"""Build (or incrementally update) the SATOPS ChromaDB vectorstore.

Handles the full pipeline in one script:
  1. Ingest raw SATOPS files (DOCX via Docling, MD via text splitter) → satops.json
  2. Enrich chunks with metadata (doc_type, topic, chunk_id)
  3. Apply semantic chunking
  4. Load into ChromaDB (idempotent — duplicate chunks are skipped)

Usage:
    # Full pipeline: ingest SATOPS/ dir and build vectorstore
    python scripts/build_satops_vectorstore.py --satops-dir ./SATOPS/

    # Re-build vectorstore from an existing satops.json (skip ingestion)
    python scripts/build_satops_vectorstore.py --input ./satops.json

    # Custom paths
    python scripts/build_satops_vectorstore.py --satops-dir ./SATOPS/ --output ./databases/chroma_satops

After running, restart the app — retrieve_context_satops will surface the new content automatically.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings

try:
    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings  # type: ignore
except ImportError:
    NVIDIAEmbeddings = None

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

MIN_CHUNK_LEN = 20

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".md"}
# Zone.Identifier files are Windows metadata — always skip them.
_SKIP_SUFFIXES = {".identifier"}

# ── doc_type label inference ──────────────────────────────────────────────────
# Files whose names contain these substrings get a more specific doc_type label,
# which lets the agent filter by doc_type when calling retrieve_context_satops.

_COMMAND_DOC_KEYWORDS = [
    "apid", "clear", "cmd", "db_stat", "evr", "file_download", "file_upload",
    "geometry", "script_output", "stalecolor", "staletimeout", "textdisplay",
    "title", "tlm", "ui_timestamp",
]

_PROCEDURE_DOC_KEYWORDS = [
    "manual_passes", "unscheduled_passes", "file_download", "file_upload",
]


def _infer_doc_type(filename: str) -> str:
    name_lower = filename.lower()
    for kw in _COMMAND_DOC_KEYWORDS:
        if kw in name_lower:
            return "satops_commands"
    for kw in _PROCEDURE_DOC_KEYWORDS:
        if kw in name_lower:
            return "satops_procedures"
    return "satops"


def _infer_topic(text: str, filename: str) -> str:
    name_lower = filename.lower()
    if any(k in name_lower for k in ("tlm", "telemetry")):
        return "satops_telemetry"
    if any(k in name_lower for k in ("file_upload", "file_download")):
        return "satops_file_transfer"
    if any(k in name_lower for k in ("apid", "cmd", "evr", "script")):
        return "satops_commands"
    if any(k in name_lower for k in ("passes", "manual_passes", "unscheduled")):
        return "satops_pass_scheduling"
    if any(k in name_lower for k in ("geometry",)):
        return "satops_geometry"
    return "satops_general"


def _chunk_id(source: str, text: str, doc_type: str) -> str:
    content_hash = hashlib.md5(text.encode()).hexdigest()[:12]
    return f"{doc_type}_{source}_{content_hash}"


# ── Ingestion helpers ─────────────────────────────────────────────────────────

def _ingest_with_docling(file_path: Path) -> list[dict]:
    """Convert a PDF/PPTX/DOCX with Docling (structure-aware, high quality)."""
    from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from transformers import AutoTokenizer

    logger.info("Docling: processing %s", file_path.name)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = AcceleratorOptions(num_threads=1, device="cpu")
    pipeline_options.images_scale = 1.0

    converter = DocumentConverter(
        format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
    )
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512)

    conv_result = converter.convert(file_path)
    doc = conv_result.document

    chunks = []
    for chunk in chunker.chunk(doc):
        page_num = 0
        if chunk.meta.doc_items and chunk.meta.doc_items[0].prov:
            page_num = chunk.meta.doc_items[0].prov[0].page_no
        chunks.append({
            "text": chunk.text,
            "source": str(file_path.resolve()),
            "filename": file_path.name,
            "page": page_num,
            "filetype": file_path.suffix.lower(),
        })

    logger.info("  → %d chunks", len(chunks))
    return chunks


def _ingest_markdown(file_path: Path) -> list[dict]:
    """Split a markdown file by heading/paragraph boundaries.

    Docling is intentionally not used here. Although HybridChunker accepts
    max_tokens=512, it does not account for the special tokens ([CLS], [SEP])
    that all-MiniLM-L6-v2 adds at embedding time (~11 extra tokens). In
    practice this produces chunks of 514+ tokens, causing indexing errors in
    the embedding model. RecursiveCharacterTextSplitter at 512 characters
    (~128 tokens) stays well under the 512-token hard limit.
    """
    import re
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logger.info("Markdown: processing %s", file_path.name)
    text = file_path.read_text(encoding="utf-8")
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
    )
    chunks = []
    for chunk_text in splitter.split_text(text):
        if chunk_text.strip():
            chunks.append({
                "text": chunk_text.strip(),
                "source": str(file_path.resolve()),
                "filename": file_path.name,
                "page": 0,
                "filetype": ".md",
            })

    logger.info("  → %d chunks", len(chunks))
    return chunks


def ingest_satops_dir(satops_dir: str, output_json: str, append: bool = False) -> list[dict]:
    """Ingest all supported files in satops_dir and write to output_json.

    Args:
        satops_dir:  Path to directory containing SATOPS documents.
        output_json: Path to write the resulting chunk JSON.
        append:      If True, extend an existing JSON rather than overwriting.

    Returns:
        Full list of raw chunk dicts written to the JSON.
    """
    sdir = Path(satops_dir)
    if not sdir.exists():
        raise FileNotFoundError(f"SATOPS directory not found: {sdir}")

    files = sorted(
        f for f in sdir.iterdir()
        if f.is_file()
        and f.suffix.lower() in SUPPORTED_EXTENSIONS
        and f.suffix.lower() not in _SKIP_SUFFIXES
        and ":Zone.Identifier" not in f.name
    )

    if not files:
        raise ValueError(f"No supported files ({', '.join(SUPPORTED_EXTENSIONS)}) found in {sdir}")

    logger.info("Found %d file(s) to ingest in %s:", len(files), sdir)
    for f in files:
        logger.info("  %s", f.name)

    existing: list[dict] = []
    if append:
        out_path = Path(output_json)
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            logger.info("Appending to existing %s (%d chunks)", out_path.name, len(existing))

    new_chunks: list[dict] = []
    for file_path in files:
        if file_path.suffix.lower() == ".md":
            new_chunks.extend(_ingest_markdown(file_path))
        else:
            new_chunks.extend(_ingest_with_docling(file_path))

    all_chunks = existing + new_chunks
    out_path = Path(output_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(
        "Wrote %d total chunks (%d new) to %s",
        len(all_chunks), len(new_chunks), out_path,
    )
    return all_chunks


# ── Embedding setup ───────────────────────────────────────────────────────────
# Priority: 1) NVIDIA hosted embeddings (nvapi- key), 2) HF endpoint server,
#           3) local all-MiniLM-L6-v2 fallback.
# Mirrors the same priority chain used in agent/agent.py.

def _setup_embeddings():
    embed_model = os.getenv("NVIDIA_EMBED_MODEL", "nvidia/nv-embed-v1")
    embed_server_url = os.getenv(
        "EMBEDDING_SERVER_URL",
        "http://trac-malenia.ern.nps.edu:8080/gpu5/embed",
    )
    fallback_model = "all-MiniLM-L6-v2"
    nvidia_key = os.getenv("ALTERNATE_EMBEDDING_MODEL_KEY")
    nvidia_url = os.getenv("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")

    # 1) NVIDIA hosted embeddings
    if NVIDIAEmbeddings is not None and nvidia_key:
        try:
            hfe = NVIDIAEmbeddings(model=embed_model, api_key=nvidia_key, base_url=nvidia_url)
            hfe.embed_query("test")
            logger.info("Embedding function set from NVIDIA API: %s", embed_model)
            return hfe
        except Exception as e:
            logger.warning("NVIDIA embeddings failed (%s) — trying HF endpoint", e)

    # 2) NPS HuggingFace endpoint server
    try:
        hfe = HuggingFaceEndpointEmbeddings(model=embed_server_url)
        hfe.embed_query("test")
        logger.info("Embedding function set from HF endpoint server: %s", embed_server_url)
        return hfe
    except Exception as e:
        logger.info("HF endpoint unreachable (%s) — using local embeddings", e)

    # 3) Local fallback
    hfe = HuggingFaceEmbeddings(
        model_name=fallback_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": False},
    )
    logger.info("Embedding function set locally with %s", fallback_model)
    return hfe


# ── Build ─────────────────────────────────────────────────────────────────────

def build_satops_vectorstore(
    input_json: str = "./satops.json",
    output_dir: str = "./databases/chroma_satops",
    collection_name: str = "satops_course",
    raw_chunks: list[dict] | None = None,
) -> Chroma:
    """Build or update the SATOPS vectorstore from a JSON file or pre-loaded chunks.

    Args:
        input_json:   Path to satops.json. Ignored if raw_chunks is provided.
        output_dir:   Chroma persist directory.
        collection_name: Chroma collection name.
        raw_chunks:   Pre-loaded chunk list (skips reading input_json).
    """
    if raw_chunks is None:
        input_path = Path(input_json)
        if not input_path.exists():
            raise FileNotFoundError(
                f"Input file not found: {input_path}\n"
                "Provide --satops-dir to ingest documents first, or "
                "run: python scripts/ingest_pdf.py ./SATOPS/ --output satops.json"
            )
        with open(input_path, encoding="utf-8") as f:
            raw_chunks = json.load(f)

    logger.info("Loaded %d raw chunks", len(raw_chunks))

    docs: list[Document] = []
    for chunk in raw_chunks:
        text = chunk.get("text", "").strip()
        if len(text) < MIN_CHUNK_LEN:
            continue
        filename = chunk.get("filename", "unknown")
        source = chunk.get("source", filename)
        doc_type = _infer_doc_type(filename)
        docs.append(Document(
            page_content=text,
            metadata={
                "source": source,
                "filename": filename,
                "page": chunk.get("page", 0),
                "filetype": chunk.get("filetype", "unknown"),
                "doc_type": doc_type,
                "topic": _infer_topic(text, filename),
                "chunk_id": _chunk_id(source, text, doc_type),
            }
        ))

    logger.info("Enriched %d documents", len(docs))

    hfe = _setup_embeddings()

    logger.info("Applying semantic chunking...")
    chunker = SemanticChunker(embeddings=hfe, breakpoint_threshold_type="percentile")
    chunked = chunker.transform_documents(docs)

    # Re-compute chunk_ids after semantic splitting (content may have changed).
    for doc in chunked:
        source = doc.metadata.get("source", "unknown")
        doc_type = doc.metadata.get("doc_type", "satops")
        content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:12]
        doc.metadata["chunk_id"] = f"{doc_type}_{source}_{content_hash}"

    logger.info("%d chunks after semantic chunking", len(chunked))

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=hfe,
        persist_directory=output_dir,
    )

    existing_ids = set(vectorstore.get()["ids"])
    logger.info("Existing chunks in store: %d", len(existing_ids))

    seen_ids: set[str] = set(existing_ids)
    new_docs = []
    for d in chunked:
        cid = d.metadata["chunk_id"]
        if cid not in seen_ids:
            seen_ids.add(cid)
            new_docs.append(d)

    logger.info("New chunks to add: %d (skipping %d duplicates)", len(new_docs), len(chunked) - len(new_docs))

    if not new_docs:
        logger.info("Vectorstore is already up to date — nothing to add.")
        return vectorstore

    ids = [d.metadata["chunk_id"] for d in new_docs]
    batch_size = 64
    for i in range(0, len(new_docs), batch_size):
        batch = new_docs[i:i + batch_size]
        vectorstore.add_documents(documents=batch, ids=ids[i:i + batch_size])
        logger.info("  Added batch %d–%d", i + 1, i + len(batch))

    total = len(vectorstore.get()["ids"])
    logger.info("Done. Total chunks in SATOPS store: %d", total)
    return vectorstore


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    _project_root = Path(__file__).parent.parent
    _default_output = os.getenv("SATOPS_CHROMA_DB_PATH", str(_project_root / "databases" / "chroma_satops"))

    parser = argparse.ArgumentParser(description="Build the SATOPS ChromaDB vectorstore.")
    parser.add_argument("--satops-dir", default=None, help="Directory of raw SATOPS files to ingest.")
    parser.add_argument("--input", default=None, help="Path to existing satops.json (skips ingestion).")
    parser.add_argument("--output", default=_default_output, help="Chroma persist directory.")
    args = parser.parse_args()

    if args.input:
        build_satops_vectorstore(input_json=args.input, output_dir=args.output)
    else:
        _satops_dir = args.satops_dir or str(_project_root / "SATOPS")
        _output_json = str(_project_root / "satops.json")
        raw = ingest_satops_dir(satops_dir=_satops_dir, output_json=_output_json)
        build_satops_vectorstore(input_json=_output_json, output_dir=args.output, raw_chunks=raw)
