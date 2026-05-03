"""Ingest documents (PDF, PPTX, DOCX) into a JSON chunk file using Docling.

Mirrors the ingestion approach from NPS_tutor.ipynb (cells 9-20):
  - DocumentConverter with CPU pipeline options
  - HybridChunker (structure-aware, max 512 tokens)
  - Saves chunks as {"text", "source", "filename", "page", "filetype"}

The output JSON is suitable as input to scripts/build_vectorstore.py via
a DocTypeConfig entry.

Usage examples:

    # Ingest a single PDF into satops.json
    python scripts/ingest_pdf.py manual.pdf --output satops.json

    # Ingest all PDFs in a directory
    python scripts/ingest_pdf.py ./SatOps/ --glob "*.pdf" --output satops.json

    # Append new files to an existing JSON (skips nothing — deduplication
    # happens in build_vectorstore.py via chunk IDs)
    python scripts/ingest_pdf.py new_manual.pdf --output satops.json --append

Supported file types: .pdf, .pptx, .docx
"""

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".pptx", ".docx", ".md"}

# ── Converter / chunker setup (mirrors NPS_tutor.ipynb cell 11) ───────────────

pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=1,   # single-threaded to prevent memory explosion on large PDFs
    device="cpu",
)
pipeline_options.images_scale = 1.0

converter = DocumentConverter(
    format_options={
        "pdf": PdfFormatOption(pipeline_options=pipeline_options)
    }
)

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
hybrid_chunker = HybridChunker(tokenizer=tokenizer, max_tokens=512)


# ── Core loader (mirrors load_with_docling in NPS_tutor.ipynb cell 11) ────────

def load_with_docling(file_path: Path) -> list[dict]:
    """Convert one file with Docling and return a list of serializable chunk dicts."""
    logger.info("Processing: %s", file_path.name)
    conv_result = converter.convert(file_path)
    doc = conv_result.document

    chunks = []
    for chunk in hybrid_chunker.chunk(doc):
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


def load_markdown(file_path: Path) -> list[dict]:
    """Split a markdown file into chunks by heading/paragraph boundaries.

    Docling is intentionally not used here. Although HybridChunker accepts
    max_tokens=512, it does not account for the special tokens ([CLS], [SEP])
    that all-MiniLM-L6-v2 adds at embedding time (~11 extra tokens). In
    practice this produces chunks of 514+ tokens, causing indexing errors in
    the embedding model. RecursiveCharacterTextSplitter at 512 characters
    (~128 tokens) stays well under the 512-token hard limit.
    """
    import re
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logger.info("Processing (markdown): %s", file_path.name)
    text = file_path.read_text(encoding="utf-8")
    # Strip YAML frontmatter if present.
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


# ── File collection helpers ───────────────────────────────────────────────────

def _collect_files(inputs: list[str], glob_pattern: str | None) -> list[Path]:
    """Resolve CLI inputs (files or directories) to a sorted list of file paths."""
    files: list[Path] = []
    for inp in inputs:
        p = Path(inp)
        if p.is_dir():
            pattern = glob_pattern or "*"
            matched = sorted(p.glob(pattern))
            supported = [f for f in matched if f.suffix.lower() in SUPPORTED_EXTENSIONS]
            if not supported:
                logger.warning("No supported files found in %s with pattern '%s'", p, pattern)
            files.extend(supported)
        elif p.is_file():
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                logger.warning("Unsupported file type '%s' — skipping %s", p.suffix, p.name)
            else:
                files.append(p)
        else:
            logger.error("Path not found: %s", p)
    return files


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest(
    inputs: list[str],
    output: str,
    glob_pattern: str | None = None,
    append: bool = False,
) -> None:
    """Ingest documents and write chunk JSON.

    Args:
        inputs:       List of file paths or directories to ingest.
        output:       Output JSON path.
        glob_pattern: Glob pattern when inputs include directories (e.g. "*.pdf").
        append:       If True, load existing JSON at output path and extend it.
    """
    files = _collect_files(inputs, glob_pattern)
    if not files:
        logger.error("No files to ingest. Exiting.")
        return

    logger.info("Ingesting %d file(s):", len(files))
    for f in files:
        logger.info("  %s", f)

    existing: list[dict] = []
    if append:
        out_path = Path(output)
        if out_path.exists():
            with open(out_path, encoding="utf-8") as f:
                existing = json.load(f)
            logger.info("Appending to existing %s (%d chunks)", out_path.name, len(existing))

    new_chunks: list[dict] = []
    for file_path in files:
        if file_path.suffix.lower() == ".md":
            new_chunks.extend(load_markdown(file_path))
        else:
            new_chunks.extend(load_with_docling(file_path))

    all_chunks = existing + new_chunks
    out_path = Path(output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info(
        "Wrote %d total chunks (%d new) to %s",
        len(all_chunks), len(new_chunks), out_path,
    )
    logger.info(
        "Next step: run 'python scripts/build_vectorstore.py' to add these chunks to ChromaDB."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDF/PPTX/DOCX files into a JSON chunk file for build_vectorstore.py."
    )
    parser.add_argument(
        "inputs", nargs="+",
        help="One or more file paths or directories to ingest.",
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Output JSON file path (e.g. satops.json).",
    )
    parser.add_argument(
        "--glob", default=None,
        help="Glob pattern when inputs include directories (default: all supported types).",
    )
    parser.add_argument(
        "--append", action="store_true",
        help="Append to existing output JSON instead of overwriting.",
    )
    args = parser.parse_args()

    ingest(
        inputs=args.inputs,
        output=args.output,
        glob_pattern=args.glob,
        append=args.append,
    )
