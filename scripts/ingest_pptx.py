"""Ingest individual lecture .pptx files into lectures.json using Docling.

Mirrors the ingestion approach in NPS_tutor.ipynb (cells 9-14):
  - DocumentConverter with CPU pipeline options
  - HybridChunker (structure-aware, max 512 tokens)
  - Saves chunks as {"text", "source", "filename", "page", "filetype"}

Only processes files matching 'SS3861 - N - *.pptx' so the compiled PDF
and the Lab reference pptx are excluded.  The resulting lectures.json will
have per-file filenames that allow build_vectorstore.py to extract
lecture_number metadata via _extract_lecture_number().

Usage:
    python scripts/ingest_pptx.py
    python scripts/ingest_pptx.py --lectures-dir ./Lectures --output ./lectures.json
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from docling.datamodel.pipeline_options import AcceleratorOptions, PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Converter / chunker setup (mirrors NPS_tutor.ipynb cell 11) ───────────────

pipeline_options = PdfPipelineOptions()
pipeline_options.accelerator_options = AcceleratorOptions(
    num_threads=1,
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
    """Convert one file with Docling and return a list of chunk dicts."""
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


# ── Main ──────────────────────────────────────────────────────────────────────

def ingest_all(lectures_dir: str = "./Lectures", output: str = "./lectures.json") -> None:
    ldir = Path(lectures_dir)
    pattern = re.compile(r"SS3861\s*-\s*\d+\s*-.*\.pptx$", re.IGNORECASE)
    pptx_files = sorted(p for p in ldir.glob("*.pptx") if pattern.match(p.name))

    if not pptx_files:
        logger.error("No matching .pptx files found in %s", ldir)
        return

    logger.info("Found %d lecture files to ingest", len(pptx_files))

    all_chunks: list[dict] = []
    for pptx_path in pptx_files:
        all_chunks.extend(load_with_docling(pptx_path))

    # Also ingest the compiled lecture notes PDF (foundational course text).
    # Page number is the only lecture-specific metadata available for this file.
    pdf_path = ldir / "SS3861_Lectures.pdf"
    if pdf_path.exists():
        logger.info("Ingesting compiled lecture notes PDF: %s", pdf_path.name)
        all_chunks.extend(load_with_docling(pdf_path))
    else:
        logger.warning("Compiled lecture notes PDF not found at %s — skipping", pdf_path)

    out_path = Path(output)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info("Wrote %d total chunks to %s", len(all_chunks), out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest lecture .pptx files into lectures.json using Docling.")
    parser.add_argument(
        "--lectures-dir",
        default=str(Path(__file__).parent.parent / "Lectures"),
        help="Directory containing .pptx files (default: ../Lectures relative to this script)",
    )
    parser.add_argument(
        "--output",
        default=str(Path(__file__).parent.parent / "lectures.json"),
        help="Output JSON path (default: ../lectures.json relative to this script)",
    )
    args = parser.parse_args()
    ingest_all(lectures_dir=args.lectures_dir, output=args.output)
