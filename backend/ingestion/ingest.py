"""
Main ingestion entry point. Runs the full offline pipeline for one manual PDF.
Usage: python -m ingestion.ingest --pdf path/to/manual.pdf [--mock]
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")


def run_ingestion(
    pdf_path: str,
    doc_meta: dict,
    use_mocks: bool = False,
    images_dir: Optional[Path] = None,
) -> str:
    """
    Full ingestion pipeline: parse → chunk → embed → index → generate document index.
    Returns the document_id.
    """
    from .chunker import chunk_blocks
    from .document_index import generate_document_index
    from .embedder import embed_chunks
    from .indexer import build_indexes
    from .parser import parse_pdf

    print(f"[ingest] Parsing {pdf_path} ...")
    parsed = parse_pdf(pdf_path, doc_meta, use_mocks=use_mocks, images_dir=images_dir)
    document_id = parsed["document_id"]
    blocks = parsed["blocks"]

    content_counts = {}
    for b in blocks:
        ct = b["content_type"]
        content_counts[ct] = content_counts.get(ct, 0) + 1
    print(f"[ingest] Blocks parsed: {content_counts}")

    print("[ingest] Chunking ...")
    chunks = chunk_blocks(blocks, doc_meta, document_id)
    print(f"[ingest] Total chunks: {len(chunks)}")

    print("[ingest] Embedding ...")
    chunks = embed_chunks(chunks, use_mocks=use_mocks)

    print("[ingest] Indexing (Qdrant + BM25) ...")
    result = build_indexes(chunks, document_id)
    print(f"[ingest] Indexed {result['qdrant_count']} vectors, BM25 at {result['bm25_path']}")

    print("[ingest] Generating document index ...")
    index = generate_document_index(document_id, doc_meta, chunks)
    print(f"[ingest] Document index: {_ROOT / 'data/indexes' / (document_id + '_index.json')}")

    return document_id


def main():
    parser = argparse.ArgumentParser(description="Ingest a bike manual PDF")
    parser.add_argument("--pdf", required=True, help="Path to the manual PDF")
    parser.add_argument("--brand", default="", help="Bike brand (e.g. Royal Enfield)")
    parser.add_argument("--model", default="", help="Bike model (e.g. Meteor 350)")
    parser.add_argument("--year", default="", help="Model year (e.g. 2022)")
    parser.add_argument("--manual-type", default="service_manual",
                        choices=["owner_manual", "service_manual", "user_guide"])
    parser.add_argument("--mock", action="store_true",
                        help="Use mocks for all external APIs (LlamaParse, OpenAI, etc.)")
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        print(f"Error: PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    use_mocks = args.mock or os.getenv("USE_MOCKS", "false").lower() == "true"
    doc_meta = {
        "bike_brand": args.brand,
        "bike_model": args.model,
        "bike_year": args.year,
        "manual_type": args.manual_type,
        "manual_source": "library",
    }

    document_id = run_ingestion(args.pdf, doc_meta, use_mocks=use_mocks)
    print(f"[ingest] Done. document_id={document_id}")


if __name__ == "__main__":
    main()
