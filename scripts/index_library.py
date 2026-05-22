#!/usr/bin/env python3
"""
Pre-index library bikes into the vector store and BM25 index.

Usage:
  python scripts/index_library.py --pdf path/to/manual.pdf \
      --brand "Royal Enfield" --model "Meteor 350" --year 2022 \
      --type service_manual [--mock]

  # Or index all PDFs in a directory:
  python scripts/index_library.py --dir data/manuals/library/ --mock

Each indexed manual gets manual_source=library so it appears on the Bike Library tab.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))


def index_one(pdf_path: str, brand: str, model: str, year: str,
               manual_type: str, use_mocks: bool) -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")

    from ingestion.parser import parse_pdf
    from ingestion.chunker import chunk_blocks
    from ingestion.embedder import embed_chunks
    from ingestion.indexer import build_indexes
    from ingestion.document_index import generate_document_index

    doc_meta = {
        "bike_brand": brand,
        "bike_model": model,
        "bike_year": str(year),
        "manual_type": manual_type,
        "manual_source": "library",
    }

    print(f"\n[1/5] Parsing  {Path(pdf_path).name} ...")
    parsed = parse_pdf(pdf_path, doc_meta, use_mocks=use_mocks)
    document_id = parsed["document_id"]
    print(f"      document_id = {document_id}")

    print("[2/5] Chunking ...")
    chunks = list(chunk_blocks(parsed["blocks"], doc_meta, document_id))
    print(f"      {len(chunks)} chunks")

    print("[3/5] Embedding ...")
    embedded = embed_chunks(chunks, use_mocks=use_mocks)

    print("[4/5] Building indexes ...")
    result = build_indexes(embedded, document_id)
    print(f"      Qdrant: {result['qdrant_count']} vectors")

    print("[5/5] Writing document index ...")
    idx = generate_document_index(document_id, doc_meta, embedded)
    print(f"      {idx['total_chunks']} chunks indexed → data/indexes/{document_id}_index.json")
    print(f"\nDone. {brand} {model} ({year}) is now in the library.")


def main():
    ap = argparse.ArgumentParser(description="Index a bike manual into the library.")
    ap.add_argument("--pdf", help="Path to PDF file")
    ap.add_argument("--dir", help="Directory of PDFs (uses filename as model name)")
    ap.add_argument("--brand", default="Unknown Brand")
    ap.add_argument("--model", default="Unknown Model")
    ap.add_argument("--year", default="2024")
    ap.add_argument("--type", dest="manual_type", default="service_manual",
                    choices=["service_manual", "owner_manual", "user_guide"])
    ap.add_argument("--mock", action="store_true", help="Use mock embeddings (no OpenAI key needed)")
    args = ap.parse_args()

    if args.pdf:
        index_one(args.pdf, args.brand, args.model, args.year, args.manual_type, args.mock)
    elif args.dir:
        d = Path(args.dir)
        pdfs = list(d.glob("*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {d}")
            sys.exit(1)
        for pdf in pdfs:
            # Derive model name from filename: "Royal_Enfield_Meteor_350.pdf" → "Royal Enfield Meteor 350"
            stem = pdf.stem.replace("_", " ").replace("-", " ")
            index_one(str(pdf), args.brand, stem, args.year, args.manual_type, args.mock)
    else:
        ap.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
