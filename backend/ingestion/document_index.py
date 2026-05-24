"""
Generates the per-manual document index JSON after ingestion.
Used for ingestion validation, query routing hints, and manual coverage display.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from .classifier import ContentType

import os

_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_INDEX_DIR = "/tmp/indexes" if os.getenv("VERCEL") else str(_ROOT / "data" / "indexes")
_INDEX_DIR = Path(os.getenv("INDEX_DIR", _DEFAULT_INDEX_DIR))


def generate_document_index(
    document_id: str,
    doc_meta: Dict,
    chunks: List[Dict],
    confidence_threshold: float = 0.0,
) -> Dict:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # Count by content type
    type_counts: Dict[str, int] = {ct.value: 0 for ct in ContentType}
    chapters: Dict[str, Dict] = {}

    for chunk in chunks:
        ct = chunk.get("content_type", ContentType.PROSE)
        type_counts[ct] = type_counts.get(ct, 0) + 1

        ch_num = chunk.get("chapter_number", "")
        ch_title = chunk.get("chapter_title", "")
        pg = chunk.get("page_number", 0)
        if ch_num and ch_num not in chapters:
            chapters[ch_num] = {"number": ch_num, "title": ch_title, "page_start": pg, "page_end": pg}
        elif ch_num:
            chapters[ch_num]["page_end"] = max(chapters[ch_num]["page_end"], pg)

    index = {
        "document_id": document_id,
        "bike_brand": doc_meta.get("bike_brand", ""),
        "bike_model": doc_meta.get("bike_model", ""),
        "bike_year": doc_meta.get("bike_year", ""),
        "manual_type": doc_meta.get("manual_type", ""),
        "manual_source": doc_meta.get("manual_source", "user_uploaded"),
        "chapters": sorted(chapters.values(), key=lambda c: c["number"]),
        "content_type_counts": type_counts,
        "total_chunks": len(chunks),
        "confidence_threshold": confidence_threshold,
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    path = _INDEX_DIR / f"{document_id}_index.json"
    path.write_text(json.dumps(index, indent=2))
    return index
