"""
GET /bikes/library — returns list of pre-indexed manuals available in the library.
GET /bikes/{document_id}/starters — returns 4 suggested questions derived from the manual's chapters.
Scans data/indexes/ for *_index.json files written by generate_document_index().
"""
import json
import os
from pathlib import Path
from fastapi import APIRouter
import store
from ingestion.indexer import get_qdrant_client

router = APIRouter()

_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_INDEX_DIR = "/tmp/indexes" if os.getenv("VERCEL") else str(_ROOT / "data" / "indexes")
_INDEX_DIR = Path(os.getenv("INDEX_DIR", _DEFAULT_INDEX_DIR))

_SKIP_WORDS = {
    "safety", "index", "warranty", "introduction", "contents",
    "appendix", "glossary", "foreword", "preface", "general information",
}

_QUESTION_MAP = [
    (["engine oil", "oil change", "lubrication"],       "How do I check and change the engine oil?"),
    (["maintenance schedule", "periodic maintenance"],   "What is the recommended maintenance schedule?"),
    (["brake", "braking"],                               "How do I inspect and adjust the brakes?"),
    (["fuel", "carburetor", "carburettor"],              "How do I check and clean the fuel system?"),
    (["spark plug", "ignition"],                         "When and how should I replace the spark plug?"),
    (["chain", "drive chain"],                           "How do I adjust and lubricate the drive chain?"),
    (["tire", "tyre", "wheel"],                          "How do I check tire pressure and condition?"),
    (["electrical", "battery", "wiring"],                "How do I troubleshoot electrical issues?"),
    (["cooling", "coolant"],                             "How does the cooling system work?"),
    (["valve", "valve clearance"],                       "What are the valve clearance specifications?"),
    (["air filter", "air cleaner"],                      "How do I clean or replace the air filter?"),
    (["clutch"],                                         "How do I adjust the clutch cable?"),
    (["suspension", "fork"],                             "How do I check and adjust the suspension?"),
    (["engine", "cylinder", "piston"],                   "What are the engine torque and service specifications?"),
]

_FALLBACK_STARTERS = [
    "What does white smoke from the exhaust mean?",
    "How do I check and change the engine oil?",
    "What is the valve clearance specification?",
    "My bike is making a knocking sound — what could it be?",
]


def _chapter_to_question(title: str) -> str:
    t = title.lower()
    for keywords, question in _QUESTION_MAP:
        if any(kw in t for kw in keywords):
            return question
    return f"What does the manual say about {title}?"


def _library_from_qdrant() -> list:
    """
    Build library summaries from indexed chunk payloads when metadata store is empty.
    """
    try:
        client = get_qdrant_client()
        rows, _ = client.scroll(
            collection_name="bike_manuals",
            limit=5000,
            with_payload=True,
            with_vectors=False,
        )
        seen = {}
        for p in rows:
            payload = getattr(p, "payload", {}) or {}
            if payload.get("manual_source", "user_uploaded") != "library":
                continue
            doc_id = payload.get("document_id", "")
            if not doc_id:
                continue
            if doc_id not in seen:
                seen[doc_id] = {
                    "document_id": doc_id,
                    "bike_brand": payload.get("bike_brand", ""),
                    "bike_model": payload.get("bike_model", ""),
                    "bike_year": payload.get("bike_year", ""),
                    "manual_type": payload.get("manual_type", ""),
                    "manual_source": payload.get("manual_source", "user_uploaded"),
                    "total_chunks": 0,
                    "ingestion_timestamp": "",
                }
            seen[doc_id]["total_chunks"] += 1
        return list(seen.values())
    except Exception:
        return []


@router.get("/bikes/library")
def library():
    # Primary: durable Supabase-backed document metadata store.
    supabase_rows = store.list_library_documents()
    if supabase_rows:
        # Dedup: same brand + model + manual_type → keep most recent entry.
        seen: dict = {}
        for b in supabase_rows:
            key = (b["bike_brand"].lower(), b["bike_model"].lower(), b["manual_type"])
            if key not in seen or b["ingestion_timestamp"] > seen[key]["ingestion_timestamp"]:
                seen[key] = b
        return {"bikes": list(seen.values())}

    # Fallback 1: derive library docs directly from Qdrant payloads.
    qdrant_rows = _library_from_qdrant()
    if qdrant_rows:
        return {"bikes": qdrant_rows}

    # Fallback 2: local index JSON files (dev / non-Supabase mode).
    raw = []
    if _INDEX_DIR.exists():
        for path in sorted(_INDEX_DIR.glob("*_index.json")):
            try:
                data = json.loads(path.read_text())
                if data.get("manual_source", "user_uploaded") != "library":
                    continue
                raw.append({
                    "document_id": data.get("document_id", ""),
                    "bike_brand": data.get("bike_brand", ""),
                    "bike_model": data.get("bike_model", ""),
                    "bike_year": data.get("bike_year", ""),
                    "manual_type": data.get("manual_type", ""),
                    "manual_source": data.get("manual_source", "user_uploaded"),
                    "total_chunks": data.get("total_chunks", 0),
                    "ingestion_timestamp": data.get("ingestion_timestamp", ""),
                })
            except Exception:
                continue

    # Dedup: same brand + model + manual_type → keep the most recently ingested entry
    seen: dict = {}
    for b in raw:
        key = (b["bike_brand"].lower(), b["bike_model"].lower(), b["manual_type"])
        if key not in seen or b["ingestion_timestamp"] > seen[key]["ingestion_timestamp"]:
            seen[key] = b

    return {"bikes": list(seen.values())}


@router.get("/bikes/{document_id}/starters")
def get_starters(document_id: str):
    """Return 4 suggested questions derived from this manual's chapter titles."""
    data = store.get_document_index(document_id)
    if data is None:
        index_path = _INDEX_DIR / f"{document_id}_index.json"
        if not index_path.exists():
            return {"starters": _FALLBACK_STARTERS}
        try:
            data = json.loads(index_path.read_text())
        except Exception:
            return {"starters": _FALLBACK_STARTERS}

    try:
        chapters = data.get("chapters", [])

        # Skip generic front-matter chapters
        useful = [
            ch for ch in chapters
            if not any(skip in ch.get("title", "").lower() for skip in _SKIP_WORDS)
        ]

        seen_questions: set = set()
        questions: list = []
        for ch in useful:
            q = _chapter_to_question(ch.get("title", ""))
            if q not in seen_questions:
                seen_questions.add(q)
                questions.append(q)
            if len(questions) == 4:
                break

        # Pad with fallbacks if fewer than 4 mapped
        for fb in _FALLBACK_STARTERS:
            if len(questions) >= 4:
                break
            if fb not in seen_questions:
                questions.append(fb)

        return {"starters": questions[:4]}
    except Exception:
        return {"starters": _FALLBACK_STARTERS}
