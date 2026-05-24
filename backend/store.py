"""
State store with Supabase support and in-memory fallback.

Supabase is used when both SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are set.
Otherwise we keep existing in-memory behavior for local/dev simplicity.
"""
import os
import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def _sanitize_ssl_env() -> None:
    """
    Remove broken CA bundle overrides copied from a different machine path.
    """
    for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        value = os.getenv(key, "").strip()
        if value and not os.path.exists(value):
            logger.warning("Ignoring invalid %s path: %s", key, value)
            os.environ.pop(key, None)


_sanitize_ssl_env()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
_SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
_SUPABASE_ENABLED = bool(_SUPABASE_URL and _SUPABASE_KEY and _is_truthy(os.getenv("USE_SUPABASE", "true")))

_JOBS_TABLE = os.getenv("SUPABASE_JOBS_TABLE", "ingest_jobs")
_DOCS_TABLE = os.getenv("SUPABASE_DOCS_TABLE", "documents")


def _sb_headers(prefer: str = "") -> Dict[str, str]:
    h = {
        "apikey": _SUPABASE_KEY,
        "Authorization": f"Bearer {_SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        h["Prefer"] = prefer
    return h


def _sb_url(table: str) -> str:
    return f"{_SUPABASE_URL}/rest/v1/{table}"


def _sb_upsert(table: str, rows: List[Dict], on_conflict: str) -> None:
    if not _SUPABASE_ENABLED:
        return
    try:
        resp = requests.post(
            _sb_url(table),
            params={"on_conflict": on_conflict},
            headers=_sb_headers("resolution=merge-duplicates,return=minimal"),
            json=rows,
            timeout=20,
        )
        resp.raise_for_status()
    except Exception:
        logger.exception("supabase upsert failed table=%s", table)


def _sb_select_one(table: str, filters: Dict[str, str]) -> Optional[Dict]:
    if not _SUPABASE_ENABLED:
        return None
    try:
        params = {"select": "*", "limit": "1"}
        for key, value in filters.items():
            params[key] = f"eq.{value}"
        resp = requests.get(_sb_url(table), headers=_sb_headers(), params=params, timeout=20)
        resp.raise_for_status()
        rows = resp.json()
        return rows[0] if rows else None
    except Exception:
        logger.exception("supabase select one failed table=%s", table)
        return None


def _sb_select_many(table: str, params: Dict[str, str]) -> List[Dict]:
    if not _SUPABASE_ENABLED:
        return []
    try:
        resp = requests.get(_sb_url(table), headers=_sb_headers(), params=params, timeout=20)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        logger.exception("supabase select many failed table=%s", table)
        return []


# session_id -> {created_at, language}
_sessions: Dict[str, Dict] = {}

# session_id -> list of thread dicts
_threads: Dict[str, List[Dict]] = {}

# job_id -> {status, progress_pct, message, document_id}
_jobs: Dict[str, Dict] = {}

# document_id -> full document index dict
_documents: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session() -> str:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"session_id": session_id, "created_at": _now(), "language": "en"}
    _threads[session_id] = []
    return session_id


def ensure_session(session_id: str) -> Dict:
    """Ensure session exists in memory; recreate if missing."""
    if session_id not in _sessions:
        _sessions[session_id] = {"session_id": session_id, "created_at": _now(), "language": "en"}
    _threads.setdefault(session_id, [])
    return _sessions[session_id]


def get_session(session_id: str) -> Optional[Dict]:
    return _sessions.get(session_id)


def update_session_language(session_id: str, language: str) -> None:
    if session_id in _sessions:
        _sessions[session_id]["language"] = language


# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

def create_thread(session_id: str, title: str = "") -> Dict:
    thread_id = str(uuid.uuid4())
    thread = {
        "thread_id": thread_id,
        "session_id": session_id,
        "title": title or "New Issue",
        "created_at": _now(),
        "status": "open",
    }
    _threads.setdefault(session_id, []).append(thread)
    return thread


def get_threads(session_id: str) -> List[Dict]:
    return _threads.get(session_id, [])


def get_history(session_id: str) -> List[Dict]:
    return [t for t in _threads.get(session_id, []) if t.get("status") == "resolved"]


# ---------------------------------------------------------------------------
# Ingestion jobs
# ---------------------------------------------------------------------------

def create_job() -> str:
    job_id = str(uuid.uuid4())
    job = {
        "job_id": job_id,
        "status": "pending",
        "progress_pct": 0,
        "message": "Queued...",
        "document_id": "",
        "updated_at": _now(),
    }
    _jobs[job_id] = job
    _sb_upsert(_JOBS_TABLE, [job], on_conflict="job_id")
    return job_id


def update_job(job_id: str, **kwargs) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)
        _jobs[job_id]["updated_at"] = _now()
    else:
        _jobs[job_id] = {"job_id": job_id, "updated_at": _now(), **kwargs}
    _sb_upsert(_JOBS_TABLE, [_jobs[job_id]], on_conflict="job_id")


def get_job(job_id: str) -> Optional[Dict]:
    job = _jobs.get(job_id)
    if job:
        return job
    row = _sb_select_one(_JOBS_TABLE, {"job_id": job_id})
    if row:
        _jobs[job_id] = row
    return row


# ---------------------------------------------------------------------------
# Documents (for library metadata persistence on serverless)
# ---------------------------------------------------------------------------

def save_document_index(index: Dict) -> None:
    document_id = index.get("document_id", "")
    if not document_id:
        return
    _documents[document_id] = index

    if not _SUPABASE_ENABLED:
        return
    row = {
        "document_id": document_id,
        "bike_brand": index.get("bike_brand", ""),
        "bike_model": index.get("bike_model", ""),
        "bike_year": index.get("bike_year", ""),
        "manual_type": index.get("manual_type", ""),
        "manual_source": index.get("manual_source", "user_uploaded"),
        "total_chunks": index.get("total_chunks", 0),
        "ingestion_timestamp": index.get("ingestion_timestamp", _now()),
        "index_json": index,
    }
    _sb_upsert(_DOCS_TABLE, [row], on_conflict="document_id")


def list_library_documents() -> List[Dict]:
    if _SUPABASE_ENABLED:
        rows = _sb_select_many(
            _DOCS_TABLE,
            {
                "select": "document_id,bike_brand,bike_model,bike_year,manual_type,manual_source,total_chunks,ingestion_timestamp",
                "manual_source": "eq.library",
                "order": "ingestion_timestamp.desc",
            },
        )
        if rows:
            return rows
    # Fallback in-memory
    out = []
    for d in _documents.values():
        if d.get("manual_source", "user_uploaded") != "library":
            continue
        out.append({
            "document_id": d.get("document_id", ""),
            "bike_brand": d.get("bike_brand", ""),
            "bike_model": d.get("bike_model", ""),
            "bike_year": d.get("bike_year", ""),
            "manual_type": d.get("manual_type", ""),
            "manual_source": d.get("manual_source", "user_uploaded"),
            "total_chunks": d.get("total_chunks", 0),
            "ingestion_timestamp": d.get("ingestion_timestamp", ""),
        })
    return sorted(out, key=lambda x: x.get("ingestion_timestamp", ""), reverse=True)


def get_document_index(document_id: str) -> Optional[Dict]:
    if document_id in _documents:
        return _documents[document_id]

    if _SUPABASE_ENABLED:
        row = _sb_select_many(
            _DOCS_TABLE,
            {"select": "index_json", "document_id": f"eq.{document_id}", "limit": "1"},
        )
        if row and row[0].get("index_json"):
            data = row[0]["index_json"]
            _documents[document_id] = data
            return data
    return None
