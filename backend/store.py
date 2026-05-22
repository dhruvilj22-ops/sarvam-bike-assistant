"""
In-memory state for sessions, threads, and ingestion jobs.
SQLite persistence is wired in Part 7 — this module keeps the same interface.
"""
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# session_id -> {created_at, language}
_sessions: Dict[str, Dict] = {}

# session_id -> list of thread dicts
_threads: Dict[str, List[Dict]] = {}

# job_id -> {status, progress_pct, message, document_id}
_jobs: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------

def create_session() -> str:
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {"session_id": session_id, "created_at": _now(), "language": "en"}
    _threads[session_id] = []
    return session_id


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
    """Return resolved threads — resolved status set externally when issue is closed."""
    return [t for t in _threads.get(session_id, []) if t.get("status") == "resolved"]


# ---------------------------------------------------------------------------
# Ingestion jobs
# ---------------------------------------------------------------------------

def create_job() -> str:
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress_pct": 0,
        "message": "Queued...",
        "document_id": "",
    }
    return job_id


def update_job(job_id: str, **kwargs) -> None:
    if job_id in _jobs:
        _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> Optional[Dict]:
    return _jobs.get(job_id)
