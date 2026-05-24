"""
Session and thread management routes.
POST /session, GET /session/{id}/threads, POST /session/{id}/threads, GET /session/{id}/history
"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

import store

router = APIRouter()


class CreateThreadRequest(BaseModel):
    title: Optional[str] = ""


@router.post("/session", status_code=201)
def create_session():
    session_id = store.create_session()
    return {"session_id": session_id}


@router.get("/session/{session_id}/threads")
def list_threads(session_id: str):
    store.ensure_session(session_id)
    return {"session_id": session_id, "threads": store.get_threads(session_id)}


@router.post("/session/{session_id}/threads", status_code=201)
def create_thread(session_id: str, body: CreateThreadRequest = None):
    store.ensure_session(session_id)
    title = (body.title if body else "") or ""
    thread = store.create_thread(session_id, title=title)
    return thread


@router.get("/session/{session_id}/history")
def get_history(session_id: str):
    store.ensure_session(session_id)
    return {"session_id": session_id, "history": store.get_history(session_id)}
