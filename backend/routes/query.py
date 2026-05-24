"""
POST /query — unified query assembly from text + voice transcript + image description,
then runs the inference pipeline and returns a structured response.
"""
import os
import logging
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from inference.pipeline import run_query
from output.tts import synthesize
from store import update_session_language

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    text: str
    session_id: str
    document_id: str
    thread_id: str
    transcript: Optional[str] = None        # from /input/voice
    image_description: Optional[str] = None  # from /input/image
    voice_initiated: bool = False            # triggers TTS on spoken_summary


def assemble_query(
    text: str,
    transcript: Optional[str] = None,
    image_description: Optional[str] = None,
) -> str:
    """Merge text, voice transcript, and image context into one query string."""
    parts = []
    if transcript and transcript.strip():
        parts.append(transcript.strip())
    if text and text.strip():
        parts.append(text.strip())
    if image_description and image_description.strip():
        parts.append(f"Image context: {image_description.strip()}")
    return " ".join(parts) if parts else text


@router.post("/query")
def query(body: QueryRequest):
    unified = assemble_query(body.text, body.transcript, body.image_description)
    if not unified.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"

    try:
        result = run_query(
            query=unified,
            document_id=body.document_id,
            thread_id=body.thread_id,
            use_mocks=use_mocks,
        )
    except Exception as exc:
        logger.exception("query pipeline failed")
        result = {
            "answer_text": (
                "I couldn't find this in your manual. "
                "For this issue, I'd recommend visiting an authorised service center."
            ),
            "spoken_summary": "I could not complete retrieval from the manual right now.",
            "citations": [],
            "severity_label": "N/A",
            "confidence": "low",
            "suggested_followups": [],
            "intent": "diagnostic",
            "language": "en",
            "context_confidence": "low",
            "_debug_error": repr(exc),
        }

    result["session_id"] = body.session_id
    result["thread_id"] = body.thread_id
    result["document_id"] = body.document_id

    # Persist detected language on every turn so TTS/UI can track switches
    detected_lang = result.get("language", "en") or "en"
    update_session_language(body.session_id, detected_lang)

    if body.voice_initiated:
        spoken = result.get("spoken_summary", "") or ""
        language = result.get("language", "en") or "en"
        tts_result = synthesize(spoken, language, use_mocks=use_mocks)
        result["tts"] = {
            "mocked": tts_result["mocked"],
            "engine": tts_result["engine"],
            "text": tts_result["text"],
        }
    else:
        result["tts"] = None

    return result
