"""
POST /input/voice — transcribes audio, stores language in session.
POST /input/image — describes image using GPT-4o vision.
"""
import os
from typing import Optional
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

import store
from input.stt import transcribe
from input.vision import describe_image

router = APIRouter()

_ALLOWED_AUDIO = {".wav", ".mp3", ".m4a", ".mp4", ".ogg", ".webm", ".flac"}
_ALLOWED_IMAGE = {".jpg", ".jpeg", ".png", ".webp", ".gif"}


@router.post("/input/voice")
async def voice_input(
    audio: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    language_hint: str = Form("auto"),
):
    suffix = "." + (audio.filename or "audio.wav").rsplit(".", 1)[-1].lower()
    if suffix not in _ALLOWED_AUDIO:
        raise HTTPException(status_code=400, detail="Unsupported audio format")

    use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"
    audio_bytes = await audio.read()

    result = transcribe(
        audio_bytes=audio_bytes,
        filename=audio.filename or "audio.wav",
        language_hint=language_hint,
        use_mocks=use_mocks,
    )

    # Store language in session if provided and session exists
    if session_id and store.get_session(session_id):
        store.update_session_language(session_id, result["language"])

    return result


@router.post("/input/image")
async def image_input(
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
):
    suffix = "." + (image.filename or "image.jpg").rsplit(".", 1)[-1].lower()
    if suffix not in _ALLOWED_IMAGE:
        raise HTTPException(status_code=400, detail="Unsupported image format")

    use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"
    image_bytes = await image.read()

    mime = "image/jpeg"
    if suffix == ".png":
        mime = "image/png"
    elif suffix == ".webp":
        mime = "image/webp"
    elif suffix == ".gif":
        mime = "image/gif"

    result = describe_image(image_bytes=image_bytes, mime_type=mime, use_mocks=use_mocks)
    return result
