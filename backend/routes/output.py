"""POST /output/tts — text-to-speech endpoint."""
import os
from fastapi import APIRouter, Response
from pydantic import BaseModel

from output.tts import synthesize

router = APIRouter()


class TTSRequest(BaseModel):
    text: str
    language: str = "en"


@router.post("/output/tts")
def tts(body: TTSRequest):
    use_mocks = os.getenv("USE_MOCKS", "false").lower() == "true"
    result = synthesize(body.text, body.language, use_mocks=use_mocks)
    if result["mocked"]:
        return {"mocked": True, "text": result["text"], "engine": result["engine"]}
    return Response(content=result["audio_bytes"], media_type=result["content_type"])
