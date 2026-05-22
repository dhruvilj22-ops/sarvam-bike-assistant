"""
Speech-to-text routing.
Indic languages → Sarvam STT.
English (and undetected) → OpenAI Whisper.
USE_MOCKS=true bypasses all real API calls.
"""
import io
import logging
import os
from typing import Dict

import requests

logger = logging.getLogger(__name__)

STT_CONFIDENCE_THRESHOLD = 0.7

# ISO 639-1 and BCP-47 codes considered Indic
INDIC_LANGS = {
    "hi", "hi-IN", "ta", "ta-IN", "te", "te-IN",
    "kn", "kn-IN", "mr", "mr-IN", "bn", "bn-IN",
    "gu", "gu-IN", "pa", "pa-IN", "ml", "ml-IN",
    "or", "or-IN", "as",
}

_MOCK_HINDI = {
    "transcript": "इंजन से आवाज आ रही है",
    "language": "hi",
    "confidence": 0.92,
    "engine": "sarvam",
    "needs_retry": False,
}
_MOCK_ENGLISH = {
    "transcript": "white smoke from exhaust",
    "language": "en",
    "confidence": 0.95,
    "engine": "whisper",
    "needs_retry": False,
}


def _transcribe_sarvam(audio_bytes: bytes, language_hint: str = "hi-IN") -> Dict:
    url = "https://api.sarvam.ai/speech-to-text"
    lang_code = language_hint if "-" in language_hint else f"{language_hint}-IN"
    resp = requests.post(
        url,
        headers={"api-subscription-key": os.getenv("SARVAM_API_KEY", "")},
        files={"file": ("audio.wav", io.BytesIO(audio_bytes), "audio/wav")},
        data={"model": "saarika:v2.5", "language_code": lang_code},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    transcript = data.get("transcript", "")
    confidence = 0.85  # Sarvam doesn't return confidence; use a safe default
    return {
        "transcript": transcript,
        "language": data.get("language_code", lang_code).split("-")[0],
        "confidence": confidence,
        "engine": "sarvam",
        "needs_retry": confidence < STT_CONFIDENCE_THRESHOLD,
    }


def _transcribe_whisper(audio_bytes: bytes, filename: str = "audio.wav") -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    result = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="verbose_json",
    )
    # avg_logprob is log probability; convert to a 0-1 confidence approximation
    log_prob = getattr(result, "avg_logprob", -0.1)
    import math
    confidence = min(1.0, max(0.0, math.exp(log_prob)))
    lang = getattr(result, "language", "en")
    return {
        "transcript": result.text,
        "language": lang,
        "confidence": confidence,
        "engine": "whisper",
        "needs_retry": confidence < STT_CONFIDENCE_THRESHOLD,
    }


def transcribe(
    audio_bytes: bytes,
    filename: str = "audio.wav",
    language_hint: str = "auto",
    use_mocks: bool = False,
) -> Dict:
    """
    Transcribe audio. Routes to Sarvam for Indic, Whisper for English.
    Returns {transcript, language, confidence, engine, needs_retry}.
    """
    if use_mocks:
        is_indic = language_hint in INDIC_LANGS or "hindi" in filename.lower()
        return _MOCK_HINDI if is_indic else _MOCK_ENGLISH

    is_indic = language_hint in INDIC_LANGS
    try:
        if is_indic:
            return _transcribe_sarvam(audio_bytes, language_hint)
        else:
            result = _transcribe_whisper(audio_bytes, filename)
            # If Whisper detected an Indic language, retry with Sarvam
            if result["language"] in INDIC_LANGS:
                logger.info("Whisper detected Indic language %s — retrying with Sarvam", result["language"])
                try:
                    return _transcribe_sarvam(audio_bytes, result["language"])
                except Exception as e:
                    logger.warning("Sarvam fallback failed: %s — using Whisper result", e)
            return result
    except Exception as exc:
        logger.error("STT failed: %s", exc)
        return {
            "transcript": "",
            "language": "en",
            "confidence": 0.0,
            "engine": "error",
            "needs_retry": True,
        }
