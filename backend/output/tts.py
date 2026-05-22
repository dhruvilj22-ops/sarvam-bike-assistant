"""
TTS routing.
Indic languages → Sarvam TTS (bulbul:v1).
English (and undetected) → OpenAI TTS (tts-1).
USE_MOCKS=true skips audio generation entirely — returns text confirmation only.
Never sends more than 3 sentences to TTS regardless of input length.
"""
import base64
import logging
import os
import re
from typing import Dict

import requests

from input.stt import INDIC_LANGS

logger = logging.getLogger(__name__)

_INDIC_BASES = {lang.split("-")[0] for lang in INDIC_LANGS}

# Matches sentence-ending punctuation followed by whitespace
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+')


def _truncate_to_3_sentences(text: str) -> str:
    parts = _SENTENCE_END.split(text.strip())
    return " ".join(parts[:3])


def _sarvam_tts(text: str, language: str) -> Dict:
    lang_code = language if "-" in language else f"{language}-IN"
    resp = requests.post(
        "https://api.sarvam.ai/text-to-speech",
        headers={"api-subscription-key": os.getenv("SARVAM_API_KEY", "")},
        json={
            "inputs": [text],
            "target_language_code": lang_code,
            "speaker": "meera",
            "model": "bulbul:v1",
            "enable_preprocessing": True,
        },
        timeout=30,
    )
    resp.raise_for_status()
    audio_b64 = resp.json()["audios"][0]
    return {
        "audio_bytes": base64.b64decode(audio_b64),
        "content_type": "audio/wav",
        "mocked": False,
        "engine": "sarvam",
        "text": text,
    }


def _openai_tts(text: str) -> Dict:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )
    return {
        "audio_bytes": response.content,
        "content_type": "audio/mpeg",
        "mocked": False,
        "engine": "openai",
        "text": text,
    }


def synthesize(text: str, language: str = "en", use_mocks: bool = False) -> Dict:
    """
    Convert text to speech.
    Returns {audio_bytes, content_type, mocked, engine, text}.
    Truncates to 3 sentences before synthesis — spoken_summary should never be longer.
    """
    text = _truncate_to_3_sentences(text)

    is_indic = language.split("-")[0] in _INDIC_BASES

    if use_mocks:
        return {
            "audio_bytes": None,
            "content_type": "audio/mpeg",
            "mocked": True,
            "engine": "sarvam" if is_indic else "openai",
            "text": text,
        }

    try:
        if is_indic:
            return _sarvam_tts(text, language)
        else:
            return _openai_tts(text)
    except Exception as exc:
        logger.error("TTS failed: %s", exc)
        return {
            "audio_bytes": None,
            "content_type": "audio/mpeg",
            "mocked": True,
            "engine": "sarvam" if is_indic else "openai",
            "text": text,
            "error": str(exc),
        }
