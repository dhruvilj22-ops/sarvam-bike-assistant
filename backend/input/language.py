"""
Sarvam language utilities for multi-language support.
  - Language detection  : POST /text-lid
  - Transliteration     : POST /transliterate  (Romanized Hindi → Devanagari)
  - Translation         : POST /translate       (Hindi ↔ English)

Mock mode is fully functional — no Sarvam credits spent during tests.
"""
import logging
import os
import re
from typing import Dict

import requests

logger = logging.getLogger(__name__)

_SARVAM_API = "https://api.sarvam.ai"
_INDIC_BASES = {"hi", "ta", "te", "kn", "mr", "bn", "gu", "pa", "ml", "or", "as"}
_DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")

# Common Romanized Hindi motorcycle symptom phrases → English retrieval terms
# At least 5 required by PLAN.md; 20 provided for real-world coverage.
TRANSLITERATION_MAP: Dict[str, str] = {
    "thak thak awaaz":              "knocking sound engine",
    "thak thak sound":              "knocking sound engine",
    "thak thak ki awaaz":           "knocking sound engine",
    "engine garam ho rahi hai":     "engine overheating high temperature",
    "engine garam":                 "engine overheating",
    "garam ho raha hai":            "overheating",
    "safed dhuaan":                 "white smoke exhaust",
    "safed dhuan":                  "white smoke exhaust",
    "dhuaan aa raha hai":           "smoke from exhaust pipe",
    "tel ka rissa":                 "oil leak engine",
    "tel nikal raha hai":           "oil leaking",
    "tel leak":                     "oil leak",
    "start nahi ho raha":           "engine not starting cranking",
    "start nahi hoti":              "engine not starting",
    "band ho gaya":                 "engine stalled stopped",
    "awaaz aa rahi hai":            "unusual noise vibration",
    "jyada awaaz":                  "excessive noise engine",
    "mileage kam ho gaya":          "fuel economy reduced poor mileage",
    "mileage kam":                  "reduced fuel economy",
    "petrol jyada lag raha hai":    "high fuel consumption",
    "brake kaam nahi kar raha":     "brake failure not working",
    "brake tight ho gaya":          "brake sticking adjustment",
    "gear nahi lag raha":           "gear shifting problem transmission",
    "vibration aa rahi hai":        "excessive vibration chassis",
    "light nahi jal rahi":          "electrical fault light not working",
}


def _sarvam_headers() -> Dict[str, str]:
    return {"api-subscription-key": os.getenv("SARVAM_API_KEY", "")}


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def detect_language(text: str, use_mocks: bool = False) -> Dict:
    """
    Returns {language_code, script_code, is_indic, base_lang}.
    Real: Sarvam text-lid API.
    Mock: Devanagari regex + transliteration map pattern match.
    """
    if use_mocks or not os.getenv("SARVAM_API_KEY", "").strip():
        return _detect_mock(text)
    try:
        resp = requests.post(
            f"{_SARVAM_API}/text-lid",
            headers=_sarvam_headers(),
            json={"input": text[:1000]},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        lang_code = data.get("language_code", "en-IN")
        base = lang_code.split("-")[0]
        return {
            "language_code": lang_code,
            "script_code": data.get("script_code", "Latn"),
            "is_indic": base in _INDIC_BASES,
            "base_lang": base,
        }
    except Exception as exc:
        logger.warning("Sarvam language detection failed: %s — using mock", exc)
        return _detect_mock(text)


def _detect_mock(text: str) -> Dict:
    if _DEVANAGARI_RE.search(text):
        return {"language_code": "hi-IN", "script_code": "Deva", "is_indic": True, "base_lang": "hi"}
    if any(phrase in text.lower() for phrase in TRANSLITERATION_MAP):
        return {"language_code": "hi-IN", "script_code": "Latn", "is_indic": True, "base_lang": "hi"}
    return {"language_code": "en-IN", "script_code": "Latn", "is_indic": False, "base_lang": "en"}


# ---------------------------------------------------------------------------
# Transliteration — Romanized Hindi → Devanagari
# ---------------------------------------------------------------------------

def transliterate_to_devanagari(text: str, use_mocks: bool = False) -> str:
    """Converts Romanized Hindi to Devanagari. Mock: returns text unchanged."""
    if use_mocks or not os.getenv("SARVAM_API_KEY", "").strip():
        return text
    try:
        resp = requests.post(
            f"{_SARVAM_API}/transliterate",
            headers=_sarvam_headers(),
            json={
                "input": text,
                "source_language_code": "hi-Latn",
                "target_language_code": "hi-IN",
            },
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json().get("transliterated_text", text)
    except Exception as exc:
        logger.warning("Sarvam transliteration failed: %s", exc)
        return text


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------

def translate_to_english(text: str, source_lang: str = "hi-IN", use_mocks: bool = False) -> str:
    """
    Translates Hindi (or other Indic) text to English for retrieval.
    Mock: applies TRANSLITERATION_MAP phrase substitutions.
    """
    if use_mocks or not os.getenv("SARVAM_API_KEY", "").strip():
        return _apply_transliteration_map(text)
    try:
        resp = requests.post(
            f"{_SARVAM_API}/translate",
            headers=_sarvam_headers(),
            json={
                "input": text,
                "source_language_code": source_lang,
                "target_language_code": "en-IN",
                "mode": "formal",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("translated_text", text)
    except Exception as exc:
        logger.warning("Sarvam translation to English failed: %s", exc)
        return _apply_transliteration_map(text)


def translate_to_indic(text: str, target_lang: str = "hi-IN", use_mocks: bool = False) -> str:
    """Translates English answer to target Indic language. Mock: returns as-is."""
    if use_mocks or not os.getenv("SARVAM_API_KEY", "").strip():
        return text
    try:
        resp = requests.post(
            f"{_SARVAM_API}/translate",
            headers=_sarvam_headers(),
            json={
                "input": text,
                "source_language_code": "en-IN",
                "target_language_code": target_lang,
                "mode": "formal",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("translated_text", text)
    except Exception as exc:
        logger.warning("Sarvam translation to %s failed: %s", target_lang, exc)
        return text


def _apply_transliteration_map(text: str) -> str:
    """Apply TRANSLITERATION_MAP substitutions for mock mode query expansion."""
    result = text
    for phrase, expansion in TRANSLITERATION_MAP.items():
        if phrase in result.lower():
            result = re.sub(re.escape(phrase), expansion, result, flags=re.IGNORECASE)
    return result
