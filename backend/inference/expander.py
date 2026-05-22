"""
Query expansion and intent classification.
Mock: keyword regex (no LLM call) + Sarvam language detection.
Real: OpenRouter GPT-4o-mini returns {expanded_query, intent, language}.
"""
import json
import logging
import os
import re
from typing import Dict

from input.language import detect_language, translate_to_english

logger = logging.getLogger(__name__)

_OOS_RE = re.compile(
    r'\b(price|cost|buy|color|colour|compare|insurance|better|worse|review|purchase|worth it)\b',
    re.IGNORECASE,
)
_SPEC_RE = re.compile(
    r'\b(torque|specification|spec|capacity|dimension|pressure|voltage|clearance|mm|nm|rpm|bhp|litre|liter|cc|kpa)\b',
    re.IGNORECASE,
)
_PROC_RE = re.compile(
    r'\b(how to|how do|steps|procedure|change|replace|install|remove|drain|adjust|clean|inspect|tighten)\b',
    re.IGNORECASE,
)


def _classify_mock(text: str) -> str:
    if _OOS_RE.search(text):
        return "out_of_scope"
    if _SPEC_RE.search(text):
        return "specification"
    if _PROC_RE.search(text):
        return "procedure"
    return "diagnostic"


def expand_query(text: str, use_mocks: bool = False) -> Dict:
    """
    Returns {original, expanded, intent, language}.
    intent: diagnostic | specification | procedure | out_of_scope
    language: base language code (e.g. "en", "hi", "ta")
    """
    if use_mocks or not os.getenv("OPENROUTER_API_KEY", "").strip():
        lang_info = detect_language(text, use_mocks=True)
        base_lang = lang_info.get("base_lang", "en")
        if lang_info.get("is_indic"):
            expanded = translate_to_english(
                text,
                source_lang=lang_info.get("language_code", "hi-IN"),
                use_mocks=True,
            )
        else:
            expanded = text
        return {
            "original": text,
            "expanded": expanded,
            "intent": _classify_mock(expanded),
            "language": base_lang,
        }

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "You are a query classifier for a motorcycle service manual assistant.\n"
                    "Classify the user query and expand it with relevant technical synonyms.\n"
                    "Return JSON with:\n"
                    '- "intent": one of "diagnostic" (symptoms/problems), "specification" '
                    '(specs/values/measurements), "procedure" (how-to/steps), '
                    '"out_of_scope" (not about bike maintenance)\n'
                    '- "expanded_query": rewrite with technical synonyms for better retrieval\n'
                    '- "language": ISO 639-1 code\n\n'
                    f"Query: {text}\n\nReturn only valid JSON."
                ),
            }],
            response_format={"type": "json_object"},
            temperature=0,
        )
        data = json.loads(resp.choices[0].message.content)
        return {
            "original": text,
            "expanded": data.get("expanded_query", text),
            "intent": data.get("intent", "diagnostic"),
            "language": data.get("language", "en"),
        }
    except Exception:
        logger.warning("Expander LLM call failed — falling back to mock classification")
        return {
            "original": text,
            "expanded": text,
            "intent": _classify_mock(text),
            "language": "en",
        }
