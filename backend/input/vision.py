"""
GPT-4o vision: converts uploaded image to a retrieval-useful text description.
USE_MOCKS=true returns the same fixture description used in ingestion.
"""
import base64
import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)

_MOCK_DESCRIPTION = (
    "Image shows white smoke from rear exhaust pipe, "
    "indicating possible oil burning or coolant leak"
)
_MOCK_TECHNICAL_TERMS = ["exhaust smoke", "oil burning", "coolant leak", "rear exhaust"]

_VISION_PROMPT = (
    "Describe this motorcycle image for a service manual assistant. "
    "List: (1) what component or issue is visible, "
    "(2) any symptoms like smoke, leaks, or damage, "
    "(3) relevant technical terms a mechanic would use. "
    "Be concise and specific — this description will be used to search a service manual."
)


def _extract_terms(description: str) -> List[str]:
    """Pull likely technical terms from the description (simple heuristic)."""
    import re
    words = re.findall(r'\b[a-z][a-z\-]{3,}\b', description.lower())
    stop = {"this", "that", "with", "from", "into", "over", "also", "will", "been",
            "have", "would", "could", "should", "their", "there", "which", "when"}
    seen = set()
    terms = []
    for w in words:
        if w not in stop and w not in seen:
            seen.add(w)
            terms.append(w)
    return terms[:8]


def describe_image(
    image_bytes: bytes,
    mime_type: str = "image/jpeg",
    use_mocks: bool = False,
) -> Dict:
    """
    Returns {description, technical_terms}.
    """
    if use_mocks:
        return {
            "description": _MOCK_DESCRIPTION,
            "technical_terms": _MOCK_TECHNICAL_TERMS,
        }

    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        b64 = base64.b64encode(image_bytes).decode()
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    {"type": "text", "text": _VISION_PROMPT},
                ],
            }],
            max_tokens=300,
        )
        description = resp.choices[0].message.content.strip()
        return {
            "description": description,
            "technical_terms": _extract_terms(description),
        }
    except Exception as exc:
        logger.error("Vision call failed: %s", exc)
        return {
            "description": "",
            "technical_terms": [],
        }
