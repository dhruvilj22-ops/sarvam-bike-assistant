"""
LLM response generation with the anti-hallucination contract enforced at prompt level.
Layer 2: hard constraints in system prompt.
Layer 3: output validation — citation check, one regeneration attempt if missing.
Sarvam-m LLM is used for Indic language queries when SARVAM_API_KEY is set.
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

_FIXTURE_PATH = Path(__file__).parent.parent.parent / "tests" / "fixtures" / "sample_response.json"
_OOS_KEYWORDS = ("price", "cost", "buy", "color", "colour", "compare", "insurance",
                  "better", "worse", "purchase", "worth it", "should i buy",
                  "resale", "review", "rating", "second hand", "used bike")

_REQUIRED_FIELDS = ["answer_text", "spoken_summary", "citations",
                     "severity_label", "confidence", "suggested_followups"]

_INDIC_BASES = {"hi", "ta", "te", "kn", "mr", "bn", "gu", "pa", "ml", "or", "as"}

_SYSTEM_PROMPT = """\
You are a motorcycle service assistant for {bike} owners and mechanics in India.
Your job is to answer questions STRICTLY using the provided [CONTEXT] sections only.

CONTEXT CONFIDENCE: {confidence_level}
{language_instruction}
HARD RULES — follow these exactly:
1. Answer ONLY from the provided [CONTEXT] sections. Never use general automotive knowledge.
2. If [CONTEXT] does not contain the answer, respond with EXACTLY:
   "I couldn't find this in your {bike} manual. For this issue, I'd recommend visiting an authorised service center."
3. Every in-scope answer MUST cite the source section (section_number, section_title, page_number).
4. When CONTEXT CONFIDENCE is LOW, explicitly state limited context was found and recommend a service center.
5. Choose exactly one severity_label from:
   "Immediate Action Required" | "Get Checked Soon" | "Monitor for Now" | "Informational" | "N/A"

FORMATTING RULES for answer_text (markdown is rendered in the UI):
- Multi-step procedures: use a numbered list (1. 2. 3.).
- Specs or multiple items: use a bullet list (- item).
- Important warnings or values: use **bold**.
- Synthesise across all [CONTEXT] sections — do not repeat the same point from multiple sections.
- Keep answers focused: no filler phrases like "According to the manual..." or "Based on the context...".
- suggested_followups MUST only be questions answerable from the provided [CONTEXT]. Never suggest topics not covered in the retrieved sections.

Return ONLY valid JSON matching this exact schema (no markdown wrapper, no explanation):
{{
  "answer_text": "full answer with inline citation reference",
  "spoken_summary": "2-3 sentence voice-friendly summary",
  "citations": [{{"section_number": "", "section_title": "", "page_number": 0}}],
  "severity_label": "...",
  "confidence": "high or low",
  "suggested_followups": ["question1", "question2", "question3"]
}}\
"""


def _load_mock_fixture() -> Dict:
    return json.loads(_FIXTURE_PATH.read_text())


def _is_oos(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _OOS_KEYWORDS)


def _format_context(chunks: List[Tuple[Dict, float]]) -> str:
    parts = []
    for i, (chunk, score) in enumerate(chunks, 1):
        sec_num = chunk.get("section_number") or chunk.get("chapter_number") or ""
        sec_title = chunk.get("section_title") or chunk.get("chapter_title") or "Section"
        page = chunk.get("page_number", 0)
        label = f"Section {sec_num} — {sec_title} (page {page})" if sec_num else f"{sec_title} (page {page})"
        parts.append(f"[CONTEXT {i}] {label}\n{chunk['text']}")
    return "\n\n".join(parts)


def _call_llm(messages: List[Dict], language: str = "en") -> Dict:
    from openai import OpenAI

    base_lang = language.split("-")[0].lower()
    sarvam_key = os.getenv("SARVAM_API_KEY", "").strip()

    if base_lang in _INDIC_BASES and sarvam_key:
        client = OpenAI(
            api_key=sarvam_key,
            base_url="https://api.sarvam.ai/v1",
        )
        resp = client.chat.completions.create(
            model="sarvam-m",
            messages=messages,
            temperature=0.1,
        )
        raw = resp.choices[0].message.content
        # sarvam-m may return plain JSON or a JSON-fenced block
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].lstrip("json").strip()
        return json.loads(raw)

    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
    )
    resp = client.chat.completions.create(
        model="openai/gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw = resp.choices[0].message.content
    return json.loads(raw)


def validate_citation_against_retrieved(citation: Dict, retrieved_chunks: List[Tuple[Dict, float]]) -> bool:
    """
    Returns True if the citation's section_number or page_number matches at least one
    retrieved chunk — i.e., the citation is grounded in context, not hallucinated.
    Both section_number and page_number are checked; either match is sufficient.
    """
    cited_sec = str(citation.get("section_number", "")).strip()
    cited_page = int(citation.get("page_number", 0))
    for chunk, _ in retrieved_chunks:
        if cited_sec and str(chunk.get("section_number", "")).strip() == cited_sec:
            return True
        if cited_page and int(chunk.get("page_number", 0)) == cited_page:
            return True
    return False


def _safe_defaults(result: Dict, context_confidence: str) -> Dict:
    result.setdefault("answer_text", "")
    result.setdefault("spoken_summary", "")
    result.setdefault("citations", [])
    result.setdefault("severity_label", "Informational")
    result.setdefault("confidence", context_confidence)
    result.setdefault("suggested_followups", [])
    return result


def generate(
    query: str,
    chunks: List[Tuple[Dict, float]],
    context_confidence: str = "high",
    history_context: str = "",
    use_mocks: bool = False,
    language: str = "en",
) -> Dict:
    """
    Generate structured response. Validates citation presence — regenerates once if missing.
    Uses sarvam-m for Indic language queries when SARVAM_API_KEY is set.
    """
    if use_mocks:
        if _is_oos(query):
            return {
                "answer_text": (
                    "I couldn't find this in your manual. "
                    "For this issue, I'd recommend visiting an authorised service center."
                ),
                "spoken_summary": "This question is outside the scope of your manual.",
                "citations": [],
                "severity_label": "N/A",
                "confidence": "low",
                "suggested_followups": [],
            }
        return _load_mock_fixture()

    bike = "your bike"
    if chunks:
        chunk = chunks[0][0]
        brand = chunk.get("bike_brand", "")
        model = chunk.get("bike_model", "")
        bike = f"{brand} {model}".strip() or bike

    base_lang = language.split("-")[0].lower()
    if base_lang in _INDIC_BASES:
        lang_names = {
            "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada",
            "mr": "Marathi", "bn": "Bengali", "gu": "Gujarati",
            "pa": "Punjabi", "ml": "Malayalam", "or": "Odia", "as": "Assamese",
        }
        lang_name = lang_names.get(base_lang, "the user's language")
        language_instruction = (
            f"\nLANGUAGE: The user is writing in {lang_name}. "
            f"Respond in {lang_name}. Keep technical terms (section numbers, torque values, "
            f"part names) in English within the {lang_name} response.\n"
        )
    else:
        language_instruction = ""

    system = _SYSTEM_PROMPT.format(
        bike=bike,
        confidence_level=context_confidence.upper(),
        language_instruction=language_instruction,
    )
    logger.info(
        "generate_start language=%s context_confidence=%s chunks=%s bike=%s",
        language,
        context_confidence,
        len(chunks),
        bike,
    )

    context_block = _format_context(chunks)
    user_content = ""
    if history_context:
        user_content += f"[CONVERSATION HISTORY]\n{history_context}\n\n"
    user_content += f"[CONTEXT]\n{context_block}\n\n[QUESTION]\n{query}"

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]

    try:
        result = _call_llm(messages, language=language)
    except Exception as e:
        logger.error("Generator LLM call failed: %s", e)
        return _safe_defaults({}, context_confidence)

    # Layer 3 — output validation: citation check
    if not result.get("citations"):
        logger.warning("No citation in first generation — regenerating with stronger constraint")
        messages.append({
            "role": "user",
            "content": (
                "Your previous response was missing citations. "
                "You MUST include at least one citation from the [CONTEXT] provided. "
                "If no relevant context exists, use the exact refusal phrase."
            ),
        })
        try:
            result = _call_llm(messages, language=language)
        except Exception as e:
            logger.error("Generator regeneration failed: %s", e)
        if not result.get("citations"):
            logger.warning("No citation after regeneration — returning response without citation")
    logger.info(
        "generate_end citations=%s severity=%s confidence=%s answer_chars=%s",
        len(result.get("citations") or []),
        result.get("severity_label", ""),
        result.get("confidence", ""),
        len(result.get("answer_text", "") or ""),
    )

    return _safe_defaults(result, context_confidence)
