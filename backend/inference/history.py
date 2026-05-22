"""
Per-thread conversation history management.
Compresses older turns into a summary after SUMMARY_THRESHOLD complete turns.
"""
import logging
import os
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SUMMARY_THRESHOLD = 4  # compress after this many complete turns (user+assistant pairs)

_history: Dict[str, List[Dict]] = {}
_summaries: Dict[str, str] = {}


def add_turn(thread_id: str, user_query: str, assistant_response: str) -> None:
    if thread_id not in _history:
        _history[thread_id] = []
    _history[thread_id].append({"role": "user", "content": user_query})
    _history[thread_id].append({"role": "assistant", "content": assistant_response})


def _summarize(turns: List[Dict], use_mocks: bool = False) -> str:
    if use_mocks or not os.getenv("OPENROUTER_API_KEY", "").strip():
        n = len(turns) // 2
        return f"[Earlier conversation summary: {n} turns about motorcycle maintenance.]"

    try:
        from openai import OpenAI

        client = OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
        )
        formatted = "\n".join(
            f"{t['role'].upper()}: {t['content']}" for t in turns
        )
        resp = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": (
                    "Summarize this motorcycle service conversation in 2-3 sentences. "
                    "Keep all technical details, symptoms, and recommendations.\n\n"
                    + formatted
                ),
            }],
            temperature=0,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning("History summarization failed: %s", e)
        n = len(turns) // 2
        return f"[Earlier conversation: {n} turns about motorcycle issue.]"


def get_context_history(
    thread_id: str,
    use_mocks: bool = False,
    max_recent_turns: int = 2,
) -> str:
    """
    Return formatted history string for the LLM context window.
    Compresses turns older than max_recent_turns into a summary after SUMMARY_THRESHOLD turns.
    """
    turns = _history.get(thread_id, [])
    if not turns:
        return ""

    total_turns = len(turns) // 2  # each turn = 1 user + 1 assistant message
    recent_messages = max_recent_turns * 2  # messages = turns * 2

    if total_turns >= SUMMARY_THRESHOLD and thread_id not in _summaries:
        older = turns[:-recent_messages]
        if older:
            _summaries[thread_id] = _summarize(older, use_mocks=use_mocks)
            _history[thread_id] = turns[-recent_messages:]
            turns = _history[thread_id]

    recent = turns[-recent_messages:] if len(turns) > recent_messages else turns
    lines = []

    if thread_id in _summaries:
        lines.append(f"[Earlier conversation summary]: {_summaries[thread_id]}")

    for msg in recent:
        lines.append(f"{msg['role'].upper()}: {msg['content']}")

    return "\n".join(lines)


def reset_thread(thread_id: str) -> None:
    """Remove all history and summary for a thread. Used in tests."""
    _history.pop(thread_id, None)
    _summaries.pop(thread_id, None)
