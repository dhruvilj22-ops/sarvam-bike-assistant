"""
Part 7 tests: TTS output.
Run: pytest tests/test_part_7.py -v
All tests use USE_MOCKS=true — zero real Sarvam/OpenAI TTS calls.
"""
import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from main import app
from output.tts import synthesize, _truncate_to_3_sentences


@pytest.fixture(scope="module", autouse=True)
def _mock_env():
    prev = {k: os.environ.get(k) for k in ("USE_MOCKS", "QDRANT_URL")}
    os.environ["USE_MOCKS"] = "true"
    os.environ["QDRANT_URL"] = ""
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


# ---------------------------------------------------------------------------
# Sentence truncation unit tests
# ---------------------------------------------------------------------------

def test_truncation_three_sentences():
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    result = _truncate_to_3_sentences(text)
    assert "Fourth" not in result
    assert "First" in result and "Third" in result


def test_truncation_two_sentences_unchanged():
    text = "First sentence. Second sentence."
    assert _truncate_to_3_sentences(text) == text


def test_truncation_one_sentence_unchanged():
    text = "Just one sentence here."
    assert _truncate_to_3_sentences(text) == text


def test_truncation_empty_string():
    assert _truncate_to_3_sentences("") == ""


def test_truncation_question_marks():
    text = "What is this? Is it serious? Should I worry? Maybe not."
    result = _truncate_to_3_sentences(text)
    assert "Maybe not" not in result


def test_truncation_exclamation_marks():
    text = "Warning! Stop immediately! Do not proceed! Call service."
    result = _truncate_to_3_sentences(text)
    assert "Call service" not in result


# ---------------------------------------------------------------------------
# synthesize() unit tests — routing and mock behavior
# ---------------------------------------------------------------------------

def test_synthesize_hindi_routes_to_sarvam():
    result = synthesize("इंजन की समस्या है।", language="hi", use_mocks=True)
    assert result["engine"] == "sarvam"


def test_synthesize_hindi_in_routes_to_sarvam():
    result = synthesize("white smoke", language="hi-IN", use_mocks=True)
    assert result["engine"] == "sarvam"


def test_synthesize_tamil_routes_to_sarvam():
    result = synthesize("engine problem", language="ta", use_mocks=True)
    assert result["engine"] == "sarvam"


def test_synthesize_english_routes_to_openai():
    result = synthesize("white smoke from exhaust", language="en", use_mocks=True)
    assert result["engine"] == "openai"


def test_synthesize_mock_returns_no_audio_bytes():
    result = synthesize("some text", language="en", use_mocks=True)
    assert result["audio_bytes"] is None


def test_synthesize_mock_returns_mocked_true():
    result = synthesize("some text", language="en", use_mocks=True)
    assert result["mocked"] is True


def test_synthesize_returns_text_field():
    result = synthesize("check the exhaust system.", language="en", use_mocks=True)
    assert "text" in result
    assert result["text"].strip() != ""


def test_synthesize_truncates_before_synthesis():
    long_text = "Sentence one. Sentence two. Sentence three. Sentence four. Sentence five."
    result = synthesize(long_text, language="en", use_mocks=True)
    assert "Sentence four" not in result["text"]
    assert "Sentence five" not in result["text"]


def test_synthesize_answer_text_too_long_gets_truncated():
    """Full answer_text should never reach TTS at full length."""
    answer = (
        "The Royal Enfield Meteor 350 uses a wet sump lubrication system. "
        "Engine oil capacity is 1.5 litres. "
        "Recommended grade is 15W-50. "
        "This is a fourth sentence that should be cut. "
        "And a fifth that definitely should not appear."
    )
    result = synthesize(answer, language="en", use_mocks=True)
    assert "fourth sentence" not in result["text"]


def test_synthesize_required_fields():
    result = synthesize("text", language="en", use_mocks=True)
    for field in ("audio_bytes", "content_type", "mocked", "engine", "text"):
        assert field in result, f"synthesize() missing field: {field}"


# ---------------------------------------------------------------------------
# POST /output/tts route tests
# ---------------------------------------------------------------------------

def test_tts_route_english_returns_200(client):
    r = client.post("/output/tts", json={"text": "white smoke from exhaust", "language": "en"})
    assert r.status_code == 200


def test_tts_route_hindi_returns_200(client):
    r = client.post("/output/tts", json={"text": "इंजन की जांच करें।", "language": "hi"})
    assert r.status_code == 200


def test_tts_route_mock_returns_mocked_true(client):
    r = client.post("/output/tts", json={"text": "check engine", "language": "en"})
    body = r.json()
    assert body["mocked"] is True


def test_tts_route_mock_returns_text(client):
    r = client.post("/output/tts", json={"text": "check engine oil level", "language": "en"})
    body = r.json()
    assert "text" in body
    assert body["text"].strip() != ""


def test_tts_route_hindi_mock_engine_is_sarvam(client):
    r = client.post("/output/tts", json={"text": "exhaust check", "language": "hi"})
    assert r.json()["engine"] == "sarvam"


def test_tts_route_english_mock_engine_is_openai(client):
    r = client.post("/output/tts", json={"text": "exhaust check", "language": "en"})
    assert r.json()["engine"] == "openai"


def test_tts_route_default_language_is_english(client):
    r = client.post("/output/tts", json={"text": "check exhaust"})
    assert r.status_code == 200
    assert r.json()["engine"] == "openai"


# ---------------------------------------------------------------------------
# /query route — voice_initiated wiring tests
# ---------------------------------------------------------------------------

def test_query_voice_initiated_true_has_tts(client):
    r = client.post("/query", json={
        "text": "white smoke from exhaust",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
        "voice_initiated": True,
    })
    assert r.status_code == 200
    body = r.json()
    assert "tts" in body
    assert body["tts"] is not None


def test_query_voice_initiated_false_tts_is_null(client):
    r = client.post("/query", json={
        "text": "white smoke from exhaust",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
        "voice_initiated": False,
    })
    assert r.status_code == 200
    assert r.json()["tts"] is None


def test_query_default_no_voice_initiated_tts_is_null(client):
    r = client.post("/query", json={
        "text": "engine oil check",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
    })
    assert r.status_code == 200
    assert r.json()["tts"] is None


def test_query_voice_tts_is_mocked(client):
    r = client.post("/query", json={
        "text": "white smoke",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
        "voice_initiated": True,
    })
    assert r.json()["tts"]["mocked"] is True


def test_query_voice_tts_text_is_spoken_summary_not_answer_text(client):
    """spoken_summary (short) must be used for TTS, not the full answer_text."""
    r = client.post("/query", json={
        "text": "white smoke",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
        "voice_initiated": True,
    })
    body = r.json()
    tts_text = body["tts"]["text"]
    answer_text = body.get("answer_text", "")
    # spoken_summary is shorter than answer_text
    assert len(tts_text) <= len(answer_text) + 5  # +5 for minor whitespace trim


def test_query_voice_tts_text_max_3_sentences(client):
    r = client.post("/query", json={
        "text": "white smoke",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
        "voice_initiated": True,
    })
    tts_text = r.json()["tts"]["text"]
    # Count sentence-ending punctuation occurrences
    import re
    sentences = re.split(r'(?<=[.!?])\s+', tts_text.strip())
    assert len(sentences) <= 3


def test_query_voice_tts_has_engine_field(client):
    r = client.post("/query", json={
        "text": "white smoke",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
        "voice_initiated": True,
    })
    assert "engine" in r.json()["tts"]
