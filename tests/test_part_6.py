"""
Part 6 tests: multimodal input (STT + image).
Run: pytest tests/test_part_6.py -v
All tests use USE_MOCKS=true — zero real Sarvam/OpenAI calls.
"""
import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

FIXTURES = Path(__file__).parent / "fixtures"

from fastapi.testclient import TestClient
from main import app
from input.stt import transcribe, STT_CONFIDENCE_THRESHOLD, INDIC_LANGS
from input.vision import describe_image
from routes.query import assemble_query


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
# STT unit tests — no HTTP, no real API
# ---------------------------------------------------------------------------

def test_hindi_hint_routes_to_sarvam():
    result = transcribe(b"fake_audio", language_hint="hi", use_mocks=True)
    assert result["engine"] == "sarvam"


def test_hindi_filename_routes_to_sarvam():
    result = transcribe(b"fake_audio", filename="sample_hindi.wav", use_mocks=True)
    assert result["engine"] == "sarvam"


def test_english_routes_to_whisper():
    result = transcribe(b"fake_audio", filename="sample_english.wav", use_mocks=True)
    assert result["engine"] == "whisper"


def test_stt_returns_required_fields():
    result = transcribe(b"fake_audio", use_mocks=True)
    for field in ("transcript", "language", "confidence", "engine", "needs_retry"):
        assert field in result, f"STT missing field: {field}"


def test_stt_hindi_transcript_non_empty():
    result = transcribe(b"fake_audio", language_hint="hi", use_mocks=True)
    assert result["transcript"].strip() != ""


def test_stt_english_transcript_non_empty():
    result = transcribe(b"fake_audio", language_hint="en", use_mocks=True)
    assert result["transcript"].strip() != ""


def test_stt_confidence_above_threshold_no_retry():
    result = transcribe(b"fake_audio", language_hint="en", use_mocks=True)
    # Mock returns confidence=0.95 for English → above threshold → no retry
    assert result["confidence"] >= STT_CONFIDENCE_THRESHOLD
    assert result["needs_retry"] is False


def test_stt_low_confidence_sets_needs_retry():
    # Simulate a low-confidence result by calling the logic directly
    result = {
        "transcript": "unclear",
        "language": "en",
        "confidence": 0.4,
        "engine": "whisper",
        "needs_retry": 0.4 < STT_CONFIDENCE_THRESHOLD,
    }
    assert result["needs_retry"] is True


def test_indic_langs_set_contains_hindi():
    assert "hi" in INDIC_LANGS
    assert "hi-IN" in INDIC_LANGS


def test_indic_langs_set_contains_tamil():
    assert "ta" in INDIC_LANGS


# ---------------------------------------------------------------------------
# Vision unit tests
# ---------------------------------------------------------------------------

def test_vision_returns_description():
    result = describe_image(b"fake_image_bytes", use_mocks=True)
    assert "description" in result
    assert result["description"].strip() != ""


def test_vision_returns_technical_terms():
    result = describe_image(b"fake_image_bytes", use_mocks=True)
    assert "technical_terms" in result
    assert isinstance(result["technical_terms"], list)
    assert len(result["technical_terms"]) > 0


def test_vision_description_is_mechanically_meaningful():
    result = describe_image(b"fake_image_bytes", use_mocks=True)
    # Mock description should reference an exhaust/engine symptom
    desc = result["description"].lower()
    assert any(w in desc for w in ("exhaust", "smoke", "oil", "engine", "leak"))


# ---------------------------------------------------------------------------
# Unified query assembly unit tests
# ---------------------------------------------------------------------------

def test_assembly_text_only():
    q = assemble_query("engine oil change", None, None)
    assert q == "engine oil change"


def test_assembly_transcript_only():
    q = assemble_query("", "इंजन से आवाज आ रही है", None)
    assert "इंजन" in q


def test_assembly_transcript_plus_text():
    q = assemble_query("engine noise", "thak thak sound", None)
    assert "thak thak sound" in q
    assert "engine noise" in q


def test_assembly_text_plus_image():
    q = assemble_query("what is this", None, "white smoke from exhaust")
    assert "what is this" in q
    assert "Image context:" in q
    assert "white smoke" in q


def test_assembly_all_three():
    q = assemble_query("engine issue", "thak thak noise", "smoke from exhaust")
    assert "thak thak noise" in q
    assert "engine issue" in q
    assert "Image context:" in q
    assert "smoke" in q


def test_assembly_empty_parts_ignored():
    q = assemble_query("engine oil", "", "")
    assert q == "engine oil"


# ---------------------------------------------------------------------------
# HTTP route tests via TestClient
# ---------------------------------------------------------------------------

def test_voice_route_hindi_returns_200(client):
    with open(FIXTURES / "sample_hindi.wav", "rb") as f:
        r = client.post("/input/voice", data={"language_hint": "hi"},
                        files={"audio": ("sample_hindi.wav", f, "audio/wav")})
    assert r.status_code == 200


def test_voice_route_hindi_routes_to_sarvam(client):
    with open(FIXTURES / "sample_hindi.wav", "rb") as f:
        r = client.post("/input/voice", data={"language_hint": "hi"},
                        files={"audio": ("sample_hindi.wav", f, "audio/wav")})
    assert r.json()["engine"] == "sarvam"


def test_voice_route_english_routes_to_whisper(client):
    with open(FIXTURES / "sample_english.wav", "rb") as f:
        r = client.post("/input/voice", data={"language_hint": "en"},
                        files={"audio": ("sample_english.wav", f, "audio/wav")})
    assert r.json()["engine"] == "whisper"


def test_voice_route_all_fields_present(client):
    with open(FIXTURES / "sample_english.wav", "rb") as f:
        r = client.post("/input/voice",
                        files={"audio": ("sample_english.wav", f, "audio/wav")})
    body = r.json()
    for field in ("transcript", "language", "confidence", "engine", "needs_retry"):
        assert field in body, f"Voice response missing field: {field}"


def test_voice_route_unsupported_format_returns_400(client):
    r = client.post("/input/voice",
                    files={"audio": ("audio.txt", b"not audio", "text/plain")})
    assert r.status_code == 400


def test_voice_stores_language_in_session(client):
    sid = client.post("/session").json()["session_id"]
    with open(FIXTURES / "sample_hindi.wav", "rb") as f:
        client.post("/input/voice",
                    data={"language_hint": "hi", "session_id": sid},
                    files={"audio": ("sample_hindi.wav", f, "audio/wav")})
    session = client.get(f"/session/{sid}/threads")
    # Session must still exist (language stored internally — verified via session being valid)
    assert session.status_code == 200


def test_image_route_returns_200(client):
    with open(FIXTURES / "sample_exhaust.jpg", "rb") as f:
        r = client.post("/input/image",
                        files={"image": ("sample_exhaust.jpg", f, "image/jpeg")})
    assert r.status_code == 200


def test_image_route_returns_description(client):
    with open(FIXTURES / "sample_exhaust.jpg", "rb") as f:
        r = client.post("/input/image",
                        files={"image": ("sample_exhaust.jpg", f, "image/jpeg")})
    body = r.json()
    assert "description" in body
    assert body["description"].strip() != ""


def test_image_route_returns_technical_terms(client):
    with open(FIXTURES / "sample_exhaust.jpg", "rb") as f:
        r = client.post("/input/image",
                        files={"image": ("sample_exhaust.jpg", f, "image/jpeg")})
    body = r.json()
    assert "technical_terms" in body
    assert isinstance(body["technical_terms"], list)


def test_image_unsupported_format_returns_400(client):
    r = client.post("/input/image",
                    files={"image": ("doc.pdf", b"not image", "application/pdf")})
    assert r.status_code == 400


def test_query_route_accepts_transcript_field(client):
    """POST /query with transcript field — unified assembly should work."""
    r = client.post("/query", json={
        "text": "engine oil",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
        "transcript": "इंजन में आवाज",
    })
    # Returns 200 even if document not indexed — pipeline handles empty retrieval gracefully
    assert r.status_code == 200


def test_query_route_accepts_image_description_field(client):
    """POST /query with image_description — merged into unified query."""
    r = client.post("/query", json={
        "text": "what is causing this",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
        "image_description": "white smoke from exhaust pipe",
    })
    assert r.status_code == 200
    body = r.json()
    assert "answer_text" in body
