"""
Part 10 tests: multi-language support via Sarvam APIs.
- Language detection (Devanagari, Romanized Hindi, English)
- Transliteration map coverage (20+ phrases)
- translate_to_english expands Romanized Hindi for retrieval
- expand_query detects Hindi and expands it
- generator.generate() respects language parameter (mock mode)
- /query endpoint persists detected language to session
- Mid-conversation language switch is handled correctly
Run: pytest tests/test_part_10.py -v
All tests use USE_MOCKS=true — no real API calls.
"""
import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture(scope="module", autouse=True)
def _mock_env():
    prev = {k: os.environ.get(k) for k in ("USE_MOCKS", "SARVAM_API_KEY", "OPENROUTER_API_KEY", "QDRANT_URL")}
    os.environ["USE_MOCKS"] = "true"
    os.environ["SARVAM_API_KEY"] = ""
    os.environ["OPENROUTER_API_KEY"] = ""
    os.environ["QDRANT_URL"] = ""
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

from input.language import (
    detect_language,
    translate_to_english,
    translate_to_indic,
    transliterate_to_devanagari,
    TRANSLITERATION_MAP,
)


def test_detect_english():
    r = detect_language("How do I change the engine oil?", use_mocks=True)
    assert r["base_lang"] == "en"
    assert not r["is_indic"]


def test_detect_devanagari():
    r = detect_language("इंजन से आवाज आ रही है", use_mocks=True)
    assert r["base_lang"] == "hi"
    assert r["is_indic"]
    assert r["script_code"] == "Deva"


def test_detect_romanized_hindi_phrase():
    r = detect_language("engine garam ho rahi hai", use_mocks=True)
    assert r["base_lang"] == "hi"
    assert r["is_indic"]
    assert r["script_code"] == "Latn"


def test_detect_romanized_hindi_phrase_thak():
    r = detect_language("thak thak awaaz aa rahi hai", use_mocks=True)
    assert r["is_indic"]


def test_detect_mixed_english_unrecognized():
    """Pure English that matches no Hindi pattern → detected as English."""
    r = detect_language("spark plug replacement steps", use_mocks=True)
    assert r["base_lang"] == "en"
    assert not r["is_indic"]


# ---------------------------------------------------------------------------
# TRANSLITERATION_MAP coverage
# ---------------------------------------------------------------------------

def test_transliteration_map_has_at_least_20_entries():
    assert len(TRANSLITERATION_MAP) >= 20


def test_transliteration_map_has_engine_overheating():
    assert any("garam" in k for k in TRANSLITERATION_MAP)


def test_transliteration_map_has_oil_leak():
    assert any("tel" in k for k in TRANSLITERATION_MAP)


def test_transliteration_map_has_start_problem():
    assert any("start" in k for k in TRANSLITERATION_MAP)


def test_transliteration_map_has_brake_issue():
    assert any("brake" in k for k in TRANSLITERATION_MAP)


def test_transliteration_map_has_mileage():
    assert any("mileage" in k for k in TRANSLITERATION_MAP)


def test_transliteration_map_has_smoke():
    assert any("dhuan" in k or "dhuaan" in k for k in TRANSLITERATION_MAP)


# ---------------------------------------------------------------------------
# translate_to_english — phrase expansion
# ---------------------------------------------------------------------------

def test_translate_engine_garam_expands():
    result = translate_to_english("engine garam ho rahi hai", use_mocks=True)
    assert "overheat" in result.lower() or "temperature" in result.lower()


def test_translate_thak_thak_expands():
    result = translate_to_english("thak thak awaaz", use_mocks=True)
    assert "knock" in result.lower()


def test_translate_tel_leak_expands():
    result = translate_to_english("tel ka rissa", use_mocks=True)
    assert "oil" in result.lower()


def test_translate_safed_dhuan_expands():
    result = translate_to_english("safed dhuaan", use_mocks=True)
    assert "smoke" in result.lower() or "white" in result.lower()


def test_translate_start_nahi_expands():
    result = translate_to_english("start nahi ho raha", use_mocks=True)
    assert "start" in result.lower() or "crank" in result.lower()


def test_translate_english_passthrough():
    original = "How to change engine oil?"
    result = translate_to_english(original, use_mocks=True)
    assert result == original


def test_translate_to_indic_mock_passthrough():
    text = "Check the engine oil level."
    result = translate_to_indic(text, target_lang="hi-IN", use_mocks=True)
    assert result == text


def test_transliterate_mock_passthrough():
    text = "thak thak awaaz"
    result = transliterate_to_devanagari(text, use_mocks=True)
    assert result == text


# ---------------------------------------------------------------------------
# expand_query — language-aware expansion
# ---------------------------------------------------------------------------

from inference.expander import expand_query


def test_expand_query_english_no_change():
    r = expand_query("engine oil change steps", use_mocks=True)
    assert r["language"] == "en"
    assert r["expanded"] == "engine oil change steps"


def test_expand_query_romanized_hindi_detects_hi():
    r = expand_query("engine garam ho rahi hai", use_mocks=True)
    assert r["language"] == "hi"


def test_expand_query_romanized_hindi_expands_for_retrieval():
    r = expand_query("engine garam ho rahi hai", use_mocks=True)
    assert "overheat" in r["expanded"].lower() or "temperature" in r["expanded"].lower()


def test_expand_query_devanagari_detects_hi():
    r = expand_query("इंजन से आवाज आ रही है", use_mocks=True)
    assert r["language"] == "hi"


def test_expand_query_thak_thak_intent():
    r = expand_query("thak thak awaaz aa rahi hai", use_mocks=True)
    # knocking is a diagnostic symptom
    assert r["intent"] == "diagnostic"


def test_expand_query_has_all_fields():
    r = expand_query("brake kaam nahi kar raha", use_mocks=True)
    for field in ("original", "expanded", "intent", "language"):
        assert field in r, f"expand_query missing field: {field}"


def test_expand_query_original_preserved():
    text = "mileage kam ho gaya"
    r = expand_query(text, use_mocks=True)
    assert r["original"] == text


# ---------------------------------------------------------------------------
# generator.generate() — language parameter respected in mock
# ---------------------------------------------------------------------------

from inference.generator import generate


def test_generate_mock_returns_fixture():
    result = generate("engine oil change", [], use_mocks=True)
    assert "answer_text" in result
    assert "citations" in result


def test_generate_mock_oos_returns_refusal():
    result = generate("what is the price of this bike", [], use_mocks=True)
    assert "couldn't find" in result["answer_text"].lower() or "service center" in result["answer_text"].lower()
    assert result["severity_label"] == "N/A"


def test_generate_mock_hindi_returns_fixture():
    result = generate("इंजन ऑयल कैसे बदलें", [], use_mocks=True, language="hi")
    assert "answer_text" in result


def test_generate_accepts_language_param():
    for lang in ("en", "hi", "ta", "en-IN", "hi-IN"):
        result = generate("engine check", [], use_mocks=True, language=lang)
        assert "answer_text" in result


# ---------------------------------------------------------------------------
# /query endpoint — language persisted to session
# ---------------------------------------------------------------------------

from fastapi.testclient import TestClient
from main import app
from store import create_session, get_session


@pytest.fixture(scope="module")
def client():
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_query_returns_language_field(client):
    session_id = create_session()
    from ingestion.indexer import build_indexes, reset_qdrant_client
    from ingestion.embedder import embed_chunks

    reset_qdrant_client()
    doc_id = "test_lang_doc_10"
    chunks = [
        {
            "chunk_id": f"{doc_id}_c0",
            "document_id": doc_id,
            "text": "Engine oil: check level every 3000 km using dipstick.",
            "content_type": "prose",
            "chapter_number": "2",
            "chapter_title": "Maintenance",
            "section_number": "2.1",
            "section_title": "Engine Oil",
            "page_number": 5,
            "vector": [0.0] * 1536,
        }
    ]
    embedded = embed_chunks(chunks, use_mocks=True)
    build_indexes(embedded, doc_id)

    from store import create_thread
    thread = create_thread(session_id)

    r = client.post("/query", json={
        "text": "engine oil level check",
        "session_id": session_id,
        "document_id": doc_id,
        "thread_id": thread["thread_id"],
    })
    assert r.status_code == 200
    data = r.json()
    assert "language" in data


def test_query_english_sets_session_language_en(client):
    session_id = create_session()
    from ingestion.indexer import build_indexes, reset_qdrant_client
    from ingestion.embedder import embed_chunks

    reset_qdrant_client()
    doc_id = "test_lang_en_10"
    chunks = [
        {
            "chunk_id": f"{doc_id}_c0",
            "document_id": doc_id,
            "text": "Torque specification: tighten cylinder head bolts to 25 Nm.",
            "content_type": "prose",
            "chapter_number": "3",
            "chapter_title": "Engine",
            "section_number": "3.2",
            "section_title": "Cylinder Head",
            "page_number": 12,
            "vector": [0.0] * 1536,
        }
    ]
    embedded = embed_chunks(chunks, use_mocks=True)
    build_indexes(embedded, doc_id)

    from store import create_thread
    thread = create_thread(session_id)

    r = client.post("/query", json={
        "text": "what is the torque for cylinder head bolts",
        "session_id": session_id,
        "document_id": doc_id,
        "thread_id": thread["thread_id"],
    })
    assert r.status_code == 200
    session = get_session(session_id)
    assert session["language"] == "en"


def test_query_hindi_sets_session_language_hi(client):
    session_id = create_session()
    from ingestion.indexer import build_indexes, reset_qdrant_client
    from ingestion.embedder import embed_chunks

    reset_qdrant_client()
    doc_id = "test_lang_hi_10"
    chunks = [
        {
            "chunk_id": f"{doc_id}_c0",
            "document_id": doc_id,
            "text": "Engine oil change procedure: drain, refill with 10W-40.",
            "content_type": "prose",
            "chapter_number": "2",
            "chapter_title": "Maintenance",
            "section_number": "2.1",
            "section_title": "Engine Oil",
            "page_number": 8,
            "vector": [0.0] * 1536,
        }
    ]
    embedded = embed_chunks(chunks, use_mocks=True)
    build_indexes(embedded, doc_id)

    from store import create_thread
    thread = create_thread(session_id)

    # Romanized Hindi query — detect_language will flag is_indic=True
    r = client.post("/query", json={
        "text": "engine garam ho rahi hai",
        "session_id": session_id,
        "document_id": doc_id,
        "thread_id": thread["thread_id"],
    })
    assert r.status_code == 200
    session = get_session(session_id)
    assert session["language"] == "hi"


def test_query_language_switch_mid_conversation(client):
    """Session language updates on each turn; a switch from Hindi to English is tracked."""
    session_id = create_session()
    from ingestion.indexer import build_indexes, reset_qdrant_client
    from ingestion.embedder import embed_chunks

    reset_qdrant_client()
    doc_id = "test_switch_10"
    chunks = [
        {
            "chunk_id": f"{doc_id}_c0",
            "document_id": doc_id,
            "text": "Brake adjustment: turn adjuster clockwise to tighten cable.",
            "content_type": "prose",
            "chapter_number": "4",
            "chapter_title": "Brakes",
            "section_number": "4.1",
            "section_title": "Brake Adjustment",
            "page_number": 20,
            "vector": [0.0] * 1536,
        }
    ]
    embedded = embed_chunks(chunks, use_mocks=True)
    build_indexes(embedded, doc_id)

    from store import create_thread
    thread = create_thread(session_id)
    tid = thread["thread_id"]

    # Turn 1 — Hindi
    r1 = client.post("/query", json={
        "text": "brake kaam nahi kar raha",
        "session_id": session_id,
        "document_id": doc_id,
        "thread_id": tid,
    })
    assert r1.status_code == 200
    assert get_session(session_id)["language"] == "hi"

    # Turn 2 — English
    r2 = client.post("/query", json={
        "text": "how to adjust the brake cable",
        "session_id": session_id,
        "document_id": doc_id,
        "thread_id": tid,
    })
    assert r2.status_code == 200
    assert get_session(session_id)["language"] == "en"


# ---------------------------------------------------------------------------
# TTS routing — Indic language uses sarvam engine in mock
# ---------------------------------------------------------------------------

from output.tts import synthesize


def test_tts_indic_language_uses_sarvam_engine_mock():
    result = synthesize("engine oil kaise badlein", "hi", use_mocks=True)
    assert result["engine"] == "sarvam"
    assert result["mocked"] is True


def test_tts_english_uses_openai_engine_mock():
    result = synthesize("Check the engine oil level.", "en", use_mocks=True)
    assert result["engine"] == "openai"
    assert result["mocked"] is True


def test_tts_hindi_in_returns_sarvam():
    result = synthesize("brake tight ho gaya hai", "hi-IN", use_mocks=True)
    assert result["engine"] == "sarvam"


def test_tts_tamil_uses_sarvam():
    result = synthesize("engine oil maaruvadhu eppadhi", "ta", use_mocks=True)
    assert result["engine"] == "sarvam"
