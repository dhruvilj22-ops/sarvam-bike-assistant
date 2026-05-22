"""
Proper regression tests for Bucket B, C, D changes.

Bucket B: library source filtering, dedup, job status strings, document_index default
Bucket C: system prompt formatting rules, top_n=5, reranker mock scores
Bucket D: starters endpoint — chapter mapping, skip-word filtering, dedup, fallback, padding

Run: pytest tests/test_bucket_fixes.py -v
All tests use USE_MOCKS=true.
"""
import json
import os
import sys
import pytest
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "backend"))


@pytest.fixture(scope="module", autouse=True)
def _mock_env():
    prev = {k: os.environ.get(k) for k in ("USE_MOCKS", "QDRANT_URL", "OPENROUTER_API_KEY")}
    os.environ["USE_MOCKS"] = "true"
    os.environ["QDRANT_URL"] = ""
    os.environ["OPENROUTER_API_KEY"] = ""
    yield
    for k, v in prev.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _write_index(
    directory: Path,
    document_id: str,
    brand: str = "Royal Enfield",
    model: str = "Meteor 350",
    manual_type: str = "service_manual",
    source: str = "library",
    timestamp: str = "2024-01-01T00:00:00+00:00",
    chapters: list = None,
) -> None:
    data = {
        "document_id": document_id,
        "bike_brand": brand,
        "bike_model": model,
        "bike_year": "2022",
        "manual_type": manual_type,
        "manual_source": source,
        "total_chunks": 50,
        "ingestion_timestamp": timestamp,
        "chapters": chapters or [],
    }
    (directory / f"{document_id}_index.json").write_text(json.dumps(data))


# ===========================================================================
# Bucket B — Library filtering
# ===========================================================================

import routes.bikes as bikes_module


def test_library_excludes_user_uploaded(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    _write_index(tmp_path, "doc_user", source="user_uploaded")
    result = bikes_module.library()
    assert result["bikes"] == [], "user_uploaded entries must not appear in /bikes/library"


def test_library_includes_library_source(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    _write_index(tmp_path, "doc_lib", brand="Honda", model="CB300", source="library")
    result = bikes_module.library()
    assert len(result["bikes"]) == 1
    assert result["bikes"][0]["bike_brand"] == "Honda"


def test_library_mixed_sources_only_returns_library(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    _write_index(tmp_path, "doc_a", brand="Honda", model="CB300", source="library")
    _write_index(tmp_path, "doc_b", brand="Yamaha", model="R15", source="user_uploaded")
    result = bikes_module.library()
    ids = [b["document_id"] for b in result["bikes"]]
    assert "doc_a" in ids
    assert "doc_b" not in ids


def test_library_no_manual_source_field_treated_as_user_uploaded(tmp_path, monkeypatch):
    """Old index files without manual_source must NOT pollute the library tab."""
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    # Write a file with no manual_source key (simulates a pre-fix index file)
    data = {
        "document_id": "old_doc",
        "bike_brand": "Bajaj",
        "bike_model": "Pulsar",
        "bike_year": "2020",
        "manual_type": "service_manual",
        "total_chunks": 10,
        "ingestion_timestamp": "2023-01-01T00:00:00+00:00",
        "chapters": [],
    }
    (tmp_path / "old_doc_index.json").write_text(json.dumps(data))
    result = bikes_module.library()
    assert result["bikes"] == [], "Index file without manual_source must default to user_uploaded"


# ===========================================================================
# Bucket B — Library dedup
# ===========================================================================

def test_library_dedup_same_brand_model_type_keeps_latest(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    _write_index(tmp_path, "doc_old", timestamp="2024-01-01T00:00:00+00:00")
    _write_index(tmp_path, "doc_new", timestamp="2025-06-01T00:00:00+00:00")
    result = bikes_module.library()
    assert len(result["bikes"]) == 1, "Same brand+model+type must dedup to one entry"
    assert result["bikes"][0]["document_id"] == "doc_new"


def test_library_dedup_different_manual_types_both_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    _write_index(tmp_path, "doc_svc", manual_type="service_manual")
    _write_index(tmp_path, "doc_own", manual_type="owner_manual")
    result = bikes_module.library()
    assert len(result["bikes"]) == 2, "Different manual_types must not be merged"


def test_library_dedup_different_models_both_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    _write_index(tmp_path, "doc_m1", model="Meteor 350")
    _write_index(tmp_path, "doc_m2", model="Classic 350")
    result = bikes_module.library()
    assert len(result["bikes"]) == 2


def test_library_returns_empty_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path / "nonexistent")
    result = bikes_module.library()
    assert result["bikes"] == []


def test_library_skips_malformed_json_gracefully(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    (tmp_path / "broken_index.json").write_text("{ not valid json }")
    _write_index(tmp_path, "doc_good", brand="Honda", model="CB300", source="library")
    result = bikes_module.library()
    assert len(result["bikes"]) == 1, "Malformed JSON must be skipped without crashing"


# ===========================================================================
# Bucket B — Job status strings
# ===========================================================================

import store


def test_job_initial_status_is_pending():
    job_id = store.create_job()
    assert store.get_job(job_id)["status"] == "pending", \
        "New jobs must start as 'pending', not 'submitted'"


def test_job_initial_progress_is_zero():
    job_id = store.create_job()
    assert store.get_job(job_id)["progress_pct"] == 0


def test_job_can_be_set_to_error():
    job_id = store.create_job()
    store.update_job(job_id, status="error", message="Disk full")
    assert store.get_job(job_id)["status"] == "error", \
        "Error path must set status='error' (frontend polls for this exact string)"


def test_job_error_message_is_preserved():
    job_id = store.create_job()
    store.update_job(job_id, status="error", message="PDF parse failed")
    assert store.get_job(job_id)["message"] == "PDF parse failed"


def test_job_complete_sets_document_id():
    job_id = store.create_job()
    store.update_job(job_id, status="complete", progress_pct=100, document_id="doc_xyz")
    job = store.get_job(job_id)
    assert job["status"] == "complete"
    assert job["document_id"] == "doc_xyz"


# ===========================================================================
# Bucket B — document_index default source
# ===========================================================================

import ingestion.document_index as di_module
from ingestion.document_index import generate_document_index


def test_document_index_defaults_to_user_uploaded(tmp_path, monkeypatch):
    monkeypatch.setattr(di_module, "_INDEX_DIR", tmp_path)
    doc_meta = {
        "bike_brand": "Honda", "bike_model": "CB300",
        "bike_year": "2022", "manual_type": "service_manual",
        # no manual_source key
    }
    index = generate_document_index("test_default_src", doc_meta, [])
    assert index["manual_source"] == "user_uploaded", \
        "generate_document_index must default to 'user_uploaded', not 'library'"


def test_document_index_respects_explicit_library_source(tmp_path, monkeypatch):
    monkeypatch.setattr(di_module, "_INDEX_DIR", tmp_path)
    doc_meta = {
        "bike_brand": "Honda", "bike_model": "CB300",
        "bike_year": "2022", "manual_type": "service_manual",
        "manual_source": "library",
    }
    index = generate_document_index("test_library_src", doc_meta, [])
    assert index["manual_source"] == "library"


def test_document_index_written_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(di_module, "_INDEX_DIR", tmp_path)
    doc_meta = {
        "bike_brand": "Honda", "bike_model": "CB300",
        "bike_year": "2022", "manual_type": "service_manual",
    }
    generate_document_index("test_disk_write", doc_meta, [])
    assert (tmp_path / "test_disk_write_index.json").exists()


# ===========================================================================
# Bucket C — System prompt formatting rules
# ===========================================================================

from inference.generator import _SYSTEM_PROMPT


def test_system_prompt_has_formatting_rules_section():
    assert "FORMATTING RULES" in _SYSTEM_PROMPT


def test_system_prompt_numbered_list_instruction():
    assert "numbered list" in _SYSTEM_PROMPT or "1. 2. 3." in _SYSTEM_PROMPT


def test_system_prompt_bullet_list_instruction():
    assert "bullet list" in _SYSTEM_PROMPT or "- item" in _SYSTEM_PROMPT


def test_system_prompt_bold_instruction():
    assert "**bold**" in _SYSTEM_PROMPT or "bold" in _SYSTEM_PROMPT.lower()


def test_system_prompt_followups_grounding_constraint():
    assert "suggested_followups" in _SYSTEM_PROMPT
    assert "CONTEXT" in _SYSTEM_PROMPT


def test_system_prompt_no_filler_phrases_instruction():
    assert "filler" in _SYSTEM_PROMPT or "According to the manual" in _SYSTEM_PROMPT


def test_system_prompt_synthesise_instruction():
    assert "ynthesi" in _SYSTEM_PROMPT  # "Synthesise" or "Synthesize"


def test_system_prompt_hard_rules_still_present():
    assert "HARD RULES" in _SYSTEM_PROMPT
    assert "authorised service center" in _SYSTEM_PROMPT


# ===========================================================================
# Bucket C — Reranker: mock scores and top_n=5
# ===========================================================================

from inference.reranker import rerank, _MOCK_SCORES


def test_mock_scores_has_five_entries():
    assert len(_MOCK_SCORES) == 5, \
        f"_MOCK_SCORES must have 5 entries for top_n=5, got {len(_MOCK_SCORES)}"


def test_mock_scores_are_descending():
    assert _MOCK_SCORES == sorted(_MOCK_SCORES, reverse=True)


def test_rerank_returns_five_results_in_mock():
    chunks = [{"text": f"chunk {i}", "chunk_id": f"c{i}"} for i in range(7)]
    results = rerank("test query", chunks, top_n=5, use_mocks=True)
    assert len(results) == 5


def test_rerank_scores_descending():
    chunks = [{"text": f"chunk {i}", "chunk_id": f"c{i}"} for i in range(7)]
    results = rerank("test query", chunks, top_n=5, use_mocks=True)
    scores = [s for _, s in results]
    assert scores == sorted(scores, reverse=True)


def test_rerank_does_not_exceed_chunk_count():
    chunks = [{"text": f"chunk {i}", "chunk_id": f"c{i}"} for i in range(3)]
    results = rerank("test query", chunks, top_n=5, use_mocks=True)
    assert len(results) == 3, "top_n must be capped at available chunk count"


def test_pipeline_source_uses_top_n_five():
    """Pipeline.py must pass top_n=5 to rerank — source-level assertion."""
    pipeline_src = (ROOT / "backend" / "inference" / "pipeline.py").read_text()
    assert "top_n=5" in pipeline_src, \
        "pipeline.py must call rerank(... top_n=5 ...) after Bucket C change"


# ===========================================================================
# Bucket D — Chapter-to-question mapping
# ===========================================================================

from routes.bikes import _chapter_to_question, _FALLBACK_STARTERS, get_starters


def test_chapter_to_question_engine_oil():
    assert "oil" in _chapter_to_question("Engine Oil Maintenance").lower()


def test_chapter_to_question_oil_change_variant():
    assert "oil" in _chapter_to_question("Oil Change Procedure").lower()


def test_chapter_to_question_brakes():
    assert "brake" in _chapter_to_question("Brake System Inspection").lower()


def test_chapter_to_question_spark_plug():
    assert "spark plug" in _chapter_to_question("Spark Plug Replacement").lower()


def test_chapter_to_question_valve_clearance():
    assert "valve" in _chapter_to_question("Valve Clearance Adjustment").lower()


def test_chapter_to_question_air_filter():
    assert "air filter" in _chapter_to_question("Air Filter Cleaning").lower()


def test_chapter_to_question_electrical():
    assert "electrical" in _chapter_to_question("Electrical System").lower()


def test_chapter_to_question_chain():
    assert "chain" in _chapter_to_question("Drive Chain Adjustment").lower()


def test_chapter_to_question_unknown_returns_generic():
    q = _chapter_to_question("Torque Specifications Table")
    assert "torque specifications table" in q.lower()


def test_chapter_to_question_unknown_is_readable():
    q = _chapter_to_question("Cooling Fan Assembly")
    assert len(q) > 10 and "?" in q


# ===========================================================================
# Bucket D — Starters endpoint unit tests
# ===========================================================================

def test_starters_fallback_when_index_file_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    result = get_starters("totally_nonexistent_doc")
    assert result["starters"] == _FALLBACK_STARTERS


def test_starters_always_returns_four(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    chapters = [
        {"number": "1", "title": "Engine Oil Maintenance"},
        {"number": "2", "title": "Brake System"},
        {"number": "3", "title": "Spark Plug"},
        {"number": "4", "title": "Valve Clearance"},
        {"number": "5", "title": "Air Filter"},
    ]
    _write_index(tmp_path, "doc_full", chapters=chapters)
    result = get_starters("doc_full")
    assert len(result["starters"]) == 4


def test_starters_questions_are_non_empty_strings(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    chapters = [{"number": str(i), "title": f"Chapter {i}"} for i in range(6)]
    _write_index(tmp_path, "doc_strings", chapters=chapters)
    result = get_starters("doc_strings")
    for q in result["starters"]:
        assert isinstance(q, str) and len(q) > 5


def test_starters_skips_safety_chapter(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    chapters = [
        {"number": "1", "title": "Safety Information"},
        {"number": "2", "title": "Engine Oil Maintenance"},
        {"number": "3", "title": "Brake System"},
        {"number": "4", "title": "Spark Plug"},
        {"number": "5", "title": "Air Filter"},
    ]
    _write_index(tmp_path, "doc_safety_skip", chapters=chapters)
    result = get_starters("doc_safety_skip")
    for q in result["starters"]:
        assert "safety" not in q.lower(), "Safety chapter must not produce a starter question"


def test_starters_skips_index_and_warranty_chapters(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    chapters = [
        {"number": "0", "title": "Index"},
        {"number": "1", "title": "Warranty Information"},
        {"number": "2", "title": "Introduction"},
        {"number": "3", "title": "Engine Oil Maintenance"},
    ]
    _write_index(tmp_path, "doc_skip_generics", chapters=chapters)
    result = get_starters("doc_skip_generics")
    for q in result["starters"]:
        assert "warranty" not in q.lower()
        assert "introduction" not in q.lower()


def test_starters_pads_with_fallbacks_when_few_chapters(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    chapters = [{"number": "1", "title": "Engine Oil Maintenance"}]
    _write_index(tmp_path, "doc_sparse", chapters=chapters)
    result = get_starters("doc_sparse")
    assert len(result["starters"]) == 4, "Must pad to 4 even with sparse chapters"


def test_starters_deduplicates_identical_questions(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    # Two chapters that map to the same question
    chapters = [
        {"number": "1", "title": "Engine Oil Change"},
        {"number": "2", "title": "Engine Oil Level Check"},  # same question as above
        {"number": "3", "title": "Brake System"},
        {"number": "4", "title": "Spark Plug"},
        {"number": "5", "title": "Air Filter"},
    ]
    _write_index(tmp_path, "doc_dedup", chapters=chapters)
    result = get_starters("doc_dedup")
    assert len(result["starters"]) == len(set(result["starters"])), \
        "Starter questions must be unique"


def test_starters_all_chapters_skipped_returns_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    chapters = [
        {"number": "1", "title": "Safety"},
        {"number": "2", "title": "Index"},
        {"number": "3", "title": "Warranty"},
        {"number": "4", "title": "Glossary"},
    ]
    _write_index(tmp_path, "doc_all_skip", chapters=chapters)
    result = get_starters("doc_all_skip")
    assert len(result["starters"]) == 4
    # All 4 must come from fallbacks
    for q in result["starters"]:
        assert q in _FALLBACK_STARTERS


def test_starters_empty_chapters_returns_fallbacks(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    _write_index(tmp_path, "doc_no_chapters", chapters=[])
    result = get_starters("doc_no_chapters")
    assert result["starters"] == _FALLBACK_STARTERS


# ===========================================================================
# Bucket D — Starters via HTTP endpoint
# ===========================================================================

from fastapi.testclient import TestClient
from main import app


def test_starters_http_returns_200(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    chapters = [
        {"number": "2", "title": "Engine Oil"},
        {"number": "3", "title": "Brake System"},
        {"number": "4", "title": "Spark Plug"},
        {"number": "5", "title": "Drive Chain"},
    ]
    _write_index(tmp_path, "doc_http_ok", chapters=chapters)
    with TestClient(app) as c:
        r = c.get("/bikes/doc_http_ok/starters")
    assert r.status_code == 200


def test_starters_http_response_shape(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    chapters = [
        {"number": "2", "title": "Engine Oil"},
        {"number": "3", "title": "Brake System"},
        {"number": "4", "title": "Spark Plug"},
        {"number": "5", "title": "Drive Chain"},
    ]
    _write_index(tmp_path, "doc_http_shape", chapters=chapters)
    with TestClient(app) as c:
        r = c.get("/bikes/doc_http_shape/starters")
    data = r.json()
    assert "starters" in data
    assert isinstance(data["starters"], list)
    assert len(data["starters"]) == 4


def test_starters_http_fallback_for_missing_doc(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    with TestClient(app) as c:
        r = c.get("/bikes/nonexistent_doc_abc123/starters")
    assert r.status_code == 200
    assert r.json()["starters"] == _FALLBACK_STARTERS


def test_library_http_excludes_user_uploaded(tmp_path, monkeypatch):
    monkeypatch.setattr(bikes_module, "_INDEX_DIR", tmp_path)
    _write_index(tmp_path, "http_user", source="user_uploaded")
    _write_index(tmp_path, "http_lib", brand="Honda", model="CB500", source="library")
    with TestClient(app) as c:
        r = c.get("/bikes/library")
    assert r.status_code == 200
    bikes = r.json()["bikes"]
    ids = [b["document_id"] for b in bikes]
    assert "http_lib" in ids
    assert "http_user" not in ids
