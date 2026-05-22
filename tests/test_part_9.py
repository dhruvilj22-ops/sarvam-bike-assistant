"""
Part 9 tests: multi-bike library, namespace isolation, save_to_library, upload precedence.
Run: pytest tests/test_part_9.py -v
All tests use USE_MOCKS=true — no real API calls.
"""
import json
import os
import sys
import pickle
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from main import app
from ingestion.indexer import build_indexes, reset_qdrant_client, vector_search, bm25_search
from ingestion.embedder import embed_chunks
from ingestion.document_index import generate_document_index
from ingestion.chunker import chunk_blocks
from ingestion.parser import parse_pdf

INDEX_DIR = ROOT / "data" / "indexes"


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
# Helpers — build two synthetic manuals to test isolation
# ---------------------------------------------------------------------------

def _make_meta(brand, model, source="library"):
    return {
        "bike_brand": brand,
        "bike_model": model,
        "bike_year": "2022",
        "manual_type": "service_manual",
        "manual_source": source,
    }


def _synthetic_chunks(document_id: str, brand: str, model: str, count: int = 5):
    """Return minimal chunk dicts with unique text per bike."""
    return [
        {
            "chunk_id": f"{document_id}_chunk_{i}",
            "document_id": document_id,
            "text": f"{brand} {model} service procedure step {i}: check the engine oil level.",
            "content_type": "prose",
            "chapter_number": "1",
            "chapter_title": "Engine",
            "section_number": "1.1",
            "section_title": "Oil System",
            "page_number": i + 1,
            "vector": [0.0] * 1536,
        }
        for i in range(count)
    ]


@pytest.fixture(scope="module")
def two_bikes(tmp_path_factory):
    """Index two synthetic bikes under separate document_ids and return their ids."""
    reset_qdrant_client()

    meta_re = _make_meta("Royal Enfield", "Meteor 350", source="library")
    meta_tvs = _make_meta("TVS", "Apache RTR 160", source="library")

    doc_id_re = "test_re_meteor350_9"
    doc_id_tvs = "test_tvs_apache160_9"

    chunks_re = _synthetic_chunks(doc_id_re, "Royal Enfield", "Meteor 350")
    chunks_tvs = _synthetic_chunks(doc_id_tvs, "TVS", "Apache RTR 160")

    # Embed with mocks (returns zero vectors of correct dim)
    embedded_re = embed_chunks(chunks_re, use_mocks=True)
    embedded_tvs = embed_chunks(chunks_tvs, use_mocks=True)

    build_indexes(embedded_re, doc_id_re)
    build_indexes(embedded_tvs, doc_id_tvs)

    generate_document_index(doc_id_re, meta_re, embedded_re)
    generate_document_index(doc_id_tvs, meta_tvs, embedded_tvs)

    yield doc_id_re, doc_id_tvs

    # Cleanup index files after module
    for did in (doc_id_re, doc_id_tvs):
        for ext in ("_index.json", "_bm25.pkl"):
            p = INDEX_DIR / f"{did}{ext}"
            if p.exists():
                p.unlink()


# ---------------------------------------------------------------------------
# Document index schema tests
# ---------------------------------------------------------------------------

def test_document_index_has_manual_source(two_bikes):
    doc_id_re, _ = two_bikes
    idx_path = INDEX_DIR / f"{doc_id_re}_index.json"
    assert idx_path.exists()
    data = json.loads(idx_path.read_text())
    assert "manual_source" in data
    assert data["manual_source"] == "library"


def test_document_index_user_upload_source(tmp_path):
    """manual_source=user_uploaded is written when save_to_library is False."""
    meta = _make_meta("Bajaj", "Pulsar NS200", source="user_uploaded")
    doc_id = "test_bajaj_pulsar_ns200_9"
    chunks = _synthetic_chunks(doc_id, "Bajaj", "Pulsar NS200", count=3)
    embedded = embed_chunks(chunks, use_mocks=True)
    build_indexes(embedded, doc_id)
    idx = generate_document_index(doc_id, meta, embedded)

    assert idx["manual_source"] == "user_uploaded"

    # Cleanup
    for ext in ("_index.json", "_bm25.pkl"):
        p = INDEX_DIR / f"{doc_id}{ext}"
        if p.exists():
            p.unlink()


# ---------------------------------------------------------------------------
# /bikes/library filter tests
# ---------------------------------------------------------------------------

def test_library_endpoint_returns_bikes(client, two_bikes):
    r = client.get("/bikes/library")
    assert r.status_code == 200
    assert "bikes" in r.json()


def test_library_only_shows_library_source(client, two_bikes):
    """User-uploaded manuals must NOT appear in the library."""
    meta = _make_meta("Honda", "Shine", source="user_uploaded")
    doc_id = "test_honda_shine_9"
    chunks = _synthetic_chunks(doc_id, "Honda", "Shine", count=2)
    embedded = embed_chunks(chunks, use_mocks=True)
    generate_document_index(doc_id, meta, embedded)

    r = client.get("/bikes/library")
    bikes = r.json()["bikes"]
    doc_ids = [b["document_id"] for b in bikes]
    assert doc_id not in doc_ids, "User-uploaded manual must not appear in library"

    # Cleanup
    idx_path = INDEX_DIR / f"{doc_id}_index.json"
    if idx_path.exists():
        idx_path.unlink()


def test_library_shows_library_source_bikes(client, two_bikes):
    """Both library bikes appear in /bikes/library."""
    doc_id_re, doc_id_tvs = two_bikes
    r = client.get("/bikes/library")
    doc_ids = [b["document_id"] for b in r.json()["bikes"]]
    assert doc_id_re in doc_ids
    assert doc_id_tvs in doc_ids


def test_library_entry_has_all_fields(client, two_bikes):
    r = client.get("/bikes/library")
    for bike in r.json()["bikes"]:
        for field in ("document_id", "bike_brand", "bike_model", "bike_year",
                      "manual_type", "manual_source", "total_chunks", "ingestion_timestamp"):
            assert field in bike, f"Library entry missing field: {field}"


# ---------------------------------------------------------------------------
# Namespace isolation tests — core correctness guarantee
# ---------------------------------------------------------------------------

def test_vector_search_returns_only_re_chunks(two_bikes):
    """Vector search for Royal Enfield doc_id must not return TVS chunks."""
    doc_id_re, doc_id_tvs = two_bikes
    query_vector = [0.0] * 1536
    results = vector_search(query_vector, doc_id_re, top_k=10)
    for payload, _ in results:
        assert payload.get("document_id") == doc_id_re, (
            f"Cross-namespace leak: RE search returned chunk from {payload.get('document_id')}"
        )


def test_vector_search_returns_only_tvs_chunks(two_bikes):
    """Vector search for TVS doc_id must not return Royal Enfield chunks."""
    doc_id_re, doc_id_tvs = two_bikes
    query_vector = [0.0] * 1536
    results = vector_search(query_vector, doc_id_tvs, top_k=10)
    for payload, _ in results:
        assert payload.get("document_id") == doc_id_tvs, (
            f"Cross-namespace leak: TVS search returned chunk from {payload.get('document_id')}"
        )


def test_bm25_isolation_re(two_bikes):
    """BM25 for Royal Enfield must not return TVS chunks."""
    doc_id_re, _ = two_bikes
    results = bm25_search(["Meteor", "engine"], doc_id_re, top_k=5)
    for payload, _ in results:
        assert payload.get("document_id") == doc_id_re


def test_bm25_isolation_tvs(two_bikes):
    """BM25 for TVS must not return Royal Enfield chunks."""
    _, doc_id_tvs = two_bikes
    results = bm25_search(["Apache", "engine"], doc_id_tvs, top_k=5)
    for payload, _ in results:
        assert payload.get("document_id") == doc_id_tvs


def test_re_chunks_contain_re_text(two_bikes):
    doc_id_re, _ = two_bikes
    results = bm25_search(["Royal", "Enfield"], doc_id_re, top_k=5)
    assert len(results) > 0
    for payload, _ in results:
        assert "Royal Enfield" in payload.get("text", "")


def test_tvs_chunks_contain_tvs_text(two_bikes):
    _, doc_id_tvs = two_bikes
    results = bm25_search(["TVS", "Apache"], doc_id_tvs, top_k=5)
    assert len(results) > 0
    for payload, _ in results:
        assert "TVS" in payload.get("text", "")


# ---------------------------------------------------------------------------
# Upload precedence — user upload supersedes library entry
# ---------------------------------------------------------------------------

def test_upload_precedence_overwrites_library_entry(two_bikes):
    """
    Re-indexing a document_id replaces existing chunks.
    User upload with same doc_id takes over retrieval.
    """
    doc_id_re, _ = two_bikes

    # Simulate user-uploaded version with different text
    user_chunks = [
        {
            "chunk_id": f"{doc_id_re}_user_chunk_0",
            "document_id": doc_id_re,
            "text": "User uploaded: custom oil spec for my specific RE Meteor.",
            "content_type": "prose",
            "chapter_number": "1",
            "chapter_title": "Engine",
            "section_number": "1.1",
            "section_title": "Oil",
            "page_number": 1,
            "vector": [0.0] * 1536,
        }
    ]
    user_embedded = embed_chunks(user_chunks, use_mocks=True)
    build_indexes(user_embedded, doc_id_re)

    results = bm25_search(["custom", "oil", "spec"], doc_id_re, top_k=5)
    texts = [p.get("text", "") for p, _ in results]
    assert any("User uploaded" in t for t in texts), "User upload did not replace library chunks"

    # Restore original chunks for remaining tests
    original_chunks = _synthetic_chunks(doc_id_re, "Royal Enfield", "Meteor 350")
    original_embedded = embed_chunks(original_chunks, use_mocks=True)
    build_indexes(original_embedded, doc_id_re)


# ---------------------------------------------------------------------------
# /ingest save_to_library field tests (HTTP level)
# ---------------------------------------------------------------------------

def test_ingest_save_to_library_true_accepted(client):
    """save_to_library=true is accepted without error."""
    import io
    # Minimal 1-byte PDF header — parser will mock-parse it
    fake_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    r = client.post(
        "/ingest",
        data={
            "brand": "Test Brand",
            "model": "Test Model",
            "year": "2023",
            "manual_type": "service_manual",
            "save_to_library": "true",
        },
        files={"file": ("test.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert r.status_code == 202
    assert "job_id" in r.json()


def test_ingest_save_to_library_false_accepted(client):
    """save_to_library=false (default) is accepted."""
    import io
    fake_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    r = client.post(
        "/ingest",
        data={
            "brand": "Private Brand",
            "model": "Private Model",
            "year": "2023",
            "manual_type": "owner_manual",
            "save_to_library": "false",
        },
        files={"file": ("private.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert r.status_code == 202


def test_ingest_without_save_to_library_defaults_to_false(client):
    """Omitting save_to_library defaults to false (user_uploaded)."""
    import io
    fake_pdf = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n"
    r = client.post(
        "/ingest",
        data={"brand": "Brand X", "model": "Model Y", "year": "2024", "manual_type": "service_manual"},
        files={"file": ("x.pdf", io.BytesIO(fake_pdf), "application/pdf")},
    )
    assert r.status_code == 202


# ---------------------------------------------------------------------------
# Frontend source tests
# ---------------------------------------------------------------------------

FRONTEND_SRC = ROOT / "frontend"


def test_frontend_upload_form_has_save_to_library():
    content = (FRONTEND_SRC / "app" / "page.tsx").read_text()
    assert "saveToLibrary" in content or "save_to_library" in content


def test_frontend_upload_sends_save_to_library():
    content = (FRONTEND_SRC / "lib" / "api.ts").read_text()
    assert "save_to_library" in content


def test_frontend_checkbox_has_description():
    content = (FRONTEND_SRC / "app" / "page.tsx").read_text()
    assert "Save to library" in content


def test_index_library_script_exists():
    assert (ROOT / "scripts" / "index_library.py").exists()


def test_index_library_script_has_brand_model_args():
    content = (ROOT / "scripts" / "index_library.py").read_text()
    assert "--brand" in content
    assert "--model" in content


def test_index_library_script_sets_library_source():
    content = (ROOT / "scripts" / "index_library.py").read_text()
    assert "manual_source" in content and "library" in content
