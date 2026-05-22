"""
Part 5 tests: FastAPI backend routes.
Run: pytest tests/test_part_5.py -v
Uses TestClient (no live server needed). All external API calls use USE_MOCKS=true.
"""
import io
import os
import sys
import time
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from main import app


@pytest.fixture(scope="module", autouse=True)
def _mock_env():
    """Isolate env vars for Part 5 — restore them after all tests in this module complete."""
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


@pytest.fixture(scope="module")
def small_pdf(tmp_path_factory):
    """Minimal 2-page PDF for ingest upload tests."""
    import fitz
    tmp = tmp_path_factory.mktemp("p5_pdf")
    path = tmp / "upload_test.pdf"
    doc = fitz.open()
    font = fitz.Font("helv")

    p = doc.new_page()
    tw = fitz.TextWriter(p.rect)
    tw.append((50, 60), "1 ENGINE SYSTEM", font=font, fontsize=14)
    prose = (
        "The Meteor 350 uses a 349cc SOHC engine. "
        "Engine oil capacity is 1.5 litres. Drain plug torque is 24 Nm."
    )
    tw.append((50, 100), prose, font=font, fontsize=9)
    warning = "WARNING: Hot oil causes burns. Wait 30 minutes before draining."
    tw.append((50, 150), warning, font=font, fontsize=9)
    procedure = (
        "1. Place bike on centre stand.\n"
        "2. Remove drain plug with 10mm socket.\n"
        "3. Drain oil completely and replace plug to 24 Nm.\n"
        "4. Fill with 1.5 litres of 15W-50 oil."
    )
    tw.append((50, 200), procedure, font=font, fontsize=9)
    tw.write_text(p)

    doc.save(str(path))
    doc.close()
    return path


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Session routes
# ---------------------------------------------------------------------------

def test_create_session(client):
    r = client.post("/session")
    assert r.status_code == 201
    assert "session_id" in r.json()
    assert len(r.json()["session_id"]) == 36  # UUID format


def test_session_isolation(client):
    sid_a = client.post("/session").json()["session_id"]
    sid_b = client.post("/session").json()["session_id"]
    assert sid_a != sid_b

    client.post(f"/session/{sid_a}/threads", json={"title": "Issue A"})
    threads_a = client.get(f"/session/{sid_a}/threads").json()["threads"]
    threads_b = client.get(f"/session/{sid_b}/threads").json()["threads"]
    assert len(threads_a) == 1
    assert len(threads_b) == 0


def test_get_threads_empty(client):
    sid = client.post("/session").json()["session_id"]
    r = client.get(f"/session/{sid}/threads")
    assert r.status_code == 200
    assert r.json()["threads"] == []


def test_create_thread(client):
    sid = client.post("/session").json()["session_id"]
    r = client.post(f"/session/{sid}/threads", json={"title": "Engine noise"})
    assert r.status_code == 201
    body = r.json()
    assert "thread_id" in body
    assert body["session_id"] == sid
    assert body["title"] == "Engine noise"
    assert body["status"] == "open"


def test_get_threads_after_create(client):
    sid = client.post("/session").json()["session_id"]
    client.post(f"/session/{sid}/threads", json={"title": "Oil leak"})
    client.post(f"/session/{sid}/threads", json={"title": "Smoke"})
    r = client.get(f"/session/{sid}/threads")
    assert r.status_code == 200
    assert len(r.json()["threads"]) == 2


def test_get_history_empty(client):
    sid = client.post("/session").json()["session_id"]
    r = client.get(f"/session/{sid}/history")
    assert r.status_code == 200
    assert r.json()["history"] == []


def test_unknown_session_returns_404(client):
    r = client.get("/session/nonexistent-id/threads")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] is True
    assert "message" in body


def test_unknown_session_history_returns_404(client):
    r = client.get("/session/nonexistent-id/history")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# Query route
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def indexed_doc_for_query(small_pdf):
    """Ingest the small PDF in-process so queries can find content."""
    import fitz
    from ingestion.parser import parse_pdf
    from ingestion.chunker import chunk_blocks
    from ingestion.embedder import embed_chunks
    from ingestion.indexer import build_indexes, reset_qdrant_client
    from ingestion.document_index import generate_document_index

    doc_meta = {
        "bike_brand": "Royal Enfield",
        "bike_model": "Meteor 350",
        "bike_year": "2022",
        "manual_type": "owner_manual",
        "manual_source": "library",
    }
    reset_qdrant_client()
    parsed = parse_pdf(str(small_pdf), doc_meta, use_mocks=True)
    document_id = parsed["document_id"]
    chunks = chunk_blocks(parsed["blocks"], doc_meta, document_id)
    embedded = embed_chunks(list(chunks), use_mocks=True)
    build_indexes(embedded, document_id)
    generate_document_index(document_id, doc_meta, embedded)
    return document_id


def test_query_returns_200(client, indexed_doc_for_query):
    sid = client.post("/session").json()["session_id"]
    tid = client.post(f"/session/{sid}/threads", json={}).json()["thread_id"]
    r = client.post("/query", json={
        "text": "engine oil capacity",
        "session_id": sid,
        "document_id": indexed_doc_for_query,
        "thread_id": tid,
    })
    assert r.status_code == 200


def test_query_all_output_fields_present(client, indexed_doc_for_query):
    sid = client.post("/session").json()["session_id"]
    tid = client.post(f"/session/{sid}/threads", json={}).json()["thread_id"]
    r = client.post("/query", json={
        "text": "how to change engine oil",
        "session_id": sid,
        "document_id": indexed_doc_for_query,
        "thread_id": tid,
    })
    body = r.json()
    for field in ("answer_text", "spoken_summary", "citations", "severity_label",
                  "confidence", "suggested_followups", "session_id", "thread_id", "document_id"):
        assert field in body, f"Missing field: {field}"


def test_query_in_scope_has_citation(client, indexed_doc_for_query):
    sid = client.post("/session").json()["session_id"]
    tid = client.post(f"/session/{sid}/threads", json={}).json()["thread_id"]
    r = client.post("/query", json={
        "text": "drain plug torque specification",
        "session_id": sid,
        "document_id": indexed_doc_for_query,
        "thread_id": tid,
    })
    body = r.json()
    assert isinstance(body["citations"], list)
    assert len(body["citations"]) > 0


def test_query_out_of_scope_refusal(client, indexed_doc_for_query):
    sid = client.post("/session").json()["session_id"]
    tid = client.post(f"/session/{sid}/threads", json={}).json()["thread_id"]
    r = client.post("/query", json={
        "text": "what is the price of this motorcycle",
        "session_id": sid,
        "document_id": indexed_doc_for_query,
        "thread_id": tid,
    })
    body = r.json()
    assert body["citations"] == []
    assert "couldn't find" in body["answer_text"].lower() or "service center" in body["answer_text"].lower()


def test_query_empty_text_returns_422(client, indexed_doc_for_query):
    r = client.post("/query", json={
        "text": "   ",
        "session_id": "s1",
        "document_id": indexed_doc_for_query,
        "thread_id": "t1",
    })
    assert r.status_code == 422


def test_query_missing_fields_returns_422(client):
    r = client.post("/query", json={"text": "engine oil"})
    assert r.status_code == 422
    body = r.json()
    assert body["error"] is True
    assert "message" in body


# ---------------------------------------------------------------------------
# Ingest routes
# ---------------------------------------------------------------------------

def test_ingest_returns_job_id(client, small_pdf):
    with open(small_pdf, "rb") as f:
        r = client.post("/ingest", data={
            "brand": "Royal Enfield",
            "model": "Meteor 350",
            "year": "2022",
            "manual_type": "owner_manual",
        }, files={"file": ("test.pdf", f, "application/pdf")})
    assert r.status_code == 202
    assert "job_id" in r.json()


def test_ingest_status_submitted(client, small_pdf):
    with open(small_pdf, "rb") as f:
        resp = client.post("/ingest", data={
            "brand": "Royal Enfield",
            "model": "Meteor 350",
            "year": "2022",
        }, files={"file": ("test.pdf", f, "application/pdf")})
    job_id = resp.json()["job_id"]
    r = client.get(f"/ingest/status/{job_id}")
    assert r.status_code == 200
    body = r.json()
    assert "status" in body
    assert "progress_pct" in body
    assert "message" in body
    assert body["status"] in ("submitted", "processing", "complete", "failed")


def test_ingest_job_completes(client, small_pdf):
    with open(small_pdf, "rb") as f:
        resp = client.post("/ingest", data={
            "brand": "Royal Enfield",
            "model": "Meteor 350",
            "year": "2022",
        }, files={"file": ("test.pdf", f, "application/pdf")})
    job_id = resp.json()["job_id"]

    deadline = time.time() + 30
    status = "submitted"
    while time.time() < deadline:
        body = client.get(f"/ingest/status/{job_id}").json()
        status = body["status"]
        if status in ("complete", "failed"):
            break
        time.sleep(0.3)

    assert status in ("complete", "failed"), f"Job did not finish, last status: {status}"


def test_ingest_non_pdf_returns_400(client):
    r = client.post("/ingest", data={
        "brand": "Test", "model": "Bike", "year": "2022",
    }, files={"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")})
    assert r.status_code == 400
    assert r.json()["error"] is True


def test_ingest_status_unknown_job(client):
    r = client.get("/ingest/status/nonexistent-job-id")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] is True


# ---------------------------------------------------------------------------
# Bikes library route
# ---------------------------------------------------------------------------

def test_bikes_library_returns_200(client):
    r = client.get("/bikes/library")
    assert r.status_code == 200


def test_bikes_library_response_shape(client):
    r = client.get("/bikes/library")
    body = r.json()
    assert "bikes" in body
    assert isinstance(body["bikes"], list)


def test_bikes_library_items_have_required_fields(client):
    r = client.get("/bikes/library")
    bikes = r.json()["bikes"]
    for bike in bikes:
        for field in ("document_id", "bike_brand", "bike_model", "bike_year", "manual_type"):
            assert field in bike, f"Library bike missing field: {field}"
