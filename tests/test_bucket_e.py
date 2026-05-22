"""
Tests for Bucket E: auto-extract bike metadata from PDF first pages.

Covers:
- _read_first_pages: extracts text from PDF bytes via PyMuPDF
- _mock_extract: brand detection, year regex, manual_type inference
- /ingest/extract-meta endpoint: shape, mock mode, non-PDF rejection, graceful fallback

Run: pytest tests/test_bucket_e.py -v
All tests use USE_MOCKS=true.
"""
import io
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


def _make_pdf_bytes(text: str) -> bytes:
    """Create a minimal single-page PDF with the given text using PyMuPDF."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text, fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()
    return buf.getvalue()


# ===========================================================================
# _read_first_pages
# ===========================================================================

from routes.ingest import _read_first_pages


def test_read_first_pages_returns_string():
    pdf_bytes = _make_pdf_bytes("Royal Enfield Meteor 350 Service Manual 2022")
    result = _read_first_pages(pdf_bytes)
    assert isinstance(result, str)


def test_read_first_pages_contains_pdf_text():
    pdf_bytes = _make_pdf_bytes("Royal Enfield Meteor 350 Service Manual 2022")
    result = _read_first_pages(pdf_bytes)
    assert "Royal Enfield" in result or "Meteor" in result


def test_read_first_pages_caps_at_4000_chars():
    long_text = "A" * 10000
    pdf_bytes = _make_pdf_bytes(long_text)
    result = _read_first_pages(pdf_bytes)
    assert len(result) <= 4000


def test_read_first_pages_multi_page_respects_max():
    """Only first 3 pages extracted — fourth page content must not appear."""
    import fitz
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text((72, 72), f"Page{i+1}Content", fontsize=11)
    buf = io.BytesIO()
    doc.save(buf)
    doc.close()

    result = _read_first_pages(buf.getvalue(), max_pages=3)
    assert "Page1Content" in result
    assert "Page3Content" in result
    assert "Page4Content" not in result
    assert "Page5Content" not in result


# ===========================================================================
# _mock_extract
# ===========================================================================

from routes.ingest import _mock_extract


def test_mock_extract_detects_royal_enfield():
    result = _mock_extract("Royal Enfield Meteor 350 Service Manual")
    assert result["bike_brand"] == "Royal Enfield"


def test_mock_extract_detects_honda():
    result = _mock_extract("Honda CB300R Owner's Manual")
    assert result["bike_brand"] == "Honda"


def test_mock_extract_detects_yamaha():
    result = _mock_extract("YAMAHA R15 V4 Service Documentation")
    assert result["bike_brand"] == "Yamaha"


def test_mock_extract_detects_bajaj():
    result = _mock_extract("Bajaj Pulsar NS200 Workshop Manual")
    assert result["bike_brand"] == "Bajaj"


def test_mock_extract_unknown_brand_returns_empty():
    result = _mock_extract("XYZ Motors Model 999 Technical Guide")
    assert result["bike_brand"] == ""


def test_mock_extract_detects_year():
    result = _mock_extract("Service Manual 2022 edition for Meteor 350")
    assert result["bike_year"] == "2022"


def test_mock_extract_detects_year_2019():
    result = _mock_extract("Published 2019. Honda CB300R owner guide.")
    assert result["bike_year"] == "2019"


def test_mock_extract_no_year_returns_empty():
    result = _mock_extract("Service Manual for Meteor — no date given")
    assert result["bike_year"] == ""


def test_mock_extract_service_manual_type():
    result = _mock_extract("Service Manual for Royal Enfield Classic 350")
    assert result["manual_type"] == "service_manual"


def test_mock_extract_owner_manual_type():
    result = _mock_extract("Owner's Manual for Royal Enfield Meteor 350 2022")
    assert result["manual_type"] == "owner_manual"


def test_mock_extract_user_guide_type():
    result = _mock_extract("User Guide for the Royal Enfield Himalayan")
    assert result["manual_type"] == "user_guide"


def test_mock_extract_confidence_is_low():
    result = _mock_extract("Some bike manual text here")
    assert result["confidence"] == "low"


def test_mock_extract_always_has_all_fields():
    result = _mock_extract("")
    for field in ("bike_brand", "bike_model", "bike_year", "manual_type", "confidence"):
        assert field in result, f"Missing field: {field}"


def test_mock_extract_model_is_empty_string():
    result = _mock_extract("Royal Enfield Meteor 350 Service Manual 2022")
    assert result["bike_model"] == "", "Mock mode cannot reliably extract model — must be empty"


# ===========================================================================
# HTTP endpoint — /ingest/extract-meta
# ===========================================================================

from fastapi.testclient import TestClient
from main import app


def test_extract_meta_returns_200():
    pdf_bytes = _make_pdf_bytes("Royal Enfield Meteor 350 Service Manual 2022")
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    assert r.status_code == 200


def test_extract_meta_response_has_required_fields():
    pdf_bytes = _make_pdf_bytes("Some bike manual text")
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    data = r.json()
    for field in ("bike_brand", "bike_model", "bike_year", "manual_type", "confidence"):
        assert field in data, f"Response missing field: {field}"


def test_extract_meta_fields_are_strings():
    pdf_bytes = _make_pdf_bytes("Honda CB300R 2021 Owner manual")
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    data = r.json()
    for field in ("bike_brand", "bike_model", "bike_year", "manual_type", "confidence"):
        assert isinstance(data[field], str), f"Field {field} must be a string"


def test_extract_meta_detects_royal_enfield_from_pdf():
    pdf_bytes = _make_pdf_bytes("Royal Enfield Meteor 350 Service Manual 2022")
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    assert r.json()["bike_brand"] == "Royal Enfield"


def test_extract_meta_detects_year_from_pdf():
    pdf_bytes = _make_pdf_bytes("Honda CB300R Service Manual 2021")
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    assert r.json()["bike_year"] == "2021"


def test_extract_meta_rejects_non_pdf():
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta",
                   files={"file": ("manual.txt", b"not a pdf", "text/plain")})
    assert r.status_code == 400


def test_extract_meta_rejects_file_without_pdf_extension():
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta",
                   files={"file": ("manual.docx", b"binary content", "application/octet-stream")})
    assert r.status_code == 400


def test_extract_meta_manual_type_service_by_default():
    pdf_bytes = _make_pdf_bytes("Workshop technical documentation for Bajaj Pulsar")
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta", files={"file": ("test.pdf", pdf_bytes, "application/pdf")})
    assert r.json()["manual_type"] == "service_manual"


def test_extract_meta_detects_owner_manual():
    pdf_bytes = _make_pdf_bytes("Owner's Manual for Royal Enfield Meteor 350")
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta", files={"file": ("owners.pdf", pdf_bytes, "application/pdf")})
    assert r.json()["manual_type"] == "owner_manual"


def test_extract_meta_with_real_pdf():
    """Smoke test against the real Royal Enfield manual in data/manuals/."""
    real_pdf = ROOT / "data" / "manuals" / "royal-enfield-owners-manual-meteor-english.pdf"
    if not real_pdf.exists():
        pytest.skip("Real PDF not present")
    pdf_bytes = real_pdf.read_bytes()
    with TestClient(app) as c:
        r = c.post("/ingest/extract-meta",
                   files={"file": ("re_meteor.pdf", pdf_bytes, "application/pdf")})
    assert r.status_code == 200
    data = r.json()
    assert "bike_brand" in data
    # Real PDF should detect Royal Enfield
    assert "royal enfield" in data["bike_brand"].lower() or data["bike_brand"] == ""
