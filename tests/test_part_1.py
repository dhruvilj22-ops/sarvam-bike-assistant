"""
Part 1 tests: environment setup and API connectivity checks.
Run: pytest tests/test_part_1.py -v
"""

import os
import json
import pytest
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent


def _key(name: str) -> str:
    val = os.getenv(name, "").strip()
    return val if val else ""


# ---------------------------------------------------------------------------
# Directory structure
# ---------------------------------------------------------------------------

def test_directory_structure():
    required = ["backend", "frontend", "tests", "tests/fixtures", "scripts", "docs", "data/indexes"]
    for d in required:
        assert (ROOT / d).is_dir(), f"Missing directory: {d}"


# ---------------------------------------------------------------------------
# .env and .env.example
# ---------------------------------------------------------------------------

def test_env_example_exists():
    assert (ROOT / ".env.example").exists(), ".env.example missing"


def test_env_file_exists():
    assert (ROOT / ".env").exists(), ".env missing — copy .env.example and fill in keys"


def test_use_mocks_key_present():
    assert os.getenv("USE_MOCKS") is not None, "USE_MOCKS not set in .env"


def test_all_expected_keys_declared_in_env_example():
    expected = [
        "OPENROUTER_API_KEY", "SARVAM_API_KEY", "OPENAI_API_KEY",
        "COHERE_API_KEY", "QDRANT_URL", "QDRANT_API_KEY",
        "LLAMAPARSE_API_KEY", "USE_MOCKS",
    ]
    content = (ROOT / ".env.example").read_text()
    for key in expected:
        assert key in content, f"{key} missing from .env.example"


# ---------------------------------------------------------------------------
# Fixture files
# ---------------------------------------------------------------------------

def test_sample_table_fixture_exists_and_valid():
    path = ROOT / "tests/fixtures/sample_table.json"
    assert path.exists(), "sample_table.json missing"
    data = json.loads(path.read_text())
    assert "table_title" in data, "sample_table.json missing table_title"
    assert "rows" in data, "sample_table.json missing rows"
    assert len(data["rows"]) > 0, "sample_table.json has empty rows"
    # Each row must be a dict (key-value), not a flat string — core contract
    for row in data["rows"]:
        assert isinstance(row, dict), "Table rows must be key-value dicts, not flat strings"


def test_sample_response_fixture_exists_and_valid():
    path = ROOT / "tests/fixtures/sample_response.json"
    assert path.exists(), "sample_response.json missing"
    data = json.loads(path.read_text())
    required_fields = [
        "answer_text", "spoken_summary", "citations",
        "severity_label", "confidence", "suggested_followups",
    ]
    for field in required_fields:
        assert field in data, f"sample_response.json missing field: {field}"
    assert len(data["citations"]) > 0, "citations must be non-empty"
    for c in data["citations"]:
        assert "section_number" in c and "section_title" in c and "page_number" in c
    assert isinstance(data["suggested_followups"], list)


# ---------------------------------------------------------------------------
# Mock mode
# ---------------------------------------------------------------------------

def test_mock_mode_flag_toggles():
    original = os.environ.get("USE_MOCKS")
    os.environ["USE_MOCKS"] = "true"
    assert os.getenv("USE_MOCKS") == "true"
    os.environ["USE_MOCKS"] = "false"
    assert os.getenv("USE_MOCKS") == "false"
    if original is not None:
        os.environ["USE_MOCKS"] = original
    else:
        del os.environ["USE_MOCKS"]


def test_mock_fixtures_cover_all_mocked_apis():
    """All fixture files that mock mode depends on must exist."""
    fixtures = [
        "tests/fixtures/sample_table.json",
        "tests/fixtures/sample_response.json",
    ]
    for f in fixtures:
        path = ROOT / f
        assert path.exists(), f"Mock fixture missing: {f}"
        data = json.loads(path.read_text())
        assert data, f"Mock fixture is empty: {f}"


# ---------------------------------------------------------------------------
# Live connectivity checks (skipped if key not set)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _key("OPENROUTER_API_KEY"), reason="OPENROUTER_API_KEY not configured")
def test_openrouter_connectivity():
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {_key('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        },
        timeout=15,
    )
    assert resp.status_code == 200, f"OpenRouter {resp.status_code}: {resp.text[:200]}"


@pytest.mark.skipif(not _key("OPENAI_API_KEY"), reason="OPENAI_API_KEY not configured")
def test_openai_connectivity():
    resp = requests.get(
        "https://api.openai.com/v1/models",
        headers={"Authorization": f"Bearer {_key('OPENAI_API_KEY')}"},
        timeout=15,
    )
    assert resp.status_code == 200, f"OpenAI {resp.status_code}: {resp.text[:200]}"


@pytest.mark.skipif(not _key("SARVAM_API_KEY"), reason="SARVAM_API_KEY not configured")
def test_sarvam_connectivity():
    # Transliterate is the lightest Sarvam endpoint — no audio file needed
    resp = requests.post(
        "https://api.sarvam.ai/transliterate",
        headers={
            "api-subscription-key": _key("SARVAM_API_KEY"),
            "Content-Type": "application/json",
        },
        json={
            "input": "namaste",
            "source_language_code": "hi-IN",
            "target_language_code": "en-IN",
        },
        timeout=15,
    )
    assert resp.status_code == 200, f"Sarvam {resp.status_code}: {resp.text[:200]}"


@pytest.mark.skipif(not _key("COHERE_API_KEY"), reason="COHERE_API_KEY not configured")
def test_cohere_connectivity():
    resp = requests.post(
        "https://api.cohere.com/v2/rerank",
        headers={
            "Authorization": f"Bearer {_key('COHERE_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": "rerank-v3.5",
            "query": "engine oil level check",
            "documents": [
                "Check oil level using the dipstick before every ride.",
                "Clean the air filter every 3000 km.",
            ],
            "top_n": 2,
        },
        timeout=15,
    )
    assert resp.status_code == 200, f"Cohere {resp.status_code}: {resp.text[:200]}"


@pytest.mark.skipif(not _key("QDRANT_URL"), reason="QDRANT_URL not configured")
def test_qdrant_connectivity():
    url = _key("QDRANT_URL").rstrip("/")
    headers = {}
    if _key("QDRANT_API_KEY"):
        headers["api-key"] = _key("QDRANT_API_KEY")
    resp = requests.get(f"{url}/collections", headers=headers, timeout=15)
    assert resp.status_code == 200, f"Qdrant {resp.status_code}: {resp.text[:200]}"


@pytest.mark.skipif(not _key("LLAMAPARSE_API_KEY"), reason="LLAMAPARSE_API_KEY not configured")
def test_llamaparse_connectivity():
    resp = requests.get(
        "https://api.cloud.llamaindex.ai/api/parsing/job",
        headers={"Authorization": f"Bearer {_key('LLAMAPARSE_API_KEY')}"},
        timeout=15,
    )
    # 200 (empty list) or 404 both confirm auth passed; 401/403 means bad key
    assert resp.status_code in (200, 404), f"LlamaParse {resp.status_code}: {resp.text[:200]}"
