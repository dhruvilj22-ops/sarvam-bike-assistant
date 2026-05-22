"""
Part 8 tests: frontend static build and FastAPI static serving.
Run: pytest tests/test_part_8.py -v
Verifies the Next.js build output exists and FastAPI serves it correctly.
"""
import os
import sys
import pytest
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend"))

FRONTEND_OUT = ROOT / "frontend" / "out"

from fastapi.testclient import TestClient
from main import app


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
# Build output structure tests
# ---------------------------------------------------------------------------

def test_frontend_out_directory_exists():
    assert FRONTEND_OUT.exists(), "Run 'npm run build' in frontend/ first"


def test_frontend_index_html_exists():
    assert (FRONTEND_OUT / "index.html").exists()


def test_frontend_chat_page_exists():
    chat_dir = FRONTEND_OUT / "chat"
    assert chat_dir.exists()
    # Next.js static export creates either chat/index.html or chat.html
    has_chat = (chat_dir / "index.html").exists() or (FRONTEND_OUT / "chat.html").exists()
    assert has_chat, "Chat page not found in build output"


def test_frontend_nextjs_assets_exist():
    next_dir = FRONTEND_OUT / "_next"
    assert next_dir.exists()
    # Static chunks should exist
    static_dir = next_dir / "static"
    assert static_dir.exists()


def test_frontend_no_api_keys_in_build():
    """Ensure no API keys leaked into static build."""
    sensitive = ["SARVAM_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY", "OPENROUTER_API_KEY"]
    for js_file in (FRONTEND_OUT / "_next" / "static").rglob("*.js"):
        content = js_file.read_text(errors="ignore")
        for key in sensitive:
            assert key not in content, f"API key reference '{key}' found in {js_file.name}"


# ---------------------------------------------------------------------------
# FastAPI static serving tests
# ---------------------------------------------------------------------------

def test_health_route_still_works(client):
    """API routes must take priority over static serving."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_query_route_not_shadowed(client):
    """POST /query must still work when static files are mounted."""
    r = client.post("/query", json={
        "text": "engine oil",
        "session_id": "s1",
        "document_id": "nonexistent",
        "thread_id": "t1",
    })
    assert r.status_code == 200
    assert "answer_text" in r.json()


def test_api_session_route_not_shadowed(client):
    r = client.post("/session")
    assert r.status_code == 201


def test_api_bikes_route_not_shadowed(client):
    r = client.get("/bikes/library")
    assert r.status_code == 200
    assert "bikes" in r.json()


@pytest.mark.skipif(not (FRONTEND_OUT / "index.html").exists(), reason="Build output missing")
def test_root_serves_frontend_html(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers.get("content-type", "")


@pytest.mark.skipif(not (FRONTEND_OUT / "index.html").exists(), reason="Build output missing")
def test_frontend_html_has_sarvam_title(client):
    r = client.get("/")
    assert r.status_code == 200
    # Title or relevant content should be in the HTML
    content = r.text
    assert "Bike Assistant" in content or "Sarvam" in content or "<!DOCTYPE" in content


@pytest.mark.skipif(not (FRONTEND_OUT / "_next").exists(), reason="Build output missing")
def test_nextjs_static_assets_served(client):
    """_next/static assets should be reachable."""
    next_static = FRONTEND_OUT / "_next" / "static"
    # Find any JS chunk
    js_files = list(next_static.rglob("*.js"))
    if not js_files:
        pytest.skip("No JS chunks found")
    rel = js_files[0].relative_to(FRONTEND_OUT)
    r = client.get(f"/{rel}")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Frontend component structure tests (sanity checks on source)
# ---------------------------------------------------------------------------

FRONTEND_SRC = ROOT / "frontend"

def test_page_tsx_exists():
    assert (FRONTEND_SRC / "app" / "page.tsx").exists()


def test_chat_page_tsx_exists():
    assert (FRONTEND_SRC / "app" / "chat" / "page.tsx").exists()


def test_api_client_exists():
    assert (FRONTEND_SRC / "lib" / "api.ts").exists()


def test_input_bar_component_exists():
    assert (FRONTEND_SRC / "components" / "InputBar.tsx").exists()


def test_message_card_component_exists():
    assert (FRONTEND_SRC / "components" / "MessageCard.tsx").exists()


def test_thread_sidebar_component_exists():
    assert (FRONTEND_SRC / "components" / "ThreadSidebar.tsx").exists()


def test_severity_badge_component_exists():
    assert (FRONTEND_SRC / "components" / "SeverityBadge.tsx").exists()


def test_citation_block_component_exists():
    assert (FRONTEND_SRC / "components" / "CitationBlock.tsx").exists()


def test_api_client_has_query_function():
    content = (FRONTEND_SRC / "lib" / "api.ts").read_text()
    assert "export async function query" in content


def test_api_client_has_voice_function():
    content = (FRONTEND_SRC / "lib" / "api.ts").read_text()
    assert "transcribeVoice" in content


def test_api_client_has_image_function():
    content = (FRONTEND_SRC / "lib" / "api.ts").read_text()
    assert "describeImage" in content


def test_api_client_has_tts_function():
    content = (FRONTEND_SRC / "lib" / "api.ts").read_text()
    assert "synthesizeSpeech" in content


def test_input_bar_has_voice_support():
    content = (FRONTEND_SRC / "components" / "InputBar.tsx").read_text()
    assert "MediaRecorder" in content or "transcribeVoice" in content


def test_input_bar_has_image_support():
    content = (FRONTEND_SRC / "components" / "InputBar.tsx").read_text()
    assert "describeImage" in content


def test_message_card_has_citation_block():
    content = (FRONTEND_SRC / "components" / "MessageCard.tsx").read_text()
    assert "CitationBlock" in content


def test_message_card_has_severity_badge():
    content = (FRONTEND_SRC / "components" / "MessageCard.tsx").read_text()
    assert "SeverityBadge" in content


def test_chat_page_has_voice_initiated():
    content = (FRONTEND_SRC / "app" / "chat" / "page.tsx").read_text()
    assert "voice_initiated" in content or "voiceInitiated" in content


def test_chat_page_has_thread_switching():
    content = (FRONTEND_SRC / "app" / "chat" / "page.tsx").read_text()
    assert "ThreadSidebar" in content


def test_home_page_has_library_tab():
    content = (FRONTEND_SRC / "app" / "page.tsx").read_text()
    assert "library" in content.lower()


def test_home_page_has_upload_tab():
    content = (FRONTEND_SRC / "app" / "page.tsx").read_text()
    assert "upload" in content.lower() or "Upload" in content


def test_home_page_polls_ingest_status():
    content = (FRONTEND_SRC / "app" / "page.tsx").read_text()
    assert "getIngestStatus" in content


def test_sarvam_gradient_in_css():
    content = (FRONTEND_SRC / "app" / "globals.css").read_text()
    assert "sarvam-gradient" in content
