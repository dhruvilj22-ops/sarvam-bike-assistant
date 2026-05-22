"""
Part 2 tests: backend scaffolding, hello world, and health route.
Run: pytest tests/test_part_2.py -v
Requires server running: bash scripts/start_local.sh
"""

import subprocess
import time
import signal
import os
import pytest
import requests
from pathlib import Path

BASE_URL = "http://localhost:8000"
ROOT = Path(__file__).parent.parent


# ---------------------------------------------------------------------------
# Fixtures — start/stop server for the test session
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def server():
    env = os.environ.copy()
    env["PATH"] = f"{os.path.expanduser('~/.local/bin')}:{env['PATH']}"
    proc = subprocess.Popen(
        ["uv", "run", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=ROOT / "backend",
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # Wait for server to be ready (up to 10s)
    for _ in range(20):
        try:
            requests.get(f"{BASE_URL}/health", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    yield proc
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=5)


# ---------------------------------------------------------------------------
# Route tests
# ---------------------------------------------------------------------------

def test_root_returns_200():
    resp = requests.get(BASE_URL + "/", timeout=5)
    assert resp.status_code == 200


def test_root_returns_html():
    resp = requests.get(BASE_URL + "/", timeout=5)
    assert "text/html" in resp.headers.get("content-type", "")


def test_health_returns_200():
    resp = requests.get(BASE_URL + "/health", timeout=5)
    assert resp.status_code == 200


def test_health_returns_ok():
    resp = requests.get(BASE_URL + "/health", timeout=5)
    assert resp.json() == {"status": "ok"}


# ---------------------------------------------------------------------------
# Script existence checks
# ---------------------------------------------------------------------------

def test_start_local_script_exists_and_executable():
    path = ROOT / "scripts/start_local.sh"
    assert path.exists(), "scripts/start_local.sh missing"
    assert os.access(path, os.X_OK), "scripts/start_local.sh not executable"


def test_stop_local_script_exists_and_executable():
    path = ROOT / "scripts/stop_local.sh"
    assert path.exists(), "scripts/stop_local.sh missing"
    assert os.access(path, os.X_OK), "scripts/stop_local.sh not executable"


def test_start_docker_script_exists():
    assert (ROOT / "scripts/start_docker.sh").exists()


def test_stop_docker_script_exists():
    assert (ROOT / "scripts/stop_docker.sh").exists()


# ---------------------------------------------------------------------------
# pyproject.toml sanity
# ---------------------------------------------------------------------------

def test_pyproject_has_fastapi_and_uvicorn():
    content = (ROOT / "backend/pyproject.toml").read_text()
    assert "fastapi" in content
    assert "uvicorn" in content
