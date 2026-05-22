#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT/backend"
source "$HOME/.local/bin/env" 2>/dev/null || true

echo "Starting backend at http://localhost:8000 ..."
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
echo $! > "$ROOT/.backend.pid"
echo "PID $! written to .backend.pid"
