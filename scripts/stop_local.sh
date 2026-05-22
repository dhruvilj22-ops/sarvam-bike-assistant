#!/usr/bin/env bash

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PID_FILE="$ROOT/.backend.pid"

if [ -f "$PID_FILE" ]; then
  PID=$(cat "$PID_FILE")
  kill "$PID" 2>/dev/null && echo "Stopped backend (PID $PID)" || echo "Process $PID already stopped"
  rm -f "$PID_FILE"
else
  # Fallback: kill any uvicorn bound to port 8000
  pkill -f "uvicorn main:app" 2>/dev/null && echo "Stopped uvicorn" || echo "No running backend found"
fi
