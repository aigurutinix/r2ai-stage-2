#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" -m uvicorn demo.app:app \
  --host "${DEMO_HOST:-127.0.0.1}" \
  --port "${DEMO_PORT:-8000}" \
  --workers 1
