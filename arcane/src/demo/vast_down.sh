#!/usr/bin/env bash
# Stop everything demo/vast_up.sh started, here and on the box.
set -euo pipefail

cd "$(dirname "$0")/.."
STATE_DIR="${STATE_DIR:-demo/.demo-state}"
SOCKET="$STATE_DIR/ssh.sock"

if [[ -f "$STATE_DIR/demo.pid" ]]; then
  kill "$(cat "$STATE_DIR/demo.pid")" 2>/dev/null || true
  rm -f "$STATE_DIR/demo.pid"
fi
fuser -k -n tcp "${DEMO_PORT:-8000}" 2>/dev/null || true
echo "local UI stopped"

if [[ -S "$SOCKET" && -f "$STATE_DIR/target" ]]; then
  TARGET=$(head -1 "$STATE_DIR/target")
  # Free the GPU so the rented box is not billing for idle weights.
  ssh -S "$SOCKET" "$TARGET" 'pkill -f "vllm serve" || true; pkill -f "VLLM::EngineCore" || true; echo "model servers stopped"' || true
  ssh -S "$SOCKET" -O exit "$TARGET" 2>/dev/null || true
  echo "connection closed"
fi

echo "Destroy the Vast instance from the web console to stop billing."
