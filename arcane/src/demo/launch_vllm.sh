#!/usr/bin/env bash
set -euo pipefail

MODEL="${VLLM_MODEL:?Set VLLM_MODEL (see vllm.env.example)}"
ARGS=(serve "$MODEL" --host "${VLLM_HOST:-127.0.0.1}" --port "${VLLM_PORT:-8001}")
if [[ -n "${VLLM_SERVED_MODEL_NAME:-}" ]]; then
  ARGS+=(--served-model-name "$VLLM_SERVED_MODEL_NAME")
fi
if [[ -n "${VLLM_MODEL_REVISION:-}" ]]; then
  ARGS+=(--revision "$VLLM_MODEL_REVISION")
fi
# Extra flags (quantisation, memory split, task) pass straight through.
exec vllm "${ARGS[@]}" "$@"
