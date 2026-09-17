#!/bin/bash
# Copy adapters off a rented box as they appear, for as long as it exists.
#
# A 5.5-hour training run finished, wrote its adapter, and the machine was
# reclaimed before the file could be copied. Nothing about the run was
# recoverable — not the adapter, not the evaluation that was running against it.
# The data and the code survived only because they had always lived elsewhere.
#
# The rule this encodes: the moment an artefact exists that cannot be
# regenerated cheaply, it should exist in two places. Measurement can wait;
# retrieval cannot.
#
# Usage:
#   bash scripts/pull_checkpoints.sh <port> <host> [interval_seconds]
#
# Run it in the background the moment training starts, not when it finishes.
set -uo pipefail

PORT="${1:?usage: pull_checkpoints.sh <port> <host> [interval]}"
HOST="${2:?usage: pull_checkpoints.sh <port> <host> [interval]}"
INTERVAL="${3:-600}"
REMOTE="${REMOTE_DIR:-/workspace/lora_locate}"
LOCAL="${LOCAL_DIR:-artifacts/lora_pulled}"

mkdir -p "$LOCAL"
echo "pulling $REMOTE from $HOST:$PORT into $LOCAL every ${INTERVAL}s"

# Git Bash on Windows ships no rsync, and this has to work from the machine the
# operator actually has. scp moves whole files, so the loop asks the box which
# checkpoint is newest and fetches only that one — a 275 MB adapter re-sent every
# cycle would cost more in bandwidth than the instance costs in compute.
SSH_OPTS=(-p "$PORT" -o StrictHostKeyChecking=no -o ConnectTimeout=30)

while true; do
  newest=$(ssh "${SSH_OPTS[@]}" "root@$HOST" \
    "ls -dt $REMOTE/checkpoint-* 2>/dev/null | head -1" 2>/dev/null | tr -d '\r')
  if [ -z "$newest" ]; then
    # No checkpoint yet is normal early on; unreachable is not. They look the
    # same here, which is why the failure branch below prints rather than exits.
    if ssh "${SSH_OPTS[@]}" "root@$HOST" true 2>/dev/null; then
      echo "$(date +%H:%M:%S) reachable, no checkpoint yet"
    else
      echo "$(date +%H:%M:%S) UNREACHABLE — box gone or restarting"
    fi
  else
    name="${newest##*/}"
    if [ -d "$LOCAL/$name" ]; then
      echo "$(date +%H:%M:%S) $name already local"
    else
      echo "$(date +%H:%M:%S) fetching $name"
      if scp -q "${SSH_OPTS[@]/-p/-P}" -r "root@$HOST:$newest" "$LOCAL/" 2>/dev/null; then
        echo "$(date +%H:%M:%S) got $name"
      else
        echo "$(date +%H:%M:%S) fetch FAILED for $name"
      fi
    fi
    # The final adapter is written beside the checkpoints, not inside one, and it
    # is the artefact that was lost — take it whenever it appears.
    if ssh "${SSH_OPTS[@]}" "root@$HOST" "test -f $REMOTE/adapter_model.safetensors" 2>/dev/null \
       && [ ! -f "$LOCAL/adapter_model.safetensors" ]; then
      echo "$(date +%H:%M:%S) fetching FINAL adapter"
      scp -q "${SSH_OPTS[@]/-p/-P}" "root@$HOST:$REMOTE/adapter_model.safetensors" \
          "root@$HOST:$REMOTE/adapter_config.json" "$LOCAL/" 2>/dev/null \
        && echo "$(date +%H:%M:%S) FINAL adapter secured"
    fi
  fi
  sleep "$INTERVAL"
done
