#!/usr/bin/env bash
set -euo pipefail

curl --fail --silent --show-error "${DEMO_URL:-http://127.0.0.1:8000}/api/health"
printf '\n'
