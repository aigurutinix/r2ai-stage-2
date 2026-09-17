#!/usr/bin/env bash
set -euo pipefail

curl --fail --silent --show-error --no-buffer \
  -H 'content-type: application/json' \
  -d '{"question":"Tổng tài sản của VJC năm 2018 trên báo cáo riêng là bao nhiêu tỷ đồng?"}' \
  "${DEMO_URL:-http://127.0.0.1:8000}/api/chat"
