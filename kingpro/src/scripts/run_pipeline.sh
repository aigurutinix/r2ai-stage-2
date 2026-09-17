#!/bin/bash
# =============================================================================
# CHUỖI HÀNH ĐỘNG CHUẨN R2AI Stage 2 — "winning config" (đã đạt EXECUTION 0.1621)
# Chạy 1 lệnh: chờ-vLLM -> BUILD (đúng cờ thắng) -> grader_check -> copy zip ra để nộp.
#
# CỜ THẮNG (ĐỪNG BỎ SÓT — bỏ decompose/maso/rel-k là retrieval sập, tụt điểm):
#   --decompose      : câu đa-thực-thể truy hồi nhiều báo cáo (docs rộng); nhánh này TỰ nhồi 6-10 bảng/câu
#   --engine sc      : self-consistency voting
#   --n-vote N       : số bản vote (5 = rẻ lấy phần lớn lợi; 10 = tối ưu, đắt gấp đôi)
#   --maso           : Mã số TT200 verify/cứu đáp án (pandas thật)
#   --rel-k 5        : nới relevant_tables top-5/báo cáo (DOCS/TABLES recall)
#   --workers 24     : song song (vLLM batch)
#
# DÙNG:
#   bash scripts/run_pipeline.sh <FPT_ENDPOINT> [out_name] [n_vote]
#   vd: bash scripts/run_pipeline.sh https://kingpro-xxx-8000.serverless.fptcloud.com sub_new 5
#
# CHUỖI ĐẦY ĐỦ (gồm bước tay trên trình duyệt):
#   B1. FPT console -> Create Container -> template vllm-openai-v0.10.1 -> H100
#       args: --model Qwen/Qwen2.5-Coder-14B-Instruct --max-model-len 16384 --api-key kingpro2026
#       port HTTP 8000. Lấy endpoint = https://<ten-container>-8000.serverless.fptcloud.com
#   B2. bash scripts/run_pipeline.sh <endpoint>   (script này: chờ+build+grader+copy)
#   B3. Nộp <out>.zip trên leaderboard (tab "Kiểm thử công khai") -> đọc EXECUTION
#   B4. Điểm > best -> bấm add-to-leaderboard; điểm thấp -> BỎ (đừng add)
#   B5. XOÁ container FPT (menu ... -> Delete -> gõ "delete") để DỪNG ĐỐT TIỀN
# =============================================================================
set +e
ENDPOINT="${1:?Thieu FPT endpoint, vd https://<ten>-8000.serverless.fptcloud.com}"
OUT="${2:-sub_pipeline}"
NVOTE="${3:-5}"
WORKERS="${4:-24}"   # tang (32-40) de rut wall-clock khi budget FPT eo hep
API_KEY="${FPT_API_KEY:-kingpro2026}"
MODEL="${FPT_MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}"
WORKSPACE="${WORKSPACE_COPY:-/c/Users/vinh/Downloads/VAIC-DDAY}"   # nơi copy zip để tool upload đọc được

cd "$(dirname "$0")/.." || exit 2
export KINGPRO_LLM_API_KEY="$API_KEY"
export BASE_CODER="$ENDPOINT/v1"
export MODEL_CODER="$MODEL"

echo "=== [1/4] $(date +%H:%M:%S) CHO vLLM san sang: $BASE_CODER ==="
R=0
for i in $(seq 1 60); do
  c=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $API_KEY" "$BASE_CODER/models" --max-time 20)
  echo "$(date +%H:%M:%S) poll $i: HTTP $c"
  [ "$c" = "200" ] && { R=1; echo "READY"; break; }
  sleep 30
done
[ "$R" != "1" ] && { echo "TIMEOUT: vLLM chua san sang sau 30 phut"; exit 1; }

echo "=== [2/4] $(date +%H:%M:%S) BUILD winning-config (decompose+sc+maso+rel-k5, n-vote=$NVOTE) ==="
python scripts/build_full_submission.py --n 1012 --decompose --engine sc --n-vote "$NVOTE" \
  --maso --rel-k 5 --workers "$WORKERS" --out "$OUT"

echo "=== [3/4] $(date +%H:%M:%S) GRADER_CHECK (pandas 1.1.5 — chi kiem code chay duoc, KHONG phai diem gold) ==="
.venv-grader/Scripts/python.exe scripts/grader_check.py "$OUT" 2>/dev/null | tail -12

echo "=== [4/4] $(date +%H:%M:%S) copy zip ra workspace de upload ==="
cp "$OUT.zip" "$WORKSPACE/$OUT.zip" && echo "SAN SANG NOP: $WORKSPACE/$OUT.zip"
echo "=== XONG $(date +%H:%M:%S). NHO: nop leaderboard + XOA container FPT (dung dot tien) ==="
