#!/bin/bash
# Cài môi trường fine-tune trên GPU container/VM (Linux, CUDA). Chạy 1 lần sau khi vào terminal.
# 3 venv TÁCH BIỆT: TRAIN (unsloth) | SERVE (vllm) | GRADER (pandas 1.1.5) — tránh xung đột torch/pandas.
set -e

echo "=== [1] TRAIN env: Unsloth (để Unsloth TỰ ghim trl/transformers/peft — KHÔNG -U đè) ==="
pip install -U pip
pip install -U "unsloth" "unsloth_zoo"
# CỐ Ý không '-U trl transformers': đè lên pin của unsloth là vỡ API SFTTrainer. Chỉ thêm phần thiếu:
pip install "datasets" "accelerate" "bitsandbytes"
python -c "import unsloth, trl, peft, datasets, transformers; \
print('TRAIN OK | trl', trl.__version__, '| transformers', transformers.__version__)"

echo "=== [2] SERVE env RIÊNG: vLLM (tách torch khỏi unsloth) ==="
python -m venv .venv-serve
./.venv-serve/bin/pip install -U pip
./.venv-serve/bin/pip install vllm
./.venv-serve/bin/python -c "import vllm; print('SERVE vLLM', vllm.__version__)"

echo "=== [3] GRADER env RIÊNG: pandas 1.1.5 (giống hệt máy chấm; chỉ cần nếu chấm TRÊN container) ==="
python -m venv .venv-grader
./.venv-grader/bin/pip install -U pip
./.venv-grader/bin/pip install "pandas==1.1.5" "numpy==1.19.5"
./.venv-grader/bin/python -c "import pandas; print('GRADER pandas', pandas.__version__)"

echo ""
echo "SETUP XONG."
echo "  TRAIN:  python scripts/train_lora.py --data build/sft_chatml.jsonl --out outputs/kingpro-lora --epochs 2 --r 32"
echo "  SERVE:  ./.venv-serve/bin/python -m vllm.entrypoints.openai.api_server \\"
echo "            --model outputs/kingpro-lora-merged --served-model-name kingpro-lora-merged \\"
echo "            --max-model-len 16384 --api-key kingpro2026 --port 8000"
echo "  (eval + nộp chạy Ở MÁY LOCAL, trỏ vào URL public cổng 8000 của container). Xem RUNBOOK_FINETUNE.md"
