"""FILE THAM CHIEU — KHONG dung trong bai nop.
Ban goc pipeline/modal_app.py (da xoa khoi repo chinh vi khong con dung).
Giu lai lam khuon mau cho service sau. Doc README.md canh file nay truoc khi dung.
Nho: doi API_KEY, doi ten app, va `modal app stop` sau khi xong.
"""
"""Modal app: serve LLM ≤14B (vLLM, OpenAI-compatible) trên GPU A10G — KHÔNG cần thẻ như H100 endpoint.
Sinh pandas cho câu hard-conditional (ViFinQA). Qwen2.5-7B-Instruct: non-thinking, ≤14B, vừa A10G 24GB.

Deploy:  python -m modal deploy modal_app.py
URL:     python -m modal app list   (hoặc in ra khi deploy) → dạng https://<workspace>--r2ai-llm-serve.modal.run
Điền pipeline/.env:
  LLM_BASE_URL=<URL>/v1
  LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
  LLM_API_KEY=<tự đặt một chuỗi ngẫu nhiên>
  LLM_NO_THINK=0
Đổi model: sửa MODEL (vd Qwen/Qwen2.5-14B-Instruct-AWQ cho mạnh hơn — thêm --quantization awq)."""
import modal

MODEL = "QuantTrio/Qwen3.5-9B-AWQ"         # Qwen3.5-9B (≤14B, 2/2026, thinking) AWQ INT4 (~6GB) — vừa A10G, KHÔNG cần payment GPU
QUANT = "awq"                              # AWQ INT4 chạy trên A10G (Ampere); FP8 cần Ada/Hopper (L4+)
import os as _os
API_KEY = _os.environ.get("LLM_API_KEY", "")   # ĐỌC TỪ MÔI TRƯỜNG — không hard-code (repo public)
assert API_KEY, "Đặt LLM_API_KEY trước khi deploy (cùng giá trị với LLM_API_KEY trong .env)"
PORT = 8000
GPU = "A10G"                               # commodity (không payment); AWQ ~6GB thừa chỗ trên 24GB
TP = 1                                      # 1 GPU → không tensor-parallel

vol = modal.Volume.from_name("r2ai-llm-weights", create_if_missing=True)   # cache weights → cold-start nhanh
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm", "huggingface_hub[hf_transfer]")   # vLLM mới nhất → hỗ trợ kiến trúc qwen3_5
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/root/.cache/huggingface",
        "VLLM_USE_FLASHINFER_SAMPLER": "0",       # sampler PyTorch native (không JIT nvcc)
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",   # giảm phân mảnh VRAM
        # KHÔNG ép VLLM_ATTENTION_BACKEND: Qwen3.5 hybrid (Gated Delta + linear attention) cần vLLM tự chọn backend
    })
)
app = modal.App("r2ai-llm", image=image)


@app.function(
    gpu=GPU,
    volumes={"/root/.cache/huggingface": vol},
    scaledown_window=300,             # giữ nóng 5 phút sau request cuối
    timeout=3600,
)
@modal.concurrent(max_inputs=16)      # vLLM continuous batching (thinking sinh nhiều token → hạ concurrency)
@modal.web_server(port=PORT, startup_timeout=1200)   # đủ thời gian tải model lần đầu (9B bf16)
def serve():
    import subprocess
    # vLLM serve = OpenAI-compatible API tại /v1/chat/completions; --reasoning-parser tách <think> khỏi content
    cmd = (
        f"vllm serve {MODEL} --host 0.0.0.0 --port {PORT} --api-key {API_KEY} "
        f"--max-model-len 16384 --gpu-memory-utilization 0.92 --reasoning-parser qwen3 "
        f"--tensor-parallel-size {TP}"
    )
    if QUANT:
        cmd += f" --quantization {QUANT}"
    subprocess.Popen(cmd, shell=True)
