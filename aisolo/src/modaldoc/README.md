# Modal — cách dùng nhanh (không cần gắn thẻ)

Ghi lại từ dự án ViFinQA 08/2026. Khuôn mẫu đầy đủ: [`modal_app.reference.py`](modal_app.reference.py).

---

## 1. Chọn GPU không bị khoá

| Nhóm | GPU | Cần payment method? |
|---|---|---|
| Commodity | **A10G** (24 GB), T4, L4 | ❌ không |
| Cao cấp | A100, H100, H200 | ✅ có |

**Cách làm:** chọn `A10G` rồi ép model vừa 24 GB bằng lượng tử hoá **AWQ INT4** (model ~9B chỉ chiếm ~6 GB).

> AWQ hay FP8? Trên A10G (Ampere) **phải AWQ** — FP8 cần Ada/Hopper trở lên. Chọn sai chỉ báo lỗi
> lúc khởi động, không phải lúc deploy.

Chi phí: A10G ~**$1.1/giờ**.

---

## 2. Ba tham số phải có

```python
@app.function(
    gpu="A10G",
    volumes={"/root/.cache/huggingface": vol},   # cache weights + torch.compile → cold start sau nhanh hơn
    scaledown_window=300,                        # ngủ sau 5 phút rảnh → KHÔNG đốt tiền
)
@modal.web_server(port=8000, startup_timeout=1200)   # cold start có thể vài phút
```

Lệnh serve (API tương thích OpenAI):

```python
vllm serve <MODEL> --host 0.0.0.0 --port 8000 --api-key <KEY> \
  --max-model-len 16384 --gpu-memory-utilization 0.92 --quantization awq
```

---

## 3. Quy trình chạy

```bash
python -m modal deploy modal_app.py        # → in ra URL
python -m modal app logs <app-name>        # chờ vLLM mở cổng (xem log, đừng đoán)
# warm-up 1 request nhỏ trước khi chạy batch
python -m modal app stop <app-name>        # DỪNG khi xong — dễ quên nhất
```

Gọi endpoint: `POST <URL>/v1/chat/completions`, header `Authorization: Bearer <KEY>`.

---

## 4. Bốn lỗi hay gặp

| Triệu chứng | Nguyên nhân | Xử lý |
|---|---|---|
| Endpoint trả **rỗng**, không báo lỗi | vLLM chưa mở cổng (còn tải model) | Xem `app logs`; viết vòng thử lại có backoff |
| Trả rỗng, `finish_reason: "length"` | Model "thinking" ăn hết `max_tokens` | Thêm `"chat_template_kwargs": {"enable_thinking": false}` |
| Lỗi khởi động, sai attention backend | Ép `VLLM_ATTENTION_BACKEND` với model hybrid | Đừng ép — để vLLM tự chọn |
| "unknown architecture" | `vllm` bản cũ | Cài bản mới nhất |

---

## 5. Một lời khuyên

**Thử trên model local nhỏ trước.** Ở dự án này, bốn lỗi trong chính script đo (không tắt thinking,
phân tầng mẫu hỏng, rổ ứng viên sai, parser nhặt trúng số năm) đều được phát hiện bằng model 4B chạy
local. Nếu chạy thẳng lên GPU thuê thì đã trả tiền để thu về số liệu rác.
