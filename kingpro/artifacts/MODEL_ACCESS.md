# Mô hình và checkpoint — KINGPRO V297

## Checkpoint chính duy nhất

```text
model: Qwen/Qwen2.5-Coder-14B-Instruct
revision: aedcc2d42b622764e023cf882b6652e646b95671
license: Apache-2.0
parameters: 14.7B total, 13.1B non-embedding (theo model card)
```

Link tải khóa revision:

```text
https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/tree/aedcc2d42b622764e023cf882b6652e646b95671
```

KINGPRO không cần checkpoint fine-tune riêng để chấm V297. Artifact nộp đã chứa
Pandas programs và CSV evidence; scorer chạy offline, không gọi model/API/mạng.
Trong sản phẩm live, Qwen là fallback sinh code cho câu ngoài registry/compiler,
chỉ mở khi endpoint và revision được attested. Hiện dynamic branch vẫn khóa
fail-closed; demo mặc định dùng registry/compiler đã kiểm toán.

## Tải bằng Hugging Face

```bash
python -m pip install "huggingface_hub>=0.26"
huggingface-cli download Qwen/Qwen2.5-Coder-14B-Instruct \
  --revision aedcc2d42b622764e023cf882b6652e646b95671 \
  --local-dir models/Qwen2.5-Coder-14B-Instruct
```

## Chạy bằng vLLM

```bash
python -m pip install -r requirements-llm.txt
vllm serve models/Qwen2.5-Coder-14B-Instruct \
  --served-model-name Qwen/Qwen2.5-Coder-14B-Instruct \
  --max-model-len 16384
```

Cấu hình `.env`:

```dotenv
KINGPRO_LLM_BASE_URL=http://127.0.0.1:8000/v1
KINGPRO_LLM_MODEL=Qwen/Qwen2.5-Coder-14B-Instruct
KINGPRO_LLM_ATTESTED=false
```

Chỉ chuyển `KINGPRO_LLM_ATTESTED=true` sau khi `/v1/models`, checkpoint revision,
license và worker runtime được đối chiếu. Không ghi API key vào source, tài liệu,
log hoặc bundle.

