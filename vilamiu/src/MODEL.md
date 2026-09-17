# Tài liệu mô hình sử dụng — đội vilamiu

## 1. Mô hình

| | |
|---|---|
| Tên | **Qwen3-14B** |
| Định danh checkpoint | `Qwen/Qwen3-14B` (Hugging Face) |
| Tham số | 14,8 tỷ |
| Ngày phát hành | tháng 4/2025 — **trước 01/06/2026** theo quy định cuộc thi |
| Giấy phép | **Apache-2.0**, open-weight, trọng số công khai |
| Độ chính xác khi chạy | **bfloat16** |
| Chế độ suy luận | `enable_thinking: true` |
| Fine-tune / adapter | **KHÔNG** — dùng checkpoint gốc |

Đội **không sử dụng bất kỳ mô hình đóng nào** (GPT, Gemini, Claude…) ở bất kỳ khâu nào:
không sinh dữ liệu, không huấn luyện, không suy luận, không chấm chọn.

`src/vifin/llm/client.py` giữ danh sách trắng `ALLOWED_MODELS`; mọi lời gọi đi qua đó, nên
một mô hình không hợp lệ không thể lọt vào bài nộp do sơ suất cấu hình.

### 1.1 Mô hình phụ trợ

Không có. Mọi khâu còn lại — truy hồi bảng, phân giải mã công ty, đọc số, giải đơn vị, các
cổng lọc, bộ giải theo Mã số Thông tư 200 — đều **thuần luật và code**, không dùng mô hình.

Đội có **thử** và **loại bỏ** `Qwen/Qwen3-Reranker-8B` sau khi đo: nó cho độ hiện diện bảng
vàng 89,9% so với 90,4% của xếp hạng trùng-từ ở cùng ngân sách, tức không hơn. Bài nộp cuối
không dùng reranker.

## 2. Phiên bản checkpoint

| | |
|---|---|
| Repo | `Qwen/Qwen3-14B` |
| Revision | `main` tại thời điểm chạy (**tháng 8/2026**) |
| Tệp trọng số | `model-0000{1..8}-of-00008.safetensors`, tổng ~29,5 GB |
| Tokenizer | `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt` |
| Cấu hình | `config.json` (`Qwen3ForCausalLM`, `max_position_embeddings: 40960`) |

Để cố định tuyệt đối, ghim revision khi tải:

```bash
huggingface-cli download Qwen/Qwen3-14B --revision main \
  --local-dir ./models/Qwen3-14B
# ghi lai SHA da tai de tai lap ve sau:
python -c "from huggingface_hub import HfApi; \
print(HfApi().model_info('Qwen/Qwen3-14B').sha)"
```

## 3. Hướng dẫn tải checkpoint

```bash
pip install huggingface_hub

# Cach 1 — Hugging Face truc tiep
huggingface-cli download Qwen/Qwen3-14B --local-dir ./models/Qwen3-14B

# Cach 2 — neu mang toi huggingface.co cham (doi da gap 4 KB/s tren mot may thue),
# dung mirror; toc do do duoc 13,5 MB/s tren cung may:
HF_ENDPOINT=https://hf-mirror.com \
  huggingface-cli download Qwen/Qwen3-14B --local-dir ./models/Qwen3-14B
```

Bản sao checkpoint đội đã dùng được chia sẻ tại:

> **Checkpoint gốc:** <https://huggingface.co/Qwen/Qwen3-14B>
> **Bản sao của đội:** https://drive.google.com/drive/folders/1K_guQUhY6CHHN572iXcRFmyfpDiJmgm9
>
> Checkpoint là mô hình open-weight công khai; đường dẫn Hugging Face ở trên là nguồn
> chính thức và cố định. Bản sao trên Drive dành cho trường hợp không truy cập được
> Hugging Face.

## 4. Hướng dẫn sử dụng checkpoint

### 4.1 Tự host bằng vLLM (tái lập hoàn toàn)

Yêu cầu: **1 GPU ≥ 48 GB** (đã chạy trên RTX A6000 48 GB và RTX 6000 Ada 48 GB).

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B --served-model-name qwen3 \
  --max-model-len 28000 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --host 0.0.0.0 --port 18000
```

Rồi trỏ pipeline vào đó:

```bash
python scripts/fresh/run_cot.py --prompts artifacts/fresh/prompts.jsonl \
  --out artifacts/fresh/passA.jsonl --temperature 0.0 --max-tokens 6000 \
  --endpoint http://127.0.0.1:18000/v1/chat/completions --model qwen3
```

### 4.2 Qua API OpenAI-compatible

Bất kỳ nhà cung cấp nào phục vụ **đúng checkpoint open-weight** `Qwen/Qwen3-14B`. Đặt khoá
vào `.env` theo mẫu `.env.example`, rồi bỏ `--endpoint`/`--model` để dùng mặc định.

## 5. Ba điều phải nhớ khi chạy lại

Cả ba đều đã đo được, và bỏ qua thì kết quả sai lệch chứ không báo lỗi:

1. **`--max-tokens` không được nhỏ hơn 5000** khi bật suy luận. Mô hình dùng hết ngân sách
   trong khối `<think>` rồi trả về nội dung rỗng. Đo được: 3.000 token → **57% phản hồi
   rỗng**; 5.600 token → **0%**.
2. **Không lượng tử hoá.** AWQ 4-bit làm giảm độ chính xác trên tác vụ này; bài nộp dùng
   bfloat16.
3. **`gpu_memory_utilization` chỉ áp khi GPU trống.** Một tiến trình `VLLM::EngineCore` mồ
   côi từ lần khởi động hỏng có thể giữ hơn 40 GB và làm lần sau báo "hết bộ nhớ" gây hiểu
   nhầm. `pkill -f api_server` **không** bắt được nó — phải `kill` theo PID lấy từ
   `nvidia-smi --query-compute-apps=pid --format=csv,noheader`.

## 6. Prompt

Đội dùng **nguyên văn** prompt hệ thống của BTC tại
`vifinqa-official/prompts/answering/program_system.txt`, chỉ nối thêm một khối `<no_constants>`
cấm gán hằng số — đúng theo yêu cầu "không được gán cứng kết quả" của Ban Tổ chức. Toàn bộ
nội dung nối thêm nằm trong `scripts/fresh/build_prompts_program.py` (hằng `NO_CONSTANTS`).
