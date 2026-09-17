# ViFinQA — Truy hồi bảng & sinh truy vấn Pandas trên BCTC tiếng Việt

Đội **vilamiu**. Tài liệu này mô tả đầy đủ cách cài đặt, cấu hình và **chạy lại sản phẩm từ
đầu** để tái hiện bài nộp `submissions/legal2.zip`.

---

## 1. Tóm tắt phương pháp

Hệ thống trả lời câu hỏi tài chính tiếng Việt bằng cách **sinh chương trình pandas đọc ô
thật từ bảng CSV**, không tra cứu chỉ tiêu đã chuẩn hoá sẵn. Bốn tầng:

| tầng | việc | công cụ |
|---|---|---|
| **1. Phân giải câu hỏi** | mã công ty, năm, phạm vi (hợp nhất / công ty mẹ), đơn vị tiền | luật + bảng bí danh (`data/ticker_aliases.csv`) |
| **2. Truy hồi bảng** | lọc theo (công ty, năm, phạm vi) rồi xếp hạng bảng trong đúng tài liệu bằng trùng-từ có trọng số ngữ cảnh | `scripts/fresh/build_prompts_program.py` |
| **3. Sinh chương trình** | Qwen3-14B (bật suy luận) viết `pandas_query` đọc ô; hai lượt nhiệt độ khác nhau | `scripts/fresh/run_cot.py` |
| **4. Kiểm và chọn** | thi hành trong sandbox của BTC, giao chéo giữa các cơ chế, năm cổng loại bỏ | `scripts/fresh/run_programs.py` + cổng |

Bốn cơ chế độc lập cùng chạy và **giao nhau** để chọn đáp án: chương trình pandas, truy vấn
SQL trên cùng bảng, bộ giải bằng code theo Mã số Thông tư 200, và bỏ phiếu tự nhất quán
(self-consistency) nhiều mẫu.

Chi tiết thiết kế và toàn bộ kết quả đo: [`BAO-CAO-PHUONG-PHAP.md`](BAO-CAO-PHUONG-PHAP.md).

---

## 2. Yêu cầu môi trường

| | |
|---|---|
| Python | **3.12 hoặc 3.13** (đã chạy trên 3.13.14) |
| Hệ điều hành | Windows 11 / Linux — không phụ thuộc nền tảng |
| Thư viện Python | `pandas`, `numpy` (xem `requirements.txt`) |
| GPU | **không bắt buộc** nếu gọi model qua API; cần **1 GPU ≥ 48 GB** nếu tự host |
| Dung lượng đĩa | ~40 GB cho kho BCTC, ~30 GB nếu tải checkpoint model |

```bash
python -m venv .venv
# Linux/macOS:  source .venv/bin/activate
# Windows:      .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Dữ liệu

### 3.1 Nguồn

Toàn bộ dữ liệu lấy từ **kho chính thức của Ban Tổ chức** trên Hugging Face
(dataset ViFinQA). Không dùng dữ liệu ngoài nào khác.

| | |
|---|---|
| Công ty | 100 mã niêm yết |
| Báo cáo | 1.965 tài liệu (hợp nhất và công ty mẹ, 10 năm) |
| Bảng CSV | 146.246 bảng, do BTC trích sẵn |
| Câu hỏi | 1.012 (`data/questions/questions.jsonl`) |

### 3.2 Cấu trúc thư mục

```
data/
├── official_corpus/
│   └── <MÃ_CK>/<NĂM>/<tên_báo_cáo>/
│       ├── <tên_báo_cáo>_extracted.txt          # văn bản, có neo [table_N](...)
│       └── <tên_báo_cáo>_extracted_tables/
│           ├── table_0.csv
│           └── ...
├── questions/questions.jsonl                     # {"id": int, "question": str}
├── code_stock.csv                                # mã CK → tên công ty (BTC cung cấp)
└── ticker_aliases.csv                            # bí danh do đội bổ sung (mục 3.3)
```

`<tên_báo_cáo>_extracted.txt` chứa các neo dạng `[table_N](…/table_N.csv)`. **Số dòng của
neo trong file `.txt`** chính là "vị trí bảng trong báo cáo" mà trường `relevant_tables`
yêu cầu — đây là điểm đã xác nhận với BTC.

### 3.3 Dữ liệu do đội bổ sung

Một tệp duy nhất: **`data/ticker_aliases.csv`** (129 dòng) — ánh xạ tên thường gọi sang mã
chứng khoán (`Hoà Phát → HPG`, `Vinamilk → VNM`, `Eximbank → EIB`…). Được soạn thủ công từ
chính tên công ty trong đề bài; **không lấy từ nguồn ngoài nào**. Lý do: sổ đăng ký của BTC
ghi `CTCP Tập đoàn Hòa Phát` trong khi đề hỏi `Hoà Phát`, khiến 35 câu không nhận ra được
công ty và ~20 câu nhận thiếu công ty trong nhóm so sánh.

### 3.4 Đường dẫn tải

Dữ liệu và checkpoint được chia sẻ tại:

> **https://drive.google.com/drive/folders/1K_guQUhY6CHHN572iXcRFmyfpDiJmgm9**
>
> Thư mục `AI Guru 2026 Stage 2 - vilamiu` trên Google Drive.

Gói chia sẻ gồm: `data/` (kho BCTC + câu hỏi + bảng bí danh), `artifacts/` (kết quả trung
gian), `submissions/legal2.zip` (bài nộp cuối).

---

## 4. Mô hình

| | |
|---|---|
| Mô hình | **Qwen/Qwen3-14B** |
| Tham số | 14,8 tỷ |
| Ngày phát hành | tháng 4/2025 (**trước 01/06/2026** theo quy định) |
| Giấy phép | Apache-2.0, open-weight |
| Checkpoint | `Qwen/Qwen3-14B` trên Hugging Face, revision `main` |
| Độ chính xác | **bfloat16** (không lượng tử hoá — AWQ 4-bit làm giảm điểm, xem báo cáo) |
| Chế độ | `enable_thinking: true` |

Không dùng bất kỳ mô hình đóng nào (GPT, Gemini…) ở bất kỳ khâu nào — sinh dữ liệu, huấn
luyện hay suy luận. Không fine-tune: bài nộp dùng **checkpoint gốc**, không có adapter.

### 4.1 Cách lấy checkpoint

```bash
# Cách 1 — Hugging Face trực tiếp
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-14B --local-dir ./models/Qwen3-14B

# Cách 2 — nếu mạng tới huggingface.co chậm (đã gặp: 4 KB/s), dùng mirror
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download Qwen/Qwen3-14B \
  --local-dir ./models/Qwen3-14B
```

### 4.2 Hai cách chạy suy luận

**(a) Tự host bằng vLLM** — tái lập hoàn toàn, cần 1 GPU ≥ 48 GB:

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-14B --served-model-name qwen3 \
  --max-model-len 28000 --gpu-memory-utilization 0.90 \
  --reasoning-parser qwen3 --host 0.0.0.0 --port 18000
```

**(b) Qua API OpenAI-compatible** phục vụ cùng checkpoint open-weight đó. Đặt khoá trong
`.env`:

```
OPEN_ROUTER_KEY=<khoá>
```

`src/vifin/llm/client.py` giữ danh sách trắng `ALLOWED_MODELS` để một model không hợp lệ
không thể lọt vào bài nộp do sơ suất.

---

## 5. Chạy lại từ đầu

Bốn lệnh, theo thứ tự. Mỗi bước ghi kết quả ra `artifacts/fresh/` và **có thể chạy lại từ
giữa chừng** (bỏ qua id đã có).

```bash
# 1. Dựng prompt: truy hồi bảng cho từng câu hỏi          (~15 phút, CPU)
python scripts/fresh/build_prompts_program.py \
    --out artifacts/fresh/prompts.jsonl

# 2. Sinh chương trình, hai lượt nhiệt độ khác nhau        (~40 phút mỗi lượt)
python scripts/fresh/run_cot.py --prompts artifacts/fresh/prompts.jsonl \
    --out artifacts/fresh/passA.jsonl --temperature 0.0  --max-tokens 6000
python scripts/fresh/run_cot.py --prompts artifacts/fresh/prompts.jsonl \
    --out artifacts/fresh/passB.jsonl --temperature 0.35 --max-tokens 6000

# 3. Thi hành trong sandbox của BTC                        (~10 phút, CPU)
python scripts/fresh/run_programs.py --replies artifacts/fresh/passA.jsonl \
    --prompts artifacts/fresh/prompts.jsonl --out artifacts/fresh/progA.jsonl
python scripts/fresh/run_programs.py --replies artifacts/fresh/passB.jsonl \
    --prompts artifacts/fresh/prompts.jsonl --out artifacts/fresh/progB.jsonl

# 4. Giao chéo, lọc qua các cổng, đóng gói bài nộp         (~20 phút, CPU)
python scripts/fresh/build_submission.py \
    --first artifacts/fresh/progA.jsonl \
    --second artifacts/fresh/progB.jsonl \
    --out submissions/submission.zip
```

Nếu tự host, thêm `--endpoint http://127.0.0.1:18000/v1/chat/completions --model qwen3` vào
bước 2.

**Lưu ý quan trọng khi tự host:** `--max-tokens` **không được nhỏ hơn 5000** khi bật suy
luận. Ngân sách nhỏ hơn khiến mô hình dùng hết token trong khối `<think>` và trả về nội dung
rỗng — đã đo: 3.000 token cho 57% mẫu rỗng, 5.600 token cho 0%.

---

## 6. Kiểm tra bài nộp trước khi nộp

```bash
python scripts/fresh/validate_submission.py submissions/legal2.zip
```

Kiểm bốn điều kiện của BTC:

1. đủ **1.012** câu, không thiếu, không sai định dạng;
2. mọi `csv_path` bắt đầu bằng `data/` và **file tương ứng có trong ZIP**;
3. mọi `pandas_query` **đọc dữ liệu từ DataFrame**, không gán hằng số
   (`result = 2.06`, `result = 1.03 * 2`, `result = df["answer"].iloc[0]` đều bị bắt);
4. chạy lại từng `pandas_query` trong sandbox và đối chiếu với trường `answer`.

---

## 7. Cấu trúc mã nguồn

```
scripts/fresh/          các bước của pipeline (xem mục 5)
├── build_prompts_program.py    truy hồi bảng + dựng prompt
├── run_cot.py                  gọi mô hình, hỗ trợ tự host qua --endpoint
├── run_programs.py             thi hành pandas_query trong sandbox BTC
├── build_submission.py         giao chéo, lọc cổng, đóng gói ZIP
├── validate_submission.py      kiểm tra tuân thủ trước khi nộp
├── hard_hop.py                 bộ giải bằng code cho câu nhiều công ty / nhiều năm
├── resolve_ticker.py           phân giải mã công ty (bảng bí danh)
├── parse_statements.py         đọc số kiểu Việt Nam, nhận hệ số đơn vị
└── find_statements.py          nhận diện báo cáo qua đẳng thức Thông tư 200

src/vifin/llm/client.py         danh sách trắng model hợp lệ
vifinqa-official/               mã nguồn tham chiếu của BTC (sandbox thi hành)
data/                           dữ liệu (mục 3)
artifacts/fresh/                kết quả trung gian
submissions/                    bài nộp đã đóng gói
```

---

## 8. Bài nộp

`submissions/legal2.zip` — 3,5 MB, gồm `submission.json` (1.012 dòng) và `data/` với 4.639
bảng CSV.

Mỗi bảng CSV là **tập con trích trực tiếp** từ kho của BTC, giữ nguyên nội dung gốc và truy
vết được về `data/official_corpus/<MÃ>/<NĂM>/<báo_cáo>/…/table_N.csv`. Mọi `pandas_query`
tính kết quả **tại thời điểm thực thi** từ các bảng này.
