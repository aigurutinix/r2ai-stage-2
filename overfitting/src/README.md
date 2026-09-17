# ViFinQA Financial QA Pipeline

Pipeline truy hồi bảng và sinh biểu thức Pandas để trả lời 1.012 câu hỏi tài
chính tiếng Việt trên 1.973 báo cáo tài chính OCR. Hệ thống kết hợp retrieval,
Qwen2.5-Coder-7B-Instruct, challenger Qwen2.5-Coder-14B-Instruct và các financial solver tất định. Mọi đáp án hợp lệ
đều được tính tại thời điểm thực thi từ CSV evidence đi kèm submission.

## Checkpoint private mới nhất: V67

**V67 đạt Answer Accuracy và Execution Accuracy `0.5277` trên private test**,
theo kết quả leaderboard đội xác nhận. Table F2 là `0.5482`; Docs F2 là
`0.9518`. Đây là checkpoint pipeline hybrid, không phải checkpoint trọng số
được fine-tune riêng.

- [Thông tin V67, artifact và phạm vi tái hiện](CHECKPOINT_V67.md).
- [Slide Demo Day V67 (PowerPoint)](deliverables/demo_day/output/Overfitting_DemoDay_2026_V67.pptx).
- [Slide Demo Day V67 (PDF)](deliverables/demo_day/output/Overfitting_DemoDay_2026_V67.pdf).
- [Kịch bản 5 phút và hỏi đáp](deliverables/demo_day/output/KICH_BAN_5_PHUT_VA_HOI_DAP.md).

Các mục cài đặt và chạy baseline bên dưới được giữ lại từ hồ sơ public V28.
Để tái hiện chính xác V67 cần thêm các artifact private nêu trong
`CHECKPOINT_V67.md`; chỉ clone mã nguồn không đủ. Không so trực tiếp điểm
public `0.4644` với private `0.5277`.

## 1. Checkpoint public V28 (lịch sử)

Checkpoint public được ghi nhận ở phần này là
`ViFinQA V28 - exact direct-growth audited3`.

| Thành phần | Phiên bản |
|---|---|
| Git branch | `improve_baseline_kien` |
| Git commit | `1d52e9d` |
| Package | `vifinqa==0.2.0` |
| Retrieval | `retrieval_v21_failed21_probe_depth112_w010.jsonl` |
| Codegen | `codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl` |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Model revision | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| Submission | `submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip` |
| SHA-256 artifact leaderboard gốc | `ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e` |

Kết quả leaderboard:

| Metric | Điểm |
|---|---:|
| `TABLES_F2MACRO` | 0.5530 |
| `DOCS_F2MACRO` | 0.9420 |
| `TABLES_PRECISION` | 0.3621 |
| `TABLES_RECALL` | 0.7480 |
| `TABLES_MRR5` | 0.6181 |
| `DOCS_PRECISION` | 0.9494 |
| `DOCS_RECALL` | 0.9457 |
| `DOCS_MRR5` | 0.9743 |
| `ANSWER_ACCURACY` | **0.4644** |
| `EXECUTION_ACCURACY` | **0.4644** |

Điểm `0.4644` là kết quả của toàn bộ pipeline hybrid, không phải độ chính xác
độc lập của riêng Qwen.

## 2. Kiến trúc hệ thống

```text
Dữ liệu OCR của BTC
    |
    v
HTML table parser + normalized Parquet store
    |
    v
Vietnamese router + canonical metric dictionary
    |
    v
BM25 + row-level retrieval + evidence requirements
    |
    +--------------------------+
    |                          |
    v                          v
Qwen2.5-Coder-7B          Typed financial planners/solvers
Text-to-Pandas            lookup/count/ranking/ratio/growth/notes
    |                          |
    +-------------+------------+
                  |
                  v
Exact challengers + audited allowlist merge
                  |
                  v
Semantic guard + DataFrame replay + unit/evidence checks
                  |
                  v
results.json + data/*.csv -> submission.zip
```

Các thành phần chính:

- `vifinqa/extraction/`: đọc OCR, tách `<table>`, xử lý `rowspan/colspan`, đơn
  vị và tạo normalized store.
- `vifinqa/router/`: nhận diện doanh nghiệp, năm, loại báo cáo, đơn vị, metric
  và phép toán.
- `vifinqa/finance/metrics.py`: canonical metric dictionary và qualifier tài
  chính.
- `vifinqa/retrieval/`: BM25, row shortlist và table ranking.
- `vifinqa/codegen/`: Qwen runner, typed planner, formula solver, exact
  challengers và semantic guard.
- `vifinqa/submission/`: dựng CSV evidence, replay query và đóng gói ZIP.
- `scripts/01..15`: CLI build, retrieval, codegen, eval, audit và merge.

## 3. Dữ liệu và checkpoint

### 3.1. Dữ liệu ViFinQA

Dữ liệu được tải từ:

- [AIGuruTinix/ViFinQA trên Hugging Face](https://huggingface.co/datasets/AIGuruTinix/ViFinQA)

Dataset gồm 1.012 câu hỏi, 1.973 báo cáo của 100 doanh nghiệp trong giai đoạn
2015-2025. Báo cáo là file `.txt` UTF-8 có nội dung OCR và bảng HTML inline.

Cài Hugging Face CLI rồi tải toàn bộ dataset:

```bash
python -m pip install "huggingface_hub>=0.27"
hf download AIGuruTinix/ViFinQA \
  --repo-type dataset \
  --local-dir data/ViFinQA
```

Sau khi tải, cấu trúc bắt buộc là:

```text
data/ViFinQA/
├── code_stock.csv
├── questions/
│   └── questions.jsonl
└── financial_statements/
    └── TICKER/YEAR/DOCUMENT/DOCUMENT_extracted.txt
```

Kiểm tra nhanh:

```bash
python -c "from vifinqa import config; print(config.QUESTIONS_JSONL.exists(), config.FS_DIR.exists(), config.CODE_STOCK_CSV.exists())"
```

Kết quả phải là `True True True`.

Lưu ý: `datasets.load_dataset("AIGuruTinix/ViFinQA")` chỉ nạp split câu hỏi
qua Hugging Face Datasets. Pipeline này cần toàn bộ repository, đặc biệt thư
mục `financial_statements/`, nên hãy dùng `hf download` như trên.

### 3.2. Trọng số mô hình

Hệ thống sử dụng checkpoint công khai:

- [Qwen2.5-Coder-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [Revision cố định `c03e6d3`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242)
- Giấy phép: Apache-2.0

Tải thủ công khi cần chạy offline:

```bash
hf download Qwen/Qwen2.5-Coder-7B-Instruct \
  --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 \
  --local-dir checkpoints/Qwen2.5-Coder-7B-Instruct
```

Pipeline không fine-tune, không có LoRA/adaptor riêng. Khi chạy trên Kaggle,
model được lượng tử hóa 4-bit NF4, double quantization, compute `float16`.

### 3.3. Artifact frozen của V28

Để tái tạo nội dung submission leaderboard `0.4644`, bộ bàn giao phải có hai
file sau trong `artifacts/`:

```text
artifacts/
├── retrieval_v21_failed21_probe_depth112_w010.jsonl
└── codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl
```

| File | SHA-256 |
|---|---|
| Retrieval V21 | `f3a5f55b737c431ed8a927587a7b4097280f023d54eeb68ec3cab0bde2fca608` |
| Codegen V28 | `b22adf22d34315ac10411451f0a153947c28382c314a79eba9fdfadd93d97d8d` |

Hai file này chứa retrieval và dự đoán của cuộc thi, không phải trọng số model,
nên không được commit vào Git. Chúng phải được đặt trong gói artifact bàn giao
hoặc thư mục Drive/OneDrive được chia sẻ cho Ban Tổ chức. Sau khi tải, chép đúng
vào `artifacts/` và kiểm tra SHA theo mục 8.

> Chủ sở hữu repository cần điền đường dẫn Drive/OneDrive của gói artifact V28
> vào đây trước khi gửi hồ sơ nghiệm thu. Không tạo link giả hoặc link yêu cầu
> quyền mà Ban Tổ chức chưa được cấp.

## 4. Yêu cầu môi trường

### 4.1. Pipeline local

- Hệ điều hành: Windows, Linux hoặc macOS.
- Python: `3.11` đến `3.13`.
- CPU pipeline không yêu cầu GPU.
- Khuyến nghị tối thiểu 8 GB RAM và 5 GB dung lượng trống.
- Chạy lệnh từ thư mục gốc repository.

Môi trường local đã kiểm thử:

| Phần mềm | Phiên bản |
|---|---:|
| Python | 3.13.2 |
| pandas | 3.0.5 |
| numpy | 2.5.1 |
| pyarrow | 25.0.0 |
| beautifulsoup4 | 4.15.0 |
| lxml | 6.1.1 |
| rapidfuzz | 3.14.5 |
| tqdm | 4.70.0 |

### 4.2. Suy luận Qwen

- Môi trường đã dùng: Kaggle Notebook.
- GPU đã dùng: NVIDIA T4, model 7B ở chế độ NF4 4-bit.
- Internet phải bật để tải model từ Hugging Face.
- `transformers>=4.45,<5`
- `accelerate>=1,<2`
- `bitsandbytes>=0.45,<1`
- Backend: Hugging Face Transformers, không dùng vLLM mới trên T4.

## 5. Cài đặt

### 5.1. Clone đúng checkpoint mã nguồn

```bash
git clone https://github.com/quangkhai5122/R2AI.git
cd R2AI
git checkout 1d52e9d
```

Repository phải được chuyển sang public hoặc Ban Tổ chức phải được cấp quyền
trước khi dùng link trên.

### 5.2. Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest "huggingface_hub>=0.27"
```

### 5.3. Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pytest "huggingface_hub>=0.27"
```

Chạy test sau khi cài đặt:

```bash
python -m pytest tests -q
```

Checkpoint V28 đã được xác nhận với `401 passed`.

## 6. Cấu hình

Cấu hình mặc định nằm trong `vifinqa/config.py`:

| Tham số | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `DATA_DIR` | `data/ViFinQA` | dữ liệu đầu vào |
| `STORE_DIR` | `artifacts/store` | Parquet store đã parse |
| `RETRIEVAL_JSONL` | `artifacts/retrieval.jsonl` | retrieval hiện hành |
| `CODEGEN_JSONL` | `artifacts/codegen_results.jsonl` | codegen hiện hành |
| `SUBMISSION_DIR` | `artifacts/submission` | thư mục bài nộp |
| `TABLE_POS_MODE` | `line` | dùng line number của `<table>` |
| `SUBMISSION_K` | `5` | số bảng nộp tối đa mỗi câu |
| `CODEGEN_K` | `6` | số bảng mặc định đưa cho codegen |

Các đường dẫn đều có thể ghi đè bằng tham số CLI. Xem đầy đủ bằng:

```bash
python scripts/01_build_store.py --help
python scripts/02_retrieve.py --help
python scripts/04_make_kaggle_payload.py --help
python scripts/05_build_submission.py --help
```

Quy tắc quan trọng: `relevant_tables` phải dùng số dòng 1-based nơi thẻ
`<table>` bắt đầu trong file OCR, không phải thứ tự xuất hiện của bảng. Store
lưu ánh xạ `table_pos -> line_no`; submission builder áp dụng ánh xạ này.

## 7. Chạy lại sản phẩm từ dữ liệu gốc

Phần này tạo lại pipeline end-to-end từ dataset. Kết quả Qwen mới có thể không
byte-identical với artifact frozen do phiên bản CUDA/GPU/thư viện, dù dùng
`temperature=0`. Muốn dựng đúng submission đã chấm `0.4644`, dùng mục 8.

### 7.1. Build normalized store

```bash
python scripts/01_build_store.py
```

Output:

```text
artifacts/store/
├── reports.parquet
├── tables/*.parquet
└── cells/*.parquet
```

Smoke test với một số doanh nghiệp:

```bash
python scripts/01_build_store.py \
  --tickers VNM,VJC,ACB \
  --store-dir artifacts/store_smoke
```

Dùng thư mục output riêng cho smoke test để không ghi đè full store.

### 7.2. Chạy retrieval

Baseline retrieval:

```bash
python scripts/02_retrieve.py \
  --questions data/ViFinQA/questions/questions.jsonl \
  --store-dir artifacts/store \
  --code-stock data/ViFinQA/code_stock.csv \
  --out artifacts/retrieval.jsonl
```

Retrieval row-rerank dùng trọng số `w=0.10`:

```bash
python scripts/02_retrieve.py \
  --depth 112 \
  --row-rerank \
  --row-score-weight 0.10 \
  --out artifacts/retrieval_w010.jsonl
```

`w=0.10` là trọng số của row-level shortlist score khi trộn vào điểm table
retrieval. Đây không phải số bảng nộp và không phải temperature của model.

### 7.3. Chạy deterministic baseline

```bash
python scripts/03_rule_baseline.py \
  --retrieval artifacts/retrieval.jsonl \
  --out artifacts/codegen_results.jsonl
```

Pipeline hiện tại có canonical metric dictionary và typed formula solver cho
lookup, count, ranking, ratio, margin, difference, growth, average balance,
temporal event, note matrix và các phép aggregate.

### 7.4. Build submission baseline

```bash
python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval.jsonl \
  --codegen artifacts/codegen_results.jsonl \
  --out-dir artifacts/submission_baseline
```

Output cần nộp:

```text
artifacts/submission_baseline/submission.zip
```

### 7.5. Tạo Kaggle payload

```bash
python scripts/04_make_kaggle_payload.py \
  --retrieval artifacts/retrieval.jsonl \
  --store-dir artifacts/store \
  --out artifacts/kaggle_payload \
  --dataset-slug vifinqa-payload \
  --dataset-id <KAGGLE_USERNAME>/vifinqa-payload
```

Thay `<KAGGLE_USERNAME>` bằng username Kaggle thực tế trước khi chạy.

Nếu dùng Kaggle CLI, tạo dataset lần đầu:

```bash
python -m pip install kaggle
kaggle datasets create -p artifacts/kaggle_payload --dir-mode zip
```

Các lần sau:

```bash
kaggle datasets version \
  -p artifacts/kaggle_payload \
  --dir-mode zip \
  -m "refresh ViFinQA payload"
```

Payload có `payload-manifest.json` và SHA-256 của code/store/retrieval. Notebook
sẽ từ chối payload thiếu file hoặc sai hash.

### 7.6. Chạy Qwen trên Kaggle

1. Tạo Kaggle Notebook và chọn GPU T4.
2. Bật Internet.
3. Attach đúng một dataset payload vừa tạo.
4. Import `kaggle/vifinqa-codegen.ipynb`.
5. Chạy smoke test trước, sau đó chạy full inference.

Lệnh full inference tương ứng:

```bash
python /kaggle/working/code/kaggle_codegen.py \
  --payload "$PAYLOAD" \
  --backend hf \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --load-4bit \
  --out /kaggle/working/codegen_results.jsonl \
  --n 1 \
  --temperature 0 \
  --k 4 \
  --max-tokens 256 \
  --batch-size 4 \
  --checkpoint-every 32 \
  --time-budget-min 400 \
  --seed 13
```

Runner ghi baseline đủ 1.012 dòng trước khi gọi model, checkpoint theo chunk và
tự giảm batch `4 -> 2 -> 1` nếu CUDA OOM. Có thể resume khi giữ nguyên payload,
run signature và file output.

Sau khi hoàn tất, tải `codegen_results.jsonl` từ Kaggle Output về máy và chạy:

```bash
python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval.jsonl \
  --codegen /path/to/codegen_results.jsonl \
  --out-dir artifacts/submission_qwen
```

### 7.7. Audit và merge challenger

Các exact challenger không tự ghi đè output Qwen. Script audit tạo shadow
candidate; chỉ các ID đã được kiểm tra thủ công mới được merge bằng allowlist:

```bash
python scripts/15_audit_checkpoint_challengers.py --help
python scripts/11_merge_codegen.py --help
```

Không merge toàn bộ disagreement tự động. Canonical parent row có thể sai với
câu hỏi metric con, kỳ đầu/cuối năm, đối tác, maturity hoặc qualifier ngân hàng.
Lịch sử allowlist và từng checkpoint V1-V28 nằm trong `CHECKPOINTS.md` và
`RUNBOOK.md`.

## 8. Tái tạo nội dung checkpoint V28

Luồng này không rerun Qwen. Nó dùng retrieval và codegen frozen đã được audit.

### 8.1. Kiểm tra artifact đầu vào

Linux:

```bash
sha256sum \
  artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl \
  artifacts/codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl
```

macOS:

```bash
shasum -a 256 \
  artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl \
  artifacts/codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl
```

Expected:

```text
f3a5f55b737c431ed8a927587a7b4097280f023d54eeb68ec3cab0bde2fca608  retrieval_v21_failed21_probe_depth112_w010.jsonl
b22adf22d34315ac10411451f0a153947c28382c314a79eba9fdfadd93d97d8d  codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl
```

### 8.2. Build submission

```bash
python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl \
  --codegen artifacts/codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl \
  --store-dir artifacts/store \
  --out-dir artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010
```

### 8.3. Kiểm tra ZIP

```bash
unzip -t artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

Kết quả phải có `No errors detected in compressed data`.

Nếu đang kiểm đúng artifact leaderboard gốc đã tải từ gói bàn giao, dùng một
trong hai lệnh sau:

Linux:

```bash
sha256sum artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

macOS:

```bash
shasum -a 256 artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

SHA-256 của đúng file ZIP đã được nộp lên leaderboard là:

```text
ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e
```

SHA này dùng để nhận diện artifact leaderboard gốc. Khi build lại, nội dung
`results.json` và 2.092 CSV không đổi nhưng raw SHA của ZIP có thể khác vì
`zipfile` lưu thời gian tạo file trong metadata. Đây không phải khác biệt về
answer, query hoặc evidence.

SHA-256 của `results.json` sau khi giải nén phải là:

```text
643d531536323d4b207486dd05764367c987bcaa9419a17226c3fdbeee616d1f
```

Kiểm tra:

```bash
python -c "import hashlib,pathlib; p=pathlib.Path('artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/results.json'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

Trong lần xác minh README, bản build mới có `1.012` entry, `2.092` CSV và toàn
bộ cây file sau giải nén giống artifact leaderboard (`diff -qr` không có khác
biệt).

## 9. Định dạng và kiểm soát submission

`submission.zip` có cấu trúc:

```text
submission.zip
├── results.json
└── data/
    └── *.csv
```

CSV evidence dùng schema dạng dài:

```text
row,label,code,col,col_name,value,unit_scale
```

Mỗi giá trị trong `pandas_query` phải được đọc hoặc tính từ các DataFrame CSV
đi kèm. Submission builder thực hiện các kiểm tra:

- query chỉ tham chiếu các DataFrame có evidence;
- query compile và replay được;
- kết quả replay khớp trường `answer`;
- evidence có thể truy vết về `report_id` và `line_no`;
- CSV không chứa cột đáp án cuối được chuẩn bị sẵn;
- đơn vị và output scalar hợp lệ;
- ZIP đủ `results.json` và các CSV được tham chiếu.

Trạng thái V28:

| Kiểm tra | Kết quả |
|---|---:|
| Entry trong `results.json` | 1.012 |
| CSV evidence | 2.092 |
| Expression compile/replay | 1.012/1.012 |
| Query phụ thuộc trực tiếp DataFrame | 1.001 |
| Query chưa giải, placeholder `0.0` | 11 |
| ZIP integrity | Pass |

Các ID chưa giải: `424, 425, 426, 464, 497, 510, 538, 805, 907, 950, 995`.
Chúng được đánh dấu thất bại rõ ràng, không phải đáp án gán cứng.

## 10. Kiểm thử và offline eval

Chạy toàn bộ source tests:

```bash
python -m pytest tests -q
```

Sinh bộ eval kỹ thuật:

```bash
python scripts/09_gen_eval_suite.py --per-class 60
python scripts/02_retrieve.py \
  --questions artifacts/eval/eval_questions.jsonl \
  --out artifacts/eval/eval_retrieval.jsonl
python scripts/03_rule_baseline.py \
  --retrieval artifacts/eval/eval_retrieval.jsonl \
  --out artifacts/eval/eval_codegen.jsonl
python scripts/05_build_submission.py \
  --retrieval artifacts/eval/eval_retrieval.jsonl \
  --codegen artifacts/eval/eval_codegen.jsonl \
  --out-dir artifacts/eval/eval_submission \
  --offline-eval
python scripts/07_evaluate.py \
  --submission artifacts/eval/eval_submission \
  --gold artifacts/eval/eval_gold.json \
  --by-class
```

Không nộp ZIP của bộ eval lên leaderboard. Bộ này được sinh từ parser/store
hiện tại và chỉ dùng làm regression/smoke test, không thay thế gold của BTC.

## 11. Troubleshooting

| Hiện tượng | Cách xử lý |
|---|---|
| Không tìm thấy dữ liệu | Kiểm tra dataset nằm đúng tại `data/ViFinQA/` |
| `load_dataset()` chỉ có câu hỏi | Dùng `hf download` để lấy cả `financial_statements/` |
| Retrieval thiếu `plan` | Chạy lại `scripts/02_retrieve.py` sau khi đổi router/metrics |
| Qwen output toàn `source=rule` | Xem log model, tăng `max_tokens`, kiểm tra Internet/GPU |
| CUDA OOM | Giảm batch, `--k`, `--max-input-tokens` hoặc `--max-tokens` |
| Payload hash mismatch | Build và upload lại toàn bộ `artifacts/kaggle_payload/` |
| Vị trí bảng sai | Bắt buộc dùng `line_no` 1-based của dòng `<table>` |
| SHA ZIP rebuild khác artifact gốc | ZIP chứa timestamp; kiểm SHA retrieval/codegen, SHA `results.json` và nội dung giải nén |
| GitHub trả về 404 | Repository đang private; cấp quyền hoặc chuyển public |

## 12. Tài liệu đi kèm

- `RUNBOOK.md`: nguồn sự thật về lệnh chạy và trạng thái pipeline.
- `CHECKPOINTS.md`: lịch sử checkpoint, audit, allowlist và leaderboard.
- `MODEL_CHECKPOINT_0.4644.md`: model, revision và hướng dẫn checkpoint.
- `DATA_CSV_PANDAS_QUERY_0.4644.md`: schema CSV và quy định Pandas query.
- `SUBMISSION_0.4644.md`: cấu trúc, truy vết và kiểm chứng bài nộp V28.
- `deliverables/csv_evidence_samples_0.4644/`: CSV evidence mẫu và script verify.
- `ViFinQA_Claude_Strategy.md`: chiến lược nghiên cứu ban đầu.

## 13. Giấy phép và lưu ý sử dụng

- Qwen2.5-Coder-7B-Instruct được phát hành theo Apache-2.0.
- Dữ liệu báo cáo tài chính kế thừa điều kiện sử dụng của nguồn dữ liệu trên
  Hugging Face; người dùng phải tuân thủ attribution và giới hạn sử dụng tương
  ứng.
- OCR có thể sai dấu, số, cấu trúc bảng hoặc thứ tự đọc. Không sử dụng output
  của hệ thống như tư vấn tài chính hoặc nguồn số liệu chính thức nếu chưa kiểm
  tra lại báo cáo gốc.
- Dữ liệu, artifact dự đoán và submission không được commit vào Git vì dung
  lượng và yêu cầu quản lý dữ liệu cuộc thi; chúng phải được chia sẻ qua kênh
  bàn giao có kiểm soát và kèm SHA-256.

## 14. Checklist bàn giao

- [ ] Repository/commit `1d52e9d` truy cập được bởi Ban Tổ chức.
- [ ] Link dataset Hugging Face truy cập được.
- [ ] Link model revision cố định truy cập được.
- [ ] Link Drive/OneDrive chứa retrieval/codegen/submission V28 đã được điền.
- [ ] Ban Tổ chức có quyền tải gói artifact.
- [ ] SHA-256 retrieval, codegen và `results.json` khớp tài liệu.
- [ ] Nếu nộp lại đúng ZIP leaderboard gốc, raw SHA của ZIP khớp tài liệu.
- [ ] `python -m pytest tests -q` chạy thành công.
- [ ] `unzip -t submission.zip` chạy thành công.
- [ ] README và ba tài liệu nghiệm thu được commit cùng checkpoint.
