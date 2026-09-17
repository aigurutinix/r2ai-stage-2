# Hướng dẫn clone và tái lập

## 1. Yêu cầu

- Windows 10/11, PowerShell;
- Python 3.11+ (môi trường phát triển hiện dùng Python 3.12);
- đủ dung lượng cho OCR corpus, SQLite index (~1,1 GB), panel và model;
- NVIDIA GPU được khuyến nghị cho Qwen GGUF; deterministic routes chạy CPU;
- llama.cpp server tương thích OpenAI API cho Easy/Note LLM route.

## 2. Clone và môi trường Python

```powershell
git clone https://github.com/hieulenho/Road2AI.git
cd Road2AI
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
$env:PYTHONPATH = (Resolve-Path "src").Path
```

Các tool phụ có extras riêng để người chỉ chạy core pipeline không phải cài
Selenium/SciPy:

```powershell
python -m pip install -e ".[model-download]"  # tải checkpoint có hash
python -m pip install -e ".[analysis]"        # score-history/optimization
python -m pip install -e ".[codabench]"       # browser automation
# hoặc: python -m pip install -e ".[dev]"
```

## 3. Dữ liệu và model không nằm trong Git

`.gitignore` cố ý loại dataset/model/generated artifacts. Cần chuẩn bị layout:

```text
Road2AI/
├─ data/
│  └─ source/ViFinQA/
│     ├─ code_stock.csv
│     ├─ questions/questions.jsonl
│     └─ financial_statements/ # 1.973 OCR reports
├─ artifacts/
│  ├─ models/
│  │  └─ Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf
│  ├─ tables.sqlite3
│  └─ financial_panel.json
└─ tools/llama.cpp/runtime/llama-server.exe
```

Tên file nguồn thực tế được tập trung trong `paths.py`; nếu dataset của bạn có
layout khác, sửa biến môi trường/đường dẫn ở một nơi thay vì hard-code trong
solver. Tuân thủ license của dataset, Qwen và llama.cpp; model weights không
được tái phân phối qua repository này.

## 4. Build sạch

```powershell
python -m road2ai_vifinqa.build_index `
  --force --output artifacts/tables_v2.sqlite3 `
  --manifest artifacts/index_v2_manifest.json

python -m road2ai_vifinqa.build_panel `
  --force --index artifacts/tables_v2.sqlite3 `
  --output artifacts/financial_panel_v2.json `
  --manifest artifacts/financial_panel_v2_manifest.json

python tools/audit_artifacts.py `
  --index artifacts/tables_v2.sqlite3 `
  --panel artifacts/financial_panel_v2.json `
  --integrity --report runs/optimization/artifact_audit.json
```

Không overwrite baseline. Dùng tên v2/candidate, benchmark rồi mới promote.

## 5. Smoke test từng route

```powershell
# Easy (cần Qwen3-8B + llama.cpp)
python -m road2ai_vifinqa.solve --ids 1-3 --run-dir runs/smoke_easy --fail-fast

# Hard
python -m road2ai_vifinqa.solve --ids 362-365 --run-dir runs/smoke_hard --fail-fast

# Note
python -m road2ai_vifinqa.solve --ids 427-429 --run-dir runs/smoke_note --fail-fast

# Template
python -m road2ai_vifinqa.solve --ids 578-580 --run-dir runs/smoke_template --fail-fast
```

Mỗi run có cache/checkpoint, JSON summary, error logs, LLM logs và output build.
Resume là mặc định. Dùng `--no-resume` để bỏ cache; checkpoint stale tự bị reject
khi index/panel đổi.

## 6. Full solve và release

```powershell
python -m road2ai_vifinqa.solve `
  --iteration 1 --run-dir runs/release_candidate `
  --index artifacts/tables_v2.sqlite3 `
  --panel artifacts/financial_panel_v2.json `
  --ids 1-1012 --fail-fast

python tools/release_audit.py `
  --zip runs/release_candidate/submission.zip `
  --run-dir runs/release_candidate `
  --table-ref-mode one-based --replays 3 `
  --report runs/release_candidate/final_release_audit.json
```

Chỉ publish khi đủ 1.012 rows/CSVs, mọi reference tồn tại, query chạy nhiều lần
cho cùng kết quả và không có orphan file. Không dùng `--publish` cho partial run.

## 7. Test dành cho contributor

```powershell
python -m unittest discover -s tests -v
python -m compileall -q src tools tests
git status --short
```

Khi sửa retrieval: chạy retrieval benchmark. Khi sửa panel: replay 159 Hard và
435 Template. Khi sửa arithmetic policy: test mutation, replay đúng family và
full release audit. Ghi rõ official score khác local metric.

## 8. Lỗi thường gặp

- **Thiếu Qwen/checkpoint incomplete:** kiểm tra path và file >4 GB cho Easy.
- **Port 8087 đã dùng:** Easy solver sẽ từ chối nếu `/v1/models` báo model khác.
- **Checkpoint mismatch:** artifact đã đổi; để solver regenerate hoặc dùng run mới.
- **Sai table reference:** dùng OCR one-based start line, không phải table ordinal.
- **Khác answer sau clone:** xác minh dataset snapshot, artifact manifest, model
  revision/hash và policy flags; không copy checkpoint từ artifact set khác.
