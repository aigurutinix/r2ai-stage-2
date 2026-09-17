# Hướng dẫn tái lập

## Môi trường

- Linux hoặc macOS 64-bit;
- Python 3.11;
- SQLite 3 có JSON1 và FTS5;
- tối thiểu 8 GB RAM, 4 GB disk trống cho source, DB và artifact;
- không cần GPU nếu chỉ build/replay submission;
- H200/A100 80 GB chỉ cần khi chạy lại LLM audit tùy chọn.

## 1. Cài dependency

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements-dev.txt
```

## 2. Chuẩn bị dữ liệu

Giải nén dữ liệu BTC vào `ViFinQA/`. Kiểm tra:

```bash
test -f ViFinQA/questions/questions.jsonl
test -f ViFinQA/code_stock.csv
```

## 3. Build end-to-end

Từ raw OCR đến ZIP cuối:

```bash
PYTHONPATH=src python -m vifinqa.release \
  --config configs/release.json build \
  --output-dir outputs/release_build/submission \
  --rebuild-database
```

Nếu đã có `artifacts/vifinqa.db`, bỏ `--rebuild-database`. Builder kiểm tra
checksum của `configs/release_plan.jsonl`, đọc lại từng bảng theo
`evidence_key`, kiểm tra checksum grid, sinh CSV và thực thi Pandas query.
Execution plan không lưu answer hoặc bất kỳ giá trị tài chính nguồn nào.

Nếu một stage bị gián đoạn, giữ nguyên work directory và dùng
`--start-stage N` để tiếp tục từ command thứ N; không kết hợp tùy chọn này với
`--rebuild-database`.

Output cuối là `outputs/release_build/submission.zip`. Nếu golden ZIP có
trên máy, pipeline còn so byte nội dung JSON/CSV với golden và dừng khi drift.

Thời gian tham khảo trên máy phát triển: build DB 10–30 phút; build ZIP từ DB
khoảng 10–20 giây; replay 1.012 câu dưới 10 giây. Không có Modal call trong
đường build này.

`research-rebuild` là entrypoint riêng để chạy lại toàn bộ 20 stage retrieval
và đo drift của thuật toán nghiên cứu. Nó không được dùng thay cho execution
plan đã khóa khi đóng gói release chính thức.

## 4. Kiểm tra golden và tạo hồ sơ release

```bash
PYTHONPATH=src python -m vifinqa.release \
  --config configs/release.json check

PYTHONPATH=src python -m vifinqa.release \
  --config configs/release.json package
```

Kỳ vọng:

```text
items               1012
replayed            1012
evidence_files      5941
provenance_bindings 5941
dataflow_gate       PASS
zero_weight_gate    PASS
```

## 5. Chạy test

```bash
PYTHONPATH=src pytest -q
```

## Lỗi thường gặp

- `DATA_SHARE_URL_NOT_SET`: điền link chia sẻ và ngày xác minh trong config;
- `Golden submission hash changed`: đang dùng nhầm ZIP hoặc ZIP đã bị sửa;
- `evidence is not an exact cited BTC table`: DB/data version không khớp;
- `query must read at least one evidence dataframe`: query vi phạm data-flow;
- `multiplication by literal zero`: biểu thức bỏ qua nguồn bằng zero-weight;
- thiếu FTS5/JSON1: dùng SQLite từ Python 3.11 official hoặc distro package đầy đủ.

## Credential

Core pipeline không cần secret. Modal/Hugging Face token chỉ truyền qua secret
manager hoặc environment cục bộ khi chạy lại model; không ghi vào `.env` được
commit, source, log hoặc release artifact.
