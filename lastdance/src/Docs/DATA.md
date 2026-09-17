# Dữ liệu

## Nguồn sử dụng

Pipeline chỉ sử dụng dữ liệu tài chính và câu hỏi do BTC cung cấp trong thư mục
`ViFinQA/`. Không có số liệu tài chính bên ngoài được bổ sung vào retrieval hay
evidence của bài nộp.

| Nhóm | Nội dung | Vai trò |
| --- | --- | --- |
| BTC gốc | 1.973 báo cáo OCR/TXT của 100 mã, năm 2015–2025 | nguồn duy nhất của evidence |
| BTC gốc | `questions/questions.jsonl`, 1.012 câu | input truy vấn |
| BTC gốc | `code_stock.csv` | ticker và tên doanh nghiệp |
| Dẫn xuất | `artifacts/vifinqa.db` | index/table warehouse tái tạo được |
| Dẫn xuất | `analysis/*_manual_overrides.csv` | tọa độ ô nguồn đã audit |
| Dẫn xuất | `data/*.csv` trong submission | toàn bộ grid của bảng nguồn được trích |

Các context gửi lên Modal được mask toàn bộ số. LLM chỉ thấy nhãn dòng/cột và
metadata cần thiết để xếp hạng candidate; không có dữ liệu ngoài hoặc answer.

## Cấu trúc và schema

Warehouse hiện có 1.973 documents, 121.756 pages, 146.246 tables và 146.246
records FTS. Scope báo cáo gồm 957 consolidated, 954 separate, 55 unspecified
và 7 aggregated.

Các bảng chính:

- `documents(document_id, ticker, year, report_scope, relative_path, sha256, …)`;
- `pages(document_id, page_ordinal, page_number, start_line_1, end_line_1, …)`;
- `tables(table_id, evidence_key, document_id, source_line_1, unit_code,
  table_kind, grid_json, raw_html, …)`;
- `table_fts`: full-text index cho title, context và nội dung bảng;
- `table_catalog`: view phẳng phục vụ inspection/retrieval.

`evidence_key` có dạng `<document_id>|<source_line_1>`. Evidence CSV dùng header
vị trí `col_0`, `col_1`, … và giữ nguyên grid OCR; vì vậy `df.iloc[r, c]` ánh xạ
ổn định trở lại `grid_json[r][c]`.

## Tải và đặt dữ liệu

1. Tải bộ dữ liệu được đội chia sẻ hoặc từ nguồn BTC được cấp quyền.
2. Giải nén sao cho tồn tại `ViFinQA/questions/questions.jsonl` và
   `ViFinQA/code_stock.csv`.
3. Điền `data_distribution.share_url` và `access_verified_at` trong
   `configs/release.json`.
4. Kiểm tra link bằng một tài khoản không thuộc đội trước khi nộp hồ sơ.

Link chia sẻ chưa thể được tự động tạo bởi source code và hiện được validator
báo thành `DATA_SHARE_URL_NOT_SET`. Đây là blocker vận hành duy nhất cần chủ
sở hữu đội hoàn tất.

## Build và kiểm tra checksum

```bash
PYTHONPATH=src python -m vifinqa.corpus_db build \
  --data-root ViFinQA --database artifacts/vifinqa.db

PYTHONPATH=src python -m vifinqa.release \
  --config configs/release.json package
```

`release_manifest.json` lưu SHA-256 của questions, company catalog, database và
submission. `evidence_provenance.jsonl` lưu `question_id`, `document_id`,
`source_line_1`, checksum grid nguồn, checksum CSV và extractor version cho
từng evidence file.

Không sửa trực tiếp database hoặc CSV release. Khi dữ liệu gốc đổi, build lại
warehouse và coi đây là một data version mới.
