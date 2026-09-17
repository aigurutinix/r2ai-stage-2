# Data Card — ViFinQA trong KINGPRO

## Tổng quan

Repo dùng bản phát hành công khai ViFinQA (`AIGuruTinix/ViFinQA`) gồm 1.012 câu hỏi tiếng Việt và 1.973 báo cáo OCR của 100 doanh nghiệp niêm yết, giai đoạn 2015–2025. Báo cáo là văn bản UTF-8, có phân trang và bảng HTML nội dòng.

| Trường | Giá trị |
| --- | --- |
| Ngôn ngữ | Tiếng Việt |
| Câu hỏi | 1.012 |
| Báo cáo | 1.973 |
| Doanh nghiệp | 100 mã chứng khoán |
| Thời gian | 2015–2025 |
| Input | OCR text + inline HTML tables |
| Gold public đi kèm | Không có answer/program/evidence trong package câu hỏi |

## Cấu trúc dữ liệu gốc

```text
data/
  code_stock.csv
  questions/questions.jsonl
  financial_statements/TICKER/YEAR/DOCUMENT/DOCUMENT_extracted.txt
```

`code_stock.csv` ánh xạ mã chứng khoán sang tên công ty. `questions.jsonl` chứa `id` và `question`. Đường dẫn báo cáo cung cấp ticker, năm và document ID.

## Biến đổi do KINGPRO thực hiện

1. Tìm từng thẻ `<table>...</table>` trong OCR.
2. Ghi lại dòng bắt đầu 1-based và trang gần nhất.
3. Parse bằng `pandas.read_html(..., flavor="lxml")`.
4. Xuất DataFrame thành CSV.
5. Tạo table reference chuẩn `DOCUMENT|LINE`.
6. Bổ sung metadata ticker, năm, scope, schema và row labels vào catalog retrieval.

CSV evidence thông thường là bản sao của bảng đã trích. Với câu nhiều nguồn, manifest `q<ID>_source_cells.csv` chứa tập ô tối thiểu cần tính và các trường truy vết:

- `source_table`: `DOCUMENT|LINE`;
- `source_csv`: CSV bảng gốc;
- `row_idx`, `col_idx`: tọa độ ô;
- `raw`: giá trị thô;
- metadata tài chính như ticker, year, metric và scale khi có.

Audit V297 hiện tại đã đọc lại 7.299 source cells trên đủ 1.012 câu. Trong đó
7.041 audited-manifest cells thuộc 817 câu và 258 causal cells thuộc 195 legacy
programs. Mọi cell đều khớp tọa độ/raw trong bảng vật lý và thuộc
`relevant_tables`; 0 missing, 0 citation-unbound. Candidate cuối khai table
references lấy trực tiếp từ evidence thực được query đọc.

## Provenance

`scripts/build_compliance_safe_candidate.py` là nguồn sự thật cho pass provenance:

- CSV tên `DOCUMENT_LINE.csv` được ánh xạ thành `DOCUMENT|LINE`.
- Manifest được mở và lấy từng `source_table`.
- AST xác định `df1`, `df2`, ... thực sự được load; nếu không thể chứng minh tập hẹp hơn, toàn bộ evidence được giữ để không làm mất nguồn.
- `relevant_docs` được suy ra từ phần trước dấu `|` của `relevant_tables`.

`scripts/check_submission_compliance.py` dựng lại phép ánh xạ độc lập lúc kiểm tra và yêu cầu danh sách khai báo khớp chính xác.

## Chất lượng và giới hạn

- OCR có thể làm hỏng dấu tiếng Việt, dấu phân cách hàng nghìn, header nhiều tầng và thứ tự đọc.
- Một công ty/năm có thể có báo cáo hợp nhất, riêng lẻ hoặc tài liệu bổ sung; chọn sai scope gây sai số dù chỉ tiêu giống tên.
- Đơn vị có thể xuất hiện ngoài bảng, trong header hoặc ghi chú.
- Dữ liệu không đại diện cho thị trường hiện tại và không dùng trực tiếp làm tư vấn tài chính.
- V297 có 1.012/1.012 chương trình chạy được; điều này không đồng nghĩa mọi đáp
  án đúng. Semantic errors do OCR/scope/unit vẫn được theo dõi riêng.

## Giấy phép và quyền sử dụng

Dữ liệu BCTC có nguồn từ TiniX Vietnam OCR Annual Financial Statements, được công bố theo CC BY-NC 4.0. Bản ViFinQA trong repo không kèm giấy phép riêng cho annotation câu hỏi; người dùng không nên suy diễn quyền rộng hơn tài liệu nguồn. Khi công bố lại, cần ghi nguồn ViFinQA và TiniX, tuân thủ điều kiện phi thương mại và quy định dữ liệu áp dụng.
