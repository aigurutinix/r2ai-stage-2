# Mô tả và hướng dẫn dữ liệu — KINGPRO V297

## Nguồn dữ liệu

KINGPRO chỉ dùng dữ liệu do BTC cung cấp trong ViFinQA, nguồn công khai:

```text
https://huggingface.co/datasets/AIGuruTinix/ViFinQA
revision: 0450088ab22ec946f04f097586967ca405955b3b
```

Snapshot gồm 1.012 câu hỏi tiếng Việt, 1.973 báo cáo OCR của 100 doanh nghiệp,
giai đoạn 2015–2025. Catalog do KINGPRO trích được 146.246 bảng.

## Cấu trúc gốc

```text
data/
  code_stock.csv
  questions/questions.jsonl
  financial_statements/<ticker>/<year>/<document>/<document>_extracted.txt
```

- `questions.jsonl`: `id`, `question`.
- `code_stock.csv`: mã chứng khoán và tên doanh nghiệp.
- `*_extracted.txt`: OCR UTF-8; bảng nằm trong thẻ HTML `<table>`.

## Dữ liệu trung gian

KINGPRO tách mỗi HTML table thành CSV, giữ metadata:

```text
table_ref = <document>|<dòng bắt đầu table, 1-based>
```

Catalog giữ ticker, năm, scope, trang, dòng, số hàng/cột, tiêu đề và nhãn dòng.
Header nhiều tầng và unit marker gần bảng được giữ để tránh chọn nhầm cột hoặc
nhân sai đơn vị.

## CSV evidence trong bài nộp

`sub_v297_scope2_a.zip` chứa `submission.json` và thư mục `data/`. Mỗi evidence
CSV là:

1. bản sao một bảng được trích từ dữ liệu BTC; hoặc
2. bảng compact chỉ chứa các ô/hàng/cột cần cho phép tính, kèm `source_table`,
   `source_csv`, `row_idx`, `col_idx`, `raw`, `scale` để truy ngược.

Runtime audit đã đọc lại 7.299 source cells trên đủ 1.012 câu, kiểm coordinate,
raw token và ràng buộc với `relevant_tables`; 0 missing, 0 citation-unbound.
Con số này chứng minh provenance của chương trình, không đồng nghĩa 100% answer
accuracy.

## Tải và dùng dữ liệu

```bash
git clone https://huggingface.co/datasets/AIGuruTinix/ViFinQA data/ViFinQA
git -C data/ViFinQA checkout 0450088ab22ec946f04f097586967ca405955b3b
```

Đặt/symlink các thư mục vào layout ở trên, sau đó chạy pipeline theo
`SOURCE_AND_DEPLOYMENT.md`. `KINGPRO_V297_DATA_PROVENANCE.zip` đã kèm final
submission, data card, source audit và runtime-lineage report để BTC kiểm tra
mà không phải tải toàn bộ corpus trước.

## Quyền sử dụng và giới hạn

Báo cáo OCR gốc được mô tả theo CC BY-NC 4.0 trong tài liệu dự án. Dataset card
ViFinQA không nên được suy diễn thành quyền rộng hơn nguồn. Hồ sơ này phục vụ
đánh giá/nghiệm thu cuộc thi; không dùng dữ liệu để tư vấn đầu tư thời gian thực.
