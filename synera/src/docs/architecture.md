# Kiến trúc và pipeline Road2AI ViFinQA

## 1. Bài toán và hợp đồng đầu ra

Mỗi câu hỏi tiếng Việt phải được nối với một hoặc nhiều báo cáo tài chính OCR,
chọn đúng bảng/ô, thực hiện phép tính và đóng gói thành `SubmissionSolution`:

- `answer`: scalar hữu hạn;
- `relevant_docs`: ID tài liệu thực sự dùng;
- `relevant_tables`: `document_id|OCR_start_line`;
- evidence CSV: các toán hạng cùng provenance;
- `pandas_query`: biểu thức chạy trên các CSV và tái tạo đúng `answer`.

Thiết kế ưu tiên **grounding và replay**: LLM không được tự cung cấp số cuối nếu
số đó không đến từ ô nguồn. Release audit chạy lại query, kiểm tra schema, nguồn,
table reference, file thiếu/thừa và tính deterministic.

## 2. Sơ đồ thành phần

```text
data/source (OCR reports, questions)
  │
  ├── HTML/OCR parser ── rowspan/colspan expansion ── table semantics
  │                                                  │
  ├── build_index.py ─────────────────────────────> tables.sqlite3
  │                                                  │
  │                          ┌───────────────────────┴──────────────┐
  │                          │                                      │
  │                    Easy/Direct                           Hard Note
  │               candidate generation                 curated retrieval
  │               58-feature reranker                  expression compiler
  │               Qwen3-8B selection                    + execution checks
  │                          │                                      │
  └── build_panel.py ──> financial_panel.json                       │
                             │                                      │
                       Hard formulas                         Note solution
                       Template registry                            │
                             └──────────────────┬───────────────────┘
                                                │
                                    SubmissionSolution adapters
                                                │
                              checkpoint → evidence CSV → ZIP
                                                │
                                     validate/replay/release audit
```

## 3. Tiền xử lý dữ liệu

### 3.1 `build_index`

`src/road2ai_vifinqa/build_index.py` đọc báo cáo, nhận dạng `<table>`, bung ô
`rowspan`/`colspan`, lưu document/table/row/cell vào SQLite và giữ tọa độ OCR.
Context quanh bảng, page number, report scope và fingerprint nguồn được lưu để
truy hồi và phát hiện artifact cũ. Baseline đã đo:

- 1.973 tài liệu;
- 146.246 bảng;
- 1.535.824 dòng;
- 584,849 giây build;
- SQLite 1.175.851.008 bytes.

### 3.2 `table_semantics`

`table_semantics.py`, `html_tables.py` và `source_units.py` phục hồi header nhiều
tầng, section, current/prior/opening period, đơn vị, loại bảng stock/flow/movement,
dòng tổng và scope consolidated/parent. Tầng này đặc biệt quan trọng vì OCR có
thể tách unit khỏi bảng hoặc lặp comparative column ở năm kế tiếp.

### 3.3 `build_panel`

`build_panel.py` chuẩn hóa các metric phổ biến thành key `(ticker, year, metric)`.
Mỗi ô ứng viên được chấm theo code, label, header, period, scope, unit và chất
lượng nguồn; chỉ sau đó mới giải quyết bảng trùng. Panel v2 có 52.066 cells và
build trong 63,918 giây. Comparative backfill chỉ nhận cột prior-year được ghi
rõ trong cùng báo cáo thường niên, không tự relabel năm tài liệu.

## 4. Bốn route giải câu hỏi

Route được khóa theo ID trong `solve.py`:

| Route | IDs | Số câu | Cơ chế |
|---|---|---:|---|
| Direct/Easy | 1–361 | 361 | exhaustive retrieval → deterministic reranker → Qwen chọn ô grounded |
| Hard | 362–426, 440–494, 539–577 | 159 | công thức tài chính deterministic trên panel |
| Note | 427–439, 495–538 | 57 | truy hồi disclosure chuyên biệt; compiler bị giới hạn trên candidate source |
| Template | 578–1012 | 435 | registry câu hỏi/toán hạng/phép tính deterministic |

### Direct/Easy

Candidate generator lọc entity/year/scope, rank table/row/cell và tính 58 feature
về lexical overlap, BM25, header/period/unit, table type, total/component và độ
tin cậy ô. Linear pairwise reranker chỉ đổi thứ tự shortlist; Qwen3-8B Q4_K_M
là selector cuối và chỉ trả candidate ID + operation. Code tự dựng biểu thức từ
các candidate đã chọn, tính đáp án và áp audit guard. Lexical direct solver là
fallback có confidence thấp.

### Hard

`hard_solver.py` và `panel_solver.py` nhận diện metric/công thức như margin,
growth, ROA/ROE, current ratio, inventory days, DOL và aggregation. Toán hạng
đến từ panel, công thức do code kiểm soát. Các metric cần average balance lấy
cả năm hiện tại và năm trước; rolling metric yêu cầu hai năm tài chính liền kề.

### Note

`hard_note_solver.py` chứa spec cho disclosure dài và các trường hợp statement
code. Candidate được dựng từ phrase, context, row label, column header và unit.
Nếu dùng LLM, model chỉ compile expression trên source IDs có sẵn; expression
phải qua parser, grounding và execution check trước khi chấp nhận.

### Template

`template_solver.py` ánh xạ family câu hỏi sang recipe, khóa nguồn, period,
operation và unit. Đây là route ổn định nhất cho các mẫu sinh có cấu trúc. Các
policy thay đổi dấu/ratio được opt-in và phải replay toàn family trước promotion.

## 5. Checkpoint và tính tái lập

Mỗi câu tạo pickle nội bộ và JSON summary trong run directory. Checkpoint chứa
schema, route, identity câu hỏi và chữ ký index/panel (path, size, mtime). Nếu
artifact đổi, checkpoint bị từ chối và câu được chạy lại. Pickle chỉ được tin
cậy khi do local command tạo, không bao giờ đọc từ submission bên ngoài.

Các write quan trọng dùng atomic replace. Submission chỉ publish khi đủ 1.012
câu; partial run vẫn hữu ích để debug nhưng không phải release.

## 6. Ranh giới tin cậy

- OCR và model output là input không tin cậy.
- Model không được tạo literal answer ngoài candidate evidence.
- Source coordinate và unit phải đi cùng toán hạng.
- Không dùng aggregate leaderboard để suy ra nhãn từng câu.
- Local audit chứng minh consistency/replay, không chứng minh hidden correctness.
- Artifact/release cũ không bị ghi đè; candidate mới build song song rồi promote.

## 7. Bản đồ source code

Chi tiết trách nhiệm từng file nằm tại
[`src/road2ai_vifinqa/README.md`](../src/road2ai_vifinqa/README.md).
