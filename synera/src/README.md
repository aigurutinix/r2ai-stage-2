# Road2AI ViFinQA

Hệ thống end-to-end cho bài toán **Vietnamese Financial Question Answering** trên
1.973 báo cáo tài chính OCR. Repository kết hợp truy hồi bảng, phân tích ngữ
nghĩa tài chính, mô hình ngôn ngữ cục bộ và bộ giải công thức deterministic để
tạo đủ bốn thành phần mà evaluator yêu cầu: đáp án số, tài liệu nguồn, bảng
nguồn và biểu thức Pandas có thể chạy lại.

> Trạng thái: pipeline nghiên cứu/thi đấu có checkpoint và release gate. Kết quả
> official tốt nhất được lưu trong lịch sử dự án là **0,7016 Execution Accuracy
> và 0,7016 Answer Accuracy (VN70)**. Đây là kết quả aggregate trên hệ thống chấm,
> không phải cam kết cho một lần chạy mới.

## Bắt đầu đọc từ đâu?

| Nhu cầu | Tài liệu |
|---|---|
| Hiểu toàn bộ hệ thống để làm slide/báo cáo | [Kiến trúc và pipeline](docs/architecture.md) |
| Clone và chạy từ đầu | [Hướng dẫn tái lập](docs/reproduction.md) |
| Hiểu model, prompting và vai trò của LLM | [Model và cơ chế suy luận](docs/models.md) |
| Xem số liệu benchmark/leaderboard | [Kết quả thực nghiệm](docs/experiments.md) |
| Đọc từng module Python | [README của package](src/road2ai_vifinqa/README.md) |
| Dùng công cụ audit/benchmark/release | [README của tools](tools/README.md) |
| Hiểu test và release gate | [README của tests](tests/README.md) |

Pipeline có thể tái lập cho bài thi **Financial Table Retrieval & Text-to-Pandas**.
Mục tiêu của dự án không chỉ là tạo một số đúng, mà là tạo đồng thời:

- đáp án số;
- tài liệu và bảng nguồn;
- CSV bằng chứng;
- biểu thức Pandas chạy lại được và sinh đúng đáp án.

## Kiến trúc

1. `build_index`: đọc 1.973 báo cáo, bung `rowspan`/`colspan` và lập chỉ mục SQLite.
2. `table_semantics`: phục hồi header nhiều tầng, section, kỳ, đơn vị, dòng tổng/thành phần và loại báo cáo.
3. `build_panel`: gom các chỉ tiêu chuẩn; chấm điểm mọi ô ứng viên rồi mới giải quyết bảng trùng.
4. Các solver chuyên biệt:
   - Easy/direct: truy hồi theo công ty–năm–scope, rank bảng rồi rank dòng/ô;
   - Hard: công thức tài chính deterministic trên panel;
   - Note: truy hồi disclosure chuyên biệt;
   - Template: khóa toán hạng và thực thi phép tính deterministic.
5. `solve`: checkpoint theo câu, ghép ZIP và replay lại toàn bộ Pandas query.
6. `release_audit`: kiểm tra độc lập schema, nguồn, table reference, CSV và thực thi.

Luồng rút gọn:

```text
OCR reports + questions
        │
        ├─ build_index ──> SQLite corpus (document/table/row/cell)
        │                         │
        │                         ├─ Direct/Easy retrieval ──> Qwen3-8B selector
        │                         └─ Note retrieval ─────────> checked expression
        │
        └─ build_panel ──> canonical financial panel
                                  ├─ Hard formulas
                                  └─ Template registry
                                           │
                        SubmissionSolution + evidence CSV + Pandas query
                                           │
                           ZIP validation + repeated replay + release audit
```

Các submission lịch sử không bị ghi đè. Artifact mới nên luôn được dựng song song,
benchmark và audit trước khi promote.

## Cài đặt

```powershell
python -m pip install -e .
$env:PYTHONPATH = (Resolve-Path "src").Path
```

Python 3.11+ được hỗ trợ. Model cục bộ mặc định là Qwen3-8B GGUF tại
`artifacts/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`.

Dataset, model GGUF, SQLite index, panel JSON, run logs và submission ZIP không
được commit vì dung lượng lớn hoặc có thể tái tạo. Sau khi clone, người dùng cần
đặt dữ liệu/model đúng layout trong [hướng dẫn tái lập](docs/reproduction.md).
Không nên chạy full solve trước khi `build_index`, `build_panel` và artifact audit
đều thành công.

## Dựng artifact an toàn

Dựng index/panel mới sang đường dẫn riêng:

```powershell
python -m road2ai_vifinqa.build_index `
  --force `
  --output artifacts/tables_v2.sqlite3 `
  --manifest artifacts/index_v2_manifest.json

python -m road2ai_vifinqa.build_panel `
  --force `
  --index artifacts/tables_v2.sqlite3 `
  --output artifacts/financial_panel_v2.json `
  --manifest artifacts/financial_panel_v2_manifest.json
```

So sánh panel mới với baseline và replay toàn bộ Hard/Template:

```powershell
python tools/benchmark_preprocessing.py `
  --baseline artifacts/financial_panel.json `
  --candidate artifacts/financial_panel_v2.json `
  --report runs/optimization/panel_v2_benchmark.json

python tools/audit_artifacts.py `
  --index artifacts/tables_v2.sqlite3 `
  --panel artifacts/financial_panel_v2.json `
  --integrity `
  --report runs/optimization/artifact_audit.json
```

Không promote nếu có coordinate lỗi, raw/value không replay được, kỳ nguồn lệch năm
hoặc solver regression chưa được source-audit.

## Chạy solver

```powershell
python -m road2ai_vifinqa.solve `
  --iteration 1 `
  --run-dir runs/release_candidate `
  --index artifacts/tables_v2.sqlite3 `
  --panel artifacts/financial_panel_v2.json `
  --ids 1-1012 `
  --fail-fast
```

Checkpoint chứa chữ ký của index và panel. Khi một artifact thay đổi, checkpoint cũ
bị từ chối tự động thay vì âm thầm tái sử dụng kết quả lỗi thời.

## Kiểm thử và phát hành

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m unittest discover -s tests -v
python -m compileall -q src tools tests

python tools/release_audit.py `
  --zip runs/release_candidate/submission.zip `
  --run-dir runs/release_candidate `
  --table-ref-mode one-based `
  --replays 3 `
  --report runs/release_candidate/final_release_audit.json
```

Release gate yêu cầu đủ 1.012 câu và 1.012 CSV; tất cả tài liệu/bảng phải tồn tại;
mọi query phải chạy deterministic nhiều lần và khớp trường `answer`.

## Nguyên tắc tối ưu điểm

- Không dùng leaderboard để dò từng đáp án.
- Không đổi nguồn chỉ vì một số ở bảng khác có vẻ “hợp lý hơn”; bộ Easy khóa bảng
  trước khi sinh câu hỏi nên nguồn tương đương vẫn có thể khác gold.
- Tách rõ lỗi retrieval, table, row, column, unit, period, scope và formula.
- Chỉ merge correction có tọa độ nguồn, công thức và replay; giữ regression suite cho
  các thay đổi đã được leaderboard xác nhận dương.
- Answer Accuracy bằng Execution Accuracy khi query chỉ đọc scalar đã tính sẵn. Muốn
  tăng hai điểm này phải sửa semantic grounding/toán hạng, không phải chỉ sửa cú pháp Pandas.

Gold chi tiết không được công bố, vì vậy kiểm định cục bộ bảo đảm tính hợp lệ và giảm
hồi quy nhưng không thể cam kết trước một mức điểm leaderboard cụ thể.

Kết quả benchmark, quyết định bật/tắt từng tầng retrieval và promotion gate của
artifact v2 được ghi tại [`docs/optimization.md`](docs/optimization.md).

## Phạm vi công khai trên GitHub

Repository commit mã nguồn, cấu hình reranker nhỏ, tests, công cụ và tài liệu.
Các mục sau được chủ động bỏ qua: `data/source`, `external`, `artifacts`, `runs`,
runtime nhị phân của llama.cpp và `submission*.zip`. Cách này giúp clone nhẹ,
không phát tán nhầm dataset/model có license riêng và tránh coi generated output
là source of truth. Mỗi artifact quan trọng đều có manifest/hash hoặc audit report
để kiểm tra khi tái tạo.
