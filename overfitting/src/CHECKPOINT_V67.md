# Checkpoint private V67

Đội: **Overfitting**. Nhánh mã nguồn: `improve_baseline_kien`.

V67 là checkpoint của toàn bộ pipeline ViFinQA hybrid: retrieval, canonical
metrics, financial solvers, kết quả Qwen và các challenger đã audit. Không có
checkpoint trọng số fine-tune riêng. Điểm dưới đây do đội xác nhận từ leaderboard
private; test cục bộ không thay thế đánh giá trên nhãn private.

## Kết quả đã xác nhận

| Metric | V67 |
|---|---:|
| TABLES_F2MACRO | 0.5482 |
| DOCS_F2MACRO | 0.9518 |
| TABLES_PRECISION | 0.3517 |
| TABLES_RECALL | 0.7419 |
| TABLES_MRR5 | 0.5994 |
| DOCS_PRECISION | 0.9504 |
| DOCS_RECALL | 0.9586 |
| DOCS_MRR5 | 0.9664 |
| ANSWER_ACCURACY | **0.5277** |
| EXECUTION_ACCURACY | **0.5277** |

Trên cùng private test, V28 đạt `0.4783`, V31 đạt `0.4921`, V63 đạt `0.5217`.
V67 tăng **0,60 điểm phần trăm** so với V63 và **4,94 điểm phần trăm** so với
V28 private. Điểm `0.4644` của V28 thuộc public test, không dùng làm mốc so sánh
trực tiếp với private.

## Thay đổi và mã nguồn

So với V63, V67 giữ nguyên nền checkpoint và thay 10 đáp án: 9 formula
challenger đã audit và 1 sửa đổi chọn ô dữ liệu nguồn. Các ID thay đổi là
`111, 324, 494, 512, 520, 521, 529, 677, 725, 726`. Không có nhãn đúng từng câu
nên không diễn giải thành 10 câu đúng thêm.

- `scripts/24_build_private_formula_challengers.py`: ghép 9 challenger từ
  artifact V61 vào V63 để tạo V64; script cũng tạo nhánh thử nghiệm V65.
- `scripts/25_build_private_v66_exact_repairs.py`: tên script giữ từ phiên thử
  nghiệm V66, nhưng đầu ra thực tế là **V67**; sửa evidence/query câu 324 trên
  nền V64. Query tính tổng trực tiếp từ CSV khi thực thi.
- `scripts/16..22` và các module consensus/schema/exact solver lưu quy trình
  tạo, kiểm tra và ghép challenger private trước đó.
- `kaggle/private-v29..35-*.ipynb` và `kaggle/kaggle_codegen.py` lưu cấu hình
  inference private, gồm Qwen2.5-Coder-14B-Instruct N=5. Không có nghĩa mọi
  output trong các lượt này đều được đưa vào V67.

Các sửa đổi theo ID là artifact audit của bộ câu hỏi cuộc thi, không phải bằng
chứng solver đã khái quát được cho mọi câu hỏi mới.

## Artifact bài nộp

```text
artifacts/codegen_private_v67_formula_core9_exact1.jsonl
artifacts/submission_private_v67_formula_core9_exact1/submission.zip
```

SHA-256 của ZIP đã đối chiếu tại máy:

```text
16a015429c533f85279515cc0733a6204f8e9342ce62f7542ce6926c7b1ad750
```

Gói chứa **1.012 entry** trong `results.json` và **2.083 CSV**. Đây là số lượng
trong bài nộp, không phải số câu được chấm riêng trên private test.
`relevant_tables` phải dùng số dòng **1-based của dòng bắt đầu `<table>`** trong
OCR, không phải thứ tự bảng. Kết quả query phải tính từ evidence tại thời điểm
thực thi, không đọc cột đáp án lưu sẵn.

## Phạm vi tái hiện

Mã nguồn, tests, notebook và tài liệu được đưa lên Git. Dữ liệu BTC, normalized
store, kết quả inference và ZIP bài nộp nằm ngoài Git theo `.gitignore`.
Tài liệu này không cung cấp một đường link artifact công khai mới.

Sau khi cài môi trường theo [README](README.md), muốn dựng lại bước ghép V67
cần khôi phục hai đầu vào đóng băng tại đúng đường dẫn:

```text
artifacts/codegen_private_v63_final_audited_from_v60.jsonl
artifacts/codegen_private_v61_balanced43_exact_from_v31.jsonl
```

Chạy từ thư mục gốc repo:

```bash
python scripts/24_build_private_formula_challengers.py
python scripts/25_build_private_v66_exact_repairs.py
```

Đóng gói lại cần normalized store và retrieval tương ứng, không chỉ hai
JSONL phía trên. CLI đóng gói là `scripts/05_build_submission.py`, dùng
`--pos-mode line`. Khi bàn giao cần chia sẻ riêng các artifact đầu vào và
đối chiếu nội dung với ZIP đã xác nhận; không khẳng định một lần chạy mới
sẽ cho ZIP có byte/hash giống hệt do metadata đóng gói.

Kiểm tra mã nguồn tại thời điểm chuẩn bị push V67:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests -q
```

Kết quả: **466 passed**. Bộ test kiểm tra hành vi mã nguồn, không xác nhận lại
Execution Accuracy `0.5277` hay toàn bộ tính đúng của evidence trong bài nộp.

## Demo Day

- [Slide PowerPoint V67](deliverables/demo_day/output/Overfitting_DemoDay_2026_V67.pptx)
- [Slide PDF V67](deliverables/demo_day/output/Overfitting_DemoDay_2026_V67.pdf)
- [Kịch bản 5 phút và hỏi đáp](deliverables/demo_day/output/KICH_BAN_5_PHUT_VA_HOI_DAP.md)

Thành viên: Lê Quang Khải, Nguyễn Trung Kiên, Trần Tuấn Huy.
