# AI_Guru — ViFinQA · Financial Table Retrieval & Text-to-Pandas

Hệ thống hỏi–đáp trên báo cáo tài chính tiếng Việt. Nhận câu hỏi bằng ngôn ngữ tự nhiên,
tìm đúng bảng trong **146.246 bảng / 1.973 báo cáo**, sinh `pandas_query` **đọc thật từ dữ
liệu**, và dẫn nguồn tới từng ô.

> **Cuộc thi:** ROAD TO AI 2026 – Stage 2 · **Đội:** AI_Guru
> **Bản nộp tham chiếu:** `submissions/v58_va74_sau_continue.zip`

---

## 0. Bộ tài liệu nộp — bản đồ nhanh

| BTC yêu cầu | File trong repo |
|---|---|
| Thuyết minh sản phẩm | [`THUYETMINH.pdf`](THUYETMINH.pdf) · [`THUYETMINH.md`](THUYETMINH.md) |
| **Mô tả dữ liệu** | [`DATA.md`](DATA.md) |
| **Mô hình + checkpoint** | [`MODEL.md`](MODEL.md) |
| **Mã nguồn + phụ thuộc** | repo này · [`requirements.txt`](requirements.txt) |
| **Bài nộp** | `submissions/` · kiểm tuân thủ: `scripts/audit_query_compliance.py` |
| **Hướng dẫn tái lập** | file này (mục 2–4) |

---

## 1. Yêu cầu môi trường

| | |
|---|---|
| **Python** | **3.12.2** (đã kiểm; 3.11+ nhiều khả năng chạy được) |
| **Hệ điều hành** | Windows 11 (đã chạy) · Linux/macOS chạy được, xem lưu ý mục 5 |
| **RAM** | **16 GB** — bước dựng bảng đọc corpus 370 MB và giữ 146.246 bảng trong bộ nhớ |
| **Đĩa trống** | **~6 GB** (corpus 370 MB + parquet ~80 MB + CSV materialize khi đóng gói) |
| **GPU** | **KHÔNG cần** cho đường ống chính. Hai bước tăng cường chạy trên **Kaggle free 2×T4** — xem [`MODEL.md`](MODEL.md) |

```bash
git clone <repo>              # hoặc giải nén bản nộp
cd AI_Guru
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

---

## 2. Chạy lại từ đầu — 5 lệnh

Mỗi lệnh in ra số liệu kiểm chứng; nếu con số lệch nhiều so với cột "kỳ vọng" thì dừng lại
và xem mục 5.

```bash
# (1) Tải corpus gốc từ HuggingFace  ─────────────────  ~3-8 phút, 370 MB
python scripts/download_corpus.py

# (2) Dựng bảng: HTML → lưới → parquet  ──────────────  ~45 giây
python scripts/build_tables.py --workers 6

# (3) Bóc tách thực thể từ câu hỏi  ──────────────────  ~10 giây
python scripts/map_questions.py

# (4) Truy hồi tài liệu + bảng ứng viên  ─────────────  ~8 phút
python scripts/retrieve.py

# (5) Sinh bài nộp  ─────────────────────────────────  ~6 phút
python scripts/make_submission.py --selftest --use-rowname \
       --cohort-llm cohort_plan_merged2.parquet --cell-pick \
       --out submissions/tai_lap.zip
```

| bước | kỳ vọng in ra |
|---|---|
| (2) | `146.246 bảng` · `111.070 đủ điều kiện` |
| (3) | `1.011/1.012` câu nhận diện được công ty |
| (4) | `~168` ứng viên trung bình mỗi câu |
| (5) | `Kiểm tra hợp lệ: ĐẠT` · `lệch so với answer: 0` |

⚠️ Bước (5) cần hai tạo tác từ GPU: `cohort_plan_merged2.parquet` và `cell_pick.parquet`.
**Cả hai đã kèm sẵn** trong `data/processed/`. Muốn dựng lại chúng từ đầu, xem
[`MODEL.md`](MODEL.md). **Bỏ hai cờ đó thì đường ống vẫn chạy** và vẫn cho bài nộp hợp lệ,
chỉ kém ~259 câu:

```bash
python scripts/make_submission.py --selftest --use-rowname --out submissions/khong_gpu.zip
```

---

## 3. Kiểm chất lượng — không tốn lượt nộp

```bash
python scripts/audit_query_compliance.py --zip submissions/tai_lap.zip   # ⭐ tuân thủ BTC
python scripts/check_generality.py                                        # cổng chống overfitting
python scripts/check_invariants.py  --zip submissions/tai_lap.zip         # bộ dò lỗi 1
python scripts/check_consistency.py --zip submissions/tai_lap.zip         # bộ dò lỗi 2
python scripts/verify_answers.py    --zip submissions/tai_lap.zip         # bộ dò lỗi 3
python scripts/cite.py              --zip submissions/tai_lap.zip         # xuất trích dẫn
```

**`audit_query_compliance.py` là bộ kiểm quan trọng nhất.** Nó phân tích **cú pháp** từng
`pandas_query` và bắt đúng ba lớp BTC cấm:

| BTC nêu là không hợp lệ | bộ kiểm bắt được |
|---|---|
| `result = 2.06` | ✅ |
| `result = 1.03 * 2` | ✅ |
| `result = df["answer"].iloc[0]` | ✅ |
| *(thêm)* hằng số truyền qua biến trung gian | ✅ |

Trên bản nộp hiện tại: **1012/1012 câu HỢP LỆ**.

---

## 4. Cấu hình

Không có file cấu hình ngoài — mọi tham số nằm trong mã nguồn và **cờ dòng lệnh**.

### Cờ PHẢI BẬT khi sinh bài nộp

| cờ | vai trò |
|---|---|
| `--selftest` | chạy thử mọi `pandas_query` trong sandbox và so với `answer`. **Bắt buộc** — đây là cổng duy nhất bắt được lớp lỗi "đáp án và truy vấn nói hai chuyện khác nhau" |
| `--use-rowname` | dùng nhãn dòng do LLM nêu (934 câu) |
| `--cohort-llm <file>` | kế hoạch peer-group do LLM điền |
| `--cell-pick` | ô do LLM chọn trong 5 ứng viên (Vá 44) |

### Cờ ĐÃ ĐO LÀ ÂM — mặc định TẮT, đừng bật

Repo giữ lại 13 cơ chế đã thử và đo ra âm hoặc bằng 0, để tái lập được kết luận. Bảng đầy
đủ kèm số đo ở [`RUNBOOK_PRIVATE.md`](RUNBOOK_PRIVATE.md) mục 3. Ví dụ:

| cờ | đo được |
|---|---|
| `--evidence-any` | `TABLES_MRR5` −0,0284 |
| `--khac-gate` | `EXECUTION` +0,0000 · `TABLES_MRR5` −0,0020 |
| `--year-plan` | `EXECUTION` −0,0020 |

### Đường dẫn gốc

Vài script dùng đường dẫn tuyệt đối `d:/AI_Guru`. Nếu đặt repo ở nơi khác, sửa hằng `ROOT`
ở đầu các file trong `scripts/`.

---

## 5. Lưu ý khi chạy trên môi trường khác

* **Console Windows dùng cp1252** — mọi script đã có
  `sys.stdout.reconfigure(encoding="utf-8")`. Nếu thấy `UnicodeEncodeError`, đặt
  `PYTHONIOENCODING=utf-8`.
* **Không ghi 146k file CSV ra đĩa** — rất chậm trên NTFS. Đường ống chỉ materialize CSV
  cho những bảng thật sự xuất hiện trong bài nộp (~2.300 file).
* **`make_submission.py` đọc `retrieved.parquet` dựng sẵn.** Sửa `questions.py` mà không
  chạy lại `retrieve.py` thì **0 tài liệu đổi**. Đây là cái bẫy đã trả giá một lần.

---

## 6. Cấu trúc thư mục

```
src/vifin/            thư viện lõi
  tables.py           parser HTML → lưới, phân loại bảng, đơn vị, số kiểu Việt
  questions.py        bóc tách mã CK / năm / phạm vi / đơn vị từ câu hỏi
  submission.py       chấm ô, dựng pandas_query, đóng gói ZIP
  cohort.py           lập kế hoạch peer-group và trục năm
  catalog_solver.py   nhận diện chỉ tiêu theo danh mục BTC

scripts/              ~60 script: dựng dữ liệu · truy hồi · sinh bài nộp · KIỂM
notebooks/            notebook Kaggle (2 cái dùng cho bản nộp)
data/raw/             corpus gốc (tải bằng script, không đưa vào repo)
data/processed/       parquet trung gian
submissions/          các bản nộp ZIP
scoring/              kết quả chấm BTC trả về
changes/              98 tài liệu thay đổi — mỗi cái có dự đoán đặt trước + kết quả đo
```

### Tài liệu kỹ thuật

| file | nội dung |
|---|---|
| [`THUYETMINH.md`](THUYETMINH.md) | **thuyết minh sản phẩm** |
| [`DATA.md`](DATA.md) | mô tả dữ liệu |
| [`MODEL.md`](MODEL.md) | mô hình và checkpoint |
| [`CLAUDE.md`](CLAUDE.md) | quy ước bất biến · mọi kết luận đã đo · mọi hướng đã đóng kèm số |
| [`TRANGTHAI.md`](TRANGTHAI.md) | trạng thái kỹ thuật hiện tại |
| [`GOAL.md`](GOAL.md) | chỉ số qua tất cả bản nộp |
| [`RUNBOOK_PRIVATE.md`](RUNBOOK_PRIVATE.md) | quy trình vòng private + bảng cờ đã đo âm |
| [`SOURCES.md`](SOURCES.md) | khai báo nguồn dữ liệu ngoài |

---

## 7. Kết quả

| chỉ số | giá trị | hạng |
|---|---|---|
| **DOCS_PRECISION** | **0,9707** | **2** |
| **DOCS_MRR5** | **0,9783** | **3** |
| TABLES_RECALL · TABLES_MRR5 · TABLES_F2 | 0,6760 · 0,6543 · 0,5588 | 5 |
| DOCS_F2 | 0,9653 | 6 |
| DOCS_RECALL | 0,9669 | 8 |
| EXECUTION_ACCURACY | 0,3696 | 8 |
| ANSWER_ACCURACY | 0,3715 | 9 |
| TABLES_PRECISION | 0,3585 | 9 |

**5/10 chỉ số top-5 · 2/10 top-3 · 1012/1012 `pandas_query` hợp lệ · 100% câu có trích dẫn.**
