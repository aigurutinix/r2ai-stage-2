# Tài liệu mô tả dữ liệu — đội vilamiu

## 1. Nguồn dữ liệu

| nguồn | mô tả | ghi chú |
|---|---|---|
| **Kho BCTC của Ban Tổ chức** | 100 công ty niêm yết, 10 năm, 1.965 báo cáo | tải từ HuggingFace Dataset ViFinQA do BTC công bố |
| **Bộ câu hỏi kiểm thử** | 1.012 câu, mỗi câu `{id, question}` | BTC cung cấp |
| **`data/code_stock.csv`** | ánh xạ mã chứng khoán → tên công ty | BTC cung cấp |
| **`data/ticker_aliases.csv`** | 129 bí danh tên công ty | **đội tự soạn**, xem mục 4 |

Ngoài `ticker_aliases.csv`, đội **không sử dụng bất kỳ dữ liệu ngoài nào**. Không dùng dữ
liệu thị trường, không dùng API tài chính, không thu thập thêm báo cáo.

## 2. Cấu trúc và định dạng

```
data/
├── official_corpus/
│   └── <MÃ_CK>/<NĂM>/<tên_báo_cáo>/
│       ├── <tên_báo_cáo>_extracted.txt
│       └── <tên_báo_cáo>_extracted_tables/
│           ├── table_0.csv … table_N.csv
├── questions/questions.jsonl
├── code_stock.csv
└── ticker_aliases.csv
```

**Quy mô:** 100 công ty · 1.965 báo cáo · **146.246 bảng CSV**.

### 2.1 Tệp `.txt`

Văn bản của báo cáo, phân trang bằng `===== PAGE n =====`. Mỗi bảng được thay bằng một neo:

```
[table_7](AAA_financial_statements_2015_consolidated_extracted_tables/table_7.csv)
```

**Số dòng của neo trong tệp `.txt` chính là "vị trí bảng trong báo cáo"** mà trường
`relevant_tables` yêu cầu (`<mã_báo_cáo>|<số_dòng>`). Đây là điểm đã xác nhận với BTC và là
quy ước bài nộp của đội tuân theo.

### 2.2 Tệp `.csv`

Không có dòng tiêu đề chuẩn; dòng đầu là tiêu đề cột **như in trong báo cáo**, các dòng sau
là dữ liệu. Đọc bằng `pandas.read_csv(path, encoding="utf-8-sig", dtype=str,
keep_default_na=False, index_col=None)` — đúng như sandbox của BTC, nên `df.iloc[0]` là
**dòng dữ liệu đầu tiên**, không phải tiêu đề.

Ba đặc điểm định dạng phải xử lý bằng luật:

| hiện tượng | ví dụ | ý nghĩa |
|---|---|---|
| dấu chấm ngăn nghìn | `2.720.958` | 2.720.958 (không phải 2,72) |
| ngoặc đơn | `(187.902)` | số âm |
| đơn vị nằm ở tiêu đề cột | `Số cuối nămTriệu đồng` | mọi số trong cột tính bằng triệu đồng |

Đơn vị được giải theo thứ tự: tiêu đề cột → tiêu đề bảng → dòng "Đơn vị tính" của tài liệu.

## 3. Hướng dẫn truy cập và sử dụng

```bash
# 1. Tải kho dữ liệu (một trong hai cách)
huggingface-cli download <ViFinQA-dataset-id> --repo-type dataset --local-dir ./data
# hoac giai nen goi chia se cua doi (muc 5) vao thu muc data/

# 2. Kiem tra da day du
python -c "import pathlib; c=pathlib.Path('data/official_corpus'); \
print('cong ty:', len(list(c.iterdir()))); \
print('bang CSV:', sum(1 for _ in c.glob('*/*/*/*_extracted_tables/table_*.csv')))"
# ky vong: cong ty 100, bang CSV 146246

# 3. Doc mot bang dung quy uoc cua BTC
python -c "import pandas as pd; \
d=pd.read_csv('data/official_corpus/AAA/2015/AAA_financial_statements_2015_consolidated/\
AAA_financial_statements_2015_consolidated_extracted_tables/table_4.csv', \
encoding='utf-8-sig', dtype=str, keep_default_na=False, index_col=None); print(d.head())"
```

## 4. Dữ liệu do đội bổ sung: `data/ticker_aliases.csv`

129 dòng, định dạng `alias,code`:

```csv
alias,code
Hòa Phát,HPG
Vinamilk,VNM
Eximbank,EIB
```

**Nguồn:** soạn thủ công từ chính tên công ty xuất hiện trong bộ câu hỏi của BTC và trong
`code_stock.csv`. Không lấy từ nguồn bên ngoài.

**Lý do cần:** sổ đăng ký ghi `CTCP Tập đoàn Hòa Phát`, `CTCP Sữa Việt Nam`,
`Ngân hàng TMCP Xuất nhập khẩu Việt Nam`; còn đề hỏi `Hoà Phát`, `Vinamilk`, `Eximbank`. Nếu
khớp nguyên văn thì **35 câu không nhận ra công ty nào** và khoảng **20 câu nhận thiếu công
ty** trong nhóm so sánh.

**Cách khớp:** so trên dạng đã bỏ dấu, đã mở rộng viết tắt (`CTCP`→`công ty cổ phần`,
`TMCP`→`thương mại cổ phần`), **khớp theo biên từ**. Biên từ là bắt buộc: khớp chuỗi con làm
`No Va` trúng trong "nợ vay" và `An Bình` trúng trong "toàn bình quân". Các bí danh quá ngắn
đã bị loại bỏ hoặc thay bằng tên dài (`Ngân hàng An Bình`, `Bất động sản Phát Đạt`).

## 5. Đường dẫn chia sẻ

> **https://drive.google.com/drive/folders/1K_guQUhY6CHHN572iXcRFmyfpDiJmgm9**
>
> Thư mục `AI Guru 2026 Stage 2 - vilamiu` trên Google Drive.

Gói chia sẻ gồm:

| thư mục | nội dung |
|---|---|
| `data/` | kho BCTC, bộ câu hỏi, `code_stock.csv`, `ticker_aliases.csv` |
| `artifacts/fresh/` | kết quả trung gian (prompt, phản hồi model, chương trình đã thi hành) |
| `submissions/legal2.zip` | bài nộp cuối |

## 6. Dữ liệu trong bài nộp

`submissions/legal2.zip` chứa **4.639 bảng CSV** trong thư mục `data/`. Mỗi tệp là bản sao
**nguyên vẹn** của một bảng trong kho BTC, đặt tên `<mã_báo_cáo>_table_<N>.csv` nên truy vết
được về `data/official_corpus/<MÃ>/<NĂM>/<báo_cáo>/<báo_cáo>_extracted_tables/table_<N>.csv`.

BTC cho phép chỉ giữ hàng/cột cần thiết; đội **giữ nguyên toàn bộ bảng** để người chấm đối
chiếu trực tiếp với nguồn mà không phải tin vào một bước cắt gọt trung gian.
