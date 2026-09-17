# Tài liệu thuyết minh sản phẩm - Bài nộp

## 1. Thông tin bài nộp

| Thuộc tính | Giá trị |
|---|---|
| Tên checkpoint | `ViFinQA V28 - exact direct-growth audited3` |
| Điểm Answer Accuracy | **0.4644** |
| Điểm Execution Accuracy | **0.4644** |
| File kết quả | `results.json` |
| File nộp | `artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip` |
| SHA-256 | `ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e` |
| Git checkpoint | `improve_baseline_kien` tại commit `1d52e9d` |

Các metric leaderboard của đúng file ZIP trên:

| Metric | Điểm |
|---|---:|
| `TABLES_F2MACRO` | 0.5530 |
| `DOCS_F2MACRO` | 0.9420 |
| `TABLES_PRECISION` | 0.3621 |
| `TABLES_RECALL` | 0.7480 |
| `TABLES_MRR5` | 0.6181 |
| `DOCS_PRECISION` | 0.9494 |
| `DOCS_RECALL` | 0.9457 |
| `DOCS_MRR5` | 0.9743 |
| `ANSWER_ACCURACY` | **0.4644** |
| `EXECUTION_ACCURACY` | **0.4644** |

## 2. Cấu trúc file nộp

Bài nộp là một file ZIP có cấu trúc:

```text
submission.zip
├── results.json
└── data/
    ├── <report_id>_table_<line_no>.csv
    └── ...
```

`results.json` chứa đủ 1.012 câu hỏi. Mỗi phần tử có các trường:

```json
{
  "id": 586,
  "question": "...",
  "answer": 335.82,
  "relevant_docs": ["<report_id>"],
  "relevant_tables": ["<report_id>|<line_no>"],
  "evidence": [
    {
      "variable": "df1",
      "csv_path": "data/<source_table>.csv"
    }
  ],
  "pandas_query": "<biểu thức được thực thi trên các DataFrame evidence>"
}
```

Trường `answer` là output được yêu cầu bởi schema nộp bài. Giá trị này được
kiểm tra phải bằng kết quả thực thi lại `pandas_query`; đáp án không được lưu
trong các CSV evidence.

## 3. Nguồn gốc và khả năng truy vết của CSV

Toàn bộ CSV trong `data/` được tạo từ các bảng OCR thuộc bộ dữ liệu do Ban Tổ
chức cung cấp. Pipeline không bổ sung dữ liệu tài chính bên ngoài.

Chuỗi truy vết của mỗi bảng:

```text
tệp OCR của BTC
  -> report_id
  -> dòng 1-based bắt đầu thẻ <table>
  -> bảng nguồn được parser lưu trong store
  -> CSV chuẩn hóa trong data/
  -> evidence.variable
  -> pandas_query
  -> answer
```

Quy ước quan trọng:

- `relevant_tables` dùng **line number 1-based của dòng bắt đầu `<table>`**,
  không dùng số thứ tự bảng.
- Tên CSV có dạng `<report_id>_table_<line_no>.csv`.
- `csv_path` là đường dẫn tương đối và luôn bắt đầu bằng `data/`.
- CSV có thể chứa toàn bộ bảng nguồn đã chuẩn hóa hoặc chỉ phần cần thiết.
- Mọi giá trị số trong CSV phải parse được từ ô thuộc bảng nguồn.
- CSV không có cột chứa đáp án cuối cùng.

Schema CSV chuẩn hóa của hệ thống:

```text
row,label,code,col,col_name,value,unit_scale
```

| Cột | Ý nghĩa |
|---|---|
| `row` | Chỉ số hàng trong bảng nguồn |
| `label` | Nhãn chỉ tiêu tài chính từ bảng nguồn |
| `code` | Mã số chỉ tiêu nếu có |
| `col` | Chỉ số cột trong bảng nguồn |
| `col_name` | Tên cột hoặc kỳ báo cáo |
| `value` | Giá trị số parse trực tiếp từ ô nguồn |
| `unit_scale` | Hệ số đổi đơn vị bảng sang VND |

## 4. Minh chứng thực tế từ bài nộp

Ví dụ dưới đây lấy trực tiếp từ entry ID `586` trong submission đã đạt
`EXECUTION_ACCURACY=0.4644`.

### 4.1. Câu hỏi và nguồn được khai báo

```text
Tốc độ tăng trưởng tiền và các khoản tương đương tiền của Tổng Công ty
Cảng Hàng không Việt Nam từ cuối năm 2021 đến cuối năm 2022 là bao nhiêu
phần trăm?
```

Hai bảng được query thực sự sử dụng:

| Biến | CSV evidence | Nguồn |
|---|---|---|
| `df4` | `data/ACV_financial_statements_2022_consolidated_table_212.csv` | `ACV_financial_statements_2022_consolidated|212` |
| `df7` | `data/ACV_financial_statements_2021_consolidated_table_219.csv` | `ACV_financial_statements_2021_consolidated|219` |

Hai số liệu nguồn trong các CSV:

```csv
# df4: ACV_financial_statements_2022_consolidated_table_212.csv
row,label,code,col,col_name,value,unit_scale
2,I. Tiền và các khoản tương đương tiền,110,3,Số cuối năm VND,2496515921711.0,1.0
```

```csv
# df7: ACV_financial_statements_2021_consolidated_table_219.csv
row,label,code,col,col_name,value,unit_scale
2,I. Tiền và các khoản tương đương tiền,110,3,Số cuối năm VND,572833249811.0,1.0
```

Các dòng bắt đầu bằng `#` ở trên chỉ là chú thích trong tài liệu; chúng không
có trong file CSV thật.

### 4.2. Pandas query thực tế

```python
round(
    (
        float(
            df4.loc[
                (df4["row"] == 2)
                & df4["label"].str.strip().eq(
                    "I. Tiền và các khoản tương đương tiền"
                )
                & (df4["col"] == 3),
                "value",
            ].iloc[0]
        )
        - float(
            df7.loc[
                (df7["row"] == 2)
                & df7["label"].str.strip().eq(
                    "I. Tiền và các khoản tương đương tiền"
                )
                & (df7["col"] == 3),
                "value",
            ].iloc[0]
        )
    )
    / abs(
        float(
            df7.loc[
                (df7["row"] == 2)
                & df7["label"].str.strip().eq(
                    "I. Tiền và các khoản tương đương tiền"
                )
                & (df7["col"] == 3),
                "value",
            ].iloc[0]
        )
    )
    * 100,
    2,
)
```

Phép tính được thực hiện tại thời điểm chạy:

```text
(2.496.515.921.711 - 572.833.249.811)
------------------------------------------------ x 100 = 335,82%
                 572.833.249.811
```

Kết quả `335.82` được suy ra trực tiếp từ `df4` và `df7`. CSV không chứa kết
quả này và query không gán cứng đáp án.

## 5. Kiểm soát chống gán cứng đáp án

Code hiện tại áp dụng các kiểm soát:

1. `vifinqa/retrieval/serialize.py` chỉ sinh CSV từ grid của bảng nguồn.
2. `vifinqa/codegen/semantic.py` phân tích AST của query và yêu cầu biểu thức
   cuối cùng phụ thuộc vào ít nhất một DataFrame `dfN`.
3. Các DataFrame được query tham chiếu phải xuất hiện trong `evidence`.
4. `vifinqa/codegen/executor.py` thực thi query trên CSV đã parse lại.
5. Submission builder replay query và so kết quả với trường `answer`.
6. Chỉ CSV được query dùng mới được khai báo trong `evidence` và đóng gói.

Các query sau không hợp lệ:

```python
2.06
```

```python
1.03 * 2
```

```python
float(df1["answer"].iloc[0])
```

```python
float(df1["value"].iloc[0]) * 0 + 2.06
```

Ba trường hợp đầu trả về hoặc đọc một đáp án đã lưu sẵn. Trường hợp cuối cố
tạo tham chiếu DataFrame giả nhưng kết quả vẫn là hằng số. Tất cả đều vi phạm
quy định và không được sử dụng cho câu trả lời hợp lệ.

Các hằng số mang ý nghĩa công thức được phép sử dụng, ví dụ:

- `100` để chuyển tỷ lệ sang phần trăm;
- `365` trong công thức vòng quay;
- `1e3`, `1e6`, `1e9` để quy đổi đơn vị;
- `2` khi tính trung bình của đúng hai số dư đã đọc từ CSV.

## 6. Kết quả kiểm tra toàn bộ submission

| Kiểm tra | Kết quả |
|---|---:|
| Số entry trong `results.json` | 1.012 |
| Số CSV evidence trong ZIP | 2.092 |
| Entry có ít nhất một evidence | 1.012 |
| `csv_path` bị thiếu trong ZIP | 0 |
| CSV sai schema chuẩn hóa | 0 |
| `relevant_tables` sai định dạng `<report_id>\|<line_no>` | 0 |
| Query tham chiếu DataFrame không có trong evidence | 0 |
| Query phụ thuộc trực tiếp vào DataFrame | 1.001 |
| Query chưa giải, còn placeholder `0.0` | 11 |
| Query compile/replay khi build | 1.012/1.012 |
| Kiểm tra toàn vẹn ZIP | Pass |

Mười một ID chưa giải:

```text
424, 425, 426, 464, 497, 510, 538, 805, 907, 950, 995
```

Các entry này có query `0.0` và được đánh dấu là câu chưa giải. Chúng không
được mô tả là query hợp lệ và không được kỳ vọng tính điểm. Đây là placeholder
thất bại rõ ràng, không phải đáp án được mã hóa hoặc lưu trong CSV.

`1.001` entry còn lại đều tham chiếu DataFrame evidence. Mọi expression trong
file đã compile và replay đúng trường `answer` khi build submission.

## 7. Hướng dẫn tái tạo và kiểm tra

### 7.1. Build lại submission từ checkpoint frozen

```bash
python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl \
  --codegen artifacts/codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl \
  --out-dir artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010
```

### 7.2. Kiểm tra cấu trúc và tính toàn vẹn

```bash
unzip -t artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

Kết quả mong đợi:

```text
No errors detected in compressed data of submission.zip.
```

### 7.3. Kiểm tra SHA-256

Linux:

```bash
sha256sum artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

macOS:

```bash
shasum -a 256 artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

Hash hợp lệ:

```text
ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e
```

## 8. Kết luận tuân thủ

- CSV evidence được trích từ dữ liệu do Ban Tổ chức cung cấp.
- Mỗi CSV có thể truy vết về `report_id` và `line_no` của bảng OCR nguồn.
- Submission không lưu đáp án cuối cùng trong CSV.
- Các query thành công tính toán đáp án tại thời điểm thực thi từ DataFrame.
- Query, evidence, answer và nguồn bảng được đóng gói cùng nhau để BTC replay.
- Các câu chưa giải được khai báo minh bạch và không dùng kỹ thuật gán cứng.

Ví dụ dữ liệu minh họa và checklist chi tiết được trình bày thêm tại
`DATA_CSV_PANDAS_QUERY_0.4644.md`.
