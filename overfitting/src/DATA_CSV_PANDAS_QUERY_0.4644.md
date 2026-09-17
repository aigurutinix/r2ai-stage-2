# 4. Dữ liệu CSV và Pandas query

## 4.1. Quy định về dữ liệu CSV

Các bảng CSV được dùng để trả lời câu hỏi phải là tập con được trích xuất từ
dữ liệu do Ban Tổ chức cung cấp và phải truy vết được về nguồn dữ liệu.

Mỗi bảng CSV có thể chỉ bao gồm các hàng, cột hoặc ô dữ liệu cần thiết, không
nhất thiết phải chứa toàn bộ bảng nguồn. Tuy nhiên, dữ liệu trong CSV không
được tự tạo, sửa số liệu hoặc chứa sẵn đáp án cuối cùng.

Mỗi CSV trong submission của hệ thống được liên kết với:

- `report_id`: mã báo cáo tài chính gốc;
- `table_pos`: thứ tự nội bộ của bảng trong parser;
- `line_no`: số dòng **1-based** nơi thẻ `<table>` bắt đầu trong file OCR do
  Ban Tổ chức cung cấp;
- `variable`: tên DataFrame được sử dụng trong `pandas_query`;
- `csv_path`: đường dẫn tương đối tới CSV trong thư mục `data/` của submission.

Giá trị `relevant_tables` được xuất theo định dạng:

```text
<report_id>|<line_no>
```

Trong đó `<line_no>` là số dòng bắt đầu của bảng, không phải số thứ tự bảng.

## 4.2. Ví dụ dữ liệu nguồn và CSV hợp lệ

### Dữ liệu nguồn do Ban Tổ chức cung cấp

| Chỉ tiêu | 2021 | 2022 | 2023 | 2024 |
|---|---:|---:|---:|---:|
| Doanh thu thuần | 800 | 1.000 | 1.200 | 1.400 |
| Lợi nhuận sau thuế | 60 | 90 | 120 | 140 |
| Tổng tài sản | 2.000 | 2.200 | 2.500 | 2.800 |
| Nợ phải trả | 900 | 1.000 | 1.100 | 1.200 |

### Một bảng CSV hợp lệ chỉ chứa phần dữ liệu cần thiết

| Chỉ tiêu | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| Doanh thu thuần | 1.000 | 1.200 | 1.400 |
| Lợi nhuận sau thuế | 90 | 120 | 140 |

Bảng CSV thứ hai hợp lệ vì mọi giá trị đều được trích từ bảng nguồn và chỉ loại
bỏ những hàng/cột không cần dùng. Bảng không chứa đáp án được tính sẵn.

## 4.3. Dạng CSV thực tế của hệ thống

Pipeline checkpoint `0.4644` chuẩn hóa bảng nguồn sang dạng dài với schema:

```text
row,label,code,col,col_name,value,unit_scale
```

Ví dụ minh họa:

```csv
row,label,code,col,col_name,value,unit_scale
1,Doanh thu thuần,10,2,2022,1000,1000000
1,Doanh thu thuần,10,3,2023,1200,1000000
1,Doanh thu thuần,10,4,2024,1400,1000000
2,Lợi nhuận sau thuế,60,2,2022,90,1000000
2,Lợi nhuận sau thuế,60,3,2023,120,1000000
2,Lợi nhuận sau thuế,60,4,2024,140,1000000
```

Ý nghĩa các cột:

| Cột | Ý nghĩa |
|---|---|
| `row` | Chỉ số hàng trong bảng nguồn |
| `label` | Nhãn chỉ tiêu tài chính lấy từ bảng nguồn |
| `code` | Mã số chỉ tiêu nếu bảng nguồn có cung cấp |
| `col` | Chỉ số cột trong bảng nguồn |
| `col_name` | Tên cột hoặc kỳ báo cáo |
| `value` | Giá trị số được parse trực tiếp từ ô nguồn |
| `unit_scale` | Hệ số quy đổi đơn vị của bảng nguồn sang VND |

CSV không có cột `answer`, không chứa kết quả cuối cùng và không thay đổi giá
trị kinh tế của ô nguồn. Việc chuyển sang dạng dài chỉ chuẩn hóa cấu trúc để
Pandas truy cập ổn định.

## 4.4. Quy định về Pandas query

Kết quả của mỗi `pandas_query` phải được tính trực tiếp từ dữ liệu trong các
bảng CSV tại thời điểm thực thi. Không được gán cứng, mã hóa hoặc lưu sẵn đáp
án dưới bất kỳ hình thức nào.

Một query hợp lệ phải đáp ứng đồng thời:

1. Đọc ít nhất một DataFrame được khai báo trong `evidence`.
2. Chọn đúng hàng, cột hoặc ô dữ liệu từ CSV.
3. Thực hiện phép tính từ các giá trị vừa đọc.
4. Trả về một số hữu hạn theo đúng đơn vị câu hỏi.
5. Có thể thực thi lại độc lập bằng các CSV được đóng gói trong submission.

Các hằng số mang ý nghĩa công thức được phép sử dụng, ví dụ `100` để đổi tỷ lệ
sang phần trăm, `365` trong vòng quay hoặc `1e6` để quy đổi đơn vị. Hằng số
không được là đáp án cuối cùng được đưa vào để né việc đọc dữ liệu.

### Ví dụ query hợp lệ

Với câu hỏi: “Tốc độ tăng trưởng doanh thu thuần năm 2024 so với năm 2022 là
bao nhiêu phần trăm?”, một query hợp lệ trên bảng minh họa là:

```python
round(
    (
        float(df1.loc[df1["Chỉ tiêu"].eq("Doanh thu thuần"), "2024"].iloc[0])
        - float(df1.loc[df1["Chỉ tiêu"].eq("Doanh thu thuần"), "2022"].iloc[0])
    )
    / abs(float(df1.loc[df1["Chỉ tiêu"].eq("Doanh thu thuần"), "2022"].iloc[0]))
    * 100,
    2,
)
```

Kết quả `40.0` được tạo tại thời điểm chạy từ hai ô `2022=1.000` và
`2024=1.400`, không được lưu sẵn trong CSV hoặc query.

Query thực tế trong submission dùng schema dài và lọc theo tổ hợp
`row + label + col` để giữ đúng danh tính ô nguồn. Ví dụ rút gọn:

```python
round(
    (
        float(df1.loc[(df1["row"] == 1) & (df1["col"] == 4), "value"].iloc[0])
        - float(df1.loc[(df1["row"] == 1) & (df1["col"] == 2), "value"].iloc[0])
    )
    / abs(float(df1.loc[(df1["row"] == 1) & (df1["col"] == 2), "value"].iloc[0]))
    * 100,
    2,
)
```

### Ví dụ query không hợp lệ

Gán trực tiếp đáp án:

```python
2.06
```

Che đáp án bằng phép toán hằng số:

```python
1.03 * 2
```

Lưu sẵn đáp án trong CSV rồi đọc lại:

```python
float(df1["answer"].iloc[0])
```

Tham chiếu DataFrame giả nhưng kết quả vẫn là hằng số:

```python
float(df1["value"].iloc[0]) * 0 + 2.06
```

Dùng một bảng không liên quan chỉ để query có tên DataFrame cũng không hợp lệ.
Các trường hợp trên không chứng minh được đáp án được tính từ evidence và sẽ
không được tính điểm.

## 4.5. Cơ chế bảo đảm trong code hiện tại

Checkpoint V28 triển khai các lớp kiểm soát sau:

| Thành phần | Kiểm soát |
|---|---|
| `vifinqa/extraction/report_parser.py` | Lưu nguồn báo cáo, trang, `table_pos` và `line_no` của bảng OCR |
| `vifinqa/retrieval/serialize.py` | Parse trực tiếp ô nguồn thành CSV dạng dài; không tạo cột đáp án |
| `vifinqa/codegen/semantic.py` | Phân tích AST và yêu cầu biểu thức đáp án phụ thuộc vào ít nhất một `dfN` |
| `vifinqa/codegen/executor.py` | Chạy query trên đúng DataFrame evidence và từ chối lỗi/giá trị không hữu hạn |
| `vifinqa/submission/build.py` | Đóng gói đúng CSV được query sử dụng và tạo đường dẫn `data/...` |
| `scripts/05_build_submission.py` | Kiểm tra biểu thức, replay đáp án và tạo `submission.zip` |

Semantic guard lần theo cả biến trung gian. Ví dụ `x = df1; answer = 0` vẫn bị
từ chối vì DataFrame không ảnh hưởng đến biểu thức cuối cùng.

## 4.6. Đối chiếu checkpoint leaderboard 0.4644

Artifact được kiểm tra:

```text
artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

Kết quả audit:

| Hạng mục | Số lượng |
|---|---:|
| Tổng số câu | 1.012 |
| Query có tham chiếu DataFrame trực tiếp | 1.001 |
| CSV evidence được đóng gói | 2.092 |
| Câu chưa giải, còn placeholder `0.0` | 11 |

Mười một câu chưa giải có ID:

```text
424, 425, 426, 464, 497, 510, 538, 805, 907, 950, 995
```

Các dòng này là trạng thái thất bại rõ ràng, không phải đáp án được mã hóa. Query
`0.0` của chúng không đáp ứng quy định grounding và không được kỳ vọng tính
điểm. `1.001` query còn lại đọc trực tiếp các DataFrame evidence và đã được
compile/replay khi build submission.

SHA-256 của submission đã chấm leaderboard:

```text
ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e
```

## 4.7. Checklist trước khi nộp

- [ ] Mọi CSV đều được sinh từ bảng nguồn do Ban Tổ chức cung cấp.
- [ ] Mọi `csv_path` tồn tại trong thư mục `data/` của ZIP.
- [ ] Mọi `relevant_tables` dùng `line_no` 1-based của dòng bắt đầu `<table>`.
- [ ] CSV không có cột chứa đáp án được tính sẵn.
- [ ] Mọi query cần tính điểm đều phụ thuộc vào ít nhất một DataFrame evidence.
- [ ] Query không che đáp án bằng phép toán trên hằng số.
- [ ] Đơn vị được quy đổi từ `unit_scale` hoặc ngữ cảnh bảng nguồn.
- [ ] Mọi query đều compile và replay ra đúng trường `answer`.
- [ ] `unzip -t submission.zip` không báo lỗi.
- [ ] SHA-256 của file nộp trùng với checkpoint đã ghi nhận.
