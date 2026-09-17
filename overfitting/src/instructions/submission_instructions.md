# Submission Instructions

## Dashboard kết quả

Các đội thi nộp kết quả dự đoán trực tiếp trên hệ thống Dashboard chính thức của cuộc thi. Mỗi lần nộp bài cần đảm bảo các yêu cầu sau:

- **Định dạng file:** Kết quả được nộp dưới dạng file chuẩn theo mẫu do Ban Tổ chức quy định, với cấu trúc trường dữ liệu tuân thủ đúng đặc tả.
- **Nội dung file:** Bao gồm kết quả dự đoán cho toàn bộ câu hỏi trong bộ dữ liệu kiểm thử. Các câu hỏi bị thiếu hoặc sai định dạng sẽ bị tính là dự đoán không hợp lệ.
- **Số lần nộp:** Mỗi đội được giới hạn số lần nộp bài mỗi ngày (chi tiết sẽ được công bố trên Dashboard) nhằm đảm bảo tính công bằng và tránh hiện tượng dò đáp án.

## Định dạng nộp bài

Bạn phải nộp một file dự đoán duy nhất ở định dạng `.json`. File phải tuân theo cấu trúc sau:

```json
[
  {
    "id": <integer>,
    "question": "<string>",
    "answer": <float>,
    "relevant_docs": ["<id_báo_cáo>"],
    "relevant_tables": ["<id_báo_cáo>|<vị trí trong báo cáo>"],
    "evidence": [
      {
        "variable": "<tên_biến_dataframe>",
        "csv_path": "<string>"
      }
    ],
    "pandas_query": "<string>"
  },
  ...
]
```

### Giải thích các trường dữ liệu:

- **`id`**: Mã định danh của câu hỏi, kiểu số nguyên (`integer`).
- **`question`**: Nội dung câu hỏi tài chính, kiểu chuỗi (`string`).
- **`relevant_docs`**: Danh sách mã định danh của các báo cáo hoặc tài liệu có liên quan đến câu hỏi. Mã báo cáo được xác định từ tên file cuối cùng trong đường dẫn tài liệu và loại bỏ phần mở rộng `.txt`.

  Ví dụ, với đường dẫn:

  ```text
  ocr_filter\AAA\2015\AAA_financial_statements_2015_consolidated
  ```

  mã báo cáo được sử dụng là:

  ```text
  AAA_financial_statements_2015_consolidated
  ```

- **`relevant_tables`**: Danh sách các bảng dữ liệu có liên quan trực tiếp đến câu trả lời. Mỗi phần tử có định dạng:

  ```text
  <id_báo_cáo>|<vị trí bảng trong báo cáo>
  ```

  Trong đó:

  - **`id_báo_cáo`**: Tên file cuối cùng trong đường dẫn tài liệu sau khi loại bỏ phần mở rộng `.txt`.
  - **`vị trí bảng trong báo cáo`**: Line bắt đầu của bảng trong báo cáo theo dữ liệu do Ban Tổ chức cung cấp.

  Ví dụ:

  ```text
  AAA_financial_statements_2015_consolidated|350
  ```

- **`answer`**: Kết quả số liệu kiểu số thực (`float`).
- **`evidence`**: Danh sách các bảng dữ liệu được sử dụng để thực thi `pandas_query`. Mỗi phần tử gồm:

  - **`variable`**: Tên biến DataFrame đại diện cho bảng và được sử dụng trực tiếp trong `pandas_query`. Tên biến phải hợp lệ trong Python và không được trùng nhau trong cùng một câu hỏi.
  - **`csv_path`**: Đường dẫn tương đối tới file CSV chứa dữ liệu mà `pandas_query` đã sử dụng để tính ra `answer`. Đường dẫn phải nằm trong thư mục `data/` của gói nộp bài.

- **`pandas_query`**: Câu lệnh pandas được sinh ra để trích xuất hoặc tính toán ra đáp án, kiểu chuỗi (`string`), có thể chạy lại được trên dữ liệu đã chuẩn hóa.

### Ví dụ bài nộp (JSON Schema Example)

```json
[
  {
    "id": 1,
    "question": "Doanh thu thuần của Công ty CP Sữa Việt Nam (VNM) năm 2023 là bao nhiêu?",
    "answer": 63075000000,
    "relevant_docs": ["AAA_financial_statements_2015_consolidated"],
    "relevant_tables": ["AAA_financial_statements_2015_consolidated|350"],
    "evidence": [
      {
        "variable": "df1",
        "csv_path": "data/AAA_financial_statements_2015_consolidated_table_1.csv"
      }
    ],
    "pandas_query": "df1[(df1.company=='VNM') & (df1.year==2023)]['net_revenue'].values[0]"
  }
]
```

Bài nộp phải được đóng gói dưới dạng một file ZIP, bao gồm một file kết quả `.json` và thư mục `data/` chứa đầy đủ các file CSV được tham chiếu bởi `csv_path` trong file kết quả.

### Cấu trúc đóng gói File ZIP

```text
submission.zip
├── <tên_file_kết_quả>.json
└── data/
    ├── <bảng_1>.csv
    ├── <bảng_2>.csv
    └── ...
```

### Lưu ý quan trọng khi nộp bài:

- File `.json` và thư mục `data/` phải nằm trực tiếp ở cấp ngoài cùng của file ZIP, không được đặt trong một thư mục cha khác.
- File ZIP chỉ được chứa một file kết quả `.json`.
- Mọi `csv_path`, bao gồm `csv_path` trong `evidence`, phải là đường dẫn tương đối bắt đầu bằng `data/`.
- Xin lưu ý rằng các bài nộp bị thiếu file hoặc thiếu câu sẽ không được đánh giá và sẽ không bị tính vào số lần nộp tối đa cho phép.
