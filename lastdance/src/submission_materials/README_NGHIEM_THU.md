# HỒ SƠ THUYẾT MINH SẢN PHẨM VIFINQA

**Đội thi:** lastdance  
**Sản phẩm:** Hệ thống truy hồi bảng dữ liệu và sinh truy vấn Pandas trên báo cáo tài chính  
**Cuộc thi:** ROAD TO AI 2026 – Stage 2 Text-to-Pandas  
**Ngày nộp:** 31/08/2026

> Phạm vi nghiệm thu của hồ sơ này là pipeline tạo và kiểm chứng bài nộp từ dữ liệu do Ban Tổ chức (BTC) cung cấp. Các thành phần nghiên cứu hoặc thử nghiệm ngoài luồng release không phải điều kiện để tái lập bài nộp.

## 1. Tổng quan sản phẩm

Sản phẩm tiếp nhận câu hỏi tài chính tiếng Việt, xác định doanh nghiệp, kỳ báo cáo, phạm vi báo cáo và chỉ tiêu liên quan; truy hồi bảng nguồn; chuẩn hóa kế hoạch tính toán; trích xuất bảng CSV làm bằng chứng; sinh câu lệnh Pandas; thực thi và kiểm tra kết quả.

Luồng release chính:

```text
Câu hỏi → warehouse/index → truy hồi bảng và ô nguồn
→ execution plan → trích xuất evidence CSV
→ sinh/thực thi pandas query → validation/provenance → submission.zip
```

Kết quả kiểm chứng trên artifact bàn giao:

| Hạng mục | Kết quả |
| --- | ---: |
| Số câu hỏi | 1.012 |
| Câu lệnh Pandas replay thành công | 1.012/1.012 |
| Tệp evidence CSV | 5.941 |
| Liên kết provenance về bảng nguồn | 5.941/5.941 |
| Số bảng nguồn được tham chiếu | 2.842 |
| Data-flow gate | PASS |
| Zero-weight/hardcoded-result gate | PASS |

## 2. Tài liệu mô tả dữ liệu

### 2.1. Nguồn dữ liệu

Pipeline chỉ sử dụng dữ liệu tài chính và câu hỏi do BTC cung cấp trong bộ ViFinQA. Không bổ sung số liệu tài chính từ nguồn bên ngoài vào evidence hoặc phép tính của bài nộp.

| Nhóm dữ liệu | Quy mô/định dạng | Vai trò |
| --- | --- | --- |
| Báo cáo tài chính BTC | 1.973 báo cáo OCR/TXT, 100 mã, giai đoạn 2015–2025 | Nguồn dữ liệu gốc |
| Câu hỏi | `questions/questions.jsonl`, 1.012 bản ghi | Đầu vào truy vấn |
| Danh mục doanh nghiệp | `code_stock.csv` | Mã chứng khoán và tên doanh nghiệp |
| Warehouse dẫn xuất | SQLite `artifacts/vifinqa.db` | Chỉ mục tài liệu, trang, bảng và FTS |
| Evidence bài nộp | CSV trong thư mục `data/` của ZIP | Bảng nguồn phục vụ từng pandas query |

### 2.2. Cấu trúc dữ liệu

Warehouse SQLite gồm các bảng chính:

- `documents`: định danh báo cáo, ticker, năm, phạm vi báo cáo, đường dẫn và checksum;
- `pages`: vị trí trang và dòng nguồn;
- `tables`: bảng OCR, `evidence_key`, đơn vị, loại bảng, `grid_json` và nội dung HTML;
- `table_fts`: chỉ mục toàn văn phục vụ truy hồi;
- `table_catalog`: view hỗ trợ kiểm tra và truy vết.

Kho hiện có 1.973 documents và 146.246 tables. `evidence_key` có dạng `<document_id>|<source_line_1>`. Mỗi CSV evidence giữ nguyên grid của bảng nguồn, nhờ đó biểu thức `df.iloc[row, column]` có thể ánh xạ trực tiếp về ô nguồn tương ứng.

### 2.3. Truy cập dữ liệu

- Link dữ liệu chia sẻ: **https://huggingface.co/datasets/AIGuruTinix/ViFinQA**
- Quyền truy cập: **[Anyone with the link / theo yêu cầu của BTC]**
- Ngày kiểm tra quyền tải bằng tài khoản ngoài đội: **[ĐIỀN NGÀY GIỜ]**
- Checksum dữ liệu/manifest: cung cấp trong `outputs/baselines/benchmark_locked/manifest.json` và release manifest.

Sau khi tải, giải nén sao cho tồn tại:

```text
ViFinQA/
├── questions/questions.jsonl
├── code_stock.csv
└── dữ liệu báo cáo OCR/TXT do BTC cung cấp
```

## 3. Mô hình sử dụng

### 3.1. Checkpoint

| Mô hình | Revision cố định | License | Vai trò |
| --- | --- | --- | --- |
| `Qwen/Qwen2.5-14B-Instruct` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | Apache-2.0 | Semantic planner/reranker |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `1df8507178afcc1bef68cd8c393f61a886323761` | MIT | Critic độc lập |
| `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` | Apache-2.0 | Tiebreak/critic |

Nguồn checkpoint chính thức:

- https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- https://huggingface.co/Qwen/Qwen3-14B

Link checkpoint do đội chia sẻ (nếu BTC yêu cầu mirror): **[ĐIỀN LINK]**

### 3.2. Cấu hình inference

- Python 3.11;
- Modal H200/A100 80 GB cho giai đoạn chạy lại semantic audit;
- `torch==2.6.0`, `transformers==4.51.3`, `accelerate==1.6.0`, `safetensors==0.5.3`;
- suy luận deterministic, không sampling;
- numeric masking đối với context đưa vào mô hình;
- LLM không nhận đáp án tham chiếu hoặc giá trị tài chính nguồn trong prompt semantic audit.

Checkpoint không cần thiết cho bước replay bài nộp đã khóa. Người nghiệm thu chỉ cần checkpoint khi muốn tái chạy giai đoạn semantic audit nghiên cứu.

### 3.3. Tải checkpoint

```bash
python -m pip install -r requirements-llm.txt

huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
  --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8

huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --revision 1df8507178afcc1bef68cd8c393f61a886323761

huggingface-cli download Qwen/Qwen3-14B \
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18
```

## 4. Mã nguồn và dependencies

### 4.1. Phạm vi mã nguồn

Bộ mã nguồn bàn giao gồm:

- build warehouse SQLite từ dữ liệu BTC;
- truy hồi và grounding bảng/ô dữ liệu;
- biểu diễn `FinancialPlan` và compiler Pandas;
- các solver theo nhóm nghiệp vụ tài chính;
- trích xuất evidence CSV;
- sandbox/validator cho pandas query;
- đóng gói, replay và kiểm tra provenance;
- script Modal dùng cho semantic planner/critic;
- cấu hình release và execution plan đã khóa;
- tài liệu dữ liệu, mô hình và hướng dẫn tái lập.

Link mã nguồn: **https://github.com/Viendeptrai1/lastdance-vifinqa**

### 4.2. Dependencies

Core release:

```text
pandas==2.2.2
rapidfuzz==3.14.5
pytest==9.1.1
```

LLM/research:

```text
modal==1.5.3
torch==2.6.0
transformers==4.51.3
accelerate==1.6.0
safetensors==0.5.3
```

Các dependency chính được khai báo trong `pyproject.toml`, `requirements.txt`, `requirements-dev.txt` và `requirements-llm.txt`.

### 4.3. Tệp cấu hình

- `configs/release.json`: metadata release, đường dẫn, checksum, model và inference;
- `configs/release_plan.jsonl`: 1.012 execution items gồm câu hỏi, provenance và pandas query;
- `analysis/financial_plan.schema.json`: schema kế hoạch tài chính;
- `scripts/modal_auto_planner.py`: cấu hình chạy planner trên Modal;
- `Docs/`: mô tả kiến trúc, dữ liệu, mô hình và tái lập.

## 5. Bài nộp và tuân thủ dữ liệu

### 5.1. Cấu trúc ZIP

```text
submission.zip
├── submission.json
└── data/
    ├── q0001_evidence.csv
    ├── ...
    └── các CSV evidence khác
```

Mỗi item trong `submission.json` gồm:

- `id`, `question`, `answer`;
- `relevant_docs`, `relevant_tables`;
- danh sách evidence variable → CSV;
- `pandas_query`.

### 5.2. Quan hệ giữa CSV và dữ liệu BTC

Mỗi CSV evidence được trích từ một bảng trong dữ liệu BTC và mang liên kết provenance gồm document, vị trí dòng nguồn, `evidence_key` và checksum grid. Validator đối chiếu nội dung CSV với `grid_json` của bảng nguồn trong warehouse.

CSV có thể chứa toàn bộ grid của bảng nguồn hoặc tập con cần thiết. Trong artifact hiện tại, pipeline ưu tiên giữ grid nguồn để việc kiểm tra tọa độ hàng/cột rõ ràng và ổn định.

### 5.3. Không lưu sẵn kết quả

Kết quả được tính trực tiếp khi thực thi pandas query trên các dataframe evidence. Execution plan không lưu trường answer hoặc số liệu tài chính nguồn. Validator thực hiện các gate:

- query phải đọc ít nhất một evidence dataframe;
- từ chối import, filesystem và network access;
- từ chối biểu thức bỏ qua dữ liệu nguồn bằng zero-weight;
- replay lại toàn bộ 1.012 pandas query;
- đối chiếu mọi evidence CSV về bảng BTC tương ứng.

## 6. Hướng dẫn cài đặt và tái lập

### 6.1. Môi trường

- Linux hoặc macOS 64-bit;
- Python 3.11;
- SQLite có JSON1 và FTS5;
- tối thiểu 8 GB RAM và 4 GB dung lượng trống;
- không cần GPU cho build/replay bài nộp.

### 6.2. Cài đặt

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e . -r requirements-dev.txt
```

### 6.3. Build warehouse và bài nộp

```bash
PYTHONPATH=src python -m vifinqa.release \
  --config configs/release.json build \
  --output-dir outputs/release_build/submission \
  --rebuild-database
```

Nếu đã có `artifacts/vifinqa.db`, có thể bỏ `--rebuild-database`.

Output:

```text
outputs/release_build/submission.zip
```

### 6.4. Replay và validation

```bash
PYTHONPATH=src python -m vifinqa.submission_validation \
  --zip outputs/release_build/submission.zip \
  --database artifacts/vifinqa.db
```

Kết quả kỳ vọng:

```text
items               1012
replayed            1012
evidence_files      5941
provenance_bindings 5941
dataflow_gate       PASS
zero_weight_gate    PASS
```

### 6.5. Chạy kiểm thử

```bash
PYTHONPATH=src pytest -q
```

## 7. Danh sách artifact bàn giao

- Tài liệu thuyết minh sản phẩm này;
- toàn bộ source code và dependencies;
- link dữ liệu BTC và warehouse dẫn xuất;
- link checkpoint/model revisions;
- `submission.zip`;
- `configs/release_plan.jsonl`;
- baseline manifest và checksum;
- README/hướng dẫn tái lập.

## 8. Thông tin liên hệ

- Đại diện đội: **Phan Quốc Viễn**
- Email: **[ĐIỀN EMAIL]**
- Số điện thoại: **0914426099**
