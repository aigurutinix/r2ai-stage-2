# MODEL.md — mô hình, checkpoint và hướng sử dụng

## 1. Tổng quan

Hệ thống AI_Guru là một pipeline bài toán tài chính dạng text-to-pandas, không dựa trên một mô hình duy nhất làm toàn bộ nhiệm vụ. Nền tảng chính của hệ thống là:

1. Truy hồi tài liệu/bảng bằng BM25 + lọc thực thể theo điều kiện câu hỏi
2. Chấm các ô/bảng theo quy tắc cấu trúc dữ liệu
3. Sinh `pandas_query` từ bảng/ô đã chọn
4. Kiểm tính hợp lệ bằng sandbox nội bộ trước khi chốt bài nộp

## 2. Mô hình và thành phần đã dùng

### 2.1 BM25 / truy hồi từ vựng
- Mục đích: chọn bảng ứng viên trong kho dữ liệu lớn
- Công cụ: Python + `pandas`, BM25-style lexical ranking
- Không cần checkpoint ngoài
- Thực thi: `scripts/retrieve.py`

### 2.2 Reranker ngữ nghĩa (nếu dùng trong các vòng phát triển)
- Model: `BAAI/bge-reranker-v2-m3`
- Link chính thức: https://huggingface.co/BAAI/bge-reranker-v2-m3
- Dùng trong giai đoạn tăng cường chọn bảng lại trên top-N ứng viên
- Dùng như một bước xếp hạng lại, không phải sinh đáp án

### 2.3 LLM hỗ trợ (nếu chạy trên Kaggle / GPU)
Có hai lựa chọn mô hình được ghi rõ trong quá trình phát triển:

#### a) `Qwen/Qwen3-14B-AWQ`
- Link chính thức: https://huggingface.co/Qwen/Qwen3-14B-AWQ
- Mô tả: 14B, AWQ 4-bit, hỗ trợ sinh `pandas_query` trong môi trường GPU
- Dùng trong notebook: `notebooks/generate_pandas_kaggle.py` và `notebooks/generate_pandas_awq.py`

#### b) `Qwen/Qwen3-8B-AWQ`
- Link chính thức: https://huggingface.co/Qwen/Qwen3-8B-AWQ
- Mô tả: 8B, AWQ 4-bit, phù hợp với nhớ RAM thấp hơn so với 14B
- Dùng trong một số vòng thử nghiệm trên Kaggle

## 3. Phiên bản checkpoint và nơi lưu

Các checkpoint mô hình được tải từ Hugging Face theo tên model mở trên Hub. Không có checkpoint nội bộ được đóng gói vào repo vì kích thước lớn và không phù hợp với lưu trữ Git.

Các link truy cập mô hình đã sử dụng:
- Qwen3-14B-AWQ: https://huggingface.co/Qwen/Qwen3-14B-AWQ
- Qwen3-8B-AWQ: https://huggingface.co/Qwen/Qwen3-8B-AWQ
- bge-reranker-v2-m3: https://huggingface.co/BAAI/bge-reranker-v2-m3

## 4. Môi trường chạy mô hình

### 4.1 Môi trường local (không cần GPU)
Đường ống chính chạy được trên máy cá nhân thông thường:
- Python 3.12.2
- RAM 16 GB
- Không cần GPU cho flow chính

Cách chạy:
```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
python scripts/download_corpus.py
python scripts/build_tables.py --workers 6
python scripts/map_questions.py
python scripts/retrieve.py
python scripts/make_submission.py --selftest --use-rowname --out submissions/khong_gpu.zip
```

### 4.2 Môi trường GPU / Kaggle
Nếu cần chạy các bước tăng cường bằng mô hình LLM trên Kaggle, dùng các notebook:
- `notebooks/generate_pandas_kaggle.py`
- `notebooks/generate_pandas_awq.py`
- `notebooks/pick_cell_kaggle.py`
- `notebooks/rowname_probe_kaggle.py`

Lưu ý:
- Đây là các notebook phát triển, không phải phần bắt buộc để chu trình chính chạy locally.
- Mô hình được lấy từ Hugging Face Hub theo link đã nêu ở trên.

## 5. Hướng dẫn tải và sử dụng checkpoint

1. Truy cập link model chính thức trên Hugging Face.
2. Chọn phiên bản AWQ hoặc checkpoint tương ứng đã ghi trong mô tả model.
3. Dùng script/Notebook tương ứng để nạp mô hình.
4. Đảm bảo không dùng model đóng (GPT-4o, Gemini, Claude, v.v.) theo ràng buộc BTC.

Ví dụ khởi tạo model (mẫu):
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-14B-AWQ",
    device_map={"": 0},
    trust_remote_code=False,
)
```

## 6. Quy định về dữ liệu và mô hình

- Mọi mô hình dùng phải là mô hình mở, không phải model đóng.
- Mô hình phải phát hành trước 01/06/2026 theo quy định.
- Không dùng tri thức ngoài dữ liệu BTC cấp nếu không được khai báo trong `SOURCES.md`.
- Mọi checkpoint và model đều phải được tham chiếu rõ ràng bằng đường link chính thức trên Hugging Face.

## 7. Tài liệu tham khảo

- `SOURCES.md`
- `README.md`
- `reference/ViFinQA/README.md`
- `scripts/download_corpus.py`
- `notebooks/generate_pandas_awq.py`
- `notebooks/generate_pandas_kaggle.py`

## 8. Kết luận

Pipeline hiện tại không phụ thuộc vào một checkpoint cục bộ nào trong repo; nó sử dụng dữ liệu BTC và các checkpoint mở công khai trên Hugging Face. Đây là cấu hình dễ tái lập và đáp ứng quy định “cung cấp link truy cập dữ liệu/checkpoint” theo đúng yêu cầu nộp tài liệu thuyết minh sản phẩm.
