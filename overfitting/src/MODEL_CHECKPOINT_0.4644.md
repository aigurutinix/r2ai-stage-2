# 3. Mô hình sử dụng

## 3.1. Thông tin về mô hình

Hệ thống sử dụng mô hình ngôn ngữ mở
[`Qwen/Qwen2.5-Coder-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
để hiểu câu hỏi tài chính và sinh/chọn biểu thức Pandas trên các bảng đã truy
hồi.

| Thuộc tính | Giá trị |
|---|---|
| Nhà phát triển | Qwen Team, Alibaba Cloud |
| Loại mô hình | Causal Language Model, instruction-tuned cho sinh và suy luận code |
| Kiến trúc | Qwen2 / Transformer |
| Số tham số | 7,61 tỷ tham số, trong đó 6,53 tỷ tham số non-embedding |
| Số lớp | 28 |
| Context | 32.768 token theo cấu hình mặc định; hỗ trợ tối đa 131.072 token khi bật YaRN |
| Giấy phép | Apache-2.0 |
| Thư viện suy luận | Hugging Face `transformers` |
| Lượng tử hóa khi chạy | 4-bit NF4, double quantization, compute `float16` |
| Backend/GPU đã dùng | Hugging Face backend trên Kaggle GPU T4 |

Đây là mô hình công khai, kích thước nhỏ hơn 14B và được phát hành trước ngày
01/06/2026. Hệ thống **không fine-tune và không tạo LoRA/adaptor riêng** cho
Qwen. Điểm số cuối cùng đến từ pipeline lai gồm:

1. Router tiếng Việt xác định doanh nghiệp, năm, loại báo cáo, đơn vị và phép
   toán.
2. BM25 và canonical metric dictionary truy hồi bảng/dòng tài chính.
3. Qwen2.5-Coder-7B-Instruct sinh hoặc chọn logic Pandas cho các câu phù hợp.
4. Deterministic financial planners/solvers xử lý lookup, count, ranking,
   ratio, margin, tăng trưởng, trung bình và các bảng thuyết minh nhiều trục.
5. Semantic guard thực thi lại biểu thức, kiểm tra đơn vị, evidence và từ chối
   kết quả không được grounding vào DataFrame.

Vì vậy, `EXECUTION_ACCURACY=0.4644` là điểm của **toàn bộ hệ thống**, không phải
độ chính xác độc lập của riêng mô hình Qwen.

## 3.2. Phiên bản checkpoint

### Checkpoint trọng số mô hình

| Thuộc tính | Giá trị |
|---|---|
| Model ID | `Qwen/Qwen2.5-Coder-7B-Instruct` |
| Hugging Face revision | `c03e6d358207e414f1eca0bb1891e29f1db0e242` |
| Định dạng | Safetensors, bốn shard |
| Link cố định | [Hugging Face checkpoint theo revision](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242) |

### Checkpoint hệ thống ViFinQA

| Thuộc tính | Giá trị |
|---|---|
| Tên checkpoint | `ViFinQA V28 - exact direct-growth audited3` |
| Git branch | `improve_baseline_kien` |
| Git commit | `1d52e9d` |
| Link mã nguồn cố định | [GitHub commit 1d52e9d](https://github.com/quangkhai5122/R2AI/tree/1d52e9d) |
| Retrieval | `artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl` |
| Codegen checkpoint | `artifacts/codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl` |
| Submission | `artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip` |
| Submission SHA-256 | `ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e` |

Kết quả leaderboard của checkpoint V28:

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

## 3.3. Hướng dẫn tải checkpoint

### Tải mã nguồn đúng phiên bản

```bash
git clone https://github.com/quangkhai5122/R2AI.git
cd R2AI
git checkout 1d52e9d
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Trên Windows, thay lệnh kích hoạt môi trường bằng:

```powershell
.venv\Scripts\activate
```

### Tải trọng số Qwen

Cài Hugging Face CLI và tải đúng revision đã khóa:

```bash
pip install "huggingface_hub>=0.27" "transformers>=4.45,<5" \
  "accelerate>=1,<2" "bitsandbytes>=0.45,<1"
hf download Qwen/Qwen2.5-Coder-7B-Instruct \
  --revision c03e6d358207e414f1eca0bb1891e29f1db0e242 \
  --local-dir checkpoints/Qwen2.5-Coder-7B-Instruct
```

Checkpoint là public và không yêu cầu Hugging Face access token. Trên Kaggle
có thể bỏ bước tải thủ công; `transformers.from_pretrained()` sẽ tải model từ
Model ID khi Internet được bật.

## 3.4. Hướng dẫn sử dụng checkpoint

### Chạy suy luận Qwen trên Kaggle

1. Build store và retrieval từ dữ liệu cuộc thi theo `README.md`/`RUNBOOK.md`.
2. Tạo Kaggle payload bằng `scripts/04_make_kaggle_payload.py`.
3. Import notebook `kaggle/vifinqa-codegen.ipynb` vào Kaggle.
4. Chọn GPU T4, bật Internet và attach đúng một payload dataset.
5. Chạy codegen với cấu hình chuẩn:

```bash
python /kaggle/working/code/kaggle_codegen.py \
  --payload "$PAYLOAD" \
  --backend hf \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --load-4bit \
  --out /kaggle/working/codegen_results.jsonl \
  --n 1 \
  --temperature 0 \
  --k 4 \
  --max-tokens 256 \
  --batch-size 4 \
  --checkpoint-every 32 \
  --time-budget-min 400
```

Runner sử dụng `transformers`, attention `sdpa`, quantization NF4 4-bit và tự
giảm batch `4 -> 2 -> 1` khi gặp CUDA OOM. Không dùng vLLM mới trên T4 vì engine
V1 không hỗ trợ ổn định kiến trúc GPU này.

### Đóng gói checkpoint V28 thành submission

Khi đã có đúng hai artifact frozen của V28 trong thư mục `artifacts/`, chạy:

```bash
python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl \
  --codegen artifacts/codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl \
  --out-dir artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010
```

Kiểm tra file tạo ra trước khi nộp:

```bash
unzip -t artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
sha256sum artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

SHA-256 phải là:

```text
ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e
```

Các artifact chứa dữ liệu và dự đoán của cuộc thi không phải trọng số model và
không được lưu trong Git. Chúng được tái tạo từ dữ liệu cuộc thi bằng mã nguồn
ở commit đã khóa. Trọng số model bắt buộc chia sẻ đã được công khai qua link
Hugging Face ở trên.

## 3.5. Đường link truy cập checkpoint

- Trọng số mô hình: [Qwen2.5-Coder-7B-Instruct, revision `c03e6d3`](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct/tree/c03e6d358207e414f1eca0bb1891e29f1db0e242)
- Model card và giấy phép: [Hugging Face model card](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- Mã nguồn pipeline `0.4644`: [GitHub checkpoint `1d52e9d`](https://github.com/quangkhai5122/R2AI/tree/1d52e9d)
- Nhánh phát triển: [GitHub branch `improve_baseline_kien`](https://github.com/quangkhai5122/R2AI/tree/improve_baseline_kien)

**Trạng thái truy cập ngày 29/08/2026:** link Hugging Face theo revision đã
được kiểm tra và truy cập công khai thành công. Repository GitHub hiện trả về
`404` khi truy cập không đăng nhập, tức là đang ở chế độ private. Trước khi gửi
tài liệu cho Ban Tổ chức, cần chuyển repository sang public hoặc cấp quyền truy
cập repository cho tài khoản do Ban Tổ chức cung cấp. Yêu cầu chia sẻ trọng số
mô hình vẫn đã được đáp ứng bởi checkpoint Hugging Face công khai.
