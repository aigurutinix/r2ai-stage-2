# Mô hình sử dụng

## Danh sách checkpoint đã khóa

| Checkpoint | Revision | Phát hành | License | Vai trò |
| --- | --- | --- | --- | --- |
| `Qwen/Qwen2.5-14B-Instruct` | `cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` | 16/09/2024 | Apache-2.0 | planner/reranker chính |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | `1df8507178afcc1bef68cd8c393f61a886323761` | 20/01/2025 | MIT | critic độc lập |
| `Qwen/Qwen3-14B` | `40c069824f4251a91eefaf281ebe4c544efd3e18` | 27/04/2025 | Apache-2.0 | tiebreak/critic thứ ba |

Ba checkpoint đều thuộc phân khúc 14B và phát hành trước cutoff 01/06/2026.
Qwen2.5-14B-Instruct có 14,7B tham số; Qwen3-14B có 14,8B tham số theo model
card nhưng checkpoint chính thức được công bố trong phân khúc 14B.

Link checkpoint được pin trực tiếp trong `configs/release.json`; revision Git
40 ký tự là định danh bất biến của toàn bộ snapshot thay cho tag `main` có thể
thay đổi.

## Vai trò và ranh giới an toàn

LLM là thành phần semantic audit, không phải máy tính đáp án:

1. Qwen nhận câu hỏi và danh sách candidate đã mask số, rồi chọn `cell_ref` hoặc
   sinh `FinancialPlan` JSON bị giới hạn schema.
2. DeepSeek/Qwen3 chỉ kiểm tra đề xuất alternative ở các batch cần critic.
3. Promotion cần consensus/gate deterministic và source audit.
4. LLM không thấy answer, Pandas query hay giá trị nguồn.
5. Retrieval cuối, evidence CSV, phép toán và answer đều được tái tạo cục bộ từ
   dữ liệu BTC.

Do các quyết định được chấp nhận đã lưu thành tọa độ nguồn, việc replay/nộp bài
không cần GPU hoặc tải model. Chỉ chạy lại LLM khi muốn tái tạo toàn bộ quá
trình semantic audit.

## Cấu hình inference

- Python 3.11 trên Modal;
- GPU H200; `torch_dtype=torch.bfloat16`; `device_map="auto"`;
- `torch==2.6.0`, `transformers==4.51.3`, `accelerate==1.6.0`,
  `safetensors==0.5.3`;
- decoding deterministic (`do_sample=False`);
- Qwen3 dùng `enable_thinking=False` trong reviewer;
- seed không ảnh hưởng do không sampling;
- prompt/context theo từng script `scripts/modal_*.py`, output bắt buộc JSON.

## Tải checkpoint

```bash
python -m pip install -r requirements-llm.txt

huggingface-cli download Qwen/Qwen2.5-14B-Instruct \
  --revision cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8
huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --revision 1df8507178afcc1bef68cd8c393f61a886323761
huggingface-cli download Qwen/Qwen3-14B \
  --revision 40c069824f4251a91eefaf281ebe4c544efd3e18
```

Nguồn chính thức:

- https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
- https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
- https://huggingface.co/Qwen/Qwen3-14B

Các repository trên là link chia sẻ checkpoint công khai. Trước nghiệm thu,
kiểm tra lại quyền tải ở chế độ chưa đăng nhập và lưu kết quả vào release note.
