# Model, reranker và cơ chế suy luận

## 1. Qwen3-8B dùng trong production Easy route

- Checkpoint: `Qwen/Qwen3-8B-GGUF`, file `Qwen3-8B-Q4_K_M.gguf`.
- Layout local: `artifacts/models/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf`.
- Runtime: `llama-server.exe` từ llama.cpp.
- Server: `127.0.0.1:8087`, `-ngl 99`, context 16.384, parallel 1, Jinja chat template.
- Generation: temperature 0, `/no_think`, tối đa 3 lần sửa mặc định.

Model làm **semantic selector**, không phải retriever độc lập và không được tự
điền đáp án. Prompt chứa question cùng candidate cells đã lấy từ OCR. Output
phải là JSON có candidate IDs/operation hợp lệ. Đáp án cuối được code tính từ
`raw_number`, source scale và requested scale. Trước khi dùng server đang chạy,
Easy solver kiểm tra `/v1/models` để chắc chắn port 8087 thực sự phục vụ đúng
checkpoint 8B và đủ tham số, tránh vô tình dùng model khác.

## 2. Deterministic Easy reranker

Artifact nhỏ `easy_reranker_v2.json` được commit cùng package:

- thuật toán: pairwise logistic linear ranker;
- 58 features;
- train trên 101 câu source-audited;
- seed `20260809`;
- validation: 5-fold question-disjoint (`qid mod 5`) và issuer-grouped CV;
- mục tiêu: gold cell có mặt trong shortlist, không dự đoán answer.

Kết quả production-style shortlist trên 101 câu: legacy 72 hits, v1 91 hits,
v2 95 hits. OOF generator ranks của protocol qid-mod-5: cell top-1 34, row
top-1 37, table top-1 54, table top-10 98. Runtime và training feature score
khớp với sai số cực đại `3,55e-15`.

Feature groups gồm BM25/IDF table-row-page, phrase overlap, header/section,
period/opening/ending, unit match, stock/flow/movement, total/component,
identifier/nonzero và small-table prior. Reranker không được gọi là end-to-end
answer model; nó chỉ tăng recall của shortlist đưa sang Qwen.

## 3. Qwen3.5-9B cho audit/reasoning thử nghiệm

- Base: `Qwen/Qwen3.5-9B`.
- Quantization: `unsloth/Qwen3.5-9B-GGUF`, Q4_K_M.
- Revision pin: `3885219b6810b007914f3a7950a8d1b469d598a5`.
- Size: 5.680.522.464 bytes.
- SHA-256: `03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8`.
- Script: `tools/download_qwen35_checkpoint.py` (download resumable + hash check).

Model này là **opt-in research/audit**, không thay mặc định 8B của Easy route.
`reasoning_llm.py` tách reasoning/final, dùng temperature 0,6, top-p 0,95,
top-k 20 và seed 20260827. Các tool audit chỉ đưa question + source tables,
không đưa submitted answer để tránh confirmation leakage. Kết quả model chỉ
được dùng làm gợi ý vị trí cần kiểm tra; thay đổi release cần tọa độ và công
thức xác minh độc lập.

## 4. Các model thử nghiệm khác

Local workspace từng chứa Qwen3-4B/14B GGUF, Qwen3-Reranker-0.6B,
`bge-reranker-v2-m3` và LoRA cell/temporal experiments. Chúng không phải
dependency bắt buộc của release hiện tại. Benchmark cho thấy semantic row prior
giảm R@1/MRR so với calibrated legacy row, nên không bật global. Tài liệu không
đánh đồng việc model tồn tại trong `artifacts/` với việc nó tham gia production.

## 5. Chạy model

Full solver tự khởi động server khi cần. Để đổi model cho các client generic:

```powershell
$env:VIFINQA_MODEL = "D:\models\model.gguf"
$env:VIFINQA_MODEL_SOURCE = "repo/revision:file.gguf"
python -m road2ai_vifinqa.solve --ids 1-10 --run-dir runs/smoke
```

Easy production vẫn pin Qwen3-8B tại layout chuẩn. Không đặt model nhị phân vào
Git; ghi source, revision, size và SHA-256 trong manifest riêng.
