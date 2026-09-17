# Session 2026-08-03 — P0 correctness và phục hồi Kaggle Qwen

## Mục tiêu

1. Chẩn đoán vì sao Kaggle vẫn in `LLM round 1: 1012 prompts`, không có
   `baseline written`, `LLM queue` hay `[chunk ...]`, rồi CUDA OOM.
2. Làm runner crash-safe, có checkpoint/resume và tự hạ batch khi OOM.
3. Khóa đồng bộ local ↔ Kaggle bằng payload manifest SHA-256, không còn hot-patch.
4. Tiếp tục P0 correctness: parser số/đơn vị, entity routing, report mapping,
   semantic code validation và submission replay guard.
5. Ghi nhận submission #5 fixed k=5 làm control cho lượt Qwen đầu tiên.

## Kết luận nguyên nhân Kaggle

Log `LLM round 1: 1012 prompts` thuộc runner cũ. Notebook cũ chỉ vá một số file
trên Kaggle nhưng không vá `vifinqa/codegen/generate.py`, nên payload cũ vẫn chạy
toàn bộ 1.012 prompt trong một round lớn. Bản đó không có bước ghi baseline đầy
đủ trước, không chia chunk/checkpoint và không tự hạ batch, vì vậy CUDA OOM làm
mất tiến độ.

Notebook hiện hành không vá code. Nó yêu cầu đúng một payload schema v2, kiểm
SHA-256 tất cả runtime input, copy nguyên trạng code đã fingerprint sang
`/kaggle/working`, rồi kiểm lại bản copy. Payload sai/cũ sẽ fail fast.

## Submission #5 — baseline đối chứng đã đo trên leaderboard

Cấu hình: `pos=line`, fixed `sub-k=5`, `expand-docs=False`, rule-only, không LLM.

| Metric | Score |
|---|---:|
| TABLES_F2MACRO | 0.4092 |
| TABLES_PRECISION | 0.2621 |
| TABLES_RECALL | 0.5852 |
| TABLES_MRR5 | 0.5882 |
| DOCS_F2MACRO | 0.8093 |
| DOCS_PRECISION | 0.9168 |
| DOCS_RECALL | 0.7999 |
| DOCS_MRR5 | 0.9457 |
| ANSWER_ACCURACY | 0.085 |
| EXECUTION_ACCURACY | 0.085 |

Audit artifact:

- 1.012/1.012 `relevant_tables` là đúng top-5 prefix của submission #3 k=10.
- 1.012/1.012 answer và pandas query giữ nguyên so với #3.
- Source distribution: 188 `rule`, 824 `none`, 0 `llm`.
- 5.060 bảng dự đoán; 2.018 doc dự đoán, trung bình 1,9941 doc/câu.
- TABLES_F2 tăng 0.3641→0.4092 (+0.0451; +12,4% tương đối) nhưng recall giảm.
- k=5 tốt hơn k=10 trên dữ liệu đã thử; chưa có bằng chứng k=5 tốt hơn k=7.
- F2 là macro-average theo từng câu, không được suy lại từ aggregate P/R.

SHA-256 snapshot submission #5:

- `artifacts/retrieval.jsonl`:
  `1A9824214A08C4E82CD0A65F29F8FF7616DD71D73342C5392D17A26BEEE0F8A8`
- `artifacts/codegen_results.jsonl`:
  `7356D72BF59F5E988E649DF12DA51A6DE905D85F13332D66AA44969C0097350C`
- `artifacts/submission/results.json`:
  `16FF58AA6E19E2366EB5C560D43A0F4C097BD4EC982BAB2E19BAD91D15E8E2B4`
- `artifacts/submission/submission.zip`:
  `FDED842DD036BF7E814D85B665098EDA37AC6DBCC3823CC603A6D442B411F67A`

## Những file đã sửa

### Runner và Kaggle

- `vifinqa/codegen/generate.py`
  - Ghi và flush rule baseline đủ mọi ID trước khi gọi LLM.
  - Chia LLM theo chunk; checkpoint sau round 1 và sau self-debug.
  - Kiểm time budget trước chunk; lỗi generation/debug vẫn giữ checkpoint hợp lệ.
  - Resume chỉ tái dùng LLM result có cùng `run_signature`.
  - Thêm semantic validation vào round 1, debug và rule result.
  - Ghi evidence cho mọi DataFrame chương trình truy cập; grounding của answer
    vẫn được kiểm riêng để không chấp nhận constant hallucination.
- `vifinqa/codegen/llm_client.py`
  - CUDA OOM backoff theo batch `4→2→1`, không bỏ sót prompt.
  - Clear CUDA cache giữa các lần thử; báo lỗi rõ nếu batch 1 vẫn OOM.
  - Seed deterministic cho vLLM; ràng buộc temperature khi `n>1`.
- `kaggle/kaggle_codegen.py`
  - Verify payload schema v2 + SHA-256; tạo `run_signature` từ payload/model/cấu hình.
  - Seed Python, NumPy và Torch.
- `scripts/04_make_kaggle_payload.py`
  - Sinh `payload-manifest.json`; giữ dataset ID cũ; thêm `--dry-run`.
  - Validate source trước khi thay payload và chỉ cho phép target dưới `artifacts/`.
- `kaggle/vifinqa-codegen.ipynb`
  - Xóa hot-patch; chỉ attach đúng một payload.
  - Smoke 12 câu và full run 7B NF4: `n=1`, `temperature=0`, codegen `k=4`,
    `max_tokens=256`, `batch_size=4`, checkpoint 32, time budget 400 phút, seed 13.
  - QA bắt buộc 1.012 ID duy nhất và answer hữu hạn.
- `vifinqa/__init__.py`: version `0.2.0`.

### P0 data/routing correctness

- `vifinqa/utils/viet_num.py`
  - Chuẩn hóa dấu phẩy thập phân tiếng Việt, gồm `5,832`, `0,9237`.
- `vifinqa/extraction/report_parser.py`
  - Hỗ trợ `trăm tỷ`; phân biệt `separate`, `consolidated`, `aggregated`, `other`.
- `vifinqa/extraction/build_store.py`
  - Thêm one-to-many report index và `find_reports`, giữ API legacy.
- `vifinqa/router/entities.py`
  - Bổ sung đơn vị trăm tỷ, triệu/nghìn cổ phiếu, cổ phiếu và triệu USD.
  - Multi-company alias, ticker chữ thường có guard, ưu tiên target trong ngoặc.
  - Mở rộng year range; phân loại output `number`, `percent`,
    `percentage_point`, `ratio`, `year`, `count`.
- `vifinqa/router/router.py`: truyền `output_type`, dùng one-to-many report lookup.
- `vifinqa/codegen/prompts.py`: truyền output type và quy ước `1e11` cho trăm tỷ.
- `vifinqa/codegen/rule_codegen.py`: rule deterministic chỉ áp dụng output number.
- `vifinqa/config.py`: thêm unit scale trăm tỷ và ghi lại score k=5 thực đo.

### Semantic/submission correctness

- `vifinqa/codegen/semantic.py` (mới)
  - Phân tích AST/dataflow để chỉ nhận DataFrame thực sự ảnh hưởng `answer`.
  - Từ chối constant/dead/unknown DataFrame reference.
  - Sanity check cứng cho `year`, `count`; cảnh báo magnitude cho percent/ratio.
- `vifinqa/submission/build.py`
  - Từ chối duplicate/missing ID, answer không hữu hạn, query thành công nhưng
    constant, DataFrame không có evidence, mapping line/evidence bị thiếu.
  - Tự bổ sung evidence nếu codegen dùng bảng ngoài submission cutoff.
  - Dùng AST thay regex để nhận DataFrame, bỏ false positive trong comment/string.
  - Kiểm exact zip layout và replay toàn bộ query trên đúng CSV sẽ nộp.

### Test và tài liệu

- Thêm/cập nhật sáu module test:
  `test_codegen_runner.py`, `test_codegen_semantic.py`, `test_submission_guards.py`,
  `test_viet_num.py`, `test_entities.py`, `test_report_store.py`.
- Cập nhật `README.md` và `CLAUDE.md` để bỏ hướng dẫn hot-patch/forecast cũ,
  ghi score #5, cấu hình 7B và quy trình payload schema v2.
- Sửa ví dụ CLI trong `scripts/05_build_submission.py`: position chính thức là
  `line`; minh họa ablation `sub-k=7` bằng output directory riêng.

## Kiểm chứng đã chạy

- Full test suite sau bản vá cuối: **65 passed in 13.32s**.
- Runner + semantic + submission guard: **24 unittest passed**.
- Rule smoke 20 câu: 13 rule, 7 none; semantic results grounded.
- Submission smoke 12 câu: build strict + replay thành công.
- Replay artifact baseline hiện có: 1.012 result, 954 evidence CSV; strict
  line/evidence/zip validation và full pandas-query replay đều thành công.
- Snapshot P0 mới: 1.973 reports, 146.246 tables, 2.722.031 cells; retrieval
  đủ 1.012 ID, không duplicate/missing/nonfinite; rule baseline 188 rule/824 none.
- Submission P0 mới: 1.012 entries, 951 evidence CSV; strict build và full
  pandas-query replay trên đúng CSV nộp đều thành công.
- Source package và staged payload: 29/29 file khớp SHA-256, 0 khác biệt.
- Control store và staged payload: 201/201 file khớp SHA-256; retrieval khớp hash.

## Payload local đã đóng băng cho lượt Qwen control

- Path: `artifacts/kaggle_payload/`
- Dataset ID: `lequangkhai5122005/vifinqa-payload`
- Schema: 2; package version: 0.2.0; 232 file được fingerprint.
- Kích thước: khoảng 100 MB.
- Stable manifest digest:
  `5d3797f8837a8274250b1b6fe981ce3bad48087045c9674216c4e4137c89c406`
- SHA-256 file `payload-manifest.json`:
  `DC7B2F831101421B390A279402591BAEC46F4D0E2AFBADC3B07710DA2BAA92AF`
- SHA-256 notebook cần import riêng `kaggle/vifinqa-codegen.ipynb`:
  `865A3AAB6303C9202F35E1B388D9DC4A24CB9E010890E63903E7184984D3441C`

Payload này cố ý giữ `artifacts/store/` + `artifacts/retrieval.jsonl` của
submission #5 làm control, nhưng dùng runner/P0 guards mới. Như vậy lượt Qwen
đầu chỉ thay trục codegen; TABLES/DOCS vẫn so được trực tiếp với #5.

Các sửa parser/report routing P0 đã được dựng và kiểm ở đường dẫn riêng:

- `artifacts/store_p0/`: 1.973 reports; doc type gồm 957 consolidated,
  954 separate, 55 other, 7 aggregated; không duplicate/null report ID.
- `artifacts/retrieval_p0.jsonl`: 1.012 records; output types gồm 646 number,
  217 percent, 53 year, 51 ratio, 23 count, 22 percentage_point.
- `artifacts/p0_rule_full.jsonl` và `artifacts/p0_submission_full/`: full rule/replay
  validation thành công.

SHA-256 P0 snapshot:

- `store_p0/reports.parquet`:
  `5A6EE350BFB8EC5DDDC33D8B8B6AD2182EFC8C73774CB0921B26AED3A3B3347D`
- `retrieval_p0.jsonl`:
  `C2FDC67867476D39165C2861775CAAD7C0F7A4E8D8A9DB9148850DF03BD9A6A0`
- `p0_rule_full.jsonl`:
  `AB1397C3269E5434A43AB61A36DC1E24FC9E9BF16FD1B40DC705CD17788B7A6F`
- `p0_submission_full/submission.zip`:
  `0A2239BB1C78BF1392957696CC677DDB9F70D3621F66D09DCA657C2D4B7053E2`

P0-data không phải drop-in tương đương control: top-5 retrieval giữ nguyên ở
831/1.012 câu, đổi ở 181 câu; relevant docs giữ nguyên ở 835/1.012; rule answer
giữ nguyên ở 982/1.012. Vì chưa có gold offline độc lập, đưa P0-data và Qwen vào
cùng một lượt sẽ làm mất khả năng quy nguyên nhân. Do đó chưa đóng gói P0-data
thành dataset upload; control Qwen được ưu tiên trước.

## Đồng bộ và chạy trên Kaggle

Từ gốc repo local:

```powershell
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P0 crash-safe runner, manifest schema v2"
```

Sau đó:

1. Lưu riêng log/output của run OOM cũ nếu cần hậu kiểm, rồi dừng run cũ.
2. Trên Kaggle dataset `vifinqa-payload`, kiểm tra version mới đã xử lý xong.
3. Import lại `kaggle/vifinqa-codegen.ipynb` hiện hành; không dùng notebook cũ.
4. Gỡ payload input cũ/duplicate; chỉ attach đúng version mới của dataset.
5. Chạy smoke 12 câu. Log đúng phải có:
   - `payload verified: schema=2 files=232`
   - `run signature: ...`
   - `baseline written (...)`
   - `LLM queue: ...`
   - `[chunk 1/...] ...`
6. Nếu smoke sạch, chạy full 7B đúng cấu hình notebook.
7. Download cả `codegen_results.jsonl` và notebook log.

Nếu tiếp tục ở Kaggle session mới, đưa checkpoint cũ về đúng
`/kaggle/working/codegen_results.jsonl` trước full cell và giữ nguyên payload,
model, `n`, `k`, max tokens, debug rounds, seed. `run_signature` khác sẽ làm
runner bỏ qua LLM answer cũ nhưng vẫn tạo baseline đầy đủ, tránh resume sai.

## Quyết định và bước kế tiếp

1. Chạy Qwen 7B control trên đúng payload đã đóng băng; build submission với
   `sub-k=5`, `expand-docs=False` và replay guard.
2. So ANSWER/EXEC với 0.085; không gán cải thiện TABLES/DOCS cho LLM.
3. Từ cùng Qwen result, build thêm `sub-k=7` nếu còn lượt nộp; không rerun GPU.
4. Sau score Qwen control, đánh giá P0-data bằng một submission rule-only riêng
   hoặc leaderboard ablation trước khi thay payload control.
5. Sau khi có Qwen score, mới sang P1 retrieval: BGE-M3 + reranker trong scope.

## Ranh giới bằng chứng

- **Leaderboard-confirmed:** toàn bộ metric submission #5 ở trên.
- **Artifact-confirmed:** prefix k=5, source distribution, SHA-256 và replay.
- **Test-confirmed:** hành vi guard/checkpoint/OOM fallback ở mức test/smoke local.
- **Chưa xác nhận:** runtime/OOM thực tế của runner mới trên Kaggle T4 và score
  Qwen; chỉ có thể kết luận sau smoke/full remote và một lượt submission.
- **Known P1 caveat:** AST grounding hiện ưu tiên code PoT thẳng; control flow
  phức tạp (`if/else` gán answer ở nhiều nhánh, loop/augmented assignment, tuple
  unpacking) có thể bị reject bảo thủ và rơi về rule baseline, nhưng không làm
  mất checkpoint. Failed placeholder vẫn giữ evidence bảng đầu như baseline đã
  được grader chấp nhận; nên tái kiểm schema nếu sau này chuyển sang evidence rỗng.
