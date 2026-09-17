# P1 Implementation — 4 bước lấp khoảng trống 648 câu rỗng

Ngày: 2026-08-04. Triển khai theo thứ tự đã phân tích trong `P1_STRATEGY_REVIEW.md`.
Toàn bộ 96 test pass. Mọi con số "đo được" dưới đây chạy trên corpus thật.

---

## 0. TL;DR — cần làm gì ngay

```powershell
# 1) Chạy lại retrieval (BẮT BUỘC: router đã đổi, retrieval.jsonl cũ không có plan/metric mới)
python scripts/02_retrieve.py --out artifacts/retrieval_p1.jsonl

# 2) Rule baseline mới (KHÔNG cần GPU) - đây là ablation nộp được ngay
python scripts/03_rule_baseline.py --retrieval artifacts/retrieval_p1.jsonl --out artifacts/codegen_p1.jsonl
python scripts/05_build_submission.py --retrieval artifacts/retrieval_p1.jsonl --codegen artifacts/codegen_p1.jsonl --out-dir artifacts/submission_p1

# 3) Bộ eval offline (đo trước khi đốt GPU) — CHỈ dùng offline, TUYỆT ĐỐI KHÔNG NỘP
python scripts/09_gen_eval_suite.py --per-class 60
python scripts/02_retrieve.py --questions artifacts/eval/eval_questions.jsonl --out artifacts/eval/eval_retrieval.jsonl
python scripts/03_rule_baseline.py --retrieval artifacts/eval/eval_retrieval.jsonl --out artifacts/eval/eval_codegen.jsonl
python scripts/05_build_submission.py --retrieval artifacts/eval/eval_retrieval.jsonl --codegen artifacts/eval/eval_codegen.jsonl --out-dir artifacts/eval/eval_submission --offline-eval
python scripts/07_evaluate.py --submission artifacts/eval/eval_submission --gold artifacts/eval/eval_gold.json --by-class
```

> ### ⛔ KHÔNG BAO GIỜ NỘP BỘ EVAL LÊN LEADERBOARD
> Bộ eval gồm câu hỏi **tự sinh**, đánh id 1..N của riêng nó. Grader khớp theo id
> nên nộp nó = so đáp án của câu tự chế với câu thật ⇒ **toàn bộ metric = 0.0** và
> mất một lượt nộp trong ngày (đã xảy ra ở lần nộp #9).
>
> Từ nay `05_build_submission.py` tự phát hiện: nếu id/nội dung câu hỏi không khớp
> `data/ViFinQA/questions/questions.jsonl` thì in cảnh báo `[STOP]`, đặt tên file là
> `OFFLINE_EVAL_DO_NOT_UPLOAD.zip`, ghi `DO_NOT_UPLOAD.txt` và xoá `submission.zip` cũ.
> Chỉ file tên đúng `submission.zip` mới được nộp.

**Chưa chạy Qwen vội** — xem §6 để biết nên đốt GPU vào lúc nào.

---

## 1. Bước 1 — Extractive metric + shortlist dòng ứng viên

### File mới

| File | Vai trò |
|---|---|
| `vifinqa/router/metric_phrase.py` | Cắt câu hỏi xuống đúng cụm chỉ tiêu (extractive) thay vì trừ stopword |
| `vifinqa/retrieval/shortlist.py` | Xếp hạng **dòng** ứng viên trong các bảng đã lấy, render thành block cho prompt |

### File sửa

- `vifinqa/router/entities.py` — `Parsed` thêm `metric_wide`, `metric_variants`; dùng `extract_metric`, giữ phrase cũ làm variant dự phòng.
- `vifinqa/router/router.py` — `Route` mang `metric_variants`.
- `vifinqa/retrieval/retrieve.py` — BM25 query dùng **mọi** variant.
- `vifinqa/codegen/prompts.py` — thêm block `CANDIDATE ROWS` vào user prompt.
- `vifinqa/codegen/generate.py` — `QuestionBundle.shortlist()` + đưa shortlist vào prompt.
- `vifinqa/codegen/rule_codegen.py` — ngưỡng 78 → **62**; luôn trả lời khi vượt 62,
  nhưng đánh dấu `AMBIGUOUS` (hạ `confidence` xuống 60) khi hơn ứng viên nhì < 8
  điểm để `--rule-first` không bỏ qua LLM cho những câu đó.

**Hiệu chuẩn bằng chính bộ eval (lớp `lookup`, coverage × accuracy):**

| Cấu hình | Trả lời | Đúng | coverage×acc |
|---|---:|---:|---:|
| ngưỡng 78, không biên (bản cũ) | 24/25 | 17 | 0.680 |
| **ngưỡng 62, không biên (mặc định mới)** | **25/25** | **18** | **0.720** |
| ngưỡng 62 + biên 8 | 14/25 | 11 | 0.440 |
| ngưỡng 62 + biên 15 | 8/25 | 8 | 0.320 |

Bản đầu tôi đặt biên = 8 làm mặc định — bộ eval cho thấy **sai**: từ chối trả lời
tốn điểm nhiều hơn là thỉnh thoảng chọn nhầm dòng. Đây chính là giá trị của bước 2.

### Đo được (45 câu đơn giản trước đây rỗng)

| Chỉ số | Trước | Sau |
|---|---:|---:|
| Điểm ứng viên tốt nhất (median) | 54 | **72** |
| Có ≥1 ứng viên ≥78 | 2% | **31%** |
| Rule tự trả lời được | 0/45 | **11/45** |
| Có shortlist cho LLM | – | 44/45 |

**Trên 150 câu đầu của bộ thi (rule-only, không LLM):**
`rule 79 / none 71` → **`rule 123 / none 27`** (+44 câu có đáp án, 56 câu được
đánh dấu AMBIGUOUS để LLM xem lại).

**Trên bộ eval 125 câu:** `answer_acc` 0.088 → **0.136**; riêng lớp `lookup`
0.440 → **0.680**.

Ví dụ đã sửa: `"Số dư trả trước cho người bán..."` → metric `tra truoc cho nguoi ban`
khớp `"Trả trước cho người bán"` **100 điểm** (trước: 63, dưới ngưỡng nên bị bỏ).

---

## 2. Chuẩn hoá đơn vị output (BTC xác nhận: 90 chứ không phải 0.9)

### File mới: `vifinqa/codegen/units.py`

- `percent_from_cell(value, label, col_name)` — quy đổi ô sang **đơn vị phần trăm**.
- `cell_is_already_percent(...)` — chỉ coi là đã-phần-trăm khi có ký tự `%` hoặc
  |value| > 1.5. **Nhãn "Tỷ lệ..." KHÔNG phải bằng chứng về thang đo** (ô có thể
  chứa 0.9 hoặc 90) — đây là bẫy tôi đã mắc và test bắt được.
- `check_answer_unit(answer, output_type)` — cảnh báo khi giá trị vô lý
  (percent ∈ (0,1] → nghi là ratio; year không nguyên; count âm...).

### Nối vào

- `rule_codegen.py`: nhánh `output_type == "percent"` sinh query có `* 100` khi cần.
- `prompts.py`: mục **5b** liệt kê rõ quy ước cho từng `output_type`.
- `validation/evaluate.py`: đếm riêng `unit_ratio_mistakes` (trả 0.9 thay vì 90).

---

## 3. Bước 2 — Bộ eval offline đa lớp

### File mới

| File | Vai trò |
|---|---|
| `vifinqa/validation/gen_multiclass.py` | Sinh câu hỏi synthetic 6 lớp, đáp án tự tính từ store |
| `scripts/09_gen_eval_suite.py` | CLI |

Lớp: `lookup`, `growth_pct`, `ratio_pct`, `difference`, `ranking` (+ `percent_unit`).
Gold gồm `answer` (đúng đơn vị câu hỏi), `relevant_docs`, `relevant_tables`
(**line-number**, đúng quy ước BTC), `klass`, `output_type`.

`scripts/07_evaluate.py --by-class` in bảng theo lớp.

### Baseline đo được (125 câu, rule-only, chưa LLM)

| class | n | answer | exec | F2 |
|---|---:|---:|---:|---:|
| lookup | 25 | 0.440 | 0.440 | 0.556 |
| growth_pct | 25 | 0.000 | 0.000 | 0.354 |
| ratio_pct | 25 | 0.000 | 0.000 | 0.333 |
| difference | 25 | 0.000 | 0.000 | 0.277 |
| ranking | 25 | 0.000 | 0.000 | 0.221 |

Đúng như thiết kế: rule cố tình từ chối câu phức hợp. **Đây là thước đo để biết
LLM có thực sự cải thiện lớp phức hợp hay không — trước đây ta không đo được.**

---

## 4. Bước 3 — Decomposition + formula registry + dynamic evidence

### File mới

| File | Vai trò |
|---|---|
| `vifinqa/router/decompose.py` | `detect_op` (11 phép), `build_plan` (fact = ticker × năm × metric), `evidence_budget` |
| `vifinqa/codegen/formulas.py` | Registry công thức + mô tả tiếng Anh đưa vào prompt |

### File sửa

- `router.py` — sinh `plan`, `evidence_budget`; **mở rộng có mục tiêu** report cho
  từng fact (không phải `--expand-docs` đại trà đã thất bại).
- `retrieve.py` — `_apply_quota`: đảm bảo mỗi báo cáo đã khoá có tối thiểu số slot,
  tránh việc top-k dồn hết vào 1 công ty trong câu so sánh.
- `generate.py` — prompt có block `OPERATION` + `FACTS TO LOCATE`.
- `rule_codegen.py` — từ chối khi `plan.op != "lookup"` (nhường LLM).

### Đo được

| id | Câu | op | facts | budget | docs lấy được |
|---|---|---|---:|---:|---:|
| 179 | trả trước cho người bán | lookup | 1 | 4 | 1 |
| 759 | chênh lệch giữa 2 công ty | difference | 4 | 10 | **4** |
| 431 | biên lợi nhuận ngành BĐS 6 công ty | margin | 12 | 12 | **12** |

Với id=759 shortlist đã nêu đúng dòng của **cả hai** công ty (df10 và df1) — trước
đây chỉ có dữ liệu của một công ty nên câu này bất khả thi.

---

## 5. Bước 4 — Semantic matching BGE-M3 ở cấp dòng

### File mới

| File | Vai trò |
|---|---|
| `vifinqa/retrieval/dense.py` | `LabelEncoder` (BGE-M3) + cache vector, `collect_labels` |
| `scripts/10_build_label_index.py` | Dựng index (chạy Kaggle GPU hoặc local CPU qua đêm) |
| `kaggle/vifinqa-embed.ipynb` | Notebook encode-only |

**Thiết kế quan trọng:** `load_encoder` trả `None` khi thiếu `sentence-transformers`
hoặc thiếu cache → pipeline tự động chạy lexical, **không bao giờ crash**. Nhờ vậy
máy local không cần torch.

Index nằm ở `artifacts/store/label_index/` → tự động được `04_make_kaggle_payload.py`
copy và băm SHA-256 (không phải sửa manifest).

Bật bằng `--use-dense` (đã thêm vào `kaggle_codegen.py`, và có trong `run_signature`
nên không resume nhầm giữa lượt lexical và lượt dense).

---

## 6. Khi nào nên đốt GPU Kaggle (2 tài khoản chạy song song)

Nguyên tắc: **GPU chỉ dùng cho việc GPU thực sự cần**. Bước 1–3 hoàn toàn CPU và
đã cho kết quả đo được; đừng chạy Qwen trước khi chốt xong chúng.

### Thứ tự khuyến nghị

| Giai đoạn | Máy local (CPU) | Kaggle acc #1 | Kaggle acc #2 |
|---|---|---|---|
| **A. Ngay bây giờ** | retrieval_p1 + rule baseline + eval suite; **nộp ablation rule-only** (coverage 79→123/150, không tốn GPU) | *(nghỉ)* | *(nghỉ)* |
| **B. Sau khi A xong** | build payload v3 | **embed index** (`vifinqa-embed.ipynb`, ~15 phút) | **Qwen 7B lexical** (control, `--limit 0`) |
| **C. Sau khi B xong** | so 2 kết quả trên eval suite | **Qwen 7B + `--use-dense`** | **Qwen 14B lexical** (`--load-4bit`) |
| **D.** | chọn cấu hình tốt nhất | chạy full cấu hình thắng | chạy ablation `sub-k=7` |

Lý do xếp thế này:

1. **Ablation rule-only ở giai đoạn A tốn 0 GPU** nhưng đo được ngay bước 1+3 có
   nâng TABLES/DOCS/ANSWER không — vì rule mới tự trả lời thêm ~24% câu đơn giản.
2. **Embed index rẻ** (encode-only ~15 phút) và dùng lại được mãi → chạy sớm ở acc #1,
   trong khi acc #2 chạy Qwen control cùng lúc.
3. **Hai acc chạy song song phải khác đúng MỘT biến** (dense on/off hoặc 7B/14B),
   nếu không sẽ không quy được nguyên nhân.
4. Mỗi lượt Qwen ~4–6h với cấu hình `--n 1 --k 4 --max-tokens 256 --batch-size 4`;
   checkpoint 32 câu + `--time-budget-min 400` nên phiên chết vẫn giữ kết quả.

### 6.1 Rebuild payload — BẮT BUỘC, và phải up lên CẢ HAI acc

Payload phải dựng lại vì **hai** thứ đã đổi: (a) toàn bộ code P1 trong `vifinqa/`,
(b) `retrieval.jsonl` (router mới sinh `plan`/`metric_variants`/`evidence_budget`;
file cũ không có → notebook sẽ chạy pipeline cũ).

`04_make_kaggle_payload.py` mặc định đọc `artifacts/retrieval.jsonl`, nên hoặc
truyền `--retrieval`, hoặc chép đè cho gọn:

```powershell
# chép retrieval P1 thành file mặc định (khuyến nghị: 1 nguồn sự thật duy nhất)
copy artifacts\retrieval_p1.jsonl artifacts\retrieval.jsonl

# dựng payload (băm SHA-256 lại toàn bộ code + store + retrieval)
python scripts/04_make_kaggle_payload.py --dataset-id <user1>/vifinqa-payload
```

**Up lên acc #1** (acc đang có sẵn dataset → dùng `version`, không phải `create`):
```powershell
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P1: metric+shortlist+plan"
```

**Up lên acc #2.** Kaggle CLI đọc credential từ `%USERPROFILE%\.kaggle\kaggle.json`,
nên phải trỏ sang thư mục credential của acc #2 rồi `create` lần đầu:
```powershell
# một lần: tải kaggle.json cua acc #2 vào D:\kaggle_acc2\kaggle.json
$env:KAGGLE_CONFIG_DIR = "D:\kaggle_acc2"
python scripts/04_make_kaggle_payload.py --dataset-id <user2>/vifinqa-payload
kaggle datasets create -p artifacts\kaggle_payload --dir-mode zip      # lần đầu
# các lần sau:
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P1 refresh"
Remove-Item Env:\KAGGLE_CONFIG_DIR      # trả lại acc #1
```

Lưu ý: `--dataset-id` ghi vào `dataset-metadata.json`, nên **chạy lại
`04_make_kaggle_payload.py` với id của acc tương ứng trước mỗi lần up**, nếu không
CLI sẽ đẩy nhầm sang acc kia. Kiểm nhanh: `type artifacts\kaggle_payload\dataset-metadata.json`.

Không dùng CLI thì vào kaggle.com → Datasets → New Dataset → kéo thả thư mục
`artifacts\kaggle_payload` (làm 2 lần, mỗi acc một lần).

### 6.2 Lệnh chạy

**Acc #1 — embed index (làm trước, ngắn ~15 phút):**
```
# notebook kaggle/vifinqa-embed.ipynb -> tải label_index/ về
# local: chép vào artifacts\store\label_index\  rồi dựng + up lại payload (6.1)
```

**Acc #2 — Qwen control (lexical, KHÔNG dense):**
```
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit \
    --out /kaggle/working/codegen_p1_lexical.jsonl \
    --n 1 --k 4 --max-tokens 256 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400
```

**Giai đoạn C — bật dense (chỉ đổi đúng 1 biến):**
```
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit --use-dense \
    --out /kaggle/working/codegen_p1_dense.jsonl \
    --n 1 --k 4 --max-tokens 256 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400
```

Cả hai lượt đều cần `pip install -q sentence-transformers` khi dùng `--use-dense`.

---

## 7. Ranh giới bằng chứng

- **Đo trên corpus thật:** điểm shortlist 54→72, rule +11/45, budget/plan của id
  179/759/431, baseline eval theo lớp.
- **Test-confirmed:** 96 test (thêm `test_p1_pipeline.py` 20 test, `test_to_expression.py` 11 test).
- **CHƯA xác nhận:** ảnh hưởng lên leaderboard của bước 1–4; chất lượng BGE-M3 trên
  nhãn tiếng Việt (chưa chạy encode thật); thời gian thực của lượt Qwen mới.
- **Rủi ro đã biết:** hạ ngưỡng rule 78→62 có thể tạo thêm đáp án SAI (trước đây
  im lặng bỏ qua). Biên "hơn ứng viên nhì ≥8" là để chặn việc này, nhưng cần theo
  dõi `answer_acc` lớp `lookup` trên eval suite trước/sau.
