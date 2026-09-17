# RUNBOOK — sổ tay lệnh ViFinQA

> **File này là nguồn sự thật duy nhất về LỆNH CHẠY.** Mỗi khi pipeline đổi,
> ghi đè trực tiếp vào đây (đừng tạo file mới) để không bị lạc phiên bản.
>
> Cập nhật lần cuối: **2026-08-27** — leaderboard xác nhận exact growth v28
> ở execution 0.4644; direct-ratio challenger là batch tiếp theo.

---

## 0. Trạng thái hiện tại (đọc trước khi chạy bất cứ thứ gì)

| Thành phần | Phiên bản/giá trị đang dùng |
|---|---|
| `vifinqa` package | 0.2.0 |
| Payload schema | 2 (có `payload-manifest.json`, SHA-256) |
| `TABLE_POS_MODE` | `line` (BTC xác nhận: vị trí = **số dòng** của `<table>`) |
| `SUBMISSION_K` | 5 |
| Retrieval chuẩn hiện tại | `artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl` |
| Rule candidate hiện tại | direct-ratio challenger v29 (planned) |
| Điểm tốt nhất đã nộp | TABLES_F2 .5530 / DOCS_F2 .9420 / ANSWER .4644 / EXEC .4644 |
| Checkpoint codegen | `artifacts/codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl` |
| Checkpoint submission | `artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip` |
| Candidate chờ nộp | chưa có; audit direct ratio trên checkpoint V28 |
| Backend LLM Kaggle | `hf` (transformers). **vLLM không chạy trên T4** |

V23 audit toàn bộ 383 câu LLM single-vote, thay đúng 15 lookup exact và giữ
nguyên 997 câu còn lại. Leaderboard xác nhận execution tăng
`0.4150 -> 0.4407`, tương đương khoảng 13 câu đúng ròng. Dùng codegen V23 làm
checkpoint. V24 thay bảy câu difference và tăng thêm `0.0020`; V25 thay năm
câu direct ranking và tăng tiếp `0.0039`. V26 audit lại toàn checkpoint, thay
15 lookup có canonical row/column/period evidence chính xác.

`submission_formula_consensus_safe2_w010` là ablation thất bại: hai candidate
`3/3` (ID 24, 996) đều không tăng Execution. Không dùng consensus/confidence
làm verifier nếu chưa kiểm tra exact child-row và qualifier evidence.

**Kiểm tra nhanh trạng thái trước mỗi phiên làm việc:**

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
.venv\Scripts\activate
python -c "import json;d=[json.loads(l) for l in open('artifacts/retrieval.jsonl',encoding='utf-8')];print('retrieval:',len(d),'| co plan P1:', 'plan' in d[0]['route'])"
python -m pytest tests -q
```

`co plan P1: False` ⇒ retrieval là bản CŨ, phải chạy lại §2 trước khi làm gì khác.

---

## 0bis. ĐỔI GÌ THÌ PHẢI CHẠY LẠI GÌ (tra bảng này trước khi chạy)

| Sửa file ở... | 01 store | 02 retrieve | 03 rule | 04+upload payload | chạy Kaggle |
|---|:--:|:--:|:--:|:--:|:--:|
| `extraction/`, `utils/viet_num.py` | ✅ | ✅ | ✅ | ✅ | ✅ |
| `router/`, `retrieval/` | – | ✅ | ✅ | ✅ | ✅ |
| `codegen/rule_*.py`, `units.py` | – | – | ✅ | ✅ | ✅ |
| `codegen/prompts.py`, `selection.py`, `generate.py`, `llm_client.py` | – | – | – | ✅ | ✅ |
| `submission/build.py`, `scripts/05` | – | – | – | – | – |
| chỉ `tests/`, `*.md` | – | – | – | – | – |

Payload luôn phải dựng lại khi **bất kỳ** file nào trong `vifinqa/` hoặc
`kaggle/kaggle_codegen.py` đổi, vì manifest băm SHA-256 toàn bộ code.

### Riêng bản P2.0 (`--llm-mode select`) — đang ở trạng thái này

Chỉ đụng `codegen/{selection,prompts,generate}.py` + `kaggle_codegen.py`.
Đã kiểm chứng: rule baseline trước/sau **giống hệt 150/150 bản ghi**.

```powershell
# BỎ QUA 01, 02, 03 — store, retrieval.jsonl, codegen_results.jsonl vẫn dùng được
python scripts/04_make_kaggle_payload.py --dataset-id <user1>/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P2.0 selection mode"

# acc #2
$env:KAGGLE_CONFIG_DIR = "D:\kaggle_acc2"
python scripts/04_make_kaggle_payload.py --dataset-id <user2>/vifinqa-payload
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P2.0 selection mode"
Remove-Item Env:\KAGGLE_CONFIG_DIR
```

Rồi chạy notebook theo §7bis với `--llm-mode select`.

## 1. Cài đặt (một lần)

```powershell
cd D:\Python_Project\Hackathon\R2AI_2026
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install pytest            # để chạy test
pip install kaggle            # để up payload bằng CLI
```

---

## 2. Pipeline local (CPU) — chạy theo đúng thứ tự này

### 2.1 Build store (chỉ chạy lại khi đổi code `extraction/`)

```powershell
python scripts/01_build_store.py
```

- ~30 phút cho 1.973 báo cáo. Ra `artifacts/store/`.
- Smoke: `python scripts/01_build_store.py --tickers VNM,VJC,ACB`
- **Bắt buộc chạy lại nếu** sửa `html_tables.py`, `report_parser.py`, `build_store.py`,
  hoặc `viet_num.py`.

### 2.2 Retrieval (chạy lại khi đổi `router/` hoặc `retrieval/`)

```powershell
python scripts/02_retrieve.py --out artifacts/retrieval.jsonl
```

- ~5 phút. **Đây là bước hay bị quên nhất** — mọi thay đổi router/metric/plan
  đều nằm trong file này.
- Smoke: thêm `--limit 150`.

### 2.3 Rule baseline (không cần GPU)

```powershell
python scripts/03_rule_baseline.py --retrieval artifacts/retrieval.jsonl --out artifacts/codegen_results.jsonl
```

Kỳ vọng hiện tại: có cả `rule` (lookup) lẫn `rule_composite`
(growth/difference/ratio/ranking). Trên 150 câu đầu: `rule 123 / rule_composite 5 / none 22`.

Nếu thấy `rule_composite 0` trên toàn bộ 1.012 câu ⇒ retrieval là bản cũ
(thiếu `plan`), quay lại §2.2.

### 2.4 Build submission

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen artifacts/codegen_results.jsonl --out-dir artifacts/submission
```

Nộp file: `artifacts\submission\submission.zip`

Log phải thấy: `expression-form check: all 1012 queries eval-compilable`.
Nếu thấy `[WARN] ... NOT single expressions` → xem §6.3.

**Biến thể (mỗi lần nộp chỉ đổi MỘT biến):**

```powershell
# ablation k=7
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen artifacts/codegen_results.jsonl --out-dir artifacts/submission_k7 --sub-k 7

# chỉ nộp các id trong file câu hỏi chính thức (nếu BTC phát bộ riêng)
python scripts/05_build_submission.py ... --questions <duong_dan>\questions.jsonl
```

**KHÔNG dùng `--expand-docs`** — đã kiểm chứng làm sập DOCS_F2 (.84 → .61).

---

## 3. Bộ eval offline (đo trước khi tốn GPU / lượt nộp)

> ⛔ **KHÔNG BAO GIỜ NỘP BỘ EVAL.** Câu hỏi tự sinh, id riêng ⇒ leaderboard 0.0
> và mất một lượt nộp (đã dính lần #9). Luôn dùng `--offline-eval`.

```powershell
python scripts/09_gen_eval_suite.py --per-class 60
python scripts/02_retrieve.py --questions artifacts/eval/eval_questions.jsonl --out artifacts/eval/eval_retrieval.jsonl
python scripts/03_rule_baseline.py --retrieval artifacts/eval/eval_retrieval.jsonl --out artifacts/eval/eval_codegen.jsonl
python scripts/05_build_submission.py --retrieval artifacts/eval/eval_retrieval.jsonl --codegen artifacts/eval/eval_codegen.jsonl --out-dir artifacts/eval/eval_submission --offline-eval
python scripts/07_evaluate.py --submission artifacts/eval/eval_submission --gold artifacts/eval/eval_gold.json --by-class
```

**Baseline rule-only hiện tại (300 câu) — mốc so sánh cho mọi thay đổi sau này:**

| class | n | answer | exec | F2 |
|---|---:|---:|---:|---:|
| lookup | 60 | 0.783 | 0.783 | 0.556 |
| ratio_pct | 60 | 0.400 | 0.400 | 0.306 |
| growth_pct | 60 | 0.383 | 0.383 | 0.350 |
| difference | 60 | 0.283 | 0.283 | 0.329 |
| ranking | 60 | 0.133 | 0.133 | 0.332 |
| **tổng** | **300** | **0.3967** | **0.3967** | **0.374** |

(trước rule composite: 0.157 tổng, 4 lớp phức hợp đều 0.000)

`ranking` thấp là do phải đúng **cả 4** công ty trong một câu; suy ra độ chính xác
mỗi fact ≈ 0.6.

Nếu lỡ tạo `submission.zip` trong thư mục eval: xoá tay
`artifacts\eval\eval_submission\submission.zip`.

### 3.1 Formula eval v2 (bắt buộc trước khi sửa solver/retrieval phức hợp)

Bộ này dùng `2.707` fact được xác minh bằng mã VAS + canonical label + cột năm
tường minh. Với chỉ tiêu stock, gold bắt buộc lấy cột closing `31/12`, không lấy
opening/interim cùng năm. Có 288 câu, cân bằng 24 câu cho mỗi lớp công thức.

```powershell
python scripts/12_gen_formula_eval.py --per-class 24
python scripts/02_retrieve.py --questions artifacts/formula_eval/formula_questions.jsonl --out artifacts/formula_eval/formula_retrieval_operand_cached.jsonl --depth 20 --row-rerank --row-score-weight 0.10
python scripts/03_rule_baseline.py --retrieval artifacts/formula_eval/formula_retrieval_operand_cached.jsonl --out artifacts/formula_eval/formula_codegen_operand_cached_k15.jsonl --k 15
python scripts/05_build_submission.py --retrieval artifacts/formula_eval/formula_retrieval_operand_cached.jsonl --codegen artifacts/formula_eval/formula_codegen_operand_cached_k15.jsonl --out-dir artifacts/formula_eval/formula_submission_operand_cached_sub5 --sub-k 5 --offline-eval
python scripts/07_evaluate.py --submission artifacts/formula_eval/formula_submission_operand_cached_sub5 --gold artifacts/formula_eval/formula_gold.json --by-class
python scripts/13_audit_formula_eval.py --retrieval artifacts/formula_eval/formula_retrieval_operand_cached.jsonl --codegen artifacts/formula_eval/formula_codegen_operand_cached_k15.jsonl --out artifacts/formula_eval/formula_audit_operand_cached_k15.json --k 15
```

Mốc rule-only đầu tiên:

| class | Exec | evidence recall@15 | all evidence@15 |
|---|---:|---:|---:|
| growth_pct | 1.000 | 1.000 | 1.000 |
| gross_margin | 1.000 | 1.000 | 1.000 |
| average_margin_change | 1.000 | 1.000 | 1.000 |
| margin_change | 0.917 | 1.000 | 1.000 |
| debt_equity | 0.875 | 1.000 | 1.000 |
| margin_difference | 0.833 | 1.000 | 1.000 |
| ranking_ratio | 0.833 | 0.951 | 0.792 |
| count_years | 0.667 | 0.694 | 0.583 |
| count_threshold | 0.125 | 0.427 | 0.000 |
| temporal_count | 0.125 | 0.670 | 0.083 |
| count_multi_condition | 0.000 | 0.356 | 0.000 |
| nested_ranking | 0.000 | 0.604 | 0.000 |
| **tổng** | **0.6146** | | |

Mốc hiện tại sau operand requirements, exact row identity, greedy set-cover và
formula fallback guard:

| class | Exec |
|---|---:|
| average_margin_change | 1.000 |
| count_multi_condition | 0.125 |
| count_threshold | 0.917 |
| count_years | 1.000 |
| debt_equity | 0.875 |
| gross_margin | 1.000 |
| growth_pct | 0.958 |
| margin_change | 0.917 |
| margin_difference | 0.833 |
| nested_ranking | 0.292 |
| ranking_ratio | 1.000 |
| temporal_count | 0.333 |
| **tổng** | **0.7708** |

TABLES_F2 offline ở `sub-k=5` là `0.7732` (precision `0.6249`, recall
`0.9099`). So với mốc đầu, Execution tăng `+0.1562`. Đây vẫn là synthetic
benchmark; leaderboard chính thức vẫn `0.2866` cho đến khi chạy payload dưới đây.

`formula_audit_k15.json` tách riêng retrieval coverage, solver coverage,
answer accuracy khi solver đã trả lời và ba lý do từ chối phổ biến theo lớp.
Kết quả này chỉ là synthetic benchmark, tuyệt đối không upload file
`OFFLINE_EVAL_DO_NOT_UPLOAD.zip` lên leaderboard.

---

## 4. Kaggle — payload

### 4.1 Khi nào phải rebuild payload

Rebuild nếu **bất kỳ** thứ nào sau đây đổi: code trong `vifinqa/`,
`kaggle/kaggle_codegen.py`, `artifacts/store/`, `artifacts/retrieval.jsonl`.

### 4.2 Dựng payload

```powershell
python scripts/04_make_kaggle_payload.py --dataset-id <user1>/vifinqa-payload
python scripts/04_make_kaggle_payload.py --dry-run        # kiểm tra trước, không ghi đè
```

Ra `artifacts\kaggle_payload\` (~100 MB) + `payload-manifest.json`.

Payload formula-aware hiện tại:

```powershell
python scripts/02_retrieve.py --out artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl --depth 20 --row-rerank --row-score-weight 0.10
python scripts/03_rule_baseline.py --retrieval artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl --out artifacts/codegen_rule_canonical_v2_formula_operand_exact_w010_k15.jsonl --k 15
python scripts/04_make_kaggle_payload.py --store-dir artifacts/store --retrieval artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl --out artifacts/kaggle-payload-formula-operand-exact-w010 --dataset-slug kaggle-payload-formula-operand-exact-w010 --dataset-id kien2005/kaggle-payload-formula-operand-exact-w010
```

Upload folder/ZIP `artifacts/kaggle-payload-formula-operand-exact-w010`, attach
dataset đó rồi import notebook
`r2ai-qwen2-5-coder-7b-formula-operand-exact-w010.ipynb`. Full run dùng
`--llm-mode select --llm-target all --k 15`; tải file
`codegen_formula_operand_sel7b_k15.jsonl` sau khi cell kiểm tra cuối cùng pass.

### 4.3 Up lên acc #1

```powershell
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "P1 metric+shortlist+plan"
```

(Lần đầu tiên với một acc: đổi `version` → `create`.)

### 4.4 Up lên acc #2

```powershell
# một lần: tải kaggle.json của acc #2 vào D:\kaggle_acc2\kaggle.json
$env:KAGGLE_CONFIG_DIR = "D:\kaggle_acc2"
python scripts/04_make_kaggle_payload.py --dataset-id <user2>/vifinqa-payload   # PHẢI chạy lại, id nằm trong metadata
kaggle datasets create -p artifacts\kaggle_payload --dir-mode zip               # lần đầu
kaggle datasets version -p artifacts\kaggle_payload --dir-mode zip -m "refresh" # các lần sau
Remove-Item Env:\KAGGLE_CONFIG_DIR
```

**Bẫy:** `--dataset-id` được ghi vào `dataset-metadata.json`; nếu không chạy lại
`04_...` trước khi đổi acc, CLI sẽ đẩy nhầm acc. Kiểm tra:

```powershell
type artifacts\kaggle_payload\dataset-metadata.json
```

---

## 5. Kaggle — chạy notebook

### 5.1 Codegen (Qwen)

Notebook: `kaggle/vifinqa-codegen.ipynb` → File → Import Notebook.
Settings: **GPU T4 x2**, **Internet On**, Add Input = dataset payload (chỉ MỘT bản).

Log đúng phải có: `payload verified: schema=2` → `run signature: ...` →
`baseline written (...)` → `LLM queue: ...` → `[chunk 1/N] ...`

Cấu hình đang khuyến nghị (7B, 4-bit):

```
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit \
    --out /kaggle/working/codegen_results.jsonl \
    --n 1 --k 4 --max-tokens 256 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400
```

Biến thể:

| Mục đích | Thêm/đổi cờ |
|---|---|
| Bật semantic matching | `--use-dense` (cần `pip install -q sentence-transformers` + `store/label_index/`) |
| Model 14B | `--model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit --batch-size 2` |
| Bỏ qua câu rule đã chắc | `--rule-first` |
| Chạy tiếp phiên trước | đặt file cũ vào `/kaggle/working/codegen_results.jsonl`, giữ NGUYÊN mọi cờ (khác cờ ⇒ `run_signature` khác ⇒ không resume) |
| OOM | `--batch-size 2`, `--k 3`, `--max-tokens 192` |

Tải `codegen_results.jsonl` từ tab Output → về local:

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <duong_dan>\codegen_results.jsonl --out-dir artifacts/submission_qwen
```

### 5.2 Embed label index (encode-only, ~15 phút)

Notebook: `kaggle/vifinqa-embed.ipynb`. GPU T4 x1 là đủ.
Tải `label_index/` → chép vào `artifacts\store\label_index\` → rebuild payload (§4).

---

## 6. Sửa lỗi thường gặp

### 6.1 Leaderboard trả 0.0 toàn bộ
Đã nộp nhầm bộ eval offline. Chỉ nộp file tên đúng `submission.zip` được sinh từ
`artifacts/retrieval.jsonl` (câu hỏi thật).

### 6.2 EXEC thấp hơn ANSWER
Có `pandas_query` nhiều dòng → grader `eval` báo SyntaxError. Sửa không cần GPU:

```powershell
python scripts/08_repair_expressions.py --submission artifacts/submission --out-dir artifacts/submission_fixed
python scripts/08_repair_expressions.py --codegen artifacts/codegen_results.jsonl   # sửa luôn ở gốc
```

### 6.3 `[WARN] ... NOT single expressions` khi build
Chạy `08_repair_expressions.py --codegen ...` rồi build lại.

### 6.4 Kaggle: `Engine core initialization failed`
vLLM V1 không chạy trên T4. Dùng `--backend hf` (mặc định notebook).

### 6.5 Kaggle: payload hash mismatch / schema
Payload cũ hoặc attach nhiều bản. Gỡ hết input, up lại theo §4, attach đúng 1 bản.

### 6.6 Windows in ra ký tự lỗi
Mọi script đã gọi `setup_stdout()` (UTF-8). Nếu vẫn lỗi: `chcp 65001`.

---

## 7. Nhật ký nộp bài (cập nhật sau mỗi lần nộp)

| # | Cấu hình | TABLES_F2 | DOCS_F2 | ANSWER | EXEC |
|---|---|---:|---:|---:|---:|
| 3 | pos=line, k=10, rule | .3641 | .8399 | .085 | .085 |
| 4 | + expand-docs ❌ | .3641 | .6066 | .085 | .085 |
| 5 | pos=line, k=5, rule | .4092 | .8093 | .085 | .085 |
| 6 | + Qwen 7B (query nhiều dòng ❌) | .4092 | .8093 | .1047 | .0613 |
| 8 | **P1 rule-only** | **.4241** | **.8628** | **.1285** | **.1285** |
| 9 | nhầm nộp bộ eval ❌ | 0 | 0 | 0 | 0 |
| 10 | **P1.5 rule composite** | **.4337** | **.8777** | **.1542** | **.1542** |
| 11 | P1.6 hợp nhất scoring | .4337 | .8777 | .1522 | .1522 |
| 12 | Qwen 7B `--llm-target empty` | .4334 | .8774 | **.1561** | .1561 |
| _ | _(điền lần nộp tiếp theo)_ | | | | |

**#12: 175 đáp án LLM mới → chỉ ~+2 câu đúng (≈2%).** Nguyên nhân đã audit:
35% query không lọc cột năm, 15% quên chia ANSWER_SCALE, 90% thiếu `regex=False`.
Chi tiết + hướng sửa: `CLAUDE.md` mục "CHẨN ĐOÁN LƯỢT QWEN #12".

**#11 trung tính** (−0.002 = 1 câu/506): bỏ 24 đáp án nhưng trong đó chỉ ~1 câu
đúng. Eval dự báo lookup +0.10 nhưng thực tế ≈ 0 ⇒ **lần thứ hai eval dự báo sai**.
Rule đã tới điểm lợi tức giảm dần: P1.5 +0.026, P1.6 −0.002.

## 7bis. Chạy Qwen song song 2 acc (giai đoạn hiện tại)

Điều kiện: đã chạy §2.2 + §2.3 với code P1.6 và rebuild payload (§4).

| | Acc #1 | Acc #2 |
|---|---|---|
| Notebook | `vifinqa-codegen.ipynb` | `vifinqa-codegen.ipynb` |
| Model | `Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit` | `Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit --batch-size 2` |
| Out | `codegen_qwen7b.jsonl` | `codegen_qwen14b.jsonl` |
| Khác nhau đúng 1 biến | kích thước model | |

**`--llm-target` quyết định GPU tiêu vào đâu** (đo trên artifact hiện tại):

| target | số câu | ước tính 7B | ghi chú |
|---|---:|---:|---|
| `all` | 1012 | ~5.0h | phủ hết, nhưng phần lớn rule đã trả lời đúng |
| `weak` | 787 | ~3.9h | rỗng + rule confidence < 78 |
| **`empty`** | **569** | **~2.8h** | **ROI cao nhất**: chỉ câu rule bó tay |

Arbitration giữ đáp án rule khi bất đồng mà rule tự tin, nên chạy `all` phần lớn
là tiêu GPU vào việc xác nhận lại thứ đã đúng.

### `--llm-mode select` — DÙNG CÁI NÀY từ nay

Sau lượt #12 (model tự viết pandas): 35% query không lọc cột năm, 15% quên chia
`ANSWER_SCALE`, 90% thiếu `regex=False` → 175 đáp án mới chỉ ra ~2 câu đúng.

Ở `--llm-mode select`, model **chỉ xuất JSON** `{"op":..., "operands":[...]}`
chọn dòng trong shortlist; **ta** sinh biểu thức pandas với cột và đơn vị đúng.
Mô phỏng end-to-end: query thiếu `regex=False` **0%**, query không lọc cột **0%**.
Output ~30 token thay vì 256 → nhanh hơn nhiều, chạy được `--llm-target all`.

```
# ACC #1 — 7B, selection mode, TOÀN BỘ câu (arbitration mới có việc để làm)
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-7B-Instruct --load-4bit \
    --llm-mode select --llm-target all \
    --out /kaggle/working/codegen_sel7b.jsonl \
    --n 1 --k 4 --max-tokens 96 --batch-size 8 \
    --checkpoint-every 32 --time-budget-min 400

# ACC #2 — 14B, khác đúng MỘT biến (kích thước model)
!python /kaggle/working/code/kaggle_codegen.py --payload $PAYLOAD --backend hf \
    --model Qwen/Qwen2.5-Coder-14B-Instruct --load-4bit \
    --llm-mode select --llm-target all \
    --out /kaggle/working/codegen_sel14b.jsonl \
    --n 1 --k 4 --max-tokens 96 --batch-size 4 \
    --checkpoint-every 32 --time-budget-min 400
```

`--max-tokens 96` là đủ cho một object JSON; `--batch-size` tăng được vì output ngắn.

Chế độ cũ `--llm-mode code` vẫn giữ để đối chứng, nhưng không khuyến nghị.

Nếu phiên còn dư thời gian sau khi xong `empty`: chạy lại **cùng lệnh** đổi
`--llm-target weak` — resume sẽ giữ nguyên phần đã làm và chỉ bổ sung phần yếu.

Về local, mỗi file build một submission riêng rồi so:

```powershell
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <path>\codegen_qwen7b.jsonl  --out-dir artifacts/submission_q7
python scripts/05_build_submission.py --retrieval artifacts/retrieval.jsonl --codegen <path>\codegen_qwen14b.jsonl --out-dir artifacts/submission_q14
```

Kiểm nhanh Qwen có thực sự thêm giá trị (trước khi tốn lượt nộp):

```powershell
python -c "import json;from collections import Counter;r=[json.loads(l) for l in open(r'<path>\codegen_qwen7b.jsonl',encoding='utf-8')];print(Counter(x['source'] for x in r));print('arbitration:',Counter((x.get('arbitration') or {}).get('reason','n/a') for x in r))"
```

Đọc kết quả: `rule and llm agree` nhiều = tín hiệu tốt; `disagree; rule weak -> llm`
là phần LLM thực sự đóng góp; `source=none` còn nhiều = LLM cũng bó tay.

Ở lượt #12 toàn bộ 183 bản ghi đều là `rule produced nothing` — vì
`--llm-target empty` khiến LLM chỉ thấy câu rule bó tay nên **arbitration chưa
từng chạy**. Với `--llm-target all` sẽ thấy đủ 4 nhóm lý do.

## 8. Hiệu chuẩn eval offline ↔ leaderboard (QUAN TRỌNG)

| | eval offline | leaderboard |
|---|---:|---:|
| trước P1.5 | 0.157 | 0.1285 |
| sau P1.5 | 0.3967 | 0.1542 |
| mức tăng | **+153%** | **+20%** |

**Bộ eval phóng đại mức cải thiện khoảng 7 lần.** Hai lý do: (a) câu synthetic
dùng đúng 6 mã VAS mà rule engine biết; (b) phân bố lớp khác thực tế
(eval chia đều 20% mỗi lớp; bộ thi thật: lookup 46%, ranking 21%).

⇒ Dùng eval để biết **hướng** (tăng hay giảm), KHÔNG dùng để dự đoán **mức**.
Chỉ leaderboard mới là trọng tài về độ lớn.

## 9. Bản đồ khoảng trống hiện tại (đo trên artifact #10)

467/1012 câu có đáp án (46%); độ chính xác trên phần đã trả lời ≈ 33%.

| op | số câu | % rỗng | ghi chú |
|---|---:|---:|---|
| lookup | 468 | 25% | 118 câu — lỗ hổng dễ lấp nhất |
| **ranking** | **214** | **95%** | 203 câu — lớn nhất, nhưng là **composite lồng nhau** |
| difference | 147 | 66% | 97 câu |
| average | 70 | 91% | 64 câu |
| growth_pct | 50 | 68% | 34 câu |
| ratio | 38 | 37% | 14 câu |

**Vì sao `ranking` gần như rỗng hoàn toàn:** câu thật lồng 3 tầng, ví dụ
*"Trong nhóm HPG, HSG, MSR và NKG, doanh nghiệp có mức tăng lớn nhất từ 2023
sang 2024 của tỷ lệ X"* = rank(growth(ratio)). Rule engine hiện chỉ giải một
tầng. Bộ eval của tôi chỉ có ranking một tầng nên không lộ ra điều này.

## 10. Candidate schema-linking v3 w010 (2026-08-21)

Checkpoint chính thức vẫn là execution `0.2866`. Candidate này đã pass offline
nhưng chưa có kết quả leaderboard.

1. Upload file sau thành Kaggle Dataset, slug phải là
   `kaggle-payload-schema-linking-v3-w010`:

```text
artifacts/kaggle-payload-schema-linking-v3-w010.zip
```

2. Import và chạy notebook:

```text
r2ai-qwen2-5-coder-7b-schema-linking-v3-w010.ipynb
```

Notebook dùng input path:

```text
/kaggle/input/datasets/kien2005/kaggle-payload-schema-linking-v3-w010
```

Giữ GPU `T4 x2`, Internet On, và chạy toàn bộ cells. Cấu hình A/B được giữ
nguyên so với consensus trước: Qwen 2.5 Coder 7B, selection mode, empty-only,
`n=3`, consensus 2/3, `k=15`, temperature `0.35`, seed `13`.

3. Tải output sau về và đặt trong một folder kết quả mới:

```text
/kaggle/working/codegen_schema_linking_v3_consensus_empty_sel7b_k15_n3.jsonl
```

Output đã nhận tại
`kaggle/results-10/codegen_schema_linking_v3_consensus_empty_sel7b_k15_n3.jsonl`.
Không nộp file raw: nó làm 177 câu đang chạy được chuyển thành `failed`.

Hai override `safe2` cũ của `24` và `996` vẫn bị cấm vì chúng dùng sai
row cha. Schema-linking v3 đã sinh lại hai câu này bằng row con chính
xác; bản audited mới chỉ nhận `24,440,562,574,996` sau khi truy ngược
từng ô dữ liệu.

File cần nộp leaderboard:

```text
artifacts/submission_result10_schema_linking_v3_audited5_w010/submission.zip
```

SHA-256: `d229bf51a5d3e368ea0a33defbd7bdad9b6b15739e310833aacdbca0bcd47eea`.
Leaderboard của file này trả về `EXECUTION_ACCURACY=0.2866`, không tăng.
Với 506 câu được chấm, chỉ một câu đúng mới đã phải đẩy điểm
lên khoảng `0.2885`; do đó audited5 không tạo gain quan sát được. Không
nộp lại artifact này. Checkpoint chính thức tiếp tục là `0.2866`:

```text
artifacts/submission_formula_operand_retrieval_best2866_answers_w010/submission.zip
```

## 11. Checkpoint kết hợp chính thức 0.3004

Artifact từ nhánh `tranhuy` đã được leaderboard xác nhận:

```text
artifacts/tranhuy_02964_stockflow_exact_v11_oldretr_k5/submission.zip
```

Gói này đạt `EXECUTION_ACCURACY=0.2964`, cao hơn checkpoint cũ `0.2866`
đúng năm câu trên 506 câu được chấm. Tên thí nghiệm bên nguồn vẫn chứa
`02925`; dùng SHA-256 sau để nhận diện đúng file đã chấm:

```text
6773532be7dbe038310b9a0663090ebf17878111a8b27b53403cd24de5f23e9a
```

Không cần rerun Qwen. Bản kết hợp đã lấy nguyên answer, pandas query và
evidence của gói `0.2964`, rồi rebuild với formula-operand retrieval `w=0.10`:

```text
artifacts/submission_tranhuy_02964_answers_formula_operand_retrieval_w010/submission.zip
```

SHA-256:

```text
17750e0290d181ce13d35ef87b589ee6bdf9bcf9fa4831a6c749e8744e8e93b7
```

Offline validation: đủ 1.012 ID; answer/query/evidence giống gói `0.2964` ở
toàn bộ 1.012 câu; mọi query compile và replay khớp answer; ZIP gồm 1.553 CSV
và pass `unzip -t`. Leaderboard xác nhận kết quả kết hợp:

```text
TABLES_F2MACRO      0.4901
DOCS_F2MACRO        0.9079
EXECUTION_ACCURACY  0.2964
```

So với gói Trần Huy dùng old retrieval, TABLES_F2 tăng `0.0138`, DOCS_F2 tăng
`0.0149` và execution được giữ nguyên. So với checkpoint retrieval tốt nhất
trước đó, TABLES_F2 tăng thêm `0.0009`, DOCS_F2 tăng thêm `0.0011`, đồng thời
execution tăng năm câu đúng. Đây là checkpoint chính thức cần dùng làm nền
cho mọi thử nghiệm tiếp theo.

Batch canonical-v2 restore4, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_02964_plus_canonical_restore4_w010/submission.zip
```

SHA-256:

```text
cdeed91318c8d6eee2cc009ca4b42e5f67acf547c5072361dfb5cbae3107a7e4
```

Bản này phục hồi theo batch bốn canonical-v2 answer `589,591,635,838` mà
codegen Trần Huy đã chuyển từ `ok` về `failed`. So với checkpoint chính thức,
nó chỉ đổi bốn answer/query/evidence, không đổi relevant docs và chỉ đổi
relevant tables ở ID `838`. Toàn bộ 1.012 query replay thành công và ZIP pass
kiểm tra cấu trúc. Leaderboard xác nhận kết quả mới:

```text
TABLES_F2MACRO      0.4902
DOCS_F2MACRO        0.9079
EXECUTION_ACCURACY  0.3004
```

Batch bốn câu mang lại hai câu đúng ròng trên tập 506 câu chấm, tương đương
`+0.0040` execution. TABLES_F2 tăng nhẹ `0.0001`, DOCS_F2 giữ nguyên. Đây là
checkpoint chính thức dùng làm nền cho mọi thử nghiệm tiếp theo.

## 12. Checkpoint deterministic year-ranking exact10: 0.3083

Không cần rerun Qwen cho batch này. Solver deterministic đã xử lý các câu hỏi
trả về năm bằng phép `argmax/argmin` trên cùng một metric qua toàn bộ các năm.
Nó chỉ fill những câu đang `failed` trong checkpoint `0.3004`.

File duy nhất cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03004_plus_yearrank_exact10_w010/submission.zip
```

SHA-256:

```text
e4aa4fd2eb797e3274d4e5b9869b52c8bae09a7e4769e1f39393d21ad4898203
```

Allowlist gồm 10 ID:

```text
842,850,878,884,889,904,948,960,997,1000
```

Các guard trước khi build đều đã pass: đủ 1.012 ID, chỉ 10 answer/query thay
đổi so với checkpoint `0.3004`, mọi query compile và replay đúng answer trên
CSV đóng gói, `relevant_tables` dùng line number 1-based chính thức và ZIP pass
`unzip -t`. Không nộp artifact rule thô; chỉ nộp `submission.zip` ở trên.

Leaderboard xác nhận `ANSWER_ACCURACY=EXECUTION_ACCURACY=0.3083`, tăng bốn câu
đúng ròng trên 506 câu so với `0.3004`. `TABLES_F2MACRO=0.4902` và
`DOCS_F2MACRO=0.9079` giữ nguyên, nên gain đến hoàn toàn từ logic answer. Đây
là checkpoint chính thức mới và là base cho canonical note dictionary của 23
câu year-ranking còn lại.

## 13. Checkpoint canonical note v3 + year-ranking v2 exact13: 0.3202

Batch này không cần rerun Qwen. Canonical note dictionary v3 và direct
year-ranking resolver v2 đã xử lý 13/16 câu đơn giản còn lại bằng query
deterministic, rồi merge fill-only trên checkpoint `0.3083`.

File duy nhất cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03083_plus_note_v3_yearrank_v2_exact13_w010/submission.zip
```

SHA-256:

```text
a27e20365e3f4fa3a081014fe69e64eeb6b78dd4047863f119da64d3596f4a05
```

Allowlist gồm 13 ID:

```text
813,829,852,890,897,921,936,946,959,971,978,981,1008
```

Ba ID `907,910,986` vẫn fail closed vì thiếu ô dữ liệu năm hiện tại hoặc ô là
dấu `-`; không được thêm thủ công giá trị của năm trước. Submission mới chỉ
đổi đúng 13 ID trên, không ghi đè câu `ok` nào của checkpoint `0.3083`. Tất cả
1.012 query đã replay đúng answer, 229 test pass, line number bảng dùng chuẩn
1-based và ZIP pass `unzip -t`.

Leaderboard xác nhận:

```text
TABLES_F2MACRO      0.4913
DOCS_F2MACRO        0.9103
ANSWER_ACCURACY     0.3202
EXECUTION_ACCURACY  0.3202
```

So với `0.3083`, batch tăng sáu câu đúng trên tập chấm 506 câu, đồng thời tăng
table F2 `+0.0011` và docs F2 `+0.0024`. Đây là checkpoint chính thức mới và
là base bắt buộc cho mọi merge fill-only tiếp theo.

## 14. Checkpoint typed compositional ranking v3 exact12: 0.3300

Batch này không cần rerun Qwen. Typed planner tách rõ phép chọn theo công
thức/chỉ tiêu A và phép trả về công thức/chỉ tiêu B, sau đó solver deterministic
thực hiện `select-then-project` với exact metric, exact period và tie guard.

File cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03202_plus_compositional_ranking_v3_exact12_w010/submission.zip
```

SHA-256:

```text
4c674ca7e75db0e1b279ff2ea997799b7e7c1cdd96a0945fdcca6f308b92406d
```

Allowlist gồm 12 ID:

```text
415,440,459,475,479,491,505,507,543,556,558,575
```

Bốn ID `495,497,511,524` vẫn fail closed do thiếu exact row cho metric note,
EPS hoặc thuế; không nới fuzzy matching. Nhóm inventory-days dùng operand có
kiểu kỳ `inventory(t-1)`, `inventory(t)`, `COGS(t)` và đọc đầy đủ evidence của
mọi doanh nghiệp trước khi chọn công ty thắng. Column resolver cũng nhận năm bị
OCR tách thành hàng header riêng hoặc ngày bị ép thành số như `3092024`.

Submission chỉ đổi đúng 12 entry so với checkpoint chính thức `0.3202`, không
ghi đè câu `ok`. Tất cả 1.012 query compile và replay đúng answer, 240 test
pass, `relevant_tables` dùng line number 1-based và ZIP pass `unzip -t`.
Leaderboard xác nhận:

```text
TABLES_F2MACRO      0.4958
DOCS_F2MACRO        0.9139
ANSWER_ACCURACY     0.3300
EXECUTION_ACCURACY  0.3300
```

So với `0.3202`, batch tăng năm câu đúng ròng trên tập chấm 506 câu, table F2
tăng `0.0045` và docs F2 tăng `0.0036`. Đây là checkpoint chính thức mới và
là base bắt buộc cho median-filter planner v4.

## 15. Checkpoint typed median-filter planner v4 audited9

Batch này không cần rerun Qwen. File duy nhất cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03300_plus_median_filter_v4_audited9_w010/submission.zip
```

SHA-256:

```text
a9be6ad8d2188e1a4f9a6db4f939031a7aab83eea00091a1a47744e85b4d0873
```

Allowlist gồm 9 ID:

```text
369,378,379,439,441,447,448,454,455
```

Planner v4 biểu diễn bộ lọc trung vị bằng node có kiểu gồm công thức lọc,
toán tử so sánh, chế độ thời gian và kỳ áp dụng. Executor tính filter trên toàn
bộ quần thể, tính median đúng cho số phần tử chẵn/lẻ, lọc rồi mới chọn và
project. Query nộp bài tự tính lại median và kiểm tra động cả những ứng viên
ban đầu nằm ngoài tập lọc.

Candidate chỉ đổi đúng 9 entry so với checkpoint `0.3300` ở answer, query,
evidence, relevant tables và relevant docs; 1.003 entry còn lại giữ nguyên.
Tất cả 1.012 query compile và replay đúng answer, 247 test pass, ZIP chứa 1.690
CSV dùng line number 1-based chính thức và `unzip -t` không lỗi.

Leaderboard xác nhận:

```text
TABLES_F2MACRO      0.4996
DOCS_F2MACRO        0.9178
TABLES_PRECISION    0.3440
TABLES_RECALL       0.6872
TABLES_MRR5         0.6101
DOCS_PRECISION      0.9509
DOCS_RECALL         0.9169
DOCS_MRR5           0.9753
ANSWER_ACCURACY     0.3300
EXECUTION_ACCURACY  0.3300
```

Median solver không tạo gain execution ròng, nhưng package tăng table F2
`0.0038` và docs F2 `0.0039` nhờ recall. Vì vậy đây là retrieval-best base ở
mốc execution `0.3300`; không tiếp tục mở rộng riêng nhóm median.

## 16. Checkpoint typed filter-then-aggregate planner v5 audited6

Batch này không cần rerun Qwen. File duy nhất cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03300_plus_filter_aggregate_v5_audited6_w010/submission.zip
```

SHA-256:

```text
1c6f62ebc24aff0177d874fe2f1593f91547e866c1c23087b167b1fdfc0a4a43
```

Allowlist gồm 6 ID:

```text
367,375,385,408,451,484
```

Planner v5 tách `PopulationNode`, `PredicateNode`, `ValueNode` và
`AggregateNode`. Nó xử lý filter hằng số hoặc trung vị, điều kiện nhiều năm,
giá trị level/growth/delta, chênh lệch hai công thức và bốn phép tổng hợp
`mean`, `sum`, `share`, `difference_of_means`. Executor bắt buộc resolve exact
predicate lẫn value cho mọi thành viên trước khi lọc. Query nộp bài tự tính lại
membership và aggregate, đồng thời đọc toàn bộ evidence đã audit.

Targeted retrieval depth 72 đạt coverage đầy đủ cho 17/19 route, nhưng chỉ sáu
câu vượt qua toàn bộ exact row, period, unit và completeness guard. Mười ba câu
còn lại giữ `failed`; không hạ ngưỡng fuzzy. Merge chỉ đổi đúng sáu ID trên so
với package `0.3300`, còn 1.006 entry giữ nguyên.

Toàn bộ 256 test pass, 1.012 query compile và replay đúng answer, ZIP chứa
1.718 CSV với line number `<table>` 1-based và `unzip -t` pass. Leaderboard xác
nhận:

```text
TABLES_F2MACRO      0.5031
DOCS_F2MACRO        0.9200
TABLES_PRECISION    0.3452
TABLES_RECALL       0.6910
TABLES_MRR5         0.6078
DOCS_PRECISION      0.9508
DOCS_RECALL         0.9196
DOCS_MRR5           0.9733
ANSWER_ACCURACY     0.3360
EXECUTION_ACCURACY  0.3360
```

Batch tăng khoảng ba câu đúng ròng và trở thành base chính thức cho v6.

## 17. Checkpoint typed period-aware average-balance planner v6 audited7

Batch này không cần rerun Qwen. File duy nhất cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03360_plus_average_balance_v6_audited7_w010/submission.zip
```

SHA-256:

```text
09ffd0a7b240478ebf1fe51d1fc72ec45021266ba597b64f1d8e8a54dafe7ba3
```

Allowlist gồm 7 ID:

```text
405,410,429,449,450,462,468
```

Planner v6 thêm `PeriodRef` và `AverageBalanceNode` cho số dư đầu/cuối kỳ,
sau đó ghép ROA, ROE, vòng quay tổng/TSCĐ và tỷ lệ `(LNST-CFO)/tài sản bình
quân` với filter, median, ranking, projection và aggregate hiện có. Ranking
theo năm hỗ trợ selector/predicate tăng trưởng so với chính năm liền trước.
Query nộp bài tự kiểm tra lại filter, median và thứ tự lựa chọn bằng toàn bộ
evidence, không dùng membership tính sẵn.

Targeted retrieval depth 72 đạt coverage đầy đủ 9/9. Chỉ 7 câu vượt exact row,
period, distinct-cell và unit guards. ID `398` fail do header HHV mất năm;
`572` fail do thiếu exact rows của GEX. Final retrieval và codegen chỉ đổi đúng
7 ID so với checkpoint `0.3360`; 1.005 entry còn lại giữ nguyên.

Toàn bộ 265 test pass, 1.012 query compile và replay đúng answer, ZIP chứa
1.748 CSV với line number `<table>` 1-based chính thức và `unzip -t` pass.

Leaderboard xác nhận:

```text
TABLES_F2MACRO      0.5059
DOCS_F2MACRO        0.9212
TABLES_PRECISION    0.3443
TABLES_RECALL       0.6943
TABLES_MRR5         0.6069
DOCS_PRECISION      0.9494
DOCS_RECALL         0.9215
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3399
EXECUTION_ACCURACY  0.3399
```

Batch tăng hai câu đúng ròng trên tập chấm 506 câu. Đây là checkpoint chính
thức và là base cho quantified-cohort planner v7.

## 18. Candidate typed quantified-cohort planner v7 audited11

Batch này không cần rerun Qwen. File duy nhất cần nộp leaderboard:

```text
artifacts/submission_tranhuy_03399_plus_quantified_cohort_v7_audited11_w010/submission.zip
```

SHA-256:

```text
a6fd879fd0b9bffdda8dde1be9a415290a036700c1514f048e152d4a1369e6bd
```

Allowlist gồm 11 ID:

```text
364,404,414,437,460,467,489,540,542,569,574
```

Planner v7 thêm `PeriodQuantifierNode` cho điều kiện đúng ở tất cả hoặc ít
nhất một kỳ, `RankSliceNode` cho top-k, denominator predicate có scope riêng và
aggregate `partition_ratio` cho nhóm trên trung vị so với nhóm bù. Executor
resolve toàn bộ thành viên và năm, fail khi top-k hòa ở biên, rồi tự tính lại
membership, rank, partition và denominator trong pandas query nộp bài.

Parser giữ `CFO/LNST` và `hàng tồn kho/nợ ngắn hạn` thành một công thức thay vì
tách thành operand rời. Entity resolver bổ sung `Nam Kim -> NKG`; chi phí lãi
vay được chuẩn hóa trị tuyệt đối khi tổng hợp các báo cáo dùng quy ước dấu khác
nhau.

Targeted retrieval depth 72 đạt coverage đầy đủ 18/20 route. Chín ID
`401,438,446,458,466,469,470,552,566` vẫn fail closed do thiếu hoặc mơ hồ ít
nhất một exact operand. Final retrieval và codegen chỉ đổi đúng 11 ID allowlist
so với checkpoint `0.3399`; 1.001 entry còn lại giữ nguyên.

## V11 note-axis checkpoint 0.3696

File nộp trực tiếp, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_03597_plus_note_axis_v11_audited11_w010/submission.zip
```

Payload dự phòng cho Kaggle:

```text
artifacts/kaggle-payload-note-axis-v11-w010.zip
```

V11 chạy deterministic ở `k=72` để lấy đủ bốn bảng vốn chủ sở hữu ACB đang ở
rank 31-43, rồi merge fill-only 11 ID lên codegen v10. Khi thử nghiệm tiếp,
không dùng output Qwen mới để thay toàn bộ checkpoint `0.3696`; luôn merge theo
allowlist để giữ nguyên các câu đã được leaderboard xác nhận.

Toàn bộ 311 test pass, 1.012 query compile và replay, ZIP chứa 1.872 CSV với
line number `<table>` 1-based chính thức và `unzip -t` pass. Leaderboard xác
nhận Execution Accuracy `0.3696`, tăng `0.0099` so với v10; dùng v11 làm base
cho batch fill-only tiếp theo.

## V12 lease-schedule checkpoint 0.3755

File nộp trực tiếp, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_03696_plus_lease_schedule_v12_audited7_w010/submission.zip
```

SHA-256:

```text
ff5157213d24df894674bd149cbdc67d40cd530f45cbc0e2233114609e5548e8
```

Allowlist gồm 7 ID:

```text
37,125,128,233,638,882,895
```

Planner phân biệt lịch bên cho thuê và bên đi thuê bằng context, resolve cột
cuối năm cùng các bucket dưới một năm, một đến năm năm và trên năm năm. Dòng
tổng chỉ được chấp nhận khi bằng tổng các bucket; pandas query nộp bài đọc và
kiểm tra lại toàn bộ identity này. Retrieval chỉ refresh ID 882 để đưa đủ bảng
cho thuê của VIC vào evidence quota; 1.011 route còn lại được giữ nguyên.

Payload Kaggle dự phòng:

```text
artifacts/kaggle-payload-lease-schedule-v12-w010.zip
```

Toàn bộ 318 test pass, 1.012 expression compile và replay. Submission chỉ đổi
đúng 7 entry so với v11, chứa 1.879 CSV dùng line number `<table>` 1-based và
`unzip -t` pass. Leaderboard xác nhận Execution Accuracy `0.3755`, tăng
`0.0059` tương đương ba câu đúng ròng; dùng v12 làm base cho batch fill-only
tiếp theo.

## V13 select-then-project checkpoint 0.3814

File leaderboard đã xác nhận:

```text
artifacts/submission_tranhuy_03755_plus_select_project_v13_audited6_w010/submission.zip
```

SHA-256:

```text
5a7eda65bb1217c347c490d53ac2bafcb585965030441193cad826b29178e531
```

Allowlist gồm 6 ID:

```text
495,501,503,522,524,533
```

Planner xếp hạng theo chỉ tiêu A rồi chiếu chỉ tiêu B của kỳ thắng; ID 501 có
thêm tie-breaker thứ hai. Resolver hỗ trợ company block, header nhiều tầng và
report ID cũ. Toàn bộ 322 test pass, 1.012 expression compile và replay, ZIP
chứa 1.907 CSV dùng line number `<table>` 1-based. Leaderboard xác nhận
Execution Accuracy `0.3814`, tăng `0.0059`; table F2 tăng `0.0022` và docs F2
tăng `0.0011`. Dùng V13 làm base fill-only cho V14.

## V14 financial-scenario checkpoint 0.3913

File cần nộp trực tiếp, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_03814_plus_scenario_v14_audited17_w010/submission.zip
```

SHA-256:

```text
9043411f3f28f7b3e14f5ea2f709e546ec4d8215169ac07c72c7ee1001f68f3e
```

Allowlist gồm 17 ID:

```text
368,387,394,409,416,419,423,432,433,434,436,453,458,469,470,545,566
```

Planner V14 biểu diễn các kịch bản tăng chi phí lãi vay, giảm EBIT, tăng giá
vốn, tăng doanh thu đến trung vị, stress thanh lý và headroom lãi vay. Canonical
inventory tách dòng thuần mã `140` khỏi giá gốc mã `141`; `chi phí đi vay` được
chuẩn hóa về interest expense và route nhận đủ BSR/PVT/GEE/GEX. Thay đổi này
cũng mở khóa 11 câu cohort/temporal cũ có cùng exact balance-sheet operands.

Hai family inventory-days và EPS dilution vẫn fail closed do thiếu kỳ đầu exact;
không đưa `424,425` vào allowlist. Các output planner sai semantics hoặc generic
composite cũng bị loại. Merge chỉ đổi đúng 17 entry đang failed, còn 995 entry
của checkpoint `0.3814` giữ nguyên.

Toàn bộ 325 test pass; 1.012 expression compile và replay đúng answer. ZIP chứa
1.993 CSV với line number `<table>` 1-based và `unzip -t` pass. Payload Kaggle
dự phòng:

```text
artifacts/kaggle-payload-scenario-v14-w010.zip
```

Leaderboard xác nhận V14 đạt execution `0.3913`, tăng `0.0099` tương đương năm
câu đúng ròng. Table F2 tăng `0.0077`, docs F2 tăng `0.0023`; dùng V14 làm base
chính thức cho V15.

## V15 matrix sensitivity/FX checkpoint 0.3953

File nộp trực tiếp, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_03913_plus_matrix_risk_v15_audited6_w010/submission.zip
```

SHA-256:

```text
f75f0d5d048360e50e29692aa3e6d71cb59181cfcb425d936c2f202ff3068d0f
```

Allowlist gồm 6 ID:

```text
156,213,275,427,428,727
```

Planner V15 đọc trực tiếp các giao điểm hàng/cột/block cho tổng tài sản tài
chính chịu rủi ro tín dụng, số dư USD, tổng giá trị hợp đồng phái sinh, độ nhạy
ngoại tệ, trạng thái tiền tệ nội/ngoại bảng và VaR một ngày. Câu FPT kiểm tra
đủ bốn đồng tiền trước khi cộng phần lỗ của các đồng có công nợ lớn hơn tài
sản. Câu ACB đọc đủ sáu cột ngoại tệ, chọn trạng thái âm lớn nhất, áp shock 5%
và chia cho LNTT exact năm 2024.

Đáp án đã audit:

```text
156       2991.04
213          6.16
275    3080776.00
427         55.55
428          0.29
727          8.56
```

Merge chỉ đổi 6 entry đang failed và giữ nguyên 1.006 entry của V14. Toàn bộ
332 test pass; 1.012 expression compile và replay đúng answer từ CSV trong ZIP.
Submission chứa 1.996 CSV, dùng line number `<table>` 1-based; submission và
payload ZIP đều qua kiểm tra integrity. Payload dự phòng:

```text
artifacts/kaggle-payload-matrix-risk-v15-w010.zip
```

Leaderboard xác nhận V15 đạt execution `0.3953`, tăng `0.0040`, tương đương hai
câu đúng ròng. Table F2 là `0.5357`, docs F2 là `0.9316`; dùng V15 làm base
fill-only chính thức cho V16.

## V16 note-temporal aggregate checkpoint 0.4012

File nộp trực tiếp, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_03953_plus_note_temporal_v16_audited12_w010/submission.zip
```

SHA-256:

```text
22cea3bde5839b43952ae17dc03ef06e6f4591a3c3b243db995881fc11e4d2b5
```

Allowlist gồm 12 ID:

```text
102,260,617,652,663,685,836,854,887,912,939,941
```

Planner V16 xử lý direct matrix lookup, chênh lệch cuối kỳ, tỷ lệ từ mã BCTC,
bao phủ nợ xấu và các phép max/sum/mean qua nhiều năm. Serializer chỉ giữ thêm
ô số cho ba dạng ma trận note đã xác định, tránh làm thay đổi query cũ.

Merge chỉ đổi 12 entry đang failed và giữ nguyên 1.000 entry của checkpoint
V15. Toàn bộ 341 test pass; 1.012 expression compile và replay đúng answer.
Submission chứa 2.013 CSV với line number `<table>` 1-based; submission và
payload ZIP đều qua kiểm tra integrity. Payload dự phòng:

```text
artifacts/kaggle-payload-note-temporal-v16-w010.zip
```

Payload SHA-256:

```text
a986faf26d78de466fa05c162d23b93246561e36fa0dedf939a348413320595b
```

Leaderboard xác nhận V16 đạt execution `0.4012`, tăng `0.0059`, tương đương ba
câu đúng ròng. Table F2 là `0.5377`, docs F2 là `0.9320`; dùng V16 làm base
fill-only chính thức cho V17.

## V17 note-ratio/maturity checkpoint 0.4032

File nộp trực tiếp, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_04012_plus_note_ratio_v17_audited6_w010/submission.zip
```

SHA-256:

```text
46e6cf543c237e9d3fed6d71af8d2a3f2c364dfa9eecebd9f388aa4c066a0beb
```

Allowlist gồm 6 ID:

```text
826,863,885,892,945,955
```

Planner V17 xử lý tỷ trọng giá vốn cho thuê đất/hạ tầng, đếm khoản phải trả
bên liên quan dương, tỷ trọng chứng chỉ tiền gửi dưới 12 tháng, tỷ trọng doanh
thu tại Lào, tỷ trọng USD ngoại bảng và tỷ trọng cho vay khách hàng ngắn hạn.
Mỗi công thức khóa metric, note block, kỳ và mẫu số trong cùng bảng; resolver
fail closed nếu thiếu operand hoặc gặp hàng mơ hồ.

Audited answers:

```text
826    2016.00
863       1.00
885       8.63
892      29.17
945       9.50
955      54.66
```

Merge chỉ đổi đúng 6 entry đang failed và giữ nguyên 1.006 entry của checkpoint
V16. Toàn bộ 347 test pass; 1.012 expression compile và replay đúng answer.
Submission chứa 2.028 CSV với line number `<table>` 1-based; submission và
payload ZIP đều qua kiểm tra integrity. Payload dự phòng:

```text
artifacts/kaggle-payload-note-ratio-v17-w010.zip
```

Payload SHA-256:

```text
f1abf8f40a5e79e64b8c822b17e201cf4a7e795d42bf7878e4e17cb7f61ac38c
```

Leaderboard xác nhận V17 đạt execution `0.4032`, tăng `0.0020`, tương đương một
câu đúng ròng trên tập 506 câu. Table F2 tăng từ `0.5377` lên `0.5395`; docs F2
giữ nguyên `0.9320`. Dùng V17 làm base fill-only chính thức cho V18.

## V18 exact VAS-code cohort checkpoint 0.4071

File nộp trực tiếp, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_04032_plus_vas_cohort_v18_audited12_w010/submission.zip
```

SHA-256:

```text
1312ee61da8ad3ed2d373ff13e1450497832b1a5a37c249a0e63ece4e80a30d6
```

Allowlist gồm 12 ID:

```text
389,390,397,401,403,406,420,438,444,445,552,572
```

Resolver V18 đọc exact VAS code trước fuzzy matching cho ba loại báo cáo chính,
chọn kỳ cuối/current, loại cột đầu kỳ và fail closed nếu cùng mã có giá trị
xung đột. Guard canonical-label ngăn các số thứ tự ở bảng thuyết minh bị hiểu
nhầm là mã chỉ tiêu. Nhờ đó các cohort formula có nhãn OCR hỏng, đặc biệt tài
sản ngắn hạn mã `100`, vẫn được chứng minh đúng metric.

Audited answers:

```text
389      1.10
390      6.69
397    146.61
401     47.45
403     -0.06
406     -3.56
420      4.77
438    -27.22
444      0.91
445      2.01
552    150.60
572     12.88
```

Retrieval chỉ đổi đúng 12 ID trên và tăng depth riêng lên 96. Merge fill-only
nhận đủ 12 câu ở confidence `93-94`, giữ nguyên 1.000 record của V17 và giảm
failed từ 45 xuống 33. Toàn bộ 352 test pass; 1.012 expression compile và replay
đúng answer. Submission chứa 2.043 CSV dùng line number `<table>` 1-based; cả
submission và payload ZIP đều qua integrity check.

Payload Kaggle dự phòng:

```text
artifacts/kaggle-payload-vas-cohort-v18-w010.zip
```

Payload SHA-256:

```text
cd5a361963861b6359c67a901078d8cd8529ec78def8bea4df65f1bab293d78d
```

Leaderboard xác nhận V18 đạt execution `0.4071`, tăng `0.0039`, tương đương hai
câu đúng ròng trên tập 506 câu. Table F2 tăng từ `0.5395` lên `0.5446`; docs F2
tăng từ `0.9320` lên `0.9354`. Dùng V18 làm base fill-only chính thức cho V19.

## V19 typed derived-selector candidate audited9

File nộp trực tiếp, không cần rerun Qwen:

```text
artifacts/submission_tranhuy_04071_plus_derived_selector_v19_audited9_w010/submission.zip
```

SHA-256:

```text
dee6de45c5ef514bf0584fbd83b9453b0bf4bd5ce5f67e737106de8088e4ea14
```

Allowlist gồm 9 ID:

```text
376,377,381,382,391,417,418,422,461
```

Planner V19 bổ sung selector chênh lệch cùng kỳ, selector biến động theo thời
gian và phép lấy projection tại hai cực trị rồi trừ nhau. Ba derived formula
mới là `CFO margin - net margin`, `LNST - CFO` và
`(CFO - LNST) / doanh thu thuần`. SG&A intensity dùng trị tuyệt đối của chi
phí để thống nhất báo cáo trình bày chi phí âm/dương. Router nhận đúng tên giao
dịch DPM/DCM/HT1 và phân biệt cách viết `tỉ số` với `tỷ số`.

Audited answers:

```text
376    61.66
377     6.86
381    22.16
382     0.64
391    21.37
417     0.94
418    -4.58
422     0.55
461     0.97
```

Merge chỉ đổi đúng 9 entry failed, giữ nguyên 1.003 record của V18 và giảm
failed từ 33 xuống 24. ID 426 được refresh retrieval nhưng vẫn fail closed và
không đi vào allowlist. Toàn bộ 359 test pass; 1.012 expression compile và
replay đúng answer. Submission chứa 2.054 CSV dùng line number `<table>`
1-based; cả submission và payload ZIP đều qua integrity check.

Payload Kaggle dự phòng:

```text
artifacts/kaggle-payload-derived-selector-v19-w010.zip
```

Payload SHA-256:

```text
f4f28846f45eeacd7957f1a1c1e6ae0203d5935e5eba316078428b5f0edf637d
```

Trần lý thuyết nếu cả 9 câu đúng trên tập leaderboard 506 câu là execution
khoảng `0.4249`. Chỉ nâng V19 thành checkpoint sau khi leaderboard xác nhận.

## V20 implicit-period select/project candidate audited3

V19 đã được leaderboard xác nhận ở Execution Accuracy `0.4111`. V20 là batch
deterministic kế tiếp, không cần rerun Qwen để nộp:

```text
artifacts/submission_tranhuy_04111_plus_implicit_period_v20_audited3_w010/submission.zip
```

SHA-256:

```text
37c68ba6111d1ad21fb6b863de2e54e5dc58fb1729e26c0fb2ce96282576f82a
```

Allowlist fill-only:

```text
384,430,431
```

V20 interprets an implicit opening-to-closing change when a question provides
only the closing year. It evaluates every entity over the prior and closing
periods, applies exact filters before ranking, and projects the requested
metric from the unique winner. The audited results are `384=10.86`,
`430=17.51`, and `431=5.99`.

ID 511 was audited but intentionally not merged. Its provisional EPS fix
changed global CSV serialization and caused replay mismatches for three
existing checkpoint rows; the replay gate therefore rejected it and the
serializer change was reverted.

Verification: 362 tests pass; all 1,012 submitted expressions compile and
replay, with 2,054 CSV files and official 1-based `<table>` line positions.

Leaderboard V20: `TABLES_F2MACRO=0.5498`, `DOCS_F2MACRO=0.9400`,
`EXECUTION_ACCURACY=0.4111`. The retrieval gain is real, but the three new
implicit-period answers did not increase execution. Do not expand this branch
until new labeled evidence justifies it.

Payload Kaggle dự phòng, chỉ dùng nếu muốn chạy Qwen lại trên retrieval V20:

```text
artifacts/kaggle-payload-implicit-period-v20-w010.zip
```

SHA-256:

```text
28e1d192ccd4838b68379987f7fb24ee7dee7910715a046ada5797055ffd7bf0
```

## V21 exact period, EPS and lease resolver candidate audited5

Submission to upload directly, without rerunning Qwen:

```text
artifacts/submission_tranhuy_04111_plus_exact_period_v21_audited5_w010/submission.zip
```

SHA-256:

```text
06c5f1ffdcf7bc40f826bcc5007b4f35d42e5156523dc483e8474439b36a2643
```

Allowlist fill-only:

```text
336,398,452,511,550
```

V21 recovers dotted opening/closing dates from split table headers, handles an
EPS value swallowed into the row-code field through an EPS-only raw-grid
cross-check, and resolves HND's full future minimum lease schedule. It changes
only the five listed failed rows; their answers are respectively `387.66`,
`0.09`, `35.22`, `21.17`, and `0.20`.

Verification: 365 tests pass. All 1,012 queries compile and replay from 2,071
packaged CSV files. Submission and payload ZIPs both pass integrity checks.

Payload Kaggle dự phòng:

```text
artifacts/kaggle-payload-exact-period-v21-w010.zip
```

Payload SHA-256:

```text
e36d2f776dc2e2be02f45493c0bc1a5fddb7db7ffcbd076b562a6a0153f33ef6
```

Leaderboard V21: `TABLES_F2MACRO=0.5510`, `DOCS_F2MACRO=0.9392`,
`EXECUTION_ACCURACY=0.4150`. This is an execution gain of `0.0039` over V20,
so V21 becomes the new base checkpoint.

## V22 grid-backed exact cohort resolver candidate audited5

Submission to upload directly, without rerunning Qwen:

```text
artifacts/submission_tranhuy_04150_plus_grid_exact_v22_audited5_w010/submission.zip
```

SHA-256:

```text
d47fb250f3452a2d90e8eb95ba23d86e62ec734c65f5ae34facbc1b059761129
```

Allowlist fill-only:

```text
388,442,443,446,466
```

V22 reads canonical VAS rows from `grid_json` only when the expected code,
exact row label, requested period and existing tidy CSV cell all agree. It also
rejects a comparative column that explicitly names another year and accepts a
generic prior-period column from the following filing. These changes recover
CEO, VNM and VIC evidence needed by five cohort questions without modifying
global CSV serialization.

Audited answers are `388=22.88`, `442=47.70`, `443=18.44`, `446=50.38`, and
`466=73.66`. Merge changes exactly five rows and leaves 11 failed questions.
Verification: 368 tests pass; all 1,012 queries compile and replay from 2,102
packaged CSV files. Submission and payload ZIPs pass integrity checks.

Payload Kaggle dự phòng:

```text
artifacts/kaggle-payload-grid-exact-v22-w010.zip
```

Payload SHA-256:

```text
e512a6ddd08f68384c282d9ad4b00f85d9433a7c32a253e4a614ee199b36f2bf
```

## V23 full-checkpoint exact lookup challenger audited15

Submission to upload directly, without rerunning Qwen:

```text
artifacts/submission_tranhuy_04150_plus_exact_lookup_v23_audited15_w010/submission.zip
```

SHA-256:

```text
1af07fb2bbfd7c092af7a42d5885ea204bb4ed86b2332580da7ae68f405955d6
```

Allowlist replacing successful LLM single-vote rows:

```text
67,69,73,77,85,94,177,178,183,220,237,282,316,346,681
```

Reproduce the 383-row shadow audit:

```bash
python scripts/15_audit_checkpoint_challengers.py \
  --retrieval artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl \
  --checkpoint artifacts/codegen_tranhuy_04150_plus_grid_exact_v22_audited5_w010.jsonl \
  --out-dir artifacts/challenger-audit-v23-llm-single-vote \
  --k 112
```

The lookup slice contains 175 rows. The fail-closed challenger agrees on 12,
replaces 15 audited disagreements and refuses 148. The final merge changes only
the allowlist and preserves 997 rows byte-for-byte on answer/query. Verification:
374 tests pass, all 1,012 expressions compile and replay, and `unzip -t` passes.

Do not merge every exact disagreement automatically. In particular, retain the
guards for opening dates and qualifier words such as bank, related party,
depreciation, foreign currency, deposit interest and common shareholder. These
were observed false-exact cases where the router mapped a child question to a
canonical parent total.

Leaderboard V23: `TABLES_F2MACRO=0.5534`, `DOCS_F2MACRO=0.9416`, and
`EXECUTION_ACCURACY=0.4407`. This is a `+0.0257` execution gain over V21, so
V23 is the new checkpoint.

## V24 exact two-operand difference challenger audited7

Submission to upload directly, without rerunning Qwen:

```text
artifacts/submission_tranhuy_04407_plus_exact_difference_v24_audited7_w010/submission.zip
```

SHA-256:

```text
30ae5e0f5474dc8d45e7aed7a8b3a972976438b896d7e7ff9e689114011750a8
```

Allowlist replacing successful LLM single-vote rows:

```text
581,592,737,776,777,798,808
```

Reproduce the difference audit:

```bash
python scripts/15_audit_checkpoint_challengers.py \
  --retrieval artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl \
  --checkpoint artifacts/codegen_tranhuy_04150_plus_exact_lookup_v23_audited15_w010.jsonl \
  --out-dir artifacts/challenger-audit-v24-difference-single-vote \
  --operation difference --k 112
```

The 60-row difference slice yields one exact agreement, seven audited
replacements and 52 refusals. Generic difference questions return an absolute
gap; explicit temporal-change wording remains directional. The closing date
`31/12` is explicitly protected from the opening-date `01/01` guard.

Verification: exactly seven rows change and 1,005 remain unchanged; 381 tests
pass, all 1,012 expressions compile and replay, and `unzip -t` passes.

Leaderboard V24: `TABLES_F2MACRO=0.5534`, `DOCS_F2MACRO=0.9416`, and
`EXECUTION_ACCURACY=0.4427`. The seven replacements produced only one net
correct answer, so do not expand generic difference semantics further.

## V25 exact direct-ranking challenger audited5

Submission to upload directly, without rerunning Qwen:

```text
artifacts/submission_tranhuy_04427_plus_exact_ranking_v25_audited5_w010/submission.zip
```

SHA-256:

```text
9ad5900bbb0903806b1e03d307e9fa76508accd04bc2dc1bfef1129c7ec4abe4
```

Allowlist replacing successful LLM single-vote rows:

```text
859,886,902,911,967
```

Reproduce the ranking audit:

```bash
python scripts/15_audit_checkpoint_challengers.py \
  --retrieval artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl \
  --checkpoint artifacts/codegen_tranhuy_04407_plus_exact_difference_v24_audited7_w010.jsonl \
  --out-dir artifacts/challenger-audit-v25-ranking-single-vote \
  --operation ranking --k 112
```

The 67-row ranking slice yields eight exact agreements, five audited
replacements and 54 refusals. The challenger supports only a direct max/min of
one canonical metric along one entity/year axis. Select-then-project, derived
ratios, missing candidates and child qualifiers are refused.

Verification: exactly five rows change and 1,007 remain unchanged; 386 tests
pass, all 1,012 expressions compile and replay, and `unzip -t` passes.

Leaderboard V25: `TABLES_F2MACRO=0.5533`, `DOCS_F2MACRO=0.9420`, and
`EXECUTION_ACCURACY=0.4466`. This is a `+0.0039` execution gain over V24, so
V25 is the new checkpoint.

## V26 full-checkpoint exact lookup coverage audited15

Submission to upload directly, without rerunning Qwen:

```text
artifacts/submission_tranhuy_04466_plus_exact_lookup_v26_audited15_w010/submission.zip
```

SHA-256:

```text
21815fa72f847776e5810d543b5140e5b3a44c499173abd2f23213e1ac0f091b
```

Allowlist replacing successful LLM single-vote rows:

```text
29,30,80,100,138,140,144,163,188,217,229,244,262,263,360
```

V26 expands exact lookup coverage with standard VAS codes `132`, `136`,
`212`, `252`, `253`, and typed note-matrix row/column resolution. A candidate
must still have one entity, one year, one atomic metric, exact context, exact
period and one unambiguous value. Named-counterparty questions remain blocked.

The audit finds 14 exact agreements and 15 audited disagreements among 160
lookup rows in the 356-row LLM single-vote cohort. The merge changes exactly
the 15 allowlisted rows and preserves the other 997 rows. Verification: 394
tests pass; all 1,012 expressions compile and replay from 2,088 packaged CSV
files; `unzip -t` passes. No Qwen rerun is required.

Leaderboard V26: `TABLES_F2MACRO=0.5533`, `DOCS_F2MACRO=0.9420`, and
`EXECUTION_ACCURACY=0.4565`. This is a `+0.0099` execution gain over V25, so
V26 is the new checkpoint.

## V27 exact direct-average challenger audited8

Submission to upload directly, without rerunning Qwen:

```text
artifacts/submission_tranhuy_04565_plus_exact_average_v27_audited8_w010/submission.zip
```

SHA-256:

```text
ffbc712d2963ed1b3ba583766e98191e6400ae14890b63f680f3d3395e858498
```

Allowlist replacing successful LLM single-vote rows:

```text
816,856,867,919,940,943,947,954
```

V27 averages one atomic canonical metric over exactly one varying axis: one
company across years or multiple companies in one year. It rejects derived
ratios, filtered cohorts, mixed metrics, duplicate scopes, child qualifiers
and any unresolved operand. All eight replacements use current-filing VAS
rows; two additional candidates agree with Qwen and are left unchanged.

Verification: exactly eight rows change and 1,004 remain unchanged; 397 tests
pass; all 1,012 queries compile and replay from 2,092 packaged CSV files; ZIP
integrity passes. No Qwen rerun is required.

Leaderboard V27: `TABLES_F2MACRO=0.5530`, `DOCS_F2MACRO=0.9420`, and
`EXECUTION_ACCURACY=0.4625`. This is a `+0.0060` gain over V26, approximately
three additional correct answers, so V27 is the new checkpoint.

## V28 exact direct-growth challenger audited3

Submission to upload directly, without rerunning Qwen:

```text
artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip
```

SHA-256:

```text
ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e
```

Allowlist replacing successful LLM single-vote rows:

```text
586,631,647
```

V28 audits the 18 remaining `growth_pct` routes and accepts only one atomic
canonical metric for one company across at least two distinct fiscal years.
The earliest year is the base and the latest is the end; accounting expense
and provision signs are normalized before applying
`(end - base) / abs(base) * 100`. Zero bases, cohort questions, mixed metrics,
missing periods and child-detail qualifiers fail closed.

The exact resolver produced four disagreements. ID 633 was deliberately
excluded because the question asks for the VietsovPetro counterparty detail
while the resolved VAS cell is the aggregate supplier-prepayment line. The
three accepted answers are `586=335.82`, `631=31.31`, and `647=37.46`.

Verification: exactly three rows change and 1,009 remain unchanged; 401 tests
pass; all 1,012 queries compile and replay from 2,092 packaged CSV files; ZIP
integrity passes. No Qwen rerun is required.

Leaderboard V28: `TABLES_F2MACRO=0.5530`, `DOCS_F2MACRO=0.9420`, and
`EXECUTION_ACCURACY=0.4644`. This is a `+0.0019` gain over V27, approximately
one additional correct answer on the inferred 506-question scoring subset.
V28 is the new checkpoint, but the direct-growth cohort is closed because the
three replacements produced only one net correct answer. Continue with a
separate direct-ratio audit so its impact can be isolated.
