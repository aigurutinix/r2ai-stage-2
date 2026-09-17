# ViFinQA — bàn giao

Đọc hết mục **Bẫy** và **Thử nghiệm đã thất bại** trước khi sửa bất cứ thứ gì. Mỗi mục ở đó
từng làm mất điểm hoặc mất một lượt nộp, và không mục nào đọc được từ tài liệu của BTC.

`PHAN-TICH.md` là ghi chú chiến lược ngắn hơn. Khi hai file lệch nhau, tin file này.

## ⛔ 10/08 — MÃ NGUỒN TRÊN ĐĨA KHÔNG DỰNG LẠI ĐƯỢC BẢN TỐT NHẤT

Một lần đồng bộ lúc ~9:05 sáng 10/08 ghi đè `scripts/run_submit.py` (852 → **531 dòng**) và
`src/vifin/answering/lookup.py` bằng bản cũ hơn. `compose.py` thì vẫn mới.

Đã mất: `ADAPTIVE_DECLARE_K`, `DECLARE_K_SCALE/CAP`, `DECLARE_POOL_K`, `USE_SHAPE_LOCK`,
`USE_LABEL_RESCAN`, `USE_LABEL_DECLARE`, `USE_DOC_LABEL_DECLARE`, `USE_BM25_DECLARE_PAD`,
`USE_EVIDENCE_REFS`, `USE_CONSENSUS`, `USE_CREDIBLE_GATE`, `_flag`; và **toàn bộ đường mã số**
trong `lookup.py` (`match_row_by_code`, `MA_SO_BY_LABEL` — không còn chỗ nào đọc `ma_so.json`).

Đo được, không phải suy đoán — `python scripts/_diff_zip.py screen_ratio_gated.zip final.zip`:

| | `screen_ratio_gated.zip` (tốt nhất) | dựng lại bằng mã hiện tại |
|---|---|---|
| đáp án khác nhau | — | **90 / 1012** |
| histogram số ref | `{3:388, 4:100, 6:233, 8:115, 10:72, 11:104}` | `{4: 1012}` phẳng |
| `result = 0.0` | 17 | 0 (không còn shape lock) |

> **ĐỪNG dựng bản nộp bằng `run_submit.py` hiện tại — nó tụt điểm.**
> Nộp bản mới bằng cách **vá `screen_ratio_gated.zip`**: tính đáp án cho các id cơ chế mới chạm
> rồi ghép vào `submission.json` + `data/*.csv`. Đó cũng là cách `screen_ratio_gated` và
> `threshold_cohort` đã được tạo.

### ✅ 11/08 17:10 — ĐÃ KHÔI PHỤC tầng cờ và chính sách khai bảng từ `.pyc`

`scripts/_recover_flags.py` (mới) lấy được đối số của mọi hằng `<computed>` mà
`_recover_pyc.py` chỉ báo là `CALL 1`. Giá trị của bản 08/08:

| hằng | giá trị |
|---|---|
| `USE_RATIO_LOOKUP` `USE_COMPOSE` `USE_SCREEN` `USE_SCREEN_RATIO` `USE_RATIO_DIVIDE` | `True` |
| `USE_EVIDENCE_REFS` `USE_LABEL_DECLARE` `USE_SHAPE_LOCK` `USE_LOCATE` `USE_TIEBREAK` `USE_PLAN` | `True` |
| `ADAPTIVE_DECLARE_K` | `True` |
| `USE_PANEL` `USE_RATIO` `USE_CREDIBLE_GATE` `LOCATE_WINS` | `False` |
| `USE_CONSENSUS` | `_flag("USE_CONSENSUS", default=False)` |
| `USE_LABEL_RESCAN` | `_flag("USE_LABEL_RESCAN", default=True)` |
| `USE_DOC_LABEL_DECLARE` `USE_BM25_DECLARE_PAD` | `_flag(...)` — mặc định **True** (các zip `nodoclabel`/`nobm25pad` là ablation tắt chúng) |
| `DECLARE_K_SCALE` | `float(os.environ.get("DECLARE_K_SCALE", "2.0"))` = **2,0** |
| `DECLARE_K_CAP` | `int(os.environ.get("DECLARE_K_CAP", "11"))` = **11** |
| `DECLARE_POOL_K` | `30` |
| `DECLARE_SOURCE` `SEARCH_SOURCE` | `'anchor'` |
| `LOCATE_SOURCE` | `'union'` |
| `RATIO_SANITY_LIMIT` | `1000.0` |
| `RATIO_UNITS` | `frozenset({'phan_tram', 'lan', 'vong'})` |

Và chính sách khai bảng, đọc trực tiếp từ disassembly `main`:

```python
if ADAPTIVE_DECLARE_K:
    span = max(1, len(question.tickers)) * max(1, len(question.years))
    predicted = max(len(set(evidence)), min(span, 8))
    target = min(max(round(DECLARE_K_SCALE * predicted), 3), DECLARE_K_CAP)
else:
    target = max(declare_k, len(evidence))
target = max(target, len(set(evidence)))
```

> **Kiểm chứng, không phải suy đoán.** Thay `predicted = 1..6` vào công thức cho
> `target = 3, 4, 6, 8, 10, 12→11`. Histogram số ref của bản nộp tốt nhất là
> `{3:388, 4:100, 6:233, 8:115, 10:72, 11:104}` — **đúng tập giá trị đó, không sót không thừa**.
> Công thức khôi phục tái tạo chính xác artefact mà nó được cho là đã tạo ra.

Một tiện lợi tình cờ: docstring `load_reranked` cũng nằm trong `.pyc`, nên số đo bẫy #9 còn
nguyên — `rerank(metric)` top-1 **41,4%**, đúng con số mà bộ chỉ số lexical mới đo lại được là
41,7% (mục 12).

Còn lại để port: nội dung hàm `shape_lock`, `label_rescan`, `label_declare`, `doc_label_declare`,
`bm25_declare_pad` và đường `ma_so` trong `lookup.py` — bytecode có đủ, cần đọc tiếp disassembly.

Khôi phục logic đã mất: `scripts/_recover_pyc.py` giải mã `scripts/__pycache__/run_submit.cpython-313.pyc`
(bản 08/08) — `marshal.loads(raw[16:])` + `dis` trả lại **chính xác** hằng số, tên toàn cục và số
dòng nguồn. Phần thêm của 09–10/08 (`SPAN_ONLY_K`, `USE_LAST_RESORT`) ghi trong `PHAN-TICH.md`.
`lookup.cpython-313.pyc` **không** cứu được — nó cùng ngày với bản .py đã hỏng.

**Chép `__pycache__/*.pyc` ra chỗ khác ngay sau mỗi bản build tốt.** Đó là bản sao lưu mã nguồn
duy nhất tồn tại trong repo này, và nó bị ghi đè ở lần chạy kế tiếp sau khi .py đổi.

## 11/08 — đường fine-tune: dựng xong, đang sinh dữ liệu

Vì sao chuyển hướng: Vương đi 0,3538 → 0,5119 → **0,6502** EXEC trong một ngày với **bốn chỉ số
TABLES giống hệt từng chữ số** ở cả ba lần nộp. Không đụng truy hồi. Họ dùng model 9B và đã hỏi BTC
về fine-tune. Ba giả thuyết rẻ hơn đều đã đo và đều âm: tự nhất quán đa số-của-3 **−7,1%**, đơn vị
tiền sạch, ngân sách bảng 8→20 **phẳng (+1,1%)**.

**Bằng chứng quyết định nằm trong `generation/schemas.py` của chính BTC:**

```python
class QARecord(BaseModel):
    id: int; question: str; answer: Answer
    relevant_docs: list[str]; relevant_tables: list[str]
    pandas_query: str; csv_path: str | list[str]; difficulty: Difficulty
```

Máy sinh của họ phát ra **giám sát gold đầy đủ, đúng định dạng bài nộp**, và được **xác thực bằng
thực thi**. Ta không cần tập gold riêng của họ — chạy máy của họ trên corpus công khai là ra dữ liệu
đúng phân bố đề thi theo cấu tạo.

### Bốn cửa phải mở, và cách mở

| cửa | cách qua |
|---|---|
| `catalog` báo `tables: 0` | bộ đọc của họ chờ neo `[table_N](...)` + thư mục `*_extracted_tables/`; ta có `<table>` HTML → `scripts/build_official_corpus.py`. **146.246 bảng, 0 lệch thứ tự** |
| `ModuleNotFoundError: fcntl` | dây chuyền chỉ chạy Linux → `vendor/win_shims/fcntl.py` (khoá tư vấn no-op; một tiến trình thì không tranh chấp) |
| `file_filter.csv` thiếu | `code_stock.csv` đã đúng hai cột đầu → sao chép |
| **thiếu cột ngành → chặn hai tầng dẫn xuất** | xem bên dưới |

### Thiếu ngành là cửa nguy hiểm nhất — nó im lặng

```python
medium.py:179        if seed_company is None or not seed_company.industry_l3: return []
intermediate.py:127  if seed_company is None or not seed_company.industry_l3: return []
```

Ngành rỗng thì **chỉ ra được tầng `easy`** — tra cứu một ô, đúng nửa bộ khớp nhãn đã giải ở 42,8%.
Train trên đó là học nửa đang thắng. `scripts/mine_industries.py` khôi phục phân hoạch từ **chính
bộ câu hỏi**: 19 câu gọi tên ngành, nhưng **288 câu liệt kê nhóm mã** → đồ thị đồng xuất hiện, ngưỡng
2 → **9 nhóm phủ 85/100 công ty**, nhóm đúng về tài chính.

> Nhóm 17 ngân hàng bị gán nhãn "công nghiệp chế biến" (trôi từ một câu về dư nợ ngành chế biến).
> `registry.py:136` chọn công thức tổ chức tín dụng bằng `industry_l2`, nên phải quyết bằng **thành
> viên chứ không bằng nhãn**: tên đăng ký có "Ngân hàng" → 21 mã gắn cờ đúng.

### Chọn model sinh — đo, không đoán

| model | kết quả |
|---|---|
| `qwen/qwen3-14b` (OpenRouter) | **5/5** ✅ |
| `Qwen3.5-9B` (local) | 0/20 — timeout, suy luận văn xuôi đốt hết token |
| `Qwen2.5-Coder-7B-AWQ` | **0/685** — trả lời đủ, validation loại sạch |
| **`Qwen3-14B-AWQ`** (local) | **dùng bản này** |

Kết quả 7B bác giả thuyết "coder thắng reasoner": phép đo cũ 61%-vs-9,4% đo **prompt của ta**, một
lần xuất chương trình. Dây chuyền BTC là phán đoán lồng nhau trên schema có kiểu. **Trục là năng lực,
không phải coder hay reasoner.**

### ⚠️ Bẫy #11 phải sửa: nó PHỤ THUỘC MODEL

- Qwen3.5-**9B** suy luận văn xuôi **không có thẻ `<think>`** → bật `--reasoning-parser` là `content`
  rỗng → **phải TẮT**
- Qwen3-**14B** phát đúng thẻ `<think>` → không bật thì `<think>` lẫn vào `content`, mọi phép đọc
  JSON hỏng → **phải BẬT**

Kiểm chứng: sau khi bật, `content = '{"ok": true}'` sạch, suy luận nằm ở trường riêng.

### Bẫy vận hành mới

- **`pkill -f <pattern>` tự giết phiên ssh** khi chuỗi lệnh từ xa chứa pattern đó — cắn 3 lần. Bracket
  trick (`Engine[C]ore`) không đủ nếu dòng lệnh còn chỗ khác chứa chuỗi. Dùng `ps -eo pid,cmd` rồi
  `kill` theo PID.
- **Hết 50 GB đĩa**: ba model cùng trong `/workspace/models`. Giữ đúng model đang phục vụ.
- **`sentence-transformers` phá cặp `transformers` của venv vLLM.** Tách `/workspace/genv` riêng, torch
  bản **CPU**, và `CUDA_VISIBLE_DEVICES=""` — vLLM giữ 22,9/24,5 GB, một embedder 2 GB rơi lên cùng
  card là sập server.
- **tar 148k file nhỏ cực chậm** (21 MB/10 phút). Gửi **nguồn** rồi dựng lại corpus trên máy: 105 MB/58s,
  dựng lại chỉ **14 giây**.

### Hai lỗi dữ liệu mà cổng thực thi KHÔNG bắt được

**1. Chương trình chứa chính đáp án của nó** — 6,6% số bản ghi:

```python
result = df.loc[df['Ngày 31/12/2018'] == '2.780.789', 'Ngày 31/12/2018'].values[0]
```

Chạy được, tái tạo đúng `answer` **theo cấu tạo**, qua sạch mọi validation. Lúc suy luận con số đó
chính là thứ chưa biết. Cùng lớp với `reads_no_frame`, khoác DataFrame nguỵ trang. Phải lọc theo
**hình dạng chương trình** — `_audit_generated.classify()`, dùng chung bởi cả audit lẫn `build_sft`.

**2. Seed cố định sinh lại y hệt.** `seed = round*7+13` → chain khởi động lại, round về 1, ra đúng
seed cũ, **7/9 câu là bản sao**. Sửa: seed từ đồng hồ + kích thước pool. Sau sửa: `deduped: kept 60`,
**0 trùng**, 91 bản ghi / 91 bảng khác nhau.

### Bộ dựng cặp SFT — hai quyết định bắt buộc

1. **Đổi quy ước đầu ra.** Gold kết bằng `df.iloc[r,c]` trả **chuỗi**; EXECUTION chấm bằng so số.
   Bằng chứng cái giá: dòng `BM25 Baseline` của BTC có ANSWER **1,0** / EXEC **0,3577**. Giữ nguyên ô,
   viết lại đuôi thành float (`num()` hoặc `as_float()` mới thêm vào `PRELUDE`).
2. **Prompt huấn luyện phải chứa nhiễu như lúc suy luận.** Gold trích **một** bảng; suy luận đưa
   **tám**. Train trên prompt chỉ có bảng đáp án dạy "đáp án nằm trong `df`". Nên mỗi cặp dựng lại qua
   **đúng bộ truy hồi của đường suy luận**, gold đặt vào trong shortlist, chương trình viết lại để gọi
   đúng tên frame.

> Bộ so số của tôi từng vứt nhầm 10/91 bản ghi tốt: `435.178` là 435178 (VN) hay 435,178 (Tây) —
> chuỗi không nói được. **Thứ dùng để lọc không được tự đoán.** Chấp nhận cả hai cách đọc: 68 → 78 cặp.

### Số đo hiện tại

| | |
|---|---|
| nhịp sinh (trên máy, 16 luồng) | **3,3 bản ghi/phút** |
| chất lượng: bộ khớp nhãn giải được | **25%** (đề thật 42,8% → sinh ra *khó hơn*) |
| chất lượng: câu chép nhãn dòng | **0%** |
| năng suất thành cặp huấn luyện | **85,7%** |

### Chạy lại toàn bộ

```bash
# local
python scripts/mine_industries.py --write
python scripts/build_official_corpus.py --out data/official_corpus
# máy thuê: bootstrap + vòng ba tầng
bash scripts/remote_bootstrap.sh
setsid nohup ./run_tiers.sh > gen_all.log 2>&1 < /dev/null &
# rồi
python scripts/build_sft.py runs/*_pool.jsonl --out artifacts/sft.jsonl
python scripts/train_qlora.py --data artifacts/sft.jsonl --out /workspace/lora
```

## 11/08 chiều — bốn phép đo đổi hướng ưu tiên

### 1. EXEC không phụ thuộc truy hồi. Bằng chứng nằm trên bảng xếp hạng

| đội | TABLES_F2 | EXEC |
|---|---|---|
| synera | **0,0** | 0,6601 |
| Trần Đình Minh Vương | 0,4118 | **0,6621** |
| ta (2795) | **0,5641** | 0,3696 |

synera khai **không bảng nào** mà vẫn 0,66. Bộ chấm chạy `pandas_query` trên `csv_path`
trong `evidence`, còn `relevant_tables` chỉ vào công thức F2 — hai trường tách rời, và
`package.py` đã tách sẵn (`tables` vs `ref_tables`). **Mọi đầu tư thêm vào TABLES không
trả một điểm EXEC nào.** Ta dẫn TABLES mà thua EXEC gần 300 câu.

### 2. Bản 2795 là nhánh 14B, và nó là bước nhảy lớn nhất kể từ `anchor_search`

`retry14b.zip` so với 2752: **282 đáp án đổi, 419 chương trình viết lại** → +0,0415 EXEC
≈ **+42 câu**. Kèm theo: ratio sai độ lớn 95 → 41, đáp án 0 là 28 → 10, chương trình hằng
số 17 → 4.

### 3. Cache 14B đã cạn — không vá thêm được nữa

Dựng bản đầy đủ nhất có thể từ `gen14b_merged.jsonl` (`--weak-branches`, báo vá 46 câu)
rồi so với `retry14b`: **0 đáp án đổi, 0 chương trình đổi**. Bản đang nộp đã tiêu thụ hết
cache đó. Muốn hơn thì phải **sinh mẻ mới**, không phải vá lại. `scripts/_zip_footprint.py`
và `_llm_headroom.py` dựng ra để đo đúng hai việc này.

| tình trạng của 1.012 câu | số câu |
|---|---|
| đang dùng chương trình 14B | 419 |
| có chương trình, đáp án trùng rule | 140 |
| **có chương trình, đáp án KHÁC rule** | **344** |
| không có chương trình chạy được | 109 |

**344 câu là toàn bộ ván bài còn lại**, và bản nộp đang chọn rule cho tất cả vì chưa có gì
để đo. Lỗi *không* tập trung ở lớp lạ: cohort/nhóm mã chỉ 116 câu, "chắc chắn sai theo quy
ước" chỉ 52 (`_class_sizes.py`, `_convention_audit.py`). ~630 câu sai nằm ở lớp thường,
nên phải nâng chất lượng đọc bảng đại trà chứ không vá từng lớp.

### 4. Lần đầu có thước đo offline: `scripts/eval_gold.py`

416 bản ghi trong `easy_pool.jsonl` là gold thật của generator BTC. Ánh xạ
`DOC|table_N` → `TableKey(DOC, N)` sang `tables.parquet` **không mất bản nào** (0/416 rơi),
nên chấm được ngay. Hai chế độ, và hiệu số giữa chúng mới là thứ cần: `--tables oracle`
đưa đúng bảng gold (đo *đọc*), `--tables retrieved` đưa shortlist tự truy hồi (đo *đầu-cuối*).
Chưa phép đo nào của dự án tách được hai nguồn lỗi đó, và chúng đòi hai cách sửa ngược nhau.

Số đầu tiên, nhánh không LLM: bộ khớp nhãn **kể cả khi được đưa đúng bảng gold** chỉ trả
lời 26,7% và đúng 10,0% (n=60). Đó là sàn — mới gọi `lookup.find`, chưa có tầng quy đổi đơn
vị và `compose`.

> **Bẫy: đừng chấm adapter trên chính 416 bản ghi đó.** `sft_easy.jsonl` được dựng từ chúng.
> Mẻ train đang chạy giữ lại 15% (`--eval-frac 0.15`, 323 học / 57 chấm); 57 mẫu đó là tập
> giữ lại duy nhất tồn tại. Phát hiện sau khi train xong thì không sửa được nữa.

### 5. `direct_answer` nhắm sai metric — và bỏ phiếu không cứu được 344 câu

Mã BTC có hai chiến lược, ta chỉ dùng `pandas_query`. Nhưng cơ chế "hỏi giá trị rồi định vị
ô" thì **đã tồn tại ở dạng mạnh hơn**: `plan_cells.py` + `scripts/run_plan.py` hỏi thẳng
**toạ độ ô + một phép toán từ tập đóng** trong một lượt, rồi ta tự biên dịch pandas — không
cần bước tìm ngược giá trị về ô. Phân bố phép toán trong 368 kế hoạch đã lưu là trần cứng của
hướng value-first:

| phép toán | số câu | |
|---|---|---|
| `value` — đáp án **là** một ô | 111 | 30% |
| `ratio_pct` / `diff` / `ratio` / `growth_pct` | 238 | 65% |

**70% câu không có ô nào chứa đáp án.** Direct-answer trả 51,29 cho một tỷ suất thì không ô
nào bằng 51,29 — bước định vị không có gì để tìm.

Dùng `planned.jsonl` làm **phiếu thứ ba** trên 344 câu bất định (`scripts/_plan_vote.py`), vì
nó sai khác kiểu với cả rule (khớp nhãn) và program (viết pandas):

| trên 344 câu | số câu |
|---|---|
| kế hoạch nói ra **số thứ ba** | **125** |
| không có kế hoạch để phá thế | 190 |
| ủng hộ rule / ủng hộ program | 17 / 12 |

Trong 154 câu có phiếu thứ ba, **81% cho ra số thứ ba**. Ba cơ chế độc lập rơi vào ba đáp án
khác nhau ⇒ **cả ba đều sai phần lớn**, không phải "một đúng một nhiễu". Không ngưỡng đồng
thuận nào cứu được pool này; thứ thiếu là **nhãn**, không phải thêm cơ chế.

> **Chỗ `direct_answer` thật sự đáng dùng là metric ANSWER, không phải EXEC.** Trường `answer`
> được chấm **riêng** với `pandas_query`, và ANSWER là một phần ba điểm macro. Ta đang để
> `answer` bám kết quả chương trình nên nó thừa hưởng mọi cú chết của chương trình miễn phí
> (ANSWER 0.3715 vs EXEC 0.3696, dính nhau). Một đáp án đọc thẳng đúng vẫn ăn ANSWER dù pandas
> crash. `report.py` của BTC: "Direct answering has no execution step" — đó là **giá trị**, không
> phải hạn chế.

### 6. 26 chương trình nộp kèm đáp án hằng số — đã sửa, `scripts/fix_constants.py`

`compile_plan(values=...)` phát đáp án thành literal rồi thêm `_ = df.iloc[0, 0]` để chương
trình "có chạm frame". Nó qua được kiểm tra tự động và **thủ tiêu đúng mục đích của kiểm tra
đó**. `retry14b.zip` có **29 câu** như vậy cộng **4 câu `result = 0.0` không tham chiếu frame
nào**. Vòng private có kiểm duyệt thủ công, và bản nộp của ta từng có 523/1012 hằng số.

Sửa không cần model: literal mang sẵn xuất xứ trong comment
(`v0 = 787973937.0 * 1.0  # df.iloc[8, 1]`), và `num()` đã có trong prelude. Mỗi lần viết lại
đều **thi hành trên đúng CSV trong zip** và chỉ giữ khi kết quả không lệch quá 0,01 →
`noconst.zip`: **26 câu đọc ô thật, 0 đáp án dịch chuyển**, cấu trúc y nguyên (1012 bản ghi,
4823 thành viên).

Ba câu bị từ chối là **lỗi thật mà literal đang che**, không phải rủi ro cần bỏ qua:

| id | `num()` đọc ô đó ra | nghĩa |
|---|---|---|
| 580 | `ValueError: '-'` | `num()` trong zip là bản **cũ, thiếu chốt nil**; bản hiện tại trả 0.0 |
| 896 | `ValueError: 'Vay ngắn hạn'` | toạ độ trỏ vào **ô nhãn** — đáp án đang nộp là bịa |
| 991 | 472154,05 ≠ 505171,0 đang nộp | `lookup._parse_cell` và `num()` **đọc lệch nhau** (bẫy #7 tái diễn) |

4 câu `result = 0.0` thì không nên trang trí cho hợp lệ: chúng là **đáp án sai**, và cách sửa
trung thực là trả lời được chúng — việc đó cần LLM, xếp vào hàng chờ GPU.

### 7. Chương trình của ta KHÔNG chết — sai số, không sai code (11/08 16:20)

`scripts/_exec_local.py` thi hành cả 1012 chương trình trên **chính CSV nằm trong zip**, tức
đầu vào y hệt grader:

| | |
|---|---|
| chạy không crash | **1012/1012 (100%)** |
| trả đúng con số ở trường `answer` | **1012/1012** |

Harness này **đã được kiểm định bằng một lượt nộp**: `noconst.zip` ghi điểm giống `retry14b`
tới bốn chữ số ở cả mười cột, đúng như nó dự đoán. Nên hai kết luận chốt lại:

- **Chất lượng sinh code không phải nút thắt của ta.** Toàn bộ khoảng cách 0,37 → 0,66 là
  *những con số sai*. Điều này phủ định cách khung hoá EXEC ở các mục trước, và làm yếu hẳn lý
  do fine-tune "để viết pandas giỏi hơn".
- **ANSWER không phải trục tự do.** Hai trường luôn nói cùng một số (0.3715 vs 0.3696 lệch đúng
  ~1 câu), nên ý "dùng `answer` như kênh riêng vì nó không phải chạy code" **không có đất** —
  trục đó đã bị dùng hết.

### 8. Ta đọc ô thật, chỉ là **sai ô** — và không phép thử tất định nào thấy được

`scripts/_answer_grounding.py`: với câu dạng tra cứu đơn (đáp án gold *là* một ô), con số ta
nộp quy về đồng có bằng ô nào trong bảng ta gửi kèm không?

| trên 411 câu tra cứu đơn | số câu |
|---|---|
| **đáp án là một ô có thật** | **352 (85,6%)** |
| không ở ô nào, nhưng câu hỏi hàm ý tổng | 31 |
| không ở ô nào, câu hỏi là một chỉ tiêu đơn | **28** |

> **Lần đầu phép đo này chạy nó báo 236 (57,4%) — sai gấp bốn lần.** Dung sai đặt theo *độ lớn*
> (`|target| × 1e-6`) trong khi đáp án được làm tròn 2 chữ số **trong đơn vị của câu hỏi**, nên
> ô đứng sau nó chỉ bị ghim trong `±0,005 × unit_scale` — với "tỷ đồng" là **5 triệu đồng**,
> rộng gấp mười lần cửa sổ tôi đặt. id=4 phơi ra lỗi: nó đọc đúng một ô qua `find_row` mà vẫn
> bị gắn cờ. Bài học lặp lại bẫy cũ: **thứ dùng để lọc không được tự đoán thang đo của nó.**

Kết quả sau khi sửa là một **kết quả âm có giá trị**: giả thuyết "ta dẫn xuất ở chỗ chỉ cần
đọc" **bị bác bỏ** — 85,6% câu tra cứu đã đọc một ô có thật. Vậy lỗi nằm ở **chọn sai ô**
(sai dòng / sai cột / sai bảng), và một ô sai vẫn là một ô, nên **không phép thử tất định nào
phát hiện được**. Đường duy nhất còn lại là **nhãn** (`eval_gold.py`) hoặc **đối chiếu chéo tài
liệu** (`corroborate.py`: số năm 2018 xuất hiện lại ở cột "năm trước" của báo cáo 2019).
Cũng khớp với ghi chú cũ trong `PHAN-TICH.md`: *"model kém ở việc quyết định con số nào"*.

### 9. Đường của top1 vẫn mở — nút thắt là một constructor, không phải công thức

Top1 (0.6621) dùng **9B fine-tune** trên dữ liệu tự sinh từ kho BTC, đáp án kiểm định tất định
bằng cách **thi hành truy vấn trên bảng nguồn**, model sinh ≤14B. Đó đúng là `run_tiers.sh`.
Ta dừng vì medium 3 giờ ra 2 bản ghi (≈12 ngày), nhưng nguyên nhân **không thuộc công thức**:

`generation/embedding_index.py::TableIndex.__init__` embed **toàn bộ** entries ngay trong
constructor, và `build_pool_index` được gọi **mỗi seed** với tới 300 bảng, trên CPU. Việc nó
dùng để làm chỉ là tìm `top_k` bảng liên quan — mà repo BTC **đã có** `BM25RetrievalIndex` với
đúng chữ ký `search(query, *, top_k)`, và đo lường của chính ta ghi *"Lexical thắng model, không
ngoại lệ"* trên kho này. Thay dense bằng lexical: nhanh hơn hàng nghìn lần, **đúng hơn** theo số
của ta, và không chạm phần luật quan tâm (đáp án vẫn do thi hành xác nhận, model sinh vẫn ≤14B).

Ràng buộc đã xác nhận trong `client.py`: open-weight **≤14B, trước 2026-06-01**. Top1 chịu
**cùng** ràng buộc, nên khoảng cách 0,37 → 0,66 là phương pháp, không phải phần cứng hay model đóng.

### 10. Thước đo offline đầu tiên chạy được — và bảng gold KÉM hơn shortlist của ta

`scripts/eval_gold.py`, 416 bản ghi gold, nhánh khớp nhãn, không LLM:

| | `--tables oracle` (đưa đúng bảng gold) | `--tables retrieved` (shortlist 6 bảng của ta) |
|---|---|---|
| bảng gold có trong tập được đưa | 100% | 69,2% |
| khớp nhãn **chịu trả lời** | 24,3% | **41,1%** |
| **đúng** (signed) | 8,7% | **10,8%** |
| đúng (magnitude) | 7,5% | 10,6% |
| đọc từ một bảng gold | 101/101 câu đã trả lời | **51/171** |
| bảng đúng nhưng **sai ô** | 64,4% | 60,8% |

> **Đưa đúng bảng gold lại cho điểm THẤP hơn.** Không phải nghịch lý: oracle thường chỉ có 1–2
> bảng nên bộ khớp nhãn không có gì để bám và **bỏ trả lời 75,7% câu**; shortlist 6 bảng cho nó
> nhiều chỗ bám hơn nên nó bắn nhiều hơn và đúng nhiều hơn.
>
> Quan trọng hơn: ở chế độ retrieved có **45 câu đúng** nhưng chỉ **51 câu đọc từ bảng gold** —
> nghĩa là **nhiều đáp án đúng được đọc từ bảng KHÔNG thuộc tập gold**. Cùng một con số nằm ở
> nhiều bảng (bảng tổng hợp và bảng chi tiết). Đây chính là cơ chế của synera: **TABLES_F2 = 0,0
> mà EXEC 0,66**. Suy ra: **tối ưu cho việc trùng bảng gold KHÔNG đồng nghĩa tối ưu cho việc trả
> lời được**, và TABLES_F2 cao của ta (0,5641 > 0,4118 của top1) không mua được điểm đọc số.

Hai hệ quả hành động:

1. **Nút thắt trội là *bỏ trả lời*, không phải chọn sai ô.** Bộ khớp nhãn mạnh nhất của ta không
   chịu bắn ở 59–76% câu **dù đang giữ bảng gold trong tay**. Và nó bắn nhiều hơn khi được đưa
   *nhiều* bảng hơn — nên "tăng số bảng ứng viên" là đòn rẻ có thể đang bị dùng thiếu.
2. **Quy ước dấu: signed ≥ magnitude ở CẢ hai chế độ** (8,7 vs 7,5 và 10,8 vs 10,6). `lookup.
   synthesize` ghi rằng điều này "unmeasurable without labels" — giờ đo được, và nó nghiêng về
   **không bọc `abs()` quanh giá trị đơn**. `compose.py` và `ratio.py` đang truyền
   `magnitude=True`; đó là điều đáng thử đảo trên bản nộp thật. Lưu ý phân biệt: quy tắc `abs()`
   của BTC nói về **hiệu không nêu chiều**, không phải về từng toán hạng.

> **Bẫy đo lường thứ hai trong ngày.** Bản đầu của `rule_reading` trả `Lookup.value` **thô**, nên
> ô ở đơn vị triệu/tỷ bị so trực tiếp với gold ở đơn vị câu hỏi — sai cả triệu lần, và tôi đã báo
> "10% dù có bảng gold" dựa trên nó. Đã nối `column_scale`, chính bộ quy đổi mà bản nộp dùng.
> Cùng một lỗi hình dạng với bẫy dung sai ở mục 8: **thứ dùng để đo không được tự đoán thang đo.**

**Chưa so được với 0,37 trên bảng xếp hạng.** Các số trên là `lookup.find` **đơn lẻ**; bản nộp
chạy cả chuỗi dự phòng (`find_ratio` → `compose.resolve` → `find_best_effort` →
`synthesize_scan`) và chuỗi đó nằm **inline trong `run_submit.py`**, vốn đã regressed. Nối chuỗi
đó vào harness là việc kế tiếp, và cũng là điều kiện để chấm adapter cho có nghĩa.

### 11. Chuỗi thật đã vào harness — và nó tự kiểm chứng

`eval_gold.py --chain` chạy đúng thứ tự ưu tiên của `run_submit.py`
(`ratio.resolve` → `screen*` → `compose.resolve` → `corroborator.choose`+`lookup.find` →
`find_ratio` → `choose_best_effort`+`find_best_effort` → `synthesize_scan`). Bỏ các nhánh
`located`/`panel`/`generated`/`planned` vì chúng đọc cache theo id đề BTC, không tồn tại cho
bản ghi gold — nên đây là **sàn tất định** mà các nhánh model được đắp lên.

> **Bằng chứng chuỗi được nối đúng:** nhánh `fallback` đo được **5%** đúng, còn ghi chú cũ của
> dự án ghi **5,9%**. Một con số đã biết được tái tạo từ mã độc lập.

Hai câu hỏi treo, trả lời trên 416 bản ghi, **không tốn lượt nộp nào**:

| `search_k` | đúng | `fallback` (số câu / % đúng) | `lookup` (số câu / % đúng) |
|---|---|---|---|
| 8 | 14,4% (60) | 277 / 12% | 134 / 20% |
| 20 | 15,6% (65) | 267 / 12% | 149 / 22% |
| **40** | **17,8% (74)** | 244 / 13% | **172 / 24%** |

Cơ chế đọc được rõ, không phải nhiễu: **nới ứng viên đẩy câu từ nhánh `fallback` (đúng 12%)
sang nhánh `lookup` (đúng 24%)**. `run_submit.py` mặc định `search_k=10`. Đây là lần đầu độ sâu
truy hồi *cho khâu trả lời* được đo — vòng quét `ksweep_*` cũ chỉ đổi **số bảng khai**, giữ
nguyên đáp án và evidence (mục 536), nên nó đo TABLES_F2 chứ không đo cái này.

| `search_k=20` | đúng |
|---|---|
| magnitude (đang dùng) | 15,6% (65) |
| `--signed` | 15,6% (65) |

> **Quy ước dấu: hoà tuyệt đối — ĐỪNG đảo.** Ở mục 10 tôi báo signed ≥ magnitude và đề nghị
> đảo `magnitude=True`; đó là **artefact của việc đo `lookup.find` đơn lẻ**. Trên chuỗi thật,
> hai cấu hình cho đúng cùng số câu. Bài học: một hiệu ứng đo trên **một thành phần** không suy
> ra được cho **cả chuỗi**, vì chuỗi có các nhánh khác hấp thụ đúng những câu mà thành phần đó
> đổi ý.

### 12. Nút thắt generator đã vá — `VIFINQA_POOL_INDEX`

`generation/embedding_index.py` giờ có **`LexicalTableIndex`** (BM25 thuần Python, không thêm
dependency) bên cạnh `TableIndex` dense, cùng chữ ký `search(query, *, top_k, filter_fn)`.
`build_pool_index` gọi `make_table_index`, **mặc định lexical**; đặt `VIFINQA_POOL_INDEX=dense`
để trả lại hành vi cũ.

Vì sao dám đổi mặc định — đo trên bảng thật, pool 300 bảng đúng bằng `POOL_TABLE_CAP`
(`scripts/_probe_lexical_pool.py`, 60 câu gold):

| | |
|---|---|
| dựng chỉ số | **58 ms** / seed (dense: hàng phút, embed 300 bảng trên CPU) |
| truy vấn | 2,3 ms |
| bảng gold trong top-1 / top-5 / top-10 | **41,7% / 70,0% / 78,3%** |

Tốc độ không phải lý do đủ — một chỉ số dựng tức thì mà xếp hạng tệ chỉ biến 12 ngày thành cách
sinh rác nhanh hơn. Nên chất lượng được đo trước, và **top-1 41,7% trùng con số 41,4%** dự án đã
đo cho reranker theo cụm chỉ tiêu (bẫy #9). Sau khi vá, phần chỉ số tốn 60 ms/seed nên nút thắt
chuyển về đúng chỗ nó phải ở: **các lượt gọi LLM**.

> Còn một đường dense thứ hai chưa chạm: `generation/table_index/store.py::NumpyTableIndexStore`
> (dùng bởi `peer_group_longitudinal_report`). Nó cache theo `(ticker, scope, period)` nên khoá
> **ổn định giữa các seed** và tái dùng được — khác hẳn `build_pool_index`. Chưa cần đổi, nhưng
> nếu tầng intermediate lại chậm thì đây là chỗ nhìn tiếp.

### 13. ⛔ Không vá `search_k=40` vào bản nộp — phép đo không chuyển giao

Mục 11 đo được chuỗi tất định tốt hơn khi nới ứng viên (14,4% ở `search_k=8` → 17,8% ở 40) và
tôi định vá phần đó vào ~430 câu tất định. `scripts/_chain_vs_shipped.py` kiểm điều kiện tiên
quyết: **ở cùng k=10, chuỗi trong harness chỉ tái tạo 50,6% đáp án tất định đang nộp** (43/85
trên mẫu 200 câu).

Nên chuỗi harness **không phải** đường ống đã tạo ra bản nộp — nó thiếu tầng cờ mất hôm 10/08
(`shape_lock`, `label_rescan`, `label_declare`, `consensus`, `credible_gate`, đường `ma_so`) và
các nhánh `located`/`panel`/`planned`. Vá k=40 vào sẽ đổi **27,1%** đáp án của tập con đó, mù
hoàn toàn về việc đổi thành tốt hay xấu, và nó là bẫy *"cơ chế mới bị cơ chế cũ ghi đè"* chạy
ngược chiều.

> **Điều kiện để lấy được phần điểm này:** khôi phục tầng cờ đã mất từ
> `scripts/__pycache__/run_submit.cpython-313.pyc` bằng `scripts/_recover_pyc.py` (xem mục ⛔
> đầu file), rồi mới nới `search_k`. Không có bước đó thì con số 17,8% chỉ nói về một đường ống
> ta không nộp.

Đây là **đề xuất thứ tư của chính tôi bị harness bác trong một buổi chiều** — sau "dùng ANSWER
làm trục riêng", "236 câu dẫn xuất sai chỗ", và "đảo `magnitude`". Giá của mỗi lần bác là vài
phút; giá nếu không bác là một lượt nộp và một kết luận sai nằm lại trong tài liệu.

### 14. Dừng sinh medium (11/08 15:50)

Round 1 chạy **3 giờ 7 phút** ra **2 bản ghi**. Ngoại suy 500 bản ghi ≈ **12 ngày**, chưa
tính tầng intermediate. Nút thắt: mỗi seed dựng lại pool index tới 300 bảng bằng bge-m3
**trên CPU**, seed đổi mỗi vòng nên cache trúng thấp — cache phình 1.402 → 45.286 file trong
3 giờ, phần lớn đổ vào seed bị `feasible=false` loại. Máy gen đã huỷ; **Qwen3-14B mất theo**.
Hệ quả: đường fine-tune chốt ở **chỉ tầng easy** — dạy đúng nửa mà bộ khớp nhãn vốn đã thắng.
Nếu quay lại sinh dữ liệu, **phải sửa nút thắt embedding trước**, đừng chạy lại y nguyên.

### 15. Bẫy #12 — `per_device_eval_batch_size` mặc định là 8

`train_qlora.py` đặt `per_device_train_batch_size=1` nhưng bỏ trống cái tương ứng cho eval,
mà HuggingFace mặc định nó là **8**. Bước train ở 12k vừa khít, rồi lượt eval ngay sau đó
gánh tám chuỗi 12k cùng lúc và OOM — traceback nằm trong `prediction_step`, không phải trong
`training_step`, và adapter không bao giờ được ghi. Thêm `per_device_eval_batch_size=1`,
`prediction_loss_only=True`, `eval_accumulation_steps=1`: đỉnh VRAM 23,5 GB (OOM) → **16,9 GB**.
`prediction_loss_only` bỏ logits `[1, 12288, ~152k]` mà Trainer giữ lại và không ai đọc.

### 16. `eval_loss` của run train KHÔNG đo được việc ta cần (11/08 17:30)

Run thật: `--data sft_easy.jsonl --out /workspace/lora_easy --epochs 2 --max-len 12288
--eval-frac 0.15` → 380 cặp = 323 train / 57 eval, 42 bước. Epoch 1: `eval_loss` **0,5646**,
`eval_mean_token_accuracy` **0,8263**, `checkpoint-21` đã ghi. Cơ học lành: VRAM 20.830 MiB
ổn định, util 89–93%, bước 23/42 ở 1:37:40, ETA **≈18:36**.

**Nhưng hai con số đó gần như không nói gì về EXEC.** `dataset_text_field="text"` nhận cả
chat template đã render thành một chuỗi, không có collator completion-only, không
`assistant_only_loss` → loss tính trên **mọi** token, prompt vào hết. Đo bằng
`scripts/_probe_sft_split.py` trên chính file train:

| | median |
|---|---|
| prompt (system 6.4k + bảng) | **20.052 ký tự** |
| completion (`result = num(df1, 4, 1)`) | **55 ký tự** |
| completion / tổng | **0,26%** |

Nên **99,74% mục tiêu là "đoán ô tiếp theo của bảng cân đối"** cộng với việc học thuộc một
system prompt 6,4k ký tự **giống nhau ở cả 380 mẫu**. Cú rơi 2,124 → 0,5646 (so với smoke
2 bước) chủ yếu là chuyện đó, và nó xảy ra trong chưa tới một epoch. Phần token thực sự
quyết định EXEC — chỉ số frame, dòng, cột — là ~4 token/mẫu: cả run 42 bước × 16 mẫu ≈
**2.700 lượt cập nhật có ích** sau 3 giờ GPU.

Hệ quả không phải "vô ích" mà là **có thể tệ hơn base**: ta đang tối ưu một mục tiêu khác
(mô hình hoá văn bản bảng) nên adapter có quyền trôi khỏi khả năng viết chương trình của
base. **Bắt buộc A/B với base, đừng mặc định là cải thiện.**

Đã vá cho lần sau (`scripts/train_qlora.py`): tách `prompt`/`completion` thay vì render
`text`, thêm `completion_only_loss=True` — trl **1.9.2** có sẵn cả `completion_only_loss`
và `assistant_only_loss`. Cùng 3 giờ đó, toàn bộ gradient rơi vào target. Không nhanh hơn
(forward vẫn phải nuốt cả prompt), nhưng `eval_mean_token_accuracy` từ đó trở thành số
đọc được: tỷ lệ token chương trình đúng.

Phụ: `pairs over the cap: 1/64` — ~1,5% cặp mất completion do truncation ở 12.288. Bỏ qua được.

### 17. A/B held-out: adapter **thua** base (11/08 18:50)

`scripts/ab_heldout.py` trên đúng 57 câu slice đầu của `sft_easy.jsonl` (cái
`train_qlora.py` giữ lại), chấm bằng **thực thi** với `abs_tol=0.01` — không chấm
tuple ô, vì base viết `find_row` còn target là `num(df,r,c)` cứng.

| | runnable | **correct** | cell-match (SFT dialect) | scale-match |
|---|---|---|---|---|
| `Qwen/Qwen3.5-9B` (base) | 33/57 (57.9%) | **17/57 (29.8%)** | 0/57 | 0/57 |
| `vifin-easy-lora` | 44/57 (77.2%) | **13/57 (22.8%)** | 14/57 (24.6%) | 50/57 (87.7%) |

Adapter học **đúng thứ nó được thưởng** trong loss không mask: dialect
(`num(...)`, scale ≈ luôn có vì `1.0` là majority), runnable tăng vì format ổn
hơn — và **đúng ô thì kém hơn base** (−4 câu / −7 điểm phần trăm). Đúng dự đoán
mục 16: tối ưu mô hình hoá bảng → trôi khỏi kỹ năng tìm số. **Không dùng adapter
này để gen submission.**

Hành động tiếp: (a) train lại với bản đã vá `completion_only_loss` — cùng 380 cặp,
cùng 3 giờ, nhưng loss chỉ trên target; hoặc (b) bỏ fine-tune easy-only và dùng
base/`retry14b` cache. Không gen submission bằng adapter vừa train.

### 18. ⛔ Tập SFT lệch phân bố đề **56 điểm** — đây mới là lý do exam tụt (14/08)

`scripts/_probe_exam_classes.py` phân loại cả 1.012 câu đề và 1.614 cặp
`sft_locate2.jsonl` bằng **cùng** bộ luật của `_class_sizes.py`:

| lớp câu | đề | tập train | lệch |
|---|---|---|---|
| tra cứu **một ô** | 36,6% | **92,7%** | **+56,1** |
| sàng lọc nhóm / xếp hạng | 19,8% | 1,5% | −18,3 |
| cực trị (cao/thấp nhất) | 14,7% | 0,1% | −14,7 |
| số học **hai ô** (chênh lệch, so-với-năm) | 14,3% | 0,2% | −14,1 |
| tỷ lệ / phần trăm | 11,7% | 3,8% | −7,9 |
| số học hai công ty | 1,5% | 1,8% | +0,3 |
| đếm trong nhóm | 1,4% | 0,0% | −1,4 |

Nghĩa là: **held-out 69,4% đo trên phân bố 92,7% một-ô, còn đề chỉ 36,6% là dạng
đó.** Held-out cao không dự báo được exam vì hai phân bố khác nhau về bản chất.
Và nó không chỉ là *trần*: LoRA dạy "đáp án luôn là một ô" nên **ghi đè** khả năng
viết chương trình nhiều bước của base — khớp đúng mục 17 (dialect lên, đúng ô
xuống). Đó là **cơ chế gây tụt**, mạnh hơn giả thuyết "thiếu cặp mù".

63,4% đề cần nhiều hơn một ô hoặc một phép lọc. Bốn lớp thiếu nhất
(sàng lọc + cực trị + hai ô + tỷ lệ) = **60,5% đề** nhưng chỉ **5,6% tập train**.

Lưu ý cấu trúc cho hướng fine-tune: câu sàng lọc nhóm thường cần 7+ bảng, prompt
ta chỉ mang 8 bảng và mỗi bảng bị cắt ở 7.000 ký tự — model **không nhìn thấy đủ
dữ liệu** để trả lời lớp 19,8% này. Lớp đó thuộc mã tất định, fine-tune không vá
được.

### 19. Pool medium 30% sai theo cấu tạo — đừng cứu, đừng sinh thêm (14/08)

`scripts/_audit_medium.py` trên 343 bản ghi medium:

- **103/343 (30,0%)** câu hỏi nêu một mã **không có bảng nào** trong
  `relevant_tables` để trả lời → đáp án ghi lại **sai so với câu hỏi**. Ví dụ:
  hỏi tổng của ASM + NKG + VPI, chỉ cite 2 bảng, `answer` trùng **từng chữ số**
  với bản ghi chỉ hỏi ASM + NKG.
- **204/343 (59%)** dùng chung đáp án với một câu khác (67 nhóm) — biến thể câu
  hỏi quanh cùng một con số, trong đó có biến thể đúng và biến thể sai.
- Số ô mà chương trình gold đọc: **1 đến 12** (chỉ 86 bản ghi đọc đúng 2).

Nên `build_sft.py` giữ được 44/343 **không phải vì bộ dựng dở** mà vì phần lớn
pool không diễn đạt được bằng một ô — và 30% thì không nên diễn đạt bằng gì cả.
Một `locate_two_cells` brute-force đã được viết rồi **xoá**; lý do ghi tại chỗ
trong `build_sft.py`, đọc trước khi viết lại lần thứ ba.

### 20. Tập train khớp phân bố đề, đáp án đúng theo cấu tạo — `sft_mixed.jsonl` (14/08)

Không LLM nào tham gia sinh. Chuỗi: `gen_shapes.py` → `rank_context.py` →
`build_sft.py` (đường nhanh `pregenerated`) → `blend_sft.py`.

Vì sao tin được đáp án:

- Số liệu lấy từ panel Thông tư 200; **3 đẳng thức kế toán đúng 100%** trên
  478/455/493 nhóm kiểm được.
- Câu hỏi được viết **từ** phép tính, nên không thể lệch khỏi đáp án — đúng chỗ
  pool medium sai (30% hỏi mã không có bảng).
- `build_panel` giờ mang **địa chỉ ô** (`Source.row/column/scale`), và
  `_verify_cells.py` xác nhận **16.430/16.430 = 100%** địa chỉ đọc lại đúng số qua
  chính `num()` mà model được cấp.
- Mỗi bản ghi được **thực thi** rồi đối chiếu đáp án trước khi ghi; `build_sft`
  thực thi **lần hai** sau khi gán tên frame. 4.900/4.900 cặp qua cả hai.

Phân bố đạt được (chuẩn hoá trên 797/1012 câu train được):

| lớp | đề | `sft_mixed` |
|---|---|---|
| tra cứu một ô | 46,4% | **46,4%** |
| cực trị | 18,7% | **18,7%** |
| số học hai ô | 18,2% | **18,2%** |
| tỷ lệ / phần trăm | 14,8% | **14,8%** |
| số học hai công ty | 1,9% | **1,9%** |

2.662 cặp, median ≈ 8.626 token, p90 ≈ 10.807, **3,0% vượt trần 12.288** (bị
`under_cap` loại). Phần được tính gradient dài hơn hẳn cho hình dạng nhiều bước:
`argmax_year` 211 ký tự vs một-ô 55 — tức loss lần này thật sự đo cấu trúc
chương trình chứ không chỉ một lời gọi `num`.

**Loại trừ có chủ ý:** sàng lọc nhóm + đếm = 21,2% đề. Chúng cần một báo cáo cho
mỗi công ty (4–7 công ty), prompt chỉ chở 8 bảng cắt ở 7.000 ký tự trong 12k
token — model **không nhìn thấy đủ** để trả lời, train bao nhiêu cũng vậy. Lớp đó
thuộc nhánh panel tất định.

**Hai bẫy đã cắn, ghi để không cắn lại:**

1. `run_query` chỉ bind tên `df` khi có **đúng một** frame; nhiều frame thì bind
   `df1..dfn` **theo vị trí**. Tự đặt `["df","df1",...]` làm `df` không tồn tại
   *và* lệch mọi frame còn lại → **45.492/45.492** ứng viên hai-bảng chết im
   lặng trong khi hình dạng một-bảng vẫn qua. Dùng `variable_names()`.
2. `portability_problems` chặn `lambda` và comprehension (grader py37) →
   **2.137/2.137** ứng viên argmax chết. Viết argmax thành chuỗi `if` tường minh.

**Chưa kiểm vòng này:** cặp mù (abstention). Dạng đích chương trình buộc trả về
một số nên không nói được "không có"; chỉ dạng locate có `KHONG_CO`. Ưu tiên lệch
phân bố trước vì nó có **cơ chế gây tụt** đo được, còn abstention chỉ là giả
thuyết.

### 21. Panel chết vì đọc cột Mã số như nhãn — đã vá (20/08)

Circular 200 chuẩn: `[Mã số | Chỉ tiêu | …]`. `detect_statement` / extract đọc
`row[0]` → thấy `"100"`/`"110"` → `stmt=None`. VIC 2022 có Bảng CĐKT hợp nhất rõ
mà panel gần như trống (`net_revenue` thôi).

Vá: dùng `label_column` từ `lookup.py`, cho `find_code_column` tìm từ cột 0,
truyền `label_col` vào `value_columns`. Đo lại:

| | trước | sau |
|---|---|---|
| nhóm (ticker,year,scope) | 1.348 | **1.943** |
| `liabilities_short` có mặt | 32% | **58%** |
| đẳng thức kế toán | 100% | 100% (nhiều nhóm hơn) |
| câu prize COMPLETE | 71 (7%) | **180 (17.8%)** |
| cohort COMPLETE | 11 | **66** |

COMPLETE = mọi ô cần cho câu hỏi có trong panel → viết được chương trình tất định.
Catalog tỷ số của BTC (`vifinqa-official/.../panel/catalog.py`) khớp vocabulary
đề (D/E, current ratio, interest coverage, CFO margin…) — bước tiếp theo là port
sang nhánh trả lời, không phải sinh thêm dữ liệu LLM.

## Trạng thái (10/08/2026)

Bản tốt nhất trên leaderboard (giữ nộp): **`screen_ratio_gated.zip` (ID 2752)** —
EXEC **0.3281** / ANSWER **0.3300** / TABLES_F2 **0.5641**.

> Đã lỗi thời: bản **2795** (`retry14b.zip`) trên bảng ngày 11/08 đạt
> EXEC **0.3696** / ANSWER **0.3715** / TABLES_F2 **0.5641** — xem mục 11/08 chiều ở trên.

Nền liền trước: `helpers.zip` (num+find_row) 0.3261 / 0.3281 / ~0.5642. Một câu ăn điểm: **id=390**.

`threshold_cohort.zip` (2755) hòa tuyệt đối với 2752 — **không dùng làm nền**.

Suy luận chi tiết 10/08: `PHAN-TICH.md` mục cùng ngày. Khi lệch, tin bảng số ở đây + zip trong
`submissions/`.

### Lịch sử ngắn (macro / mốc)

| Bản | EXEC | ANSWER | TABLES_F2 | Ghi chú |
|---|---|---|---|---|
| `anchor_search` | 0.2984 | 0.3004 | 0.5618 | chỉ mục caption |
| `spank_lastresort` | 0.3241 | 0.3261 | 0.5620 | k=2·span; lastresort |
| `helpers` | 0.3261 | 0.3281 | ~0.564 | `num`/`find_row` |
| **`screen_ratio_gated` (2752)** | **0.3281** | **0.3300** | **0.5641** | +390 NKG |
| `threshold_cohort` (2755) | 0.3281 | 0.3300 | 0.5641 | +397 không điểm |

---

## Trạng thái cũ (07/08/2026) — giữ để lần ngược

Bản tốt nhất lúc đó: **`anchor_search.zip` — EXEC 0.2984, ANSWER 0.3004, TABLES_F2 0.5618.**

### ⚠️ Bảng xếp hạng KHÔNG phải điểm cuối

`context.txt` dòng 45: điểm cuối = **trung bình macro BA tiêu chí** (TABLES_F2, ANSWER, EXEC).
Leaderboard chỉ *sắp xếp* theo EXEC và chỉ hiển thị bản nộp mới nhất. Tính lại theo thước đo thật:

| Đội | TABLES_F2 | ANSWER | EXEC | **Macro** |
|---|---|---|---|---|
| BM25 Baseline (BTC, đáp án gốc) | 0.8934 | 1.0 | 0.3577 | 0.7504 |
| synera | **0.0** | 0.6601 | 0.6601 | 0.4401 |
| nguyenvuhoanglong | 0.5535 | 0.3142 | 0.3123 | **0.3933** |
| **ta (vilamiu)** | **0.5618** | 0.3004 | 0.2984 | **0.3869** |
| yyy | 0.3526 | 0.2885 | 0.2885 | 0.3099 |

**Ta đang hạng 3 theo thước đo thật, không phải hạng 5.** Đã vượt nguyenvuhoanglong ở TABLES_F2;
còn cách họ **0.0064 macro**, tương đương khoảng **10 câu đúng** trên EXEC + ANSWER.

### Sáu lượt nộp trong hai ngày

| Bản | Đổi đáp án | EXEC | ANSWER | Δ câu | Thay đổi chính |
|---|---|---|---|---|---|
| `bestlabel` | — | 0.2016 | 0.2036 | — | nền |
| `compose` | 157 | 0.2273 | 0.2292 | **+13** | `extract_metric` + `label_column` + hợp thành theo năm |
| `dtypefix` | 4 | 0.2273 | 0.2292 | 0 | Bẫy #15 (`read_csv` suy kiểu) |
| `axis` | 19 | 0.2352 | 0.2391 | +4 | hợp thành theo công ty + 3 lỗi thật |
| `scalefix` | 45 | 0.2391 | 0.2431 | +2 | `_checked_scale` + loại độ lớn bất khả thi |
| `divide` | 32 | 0.2431 | 0.2470 | +2 | sàng lọc hai tầng + chia tỷ lệ |

**+21 câu đúng, không thuê GPU, không model mới.** Nhưng lợi ích **giảm dần rõ rệt**: +13, 0, +4,
+2, +2 — vá từng lớp câu đã cạn.

### Bước nhảy 07/08 — đổi thứ được đánh chỉ mục, không vá thêm nhánh

| Bản | Đổi đáp án | EXEC | ANSWER | TABLES_F2 | Thay đổi |
|---|---|---|---|---|---|
| `label_doc` | — | 0.2846 | 0.2866 | 0.4718 | nền |
| `decl_k12` | 16 | 0.2866 | 0.2885 | 0.3577 | khai 12 bảng — **thua** |
| `decl_k20` | 16 | 0.2866 | 0.2885 | 0.2833 | khai 20 bảng — **thua nặng** |
| `anchor_search` | 128 | **0.2984** | **0.3004** | **0.5618** | chỉ mục theo văn xuôi + k thích ứng + đường trả lời |
| `ksweep_lo/hi/max` | 0 | 0.2984 | 0.3004 | 0.5561 / 0.5577 / 0.5498 | quét k — xác nhận 5.94 là đỉnh |

Một thay đổi ăn **+0.090 TABLES_F2 và +0.012 cả EXEC lẫn ANSWER** — nhiều hơn tổng năm lượt trước
đó cộng lại. Nguyên nhân gốc ở mục "Vì sao khớp nhãn có trần 42%" bên dưới.

### Độ lớn đáp án trong bản hiện tại

| | Câu |
|---|---|
| tiền, độ lớn hợp lý | 611 |
| tỷ lệ, độ lớn hợp lý | 184 |
| **tỷ lệ > 1e4 (bất khả thi)** | **111** |
| không có đơn vị (câu đếm…) | 91 |
| đáp án = 0 | 10 |
| **> 1e16 đồng (bất khả thi)** | **5** |

111 câu tỷ lệ bất khả thi còn lại **phần lớn là sàng lọc hai tầng theo tỷ số**. Kết luận “đừng
ghép” đã **hết hiệu lực có điều kiện** (có registry) — xem PHAN-TICH 07/08 tối và 10/08 (clean
screen +390; threshold 397 hòa).

## Vì sao khớp nhãn có trần 42% — đọc `vifinqa-official/` trước khi tự suy diễn

Thư mục **`vifinqa-official/` trong repo là gói mã đầy đủ do BTC phát hành** (generation +
retrieval + evaluation). Nó nằm đó suốt nhiều phiên và trả lời trực tiếp những câu ta đã đoán sai
bằng thực nghiệm đắt tiền. Ba file quyết định:

- `src/vifinqa/generation/common.py:433` — `relevant_tables = used_refs`, tức gold **đúng bằng các
  bảng mà `pandas_query` đọc**. Nên g ≈ số cặp (mã, năm) câu hỏi chạm tới.
- cùng file, `judge_and_maybe_rewrite` — bộ duyệt "tự nhiên" **loại mọi câu hỏi chép nguyên văn
  nhãn dòng hoặc nhãn cột**.
- `src/vifinqa/encoding/table_text.py` — `table_retrieval_text` là văn bản họ đánh chỉ mục: tên
  đầy đủ công ty + cột + 20 nhãn dòng đầu + **2.500 ký tự văn xuôi quanh bảng**.

Ghép hai điều đầu: **theo thiết kế của bộ sinh dữ liệu, từ ngữ câu hỏi là từ ngữ KHÔNG có trong
bảng.** Nó nằm ở tiêu đề thuyết minh giới thiệu bảng ("5.2. Phải thu ngắn hạn của khách hàng").
`lexical.py:_table_tokens` chỉ đánh chỉ mục caption + nhãn — tức đúng cái phần được viết ra để
tránh. Đó là trần 42%, và nó là lỗi thiết kế chứ không phải thiếu tinh chỉnh.

Tỷ lệ bảng chứng cứ lấy lại được, đo trên chính xếp hạng đã tìm ra chúng (nên **thiên vị chống
lại** hai cột phải):

| cắt | bm25 (nhãn) | context (cả trang) | **anchor (đoạn sát bảng)** |
|---|---|---|---|
| 3 | 60.1% | 58.9% | **62.9%** |
| 4 | 65.0% | 64.3% | **69.4%** |
| 5 | 69.0% | 69.6% | **74.5%** |
| 30 | 86.2% | 92.6% | **94.4%** |

`context` (cả trang) thua vì **mọi bảng trên cùng một trang nhận y hệt một đoạn văn**. RRF hợp nhất
thắng ở top-30 (95.3%) nhưng không nhúc nhích ở top-4 — vô dụng, xem công thức F2 ngay dưới.

## Công thức F2 — trực giác "β=2 nên ưu tiên recall" dẫn tới quyết định SAI

Với g bảng gold, k bảng khai, h bảng khai đúng: **F2 = 5h / (4g + k)**. Khớp cả bảy điểm đo.

Trực giác "β=2 nên recall quan trọng gấp 4 lần" **chỉ đúng khi P ≈ R**. Khi P ≪ R thì mẫu số 4P+R
bị R chi phối và F2 ≈ 5P — F2 trở thành hàm của **precision**. Đó là lý do khai sâu hơn *giảm*
điểm dù recall tăng mạnh (0.54 → 0.78 mà F2 rơi 0.4718 → 0.2833).

Quét k trên xếp hạng anchor, **cùng đáp án cùng evidence, chỉ khác số bảng khai**:

| avg k | TABLES_F2 | P | R |
|---|---|---|---|
| 5.00 | 0.5561 | 0.3657 | 0.6863 |
| **5.94** | **0.5618** | 0.3314 | 0.7229 |
| 6.82 | 0.5577 | 0.3060 | 0.7454 |
| 7.81 | 0.5498 | 0.2846 | 0.7622 |

Đối xứng quanh 5.94 → **đó là đỉnh, đòn k đã hết**. Muốn tăng F2 chỉ còn cách tăng h.

k **phải thay đổi theo câu**: 441/1012 câu chỉ hỏi một mã một năm (g=1), khai 4 bảng thì F2 trần
chỉ 5/(4+4) = 0.625 dù truy hồi hoàn hảo; câu span=8 mà khai 4 thì bị khai *thiếu* nghiêm trọng.
Luật đang dùng: `k = clamp(2 · ĝ, 3, 11)` với `ĝ = max(#evidence, min(span, 8))`, span = #mã × #năm.
Trung bình span của 1012 câu = 2.65, g đo từ leaderboard = 2.40–2.72 — **span dự đoán được g**.

## Tái lập chính xác bản 2410 (`bestlabel.zip`, EXEC 0.2016 / ANSWER 0.2036)

**Một lệnh duy nhất**, với 6 artifact đã có sẵn trong `artifacts/`:

```bash
PYTHONPATH=src python scripts/run_submit.py 10 5 artifacts/generated_full.jsonl
# -> submissions/final.zip
```

Đối số: `search_k=10` (độ sâu tìm kiếm), `declare_k=5` (số ref khai vào `relevant_tables`).

Cờ trong `scripts/run_submit.py` phải đúng như sau — đây là phần dễ sai nhất khi tái lập:

```python
USE_PANEL = False      LOCATE_SOURCE = "union"
USE_RATIO = False      USE_TIEBREAK = True
USE_LOCATE = True      USE_PLAN = True
LOCATE_WINS = False
```

### 6 artifact được `run_submit.py` đọc

| Artifact | Sinh bằng | Cần GPU | Thời gian |
|---|---|---|---|
| `tables.parquet` | `run_extract.py` | không | 18s |
| `reranked_metric.jsonl` | `run_rerank.py --query-mode metric` | GPU 8 GB local | 7 ph |
| `located.jsonl` | `run_locate.py` | **GPU thuê** | 6 ph |
| `embed_located.jsonl` | `run_embed_locate.py --min-score 0.55` | GPU 8 GB local | 33s |
| `planned.jsonl` | `run_plan.py` | **GPU thuê** | 8 ph |
| `generated_full.jsonl` | `run_generate.py --workers 6` | OpenRouter | 40 ph |

`panel_answers.jsonl` cũng được đọc nhưng **bị bỏ qua** vì `USE_PANEL = False`.

### Artifact từ điển mã số (07/08 khuya)

| Artifact | Sinh bằng | Cần GPU | Thời gian |
|---|---|---|---|
| `ma_so.json` | `build_ma_so.py` | không | 11s |

344 mã theo Thông tư 200, mỗi mã kèm tối đa 25 biến thể nhãn thực tế và tần suất, dựng từ 6.831
bảng có cột Mã số. Kiểm bằng `verify_ma_so.py`: **4.472/4.488 đẳng thức kế toán đúng (99,6%)**.

**Khớp theo mã là khớp CHÍNH XÁC**, thay cho khớp nhãn mờ đang chặn đường trả lời ở 42%. Mã chỉ có
ở các báo cáo chính (bảng cân đối, kết quả kinh doanh, lưu chuyển tiền tệ) — thuyết minh không có
— nhưng đó đúng là nơi chứa hầu hết chỉ tiêu mà câu hỏi nhắc tới.

⚠️ **Chuẩn hoá dấu trước khi tính.** Giá vốn (`kqkd:11`) in trong ngoặc kế toán ở một số báo cáo
(âm) và in dương ở số khác. Công thức nào đọc mã chi phí mà không lấy trị tuyệt đối sẽ sai ngẫu
nhiên tuỳ báo cáo, và sai kiểu đó **không nhìn ra được từ kết quả** vì con số vẫn hợp lý.

### Artifact truy hồi theo văn xuôi (07/08) — nguồn của bước nhảy

| Artifact | Sinh bằng | Cần GPU | Thời gian | Dùng ở đâu |
|---|---|---|---|---|
| `anchor_index.jsonl` (136 MB) | `build_anchor_index.py` | không | 32s | đầu vào của xếp hạng |
| `context_index.jsonl` (183 MB) | `build_context_index.py` | không | 67s | chỉ để so sánh — **thua**, đừng dùng |
| `anchor_rank.jsonl` | `rank_context.py 30 …anchor_index.jsonl …anchor_rank.jsonl` | không | 82s | khai báo + đường trả lời |
| `anchor_keys.jsonl` | chuyển định dạng từ `anchor_rank.jsonl` | không | tức thì | `run_plan`, `run_locate` |
| `anchor_group.jsonl` | `PER_GROUP=6 GROUP_CAP=30 rank_context.py … balanced` | không | 53s | `run_generate` |

Ba định dạng vì ba nơi tiêu thụ khác nhau, **không phải trùng lặp**:

- `anchor_rank.jsonl` — phẳng top-30, khoá `refs`. `run_submit` đọc cái này.
- `anchor_keys.jsonl` — cùng nội dung, khoá `keys`, vì `run_plan`/`run_locate` đọc định dạng của
  `reranked_metric.jsonl`.
- `anchor_group.jsonl` — **sâu 6 bảng mỗi (mã, năm)**, tb 14.6 khoá / 2.83 tài liệu. Bắt buộc cho
  `run_generate`: docstring của nó ghi rõ danh sách *phẳng* làm tỷ lệ chương trình chạy được tụt
  49.9% → 26.4%, vì câu so sánh năm công ty bị dồn hết vào một công ty và chương trình thiếu toán
  hạng thì không chạy nổi. Đây là Bẫy #10 ở dạng khác — **rerank giúp nơi vấn đề là *liên quan*,
  hại nơi vấn đề là *phủ sóng***.

### Dựng lại từ số 0

```bash
# 1. Không cần GPU
PYTHONPATH=src python scripts/run_extract.py
PYTHONPATH=src python scripts/run_parse.py

# 1b. Chỉ mục theo văn xuôi (không GPU, ~3 phút tất cả)
PYTHONPATH=src python scripts/build_anchor_index.py
PYTHONPATH=src python scripts/rank_context.py 30 artifacts/anchor_index.jsonl artifacts/anchor_rank.jsonl
PER_GROUP=6 GROUP_CAP=30 PYTHONPATH=src python scripts/rank_context.py 30 \
    artifacts/anchor_index.jsonl artifacts/anchor_group.jsonl balanced
python -c "import json;out=open('artifacts/anchor_keys.jsonl','w',encoding='utf-8');\
[out.write(json.dumps({'id':r['id'],'keys':[[x['doc_name'],x['table_id']] for x in r['refs']]})+'\n')\
 for r in map(json.loads,open('artifacts/anchor_rank.jsonl',encoding='utf-8'))]"

# 2. GPU local 8 GB (bge-reranker-v2-m3 và bge-m3, mỗi model ~1,2 GB fp16)
PYTHONPATH=src python scripts/run_rerank.py --candidates 30 --query-mode metric
PYTHONPATH=src python scripts/run_embed_locate.py --min-score 0.55

# 3. OpenRouter (cần OPEN_ROUTER_KEY trong .env)
PYTHONPATH=src python scripts/run_generate.py --workers 8 \
    --use-rerank --reranked artifacts/anchor_group.jsonl --cache artifacts/gen_anchor.jsonl

# 4. GPU thuê 24 GB — xem mục "Dựng máy thuê" bên dưới
PYTHONPATH=src python scripts/run_locate.py --workers 14 --reranked artifacts/anchor_keys.jsonl \
    --cache artifacts/located_anchor.jsonl
PYTHONPATH=src python scripts/run_plan.py --workers 16 --reranked artifacts/anchor_keys.jsonl \
    --cache artifacts/planned_anchor.jsonl

# 5. Đóng gói
SUBMIT_NAME=anchor_regen PYTHONPATH=src python scripts/run_submit.py 10 4 artifacts/gen_anchor.jsonl
```

Thứ tự bắt buộc: (1) trước tất cả; (1b) trước (3) và (4); (5) sau cùng.

**Cả `run_plan` và `run_locate` ghi APPEND vào cache và bỏ qua câu đã có.** Muốn sinh lại thì phải
trỏ sang file cache mới (như trên), đừng ghi đè file cũ — nếu bản mới tệ hơn thì không quay lại
được, và repo không phải git repo.

### Dựng máy thuê (vast.ai)

Máy: 1×RTX 4090 24 GB, **Max CUDA ≥ 13** (template `vLLM` là CUDA 13), 50 GB disk là đủ
(~22 GB dùng thật). Ưu tiên reliability và **băng thông mạng** — tải 9 GB weight là chi phí
dựng lớn nhất, 903 Mbps mất 80s còn 226 Mbps mất 320s.

```bash
# Trên máy thuê, sau khi SSH vào:
# Bẫy #11 — bỏ reasoning parser, nếu không mọi output về rỗng:
sed -i "s/ --reasoning-parser qwen3//g" /etc/environment
sed -i "s/ --tool-call-parser qwen3_coder//g; s/ --enable-auto-tool-choice//g" /etc/environment

# Bẫy #12 — supervisor báo đã dừng nhưng EngineCore còn giữ ~21 GB VRAM:
supervisorctl stop vllm; sleep 5
P=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | head -1)
[ -n "$P" ] && kill -9 $P; sleep 4
nvidia-smi --query-gpu=memory.used --format=csv,noheader   # phải thấy ~1 MiB

# Chạy vLLM trực tiếp, KHÔNG qua supervisor (Bẫy #13):
source /venv/main/bin/activate
setsid nohup vllm serve Qwen/Qwen2.5-Coder-14B-Instruct-AWQ --host 127.0.0.1 --port 18000 --max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 16 --download-dir /workspace/models > /workspace/vllm.log 2>&1 < /dev/null &
uv pip install -q pandas pyarrow
```

Upload (~33 MB): `src scripts artifacts/tables.parquet artifacts/reranked_metric.jsonl
artifacts/anchor_keys.jsonl data/questions/questions.jsonl data/code_stock.csv
vifinqa-official/prompts`.

**Lần thuê 07/08 trở đi: nhớ `artifacts/anchor_keys.jsonl` và truyền `--reranked`.** Không truyền
thì cả hai script im lặng rơi về `reranked_metric.jsonl` cũ và cả lượt thuê thành vô ích.

### Quy trình đã chạy thật 07/08 (template vast.ai `vllm/vllm-openai:v0.25.1`)

```bash
# --- trên máy thuê ---
sed -i "s/ --reasoning-parser qwen3//g; s/ --tool-call-parser qwen3_coder//g; \
        s/ --enable-auto-tool-choice//g" /etc/environment          # Bẫy #11
supervisorctl stop vllm; sleep 5
for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q nvidia && kill -9 $(basename $p); done
sleep 8; nvidia-smi --query-gpu=memory.used --format=csv,noheader   # PHẢI thấy 0 MiB — Bẫy #19
supervisorctl start vllm                                            # ~210s khi weight đã cache

# --- ở local, tunnel thay vì upload cả repo ---
ssh -N -L 18000:localhost:18000 -p <PORT> root@<HOST>
```

Tunnel gọn hơn hẳn cách upload 33 MB rồi tải cache về: chỉ prompt đi qua mạng, và **tránh sạch
chênh lệch pandas** giữa remote (3.0.5) và local (2.3.3) vì `run_submit` vẫn chạy ở local.

```bash
PYTHONPATH=src python scripts/run_plan.py --workers 8 --max-tokens 1600 \
    --model Qwen/Qwen3.5-9B --local-url http://localhost:18000/v1 \
    --reranked artifacts/anchor_keys.jsonl --cache artifacts/planned_anchor.jsonl
PYTHONPATH=src python scripts/run_locate.py --workers 8 --max-tokens 1200 \
    --model Qwen/Qwen3.5-9B --local-url http://localhost:18000/v1 \
    --reranked artifacts/anchor_keys.jsonl --cache artifacts/located_anchor.jsonl
```

`--workers 8` khớp `--max-num-seqs 8` của template; đặt cao hơn chỉ xếp hàng chứ không nhanh thêm.

Chạy với `--local-url http://localhost:18000/v1`. **Tải cache về và chạy lại `run_submit` ở
local** — remote có pandas 3.0.5, local 2.3.3, và chỉ code chạy được ở local mới nên nộp.

### ⚠️ Repo KHÔNG phải git repo

Không có lịch sử, không revert được. Nếu tiếp tục dự án thì `git init` ngay và commit
`submissions/bestlabel.zip` cùng bảng cờ ở trên — bản tốt nhất hiện chỉ tồn tại dưới dạng một
file zip và một tổ hợp cờ trong file nguồn.

## Các nhánh trả lời, theo thứ tự ưu tiên trong `run_submit.py`

| # | Nhánh | Câu | Độ chính xác đo được | Loại |
|---|---|---|---|---|
| 1 | Tra cứu regex + xếp theo điểm nhãn | 272 | **~55%** | lexical |
| 2 | Định vị 1 ô (union model ∪ embed) | 163 | 13,3% (phần dư) | model |
| 3 | LLM sinh pandas | 229 | 19,5% | model |
| 4 | Kế hoạch đa ô (k ô + 1 phép toán) | 158 | 7,2% | model |
| 5 | Fallback (khớp nhãn không ngưỡng) | 212 | 5,9% | lexical |
| 6 | Quét cột | 9 | ~0% | — |

**Kết luận quan trọng nhất của cả dự án: mọi cơ chế lexical đều thắng mọi cơ chế model, không
ngoại lệ.** Được đo lại nhiều lần:
- Định vị bằng model trên pool regex: **~21%** so với **42,8%** của khớp nhãn F1 (mất 13 câu).
- Thay embed cho model ở cùng vị trí: mất 9 câu.
- Qwen2.5-Coder-14B tự host so với Qwen3-8B: 16 so với 17 câu dùng được.
- Thắng lợi lớn nhất từ đầu (+2 câu, đưa lên 0.2036) là **một phép sửa logic chọn ưu tiên
  deterministic**, không có model nào tham gia.

Hệ quả: đầu tư vào đường deterministic. Đã thuê GPU hai lần và dựng ba cơ chế model, tổng cộng
chúng ăn ít hơn một lần sửa `Corroborator.choose`.

**Bổ sung 07/08 — có một loại đầu tư thứ ba, và nó lớn hơn cả hai.** Kết luận trên nói về *cơ
chế trả lời*. Nhưng cả cơ chế lexical lẫn model đều đọc cùng một danh sách bảng, nên **sửa danh
sách đó nâng tất cả cùng lúc**: đổi chỉ mục sang văn xuôi ăn +0.090 TABLES_F2 *và* +0.012 EXEC
*và* +0.012 ANSWER trong một lượt. Trước khi tối ưu một nhánh, hỏi xem đầu vào chung của mọi
nhánh có sai không.

## Vì sao mỗi cơ chế mới chỉ ăn được vài câu

Độ chính xác giảm đơn điệu theo thứ tự thêm vào: 42,8% → 19,5% → 13,3% → 7,2% → 5,9%.
**Mỗi cơ chế mới chỉ chạy trên phần dư của cơ chế trước**, và phần dư khó hơn *theo cấu trúc* —
nó khó chính vì mọi cơ chế dễ hơn đã lấy phần của mình. Đây không phải chuyện làm dở ở từng
bước, mà là hệ quả toán học của cách tấn công.

Nên **thêm cơ chế thứ bảy là đường cụt**: phần dư còn ~106 câu công khai ở 5,9%, nâng lên 20%
cũng chỉ +15 câu. Chỗ có đủ độ lớn nằm ở **pool đã phủ**: nâng regex 136 câu từ 42,8% lên 60%
là +23 câu; nâng LLM 114 câu từ 19,5% lên 30% là +12 câu. Hai pool đó chứa 170 câu sai, nhiều
hơn cả phần dư.

## Cờ trong `run_submit.py`

| Cờ | Giá trị | Trạng thái |
|---|---|---|
| `LOCATE_WINS` | `False` | đã đo: bật = mất 13 câu |
| `LOCATE_SOURCE` | `"union"` | model trước, embed lấp 77 câu model thiếu |
| `USE_TIEBREAK` | `True` | đã đo 08/08: tắt = −6 EXEC — giữ True |
| `USE_PLAN` | `True` | +1 câu |
| `USE_PANEL` | `False` | đã đo: bật = trung tính, −1 exec |
| `USE_RATIO` | `False` | đã đo: bật = −1 câu, +7 IndexError |
| `USE_RATIO_LOOKUP` | `True` | +6 câu; nằm trong bản 0.2273. Khác `USE_RATIO`: chỉ *đọc* ô % có sẵn |
| `USE_COMPOSE` | `True` | 58 câu (2 trục); **đã đo: 0.2016 → 0.2273 → 0.2352** |
| `USE_SCREEN` | `True` | 5 câu, và loại 10 câu compose gom sai |
| `USE_SCREEN_RATIO` | `True` | 10/08; screen filter = tỷ số sạch + đủ mọi ticker. **+1 (390)** |
| `USE_RATIO_DIVIDE` | `True` | 24 câu. Chia hai ô, khác `USE_RATIO` |
| `DECLARE_SOURCE` | `"anchor"` | nguồn xếp hạng cho `relevant_tables` |
| `SEARCH_SOURCE` | `"anchor"` | xếp hạng cho đường trả lời |
| `ADAPTIVE_DECLARE_K` | `True` (kỳ vọng spank) | k≈`clamp(2·span,3,11)`; evidence chỉ là sàn — xem PHAN-TICH spank |
| `DECLARE_K_SCALE` | `2.0` (env) | ⚠️ đỉnh **dịch theo bộ evidence** — xem cảnh báo dưới bảng |
| `DECLARE_K_CAP` | `11` (env) | trần k cho câu span lớn |

Biến môi trường (không phải cờ trong file): `SUBMIT_NAME` đặt tên zip đầu ra, `LOCATED_FILE` và
`PLANNED_FILE` trỏ sang cache khác `located.jsonl` / `planned.jsonl`.

> ⚠️ **`DECLARE_K_SCALE`: đỉnh là hàm của bộ evidence, không phải hằng số.** Quét trên đáp án
> `anchor_search` cho đỉnh ở k≈5,94 (scale 2.0). Sau khi sinh lại bộ đệm LLM, k tự trôi lên 6,46 và
> đỉnh **dịch lên trên 6,46** — hạ về 5,85 làm F2 rơi tiếp 0,5526 → 0,5496 và mất một lượt nộp.
> Chương trình mới đọc nhiều bảng hơn, mà evidence có xác suất là gold cao hơn phần đệm BM25, nên
> cắt k là cắt vào phần đang có lãi. **Mỗi lần đáp án đổi thì phải quét lại.**

`USE_TIEBREAK` dùng embed làm trọng tài giữa regex và model. **Đã đo (08/08): tắt mất 6 EXEC** —
giữ `True`. Suy luận “embed yếu trung bình nên tắt” sai trên tập mâu thuẫn có điều kiện
(xem PHAN-TICH).

### 10/08 — screen-on-ratio

- `compose.resolve_screen_ratio` + `USE_SCREEN_RATIO`: filter tỷ số sạch, đủ operand, không nest.
- `compose.resolve_screen_ratio_threshold`: cohort thanh toán hiện hành ≷ τ rồi rank — **2755 hòa**,
  không lấy làm nền.
- Alias `nam kim`/`hoa sen` trong `BRAND_ALIASES`.
- Zip surgical từ helpers: `submissions/screen_ratio_gated.zip`, `threshold_cohort.zip`.
- Audit: `scripts/_exec_residual_audit.py`, `scripts/_probe_ripe_ratio_screen.py`.

## Vòng cải thiện 04/08 chiều — `metricfix.zip` (đã nằm trong bản 0.2273)

Xuất phát từ một câu hỏi: *thành phần nào ảnh hưởng tới output nhiều nhất?* Trả lời:
`extract_metric` — mọi nhánh khớp nhãn đều gọi nó. Nó có ba khuyết điểm, đo được mà không cần
đáp án:

| Khuyết điểm | Quy mô | Nguyên nhân |
|---|---|---|
| Cắt mệnh đề đầu dừng ở dấu phẩy **đầu tiên** | 84 câu chỉ tiêu bắt đầu bằng chữ số | "Trong các năm 2017, 2019 và 2022, …" còn lại "2019 và 2022, …" |
| Dạng công-ty-trước không có "của" để cắt | trong 236 câu còn chứa năm | "CRE có tổng doanh thu … năm 2019" trả về **nguyên câu hỏi** |
| Cắt ở "của" đầu tiên bất kể ngữ nghĩa | 152 câu (15%) | "Giá trị còn lại **của quyền sử dụng đất**" → còn "Giá trị còn lại", khớp bừa mọi dòng trong thuyết minh TSCĐ |

Và một khuyết điểm thứ tư, ở `match_row`: nó đọc cứng `row[0]`, nhưng **9,4% bảng (1.794/19.144)
có cột nhãn ở chỉ số 1** — gồm chính bảng cân đối kế toán có cột "Mã số" (cột 0 = 100/110/111,
cột 1 = "A. TÀI SẢN NGẮN HẠN"). Những bảng đó trước đây mất *toàn bộ* khả năng khớp nhãn.

Kết quả đo (cùng `reranked_metric.jsonl` cũ, không cần GPU):

| Bước | Câu nhánh đơn trả lời | Nhãn khớp tuyệt đối (1,00) |
|---|---|---|
| nền (bản 2410) | 273 | 122 |
| + `extract_metric` viết lại | 294 | 137 |
| + chỉ cắt "của" trước pháp nhân | 287 | 139 |
| + `metric_variants` (thử cả hai cách đọc) | 291 | 139 |
| + `label_column` | **303** | **140** |

`label_column` chọn cột nhãn theo tỷ lệ ô "có chữ" (≥4 chữ cái), giới hạn 3 cột đầu, **cột 0
thắng khi hoà** để giữ hành vi cũ làm mặc định. `Lookup` mang thêm `label_col` và mọi bộ sinh
code dùng `df.iloc[:, {label_col}]` — 43 câu trong bản nộp mới đọc cột khác 0.

`metric_variants` là chỗ đáng chú ý về phương pháp: hai cách đọc "Giá trị còn lại của quyền sử
dụng đất" (cả cụm là dòng / "giá trị còn lại" là cột) đều đúng ở một số bảng, nên **không chọn
bừa một cái** — thử cả hai, lấy điểm nhãn cao hơn. Chọn cứng một phía cho 287 hoặc 294; thử cả
hai cho 291 với nhãn chính xác hơn.

Nhánh mới `USE_RATIO_LOOKUP`: `value_columns` loại mọi cột có độ lớn trung vị < 1000, tức loại
**đúng cột phần trăm** mà câu hỏi "bao nhiêu %" cần → những câu đó trả lời từ cột đồng và ra
"1.028.364.192.393 phần trăm". `ratio_columns` nhận lại chúng, **bắt buộc header phải có**
`%|tỷ lệ|sở hữu|biểu quyết|lợi ích` vì hai ca nhiễu duy nhất ("Thuyết minh" chứa 28/29, "Mã số"
chứa 1–5) đều không có từ nào trong đó. Chỉ +6 câu — trần của lớp này chỉ khoảng 11 câu, phần
còn lại trong 297 câu đơn vị tỷ lệ cần *tính* thật.

**Khác biệt với `USE_RATIO` đã thất bại:** `USE_RATIO` *đoán* tỷ lệ bằng tỷ trọng trên tổng cột;
nhánh này chỉ *đọc* con số đã có trong bảng. Bài học đã ghi ở dưới vẫn đúng — biến bất khả thi
thành khả thi không được gì; phải biến nó thành **đúng**.

Rủi ro của lượt nộp: **121/1012 đáp án đổi (12%)**, tức ~60 câu được chấm. Bằng chứng nghiêng về
phía tốt (nhãn khớp tuyệt đối +18 câu, tức +15%) nhưng đây vẫn là một cú đánh cược.

Còn dư địa đã đo nhưng chưa làm: 157 đáp án vẫn có **độ lớn bất khả thi** (100 câu % > 1e4,
39 câu > 1e16 đồng, 18 câu "lần" > 1e4) — khoảng 78 câu được chấm, tất cả chắc chắn sai.

## Cơ chế hợp thành cho câu dẫn xuất — `compose.zip` (ĐÃ NỘP: 0.2273 / 0.2292)

`src/vifin/answering/compose.py`, cờ `USE_COMPOSE`, chạy **trước** nhánh đơn. Không cần GPU,
thuần lexical. Tái lập: `PYTHONPATH=src python scripts/run_submit.py 10 5 artifacts/generated_full.jsonl`.

Ý tưởng: câu dẫn xuất là **hợp thành**. Thay vì hỏi model, gọi lại chính bộ khớp nhãn — thành
phần mạnh nhất trong pipeline — một lần cho mỗi năm, rồi áp một phép trong tập đóng
`{diff, growth, max, min, avg, sum}` qua `plan_cells.compile_plan` (bộ sinh đã chạy được, an
toàn py37, đã qua cổng `reads_no_frame`).

Hai quyết định thiết kế mang phần lớn độ chính xác:

**Khoá nhãn.** Không để mỗi năm tự khớp độc lập. Năm có điểm nhãn cao nhất làm *mốc*, mọi năm
khác phải trả về **đúng nhãn đó** (so sau khi chuẩn hoá) hoặc bị bỏ. Khớp độc lập cho cùng nhãn
ở 70% ca và khác nhau ở 30% — khoá nhãn biến 30% đó thành "cùng nhãn hoặc không trả lời", thay vì
âm thầm trộn hai khoản mục khác nhau vào một phép trừ.

**Kiểm độ lớn chéo năm.** Sai đơn vị hiện ra thành một năm lệch vài bậc so với các năm còn lại
(GAS: 4,17e17 cạnh 4,55e11). Cơ chế đơn năm không có gì để so; ở đây **trung vị các năm khác là
bộ phát hiện miễn phí**, và năm lệch bị loại thay vì bị tính vào. Nếu số năm lệch nhiều hơn
`n − 2` thì bỏ cả câu, vì lúc đó chính trung vị đã không đáng tin.

Ngoài ra: mỗi năm **bắt buộc lấy từ báo cáo của đúng năm đó** (`_year_candidates` chặn việc
`candidate_docs` tự nới sang năm liền kề). Nới năm là đúng cho tra cứu đơn đọc cột so sánh,
nhưng ở đây nó sẽ dựng phép tính từ sai kỳ.

### Đo lường

Phạm vi: 1 mã, ≥2 năm, nhận diện được phép tính, đơn vị tiền (hoặc tăng trưởng) → **150 câu**.

| Bước | Trả lời được | Nhãn khớp tuyệt đối |
|---|---|---|
| bản đầu | 39/150 | 18 |
| + làm giàu `metric_variants` | **50/150** | **29** |

Phân theo phép tính: max 24 · diff 10 · growth 9 · avg 3 · sum 3 · min 1.

Chẩn đoán 111 ca thất bại ban đầu chỉ ra **99 ca là "có bảng nhưng không khớp được nhãn nào"** —
không phải thiếu tài liệu, không phải do khoá nhãn. Nguyên nhân là các bộ cắt còn thiếu, và cách
sửa **an toàn hơn hẳn** việc đổi `extract_metric`:

> **Làm giàu `metric_variants` thay vì sửa `extract_metric`.** `match_row` thử mọi biến thể rồi
> lấy điểm cao nhất, `MIN_LABEL_SCORE` vẫn giữ nguyên — nên thêm một cách đọc chỉ có thể tìm ra
> dòng khớp *tốt hơn*. Sửa `extract_metric` thì mỗi lần đánh đổi lớp câu này lấy lớp câu khác
> (đã thấy: 294 → 287 khi thêm luật "của").

Biến thể đã thêm: cắt đuôi so sánh nhất ("doanh thu thuần **thấp nhất**") · cắt mệnh đề sở hữu
dạng "tại CTCP/Tập đoàn" (loại trừ "Ngân hàng", vì "Tiền gửi tại Ngân hàng Nhà nước" là tên tài
khoản) · cắt "tổng" đứng đầu (trừ "tổng cộng"). Và một lỗi thật trong `LEADING_OP_RE`: nó thiếu
`tăng trưởng` / `tốc độ tăng` — cùng lớp với `chênh lệch`, rõ ràng là từ chỉ phép tính. Riêng nó
đưa nhánh growth từ 1 lên 9 câu.

Hiệu ứng phụ có lợi: nhánh đơn cũng lên **303 → 312 câu, nhãn khớp tuyệt đối 140 → 154**.

### Compose lấy 37 câu khỏi nhánh đơn — và đó là đúng

Nhánh đơn mạnh nhất (~55%), nên việc chiếm chỗ phải kiểm. Kiểm bằng số học, không bằng niềm tin:

| id | câu hỏi | ô đơn | hợp thành |
|---|---|---|---|
| 646 | "chi phí nhân viên năm 2021 chênh lệch mấy tỷ so với 2020" | 28,60 | **10,34** = 22,29 − 11,95 |
| 627 | "chênh lệch … cuối năm 2018 so với cuối 2016" | 13.838 | **27.838** = 41.677 − 13.838 |
| 514 | "min qua 5 năm" | 62,13 | **16,02** (đúng giá trị nhỏ nhất) |

Ô đơn trả về **giá trị của một năm**, không phải phép tính được hỏi. Con số 55% của nhánh đơn đo
trên pool của nó, chủ yếu là tra cứu đơn thật — nó không áp dụng cho những câu này.

### Rủi ro

`compose.zip` đổi **157/1012 đáp án so với `bestlabel.zip`** (48 câu so với `metricfix.zip`).
0 lỗi py37, 0 câu gán hằng số. Độ lớn bất khả thi 161 → 154.

## Câu dẫn xuất: tiền đề ban đầu (đã dẫn tới cơ chế trên)

`scripts/probe_derived.py`. **550/1012 câu là dẫn xuất** (khớp `DERIVED_RE`), và cả ba cơ chế
đang phục vụ chúng đều là model: LLM 19,5%, cell-plan 7,2%, fallback 5,9%. Nhưng câu dẫn xuất là
**hợp thành**: "chênh lệch X giữa năm A và B" = hai lần khớp nhãn + một phép trừ, mà khớp nhãn
là thành phần mạnh nhất trong pipeline.

Probe lấy 126 câu đa năm / một mã / đơn vị tiền, truy hồi **riêng từng năm** bằng truy vấn
chỉ-chỉ-tiêu, rồi `find` trên tài liệu của năm đó:

| Cấu hình | Giải đủ mọi năm |
|---|---|
| `search_balanced`, truy vấn toàn câu | 14/126 |
| truy hồi từng năm, truy vấn chỉ tiêu | 22/126 |
| + `extract_metric` đã sửa | **37/126** |

70% số câu giải được có **cùng một nhãn ở mọi năm** — bằng chứng thật, không vòng vo, vì không
có gì trong quy trình chọn tài liệu dựa trên giá trị. Giá trị đọc ra rất khả tín (EPS BVH
2017–2024: 2.286 / 1.689 / 2.089 / 2.843).

Hai điều probe phát hiện thêm:
- `choose()` bỏ mọi câu đa năm sang `_first()` (lấy ứng viên đầu, **không xếp điểm**) vì
  `len(years) != 1`; và dòng gọi `choose()` **không có cổng `is_single_lookup`** trong khi nhánh
  locate thì có. Nên 41 câu dẫn xuất đang được trả bằng **một ô đơn**, dù cả 41 đều đã có sẵn
  plan lẫn code LLM bị chặn phía sau.
- Sai hệ số đơn vị lộ ra khi so các năm: GAS 2024 ra 4,17e17 cạnh 2020 ra 4,55e11. **Trung vị
  qua các năm là bộ phát hiện sai scale miễn phí** — cơ chế đơn năm không có được.

`DERIVED_RE` cũng bắt sai: "chênh lệch tỷ giá", "tỷ lệ quyền biểu quyết", "thu nhập bình quân"
là **tên chỉ tiêu**, không phải phép tính. `probe_derived.classify()` có `NAME_IDIOMS` xử lý
việc này; `is_single_lookup` thì chưa.

## Trục thứ hai cho compose — `axis.zip` (ĐÃ NỘP: 0.2352 / 0.2391, +4 câu)

Compose ban đầu gom theo **năm**. Tổng quát hoá: gom theo bất cứ trục nào. "X cao nhất qua các
năm 2020-2024" và "X cao nhất trong số HPG, HSG và NKG" là **cùng một phép tính trên bộ toán hạng
khác** — nên `resolve()` giờ chạy trên danh sách `(tên, câu hỏi đã thu hẹp)`, và `eligible()` trả
`(phép tính, trục)`.

Chọn trục này bằng số, không bằng cảm giác — đo pool trước:

| Pool | Câu | Ghi chú |
|---|---|---|
| compose đã phủ (1 mã, ≥2 năm) | 150 | |
| **nhiều công ty + gom + đơn vị tiền** | **55** | 42 câu gom thuần, 13 câu có điều kiện |
| nhiều công ty, phép/đơn vị khác | 210 | phần lớn là sàng lọc hai tầng |
| dạng "X trên Y" | 59 | |
| 1 mã, <2 năm, "có phép tính" | 143 | **pool giả** — xem dưới |

Pool 143 câu là **giả**: 139 câu bị gán `sum` chỉ vì có chữ "Tổng", mà "Tổng tài sản", "Tổng nợ
phải trả", "Tổng quỹ lương" là **tên chỉ tiêu**. Mẫu `tổng` quá lỏng, nhưng vô hại vì
`eligible()` đòi ≥2 toán hạng. Nếu sau này nới điều kiện đó thì phải sửa mẫu trước.

### Kết quả

| Trục | Trong phạm vi | Trả lời được | Nhãn khớp tuyệt đối |
|---|---|---|---|
| năm | 129 | 51 | 29 |
| **công ty** | 97 | **17** | 7 |

Tổng compose **50 → 68 câu**.

### Ba lỗi thật tìm được bằng cách đọc output

**1. Dấu ngoặc kế toán làm sai phép cộng.** id=862 "Tổng chi phí tài chính" của 3 công ty ra
**5,40** thay vì 75,42: báo cáo Việt Nam in chi phí là `(6.154.448.991)`, tức âm, nên tổng thành
−6,15 + 40,41 − 28,86. Nhánh đơn đã xử lý đúng từ đầu bằng `synthesize(magnitude=True)`; compose
thì không. Đã thêm `magnitude` vào `compile_plan` → mọi toán hạng đọc qua `abs()`.

Đáng ghi: **chính báo cáo chẩn đoán của tôi che dấu âm**, vì `Composition.values` lưu `abs()`.
Bộ hiển thị làm mất đúng thông tin cần để thấy lỗi.

**2. `compile_plan(values=…)` phát hằng số vào code.** Nó sinh `v0 = -6154448991.0 * 1.0`, đúng
thứ BTC nói sẽ bị loại khi kiểm thủ công ở private. Compose giờ sinh bản **đọc frame**
(`abs(num(df1, 5, 5))`), tự chạy lại và so với giá trị tính bằng Python (`_expected`), chỉ rơi về
hằng số khi `num()` không phân tích được ô. **Kiểm bản nộp: 0 chương trình chứa hằng số** — 217
câu dùng `num()`, 699 câu dùng `synthesize`, 96 câu do LLM.

**3. Tiebreak ghi đè compose.** 3 câu hợp thành bị nhánh `locate` (model, đo 13–21%) thay bằng
một ô đơn, vì `is_single_lookup` trả True cho chúng ("trừ đi", "Tổng" không có trong `DERIVED_RE`)
và tiebreak chạy khi `query is not PLACEHOLDER`. id=934 mất một tổng nhiều dòng để lấy một ô.
Đã thêm biến `source`: tiebreak **chỉ** phân xử khi đáp án đến từ nhánh regex — đúng nhánh nó
được đo trên. `tiebreaks: model won` 20 → 8.

### Hai quyết định KHÔNG sửa, có lý do

**SCREEN_RE chỉ áp cho trục công ty.** Câu sàng lọc ("doanh nghiệp **có** hệ số thanh toán nhanh
thấp nhất thì hàng tồn kho bằng bao nhiêu") cần hai tầng, và gom danh sách mã ở đó là sai. Nhưng
áp cùng bộ lọc cho trục năm làm **mất 10/50 câu** của một trục đã có kết quả leaderboard thật.
Không được đánh đổi một thắng lợi đã đo bằng một suy đoán chưa đo.

**3 câu 2-toán-hạng lệch >100x: để nguyên.** Bộ kiểm độ lớn cần ≥3 toán hạng. Với 2 toán hạng,
lệch 842x *trông như* sai đơn vị — nhưng thu nhập mua bán chứng khoán của hai ngân hàng có thể
chênh 1.000 lần thật. Bác bỏ là đoán.

## Sai hệ số đơn vị — `scalefix.zip` (ĐÃ NỘP: 0.2391 / 0.2431, +2 câu)

Xuất phát từ 155 đáp án có **độ lớn bất khả thi**. Phân loại chúng đã **bác bỏ kế hoạch tôi tự
ghi ở bản trước** (bảng công thức tài chính + bộ phân tích "X trên Y"):

| Lớp | Câu |
|---|---|
| **sàng lọc hai tầng** | **91** |
| sai hệ số đơn vị tiền (>1e16 đồng) | 35 |
| khác | 15 |
| dạng "X trên Y" | 15 |
| công thức có tên (ROA, biên LN…) | 3 |
| tăng trưởng | 2 |

Bảng công thức sẽ chạm được **18 câu**, còn 91 câu là chuyện khác hẳn. Đo trước khi code lại tiết
kiệm một lần viết sai hướng.

### Lỗi: đơn vị đọc từ văn bản quanh bảng, không thuộc bảng đó

id=117 (FPT, "Giá vốn của dịch vụ đã cung cấp", hỏi nghìn tỷ) nộp **34.732.057**. Truy ra:
`column_scale` trả về 1e6 vì thuyết minh ghi "triệu đồng)", nhưng ô gốc là
`34.732.056.920.000` — **14 chữ số, đã là đồng đầy đủ**. Nhân thêm một triệu thành 3,5e19 đồng,
khoảng một nghìn lần GDP Việt Nam. Đáp án đúng ≈ 34,73.

**Số chữ số là bằng chứng quyết định, không phải suy đoán khả thi.** Không báo cáo nào in số 14
chữ số theo đơn vị triệu đồng. `_checked_scale` bác bỏ hệ số mà chính các con số trong cột phủ
định: nếu trung vị cột × hệ số vượt `DONG_CEILING = 1e16` (tổng tài sản VCB ≈ 2e15) thì cột đã ở
đơn vị đồng.

Đây là chỗ phân biệt với thí nghiệm `USE_RATIO` đã thất bại: ở đó tôi *thay* giá trị bất khả thi
bằng một giá trị khả thi khác. Ở đây tôi chọn giữa **hai hệ số ứng viên cụ thể** bằng bằng chứng
về đơn vị mà bảng thực sự dùng.

Nhánh đơn: **312 câu / 154 nhãn tuyệt đối — y nguyên, không hồi quy — và 0 câu bất khả thi.**

### Ba nhánh không sửa được bằng cách đó

`located.jsonl` và `planned.jsonl` là output GPU đã cache, code LLM thì tự viết bộ phân tích —
không nhánh nào gọi `column_scale`, nên phải **loại và cho rơi xuống nhánh sau** (`impossible()`
trong `run_submit.py`). Vẫn khác `USE_RATIO`: không thay bằng giá trị đoán, chỉ để cơ chế khác trả
lời. LLM bị loại 13 câu, locate 152 → 143, và plan/fallback nhận thêm.

**Sai hệ số tiền: 35 → 5 câu** (5 câu còn lại ở nhánh cuối, không còn chỗ rơi). Tổng bất khả thi
155 → 125. Đổi **45 đáp án** so với `axis.zip`.

Còn lại **120 câu đơn vị tỷ lệ > 1e4**, trong đó 91 câu là sàng lọc hai tầng — pool lớn nhất chưa
ai chạm, và là việc tiếp theo.

## Sàng lọc hai tầng và chia tỷ lệ — `divide.zip` (ĐÃ NỘP: 0.2431 / 0.2470, +2 câu)

### Trục thứ ba: sàng lọc (`USE_SCREEN`)

99 câu có hình thái `<A> ... tại năm có <B> lớn nhất` hoặc `... doanh nghiệp có <B> thấp nhất` —
xếp hạng tập toán hạng theo B rồi đọc A của kẻ thắng. `classify` đọc từ so sánh nhất đó thành
**phép tính**, nên compose trả về `max(A)` qua các năm thay vì A ở năm B lớn nhất. **11 câu sai
kiểu này đã nằm trong bản 0.2391.**

Đọc trọn văn 11 câu mới thấy rõ, và nó buộc tôi **tự sửa một kết luận cũ**: lần trước tôi "kiểm
tay" id=514 và xác nhận 16,02 là giá trị nhỏ nhất đúng — nhưng câu hỏi đòi chỉ tiêu A **ở năm mà
chỉ tiêu B nhỏ nhất**. Tôi đã kiểm đúng một tuyên bố sai.

`resolve_screen` chạy `_pick_operands` nhắm vào B (tham số `metric` mới) để chọn kẻ thắng, rồi tra
cứu đơn A trong đúng kỳ/công ty đó. Chỉ tiêu A được trích từ câu **đã cắt bỏ mệnh đề sàng lọc**.

Yêu cầu **mọi toán hạng phải giải được** — argmax trên tập con là sai. Đó là lý do độ phủ thấp:

| | Câu |
|---|---|
| hình thái sàng lọc, B là khoản mục thường, đơn vị tiền | 25 |
| giải đủ toán hạng | 8 |
| **trả lời được** | **5** |

Nhỏ, nhưng `eligible()` giờ **từ chối** hình thái này nên 10 câu gom sai không còn được nộp.
`SCREEN_YEAR_RE`/`SCREEN_TICKER_RE` hẹp hơn `SCREEN_RE` nhiều (phải bắt được cả B lẫn từ so sánh
nhất), nên dùng để từ chối trên trục năm mà **không** mất 10 câu như bộ lọc rộng từng gây ra.

### Trục thứ tư: chia tỷ lệ (`USE_RATIO_DIVIDE`)

Đo lại 155 câu bất khả thi cho thấy **150 câu có dạng chung "X trên Y"** — nhiều hơn mọi công thức
có tên cộng lại. Nên **không** viết bảng công thức làm cơ chế chính; viết bộ giải X/Y tổng quát,
rồi để công thức có tên **viết lại** thành cặp X/Y (`FORMULAS`, 6 dòng: biên LN gộp, biên LN
ròng/ROS, ROA, ROE, hệ số thanh toán hiện hành, hệ số nợ).

`src/vifin/answering/ratio.py`. Tra cứu hai chỉ tiêu trong **cùng một kỳ của cùng một công ty**,
áp hệ số riêng từng ô (nên đơn vị tự triệt tiêu), rồi chia qua `compile_plan` op `ratio`/`ratio_pct`
với `magnitude=True`.

| | Câu |
|---|---|
| trong phạm vi (1 mã, đơn vị tỷ lệ, khớp dạng) | 64 |
| **trả lời được** | **24** |
| kết quả trong 0–200% | 23 |

**Từ chối khi kết quả > 1000%** (`SANITY_LIMIT`): 29.460% không phải sai lệch nhỏ mà là sai phân
giải một trong hai toán hạng. Đây vẫn khác `USE_RATIO` đã thất bại — nó *thay* giá trị bất khả thi
bằng một giá trị khả thi khác; cơ chế này **tính đúng thương số câu hỏi nêu** và **giữ lại không
trả lời** nếu kết quả không phải một tỷ lệ.

Bất khả thi 125 → **116**. Đổi 32 đáp án so với `scalefix.zip`.

### Còn lại trong pool này — và vì sao ĐỪNG ghép tiếp

> 🔄 **KẾT LUẬN NÀY ĐÃ HẾT HIỆU LỰC (07/08 tối).** Điều kiện của nó là *không có bảng công thức*.
> `vifinqa-official/src/vifinqa/generation/intermediate_formulas/registry.py` chính là bảng đó,
> kèm `concept_names` — các biến thể nhãn tiếng Việt cho từng vai trò trong công thức.
>
> Đo lại pool "sàng lọc hoặc so ngưỡng theo tỷ số phải tự tính": **133 câu, đang đúng ~12 (9%)**,
> hầu hết rơi vào fallback (51) và plan (48). Đạt 15% → +8 câu; đạt 25% → +21 câu. Biên độ đang
> giữ so với nguyenvuhoanglong chỉ **0,001 macro**. Chi tiết và danh sách archetype cần hỗ trợ ở
> `PHAN-TICH.md`, mục "07/08 tối".
>
> Cũng phát hiện lỗi định nghĩa trong `src/vifin/answering/ratio.py`: ROA và ROE của ta chia cho số
> **cuối năm**, định nghĩa chính thức chia cho **bình quân đầu năm và cuối năm**; `quick_ratio` =
> (TSNH − hàng tồn kho) / nợ ngắn hạn thì chưa có (15 câu). Dung sai là 0,01 **tuyệt đối** với
> `rel_tol=0`, nên đây là sai hẳn, không phải sai số.
>
> Bài học chung: **một kết luận "đừng làm" luôn kèm điều kiện — hãy ghi điều kiện đó cùng kết luận**,
> nếu không nó sẽ sống lâu hơn lý do tồn tại của nó.

48/99 câu sàng lọc xếp hạng theo **tỷ số**. Tôi từng ghi rằng bước tiếp theo tự nhiên là cho
`ratio.py` chạy trong vòng toán hạng của `resolve_screen`. **Số học bác bỏ:**

- `ratio.shape` viết lại được 29/48 câu.
- `resolve_screen` đòi **mọi** toán hạng giải được (argmax trên tập con là sai).
- `ratio.py` giải được 37% số câu trong phạm vi của nó; `resolve_screen` giải đủ toán hạng ở 32%.

Cần tỷ số giải được đồng thời cho 4–7 công ty. **29 câu → gần như chắc chắn dưới 2 câu được chấm.**
Chi tiết và việc nên làm thay thế ở `PHAN-TICH.md`.

## Bẫy — đọc trước khi sửa

### 0. VÒNG PRIVATE CHẤM KHÁC VÒNG PUBLIC
- Public xếp hạng **chỉ theo `EXECUTION_ACCURACY`**. Private dùng **cả truy hồi + answer +
  execution**. Đừng kết luận `TABLES_F2` vô giá trị vì hai đội đầu bảng công khai để nó bằng 0.
- **`pandas_query` bị kiểm THỦ CÔNG ở private; gán hằng số bị loại.** `reads_no_frame()` trong
  `validate.py` chặn bằng AST. Bản nộp từng có **523/1012 (52%)** query là hằng số.
- `relevant_tables` rỗng vẫn hợp lệ nhưng mất điểm truy hồi.

### 1. `relevant_tables` là SỐ DÒNG, không phải chỉ số bảng
Số dòng 1-based nơi thẻ `<table>` bắt đầu trong file OCR, viết trần: `doc|350`. Repo BTC code
`f"{doc}|table_{id}"` — **sai so với scorer**. Cả 4 tổ hợp ordinal đều cho `TABLES_F2 = 0.0`.

### 2. Scorer bind CSV theo `evidence.variable`, không theo `df`
`program_system.txt` của BTC mâu thuẫn với đặc tả nộp bài. Khai `df1` mà code dùng `df` →
`NameError` mọi câu. Bảng đơn đặt tên `df` (thoả cả hai cách hiểu), nhiều bảng thì `df1..dfn`.

### 3. Dung sai **0,01 TUYỆT ĐỐI**, `rel_tol` bằng 0

Ghi chú cũ ở đây viết "0,02% tương đối" — **sai, và sai theo hướng nguy hiểm**.
`answer_match.py:57` là `math.isclose(exp, act, rel_tol=0.0, abs_tol=ANSWER_ABS_TOL)` với
`ANSWER_ABS_TOL = 1e-2` (`constants.py:5`), và test của BTC chốt cả hai phía:
`is_correct(100.0, 100.005)` đúng, `is_correct(100.0, 100.02)` **sai**.

Không có dung sai tương đối nghĩa là **số càng lớn càng khắt khe**, chứ không phải càng dễ.
Đáp án theo đồng cỡ 1e12 phải đúng tới hai chữ số thập phân — sai một đồng là mất điểm.
Hệ quả cho cách chọn cơ chế: một chương trình nhiều bước tích luỹ sai số làm tròn sẽ trượt ở
chỗ mà **đọc thẳng một ô rồi quy đổi đơn vị** thì không, vì nó chỉ có một phép nhân. BTC còn
nói rõ trong `direct_system.txt`: "Do not round during intermediate calculations. Round only
the final result to 2 decimal places." Mọi chỗ trong repo ta làm tròn giữa chừng là đang tự
bắn vào chân.

### 4. Scorer chạy Python 3.7 (`codalab/codalab-legacy:py37`)
Comprehension trong `exec` với globals/locals tách rời → `NameError`. Python 3.13 ở local đã
inline comprehension nên **che hoàn toàn lỗi này**. `portability_problems()` chặn trước.

### 5. Nộp đủ 1.012 câu, dù chỉ chấm 506 (tập 506 là ngẫu nhiên)

### 6. Đơn vị nằm trong tiêu đề cột, không phải dòng "Đơn vị tính"

### 7. `strip_tones` bỏ dấu thanh nhưng GIỮ dấu mũ
`"thuần"` → `"thuân"`. Mọi chuỗi dò phải đi qua `_fold`. Lỗi này đã xảy ra **hai lần**.

### 8. Cạm bẫy tiếng Việt
36 câu viết "Hoà" vs roster "Hòa" · 12 cặp tên công ty lồng nhau · **64 câu "trăm tỷ đồng"**
(1e11) · STB = "Sài Gòn Tài Lộc" không phải Sacombank · 2,4% bảng ghi số kiểu Anh.

### 9. Reranker: PHẢI dùng `--query-mode metric`
Rerank theo **cả câu hỏi** tệ hơn không rerank (top-1 33,8% → 17,0%): câu hỏi chứa tên công ty,
năm, đơn vị mà bộ lọc metadata đã dùng xong, làm loãng tín hiệu tên chỉ tiêu. Theo **cụm chỉ
tiêu** thì thắng BM25 ở mọi k (top-1 41,4%, top-5 55,2%).

### 10. Rerank giúp nhánh tra cứu, HẠI nhánh LLM
Đưa thứ tự rerank vào `run_generate` kéo tỷ lệ chạy được 49,9% → 26,4%: rerank là xếp hạng độ
liên quan phẳng, không đảm bảo có mặt đủ các công ty trong câu so sánh. Nhánh LLM giữ
`search_balanced`. **Rerank giúp chỗ vấn đề là *độ liên quan*, hại chỗ vấn đề là *độ phủ*.**

### 11. Máy thuê: `--reasoning-parser qwen3` bật mặc định trong `VLLM_ARGS`
Nó dồn toàn bộ output vào `message.reasoning` và để `content: null`. Qwen2.5-Coder không phải
model thinking nên **mọi chương trình trả về rỗng** — rất dễ kết luận oan là model không sinh
được code. `ChatClient` đã có fallback đọc `reasoning`, nhưng vẫn nên bỏ tham số.

### 12. Máy thuê: `EngineCore` mồ côi giữ VRAM
`supervisorctl restart vllm` báo thành công nhưng process cũ vẫn giữ ~21/23 GB, engine mới chết
với `ValueError: Free memory ... 1.98 GiB`. Phải `kill -9` pid trong `nvidia-smi` trước khi start.

### 13. Máy thuê: supervisor không đọc lại `/etc/environment`
Ghi `VLLM_MODEL` vào `/etc/environment` **sau** khi supervisord đã khởi động thì service không
thấy — nó `EXITED` với log gần như rỗng. Chạy `vllm serve` trực tiếp bằng `setsid nohup`.

### 14. `ssh ... 'cmd &'` treo kênh
Chạy tiến trình nền qua ssh mà không redirect đủ fd sẽ làm ssh treo chờ kênh đóng. Dùng
`setsid nohup ... > log 2>&1 < /dev/null &`, và kiểm tiến độ bằng một kết nối ssh riêng.

### 15. Scorer đọc CSV bằng `read_csv` — nó TỰ SUY KIỂU từng cột
Bẫy đắt nhất về mặt "im lặng": `sandbox.frame_from_rows` từng dựng frame với `dtype=object`, tức
mọi ô là chuỗi. Scorer thì không. Bằng chứng: leaderboard báo `2 AttributeError, 1 IndexError`
trên bản `compose`, và lỗi tái lập được **chính xác** khi dựng frame bằng `pd.read_csv`:

```
AttributeError: 'numpy.float64' object has no attribute 'replace'
```

Chương trình gọi `cell.replace(".", "")` chạy đúng trên chuỗi và nổ trên cột được suy thành số.
Nghĩa là **bộ kiểm cục bộ sai ở đúng cái nó tồn tại để dự đoán** — nó phê duyệt code sẽ crash
khi chấm.

Đo trên 1012 chương trình: 1006 câu giống nhau, **5 câu chỉ crash khi suy kiểu**, 1 câu lệch giá
trị (0,01 cục bộ so với 0,17 khi chấm). Đã sửa: `frame_from_rows` giờ round-trip qua đúng CSV sẽ
nộp rồi `read_csv`. Việc round-trip cũng tái hiện cách `read_csv` đổi tên cột trùng — bảng OCR
đầy cột trùng tên.

**Sau khi sửa: 1012/1012 chương trình chạy được theo cách đọc của scorer và tái lập đúng đáp án
đã nộp.** Đây là bất biến nên kiểm lại trước mỗi lần nộp.

Ghi chú cũ trong `package._render_csv` nói "scorer đọc bằng `csv.reader`" là **sai** — đã sửa.

### 16. Import `vifinqa` kéo cả CLI, và corpus công khai khác corpus nội bộ

`vifinqa/__init__.py` import `cli` → `legacy_cli` → `bm25s`, nên `import vifinqa.common.corpus...`
chết ngay dù module đích chỉ dùng stdlib. Nạp module lẻ bằng `importlib.util.spec_from_file_location`,
và **phải `sys.modules[name] = m` TRƯỚC `spec.loader.exec_module(m)`** — nếu không `@dataclass(slots=True)`
nổ với `AttributeError: 'NoneType' object has no attribute '__dict__'`.

Quan trọng hơn: `parse_document` của họ tìm neo `[table_N](path)` và **corpus công khai không có
neo đó** — bảng nhúng thẳng bằng `<table>...</table>`. Chạy nguyên si trả về 0 neo và mọi ngữ cảnh
rỗng, *im lặng*. `table_id` = **thứ tự xuất hiện của `<table>`** trong tài liệu; đã kiểm khớp 100%
với `page_no` đã lưu trên ba tài liệu mẫu trước khi dựng 111.728 bảng.

### 17. F2 phạt precision khi P ≪ R — xem mục công thức F2 ở đầu file

Đã mất hai lượt nộp (`decl_k12`, `decl_k20`) vì tin "β=2 nên khai nhiều bảng hơn thì tốt".

### 18. Model suy luận đốt sạch `max_tokens` TRƯỚC khi tới đáp án

`run_plan` đặt `max_tokens=200`, `run_locate` đặt `120` — đủ cho model instruct, **chết hoàn toàn
với model suy luận**. Qwen3.5-9B viết ~3.500 ký tự "Thinking Process:" rồi mới tới JSON:
`usable plans 0/20`. Nâng lên 1600 → **9/20**. Ngân sách, không phải bộ phân tích.

Nó **không phát ra thẻ `<think>`** nên không reasoning parser nào tách được. Đã đo và **cả ba đều
thất bại**: `/no_think` trong prompt, chỉ thị hệ thống "no preamble, start with {", và prefill
assistant bằng `{`. Model vẫn dạo đầu. `plan_json.py` đã regex bóc JSON ra khỏi văn xuôi nên chỉ
cần cho đủ token.

Cả hai script giờ có `--max-tokens`, **mặc định giữ nguyên giá trị cũ** để không đổi hành vi với
model instruct. Dùng model suy luận thì phải truyền tay.

### 19. Diệt `EngineCore` mồ côi: `nvidia-smi` báo PID SAI

Bẫy #12 nói `kill -9 $(nvidia-smi --query-compute-apps=pid ...)`. **Không chạy được** — PID đó
thuộc namespace host, trong container báo "No such process", và `nvidia-smi` hiển thị
`process_name = [Not Found]` trong khi vẫn giữ 19 GB. `pkill -f "EngineCore"` cũng trượt.

Cách đúng — quét tiến trình nào đang mở thiết bị nvidia:

```bash
for p in /proc/[0-9]*; do
  ls -l $p/fd 2>/dev/null | grep -q nvidia && \
    echo "PID $(basename $p): $(tr '\0' ' ' < $p/cmdline | cut -c1-110)"
done
# hoặc: fuser -v /dev/nvidia*
kill -9 <PID thật>   # rồi kiểm: nvidia-smi --query-gpu=memory.used --format=csv,noheader -> 0 MiB
```

Không dọn thì engine mới chết với `ValueError: Free memory on device cuda:0 (1.98/23.52 GiB) ...
is less than desired GPU memory utilization (0.92, ...)`.

### 20. Template vast.ai phục vụ model KHÁC với model ta cần

Template `vLLM` mặc định serve `Qwen/Qwen3.5-9B` (hợp lệ: open-weight, 9B, phát hành Feb/Mar 2026,
trong hạn 01/06/2026 — đã thêm vào `ALLOWED_MODELS`). Weight tải sẵn nên **dùng nó rẻ hơn kéo về
9 GB Qwen2.5-Coder-14B-AWQ**. Nhưng phải:

1. `sed -i` gỡ ba cờ khỏi `/etc/environment` (Bẫy #11) — `--reasoning-parser qwen3`,
   `--tool-call-parser qwen3_coder`, `--enable-auto-tool-choice`.
2. `supervisorctl stop vllm`, dọn `EngineCore` mồ côi (Bẫy #19), rồi `supervisorctl start vllm`.
   Supervisor đọc lại `/etc/environment` khi start nên **không cần chạy vllm trực tiếp** như Bẫy
   #13 mô tả cho template cũ.
3. Truyền `--model Qwen/Qwen3.5-9B` và `--max-tokens 1600` (Bẫy #18).

Khởi động lại mất ~210 giây khi weight đã có cache.

## Thử nghiệm đã THẤT BẠI — đừng làm lại

| Thay đổi | Kết quả |
|---|---|
| `LOCATE_WINS=True` (định vị thắng trên pool regex) | **−13 câu** (0.1976 → 0.1719) |
| Thay embed cho model ở vị trí lấp chỗ trống | **−9 câu** (0.1957 → 0.1779) |
| Xếp lại nhóm fallback theo (xác nhận chéo năm, điểm nhãn) | **−8 câu** |
| Sửa 273 đáp án %/lần "bất khả thi" thành "hợp lý" | **0 câu tăng** |
| Gộp panel chỉ tiêu | trung tính |
| Hạ `declare_k` xuống 3 | `TABLES_F2` 0.3817 → 0.3775 |
| Qwen2.5-Coder-14B thay Qwen3-8B | 16 vs 17 câu dùng được |
| Ví dụ `WRONG:` trong prompt | model bắt chước dòng sai, 19/30 → 16/30 |
| Khớp nhãn tự do khi không có cột Mã số | đẳng thức tài sản 97,8% → 73,2% |
| Danh sách trắng từ vựng lọc câu hỏi panel | loại sạch 287/287 ứng viên |
| Khai 12 bảng / 20 bảng (`decl_k12`, `decl_k20`) | `TABLES_F2` 0.4718 → **0.3577 / 0.2833** |
| Chỉ mục ngữ cảnh **cả trang** (`context_index`) | top-4 64.3% so với 65.0% của bm25 — mọi bảng cùng trang nhận y hệt một đoạn văn |
| RRF hợp nhất bm25 + anchor | top-30 tốt hơn (95.3%) nhưng **top-4 không đổi** — vô dụng vì F2 giới hạn bởi precision |
| `DECLARE_K_SCALE` 1.5 / 2.5 / 3.0 | 0.5561 / 0.5577 / 0.5498 so với **0.5618** ở 2.0 |
| Dùng nguyên 987 cặp nhãn khai thác (`label_pairs.jsonl`) | **chưa nộp, đừng làm** — nhiễu hệ thống: "Thắng dư vốn cổ phần" ↔ "Số đầu năm" 13 lần, "Chi phí tài chính" ↔ "TỔNG CỘNG" 8 lần. Chỉ tập viết tắt (`abbreviations.json`) là dùng được |
| `threshold_cohort` / id=397 (hiện hành >1,5 → min nhanh → tồn kho) | **hòa EXEC/ANSWER/F2** vs 2752 — câu đổi không điểm |
| Bỏ BM25 pad declare (`label_t3`) | F2 **0.4718→0.4127** (P↑ R↓ nặng) |
| Synth train từ ô (`probe_synth`) | model bịa chỉ tiêu; bỏ |
| Train/SFT trên gen BTC tự chạy | chưa làm; kết luận: không ưu tiên trước median/registry |

**Hai thành phần dùng chung KHÔNG phải nút thắt** (đo rồi, đừng sửa): `pick_column` đúng ~89%
(273/273 luôn chọn cột giá trị đầu; chỉ 30 câu sai kỳ), `column_scale` đúng ~96% (phân bố độ
lớn tập trung 10⁹–10¹³ VNĐ, chỉ 12/273 bất thường).

## Bài học về phương pháp — đã lặp lại nhiều lần

**Áp một phép đo ra ngoài phạm vi của nó.** Lỗi phổ biến nhất của tôi trong dự án này:
- Dùng 13,3% (đo trên *phần dư*) để bác bỏ `LOCATE_WINS` (áp trên *pool regex*) — phải nộp thử
  mới biết, và hoá ra kết luận đúng nhưng lý do sai.
- Coi 81,5% xác nhận chéo năm là độ chính xác. Số học bác bỏ ngay: trần trên chỉ 69%.
- Coi 232 câu "đa số 3 phương pháp đồng thuận" là dấu hiệu đúng. Ba phương pháp chia sẻ
  `pick_column` và `column_scale` nên chỉ độc lập ở bước chọn dòng; embed đồng thuận nhiều nhất
  nhưng đo ra lại **yếu nhất**.

**Biến quan sát rời rạc thành kết luận rồi ra quyết định.** "Ba lỗi đã chẩn đoán của nhánh tra
cứu" dựa trên ~12 ví dụ nhìn mắt; đo đúng ra 2 + 2 + 23 câu trên 273, và 2 câu "nhãn tên công
ty" thực ra **đúng**.

**Vá bằng `str.replace` mà không kiểm kết quả.** Xảy ra **hai lần**, cả hai đều thất bại im
lặng: bộ lọc không-cache-lỗi-mạng (ghi 40 lỗi HTTP 400 vào cache như kết quả thật) và phép
"gộp union" (file đã bị sửa nên chuỗi đích không còn). **Luôn `grep` lại sau khi vá.**

**Kiểm số học trước khi đốt lượt nộp.** `declare_k=3` đã đo ở bản 2276 — trùng từng chữ số —
mà tôi vẫn nộp lại. Ngược lại, phép chia đơn giản đã bác bỏ tuyên bố "tra cứu đúng 81,5%" mà
không tốn lượt nào.

## Đo lường không cần đáp án

**Đẳng thức kế toán** (`run_metrics.py`): `TS = TSNH + TSDH`, `Tổng nguồn vốn = NPT + VCSH`.
Hiện **100% trên 478/455/493 ca** — panel lấy lưới rộng nhất rồi bỏ nhóm vi phạm.

**Đối chiếu chéo năm** (`check_crossyear.py`): BCTC năm Y+1 phục hồi số liệu năm Y.
⚠️ Chỉ chứng minh việc trích **nhất quán**, không chứng minh dòng đó **liên quan**. Đừng đọc nó
như độ chính xác.

**Đọc mã của ban tổ chức trước khi tự suy diễn.** `vifinqa-official/` nằm sẵn trong repo suốt
nhiều phiên. Nó trả lời trực tiếp "gold là gì" và "vì sao truy hồi kém" — hai câu đã đoán sai
nhiều lần bằng thực nghiệm tốn lượt nộp.

**Một công thức khớp số đo đáng giá hơn một trực giác về công thức.** "β=2 nên ưu tiên recall"
đúng về mặt chữ nhưng dẫn tới hai lượt nộp mất điểm; `F2 = 5h/(4g+k)` khớp cả bảy điểm và nói
ngược lại.

**Thước đo offline thiên vị vẫn dùng được, nếu biết chiều thiên vị.** Bảng chứng cứ thiên vị
chống lại xếp hạng mới (chính xếp hạng cũ đã tìm ra chúng), nên "hoà" là tín hiệu tốt và "thắng"
là tín hiệu rất mạnh. Đó là căn cứ để dám đổi 128 đáp án.

## Việc tiếp theo (cập nhật 10/08 sáng)

Best giữ: **`screen_ratio_gated.zip` (2752)**. EXEC chết vì **~336 câu sai có số**, không vì 17 zero.
Gap Vương EXEC ≈ 13 câu chấm — nhưng ta **dẫn macro ba tiêu chí** trong mọi đội nộp đủ trường
`relevant_tables`; xem `PHAN-TICH.md` mục "đọc lại BXH".

Xếp theo đòn bẩy đã đo, **không** theo cảm giác:

1. **Hạ `k/g` mà giữ recall** — đòn lớn nhất còn lại và không cần model, không cần tiền.
   `k/g = 2,39`; **65,4% ref khai nằm đúng tài liệu, sai dòng**. Mỗi 1,0 giảm ở k/g đáng
   **+0,03 macro**. Cắt cứng theo hạng đã hết (4 lần quét, đỉnh 5,62) và cổng `lookup.find` đã ĐO
   là hỏng (13% ref có điểm). Ứng viên còn lại: **ngưỡng tương đối theo điểm anchor** — cần sửa
   `rank_context.py` xuất điểm kèm khoá.
2. **18 câu đếm**, ≥10 đang nộp con số không thể là phép đếm; đáp án chặn trên bởi `question.tickers`.
   Free-roll. Chưa code. Đo tỷ lệ giải điều kiện lọc **trước** khi code.
3. **4 câu thiếu phép chia đơn vị** (748/793/824 triệu cổ phiếu, 213 triệu USD) — sửa được chắc
   chắn, nhưng phải chặn câu đếm trước, nếu không id 990/1005 (đang đúng) sẽ hỏng.
4. **Median cohort** (lọc >/&lt; trung vị rồi rank theo tỷ số) — ~13 câu nested. Chưa code.
5. **Regen `planned`/`located` trên `anchor_keys`** — nhớ bài học helpers: chạy được ≠ đúng.
6. **Đừng** bật `implausible()` (đã đo: 57/94 không có nhánh thay thế, 7 câu còn lại ra số rác),
   đừng chồng threshold mù, `label_t3`, hạ `MIN_LABEL_SCORE`, train synth, tối ưu % runnable.
7. **Hỏi BTC** công thức private (macro 3 vs nghiêng EXEC) nếu chưa có trả lời chính thức. Câu trả
   lời quyết định mục 1 hay mục 2–5 mới là ưu tiên.

### 21. `impossible()` không bao giờ chạy trên câu đơn vị tỷ lệ

`impossible()` mở đầu bằng `if scale is None: return False`, và `unit_scale` là None cho
`phan_tram`/`lan`/`vong` vì `UNIT_SCALE` chỉ ánh xạ từ tiền tệ. **298 câu đi qua mọi nhánh không
có cổng vệ sinh.** Biết điều này trước khi tin rằng một nhánh "đã được kiểm tra tính hợp lý".
Bật cổng lên thì vô ích (đã đo) — nhưng đừng lặp lại giả định rằng nó đang chạy.

### Rủi ro private (giữ từ 07/08)

- Hằng số `result = 0.0` còn lại (lastresort đã giảm): giữ trung thực, không ngụy trang.
- Private tính retrieval + answer + exec; TABLES vẫn là lợi thế vs synera (TABLES=0).
