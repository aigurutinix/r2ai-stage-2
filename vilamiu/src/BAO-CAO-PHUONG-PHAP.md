# ViFinQA — Báo cáo phương pháp

**Đội vilamiu** · cập nhật 09/08/2026

Tài liệu này là bản trình bày cho ban tổ chức. Mỗi mục ghi *phát hiện là gì*, *đo được bao nhiêu*,
và *đã thay đổi gì trong hệ thống*. Mọi con số đều lấy từ leaderboard công khai hoặc từ script
chạy lại được, không có con số ước lượng nào được trình bày như kết quả đo.

---

## 0. Tóm tắt

Hệ thống trả lời câu hỏi tài chính tiếng Việt đầu-cuối: từ 1.973 báo cáo và 146.246 bảng, với mỗi
câu hỏi phải trả về (a) danh sách bảng liên quan, (b) đáp án số, (c) chương trình pandas tái lập
được đáp án đó.

Kết quả công khai:

| | TABLES_F2 | ANSWER | EXEC | Trung bình macro |
|---|---|---|---|---|
| **vilamiu** | 0,5620 | 0,3261 | 0,3241 | **0,4041** |

Ba phát hiện tạo ra phần lớn kết quả, và **không phát hiện nào đến từ việc thêm mô hình**:

1. **Bộ dữ liệu được sinh ra với ràng buộc cấm câu hỏi chép nhãn bảng** → mọi hệ thống đánh chỉ
   mục theo nhãn đều có trần cứng. Đổi sang đánh chỉ mục văn xuôi giới thiệu bảng: TABLES_F2
   0,4718 → 0,5618, đồng thời EXEC và ANSWER mỗi cái +0,012.
2. **`F2 = 5h/(4g+k)`** — cách hiểu phổ biến "β=2 nên ưu tiên recall" dẫn tới quyết định sai trong
   vùng vận hành thực tế. Khai nhiều bảng hơn làm điểm *giảm*.
3. **Mã số theo Thông tư 200 cho phép khớp chính xác** thay cho khớp nhãn mờ. Từ điển 344 mã dựng
   từ corpus, kiểm bằng đẳng thức kế toán ở 99,6%. Nối vào tầng tra cứu cùng với việc sửa định
   nghĩa tỷ số: +6 câu EXEC và +6 câu ANSWER.
4. **Số bảng khai báo phải suy từ câu hỏi, không từ chương trình của chính mình.** Chi tiết ở mục
   1; đây là thay đổi cho TABLES_F2 lớn nhất sau chỉ mục truy hồi.

---

## 1. Thước đo quyết định thiết kế

Điểm cuối là **trung bình macro của ba tiêu chí** — TABLES_F2, Answer Accuracy, Execution Accuracy.
Điều này đổi hoàn toàn thứ tự ưu tiên: một hệ thống mạnh về sinh chương trình nhưng bỏ trống phần
truy hồi mất một phần ba điểm.

### Công thức F2, và vì sao trực giác phổ biến sai

Với `g` bảng gold, `k` bảng ta khai, `h` bảng khai đúng:

```
F2 = (1+4)·P·R / (4P + R),  P = h/k,  R = h/g   =>   F2 = 5h / (4g + k)
```

Cách đọc thông thường là "β=2 nên recall quan trọng gấp bốn lần precision, vậy cứ khai rộng".
**Điều đó chỉ đúng khi P ≈ R.** Khi P ≪ R, mẫu số `4P + R` bị `R` chi phối, và

```
F2 ≈ 5PR / R = 5P
```

tức F2 trở thành hàm của **precision**. Chúng tôi đã trả giá để học điều này — hai lượt nộp, mỗi
lượt chỉ đổi số bảng khai và giữ đáp án cố định tuyệt đối:

| số bảng khai / câu | TABLES_F2 | Precision | Recall |
|---|---|---|---|
| 4,21 | **0,4718** | ~0,31 | ~0,54 |
| 12 | 0,3577 | 0,1485 | 0,7416 |
| 20 | 0,2833 | 0,0994 | 0,7773 |

Recall tăng từ 0,54 lên 0,78 mà điểm rơi 40%. Công thức khớp cả ba điểm.

### Số bảng khai phải thay đổi theo từng câu

Từ công thức, `k` tối ưu phụ thuộc `g`. Mà `g` **dự đoán được**: gold là đúng các bảng mà chương
trình tham chiếu đọc, nên `g ≈ số cặp (mã chứng khoán × năm)` câu hỏi chạm tới. Span trung bình
của 1.012 câu là 2,65; `g` suy ngược từ leaderboard là 2,40–2,72.

441/1.012 câu chỉ hỏi một công ty trong một năm. Với `g = 1`, khai 4 bảng chặn F2 ở `5/(4+4) =
0,625` **dù truy hồi hoàn hảo**. Ngược lại câu hỏi 8 cặp mà khai 4 bảng thì bị khai thiếu nghiêm
trọng.

Chính sách hiện tại: `k = clamp(2·ĝ, 3, 11)` với `ĝ = max(số bảng chương trình đọc, min(span, 8))`.
Quét bốn điểm với đáp án cố định để xác định hệ số:

| k trung bình | 5,00 | **5,94** | 6,82 | 7,81 |
|---|---|---|---|---|
| TABLES_F2 | 0,5561 | **0,5618** | 0,5577 | 0,5498 |

**Lưu ý đã đo được:** đỉnh này *không cố định*. Sau khi cải thiện tầng sinh chương trình, số bảng
chương trình đọc tăng lên, và đỉnh dịch lên trên 6,46. Chúng tôi đã mất một lượt nộp vì giả định
đỉnh cũ còn đúng.

### Lỗi khái niệm: `#evidence` không phải ước lượng của `g`

Chính sách ban đầu ước lượng `g` bằng `max(#evidence, span)`. Nhưng `#evidence` là số bảng **chương
trình của chúng tôi** đọc, còn `g` là số bảng **chương trình gold** đọc — hai đại lượng khác nhau,
và chúng tôi đã lấy cái thứ nhất ước lượng cái thứ hai rồi nhân đôi.

Chỗ lệch nhất là câu một-công-ty-hai-kỳ: 167 câu, trung bình 2,99 bảng evidence, khai 6,35 bảng
trong khi tối ưu là ~4. Sửa thành `k = clamp(2 · span, 3, 11)` với evidence chỉ còn là *sàn* của k:

| số cặp (mã × năm) | câu | k cũ | k mới |
|---|---|---|---|
| 1 | 441 | 3,82 | 3,29 |
| **2** | **167** | **6,35** | **4,67** |
| 3 | 114 | 7,75 | 6,00 |
| 4 | 115 | 8,94 | 8,00 |

Precision 0,3184 → **0,3369** trong khi recall chỉ giảm 0,7315 → 0,7164 — phần cắt đi gần như toàn
bảng sai. TABLES_F2 **+0,0126**.

Điều đáng nói về phương pháp: chúng tôi đã quét hệ số nhân bốn lần (1,5 / 2,0 / 2,5 / 3,0) và cả
bốn đều xác nhận 2,0 là tối ưu — **vì quét sai trục**. Hệ số đúng; sai ở đại lượng được nhân. Không
phép quét tham số nào phát hiện được loại lỗi này; chỉ có việc mổ xẻ phân bố k theo từng nhóm câu.

---

## 2. Kiến trúc: thang deterministic-first

Nguyên tắc: **cơ chế xác định chạy trước, mô hình chỉ nhận phần dư.** Lý do là số liệu, không phải
sở thích — mọi cơ chế xác định đều đo được cao hơn mọi cơ chế mô hình trên cùng tập câu.

```
   Câu hỏi
      │
      ├─ Phân tích câu hỏi (xác định)  ──► mã chứng khoán, năm, phạm vi báo cáo, chỉ tiêu, đơn vị
      │
      ├─ Truy hồi bảng ─── lọc metadata cứng ──► BM25 trên văn xuôi giới thiệu bảng
      │
      ├─ TẦNG 1  Tra cứu một ô bằng khớp nhãn + xác nhận chéo năm      284 câu   ~55%
      ├─ TẦNG 2  Hợp thành theo trục năm / trục công ty                 79 câu   ~55%
      ├─ TẦNG 3  Tỷ số và sàng lọc hai tầng                             39 câu   ~30%
      │            (chia hai ô 27 · đọc ô % có sẵn 7 · sàng lọc 5)
      ├─ TẦNG 4  SLM sinh chương trình pandas (Qwen3-8B)               201 câu   19,5%
      ├─ TẦNG 5  SLM định vị một ô (Qwen2.5-Coder-14B)                  93 câu   13,3%
      ├─ TẦNG 6  SLM lập kế hoạch k ô + một phép toán                  158 câu    7,2%
      ├─ TẦNG 7  Dự phòng: khớp nhãn không ngưỡng                      106 câu    5,9%
      ├─ TẦNG 8  Quét cột                                                4 câu    ~0%
      └─ TẦNG 9  Nỗ lực cuối cho câu bị khoá hình dạng                  30 câu    ~5,9%
      │
      └─ Kiểm chứng: chạy lại chương trình trên đúng CSV sẽ nộp
```

**994/1.012 câu (98,2%)** có chương trình pandas thật. Trong đó **854 câu** đến từ sáu tầng đầu,
140 câu từ ba tầng cuối vốn đóng góp rất ít. **18 câu** không có chương trình nào và được nộp trung
thực dưới dạng `result = 0.0` thay vì đoán.

Tầng 9 tồn tại vì luật vòng private: khi chỉ EXECUTION được chấm, giữ im lặng và đoán sai cho cùng
kết quả nên im lặng an toàn hơn. Khi ANSWER được chấm riêng và `pandas_query` bị soi tay, im lặng
thôi miễn phí — 48 chỗ trống thành 48 đáp án sai cộng 48 hằng số cứng. Đổi 30 trong số đó sang
chương trình thật: **+2 câu** trên cả hai tiêu chí.

### Vì sao độ chính xác giảm đơn điệu

`42,8% → 19,5% → 13,3% → 7,2% → 5,9%`

Đây **không phải** chuyện làm dở dần. Mỗi tầng chỉ chạy trên phần dư của tầng trước, và phần dư khó
hơn *theo cấu trúc* — nó khó chính vì mọi cơ chế dễ hơn đã lấy phần của mình. Hệ quả thực hành:
thêm tầng thứ tám là đường cụt. Phần dư còn ~106 câu ở 5,9%; nâng lên 20% cũng chỉ được +15 câu.
Chỗ có đủ độ lớn nằm ở **pool đã phủ**, và đó là nơi chúng tôi tập trung.

### Kết luận đo được nhiều lần: cơ chế xác định thắng cơ chế mô hình

| So sánh | Kết quả |
|---|---|
| Định vị bằng mô hình thay khớp nhãn trên cùng pool | **−13 câu** |
| Thay embedding cho mô hình ở cùng vị trí | **−9 câu** |
| Qwen2.5-Coder-14B so với Qwen3-8B (sinh chương trình) | 16 so với 17 câu — hoà |
| Sửa logic chọn ưu tiên xác định | **+2 câu**, không mô hình nào tham gia |

**Nhưng có loại đầu tư thứ ba, và nó lớn hơn cả hai.** Kết luận trên nói về *cơ chế trả lời*. Cả
cơ chế xác định lẫn cơ chế mô hình đều đọc **cùng một danh sách bảng**, nên sửa danh sách đó nâng
tất cả cùng lúc — đó chính là mục 3.

---

## 3. Phát hiện chính: bộ dữ liệu được sinh ra để tránh nhãn bảng

### Quan sát

Bộ so khớp nhãn của chúng tôi bão hoà ở 42% *ngay cả khi lấy 50 ứng viên*. Tăng độ sâu truy hồi
không giúp gì. Điều đó nói nút thắt không nằm ở xếp hạng.

### Nguyên nhân

Gói mã ViFinQA do ban tổ chức phát hành có một bộ duyệt chất lượng câu hỏi
(`judge_and_maybe_rewrite`) **loại bỏ mọi câu hỏi chép nguyên văn nhãn dòng hoặc nhãn cột**, và
viết lại chúng bằng ngôn ngữ tài chính tự nhiên.

Hệ quả: **theo thiết kế của bộ sinh dữ liệu, từ ngữ của câu hỏi là từ ngữ không có trong bảng.**
Nó nằm ở đoạn văn giới thiệu bảng — tiêu đề thuyết minh kiểu "5.2. Phải thu ngắn hạn của khách
hàng". Chỉ mục của chúng tôi khi đó chứa caption, dòng đơn vị, hàng tiêu đề và cột nhãn: đúng cái
phần dữ liệu được viết ra để tránh.

### Giải pháp và đo lường

Ba biến thể chỉ mục, **cùng BM25, cùng bộ tách từ, cùng bộ lọc metadata** — chỉ khác văn bản được
chấm điểm:

| mức cắt | nhãn bảng | ngữ cảnh cả trang | **đoạn văn sát bảng** |
|---|---|---|---|
| 3 | 60,1% | 58,9% | **62,9%** |
| 4 | 65,0% | 64,3% | **69,4%** |
| 5 | 69,0% | 69,6% | **74,5%** |
| 30 | 86,2% | 92,6% | **94,4%** |

*Thước đo: tỷ lệ bảng mà đường trả lời thật sự đã đọc, được xếp hạng lại tìm thấy. Phép đo này
thiên vị **chống lại** hai cột phải, vì những bảng đó vốn do xếp hạng cũ tìm ra.*

Ngữ cảnh cả trang thua vì **mọi bảng trên cùng một trang nhận y hệt một đoạn văn** — không phân
biệt được. Hợp nhất RRF thắng ở mức cắt 30 (95,3%) nhưng không nhúc nhích ở mức cắt 4, nên vô dụng:
F2 bị giới hạn bởi precision (mục 1).

### Kết quả trên leaderboard

| | trước | sau |
|---|---|---|
| TABLES_F2 | 0,4718 | **0,5618** |
| ANSWER | 0,2885 | **0,3004** |
| EXEC | 0,2846 | **0,2984** |

Một thay đổi, ba tiêu chí cùng lên. Precision *và* recall cùng tăng (0,31 → 0,331 và 0,54 → 0,723)
— dấu hiệu xếp hạng tốt hơn thật, không phải đánh đổi hai đầu.

### Hệ quả kéo theo

Ba bộ nhớ đệm của mô hình (sinh chương trình, định vị ô, lập kế hoạch — chiếm 45% bài nộp) đều
được tạo trên xếp hạng cũ. Sinh lại tầng sinh chương trình trên xếp hạng mới: **+16 câu EXEC và
+16 câu ANSWER**, chi phí bằng không (chạy qua API mô hình mở).

---

## 4. Từ điển mã số: khớp chính xác thay cho khớp mờ

### Ý tưởng

Báo cáo tài chính Việt Nam có cột **Mã số** theo Thông tư 200/2014/TT-BTC. Mã là định danh pháp lý;
nhãn chỉ là cách diễn đạt. `cdkt:270` luôn là tổng cộng tài sản dù bảng viết "TỔNG CỘNG TÀI SẢN",
"Tổng tài sản" hay bị OCR thành "TÔNG CÔNG TAI SAN". **Khớp theo mã là khớp chính xác.**

### Dựng

Quét 111.728 bảng, tìm cột bị chi phối bởi số nguyên 2–3 chữ số, ghép mã với nhãn bên cạnh, phân
loại báo cáo bằng các dòng neo trong chính bảng. Kết quả: **344 mã phân biệt**, mỗi mã kèm tới 25
biến thể nhãn thực tế và tần suất, từ 6.831 bảng có cột mã. Thời gian: 11 giây.

### Kiểm chứng độc lập

Từ điển được kiểm bằng số học mà dữ liệu không thể thoả mãn do trùng hợp:

| đẳng thức | đúng / kiểm | |
|---|---|---|
| `cdkt:270 = 100 + 200` | 536 / 536 | 100% |
| `cdkt:440 = 300 + 400` | 1.106 / 1.112 | 99,5% |
| `cdkt:270 = 440` | 11 / 11 | 100% |
| `kqkd:20 = 10 − \|11\|` | 1.365 / 1.372 | 99,5% |
| `kqkd:50 = 30 + 40` | 1.454 / 1.457 | 99,8% |
| **tổng** | **4.472 / 4.488** | **99,6%** |

Nếu cột mã bị đọc sai, hoặc nhãn bị gán nhầm mã, các đẳng thức này sẽ hỏng.

### Hai lỗi tự tìm ra trong quá trình, và cả hai đều thành phát hiện

**Lỗi 1 — nhận dạng loại báo cáo từ caption là sai.** Bản đầu cho 2.692 bảng lưu chuyển tiền tệ so
với 331 bảng cân đối kế toán. Con số đó tự tố cáo bug: mọi báo cáo đều có bảng cân đối. Nguyên
nhân là tiêu đề báo cáo nằm ở văn bản *trên* bảng chứ không nằm trong bảng. Chuyển sang nhận dạng
bằng các dòng neo: 3.056 → 6.831 bảng, ba loại cân bằng.

**Lỗi 2 — quy ước dấu không nhất quán.** Đẳng thức `20 = 10 − 11` ban đầu chỉ đạt 69,4%, trông y
hệt một lỗi ánh xạ mã. Kiểm tay một ví dụ:

```
kqkd:10  =  9.258.073.280.674     doanh thu thuần
kqkd:11  = -8.215.933.902.107     giá vốn hàng bán — mang dấu ÂM
kqkd:20  =  1.042.139.378.567     = 10 + 11, khớp đúng từng chữ số
```

**Giá vốn được in trong ngoặc kế toán ở một số báo cáo và in dương ở số khác.** Phép kiểm sai, dữ
liệu đúng. Sửa thành `10 − |11|` → 99,5%.

Hệ quả vượt ra ngoài script: **mọi công thức đọc mã chi phí phải chuẩn hoá dấu trước khi tính.**
Nếu không, các tỷ số dùng giá vốn sẽ sai ngẫu nhiên tuỳ báo cáo — và sai kiểu đó không phát hiện
được từ kết quả, vì con số vẫn nằm trong khoảng hợp lý.

---

## 5. Kiểm chứng: mọi chương trình nộp đi đều chạy lại tại chỗ

Trước mỗi lần nộp, cả 1.012 chương trình được chạy lại trên **đúng các byte CSV sẽ nộp**, đọc bằng
`pandas.read_csv` — đúng bộ đọc mà hệ thống chấm dùng — và phải tái lập đúng đáp án đã khai. Bất
biến hiện tại: **1.012/1.012**.

Bất biến này ra đời từ một lỗi đáng nhớ. Bộ kiểm cục bộ ban đầu dựng DataFrame từ lưới chuỗi, nên
mọi ô là chuỗi. `read_csv` thì **tự suy kiểu từng cột**. Hệ quả: chương trình gọi
`cell.replace(".", "")` chạy đúng ở cục bộ và nổ khi chấm với
`AttributeError: 'numpy.float64' object has no attribute 'replace'`.

Nói cách khác, **bộ kiểm sai ở đúng cái nó tồn tại để dự đoán** — nó phê duyệt code sẽ crash. Sau
khi sửa để round-trip qua CSV rồi `read_csv`: 5 câu chỉ crash khi suy kiểu, 1 câu lệch giá trị,
tất cả được phát hiện trước khi nộp.

---

## 6. Giới hạn

Chúng tôi nêu trước những điểm yếu thay vì để chúng bị phát hiện:

- **18/1.012 câu không có đáp án.** Chúng được nộp dưới dạng `result = 0.0`. Có thể làm chúng đọc
  một DataFrame để qua được bộ kiểm tự động, nhưng đó là nguỵ trang một hằng số. Chúng tôi giữ
  nguyên vì đó là biểu thị trung thực rằng hệ thống không trả lời được.
- **452 câu (45%) đến từ mô hình ngôn ngữ nhỏ** ở độ chính xác 7,2%–19,5%, và **140 câu nữa** đến
  từ ba tầng cuối gần như không đóng góp điểm. Đây là phần yếu nhất của hệ thống: hơn một nửa bài
  nộp nằm ở vùng dưới 20% chính xác.
- **Execution Accuracy 0,3142** còn cách xa đội dẫn đầu về tiêu chí này.
- **Pool 133 câu chưa khai thác:** các câu sàng lọc hoặc so ngưỡng theo một tỷ số phải tự tính
  (ROA, ROE, thanh toán nhanh, biên lợi nhuận). Hiện đúng khoảng 12/133 (9%) vì chúng rơi vào các
  tầng yếu nhất. Đây là hướng phát triển tiếp theo.
- **Định nghĩa tỷ số của chúng tôi có lỗi đã xác nhận:** ROA và ROE đang chia cho số cuối năm thay
  vì bình quân đầu–cuối năm. Với dung sai 0,02%, đây là sai chứ không phải sai số.

---

## 7. Tái lập

Toàn bộ hệ thống là một số artifact xác định cộng một lệnh đóng gói. Không có bước nào phụ thuộc
trạng thái thủ công.

```bash
# 1. Trích bảng và phân tích câu hỏi (CPU, ~40 giây)
python scripts/run_extract.py
python scripts/run_parse.py

# 2. Chỉ mục truy hồi theo văn xuôi (CPU, ~3 phút)
python scripts/build_anchor_index.py
python scripts/rank_context.py 30 artifacts/anchor_index.jsonl artifacts/anchor_rank.jsonl

# 3. Từ điển mã số và kiểm chứng (CPU, ~25 giây)
python scripts/build_ma_so.py
python scripts/verify_ma_so.py

# 4. Các tầng mô hình (mô hình mở ≤14B)
python scripts/run_generate.py --use-rerank --reranked artifacts/anchor_group.jsonl
python scripts/run_locate.py  --reranked artifacts/anchor_keys.jsonl
python scripts/run_plan.py    --reranked artifacts/anchor_keys.jsonl

# 5. Đóng gói và kiểm chứng
python scripts/run_submit.py 10 4 artifacts/gen_anchor.jsonl
```

Mọi mô hình đều là mô hình mở ≤14B tham số phát hành trước 01/06/2026, được cưỡng chế bằng một
danh sách cho phép trong mã nguồn (`ALLOWED_MODELS`).

---

## 8. Nguồn tham chiếu

- **Thông tư 200/2014/TT-BTC** — hệ thống mã số chỉ tiêu báo cáo tài chính doanh nghiệp.
- **Gói mã ViFinQA do ban tổ chức phát hành** (`vifinqa-official/`) — chúng tôi đã đọc mã sinh dữ
  liệu và mã đánh giá để hiểu định nghĩa gold và ràng buộc sinh câu hỏi. Phát hiện ở mục 3 đến từ
  việc đọc này. Chúng tôi **không truy cập `reference_data`** (đáp án gold, `is_public: false`).
- **AITeamVN/Vietnamese_Embedding** — dùng trong các thí nghiệm đối chiếu bộ so khớp nhãn.
- Định nghĩa tỷ số tài chính theo chuẩn phân tích tài chính doanh nghiệp phổ thông.

---

## Phụ lục: nhật ký tiến triển

| Ngày | Bản | EXEC | ANSWER | TABLES_F2 | Thay đổi chính |
|---|---|---|---|---|---|
| 02/08 | nền | 0,2016 | 0,2036 | — | tra cứu nhãn + BM25 |
| 04/08 | compose | 0,2273 | 0,2292 | — | hợp thành theo năm, sửa bộ trích chỉ tiêu |
| 05/08 | axis | 0,2352 | 0,2391 | — | hợp thành theo công ty |
| 05/08 | scalefix | 0,2391 | 0,2431 | — | sửa hệ số đơn vị |
| 06/08 | divide | 0,2431 | 0,2470 | — | sàng lọc hai tầng, chia tỷ số |
| 06/08 | label_doc | 0,2846 | 0,2866 | 0,4718 | khai bảng theo nhãn toàn tài liệu |
| 07/08 | decl_k12/20 | — | — | 0,3577 / 0,2833 | **thí nghiệm thất bại**, xem mục 1 |
| 07/08 | anchor_search | 0,2984 | 0,3004 | 0,5618 | **chỉ mục văn xuôi** — mục 3 |
| 07/08 | regen_llm | 0,3142 | 0,3162 | 0,5526 | sinh lại tầng chương trình |
| 07/08 | maso_ratio | 0,3202 | 0,3221 | 0,5494 | mã số + sửa định nghĩa tỷ số — mục 4 |
| 08/08 | searchk20 | 0,3221 | 0,3241 | 0,5494 | độ sâu truy hồi đo lại trên xếp hạng mới |
| 09/08 | **spank_lastresort** | **0,3241** | **0,3261** | **0,5620** | **k suy từ câu hỏi + bỏ 30 hằng số** |

Điểm trung bình macro: 0,3477 → **0,4041**.
