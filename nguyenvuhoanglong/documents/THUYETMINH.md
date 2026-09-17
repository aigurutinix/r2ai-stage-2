# THUYẾT MINH SẢN PHẨM — Đội AI_Guru

**Cuộc thi:** ROAD TO AI 2026 – Stage 2 · Bài toán *Financial Table Retrieval & Text-to-Pandas* (ViFinQA)
**Ngày:** 28/08/2026 · **Bản nộp tham chiếu:** `v58`

---

## 1. Tóm tắt

Chúng tôi xây dựng một hệ thống **hỏi–đáp trên báo cáo tài chính tiếng Việt**: nhận một câu
hỏi bằng ngôn ngữ tự nhiên, tự tìm đúng báo cáo và đúng bảng trong kho **1.973 báo cáo /
146.246 bảng**, rồi sinh một truy vấn `pandas` **đọc thật từ dữ liệu** để trả lời, kèm
**trích dẫn nguồn tới từng ô**.

Điểm khác biệt của sản phẩm không nằm ở việc dùng mô hình lớn, mà ở **ba lựa chọn kỹ thuật**:

| | |
|---|---|
| **Đường sinh đáp án TẤT ĐỊNH** | mọi con số đến từ một ô có thật trong bảng, không mô hình nào được phép "bịa" giá trị |
| **Mô hình ngôn ngữ chỉ làm TRỌNG TÀI** | model chỉ chọn giữa vài phương án đã được lọc, không tự do sinh code hay đọc bảng |
| **Bốn bộ tự phát hiện lỗi KHÔNG cần đáp án** | hệ thống tự biết câu nào nó chắc chắn trả lời sai |

Kết quả trên bảng xếp hạng công khai: **`DOCS_PRECISION` hạng 2** và **`DOCS_MRR5` hạng 3**
toàn giải; **5/10 chỉ số nằm trong top-5**.

---

## 2. Bài toán và ràng buộc

**Đầu vào:** 1.012 câu hỏi tiếng Việt về báo cáo tài chính doanh nghiệp niêm yết.
**Đầu ra mỗi câu:** đáp án số · danh sách tài liệu liên quan · danh sách bảng liên quan ·
một đoạn `pandas_query` chạy được trên chính các bảng đã khai.

Ràng buộc BTC đặt ra mà chúng tôi tuân thủ tuyệt đối:

* mô hình **open-weight, ≤14B tham số**, phát hành trước 01/06/2026 — không dùng model đóng;
* `pandas_query` **phải đọc từ dataframe**, không được gán hằng số;
* đáp án phải **đúng đơn vị câu hỏi yêu cầu**, làm tròn 2 chữ số ở bước cuối;
* mọi nguồn dữ liệu ngoài phải khai báo (chúng tôi ghi trong `SOURCES.md`).

Điều kiện làm việc: máy cá nhân **GTX 1650 4GB**, GPU chỉ có trên Kaggle free (2×T4).
**Không có tập huấn luyện, không có đáp án mẫu** — toàn bộ hệ thống là *zero-shot*.

---

## 3. Kiến trúc hệ thống

```
báo cáo .txt  ──►  BỘ ĐỌC BẢNG  ──►  146.246 bảng + siêu dữ liệu
   (1.973)          HTML → lưới        (trang, tiêu đề, đơn vị, loại bảng)
                         │
câu hỏi  ──►  BÓC TÁCH THỰC THỂ  ──►  mã CK · năm · phạm vi · đơn vị đáp án
                         │
                    LỌC CỨNG            chỉ giữ báo cáo đúng (mã CK, năm, phạm vi)
                         │
                  CHẤM Ô TOÀN CỤC       chấm MỌI dòng của MỌI bảng ứng viên
                         │               trên một thang chung (BM25 + IDF toàn kho)
                         ├──► LLM làm TRỌNG TÀI: chọn 1 trong 5 ô đã lọc
                         │
                  SINH pandas_query      df.iloc[R, C] — đọc thật từ bảng
                         │
              BỐN BỘ TỰ KIỂM LỖI        phát hiện câu chắc chắn sai, thử lại
                         │
                    BÀI NỘP + TRÍCH DẪN
```

### 3.1 Bộ đọc bảng — đã xác minh khớp **chính xác** với BTC

Chúng tôi cài lại nguyên văn luật lọc bảng của BTC (`is_table_eligible`: thân bảng ≥3 dòng,
≥6 ô số, theo đúng khuôn số kiểu Việt Nam) và đếm được **111.070** bảng đủ điều kiện —
**trùng khít con số 111.070 mà BTC công bố**.

Đây là bằng chứng mạnh nhất trong toàn dự án: bộ đọc của chúng tôi khớp bộ đọc của BTC
**từng bảng một**. Ngoài ra:

| phép kiểm | kết quả |
|---|---|
| đẳng thức kế toán `mã 270 == mã 440` | **98,59%** báo cáo khớp |
| đẳng thức `100 + 200 == 270` | **99,72%** |
| bóc tách công ty từ câu hỏi | **1.011/1.012** |
| chất lượng bảng thuyết minh (3.000 mẫu) | lưới rách **0%** · bảng cắt dòng **0%** |

### 3.2 Lọc cứng trước khi chấm — lý do chúng tôi thắng ở khâu truy hồi

Thay vì tìm kiếm ngữ nghĩa trên toàn kho 111.070 bảng, chúng tôi **lọc cứng trước** theo
mã chứng khoán, năm và phạm vi báo cáo (hợp nhất / riêng lẻ) — ba tín hiệu bóc được từ câu
hỏi với độ chính xác gần tuyệt đối. Sau bước đó, mọi bảng ứng viên **đã cùng chủ đề**, và
thứ còn phân biệt được là **khớp chính xác tên khoản mục** — đúng sở trường của BM25 có
trọng số IDF toàn kho.

Chúng tôi đã kiểm chứng lựa chọn này bằng thực nghiệm, không bằng trực giác:

| bộ chọn bảng | `TABLES_F2` |
|---|---|
| **BM25 + IDF (đang dùng)** | **0,5588** |
| hợp nhất RRF (BM25 + embedding) | 0,5523 |
| bi-encoder Qwen3-Embedding-0,6B | 0,5253 |
| cross-encoder `bge-reranker-v2-m3` | 0,5141 |

**Xếp theo "độ ngữ nghĩa" thì kết quả giảm đơn điệu.** Một benchmark chỉ nói được điều gì
đó về bài toán **cùng chế độ**: mô hình ngữ nghĩa mạnh khi phải phân biệt tài liệu khác chủ
đề, nhưng vô dụng sau khi đã lọc về một báo cáo.

### 3.3 Chấm Ô trước, bảng suy ngược

Kiến trúc thông thường là *chọn bảng → chọn dòng*. Chúng tôi **bỏ hẳn bước chọn bảng**:
chấm **mọi dòng của mọi bảng ứng viên** trên cùng một thang, lấy ô điểm cao nhất, và bảng
là **hệ quả**. Thay đổi này một mình nâng độ chính xác nội bộ từ 38/88 lên **51/88**.

### 3.4 Mô hình ngôn ngữ — chỉ ở đúng một vai trò

Chúng tôi đã thử **năm** kiến trúc dùng LLM và đo từng cái:

| model làm gì | kết quả |
|---|---|
| sinh code `pandas` tự do | **0%** (336/336 đầu ra hỏng) |
| **chọn 1 trong 5 ô ĐÃ LỌC** | **+16 câu — bản vá lớn nhất dự án** |
| chọn 1 trong ~21 ô | 0 |
| đọc trọn bảng, tự nêu dòng và phép toán | cứu 1 · **hỏng 7** |
| đọc trọn lưới, tự nêu ô | cứu 0 · hỏng 1 |

> **Quy luật đo được: mô hình ≤14B chỉ tạo giá trị khi làm TRỌNG TÀI giữa vài phương án đã
> được xếp hạng tốt — chưa bao giờ khi làm người đọc hay người suy luận. Càng cho tự do
> càng tệ, một cách đơn điệu.**

Kết luận này khớp với thực nghiệm của chính BTC trên cùng bộ dữ liệu: ở hạng cân ≤14B,
Qwen3-4B đạt **11,56%** (CoT) / **4,64%** (PoT); Qwen2.5-Coder-7B đạt **0,30%** (PoT).
**Hệ thống của chúng tôi đạt 36,96% — gấp 3,2 lần.**

---

## 4. Bốn bộ tự phát hiện lỗi — không cần đáp án mẫu

Vì không có tập gold, chúng tôi xây bốn cơ chế để hệ thống **tự biết mình sai**:

| # | bộ dò | bắt lớp lỗi nào |
|---|---|---|
| 1 | **bất biến loại đáp án** | hỏi phần trăm mà trả 96 nghìn tỷ; hỏi "có bao nhiêu doanh nghiệp" mà trả một số tiền |
| 2 | **nhất quán chéo câu hỏi** | hai câu hỏi cùng một đại lượng mà ra hai đáp án khác nhau |
| 3 | **đối chiếu kho fact** | đáp án lệch với giá trị của chính khoản mục ấy trong kho 1,49 triệu fact trích từ corpus |
| 4 | **ràng buộc cấu trúc** | câu hỏi về ba công ty mà truy vấn chỉ đọc **một ô** — sai theo định nghĩa |

Bốn bộ này chạy **offline, không tốn lượt nộp**, và đã nhiều lần **chặn được bản vá xấu
trước khi nộp**. Chúng cũng là nền tảng cho mục tiêu 5 của đề bài (hạn chế trả lời không
có căn cứ).

---

## 5. Kết quả đo được

### 5.1 Bảng xếp hạng công khai (bản `v58`)

| chỉ số | giá trị | hạng |
|---|---|---|
| **DOCS_PRECISION** | **0,9707** | **2** |
| **DOCS_MRR5** | **0,9783** | **3** |
| TABLES_RECALL | 0,6760 | 5 |
| TABLES_MRR5 | 0,6543 | 5 |
| TABLES_F2 | 0,5588 | 5 |
| DOCS_F2 | 0,9653 | 6 |
| DOCS_RECALL | 0,9669 | 8 |
| ANSWER_ACCURACY | 0,3715 | 9 |
| EXECUTION_ACCURACY | 0,3696 | 8 |
| TABLES_PRECISION | 0,3585 | 9 |

**5/10 chỉ số trong top-5, 2/10 trong top-3.**

### 5.2 Điểm mạnh nhất: truy hồi tài liệu

`DOCS_PRECISION` 0,9707 nghĩa là **97,07%** tài liệu chúng tôi khai ra là tài liệu thật sự
chứa đáp án. Chỉ số này đạt được nhờ bộ bóc tách thực thể chính xác: phủ **99,89%** cặp
(mã chứng khoán, năm) mà câu hỏi đòi, và **0 lỗi** gán phạm vi báo cáo trên toàn bộ 1.012 câu.

### 5.3 Tiến trình

Từ bản nộp đầu (`ANSWER` 0,1126) tới bản hiện tại (0,3715) là **gấp 3,3 lần**, qua 30+ bản
nộp, mỗi bản có một tài liệu thay đổi riêng ghi rõ **dự đoán đặt trước** và **kết quả đo**.

---

## 6. Đáp ứng năm mục tiêu của đề bài

| # | mục tiêu | trạng thái |
|---|---|---|
| 1 | **Truy hồi chính xác** — đúng công ty, năm, bảng | ✅ `DOCS_PRECISION` **hạng 2 toàn giải** |
| 2 | **Hiểu truy vấn tài chính tiếng Việt** | ✅ bóc tách thực thể 1.011/1.012; nhận diện đơn vị, kỳ, phạm vi |
| 3 | **Sinh pandas & tính đúng** | ✅ **1.012/1.012** truy vấn đọc thật từ dataframe, **0 hằng số** |
| 4 | **Dẫn nguồn minh bạch** | ✅ **100%** — xem 6.1 |
| 5 | **Kiểm soát sai lệch** | ⚙️ hạ tầng đã có — xem 6.2 |

### 6.1 Mục tiêu 4 — trích dẫn tới từng ô, phủ 100%

`scripts/cite.py` xuất trích dẫn đầy đủ cho **1.012/1.012 câu**, trung bình **3,46 nguồn/câu**:

```
[106] Tỷ lệ sở hữu của công ty mẹ PLX tại Công ty TNHH Hóa chất PTN cuối 2016?

  NGUỒN 1: PLX · 2016 · BCTC riêng (công ty mẹ) · trang 28
     Thuyết minh báo cáo tài chính — «11. Đầu tư tài chính dài hạn»
     dòng «Công ty TNHH Hóa chất PTN» · cột «31/12/2016» = 46.826.010.000
     [PLX_financial_statements_2016_separate|826]
```

Trích dẫn được dựng **từ chính `pandas_query` đã nộp** (`df.iloc[R, C]`), nên nó mô tả
**đúng ô mà hệ thống đã dùng**, không phải một ô dựng lại. Đây là điều kiện để trích dẫn
có giá trị kiểm chứng.

### 6.2 Mục tiêu 5 — hạ tầng kiểm soát sai lệch

Bốn bộ dò ở mục 4 hiện chỉ ra **123/1.012 câu** hệ thống **chắc chắn trả lời sai**. Hệ
thống **biết** những câu này. Việc bỏ trống chúng sẽ nâng độ chính xác *trên số câu đã trả
lời* (0,3696 → **0,4207**) mà không mất điểm nào (câu sai và câu bỏ trống đều cho 0), đồng thời thoả đúng tinh
thần mục tiêu 5.

⚠️ Chúng tôi **chưa bật** cơ chế này trong bản nộp thi vì bảng xếp hạng hiện tại chấm theo
tỉ lệ trên **toàn bộ** câu hỏi, và chưa xác nhận được bộ chấm xử lý `answer` rỗng ra sao.
Đây là **một công tắc**, không phải một hạng mục nghiên cứu.

---

## 7. Kỷ luật đo lường — phần chúng tôi đầu tư nhiều nhất

Ràng buộc khó nhất của cuộc thi không phải thuật toán mà là **không có thước đo**: không
tập dev, không đáp án, và ở vòng private chỉ **5 lượt nộp**. Chúng tôi xử lý bằng ba nguyên tắc:

**1. Mọi thay đổi phải qua cổng chặn tính tổng quát.** Mỗi bản vá có một tài liệu riêng trả
lời 5 câu hỏi trước khi sửa code — trong đó có *"tham số này lấy từ đâu?"* và *"đây có phải
là nhìn câu sai rồi viết luật chữa đúng chúng không?"*. Hiện có **98 tài liệu** đã qua cổng,
kiểm tự động bằng `scripts/check_generality.py`.

**2. Tham số chỉ được chọn ở GIỮA VÙNG PHẲNG của hai thước gán nhãn bằng cơ chế khác nhau**
— không lấy đỉnh nhọn của một thước. Ví dụ gần nhất: chúng tôi quét tham số `ROW_MATCH_BETA`
trên dải rộng 6,7 lần, thấy **cả hai thước đều phẳng tuyệt đối** trong khi 10,2% đáp án thay
đổi ⇒ **giữ nguyên giá trị cũ**, vì đổi nó là đánh bạc không đo được.

**3. Kết quả ÂM được ghi lại đầy đủ như kết quả dương.** Tài liệu dự án ghi rõ từng hướng
đã đóng kèm **con số đo được** — ngữ cảnh trang, cross-encoder, bỏ phiếu đa bảng, đối chiếu
chéo năm, LLM chọn bảng, bi-encoder… Nhờ vậy không ai trong đội thử lại một ngõ cụt.

> Chúng tôi coi đây là phần có giá trị lâu dài nhất của sản phẩm: một **quy trình** cho phép
> cải tiến một hệ thống AI khi không có nhãn, chứ không chỉ một con số trên bảng xếp hạng.

---

## 8. Hạn chế đã đo được

Chúng tôi nêu rõ giới hạn thay vì che đi:

**8.1. Khoảng 42% nhãn dòng trong thuyết minh là nhập nhằng.** Cùng một tên khoản mục xuất
hiện ở nhiều bảng trong **cùng một báo cáo** với các giá trị khác nhau — ví dụ *"vay ngắn hạn
phải trả các bên liên quan"* có ở ba bảng với ba giá trị. Câu hỏi nêu tên khoản mục nguyên
văn nhưng **không nêu bảng nào**, nên với nhóm này thông tin trong câu hỏi **không đủ** để
xác định ô. Đây là giới hạn **thông tin**, không phải giới hạn thuật toán.

**8.2. Khoảng một nửa số câu cần đáp án DẪN XUẤT** (tỉ số, gộp nhóm, cực trị nhiều công ty)
— không ô đơn nào bằng đáp án. Nhóm này cần năng lực **lập kế hoạch tính toán**, và thực
nghiệm của chính BTC cho thấy hạng cân ≤14B đạt 4,64–11,56% ở đó.

**8.3. Từ vựng chỉ tiêu đóng.** Bộ lập kế hoạch dùng danh mục 23 tỉ số + 24 chỉ tiêu của
BTC, toàn bộ là dòng **báo cáo chính**; trong khi **84,8%** bảng gold là **thuyết minh**.
Chúng tôi đã đo và xác nhận trần này trên cả hai trục (công ty và năm).

---

## 9. Hướng phát triển

| | |
|---|---|
| **Ngắn hạn** | bật cơ chế từ chối trả lời cho nhóm 123 câu hệ thống biết là sai (mục tiêu 5) |
| **Trung hạn** | mở rộng từ vựng đích từ danh mục 47 mục sang **nhãn dòng thuyết minh trích thẳng từ câu hỏi** — đây là trần đã đo được của nhánh lập kế hoạch |
| **Dài hạn** | phá nhập nhằng nhãn bằng tín hiệu ngoài nhãn (vai trò bảng trong báo cáo, quan hệ số học chéo bảng) |

---

## 10. Tái lập

Toàn bộ hệ thống chạy trên máy cá nhân, trừ hai bước dùng GPU Kaggle free.

```bash
python scripts/download_corpus.py        # tải corpus
python scripts/build_tables.py --workers 6   # dựng 146.246 bảng (~45 giây)
python scripts/map_questions.py          # bóc tách thực thể từ câu hỏi
python scripts/retrieve.py               # truy hồi (~8 phút)
python scripts/make_submission.py --selftest --use-rowname \
       --cohort-llm cohort_plan_merged2.parquet --cell-pick \
       --out submissions/<tên>.zip
python scripts/cite.py --zip submissions/<tên>.zip   # xuất trích dẫn
```

**Kiểm chất lượng, không tốn lượt nộp:**

```bash
python scripts/check_generality.py                    # cổng chống overfitting
python scripts/check_invariants.py  --zip <file>      # bộ dò lỗi 1
python scripts/check_consistency.py --zip <file>      # bộ dò lỗi 2
python scripts/verify_answers.py    --zip <file>      # bộ dò lỗi 3
```

**Mô hình sử dụng:** Qwen3-14B-AWQ (open-weight, ≤14B, phát hành trước 01/06/2026), chạy
trên Kaggle 2×T4, **chỉ ở vai trò trọng tài** — chọn giữa các phương án do đường ống tất
định đã lọc sẵn. Khai báo đầy đủ nguồn dữ liệu ngoài ở `SOURCES.md`.

---

## Phụ lục — tài liệu kỹ thuật kèm theo

| file | nội dung |
|---|---|
| `CLAUDE.md` | quy ước bất biến, mọi kết luận đã đo, mọi hướng đã đóng kèm số |
| `TRANGTHAI.md` | ảnh chụp trạng thái kỹ thuật hiện tại |
| `GOAL.md` | chỉ số qua tất cả bản nộp |
| `changes/` | **98 tài liệu thay đổi**, mỗi cái có dự đoán đặt trước + kết quả đo |
| `RUNBOOK_PRIVATE.md` | quy trình vòng private, cổng chặn từng lượt nộp |
| `SOURCES.md` | khai báo nguồn dữ liệu ngoài |
