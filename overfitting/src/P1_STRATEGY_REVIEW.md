# Nhận xét kế hoạch P1/P2 + chiến lược lấp 648 câu rỗng

Ngày: 2026-08-04. Mọi con số dưới đây đo trực tiếp trên artifact submission #6
và `artifacts/retrieval.jsonl`, không phải ước lượng.

## 1. Khoảng trống thật sự nằm ở đâu

| Lớp câu hỏi | Số câu | % rỗng | Đóng góp vào 648 |
|---|---:|---:|---:|
| Đơn giản (1 ticker, 1 năm, không aggregate) | 505 | 43% | 217 |
| Phức hợp (≥2 ticker / ≥2 năm / aggregate / "giả sử") | 507 | 85% | 431 |

Ba sự thật đã kiểm chứng:

1. **100% câu rỗng đều CÓ bảng trong context.** Không câu nào thiếu dữ liệu đầu vào.
2. **Router bắt entity đúng** — chỉ 2/176 câu multi-ticker bị bắt thiếu.
3. **Với câu đơn giản, tăng k hoặc rerank bảng KHÔNG cứu được.** Điểm khớp nhãn
   tốt nhất: median 54 (k=4) → 58 (top-20) → 59 (quét toàn bộ báo cáo). Đường
   cong phẳng nghĩa là dòng cần tìm không "ở xa hơn trong danh sách"; nó nằm
   ngay đó nhưng **tầng khớp không nhận ra**.

Soi tay xác nhận điểm 3:

| id | metric_norm | Nhãn đúng trong báo cáo | Điểm |
|---|---|---|---:|
| 179 | `so du tra truoc cho nguoi ban tinh dong` | "Trả trước cho người bán" | 63 |
| 247 | `tong so tien tra truoc cho nguoi ban ngan han co phan` | "2. Trả trước cho người bán ngắn hạn" | 64 |

Ngưỡng rule là 78 → từ chối. Hai lỗi cộng hưởng:

- `metric_norm` sinh bằng cách **trừ stopword khỏi câu** nên còn rác
  ("so du", "tinh dong", "co phan", mảnh tên công ty) làm loãng điểm.
- Khớp thuần từ vựng không xử lý được diễn đạt khác nhau.

## 2. Nhận xét kế hoạch P1/P2 được đề xuất

### Đúng và nên giữ

- **P1.4 — dynamic evidence count.** Được dữ liệu ủng hộ mạnh nhất trong nhóm P1.
  `--expand-docs` đại trà đã thất bại (DOCS precision 0.91→0.28); mở rộng *có
  điều kiện* theo số kỳ/số công ty mà câu hỏi thực sự yêu cầu là cách đúng, và
  nó chính là thứ kéo DOCS_RECALL 0.80 lên mà không giết precision.
- **P1.3 — formula registry.** Nhắm đúng 507 câu phức hợp. Ưu tiên trong nhóm
  này: ranking/so sánh 2 công ty và growth 2 kỳ (nhiều câu nhất), không phải CAGR.
- **P1.1 — query decomposition.** Cần cho câu đa thực thể. Nhưng xem mục 3: nó
  không phải bước đầu tiên.
- **P1.5 — context serialization.** Đúng hướng, nhưng phần "unit theo từng row"
  đã có sẵn (`unit_scale` trong tidy CSV). Phần thiếu thật là **thu hẹp bảng
  thành shortlist dòng ứng viên** trước khi đưa vào prompt.

### Cần đặt lại thứ tự hoặc thu hẹp phạm vi

- **P1.2 — canonical fact store: rủi ro cao nhất, hoãn lại.** Đây là viết lại
  tầng dữ liệu, tốn nhiều ngày, trong khi `cells/{ticker}.parquet` đã là long
  index sẵn có. Thứ còn thiếu chỉ là *tầng canonical hoá tên chỉ tiêu*
  (từ điển đồng nghĩa + mã VAS), không phải một store mới. Làm bản mỏng đó
  trước; chỉ dựng store đầy đủ nếu thật sự chạm trần.
- **P2 xếp sau cùng là sai với dữ liệu.** Kế hoạch coi P2 (BGE-M3 + cross-encoder)
  là "sau khi P0/P1 ổn". Nhưng đo đạc cho thấy nút thắt của 217 câu đơn giản là
  **khớp ngữ nghĩa** — đúng thứ mà dense retrieval/cross-encoder giải quyết.
  Nên kéo phần *matching ngữ nghĩa ở cấp DÒNG* lên sớm, còn phần
  *rerank bảng* thì để sau (đã chứng minh không giúp: đường cong 54→58→59).

### Thiếu hẳn trong kế hoạch

- **Bộ đánh giá offline theo đúng phân bố câu hỏi.** Hiện `val_gold.json` chỉ phủ
  6 chỉ tiêu VAS trên câu đơn giản — tức là **không đo được** chính lớp câu đang
  hỏng. Không có nó, mỗi thay đổi P1/P2 phải trả giá bằng một lượt nộp.
  Có thể sinh deterministic từ store: growth 2 kỳ, tỷ lệ 2 chỉ tiêu, max/min
  nhiều công ty — đáp án tự tính được nên không cần gold của BTC.
- **Hiệu chuẩn ngưỡng + xử lý mơ hồ.** Rule hiện tại là nhị phân (≥78 nhận, dưới
  thì bỏ). Với các trường hợp 62–64 điểm, đúng cách là đưa **shortlist 5–10 dòng
  ứng viên** cho LLM chọn, thay vì im lặng bỏ qua.

## 3. Thứ tự đề xuất (theo tỷ lệ điểm-được/công-sức)

**Bước 1 — Extractive metric + shortlist dòng ứng viên** (rẻ nhất, ~217 câu)
- Trích metric bằng cách *lấy ra* cụm chỉ tiêu (trước "của/năm/cuối/tính bằng"),
  thay vì trừ stopword.
- Hạ ngưỡng rule xuống ~60 nhưng khi có nhiều ứng viên thì **không tự quyết**:
  đưa top-8 dòng (kèm mã VAS, cột, đơn vị) vào prompt cho LLM chọn.
- Đây là schema-linking của text-to-SQL, đúng bài học CHESS/DIN-SQL trong strategy.

**Bước 2 — Bộ eval offline đa lớp** (bắt buộc trước khi làm tiếp)
- Sinh synthetic cho: growth, ratio, so sánh 2 công ty, max/min nhiều công ty.
- Không có nó, bước 3–4 không thể đo được hiệu quả.

**Bước 3 — Decomposition + formula registry + dynamic evidence** (~431 câu)
- Ba việc này đi cùng nhau: tách câu thành các fact (ticker, năm, chỉ tiêu),
  lấy evidence theo từng fact, rồi ráp bằng công thức.
- Bắt đầu bằng 2 mẫu phổ biến nhất: **so sánh/chênh lệch 2 thực thể** và
  **growth 2 kỳ**, chưa cần bao phủ hết CAGR/ranking phức tạp.

**Bước 4 — Semantic matching (BGE-M3) ở cấp dòng, rồi mới tới rerank bảng**
- Embed nhãn dòng + ngữ cảnh bảng; dùng cho cả bước 1 và bước 3.
- Rerank bảng bằng cross-encoder chỉ để nâng TABLES_F2 (chỉ số riêng), không
  kỳ vọng nâng ANSWER.

## 4. Về "các đội khác có retrieval docs mạnh hơn"

DOCS_F2 của ta 0.8093 (P 0.9168 / R 0.7999). Precision đã cao — phần thiếu là
recall 20%, mà phần lớn nằm ở **câu đa thực thể/đa kỳ** (báo cáo thứ hai không
được lấy). Nghĩa là: cách nâng DOCS_RECALL không phải là "lấy rộng hơn" (đã thử,
thất bại) mà chính là **P1.4 dynamic evidence theo cấu trúc câu hỏi**. Hai mục
tiêu này trùng nhau — làm bước 3 sẽ nâng cả ANSWER lẫn DOCS_RECALL.

## 5. Ranh giới bằng chứng

- **Leaderboard-confirmed:** mọi metric #1–#6.
- **Artifact-confirmed:** phân lớp câu hỏi, tỷ lệ rỗng theo lớp, 100% câu rỗng có
  bảng, đường cong khớp nhãn 54→58→59, ví dụ id 179/247/707/759/780.
- **Suy luận (chưa kiểm chứng):** ước lượng phần điểm thu được của từng bước.
  `label_metric_score` là proxy từ vựng do ta tự định nghĩa, nên "điểm 59" đo
  mức khớp *từ vựng*, không phải giới hạn ngữ nghĩa tuyệt đối.
- **Chưa biết:** công thức xếp hạng cuối của BTC (trọng số giữa TABLES/DOCS/
  ANSWER/EXEC) — ảnh hưởng trực tiếp tới việc nên dồn sức vào trục nào.
