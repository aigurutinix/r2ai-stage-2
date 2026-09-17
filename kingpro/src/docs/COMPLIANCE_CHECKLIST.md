# Checklist tuân thủ R2AI Stage 2

Tài liệu này phân biệt kiểm tra tự động với điều kiện cần BTC/người dự thi xác nhận thủ công.

## Kiểm tra tự động đã đạt

| Yêu cầu | Cách đáp ứng | Kết quả |
| --- | --- | ---: |
| Đủ schema bài nộp | `submission.json` có 1.012 entry và các trường output | Đạt |
| Query dùng evidence | AST yêu cầu dependency `df`/`dfs` | 1.012/1.012 |
| Không gán đáp án số trực tiếp | scan mọi assignment vào `result` | 0 lỗi |
| Evidence tồn tại | kiểm từng `csv_path` | 0 thiếu |
| Provenance khớp bảng được đọc | dựng lại từ CSV/manifest; exact order mặc định, equality theo membership/multiplicity khi audit v192 order-only | 0 lệch |
| Query chạy được | grader pandas 1.1.5, string và typed runtime, có cả `warnings-as-errors` | 1.012/1.012 ở cả bốn chế độ trên v192 |
| Query trả số | kiểm kiểu kết quả | 1.012/1.012 |
| Kết quả chạy lại khớp answer lưu | tolerance theo checker | 1.012/1.012 |
| Source-coordinate | so raw token với bảng BTC gốc | 2.203/2.203 |
| Panel source-coordinate | kiểm các toán hạng panel trên BCTC gốc | 4.783/4.783 |
| Python 3.7 compatibility | parse toàn bộ query bằng grammar 3.7 và chặn API mới | 0 findings |
| Kỳ, intent, table family, source scope | chạy toàn bộ semantic/provenance gates | 0 high-risk findings |
| Đường dẫn ZIP portable | kiểm exact path, dấu `/`, case collision và CRC/full read | 1.940 refs đều có; 0 collision |
| Metric-code | audit mã chỉ tiêu BCTC | 0 findings |
| Nhãn năm phụ thuộc dữ liệu | trace dependency của 53 câu trả năm; không chấp nhận nhãn năm chỉ đến từ literal trong code | v188: 53/53 sạch, 213 nhãn lấy từ DataFrame |
| Grounded compiler phục vụ demo | grammar bảo thủ, source-coordinate Pandas, sandbox + đối chiếu cube; chặn qualifier/đầu kỳ/đơn vị mơ hồ và topology đếm/lọc không hỗ trợ | 100/100 trên compiler-selected registry subset; adversarial gate từ chối 29/29 biến thể sai time-basis/scope/unit/intent; HTTP smoke phải PASS 7/7 ở lần readiness live gần nhất |
| Truy hồi tài liệu theo facet nguồn | tách công ty–năm–scope, mixed scope theo năm, alias non-overlap; chỉ mở rộng kỳ theo tín hiệu kế toán hữu hạn và backfill catalog duy nhất, nhóm mơ hồ fail-closed; không dùng question ID/answer/leaderboard feedback | Full 1.012 source-binding audit: precision `0,979938`, recall `0,988322`, F2 `0,983827`, miss `182 -> 29`; đây là regression nội bộ trên nhãn thực thi v184, không phải BTC Docs F2/gold |
| Table reranker theo CSV nguồn | BM25 pool 40, tự chọn cột nhãn thật thay vì giả định cột đầu; fallback nguyên thứ tự BM25 nếu CSV lỗi | Full 1.012 source-binding A/B trên document stage mới: recall@8 `0,471960 -> 0,552890`, MRR@40 `0,438546 -> 0,532769`; 0 bucket regression; không diễn giải thành BTC Tables F2/gold |
| ZIP đúng layout và tái lập | hai lần deterministic referenced-only packaging cho cùng byte/hash | v192 PASS, 1.935 entries |

## Trạng thái champion và candidate runtime-safe

- Champion đã đo là v192/ID 3552: Execution `0.6877`, Answer `0.6897`, Tables F2-macro `0.6014`, Docs F2-macro `0.9611`; selected và #1 public tại lần kiểm tra gần nhất.
- Candidate v187 chỉ sửa tính portable của ba `pandas_query` q170/q222/q369, giữ nguyên mọi answer, retrieval label và evidence của v184.
- v188 kế thừa v187, chỉ thay code của 53 câu trả năm để nhãn năm được đọc từ ô dữ liệu đã xác minh; BTC đã đo Execution `0.6838`.
- v190 kế thừa v188 và có hai sửa lỗi nguồn q375/q411; BTC đo tăng lên `0.6858`. v191 bổ sung sửa EPS q511 bằng ba ô nguồn.
- v192 kế thừa v191 và chỉ đổi order của đúng cùng tập `relevant_tables` tại 81 panel. BTC đo tăng Execution/Answer đúng một câu, Tables F2 `0.6011 -> 0.6014` và MRR@5 `0.6361 -> 0.6384`; independent replay vẫn chỉ là bằng chứng cơ chế, không phải hidden-gold claim.
- `relevant_docs`/`relevant_tables` có nguồn thật; mọi evidence path trong ZIP khớp exact kể cả trên filesystem phân biệt hoa/thường.
- Mỗi source-audited token có tọa độ đối chiếu dữ liệu BTC gốc; v192 có 2.203/2.203 source cells và 4.783/4.783 panel cells khớp.
- Live product chỉ cho phép model trong allowlist, thực thi ở subprocess có timeout, bind citation theo dependency DataFrame và replay trước khi trả lời.
- Coding assistant chỉ là công cụ phát triển; không nằm trong dependency/runtime của pipeline demo hay artifact chấm.

## Điều kiện còn phải xác nhận thủ công

| Điều kiện | Trạng thái/việc cần làm |
| --- | --- |
| Giới hạn kích thước model | Đạt theo xác nhận BTC do đội thi cung cấp: tổng tham số không quá 15B; Qwen 14,7–14,8B nằm trong ngưỡng. Lưu ảnh/email xác nhận cùng hồ sơ. |
| Hạn phát hành model | Checkpoint dùng trong thử nghiệm đều trước 01/06/2026; giữ model card/commit hoặc snapshot metadata để chứng minh. |
| Coding assistant khi phát triển | Đạt theo cách áp dụng được BTC xác nhận: coding assistant chỉ hỗ trợ lập trình/review, không tham gia pipeline retrieval/Text-to-Pandas, không sinh kết quả lúc chấm và không là dependency runtime. |
| Quyền dữ liệu | Lưu inventory nguồn, điều khoản BTC và license tương ứng; chỉ dùng kho BTC hoặc nguồn mở/hợp pháp mà đội có quyền tiếp cận. |
| Tài khoản/đội thi/số lượt nộp | Chỉ người dự thi và cổng thi xác nhận được. |
| README, source code và tài liệu nghiệm thu | Đã bổ sung trong repo; cần nộp đúng kênh/hạn BTC yêu cầu. |
| Công bố điểm và quyền nghiệm thu | Đội chấp nhận quyền công bố điểm, đánh giá định lượng/định tính và xác minh của BTC theo điều khoản; không tuyên bố public score là kết quả chung cuộc. |
| Quyền sử dụng bài thắng giải | Nếu đoạt giải, chuẩn bị bàn giao đúng source/data/artifact mà điều khoản cho phép BTC sử dụng; không đưa bí mật/token của bên thứ ba vào gói bàn giao. |

## Lệnh audit chuẩn

```powershell
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v192_data_derived_table_order `
  --archive sub_top123_candidate_v192_data_derived_table_order.zip `
  --expected-sha256 D5D16C202863022670149AAB6C3BF13BB3F77D1E94C8546C20789346E5539E27 `
  --require-data-derived-labels `
  --allow-relevant-table-order `
  --product-python C:\Users\vinh\AppData\Local\Programs\Python\Python314\python.exe
```

Chỉ dùng artifact khi lệnh trả `FINAL RELEASE GATE: PASS`. Vector điểm v192 chỉ gắn với đúng ZIP SHA-256 đã khóa và submission ID 3552.
