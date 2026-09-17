# Hồ sơ sản phẩm KINGPRO R2AI Stage 2

> Cập nhật 28/08/2026: champion public và replay mặc định là V297 / ID3747
> (Execution/Answer `0.7115`, Tables F2 `0.6120`, Docs F2 `0.9618`). V297
> dùng artifact `sub_v297_scope2_a.zip` (SHA-256 `90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC`) với full gate 522 tests + 25 subtests.
> V297 inherits the q224 physical union and q966 consolidated-source repair. V290 là rollback trực tiếp; V276 là rollback thứ hai và exact-10 public-metric tie reference; V269
> là measured rollback fallback thứ ba/source-clean; v206/v207 vẫn byte-locked. Phần v192 phía dưới được giữ
> để truy lịch sử; trạng thái vận hành hiện hành nằm trong
> [HANOI_DEMO_HANDOFF.md](HANOI_DEMO_HANDOFF.md).

## 1. Mục tiêu

KINGPRO là trợ lý truy vấn số liệu báo cáo tài chính tiếng Việt. Đầu vào là câu hỏi tự nhiên; đầu ra là tài liệu/bảng nguồn, các CSV evidence, chương trình Pandas và kết quả số. Thiết kế ưu tiên khả năng kiểm tra và tái thực thi hơn câu trả lời sinh tự do.

## 2. Người dùng và tình huống sử dụng

- Người phân tích cần tra một chỉ tiêu theo công ty, năm và loại báo cáo.
- Người dùng cần tính tăng trưởng, tỷ lệ hoặc so sánh nhiều doanh nghiệp.
- Ban giám khảo cần chạy lại chương trình trên đúng evidence và truy vết về BCTC OCR.

Sản phẩm không phải tư vấn đầu tư và không thay thế việc đối chiếu báo cáo gốc trong quyết định có rủi ro cao.

## 3. Thành phần

1. **Corpus builder:** tách bảng HTML nội dòng từ OCR thành CSV và giữ `report_id|line`.
2. **Retriever:** thu hẹp theo mã chứng khoán, năm, phạm vi báo cáo rồi xếp hạng nội dung bảng.
3. **Answering pipeline:** tạo hoặc chuẩn hoá Pandas, kiểm schema bằng AST, chạy sandbox và tự sửa lỗi trong giai đoạn phát triển.
4. **Deterministic financial layer:** xử lý số Việt Nam, đơn vị, lookup, ratio và panel metrics.
5. **Submission builder:** xuất schema cuộc thi với `answer`, `evidence`, `pandas_query`, `relevant_docs`, `relevant_tables`.
6. **Compliance gate:** từ chối query không phụ thuộc dữ liệu, gán literal trực tiếp, thiếu evidence hoặc khai provenance lệch evidence.
7. **Live product service:** exact-match hoặc paraphrase duy nhất vượt các cổng bảo thủ được replay từ verified registry. Câu một công ty–một năm thuộc grammar chỉ tiêu chuẩn và một topology argmax/argmin panel hẹp có thể đi qua grounded compiler: đọc tọa độ primary-statement đã chuẩn hoá, dựng Pandas, chạy sandbox và đối chiếu độc lập với cube trước khi trả citation. Câu còn lại đi qua domain-boundary, truy hồi phân rã và model mở. Mọi mode được công khai; compiler không giả làm LLM hay registry hit.
8. **Evidence-first UI:** Next.js 16/React 19 với Answer/Pandas/Sources, evidence inspector, confidence dial, citation switcher, audit-trace export, trạng thái bốn cổng xác minh và refusal card thay vì đoán. Data room đọc catalog runtime 100 doanh nghiệp/146.246 bảng; Audit trails lưu và xuất trace của phiên. Judge view ánh xạ năm tiêu chí chấm sang bằng chứng tương tác, công khai allowlist/runtime và có năm ca demo gồm câu mới không phụ thuộc endpoint. Mọi demo gọi pipeline thật, không hiển thị số liệu mock.

## 4. Hợp đồng đầu ra

Mỗi phần tử `submission.json` có:

- `id`: định danh câu hỏi;
- `answer`: kết quả số đã lưu để đối chiếu;
- `relevant_docs`: danh sách report ID nguồn;
- `relevant_tables`: danh sách `report_id|line` nguồn;
- `evidence`: ánh xạ DataFrame tới CSV trong ZIP;
- `pandas_query`: chương trình gán kết quả cuối vào `result`.

Ở candidate hiện tại, provenance được suy ra từ các evidence position mà AST xác định là thực sự được query đọc. Với manifest thu gọn, mỗi dòng vẫn giữ `source_table`, `source_csv`, tọa độ ô và giá trị thô để truy ngược.

## 5. Chỉ tiêu nghiệm thu hiện tại

Artifact đã đo tốt nhất là v192 / ID 3552. BTC đo Execution `0.6877`, Answer
`0.6897`, Tables F2-macro `0.6014`, Docs F2-macro `0.9611`; đây là selected
submission và hạng 1 public tại lần kiểm tra gần nhất. ZIP v192 được khóa SHA-256
`D5D16C202863022670149AAB6C3BF13BB3F77D1E94C8546C20789346E5539E27`.
Verified replay registry phục vụ demo mặc định là v192; audit kỹ thuật hiện tại
biên dịch đúng 106/106 câu trong grammar bảo thủ, từ chối 29/29 ca đối nghịch và
chạy 7/7 ca HTTP smoke trong `4.206 ms` tổng thời gian đo bởi readiness report.

v192 kế thừa sửa EPS có nguồn của v191 và chỉ đổi thứ tự các bảng đã có ở 81
câu panel theo ticker/năm được chính query chọn; không thêm hoặc bỏ bảng, không
đổi query/evidence/document ngoài q511. v192 đạt
1.012/1.012 ở bốn runtime, 2.203/2.203 source cells, 4.783/4.783 panel cells,
166 tests + 25 subtests. BTC đo nó tăng một câu Execution/Answer so với v190 và
tăng Tables F2/Precision/Recall/MRR@5 lần lượt lên `0.6014/0.5808/0.6159/0.6384`.
Judge View đọc readiness report thật và hiển thị tách biệt technical, smoke,
dynamic và manual readiness; model chưa attested vẫn phải hiện `LOCKED`.
Product retrieval dùng parser scope hiển ngôn/mixed-year, alias punctuation không
chồng lên tên pháp lý dài hơn, unique preferred/scope-neutral backfill
fail-closed và ba dependency năm bảo thủ cho đầu kỳ, số dư bình quân và growth
scan theo giai đoạn. Document macro F2 source-bound tăng `0,907888 -> 0,983827`,
precision `0,935530 -> 0,979938`, recall `0,912056 -> 0,988322`, miss `182 -> 29`.
CSV-label reranker nâng recall@8 `0,471960 -> 0,552890` và MRR@40
`0,438546 -> 0,532769`. Đây là regression nội bộ trên nguồn thực thi, không phải
metric gold ẩn.

| Kiểm tra | Kết quả |
| --- | ---: |
| Tổng câu | 1.012 |
| Query có nội dung | 1.012 |
| Query chạy không exception, string runtime | 1.012 |
| Query chạy không exception, typed runtime | 1.012 |
| Query trả kết quả số và chạy lại khớp `answer` | 1.012 |
| Source cells khớp dữ liệu gốc | 2.203/2.203 trên candidate local v192 |
| Source cells khớp dữ liệu gốc | 2.145/2.145 trên candidate nghiên cứu v166 |
| Source cells khớp dữ liệu gốc | 2.149/2.149 trên candidate nghiên cứu v167 |
| Source cells khớp dữ liệu gốc | 2.150/2.150 trên candidate nghiên cứu v168 |
| Source cells khớp dữ liệu gốc | 2.150/2.150 trên candidate nghiên cứu v170 |
| Panel source cells khớp dữ liệu gốc | 4.780/4.780 trên ablation truy hồi v171 |
| Source cells khớp dữ liệu gốc | 2.148/2.148 trên candidate v172; panel 4.780/4.780 |
| Source cells khớp dữ liệu gốc | 2.149/2.149 trên candidate v173; panel 4.780/4.780 |
| Source cells khớp dữ liệu gốc | 2.151/2.151 trên candidate v174; panel 4.780/4.780 |
| Source cells khớp dữ liệu gốc | 2.153/2.153 trên candidate v175; panel 4.780/4.780 |
| Source cells khớp dữ liệu gốc | 2.154/2.154 trên candidate v176; panel 4.780/4.780 |
| Source cells khớp dữ liệu gốc | 2.155/2.155 trên candidate v177; panel 4.780/4.780 |
| Source cells khớp dữ liệu gốc | 2.157/2.157 trên candidate v178; panel 4.780/4.780 |
| Source/panel cells khớp dữ liệu gốc | v192: 2.203/2.203 source; 4.783/4.783 panel |
| Direct numeric assignment vào `result` | 0 |
| Lỗi provenance/evidence từ compliance checker | 0 |
| Registry exact-match được nạp và replay theo yêu cầu | 1.012 entries |
| Registry rows đủ facet/term để xét paraphrase bảo thủ | 1.009 entries |
| Registry paraphrase gates | entity/year/scope/unit/intent/time + score/precision/recall/margin/shared terms |
| Nhất quán câu hỏi v178 | 0/29 disagreement cùng đơn vị; 0/39 sau quy đổi tiền tệ |
| Mixed-condition unit parser | lấy explicit output unit cuối câu; không để ngưỡng `%` lấn đơn vị tiền |
| Direct one-cell unit mismatches trên header có ĐVT rõ | 0/22 |
| Lighthouse desktop/mobile | 100/100 Accessibility, Best Practices, SEO, Agentic Browsing |
| Judge View HTTP smoke | 7/7 ca qua frontend proxy: ba ca sân khấu exact, paraphrase, grounded compiler, refusal thiếu thực thể và prompt injection |
| Open-model control plane | Cấu hình endpoint/template đúng Qwen2.5-Coder-14B và image tag bất biến; operational audit đang fail vì pod inventory chỉ có worker `EXITED`, native/OpenAI request kẹt queue; inference vẫn khóa |
| Refusal calibration nội bộ | 32 positive + 192 structured negatives; tại 0,90: accept 90,6%, reject 100% trên suite hữu hạn |

Các con số gate là kiểm tra nội bộ về khả năng thực thi, provenance và sản phẩm; không được diễn giải thành điểm gold ẩn. Riêng vector điểm v192 nêu trên là kết quả public do BTC đo, không phải điểm private hoặc kết quả chung cuộc.

## 6. Bàn giao

- Candidate bảo toàn: `sub_top123_candidate_v161_legacy_source43/` và ZIP cùng tên.
- SHA-256 ZIP v161: `1D4917D7D4C5BA27C2303DC661EF0AD7D6F38AECF140C3F8DFADC19F3705D4AB`.
- Champion product/runtime: `sub_top123_candidate_v192_data_derived_table_order/`; SHA-256 ZIP `D5D16C202863022670149AAB6C3BF13BB3F77D1E94C8546C20789346E5539E27`; BTC public ID 3552, selected submission.
- Measured rollback: `sub_top123_candidate_v190_pdr_growth_count/`; SHA-256 ZIP `B474FAA474850A490E0574AE52DBE76134194048ED6C8BC692D89919009BDD32`; BTC public ID 3545.
- Stable fallback: `sub_top123_candidate_v165_hag_long_receivables/`; SHA-256 ZIP `01DD51D4AA0E4B42A822E1F18493027FF6AD5678F4478CCF17C810B6EE6DB23C`.
- Candidate nghiên cứu q81: `sub_top123_candidate_v166_mbb_afs_debt/`; SHA-256 ZIP `C879ACA3147467D47742ACB5F366E6D2A9B60A9DC9195EB303D897388F09C433`; chưa nộp, chưa thay registry mặc định v165.
- Candidate nghiên cứu q81+q937: `sub_top123_candidate_v167_ctg_afs_debt_mean/`; SHA-256 ZIP `8D807C0DC36BA55707E05E5E6DB5551C4D9D9F4352066DE206AC24BAC9C499A1`; chưa nộp, chưa thay registry mặc định v165.
- Candidate nghiên cứu q35+q81+q937: `sub_top123_candidate_v168_bvh_receivable_summary/`; SHA-256 ZIP `1269C031B715377854DC33D0CEA12CAA738E32F97DD8D45A0F9FF3EA66B7ADA0`; chưa nộp, chưa thay registry mặc định v165.
- Candidate nghiên cứu source-hardened q35+q81+q797+q937: `sub_top123_candidate_v170_employee_direction_hardened/`; SHA-256 ZIP `855065C4A7E18509D518A71BB8AAE9104344BBAC8994BE1C178D72346D3E384F`; chưa nộp, chưa đo, chưa thay registry mặc định v165.
- Candidate ablation truy hồi panel, giữ nguyên toàn bộ đáp án v170: `sub_top123_candidate_v171_panel_minimal/`; SHA-256 ZIP `16F92E09A3BF0D118E762E584908D858EB9F80F66C7587A7B556A9CB6920C9C9`; chưa nộp, chưa đo, chưa thay registry mặc định v165.
- Candidate sửa đúng năm nguồn q336 trên v171: `sub_top123_candidate_v172_hnd_lease_2025/`; SHA-256 ZIP `5E1657BB0AD2752C9851CF384DE0F42E8CDAA64C6D305FB773873EAE0A3494C7`; chưa nộp, chưa đo, chưa thay registry mặc định v165.
- Candidate sửa đúng nhóm vay ngắn hạn AAA q354: `sub_top123_candidate_v173_aaa_short_term_bank_loan/`; SHA-256 ZIP `E4BD93CE0477D5C392E4627262635A651FE679FE59E165F675F3D94A606947C7`; chưa nộp, chưa đo.
- Candidate sửa đúng dòng chi phí nhân viên KHG q646: `sub_top123_candidate_v174_khg_employee_expense/`; SHA-256 ZIP `29CF55B6DE32FF61B18CE6E546673A39F13BD233E87B89AD8325F84107EE3F74`; chưa nộp, chưa đo.
- Candidate tổng hợp đúng trái phiếu Chính phủ STB q169: `sub_top123_candidate_v175_stb_government_bonds/`; SHA-256 ZIP `082D41694B245C14EE024EF9AA791C67A2C5CFBBABEA02A79806BEA6DEA1F272`; chưa nộp, chưa đo.
- Candidate sửa số dư cuối năm lãi vay phải trả DLG q26: `sub_top123_candidate_v176_dlg_interest_payable/`; SHA-256 ZIP `2CF4BE2BE349C00F449310568D496CFA09F4090B7088EE67E43A3702E11A0104`; chưa nộp, chưa đo.
- Candidate mới nhất sửa tổng XDCB dở dang dài hạn hợp nhất VIC q244: `sub_top123_candidate_v177_vic_long_term_cip/`; SHA-256 ZIP `A8713B9B8528AC001D69CD6CE2108FFEF448A2EAAED9FC03E1C09B767F35112D`; chưa nộp, chưa đo, chưa thay registry mặc định v165.
- Candidate mới nhất sửa mẫu số tổng tài sản cuối năm NVB q717: `sub_top123_candidate_v178_nvb_off_balance_assets/`; SHA-256 ZIP `3514C00115E3665FB5A2F74C8F244E44762311468720C29A0C05F84E780871CD`; chưa nộp, chưa đo, chưa thay registry mặc định v165.
- Candidate mới nhất sửa tổng dự phòng rủi ro cho vay khách hàng CTG q356 trên v182: `sub_top123_candidate_v183_ctg_total_loan_provision/`; SHA-256 ZIP `D96119B2E51EF71196A98CEA7CC44A2C32A6DC04634679A2F02BA5DEF6AC40DD`; chỉ q356 đổi `6.071.288 -> 12.788.628` triệu đồng và thêm bảng cân đối `|264` đã đối soát với Note 11 `|1212`; chưa nộp, chưa đo, chưa thay registry mặc định v165. v182 là rollback retrieval trực tiếp.
- Giao diện: `frontend/`; API sản phẩm: `src/kingpro/product/` và `scripts/serve_product.py`.
- Lệnh tái lập: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
- Đối chiếu luật: [COMPLIANCE_CHECKLIST.md](COMPLIANCE_CHECKLIST.md).
- Kịch bản chứng minh tuân thủ khi demo: [DEMO_DAY_COMPLIANCE.md](DEMO_DAY_COMPLIANCE.md).
- Review thủ công sáu cảnh báo source-scope mức medium: [LEGACY_SOURCE_SCOPE_REVIEW.md](LEGACY_SOURCE_SCOPE_REVIEW.md).
