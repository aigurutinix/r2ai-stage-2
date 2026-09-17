# KINGPRO — R2AI 2026 Stage 2

Hệ thống truy hồi bảng tài chính và sinh truy vấn Pandas có dẫn nguồn cho bộ dữ liệu ViFinQA. Mục tiêu của hệ thống là biến câu hỏi tiếng Việt thành một chương trình có thể kiểm tra được: đọc đúng CSV evidence, tính kết quả ở thời gian chạy và gắn lại đúng tài liệu/bảng nguồn.

## Trạng thái hiện tại

> Cập nhật Hà Nội ngày 28/08/2026: champion public là **V297 / ID 3747**
> (`Execution 0.7115`, `Answer 0.7115`, `Tables F2 0.6120`,
> `Docs F2 0.9618`, `on_leaderboard=true`). Product replay mặc định là V297;
> artifact `sub_v297_scope2_a.zip` (SHA-256 `90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC`) là bản được chọn;
> full gate 522 tests + 25 subtests. V297 inherits the q224 physical union and q966 consolidated-source repair. V290 là rollback trực tiếp, V276 là rollback thứ hai và exact-10 public-metric tie reference; V269 là rollback thứ ba/source-clean, v206/v207 được bảo toàn byte/hash.
> full release gate PASS. Không gọi public score là private/final. Xem
> [HANOI_DEMO_HANDOFF.md](docs/HANOI_DEMO_HANDOFF.md) để lấy trạng thái và lệnh
> hiện hành; bảng dài phía dưới là snapshot lịch sử.

| Hạng mục | Trạng thái ngày 25/08/2026 |
| --- | --- |
| Bản selected hiện tại | `V297`, ID `3747`: Execution/Answer `0.7115/0.7115`, Tables F2 `0.6120`, Docs F2 `0.9618`; artifact `sub_v297_scope2_a.zip` (SHA-256 `90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC`), full gate 522 tests + 25 subtests |
| Kết quả sửa nguồn q1007 | `v197` thay nhầm tổng phân khúc địa lý 2021/2022 bằng cột `Dự án BOT`, đáp án `94,71 -> 92,13%`; BTC đo không đổi metric nào so với v192. Full gate PASS 181 test + 25 subtest; ZIP SHA-256 `7D56579F1FB8D1F44CFCF2893141E4D2B398B72E31637192CDBD59DB03520FC0` |
| Product/runtime mặc định | `sub_top123_candidate_v192_data_derived_table_order/`; fallback lần lượt về v190, v184, v165 rồi v161 nếu artifact mới hơn không tồn tại |
| Artifact retrieval đối chứng | `v185`, ID `3493`: toàn bộ vector điểm bằng v184; thêm 11 bảng tương đương không tạo thay đổi đo được |
| Candidate runtime-safe chưa đo | `sub_top123_candidate_v187_warning_clean/`; so với v184 chỉ sửa code q369/q170/q222, giữ nguyên mọi answer/retrieval/evidence; 1.012/1.012 cả hai mode dưới `-W error`; full release gate PASS; ZIP SHA-256 `9A2443DFA6449246FDB36E317A9E388C7895CAD3773D02FEADA6E65FACB93121`; chưa được phép gọi là cải thiện điểm |
| Artifact Demo Day/compliance đã đo | `v188`, ID `3539`, Execution `0.6838`; kế thừa v187 và thay 213 nhãn năm literal trong 53 query bằng năm đọc từ DataFrame nguồn; answer/retrieval/evidence giữ nguyên; data-derived-label audit 0 lỗi và full release gate PASS; ZIP SHA-256 `1DD5EABD37ABE2931CC78A548330FBB7C545735D2912C36C449B266238F568B8` |
| Sửa EPS đã được đo trong v192 | q511 bổ sung đủ EPS DPM/HT1/HPG `1.551/1.681/4.037`, chọn HPG và tính NPAT/VCSH `21,17%`; cùng table-order change đưa Execution/Answer tăng đúng một câu so với v190 |
| Retrieval-order đã được BTC đo | v192 chỉ đổi thứ tự `relevant_tables` ở 81/1.012 câu panel, giữ nguyên membership. BTC đo Tables F2 `0.6011 -> 0.6014`, Precision `0.5806 -> 0.5808`, Recall `0.6155 -> 0.6159`, MRR@5 `0.6361 -> 0.6384`; ZIP SHA-256 `D5D16C202863022670149AAB6C3BF13BB3F77D1E94C8546C20789346E5539E27` |
| Số câu | 1.012 |
| Query có thể thực thi | 1.012/1.012 trong runtime string và typed |
| Release gate | v192 artifact gate: 1.012/1.012 ở bốn runtime; 2.203/2.203 source cells; 4.783/4.783 panel cells; 166/166 tests + 25 subtests; Python 3.7, provenance-membership, period, intent, metric, archive, data-flow và data-derived-label gates đều PASS |
| Khóa artifact đã được BTC đo | v192 ZIP SHA-256 `D5D16C202863022670149AAB6C3BF13BB3F77D1E94C8546C20789346E5539E27`; rollback v190 `B474FAA474850A490E0574AE52DBE76134194048ED6C8BC692D89919009BDD32`; rollback sâu v184 `3C025FD1331540C9C8E30C9FB29EDF9A12B58B40AE62A7B5CCBB7CA359C0BE04` |
| Product API | Có `/health`, `/catalog`, `/ask`, domain-boundary + refusal gate, citation binding, 1.012-entry verified replay registry V297, grounded compiler source-bound đạt 106/106 và CSV-label table reranker. Parser scope hiển ngôn/mixed-year, alias non-overlap, scope-neutral fail-closed và các dependency năm tài chính bảo thủ tăng document F2 nội bộ `0,907888 -> 0,983827`, precision `0,935530 -> 0,979938`, recall `0,912056 -> 0,988322`, miss `182 -> 29`; trên cùng document stage, table recovery@8 sau rerank tăng `0,471960 -> 0,552890`. Đây là local source-bound regression, không phải BTC-gold claim |
| Demo readiness live | `/health` đọc báo cáo readiness đã audit và chỉ xuất summary không nhạy cảm: hiện `12/12` technical, `7/7` HTTP smoke, `0/8` hồ sơ thủ công, `stage_safe_now=true`, `full_demo_ready_now=false`; model động vẫn hiển thị `LOCKED`, không đánh tráo cấu hình allowlist thành serving attestation |
| Runtime model | RunPod configuration PASS: endpoint/template cùng cấu hình Qwen2.5-Coder-14B và image tag bất biến; operational audit hiện FAIL vì inventory có 5 worker `EXITED`, job native/OpenAI kẹt queue dù health stale báo ready; dynamic generation vẫn fail-closed |

Các đoạn tiếp theo là lịch sử phát triển theo thời điểm tạo candidate; trạng thái hiện hành luôn lấy từ bảng trên và [submissions-log.md](docs/submissions-log.md).

## Bộ nhớ thí nghiệm Vô Thượng

Repo có kênh ghi append-only tại `knowledge/vothuong/`: `experiments.jsonl`
lưu sự kiện máy đọc được, còn `playbook.md` lưu bài học bền. Cơ chế này được
chuyển từ tool Vô Thượng toàn máy sang bản repo-local, giữ cổng chống trùng,
delta `supersedes` và phân biệt oracle thật với dev/offline thiên lệch.

`answer_forensics.py` tự ghi mỗi lần quét; `experiment.py` tự ghi package bị
chặn, package thành công và điểm leaderboard. Không có lệnh nào tự upload bài.
`leaderboard_scores.py` đọc toàn bộ lịch sử điểm qua chính API của trang trong
tab Chrome đã đăng nhập; script chỉ GET và không đọc/in/lưu cookie.

```powershell
# Ghi bài học bền
python scripts\vothuong_log.py add "Raw source đã là phần trăm thì không ép phép chia" --conf cao --tags source,percent

# Ghi review một hoặc nhiều câu
python scripts\vothuong_log.py review --question-ids 97,19 --verdict source_confirmed --summary "Nguồn chứa trực tiếp tỷ lệ phần trăm"

# Chạy bất kỳ gate nào và tự ghi exit code/thời gian
python scripts\vothuong_log.py run --name forensic-v197 -- python scripts\answer_forensics.py sub_top123_candidate_v196_effective_tax_sign

# Xem lịch sử gần nhất hoặc riêng một câu
python scripts\vothuong_log.py recent 20
python scripts\vothuong_log.py recent 20 --question-id 97

# Đọc toàn bộ điểm API (hoặc chỉ 10 bài mới nhất)
python scripts\leaderboard_scores.py
python scripts\leaderboard_scores.py --limit 10
```

Oracle `leaderboard`, `source`, `measurement`/`do-luong` được gắn `that`;
`dev`, `offline`, `val`, `heuristic` bị gắn `thien-lech`. Tool tự che các trường
key/token/password/secret trước khi ghi, nhưng vẫn không được truyền credential
vào ghi chú hay dòng lệnh.

Artifact v161:

- dựng `relevant_docs` và `relevant_tables` từ chính evidence mà query đọc;
- mọi câu đều có Pandas query tái lập từ CSV nguồn;
- có cổng compliance, source-coordinate, entity, intent, period và metric-code;
- khác bản đã đo v106 ở 43 đáp án, nhưng chưa được phép suy diễn điểm cho tới khi BTC chấm xong.

Successor v165 giữ nguyên artifact v161 và bổ sung bốn sửa chữa có đối chiếu BCTC: q325 đúng bảng KQKD công ty mẹ TTF, q611 đúng năm nhóm thanh khoản trong hạn EIB, q337 dùng đúng dòng tổng tiền và tương đương tiền BAF, q723 tính đúng tỷ trọng khoản cho vay dài hạn bên liên quan trong tổng phải thu dài hạn từ bên liên quan của HAG. v165 chưa được BTC đo nên không được gọi là tăng điểm đã xác nhận.

Candidate tách biệt v166 chỉ thay q81: câu hỏi MBB 2023 dùng cụm không định tính
“chứng khoán nợ”. Trong test set, q880 dùng cùng cách gọi cho nhóm chứng khoán
sẵn sàng để bán, còn q883 ghi rõ “chứng khoán kinh doanh nợ” khi hỏi nhóm
trading. Vì vậy v166 đọc dòng AFS `Chứng khoán nợ` tại bảng `|1596`, đổi
`4.551.746 -> 143.010.711` triệu đồng. Đây là suy luận từ quy ước của test set,
không phải điểm BTC đã đo; product registry vẫn mặc định v165.

Candidate v167 kế thừa q81 và sửa thêm q937. Chương trình cũ của q937 lấy
bốn số từ bảng biến động dự phòng; chương trình mới lấy dòng AFS `Chứng khoán
nợ` của CTG cho đủ 2017–2020 và tính trung bình `105.632.734,75` triệu đồng.
Detector table-family về 0 findings, nhưng đây vẫn là candidate nghiên cứu chưa
được BTC đo; v166 được giữ làm ablation q81-only.

Candidate v168 kế thừa v167 và thay q35 bằng đúng dòng tổng hợp `Phải thu: -
Bảo Việt Nhân thọ` tại bảng `|1232`, cho kết quả `222.575,01` triệu đồng. Dòng
này trực tiếp hơn cả đáp án cũ chọn một tiểu mục (`208.334,22`) lẫn giả thuyết
v107 cộng hai khoản phải thu gộp (`232.568,96`). v168 chưa được BTC đo và
product registry vẫn mặc định v165.

Candidate v171 không đổi bất kỳ đáp án nào của v170. Chế độ
`--minimal-dependencies` phân tích AST của 175 chương trình panel, chỉ giữ các
metric thô và metric dẫn xuất cần cho từng câu. Có 172 query/table-list được
thu gọn; ba câu dữ liệu tài sản cố định thưa (q429, q432, q506) dùng fallback
nguyên trạng. Tổng số `relevant_tables` trong nhóm panel giảm từ 8.721 xuống
2.971 (-65,93%), median từ 43 xuống 12. Cả runtime string/typed vẫn đạt
1.012/1.012; verifier panel truy ngược 4.780/4.780 ô về CSV BCTC gốc. Đây là
ablation truy hồi chưa được BTC đo, không phải bằng chứng tăng điểm và không
thay product registry v165.

Candidate v172 kế thừa toàn bộ v171 và chỉ sửa q336. Câu hỏi yêu cầu tổng tiền
thuê tối thiểu của HND tại 31/12/2025, nhưng bản cũ đã gắn nhầm số cuối năm của
BCTC 2024 (`378,05` tỷ) thành năm 2025. v172 đọc trực tiếp dòng tổng tại
`HND_financial_statements_2025|912`, raw `387.656.354.540` VND, trả `387,66`
tỷ. Gate structural mới phát hiện tự động trường hợp ngày cuối năm không hiện
diện trong năm của `relevant_docs`; 1.011 câu còn lại giữ nguyên. v172 chưa
được BTC đo và không thay product registry v165.

Candidate v173 kế thừa toàn bộ v172 và chỉ sửa q354. Query cũ lấy dòng
`Các khoản vay ngân hàng bằng VND` trong mục `19.2 Vay ngân hàng dài hạn` tại
`AAA_financial_statements_2021_separate|1097`, nên trả `2,95` trăm tỷ đồng.
Bảng biến động vay ngắn hạn tại `AAA_financial_statements_2021_separate|1058`
ghi trực tiếp dòng `Vay ngân hàng`, số cuối năm `1.401.195.977.583` VND; v173
trả `14,01` trăm tỷ đồng. Chỉ q354 thay đổi, 1.011 câu khác giữ nguyên. v173
chưa được BTC đo và product registry vẫn mặc định v165.

Candidate v174 kế thừa v173 và chỉ sửa q646. Câu hỏi dùng đúng tên dòng
`Chi phí nhân viên`, nhưng query cũ lấy dòng khác là `Chi phí nhân viên quản lý`
tại `|1025`. v174 đọc dòng chính xác trong thuyết minh chi phí bán hàng tại
`KHG_financial_statements_2021_consolidated|1021`: 22.287.552.828 VND năm
2021 và 11.949.173.962 VND năm 2020, nên chênh lệch là `10,34` tỷ thay vì
`13,17` tỷ.

Candidate v175 kế thừa v174 và chỉ sửa q169. Query cũ lấy riêng phần trái phiếu
Chính phủ đem thế chấp tại `|1971` (`9.636.738` triệu đồng), dù câu hỏi không
có điều kiện thế chấp. Note 13 tại `|1313` ghi số đầu năm của hai phân loại AFS
`27.045.792` và HTM `991.387`; theo cùng quy ước tổng mọi phân loại của q888,
v175 trả `28.037.179` triệu đồng. v174/v175 đều chưa được BTC đo; product
registry phục vụ demo vẫn khóa ở v165.

Candidate v176 kế thừa v175 và chỉ sửa q26. Query cũ lấy dòng giao dịch với một
bên liên quan tại `DLG_financial_statements_2023_consolidated|1878`, là số phát
sinh trong năm `26.433.460` VND chứ không phải số dư cuối năm. Thuyết minh 20
`Chi phí phải trả ngắn hạn` tại bảng `|1455` có đúng dòng `Lãi vay phải trả`,
cột `Số cuối năm` là `350.187.565.073` VND; v176 trả `350.187,57` triệu đồng.
Artifact đạt 1.012/1.012 ở cả hai runtime, source 2.154/2.154, panel
4.780/4.780 và 51/51 tests. v176 chưa được BTC đo; product registry phục vụ
demo vẫn khóa ở v165.

Candidate v177 kế thừa v176 và chỉ sửa q244. Query cũ lấy `6.943.642.868.303`
VND từ bảng giá trị hợp lý tạm thời của riêng Vicentra tại ngày mua (`|1295`).
Câu hỏi yêu cầu tổng chi phí xây dựng cơ bản dở dang dài hạn của toàn Tập đoàn
tại 31/12/2016. Bảng cân đối hợp nhất mã 242 tại `|209` ghi trực tiếp số cuối
năm `33.991.567.265.462` VND; v177 trả `339,92` trăm tỷ đồng. Artifact đạt
1.012/1.012 ở cả hai runtime, source 2.155/2.155, panel 4.780/4.780 và 65/65
tests. Audit phạm vi nguồn kiểm tra 222 legacy rows, truy được 222 báo cáo gốc,
không còn high-risk finding; sáu cảnh báo medium đã được đối chiếu thủ công.
Audit nhất quán chéo không tìm thấy mâu thuẫn ở 29 cặp cùng entity/năm/scope/
unit/intent/time; chế độ quy đổi tiền tệ cũng sạch trên 39 cặp. Product parser
đã sửa để lấy đơn vị đầu ra cuối cùng trong câu hỏi hỗn hợp điều kiện `%` và
đơn vị tiền, ngăn replay paraphrase giữa `tỷ` và `nghìn tỷ`.
v177 chưa được BTC đo; product registry phục vụ demo vẫn khóa ở v165.

Candidate v178 kế thừa v177 và chỉ sửa q717. Tử số cam kết ngoại bảng
`12.053.691` triệu đồng tại `NVB_financial_statements_2019_separate|1423`
đã đúng, nhưng query cũ chia cho `73.007.228` triệu đồng từ bảng phân khúc có
ngữ cảnh `01/01/2019` tại `|1571`, tức số mở đầu kỳ. Bảng cân đối công ty mẹ
tại `|268` ghi `TỔNG TÀI SẢN CÓ` cuối 2019 là `80.405.111` triệu đồng; v178
trả `14,99%` thay vì `16,51%`. Rule source-scope mới phát hiện đúng q717 là
high-risk duy nhất trên v177 và về 0 trên v178. Artifact đạt 1.012/1.012 ở cả
hai runtime, source 2.157/2.157, panel 4.780/4.780 và 70/70 tests. v178 chưa
được BTC đo; product registry phục vụ demo vẫn khóa ở v165.

## Kiến trúc

```text
ViFinQA OCR
  -> trích bảng HTML thành CSV, giữ report_id|line
  -> truy hồi theo mã CK, năm, scope và nội dung bảng
  -> sinh/chuẩn hoá chương trình Pandas
  -> kiểm AST và thực thi trong sandbox pandas 1.1.5
  -> answer + evidence + relevant_docs/relevant_tables
  -> compliance gate -> ZIP nộp bài

Câu hỏi live
  -> kiểm đủ công ty + năm
  -> nếu khớp chính xác registry đã audit: chạy lại Pandas trên evidence rồi trả citation
  -> nếu là paraphrase duy nhất vượt năm cổng bảo thủ: replay cùng chương trình và công khai match score
  -> nếu không khớp: tiếp tục retrieval decomposition theo mọi cặp công ty–năm
  -> sinh Pandas bằng model mở trong allowlist
  -> subprocess timeout + safety AST
  -> chạy lại kết quả + bind citation theo DataFrame thực sự được đọc
  -> trả grounded=true hoặc từ chối có lý do
```

Các query trong ZIP không gọi mạng hay mô hình. Mô hình chỉ tham gia ở giai đoạn phát triển/sinh candidate; kết quả cuối được tính lại từ CSV bằng Pandas trong môi trường chấm.

## Cấu trúc repo

```text
src/kingpro/                 mã nguồn trích bảng, retrieval, answering, sandbox
src/kingpro/product/         service live, refusal policy và citation contract
frontend/                    Next.js 16 evidence-first financial cockpit
scripts/                     build, audit, grader check và đóng gói
tests/                       unit test sandbox, provenance và product gate
data/                        ViFinQA OCR và câu hỏi
build/                       catalog/bảng trung gian
sub_compliant_all/           candidate đầu vào trước pass tuân thủ
sub_compliance_safe/         candidate tuân thủ đã dựng lại
docs/                        hồ sơ sản phẩm, dữ liệu, mô hình và tái lập
```

## Cài đặt

Phần xử lý dữ liệu và kiểm tra artifact:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONPATH = "$PWD\src"
```

Phần sinh bằng LLM là tùy chọn, phù hợp nhất trên Linux/CUDA:

```bash
python -m pip install -r requirements-llm.txt
```

Không commit `.env`, token Hugging Face hoặc khóa endpoint. Các biến endpoint/model chỉ cần thiết khi chạy inference, không cần cho checker hay grader offline.

Nếu cần inference, copy `.env.example` thành `.env`, thay endpoint/key cục bộ rồi nạp các biến đó vào shell. File mẫu không chứa secret.

## Chạy sản phẩm live

Nạp `.env` theo `.env.example`, phục vụ Qwen bằng endpoint vLLM tương thích OpenAI rồi chạy:

```powershell
$env:PYTHONPATH = "$PWD\src"
python scripts\serve_product.py
```

Kiểm tra:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/ask -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body '{"question":"Doanh thu thuần về bán hàng và cung cấp dịch vụ của GEG trong năm 2025 là bao nhiêu tỷ đồng?"}'
```

Có thể chạy một câu không qua HTTP bằng `python scripts/ask_product.py "<câu hỏi>"`. Exact-match registry chỉ chuẩn hoá dấu câu/khoảng trắng. Một paraphrase chỉ được replay khi tập công ty, năm, scope, đơn vị, intent và time-basis tương thích; đồng thời vượt score `0.78`, candidate recall `0.75`, query precision `0.65`, tối thiểu bốn từ tài chính chung và margin `0.12` với ứng viên kế tiếp. Mode trả về là `verified_registry_paraphrase`, không giả làm exact hit. Mỗi hit vẫn phải chạy lại query và khớp answer trước khi trả. Mọi câu không vượt đủ cổng tiếp tục đi qua pipeline model mở hoặc bị từ chối. Rule domain-boundary chạy trước model và giải thích khi câu hỏi cần dữ liệu ngoài kho BCTC như giá thị trường, xếp hạng ESG ngoài, analytics ứng dụng, khảo sát nhân viên hoặc dữ liệu thương hiệu. Runtime cũng từ chối nếu thiếu công ty/năm, thiếu nguồn, retrieval dưới ngưỡng, code không an toàn, không có citation thật hoặc chạy lại không khớp.

Chạy batch private-safe với checkpoint nguyên tử, ordered concurrency, trace và
run manifest:

```powershell
python scripts/run_product_batch.py data/questions/questions.jsonl `
  --workers 4 --expected-count 1012
```

Nếu bị ngắt, chạy tiếp bằng `--resume outputs/product_batch/<run_id>`. Fallback
v297 chỉ được dùng sau lỗi và chỉ khi cả ID lẫn nguyên văn câu hỏi khớp; không
fuzzy-fallback câu private mới. Chi tiết xem
[`docs/MSCAI_OPERATIONAL_HARDENING_20260828.md`](docs/MSCAI_OPERATIONAL_HARDENING_20260828.md).
Ma trận còn thiếu để đạt end-to-end hoàn chỉnh xem tại
[`docs/MSCAI_END_TO_END_GAP_MATRIX_20260828.md`](docs/MSCAI_END_TO_END_GAP_MATRIX_20260828.md).

### Chạy toàn bộ bằng Docker

Docker stack production dùng cùng backend/frontend ở trên, giữ source data và
v297 replay read-only, còn checkpoint/trace/submission ZIP được ghi bền vững về
`outputs/` trên host:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/docker_product.ps1 up -Rebuild
```

Sau khi hai healthcheck PASS, mở `http://127.0.0.1:3000`. Backend entrypoint sẽ
fail-fast nếu thiếu catalog/BM25/tables hoặc SHA-256 của v297 replay không khớp.
Chi tiết vận hành, model endpoint tùy chọn và lệnh doctor/down/logs nằm tại
[`docs/DOCKER_DEPLOYMENT.md`](docs/DOCKER_DEPLOYMENT.md).

Security/governance product hiện có authority contract, audited LLM gateway,
PII/credential egress guard, token/OIDC RBAC, tenant-scoped batch/upload/feedback,
explicit write approval, upload quarantine, AES-GCM tenant storage, lifecycle
monitor, recoverable retention, backup/restore drill, rolling metrics và
fail-closed OS-sandbox attestation. Contract và giới hạn provider còn thiếu nằm
tại [`docs/SECURITY_AND_GOVERNANCE.md`](docs/SECURITY_AND_GOVERNANCE.md).

Trong nhánh retrieval, BM25 lấy pool 40 bảng rồi bộ reranker đọc trực tiếp CSV để
nhận diện cột nhãn thật. Cách này sửa lỗi OCR phổ biến khi cột đầu chỉ là `Mã số`
còn tên chỉ tiêu nằm ở cột kế tiếp. A/B trên cùng 1.012 source bindings đã audit
cho pipeline recall@8 `0,471960 -> 0,552890` và MRR@40 `0,438546 -> 0,532769`,
không bucket lookup/analytic/ratio/multi-table nào giảm. Đây là regression sản
phẩm theo nguồn thực thi, không phải F2 gold ẩn hoặc dự báo điểm leaderboard.

Pipeline tạo submission cũng ưu tiên cùng grounded compiler trước model cho grammar
một công ty–một năm và topology argmax/argmin panel bảo thủ. Local audit hiện nhận
106/1.012 câu và khớp 106/106; adversarial gate từ chối 29/29 biến thể sai time-basis,
scope, aggregation, unit hoặc business meaning; 32 câu được mở rộng an toàn sau mốc compiler 67; đây không phải
hidden-set hay private-score claim. Dùng
`scripts/build_full_submission.py --compiler-only` chỉ
để pilot/CI vì chế độ này fail toàn lượt khi gặp câu ngoài grammar; build private
đầy đủ phải giữ model mở làm fallback. Script mặc định không ghi đè `--out` hoặc
ZIP đã tồn tại và đóng ZIP deterministic. Xem mục 28 trong
[Reproducibility](docs/REPRODUCIBILITY.md).

## Chạy giao diện demo

Frontend dùng Next.js 16 + React 19 và gọi backend qua route proxy nội bộ, vì vậy khóa endpoint không bị đưa xuống trình duyệt. Mở terminal thứ hai:

```powershell
cd frontend
npm install
npm run dev -- --hostname 127.0.0.1 --port 3010
```

Mở `http://127.0.0.1:3010`. Trước Demo Day nên kiểm tra bản production:

```powershell
cd frontend
npm run lint
npm run build
npm start -- --hostname 127.0.0.1 --port 3010
```

Dừng tiến trình `next start` đang nghe cổng 3010 trước khi chạy lại `next build`, rồi
khởi động lại sau khi build hoàn tất. Không ghi đè thư mục `.next` khi production server
đang phục vụ vì browser có thể giữ HTML cũ nhưng yêu cầu chunk mới và nhận lỗi 500.

Giao diện chính là workspace ChatGPT-style tối giản: `Chat/Batch`, lịch sử trong
phiên, measured backend trace live qua SSE có thể co/mở, code Pandas dạng disclosure và evidence
panel cố định bên phải. Evidence panel chỉ hiện nguồn đang chọn, metadata vật lý và
bảng CSV thật; có thể thu thành rail 52px. Motion được giới hạn ở những thay đổi trạng
thái có ý nghĩa: Chat/Batch trượt tab, trace co/mở, evidence panel trượt, card hover/
press và composer focus. Không dùng dữ liệu mock. Mỗi kết quả vẫn công khai mode
`verified_registry`, `verified_registry_paraphrase`, `deterministic_compiler` hoặc
`generated`; nếu thiếu bằng chứng thì hiển thị refusal. Nếu backend chạy ở địa chỉ
khác, đặt `KINGPRO_API_URL` cho tiến trình Next.js.

Với registry V297, runtime nạp manifest/source audit nhưng chỉ công khai cell lineage
khi tọa độ vật lý và raw token được đọc lại từ `build/tables` và khớp tuyệt đối. Lớp
audit gốc tái chứng minh 7.041 ô trên 817 câu. Với 195 chương trình legacy còn lại,
counterfactual analyzer perturb cell rồi replay đúng Pandas, chỉ nhận ô làm kết quả
thay đổi và bind được duy nhất vào một bảng relevant. Lớp này bổ sung 258 ô/195 câu,
bao gồm text-label dependency của các phép đếm tồn tại. Tổng runtime hiện có
**7.299 ô trên 1.012/1.012 câu (100%)**, không có ô nằm ngoài `relevant_tables`.
Modal gộp header nhiều tầng thành nhãn kiểu Excel (`Trong hạn · Tổng cộng`), tô riêng
đúng ô/hàng đã dùng và tự cuộn dọc/ngang tới ô đó.

Batch UI tạo job ở backend, stream tiến độ theo item và tải kết quả từ durable
runner; tab trình duyệt không trực tiếp sở hữu 1.012 request. API gồm
`POST /batch/jobs`, `GET /batch/jobs/{id}`, `GET /batch/jobs/{id}/events` và
`GET /batch/jobs/{id}/download`.
Khi mọi câu đều answer/fallback hợp lệ, nút **Xuất ZIP nộp** chạy lại từng
`pandas_query` trên CSV đã copy, kiểm schema/provenance path, đóng ZIP
deterministic và hiển thị SHA-256. Refusal/error làm export fail-closed.

Audit browser cô lập kiểm tra claim và tương tác mà không dùng profile đăng nhập của đội:

```powershell
python scripts\audit_demo_ui_truth.py
python scripts\build_counterfactual_cell_lineage.py --resume
python scripts\audit_runtime_cell_lineage.py
python scripts\audit_cell_lineage_ui.py --url http://127.0.0.1:3000/
```

Gate yêu cầu workspace tối giản và live health xuất hiện, claim cũ gây hiểu nhầm
vắng mặt, ca VJC chạy qua API thật, measured trace/evidence đúng nguồn, motion của
Trace/Evidence/Chat–Batch hoạt động, console sạch và screenshot được tạo.
Hai audit lineage lần lượt chứng minh toàn bộ 1.012 câu có coordinate/raw đã tái kiểm và
kịch bản q69 hiển thị đúng ô `259.236.746` ở cột `Trong hạn · Tổng cộng`.

Ngay trước private/final handoff, chạy doctor fail-closed:

```powershell
python scripts\private_ready_doctor.py --require-live
```

Doctor PASS mới cho phép dùng file thi `sub_v297_scope2_a.zip`; mọi runtime cube,
lineage sidecar và `private_final_handoff_*.zip` chỉ là sản phẩm vận hành/tài liệu,
không phải payload nộp BTC.

## Kiểm tra candidate cuối

Chạy tại thư mục gốc repo:

```powershell
python scripts\check_submission_compliance.py sub_top123_candidate_v161_legacy_source43
.venv-grader\Scripts\python.exe scripts\grader_check.py sub_top123_candidate_v161_legacy_source43
python scripts\verify_source_audit.py sub_top123_candidate_v161_legacy_source43
python scripts\audit_metric_codes.py sub_top123_candidate_v161_legacy_source43 --fail-on-findings
python scripts\audit_direct_units.py sub_top123_candidate_v161_legacy_source43 --include-audited
```

Kết quả kỳ vọng: compliance `issues: 0`, hai runtime `1012/1012`, source audit `2130/2130`, metric-code `finding_count: 0`, direct-unit `finding_count: 0`. ZIP phải có `submission.json` và `data/` ngay ở archive root.

## Dựng lại catalog từ OCR

```powershell
$env:PYTHONPATH = "$PWD\src"
python src\kingpro\corpus\build_catalog.py --out build
```

Mỗi bảng được định danh bằng `report_id|line`, trong đó `line` là dòng 1-based chứa thẻ `<table>` trong file OCR. Xem chi tiết tại [Data Card](docs/DATA_CARD.md).

## Hồ sơ bàn giao

- [Product Profile](docs/PRODUCT_PROFILE.md): phạm vi, luồng xử lý và tiêu chí nghiệm thu.
- [Data Card](docs/DATA_CARD.md): nguồn, biến đổi và giới hạn dữ liệu.
- [Model Card](docs/MODEL_CARD.md): checkpoint, vai trò, giấy phép và đối chiếu ngưỡng model do BTC xác nhận.
- [Reproducibility](docs/REPRODUCIBILITY.md): quy trình tái lập và kiểm thử.
- [Compliance Checklist](docs/COMPLIANCE_CHECKLIST.md): đối chiếu yêu cầu cuộc thi với bằng chứng kỹ thuật.
- [Winning Audit](docs/FINAL_WINNING_AUDIT.md): ma trận bằng chứng đã đo/local/manual và quyết định artifact.
- [Demo Day Compliance](docs/DEMO_DAY_COMPLIANCE.md): kịch bản chứng minh luật, điểm và anti-hallucination trước giám khảo.
- `scripts/build_demo_evidence_bundle.py`: đóng gói dossier/báo cáo kỹ thuật theo allowlist, quét secret và tạo ZIP deterministic để xuất trình; không nhúng artifact thi, email BTC hoặc credential.
- `scripts/audit_demo_compliance.py`: audit model ID, endpoint egress, archive, hash và hồ sơ mà không làm lộ API key.
- `scripts/audit_demo_ui_truth.py`: chạy browser cô lập để kiểm claim readiness, tương tác và console của Judge View.
- [Legacy Source Review](docs/LEGACY_SOURCE_SCOPE_REVIEW.md): review thủ công sáu cảnh báo scope mức medium.
- [Working Notes](WORKING_NOTES.md): thuyết minh nghiên cứu chi tiết và lịch sử thử nghiệm.

## Giới hạn

- OCR có thể sai dấu, đơn vị, thứ tự ô hoặc cấu trúc header; kết quả cần được kiểm chứng trước ứng dụng tài chính thực tế.
- Điểm v192/ID 3552 là public score, không phải private score hoặc kết quả chung cuộc; v190 được giữ nguyên làm rollback đã đo.
- Full refusal stack ở ngưỡng `0.90` được kiểm trên 32 clean positives và 192 structured out-of-corpus negatives nội bộ: accept 90,6% positive, reject 100% negative trong đúng development suite hữu hạn này. Đây không phải claim test ẩn BTC; phải hiệu chỉnh lại khi đổi corpus/retriever hoặc taxonomy.
- Theo xác nhận BTC do đội thi lưu giữ, model có tổng tham số không quá 15B được chấp nhận. Hai checkpoint Qwen dùng trong pipeline có khoảng 14,7–14,8B tổng tham số và nằm trong ngưỡng này; cần lưu kèm ảnh/email xác nhận khi nghiệm thu.

## Giấy phép và sử dụng dữ liệu

Mã nguồn dự án và dữ liệu không mặc nhiên cùng giấy phép. Dữ liệu OCR nguồn tuân theo CC BY-NC 4.0; annotation ViFinQA trong bản dữ liệu hiện có không kèm giấy phép riêng. Xem [Data Card](docs/DATA_CARD.md) trước khi phân phối hoặc sử dụng ngoài cuộc thi.
