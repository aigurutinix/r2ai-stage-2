# Demo Day — kịch bản chứng minh tuân thủ

Tài liệu này là checklist trình bày, không thay thế Điều khoản Tham gia hoặc xác nhận chính thức của BTC. Chỉ phát biểu những gì có bằng chứng đi kèm.

> **Trạng thái Hà Nội hiện hành:** champion public là V297 / ID3747 với
> Execution/Answer `0.7115`, Tables F2 `0.6114`, Docs F2 `0.9611` và
> `on_leaderboard=true`. Replay mặc định là V297 (`sub_v297_scope2_a.zip`, SHA-256 `90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC`);
> full gate 522 tests + 25 subtests. V290 là rollback trực tiếp; V276 là rollback thứ hai và exact-10 public-metric tie reference; V269 là measured rollback thứ ba/source-clean;
> v206/v207 là lineage byte-locked. V297 inherits q224 union + q966 consolidated-source repair; V276 remains the exact-10 public-metric tie reference
> F2/Precision/Recall so với V217/V269. Không suy diễn public score thành private gold. Các tham chiếu v192 ở phần lịch sử phía
> dưới không còn là “current”. Xem [HANOI_DEMO_HANDOFF.md](HANOI_DEMO_HANDOFF.md).

## Một câu mở đầu

“KINGPRO không trả lời bằng một con số sinh tự do. Hệ thống tìm bảng, chạy Pandas trên CSV nguồn, khóa citation và chỉ xuất kết quả khi deterministic replay khớp; thiếu bất kỳ mắt xích nào thì từ chối.”

## Bằng chứng nên mở theo thứ tự

1. **Public score V297 / ID3747:** Execution `0.7115`, Answer `0.7115`, Tables F2-macro `0.6120`, Docs F2-macro `0.9618`; artifact `sub_v297_scope2_a.zip`, SHA-256 `90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC`, full gate 522 tests + 25 subtests.
2. **Một câu trực tiếp:** mở Answer, Pandas và Sources để chỉ ra cùng một trace.
3. **Một câu so sánh hoặc đa năm:** chứng minh hiểu tài chính tiếng Việt và nhiều DataFrame.
4. **Một câu thiếu công ty hoặc kỳ:** chứng minh hard refusal, không bịa số.
5. **Runtime proof:** model mở trong allowlist, closed-model API disabled, registry mặc định V297, 1.012 chương trình đã xác minh.
6. **Endpoint-resilience proof:** chạy câu mới FPT 2024 qua `deterministic_compiler`; mở Pandas và source cell, đồng thời chỉ rõ đây không phải registry replay hay LLM.
7. **Release proof:** V297 đạt 1.012/1.012 query ở bốn runtime; source/panel verifier kiểm tra 2.262/4.789 cells; full gate đạt 522 tests + 25 subtests và SHA-256 khóa artifact. Policy table-order là `membership`, không cho phép thiếu hoặc thêm bảng.
8. **UI proof:** 7/7 ca HTTP sân khấu, console sạch; Lighthouse desktop/mobile đều 100 ở bốn hạng mục được audit.
9. **Candidate proof v225:** ledger từng câu 1.012/1.012, bốn runtime 1.012/1.012, 2.255 source cell, 4.789 panel cell và release gate PASS; nhấn mạnh `measured=true`, ID 3723, public `0.7095`, không phải champion/private score.

## Đối chiếu luật và yêu cầu sản phẩm

| Nội dung | Cách chứng minh khi demo | Bằng chứng |
| --- | --- | --- |
| Truy hồi đúng bảng | Mở Data Room và Sources của câu trả lời | `relevant_docs`, `relevant_tables`, citation binding |
| Sinh và thực thi Pandas | Mở tab Pandas rồi replay cùng câu | query phụ thuộc DataFrame; output khớp answer |
| Dẫn nguồn minh bạch | Chỉ công ty, năm, scope, report/table reference và preview | source/panel verifier |
| Anti-hallucination | Chạy câu thiếu công ty hoặc thiếu kỳ | hard refusal, không citation thì không answer |
| Câu mới khi endpoint lỗi | Chạy “Tổng tài sản của FPT năm 2024…” | mode `deterministic_compiler`; source-bound Pandas; không nhận grammar ngoài phạm vi |
| Không hardcode đáp án | Giải thích AST/compliance gate và dependency vào CSV | compliance gate 0 lỗi |
| Tương thích grader | Nêu cú pháp Python 3.7 và Pandas 1.1.5 | compatibility audit và typed replay |
| Model mở | Mở runtime card và model card | Qwen2.5-Coder-14B-Instruct; closed-model API disabled |
| Giới hạn tham số | Không nói “luật công khai là 15B”; nói đúng rằng đội có xác nhận riêng của BTC | mang email/ảnh xác nhận ngưỡng `≤15B` |
| Công cụ hỗ trợ phát triển | Nêu coding assistant không nằm trong runtime, không sinh kết quả chấm | source code, dependency/runtime trace |
| Nguồn dữ liệu hợp pháp | Chỉ dùng kho BTC và nguồn mở/hợp pháp đã ghi hồ sơ | data inventory, license/source note |
| Tái lập bài nộp | Đưa đúng tên ZIP và hash; giữ rollback đã đo | v192 `D5D16C...E5539E27`; rollback v190 `B474FA...BDD32` |
| Hạn phát hành model | Chứng minh checkpoint có trước 01/06/2026 | model card, release/commit hoặc snapshot metadata |
| Đội và tài khoản | Một người không thuộc nhiều đội; chỉ tài khoản đội được duyệt nộp | xác nhận thành viên và lịch sử tài khoản nộp |
| Quyền công bố/nghiệm thu | Không gọi public rank là kết quả chung cuộc; chấp nhận BTC kiểm tra định lượng, định tính và tư cách | điều khoản tham gia và hồ sơ nghiệm thu |
| Quyền sử dụng bài thắng giải | Chuẩn bị source/data/artifact bàn giao đúng điều khoản, không kèm secret bên thứ ba | inventory bàn giao và secret scan |

## Cách giải thích điểm macro

Không nói các cột trên Dashboard được cộng hoặc lấy trung bình với nhau. Precision, Recall và F2-macro được tính theo từng câu rồi trung bình trên toàn bộ câu hỏi. Ở cấu hình leaderboard công khai đã kiểm tra, `Execution Accuracy` là chỉ số xếp hạng chính. BTC vẫn giữ quyền đánh giá thủ công, định tính hoặc dùng chỉ số khác khi xét giải cuối, nên UI, citation, refusal, compliance và hồ sơ tái lập vẫn là phần bắt buộc của bài demo.

## Hồ sơ phải mang theo

- email/ảnh BTC xác nhận ngưỡng model `≤15B`;
- model card và license của checkpoint mở;
- README, Product Profile, Compliance Checklist và Reproducibility;
- ZIP v192 cùng SHA-256 đầy đủ; hash v190 để trình bày rollback đã đo riêng;
- log release gate JSON;
- danh sách nguồn dữ liệu và giấy phép/quyền sử dụng;
- xác nhận nội bộ rằng mỗi thành viên chỉ thuộc một đội và chỉ tài khoản đội được duyệt dùng để nộp.

## Không nên tuyên bố

- Không gọi v186 là tốt hơn trước khi BTC đo.
- Không gọi điểm public là điểm private hoặc kết quả chung cuộc.
- Không nói retrieval “chiếm đúng 50% công thức leaderboard hiện tại” nếu Dashboard/Evaluation không hiển thị công thức đó.
- Không nói Qwen có tổng tham số “≤14B”; dùng cụm “14B class, khoảng 14,7–14,8B tổng tham số, nằm trong ngưỡng ≤15B theo xác nhận BTC được đội lưu giữ”.
- Không nói mọi câu hỏi tự do đều trả lời được; khả năng từ chối là một tính năng an toàn có chủ đích.

## Luật nào được tính vào phần demo

Trang chương trình chính thức mô tả Vòng 2 là **Product Optimization** gồm
Citation, Hallucination Control, Product Improvement và Pitching Preparation;
Top 10 Demo Day gồm Live Demo, Pitching và Q&A. Vì vậy đội không chỉ trình bày
điểm leaderboard mà phải chứng minh được toàn bộ chuỗi sản phẩm. Nguồn công khai:
[Road to AI 2026](https://r2ai.aiguru.com.vn/).

BTC hiện chưa công bố công khai trọng số chi tiết giữa Live Demo, Pitching và
Q&A. Không tự đặt công thức điểm hoặc tuyên bố điểm Dashboard sẽ quyết định toàn
bộ kết quả Demo Day. Mọi quyết định sản phẩm từ thời điểm này phải qua đồng thời
hai cổng:

1. **Cổng kỹ thuật:** retrieval, Pandas và answer chạy đúng; provenance, archive
   và runtime tương thích grader; không làm suy giảm champion đã đo.
2. **Cổng Demo Day:** trace nhìn thấy được, citation kiểm chứng được, refusal an
   toàn, model/dữ liệu hợp luật, tái lập được và người trình bày trả lời được Q&A.

## Definition of ready cho Demo Day

| Cổng | Điều kiện đạt | Trạng thái 24/08/2026 |
| --- | --- | --- |
| Hiệu năng đã được BTC đo | Có một submission được chọn với vector điểm và ID rõ ràng | **PASS:** v192 / ID 3552 |
| Artifact tái lập | ZIP/hash cố định; query, source và panel replay đều qua release gate | **PASS:** v192 đã đo; v190 là rollback đã đo |
| Live trace | Answer, Pandas, Sources và trace ID cùng một lần chạy; không dùng mock | **PASS trên 7/7 ca sân khấu đã kiểm tra** |
| Citation | Mỗi kết quả truy ngược được về DataFrame, bảng và tài liệu thực sự được đọc | **PASS trên suite nội bộ** |
| Anti-hallucination | Thiếu entity/kỳ/nguồn hoặc câu ngoài phạm vi thì từ chối, không xuất số | **PASS trên suite nội bộ** |
| Câu mới không phụ thuộc endpoint | Grounded compiler xử lý grammar bảo thủ và công khai mode | **PASS 106/106 câu compiler chọn và từ chối 29/29 adversarial variants; không phải hidden-set hay universal-safety claim** |
| Retrieval tài liệu cho câu mới | Tách công ty–năm–scope, bind mixed scope theo năm, chặn alias ngắn nằm trong tên dài; chỉ mở rộng kỳ theo tín hiệu kế toán hữu hạn và backfill catalog duy nhất, trường hợp mơ hồ fail-closed; không dùng question ID/answer/leaderboard feedback | **PASS regression full 1.012: precision `0,979938`, recall `0,988322`, F2 `0,983827`, miss `182 -> 29`; nhãn thực thi v184, không phải BTC Docs F2/gold** |
| Retrieval bảng cho câu mới | Rerank pool BM25 bằng nhãn đọc từ CSV nguồn; không dùng question ID/answer/leaderboard feedback | **PASS regression: recall@8 source-binding `0,471960 -> 0,552890`, MRR@40 `0,438546 -> 0,532769`, không bucket nào giảm; không phải BTC Tables F2/gold** |
| Model live | Endpoint trả đúng checkpoint mở, worker đang chạy và probe hoàn tất | **FAIL vận hành:** cấu hình đúng nhưng inventory chỉ có worker `EXITED`; dynamic generation phải giữ khóa |
| Hồ sơ model | Model card, license, ngày phát hành và xác nhận ngưỡng tham số của BTC | **Cần mang bản xác nhận `<=15B` của BTC** |
| Tư cách đội/tài khoản | Một người không ở nhiều đội; chỉ tài khoản đội được duyệt nộp | **Cần đội xác nhận thủ công** |
| Dữ liệu/IP/bàn giao | Inventory/license rõ ràng, secret scan sạch, sẵn sàng bàn giao nếu thắng | **Hồ sơ kỹ thuật có; cần đối chiếu điều khoản/email gốc trước Demo Day** |
| Pitching và Q&A | Chạy trọn runbook 5 phút, có fallback và trả lời đúng phạm vi bằng chứng | **Runbook có; cần diễn tập có bấm giờ và quay lại một lượt hoàn chỉnh** |

Chỉ gọi hệ thống **Demo-ready** khi không còn dòng `FAIL` và các mục thủ công đã
được đặt trong một thư mục nghiệm thu có thể mở ngay trên sân khấu. Nếu RunPod
chưa sẵn sàng, vẫn được demo registry và compiler nhưng phải nói rõ dynamic
generation đang fail-closed; không được trình bày như một runtime model đang hoạt
động bình thường.

| Nhóm luật/yêu cầu | Điều giám khảo có thể hỏi | Cách trả lời và chứng minh |
| --- | --- | --- |
| Sản phẩm | Có thật sự truy hồi, sinh Pandas và chạy không? | Chạy một trace mới; mở bảng, code, output và citation cùng trace ID |
| Citation | Nguồn có tồn tại và đúng công ty/năm không? | Mở preview dòng nguồn và table reference; nêu source/panel verifier |
| Anti-hallucination | Nếu thiếu dữ liệu thì sao? | Chạy ca thiếu entity/kỳ và ca ngoài BCTC; hệ thống phải từ chối, không xuất số |
| Endpoint resilience | Nếu model endpoint timeout ngay trên sân khấu thì sao? | Chạy ca compiler một công ty–một năm; công khai mode và giới hạn grammar; câu phức tạp vẫn fail-closed |
| Model mở và giới hạn kích thước | Model nào đang chạy, có hợp ngưỡng không? | Health card + model card/license + ảnh RunPod ghi checkpoint/revision + xác nhận BTC về ngưỡng |
| Không dùng model đóng ở runtime | Có gọi OpenAI/Claude để trả lời không? | Model ID và endpoint qua allowlist; host hiện tại `api.runpod.ai`/HTTPS; coding assistant chỉ hỗ trợ phát triển |
| Tái lập | Có chạy lại đúng kết quả được không? | Replay cùng Pandas/CSV, release report và SHA-256 của ZIP |
| Dữ liệu và IP | Nguồn nào được dùng, có quyền dùng không? | DATA_CARD, inventory/license; sẵn sàng cung cấp source/data theo điều khoản bài thắng giải |
| Đội/tài khoản | Có đúng một đội và một tài khoản nộp không? | Thành viên xác nhận; đưa thông tin tài khoản đội khi BTC yêu cầu |
| Điểm và xét giải | Điểm cuối có phải trung bình các cột không? | Không. Macro là trung bình theo câu; public rank hiện theo Execution. BTC vẫn có quyền xét thủ công/định tính |
| Công bố kết quả | Có đồng ý công bố điểm không? | Xác nhận đội đã chấp nhận điều khoản tham gia và quyền công bố của BTC |
| Hạn phát hành model | Checkpoint có được phát hành trước mốc luật không? | Mở model card/snapshot metadata thể hiện ngày trước 01/06/2026 |
| Source/data khi thắng giải | BTC có thể nghiệm thu và sử dụng bài thắng giải không? | Mở manifest bàn giao; source, data phát sinh và hash artifact; secret scan sạch |

## Ba lớp bằng chứng phải phân biệt khi nói trước giám khảo

1. **Đã được BTC đo:** v192/ID 3552 và vector điểm công khai. Đây là bằng chứng hiệu năng public, không phải điểm private/chung cuộc.
2. **Đã được local gate chứng minh:** v192 chạy 1.012/1.012 ở bốn runtime; 2.203/2.203 source cell, 4.783/4.783 panel cell và table-order data flow đều sạch. Đây là bằng chứng kỹ thuật bổ sung cho chính artifact đã đo.
3. **Phải mang hồ sơ thủ công:** xác nhận ngưỡng model của BTC, ngày phát hành/checkpoint, license dữ liệu/model, tư cách đội/tài khoản và chấp nhận điều khoản công bố/bàn giao.

Khi demo, mỗi tuyên bố phải được gắn đúng một trong ba lớp này. Không đổi nhãn
“local PASS” thành “BTC verified”, và không dùng điểm v192 để quảng bá một successor chưa được chấm.

## Audit kỹ thuật chạy trước khi lên sân khấu

```powershell
python scripts\audit_demo_compliance.py `
  --artifact sub_top123_candidate_v192_data_derived_table_order.zip `
  --output build\demo_compliance\v192_current_champion.json
```

Kết quả ngày 24/08/2026: `technical_pass: true`; model identifier, endpoint
host/transport, archive, compiler regression 106/106, adversarial refusal 29/29,
source-bound document-retrieval regression, source-proven table-reranker regression và bộ hồ sơ đều PASS. Báo cáo không in API key. Báo cáo
cũng cố ý liệt kê các bằng chứng thủ công còn phải mang theo; không biến điều
không thể kiểm chứng bằng code thành tuyên bố tự động.

Audit control-plane bổ sung chạy bằng:

```powershell
python scripts\audit_runtime_attestation.py `
  --output build\demo_compliance\runtime_attestation_20260824.json
```

Snapshot ngày 24/08/2026 xác nhận endpoint và template RunPod cùng cấu hình đúng
`Qwen/Qwen2.5-Coder-14B-Instruct`, image có tag bất biến và không lộ secret.
Nhưng pod inventory chỉ có 5 worker `EXITED` và lần kiểm tra mới nhất
`GET /models` timeout 25 giây, dù health báo 3 worker ready. Vì vậy
`control_plane_attested=false`, `runtime_identity_attested=false`. Demo vẫn chạy
an toàn bằng registry v192 và grounded compiler; free-form generation tiếp tục
fail-closed bằng `KINGPRO_LLM_ATTESTED=false`. Xem
[RUNTIME_ATTESTATION.md](RUNTIME_ATTESTATION.md).

Gate tổng hợp cho buổi diễn chạy bằng:

```powershell
python scripts\audit_demo_readiness.py `
  --artifact sub_top123_candidate_v192_data_derived_table_order.zip `
  --runtime-report build\demo_compliance\runtime_attestation_live_identity_20260824.json `
  --manual-evidence docs\demo_manual_evidence.json `
  --base-url http://127.0.0.1:3010 `
  --stage-request-budget 120 `
  --output build\demo_compliance\demo_readiness_v192_current.json
```

Exit code `0` chỉ xuất hiện khi `full_demo_ready=true`. Trạng thái hiện tại là
`stage_safe_local=true`, `full_demo_ready=false`: technical artifact, cấu hình
model và 7/7 HTTP smoke đều PASS; lượt live trên đúng v192 chạy `7.090 ms`
toàn suite, các ca trả lời `1.158–1.652 ms` và refusal `9–24 ms`, trong ngân sách
request 120 giây. Runtime model chưa operational và manifest
bằng chứng thủ công vẫn cố ý để `false`. Chỉ điền manifest bằng đường dẫn hồ sơ
đội thực sự giữ; không đưa nội dung email riêng tư hoặc secret vào repo.

Giao diện Judge View không được tự suy diễn trạng thái từ cấu hình model. Health
đọc báo cáo readiness gần nhất và hiển thị tách biệt `12/12 technical`, `7/7 smoke`,
`Dynamic LOCKED` và `0/8 manual evidence`. Audit browser cô lập chạy bằng:

```powershell
python scripts\audit_demo_ui_truth.py `
  --output build\demo_compliance\demo_ui_truth_v3.json `
  --screenshot build\demo_compliance\demo_ui_truth_v3.png
```

Kết quả hiện tại PASS toàn bộ tám check: claim đúng xuất hiện, claim cũ gây hiểu
nhầm không còn, chọn/chạy ca compiler thật thành công, điều hướng Demo Day/Data
Room tương tác được, console sạch và screenshot được tạo. Audit này chỉ chứng
minh UI local; không thay attestation model từ xa hoặc xác minh tư cách của BTC.

Tạo bộ bằng chứng kỹ thuật công khai, không kèm artifact thi và credential:

```powershell
python scripts\build_demo_evidence_bundle.py `
  --out build\demo_evidence_public_RELEASE_A
```

Mỗi release phải được dựng hai lần vào hai thư mục mới; hai ZIP phải
byte-identical, gồm 18 entry, không trùng tên và đọc kiểm tra không lỗi. Ghi
SHA-256 của archive ở biên bản bàn giao nằm ngoài bundle để tránh tài liệu bên
trong tự tham chiếu vào hash của chính nó.
Bundle chỉ chứa dossier và báo cáo kỹ thuật được allowlist; v190/v192 chỉ xuất hiện
bằng tên, kích thước và SHA-256. Email BTC, ảnh tài khoản, bằng chứng danh tính,
`.env`, API key và token bị loại có chủ đích. Bundle này không thay thế xác nhận
tư cách của BTC, attestation runtime trực tiếp hoặc bằng chứng thủ công của đội.

## Trả lời ngắn nếu bị hỏi về “điểm trung bình”

“Không có quy định nào trong phần luật đội đang giữ nói điểm cuối là trung bình
các lần nộp hay trung bình các cột Dashboard. F2-macro là trung bình metric theo
từng câu. Public leaderboard hiện dùng Execution Accuracy làm chỉ số chính.
Tuy nhiên BTC bảo lưu quyền đánh giá định lượng, thủ công và định tính khi xét
giải, nên phần sản phẩm, compliance và live demo vẫn được đội chuẩn bị đầy đủ.”

## Runbook pitching 5 phút

Mốc dưới đây có tổng thời lượng 5 phút. Người trình bày chỉ chuyển sang bước kế tiếp
khi giao diện đã hiện trạng thái cuối; nếu một request quá 10 giây thì dùng ngay ca
compiler hoặc registry replay đã kiểm tra, không đứng chờ endpoint.

| Thời gian | Thao tác | Câu nói bắt buộc | Bằng chứng trên màn hình |
| ---: | --- | --- | --- |
| `00:00–00:20` | Mở Judge View | “KINGPRO không sinh tự do một con số; hệ thống chỉ trả lời sau khi Pandas chạy lại và citation được khóa.” | health, corpus và bốn verification gates |
| `00:20–01:05` | Ca 01 — chỉ tiêu trực tiếp | “Đây là một trace từ tiếng Việt tới đúng ô BCTC.” | Answer → Pandas → Sources, cùng trace ID |
| `01:05–02:20` | Ca 02 — bốn doanh nghiệp | “Retrieval phải đủ toàn bộ entity và bảng trước khi phép tính được chấp nhận.” | danh sách entity, 12 citation đã bind và code lọc/tính |
| `02:20–03:35` | Ca 03 — bảy năm | “Hệ thống thực hiện điều kiện nhiều bước; đáp án vẫn là output replay từ DataFrame.” | 2018–2024, 22 citation đã bind và kết quả điều kiện |
| `03:35–04:20` | Ca 04 — câu mới | “Mode này là grounded compiler, không phải registry replay và không cần endpoint model.” | `deterministic_compiler`, source cell và Pandas |
| `04:20–04:50` | Ca 05 — prompt injection | “Thiếu an toàn hoặc căn cứ thì hệ thống từ chối trước model và không xuất số.” | `unsafe_instruction`, 0 citation, blocked-before-model |
| `04:50–05:00` | Chốt | “Điểm public chứng minh hiệu năng; SHA, replay và hồ sơ model/dữ liệu chứng minh khả năng nghiệm thu.” | v192 score/gate/hash và rollback v190 |

Không nhập câu mới tùy hứng trong luồng 5 phút. Nếu giám khảo yêu cầu câu tự do, nói rõ
trước rằng grammar compiler chỉ bao phủ một tập chỉ tiêu chuẩn; câu ngoài phạm vi sẽ đi
qua model mở đã attested hoặc fail-closed.

## Q&A phản biện — câu trả lời ngắn có thể kiểm chứng

| Câu hỏi | Trả lời trong 15–25 giây | Mở bằng chứng |
| --- | --- | --- |
| Có hardcode đáp án không? | Không gán số vào `result`. Registry là các chương trình đã audit; live compiler đọc ô nguồn, dựng Pandas rồi replay. Mode luôn hiện công khai để không đánh tráo hai cơ chế. | tab Pandas, source coordinates, compliance gate |
| Nếu RunPod chết thì sao? | Exact/paraphrase đã audit và grounded compiler vẫn chạy. Sinh tự do bị khóa; hệ thống không hạ chuẩn để bịa câu trả lời. | health card + ca compiler + refusal |
| Model có đúng luật không? | Runtime khai Qwen2.5-Coder-14B-Instruct, model mở Apache-2.0, phát hành 12/11/2024. Tổng tham số khoảng 14,7B; đội chỉ viện dẫn ngưỡng `≤15B` khi xuất trình xác nhận riêng bằng văn bản của BTC. | model card, RunPod control plane, email BTC |
| Có dùng GPT/Claude để trả lời không? | Không trong runtime hoặc artifact chấm. Coding assistant chỉ hỗ trợ phát triển mã; mọi số cuối được Pandas tính lại từ CSV và có thể tái lập offline. | dependencies, runtime trace, disclosure |
| Vì sao không trả lời mọi câu? | Từ chối là kiểm soát ảo giác. Thiếu entity, kỳ, nguồn, code an toàn hoặc replay khớp thì không có đáp án. | một ca thiếu công ty và một ca ngoài corpus |
| Citation có phải chỉ trang trí? | Không. Citation được bind từ chính DataFrame mà query đọc; verifier truy ngược 2.203 source cell và 4.783 panel cell về BCTC gốc. | Evidence Inspector và release report |
| Điểm cuối có phải trung bình các cột/lần nộp? | Không có điều khoản đội đang giữ nói như vậy. Macro trung bình theo câu; public rank hiện theo Execution. BTC vẫn có quyền xét định tính và nghiệm thu khi trao giải. | Evaluation/Terms + dashboard |
| v192 có tốt hơn v190 không? | Có trên public set: Execution/Answer tăng một câu và cả bốn table metrics đều tăng. Không suy rộng điều này thành private/chung cuộc. | bảng ba lớp bằng chứng |
| Có tái lập được không? | Có: môi trường grader, archive layout, 1.012 chương trình, verifier và SHA-256 đều được khóa. Bản dựng lại phải trùng hash. | Reproducibility + release JSON + SHA |
| Dữ liệu ngoài đến từ đâu? | Chỉ nguồn hợp pháp có inventory và license; không dùng gold/đáp án kín. Nguồn BTC và biến đổi OCR được ghi trong Data Card. | Data Card và inventory |
| API key có bị lộ xuống trình duyệt không? | Không. Frontend gọi proxy nội bộ; secret chỉ ở server, archive và báo cáo đều qua secret scan. | network/proxy route + archive audit |
| Vì sao Tables F2 thấp hơn Docs F2? | Xác định đúng báo cáo dễ hơn xác định đủ và vừa chính xác tập bảng cần cho phép tính nhiều bước. Đội trình bày đúng vector đã đo, không che metric yếu. | score detail v192 + một trace nhiều bảng |

## Quy tắc fallback trên sân khấu

1. Nếu UI mất kết nối: refresh health đúng một lần, sau đó mở release report và video/screenshot
   dự phòng; không sửa cấu hình live trước giám khảo.
2. Nếu endpoint timeout: chuyển thẳng ca compiler; giữ `KINGPRO_LLM_ATTESTED=false`.
3. Nếu một source preview lỗi: mở trace JSON và CSV/table reference tương ứng; không thay citation.
4. Nếu câu hỏi giám khảo ngoài BCTC: trả lời phạm vi và cho hệ thống từ chối; không đoán số.
5. Nếu bị hỏi ngưỡng `14B`/`15B`: xuất trình văn bản BTC, nói đúng 14,7B tổng tham số và
   xin giám khảo xác nhận áp dụng; tuyệt đối không gọi checkpoint là `≤14B`.
