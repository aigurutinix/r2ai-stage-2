# Reproducibility Runbook

## 24. Rebuild second balanced table-recall candidate v185

```powershell
python scripts\build_second_balanced_retrieval_recall_candidate.py
.venv-grader\Scripts\python.exe scripts\grader_check.py sub_top123_candidate_v185_second_balanced_table_recall
.venv-grader\Scripts\python.exe scripts\grader_check.py sub_top123_candidate_v185_second_balanced_table_recall --official
python scripts\check_submission_compliance.py sub_top123_candidate_v185_second_balanced_table_recall
python scripts\verify_source_audit.py sub_top123_candidate_v185_second_balanced_table_recall
python scripts\verify_panel_source_cells.py sub_top123_candidate_v185_second_balanced_table_recall
python -m pytest -q
python scripts\package_submission.py sub_top123_candidate_v185_second_balanced_table_recall --out sub_top123_candidate_v185_second_balanced_table_recall.zip --referenced-only
```

Expected package: 1,946 entries, 1,258,080 bytes, SHA-256
`F288E5DFF2E4C456F529E074D0753595FC14D4FC19DFCD61097D965C92F08E40`.
These commands do not authorize an upload.

## 1. Môi trường

Có hai môi trường tách biệt:

- **Development:** Python hiện đại với dependency trong `requirements.txt`.
- **Offline grader:** `.venv-grader`, Python 3.9.25, pandas 1.1.5, numpy 1.19.5. Môi trường này mô phỏng phiên bản Pandas của grader; không khẳng định là image Python 3.7 chính thức của BTC.

Kiểm phiên bản:

```powershell
.venv-grader\Scripts\python.exe -c "import sys,pandas,numpy; print(sys.version); print(pandas.__version__, numpy.__version__)"
```

## 2. Candidate chuẩn hiện tại

Candidate đang được bảo toàn là `sub_top123_candidate_v161_legacy_source43/`. Không chạy lại builder hoặc sửa trực tiếp thư mục này khi chỉ cần audit. Các thử nghiệm mới phải tạo ở một thư mục version khác.

```powershell
Get-FileHash sub_top123_candidate_v161_legacy_source43.zip -Algorithm SHA256
```

Expected SHA-256: `1D4917D7D4C5BA27C2303DC661EF0AD7D6F38AECF140C3F8DFADC19F3705D4AB`.

## 22. Tái lập sửa tổng dự phòng cho vay CTG q356 trên v183

```powershell
python scripts\build_ctg_total_loan_provision_candidate.py

.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v183_ctg_total_loan_provision
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v183_ctg_total_loan_provision --official
python scripts\check_submission_compliance.py `
  sub_top123_candidate_v183_ctg_total_loan_provision
python scripts\verify_source_audit.py `
  sub_top123_candidate_v183_ctg_total_loan_provision
python scripts\verify_panel_source_cells.py `
  sub_top123_candidate_v183_ctg_total_loan_provision
python scripts\audit_metric_codes.py `
  sub_top123_candidate_v183_ctg_total_loan_provision --fail-on-findings
python scripts\audit_audited_period_columns.py `
  sub_top123_candidate_v183_ctg_total_loan_provision --fail-on-findings
python scripts\audit_legacy_query_sources.py `
  sub_top123_candidate_v183_ctg_total_loan_provision `
  --out build\v183_legacy_query_sources.json
python scripts\audit_legacy_source_scope.py `
  sub_top123_candidate_v183_ctg_total_loan_provision `
  build\v183_legacy_query_sources.json --fail-on-high
python -m pytest -q

python scripts\package_submission.py `
  sub_top123_candidate_v183_ctg_total_loan_provision `
  --out sub_top123_candidate_v183_ctg_total_loan_provision.zip `
  --referenced-only
```

Kết quả: chỉ q356 đổi `6.071.288 -> 12.788.628` triệu đồng; Note 11 tổng
đối soát tuyệt đối với số âm trình bày trên bảng cân đối. Hai runtime
1.012/1.012; source 2.200/2.200; panel 4.780/4.780; 97 tests. ZIP có 1.940
entry, 1.251.042 byte và SHA-256
`D96119B2E51EF71196A98CEA7CC44A2C32A6DC04634679A2F02BA5DEF6AC40DD`.
Các lệnh trên không cấp quyền upload; v165/v182 không bị sửa.

## 23. Tái lập tỷ lệ chi phí dự phòng rủi ro MBB q671 trên v184

```powershell
python scripts\build_mbb_credit_provision_ratio_candidate.py

.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio --official
python scripts\check_submission_compliance.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio
python scripts\verify_source_audit.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio
python scripts\verify_panel_source_cells.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio
python scripts\audit_metric_codes.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio --fail-on-findings
python scripts\audit_audited_period_columns.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio --fail-on-findings
python scripts\audit_legacy_query_sources.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio `
  --out build\v184_legacy_query_sources.json
python scripts\audit_legacy_source_scope.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio `
  build\v184_legacy_query_sources.json --fail-on-high
python -m pytest -q

python scripts\package_submission.py `
  sub_top123_candidate_v184_mbb_credit_provision_ratio `
  --out sub_top123_candidate_v184_mbb_credit_provision_ratio.zip `
  --referenced-only
```

Kết quả: chỉ q671 đổi `57,44 -> 57,24%`. Note 35 và báo cáo kết quả
kinh doanh cùng xác nhận chi phí dự phòng rủi ro ròng `6.118.440` triệu đồng;
mẫu số PBT là `10.688.276` triệu đồng. Hai runtime 1.012/1.012; source
2.203/2.203; panel 4.780/4.780; 97 tests. ZIP có 1.935 entry, 1.247.478 byte
và SHA-256
`3C025FD1331540C9C8E30C9FB29EDF9A12B58B40AE62A7B5CCBB7CA359C0BE04`.
Các lệnh trên không cấp quyền upload; v165/v183 không bị sửa.

## 3. Chạy cổng tuân thủ

```powershell
python scripts\check_submission_compliance.py sub_top123_candidate_v161_legacy_source43
```

Expected:

```json
{
  "entries": 1012,
  "queries_checked": 1012,
  "issues": 0,
  "by_kind": {}
}
```

Checker fail nếu query không phụ thuộc `df`/`dfs`, gán số trực tiếp vào `result`, thiếu evidence, không thể truy nguồn, hoặc `relevant_docs`/`relevant_tables` không khớp evidence.

## 4. Chạy lại toàn bộ query

```powershell
.venv-grader\Scripts\python.exe scripts\grader_check.py sub_top123_candidate_v161_legacy_source43
```

Expected:

- entries: 1.012;
- with_query: 1.012;
- empty_query: 0;
- ran_no_exception: 1.012;
- numeric_result: 1.012;
- match_stored_answer: 1.012;
- errors: `{}`.

`match_stored_answer` xác nhận tính nhất quán nội bộ, không phải điểm gold ẩn của BTC.

## 5. Đóng gói tối thiểu

```powershell
python scripts\package_submission.py sub_top123_candidate_v161_legacy_source43 `
  --out sub_top123_candidate_v161_legacy_source43_repeat.zip `
  --referenced-only
```

Xác nhận layout và SHA-256:

```powershell
tar -tf sub_top123_candidate_v161_legacy_source43_repeat.zip | Select-Object -First 10
Get-FileHash sub_top123_candidate_v161_legacy_source43_repeat.zip -Algorithm SHA256
```

Archive root bắt buộc có `submission.json` và `data/`; `compliance_audit.json` là hồ sơ nội bộ nên không cần đưa vào ZIP thi.

## 6. Đối chiếu với candidate đầu vào

Các invariant cần giữ:

- đủ 1.012 ID và đúng thứ tự;
- mọi `answer` không đổi;
- candidate v161 gốc và ZIP khóa SHA không bị sửa/overwrite;
- evidence path/content không đổi;
- provenance thay đổi có audit ID đầy đủ;
- artifact top 10 cũ không bị overwrite.

## 7. Typed CSV runtime gate

Current top candidates must also pass pandas default dtype inference while retaining the Stage-2 `dfs` namespace proven by measured submissions:

```powershell
.venv-grader\Scripts\python.exe scripts\grader_check.py sub_top123_candidate_v161_legacy_source43 --typed-dfs
```

Both string-dfs and typed-dfs runs must execute, return a number and match the stored answer for all 1,012 entries. This catches Vietnamese single-dot thousands tokens that pandas can otherwise coerce to decimal floats before the query parser sees them.

## 8. Source-coordinate gate

For every source-audited repair, verify the recorded raw token against the
original BTC table using Pandas `iloc` coordinates:

```powershell
.venv-grader\Scripts\python.exe scripts\verify_source_audit.py `
  sub_top123_candidate_v161_legacy_source43
```

The command must report `issues: 0`. This gate is independent of the generated
runtime evidence CSV and catches a manifest that contains the correct token at
the wrong original-table coordinate.

For v161 the full cumulative gate checks 2,130 source cells, not only the cells
belonging to the 43 answer deltas from measured v106. The builder resolves an exact raw
token before writing each coordinate and records the source-row labels used by
the semantic audit:

```powershell
.venv-grader\Scripts\python.exe scripts\audit_semantic_alignment.py `
  sub_top123_candidate_v161_legacy_source43 `
  --include-audited --limit 1012
```

Expected coverage is 766 scored programs, 587 audited IDs with resolved row
labels, and zero unresolved audit sources. A low lexical score remains a
manual-review signal, never an automatic answer replacement.

Unambiguous question phrases must also agree with their normalized statement
codes. This catches parent/component substitutions such as 220/221, 240/242,
130/132, 110/111 and 140/141 across both standard and panel evidence:

```powershell
.venv-grader\Scripts\python.exe scripts\audit_metric_codes.py `
  sub_top123_candidate_v161_legacy_source43 --fail-on-findings
```

The current gate must report `finding_count: 0`, and keeps
q622 visible as one documented source-reconciled exception.

Direct one-cell programs with an explicit source unit also have an independent
power-of-ten conversion gate. Bare `VND` headers are intentionally skipped
because several banking notes declare “triệu VND” outside the extracted table:

```powershell
python scripts\audit_direct_units.py `
  sub_top123_candidate_v161_legacy_source43 --include-audited `
  --out build\v161_direct_unit_audit_all_v3.json
```

The current run checks 22 unambiguous direct reads and reports
`finding_count: 0`.

Fixed-position legacy programs also need semantic triage because the older
`str.contains` audit cannot see a wrong `iloc[row, column]` selector:

```powershell
.venv-grader\Scripts\python.exe scripts\audit_positional_semantics.py `
  sub_top123_candidate_v161_legacy_source43 --limit 100
```

This audit exposed q64, q146, q151 and q255. Its remaining low-overlap items require
manual review; low overlap alone is not permission to rewrite an answer.

Company-name alignment must also be checked against the authoritative BTC
stock-code catalog. This catches issuer substitutions that ticker-token
heuristics miss, while treating an investee/counterparty as an object rather
than the reporting entity:

```powershell
.venv-grader\Scripts\python.exe scripts\audit_entity_names.py `
  sub_top123_candidate_v161_legacy_source43 --limit 100
```

The command scans all 1,012 questions, resolves a catalog entity in 976 of
them, and must report `finding_count: 0`. It exposed q195 and q656.

## 9. Smoke test sản phẩm và verified registry

`ProductService` tự nạp candidate bảo toàn nếu file tồn tại; có thể chỉ định
candidate khác bằng `KINGPRO_REPLAY_SUBMISSION`. Một registry hit chỉ được trả
sau khi query chạy lại từ CSV và khớp answer:

```powershell
python scripts\serve_product.py
Invoke-RestMethod http://127.0.0.1:8080/health
Invoke-RestMethod http://127.0.0.1:8080/ask -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body '{"question":"Doanh thu thuần về bán hàng và cung cấp dịch vụ của GEG trong năm 2025 là bao nhiêu tỷ đồng?"}'
```

Health phải báo `replay_registry: true`, `replay_entries: 1012`,
`paraphrase_replay: true`, `paraphrase_entries: 1010`,
`paraphrase_exact_only_entries: 2` và công khai toàn bộ `paraphrase_policy`.
Hai câu q412/q464 thiếu lần lượt facet năm/ticker trong chính câu hỏi nên chỉ
được exact replay, không được fuzzy replay; câu smoke exact trả `2998.87`,
`verification.mode: verified_registry` và một citation.

Một paraphrase chỉ được replay khi công ty, năm, scope, đơn vị, intent và
time-basis tương thích, rồi đồng thời vượt năm ngưỡng khai báo trong
`.env.example`. Kết quả phải công khai
`verification.mode: verified_registry_paraphrase`, score, precision, recall,
margin và ID câu nguồn; query vẫn được chạy lại từ CSV. Câu đổi metric, đổi
time-basis hoặc hòa hai ứng viên phải không match.

Judge View được kiểm tra qua đúng proxy HTTP của frontend:

```powershell
python scripts\smoke_demo_day.py --base-url http://127.0.0.1:3010
```

Kết quả hợp lệ hiện tại là `7/7`: ba ca Judge View trả đúng VJC `208253.2` với 1
citation, nhóm HPG/HSG/MSR/NKG `0.59` với 12 citation và HPG 2018–2024 `9.04`
với 22 citation; GEG paraphrase trả `2998.87` với mode
`verified_registry_paraphrase`; câu thiếu công ty/năm bị từ chối bằng
`missing_company`; câu prompt injection bị từ chối; ca compiler mới phải trả lời từ source cell và có `verification.mode` bằng
`deterministic_compiler`. Các ca registry phải đồng thời qua replay và
citation-binding gate; ca compiler phải qua sandbox execution và đối chiếu cube độc lập.

## 10. Tái lập catalog (tùy chọn, tốn thời gian)

```powershell
$env:PYTHONPATH = "$PWD\src"
python src\kingpro\corpus\build_catalog.py --out build
```

Đây là bước tái tạo dữ liệu trung gian từ OCR, không cần chạy lại để chỉ audit candidate đã đóng gói.

## 11. Tái lập hai debt-security ablation

Builder giữ công thức mặc định cũ để v127/q937 trading vẫn tái lập đúng. Chỉ
khi truyền `--formula-variant debt-afs` mới dùng quy ước AFS đã đối chiếu trong
test set:

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v165_hag_long_receivables `
  --out sub_top123_candidate_v166_mbb_afs_debt `
  --include-ids 81 --formula-variant debt-afs

python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v166_mbb_afs_debt `
  --out sub_top123_candidate_v167_ctg_afs_debt_mean `
  --include-ids 937 --formula-variant debt-afs
```

Sau toàn bộ gate, đóng gói bằng `scripts/package_submission.py
--referenced-only`. Hash kỳ vọng là
`C879ACA3147467D47742ACB5F366E6D2A9B60A9DC9195EB303D897388F09C433`
cho v166 và
`8D807C0DC36BA55707E05E5E6DB5551C4D9D9F4352066DE206AC24BAC9C499A1`
cho v167. Đây là hai ablation chưa nộp; lệnh tái lập không cấp quyền upload.

q35 dùng một variant tách biệt để không làm thay đổi khả năng tái lập công thức
gross-sum cũ:

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v167_ctg_afs_debt_mean `
  --out sub_top123_candidate_v168_bvh_receivable_summary `
  --include-ids 35 --formula-variant receivable-summary
```

Sau gate và đóng gói `--referenced-only`, hash v168 kỳ vọng là
`1269C031B715377854DC33D0CEA12CAA738E32F97DD8D45A0F9FF3EA66B7ADA0`.

q797 dùng phép trừ có hướng qua variant riêng để giữ nguyên công thức mặc định
v106 và tạo được cả ablation đo điểm lẫn artifact đã harden nguồn:

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v106_final_burst `
  --out sub_top123_candidate_v169_v106_employee_direction `
  --include-ids 797 --formula-variant employee-direction

python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v168_bvh_receivable_summary `
  --out sub_top123_candidate_v170_employee_direction_hardened `
  --include-ids 797 --formula-variant employee-direction
```

v169 chỉ dùng để đo delta q797 trên nền đã có điểm; audit riêng q797 phải báo
2/2 source cells, còn full audit kế thừa nợ 222 coordinate cũ của v106. v170
phải qua full source audit 2,150/2,150 và hai runtime 1,012/1,012. Sau đóng gói
`--referenced-only`, hash kỳ vọng lần lượt là
`15CA1A7209E4C4D497FDB181C288D83ADCC825D295C6534435EF2EBEFEC3F980`
và `855065C4A7E18509D518A71BB8AAE9104344BBAC8994BE1C178D72346D3E384F`.
Hai lệnh này không cấp quyền upload.

## 12. Tái lập ablation truy hồi panel v171

Chế độ mặc định của builder panel vẫn giữ hành vi cũ. Chỉ cờ tường minh dưới
đây mới bật dependency tối thiểu:

```powershell
python scripts\build_compliant_panel_candidate.py `
  --base sub_top123_candidate_v170_employee_direction_hardened `
  --out sub_top123_candidate_v171_panel_minimal `
  --minimal-dependencies

python scripts\grader_check.py sub_top123_candidate_v171_panel_minimal
python scripts\grader_check.py sub_top123_candidate_v171_panel_minimal --typed-dfs
python scripts\check_submission_compliance.py sub_top123_candidate_v171_panel_minimal
python scripts\verify_source_audit.py sub_top123_candidate_v171_panel_minimal
python scripts\verify_panel_source_cells.py sub_top123_candidate_v171_panel_minimal
python scripts\audit_metric_codes.py sub_top123_candidate_v171_panel_minimal --fail-on-findings
python -m pytest -q

python scripts\package_submission.py sub_top123_candidate_v171_panel_minimal `
  --out sub_top123_candidate_v171_panel_minimal.zip --referenced-only
```

Kết quả kỳ vọng: hai runtime 1.012/1.012; compliance 0; source audit
2.150/2.150; panel source audit 4.780/4.780; metric findings 0; 32 tests pass.
ZIP có 1.974 entry, 1.258.806 byte và SHA-256
`16F92E09A3BF0D118E762E584908D858EB9F80F66C7587A7B556A9CB6920C9C9`.
Lệnh build/package không cấp quyền upload; v171 chưa được BTC đo.

## 13. Tái lập sửa đúng năm HND q336 trên v172

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v171_panel_minimal `
  --out sub_top123_candidate_v172_hnd_lease_2025 `
  --include-ids 336 --formula-variant hnd-lease-2025

python scripts\grader_check.py sub_top123_candidate_v172_hnd_lease_2025
python scripts\grader_check.py sub_top123_candidate_v172_hnd_lease_2025 --typed-dfs
python scripts\check_submission_compliance.py sub_top123_candidate_v172_hnd_lease_2025
python scripts\verify_source_audit.py sub_top123_candidate_v172_hnd_lease_2025
python scripts\verify_panel_source_cells.py sub_top123_candidate_v172_hnd_lease_2025
python scripts\audit_metric_codes.py sub_top123_candidate_v172_hnd_lease_2025 --fail-on-findings
python scripts\audit_structural_risks.py sub_top123_candidate_v172_hnd_lease_2025 --include-audited
python -m pytest -q

python scripts\package_submission.py sub_top123_candidate_v172_hnd_lease_2025 `
  --out sub_top123_candidate_v172_hnd_lease_2025.zip --referenced-only
```

Kết quả kỳ vọng: chỉ q336 đổi `378,05 -> 387,66`; hai runtime 1.012/1.012;
compliance 0; source 2.148/2.148; panel source 4.780/4.780; 37 tests pass.
ZIP có 1.974 entry, 1.258.712 byte và SHA-256
`5E1657BB0AD2752C9851CF384DE0F42E8CDAA64C6D305FB773873EAE0A3494C7`.
Không lệnh nào ở trên cấp quyền upload.

## 21. Tái lập candidate runtime-warning-clean v187

```powershell
python scripts\build_q369_copy_safe_candidate.py
python scripts\build_warning_clean_candidate.py

python scripts\package_submission.py sub_top123_candidate_v187_warning_clean `
  --out sub_top123_candidate_v187_warning_clean.zip --referenced-only

.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v187_warning_clean `
  --archive sub_top123_candidate_v187_warning_clean.zip `
  --expected-sha256 9A2443DFA6449246FDB36E317A9E388C7895CAD3773D02FEADA6E65FACB93121 `
  --product-python C:\Users\vinh\AppData\Local\Programs\Python\Python314\python.exe
```

Nếu thư mục v186/v187 đã tồn tại, builder chủ động dừng để không ghi đè; dùng
artifact hiện có và chạy từ bước package/gate. So với measured champion v184,
chỉ `pandas_query` của q170/q222/q369 thay đổi. Mọi answer, question,
`relevant_docs`, `relevant_tables` và evidence được giữ nguyên. Hai grader mode
thường và hai mode `-W error` đều đạt 1.012/1.012; source 2.203/2.203; panel
4.780/4.780; 97/97 tests. Hai lần package độc lập tạo ZIP byte-identical,
1.935 entries, 1.245.989 bytes, SHA-256
`9A2443DFA6449246FDB36E317A9E388C7895CAD3773D02FEADA6E65FACB93121`.
Release gate không build candidate và không upload.

## 15. Tái lập sửa đúng dòng chi phí nhân viên KHG q646 trên v174

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v173_aaa_short_term_bank_loan `
  --out sub_top123_candidate_v174_khg_employee_expense `
  --include-ids 646 --formula-variant khg-employee-expense

python scripts\grader_check.py sub_top123_candidate_v174_khg_employee_expense
python scripts\grader_check.py sub_top123_candidate_v174_khg_employee_expense --typed-dfs
python scripts\verify_source_audit.py sub_top123_candidate_v174_khg_employee_expense
python scripts\verify_panel_source_cells.py sub_top123_candidate_v174_khg_employee_expense
python scripts\audit_legacy_query_sources.py sub_top123_candidate_v174_khg_employee_expense `
  --out build\v174_legacy_query_sources.json
python -m pytest -q
python scripts\package_submission.py sub_top123_candidate_v174_khg_employee_expense `
  --out sub_top123_candidate_v174_khg_employee_expense.zip --referenced-only
```

Kết quả: chỉ q646 đổi `13,17 -> 10,34`; source 2.151/2.151; panel
4.780/4.780; 47 tests; ZIP 1.960 entry, 1.250.676 byte, SHA-256
`29CF55B6DE32FF61B18CE6E546673A39F13BD233E87B89AD8325F84107EE3F74`.

## 16. Tái lập tổng trái phiếu Chính phủ STB q169 trên v175

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v174_khg_employee_expense `
  --out sub_top123_candidate_v175_stb_government_bonds `
  --include-ids 169 --formula-variant stb-government-bonds-opening-2017

python scripts\grader_check.py sub_top123_candidate_v175_stb_government_bonds
python scripts\grader_check.py sub_top123_candidate_v175_stb_government_bonds --typed-dfs
python scripts\check_submission_compliance.py sub_top123_candidate_v175_stb_government_bonds
python scripts\verify_source_audit.py sub_top123_candidate_v175_stb_government_bonds
python scripts\verify_panel_source_cells.py sub_top123_candidate_v175_stb_government_bonds
python scripts\audit_metric_codes.py sub_top123_candidate_v175_stb_government_bonds --fail-on-findings
python scripts\audit_legacy_query_sources.py sub_top123_candidate_v175_stb_government_bonds `
  --out build\v175_legacy_query_sources.json
python -m pytest -q
python scripts\package_submission.py sub_top123_candidate_v175_stb_government_bonds `
  --out sub_top123_candidate_v175_stb_government_bonds.zip --referenced-only
```

Kết quả: chỉ q169 đổi `9.636.738 -> 28.037.179` triệu đồng; source
2.153/2.153; panel 4.780/4.780; 49 tests; ZIP 1.955 entry, 1.247.718 byte,
SHA-256 `082D41694B245C14EE024EF9AA791C67A2C5CFBBABEA02A79806BEA6DEA1F272`.
Không lệnh nào ở hai mục trên cấp quyền upload.

## 17. Tái lập số dư cuối năm lãi vay phải trả DLG q26 trên v176

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v175_stb_government_bonds `
  --out sub_top123_candidate_v176_dlg_interest_payable `
  --include-ids 26 --formula-variant dlg-interest-payable-ending

python scripts\grader_check.py sub_top123_candidate_v176_dlg_interest_payable
python scripts\grader_check.py sub_top123_candidate_v176_dlg_interest_payable --typed-dfs
python scripts\check_submission_compliance.py sub_top123_candidate_v176_dlg_interest_payable
python scripts\verify_source_audit.py sub_top123_candidate_v176_dlg_interest_payable
python scripts\verify_panel_source_cells.py sub_top123_candidate_v176_dlg_interest_payable
python scripts\audit_metric_codes.py sub_top123_candidate_v176_dlg_interest_payable --fail-on-findings
python scripts\audit_legacy_query_sources.py sub_top123_candidate_v176_dlg_interest_payable `
  --out build\v176_legacy_query_sources.json
python -m pytest -q
python scripts\package_submission.py sub_top123_candidate_v176_dlg_interest_payable `
  --out sub_top123_candidate_v176_dlg_interest_payable.zip --referenced-only
```

Kết quả: chỉ q26 đổi `26,43 -> 350.187,57` triệu đồng; source 2.154/2.154;
panel 4.780/4.780; 51 tests; ZIP 1.950 entry, 1.244.912 byte, SHA-256
`2CF4BE2BE349C00F449310568D496CFA09F4090B7088EE67E43A3702E11A0104`.
Các lệnh trên không cấp quyền upload.

## 20. Tái lập domain-boundary và refusal calibration của sản phẩm

```powershell
python scripts\eval_refusal_gate.py --out outputs\refusal_calibration_v4.json
python -m pytest -q
python scripts\smoke_demo_day.py
cd frontend
npm run build
```

Development suite gồm 32 câu BCTC sạch và 192 negative có đủ công ty/năm nhưng
cần nguồn ngoài kho BCTC (giá thị trường, brand, khảo sát nhân viên, analytics
ứng dụng, external ESG và câu phi tài chính). Full stack tại ngưỡng `0,90`
accept 90,6% positive và reject 100% negative trong đúng suite hữu hạn này;
không diễn giải thành kết quả test ẩn. Full regression đạt 77 tests, Judge View
smoke 4/4 và Next.js production build pass. Thay đổi chỉ thuộc product/audit,
không sửa `submission.json`, ZIP v178 hay registry v165 và không cấp quyền
upload.

## 18. Tái lập tổng XDCB dở dang dài hạn VIC q244 trên v177

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v176_dlg_interest_payable `
  --out sub_top123_candidate_v177_vic_long_term_cip `
  --include-ids 244 --formula-variant vic-ending-long-term-cip

python scripts\grader_check.py sub_top123_candidate_v177_vic_long_term_cip
python scripts\grader_check.py sub_top123_candidate_v177_vic_long_term_cip --typed-dfs
python scripts\check_submission_compliance.py sub_top123_candidate_v177_vic_long_term_cip
python scripts\verify_source_audit.py sub_top123_candidate_v177_vic_long_term_cip
python scripts\verify_panel_source_cells.py sub_top123_candidate_v177_vic_long_term_cip
python scripts\audit_metric_codes.py sub_top123_candidate_v177_vic_long_term_cip --fail-on-findings
python scripts\audit_legacy_query_sources.py sub_top123_candidate_v177_vic_long_term_cip `
  --out build\v177_legacy_query_sources.json
python scripts\audit_legacy_source_scope.py sub_top123_candidate_v177_vic_long_term_cip `
  build\v177_legacy_query_sources.json --out build\v177_legacy_source_scope.json `
  --fail-on-high
python scripts\audit_cross_question_consistency.py `
  sub_top123_candidate_v177_vic_long_term_cip
python scripts\audit_cross_question_consistency.py `
  sub_top123_candidate_v177_vic_long_term_cip --cross-currency
python -m pytest -q
python scripts\package_submission.py sub_top123_candidate_v177_vic_long_term_cip `
  --out sub_top123_candidate_v177_vic_long_term_cip.zip --referenced-only
```

Kết quả: chỉ q244 đổi `69,44 -> 339,92` trăm tỷ đồng; source 2.155/2.155;
panel 4.780/4.780; source-scope 0 high-risk trên 222 legacy rows; audit nhất
quán chéo 0 finding trên 29 cặp cùng đơn vị và 0 finding trên 39 cặp sau quy
đổi tiền tệ; 65 tests;
ZIP 1.945 entry, 1.241.004 byte, SHA-256
`A8713B9B8528AC001D69CD6CE2108FFEF448A2EAAED9FC03E1C09B767F35112D`.
Các lệnh trên không cấp quyền upload.

## 19. Tái lập tỷ lệ cam kết ngoại bảng trên tài sản NVB q717 trên v178

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v177_vic_long_term_cip `
  --out sub_top123_candidate_v178_nvb_off_balance_assets `
  --include-ids 717 --formula-variant nvb-off-balance-assets-2019

python scripts\grader_check.py sub_top123_candidate_v178_nvb_off_balance_assets
python scripts\grader_check.py sub_top123_candidate_v178_nvb_off_balance_assets --typed-dfs
python scripts\check_submission_compliance.py sub_top123_candidate_v178_nvb_off_balance_assets
python scripts\verify_source_audit.py sub_top123_candidate_v178_nvb_off_balance_assets
python scripts\verify_panel_source_cells.py sub_top123_candidate_v178_nvb_off_balance_assets
python scripts\audit_metric_codes.py sub_top123_candidate_v178_nvb_off_balance_assets --fail-on-findings
python scripts\audit_legacy_query_sources.py sub_top123_candidate_v178_nvb_off_balance_assets `
  --out build\v178_legacy_query_sources.json
python scripts\audit_legacy_source_scope.py sub_top123_candidate_v178_nvb_off_balance_assets `
  build\v178_legacy_query_sources.json --out build\v178_legacy_source_scope.json `
  --fail-on-high
python scripts\audit_cross_question_consistency.py `
  sub_top123_candidate_v178_nvb_off_balance_assets
python scripts\audit_cross_question_consistency.py `
  sub_top123_candidate_v178_nvb_off_balance_assets --cross-currency
python -m pytest -q
python scripts\package_submission.py sub_top123_candidate_v178_nvb_off_balance_assets `
  --out sub_top123_candidate_v178_nvb_off_balance_assets.zip --referenced-only
```

Kết quả: chỉ q717 đổi `16,51 -> 14,99%`; mẫu số đổi từ bảng phân khúc có
ngữ cảnh `01/01/2019` sang `TỔNG TÀI SẢN CÓ` cuối năm tại bảng cân đối công ty
mẹ; source 2.157/2.157; panel 4.780/4.780; source-scope 0 high-risk trên 221
legacy rows; audit nhất quán chéo 0 finding trên 29 cặp cùng đơn vị và 0 trên
39 cặp sau quy đổi tiền tệ; 70 tests. Hai ZIP có 1.940 entry, 1.235.798 byte,
byte-identical, SHA-256
`3514C00115E3665FB5A2F74C8F244E44762311468720C29A0C05F84E780871CD`.
Các lệnh trên không cấp quyền upload.

## 14. Tái lập sửa đúng nhóm vay AAA q354 trên v173

```powershell
python scripts\build_compliant_standard_candidate.py `
  --base sub_top123_candidate_v172_hnd_lease_2025 `
  --out sub_top123_candidate_v173_aaa_short_term_bank_loan `
  --include-ids 354 --formula-variant aaa-short-bank-2021

python scripts\grader_check.py sub_top123_candidate_v173_aaa_short_term_bank_loan
python scripts\grader_check.py sub_top123_candidate_v173_aaa_short_term_bank_loan --typed-dfs
python scripts\check_submission_compliance.py sub_top123_candidate_v173_aaa_short_term_bank_loan
python scripts\verify_source_audit.py sub_top123_candidate_v173_aaa_short_term_bank_loan
python scripts\verify_panel_source_cells.py sub_top123_candidate_v173_aaa_short_term_bank_loan
python scripts\audit_metric_codes.py sub_top123_candidate_v173_aaa_short_term_bank_loan --fail-on-findings
python scripts\audit_legacy_query_sources.py sub_top123_candidate_v173_aaa_short_term_bank_loan `
  --out build\v173_legacy_query_sources.json
python -m pytest -q

python scripts\package_submission.py sub_top123_candidate_v173_aaa_short_term_bank_loan `
  --out sub_top123_candidate_v173_aaa_short_term_bank_loan.zip --referenced-only
```

Kết quả kỳ vọng: chỉ q354 đổi `2,95 -> 14,01`; hai runtime 1.012/1.012;
compliance 0; source 2.149/2.149; panel source 4.780/4.780; legacy tracer
226/226 dòng, 0 unresolved, 0 provenance mismatch; 45 tests pass. Hai lần đóng
gói độc lập tạo ZIP 1.969 entry, 1.255.512 byte và cùng SHA-256
`E4BD93CE0477D5C392E4627262635A651FE679FE59E165F675F3D94A606947C7`.
Không lệnh nào ở trên cấp quyền upload.
## 25. Tái lập candidate nhãn năm phụ thuộc dữ liệu v188

```powershell
python scripts\build_data_derived_year_labels_candidate.py
python scripts\audit_data_derived_labels.py `
  sub_top123_candidate_v188_data_derived_year_labels
python scripts\package_submission.py `
  sub_top123_candidate_v188_data_derived_year_labels `
  --out sub_top123_candidate_v188_data_derived_year_labels.zip `
  --referenced-only
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v188_data_derived_year_labels `
  --archive sub_top123_candidate_v188_data_derived_year_labels.zip `
  --expected-sha256 1DD5EABD37ABE2931CC78A548330FBB7C545735D2912C36C449B266238F568B8 `
  --require-data-derived-labels `
  --product-python C:\Users\vinh\AppData\Local\Programs\Python\Python314\python.exe
```

Kết quả đã khóa: 53 câu trả năm, 213 nhãn năm đọc từ DataFrame, 0 issue;
1.012/1.012 ở normal/strict-warning string/typed runtime; 121 tests; hai ZIP
byte-identical, 1.247.943 byte, 1.935 entry; SHA-256
`1DD5EABD37ABE2931CC78A548330FBB7C545735D2912C36C449B266238F568B8`.
Builder không ghi đè thư mục đích và các lệnh trên không upload.
## 26. Tái lập grounded compiler và ca demo không phụ thuộc endpoint

```powershell
python scripts\audit_deterministic_compiler.py `
  --require-perfect --min-compiled 67 `
  --out build\demo_compliance\deterministic_compiler_v1.json
python scripts\audit_demo_compliance.py `
  --artifact sub_top123_candidate_v192_data_derived_table_order.zip `
  --output build\demo_compliance\v192_current_champion.json
python scripts\smoke_demo_day.py --base-url http://127.0.0.1:3010 --timeout 30
python -m pytest -q
```

Kết quả ngày 24/08/2026: compiler tự nhận 100/1.012 câu registry thuộc grammar
bảo thủ và khớp 100/100 trong tolerance làm tròn 0,005 đơn vị đầu ra; 9 câu panel
thuộc topology argmax/argmin theo một công ty–nhiều năm hoặc nhiều công ty–một
năm. Mười một câu công thức trực tiếp mới gồm kết quả tài chính ròng, thuế hiện
hành, tỷ suất lợi nhuận ròng, SG&A intensity và nợ ngắn hạn/vốn chủ. Câu đếm/lọc
nhiều điều kiện q380 bị từ chối có chủ đích; 7/7 HTTP smoke sau backend reload PASS trong
`7.090 ms` toàn suite, các ca trả lời `1.158–1.652 ms` và refusal `9–24 ms`, dưới
ngân sách request 120 giây; 166/166 tests cùng 25 subtests PASS. Đây là local regression trên compiler-selected
subset, không phải tuyên bố coverage ngôn ngữ tự do, hidden set hoặc private
leaderboard. Câu ngoài grammar tiếp tục đi qua model mở đã attested hoặc bị từ
chối an toàn.

## 27. Attestation runtime model mở

```powershell
python scripts\audit_runtime_attestation.py `
  --output build\demo_compliance\runtime_attestation_20260824.json `
  --timeout 20
```

Kết quả ngày 24/08/2026: model card Qwen công bố 14,7B, Apache-2.0 và
ngày 12/11/2024; endpoint/template cùng khai đúng model ID, image tag bất biến.
Tuy nhiên pod inventory chỉ có 5 worker `EXITED`; native `/run` kẹt queue 30
giây và được hủy đúng job probe, còn `/models` timeout 120 giây. Health báo 3
ready là không nhất quán với inventory, nên `control_plane_attested=false`,
`runtime_identity_attested=false` và `dynamic_enable_recommended=false`.
Audit không tự sửa `.env`; free-form generation tiếp tục fail-closed. Bật một
active worker có chi phí và cần operator phê duyệt trước khi thay cấu hình cloud.

## 28. Grounded compiler trong pipeline private set

`scripts/build_full_submission.py` hiện thử grounded compiler trước model cho
các câu một công ty–một năm và topology argmax/argmin panel thuộc grammar bảo
thủ. Nếu compiler không nhận câu,
pipeline bình thường mới chuyển sang retrieval/model; không tự điền một đáp án
suy đoán. Mỗi query compiler mang binding biến DataFrame → `table_ref`, sau đó
builder copy đúng CSV vào evidence của submission.

Pilot offline không cần API key:

```powershell
python scripts\build_full_submission.py `
  --compiler-only `
  --questions tests\fixtures\compiler_submission_questions.jsonl `
  --n 1 `
  --out build\private_compiler_pilot

.venv-grader\Scripts\python.exe scripts\grader_check.py `
  build\private_compiler_pilot
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  build\private_compiler_pilot --official
python scripts\check_submission_compliance.py build\private_compiler_pilot
```

`--compiler-only` là cổng CI/pilot: chỉ một câu ngoài grammar cũng làm cả lượt
fail, vì vậy không được dùng nó để tạo một private submission với các hàng 0.
Chế độ build bình thường mới dùng model mở làm fallback. Builder mặc định từ
chối nếu `--out` hoặc ZIP cùng tên đã tồn tại; chỉ `--resume` hoặc
`--overwrite` tường minh mới cho phép tiếp tục/thay đúng target đã kiểm tra.

Kết quả pilot ngày 24/08/2026: một câu FPT 2024 chạy và khớp answer ở bốn
runtime string/typed, compliance 0 issue; hai lần build byte-identical, ZIP
SHA-256 `B0C15D6F14FD7F5BB772C10F88A1A91DCAE88147744806F651D6F8ADB7C59C57`.
Pilot panel v2 gồm q363/q381/q392/q398 tạo 64 CSV evidence, qua 4/4
string/official-typed runtime và compliance 0; hai ZIP dựng độc lập cùng SHA-256
`C688E142D0C8FB6F904A40FC446F08DD06A6D3EDE9A2E383F785C001079B77E9`.
Pilot công thức trực tiếp v3 gồm 11 câu mới tạo 11 CSV evidence, qua 11/11
string runtime, 11/11 official-typed runtime và compliance 0; hai ZIP dựng độc
lập byte-identical với SHA-256
`32325624FD674A83D2C861B3DCA338B4E98739BC424EF67B73F6F162B4C92991`.
Pilot alias chuẩn v5 gồm 21 câu mới tạo 21 CSV evidence, qua 21/21 string runtime,
21/21 official-typed runtime và compliance 0; hai ZIP dựng độc lập byte-identical
với SHA-256 `0560901DA4E2ADA355AF4FCF544F8CCBDAE4388E22D6AB7F6468FDFDF5A1DD0A`.
Audit registry sau mở rộng xác nhận 100/100 answer và 100/100 binding provenance,
0 mismatch. `scripts/audit_compiler_adversarial.py` giữ 7 positive controls và
từ chối 29/29 biến thể sai time-basis, comparison, scenario, scope, unit hoặc
note qualifier. Đây là finite local suite, không phải chứng minh an toàn phổ quát
hay private score.

## 29. Hồ sơ bằng chứng Demo Day công khai

```powershell
python scripts\build_demo_evidence_bundle.py `
  --out build\demo_evidence_public_20260824
```

Builder chỉ copy các tài liệu và báo cáo kỹ thuật trong allowlist, quét mẫu secret,
ghi hash artifact thi vào manifest nhưng không nhúng payload ZIP dự thi. Nó từ chối
ghi đè target và chỉ cho phép output nằm dưới `build/`. Hai lần dựng độc lập phải
cho cùng SHA-256, 16 entry, 0 duplicate và `ZipFile.testzip() = None`. SHA được ghi
ở biên bản bàn giao bên ngoài bundle để không tạo vòng tự tham chiếu. Đây là bằng chứng kỹ thuật
công khai; hồ sơ thủ công và attestation runtime trực tiếp vẫn là các cổng riêng.

## 30. Table reranker theo nhãn CSV nguồn

Baseline và bản rerank đều chạy đúng cùng 1.012 câu và lấy target từ table
bindings mà query source-audited thực sự đọc; không dùng hidden gold, answer hay
biến động leaderboard:

```powershell
python scripts\audit_retrieval_provenance_recovery.py `
  --facet-backfill --scope-neutral-backfill `
  --opening-year-backfill --formula-year-backfill --growth-scan-year-backfill `
  --output build\demo_compliance\retrieval_provenance_semantic_v6_baseline.json

python scripts\audit_retrieval_provenance_recovery.py `
  --facet-backfill --scope-neutral-backfill `
  --opening-year-backfill --formula-year-backfill --growth-scan-year-backfill `
  --label-weight 4 `
  --output build\demo_compliance\retrieval_provenance_semantic_v6_reranked.json

python scripts\audit_retrieval_reranker.py `
  --output build\demo_compliance\table_reranker_regression_semantic_v6.json
```

Gate yêu cầu đủ 1.012 câu, document stage không đổi, recall@8 và MRR@40 toàn
tập cùng tăng, đồng thời không bucket lookup/analytic/banking/ratio/multi-table
nào giảm ở hai metric này. Snapshot semantic retrieval hiện hành ngày 24/08/2026 PASS:
document recall `0,988322`; pipeline recall@8 `0,471960 -> 0,552890`, MRR@40
`0,438546 -> 0,532769`. Đây chỉ là regression
khôi phục bảng nguồn thực thi; không được gọi là Tables F2 BTC hoặc dự báo điểm
ẩn. Product dùng pool BM25 40, rerank theo cột nhãn tự nhận diện từ CSV rồi mới
cắt về table budget; nếu CSV lỗi, code giữ nguyên thứ tự BM25.

## 31. Document facets theo alias và khoảng năm

Audit dưới đây so sánh đúng cùng 1.012 câu với target lấy từ `relevant_docs` của
chương trình v184 có thể thực thi; không sử dụng hidden gold hoặc biến động
leaderboard:

```powershell
python scripts\audit_document_retrieval_coverage.py `
  --per-pair 1 `
  --facet-backfill --scope-neutral-backfill `
  --opening-year-backfill --formula-year-backfill --growth-scan-year-backfill `
  --output build\demo_compliance\document_retrieval_coverage_semantic_v19_final.json

python scripts\audit_document_retrieval_facets.py `
  --output build\demo_compliance\document_retrieval_facets_semantic_v2.json
```

Parser scope hiển ngôn/mixed-year, alias punctuation không chồng lên tên pháp lý
dài hơn, unique preferred/scope-neutral backfill và ba dependency năm bảo thủ tăng
macro precision `0,935530 -> 0,979938`, recall `0,912056 -> 0,988322`, macro F2
`0,907888 -> 0,983827` và giảm số câu miss `182 -> 29`. Nhóm facet có nhiều report,
câu không nêu universe và scope không suy chắc đều fail-closed. Một thử nghiệm mở
rộng năm ngầm chỉ đạt F2 `0,909472` nên bị loại; report production trước đạt
`0,951868`, thấp hơn bản được chọn. Gate lưu hash của baseline, bản trung gian, bản được chọn và
bản bị loại, đồng thời từ chối nếu tập 1.012 câu/candidate/cap thay đổi. Đây là regression source-bound
nội bộ, không phải Docs F2 của BTC hoặc dự báo điểm private.

## 32. Tái lập rollback v190 và candidate EPS v191

v190 kế thừa v189 và chỉ bổ sung toán hạng doanh thu so sánh 2024 của PDR cho
q411. v189 kế thừa v188 và sửa toán tử q375 từ thương thành hiệu tuyệt đối giữa
hai trung bình. BTC đã đo đúng ZIP v190/ID 3545: Execution `0.6858`, Answer
`0.6877`, Tables F2 `0.6011`, Docs F2 `0.9611`. Artifact đã đo phải giữ nguyên.

```powershell
Get-FileHash sub_top123_candidate_v190_pdr_growth_count.zip -Algorithm SHA256
# B474FAA474850A490E0574AE52DBE76134194048ED6C8BC692D89919009BDD32
```

v191 là successor cô lập dùng để xây v192. Builder đọc v190, bổ sung hai ô EPS bị thiếu
cho q511, chọn HPG vì EPS `4.037` lớn hơn HT1 `1.681` và DPM `1.551`, rồi tính
`8.600.550.706.227 / 40.622.949.840.810 * 100 = 21,17%`.

```powershell
python scripts\build_eps_selection_candidate.py
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v191_eps_selection --ids 375,411,511
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v191_eps_selection --ids 375,411,511 --official
python scripts\verify_panel_source_cells.py `
  sub_top123_candidate_v191_eps_selection
python scripts\package_submission.py `
  sub_top123_candidate_v191_eps_selection `
  --out sub_top123_candidate_v191_eps_selection.zip
python scripts\final_release_gate.py `
  sub_top123_candidate_v191_eps_selection `
  --archive sub_top123_candidate_v191_eps_selection.zip `
  --expected-sha256 1548EC8861EFA162623D2EA8B1A5410DC029FB1D5EDA475E614963EC2FA8A7D7 `
  --product-python C:\Users\vinh\AppData\Local\Programs\Python\Python314\python.exe `
  --out build\release_gate\sub_top123_candidate_v191_eps_selection\final.json `
  --require-data-derived-labels
```

Kết quả: hai ZIP độc lập byte-identical, 5.218 entry, 3.191.563 byte, SHA-256
`1548EC8861EFA162623D2EA8B1A5410DC029FB1D5EDA475E614963EC2FA8A7D7`.
Full release gate PASS 1.012/1.012 ở bốn runtime, source 2.203/2.203, panel
4.783/4.783, 163 test + 25 subtest. Không upload trong quy trình build/gate.

## 33. Tái lập champion data-derived table order v192

v192 đọc v191 và chỉ sắp lại danh sách `relevant_tables` ở 81 câu panel. Builder
chạy query trên đúng CSV evidence rồi ưu tiên ticker/năm được chương trình chọn.
Không dùng question ID, answer, hidden gold hay biến động leaderboard để xếp hạng.
Membership và multiplicity của mọi danh sách bảng được giữ nguyên.

```powershell
python scripts\build_data_derived_table_order_candidate.py
python scripts\verify_data_derived_table_order.py
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v192_data_derived_table_order
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v192_data_derived_table_order --official
python scripts\package_submission.py `
  sub_top123_candidate_v192_data_derived_table_order `
  --out sub_top123_candidate_v192_data_derived_table_order.zip `
  --referenced-only
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v192_data_derived_table_order `
  --archive sub_top123_candidate_v192_data_derived_table_order.zip `
  --expected-sha256 D5D16C202863022670149AAB6C3BF13BB3F77D1E94C8546C20789346E5539E27 `
  --require-data-derived-labels `
  --allow-relevant-table-order
```

Verifier độc lập replay 175 panel, xác định được 136, không có biến selector nào
nằm ngoài static dependency path tới `result`, và không có selected-table rank
regression. MRR@5 proxy source-derived tăng `0,509191 -> 1,000000`; đây không
phải điểm BTC hay cam kết hidden gold. Gate mặc định của compliance/provenance
vẫn yêu cầu exact order; cờ `--allow-relevant-table-order` chỉ thay kiểm tra đó
bằng equality theo membership/multiplicity, còn 4.783 source cells vẫn được so
tọa độ và raw token nghiêm ngặt.

Hai ZIP referenced-only độc lập byte-identical, 1.935 entry, 1.248.122 byte,
SHA-256 `D5D16C202863022670149AAB6C3BF13BB3F77D1E94C8546C20789346E5539E27`.
Full release gate PASS 1.012/1.012 ở bốn runtime, source 2.203/2.203, panel
4.783/4.783, 166 test + 25 subtest. BTC đã đo đúng ZIP này ở submission ID 3552:
Execution `0.6877`, Answer `0.6897`, Tables F2/Precision/Recall/MRR@5
`0.6014/0.5808/0.6159/0.6384`, Docs F2/Precision/Recall/MRR@5
`0.9611/0.9580/0.9672/0.9806`. So với v190, Execution và Answer tăng đúng
một câu; cả bốn table metrics tăng và document metrics giữ nguyên. Không upload
trong quy trình build/gate; submission ID 3552 do người dự thi thực hiện riêng.

## 34. Ứng viên exact-metric table order v193

v193 giữ nguyên toàn bộ v192 và chỉ tiếp tục xử lý 39 panel mà extractor đầu
tiên không xác định được nhóm kết quả. Audit mới chỉ chấp nhận biểu thức chọn
scalar/max/min, loại trừ mean/sum/count, rồi yêu cầu giá trị kết quả đã làm tròn
khớp duy nhất một derived-metric cell có ticker/năm trong chính runtime query.
Quy tắc không dùng question ID, answer override, hidden gold hay biến động điểm.

Audit xác định đúng 3 trường hợp: q364 DPM–2021, q369 HSG–2023 và q382
DPM–2023. Thứ hạng bảng của nhóm sinh kết quả chuyển lần lượt `14 -> 1`,
`12 -> 1` và `4 -> 1`; membership/multiplicity của `relevant_tables`, answer,
query, evidence và docs đều bất biến. 36 panel aggregate/mơ hồ còn lại được giữ
nguyên.

```powershell
python scripts\audit_unresolved_table_order.py
python scripts\build_exact_metric_table_order_candidate.py
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v193_exact_metric_table_order
.venv-grader\Scripts\python.exe scripts\grader_check.py `
  sub_top123_candidate_v193_exact_metric_table_order --official
python scripts\package_submission.py `
  sub_top123_candidate_v193_exact_metric_table_order `
  --out sub_top123_candidate_v193_exact_metric_table_order.zip `
  --referenced-only
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v193_exact_metric_table_order `
  --archive sub_top123_candidate_v193_exact_metric_table_order.zip `
  --expected-sha256 9BD041C940906EFF7ABE9CA7CC4575A1A8B14110774991574C1F784D6707D434 `
  --require-data-derived-labels `
  --allow-relevant-table-order
```

Hai ZIP referenced-only độc lập byte-identical, 1.935 entry, 1.248.124 byte,
SHA-256 `9BD041C940906EFF7ABE9CA7CC4575A1A8B14110774991574C1F784D6707D434`.
Full release gate PASS 1.012/1.012 ở bốn runtime, source 2.203/2.203, panel
4.783/4.783, 167 test + 25 subtest. Đây là ứng viên retrieval chưa được BTC đo;
v192/ID 3552 vẫn là champion đã đo và rollback mặc định. Quy trình không upload.

## 35. Ứng viên attribute-metric table order v194

v194 nối tiếp v193 và mở rộng cùng audit exact-metric cho cú pháp Pandas
attribute (`frame.metric.max()`) bên cạnh cú pháp subscript
(`frame["metric"].max()`). Điều kiện bảo thủ không đổi: không reverse-match
mean/sum/count, kết quả đã làm tròn phải khớp duy nhất một derived-metric cell,
và ticker/năm phải thuộc panel audit.

Sáu câu được sắp lại: q464 HHS–2016 (`130 -> 1`), q472/q489/q547 DPM–2022
(`17 -> 1`) và q551/q574 GEE–2024 (`5 -> 1`). Đây là ba topology câu hỏi,
trong đó các ID lặp là các cách diễn đạt khác nhau của cùng phép toán và cùng
nguồn. v194 cộng dồn 9 reorder exact-metric trên v192; 30 panel aggregate/mơ hồ
còn lại không thay đổi.

```powershell
python scripts\audit_unresolved_table_order.py
python scripts\build_attribute_metric_table_order_candidate.py
python scripts\package_submission.py `
  sub_top123_candidate_v194_attribute_metric_table_order `
  --out sub_top123_candidate_v194_attribute_metric_table_order.zip `
  --referenced-only
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v194_attribute_metric_table_order `
  --archive sub_top123_candidate_v194_attribute_metric_table_order.zip `
  --expected-sha256 B2BEFD5C5F1859D3E25D1023EF2096AE1A05F0897283175430C7A731343DBA47 `
  --require-data-derived-labels `
  --allow-relevant-table-order
```

Hai ZIP referenced-only độc lập byte-identical, 1.935 entry, 1.248.141 byte,
SHA-256 `B2BEFD5C5F1859D3E25D1023EF2096AE1A05F0897283175430C7A731343DBA47`.
Full release gate PASS 1.012/1.012 ở bốn runtime, source 2.203/2.203, panel
4.783/4.783, 167 test + 25 subtest. Đây là ứng viên retrieval chưa được BTC đo;
không có upload trong quá trình build/gate.

## 36. Accuracy triage bảo thủ sau v192

Sau khi v192 đạt Execution `348/506`, một vòng đối chiếu độc lập được thực hiện
trên các câu legacy có semantic-overlap thấp và các câu mà cache lời giải cũ
khác đáng kể với artifact hiện hành. Cache cũ chỉ được dùng để xếp hàng kiểm
tra; mọi kết luận đều quay lại ô BCTC gốc, không dùng biến động leaderboard làm
nhãn và không tạo candidate nếu nguồn chưa chứng minh một đáp án khác.

Các trường hợp đã loại khỏi diện sửa gồm:

- q249: bảng HDG `|1135` cho cả vốn được duyệt và vốn đã phát hành cuối 2016 là
  `759.680.800.000` VND, đúng `7,60` trăm tỷ.
- q71: bảng GEG `|2016` cho hai cấu phần `2.935.428.348.323` và
  `63.438.994.258` VND, tổng đúng `2.998.867.342.581` VND hay `2.998,87` tỷ.
- q171: lưu chuyển tiền thuần từ hoạt động kinh doanh NLG 2021, mã 20 tại
  `|306`, là `1.295.542.317.371` VND, đúng `12,96` trăm tỷ; giá trị cache cũ
  `2,07` không khớp nguồn.
- q388: nhóm CFO-margin âm gồm NVL, DIG, IJC, CEO và CRE; trung bình năm biên
  lợi nhuận gộp tính từ mã 10/20 là `22,88%`. Giá trị cache cũ `22,92%` chỉ
  trùng DIG và bỏ bốn doanh nghiệp còn lại.
- q481/q541/q555: biên LNST/doanh thu CEO lần lượt khoảng `12,19%`, `8,70%`
  và `12,69%` cho 2022-2024; hai năm đạt ngưỡng là 2022 và 2024, doanh thu nhỏ
  nhất là `1.307.936.213.643` VND, đúng `1.307,94` tỷ hay `1,31` nghìn tỷ.
- q35: bảng tổng hợp BVH `|1232` ghi trực tiếp phải thu Bảo Việt Nhân thọ
  `222.575.005.778` VND. Con số này cũng bằng hai khoản phải thu chi tiết trừ
  hai khoản phải trả trong bảng `|1514`, nên đáp án hiện hành `222.575,01`
  triệu là số dư thuần được báo cáo, không phải một tổng tùy ý.

Audit semantic thấp tiếp tục xác nhận q60, q67, q204, q210, q222, q294, q314,
q634, q683, q691 và q796 từ chính các dòng/tổng nguồn. Vòng này không phát hiện
được answer delta nào đủ bằng chứng để dựng v195. Quyết định không sửa là một
gate chống regression: v192 vẫn là champion đã đo; v194 chỉ tối ưu thứ tự bảng.

Sau khi làm rõ coverage của registry, health runtime báo `1.012` exact entries,
`1.010` paraphrase-safe entries và `2` exact-only entries; q412/q464 không được
fuzzy replay vì thiếu facet năm/ticker. Exact replay của chúng vẫn giữ nguyên.
Full product regression sau thay đổi này PASS `168/168` tests + `25` subtests;
smoke qua proxy PASS `7/7` và UI truth audit PASS toàn bộ checks.

## 37. Aggregate contributor table order v195

v195 nối tiếp v194 và chỉ sắp lại `relevant_tables` cho 23 câu aggregate
`mean`/`sum`/`count`. Thứ tự mới được suy ra bằng cách chạy lại chính truy vấn Pandas,
xác định các doanh nghiệp thực sự đóng góp vào kết quả rồi đưa các bảng nguồn của
nhóm đóng góp lên trước. Không thay answer, query, evidence, relevant-doc, membership
hay multiplicity của relevant-table.

```powershell
python scripts\audit_aggregate_table_order.py
python scripts\build_aggregate_contributor_table_order_candidate.py
python scripts\verify_aggregate_contributor_table_order.py
python scripts\package_submission.py `
  sub_top123_candidate_v195_aggregate_contributor_order `
  --out sub_top123_candidate_v195_aggregate_contributor_order.zip `
  --referenced-only
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v195_aggregate_contributor_order `
  --archive sub_top123_candidate_v195_aggregate_contributor_order.zip `
  --expected-sha256 2C52CDF6BE8064689D0DC5C5871F0C9BBFEA2FAA85416BA995E1BA34A4942BB4 `
  --require-data-derived-labels `
  --allow-relevant-table-order
```

Verifier xác nhận 23/23 thay đổi chỉ là thứ tự bảng, số contributor source trong
top 5 tăng `28 -> 91`, replay runtime khớp toàn bộ. ZIP referenced-only có 1.935
entry, 1.246.776 byte, SHA-256
`2C52CDF6BE8064689D0DC5C5871F0C9BBFEA2FAA85416BA995E1BA34A4942BB4`.
Full release gate PASS 1.012/1.012 ở bốn runtime, source 2.203/2.203, panel
4.783/4.783, toàn bộ hard audit, 168 test + 25 subtest.

BTC đo artifact này dưới tên upload `sub_top123_candidate_v195_aggregate.zip`,
submission ID 3555. Execution `0.6877`, Answer `0.6897`, Tables
F2/Precision/Recall/MRR@5 `0.6014/0.5808/0.6159/0.6415`, Docs
F2/Precision/Recall/MRR@5 `0.9611/0.9580/0.9672/0.9806`. So với v192, chỉ
Tables MRR@5 tăng `0.6384 -> 0.6415`; mọi metric khác giữ nguyên. v195 chưa được
gắn leaderboard, còn v192/ID 3552 vẫn là hàng đang selected.

## 38. Effective-tax deferred-sign repair v196

Research chéo SpreadsheetBench Verified, RePairTQA và Financial-QA error analysis
được ánh xạ vào `audit_cross_task_risk_matrix.py`. Trục missingness + dấu kế toán
đưa q993 lên hàng review và phát hiện v195 đã lấy trị tuyệt đối của khoản thuế TNDN
hoãn lại âm tại SJG 2022. Bốn dòng PBT/current-tax/deferred-tax/PAT tự hòa giải chỉ
khi giữ dấu âm; trung bình thuế suất thực tế của GEE/VGC/SJG là `10,65%`.

```powershell
python scripts\build_effective_tax_rate_sign_candidate.py
python scripts\package_submission.py `
  sub_top123_candidate_v196_effective_tax_sign `
  --out sub_top123_candidate_v196_effective_tax_sign.zip `
  --referenced-only
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v196_effective_tax_sign `
  --archive sub_top123_candidate_v196_effective_tax_sign.zip `
  --expected-sha256 4A859AB01AFF2FDD4A1DD243B6DA47868F43381086A725FC63BBB7BC4213D4A1 `
  --require-data-derived-labels `
  --allow-relevant-table-order
```

Full release gate PASS 1.012/1.012 ở bốn runtime, source 2.203/2.203, panel
4.783/4.783, toàn bộ hard audit, 168 test + 25 subtest. BTC đo artifact dưới
tên upload rút gọn `sub_top123_candidate_v196_effective.zip`, submission ID
3557: Execution `0.6877`, bằng v195. Trang detailed-result tại thời điểm kiểm
tra trả HTML rỗng nên chưa xác minh được Answer Accuracy và các metric retrieval.

Sau phép đo, bộ sinh chuẩn được sửa ở mức mô hình dữ liệu: `Operand` có cờ
`preserve_sign` opt-in. q993 chỉ bật cờ cho ô chi phí thuế hoãn lại SJG; VGC
vẫn dùng magnitude vì công thức trừ khoản thu nhập thuế. Hai test hồi quy khóa
cả hành vi mới lẫn hành vi magnitude mặc định. Toàn bộ suite sau thay đổi đạt
170 test + 25 subtest.

## 39. Bộ nhớ thí nghiệm append-only

Mọi lần chạy `scripts/answer_forensics.py` được ghi tự động vào
`knowledge/vothuong/experiments.jsonl` với top ID, counts, artifact và oracle
`heuristic` (bị đánh dấu thiên lệch). `scripts/experiment.py` ghi cả package
gate và kết quả leaderboard; leaderboard là oracle thật. Các review nguồn và
bài học dùng `scripts/vothuong_log.py`. Cập nhật không sửa lịch sử: thêm delta
mới và trỏ `supersedes` về ID cũ.

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest tests\test_experiment_logbook.py tests\test_answer_forensics.py -q
python scripts\vothuong_log.py recent 20
```

Log repo-local không nhập các event của dự án khác từ `~/.claude/vothuong` và
không chứa credential. Nó là bằng chứng vận hành, không tự biến heuristic/dev
thành nhãn đúng và không tự upload submission.

## 40. HHV BOT activity-segment repair v197

q1007 của v192 đọc tổng tài sản từ bảng phân khúc theo địa bàn cho 2021/2022
và gắn nhầm chúng thành tài sản BOT. Hai BCTC có bảng phân khúc theo hoạt động
riêng với cột `Dự án BOT`: `HHV...2021|2563` là
`32.355.512.700.711` VND và `HHV...2022|2450` là
`33.657.835.517.377` VND. Tỷ trọng bốn năm 2021/2022/2024/2025 lần lượt là
`95,27/94,40/90,78/88,07%`; trung bình `92,13%`.

```powershell
python scripts\build_hhv_bot_segment_assets_candidate.py
python scripts\package_submission.py `
  sub_top123_candidate_v197_hhv_bot_segment_assets `
  --out sub_top123_candidate_v197_hhv_bot_segment_assets.zip `
  --referenced-only
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v197_hhv_bot_segment_assets `
  --archive sub_top123_candidate_v197_hhv_bot_segment_assets.zip `
  --expected-sha256 7D56579F1FB8D1F44CFCF2893141E4D2B398B72E31637192CDBD59DB03520FC0 `
  --require-data-derived-labels `
  --allow-relevant-table-order
```

Gate PASS 1.012/1.012 ở bốn runtime, source 2.203/2.203, panel
4.783/4.783, 181 test + 25 subtest. ZIP referenced-only có 1.935 entry,
1.248.141 byte; hai lần dựng byte-identical. Artifact chưa được BTC đo.

## 41. Runtime scalar-round hardening v216

Log public báo đúng một câu `AttributeError`. Audit receiver kiểu dữ liệu mở
rộng phát hiện q368 là chương trình duy nhất còn gọi `.round(2)` trực tiếp trên
kết quả scalar của `Series.mean()`. NumPy scalar có method này nhưng Python
`float` không có; v216 chuẩn hóa kết quả reduction về `float` rồi dùng dòng
`round(float(result), 2)` vốn đã tồn tại. Thay đổi chỉ nằm trong
`pandas_query` q368; toàn bộ answer, evidence và relevant tables giữ nguyên.

```powershell
python scripts\build_v216_q368_scalar_round_hardening.py
python scripts\audit_runtime_attribute_hazards.py `
  sub_top123_candidate_v216_q368_scalar_round_hardened
python scripts\package_submission.py `
  sub_top123_candidate_v216_q368_scalar_round_hardened `
  --out sub_top123_candidate_v216_q368_scalar_round_hardened.zip `
  --referenced-only
.venv-grader\Scripts\python.exe scripts\final_release_gate.py `
  sub_top123_candidate_v216_q368_scalar_round_hardened `
  --archive sub_top123_candidate_v216_q368_scalar_round_hardened.zip `
  --expected-sha256 8029D3475BE060CFA0F57A883A609B418304C6BBAFCF00C22BF262714C1857F7 `
  --require-data-derived-labels `
  --allow-relevant-table-order `
  --out build\release_gate\sub_top123_candidate_v216_q368_scalar_round_hardened\release_strict.json
```

Strict release gate PASS: bốn runtime đều 1.012/1.012, source 2.255/2.255,
panel 4.783/4.783, data-derived-label audit 53/53 sạch, 377 test + 25
subtest. ZIP referenced-only có 1.890 entry, 1.218.138 byte và SHA-256
`8029D3475BE060CFA0F57A883A609B418304C6BBAFCF00C22BF262714C1857F7`.
Candidate chưa được BTC đo; cảnh báo server không tiết lộ ID nên q368 vẫn là
giả thuyết runtime có bằng chứng mạnh, chưa được gọi là sửa trúng public.

Sau khi khóa candidate, `audit_runtime_attribute_hazards.py` được thêm cờ
`--fail-on-findings` và tích hợp thành hard gate mặc định trong
`final_release_gate.py`. Báo cáo tương ứng nằm tại
`runtime_attribute_hazards.json`; lần chạy tích hợp trên v216 đạt 0 finding.

## Candidate v224: audit từng câu và provenance exception fail-closed

v224 giữ public champion/replay ở v217 và không tự nhận điểm. Candidate này có
ledger `docs/INDIVIDUAL_QUESTION_AUDIT_20260827.md` phủ đủ ID 1..1.012;
`scripts/verify_individual_question_audit.py` kiểm tra coverage, registry ID và
SHA của `submission.json`. Năm metadata warning đã giám định trên q224/q714/q764
được khóa bằng payload đầy đủ trong `config/v224_provenance_allowlist.json`;
warning mới, payload thay đổi hoặc exception thừa đều làm gate fail.

Artifact referenced-only có 1.885 member, 1.214.006 byte, SHA-256
`45268D7C7BE4CA500678536B9304C590BB0D55787C99FD3BF60F72DE561EDFA6`.
Báo cáo `build/v224_final_release_gate.json` ghi 25 command PASS, bốn runtime
1.012/1.012, source 2.255/2.255, panel 4.789/4.789, 409 test + 25 subtest và
hash v206/v207 không đổi trước–sau.

## Candidate v225: rollback q98 theo masthead vật lý

Vòng scope audit sau đó chứng minh tên container HUT 2024 bị đảo: bảng
trong container `consolidated|325` có masthead `BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG`,
còn `separate|328` có masthead `... HỢP NHẤT`. v224 vì tin suffix container
đã đổi sai q98 từ 146,47 tỷ thành 3.177,37 tỷ. v225 khôi phục toàn
bộ payload q98 đã được chứng minh của v223 và không thay đổi câu khác.

Artifact referenced-only v225 có SHA-256
`9E452A3814CB5DA5C62493B5B902ABF5E42F78D4ABC745F270F7E00A6CDF8888`;
`submission.json` có SHA-256
`45BCF57FDB5CF06ACD0E0669CFA88166E376130815FF644AEA88A71154DA64B0`.
`build/v225_final_release_gate.json` PASS bốn runtime 1.012/1.012, source
2.255/2.255, panel 4.789/4.789, 409 test + 25 subtest; bảy metadata warning
q98/q224/q714/q764 được khóa exact trong
`config/v225_provenance_allowlist.json`. Hash v206/v207 vẫn bất biến.
