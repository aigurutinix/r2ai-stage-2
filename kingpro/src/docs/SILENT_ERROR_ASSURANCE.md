# Silent-error assurance

Một chương trình chạy thành công không chứng minh đáp án đúng. Những câu chưa
thấy lỗi được giữ ở trạng thái **unfalsified, not proven** cho tới khi có bằng
chứng độc lập; không dùng confidence tự khai của LLM làm oracle.

## Trạng thái

- `falsified`: tái tính độc lập từ nguồn vật lý cho kết quả khác.
- `challenged`: lineage thiếu, metadata xung đột, selector gần hòa hoặc các
  đường giải bất đồng; phải review, chưa được tự động đổi đáp án.
- `source_confirmed`: company/year/scope/table/row/column/unit/formula/answer
  đã được đối chiếu với nguồn vật lý và ghi vào ledger.
- `unfalsified_not_proven`: các phép thử hiện có chưa tìm thấy lỗi nhưng còn
  chung semantic gate hoặc chung nguồn chuẩn hóa.
- `under_tested`: thiếu ít nhất một lớp kiểm tra quan trọng.

## Bằng chứng hiện tại (2026-08-28)

- Tổng corpus: 1.012 câu.
- Ledger nguồn bền vững: 218 câu đã đóng (156 no-change, 59 chờ mutation/
  cleanup); public-100 và batch-150 đều còn 0 câu chưa đóng.
- Typed-plan/dual execution: compiler nhận 106 câu; 106/106 khớp giữa compiler,
  typed arithmetic, Pandas sandbox và tập source cell; 906 explicit refusal.
- Metamorphic compiler audit: 106/106 bất biến khi đảo cube, đảo registry và
  chèn một cell metric thật dưới ticker/năm nhiễu; 0 disagreement.
- Falsification batch gần nhất: 10/10 đáp án tái tính khớp; q749 thiếu compact
  manifest, q780 xung đột typed-factor/unit, q401 có selector gap 0,040385 điểm %.
- Local V272 đã cleanup q749/q780 trên V269 mà không đổi byte `submission.json`:
  q749 factor qua pandas được khóa thành `[380, 450276]`; q780 factor string-source
  về 1. Full source gate 2.257/2.257 cells, ba runtime mode cho q749/q780 và
  typed-plan strict 106/106 đều pass. Chưa package/nộp; q401 vẫn `challenged`.
- Audit dtype-inference mới quét 811 manifest/6.534 cells của V272: q749 đã qua
  cả current và latent inference; phát hiện q986 blank→NaN khác string→0 nhưng
  answer 2016 ổn định ở cả hai grader mode. q986 được source-confirm và queue
  parser hardening; q429/q432 là legacy schema không có `typed_factor`, được
  ghi `unsupported`, không bị gọi nhầm là numeric finding.
- Local V274 kế thừa V272 và chỉ harden code q986: non-string NaN được chuẩn
  hóa thành 0 trước khi chọn max. Không đổi answer/docs/tables/evidence. Full
  string/typed/official runtime đều 1.012/1.012, source gate 2.257 cells và
  typed-plan 106/106 pass. V274 chưa package/nộp và không thay trạng thái public.
- V275 residual audit đóng nốt 32/32 câu: 29 no-change, q613/q627 cleanup-only,
  q638 answer change -56.62→5.74, 0 ambiguous. q638 dùng nhầm disclosure
  “cam kết cho thuê” thay cho “cam kết thuê”; raw report và physical CSV đều
  chứng minh operand 2022 đúng là 25,779,332,206.
- Risk matrix gần nhất trên answer vector cùng SHA với V269: 521 `under_tested`,
  491 `unfalsified_not_proven`. Các tập đếm có thể giao nhau với ledger/106 câu,
  nên tuyệt đối không cộng chúng như coverage độc lập.

## Giới hạn độc lập

Typed oracle không dùng `compiler.answer` hay expression Pandas, nhưng vẫn dùng
`compiled.metric`/topology và cùng đọc `FinancialCube`. Nó là kiểm chứng số học
khác implementation, **không phải blind semantic solver**. Metamorphic audit
hiện kiểm tra thứ tự cube/registry và distractor isolation; chưa hoán vị hàng/cột
trong mọi CSV vật lý.

Blind solver thật phải tự suy ra metric/company/year/scope, tự retrieve từ bảng
vật lý, không nhìn plan/code/answer của solver A, rồi mới so kết quả và lineage.
Cho tới khi đường này tồn tại, `0 mismatch` không được diễn giải thành `đúng`.

## Quy tắc mutation

Chỉ đổi answer khi có tái tính độc lập từ nguồn vật lý và giải thích được mọi
hard negative gần nhất. Static warning, LLM voting, public-neutral pair hoặc
agreement giữa các đường có chung dependency chỉ được quyền mở queue review.

## Công cụ và artifact

- `scripts/audit_typed_plan_dual_execution.py`
- `build/typed_plan_dual_execution_v269.json`
- `scripts/audit_compiler_metamorphic.py`
- `build/v272_compiler_metamorphic_v269.json`
- `scripts/audit_silent_batch_falsification.py`
- `build/v261_silent_batch_c_falsification.json`
- `build/v272_q749_q780_cleanup_gates.json`
- `build/v272_typed_plan_dual_execution.json`
- `sub_v272_q749_q780/`
- `sub_v274_q986_nan/`
- `build/v274_q986_nan_guard_gates.json`
- `build/v275_residual_source_adjudication.json`
- `scripts/build_silent_error_risk_queue.py`
- `scripts/audit_manifest_dtype_inference.py`
- `build/v273_manifest_dtype_v272.json`
- `knowledge/vothuong/question_source_verdicts.json`
