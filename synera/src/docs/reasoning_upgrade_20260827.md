# Source-grounded reasoning upgrade, 2026-08-27

## Scored baseline

The best observed Execution Accuracy is **0.6937**, not a local estimate.
`submission_vn53.zip` remains the protected baseline (SHA-256
`f30b3890ac00af812e39eef1cfe234ca4146f083020592bd8f5047524e67cd67`).
The live history in `runs/reasoning_selector/remote_history_before_release.json`
confirms VN64 at .6917 and VN65/VN66 at .6937. These are the three submissions
made on 2026-08-27 Vietnam time before this audit. The subsequent VN67 upload
and official result are recorded at the end of this document.

## New local model

Qwen3.5-9B Q4_K_M is available at
`D:\Road2AI-models\Qwen3.5-9B-Q4_K_M.gguf`. Its pinned revision, verified hash,
base-model provenance and release date are in
`artifacts/models/qwen35_9b_manifest.json`. Download reproduction uses
`tools/download_qwen35_checkpoint.py`. The base release was 2026-03-02,
before the competition cutoff; see the [official release news](https://github.com/QwenLM/Qwen3.5#news)
and [official model card](https://huggingface.co/Qwen/Qwen3.5-9B).

The new opt-in reasoning client keeps final output separate from reasoning,
rejects truncated responses and leaves the historical no-thinking client alone.
Inference is approximately 40 tokens/s on this machine.

## Completed checks (not official accuracy measurements)

- Six source-audited controls selected the correct source coordinates. Q120
  exposed lost million-VND units across a page break; exact repeated headers
  and the previous table's explicit unit now allow conservative inheritance.
- Eighteen high-risk Easy questions produced the same numeric answers as VN53.
  Agreement with the baseline is **not** proof of correctness.
- Explicit source-unit audit: 344 cells, no answer-changing mismatch.
- Opening-date audit: three false alarms came from unpadded `1/1/YYYY` dates;
  the date parser is fixed and the submitted answers were retained.
- Panel rebuilding now distinguishes explicit nil dashes from empty OCR cells,
  and permits small amounts in explicitly scaled currency columns. Numeric
  column ordinals and ambiguous base-VND note references remain excluded.
- VN53 repairs for Q755 (total interest-sensitivity bucket) and Q821 (gross
  industry-loan denominator, not net balance-sheet loans) now live in the
  deterministic registry, preventing their loss during a clean solve.
- The guarded panel `artifacts/financial_panel_guarded_values_v5.json` replayed
  594 Hard/Template questions with **zero numeric changes and zero errors**.
  Therefore these infrastructure repairs alone do not justify another upload.
- Rolling metrics now require adjacent financial years. A missing 2023 row can
  no longer make a 2024 growth or average use 2021 as the prior year. A separate
  opt-in panel backfill recovered 6,399 prior-year cells only from explicit
  comparatives in the same consolidated report and retained source coordinates.
- The comparative panel replayed all **651 deterministic Hard, Note and
  Template questions with zero numeric changes and zero errors**. This fixes a
  latent failure mode, but does not justify an answer-changing upload by itself.
- 59 unit/regression tests pass; the source tree and tools also compile cleanly.

Reports and checkpoints are under `runs/reasoning_selector/`. The first
arithmetic audit was interrupted because mixed CSV row types caused NaN fields
to mask valid fallback columns. Its outputs are diagnostic only. The corrected
guarded run is `arithmetic_qwen35_plan.json`; all six attempted calculations
were rejected by the compiler/grounding gates, while their usable explanations
agreed with the existing formula semantics. Inputs must have non-missing values
and unique source IDs before inference.

The final disagreement audit independently confirmed the current answers for
Q11, Q58, Q153 and Q164. Q49 remained a three-table collision (selling,
administrative, or cost-by-element context); Q59 contrasted a gross face-
statement balance with the detailed note balance after provision. Neither
alternative is promotable because the question text does not prove a different
organiser-locked source table. The audit was stopped after these cases rather
than spending compute on already source-resolved alternatives.

## Promotion gate

The independent arithmetic model sees only the question and input source cells,
not VN53's answer, expression or precomputed answer fields. Inspect any changed
expression against complete source tables and the question before packaging.
Do not use confidence, model disagreement or aggregate-score probing as a
substitute for source verification. Preserve the baseline, audit the complete
1,012-row ZIP, and only upload a genuinely different source-backed candidate.
## Subsequent source audit and VN67

The following corrections were verified against raw official table cells,
not inferred from hidden answer labels or aggregate score probing:

| Question | Problem | Before | New answer |
| --- | --- | ---: | ---: |
| 495 | VGT 2021 all other short-term receivables were mistaken for the related-party subset; the correct subset changes the selected year to 2020. | 375.899885116 | 385.62271692 |
| 521 | The HDG file named 2023 repeats its 2022 statements. The genuine 2024 opening cash balance supplies 2023 closing cash; 2025 now wins. | 500.688616629 | 299.780784516 |
| 904 | The same mislabeled HDG report supplied wrong financial income. The explicitly labelled 2023 comparative in 2024 makes 2024 the maximum. | 2023 | 2024 |

Four additional source-only repairs preserve numeric answers: Q501 uses gross
overdue principal, not an allowance; Q502 uses the fund's bank-account balance,
not the net fund balance; Q506 uses tangible fixed assets code 221, not code
220; Q526 uses current-year EPS, not its prior-year comparative. All seven
new rows calculate their answers from source operands in pandas, without
`computed_answer` fields. Query mutation tests verify that selectors actually
affect the result. Comparative years require explicit opt-in and an annual
prior-period header; source document years are never relabelled.

English VND-million unit parsing is also fixed, but no existing submission
operand matched that bug, so it is not claimed as a current score gain.

The report-period audit flagged FPT2015, NVL2019 and SAB2017 too. Their table
headers/next-year comparatives showed correct data despite stale titles, so
they were NOT changed. Only HDG2023 had the duplicated 2022 values (all 47
nonzero matching panel metrics) and contradictory genuine 2024 comparatives.
Source-model review outputs in `note_sources_9b.json` were used only to flag
locations for manual verification; approximate model answers were rejected.

VN67 is built by `tools/build_source_repair_release.py`. Its exact baseline is
VN53 and its SHA-256 is
`5b1c8be1b5904815eb6aaadbcbfc396f14325d6a1ad497e10431d1796bde5356`.
The manifest is `runs/reasoning_selector/vn67_manifest.json`. It preserves
1,005 JSON rows and their CSV bytes, changes seven source/query rows, and
changes only three numeric answers. Existing releases are never overwritten.

Verification completed:

- 66 unit/source-regression tests pass.
- Full 651-question deterministic replay: exactly the three expected answer
  changes, zero errors (`vn67_full_replay.json`).
- One complete ZIP audit: 1,012 rows, 1,012 fresh CSV query executions matching
  submitted answers, all source table references exist, no missing/orphan CSVs
  (`vn67_release_audit.json`, `ok: true`).

These checks prove execution consistency, not hidden-answer accuracy.
### Official VN67 result

VN67 was submitted once as **ID 3724**, finished, and received **0.6937**
Execution Accuracy and Answer Accuracy: **no observed score improvement**.
All retrieval metrics also stayed identical. The official scoring archive has
only aggregate scores and a successful exit code, not per-question labels.

Downloaded `vn67_prediction_result.zip` was compared with the local release:
all **1,013 local members are byte-identical**, including the changed answers
for Q495/Q521/Q904. Only organizer metadata and the `data/` directory entry
were added. Thus this is not an old/wrong ZIP upload or a local packaging-cache
problem. The raw scoring ZIP and structured `vn67_official_result.json` are
saved under `runs/reasoning_selector/`.

The scored best is still the protected VN53 at 0.6937. VN67 contains verified
source/query repairs but is NOT advertised as a higher-scoring version. Four
submissions have now been used on 2026-08-27 Vietnam time; six nominal daily
slots remain, subject to the server's eligibility check. Do not use these to
probe answer combinations or isolate hidden correctness from aggregate scores.
The next experiment was VN68, described below.

## VN68: signed temporal change

An opt-in `signed_temporal_changes` policy interprets explicit temporal
"biến động/thay đổi" as new-minus-old, rather than an absolute magnitude.
Unordered "chênh lệch giữa" questions are left unchanged. Source cells and
units are preserved: Q624 STB foreign-currency cash, Q642 HDB held-to-maturity
government bonds, and Q649 HSG long-term prepaid rent are decreases.

The 435-question Template replay changes exactly those three answers, with
zero errors. All 69 tests pass. The complete release audit executes all 1,012
CSV-backed queries and matches all submitted answers; no missing references
or orphan CSVs. Reproduction uses `tools/build_reviewed_variant.py` and
`runs/reasoning_selector/vn68_plan.json`, which contains source coordinates
and expressions, not expected numeric answers.

VN68 SHA-256:
`be29ba69e36f985c6783f3594070705d72ae6a9615b0111528e5a9ebb897aba2`.
Official submission **3725** finished at **0.6996 Execution Accuracy** and
**0.6996 Answer Accuracy**, an observed increase of **0.0059**. All retrieval
scores stayed unchanged. This is now the scored best; VN53/VN67 remain intact.
Five nominal daily slots remain after this fifth upload on 2026-08-27.
Do not infer individual hidden labels from the aggregate improvement.

## VN69: broader ordered comparison was NOT promoted

The separate opt-in `ordered_comparisons` policy treated explicit "A chênh
lệch so với B" as signed A-B for positive balance operands. Six numeric
changes retained original source cells: Q742, Q758, Q781, Q790, Q793, Q799.
The 435-question replay showed exactly these changes and zero errors; all 72
tests and the 1,012-query release audit passed.

Official submission **3727** nevertheless scored **0.6937**, down from VN68's
0.6996. **Reject this policy for the best release.** The untouched VN68 is the
current best. A further temporal-contrast extension exists only as an opt-in
local experiment; do not upload ambiguous sign variants in response to this
regression. Four nominal daily slots remain after six uploads.

## VN70: literal ratio units and gross-sales basis

Built from **VN68**, not the regressed VN69. The new opt-in
`literal_ratio_contracts` policy changes two answers:

- Q672: the question asks for a ratio without percent units. Return short-term
  prepayments divided by long-term prepayments, without an implicit x100.
- Q711: related-party sales share uses the gross total-sales denominator from
  the same note. Gross sales reconcile exactly to net sales plus disclosed
  reductions; do not combine a gross numerator with net-sales denominator.

The local model independently confirmed these calculations from full source
tables in `ratio_full_tables_confirmation.json`. Release calculations use
exact source numbers and pandas, not approximate model answers. Regression
checks also ensure an explicit percent request restores x100, and an explicit
net-sales question cannot switch to a gross denominator.

All **75 tests** pass. Full Template replay (435 questions) has exactly two
numeric changes and zero errors. Full ZIP audit again has **1,012/1,012**
CSV-replayed answers matching, no missing references, no orphan CSVs.

VN70 SHA-256:
`e9d0cab3e34f1bcf5d5f193866a13da6be53a900bed901e83067d43135e40d47`.
Official submission **3728** scored **0.7016 Execution Accuracy** and
**0.7016 Answer Accuracy**, the new best. Improvement over the initial 0.6937
is 0.0079 (0.79 percentage points); it is not a claim of 0.8 performance.
Three nominal daily slots remain after seven uploads on 2026-08-27.
