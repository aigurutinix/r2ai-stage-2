# Leaderboard Checkpoints

## Execution 0.2806 - canonical direct24 + semantic4

- Date: 2026-08-13
- Branch: `improve_baseline_kien`
- Source commit: `f59797907c1f343d50ff1fbe3f8f24559568b371`
- Retrieval: `artifacts/retrieval_p2_canonical_qualified_hybrid_w010.jsonl`
- Codegen: `artifacts/codegen_result6_canonical_direct24_semantic4_w010.jsonl`
- Submission: `artifacts/submission_p2_canonical_result6_direct24_semantic4_w010/submission.zip`
- Codegen SHA-256: `74c295c5b145ab79ba543b6f505eb72a458ba88eeb984886861b23f28189330a`
- Submission SHA-256: `5999f798565fcc7e9e5545f50a6701737ef846b75bc7c181a78b7bb3e6704176`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4761,
  "DOCS_F2MACRO": 0.8918,
  "TABLES_PRECISION": 0.316,
  "TABLES_RECALL": 0.6629,
  "TABLES_MRR5": 0.6359,
  "DOCS_PRECISION": 0.9487,
  "DOCS_RECALL": 0.8883,
  "DOCS_MRR5": 0.9654,
  "ANSWER_ACCURACY": 0.2806,
  "EXECUTION_ACCURACY": 0.2806
}
```

The submission keeps the audited `0.2648` core and canonical retrieval, then
overrides 24 direct/high-confidence metrics and four semantic metric cases.
Changed question IDs:

```text
4, 61, 72, 101, 109, 123, 176, 201, 255, 265, 278, 289, 307, 310,
337, 665, 676, 698, 699, 718, 719, 739, 743, 746, 762, 778, 795, 800
```

Compared with the preceding `0.2648` submission, this checkpoint gains 16
correct executions out of 1,012 questions. All 1,012 pandas expressions replay
locally and the source test suite passes (`173 passed`). Generated artifacts
remain ignored by Git; the hashes above identify the exact local files.

## Execution 0.2866 - canonical metric dictionary v2 audited6

- Date: 2026-08-20
- Branch: `improve_baseline_kien`
- Source commit: `ef77d55fcdf283750015018488af1524163b3164`
- Retrieval: `artifacts/retrieval_p2_canonical_v2_qualified_hybrid_w010.jsonl`
- Rule codegen: `artifacts/codegen_rule_canonical_v2_w010_k15.jsonl`
- Kaggle output: `kaggle/results-7/codegen_canonical_v2_sel7b.jsonl`
- Audited blend: `artifacts/codegen_result7_canonical_v2_audited6_w010.jsonl`
- Submission: `artifacts/submission_p2_canonical_v2_result7_audited6_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-canonical-v2-w010.zip`
- Kaggle notebook: `r2ai-qwen2-5-coder-7b-canonical-v2-w010.ipynb`
- Retrieval SHA-256: `c7fa88ad54294eb03881569126e24033e2ac73c8f111ae867110c1c44b03c28c`
- Kaggle-output SHA-256: `f2e03eb7aee290cf09d20f8aaa1ac5a04dbae7d15e090c5585ae8e95f0de2f96`
- Audited-blend SHA-256: `ee1c189c5968f15c2abe8e98677296fc07fc745c87cf3ccf26cf65d7f2534314`
- Submission SHA-256: `5fb8fb48a4bfa5e539baebf4e108bc8c7e8364966c43a9db25ff67a1fd0a7147`
- Payload SHA-256: `e0f681a0de9978da6a7081dbd9b423633945ddf96441e2a353b6aa4924b7cb41`

V2 registers 139 metrics and links 791/1,012 official questions (78.16%), up
from 534/1,012 (52.77%). It adds structured qualifiers and exact row identity.
On the 40-question synthetic validation set, rule execution rises from 0.700
for canonical v1 to 0.725 for v2. Kaggle inference produced 868 executable
rows, including 120 rows that failed in the 0.2806 checkpoint. A semantic audit
found that confidence-only selection admitted many false positives on nested
questions, so the earlier `safe20` artifact is deprecated and must not be
submitted. The final conservative blend preserves the complete 0.2806
checkpoint and adds only six table-verified answers: `589, 591, 635, 757, 830,
838`. All 1,012 submission expressions replay locally; the source suite passes
(`183 passed`).

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4777,
  "DOCS_F2MACRO": 0.8945,
  "TABLES_PRECISION": 0.3192,
  "TABLES_RECALL": 0.6638,
  "TABLES_MRR5": 0.6427,
  "DOCS_PRECISION": 0.9484,
  "DOCS_RECALL": 0.8917,
  "DOCS_MRR5": 0.9644,
  "ANSWER_ACCURACY": 0.2866,
  "EXECUTION_ACCURACY": 0.2866
}
```

Against the `0.2806` checkpoint, canonical v2 gains all six audited executions
(`6/1,012 = 0.00593`) while table F2 rises by `0.0016` and document F2 rises by
`0.0027`. This confirms the canonical metric dictionary v2 milestone: its
retrieval changes improve both evidence metrics, and every accepted answer
change executes correctly on the official grader.

## Execution 0.2866 - formula operand retrieval, answers preserved

- Date: 2026-08-21
- Branch: `improve_baseline_kien`
- Retrieval: `artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl`
- Codegen: `artifacts/codegen_result7_canonical_v2_audited6_w010.jsonl`
- Submission: `artifacts/submission_formula_operand_retrieval_best2866_answers_w010/submission.zip`
- Retrieval SHA-256: `ceede9c3b72675b67e3030746c0644fc034add1d415e5af1f4d9cbbcad342c27`
- Codegen SHA-256: `ee1c189c5968f15c2abe8e98677296fc07fc745c87cf3ccf26cf65d7f2534314`
- Submission SHA-256: `9a1c5608960ea88d46241e7b270c97559b1b04143af932fa9f26deba985a8f4c`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4892,
  "DOCS_F2MACRO": 0.9068,
  "TABLES_PRECISION": 0.3417,
  "TABLES_RECALL": 0.6755,
  "TABLES_MRR5": 0.6088,
  "DOCS_PRECISION": 0.9488,
  "DOCS_RECALL": 0.9048,
  "DOCS_MRR5": 0.9733,
  "ANSWER_ACCURACY": 0.2866,
  "EXECUTION_ACCURACY": 0.2866
}
```

This checkpoint preserves all 1,012 answers, pandas expressions and execution
evidence from the `0.2866` canonical-v2 checkpoint. Only table/document
retrieval changes: TABLES_F2 gains `0.0115` and DOCS_F2 gains `0.0123` without
an execution regression. The all-question Qwen `k=15` run is not a valid answer
checkpoint: it improved TABLES_F2 to `0.4924` and DOCS_F2 to `0.9126`, but
changed 438 previously successful answers and reduced execution to `0.2747`.

Consensus `n=3` was also tested on deterministic-empty questions. A strict
blend accepted only IDs `24` and `996` after `3/3` agreement, confidence 95,
retrieval coverage and route-shape checks. Leaderboard execution remained
`0.2866`, proving both additions wrong. The errors are schema-linking failures:
ID 24 selects the parent `Vay dài hạn` row instead of the named counterparty;
ID 996 canonicalizes `TSCĐ vô hình - giá trị còn lại` as parent
`fixed_assets`. Do not resubmit `submission_formula_consensus_safe2_w010`.

## Execution 0.2964 - tranhuy structured selection and guarded exact cells

- Date: 2026-08-22
- Source branch: `tranhuy` at `75e5269`
- Imported codegen: `artifacts/codegen_tranhuy_02964_stockflow_exact_v11.jsonl`
- Scored submission: `artifacts/tranhuy_02964_stockflow_exact_v11_oldretr_k5/submission.zip`
- Codegen SHA-256: `aa14d8747b5f856067cfb4edbddd8142469578acadb26c17b2d92658fa8e3b65`
- Scored submission SHA-256: `6773532be7dbe038310b9a0663090ebf17878111a8b27b53403cd24de5f23e9a`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4763,
  "DOCS_F2MACRO": 0.893,
  "TABLES_PRECISION": 0.3146,
  "TABLES_RECALL": 0.6639,
  "TABLES_MRR5": 0.6359,
  "DOCS_PRECISION": 0.9487,
  "DOCS_RECALL": 0.8897,
  "DOCS_MRR5": 0.9654,
  "ANSWER_ACCURACY": 0.2964,
  "EXECUTION_ACCURACY": 0.2964
}
```

The artifact directory retains the older experiment label `02925`; the score
above is the leaderboard result confirmed for this exact ZIP. The branch adds
semantic-grounded structured selection, typed nested IR, deterministic
compilation/unit guards, and conservative exact-cell/canonical-metric
overrides. The final audit layers each accepted one change: exact-cell IDs
`53` and `38`, then stock/flow ID `841`.

Compared with the previous `0.2866` codegen, this checkpoint changes 29 pandas
queries and 27 answers, reducing zero/empty answers from 240 to 223. On the
hidden 506-question execution subset it gains five net correct answers. All
1,012 answers, expressions and evidence entries have been copied locally with
hashes so this result remains reproducible independently of the sibling
`artifacts_tranhuy` directory.

## Execution 0.2964 - combined best with formula-operand retrieval

- Retrieval: `artifacts/retrieval_canonical_v2_formula_operand_exact_w010.jsonl`
- Codegen: `artifacts/codegen_tranhuy_02964_stockflow_exact_v11.jsonl`
- Submission: `artifacts/submission_tranhuy_02964_answers_formula_operand_retrieval_w010/submission.zip`
- Submission SHA-256: `17750e0290d181ce13d35ef87b589ee6bdf9bcf9fa4831a6c749e8744e8e93b7`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4901,
  "DOCS_F2MACRO": 0.9079,
  "TABLES_PRECISION": 0.3413,
  "TABLES_RECALL": 0.677,
  "TABLES_MRR5": 0.6088,
  "DOCS_PRECISION": 0.9489,
  "DOCS_RECALL": 0.9061,
  "DOCS_MRR5": 0.9733,
  "ANSWER_ACCURACY": 0.2964,
  "EXECUTION_ACCURACY": 0.2964
}
```

This candidate preserves all 1,012 answers, pandas expressions and evidence
entries from the leaderboard-confirmed `0.2964` submission. It changes only
retrieval: 726 table lists and 359 document lists versus the old retrieval.
Against the current best retrieval checkpoint (`TABLES_F2=0.4892`,
`DOCS_F2=0.9068`), only 15 table lists and five document lists differ because
the new expressions require additional execution evidence.

Build guards pass: 1,012 unique entries, all expressions compile, every answer
replays exactly from the packaged evidence, 1,553 CSV files are included using
official 1-based `<table>` line numbers, and `unzip -t` reports no errors.
The leaderboard confirms that execution remains `0.2964`. Relative to the
scored old-retrieval package, TABLES_F2 rises by `0.0138` and DOCS_F2 rises by
`0.0149`. It also exceeds the previous retrieval-best checkpoint by `0.0009`
TABLES_F2 and `0.0011` DOCS_F2 while gaining five net executable answers. It
is the direct base of the `0.3004` restore4 checkpoint below.

## Execution 0.3004 - combined best plus canonical-v2 restore4

- Base: `artifacts/codegen_tranhuy_02964_stockflow_exact_v11.jsonl`
- Restore source: `artifacts/codegen_result7_canonical_v2_audited6_w010.jsonl`
- Restored IDs: `589, 591, 635, 838`
- Codegen: `artifacts/codegen_tranhuy_02964_plus_canonical_restore4_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_02964_plus_canonical_restore4_w010/submission.zip`
- Codegen SHA-256: `4b524bdb3b77d83115915cc71f3bb0928f55c95a08b3dea80c1251f7bb8f2d69`
- Submission SHA-256: `cdeed91318c8d6eee2cc009ca4b42e5f67acf547c5072361dfb5cbae3107a7e4`

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4902,
  "DOCS_F2MACRO": 0.9079,
  "TABLES_PRECISION": 0.3416,
  "TABLES_RECALL": 0.677,
  "TABLES_MRR5": 0.6088,
  "DOCS_PRECISION": 0.9489,
  "DOCS_RECALL": 0.9061,
  "DOCS_MRR5": 0.9733,
  "ANSWER_ACCURACY": 0.3004,
  "EXECUTION_ACCURACY": 0.3004
}
```

The Trần Huy codegen changed these four table-audited canonical-v2 answers
back from `ok` to `failed`. This batch restores them on top of the official
combined `0.2964` checkpoint. It changes exactly four answers, expressions and
evidence lists; relevant documents are unchanged and relevant tables change
only for ID `838`. All 1,012 expressions compile and replay, the ZIP contains
1,560 evidence CSV files, and `unzip -t` passes. The leaderboard confirms two
net additional correct answers on the 506-question scored subset, increasing
execution by `0.0040`. TABLES_F2 also rises by `0.0001`, while DOCS_F2 is
unchanged. This is now the official best checkpoint.

## Superseded candidate - schema linking v3 w010

- Date: 2026-08-21
- Branch: `improve_baseline_kien`
- Retrieval: `artifacts/retrieval_schema_linking_v3_w010.jsonl`
- Rule codegen: `artifacts/codegen_rule_schema_linking_v3_w010_k15.jsonl`
- Payload folder: `artifacts/kaggle-payload-schema-linking-v3-w010`
- Upload ZIP: `artifacts/kaggle-payload-schema-linking-v3-w010.zip`
- Notebook: `r2ai-qwen2-5-coder-7b-schema-linking-v3-w010.ipynb`
- Retrieval SHA-256: `1072d9ea8c8b920597526d279b2c24276dee8888f0f111b4ebcdc456be528259`
- Payload ZIP SHA-256: `44e296fa96b50a47e215af7027ef62ba7509bc0060692c8faf43a365a3615d7a`
- Notebook SHA-256: `be11386599e4238adab0e9f39e0ed90638c2ff35ad89f8bd8baaa21a3abc2f82`

This candidate fixes the two schema-linking root causes exposed by the failed
`safe2` experiment. Named counterparties are now part of requirement identity,
and `TSCĐ vô hình - giá trị còn lại` maps to the dedicated
`intangible_fixed_assets` child instead of parent `fixed_assets`. Generic
requirements use clean canonical aliases so output wording cannot poison the
row-score cache.

Offline guards:

- official retrieval: 1,012 rows, zero empty candidate lists;
- evidence complete at depth 20: `714 -> 715`;
- evidence complete at Qwen `k=15`: `691 -> 692`;
- only 28 questions change top-5 order and 17 change the top-5 set;
- IDs `24`, `534` and `996` have complete evidence inside `k=15`;
- hard-negative row linking: `76/76` top-1, MRR `1.000`, recall@5 `1.000`;
- 288-question formula eval is unchanged from operand-exact;
- source suite: `211 passed`.

This candidate was later built and scored as the failed ablation below. It did
not improve execution over `0.2866` and is superseded by the combined `0.2964`
checkpoint above.

## Failed ablation - results-10 schema linking v3 audited5

- Date: 2026-08-21
- Raw Qwen output: `kaggle/results-10/codegen_schema_linking_v3_consensus_empty_sel7b_k15_n3.jsonl`
- Audited blend: `artifacts/codegen_result10_schema_linking_v3_audited5_w010.jsonl`
- Submission: `artifacts/submission_result10_schema_linking_v3_audited5_w010/submission.zip`
- Raw-output SHA-256: `79ed71705dcfa8c070501f757c28614b9d75b1b4a810c67e98a394ea4e638155`
- Audited-blend SHA-256: `3acefc1f4900c23a15c8b6e8cbb965ad8bec93abf7d89176917047b8c828c4bc`
- Submission SHA-256: `d229bf51a5d3e368ea0a33defbd7bdad9b6b15739e310833aacdbca0bcd47eea`

Do not submit the raw Qwen output. Relative to the official `0.2866` codegen
checkpoint it changes 524 answers, while 177 previously executable rows become
failed. The audited blend preserves every existing answer and replaces only
five previously failed rows: `24`, `440`, `562`, `574`, and `996`.

The five accepted answers were traced to their exact report cells and formulas:

- `24`: named HAGL counterparty long-term borrowing, not the parent borrowing row;
- `440`: select DIG/2024 by maximum D/E, then calculate EBIT/interest expense;
- `562`: only DCM/2022 clears net margin above 10%, then calculate CFO/current liabilities;
- `574`: GEE and GEX have positive CFO in 2022-2024 and GEE has the larger 2024 net margin;
- `996`: exact intangible-fixed-assets totals for 2016, 2018, 2020, and 2021.

The audit rejected executable but semantically wrong rows, including `364`
(returns the selector rather than accrual ratio), `407` (uses 2021 instead of
the next year), `417` (wrong selector and target), `717` (total assets divided
by itself), `805` (parent receivables instead of related parties), and `857`
(uses prior-year columns 2015-2017).

Final guards: 1,012 unique IDs, zero empty document/table lists, exactly five
answer/query deltas versus the `0.2866` submission, all 1,012 expressions
compile and replay from packaged CSV evidence, and the ZIP layout passes
`unzip -t`.

Leaderboard execution remained `0.2866`. With 506 scored questions, one net
additional correct answer would move the rounded score to about `0.2885`, so
this submission produced zero observable net gains. Because the organizer's
506-ID subset is hidden, this does not distinguish answers that are wrong from
changed IDs that are outside the scored subset. Do not resubmit this audited5
artifact. It is superseded by the combined `0.2964` checkpoint.

## Execution 0.3083 - deterministic exact year ranking

- Date: 2026-08-22
- Base checkpoint: execution `0.3004`
- Rule output: `artifacts/codegen_rule_canonical_v2_formula_operand_yearrank_exact_w010_k15.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03004_plus_yearrank_exact10_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03004_plus_yearrank_exact10_w010/submission.zip`
- Accepted IDs: `842, 850, 878, 884, 889, 904, 948, 960, 997, 1000`
- Rule SHA-256: `8724be8e46e10ea7b4ba79735a3d185c3cf6949908a940b31ada22aec7715cd4`
- Codegen SHA-256: `c0e317f271303190203c9ff736eaa2aad9d62a87225a3e8e01cceecf17460ff0`
- Submission SHA-256: `e4aa4fd2eb797e3274d4e5b9869b52c8bae09a7e4769e1f39393d21ad4898203`

This batch implements an explicit ranking plan with `dimension=year`,
`projection=year`, and `direction=max|min`. The solver resolves the same
canonical metric for every requested year, requires strong period evidence,
rejects parent/child substitutions, reads every evidence cell in the submitted
expression, and fails closed on missing years, ambiguity, unsupported ratio
selectors, and ties.

Of the 33 failed year-output ranking rows in the `0.3004` codegen, the first
pass returned 13. Exact-cell audit exposed three unsafe results: ID `826`
dropped a ratio denominator, ID `850` mixed total cost of goods sold with
forest-cost detail rows, and ID `959` used an equity child row. After the exact
identity guard, `826` and `959` fail closed and `850` resolves all five years
to VAS code `11`. Ten rows remain accepted, each with complete period evidence
and a successful pandas replay.

The final package changes exactly ten answers and expressions versus the
leaderboard-confirmed `0.3004` submission. All 1,012 expressions compile and
replay from the packaged evidence, official 1-based `<table>` line numbers are
used, the ZIP contains 1,586 CSV files, and `unzip -t` reports no errors.

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4902,
  "DOCS_F2MACRO": 0.9079,
  "TABLES_PRECISION": 0.3416,
  "TABLES_RECALL": 0.677,
  "TABLES_MRR5": 0.6088,
  "DOCS_PRECISION": 0.9489,
  "DOCS_RECALL": 0.9061,
  "DOCS_MRR5": 0.9733,
  "ANSWER_ACCURACY": 0.3083,
  "EXECUTION_ACCURACY": 0.3083
}
```

The batch gains four net correct answers on the hidden 506-question subset:
`0.3004 -> 0.3083`. Retrieval metrics are byte-for-byte unchanged, confirming
that this gain comes from answer semantics rather than wider table/document
retrieval. This is the new official best checkpoint.

## Execution 0.3202 - canonical note v3 + direct year ranking v2

- Date: 2026-08-22
- Base checkpoint: execution `0.3083`
- Target batch: 16 simple year-ranking questions
- Refreshed retrieval: `artifacts/retrieval_canonical_note_v3_yearrank_v2_exact13_w010.jsonl`
- Rule output: `artifacts/codegen_rule_canonical_note_v3_yearrank_v2_w010_k15.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03083_plus_note_v3_yearrank_v2_exact13_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03083_plus_note_v3_yearrank_v2_exact13_w010/submission.zip`
- Accepted IDs: `813, 829, 852, 890, 897, 921, 936, 946, 959, 971, 978, 981, 1008`
- Rejected IDs: `907, 910, 986`
- Submission SHA-256: `a27e20365e3f4fa3a081014fe69e64eeb6b78dd4047863f119da64d3596f4a05`

Canonical note dictionary v3 adds query aliases, exact row aliases and table
context constraints for detailed note metrics such as external receivables,
merchandise inventory, short-term unearned revenue, real-estate brokerage
expense, construction payables, LPG revenue, named-counterparty payables and
related-party service revenue. The direct resolver now respects separate versus
consolidated reports, exact note context and OCR current-period headers.

The batch fails closed when any requested year is absent. ID `907` has missing
current-period share data in the extraction store; IDs `910` and `986` have a
dash/blank in the requested current-year cell. A regression guard prevents the
prior-year value from being mistaken for the current year when OCR folds the
dash into the row label.

The submission changes exactly 13 previously failed answers, expressions and
evidence sets. It does not overwrite any successful row from the `0.3083`
checkpoint. All 1,012 expressions compile and replay from the packaged CSVs,
the package uses official 1-based `<table>` line numbers, all 229 tests pass,
and `unzip -t` reports no errors.

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4913,
  "DOCS_F2MACRO": 0.9103,
  "TABLES_PRECISION": 0.3424,
  "TABLES_RECALL": 0.6783,
  "TABLES_MRR5": 0.6098,
  "DOCS_PRECISION": 0.9509,
  "DOCS_RECALL": 0.9085,
  "DOCS_MRR5": 0.9753,
  "ANSWER_ACCURACY": 0.3202,
  "EXECUTION_ACCURACY": 0.3202
}
```

The batch gains six correct answers on the hidden 506-question subset:
`0.3083 -> 0.3202`. Table F2 rises by `0.0011` and document F2 by `0.0024`,
so the targeted canonical retrieval improves both evidence quality and answer
execution. This is the new official best checkpoint.

## Execution 0.3300 - typed compositional ranking v3 exact12

- Date: 2026-08-23
- Status: official leaderboard checkpoint; base execution `0.3202`
- Target batch: 16 nested ranking questions that select by metric/formula A and
  project metric/formula B
- Retrieval: `artifacts/retrieval_tranhuy_03202_plus_compositional_ranking_v3_exact12_w010.jsonl`
- Rule output: `artifacts/codegen_compositional_ranking_v3_offset_k48_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03202_plus_compositional_ranking_v3_exact12_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03202_plus_compositional_ranking_v3_exact12_w010/submission.zip`
- Accepted IDs: `415, 440, 459, 475, 479, 491, 505, 507, 543, 556, 558, 575`
- Rejected IDs: `495, 497, 511, 524`
- Submission SHA-256: `4c674ca7e75db0e1b279ff2ea997799b7e7c1cdd96a0945fdcca6f308b92406d`

The typed planner separates the selector and projection calculations, including
their `level`, `growth` or `delta` modes. The executor resolves every entity and
period exactly, computes the selector, chooses the unique argmax/argmin, and
only then evaluates the projection. Its replay expression rechecks the winning
condition and reads all supporting evidence.

Inventory-days now declares canonical period-aware operands:
`inventory(t-1)`, `inventory(t)` and `cost_of_goods_sold(t)`. The router locks
adjacent-year reports from those offsets, while schema linking recovers years
that OCR split into separate header rows, including flattened dates such as
`3092024`. Four unresolved note/EPS/tax questions remain fail-closed.

The final artifacts differ from the `0.3202` checkpoint on exactly the 12
allowlisted entries. All 1,012 expressions compile and replay from packaged
CSV evidence, all 240 tests pass, table references use official 1-based
`<table>` line numbers, and `unzip -t` reports no errors.

Leaderboard metrics:

```json
{
  "TABLES_F2MACRO": 0.4958,
  "DOCS_F2MACRO": 0.9139,
  "TABLES_PRECISION": 0.3439,
  "TABLES_RECALL": 0.683,
  "TABLES_MRR5": 0.6101,
  "DOCS_PRECISION": 0.9509,
  "DOCS_RECALL": 0.9127,
  "DOCS_MRR5": 0.9753,
  "ANSWER_ACCURACY": 0.33,
  "EXECUTION_ACCURACY": 0.33
}
```

The batch gains five net correct answers on the hidden 506-question subset:
`0.3202 -> 0.3300`. Table F2 rises by `0.0045` and document F2 by `0.0036`.
This is the official base for median-filter planner v4.

## Checkpoint - typed median-filter planner v4 audited9

- Date: 2026-08-23
- Status: leaderboard verified; retrieval-best at unchanged execution `0.3300`
- Base checkpoint: execution `0.3300`
- Target batch: 19 failed questions containing a median-based population filter
- Final retrieval: `artifacts/retrieval_tranhuy_03300_plus_median_filter_v4_audited9_w010.jsonl`
- Rule outputs: `artifacts/codegen_median_filter_v4_k32_w010.jsonl` and
  `artifacts/codegen_median_filter_v4_k72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03300_plus_median_filter_v4_audited9_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03300_plus_median_filter_v4_audited9_w010/submission.zip`
- Accepted IDs: `369, 378, 379, 439, 441, 447, 448, 454, 455`
- Retrieval SHA-256: `20828af15abffa42fcd87270da68ea702b050778528a0e93bedad2873bdc337f`
- Codegen SHA-256: `eb2a5de41ba2d6791d9c9c94d88d28abaf3097a524a44cb6416fef753b5f2247`
- Submission SHA-256: `a9be6ad8d2188e1a4f9a6db4f939031a7aab83eea00091a1a47744e85b4d0873`

The v4 planner represents the median predicate as a typed node containing its
formula, comparison operator, temporal mode and fixed or relative period. It
supports both entity and year populations, explicit and implied prior-year
growth, and decomposed quick-ratio and inventory-days wording. Typed temporal
bounds distinguish formula evidence years from the selector interval.

The deterministic executor resolves the filter for every population member,
computes the odd/even median, filters, performs the unique argmax/argmin, and
only then evaluates the projection. The submitted expression recomputes the
median and checks every candidate dynamically, including candidates outside
the statically selected subset. Unsupported average-equity, average-fixed-
asset and multi-formula filters remain fail-closed.

The 19-question audit produced nine stable answers across contiguous retrieval
depths. The final retrieval refreshes exactly those nine rows; all other 1,003
retrieval and codegen records remain identical to the `0.3300` base. The ZIP
changes exactly nine answers, expressions, evidence sets, relevant-table lists
and relevant-document lists. All 247 tests pass, all 1,012 expressions compile
and replay from 1,690 packaged CSV files, official 1-based `<table>` line
numbers are used, and `unzip -t` reports no errors.

Leaderboard metrics:

```text
TABLES_F2MACRO      0.4996
DOCS_F2MACRO        0.9178
TABLES_PRECISION    0.3440
TABLES_RECALL       0.6872
TABLES_MRR5         0.6101
DOCS_PRECISION      0.9509
DOCS_RECALL         0.9169
DOCS_MRR5           0.9753
ANSWER_ACCURACY     0.3300
EXECUTION_ACCURACY  0.3300
```

The nine median-filter answers produced zero net execution gain on the hidden
subset, but retrieval increased table F2 by `0.0038` and document F2 by
`0.0039`, almost entirely through recall. Keep this package as the
retrieval-best `0.3300` base, but do not expand the median family further.

## Checkpoint - typed filter-then-aggregate planner v5 audited6

- Date: 2026-08-24
- Status: leaderboard verified; execution `0.3360`
- Base checkpoint: execution `0.3300`, retrieval-best median v4
- Audited backlog: 19 filter/aggregate questions
- Exact deterministic fills: `367, 375, 385, 408, 451, 484`
- Final retrieval: `artifacts/retrieval_tranhuy_03300_plus_filter_aggregate_v5_audited6_w010.jsonl`
- Rule output: `artifacts/codegen_filter_aggregate_v5_final_k72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03300_plus_filter_aggregate_v5_audited6_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03300_plus_filter_aggregate_v5_audited6_w010/submission.zip`
- Retrieval SHA-256: `3a6a0fda79b8f81639c723a3ff9c29d5b216634003ca16d3c68a0f282034c513`
- Codegen SHA-256: `2373684fd1cc78f9553129dcc417cc701b78746a41406aec66e5f5f7244bfc85`
- Submission SHA-256: `1c6f62ebc24aff0177d874fe2f1593f91547e866c1c23087b167b1fdfc0a4a43`

Planner v5 uses typed population, predicate, value and aggregate nodes. It
supports constant and median predicates, exact multi-period filters, level,
growth and delta values, formula differences, and `mean`, `sum`, `share` plus
`difference_of_means`. The executor resolves both predicate and value evidence
for every population member before filtering. Submitted expressions recompute
membership and aggregation dynamically and force reads of all audited cells.

Targeted depth-72 retrieval made 17/19 routes coverage-complete. Thirteen rows
still fail closed because at least one exact statement row is absent,
ambiguous, or semantically different; no fuzzy threshold was relaxed. The
final merge changes exactly six failed rows and preserves all 1,006 remaining
answers from the `0.3300` package. All 256 tests pass, all 1,012 expressions
compile and replay, the ZIP contains 1,718 CSV files using official 1-based
`<table>` line positions, and `unzip -t` reports no errors.

Leaderboard metrics:

```text
TABLES_F2MACRO      0.5031
DOCS_F2MACRO        0.9200
TABLES_PRECISION    0.3452
TABLES_RECALL       0.6910
TABLES_MRR5         0.6078
DOCS_PRECISION      0.9508
DOCS_RECALL         0.9196
DOCS_MRR5           0.9733
ANSWER_ACCURACY     0.3360
EXECUTION_ACCURACY  0.3360
```

The batch gains about three net correct answers on the hidden 506-question
subset. This package is the official base for period-aware planner v6.

## Checkpoint - typed period-aware average-balance planner v6 audited7

- Date: 2026-08-24
- Status: leaderboard verified; execution `0.3399`
- Base checkpoint: execution `0.3360`, filter/aggregate v5
- Target family: 9 average-balance formula questions
- Exact deterministic fills: `405, 410, 429, 449, 450, 462, 468`
- Final retrieval: `artifacts/retrieval_tranhuy_03360_plus_average_balance_v6_audited7_w010.jsonl`
- Rule output: `artifacts/codegen_average_balance_v6_final_k72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03360_plus_average_balance_v6_audited7_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03360_plus_average_balance_v6_audited7_w010/submission.zip`
- Retrieval SHA-256: `06177c5dc52b482a746d0c32b445a08da4e158c300af03ceb0290e720c07c4f2`
- Codegen SHA-256: `1b55a519c31ee3c81eaae6aa961fd473c00ac25fa3a6148bac1b00f25fe44d69`
- Submission SHA-256: `09ffd0a7b240478ebf1fe51d1fc72ec45021266ba597b64f1d8e8a54dafe7ba3`

Planner v6 adds typed `PeriodRef` and `AverageBalanceNode` operands. It covers
average assets, average equity and average net fixed assets, then composes ROA,
ROE, total/fixed-asset turnover and `(net profit - CFO) / average assets` with
the existing filter, median, ranking, projection and aggregate nodes. Year
ranking supports year-over-year selectors and predicates relative to each
candidate year instead of collapsing the whole interval into one change.

The exact executor requires consecutive opening/closing periods, distinct
cells, matching canonical operands, normalized units and dynamic filter and
selection guards. Same-code continuation rows are accepted only when their
normalized values agree. Negative financial returns remain valid percentages.
The two unresolved targets, `398` and `572`, stay fail-closed because the former
lacks a strong year header for HHV and the latter lacks exact GEX rows.

Targeted depth-72 retrieval reached complete evidence coverage for all nine
routes. The final audited package refreshes and fills only the seven accepted
IDs; all other 1,005 retrieval/codegen rows are identical to the `0.3360` base.
All 265 tests pass, all 1,012 expressions compile and replay, the ZIP contains
1,748 CSV files using official 1-based `<table>` line positions, and `unzip -t`
reports no errors.

Leaderboard metrics:

```text
TABLES_F2MACRO      0.5059
DOCS_F2MACRO        0.9212
TABLES_PRECISION    0.3443
TABLES_RECALL       0.6943
TABLES_MRR5         0.6069
DOCS_PRECISION      0.9494
DOCS_RECALL         0.9215
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3399
EXECUTION_ACCURACY  0.3399
```

The batch gains two net correct answers on the hidden 506-question subset and
is the official base for quantified-cohort planner v7.

## Checkpoint - typed quantified-cohort planner v7 audited11

- Date: 2026-08-24
- Status: leaderboard verified; execution `0.3498`
- Base checkpoint: execution `0.3399`, period-aware average-balance v6
- Target batch: 20 quantified cohort, top-k and partition questions
- Exact deterministic fills: `364, 404, 414, 437, 460, 467, 489, 540, 542, 569, 574`
- Final retrieval: `artifacts/retrieval_tranhuy_03399_plus_quantified_cohort_v7_audited11_w010.jsonl`
- Rule output: `artifacts/codegen_quantified_cohort_v7_k72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03399_plus_quantified_cohort_v7_audited11_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03399_plus_quantified_cohort_v7_audited11_w010/submission.zip`
- Retrieval SHA-256: `b5c078323175ffc57ee2400c6b6a85232c9a16ce768b8cc25c7feebe63c5a4da`
- Codegen SHA-256: `d3f0522739b7c47d4e4c6137f717c760034b9fe8bb593be9562eac1773dd2b68`
- Submission SHA-256: `a6fd879fd0b9bffdda8dde1be9a415290a036700c1514f048e152d4a1369e6bd`

Planner v7 adds typed `PeriodQuantifierNode` (`all` or `any` over explicit
years), `RankSliceNode` for top-k cohorts, scoped denominator predicates and a
`partition_ratio` aggregate. The deterministic executor resolves every member
and period, rejects top-k boundary ties, builds the complement partition, and
recomputes membership, rank and denominator scope inside the submitted pandas
expression.

Formula and entity linking now keep `CFO/LNST` and
`inventory/current liabilities` as single derived calculations, map the short
name Nam Kim to `NKG`, and treat interest expense as an absolute expense when
aggregating reports that mix display signs. The exact resolver remains
fail-closed for unresolved or ambiguous statement rows.

Depth-72 retrieval reached complete evidence coverage for 18/20 target routes.
Nine IDs (`401, 438, 446, 458, 466, 469, 470, 552, 566`) remain failed because
at least one exact operand is unavailable or ambiguous. The final retrieval and
codegen artifacts differ from v6 on exactly the 11 audited IDs. All 278 tests
pass, all 1,012 expressions compile, the ZIP contains 1,787 CSV files using
official 1-based `<table>` line positions, and `unzip -t` reports no errors.

Leaderboard metrics:

```text
TABLES_F2MACRO      0.5112
DOCS_F2MACRO        0.9232
TABLES_PRECISION    0.3469
TABLES_RECALL       0.7001
TABLES_MRR5         0.6083
DOCS_PRECISION      0.9488
DOCS_RECALL         0.9241
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3498
EXECUTION_ACCURACY  0.3498
```

The batch gains five net correct answers on the hidden 506-question subset and
is the official base for temporal-event planner v8.

## Candidate - typed temporal-event planner v8 audited11

- Date: 2026-08-24
- Status: leaderboard-confirmed execution `0.3538`
- Base checkpoint: execution `0.3498`, quantified-cohort v7
- Target batch: 40 unresolved temporal select/project questions
- Exact deterministic fills: `363, 365, 372, 386, 400, 407, 456, 549, 562, 573, 857`
- Final retrieval: `artifacts/retrieval_tranhuy_03498_plus_temporal_event_v8_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_temporal_event_v8_k25_final_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03498_plus_temporal_event_v8_audited11_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03498_plus_temporal_event_v8_audited11_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-temporal-event-v8-w010`
- Retrieval SHA-256: `ff07bc17245cb98049e8f8baea8ff790ab0c576bb1720f11979a7472ae8f5a20`
- Codegen SHA-256: `011e62ec755881b46e6055028d433570310b605288317a5892567e0c4e07c01c`
- Submission SHA-256: `a5a01c50c2dbd1ca73384f5d493282892ab344a538324d151175617c8cc90d26`

Planner v8 adds typed event selection over a `year` or `(entity, year)` axis,
with `first`, `last`, `argmax` and `argmin` event nodes plus an explicit target
period offset. The executor resolves the selector and every predicate for all
candidates, selects one event, then evaluates a distinct projection at the
selected year or its adjacent year. Missing years, ambiguous rows, ties and
unresolved offset targets fail closed.

The evidence router now reserves prior-year operands for every growth candidate
and next-year operands for offset projections. Exact row resolution accepts two
audited CFO-negative renderings: OCR-concatenated labels and the standard
`su dung vao hoat dong kinh doanh` wording. Compact matching is enabled only
when the canonical statement, metric and line code are already exact.

Targeted depth-72 retrieval needs `k=25` for the longest six-year, two-statement
questions. Twelve initial candidates were audited; ID `503` was rejected because
its projected note metric was not canonicalized. Ten fills come from v8 and ID
`857` is a safe collateral fill from the existing exact year-ranking v4 solver.
Plain max/min year questions are explicitly deferred to that existing solver,
leaving exactly 11 fill-only changes. The submitted expressions recompute
selection and predicates and force reads of all selector/filter/project cells.
All 287 tests pass, all 1,012
expressions compile, the ZIP contains 1,810 CSV files using official 1-based
`<table>` line positions, and `unzip -t` reports no errors.

Leaderboard metrics:

```text
TABLES_F2MACRO      0.5151
DOCS_F2MACRO        0.9272
TABLES_PRECISION    0.3481
TABLES_RECALL       0.7043
TABLES_MRR5         0.6076
DOCS_PRECISION      0.9488
DOCS_RECALL         0.9287
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3538
EXECUTION_ACCURACY  0.3538
```

## Candidate - typed note-detail arithmetic planner v9 audited8

- Date: 2026-08-24
- Status: leaderboard-confirmed execution `0.3577`
- Base checkpoint: execution `0.3538`, temporal-event v8
- Target batch: 24 unresolved note-detail arithmetic questions
- Exact deterministic fills: `24, 583, 696, 853, 877, 926, 1001, 1009`
- Final retrieval: `artifacts/retrieval_tranhuy_03538_plus_note_detail_v9_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_note_detail_v9_k25_final_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03538_plus_note_detail_v9_audited8_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03538_plus_note_detail_v9_audited8_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-note-detail-v9-w010`
- Upload ZIP: `artifacts/kaggle-payload-note-detail-v9-w010.zip`
- Retrieval SHA-256: `74d08ad0a2c89a9b4ad276cc40e15d20327464b9b17af52204e1af412f172adc`
- Codegen SHA-256: `914ce3db39b9b237dbc08545683202a97fafbd7a0935fda34a106d2f9cc527be`
- Submission SHA-256: `025414acbc31d26f665e8d87c69754cf280eeb7a991f383dac7db4f3b0c122a6`

Planner v9 adds a typed `NoteDetailPlan` for one exact note calculation over
one varying axis. It supports direct lookup, growth, difference, mean, sum,
argmax and argmin. The canonical registry now separates child rows such as USD
long-term borrowings, real-estate customer loans, deposit-interest expense,
finished goods and general loan provision from their parent totals. Derived
metrics declare every numerator and denominator requirement before retrieval.

The executor resolves every operand with strong year evidence, requires all
operands of one ratio to come from the same report, rejects duplicate cells and
ties, and emits expressions that read every selected point. Eight fills clear
the confidence-94 allowlist. The other 16 targets remain untouched; most need a
matrix-aware resolver because the requested segment/category is represented by
a column header or multi-row block rather than an exact row label.

All 300 tests pass. All 1,012 submission expressions compile, the eight new
queries execute from packaged CSV evidence and reproduce their answers, and the
ZIP contains 1,833 CSV files using official 1-based `<table>` line positions.
`unzip -t` reports no errors.

Leaderboard metrics:

```text
TABLES_F2MACRO      0.5195
DOCS_F2MACRO        0.9268
TABLES_PRECISION    0.3511
TABLES_RECALL       0.7096
TABLES_MRR5         0.6125
DOCS_PRECISION      0.9488
DOCS_RECALL         0.9282
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3577
EXECUTION_ACCURACY  0.3577
```

Relative to v8, execution improves by `0.0039`, equivalent to two net-correct
answers on the 506-question leaderboard subset. Table F2 improves by `0.0044`
while docs F2 decreases slightly by `0.0004`. This confirms v9 as the new
official base checkpoint. The next batch should target matrix-style note tables
instead of adding more row-only aliases.

## Checkpoint - typed matrix-note planner v10 audited5

- Date: 2026-08-24
- Status: leaderboard-confirmed execution `0.3597`
- Base checkpoint: execution `0.3577`, note-detail v9
- Audited fill-only IDs: `806, 821, 931, 982, 1007`
- Retrieval: `artifacts/retrieval_tranhuy_03538_plus_note_detail_v9_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_matrix_note_v10_k25_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03577_plus_matrix_note_v10_audited5_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03577_plus_matrix_note_v10_audited5_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-matrix-note-v10-w010`
- Upload ZIP: `artifacts/kaggle-payload-matrix-note-v10-w010.zip`
- Retrieval SHA-256: `74d08ad0a2c89a9b4ad276cc40e15d20327464b9b17af52204e1af412f172adc`
- Codegen SHA-256: `a05af766f380f87b31baeded8cfc11cb9d140e82b373844ebe58434c455ebd9f`
- Submission SHA-256: `d1334ab090c5f685ebe52206ff9aa670bd5fb0cbcfc2333ea6483267ef2585c0`
- Payload ZIP SHA-256: `e55f2a429640af8f534c15c8fc9703940bdd699b8bcd0fc89672fad9671d4022`

Planner v10 adds typed matrix families backed by an exact row/column/block
resolver over the original `grid_json`. It supports subsidiary-only block
means, current-period row/column intersections, repeated VAS row codes,
segment columns, totals, cross-entity means and year argmax. Duplicate OCR
renderings are accepted only when their complete ordered value vectors agree;
missing axes, conflicting copies and ties fail closed.

The five audited answers are `806=6.63`, `821=13.06`, `931=57.15`,
`982=2025` and `1007=92.13`. Their generated expressions read 78 concrete
cells and replay from packaged CSV evidence. The tidy serializer has one
narrow compatibility fix so integer voting rates such as `100` are retained
instead of being mistaken for VAS line codes; all other tables preserve the
legacy behavior.

All 306 tests pass. All 1,012 expressions compile and replay validation passes.
The submission ZIP contains 1,848 CSV files using official 1-based `<table>`
line positions, and both submission/payload archives pass `unzip -t`. The five
changes only fill rows that were `failed` in v9.

Leaderboard result:

```text
TABLES_F2MACRO      0.5206
DOCS_F2MACRO        0.9274
TABLES_PRECISION    0.3515
TABLES_RECALL       0.7110
TABLES_MRR5         0.6125
DOCS_PRECISION      0.9488
DOCS_RECALL         0.9290
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3597
EXECUTION_ACCURACY  0.3597
```

Compared with v9 (`0.3577`), v10 adds one net-correct answer on the 506-question
leaderboard subset. It is now the official base for subsequent fill-only work.

## Checkpoint - typed note-axis aggregation planner v11 audited11

- Date: 2026-08-24
- Status: leaderboard-confirmed execution `0.3696`
- Base checkpoint: execution `0.3597`, matrix-note v10
- Audited fill-only IDs: `598, 690, 717, 723, 796, 910, 924, 968, 970, 986, 996`
- Retrieval: `artifacts/retrieval_tranhuy_03538_plus_note_detail_v9_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_note_axis_v11_k72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03597_plus_note_axis_v11_audited11_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03597_plus_note_axis_v11_audited11_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-note-axis-v11-w010`
- Upload ZIP: `artifacts/kaggle-payload-note-axis-v11-w010.zip`
- Retrieval SHA-256: `74d08ad0a2c89a9b4ad276cc40e15d20327464b9b17af52204e1af412f172adc`
- Rule SHA-256: `6d30d705fd124c4596db8a3febb008ec8936ecf4f68db9bf1b254237faa78b5d`
- Codegen SHA-256: `367aa94907fe733555e5b7ecbe9e0d03fd3bedc5b36e26bb7de9312bd58adcd7`
- Submission SHA-256: `37269f5e0054a4005094d02be5e6526b8c974f21f30da4d0065f60dec645baf9`
- Payload ZIP SHA-256: `eb749473c9b7eaa37a778b5b3772d2fa82d3bcb4b55fde210cf36a560fb85b73`

Planner v11 adds exact note-table axes for subtotal growth, total-to-total
ratios, absolute differences of child/parent shares, multi-period sums, means
and maxima. It prioritizes specific period headers before generic fallbacks,
requires same-table numerator/denominator provenance where appropriate, and
uses note context to distinguish repeated depreciation and inventory grids.
Dash-valued current cells are accepted only when zero is reproduced from a
numeric subtotal identity; no constant zero is inserted into a query.

Audited answers:

```text
598   -95.56
690   1236.06
717   14.99
723   94.88
796   3.32
910   2022
924   9.02
968   82.69
970   11.35
986   2016
996   4442784.00
```

The merge changes exactly these 11 failed rows and preserves the other 1,001
records from the leaderboard-confirmed v10 codegen. All 311 tests pass, all
1,012 expressions compile and replay, the submission contains 1,872 CSV files
with official 1-based `<table>` line positions, and both submission and payload
archives pass `unzip -t`. The checkpoint was submitted directly without a Qwen
rerun; its deterministic merge preserves the already confirmed v10 answers
outside the allowlist.

Leaderboard result:

```text
TABLES_F2MACRO      0.5246
DOCS_F2MACRO        0.9282
TABLES_PRECISION    0.3537
TABLES_RECALL       0.7159
TABLES_MRR5         0.6125
DOCS_PRECISION      0.9488
DOCS_RECALL         0.9299
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3696
EXECUTION_ACCURACY  0.3696
```

Compared with v10, execution improves by `0.0099`, equivalent to five
net-correct answers on the 506-question leaderboard subset. Table F2 improves
by `0.0040` and docs F2 by `0.0008`. This confirms v11 as the new official base
for subsequent fill-only work.

## Checkpoint - typed lease-schedule planner v12 audited7

- Date: 2026-08-24
- Status: leaderboard-confirmed execution `0.3755`
- Base checkpoint: execution `0.3696`, note-axis v11
- Audited fill-only IDs: `37, 125, 128, 233, 638, 882, 895`
- Retrieval: `artifacts/retrieval_tranhuy_03696_plus_lease_schedule_v12_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_lease_schedule_v12_k72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03696_plus_lease_schedule_v12_audited7_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03696_plus_lease_schedule_v12_audited7_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-lease-schedule-v12-w010`
- Upload ZIP: `artifacts/kaggle-payload-lease-schedule-v12-w010.zip`
- Retrieval SHA-256: `97702388d8221099c72de6adda3e410e9e41d95aaef8dadbdee1c0d651481df3`
- Rule SHA-256: `c178b13ecfa6f6c845f3f88c4d6f6a703747569bb598e8a3bb0d0bfb80e85d25`
- Codegen SHA-256: `098843cc4556f13c80bdf0b88140f07d5d50874a1ba30184e6cfee6eda1a4101`
- Submission SHA-256: `ff5157213d24df894674bd149cbdc67d40cd530f45cbc0e2233114609e5548e8`
- Payload ZIP SHA-256: `79d6ca66cca9863391785101128c2aef8978a903370ba574db1701bad2416c2f`

Planner v12 represents an operating-lease schedule by direction
(`receivable` or `payable`), maturity axis (`short_term`, `total` or their
share) and reduction (`direct`, `growth`, `mean` or `max`). The resolver uses
the original grid to distinguish the reporting-period column and the under-one-
year, one-to-five-year and over-five-year buckets. It accepts `TONG CONG`,
`Cong` and blank total labels, but only when the reported total reconciles to
all available maturity buckets.

The direction guard separates lessor and lessee tables that occur next to each
other in the same filing. A targeted schema-linking rule recognizes maturity
schedules whose total row is too generic for ordinary row matching; this moves
the missing VIC lessor table for ID 882 into the evidence quota without
boosting its lessee schedule. Every generated expression reads the subtotal and
all bucket cells and recomputes the reconciliation identity.

Audited answers:

```text
37     201.00
125      0.84
128     27.35
233      2.44
638      5.74
882     17.16
895  23885.08
```

The merge changes exactly these seven rows and preserves the other 1,005
records from the leaderboard-confirmed v11 codegen. ID 336 remains failed
because its lease figures occur only in narrative text, while IDs 495 and 533
are selector/project questions outside the v12 semantics. All 318 tests pass,
all 1,012 expressions compile and replay, the submission contains 1,879 CSV
files with official 1-based `<table>` line positions, and both submission and
payload archives pass `unzip -t`. Submit the local merged ZIP directly; a Qwen
rerun is not required.

Leaderboard result:

```text
TABLES_F2MACRO      0.5259
DOCS_F2MACRO        0.9282
TABLES_PRECISION    0.3546
TABLES_RECALL       0.7174
TABLES_MRR5         0.6140
DOCS_PRECISION      0.9488
DOCS_RECALL         0.9299
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3755
EXECUTION_ACCURACY  0.3755
```

Compared with v11, execution improves by `0.0059`, equivalent to three
net-correct answers on the 506-question leaderboard subset. Table F2 improves
by `0.0013`, table MRR5 by `0.0015`, and docs F2 is unchanged. This confirms
v12 as the new official base for subsequent fill-only work.

## Checkpoint - typed select-then-project planner v13 audited6

- Date: 2026-08-25
- Status: leaderboard-confirmed execution `0.3814`
- Base checkpoint: execution `0.3755`, lease-schedule v12
- Audited fill-only IDs: `495, 501, 503, 522, 524, 533`
- Retrieval: `artifacts/retrieval_tranhuy_03696_plus_lease_schedule_v12_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_select_project_v13_k72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03755_plus_select_project_v13_audited6_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03755_plus_select_project_v13_audited6_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-select-project-v13-w010`
- Upload ZIP: `artifacts/kaggle-payload-select-project-v13-w010.zip`
- Rule SHA-256: `0ea64b6eecfa1bfec084593b937651130f54acd6594e9d9e77175dc312f12537`
- Codegen SHA-256: `cf1d2bfcd3eca2e0f5e9f980cbabd8a7fa6f2ac3c2bb2b53ea0743465ec7ffce`
- Submission SHA-256: `5a7eda65bb1217c347c490d53ac2bafcb585965030441193cad826b29178e531`
- Payload ZIP SHA-256: `9c43f6bdb79239ab31b08a481a3045ef230a100f00fabbcdb0d1949db22e749f`

Planner v13 models a two-stage question explicitly: rank years by a typed
selector, then project a different note value from the winning year. ID 501
adds a lexicographic selector: maximize investment cost first, retain every
tied year, then maximize gross overdue receivables and return the winning year.
All other families require a unique primary winner.

The matrix resolver now scopes repeated row labels to an exact company block
and can combine multi-row headers such as `31/12/2023` plus `Gia goc`. It also
accepts legacy report IDs without a consolidated/separate suffix only when no
typed report competes for that ticker and year. Lease totals still have to
reconcile to all maturity buckets before they can be projected.

Audited answers:

```text
495     385.62
501    2023.00
503     217.22
522 -100562.00
524     102.42
533      10.61
```

The fill-only merge changes exactly these six failed rows and preserves the
other 1,006 records from the leaderboard-confirmed v12 checkpoint. All 322
tests pass, all 1,012 expressions compile and replay, the submission contains
1,907 CSV files with official 1-based `<table>` line positions, and both the
submission and payload archives pass `unzip -t`. The checkpoint was submitted
directly without a Qwen rerun.

Leaderboard result:

```text
TABLES_F2MACRO      0.5281
DOCS_F2MACRO        0.9293
TABLES_PRECISION    0.3560
TABLES_RECALL       0.7201
TABLES_MRR5         0.6140
DOCS_PRECISION      0.9490
DOCS_RECALL         0.9312
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3814
EXECUTION_ACCURACY  0.3814
```

Compared with v12, execution improves by `0.0059`, equivalent to three
net-correct answers on the 506-question leaderboard subset. Table F2 improves
by `0.0022` and document F2 by `0.0011`. This confirms v13 as the official base
for subsequent fill-only work.

## Checkpoint - typed financial-scenario planner v14 audited17

- Date: 2026-08-25
- Status: leaderboard-confirmed execution `0.3913`
- Base checkpoint: execution `0.3814`, select-then-project v13
- Audited fill-only IDs: `368, 387, 394, 409, 416, 419, 423, 432, 433, 434, 436, 453, 458, 469, 470, 545, 566`
- Retrieval: `artifacts/retrieval_tranhuy_03814_plus_scenario_v14_audit_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_scenario_v14_audit_k72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03814_plus_scenario_v14_audited17_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03814_plus_scenario_v14_audited17_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-scenario-v14-w010`
- Upload ZIP: `artifacts/kaggle-payload-scenario-v14-w010.zip`
- Retrieval SHA-256: `8a57799a78517bbff8ca3d4e74c182f1cafd7e3caeea8262828bba4b55c36648`
- Rule SHA-256: `0ce9c16335ea3dfa3c4eafa20196152c0f5024e43364b7d74063db5f62cc19a3`
- Codegen SHA-256: `4250d7bb477a2bf9dc0cf85e32f50384387174519bb32f25d2e4dc9f9f483049`
- Submission SHA-256: `9043411f3f28f7b3e14f5ea2f709e546ec4d8215169ac07c72c7ee1001f68f3e`
- Payload ZIP SHA-256: `baaa926aa78c6578217bb3c33c3fe4c6a016a40f08fe19307e5ad8150c7ca563`

V14 recognizes six deterministic scenario families: interest-expense shocks,
EBIT shocks, revenue growth to a median turnover target, liquidation stress,
interest-cost headroom and price growth needed to preserve operating margin.
Every family resolves the complete entity population, replays all filters and
scenario arithmetic in the submitted pandas expression, and fails closed on a
missing operand or an unruled tie.

Canonical inventory now distinguishes the net balance-sheet aggregate (code
`140`) from gross inventory (code `141`). Exact VAS codes can prove canonical
identity when OCR changes the row wording, and `chi phi di vay` is accepted as
interest expense. Entity aliases add BSR/PVT and distinguish GEE from GEX.
Besides the six scenario rows, these changes safely unlock eleven existing
cohort and temporal formulas with the same exact operands.

Audited answers:

```text
368    0.59
387    6.99
394   20.85
409   14.37
416    1.74
419    0.56
423    3.70
432   88.05
433   26.44
434   10.79
436    4.78
453    0.59
458    5.24
469    0.25
470    1.29
545    2.55
566    0.25
```

The merge changes exactly these 17 failed rows and preserves the other 995
records from the leaderboard-confirmed v13 checkpoint. IDs 424 and 425 remain
failed because the opening-period header or exact early-year profit row is not
reliable. Semantically incorrect generic candidates are excluded. All 325
tests pass, all 1,012 expressions compile and replay, the submission contains
1,993 CSV files with official 1-based `<table>` line positions, and both ZIP
archives pass integrity checks. The checkpoint was submitted directly without a
Qwen rerun.

Leaderboard result:

```text
TABLES_F2MACRO      0.5358
DOCS_F2MACRO        0.9316
TABLES_PRECISION    0.3579
TABLES_RECALL       0.7287
TABLES_MRR5         0.6150
DOCS_PRECISION      0.9480
DOCS_RECALL         0.9342
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3913
EXECUTION_ACCURACY  0.3913
```

Compared with v13, execution improves by `0.0099`, equivalent to five
net-correct answers on the 506-question leaderboard subset. Table F2 improves
by `0.0077` and document F2 by `0.0023`, mainly through recall. This confirms
v14 as the official base for subsequent fill-only work.

## Execution 0.3953 - typed matrix sensitivity/FX planner v15 audited6

- Date: 2026-08-25
- Status: leaderboard confirmed
- Base checkpoint: execution `0.3913`, financial-scenario v14
- Audited fill-only IDs: `156, 213, 275, 427, 428, 727`
- Retrieval: `artifacts/retrieval_tranhuy_03913_plus_matrix_risk_v15_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_matrix_risk_v15_depth72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03913_plus_matrix_risk_v15_audited6_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03913_plus_matrix_risk_v15_audited6_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-matrix-risk-v15-w010`
- Upload ZIP: `artifacts/kaggle-payload-matrix-risk-v15-w010.zip`
- Retrieval SHA-256: `ee536d076d6ba1e9793f25a2c51dbbd3f3df3159a8301f81d1e14475d716b516`
- Rule SHA-256: `b4115d29187b3814463ffa6a456d0fd475f37b05686e008e683feb113d74b225`
- Codegen SHA-256: `2d61dd7e7648642fe71ceab3898ba741fdc9016f08051b672d0e1640155ddd32`
- Submission SHA-256: `f75f0d5d048360e50e29692aa3e6d71cb59181cfcb425d936c2f202ff3068d0f`
- Payload ZIP SHA-256: `069e2796bd4145840dced59e49479d381bbec82b37eb836311e3461d8f9bd5c6`

V15 adds typed matrix families for a credit-risk total, native-currency USD
balance, derivative contract notional, filtered FX sensitivity loss, worst FX
position stress relative to pretax profit, and one-day listed-equity VaR
growth. Every path fixes the requested row, period column and note block, and
fails closed when any currency, year, denominator or matrix axis is missing.

Audited answers:

```text
156       2991.04
213          6.16
275    3080776.00
427         55.55
428          0.29
727          8.56
```

The fill-only merge changes exactly these six failed rows and preserves all
other 1,006 records from leaderboard-confirmed v14. All 332 tests pass, and all
1,012 submitted expressions replay to their recorded answers from the packaged
CSV evidence. The submission contains 1,996 CSV files using official 1-based
`<table>` line positions; both ZIP archives pass integrity checks.

Leaderboard result:

```text
TABLES_F2MACRO      0.5357
DOCS_F2MACRO        0.9316
TABLES_PRECISION    0.3578
TABLES_RECALL       0.7287
TABLES_MRR5         0.6150
DOCS_PRECISION      0.9480
DOCS_RECALL         0.9342
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.3953
EXECUTION_ACCURACY  0.3953
```

Execution improves by `0.0040`, equivalent to two net-correct answers on the
506-question leaderboard subset. V15 is the official base for V16.

## Execution 0.4012 - typed note-temporal aggregate planner v16 audited12

- Date: 2026-08-25
- Status: leaderboard confirmed
- Base checkpoint: execution `0.3953`, matrix sensitivity/FX v15
- Audited fill-only IDs: `102, 260, 617, 652, 663, 685, 836, 854, 887, 912, 939, 941`
- Retrieval: `artifacts/retrieval_tranhuy_03953_plus_note_temporal_v16_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_note_temporal_v16_depth72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_03953_plus_note_temporal_v16_audited12_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_03953_plus_note_temporal_v16_audited12_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-note-temporal-v16-w010`
- Upload ZIP: `artifacts/kaggle-payload-note-temporal-v16-w010.zip`
- Retrieval SHA-256: `9e9b1576a7a18770fdaeb387f6fbee9851d51c0da062b0f9a1a973bdec7d7532`
- Rule SHA-256: `134c0378c4ad507fee5aece67e092253c6403fb247fecb1acebe280fbbf66e29`
- Codegen SHA-256: `178f85816f294be4d06145340957cef0e113d4d298f2e8aff995b5481741caae`
- Submission SHA-256: `22cea3bde5839b43952ae17dc03ef06e6f4591a3c3b243db995881fc11e4d2b5`
- Payload ZIP SHA-256: `a986faf26d78de466fa05c162d23b93246561e36fa0dedf939a348413320595b`

V16 adds typed direct matrix lookup, closing-period difference, statement-code
ratio, bad-debt coverage and temporal max/sum/mean reductions. It also supports
multi-row period headers and headerless note subtotals. Canonical metrics now
cover NPL groups, total unearned revenue, tangible-fixed-asset depreciation and
other short-term subsidiary payables.

Audited answers:

```text
102    933246.00
260       100.00
617        10.40
652        28.53
663        36.76
685       107.54
836       280.54
854         4.41
887     32860.00
912        11.45
939        45.08
941      1150.34
```

The merge changes exactly these 12 failed rows and preserves the other 1,000
records from V15. All 341 tests pass; all 1,012 expressions compile and replay
from 2,013 packaged CSV files using official 1-based `<table>` line positions.
Both ZIP archives pass integrity checks.

Leaderboard result:

```text
TABLES_F2MACRO      0.5377
DOCS_F2MACRO        0.9320
TABLES_PRECISION    0.3589
TABLES_RECALL       0.7313
TABLES_MRR5         0.6135
DOCS_PRECISION      0.9480
DOCS_RECALL         0.9347
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.4012
EXECUTION_ACCURACY  0.4012
```

Execution improves by `0.0059`, equivalent to three net-correct answers on the
506-question leaderboard subset. Table F2 improves by `0.0020` and document F2
by `0.0004`. V16 is the official base for V17.

## Execution 0.4032 - typed note-ratio/maturity planner v17 audited6

- Date: 2026-08-25
- Status: leaderboard confirmed
- Base checkpoint: execution `0.4012`, note-temporal aggregate v16
- Audited fill-only IDs: `826, 863, 885, 892, 945, 955`
- Retrieval: `artifacts/retrieval_tranhuy_04012_plus_note_ratio_v17_depth72_w010.jsonl`
- Rule output: `artifacts/codegen_rule_note_ratio_v17_depth72_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_04012_plus_note_ratio_v17_audited6_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04012_plus_note_ratio_v17_audited6_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-note-ratio-v17-w010`
- Upload ZIP: `artifacts/kaggle-payload-note-ratio-v17-w010.zip`
- Retrieval SHA-256: `53b9468a6025ebc292ddc628ae925bb15298a259dce0b7e6858b8f0a30b905f5`
- Rule SHA-256: `e15846cf0ad98b44757d35820c3ee1f9da7033dfe650bb96c8fcacfb30910310`
- Codegen SHA-256: `9e121e13a7959fe395718786c58125bbb2d28c36a479f4695050ca6719d39abd`
- Submission SHA-256: `46e6cf543c237e9d3fed6d71af8d2a3f2c364dfa9eecebd9f388aa4c066a0beb`
- Payload ZIP SHA-256: `f1abf8f40a5e79e64b8c822b17e201cf4a7e795d42bf7878e4e17cb7f61ac38c`

V17 adds deterministic note-ratio families for land/infrastructure rental cost,
named related-party payables, certificate-deposit maturity, Laos geographic
revenue, off-balance USD balances and short-term customer loans. Each ratio
resolves numerator and denominator in the same note table and period. The
planner fixes the requested entity/year population and fails closed on missing
operands, ambiguous rows or incomplete cohorts.

Audited answers:

```text
826    2016.00
863       1.00
885       8.63
892      29.17
945       9.50
955      54.66
```

The fill-only merge changes exactly these six failed rows and preserves the
other 1,006 records from leaderboard-confirmed V16. Failed rows decrease from
51 to 45. All 347 tests pass; all 1,012 expressions compile and replay from
2,028 packaged CSV files using official 1-based `<table>` line positions. Both
ZIP archives pass integrity checks. The submission was evaluated directly;
the payload is only a reproducibility backup and did not require a Qwen rerun.

Leaderboard result:

```text
TABLES_F2MACRO      0.5395
DOCS_F2MACRO        0.9320
TABLES_PRECISION    0.3601
TABLES_RECALL       0.7333
TABLES_MRR5         0.6155
DOCS_PRECISION      0.9480
DOCS_RECALL         0.9347
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.4032
EXECUTION_ACCURACY  0.4032
```

Execution improves by `0.0020`, equivalent to one net-correct answer on the
506-question leaderboard subset. Table F2 improves by `0.0018`; document F2
is unchanged. V17 is the official base for V18.

## Execution 0.4071 - exact VAS-code cohort resolver v18 audited12

- Date: 2026-08-25
- Status: leaderboard confirmed
- Base checkpoint: execution `0.4032`, note-ratio/maturity v17
- Audited fill-only IDs: `389, 390, 397, 401, 403, 406, 420, 438, 444, 445, 552, 572`
- Retrieval: `artifacts/retrieval_tranhuy_04032_plus_vas_cohort_v18_depth96_w010.jsonl`
- Rule output: `artifacts/codegen_rule_vas_cohort_v18_depth96_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_04032_plus_vas_cohort_v18_audited12_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04032_plus_vas_cohort_v18_audited12_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-vas-cohort-v18-w010`
- Upload ZIP: `artifacts/kaggle-payload-vas-cohort-v18-w010.zip`
- Retrieval SHA-256: `fde2e958fe4399af3b8f92ceb0acb3ed6c07d91b505b107d77a266360b2247f4`
- Rule SHA-256: `e43822953cab4333cd0a03ecd8cd08ae2f8de32c7bff088f7cad517deab76eb4`
- Codegen SHA-256: `6e3baf01a8f82cef0ee211b5f653dddc66b99ee5e21d25e4cfa9244b6186408e`
- Submission SHA-256: `1312ee61da8ad3ed2d373ff13e1450497832b1a5a37c249a0e63ece4e80a30d6`
- Payload ZIP SHA-256: `cd5a361963861b6359c67a901078d8cd8529ec78def8bea4df65f1bab293d78d`

V18 adds an exact statement-code path before fuzzy row matching. It activates
only in recognized balance-sheet, income-statement and cash-flow contexts,
rejects rows whose readable label maps to a conflicting canonical metric, and
prefers the closing/current period over opening columns. Repeated generic OCR
headers are resolved to the leftmost current-period cell; conflicting exact
code values fail closed. This safely recovers opaque rows such as current
assets code `100` without treating numbered note rows as statement codes.

Audited answers:

```text
389      1.10
390      6.69
397    146.61
401     47.45
403     -0.06
406     -3.56
420      4.77
438    -27.22
444      0.91
445      2.01
552    150.60
572     12.88
```

The targeted retrieval changes only these 12 rows. The fill-only merge accepts
all 12 at confidence `93-94`, preserves the other 1,000 V17 records and lowers
failed rows from 45 to 33. All 352 tests pass; all 1,012 expressions compile
and replay. The submission contains 2,043 CSV files with official 1-based
`<table>` line positions, and both ZIP archives pass integrity checks. It was
submitted directly without a Qwen rerun.

Leaderboard result:

```text
TABLES_F2MACRO      0.5446
DOCS_F2MACRO        0.9354
TABLES_PRECISION    0.3610
TABLES_RECALL       0.7386
TABLES_MRR5         0.6165
DOCS_PRECISION      0.9480
DOCS_RECALL         0.9384
DOCS_MRR5           0.9723
ANSWER_ACCURACY     0.4071
EXECUTION_ACCURACY  0.4071
```

Execution improves by `0.0039`, equivalent to two net-correct answers on the
506-question leaderboard subset. Table F2 improves by `0.0051` and document
F2 by `0.0034`. V18 is the official base for V19.

## Candidate - typed derived-selector planner v19 audited9

- Date: 2026-08-25
- Status: leaderboard evaluated; retrieval improved but execution unchanged
- Base checkpoint: execution `0.4071`, exact VAS-code cohort v18
- Audited fill-only IDs: `376, 377, 381, 382, 391, 417, 418, 422, 461`
- Retrieval refresh IDs: audited IDs plus failed control `426`
- Retrieval: `artifacts/retrieval_tranhuy_04071_plus_derived_selector_v19_final_depth112_w010.jsonl`
- Rule output: `artifacts/codegen_rule_derived_selector_v19_depth112_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_04071_plus_derived_selector_v19_audited9_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04071_plus_derived_selector_v19_audited9_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-derived-selector-v19-w010`
- Upload ZIP: `artifacts/kaggle-payload-derived-selector-v19-w010.zip`
- Retrieval SHA-256: `e01fb6c32a2198dac59a5155e0d00f9f67f92276eb113b0e9ef98fdc08062f92`
- Rule SHA-256: `0d09f6de153dc05569f5d1127bc1964f95247cba3430f760fa0e516a82198548`
- Codegen SHA-256: `ccc8e8d0da69179ee3b10b6d64de59c4f0fa674d2f8b7ecac20a9c49e4489547`
- Submission SHA-256: `dee6de45c5ef514bf0584fbd83b9453b0bf4bd5ce5f67e737106de8088e4ea14`
- Payload ZIP SHA-256: `f4f28846f45eeacd7957f1a1c1e6ae0203d5935e5eba316078428b5f0edf637d`

V19 adds typed selectors for same-period derived differences, temporal changes
and an extreme-to-extreme projection spread. It recognizes CFO margin minus
net margin, net profit minus CFO, and `(CFO - net profit) / revenue`; the
submitted query replays both the selector and projected metric. For
percentage-point spread questions it proves unique maximum and minimum
selector entities before subtracting their projected values. SG&A intensity
uses absolute expense magnitudes so statements that present expenses as
negative numbers rank consistently.

Entity routing adds exact trade-name aliases for DPM, DCM and HT1. Ratio
phrasing accepts both `ty so` and `ti so`, while clause-local target mode keeps
a preceding growth filter from turning the projected current-period ratio into
a growth calculation.

Audited answers:

```text
376    61.66
377     6.86
381    22.16
382     0.64
391    21.37
417     0.94
418    -4.58
422     0.55
461     0.97
```

The fill-only merge changes exactly these nine failed rows, preserves all other
1,003 V18 records and lowers failed rows from 33 to 24. ID 426 remains failed
and is not merged. All 359 tests pass; all 1,012 expressions compile and replay
from 2,054 packaged CSV files using official 1-based `<table>` line positions.
Both ZIP archives pass integrity checks. V19 can be submitted directly without
a Qwen rerun. If all nine additions are correct on the 506-question subset,
the execution ceiling is approximately `0.4249`; this remains a projection
until leaderboard evaluation.

## Candidate - implicit-period select/project planner v20 audited3

- Date: 2026-08-25
- Status: locally audited; waiting for leaderboard
- Base checkpoint: V19 leaderboard-confirmed execution `0.4111`
- Audited fill-only IDs: `384, 430, 431`
- Retrieval refresh IDs: `384, 430, 431, 511`
- Retrieval: `artifacts/retrieval_implicit_period_v20_probe_depth112_w010.jsonl`
- Rule output: `artifacts/codegen_rule_implicit_period_v20_depth112_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_04111_plus_implicit_period_v20_audited3_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04111_plus_implicit_period_v20_audited3_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-implicit-period-v20-w010`
- Upload ZIP: `artifacts/kaggle-payload-implicit-period-v20-w010.zip`
- Retrieval SHA-256: `e06d5829277c607ec6254a1ce225b5b364fafb77041522c9528af1748b328516`
- Rule SHA-256: `fd97665012af69de5eb146aec8dcdf01b836cd60331421cb3ca7e379b52e428b`
- Codegen SHA-256: `f1f47fb0fbc26164824adfac8759897dbd09ccc7bc9bb7856faef8009c5bdd4b`
- Submission SHA-256: `37c68ba6111d1ad21fb6b863de2e54e5dc58fb1729e26c0fb2ce96282576f82a`
- Payload ZIP SHA-256: `28e1d192ccd4838b68379987f7fb24ee7dee7910715a046ada5797055ffd7bf0`

V20 adds a typed implicit-period planner for questions that name only the
closing year but require the change from the beginning to the end of that year.
It resolves the prior fiscal period explicitly, requires every entity's exact
evidence, and fails closed on any missing company or tie. It also supports a
joint revenue-and-operating-margin decline predicate before ranking by the
change in SG&A intensity.

Audited answers:

```text
384    10.86    HSG: largest 2024 inventory/assets increase; 2024 gross margin
430    17.51    KBC: revenue and operating margin both declined; largest SG&A increase
431     5.99    VPI: gross margin declined by >2pp; largest asset-turnover increase
```

The merge changes exactly these three previously failed rows, preserves the
other 1,009 V19 records, and lowers failed rows from 24 to 21. The candidate
initially included ID 511, but it was deliberately excluded: a broad serializer
change needed for one DPM EPS table altered CSV layouts used by existing
checkpoint expressions. The replay guard caught the regression on IDs 731, 901
and 932, so that change was reverted before packaging.

All 362 tests pass. All 1,012 expressions compile and replay against the 2,054
submitted CSV files; both submission and payload ZIPs pass integrity checks.
Leaderboard evaluation returned `TABLES_F2MACRO=0.5498`,
`DOCS_F2MACRO=0.9400`, and `EXECUTION_ACCURACY=0.4111`. Retrieval improved
slightly over V19, but none of the three fills produced a net execution gain.
The implicit-period branch is therefore closed; V20 remains the retrieval base
for the next exact-resolver batch, not a new execution checkpoint.

## Checkpoint - exact period, EPS and lease resolver v21 audited5

- Date: 2026-08-26
- Status: leaderboard confirmed
- Base: V20 retrieval and codegen, leaderboard execution `0.4111`
- Audited fill-only IDs: `336, 398, 452, 511, 550`
- Remaining failed IDs: `388, 424, 425, 426, 442, 443, 446, 464, 466, 497, 510, 538, 805, 907, 950, 995`
- Retrieval: `artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl`
- Rule output: `artifacts/codegen_rule_exact_period_v21_depth112_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_04111_plus_exact_period_v21_audited5_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04111_plus_exact_period_v21_audited5_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-exact-period-v21-w010`
- Upload ZIP: `artifacts/kaggle-payload-exact-period-v21-w010.zip`
- Retrieval SHA-256: `f3a5f55b737c431ed8a927587a7b4097280f023d54eeb68ec3cab0bde2fca608`
- Rule SHA-256: `96bb1f1287c33171fec96d55a732fa17d1c97dce920ba1c45c4ee6fb929e19f6`
- Codegen SHA-256: `1228d8e3fb5321b32f34ff279b3992859aea72771bd0d0d22f111f8b772fd4d2`
- Submission SHA-256: `06c5f1ffdcf7bc40f826bcc5007b4f35d42e5156523dc483e8474439b36a2643`
- Payload ZIP SHA-256: `e36d2f776dc2e2be02f45493c0bc1a5fddb7db7ffcbd076b562a6a0153f33ef6`

V21 resolves two extraction edge cases without changing global CSV
serialization. Multi-row balance-sheet headers now recover dotted dates such
as `31.12.2024` and distinguish them from `01.01.2024`. Basic EPS stored in the
serializer's `code` field is accepted only when the canonical EPS label, raw
grid value, current-period header and inferred decimal multiplier all agree.
This isolates the DPM `1.551 -> 1,551` conversion that previously regressed
IDs 731, 901 and 932.

The lease schedule resolver also recognizes unqualified rented-asset context
and the `Trong vòng 1 năm` maturity bucket. Audited answers:

```text
336   387.66    HND future minimum lease total
398     0.09    HHV total-asset turnover using average assets
452    35.22    mean gross margin for DLG and HHV below cohort median
511    21.17    HPG ROE after selecting the highest EPS
550     0.20    mean CFO/current-liabilities ratio for DLG and HHV
```

The allowlisted merge changes exactly five failed rows, preserves the other
1,007 V20 records and lowers failed rows from 21 to 16. Candidate ID 510 is
explicitly rejected because its fallback emits a unit warning and confidence
40. All 365 tests pass; all 1,012 submitted expressions compile and replay
from 2,071 CSV files with official 1-based `<table>` line positions. Both ZIP
archives pass integrity checks. No Qwen rerun is required for this deterministic
submission.

Leaderboard evaluation returned `TABLES_F2MACRO=0.5510`,
`DOCS_F2MACRO=0.9392`, and `EXECUTION_ACCURACY=0.4150`. Execution improved by
`0.0039` over V20, approximately two additional correct answers on the
506-question scoring subset. V21 is the new execution checkpoint.

## Candidate - grid-backed exact cohort resolver v22 audited5

- Date: 2026-08-27
- Status: leaderboard confirmed
- Base checkpoint: V21 leaderboard execution `0.4150`
- Audited fill-only IDs: `388, 442, 443, 446, 466`
- Remaining failed IDs: `424, 425, 426, 464, 497, 510, 538, 805, 907, 950, 995`
- Retrieval: `artifacts/retrieval_v21_failed21_probe_depth112_w010.jsonl`
- Rule output: `artifacts/codegen_rule_grid_exact_v22_depth112_w010.jsonl`
- Fill-only codegen: `artifacts/codegen_tranhuy_04150_plus_grid_exact_v22_audited5_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04150_plus_grid_exact_v22_audited5_w010/submission.zip`
- Kaggle payload: `artifacts/kaggle-payload-grid-exact-v22-w010`
- Upload ZIP: `artifacts/kaggle-payload-grid-exact-v22-w010.zip`
- Retrieval SHA-256: `f3a5f55b737c431ed8a927587a7b4097280f023d54eeb68ec3cab0bde2fca608`
- Rule SHA-256: `63129f407af2f8a6818d6b4375b034bd5fe31a1b075180134c681569b84a76cb`
- Codegen SHA-256: `10059ff7c58758522b75685a4a772a58a3294809f363b451d79c3a961348e283`
- Submission SHA-256: `d47fb250f3452a2d90e8eb95ba23d86e62ec734c65f5ae34facbc1b059761129`
- Payload ZIP SHA-256: `e512a6ddd08f68384c282d9ad4b00f85d9433a7c32a253e4a614ee199b36f2bf`

V22 adds a fail-closed raw-grid fallback for canonical VAS rows. It is used
only when the normal exact path fails and requires all four signals to agree:
canonical row label, expected VAS code, target-period header and the matching
tidy CSV cell. This recovers tables whose OCR dropped the statement title or
placed a printed row number before the actual VAS code. Explicit header years
now override generic `Đơn vị tính: VND` period heuristics, and a generic
`Số đầu năm`/`Năm trước` column in filing `Y+1` can prove period `Y`.

Audited answers:

```text
388    22.88    mean gross margin of five negative-CFO-margin firms
442    47.70    KBC wins revenue growth inside the below-median D/E cohort
443    18.44    DBC wins revenue growth inside the below-median D/E cohort
446    50.38    profit share of DBC, QNS and VNM
466    73.66    positive-profit share of MCH, MPC, QNS and VNM
```

The merge changes exactly five failed rows, preserves the other 1,007 V21
records and lowers failed rows from 16 to 11. ID 510 remains rejected due to a
unit warning and confidence 40. All 368 tests pass; all 1,012 expressions
compile and replay from 2,102 packaged CSV files with official 1-based
`<table>` line positions. Both ZIP archives pass integrity checks. V22 can be
submitted directly without rerunning Qwen.

## Candidate - full-checkpoint exact lookup challenger v23 audited15

- Date: 2026-08-26
- Status: leaderboard confirmed
- Base candidate: V22 codegen on V21 retrieval
- Leaderboard anchor: V21 execution `0.4150`
- Audit scope: all `383` successful LLM single-vote rows
- Lookup cohort: `175` rows
- Exact agreement: `12` rows
- Audited exact replacements: `15` rows
- Refused lookup rows: `148`
- Allowlist: `67, 69, 73, 77, 85, 94, 177, 178, 183, 220, 237, 282, 316, 346, 681`
- Audit directory: `artifacts/challenger-audit-v23-llm-single-vote`
- Codegen: `artifacts/codegen_tranhuy_04150_plus_exact_lookup_v23_audited15_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04150_plus_exact_lookup_v23_audited15_w010/submission.zip`
- Audit summary SHA-256: `cf192c77d65b46c731f402df35931ad77649bcfa065457c94601c3329307b45c`
- Audit matrix SHA-256: `ab8c9f3f5b8d8bf59c0e96e3a0edbac60c52b8a2ea86b15455aaa6ac35019eb3`
- Codegen SHA-256: `115b0c950cad8c796b9bf01ab34cc668eb1b64c8535e4cd40c86767d2604dbb9`
- Submission SHA-256: `1af07fb2bbfd7c092af7a42d5885ea204bb4ed86b2332580da7ae68f405955d6`

V23 is the first shadow audit of successful answers rather than another
fill-only pass over the remaining 11 failures. The exact lookup challenger
requires one ticker, one year, one non-derived canonical metric, one resolved
cell, compatible units and a replayable query. It fails closed on opening
periods, unsupported child qualifiers and implausible scaled amounts.

The 175 lookup rows produce 27 exact candidates: 12 agree with Qwen and 15
audited disagreements are accepted. Eight replacements use exact current-year
VAS rows and seven use exact note totals. Unsafe candidates involving bank-only
subtotals, related parties, depreciation, foreign currency, deposit-interest
detail and opening dates are rejected before merge.

```text
 67       890.93    94          8.30    177       40802.32
 69 259236746.00    73          1.71     77    13976356.00
 85   9556360.00   178         13.45    183    25061907.00
220 1740391368.00  237        590.76    282          1.68
316        52.50   346  290816086.00    681        570.91
```

The merge changes exactly the 15 allowlisted rows and preserves all other 997
V22 records. It keeps 1,001 successful rows and 11 failed rows. All 374 tests
pass; all 1,012 expressions compile and replay, and the submission ZIP passes
integrity checks. No Qwen rerun is required. The next challenger cohort is the
60 LLM single-vote `difference` rows, using exact two-operand resolution and
the same shadow-audit contract.

Leaderboard evaluation returned `TABLES_F2MACRO=0.5534`,
`DOCS_F2MACRO=0.9416`, and `EXECUTION_ACCURACY=0.4407`. Execution improved by
`0.0257` over V21, approximately 13 additional correct answers on the inferred
506-question scoring subset. This validates 13 of the 15 audited replacements
at net level and makes V23 the new execution checkpoint.

## Candidate - exact two-operand difference challenger v24 audited7

- Date: 2026-08-26
- Status: leaderboard confirmed
- Base checkpoint: V23 leaderboard execution `0.4407`
- Audit scope: `60` LLM single-vote difference routes
- Exact agreement: `1` row (`793`)
- Audited exact replacements: `7` rows
- Refused difference rows: `52`
- Allowlist: `581, 592, 737, 776, 777, 798, 808`
- Audit directory: `artifacts/challenger-audit-v24-difference-single-vote`
- Codegen: `artifacts/codegen_tranhuy_04407_plus_exact_difference_v24_audited7_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04407_plus_exact_difference_v24_audited7_w010/submission.zip`
- Audit summary SHA-256: `3cad66aa3401758539715d703074c4406c8232f701c91ffec55e93df7296f33f`
- Audit matrix SHA-256: `884e118ac770687a3451f84bee7ecf1332936569463708f071298be0861d9f94`
- Codegen SHA-256: `4a8a844a73ac4d1c70b996dbb78d5143c6d7f4c0efe7c41056421e52a31f677f`
- Submission SHA-256: `30ae5e0f5474dc8d45e7aed7a8b3a972976438b896d7e7ff9e689114011750a8`

V24 requires exactly two non-derived requirements for the same canonical
metric along exactly one varying axis: either two companies in one year or one
company in two years. Both cells must resolve exactly and remain distinct.
Qualifier coverage, period identity, unit plausibility and query replay are
fail-closed. Generic “chênh lệch bao nhiêu” uses absolute magnitude, while an
explicit “mức thay đổi từ ... đến ...” preserves later-minus-earlier direction.

```text
581       3053.82    DBC tangible fixed assets, 2019 versus 2015
592     187120.00    BID customer-loan provision balance, 2024 versus 2022
737      15487.70    ASM versus BAF cost of goods sold
776          2.27    MSN versus MML parent-company net profit
777       1742.62    MCH versus VNM current income-tax expense
798         67.56    HHV versus VSC parent-company tangible fixed assets
808        132.69    DCM versus DPM total assets
```

The audit also fixed an opening-period guard that previously matched `1/1`
inside closing date `31/12`. The merge changes exactly the seven allowlisted
rows and preserves all other 1,005 V23 records. It keeps 1,001 successful rows
and 11 failures. All 381 tests pass; all 1,012 expressions compile and replay,
and the submission ZIP passes integrity checks. No Qwen rerun is required.

Leaderboard evaluation returned `TABLES_F2MACRO=0.5534`,
`DOCS_F2MACRO=0.9416`, and `EXECUTION_ACCURACY=0.4427`. Execution improved by
only `0.0020` over V23, one net correct answer. V24 remains the new checkpoint
because it is positive, but generic difference replacement is closed until
labeled evidence can identify which of the seven replacements regressed.

## Candidate - exact direct-ranking challenger v25 audited5

- Date: 2026-08-26
- Status: leaderboard confirmed
- Base checkpoint: V24 leaderboard execution `0.4427`
- Audit scope: `67` LLM single-vote ranking routes
- Exact agreements: `8` rows
- Audited exact replacements: `5` rows
- Refused ranking rows: `54`
- Allowlist: `859, 886, 902, 911, 967`
- Audit directory: `artifacts/challenger-audit-v25-ranking-single-vote`
- Codegen: `artifacts/codegen_tranhuy_04427_plus_exact_ranking_v25_audited5_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04427_plus_exact_ranking_v25_audited5_w010/submission.zip`
- Audit summary SHA-256: `37278c841f3ca4d90c3afa81920acc3c071fe217473227b66f06d9c541228c98`
- Audit matrix SHA-256: `74fbb510a876349aa00d2e5f3d3549faf804c92d572469e3ea5a5b8e6ecf1ce5`
- Codegen SHA-256: `5a0590ffc0e97231c7951ef47c94cebc8fe9b81499ae7d9a188c3ff90b43f9cb`
- Submission SHA-256: `9ad5900bbb0903806b1e03d307e9fa76508accd04bc2dc1bfef1129c7ec4abe4`

V25 handles only direct max/min value ranking over one canonical metric. Every
candidate entity or year must resolve to a distinct exact cell; the solver
refuses select-then-project questions, derived metrics, unsupported outputs,
missing candidates and qualifier loss. A word-boundary fix also prevents
`ghi nhận mức` from being misread as the credit-limit qualifier `hạn mức`.

```text
859        335.75    DCM maximum closing bonus/welfare fund balance
886    4270530.00    VIB maximum customer-loan provision balance
902        325.68    VPI maximum cash and cash equivalents
911        145.18    VIF parent maximum tangible fixed assets
967   40469060.00    BID maximum customer-loan provision balance
```

The merge changes exactly five rows and preserves all other 1,007 V24 records.
It keeps 1,001 successful rows and 11 failures. All 386 tests pass; all 1,012
expressions compile and replay, and the submission ZIP passes integrity checks.
No Qwen rerun is required.

Leaderboard evaluation returned `TABLES_F2MACRO=0.5533`,
`DOCS_F2MACRO=0.9420`, and `EXECUTION_ACCURACY=0.4466`. Execution improved by
`0.0039` over V24, approximately two additional correct answers, so V25 is the
new checkpoint.

## Candidate - exact lookup coverage v26 audited15

- Date: 2026-08-26
- Status: leaderboard confirmed
- Base checkpoint: V25 leaderboard execution `0.4466`
- Audit scope: `160` lookup routes in `356` LLM single-vote rows
- Exact agreements: `14` rows
- Audited exact replacements: `15` rows
- Refused lookup rows: `131`
- Allowlist: `29, 30, 80, 100, 138, 140, 144, 163, 188, 217, 229, 244, 262, 263, 360`
- Audit directory: `artifacts/challenger-audit-v26-lookup-coverage`
- Codegen: `artifacts/codegen_tranhuy_04466_plus_exact_lookup_v26_audited15_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04466_plus_exact_lookup_v26_audited15_w010/submission.zip`
- Audit summary SHA-256: `e7dbadd3883e38710fcf544d16ff775b46b88d0ef77a43b53e9b5ad521844667`
- Audit matrix SHA-256: `c235734c8e4d377bdd29037dec139e3d2a64882c6e74b5d8bbe45b7a8e87056a`
- Codegen SHA-256: `aa1f4680296554d3aefef338c6388ace0bcf4142aaf2a270d7aa52927fdd412d`
- Submission SHA-256: `21815fa72f847776e5810d543b5140e5b3a44c499173abd2f23213e1ac0f091b`

V26 extends the full-checkpoint lookup audit instead of touching the remaining
failed rows. It adds standard VAS identity for supplier prepayments, other
short-term receivables, investments in associates and other equity
investments. Canonical note metrics now carry typed column phrases, allowing
the resolver to distinguish share counts, fixed-rate financial assets, gross
bad receivables and FVTPL fair value from neighboring matrix cells.

All 15 replacements were audited against their full source block. Nine use
current-year VAS rows and six use exact note row/column intersections. A new
counterparty guard rejects a named-company bond question that otherwise maps
to the aggregate bonds-issued row. The merge changes exactly 15 rows and
preserves the other 997 checkpoint rows. All 394 tests pass, all 1,012 queries
compile and replay, and the ZIP contains 2,088 verified CSV files. No Qwen
rerun is required.

Leaderboard evaluation returned `TABLES_F2MACRO=0.5533`,
`DOCS_F2MACRO=0.9420`, and `EXECUTION_ACCURACY=0.4565`. Execution improved by
`0.0099` over V25, approximately five additional correct answers, so V26 is
the new checkpoint.

## Candidate - exact direct-average challenger v27 audited8

- Date: 2026-08-26
- Status: leaderboard confirmed
- Base checkpoint: V26 leaderboard execution `0.4565`
- Audit scope: `45` average routes in `341` LLM single-vote rows
- Exact agreements: `2` rows
- Audited exact replacements: `8` rows
- Refused average rows: `35`
- Allowlist: `816, 856, 867, 919, 940, 943, 947, 954`
- Audit directory: `artifacts/challenger-audit-v27-average-single-vote`
- Codegen: `artifacts/codegen_tranhuy_04565_plus_exact_average_v27_audited8_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04565_plus_exact_average_v27_audited8_w010/submission.zip`
- Audit summary SHA-256: `e54205750906a9a41345b83352325708862d9b4caa7712b7fd736996abdbdc02`
- Audit matrix SHA-256: `6d254f800db5ff70a2e60fe8610bae454999147cc48171578f1679a38eb7761c`
- Codegen SHA-256: `a3be4141d00eea7e3ea9535d80a16d196508e1d357d76efa4830425cf5e3d960`
- Submission SHA-256: `ffbc712d2963ed1b3ba583766e98191e6400ae14890b63f680f3d3395e858498`

The challenger accepts only a direct arithmetic mean of one atomic canonical
metric across one entity/year axis. Every operand must resolve to a distinct
exact cell. Derived ratios, conditional cohorts, mixed metrics, regional
details and missing periods fail closed. The eight replacements cover
short-term borrowings, basic EPS, liabilities, trade payables, financial
income, cash, cost of goods sold and interest expense.

All eight replacements use current-year VAS rows. The merge preserves the
other 1,004 checkpoint rows. All 397 tests pass, all 1,012 queries compile and
replay, and the ZIP contains 2,092 verified CSV files. No Qwen rerun is
required.

Leaderboard evaluation returned `TABLES_F2MACRO=0.5530`,
`DOCS_F2MACRO=0.9420`, and `EXECUTION_ACCURACY=0.4625`. Execution improved by
`0.0060` over V26, approximately three additional correct answers, so V27 is
the new checkpoint.

## Candidate - exact direct-growth challenger v28 audited3

- Date: 2026-08-26
- Status: locally audited; waiting for leaderboard
- Base checkpoint: V27 leaderboard execution `0.4625`
- Audit scope: `18` growth routes in `333` LLM single-vote rows
- Exact disagreements: `4` rows
- Audited exact replacements: `3` rows
- Refused growth rows: `14`
- Allowlist: `586, 631, 647`
- Excluded false parent match: `633`
- Audit directory: `artifacts/challenger-audit-v28-growth-single-vote`
- Codegen: `artifacts/codegen_tranhuy_04625_plus_exact_growth_v28_audited3_w010.jsonl`
- Submission: `artifacts/submission_tranhuy_04625_plus_exact_growth_v28_audited3_w010/submission.zip`
- Audit summary SHA-256: `2b05583ac4bd58b33a0b02a589bd573d94365377f92f98efbbdff95a0c4eb5ce`
- Audit matrix SHA-256: `705db962713c07583bf6de93569872862026b89d82e9088681a2b1492023d1d0`
- Codegen SHA-256: `b22adf22d34315ac10411451f0a153947c28382c314a79eba9fdfadd93d97d8d`
- Submission SHA-256: `ac99946b373559a6635bceff89cfa4c0a4cb026eeec7ee657954b67a07373b2e`

V28 computes direct endpoint growth only when all requirements share one
atomic metric and one company. It sorts the canonical periods, uses the
earliest as base and latest as end, normalizes accounting expense/provision
signs, rejects zero bases and replays the exact two-cell pandas expression.
New scope guards reject term-deposit and construction-activity detail when a
canonical parent line cannot represent that child.

ID 633 was excluded even though it resolved to two current-year VAS cells:
the question asks for the VietsovPetro counterparty amount, while those cells
are aggregate supplier prepayments. The accepted values are:

```text
586      335.82    ACV cash and cash equivalents, 2021 to 2022
631       31.31    IJC parent-company selling expense, 2015 to 2018
647       37.46    ACB customer-loan provision balance, 2017 to 2019
```

The merge changes exactly three rows and preserves all other 1,009 V27 rows.
All 401 tests pass; all 1,012 expressions compile and replay, and the ZIP
contains 2,092 verified CSV files. No Qwen rerun is required.

Leaderboard evaluation returned `TABLES_F2MACRO=0.5530`,
`DOCS_F2MACRO=0.9420`, `TABLES_PRECISION=0.3621`,
`TABLES_RECALL=0.7480`, `DOCS_PRECISION=0.9494`,
`DOCS_RECALL=0.9457`, and `EXECUTION_ACCURACY=0.4644`. Execution improved by
`0.0019` over V27, approximately one additional correct answer. V28 is the new
checkpoint, but direct growth should not be broadened without labeled evidence:
three audited replacements yielded only one net correct answer. The next
isolated challenger cohort is direct ratio.
