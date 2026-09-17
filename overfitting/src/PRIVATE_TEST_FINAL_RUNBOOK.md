# Private Test Final Runs

Historical runbook for private challengers. The final confirmed checkpoint is
**V67 (private Execution Accuracy 0.5277)**; see [CHECKPOINT_V67.md](CHECKPOINT_V67.md)
for its artifacts and reconstruction dependencies. The submission numbers below
describe the earlier plan, not the current remaining submission budget.

This runbook preserves the five-submission budget. Never upload a raw Kaggle
codegen JSONL; every challenger must pass the local consensus/replay gate first.

## Submission #2: deterministic V29

Upload only:

```text
artifacts/submission_private_v29_exact28_w010/submission.zip
```

SHA-256:

```text
9dd0bce47afe22555affe19e9ad3ca1ca81a8f8e4e9fd7efba8b5f9c671b7a70
```

## Submission #3: Qwen 14B N=5

Kaggle Dataset folder:

```text
artifacts/kaggle-payload-private-v29-qwen14b-n5
```

Notebook:

```text
kaggle/private-v29-qwen14b-n5-consensus.ipynb
```

Download this output from Kaggle:

```text
private_v29_qwen14b_n5_raw.jsonl
```

Place it anywhere locally and run:

```bash
python scripts/16_merge_consensus_codegen.py \
  --base artifacts/codegen_private_v29_exact28_w010.jsonl \
  --candidate <PATH>/private_v29_qwen14b_n5_raw.jsonl \
  --retrieval artifacts/retrieval_v29_private_exact_metrics3_depth112_w010.jsonl \
  --ids-file artifacts/private_v29_qwen14b_n5_target_ids.json \
  --base-source-token llm_select \
  --expected-samples 5 --min-votes 4 --min-confidence 78 \
  --out artifacts/codegen_private_v30_qwen14b_consensus.jsonl

python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval_v29_private_exact_metrics3_depth112_w010.jsonl \
  --codegen artifacts/codegen_private_v30_qwen14b_consensus.jsonl \
  --out-dir artifacts/submission_private_v30_qwen14b_consensus \
  --sub-k 5
```

Upload only `artifacts/submission_private_v30_qwen14b_consensus/submission.zip`.

## Submission #4: schema-tail reranker

Kaggle Dataset folder:

```text
artifacts/kaggle-payload-private-v30-schema-qwen14b-n5
```

Notebook:

```text
kaggle/private-v30-schema-qwen14b-n5.ipynb
```

Download:

```text
private_v30_schema_qwen14b_n5_raw.jsonl
```

Set `BEST_BEFORE_SCHEMA` to the codegen checkpoint with the better observed
private score: V29 from submission #2 or V30 from submission #3.

```bash
BEST_BEFORE_SCHEMA=artifacts/codegen_private_v30_qwen14b_consensus.jsonl

python scripts/16_merge_consensus_codegen.py \
  --base "$BEST_BEFORE_SCHEMA" \
  --candidate <PATH>/private_v30_schema_qwen14b_n5_raw.jsonl \
  --retrieval artifacts/retrieval_private_v30_schema_tail15_w010.jsonl \
  --ids-file artifacts/private_v30_schema_target_ids.json \
  --expected-samples 5 --min-votes 4 --min-confidence 78 \
  --out artifacts/codegen_private_v31_schema_consensus.jsonl

python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval_private_v30_schema_tail15_w010.jsonl \
  --codegen artifacts/codegen_private_v31_schema_consensus.jsonl \
  --out-dir artifacts/submission_private_v31_schema_consensus \
  --sub-k 5
```

Upload only `artifacts/submission_private_v31_schema_consensus/submission.zip`.

## Submission #5: conservative ensemble

Set `BEST_PRIVATE_CODEGEN` to whichever checkpoint has the highest observed
private score after submissions #2-#4. The ensemble accepts a changed answer
only when both verified runs agree, or when the only verified run is 5/5.

```bash
BEST_PRIVATE_CODEGEN=artifacts/codegen_private_v31_schema_consensus.jsonl

python scripts/18_build_private_ensemble.py \
  --base "$BEST_PRIVATE_CODEGEN" \
  --candidate-a <PATH>/private_v29_qwen14b_n5_raw.jsonl \
  --candidate-b <PATH>/private_v30_schema_qwen14b_n5_raw.jsonl \
  --retrieval-a artifacts/retrieval_v29_private_exact_metrics3_depth112_w010.jsonl \
  --retrieval-b artifacts/retrieval_private_v30_schema_tail15_w010.jsonl \
  --expected-samples 5 --min-votes 4 --min-confidence 78 \
  --out artifacts/codegen_private_v32_final_ensemble.jsonl

python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval_private_v30_schema_tail15_w010.jsonl \
  --codegen artifacts/codegen_private_v32_final_ensemble.jsonl \
  --out-dir artifacts/submission_private_v32_final_ensemble \
  --sub-k 5
```

Before every upload, require all of the following:

```bash
unzip -t <submission-dir>/submission.zip
shasum -a 256 <submission-dir>/submission.zip
```

The submission builder already replays all 1,012 pandas expressions against
the exact packaged CSV evidence and stops on any mismatch.

## V33-V36: code-mode 2-of-3 challenger (do not submit raw runs)

This batch targets the 282 rows left unresolved after the earlier selection-mode
consensus. It uses code generation because these questions need arithmetic and
multi-cell evidence, not another single-row selection pass.

### Important rerun status (2026-09-03)

The first downloaded V33, V34 and V35 outputs are invalid: every one of their
1,012 rows has `source=none`, `votes=0` and `n_ok=0`. They contain no Qwen
challenger and must not be merged or submitted. The most likely failure was CUDA
OOM from generating all five return sequences together; the notebook shell call
did not propagate the non-zero process exit code.

The three payload folders below have been rebuilt with:

- independent OOM adaptation for prompt batch and return-sequence batch;
- a three-question smoke test before the full 282-question run;
- per-question raw-generation diagnostics when code cannot execute;
- `subprocess.run(..., check=True)` so Kaggle stops on a failed runner;
- a final assertion that at least one full-run LLM candidate exists.
- `private-n5-safe-v4`: left-side token truncation plus a repeated final task
  anchor, preventing OOM retries from cutting off the generation marker and
  making Qwen continue raw CSV text;
- a one-sample smoke path and 3,600-token/256-output defaults in the notebooks,
  reducing startup and per-question T4 time while the full run remains N=5.

Create a **new version** of each existing Kaggle dataset and rerun the updated
notebook. A valid download must report at least one `source` beginning with
`llm`; in practice the count should be much larger than one. Keep the `.smoke`
file only for diagnostics and download the main `*_raw.jsonl` file for merging.

Upload these three folders as private Kaggle datasets:

```text
artifacts/kaggle-payload-private-v33-code-n5
artifacts/kaggle-payload-private-v34-schema-code-n5
artifacts/kaggle-payload-private-v35-note-ops-code-n5
```

Run the corresponding notebooks. V33 and V35 are the minimum independent pair;
V34 is the third vote that increases coverage and resolves one-run failures.

```text
kaggle/private-v33-qwen14b-code-n5.ipynb
kaggle/private-v34-schema-qwen14b-code-n5.ipynb
kaggle/private-v35-note-ops-qwen14b-code-n5.ipynb
```

Download these outputs:

```text
private_v33_qwen14b_code_n5_raw.jsonl
private_v34_schema_qwen14b_code_n5_raw.jsonl
private_v35_note_ops_qwen14b_code_n5_raw.jsonl
```

Do not submit any output directly. Build the row-level verified 2-of-3 ensemble:

```bash
python scripts/21_build_private_code_ensemble.py \
  --base artifacts/codegen_private_v31_schema_consensus_from_v29.jsonl \
  --candidate-a <PATH>/private_v33_qwen14b_code_n5_raw.jsonl \
  --retrieval-a artifacts/retrieval_v29_private_exact_metrics3_depth112_w010.jsonl \
  --candidate-b <PATH>/private_v34_schema_qwen14b_code_n5_raw.jsonl \
  --retrieval-b artifacts/retrieval_private_v30_schema_tail15_w010.jsonl \
  --candidate-c <PATH>/private_v35_note_ops_qwen14b_code_n5_raw.jsonl \
  --retrieval-c artifacts/retrieval_private_v35_note_ops2_depth112_w010.jsonl \
  --expected-samples 5 --min-votes 4 --k 15 \
  --out artifacts/codegen_private_v36_code_ensemble_2of3.jsonl \
  --audit-out artifacts/codegen_private_v36_code_ensemble_2of3.audit.jsonl
```

The verifier requires a fresh replay, no semantic warnings, at least 4/5
within-run votes, the exact canonical row/column for every requirement, and the
same answer from the same source cells in at least two independent runs.
Uncanonicalized routes require unanimous 5/5 agreement in the agreeing runs.

Only after reviewing the accepted count and audit, build with the known private
retrieval checkpoint so table/doc quality does not regress:

```bash
python scripts/05_build_submission.py \
  --retrieval artifacts/retrieval_private_v30_schema_tail15_w010.jsonl \
  --codegen artifacts/codegen_private_v36_code_ensemble_2of3.jsonl \
  --out-dir artifacts/submission_private_v36_code_ensemble_2of3 \
  --sub-k 5
```

This is a challenger, not a promised `0.60` checkpoint. Spend another private
submission only after the audit shows broad, independently reproduced changes.

## Generated candidates after both Kaggle runs

Both raw outputs have now been verified. Operation-aware guards reject negative
absolute differences, ratios missing canonical evidence for either operand,
semantic/unit warnings and replay/evidence mismatches.

Recommended private upload order after deterministic V29:

1. **Schema consensus, 27 changed answers** (highest precision):

   ```text
   artifacts/submission_private_v31_schema_consensus_from_v29/submission.zip
   SHA-256 13290bbb4a5f214fc3804edf418c6c4f2b766b5ba984699a0edee29c8034d15b
   ```

2. **Cross-run ensemble, 35 changed answers** (medium breadth):

   ```text
   artifacts/submission_private_v32_final_ensemble_from_v29/submission.zip
   SHA-256 a027a73f6a4594a37188ad2d8f4ee0f257ce0a5d921e1f8a54f3d9879abcb8d3
   ```

3. **V29 Qwen14B consensus, 38 changed answers** (widest/riskier):

   ```text
   artifacts/submission_private_v30_qwen14b_consensus/submission.zip
   SHA-256 f28742d1df9f873f50db097e74a0a2319fe273e9fb216bccb76c73fab9d1e396
   ```

Do not automatically spend the last upload on the widest candidate. Use the
score trend from the first two private uploads to decide whether its additional
single-run changes are helping.
