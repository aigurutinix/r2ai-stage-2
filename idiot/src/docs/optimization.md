# Optimization baseline and promotion gates

This document records the measurable outcome of the semantic preprocessing
upgrade. Generated artifacts stay outside Git under `artifacts/` and benchmark
reports stay under `runs/optimization/`.

## Reproducible artifacts

| Artifact | Result |
|---|---:|
| `tables_v2.sqlite3` documents | 1,973 |
| tables | 146,246 |
| rows | 1,535,824 |
| build time | 584.849 s |
| index size | 1,175,851,008 bytes |
| `financial_panel_v2.json` cells | 52,066 |
| panel build time | 63.918 s |
| artifact audit | 0 errors, 0 warnings |

The original index and panel are never overwritten by these commands. The
solver checkpoint signature includes index/panel path, size and modification
time, so a checkpoint produced with one artifact set cannot silently be reused
with another.

## Retrieval evaluation

The retrieval benchmark uses 124 source-audited Easy questions.

| Retriever | R@1 | R@5 | R@10 | R@20 | MRR |
|---|---:|---:|---:|---:|---:|
| calibrated legacy row | .2661 | .5000 | .5726 | .6210 | .3728 |
| semantic row (experimental) | .2419 | .4597 | .5403 | .5887 | .3373 |
| table→legacy-row cascade | .2581 | .4919 | .5806 | .6210 | .3687 |
| semantic table | .3548 | .8065 | .9194 | .9677 | .5534 |

Decision: keep the calibrated row scorer as the default. Use semantic table
retrieval as a separate shortlist/gating stage. Semantic row priors remain
opt-in (`semantic=True`) because enabling them globally is a measured
regression. The cascade is useful as a diverse shortlist source (R@10 and
R@50 improve), but it does not replace the default ordering because top-1 and
MRR are slightly lower.

## Panel replay

The candidate panel differs from the historical panel in 3,546 stored keys
(3,472 numeric values and 1,756 source selections; the sets overlap). The large
diff is expected because v2 rejects explicit prior/opening periods and applies
units per table/cell rather than carrying stale page context forward.

Deterministic replay results:

| Route/family | Questions replayed | Answer changes | Errors |
|---|---:|---:|---:|
| Hard | 159 | 0 | 0 |
| Template 578–655 | 44 | 1 | 0 |
| Template 656–732 | 34 | 0 | 0 |
| Template 733–812 | 45 | 0 | 0 |
| Template 813–912 | 47 | 0 | 0 |
| Template 913–1012 | 34 | 0 | 0 |

The only numeric answer change is Q632. The old panel treated the 2016 MSR
balance as VND while treating the 2020 balance as thousand VND, producing an
implausible growth of 126,161.8448%. Both source tables explicitly say
`Nghìn VND`; v2 applies the same ×1,000 scale and produces 26.2618448%. The
document/table/row/column coordinates are unchanged.

The final deterministic solver was also compared directly with the submitted
`submission_vn31.zip` baseline. All 159 Hard and 435 Template answers reproduce
the release exactly (594/594, zero errors). Seven source-backed recipes which
older packaging scripts had applied only while building a ZIP are now part of
the durable Template registry, so a clean solve no longer loses the best-known
release corrections.

## Runtime

Profiling a representative un-audited Template question (Q579) exposed 3,111
full table-semantic reconstructions and 42.6 million function calls. Caching
issuer/year/scope candidates and table previews—and one `TableAnalyzer` per
table when semantic-column mode is enabled—reduced the first solve from 14.55 s
to 0.29 s and a warm repeat to 0.26 s, with the same answer.
Family benchmarks are range-addressable (`--qid-min`, `--qid-max`) and print
progress every 25 questions, so long validation runs can be checkpointed.

## Promotion policy

1. Never replace historical artifacts or submissions in place.
2. Run `audit_artifacts.py`; require zero errors and zero warnings.
3. Replay Hard and all five Template families; audit every answer-changing ID.
4. Keep learned Easy ranking calibrated; do not stack unvalidated semantic row
   priors on top of it.
5. Promote a candidate panel only into a fresh solve run and run the full ZIP
   release audit with repeated Pandas replays.

Local validation establishes integrity, reproducibility and regression safety;
it cannot guarantee a leaderboard score because per-question gold answers are
not public.
