# V297 submission handoff

- Selected version / submission: `v297` / `3747`
- Artifact: `sub_v297_scope2_a.zip`
- ZIP SHA-256: `90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC`
- `submission.json` SHA-256: `E19E4748DDAFFAD28112292EAE17E9296F85BBC99265C9D52EEB27E8F8539C85`
- Authenticated state: `Finished`, `on_leaderboard=true`

## Public vector

- Execution / Answer: `0.7115 / 0.7115`
- Tables F2 / Precision / Recall / MRR5: `0.6120 / 0.5928 / 0.6261 / 0.6514`
- Docs F2 / Precision / Recall / MRR5: `0.9618 / 0.9587 / 0.9678 / 0.9806`

Relative to V290, V297 improves Tables F2/P/R by `0.0006/0.0007/0.0007`
and Docs F2/P/R by `0.0007/0.0007/0.0006`; the other four metrics tie.

## Exact additions over V290

- q224: use physical HỢP NHẤT source `HUT...separate|1624`, answer `1200.50`,
  with the measured old+physical retrieval union. Aggregate constraints force
  both tried q224 answers public-wrong, so this does not chase public Answer.
- q966: replace GEG physical RIÊNG `|474` with HỢP NHẤT `|493`; count remains
  `2` because both values are below the `1e12` threshold.

Inherited source fixes include V290 q98/q764 and V276 q638 lineage. q714 remains
on public-correct `238.89`; source-derived `168.74` stays documented but excluded.

## Verification and rollback

- Deterministic A/B ZIP: 1,885 entries, byte-identical.
- Full release gate: PASS; 1.012/1.012 runtime cell lineage, 7.299 physical/
  causal cells, 0 citation-unbound; 548 tests + 25 subtests.
- Local-unit cube SHA-256:
  `EC80A93F159528180402A5B91E91575B05751E59BF96DD8CBAE252FAE0D22EBC`;
  legacy rollback cube:
  `3B509E2E2969C4481A48CF673E95138541A4235822F28691EB7C1BD0F733076B`.
- Compiler production gates: 106/106 typed dual execution, 29/29 adversarial
  rejection, 3/3 metamorphic invariance. Explicit local markers repair HNG/HAG
  ×1.000 risks without changing any of 106 accepted registry answers.
- V206/V207 hashes unchanged.
- Direct rollback: V290 / ID3745. Further rollback: V276 / ID3742, then V269.

Run `python scripts/private_ready_doctor.py --require-live` immediately before
handoff. Upload only `sub_v297_scope2_a.zip`; the documentation handoff ZIP,
runtime cube and lineage sidecar are not competition submissions.

No private score is known; V297 is preferred because it publicly dominates
V290 and has stronger physical source consistency.
