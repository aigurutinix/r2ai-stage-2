# Private-test final checklist — V297

## Submission to keep selected

- Version: **V297**
- Codabench submission ID: **3747**
- File: `sub_v297_scope2_a.zip`
- SHA-256: `90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC`
- Current authenticated state: `Finished`, `on_leaderboard=true`

Do not upload or select another artifact unless a new release intentionally
supersedes V297 and passes a fresh signed freeze audit. Private score is unknown;
V297 is preferred because it weakly dominates V290 on all ten public metrics,
strictly improves six retrieval metrics, and keeps stronger physical lineage.

## Final operator actions

1. On Codabench, verify row **3747** still carries the selected/on-leaderboard icon.
2. If the organizer asks teams to choose one final submission, choose **ID3747**.
3. If a file must be uploaded manually, upload only `sub_v297_scope2_a.zip` and
   verify its SHA-256 before choosing it.
4. Do not rename/repack the ZIP; the uploaded filename is already below the
   64-character limit and the archive layout/CRC is locked.
5. Keep V290/ID3745 as direct rollback, V276/ID3742 second, V269/ID3741 third.
6. Preserve v206/v207 unchanged; they are historical reproducibility artifacts,
   not final private candidates.

## One-command verification

```powershell
python scripts\private_ready_doctor.py --require-live
```

Expected result: `build/private_ready_doctor.json` has `status=PASS` and every
check is `true`. Doctor khóa đồng thời ZIP/submission, local-unit cube, rollback
cube, freeze receipt, 106/106 typed execution, 29/29 adversarial rejection,
3/3 metamorphic invariance, 1.012/1.012 cell lineage và live UI/backend.

Nếu cần refresh trạng thái leaderboard trước doctor:

```powershell
python scripts\leaderboard_scores.py --limit 5 --json `
  --out build\private_final_leaderboard_check_v297.json
python scripts\audit_private_final_freeze_v297.py `
  --out build\private_final_freeze_v297_local_units.json
```

## Hanoi/private claim boundary

- Allowed: public V297 scores, release PASS, source lineage, offline replay and
  compiler stage-safe evidence.
- Not allowed: claiming a private score, claiming the dynamic model is live
  without runtime identity attestation, or treating manual evidence 0/8 as done.
- If network/model is unavailable in Hà Nội, keep dynamic mode `LOCKED` and use
  the audited registry/compiler path.
- Local-unit cube là runtime artifact, không phải file nộp. Không upload cube,
  counterfactual sidecar hoặc private handoff ZIP lên BTC.
