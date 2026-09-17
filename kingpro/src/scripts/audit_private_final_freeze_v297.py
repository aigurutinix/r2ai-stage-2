"""Fail-closed audit for the V297 private/final submission freeze."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
V297_SHA = "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC"
LOCKS = {
    "sub_v297_scope2_a.zip": V297_SHA,
    "sub_v290_scope2_a.zip": "711A3493279387593ECA17C3C4130154E9F860CA891B91013083E7D76E5C04A4",
    "sub_v276_q638_fix.zip": "86A52DA0FE9A9121C6BB08191FDC2FABFDD97620020B9F3F46DAB81B2258478E",
    "sub_v269_lineage_control.zip": "C933B5D901836C1414DAB801DAAA77BBA8654BC64572F208A8D8730959B935A0",
    "sub_top123_candidate_v206_semantic_batch11.zip": "34B19414AA1610785163513F8D2F16CCD41A1F797A90F7121817657F553E4B5B",
    "sub_top123_candidate_v207_semantic_batch6_final.zip": "DD616AA408E4B601B921246DDFF4DA4BFD4E8C283FCEF22FF68680AAAB2ADDAF",
}


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "build/private_final_leaderboard_check_v297.json")
    parser.add_argument("--out", type=Path, default=ROOT / "build/private_final_freeze_v297.json")
    args = parser.parse_args()
    scores = {int(row["id"]): row for row in load(args.scores).get("submissions", [])}
    release = load(ROOT / "build/release_gate/sub_v297_scope2/release.json")
    dominance = load(ROOT / "build/v299_v297_public_dominance.json")
    proxy = load(ROOT / "build/demo_compliance/private_proxy_cohorts_hanoi_v297_v51.json")
    preflight = load(ROOT / "build/demo_compliance/hanoi_preflight_20260828_v297_v52_live.json")
    receipt = load(ROOT / ".leaderboard-selection/submission-3747.json")
    artifacts = []
    for name, expected in LOCKS.items():
        path = ROOT / name
        actual = sha256(path) if path.is_file() else None
        artifacts.append({"path": name, "exists": path.is_file(), "expected_sha256": expected, "sha256": actual, "ok": actual == expected})
    with zipfile.ZipFile(ROOT / "sub_v297_scope2_a.zip") as archive:
        archive.testzip()
        v297_entries = len(archive.namelist())
    current = scores.get(3747, {})
    checks = {
        "v297_finished": current.get("status") == "Finished",
        "v297_authenticated_selected": current.get("on_leaderboard") is True,
        "v290_not_selected": scores.get(3745, {}).get("on_leaderboard") is False,
        "selection_receipt_confirmed": receipt.get("state") == "confirmed" and receipt.get("sha256") == V297_SHA,
        "release_gate_pass": release.get("status") == "PASS",
        "release_archive_hash_locked": release.get("archive", {}).get("sha256") == V297_SHA,
        "release_archive_crc_pass": release.get("archive", {}).get("crc_and_full_read") == "pass",
        "public_dominance_pass": dominance.get("status") == "PASS",
        "private_proxy_pass": proxy.get("passed") is True,
        "hanoi_stage_safe": preflight.get("stage_safe") is True,
        "fallback_ready": preflight.get("fallback_ready") is True,
        "dynamic_claim_stays_locked": preflight.get("full_dynamic_ready") is False,
        "all_artifact_hashes_locked": all(item["ok"] for item in artifacts),
        "v297_archive_layout_nonempty": v297_entries == 1885,
        "no_v312_archive_exists": not any(ROOT.glob("sub_v312*.zip")),
    }
    report = {
        "schema_version": 1,
        "kind": "private_final_submission_freeze",
        "selected": {"version": "v297", "submission_id": 3747, "filename": "sub_v297_scope2_a.zip", "sha256": V297_SHA},
        "rollback_order": ["v290/3745", "v276/3742", "v269/3741"],
        "checks": checks,
        "artifacts": artifacts,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "operator_instruction": "Do not upload or select another submission unless this freeze audit is deliberately superseded by a newly signed release.",
        "claim_limit": "No private score is known. V297 is selected because it publicly dominates V290 and has stronger source lineage; dynamic/manual Hanoi attestations remain separate.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
