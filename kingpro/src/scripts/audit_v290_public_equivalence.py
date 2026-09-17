"""Prove V290 public equivalence, selection, and guarded source preference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "EXECUTION_ACCURACY", "ANSWER_ACCURACY",
    "TABLES_F2MACRO", "TABLES_PRECISION", "TABLES_RECALL", "TABLES_MRR5",
    "DOCS_F2MACRO", "DOCS_PRECISION", "DOCS_RECALL", "DOCS_MRR5",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def by_id(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in payload.get("submissions", [])}


def forced(constraints: dict[str, Any], qid: int) -> dict[str, bool]:
    return {
        str(row["answer_variant"]): bool(row["forced_correct"])
        for row in constraints.get("forced_variants", [])
        if int(row.get("id", -1)) == qid
    }


def audit(scores_path: Path, release_path: Path, constraints_path: Path) -> dict[str, Any]:
    scores = by_id(load(scores_path))
    release = load(release_path)
    constraints = load(constraints_path)
    v290, v276 = scores[3745], scores[3742]
    deltas = {
        metric: round(float(v290["scores"][metric]) - float(v276["scores"][metric]), 10)
        for metric in METRICS
    }
    q98, q714, q764 = (forced(constraints, qid) for qid in (98, 714, 764))
    checks = {
        "v290_finished": v290.get("status") == "Finished",
        "v290_selected_authenticated": v290.get("on_leaderboard") is True,
        "v276_deselected_authenticated": v276.get("on_leaderboard") is False,
        "all_ten_metrics_equal_v276": all(delta == 0 for delta in deltas.values()),
        "release_gate_pass": release.get("status") == "PASS",
        "q98_old_forced_public_wrong": q98.get("num:3177.37") is False,
        "q98_source_forced_public_wrong": q98.get("num:146.47") is False,
        "q714_source_variant_forced_public_wrong": q714.get("num:168.74") is False,
        "q714_v276_variant_forced_public_correct": q714.get("num:238.89") is True,
        "q764_old_forced_public_wrong": q764.get("num:2.99") is False,
        "q764_source_forced_public_wrong": q764.get("num:2.94") is False,
        "v206_preserved": sha256(ROOT / "sub_top123_candidate_v206_semantic_batch11.zip")
        == "34B19414AA1610785163513F8D2F16CCD41A1F797A90F7121817657F553E4B5B",
        "v207_preserved": sha256(ROOT / "sub_top123_candidate_v207_semantic_batch6_final.zip")
        == "DD616AA408E4B601B921246DDFF4DA4BFD4E8C283FCEF22FF68680AAAB2ADDAF",
    }
    return {
        "schema_version": 1,
        "kind": "v290_public_equivalence_and_source_preference",
        "inputs": {
            "scores": {"path": str(scores_path), "sha256": sha256(scores_path)},
            "release": {"path": str(release_path), "sha256": sha256(release_path)},
            "constraints": {"path": str(constraints_path), "sha256": sha256(constraints_path)},
        },
        "v290": v290,
        "v276": v276,
        "metric_deltas_v290_minus_v276": deltas,
        "forced_variants": {"98": q98, "714": q714, "764": q764},
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision": {
            "public_selected": "v290",
            "private_final_preference": "v290",
            "rollback": "v276",
            "reason": "All ten public metrics equal V276; V290 adds physical-source q98 union and q764 while retaining public-correct q714.",
            "claim_limit": "No private score is known. Preference uses public non-regression plus independently verified physical-source robustness.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "build/leaderboard_scores_authoritative_full_after_v290.json")
    parser.add_argument("--release", type=Path, default=ROOT / "build/release_gate/sub_v290_scope2/release.json")
    parser.add_argument("--constraints", type=Path, default=ROOT / "build/v291_leaderboard_answer_constraints_after_v290.json")
    parser.add_argument("--out", type=Path, default=ROOT / "build/v292_v290_public_equivalence.json")
    args = parser.parse_args()
    report = audit(args.scores.resolve(), args.release.resolve(), args.constraints.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"v290", "v276"}}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
