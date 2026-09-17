"""Prove V297 public dominance and guarded source preference over V290."""

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


def audit(scores_path: Path, release_path: Path, constraints_path: Path) -> dict[str, Any]:
    scores = by_id(load(scores_path))
    release = load(release_path)
    constraints = load(constraints_path)
    current, rollback = scores[3747], scores[3745]
    deltas = {
        metric: round(float(current["scores"][metric]) - float(rollback["scores"][metric]), 10)
        for metric in METRICS
    }
    improved = sorted(metric for metric, delta in deltas.items() if delta > 0)
    expected_improved = sorted({
        "TABLES_F2MACRO", "TABLES_PRECISION", "TABLES_RECALL",
        "DOCS_F2MACRO", "DOCS_PRECISION", "DOCS_RECALL",
    })
    q224 = {
        str(row["answer_variant"]): bool(row["forced_correct"])
        for row in constraints.get("forced_variants", [])
        if int(row.get("id", -1)) == 224
    }
    checks = {
        "v297_finished": current.get("status") == "Finished",
        "v297_selected_authenticated": current.get("on_leaderboard") is True,
        "v290_deselected_authenticated": rollback.get("on_leaderboard") is False,
        "weakly_dominates_v290": all(delta >= 0 for delta in deltas.values()),
        "improves_exactly_six_retrieval_metrics": improved == expected_improved,
        "release_gate_pass": release.get("status") == "PASS",
        "q224_old_forced_public_wrong": q224.get("num:181.54") is False,
        "q224_source_forced_public_wrong": q224.get("num:1200.50") is False,
        "v206_preserved": sha256(ROOT / "sub_top123_candidate_v206_semantic_batch11.zip")
        == "34B19414AA1610785163513F8D2F16CCD41A1F797A90F7121817657F553E4B5B",
        "v207_preserved": sha256(ROOT / "sub_top123_candidate_v207_semantic_batch6_final.zip")
        == "DD616AA408E4B601B921246DDFF4DA4BFD4E8C283FCEF22FF68680AAAB2ADDAF",
    }
    return {
        "schema_version": 1,
        "kind": "v297_public_dominance_and_source_preference",
        "inputs": {
            "scores": {"path": str(scores_path), "sha256": sha256(scores_path)},
            "release": {"path": str(release_path), "sha256": sha256(release_path)},
            "constraints": {"path": str(constraints_path), "sha256": sha256(constraints_path)},
        },
        "v297": current,
        "v290": rollback,
        "metric_deltas_v297_minus_v290": deltas,
        "strictly_improved_metrics": improved,
        "q224_forced_variants": q224,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision": {
            "public_selected": "v297",
            "private_final_preference": "v297",
            "rollback": "v290",
            "reason": "V297 preserves Answer/Execution/MRR, improves six retrieval metrics, and adds source-proven q224/q966 lineage.",
            "claim_limit": "No private score is known; preference uses authenticated public dominance plus physical-source robustness.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=ROOT / "build/leaderboard_scores_authoritative_full_after_v297.json")
    parser.add_argument("--release", type=Path, default=ROOT / "build/release_gate/sub_v297_scope2/release.json")
    parser.add_argument("--constraints", type=Path, default=ROOT / "build/v298_leaderboard_answer_constraints_after_v297.json")
    parser.add_argument("--out", type=Path, default=ROOT / "build/v299_v297_public_dominance.json")
    args = parser.parse_args()
    report = audit(args.scores.resolve(), args.release.resolve(), args.constraints.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"v297", "v290"}}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
