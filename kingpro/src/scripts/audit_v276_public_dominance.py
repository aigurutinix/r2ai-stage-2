"""Prove V276's measured public dominance and guarded final preference."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRICS = (
    "EXECUTION_ACCURACY",
    "ANSWER_ACCURACY",
    "TABLES_F2MACRO",
    "TABLES_PRECISION",
    "TABLES_RECALL",
    "TABLES_MRR5",
    "DOCS_F2MACRO",
    "DOCS_PRECISION",
    "DOCS_RECALL",
    "DOCS_MRR5",
)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def by_id(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(row["id"]): row for row in payload.get("submissions", [])}


def audit(
    full_scores: Path,
    selected_scores: Path,
    release_report: Path,
    constraints_report: Path,
    *,
    root: Path = ROOT,
) -> dict[str, Any]:
    full = by_id(load(full_scores))
    selected = by_id(load(selected_scores))
    release = load(release_report)
    constraints = load(constraints_report)
    current = full[3742]
    fallback = full[3741]
    current_selected = selected[3742]
    current_scores = current["scores"]
    fallback_scores = fallback["scores"]
    deltas = {
        metric: round(float(current_scores[metric]) - float(fallback_scores[metric]), 10)
        for metric in METRICS
    }
    weakly_dominates = all(delta >= 0 for delta in deltas.values())
    strictly_improves = [metric for metric, delta in deltas.items() if delta > 0]
    q638 = {
        str(item["answer_variant"]): item["forced_correct"]
        for item in constraints.get("forced_variants", [])
        if int(item.get("id", -1)) == 638
    }
    v206 = root / "sub_top123_candidate_v206_semantic_batch11.zip"
    v207 = root / "sub_top123_candidate_v207_semantic_batch6_final.zip"
    checks = {
        "v276_finished": current.get("status") == "Finished",
        "v276_selected_authenticated": current_selected.get("on_leaderboard") is True,
        "release_gate_pass": release.get("status") == "PASS",
        "weakly_dominates_v269": weakly_dominates,
        "strict_table_improvement_only": set(strictly_improves)
        == {"TABLES_F2MACRO", "TABLES_PRECISION", "TABLES_RECALL"},
        "q638_old_forced_public_wrong": q638.get("num:-56.62") is False,
        "q638_source_truth_forced_public_wrong": q638.get("num:5.74") is False,
        "v206_preserved": sha256(v206)
        == "34B19414AA1610785163513F8D2F16CCD41A1F797A90F7121817657F553E4B5B",
        "v207_preserved": sha256(v207)
        == "DD616AA408E4B601B921246DDFF4DA4BFD4E8C283FCEF22FF68680AAAB2ADDAF",
    }
    return {
        "schema_version": 1,
        "kind": "v276_public_dominance_and_final_preference",
        "inputs": {
            "full_scores": {"path": str(full_scores), "sha256": sha256(full_scores)},
            "selected_scores": {"path": str(selected_scores), "sha256": sha256(selected_scores)},
            "release_report": {"path": str(release_report), "sha256": sha256(release_report)},
            "constraints_report": {"path": str(constraints_report), "sha256": sha256(constraints_report)},
        },
        "v276": current,
        "v269_fallback": fallback,
        "metric_deltas_v276_minus_v269": deltas,
        "strictly_improved_metrics": strictly_improves,
        "q638_forced_variants": q638,
        "checks": checks,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "decision": {
            "public_selected": "v276",
            "private_final_preference": "v276",
            "rollback": "v269",
            "claim_limit": "No private score is known; preference uses measured public dominance and physical-source robustness.",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full-scores",
        type=Path,
        default=ROOT / "build" / "leaderboard_scores_authoritative_full_after_v276.json",
    )
    parser.add_argument(
        "--selected-scores",
        type=Path,
        default=ROOT / "build" / "leaderboard_after_v276_selected.json",
    )
    parser.add_argument(
        "--release-report",
        type=Path,
        default=ROOT / "build" / "release_gate" / "sub_v276_q638_fix" / "release.json",
    )
    parser.add_argument(
        "--constraints-report",
        type=Path,
        default=ROOT / "build" / "v277_leaderboard_answer_constraints_after_v276.json",
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "build" / "v278_v276_public_dominance.json"
    )
    args = parser.parse_args()
    report = audit(
        args.full_scores.resolve(),
        args.selected_scores.resolve(),
        args.release_report.resolve(),
        args.constraints_report.resolve(),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"v276", "v269_fallback"}}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

