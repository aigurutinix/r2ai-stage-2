"""Validate the source-bound document-retrieval facet A/B experiment.

The reference labels are the documents used by the executable v184 programs,
not BTC hidden gold.  This gate therefore protects product retrieval from a
measured local regression; it must never be presented as a leaderboard claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    ROOT / "build" / "demo_compliance" / "document_retrieval_coverage_v1.json"
)
DEFAULT_CHOSEN = (
    ROOT
    / "build"
    / "demo_compliance"
    / "document_retrieval_coverage_semantic_v20_final.json"
)
DEFAULT_PREVIOUS = (
    ROOT
    / "build"
    / "demo_compliance"
    / "document_retrieval_coverage_alias_backfill_v7_final.json"
)
DEFAULT_REJECTED = (
    ROOT
    / "build"
    / "demo_compliance"
    / "document_retrieval_coverage_implicit_years_v4.json"
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {0}".format(path))
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _mode(report: dict) -> dict:
    modes = report.get("modes") or {}
    value = modes.get("1") or modes.get(1)
    if not isinstance(value, dict):
        raise ValueError("report has no per_pair=1 metrics")
    return value


def audit(
    baseline_path: Path = DEFAULT_BASELINE,
    chosen_path: Path = DEFAULT_CHOSEN,
    rejected_path: Path = DEFAULT_REJECTED,
    previous_path: Path = DEFAULT_PREVIOUS,
) -> dict:
    baseline = _read(baseline_path)
    chosen = _read(chosen_path)
    rejected = _read(rejected_path)
    previous = _read(previous_path)
    before = _mode(baseline)
    after = _mode(chosen)
    discarded = _mode(rejected)
    prior = _mode(previous)

    precision_delta = float(after["macro_precision"]) - float(before["macro_precision"])
    recall_delta = float(after["macro_recall"]) - float(before["macro_recall"])
    f2_delta = float(after["macro_f2"]) - float(before["macro_f2"])
    checks = {
        "reports_are_source_bound_regression_not_btc_gold": all(
            report.get("kind")
            == "source_proven_document_retrieval_regression_not_btc_gold"
            for report in (baseline, chosen, rejected, previous)
        ),
        "same_candidate_cap_and_full_question_set": bool(
            baseline.get("candidate") == chosen.get("candidate") == rejected.get("candidate") == previous.get("candidate")
            and baseline.get("cap") == chosen.get("cap") == rejected.get("cap") == previous.get("cap") == 40
            and int(before.get("questions", 0))
            == int(after.get("questions", -1))
            == int(discarded.get("questions", -2))
            == int(prior.get("questions", -3))
            == 1012
        ),
        "chosen_semantic_features_are_enabled": bool(
            chosen.get("facet_backfill")
            and chosen.get("scope_neutral_backfill")
            and chosen.get("opening_year_backfill")
            and chosen.get("formula_year_backfill")
            and chosen.get("growth_scan_year_backfill")
            and chosen.get("comparative_selection_year_backfill")
        ),
        "macro_recall_improved": recall_delta > 0.0,
        "macro_f2_improved": f2_delta > 0.0,
        "macro_precision_improved": precision_delta > 0.0,
        "missed_question_count_reduced": int(after["missed_questions"]) < int(before["missed_questions"]),
        "chosen_beats_rejected_implicit_year_probe_on_f2": (
            float(after["macro_f2"]) > float(discarded["macro_f2"])
        ),
        "chosen_macro_f2_exceeds_0_95": float(after["macro_f2"]) > 0.95,
        "semantic_pipeline_improves_previous_production_report": (
            float(after["macro_f2"]) > float(prior["macro_f2"])
            and float(after["macro_recall"]) > float(prior["macro_recall"])
            and float(after["macro_precision"]) > float(prior["macro_precision"])
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_report": str(baseline_path),
        "baseline_report_sha256": _sha256(baseline_path),
        "chosen_report": str(chosen_path),
        "chosen_report_sha256": _sha256(chosen_path),
        "rejected_probe_report": str(rejected_path),
        "rejected_probe_report_sha256": _sha256(rejected_path),
        "previous_production_report": str(previous_path),
        "previous_production_report_sha256": _sha256(previous_path),
        "metrics": {
            "macro_precision_before": float(before["macro_precision"]),
            "macro_precision_after": float(after["macro_precision"]),
            "macro_precision_delta": round(precision_delta, 6),
            "macro_recall_before": float(before["macro_recall"]),
            "macro_recall_after": float(after["macro_recall"]),
            "macro_recall_delta": round(recall_delta, 6),
            "macro_f2_before": float(before["macro_f2"]),
            "macro_f2_after": float(after["macro_f2"]),
            "macro_f2_delta": round(f2_delta, 6),
            "rejected_probe_macro_f2": float(discarded["macro_f2"]),
            "previous_production_macro_f2": float(prior["macro_f2"]),
            "missed_questions_before": int(before["missed_questions"]),
            "missed_questions_after": int(after["missed_questions"]),
        },
        "claim_limit": (
            "Labels are documents referenced by v184 executable programs, not BTC hidden gold. "
            "This is a local product-regression gate, not a leaderboard F2 or score forecast."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--chosen", type=Path, default=DEFAULT_CHOSEN)
    parser.add_argument("--rejected", type=Path, default=DEFAULT_REJECTED)
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "build" / "demo_compliance" / "document_retrieval_facets_semantic_v2.json"
        ),
    )
    args = parser.parse_args()
    report = audit(
        args.baseline.resolve(),
        args.chosen.resolve(),
        args.rejected.resolve(),
        args.previous.resolve(),
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
