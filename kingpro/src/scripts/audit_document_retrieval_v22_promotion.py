"""Fail-closed promotion gate for source-bound retrieval policy v22.

The gate compares v22 with promoted v21 on the same 1,012 executable source
bindings and requires the stratified local proxy to remain clean.  These are
not BTC hidden-gold metrics and do not predict a private leaderboard score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREVIOUS = (
    ROOT / "build" / "demo_compliance" / "document_retrieval_coverage_semantic_v21_final.json"
)
DEFAULT_CANDIDATE = (
    ROOT / "build" / "demo_compliance" / "document_retrieval_coverage_semantic_v22_final.json"
)
DEFAULT_PROXY = (
    ROOT / "build" / "demo_compliance" / "private_proxy_cohorts_hanoi_v217_v5_final.json"
)


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _mode(report: dict) -> dict:
    value = (report.get("modes") or {}).get("1")
    if not isinstance(value, dict):
        raise ValueError("retrieval report has no per_pair=1 mode")
    return value


def audit(
    previous_path: Path = DEFAULT_PREVIOUS,
    candidate_path: Path = DEFAULT_CANDIDATE,
    proxy_path: Path = DEFAULT_PROXY,
) -> dict:
    previous = _read(previous_path)
    candidate = _read(candidate_path)
    proxy = _read(proxy_path)
    before = _mode(previous)
    after = _mode(candidate)
    precision_delta = float(after["macro_precision"]) - float(before["macro_precision"])
    recall_delta = float(after["macro_recall"]) - float(before["macro_recall"])
    f2_delta = float(after["macro_f2"]) - float(before["macro_f2"])
    multi = proxy["cohorts"]["entity_folds"]["multi_entity_or_unresolved"]
    checks = {
        "source_bound_reports": (
            previous.get("kind")
            == candidate.get("kind")
            == "source_proven_document_retrieval_regression_not_btc_gold"
        ),
        "same_candidate_and_question_universe": (
            previous.get("candidate") == candidate.get("candidate")
            and int(before.get("questions", 0)) == int(after.get("questions", -1)) == 1012
        ),
        "bounded_base_pool_and_caps": (
            candidate.get("pool") == 1000
            and candidate.get("cap") == 40
            and candidate.get("universe_cap") == 200
            and float(after["average_predicted_documents"]) <= 3.3
        ),
        "all_v21_rules_retained": all(
            candidate.get(name)
            for name in (
                "facet_backfill",
                "scope_neutral_backfill",
                "opening_year_backfill",
                "formula_year_backfill",
                "growth_scan_year_backfill",
                "comparative_selection_year_backfill",
                "comparative_series_year_backfill",
            )
        ),
        "v22_bounded_rules_enabled": (
            candidate.get("ambiguous_series_reports") is True
            and float(candidate.get("implicit_ownership_scope_minimum_ratio", 0.0)) == 1.2
            and candidate.get("count_scope_fallback") is True
            and candidate.get("catalog_universe_scan") is True
        ),
        "rejected_broad_rules_disabled": (
            float(candidate.get("implicit_scope_minimum_ratio", 0.0)) == 0.0
            and candidate.get("sparse_series_year_backfill") is False
            and candidate.get("common_evidence_year") is False
        ),
        "recall_gain_at_least_0_002": recall_delta >= 0.002,
        "f2_gain_at_least_0_0015": f2_delta >= 0.0015,
        "precision_floor_0_975": float(after["macro_precision"]) >= 0.975,
        "precision_drop_bounded_to_0_001": precision_delta >= -0.001,
        "missed_questions_reduced_to_at_most_4": (
            int(after["missed_questions"]) <= 4
            and int(after["missed_questions"]) < int(before["missed_questions"])
        ),
        "private_proxy_passes_without_observations": bool(
            proxy.get("passed")
            and not proxy.get("risk_flags")
            and not proxy.get("retrieval_observations")
        ),
        "compiler_precision_remains_one": float(
            proxy["global"]["compiler_precision"]
        ) == 1.0,
        "multi_entity_raw_recall_at_least_0_997": float(
            multi["document_macro_recall"]
        ) >= 0.997,
    }
    return {
        "kind": "source_bound_retrieval_v22_promotion_not_btc_hidden_gold",
        "passed": all(checks.values()),
        "checks": checks,
        "inputs": {
            "previous": str(previous_path.resolve()),
            "previous_sha256": _sha256(previous_path),
            "candidate": str(candidate_path.resolve()),
            "candidate_sha256": _sha256(candidate_path),
            "private_proxy": str(proxy_path.resolve()),
            "private_proxy_sha256": _sha256(proxy_path),
        },
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
            "missed_questions_before": int(before["missed_questions"]),
            "missed_questions_after": int(after["missed_questions"]),
            "average_predicted_documents_after": float(
                after["average_predicted_documents"]
            ),
            "multi_entity_raw_recall": float(multi["document_macro_recall"]),
        },
        "claim_limit": (
            "Targets are executable source bindings, not BTC hidden gold. "
            "This gate supports product retrieval promotion only and does not predict private score."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, default=DEFAULT_PREVIOUS)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--private-proxy", type=Path, default=DEFAULT_PROXY)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT / "build" / "demo_compliance" / "document_retrieval_v22_promotion.json"
        ),
    )
    args = parser.parse_args()
    report = audit(
        args.previous.resolve(), args.candidate.resolve(), args.private_proxy.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
