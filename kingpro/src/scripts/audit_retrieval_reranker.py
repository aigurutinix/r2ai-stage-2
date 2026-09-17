"""Validate the source-proven table-reranker A/B report.

The input labels come from executable source bindings rather than BTC hidden
gold.  Consequently this gate proves a product regression improvement only;
it does not estimate leaderboard F2 or hidden score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = ROOT / "build" / "demo_compliance" / "retrieval_provenance_semantic_v6_baseline.json"
DEFAULT_RERANKED = ROOT / "build" / "demo_compliance" / "retrieval_provenance_semantic_v6_reranked.json"


def _read(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected JSON object: {0}".format(path))
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def audit(baseline_path: Path = DEFAULT_BASELINE, reranked_path: Path = DEFAULT_RERANKED) -> dict:
    baseline = _read(baseline_path)
    reranked = _read(reranked_path)
    base_summary = baseline.get("summary") or {}
    new_summary = reranked.get("summary") or {}
    shared_buckets = sorted(set(base_summary).intersection(new_summary))
    bucket_deltas = {}
    regressions = []
    for bucket in shared_buckets:
        before = base_summary[bucket]
        after = new_summary[bucket]
        recall_delta = float(after["pipeline_recall_at_8"]) - float(before["pipeline_recall_at_8"])
        mrr_delta = float(after["pipeline_mrr40"]) - float(before["pipeline_mrr40"])
        bucket_deltas[bucket] = {
            "questions": int(before["questions"]),
            "pipeline_recall_at_8_before": float(before["pipeline_recall_at_8"]),
            "pipeline_recall_at_8_after": float(after["pipeline_recall_at_8"]),
            "pipeline_recall_at_8_delta": round(recall_delta, 6),
            "pipeline_mrr40_before": float(before["pipeline_mrr40"]),
            "pipeline_mrr40_after": float(after["pipeline_mrr40"]),
            "pipeline_mrr40_delta": round(mrr_delta, 6),
        }
        if recall_delta < -1e-12 or mrr_delta < -1e-12:
            regressions.append(bucket)

    same_question_set = bool(
        baseline.get("audited_questions") == reranked.get("audited_questions") == 1012
        and baseline.get("candidate") == reranked.get("candidate")
        and base_summary.get("all", {}).get("questions") == new_summary.get("all", {}).get("questions")
    )
    same_document_recall = bool(
        float(base_summary.get("all", {}).get("document_recall", -1.0))
        == float(new_summary.get("all", {}).get("document_recall", -2.0))
    )
    all_delta = bucket_deltas.get("all", {})
    checks = {
        "reports_are_source_proven_regression_not_btc_gold": bool(
            baseline.get("kind") == "source_proven_retrieval_regression_not_btc_gold"
            and reranked.get("kind") == "source_proven_retrieval_regression_not_btc_gold"
        ),
        "same_full_question_set": same_question_set,
        "document_stage_unchanged": same_document_recall,
        "document_stage_source_recall_exceeds_0_96": float(
            new_summary.get("all", {}).get("document_recall", 0.0)
        ) > 0.96,
        "configured_label_weight_is_conservative_four": float(reranked.get("label_weight", 0.0)) == 4.0,
        "pipeline_recall_at_8_improved": float(all_delta.get("pipeline_recall_at_8_delta", 0.0)) > 0.0,
        "pipeline_mrr40_improved": float(all_delta.get("pipeline_mrr40_delta", 0.0)) > 0.0,
        "no_bucket_regressed_on_recall_or_mrr": bool(shared_buckets and not regressions),
    }
    implementation = ROOT / "src" / "kingpro" / "retrieval" / "table_reranker.py"
    candidate_submission = (
        ROOT / str(reranked.get("candidate", "")) / "submission.json"
    )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "baseline_report": str(baseline_path),
        "baseline_report_sha256": _sha256(baseline_path),
        "reranked_report": str(reranked_path),
        "reranked_report_sha256": _sha256(reranked_path),
        "implementation_sha256": _sha256(implementation),
        "candidate_submission_sha256": _sha256(candidate_submission),
        "bucket_deltas": bucket_deltas,
        "regressions": regressions,
        "claim_limit": (
            "This proves recovery of source-audited executable table bindings, "
            "not BTC hidden-gold F2 or expected leaderboard gain."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--reranked", type=Path, default=DEFAULT_RERANKED)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "demo_compliance" / "table_reranker_regression_semantic_v6.json",
    )
    args = parser.parse_args()
    report = audit(args.baseline.resolve(), args.reranked.resolve())
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
