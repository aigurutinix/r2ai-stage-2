"""Fuse independent read-only audits into a reproducible batch review queue.

The individual audits intentionally favour recall and therefore contain many
false positives.  This script does not change a submission.  It ranks question
IDs only when at least two independent audit families agree, discounts IDs
that already have a durable source review, and emits a fixed-size queue whose
evidence can be inspected family by family.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REPORTS = {
    "operand_context": "v210_operand_table_context_consistency_v209.json",
    "unit_dimension": "v210_batch_unit_dimension.json",
    "program_intent": "v210_batch_program_intent_all_v209.json",
    "filter_cardinality": "v210_batch_filter_cardinality.json",
    "comparison_sign": "v210_batch_comparison_sign.json",
    "linguistic_sign": "v210_batch_linguistic_signs.json",
    "selector_coverage": "v210_batch_selector_coverage.json",
    "shape_additivity": "v210_batch_shape_additivity.json",
    "positional_semantics": "v210_batch_positional.json",
    "semantic_alignment": "v210_batch_semantic_alignment.json",
    "cross_document_units": "v210_batch_cross_document_units.json",
}

WEIGHTS = {
    "operand_context": 8.0,
    "unit_dimension": 8.0,
    "program_intent": 5.0,
    "filter_cardinality": 5.0,
    "comparison_sign": 4.0,
    "linguistic_sign": 3.0,
    "selector_coverage": 1.0,
    "shape_additivity": 2.0,
    "positional_semantics": 2.0,
    "semantic_alignment": 3.0,
    "cross_document_units": 1.0,
}

SOURCE_STRONG_VERDICTS = {
    "source_confirmed",
    "hypothesis_rejected",
    "retained",
    "confirmed_correct",
}


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_reviews(path: Path) -> dict[int, list[dict]]:
    reviews: dict[int, list[dict]] = defaultdict(list)
    if not path.exists():
        return reviews
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("kind") != "review":
            continue
        for qid in row.get("question_ids", []):
            reviews[int(qid)].append(
                {
                    "review_id": row.get("id"),
                    "verdict": row.get("verdict"),
                    "summary": row.get("summary"),
                }
            )
    return reviews


def normalized_strength(family: str, finding: dict) -> float:
    if family == "semantic_alignment":
        # The audit reports similarity; low similarity is the risky direction.
        return max(0.0, min(1.0, 1.0 - float(finding.get("score", 1.0))))
    if family == "selector_coverage":
        ratio = float(finding.get("coverage_ratio", 1.0))
        return max(0.25, min(1.0, 1.0 - ratio))
    if family in {"program_intent", "filter_cardinality", "shape_additivity"}:
        raw = float(finding.get("score", finding.get("risk", 1.0)))
        return max(0.25, min(1.0, raw / 6.0))
    if family == "positional_semantics":
        return max(0.25, min(1.0, 1.0 - float(finding.get("score", 0.0))))
    return 1.0


def compact_detail(family: str, finding: dict) -> dict:
    keep = {
        "score",
        "risk",
        "reasons",
        "metric_key",
        "context_group",
        "families",
        "pattern",
        "match_count",
        "axis",
        "cohort",
        "present_groups",
        "missing_groups",
        "coverage_ratio",
        "label",
        "header_path",
        "raw",
    }
    return {key: value for key, value in finding.items() if key in keep}


def build_queue(
    submission_dir: Path,
    report_dir: Path,
    ledger: Path,
    limit: int,
) -> dict:
    submission = read_json(submission_dir / "submission.json")
    questions = {int(row["id"]): row for row in submission}
    reviews = load_reviews(ledger)
    signals: dict[int, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))

    report_stats = {}
    for family, filename in REPORTS.items():
        path = report_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        payload = read_json(path)
        findings = payload.get("findings", [])
        report_stats[family] = {
            "path": display_path(path),
            "finding_count": len(findings),
            "unique_question_count": len({int(row["id"]) for row in findings}),
        }
        for finding in findings:
            signals[int(finding["id"])][family].append(finding)

    candidates = []
    for qid, by_family in signals.items():
        # Correlated high-recall audits do not earn extra votes from duplicate
        # findings inside the same family.
        family_scores = {}
        evidence = {}
        for family, findings in by_family.items():
            strongest = max(findings, key=lambda row: normalized_strength(family, row))
            strength = normalized_strength(family, strongest)
            family_scores[family] = round(WEIGHTS[family] * strength, 4)
            evidence[family] = {
                "finding_count": len(findings),
                "strength": round(strength, 4),
                "detail": compact_detail(family, strongest),
            }

        independent_family_count = len(family_scores)
        if independent_family_count < 2:
            continue
        durable = reviews.get(qid, [])
        strong_reviews = [
            row for row in durable if str(row.get("verdict")) in SOURCE_STRONG_VERDICTS
        ]
        review_penalty = 8.0 if strong_reviews else 2.0 if durable else 0.0
        score = sum(family_scores.values()) + 1.5 * (independent_family_count - 1)
        score -= review_penalty
        # q769 is the one source-proven repair discovered by the context audit;
        # the bonus keeps it visible without treating audit guesses as fixes.
        if qid == 769:
            score += 20.0
        row = questions[qid]
        candidates.append(
            {
                "id": qid,
                "risk_score": round(score, 4),
                "independent_family_count": independent_family_count,
                "families": sorted(family_scores),
                "family_scores": family_scores,
                "question": row.get("question"),
                "current_answer": row.get("answer"),
                "relevant_tables": row.get("relevant_tables", []),
                "durable_reviews": durable,
                "strong_source_review_count": len(strong_reviews),
                "evidence": evidence,
                "status": "source_proven_repair" if qid == 769 else "review_required",
            }
        )

    candidates.sort(
        key=lambda row: (
            row["status"] != "source_proven_repair",
            -row["risk_score"],
            -row["independent_family_count"],
            row["id"],
        )
    )
    queue = candidates[:limit]
    return {
        "kind": "independent_multi_audit_batch_queue",
        "submission": str(submission_dir),
        "requested_size": limit,
        "queue_size": len(queue),
        "eligible_multi_signal_questions": len(candidates),
        "source_proven_repair_count": sum(
            row["status"] == "source_proven_repair" for row in queue
        ),
        "review_required_count": sum(row["status"] == "review_required" for row in queue),
        "family_counts_in_queue": dict(
            sorted(Counter(family for row in queue for family in row["families"]).items())
        ),
        "report_stats": report_stats,
        "ranking_policy": {
            "minimum_independent_families": 2,
            "duplicate_findings_within_family": "count once; retain strongest",
            "durable_source_review_penalty": 8.0,
            "other_review_penalty": 2.0,
            "mutation_authority": "none; queue is read-only",
        },
        "queue": queue,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "submission_dir",
        nargs="?",
        type=Path,
        default=ROOT / "sub_top123_candidate_v209_q15_board_role_r2",
    )
    parser.add_argument("--report-dir", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "knowledge" / "vothuong" / "experiments.jsonl",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build" / "v210_batch100_independent_risk_queue_v209.json",
    )
    args = parser.parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")
    payload = build_queue(
        args.submission_dir.resolve(),
        args.report_dir.resolve(),
        args.ledger.resolve(),
        args.limit,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "queue"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
