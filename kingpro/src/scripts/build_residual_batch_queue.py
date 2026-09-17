"""Build a residual QA queue without confusing "reviewed" with "proved correct".

The older forensics queue removed every question mentioned by the durable audit
ledger.  That is useful for avoiding duplicate manual work, but it becomes
counterproductive once broad audit packets cover all 1,012 rows: a weak or
bulk review is not evidence that an answer is correct.  This tool keeps those
reviews as weighted evidence, fuses fresh detector families, independent model
counterfactuals and measured submission history, and always emits an explicit
review queue.  It never mutates or uploads a submission.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from infer_public_ensemble import (  # noqa: E402
    cluster_values,
    load_history,
    numeric_answer,
    read_submission,
)


FAMILY_WEIGHTS = {
    "unit_dimension_contract": 10.0,
    "runtime_denominators": 9.0,
    "missing_operand_counterfactuals": 9.0,
    "operand_table_context": 8.0,
    "legacy_operand_table_context": 8.0,
    "source_cell_semantics": 7.0,
    "filter_cardinality": 7.0,
    "merged_header_value_shift": 7.0,
    "legacy_merged_header_value_shift": 7.0,
    "exact_value_semantic_alternatives": 6.0,
    "semantic_child_lineage": 6.0,
    "legacy_semantic_child_lineage": 6.0,
    "selector_metric_coverage_strict": 6.0,
    "physical_report_scope": 6.0,
    "cross_report_comparatives": 5.0,
    "comparison_sign_policy": 5.0,
    "linguistic_premise_signs": 5.0,
    "direct_row_sibling_semantics": 5.0,
    "result_shape_null_additivity": 5.0,
    "row_total_semantics": 4.0,
    "total_bucket_semantics": 4.0,
    "selected_cell_collisions": 4.0,
    "positional_semantics": 3.0,
    "same_label_value_collisions": 2.0,
}

ARRAY_KEYS = (
    "findings",
    "records",
    "hints",
    "rows",
    "disagreements",
    "all_disagreements",
    "unreviewed",
    "high_priority",
)

DETAIL_KEYS = (
    "priority",
    "confidence",
    "review_priority",
    "reason",
    "reasons",
    "metric_key",
    "pattern",
    "current_table",
    "source_table",
    "alternative_table",
    "current_label",
    "selected_label",
    "candidate_label",
    "current_raw",
    "selected_raw",
    "candidate_raw",
    "header_path",
    "coverage_ratio",
    "missing_groups",
    "answer_changes",
    "counterfactual_result",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def detector_family(path: Path) -> str:
    stem = path.stem
    if stem.startswith("v226_"):
        stem = stem[len("v226_") :]
    if stem.endswith("_v225"):
        stem = stem[: -len("_v225")]
    return stem


def compact_detail(row: dict[str, Any]) -> dict[str, Any]:
    return {key: row[key] for key in DETAIL_KEYS if key in row}


def extract_rows(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    for key in ARRAY_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict) and "id" in row]
    return []


def review_index(path: Path) -> dict[int, list[dict[str, Any]]]:
    indexed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in load_jsonl(path):
        if event.get("kind") != "review":
            continue
        if event.get("verdict") in {"strategy_checkpoint"}:
            continue
        qids = event.get("question_ids") or []
        if not isinstance(qids, list):
            continue
        trust = str(event.get("oracle_trust") or "khong-ro")
        evidence = [str(item) for item in (event.get("evidence") or [])]
        narrow = len(qids) <= 20
        source_like = any(
            token in " ".join(evidence).casefold()
            for token in ("source", "financial_statements", "q", "csv", "table")
        )
        tier = "strong" if trust == "that" and narrow and source_like else "weak"
        record = {
            "review_id": event.get("id"),
            "verdict": event.get("verdict"),
            "oracle_trust": trust,
            "tier": tier,
            "question_count": len(qids),
            "summary": event.get("summary"),
            "evidence": evidence[:5],
        }
        for qid in qids:
            indexed[int(qid)].append(record)
    return indexed


def history_alternatives(
    history_path: Path,
    current: dict[int, dict[str, Any]],
    tolerance: float,
) -> dict[int, list[dict[str, Any]]]:
    submissions = [
        loaded
        for row in load_history(history_path)
        if (loaded := read_submission(ROOT, row)) is not None
        and len(loaded.entries) == 1012
    ]
    alternatives: dict[int, list[dict[str, Any]]] = {}
    for qid, base in current.items():
        baseline = numeric_answer(base)
        if baseline is None:
            continue
        observed: list[tuple[float, Any]] = []
        for submission in submissions:
            value = numeric_answer(submission.entries.get(qid, {}))
            if value is not None:
                observed.append((value, submission))
        centers, mapping = cluster_values([value for value, _ in observed], tolerance)
        rows = []
        for class_id, center in enumerate(centers):
            if math.isclose(center, baseline, rel_tol=0.0, abs_tol=tolerance):
                continue
            members = [sub for value, sub in observed if mapping[value] == class_id]
            if not members:
                continue
            best = max(
                members,
                key=lambda sub: (
                    sub.history.answer_accuracy,
                    sub.history.execution_accuracy,
                    sub.history.submission_id,
                ),
            )
            rows.append(
                {
                    "answer": center,
                    "best_source": best.history.stem,
                    "best_answer_accuracy": best.history.answer_accuracy,
                    "supporting_submission_count": len(members),
                }
            )
        rows.sort(
            key=lambda row: (
                -row["best_answer_accuracy"],
                -row["supporting_submission_count"],
                row["answer"],
            )
        )
        if rows:
            alternatives[qid] = rows
    return alternatives


def plausible_alternative(current: Any, alternative: Any) -> bool:
    """Reject the unit-scale garbage common in early LLM submissions."""
    try:
        baseline = float(current)
        candidate = float(alternative)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        return False
    if baseline == 0.0 or candidate == 0.0:
        return False
    ratio = abs(candidate / baseline)
    return 0.05 <= ratio <= 20.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3",
    )
    parser.add_argument("--audit-glob", default="build/v226_*_v225.json")
    parser.add_argument(
        "--counterfactuals",
        type=Path,
        default=ROOT / "build/v227_counterfactual_batches_v217_all.json",
    )
    parser.add_argument(
        "--history", type=Path, default=ROOT / "configs/leaderboard_history.csv"
    )
    parser.add_argument(
        "--reviews",
        type=Path,
        default=ROOT / "knowledge/vothuong/experiments.jsonl",
    )
    parser.add_argument(
        "--ensemble",
        type=Path,
        default=ROOT / "build/v227_public_ensemble_v217_updated.json",
    )
    parser.add_argument(
        "--panel-audit",
        type=Path,
        default=ROOT / "build/v227_panel_disagreement_audit.json",
    )
    parser.add_argument(
        "--ensemble-verdicts",
        type=Path,
        default=ROOT / "knowledge/vothuong/ensemble_candidate_verdicts.json",
    )
    parser.add_argument("--limit", type=int, default=150)
    parser.add_argument(
        "--zone", choices=("all", "first506", "second506"), default="all"
    )
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "build/v227_residual_batch_queue.json"
    )
    args = parser.parse_args()
    if args.limit < 50:
        raise SystemExit("--limit must be at least 50 for batch-first review")

    candidate = args.candidate.resolve()
    submission_rows = load_json(candidate / "submission.json")
    by_id = {int(row["id"]): row for row in submission_rows}
    if len(by_id) != 1012:
        raise SystemExit(f"candidate must contain 1,012 unique IDs, got {len(by_id)}")

    signals: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    report_stats = {}
    for path in sorted(ROOT.glob(args.audit_glob)):
        family = detector_family(path)
        if family not in FAMILY_WEIGHTS:
            continue
        rows = extract_rows(load_json(path))
        report_stats[family] = {
            "path": relative(path),
            "findings": len(rows),
            "questions": len({int(row["id"]) for row in rows}),
        }
        for row in rows:
            signals[int(row["id"])][family].append(row)

    counter_by_id = {
        int(row["id"]): row
        for row in load_json(args.counterfactuals).get("all_disagreements", [])
    }
    ensemble_by_id: dict[int, dict[str, Any]] = {}
    if args.ensemble.exists():
        ensemble_by_id = {
            int(row["id"]): row
            for row in load_json(args.ensemble).get("question_predictions", [])
        }
    panel_keep_current: dict[int, dict[str, Any]] = {}
    if args.panel_audit.exists():
        panel_keep_current = {
            int(row["id"]): row
            for row in load_json(args.panel_audit).get("disagreements", [])
            if row.get("status") == "keep_current"
        }
    ensemble_keep_current: dict[int, dict[str, Any]] = {}
    if args.ensemble_verdicts.exists():
        ensemble_keep_current = {
            int(qid): row
            for qid, row in (
                load_json(args.ensemble_verdicts).get("verdicts") or {}
            ).items()
            if row.get("verdict") == "keep_current"
        }
    reviews = review_index(args.reviews)
    history = history_alternatives(args.history, by_id, args.tolerance)

    queue = []
    for qid, row in by_id.items():
        if args.zone == "first506" and qid > 506:
            continue
        if args.zone == "second506" and qid <= 506:
            continue
        family_rows = signals.get(qid, {})
        counter = counter_by_id.get(qid)
        historical_all = history.get(qid, [])
        historical = [
            item
            for item in historical_all
            if plausible_alternative(row.get("answer"), item.get("answer"))
        ]
        review_rows = reviews.get(qid, [])
        strong_reviews = [item for item in review_rows if item["tier"] == "strong"]
        ensemble = ensemble_by_id.get(qid)
        ensemble_alternative = None
        ensemble_bonus = 0.0
        if ensemble:
            baseline_answer = float(row.get("answer"))
            choices = [
                {
                    "answer": float(item["answer"]),
                    "probability": float(item["probability"]),
                    "sources": item.get("sources") or [],
                }
                for item in ensemble.get("candidates", [])
                if plausible_alternative(baseline_answer, item.get("answer"))
            ]
            choices.sort(key=lambda item: -item["probability"])
            if choices:
                top = choices[0]
                current_probability = max(
                    [
                        float(item["probability"])
                        for item in ensemble.get("candidates", [])
                        if math.isclose(
                            float(item["answer"]),
                            baseline_answer,
                            rel_tol=0.0,
                            abs_tol=args.tolerance,
                        )
                    ]
                    or [0.0]
                )
                runner = max(current_probability, float(ensemble.get("none_probability") or 0.0))
                margin = top["probability"] - runner
                if top["probability"] >= 0.35 and margin >= 0.05:
                    ensemble_alternative = {
                        **top,
                        "current_probability": current_probability,
                        "none_probability": float(ensemble.get("none_probability") or 0.0),
                        "margin_over_current_or_none": margin,
                    }
                    if top["probability"] >= 0.60 and margin >= 0.20:
                        ensemble_bonus = 12.0
                    elif top["probability"] >= 0.45 and margin >= 0.10:
                        ensemble_bonus = 7.0
                    else:
                        ensemble_bonus = 3.0

        family_score = sum(FAMILY_WEIGHTS[name] for name in family_rows)
        independent_bonus = max(0, len(family_rows) - 1) * 2.5
        counter_bonus = 0.0
        counter_plausible = bool(
            counter
            and plausible_alternative(
                row.get("answer"), counter.get("alternative_answer")
            )
        )
        if counter:
            counter_bonus = 4.0 if counter_plausible else -4.0
            if counter_plausible and len(counter.get("dependency_groups") or []) >= 2:
                counter_bonus += 4.0
        history_bonus = min(4.0, float(len(historical)))
        # The scorer diagnostic exposes a 506-row public reference.  The exact
        # membership mapping is not public, so this is a triage prior only.
        first_half_bonus = 4.0 if qid <= 506 else 0.0
        review_penalty = min(18.0, 9.0 * len(strong_reviews))
        score = family_score + independent_bonus + counter_bonus + history_bonus
        panel_penalty = 24.0 if qid in panel_keep_current else 0.0
        ensemble_verdict_penalty = 18.0 if qid in ensemble_keep_current else 0.0
        score += (
            first_half_bonus
            + ensemble_bonus
            - review_penalty
            - panel_penalty
            - ensemble_verdict_penalty
        )

        # Keep the queue complete even when a row has only historical/model
        # disagreement.  Rows with no residual signal are intentionally absent.
        if score <= 0 or (
            not family_rows
            and not counter
            and not historical
            and ensemble_alternative is None
        ):
            continue

        reasons = [
            f"{name}: {len(items)} finding(s)"
            for name, items in sorted(
                family_rows.items(), key=lambda item: (-FAMILY_WEIGHTS[item[0]], item[0])
            )
        ]
        if counter:
            reasons.append(
                "independent-view disagreement: "
                + ",".join(counter.get("dependency_groups") or [])
            )
        if historical:
            reasons.append(f"{len(historical)} historical answer class(es)")
        if ensemble_alternative:
            reasons.append(
                "leaderboard-equation alternative: "
                f"p={ensemble_alternative['probability']:.3f}, "
                f"margin={ensemble_alternative['margin_over_current_or_none']:.3f}"
            )
        if strong_reviews:
            reasons.append(
                f"discounted by {len(strong_reviews)} narrow trusted source review(s)"
            )
        if qid in panel_keep_current:
            reasons.append("discounted by source-adjudicated false panel disagreement")
        if qid in ensemble_keep_current:
            reasons.append("discounted by source-adjudicated false ensemble alternative")

        alternative = None
        if counter and counter_plausible:
            alternative = {
                "answer": counter.get("alternative_answer"),
                "kind": counter.get("discrepancy"),
                "views": counter.get("independent_views") or [],
                "dependency_groups": counter.get("dependency_groups") or [],
                "tables": counter.get("alternative_tables") or [],
            }
        elif ensemble_alternative:
            alternative = {
                **ensemble_alternative,
                "kind": "leaderboard_equation_answer_class",
            }
        elif historical:
            alternative = {**historical[0], "kind": "historical_answer_class"}

        if alternative and len(family_rows) >= 3 and not strong_reviews:
            confidence = "high-review-priority"
        elif alternative and len(family_rows) >= 1:
            confidence = "medium-review-priority"
        else:
            confidence = "low-review-priority"

        queue.append(
            {
                "id": qid,
                "risk_score": round(score, 3),
                "confidence": confidence,
                "zone": "first506" if qid <= 506 else "second506",
                "question": row.get("question"),
                "current_answer": row.get("answer"),
                "reason": reasons,
                "detector_families": sorted(family_rows),
                "detector_evidence": {
                    name: {
                        "finding_count": len(items),
                        "sample": compact_detail(items[0]),
                    }
                    for name, items in sorted(family_rows.items())
                },
                "source": {
                    "relevant_tables": row.get("relevant_tables") or [],
                    "evidence_csv": [
                        item.get("csv_path") for item in (row.get("evidence") or [])
                    ],
                },
                "counterfactual": alternative,
                "ensemble_alternative": ensemble_alternative,
                "historical_alternatives": historical[:5],
                "implausible_historical_alternative_count": len(historical_all) - len(historical),
                "durable_reviews": review_rows[-5:],
                "strong_review_count": len(strong_reviews),
                "mutation_authority": False,
            }
        )

    queue.sort(
        key=lambda item: (
            -item["risk_score"],
            item["strong_review_count"],
            item["id"],
        )
    )
    selected = queue[: args.limit]
    payload = {
        "schema_version": 1,
        "kind": "residual_batch_review_queue",
        "candidate": relative(candidate),
        "policy": {
            "reviewed_is_not_proved_correct": True,
            "minimum_requested_queue": 50,
            "automatic_answer_changes": False,
            "source_verification_required": True,
            "leaderboard_equations_are_priority_not_oracle": True,
            "first506_note": (
                "The scorer exposes gold=506/pred=1012. first506 is a triage "
                "prior, not a proved public-membership label."
            ),
        },
        "counts": {
            "questions": len(by_id),
            "eligible_residual": len(queue),
            "queued": len(selected),
            "first506": sum(item["zone"] == "first506" for item in selected),
            "second506": sum(item["zone"] == "second506" for item in selected),
            "with_counterfactual": sum(item["counterfactual"] is not None for item in selected),
            "with_strong_review": sum(item["strong_review_count"] > 0 for item in selected),
            "with_ensemble_alternative": sum(
                item["ensemble_alternative"] is not None for item in selected
            ),
            "discounted_panel_false_positive": sum(
                item["id"] in panel_keep_current for item in selected
            ),
            "discounted_ensemble_false_positive": sum(
                item["id"] in ensemble_keep_current for item in selected
            ),
        },
        "report_stats": report_stats,
        "queue": selected,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "counts": payload["counts"],
                "top_ids": [item["id"] for item in selected[:30]],
                "out": relative(args.out),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
