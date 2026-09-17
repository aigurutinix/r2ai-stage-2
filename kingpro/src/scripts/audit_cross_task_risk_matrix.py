"""Rank submission programs by failure patterns seen in adjacent benchmarks.

This is a read-only triage tool, not a correctness oracle.  It combines the
physical source-cell report with static program features so manual review can
start with questions that have the largest cross-table, hierarchy, missingness,
aggregation and output-selection surface.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUBMISSION = ROOT / "sub_top123_candidate_v195_aggregate_contributor_order" / "submission.json"
DEFAULT_SEMANTICS = ROOT / "build" / "v196_source_cell_semantics.json"
DEFAULT_OUTPUT = ROOT / "build" / "v196_cross_task_risk_matrix.json"
DEFAULT_TAXONOMY = ROOT / "knowledge" / "financial_qa_failure_taxonomy.json"
DEFAULT_MEMORY = ROOT / "knowledge" / "vothuong" / "experiments.jsonl"

REASON_TO_FAILURE_MODES = {
    "many_operands_16_plus": ("cross_table_join_and_dependency_path",),
    "many_operands_8_plus": ("cross_table_join_and_dependency_path",),
    "many_operands_4_plus": ("cross_table_join_and_dependency_path",),
    "cross_table_6_plus": ("long_context_rot_and_evidence_drift",),
    "cross_table": ("cross_table_join_and_dependency_path",),
    "cross_document": ("long_context_rot_and_evidence_drift",),
    "multi_year_3_plus": ("period_and_year_alignment",),
    "multi_entity_3_plus": ("entity_and_scope_alignment",),
    "deep_or_noisy_header": ("hierarchical_headers_and_merged_cells",),
    "blank_dash_missingness": ("null_empty_and_missingness_policy",),
    "parenthesized_sign": ("sign_gross_net_absolute",),
    "join_cardinality": (
        "cross_table_join_and_dependency_path",
        "additivity_fanout_and_duplicate_rows",
    ),
    "argmax_argmin_output": ("ranking_comparison_ties",),
    "aggregate_denominator_or_additivity": (
        "aggregate_count_mean_missing_values",
        "operator_and_denominator_semantics",
    ),
    "sign_normalization_semantics": ("sign_gross_net_absolute",),
    "subtotal_component_overlap": (
        "hard_negative_metric_or_table",
        "additivity_fanout_and_duplicate_rows",
    ),
    "long_program": ("spurious_program_and_accidental_match",),
    "identity_output_contract": ("result_shape_null_and_execution_equivalence",),
}

RISK_BONUS = {"critical": 3, "high": 2, "medium": 1}

# Terminal outcomes that mean a source-backed review has already closed the
# question.  Limiting this to the literal ``source_confirmed`` label caused
# false-positive and rejected-hypothesis reviews to be audited again.
SOURCE_VERIFIED_VERDICTS = {
    "source_confirmed",
    "source_confirmed_fix",
    "source_confirmed_no_change",
    "source_corrected",
    "false_positive",
    "hypothesis_rejected",
    "keep",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, default=DEFAULT_SUBMISSION)
    parser.add_argument("--semantics", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--taxonomy", type=Path, default=DEFAULT_TAXONOMY)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--top", type=int, default=100)
    return parser.parse_args()


def add(reasons: list[str], name: str, points: int) -> int:
    reasons.append(name)
    return points


def load_review_memory(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Index append-only source reviews by question id."""

    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    if not path.is_file():
        return result
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("kind") != "review":
                continue
            for raw_id in event.get("question_ids", []):
                try:
                    result[int(raw_id)].append(event)
                except (TypeError, ValueError):
                    continue
    return result


def failure_modes_for_reasons(reasons: list[str]) -> list[str]:
    modes = {
        mode
        for reason in reasons
        for mode in REASON_TO_FAILURE_MODES.get(reason, ())
    }
    return sorted(modes)


def compact_review(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event.get("id"),
        "timestamp": event.get("timestamp"),
        "verdict": event.get("verdict"),
        "oracle": event.get("oracle"),
        "oracle_trust": event.get("oracle_trust"),
        "summary": event.get("summary"),
    }


def is_source_verified_review(review: dict[str, Any] | None) -> bool:
    """Return whether the latest review closed the question against source.

    Candidate fixes and semantic ambiguities deliberately remain reviewable.
    Historical logs use several source-oracle spellings, so the oracle field is
    treated as source-backed when it contains the word ``source``.  Newer logs
    additionally carry the trusted marker ``oracle_trust=that``.
    """

    if not review:
        return False
    verdict = str(review.get("verdict") or "").casefold().replace("-", "_")
    oracle = str(review.get("oracle") or "").casefold()
    source_backed = "source" in oracle or review.get("oracle_trust") == "that"
    return source_backed and verdict in SOURCE_VERIFIED_VERDICTS


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    rows = json.loads(args.submission.read_text(encoding="utf-8"))
    semantic_payload = json.loads(args.semantics.read_text(encoding="utf-8"))
    semantic_by_id = {int(item["id"]): item for item in semantic_payload.get("records", [])}
    taxonomy = json.loads(args.taxonomy.read_text(encoding="utf-8"))
    modes_by_id = {
        str(item["id"]): item for item in taxonomy.get("failure_modes", [])
    }
    known_modes_by_question: dict[int, set[str]] = defaultdict(set)
    for mode_id, mode in modes_by_id.items():
        for raw_id in mode.get("known_cases", []):
            try:
                known_modes_by_question[int(raw_id)].add(mode_id)
            except (TypeError, ValueError):
                continue
    review_memory = load_review_memory(args.memory)

    ranked: list[dict[str, object]] = []
    reason_counts: Counter[str] = Counter()
    for row in rows:
        qid = int(row["id"])
        query = str(row.get("pandas_query", ""))
        question = str(row.get("question", ""))
        cells = semantic_by_id.get(qid, {}).get("cells", [])
        tables = {str(cell.get("source_table", "")) for cell in cells if cell.get("source_table")}
        docs = {table.rsplit("|", 1)[0] for table in tables}
        years = {str(cell.get("year", "")) for cell in cells if cell.get("year")}
        tickers = {str(cell.get("ticker", "")) for cell in cells if cell.get("ticker")}
        raw_values = [str(cell.get("raw_physical", "")).strip() for cell in cells]
        labels = [str(cell.get("source_label", "")).casefold() for cell in cells]
        header_depth = max((len(cell.get("header_path", [])) for cell in cells), default=0)

        score = 0
        reasons: list[str] = []
        if len(cells) >= 16:
            score += add(reasons, "many_operands_16_plus", 4)
        elif len(cells) >= 8:
            score += add(reasons, "many_operands_8_plus", 3)
        elif len(cells) >= 4:
            score += add(reasons, "many_operands_4_plus", 1)
        if len(tables) >= 6:
            score += add(reasons, "cross_table_6_plus", 4)
        elif len(tables) >= 2:
            score += add(reasons, "cross_table", 2)
        if len(docs) >= 2:
            score += add(reasons, "cross_document", 2)
        if len(years) >= 3:
            score += add(reasons, "multi_year_3_plus", 2)
        if len(tickers) >= 3:
            score += add(reasons, "multi_entity_3_plus", 2)
        if header_depth >= 3:
            score += add(reasons, "deep_or_noisy_header", 2)
        if any(value in {"", "-", "–", "—"} for value in raw_values):
            score += add(reasons, "blank_dash_missingness", 3)
        if any(value.startswith("(") and value.endswith(")") for value in raw_values):
            score += add(reasons, "parenthesized_sign", 1)
        if ".merge(" in query or "pd.merge(" in query or ".join(" in query:
            score += add(reasons, "join_cardinality", 4)
        if re.search(r"\b(max|min)\s*\(", query) or ".idxmax(" in query or ".idxmin(" in query:
            score += add(reasons, "argmax_argmin_output", 2)
        if re.search(r"\.mean\s*\(|\bsum\s*\(", query):
            score += add(reasons, "aggregate_denominator_or_additivity", 2)
        if "abs(" in query and any(value.startswith("(") for value in raw_values):
            score += add(reasons, "sign_normalization_semantics", 1)
        if any("tổng" in label for label in labels) and len(cells) >= 3:
            score += add(reasons, "subtotal_component_overlap", 3)
        if query.count("\n") >= 80:
            score += add(reasons, "long_program", 2)
        if re.search(r"\b(năm nào|công ty nào|mã nào|doanh nghiệp nào)\b", question.casefold()):
            score += add(reasons, "identity_output_contract", 2)

        inferred_modes = set(failure_modes_for_reasons(reasons))
        known_modes = known_modes_by_question.get(qid, set())
        matched_modes = sorted(inferred_modes | known_modes)
        memory_score = score + sum(
            RISK_BONUS.get(str(modes_by_id.get(mode, {}).get("risk", "")), 0)
            for mode in matched_modes
        )
        if known_modes:
            memory_score += 2
        reviews = review_memory.get(qid, [])
        latest_review = compact_review(reviews[-1]) if reviews else None
        source_confirmed = is_source_verified_review(latest_review)

        similar_cases: list[dict[str, Any]] = []
        seen_similar: set[int] = set()
        for mode_id in matched_modes:
            for raw_id in modes_by_id.get(mode_id, {}).get("known_cases", []):
                similar_id = int(raw_id)
                if similar_id == qid or similar_id in seen_similar:
                    continue
                seen_similar.add(similar_id)
                prior_reviews = review_memory.get(similar_id, [])
                similar_cases.append({
                    "id": similar_id,
                    "shared_failure_mode": mode_id,
                    "latest_review": (
                        compact_review(prior_reviews[-1]) if prior_reviews else None
                    ),
                })
                if len(similar_cases) >= 8:
                    break
            if len(similar_cases) >= 8:
                break

        for reason in reasons:
            reason_counts[reason] += 1
        ranked.append(
            {
                "id": qid,
                "risk_score": score,
                "memory_score": memory_score,
                "question": question,
                "answer": row.get("answer"),
                "reasons": reasons,
                "matched_failure_modes": matched_modes,
                "known_failure_modes": sorted(known_modes),
                "latest_review": latest_review,
                "source_confirmed": source_confirmed,
                "similar_cases": similar_cases,
                "features": {
                    "operands": len(cells),
                    "tables": len(tables),
                    "documents": len(docs),
                    "years": len(years),
                    "tickers": len(tickers),
                    "max_header_depth": header_depth,
                },
            }
        )

    ranked.sort(key=lambda item: (-int(item["memory_score"]), int(item["id"])))
    unreviewed = [item for item in ranked if not item["source_confirmed"]]
    payload = {
        "submission": str(args.submission),
        "semantics": str(args.semantics),
        "taxonomy": str(args.taxonomy),
        "memory": str(args.memory),
        "checked": len(rows),
        "questions_with_physical_cells": len(semantic_by_id),
        "reviewed_question_count": len(review_memory),
        "source_confirmed_count": sum(bool(item["source_confirmed"]) for item in ranked),
        "policy": "Error-memory triage only. A high score or similar case is not evidence of an incorrect answer; every repair must return to BTC source cells.",
        "risk_reason_counts": dict(sorted(reason_counts.items())),
        "top": ranked[: max(args.top, 0)],
        "unreviewed_top": unreviewed[: max(args.top, 0)],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: payload[key] for key in ("checked", "questions_with_physical_cells", "risk_reason_counts")}, ensure_ascii=False, indent=2))
    print("top_ids", [item["id"] for item in payload["top"][:20]])
    print("unreviewed_top_ids", [item["id"] for item in payload["unreviewed_top"][:20]])
    print("output", args.output)


if __name__ == "__main__":
    main()
