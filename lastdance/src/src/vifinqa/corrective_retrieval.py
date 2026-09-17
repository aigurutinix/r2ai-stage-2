"""Consensus gates for corrective, retrieval-grounded FinancialPlans.

The module intentionally does not choose facts from the baseline answer.  Two
independent assignment proposals are bound to the corpus, evaluated through the
same deterministic IR, and compared fact by fact.  A changed answer is only a
candidate for a later semantic audit when every fact has exact row consensus.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional

from rapidfuzz.fuzz import ratio, token_set_ratio

from .experiments import SubmissionArchive
from .financial_ir import FinancialPlan, evaluate_plan
from .grounded_plans import GroundedBinder, GroundedCell
from .l1_fact_layer import normalize


SEMANTIC_STOPWORDS = {"cua", "tai", "trong", "tu", "va", "theo", "cho"}


@dataclass(frozen=True)
class BoundAssignment:
    question_id: int
    model: str
    row_refs: Mapping[str, str]
    bindings: Mapping[str, GroundedCell]
    answer: float


def load_valid_assignments(path: Path, model: str) -> dict[int, dict[str, Any]]:
    """Load one validated assignment per question without trusting raw text."""

    result: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("status") != "VALID":
            continue
        question_id = int(row["question_id"])
        if question_id in result:
            raise ValueError(f"{path}: duplicate valid q{question_id} for {model}")
        result[question_id] = row
    return result


def _assignment_refs(plan: FinancialPlan, row: Mapping[str, Any]) -> dict[str, str]:
    expected = {fact.id for fact in plan.facts}
    refs: dict[str, str] = {}
    assignments = row.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("assignments must be a list")
    for assignment in assignments:
        fact_id = assignment.get("fact_id")
        row_ref = assignment.get("row_ref")
        if fact_id not in expected:
            raise ValueError(f"unknown fact_id {fact_id!r}")
        if fact_id in refs:
            raise ValueError(f"duplicate fact_id {fact_id!r}")
        if not isinstance(row_ref, str):
            raise ValueError(f"fact {fact_id}: row_ref must be a string")
        refs[fact_id] = row_ref
    if set(refs) != expected:
        raise ValueError(f"missing fact_ids {sorted(expected - set(refs))}")
    return refs


def bind_assignment(
    plan: FinancialPlan,
    row: Mapping[str, Any],
    model: str,
    binder: GroundedBinder,
) -> BoundAssignment:
    """Bind exactly what a model proposed; never silently use a fallback row."""

    refs = _assignment_refs(plan, row)
    bindings: dict[str, GroundedCell] = {}
    for fact in plan.facts:
        grounded = replace(fact, row_ref=refs[fact.id])
        bindings[fact.id] = binder.bind_fact(plan, grounded)
    answer = evaluate_plan(plan, {key: value.value for key, value in bindings.items()})
    return BoundAssignment(plan.question_id, model, refs, bindings, answer)


def label_similarity(metric: str, row_text: str) -> float:
    metric_value = normalize(metric)
    row_value = normalize(row_text)
    return 0.65 * token_set_ratio(metric_value, row_value) + 0.35 * ratio(
        metric_value, row_value
    )


def metric_token_coverage(metric: str, raw_row_text: str) -> float:
    """Measure whether the row itself, not just its section, names the metric."""

    metric_tokens = {
        token
        for token in normalize(metric).split()
        if token not in SEMANTIC_STOPWORDS
    }
    if not metric_tokens:
        return 0.0
    row_tokens = set(normalize(raw_row_text).split())
    return len(metric_tokens & row_tokens) / len(metric_tokens)


def column_year_matches(fact_year: int, period: str, column_text: str) -> bool:
    """Reject a prior-year comparative column for a current/end/flow fact."""

    years = {
        int(value)
        for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", column_text)
    }
    if not years:
        return True
    if period == "start":
        return bool(years & {fact_year - 1, fact_year})
    return fact_year in years


def zero_weighted_fact_ids(plan: FinancialPlan) -> set[str]:
    """Find direct ``fact * 0`` registry constructs for external auditing."""

    fact_ids = {fact.id for fact in plan.facts}
    zero_literals = {
        node.id
        for node in plan.nodes
        if node.op == "literal"
        and float(node.params.get("value", math.nan)) == 0.0
    }
    result = set()
    for node in plan.nodes:
        if node.op != "multiply" or len(node.inputs) != 2:
            continue
        left, right = node.inputs
        if left in fact_ids and right in zero_literals:
            result.add(left)
        if right in fact_ids and left in zero_literals:
            result.add(right)
    return result


def _duplicate_cell_conflicts(
    plan: FinancialPlan, assignment: BoundAssignment
) -> list[dict[str, Any]]:
    """Flag different requested facts that collapse onto the same source cell."""

    grouped: dict[tuple[int, int, int], list[str]] = {}
    by_id = {fact.id: fact for fact in plan.facts}
    for fact_id, binding in assignment.bindings.items():
        key = (binding.table_id, binding.row_index, binding.column_index)
        grouped.setdefault(key, []).append(fact_id)
    conflicts = []
    for cell, fact_ids in grouped.items():
        if len(fact_ids) < 2:
            continue
        signatures = {
            (
                by_id[fact_id].ticker,
                by_id[fact_id].year,
                normalize(by_id[fact_id].metric),
                by_id[fact_id].period,
                by_id[fact_id].scope,
            )
            for fact_id in fact_ids
        }
        if len(signatures) > 1:
            conflicts.append({"cell": cell, "fact_ids": fact_ids})
    return conflicts


def compare_assignments(
    plan: FinancialPlan,
    first: BoundAssignment,
    second: BoundAssignment,
    baseline_answer: float,
) -> dict[str, Any]:
    """Return an auditable consensus verdict for one question."""

    facts_by_id = {fact.id: fact for fact in plan.facts}
    fact_audit = []
    exact_consensus = True
    for fact_id in sorted(facts_by_id):
        left = first.bindings[fact_id]
        right = second.bindings[fact_id]
        exact = first.row_refs[fact_id] == second.row_refs[fact_id]
        exact_consensus = exact_consensus and exact
        same_value = math.isclose(left.value, right.value, rel_tol=1e-9, abs_tol=1e-7)
        fact_audit.append(
            {
                "fact_id": fact_id,
                "metric": facts_by_id[fact_id].metric,
                "first_row_ref": first.row_refs[fact_id],
                "second_row_ref": second.row_refs[fact_id],
                "exact_row_consensus": exact,
                "same_value": same_value,
                "first_value": left.value,
                "second_value": right.value,
                "first_row_text": left.row_text,
                "second_row_text": right.row_text,
                "first_label_score": round(
                    label_similarity(facts_by_id[fact_id].metric, left.row_text), 4
                ),
                "second_label_score": round(
                    label_similarity(facts_by_id[fact_id].metric, right.row_text), 4
                ),
                "first_column": left.column_text,
                "second_column": right.column_text,
                "first_ambiguity": left.ambiguity,
                "second_ambiguity": right.ambiguity,
            }
        )

    answer_consensus = math.isclose(
        first.answer, second.answer, rel_tol=1e-9, abs_tol=1e-7
    )
    duplicate_conflicts = {
        first.model: _duplicate_cell_conflicts(plan, first),
        second.model: _duplicate_cell_conflicts(plan, second),
    }
    changed = answer_consensus and not math.isclose(
        first.answer, baseline_answer, rel_tol=1e-6, abs_tol=1e-9
    )
    if any(duplicate_conflicts.values()):
        status = "REJECTED_DUPLICATE_CELL"
    elif not answer_consensus:
        status = "REJECTED_ANSWER_DISAGREEMENT"
    elif not exact_consensus:
        status = "REJECTED_ROW_DISAGREEMENT"
    elif not changed:
        status = "CONSENSUS_BASELINE"
    else:
        status = "NEEDS_SEMANTIC_AUDIT"

    return {
        "question_id": plan.question_id,
        "status": status,
        "question": plan.question,
        "first_model": first.model,
        "second_model": second.model,
        "first_answer": first.answer,
        "second_answer": second.answer,
        "baseline_answer": baseline_answer,
        "exact_row_consensus": exact_consensus,
        "answer_consensus": answer_consensus,
        "duplicate_cell_conflicts": duplicate_conflicts,
        "facts": fact_audit,
        "consensus_assignments": [
            {"fact_id": fact.id, "row_ref": first.row_refs[fact.id]}
            for fact in plan.facts
        ]
        if exact_consensus
        else [],
    }


def build_consensus_audit(
    *,
    baseline_zip: Path,
    plans_path: Path,
    first_assignments_path: Path,
    second_assignments_path: Path,
    first_model: str,
    second_model: str,
    database: Path,
    output_path: Path,
) -> dict[str, Any]:
    plans = {
        plan.question_id: plan
        for plan in (
            FinancialPlan.from_dict(json.loads(line))
            for line in plans_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    first_rows = load_valid_assignments(first_assignments_path, first_model)
    second_rows = load_valid_assignments(second_assignments_path, second_model)
    baseline = SubmissionArchive.load(baseline_zip)
    output = []
    with GroundedBinder(database) as binder:
        for question_id, plan in sorted(plans.items()):
            record: dict[str, Any]
            if question_id not in first_rows or question_id not in second_rows:
                record = {
                    "question_id": question_id,
                    "status": "REJECTED_MISSING_PROPOSAL",
                    "first_present": question_id in first_rows,
                    "second_present": question_id in second_rows,
                }
            else:
                try:
                    first = bind_assignment(
                        plan, first_rows[question_id], first_model, binder
                    )
                    second = bind_assignment(
                        plan, second_rows[question_id], second_model, binder
                    )
                    record = compare_assignments(
                        plan,
                        first,
                        second,
                        float(baseline.by_id[question_id]["answer"]),
                    )
                except (KeyError, TypeError, ValueError, IndexError) as error:
                    record = {
                        "question_id": question_id,
                        "status": "REJECTED_BIND",
                        "reason": str(error),
                    }
            output.append(record)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output),
        encoding="utf-8",
    )
    counts: dict[str, int] = {}
    for row in output:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return {
        "questions": len(output),
        "status_counts": counts,
        "semantic_audit_ids": [
            row["question_id"]
            for row in output
            if row["status"] == "NEEDS_SEMANTIC_AUDIT"
        ],
        "output": str(output_path),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-zip", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--first-assignments", type=Path, required=True)
    parser.add_argument("--second-assignments", type=Path, required=True)
    parser.add_argument("--first-model", default="qwen")
    parser.add_argument("--second-model", default="deepseek")
    parser.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_consensus_audit(
        baseline_zip=args.baseline_zip,
        plans_path=args.plans,
        first_assignments_path=args.first_assignments,
        second_assignments_path=args.second_assignments,
        first_model=args.first_model,
        second_model=args.second_model,
        database=args.database,
        output_path=args.output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
