"""Validation and disagreement analysis for LLM FinancialPlan proposals."""

from __future__ import annotations

import argparse
import json
import copy
from collections import Counter
from pathlib import Path
from typing import Any, Optional

from rapidfuzz.fuzz import ratio

from .financial_ir import FinancialPlan
from .l1_fact_layer import normalize


def extract_json_object(raw: str) -> dict[str, Any]:
    """Extract the last complete FinancialPlan-like object from model text."""

    decoder = json.JSONDecoder()
    candidates: list[dict[str, Any]] = []
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append(value)
    if not candidates:
        raise ValueError("No complete JSON object found in model response")
    plan_like = [value for value in candidates if "schema_version" in value and "facts" in value]
    rerank_like = [
        value for value in candidates if "question_id" in value and "selected" in value
    ]
    assignment_like = [
        value for value in candidates if "question_id" in value and "assignments" in value
    ]
    audit_like = [
        value
        for value in candidates
        if "question_id" in value and "fact_reviews" in value
    ]
    return (plan_like or rerank_like or assignment_like or audit_like or candidates)[-1]


def validate_rerank_response(
    *, raw_response: str, question_id: int, context: dict[str, Any], model: str
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": question_id,
        "model": model,
        "status": "INVALID",
        "raw_response": raw_response,
    }
    try:
        payload = extract_json_object(raw_response)
        if int(payload.get("question_id", -1)) != question_id:
            raise ValueError("reranker question_id mismatch")
        selected = payload.get("selected")
        if not isinstance(selected, list) or not selected:
            raise ValueError("reranker selected must be a non-empty array")
        allowed = {
            candidate["row_ref"]
            for table in context["tables"]
            for candidate in table["candidate_rows"]
        }
        refs = [selection.get("row_ref") for selection in selected]
        if any(ref not in allowed for ref in refs):
            raise ValueError("reranker invented a row_ref")
        if len(refs) != len(set(refs)):
            raise ValueError("reranker returned duplicate row_ref")
        row.update({"status": "VALID", "selection": payload})
    except (AttributeError, TypeError, ValueError) as error:
        row["validation_error"] = str(error)
    return row


def normalize_llm_payload(raw: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Apply only semantics-preserving repairs for common model JSON shapes."""

    payload = copy.deepcopy(raw)
    repairs: list[str] = []
    facts = payload.get("facts")
    if isinstance(facts, list):
        for index, fact in enumerate(facts):
            if isinstance(fact, dict) and "source_preference" not in fact:
                fact["source_preference"] = "auto"
                repairs.append(f"facts[{index}].source_preference=auto")

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        return payload, repairs
    known = {
        fact.get("id")
        for fact in facts or []
        if isinstance(fact, dict) and isinstance(fact.get("id"), str)
    }
    normalized_nodes = []
    for index, original in enumerate(nodes):
        if not isinstance(original, dict):
            normalized_nodes.append(original)
            continue
        node = copy.deepcopy(original)
        operation = node.get("op")
        inputs = node.get("inputs")
        params = node.get("params")
        if not isinstance(inputs, list) or not isinstance(params, dict):
            normalized_nodes.append(node)
            continue

        if (
            node.get("id") in known
            and operation == "identity"
            and not inputs
            and not params
        ):
            repairs.append(f"nodes[{index}]:drop_redundant_fact_identity")
            continue

        if operation == "literal":
            reference = None
            if len(inputs) == 1 and not params:
                reference = inputs[0]
            elif not inputs and len(params) == 1:
                value = next(iter(params.values()))
                if isinstance(value, str):
                    reference = value
            if isinstance(reference, str):
                node["op"] = "identity"
                node["inputs"] = [reference]
                node["params"] = {}
                repairs.append(f"nodes[{index}]:literal_reference->identity")

        operation = node.get("op")
        inputs = node.get("inputs")
        params = node.get("params")
        if operation == "multiply" and isinstance(inputs, list) and isinstance(params, dict):
            factor = params.get("literal")
            if isinstance(factor, (int, float)) and not isinstance(factor, bool):
                real_inputs = [
                    value
                    for value in inputs
                    if value not in {"literal", f"literal_{factor}", f"literal_{int(factor)}"}
                ]
                if len(real_inputs) == 1:
                    node["op"] = "scale"
                    node["inputs"] = real_inputs
                    node["params"] = {"factor": factor}
                    repairs.append(f"nodes[{index}]:multiply_literal->scale")
            elif not params and len(inputs) == 2:
                missing_literals = [
                    value
                    for value in inputs
                    if isinstance(value, str)
                    and value not in known
                    and value.startswith("literal_")
                ]
                if len(missing_literals) == 1:
                    suffix = missing_literals[0].removeprefix("literal_")
                    try:
                        factor = float(suffix)
                    except ValueError:
                        factor = None
                    if factor is not None:
                        node["op"] = "scale"
                        node["inputs"] = [value for value in inputs if value != missing_literals[0]]
                        node["params"] = {"factor": factor}
                        repairs.append(f"nodes[{index}]:missing_literal->scale")

        if (
            node.get("op") in {"select_argmax", "select_argmin"}
            and len(node.get("inputs", [])) == 1
            and node["inputs"][0] in known
            and not node.get("params")
        ):
            node["op"] = "identity"
            repairs.append(f"nodes[{index}]:scalar_selector->identity")

        if node.get("op") in {"filter", "filter_by", "count_if"}:
            comparator = node.get("params", {}).get("comparator")
            comparator_map = {">": "gt", ">=": "ge", "<": "lt", "<=": "le", "==": "eq", "!=": "ne"}
            if comparator in comparator_map:
                node["params"]["comparator"] = comparator_map[comparator]
                repairs.append(f"nodes[{index}]:comparator_normalized")
        normalized_nodes.append(node)
        if isinstance(node.get("id"), str):
            known.add(node["id"])
    payload["nodes"] = normalized_nodes
    return payload, repairs


def _fact_alignment(teacher: FinancialPlan, candidate: FinancialPlan) -> dict[str, Any]:
    unused = set(range(len(candidate.facts)))
    rows = []
    for expected in teacher.facts:
        best_index = None
        best_score = -1.0
        best_metric_score = 0.0
        for index in unused:
            proposed = candidate.facts[index]
            metric_score = ratio(normalize(expected.metric), normalize(proposed.metric)) / 100.0
            dimension_score = sum(
                (
                    expected.ticker == proposed.ticker,
                    expected.year == proposed.year,
                    expected.scope == proposed.scope,
                    expected.period == proposed.period,
                    expected.unit == proposed.unit,
                )
            ) / 5.0
            score = 0.65 * dimension_score + 0.35 * metric_score
            if score > best_score:
                best_index = index
                best_score = score
                best_metric_score = metric_score
        if best_index is None:
            rows.append({"teacher": expected.id, "candidate": None, "score": 0.0})
            continue
        unused.remove(best_index)
        proposed = candidate.facts[best_index]
        rows.append(
            {
                "teacher": expected.id,
                "candidate": proposed.id,
                "score": round(best_score, 4),
                "metric_similarity": round(best_metric_score, 4),
                "ticker_match": expected.ticker == proposed.ticker,
                "year_match": expected.year == proposed.year,
                "scope_match": expected.scope == proposed.scope,
                "period_match": expected.period == proposed.period,
                "unit_match": expected.unit == proposed.unit,
            }
        )
    return {
        "matches": rows,
        "mean_score": round(sum(row["score"] for row in rows) / len(rows), 4),
        "unmatched_candidate_facts": [candidate.facts[index].id for index in sorted(unused)],
    }


def _canonical_operators(plan: FinancialPlan) -> Counter[str]:
    """Expand convenience operators before comparing semantic structure."""

    result: Counter[str] = Counter()
    for node in plan.nodes:
        if node.op in {"literal", "identity"}:
            continue
        if node.op == "scale":
            result["multiply"] += 1
        elif node.op == "ratio_percent":
            result.update(("divide", "multiply"))
        elif node.op == "percent_change":
            result.update(("subtract", "divide", "multiply"))
        else:
            result[node.op] += 1
    return result


def compare_plans(teacher: FinancialPlan, candidate: FinancialPlan) -> dict[str, Any]:
    alignment = _fact_alignment(teacher, candidate)
    teacher_dimensions = Counter(
        (fact.ticker, fact.year, fact.scope) for fact in teacher.facts
    )
    candidate_dimensions = Counter(
        (fact.ticker, fact.year, fact.scope) for fact in candidate.facts
    )
    teacher_ops = _canonical_operators(teacher)
    candidate_ops = _canonical_operators(candidate)
    checks = {
        "fact_count": len(teacher.facts) == len(candidate.facts),
        "dimensions": teacher_dimensions == candidate_dimensions,
        "operators": teacher_ops == candidate_ops,
        "output_unit": teacher.output_unit == candidate.output_unit,
    }
    agreement_score = (
        0.55 * alignment["mean_score"]
        + 0.15 * float(checks["fact_count"])
        + 0.15 * float(checks["dimensions"])
        + 0.10 * float(checks["operators"])
        + 0.05 * float(checks["output_unit"])
    )
    return {
        "agreement_score": round(agreement_score, 4),
        "checks": checks,
        "teacher_operators": dict(sorted(teacher_ops.items())),
        "candidate_operators": dict(sorted(candidate_ops.items())),
        "fact_alignment": alignment,
    }


def validate_llm_response(
    *,
    raw_response: str,
    teacher: FinancialPlan,
    model: str,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "question_id": teacher.question_id,
        "model": model,
        "status": "INVALID",
        "raw_response": raw_response,
    }
    try:
        payload, repairs = normalize_llm_payload(extract_json_object(raw_response))
        candidate = FinancialPlan.from_dict(payload)
        if candidate.question_id != teacher.question_id:
            raise ValueError(
                f"question_id mismatch: {candidate.question_id} != {teacher.question_id}"
            )
        if candidate.question != teacher.question:
            raise ValueError("candidate changed the question text")
        allowed_tickers = {fact.ticker for fact in teacher.facts}
        unexpected_tickers = sorted({fact.ticker for fact in candidate.facts} - allowed_tickers)
        if unexpected_tickers:
            raise ValueError(f"candidate invented tickers: {unexpected_tickers}")
        row.update(
            {
                "status": "VALID_REPAIRED" if repairs else "VALID",
                "repairs": repairs,
                "candidate": candidate.to_dict(),
                "comparison": compare_plans(teacher, candidate),
            }
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        row["validation_error"] = str(error)
    return row


def reanalyze_file(responses_path: Path, teachers_path: Path, output_path: Path) -> dict[str, Any]:
    teachers = {
        plan.question_id: plan
        for plan in (
            FinancialPlan.from_dict(json.loads(line))
            for line in teachers_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    rows = [
        json.loads(line)
        for line in responses_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    analyzed = [
        validate_llm_response(
            raw_response=row["raw_response"],
            teacher=teachers[int(row["question_id"])],
            model=row["model"],
        )
        for row in rows
    ]
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in analyzed),
        encoding="utf-8",
    )
    valid = [row for row in analyzed if row["status"].startswith("VALID")]
    return {
        "proposals": len(analyzed),
        "valid": len(valid),
        "repaired": sum(row["status"] == "VALID_REPAIRED" for row in analyzed),
        "mean_agreement": (
            sum(row["comparison"]["agreement_score"] for row in valid) / len(valid)
            if valid
            else 0.0
        ),
        "output": str(output_path),
    }


def reanalyze_rerank_file(
    responses_path: Path, contexts_path: Path, output_path: Path
) -> dict[str, Any]:
    contexts = {
        int(row["question_id"]): row
        for row in (
            json.loads(line)
            for line in contexts_path.read_text(encoding="utf-8").splitlines()
            if line
        )
    }
    rows = [
        json.loads(line)
        for line in responses_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    analyzed = [
        validate_rerank_response(
            raw_response=row["raw_response"],
            question_id=int(row["question_id"]),
            context=contexts[int(row["question_id"])],
            model=row["model"],
        )
        for row in rows
    ]
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in analyzed),
        encoding="utf-8",
    )
    return {
        "proposals": len(analyzed),
        "valid": sum(row["status"] == "VALID" for row in analyzed),
        "selected_rows": sum(
            len(row.get("selection", {}).get("selected", [])) for row in analyzed
        ),
        "output": str(output_path),
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--teachers", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument("--contexts", type=Path)
    args = parser.parse_args(argv)
    if args.rerank:
        if args.contexts is None:
            parser.error("--rerank requires --contexts")
        result = reanalyze_rerank_file(args.responses, args.contexts, args.output)
    else:
        if args.teachers is None:
            parser.error("plan reanalysis requires --teachers")
        result = reanalyze_file(args.responses, args.teachers, args.output)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
