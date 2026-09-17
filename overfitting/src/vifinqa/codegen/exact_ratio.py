"""Fail-closed canonical challenger for direct financial ratios."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..finance.metrics import get_metric, metric_keys, metric_uses_absolute_value
from ..utils.viet_text import norm
from .exact_lookup import exact_metric_scope_error
from .fact_resolver import ResolvedFact, resolve_requirement
from .units import check_answer_unit


_OPENING_MARKERS = ("dau nam", "dau ky", "ngay 01 thang 01")
_GROUPED_NUMERATOR = (
    "selling_expense", "administrative_expense", "net_revenue",
)


@dataclass
class ExactRatioAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    confidence: float = 0.0
    detail: str = ""
    tier: str = ""
    resolved: list[ResolvedFact] = field(default_factory=list)


def try_exact_ratio_answer(
        route: dict, tables: list[dict]) -> ExactRatioAnswer:
    """Compute a direct percent ratio from two or three canonical operands."""
    plan = route.get("plan") or {}
    if plan.get("op") != "ratio":
        return ExactRatioAnswer(False, detail="not a ratio route")
    output_type = str(route.get("output_type") or "percent")
    if output_type != "percent":
        return ExactRatioAnswer(
            False, detail=f"unsupported ratio output={output_type}")

    question = str(route.get("question") or "")
    question_norm = norm(question)
    opening_date = re.search(
        r"(?<!\d)0?1\s*/\s*0?1(?:\s*/\s*20\d{2}|(?!\d))",
        question_norm,
    )
    if any(marker in question_norm for marker in _OPENING_MARKERS) or opening_date:
        return ExactRatioAnswer(False, detail="opening-period ratio is ambiguous")

    requirements = route.get("evidence_requirements") or []
    facts = plan.get("facts") or []
    roles = [str(fact.get("role") or "") for fact in facts]
    if len(facts) != 2 or roles != ["numerator", "denominator"]:
        return ExactRatioAnswer(
            False, detail=f"ratio roles={roles}")
    if len(requirements) not in {2, 3}:
        return ExactRatioAnswer(
            False, detail=f"canonical requirements={len(requirements)}")

    keys = tuple(str(req.get("metric_key") or "") for req in requirements)
    if not all(keys):
        return ExactRatioAnswer(False, detail="canonical ratio metric missing")
    if len(requirements) == 3 and keys != _GROUPED_NUMERATOR:
        return ExactRatioAnswer(
            False, detail=f"unsupported grouped numerator={keys}")

    scopes = {
        (str(req.get("ticker") or "").upper(), int(req.get("year") or 0),
         str(req.get("doc_type") or ""))
        for req in requirements
    }
    if len(scopes) != 1 or any(not ticker or year <= 0
                               for ticker, year, _doc_type in scopes):
        return ExactRatioAnswer(False, detail=f"ratio scopes={sorted(scopes)}")

    metrics = [get_metric(key) for key in keys]
    if any(metric.components for metric in metrics):
        return ExactRatioAnswer(False, detail="derived ratio operand")

    role_phrases = [str(fact.get("metric") or "") for fact in facts]
    metric_groups = (
        [metrics[:2], metrics[2:]]
        if len(metrics) == 3 else
        [[metrics[0]], [metrics[1]]]
    )
    for phrase, group in zip(role_phrases, metric_groups):
        inferred = set(metric_keys([phrase], expand_derived=False))
        group_keys = {metric.key for metric in group}
        if inferred and not inferred.issubset(group_keys):
            return ExactRatioAnswer(
                False, detail=(f"ratio role mismatch phrase={phrase!r} "
                               f"inferred={sorted(inferred)} expected={sorted(group_keys)}"))
        for metric in group:
            scope_error = exact_metric_scope_error(phrase, metric)
            if scope_error:
                return ExactRatioAnswer(False, detail=scope_error)

    resolved: list[ResolvedFact] = []
    for requirement, metric in zip(requirements, metrics):
        fact = resolve_requirement(
            requirement, tables,
            question=str(requirement.get("metric_label") or metric.label),
        )
        if fact is None:
            return ExactRatioAnswer(
                False,
                detail=f"exact operand unresolved {requirement.get('requirement_id')}",
                resolved=resolved,
            )
        resolved.append(fact)

    identities = {
        (fact.report_id, fact.table_pos, fact.row, fact.col, fact.value_column)
        for fact in resolved
    }
    if len(identities) != len(resolved):
        return ExactRatioAnswer(
            False, detail="ratio operands contain duplicate cells", resolved=resolved)

    values = []
    expressions = []
    for fact, metric in zip(resolved, metrics):
        absolute = metric_uses_absolute_value(metric.label, (metric.key,))
        values.append(abs(fact.value_vnd) if absolute else fact.value_vnd)
        expression = fact.expr_vnd()
        expressions.append(f"abs({expression})" if absolute else expression)

    numerator_count = 2 if len(requirements) == 3 else 1
    numerator = sum(values[:numerator_count])
    denominator = values[-1]
    if denominator == 0:
        return ExactRatioAnswer(
            False, detail="ratio denominator is zero", resolved=resolved)
    answer = round(float(numerator / denominator * 100), 2)
    warning = check_answer_unit(answer, output_type)
    if warning:
        return ExactRatioAnswer(
            False, detail=f"unit guard: {warning}", resolved=resolved)

    numerator_expr = (
        expressions[0] if numerator_count == 1
        else f"({expressions[0]} + {expressions[1]})"
    )
    query = f"round({numerator_expr} / {expressions[-1]} * 100, 2)"
    tier, confidence = _ratio_tier(resolved, metrics)
    cells = ",".join(
        f"{fact.report_id}|{fact.table_pos}|r{fact.row}c{fact.col}"
        for fact in resolved
    )
    return ExactRatioAnswer(
        True, answer, query, confidence,
        detail=(f"exact_ratio metrics={','.join(keys)} tier={tier} "
                f"cells={cells}"),
        tier=tier, resolved=resolved,
    )


def _ratio_tier(
        resolved: list[ResolvedFact], metrics: list) -> tuple[str, float]:
    kinds = []
    for fact, metric in zip(resolved, metrics):
        code = re.sub(r"\.0$", "", str(fact.code or "").strip())
        report_year = _report_year(fact.report_id)
        expected_codes = set(metric.codes)
        if expected_codes and code in expected_codes and report_year == fact.year:
            kinds.append("vas_current")
        elif (expected_codes and code in expected_codes
              and report_year == (fact.year or 0) + 1):
            kinds.append("vas_prior")
        else:
            kinds.append("note_exact")
    if all(kind == "vas_current" for kind in kinds):
        return "vas_ratio_current", 99.0
    if all(kind in {"vas_current", "vas_prior"} for kind in kinds):
        return "vas_ratio_mixed", 97.0
    return "note_ratio_exact", 94.0


def _report_year(report_id: str) -> int | None:
    found = re.search(
        r"(?:financial_statements_|_)(20\d{2})(?:_|$)", str(report_id))
    return int(found.group(1)) if found else None
