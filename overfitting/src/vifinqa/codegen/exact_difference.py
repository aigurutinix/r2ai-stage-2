"""Fail-closed two-cell challenger for successful difference answers."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..finance.metrics import get_metric, metric_uses_absolute_value
from ..utils.viet_text import norm
from .exact_lookup import exact_metric_scope_error
from .fact_resolver import ResolvedFact, resolve_requirement
from .units import check_answer_unit


@dataclass
class ExactDifferenceAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    confidence: float = 0.0
    detail: str = ""
    tier: str = ""
    resolved: list[ResolvedFact] = field(default_factory=list)


def try_exact_difference_answer(
    route: dict, tables: list[dict],
) -> ExactDifferenceAnswer:
    """Resolve a direct A-B only when both canonical operands are exact."""
    plan = route.get("plan") or {}
    if plan.get("op") != "difference":
        return ExactDifferenceAnswer(False, detail="not a difference route")
    output_type = str(route.get("output_type") or "number")
    if output_type != "number":
        return ExactDifferenceAnswer(
            False, detail=f"unsupported difference output={output_type}")

    requirements = route.get("evidence_requirements") or []
    if len(requirements) != 2:
        return ExactDifferenceAnswer(
            False, detail=f"canonical requirements={len(requirements)}")
    metric_keys = [str(req.get("metric_key") or "") for req in requirements]
    if not metric_keys[0] or len(set(metric_keys)) != 1:
        return ExactDifferenceAnswer(
            False, detail=f"difference metrics={metric_keys}")
    metric = get_metric(metric_keys[0])
    if metric.components:
        return ExactDifferenceAnswer(
            False, detail=f"derived metric components={len(metric.components)}")

    scopes = [
        (str(req.get("ticker") or "").upper(), req.get("year"))
        for req in requirements
    ]
    tickers = {ticker for ticker, _year in scopes if ticker}
    years = {int(year) for _ticker, year in scopes if year is not None}
    valid_axis = (
        (len(tickers) == 1 and len(years) == 2)
        or (len(tickers) == 2 and len(years) == 1)
    )
    if not valid_axis:
        return ExactDifferenceAnswer(
            False, detail=f"difference axis tickers={len(tickers)} years={len(years)}")

    scope_error = exact_metric_scope_error(route.get("question", ""), metric)
    if scope_error:
        return ExactDifferenceAnswer(False, detail=scope_error)

    resolved = []
    for requirement in requirements:
        fact = resolve_requirement(
            requirement, tables,
            question=str(requirement.get("metric_label") or metric.label),
        )
        if fact is None:
            return ExactDifferenceAnswer(
                False, detail=f"exact operand unresolved {requirement.get('requirement_id')}",
                resolved=resolved)
        resolved.append(fact)
    identities = {
        (fact.report_id, fact.table_pos, fact.row, fact.col, fact.value_column)
        for fact in resolved
    }
    if len(identities) != 2:
        return ExactDifferenceAnswer(
            False, detail="difference operands resolve to the same cell",
            resolved=resolved)

    first, second = _ordered_operands(resolved, route.get("question", ""))
    absolute = metric_uses_absolute_value(
        " ".join((str(route.get("question") or ""), metric.label)),
        (metric.key,),
    )
    first_value = abs(first.value_vnd) if absolute else first.value_vnd
    second_value = abs(second.value_vnd) if absolute else second.value_vnd
    q_scale = float(route.get("unit_scale", 1.0) or 1.0)
    absolute_gap = _uses_absolute_gap(route.get("question", ""))
    difference = first_value - second_value
    if absolute_gap:
        difference = abs(difference)
    answer = round(float(difference / q_scale), 2)
    warning = check_answer_unit(answer, output_type)
    if warning:
        return ExactDifferenceAnswer(
            False, detail=f"unit guard: {warning}", resolved=resolved)
    if q_scale >= 1e6 and abs(answer) > 1e10:
        return ExactDifferenceAnswer(
            False, detail="unit guard: implausible scaled amount", resolved=resolved)

    first_expr = first.expr_vnd()
    second_expr = second.expr_vnd()
    if absolute:
        first_expr, second_expr = f"abs({first_expr})", f"abs({second_expr})"
    difference_expr = f"({first_expr} - {second_expr})"
    if absolute_gap:
        difference_expr = f"abs({difference_expr})"
    query = f"round({difference_expr} / {q_scale:g}, 2)"
    tier, confidence = _pair_tier(resolved, metric)
    cells = ",".join(
        f"{fact.report_id}|{fact.table_pos}|r{fact.row}c{fact.col}"
        for fact in (first, second)
    )
    return ExactDifferenceAnswer(
        True, answer, query, confidence,
        detail=f"exact_difference metric={metric.key} tier={tier} cells={cells}",
        tier=tier, resolved=resolved,
    )


def _ordered_operands(
    resolved: list[ResolvedFact], question: str,
) -> tuple[ResolvedFact, ResolvedFact]:
    text = norm(question)
    same_ticker = len({fact.ticker for fact in resolved}) == 1
    if same_ticker:
        ordered = sorted(resolved, key=lambda fact: int(fact.year or 0), reverse=True)
    else:
        ordered = list(resolved)
    if "thap hon" in text or "it hon" in text or "nho hon" in text:
        ordered.reverse()
    return ordered[0], ordered[1]


def _uses_absolute_gap(question: str) -> bool:
    text = norm(question)
    directional_change = any(marker in text for marker in (
        "muc thay doi", "thay doi tu", "bien dong tu", "tang tu", "giam tu",
    ))
    return "chenh lech" in text and not directional_change


def _pair_tier(resolved: list[ResolvedFact], metric) -> tuple[str, float]:
    kinds = []
    expected_codes = set(metric.codes)
    for fact in resolved:
        code = re.sub(r"\.0$", "", str(fact.code or "").strip())
        report_year = _report_year(fact.report_id)
        if expected_codes and code in expected_codes and report_year == fact.year:
            kinds.append("vas_current")
        elif expected_codes and code in expected_codes and report_year == (fact.year or 0) + 1:
            kinds.append("vas_prior")
        else:
            kinds.append("note_exact")
    if all(kind == "vas_current" for kind in kinds):
        return "vas_pair_current", 99.0
    if all(kind in {"vas_current", "vas_prior"} for kind in kinds):
        return "vas_pair_mixed", 97.0
    return "note_pair_exact", 94.0


def _report_year(report_id: str) -> int | None:
    found = re.search(
        r"(?:financial_statements_|_)(20\d{2})(?:_|$)", str(report_id))
    return int(found.group(1)) if found else None
