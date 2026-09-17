"""Fail-closed canonical challenger for direct period growth questions."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..finance.metrics import get_metric, metric_uses_absolute_value
from ..utils.viet_text import norm
from .exact_lookup import exact_metric_scope_error
from .fact_resolver import ResolvedFact, resolve_requirement
from .units import check_answer_unit


@dataclass
class ExactGrowthAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    confidence: float = 0.0
    detail: str = ""
    tier: str = ""
    resolved: list[ResolvedFact] = field(default_factory=list)


def try_exact_growth_answer(
        route: dict, tables: list[dict]) -> ExactGrowthAnswer:
    """Compute endpoint growth for one atomic metric and one company."""
    plan = route.get("plan") or {}
    if plan.get("op") != "growth_pct":
        return ExactGrowthAnswer(False, detail="not a growth_pct route")
    output_type = str(route.get("output_type") or "percent")
    if output_type != "percent":
        return ExactGrowthAnswer(
            False, detail=f"unsupported growth output={output_type}")

    requirements = route.get("evidence_requirements") or []
    if len(requirements) < 2:
        return ExactGrowthAnswer(
            False, detail=f"canonical requirements={len(requirements)}")
    metric_keys = [str(req.get("metric_key") or "") for req in requirements]
    if not metric_keys[0] or len(set(metric_keys)) != 1:
        return ExactGrowthAnswer(
            False, detail=f"growth metrics={sorted(set(metric_keys))}")
    metric = get_metric(metric_keys[0])
    if metric.components:
        return ExactGrowthAnswer(
            False, detail=f"derived metric components={len(metric.components)}")

    scopes = [
        (str(req.get("ticker") or "").upper(), int(req.get("year") or 0))
        for req in requirements
    ]
    tickers = {ticker for ticker, _year in scopes if ticker}
    years = {year for _ticker, year in scopes if year > 0}
    if len(tickers) != 1 or len(years) < 2:
        return ExactGrowthAnswer(
            False, detail=(f"growth axis tickers={len(tickers)} "
                           f"years={len(years)}"))
    if len(scopes) != len(years) or any(not ticker or year <= 0
                                        for ticker, year in scopes):
        return ExactGrowthAnswer(
            False, detail="growth scope contains duplicates or missing periods")

    scope_error = exact_metric_scope_error(route.get("question", ""), metric)
    if scope_error:
        return ExactGrowthAnswer(False, detail=scope_error)

    base_year, end_year = min(years), max(years)
    endpoint_requirements = [
        next(req for req in requirements if int(req.get("year") or 0) == year)
        for year in (base_year, end_year)
    ]
    resolved = []
    for requirement in endpoint_requirements:
        fact = resolve_requirement(
            requirement, tables,
            question=str(requirement.get("metric_label") or metric.label),
        )
        if fact is None:
            return ExactGrowthAnswer(
                False,
                detail=f"exact operand unresolved {requirement.get('requirement_id')}",
                resolved=resolved,
            )
        resolved.append(fact)
    identities = {
        (fact.report_id, fact.table_pos, fact.row, fact.col, fact.value_column)
        for fact in resolved
    }
    if len(identities) != 2:
        return ExactGrowthAnswer(
            False, detail="growth operands resolve to the same cell",
            resolved=resolved)

    base, end = resolved
    absolute = metric_uses_absolute_value(metric.label, (metric.key,))
    base_value = abs(base.value_vnd) if absolute else base.value_vnd
    end_value = abs(end.value_vnd) if absolute else end.value_vnd
    if base_value == 0:
        return ExactGrowthAnswer(
            False, detail="growth base is zero", resolved=resolved)
    asks_decrease = bool(re.search(
        r"\bgiam\s+bao nhieu\s*(?:%|phan tram\b)",
        norm(route.get("question", "")),
    ))
    first_value, last_value = (
        (base_value, end_value) if asks_decrease
        else (end_value, base_value)
    )
    answer = round(float((first_value - last_value) / abs(base_value) * 100), 2)
    warning = check_answer_unit(answer, output_type)
    if warning:
        return ExactGrowthAnswer(
            False, detail=f"unit guard: {warning}", resolved=resolved)

    base_expr, end_expr = base.expr_vnd(), end.expr_vnd()
    if absolute:
        base_expr, end_expr = f"abs({base_expr})", f"abs({end_expr})"
    first_expr, last_expr = (
        (base_expr, end_expr) if asks_decrease else (end_expr, base_expr)
    )
    query = (
        f"round(({first_expr} - {last_expr}) / abs({base_expr}) * 100, 2)"
    )
    tier, confidence = _growth_tier(resolved, metric)
    cells = ",".join(
        f"{fact.report_id}|{fact.table_pos}|r{fact.row}c{fact.col}"
        for fact in resolved
    )
    return ExactGrowthAnswer(
        True, answer, query, confidence,
        detail=(f"exact_growth metric={metric.key} tier={tier} "
                f"period={base_year}:{end_year} cells={cells}"),
        tier=tier, resolved=resolved,
    )


def _growth_tier(
        resolved: list[ResolvedFact], metric) -> tuple[str, float]:
    kinds = []
    expected_codes = set(metric.codes)
    for fact in resolved:
        code = re.sub(r"\.0$", "", str(fact.code or "").strip())
        report_year = _report_year(fact.report_id)
        if expected_codes and code in expected_codes and report_year == fact.year:
            kinds.append("vas_current")
        elif (expected_codes and code in expected_codes
              and report_year == (fact.year or 0) + 1):
            kinds.append("vas_prior")
        else:
            kinds.append("note_exact")
    if all(kind == "vas_current" for kind in kinds):
        return "vas_growth_current", 99.0
    if all(kind in {"vas_current", "vas_prior"} for kind in kinds):
        return "vas_growth_mixed", 97.0
    return "note_growth_exact", 94.0


def _report_year(report_id: str) -> int | None:
    found = re.search(
        r"(?:financial_statements_|_)(20\d{2})(?:_|$)", str(report_id))
    return int(found.group(1)) if found else None
