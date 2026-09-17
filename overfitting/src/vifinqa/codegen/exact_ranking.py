"""Fail-closed exact challenger for direct value rankings."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..finance.metrics import get_metric, metric_uses_absolute_value
from ..utils.viet_text import norm
from .exact_lookup import exact_metric_scope_error
from .fact_resolver import ResolvedFact, resolve_requirement
from .units import check_answer_unit


@dataclass
class ExactRankingAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    confidence: float = 0.0
    detail: str = ""
    tier: str = ""
    resolved: list[ResolvedFact] = field(default_factory=list)


def try_exact_ranking_answer(
    route: dict, tables: list[dict],
) -> ExactRankingAnswer:
    """Return max/min value only when every candidate fact is canonical exact."""
    plan = route.get("plan") or {}
    if plan.get("op") != "ranking":
        return ExactRankingAnswer(False, detail="not a ranking route")
    output_type = str(route.get("output_type") or "number")
    if output_type != "number":
        return ExactRankingAnswer(
            False, detail=f"unsupported ranking output={output_type}")
    question = str(route.get("question") or "")
    if _looks_select_then_project(question):
        return ExactRankingAnswer(False, detail="nested ranking projection")

    requirements = route.get("evidence_requirements") or []
    if len(requirements) < 2:
        return ExactRankingAnswer(
            False, detail=f"canonical requirements={len(requirements)}")
    metric_keys = [str(req.get("metric_key") or "") for req in requirements]
    if not metric_keys[0] or len(set(metric_keys)) != 1:
        return ExactRankingAnswer(
            False, detail=f"ranking metrics={sorted(set(metric_keys))}")
    metric = get_metric(metric_keys[0])
    if metric.components:
        return ExactRankingAnswer(
            False, detail=f"derived metric components={len(metric.components)}")

    scopes = [
        (str(req.get("ticker") or "").upper(), req.get("year"))
        for req in requirements
    ]
    tickers = {ticker for ticker, _year in scopes if ticker}
    years = {int(year) for _ticker, year in scopes if year is not None}
    valid_axis = (
        (len(tickers) == 1 and len(years) == len(requirements))
        or (len(years) == 1 and len(tickers) == len(requirements))
    )
    if not valid_axis:
        return ExactRankingAnswer(
            False, detail=f"ranking axis tickers={len(tickers)} years={len(years)}")

    scope_error = exact_metric_scope_error(question, metric)
    if scope_error:
        return ExactRankingAnswer(False, detail=scope_error)

    resolved = []
    for requirement in requirements:
        fact = resolve_requirement(
            requirement, tables,
            question=str(requirement.get("metric_label") or metric.label),
        )
        if fact is None:
            return ExactRankingAnswer(
                False, detail=f"exact candidate unresolved {requirement.get('requirement_id')}",
                resolved=resolved)
        resolved.append(fact)
    identities = {
        (fact.report_id, fact.table_pos, fact.row, fact.col, fact.value_column)
        for fact in resolved
    }
    if len(identities) != len(resolved):
        return ExactRankingAnswer(
            False, detail="ranking candidates contain duplicate cells",
            resolved=resolved)

    absolute = metric_uses_absolute_value(
        " ".join((question, metric.label)), (metric.key,))
    values_vnd = [
        abs(fact.value_vnd) if absolute else fact.value_vnd
        for fact in resolved
    ]
    direction = _direction(question)
    selected = min(values_vnd) if direction == "min" else max(values_vnd)
    q_scale = float(route.get("unit_scale", 1.0) or 1.0)
    answer = round(float(selected / q_scale), 2)
    warning = check_answer_unit(answer, output_type)
    if warning:
        return ExactRankingAnswer(
            False, detail=f"unit guard: {warning}", resolved=resolved)
    if q_scale >= 1e6 and abs(answer) > 1e10:
        return ExactRankingAnswer(
            False, detail="unit guard: implausible scaled amount", resolved=resolved)

    expressions = [fact.expr_vnd() for fact in resolved]
    if absolute:
        expressions = [f"abs({expression})" for expression in expressions]
    query = f"round({direction}({', '.join(expressions)}) / {q_scale:g}, 2)"
    tier, confidence = _ranking_tier(resolved, metric)
    cells = ",".join(
        f"{fact.report_id}|{fact.table_pos}|r{fact.row}c{fact.col}"
        for fact in resolved
    )
    return ExactRankingAnswer(
        True, answer, query, confidence,
        detail=(f"exact_ranking metric={metric.key} direction={direction} "
                f"tier={tier} cells={cells}"),
        tier=tier, resolved=resolved,
    )


def _looks_select_then_project(question: str) -> bool:
    text = norm(question)
    return bool(
        re.search(r"\b(?:tai|trong) nam co\b", text)
        or re.search(
            r"\bcua (?:doanh nghiep|cong ty|ngan hang) co\b", text)
        or "tai cong ty co" in text
    )


def _direction(question: str) -> str:
    text = norm(question)
    if any(marker in text for marker in (
        "nho nhat", "thap nhat", "it nhat", "be nhat", "toi thieu",
    )):
        return "min"
    return "max"


def _ranking_tier(
    resolved: list[ResolvedFact], metric,
) -> tuple[str, float]:
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
        return "vas_ranking_current", 99.0
    if all(kind in {"vas_current", "vas_prior"} for kind in kinds):
        return "vas_ranking_mixed", 97.0
    return "note_ranking_exact", 94.0


def _report_year(report_id: str) -> int | None:
    found = re.search(
        r"(?:financial_statements_|_)(20\d{2})(?:_|$)", str(report_id))
    return int(found.group(1)) if found else None
