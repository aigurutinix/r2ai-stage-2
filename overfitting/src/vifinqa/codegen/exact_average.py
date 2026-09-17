"""Fail-closed canonical challenger for direct arithmetic means."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..finance.metrics import get_metric, metric_uses_absolute_value
from .exact_lookup import exact_metric_scope_error
from .fact_resolver import ResolvedFact, resolve_requirement
from .units import check_answer_unit


@dataclass
class ExactAverageAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    confidence: float = 0.0
    detail: str = ""
    tier: str = ""
    resolved: list[ResolvedFact] = field(default_factory=list)


def try_exact_average_answer(
        route: dict, tables: list[dict]) -> ExactAverageAnswer:
    """Average one atomic metric over exactly one varying scope axis."""
    plan = route.get("plan") or {}
    if plan.get("op") != "average":
        return ExactAverageAnswer(False, detail="not an average route")
    output_type = str(route.get("output_type") or "number")
    if output_type != "number":
        return ExactAverageAnswer(
            False, detail=f"unsupported average output={output_type}")

    requirements = route.get("evidence_requirements") or []
    if len(requirements) < 2:
        return ExactAverageAnswer(
            False, detail=f"canonical requirements={len(requirements)}")
    metric_keys = [str(req.get("metric_key") or "") for req in requirements]
    if not metric_keys[0] or len(set(metric_keys)) != 1:
        return ExactAverageAnswer(
            False, detail=f"average metrics={sorted(set(metric_keys))}")
    metric = get_metric(metric_keys[0])
    if metric.components:
        return ExactAverageAnswer(
            False, detail=f"derived metric components={len(metric.components)}")

    scopes = [
        (str(req.get("ticker") or "").upper(), int(req.get("year") or 0))
        for req in requirements
    ]
    if any(not ticker or year <= 0 for ticker, year in scopes):
        return ExactAverageAnswer(False, detail="average scope incomplete")
    if len(set(scopes)) != len(scopes):
        return ExactAverageAnswer(False, detail="average scope contains duplicates")
    tickers = {ticker for ticker, _year in scopes}
    years = {year for _ticker, year in scopes}
    if not ((len(tickers) == 1 and len(years) == len(scopes))
            or (len(years) == 1 and len(tickers) == len(scopes))):
        return ExactAverageAnswer(
            False, detail=(f"average axis tickers={len(tickers)} "
                           f"years={len(years)} requirements={len(scopes)}"))

    scope_error = exact_metric_scope_error(route.get("question", ""), metric)
    if scope_error:
        return ExactAverageAnswer(False, detail=scope_error)

    resolved = []
    for requirement in requirements:
        fact = resolve_requirement(
            requirement, tables,
            question=str(requirement.get("metric_label") or metric.label),
        )
        if fact is None:
            return ExactAverageAnswer(
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
        return ExactAverageAnswer(
            False, detail="average operands resolve to duplicate cells",
            resolved=resolved)

    absolute = metric_uses_absolute_value(
        " ".join((str(route.get("question") or ""), metric.label)),
        (metric.key,),
    )
    values = [abs(fact.value_vnd) if absolute else fact.value_vnd
              for fact in resolved]
    q_scale = float(route.get("unit_scale", 1.0) or 1.0)
    answer = round(float(sum(values) / len(values) / q_scale), 2)
    warning = check_answer_unit(answer, output_type)
    if warning:
        return ExactAverageAnswer(
            False, detail=f"unit guard: {warning}", resolved=resolved)
    if q_scale >= 1e6 and abs(answer) > 1e10:
        return ExactAverageAnswer(
            False, detail="unit guard: implausible scaled amount",
            resolved=resolved)

    expressions = [fact.expr_vnd() for fact in resolved]
    if absolute:
        expressions = [f"abs({expr})" for expr in expressions]
    query = (
        f"round(({' + '.join(expressions)}) / {len(expressions)} "
        f"/ {q_scale:g}, 2)"
    )
    tier, confidence = _average_tier(resolved, metric)
    cells = ",".join(
        f"{fact.report_id}|{fact.table_pos}|r{fact.row}c{fact.col}"
        for fact in resolved
    )
    return ExactAverageAnswer(
        True, answer, query, confidence,
        detail=(f"exact_average metric={metric.key} tier={tier} "
                f"n={len(resolved)} cells={cells}"),
        tier=tier, resolved=resolved,
    )


def _average_tier(
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
        return "vas_average_current", 99.0
    if all(kind in {"vas_current", "vas_prior"} for kind in kinds):
        return "vas_average_mixed", 97.0
    return "note_average_exact", 94.0


def _report_year(report_id: str) -> int | None:
    found = re.search(
        r"(?:financial_statements_|_)(20\d{2})(?:_|$)", str(report_id))
    return int(found.group(1)) if found else None
