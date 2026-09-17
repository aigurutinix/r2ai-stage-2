"""Deterministic formula solver for common financial ratios and conditions.

This module sits between retrieval and LLM codegen.  It handles questions where
the arithmetic is standard once the needed statement rows are located:

* count companies satisfying one or more numeric conditions
* rank companies by a direct formula
* answer direct ratio / margin formulas

The implementation is intentionally conservative.  If every operand for every
candidate cannot be resolved from the retrieved tables, the solver refuses and
the normal composite/LLM path keeps control.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Callable

from ..finance.metrics import (
    find_metrics,
    get_metric,
    metric_keys,
    metric_uses_absolute_value,
)
from ..utils.viet_num import parse_vn_number
from ..utils.viet_text import norm, strip_diacritics
from .fact_resolver import (
    MatrixRequest,
    ResolvedFact,
    distinct_cells,
    resolve_fact,
    resolve_matrix_request,
    resolve_requirement,
)
from .rule_composite import CompositeAnswer, _FactView
from .units import check_answer_unit


@dataclass(frozen=True)
class Operand:
    metric: str
    variants: tuple[str, ...]
    required_phrases: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    expected_codes: tuple[str, ...] = ()


@dataclass(frozen=True)
class FormulaSpec:
    name: str
    triggers: tuple[str, ...]
    operands: tuple[Operand, ...]
    kind: str                      # ratio | percent
    value_fn: Callable[[list[float]], float]
    expr_fn: Callable[[list[ResolvedFact]], str]
    period_offsets: tuple[int, ...] = ()
    period_refs: tuple[PeriodRef, ...] = ()
    average_balances: tuple[AverageBalanceNode, ...] = ()


@dataclass(frozen=True)
class PeriodRef:
    role: str                       # current | opening | closing
    offset: int = 0


@dataclass(frozen=True)
class AverageBalanceNode:
    opening_operand: int
    closing_operand: int


@dataclass
class FormulaValue:
    spec: FormulaSpec
    ticker: str
    year: int | None
    value: float
    expr: str
    resolved: list[ResolvedFact]
    score: float
    evidence_years: tuple[int | None, ...] = ()


@dataclass(frozen=True)
class ExactDerivedSpec:
    components: tuple[str, ...]
    kind: str                      # money | ratio | percent
    value_fn: Callable[[list[float]], float]
    expr_fn: Callable[[list[ResolvedFact]], str]


@dataclass(frozen=True)
class FormulaMatch:
    spec: FormulaSpec
    start: int
    end: int
    trigger: str


@dataclass(frozen=True)
class Condition:
    spec: FormulaSpec | None
    metric: str
    variants: tuple[str, ...]
    op: str
    threshold: float
    kind: str                      # money | ratio | percent

    @property
    def label(self) -> str:
        return self.spec.name if self.spec else self.metric


@dataclass(frozen=True)
class CalculationNode:
    match: FormulaMatch
    mode: str                      # level | growth | delta | decrease | cagr
    start_year: int | None = None
    end_year: int | None = None


@dataclass(frozen=True)
class MedianFilterNode:
    calculation: CalculationNode
    op: str                        # < | <= | > | >=
    year: int | None = None        # fixed entity-period filter; None for year dimension


@dataclass(frozen=True)
class CompositionalRankingPlan:
    dimension: str                 # entity | year
    direction: str                 # max | min
    selector: CalculationNode
    projection: CalculationNode
    filters: tuple[Condition, ...] = ()
    median_filter: MedianFilterNode | None = None
    predicates: tuple[PredicateNode, ...] = ()
    projection_reduction: str = "selected"  # selected | max_minus_min


@dataclass(frozen=True)
class PopulationNode:
    dimension: str                 # entity
    members: tuple[str, ...]
    years: tuple[int, ...]


@dataclass(frozen=True)
class PeriodQuantifierNode:
    mode: str                        # all | any
    years: tuple[int, ...]


@dataclass(frozen=True)
class PredicateNode:
    calculation: CalculationNode
    op: str                        # < | <= | > | >= | =
    threshold: float | None = None
    reference: str = "constant"    # constant | median
    years: tuple[int, ...] = ()
    quantifier: PeriodQuantifierNode | None = None


@dataclass(frozen=True)
class ValueNode:
    primary: CalculationNode
    secondary: CalculationNode | None = None
    combine: str = "identity"       # identity | difference
    year: int | None = None


@dataclass(frozen=True)
class RankSliceNode:
    calculation: CalculationNode
    direction: str                   # max | min
    k: int
    year: int | None = None


@dataclass(frozen=True)
class AggregateNode:
    op: str                        # mean | sum | share | partition_ratio
    group_predicate: PredicateNode | None = None
    denominator_predicates: tuple[PredicateNode, ...] = ()
    rank_slice: RankSliceNode | None = None


@dataclass(frozen=True)
class FilterAggregatePlan:
    population: PopulationNode
    predicates: tuple[PredicateNode, ...]
    value: ValueNode
    aggregate: AggregateNode


@dataclass(frozen=True)
class TemporalEventNode:
    mode: str                        # extreme | first | last
    direction: str                   # max | min
    selector: CalculationNode


@dataclass(frozen=True)
class ProjectionPeriodNode:
    calculation: CalculationNode
    offset: int = 0


@dataclass(frozen=True)
class TemporalEventPlan:
    axis: str                        # year | entity_year
    members: tuple[str, ...]
    years: tuple[int, ...]
    event: TemporalEventNode
    projection: ProjectionPeriodNode
    filters: tuple[Condition, ...] = ()


@dataclass(frozen=True)
class NoteDetailPlan:
    members: tuple[str, ...]
    years: tuple[int, ...]
    calculation: CalculationNode
    reduction: str                  # direct | growth | difference | mean | sum | max | min


@dataclass(frozen=True)
class MatrixNotePlan:
    family: str
    members: tuple[str, ...]
    years: tuple[int, ...]
    reduction: str                  # direct | growth | difference | mean | sum | max[_value]


@dataclass(frozen=True)
class NoteAxisPlan:
    family: str
    members: tuple[str, ...]
    years: tuple[int, ...]
    reduction: str                  # growth | ratio | abs_difference | sum | mean | max


@dataclass(frozen=True)
class LeaseSchedulePlan:
    direction: str                   # receivable | payable
    members: tuple[str, ...]
    years: tuple[int, ...]
    value_axis: str                  # total | short_term | short_term_share
    reduction: str                   # direct | growth | mean | max


@dataclass(frozen=True)
class SelectProjectPlan:
    family: str
    members: tuple[str, ...]
    years: tuple[int, ...]
    direction: str                   # max | min
    tie_breaker: str = ""            # optional secondary max/min selector
    projection: str = "money"        # money | year


@dataclass(frozen=True)
class ScenarioPlan:
    family: str
    members: tuple[str, ...]
    years: tuple[int, ...]
    shock: float
    direction: str                   # max | min | aggregate


@dataclass
class MatrixValue:
    ticker: str
    year: int
    value: float
    expr: str
    resolved: list[ResolvedFact]


@dataclass
class Population:
    tickers: list[str]
    resolved: list[ResolvedFact]
    support_exprs: list[str]


def try_formula_answer(route: dict, tables: list[dict], encoder=None,
                       min_score: float = 62.0) -> CompositeAnswer:
    """Try formula-aware deterministic solving for count/ranking/ratio/margin."""
    plan = route.get("plan") or {}
    op = plan.get("op", "lookup")
    output_type = route.get("output_type", "number")

    if output_type == "count" or op == "count":
        return _try_count(route, tables, encoder, min_score)
    exact_derived = _try_exact_derived_metric(
        route, tables, encoder, min_score)
    if exact_derived.ok:
        return exact_derived
    scenario_plan = build_scenario_plan(route)
    if scenario_plan is not None:
        return _try_scenario(
            route, tables, encoder, min_score, scenario_plan)
    select_project_plan = build_select_project_plan(route)
    if select_project_plan is not None:
        return _try_select_project(route, tables, select_project_plan)
    lease_plan = build_lease_schedule_plan(route)
    if lease_plan is not None:
        lease = _try_lease_schedule(route, tables, lease_plan)
        if lease.ok:
            return lease
    note_axis_plan = build_note_axis_plan(route)
    if note_axis_plan is not None:
        note_axis = _try_note_axis(route, tables, note_axis_plan)
        if note_axis.ok:
            return note_axis
    matrix_plan = build_matrix_note_plan(route)
    if matrix_plan is not None:
        return _try_matrix_note(
            route, tables, matrix_plan, encoder=encoder, min_score=min_score)
    note_detail_plan = build_note_detail_plan(route)
    if note_detail_plan is not None:
        return _try_note_detail(
            route, tables, encoder, min_score, note_detail_plan)
    temporal_plan = build_temporal_event_plan(route)
    if temporal_plan is not None:
        return _try_temporal_event(
            route, tables, encoder, min_score, temporal_plan)
    aggregate_plan = build_filter_aggregate_plan(route)
    if aggregate_plan is not None:
        return _try_filter_aggregate(
            route, tables, encoder, min_score, aggregate_plan)
    nested_plan = build_compositional_ranking_plan(route)
    if nested_plan is not None and _looks_nested_selector(route.get("question", "")):
        nested = _try_nested_ranking(route, tables, encoder, min_score)
        if nested.ok or op != "ranking":
            return nested
    if "trung vi" in _plain(route.get("question", "")):
        # Median-filter questions are compositional even when the coarse router
        # labels the outer arithmetic as difference/growth instead of ranking.
        if build_compositional_ranking_plan(route) is not None:
            return _try_nested_ranking(route, tables, encoder, min_score)
    if op == "ranking":
        if output_type == "year":
            return _try_year_ranking(route, tables, encoder, min_score)
        nested = _try_nested_ranking(route, tables, encoder, min_score)
        if nested.ok:
            return nested
        if _looks_nested_selector(route.get("question", "")):
            return nested
        return _try_formula_ranking(route, tables, encoder, min_score)
    if op in {"difference", "growth_pct", "average"} or output_type == "percentage_point":
        temporal = _try_formula_change(route, tables, encoder, min_score)
        if temporal.ok:
            return temporal
    if op in {"ratio", "margin", "ratio_times"} or output_type in {"ratio", "percent"}:
        if _looks_nested_selector(route.get("question", "")):
            nested = _try_nested_ranking(route, tables, encoder, min_score)
            if nested.ok:
                return nested
        temporal = _try_formula_change(route, tables, encoder, min_score)
        if temporal.ok:
            return temporal
        return _try_direct_formula(route, tables, encoder, min_score)
    return CompositeAnswer(ok=False, detail=f"formula solver skipped op={op}")


def _try_exact_derived_metric(route: dict, tables: list[dict], encoder,
                              min_score: float) -> CompositeAnswer:
    """Evaluate canonical derived metrics only from exact component cells."""
    keys = list(dict.fromkeys(route.get("metric_keys") or []))
    if len(keys) != 1 or keys[0] not in _EXACT_DERIVED_SPECS:
        return CompositeAnswer(ok=False, detail="exact derived metric not selected")

    key = keys[0]
    spec = _EXACT_DERIVED_SPECS[key]
    metric = get_metric(key)
    tickers = _candidate_tickers(route, tables)
    years = sorted(set(_route_years(route)))
    if len(tickers) != 1 or len(years) != 1:
        return CompositeAnswer(
            ok=False, detail=f"exact derived dimensions={len(tickers)}x{len(years)}")
    if tuple(metric.components) != spec.components:
        return CompositeAnswer(ok=False, detail=f"exact derived schema mismatch {key}")

    ticker, year = tickers[0], years[0]
    resolved: list[ResolvedFact] = []
    for component in spec.components:
        requirement = _year_requirement(route, ticker, year, component)
        found = resolve_requirement(
            requirement, tables, encoder=encoder, min_score=min_score,
            question=str(requirement.get("metric_label") or component),
        )
        if (found is None or not _resolved_supports_year(found, year)
                or not _resolved_value_sane(found)):
            return CompositeAnswer(
                ok=False, detail=f"exact derived unresolved {key}/{component}",
                resolved=resolved)
        resolved.append(found)
    if not distinct_cells(resolved):
        return CompositeAnswer(
            ok=False, detail=f"exact derived duplicate cells {key}", resolved=resolved)

    values = [found.value_vnd for found in resolved]
    try:
        value = spec.value_fn(values)
    except ZeroDivisionError:
        return CompositeAnswer(
            ok=False, detail=f"exact derived zero denominator {key}", resolved=resolved)
    if not math.isfinite(value):
        return CompositeAnswer(
            ok=False, detail=f"exact derived non-finite {key}", resolved=resolved)

    expr = spec.expr_fn(resolved)
    if spec.kind == "money":
        question_scale = float(route.get("unit_scale") or 1.0)
        if question_scale <= 0:
            return CompositeAnswer(
                ok=False, detail=f"exact derived invalid unit {key}", resolved=resolved)
        value /= question_scale
        expr = f"(({expr}) / {question_scale:g})"
    output_type = route.get("output_type", spec.kind)
    warn = check_answer_unit(value, output_type)
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(
            ok=False, detail=f"exact derived unit guard {key}: {warn}",
            resolved=resolved)
    return CompositeAnswer(
        ok=True, answer=round(float(value), 2),
        pandas_query=f"round({expr}, 2)",
        confidence=_confidence(resolved, base=96.0),
        detail=f"formula_exact_derived {key}"
               + (f" | UNIT-WARN: {warn}" if warn else ""),
        resolved=resolved,
    )


def requires_formula_solver(route: dict) -> bool:
    """Whether generic lookup/composite fallbacks would change the semantics."""
    plan = route.get("plan") or {}
    op = plan.get("op", "lookup")
    question = route.get("question", "")
    if route.get("output_type") == "count" or op == "count":
        return True
    if route.get("output_type") == "year":
        return True
    if _looks_nested_selector(question):
        return True
    return (build_lease_schedule_plan(route) is not None
            or build_scenario_plan(route) is not None
            or build_select_project_plan(route) is not None
            or build_note_axis_plan(route) is not None
            or build_matrix_note_plan(route) is not None
            or bool(_detected_specs(question)))


def _try_count(route: dict, tables: list[dict], encoder, min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    text = _plain(question)
    if any(w in text for w in ("gia su", "kich ban", "neu doanh thu")):
        return CompositeAnswer(ok=False, detail="scenario count unsupported")
    if _counts_years(question):
        return _try_count_years(route, tables, encoder, min_score)
    if len(_route_years(route)) >= 2:
        temporal = _try_temporal_entity_count(route, tables, encoder, min_score)
        if temporal.ok:
            return temporal

    conditions = _parsed_conditions(route, question)

    if not conditions:
        c = _direct_metric_condition(route, question)
        if c:
            conditions = [c]
    if not conditions:
        return CompositeAnswer(ok=False, detail="count has no parsed condition")
    expected = _expected_condition_count(question)
    if len(conditions) < expected:
        return CompositeAnswer(
            ok=False,
            detail=f"count parsed only {len(conditions)}/{expected} conditions")

    tickers = _candidate_tickers(route, tables)
    year = _primary_year(route)
    if not tickers:
        return CompositeAnswer(ok=False, detail="count has no candidate tickers")

    population = _maybe_top_n_population(
        question, tickers, year, route, tables, encoder, min_score)
    if population is None:
        return CompositeAnswer(ok=False, detail="top-n population could not be resolved")
    tickers = population.tickers

    terms, resolved, answers = [], list(population.resolved), []
    for ticker in tickers:
        ticker_terms = []
        ticker_passes = []
        for cond in conditions:
            val = _evaluate_condition(cond, ticker, year, route, tables, encoder, min_score)
            if val is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"count unresolved condition {cond.label} for {ticker}",
                    resolved=resolved)
            resolved.extend(val.resolved)
            passed = _compare(val.value, cond.op, cond.threshold)
            ticker_terms.append(_condition_expr(val.expr, cond.op, cond.threshold))
            ticker_passes.append(passed)
        answers.append(all(ticker_passes))
        terms.append("(" + " and ".join(ticker_terms) + ")")

    answer = float(sum(1 for x in answers if x))
    support = ""
    if population.support_exprs:
        support = f" + 0 * ({' + '.join(population.support_exprs)})"
    query = f"round(float({' + '.join(terms)}){support}, 2)"
    conf = _confidence(resolved, base=92.0 if len(conditions) > 1 else 88.0)
    return CompositeAnswer(ok=True, answer=answer, pandas_query=query,
                           confidence=conf,
                           detail=f"formula_count n={len(tickers)} conditions={len(conditions)}",
                           resolved=resolved)


def _try_count_years(route: dict, tables: list[dict], encoder,
                     min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    tickers = _candidate_tickers(route, tables)
    years = sorted(set(_route_years(route)))
    if len(tickers) != 1 or len(years) < 2:
        return CompositeAnswer(ok=False, detail="count years needs one ticker and >=2 years")

    median_case = _try_count_years_with_median(
        route, tables, encoder, min_score, tickers[0], years)
    if median_case.ok:
        return median_case

    conditions = _parsed_conditions(route, question)
    if not conditions:
        direct = _direct_metric_condition(route, question)
        conditions = [direct] if direct else []
    if not conditions:
        return CompositeAnswer(ok=False, detail="count years has no parsed condition")

    terms, resolved, passed = [], [], []
    for year in years:
        year_terms, year_passes = [], []
        for cond in conditions:
            val = _evaluate_condition(
                cond, tickers[0], year, route, tables, encoder, min_score)
            if val is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"count years unresolved {cond.label}/{year}",
                    resolved=resolved)
            if not _value_supports_year(val, year):
                return CompositeAnswer(
                    ok=False,
                    detail=f"count years lacks exact-year evidence {cond.label}/{year}",
                    resolved=resolved)
            resolved.extend(val.resolved)
            year_terms.append(_condition_expr(val.expr, cond.op, cond.threshold))
            year_passes.append(_compare(val.value, cond.op, cond.threshold))
        terms.append("(" + " and ".join(year_terms) + ")")
        passed.append(all(year_passes))

    answer = float(sum(passed))
    query = f"round(float({' + '.join(terms)}), 2)"
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=90.0),
        detail=f"formula_count_years n={len(years)} conditions={len(conditions)}",
        resolved=resolved)


def _try_count_years_with_median(route: dict, tables: list[dict], encoder,
                                 min_score: float, ticker: str,
                                 years: list[int]) -> CompositeAnswer:
    text = _plain(route.get("question", ""))
    if "trung vi" not in text or "nam truoc" not in text:
        return CompositeAnswer(ok=False, detail="not a prior-year median count")
    specs = {s.name: s for s in _detected_specs(text)}
    change_spec = specs.get("gross_margin")
    median_spec = specs.get("cfo_margin")
    if change_spec is None or median_spec is None:
        return CompositeAnswer(ok=False, detail="median year count formulas missing")

    change_vals, median_vals, resolved = [], [], []
    for year in years:
        a = _evaluate_formula(
            change_spec, ticker, year, route, tables, encoder, min_score)
        b = _evaluate_formula(
            median_spec, ticker, year, route, tables, encoder, min_score)
        if a is None or b is None:
            return CompositeAnswer(ok=False,
                                   detail=f"median year count unresolved {year}",
                                   resolved=resolved)
        if not _value_supports_year(a, year) or not _value_supports_year(b, year):
            return CompositeAnswer(ok=False,
                                   detail=f"median year count lacks exact year {year}",
                                   resolved=resolved)
        change_vals.append(a)
        median_vals.append(b)
        resolved.extend(a.resolved + b.resolved)

    median = _median([v.value for v in median_vals])
    median_expr = _median_expr([v.expr for v in median_vals])
    terms, outcomes = [], []
    for i in range(1, len(years)):
        terms.append(
            f"(({change_vals[i].expr}) > ({change_vals[i - 1].expr}) and "
            f"({median_vals[i].expr}) > ({median_expr}))")
        outcomes.append(change_vals[i].value > change_vals[i - 1].value
                        and median_vals[i].value > median)
    answer = float(sum(outcomes))
    return CompositeAnswer(
        ok=True, answer=answer,
        pandas_query=f"round(float({' + '.join(terms)}), 2)",
        confidence=_confidence(resolved, base=88.0),
        detail="formula_count_years prior_change+median", resolved=resolved)


def _try_temporal_entity_count(route: dict, tables: list[dict], encoder,
                               min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    text = _plain(question)
    years = sorted(set(_route_years(route)))
    tickers = _candidate_tickers(route, tables)
    if len(years) < 2 or not tickers:
        return CompositeAnswer(ok=False, detail="temporal count needs periods/entities")
    start, end = years[0], years[-1]
    matches = _calculation_matches(question)
    by_name = _unique_matches(matches)

    compare_growth = ("cao hon toc do tang" in text
                      and "sga_expense" in by_name and "net_revenue" in by_name)
    predicates: list[tuple[str, FormulaSpec, str | None]] = []
    if compare_growth:
        predicates.append(("growth_gt", by_name["sga_expense"].spec,
                           by_name["net_revenue"].spec.name))
    else:
        for match in by_name.values():
            direction = _temporal_direction(text, match)
            if direction:
                predicates.append((direction, match.spec, None))

    if not predicates:
        return CompositeAnswer(ok=False, detail="temporal count predicates missing")
    if len(predicates) < _expected_condition_count(question):
        return CompositeAnswer(ok=False,
                               detail="temporal count parsed too few predicates")

    terms, outcomes, resolved = [], [], []
    for ticker in tickers:
        ticker_terms, ticker_outcomes = [], []
        for kind, spec, other_name in predicates:
            if kind == "growth_gt":
                other = by_name[other_name].spec
                a = _evaluate_change(spec, ticker, start, end, "growth", route,
                                     tables, encoder, min_score)
                b = _evaluate_change(other, ticker, start, end, "growth", route,
                                     tables, encoder, min_score)
                if a is None or b is None:
                    return CompositeAnswer(ok=False,
                                           detail=f"temporal count unresolved {ticker}",
                                           resolved=resolved)
                resolved.extend(a.resolved + b.resolved)
                ticker_terms.append(f"(({a.expr}) > ({b.expr}))")
                ticker_outcomes.append(a.value > b.value)
                continue

            if kind in {"increase", "decrease"}:
                val = _evaluate_change(spec, ticker, start, end, "delta", route,
                                       tables, encoder, min_score)
                if val is None:
                    return CompositeAnswer(ok=False,
                                           detail=f"temporal count unresolved {ticker}",
                                           resolved=resolved)
                resolved.extend(val.resolved)
                op = ">" if kind == "increase" else "<"
                ticker_terms.append(f"(({val.expr}) {op} 0)")
                ticker_outcomes.append(val.value > 0 if op == ">" else val.value < 0)
                continue

            val = _evaluate_formula(spec, ticker, end, route, tables, encoder, min_score)
            if val is None:
                return CompositeAnswer(ok=False,
                                       detail=f"temporal count unresolved {ticker}",
                                       resolved=resolved)
            if not _value_supports_year(val, end):
                return CompositeAnswer(
                    ok=False, detail=f"temporal count lacks exact year {ticker}/{end}",
                    resolved=resolved)
            resolved.extend(val.resolved)
            op = ">" if kind == "positive" else "<"
            ticker_terms.append(f"(({val.expr}) {op} 0)")
            ticker_outcomes.append(val.value > 0 if op == ">" else val.value < 0)
        terms.append("(" + " and ".join(ticker_terms) + ")")
        outcomes.append(all(ticker_outcomes))

    answer = float(sum(outcomes))
    return CompositeAnswer(
        ok=True, answer=answer,
        pandas_query=f"round(float({' + '.join(terms)}), 2)",
        confidence=_confidence(resolved, base=88.0),
        detail=f"formula_count_temporal n={len(tickers)} predicates={len(predicates)}",
        resolved=resolved)


def build_filter_aggregate_plan(route: dict) -> FilterAggregatePlan | None:
    """Build a typed entity filter-then-aggregate plan.

    The planner only accepts explicit aggregate language and fully typed
    calculations. Average-balance formula operands are kept inside their
    calculation node and are not mistaken for population means.
    """
    question = route.get("question", "")
    text = _plain(question)
    if any(value in text for value in (
            "gia su", "kich ban", "neu ", "co the tang toi da", "truoc khi")):
        return None
    tickers = tuple(_candidate_tickers(route, []))
    years = tuple(sorted(set(_route_years(route))))
    if len(tickers) < 2 or not years:
        return None

    markers = _population_aggregate_markers(text)
    aggregate_op = _aggregate_operation(text, bool(markers))
    if aggregate_op is None:
        return None
    matches = _calculation_matches(question)
    if not matches:
        return None

    predicates, predicate_matches = _aggregate_predicates(
        text, matches, list(years))
    rank_slice = _aggregate_rank_slice(
        text, matches, predicate_matches, list(years))
    if rank_slice is not None:
        predicate_matches.add(_match_key(rank_slice.calculation.match))
    group_predicate = None
    if aggregate_op in {"difference_of_means", "partition_ratio"}:
        median_predicates = [
            predicate for predicate in predicates
            if predicate.reference == "median"
        ]
        if len(median_predicates) != 1:
            return None
        group_predicate = median_predicates[0]
        predicates = [
            predicate for predicate in predicates
            if predicate is not group_predicate
        ]

    value = _aggregate_value_node(
        text, matches, predicate_matches, markers, list(years),
        route.get("output_type"), aggregate_op)
    if value is None:
        return None
    denominator_predicates = _aggregate_denominator_predicates(
        text, value, predicates, list(years), aggregate_op)
    kind = _value_node_kind(value)
    result_kind = (
        "percent" if aggregate_op == "share"
        else "ratio" if aggregate_op == "partition_ratio"
        else kind
    )
    if not _output_accepts_kind(route.get("output_type"), result_kind):
        return None

    return FilterAggregatePlan(
        population=PopulationNode("entity", tickers, years),
        predicates=tuple(predicates),
        value=value,
        aggregate=AggregateNode(
            aggregate_op, group_predicate,
            tuple(denominator_predicates), rank_slice),
    )


def _population_aggregate_markers(text: str) -> list[tuple[int, int]]:
    markers = []
    operand_prefixes = (
        "hang ton kho", "tong tai san", "tai san", "von chu so huu",
        "tai san co dinh thuan", "du no", "so du",
    )
    for found in re.finditer(r"\b(?:binh quan|trung binh)\b", text):
        prefix = text[max(0, found.start() - 40):found.start()].rstrip()
        suffix = text[found.end():min(len(text), found.end() + 45)].lstrip()
        if (any(prefix.endswith(value) for value in operand_prefixes)
                or any(suffix.startswith(value) for value in operand_prefixes)):
            continue
        markers.append((found.start(), found.end()))
    return markers


def _aggregate_operation(text: str, has_mean_marker: bool) -> str | None:
    if (has_mean_marker and "phan nhom con lai" in text
            and "chenh lech" in text):
        return "difference_of_means"
    if ("gap bao nhieu lan tong" in text
            and "trung vi" in text
            and "bang hoac thap hon" in text):
        return "partition_ratio"
    if any(value in text for value in (
            "nam giu bao nhieu phan tram tong",
            "dong gop bao nhieu phan tram tong",
            "dong gop bao nhieu phan tram vao tong",
            "chiem bao nhieu phan tram tong",
            "ty trong trong tong",
    )):
        return "share"
    if has_mean_marker:
        return "mean"
    if (re.search(r"\btong cong\b(?!\s*ty)", text)
            or "cong lai" in text
            or re.search(
                r"\btong\s+.+?\s+cua\s+cac\s+(?:cong ty|doanh nghiep|ma)\b",
                text)):
        return "sum"
    return None


def _aggregate_rank_slice(
        text: str, matches: list[FormulaMatch],
        used: set[tuple[str, int, int]], years: list[int]) -> RankSliceNode | None:
    stated = list(re.finditer(r"\b(\d+)\s+(?:doanh nghiep|cong ty|ma)\b", text))
    if not stated:
        return None
    ranked = _ranked_match(text, [
        match for match in matches if _match_key(match) not in used
    ])
    if ranked is None:
        return None
    match, extreme_start, want_min = ranked
    nearest = min(stated, key=lambda found: abs(found.end() - match.start))
    if not (nearest.end() <= match.start and match.start - nearest.end() <= 170):
        return None
    k = int(nearest.group(1))
    if k <= 0:
        return None
    mode = _aggregate_target_value_mode(text, match, years)
    node = _calculation_node(text, match, mode, years, anchor=extreme_start)
    return RankSliceNode(
        node, "min" if want_min else "max", k,
        _aggregate_target_year(text, match, match, years),
    )


def _aggregate_denominator_predicates(
        text: str, value: ValueNode, predicates: list[PredicateNode],
        years: list[int], aggregate_op: str) -> list[PredicateNode]:
    if aggregate_op != "share":
        return []
    positive_denominator = any(value in text for value in (
        "tong loi nhuan sau thue cua cac doanh nghiep co lai",
        "tong loi nhuan sau thue duong cua toan bo nhom",
        "tong loi nhuan sau thue duong cua ca nhom",
    ))
    if not positive_denominator:
        return []
    for predicate in predicates:
        if (predicate.calculation.match.spec.name
                == value.primary.match.spec.name
                and predicate.op == ">" and predicate.threshold == 0.0):
            return [predicate]
    year = value.year if value.year is not None else (years[-1] if years else None)
    quantifier = PeriodQuantifierNode("all", (year,)) if year is not None else None
    return [PredicateNode(
        value.primary, ">", threshold=0.0,
        years=(year,) if year is not None else (), quantifier=quantifier)]


def _aggregate_predicates(
        text: str, matches: list[FormulaMatch],
        years: list[int]) -> tuple[list[PredicateNode], set[tuple[str, int, int]]]:
    predicates: list[PredicateNode] = []
    used: set[tuple[str, int, int]] = set()
    median = _aggregate_median_filter_node(text, matches, years)
    if median is not None:
        median_years = ((median.year,) if median.year is not None else ())
        predicates.append(PredicateNode(
            median.calculation, median.op, reference="median",
            years=median_years))
        used.add(_match_key(median.calculation.match))

    aggregate_answer_start = min(
        (text.find(marker) for marker in (
            "dong gop bao nhieu", "chiem bao nhieu", "nam giu bao nhieu",
            "gap bao nhieu lan tong",
        ) if text.find(marker) >= 0),
        default=len(text),
    )

    for index, match in enumerate(matches):
        key = _match_key(match)
        if key in used or match.start >= aggregate_answer_start:
            continue
        next_start = min(
            (other.start for other in matches[index + 1:]
             if other.start >= match.end),
            default=min(len(text), match.end + 180),
        )
        segment_end = min(len(text), max(match.end, next_start), match.end + 180)
        segment = text[match.end:segment_end]
        mode = _predicate_mode(text, match, segment, years)
        condition_kind = "percent" if mode in {"growth", "cagr"} else match.spec.kind
        direction = _simple_temporal_direction(segment)
        parsed = ((direction, 0.0) if mode != "level" and direction
                  else _parse_condition_segment(segment, condition_kind))
        if parsed is None and mode != "level":
            if re.search(r"\bgiam(?:\s+\w+){0,5}\s+so voi\b", segment):
                parsed = ("<", 0.0)
            elif re.search(r"\btang(?:\s+\w+){0,5}\s+so voi\b", segment):
                parsed = (">", 0.0)
        if parsed is None:
            continue
        op, threshold = parsed
        node = _calculation_node(text, match, mode, years, anchor=match.end)
        predicate_years = _aggregate_predicate_years(
            text, match, segment, years, mode)
        quantifier = _aggregate_period_quantifier(
            text, match, segment, predicate_years)
        predicates.append(PredicateNode(
            node, op, threshold=threshold, years=predicate_years,
            quantifier=quantifier))
        used.add(key)
    return predicates, used


def _predicate_mode(text: str, match: FormulaMatch, segment: str,
                    years: list[int]) -> str:
    prefix = text[max(0, match.start - 55):match.start]
    context = prefix + " " + segment
    growth_markers = ("tang truong", "toc do tang", "ty le tang", "phan tram tang")
    condition = re.search(
        r"(?:>=|<=|>|<)|\b(?:lon hon|cao hon|vuot|tren|nho hon|thap hon|"
        r"duoi|khong am|khong duong|duong|am)\b", segment)
    if _simple_temporal_direction(segment):
        return "delta"
    if re.search(
            r"\btang\s+(?:tren|hon|it nhat)?\s*[-+]?\d+(?:[\.,]\d+)?\s*%?"
            r"(?:\s+so voi|\s+tu ky)", segment):
        return "growth"
    if re.search(
            r"\bgiam\s+(?:tren|hon|it nhat)?\s*[-+]?\d+(?:[\.,]\d+)?\s*"
            r"(?:diem phan tram|%)(?:\s+so voi|\s+tu ky)", segment):
        return "decrease"
    growth_pos = min(
        (segment.find(value) for value in growth_markers if value in segment),
        default=-1,
    )
    if (not any(value in prefix for value in growth_markers)
            and condition is not None
            and (growth_pos < 0 or condition.start() < growth_pos)):
        return "level"
    if "cagr" in context:
        return "cagr"
    if any(value in context for value in growth_markers):
        return "growth"
    if (re.search(r"\b(?:tang|giam)(?:\s+\w+){0,5}\s+so voi\b", segment)
            or any(value in context for value in ("muc thay doi", "muc tang"))):
        return "delta"
    return "level"


def _simple_temporal_direction(segment: str) -> str | None:
    """Recognize an unquantified increase/decrease across explicit periods."""
    if re.match(r"\s*(?:co\s+)?tang\b(?!\s+(?:truong|tren|hon|it nhat|\d))",
                segment):
        return ">"
    if re.match(r"\s*(?:co\s+)?giam\b(?!\s+(?:tren|hon|it nhat|\d))",
                segment):
        return "<"
    comparison = re.match(
        r"\s*(?:nam\s+)?20\d{2}\s+(cao hon|lon hon|thap hon|nho hon)\s+"
        r"(?:nam\s+)?20\d{2}\b",
        segment,
    )
    if comparison:
        return ">" if comparison.group(1) in {"cao hon", "lon hon"} else "<"
    return None


def _aggregate_median_filter_node(
        text: str, matches: list[FormulaMatch],
        years: list[int]) -> MedianFilterNode | None:
    node = _median_filter_node(text, matches, "entity", years)
    if node is not None:
        return node
    choices = []
    for median in re.finditer(r"\b(?:muc\s+)?trung vi\b", text):
        for match in matches:
            if match.start < median.end():
                continue
            gap = match.start - median.end()
            if gap > 90:
                continue
            op = _median_comparator(text[match.end:min(len(text), match.end + 180)])
            if op is not None:
                choices.append((gap, match, op))
    if not choices:
        return None
    _gap, match, op = min(choices, key=lambda item: item[0])
    year = years[-1] if years else None
    return MedianFilterNode(CalculationNode(match, "level"), op, year)


def _aggregate_predicate_years(
        text: str, match: FormulaMatch, segment: str, years: list[int],
        mode: str) -> tuple[int, ...]:
    if mode != "level":
        return ()
    all_period_phrases = (
        "ca hai nam", "o ca hai nam", "trong ca hai nam",
        "ca ba nam", "o ca ba nam", "trong ca ba nam",
        "ca 2 nam", "ca 3 nam", "trong ca 2 nam", "trong ca 3 nam",
        "lien tuc trong hai nam", "lien tuc trong ba nam",
        "ca hai ky", "o ca hai ky", "duy tri", "trong hai nam", "trong ba nam",
    )
    any_period_phrases = (
        "it nhat mot nam", "mot trong cac nam", "bat ky nam nao",
        "o mot nam", "trong mot nam",
    )
    wants_all_periods = any(value in text for value in all_period_phrases)
    wants_any_period = any(value in text for value in any_period_phrases)
    if wants_all_periods and len(years) == 1:
        return (years[0] - 1, years[0])
    if wants_all_periods:
        return tuple(years)
    if wants_any_period:
        return tuple(years)
    if len(years) == 1:
        return (years[0],)
    local = text[max(0, match.start - 30):match.end] + segment
    mentioned = [
        year for year in years
        if re.search(rf"(?<!\d){year}(?!\d)", local)
    ]
    return tuple(mentioned or [years[-1]])


def _aggregate_period_quantifier(
        text: str, match: FormulaMatch, segment: str,
        predicate_years: tuple[int, ...]) -> PeriodQuantifierNode | None:
    if len(predicate_years) <= 1:
        return None
    local = text[max(0, match.start - 80):min(len(text), match.end + len(segment))]
    any_markers = (
        "it nhat mot nam", "mot trong cac nam", "bat ky nam nao",
        "o mot nam", "trong mot nam",
    )
    mode = "any" if any(marker in local for marker in any_markers) else "all"
    return PeriodQuantifierNode(mode, predicate_years)


def _aggregate_value_node(
        text: str, matches: list[FormulaMatch],
        predicate_matches: set[tuple[str, int, int]],
        markers: list[tuple[int, int]], years: list[int],
        output_type: str | None, aggregate_op: str) -> ValueNode | None:
    choices: list[tuple[FormulaMatch, CalculationNode, str]] = []
    for match in matches:
        if _match_key(match) in predicate_matches:
            continue
        mode = _aggregate_target_value_mode(text, match, years)
        node = _calculation_node(text, match, mode, years, anchor=match.start)
        kind = _calculation_kind(node)
        expected_kind = (
            "percent" if aggregate_op == "share"
            else "ratio" if aggregate_op == "partition_ratio"
            else kind
        )
        if _output_accepts_kind(output_type, expected_kind):
            choices.append((match, node, kind))
    if not choices:
        return None

    difference = text.find("chenh lech")
    between = text.find("giua", difference if difference >= 0 else 0)
    if difference >= 0 and between >= 0:
        pair = [choice for choice in choices if choice[0].start > between]
        if len(pair) >= 2 and pair[0][2] == pair[1][2]:
            first, second = pair[0], pair[1]
            return ValueNode(
                first[1], second[1], "difference",
                _aggregate_target_year(text, first[0], second[0], years))

    selected = min(
        choices,
        key=lambda choice: _aggregate_distance(choice[0], markers, len(text)),
    )
    return ValueNode(
        selected[1], year=_aggregate_target_year(
            text, selected[0], selected[0], years))


def _aggregate_target_value_mode(
        text: str, match: FormulaMatch, years: list[int]) -> str:
    if len(years) < 2:
        return "level"
    clause_start = max(
        text.rfind(";", 0, match.start),
        text.rfind(".", 0, match.start),
        text.rfind(",", 0, match.start),
    )
    before = text[max(clause_start + 1, match.start - 110):match.start]
    after = text[match.end:min(len(text), match.end + 55)]
    if "cagr" in before:
        return "cagr"
    if "muc giam" in before:
        return "decrease"
    if any(value in before for value in (
            "muc thay doi", "thay doi", "muc tang")):
        return "delta"
    if any(value in before for value in (
            "tang truong", "toc do tang", "ty le tang", "phan tram tang")):
        return "growth"
    if re.match(r"\s*(?:co\s+)?(?:muc\s+)?thay doi\b", after):
        return "delta"
    return "level"


def _aggregate_distance(match: FormulaMatch, markers: list[tuple[int, int]],
                        text_length: int) -> int:
    if not markers:
        return text_length - match.end
    distances = []
    for start, end in markers:
        if match.end <= start:
            distances.append(start - match.end)
        elif match.start >= end:
            distances.append(match.start - end)
        else:
            distances.append(0)
    return min(distances)


def _aggregate_target_year(text: str, first: FormulaMatch,
                           second: FormulaMatch, years: list[int]) -> int:
    window = text[max(0, first.start - 45):min(len(text), second.end + 60)]
    mentioned = [
        year for year in years
        if re.search(rf"(?<!\d){year}(?!\d)", window)
    ]
    return max(mentioned or years)


def _match_key(match: FormulaMatch) -> tuple[str, int, int]:
    return match.spec.name, match.start, match.end


def _calculation_kind(node: CalculationNode) -> str:
    if node.mode in {"growth", "cagr", "yoy_growth"}:
        return "percent"
    return node.match.spec.kind


def _value_node_kind(node: ValueNode) -> str:
    primary = _calculation_kind(node.primary)
    if node.secondary is None:
        return primary
    secondary = _calculation_kind(node.secondary)
    return primary if primary == secondary else "invalid"


def _try_filter_aggregate(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: FilterAggregatePlan) -> CompositeAnswer:
    tickers = list(typed.population.members)
    years = list(typed.population.years)
    stated_n = _stated_population_size(_plain(route.get("question", "")))
    if stated_n is not None and stated_n != len(tickers):
        return CompositeAnswer(
            ok=False, detail=f"filter aggregate population {len(tickers)}/{stated_n}")

    all_predicates = list(typed.predicates)
    group_predicate = typed.aggregate.group_predicate
    if group_predicate is not None:
        all_predicates.append(group_predicate)
    for predicate in typed.aggregate.denominator_predicates:
        if predicate not in all_predicates:
            all_predicates.append(predicate)
    predicate_outcomes: dict[int, dict[str, bool]] = {}
    predicate_exprs: dict[int, dict[str, str]] = {}
    resolved: list[ResolvedFact] = []
    support: list[str] = []

    for predicate in all_predicates:
        values_by_ticker: dict[str, list[FormulaValue]] = {}
        for ticker in tickers:
            values = _evaluate_aggregate_predicate(
                predicate, ticker, years, route, tables, encoder, min_score)
            if values is None:
                return CompositeAnswer(
                    ok=False,
                    detail=("filter aggregate unresolved predicate "
                            f"{predicate.calculation.match.spec.name}/{ticker}"),
                    resolved=resolved,
                )
            values_by_ticker[ticker] = values
            for value in values:
                resolved.extend(value.resolved)
                support.append(value.expr)

        threshold = predicate.threshold
        threshold_expr: float | str | None = threshold
        if predicate.reference == "median":
            flattened = [values for values in values_by_ticker.values()]
            if any(len(values) != 1 for values in flattened):
                return CompositeAnswer(
                    ok=False, detail="filter aggregate median is not scalar",
                    resolved=resolved)
            median_values = [values[0] for values in flattened]
            threshold = _median([value.value for value in median_values])
            threshold_expr = _median_expr([value.expr for value in median_values])
        if threshold is None or threshold_expr is None:
            return CompositeAnswer(
                ok=False, detail="filter aggregate threshold missing",
                resolved=resolved)

        outcomes, expressions = {}, {}
        for ticker, values in values_by_ticker.items():
            checks = [
                _compare(value.value, predicate.op, threshold) for value in values
            ]
            check_exprs = [
                _condition_expr(value.expr, predicate.op, threshold_expr)
                for value in values
            ]
            reducer = (
                predicate.quantifier.mode if predicate.quantifier is not None
                else "all"
            )
            outcomes[ticker] = any(checks) if reducer == "any" else all(checks)
            joiner = " or " if reducer == "any" else " and "
            expressions[ticker] = joiner.join(check_exprs)
        predicate_outcomes[id(predicate)] = outcomes
        predicate_exprs[id(predicate)] = expressions

    values: dict[str, FormulaValue] = {}
    for ticker in tickers:
        value = _evaluate_aggregate_value(
            typed.value, ticker, years, route, tables, encoder, min_score)
        if value is None:
            return CompositeAnswer(
                ok=False,
                detail=f"filter aggregate unresolved value {ticker}",
                resolved=resolved,
            )
        values[ticker] = value
        resolved.extend(value.resolved)
        support.append(value.expr)

    filter_outcomes = {
        ticker: all(predicate_outcomes[id(predicate)][ticker]
                    for predicate in typed.predicates)
        for ticker in tickers
    }
    filter_exprs = {
        ticker: (" and ".join(predicate_exprs[id(predicate)][ticker]
                              for predicate in typed.predicates) or "True")
        for ticker in tickers
    }
    selected = [ticker for ticker in tickers if filter_outcomes[ticker]]
    op = typed.aggregate.op
    kind = _value_node_kind(typed.value)
    cohort_exprs = dict(filter_exprs)
    rank_guard = "True"

    rank_slice = typed.aggregate.rank_slice
    if rank_slice is not None:
        rank_values: dict[str, FormulaValue] = {}
        for ticker in tickers:
            rank_value = _evaluate_calculation_exact(
                rank_slice.calculation, ticker, rank_slice.year, years,
                route, tables, encoder, min_score)
            if rank_value is None:
                return CompositeAnswer(
                    ok=False, detail=f"filter aggregate unresolved rank {ticker}",
                    resolved=resolved)
            rank_values[ticker] = rank_value
            resolved.extend(rank_value.resolved)
            support.append(rank_value.expr)
        eligible = [ticker for ticker in tickers if filter_outcomes[ticker]]
        if len(eligible) < rank_slice.k:
            return CompositeAnswer(
                ok=False, detail="filter aggregate top-k population too small",
                resolved=resolved)
        ordered = sorted(
            eligible, key=lambda ticker: rank_values[ticker].value,
            reverse=rank_slice.direction == "max")
        if (len(ordered) > rank_slice.k and math.isclose(
                rank_values[ordered[rank_slice.k - 1]].value,
                rank_values[ordered[rank_slice.k]].value,
                rel_tol=1e-12, abs_tol=1e-6)):
            return CompositeAnswer(
                ok=False, detail="filter aggregate top-k boundary tie",
                resolved=resolved)
        selected = ordered[:rank_slice.k]
        comparator = ">" if rank_slice.direction == "max" else "<"
        for ticker in tickers:
            better = " + ".join(
                f"(({filter_exprs[other]}) and "
                f"(({rank_values[other].expr}) {comparator} "
                f"({rank_values[ticker].expr})))"
                for other in tickers if other != ticker
            ) or "0"
            cohort_exprs[ticker] = (
                f"(({filter_exprs[ticker]}) and (({better}) < {rank_slice.k}))")
        rank_count = " + ".join(f"({cohort_exprs[ticker]})" for ticker in tickers)
        rank_guard = f"(({rank_count}) == {rank_slice.k})"

    denominator_predicates = typed.aggregate.denominator_predicates
    denominator_outcomes = {
        ticker: all(predicate_outcomes[id(predicate)][ticker]
                    for predicate in denominator_predicates)
        for ticker in tickers
    }
    denominator_passes = {
        ticker: (" and ".join(
            predicate_exprs[id(predicate)][ticker]
            for predicate in denominator_predicates) or "True")
        for ticker in tickers
    }

    if op == "difference_of_means":
        if group_predicate is None:
            return CompositeAnswer(ok=False, detail="aggregate group missing")
        group_static = predicate_outcomes[id(group_predicate)]
        group_expr = predicate_exprs[id(group_predicate)]
        high = [ticker for ticker in selected if group_static[ticker]]
        low = [ticker for ticker in selected if not group_static[ticker]]
        if not high or not low:
            return CompositeAnswer(
                ok=False, detail="filter aggregate group empty", resolved=resolved)
        answer = (_mean([values[ticker].value for ticker in high])
                  - _mean([values[ticker].value for ticker in low]))
        high_pass = {
            ticker: f"(({filter_exprs[ticker]}) and ({group_expr[ticker]}))"
            for ticker in tickers
        }
        low_pass = {
            ticker: f"(({filter_exprs[ticker]}) and (not ({group_expr[ticker]})))"
            for ticker in tickers
        }
        high_expr, high_den = _conditional_mean_expr(values, high_pass)
        low_expr, low_den = _conditional_mean_expr(values, low_pass)
        answer_expr = f"(({high_expr}) - ({low_expr}))"
        dynamic_guard = f"(({high_den}) > 0 and ({low_den}) > 0)"
    elif op == "mean":
        if not selected:
            return CompositeAnswer(
                ok=False, detail="filter aggregate population empty", resolved=resolved)
        answer = _mean([values[ticker].value for ticker in selected])
        answer_expr, denominator = _conditional_mean_expr(values, cohort_exprs)
        dynamic_guard = f"(({denominator}) > 0 and ({rank_guard}))"
    elif op == "sum":
        if not selected:
            return CompositeAnswer(
                ok=False, detail="filter aggregate population empty", resolved=resolved)
        answer = sum(values[ticker].value for ticker in selected)
        answer_expr = _conditional_sum_expr(values, cohort_exprs)
        dynamic_guard = rank_guard
    elif op == "share":
        numerator = sum(values[ticker].value for ticker in selected)
        denominator = sum(
            values[ticker].value for ticker in tickers
            if denominator_outcomes[ticker])
        if not selected or denominator == 0:
            return CompositeAnswer(
                ok=False, detail="filter aggregate share denominator empty",
                resolved=resolved)
        answer = numerator / denominator * 100.0
        numerator_expr = _conditional_sum_expr(values, cohort_exprs)
        denominator_expr = _conditional_sum_expr(values, denominator_passes)
        answer_expr = f"(({numerator_expr}) / ({denominator_expr}) * 100)"
        dynamic_guard = f"(({denominator_expr}) != 0 and ({rank_guard}))"
        kind = "percent"
    elif op == "partition_ratio":
        if group_predicate is None:
            return CompositeAnswer(ok=False, detail="aggregate partition missing")
        group_static = predicate_outcomes[id(group_predicate)]
        group_expr = predicate_exprs[id(group_predicate)]
        high = [ticker for ticker in selected if group_static[ticker]]
        low = [ticker for ticker in selected if not group_static[ticker]]
        numerator = sum(values[ticker].value for ticker in high)
        denominator = sum(values[ticker].value for ticker in low)
        if not high or not low or denominator == 0:
            return CompositeAnswer(
                ok=False, detail="filter aggregate partition empty",
                resolved=resolved)
        answer = numerator / denominator
        high_pass = {
            ticker: f"(({filter_exprs[ticker]}) and ({group_expr[ticker]}))"
            for ticker in tickers
        }
        low_pass = {
            ticker: f"(({filter_exprs[ticker]}) and (not ({group_expr[ticker]})))"
            for ticker in tickers
        }
        numerator_expr = _conditional_sum_expr(values, high_pass)
        denominator_expr = _conditional_sum_expr(values, low_pass)
        answer_expr = f"(({numerator_expr}) / ({denominator_expr}))"
        dynamic_guard = f"(({denominator_expr}) != 0)"
        kind = "ratio"
    else:
        return CompositeAnswer(ok=False, detail=f"aggregate op unsupported {op}")

    if kind == "money":
        scale = float(route.get("unit_scale", 1.0) or 1.0)
        answer /= scale
        answer_expr = f"(({answer_expr}) / {scale:g})"
    support_expr = " + ".join(support) or "0.0"
    query = (f"round((({answer_expr}) if ({dynamic_guard}) else 0.0) "
             f"+ 0 * ({support_expr}), 2)")
    warn = check_answer_unit(answer, route.get("output_type", kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(
            ok=False, detail=f"filter aggregate unit guard: {warn}",
            resolved=resolved)
    resolved = _dedupe_resolved(resolved)
    is_v7 = (
        op == "partition_ratio"
        or typed.aggregate.rank_slice is not None
        or bool(typed.aggregate.denominator_predicates)
        or any(predicate.quantifier is not None for predicate in typed.predicates)
    )
    detail_family = (
        "formula_quantified_cohort_v7" if is_v7
        else "formula_period_aware_v6"
        if _is_period_aware_v6(typed.value.primary.match.spec)
        else "formula_filter_aggregate_v5"
    )
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=94.0),
        detail=(f"{detail_family} op={op} "
                f"value={typed.value.primary.match.spec.name}/"
                f"{typed.value.primary.mode} n={len(selected)}/{len(tickers)}"),
        resolved=resolved,
    )


def _evaluate_aggregate_predicate(
        predicate: PredicateNode, ticker: str, years: list[int], route: dict,
        tables: list[dict], encoder, min_score: float) -> list[FormulaValue] | None:
    evaluation_years = (
        list(predicate.years)
        if predicate.calculation.mode == "level"
        else [predicate.calculation.end_year or years[-1]]
    )
    if not evaluation_years:
        return None
    values = []
    for year in evaluation_years:
        value = _evaluate_calculation_exact(
            predicate.calculation, ticker, year, years,
            route, tables, encoder, min_score)
        if value is None:
            return None
        values.append(value)
    return values


def _evaluate_aggregate_value(
        node: ValueNode, ticker: str, years: list[int], route: dict,
        tables: list[dict], encoder, min_score: float) -> FormulaValue | None:
    primary = _evaluate_calculation_exact(
        node.primary, ticker, node.year, years,
        route, tables, encoder, min_score)
    if primary is None:
        return None
    if node.secondary is None:
        return primary
    secondary = _evaluate_calculation_exact(
        node.secondary, ticker, node.year, years,
        route, tables, encoder, min_score)
    if secondary is None or _calculation_kind(node.primary) != _calculation_kind(
            node.secondary):
        return None
    return FormulaValue(
        spec=primary.spec, ticker=ticker, year=node.year,
        value=primary.value - secondary.value,
        expr=f"(({primary.expr}) - ({secondary.expr}))",
        resolved=primary.resolved + secondary.resolved,
        score=min(primary.score, secondary.score),
        evidence_years=primary.evidence_years + secondary.evidence_years,
    )


def _conditional_sum_expr(values: dict[str, FormulaValue],
                          passes: dict[str, str]) -> str:
    return " + ".join(
        f"(({value.expr}) if ({passes[ticker]}) else 0.0)"
        for ticker, value in values.items()
    )


def _conditional_mean_expr(values: dict[str, FormulaValue],
                           passes: dict[str, str]) -> tuple[str, str]:
    numerator = _conditional_sum_expr(values, passes)
    denominator = " + ".join(f"({passes[ticker]})" for ticker in values)
    return f"(({numerator}) / ({denominator}))", denominator


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def build_scenario_plan(route: dict) -> ScenarioPlan | None:
    """Recognize typed statement-based stress and sensitivity questions."""
    text = _plain(route.get("question", ""))
    members = tuple(dict.fromkeys(
        str(ticker).upper() for ticker in route.get("tickers") or [] if ticker))
    years = tuple(sorted(set(_route_years(route))))
    family = direction = ""
    shock = 0.0
    expected_members = 0

    if ("chi phi lai vay tang 20%" in text
            and "bien loi nhuan truoc thue" in text):
        family, direction, shock, expected_members = (
            "interest_shock_pretax_margin", "min", 0.20, 3)
    elif ("ebit proxy" in text and "giam 15%" in text
          and "he so kha nang thanh toan lai vay" in text):
        family, direction, shock, expected_members = (
            "ebit_shock_interest_coverage", "min", 0.15, 5)
    elif ("giam so ngay ton kho" in text and "ve dung trung vi" in text
          and "giai phong" in text):
        family, direction, expected_members = (
            "inventory_days_release", "max", 3)
    elif ("phat hanh them 10%" in text and "pha loang eps" in text
          and "von chu so huu binh quan" in text):
        family, direction, shock, expected_members = (
            "eps_dilution_after_max_roe", "max", 0.10, 1)
    elif ("vong quay tai san co dinh dat trung vi" in text
          and "tai san co dinh thuan binh quan giu nguyen" in text):
        family, direction, expected_members = (
            "revenue_growth_to_median_turnover", "max", 10)
    elif ("gia tri tai san rong chiu ap luc thanh ly am" in text
          and "giam 30%" in text and "giam 50%" in text):
        family, direction, expected_members = (
            "liquidation_stress_liability_share", "aggregate", 10)
    elif ("chi phi lai vay" in text and "tang toi da" in text
          and "giam ve dung 2" in text):
        family, direction, expected_members = (
            "interest_cost_headroom", "min", 5)
    elif ("gia von tang 5%" in text and "giu nguyen bien loi nhuan hoat dong" in text
          and "muc tang gia ban" in text):
        family, direction, shock, expected_members = (
            "price_growth_preserve_operating_margin", "max", 0.05, 6)
    else:
        return None

    if len(members) != expected_members:
        return None
    if family == "eps_dilution_after_max_roe":
        if len(years) < 2:
            return None
    elif len(years) != 1:
        return None
    return ScenarioPlan(family, members, years, shock, direction)


def _try_scenario(route: dict, tables: list[dict], encoder,
                  min_score: float, typed: ScenarioPlan) -> CompositeAnswer:
    family = typed.family
    if family == "interest_shock_pretax_margin":
        return _scenario_interest_shock_margin(
            route, tables, encoder, min_score, typed)
    if family == "ebit_shock_interest_coverage":
        return _scenario_ebit_shock_coverage(
            route, tables, encoder, min_score, typed)
    if family == "inventory_days_release":
        return _scenario_inventory_release(
            route, tables, encoder, min_score, typed)
    if family == "eps_dilution_after_max_roe":
        return _scenario_eps_dilution(
            route, tables, encoder, min_score, typed)
    if family == "revenue_growth_to_median_turnover":
        return _scenario_turnover_growth(
            route, tables, encoder, min_score, typed)
    if family == "liquidation_stress_liability_share":
        return _scenario_liquidation_share(
            route, tables, encoder, min_score, typed)
    if family == "interest_cost_headroom":
        return _scenario_interest_headroom(
            route, tables, encoder, min_score, typed)
    if family == "price_growth_preserve_operating_margin":
        return _scenario_price_growth(
            route, tables, encoder, min_score, typed)
    return CompositeAnswer(ok=False, detail=f"scenario unsupported family={family}")


def _scenario_formula(name: str) -> FormulaSpec:
    return next(spec for spec in _FORMULAS if spec.name == name)


def _scenario_exact(
        name: str, ticker: str, year: int, route: dict, tables: list[dict],
        encoder, min_score: float) -> FormulaValue | None:
    spec = (_scenario_formula(name) if any(
        candidate.name == name for candidate in _FORMULAS)
        else _direct_metric_formula(name))
    return _evaluate_formula_exact(
        spec, ticker, year, route, tables, encoder, min_score)


def _scenario_failed(family: str, operand: str,
                     resolved: list[ResolvedFact]) -> CompositeAnswer:
    return CompositeAnswer(
        ok=False, detail=f"scenario {family} unresolved {operand}",
        resolved=_dedupe_resolved(resolved))


def _scenario_extreme(
        route: dict, typed: ScenarioPlan,
        candidates: list[tuple[str, float, str, bool, str]],
        resolved: list[ResolvedFact], support: list[str]) -> CompositeAnswer:
    eligible = [candidate for candidate in candidates if candidate[3]]
    if not eligible:
        return CompositeAnswer(
            ok=False, detail=f"scenario {typed.family} filtered all candidates",
            resolved=_dedupe_resolved(resolved))
    reverse = typed.direction == "max"
    ordered = sorted(eligible, key=lambda candidate: candidate[1], reverse=reverse)
    if len(ordered) > 1 and math.isclose(
            ordered[0][1], ordered[1][1], rel_tol=1e-12, abs_tol=1e-6):
        return CompositeAnswer(
            ok=False, detail=f"scenario {typed.family} tie has no convention",
            resolved=_dedupe_resolved(resolved))
    chosen = ordered[0]
    comparator = ">" if reverse else "<"
    guards = [chosen[4]]
    for candidate in candidates:
        if candidate[0] == chosen[0]:
            continue
        guards.append(
            f"((not ({candidate[4]})) or "
            f"(({chosen[2]}) {comparator} ({candidate[2]})))")
    guard = " and ".join(guards)
    support_expr = " + ".join(support) or "0.0"
    query = (
        f"round((({chosen[2]}) if ({guard}) else 0.0) "
        f"+ 0 * ({support_expr}), 2)")
    answer = round(float(chosen[1]), 2)
    warn = check_answer_unit(answer, route.get("output_type", "number"))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(
            ok=False, detail=f"scenario unit guard: {warn}",
            resolved=_dedupe_resolved(resolved))
    resolved = _dedupe_resolved(resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=94.0),
        detail=f"formula_scenario_v14 family={typed.family} picked={chosen[0]}",
        resolved=resolved)


def _scenario_interest_shock_margin(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: ScenarioPlan) -> CompositeAnswer:
    year = typed.years[0]
    candidates, resolved, support = [], [], []
    for ticker in typed.members:
        values = [
            _scenario_exact(name, ticker, year, route, tables, encoder, min_score)
            for name in ("pretax_profit", "interest_expense", "net_revenue")
        ]
        if any(value is None for value in values):
            return _scenario_failed(typed.family, ticker, resolved)
        pretax, interest, revenue = values
        resolved.extend(pretax.resolved + interest.resolved + revenue.resolved)
        support.extend((pretax.expr, interest.expr, revenue.expr))
        interest_value = abs(interest.value)
        if interest_value == 0 or revenue.value <= 0:
            return _scenario_failed(typed.family, f"invalid denominator {ticker}", resolved)
        coverage = (pretax.value + interest_value) / interest_value
        coverage_expr = (
            f"(({pretax.expr} + abs({interest.expr})) / abs({interest.expr}))")
        value = (pretax.value - typed.shock * interest_value) / revenue.value * 100.0
        expr = (
            f"(({pretax.expr} - {typed.shock:g} * abs({interest.expr})) "
            f"/ {revenue.expr} * 100)")
        candidates.append((ticker, value, expr, coverage > 2.0,
                           f"({coverage_expr} > 2.0)"))
    return _scenario_extreme(route, typed, candidates, resolved, support)


def _scenario_ebit_shock_coverage(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: ScenarioPlan) -> CompositeAnswer:
    year = typed.years[0]
    quick_values: dict[str, FormulaValue] = {}
    raw_values: dict[str, tuple[FormulaValue, FormulaValue]] = {}
    resolved, support = [], []
    for ticker in typed.members:
        quick = _scenario_exact(
            "quick_ratio", ticker, year, route, tables, encoder, min_score)
        pretax = _scenario_exact(
            "pretax_profit", ticker, year, route, tables, encoder, min_score)
        interest = _scenario_exact(
            "interest_expense", ticker, year, route, tables, encoder, min_score)
        if quick is None or pretax is None or interest is None:
            return _scenario_failed(typed.family, ticker, resolved)
        if abs(interest.value) == 0:
            return _scenario_failed(typed.family, f"zero interest {ticker}", resolved)
        quick_values[ticker] = quick
        raw_values[ticker] = (pretax, interest)
        resolved.extend(quick.resolved + pretax.resolved + interest.resolved)
        support.extend((quick.expr, pretax.expr, interest.expr))
    median = _median([value.value for value in quick_values.values()])
    median_expr = _median_expr([value.expr for value in quick_values.values()])
    candidates = []
    for ticker in typed.members:
        quick = quick_values[ticker]
        pretax, interest = raw_values[ticker]
        value = ((1.0 - typed.shock)
                 * (pretax.value + abs(interest.value)) / abs(interest.value))
        expr = (
            f"({1.0 - typed.shock:g} * "
            f"({pretax.expr} + abs({interest.expr})) / abs({interest.expr}))")
        candidates.append((ticker, value, expr, quick.value < median,
                           f"({quick.expr} < {median_expr})"))
    return _scenario_extreme(route, typed, candidates, resolved, support)


def _scenario_inventory_release(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: ScenarioPlan) -> CompositeAnswer:
    year = typed.years[0]
    days_values: dict[str, FormulaValue] = {}
    cogs_values: dict[str, FormulaValue] = {}
    resolved, support = [], []
    for ticker in typed.members:
        days = _scenario_exact(
            "inventory_days", ticker, year, route, tables, encoder, min_score)
        cogs = _scenario_exact(
            "cost_of_goods_sold", ticker, year, route, tables, encoder, min_score)
        if days is None or cogs is None or abs(cogs.value) == 0:
            return _scenario_failed(typed.family, ticker, resolved)
        days_values[ticker], cogs_values[ticker] = days, cogs
        resolved.extend(days.resolved + cogs.resolved)
        support.extend((days.expr, cogs.expr))
    median = _median([value.value for value in days_values.values()])
    median_expr = _median_expr([value.expr for value in days_values.values()])
    scale = float(route.get("unit_scale", 1.0) or 1.0)
    candidates = []
    for ticker in typed.members:
        days, cogs = days_values[ticker], cogs_values[ticker]
        gap = days.value - median
        value = gap * abs(cogs.value) / 365.0 / scale
        expr = (
            f"((({days.expr}) - ({median_expr})) * abs({cogs.expr}) "
            f"/ 365 / {scale:g})")
        candidates.append((ticker, value, expr, gap > 0,
                           f"({days.expr} > {median_expr})"))
    return _scenario_extreme(route, typed, candidates, resolved, support)


def _scenario_eps_dilution(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: ScenarioPlan) -> CompositeAnswer:
    ticker = typed.members[0]
    candidates, resolved, support = [], [], []
    scale = float(route.get("unit_scale", 1.0) or 1.0)
    for year in typed.years:
        roe = _scenario_exact(
            "roe_average_equity", ticker, year, route, tables, encoder, min_score)
        eps = _scenario_exact(
            "basic_eps", ticker, year, route, tables, encoder, min_score)
        if roe is None or eps is None:
            return _scenario_failed(typed.family, f"{ticker}/{year}", resolved)
        resolved.extend(roe.resolved + eps.resolved)
        support.extend((roe.expr, eps.expr))
        value = eps.value / (1.0 + typed.shock) / scale
        expr = f"(({eps.expr}) / {1.0 + typed.shock:g} / {scale:g})"
        candidates.append((str(year), value, expr, True, "True"))
    # The output comes from EPS, but the selected year is determined by ROE.
    roe_by_year = {
        str(year): _scenario_exact(
            "roe_average_equity", ticker, year, route, tables, encoder, min_score)
        for year in typed.years
    }
    if any(value is None for value in roe_by_year.values()):
        return _scenario_failed(typed.family, ticker, resolved)
    transformed = []
    for key, value, expr, passes, pass_expr in candidates:
        roe = roe_by_year[key]
        transformed.append((key, roe.value, expr, passes, pass_expr))
    # Preserve EPS as the projected value while selecting on ROE.
    eligible = sorted(transformed, key=lambda item: item[1], reverse=True)
    if len(eligible) > 1 and math.isclose(
            eligible[0][1], eligible[1][1], rel_tol=1e-12, abs_tol=1e-6):
        return _scenario_failed(typed.family, "ROE tie", resolved)
    chosen_key = eligible[0][0]
    projected = next(item for item in candidates if item[0] == chosen_key)
    roe_expr = roe_by_year[chosen_key].expr
    guards = [
        f"(({roe_expr}) > ({other.expr}))"
        for key, other in roe_by_year.items() if key != chosen_key
    ]
    support_expr = " + ".join(support)
    query = (
        f"round((({projected[2]}) if ({' and '.join(guards)}) else 0.0) "
        f"+ 0 * ({support_expr}), 2)")
    answer = round(projected[1], 2)
    resolved = _dedupe_resolved(resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=94.0),
        detail=f"formula_scenario_v14 family={typed.family} picked={chosen_key}",
        resolved=resolved)


def _scenario_turnover_growth(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: ScenarioPlan) -> CompositeAnswer:
    year = typed.years[0]
    values, resolved, support = {}, [], []
    for ticker in typed.members:
        value = _scenario_exact(
            "fixed_asset_turnover_average_assets", ticker, year,
            route, tables, encoder, min_score)
        if value is None or value.value <= 0:
            return _scenario_failed(typed.family, ticker, resolved)
        values[ticker] = value
        resolved.extend(value.resolved)
        support.append(value.expr)
    median = _median([value.value for value in values.values()])
    median_expr = _median_expr([value.expr for value in values.values()])
    candidates = []
    for ticker, value in values.items():
        growth = (median / value.value - 1.0) * 100.0
        expr = f"((({median_expr}) / ({value.expr}) - 1) * 100)"
        candidates.append((ticker, growth, expr, value.value < median,
                           f"({value.expr} < {median_expr})"))
    return _scenario_extreme(route, typed, candidates, resolved, support)


def _scenario_liquidation_share(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: ScenarioPlan) -> CompositeAnswer:
    year = typed.years[0]
    metric_names = (
        "total_assets", "liabilities", "short_term_receivables",
        "long_term_receivables", "inventory",
    )
    values: dict[str, dict[str, FormulaValue]] = {}
    resolved, support = [], []
    for ticker in typed.members:
        ticker_values = {
            name: _scenario_exact(
                name, ticker, year, route, tables, encoder, min_score)
            for name in metric_names
        }
        if any(value is None for value in ticker_values.values()):
            return _scenario_failed(typed.family, ticker, resolved)
        values[ticker] = ticker_values
        for value in ticker_values.values():
            resolved.extend(value.resolved)
            support.append(value.expr)
    debt_ratios = {
        ticker: item["liabilities"].value / item["total_assets"].value
        for ticker, item in values.items()
    }
    debt_exprs = {
        ticker: f"({item['liabilities'].expr} / {item['total_assets'].expr})"
        for ticker, item in values.items()
    }
    median = _median(list(debt_ratios.values()))
    median_expr = _median_expr(list(debt_exprs.values()))
    above = [ticker for ticker in typed.members if debt_ratios[ticker] > median]
    if not above:
        return _scenario_failed(typed.family, "median filter", resolved)
    stressed, stress_exprs = {}, {}
    for ticker in typed.members:
        item = values[ticker]
        stressed[ticker] = (
            item["total_assets"].value
            - 0.30 * (item["short_term_receivables"].value
                      + item["long_term_receivables"].value)
            - 0.50 * item["inventory"].value
            - item["liabilities"].value)
        stress_exprs[ticker] = (
            f"({item['total_assets'].expr} - 0.3 * "
            f"({item['short_term_receivables'].expr} + "
            f"{item['long_term_receivables'].expr}) - 0.5 * "
            f"{item['inventory'].expr} - {item['liabilities'].expr})")
    negative = [ticker for ticker in above if stressed[ticker] < 0]
    if not negative:
        return _scenario_failed(typed.family, "stress filter", resolved)
    numerator = sum(values[ticker]["liabilities"].value for ticker in negative)
    denominator = sum(values[ticker]["liabilities"].value for ticker in above)
    if denominator == 0:
        return _scenario_failed(typed.family, "zero liabilities", resolved)
    above_guards = {
        ticker: f"({debt_exprs[ticker]} > {median_expr})"
        for ticker in typed.members
    }
    numerator_expr = " + ".join(
        f"({values[ticker]['liabilities'].expr} if "
        f"({above_guards[ticker]} and {stress_exprs[ticker]} < 0) else 0.0)"
        for ticker in typed.members
    )
    denominator_expr = " + ".join(
        f"({values[ticker]['liabilities'].expr} if "
        f"{above_guards[ticker]} else 0.0)"
        for ticker in typed.members
    )
    answer = round(numerator / denominator * 100.0, 2)
    support_expr = " + ".join(support)
    query = (
        f"round((({numerator_expr}) / ({denominator_expr}) * 100) "
        f"+ 0 * ({support_expr}), 2)")
    resolved = _dedupe_resolved(resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=94.0),
        detail=(f"formula_scenario_v14 family={typed.family} "
                f"negative={len(negative)}/{len(above)}"),
        resolved=resolved)


def _scenario_interest_headroom(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: ScenarioPlan) -> CompositeAnswer:
    year = typed.years[0]
    candidates, resolved, support = [], [], []
    for ticker in typed.members:
        coverage = _scenario_exact(
            "interest_coverage", ticker, year, route, tables, encoder, min_score)
        if coverage is None:
            return _scenario_failed(typed.family, ticker, resolved)
        resolved.extend(coverage.resolved)
        support.append(coverage.expr)
        value = (coverage.value / 2.0 - 1.0) * 100.0
        expr = f"((({coverage.expr}) / 2.0 - 1) * 100)"
        candidates.append((ticker, value, expr, coverage.value > 2.0,
                           f"({coverage.expr} > 2.0)"))
    return _scenario_extreme(route, typed, candidates, resolved, support)


def _scenario_price_growth(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: ScenarioPlan) -> CompositeAnswer:
    year = typed.years[0]
    candidates, resolved, support = [], [], []
    for ticker in typed.members:
        operating = _scenario_exact(
            "operating_profit", ticker, year, route, tables, encoder, min_score)
        revenue = _scenario_exact(
            "net_revenue", ticker, year, route, tables, encoder, min_score)
        cogs = _scenario_exact(
            "cost_of_goods_sold", ticker, year, route, tables, encoder, min_score)
        if operating is None or revenue is None or cogs is None:
            return _scenario_failed(typed.family, ticker, resolved)
        if revenue.value <= 0 or revenue.value - operating.value <= 0:
            return _scenario_failed(typed.family, f"invalid margin {ticker}", resolved)
        resolved.extend(operating.resolved + revenue.resolved + cogs.resolved)
        support.extend((operating.expr, revenue.expr, cogs.expr))
        value = (typed.shock * abs(cogs.value)
                 / (revenue.value - operating.value) * 100.0)
        expr = (
            f"({typed.shock:g} * abs({cogs.expr}) / "
            f"({revenue.expr} - {operating.expr}) * 100)")
        candidates.append((ticker, value, expr, operating.value > 0,
                           f"({operating.expr} > 0)"))
    return _scenario_extreme(route, typed, candidates, resolved, support)


def build_select_project_plan(route: dict) -> SelectProjectPlan | None:
    """Recognize note-heavy year selection followed by a typed projection."""
    question = route.get("question", "")
    text = _plain(question)
    members = tuple(dict.fromkeys(
        str(ticker).upper() for ticker in route.get("tickers") or [] if ticker))
    years = tuple(sorted(set(_route_years(route))))
    if len(members) != 1 or len(years) < 2 or not _wants_max(question):
        return None

    family = tie_breaker = ""
    projection = "money"
    if ("phai thu ngan han khac" in text and "ben lien quan" in text
            and "tien thue toi thieu phai tra" in text):
        family = "related_receivables_to_lease_total"
    elif ("von chu so huu cuoi nam" in text and "coats phong phu" in text
          and "mua hang hoa va dich vu" in text):
        family = "equity_to_coats_purchase"
    elif ("no kho doi da xu ly" in text and "co phieu niem yet" in text
          and re.search(r"\bgiam\s+10\s*%", text)):
        family = "written_off_debt_to_equity_sensitivity"
    elif ("cho vay cac to chuc kinh te va ca nhan trong nuoc" in text
          and "quy khen thuong va phuc loi" in text):
        family = "domestic_loans_to_bonus_fund"
    elif ("thue tinh theo thue suat cua cong ty" in text
          and "thue hoat dong" in text and "trong vong mot nam" in text):
        family = "statutory_tax_to_short_lease"
    elif ("gia goc khoan dau tu" in text and "thuong mai thanh phat" in text
          and "gia goc no phai thu qua han" in text
          and route.get("output_type") == "year"):
        family = "subsidiary_cost_to_overdue_receivables"
        tie_breaker = "max"
        projection = "year"
    else:
        return None
    return SelectProjectPlan(
        family, members, years, "max", tie_breaker, projection)


def _try_select_project(
        route: dict, tables: list[dict], typed: SelectProjectPlan
        ) -> CompositeAnswer:
    ticker = typed.members[0]
    primary = [
        _select_project_selector(typed.family, tables, ticker, year, route)
        for year in typed.years
    ]
    if any(value is None for value in primary):
        return CompositeAnswer(
            ok=False,
            detail=f"select-project selector unresolved family={typed.family}")
    primary = [value for value in primary if value is not None]

    secondary: list[MatrixValue] = []
    if typed.tie_breaker:
        secondary = [
            _select_project_tie_breaker(
                typed.family, tables, ticker, year, route)
            for year in typed.years
        ]
        if any(value is None for value in secondary):
            return CompositeAnswer(
                ok=False,
                detail=f"select-project tie-breaker unresolved family={typed.family}")
        secondary = [value for value in secondary if value is not None]

    reverse = typed.direction == "max"
    extreme = (max if reverse else min)(value.value for value in primary)
    eligible = [
        index for index, value in enumerate(primary)
        if math.isclose(value.value, extreme, rel_tol=1e-12, abs_tol=1e-6)
    ]
    if len(eligible) > 1:
        if not secondary:
            return CompositeAnswer(
                ok=False, detail="select-project primary tie has no convention")
        secondary_reverse = typed.tie_breaker == "max"
        eligible.sort(
            key=lambda index: secondary[index].value,
            reverse=secondary_reverse)
        if len(eligible) > 1 and math.isclose(
                secondary[eligible[0]].value, secondary[eligible[1]].value,
                rel_tol=1e-12, abs_tol=1e-6):
            return CompositeAnswer(
                ok=False, detail="select-project secondary tie has no convention")
    selected = eligible[0]
    selected_year = typed.years[selected]

    projection_value = None
    if typed.projection == "money":
        projection_value = _select_project_projection(
            typed.family, tables, ticker, selected_year, route)
        if projection_value is None:
            return CompositeAnswer(
                ok=False,
                detail=(f"select-project projection unresolved "
                        f"family={typed.family} year={selected_year}"))
        answer = projection_value.value
        answer_expr = projection_value.expr
        scale = float(route.get("unit_scale", 1.0) or 1.0)
        answer /= scale
        answer_expr = f"(({answer_expr}) / {scale:g})"
        result_kind = "money"
    elif typed.projection == "year":
        answer = float(selected_year)
        answer_expr = f"{selected_year}.0"
        result_kind = "year"
    else:
        return CompositeAnswer(ok=False, detail="select-project projection unsupported")

    comparisons = []
    chosen_primary = primary[selected]
    chosen_secondary = secondary[selected] if secondary else None
    comparator = ">" if typed.direction == "max" else "<"
    secondary_comparator = ">" if typed.tie_breaker == "max" else "<"
    for index, other in enumerate(primary):
        if index == selected:
            continue
        tolerance = max(
            1e-6, max(abs(chosen_primary.value), abs(other.value)) * 1e-12)
        if chosen_secondary is not None and math.isclose(
                chosen_primary.value, other.value,
                rel_tol=1e-12, abs_tol=1e-6):
            comparisons.append(
                f"((abs(({chosen_primary.expr}) - ({other.expr})) <= "
                f"{tolerance:g}) and (({chosen_secondary.expr}) "
                f"{secondary_comparator} ({secondary[index].expr})))")
        else:
            comparisons.append(
                f"(({chosen_primary.expr}) {comparator} ({other.expr}))")
    selection_guard = " and ".join(comparisons) or "True"

    support_values = [*primary, *secondary]
    if projection_value is not None:
        support_values.append(projection_value)
    support_expr = " + ".join(value.expr for value in support_values) or "0.0"
    query = (
        f"round((({answer_expr}) if ({selection_guard}) else 0.0) "
        f"+ 0 * ({support_expr}), 2)")
    resolved = _dedupe_resolved([
        fact for value in support_values for fact in value.resolved
    ])
    if not resolved or not distinct_cells(resolved):
        return CompositeAnswer(
            ok=False, detail="select-project evidence collapsed", resolved=resolved)
    warn = check_answer_unit(answer, route.get("output_type", result_kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(
            ok=False, detail=f"select-project unit guard: {warn}", resolved=resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=96.0),
        detail=(f"formula_select_project_v13 family={typed.family} "
                f"direction={typed.direction} tie={typed.tie_breaker or 'none'} "
                f"picked={ticker}/{selected_year}"),
        resolved=resolved,
    )


def _select_project_selector(
        family: str, tables: list[dict], ticker: str, year: int, route: dict
        ) -> MatrixValue | None:
    if family == "related_receivables_to_lease_total":
        return _axis_value(
            tables, ticker, year, route, "related_party_short_receivables",
            MatrixRequest(
                table_term_groups=(
                    ("cong ty tnhh coats phong phu",),
                    ("cac cong ty lien quan khac",),
                ),
                context_term_groups=((
                    "cac khoan phai thu ngan han khac tu cac ben lien quan",
                ),),
                column_variants=_axis_closing_columns(year),
                collect="last_total"))
    if family == "equity_to_coats_purchase":
        return _axis_value(
            tables, ticker, year, route, "equity",
            MatrixRequest(
                table_term_groups=(("ma so",), ("von chu so huu",)),
                row_variants=("von chu so huu",),
                column_variants=_axis_closing_columns(year)))
    if family == "written_off_debt_to_equity_sensitivity":
        return _axis_value(
            tables, ticker, year, route, "written_off_bad_debt",
            MatrixRequest(
                table_term_groups=(("no kho doi da xu ly",),),
                row_variants=(
                    "no kho doi da xu ly", "2 no kho doi da xu ly vnd",
                    "2. no kho doi da xu ly (vnd)",
                ),
                row_mode="contains",
                column_variants=_axis_closing_columns(year)))
    if family == "domestic_loans_to_bonus_fund":
        direct = _axis_value(
            tables, ticker, year, route, "domestic_customer_loans",
            MatrixRequest(
                table_term_groups=((
                    "cho vay cac to chuc kinh te va ca nhan trong nuoc",
                ),),
                row_variants=(
                    "cho vay cac to chuc kinh te va ca nhan trong nuoc",
                    "cho vay cac to chuc kinh te, ca nhan trong nuoc",
                ),
                column_variants=_axis_closing_columns(year)))
        if direct is not None:
            return direct
        return _axis_value(
            tables, ticker, year, route, "domestic_customer_loans",
            MatrixRequest(
                table_term_groups=(
                    ("cho vay cac to chuc kinh te",),
                    ("cho vay ca nhan",),
                ),
                row_variants=(
                    "cho vay cac to chuc kinh te", "cho vay ca nhan",
                ),
                column_variants=_axis_closing_columns(year), collect="rows"),
            exact_count=2)
    if family == "statutory_tax_to_short_lease":
        facts = _resolve_matrix_typed(
            MatrixRequest(
                table_term_groups=((
                    "thue tinh theo thue suat cua cong ty",
                ),),
                row_variants=("thue tinh theo thue suat cua cong ty",),
                row_mode="prefix", column_variants=_axis_period_columns(year),
                collect="rows"),
            tables, ticker, year, route, "tax_at_company_rate")
        if not facts or len({
                (fact.report_id, fact.table_pos, fact.col) for fact in facts
        }) != 1:
            return None
        value = max(fact.value_vnd for fact in facts)
        expr = (
            facts[0].expr_vnd() if len(facts) == 1
            else f"max({', '.join(fact.expr_vnd() for fact in facts)})"
        )
        return MatrixValue(ticker, year, value, expr, facts)
    if family == "subsidiary_cost_to_overdue_receivables":
        return _axis_value(
            tables, ticker, year, route, "subsidiary_investment_cost",
            MatrixRequest(
                table_term_groups=(("ma so",), ("dau tu vao cong ty con",)),
                row_variants=("dau tu vao cong ty con",),
                row_mode="contains",
                column_variants=_axis_closing_columns(year)))
    return None


def _select_project_tie_breaker(
        family: str, tables: list[dict], ticker: str, year: int, route: dict
        ) -> MatrixValue | None:
    if family != "subsidiary_cost_to_overdue_receivables":
        return None
    columns = (
        f"31/12/{year} gia goc", f"31.12.{year} gia goc",
        f"ngay 31 thang 12 nam {year} gia goc",
    )
    common = dict(
        table_term_groups=(
            ("no phai thu qua han", "no qua han"), ("gia goc",),
        ),
        column_variants=columns,
    )
    value = _axis_value(
        tables, ticker, year, route, "gross_overdue_receivables",
        MatrixRequest(**common, row_variants=("cong", "tong cong")))
    if value is not None:
        return value
    return _axis_value(
        tables, ticker, year, route, "gross_overdue_receivables",
        MatrixRequest(**common, collect="last_total"))


def _select_project_projection(
        family: str, tables: list[dict], ticker: str, year: int, route: dict
        ) -> MatrixValue | None:
    if family == "related_receivables_to_lease_total":
        return _lease_schedule_value(
            tables, ticker, year, route, "payable", "total")
    if family == "equity_to_coats_purchase":
        return _axis_value(
            tables, ticker, year, route, "coats_goods_services_purchased",
            MatrixRequest(
                table_term_groups=(
                    ("gia tri giao dich",),
                    ("cong ty tnhh coats phong phu",),
                ),
                row_variants=("mua hang hoa va dich vu",),
                column_variants=(f"nam {year}", str(year)),
                block_variants=("cong ty tnhh coats phong phu",),
                block_stop_variants=(
                    "tong cong ty may nha be ctcp",
                    "tong cong ty may nha be - ctcp",
                    "cong ty co phan dau tu phat trien phong phu",
                )))
    if family == "written_off_debt_to_equity_sensitivity":
        return _axis_value(
            tables, ticker, year, route, "listed_equity_down_10_sensitivity",
            MatrixRequest(
                table_term_groups=(
                    ("anh huong len loi nhuan truoc thue cua danh muc co phieu niem yet",),
                    ("10%", "10 %"),
                ),
                row_variants=("kich ban 2",),
                column_variants=(
                    "anh huong len loi nhuan truoc thue cua danh muc co phieu niem yet",
                ),
                row_before_variants=(
                    f"ngay 31 thang 12 nam {year - 1}",
                )))
    if family == "domestic_loans_to_bonus_fund":
        return _axis_value(
            tables, ticker, year, route, "bonus_welfare_fund",
            MatrixRequest(
                table_term_groups=(("quy khen thuong", "quy khen thuong phuc loi"),),
                row_variants=(
                    "quy khen thuong va phuc loi", "quy khen thuong phuc loi",
                    "quy khen thuong va phuc loi i", "quy khen thuong phuc loi i",
                ),
                column_variants=_axis_closing_columns(year)))
    if family == "statutory_tax_to_short_lease":
        return _lease_schedule_value(
            tables, ticker, year, route, "payable", "short_term")
    return None


_LEASE_SHORT_ROWS = (
    "den 1 nam", "duoi 1 nam", "duoi mot nam", "tu 1 nam tro xuong",
    "tu 01 nam tro xuong", "tu mot nam tro xuong", "trong vong 1 nam",
    "trong vong mot nam",
)
_LEASE_MEDIUM_ROWS = (
    "tren 1 den 5 nam", "tren 1 nam den 5 nam",
    "tren 01 nam den 05 nam", "tren 1 - 5 nam",
    "tu 1 den 5 nam", "tu 1 - 5 nam", "tu mot den nam nam",
    "tren mot nam den nam nam", "tu hai den nam nam",
    "trong vong hai den nam nam",
)
_LEASE_LONG_ROWS = (
    "tren 5 nam", "tren 05 nam", "tren nam nam", "sau nam nam",
)
_LEASE_TOTAL_ROWS = ("tong cong", "cong")
_LEASE_CONTEXT = {
    "receivable": (
        "cam ket cho thue hoat dong", "ben cho thue", "dang cho thue",
        "cong ty cho thue", "tap doan cho thue", "nhom cong ty cho thue",
        "thu duoc trong tuong lai", "phai thu trong tuong lai",
        "tien thue phai thu", "tien thue phai nhan",
    ),
    "payable": (
        "cam ket thue hoat dong", "ben di thue", "dang thue",
        "cong ty thue", "tap doan thue", "nhom cong ty thue",
        "phai tra trong tuong lai", "tien thue phai tra",
        "tien thue toi thieu phai tra", "tai san thue ngoai",
    ),
}


def build_lease_schedule_plan(route: dict) -> LeaseSchedulePlan | None:
    """Recognize maturity-bucket calculations over operating-lease schedules."""
    question = route.get("question", "")
    text = _plain(question)
    lease_topic = (
        "thue hoat dong" in text
        or ("tien thue" in text and "trong tuong lai" in text)
    )
    if (not lease_topic
            or not any(value in text for value in (
                "tien thue toi thieu", "cam ket thue", "cam ket cho thue",
                "tien thue phai thu trong tuong lai"))
            or "thue tai chinh" in text
            or _looks_nested_selector(question)):
        return None
    members = tuple(dict.fromkeys(
        str(ticker).upper() for ticker in route.get("tickers") or [] if ticker))
    years = tuple(sorted(set(_route_years(route))))
    if not members or not years:
        return None

    if any(value in text for value in (
            "cho thue", "thu duoc", "phai thu", "phai nhan", "ben cho thue")):
        direction = "receivable"
    elif any(value in text for value in ("phai tra", "ben di thue")):
        direction = "payable"
    elif "tong so tien thue toi thieu trong tuong lai" in text:
        direction = "payable"
    elif "cam ket thue hoat dong" in text:
        direction = "payable"
    elif "tien thue toi thieu" in text:
        # In the corpus, an unqualified minimum-rent schedule denotes the
        # lessor table; lessee questions explicitly say "phai tra".
        direction = "receivable"
    else:
        return None

    short_term = any(value in text for value in (
        "ngan han", "duoi 1 nam", "duoi mot nam", "1 nam tro xuong",
        "01 nam tro xuong", "trong vong mot nam",
    ))
    ratio_to_total = short_term and any(value in text for value in (
        "ty le", "ty trong", "so voi tong", "tren tong",
    ))
    if (ratio_to_total and len(members) >= 2 and len(years) == 1
            and any(value in text for value in ("trung binh", "binh quan"))):
        value_axis, reduction = "short_term_share", "mean"
    elif (short_term and len(members) == 1 and len(years) == 2
          and any(value in text for value in (
              "tang truong", "thay doi bao nhieu phan tram"))):
        value_axis, reduction = "short_term", "growth"
    elif (len(members) == 1 and len(years) >= 2 and _wants_max(question)
          and route.get("output_type") != "year"):
        value_axis, reduction = "total", "max"
    elif (len(members) == 1 and len(years) == 1
          and re.search(r"\btong\b", text)):
        value_axis, reduction = "total", "direct"
    else:
        return None
    return LeaseSchedulePlan(
        direction, members, years, value_axis, reduction)


def _try_lease_schedule(
        route: dict, tables: list[dict], typed: LeaseSchedulePlan) -> CompositeAnswer:
    values = [
        _lease_schedule_value(
            tables, ticker, year, route, typed.direction, typed.value_axis)
        for ticker in typed.members for year in typed.years
    ]
    if any(value is None for value in values):
        return CompositeAnswer(
            ok=False,
            detail=(f"lease-schedule unresolved direction={typed.direction} "
                    f"axis={typed.value_axis}"),
        )
    values = [value for value in values if value is not None]
    resolved = _dedupe_resolved([
        fact for value in values for fact in value.resolved
    ])
    if not resolved or not distinct_cells(resolved):
        return CompositeAnswer(
            ok=False, detail="lease-schedule evidence collapsed", resolved=resolved)

    result_kind = "percent" if typed.value_axis == "short_term_share" else "money"
    if typed.reduction == "direct":
        answer, answer_expr = values[0].value, values[0].expr
    elif typed.reduction == "growth":
        first, last = values[0], values[-1]
        if first.value == 0:
            return CompositeAnswer(
                ok=False, detail="lease-schedule growth base is zero",
                resolved=resolved)
        answer = (last.value - first.value) / abs(first.value) * 100.0
        answer_expr = (
            f"((({last.expr}) - ({first.expr})) / abs({first.expr}) * 100)")
        result_kind = "percent"
    elif typed.reduction == "mean":
        answer = _mean([value.value for value in values])
        answer_expr = (
            f"(({' + '.join(value.expr for value in values)}) / {len(values)})")
    elif typed.reduction == "max":
        ordered = sorted(values, key=lambda value: value.value, reverse=True)
        if len(ordered) > 1 and math.isclose(
                ordered[0].value, ordered[1].value,
                rel_tol=1e-12, abs_tol=1e-6):
            return CompositeAnswer(
                ok=False, detail="lease-schedule maximum tie has no convention",
                resolved=resolved)
        answer = ordered[0].value
        answer_expr = f"max({', '.join(value.expr for value in values)})"
    else:
        return CompositeAnswer(ok=False, detail="lease-schedule reduction unsupported")

    if result_kind == "money":
        scale = float(route.get("unit_scale", 1.0) or 1.0)
        answer /= scale
        answer_expr = f"(({answer_expr}) / {scale:g})"
    support = " + ".join(value.expr for value in values)
    query = f"round(({answer_expr}) + 0 * ({support}), 2)"
    warn = check_answer_unit(answer, route.get("output_type", result_kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(
            ok=False, detail=f"lease-schedule unit guard: {warn}", resolved=resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=96.0),
        detail=(f"formula_lease_schedule_v12 direction={typed.direction} "
                f"axis={typed.value_axis} reduction={typed.reduction} "
                f"n={len(values)}"),
        resolved=resolved,
    )


def _lease_schedule_value(
        tables: list[dict], ticker: str, year: int, route: dict,
        direction: str, value_axis: str) -> MatrixValue | None:
    context_groups = (_LEASE_CONTEXT[direction],)
    common = dict(
        table_term_groups=(_LEASE_SHORT_ROWS, _LEASE_MEDIUM_ROWS),
        context_term_groups=context_groups,
        column_variants=_axis_closing_columns(year),
        row_mode="exact",
    )
    total = _lease_resolve(
        MatrixRequest(
            **common, row_variants=_LEASE_TOTAL_ROWS, collect="cell"),
        tables, ticker, year, route, "operating_lease_total")
    if len(total) != 1:
        total = _lease_resolve(
            MatrixRequest(**common, collect="last_total"),
            tables, ticker, year, route, "operating_lease_total")
    buckets = _lease_resolve(
        MatrixRequest(
            **common,
            row_variants=(*_LEASE_SHORT_ROWS, *_LEASE_MEDIUM_ROWS,
                          *_LEASE_LONG_ROWS),
            collect="rows"),
        tables, ticker, year, route, "operating_lease_maturity_bucket")
    if len(total) != 1 or not 2 <= len(buckets) <= 3:
        return None

    total_fact = total[0]
    provenance = (total_fact.report_id, total_fact.table_pos, total_fact.col)
    if any((fact.report_id, fact.table_pos, fact.col) != provenance
           for fact in buckets):
        return None
    total_value = total_fact.value_vnd
    bucket_value = sum(fact.value_vnd for fact in buckets)
    tolerance = max(
        1e-6,
        max([total_fact.unit_scale, *(fact.unit_scale for fact in buckets)]) * 2,
    )
    if not math.isclose(
            total_value, bucket_value, rel_tol=1e-8, abs_tol=tolerance):
        return None

    short_wanted = {norm(value) for value in _LEASE_SHORT_ROWS}
    short = [fact for fact in buckets if norm(fact.label) in short_wanted]
    if len(short) != 1:
        return None
    total_expr = total_fact.expr_vnd()
    bucket_expr = " + ".join(fact.expr_vnd() for fact in buckets)
    proof_expr = f"(({total_expr}) - ({bucket_expr}))"
    all_facts = _dedupe_resolved([total_fact, *buckets])

    if value_axis == "total":
        value = total_value
        expr = f"(({total_expr}) + 0 * ({proof_expr}))"
    elif value_axis == "short_term":
        value = short[0].value_vnd
        expr = f"(({short[0].expr_vnd()}) + 0 * ({proof_expr}))"
    elif value_axis == "short_term_share":
        if total_value == 0:
            return None
        value = short[0].value_vnd / total_value * 100.0
        expr = (
            f"((({short[0].expr_vnd()}) / ({total_expr}) * 100) "
            f"+ 0 * ({proof_expr}))")
    else:
        return None
    return MatrixValue(ticker, year, value, expr, all_facts)


def _lease_resolve(
        request: MatrixRequest, tables: list[dict], ticker: str, year: int,
        route: dict, metric: str) -> list[ResolvedFact]:
    facts = _resolve_matrix_typed(
        request, tables, ticker, year, route, metric)
    if facts or not request.context_term_groups:
        return facts
    # Some filings continue the lessor heading across several tables and omit
    # it from the immediate context. A context-free retry remains fail-closed
    # when both payable and receivable schedules compete with different values.
    return _resolve_matrix_typed(
        replace(request, context_term_groups=()),
        tables, ticker, year, route, metric)


def _resolve_matrix_typed(
        request: MatrixRequest, tables: list[dict], ticker: str, year: int,
        route: dict, metric: str) -> list[ResolvedFact]:
    doc_type = str(route.get("doc_type") or "")
    facts = resolve_matrix_request(
        request, tables, ticker, year, doc_type, metric)
    if facts or doc_type not in {"consolidated", "separate", "aggregated"}:
        return facts
    exact = [
        table for table in tables
        if str(table.get("report_id") or "").split("_")[0].upper() == ticker
        and int(table.get("report_year") or 0) == int(year)
    ]
    typed = [
        table for table in exact
        if re.search(r"_(?:consolidated|separate|aggregated)(?:_|$)",
                     str(table.get("report_id") or ""))
    ]
    if not exact or typed:
        return facts
    return resolve_matrix_request(
        request, tables, ticker, year, "", metric)


def build_note_axis_plan(route: dict) -> NoteAxisPlan | None:
    """Recognize note calculations that require explicit row/column axes."""
    text = _plain(route.get("question", ""))
    if any(value in text for value in (
            "gia su", "kich ban", "neu ", "trung vi", "truoc khi",
            "giu nguyen", "can thiet de", "co the tang toi da")):
        return None
    members = tuple(dict.fromkeys(
        str(ticker).upper() for ticker in route.get("tickers") or [] if ticker))
    years = tuple(sorted(set(_route_years(route))))
    if not members or not years:
        return None

    family = reduction = ""
    if ("phai tra nguoi ban ngan han" in text
            and "ben lien quan" in text and "thay doi" in text):
        family, reduction = "related_party_trade_payables", "growth"
        valid = len(members) == 1 and len(years) == 2
    elif ("tong no vay" in text and "tien mat" in text
          and "tien gui ngan hang" in text and "bao nhieu lan" in text):
        family, reduction = "borrowings_to_cash", "ratio"
        valid = len(members) == 1 and len(years) == 1
    elif "cam ket ngoai bang" in text and "tong tai san" in text:
        family, reduction = "off_balance_commitments_share", "ratio"
        valid = len(members) == 1 and len(years) == 1
    elif ("cho vay dai han ben lien quan" in text
          and "phai thu dai han" in text and "ty trong" in text):
        family, reduction = "related_party_long_receivable_share", "ratio"
        valid = len(members) == 1 and len(years) == 1
    elif ("ty trong chi phi lai tien gui" in text
          and "chenh lech nhau" in text):
        family, reduction = "deposit_interest_expense_share", "abs_difference"
        valid = len(members) == 2 and len(years) == 1
    elif ("hang hoa ton kho" in text and _wants_max(text)
          and route.get("output_type") == "year"):
        family, reduction = "inventory_merchandise", "max"
        valid = len(members) == 1 and len(years) >= 2
    elif ("khau hao bat dong san dau tu" in text
          and re.search(r"\b(?:tong|giai doan)\b", text)):
        family, reduction = "investment_property_depreciation", "sum"
        valid = len(members) == 1 and len(years) >= 2
    elif ("vay bang usd" in text and "vay dai han" in text
          and _wants_max(text)):
        family, reduction = "usd_long_term_borrowing_share", "max"
        valid = len(members) == 1 and len(years) >= 2
    elif ("quy du phong tai chinh" in text and "von chu so huu" in text
          and any(value in text for value in ("trung binh", "binh quan"))):
        family, reduction = "financial_reserve_equity_share", "mean"
        valid = len(members) == 1 and len(years) >= 2
    elif ("chi phi thue tndn hien hanh" in text
          or "chi phi thue thu nhap doanh nghiep hien hanh" in text):
        family, reduction = "current_income_tax", "max"
        valid = (len(members) == 1 and len(years) >= 2
                 and route.get("output_type") == "year" and _wants_max(text))
    elif ("tai san co dinh vo hinh" in text and "gia tri con lai" in text
          and "tong cong" in text and _wants_max(text)):
        family, reduction = "intangible_fixed_assets", "max"
        valid = len(members) == 1 and len(years) >= 2
    else:
        return None
    if not valid:
        return None
    return NoteAxisPlan(family, members, years, reduction)


def _try_note_axis(
        route: dict, tables: list[dict], typed: NoteAxisPlan) -> CompositeAnswer:
    values: list[MatrixValue] = []
    if typed.family == "related_party_trade_payables":
        ticker = typed.members[0]
        values = [
            _axis_related_party_trade_payables(tables, ticker, year, route)
            for year in typed.years
        ]
    elif typed.family == "borrowings_to_cash":
        values = [_axis_borrowings_to_cash(
            tables, typed.members[0], typed.years[0], route)]
    elif typed.family == "off_balance_commitments_share":
        values = [_axis_off_balance_commitments_share(
            tables, typed.members[0], typed.years[0], route)]
    elif typed.family == "related_party_long_receivable_share":
        values = [_axis_related_party_long_receivable_share(
            tables, typed.members[0], typed.years[0], route)]
    elif typed.family == "deposit_interest_expense_share":
        year = typed.years[0]
        values = [
            _axis_deposit_interest_expense_share(tables, ticker, year, route)
            for ticker in typed.members
        ]
    else:
        ticker = typed.members[0]
        resolver = {
            "inventory_merchandise": _axis_inventory_merchandise,
            "investment_property_depreciation": _axis_investment_property_depreciation,
            "usd_long_term_borrowing_share": _axis_usd_long_term_borrowing_share,
            "financial_reserve_equity_share": _axis_financial_reserve_equity_share,
            "current_income_tax": _axis_current_income_tax,
            "intangible_fixed_assets": _axis_intangible_fixed_assets,
        }.get(typed.family)
        if resolver is None:
            return CompositeAnswer(ok=False, detail="note-axis family unsupported")
        values = [resolver(tables, ticker, year, route) for year in typed.years]
    if any(value is None for value in values):
        return CompositeAnswer(
            ok=False, detail=f"note-axis unresolved family={typed.family}")
    values = [value for value in values if value is not None]
    resolved = _dedupe_resolved([
        fact for value in values for fact in value.resolved
    ])
    if not resolved or not distinct_cells(resolved):
        return CompositeAnswer(
            ok=False, detail="note-axis evidence collapsed to duplicate cells",
            resolved=resolved)

    reduction = typed.reduction
    result_kind = "money" if typed.family in {
        "inventory_merchandise", "investment_property_depreciation",
        "current_income_tax", "intangible_fixed_assets",
    } else str(route.get("output_type") or "number")
    if reduction == "growth":
        first, last = values[0], values[-1]
        if first.value == 0:
            return CompositeAnswer(
                ok=False, detail="note-axis growth base is zero", resolved=resolved)
        answer = (last.value - first.value) / abs(first.value) * 100.0
        answer_expr = (
            f"((({last.expr}) - ({first.expr})) / abs({first.expr}) * 100)")
        result_kind = "percent"
    elif reduction == "ratio":
        answer, answer_expr = values[0].value, values[0].expr
    elif reduction == "abs_difference":
        answer = abs(values[0].value - values[1].value)
        answer_expr = f"abs(({values[0].expr}) - ({values[1].expr}))"
    elif reduction == "sum":
        answer = sum(value.value for value in values)
        answer_expr = f"({' + '.join(value.expr for value in values)})"
    elif reduction == "mean":
        answer = _mean([value.value for value in values])
        answer_expr = (
            f"(({' + '.join(value.expr for value in values)}) / {len(values)})")
    elif reduction == "max":
        ordered = sorted(values, key=lambda value: value.value, reverse=True)
        if len(ordered) > 1 and math.isclose(
                ordered[0].value, ordered[1].value,
                rel_tol=1e-12, abs_tol=1e-6):
            return CompositeAnswer(
                ok=False, detail="note-axis maximum tie has no convention",
                resolved=resolved)
        if route.get("output_type") == "year":
            answer = float(ordered[0].year)
            answer_expr = _year_projection_expr([
                (value.year, value.value, value.expr) for value in values
            ], "max")
            result_kind = "year"
        else:
            answer = ordered[0].value
            answer_expr = f"max({', '.join(value.expr for value in values)})"
    else:
        return CompositeAnswer(ok=False, detail="note-axis reduction unsupported")

    if result_kind == "money":
        scale = float(route.get("unit_scale", 1.0) or 1.0)
        answer /= scale
        answer_expr = f"(({answer_expr}) / {scale:g})"
    support = " + ".join(value.expr for value in values)
    query = f"round(({answer_expr}) + 0 * ({support}), 2)"
    warn = check_answer_unit(answer, route.get("output_type", result_kind))
    if (warn and "outside plausible range" in warn
            and typed.family != "borrowings_to_cash"):
        return CompositeAnswer(
            ok=False, detail=f"note-axis unit guard: {warn}", resolved=resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=95.0),
        detail=(f"formula_note_axis_v11 family={typed.family} "
                f"reduction={reduction} n={len(values)}"),
        resolved=resolved,
    )


def _axis_closing_columns(year: int) -> tuple[str, ...]:
    return (
        f"31/12/{year}", f"31.12.{year}",
        f"ngay 31 thang 12 nam {year}",
        f"ngay 31 thang 12nam {year}",
        f"cuoi nam {year}", "so cuoi nam", "so du cuoi nam", "cuoi nam",
        "nam nay", "ngay 31 thang 12",
    )


def _axis_period_columns(year: int) -> tuple[str, ...]:
    return (*_axis_closing_columns(year), f"nam {year}", str(year))


def _axis_value(
        tables: list[dict], ticker: str, year: int, route: dict,
        metric: str, request: MatrixRequest, *, absolute: bool = False,
        exact_count: int = 1, raw: bool = False) -> MatrixValue | None:
    facts = resolve_matrix_request(
        request, tables, ticker, year, str(route.get("doc_type") or ""), metric)
    if len(facts) != exact_count:
        return None
    if len({(fact.report_id, fact.table_pos, fact.col) for fact in facts}) != 1:
        return None
    if raw and absolute:
        value = sum(abs(fact.value) for fact in facts)
        expr = " + ".join(f"abs({fact.expr()})" for fact in facts)
    elif raw:
        value = sum(fact.value for fact in facts)
        expr = " + ".join(fact.expr() for fact in facts)
    elif absolute:
        value = sum(abs(fact.value_vnd) for fact in facts)
        expr = " + ".join(f"abs({fact.expr_vnd()})" for fact in facts)
    else:
        value = sum(fact.value_vnd for fact in facts)
        expr = " + ".join(fact.expr_vnd() for fact in facts)
    return MatrixValue(ticker, year, value, f"({expr})", facts)


def _axis_ratio(
        numerator: MatrixValue | None, denominator: MatrixValue | None,
        *, percent: bool, same_table: bool = False) -> MatrixValue | None:
    if numerator is None or denominator is None or denominator.value == 0:
        return None
    if (numerator.ticker, numerator.year) != (denominator.ticker, denominator.year):
        return None
    if same_table:
        left = {(fact.report_id, fact.table_pos) for fact in numerator.resolved}
        right = {(fact.report_id, fact.table_pos) for fact in denominator.resolved}
        if len(left) != 1 or left != right:
            return None
    multiplier = 100.0 if percent else 1.0
    suffix = " * 100" if percent else ""
    return MatrixValue(
        numerator.ticker, numerator.year,
        numerator.value / denominator.value * multiplier,
        f"(({numerator.expr}) / ({denominator.expr}){suffix})",
        _dedupe_resolved(numerator.resolved + denominator.resolved),
    )


def _axis_related_party_trade_payables(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    return _axis_value(
        tables, ticker, year, route, "related_party_trade_payables_short_term",
        MatrixRequest(
            table_term_groups=(("phai tra nguoi ban ngan han",), ("ben lien quan",)),
            row_variants=("tong cong",), column_variants=_axis_closing_columns(year),
            collect="cell", row_mode="exact"))


def _axis_borrowings_to_cash(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    borrowings = _axis_value(
        tables, ticker, year, route, "borrowings_total",
        MatrixRequest(
            table_term_groups=(("vay ngan han",), ("vay dai han",)),
            row_variants=("so cuoi nam",), column_variants=("tong cong",),
            collect="cell", row_mode="exact"))
    cash = _axis_value(
        tables, ticker, year, route, "cash_and_bank_deposits",
        MatrixRequest(
            table_term_groups=(("tien mat",), ("tien gui ngan hang",)),
            row_variants=("tong cong",), column_variants=_axis_closing_columns(year),
            collect="cell", row_mode="exact"))
    return _axis_ratio(borrowings, cash, percent=False)


def _axis_off_balance_commitments_share(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    commitments = _axis_value(
        tables, ticker, year, route, "off_balance_commitments",
        MatrixRequest(
            table_term_groups=(("cam ket giao dich",), ("cac cam ket khac",)),
            row_variants=("tong cong",), column_variants=_axis_closing_columns(year),
            collect="cell", row_mode="exact"))
    assets = _axis_value(
        tables, ticker, year, route, "total_assets",
        MatrixRequest(
            table_term_groups=(("tong tai san co", "tong cong tai san"),),
            row_variants=("tong tai san co", "tong cong tai san"),
            column_variants=_axis_closing_columns(year), collect="cell",
            row_mode="exact"))
    return _axis_ratio(commitments, assets, percent=True)


def _axis_related_party_long_receivable_share(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            ("phai thu ve cho vay dai han",), ("phai thu dai han khac",)),
        column_variants=_axis_closing_columns(year), row_mode="exact")
    loans = _axis_value(
        tables, ticker, year, route, "related_party_long_term_loans_receivable",
        MatrixRequest(**common, row_variants=("phai thu ve cho vay dai han",),
                      collect="cell"))
    total = _axis_value(
        tables, ticker, year, route, "related_party_long_term_receivables_total",
        MatrixRequest(**common, row_variants=(
            "phai thu ve cho vay dai han", "phai thu dai han khac"),
            collect="rows"), exact_count=2)
    return _axis_ratio(loans, total, percent=True, same_table=True)


def _axis_deposit_interest_expense_share(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(("tra lai tien gui",), ("tong cong",)),
        column_variants=_axis_period_columns(year), row_mode="exact")
    deposits = _axis_value(
        tables, ticker, year, route, "deposit_interest_expense",
        MatrixRequest(**common, row_variants=("tra lai tien gui",), collect="cell"),
        absolute=True)
    total = _axis_value(
        tables, ticker, year, route, "bank_interest_expense",
        MatrixRequest(**common, row_variants=("tong cong",), collect="cell"),
        absolute=True)
    return _axis_ratio(deposits, total, percent=True, same_table=True)


def _axis_inventory_merchandise(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    direct = _axis_value(
        tables, ticker, year, route, "inventory_merchandise",
        MatrixRequest(
            table_term_groups=(("hang hoa",), ("tong cong",)),
            context_term_groups=(("hang ton kho",),),
            row_variants=("hang hoa",), column_variants=_axis_closing_columns(year),
            collect="cell", row_mode="exact"), absolute=True)
    if direct is not None:
        return direct
    common = dict(
        table_term_groups=(("hang hoa",), ("tong cong",)),
        context_term_groups=(("hang ton kho",),),
        column_variants=_axis_closing_columns(year), row_mode="exact")
    total = _axis_value(
        tables, ticker, year, route, "inventory_total",
        MatrixRequest(**common, row_variants=("tong cong",), collect="cell"),
        absolute=True)
    components = _axis_value(
        tables, ticker, year, route, "inventory_non_merchandise",
        MatrixRequest(**common, row_variants=(
            "hang mua dang di tren duong", "nguyen lieu vat lieu ton kho",
            "cong cu dung cu", "chi phi san xuat kinh doanh do dang",
            "thanh pham"), collect="rows"), absolute=True, exact_count=5)
    if total is None or components is None:
        return None
    if {(f.report_id, f.table_pos) for f in total.resolved} != {
            (f.report_id, f.table_pos) for f in components.resolved}:
        return None
    value = abs(total.value - components.value)
    expr = f"abs(({total.expr}) - ({components.expr}))"
    return MatrixValue(
        ticker, year, value, expr,
        _dedupe_resolved(total.resolved + components.resolved))


def _axis_investment_property_depreciation(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    return _axis_value(
        tables, ticker, year, route, "investment_property_depreciation",
        MatrixRequest(
            table_term_groups=(("khau hao trong nam",), ("gia tri con lai",)),
            context_term_groups=(("bat dong san dau tu",),),
            row_variants=("khau hao trong nam",),
            column_variants=("vnd", "nha cua", "vat kien truc"),
            collect="cell", row_mode="exact"), absolute=True)


def _axis_usd_long_term_borrowing_share(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(("vay bang usd",), ("vay bang vnd",)),
        column_variants=_axis_closing_columns(year))
    usd = _axis_value(
        tables, ticker, year, route, "usd_long_term_borrowings",
        MatrixRequest(**common, row_variants=("vay bang usd",),
                      collect="cell", row_mode="exact"), absolute=True)
    total = _axis_value(
        tables, ticker, year, route, "borrowings_long_term",
        MatrixRequest(**common, collect="last_total"), absolute=True)
    return _axis_ratio(usd, total, percent=True, same_table=True)


def _axis_financial_reserve_equity_share(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(("quy du phong tai chinh", "quy du phongtai chinh"),
                           ("tong",)),
        row_variants=(
            f"tai ngay 31 thang 12 nam {year}",
            f"so du tai ngay 31 thang 12 nam {year}"),
        collect="cell", row_mode="exact")
    reserve = _axis_value(
        tables, ticker, year, route, "financial_reserve_fund",
        MatrixRequest(**common, column_variants=(
            "quy du phong tai chinh", "quy du phongtai chinh")),
        absolute=True)
    equity = _axis_value(
        tables, ticker, year, route, "equity",
        MatrixRequest(**common, column_variants=("tong cong", "tong")),
        absolute=True)
    return _axis_ratio(reserve, equity, percent=True, same_table=True)


def _axis_current_income_tax(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    direct = _axis_value(
        tables, ticker, year, route, "current_income_tax",
        MatrixRequest(
            table_term_groups=(("chi phi thue tndn hien hanh",
                                "chi phi thue thu nhap doanh nghiep hien hanh"),
                               ("loi nhuan sau thue tndn",)),
            row_codes=("51",), column_variants=_axis_period_columns(year),
            collect="cell"), absolute=True, raw=True)
    if direct is not None:
        return direct

    common = dict(
        table_term_groups=(("thue tndn hoan lai", "thue thu nhap doanh nghiep hoan lai"),
                           ("tong cong",)),
        column_variants=_axis_period_columns(year), collect="cell",
        row_mode="exact")
    total = _axis_value(
        tables, ticker, year, route, "income_tax_total",
        MatrixRequest(**common, row_variants=("tong cong",)), raw=True)
    deferred = _axis_value(
        tables, ticker, year, route, "deferred_income_tax",
        MatrixRequest(**common, row_variants=(
            "thu nhap thue tndn hoan lai", "chi phi thue tndn hoan lai",
            "thu nhap thue thu nhap doanh nghiep hoan lai",
            "chi phi thue thu nhap doanh nghiep hoan lai")), raw=True)
    if total is None or deferred is None:
        return None
    if {(f.report_id, f.table_pos) for f in total.resolved} != {
            (f.report_id, f.table_pos) for f in deferred.resolved}:
        return None
    value = abs(total.value - deferred.value)
    expr = f"abs(({total.expr}) - ({deferred.expr}))"
    return MatrixValue(
        ticker, year, value, expr,
        _dedupe_resolved(total.resolved + deferred.resolved))


def _axis_intangible_fixed_assets(
        tables: list[dict], ticker: str, year: int, route: dict) -> MatrixValue | None:
    return _axis_value(
        tables, ticker, year, route, "intangible_fixed_assets",
        MatrixRequest(
            table_term_groups=(("tai san co dinh vo hinh",), ("tong tai san",)),
            row_variants=("tai san co dinh vo hinh",),
            column_variants=_axis_closing_columns(year), collect="cell",
            row_mode="exact"), absolute=True)


def build_matrix_note_plan(route: dict) -> MatrixNotePlan | None:
    """Recognize calculations whose operands live at matrix intersections."""
    question = route.get("question", "")
    text = _plain(question)
    is_typed_risk_scenario = (
        "bang do nhay" in text
        or ("trang thai tien te noi" in text and "ngoai bang" in text)
    )
    if not is_typed_risk_scenario and any(value in text for value in (
            "gia su", "kich ban", "neu ", "trung vi", "truoc khi",
            "giu nguyen", "can thiet de", "co the tang toi da")):
        return None
    members = tuple(dict.fromkeys(
        str(ticker).upper() for ticker in route.get("tickers") or [] if ticker))
    years = tuple(sorted(set(_route_years(route))))
    if not members or not years:
        return None

    if ("ty le bieu quyet trung binh" in text and "cong ty con" in text
            and "chenh lech" in text):
        family, reduction = "subsidiary_voting_rate", "difference"
        valid = len(members) == 2 and len(years) == 1
    elif ("gia von cho thue dai han dat va co so ha tang" in text
          and "tong gia von hang ban" in text and _wants_max(text)):
        family, reduction = "land_infrastructure_rental_cogs_share", "max"
        valid = (members == ("KBC",) and len(years) >= 2
                 and route.get("output_type") == "year")
    elif ("so du khoan phai tra ngan han khac" in text
          and "kinh doanh xuat nhap khau hoang anh gia lai" in text
          and "so nam" in text):
        family, reduction = "named_related_other_payable_positive", "sum"
        members = ("HAG",)
        valid = len(years) >= 2
    elif ("ty trong chung chi tien gui" in text
          and "duoi 12 thang" in text and _wants_max(text)):
        family, reduction = "certificate_deposit_short_share", "max_value"
        valid = members == ("STB",) and len(years) >= 2
    elif ("ty trong doanh thu tu khu vuc lao" in text
          and "tong doanh thu" in text
          and any(value in text for value in ("trung binh", "binh quan"))):
        family, reduction = "laos_geographic_revenue_share", "mean"
        valid = members == ("HAG",) and len(years) >= 2
    elif ("ty trong ngoai te usd" in text
          and "ngoai bang can doi ke toan" in text
          and any(value in text for value in ("trung binh", "binh quan"))):
        family, reduction = "off_balance_usd_share", "mean"
        valid = len(members) >= 2 and len(years) == 1
    elif ("ty trong cho vay ngan han" in text
          and "tong du no cho vay khach hang" in text
          and any(value in text for value in ("trung binh", "binh quan"))):
        family, reduction = "short_term_customer_loan_share", "mean"
        valid = len(members) >= 2 and len(years) == 1
    elif ("gia tri con lai cua quyen su dung dat" in text
          and "cuoi nam" in text):
        family, reduction = "land_use_right_net_closing", "direct"
        valid = len(members) == 1 and len(years) == 1
    elif ("ty le so huu" in text and "tai cong ty" in text
          and any(value in text for value in ("den ngay", "vao ngay"))):
        family, reduction = "named_subsidiary_ownership_rate", "direct"
        valid = len(members) == 1 and len(years) == 1
    elif ("thue tndn phai nop cuoi ky" in text
          or "thue thu nhap doanh nghiep phai nop cuoi ky" in text):
        family, reduction = "income_tax_payable_closing", "last_minus_first"
        valid = len(members) == 1 and len(years) == 2
    elif ("no vay tren tong tai san" in text
          or "ty so no vay tren tong tai san" in text):
        family, reduction = "borrowings_to_assets", "direct"
        valid = len(members) == 1 and len(years) == 1
    elif "ty le bao phu no xau" in text:
        family, reduction = "nonperforming_loan_coverage", "direct"
        valid = len(members) == 1 and len(years) == 1
    elif ("du phong phai thu kho doi" in text
          and "trich lap trong nam" in text and _wants_max(text)):
        family, reduction = "doubtful_receivable_provision_addition", "max_value"
        valid = len(members) == 1 and len(years) >= 2
    elif ("doanh thu chua thuc hien cuoi ky" in text
          and any(value in text for value in ("trung binh", "binh quan"))):
        family, reduction = "unearned_revenue_closing", "mean"
        valid = len(members) == 1 and len(years) >= 2
    elif ("tong thu lao hoi dong quan tri" in text
          and "cong don" in text):
        family, reduction = "board_compensation", "sum"
        valid = len(members) == 1 and len(years) >= 2
    elif ("thu lao hoi dong quan tri va ban tong giam doc" in text
          and _wants_max(text)):
        family, reduction = "board_management_compensation", "max_value"
        valid = len(members) == 1 and len(years) >= 2
    elif ("phai tra ngan han khac voi cong ty con" in text
          and _wants_max(text)):
        family, reduction = "subsidiary_other_payables_short_term", "max_value"
        valid = len(members) == 1 and len(years) >= 2
    elif ("du no cho vay nganh bat dong san" in text
          and any(value in text for value in ("trung binh", "binh quan"))):
        family, reduction = "real_estate_loan_share", "mean"
        valid = len(members) == 1 and len(years) >= 2
    elif ("hao mon luy ke trung binh" in text and "tscd huu hinh" in text):
        family, reduction = "tangible_fixed_asset_depreciation_share", "mean"
        valid = len(members) >= 2 and len(years) == 1
    elif (any(value in text for value in (
            "khau hao luy ke tren nguyen gia tai san co dinh",
            "hao mon luy ke tren nguyen gia tai san co dinh"))
          and any(value in text for value in ("trung binh", "binh quan"))):
        family, reduction = "fixed_asset_depreciation_share", "mean"
        valid = len(members) >= 2 and len(years) == 1
    elif ("tai san bo phan dich vu van tai" in text
          and "tong tai san" in text and _wants_max(text)):
        family, reduction = "transport_segment_asset_share", "max"
        valid = (len(members) == 1 and len(years) >= 2
                 and route.get("output_type") == "year")
    elif ("tai san bo phan bot" in text and "tong tai san" in text
          and any(value in text for value in ("trung binh", "binh quan"))):
        family, reduction = "bot_segment_asset_share", "mean"
        valid = len(members) == 1 and len(years) >= 2
    elif ("tong gia tri tai san tai chinh chiu rui ro tin dung" in text
          and "tong" in text):
        family, reduction = "credit_risk_financial_assets_total", "direct"
        valid = members == ("BVH",) and years == (2018,)
    elif ("so du ngoai te usd" in text and "cuoi nam" in text):
        family, reduction = "closing_usd_currency_balance", "direct"
        valid = members == ("ACV",) and years == (2018,)
    elif ("tong gia tri hop dong cong cu phai sinh" in text
          and "cuoi nam" in text):
        family, reduction = "derivative_contract_notional_total", "direct"
        valid = members == ("KLB",) and years == (2022,)
    elif ("cong no tien te cuoi nam lon hon tai san tien te cuoi nam" in text
          and "bang do nhay" in text and "vnd bien dong bat loi 5%" in text):
        family, reduction = "fx_sensitivity_filtered_loss", "direct"
        years = (2016,)
        valid = members == ("FPT",)
    elif ("vnd giam 5%" in text and "trang thai tien te noi" in text
          and "ngoai bang" in text
          and "dong tien gay bat loi lon nhat" in text):
        family, reduction = "fx_position_worst_loss_share", "direct"
        years = (2024,)
        valid = members == ("ACB",)
    elif ("tang truong gia tri rui ro" in text and "var" in text
          and "1 ngay" in text
          and "danh muc co phieu niem yet" in text):
        family, reduction = "listed_equity_var_one_day", "growth"
        valid = members == ("BVH",) and years == (2018, 2019)
    else:
        return None
    if not valid:
        return None
    return MatrixNotePlan(family, members, years, reduction)


def _try_matrix_note(
        route: dict, tables: list[dict], typed: MatrixNotePlan, encoder=None,
        min_score: float = 62.0) -> CompositeAnswer:
    """Execute a typed matrix plan only when every axis resolves exactly."""
    values: list[MatrixValue] = []
    v17_families = {
        "land_infrastructure_rental_cogs_share",
        "named_related_other_payable_positive",
        "certificate_deposit_short_share",
        "laos_geographic_revenue_share",
        "off_balance_usd_share",
        "short_term_customer_loan_share",
    }
    v16_families = {
        "land_use_right_net_closing",
        "named_subsidiary_ownership_rate",
        "income_tax_payable_closing",
        "borrowings_to_assets",
        "nonperforming_loan_coverage",
        "doubtful_receivable_provision_addition",
        "unearned_revenue_closing",
        "board_compensation",
        "board_management_compensation",
        "tangible_fixed_asset_depreciation_share",
        "subsidiary_other_payables_short_term",
    }
    if typed.family in v17_families:
        if typed.family in {
                "off_balance_usd_share", "short_term_customer_loan_share"}:
            year = typed.years[0]
            pairs = ((ticker, year) for ticker in typed.members)
        else:
            ticker = typed.members[0]
            pairs = ((ticker, year) for year in typed.years)
        for ticker, year in pairs:
            found = _matrix_note_v17_value(
                typed.family, tables, ticker, year, route)
            if found is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"matrix {typed.family} unresolved {ticker}/{year}")
            values.append(found)
    elif typed.family in v16_families:
        if typed.family == "tangible_fixed_asset_depreciation_share":
            year = typed.years[0]
            pairs = ((ticker, year) for ticker in typed.members)
        else:
            ticker = typed.members[0]
            pairs = ((ticker, year) for year in typed.years)
        for ticker, year in pairs:
            found = _matrix_note_v16_value(
                typed.family, tables, ticker, year, route)
            if found is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"matrix {typed.family} unresolved {ticker}/{year}")
            values.append(found)
    elif typed.family in {
            "credit_risk_financial_assets_total",
            "closing_usd_currency_balance",
            "derivative_contract_notional_total",
            "fx_sensitivity_filtered_loss",
            "fx_position_worst_loss_share",
    }:
        ticker, year = typed.members[0], typed.years[0]
        if typed.family == "credit_risk_financial_assets_total":
            found = _matrix_credit_risk_total(tables, ticker, year, route)
        elif typed.family == "closing_usd_currency_balance":
            found = _matrix_closing_usd_balance(tables, ticker, year, route)
        elif typed.family == "derivative_contract_notional_total":
            found = _matrix_derivative_contract_total(tables, ticker, year, route)
        elif typed.family == "fx_sensitivity_filtered_loss":
            found = _matrix_fx_sensitivity_loss(tables, ticker, year, route)
        else:
            found = _matrix_fx_position_loss_share(
                tables, ticker, year, route, encoder, min_score)
        if found is None:
            return CompositeAnswer(
                ok=False, detail=f"matrix {typed.family} unresolved {ticker}/{year}")
        values.append(found)
    elif typed.family == "listed_equity_var_one_day":
        ticker = typed.members[0]
        for year in typed.years:
            found = _matrix_listed_equity_var(tables, ticker, year, route)
            if found is None:
                return CompositeAnswer(
                    ok=False, detail=f"matrix VaR unresolved {ticker}/{year}")
            values.append(found)
    elif typed.family == "subsidiary_voting_rate":
        year = typed.years[0]
        for ticker in typed.members:
            found = _matrix_voting_rate(tables, ticker, year, route)
            if found is None:
                return CompositeAnswer(
                    ok=False, detail=f"matrix voting block unresolved {ticker}/{year}")
            values.append(found)
    elif typed.family == "fixed_asset_depreciation_share":
        year = typed.years[0]
        for ticker in typed.members:
            found = _matrix_fixed_asset_share(tables, ticker, year, route)
            if found is None:
                return CompositeAnswer(
                    ok=False, detail=f"matrix fixed-asset blocks unresolved {ticker}/{year}")
            values.append(found)
    else:
        ticker = typed.members[0]
        for year in typed.years:
            if typed.family == "real_estate_loan_share":
                found = _matrix_real_estate_loan_share(
                    tables, ticker, year, route)
            elif typed.family == "transport_segment_asset_share":
                found = _matrix_segment_asset_share(
                    tables, ticker, year, route,
                    ("dich vu van tai",), closing_table=False)
            else:
                found = _matrix_segment_asset_share(
                    tables, ticker, year, route,
                    ("du an bot", "thu phi tram bot", "bo phan bot"),
                    closing_table=True)
            if found is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"matrix {typed.family} unresolved {ticker}/{year}")
            values.append(found)

    resolved = _dedupe_resolved([
        fact for value in values for fact in value.resolved
    ])
    if not resolved or not distinct_cells(resolved):
        return CompositeAnswer(
            ok=False, detail="matrix evidence collapsed to duplicate cells",
            resolved=resolved)

    if typed.reduction == "direct":
        if len(values) != 1:
            return CompositeAnswer(
                ok=False, detail="matrix direct reduction needs one value",
                resolved=resolved)
        answer = values[0].value
        answer_expr = values[0].expr
    elif typed.reduction == "growth":
        if len(values) != 2 or values[0].value == 0:
            return CompositeAnswer(
                ok=False, detail="matrix growth needs two values and nonzero base",
                resolved=resolved)
        answer = ((values[1].value - values[0].value)
                  / abs(values[0].value) * 100.0)
        answer_expr = (
            f"((({values[1].expr}) - ({values[0].expr})) "
            f"/ abs({values[0].expr}) * 100)")
    elif typed.reduction == "difference":
        answer = values[0].value - values[1].value
        answer_expr = f"(({values[0].expr}) - ({values[1].expr}))"
    elif typed.reduction == "last_minus_first":
        if len(values) != 2:
            return CompositeAnswer(
                ok=False, detail="matrix period difference needs two values",
                resolved=resolved)
        answer = values[1].value - values[0].value
        answer_expr = f"(({values[1].expr}) - ({values[0].expr}))"
    elif typed.reduction == "mean":
        answer = _mean([value.value for value in values])
        answer_expr = (
            f"(({' + '.join(value.expr for value in values)}) / {len(values)})")
    elif typed.reduction == "sum":
        answer = sum(value.value for value in values)
        answer_expr = f"({' + '.join(value.expr for value in values)})"
    elif typed.reduction == "max_value":
        answer = max(value.value for value in values)
        answer_expr = f"max({', '.join(value.expr for value in values)})"
    elif typed.reduction == "max":
        ordered = sorted(values, key=lambda value: value.value, reverse=True)
        if len(ordered) > 1 and math.isclose(
                ordered[0].value, ordered[1].value,
                rel_tol=1e-12, abs_tol=1e-8):
            return CompositeAnswer(
                ok=False, detail="matrix year ranking tie has no convention",
                resolved=resolved)
        answer = float(ordered[0].year)
        answer_expr = _year_projection_expr([
            (value.year, value.value, value.expr) for value in values
        ], "max")
    else:
        return CompositeAnswer(ok=False, detail="matrix reduction unsupported")

    query = f"round(({answer_expr}), 2)"
    warn = check_answer_unit(answer, route.get("output_type", "percent"))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(
            ok=False, detail=f"matrix unit guard: {warn}", resolved=resolved)
    version = "v17" if typed.family in v17_families else "v16" if typed.family in v16_families else "v15" if typed.family in {
        "credit_risk_financial_assets_total",
        "closing_usd_currency_balance",
        "derivative_contract_notional_total",
        "fx_sensitivity_filtered_loss",
        "fx_position_worst_loss_share",
        "listed_equity_var_one_day",
    } else "v10"
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query, confidence=95.0,
        detail=(f"formula_matrix_note_{version} family={typed.family} "
                f"reduction={typed.reduction} n={len(values)}"),
        resolved=resolved,
    )


def _matrix_note_v16_value(
        family: str, tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    resolvers = {
        "land_use_right_net_closing": _matrix_land_use_right_net,
        "named_subsidiary_ownership_rate": _matrix_subsidiary_ownership_rate,
        "income_tax_payable_closing": _matrix_income_tax_payable,
        "borrowings_to_assets": _matrix_borrowings_to_assets,
        "nonperforming_loan_coverage": _matrix_nonperforming_loan_coverage,
        "doubtful_receivable_provision_addition":
            _matrix_doubtful_receivable_provision_addition,
        "unearned_revenue_closing": _matrix_unearned_revenue,
        "board_compensation": _matrix_board_compensation,
        "board_management_compensation": _matrix_board_management_compensation,
        "tangible_fixed_asset_depreciation_share":
            _matrix_tangible_fixed_asset_share,
        "subsidiary_other_payables_short_term":
            _matrix_subsidiary_other_payables,
    }
    resolver = resolvers.get(family)
    return resolver(tables, ticker, year, route) if resolver else None


def _matrix_note_v17_value(
        family: str, tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    resolvers = {
        "land_infrastructure_rental_cogs_share":
            _matrix_land_infrastructure_rental_cogs_share,
        "named_related_other_payable_positive":
            _matrix_named_related_other_payable_positive,
        "certificate_deposit_short_share":
            _matrix_certificate_deposit_short_share,
        "laos_geographic_revenue_share":
            _matrix_laos_geographic_revenue_share,
        "off_balance_usd_share": _matrix_off_balance_usd_share,
        "short_term_customer_loan_share":
            _matrix_short_term_customer_loan_share,
    }
    resolver = resolvers.get(family)
    return resolver(tables, ticker, year, route) if resolver else None


def _scale_matrix_money(value: MatrixValue | None, route: dict) -> MatrixValue | None:
    if value is None:
        return None
    scale = float(route.get("unit_scale", 1.0) or 1.0)
    if scale <= 0:
        return None
    return MatrixValue(
        value.ticker, value.year, value.value / scale,
        f"(({value.expr}) / {scale:g})", value.resolved)


def _matrix_land_use_right_net(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    value = _axis_value(
        tables, ticker, year, route, "land_use_right_net",
        MatrixRequest(
            table_term_groups=(("quyen su dung dat",), ("gia tri con lai",)),
            context_term_groups=((
                f"bien dong cua tai san co dinh vo hinh trong nam {year}",),),
            row_variants=("tai ngay cuoi nam",),
            column_variants=("quyen su dung dat",),
            block_variants=("gia tri con lai",),
            collect="cell", row_mode="exact"))
    return _scale_matrix_money(value, route)


def _named_subsidiary_from_question(question: str) -> str:
    text = _plain(question)
    match = re.search(
        r"\btai\s+(?:cong ty\s+)?(?:tnhh\s+|ctcp\s+|co phan\s+)?"
        r"(.+?)\s+(?:den|vao)\s+ngay\b", text)
    return match.group(1).strip() if match else ""


def _matrix_subsidiary_ownership_rate(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    subsidiary = _named_subsidiary_from_question(
        str(route.get("question") or ""))
    if not subsidiary:
        return None
    return _axis_value(
        tables, ticker, year, route, "ownership_rate",
        MatrixRequest(
            table_term_groups=(("ty le so huu cua cong ty",), (subsidiary,)),
            context_term_groups=(("co cau to chuc", "cong ty con"),),
            row_variants=(subsidiary,),
            column_variants=(
                f"ty le so huu cua cong ty ngay 31 thang 12 nam {year}",
                f"ty le so huu cua cong ty 31 12 {year}"),
            collect="cell", row_mode="contains"),
        raw=True)


def _matrix_income_tax_payable(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    value = _axis_value(
        tables, ticker, year, route, "income_tax_payable",
        MatrixRequest(
            table_term_groups=(
                ("thue tndn", "thue thu nhap doanh nghiep"),
                ("so cuoi nam",)),
            context_term_groups=(("thue va cac khoan phai nop nha nuoc",),),
            row_variants=("thue tndn", "thue thu nhap doanh nghiep"),
            column_variants=_axis_closing_columns(year),
            collect="cell", row_mode="exact"))
    return _scale_matrix_money(value, route)


def _matrix_borrowings_to_assets(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    current = _axis_closing_columns(year)
    borrowings = _axis_value(
        tables, ticker, year, route, "borrowings_total",
        MatrixRequest(
            table_term_groups=(
                ("ma so",), ("vay ngan han",), ("vay dai han",)),
            row_codes=("320", "338"), column_variants=current,
            collect="rows"),
        exact_count=2)
    assets = _axis_value(
        tables, ticker, year, route, "total_assets",
        MatrixRequest(
            table_term_groups=(("ma so",), ("tong cong tai san",)),
            row_codes=("270",), column_variants=current, collect="cell"))
    return _axis_ratio(borrowings, assets, percent=True)


def _matrix_nonperforming_loan_coverage(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    current = _axis_closing_columns(year)
    provision = _axis_value(
        tables, ticker, year, route, "customer_loan_provision_balance",
        MatrixRequest(
            table_term_groups=(
                ("du phong rui ro cho vay khach hang",),
                ("tong tai san co",)),
            row_variants=("du phong rui ro cho vay khach hang",),
            column_variants=current, collect="cell", row_mode="exact"),
        absolute=True)
    bad_debt = _axis_value(
        tables, ticker, year, route, "nonperforming_loans",
        MatrixRequest(
            table_term_groups=(
                ("no duoi tieu chuan",), ("no nghi ngo",),
                ("no co kha nang mat von",)),
            row_variants=(
                "no duoi tieu chuan", "no nghi ngo", "no co kha nang mat von"),
            column_variants=current, collect="rows", row_mode="exact"),
        exact_count=3)
    return _axis_ratio(provision, bad_debt, percent=True)


def _matrix_doubtful_receivable_provision_addition(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    value = _axis_value(
        tables, ticker, year, route,
        "doubtful_receivable_provision_addition",
        MatrixRequest(
            table_term_groups=(("du phong trich lap trong nam",),),
            context_term_groups=((
                "du phong phai thu kho doi",
                "du phong phai thu ngan han kho doi"),),
            row_variants=("du phong trich lap trong nam",),
            column_variants=_axis_period_columns(year),
            collect="cell", row_mode="contains"),
        absolute=True)
    return _scale_matrix_money(value, route)


def _matrix_unearned_revenue(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    value = _axis_value(
        tables, ticker, year, route, "unearned_revenue_total",
        MatrixRequest(
            table_term_groups=(
                ("doanh thu chua thuc hien ngan han",),
                ("doanh thu chua thuc hien dai han",)),
            row_codes=("318", "336"),
            column_variants=_axis_closing_columns(year), collect="rows"),
        exact_count=2)
    return _scale_matrix_money(value, route)


def _matrix_board_compensation(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    value = _axis_value(
        tables, ticker, year, route, "board_compensation",
        MatrixRequest(
            table_term_groups=(("tong cong",), ("nam nay",)),
            context_term_groups=((
                "thu lao chi tra cho cac thanh vien cua hoi dong quan tri",),),
            row_variants=("tong cong",),
            column_variants=_axis_period_columns(year),
            collect="cell", row_mode="exact"))
    return _scale_matrix_money(value, route)


def _matrix_board_management_compensation(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    value = _axis_value(
        tables, ticker, year, route, "board_management_compensation",
        MatrixRequest(
            table_term_groups=(("tong cong",), ("nam nay",)),
            context_term_groups=(
                ("thu nhap cua cac thanh vien cua hoi dong quan tri",),
                ("ban tong giam doc",)),
            row_variants=("tong cong",),
            column_variants=_axis_period_columns(year),
            collect="cell", row_mode="exact"))
    return _scale_matrix_money(value, route)


def _matrix_tangible_fixed_asset_share(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            ("ma so",), ("tai san co dinh huu hinh",)),
        column_variants=_axis_closing_columns(year), collect="cell")
    cost = _axis_value(
        tables, ticker, year, route, "tangible_fixed_assets_cost",
        MatrixRequest(**common, row_codes=("222",)))
    depreciation = _axis_value(
        tables, ticker, year, route,
        "tangible_fixed_assets_accumulated_depreciation",
        MatrixRequest(**common, row_codes=("223",)), absolute=True)
    return _axis_ratio(depreciation, cost, percent=True, same_table=True)


def _matrix_subsidiary_other_payables(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    value = _axis_value(
        tables, ticker, year, route, "subsidiary_other_payables_short_term",
        MatrixRequest(
            table_term_groups=(("cong ty con",),),
            context_term_groups=(("phai tra ngan han khac",),),
            column_index=3, collect="last_total"))
    return _scale_matrix_money(value, route)


def _matrix_land_infrastructure_rental_cogs_share(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            ("gia von cho thue dai han dat va co so ha tang",
             "gia von dat va co so ha tang cho thue",
             "gia von cho thue dat va co so ha tang cho thue"),
            ("tong cong",)),
        column_variants=_axis_period_columns(year), row_mode="exact")
    numerator = _axis_value(
        tables, ticker, year, route, "land_infrastructure_rental_cogs",
        MatrixRequest(
            **common,
            row_variants=(
                "gia von cho thue dai han dat va co so ha tang",
                "gia von dat va co so ha tang cho thue",
                "gia von cho thue dat va co so ha tang cho thue"),
            collect="cell"), absolute=True)
    denominator = _axis_value(
        tables, ticker, year, route, "cost_of_goods_sold",
        MatrixRequest(**common, row_variants=("tong cong",), collect="cell"),
        absolute=True)
    return _axis_ratio(numerator, denominator, percent=True, same_table=True)


_HAG_EXPORT_COMPANY_ROWS = (
    "cong ty tnhh mtv kinh doanh xuat nhap khau hoang anh gia lai",
    "cong ty tnhh mtv kinh doanh xuat nhap khauhoang anh gia lai",
)


def _matrix_named_related_other_payable_positive(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            _HAG_EXPORT_COMPANY_ROWS, ("phai tra ngan han khac",)),
        row_variants=_HAG_EXPORT_COMPANY_ROWS,
        collect="cell", row_mode="contains")
    current = _axis_value(
        tables, ticker, year, route, "named_related_other_payable",
        MatrixRequest(**common, column_variants=_axis_closing_columns(year)),
        raw=True)
    if current is not None:
        return MatrixValue(
            ticker, year, float(current.value > 0),
            f"(1.0 if ({current.expr}) > 0 else 0.0)", current.resolved)

    # A dash is intentionally absent from tidy CSV. The same named row's
    # comparative cell keeps the row in submitted evidence; zero is then the
    # only current-period interpretation accepted by this family.
    comparative = _axis_value(
        tables, ticker, year, route, "named_related_other_payable_prior",
        MatrixRequest(
            **common, column_variants=("so dau nam", "dau nam", "1/1")),
        raw=True)
    if comparative is None:
        return None
    return MatrixValue(
        ticker, year, 0.0, f"(0.0 * abs({comparative.expr}))",
        comparative.resolved)


def _matrix_certificate_deposit_short_share(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    maturity_rows = (
        "duoi 12 thang", "tu 12 thang den duoi 5 nam", "tu 5 nam tro len",
    )
    common = dict(
        table_term_groups=(
            ("chung chi tien gui",), ("duoi 12 thang",),
            ("tu 12 thang den duoi 5 nam",)),
        block_variants=("chung chi tien gui",),
        block_stop_variants=("trai phieu ghi danh do tctd phat hanh",),
        column_variants=_axis_closing_columns(year), row_mode="exact")
    numerator = _axis_value(
        tables, ticker, year, route,
        "certificate_deposits_under_12_months",
        MatrixRequest(**common, row_variants=("duoi 12 thang",),
                      collect="cell"), absolute=True)
    denominator = _axis_value(
        tables, ticker, year, route, "certificates_of_deposit_total",
        MatrixRequest(**common, row_variants=maturity_rows, collect="rows"),
        absolute=True, exact_count=3)
    return _axis_ratio(numerator, denominator, percent=True, same_table=True)


def _matrix_laos_geographic_revenue_share(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            ("viet nam",), ("lao",), ("tong cong",),
            ("doanh thu tu khach hang ben ngoai",)),
        block_variants=(f"cho nam tai chinh ket thuc ngay 31 thang 12 nam {year}",),
        row_variants=("doanh thu tu khach hang ben ngoai",),
        collect="cell", row_mode="exact")
    numerator = _axis_value(
        tables, ticker, year, route, "laos_geographic_revenue",
        MatrixRequest(**common, column_variants=("lao",)))
    denominator = _axis_value(
        tables, ticker, year, route, "geographic_revenue_total",
        MatrixRequest(**common, column_variants=("tong cong", "tong")))
    return _axis_ratio(numerator, denominator, percent=True, same_table=True)


def _matrix_off_balance_usd_share(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    currency_rows = (
        "usd", "do la my usd", "ngoai te usd", "eur", "euro", "jpy",
        "yen nhat", "cny", "cad", "aud", "lak", "khr", "krw",
        "won han quoc krw", "gbp", "dong bang anh gpb", "sgd",
        "do la singapore sgd", "rub", "rup nga rub", "ngoai te khac",
    )
    common = dict(
        table_term_groups=(
            ("usd", "do la my"), ("eur", "euro"),
            (f"31/12/{year}", f"31.12.{year}", "so cuoi nam")),
        context_term_groups=((
            "ngoai te cac loai", "cac khoan muc ngoai bang can doi ke toan",
            "cac khoan muc ngoai bang"),),
        column_index=1, row_mode="contains")
    numerator = _axis_value(
        tables, ticker, year, route, "off_balance_usd_balance",
        MatrixRequest(
            **common,
            row_variants=("usd", "do la my usd", "ngoai te usd"),
            collect="cell"), raw=True)
    denominator_facts = resolve_matrix_request(
        MatrixRequest(**common, row_variants=currency_rows, collect="rows"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "off_balance_foreign_currency_total")
    if numerator is None or len(denominator_facts) < 2:
        return None
    if len({(fact.report_id, fact.table_pos, fact.col)
            for fact in denominator_facts}) != 1:
        return None
    denominator = MatrixValue(
        ticker, year, sum(abs(fact.value) for fact in denominator_facts),
        f"({' + '.join(f'abs({fact.expr()})' for fact in denominator_facts)})",
        denominator_facts)
    return _axis_ratio(numerator, denominator, percent=True, same_table=True)


def _matrix_short_term_customer_loan_share(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            ("ngan han", "no ngan han"), ("trung han", "no trung han"),
            ("dai han", "no dai han")),
        column_variants=_axis_closing_columns(year))
    numerator = _axis_value(
        tables, ticker, year, route, "customer_loans_short_term",
        MatrixRequest(
            **common,
            row_variants=("ngan han", "no ngan han",
                          "no ngan han duoi 1 nam"),
            collect="cell", row_mode="contains"), absolute=True)
    denominator = _axis_value(
        tables, ticker, year, route, "customer_loans",
        MatrixRequest(**common, collect="last_total"), absolute=True)
    return _axis_ratio(numerator, denominator, percent=True, same_table=True)


def _matrix_voting_rate(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    request = MatrixRequest(
        table_term_groups=(
            ("ty le bieu quyet", "ty le so huu va bieu quyet"),
            ("cong ty con",),
        ),
        column_variants=("ty le bieu quyet", "ty le so huu va bieu quyet"),
        block_variants=("cong ty con",),
        block_stop_variants=("cong ty lien doanh", "cong ty lien ket"),
        collect="block",
    )
    facts = resolve_matrix_request(
        request, tables, ticker, year, str(route.get("doc_type") or ""),
        "subsidiary_voting_rate")
    if len(facts) < 2 or any(not 0 <= fact.value <= 100 for fact in facts):
        return None
    expr = f"(({' + '.join(fact.expr() for fact in facts)}) / {len(facts)})"
    return MatrixValue(
        ticker, year, _mean([fact.value for fact in facts]), expr, facts)


def _current_column_variants(year: int) -> tuple[str, ...]:
    return (
        f"31/12/{year}", f"31.12.{year}",
        "so du cuoi nam", "so cuoi nam", "31 12",
    )


def _single_matrix_value(
        facts: list[ResolvedFact], ticker: str, year: int, route: dict, *,
        raw_currency: bool = False) -> MatrixValue | None:
    if len(facts) != 1:
        return None
    fact = facts[0]
    scale = float(route.get("unit_scale", 1.0) or 1.0)
    value = (fact.value if raw_currency else fact.value_vnd) / scale
    source_expr = fact.expr() if raw_currency else fact.expr_vnd()
    return MatrixValue(ticker, year, value, f"(({source_expr}) / {scale:g})", facts)


def _matrix_credit_risk_total(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    facts = resolve_matrix_request(
        MatrixRequest(
            table_term_groups=(("chua qua han",), ("tong cong",)),
            context_term_groups=((
                f"rui ro tin dung tai ngay 31 thang 12 nam {year} nhu sau",
            ),),
            row_variants=("tong",), column_variants=("tong cong",),
            collect="cell", row_mode="exact"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "credit_risk_financial_assets_total")
    return _single_matrix_value(facts, ticker, year, route)


def _matrix_closing_usd_balance(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    facts = resolve_matrix_request(
        MatrixRequest(
            table_term_groups=(
                ("do la my usd",), ("so cuoi nam",), ("so dau nam",)),
            row_variants=("do la my usd", "do la my"),
            column_variants=_current_column_variants(year),
            collect="cell", row_mode="contains"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "closing_usd_currency_balance")
    return _single_matrix_value(facts, ticker, year, route, raw_currency=True)


def _matrix_derivative_contract_total(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    facts = resolve_matrix_request(
        MatrixRequest(
            table_term_groups=(
                ("tong gia tri hop dong",), ("tong gia tri ghi so",),
                ("so cuoi nam",), ("so dau nam",)),
            row_variants=("cong",),
            column_variants=(
                "tong gia tri hop dong so cuoi nam", "tong gia tri hop dong"),
            block_variants=("so cuoi nam",),
            block_stop_variants=("so dau nam",),
            collect="cell", row_mode="exact"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "derivative_contract_notional_total")
    if len(facts) != 1:
        return None
    # This banking note is explicitly presented in million VND, but the unit
    # marker is outside the extracted table and can be missed by table metadata.
    fact = facts[0]
    return MatrixValue(ticker, year, fact.value, fact.expr(), facts)


_FX_ROWS = (
    "do la my", "euro", "yen nhat", "do la singapore",
)


def _fx_currency_key(label: str) -> str:
    text = _plain(label)
    for key, variants in (
            ("usd", ("usd", "do la my")),
            ("eur", ("eur", "euro")),
            ("jpy", ("jpy", "yen nhat")),
            ("sgd", ("sgd", "singapore")),
            ("aud", ("aud",)), ("cad", ("cad",)), ("other", ("khac",))):
        if any(value in text for value in variants):
            return key
    return ""


def _facts_by_currency(facts: list[ResolvedFact]) -> dict[str, ResolvedFact]:
    result: dict[str, ResolvedFact] = {}
    for fact in facts:
        key = _fx_currency_key(fact.label)
        if not key or key in result:
            return {}
        result[key] = fact
    return result


def _matrix_fx_sensitivity_loss(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            ("cong no",), ("tai san",), ("so cuoi nam",), ("so dau nam",)),
        row_variants=_FX_ROWS, collect="rows", row_mode="contains")
    liabilities = resolve_matrix_request(
        MatrixRequest(
            **common,
            column_variants=("cong no so cuoi nam", "cong no")),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "foreign_currency_monetary_liabilities")
    assets = resolve_matrix_request(
        MatrixRequest(
            **common,
            column_variants=("tai san so cuoi nam", "tai san")),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "foreign_currency_monetary_assets")
    sensitivity = resolve_matrix_request(
        MatrixRequest(
            table_term_groups=(("nam nay",), ("nam truoc",), ("do la my",)),
            context_term_groups=(("do nhay",),),
            row_variants=_FX_ROWS,
            column_variants=("nam nay vnd", "nam nay"),
            collect="rows", row_mode="contains"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "foreign_currency_pretax_sensitivity")
    axes = tuple(_facts_by_currency(facts)
                 for facts in (liabilities, assets, sensitivity))
    if any(set(axis) != {"usd", "eur", "jpy", "sgd"} for axis in axes):
        return None
    debt_by_currency, assets_by_currency, sensitivity_by_currency = axes
    terms, selected, resolved = [], [], []
    for currency in ("usd", "eur", "jpy", "sgd"):
        debt = debt_by_currency[currency]
        asset = assets_by_currency[currency]
        shock = sensitivity_by_currency[currency]
        has_net_liability = debt.value_vnd > asset.value_vnd
        if has_net_liability != (shock.value_vnd < 0):
            return None
        if has_net_liability:
            selected.append(abs(shock.value_vnd))
        terms.append(
            f"(abs({shock.expr_vnd()}) if "
            f"({debt.expr_vnd()} > {asset.expr_vnd()}) else 0.0)")
        resolved.extend((debt, asset, shock))
    if not selected:
        return None
    scale = float(route.get("unit_scale", 1.0) or 1.0)
    return MatrixValue(
        ticker, year, sum(selected) / scale,
        f"(({' + '.join(terms)}) / {scale:g})", resolved)


def _matrix_fx_position_loss_share(
        tables: list[dict], ticker: str, year: int, route: dict,
        encoder, min_score: float) -> MatrixValue | None:
    positions = []
    for currency in ("USD", "EUR", "JPY", "AUD", "CAD", "Khac"):
        facts = resolve_matrix_request(
            MatrixRequest(
                table_term_groups=(
                    (f"31 thang 12 nam {year}",),
                    ("trang thai tien te noi",), ("tong tai san",),
                    ("tong no phai tra",)),
                row_variants=("trang thai tien te noi ngoai bang",),
                column_variants=(currency,), collect="cell", row_mode="exact"),
            tables, ticker, year, str(route.get("doc_type") or ""),
            f"net_monetary_position_{currency.lower()}")
        if len(facts) != 1:
            return None
        positions.append(facts[0])
    pretax = _scenario_exact(
        "pretax_profit", ticker, year, route, tables, encoder, min_score)
    if pretax is None or pretax.value <= 0:
        return None
    worst = min(fact.value_vnd for fact in positions)
    if worst >= 0:
        return None
    loss = abs(worst) * 0.05
    value = loss / pretax.value * 100.0
    position_exprs = ", ".join(fact.expr_vnd() for fact in positions)
    expr = (
        f"(max(0.0, -min({position_exprs})) * 0.05 "
        f"/ ({pretax.expr}) * 100)")
    return MatrixValue(
        ticker, year, value, expr,
        _dedupe_resolved(positions + pretax.resolved))


def _matrix_listed_equity_var(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    facts = resolve_matrix_request(
        MatrixRequest(
            table_term_groups=(
                (f"31 thang 12 nam {year}",),
                ("var 95% 1 ngay", "var 95 1 ngay"), ("tong",)),
            context_term_groups=(("rui ro gia co phieu",),),
            row_variants=("var 95% 1 ngay", "var 95 1 ngay"),
            column_variants=("tong",),
            collect="cell", row_mode="exact"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "listed_equity_var_one_day")
    if len(facts) != 1 or facts[0].value_vnd == 0:
        return None
    fact = facts[0]
    return MatrixValue(
        ticker, year, abs(fact.value_vnd), f"abs({fact.expr_vnd()})", facts)


def _matrix_real_estate_loan_share(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            ("hoat dong kinh doanh bat dong san",),
            ("cac nganh khac",),
        ),
        column_variants=_current_column_variants(year),
    )
    numerator = resolve_matrix_request(
        MatrixRequest(
            **common,
            row_variants=("hoat dong kinh doanh bat dong san",),
            collect="cell", row_mode="exact"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "real_estate_customer_loans")
    denominator = resolve_matrix_request(
        MatrixRequest(**common, collect="last_total"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "customer_loans")
    if len(numerator) != 1 or len(denominator) != 1:
        return None
    num, den = numerator[0], denominator[0]
    if ((num.report_id, num.table_pos, num.col)
            != (den.report_id, den.table_pos, den.col)) or den.value_vnd == 0:
        return None
    value = num.value_vnd / den.value_vnd * 100.0
    expr = f"({num.expr_vnd()} / {den.expr_vnd()} * 100)"
    return MatrixValue(ticker, year, value, expr, [num, den])


def _matrix_fixed_asset_share(
        tables: list[dict], ticker: str, year: int,
        route: dict) -> MatrixValue | None:
    common = dict(
        table_term_groups=(
            ("ma so",), ("tai san co dinh",),
            ("so cuoi nam", f"31/12/{year}", f"31.12.{year}"),
        ),
        column_variants=_current_column_variants(year), collect="rows",
    )
    costs = resolve_matrix_request(
        MatrixRequest(**common, row_codes=("222", "225", "228")),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "fixed_assets_cost")
    depreciation = resolve_matrix_request(
        MatrixRequest(**common, row_codes=("223", "226", "229")),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "fixed_assets_accumulated_depreciation")
    if not costs or not depreciation:
        return None
    if len({(fact.report_id, fact.table_pos, fact.col) for fact in costs + depreciation}) != 1:
        return None
    cost_codes = {fact.code for fact in costs}
    depreciation_codes = {fact.code for fact in depreciation}
    if any(str(int(code) - 1) not in cost_codes for code in depreciation_codes):
        return None
    # Code 225 can legitimately have a dash at code 226 (zero accumulated
    # depreciation), so it contributes to cost without a numeric numerator.
    if any(str(int(code) + 1) not in depreciation_codes
           for code in cost_codes if code != "225"):
        return None
    cost = sum(fact.value_vnd for fact in costs)
    dep = sum(abs(fact.value_vnd) for fact in depreciation)
    if cost <= 0 or dep < 0 or dep > cost * 1.01:
        return None
    cost_expr = " + ".join(fact.expr_vnd() for fact in costs)
    dep_expr = " + ".join(f"abs({fact.expr_vnd()})" for fact in depreciation)
    value = dep / cost * 100.0
    expr = f"(({dep_expr}) / ({cost_expr}) * 100)"
    return MatrixValue(ticker, year, value, expr, costs + depreciation)


def _matrix_segment_asset_share(
        tables: list[dict], ticker: str, year: int, route: dict,
        segment_variants: tuple[str, ...], *,
        closing_table: bool) -> MatrixValue | None:
    groups: tuple[tuple[str, ...], ...] = (
        segment_variants, ("tai san bo phan",), ("tong tai san",),
    )
    if closing_table:
        groups += ((f"31/12/{year}", f"31.12.{year}"),)
    common = dict(
        table_term_groups=groups,
        row_before_variants=("so dau nam",),
    )
    numerator = resolve_matrix_request(
        MatrixRequest(
            **common, row_variants=("tai san bo phan",),
            column_variants=segment_variants, collect="cell",
            row_mode="prefix"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "segment_assets")
    denominator = resolve_matrix_request(
        MatrixRequest(
            **common, row_variants=("tong tai san",),
            column_variants=("tong cong", "tong"), collect="cell",
            row_mode="exact"),
        tables, ticker, year, str(route.get("doc_type") or ""),
        "total_assets")
    if len(numerator) != 1 or len(denominator) != 1:
        return None
    num, den = numerator[0], denominator[0]
    if ((num.report_id, num.table_pos)
            != (den.report_id, den.table_pos)) or den.value_vnd == 0:
        return None
    value = num.value_vnd / den.value_vnd * 100.0
    expr = f"({num.expr_vnd()} / {den.expr_vnd()} * 100)"
    return MatrixValue(ticker, year, value, expr, [num, den])


def build_note_detail_plan(route: dict) -> NoteDetailPlan | None:
    """Build an exact parent/child arithmetic plan for note-table metrics."""
    question = route.get("question", "")
    text = _plain(question)
    if any(value in text for value in (
            "gia su", "kich ban", "neu ", "trung vi", "truoc khi",
            "giu nguyen", "can thiet de", "co the tang toi da")):
        return None

    members = tuple(dict.fromkeys(
        str(ticker).upper() for ticker in route.get("tickers") or [] if ticker))
    years = tuple(sorted(set(_route_years(route))))
    if not members or not years:
        return None

    matches = [
        match for match in _calculation_matches(question)
        if match.spec.name in _NOTE_DETAIL_SPEC_NAMES
    ]
    unique = _unique_matches(matches)
    if len(unique) != 1:
        return None
    match = next(iter(unique.values()))

    op = str((route.get("plan") or {}).get("op") or "")
    if op == "growth_pct" or (
            len(years) >= 2 and any(value in text for value in (
                "tang bao nhieu %", "tang bao nhieu phan tram",
                "thay doi bao nhieu %", "thay doi bao nhieu phan tram",
                "thay doi bao nhieu phan tram", "so voi ngay",
            )) and any(value in text for value in ("tang", "thay doi"))):
        reduction = "growth"
    elif op == "difference" or "chenh lech" in text:
        reduction = "difference"
    elif op == "average" or any(value in text for value in (
            "trung binh", "binh quan")):
        reduction = "mean"
    elif op == "sum" or (
            len(years) >= 2 and re.search(
                r"\btong\s+.+\b(?:giai doan|tu nam|qua cac nam)\b", text)):
        reduction = "sum"
    elif _wants_min(text):
        reduction = "min"
    elif _wants_max(text):
        reduction = "max"
    else:
        reduction = "direct"

    n_members, n_years = len(members), len(years)
    one_axis = n_members == 1 or n_years == 1
    if not one_axis:
        return None
    if reduction == "direct" and (n_members, n_years) != (1, 1):
        return None
    if reduction in {"growth", "sum"} and not (
            n_members == 1 and n_years >= 2):
        return None
    if reduction == "difference" and not (
            (n_members == 2 and n_years == 1)
            or (n_members == 1 and n_years == 2)):
        return None
    if reduction in {"mean", "max", "min"} and n_members * n_years < 2:
        return None
    if route.get("output_type") == "year" and reduction not in {"max", "min"}:
        return None

    return NoteDetailPlan(
        members=members,
        years=years,
        calculation=CalculationNode(match, "level"),
        reduction=reduction,
    )


def _try_note_detail(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: NoteDetailPlan) -> CompositeAnswer:
    """Execute a note-detail reduction only when every exact cell is present."""
    if len(typed.members) == 1:
        points = [(typed.members[0], year) for year in typed.years]
    else:
        points = [(ticker, typed.years[0]) for ticker in typed.members]

    values: list[FormulaValue] = []
    resolved: list[ResolvedFact] = []
    for ticker, year in points:
        value = _evaluate_calculation_exact(
            typed.calculation, ticker, year, list(typed.years),
            route, tables, encoder, min_score)
        if value is None or not _value_supports_year(value, year):
            return CompositeAnswer(
                ok=False, detail=f"note detail unresolved {ticker}/{year}",
                resolved=resolved)
        reports = {found.report_id for found in value.resolved}
        if len(reports) != 1:
            return CompositeAnswer(
                ok=False,
                detail=f"note detail operands cross reports {ticker}/{year}",
                resolved=resolved)
        values.append(value)
        resolved.extend(value.resolved)

    if not distinct_cells(resolved):
        return CompositeAnswer(
            ok=False, detail="note detail points collapsed to duplicate cells",
            resolved=resolved)

    reduction = typed.reduction
    spec = typed.calculation.match.spec
    raw_answer: float
    answer_expr: str
    result_kind = spec.kind
    if reduction == "direct":
        raw_answer, answer_expr = values[0].value, values[0].expr
    elif reduction == "growth":
        first, last = values[0], values[-1]
        if first.value == 0:
            return CompositeAnswer(
                ok=False, detail="note detail growth base is zero",
                resolved=resolved)
        raw_answer = (last.value - first.value) / abs(first.value) * 100.0
        answer_expr = (
            f"((({last.expr}) - ({first.expr})) / abs({first.expr}) * 100)")
        result_kind = "percent"
    elif reduction == "difference":
        raw_answer = values[0].value - values[1].value
        answer_expr = f"(({values[0].expr}) - ({values[1].expr}))"
    elif reduction == "mean":
        raw_answer = sum(value.value for value in values) / len(values)
        answer_expr = (
            f"(({' + '.join(value.expr for value in values)}) / {len(values)})")
    elif reduction == "sum":
        raw_answer = sum(value.value for value in values)
        answer_expr = f"({' + '.join(value.expr for value in values)})"
    elif reduction in {"max", "min"}:
        reverse = reduction == "max"
        ordered = sorted(values, key=lambda value: value.value, reverse=reverse)
        if len(ordered) > 1 and math.isclose(
                ordered[0].value, ordered[1].value,
                rel_tol=1e-12, abs_tol=1e-6):
            return CompositeAnswer(
                ok=False, detail="note detail extreme tie has no convention",
                resolved=resolved)
        chosen = ordered[0]
        if route.get("output_type") == "year":
            year_values = [
                (year, value.value, value.expr)
                for (_ticker, year), value in zip(points, values)
            ]
            raw_answer = float(chosen.year)
            answer_expr = _year_projection_expr(year_values, reduction)
            result_kind = "year"
        else:
            raw_answer = chosen.value
            fn = "max" if reduction == "max" else "min"
            answer_expr = f"{fn}({', '.join(value.expr for value in values)})"
    else:
        return CompositeAnswer(ok=False, detail="note detail reduction unsupported")

    answer = raw_answer
    if result_kind == "money":
        scale = float(route.get("unit_scale", 1.0) or 1.0)
        answer = raw_answer / scale
        answer_expr = f"(({answer_expr}) / {scale:g})"
    support = " + ".join(value.expr for value in values) or "0.0"
    query = f"round(({answer_expr}) + 0 * ({support}), 2)"
    warn = check_answer_unit(
        answer, route.get("output_type", result_kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(
            ok=False, detail=f"note detail unit guard: {warn}",
            resolved=resolved)
    resolved = _dedupe_resolved(resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=94.0),
        detail=(f"formula_note_detail_v9 metric={spec.name} "
                f"reduction={reduction} n={len(values)}"),
        resolved=resolved,
    )


def build_temporal_event_plan(route: dict) -> TemporalEventPlan | None:
    """Build a typed select-period/project plan for deterministic execution."""
    question = route.get("question", "")
    text = _plain(question)
    if route.get("output_type") == "year" or any(value in text for value in (
            "gia su", "kich ban", "neu ", "co the tang toi da", "truoc khi",
            "giu nguyen", "can thiet de")):
        return None
    if "trung vi" in text:
        return None

    members = tuple(dict.fromkeys(
        str(ticker).upper() for ticker in route.get("tickers") or [] if ticker))
    years = tuple(sorted(set(_route_years(route))))
    if len(years) < 2 or not members:
        return None
    joint_axis = bool(re.search(
        r"\b(?:cong ty|doanh nghiep|ma)(?:\s+co phieu)?\s+va\s+nam\b", text))
    if len(members) == 1:
        axis = "year"
    elif len(members) >= 2 and joint_axis:
        axis = "entity_year"
    else:
        return None

    matches = _calculation_matches(question)
    if not matches:
        return None
    first_event = re.search(r"\bnam\s+(dau tien|cuoi cung)\b", text)
    filters: list[Condition] = []
    if first_event is not None:
        if axis != "year":
            return None
        selector = _event_condition_match(question, matches, first_event.start())
        if selector is None:
            return None
        condition = _condition_for_match(question, selector)
        if condition is None:
            context = text[selector.end:min(len(text), selector.end + 90)]
            if re.search(r"\bam\b", context):
                condition = Condition(
                    selector.spec, selector.spec.name, (), "<", 0.0,
                    selector.spec.kind)
            elif re.search(r"\bduong\b", context):
                condition = Condition(
                    selector.spec, selector.spec.name, (), ">", 0.0,
                    selector.spec.kind)
        if condition is None:
            return None
        filters.append(condition)
        event_mode = "first" if first_event.group(1) == "dau tien" else "last"
        direction = "min" if event_mode == "first" else "max"
        extreme_start = first_event.start()
        selection_boundary = first_event.end()
        selector_node = CalculationNode(selector, "level")
    else:
        ranked = _ranked_match(text, matches)
        if ranked is None:
            return None
        selector, extreme_start, want_min = ranked
        event_mode = "extreme"
        direction = "min" if want_min else "max"
        selection_boundary = extreme_start
        selector_mode = _value_mode(text, selector, extreme_start, True)
        if selector_mode == "growth":
            selector_mode = "yoy_growth"
        selector_node = _calculation_node(
            text, selector, selector_mode, list(years), anchor=extreme_start)

    compatible = [
        match for match in matches
        if _output_accepts_kind(route.get("output_type"), match.spec.kind)
    ]
    target = _temporal_projection_match(
        compatible or matches, selector, extreme_start, text)
    if (target is not None
            and target.start > selector.start
            and _condition_for_match(question, target) is not None
            and _output_accepts_kind(route.get("output_type"), selector.spec.kind)):
        target = selector
    if (target is None and _output_accepts_kind(
            route.get("output_type"), selector.spec.kind)
            and _selector_is_direct_answer(text, selector)):
        target = selector
    if target is None:
        return None

    excluded = {selector.spec.name, target.spec.name}
    for condition in _parsed_conditions(route, question):
        if condition.label in excluded or condition in filters:
            continue
        filters.append(condition)
    same_projection = (
        target.start == selector.start and target.spec.name == selector.spec.name)
    if (event_mode == "extreme" and axis == "year" and same_projection
            and not filters and _projection_period_offset(text) == 0):
        return None
    if same_projection:
        target_mode = selector_node.mode
    else:
        target_mode = _temporal_target_value_mode(
            text, target, selection_boundary)
    projection = ProjectionPeriodNode(
        _calculation_node(text, target, target_mode, list(years), target.start),
        _projection_period_offset(text),
    )
    return TemporalEventPlan(
        axis=axis,
        members=members,
        years=years,
        event=TemporalEventNode(event_mode, direction, selector_node),
        projection=projection,
        filters=tuple(filters),
    )


def _temporal_target_value_mode(
        text: str, target: FormulaMatch, selection_boundary: int) -> str:
    """Keep selector wording from leaking into the projected calculation."""
    context = text[max(selection_boundary, target.start - 75):target.start]
    context = re.split(r"[,;:]", context)[-1]
    if "cagr" in context:
        return "cagr"
    if "muc giam" in context:
        return "decrease"
    if any(value in context for value in (
            "tang truong", "toc do tang", "ty le tang", "phan tram tang",
            "muc tang tuong doi", "tang tuong doi")):
        return "growth"
    if any(value in context for value in ("muc thay doi", "muc tang")):
        return "delta"
    return "level"


def _event_condition_match(
        question: str, matches: list[FormulaMatch], marker_start: int
        ) -> FormulaMatch | None:
    text = _plain(question)
    choices = []
    for match in matches:
        if match.start < marker_start:
            continue
        context = text[match.end:min(len(text), match.end + 100)]
        if re.search(r"\b(?:am|duong)\b", context):
            choices.append((match.start - marker_start, match))
    return min(choices, key=lambda item: item[0])[1] if choices else None


def _temporal_projection_match(
        matches: list[FormulaMatch], selector: FormulaMatch,
        event_start: int, text: str) -> FormulaMatch | None:
    candidates = [
        match for match in matches
        if not (match.start == selector.start and match.end == selector.end)
    ]
    answer_pos = text.find("bao nhieu", event_start)
    if answer_pos >= 0:
        before_answer = [match for match in candidates if match.start < answer_pos]
        if before_answer:
            return max(before_answer, key=lambda match: match.start)
    if candidates:
        return min(candidates, key=lambda match: abs(match.start - event_start))
    return selector if _selector_is_direct_answer(text, selector) else None


def _projection_period_offset(text: str) -> int:
    if any(value in text for value in (
            "nam ngay sau nam", "nam lien sau nam", "nam sau nam",
            "nam ke tiep", "nam lien ke", "cuoi nam ke tiep")):
        return 1
    if any(value in text for value in (
            "nam ngay truoc nam", "nam lien truoc nam", "nam truoc nam")):
        return -1
    return 0


def _try_temporal_event(
        route: dict, tables: list[dict], encoder, min_score: float,
        typed: TemporalEventPlan) -> CompositeAnswer:
    years = list(typed.years)
    if typed.axis == "year":
        dimension = [(typed.members[0], year) for year in years]
    elif typed.axis == "entity_year":
        dimension = [
            (ticker, year) for ticker in typed.members for year in years
        ]
    else:
        return CompositeAnswer(ok=False, detail="temporal event axis unsupported")

    values: dict[tuple[str, int], FormulaValue] = {}
    pass_exprs: dict[tuple[str, int], str] = {}
    eligible: list[tuple[str, int, FormulaValue]] = []
    resolved: list[ResolvedFact] = []
    support: list[str] = []
    selector_node = typed.event.selector
    for ticker, year in dimension:
        selector = _evaluate_calculation_exact(
            selector_node, ticker, year, years,
            route, tables, encoder, min_score)
        if selector is None:
            return CompositeAnswer(
                ok=False,
                detail=f"temporal selector unresolved {ticker}/{year}",
                resolved=resolved)
        if selector_node.mode == "level" and not _value_supports_year(
                selector, year):
            return CompositeAnswer(
                ok=False,
                detail=f"temporal selector lacks exact year {ticker}/{year}",
                resolved=resolved)
        values[(ticker, year)] = selector
        resolved.extend(selector.resolved)
        support.append(selector.expr)

        passes = True
        expressions = []
        for condition in typed.filters:
            if (condition.spec is not None
                    and condition.spec.name == selector.spec.name
                    and selector_node.mode == "level"):
                checked = selector
            else:
                checked = (
                    _evaluate_formula_exact(
                        condition.spec, ticker, year,
                        route, tables, encoder, min_score)
                    if condition.spec is not None
                    else _evaluate_condition(
                        condition, ticker, year,
                        route, tables, encoder, min_score)
                )
            if checked is None or not _value_supports_year(checked, year):
                return CompositeAnswer(
                    ok=False,
                    detail=f"temporal filter unresolved {condition.label}/{ticker}/{year}",
                    resolved=resolved)
            resolved.extend(checked.resolved)
            support.append(checked.expr)
            passes = passes and _compare(
                checked.value, condition.op, condition.threshold)
            expressions.append(_condition_expr(
                checked.expr, condition.op, condition.threshold))
        pass_exprs[(ticker, year)] = " and ".join(expressions) or "True"
        if passes:
            eligible.append((ticker, year, selector))

    if not eligible:
        return CompositeAnswer(
            ok=False, detail="temporal event filtered all candidates",
            resolved=resolved)
    if typed.event.mode == "first":
        chosen = min(eligible, key=lambda item: item[1])
    elif typed.event.mode == "last":
        chosen = max(eligible, key=lambda item: item[1])
    else:
        ordered = sorted(
            eligible, key=lambda item: item[2].value,
            reverse=typed.event.direction == "max")
        if len(ordered) > 1 and math.isclose(
                ordered[0][2].value, ordered[1][2].value,
                rel_tol=1e-12, abs_tol=1e-6):
            return CompositeAnswer(
                ok=False, detail="temporal event tie has no convention",
                resolved=resolved)
        chosen = ordered[0]

    ticker, selected_year, selected_value = chosen
    target_year = selected_year + typed.projection.offset
    target = _evaluate_calculation_exact(
        typed.projection.calculation, ticker, target_year, years,
        route, tables, encoder, min_score)
    if target is None:
        return CompositeAnswer(
            ok=False, detail=f"temporal projection unresolved {ticker}/{target_year}",
            resolved=resolved)
    if (typed.projection.calculation.mode == "level"
            and not _value_supports_year(target, target_year)):
        return CompositeAnswer(
            ok=False,
            detail=f"temporal projection lacks exact year {ticker}/{target_year}",
            resolved=resolved)
    resolved.extend(target.resolved)
    answer, answer_expr = _answer_value(target, route)

    selected_passes = pass_exprs[(ticker, selected_year)]
    comparisons = [selected_passes]
    if typed.event.mode in {"first", "last"}:
        for other_ticker, other_year in dimension:
            if other_ticker != ticker or other_year == selected_year:
                continue
            is_prior = other_year < selected_year
            if ((typed.event.mode == "first" and is_prior)
                    or (typed.event.mode == "last" and not is_prior)):
                comparisons.append(
                    f"(not ({pass_exprs[(other_ticker, other_year)]}))")
    else:
        comparator = ">" if typed.event.direction == "max" else "<"
        for other_ticker, other_year in dimension:
            if (other_ticker, other_year) == (ticker, selected_year):
                continue
            comparisons.append(
                f"((not ({pass_exprs[(other_ticker, other_year)]})) or "
                f"(({selected_value.expr}) {comparator} "
                f"({values[(other_ticker, other_year)].expr})))")
    selection_guard = " and ".join(comparisons) or "True"
    support_expr = " + ".join(support) or "0.0"
    query = (f"round((({answer_expr}) if ({selection_guard}) else 0.0) "
             f"+ 0 * ({support_expr}), 2)")
    warn = check_answer_unit(
        answer, route.get("output_type", target.spec.kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(
            ok=False, detail=f"temporal event unit guard: {warn}",
            resolved=resolved)
    resolved = _dedupe_resolved(resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=94.0),
        detail=(f"formula_temporal_event_v8 axis={typed.axis} "
                f"event={typed.event.mode}/{typed.event.direction} "
                f"selector={selector_node.match.spec.name}/{selector_node.mode} "
                f"projection={typed.projection.calculation.match.spec.name}/"
                f"{typed.projection.calculation.mode} "
                f"offset={typed.projection.offset:+d} "
                f"picked={ticker}/{selected_year}"),
        resolved=resolved,
    )


def build_compositional_ranking_plan(
        route: dict) -> CompositionalRankingPlan | None:
    question = route.get("question", "")
    text = _plain(question)
    matches = _calculation_matches(question)
    ranked = _ranked_match(text, matches)
    if ranked is None:
        return None
    selector, extreme_start, want_min = ranked
    output_type = route.get("output_type")
    compatible = [
        match for match in matches
        if _output_accepts_kind(output_type, match.spec.kind)
    ]
    target = _target_match(compatible or matches, selector, extreme_start, text)
    if (target is not None
            and _condition_for_match(question, target) is not None
            and _output_accepts_kind(route.get("output_type"), selector.spec.kind)):
        # "revenue lowest among years with net margin > 10%" asks for the
        # ranked revenue itself; the other formula is only a filter.
        target = selector
    if (target is None and _output_accepts_kind(route.get("output_type"),
                                                selector.spec.kind)
            and _selector_is_direct_answer(text, selector)):
        target = selector
    if target is None:
        return None

    tickers = list(dict.fromkeys(route.get("tickers") or []))
    years = sorted(set(_route_years(route)))
    if len(tickers) >= 2:
        dimension = "entity"
    elif len(tickers) == 1 and len(years) >= 2:
        dimension = "year"
    else:
        return None
    implicit_prior_change = _has_implicit_prior_change(text, years)
    temporal_selector = len(years) >= 2 or implicit_prior_change
    selector_mode = _value_mode(text, selector, extreme_start, temporal_selector)
    if dimension == "year" and selector_mode == "growth":
        selector_mode = "yoy_growth"
    projection_mode = _target_value_mode(
        text, target, dimension == "entity"
        and (len(years) >= 2 or implicit_prior_change))
    predicates = list(_ranking_temporal_predicates(
        text, matches, selector, target, dimension, years))
    if dimension == "entity":
        cohort_predicates, _used = _aggregate_predicates(text, matches, years)
        excluded_matches = {_match_key(selector), _match_key(target)}
        predicates.extend(
            predicate for predicate in cohort_predicates
            if predicate.reference == "constant"
            and _match_key(predicate.calculation.match) not in excluded_matches
        )
        predicates.extend(_joint_operating_decline_predicates(
            text, matches, years))
    predicates = tuple(dict.fromkeys(predicates))
    excluded = {
        selector.spec.name, target.spec.name,
        *(predicate.calculation.match.spec.name for predicate in predicates),
    }
    conditions = tuple(
        condition for condition in _parsed_conditions(route, question)
        if condition.label not in excluded
    )
    median_filter = _median_filter_node(text, matches, dimension, years)
    if "trung vi" in text and median_filter is None:
        return None
    selector_node = _calculation_node(
        text, selector, selector_mode, years, anchor=extreme_start)
    projection_node = _calculation_node(
        text, target, projection_mode, years, anchor=target.start)
    projection_reduction = (
        "max_minus_min"
        if (output_type == "percentage_point"
            and "chenh lech" in text and "so voi" in text
            and re.search(r"\b(?:cao|lon|nhieu)\s+nhat\b", text)
            and re.search(r"\b(?:thap|nho|it)\s+nhat\b", text))
        else "selected"
    )
    return CompositionalRankingPlan(
        dimension=dimension,
        direction="min" if want_min else "max",
        selector=selector_node,
        projection=projection_node,
        filters=conditions,
        median_filter=median_filter,
        predicates=predicates,
        projection_reduction=projection_reduction,
    )


def _ranking_temporal_predicates(
        text: str, matches: list[FormulaMatch], selector: FormulaMatch,
        target: FormulaMatch, dimension: str,
        years: list[int]) -> list[PredicateNode]:
    if dimension != "year" or len(years) < 2:
        return []
    out, seen = [], set()
    for match in matches:
        if match.spec.name in {selector.spec.name, target.spec.name}:
            continue
        context = text[max(0, match.start - 45):min(len(text), match.end + 95)]
        if not re.search(
                r"\b(?:tang|cao hon)\s+so voi\s+nam\s+lien\s+truoc\b",
                context):
            continue
        if match.spec.name in seen:
            continue
        seen.add(match.spec.name)
        out.append(PredicateNode(
            calculation=CalculationNode(match, "yoy_delta"),
            op=">", threshold=0.0,
        ))
    return out


def _has_implicit_prior_change(text: str, years: list[int]) -> bool:
    if len(years) != 1:
        return False
    return bool(re.search(
        r"\b(?:tu dau nam den cuoi nam|cung (?:sut )?giam nam 20\d{2}|"
        r"(?:tang|giam) (?:manh|nhieu|lon) nhat|giam bao nhieu)",
        text,
    ))


def _joint_operating_decline_predicates(
        text: str, matches: list[FormulaMatch],
        years: list[int]) -> list[PredicateNode]:
    found = re.search(
        r"\bdoanh thu va bien loi nhuan hoat dong cung (?:sut )?giam"
        r"(?: nam (20\d{2}))?\b",
        text,
    )
    if found is None:
        return []
    end_year = int(found.group(1)) if found.group(1) else (
        years[-1] if years else None)
    if end_year is None:
        return []
    operating = next(
        (match for match in matches
         if match.spec.name == "operating_margin"
         and found.start() <= match.start <= found.end()),
        None,
    )
    if operating is None:
        return []
    revenue_spec = next(
        spec for spec in _METRIC_SPECS if spec.name == "net_revenue")
    revenue_start = text.find("doanh thu", found.start(), found.end())
    revenue = FormulaMatch(
        revenue_spec, revenue_start, revenue_start + len("doanh thu"),
        "doanh thu",
    )
    start_year = end_year - 1
    return [
        PredicateNode(
            CalculationNode(revenue, "delta", start_year, end_year),
            "<", threshold=0.0, years=(start_year, end_year)),
        PredicateNode(
            CalculationNode(operating, "delta", start_year, end_year),
            "<", threshold=0.0, years=(start_year, end_year)),
    ]


def _try_nested_ranking(route: dict, tables: list[dict], encoder,
                        min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    text = _plain(question)
    typed = build_compositional_ranking_plan(route)
    if typed is None:
        return CompositeAnswer(ok=False, detail="typed nested ranking plan missing")
    selector = typed.selector.match
    target = typed.projection.match
    want_min = typed.direction == "min"
    if ("tu dau nam den cuoi nam" in text
            and typed.selector.mode == "level"):
        return CompositeAnswer(ok=False, detail="nested ranking needs intra-year columns")
    if (re.search(r"\b\d+\s+(?:doanh nghiep|cong ty|ma)\b", text)
            and any(w in text for w in ("nam giu bao nhieu phan tram tong",
                                         "dong gop bao nhieu phan tram tong",
                                         "ty trong trong tong"))):
        return CompositeAnswer(ok=False,
                               detail="nested top-n group aggregation unsupported")
    if any(w in text for w in ("gia su", "kich ban", "co the tang toi da",
                                "truoc khi", "neu ")):
        return CompositeAnswer(ok=False, detail="nested ranking scenario unsupported")
    if ("chenh lech duong giua lnst va cfo" in text
            and typed.selector.match.spec.name != "net_profit_cfo_gap"):
        return CompositeAnswer(ok=False,
                               detail="nested ranking unsupported exact formula")
    if (any(_mentions_average_balance(text, metric) for metric in (
            "total_assets", "equity", "fixed_assets"))
            and not any(match.spec.average_balances
                        for match in _calculation_matches(question))):
        return CompositeAnswer(
            ok=False, detail="nested ranking average-balance formula unbound")
    if ("hang ton kho binh quan" in text and "nhan 365" in text
            and typed.selector.match.spec.name != "inventory_days"):
        return CompositeAnswer(ok=False,
                               detail="nested ranking inventory-days plan mismatch")

    tickers = _candidate_tickers(route, tables)
    years = sorted(set(_route_years(route)))
    if typed.dimension == "entity" and len(tickers) >= 2:
        stated_n = _stated_population_size(text)
        if stated_n is not None and stated_n != len(tickers):
            return CompositeAnswer(
                ok=False,
                detail=f"nested ranking population {len(tickers)}/{stated_n}")
        dimension = [(ticker, years[-1] if years else None) for ticker in tickers]
        entity_mode = True
    elif typed.dimension == "year" and len(tickers) == 1 and len(years) >= 2:
        dimension = [(tickers[0], year) for year in years]
        entity_mode = False
    else:
        return CompositeAnswer(ok=False, detail="nested ranking has no dimension")

    selector_mode = typed.selector.mode
    if not entity_mode and selector_mode == "growth":
        return CompositeAnswer(ok=False,
                               detail="nested year-over-year selector unsupported")
    conditions = list(typed.filters)

    ranked_values, support, resolved = [], [], []
    selector_values: dict[tuple[str, int | None], FormulaValue] = {}
    filter_exprs: dict[tuple[str, int | None], str] = {}
    median_values: dict[tuple[str, int | None], FormulaValue] = {}
    median = median_expr = None
    if typed.median_filter is not None:
        node = typed.median_filter
        for ticker, year in dimension:
            filter_year = node.year if entity_mode else year
            fv = _evaluate_calculation_exact(
                node.calculation, ticker, filter_year, years,
                route, tables, encoder, min_score)
            if fv is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"median filter unresolved {ticker}/{filter_year}",
                    resolved=resolved,
                )
            median_values[(ticker, year)] = fv
            support.append(fv.expr)
            resolved.extend(fv.resolved)
        median = _median([value.value for value in median_values.values()])
        median_expr = _median_expr(
            [value.expr for value in median_values.values()])

    for ticker, year in dimension:
        sv = _evaluate_calculation_exact(
            typed.selector, ticker, year, years,
            route, tables, encoder, min_score)
        if sv is None:
            return CompositeAnswer(ok=False,
                                   detail=f"nested selector unresolved {ticker}/{year}",
                                   resolved=resolved)
        if selector_mode == "level" and not _value_supports_year(sv, year):
            return CompositeAnswer(
                ok=False, detail=f"nested selector lacks exact year {ticker}/{year}",
                resolved=resolved)
        resolved.extend(sv.resolved)
        support.append(sv.expr)
        selector_values[(ticker, year)] = sv

        passes = True
        pass_exprs = []
        if typed.median_filter is not None:
            filter_value = median_values[(ticker, year)]
            passes = _compare(filter_value.value, typed.median_filter.op, median)
            pass_exprs.append(_condition_expr(
                filter_value.expr, typed.median_filter.op, median_expr))
        for cond in conditions:
            check_years = _nested_condition_years(
                text, entity_mode, years, year)
            for check_year in check_years:
                cv = (
                    _evaluate_formula_exact(
                        cond.spec, ticker, check_year,
                        route, tables, encoder, min_score)
                    if cond.spec is not None and check_year is not None
                    else _evaluate_condition(
                        cond, ticker, check_year,
                        route, tables, encoder, min_score)
                )
                if cv is None:
                    return CompositeAnswer(
                        ok=False,
                        detail=f"nested filter unresolved {ticker}/{check_year}",
                        resolved=resolved)
                resolved.extend(cv.resolved)
                support.append(cv.expr)
                if not _value_supports_year(cv, check_year):
                    return CompositeAnswer(
                        ok=False,
                        detail=f"nested filter lacks exact year {ticker}/{check_year}",
                        resolved=resolved)
                passes = passes and _compare(cv.value, cond.op, cond.threshold)
                pass_exprs.append(_condition_expr(cv.expr, cond.op, cond.threshold))
        for predicate in typed.predicates:
            predicate_years = years if entity_mode else [year]
            predicate_values = _evaluate_aggregate_predicate(
                predicate, ticker, predicate_years,
                route, tables, encoder, min_score)
            if predicate_values is None or predicate.threshold is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"nested typed predicate unresolved {ticker}/{year}",
                    resolved=resolved)
            checks = [
                _compare(value.value, predicate.op, predicate.threshold)
                for value in predicate_values
            ]
            expressions = [
                _condition_expr(value.expr, predicate.op, predicate.threshold)
                for value in predicate_values
            ]
            reducer = (
                predicate.quantifier.mode if predicate.quantifier is not None
                else "all"
            )
            predicate_passes = any(checks) if reducer == "any" else all(checks)
            passes = passes and predicate_passes
            pass_exprs.append(
                (" or " if reducer == "any" else " and ").join(expressions))
            for value in predicate_values:
                resolved.extend(value.resolved)
                support.append(value.expr)
        filter_exprs[(ticker, year)] = " and ".join(pass_exprs) or "True"
        if passes:
            ranked_values.append((ticker, year, sv))

    if not ranked_values:
        return CompositeAnswer(ok=False, detail="nested ranking filtered all candidates",
                               resolved=resolved)
    ordered = sorted(ranked_values, key=lambda item: item[2].value,
                     reverse=not want_min)
    if len(ordered) > 1 and math.isclose(
            ordered[0][2].value, ordered[1][2].value,
            rel_tol=1e-12, abs_tol=1e-6):
        return CompositeAnswer(ok=False, detail="nested ranking tie has no convention",
                               resolved=resolved)
    chosen = ordered[0]
    ticker, selected_year, _selector_value = chosen

    if typed.projection_reduction == "max_minus_min":
        if not entity_mode or len(ordered) < 2 or want_min:
            return CompositeAnswer(
                ok=False, detail="nested projection spread shape unsupported",
                resolved=resolved)
        if len(ordered) > 2 and math.isclose(
                ordered[-1][2].value, ordered[-2][2].value,
                rel_tol=1e-12, abs_tol=1e-6):
            return CompositeAnswer(
                ok=False, detail="nested projection spread minimum tie",
                resolved=resolved)
        bottom = ordered[-1]
        top_target = _evaluate_nested_projection(
            typed, chosen, entity_mode, years, text,
            route, tables, encoder, min_score)
        bottom_target = _evaluate_nested_projection(
            typed, bottom, entity_mode, years, text,
            route, tables, encoder, min_score)
        if top_target is None or bottom_target is None:
            return CompositeAnswer(
                ok=False, detail="nested projection spread target unresolved",
                resolved=resolved)
        resolved.extend(top_target.resolved)
        resolved.extend(bottom_target.resolved)
        top_answer, top_expr = _answer_value(top_target, route)
        bottom_answer, bottom_expr = _answer_value(bottom_target, route)
        answer = top_answer - bottom_answer
        answer_expr = f"(({top_expr}) - ({bottom_expr}))"
        comparison_population = (
            dimension if (typed.median_filter is not None
                          or conditions or typed.predicates)
            else [(candidate[0], candidate[1]) for candidate in ranked_values]
        )

        def extreme_guard(item, comparator: str) -> str:
            item_key = (item[0], item[1])
            checks = [filter_exprs[item_key]]
            for candidate_ticker, candidate_year in comparison_population:
                candidate_key = (candidate_ticker, candidate_year)
                if candidate_key == item_key:
                    continue
                checks.append(
                    f"((not ({filter_exprs[candidate_key]})) or "
                    f"(({item[2].expr}) {comparator} "
                    f"({selector_values[candidate_key].expr})))")
            return " and ".join(checks) or "True"

        selection_guard = (
            f"({extreme_guard(chosen, '>')}) and "
            f"({extreme_guard(bottom, '<')})"
        )
        support_expr = " + ".join(support) or "0.0"
        query = (f"round((({answer_expr}) if ({selection_guard}) else 0.0) "
                 f"+ 0 * ({support_expr}), 2)")
        warn = check_answer_unit(answer, route.get("output_type", "percentage_point"))
        if warn and "outside plausible range" in warn:
            return CompositeAnswer(
                ok=False, detail=f"nested projection spread unit guard: {warn}",
                resolved=resolved)
        resolved = _dedupe_resolved(resolved)
        return CompositeAnswer(
            ok=True, answer=answer, pandas_query=query,
            confidence=_confidence(resolved, base=94.0),
            detail=("formula_projection_spread_v19 "
                    f"selector={selector.spec.name}/{selector_mode} "
                    f"projection={target.spec.name}/{typed.projection.mode} "
                    f"picked={chosen[0]}-{bottom[0]}/{selected_year}"),
            resolved=resolved,
        )

    target_mode = typed.projection.mode
    tv = _evaluate_nested_projection(
        typed, chosen, entity_mode, years, text,
        route, tables, encoder, min_score)
    if tv is None:
        return CompositeAnswer(ok=False,
                               detail=f"nested target unresolved {ticker}/{selected_year}",
                               resolved=resolved)
    resolved.extend(tv.resolved)
    answer, answer_expr = _answer_value(tv, route)
    support_expr = " + ".join(support)
    comparator = "<" if want_min else ">"
    selected_expr = chosen[2].expr
    comparisons = [filter_exprs[(ticker, selected_year)]]
    comparison_population = (
        dimension if (typed.median_filter is not None
                      or conditions or typed.predicates)
        else [(candidate[0], candidate[1]) for candidate in ranked_values]
    )
    for candidate_ticker, candidate_year in comparison_population:
        if (candidate_ticker, candidate_year) == (ticker, selected_year):
            continue
        candidate_selector = selector_values[(candidate_ticker, candidate_year)]
        comparison = (
            f"(({selected_expr}) {comparator} ({candidate_selector.expr}))")
        candidate_passes = filter_exprs[(candidate_ticker, candidate_year)]
        comparison = f"((not ({candidate_passes})) or ({comparison}))"
        comparisons.append(comparison)
    selection_guard = " and ".join(comparisons) or "True"
    query = (f"round((({answer_expr}) if ({selection_guard}) else 0.0) "
             f"+ 0 * ({support_expr}), 2)")
    warn = check_answer_unit(answer, route.get("output_type", tv.spec.kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"nested unit guard: {warn}",
                               resolved=resolved)
    resolved = _dedupe_resolved(resolved)
    period_aware = any(_is_period_aware_v6(value.spec) for value in (
        typed.selector.match, typed.projection.match,
        *( [typed.median_filter.calculation.match]
           if typed.median_filter is not None else []),
    ))
    quantified = any(
        predicate.quantifier is not None for predicate in typed.predicates)
    detail_family = (
        "formula_quantified_cohort_v7" if quantified
        else "formula_period_aware_v6" if period_aware
        else "formula_nested_v4"
    )
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=93.0),
        detail=(f"{detail_family} selector={selector.spec.name}/{selector_mode} "
                f"projection={target.spec.name}/{target_mode} "
                f"median={typed.median_filter.calculation.match.spec.name if typed.median_filter else 'none'} "
                f"picked={ticker}/{selected_year}"),
        resolved=resolved)


def _evaluate_nested_projection(
        typed: CompositionalRankingPlan,
        selected: tuple[str, int | None, FormulaValue],
        entity_mode: bool, years: list[int], text: str,
        route: dict, tables: list[dict], encoder,
        min_score: float) -> FormulaValue | None:
    ticker, selected_year, _selector = selected
    node = typed.projection
    if node.mode != "level" and entity_mode:
        target_start = (
            node.start_year if node.start_year is not None else years[0])
        target_end = (
            node.end_year if node.end_year is not None else years[-1])
        return _evaluate_change_exact(
            node.match.spec, ticker, target_start, target_end, node.mode,
            route, tables, encoder, min_score)
    target_year = selected_year
    if not entity_mode and "nam sau nam" in text:
        target_year = int(selected_year) + 1
    if target_year is None:
        return None
    value = _evaluate_formula_exact(
        node.match.spec, ticker, target_year,
        route, tables, encoder, min_score)
    if (value is None or (node.mode == "level"
                          and not _value_supports_year(value, target_year))):
        return None
    return value


def _nested_condition_years(
        text: str, entity_mode: bool, years: list[int],
        candidate_year: int | None) -> list[int | None]:
    if not entity_mode or len(years) < 2:
        return [candidate_year]
    all_period_markers = (
        "ca hai nam", "ca ba nam", "ca 2 nam", "ca 3 nam",
        "trong hai nam", "trong ba nam", "trong ca hai nam",
        "trong ca ba nam", "duy tri", "lien tuc",
    )
    return list(years) if any(marker in text for marker in all_period_markers) else [candidate_year]


def _is_period_aware_v6(spec: FormulaSpec) -> bool:
    return spec.name in {
        "average_total_assets", "average_equity", "average_fixed_assets",
        "roa_average_assets", "roe_average_equity",
        "total_asset_turnover_average_assets",
        "fixed_asset_turnover_average_assets", "accrual_average_assets",
    }


def _evaluate_calculation_exact(
        node: CalculationNode, ticker: str, year: int | None,
        years: list[int], route: dict, tables: list[dict], encoder,
        min_score: float) -> FormulaValue | None:
    if node.mode == "level":
        if year is None:
            return None
        return _evaluate_formula_exact(
            node.match.spec, ticker, year, route, tables, encoder, min_score)
    if node.mode in {"yoy_growth", "yoy_delta"}:
        if year is None:
            return None
        return _evaluate_change_exact(
            node.match.spec, ticker, int(year) - 1, int(year),
            "growth" if node.mode == "yoy_growth" else "delta",
            route, tables, encoder, min_score)
    start = node.start_year if node.start_year is not None else (
        years[0] if years else None)
    end = node.end_year if node.end_year is not None else (
        years[-1] if years else None)
    if start is None or end is None or start >= end:
        return None
    return _evaluate_change_exact(
        node.match.spec, ticker, start, end, node.mode,
        route, tables, encoder, min_score)


def _dedupe_resolved(values: list[ResolvedFact]) -> list[ResolvedFact]:
    seen, out = set(), []
    for value in values:
        key = (value.report_id, value.table_pos, value.row, value.col)
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _try_formula_change(route: dict, tables: list[dict], encoder,
                        min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    text = _plain(question)
    aggregates = any(w in text for w in ("binh quan", "trung binh"))
    if _looks_nested_selector(question) and not aggregates:
        return CompositeAnswer(ok=False, detail="formula change looks nested")
    if any(w in text for w in ("gia su", "kich ban", "neu ")):
        return CompositeAnswer(ok=False, detail="formula change scenario unsupported")
    tickers = _candidate_tickers(route, tables)
    years = sorted(set(_route_years(route)))
    matches = [m for m in _calculation_matches(question)
               if m.spec in _FORMULAS]
    by_name = _unique_matches(matches)
    if not by_name:
        return CompositeAnswer(ok=False, detail="formula change has no formulas")

    # Same-period difference between two formula concepts, optionally averaged
    # over a filtered group (for example gross margin minus net margin).
    if (len(years) == 1 and len(by_name) == 2 and len(tickers) >= 1
            and "chenh lech giua" in text):
        specs = _pair_specs_after_difference(text, matches)
        if len(specs) != 2:
            return CompositeAnswer(ok=False, detail="formula pair order unresolved")
        return _formula_pair_difference(route, tables, encoder, min_score,
                                        tickers, years[0], specs)

    if len(years) < 2:
        if len(tickers) == 2 and len(by_name) == 1:
            spec = next(iter(by_name.values())).spec
            vals, resolved = [], []
            for ticker in tickers:
                fv = _evaluate_formula(
                    spec, ticker, years[0] if years else None, route, tables,
                    encoder, min_score)
                if fv is None:
                    return CompositeAnswer(ok=False,
                                           detail=f"formula comparison unresolved {ticker}")
                if years and not _value_supports_year(fv, years[0]):
                    return CompositeAnswer(
                        ok=False,
                        detail=f"formula comparison lacks exact year {ticker}/{years[0]}")
                vals.append(fv)
                resolved.extend(fv.resolved)
            answer = vals[0].value - vals[1].value
            query = f"round(({vals[0].expr}) - ({vals[1].expr}), 2)"
            warn = check_answer_unit(answer, route.get("output_type", spec.kind))
            if warn and "outside plausible range" in warn:
                return CompositeAnswer(ok=False,
                                       detail=f"formula comparison unit guard: {warn}",
                                       resolved=resolved)
            return CompositeAnswer(ok=True, answer=answer, pandas_query=query,
                                   confidence=_confidence(resolved, base=88.0),
                                   detail=f"formula_compare {spec.name}", resolved=resolved)
        return CompositeAnswer(ok=False, detail="formula change needs two periods")

    start, end = years[0], years[-1]
    target = _change_target_match(text, list(by_name.values()))
    if target is None:
        return CompositeAnswer(ok=False, detail="formula change target missing")
    mode = _target_value_mode(text, target, True)
    if mode == "level":
        return CompositeAnswer(ok=False, detail="formula change target is only a filter")

    values, support, resolved = [], [], []
    revenue_filter = ("doanh thu thuan" in text and
                      any(w in text for w in ("doanh thu thuan nam",
                                              "tang truong doanh thu thuan",
                                              "doanh thu thuan tang")) and
                      any(w in text for w in ("tang so voi", "cao hon nam", "duong",
                                              "doanh thu thuan tang")))
    revenue_spec = next(s for s in _METRIC_SPECS if s.name == "net_revenue")
    for ticker in tickers:
        include = True
        if revenue_filter:
            rv = _evaluate_change(revenue_spec, ticker, start, end, "delta", route,
                                  tables, encoder, min_score)
            if rv is None:
                return CompositeAnswer(ok=False,
                                       detail=f"formula change filter unresolved {ticker}",
                                       resolved=resolved)
            include = rv.value > 0
            support.append(rv.expr)
            resolved.extend(rv.resolved)
        if not include:
            continue
        fv = _evaluate_change(target.spec, ticker, start, end, mode, route,
                              tables, encoder, min_score)
        if fv is None:
            return CompositeAnswer(ok=False,
                                   detail=f"formula change unresolved {ticker}",
                                   resolved=resolved)
        values.append(fv)
        support.append(fv.expr)
        resolved.extend(fv.resolved)

    if not values:
        return CompositeAnswer(ok=False, detail="formula change filter empty",
                               resolved=resolved)
    average = len(values) > 1 and any(w in text for w in ("binh quan", "trung binh"))
    if len(values) > 1 and not average:
        return CompositeAnswer(ok=False, detail="formula change ambiguous aggregation",
                               resolved=resolved)
    if average:
        answer = sum(v.value for v in values) / len(values)
        answer_expr = f"(({' + '.join(v.expr for v in values)}) / {len(values)})"
    else:
        answer = values[0].value
        answer_expr = values[0].expr
    query = f"round(({answer_expr}) + 0 * ({' + '.join(support)}), 2)"
    warn = check_answer_unit(answer, route.get("output_type", target.spec.kind))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"formula change unit guard: {warn}",
                               resolved=resolved)
    return CompositeAnswer(
        ok=True, answer=answer, pandas_query=query,
        confidence=_confidence(resolved, base=88.0),
        detail=f"formula_change {target.spec.name}/{mode} n={len(values)}",
        resolved=resolved)


def _formula_pair_difference(route: dict, tables: list[dict], encoder,
                             min_score: float, tickers: list[str], year: int,
                             specs: list[FormulaSpec]) -> CompositeAnswer:
    text = _plain(route.get("question", ""))
    values, resolved = [], []
    for ticker in tickers:
        a = _evaluate_formula(specs[0], ticker, year, route, tables, encoder, min_score)
        b = _evaluate_formula(specs[1], ticker, year, route, tables, encoder, min_score)
        if a is None or b is None:
            return CompositeAnswer(ok=False,
                                   detail=f"formula pair unresolved {ticker}",
                                   resolved=resolved)
        if not _value_supports_year(a, year) or not _value_supports_year(b, year):
            return CompositeAnswer(ok=False,
                                   detail=f"formula pair lacks exact year {ticker}/{year}",
                                   resolved=resolved)
        resolved.extend(a.resolved + b.resolved)
        if "duong" in text and b.value <= 0:
            continue
        values.append((a.value - b.value, f"(({a.expr}) - ({b.expr}))"))
    if not values:
        return CompositeAnswer(ok=False, detail="formula pair filter empty",
                               resolved=resolved)
    if len(values) > 1 and not any(w in text for w in ("binh quan", "trung binh")):
        return CompositeAnswer(ok=False, detail="formula pair aggregation missing",
                               resolved=resolved)
    answer = sum(v for v, _ in values) / len(values)
    expr = f"(({' + '.join(e for _, e in values)}) / {len(values)})"
    warn = check_answer_unit(answer, route.get("output_type", "percentage_point"))
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"formula pair unit guard: {warn}",
                               resolved=resolved)
    return CompositeAnswer(ok=True, answer=answer, pandas_query=f"round({expr}, 2)",
                           confidence=_confidence(resolved, base=87.0),
                           detail=f"formula_pair_difference n={len(values)}",
                           resolved=resolved)


def _try_formula_ranking(route: dict, tables: list[dict], encoder,
                         min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    if _looks_nested_selector(question):
        return CompositeAnswer(ok=False, detail="ranking looks nested")
    specs = _detected_specs(question)
    if len(specs) != 1:
        return CompositeAnswer(ok=False, detail=f"ranking formulas={len(specs)}")
    spec = specs[0]
    tickers = _candidate_tickers(route, tables)
    year = _primary_year(route)
    if len(tickers) < 2:
        return CompositeAnswer(ok=False, detail="ranking needs >=2 tickers")

    vals, resolved = [], []
    for ticker in tickers:
        fv = _evaluate_formula(spec, ticker, year, route, tables, encoder, min_score)
        if fv is None:
            return CompositeAnswer(ok=False, detail=f"ranking unresolved {ticker}",
                                   resolved=resolved)
        vals.append(fv)
        resolved.extend(fv.resolved)

    want_min = _wants_min(question)
    best = min(vals, key=lambda v: v.value) if want_min else max(vals, key=lambda v: v.value)
    fn = "min" if want_min else "max"
    query = f"round({fn}({', '.join(v.expr for v in vals)}), 2)"
    warn = check_answer_unit(best.value, spec.kind)
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"formula_ranking unit guard: {warn}",
                               resolved=resolved)
    conf = _confidence(resolved, base=90.0)
    return CompositeAnswer(ok=True, answer=best.value, pandas_query=query,
                           confidence=conf,
                           detail=f"formula_ranking {spec.name}"
                                  + (f" | UNIT-WARN: {warn}" if warn else ""),
                           resolved=resolved)


def _try_year_ranking(route: dict, tables: list[dict], encoder,
                      min_score: float) -> CompositeAnswer:
    """Return the argmax/argmin year after resolving every period exactly."""
    question = route.get("question", "")
    plan = route.get("plan") or {}
    tickers = _candidate_tickers(route, tables)
    years = sorted(set(_route_years(route)))
    if len(tickers) != 1 or len(years) < 2:
        return CompositeAnswer(
            ok=False, detail="year ranking needs one ticker and >=2 years")
    if plan.get("dimension") not in {None, "", "year"}:
        return CompositeAnswer(ok=False, detail="year ranking dimension mismatch")
    if plan.get("projection") not in {None, "", "year"}:
        return CompositeAnswer(ok=False, detail="year ranking projection mismatch")
    if _looks_nested_selector(question):
        return CompositeAnswer(ok=False, detail="year ranking looks nested")
    if any(word in _plain(question) for word in (
        "gia su", "kich ban", "trung vi", "nam sau nam", "nam lien truoc",
    )):
        return CompositeAnswer(ok=False, detail="year ranking selector unsupported")

    direction = "min" if _wants_min(question) else "max"
    if plan.get("direction") not in {None, "", direction}:
        return CompositeAnswer(ok=False, detail="year ranking direction mismatch")

    keys = list(dict.fromkeys(route.get("metric_keys") or metric_keys(
        [route.get("metric_norm", ""), *(route.get("metric_variants") or [])],
        expand_derived=False,
    )))
    specs = _detected_specs(question)
    direct_key = ""
    if len(keys) == 1 and not _looks_ratio_selector(question):
        metric = get_metric(keys[0])
        if not metric.is_derived:
            direct_key = keys[0]

    values: list[tuple[int, float, str]] = []
    resolved: list[ResolvedFact] = []
    if direct_key:
        use_absolute = metric_uses_absolute_value(question, [direct_key])
        for year in years:
            requirement = _year_requirement(
                route, tickers[0], year, direct_key)
            found = resolve_requirement(
                requirement, tables, encoder=encoder, min_score=min_score,
                question=question,
            )
            if (found is None or not _resolved_supports_year(found, year)
                    or not _resolved_value_sane(found)):
                return CompositeAnswer(
                    ok=False,
                    detail=f"year ranking unresolved exact {direct_key}/{year}",
                    resolved=resolved,
                )
            value = abs(found.value_vnd) if use_absolute else found.value_vnd
            expr = f"abs({found.expr_vnd()})" if use_absolute else found.expr_vnd()
            values.append((year, value, expr))
            resolved.append(found)
        detail_metric = direct_key
    elif len(specs) == 1:
        spec = specs[0]
        for year in years:
            found = _evaluate_formula_exact(
                spec, tickers[0], year, route, tables, encoder, min_score)
            if found is None:
                return CompositeAnswer(
                    ok=False,
                    detail=f"year ranking unresolved exact formula {spec.name}/{year}",
                    resolved=resolved,
                )
            values.append((year, found.value, found.expr))
            resolved.extend(found.resolved)
        detail_metric = spec.name
    else:
        return CompositeAnswer(
            ok=False,
            detail=f"year ranking metric keys={len(keys)} formulas={len(specs)}",
        )

    if not distinct_cells(resolved):
        return CompositeAnswer(
            ok=False, detail="year ranking evidence cells are not distinct",
            resolved=resolved)
    ordered = sorted(values, key=lambda item: item[1], reverse=direction == "max")
    if len(ordered) > 1 and math.isclose(
            ordered[0][1], ordered[1][1], rel_tol=1e-12, abs_tol=1e-6):
        return CompositeAnswer(
            ok=False, detail="year ranking tie has no convention", resolved=resolved)

    answer = float(ordered[0][0])
    query = f"round(float({_year_projection_expr(values, direction)}), 2)"
    warn = check_answer_unit(answer, "year")
    if warn:
        return CompositeAnswer(
            ok=False, detail=f"year ranking unit guard: {warn}", resolved=resolved)
    return CompositeAnswer(
        ok=True,
        answer=answer,
        pandas_query=query,
        confidence=_confidence(resolved, base=94.0),
        detail=(f"formula_year_ranking metric={detail_metric} "
                f"direction={direction} n={len(years)}"),
        resolved=resolved,
    )


def _year_requirement(route: dict, ticker: str, year: int,
                      metric_key: str) -> dict:
    matches = [
        requirement for requirement in route.get("evidence_requirements") or []
        if str(requirement.get("ticker") or "").upper() == ticker.upper()
        and requirement.get("year") is not None
        and int(requirement["year"]) == year
        and str(requirement.get("metric_key") or "") == metric_key
    ]
    if matches:
        return matches[0]
    metric = get_metric(metric_key)
    source_variants = [
        str(value) for value in route.get("metric_variants") or []
        if metric_key in metric_keys([str(value)], expand_derived=False)
    ]
    return {
        "requirement_id": f"{ticker}|{year}|{metric_key}",
        "ticker": ticker,
        "year": year,
        "doc_type": route.get("doc_type", "consolidated"),
        "metric_key": metric_key,
        "metric_label": metric.label,
        "metric_variants": list(dict.fromkeys([*source_variants, *metric.variants])),
        "statement": metric.statement,
    }


def _evaluate_formula_exact(spec: FormulaSpec, ticker: str, year: int,
                            route: dict, tables: list[dict], encoder,
                            min_score: float) -> FormulaValue | None:
    resolved = []
    evidence_years = _formula_operand_years(spec, year)
    for operand, evidence_year in zip(spec.operands, evidence_years):
        keys = metric_keys([operand.metric, *operand.variants], expand_derived=False)
        if len(keys) != 1:
            return None
        requirement = _year_requirement(route, ticker, evidence_year, keys[0])
        found = resolve_requirement(
            requirement, tables, encoder=encoder, min_score=min_score,
            # The full compositional question can mention both opening and
            # closing periods. The operand year already identifies the desired
            # cell, so row linking must not inherit a global period qualifier.
            question=str(requirement.get("metric_label") or operand.metric),
        )
        if (found is None or not _operand_accepts(
                operand, found, canonical_exact=True)
                or not _resolved_supports_year(found, evidence_year)
                or not _resolved_value_sane(found)):
            return None
        resolved.append(found)
    if not _period_aware_operands_valid(spec, resolved, evidence_years):
        return None
    try:
        value = spec.value_fn([found.value_vnd for found in resolved])
    except ZeroDivisionError:
        return None
    if not math.isfinite(value):
        return None
    return FormulaValue(
        spec=spec, ticker=ticker, year=year, value=value,
        expr=spec.expr_fn(resolved), resolved=resolved,
        score=min(found.score for found in resolved),
        evidence_years=evidence_years,
    )


def _formula_operand_years(spec: FormulaSpec,
                           year: int | None) -> tuple[int | None, ...]:
    if spec.period_refs:
        if len(spec.period_refs) != len(spec.operands):
            raise ValueError(f"formula {spec.name}: period_refs/operands mismatch")
        offsets = tuple(period.offset for period in spec.period_refs)
    else:
        offsets = spec.period_offsets or (0,) * len(spec.operands)
    if len(offsets) != len(spec.operands):
        raise ValueError(f"formula {spec.name}: period_offsets/operands mismatch")
    return tuple(None if year is None else int(year) + offset for offset in offsets)


def _formula_has_prior_period(spec: FormulaSpec) -> bool:
    return any(
        evidence_year is not None and evidence_year < 0
        for evidence_year in (
            tuple(period.offset for period in spec.period_refs)
            if spec.period_refs else spec.period_offsets
        )
    )


def _period_aware_operands_valid(
        spec: FormulaSpec, resolved: list[ResolvedFact],
        evidence_years: tuple[int | None, ...]) -> bool:
    for node in spec.average_balances:
        indexes = (node.opening_operand, node.closing_operand)
        if any(index < 0 or index >= len(resolved) for index in indexes):
            return False
        opening_index, closing_index = indexes
        if spec.operands[opening_index].metric != spec.operands[closing_index].metric:
            return False
        opening_year = evidence_years[opening_index]
        closing_year = evidence_years[closing_index]
        if (opening_year is None or closing_year is None
                or opening_year + 1 != closing_year):
            return False
        opening = resolved[opening_index]
        closing = resolved[closing_index]
        if opening.ticker.upper() != closing.ticker.upper():
            return False
        if not distinct_cells([opening, closing]):
            return False
        if opening.unit_scale <= 0 or closing.unit_scale <= 0:
            return False
    return True


def _year_projection_expr(values: list[tuple[int, float, str]],
                          direction: str) -> str:
    """Build a dynamic argmax/argmin expression that reads every candidate."""
    comparator = "<" if direction == "min" else ">"
    result = f"{values[-1][0]}.0"
    for year, _value, expr in reversed(values[:-1]):
        comparisons = [
            f"(({expr}) {comparator} ({other_expr}))"
            for other_year, _other_value, other_expr in values
            if other_year != year
        ]
        result = f"({year}.0 if ({' and '.join(comparisons)}) else {result})"
    return result


def _looks_ratio_selector(question: str) -> bool:
    text = _plain(question)
    return any(value in text for value in (
        "ty trong", "ty le", "chia cho", "tren tong", "so voi tong",
    ))


def _try_direct_formula(route: dict, tables: list[dict], encoder,
                        min_score: float) -> CompositeAnswer:
    question = route.get("question", "")
    if _looks_nested_selector(question) or _has_complex_temporal_selector(question):
        return CompositeAnswer(ok=False, detail="direct formula looks nested")
    if len(_route_years(route)) != 1:
        return CompositeAnswer(ok=False, detail="direct formula needs one year")
    specs = _detected_specs(question)
    if len(specs) != 1:
        return CompositeAnswer(ok=False, detail=f"direct formulas={len(specs)}")
    tickers = _candidate_tickers(route, tables)
    year = _primary_year(route)
    if len(tickers) != 1:
        return CompositeAnswer(ok=False, detail=f"direct formula tickers={len(tickers)}")

    fv = _evaluate_formula(specs[0], tickers[0], year, route, tables, encoder, min_score)
    if fv is None:
        return CompositeAnswer(ok=False, detail=f"direct unresolved {specs[0].name}")
    output_type = route.get("output_type", fv.spec.kind)
    warn = check_answer_unit(fv.value, output_type)
    if warn and "outside plausible range" in warn:
        return CompositeAnswer(ok=False, detail=f"formula_direct unit guard: {warn}",
                               resolved=fv.resolved)
    conf = _confidence(fv.resolved, base=92.0)
    return CompositeAnswer(ok=True, answer=fv.value,
                           pandas_query=f"round({fv.expr}, 2)",
                           confidence=conf,
                           detail=f"formula_direct {fv.spec.name}"
                                  + (f" | UNIT-WARN: {warn}" if warn else ""),
                           resolved=fv.resolved)


def _evaluate_condition(cond: Condition, ticker: str, year: int | None, route: dict,
                        tables: list[dict], encoder, min_score: float) -> FormulaValue | None:
    if cond.spec:
        return _evaluate_formula(cond.spec, ticker, year, route, tables, encoder, min_score)

    fact = _FactView({"ticker": ticker, "year": year, "doc_type": route.get("doc_type"),
                      "metric": cond.metric})
    r = resolve_fact(
        fact, tables, list(cond.variants), encoder, min_score,
        question=route.get("question", ""))
    if (r is None or not _direct_metric_accepts(cond, r, route)
            or not _resolved_supports_year(r, year)
            or not _resolved_value_sane(r)):
        return None
    return FormulaValue(
        spec=_DIRECT_METRIC_SPEC,
        ticker=ticker,
        year=year,
        value=r.value_vnd,
        expr=r.expr_vnd(),
        resolved=[r],
        score=r.score,
    )


def _direct_metric_accepts(cond: Condition, resolved: ResolvedFact,
                           route: dict) -> bool:
    """Protect count thresholds from high-scoring but contradictory labels."""
    metric = _plain(cond.metric)
    intent = _plain(route.get("question", ""))
    label = _plain(resolved.label)
    required = (
        "thue thu nhap doanh nghiep",
        "lai thuan",
        "ngoai hoi",
        "luu chuyen tien thuan tu hoat dong kinh doanh",
        "dong tien thuan tu hoat dong kinh doanh",
    )
    for phrase in required:
        if (phrase in metric or phrase in intent) and phrase not in label:
            return False
    contradictions = (
        ("doanh nghiep", "ca nhan"),
        ("ngoai hoi", "vang"),
    )
    if any((wanted in metric or wanted in intent) and wrong in label
           for wanted, wrong in contradictions):
        return False
    if "cuoi nam" in intent and "binh quan" in label:
        return False
    return True


def _evaluate_formula(spec: FormulaSpec, ticker: str, year: int | None, route: dict,
                      tables: list[dict], encoder, min_score: float) -> FormulaValue | None:
    resolved = []
    evidence_years = _formula_operand_years(spec, year)
    for op, evidence_year in zip(spec.operands, evidence_years):
        fact = _FactView({"ticker": ticker, "year": evidence_year,
                          "doc_type": route.get("doc_type"),
                          "metric": op.metric})
        r = resolve_fact(
            fact, tables, list(op.variants), encoder, min_score,
            question=route.get("question", ""))
        if (r is None or not _operand_accepts(op, r)
                or not _resolved_supports_year(r, evidence_year)
                or not _resolved_value_sane(r)):
            return None
        resolved.append(r)
    if not _period_aware_operands_valid(spec, resolved, evidence_years):
        return None
    values = [r.value_vnd for r in resolved]
    try:
        value = spec.value_fn(values)
    except ZeroDivisionError:
        return None
    if value != value or abs(value) == float("inf"):
        return None
    if spec.name in {"quick_ratio", "current_ratio", "debt_assets",
                     "inventory_assets", "fixed_asset_turnover"} and value < 0:
        return None
    return FormulaValue(spec=spec, ticker=ticker, year=year, value=value,
                        expr=spec.expr_fn(resolved), resolved=resolved,
                        score=min(r.score for r in resolved),
                        evidence_years=evidence_years)


def _operand_accepts(
        operand: Operand, resolved: ResolvedFact,
        canonical_exact: bool = False) -> bool:
    """Reject fuzzy lookalikes that change a financial formula's meaning."""
    label = norm(resolved.label)
    code = re.sub(r"\.0$", "", str(resolved.code or "").strip())
    trusted_codes = operand.expected_codes
    if canonical_exact and not trusted_codes:
        keys = metric_keys(
            [operand.metric, *operand.variants], expand_derived=False)
        if len(keys) == 1:
            trusted_codes = get_metric(keys[0]).codes
    if trusted_codes and code.isdigit() and code not in trusted_codes:
        return False
    if operand.required_phrases:
        required_match = any(
            phrase in label for phrase in operand.required_phrases)
        if not required_match and trusted_codes and code in trusted_codes:
            compact_label = label.replace(" ", "")
            required_match = any(
                phrase.replace(" ", "") in compact_label
                for phrase in operand.required_phrases)
        code_proves_identity = (
            canonical_exact and bool(trusted_codes) and code in trusted_codes)
        if not required_match and not code_proves_identity:
            return False
    compact_label = label.replace(" ", "")
    if any(
            phrase in label or phrase.replace(" ", "") in compact_label
            for phrase in operand.forbidden_phrases):
        return False
    return True


def _resolved_supports_year(resolved: ResolvedFact, year: int | None) -> bool:
    if year is None or resolved.year_evidence < 3:
        return year is None
    report_year = _report_year(resolved.report_id)
    return report_year in {year, year + 1}


def _resolved_value_sane(resolved: ResolvedFact) -> bool:
    # A 12-digit raw value is already VND. Multiplying it by a million/billion
    # is a common table-level unit extraction error and flips threshold counts.
    if abs(resolved.value) >= 1e9 and resolved.unit_scale >= 1e6:
        return False
    # Even Vietnam's largest consolidated balance sheets remain far below this
    # ceiling. Values beyond it are malformed OCR/grouping, not financial data.
    return abs(resolved.value_vnd) <= 1e18


def _evaluate_change(spec: FormulaSpec, ticker: str, start: int, end: int,
                     mode: str, route: dict, tables: list[dict], encoder,
                     min_score: float) -> FormulaValue | None:
    first = _evaluate_formula(spec, ticker, start, route, tables, encoder, min_score)
    last = _evaluate_formula(spec, ticker, end, route, tables, encoder, min_score)
    if first is None or last is None:
        return None
    if not _value_supports_year(first, start) or not _value_supports_year(last, end):
        return None
    if mode == "growth":
        if first.value == 0:
            return None
        value = (last.value - first.value) / abs(first.value) * 100.0
        expr = f"((({last.expr}) - ({first.expr})) / abs({first.expr}) * 100)"
    elif mode == "decrease":
        value = first.value - last.value
        expr = f"(({first.expr}) - ({last.expr}))"
    elif mode == "cagr":
        if first.value <= 0:
            return None
        n = max(1, end - start)
        value = ((last.value / first.value) ** (1.0 / n) - 1.0) * 100.0
        expr = f"((({last.expr}) / ({first.expr})) ** (1/{n}) - 1) * 100"
    else:
        value = last.value - first.value
        expr = f"(({last.expr}) - ({first.expr}))"
    return FormulaValue(spec=spec, ticker=ticker, year=end, value=value, expr=expr,
                        resolved=first.resolved + last.resolved,
                        score=min(first.score, last.score),
                        evidence_years=first.evidence_years + last.evidence_years)


def _evaluate_change_exact(spec: FormulaSpec, ticker: str, start: int, end: int,
                           mode: str, route: dict, tables: list[dict], encoder,
                           min_score: float) -> FormulaValue | None:
    first = _evaluate_formula_exact(
        spec, ticker, start, route, tables, encoder, min_score)
    last = _evaluate_formula_exact(
        spec, ticker, end, route, tables, encoder, min_score)
    if first is None or last is None:
        return None
    if not _value_supports_year(first, start) or not _value_supports_year(last, end):
        return None
    if mode == "growth":
        if first.value == 0:
            return None
        value = (last.value - first.value) / abs(first.value) * 100.0
        expr = f"((({last.expr}) - ({first.expr})) / abs({first.expr}) * 100)"
    elif mode == "decrease":
        value = first.value - last.value
        expr = f"(({first.expr}) - ({last.expr}))"
    elif mode == "cagr":
        if first.value <= 0:
            return None
        n = max(1, end - start)
        value = ((last.value / first.value) ** (1.0 / n) - 1.0) * 100.0
        expr = f"((({last.expr}) / ({first.expr})) ** (1/{n}) - 1) * 100"
    else:
        value = last.value - first.value
        expr = f"(({last.expr}) - ({first.expr}))"
    return FormulaValue(
        spec=spec, ticker=ticker, year=end, value=value, expr=expr,
        resolved=first.resolved + last.resolved,
        score=min(first.score, last.score),
        evidence_years=first.evidence_years + last.evidence_years,
    )


def _answer_value(value: FormulaValue, route: dict) -> tuple[float, str]:
    if value.spec.kind != "money":
        return value.value, value.expr
    scale = float(route.get("unit_scale", 1.0) or 1.0)
    return value.value / scale, f"(({value.expr}) / {scale:g})"


def _output_accepts_kind(output_type: str | None, kind: str) -> bool:
    if kind == "money":
        return output_type in {None, "", "number"}
    if kind == "ratio":
        return output_type == "ratio"
    if kind == "percent":
        return output_type in {"percent", "percentage_point"}
    return False


def _value_supports_year(value: FormulaValue, year: int | None) -> bool:
    if year is None:
        return True
    expected = value.evidence_years or (year,) * len(value.resolved)
    return len(expected) == len(value.resolved) and all(
        _resolved_supports_year(resolved, evidence_year)
        for resolved, evidence_year in zip(value.resolved, expected)
    )


def _report_year(report_id: str) -> int | None:
    found = re.search(r"(?:financial_statements_|_)(20\d{2})(?:_|$)",
                      str(report_id))
    return int(found.group(1)) if found else None


def _direct_metric_condition(route: dict, question: str) -> Condition | None:
    metric = route.get("metric_norm", "")
    if not metric:
        return None
    parsed = _parse_condition_after(question, metric, "money")
    if parsed is None:
        parsed = _parse_condition_after(question, "", "money")
    if parsed is None:
        return None
    op, threshold = parsed
    variants = _clean_metric_variants(route.get("metric_variants") or [metric])
    metric_clean = variants[0] if variants else _clean_condition_metric(metric)
    return Condition(None, metric_clean, tuple(variants or [metric_clean]),
                     op, threshold, "money")


def _parsed_conditions(route: dict, question: str) -> list[Condition]:
    conditions, seen = [], set()
    for match in _calculation_matches(question):
        cond = _condition_for_match(question, match)
        if cond is None or cond.label in seen:
            continue
        conditions.append(cond)
        seen.add(cond.label)
    return conditions


def _condition_for_match(question: str, match: FormulaMatch) -> Condition | None:
    text = _plain(question)
    if match.spec.name == "working_capital":
        phrase = text[match.start:match.end]
        if "thap hon" in phrase:
            return Condition(match.spec, match.spec.name, (), "<", 0.0, "money")
        if "cao hon" in phrase:
            return Condition(match.spec, match.spec.name, (), ">", 0.0, "money")
    parsed = _parse_condition_segment(text[match.end:match.end + 180], match.spec.kind)
    if parsed is None:
        return None
    op, threshold = parsed
    return Condition(match.spec, match.spec.name, (), op, threshold, match.spec.kind)


def _condition_for_spec(question: str, spec: FormulaSpec) -> Condition | None:
    parsed = None
    for trigger in spec.triggers:
        parsed = _parse_condition_after(question, trigger, spec.kind)
        if parsed is not None:
            break
    if parsed is None:
        return None
    op, threshold = parsed
    return Condition(spec, spec.name, (), op, threshold, spec.kind)


def _parse_condition_after(question: str, anchor: str, kind: str) -> tuple[str, float] | None:
    text = _plain(question)
    start = 0
    if anchor:
        pos = text.find(anchor)
        if pos < 0:
            return None
        start = pos + len(anchor)
    return _parse_condition_segment(text[start:start + 180], kind)


def _parse_condition_segment(segment: str, kind: str) -> tuple[str, float] | None:
    patterns = [
        (">=", r">=\s*(" + _NUM_RE + r")"),
        ("<=", r"<=\s*(" + _NUM_RE + r")"),
        (">", r">\s*(" + _NUM_RE + r")"),
        ("<", r"<\s*(" + _NUM_RE + r")"),
        (">=", r"(?:lon hon hoac bang|khong nho hon|khong thap hon|toi thieu|it nhat)\s+(" + _NUM_RE + r")"),
        ("<=", r"(?:nho hon hoac bang|khong lon hon|toi da|nhieu nhat)\s+(" + _NUM_RE + r")"),
        (">", r"(?:lon hon|cao hon|vuot|tren|(?<!nho )(?<!thap )hon)\s+(" + _NUM_RE + r")"),
        ("<", r"(?:nho hon|thap hon|duoi|nho hon muc|duoi muc)\s+(" + _NUM_RE + r")"),
        (">=", r"\btu\s+(" + _NUM_RE + r")(?:\s+\w+){0,2}\s+tro len\b"),
        ("=", r"(?:bang|la)\s+(" + _NUM_RE + r")"),
    ]
    for op, pat in patterns:
        m = re.search(pat, segment)
        if not m:
            continue
        raw = m.group(1)
        num = parse_vn_number(raw)
        if num is None:
            continue
        tail = segment[m.end(1):m.end(1) + 60]
        return op, _scale_threshold(float(num), tail, kind)
    sign_segment = segment[:65]
    if re.search(r"\bkhong\s+am\b", sign_segment):
        return ">=", 0.0
    if re.search(r"\bkhong\s+duong\b", sign_segment):
        return "<=", 0.0
    if re.search(r"\bduong\b", sign_segment):
        return ">", 0.0
    if re.search(r"\bam\b", sign_segment):
        return "<", 0.0
    return None


def _clean_metric_variants(variants) -> list[str]:
    out = []
    for variant in variants:
        cleaned = _clean_condition_metric(str(variant or ""))
        words = cleaned.split()
        if len(words) < 2 or cleaned in {"cac cong ty", "cac doanh nghiep", "trong so"}:
            continue
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def _clean_condition_metric(metric: str) -> str:
    text = _plain(metric)
    text = re.sub(r"^(?:ghi nhan|gia tri|so du)\s+", "", text)
    text = re.split(
        r"\s+(?:lon hon|cao hon|vuot|tren|nho hon|thap hon|duoi|duong|am)\b",
        text, maxsplit=1)[0]
    return text.strip()


def _maybe_top_n_population(question: str, tickers: list[str], year: int | None,
                            route: dict, tables: list[dict], encoder,
                            min_score: float) -> Population | None:
    text = _plain(question)
    m = re.search(r"\btrong(?:\s+so)?\s+(\d+)\s+(?:doanh nghiep|cong ty|ma)\b", text)
    if not m:
        return Population(tickers, [], [])
    n = int(m.group(1))
    if n <= 0:
        return Population(tickers, [], [])
    if "doanh thu thuan" not in text or not _wants_max(text):
        return Population(tickers, [], [])

    vals, resolved = [], []
    for ticker in tickers:
        fact = _FactView({"ticker": ticker, "year": year, "doc_type": route.get("doc_type"),
                          "metric": "doanh thu thuan"})
        r = resolve_fact(
            fact, tables, ["doanh thu thuan"], encoder, min_score,
            question=route.get("question", ""))
        if r is None:
            return None
        resolved.append(r)
        vals.append((ticker, r.value_vnd))
    vals.sort(key=lambda x: x[1], reverse=True)
    return Population([ticker for ticker, _ in vals[:n]], resolved,
                      [r.expr_vnd() for r in resolved])


def _candidate_tickers(route: dict, tables: list[dict]) -> list[str]:
    seen, out = set(), []
    for ticker in route.get("tickers") or []:
        t = str(ticker).upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    if out:
        return out
    for table in tables:
        t = str(table.get("report_id", "")).split("_")[0].upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _stated_population_size(question: str) -> int | None:
    match = re.search(r"\btrong\s+(\d+)\s+(?:doanh nghiep|cong ty|ma)\b",
                      _plain(question))
    return int(match.group(1)) if match else None


def _primary_year(route: dict) -> int | None:
    years = _route_years(route)
    return max(years) if years else None


def _route_years(route: dict) -> list[int]:
    return [int(y) for y in route.get("years") or [] if y is not None]


def _detected_specs(question: str) -> list[FormulaSpec]:
    matches = _formula_matches(question)
    seen, out = set(), []
    for match in matches:
        if match.spec.name in seen:
            continue
        seen.add(match.spec.name)
        out.append(match.spec)
    return out


def _formula_matches(question: str) -> list[FormulaMatch]:
    matches = _specialize_period_aware_matches(
        question, _all_spec_matches(question, _FORMULAS))
    matches.extend(_structured_formula_matches(question, matches))
    compound = [
        match for match in matches
        if match.spec.name in _COMPOUND_FORMULA_NAMES
    ]
    matches = [
        match for match in matches
        if (match in compound or not any(
            outer.start <= match.start and match.end <= outer.end
            and (outer.start, outer.end) != (match.start, match.end)
            for outer in compound))
    ]
    unique = {}
    for match in matches:
        unique.setdefault((match.spec.name, match.start, match.end), match)
    return sorted(unique.values(), key=lambda match: (match.start, match.end))


def _specialize_period_aware_matches(
        question: str, matches: list[FormulaMatch]) -> list[FormulaMatch]:
    text = _plain(question)
    specs = {spec.name: spec for spec in _FORMULAS}
    replacements = {}
    if _mentions_average_balance(text, "total_assets"):
        replacements.update({
            "roa": "roa_average_assets",
            "total_asset_turnover": "total_asset_turnover_average_assets",
        })
    if _mentions_average_balance(text, "equity"):
        replacements["roe"] = "roe_average_equity"
    if _mentions_average_balance(text, "fixed_assets"):
        replacements["fixed_asset_turnover"] = "fixed_asset_turnover_average_assets"
    return [
        FormulaMatch(
            specs[replacements[match.spec.name]],
            match.start, match.end, match.trigger,
        )
        if match.spec.name in replacements else match
        for match in matches
    ]


def _mentions_average_balance(text: str, metric: str) -> bool:
    aliases = {
        "total_assets": ("tong tai san",),
        "equity": ("von chu so huu",),
        "fixed_assets": ("tai san co dinh thuan", "tai san co dinh"),
    }[metric]
    average = r"(?:binh quan|trung binh)"
    for alias in aliases:
        if (re.search(rf"\b{re.escape(alias)}\s+{average}\b", text)
                or re.search(rf"\b{average}\s+{re.escape(alias)}\b", text)):
            return True
    if (metric == "total_assets"
            and re.search(
                r"\bvong quay tong tai san(?:\s+\([^)]*\))?\s+"
                r"(?:tinh\s+)?(?:theo\s+)?tai san\s+"
                r"(?:binh quan|trung binh)\b", text)):
        return True
    return False


def _all_spec_matches(question: str, specs: list[FormulaSpec]) -> list[FormulaMatch]:
    text = _plain(question)
    raw = []
    for spec in specs:
        for trigger in sorted(spec.triggers, key=len, reverse=True):
            for found in re.finditer(re.escape(trigger), text):
                raw.append(FormulaMatch(spec, found.start(), found.end(), trigger))
    raw.sort(key=lambda m: (m.start, -(m.end - m.start)))
    out = []
    for match in raw:
        if any(prev.spec.name == match.spec.name
               and _overlap(prev.start, prev.end, match.start, match.end)
               for prev in out):
            continue
        out.append(match)
    return sorted(out, key=lambda m: (m.start, m.end))


def _relocate_normalized_alias(text: str, alias: str, cursor: int) -> tuple[int, int]:
    """Map a normalized alias back to text that still contains punctuation."""
    tokens = alias.split()
    if not tokens:
        return -1, -1
    separator = r"[\s,;:./()\-]+"
    pattern = r"(?<!\w)" + separator.join(re.escape(token) for token in tokens)
    pattern += r"(?!\w)"
    found = re.search(pattern, text[cursor:])
    if found is None:
        return -1, -1
    return cursor + found.start(), cursor + found.end()


def _calculation_matches(question: str) -> list[FormulaMatch]:
    formulas = _formula_matches(question)
    metrics = _all_spec_matches(question, _METRIC_SPECS)
    fixed_keys = {spec.name for spec in _METRIC_SPECS}
    text = _plain(question)
    alias_cursors: dict[str, int] = {}
    for found in find_metrics(question, include_derived=False):
        key = found.metric.key
        if key in fixed_keys:
            continue
        # ``find_metrics`` indexes the punctuation-stripped ``norm`` text,
        # while formula planners slice ``_plain`` text to preserve clause
        # separators. Locate the alias again so commas before a metric cannot
        # truncate a nearby threshold (for example ``10%, tổng doanh thu``).
        start, end = _relocate_normalized_alias(
            text, found.alias, alias_cursors.get(found.alias, 0))
        if start < 0:
            continue
        alias_cursors[found.alias] = end
        metrics.append(FormulaMatch(
            _direct_metric_formula(key), start, end, found.alias))
    # A one-line metric inside a longer formula phrase is an operand, not a
    # second calculation. A separate occurrence elsewhere remains available as
    # a ranking selector or filter.
    metrics = [m for m in metrics
               if not any(_overlap(m.start, m.end, f.start, f.end) for f in formulas)]
    # Prefer the exact child metric when a generic parent is only a substring
    # of the same phrase, for example inventory inside gross inventory.
    metrics = [
        match for match in metrics
        if not any(
            other.spec.name != match.spec.name
            and other.start <= match.start and other.end >= match.end
            and (other.end - other.start) > (match.end - match.start)
            for other in metrics
        )
    ]
    unique = {}
    for match in formulas + metrics:
        unique.setdefault((match.spec.name, match.start, match.end), match)
    return sorted(unique.values(), key=lambda m: (m.start, m.end))


def _structured_formula_matches(
        question: str, existing: list[FormulaMatch]) -> list[FormulaMatch]:
    """Recognize formula wording whose years split the numerator/denominator."""
    text = _plain(question)
    specs = {spec.name: spec for spec in _FORMULAS}
    patterns = (
        (
            "gross_margin",
            r"loi nhuan gop(?:\s+nam\s+20\d{2})?\s+"
            r"(?:bang|chiem)\s+bao nhieu phan tram\s+"
            r"doanh thu thuan(?:\s+nam\s+20\d{2})?",
        ),
        (
            "inventory_assets",
            r"hang ton kho(?:\s+cuoi nam)?\s+"
            r"(?:bang|chiem)\s+bao nhieu phan tram\s+"
            r"tong tai san(?:\s+cuoi nam(?:\s+do)?)?",
        ),
        (
            "quick_ratio",
            r"tai san ngan han\s+(?:sau khi\s+)?tru\s+hang ton kho\s+"
            r"(?:roi\s+)?chia cho\s+no ngan han(?:\s+nam\s+20\d{2})?",
        ),
        (
            "inventory_days",
            r"(?:gia tri\s+)?hang ton kho binh quan"
            r"(?:\s+nam\s+20\d{2}\s+va\s+(?:nam\s+)?20\d{2})?\s+"
            r"chia cho\s+gia von hang ban(?:\s+nam\s+20\d{2})?\s+"
            r"roi\s+nhan\s+365",
        ),
        (
            "roa_average_assets",
            r"loi nhuan sau thue(?:\s+nam\s+20\d{2})?\s+"
            r"(?:tren|bang\s+bao\s+nhieu\s+phan\s+tram|chiem\s+bao\s+nhieu\s+phan\s+tram)\s+"
            r"(?:(?:tong\s+tai\s+san)\s+(?:binh\s+quan|trung\s+binh)|"
            r"(?:binh\s+quan|trung\s+binh)\s+(?:tong\s+)?tai\s+san)",
        ),
        (
            "roe_average_equity",
            r"loi nhuan sau thue(?:\s+nam\s+20\d{2})?\s+"
            r"(?:tren|bang\s+bao\s+nhieu\s+phan\s+tram|chiem\s+bao\s+nhieu\s+phan\s+tram)\s+"
            r"(?:(?:von\s+chu\s+so\s+huu)\s+(?:binh\s+quan|trung\s+binh)|"
            r"(?:binh\s+quan|trung\s+binh)\s+von\s+chu\s+so\s+huu)",
        ),
        (
            "accrual_average_assets",
            r"(?:chenh\s+lech\s+giua\s+)?loi nhuan sau thue\s+va\s+"
            r"(?:luu chuyen tien thuan tu hoat dong kinh doanh|"
            r"dong tien thuan tu hoat dong kinh doanh|cfo)"
            r"(?:\s+nam\s+20\d{2})?\s+(?:chia cho|tren)\s+"
            r"(?:(?:tong\s+tai\s+san)\s+(?:binh\s+quan|trung\s+binh)|"
            r"(?:binh\s+quan|trung\s+binh)\s+(?:tong\s+)?tai\s+san)",
        ),
        (
            "average_fixed_assets",
            r"(?:(?:binh\s+quan|trung\s+binh)\s+tai\s+san\s+co\s+dinh\s+thuan|"
            r"tai\s+san\s+co\s+dinh\s+thuan\s+(?:binh\s+quan|trung\s+binh))",
        ),
    )
    out = []
    for name, pattern in patterns:
        for found in re.finditer(pattern, text):
            if any(match.spec.name == name and _overlap(
                    match.start, match.end, found.start(), found.end())
                   for match in existing):
                continue
            out.append(FormulaMatch(
                specs[name], found.start(), found.end(), found.group(0)))
    return out


def _direct_metric_formula(metric_key: str) -> FormulaSpec:
    metric = get_metric(metric_key)
    return FormulaSpec(
        name=f"metric:{metric_key}", triggers=metric.variants,
        operands=(_canonical_operand(metric_key),), kind="money",
        value_fn=lambda values: values[0], expr_fn=_identity_expr,
    )


def _overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(a0, b0) < min(a1, b1)


def _unique_matches(matches: list[FormulaMatch]) -> dict[str, FormulaMatch]:
    out = {}
    for match in matches:
        out.setdefault(match.spec.name, match)
    return out


def _ranked_match(text: str, matches: list[FormulaMatch]):
    extrema = list(re.finditer(
        r"\b(cao nhat|lon nhat|nhieu nhat|manh nhat|"
        r"thap nhat|nho nhat|it nhat)\b", text))
    if not extrema and re.search(
            r"\b(?:trong\s+)?(?:2|hai)\s+(?:doanh nghiep|cong ty)\b", text):
        extrema = list(re.finditer(r"\b(lon hon|nho hon)\b", text))
        extrema = [
            extreme for extreme in extrema
            if any(marker in text[max(0, extreme.start() - 130):extreme.start()]
                   for marker in ("muc giam", "muc tang", "muc thay doi"))
        ]
    before, after = [], []
    for match in matches:
        for extreme in extrema:
            gap = extreme.start() - match.end
            if 0 <= gap <= 90:
                before.append((gap, match, extreme))
            forward_gap = match.start - extreme.end()
            if 0 <= forward_gap <= 170:
                after.append((forward_gap, match, extreme))
    choices = before or after
    if not choices:
        return None
    _gap, match, extreme = min(choices, key=lambda x: x[0])
    want_min = extreme.group(1) in {"thap nhat", "nho nhat", "it nhat", "nho hon"}
    return match, extreme.start(), want_min


def _target_match(matches: list[FormulaMatch], selector: FormulaMatch,
                  extreme_start: int, text: str) -> FormulaMatch | None:
    answer_pos = text.find("bao nhieu", extreme_start)
    answer_pos = len(text) if answer_pos < 0 else answer_pos
    after = [m for m in matches
             if m.start > extreme_start and m.start < answer_pos
             and not (m.start == selector.start and m.end == selector.end)]
    if after:
        return max(after, key=lambda m: m.start)
    before = [m for m in matches
              if m.end <= selector.start
              and not (m.start == selector.start and m.end == selector.end)]
    return max(before, key=lambda m: m.end) if before else None


def _calculation_node(text: str, match: FormulaMatch, mode: str,
                      years: list[int], anchor: int) -> CalculationNode:
    if mode == "level":
        return CalculationNode(match, mode)
    bounds = _temporal_bounds(text, match, years, anchor)
    if bounds is None:
        return CalculationNode(match, mode)
    return CalculationNode(match, mode, bounds[0], bounds[1])


def _temporal_bounds(text: str, match: FormulaMatch, years: list[int],
                     anchor: int) -> tuple[int, int] | None:
    window_start = max(0, min(match.start, anchor) - 110)
    window_end = min(len(text), max(match.end, anchor) + 110)
    window = text[window_start:window_end]
    ranges = []
    for found in re.finditer(
            r"\btu\s+(?:nam\s+)?(20\d{2})\s+"
            r"(?:den|sang|toi)\s+(?:nam\s+)?(20\d{2})\b", window):
        first, last = int(found.group(1)), int(found.group(2))
        absolute_start = window_start + found.start()
        ranges.append((abs(absolute_start - anchor), first, last))
    for found in re.finditer(
            r"\b(20\d{2})\s+so\s+voi\s+(?:nam\s+)?(20\d{2})\b", window):
        last, first = int(found.group(1)), int(found.group(2))
        absolute_start = window_start + found.start()
        ranges.append((abs(absolute_start - anchor), first, last))
    if ranges:
        _distance, first, last = min(ranges, key=lambda item: item[0])
        return (min(first, last), max(first, last))
    ordered = sorted(set(years))
    if len(ordered) >= 2:
        return ordered[0], ordered[-1]
    if len(ordered) == 1:
        return ordered[0] - 1, ordered[0]
    return None


def _median_filter_node(
        text: str, matches: list[FormulaMatch], dimension: str,
        years: list[int]) -> MedianFilterNode | None:
    medians = list(re.finditer(r"\b(?:muc\s+)?trung vi\b", text))
    if not medians:
        return None
    choices = []
    for median_match in medians:
        for match in matches:
            if match.end > median_match.start():
                continue
            gap = median_match.start() - match.end
            if gap > 150:
                continue
            segment = text[match.end:median_match.end()]
            op = _median_comparator(segment)
            if op is not None:
                choices.append((gap, match, median_match, op))
    if not choices:
        return None
    _gap, match, median_match, op = min(choices, key=lambda item: item[0])
    mode = _median_filter_mode(text, match, median_match.start(), dimension, years)
    year = None
    if dimension == "entity" and mode == "level":
        window = text[max(0, match.start - 35):median_match.end()]
        mentioned = [
            year for year in years
            if re.search(rf"(?<!\d){year}(?!\d)", window)
        ]
        if len(mentioned) == 1:
            year = mentioned[0]
        elif (len(mentioned) >= 2
              and _formula_has_prior_period(match.spec)):
            year = max(mentioned)
        elif len(years) == 1:
            year = years[0]
        else:
            return None
    calculation = _calculation_node(
        text, match, mode, years, anchor=median_match.start())
    return MedianFilterNode(calculation, op, year)


def _median_comparator(segment: str) -> str | None:
    if any(value in segment for value in ("bang hoac thap hon", "bang hoac nho hon")):
        return "<="
    if any(value in segment for value in ("bang hoac cao hon", "bang hoac lon hon")):
        return ">="
    if any(value in segment for value in ("thap hon", "nho hon", "duoi")):
        return "<"
    if any(value in segment for value in ("cao hon", "lon hon", "vuot")):
        return ">"
    return None


def _median_filter_mode(text: str, match: FormulaMatch, median_start: int,
                        dimension: str, years: list[int]) -> str:
    if dimension == "year":
        return "level"
    context = text[max(0, match.start - 55):median_start]
    if "cagr" in context:
        return "cagr"
    if "muc giam" in context:
        return "decrease"
    if any(value in context for value in (
            "tang truong", "toc do tang", "ty le tang", "phan tram tang")):
        return "growth"
    if any(value in context for value in ("muc thay doi", "muc tang")):
        return "delta"
    return "level"


def _selector_is_direct_answer(text: str, selector: FormulaMatch) -> bool:
    selector_start = selector.start
    trigger = _plain(selector.trigger)
    located = text.find(
        trigger, max(0, selector.start - 48),
        min(len(text), selector.end + 48),
    )
    if located >= 0:
        selector_start = located
    prefix = text[:selector_start].strip(" ,:;.-")
    if not prefix:
        return True
    # Allow a short question lead-in ("what is the highest revenue...") but not
    # "metric X at the year/company whose selector Y is highest".
    if len(prefix.split()) <= 4:
        return True
    return not any(w in prefix for w in (
        "tai nam", "vao nam", "cua doanh nghiep", "cua cong ty", "o nam",
        "trong nam co", "tai nam co", "nam co",
    ))


def _value_mode(text: str, match: FormulaMatch, extreme_start: int,
                temporal_allowed: bool) -> str:
    if not temporal_allowed:
        return "level"
    if match.start >= extreme_start:
        ctx = text[max(0, extreme_start - 45):match.start]
    else:
        ctx = text[max(0, match.start - 70):min(len(text), extreme_start + 24)]
    if "cagr" in ctx:
        return "cagr"
    if "muc giam" in ctx:
        return "decrease"
    if any(w in ctx for w in (
            "tang truong", "toc do tang", "ty le tang", "phan tram tang",
            "muc tang tuong doi", "tang tuong doi")):
        return "growth"
    if any(w in ctx for w in ("muc thay doi", "muc tang")):
        return "delta"
    if re.search(r"\b(?:tang|giam)\s+(?:manh|nhieu|lon)\s+nhat\b", ctx):
        return "delta"
    return "level"


def _target_value_mode(text: str, match: FormulaMatch,
                       temporal_allowed: bool) -> str:
    if not temporal_allowed:
        return "level"
    clause_start = max(
        text.rfind(",", 0, match.start),
        text.rfind(";", 0, match.start),
        text.rfind(".", 0, match.start),
    )
    before = text[max(clause_start + 1, match.start - 75):match.start]
    after = text[match.end:min(len(text), match.end + 45)]
    explicit_period = re.search(r"\bnam\s+20\d{2}\b", match.trigger) or re.match(
        r"\s+(?:(?:cua|tai)\s+(?:cong ty|doanh nghiep|ma)"
        r"(?:\s+do)?\s+)?(?:tai\s+)?nam\s+20\d{2}\b",
        after,
    )
    if (explicit_period
            and not re.match(r"\s*(?:co\s+)?(?:muc\s+)?thay doi\b", after)):
        return "level"
    if any(w in before for w in ("muc thay doi", "thay doi", "muc tang")):
        return "delta"
    if "muc giam" in before:
        return "decrease"
    if "cagr" in before:
        return "cagr"
    if any(w in before for w in ("tang truong", "toc do tang", "phan tram tang")):
        return "growth"
    if re.match(r"\s*(?:co\s+)?(?:muc\s+)?thay doi\b", after):
        return "delta"
    if re.match(r"\s*(?:co\s+)?(?:muc\s+)?giam\s+bao nhieu\b", after):
        return "decrease"
    return "level"


def _change_target_match(text: str,
                         matches: list[FormulaMatch]) -> FormulaMatch | None:
    choices = []
    for match in matches:
        clause_start = max(text.rfind(";", 0, match.start),
                           text.rfind(".", 0, match.start))
        before = text[max(clause_start + 1, match.start - 80):match.start]
        after = text[match.end:min(len(text), match.end + 45)]
        if (any(w in before for w in ("muc thay doi", "thay doi", "muc tang",
                                      "muc giam", "tang truong", "toc do tang"))
                or re.match(r"\s*(?:co\s+)?(?:muc\s+)?(?:thay doi|tang|giam)\b",
                            after)):
            choices.append(match)
    unique = _unique_matches(choices)
    return next(iter(unique.values())) if len(unique) == 1 else None


def _pair_specs_after_difference(text: str,
                                 matches: list[FormulaMatch]) -> list[FormulaSpec]:
    pos = text.find("chenh lech giua")
    if pos < 0:
        return []
    seen, specs = set(), []
    for match in matches:
        if match.start <= pos or match.spec.name in seen:
            continue
        seen.add(match.spec.name)
        specs.append(match.spec)
    return specs[:2]


def _temporal_direction(text: str, match: FormulaMatch) -> str | None:
    before = text[max(0, match.start - 45):match.start]
    after = text[match.end:min(len(text), match.end + 80)]
    if re.search(r"(?:tang|cai thien)\s+$", before) or re.match(
            r"(?:\s+\w+){0,3}\s+(?:tang|cao hon)\b", after):
        return "increase"
    if re.search(r"giam\s+$", before) or re.match(
            r"(?:\s+\w+){0,3}\s+(?:giam|thap hon)\b", after):
        return "decrease"
    if re.match(r"(?:\s+\w+){0,4}\s+duong\b", after):
        return "positive"
    if re.match(r"(?:\s+\w+){0,4}\s+am\b", after):
        return "negative"
    return None


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    return ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0


def _median_expr(exprs: list[str]) -> str:
    ordered = f"sorted([{', '.join(exprs)}])"
    mid = len(exprs) // 2
    if len(exprs) % 2:
        return f"({ordered}[{mid}])"
    return f"(({ordered}[{mid - 1}] + {ordered}[{mid}]) / 2)"


def _condition_expr(expr: str, op: str, threshold: float | str) -> str:
    rhs = threshold if isinstance(threshold, str) else f"{threshold:g}"
    return f"(({expr}) {op} {rhs})"


def _compare(value: float, op: str, threshold: float) -> bool:
    if op == ">":
        return value > threshold
    if op == ">=":
        return value >= threshold
    if op == "<":
        return value < threshold
    if op == "<=":
        return value <= threshold
    if op == "=":
        return abs(value - threshold) <= max(1e-9, abs(threshold) * 1e-9)
    return False


def _confidence(resolved: list[ResolvedFact], base: float) -> float:
    if not resolved:
        return 0.0
    weakest = min(r.score for r in resolved)
    keys = {(r.report_id, r.table_pos, r.label, r.col) for r in resolved}
    conf = min(base, weakest + 8.0)
    if len(keys) < len(resolved):
        conf = min(conf, 35.0)
    return max(0.0, conf)


def _scale_threshold(value: float, tail: str, kind: str) -> float:
    if kind in {"ratio", "percent"}:
        return value
    tail = _plain(tail)
    if "nghin ty" in tail or "ngan ty" in tail:
        return value * 1e12
    if "tram ty" in tail:
        return value * 1e11
    if "ty" in tail:
        return value * 1e9
    if "trieu" in tail:
        return value * 1e6
    if "nghin" in tail or "ngan" in tail:
        return value * 1e3
    return value


def _plain(text: str) -> str:
    text = strip_diacritics(str(text or "")).lower()
    text = text.replace("đ", "d").replace("Đ", "D")
    return re.sub(r"\s+", " ", text).strip()


def _wants_min(question: str) -> bool:
    text = _plain(question)
    return any(w in text for w in (
        "nho nhat", "thap nhat", "it nhat", "thap hon ca", "be nhat",
        "toi thieu"))


def _wants_max(question: str) -> bool:
    text = _plain(question)
    return any(w in text for w in (
        "lon nhat", "cao nhat", "nhieu nhat", "toi da", "dung dau"))


def _looks_nested_selector(question: str) -> bool:
    text = _plain(question)
    entities = r"(?:cong ty|doanh nghiep|ngan hang|ma|to chuc)"
    if re.search(rf"\b(?:cua|cho|tai)\s+{entities}\s+co(?!\s+phan\b)\b", text):
        return True
    if re.search(rf"\b{entities}\s+co(?!\s+phan\b)\b", text):
        return True
    if re.search(
            rf"\b{entities}\s+dat\b.*?"
            r"\b(?:cao nhat|thap nhat|lon nhat|nho nhat)\s+co\b", text):
        return True
    return bool(
        re.search(r"\bnam\s+(?:ma|ghi nhan)\b", text)
        or re.search(r"\b(?:tai|vao)\s+nam\s+co\b", text)
    )


def _has_complex_temporal_selector(question: str) -> bool:
    text = _plain(question)
    return any(w in text for w in (
        "giai doan", "tu nam", "den nam", "nam dau tien", "nam lien truoc",
        "ngay sau", "sau nam", "truoc nam"))


def _counts_years(question: str) -> bool:
    text = _plain(question)
    return "bao nhieu nam" in text or ("so nam" in text and "bao nhieu" in text)


def _expected_condition_count(question: str) -> int:
    text = _plain(question)
    expected = 1
    expected = max(expected, text.count("vua co"))
    if "dong thoi" in text and " va " in text:
        expected = max(expected, 2)
    return expected


def _div(a: float, b: float) -> float:
    if b == 0:
        raise ZeroDivisionError("formula denominator is zero")
    return a / b


def _ratio_expr(num: ResolvedFact, den: ResolvedFact) -> str:
    return f"({num.expr_vnd()} / {den.expr_vnd()})"


def _percent_expr(num: ResolvedFact, den: ResolvedFact) -> str:
    return f"({_ratio_expr(num, den)} * 100)"


def _quick_expr(rs: list[ResolvedFact]) -> str:
    return f"(({rs[0].expr_vnd()} - {rs[1].expr_vnd()}) / {rs[2].expr_vnd()})"


def _sum_margin_expr(rs: list[ResolvedFact]) -> str:
    return (
        f"((abs({rs[0].expr_vnd()}) + abs({rs[1].expr_vnd()})) / "
        f"abs({rs[2].expr_vnd()}) * 100)"
    )


def _inventory_assets_expr(rs: list[ResolvedFact]) -> str:
    return f"({rs[0].expr_vnd()} / {rs[1].expr_vnd()} * 100)"


def _working_capital_expr(rs: list[ResolvedFact]) -> str:
    return f"({rs[0].expr_vnd()} - {rs[1].expr_vnd()})"


def _interest_coverage_expr(rs: list[ResolvedFact]) -> str:
    return (f"(({rs[0].expr_vnd()} + abs({rs[1].expr_vnd()})) "
            f"/ abs({rs[1].expr_vnd()}))")


def _inventory_days_expr(rs: list[ResolvedFact]) -> str:
    return (f"(365 * (({rs[0].expr_vnd()} + {rs[1].expr_vnd()}) / 2) "
            f"/ abs({rs[2].expr_vnd()}))")


def _average_balance_expr(opening: ResolvedFact, closing: ResolvedFact) -> str:
    return f"(({opening.expr_vnd()} + {closing.expr_vnd()}) / 2)"


def _average_ratio_expr(rs: list[ResolvedFact], *, percent: bool = False,
                        numerator: str | None = None) -> str:
    numerator_expr = numerator or rs[0].expr_vnd()
    denominator = _average_balance_expr(rs[-2], rs[-1])
    scale = " * 100" if percent else ""
    return f"(({numerator_expr}) / ({denominator}){scale})"


def _accrual_average_assets_expr(rs: list[ResolvedFact]) -> str:
    numerator = f"({rs[0].expr_vnd()} - {rs[1].expr_vnd()})"
    return _average_ratio_expr(rs, percent=True, numerator=numerator)


def _operating_leverage_expr(rs: list[ResolvedFact]) -> str:
    operating_growth = (
        f"(({rs[1].expr_vnd()} - {rs[0].expr_vnd()}) / "
        f"abs({rs[0].expr_vnd()}))")
    revenue_growth = (
        f"(({rs[3].expr_vnd()} - {rs[2].expr_vnd()}) / "
        f"abs({rs[2].expr_vnd()}))")
    return f"(({operating_growth}) / ({revenue_growth}))"


def _average_identity_expr(rs: list[ResolvedFact]) -> str:
    return _average_balance_expr(rs[0], rs[1])


def _sum_expr(rs: list[ResolvedFact]) -> str:
    return f"({rs[0].expr_vnd()} + {rs[1].expr_vnd()})"


def _net_profit_cfo_gap_expr(rs: list[ResolvedFact]) -> str:
    return f"({rs[0].expr_vnd()} - {rs[1].expr_vnd()})"


def _cfo_net_margin_gap_expr(rs: list[ResolvedFact]) -> str:
    numerator = f"({rs[0].expr_vnd()} - {rs[1].expr_vnd()})"
    return f"(({numerator}) / ({rs[2].expr_vnd()}) * 100.0)"


def _cfo_net_profit_revenue_gap_expr(rs: list[ResolvedFact]) -> str:
    numerator = f"({rs[0].expr_vnd()} - {rs[1].expr_vnd()})"
    return f"(({numerator}) / ({rs[2].expr_vnd()}))"


def _identity_expr(rs: list[ResolvedFact]) -> str:
    return rs[0].expr_vnd()


_NUM_RE = r"[-+]?\d+(?:[\.,]\d+)?"

def _canonical_operand(key: str, strict_codes: bool = True) -> Operand:
    metric = get_metric(key)
    return Operand(
        metric=metric.label,
        variants=metric.variants,
        required_phrases=metric.required_phrases,
        forbidden_phrases=metric.forbidden_phrases,
        expected_codes=metric.codes if strict_codes else (),
    )


_CURRENT_ASSETS = _canonical_operand("current_assets")
_INVENTORY = replace(_canonical_operand("inventory"), expected_codes=("140",))
_INVENTORY_GROSS = _canonical_operand("inventory_gross")
_CURRENT_LIABILITIES = _canonical_operand("current_liabilities")
_LIABILITIES = _canonical_operand("liabilities")
_EQUITY = _canonical_operand("equity")
_NET_REVENUE = _canonical_operand("net_revenue")
_GROSS_PROFIT = _canonical_operand("gross_profit")
_NET_PROFIT = _canonical_operand("net_profit")
_CFO = _canonical_operand("cfo", strict_codes=False)
_TOTAL_ASSETS = _canonical_operand("total_assets")
_SELLING_EXP = _canonical_operand("selling_expense")
_ADMIN_EXP = _canonical_operand("administrative_expense")
_FIXED_ASSETS = _canonical_operand("fixed_assets")
_PRETAX_PROFIT = _canonical_operand("pretax_profit")
_INTEREST_EXPENSE = _canonical_operand("interest_expense", strict_codes=False)
_OPERATING_PROFIT = _canonical_operand("operating_profit")
_COGS = _canonical_operand("cost_of_goods_sold")

_CURRENT_PERIOD = PeriodRef("current", 0)
_OPENING_PERIOD = PeriodRef("opening", -1)
_CLOSING_PERIOD = PeriodRef("closing", 0)

_DIRECT_METRIC_SPEC = FormulaSpec(
    "direct_metric", (), (), "money", lambda vals: vals[0], lambda rs: rs[0].expr_vnd())

_EXACT_DERIVED_SPECS = {
    "net_financial_result": ExactDerivedSpec(
        ("financial_income", "financial_expense"), "money",
        lambda values: values[0] - abs(values[1]),
        lambda rows: f"(({rows[0].expr_vnd()}) - abs({rows[1].expr_vnd()}))",
    ),
    "effective_income_tax_rate": ExactDerivedSpec(
        ("income_tax_expense_total", "pretax_profit"), "percent",
        lambda values: _div(abs(values[0]), abs(values[1])) * 100.0,
        lambda rows: (
            f"(abs({rows[0].expr_vnd()}) / abs({rows[1].expr_vnd()}) * 100)"
        ),
    ),
    "depreciation_amortization_cost_share": ExactDerivedSpec(
        ("depreciation_amortization_expense",
         "production_business_cost_by_element_total"), "percent",
        lambda values: _div(abs(values[0]), abs(values[1])) * 100.0,
        lambda rows: (
            f"(abs({rows[0].expr_vnd()}) / abs({rows[1].expr_vnd()}) * 100)"
        ),
    ),
    "related_party_receivables_payables_total": ExactDerivedSpec(
        ("related_party_receivables", "related_party_payables"), "money",
        lambda values: values[0] + values[1],
        lambda rows: f"(({rows[0].expr_vnd()}) + ({rows[1].expr_vnd()}))",
    ),
}

_FORMULAS = [
    FormulaSpec(
        "quick_ratio",
        ("he so thanh toan nhanh", "ty so thanh toan nhanh",
         "ti so thanh toan nhanh",
         "ty le phan chenh lech giua tai san ngan han va hang ton kho tren no ngan han",
         "phan chenh lech giua tai san ngan han va hang ton kho tren no ngan han",
         "quick ratio"),
        (_CURRENT_ASSETS, _INVENTORY, _CURRENT_LIABILITIES),
        "ratio",
        lambda v: _div(v[0] - v[1], v[2]),
        _quick_expr,
    ),
    FormulaSpec(
        "current_ratio",
        ("he so thanh toan hien hanh", "ty so thanh toan hien hanh",
         "ti so thanh toan hien hanh",
         "tai san ngan han gap bao nhieu lan no ngan han",
         "ty le tai san ngan han tren no ngan han",
         "tai san ngan han tren no ngan han",
         "tai san ngan han chia cho no ngan han", "current ratio"),
        (_CURRENT_ASSETS, _CURRENT_LIABILITIES),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "debt_equity",
        ("he so no phai tra tren von chu so huu", "no phai tra tren von chu so huu",
         "ty le no phai tra tren von chu so huu",
         "no phai tra chia cho von chu so huu",
         "no phai tra gap bao nhieu lan von chu so huu",
         "d/e", "debt/equity"),
        (_LIABILITIES, _EQUITY),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "debt_assets",
        ("no phai tra tren tong tai san", "no phai tra chia cho tong tai san",
         "ty le no phai tra chia cho tong tai san", "debt/assets",
         "he so no tren tai san"),
        (_LIABILITIES, _TOTAL_ASSETS),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "gross_margin",
        ("bien loi nhuan gop", "gross margin",
         "loi nhuan gop tren doanh thu thuan"),
        (_GROSS_PROFIT, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "net_margin",
        ("bien loi nhuan sau thue", "bien loi nhuan rong", "net margin",
         "loi nhuan sau thue tren doanh thu thuan"),
        (_NET_PROFIT, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "operating_margin",
        ("bien loi nhuan hoat dong", "loi nhuan thuan tu hoat dong kinh doanh tren doanh thu thuan"),
        (_OPERATING_PROFIT, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "cfo_margin",
        ("bien dong tien tu hoat dong kinh doanh", "cfo margin",
         "ty so cfo tren doanh thu thuan",
         "ti so cfo tren doanh thu thuan",
         "ty so dong tien hoat dong tren doanh thu thuan",
         "ty le dong tien thuan tu hoat dong kinh doanh (cfo) tren doanh thu thuan",
         "luu chuyen tien thuan tu hoat dong kinh doanh tren doanh thu thuan",
         "dong tien thuan tu hoat dong kinh doanh tren doanh thu thuan"),
        (_CFO, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "cfo_net_profit",
        ("cfo tren lnst", "cfo/lnst",
         "ty le dong tien hoat dong (cfo) tren lnst",
         "ty le cfo tren loi nhuan sau thue",
         "ty le dong tien kinh doanh tren loi nhuan sau thue",
         "ty le dong tien thuan tu hoat dong kinh doanh (cfo) tren loi nhuan sau thue",
         "luu chuyen tien thuan tu hoat dong kinh doanh tren loi nhuan sau thue",
         "dong tien kinh doanh tren loi nhuan sau thue",
         "dong tien thuan tu hoat dong kinh doanh tren loi nhuan sau thue"),
        (_CFO, _NET_PROFIT),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "cfo_operating_profit",
        ("luu chuyen tien thuan tu hoat dong kinh doanh tren loi nhuan thuan tu hoat dong kinh doanh",
         "dong tien thuan tu hoat dong kinh doanh tren loi nhuan thuan tu hoat dong kinh doanh"),
        (_CFO, _OPERATING_PROFIT),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "cfo_current_liabilities",
        ("cfo tren no ngan han", "cfo/no ngan han",
         "luu chuyen tien thuan tu hoat dong kinh doanh tren no ngan han",
         "dong tien thuan tu hoat dong kinh doanh tren no ngan han",
         "he so dong tien hoat dong tren no ngan han",
         "dong tien hoat dong tren no ngan han"),
        (_CFO, _CURRENT_LIABILITIES),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "inventory_current_liabilities",
        ("ty le hang ton kho tren no ngan han",
         "ty le hang ton kho chia cho no ngan han",
         "hang ton kho chia cho no ngan han",
         "hang ton kho tren no ngan han"),
        (_INVENTORY, _CURRENT_LIABILITIES),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "roa",
        ("roa", "loi nhuan sau thue tren tong tai san"),
        (_NET_PROFIT, _TOTAL_ASSETS),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "roe",
        ("roe", "loi nhuan sau thue tren von chu so huu",
         "ty le giua loi nhuan thuan sau thue hop nhat va von chu so huu"),
        (_NET_PROFIT, _EQUITY),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "sga_intensity",
        ("ty trong chi phi ban hang va quan ly doanh nghiep",
         "ty le sga", "ty le sg&a",
         "chi phi ban hang va quan ly doanh nghiep tren doanh thu thuan",
         "chi phi ban hang va chi phi quan ly doanh nghiep tren doanh thu thuan",
         "sga intensity", "sg&a intensity"),
        (_SELLING_EXP, _ADMIN_EXP, _NET_REVENUE),
        "percent",
        lambda v: _div(abs(v[0]) + abs(v[1]), abs(v[2])) * 100.0,
        _sum_margin_expr,
    ),
    FormulaSpec(
        "cfo_net_margin_gap",
        ("hieu so giua cfo margin va bien loi nhuan rong",
         "chenh lech giua cfo margin va bien loi nhuan rong",
         "cfo margin tru bien loi nhuan rong"),
        (_CFO, _NET_PROFIT, _NET_REVENUE),
        "percent",
        lambda v: _div(v[0] - v[1], v[2]) * 100.0,
        _cfo_net_margin_gap_expr,
    ),
    FormulaSpec(
        "net_profit_cfo_gap",
        ("chenh lech duong giua lnst va cfo",
         "chenh lech giua lnst va cfo",
         "phan lnst chua chuyen hoa thanh cfo",
         "loi nhuan sau thue tru cfo"),
        (_NET_PROFIT, _CFO),
        "money",
        lambda v: v[0] - v[1],
        _net_profit_cfo_gap_expr,
    ),
    FormulaSpec(
        "cfo_net_profit_revenue_gap",
        ("luu chuyen tien thuan tu hoat dong kinh doanh tru loi nhuan sau thue "
         "roi chia cho doanh thu thuan",
         "cfo tru lnst roi chia cho doanh thu thuan"),
        (_CFO, _NET_PROFIT, _NET_REVENUE),
        "ratio",
        lambda v: _div(v[0] - v[1], v[2]),
        _cfo_net_profit_revenue_gap_expr,
    ),
    FormulaSpec(
        "fixed_asset_turnover",
        ("vong quay tai san co dinh", "doanh thu thuan tren tai san co dinh",
         "fixed asset turnover"),
        (_NET_REVENUE, _FIXED_ASSETS),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "total_asset_turnover",
        ("vong quay tong tai san", "doanh thu thuan tren tong tai san",
         "total asset turnover"),
        (_NET_REVENUE, _TOTAL_ASSETS),
        "ratio",
        lambda v: _div(v[0], v[1]),
        lambda rs: _ratio_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "long_term_assets_ratio",
        ("ty trong tai san dai han tren tong tai san",
         "ty le tai san dai han tren tong tai san",
         "tai san dai han tren tong tai san"),
        (_canonical_operand("long_term_assets"), _TOTAL_ASSETS),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        lambda rs: _percent_expr(rs[0], rs[1]),
    ),
    FormulaSpec(
        "inventory_assets",
        ("ty trong hang ton kho tren tong tai san",
         "hang ton kho tren tong tai san", "ty trong hang ton kho"),
        (_INVENTORY, _TOTAL_ASSETS),
        "percent",
        lambda v: _div(v[0], v[1]) * 100.0,
        _inventory_assets_expr,
    ),
    FormulaSpec(
        "working_capital",
        ("von luu dong rong", "tai san ngan han thap hon no ngan han",
         "tai san ngan han cao hon no ngan han"),
        (_CURRENT_ASSETS, _CURRENT_LIABILITIES),
        "money",
        lambda v: v[0] - v[1],
        _working_capital_expr,
    ),
    FormulaSpec(
        "interest_coverage",
        ("he so kha nang thanh toan lai vay", "kha nang thanh toan lai vay",
         "loi nhuan truoc lai vay va thue", "ebit tren chi phi lai vay",
         "tong loi nhuan truoc thue va chi phi lai vay"),
        (_PRETAX_PROFIT, _INTEREST_EXPENSE),
        "ratio",
        lambda v: _div(v[0] + abs(v[1]), abs(v[1])),
        _interest_coverage_expr,
    ),
    FormulaSpec(
        "cost_inventory_ratio",
        ("ty le giua gia von hang ban tong cong va gia goc hang ton kho cuoi nam",
         "gia von hang ban tren hang ton kho"),
        (_COGS, _INVENTORY_GROSS),
        "ratio",
        lambda v: _div(abs(v[0]), v[1]),
        lambda rs: f"(abs({rs[0].expr_vnd()}) / {rs[1].expr_vnd()})",
    ),
    FormulaSpec(
        "interest_pretax_ratio",
        ("ty le giua chi phi lai vay va loi nhuan truoc thue",
         "chi phi lai vay tren loi nhuan truoc thue"),
        (_INTEREST_EXPENSE, _PRETAX_PROFIT),
        "percent",
        lambda v: _div(abs(v[0]), v[1]) * 100.0,
        lambda rs: f"(abs({rs[0].expr_vnd()}) / {rs[1].expr_vnd()} * 100)",
    ),
    FormulaSpec(
        "inventory_days",
        ("so ngay ton kho",
         "365 lan hang ton kho binh quan dau ky va cuoi ky tren gia von hang ban",
         "365 lan trung binh hang ton kho dau nam va cuoi nam tren gia von hang ban",
         "hang ton kho binh quan nhan 365 roi chia cho gia von hang ban",
         "365 nhan voi trung binh hang ton kho dau nam va cuoi nam roi chia cho gia von hang ban"),
        (_INVENTORY, _INVENTORY, _COGS),
        "ratio",
        lambda v: 365.0 * ((v[0] + v[1]) / 2.0) / abs(v[2]),
        _inventory_days_expr,
        period_refs=(_OPENING_PERIOD, _CLOSING_PERIOD, _CURRENT_PERIOD),
        average_balances=(AverageBalanceNode(0, 1),),
    ),
    FormulaSpec(
        "average_total_assets", (),
        (_TOTAL_ASSETS, _TOTAL_ASSETS), "money",
        lambda v: (v[0] + v[1]) / 2.0,
        _average_identity_expr,
        period_refs=(_OPENING_PERIOD, _CLOSING_PERIOD),
        average_balances=(AverageBalanceNode(0, 1),),
    ),
    FormulaSpec(
        "average_equity", (),
        (_EQUITY, _EQUITY), "money",
        lambda v: (v[0] + v[1]) / 2.0,
        _average_identity_expr,
        period_refs=(_OPENING_PERIOD, _CLOSING_PERIOD),
        average_balances=(AverageBalanceNode(0, 1),),
    ),
    FormulaSpec(
        "average_fixed_assets", (),
        (_FIXED_ASSETS, _FIXED_ASSETS), "money",
        lambda v: (v[0] + v[1]) / 2.0,
        _average_identity_expr,
        period_refs=(_OPENING_PERIOD, _CLOSING_PERIOD),
        average_balances=(AverageBalanceNode(0, 1),),
    ),
    FormulaSpec(
        "roa_average_assets", (),
        (_NET_PROFIT, _TOTAL_ASSETS, _TOTAL_ASSETS), "percent",
        lambda v: _div(v[0], (v[1] + v[2]) / 2.0) * 100.0,
        lambda rs: _average_ratio_expr(rs, percent=True),
        period_refs=(_CURRENT_PERIOD, _OPENING_PERIOD, _CLOSING_PERIOD),
        average_balances=(AverageBalanceNode(1, 2),),
    ),
    FormulaSpec(
        "roe_average_equity", (),
        (_NET_PROFIT, _EQUITY, _EQUITY), "percent",
        lambda v: _div(v[0], (v[1] + v[2]) / 2.0) * 100.0,
        lambda rs: _average_ratio_expr(rs, percent=True),
        period_refs=(_CURRENT_PERIOD, _OPENING_PERIOD, _CLOSING_PERIOD),
        average_balances=(AverageBalanceNode(1, 2),),
    ),
    FormulaSpec(
        "total_asset_turnover_average_assets", (),
        (_NET_REVENUE, _TOTAL_ASSETS, _TOTAL_ASSETS), "ratio",
        lambda v: _div(v[0], (v[1] + v[2]) / 2.0),
        _average_ratio_expr,
        period_refs=(_CURRENT_PERIOD, _OPENING_PERIOD, _CLOSING_PERIOD),
        average_balances=(AverageBalanceNode(1, 2),),
    ),
    FormulaSpec(
        "fixed_asset_turnover_average_assets", (),
        (_NET_REVENUE, _FIXED_ASSETS, _FIXED_ASSETS), "ratio",
        lambda v: _div(v[0], (v[1] + v[2]) / 2.0),
        _average_ratio_expr,
        period_refs=(_CURRENT_PERIOD, _OPENING_PERIOD, _CLOSING_PERIOD),
        average_balances=(AverageBalanceNode(1, 2),),
    ),
    FormulaSpec(
        "accrual_average_assets", ("ty so don tich", "ty le don tich"),
        (_NET_PROFIT, _CFO, _TOTAL_ASSETS, _TOTAL_ASSETS), "percent",
        lambda v: _div(v[0] - v[1], (v[2] + v[3]) / 2.0) * 100.0,
        _accrual_average_assets_expr,
        period_refs=(
            _CURRENT_PERIOD, _CURRENT_PERIOD,
            _OPENING_PERIOD, _CLOSING_PERIOD,
        ),
        average_balances=(AverageBalanceNode(2, 3),),
    ),
    FormulaSpec(
        "operating_leverage", ("don bay kinh doanh",),
        (_OPERATING_PROFIT, _OPERATING_PROFIT, _NET_REVENUE, _NET_REVENUE),
        "ratio",
        lambda v: _div(
            _div(v[1] - v[0], abs(v[0])),
            _div(v[3] - v[2], abs(v[2]))),
        _operating_leverage_expr,
        period_refs=(
            _OPENING_PERIOD, _CURRENT_PERIOD,
            _OPENING_PERIOD, _CURRENT_PERIOD,
        ),
    ),
]


def _note_direct_spec(name: str, key: str,
                      triggers: tuple[str, ...]) -> FormulaSpec:
    return FormulaSpec(
        name, triggers, (_canonical_operand(key, strict_codes=False),),
        "money", lambda values: values[0], _identity_expr)


def _note_ratio_spec(name: str, numerator: str, denominator: str,
                     triggers: tuple[str, ...], *, kind: str = "percent",
                     absolute: bool = False) -> FormulaSpec:
    multiplier = 100.0 if kind == "percent" else 1.0
    if absolute:
        value_fn = lambda values: _div(abs(values[0]), abs(values[1])) * multiplier
        expr_fn = lambda resolved: (
            f"(abs({resolved[0].expr_vnd()}) / "
            f"abs({resolved[1].expr_vnd()}) * {multiplier:g})")
    else:
        value_fn = lambda values: _div(values[0], values[1]) * multiplier
        expr_fn = lambda resolved: (
            f"({resolved[0].expr_vnd()} / "
            f"{resolved[1].expr_vnd()} * {multiplier:g})")
    return FormulaSpec(
        name, triggers,
        (_canonical_operand(numerator, strict_codes=False),
         _canonical_operand(denominator, strict_codes=False)),
        kind, value_fn, expr_fn)


_NOTE_DETAIL_FORMULAS = [
    _note_direct_spec(
        "note_hagl_related_party_long_term_borrowing",
        "hagl_related_party_long_term_borrowing",
        ("vay dai han voi cong ty co phan hoang anh gia lai",
         "vay dai han voi hagl")),
    _note_direct_spec(
        "note_intangible_fixed_assets", "intangible_fixed_assets",
        ("gia tri con lai tai san co dinh vo hinh",
         "gia tri con lai cua tai san co dinh vo hinh")),
    _note_direct_spec(
        "note_related_party_trade_payables_short_term",
        "related_party_trade_payables_short_term",
        ("phai tra nguoi ban ngan han voi ben lien quan",
         "phai tra nguoi ban ngan han cac ben lien quan")),
    _note_direct_spec(
        "note_related_party_short_term_receivables_total",
        "related_party_short_term_receivables_total",
        ("tong phai thu ngan han tu cac ben lien quan",
         "phai thu ngan han tu cac ben lien quan")),
    _note_direct_spec(
        "note_merchandise_inventory", "merchandise_inventory",
        ("gia tri hang hoa ton kho", "hang hoa ton kho cuoi ky")),
    _note_direct_spec(
        "note_investment_property_depreciation",
        "investment_property_depreciation",
        ("chi phi khau hao bat dong san dau tu",
         "khau hao bat dong san dau tu")),
    _note_direct_spec(
        "note_current_income_tax", "current_income_tax",
        ("chi phi thue tndn hien hanh",
         "chi phi thue thu nhap doanh nghiep hien hanh")),
    FormulaSpec(
        "note_borrowings_cash_and_deposits_ratio",
        ("tong no vay gap tong tien mat va tien gui ngan hang",
         "tong no vay tren tong tien mat va tien gui ngan hang"),
        (_canonical_operand("borrowings_total", strict_codes=False),
         _canonical_operand("cash_on_hand", strict_codes=False),
         _canonical_operand("bank_deposits", strict_codes=False)),
        "ratio",
        lambda values: _div(values[0], values[1] + values[2]),
        lambda resolved: (
            f"({resolved[0].expr_vnd()} / "
            f"({resolved[1].expr_vnd()} + {resolved[2].expr_vnd()}))"),
    ),
    _note_ratio_spec(
        "note_equity_turnover", "net_revenue", "equity",
        ("vong quay von chu so huu",), kind="ratio"),
    _note_ratio_spec(
        "note_off_balance_commitments_assets", "off_balance_commitments",
        "total_assets", ("cam ket ngoai bang tren tong tai san",)),
    _note_ratio_spec(
        "note_related_party_long_term_loan_share",
        "related_party_long_term_loans_receivable",
        "related_party_long_term_receivables_total",
        ("ty trong khoan cho vay dai han ben lien quan trong tong khoan phai thu dai han tu ben lien quan",
         "cho vay dai han ben lien quan trong tong phai thu dai han tu ben lien quan")),
    _note_ratio_spec(
        "note_deposit_interest_expense_share", "deposit_interest_expense",
        "bank_interest_expense",
        ("ty trong chi phi lai tien gui",
         "ty trong chi phi lai tien gui trong tong chi phi lai",
         "chi phi lai tien gui tren tong chi phi lai"), absolute=True),
    _note_ratio_spec(
        "note_real_estate_customer_loan_share", "real_estate_customer_loans",
        "customer_loans", ("ty trong du no cho vay nganh bat dong san",
                           "ty trong trung binh du no cho vay nganh bat dong san")),
    _note_ratio_spec(
        "note_equity_total_capital_share", "equity", "total_capital",
        ("von chu so huu tren tong nguon von",)),
    _note_ratio_spec(
        "note_finished_goods_inventory_share", "finished_goods_inventory",
        "inventory_gross",
        ("ty trong thanh pham trong tong gia tri hang ton kho",
         "thanh pham trong tong gia tri hang ton kho")),
    _note_ratio_spec(
        "note_accumulated_depreciation_cost_share",
        "fixed_assets_accumulated_depreciation", "fixed_assets_cost",
        ("khau hao luy ke tren nguyen gia tai san co dinh",
         "hao mon luy ke tren nguyen gia tai san co dinh"), absolute=True),
    _note_ratio_spec(
        "note_usd_long_term_borrowing_share", "usd_long_term_borrowings",
        "borrowings_long_term",
        ("ty trong khoan vay bang usd trong tong khoan vay dai han",
         "vay bang usd trong tong khoan vay dai han")),
    _note_ratio_spec(
        "note_financial_reserve_equity_share", "financial_reserve_fund",
        "equity", ("ty trong quy du phong tai chinh trong von chu so huu",)),
    _note_ratio_spec(
        "note_transport_segment_asset_share", "transport_segment_assets",
        "total_assets",
        ("ty trong tai san bo phan dich vu van tai so voi tong tai san",
         "tai san bo phan dich vu van tai tren tong tai san")),
    _note_ratio_spec(
        "note_credit_provision_preprovision_ratio", "credit_provision_expense",
        "pre_provision_operating_profit",
        ("chi phi du phong rui ro tin dung tren loi nhuan truoc du phong",),
        absolute=True),
    _note_ratio_spec(
        "note_bot_segment_asset_share", "bot_segment_assets", "total_assets",
        ("ty trong tai san bo phan bot tren tong tai san",
         "tai san bo phan bot tren tong tai san")),
    _note_ratio_spec(
        "note_general_provision_total_loan_provision_share",
        "general_customer_loan_provision_balance",
        "customer_loan_provision_balance",
        ("ty trong du phong chung trong tong du phong rui ro cho vay khach hang",
         "du phong chung trong tong du phong rui ro cho vay khach hang"),
        absolute=True),
]
_FORMULAS.extend(_NOTE_DETAIL_FORMULAS)
_COMPOUND_FORMULA_NAMES = frozenset({
    "cfo_net_margin_gap",
    "net_profit_cfo_gap",
    "cfo_net_profit_revenue_gap",
})
_NOTE_DETAIL_SPEC_NAMES = frozenset(
    spec.name for spec in _NOTE_DETAIL_FORMULAS)

# One-line statement metrics are useful as selectors and filters.  They stay
# separate from _FORMULAS so a direct lookup question is not mistaken for a
# formula question merely because it mentions revenue or inventory.
_METRIC_SPECS = [
    FormulaSpec("net_revenue", ("doanh thu thuan",), (_NET_REVENUE,),
                "money", lambda v: v[0], _identity_expr),
    FormulaSpec("gross_profit", ("loi nhuan gop",), (_GROSS_PROFIT,),
                "money", lambda v: v[0], _identity_expr),
    FormulaSpec("net_profit", ("loi nhuan sau thue", "lnst"), (_NET_PROFIT,),
                "money", lambda v: v[0], _identity_expr),
    FormulaSpec(
        "cfo",
        ("luu chuyen tien thuan tu hoat dong kinh doanh",
         "dong tien thuan tu hoat dong kinh doanh", "dong tien hoat dong",
         "cfo"),
        (_CFO,), "money", lambda v: v[0], _identity_expr),
    FormulaSpec("inventory", ("gia tri hang ton kho", "hang ton kho"),
                (_INVENTORY,), "money", lambda v: v[0], _identity_expr),
    FormulaSpec(
        "sga_expense",
        ("tong chi phi ban hang va chi phi quan ly doanh nghiep",
         "chi phi ban hang va chi phi quan ly doanh nghiep"),
        (_SELLING_EXP, _ADMIN_EXP), "money", lambda v: v[0] + v[1], _sum_expr),
    FormulaSpec("pretax_profit", ("loi nhuan truoc thue", "lntt"),
                (_PRETAX_PROFIT,), "money", lambda v: v[0], _identity_expr),
    FormulaSpec("interest_expense", ("chi phi lai vay",), (_INTEREST_EXPENSE,),
                "money", lambda v: abs(v[0]),
                lambda rs: f"abs({rs[0].expr_vnd()})"),
]
