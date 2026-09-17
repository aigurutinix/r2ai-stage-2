
from __future__ import annotations

import hashlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.hard.recipe import operations  # noqa: F401 — side-effect: register operations
from vifinqa.generation.hard.recipe.audit.base import DependencyAuditor
from vifinqa.generation.hard.recipe.audit.cache import AuditCache
from vifinqa.generation.hard.recipe.audit.service import DependencyAuditRejected
from vifinqa.generation.hard.recipe.base import CandidateRejected, MetricRoleInput, ReasoningGraph, ValueKind
from vifinqa.generation.hard.recipe.bindings import resolve_metric_terms, resolve_operating_accruals_terms
from vifinqa.generation.hard.recipe.compiler import terminal_value_kind
from vifinqa.generation.hard.recipe.evaluator import EvaluationError
from vifinqa.generation.hard.recipe.finalize import company_names, finalize_candidate, unit_label
from vifinqa.generation.hard.recipe.grounded.domains import (
    MIN_PEER_ENTITIES,
    adjacent_period_domains,
    same_period_domains,
)
from vifinqa.generation.hard.recipe.grounded.recipes import (
    EQ_01_ID,
    EQ_01_TERMINAL_KEY,
    GRO_03_ID,
    GRO_03_TERMINAL_KEY,
    GROUNDED_INTERPRETATION_LIMITS,
    GROUNDED_RECIPE_MEANINGS,
    LEV_05_ID,
    LIQ_02_ID,
    PRO_04_ID,
    WCA_10_ID,
    build_eq01_graph,
    build_gro03_graph,
    build_lev05_graph,
    build_liq02_graph,
    build_pro04_graph,
    build_wca10_graph,
)
from vifinqa.generation.hard.recipe.planner import PublicSpec, RecipeCandidate, table_ref_to_path_map
from vifinqa.generation.panel.base import Cube
from vifinqa.generation.panel.catalog import display_name, get_ratio
from vifinqa.generation.panel.measurements.catalog import GRO_03 as GRO_03_MEASUREMENT
from vifinqa.generation.panel.measurements.catalog import LEV_05 as LEV_05_MEASUREMENT
from vifinqa.generation.panel.measurements.catalog import LIQ_02 as LIQ_02_MEASUREMENT
from vifinqa.generation.panel.measurements.catalog import PRO_04 as PRO_04_MEASUREMENT

logger = logging.getLogger(__name__)

REPORT_SCOPE = "consolidated"

_RATIO_KEYS = {
    LIQ_02_ID: ("current_ratio", "inventory_to_current_liabilities"),
    PRO_04_ID: ("gross_margin", "gross_to_net_margin_difference"),
    LEV_05_ID: ("liabilities_to_equity", "interest_coverage"),
    WCA_10_ID: ("current_ratio", "operating_cash_flow_ratio"),
}


@dataclass(frozen=True, slots=True)
class GroundedDomainAttempt:

    intent_id: str
    industry: str
    entities: tuple[str, ...]
    periods: tuple[str, ...]
    graph: ReasoningGraph | None
    terminal_metric_key: str
    reject_reason: str | None


def _candidate_signature(*parts: object) -> str:
    raw = "|".join(repr(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _entity_role(bindings: dict, role_id: str, metric_key: str) -> MetricRoleInput:
    return MetricRoleInput(role_id, metric_key, ValueKind.NUMERIC_SERIES_ENTITY, bindings)


def _entity_period_role(bindings: dict, role_id: str, metric_key: str) -> MetricRoleInput:
    return MetricRoleInput(role_id, metric_key, ValueKind.NUMERIC_SERIES_ENTITY_PERIOD, bindings)


# --- EQ_01 ---------------------------------------------------------------------------------------


def _eq01_eligible(cube: Cube, ticker: str, period_prior: str, period: str) -> bool:
    npat_cell = cube.cell(ticker, period, "kqkd:60")
    cfo_cell = cube.cell(ticker, period, "lctt:20")
    assets_t = cube.cell(ticker, period, "cdkt:270")
    assets_prior = cube.cell(ticker, period_prior, "cdkt:270")
    if None in (npat_cell, cfo_cell, assets_t, assets_prior):
        return False
    assert assets_t is not None and assets_prior is not None
    return (assets_t.value + assets_prior.value) / 2 > 0


def eq01_attempt(cube: Cube, industry: str, entities: tuple[str, ...], period_prior: str, period: str) -> GroundedDomainAttempt:
    eligible = tuple(sorted(t for t in entities if _eq01_eligible(cube, t, period_prior, period)))
    if len(eligible) < MIN_PEER_ENTITIES:
        return GroundedDomainAttempt(EQ_01_ID, industry, eligible, (period,), None, EQ_01_TERMINAL_KEY, "insufficient_coverage")

    npat_bindings = {t: resolve_metric_terms(cube, t, period, "kqkd:60") for t in eligible}
    accrual_bindings = {t: resolve_operating_accruals_terms(cube, t, period_prior, period) for t in eligible}
    npat_role = _entity_role(npat_bindings, "npat", "kqkd:60")
    accrual_role = _entity_role(accrual_bindings, "accrual", EQ_01_TERMINAL_KEY)
    graph = build_eq01_graph(
        entities=eligible, reference_period=period, report_scope=REPORT_SCOPE,
        npat_role=npat_role, accrual_role=accrual_role,
    )
    return GroundedDomainAttempt(EQ_01_ID, industry, eligible, (period,), graph, EQ_01_TERMINAL_KEY, None)


# --- LIQ_02 --------------------------------------------------------------------------------------


def liq02_attempt(cube: Cube, industry: str, entities: tuple[str, ...], period: str) -> GroundedDomainAttempt:
    eligible = tuple(sorted(t for t in entities if LIQ_02_MEASUREMENT.evaluate(cube, t, (period,)).value is not None))
    if len(eligible) < MIN_PEER_ENTITIES:
        return GroundedDomainAttempt(LIQ_02_ID, industry, eligible, (period,), None, "inventory_to_current_liabilities", "insufficient_coverage")

    rank_key, target_key = _RATIO_KEYS[LIQ_02_ID]
    rank_bindings = {t: resolve_metric_terms(cube, t, period, rank_key) for t in eligible}
    target_bindings = {t: resolve_metric_terms(cube, t, period, target_key) for t in eligible}
    rank_role = _entity_role(rank_bindings, "current_ratio", rank_key)
    target_role = _entity_role(target_bindings, "target", target_key)
    graph = build_liq02_graph(
        entities=eligible, reference_period=period, report_scope=REPORT_SCOPE,
        current_ratio_role=rank_role, target_role=target_role,
    )
    return GroundedDomainAttempt(LIQ_02_ID, industry, eligible, (period,), graph, target_key, None)


# --- GRO_03 --------------------------------------------------------------------------------------


def gro03_attempt(cube: Cube, industry: str, entities: tuple[str, ...], period_prior: str, period: str) -> GroundedDomainAttempt:
    eligible = tuple(
        sorted(t for t in entities if GRO_03_MEASUREMENT.evaluate(cube, t, (period_prior, period)).value is not None)
    )
    periods = (period_prior, period)
    if len(eligible) < MIN_PEER_ENTITIES:
        return GroundedDomainAttempt(GRO_03_ID, industry, eligible, periods, None, GRO_03_TERMINAL_KEY, "insufficient_coverage")

    revenue_bindings = {(t, p): resolve_metric_terms(cube, t, p, "kqkd:10") for t in eligible for p in periods}
    gpm_bindings = {(t, p): resolve_metric_terms(cube, t, p, "gross_margin") for t in eligible for p in periods}
    revenue_role = _entity_period_role(revenue_bindings, "revenue", "kqkd:10")
    gpm_role = _entity_period_role(gpm_bindings, "gross_margin", "gross_margin")
    graph = build_gro03_graph(
        entities=eligible, periods=periods, report_scope=REPORT_SCOPE,
        revenue_role=revenue_role, gross_margin_role=gpm_role,
    )
    return GroundedDomainAttempt(GRO_03_ID, industry, eligible, periods, graph, GRO_03_TERMINAL_KEY, None)


# --- PRO_04 --------------------------------------------------------------------------------------


def pro04_attempt(cube: Cube, industry: str, entities: tuple[str, ...], period: str) -> GroundedDomainAttempt:
    eligible = tuple(sorted(t for t in entities if PRO_04_MEASUREMENT.evaluate(cube, t, (period,)).value is not None))
    if len(eligible) < MIN_PEER_ENTITIES:
        return GroundedDomainAttempt(PRO_04_ID, industry, eligible, (period,), None, "gross_to_net_margin_difference", "insufficient_coverage")

    rank_key, target_key = _RATIO_KEYS[PRO_04_ID]
    rank_bindings = {t: resolve_metric_terms(cube, t, period, rank_key) for t in eligible}
    target_bindings = {t: resolve_metric_terms(cube, t, period, target_key) for t in eligible}
    rank_role = _entity_role(rank_bindings, "gross_margin", rank_key)
    target_role = _entity_role(target_bindings, "target", target_key)
    graph = build_pro04_graph(
        entities=eligible, reference_period=period, report_scope=REPORT_SCOPE,
        gross_margin_role=rank_role, target_role=target_role,
    )
    return GroundedDomainAttempt(PRO_04_ID, industry, eligible, (period,), graph, target_key, None)


# --- LEV_05 --------------------------------------------------------------------------------------


def lev05_attempt(cube: Cube, industry: str, entities: tuple[str, ...], period: str) -> GroundedDomainAttempt:
    eligible = tuple(sorted(t for t in entities if LEV_05_MEASUREMENT.evaluate(cube, t, (period,)).value is not None))
    if len(eligible) < MIN_PEER_ENTITIES:
        return GroundedDomainAttempt(LEV_05_ID, industry, eligible, (period,), None, "interest_coverage", "insufficient_coverage")

    rank_key, target_key = _RATIO_KEYS[LEV_05_ID]
    rank_bindings = {t: resolve_metric_terms(cube, t, period, rank_key) for t in eligible}
    target_bindings = {t: resolve_metric_terms(cube, t, period, target_key) for t in eligible}
    rank_role = _entity_role(rank_bindings, "liabilities_to_equity", rank_key)
    target_role = _entity_role(target_bindings, "target", target_key)
    graph = build_lev05_graph(
        entities=eligible, reference_period=period, report_scope=REPORT_SCOPE,
        liabilities_to_equity_role=rank_role, target_role=target_role,
    )
    return GroundedDomainAttempt(LEV_05_ID, industry, eligible, (period,), graph, target_key, None)


# --- WCA_10 --------------------------------------------------------------------------------------


def _wca10_eligible(cube: Cube, ticker: str, period: str) -> bool:
    cl_cell = cube.cell(ticker, period, "cdkt:310")
    ca_cell = cube.cell(ticker, period, "cdkt:100")
    cfo_cell = cube.cell(ticker, period, "lctt:20")
    if None in (cl_cell, ca_cell, cfo_cell):
        return False
    assert cl_cell is not None
    return cl_cell.value > 0


def wca10_attempt(cube: Cube, industry: str, entities: tuple[str, ...], period: str) -> GroundedDomainAttempt:
    eligible = tuple(sorted(t for t in entities if _wca10_eligible(cube, t, period)))
    if len(eligible) < MIN_PEER_ENTITIES:
        return GroundedDomainAttempt(WCA_10_ID, industry, eligible, (period,), None, "operating_cash_flow_ratio", "insufficient_coverage")

    rank_key, target_key = _RATIO_KEYS[WCA_10_ID]
    rank_bindings = {t: resolve_metric_terms(cube, t, period, rank_key) for t in eligible}
    target_bindings = {t: resolve_metric_terms(cube, t, period, target_key) for t in eligible}
    rank_role = _entity_role(rank_bindings, "current_ratio", rank_key)
    target_role = _entity_role(target_bindings, "target", target_key)
    graph = build_wca10_graph(
        entities=eligible, reference_period=period, report_scope=REPORT_SCOPE,
        current_ratio_role=rank_role, target_role=target_role,
    )
    return GroundedDomainAttempt(WCA_10_ID, industry, eligible, (period,), graph, target_key, None)


# --- Shortcut cho feasibility.py: intent_id -> (domain_kind, attempt_fn) -------------------------

GROUNDED_ATTEMPT_BUILDERS: dict[str, tuple[str, object]] = {
    EQ_01_ID: ("adjacent_period", eq01_attempt),
    LIQ_02_ID: ("same_period", liq02_attempt),
    GRO_03_ID: ("adjacent_period", gro03_attempt),
    PRO_04_ID: ("same_period", pro04_attempt),
    LEV_05_ID: ("same_period", lev05_attempt),
    WCA_10_ID: ("same_period", wca10_attempt),
}


# --- Public spec + production candidate ----------------------------------------------------------


def _metric_role_pair(intent_id: str) -> tuple[str, str]:
    if intent_id == EQ_01_ID:
        return "kqkd:60", EQ_01_TERMINAL_KEY
    if intent_id == GRO_03_ID:
        return "kqkd:10", GRO_03_TERMINAL_KEY
    return _RATIO_KEYS[intent_id]


_DUAL_FILTER_RANK_INTENTS = frozenset({EQ_01_ID, WCA_10_ID})


def _build_public_spec(
    attempt: GroundedDomainAttempt,
    *,
    candidate_id: str,
    company_meta: dict[str, CompanyInfo],
    selector_direction: str,
    threshold_operator: str | None,
    threshold_convention_value: float | None,
) -> PublicSpec:
    rank_key, target_key = _metric_role_pair(attempt.intent_id)
    value_kind = terminal_value_kind(attempt.terminal_metric_key)
    rank_label = display_name(rank_key) or rank_key
    target_label = _terminal_label(attempt.intent_id, target_key)
    reference_period = attempt.periods[-1]
    if attempt.intent_id in _DUAL_FILTER_RANK_INTENTS:
        role_a: tuple[str, ...] = ("filter", "rank")
    elif threshold_operator:
        role_a = ("filter",)
    else:
        role_a = ("rank",)
    return PublicSpec(
        recipe_id=attempt.intent_id,
        candidate_id=candidate_id,
        entities=attempt.entities,
        company_names=company_names(attempt.entities, company_meta),
        periods=attempt.periods,
        reference_period=reference_period if len(attempt.periods) == 1 else None,
        metric_labels={"A": rank_label, "B": target_label},
        metric_keys={"A": rank_key, "B": target_key},
        metric_roles={"A": role_a, "B": ("target",)},
        threshold_operator=threshold_operator,
        threshold_source="convention" if threshold_operator else None,
        threshold_statistic=None,
        threshold_convention_value=threshold_convention_value,
        selector_direction=selector_direction,
        aggregate_operation=None,
        target_value_kind=value_kind,
        target_unit_label=unit_label(value_kind),
        intent_id=attempt.intent_id,
        analytical_frame=GROUNDED_RECIPE_MEANINGS[attempt.intent_id],
        terminal_measurement_name=target_label,
        interpretation_limits=GROUNDED_INTERPRETATION_LIMITS[attempt.intent_id],
    )


def _terminal_label(intent_id: str, target_key: str) -> str:
    if intent_id == EQ_01_ID:
        return "Chênh lệch giữa lợi nhuận sau thuế và dòng tiền từ hoạt động kinh doanh so với tài sản bình quân"
    if intent_id == GRO_03_ID:
        return "Thay đổi biên lợi nhuận gộp so với kỳ trước"
    ratio = get_ratio(target_key)
    return ratio.name if ratio is not None else (display_name(target_key) or target_key)


_SELECTOR_DIRECTION = {
    EQ_01_ID: "argmax",
    LIQ_02_ID: "argmax",
    GRO_03_ID: "argmax",
    PRO_04_ID: "argmax",
    LEV_05_ID: "argmax",
    WCA_10_ID: "argmin",
}
_THRESHOLD_INFO: dict[str, tuple[str, float] | None] = {
    EQ_01_ID: (">", 0.0),
    LIQ_02_ID: None,
    GRO_03_ID: None,
    PRO_04_ID: None,
    LEV_05_ID: None,
    WCA_10_ID: ("<", 1.0),
}


def _make_candidate(
    attempt: GroundedDomainAttempt,
    *,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
) -> RecipeCandidate:
    assert attempt.graph is not None
    validated_graph, trace, compiled, formatted_answer, audit_stats = finalize_candidate(
        provisional_graph=attempt.graph,
        terminal_metric_key=attempt.terminal_metric_key,
        docs_by_name=docs_by_name,
        company_meta=company_meta,
        table_ref_to_path=table_ref_to_path,
        auditor=auditor,
        audit_cache=audit_cache,
    )
    candidate_id = _candidate_signature(attempt.intent_id, attempt.entities, attempt.periods)
    threshold = _THRESHOLD_INFO[attempt.intent_id]
    public_spec = _build_public_spec(
        attempt,
        candidate_id=candidate_id,
        company_meta=company_meta,
        selector_direction=_SELECTOR_DIRECTION[attempt.intent_id],
        threshold_operator=threshold[0] if threshold else None,
        threshold_convention_value=threshold[1] if threshold else None,
    )
    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=attempt.intent_id,
        entities=attempt.entities,
        graph=validated_graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=formatted_answer,
        audit_stats=audit_stats,
        public_spec=public_spec,
    )


def _iter_candidates_for_intent(
    intent_id: str,
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    del seed  # Keep period handling explicit and deterministic.
    domain_kind, attempt_fn = GROUNDED_ATTEMPT_BUILDERS[intent_id]
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)

    if domain_kind == "same_period":
        domains = same_period_domains(cube, company_meta)
        attempts = (attempt_fn(cube, industry, entities, period) for industry, entities, period in domains)
    else:
        domains = adjacent_period_domains(cube, company_meta)
        attempts = (
            attempt_fn(cube, industry, entities, prior, current) for industry, entities, prior, current in domains
        )

    yielded = 0
    for attempt in attempts:
        if yielded >= max_candidates:
            return
        if attempt.graph is None:
            continue
        try:
            candidate = _make_candidate(
                attempt, docs_by_name=docs_by_name, company_meta=company_meta,
                table_ref_to_path=table_ref_to_path, auditor=auditor, audit_cache=audit_cache,
            )
        except DependencyAuditRejected as exc:
            logger.info("%s candidate reject (JIT audit) industry=%s periods=%s: %s", intent_id, attempt.industry, attempt.periods, exc.reasons)
            continue
        except EvaluationError as exc:
            logger.debug("%s candidate reject (evaluate: non-triviality/tie) industry=%s periods=%s: %s", intent_id, attempt.industry, attempt.periods, exc)
            continue
        except CandidateRejected as exc:
            logger.debug("%s candidate reject (hardness gate/BUG) industry=%s periods=%s: %s", intent_id, attempt.industry, attempt.periods, exc)
            continue
        yielded += 1
        yield candidate


def iter_eq01_candidates(**kwargs) -> Iterator[RecipeCandidate]:
    yield from _iter_candidates_for_intent(EQ_01_ID, **kwargs)


def iter_liq02_candidates(**kwargs) -> Iterator[RecipeCandidate]:
    yield from _iter_candidates_for_intent(LIQ_02_ID, **kwargs)


def iter_gro03_candidates(**kwargs) -> Iterator[RecipeCandidate]:
    yield from _iter_candidates_for_intent(GRO_03_ID, **kwargs)


def iter_pro04_candidates(**kwargs) -> Iterator[RecipeCandidate]:
    yield from _iter_candidates_for_intent(PRO_04_ID, **kwargs)


def iter_lev05_candidates(**kwargs) -> Iterator[RecipeCandidate]:
    yield from _iter_candidates_for_intent(LEV_05_ID, **kwargs)


def iter_wca10_candidates(**kwargs) -> Iterator[RecipeCandidate]:
    yield from _iter_candidates_for_intent(WCA_10_ID, **kwargs)


ALL_GROUNDED_RECIPE_ITERATORS: dict[str, object] = {
    EQ_01_ID: iter_eq01_candidates,
    LIQ_02_ID: iter_liq02_candidates,
    GRO_03_ID: iter_gro03_candidates,
    PRO_04_ID: iter_pro04_candidates,
    LEV_05_ID: iter_lev05_candidates,
    WCA_10_ID: iter_wca10_candidates,
}
