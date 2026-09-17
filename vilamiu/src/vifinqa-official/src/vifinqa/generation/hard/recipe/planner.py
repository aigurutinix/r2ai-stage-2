
from __future__ import annotations

import hashlib
import logging
import random
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.hard.recipe.audit.base import DependencyAuditor
from vifinqa.generation.hard.recipe.audit.cache import AuditCache
from vifinqa.generation.hard.recipe.audit.service import (
    AuditStats,
    DependencyAuditRejected,
)
from vifinqa.generation.hard.recipe.base import (
    CandidateRejected,
    MetricRoleInput,
    MetricTerms,
    ReasoningGraph,
    ValueKind,
)
from vifinqa.generation.hard.recipe.bindings import (
    longest_contiguous_run as _longest_contiguous_run,
)
from vifinqa.generation.hard.recipe.bindings import (
    resolve_metric_terms as _resolve_metric_terms,
)
from vifinqa.generation.hard.recipe.compiler import CompiledQuery, terminal_value_kind
from vifinqa.generation.hard.recipe.evaluator import EvaluationError, EvaluationTrace
from vifinqa.generation.hard.recipe.finalize import (
    company_names,
    finalize_candidate,
    unit_label,
)
from vifinqa.generation.hard.recipe.recipes import (
    AGGREGATE_OPERATIONS,
    DERIVED_THRESHOLD_STATISTICS,
    R1_RECIPE_ID,
    R2_RECIPE_ID,
    R3_RECIPE_ID,
    R4_RECIPE_ID,
    R5_RECIPE_ID,
    R6_RECIPE_ID,
    R7_RECIPE_ID,
    SELECT_DIRECTIONS,
    build_r1_graph,
    build_r2_graph,
    build_r3_graph,
    build_r4_graph,
    build_r5_graph,
    build_r6_graph,
    build_r7_graph,
)
from vifinqa.generation.panel.base import Cube
from vifinqa.generation.panel.catalog import display_name, get_ratio, peer_tickers

logger = logging.getLogger(__name__)

MIN_ENTITIES = 3
MAX_ENTITIES = 6
MIN_CANDIDATE_YEARS = 3
MAX_CANDIDATE_YEARS = 5
REVENUE_METRIC = "kqkd:10"
REPORT_SCOPE = "consolidated"

TEMPORAL_METRIC_CANDIDATES = ("lctt:20", "kqkd:60")
MIN_TEMPORAL_YEARS = 3
MAX_TEMPORAL_YEARS = 5
MIN_R3_ENTITIES = 4
MAX_R3_ENTITIES = 6
MIN_R4_ENTITIES = 3
MAX_R4_ENTITIES = 6
MIN_R5_ENTITIES = 4
MAX_R5_ENTITIES = 6
MIN_R6_PERIODS = 3
MAX_R6_PERIODS = 6
MIN_R7_PERIODS = 4
MAX_R7_PERIODS = 6
PERCENTILE_CANDIDATES = (25.0, 75.0)

_APPROVED_RATIO_KEYS = frozenset(
    {
        "gross_margin",
        "net_margin",
        "current_ratio",
        "quick_ratio",
        "liabilities_to_equity",
        "debt_to_assets",
    }
)
_RAW_TARGET_METRICS = (
    "kqkd:20",
    "kqkd:50",
    "kqkd:60",
    "cdkt:100",
    "cdkt:140",
    "cdkt:270",
    "cdkt:300",
    "cdkt:310",
    "cdkt:400",
    "lctt:20",
)
TARGET_METRIC_CANDIDATES = _RAW_TARGET_METRICS + tuple(sorted(_APPROVED_RATIO_KEYS))


def _candidate_signature(*parts: object) -> str:
    raw = "|".join(repr(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


CONVENTION_THRESHOLDS: dict[str, tuple[str, float]] = {
    "net_margin": (">", 0.10),
    "current_ratio": (">", 1.50),
}


_unit_label = unit_label


def _aggregate_operations_for(target_metric: str) -> tuple[str, ...]:
    if get_ratio(target_metric) is not None:
        return tuple(op for op in AGGREGATE_OPERATIONS if op != "sum")
    return AGGREGATE_OPERATIONS


@dataclass(frozen=True, slots=True)
class PublicPredicate:
    """One public filter condition; multiple items are conjunctive unless ``group`` differs."""

    predicate_id: str
    metric_role: str
    operator: str
    threshold_source: Literal["fixed", "derived", "zero", "relation"]
    threshold_value: float | None = None
    statistic: str | None = None
    related_metric_role: str | None = None
    periods: tuple[str, ...] = ()
    group: str = "main"
    transform: str | None = None


@dataclass(frozen=True, slots=True)
class PublicCohort:
    cohort_id: str
    construction: Literal[
        "predicate",
        "intersection",
        "top_n",
        "top_quantile",
        "bottom_quantile",
        "median_split",
    ]
    source_roles: tuple[str, ...]
    predicate_ids: tuple[str, ...] = ()
    size: int | None = None
    quantile_percent: float | None = None
    equality_policy: str | None = None
    rounding_policy: str | None = None
    minimum_size: int | None = None


@dataclass(frozen=True, slots=True)
class PublicTerminalBranch:
    branch_id: str
    operation: str
    input_branches: tuple[str, ...] = ()
    metric_role: str | None = None
    cohort_id: str | None = None
    selector_step: str | None = None


@dataclass(frozen=True, slots=True)
class PublicSpec:

    recipe_id: str
    candidate_id: str
    entities: tuple[str, ...]
    company_names: tuple[str, ...]
    periods: tuple[str, ...]
    reference_period: str | None
    metric_labels: Mapping[str, str]
    metric_keys: Mapping[str, str]
    metric_roles: Mapping[str, tuple[Literal["filter", "rank", "target"], ...]]
    threshold_operator: str | None
    threshold_source: str | None  # "convention" | "derived" | None
    threshold_statistic: (
        str | None
    )  # "median" | "average_threshold" | "percentile" | None
    threshold_convention_value: float | None
    selector_direction: str | None  # "argmax" | "argmin" | None
    aggregate_operation: (
        str | None
    )  # "sum" | "average" | "minimum" | "maximum" | "maximum"(R2) | None
    target_value_kind: str
    target_unit_label: str
    intent_id: str | None = None
    analytical_frame: str | None = None
    story_family: str | None = None
    filter_transform: str | None = None
    rank_transform: str | None = None
    target_transform: str | None = None
    terminal_measurement_name: str | None = None
    interpretation_limits: tuple[str, ...] = ()
    required_question_terms: tuple[str, ...] = ()
    # Reviewed-template contract.  Empty for legacy R1-R7/analytical frames; populated by the
    # template compiler.  These fields preserve every predicate/cohort/terminal branch instead of
    # flattening an intent to one filter/rank/target triple.
    template_id: str | None = None
    universe_kind: str | None = None
    universe_description: str | None = None
    operation_sequence: tuple[str, ...] = ()
    predicates: tuple[PublicPredicate, ...] = ()
    cohorts: tuple[PublicCohort, ...] = ()
    terminal_branches: tuple[PublicTerminalBranch, ...] = ()
    tie_policy: str | None = None


@dataclass(frozen=True, slots=True)
class RecipeCandidate:
    candidate_id: str
    recipe_id: str
    entities: tuple[str, ...]
    graph: ReasoningGraph
    trace: EvaluationTrace
    compiled: CompiledQuery
    formatted_answer: float
    audit_stats: AuditStats
    public_spec: PublicSpec


_finalize_candidate = finalize_candidate
_company_names = company_names


def table_ref_to_path_map(docs: list[DocumentRef]) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for doc in docs:
        if doc.tables_dir is None:
            continue
        for table_id in doc.table_ids:
            mapping[f"{doc.doc_name}|table_{table_id}"] = doc.table_csv_path(table_id)
    return mapping


def _build_r1_candidate(
    *,
    cube: Cube,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    target_metric: str,
) -> RecipeCandidate:
    candidate_periods = periods[1:]

    revenue_bindings: dict[tuple[str, str], MetricTerms] = {}
    for entity in entities:
        for period in periods:
            terms = _resolve_metric_terms(cube, entity, period, REVENUE_METRIC)
            if terms is None or terms.value is None:
                raise CandidateRejected(f"Missing revenue coverage for {entity}/{period}")
            revenue_bindings[(entity, period)] = terms

    target_bindings: dict[tuple[str, str], MetricTerms] = {}
    for entity in entities:
        for period in candidate_periods:
            terms = _resolve_metric_terms(cube, entity, period, target_metric)
            if terms is None or terms.value is None:
                raise CandidateRejected(
                    f"Missing target coverage for target={target_metric} cho {entity}/{period}"
                )
            target_bindings[(entity, period)] = terms

    revenue_role = MetricRoleInput(
        "revenue",
        REVENUE_METRIC,
        ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        revenue_bindings,
    )
    target_role = MetricRoleInput(
        "target", target_metric, ValueKind.NUMERIC_SERIES_ENTITY_PERIOD, target_bindings
    )

    provisional_graph = build_r1_graph(
        entities=entities,
        periods=periods,
        report_scope=REPORT_SCOPE,
        revenue_role=revenue_role,
        target_role=target_role,
    )
    validated_graph, trace, compiled, formatted_answer, audit_stats = (
        _finalize_candidate(
            provisional_graph=provisional_graph,
            terminal_metric_key=target_metric,
            docs_by_name=docs_by_name,
            company_meta=company_meta,
            table_ref_to_path=table_ref_to_path,
            auditor=auditor,
            audit_cache=audit_cache,
        )
    )

    candidate_id = _candidate_signature(R1_RECIPE_ID, entities, periods, target_metric)
    value_kind = terminal_value_kind(target_metric)
    public_spec = PublicSpec(
        recipe_id=R1_RECIPE_ID,
        candidate_id=candidate_id,
        entities=entities,
        company_names=_company_names(entities, company_meta),
        periods=periods,
        reference_period=None,
        metric_labels={
            "A": display_name(REVENUE_METRIC) or REVENUE_METRIC,
            "B": display_name(target_metric) or target_metric,
        },
        metric_keys={"A": REVENUE_METRIC, "B": target_metric},
        metric_roles={"A": ("rank",), "B": ("target",)},
        threshold_operator=None,
        threshold_source=None,
        threshold_statistic=None,
        threshold_convention_value=None,
        selector_direction="argmax",
        aggregate_operation=None,
        target_value_kind=value_kind,
        target_unit_label=_unit_label(value_kind),
    )

    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=R1_RECIPE_ID,
        entities=entities,
        graph=validated_graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=formatted_answer,
        audit_stats=audit_stats,
        public_spec=public_spec,
    )


def iter_r1_candidates(
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    rng = random.Random(seed)
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)
    tickers = [t for t in cube.tickers() if company_meta.get(t) is not None]

    seen_signatures: set[str] = set()
    tried = 0
    while tried < max_candidates and tickers:
        tried += 1
        seed_ticker = rng.choice(tickers)
        peers = sorted(peer_tickers(seed_ticker, company_meta) & set(cube.tickers()))
        if len(peers) < MIN_ENTITIES:
            continue
        entity_count = rng.randint(MIN_ENTITIES, min(MAX_ENTITIES, len(peers)))
        entities = tuple(sorted(rng.sample(peers, entity_count)))

        common_years: set[str] | None = None
        for entity in entities:
            years = {
                y
                for y in cube.years(entity)
                if _resolve_metric_terms(cube, entity, y, REVENUE_METRIC) is not None
            }
            common_years = years if common_years is None else common_years & years
        if not common_years:
            continue
        contiguous = _longest_contiguous_run(common_years)
        if len(contiguous) < MIN_CANDIDATE_YEARS + 1:
            continue

        candidate_year_count = rng.randint(
            MIN_CANDIDATE_YEARS, min(MAX_CANDIDATE_YEARS, len(contiguous) - 1)
        )
        window_len = candidate_year_count + 1
        start = rng.randint(0, len(contiguous) - window_len)
        periods = tuple(contiguous[start : start + window_len])

        target_metric = rng.choice(TARGET_METRIC_CANDIDATES)
        signature = _candidate_signature(R1_RECIPE_ID, entities, periods, target_metric)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        try:
            candidate = _build_r1_candidate(
                cube=cube,
                docs_by_name=docs_by_name,
                company_meta=company_meta,
                table_ref_to_path=table_ref_to_path,
                auditor=auditor,
                audit_cache=audit_cache,
                entities=entities,
                periods=periods,
                target_metric=target_metric,
            )
        except CandidateRejected as exc:
            logger.debug("Candidate reject (coverage/hardness/BUG): %s", exc)
            continue
        except DependencyAuditRejected as exc:
            logger.info("Candidate reject (JIT audit): %s", exc.reasons)
            continue
        except EvaluationError as exc:
            logger.debug(
                "Candidate rejected during evaluation (tie, zero denominator, or missing coverage): %s", exc
            )
            continue

        yield candidate




def _sample_peer_group(
    rng: random.Random,
    cube: Cube,
    company_meta: dict[str, CompanyInfo],
    *,
    min_entities: int,
    max_entities: int,
) -> tuple[str, ...] | None:
    tickers = [t for t in cube.tickers() if company_meta.get(t) is not None]
    if not tickers:
        return None
    seed_ticker = rng.choice(tickers)
    peers = sorted(peer_tickers(seed_ticker, company_meta) & set(cube.tickers()))
    if len(peers) < min_entities:
        return None
    entity_count = rng.randint(min_entities, min(max_entities, len(peers)))
    return tuple(sorted(rng.sample(peers, entity_count)))


def _years_with_coverage(
    cube: Cube, entities: tuple[str, ...], metric_key: str
) -> set[str]:
    common: set[str] | None = None
    for entity in entities:
        years = {
            y
            for y in cube.years(entity)
            if _resolve_metric_terms(cube, entity, y, metric_key) is not None
        }
        common = years if common is None else common & years
        if not common:
            return set()
    return common or set()


def _company_years_with_coverage(cube: Cube, ticker: str, metric_key: str) -> set[str]:
    return {
        y
        for y in cube.years(ticker)
        if _resolve_metric_terms(cube, ticker, y, metric_key) is not None
    }


def _build_entity_role(
    cube: Cube, role_id: str, metric_key: str, entities: tuple[str, ...], period: str
) -> MetricRoleInput:
    bindings: dict[str, MetricTerms] = {}
    for entity in entities:
        terms = _resolve_metric_terms(cube, entity, period, metric_key)
        if terms is None or terms.value is None:
            raise CandidateRejected(
                f"Missing coverage for {metric_key} cho {entity}/{period}"
            )
        bindings[entity] = terms
    return MetricRoleInput(
        role_id, metric_key, ValueKind.NUMERIC_SERIES_ENTITY, bindings
    )


def _build_entity_period_role(
    cube: Cube,
    role_id: str,
    metric_key: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
) -> MetricRoleInput:
    bindings: dict[tuple[str, str], MetricTerms] = {}
    for entity in entities:
        for period in periods:
            terms = _resolve_metric_terms(cube, entity, period, metric_key)
            if terms is None or terms.value is None:
                raise CandidateRejected(
                    f"Missing coverage for {metric_key} cho {entity}/{period}"
                )
            bindings[(entity, period)] = terms
    return MetricRoleInput(
        role_id, metric_key, ValueKind.NUMERIC_SERIES_ENTITY_PERIOD, bindings
    )


def _build_period_role(
    cube: Cube, role_id: str, metric_key: str, ticker: str, periods: tuple[str, ...]
) -> MetricRoleInput:
    bindings: dict[str, MetricTerms] = {}
    for period in periods:
        terms = _resolve_metric_terms(cube, ticker, period, metric_key)
        if terms is None or terms.value is None:
            raise CandidateRejected(
                f"Missing coverage for {metric_key} cho {ticker}/{period}"
            )
        bindings[period] = terms
    return MetricRoleInput(
        role_id, metric_key, ValueKind.NUMERIC_SERIES_PERIOD, bindings
    )


def _pick_derived_threshold(rng: random.Random) -> tuple[str, float | None]:
    statistic = rng.choice(DERIVED_THRESHOLD_STATISTICS)
    percentile_value = (
        rng.choice(PERCENTILE_CANDIDATES) if statistic == "percentile" else None
    )
    return statistic, percentile_value


def _statistic_label(statistic: str, percentile_value: float | None) -> str:
    return {
        "median": "trung vị",
        "average_threshold": "trung bình",
        "percentile": f"phân vị {percentile_value}",
    }[statistic]


# --- R2 — temporal_filter_maximum -------------------------------------------------------------


def _build_r2_candidate(
    *,
    cube: Cube,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    entities: tuple[str, ...],
    temporal_periods: tuple[str, ...],
    reference_period: str,
    temporal_metric: str,
    target_metric: str,
) -> RecipeCandidate:
    temporal_role = _build_entity_period_role(
        cube, "temporal", temporal_metric, entities, temporal_periods
    )
    target_role = _build_entity_role(
        cube, "target", target_metric, entities, reference_period
    )

    provisional_graph = build_r2_graph(
        entities=entities,
        temporal_periods=temporal_periods,
        reference_period=reference_period,
        report_scope=REPORT_SCOPE,
        temporal_role=temporal_role,
        temporal_operator=">",
        temporal_threshold=0.0,
        target_role=target_role,
    )
    validated_graph, trace, compiled, formatted_answer, audit_stats = (
        _finalize_candidate(
            provisional_graph=provisional_graph,
            terminal_metric_key=target_metric,
            docs_by_name=docs_by_name,
            company_meta=company_meta,
            table_ref_to_path=table_ref_to_path,
            auditor=auditor,
            audit_cache=audit_cache,
        )
    )

    candidate_id = _candidate_signature(
        R2_RECIPE_ID,
        entities,
        temporal_periods,
        reference_period,
        temporal_metric,
        target_metric,
    )
    value_kind = terminal_value_kind(target_metric)
    public_spec = PublicSpec(
        recipe_id=R2_RECIPE_ID,
        candidate_id=candidate_id,
        entities=entities,
        company_names=_company_names(entities, company_meta),
        periods=temporal_periods,
        reference_period=reference_period,
        metric_labels={
            "A": display_name(temporal_metric) or temporal_metric,
            "B": display_name(target_metric) or target_metric,
        },
        metric_keys={"A": temporal_metric, "B": target_metric},
        metric_roles={"A": ("filter",), "B": ("target",)},
        threshold_operator=">",
        threshold_source="convention",
        threshold_statistic=None,
        threshold_convention_value=0.0,
        selector_direction=None,
        aggregate_operation="maximum",
        target_value_kind=value_kind,
        target_unit_label=_unit_label(value_kind),
    )
    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=R2_RECIPE_ID,
        entities=entities,
        graph=validated_graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=formatted_answer,
        audit_stats=audit_stats,
        public_spec=public_spec,
    )


def iter_r2_candidates(
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    rng = random.Random(seed)
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)

    seen_signatures: set[str] = set()
    tried = 0
    while tried < max_candidates:
        tried += 1
        entities = _sample_peer_group(
            rng,
            cube,
            company_meta,
            min_entities=MIN_ENTITIES,
            max_entities=MAX_ENTITIES,
        )
        if entities is None:
            continue
        temporal_metric = rng.choice(TEMPORAL_METRIC_CANDIDATES)
        temporal_years = _years_with_coverage(cube, entities, temporal_metric)
        contiguous = _longest_contiguous_run(temporal_years)
        if len(contiguous) < MIN_TEMPORAL_YEARS:
            continue
        window_len = rng.randint(
            MIN_TEMPORAL_YEARS, min(MAX_TEMPORAL_YEARS, len(contiguous))
        )
        start = rng.randint(0, len(contiguous) - window_len)
        temporal_periods = tuple(contiguous[start : start + window_len])
        reference_period = temporal_periods[-1]

        target_metric = rng.choice(TARGET_METRIC_CANDIDATES)
        if target_metric == temporal_metric:
            continue
        if not all(
            _resolve_metric_terms(cube, e, reference_period, target_metric) is not None
            for e in entities
        ):
            continue

        signature = _candidate_signature(
            R2_RECIPE_ID,
            entities,
            temporal_periods,
            reference_period,
            temporal_metric,
            target_metric,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        try:
            candidate = _build_r2_candidate(
                cube=cube,
                docs_by_name=docs_by_name,
                company_meta=company_meta,
                table_ref_to_path=table_ref_to_path,
                auditor=auditor,
                audit_cache=audit_cache,
                entities=entities,
                temporal_periods=temporal_periods,
                reference_period=reference_period,
                temporal_metric=temporal_metric,
                target_metric=target_metric,
            )
        except CandidateRejected as exc:
            logger.debug("R2 candidate reject (coverage/hardness/BUG): %s", exc)
            continue
        except DependencyAuditRejected as exc:
            logger.info("R2 candidate reject (JIT audit): %s", exc.reasons)
            continue
        except EvaluationError as exc:
            logger.debug(
                "R2 candidate rejected during evaluation (non-triviality or missing coverage): %s", exc
            )
            continue

        yield candidate


# --- R3 — temporal_filter_argmax_lookup -------------------------------------------------------


def _build_r3_candidate(
    *,
    cube: Cube,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    entities: tuple[str, ...],
    temporal_periods: tuple[str, ...],
    reference_period: str,
    temporal_metric: str,
    rank_metric: str,
    rank_direction: str,
    target_metric: str,
) -> RecipeCandidate:
    temporal_role = _build_entity_period_role(
        cube, "temporal", temporal_metric, entities, temporal_periods
    )
    rank_role = _build_entity_role(
        cube, "rank", rank_metric, entities, reference_period
    )
    target_role = _build_entity_role(
        cube, "target", target_metric, entities, reference_period
    )

    provisional_graph = build_r3_graph(
        entities=entities,
        temporal_periods=temporal_periods,
        reference_period=reference_period,
        report_scope=REPORT_SCOPE,
        temporal_role=temporal_role,
        temporal_operator=">",
        temporal_threshold=0.0,
        rank_role=rank_role,
        rank_direction=rank_direction,
        target_role=target_role,
    )
    validated_graph, trace, compiled, formatted_answer, audit_stats = (
        _finalize_candidate(
            provisional_graph=provisional_graph,
            terminal_metric_key=target_metric,
            docs_by_name=docs_by_name,
            company_meta=company_meta,
            table_ref_to_path=table_ref_to_path,
            auditor=auditor,
            audit_cache=audit_cache,
        )
    )

    candidate_id = _candidate_signature(
        R3_RECIPE_ID,
        entities,
        temporal_periods,
        reference_period,
        temporal_metric,
        rank_metric,
        rank_direction,
        target_metric,
    )
    value_kind = terminal_value_kind(target_metric)
    public_spec = PublicSpec(
        recipe_id=R3_RECIPE_ID,
        candidate_id=candidate_id,
        entities=entities,
        company_names=_company_names(entities, company_meta),
        periods=temporal_periods,
        reference_period=reference_period,
        metric_labels={
            "A": display_name(temporal_metric) or temporal_metric,
            "B": display_name(rank_metric) or rank_metric,
            "C": display_name(target_metric) or target_metric,
        },
        metric_keys={"A": temporal_metric, "B": rank_metric, "C": target_metric},
        metric_roles={"A": ("filter",), "B": ("rank",), "C": ("target",)},
        threshold_operator=">",
        threshold_source="convention",
        threshold_statistic=None,
        threshold_convention_value=0.0,
        selector_direction=rank_direction,
        aggregate_operation=None,
        target_value_kind=value_kind,
        target_unit_label=_unit_label(value_kind),
    )
    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=R3_RECIPE_ID,
        entities=entities,
        graph=validated_graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=formatted_answer,
        audit_stats=audit_stats,
        public_spec=public_spec,
    )


def iter_r3_candidates(
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    rng = random.Random(seed)
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)

    seen_signatures: set[str] = set()
    tried = 0
    while tried < max_candidates:
        tried += 1
        entities = _sample_peer_group(
            rng,
            cube,
            company_meta,
            min_entities=MIN_R3_ENTITIES,
            max_entities=MAX_R3_ENTITIES,
        )
        if entities is None:
            continue
        temporal_metric = rng.choice(TEMPORAL_METRIC_CANDIDATES)
        temporal_years = _years_with_coverage(cube, entities, temporal_metric)
        contiguous = _longest_contiguous_run(temporal_years)
        if len(contiguous) < MIN_TEMPORAL_YEARS:
            continue
        window_len = rng.randint(
            MIN_TEMPORAL_YEARS, min(MAX_TEMPORAL_YEARS, len(contiguous))
        )
        start = rng.randint(0, len(contiguous) - window_len)
        temporal_periods = tuple(contiguous[start : start + window_len])
        reference_period = temporal_periods[-1]

        role_pool = [m for m in TARGET_METRIC_CANDIDATES if m != temporal_metric]
        if len(role_pool) < 2:
            continue
        rank_metric, target_metric = rng.sample(role_pool, 2)
        if not all(
            _resolve_metric_terms(cube, e, reference_period, m) is not None
            for e in entities
            for m in (rank_metric, target_metric)
        ):
            continue
        rank_direction = rng.choice(SELECT_DIRECTIONS)

        signature = _candidate_signature(
            R3_RECIPE_ID,
            entities,
            temporal_periods,
            reference_period,
            temporal_metric,
            rank_metric,
            rank_direction,
            target_metric,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        try:
            candidate = _build_r3_candidate(
                cube=cube,
                docs_by_name=docs_by_name,
                company_meta=company_meta,
                table_ref_to_path=table_ref_to_path,
                auditor=auditor,
                audit_cache=audit_cache,
                entities=entities,
                temporal_periods=temporal_periods,
                reference_period=reference_period,
                temporal_metric=temporal_metric,
                rank_metric=rank_metric,
                rank_direction=rank_direction,
                target_metric=target_metric,
            )
        except CandidateRejected as exc:
            logger.debug("R3 candidate reject (coverage/hardness/BUG): %s", exc)
            continue
        except DependencyAuditRejected as exc:
            logger.info("R3 candidate reject (JIT audit): %s", exc.reasons)
            continue
        except EvaluationError as exc:
            logger.debug(
                "R3 candidate rejected during evaluation (tie, non-triviality, or missing coverage): %s",
                exc,
            )
            continue

        yield candidate


# --- R4 — entity_convention_filter_aggregate --------------------------------------------------


def _build_r4_candidate(
    *,
    cube: Cube,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    entities: tuple[str, ...],
    reference_period: str,
    filter_metric: str,
    target_metric: str,
    aggregate_operation: str,
) -> RecipeCandidate:
    filter_operator, filter_threshold_value = CONVENTION_THRESHOLDS[filter_metric]
    filter_role = _build_entity_role(
        cube, "filter_a", filter_metric, entities, reference_period
    )
    target_role = _build_entity_role(
        cube, "target_b", target_metric, entities, reference_period
    )

    provisional_graph = build_r4_graph(
        entities=entities,
        reference_period=reference_period,
        report_scope=REPORT_SCOPE,
        filter_role=filter_role,
        filter_operator=filter_operator,
        filter_threshold_value=filter_threshold_value,
        target_role=target_role,
        aggregate_operation=aggregate_operation,
    )
    validated_graph, trace, compiled, formatted_answer, audit_stats = (
        _finalize_candidate(
            provisional_graph=provisional_graph,
            terminal_metric_key=target_metric,
            docs_by_name=docs_by_name,
            company_meta=company_meta,
            table_ref_to_path=table_ref_to_path,
            auditor=auditor,
            audit_cache=audit_cache,
        )
    )

    candidate_id = _candidate_signature(
        R4_RECIPE_ID,
        entities,
        reference_period,
        filter_metric,
        target_metric,
        aggregate_operation,
    )
    value_kind = terminal_value_kind(target_metric)
    public_spec = PublicSpec(
        recipe_id=R4_RECIPE_ID,
        candidate_id=candidate_id,
        entities=entities,
        company_names=_company_names(entities, company_meta),
        periods=(reference_period,),
        reference_period=reference_period,
        metric_labels={
            "A": display_name(filter_metric) or filter_metric,
            "B": display_name(target_metric) or target_metric,
        },
        metric_keys={"A": filter_metric, "B": target_metric},
        metric_roles={"A": ("filter",), "B": ("target",)},
        threshold_operator=filter_operator,
        threshold_source="convention",
        threshold_statistic=None,
        threshold_convention_value=filter_threshold_value,
        selector_direction=None,
        aggregate_operation=aggregate_operation,
        target_value_kind=value_kind,
        target_unit_label=_unit_label(value_kind),
    )
    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=R4_RECIPE_ID,
        entities=entities,
        graph=validated_graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=formatted_answer,
        audit_stats=audit_stats,
        public_spec=public_spec,
    )


def iter_r4_candidates(
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    rng = random.Random(seed)
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)
    filter_metrics = tuple(CONVENTION_THRESHOLDS)

    seen_signatures: set[str] = set()
    tried = 0
    while tried < max_candidates:
        tried += 1
        entities = _sample_peer_group(
            rng,
            cube,
            company_meta,
            min_entities=MIN_R4_ENTITIES,
            max_entities=MAX_R4_ENTITIES,
        )
        if entities is None:
            continue
        filter_metric = rng.choice(filter_metrics)
        target_metric = rng.choice(TARGET_METRIC_CANDIDATES)
        if target_metric == filter_metric:
            continue

        candidate_years = _years_with_coverage(
            cube, entities, filter_metric
        ) & _years_with_coverage(cube, entities, target_metric)
        if not candidate_years:
            continue
        reference_period = rng.choice(sorted(candidate_years))
        aggregate_operation = rng.choice(_aggregate_operations_for(target_metric))

        signature = _candidate_signature(
            R4_RECIPE_ID,
            entities,
            reference_period,
            filter_metric,
            target_metric,
            aggregate_operation,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        try:
            candidate = _build_r4_candidate(
                cube=cube,
                docs_by_name=docs_by_name,
                company_meta=company_meta,
                table_ref_to_path=table_ref_to_path,
                auditor=auditor,
                audit_cache=audit_cache,
                entities=entities,
                reference_period=reference_period,
                filter_metric=filter_metric,
                target_metric=target_metric,
                aggregate_operation=aggregate_operation,
            )
        except CandidateRejected as exc:
            logger.debug("R4 candidate reject (coverage/hardness/BUG): %s", exc)
            continue
        except DependencyAuditRejected as exc:
            logger.info("R4 candidate reject (JIT audit): %s", exc.reasons)
            continue
        except EvaluationError as exc:
            logger.debug(
                "R4 candidate rejected during evaluation (non-triviality or missing coverage): %s", exc
            )
            continue

        yield candidate


# --- R5 — entity_derived_threshold_filter_aggregate --------------------------------------------


def _build_r5_candidate(
    *,
    cube: Cube,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    entities: tuple[str, ...],
    reference_period: str,
    filter_metric: str,
    filter_operator: str,
    threshold_statistic: str,
    percentile_value: float | None,
    target_metric: str,
    aggregate_operation: str,
) -> RecipeCandidate:
    filter_role = _build_entity_role(
        cube, "filter_a", filter_metric, entities, reference_period
    )
    target_role = _build_entity_role(
        cube, "target_b", target_metric, entities, reference_period
    )

    provisional_graph = build_r5_graph(
        entities=entities,
        reference_period=reference_period,
        report_scope=REPORT_SCOPE,
        filter_role=filter_role,
        filter_operator=filter_operator,
        threshold_statistic=threshold_statistic,
        percentile_value=percentile_value,
        target_role=target_role,
        aggregate_operation=aggregate_operation,
    )
    validated_graph, trace, compiled, formatted_answer, audit_stats = (
        _finalize_candidate(
            provisional_graph=provisional_graph,
            terminal_metric_key=target_metric,
            docs_by_name=docs_by_name,
            company_meta=company_meta,
            table_ref_to_path=table_ref_to_path,
            auditor=auditor,
            audit_cache=audit_cache,
        )
    )

    candidate_id = _candidate_signature(
        R5_RECIPE_ID,
        entities,
        reference_period,
        filter_metric,
        filter_operator,
        threshold_statistic,
        percentile_value,
        target_metric,
        aggregate_operation,
    )
    value_kind = terminal_value_kind(target_metric)
    public_spec = PublicSpec(
        recipe_id=R5_RECIPE_ID,
        candidate_id=candidate_id,
        entities=entities,
        company_names=_company_names(entities, company_meta),
        periods=(reference_period,),
        reference_period=reference_period,
        metric_labels={
            "A": display_name(filter_metric) or filter_metric,
            "B": display_name(target_metric) or target_metric,
        },
        metric_keys={"A": filter_metric, "B": target_metric},
        metric_roles={"A": ("filter",), "B": ("target",)},
        threshold_operator=filter_operator,
        threshold_source="derived",
        threshold_statistic=threshold_statistic,
        threshold_convention_value=None,
        selector_direction=None,
        aggregate_operation=aggregate_operation,
        target_value_kind=value_kind,
        target_unit_label=_unit_label(value_kind),
    )
    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=R5_RECIPE_ID,
        entities=entities,
        graph=validated_graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=formatted_answer,
        audit_stats=audit_stats,
        public_spec=public_spec,
    )


def iter_r5_candidates(
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    rng = random.Random(seed)
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)

    seen_signatures: set[str] = set()
    tried = 0
    while tried < max_candidates:
        tried += 1
        entities = _sample_peer_group(
            rng,
            cube,
            company_meta,
            min_entities=MIN_R5_ENTITIES,
            max_entities=MAX_R5_ENTITIES,
        )
        if entities is None:
            continue
        filter_metric, target_metric = rng.sample(TARGET_METRIC_CANDIDATES, 2)
        candidate_years = _years_with_coverage(
            cube, entities, filter_metric
        ) & _years_with_coverage(cube, entities, target_metric)
        if not candidate_years:
            continue
        reference_period = rng.choice(sorted(candidate_years))
        filter_operator = rng.choice(("<", ">"))
        threshold_statistic, percentile_value = _pick_derived_threshold(rng)
        aggregate_operation = rng.choice(_aggregate_operations_for(target_metric))

        signature = _candidate_signature(
            R5_RECIPE_ID,
            entities,
            reference_period,
            filter_metric,
            filter_operator,
            threshold_statistic,
            percentile_value,
            target_metric,
            aggregate_operation,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        try:
            candidate = _build_r5_candidate(
                cube=cube,
                docs_by_name=docs_by_name,
                company_meta=company_meta,
                table_ref_to_path=table_ref_to_path,
                auditor=auditor,
                audit_cache=audit_cache,
                entities=entities,
                reference_period=reference_period,
                filter_metric=filter_metric,
                filter_operator=filter_operator,
                threshold_statistic=threshold_statistic,
                percentile_value=percentile_value,
                target_metric=target_metric,
                aggregate_operation=aggregate_operation,
            )
        except CandidateRejected as exc:
            logger.debug("R5 candidate reject (coverage/hardness/BUG): %s", exc)
            continue
        except DependencyAuditRejected as exc:
            logger.info("R5 candidate reject (JIT audit): %s", exc.reasons)
            continue
        except EvaluationError as exc:
            logger.debug(
                "R5 candidate rejected during evaluation (non-triviality or missing coverage): %s", exc
            )
            continue

        yield candidate


# --- R6 — period_convention_filter_select_lookup -----------------------------------------------


def _build_r6_candidate(
    *,
    cube: Cube,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    ticker: str,
    periods: tuple[str, ...],
    filter_metric: str,
    select_metric: str,
    select_direction: str,
    target_metric: str,
) -> RecipeCandidate:
    filter_operator, filter_threshold_value = CONVENTION_THRESHOLDS[filter_metric]
    filter_role = _build_period_role(cube, "filter_a", filter_metric, ticker, periods)
    select_role = _build_period_role(cube, "select_b", select_metric, ticker, periods)
    target_role = _build_period_role(cube, "target_c", target_metric, ticker, periods)

    provisional_graph = build_r6_graph(
        ticker=ticker,
        periods=periods,
        report_scope=REPORT_SCOPE,
        filter_role=filter_role,
        filter_operator=filter_operator,
        filter_threshold_value=filter_threshold_value,
        select_role=select_role,
        select_direction=select_direction,
        target_role=target_role,
    )
    validated_graph, trace, compiled, formatted_answer, audit_stats = (
        _finalize_candidate(
            provisional_graph=provisional_graph,
            terminal_metric_key=target_metric,
            docs_by_name=docs_by_name,
            company_meta=company_meta,
            table_ref_to_path=table_ref_to_path,
            auditor=auditor,
            audit_cache=audit_cache,
        )
    )

    candidate_id = _candidate_signature(
        R6_RECIPE_ID,
        ticker,
        periods,
        filter_metric,
        select_metric,
        select_direction,
        target_metric,
    )
    value_kind = terminal_value_kind(target_metric)
    public_spec = PublicSpec(
        recipe_id=R6_RECIPE_ID,
        candidate_id=candidate_id,
        entities=(ticker,),
        company_names=_company_names((ticker,), company_meta),
        periods=periods,
        reference_period=None,
        metric_labels={
            "A": display_name(filter_metric) or filter_metric,
            "B": display_name(select_metric) or select_metric,
            "C": display_name(target_metric) or target_metric,
        },
        metric_keys={"A": filter_metric, "B": select_metric, "C": target_metric},
        metric_roles={"A": ("filter",), "B": ("rank",), "C": ("target",)},
        threshold_operator=filter_operator,
        threshold_source="convention",
        threshold_statistic=None,
        threshold_convention_value=filter_threshold_value,
        selector_direction=select_direction,
        aggregate_operation=None,
        target_value_kind=value_kind,
        target_unit_label=_unit_label(value_kind),
    )
    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=R6_RECIPE_ID,
        entities=(ticker,),
        graph=validated_graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=formatted_answer,
        audit_stats=audit_stats,
        public_spec=public_spec,
    )


def iter_r6_candidates(
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    rng = random.Random(seed)
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)
    filter_metrics = tuple(CONVENTION_THRESHOLDS)
    tickers = [t for t in cube.tickers() if company_meta.get(t) is not None]

    seen_signatures: set[str] = set()
    tried = 0
    while tried < max_candidates and tickers:
        tried += 1
        ticker = rng.choice(tickers)
        filter_metric = rng.choice(filter_metrics)
        role_pool = [m for m in TARGET_METRIC_CANDIDATES if m != filter_metric]
        if len(role_pool) < 2:
            continue
        select_metric, target_metric = rng.sample(role_pool, 2)

        common = (
            _company_years_with_coverage(cube, ticker, filter_metric)
            & _company_years_with_coverage(cube, ticker, select_metric)
            & _company_years_with_coverage(cube, ticker, target_metric)
        )
        contiguous = _longest_contiguous_run(common)
        if len(contiguous) < MIN_R6_PERIODS:
            continue
        window_len = rng.randint(MIN_R6_PERIODS, min(MAX_R6_PERIODS, len(contiguous)))
        start = rng.randint(0, len(contiguous) - window_len)
        periods = tuple(contiguous[start : start + window_len])
        select_direction = rng.choice(SELECT_DIRECTIONS)

        signature = _candidate_signature(
            R6_RECIPE_ID,
            ticker,
            periods,
            filter_metric,
            select_metric,
            select_direction,
            target_metric,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        try:
            candidate = _build_r6_candidate(
                cube=cube,
                docs_by_name=docs_by_name,
                company_meta=company_meta,
                table_ref_to_path=table_ref_to_path,
                auditor=auditor,
                audit_cache=audit_cache,
                ticker=ticker,
                periods=periods,
                filter_metric=filter_metric,
                select_metric=select_metric,
                select_direction=select_direction,
                target_metric=target_metric,
            )
        except CandidateRejected as exc:
            logger.debug("R6 candidate reject (coverage/hardness/BUG): %s", exc)
            continue
        except DependencyAuditRejected as exc:
            logger.info("R6 candidate reject (JIT audit): %s", exc.reasons)
            continue
        except EvaluationError as exc:
            logger.debug(
                "R6 candidate rejected during evaluation (tie, non-triviality, or missing coverage): %s",
                exc,
            )
            continue

        yield candidate


# --- R7 — period_derived_threshold_select_lookup -----------------------------------------------


def _build_r7_candidate(
    *,
    cube: Cube,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    ticker: str,
    periods: tuple[str, ...],
    filter_metric: str,
    filter_operator: str,
    threshold_statistic: str,
    percentile_value: float | None,
    select_metric: str,
    select_direction: str,
    target_metric: str,
) -> RecipeCandidate:
    filter_role = _build_period_role(cube, "filter_a", filter_metric, ticker, periods)
    select_role = _build_period_role(cube, "select_b", select_metric, ticker, periods)
    target_role = _build_period_role(cube, "target_c", target_metric, ticker, periods)

    provisional_graph = build_r7_graph(
        ticker=ticker,
        periods=periods,
        report_scope=REPORT_SCOPE,
        filter_role=filter_role,
        filter_operator=filter_operator,
        threshold_statistic=threshold_statistic,
        percentile_value=percentile_value,
        select_role=select_role,
        select_direction=select_direction,
        target_role=target_role,
    )
    validated_graph, trace, compiled, formatted_answer, audit_stats = (
        _finalize_candidate(
            provisional_graph=provisional_graph,
            terminal_metric_key=target_metric,
            docs_by_name=docs_by_name,
            company_meta=company_meta,
            table_ref_to_path=table_ref_to_path,
            auditor=auditor,
            audit_cache=audit_cache,
        )
    )

    candidate_id = _candidate_signature(
        R7_RECIPE_ID,
        ticker,
        periods,
        filter_metric,
        filter_operator,
        threshold_statistic,
        percentile_value,
        select_metric,
        select_direction,
        target_metric,
    )
    value_kind = terminal_value_kind(target_metric)
    public_spec = PublicSpec(
        recipe_id=R7_RECIPE_ID,
        candidate_id=candidate_id,
        entities=(ticker,),
        company_names=_company_names((ticker,), company_meta),
        periods=periods,
        reference_period=None,
        metric_labels={
            "A": display_name(filter_metric) or filter_metric,
            "B": display_name(select_metric) or select_metric,
            "C": display_name(target_metric) or target_metric,
        },
        metric_keys={"A": filter_metric, "B": select_metric, "C": target_metric},
        metric_roles={"A": ("filter",), "B": ("rank",), "C": ("target",)},
        threshold_operator=filter_operator,
        threshold_source="derived",
        threshold_statistic=threshold_statistic,
        threshold_convention_value=None,
        selector_direction=select_direction,
        aggregate_operation=None,
        target_value_kind=value_kind,
        target_unit_label=_unit_label(value_kind),
    )
    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=R7_RECIPE_ID,
        entities=(ticker,),
        graph=validated_graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=formatted_answer,
        audit_stats=audit_stats,
        public_spec=public_spec,
    )


def iter_r7_candidates(
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    rng = random.Random(seed)
    docs_by_name = {doc.doc_name: doc for doc in docs}
    table_ref_to_path = table_ref_to_path_map(docs)
    tickers = [t for t in cube.tickers() if company_meta.get(t) is not None]

    seen_signatures: set[str] = set()
    tried = 0
    while tried < max_candidates and tickers:
        tried += 1
        ticker = rng.choice(tickers)
        filter_metric, select_metric, target_metric = rng.sample(
            TARGET_METRIC_CANDIDATES, 3
        )

        common = (
            _company_years_with_coverage(cube, ticker, filter_metric)
            & _company_years_with_coverage(cube, ticker, select_metric)
            & _company_years_with_coverage(cube, ticker, target_metric)
        )
        contiguous = _longest_contiguous_run(common)
        if len(contiguous) < MIN_R7_PERIODS:
            continue
        window_len = rng.randint(MIN_R7_PERIODS, min(MAX_R7_PERIODS, len(contiguous)))
        start = rng.randint(0, len(contiguous) - window_len)
        periods = tuple(contiguous[start : start + window_len])

        filter_operator = rng.choice(("<", ">"))
        threshold_statistic, percentile_value = _pick_derived_threshold(rng)
        select_direction = rng.choice(SELECT_DIRECTIONS)

        signature = _candidate_signature(
            R7_RECIPE_ID,
            ticker,
            periods,
            filter_metric,
            filter_operator,
            threshold_statistic,
            percentile_value,
            select_metric,
            select_direction,
            target_metric,
        )
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        try:
            candidate = _build_r7_candidate(
                cube=cube,
                docs_by_name=docs_by_name,
                company_meta=company_meta,
                table_ref_to_path=table_ref_to_path,
                auditor=auditor,
                audit_cache=audit_cache,
                ticker=ticker,
                periods=periods,
                filter_metric=filter_metric,
                filter_operator=filter_operator,
                threshold_statistic=threshold_statistic,
                percentile_value=percentile_value,
                select_metric=select_metric,
                select_direction=select_direction,
                target_metric=target_metric,
            )
        except CandidateRejected as exc:
            logger.debug("R7 candidate reject (coverage/hardness/BUG): %s", exc)
            continue
        except DependencyAuditRejected as exc:
            logger.info("R7 candidate reject (JIT audit): %s", exc.reasons)
            continue
        except EvaluationError as exc:
            logger.debug(
                "R7 candidate rejected during evaluation (tie, non-triviality, or missing coverage): %s",
                exc,
            )
            continue

        yield candidate


RECIPE_ITERATORS = {
    R1_RECIPE_ID: iter_r1_candidates,
    R2_RECIPE_ID: iter_r2_candidates,
    R3_RECIPE_ID: iter_r3_candidates,
    R4_RECIPE_ID: iter_r4_candidates,
    R5_RECIPE_ID: iter_r5_candidates,
    R6_RECIPE_ID: iter_r6_candidates,
    R7_RECIPE_ID: iter_r7_candidates,
}
