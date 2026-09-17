
from __future__ import annotations

import itertools
import random
import re
from dataclasses import dataclass
from typing import Literal

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.table_index.base import TableIndexStore
from vifinqa.generation.common import CandidateTable, candidate_table_from_doc
from vifinqa.generation.retrieval.time_series import enumerate_time_series_windows

DEFAULT_PERIOD_COUNT = 3
DEFAULT_MIN_ENTITIES = 3
DEFAULT_MAX_TABLES_PER_CELL = 3
DEFAULT_SEMANTIC_POOL_SIZE = 30
DEFAULT_MAX_TRIPLES_PER_GROUP = 5

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


@dataclass(frozen=True, slots=True)
class PeerLongitudinalGroup:

    industry_l3: str
    report_scope: Literal["consolidated", "parent"]
    periods: tuple[str, str, str]
    eligible_tickers: tuple[str, ...]
    docs_by_ticker_period: dict[tuple[str, str], tuple[DocumentRef, ...]]


def enumerate_peer_longitudinal_groups(
    all_docs: list[DocumentRef],
    companies: dict[str, CompanyInfo],
    *,
    period_count: int = DEFAULT_PERIOD_COUNT,
    min_entities: int = DEFAULT_MIN_ENTITIES,
) -> list[PeerLongitudinalGroup]:
    windows = enumerate_time_series_windows(all_docs, period_count=period_count)
    by_group: dict[tuple[str, str, tuple[str, ...]], list] = {}
    for window in windows:
        company = companies.get(window.ticker)
        if company is None or not company.industry_l3:
            continue
        key = (company.industry_l3, window.report_scope, window.periods)
        by_group.setdefault(key, []).append(window)

    groups: list[PeerLongitudinalGroup] = []
    for (industry, scope, periods), ts_windows in sorted(by_group.items()):
        tickers = sorted({w.ticker for w in ts_windows})
        if len(tickers) < min_entities:
            continue
        docs_by_ticker_period: dict[tuple[str, str], tuple[DocumentRef, ...]] = {}
        for w in ts_windows:
            for period in periods:
                docs_by_ticker_period[(w.ticker, period)] = w.docs_by_period[period]
        groups.append(
            PeerLongitudinalGroup(
                industry_l3=industry,
                report_scope=scope,  # type: ignore[arg-type]
                periods=periods,  # type: ignore[arg-type]
                eligible_tickers=tuple(tickers),
                docs_by_ticker_period=docs_by_ticker_period,
            )
        )
    return groups


def iter_ticker_triples(
    eligible_tickers: tuple[str, ...],
    *,
    rng: random.Random,
    max_triples: int = DEFAULT_MAX_TRIPLES_PER_GROUP,
) -> list[tuple[str, str, str]]:
    all_triples = [tuple(sorted(triple)) for triple in itertools.combinations(eligible_tickers, 3)]
    rng.shuffle(all_triples)
    return all_triples[:max_triples]


@dataclass(frozen=True, slots=True)
class PeerLongitudinalBundle:

    industry_l3: str
    report_scope: Literal["consolidated", "parent"]
    periods: tuple[str, str, str]
    tickers: tuple[str, str, str]
    candidates_by_cell: dict[tuple[str, str], list[CandidateTable]]
    min_score: float
    avg_score: float


def retrieve_peer_longitudinal_bundle(
    *,
    group: PeerLongitudinalGroup,
    tickers: tuple[str, str, str],
    concept_names: tuple[str, ...],
    index_store: TableIndexStore,
    companies: dict[str, CompanyInfo],
    max_tables_per_cell: int = DEFAULT_MAX_TABLES_PER_CELL,
    semantic_pool_size: int = DEFAULT_SEMANTIC_POOL_SIZE,
) -> PeerLongitudinalBundle | None:
    query = ". ".join(concept_names)
    normalized_concepts = {_normalize(name) for name in concept_names}
    candidates_by_cell: dict[tuple[str, str], list[CandidateTable]] = {}
    top_scores: list[float] = []

    for ticker in tickers:
        for period in group.periods:
            docs = list(group.docs_by_ticker_period.get((ticker, period), ()))
            if not docs:
                return None
            index = index_store.get(
                ticker=ticker, report_scope=group.report_scope, period=period, docs=docs
            )
            hits = index.search(query, top_k=semantic_pool_size)
            if not hits:
                return None
            ranked = sorted(
                hits,
                key=lambda hit: (
                    any(concept in _normalize(hit[0].text) for concept in normalized_concepts),
                    hit[1],
                ),
                reverse=True,
            )
            doc_by_name = {doc.doc_name: doc for doc in docs}
            company = companies.get(ticker)
            cell_candidates: list[CandidateTable] = []
            for indexed, _score in ranked[:max_tables_per_cell]:
                doc = doc_by_name.get(indexed.doc_name)
                if doc is None:
                    continue
                cell_candidates.append(
                    candidate_table_from_doc(
                        doc,
                        indexed.table_id,
                        company_name=company.name if company else ticker,
                        context_before=0,
                        context_after=0,
                    )
                )
            if not cell_candidates:
                return None
            candidates_by_cell[(ticker, period)] = cell_candidates
            top_scores.append(ranked[0][1])

    return PeerLongitudinalBundle(
        industry_l3=group.industry_l3,
        report_scope=group.report_scope,
        periods=group.periods,
        tickers=tickers,
        candidates_by_cell=candidates_by_cell,
        min_score=min(top_scores),
        avg_score=sum(top_scores) / len(top_scores),
    )


def select_best_bundle(
    bundles: list[PeerLongitudinalBundle],
) -> PeerLongitudinalBundle | None:
    if not bundles:
        return None
    return max(bundles, key=lambda b: (b.min_score, b.avg_score))
