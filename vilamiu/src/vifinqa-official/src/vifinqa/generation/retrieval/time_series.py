
from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.embeddings.base import Embedder
from vifinqa.generation.embedding_index import IndexedTable
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.common import (
    CandidateTable,
    build_pool_index,
    candidate_table_from_doc,
    report_scope,
    table_retrieval_text,
)
from vifinqa.generation.retrieval.base import RetrievalResult

DEFAULT_PERIOD_COUNT = 3
TOP_K_PER_YEAR = 3


def _time_series_pool_docs(all_docs: list[DocumentRef], seed_doc: DocumentRef) -> dict[str, list[DocumentRef]]:
    seed_scope = report_scope(seed_doc.doc_name)
    by_year: dict[str, list[DocumentRef]] = {}
    for d in all_docs:
        if d.tables_dir is None or d.text_path is None:
            continue
        if d.ticker != seed_doc.ticker or report_scope(d.doc_name) != seed_scope:
            continue
        by_year.setdefault(d.year, []).append(d)
    return by_year


def _build_year_hits(
    by_year: dict[str, list[DocumentRef]],
    seed_table_text: str,
    embedder: Embedder,
    rng: random.Random,
    *,
    top_k_per_year: int = TOP_K_PER_YEAR,
) -> dict[str, list[tuple[IndexedTable, float]]]:
    hits_by_year: dict[str, list[tuple[IndexedTable, float]]] = {}
    for year, docs in by_year.items():
        index, lookup = build_pool_index(docs, embedder, rng=rng)
        if not lookup:
            continue
        hits = index.search(seed_table_text, top_k=top_k_per_year)
        if hits:
            hits_by_year[year] = hits
    return hits_by_year


def find_consecutive_windows(
    available_years: list[str], seed_year: str, period_count: int = DEFAULT_PERIOD_COUNT
) -> list[tuple[str, ...]]:
    available = set(available_years)
    seed = int(seed_year)
    windows: list[tuple[str, ...]] = []
    for start in range(seed - period_count + 1, seed + 1):
        years = [start + i for i in range(period_count)]
        year_strs = tuple(str(y) for y in years)
        if all(y in available for y in year_strs):
            windows.append(year_strs)
    return windows


def score_time_series_window(
    window: tuple[str, ...],
    hits_by_year: dict[str, list[tuple[IndexedTable, float]]],
    seed_year: str,
) -> tuple[float, float, float]:
    sims = [hits_by_year[year][0][1] for year in window]
    min_sim = min(sims)
    avg_sim = sum(sims) / len(sims)
    seed = int(seed_year)
    bounds = [int(window[0]), int(window[-1])]
    distance = min(abs(bounds[0] - seed), abs(bounds[1] - seed))
    return (min_sim, avg_sim, -float(distance))


def select_best_time_series_window(
    windows: list[tuple[str, ...]],
    hits_by_year: dict[str, list[tuple[IndexedTable, float]]],
    seed_year: str,
) -> tuple[str, ...] | None:
    if not windows:
        return None
    return max(windows, key=lambda w: score_time_series_window(w, hits_by_year, seed_year))


def time_series_preflight_ok(
    *,
    all_docs: list[DocumentRef],
    seed_doc: DocumentRef,
    seed_table_id: int,
    period_count: int = DEFAULT_PERIOD_COUNT,
) -> bool:
    seed_table = load_table(
        seed_doc.table_csv_path(seed_table_id),
        ticker=seed_doc.ticker,
        year=seed_doc.year,
        doc_name=seed_doc.doc_name,
        table_id=seed_table_id,
    )
    if not is_table_eligible(seed_table):
        return False
    by_year = _time_series_pool_docs(all_docs, seed_doc)
    if seed_doc.year not in by_year:
        return False
    return bool(find_consecutive_windows(list(by_year), seed_doc.year, period_count))


def retrieve_time_series_candidates(
    *,
    all_docs: list[DocumentRef],
    seed_doc: DocumentRef,
    seed_table_id: int,
    embedder: Embedder,
    companies: dict[str, CompanyInfo],
    rng: random.Random,
    period_count: int = DEFAULT_PERIOD_COUNT,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
) -> RetrievalResult:
    seed_table = load_table(
        seed_doc.table_csv_path(seed_table_id),
        ticker=seed_doc.ticker,
        year=seed_doc.year,
        doc_name=seed_doc.doc_name,
        table_id=seed_table_id,
    )
    if not is_table_eligible(seed_table):
        return None

    by_year = _time_series_pool_docs(all_docs, seed_doc)
    if seed_doc.year not in by_year:
        return None

    seed_document = parse_document(seed_doc.text_path)  # type: ignore[arg-type]
    seed_context = seed_document.table_context(seed_table_id, before=0, after=0)
    seed_text = table_retrieval_text(seed_table, seed_context)

    hits_by_year = _build_year_hits(by_year, seed_text, embedder, rng)
    if seed_doc.year not in hits_by_year:
        return None

    windows = find_consecutive_windows(list(hits_by_year), seed_doc.year, period_count)
    window = select_best_time_series_window(windows, hits_by_year, seed_doc.year)
    if window is None:
        return None

    doc_by_name = {d.doc_name: d for docs in by_year.values() for d in docs}
    candidates: list[CandidateTable] = []
    for year in window:
        if year == seed_doc.year:
            hit_doc, hit_table_id = seed_doc, seed_table_id
        else:
            top_hit, _score = hits_by_year[year][0]
            hit_doc, hit_table_id = doc_by_name[top_hit.doc_name], top_hit.table_id
        company = companies.get(hit_doc.ticker)
        candidates.append(
            candidate_table_from_doc(
                hit_doc,
                hit_table_id,
                company_name=company.name if company else hit_doc.ticker,
                context_before=context_pages_before,
                context_after=context_pages_after,
            )
        )
    return candidates


#
# Keep period handling explicit and deterministic.
# Keep LLM behavior within the declared contract.
# Keep LLM behavior within the declared contract.


@dataclass(frozen=True, slots=True)
class TimeSeriesWindow:
    ticker: str
    report_scope: Literal["consolidated", "parent"]
    periods: tuple[str, ...]
    docs_by_period: Mapping[str, tuple[DocumentRef, ...]]
    anchor_period: str


def _pick_anchor_period(periods: tuple[str, ...], docs_by_period: Mapping[str, tuple[DocumentRef, ...]]) -> str:

    def _table_count(period: str) -> int:
        return sum(len(d.table_ids) for d in docs_by_period[period])

    return max(periods, key=lambda p: (_table_count(p), int(p)))


def enumerate_time_series_windows(
    all_docs: list[DocumentRef],
    *,
    period_count: int = DEFAULT_PERIOD_COUNT,
) -> list[TimeSeriesWindow]:
    by_ticker_scope: dict[tuple[str, str], dict[str, list[DocumentRef]]] = {}
    for doc in all_docs:
        if doc.tables_dir is None or doc.text_path is None:
            continue
        scope = report_scope(doc.doc_name)
        if scope not in ("consolidated", "parent"):
            continue
        by_ticker_scope.setdefault((doc.ticker, scope), {}).setdefault(doc.year, []).append(doc)

    windows: list[TimeSeriesWindow] = []
    for (ticker, scope), by_year in sorted(by_ticker_scope.items()):
        available_years = set(by_year)
        start_years = sorted(int(y) for y in by_year)
        for start in start_years:
            candidate_periods = tuple(str(start + i) for i in range(period_count))
            if not all(y in available_years for y in candidate_periods):
                continue
            docs_by_period = {p: tuple(by_year[p]) for p in candidate_periods}
            windows.append(
                TimeSeriesWindow(
                    ticker=ticker,
                    report_scope=scope,  # type: ignore[arg-type]
                    periods=candidate_periods,
                    docs_by_period=docs_by_period,
                    anchor_period=_pick_anchor_period(candidate_periods, docs_by_period),
                )
            )
    return windows
