
from __future__ import annotations

import random

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.embeddings.base import Embedder
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.common import (
    CandidateTable,
    build_pool_index,
    candidate_table_from_doc,
    report_scope,
    table_retrieval_text,
)
from vifinqa.generation.retrieval.base import RetrievalResult

MIN_ENTITIES = 3
MAX_EXTRA_TABLES = 4
TOP_K_SEARCH = 30


def _pool_docs_for_year(
    all_docs: list[DocumentRef],
    seed_doc: DocumentRef,
    year: str,
    companies: dict[str, CompanyInfo],
) -> list[DocumentRef]:
    seed_company = companies.get(seed_doc.ticker)
    if seed_company is None or not seed_company.industry_l3:
        return []
    seed_scope = report_scope(seed_doc.doc_name)

    def matches(d: DocumentRef) -> bool:
        if d.year != year:
            return False
        other = companies.get(d.ticker)
        return other is not None and other.industry_l3 == seed_company.industry_l3

    return [
        d
        for d in all_docs
        if d.tables_dir is not None and d.text_path is not None and matches(d) and report_scope(d.doc_name) == seed_scope
    ]


def peer_group_same_period_preflight_ok(
    *,
    all_docs: list[DocumentRef],
    seed_doc: DocumentRef,
    seed_table_id: int,
    companies: dict[str, CompanyInfo],
    min_entities: int = MIN_ENTITIES,
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

    pool_docs = _pool_docs_for_year(
        all_docs,
        seed_doc,
        seed_doc.year,
        companies,
    )
    return len({doc.ticker for doc in pool_docs}) >= min_entities


def retrieve_peer_group_same_period_candidates(
    *,
    all_docs: list[DocumentRef],
    seed_doc: DocumentRef,
    seed_table_id: int,
    embedder: Embedder,
    companies: dict[str, CompanyInfo],
    rng: random.Random,
    min_entities: int = MIN_ENTITIES,
    max_extra_tables: int = MAX_EXTRA_TABLES,
    top_k_search: int = TOP_K_SEARCH,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
) -> RetrievalResult:
    from vifinqa.generation.intermediate import _pool_docs, _select_diverse_hits

    seed_table = load_table(
        seed_doc.table_csv_path(seed_table_id),
        ticker=seed_doc.ticker,
        year=seed_doc.year,
        doc_name=seed_doc.doc_name,
        table_id=seed_table_id,
    )
    if not is_table_eligible(seed_table):
        return None
    seed_company = companies.get(seed_doc.ticker)
    if seed_company is None or not seed_company.industry_l3:
        return None

    seed_document = parse_document(seed_doc.text_path)  # type: ignore[arg-type]
    seed_context = seed_document.table_context(seed_table_id, before=0, after=0)
    seed_text = table_retrieval_text(seed_table, seed_context)

    pool_docs = _pool_docs(all_docs, seed_doc, "multi_company_same_year", companies)
    if not pool_docs:
        return None
    index, lookup = build_pool_index(pool_docs, embedder, rng=rng)
    if not lookup:
        return None
    hits = index.search(seed_text, top_k=top_k_search)
    if not hits:
        return None
    diverse = _select_diverse_hits(
        hits,
        mode="multi_company_same_year",
        seed_key=seed_doc.ticker,
        min_distinct=min_entities,
        max_extra=max_extra_tables,
    )
    if not diverse:
        return None

    candidates: list[CandidateTable] = [
        candidate_table_from_doc(
            seed_doc,
            seed_table_id,
            company_name=seed_company.name,
            context_before=context_pages_before,
            context_after=context_pages_after,
        )
    ]
    seen_tickers = {seed_doc.ticker}
    for indexed_table in diverse:
        if indexed_table.ticker in seen_tickers:
            continue
        ref = f"{indexed_table.doc_name}|table_{indexed_table.table_id}"
        hit_doc, _hit_table = lookup[ref]
        hit_company = companies.get(hit_doc.ticker)
        candidates.append(
            candidate_table_from_doc(
                hit_doc,
                indexed_table.table_id,
                company_name=hit_company.name if hit_company else hit_doc.ticker,
                context_before=context_pages_before,
                context_after=context_pages_after,
            )
        )
        seen_tickers.add(indexed_table.ticker)

    return candidates


def retrieve_peer_group_two_periods_candidates(
    *,
    all_docs: list[DocumentRef],
    seed_doc: DocumentRef,
    seed_table_id: int,
    embedder: Embedder,
    companies: dict[str, CompanyInfo],
    rng: random.Random,
    min_entities: int = MIN_ENTITIES,
    max_extra_tables: int = MAX_EXTRA_TABLES,
    top_k_search: int = TOP_K_SEARCH,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
) -> RetrievalResult:
    # Keep period handling explicit and deterministic.
    from vifinqa.generation.intermediate import _pool_docs, _select_diverse_hits

    seed_table = load_table(
        seed_doc.table_csv_path(seed_table_id),
        ticker=seed_doc.ticker,
        year=seed_doc.year,
        doc_name=seed_doc.doc_name,
        table_id=seed_table_id,
    )
    if not is_table_eligible(seed_table):
        return None
    seed_company = companies.get(seed_doc.ticker)
    if seed_company is None or not seed_company.industry_l3:
        return None

    seed_document = parse_document(seed_doc.text_path)  # type: ignore[arg-type]
    seed_context = seed_document.table_context(seed_table_id, before=0, after=0)
    seed_text = table_retrieval_text(seed_table, seed_context)

    # Keep period handling explicit and deterministic.
    pool_docs_a = _pool_docs(all_docs, seed_doc, "multi_company_same_year", companies)
    if not pool_docs_a:
        return None
    index_a, lookup_a = build_pool_index(pool_docs_a, embedder, rng=rng)
    if not lookup_a:
        return None
    hits_a = index_a.search(seed_text, top_k=top_k_search)
    if not hits_a:
        return None
    diverse_a = _select_diverse_hits(
        hits_a,
        mode="multi_company_same_year",
        seed_key=seed_doc.ticker,
        min_distinct=min_entities,
        max_extra=max_extra_tables,
    )
    if not diverse_a:
        return None

    table_by_ticker_a: dict[str, tuple[DocumentRef, int]] = {seed_doc.ticker: (seed_doc, seed_table_id)}
    ordered_tickers_a: list[str] = [seed_doc.ticker]
    for hit in diverse_a:
        if hit.ticker in table_by_ticker_a:
            continue
        ref = f"{hit.doc_name}|table_{hit.table_id}"
        hit_doc, _hit_table = lookup_a[ref]
        table_by_ticker_a[hit.ticker] = (hit_doc, hit.table_id)
        ordered_tickers_a.append(hit.ticker)

    # Keep period handling explicit and deterministic.
    seed_year_int = int(seed_doc.year)
    candidate_years = sorted(
        {
            d.year
            for d in all_docs
            if d.year != seed_doc.year
            and report_scope(d.doc_name) == report_scope(seed_doc.doc_name)
            and (other := companies.get(d.ticker)) is not None
            and other.industry_l3 == seed_company.industry_l3
        },
        key=lambda y: abs(int(y) - seed_year_int),
    )

    for year_b in candidate_years:
        pool_docs_b = _pool_docs_for_year(all_docs, seed_doc, year_b, companies)
        if not pool_docs_b:
            continue
        index_b, lookup_b = build_pool_index(pool_docs_b, embedder, rng=rng)
        if not lookup_b:
            continue
        hits_b = index_b.search(seed_text, top_k=top_k_search)
        if not hits_b:
            continue

        table_by_ticker_b: dict[str, tuple[DocumentRef, int]] = {}
        for indexed_table, _score in hits_b:
            if indexed_table.ticker in table_by_ticker_b:
                continue
            ref = f"{indexed_table.doc_name}|table_{indexed_table.table_id}"
            hit_doc, _hit_table = lookup_b[ref]
            table_by_ticker_b[indexed_table.ticker] = (hit_doc, indexed_table.table_id)

        common_tickers = [tk for tk in ordered_tickers_a if tk in table_by_ticker_b]
        if len(common_tickers) < min_entities:
            continue

        selected_tickers = common_tickers[: max_extra_tables + 1]
        candidates: list[CandidateTable] = []
        for ticker in selected_tickers:
            doc_a, table_id_a = table_by_ticker_a[ticker]
            doc_b, table_id_b = table_by_ticker_b[ticker]
            company = companies.get(ticker)
            company_name = company.name if company else ticker
            candidates.append(
                candidate_table_from_doc(
                    doc_a,
                    table_id_a,
                    company_name=company_name,
                    context_before=context_pages_before,
                    context_after=context_pages_after,
                )
            )
            candidates.append(
                candidate_table_from_doc(
                    doc_b,
                    table_id_b,
                    company_name=company_name,
                    context_before=context_pages_before,
                    context_after=context_pages_after,
                )
            )
        return candidates

    return None
