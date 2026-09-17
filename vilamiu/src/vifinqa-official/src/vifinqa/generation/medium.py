
from __future__ import annotations

import logging
import random
from functools import partial
from pathlib import Path
from typing import Literal

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import DocumentRef, scan_catalog
from vifinqa.common.corpus.company_meta import CompanyInfo, load_company_meta
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.embeddings.base import Embedder
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.common import (
    CandidateTable,
    ChainPromptBuilders,
    build_candidate_table,
    build_pool_index,
    candidate_table_from_doc,
    report_scope,
    run_prompt_chain,
    table_retrieval_text,
)
from vifinqa.generation.parallel import DEFAULT_MAX_WORKERS, run_parallel_generation
from vifinqa.generation.prompts.medium import (
    MediumCase,
    build_concept_prompt,
    build_mapping_prompt,
    build_query_prompt,
    build_question_prompt,
)
from vifinqa.generation.scenarios import ScenarioSpec
from vifinqa.generation.schemas import Difficulty, QARecord
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.output.writer import JsonlWriter

logger = logging.getLogger(__name__)

CrossDocMode = Literal["same_company_diff_year", "same_year_diff_company"]
MediumMode = Literal["same_doc", "same_company_diff_year", "same_year_diff_company"]
_AUTO_MODES: tuple[MediumMode, ...] = ("same_doc", "same_company_diff_year", "same_year_diff_company")

def _chain_builders(case: MediumCase) -> ChainPromptBuilders:
    return ChainPromptBuilders(
        concept=partial(build_concept_prompt, case=case),
        mapping=build_mapping_prompt,
        question=build_question_prompt,
        query=build_query_prompt,
    )


def _chain_builders_scenario(case: MediumCase, scenario: ScenarioSpec) -> ChainPromptBuilders:
    return ChainPromptBuilders(
        concept=partial(build_concept_prompt, case=case, scenario=scenario),
        mapping=partial(build_mapping_prompt, scenario=scenario),
        question=partial(build_question_prompt, scenario=scenario),
        query=partial(build_query_prompt, scenario=scenario),
    )


def _eligible_table_ids(doc: DocumentRef) -> list[int]:
    ids = []
    for tid in doc.table_ids:
        table = load_table(doc.table_csv_path(tid), ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=tid)
        if is_table_eligible(table):
            ids.append(tid)
    return sorted(ids)


def _split_into_groups(ids: list[int], n_groups: int = 3) -> list[list[int]]:
    if not ids:
        return []
    size = max(1, -(-len(ids) // n_groups))  # ceil division
    return [ids[i : i + size] for i in range(0, len(ids), size)]


def _sample_region_spanning(ids: list[int], rng: random.Random, *, per_region_cap: int = 3, n_groups: int = 3) -> list[int]:
    sampled: list[int] = []
    for group in _split_into_groups(ids, n_groups=n_groups):
        take = min(per_region_cap, len(group))
        sampled.extend(rng.sample(group, take))
    return sampled


def generate_medium_same_doc(
    *,
    settings: Settings,
    llm: ChatLLM,
    count: int,
    out_path: Path,
    seed: int | None = None,
    per_region_cap: int = 3,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
    difficulty: Difficulty = "medium",
    min_tables: int = 2,
    scenario: ScenarioSpec | None = None,
    builders: ChainPromptBuilders | None = None,
) -> int:
    rng = random.Random(seed)
    docs = [
        d
        for d in scan_catalog(settings.data_root)
        if d.tables_dir is not None and d.text_path is not None and len(d.table_ids) >= 2
    ]
    rng.shuffle(docs)
    companies = load_company_meta(settings.company_meta_path)
    writer = JsonlWriter(out_path)

    def _build_record(doc: DocumentRef) -> QARecord | None:
        eligible_ids = _eligible_table_ids(doc)
        if len(eligible_ids) < 2:
            return None

        window = _sample_region_spanning(eligible_ids, rng, per_region_cap=per_region_cap)
        if len(window) < 2:
            return None

        document = parse_document(doc.text_path)  # type: ignore[arg-type]
        company = companies.get(doc.ticker)
        company_name = company.name if company else doc.ticker

        candidates: list[CandidateTable] = []
        for table_id in window:
            table = load_table(
                doc.table_csv_path(table_id), ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=table_id
            )
            candidates.append(
                build_candidate_table(
                    doc=doc,
                    table=table,
                    document=document,
                    company_name=company_name,
                    context_before=context_pages_before,
                    context_after=context_pages_after,
                )
            )

        chain_builders = builders if builders is not None else _chain_builders("same_doc")
        return run_prompt_chain(
            llm=llm,
            candidates=candidates,
            builders=chain_builders,
            record_id=0,
            difficulty=difficulty,
            min_tables=min_tables,
            scenario=scenario,
        )

    generated = run_parallel_generation(
        work_items=docs,
        build_record=_build_record,
        count=count,
        writer=writer,
        max_workers=max_workers,
        max_candidates=max_candidates,
    )

    if generated < count:
        logger.warning("Generated only %d/%d Medium questions (same-doc).", generated, count)
    return generated


def _cross_doc_pool(
    all_docs: list[DocumentRef],
    seed_doc: DocumentRef,
    mode: CrossDocMode,
    companies: dict[str, CompanyInfo],
) -> list[DocumentRef]:
    if mode == "same_company_diff_year":
        matches = lambda d: d.ticker == seed_doc.ticker and d.year != seed_doc.year  # noqa: E731
    else:
        seed_company = companies.get(seed_doc.ticker)
        if seed_company is None or not seed_company.industry_l3:
            return []

        def matches(d: DocumentRef) -> bool:
            if d.year != seed_doc.year or d.ticker == seed_doc.ticker:
                return False
            other = companies.get(d.ticker)
            return other is not None and other.industry_l3 == seed_company.industry_l3
    seed_scope = report_scope(seed_doc.doc_name)

    def scope_matches(doc: DocumentRef) -> bool:
        candidate_scope = report_scope(doc.doc_name)
        if seed_scope == "unknown" or candidate_scope == "unknown":
            return seed_scope == candidate_scope
        return seed_scope == candidate_scope

    return [
        d
        for d in all_docs
        if d.tables_dir is not None and d.text_path is not None and matches(d) and scope_matches(d)
    ]


def generate_medium_cross_doc(
    *,
    settings: Settings,
    llm: ChatLLM,
    embedder: Embedder,
    count: int,
    out_path: Path,
    seed: int | None = None,
    mode: CrossDocMode = "same_company_diff_year",
    top_k: int = 5,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
    scenario: ScenarioSpec | None = None,
) -> int:
    rng = random.Random(seed)
    all_docs = scan_catalog(settings.data_root)
    seed_candidates = [
        (d, tid) for d in all_docs if d.tables_dir is not None and d.text_path is not None for tid in d.table_ids
    ]
    rng.shuffle(seed_candidates)
    companies = load_company_meta(settings.company_meta_path)
    writer = JsonlWriter(out_path)

    def _build_record(item: tuple[DocumentRef, int]) -> QARecord | None:
        seed_doc, seed_table_id = item
        seed_table = load_table(
            seed_doc.table_csv_path(seed_table_id),
            ticker=seed_doc.ticker,
            year=seed_doc.year,
            doc_name=seed_doc.doc_name,
            table_id=seed_table_id,
        )
        if not is_table_eligible(seed_table):
            return None

        pool_docs = _cross_doc_pool(all_docs, seed_doc, mode, companies)
        if not pool_docs:
            return None

        seed_document = parse_document(seed_doc.text_path)  # type: ignore[arg-type]
        index, pool_lookup = build_pool_index(pool_docs, embedder, rng=rng)
        if not pool_lookup:
            return None

        seed_context = seed_document.table_context(seed_table_id, before=0, after=0)
        hits = index.search(table_retrieval_text(seed_table, seed_context), top_k=top_k)
        if not hits:
            return None

        seed_company = companies.get(seed_doc.ticker)
        candidates = [
            build_candidate_table(
                doc=seed_doc,
                table=seed_table,
                document=seed_document,
                company_name=seed_company.name if seed_company else seed_doc.ticker,
                context_before=context_pages_before,
                context_after=context_pages_after,
            )
        ]
        for indexed_table, _score in hits:
            ref = f"{indexed_table.doc_name}|table_{indexed_table.table_id}"
            hit_doc, _hit_table = pool_lookup[ref]
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

        chain_builders = _chain_builders_scenario(mode, scenario) if scenario is not None else _chain_builders(mode)
        return run_prompt_chain(
            llm=llm,
            candidates=candidates,
            builders=chain_builders,
            record_id=0,
            difficulty="medium",
            scenario=scenario,
        )

    generated = run_parallel_generation(
        work_items=seed_candidates,
        build_record=_build_record,
        count=count,
        writer=writer,
        max_workers=max_workers,
        max_candidates=max_candidates,
    )

    if generated < count:
        logger.warning("Generated only %d/%d Medium questions (cross-doc).", generated, count)
    return generated


def generate_medium(
    *,
    settings: Settings,
    llm: ChatLLM,
    embedder: Embedder,
    count: int,
    out_path: Path,
    seed: int | None = None,
    top_k: int = 5,
    per_region_cap: int = 3,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
) -> int:
    rng = random.Random(seed)
    all_docs = scan_catalog(settings.data_root)
    seed_candidates = [
        (doc, table_id)
        for doc in all_docs
        if doc.tables_dir is not None and doc.text_path is not None
        for table_id in doc.table_ids
    ]
    rng.shuffle(seed_candidates)
    work_items = [
        (doc, table_id, rng.choice(_AUTO_MODES), rng.randrange(2**32))
        for doc, table_id in seed_candidates
    ]
    companies = load_company_meta(settings.company_meta_path)
    writer = JsonlWriter(out_path)

    def _same_doc_candidates(doc: DocumentRef, item_rng: random.Random) -> list[CandidateTable]:
        eligible_ids = _eligible_table_ids(doc)
        if len(eligible_ids) < 2:
            return []
        window = _sample_region_spanning(eligible_ids, item_rng, per_region_cap=per_region_cap)
        if len(window) < 2:
            return []
        document = parse_document(doc.text_path)  # type: ignore[arg-type]
        company = companies.get(doc.ticker)
        return [
            build_candidate_table(
                doc=doc,
                table=load_table(
                    doc.table_csv_path(table_id),
                    ticker=doc.ticker,
                    year=doc.year,
                    doc_name=doc.doc_name,
                    table_id=table_id,
                ),
                document=document,
                company_name=company.name if company else doc.ticker,
                context_before=context_pages_before,
                context_after=context_pages_after,
            )
            for table_id in window
        ]

    def _cross_doc_candidates(
        seed_doc: DocumentRef,
        seed_table_id: int,
        record_mode: CrossDocMode,
        item_rng: random.Random,
    ) -> list[CandidateTable]:
        seed_table = load_table(
            seed_doc.table_csv_path(seed_table_id),
            ticker=seed_doc.ticker,
            year=seed_doc.year,
            doc_name=seed_doc.doc_name,
            table_id=seed_table_id,
        )
        if not is_table_eligible(seed_table):
            return []
        pool_docs = _cross_doc_pool(all_docs, seed_doc, record_mode, companies)
        if not pool_docs:
            return []
        seed_document = parse_document(seed_doc.text_path)  # type: ignore[arg-type]
        index, pool_lookup = build_pool_index(pool_docs, embedder, rng=item_rng)
        seed_context = seed_document.table_context(seed_table_id, before=0, after=0)
        hits = index.search(table_retrieval_text(seed_table, seed_context), top_k=top_k)
        if not hits:
            return []
        seed_company = companies.get(seed_doc.ticker)
        candidates = [
            build_candidate_table(
                doc=seed_doc,
                table=seed_table,
                document=seed_document,
                company_name=seed_company.name if seed_company else seed_doc.ticker,
                context_before=context_pages_before,
                context_after=context_pages_after,
            )
        ]
        for indexed_table, _score in hits:
            ref = f"{indexed_table.doc_name}|table_{indexed_table.table_id}"
            hit_doc, _hit_table = pool_lookup[ref]
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
        return candidates

    def _build_record(item: tuple[DocumentRef, int, MediumMode, int]) -> QARecord | None:
        seed_doc, seed_table_id, record_mode, item_seed = item
        item_rng = random.Random(item_seed)
        if record_mode == "same_doc":
            candidates = _same_doc_candidates(seed_doc, item_rng)
        else:
            candidates = _cross_doc_candidates(seed_doc, seed_table_id, record_mode, item_rng)
        if len(candidates) < 2:
            return None
        return run_prompt_chain(
            llm=llm,
            candidates=candidates,
            builders=_chain_builders(record_mode),
            record_id=0,
            difficulty="medium",
        )

    generated = run_parallel_generation(
        work_items=work_items,
        build_record=_build_record,
        count=count,
        writer=writer,
        max_workers=max_workers,
        max_candidates=max_candidates,
    )
    if generated < count:
        logger.warning("Generated only %d/%d Medium questions (auto).", generated, count)
    return generated
