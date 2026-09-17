"""Tier 3: apply hard candidate filters, then retrieve for entity/period diversity.
"""

from __future__ import annotations

import logging
import random
from functools import partial
from pathlib import Path
from threading import Lock
from typing import Literal

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import DocumentRef, scan_catalog
from vifinqa.common.corpus.company_meta import CompanyInfo, load_company_meta
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.embeddings.base import Embedder
from vifinqa.generation.embedding_index import IndexedTable
from vifinqa.generation.table_index.base import TableIndexStore
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.common import (
    CandidateTable,
    ChainPromptBuilders,
    build_candidate_table,
    build_pool_index,
    call_structured,
    candidate_table_from_doc,
    judge_and_maybe_rewrite,
    load_table_for_candidate,
    question_style_error,
    report_scope,
    run_prompt_chain,
    table_retrieval_text,
)
from vifinqa.generation.medium import generate_medium_same_doc
from vifinqa.generation.intermediate_compiler import compile_formula_query, compile_longitudinal_query
from vifinqa.generation.intermediate_formulas.base import FormulaScenarioPlan, ResolvedCell
from vifinqa.generation.intermediate_formulas.longitudinal import (
    LongitudinalScenarioPlan,
    enabled_input_metrics,
    enabled_transforms,
    get_input_metric,
    get_reducer_ids,
    get_transform,
)
from vifinqa.generation.intermediate_formulas.registry import enabled_formulas
from vifinqa.generation.intermediate_gates import (
    VnNumberError,
    formula_answer_finite_error,
    formula_denominator_positive_error,
    formula_industry_policy_error,
    formula_question_alignment_error,
    formula_report_scope_error,
    formula_role_coverage_error,
    formula_scale_consistency_error,
    formula_sign_policy_error,
    longitudinal_coverage_error,
    longitudinal_near_zero_base_error,
    longitudinal_question_alignment_error,
    longitudinal_scale_consistency_error,
    parse_vn_number,
    resolve_role_cell,
)
from vifinqa.generation.parallel import DEFAULT_MAX_WORKERS, run_parallel_generation
from vifinqa.generation.prompts.intermediate import (
    IntermediateMode,
    build_concept_prompt,
    build_mapping_prompt,
    build_query_prompt,
    build_question_prompt,
    build_same_doc_multi_concept_prompt,
    build_same_doc_multi_question_prompt,
)
from vifinqa.generation.prompts.intermediate_formula import (
    build_formula_finance_judge_prompt,
    build_formula_question_prompt,
    build_formula_role_mapping_prompt,
)
from vifinqa.generation.prompts.intermediate_longitudinal import (
    build_longitudinal_cell_mapping_prompt,
    build_longitudinal_finance_judge_prompt,
    build_longitudinal_question_prompt,
    cell_role_id,
)
from vifinqa.generation.retrieval.peer_group import retrieve_peer_group_two_periods_candidates
from vifinqa.generation.retrieval.peer_group_longitudinal import (
    DEFAULT_MAX_TRIPLES_PER_GROUP,
    PeerLongitudinalGroup,
    enumerate_peer_longitudinal_groups,
    iter_ticker_triples,
    retrieve_peer_longitudinal_bundle,
)
from vifinqa.generation.retrieval.same_doc_roles import DocumentRoleRequest, retrieve_document_roles
from vifinqa.generation.retrieval.time_series import (
    DEFAULT_PERIOD_COUNT,
    retrieve_time_series_candidates,
)
from vifinqa.generation.scenarios import ScenarioSpec
from vifinqa.generation.schemas import FinancialValidityJudgment, FormulaMappingResult, QARecord, QuestionDraft
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.output.writer import JsonlWriter
from vifinqa.generation.validation.pandas_check import execute_query

logger = logging.getLogger(__name__)

MIN_DISTINCT = 3
MAX_EXTRA_TABLES = 4  # Additional tables beyond the seed.
TOP_K_SEARCH = 30
AutoIntermediateMode = Literal["auto", "multi_company_same_year", "multi_year_same_company"]
_AUTO_MODES: tuple[IntermediateMode, ...] = ("multi_company_same_year", "multi_year_same_company")
IntermediateWorkItem = tuple[DocumentRef, int, IntermediateMode, int]


def _diversity_key(mode: IntermediateMode, table: IndexedTable) -> str:
    return table.ticker if mode == "multi_company_same_year" else table.year


def _pool_docs(
    all_docs: list[DocumentRef],
    seed_doc: DocumentRef,
    mode: IntermediateMode,
    companies: dict[str, CompanyInfo],
) -> list[DocumentRef]:
    if mode == "multi_company_same_year":
        seed_company = companies.get(seed_doc.ticker)
        if seed_company is None or not seed_company.industry_l3:
            return []

        def matches(d: DocumentRef) -> bool:
            if d.year != seed_doc.year or d.ticker == seed_doc.ticker:
                return False
            other = companies.get(d.ticker)
            return other is not None and other.industry_l3 == seed_company.industry_l3
    else:
        matches = lambda d: d.ticker == seed_doc.ticker and d.year != seed_doc.year  # noqa: E731
    seed_scope = report_scope(seed_doc.doc_name)
    return [
        d
        for d in all_docs
        if d.tables_dir is not None
        and d.text_path is not None
        and matches(d)
        and report_scope(d.doc_name) == seed_scope
    ]


def _select_diverse_hits(
    hits: list[tuple[IndexedTable, float]],
    *,
    mode: IntermediateMode,
    seed_key: str,
    min_distinct: int,
    max_extra: int,
) -> list[IndexedTable]:
    selected: list[IndexedTable] = []
    seen_keys = {seed_key}
    for indexed_table, _score in hits:
        key = _diversity_key(mode, indexed_table)
        if key in seen_keys:
            continue
        selected.append(indexed_table)
        seen_keys.add(key)
        if len(selected) >= max_extra:
            break
    if len(seen_keys) < min_distinct:
        return []
    return selected


def _preflight_work_items(
    work_items: list[IntermediateWorkItem],
    *,
    all_docs: list[DocumentRef],
    companies: dict[str, CompanyInfo],
    min_distinct: int,
):
    for item in work_items:
        seed_doc, seed_table_id, record_mode, _item_seed = item
        seed_table = load_table(
            seed_doc.table_csv_path(seed_table_id),
            ticker=seed_doc.ticker,
            year=seed_doc.year,
            doc_name=seed_doc.doc_name,
            table_id=seed_table_id,
        )
        if not is_table_eligible(seed_table):
            continue

        pool_docs = _pool_docs(all_docs, seed_doc, record_mode, companies)
        seed_key = seed_doc.ticker if record_mode == "multi_company_same_year" else seed_doc.year
        distinct_pool_keys = {
            doc.ticker if record_mode == "multi_company_same_year" else doc.year for doc in pool_docs
        }
        if len(distinct_pool_keys | {seed_key}) < min_distinct:
            continue
        yield item


def generate_intermediate(
    *,
    settings: Settings,
    llm: ChatLLM,
    embedder: Embedder,
    count: int,
    out_path: Path,
    seed: int | None = None,
    mode: AutoIntermediateMode = "auto",
    min_distinct: int = MIN_DISTINCT,
    max_extra_tables: int = MAX_EXTRA_TABLES,
    top_k_search: int = TOP_K_SEARCH,
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
    raw_work_items = [
        (doc, table_id, rng.choice(_AUTO_MODES) if mode == "auto" else mode, rng.randrange(2**32))
        for doc, table_id in seed_candidates
    ]
    companies = load_company_meta(settings.company_meta_path)
    work_items = _preflight_work_items(
        raw_work_items,
        all_docs=all_docs,
        companies=companies,
        min_distinct=min_distinct,
    )
    writer = JsonlWriter(out_path)

    def _build_record(item: IntermediateWorkItem) -> QARecord | None:
        seed_doc, seed_table_id, record_mode, item_seed = item
        item_rng = random.Random(item_seed)
        diversity_key = (
            (lambda c: c.ticker) if record_mode == "multi_company_same_year" else (lambda c: c.year)
        )
        builders = ChainPromptBuilders(
            concept=partial(build_concept_prompt, mode=record_mode, scenario=scenario),
            mapping=partial(build_mapping_prompt, scenario=scenario),
            question=partial(build_question_prompt, mode=record_mode, scenario=scenario),
            query=partial(build_query_prompt, scenario=scenario),
        )
        seed_table = load_table(
            seed_doc.table_csv_path(seed_table_id),
            ticker=seed_doc.ticker,
            year=seed_doc.year,
            doc_name=seed_doc.doc_name,
            table_id=seed_table_id,
        )
        if not is_table_eligible(seed_table):
            return None

        pool_docs = _pool_docs(all_docs, seed_doc, record_mode, companies)
        if not pool_docs:
            return None

        seed_document = parse_document(seed_doc.text_path)  # type: ignore[arg-type]
        index, pool_lookup = build_pool_index(pool_docs, embedder, rng=item_rng)
        if not pool_lookup:
            return None

        seed_context = seed_document.table_context(seed_table_id, before=0, after=0)
        hits = index.search(table_retrieval_text(seed_table, seed_context), top_k=top_k_search)
        if not hits:
            return None

        seed_key = seed_doc.ticker if record_mode == "multi_company_same_year" else seed_doc.year
        diverse = _select_diverse_hits(
            hits,
            mode=record_mode,
            seed_key=seed_key,
            min_distinct=min_distinct,
            max_extra=max_extra_tables,
        )
        if not diverse:
            return None

        seed_company = companies.get(seed_doc.ticker)
        candidates: list[CandidateTable] = [
            build_candidate_table(
                doc=seed_doc,
                table=seed_table,
                document=seed_document,
                company_name=seed_company.name if seed_company else seed_doc.ticker,
                context_before=context_pages_before,
                context_after=context_pages_after,
            )
        ]
        for indexed_table in diverse:
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

        return run_prompt_chain(
            llm=llm,
            candidates=candidates,
            builders=builders,
            record_id=0,
            difficulty="intermediate",
            min_tables=min_distinct,
            diversity_key=diversity_key,
            min_distinct=min_distinct,
            max_extra_tables=max_extra_tables,
            scenario=scenario,
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
        logger.warning("Generated only %d/%d Intermediate questions.", generated, count)
    return generated


TimeSeriesWorkItem = tuple[DocumentRef, int, int]


def generate_time_series_intermediate(
    *,
    settings: Settings,
    llm: ChatLLM,
    embedder: Embedder,
    count: int,
    out_path: Path,
    seed: int | None = None,
    scenario: ScenarioSpec,
    period_count: int = DEFAULT_PERIOD_COUNT,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
) -> int:
    rng = random.Random(seed)
    all_docs = scan_catalog(settings.data_root)
    seed_candidates = [
        (d, tid) for d in all_docs if d.tables_dir is not None and d.text_path is not None for tid in d.table_ids
    ]
    rng.shuffle(seed_candidates)
    work_items: list[TimeSeriesWorkItem] = [
        (doc, table_id, rng.randrange(2**32)) for doc, table_id in seed_candidates
    ]
    companies = load_company_meta(settings.company_meta_path)
    writer = JsonlWriter(out_path)

    def _build_record(item: TimeSeriesWorkItem) -> QARecord | None:
        seed_doc, seed_table_id, item_seed = item
        item_rng = random.Random(item_seed)
        candidates = retrieve_time_series_candidates(
            all_docs=all_docs,
            seed_doc=seed_doc,
            seed_table_id=seed_table_id,
            embedder=embedder,
            companies=companies,
            rng=item_rng,
            period_count=period_count,
            context_pages_before=context_pages_before,
            context_pages_after=context_pages_after,
        )
        if candidates is None:
            return None
        builders = ChainPromptBuilders(
            concept=partial(build_concept_prompt, mode="multi_year_same_company", scenario=scenario),
            mapping=partial(build_mapping_prompt, scenario=scenario),
            question=partial(build_question_prompt, mode="multi_year_same_company", scenario=scenario),
            query=partial(build_query_prompt, scenario=scenario),
        )
        return run_prompt_chain(
            llm=llm,
            candidates=candidates,
            builders=builders,
            record_id=0,
            difficulty="intermediate",
            min_tables=scenario.min_observations,
            scenario=scenario,
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
        logger.warning("Generated only %d/%d Intermediate questions (time-series).", generated, count)
    return generated


def generate_peer_group_two_periods(
    *,
    settings: Settings,
    llm: ChatLLM,
    embedder: Embedder,
    count: int,
    out_path: Path,
    seed: int | None = None,
    scenario: ScenarioSpec,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
) -> int:
    rng = random.Random(seed)
    all_docs = scan_catalog(settings.data_root)
    seed_candidates = [
        (d, tid) for d in all_docs if d.tables_dir is not None and d.text_path is not None for tid in d.table_ids
    ]
    rng.shuffle(seed_candidates)
    work_items: list[TimeSeriesWorkItem] = [
        (doc, table_id, rng.randrange(2**32)) for doc, table_id in seed_candidates
    ]
    companies = load_company_meta(settings.company_meta_path)
    writer = JsonlWriter(out_path)

    def _build_record(item: TimeSeriesWorkItem) -> QARecord | None:
        seed_doc, seed_table_id, item_seed = item
        item_rng = random.Random(item_seed)
        candidates = retrieve_peer_group_two_periods_candidates(
            all_docs=all_docs,
            seed_doc=seed_doc,
            seed_table_id=seed_table_id,
            embedder=embedder,
            companies=companies,
            rng=item_rng,
            min_entities=scenario.min_entities,
            context_pages_before=context_pages_before,
            context_pages_after=context_pages_after,
        )
        if candidates is None:
            return None
        builders = ChainPromptBuilders(
            concept=partial(build_concept_prompt, mode="multi_company_same_year", scenario=scenario),
            mapping=partial(build_mapping_prompt, scenario=scenario),
            question=partial(build_question_prompt, mode="multi_company_same_year", scenario=scenario),
            query=partial(build_query_prompt, scenario=scenario),
        )
        return run_prompt_chain(
            llm=llm,
            candidates=candidates,
            builders=builders,
            record_id=0,
            difficulty="intermediate",
            min_tables=scenario.min_observations,
            scenario=scenario,
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
        logger.warning("Generated only %d/%d Intermediate questions (two-period peer group).", generated, count)
    return generated


def generate_same_doc_multi_inputs(
    *,
    settings: Settings,
    llm: ChatLLM,
    count: int,
    out_path: Path,
    seed: int | None = None,
    scenario: ScenarioSpec,
    per_region_cap: int = 3,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
) -> int:
    builders = ChainPromptBuilders(
        concept=partial(build_same_doc_multi_concept_prompt, scenario=scenario),
        mapping=partial(build_mapping_prompt, scenario=scenario),
        question=partial(build_same_doc_multi_question_prompt, scenario=scenario),
        query=partial(build_query_prompt, scenario=scenario),
    )
    return generate_medium_same_doc(
        settings=settings,
        llm=llm,
        count=count,
        out_path=out_path,
        seed=seed,
        per_region_cap=per_region_cap,
        context_pages_before=context_pages_before,
        context_pages_after=context_pages_after,
        max_workers=max_workers,
        max_candidates=max_candidates,
        difficulty="intermediate",
        min_tables=scenario.min_observations,
        scenario=scenario,
        builders=builders,
    )


# ==================== same_doc_multi_role_formula (CP1, intermediate_v2_rework_plan.md §4) ====================
#
# Keep LLM behavior within the declared contract.
# Keep LLM behavior within the declared contract.

FormulaWorkItem = tuple[DocumentRef, int]

MAX_FORMULA_QUESTION_ATTEMPTS = 2


def generate_same_doc_multi_role_formula(
    *,
    settings: Settings,
    llm: ChatLLM,
    index_store: TableIndexStore,
    count: int,
    out_path: Path,
    seed: int | None = None,
    scenario: ScenarioSpec,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
) -> int:
    del scenario
    rng = random.Random(seed)
    all_docs = scan_catalog(settings.data_root)
    eligible_docs = [
        d for d in all_docs if d.tables_dir is not None and d.text_path is not None and d.table_ids
    ]
    rng.shuffle(eligible_docs)
    work_items: list[FormulaWorkItem] = [(doc, rng.randrange(2**32)) for doc in eligible_docs]
    companies = load_company_meta(settings.company_meta_path)
    writer = JsonlWriter(out_path)

    formulas = enabled_formulas()
    formula_counts: dict[str, int] = {f.formula_id: 0 for f in formulas}
    counts_lock = Lock()

    def _formulas_by_ascending_count() -> list:
        with counts_lock:
            return sorted(formulas, key=lambda f: formula_counts[f.formula_id])

    def _build_record(item: FormulaWorkItem) -> QARecord | None:
        doc, _item_seed = item
        company = companies.get(doc.ticker)
        scope = report_scope(doc.doc_name)
        if scope not in ("consolidated", "parent"):
            return None

        for formula in _formulas_by_ascending_count():
            if formula_industry_policy_error(formula, company):
                continue
            if formula_report_scope_error(formula, scope):
                continue

            requests = [
                DocumentRoleRequest(role_id=role.role_id, concept_names=role.concept_names)
                for role in formula.required_roles
            ]
            tables_by_role = retrieve_document_roles(
                doc=doc,
                requests=requests,
                index_store=index_store,
                company_name=company.name if company else doc.ticker,
                report_scope=scope,  # type: ignore[arg-type]
            )
            if tables_by_role is None:
                continue

            candidate_by_ref: dict[str, CandidateTable] = {}
            for candidates in tables_by_role.values():
                for candidate in candidates:
                    candidate_by_ref.setdefault(candidate.table_ref, candidate)

            system, user = build_formula_role_mapping_prompt(formula, tables_by_role)
            mapping_result, _err = call_structured(
                llm, system=system, user=user, schema=FormulaMappingResult
            )
            if mapping_result is None:
                continue
            mapping_by_role = {m.role_id: m for m in mapping_result.mappings}

            bindings: dict[str, ResolvedCell] = {}
            mapping_failed = False
            for role in formula.required_roles:
                mapping = mapping_by_role.get(role.role_id)
                if mapping is None:
                    logger.info("Formula chain: LLM returned no mapping for role %s.", role.role_id)
                    mapping_failed = True
                    break
                candidate = candidate_by_ref.get(mapping.table_ref)
                if candidate is None:
                    logger.info(
                        "Formula chain: role %s references a table_ref outside candidates: %s",
                        role.role_id,
                        mapping.table_ref,
                    )
                    mapping_failed = True
                    break
                table = load_table_for_candidate(candidate)
                unit_context = f"{candidate.csv_header} {' '.join(candidate.unit_snippets)}"
                resolved = resolve_role_cell(mapping, table, unit_context=unit_context)
                if isinstance(resolved, str):
                    logger.info("Chain formula: %s", resolved)
                    mapping_failed = True
                    break
                bindings[role.role_id] = resolved
            if mapping_failed:
                continue

            coverage_error = formula_role_coverage_error(formula, bindings)
            if coverage_error:
                logger.info("Chain formula: %s", coverage_error)
                continue

            scale_error = formula_scale_consistency_error(bindings)
            if scale_error:
                logger.info("Chain formula: %s", scale_error)
                continue

            try:
                values = {
                    role_id: parse_vn_number(cell.raw_value) for role_id, cell in bindings.items()
                }
            except VnNumberError as exc:
                logger.info("Formula chain: failed to parse a Vietnamese number: %s", exc)
                continue

            denominator_error = formula_denominator_positive_error(formula.formula_id, values)
            if denominator_error:
                logger.info("Chain formula: %s", denominator_error)
                continue
            sign_error = formula_sign_policy_error(formula.formula_id, values)
            if sign_error:
                logger.info("Chain formula: %s", sign_error)
                continue

            plan = FormulaScenarioPlan(
                scenario="same_doc_multi_role_formula",
                formula_id=formula.formula_id,
                entity_ticker=doc.ticker,
                entity_name=company.name if company else doc.ticker,
                document_doc_name=doc.doc_name,
                year=doc.year,
                report_scope=scope,  # type: ignore[arg-type]
                answer_type=formula.answer_type,
                unit=formula.unit,
                bindings=bindings,
            )

            system, user = build_formula_finance_judge_prompt(formula, plan)
            finance_judgment, err = call_structured(
                llm, system=system, user=user, schema=FinancialValidityJudgment, max_attempts=1
            )
            if finance_judgment is None or not finance_judgment.valid:
                logger.info(
                    "Chain formula: finance judge reject: %s",
                    finance_judgment.reason if finance_judgment else err,
                )
                continue

            pandas_query = compile_formula_query(plan)
            relevant_tables = plan.relevant_tables()
            csv_map = {ref: candidate_by_ref[ref].csv_path for ref in relevant_tables}
            execution = execute_query(pandas_query, csv_map)
            if not execution.ok:
                logger.info("Formula chain: query failed on the original CSV: %s", execution.detail)
                continue
            finite_error = formula_answer_finite_error(execution.actual)
            if finite_error:
                logger.info("Chain formula: %s", finite_error)
                continue
            missing_refs = set(csv_map) - set(execution.accessed_refs)
            if missing_refs:
                logger.info(
                    "Formula chain: pandas_query did not read all selected tables: %s",
                    sorted(missing_refs),
                )
                continue

            question_feedback = ""
            for _question_attempt in range(MAX_FORMULA_QUESTION_ATTEMPTS):
                system, user = build_formula_question_prompt(formula, plan)
                if question_feedback:
                    user = (
                        f"{user}\n\nPrevious error: {question_feedback}\n"
                        "Rewrite it and return the required JSON."
                    )
                draft, err = call_structured(
                    llm, system=system, user=user, schema=QuestionDraft, max_attempts=1
                )
                if draft is None:
                    question_feedback = err
                    continue
                style_error = question_style_error(draft.question)
                if style_error:
                    question_feedback = style_error
                    continue
                alignment_error = formula_question_alignment_error(draft.question, plan)
                if alignment_error:
                    question_feedback = alignment_error
                    continue
                question, natural_ok, judge_detail = judge_and_maybe_rewrite(
                    llm=llm, question=draft.question, table_labels=formula.name_vi
                )
                if not natural_ok:
                    question_feedback = f"Question is unnatural: {judge_detail}"
                    continue

                with counts_lock:
                    formula_counts[formula.formula_id] += 1
                return QARecord(
                    id=0,
                    question=question,
                    answer=execution.actual,
                    relevant_docs=[doc.doc_name],
                    relevant_tables=relevant_tables,
                    pandas_query=pandas_query,
                    csv_path=[str(csv_map[ref]) for ref in relevant_tables],
                    difficulty="intermediate",
                )

            logger.info(
            "Formula chain: could not write an acceptable question after locking the query: %s",
                question_feedback,
            )

        return None

    generated = run_parallel_generation(
        work_items=work_items,
        build_record=_build_record,
        count=count,
        writer=writer,
        max_workers=max_workers,
        max_candidates=max_candidates,
    )
    if generated < count:
        logger.warning(
            "Generated only %d/%d Intermediate questions (same_doc_multi_role_formula).", generated, count
        )
    return generated


# ==================== peer_group_longitudinal (CP3, intermediate_v2_rework_plan.md §5) ====================
#
# Keep LLM behavior within the declared contract.
# Keep report-scope handling explicit.
# Keep LLM behavior within the declared contract.

LongitudinalWorkItem = tuple[PeerLongitudinalGroup, tuple[str, str, str], str, str, str, int]

MAX_LONGITUDINAL_QUESTION_ATTEMPTS = 2


def generate_peer_group_longitudinal(
    *,
    settings: Settings,
    llm: ChatLLM,
    index_store: TableIndexStore,
    count: int,
    out_path: Path,
    seed: int | None = None,
    scenario: ScenarioSpec,
    max_triples_per_group: int = DEFAULT_MAX_TRIPLES_PER_GROUP,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
) -> int:
    del scenario
    rng = random.Random(seed)
    all_docs = scan_catalog(settings.data_root)
    companies = load_company_meta(settings.company_meta_path)
    groups = enumerate_peer_longitudinal_groups(all_docs, companies)

    work_items: list[LongitudinalWorkItem] = []
    for group in groups:
        group_rng = random.Random(rng.randrange(2**32))
        triples = iter_ticker_triples(
            group.eligible_tickers, rng=group_rng, max_triples=max_triples_per_group
        )
        for triple in triples:
            for metric in enabled_input_metrics():
                for transform in enabled_transforms():
                    for reducer_id in get_reducer_ids():
                        work_items.append(
                            (
                                group,
                                triple,
                                metric.metric_id,
                                transform.transform_id,
                                reducer_id,
                                rng.randrange(2**32),
                            )
                        )
    rng.shuffle(work_items)
    writer = JsonlWriter(out_path)

    def _build_record(item: LongitudinalWorkItem) -> QARecord | None:
        group, triple, metric_id, transform_id, reducer_id, _item_seed = item
        metric = get_input_metric(metric_id)
        transform = get_transform(transform_id)

        bundle = retrieve_peer_longitudinal_bundle(
            group=group,
            tickers=triple,
            concept_names=metric.concept_names,
            index_store=index_store,
            companies=companies,
        )
        if bundle is None:
            return None

        entity_names = {ticker: (companies[ticker].name if ticker in companies else ticker) for ticker in triple}
        candidate_by_ref: dict[str, CandidateTable] = {}
        for candidates in bundle.candidates_by_cell.values():
            for candidate in candidates:
                candidate_by_ref.setdefault(candidate.table_ref, candidate)

        system, user = build_longitudinal_cell_mapping_prompt(
            metric, triple, entity_names, group.periods, bundle.candidates_by_cell
        )
        mapping_result, _err = call_structured(
            llm, system=system, user=user, schema=FormulaMappingResult
        )
        if mapping_result is None:
            return None
        mapping_by_role = {m.role_id: m for m in mapping_result.mappings}

        bindings: dict[tuple[str, str], ResolvedCell] = {}
        mapping_failed = False
        for ticker in triple:
            for period in group.periods:
                role_id = cell_role_id(ticker, period)
                mapping = mapping_by_role.get(role_id)
                if mapping is None:
                    logger.info("Longitudinal chain: LLM returned no mapping for cell %s.", role_id)
                    mapping_failed = True
                    break
                candidate = candidate_by_ref.get(mapping.table_ref)
                if candidate is None:
                    logger.info(
                        "Longitudinal chain: cell %s references a table_ref outside candidates: %s",
                        role_id,
                        mapping.table_ref,
                    )
                    mapping_failed = True
                    break
                table = load_table_for_candidate(candidate)
                unit_context = f"{candidate.csv_header} {' '.join(candidate.unit_snippets)}"
                resolved = resolve_role_cell(mapping, table, unit_context=unit_context)
                if isinstance(resolved, str):
                    logger.info("Chain longitudinal: %s", resolved)
                    mapping_failed = True
                    break
                bindings[(ticker, period)] = resolved
            if mapping_failed:
                break
        if mapping_failed:
            return None

        coverage_error = longitudinal_coverage_error(triple, group.periods, bindings)
        if coverage_error:
            logger.info("Chain longitudinal: %s", coverage_error)
            return None

        scale_error = longitudinal_scale_consistency_error(triple, group.periods, bindings)
        if scale_error:
            logger.info("Chain longitudinal: %s", scale_error)
            return None

        try:
            parsed_values = {
                key: parse_vn_number(cell.raw_value) * cell.scale for key, cell in bindings.items()
            }
        except VnNumberError as exc:
            logger.info("Longitudinal chain: failed to parse a Vietnamese number: %s", exc)
            return None

        near_zero_error = ""
        for ticker in triple:
            x1, x2, x3 = (parsed_values[(ticker, period)] for period in group.periods)
            near_zero_error = longitudinal_near_zero_base_error(x1, x2, x3)
            if near_zero_error:
                near_zero_error = f"Entity {ticker}: {near_zero_error}"
                break
        if near_zero_error:
            logger.info("Chain longitudinal: %s", near_zero_error)
            return None

        plan = LongitudinalScenarioPlan(
            scenario="peer_group_longitudinal",
            metric_family=metric.metric_id,
            input_metric_kind=metric.kind,
            entities=triple,
            entity_names=entity_names,
            periods=group.periods,
            report_scope=group.report_scope,
            per_entity_transform=transform_id,
            terminal_reducer=reducer_id,
            time_basis="Use exactly one observation from each year and transform each company independently.",
            measurement_basis="not_applicable",
            unit="%",
            answer_type="percentage",
            bindings=bindings,
        )

        system, user = build_longitudinal_finance_judge_prompt(metric, plan)
        finance_judgment, err = call_structured(
            llm, system=system, user=user, schema=FinancialValidityJudgment, max_attempts=1
        )
        if finance_judgment is None or not finance_judgment.valid:
            logger.info(
                "Chain longitudinal: finance judge reject: %s",
                finance_judgment.reason if finance_judgment else err,
            )
            return None

        pandas_query = compile_longitudinal_query(plan)
        relevant_tables = plan.relevant_tables()
        csv_map = {ref: candidate_by_ref[ref].csv_path for ref in relevant_tables}
        execution = execute_query(pandas_query, csv_map)
        if not execution.ok:
            logger.info("Longitudinal chain: query failed on the original CSV: %s", execution.detail)
            return None
        finite_error = formula_answer_finite_error(execution.actual)
        if finite_error:
            logger.info("Chain longitudinal: %s", finite_error)
            return None
        missing_refs = set(csv_map) - set(execution.accessed_refs)
        if missing_refs:
            logger.info(
                "Longitudinal chain: pandas_query did not read all selected tables: %s",
                sorted(missing_refs),
            )
            return None

        question_feedback = ""
        for _question_attempt in range(MAX_LONGITUDINAL_QUESTION_ATTEMPTS):
            system, user = build_longitudinal_question_prompt(metric, transform, reducer_id, plan)
            if question_feedback:
                user = (
                    f"{user}\n\nPrevious error: {question_feedback}\n"
                    "Rewrite it and return the required JSON."
                )
            draft, err = call_structured(
                llm, system=system, user=user, schema=QuestionDraft, max_attempts=1
            )
            if draft is None:
                question_feedback = err
                continue
            style_error = question_style_error(draft.question)
            if style_error:
                question_feedback = style_error
                continue
            alignment_error = longitudinal_question_alignment_error(draft.question, plan)
            if alignment_error:
                question_feedback = alignment_error
                continue
            question, natural_ok, judge_detail = judge_and_maybe_rewrite(
                llm=llm, question=draft.question, table_labels=metric.label_vi
            )
            if not natural_ok:
                question_feedback = f"Question is unnatural: {judge_detail}"
                continue

            return QARecord(
                id=0,
                question=question,
                answer=execution.actual,
                relevant_docs=sorted({candidate_by_ref[ref].doc_name for ref in relevant_tables}),
                relevant_tables=relevant_tables,
                pandas_query=pandas_query,
                csv_path=[str(csv_map[ref]) for ref in relevant_tables],
                difficulty="intermediate",
            )

        logger.info(
            "Longitudinal chain: could not write an acceptable question after locking the query: %s",
            question_feedback,
        )
        return None

    generated = run_parallel_generation(
        work_items=work_items,
        build_record=_build_record,
        count=count,
        writer=writer,
        max_workers=max_workers,
        max_candidates=max_candidates,
    )
    if generated < count:
        logger.warning(
            "Generated only %d/%d Intermediate questions (peer_group_longitudinal).", generated, count
        )
    return generated
