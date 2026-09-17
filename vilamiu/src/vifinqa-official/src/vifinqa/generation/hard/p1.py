
from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import DocumentRef, scan_catalog
from vifinqa.common.corpus.company_meta import CompanyInfo, load_company_meta
from vifinqa.embeddings.base import Embedder
from vifinqa.generation.common import (
    MAX_ATTEMPTS,
    CandidateTable,
    call_structured,
    judge_and_maybe_rewrite,
    load_table_for_candidate,
    question_style_error,
    report_scope,
    report_scope_error,
)
from vifinqa.generation.hard.compiler.p1 import P1CompileError, P1Compiler, locate_cell
from vifinqa.generation.hard.gate import (
    graph_gate_error,
    p1_coverage_error,
    p1_dependency_error,
)
from vifinqa.generation.hard.numbers import VnNumberError, parse_vn_number, resolve_unit_claim
from vifinqa.generation.hard.schemas import (
    ExecutionTrace,
    HardMappingResult,
    HardPlan,
    HardPlanDraft,
    HardTableMapping,
    MetricBinding,
    MetricRoleInput,
    ReasoningStep,
    RejectionCounters,
    StepOutputInput,
    derive_calculation_steps,
)
from vifinqa.generation.hard.scenarios import get_hard_scenario
from vifinqa.generation.hard.threshold import generate_threshold
from vifinqa.generation.parallel import DEFAULT_MAX_WORKERS, run_parallel_generation
from vifinqa.generation.prompts.hard_plan import (
    build_hard_alignment_judge_prompt,
    build_hard_finance_judge_prompt,
    build_hard_mapping_prompt,
    build_hard_plan_prompt,
    build_hard_question_prompt,
)
from vifinqa.generation.retrieval.peer_group import (
    peer_group_same_period_preflight_ok,
    retrieve_peer_group_same_period_candidates,
)
from vifinqa.generation.schemas import FinancialValidityJudgment, QARecord, QuestionDraft
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.output.writer import JsonlWriter
from vifinqa.generation.validation.pandas_check import execute_query, values_match

logger = logging.getLogger(__name__)

MIN_ENTITIES = 3
MAX_EXTRA_TABLES = 4
TOP_K_SEARCH = 30
SCENARIO = get_hard_scenario("p1_filter_aggregate")

HardP1WorkItem = tuple[DocumentRef, int]


def _preflight_work_items(
    work_items: Iterable[HardP1WorkItem],
    *,
    all_docs: list[DocumentRef],
    companies: dict[str, CompanyInfo],
    min_entities: int,
) -> Iterator[HardP1WorkItem]:
    for seed_doc, seed_table_id in work_items:
        if peer_group_same_period_preflight_ok(
            all_docs=all_docs,
            seed_doc=seed_doc,
            seed_table_id=seed_table_id,
            companies=companies,
            min_entities=min_entities,
        ):
            yield seed_doc, seed_table_id


def _plan_draft_error(draft: HardPlanDraft) -> str:
    required = {
        "concept_name": draft.concept_name,
        "metric_role": draft.metric_role,
        "concept_formula": draft.concept_formula,
        "financial_rationale": draft.financial_rationale,
        "population": draft.population,
        "unit": draft.unit,
    }
    missing = [name for name, value in required.items() if not value.strip()]
    if missing:
        return f"Plan is missing required fields: {missing}"
    if draft.final_operation == "boolean" and draft.answer_type != "boolean":
        return "final_operation=boolean requires answer_type=boolean."
    if draft.final_operation == "count" and draft.answer_type != "number":
        return "final_operation=count requires answer_type=number."
    return ""


def _canonicalize_final_answer_type(draft: HardPlanDraft) -> HardPlanDraft:
    canonical = {
        "count": "number",
        "boolean": "boolean",
    }.get(draft.final_operation)
    if canonical is None or draft.answer_type == canonical:
        return draft
    return draft.model_copy(update={"answer_type": canonical})


def _measurement_basis_error(draft: HardPlanDraft, mappings: list[HardTableMapping]) -> str:
    if draft.measurement_basis == "unknown":
        return "Plan has not locked measurement_basis (gross/net/not_applicable)."
    unknown_refs = sorted(m.table_ref for m in mappings if m.measurement_basis == "unknown")
    if unknown_refs:
        return f"Unable to determine measurement_basis for: {unknown_refs}"
    mismatched = sorted(m.table_ref for m in mappings if m.measurement_basis != draft.measurement_basis)
    if mismatched:
        return f"measurement_basis does not match the plan ({draft.measurement_basis}): {mismatched}"
    return ""


def _resolve_scale(
    candidate: CandidateTable, mapping: HardTableMapping, draft: HardPlanDraft
) -> tuple[float, str] | None:
    resolved = resolve_unit_claim(
        mapping.unit,
        csv_header=candidate.csv_header,
        unit_snippets=list(candidate.unit_snippets),
        value_kind=draft.metric_value_kind,
    )
    if resolved is None:
        return None
    return resolved.scale, resolved.normalized_unit


def build_reasoning_steps(bindings: list[MetricBinding], draft: HardPlanDraft) -> list[ReasoningStep]:
    steps: list[ReasoningStep] = []
    extract_ids: list[str] = []
    for binding in bindings:
        step_id = f"extract_{binding.ticker}"
        steps.append(
            ReasoningStep(
                step_id=step_id,
                operation="extract",
                inputs=[MetricRoleInput(metric_role=binding.metric_role)],
                group_by="company",
                output=f"raw_{binding.ticker}",
                description=f"Lấy {draft.concept_name} của {binding.ticker} năm {binding.period}.",
            )
        )
        extract_ids.append(step_id)
    steps.append(
        ReasoningStep(
            step_id="filter_1",
            operation="filter",
            inputs=[StepOutputInput(step_id=i) for i in extract_ids],
            group_by="company",
            output="passed",
            description=f"Lọc công ty có {draft.concept_name} {draft.comparison} ngưỡng đã chọn.",
        )
    )
    steps.append(
        ReasoningStep(
            step_id="final",
            operation=draft.final_operation,
            inputs=[StepOutputInput(step_id="filter_1")],
            group_by="none",
            output="answer",
            description=f"Tính {draft.final_operation} trên tập công ty đã lọc.",
        )
    )
    return steps


def generate_hard_p1(
    *,
    settings: Settings,
    llm: ChatLLM,
    embedder: Embedder,
    count: int,
    out_path: Path,
    seed: int | None = None,
    min_entities: int = MIN_ENTITIES,
    max_extra_tables: int = MAX_EXTRA_TABLES,
    top_k_search: int = TOP_K_SEARCH,
    context_pages_before: int = 1,
    context_pages_after: int = 1,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
) -> int:
    required_entities = max(min_entities, SCENARIO.min_entities)
    rng = random.Random(seed)
    all_docs = scan_catalog(settings.data_root)
    raw_seed_candidates = [
        (d, tid) for d in all_docs if d.tables_dir is not None and d.text_path is not None for tid in d.table_ids
    ]
    rng.shuffle(raw_seed_candidates)
    companies = load_company_meta(settings.company_meta_path)
    seed_candidates = _preflight_work_items(
        raw_seed_candidates,
        all_docs=all_docs,
        companies=companies,
        min_entities=required_entities,
    )
    writer = JsonlWriter(out_path)
    counters = RejectionCounters()

    def _build_record(item: tuple[DocumentRef, int]) -> QARecord | None:
        seed_doc, seed_table_id = item
        item_rng = random.Random(rng.randrange(2**32))

        candidates = retrieve_peer_group_same_period_candidates(
            all_docs=all_docs,
            seed_doc=seed_doc,
            seed_table_id=seed_table_id,
            embedder=embedder,
            companies=companies,
            rng=item_rng,
            min_entities=required_entities,
            max_extra_tables=max_extra_tables,
            top_k_search=top_k_search,
            context_pages_before=context_pages_before,
            context_pages_after=context_pages_after,
        )
        if candidates is None:
            counters.retrieval += 1
            return None
        candidate_by_ref = {c.table_ref: c for c in candidates}

        system, user = build_hard_plan_prompt(candidates, min_entities=required_entities)
        draft, _err = call_structured(llm, system=system, user=user, schema=HardPlanDraft)
        if draft is None:
            logger.debug("Hard P1: plan could not be parsed (candidates=%s): %s", list(candidate_by_ref), _err)
            counters.plan += 1
            return None
        if not draft.feasible:
            logger.debug(
                "Hard P1: plan reported infeasible (candidates=%s): %s", list(candidate_by_ref), draft.reason
            )
            counters.plan += 1
            return None
        draft = _canonicalize_final_answer_type(draft)
        draft_error = _plan_draft_error(draft)
        if draft_error:
            logger.debug("Hard P1: invalid plan draft (candidates=%s): %s", list(candidate_by_ref), draft_error)
            counters.plan += 1
            return None

        system, user = build_hard_mapping_prompt(candidates, draft.concept_formula, draft.metric_role)
        mapping_result, _err = call_structured(llm, system=system, user=user, schema=HardMappingResult)
        if mapping_result is None:
            counters.mapping += 1
            return None
        valid_mappings = [
            m
            for m in mapping_result.mappings
            if m.has_concept and m.table_ref in candidate_by_ref and m.row_label and m.column_label
        ]
        if len({candidate_by_ref[m.table_ref].ticker for m in valid_mappings}) < required_entities:
            counters.mapping += 1
            return None

        basis_mappings = [
            mapping
            for mapping in valid_mappings
            if draft.measurement_basis != "unknown"
            and mapping.measurement_basis == draft.measurement_basis
        ]
        basis_error = _measurement_basis_error(draft, basis_mappings)
        if basis_error:
            counters.scope_or_basis += 1
            return None
        if len({candidate_by_ref[m.table_ref].ticker for m in basis_mappings}) < required_entities:
            counters.scope_or_basis += 1
            return None
        selected_tables = [candidate_by_ref[m.table_ref] for m in basis_mappings]
        scope_error = report_scope_error(selected_tables)
        if scope_error:
            counters.scope_or_basis += 1
            return None
        table_scope = report_scope(selected_tables[0].doc_name)
        if table_scope not in ("consolidated", "parent"):
            counters.scope_or_basis += 1
            return None

        bindings: list[MetricBinding] = []
        for mapping in basis_mappings:
            candidate = candidate_by_ref[mapping.table_ref]
            resolved = _resolve_scale(candidate, mapping, draft)
            if resolved is None:
                continue
            scale, normalized_unit = resolved
            bindings.append(
                MetricBinding(
                    metric_role=draft.metric_role,
                    table_ref=mapping.table_ref,
                    ticker=candidate.ticker,
                    period=candidate.year,
                    report_scope=table_scope,  # type: ignore[arg-type]
                    row_label=mapping.row_label,
                    column_label=mapping.column_label,
                    raw_unit=mapping.unit.evidence,
                    scale=scale,
                    normalized_unit=normalized_unit,
                    measurement_basis=mapping.measurement_basis,
                    evidence=mapping.unit.evidence,
                )
            )
        if len({b.ticker for b in bindings}) < required_entities:
            counters.unit_gate += 1
            return None
        # Keep unit and scale handling explicit.
        selected_tables = [candidate_by_ref[b.table_ref] for b in bindings]

        reasoning_steps = build_reasoning_steps(bindings, draft)
        plan = HardPlan(
            draft=draft,
            reasoning_steps=reasoning_steps,
            calculation_steps=derive_calculation_steps(reasoning_steps),
        )
        if graph_gate_error(plan):
            counters.graph_gate += 1
            return None
        if p1_dependency_error(plan, SCENARIO):
            counters.graph_gate += 1
            return None
        if p1_coverage_error(bindings, [b.table_ref for b in bindings]):
            counters.coverage_gate += 1
            return None

        tables_by_ref = {b.table_ref: load_table_for_candidate(candidate_by_ref[b.table_ref]) for b in bindings}
        raw_values: dict[str, float] = {}
        for binding in bindings:
            location = locate_cell(tables_by_ref[binding.table_ref], binding.row_label, binding.column_label)
            if location is None:
                counters.compile += 1
                return None
            raw_cell = tables_by_ref[binding.table_ref].rows[location[0]][location[1]]
            try:
                raw_values[binding.ticker] = parse_vn_number(raw_cell) * binding.scale
            except VnNumberError:
                counters.compile += 1
                return None

        threshold = generate_threshold(list(raw_values.values()), rng=item_rng, comparison=draft.comparison)
        if threshold is None:
            counters.threshold += 1
            return None
        plan.threshold = threshold

        try:
            compiled = P1Compiler().compile(plan, bindings, tables_by_ref)
        except P1CompileError:
            counters.compile += 1
            return None

        csv_map = {b.table_ref: candidate_by_ref[b.table_ref].csv_path for b in bindings}
        execution = execute_query(compiled.pandas_query, csv_map)
        if not execution.ok or not values_match(compiled.expected_answer, execution.actual):
            counters.execute += 1
            logger.warning(
                "Hard P1: executed query differs from the compiler value (expected=%r actual=%r ok=%s)",
                compiled.expected_answer,
                execution.actual,
                execution.ok,
            )
            return None
        missing_refs = set(csv_map) - set(execution.accessed_refs)
        if missing_refs:
            counters.execute += 1
            return None
        answer = execution.actual

        system, user = build_hard_finance_judge_prompt(
            plan, bindings, pandas_query=compiled.pandas_query, actual_result=answer
        )
        judgment, _err = call_structured(llm, system=system, user=user, schema=FinancialValidityJudgment, max_attempts=1)
        if judgment is None or not judgment.valid:
            counters.finance_judge += 1
            return None

        question_feedback = ""
        table_labels = "\n".join(candidate_by_ref[b.table_ref].table_labels for b in bindings)
        for _attempt in range(MAX_ATTEMPTS):
            system, user = build_hard_question_prompt(plan, selected_tables)
            if question_feedback:
                user = f"{user}\n\nLần trước bị lỗi: {question_feedback}\nHãy viết lại và trả đúng JSON yêu cầu."
            draft_q, _err = call_structured(llm, system=system, user=user, schema=QuestionDraft, max_attempts=1)
            if draft_q is None:
                question_feedback = _err
                continue
            style_error = question_style_error(draft_q.question)
            if style_error:
                counters.question_style += 1
                question_feedback = style_error
                continue

            identities = "\n".join(
                f"- {c.company_name} ({c.ticker}), năm {c.year}, phạm vi {report_scope(c.doc_name)}"
                for c in selected_tables
            )
            system, user = build_hard_alignment_judge_prompt(plan, draft_q.question, identities)
            alignment, _err = call_structured(llm, system=system, user=user, schema=FinancialValidityJudgment, max_attempts=1)
            if alignment is None or not alignment.valid:
                counters.alignment_judge += 1
                question_feedback = alignment.reason if alignment else _err
                continue

            question, natural_ok, judge_detail = judge_and_maybe_rewrite(
                llm=llm, question=draft_q.question, table_labels=table_labels
            )
            if not natural_ok:
                counters.naturalness_judge += 1
                question_feedback = f"Question is not natural: {judge_detail}"
                continue

            used_refs = [b.table_ref for b in bindings]
            doc_names = sorted({candidate_by_ref[ref].doc_name for ref in used_refs})
            trace = ExecutionTrace(
                scenario=SCENARIO.name,
                plan=plan,
                bindings=bindings,
                accessed_refs=execution.accessed_refs,
            )
            logger.debug("Hard P1 execution trace: %s", trace)
            return QARecord(
                id=0,
                question=question,
                answer=answer,
                relevant_docs=doc_names,
                relevant_tables=used_refs,
                pandas_query=compiled.pandas_query,
                csv_path=[str(csv_map[ref]) for ref in used_refs],
                difficulty="hard",
            )

        logger.info("Hard P1: could not write an acceptable question after locking the query: %s", question_feedback)
        return None

    generated = run_parallel_generation(
        work_items=seed_candidates,
        build_record=_build_record,
        count=count,
        writer=writer,
        max_workers=max_workers,
        max_candidates=max_candidates,
    )

    logger.info("Hard P1 rejection counters: %s", counters.as_dict())
    if generated < count:
        logger.warning("Generated only %d/%d Hard questions (P1).", generated, count)
    return generated
