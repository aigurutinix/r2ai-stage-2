
from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from threading import Lock

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import DocumentRef, scan_catalog
from vifinqa.common.corpus.company_meta import CompanyInfo, load_company_meta
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.generation.table_index.base import TableIndexStore, TableSearchIndex
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.budget import LLMCallBudget
from vifinqa.generation.common import (
    MAX_ATTEMPTS,
    CandidateTable,
    call_structured,
    candidate_table_from_doc,
    load_table_for_candidate,
    question_style_error,
    report_scope,
)
from vifinqa.generation.hard.compiler.p3 import P3CompileError, P3Compiler
from vifinqa.generation.hard.gate import (
    graph_gate_error,
    p3_coverage_error,
    p3_dependency_error,
)
from vifinqa.generation.hard.numbers import resolve_unit_claim
from vifinqa.generation.hard.schemas import (
    ExecutionTrace,
    HardMultiMetricPlanDraft,
    HardMultiRoleMappingResult,
    HardMultiRoleTableMapping,
    HardP3PairJudgment,
    HardP3Plan,
    HardP3QuestionJudgment,
    MetricBinding,
    MetricRoleInput,
    ReasoningStep,
    RejectionCounters,
    StepOutputInput,
    derive_calculation_steps,
)
from vifinqa.generation.hard.scenarios import get_hard_scenario
from vifinqa.generation.parallel import DEFAULT_MAX_WORKERS, run_parallel_generation
from vifinqa.generation.prompts.hard_plan import (
    build_hard_p3_finance_judge_prompt,
    build_hard_p3_mapping_prompt,
    build_hard_p3_pair_judge_prompt,
    build_hard_p3_plan_prompt,
    build_hard_p3_question_judge_prompt,
    build_hard_p3_question_prompt,
)
from vifinqa.generation.retrieval.base import TableDescriptor, build_table_descriptor
from vifinqa.generation.retrieval.time_series import (
    DEFAULT_PERIOD_COUNT,
    TimeSeriesWindow,
    enumerate_time_series_windows,
)
from vifinqa.generation.schemas import FinancialValidityJudgment, QARecord, QuestionDraft
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.output.writer import JsonlWriter
from vifinqa.generation.validation.pandas_check import execute_query, values_match

logger = logging.getLogger(__name__)

DEFAULT_TABLES_PER_PERIOD_ROLE = 2
SCENARIO = get_hard_scenario("p3_period_selector")


def _plan_draft_error(draft: HardMultiMetricPlanDraft, inventory_by_ref: dict[str, TableDescriptor]) -> str:
    required = {
        "selector_anchor_ref": draft.selector_anchor_ref,
        "selector_concept_name": draft.selector_concept_name,
        "selector_metric_role": draft.selector_metric_role,
        "selector_concept_formula": draft.selector_concept_formula,
        "selector_financial_rationale": draft.selector_financial_rationale,
        "selector_unit": draft.selector_unit,
        "answer_anchor_ref": draft.answer_anchor_ref,
        "answer_concept_name": draft.answer_concept_name,
        "answer_metric_role": draft.answer_metric_role,
        "answer_concept_formula": draft.answer_concept_formula,
        "answer_financial_rationale": draft.answer_financial_rationale,
        "answer_unit": draft.answer_unit,
        "population": draft.population,
        "analysis_intent": draft.analysis_intent,
        "relationship_rationale": draft.relationship_rationale,
    }
    missing = [name for name, value in required.items() if not value.strip()]
    if missing:
        return f"Plan is missing required fields: {missing}"
    if draft.selector_metric_role == draft.answer_metric_role:
        return "selector_metric_role and answer_metric_role must be different financial metrics."
    if draft.selector_anchor_ref not in inventory_by_ref:
        return f"selector_anchor_ref does not exist in the anchor inventory: {draft.selector_anchor_ref!r}"
    if draft.answer_anchor_ref not in inventory_by_ref:
        return f"answer_anchor_ref does not exist in the anchor inventory: {draft.answer_anchor_ref!r}"
    if draft.answer_type not in ("money", "percentage", "number"):
        return f"answer_type (P3) must be money|percentage|number; got {draft.answer_type!r}."
    return ""


def _build_anchor_inventory(
    anchor_docs: tuple[DocumentRef, ...], companies: dict[str, CompanyInfo]
) -> list[TableDescriptor]:
    inventory: list[TableDescriptor] = []
    for doc in anchor_docs:
        if doc.tables_dir is None or doc.text_path is None:
            continue
        scope = report_scope(doc.doc_name)
        if scope not in ("consolidated", "parent"):
            continue
        document = parse_document(doc.text_path)
        company = companies.get(doc.ticker)
        for table_id in doc.table_ids:
            table = load_table(
                doc.table_csv_path(table_id), ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=table_id
            )
            if not is_table_eligible(table):
                continue
            inventory.append(
                build_table_descriptor(
                    doc=doc,
                    table=table,
                    document=document,
                    company_name=company.name if company else doc.ticker,
                    report_scope=scope,  # type: ignore[arg-type]
                )
            )
    return inventory


def _table_ref_parts(table_ref: str) -> tuple[str, int]:
    doc_name, _, table_part = table_ref.partition("|table_")
    return doc_name, int(table_part)


def _shortlist_for_role(
    *,
    index: TableSearchIndex,
    query: str,
    doc_by_name: dict[str, DocumentRef],
    companies: dict[str, CompanyInfo],
    max_tables: int,
    keep_ref: str | None,
) -> list[CandidateTable]:
    result: list[CandidateTable] = []
    seen_refs: set[str] = set()

    def _add(doc_name: str, table_id: int) -> None:
        if len(result) >= max_tables:
            return
        ref = f"{doc_name}|table_{table_id}"
        if ref in seen_refs:
            return
        doc = doc_by_name.get(doc_name)
        if doc is None:
            return
        company = companies.get(doc.ticker)
        result.append(
            candidate_table_from_doc(
                doc,
                table_id,
                company_name=company.name if company else doc.ticker,
                context_before=0,
                context_after=0,
            )
        )
        seen_refs.add(ref)

    if keep_ref is not None:
        doc_name, table_id = _table_ref_parts(keep_ref)
        _add(doc_name, table_id)

    hits = index.search(query, top_k=max(max_tables, 5))
    for hit_table, _score in hits:
        _add(hit_table.doc_name, hit_table.table_id)

    return result


def _role_query(concept_name: str, formula: str, descriptor: TableDescriptor) -> str:
    return f"{concept_name}. {formula}. {descriptor.table_labels}. {descriptor.anchor_context}"


def _role_specific_retrieval(
    *,
    window: TimeSeriesWindow,
    draft: HardMultiMetricPlanDraft,
    index_store: TableIndexStore,
    companies: dict[str, CompanyInfo],
    inventory_by_ref: dict[str, TableDescriptor],
    max_tables_per_period: int,
) -> tuple[dict[str, list[CandidateTable]], dict[str, list[CandidateTable]]] | None:
    selector_query = _role_query(
        draft.selector_concept_name, draft.selector_concept_formula, inventory_by_ref[draft.selector_anchor_ref]
    )
    answer_query = _role_query(
        draft.answer_concept_name, draft.answer_concept_formula, inventory_by_ref[draft.answer_anchor_ref]
    )

    selector_by_period: dict[str, list[CandidateTable]] = {}
    answer_by_period: dict[str, list[CandidateTable]] = {}
    for period in window.periods:
        docs = list(window.docs_by_period[period])
        doc_by_name = {d.doc_name: d for d in docs}
        index = index_store.get(ticker=window.ticker, report_scope=window.report_scope, period=period, docs=docs)

        selector_tables = _shortlist_for_role(
            index=index,
            query=selector_query,
            doc_by_name=doc_by_name,
            companies=companies,
            max_tables=max_tables_per_period,
            keep_ref=draft.selector_anchor_ref if period == window.anchor_period else None,
        )
        if not selector_tables:
            return None
        selector_by_period[period] = selector_tables

        answer_tables = _shortlist_for_role(
            index=index,
            query=answer_query,
            doc_by_name=doc_by_name,
            companies=companies,
            max_tables=max_tables_per_period,
            keep_ref=draft.answer_anchor_ref if period == window.anchor_period else None,
        )
        if not answer_tables:
            return None
        answer_by_period[period] = answer_tables
    return selector_by_period, answer_by_period


def _mappings_by_period(
    mappings: list[HardMultiRoleTableMapping], role: str, candidate_by_ref: dict[str, CandidateTable]
) -> dict[str, list[HardMultiRoleTableMapping]]:
    by_period: dict[str, list[HardMultiRoleTableMapping]] = {}
    for m in mappings:
        candidate = candidate_by_ref.get(m.table_ref)
        if candidate is None:
            continue
        if role == "selector":
            has, row, col = m.has_selector, m.selector_row_label, m.selector_column_label
        else:
            has, row, col = m.has_answer, m.answer_row_label, m.answer_column_label
        if not (has and row and col):
            continue
        by_period.setdefault(candidate.year, []).append(m)
    return by_period


def _resolve_unique_binding_per_period(
    by_period: dict[str, list[HardMultiRoleTableMapping]], periods: tuple[str, ...]
) -> dict[str, HardMultiRoleTableMapping] | None:
    resolved: dict[str, HardMultiRoleTableMapping] = {}
    for period in periods:
        entries = by_period.get(period, [])
        if len(entries) != 1:
            return None
        resolved[period] = entries[0]
    return resolved


def _mapping_rejection_reason(
    mappings: list[HardMultiRoleTableMapping],
    candidate_by_ref: dict[str, CandidateTable],
    periods: tuple[str, ...],
) -> str:
    unknown_refs = sorted({m.table_ref for m in mappings if m.table_ref not in candidate_by_ref})

    def _role_stats(role: str) -> tuple[dict[str, int], int, int]:
        valid_by_period = {period: 0 for period in periods}
        marked_absent = 0
        missing_location = 0
        for mapping in mappings:
            candidate = candidate_by_ref.get(mapping.table_ref)
            if candidate is None:
                continue
            if role == "selector":
                has = mapping.has_selector
                row = mapping.selector_row_label
                column = mapping.selector_column_label
            else:
                has = mapping.has_answer
                row = mapping.answer_row_label
                column = mapping.answer_column_label
            if not has:
                marked_absent += 1
            elif not row or not column:
                missing_location += 1
            elif candidate.year in valid_by_period:
                valid_by_period[candidate.year] += 1
        return valid_by_period, marked_absent, missing_location

    selector_counts, selector_absent, selector_missing_location = _role_stats("selector")
    answer_counts, answer_absent, answer_missing_location = _role_stats("answer")
    return (
        f"each role must have exactly one mapping per period; returned_mappings={len(mappings)} "
        f"unknown_refs={unknown_refs} selector_counts={selector_counts} "
        f"selector_has_false={selector_absent} selector_missing_location={selector_missing_location} "
        f"answer_counts={answer_counts} answer_has_false={answer_absent} "
        f"answer_missing_location={answer_missing_location}"
    )


def _build_role_bindings(
    *,
    metric_role: str,
    resolved_by_period: dict[str, HardMultiRoleTableMapping],
    candidate_by_ref: dict[str, CandidateTable],
    value_kind: str,
    measurement_basis: str,
    report_scope_value: str,
    role: str,
) -> list[MetricBinding] | None:
    kinds: set[str] = set()
    bindings: list[MetricBinding] = []
    for period, mapping in resolved_by_period.items():
        candidate = candidate_by_ref[mapping.table_ref]
        basis = mapping.selector_measurement_basis if role == "selector" else mapping.answer_measurement_basis
        if basis != measurement_basis:
            return None
        unit_claim = mapping.selector_unit if role == "selector" else mapping.answer_unit
        resolved = resolve_unit_claim(
            unit_claim,
            csv_header=candidate.csv_header,
            unit_snippets=list(candidate.unit_snippets),
            value_kind=value_kind,  # type: ignore[arg-type]
        )
        if resolved is None:
            return None
        kinds.add(unit_claim.kind)
        row_label = mapping.selector_row_label if role == "selector" else mapping.answer_row_label
        column_label = mapping.selector_column_label if role == "selector" else mapping.answer_column_label
        bindings.append(
            MetricBinding(
                metric_role=metric_role,
                table_ref=mapping.table_ref,
                ticker=candidate.ticker,
                period=period,
                report_scope=report_scope_value,  # type: ignore[arg-type]
                row_label=row_label,
                column_label=column_label,
                raw_unit=unit_claim.evidence,
                scale=resolved.scale,
                normalized_unit=resolved.normalized_unit,
                measurement_basis=basis,
                evidence=unit_claim.evidence,
            )
        )
    if len(kinds) > 1:
        return None
    return bindings


def build_reasoning_steps(periods: list[str], draft: HardMultiMetricPlanDraft) -> list[ReasoningStep]:
    steps: list[ReasoningStep] = []
    selector_ids: list[str] = []
    for period in periods:
        step_id = f"extract_selector_{period}"
        steps.append(
            ReasoningStep(
                step_id=step_id,
                operation="extract",
                inputs=[MetricRoleInput(metric_role=draft.selector_metric_role)],
                group_by="period",
                output=f"selector_{period}",
                description=f"Lấy {draft.selector_concept_name} năm {period}.",
            )
        )
        selector_ids.append(step_id)

    direction_vn = "cao nhất" if draft.selector_operation == "argmax" else "thấp nhất"
    steps.append(
        ReasoningStep(
            step_id="select_period",
            operation=draft.selector_operation,
            inputs=[StepOutputInput(step_id=i) for i in selector_ids],
            group_by="period",
            output="selected_period",
            description=f"Chọn năm có {draft.selector_concept_name} {direction_vn}.",
        )
    )

    answer_ids: list[str] = []
    for period in periods:
        step_id = f"extract_answer_{period}"
        steps.append(
            ReasoningStep(
                step_id=step_id,
                operation="extract",
                inputs=[MetricRoleInput(metric_role=draft.answer_metric_role)],
                group_by="period",
                output=f"answer_{period}",
                description=f"Lấy {draft.answer_concept_name} năm {period}.",
            )
        )
        answer_ids.append(step_id)

    steps.append(
        ReasoningStep(
            step_id="final",
            operation="lookup",
            inputs=[
                StepOutputInput(step_id="select_period"),
                *[StepOutputInput(step_id=i) for i in answer_ids],
            ],
            group_by="none",
            output="answer",
            description=f"Lấy {draft.answer_concept_name} tại năm đã chọn ở bước trước.",
        )
    )
    return steps


def generate_hard_p3(
    *,
    settings: Settings,
    llm: ChatLLM,
    index_store: TableIndexStore,
    count: int,
    out_path: Path,
    seed: int | None = None,
    period_count: int = DEFAULT_PERIOD_COUNT,
    max_tables_per_period_role: int = DEFAULT_TABLES_PER_PERIOD_ROLE,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
    max_llm_calls: int | None = None,
) -> int:
    required_periods = max(period_count, SCENARIO.min_periods)
    rng = random.Random(seed)
    all_docs = scan_catalog(settings.data_root)
    companies = load_company_meta(settings.company_meta_path)
    windows = enumerate_time_series_windows(all_docs, period_count=required_periods)
    rng.shuffle(windows)
    writer = JsonlWriter(out_path)
    counters = RejectionCounters()
    budget = LLMCallBudget(max_llm_calls)
    telemetry_lock = Lock()
    attempted = 0

    def _build_record(window: TimeSeriesWindow) -> QARecord | None:
        nonlocal attempted
        with telemetry_lock:
            attempted += 1
            candidate_number = attempted
        window_key = (window.ticker, window.report_scope, window.periods)
        candidate_started = time.monotonic()
        candidate_calls_started = budget.snapshot().calls

        def _before_call(stage: str) -> None:
            budget.before_call(candidate=window_key, stage=stage)

        def _reject(stage: str, reason: str) -> None:
            with telemetry_lock:
                setattr(counters, stage, getattr(counters, stage) + 1)
            snapshot = budget.snapshot()
            compact_reason = " ".join(reason.split())[:1000]
            logger.info(
                "Hard P3 reject candidate=%d window=%s reject_stage=%s reason=%s "
                "llm_calls=%d llm_calls_total=%d elapsed=%.3fs elapsed_total=%.3fs",
                candidate_number,
                window_key,
                stage,
                compact_reason,
                snapshot.calls - candidate_calls_started,
                snapshot.calls,
                time.monotonic() - candidate_started,
                snapshot.elapsed_seconds,
            )

        t_inventory = time.monotonic()
        anchor_docs = window.docs_by_period[window.anchor_period]
        inventory = _build_anchor_inventory(anchor_docs, companies)
        if not inventory:
            _reject("inventory", "anchor period has no eligible tables")
            return None
        inventory_by_ref = {d.table_ref: d for d in inventory}
        logger.debug(
            "Hard P3 stage=inventory window=%s anchor=%s tables=%d elapsed=%.3fs",
            window_key, window.anchor_period, len(inventory), time.monotonic() - t_inventory,
        )

        t_plan = time.monotonic()
        system, user = build_hard_p3_plan_prompt(inventory, min_periods=len(window.periods))
        draft, _err = call_structured(
            llm,
            system=system,
            user=user,
            schema=HardMultiMetricPlanDraft,
            before_call=lambda: _before_call("plan"),
        )
        if draft is None:
            _reject("plan", _err or "plan could not be parsed")
            return None
        if not draft.feasible:
            _reject("plan", f"plan reported infeasible: {draft.reason}")
            return None
        draft_error = _plan_draft_error(draft, inventory_by_ref)
        if draft_error:
            _reject("plan", draft_error)
            return None
        logger.debug(
            "Hard P3 stage=plan window=%s prompt_chars=%d elapsed=%.3fs",
            window_key, len(system) + len(user), time.monotonic() - t_plan,
        )
        logger.debug(
            "Hard P3 plan window=%s selector=%r operation=%s answer=%r "
            "selector_anchor=%s answer_anchor=%s",
            window_key,
            draft.selector_concept_name,
            draft.selector_operation,
            draft.answer_concept_name,
            draft.selector_anchor_ref,
            draft.answer_anchor_ref,
        )

        t_pair = time.monotonic()
        system, user = build_hard_p3_pair_judge_prompt(
            draft, inventory_by_ref[draft.selector_anchor_ref], inventory_by_ref[draft.answer_anchor_ref]
        )
        pair_judgment, _err = call_structured(
            llm,
            system=system,
            user=user,
            schema=HardP3PairJudgment,
            max_attempts=1,
            before_call=lambda: _before_call("pair_judge"),
        )
        if pair_judgment is None or not pair_judgment.valid:
            _reject("pair_judge", pair_judgment.reason if pair_judgment else _err)
            return None
        logger.debug("Hard P3 stage=pair_judge window=%s elapsed=%.3fs", window_key, time.monotonic() - t_pair)

        t_retrieval = time.monotonic()
        retrieval = _role_specific_retrieval(
            window=window,
            draft=draft,
            index_store=index_store,
            companies=companies,
            inventory_by_ref=inventory_by_ref,
            max_tables_per_period=max_tables_per_period_role,
        )
        if retrieval is None:
            _reject("role_retrieval", "missing a selector or answer candidate for at least one period")
            return None
        selector_by_period, answer_by_period = retrieval
        logger.debug(
            "Hard P3 stage=role_retrieval window=%s selector_hits=%s answer_hits=%s elapsed=%.3fs",
            window_key,
            {p: len(v) for p, v in selector_by_period.items()},
            {p: len(v) for p, v in answer_by_period.items()},
            time.monotonic() - t_retrieval,
        )
        logger.debug(
            "Hard P3 role_retrieval refs window=%s selector_refs=%s answer_refs=%s",
            window_key,
            {period: [table.table_ref for table in tables] for period, tables in selector_by_period.items()},
            {period: [table.table_ref for table in tables] for period, tables in answer_by_period.items()},
        )

        candidate_by_ref: dict[str, CandidateTable] = {}
        for by_period in (selector_by_period, answer_by_period):
            for tables in by_period.values():
                for table in tables:
                    candidate_by_ref[table.table_ref] = table

        t_mapping = time.monotonic()
        system, user = build_hard_p3_mapping_prompt(selector_by_period, answer_by_period, draft)
        mapping_result, _err = call_structured(
            llm,
            system=system,
            user=user,
            schema=HardMultiRoleMappingResult,
            before_call=lambda: _before_call("mapping"),
        )
        if mapping_result is None:
            _reject("mapping", _err or "mapping could not be parsed")
            return None

        selector_periods = _mappings_by_period(mapping_result.mappings, "selector", candidate_by_ref)
        answer_periods = _mappings_by_period(mapping_result.mappings, "answer", candidate_by_ref)
        resolved_selector = _resolve_unique_binding_per_period(selector_periods, window.periods)
        resolved_answer = _resolve_unique_binding_per_period(answer_periods, window.periods)
        if resolved_selector is None or resolved_answer is None:
            _reject("mapping", _mapping_rejection_reason(mapping_result.mappings, candidate_by_ref, window.periods))
            return None
        logger.debug(
            "Hard P3 stage=mapping window=%s table_count=%d prompt_chars=%d elapsed=%.3fs",
            window_key, len(candidate_by_ref), len(system) + len(user), time.monotonic() - t_mapping,
        )

        if draft.selector_measurement_basis == "unknown" or draft.answer_measurement_basis == "unknown":
            _reject(
                "scope_or_basis",
                "plan has not locked measurement_basis for the selector or answer",
            )
            return None

        selector_bindings = _build_role_bindings(
            metric_role=draft.selector_metric_role,
            resolved_by_period=resolved_selector,
            candidate_by_ref=candidate_by_ref,
            value_kind=draft.selector_value_kind,
            measurement_basis=draft.selector_measurement_basis,
            report_scope_value=window.report_scope,
            role="selector",
        )
        answer_bindings = _build_role_bindings(
            metric_role=draft.answer_metric_role,
            resolved_by_period=resolved_answer,
            candidate_by_ref=candidate_by_ref,
            value_kind=draft.answer_type,
            measurement_basis=draft.answer_measurement_basis,
            report_scope_value=window.report_scope,
            role="answer",
        )
        if selector_bindings is None or answer_bindings is None:
            _reject(
                "unit_gate",
                "unit claim or measurement_basis does not match the evidence for at least one role/period",
            )
            return None

        reasoning_steps = build_reasoning_steps(list(window.periods), draft)
        plan = HardP3Plan(
            draft=draft, reasoning_steps=reasoning_steps, calculation_steps=derive_calculation_steps(reasoning_steps)
        )
        graph_error = graph_gate_error(plan)
        if graph_error:
            _reject("graph_gate", graph_error)
            return None
        dependency_error = p3_dependency_error(plan, SCENARIO)
        if dependency_error:
            _reject("graph_gate", dependency_error)
            return None
        coverage_error = p3_coverage_error(selector_bindings, answer_bindings, list(window.periods))
        if coverage_error:
            _reject("coverage_gate", coverage_error)
            return None

        t_compile = time.monotonic()
        all_bindings = [*selector_bindings, *answer_bindings]
        tables_by_ref = {b.table_ref: load_table_for_candidate(candidate_by_ref[b.table_ref]) for b in all_bindings}
        try:
            compiled = P3Compiler().compile(plan, all_bindings, tables_by_ref)
        except P3CompileError as exc:
            _reject("compile", str(exc))
            return None

        csv_map = {ref: candidate_by_ref[ref].csv_path for ref in tables_by_ref}
        execution = execute_query(compiled.pandas_query, csv_map)
        if not execution.ok or not values_match(compiled.expected_answer, execution.actual):
            _reject(
                "execute",
                f"query does not match the compiler: expected={compiled.expected_answer!r} "
                f"actual={execution.actual!r} ok={execution.ok} detail={execution.detail!r}",
            )
            return None
        missing_refs = set(csv_map) - set(execution.accessed_refs)
        if missing_refs:
            _reject("execute", f"query did not access all locked tables: missing_refs={sorted(missing_refs)}")
            return None
        answer = execution.actual
        logger.debug("Hard P3 stage=compile_execute window=%s elapsed=%.3fs", window_key, time.monotonic() - t_compile)

        t_finance = time.monotonic()
        system, user = build_hard_p3_finance_judge_prompt(
            plan, selector_bindings, answer_bindings,
            pandas_query=compiled.pandas_query, actual_result=answer, selected_period=compiled.selected_period,
        )
        finance_judgment, _err = call_structured(
            llm,
            system=system,
            user=user,
            schema=FinancialValidityJudgment,
            max_attempts=1,
            before_call=lambda: _before_call("finance_judge"),
        )
        if finance_judgment is None or not finance_judgment.valid:
            _reject("finance_judge", finance_judgment.reason if finance_judgment else _err)
            return None
        logger.debug("Hard P3 stage=finance_judge window=%s elapsed=%.3fs", window_key, time.monotonic() - t_finance)

        company_name = candidate_by_ref[selector_bindings[0].table_ref].company_name
        ticker = window.ticker
        canonical_answer_unit = answer_bindings[0].normalized_unit
        used_refs = sorted({b.table_ref for b in all_bindings})

        t_question = time.monotonic()
        question_feedback = ""
        question_reject_stage = "question_style"
        for _attempt in range(MAX_ATTEMPTS):
            system, user = build_hard_p3_question_prompt(
                plan,
                company_name=company_name,
                ticker=ticker,
                periods=list(window.periods),
                report_scope=window.report_scope,
                canonical_answer_unit=canonical_answer_unit,
            )
            if question_feedback:
                user = f"{user}\n\nLần trước bị lỗi: {question_feedback}\nHãy viết lại và trả đúng JSON yêu cầu."
            draft_q, _err = call_structured(
                llm,
                system=system,
                user=user,
                schema=QuestionDraft,
                max_attempts=1,
                before_call=lambda: _before_call("question_writer"),
            )
            if draft_q is None:
                question_feedback = _err
                question_reject_stage = "question_style"
                continue
            style_error = question_style_error(draft_q.question)
            if style_error:
                question_feedback = style_error
                question_reject_stage = "question_style"
                continue

            system, user = build_hard_p3_question_judge_prompt(
                plan, draft_q.question,
                company_name=company_name, ticker=ticker,
                periods=list(window.periods), report_scope=window.report_scope,
            )
            question_judgment, _err = call_structured(
                llm,
                system=system,
                user=user,
                schema=HardP3QuestionJudgment,
                max_attempts=1,
                before_call=lambda: _before_call("question_judge"),
            )
            if question_judgment is None or not question_judgment.valid:
                question_feedback = question_judgment.reason if question_judgment else _err
                question_reject_stage = "question_judge"
                continue

            logger.debug("Hard P3 stage=question window=%s elapsed=%.3fs", window_key, time.monotonic() - t_question)
            doc_names = sorted({candidate_by_ref[ref].doc_name for ref in used_refs})
            trace = ExecutionTrace(
                scenario=SCENARIO.name, plan=plan, bindings=all_bindings, accessed_refs=execution.accessed_refs
            )
            logger.debug("Hard P3 execution trace: %s", trace)
            snapshot = budget.snapshot()
            logger.info(
                "Hard P3 success candidate=%d window=%s llm_calls=%d llm_calls_total=%d "
                "elapsed=%.3fs elapsed_total=%.3fs",
                candidate_number,
                window_key,
                snapshot.calls - candidate_calls_started,
                snapshot.calls,
                time.monotonic() - candidate_started,
                snapshot.elapsed_seconds,
            )
            return QARecord(
                id=0,
                question=draft_q.question,
                answer=answer,
                relevant_docs=doc_names,
                relevant_tables=used_refs,
                pandas_query=compiled.pandas_query,
                csv_path=[str(csv_map[ref]) for ref in used_refs],
                difficulty="hard",
            )

        _reject(question_reject_stage, question_feedback or "question writing/judging attempts exhausted")
        return None

    generated = run_parallel_generation(
        work_items=windows, build_record=_build_record, count=count, writer=writer,
        max_workers=max_workers, max_candidates=max_candidates,
        should_stop=budget.exhausted,
    )

    snapshot = budget.snapshot()
    logger.info(
        "Hard P3 run summary: attempted_candidates=%d generated=%d/%d llm_calls=%d "
        "max_llm_calls=%s elapsed=%.3fs budget_exhausted=%s",
        attempted,
        generated,
        count,
        snapshot.calls,
        snapshot.max_calls,
        snapshot.elapsed_seconds,
        budget.exhausted(),
    )
    logger.info("Hard P3 rejection counters: %s", counters.as_dict())
    if generated < count:
        logger.warning("Generated only %d/%d Hard questions (P3).", generated, count)
    return generated
