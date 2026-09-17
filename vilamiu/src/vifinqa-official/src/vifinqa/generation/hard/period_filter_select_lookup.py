
from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from threading import Lock

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import scan_catalog
from vifinqa.common.corpus.company_meta import load_company_meta
from vifinqa.generation.table_index.base import TableIndexStore
from vifinqa.generation.budget import LLMCallBudget
from vifinqa.generation.common import (
    CandidateTable,
    call_structured,
    load_table_for_candidate,
    question_style_error,
)
from vifinqa.generation.hard.compiler.period_filter_select_lookup import (
    PeriodFilterSelectLookupCompileError,
    PeriodFilterSelectLookupCompiler,
)
from vifinqa.generation.hard.depth3_schemas import (
    KeyedMetricRoleDraft,
    MultiRoleMappingResult,
    PeriodFilterSelectLookupDraft,
    ResolvedMetricExpression,
    RolePeriodMetricExpression,
)
from vifinqa.generation.hard.numbers import resolve_unit_claim
from vifinqa.generation.hard.schemas import MetricBinding
from vifinqa.generation.parallel import DEFAULT_MAX_WORKERS, run_parallel_generation
from vifinqa.generation.prompts.hard_depth3 import (
    build_depth3_final_judge_prompt,
    build_depth3_mapping_prompt,
    build_depth3_plan_prompt,
    build_depth3_question_prompt,
)
from vifinqa.generation.retrieval.roles import (
    RoleCandidates,
    RoleRequest,
    build_period_inventories,
    retrieve_period_roles,
    select_cross_period_anchor_inventory,
)
from vifinqa.generation.retrieval.time_series import DEFAULT_PERIOD_COUNT, TimeSeriesWindow, enumerate_time_series_windows
from vifinqa.generation.schemas import FinancialValidityJudgment, QARecord, QuestionDraft
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.output.writer import JsonlWriter
from vifinqa.generation.validation.pandas_check import execute_query, values_match

logger = logging.getLogger(__name__)

DEFAULT_MAX_INVENTORY_TABLES = 24
DEFAULT_TABLES_PER_ROLE_PERIOD = 3
DEFAULT_DEPTH3_MAX_LLM_CALLS = 4


def _plan_error(plan: PeriodFilterSelectLookupDraft, inventory_refs: set[str]) -> str:
    if not plan.feasible:
        return plan.reason or "plan reported infeasible"
    for name, role in (
        ("filter", plan.filter_role),
        ("selector", plan.selector_role),
        ("answer", plan.answer_role),
    ):
        if role.anchor_ref not in inventory_refs:
            return f"{name}_role.anchor_ref is not in the inventory: {role.anchor_ref!r}"
        if role.measurement_basis == "unknown":
            return f"{name}_role has not locked measurement_basis"
    return ""


def _candidate_lookup(role_candidates: RoleCandidates) -> dict[str, CandidateTable]:
    result: dict[str, CandidateTable] = {}
    for by_period in role_candidates.values():
        for candidates in by_period.values():
            for candidate in candidates:
                result[candidate.table_ref] = candidate
    return result


def _build_expressions(
    *,
    plan: PeriodFilterSelectLookupDraft,
    mapping_result: MultiRoleMappingResult,
    role_candidates: RoleCandidates,
    periods: tuple[str, ...],
    report_scope: str,
) -> tuple[list[ResolvedMetricExpression] | None, str]:
    if not mapping_result.feasible:
        return None, mapping_result.reason or "mapping reported infeasible"
    resolved_expressions: list[ResolvedMetricExpression] = []
    expressions_by_key: dict[tuple[str, str], list[RolePeriodMetricExpression]] = {}
    for expression in mapping_result.expressions:
        expressions_by_key.setdefault((expression.metric_role, expression.period), []).append(expression)

    roles: tuple[KeyedMetricRoleDraft, ...] = (plan.filter_role, plan.selector_role, plan.answer_role)
    expected_keys = {(role.metric_role, period) for role in roles for period in periods}
    if set(expressions_by_key) != expected_keys:
        missing = sorted(expected_keys - set(expressions_by_key))
        extra = sorted(set(expressions_by_key) - expected_keys)
        return None, f"mapping expression grid sai: missing={missing} extra={extra}"

    for role in roles:
        unit_kinds: set[str] = set()
        for period in periods:
            mapped = expressions_by_key[(role.metric_role, period)]
            if len(mapped) != 1:
                return None, f"role={role.metric_role} period={period} requires exactly one expression; got {len(mapped)}"
            expression = mapped[0]
            if expression.measurement_basis != role.measurement_basis:
                return None, (
                    f"role={role.metric_role} period={period} basis={expression.measurement_basis} "
                    f"differs from plan={role.measurement_basis}"
                )
            candidates_by_ref = {
                candidate.table_ref: candidate for candidate in role_candidates[role.metric_role][period]
            }
            cell_bindings: list[MetricBinding] = []
            expression_unit_kinds: set[str] = set()
            for cell in expression.cells:
                candidate = candidates_by_ref.get(cell.table_ref)
                if candidate is None:
                    return None, (
                        f"role={role.metric_role} period={period} selected a table_ref outside the shortlist: {cell.table_ref}"
                    )
                resolved = resolve_unit_claim(
                    cell.unit,
                    csv_header=candidate.csv_header,
                    unit_snippets=list(candidate.unit_snippets),
                    value_kind=role.value_kind,
                )
                if resolved is None:
                    return None, (
                        f"role={role.metric_role} period={period} has an invalid unit claim: "
                        f"table_ref={candidate.table_ref} kind={cell.unit.kind} source={cell.unit.source} "
                        f"evidence={cell.unit.evidence!r} header={candidate.csv_header!r} "
                        f"unit_snippets={candidate.unit_snippets!r}"
                    )
                expression_unit_kinds.add(cell.unit.kind)
                cell_bindings.append(
                    MetricBinding(
                        metric_role=role.metric_role,
                        table_ref=candidate.table_ref,
                        ticker=candidate.ticker,
                        period=period,
                        report_scope=report_scope,  # type: ignore[arg-type]
                        row_label=cell.row_label,
                        column_label=cell.column_label,
                        raw_unit=resolved.raw_evidence or cell.unit.kind,
                        scale=resolved.scale,
                        normalized_unit=resolved.normalized_unit,
                        measurement_basis=expression.measurement_basis,
                        evidence=resolved.raw_evidence,
                    )
                )
            if len(expression_unit_kinds) != 1:
                return None, (
                    f"role={role.metric_role} period={period} expression mixes UnitKind values: "
                    f"{sorted(expression_unit_kinds)}"
                )
            unit_kinds.update(expression_unit_kinds)
            resolved_expressions.append(
                ResolvedMetricExpression(
                    metric_role=role.metric_role,
                    period=period,
                    operation=expression.operation,
                    cells=cell_bindings,
                )
            )
        if len(unit_kinds) != 1:
            return None, f"role={role.metric_role} has inconsistent UnitKind values across periods: {sorted(unit_kinds)}"
    return resolved_expressions, ""


def generate_period_filter_select_lookup(
    *,
    settings: Settings,
    llm: ChatLLM,
    index_store: TableIndexStore,
    count: int,
    out_path: Path,
    seed: int | None = None,
    period_count: int = DEFAULT_PERIOD_COUNT,
    max_inventory_tables: int = DEFAULT_MAX_INVENTORY_TABLES,
    max_tables_per_role_period: int = DEFAULT_TABLES_PER_ROLE_PERIOD,
    max_workers: int = DEFAULT_MAX_WORKERS,
    max_candidates: int | None = None,
    max_llm_calls: int | None = None,
) -> int:
    if max_workers != 1:
        raise ValueError("The depth3 walking skeleton supports only max_workers=1")
    effective_max_candidates = 1 if max_candidates is None else max_candidates
    effective_max_llm_calls = DEFAULT_DEPTH3_MAX_LLM_CALLS if max_llm_calls is None else max_llm_calls

    all_docs = scan_catalog(settings.data_root)
    companies = load_company_meta(settings.company_meta_path)
    windows = enumerate_time_series_windows(all_docs, period_count=max(period_count, 3))
    random.Random(seed).shuffle(windows)
    writer = JsonlWriter(out_path)
    budget = LLMCallBudget(effective_max_llm_calls)
    lock = Lock()
    attempted = 0
    rejection_counters: dict[str, int] = {}

    def _build_record(window: TimeSeriesWindow) -> QARecord | None:
        nonlocal attempted
        with lock:
            attempted += 1
            candidate_number = attempted
        window_key = (window.ticker, window.report_scope, window.periods)
        candidate_started = time.monotonic()
        candidate_calls_started = budget.snapshot().calls

        def _before_call(stage: str) -> None:
            budget.before_call(candidate=window_key, stage=stage)

        def _reject(stage: str, reason: str) -> None:
            with lock:
                rejection_counters[stage] = rejection_counters.get(stage, 0) + 1
            snapshot = budget.snapshot()
            logger.info(
                "Depth3 reject candidate=%d window=%s reject_stage=%s reason=%s "
                "llm_calls=%d llm_calls_total=%d elapsed=%.3fs",
                candidate_number,
                window_key,
                stage,
                " ".join(reason.split())[:1000],
                snapshot.calls - candidate_calls_started,
                snapshot.calls,
                time.monotonic() - candidate_started,
            )

        inventories = build_period_inventories(window, companies)
        if any(not inventories.get(period) for period in window.periods):
            _reject("inventory", "at least one period has no eligible tables")
            return None
        inventory = select_cross_period_anchor_inventory(
            inventories,
            anchor_period=window.anchor_period,
            max_tables=max_inventory_tables,
        )
        if not inventory:
            _reject("inventory", f"anchor inventory contains only {len(inventory)} tables")
            return None

        system, user = build_depth3_plan_prompt(inventory, periods=window.periods)
        plan, err = call_structured(
            llm,
            system=system,
            user=user,
            schema=PeriodFilterSelectLookupDraft,
            max_attempts=1,
            before_call=lambda: _before_call("plan"),
        )
        if plan is None:
            _reject("plan", err)
            return None
        plan_error = _plan_error(plan, {descriptor.table_ref for descriptor in inventory})
        if plan_error:
            _reject("plan", plan_error)
            return None
        logger.debug(
            "Depth3 plan window=%s filter=%r selector=%r answer=%r prompt_chars=%d",
            window_key,
            plan.filter_role.concept_name,
            plan.selector_role.concept_name,
            plan.answer_role.concept_name,
            len(system) + len(user),
        )

        inventory_by_ref = {descriptor.table_ref: descriptor for descriptor in inventory}
        requests = [
            RoleRequest(
                metric_role=role.metric_role,
                concept_name=role.concept_name,
                concept_formula=role.concept_formula,
                anchor_ref=role.anchor_ref,
                anchor_text=(
                    f"{inventory_by_ref[role.anchor_ref].table_labels}. "
                    f"{inventory_by_ref[role.anchor_ref].anchor_context}"
                ),
            )
            for role in (plan.filter_role, plan.selector_role, plan.answer_role)
        ]
        role_candidates = retrieve_period_roles(
            window=window,
            requests=requests,
            index_store=index_store,
            companies=companies,
            max_tables_per_role_period=max_tables_per_role_period,
        )
        if role_candidates is None:
            _reject("role_retrieval", "could not build candidates for every role/period")
            return None
        logger.debug(
            "Depth3 role refs window=%s refs=%s",
            window_key,
            {
                role: {period: [candidate.table_ref for candidate in candidates] for period, candidates in by_period.items()}
                for role, by_period in role_candidates.items()
            },
        )
        candidate_by_ref = _candidate_lookup(role_candidates)

        system, user = build_depth3_mapping_prompt(role_candidates, plan)
        mapping_result, err = call_structured(
            llm,
            system=system,
            user=user,
            schema=MultiRoleMappingResult,
            max_attempts=1,
            before_call=lambda: _before_call("mapping"),
        )
        if mapping_result is None:
            _reject("mapping", err)
            return None
        expressions, binding_error = _build_expressions(
            plan=plan,
            mapping_result=mapping_result,
            role_candidates=role_candidates,
            periods=window.periods,
            report_scope=window.report_scope,
        )
        if expressions is None:
            _reject("mapping", binding_error)
            return None

        bindings = [binding for expression in expressions for binding in expression.cells]
        tables_by_ref = {
            binding.table_ref: load_table_for_candidate(candidate_by_ref[binding.table_ref]) for binding in bindings
        }
        try:
            compiled = PeriodFilterSelectLookupCompiler().compile(plan, expressions, tables_by_ref)
        except PeriodFilterSelectLookupCompileError as exc:
            _reject("compile", str(exc))
            return None
        csv_map = {ref: candidate_by_ref[ref].csv_path for ref in tables_by_ref}
        execution = execute_query(compiled.pandas_query, csv_map)
        if (
            not execution.ok
            or not values_match(compiled.expected_answer, execution.actual)
            or not isinstance(execution.actual, (int, float, bool))
            or (isinstance(execution.actual, float) and not math.isfinite(execution.actual))
        ):
            _reject(
                "execute",
                f"expected={compiled.expected_answer!r} actual={execution.actual!r} "
                f"ok={execution.ok} detail={execution.detail}",
            )
            return None
        missing_refs = set(csv_map) - set(execution.accessed_refs)
        if missing_refs:
            _reject("execute", f"query did not access all binding refs: {sorted(missing_refs)}")
            return None
        answer = execution.actual

        company_name = next(iter(candidate_by_ref.values())).company_name
        answer_unit = next(
            binding.normalized_unit for binding in bindings if binding.metric_role == plan.answer_role.metric_role
        )
        system, user = build_depth3_question_prompt(
            plan,
            company_name=company_name,
            ticker=window.ticker,
            periods=window.periods,
            answer_unit=answer_unit,
        )
        question_draft, err = call_structured(
            llm,
            system=system,
            user=user,
            schema=QuestionDraft,
            max_attempts=1,
            before_call=lambda: _before_call("question"),
        )
        if question_draft is None:
            _reject("question", err)
            return None
        style_error = question_style_error(question_draft.question)
        if style_error:
            _reject("question", style_error)
            return None

        system, user = build_depth3_final_judge_prompt(
            plan,
            expressions,
            question=question_draft.question,
            pandas_query=compiled.pandas_query,
            answer=answer,
            periods=window.periods,
            company_name=company_name,
        )
        judgment, err = call_structured(
            llm,
            system=system,
            user=user,
            schema=FinancialValidityJudgment,
            max_attempts=1,
            before_call=lambda: _before_call("final_judge"),
        )
        if judgment is None or not judgment.valid:
            _reject("final_judge", judgment.reason if judgment else err)
            return None

        snapshot = budget.snapshot()
        logger.info(
            "Depth3 success candidate=%d window=%s allowed_periods=%s selected_period=%s "
            "llm_calls=%d llm_calls_total=%d elapsed=%.3fs",
            candidate_number,
            window_key,
            sorted(compiled.allowed_periods, key=int),
            compiled.selected_period,
            snapshot.calls - candidate_calls_started,
            snapshot.calls,
            time.monotonic() - candidate_started,
        )
        used_refs = sorted(tables_by_ref)
        return QARecord(
            id=0,
            question=question_draft.question,
            answer=answer,
            relevant_docs=sorted({candidate_by_ref[ref].doc_name for ref in used_refs}),
            relevant_tables=used_refs,
            pandas_query=compiled.pandas_query,
            csv_path=[str(csv_map[ref]) for ref in used_refs],
            difficulty="hard",
        )

    generated = run_parallel_generation(
        work_items=windows,
        build_record=_build_record,
        count=count,
        writer=writer,
        max_workers=max_workers,
        max_candidates=effective_max_candidates,
        should_stop=budget.exhausted,
    )
    snapshot = budget.snapshot()
    logger.info(
        "Depth3 run summary: attempted_candidates=%d generated=%d/%d llm_calls=%d "
        "max_llm_calls=%d elapsed=%.3fs rejection_counters=%s",
        attempted,
        generated,
        count,
        snapshot.calls,
        effective_max_llm_calls,
        snapshot.elapsed_seconds,
        rejection_counters,
    )
    if generated < count:
        logger.warning("Generated only %d/%d Hard depth3 questions.", generated, count)
    return generated
