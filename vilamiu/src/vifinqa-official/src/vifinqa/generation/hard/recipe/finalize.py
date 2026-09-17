
from __future__ import annotations

from pathlib import Path

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.hard.recipe.audit.base import DependencyAuditor
from vifinqa.generation.hard.recipe.audit.cache import AuditCache
from vifinqa.generation.hard.recipe.audit.service import (
    AuditStats,
    audit_dependency_closure,
    rebuild_validated_graph,
)
from vifinqa.generation.hard.recipe.base import CandidateRejected, ReasoningGraph
from vifinqa.generation.hard.recipe.compiler import (
    CompiledQuery,
    compile_graph,
    format_terminal,
    transformed_terminal_value_kind,
)
from vifinqa.generation.hard.recipe.evaluator import EvaluationTrace, evaluate_graph
from vifinqa.generation.hard.recipe.gates import compute_hardness, hardness_gate_error
from vifinqa.generation.hard.recipe.runtime_gates import enforce_runtime_gates
from vifinqa.generation.validation.pandas_check import check_answer


def finalize_candidate(
    *,
    provisional_graph: ReasoningGraph,
    terminal_metric_key: str,
    terminal_transform: str | None = None,
    docs_by_name: dict[str, DocumentRef],
    company_meta: dict[str, CompanyInfo],
    table_ref_to_path: dict[str, Path],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    enforce_objective_gates: bool = False,
) -> tuple[ReasoningGraph, EvaluationTrace, CompiledQuery, float, AuditStats]:
    audited = audit_dependency_closure(
        provisional_graph,
        docs_by_name=docs_by_name,
        company_meta=company_meta,
        cache=audit_cache,
        auditor=auditor,
    )
    validated_graph = rebuild_validated_graph(provisional_graph, audited)

    return finalize_validated_candidate(
        validated_graph=validated_graph,
        terminal_metric_key=terminal_metric_key,
        terminal_transform=terminal_transform,
        table_ref_to_path=table_ref_to_path,
        audit_stats=audited.stats,
        enforce_objective_gates=enforce_objective_gates,
    )


def finalize_validated_candidate(
    *,
    validated_graph: ReasoningGraph,
    terminal_metric_key: str,
    terminal_transform: str | None = None,
    table_ref_to_path: dict[str, Path],
    audit_stats: AuditStats,
    enforce_objective_gates: bool = False,
) -> tuple[ReasoningGraph, EvaluationTrace, CompiledQuery, float, AuditStats]:
    """Evaluate/compile/replay a graph whose complete dependency closure was already audited."""

    trace = evaluate_graph(validated_graph)
    if enforce_objective_gates:
        enforce_runtime_gates(validated_graph, trace)

    hardness = compute_hardness(validated_graph)
    hardness_error = hardness_gate_error(hardness)
    if hardness_error:
        raise CandidateRejected(f"Hardness gate: {hardness_error}")

    compiled = compile_graph(
        validated_graph,
        table_ref_to_path=table_ref_to_path,
        terminal_metric_key=terminal_metric_key,
        terminal_transform=terminal_transform,
    )
    formatted_answer = format_terminal(
        trace.expected,
        transformed_terminal_value_kind(terminal_metric_key, terminal_transform),
    )
    result = check_answer(compiled.pandas_query, compiled.csv_path, formatted_answer)
    if not result.ok:
        raise CandidateRejected(
            f"BUG: pandas_query does not match the evaluator: {result.detail}"
        )

    return validated_graph, trace, compiled, formatted_answer, audit_stats


def company_names(
    entities: tuple[str, ...], company_meta: dict[str, CompanyInfo]
) -> tuple[str, ...]:
    return tuple(company_meta[e].name if e in company_meta else e for e in entities)


def unit_label(value_kind: str) -> str:
    return {
        "percentage": "%",
        "percentage_point": "điểm phần trăm",
        "number": "lần",
        "money": "đồng",
    }[value_kind]
