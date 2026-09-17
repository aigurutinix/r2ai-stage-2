
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vifinqa.generation.hard.numbers import VN_NUMBER_PARSER_SOURCE
from vifinqa.generation.hard.recipe.base import (
    EmitContext,
    MetricRoleInput,
    ReasoningGraph,
    StepOutputInput,
)
from vifinqa.generation.hard.recipe.gates import validate_schema
from vifinqa.generation.hard.recipe.registry import get_operation
from vifinqa.generation.panel.catalog import GROUNDED_DERIVED_VALUE_KINDS, get_ratio

_CELL_HELPER_SOURCE = """\
def _cell(table_ref, row_idx, col_idx, scale, is_cost):
    raw = dfs[table_ref].iloc[row_idx, col_idx]
    value = _parse_vn_number(raw) * scale
    return abs(value) if is_cost else value
"""


@dataclass(frozen=True, slots=True)
class CompiledQuery:
    pandas_query: str
    relevant_tables: tuple[str, ...]
    csv_path: dict[str, Path]


def terminal_value_kind(metric_key: str) -> str:
    ratio = get_ratio(metric_key)
    if ratio is not None:
        return ratio.value_kind
    return GROUNDED_DERIVED_VALUE_KINDS.get(metric_key, "money")


def transformed_terminal_value_kind(
    metric_key: str, transform: str | None = None
) -> str:
    """Return the output kind after an optional temporal transform."""
    value_kind = terminal_value_kind(metric_key)
    if transform == "growth":
        return "percentage"
    if transform == "period_difference" and value_kind == "percentage":
        return "percentage_point"
    return value_kind


def format_terminal(value: float, value_kind: str) -> float:
    if value_kind in ("percentage", "percentage_point"):
        return round(value * 100, 2)
    if value_kind == "number":
        return round(value, 2)
    return value


def compile_graph(
    graph: ReasoningGraph,
    *,
    table_ref_to_path: dict[str, Path],
    terminal_metric_key: str,
    terminal_transform: str | None = None,
) -> CompiledQuery:
    ordered = validate_schema(graph)
    context = EmitContext(graph=graph)

    lines: list[str] = [
        VN_NUMBER_PARSER_SOURCE.rstrip("\n"),
        "",
        _CELL_HELPER_SOURCE.rstrip("\n"),
        "",
    ]

    for node in ordered:
        operation = get_operation(node.operation)
        input_exprs: list[str] = []
        for node_input in node.inputs:
            if isinstance(node_input, MetricRoleInput):
                input_exprs.append("")
            else:
                assert isinstance(node_input, StepOutputInput)
                input_exprs.append(context.var_for(node_input.source_step_id))
        lines.append(f"# {node.step_id}: {node.description}")
        lines.extend(operation.emit(node, tuple(input_exprs), context))

    value_kind = transformed_terminal_value_kind(
        terminal_metric_key, terminal_transform
    )
    terminal_var = context.var_for(graph.terminal_step_id)
    if value_kind in ("percentage", "percentage_point"):
        lines.append(f"result = round({terminal_var} * 100, 2)")
    elif value_kind == "number":
        lines.append(f"result = round({terminal_var}, 2)")
    else:
        lines.append(f"result = {terminal_var}")

    pandas_query = "\n".join(lines) + "\n"
    relevant_tables = tuple(sorted(context.used_table_refs))
    csv_path = {ref: table_ref_to_path[ref] for ref in relevant_tables}
    return CompiledQuery(
        pandas_query=pandas_query, relevant_tables=relevant_tables, csv_path=csv_path
    )
