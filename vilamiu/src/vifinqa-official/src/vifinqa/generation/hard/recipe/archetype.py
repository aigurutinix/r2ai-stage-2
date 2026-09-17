"""Coarse reasoning archetypes derived from the typed DAG.

``operator_signature`` distinguishes exact graph shapes. This module intentionally groups graphs
that ask the same broad kind of analytical question, so batch diversity cannot be satisfied by
minor node or reducer changes alone.
"""

from __future__ import annotations

from vifinqa.generation.hard.recipe.base import ReasoningGraph, StepOutputInput, ValueKind

_TRANSFORMS = frozenset({"growth", "period_difference"})
_REDUCERS = frozenset(
    {"sum", "average", "minimum", "maximum", "cohort_average", "restricted_sum"}
)


def reasoning_archetype(graph: ReasoningGraph) -> str:
    nodes = {node.step_id: node for node in graph.nodes}
    terminal = nodes[graph.terminal_step_id]
    operations = {node.operation for node in graph.nodes}

    if terminal.operation == "scalar_difference" and "cohort_average" in operations:
        return "cohort_comparison"
    if terminal.operation == "scalar_share":
        return "cohort_contribution_share"
    if terminal.operation in {"set_count", "count"}:
        return "intersection_count"

    if terminal.operation == "lookup":
        target_source = _source_node(terminal, 0, nodes)
        winner = _source_node(terminal, 1, nodes)
        if target_source is not None and target_source.operation in _TRANSFORMS:
            return "transformed_rank_transformed_lookup"
        if winner is not None and len(winner.inputs) == 2:
            return "filtered_rank_lookup"
        rank_source = _source_node(winner, 0, nodes) if winner is not None else None
        if rank_source is not None and rank_source.operation in _TRANSFORMS:
            return "transformed_rank_lookup"
        return "rank_lookup"

    if terminal.operation in _REDUCERS:
        if "median" in operations:
            return "derived_threshold_reduce"
        if operations & _TRANSFORMS:
            return "transformed_filter_reduce"
        if "temporal_all" in operations:
            return "persistent_filter_reduce"
        source = _source_node(terminal, 0, nodes)
        if source is not None and source.output_kind == ValueKind.NUMERIC_SERIES_PERIOD:
            return "period_filter_reduce"
        if terminal.operation == "sum":
            return "filtered_sum"
        return "filtered_reduce"

    return f"other:{terminal.operation}"


def _source_node(node, input_index: int, nodes: dict):
    if node is None or input_index >= len(node.inputs):
        return None
    node_input = node.inputs[input_index]
    if not isinstance(node_input, StepOutputInput):
        return None
    return nodes.get(node_input.source_step_id)
