
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from vifinqa.generation.hard.recipe.base import (
    GraphError,
    MetricRoleInput,
    OperationError,
    ReasoningGraph,
    StepOutputInput,
    ValueKind,
)
from vifinqa.generation.hard.recipe.gates import validate_schema
from vifinqa.generation.hard.recipe.registry import get_operation

_ADAPTIVE_KEY_KINDS = frozenset(
    {ValueKind.ENTITY_KEY, ValueKind.PERIOD_KEY, ValueKind.ENTITY_PERIOD_KEY}
)
_ADAPTIVE_SET_KINDS = frozenset({ValueKind.ENTITY_SET, ValueKind.PERIOD_SET})


@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    expected: float
    node_outputs: Mapping[str, object]
    runtime_thresholds: Mapping[str, float]
    selected_keys: Mapping[str, object]
    survivor_sets: Mapping[str, frozenset]
    contributing_keys: Mapping[str, frozenset]


class EvaluationError(ValueError):
    pass


def evaluate_graph(graph: ReasoningGraph) -> EvaluationTrace:
    ordered = validate_schema(graph)

    node_outputs: dict[str, object] = {}
    runtime_thresholds: dict[str, float] = {}
    selected_keys: dict[str, object] = {}
    survivor_sets: dict[str, frozenset] = {}
    contributing_keys: dict[str, frozenset] = {}

    for node in ordered:
        operation = get_operation(node.operation)
        resolved_inputs: list[object] = []
        for node_input in node.inputs:
            if isinstance(node_input, MetricRoleInput):
                resolved_inputs.append(node_input)
            else:
                assert isinstance(node_input, StepOutputInput)
                resolved_inputs.append(node_outputs[node_input.source_step_id])

        try:
            output = operation.evaluate(node, tuple(resolved_inputs))
        except OperationError as exc:
            raise EvaluationError(f"{node.step_id}: {exc}") from exc

        node_outputs[node.step_id] = output
        contributing_keys[node.step_id] = _contributing_keys(node, resolved_inputs, output)

        if node.output_kind == ValueKind.THRESHOLD and isinstance(output, (int, float)):
            runtime_thresholds[node.step_id] = float(output)
        if node.output_kind in _ADAPTIVE_KEY_KINDS:
            selected_keys[node.step_id] = output
        if node.output_kind in _ADAPTIVE_SET_KINDS and isinstance(output, (frozenset, set)):
            survivor_sets[node.step_id] = frozenset(output)

    terminal_output = node_outputs[graph.terminal_step_id]
    if not isinstance(terminal_output, (int, float)) or isinstance(terminal_output, bool):
        raise GraphError(f"terminal must be a finite int or float; got {terminal_output!r}")

    return EvaluationTrace(
        expected=float(terminal_output),
        node_outputs=node_outputs,
        runtime_thresholds=runtime_thresholds,
        selected_keys=selected_keys,
        survivor_sets=survivor_sets,
        contributing_keys=contributing_keys,
    )


def _contributing_keys(node, inputs: list[object], output: object) -> frozenset:
    if node.operation == "extract" and isinstance(output, dict):
        return frozenset(output.keys())
    if isinstance(output, (set, frozenset)):
        return frozenset(output)
    if node.operation in {"argmax", "argmin"}:
        return frozenset((output,))
    if node.operation == "lookup":
        return frozenset((inputs[1],))
    if node.operation in {
        "sum",
        "average",
        "minimum",
        "maximum",
        "count",
        "restricted_sum",
        "cohort_average",
    }:
        series = inputs[0]
        if isinstance(series, dict):
            if len(inputs) == 2 and isinstance(inputs[1], (set, frozenset)):
                return frozenset(key for key in series if key in inputs[1])
            return frozenset(series.keys())
    result: set = set()
    for value in inputs:
        if isinstance(value, dict):
            result.update(value.keys())
        elif isinstance(value, (set, frozenset)):
            result.update(value)
        elif isinstance(value, (str, tuple)):
            result.add(value)
    return frozenset(result)
