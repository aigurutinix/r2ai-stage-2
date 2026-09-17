"""Runtime non-triviality and effective-dependency gates.

These checks run after JIT materialization and graph evaluation. They use only
typed operation semantics plus runtime node outputs; no question text, frame id,
ticker, year, or expected answer is consulted.
"""

from __future__ import annotations

from collections.abc import Iterable

from vifinqa.generation.hard.recipe.base import (
    CandidateRejected,
    ReasoningGraph,
    ReasoningNode,
    StepOutputInput,
    ValueKind,
)
from vifinqa.generation.hard.recipe.evaluator import EvaluationTrace

MIN_DERIVED_THRESHOLD_POPULATION = 5
MIN_DERIVED_SELECTED_SET = 2
MIN_COHORT_AVERAGE_MEMBERS = 2


class RuntimeGateReason:
    EMPTY_FULL_PREDICATE = "empty_or_full_predicate"
    REDUNDANT_PREDICATE = "redundant_predicate_or_intersection_input"
    THRESHOLD_POPULATION_TOO_SMALL = "median_or_average_threshold_population_too_small"
    DERIVED_SELECTED_SET_TOO_SMALL = "derived_selected_set_too_small"
    COHORT_TOO_SMALL = "cohort_too_small"
    RESTRICTED_RANK_WINNER_UNCHANGED = "restricted_rank_winner_unchanged"
    INEFFECTIVE_ADAPTIVE_EDGE = "ineffective_adaptive_edge"


def enforce_runtime_gates(graph: ReasoningGraph, trace: EvaluationTrace) -> None:
    by_id = {node.step_id: node for node in graph.nodes}
    terminal_path = _terminal_path(graph)
    for node in graph.nodes:
        if node.step_id not in terminal_path:
            continue
        if node.operation == "predicate_set":
            _check_predicate_set(node, by_id, trace)
        elif node.operation == "set_intersection":
            _check_set_intersection(node, by_id, trace)
        elif node.operation in {"median", "average_threshold"}:
            _check_threshold_population(node, by_id, trace)
        elif node.operation == "filter":
            _check_filter(node, by_id, trace)
        elif node.operation == "cohort_average":
            _check_cohort_average(node, by_id, trace)
        elif node.operation in {"argmax", "argmin"} and len(node.inputs) == 2:
            _check_restricted_rank_effect(
                node, by_id, trace, pick_max=node.operation == "argmax"
            )
        elif (
            node.operation in {"sum", "average", "minimum", "maximum", "restricted_sum"}
            and len(node.inputs) == 2
        ):
            _check_restricted_reducer_effect(node, by_id, trace)
        elif node.operation == "lookup":
            _check_lookup_effect(node, by_id, trace)


def _terminal_path(graph: ReasoningGraph) -> set[str]:
    by_id = {node.step_id: node for node in graph.nodes}
    reachable: set[str] = set()

    def visit(step_id: str) -> None:
        if step_id in reachable:
            return
        reachable.add(step_id)
        for item in by_id[step_id].inputs:
            if isinstance(item, StepOutputInput):
                visit(item.source_step_id)

    visit(graph.terminal_step_id)
    return reachable


def _step_input(node: ReasoningNode, index: int) -> StepOutputInput:
    item = node.inputs[index]
    assert isinstance(item, StepOutputInput)
    return item


def _input_output(node: ReasoningNode, index: int, trace: EvaluationTrace) -> object:
    return trace.node_outputs[_step_input(node, index).source_step_id]


def _universe(
    node: ReasoningNode, trace: EvaluationTrace, by_id: dict[str, ReasoningNode]
) -> frozenset:
    if node.output_kind == ValueKind.ENTITY_SET:
        return frozenset(node.domain.entities)
    if node.output_kind == ValueKind.PERIOD_SET:
        return frozenset(node.domain.periods)
    if node.inputs and isinstance(node.inputs[0], StepOutputInput):
        source_output = trace.node_outputs[node.inputs[0].source_step_id]
        if isinstance(source_output, dict):
            return frozenset(source_output.keys())
    del by_id
    return frozenset()


def _reject(reason: str, detail: str) -> None:
    raise CandidateRejected(f"{reason}: {detail}")


def _proper_non_empty(selected: Iterable, universe: Iterable) -> bool:
    selected_set = frozenset(selected)
    universe_set = frozenset(universe)
    return bool(selected_set) and selected_set != universe_set


def _check_predicate_set(
    node: ReasoningNode, by_id: dict[str, ReasoningNode], trace: EvaluationTrace
) -> None:
    if node.params.get("eligibility_gate") is True:
        return
    series = _input_output(node, 0, trace)
    output = trace.node_outputs[node.step_id]
    assert isinstance(series, dict)
    assert isinstance(output, (set, frozenset))
    if not _proper_non_empty(output, series.keys()):
        _reject(
            RuntimeGateReason.EMPTY_FULL_PREDICATE,
            f"{node.step_id} selected {len(output)} of {len(series)}",
        )
    del by_id


def _check_set_intersection(
    node: ReasoningNode, by_id: dict[str, ReasoningNode], trace: EvaluationTrace
) -> None:
    input_sets = [
        frozenset(_input_output(node, index, trace))
        for index in range(len(node.inputs))
    ]
    eligibility_indexes = {
        index
        for index in range(len(node.inputs))
        if by_id[_step_input(node, index).source_step_id].params.get("eligibility_gate")
        is True
    }
    output = frozenset(trace.node_outputs[node.step_id])
    universe = _universe(node, trace, by_id)
    for index, selected in enumerate(input_sets):
        if index in eligibility_indexes:
            continue
        if not _proper_non_empty(selected, universe):
            _reject(
                RuntimeGateReason.EMPTY_FULL_PREDICATE,
                f"{node.step_id} input {index} is empty/full",
            )
    if not _proper_non_empty(output, universe):
        _reject(
            RuntimeGateReason.EMPTY_FULL_PREDICATE,
            f"{node.step_id} output is empty/full",
        )
    for index in range(len(input_sets)):
        if index in eligibility_indexes:
            continue
        without = [selected for i, selected in enumerate(input_sets) if i != index]
        if without and frozenset.intersection(*without) == output:
            _reject(
                RuntimeGateReason.REDUNDANT_PREDICATE,
                f"{node.step_id} input {index} does not change intersection",
            )


def _check_threshold_population(
    node: ReasoningNode, by_id: dict[str, ReasoningNode], trace: EvaluationTrace
) -> None:
    series = _input_output(node, 0, trace)
    assert isinstance(series, dict)
    if len(series) < MIN_DERIVED_THRESHOLD_POPULATION:
        _reject(
            RuntimeGateReason.THRESHOLD_POPULATION_TOO_SMALL,
            f"{node.step_id} population={len(series)} < {MIN_DERIVED_THRESHOLD_POPULATION}",
        )
    del by_id


def _check_filter(
    node: ReasoningNode, by_id: dict[str, ReasoningNode], trace: EvaluationTrace
) -> None:
    series = _input_output(node, 0, trace)
    output = frozenset(trace.node_outputs[node.step_id])
    assert isinstance(series, dict)
    if not _proper_non_empty(output, series.keys()):
        _reject(
            RuntimeGateReason.EMPTY_FULL_PREDICATE,
            f"{node.step_id} selected {len(output)} of {len(series)}",
        )
    if node.params.get("threshold_source") == "derived":
        if len(output) < MIN_DERIVED_SELECTED_SET:
            _reject(
                RuntimeGateReason.DERIVED_SELECTED_SET_TOO_SMALL,
                f"{node.step_id} selected={len(output)} < {MIN_DERIVED_SELECTED_SET}",
            )
        if not _has_alternative_threshold_cut(series, node.params["operator"], output):
            _reject(
                RuntimeGateReason.INEFFECTIVE_ADAPTIVE_EDGE,
                f"{node.step_id} derived threshold has no alternative cut changing selected set",
            )
    del by_id


def _check_cohort_average(
    node: ReasoningNode, by_id: dict[str, ReasoningNode], trace: EvaluationTrace
) -> None:
    pool = trace.contributing_keys[node.step_id]
    if len(pool) < MIN_COHORT_AVERAGE_MEMBERS:
        _reject(
            RuntimeGateReason.COHORT_TOO_SMALL,
            f"{node.step_id} cohort members={len(pool)} < {MIN_COHORT_AVERAGE_MEMBERS}",
        )
    del by_id


def _check_restricted_rank_effect(
    node: ReasoningNode,
    by_id: dict[str, ReasoningNode],
    trace: EvaluationTrace,
    *,
    pick_max: bool,
) -> None:
    series = _input_output(node, 0, trace)
    restricted_winner = trace.node_outputs[node.step_id]
    assert isinstance(series, dict)
    unrestricted_winner = _unique_winner(series, pick_max=pick_max)
    if unrestricted_winner == restricted_winner:
        _reject(
            RuntimeGateReason.RESTRICTED_RANK_WINNER_UNCHANGED,
            f"{node.step_id} restricted and unrestricted winner are both {restricted_winner!r}",
        )
    del by_id


def _check_restricted_reducer_effect(
    node: ReasoningNode, by_id: dict[str, ReasoningNode], trace: EvaluationTrace
) -> None:
    series = _input_output(node, 0, trace)
    restricted_keys = trace.contributing_keys[node.step_id]
    assert isinstance(series, dict)
    if len(node.inputs) == 2 and len(restricted_keys) < MIN_DERIVED_SELECTED_SET:
        _reject(
            RuntimeGateReason.DERIVED_SELECTED_SET_TOO_SMALL,
            f"{node.step_id} restricted selected={len(restricted_keys)} < {MIN_DERIVED_SELECTED_SET}",
        )
    if restricted_keys == frozenset(series.keys()):
        _reject(
            RuntimeGateReason.INEFFECTIVE_ADAPTIVE_EDGE,
            f"{node.step_id} restricted pool equals full population",
        )
    del by_id


def _check_lookup_effect(
    node: ReasoningNode, by_id: dict[str, ReasoningNode], trace: EvaluationTrace
) -> None:
    series = _input_output(node, 0, trace)
    key = _input_output(node, 1, trace)
    assert isinstance(series, dict)
    if key not in series:
        _reject(
            RuntimeGateReason.INEFFECTIVE_ADAPTIVE_EDGE,
            f"{node.step_id} key {key!r} not covered",
        )
    if len(series) < 2:
        _reject(
            RuntimeGateReason.INEFFECTIVE_ADAPTIVE_EDGE,
            f"{node.step_id} lookup has no alternative covered key",
        )
    del by_id


def _unique_winner(series: dict, *, pick_max: bool) -> object | None:
    if not series:
        return None
    best = max(series.values()) if pick_max else min(series.values())
    winners = [key for key, value in series.items() if value == best]
    return winners[0] if len(winners) == 1 else None


def _apply_operator(operator: str, value: float, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == ">=":
        return value >= threshold
    if operator == "<":
        return value < threshold
    if operator == "<=":
        return value <= threshold
    raise CandidateRejected(f"unsupported_filter_operator: {operator!r}")


def _has_alternative_threshold_cut(
    series: dict, operator: object, current: frozenset
) -> bool:
    if not isinstance(operator, str):
        return False
    values = sorted({float(value) for value in series.values()})
    if len(values) < 2:
        return False
    thresholds = [
        (left + right) / 2 for left, right in zip(values, values[1:], strict=False)
    ]
    thresholds.extend(values)
    for threshold in thresholds:
        selected = frozenset(
            key
            for key, value in series.items()
            if _apply_operator(operator, float(value), threshold)
        )
        if not _proper_non_empty(selected, series.keys()):
            continue
        if len(selected) < MIN_DERIVED_SELECTED_SET:
            continue
        if selected != current:
            return True
    return False
