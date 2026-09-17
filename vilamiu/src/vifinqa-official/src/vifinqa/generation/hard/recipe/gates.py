
from __future__ import annotations

from dataclasses import dataclass

from vifinqa.generation.hard.recipe.base import (
    ADAPTIVE_VALUE_KINDS,
    GraphError,
    MetricRoleInput,
    ReasoningGraph,
    ReasoningNode,
    StepOutputInput,
    ValueKind,
)
from vifinqa.generation.hard.recipe.registry import get_operation


def _topo_order(graph: ReasoningGraph) -> list[ReasoningNode]:
    """Run Kahn's algorithm and raise ``GraphError`` when a cycle exists."""
    by_id = {node.step_id: node for node in graph.nodes}
    if len(by_id) != len(graph.nodes):
        raise GraphError("duplicate step_id in graph")

    outputs = [node.output for node in graph.nodes]
    if len(set(outputs)) != len(outputs):
        raise GraphError("duplicate output symbol in graph")

    dependents: dict[str, list[str]] = {node.step_id: [] for node in graph.nodes}
    indegree: dict[str, int] = {node.step_id: 0 for node in graph.nodes}
    for node in graph.nodes:
        for node_input in node.inputs:
            if isinstance(node_input, StepOutputInput):
                if node_input.source_step_id not in by_id:
                    raise GraphError(
                        f"{node.step_id}: references a missing step_id: {node_input.source_step_id}"
                    )
                dependents[node_input.source_step_id].append(node.step_id)
                indegree[node.step_id] += 1

    queue = [step_id for step_id, degree in indegree.items() if degree == 0]
    ordered: list[str] = []
    queue_index = 0
    while queue_index < len(queue):
        step_id = queue[queue_index]
        queue_index += 1
        ordered.append(step_id)
        for dependent in dependents[step_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)

    if len(ordered) != len(graph.nodes):
        raise GraphError("graph contains a cycle")
    return [by_id[step_id] for step_id in ordered]


def validate_schema(graph: ReasoningGraph) -> list[ReasoningNode]:
    if not graph.nodes:
        raise GraphError("graph is empty")
    ordered = _topo_order(graph)

    terminal = next(
        (n for n in graph.nodes if n.step_id == graph.terminal_step_id), None
    )
    if terminal is None:
        raise GraphError(f"terminal_step_id does not exist: {graph.terminal_step_id}")
    if terminal.output_kind != ValueKind.NUMERIC_SCALAR:
        raise GraphError(f"terminal must be NumericScalar; got {terminal.output_kind}")

    reachable_to_terminal: set[str] = set()

    def mark(step_id: str) -> None:
        if step_id in reachable_to_terminal:
            return
        reachable_to_terminal.add(step_id)
        node = next(n for n in graph.nodes if n.step_id == step_id)
        for node_input in node.inputs:
            if isinstance(node_input, StepOutputInput):
                mark(node_input.source_step_id)

    mark(graph.terminal_step_id)
    dead = [n.step_id for n in graph.nodes if n.step_id not in reachable_to_terminal]
    if dead:
        raise GraphError(f"nodes do not contribute to the terminal (dead nodes): {dead}")

    for node in graph.nodes:
        for node_input in node.inputs:
            if isinstance(node_input, MetricRoleInput):
                continue
            source_node = next(
                n for n in graph.nodes if n.step_id == node_input.source_step_id
            )
            if source_node.output_kind != node_input.expected_kind:
                raise GraphError(
                    f"{node.step_id}: input references {node_input.source_step_id} has output_kind "
                    f"{source_node.output_kind}, not expected_kind {node_input.expected_kind}"
                )
        operation = get_operation(node.operation)
        operation.validate(node, graph)

    return ordered


@dataclass(frozen=True, slots=True)
class HardnessMetrics:
    reasoning_depth: int
    adaptive_edges: int


def compute_hardness(graph: ReasoningGraph) -> HardnessMetrics:
    by_id = {node.step_id: node for node in graph.nodes}

    depth_cache: dict[str, int] = {}
    adaptive_cache: dict[str, int] = {}

    def depth_and_adaptive(step_id: str) -> tuple[int, int]:
        if step_id in depth_cache:
            return depth_cache[step_id], adaptive_cache[step_id]
        node = by_id[step_id]
        best_depth = 0
        best_adaptive = 0
        for node_input in node.inputs:
            if not isinstance(node_input, StepOutputInput):
                continue
            source_node = by_id[node_input.source_step_id]
            upstream_depth, upstream_adaptive = depth_and_adaptive(
                node_input.source_step_id
            )
            edge_adaptive = 1 if source_node.output_kind in ADAPTIVE_VALUE_KINDS else 0
            candidate_depth = upstream_depth
            candidate_adaptive = upstream_adaptive + edge_adaptive
            best_depth = max(best_depth, candidate_depth)
            best_adaptive = max(best_adaptive, candidate_adaptive)
        if node.operation != "extract":
            best_depth += 1
        depth_cache[step_id] = best_depth
        adaptive_cache[step_id] = best_adaptive
        return best_depth, best_adaptive

    depth, adaptive_edges = depth_and_adaptive(graph.terminal_step_id)
    return HardnessMetrics(reasoning_depth=depth, adaptive_edges=adaptive_edges)


HARD_BASIC_MIN_DEPTH = 2
HARD_BASIC_MIN_ADAPTIVE = 1
DEEP_HARD_MIN_DEPTH = 3
DEEP_HARD_MIN_ADAPTIVE = 2


def hardness_gate_error(metrics: HardnessMetrics) -> str:
    if metrics.reasoning_depth < HARD_BASIC_MIN_DEPTH:
        return f"reasoning_depth={metrics.reasoning_depth} < {HARD_BASIC_MIN_DEPTH} (basic Hard)"
    if metrics.adaptive_edges < HARD_BASIC_MIN_ADAPTIVE:
        return f"adaptive_edges={metrics.adaptive_edges} < {HARD_BASIC_MIN_ADAPTIVE} (basic Hard)"
    return ""


def deep_hardness_gate_error(metrics: HardnessMetrics) -> str:
    """Return an error unless ``metrics`` satisfy the Deep Hard contract.

    Keep this separate from :func:`hardness_gate_error`: legacy Hard scenarios and
    lower-level graph tests still intentionally exercise the basic Hard boundary,
    while production Hard Cube checkpoints can opt into the stricter contract.
    """
    if metrics.reasoning_depth < DEEP_HARD_MIN_DEPTH:
        return (
            f"reasoning_depth={metrics.reasoning_depth} < {DEEP_HARD_MIN_DEPTH} "
            "(Deep Hard)"
        )
    if metrics.adaptive_edges < DEEP_HARD_MIN_ADAPTIVE:
        return (
            f"adaptive_edges={metrics.adaptive_edges} < {DEEP_HARD_MIN_ADAPTIVE} "
            "(Deep Hard)"
        )
    return ""
