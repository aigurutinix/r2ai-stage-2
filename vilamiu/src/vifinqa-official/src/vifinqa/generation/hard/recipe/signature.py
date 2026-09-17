
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from vifinqa.generation.hard.recipe.base import MetricRoleInput, ReasoningGraph, StepOutputInput
from vifinqa.generation.hard.recipe.gates import validate_schema

_STRUCTURAL_PARAMS = frozenset(
    {"operator", "threshold_source", "statistic", "require_positive"}
)
_COMMUTATIVE_OPERATIONS = frozenset({"set_intersection"})


@dataclass(frozen=True, slots=True)
class OperatorSignature:
    canonical: str
    digest: str


def operator_signature(graph: ReasoningGraph) -> OperatorSignature:
    validate_schema(graph)
    nodes = {node.step_id: node for node in graph.nodes}
    role_positions = {id(role): i for i, role in enumerate(graph.metric_roles)}
    cache: dict[str, str] = {}

    def expression(step_id: str) -> str:
        if step_id in cache:
            return cache[step_id]
        node = nodes[step_id]
        children: list[str] = []
        for item in node.inputs:
            if isinstance(item, MetricRoleInput):
                position = role_positions[id(item)]
                children.append(f"role{position}:{item.expected_kind.value}")
            else:
                assert isinstance(item, StepOutputInput)
                children.append(expression(item.source_step_id))
        if node.operation in _COMMUTATIVE_OPERATIONS:
            children.sort()
        params = ",".join(
            f"{key}={node.params[key]!r}"
            for key in sorted(_STRUCTURAL_PARAMS & node.params.keys())
        )
        head = f"{node.operation}:{node.output_kind.value}"
        if params:
            head += f"[{params}]"
        result = f"{head}({','.join(children)})"
        cache[step_id] = result
        return result

    canonical = expression(graph.terminal_step_id)
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return OperatorSignature(canonical=canonical, digest=digest)
