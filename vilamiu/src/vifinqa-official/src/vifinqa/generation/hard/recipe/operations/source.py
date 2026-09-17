"""The ``extract`` operation converts metric bindings into domain-keyed values.

It is the only operation allowed to read ``MetricRoleInput`` directly. Every other
operation consumes its output through ``StepOutputInput``.
"""

from __future__ import annotations

from vifinqa.generation.hard.recipe.base import (
    EmitContext,
    GraphError,
    MetricRoleInput,
    MetricTerms,
    OperationError,
    ReasoningGraph,
    ReasoningNode,
)
from vifinqa.generation.hard.recipe.registry import register_operation


def _terms_expr(terms: MetricTerms, context: EmitContext) -> str:
    numerator_expr = " + ".join(
        f"({term.coefficient!r} * {context.cell_expr(term.cell, term.metric_key)})" for term in terms.numerator
    )
    if not terms.denominator:
        return numerator_expr
    denominator_expr = " + ".join(
        f"({term.coefficient!r} * {context.cell_expr(term.cell, term.metric_key)})" for term in terms.denominator
    )
    return f"(({numerator_expr}) / ({denominator_expr}))"


class ExtractOperation:
    name = "extract"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 1:
            raise GraphError(f"{node.step_id}: extract requires one input; got {len(node.inputs)}")
        role = node.inputs[0]
        if not isinstance(role, MetricRoleInput):
            raise GraphError(f"{node.step_id}: extract accepts MetricRoleInput, not StepOutputInput")
        if role.expected_kind != node.output_kind:
            raise GraphError(
                f"{node.step_id}: role expected_kind ({role.expected_kind}) differs from node output_kind ({node.output_kind})"
            )
        if not role.bindings:
            raise GraphError(f"{node.step_id}: role {role.role_id} has no bindings")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        role = node.inputs[0]
        assert isinstance(role, MetricRoleInput)
        result: dict[object, float] = {}
        for key, terms in role.bindings.items():
            value = terms.value
            if value is None:
                raise OperationError(
                    f"{node.step_id}: zero denominator while resolving metric_key={role.metric_key} at {key!r}"
                )
            result[key] = value
        return result

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        del inputs
        role = node.inputs[0]
        assert isinstance(role, MetricRoleInput)
        var = context.var_for(node.step_id)
        lines = [f"{var} = {{"]
        for key in sorted(role.bindings):
            terms = role.bindings[key]
            lines.append(f"    {key!r}: {_terms_expr(terms, context)},")
        lines.append("}")
        return lines


register_operation(ExtractOperation())
