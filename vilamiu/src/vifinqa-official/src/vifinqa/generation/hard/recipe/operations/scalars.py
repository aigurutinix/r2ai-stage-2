"""Scalar arithmetic primitives for combining independent reasoning branches."""

from __future__ import annotations

from vifinqa.generation.hard.recipe.base import (
    EmitContext,
    GraphError,
    OperationError,
    ReasoningGraph,
    ReasoningNode,
    StepOutputInput,
    ValueKind,
)
from vifinqa.generation.hard.recipe.registry import register_operation


class ScalarDifferenceOperation:
    name = "scalar_difference"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 2:
            raise GraphError(f"{node.step_id}: scalar_difference requires exactly two inputs")
        if any(
            not isinstance(item, StepOutputInput)
            or item.expected_kind != ValueKind.NUMERIC_SCALAR
            for item in node.inputs
        ):
            raise GraphError(
                f"{node.step_id}: scalar_difference accepts only two NumericScalar inputs"
            )
        if node.output_kind != ValueKind.NUMERIC_SCALAR:
            raise GraphError(
                f"{node.step_id}: scalar_difference must output NumericScalar"
            )

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        result = float(inputs[0]) - float(inputs[1])
        if node.params.get("require_positive") is True and result <= 0:
            raise OperationError(
                f"{node.step_id}: the difference must be positive to match the 'higher' premise; got {result}"
            )
        return result

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return [f"{context.var_for(node.step_id)} = {inputs[0]} - {inputs[1]}"]


class ScalarShareOperation:
    """Generic ``part / whole`` share; the operation name is not exposed in questions."""

    name = "scalar_share"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 2:
            raise GraphError(f"{node.step_id}: scalar_share requires exactly (part, whole)")
        if any(
            not isinstance(item, StepOutputInput)
            or item.expected_kind != ValueKind.NUMERIC_SCALAR
            for item in node.inputs
        ):
            raise GraphError(f"{node.step_id}: scalar_share accepts only two NumericScalar inputs")
        if node.output_kind != ValueKind.NUMERIC_SCALAR:
            raise GraphError(f"{node.step_id}: scalar_share must output NumericScalar")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        denominator = float(inputs[1])
        if denominator == 0:
            raise OperationError(f"{node.step_id}: share denominator is zero")
        return float(inputs[0]) / denominator

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return [f"{context.var_for(node.step_id)} = {inputs[0]} / {inputs[1]}"]


class ScalarAbsoluteOperation:
    """Absolute magnitude of a runtime scalar, used when wording asks for a decrease."""

    name = "scalar_absolute"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 1:
            raise GraphError(f"{node.step_id}: scalar_absolute requires exactly one input")
        source = node.inputs[0]
        if (
            not isinstance(source, StepOutputInput)
            or source.expected_kind != ValueKind.NUMERIC_SCALAR
        ):
            raise GraphError(f"{node.step_id}: scalar_absolute accepts only NumericScalar")
        if node.output_kind != ValueKind.NUMERIC_SCALAR:
            raise GraphError(
                f"{node.step_id}: scalar_absolute must output NumericScalar"
            )

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        del node
        return abs(float(inputs[0]))

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return [f"{context.var_for(node.step_id)} = abs(float({inputs[0]}))"]


register_operation(ScalarDifferenceOperation())
register_operation(ScalarShareOperation())
register_operation(ScalarAbsoluteOperation())
