"""Numeric reducers over entity- or period-indexed series.

Reducers accept either a whole population or a population plus a runtime restriction
set. Restricted aggregate operations require at least two survivors so the result is
non-trivial. ``count`` is exempt because a single survivor can be a valid count.
"""

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

_SERIES_TO_SET: dict[ValueKind, ValueKind] = {
    ValueKind.NUMERIC_SERIES_ENTITY: ValueKind.ENTITY_SET,
    ValueKind.NUMERIC_SERIES_PERIOD: ValueKind.PERIOD_SET,
}


def _validate_reducer(node: ReasoningNode, op_name: str) -> None:
    if len(node.inputs) not in (1, 2):
        raise GraphError(f"{node.step_id}: {op_name} requires one or two inputs; got {len(node.inputs)}")
    source = node.inputs[0]
    if not isinstance(source, StepOutputInput) or source.expected_kind not in _SERIES_TO_SET:
        raise GraphError(f"{node.step_id}: the first {op_name} input must be NumericSeriesEntity or NumericSeriesPeriod")
    if node.output_kind != ValueKind.NUMERIC_SCALAR:
        raise GraphError(f"{node.step_id}: {op_name} must output NumericScalar")
    if len(node.inputs) == 2:
        restrict = node.inputs[1]
        if not isinstance(restrict, StepOutputInput):
            raise GraphError(f"{node.step_id}: the second {op_name} input must be a StepOutputInput restriction set")
        expected_set = _SERIES_TO_SET[source.expected_kind]
        if restrict.expected_kind != expected_set:
            raise GraphError(
                f"{node.step_id}: {op_name} restriction set must be {expected_set} to match {source.expected_kind}"
            )


def _resolve_pool(
    node: ReasoningNode,
    inputs: tuple[object, ...],
    *,
    require_nontrivial_restricted_pool: bool = False,
) -> dict:
    series = inputs[0]
    assert isinstance(series, dict)
    if len(inputs) == 1:
        pool = series
    else:
        restrict_set = inputs[1]
        assert isinstance(restrict_set, (set, frozenset))
        pool = {k: v for k, v in series.items() if k in restrict_set}
    if not pool:
        raise OperationError(f"{node.step_id}: population is empty after applying the restriction set")
    if require_nontrivial_restricted_pool and len(inputs) == 2 and len(pool) < 2:
        raise OperationError(
            f"{node.step_id}: restricted pool has only {len(pool)} item(s); the aggregate is trivial"
        )
    return pool


def _emit_pool(
    node: ReasoningNode,
    inputs: tuple[str, ...],
    context: EmitContext,
    *,
    require_nontrivial_restricted_pool: bool = False,
) -> tuple[str, list[str]]:
    var = context.var_for(node.step_id)
    source_var = inputs[0]
    if len(inputs) == 1:
        return source_var, []
    restrict_var = inputs[1]
    pool_var = f"_pool{var}"
    lines = [f"{pool_var} = {{k: v for k, v in {source_var}.items() if k in {restrict_var}}}"]
    if require_nontrivial_restricted_pool:
        lines += [
            f"if len({pool_var}) < 2:",
            f"    assert False, '{node.step_id}: restricted pool too small'",
        ]
    else:
        lines += [
            f"if not {pool_var}:",
            f"    assert False, '{node.step_id}: restricted pool is empty'",
        ]
    return pool_var, lines


class SumOperation:

    name = "sum"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        _validate_reducer(node, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return sum(_resolve_pool(node, inputs, require_nontrivial_restricted_pool=True).values())

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        var = context.var_for(node.step_id)
        pool_expr, lines = _emit_pool(node, inputs, context, require_nontrivial_restricted_pool=True)
        lines.append(f"{var} = sum({pool_expr}.values())")
        return lines


class AverageOperation:
    name = "average"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        _validate_reducer(node, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        pool = _resolve_pool(node, inputs, require_nontrivial_restricted_pool=True)
        return sum(pool.values()) / len(pool)

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        var = context.var_for(node.step_id)
        pool_expr, lines = _emit_pool(node, inputs, context, require_nontrivial_restricted_pool=True)
        lines.append(f"{var} = sum({pool_expr}.values()) / len({pool_expr})")
        return lines


class MinimumOperation:
    name = "minimum"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        _validate_reducer(node, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return min(_resolve_pool(node, inputs, require_nontrivial_restricted_pool=True).values())

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        var = context.var_for(node.step_id)
        pool_expr, lines = _emit_pool(node, inputs, context, require_nontrivial_restricted_pool=True)
        lines.append(f"{var} = min({pool_expr}.values())")
        return lines


class MaximumOperation:
    name = "maximum"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        _validate_reducer(node, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return max(_resolve_pool(node, inputs, require_nontrivial_restricted_pool=True).values())

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        var = context.var_for(node.step_id)
        pool_expr, lines = _emit_pool(node, inputs, context, require_nontrivial_restricted_pool=True)
        lines.append(f"{var} = max({pool_expr}.values())")
        return lines


class CountOperation:

    name = "count"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        _validate_reducer(node, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return len(_resolve_pool(node, inputs))

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        var = context.var_for(node.step_id)
        pool_expr, lines = _emit_pool(node, inputs, context)
        lines.append(f"{var} = len({pool_expr})")
        return lines


class CohortAverageOperation:

    name = "cohort_average"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        _validate_reducer(node, self.name)
        if len(node.inputs) != 2:
            raise GraphError(f"{node.step_id}: cohort_average requires exactly (series, cohort set)")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        pool = _resolve_pool(node, inputs)
        return sum(pool.values()) / len(pool)

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        var = context.var_for(node.step_id)
        pool_expr, lines = _emit_pool(node, inputs, context, require_nontrivial_restricted_pool=True)
        lines.append(f"{var} = sum({pool_expr}.values()) / len({pool_expr})")
        return lines


class RestrictedSumOperation:

    name = "restricted_sum"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        _validate_reducer(node, self.name)
        if len(node.inputs) != 2:
            raise GraphError(f"{node.step_id}: restricted_sum requires exactly (series, cohort set)")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return sum(_resolve_pool(node, inputs).values())

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        var = context.var_for(node.step_id)
        pool_expr, lines = _emit_pool(node, inputs, context)
        lines.append(f"{var} = sum({pool_expr}.values())")
        return lines


register_operation(SumOperation())
register_operation(AverageOperation())
register_operation(MinimumOperation())
register_operation(MaximumOperation())
register_operation(CountOperation())
register_operation(CohortAverageOperation())
register_operation(RestrictedSumOperation())
