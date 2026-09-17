"""Year-over-year growth over an entity-period numeric series.

``period_universe`` is the complete, sorted, consecutive input range. It must contain
exactly one leading period beyond ``node.domain.periods`` because the first growth
period requires its immediately preceding value.
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


class GrowthOperation:
    name = "growth"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        if len(node.inputs) != 1:
            raise GraphError(
                f"{node.step_id}: growth requires one input; got {len(node.inputs)}"
            )
        source = node.inputs[0]
        if not isinstance(source, StepOutputInput):
            raise GraphError(f"{node.step_id}: growth accepts only StepOutputInput")
        if source.expected_kind != ValueKind.NUMERIC_SERIES_ENTITY_PERIOD:
            raise GraphError(
                f"{node.step_id}: growth requires NumericSeriesEntityPeriod input"
            )
        if node.output_kind != ValueKind.NUMERIC_SERIES_ENTITY_PERIOD:
            raise GraphError(
                f"{node.step_id}: growth must output NumericSeriesEntityPeriod"
            )
        period_universe = node.params.get("period_universe")
        if not isinstance(period_universe, tuple) or len(period_universe) < 2:
            raise GraphError(
                f"{node.step_id}: params.period_universe must be a tuple of length >= 2"
            )
        if tuple(period_universe[1:]) != tuple(node.domain.periods):
            raise GraphError(
                f"{node.step_id}: node.domain.periods must equal period_universe[1:] "
                f"({period_universe[1:]} != {node.domain.periods})"
            )
        if not node.domain.entities:
            raise GraphError(f"{node.step_id}: domain.entities is empty")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series = inputs[0]
        assert isinstance(series, dict)
        period_universe = node.params["period_universe"]
        result: dict[tuple[str, str], float] = {}
        for entity in node.domain.entities:
            for i in range(1, len(period_universe)):
                cur = (entity, period_universe[i])
                prev = (entity, period_universe[i - 1])
                if cur not in series or prev not in series:
                    raise OperationError(
                        f"{node.step_id}: missing revenue coverage for {cur} or {prev}"
                    )
                prev_value = series[prev]
                if prev_value == 0:
                    raise OperationError(
                        f"{node.step_id}: previous-period revenue is zero at {prev}; growth is undefined"
                    )
                result[cur] = series[cur] / prev_value - 1
        return result

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        periods_var = f"_periods{var}"
        period_universe = node.params["period_universe"]
        return [
            f"{periods_var} = {period_universe!r}",
            f"{var} = {{}}",
            f"for _entity in {node.domain.entities!r}:",
            f"    for _i in range(1, len({periods_var})):",
            f"        _cur = (_entity, {periods_var}[_i])",
            f"        _prev = (_entity, {periods_var}[_i - 1])",
            f"        {var}[_cur] = {inputs[0]}[_cur] / {inputs[0]}[_prev] - 1",
        ]


class PeriodDifferenceOperation:
    """Return ``current - previous`` for each entity instead of a growth rate.

    This supports ratio metrics whose cross-period difference cannot be represented as
    one linear ``MetricTerms`` expression because the periods have different denominators.
    """

    name = "period_difference"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        if len(node.inputs) != 1:
            raise GraphError(
                f"{node.step_id}: period_difference requires one input; got {len(node.inputs)}"
            )
        source = node.inputs[0]
        if not isinstance(source, StepOutputInput):
            raise GraphError(
                f"{node.step_id}: period_difference accepts only StepOutputInput"
            )
        if source.expected_kind != ValueKind.NUMERIC_SERIES_ENTITY_PERIOD:
            raise GraphError(
                f"{node.step_id}: period_difference requires NumericSeriesEntityPeriod input"
            )
        if node.output_kind != ValueKind.NUMERIC_SERIES_ENTITY_PERIOD:
            raise GraphError(
                f"{node.step_id}: period_difference must output NumericSeriesEntityPeriod"
            )
        period_universe = node.params.get("period_universe")
        if not isinstance(period_universe, tuple) or len(period_universe) < 2:
            raise GraphError(
                f"{node.step_id}: params.period_universe must be a tuple of length >= 2"
            )
        if tuple(period_universe[1:]) != tuple(node.domain.periods):
            raise GraphError(
                f"{node.step_id}: node.domain.periods must equal period_universe[1:] "
                f"({period_universe[1:]} != {node.domain.periods})"
            )
        if not node.domain.entities:
            raise GraphError(f"{node.step_id}: domain.entities is empty")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series = inputs[0]
        assert isinstance(series, dict)
        period_universe = node.params["period_universe"]
        result: dict[tuple[str, str], float] = {}
        for entity in node.domain.entities:
            for i in range(1, len(period_universe)):
                cur = (entity, period_universe[i])
                prev = (entity, period_universe[i - 1])
                if cur not in series or prev not in series:
                    raise OperationError(
                        f"{node.step_id}: missing coverage for {cur} or {prev}"
                    )
                result[cur] = series[cur] - series[prev]
        return result

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        periods_var = f"_periods{var}"
        period_universe = node.params["period_universe"]
        return [
            f"{periods_var} = {period_universe!r}",
            f"{var} = {{}}",
            f"for _entity in {node.domain.entities!r}:",
            f"    for _i in range(1, len({periods_var})):",
            f"        _cur = (_entity, {periods_var}[_i])",
            f"        _prev = (_entity, {periods_var}[_i - 1])",
            f"        {var}[_cur] = {inputs[0]}[_cur] - {inputs[0]}[_prev]",
        ]


class SlicePeriodOperation:
    """Project an entity-period panel onto one period.

    The selected period is a public graph parameter, not a precomputed winner or value.
    """

    name = "slice_period"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 1:
            raise GraphError(
                f"{node.step_id}: slice_period requires one input; got {len(node.inputs)}"
            )
        source = node.inputs[0]
        if not isinstance(source, StepOutputInput):
            raise GraphError(f"{node.step_id}: slice_period accepts only StepOutputInput")
        if source.expected_kind != ValueKind.NUMERIC_SERIES_ENTITY_PERIOD:
            raise GraphError(
                f"{node.step_id}: slice_period requires NumericSeriesEntityPeriod"
            )
        if node.output_kind != ValueKind.NUMERIC_SERIES_ENTITY:
            raise GraphError(
                f"{node.step_id}: slice_period must output NumericSeriesEntity"
            )
        period = node.params.get("period")
        if not isinstance(period, str) or period not in node.domain.periods:
            raise GraphError(
                f"{node.step_id}: params.period must be in domain.periods {node.domain.periods!r}"
            )
        if not node.domain.entities:
            raise GraphError(f"{node.step_id}: domain.entities is empty")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series = inputs[0]
        assert isinstance(series, dict)
        period = node.params["period"]
        result: dict[str, float] = {}
        for entity in node.domain.entities:
            key = (entity, period)
            if key not in series:
                raise OperationError(f"{node.step_id}: missing coverage for {key}")
            result[entity] = series[key]
        return result

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        period = node.params["period"]
        return [
            f"{var} = {{e: {inputs[0]}[(e, {period!r})] for e in {node.domain.entities!r}}}"
        ]


class SeriesDivideOperation:
    """Divide two numeric series aligned to the same keys.

    This supports formulas whose numerator and denominator are runtime transform results,
    such as DOL, without embedding the formula in the planner or compiler.
    """

    name = "series_divide"

    _KINDS = {
        ValueKind.NUMERIC_SERIES_ENTITY,
        ValueKind.NUMERIC_SERIES_PERIOD,
        ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
    }

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 2:
            raise GraphError(f"{node.step_id}: series_divide requires exactly two inputs")
        first, second = node.inputs
        if not isinstance(first, StepOutputInput) or not isinstance(
            second, StepOutputInput
        ):
            raise GraphError(f"{node.step_id}: both inputs must be StepOutputInput")
        if (
            first.expected_kind != second.expected_kind
            or first.expected_kind not in self._KINDS
        ):
            raise GraphError(f"{node.step_id}: both numeric series must have the same kind")
        if node.output_kind != first.expected_kind:
            raise GraphError(f"{node.step_id}: output must have the same kind as both inputs")
        epsilon = node.params.get("denominator_epsilon", 0.0)
        if not isinstance(epsilon, (int, float)) or float(epsilon) < 0:
            raise GraphError(f"{node.step_id}: denominator_epsilon must be non-negative")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        numerator, denominator = inputs
        assert isinstance(numerator, dict) and isinstance(denominator, dict)
        if set(numerator) != set(denominator):
            raise OperationError(f"{node.step_id}: the two series have different keys")
        epsilon = float(node.params.get("denominator_epsilon", 0.0))
        if any(abs(float(denominator[key])) <= epsilon for key in denominator):
            raise OperationError(
                f"{node.step_id}: a denominator does not exceed epsilon {epsilon}"
            )
        return {
            key: float(numerator[key]) / float(denominator[key]) for key in numerator
        }

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        epsilon = float(node.params.get("denominator_epsilon", 0.0))
        return [
            f"assert set({inputs[0]}) == set({inputs[1]})",
            f"assert all(abs(float(v)) > {epsilon!r} for v in {inputs[1]}.values())",
            f"{var} = {{k: float({inputs[0]}[k]) / float({inputs[1]}[k]) for k in {inputs[0]}}}",
        ]


register_operation(GrowthOperation())
register_operation(PeriodDifferenceOperation())
register_operation(SlicePeriodOperation())
register_operation(SeriesDivideOperation())
