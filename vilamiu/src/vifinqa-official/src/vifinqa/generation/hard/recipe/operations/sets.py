
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

_SET_KINDS = frozenset({ValueKind.ENTITY_SET, ValueKind.PERIOD_SET})
_SERIES_TO_SET = {
    ValueKind.NUMERIC_SERIES_ENTITY: ValueKind.ENTITY_SET,
    ValueKind.NUMERIC_SERIES_PERIOD: ValueKind.PERIOD_SET,
}
_OPERATORS = {
    ">": lambda value, threshold: value > threshold,
    ">=": lambda value, threshold: value >= threshold,
    "<": lambda value, threshold: value < threshold,
    "<=": lambda value, threshold: value <= threshold,
}
_OP_EXPR = {">": "v > {t}", ">=": "v >= {t}", "<": "v < {t}", "<=": "v <= {t}"}
_COMPARE_EXPR = {
    ">": "{a} > {b}",
    ">=": "{a} >= {b}",
    "<": "{a} < {b}",
    "<=": "{a} <= {b}",
}


class PredicateSetOperation:

    name = "predicate_set"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 1:
            raise GraphError(f"{node.step_id}: predicate_set requires exactly one input")
        source = node.inputs[0]
        if (
            not isinstance(source, StepOutputInput)
            or source.expected_kind not in _SERIES_TO_SET
        ):
            raise GraphError(
                f"{node.step_id}: predicate_set requires NumericSeriesEntity or NumericSeriesPeriod"
            )
        if node.output_kind != _SERIES_TO_SET[source.expected_kind]:
            raise GraphError(f"{node.step_id}: output set does not match the input axis")
        if node.params.get("operator") not in _OPERATORS:
            raise GraphError(f"{node.step_id}: invalid predicate_set operator")
        if not isinstance(node.params.get("threshold_value"), (int, float)):
            raise GraphError(
                f"{node.step_id}: predicate_set threshold_value must be numeric"
            )

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series = inputs[0]
        assert isinstance(series, dict)
        operator = _OPERATORS[node.params["operator"]]
        threshold = float(node.params["threshold_value"])
        return frozenset(
            key for key, value in series.items() if operator(value, threshold)
        )

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        threshold = float(node.params["threshold_value"])
        expression = _OP_EXPR[node.params["operator"]].format(t=repr(threshold))
        return [f"{var} = {{k for k, v in {inputs[0]}.items() if {expression}}}"]


class SetIntersectionOperation:
    name = "set_intersection"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) < 2:
            raise GraphError(f"{node.step_id}: set_intersection requires at least two inputs")
        if node.output_kind not in _SET_KINDS:
            raise GraphError(
                f"{node.step_id}: set_intersection must output EntitySet or PeriodSet"
            )
        for item in node.inputs:
            if (
                not isinstance(item, StepOutputInput)
                or item.expected_kind != node.output_kind
            ):
                raise GraphError(
                    f"{node.step_id}: every set_intersection input must be StepOutputInput of kind {node.output_kind}"
                )

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        del node
        sets = tuple(set(value) for value in inputs)  # type: ignore[arg-type]
        return frozenset.intersection(*(frozenset(value) for value in sets))

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        expression = " & ".join(f"set({value})" for value in inputs)
        return [f"{var} = {expression}"]


class SetCountOperation:

    name = "set_count"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 1:
            raise GraphError(f"{node.step_id}: set_count requires exactly one input")
        source = node.inputs[0]
        if (
            not isinstance(source, StepOutputInput)
            or source.expected_kind not in _SET_KINDS
        ):
            raise GraphError(f"{node.step_id}: set_count accepts only EntitySet or PeriodSet")
        if node.output_kind != ValueKind.NUMERIC_SCALAR:
            raise GraphError(f"{node.step_id}: set_count must output NumericScalar")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        del node
        return len(inputs[0])  # type: ignore[arg-type]

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return [f"{context.var_for(node.step_id)} = len({inputs[0]})"]


class DerivedPredicateSetOperation:
    """Predicate against a runtime threshold without forcing a non-trivial survivor count."""

    name = "derived_predicate_set"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 2:
            raise GraphError(
                f"{node.step_id}: derived_predicate_set requires (series, threshold)"
            )
        source, threshold = node.inputs
        if (
            not isinstance(source, StepOutputInput)
            or source.expected_kind not in _SERIES_TO_SET
        ):
            raise GraphError(
                f"{node.step_id}: first input must be NumericSeriesEntity or NumericSeriesPeriod"
            )
        if (
            not isinstance(threshold, StepOutputInput)
            or threshold.expected_kind != ValueKind.THRESHOLD
        ):
            raise GraphError(f"{node.step_id}: second input must be Threshold")
        if node.output_kind != _SERIES_TO_SET[source.expected_kind]:
            raise GraphError(f"{node.step_id}: output set does not match the series")
        if node.params.get("operator") not in _OPERATORS:
            raise GraphError(f"{node.step_id}: invalid operator")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series, threshold = inputs
        assert isinstance(series, dict)
        predicate = _OPERATORS[node.params["operator"]]
        return frozenset(
            key for key, value in series.items() if predicate(value, float(threshold))
        )

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        expression = _OP_EXPR[node.params["operator"]].format(t=f"float({inputs[1]})")
        return [
            f"{context.var_for(node.step_id)} = {{k for k, v in {inputs[0]}.items() if {expression}}}"
        ]


class SeriesCompareSetOperation:
    """Compare two aligned series key-by-key and return keys satisfying the relation."""

    name = "series_compare_set"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 2:
            raise GraphError(f"{node.step_id}: series_compare_set requires exactly two series")
        first, second = node.inputs
        if not isinstance(first, StepOutputInput) or not isinstance(
            second, StepOutputInput
        ):
            raise GraphError(f"{node.step_id}: both inputs must be StepOutputInput")
        if (
            first.expected_kind != second.expected_kind
            or first.expected_kind not in _SERIES_TO_SET
        ):
            raise GraphError(f"{node.step_id}: both series must use the same entity or period axis")
        if node.output_kind != _SERIES_TO_SET[first.expected_kind]:
            raise GraphError(f"{node.step_id}: output set does not match the series")
        if node.params.get("operator") not in _OPERATORS:
            raise GraphError(f"{node.step_id}: invalid operator")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        first, second = inputs
        assert isinstance(first, dict) and isinstance(second, dict)
        if set(first) != set(second):
            raise OperationError(f"{node.step_id}: the two series have different keys")
        predicate = _OPERATORS[node.params["operator"]]
        return frozenset(key for key in first if predicate(first[key], second[key]))

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        first, second = inputs
        expression = _COMPARE_EXPR[node.params["operator"]].format(
            a=f"{first}[k]", b=f"{second}[k]"
        )
        return [
            f"{context.var_for(node.step_id)} = {{k for k in {first} if {expression}}}"
        ]


class RankedCohortOperation:
    """Exact top/bottom p% cohort with ceil sizing and boundary-tie rejection."""

    name = "ranked_cohort"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 1:
            raise GraphError(f"{node.step_id}: ranked_cohort requires one series")
        source = node.inputs[0]
        if (
            not isinstance(source, StepOutputInput)
            or source.expected_kind not in _SERIES_TO_SET
        ):
            raise GraphError(
                f"{node.step_id}: ranked_cohort requires NumericSeriesEntity or NumericSeriesPeriod"
            )
        if node.output_kind != _SERIES_TO_SET[source.expected_kind]:
            raise GraphError(f"{node.step_id}: output set does not match the series")
        if node.params.get("side") not in {"top", "bottom"}:
            raise GraphError(f"{node.step_id}: side must be top|bottom")
        percent = node.params.get("percent")
        if not isinstance(percent, (int, float)) or not 0 < float(percent) < 50:
            raise GraphError(f"{node.step_id}: percent must be in (0, 50)")
        minimum_size = node.params.get("minimum_size", 1)
        if not isinstance(minimum_size, int) or minimum_size < 1:
            raise GraphError(f"{node.step_id}: minimum_size must be a positive integer")

    @staticmethod
    def _select(node: ReasoningNode, series: dict) -> frozenset:
        import math

        n = max(
            int(node.params.get("minimum_size", 1)),
            math.ceil(len(series) * float(node.params["percent"]) / 100.0),
        )
        if len(series) < n * 2:
            raise OperationError(
                f"{node.step_id}: population {len(series)} insufficient size for two cohorts {n}"
            )
        reverse = node.params["side"] == "top"
        ordered = sorted(series.items(), key=lambda item: item[1], reverse=reverse)
        if n < len(ordered) and ordered[n - 1][1] == ordered[n][1]:
            raise OperationError(f"{node.step_id}: tie at the cohort boundary")
        return frozenset(key for key, _value in ordered[:n])

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series = inputs[0]
        assert isinstance(series, dict)
        return self._select(node, series)

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        source = inputs[0]
        percent = float(node.params["percent"])
        minimum_size = int(node.params.get("minimum_size", 1))
        reverse = node.params["side"] == "top"
        return [
            f"_n{var} = max({minimum_size}, int(-(-(len({source}) * {percent!r} / 100.0) // 1)))",
            f"if len({source}) < _n{var} * 2:",
            f"    assert False, '{node.step_id}: population too small for two cohorts'",
            f"_ordered{var} = sorted({source}.items(), key=lambda item: item[1], reverse={reverse!r})",
            f"if _n{var} < len(_ordered{var}) and _ordered{var}[_n{var} - 1][1] == _ordered{var}[_n{var}][1]:",
            f"    assert False, '{node.step_id}: boundary tie'",
            f"{var} = {{key for key, _value in _ordered{var}[:_n{var}]}}",
        ]


class TopNOperation:

    name = "top_n"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 1:
            raise GraphError(f"{node.step_id}: top_n requires exactly one series")
        source = node.inputs[0]
        if (
            not isinstance(source, StepOutputInput)
            or source.expected_kind not in _SERIES_TO_SET
        ):
            raise GraphError(f"{node.step_id}: top_n requires NumericSeriesEntity or NumericSeriesPeriod")
        if node.output_kind != _SERIES_TO_SET[source.expected_kind]:
            raise GraphError(f"{node.step_id}: output set does not match the series")
        n = node.params.get("n")
        if not isinstance(n, int) or n < 1:
            raise GraphError(f"{node.step_id}: n must be a positive integer")

    @staticmethod
    def _select(node: ReasoningNode, series: dict) -> frozenset:
        n = int(node.params["n"])
        if n > len(series):
            raise OperationError(f"{node.step_id}: n={n} exceeds population={len(series)}")
        ordered = sorted(series.items(), key=lambda item: item[1], reverse=True)
        if n < len(ordered) and ordered[n - 1][1] == ordered[n][1]:
            raise OperationError(f"{node.step_id}: tie at the top-N boundary")
        return frozenset(key for key, _value in ordered[:n])

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series = inputs[0]
        assert isinstance(series, dict)
        return self._select(node, series)

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        ordered = f"_ordered{var}"
        n = int(node.params["n"])
        return [
            f"{ordered} = sorted({inputs[0]}.items(), key=lambda item: item[1], reverse=True)",
            f"assert {n} <= len({ordered})",
            f"assert {n} == len({ordered}) or {ordered}[{n - 1}][1] != {ordered}[{n}][1]",
            f"{var} = {{key for key, _value in {ordered}[:{n}]}}",
        ]


class SeriesRestrictOperation:
    """Restrict a numeric series by an aligned runtime set while preserving series kind."""

    name = "series_restrict"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 2:
            raise GraphError(f"{node.step_id}: series_restrict requires (series, set)")
        series, selected = node.inputs
        if not isinstance(series, StepOutputInput):
            raise GraphError(f"{node.step_id}: first input must be StepOutputInput")
        expected_set = _SERIES_TO_SET.get(series.expected_kind)
        if expected_set is None:
            raise GraphError(f"{node.step_id}: first input must be a numeric series")
        if (
            not isinstance(selected, StepOutputInput)
            or selected.expected_kind != expected_set
        ):
            raise GraphError(f"{node.step_id}: set does not match the series axis")
        if node.output_kind != series.expected_kind:
            raise GraphError(f"{node.step_id}: output must preserve the series kind")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series, selected = inputs
        assert isinstance(series, dict) and isinstance(selected, (set, frozenset))
        result = {key: value for key, value in series.items() if key in selected}
        if not result:
            raise OperationError(f"{node.step_id}: series is empty after restriction")
        return result

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        return [
            f"{var} = {{k: v for k, v in {inputs[0]}.items() if k in {inputs[1]}}}",
            f"assert {var}",
        ]


register_operation(PredicateSetOperation())
register_operation(SetIntersectionOperation())
register_operation(SetCountOperation())
register_operation(DerivedPredicateSetOperation())
register_operation(SeriesCompareSetOperation())
register_operation(RankedCohortOperation())
register_operation(TopNOperation())
register_operation(SeriesRestrictOperation())
