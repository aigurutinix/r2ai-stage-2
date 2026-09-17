"""Operations that produce or consume adaptive threshold, set, and key values.

Selectors enforce a non-trivial ranking population. Filters enforce a nonempty proper
subset of the universe. Statistical threshold nodes derive their values at runtime;
fixed financial conventions remain ordinary graph parameters.
"""

from __future__ import annotations

import pandas as pd

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

_SERIES_TO_KEY: dict[ValueKind, ValueKind] = {
    ValueKind.NUMERIC_SERIES_ENTITY: ValueKind.ENTITY_KEY,
    ValueKind.NUMERIC_SERIES_PERIOD: ValueKind.PERIOD_KEY,
    ValueKind.NUMERIC_SERIES_ENTITY_PERIOD: ValueKind.ENTITY_PERIOD_KEY,
}
_SERIES_TO_SET: dict[ValueKind, ValueKind] = {
    ValueKind.NUMERIC_SERIES_ENTITY: ValueKind.ENTITY_SET,
    ValueKind.NUMERIC_SERIES_PERIOD: ValueKind.PERIOD_SET,
}
_MIN_SURVIVORS_BEFORE_SELECTOR = 2

_OPERATORS = {">", "<", ">=", "<="}
_OP_EXPR = {">": "{a} > {b}", "<": "{a} < {b}", ">=": "{a} >= {b}", "<=": "{a} <= {b}"}


def _apply_operator(operator: str, value: float, threshold: float) -> bool:
    if operator == ">":
        return value > threshold
    if operator == "<":
        return value < threshold
    if operator == ">=":
        return value >= threshold
    return value <= threshold


class ArgmaxOperation:
    name = "argmax"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        _validate_selector(node, graph, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return _evaluate_selector(node, inputs, pick_max=True)

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return _emit_selector(node, inputs, context, pick_max=True)


class ArgminOperation:
    name = "argmin"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        _validate_selector(node, graph, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return _evaluate_selector(node, inputs, pick_max=False)

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return _emit_selector(node, inputs, context, pick_max=False)


def _validate_selector(
    node: ReasoningNode, graph: ReasoningGraph, op_name: str
) -> None:
    del graph
    if len(node.inputs) not in (1, 2):
        raise GraphError(
            f"{node.step_id}: {op_name} requires one or two inputs; got {len(node.inputs)}"
        )
    source = node.inputs[0]
    if not isinstance(source, StepOutputInput):
        raise GraphError(
            f"{node.step_id}: the first {op_name} input must be StepOutputInput"
        )
    expected_output = _SERIES_TO_KEY.get(source.expected_kind)
    if expected_output is None:
        raise GraphError(
            f"{node.step_id}: {op_name} does not support input kind {source.expected_kind}"
        )
    if node.output_kind != expected_output:
        raise GraphError(
            f"{node.step_id}: input {source.expected_kind} must output {expected_output}, "
            f"node khai output_kind={node.output_kind}"
        )
    if len(node.inputs) == 2:
        restrict = node.inputs[1]
        if not isinstance(restrict, StepOutputInput):
            raise GraphError(
                f"{node.step_id}: the second {op_name} input must be a StepOutputInput restriction set"
            )
        expected_set = _SERIES_TO_SET.get(source.expected_kind)
        if expected_set is None or restrict.expected_kind != expected_set:
            raise GraphError(
                f"{node.step_id}: {op_name} restriction set must be {expected_set} to match {source.expected_kind}"
            )
    require_positive = node.params.get("require_positive", False)
    if not isinstance(require_positive, bool):
        raise GraphError(f"{node.step_id}: params.require_positive must be bool")


def _restrict_keys(series: dict, inputs: tuple[object, ...]) -> dict:
    if len(inputs) == 1:
        return series
    restrict_set = inputs[1]
    assert isinstance(restrict_set, (set, frozenset))
    restricted = {k: v for k, v in series.items() if k in restrict_set}
    if len(restricted) < _MIN_SURVIVORS_BEFORE_SELECTOR:
        raise OperationError(
            f"restriction leaves {len(restricted)} item(s); at least "
            f"{_MIN_SURVIVORS_BEFORE_SELECTOR} are required before ranking"
        )
    return restricted


def _evaluate_selector(
    node: ReasoningNode, inputs: tuple[object, ...], *, pick_max: bool
) -> object:
    series = inputs[0]
    assert isinstance(series, dict)
    if not series:
        raise OperationError(f"{node.step_id}: population is empty; there is nothing to rank")
    pool = _restrict_keys(series, inputs)
    if len(pool) < _MIN_SURVIVORS_BEFORE_SELECTOR:
        raise OperationError(
            f"{node.step_id}: rank pool has {len(pool)} item(s); at least {_MIN_SURVIVORS_BEFORE_SELECTOR} are required"
        )
    best = max(pool.values()) if pick_max else min(pool.values())
    if node.params.get("require_positive", False) and best <= 0:
        raise OperationError(f"{node.step_id}: selected value must be positive; got {best}")
    winners = [key for key, value in pool.items() if value == best]
    if len(winners) != 1:
        raise OperationError(
            f"{node.step_id}: no unique winner because of a tie: {winners}"
        )
    return winners[0]


def _emit_selector(
    node: ReasoningNode,
    inputs: tuple[str, ...],
    context: EmitContext,
    *,
    pick_max: bool,
) -> list[str]:
    var = context.var_for(node.step_id)
    source_var = inputs[0]
    lines: list[str] = []
    if len(inputs) == 2:
        restrict_var = inputs[1]
        pool_var = f"_pool{var}"
        lines.append(
            f"{pool_var} = {{k: v for k, v in {source_var}.items() if k in {restrict_var}}}"
        )
        pool_ref = pool_var
    else:
        pool_ref = source_var
    fn = "max" if pick_max else "min"
    lines += [
        f"if len({pool_ref}) < {_MIN_SURVIVORS_BEFORE_SELECTOR}:",
        f"    assert False, '{node.step_id}: rank pool too small'",
        f"_best{var} = {fn}({pool_ref}.values())",
    ]
    if node.params.get("require_positive", False):
        lines += [
            f"if _best{var} <= 0:",
            f"    assert False, '{node.step_id}: selected value must be positive'",
        ]
    lines += [
        f"_winners{var} = [k for k, v in {pool_ref}.items() if v == _best{var}]",
        f"if len(_winners{var}) != 1:",
        f"    assert False, '{node.step_id}: rank winner is not unique'",
        f"{var} = _winners{var}[0]",
    ]
    return lines


_KEY_KINDS = frozenset(
    {ValueKind.ENTITY_KEY, ValueKind.PERIOD_KEY, ValueKind.ENTITY_PERIOD_KEY}
)
_SERIES_KINDS = frozenset(_SERIES_TO_KEY)


class LookupOperation:
    name = "lookup"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 2:
            raise GraphError(
                f"{node.step_id}: lookup requires two inputs (series, key); got {len(node.inputs)}"
            )
        series_input, key_input = node.inputs
        if not isinstance(series_input, StepOutputInput) or not isinstance(
            key_input, StepOutputInput
        ):
            raise GraphError(
                f"{node.step_id}: lookup requires StepOutputInput for both inputs"
            )
        if series_input.expected_kind not in _SERIES_KINDS:
            raise GraphError(
                f"{node.step_id}: first lookup input must be a NumericSeries"
            )
        if key_input.expected_kind not in _KEY_KINDS:
            raise GraphError(f"{node.step_id}: second lookup input must be a Key")
        if _SERIES_TO_KEY[series_input.expected_kind] != key_input.expected_kind:
            raise GraphError(
                f"{node.step_id}: series kind ({series_input.expected_kind}) and key kind ({key_input.expected_kind}) use different axes"
            )
        if node.output_kind != ValueKind.NUMERIC_SCALAR:
            raise GraphError(f"{node.step_id}: lookup must output NumericScalar")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series, key = inputs
        assert isinstance(series, dict)
        if key not in series:
            raise OperationError(
                f"{node.step_id}: value is missing at selected key {key!r}"
            )
        return series[key]

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        series_var, key_var = inputs
        return [f"{var} = {series_var}[{key_var}]"]


class FilterOperation:
    """Filter a numeric series into a same-axis entity or period set.

    A conventional threshold is a fixed graph parameter and uses one input. A derived
    threshold is recomputed from the runtime population and uses a second input.
    """

    name = "filter"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) not in (1, 2):
            raise GraphError(
                f"{node.step_id}: filter requires one or two inputs; got {len(node.inputs)}"
            )
        source = node.inputs[0]
        if (
            not isinstance(source, StepOutputInput)
            or source.expected_kind not in _SERIES_TO_SET
        ):
            raise GraphError(
                f"{node.step_id}: first filter input must be NumericSeriesEntity or NumericSeriesPeriod"
            )
        expected_set = _SERIES_TO_SET[source.expected_kind]
        if node.output_kind != expected_set:
            raise GraphError(
                f"{node.step_id}: filtering {source.expected_kind} must output {expected_set}"
            )
        operator = node.params.get("operator")
        if operator not in _OPERATORS:
            raise GraphError(
                f"{node.step_id}: params.operator must be one of {_OPERATORS}; got {operator!r}"
            )
        threshold_source = node.params.get("threshold_source")
        if threshold_source not in ("convention", "derived"):
            raise GraphError(
                f"{node.step_id}: params.threshold_source must be 'convention' or 'derived'"
            )
        if threshold_source == "convention":
            if len(node.inputs) != 1:
                raise GraphError(
                    f"{node.step_id}: threshold_source=convention accepts only one input"
                )
            if not isinstance(node.params.get("threshold_value"), (int, float)):
                raise GraphError(
                    f"{node.step_id}: threshold_source=convention requires numeric params.threshold_value"
                )
        else:
            if len(node.inputs) != 2:
                raise GraphError(
                    f"{node.step_id}: threshold_source=derived requires two inputs (series, threshold node)"
                )
            threshold_input = node.inputs[1]
            if (
                not isinstance(threshold_input, StepOutputInput)
                or threshold_input.expected_kind != ValueKind.THRESHOLD
            ):
                raise GraphError(
                    f"{node.step_id}: second input must be StepOutputInput with kind=Threshold"
                )

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series = inputs[0]
        assert isinstance(series, dict)
        if not series:
            raise OperationError(
                f"{node.step_id}: population is empty; there is nothing to filter"
            )
        operator = node.params["operator"]
        threshold = (
            float(inputs[1])
            if len(inputs) == 2
            else float(node.params["threshold_value"])
        )
        survivors = {
            key
            for key, value in series.items()
            if _apply_operator(operator, value, threshold)
        }
        if not survivors:
            raise OperationError(
                f"{node.step_id}: filter output is empty"
            )
        if len(survivors) == len(series):
            raise OperationError(
                f"{node.step_id}: filter output equals the entire universe"
            )
        return survivors

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        source_var = inputs[0]
        operator = node.params["operator"]
        if len(inputs) == 2:
            threshold_expr = f"float({inputs[1]})"
        else:
            threshold_expr = repr(float(node.params["threshold_value"]))
        expr = _OP_EXPR[operator].format(a="v", b=f"_threshold{var}")
        lines = [
            f"_threshold{var} = {threshold_expr}",
            f"{var} = {{k for k, v in {source_var}.items() if {expr}}}",
            f"if not {var} or len({var}) == len({source_var}):",
            f"    assert False, '{node.step_id}: filter output is empty or full'",
        ]
        if node.params.get("threshold_source") == "derived":
            lines += [
                f"if len({var}) < {_MIN_SURVIVORS_BEFORE_SELECTOR}:",
                f"    assert False, '{node.step_id}: derived selected set too small'",
            ]
        return lines


class TemporalAllOperation:
    """Return entities that satisfy a predicate in every period of a complete panel."""

    name = "temporal_all"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        if len(node.inputs) != 1:
            raise GraphError(
                f"{node.step_id}: temporal_all requires one input; got {len(node.inputs)}"
            )
        source = node.inputs[0]
        if (
            not isinstance(source, StepOutputInput)
            or source.expected_kind != ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
        ):
            raise GraphError(
                f"{node.step_id}: temporal_all accepts only NumericSeriesEntityPeriod input"
            )
        if node.output_kind != ValueKind.ENTITY_SET:
            raise GraphError(f"{node.step_id}: temporal_all must output EntitySet")
        operator = node.params.get("operator")
        if operator not in _OPERATORS:
            raise GraphError(f"{node.step_id}: params.operator must be one of {_OPERATORS}")
        if not isinstance(node.params.get("threshold_value"), (int, float)):
            raise GraphError(f"{node.step_id}: params.threshold_value must be numeric")
        period_universe = node.params.get("period_universe")
        if period_universe is not None:
            if (
                not isinstance(period_universe, tuple)
                or not period_universe
                or not all(isinstance(p, str) for p in period_universe)
            ):
                raise GraphError(
                    f"{node.step_id}: params.period_universe must be a nonempty tuple[str, ...]"
                )
        if not node.domain.entities:
            raise GraphError(f"{node.step_id}: domain.entities is empty")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series = inputs[0]
        assert isinstance(series, dict)
        operator = node.params["operator"]
        threshold = float(node.params["threshold_value"])
        period_universe = node.params.get("period_universe")
        periods = (
            tuple(period_universe)
            if period_universe is not None
            else tuple(sorted({period for (_entity, period) in series}))
        )
        if not periods:
            raise OperationError(
                f"{node.step_id}: series is empty; there are no periods to check"
            )
        survivors: set[str] = set()
        for entity in node.domain.entities:
            observations = []
            for period in periods:
                key = (entity, period)
                if key not in series:
                    raise OperationError(f"{node.step_id}: missing observation for {key}")
                observations.append(series[key])
            if all(_apply_operator(operator, v, threshold) for v in observations):
                survivors.add(entity)
        if not survivors:
            raise OperationError(
                f"{node.step_id}: temporal_all output is empty"
            )
        if len(survivors) == len(node.domain.entities):
            raise OperationError(
                f"{node.step_id}: temporal_all output equals the entire universe"
            )
        return survivors

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        source_var = inputs[0]
        operator = node.params["operator"]
        threshold = float(node.params["threshold_value"])
        expr = _OP_EXPR[operator].format(a="v", b=repr(threshold))
        entities_var = f"_entities{var}"
        periods = tuple(node.params.get("period_universe") or ())
        periods_var = f"_periods{var}"
        period_line = (
            f"{periods_var} = {periods!r}"
            if periods
            else f"{periods_var} = sorted({{p for (_e, p) in {source_var}}})"
        )
        return [
            f"{entities_var} = {node.domain.entities!r}",
            period_line,
            f"{var} = {{",
            f"    _e for _e in {entities_var}",
            f"    if all(({expr.replace('v', f'{source_var}[(_e, _p)]')}) for _p in {periods_var})",
            "}",
        ]


class MedianThresholdOperation:
    """Reduce a numeric series to a median threshold with ``pandas.Series``.

    Evaluation and emitted code use the same implementation to prevent formula drift.
    """

    name = "median"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        _validate_threshold_reducer(node, graph, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return _evaluate_threshold_reducer(node, inputs, statistic="median")

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return _emit_threshold_reducer(node, inputs, context, statistic="median")


class AverageThresholdOperation:
    """Reduce an entity- or period-indexed numeric series to a mean threshold."""

    name = "average_threshold"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        _validate_threshold_reducer(node, graph, self.name)

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return _evaluate_threshold_reducer(node, inputs, statistic="mean")

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return _emit_threshold_reducer(node, inputs, context, statistic="mean")


class CohortMedianThresholdOperation:
    """Median of a numeric series after applying a runtime entity or period set."""

    name = "cohort_median"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        del graph
        if len(node.inputs) != 2:
            raise GraphError(f"{node.step_id}: cohort_median requires (series, set)")
        series, cohort = node.inputs
        expected_set = {
            ValueKind.NUMERIC_SERIES_ENTITY: ValueKind.ENTITY_SET,
            ValueKind.NUMERIC_SERIES_PERIOD: ValueKind.PERIOD_SET,
        }.get(series.expected_kind if isinstance(series, StepOutputInput) else None)
        if expected_set is None:
            raise GraphError(f"{node.step_id}: series must be indexed by entity or period")
        if (
            not isinstance(cohort, StepOutputInput)
            or cohort.expected_kind != expected_set
        ):
            raise GraphError(f"{node.step_id}: restriction set does not match the series axis")
        if node.output_kind != ValueKind.THRESHOLD:
            raise GraphError(f"{node.step_id}: cohort_median must output Threshold")

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        series, cohort = inputs
        assert isinstance(series, dict) and isinstance(cohort, (set, frozenset))
        values = [float(value) for key, value in series.items() if key in cohort]
        if not values:
            raise OperationError(f"{node.step_id}: cohort is empty; median is undefined")
        return float(pd.Series(values, dtype="float64").median())

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        var = context.var_for(node.step_id)
        pool = f"_pool{var}"
        return [
            f"{pool} = [float(v) for k, v in {inputs[0]}.items() if k in {inputs[1]}]",
            f"assert {pool}",
            f"{var} = float(pd.Series({pool}, dtype='float64').median())",
        ]


class PercentileOperation:
    """Reduce a numeric series to the requested percentile threshold (0-100)."""

    name = "percentile"

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        _validate_threshold_reducer(node, graph, self.name)
        percentile = node.params.get("percentile")
        if not isinstance(percentile, (int, float)) or not (0 < percentile < 100):
            raise GraphError(
                f"{node.step_id}: params.percentile must be numeric and in (0, 100)"
            )

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        return _evaluate_threshold_reducer(node, inputs, statistic="quantile")

    def emit(
        self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext
    ) -> list[str]:
        return _emit_threshold_reducer(node, inputs, context, statistic="quantile")


def _validate_threshold_reducer(
    node: ReasoningNode, graph: ReasoningGraph, op_name: str
) -> None:
    del graph
    if len(node.inputs) != 1:
        raise GraphError(
            f"{node.step_id}: {op_name} requires one input; got {len(node.inputs)}"
        )
    source = node.inputs[0]
    if (
        not isinstance(source, StepOutputInput)
        or source.expected_kind not in _SERIES_TO_SET
    ):
        raise GraphError(
            f"{node.step_id}: {op_name} input must be NumericSeriesEntity or NumericSeriesPeriod"
        )
    if node.output_kind != ValueKind.THRESHOLD:
        raise GraphError(f"{node.step_id}: {op_name} must output Threshold")


def _evaluate_threshold_reducer(
    node: ReasoningNode, inputs: tuple[object, ...], *, statistic: str
) -> object:
    series = inputs[0]
    assert isinstance(series, dict)
    if not series:
        raise OperationError(
            f"{node.step_id}: population is empty; {statistic} is undefined"
        )
    values = pd.Series(list(series.values()), dtype="float64")
    if statistic == "median":
        return float(values.median())
    if statistic == "mean":
        return float(values.mean())
    q = float(node.params["percentile"]) / 100.0
    return float(values.quantile(q))


def _emit_threshold_reducer(
    node: ReasoningNode,
    inputs: tuple[str, ...],
    context: EmitContext,
    *,
    statistic: str,
) -> list[str]:
    var = context.var_for(node.step_id)
    source_var = inputs[0]
    if statistic == "median":
        call = "median()"
    elif statistic == "mean":
        call = "mean()"
    else:
        q = float(node.params["percentile"]) / 100.0
        call = f"quantile({q!r})"
    lines: list[str] = []
    if statistic in {"median", "mean"}:
        lines += [
            f"if len({source_var}) < 5:",
            f"    assert False, '{node.step_id}: threshold population too small'",
        ]
    lines.append(
        f"{var} = float(pd.Series(list({source_var}.values()), dtype='float64').{call})"
    )
    return lines


register_operation(ArgmaxOperation())
register_operation(ArgminOperation())
register_operation(LookupOperation())
register_operation(FilterOperation())
register_operation(TemporalAllOperation())
register_operation(MedianThresholdOperation())
register_operation(AverageThresholdOperation())
register_operation(CohortMedianThresholdOperation())
register_operation(PercentileOperation())
