
from __future__ import annotations

from vifinqa.generation.hard.recipe.base import (
    Domain,
    MetricRoleInput,
    ReasoningGraph,
    ReasoningNode,
    StepOutputInput,
    ValueKind,
)


def _node(
    step_id: str,
    operation: str,
    inputs: tuple,
    output_kind: ValueKind,
    domain: Domain,
    description: str,
    **params: object,
) -> ReasoningNode:
    return ReasoningNode(
        step_id=step_id,
        operation=operation,
        inputs=inputs,
        output_kind=output_kind,
        domain=domain,
        params=params,
        output=step_id,
        description=description,
    )


def _panel_transform_nodes(
    *,
    prefix: str,
    role: MetricRoleInput,
    transform: str,
    full_domain: Domain,
    current_domain: Domain,
    periods: tuple[str, ...],
) -> tuple[ReasoningNode, ...]:
    if transform not in {"growth", "period_difference"}:
        raise ValueError(f"unsupported panel transform: {transform!r}")
    return (
        _node(
            f"{prefix}_source",
            "extract",
            (role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            full_domain,
            f"trích panel {prefix}",
        ),
        _node(
            f"{prefix}_panel",
            transform,
            (
                StepOutputInput(
                    f"{prefix}_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            current_domain,
            f"tính biến đổi {prefix}",
            period_universe=periods,
        ),
        _node(
            f"{prefix}_current",
            "slice_period",
            (
                StepOutputInput(
                    f"{prefix}_panel", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            f"chiếu {prefix} về kỳ cuối",
            period=periods[-1],
        ),
    )


def build_extreme_lookup_difference(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    first_selector: str,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    if first_selector not in {"argmax", "argmin"}:
        raise ValueError("first_selector must be argmax|argmin")
    second_selector = "argmin" if first_selector == "argmax" else "argmax"
    nodes = (
        _node(
            "rank_values",
            "extract",
            (rank_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu xếp hạng",
        ),
        _node(
            "first_key",
            first_selector,
            (StepOutputInput("rank_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_KEY,
            domain,
            "chọn cực thứ nhất",
        ),
        _node(
            "second_key",
            second_selector,
            (StepOutputInput("rank_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_KEY,
            domain,
            "chọn cực thứ hai",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu đích",
        ),
        _node(
            "first_value",
            "lookup",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("first_key", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tra chỉ tiêu tại cực thứ nhất",
        ),
        _node(
            "second_value",
            "lookup",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("second_key", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tra chỉ tiêu tại cực thứ hai",
        ),
        _node(
            "result",
            "scalar_difference",
            (
                StepOutputInput("first_value", ValueKind.NUMERIC_SCALAR),
                StepOutputInput("second_value", ValueKind.NUMERIC_SCALAR),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "lấy chênh lệch hai cực",
        ),
    )
    return ReasoningGraph(recipe_id, domain, (rank_role, target_role), nodes, "result")


def build_entity_filter_rank_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    filter_role: MetricRoleInput,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    filter_operator: str,
    threshold: float,
    selector_direction: str,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    if selector_direction not in {"argmax", "argmin"}:
        raise ValueError("selector_direction must be argmax|argmin")
    nodes = (
        _node(
            "filter_values",
            "extract",
            (filter_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu lọc",
        ),
        _node(
            "selected",
            "filter",
            (StepOutputInput("filter_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort",
            operator=filter_operator,
            threshold_source="convention",
            threshold_value=threshold,
        ),
        _node(
            "rank_values",
            "extract",
            (rank_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu xếp hạng",
        ),
        _node(
            "winner",
            selector_direction,
            (
                StepOutputInput("rank_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_KEY,
            domain,
            "chọn công ty trong cohort",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu đích",
        ),
        _node(
            "result",
            "lookup",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("winner", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tra chỉ tiêu đích tại winner",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (filter_role, rank_role, target_role), nodes, "result"
    )


def build_transformed_filter_aggregate(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    filter_role: MetricRoleInput,
    target_role: MetricRoleInput,
    operator: str,
    threshold: float,
    aggregate_operation: str,
) -> ReasoningGraph:
    full = Domain(entities, periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = [
        *_panel_transform_nodes(
            prefix="filter",
            role=filter_role,
            transform="growth",
            full_domain=full,
            current_domain=current,
            periods=periods,
        ),
        _node(
            "selected",
            "filter",
            (StepOutputInput("filter_current", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc theo tăng trưởng",
            operator=operator,
            threshold_source="convention",
            threshold_value=threshold,
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current,
            "trích chỉ tiêu đích năm cuối",
        ),
        _node(
            "result",
            aggregate_operation,
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            current,
            "tổng hợp chỉ tiêu trong cohort",
        ),
    ]
    return ReasoningGraph(
        recipe_id, full, (filter_role, target_role), tuple(nodes), "result"
    )


def build_median_split_ratio(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    cohort_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "cohort_values",
            "extract",
            (cohort_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu chia cohort",
        ),
        _node(
            "median",
            "median",
            (StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.THRESHOLD,
            domain,
            "tính trung vị",
        ),
        _node(
            "high",
            "derived_predicate_set",
            (
                StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("median", ValueKind.THRESHOLD),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "cohort trên trung vị",
            operator=">",
        ),
        _node(
            "low",
            "derived_predicate_set",
            (
                StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("median", ValueKind.THRESHOLD),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "cohort thấp hơn hoặc bằng trung vị",
            operator="<=",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu cộng",
        ),
        _node(
            "high_sum",
            "restricted_sum",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("high", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tổng cohort cao",
        ),
        _node(
            "low_sum",
            "restricted_sum",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("low", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tổng cohort thấp",
        ),
        _node(
            "result",
            "scalar_share",
            (
                StepOutputInput("high_sum", ValueKind.NUMERIC_SCALAR),
                StepOutputInput("low_sum", ValueKind.NUMERIC_SCALAR),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tỷ số hai tổng",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (cohort_role, target_role), nodes, "result"
    )


def build_ranked_cohort_terminal(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    cohort_role: MetricRoleInput,
    target_role: MetricRoleInput,
    percent: float,
    terminal: str,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes: list[ReasoningNode] = [
        _node(
            "cohort_values",
            "extract",
            (cohort_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu xếp cohort",
        ),
        _node(
            "top",
            "ranked_cohort",
            (StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lập cohort phân vị cao",
            side="top",
            percent=percent,
            minimum_size=3,
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu đích",
        ),
    ]
    if terminal == "average":
        nodes.append(
            _node(
                "result",
                "cohort_average",
                (
                    StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput("top", ValueKind.ENTITY_SET),
                ),
                ValueKind.NUMERIC_SCALAR,
                domain,
                "tính bình quân cohort cao",
            )
        )
    elif terminal == "share":
        nodes.extend(
            (
                _node(
                    "top_sum",
                    "restricted_sum",
                    (
                        StepOutputInput(
                            "target_values", ValueKind.NUMERIC_SERIES_ENTITY
                        ),
                        StepOutputInput("top", ValueKind.ENTITY_SET),
                    ),
                    ValueKind.NUMERIC_SCALAR,
                    domain,
                    "tổng cohort cao",
                ),
                _node(
                    "all_sum",
                    "sum",
                    (
                        StepOutputInput(
                            "target_values", ValueKind.NUMERIC_SERIES_ENTITY
                        ),
                    ),
                    ValueKind.NUMERIC_SCALAR,
                    domain,
                    "tổng universe",
                ),
                _node(
                    "result",
                    "scalar_share",
                    (
                        StepOutputInput("top_sum", ValueKind.NUMERIC_SCALAR),
                        StepOutputInput("all_sum", ValueKind.NUMERIC_SCALAR),
                    ),
                    ValueKind.NUMERIC_SCALAR,
                    domain,
                    "tỷ trọng cohort",
                ),
            )
        )
    elif terminal == "gap":
        nodes.insert(
            2,
            _node(
                "bottom",
                "ranked_cohort",
                (StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),),
                ValueKind.ENTITY_SET,
                set_domain,
                "lập cohort phân vị thấp",
                side="bottom",
                percent=percent,
                minimum_size=3,
            ),
        )
        nodes.extend(
            (
                _node(
                    "top_avg",
                    "cohort_average",
                    (
                        StepOutputInput(
                            "target_values", ValueKind.NUMERIC_SERIES_ENTITY
                        ),
                        StepOutputInput("top", ValueKind.ENTITY_SET),
                    ),
                    ValueKind.NUMERIC_SCALAR,
                    domain,
                    "bình quân cohort cao",
                ),
                _node(
                    "bottom_avg",
                    "cohort_average",
                    (
                        StepOutputInput(
                            "target_values", ValueKind.NUMERIC_SERIES_ENTITY
                        ),
                        StepOutputInput("bottom", ValueKind.ENTITY_SET),
                    ),
                    ValueKind.NUMERIC_SCALAR,
                    domain,
                    "bình quân cohort thấp",
                ),
                _node(
                    "result",
                    "scalar_difference",
                    (
                        StepOutputInput("top_avg", ValueKind.NUMERIC_SCALAR),
                        StepOutputInput("bottom_avg", ValueKind.NUMERIC_SCALAR),
                    ),
                    ValueKind.NUMERIC_SCALAR,
                    domain,
                    "chênh lệch hai bình quân",
                ),
            )
        )
    else:
        raise ValueError(f"invalid ranked-cohort terminal: {terminal!r}")
    return ReasoningGraph(
        recipe_id, domain, (cohort_role, target_role), tuple(nodes), "result"
    )


def build_lower_median_positive_share(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    cohort_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "cohort_values",
            "extract",
            (cohort_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu cohort",
        ),
        _node(
            "median",
            "median",
            (StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.THRESHOLD,
            domain,
            "tính trung vị",
        ),
        _node(
            "low",
            "derived_predicate_set",
            (
                StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("median", ValueKind.THRESHOLD),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc dưới trung vị",
            operator="<",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích lợi nhuận",
        ),
        _node(
            "positive",
            "predicate_set",
            (StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc lợi nhuận dương",
            operator=">",
            threshold_value=0.0,
        ),
        _node(
            "selected",
            "set_intersection",
            (
                StepOutputInput("low", ValueKind.ENTITY_SET),
                StepOutputInput("positive", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "giao cohort thấp và lợi nhuận dương",
        ),
        _node(
            "part",
            "restricted_sum",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tổng lợi nhuận cohort",
        ),
        _node(
            "whole",
            "restricted_sum",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("positive", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tổng lợi nhuận dương",
        ),
        _node(
            "result",
            "scalar_share",
            (
                StepOutputInput("part", ValueKind.NUMERIC_SCALAR),
                StepOutputInput("whole", ValueKind.NUMERIC_SCALAR),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tỷ trọng lợi nhuận",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (cohort_role, target_role), nodes, "result"
    )


def build_dual_transform_count(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    first_role: MetricRoleInput,
    second_role: MetricRoleInput,
    first_transform: str,
    second_transform: str,
    mode: str,
) -> ReasoningGraph:
    full = Domain(entities, periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = [
        *_panel_transform_nodes(
            prefix="first",
            role=first_role,
            transform=first_transform,
            full_domain=full,
            current_domain=current,
            periods=periods,
        ),
        *_panel_transform_nodes(
            prefix="second",
            role=second_role,
            transform=second_transform,
            full_domain=full,
            current_domain=current,
            periods=periods,
        ),
    ]
    if mode == "positive_negative":
        nodes.extend(
            (
                _node(
                    "first_set",
                    "predicate_set",
                    (
                        StepOutputInput(
                            "first_current", ValueKind.NUMERIC_SERIES_ENTITY
                        ),
                    ),
                    ValueKind.ENTITY_SET,
                    set_domain,
                    "lọc biến thứ nhất tăng",
                    operator=">",
                    threshold_value=0.0,
                ),
                _node(
                    "second_set",
                    "predicate_set",
                    (
                        StepOutputInput(
                            "second_current", ValueKind.NUMERIC_SERIES_ENTITY
                        ),
                    ),
                    ValueKind.ENTITY_SET,
                    set_domain,
                    "lọc biến thứ hai giảm",
                    operator="<",
                    threshold_value=0.0,
                ),
                _node(
                    "matched",
                    "set_intersection",
                    (
                        StepOutputInput("first_set", ValueKind.ENTITY_SET),
                        StepOutputInput("second_set", ValueKind.ENTITY_SET),
                    ),
                    ValueKind.ENTITY_SET,
                    set_domain,
                    "giao hai điều kiện",
                ),
            )
        )
    elif mode == "first_above_second":
        nodes.append(
            _node(
                "matched",
                "series_compare_set",
                (
                    StepOutputInput("first_current", ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput("second_current", ValueKind.NUMERIC_SERIES_ENTITY),
                ),
                ValueKind.ENTITY_SET,
                set_domain,
                "lọc biến thứ nhất tăng nhanh hơn biến thứ hai",
                operator=">",
            )
        )
    else:
        raise ValueError(f"invalid dual-transform-count mode: {mode!r}")
    nodes.append(
        _node(
            "result",
            "set_count",
            (StepOutputInput("matched", ValueKind.ENTITY_SET),),
            ValueKind.NUMERIC_SCALAR,
            full,
            "đếm survivor",
        )
    )
    return ReasoningGraph(
        recipe_id, full, (first_role, second_role), tuple(nodes), "result"
    )


def build_filtered_dual_transform_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    filter_role: MetricRoleInput,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    full = Domain(entities, periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = [
        *_panel_transform_nodes(
            prefix="filter",
            role=filter_role,
            transform="growth",
            full_domain=full,
            current_domain=current,
            periods=periods,
        ),
        _node(
            "selected",
            "predicate_set",
            (StepOutputInput("filter_current", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc doanh thu giảm",
            operator="<",
            threshold_value=0.0,
        ),
        *_panel_transform_nodes(
            prefix="rank",
            role=rank_role,
            transform="growth",
            full_domain=full,
            current_domain=current,
            periods=periods,
        ),
        _node(
            "winner",
            "argmin",
            (
                StepOutputInput("rank_current", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_KEY,
            current,
            "chọn công ty giảm COGS mạnh nhất",
        ),
        *_panel_transform_nodes(
            prefix="target",
            role=target_role,
            transform="period_difference",
            full_domain=full,
            current_domain=current,
            periods=periods,
        ),
        _node(
            "result",
            "lookup",
            (
                StepOutputInput("target_current", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("winner", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            current,
            "tra thay đổi biên gộp",
        ),
    ]
    return ReasoningGraph(
        recipe_id, full, (filter_role, rank_role, target_role), tuple(nodes), "result"
    )


def build_multi_transform_max(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    predicate_roles: tuple[MetricRoleInput, ...],
    operators: tuple[str, ...],
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    if len(predicate_roles) != len(operators) or len(predicate_roles) < 2:
        raise ValueError("multi_transform_max requires at least two predicate roles matching operators")
    full = Domain(entities, periods, report_scope)
    transform_periods = periods[-2:]
    transform_domain = Domain(entities, transform_periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes: list[ReasoningNode] = []
    set_inputs: list[StepOutputInput] = []
    for index, (role, operator) in enumerate(
        zip(predicate_roles, operators, strict=True)
    ):
        prefix = f"predicate_{index}"
        nodes.extend(
            _panel_transform_nodes(
                prefix=prefix,
                role=role,
                transform="period_difference",
                full_domain=transform_domain,
                current_domain=current,
                periods=transform_periods,
            )
        )
        set_id = f"set_{index}"
        nodes.append(
            _node(
                set_id,
                "predicate_set",
                (
                    StepOutputInput(
                        f"{prefix}_current", ValueKind.NUMERIC_SERIES_ENTITY
                    ),
                ),
                ValueKind.ENTITY_SET,
                set_domain,
                "lập predicate set",
                operator=operator,
                threshold_value=0.0,
            )
        )
        set_inputs.append(StepOutputInput(set_id, ValueKind.ENTITY_SET))
    nodes.extend(
        (
            _node(
                "selected",
                "set_intersection",
                tuple(set_inputs),
                ValueKind.ENTITY_SET,
                set_domain,
                "giao toàn bộ điều kiện",
            ),
            _node(
                "target_values",
                "extract",
                (target_role,),
                ValueKind.NUMERIC_SERIES_ENTITY,
                current,
                "trích chỉ tiêu đích năm cuối",
            ),
            _node(
                "result",
                "maximum",
                (
                    StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput("selected", ValueKind.ENTITY_SET),
                ),
                ValueKind.NUMERIC_SCALAR,
                current,
                "lấy giá trị cao nhất trong survivor",
            ),
        )
    )
    return ReasoningGraph(
        recipe_id, full, (*predicate_roles, target_role), tuple(nodes), "result"
    )


def build_sign_cohort_gap(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    cohort_role: MetricRoleInput,
    target_role: MetricRoleInput,
    periods: tuple[str, ...] | None = None,
) -> ReasoningGraph:
    domain = Domain(entities, periods or (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "cohort_values",
            "extract",
            (cohort_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích biến chia cohort",
        ),
        _node(
            "positive_cohort",
            "filter",
            (StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort có giá trị dương",
            operator=">",
            threshold_source="convention",
            threshold_value=0.0,
        ),
        _node(
            "negative_cohort",
            "filter",
            (StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort có giá trị âm",
            operator="<",
            threshold_source="convention",
            threshold_value=0.0,
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu đích theo công ty",
        ),
        _node(
            "positive_average",
            "cohort_average",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("positive_cohort", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính bình quân chỉ tiêu ở cohort dương",
        ),
        _node(
            "negative_average",
            "cohort_average",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("negative_cohort", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính bình quân chỉ tiêu ở cohort âm",
        ),
        _node(
            "result",
            "scalar_difference",
            (
                StepOutputInput("positive_average", ValueKind.NUMERIC_SCALAR),
                StepOutputInput("negative_average", ValueKind.NUMERIC_SCALAR),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "lấy phần cao hơn giữa hai mức bình quân",
            require_positive=True,
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (cohort_role, target_role), nodes, "result"
    )


def build_dual_panel_predicate_count(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    prior: str,
    current: str,
    report_scope: str,
    growth_role: MetricRoleInput,
    difference_role: MetricRoleInput,
) -> ReasoningGraph:
    periods = (prior, current)
    domain = Domain(entities, periods, report_scope)
    current_domain = Domain(entities, (current,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "growth_source",
            "extract",
            (growth_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            domain,
            "trích panel chỉ tiêu tăng trưởng",
        ),
        _node(
            "growth_panel",
            "growth",
            (StepOutputInput("growth_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            current_domain,
            "tính tăng trưởng giữa hai kỳ",
            period_universe=periods,
        ),
        _node(
            "growth_current",
            "slice_period",
            (StepOutputInput("growth_panel", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            "chiếu tăng trưởng về kỳ cuối",
            period=current,
        ),
        _node(
            "positive_growth",
            "predicate_set",
            (StepOutputInput("growth_current", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lập tập công ty có tăng trưởng dương",
            operator=">",
            threshold_value=0.0,
        ),
        _node(
            "difference_source",
            "extract",
            (difference_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            domain,
            "trích panel chỉ tiêu biên",
        ),
        _node(
            "difference_panel",
            "period_difference",
            (
                StepOutputInput(
                    "difference_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            current_domain,
            "tính thay đổi chỉ tiêu biên",
            period_universe=periods,
        ),
        _node(
            "difference_current",
            "slice_period",
            (
                StepOutputInput(
                    "difference_panel", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            "chiếu thay đổi về kỳ cuối",
            period=current,
        ),
        _node(
            "negative_difference",
            "predicate_set",
            (StepOutputInput("difference_current", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lập tập công ty có chỉ tiêu biên giảm",
            operator="<",
            threshold_value=0.0,
        ),
        _node(
            "matched",
            "set_intersection",
            (
                StepOutputInput("positive_growth", ValueKind.ENTITY_SET),
                StepOutputInput("negative_difference", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lấy giao hai cohort",
        ),
        _node(
            "result",
            "set_count",
            (StepOutputInput("matched", ValueKind.ENTITY_SET),),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "đếm công ty thỏa đồng thời hai điều kiện",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (growth_role, difference_role), nodes, "result"
    )


def build_temporal_growth_average(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    prior: str,
    current: str,
    report_scope: str,
    temporal_role: MetricRoleInput,
    growth_role: MetricRoleInput,
    threshold: float = 0.0,
) -> ReasoningGraph:
    periods = (prior, current)
    domain = Domain(entities, periods, report_scope)
    current_domain = Domain(entities, (current,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "temporal_values",
            "extract",
            (temporal_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            domain,
            "trích panel điều kiện theo thời gian",
        ),
        _node(
            "persistent_set",
            "temporal_all",
            (
                StepOutputInput(
                    "temporal_values", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc công ty thỏa điều kiện ở cả hai kỳ",
            operator=">",
            threshold_value=threshold,
        ),
        _node(
            "growth_source",
            "extract",
            (growth_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            domain,
            "trích panel chỉ tiêu tăng trưởng",
        ),
        _node(
            "growth_panel",
            "growth",
            (StepOutputInput("growth_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            current_domain,
            "tính tăng trưởng giữa hai kỳ",
            period_universe=periods,
        ),
        _node(
            "growth_current",
            "slice_period",
            (StepOutputInput("growth_panel", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            "chiếu tăng trưởng về kỳ cuối",
            period=current,
        ),
        _node(
            "result",
            "average",
            (
                StepOutputInput("growth_current", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("persistent_set", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            current_domain,
            "tính tăng trưởng bình quân trong cohort",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (temporal_role, growth_role), nodes, "result"
    )


def build_temporal_growth_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    prior: str,
    current: str,
    report_scope: str,
    temporal_role: MetricRoleInput,
    growth_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    periods = (prior, current)
    domain = Domain(entities, periods, report_scope)
    current_domain = Domain(entities, (current,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "temporal_values",
            "extract",
            (temporal_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            domain,
            "trích panel điều kiện theo thời gian",
        ),
        _node(
            "persistent_set",
            "temporal_all",
            (
                StepOutputInput(
                    "temporal_values", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc công ty thỏa điều kiện ở cả hai kỳ",
            operator=">",
            threshold_value=0.0,
        ),
        _node(
            "growth_source",
            "extract",
            (growth_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            domain,
            "trích panel doanh thu",
        ),
        _node(
            "growth_panel",
            "growth",
            (StepOutputInput("growth_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            current_domain,
            "tính tăng trưởng doanh thu",
            period_universe=periods,
        ),
        _node(
            "growth_current",
            "slice_period",
            (StepOutputInput("growth_panel", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            "chiếu tăng trưởng về kỳ cuối",
            period=current,
        ),
        _node(
            "winner",
            "argmax",
            (
                StepOutputInput("growth_current", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("persistent_set", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_KEY,
            current_domain,
            "chọn công ty tăng trưởng cao nhất trong cohort",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            "trích chỉ tiêu đích tại kỳ cuối",
        ),
        _node(
            "result",
            "lookup",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("winner", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            current_domain,
            "tra chỉ tiêu đích tại công ty đã chọn",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (temporal_role, growth_role, target_role), nodes, "result"
    )


def build_panel_filter_average(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    prior: str,
    current: str,
    report_scope: str,
    growth_role: MetricRoleInput,
    difference_role: MetricRoleInput,
) -> ReasoningGraph:
    periods = (prior, current)
    domain = Domain(entities, periods, report_scope)
    current_domain = Domain(entities, (current,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "growth_source",
            "extract",
            (growth_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            domain,
            "trích panel doanh thu",
        ),
        _node(
            "growth_panel",
            "growth",
            (StepOutputInput("growth_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            current_domain,
            "tính tăng trưởng doanh thu",
            period_universe=periods,
        ),
        _node(
            "growth_current",
            "slice_period",
            (StepOutputInput("growth_panel", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            "chiếu tăng trưởng về kỳ cuối",
            period=current,
        ),
        _node(
            "selected",
            "filter",
            (StepOutputInput("growth_current", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc công ty có doanh thu tăng",
            operator=">",
            threshold_source="convention",
            threshold_value=0.0,
        ),
        _node(
            "difference_source",
            "extract",
            (difference_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            domain,
            "trích panel biên lợi nhuận",
        ),
        _node(
            "difference_panel",
            "period_difference",
            (
                StepOutputInput(
                    "difference_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            current_domain,
            "tính thay đổi biên lợi nhuận",
            period_universe=periods,
        ),
        _node(
            "difference_current",
            "slice_period",
            (
                StepOutputInput(
                    "difference_panel", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            "chiếu thay đổi biên về kỳ cuối",
            period=current,
        ),
        _node(
            "result",
            "average",
            (
                StepOutputInput("difference_current", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            current_domain,
            "tính thay đổi biên bình quân trong cohort doanh thu tăng",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (growth_role, difference_role), nodes, "result"
    )


def build_entity_filter_aggregate(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    filter_role: MetricRoleInput,
    target_role: MetricRoleInput,
    operator: str,
    threshold: float,
    aggregate_operation: str,
) -> ReasoningGraph:
    if aggregate_operation not in {"sum", "average", "minimum", "maximum"}:
        raise ValueError(
            f"invalid entity_filter_aggregate aggregate_operation: {aggregate_operation!r}"
        )
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "filter_values",
            "extract",
            (filter_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu điều kiện",
        ),
        _node(
            "selected",
            "filter",
            (StepOutputInput("filter_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort theo điều kiện tài chính",
            operator=operator,
            threshold_source="convention",
            threshold_value=threshold,
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu đích",
        ),
        _node(
            "result",
            aggregate_operation,
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính reducer chỉ tiêu đích trong cohort",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (filter_role, target_role), nodes, "result"
    )


def build_temporal_filter_aggregate(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    filter_role: MetricRoleInput,
    target_role: MetricRoleInput,
    operator: str,
    threshold: float,
    aggregate_operation: str,
) -> ReasoningGraph:
    if aggregate_operation not in {"sum", "average", "minimum", "maximum"}:
        raise ValueError(
            f"invalid temporal_filter_aggregate aggregate_operation: {aggregate_operation!r}"
        )
    full_domain = Domain(entities, periods, report_scope)
    reference_domain = Domain(entities, (reference_period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "temporal_values",
            "extract",
            (filter_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            full_domain,
            "trích panel chỉ tiêu điều kiện theo toàn bộ window",
        ),
        _node(
            "selected",
            "temporal_all",
            (
                StepOutputInput(
                    "temporal_values", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort thỏa điều kiện ở mọi năm trong window",
            operator=operator,
            threshold_value=threshold,
            period_universe=periods,
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            reference_domain,
            "trích chỉ tiêu đích tại năm tham chiếu",
        ),
        _node(
            "result",
            aggregate_operation,
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            reference_domain,
            "tính reducer chỉ tiêu đích tại năm tham chiếu trong cohort",
        ),
    )
    return ReasoningGraph(
        recipe_id, full_domain, (filter_role, target_role), nodes, "result"
    )


def build_temporal_filter_rank_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    filter_role: MetricRoleInput,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    operator: str,
    threshold: float,
    transform: str,
    selector_direction: str,
) -> ReasoningGraph:
    full_domain = Domain(entities, periods, report_scope)
    reference_domain = Domain(entities, (reference_period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes: list[ReasoningNode] = [
        _node(
            "temporal_values",
            "extract",
            (filter_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            full_domain,
            "trích panel chỉ tiêu điều kiện theo toàn bộ window",
        ),
        _node(
            "selected",
            "temporal_all",
            (
                StepOutputInput(
                    "temporal_values", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort thỏa điều kiện ở mọi năm trong window",
            operator=operator,
            threshold_value=threshold,
            period_universe=periods,
        ),
    ]

    if transform == "identity":
        rank_step = "rank_values"
        nodes.append(
            _node(
                "rank_values",
                "extract",
                (rank_role,),
                ValueKind.NUMERIC_SERIES_ENTITY,
                reference_domain,
                "trích chỉ tiêu xếp hạng tại năm tham chiếu",
            )
        )
    elif transform in {"growth", "period_difference"}:
        transformed_domain = Domain(entities, periods[1:], report_scope)
        rank_step = "rank_reference_values"
        nodes.extend(
            [
                _node(
                    "rank_source",
                    "extract",
                    (rank_role,),
                    ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
                    full_domain,
                    "trích panel chỉ tiêu xếp hạng",
                ),
                _node(
                    "rank_panel",
                    transform,
                    (
                        StepOutputInput(
                            "rank_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                        ),
                    ),
                    ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
                    transformed_domain,
                    "biến đổi panel chỉ tiêu xếp hạng theo thời gian",
                    period_universe=periods,
                ),
                _node(
                    "rank_reference_values",
                    "slice_period",
                    (
                        StepOutputInput(
                            "rank_panel", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                        ),
                    ),
                    ValueKind.NUMERIC_SERIES_ENTITY,
                    reference_domain,
                    "chiếu chỉ tiêu xếp hạng về năm tham chiếu",
                    period=reference_period,
                ),
            ]
        )
    else:
        raise ValueError(
            f"invalid temporal_filter_rank_lookup transform: {transform!r}"
        )

    selector_operation = {"argmax": "argmax", "argmin": "argmin"}.get(
        selector_direction
    )
    if selector_operation is None:
        raise ValueError(
            f"invalid temporal_filter_rank_lookup selector_direction: {selector_direction!r}"
        )

    nodes.extend(
        [
            _node(
                "winner",
                selector_operation,
                (
                    StepOutputInput(rank_step, ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput("selected", ValueKind.ENTITY_SET),
                ),
                ValueKind.ENTITY_KEY,
                reference_domain,
                "chọn công ty theo chỉ tiêu xếp hạng trong cohort đã lọc",
            ),
            _node(
                "target_values",
                "extract",
                (target_role,),
                ValueKind.NUMERIC_SERIES_ENTITY,
                reference_domain,
                "trích chỉ tiêu đích tại năm tham chiếu",
            ),
            _node(
                "result",
                "lookup",
                (
                    StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput("winner", ValueKind.ENTITY_KEY),
                ),
                ValueKind.NUMERIC_SCALAR,
                reference_domain,
                "tra chỉ tiêu đích tại đúng công ty đã chọn",
            ),
        ]
    )
    return ReasoningGraph(
        recipe_id,
        full_domain,
        (filter_role, rank_role, target_role),
        tuple(nodes),
        "result",
    )


def build_period_filter_aggregate(
    *,
    recipe_id: str,
    entity: str,
    periods: tuple[str, ...],
    report_scope: str,
    filter_role: MetricRoleInput,
    target_role: MetricRoleInput,
    operator: str,
    threshold: float,
    aggregate_operation: str,
) -> ReasoningGraph:
    if aggregate_operation not in {"sum", "average", "minimum", "maximum"}:
        raise ValueError(
            f"invalid period_filter_aggregate aggregate_operation: {aggregate_operation!r}"
        )
    domain = Domain((entity,), periods, report_scope)
    nodes = (
        _node(
            "filter_values",
            "extract",
            (filter_role,),
            ValueKind.NUMERIC_SERIES_PERIOD,
            domain,
            "trích chuỗi chỉ tiêu điều kiện theo năm",
        ),
        _node(
            "selected",
            "filter",
            (StepOutputInput("filter_values", ValueKind.NUMERIC_SERIES_PERIOD),),
            ValueKind.PERIOD_SET,
            domain,
            "lọc các năm theo điều kiện tài chính",
            operator=operator,
            threshold_source="convention",
            threshold_value=threshold,
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_PERIOD,
            domain,
            "trích chuỗi chỉ tiêu đích theo năm",
        ),
        _node(
            "result",
            aggregate_operation,
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_PERIOD),
                StepOutputInput("selected", ValueKind.PERIOD_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính reducer chỉ tiêu đích trên các năm được lọc",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (filter_role, target_role), nodes, "result"
    )


def build_period_filter_rank_lookup(
    *,
    recipe_id: str,
    entity: str,
    periods: tuple[str, ...],
    report_scope: str,
    filter_role: MetricRoleInput,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    operator: str,
    threshold: float,
    selector_direction: str,
) -> ReasoningGraph:
    selector_operation = {"argmax": "argmax", "argmin": "argmin"}.get(
        selector_direction
    )
    if selector_operation is None:
        raise ValueError(
            f"invalid period_filter_rank_lookup selector_direction: {selector_direction!r}"
        )
    domain = Domain((entity,), periods, report_scope)
    nodes = (
        _node(
            "filter_values",
            "extract",
            (filter_role,),
            ValueKind.NUMERIC_SERIES_PERIOD,
            domain,
            "trích chuỗi chỉ tiêu điều kiện theo năm",
        ),
        _node(
            "selected",
            "filter",
            (StepOutputInput("filter_values", ValueKind.NUMERIC_SERIES_PERIOD),),
            ValueKind.PERIOD_SET,
            domain,
            "lọc các năm theo điều kiện tài chính",
            operator=operator,
            threshold_source="convention",
            threshold_value=threshold,
        ),
        _node(
            "rank_values",
            "extract",
            (rank_role,),
            ValueKind.NUMERIC_SERIES_PERIOD,
            domain,
            "trích chuỗi chỉ tiêu xếp hạng theo năm",
        ),
        _node(
            "winner",
            selector_operation,
            (
                StepOutputInput("rank_values", ValueKind.NUMERIC_SERIES_PERIOD),
                StepOutputInput("selected", ValueKind.PERIOD_SET),
            ),
            ValueKind.PERIOD_KEY,
            domain,
            "chọn năm theo chỉ tiêu xếp hạng trong các năm đã lọc",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_PERIOD,
            domain,
            "trích chuỗi chỉ tiêu đích theo năm",
        ),
        _node(
            "result",
            "lookup",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_PERIOD),
                StepOutputInput("winner", ValueKind.PERIOD_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tra chỉ tiêu đích tại đúng năm đã chọn",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (filter_role, rank_role, target_role), nodes, "result"
    )


def build_filter_average(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    filter_role: MetricRoleInput,
    target_role: MetricRoleInput,
    operator: str,
    threshold: float,
) -> ReasoningGraph:
    return build_entity_filter_aggregate(
        recipe_id=recipe_id,
        entities=entities,
        period=period,
        report_scope=report_scope,
        filter_role=filter_role,
        target_role=target_role,
        operator=operator,
        threshold=threshold,
        aggregate_operation="average",
    )


def build_dual_predicate_count(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    first_role: MetricRoleInput,
    first_operator: str,
    second_role: MetricRoleInput,
    second_operator: str,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "first_values",
            "extract",
            (first_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu điều kiện thứ nhất",
        ),
        _node(
            "first_set",
            "predicate_set",
            (StepOutputInput("first_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lập tập theo điều kiện thứ nhất",
            operator=first_operator,
            threshold_value=0.0,
        ),
        _node(
            "second_values",
            "extract",
            (second_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu điều kiện thứ hai",
        ),
        _node(
            "second_set",
            "predicate_set",
            (StepOutputInput("second_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lập tập theo điều kiện thứ hai",
            operator=second_operator,
            threshold_value=0.0,
        ),
        _node(
            "matched",
            "set_intersection",
            (
                StepOutputInput("first_set", ValueKind.ENTITY_SET),
                StepOutputInput("second_set", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lấy giao hai cohort",
        ),
        _node(
            "result",
            "set_count",
            (StepOutputInput("matched", ValueKind.ENTITY_SET),),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "đếm công ty thỏa đồng thời hai điều kiện",
        ),
    )
    return ReasoningGraph(recipe_id, domain, (first_role, second_role), nodes, "result")


def build_derived_threshold_share(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    selector_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    """Above-median cohort contribution / total population contribution."""
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "selector_values",
            "extract",
            (selector_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu chia cohort",
        ),
        _node(
            "peer_median",
            "median",
            (StepOutputInput("selector_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.THRESHOLD,
            domain,
            "tính trung vị runtime của peer group",
        ),
        _node(
            "selected",
            "filter",
            (
                StepOutputInput("selector_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("peer_median", ValueKind.THRESHOLD),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort cao hơn trung vị",
            operator=">",
            threshold_source="derived",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích contribution metric",
        ),
        _node(
            "selected_total",
            "restricted_sum",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính contribution của cohort",
        ),
        _node(
            "group_total",
            "sum",
            (StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính contribution của toàn nhóm",
        ),
        _node(
            "result",
            "scalar_share",
            (
                StepOutputInput("selected_total", ValueKind.NUMERIC_SCALAR),
                StepOutputInput("group_total", ValueKind.NUMERIC_SCALAR),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính tỷ trọng cohort trên toàn nhóm",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (selector_role, target_role), nodes, "result"
    )


def _panel_value_nodes(
    *,
    prefix: str,
    role: MetricRoleInput,
    transform: str,
    full_domain: Domain,
    periods: tuple[str, ...],
    period: str,
) -> tuple[list[ReasoningNode], str]:
    """Materialize one entity series at ``period`` from a bound panel."""
    if period not in periods:
        raise ValueError(f"{prefix}: periods {period!r} is not in the panel {periods!r}")
    current_domain = Domain(full_domain.entities, (period,), full_domain.report_scope)
    if (
        transform == "identity"
        and role.expected_kind == ValueKind.NUMERIC_SERIES_ENTITY
    ):
        return [
            _node(
                f"{prefix}_values",
                "extract",
                (role,),
                ValueKind.NUMERIC_SERIES_ENTITY,
                current_domain,
                f"trích series {prefix} tại {period}",
            )
        ], f"{prefix}_values"
    if role.expected_kind != ValueKind.NUMERIC_SERIES_ENTITY_PERIOD:
        raise ValueError(
            f"{prefix}: transform {transform!r} requires an entity-period role; got {role.expected_kind}"
        )
    nodes = [
        _node(
            f"{prefix}_source",
            "extract",
            (role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            full_domain,
            f"trích panel {prefix}",
        )
    ]
    source_step = f"{prefix}_source"
    if transform in {"growth", "period_difference"}:
        transformed_domain = Domain(
            full_domain.entities, periods[1:], full_domain.report_scope
        )
        nodes.append(
            _node(
                f"{prefix}_transformed",
                transform,
                (StepOutputInput(source_step, ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
                ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
                transformed_domain,
                f"biến đổi panel {prefix}",
                period_universe=periods,
            )
        )
        source_step = f"{prefix}_transformed"
    elif transform != "identity":
        raise ValueError(f"{prefix}: unsupported transform {transform!r}")
    nodes.append(
        _node(
            f"{prefix}_values",
            "slice_period",
            (StepOutputInput(source_step, ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current_domain,
            f"chiếu {prefix} về {period}",
            period=period,
        )
    )
    return nodes, f"{prefix}_values"


def build_filtered_transform_rank_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    rank_transform: str,
    target_transform: str,
    selector_direction: str,
    filter_role: MetricRoleInput | None = None,
    filter_transform: str = "identity",
    filter_operator: str = ">",
    filter_threshold: float = 0.0,
    filter_period: str | None = None,
    rank_period: str | None = None,
    target_period: str | None = None,
    transform_periods: tuple[str, ...] | None = None,
) -> ReasoningGraph:
    """Optional fixed filter -> transformed rank -> same-key transformed lookup."""
    if len(periods) < 2:
        raise ValueError("filtered_transform_rank_lookup requires at least two periods")
    selector_operation = {"argmax": "argmax", "argmin": "argmin"}.get(
        selector_direction
    )
    if selector_operation is None:
        raise ValueError(f"invalid selector_direction: {selector_direction!r}")
    full = Domain(entities, periods, report_scope)
    set_domain = Domain(entities, (), report_scope)
    filter_period = filter_period or periods[-1]
    rank_period = rank_period or periods[-1]
    target_period = target_period or periods[-1]
    transform_periods = transform_periods or periods
    transform_domain = Domain(entities, transform_periods, report_scope)
    nodes: list[ReasoningNode] = []
    selected_input: StepOutputInput | None = None
    roles: list[MetricRoleInput] = []
    if filter_role is not None:
        filter_nodes, filter_step = _panel_value_nodes(
            prefix="filter",
            role=filter_role,
            transform=filter_transform,
            full_domain=transform_domain if filter_transform != "identity" else full,
            periods=transform_periods if filter_transform != "identity" else periods,
            period=filter_period,
        )
        nodes.extend(filter_nodes)
        nodes.append(
            _node(
                "selected",
                "predicate_set",
                (StepOutputInput(filter_step, ValueKind.NUMERIC_SERIES_ENTITY),),
                ValueKind.ENTITY_SET,
                set_domain,
                "lọc universe trước khi xếp hạng",
                operator=filter_operator,
                threshold_value=filter_threshold,
            )
        )
        selected_input = StepOutputInput("selected", ValueKind.ENTITY_SET)
        roles.append(filter_role)
    rank_nodes, rank_step = _panel_value_nodes(
        prefix="rank",
        role=rank_role,
        transform=rank_transform,
        full_domain=transform_domain if rank_transform != "identity" else full,
        periods=transform_periods if rank_transform != "identity" else periods,
        period=rank_period,
    )
    nodes.extend(rank_nodes)
    selector_inputs = [StepOutputInput(rank_step, ValueKind.NUMERIC_SERIES_ENTITY)]
    if selected_input is not None:
        selector_inputs.append(selected_input)
    nodes.append(
        _node(
            "winner",
            selector_operation,
            tuple(selector_inputs),
            ValueKind.ENTITY_KEY,
            Domain(entities, (rank_period,), report_scope),
            "chọn entity trong cohort đã khóa",
        )
    )
    target_nodes, target_step = _panel_value_nodes(
        prefix="target",
        role=target_role,
        transform=target_transform,
        full_domain=transform_domain if target_transform != "identity" else full,
        periods=transform_periods if target_transform != "identity" else periods,
        period=target_period,
    )
    nodes.extend(target_nodes)
    nodes.append(
        _node(
            "result",
            "lookup",
            (
                StepOutputInput(target_step, ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("winner", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            Domain(entities, (target_period,), report_scope),
            "tra target tại đúng entity được chọn",
        )
    )
    return ReasoningGraph(
        recipe_id, full, (*roles, rank_role, target_role), tuple(nodes), "result"
    )


def build_filter_share(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    filter_role: MetricRoleInput,
    target_role: MetricRoleInput,
    operator: str,
    threshold: float,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "filter_values",
            "extract",
            (filter_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích metric lọc",
        ),
        _node(
            "selected",
            "predicate_set",
            (StepOutputInput("filter_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort",
            operator=operator,
            threshold_value=threshold,
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích metric đóng góp",
        ),
        _node(
            "part",
            "restricted_sum",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tổng trong cohort",
        ),
        _node(
            "whole",
            "sum",
            (StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tổng universe",
        ),
        _node(
            "result",
            "scalar_share",
            (
                StepOutputInput("part", ValueKind.NUMERIC_SCALAR),
                StepOutputInput("whole", ValueKind.NUMERIC_SCALAR),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính tỷ trọng",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (filter_role, target_role), nodes, "result"
    )


def build_transform_rank_share(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput | None,
    selector_direction: str = "argmax",
    positive_only: bool = False,
) -> ReasoningGraph:
    full = Domain(entities, periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    rank_nodes, rank_step = _panel_value_nodes(
        prefix="rank",
        role=rank_role,
        transform="period_difference",
        full_domain=full,
        periods=periods,
        period=periods[-1],
    )
    nodes: list[ReasoningNode] = list(rank_nodes)
    selected: StepOutputInput | None = None
    if positive_only:
        nodes.append(
            _node(
                "positive",
                "predicate_set",
                (StepOutputInput(rank_step, ValueKind.NUMERIC_SERIES_ENTITY),),
                ValueKind.ENTITY_SET,
                set_domain,
                "giữ các mức tăng dương",
                operator=">",
                threshold_value=0.0,
            )
        )
        selected = StepOutputInput("positive", ValueKind.ENTITY_SET)
    selector_inputs = [StepOutputInput(rank_step, ValueKind.NUMERIC_SERIES_ENTITY)]
    if selected is not None:
        selector_inputs.append(selected)
    nodes.append(
        _node(
            "winner",
            selector_direction,
            tuple(selector_inputs),
            ValueKind.ENTITY_KEY,
            current,
            "chọn mức tăng lớn nhất",
        )
    )
    if target_role is None:
        target_step = rank_step
        roles = (rank_role,)
    else:
        nodes.append(
            _node(
                "target_values",
                "extract",
                (target_role,),
                ValueKind.NUMERIC_SERIES_ENTITY,
                current,
                "trích target share",
            )
        )
        target_step = "target_values"
        roles = (rank_role, target_role)
    nodes.append(
        _node(
            "part",
            "lookup",
            (
                StepOutputInput(target_step, ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("winner", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            current,
            "tra đóng góp của winner",
        )
    )
    if target_role is None and selected is not None:
        whole_inputs = (
            StepOutputInput(target_step, ValueKind.NUMERIC_SERIES_ENTITY),
            selected,
        )
        whole_operation = "restricted_sum"
    else:
        whole_inputs = (StepOutputInput(target_step, ValueKind.NUMERIC_SERIES_ENTITY),)
        whole_operation = "sum"
    nodes.append(
        _node(
            "whole",
            whole_operation,
            whole_inputs,
            ValueKind.NUMERIC_SCALAR,
            current,
            "tính tổng mẫu số",
        )
    )
    nodes.append(
        _node(
            "result",
            "scalar_share",
            (
                StepOutputInput("part", ValueKind.NUMERIC_SCALAR),
                StepOutputInput("whole", ValueKind.NUMERIC_SCALAR),
            ),
            ValueKind.NUMERIC_SCALAR,
            current,
            "tính tỷ trọng",
        )
    )
    return ReasoningGraph(recipe_id, full, roles, tuple(nodes), "result")


def build_top_n_dual_predicate_count(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    rank_role: MetricRoleInput,
    first_role: MetricRoleInput,
    second_role: MetricRoleInput,
    n: int,
    first_operator: str,
    first_threshold: float,
    second_operator: str,
    second_threshold: float,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "rank_values",
            "extract",
            (rank_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích metric top-N",
        ),
        _node(
            "top",
            "top_n",
            (StepOutputInput("rank_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "chọn top-N",
            n=n,
        ),
        _node(
            "first_values",
            "extract",
            (first_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích predicate một",
        ),
        _node(
            "first_set",
            "predicate_set",
            (StepOutputInput("first_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc predicate một",
            operator=first_operator,
            threshold_value=first_threshold,
        ),
        _node(
            "second_values",
            "extract",
            (second_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích predicate hai",
        ),
        _node(
            "second_set",
            "predicate_set",
            (StepOutputInput("second_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc predicate hai",
            operator=second_operator,
            threshold_value=second_threshold,
        ),
        _node(
            "selected",
            "set_intersection",
            (
                StepOutputInput("top", ValueKind.ENTITY_SET),
                StepOutputInput("first_set", ValueKind.ENTITY_SET),
                StepOutputInput("second_set", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "giao top-N và hai predicate",
        ),
        _node(
            "result",
            "set_count",
            (StepOutputInput("selected", ValueKind.ENTITY_SET),),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "đếm survivor",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (rank_role, first_role, second_role), nodes, "result"
    )


def build_nested_median_aggregate(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    first_role: MetricRoleInput,
    second_role: MetricRoleInput,
    target_role: MetricRoleInput,
    first_operator: str,
    first_threshold: float,
    second_operator: str,
    aggregate_operation: str,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "first_values",
            "extract",
            (first_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích điều kiện đầu",
        ),
        _node(
            "first_cohort",
            "predicate_set",
            (StepOutputInput("first_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort đầu",
            operator=first_operator,
            threshold_value=first_threshold,
        ),
        _node(
            "second_values",
            "extract",
            (second_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích metric median",
        ),
        _node(
            "cohort_median",
            "cohort_median",
            (
                StepOutputInput("second_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("first_cohort", ValueKind.ENTITY_SET),
            ),
            ValueKind.THRESHOLD,
            domain,
            "tính median trong cohort đầu",
        ),
        _node(
            "below_median",
            "derived_predicate_set",
            (
                StepOutputInput("second_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("cohort_median", ValueKind.THRESHOLD),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc theo median cohort",
            operator=second_operator,
        ),
        _node(
            "selected",
            "set_intersection",
            (
                StepOutputInput("first_cohort", ValueKind.ENTITY_SET),
                StepOutputInput("below_median", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "giữ predicate median bên trong cohort đầu",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích target",
        ),
        _node(
            "result",
            aggregate_operation,
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "aggregate trong cohort lồng",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (first_role, second_role, target_role), nodes, "result"
    )


def build_multi_predicate_terminal(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    predicate_roles: tuple[MetricRoleInput, ...],
    predicate_transforms: tuple[str, ...],
    predicate_operators: tuple[str, ...],
    predicate_thresholds: tuple[float, ...],
    target_role: MetricRoleInput | None,
    aggregate_operation: str,
) -> ReasoningGraph:
    if not (
        len(predicate_roles)
        == len(predicate_transforms)
        == len(predicate_operators)
        == len(predicate_thresholds)
    ):
        raise ValueError("multi_predicate_terminal configuration arrays have different lengths")
    full = Domain(entities, periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes: list[ReasoningNode] = []
    set_inputs: list[StepOutputInput] = []
    for index, (role, transform, operator, threshold) in enumerate(
        zip(
            predicate_roles,
            predicate_transforms,
            predicate_operators,
            predicate_thresholds,
            strict=True,
        )
    ):
        prefix = f"predicate_{index}"
        role_nodes, value_step = _panel_value_nodes(
            prefix=prefix,
            role=role,
            transform=transform,
            full_domain=full,
            periods=periods,
            period=periods[-1],
        )
        nodes.extend(role_nodes)
        set_id = f"set_{index}"
        nodes.append(
            _node(
                set_id,
                "predicate_set",
                (StepOutputInput(value_step, ValueKind.NUMERIC_SERIES_ENTITY),),
                ValueKind.ENTITY_SET,
                set_domain,
                "lập predicate set",
                operator=operator,
                threshold_value=threshold,
            )
        )
        set_inputs.append(StepOutputInput(set_id, ValueKind.ENTITY_SET))
    nodes.append(
        _node(
            "selected",
            "set_intersection",
            tuple(set_inputs),
            ValueKind.ENTITY_SET,
            set_domain,
            "giao các predicate",
        )
    )
    roles: tuple[MetricRoleInput, ...]
    if target_role is None:
        if aggregate_operation != "set_count":
            raise ValueError("terminal must be set_count when there is no target")
        nodes.append(
            _node(
                "result",
                "set_count",
                (StepOutputInput("selected", ValueKind.ENTITY_SET),),
                ValueKind.NUMERIC_SCALAR,
                full,
                "đếm survivor",
            )
        )
        roles = predicate_roles
    else:
        nodes.append(
            _node(
                "target_values",
                "extract",
                (target_role,),
                ValueKind.NUMERIC_SERIES_ENTITY,
                current,
                "trích target",
            )
        )
        nodes.append(
            _node(
                "result",
                aggregate_operation,
                (
                    StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput("selected", ValueKind.ENTITY_SET),
                ),
                ValueKind.NUMERIC_SCALAR,
                current,
                "aggregate target",
            )
        )
        roles = (*predicate_roles, target_role)
    return ReasoningGraph(recipe_id, full, roles, tuple(nodes), "result")


def build_dol_terminal(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    revenue_role: MetricRoleInput,
    operating_profit_role: MetricRoleInput,
    revenue_growth_threshold: float,
    target_role: MetricRoleInput | None = None,
    size_role: MetricRoleInput | None = None,
    size_threshold: float | None = None,
    selector_direction: str = "argmax",
    denominator_epsilon: float = 0.01,
) -> ReasoningGraph:
    full = Domain(entities, periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    revenue_nodes, revenue_step = _panel_value_nodes(
        prefix="revenue",
        role=revenue_role,
        transform="growth",
        full_domain=full,
        periods=periods,
        period=periods[-1],
    )
    profit_nodes, profit_step = _panel_value_nodes(
        prefix="profit",
        role=operating_profit_role,
        transform="growth",
        full_domain=full,
        periods=periods,
        period=periods[-1],
    )
    nodes: list[ReasoningNode] = [*revenue_nodes, *profit_nodes]
    profit_positive_steps: list[str] = []
    for index, period in enumerate(periods):
        level_step = f"profit_level_{index}"
        positive_step = f"profit_positive_{index}"
        nodes.extend(
            (
                _node(
                    level_step,
                    "slice_period",
                    (
                        StepOutputInput(
                            "profit_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                        ),
                    ),
                    ValueKind.NUMERIC_SERIES_ENTITY,
                    Domain(entities, (period,), report_scope),
                    f"chiếu lợi nhuận hoạt động về kỳ {period}",
                    period=period,
                ),
                _node(
                    positive_step,
                    "predicate_set",
                    (StepOutputInput(level_step, ValueKind.NUMERIC_SERIES_ENTITY),),
                    ValueKind.ENTITY_SET,
                    set_domain,
                    f"giữ lợi nhuận hoạt động dương kỳ {period}",
                    operator=">",
                    threshold_value=0.0,
                    eligibility_gate=True,
                ),
            )
        )
        profit_positive_steps.append(positive_step)
    nodes.append(
        _node(
            "growth_set",
            "predicate_set",
            (StepOutputInput(revenue_step, ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc tăng trưởng doanh thu",
            operator=">=",
            threshold_value=revenue_growth_threshold,
        )
    )
    cohort_inputs = [
        StepOutputInput("growth_set", ValueKind.ENTITY_SET),
        *(
            StepOutputInput(step_id, ValueKind.ENTITY_SET)
            for step_id in profit_positive_steps
        ),
    ]
    roles: list[MetricRoleInput] = [revenue_role, operating_profit_role]
    if size_role is not None:
        if size_threshold is None:
            raise ValueError("size_role requires size_threshold")
        size_nodes, size_step = _panel_value_nodes(
            prefix="size",
            role=size_role,
            transform="identity",
            full_domain=full,
            periods=periods,
            period=periods[-1],
        )
        nodes.extend(size_nodes)
        nodes.append(
            _node(
                "size_set",
                "predicate_set",
                (StepOutputInput(size_step, ValueKind.NUMERIC_SERIES_ENTITY),),
                ValueKind.ENTITY_SET,
                set_domain,
                "lọc quy mô",
                operator=">",
                threshold_value=size_threshold,
            )
        )
        cohort_inputs.append(StepOutputInput("size_set", ValueKind.ENTITY_SET))
        roles.insert(0, size_role)
    nodes.append(
        _node(
            "selected",
            "set_intersection",
            tuple(cohort_inputs),
            ValueKind.ENTITY_SET,
            set_domain,
            "giao bộ lọc DOL",
        )
    )
    selected_step = "selected"
    nodes.append(
        _node(
            "dol_values",
            "series_divide",
            (
                StepOutputInput(profit_step, ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput(revenue_step, ValueKind.NUMERIC_SERIES_ENTITY),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current,
            "tính DOL",
            denominator_epsilon=denominator_epsilon,
        )
    )
    if target_role is None:
        nodes.append(
            _node(
                "result",
                "maximum",
                (
                    StepOutputInput("dol_values", ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput(selected_step, ValueKind.ENTITY_SET),
                ),
                ValueKind.NUMERIC_SCALAR,
                current,
                "lấy DOL cao nhất",
            )
        )
    else:
        nodes.append(
            _node(
                "winner",
                selector_direction,
                (
                    StepOutputInput("dol_values", ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput(selected_step, ValueKind.ENTITY_SET),
                ),
                ValueKind.ENTITY_KEY,
                current,
                "chọn DOL cực trị",
            )
        )
        nodes.append(
            _node(
                "target_values",
                "extract",
                (target_role,),
                ValueKind.NUMERIC_SERIES_ENTITY,
                current,
                "trích target",
            )
        )
        nodes.append(
            _node(
                "result",
                "lookup",
                (
                    StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                    StepOutputInput("winner", ValueKind.ENTITY_KEY),
                ),
                ValueKind.NUMERIC_SCALAR,
                current,
                "lookup target tại winner",
            )
        )
        roles.append(target_role)
    return ReasoningGraph(recipe_id, full, tuple(roles), tuple(nodes), "result")


def build_transformed_filter_aggregate_terminal(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    filter_role: MetricRoleInput,
    target_role: MetricRoleInput,
    filter_transform: str,
    target_transform: str,
    filter_operator: str,
    filter_threshold: float,
    aggregate_operation: str,
    absolute_terminal: bool = False,
    transform_periods: tuple[str, ...] | None = None,
) -> ReasoningGraph:
    full = Domain(entities, periods, report_scope)
    transform_periods = transform_periods or periods
    transform_domain = Domain(entities, transform_periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    filter_nodes, filter_step = _panel_value_nodes(
        prefix="filter",
        role=filter_role,
        transform=filter_transform,
        full_domain=transform_domain,
        periods=transform_periods,
        period=periods[-1],
    )
    target_nodes, target_step = _panel_value_nodes(
        prefix="target",
        role=target_role,
        transform=target_transform,
        full_domain=transform_domain,
        periods=transform_periods,
        period=periods[-1],
    )
    nodes: list[ReasoningNode] = [*filter_nodes]
    nodes.append(
        _node(
            "selected",
            "predicate_set",
            (StepOutputInput(filter_step, ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort theo transform",
            operator=filter_operator,
            threshold_value=filter_threshold,
        )
    )
    nodes.extend(target_nodes)
    terminal_id = "aggregate_result" if absolute_terminal else "result"
    nodes.append(
        _node(
            terminal_id,
            aggregate_operation,
            (
                StepOutputInput(target_step, ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            current,
            "aggregate target đã transform",
        )
    )
    if absolute_terminal:
        nodes.append(
            _node(
                "result",
                "scalar_absolute",
                (StepOutputInput(terminal_id, ValueKind.NUMERIC_SCALAR),),
                ValueKind.NUMERIC_SCALAR,
                current,
                "lấy độ lớn tuyệt đối",
            )
        )
    return ReasoningGraph(
        recipe_id, full, (filter_role, target_role), tuple(nodes), "result"
    )


def build_filtered_ranked_cohort_gap(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    size_role: MetricRoleInput,
    cohort_role: MetricRoleInput,
    target_role: MetricRoleInput,
    size_threshold: float,
    percent: float,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "size_values",
            "extract",
            (size_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích quy mô",
        ),
        _node(
            "survivors",
            "predicate_set",
            (StepOutputInput("size_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc quy mô",
            operator=">",
            threshold_value=size_threshold,
        ),
        _node(
            "cohort_values",
            "extract",
            (cohort_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích metric chia cohort",
        ),
        _node(
            "survivor_values",
            "series_restrict",
            (
                StepOutputInput("cohort_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("survivors", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "restrict metric cohort",
        ),
        _node(
            "top",
            "ranked_cohort",
            (StepOutputInput("survivor_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "chọn quantile cao",
            side="top",
            percent=percent,
            minimum_size=3,
        ),
        _node(
            "bottom",
            "ranked_cohort",
            (StepOutputInput("survivor_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "chọn quantile thấp",
            side="bottom",
            percent=percent,
            minimum_size=3,
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích target",
        ),
        _node(
            "top_average",
            "cohort_average",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("top", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "bình quân top",
        ),
        _node(
            "bottom_average",
            "cohort_average",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("bottom", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "bình quân bottom",
        ),
        _node(
            "result",
            "scalar_difference",
            (
                StepOutputInput("top_average", ValueKind.NUMERIC_SCALAR),
                StepOutputInput("bottom_average", ValueKind.NUMERIC_SCALAR),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "chênh lệch hai cohort",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (size_role, cohort_role, target_role), nodes, "result"
    )


def build_transition_filter_rank_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    transition_role: MetricRoleInput,
    size_role: MetricRoleInput,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    size_threshold: float,
    selector_direction: str,
) -> ReasoningGraph:
    full = Domain(entities, periods, report_scope)
    current = Domain(entities, (periods[-1],), report_scope)
    prior = Domain(entities, (periods[-2],), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "transition_source",
            "extract",
            (transition_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            full,
            "trích panel chuyển trạng thái",
        ),
        _node(
            "prior_values",
            "slice_period",
            (
                StepOutputInput(
                    "transition_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY,
            prior,
            "chiếu kỳ trước",
            period=periods[-2],
        ),
        _node(
            "prior_set",
            "predicate_set",
            (StepOutputInput("prior_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc kỳ trước không âm",
            operator=">=",
            threshold_value=0.0,
        ),
        _node(
            "current_values",
            "slice_period",
            (
                StepOutputInput(
                    "transition_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
            ),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current,
            "chiếu kỳ cuối",
            period=periods[-1],
        ),
        _node(
            "current_set",
            "predicate_set",
            (StepOutputInput("current_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc kỳ cuối âm",
            operator="<",
            threshold_value=0.0,
        ),
        _node(
            "size_values",
            "extract",
            (size_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current,
            "trích quy mô",
        ),
        _node(
            "size_set",
            "predicate_set",
            (StepOutputInput("size_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc quy mô",
            operator=">",
            threshold_value=size_threshold,
        ),
        _node(
            "selected",
            "set_intersection",
            (
                StepOutputInput("prior_set", ValueKind.ENTITY_SET),
                StepOutputInput("current_set", ValueKind.ENTITY_SET),
                StepOutputInput("size_set", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "giao transition và quy mô",
        ),
        _node(
            "rank_values",
            "extract",
            (rank_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current,
            "trích rank",
        ),
        _node(
            "winner",
            selector_direction,
            (
                StepOutputInput("rank_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.ENTITY_KEY,
            current,
            "chọn winner",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            current,
            "trích target",
        ),
        _node(
            "result",
            "lookup",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("winner", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            current,
            "lookup target",
        ),
    )
    return ReasoningGraph(
        recipe_id,
        full,
        (transition_role, size_role, rank_role, target_role),
        nodes,
        "result",
    )


def build_rank_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    selector_direction: str = "argmax",
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    selector_operation = {"argmax": "argmax", "argmin": "argmin"}.get(
        selector_direction
    )
    if selector_operation is None:
        raise ValueError(
            f"invalid rank_lookup selector_direction: {selector_direction!r}"
        )
    nodes = (
        _node(
            "rank_values",
            "extract",
            (rank_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu xếp hạng",
        ),
        _node(
            "winner",
            selector_operation,
            (StepOutputInput("rank_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.ENTITY_KEY,
            domain,
            "chọn công ty theo chỉ tiêu xếp hạng",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu đích",
        ),
        _node(
            "result",
            "lookup",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("winner", ValueKind.ENTITY_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tra chỉ tiêu đích tại công ty đã chọn",
        ),
    )
    return ReasoningGraph(recipe_id, domain, (rank_role, target_role), nodes, "result")


def build_panel_rank_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    transform: str,
    selector_direction: str,
) -> ReasoningGraph:
    full_domain = Domain(entities, periods, report_scope)
    candidate_periods = (
        periods[1:] if transform in {"growth", "period_difference"} else periods
    )
    candidate_domain = Domain(entities, candidate_periods, report_scope)
    rank_step = "rank_source"
    nodes: list[ReasoningNode] = [
        _node(
            "rank_source",
            "extract",
            (rank_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            full_domain
            if transform in {"growth", "period_difference"}
            else candidate_domain,
            "trích panel chỉ tiêu xếp hạng",
        )
    ]
    if transform == "identity":
        rank_step = "rank_source"
    elif transform in {"growth", "period_difference"}:
        rank_step = "rank_values"
        nodes.append(
            _node(
                "rank_values",
                transform,
                (
                    StepOutputInput(
                        "rank_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                    ),
                ),
                ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
                candidate_domain,
                "biến đổi panel chỉ tiêu xếp hạng theo thời gian",
                period_universe=periods,
            )
        )
    else:
        raise ValueError(f"invalid panel_rank_lookup transform: {transform!r}")

    selector_operation = {"argmax": "argmax", "argmin": "argmin"}.get(
        selector_direction
    )
    if selector_operation is None:
        raise ValueError(
            f"invalid panel_rank_lookup selector_direction: {selector_direction!r}"
        )

    nodes.extend(
        [
            _node(
                "winner",
                selector_operation,
                (StepOutputInput(rank_step, ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
                ValueKind.ENTITY_PERIOD_KEY,
                candidate_domain,
                "chọn company-year theo chỉ tiêu xếp hạng",
            ),
            _node(
                "target_values",
                "extract",
                (target_role,),
                ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
                candidate_domain,
                "trích panel chỉ tiêu đích cho toàn bộ company-year có thể được chọn",
            ),
            _node(
                "result",
                "lookup",
                (
                    StepOutputInput(
                        "target_values", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                    ),
                    StepOutputInput("winner", ValueKind.ENTITY_PERIOD_KEY),
                ),
                ValueKind.NUMERIC_SCALAR,
                candidate_domain,
                "tra chỉ tiêu đích tại đúng company-year đã chọn",
            ),
        ]
    )
    return ReasoningGraph(
        recipe_id, full_domain, (rank_role, target_role), tuple(nodes), "result"
    )


def build_dual_transform_panel_rank_lookup(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    graph_periods: tuple[str, ...],
    source_periods: tuple[str, ...],
    report_scope: str,
    rank_role: MetricRoleInput,
    target_role: MetricRoleInput,
    rank_transform: str,
    target_transform: str,
    selector_direction: str,
    require_positive_selector_value: bool = False,
) -> ReasoningGraph:
    if len(source_periods) < 2:
        raise ValueError("dual_transform_panel_rank_lookup requires at least two source periods")
    transforms = {"growth", "period_difference"}
    if rank_transform not in transforms or target_transform not in transforms:
        raise ValueError(
            "dual_transform_panel_rank_lookup supports only growth|period_difference for both branches"
        )
    selector_operation = {"argmax": "argmax", "argmin": "argmin"}.get(
        selector_direction
    )
    if selector_operation is None:
        raise ValueError(f"invalid selector_direction: {selector_direction!r}")

    full_domain = Domain(entities, graph_periods, report_scope)
    source_domain = Domain(entities, source_periods, report_scope)
    candidate_periods = source_periods[1:]
    candidate_domain = Domain(entities, candidate_periods, report_scope)
    nodes = (
        _node(
            "rank_source",
            "extract",
            (rank_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            source_domain,
            "trích panel chỉ tiêu xếp hạng",
        ),
        _node(
            "rank_values",
            rank_transform,
            (StepOutputInput("rank_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            candidate_domain,
            "tính biến động chỉ tiêu xếp hạng theo thời gian",
            period_universe=source_periods,
        ),
        _node(
            "winner",
            selector_operation,
            (StepOutputInput("rank_values", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.ENTITY_PERIOD_KEY,
            candidate_domain,
            "chọn company-year theo biến động chỉ tiêu xếp hạng",
            require_positive=require_positive_selector_value,
        ),
        _node(
            "target_source",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            source_domain,
            "trích panel chỉ tiêu đích",
        ),
        _node(
            "target_values",
            target_transform,
            (StepOutputInput("target_source", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
            candidate_domain,
            "tính biến động chỉ tiêu đích theo thời gian",
            period_universe=source_periods,
        ),
        _node(
            "result",
            "lookup",
            (
                StepOutputInput(
                    "target_values", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD
                ),
                StepOutputInput("winner", ValueKind.ENTITY_PERIOD_KEY),
            ),
            ValueKind.NUMERIC_SCALAR,
            candidate_domain,
            "tra biến động chỉ tiêu đích tại đúng company-year đã chọn",
        ),
    )
    return ReasoningGraph(
        recipe_id, full_domain, (rank_role, target_role), nodes, "result"
    )


def build_derived_threshold_average(
    *,
    recipe_id: str,
    entities: tuple[str, ...],
    period: str,
    report_scope: str,
    selector_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    domain = Domain(entities, (period,), report_scope)
    set_domain = Domain(entities, (), report_scope)
    nodes = (
        _node(
            "selector_values",
            "extract",
            (selector_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu chia cohort",
        ),
        _node(
            "peer_median",
            "median",
            (StepOutputInput("selector_values", ValueKind.NUMERIC_SERIES_ENTITY),),
            ValueKind.THRESHOLD,
            domain,
            "tính trung vị runtime của peer group",
        ),
        _node(
            "selected",
            "filter",
            (
                StepOutputInput("selector_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("peer_median", ValueKind.THRESHOLD),
            ),
            ValueKind.ENTITY_SET,
            set_domain,
            "lọc cohort cao hơn trung vị",
            operator=">",
            threshold_source="derived",
        ),
        _node(
            "target_values",
            "extract",
            (target_role,),
            ValueKind.NUMERIC_SERIES_ENTITY,
            domain,
            "trích chỉ tiêu đích",
        ),
        _node(
            "result",
            "average",
            (
                StepOutputInput("target_values", ValueKind.NUMERIC_SERIES_ENTITY),
                StepOutputInput("selected", ValueKind.ENTITY_SET),
            ),
            ValueKind.NUMERIC_SCALAR,
            domain,
            "tính bình quân chỉ tiêu đích trong cohort",
        ),
    )
    return ReasoningGraph(
        recipe_id, domain, (selector_role, target_role), nodes, "result"
    )
