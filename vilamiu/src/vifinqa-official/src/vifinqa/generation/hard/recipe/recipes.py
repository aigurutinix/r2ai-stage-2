
from __future__ import annotations

from vifinqa.generation.hard.recipe.base import (
    Domain,
    MetricRoleInput,
    ReasoningGraph,
    ReasoningNode,
    StepOutputInput,
    ValueKind,
)

R1_RECIPE_ID = "entity_period_growth_argmax_lookup"


def build_r1_graph(
    *,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    report_scope: str,
    revenue_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    if len(periods) < 2:
        raise ValueError("R1 requires a period_window with at least two years (one base year and at least one candidate year)")
    candidate_periods = periods[1:]
    full_domain = Domain(entities=entities, periods=periods, report_scope=report_scope)
    candidate_domain = Domain(entities=entities, periods=candidate_periods, report_scope=report_scope)

    extract_revenue = ReasoningNode(
        step_id="extract_revenue",
        operation="extract",
        inputs=(revenue_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=full_domain,
        params={},
        output="revenue_series",
        description=f"trích {revenue_role.metric_key} theo từng công ty-năm trong giai đoạn xét",
    )
    growth = ReasoningNode(
        step_id="growth",
        operation="growth",
        inputs=(StepOutputInput("extract_revenue", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=candidate_domain,
        params={"period_universe": periods},
        output="growth_series",
        description="tăng trưởng doanh thu thuần so với năm liền trước, theo từng công ty-năm",
    )
    winner = ReasoningNode(
        step_id="winner",
        operation="argmax",
        inputs=(StepOutputInput("growth", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
        output_kind=ValueKind.ENTITY_PERIOD_KEY,
        domain=candidate_domain,
        params={},
        output="winner_key",
        description="chọn công ty-năm có tăng trưởng doanh thu thuần cao nhất trong nhóm",
    )
    extract_target = ReasoningNode(
        step_id="extract_target",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=candidate_domain,
        params={},
        output="target_series",
        description=f"trích {target_role.metric_key} theo từng công ty-năm ứng viên",
    )
    result = ReasoningNode(
        step_id="result",
        operation="lookup",
        inputs=(
            StepOutputInput("extract_target", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),
            StepOutputInput("winner", ValueKind.ENTITY_PERIOD_KEY),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=candidate_domain,
        params={},
        output="result",
        description=f"tra {target_role.metric_key} tại công ty-năm đã chọn",
    )

    return ReasoningGraph(
        recipe_id=R1_RECIPE_ID,
        domain=full_domain,
        metric_roles=(revenue_role, target_role),
        nodes=(extract_revenue, growth, winner, extract_target, result),
        terminal_step_id="result",
    )


R2_RECIPE_ID = "temporal_filter_maximum"
R3_RECIPE_ID = "temporal_filter_argmax_lookup"
R4_RECIPE_ID = "entity_convention_filter_aggregate"
R5_RECIPE_ID = "entity_derived_threshold_filter_aggregate"
R6_RECIPE_ID = "period_convention_filter_select_lookup"
R7_RECIPE_ID = "period_derived_threshold_select_lookup"

AGGREGATE_OPERATIONS = ("sum", "average", "minimum", "maximum")
SELECT_DIRECTIONS = ("argmax", "argmin")
DERIVED_THRESHOLD_STATISTICS = ("median", "average_threshold", "percentile")
FILTER_OPERATORS = (">", "<", ">=", "<=")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def build_r2_graph(
    *,
    entities: tuple[str, ...],
    temporal_periods: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    temporal_role: MetricRoleInput,
    temporal_operator: str,
    temporal_threshold: float,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    _require(len(entities) >= 2, "R2 cần >= 2 entities")
    _require(reference_period in temporal_periods, "R2: reference_period phải nằm trong temporal_periods")
    _require(temporal_operator in FILTER_OPERATORS, f"R2: temporal_operator phải thuộc {FILTER_OPERATORS}")
    _require(
        target_role.metric_key != temporal_role.metric_key,
        "R2: target metric phải khác temporal metric (§8.4 vai trò phân tích khác nhau)",
    )

    panel_domain = Domain(entities=entities, periods=temporal_periods, report_scope=report_scope)
    entity_set_domain = Domain(entities=entities, periods=(), report_scope=report_scope)
    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)

    extract_temporal = ReasoningNode(
        step_id="extract_temporal",
        operation="extract",
        inputs=(temporal_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=panel_domain,
        params={},
        output="temporal_series",
        description=f"trích {temporal_role.metric_key} theo từng công ty-năm trong giai đoạn xét",
    )
    filtered_set = ReasoningNode(
        step_id="filtered_set",
        operation="temporal_all",
        inputs=(StepOutputInput("extract_temporal", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
        output_kind=ValueKind.ENTITY_SET,
        domain=entity_set_domain,
        params={"operator": temporal_operator, "threshold_value": temporal_threshold},
        output="temporal_survivors",
        description=f"lọc công ty có {temporal_role.metric_key} {temporal_operator} {temporal_threshold} ở mọi năm trong giai đoạn",
    )
    extract_target = ReasoningNode(
        step_id="extract_target",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="target_series",
        description=f"trích {target_role.metric_key} tại năm tham chiếu {reference_period}",
    )
    result = ReasoningNode(
        step_id="result",
        operation="maximum",
        inputs=(
            StepOutputInput("extract_target", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("filtered_set", ValueKind.ENTITY_SET),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=reference_domain,
        params={},
        output="result",
        description=f"giá trị {target_role.metric_key} lớn nhất trong nhóm công ty thỏa điều kiện liên tục",
    )

    return ReasoningGraph(
        recipe_id=R2_RECIPE_ID,
        domain=panel_domain,
        metric_roles=(temporal_role, target_role),
        nodes=(extract_temporal, filtered_set, extract_target, result),
        terminal_step_id="result",
    )


def build_r3_graph(
    *,
    entities: tuple[str, ...],
    temporal_periods: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    temporal_role: MetricRoleInput,
    temporal_operator: str,
    temporal_threshold: float,
    rank_role: MetricRoleInput,
    rank_direction: str,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    _require(len(entities) >= 3, "R3 cần >= 3 entities (để survivor set có thể >= 2 sau filter)")
    _require(reference_period in temporal_periods, "R3: reference_period phải nằm trong temporal_periods")
    _require(temporal_operator in FILTER_OPERATORS, f"R3: temporal_operator phải thuộc {FILTER_OPERATORS}")
    _require(rank_direction in SELECT_DIRECTIONS, f"R3: rank_direction phải thuộc {SELECT_DIRECTIONS}")
    _require(rank_role.metric_key != target_role.metric_key, "R3: rank metric (A) phải khác target metric (B)")
    _require(rank_role.metric_key != temporal_role.metric_key, "R3: rank metric phải khác temporal metric")

    panel_domain = Domain(entities=entities, periods=temporal_periods, report_scope=report_scope)
    entity_set_domain = Domain(entities=entities, periods=(), report_scope=report_scope)
    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)

    extract_temporal = ReasoningNode(
        step_id="extract_temporal",
        operation="extract",
        inputs=(temporal_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=panel_domain,
        params={},
        output="temporal_series",
        description=f"trích {temporal_role.metric_key} theo từng công ty-năm trong giai đoạn xét",
    )
    filtered_set = ReasoningNode(
        step_id="filtered_set",
        operation="temporal_all",
        inputs=(StepOutputInput("extract_temporal", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
        output_kind=ValueKind.ENTITY_SET,
        domain=entity_set_domain,
        params={"operator": temporal_operator, "threshold_value": temporal_threshold},
        output="temporal_survivors",
        description=f"lọc công ty có {temporal_role.metric_key} {temporal_operator} {temporal_threshold} ở mọi năm trong giai đoạn",
    )
    extract_rank = ReasoningNode(
        step_id="extract_rank",
        operation="extract",
        inputs=(rank_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="rank_series",
        description=f"trích {rank_role.metric_key} tại năm tham chiếu {reference_period}",
    )
    direction_desc = "cao nhất" if rank_direction == "argmax" else "thấp nhất"
    winner = ReasoningNode(
        step_id="winner",
        operation=rank_direction,
        inputs=(
            StepOutputInput("extract_rank", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("filtered_set", ValueKind.ENTITY_SET),
        ),
        output_kind=ValueKind.ENTITY_KEY,
        domain=reference_domain,
        params={},
        output="winner_key",
        description=f"chọn công ty có {rank_role.metric_key} {direction_desc} trong nhóm thỏa điều kiện",
    )
    extract_target = ReasoningNode(
        step_id="extract_target",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="target_series",
        description=f"trích {target_role.metric_key} tại năm tham chiếu {reference_period}",
    )
    result = ReasoningNode(
        step_id="result",
        operation="lookup",
        inputs=(
            StepOutputInput("extract_target", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("winner", ValueKind.ENTITY_KEY),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=reference_domain,
        params={},
        output="result",
        description=f"tra {target_role.metric_key} tại công ty đã chọn",
    )

    return ReasoningGraph(
        recipe_id=R3_RECIPE_ID,
        domain=panel_domain,
        metric_roles=(temporal_role, rank_role, target_role),
        nodes=(extract_temporal, filtered_set, extract_rank, winner, extract_target, result),
        terminal_step_id="result",
    )


def build_r4_graph(
    *,
    entities: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    filter_role: MetricRoleInput,
    filter_operator: str,
    filter_threshold_value: float,
    target_role: MetricRoleInput,
    aggregate_operation: str,
) -> ReasoningGraph:
    _require(len(entities) >= 2, "R4 cần >= 2 entities")
    _require(filter_operator in FILTER_OPERATORS, f"R4: filter_operator phải thuộc {FILTER_OPERATORS}")
    _require(aggregate_operation in AGGREGATE_OPERATIONS, f"R4: aggregate_operation must be one of {AGGREGATE_OPERATIONS}")
    _require(
        filter_role.metric_key != target_role.metric_key,
        "R4: filter metric (A) phải khác target metric (B, §8.4 vai trò phân tích khác nhau)",
    )

    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)
    entity_set_domain = Domain(entities=entities, periods=(), report_scope=report_scope)

    extract_a = ReasoningNode(
        step_id="extract_a",
        operation="extract",
        inputs=(filter_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="metric_a",
        description=f"trích {filter_role.metric_key} của từng công ty tại năm {reference_period}",
    )
    filtered_set = ReasoningNode(
        step_id="filtered_set",
        operation="filter",
        inputs=(StepOutputInput("extract_a", ValueKind.NUMERIC_SERIES_ENTITY),),
        output_kind=ValueKind.ENTITY_SET,
        domain=entity_set_domain,
        params={"operator": filter_operator, "threshold_source": "convention", "threshold_value": filter_threshold_value},
        output="survivors",
        description=f"lọc công ty có {filter_role.metric_key} {filter_operator} {filter_threshold_value}",
    )
    extract_b = ReasoningNode(
        step_id="extract_b",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="metric_b",
        description=f"trích {target_role.metric_key} của từng công ty tại năm {reference_period}",
    )
    result = ReasoningNode(
        step_id="result",
        operation=aggregate_operation,
        inputs=(
            StepOutputInput("extract_b", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("filtered_set", ValueKind.ENTITY_SET),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=reference_domain,
        params={},
        output="result",
        description=f"{aggregate_operation} {target_role.metric_key} của các công ty thỏa điều kiện",
    )

    return ReasoningGraph(
        recipe_id=R4_RECIPE_ID,
        domain=reference_domain,
        metric_roles=(filter_role, target_role),
        nodes=(extract_a, filtered_set, extract_b, result),
        terminal_step_id="result",
    )


def build_r5_graph(
    *,
    entities: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    filter_role: MetricRoleInput,
    filter_operator: str,
    threshold_statistic: str,
    percentile_value: float | None,
    target_role: MetricRoleInput,
    aggregate_operation: str,
) -> ReasoningGraph:
    _require(len(entities) >= 3, "R5 cần >= 3 entities (threshold cần population đủ lớn để có ý nghĩa)")
    _require(filter_operator in FILTER_OPERATORS, f"R5: filter_operator phải thuộc {FILTER_OPERATORS}")
    _require(
        threshold_statistic in DERIVED_THRESHOLD_STATISTICS,
        f"R5: threshold_statistic phải thuộc {DERIVED_THRESHOLD_STATISTICS}",
    )
    _require(aggregate_operation in AGGREGATE_OPERATIONS, f"R5: aggregate_operation must be one of {AGGREGATE_OPERATIONS}")
    _require(filter_role.metric_key != target_role.metric_key, "R5: filter metric (A) phải khác target metric (B)")
    if threshold_statistic == "percentile":
        _require(percentile_value is not None and 0 < percentile_value < 100, "R5: percentile_value phải trong (0,100)")

    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)
    entity_set_domain = Domain(entities=entities, periods=(), report_scope=report_scope)
    threshold_domain = Domain(entities=entities, periods=(), report_scope=report_scope)

    extract_a = ReasoningNode(
        step_id="extract_a",
        operation="extract",
        inputs=(filter_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="metric_a",
        description=f"trích {filter_role.metric_key} của từng công ty tại năm {reference_period}",
    )
    threshold_params: dict[str, object] = {"percentile": percentile_value} if threshold_statistic == "percentile" else {}
    statistic_desc = {
        "median": "trung vị",
        "average_threshold": "trung bình",
        "percentile": f"phân vị {percentile_value}",
    }[threshold_statistic]
    threshold_node = ReasoningNode(
        step_id="threshold",
        operation=threshold_statistic,
        inputs=(StepOutputInput("extract_a", ValueKind.NUMERIC_SERIES_ENTITY),),
        output_kind=ValueKind.THRESHOLD,
        domain=threshold_domain,
        params=threshold_params,
        output="threshold_value",
        description=f"tính {statistic_desc} của {filter_role.metric_key} trong nhóm",
    )
    filtered_set = ReasoningNode(
        step_id="filtered_set",
        operation="filter",
        inputs=(
            StepOutputInput("extract_a", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("threshold", ValueKind.THRESHOLD),
        ),
        output_kind=ValueKind.ENTITY_SET,
        domain=entity_set_domain,
        params={"operator": filter_operator, "threshold_source": "derived"},
        output="survivors",
        description=f"lọc công ty có {filter_role.metric_key} {filter_operator} {statistic_desc} của nhóm",
    )
    extract_b = ReasoningNode(
        step_id="extract_b",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="metric_b",
        description=f"trích {target_role.metric_key} của từng công ty tại năm {reference_period}",
    )
    result = ReasoningNode(
        step_id="result",
        operation=aggregate_operation,
        inputs=(
            StepOutputInput("extract_b", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("filtered_set", ValueKind.ENTITY_SET),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=reference_domain,
        params={},
        output="result",
        description=f"{aggregate_operation} {target_role.metric_key} của các công ty thỏa điều kiện",
    )

    return ReasoningGraph(
        recipe_id=R5_RECIPE_ID,
        domain=reference_domain,
        metric_roles=(filter_role, target_role),
        nodes=(extract_a, threshold_node, filtered_set, extract_b, result),
        terminal_step_id="result",
    )


def build_r6_graph(
    *,
    ticker: str,
    periods: tuple[str, ...],
    report_scope: str,
    filter_role: MetricRoleInput,
    filter_operator: str,
    filter_threshold_value: float,
    select_role: MetricRoleInput,
    select_direction: str,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    _require(len(periods) >= 3, "R6 cần >= 3 kỳ (để PeriodSet có thể >= 2 sau filter)")
    _require(filter_operator in FILTER_OPERATORS, f"R6: filter_operator phải thuộc {FILTER_OPERATORS}")
    _require(select_direction in SELECT_DIRECTIONS, f"R6: select_direction phải thuộc {SELECT_DIRECTIONS}")
    keys = {filter_role.metric_key, select_role.metric_key, target_role.metric_key}
    _require(len(keys) == 3, "R6: metric A (filter)/B (select)/C (target) phải là 3 metric khác nhau")

    entities = (ticker,)
    panel_domain = Domain(entities=entities, periods=periods, report_scope=report_scope)
    period_set_domain = Domain(entities=entities, periods=(), report_scope=report_scope)

    extract_a = ReasoningNode(
        step_id="extract_a",
        operation="extract",
        inputs=(filter_role,),
        output_kind=ValueKind.NUMERIC_SERIES_PERIOD,
        domain=panel_domain,
        params={},
        output="metric_a",
        description=f"trích {filter_role.metric_key} của {ticker} qua từng năm",
    )
    filtered_set = ReasoningNode(
        step_id="filtered_set",
        operation="filter",
        inputs=(StepOutputInput("extract_a", ValueKind.NUMERIC_SERIES_PERIOD),),
        output_kind=ValueKind.PERIOD_SET,
        domain=period_set_domain,
        params={"operator": filter_operator, "threshold_source": "convention", "threshold_value": filter_threshold_value},
        output="survivor_periods",
        description=f"lọc các năm có {filter_role.metric_key} {filter_operator} {filter_threshold_value}",
    )
    extract_b = ReasoningNode(
        step_id="extract_b",
        operation="extract",
        inputs=(select_role,),
        output_kind=ValueKind.NUMERIC_SERIES_PERIOD,
        domain=panel_domain,
        params={},
        output="metric_b",
        description=f"trích {select_role.metric_key} của {ticker} qua từng năm",
    )
    direction_desc = "cao nhất" if select_direction == "argmax" else "thấp nhất"
    selected_period = ReasoningNode(
        step_id="selected_period",
        operation=select_direction,
        inputs=(
            StepOutputInput("extract_b", ValueKind.NUMERIC_SERIES_PERIOD),
            StepOutputInput("filtered_set", ValueKind.PERIOD_SET),
        ),
        output_kind=ValueKind.PERIOD_KEY,
        domain=period_set_domain,
        params={},
        output="selected_period",
        description=f"chọn năm có {select_role.metric_key} {direction_desc} trong các năm thỏa điều kiện",
    )
    extract_c = ReasoningNode(
        step_id="extract_c",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_PERIOD,
        domain=panel_domain,
        params={},
        output="metric_c",
        description=f"trích {target_role.metric_key} của {ticker} qua từng năm",
    )
    result = ReasoningNode(
        step_id="result",
        operation="lookup",
        inputs=(
            StepOutputInput("extract_c", ValueKind.NUMERIC_SERIES_PERIOD),
            StepOutputInput("selected_period", ValueKind.PERIOD_KEY),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=panel_domain,
        params={},
        output="result",
        description=f"tra {target_role.metric_key} tại năm đã chọn",
    )

    return ReasoningGraph(
        recipe_id=R6_RECIPE_ID,
        domain=panel_domain,
        metric_roles=(filter_role, select_role, target_role),
        nodes=(extract_a, filtered_set, extract_b, selected_period, extract_c, result),
        terminal_step_id="result",
    )


def build_r7_graph(
    *,
    ticker: str,
    periods: tuple[str, ...],
    report_scope: str,
    filter_role: MetricRoleInput,
    filter_operator: str,
    threshold_statistic: str,
    percentile_value: float | None,
    select_role: MetricRoleInput,
    select_direction: str,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    _require(len(periods) >= 4, "R7 cần >= 4 kỳ (threshold cần population đủ lớn + còn >=2 kỳ sau filter)")
    _require(filter_operator in FILTER_OPERATORS, f"R7: filter_operator phải thuộc {FILTER_OPERATORS}")
    _require(
        threshold_statistic in DERIVED_THRESHOLD_STATISTICS,
        f"R7: threshold_statistic phải thuộc {DERIVED_THRESHOLD_STATISTICS}",
    )
    _require(select_direction in SELECT_DIRECTIONS, f"R7: select_direction phải thuộc {SELECT_DIRECTIONS}")
    keys = {filter_role.metric_key, select_role.metric_key, target_role.metric_key}
    _require(len(keys) == 3, "R7: metric A (filter)/B (select)/C (target) phải là 3 metric khác nhau")
    if threshold_statistic == "percentile":
        _require(percentile_value is not None and 0 < percentile_value < 100, "R7: percentile_value phải trong (0,100)")

    entities = (ticker,)
    panel_domain = Domain(entities=entities, periods=periods, report_scope=report_scope)
    period_set_domain = Domain(entities=entities, periods=(), report_scope=report_scope)

    extract_a = ReasoningNode(
        step_id="extract_a",
        operation="extract",
        inputs=(filter_role,),
        output_kind=ValueKind.NUMERIC_SERIES_PERIOD,
        domain=panel_domain,
        params={},
        output="metric_a",
        description=f"trích {filter_role.metric_key} của {ticker} qua từng năm",
    )
    threshold_params: dict[str, object] = {"percentile": percentile_value} if threshold_statistic == "percentile" else {}
    statistic_desc = {
        "median": "trung vị",
        "average_threshold": "trung bình",
        "percentile": f"phân vị {percentile_value}",
    }[threshold_statistic]
    threshold_node = ReasoningNode(
        step_id="threshold",
        operation=threshold_statistic,
        inputs=(StepOutputInput("extract_a", ValueKind.NUMERIC_SERIES_PERIOD),),
        output_kind=ValueKind.THRESHOLD,
        domain=period_set_domain,
        params=threshold_params,
        output="threshold_value",
        description=f"tính {statistic_desc} của {filter_role.metric_key} qua các năm",
    )
    filtered_set = ReasoningNode(
        step_id="filtered_set",
        operation="filter",
        inputs=(
            StepOutputInput("extract_a", ValueKind.NUMERIC_SERIES_PERIOD),
            StepOutputInput("threshold", ValueKind.THRESHOLD),
        ),
        output_kind=ValueKind.PERIOD_SET,
        domain=period_set_domain,
        params={"operator": filter_operator, "threshold_source": "derived"},
        output="survivor_periods",
        description=f"lọc các năm có {filter_role.metric_key} {filter_operator} {statistic_desc}",
    )
    extract_b = ReasoningNode(
        step_id="extract_b",
        operation="extract",
        inputs=(select_role,),
        output_kind=ValueKind.NUMERIC_SERIES_PERIOD,
        domain=panel_domain,
        params={},
        output="metric_b",
        description=f"trích {select_role.metric_key} của {ticker} qua từng năm",
    )
    direction_desc = "cao nhất" if select_direction == "argmax" else "thấp nhất"
    selected_period = ReasoningNode(
        step_id="selected_period",
        operation=select_direction,
        inputs=(
            StepOutputInput("extract_b", ValueKind.NUMERIC_SERIES_PERIOD),
            StepOutputInput("filtered_set", ValueKind.PERIOD_SET),
        ),
        output_kind=ValueKind.PERIOD_KEY,
        domain=period_set_domain,
        params={},
        output="selected_period",
        description=f"chọn năm có {select_role.metric_key} {direction_desc} trong các năm thỏa điều kiện",
    )
    extract_c = ReasoningNode(
        step_id="extract_c",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_PERIOD,
        domain=panel_domain,
        params={},
        output="metric_c",
        description=f"trích {target_role.metric_key} của {ticker} qua từng năm",
    )
    result = ReasoningNode(
        step_id="result",
        operation="lookup",
        inputs=(
            StepOutputInput("extract_c", ValueKind.NUMERIC_SERIES_PERIOD),
            StepOutputInput("selected_period", ValueKind.PERIOD_KEY),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=panel_domain,
        params={},
        output="result",
        description=f"tra {target_role.metric_key} tại năm đã chọn",
    )

    return ReasoningGraph(
        recipe_id=R7_RECIPE_ID,
        domain=panel_domain,
        metric_roles=(filter_role, select_role, target_role),
        nodes=(extract_a, threshold_node, filtered_set, extract_b, selected_period, extract_c, result),
        terminal_step_id="result",
    )


RECIPE_MEANINGS: dict[str, str] = {
    R1_RECIPE_ID: (
        "Trong một peer group và giai đoạn, tại công ty-năm có tốc độ tăng trưởng doanh thu thuần "
        "cao nhất, metric B là bao nhiêu?"
    ),
    R2_RECIPE_ID: (
        "Trong các công ty có metric A dương liên tục suốt giai đoạn, giá trị metric B lớn nhất tại "
        "năm tham chiếu là bao nhiêu?"
    ),
    R3_RECIPE_ID: (
        "Trong các công ty thỏa điều kiện metric A liên tục, metric C của công ty có metric B cao/"
        "thấp nhất tại năm tham chiếu là bao nhiêu?"
    ),
    R4_RECIPE_ID: (
        "Tại năm tham chiếu, tổng hợp metric B của các công ty có metric A vượt ngưỡng quy ước tài "
        "chính chuẩn là bao nhiêu?"
    ),
    R5_RECIPE_ID: (
        "Tổng hợp metric B của các công ty có metric A cao/thấp hơn trung vị hoặc trung bình của "
        "nhóm là bao nhiêu?"
    ),
    R6_RECIPE_ID: (
        "Trong các năm công ty có metric A vượt ngưỡng quy ước tài chính chuẩn, tại năm metric B "
        "thấp/cao nhất, metric C là bao nhiêu?"
    ),
    R7_RECIPE_ID: (
        "Trong các năm công ty có metric A cao/thấp hơn trung vị hoặc trung bình của chính công ty "
        "đó qua các năm, tại năm metric B thấp/cao nhất, metric C là bao nhiêu?"
    ),
}
