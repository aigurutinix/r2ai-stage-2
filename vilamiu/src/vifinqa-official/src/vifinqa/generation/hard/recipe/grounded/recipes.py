
from __future__ import annotations

from vifinqa.generation.hard.recipe.base import (
    Domain,
    MetricRoleInput,
    ReasoningGraph,
    ReasoningNode,
    StepOutputInput,
    ValueKind,
)

EQ_01_ID = "EQ_01"
LIQ_02_ID = "LIQ_02"
GRO_03_ID = "GRO_03"
PRO_04_ID = "PRO_04"
LEV_05_ID = "LEV_05"
WCA_10_ID = "WCA_10"

GROUNDED_RECIPE_IDS: tuple[str, ...] = (EQ_01_ID, LIQ_02_ID, GRO_03_ID, PRO_04_ID, LEV_05_ID, WCA_10_ID)

ENABLED_GROUNDED_RECIPE_IDS: tuple[str, ...] = (
    LIQ_02_ID,
    GRO_03_ID,
    PRO_04_ID,
    LEV_05_ID,
)

# Keep period handling explicit and deterministic.
EQ_01_TERMINAL_KEY = "eq01_operating_accruals_ratio"
GRO_03_TERMINAL_KEY = "gro03_delta_gross_margin"

EQ_01_PROFITABLE_THRESHOLD = 0.0  # NPAT_t > 0
WCA_10_DEFICIT_THRESHOLD = 1.0

GROUNDED_RECIPE_MEANINGS: dict[str, str] = {
    EQ_01_ID: (
        "Trong nhóm công ty cùng ngành có lợi nhuận sau thuế dương trong năm, tại công ty có lợi "
        "nhuận sau thuế cao nhất, phần chênh lệch giữa lợi nhuận sau thuế và dòng tiền từ hoạt động "
        "kinh doanh chiếm bao nhiêu phần trăm tài sản bình quân của công ty đó?"
    ),
    LIQ_02_ID: (
        "Trong nhóm công ty cùng ngành, tại công ty có hệ số thanh toán hiện hành cao nhất, hàng "
        "tồn kho tương đương bao nhiêu lần nợ ngắn hạn?"
    ),
    GRO_03_ID: (
        "Trong nhóm công ty cùng ngành, tại công ty có mức thay đổi doanh thu thuần lớn nhất so với "
        "năm liền trước, biên lợi nhuận gộp thay đổi bao nhiêu điểm phần trăm so với năm liền trước?"
    ),
    PRO_04_ID: (
        "Trong nhóm công ty cùng ngành, tại công ty có biên lợi nhuận gộp cao nhất, biên lợi nhuận "
        "gộp cao hơn biên lợi nhuận ròng bao nhiêu điểm phần trăm?"
    ),
    LEV_05_ID: (
        "Trong nhóm công ty cùng ngành có hệ số nợ phải trả trên vốn chủ sở hữu cao nhất, hệ số khả "
        "năng thanh toán lãi vay của công ty đó là bao nhiêu?"
    ),
    WCA_10_ID: (
        "Trong nhóm công ty cùng ngành có tài sản ngắn hạn thấp hơn nợ ngắn hạn, tại công ty có hệ "
        "số thanh toán hiện hành thấp nhất, hệ số dòng tiền hoạt động trên nợ ngắn hạn là bao nhiêu?"
    ),
}

# Keep LLM behavior within the declared contract.
GROUNDED_INTERPRETATION_LIMITS: dict[str, tuple[str, ...]] = {
    EQ_01_ID: (
        "Đây là proxy/signal chất lượng chuyển đổi lợi nhuận thành tiền, KHÔNG phải bằng chứng thao "
        "túng lợi nhuận hay gian lận kế toán.",
        "Không đặt tên là Sloan Accrual Measure/Beneish TATA/Jones Model.",
        "Diễn đạt target bằng công thức lời (chênh lệch giữa lợi nhuận sau thuế và dòng tiền từ hoạt "
        "động kinh doanh so với tài sản bình quân) — KHÔNG ném riêng thuật ngữ 'tỷ lệ dồn tích' như "
        "một cái tên tự nó đã rõ nghĩa.",
    ),
    LIQ_02_ID: (
        "Chỉ đo quy mô tồn kho so với nợ ngắn hạn theo convention quick ratio của dự án, KHÔNG đo "
        "tốc độ luân chuyển tồn kho hay xác suất mất khả năng thanh toán.",
        "Không dùng ngưỡng current ratio cố định (vd 1.5) trong câu hỏi.",
    ),
    GRO_03_ID: (
        "Không kết luận nhân quả kiểu 'hy sinh biên gộp để mua tăng trưởng' — giá, sản lượng, cơ cấu "
        "và giá vốn đều có thể làm biên gộp đổi.",
        "Nếu tăng trưởng doanh thu vẫn âm, diễn đạt là 'mức thay đổi doanh thu cao nhất', không tự "
        "gọi là tăng trưởng dương.",
    ),
    PRO_04_ID: (
        "Đây là tác động ròng cộng dồn của mọi khoản mục giữa lợi nhuận gộp và lợi nhuận sau thuế, "
        "KHÔNG được quy toàn bộ chênh lệch cho chi phí bán hàng/quản lý, lãi vay hay thuế riêng lẻ.",
        "Không gọi là 'Gross-to-Net Margin Spread' như một ratio chuẩn có tên riêng.",
    ),
    LEV_05_ID: (
        "Selector là Nợ phải trả/Vốn chủ sở hữu — KHÔNG gọi là Debt-to-Equity/D/E chuẩn CFA (dự án "
        "chưa có tổng nợ vay chịu lãi riêng).",
        "Không kết luận xác suất phá sản hay khả năng chống chịu cú sốc từ 1 con số coverage.",
    ),
    WCA_10_ID: (
        "Không nói CFO 'bù trực tiếp' cho vốn lưu động âm — dòng tiền cả năm và số dư cuối năm không "
        "cộng trừ trực tiếp với nhau.",
        "Không kết luận doanh nghiệp chắc chắn vỡ nợ; ratio chịu mùa vụ, cần so với peer/ngành.",
    ),
}


def build_eq01_graph(
    *,
    entities: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    npat_role: MetricRoleInput,
    accrual_role: MetricRoleInput,
) -> ReasoningGraph:
    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)
    entity_set_domain = Domain(entities=entities, periods=(), report_scope=report_scope)

    extract_npat = ReasoningNode(
        step_id="extract_npat",
        operation="extract",
        inputs=(npat_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="npat_series",
        description="trích lợi nhuận sau thuế của từng công ty trong năm",
    )
    profitable_set = ReasoningNode(
        step_id="profitable_set",
        operation="filter",
        inputs=(StepOutputInput("extract_npat", ValueKind.NUMERIC_SERIES_ENTITY),),
        output_kind=ValueKind.ENTITY_SET,
        domain=entity_set_domain,
        params={"operator": ">", "threshold_source": "convention", "threshold_value": EQ_01_PROFITABLE_THRESHOLD},
        output="profitable_survivors",
        description="lọc công ty có lợi nhuận sau thuế dương trong năm",
    )
    winner = ReasoningNode(
        step_id="winner",
        operation="argmax",
        inputs=(
            StepOutputInput("extract_npat", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("profitable_set", ValueKind.ENTITY_SET),
        ),
        output_kind=ValueKind.ENTITY_KEY,
        domain=reference_domain,
        params={},
        output="winner_key",
        description="chọn công ty có lợi nhuận sau thuế cao nhất trong nhóm có lãi",
    )
    extract_accrual = ReasoningNode(
        step_id="extract_accrual",
        operation="extract",
        inputs=(accrual_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="accrual_series",
        description="trích tỷ lệ dồn tích hoạt động trên tài sản bình quân của từng công ty",
    )
    result = ReasoningNode(
        step_id="result",
        operation="lookup",
        inputs=(
            StepOutputInput("extract_accrual", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("winner", ValueKind.ENTITY_KEY),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=reference_domain,
        params={},
        output="result",
        description="tra tỷ lệ dồn tích hoạt động trên tài sản bình quân tại công ty đã chọn",
    )

    return ReasoningGraph(
        recipe_id=EQ_01_ID,
        domain=reference_domain,
        metric_roles=(npat_role, accrual_role),
        nodes=(extract_npat, profitable_set, winner, extract_accrual, result),
        terminal_step_id="result",
    )


def build_liq02_graph(
    *,
    entities: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    current_ratio_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)

    extract_cr = ReasoningNode(
        step_id="extract_current_ratio",
        operation="extract",
        inputs=(current_ratio_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="current_ratio_series",
        description="trích hệ số thanh toán hiện hành của từng công ty trong năm",
    )
    winner = ReasoningNode(
        step_id="winner",
        operation="argmax",
        inputs=(StepOutputInput("extract_current_ratio", ValueKind.NUMERIC_SERIES_ENTITY),),
        output_kind=ValueKind.ENTITY_KEY,
        domain=reference_domain,
        params={},
        output="winner_key",
        description="chọn công ty có hệ số thanh toán hiện hành cao nhất trong nhóm",
    )
    extract_target = ReasoningNode(
        step_id="extract_target",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="target_series",
        description="trích hàng tồn kho trên nợ ngắn hạn của từng công ty trong năm",
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
        description="tra hàng tồn kho trên nợ ngắn hạn tại công ty đã chọn",
    )

    return ReasoningGraph(
        recipe_id=LIQ_02_ID,
        domain=reference_domain,
        metric_roles=(current_ratio_role, target_role),
        nodes=(extract_cr, winner, extract_target, result),
        terminal_step_id="result",
    )


def build_gro03_graph(
    *,
    entities: tuple[str, ...],
    periods: tuple[str, str],
    report_scope: str,
    revenue_role: MetricRoleInput,
    gross_margin_role: MetricRoleInput,
) -> ReasoningGraph:
    prior_period, period = periods
    full_domain = Domain(entities=entities, periods=periods, report_scope=report_scope)
    candidate_domain = Domain(entities=entities, periods=(period,), report_scope=report_scope)

    extract_revenue = ReasoningNode(
        step_id="extract_revenue",
        operation="extract",
        inputs=(revenue_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=full_domain,
        params={},
        output="revenue_series",
        description="trích doanh thu thuần theo từng công ty-năm",
    )
    growth = ReasoningNode(
        step_id="growth",
        operation="growth",
        inputs=(StepOutputInput("extract_revenue", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=candidate_domain,
        params={"period_universe": periods},
        output="growth_series",
        description="mức thay đổi doanh thu thuần so với năm liền trước, theo từng công ty",
    )
    winner = ReasoningNode(
        step_id="winner",
        operation="argmax",
        inputs=(StepOutputInput("growth", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
        output_kind=ValueKind.ENTITY_PERIOD_KEY,
        domain=candidate_domain,
        params={},
        output="winner_key",
        description="chọn công ty-năm có mức thay đổi doanh thu thuần lớn nhất trong nhóm",
    )
    extract_gpm = ReasoningNode(
        step_id="extract_gross_margin",
        operation="extract",
        inputs=(gross_margin_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=full_domain,
        params={},
        output="gpm_series",
        description="trích biên lợi nhuận gộp theo từng công ty-năm",
    )
    delta_gpm = ReasoningNode(
        step_id="delta_gross_margin",
        operation="period_difference",
        inputs=(StepOutputInput("extract_gross_margin", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        domain=candidate_domain,
        params={"period_universe": periods},
        output="delta_gpm_series",
        description="thay đổi biên lợi nhuận gộp so với năm liền trước, theo từng công ty",
    )
    result = ReasoningNode(
        step_id="result",
        operation="lookup",
        inputs=(
            StepOutputInput("delta_gross_margin", ValueKind.NUMERIC_SERIES_ENTITY_PERIOD),
            StepOutputInput("winner", ValueKind.ENTITY_PERIOD_KEY),
        ),
        output_kind=ValueKind.NUMERIC_SCALAR,
        domain=candidate_domain,
        params={},
        output="result",
        description="tra thay đổi biên lợi nhuận gộp tại công ty-năm đã chọn",
    )

    return ReasoningGraph(
        recipe_id=GRO_03_ID,
        domain=full_domain,
        metric_roles=(revenue_role, gross_margin_role),
        nodes=(extract_revenue, growth, winner, extract_gpm, delta_gpm, result),
        terminal_step_id="result",
    )


def build_pro04_graph(
    *,
    entities: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    gross_margin_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)

    extract_gm = ReasoningNode(
        step_id="extract_gross_margin",
        operation="extract",
        inputs=(gross_margin_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="gross_margin_series",
        description="trích biên lợi nhuận gộp của từng công ty trong năm",
    )
    winner = ReasoningNode(
        step_id="winner",
        operation="argmax",
        inputs=(StepOutputInput("extract_gross_margin", ValueKind.NUMERIC_SERIES_ENTITY),),
        output_kind=ValueKind.ENTITY_KEY,
        domain=reference_domain,
        params={},
        output="winner_key",
        description="chọn công ty có biên lợi nhuận gộp cao nhất trong nhóm",
    )
    extract_target = ReasoningNode(
        step_id="extract_target",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="target_series",
        description="trích chênh lệch giữa biên lợi nhuận gộp và biên lợi nhuận ròng của từng công ty",
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
        description="tra chênh lệch biên lợi nhuận gộp - biên lợi nhuận ròng tại công ty đã chọn",
    )

    return ReasoningGraph(
        recipe_id=PRO_04_ID,
        domain=reference_domain,
        metric_roles=(gross_margin_role, target_role),
        nodes=(extract_gm, winner, extract_target, result),
        terminal_step_id="result",
    )


def build_lev05_graph(
    *,
    entities: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    liabilities_to_equity_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)

    extract_lte = ReasoningNode(
        step_id="extract_liabilities_to_equity",
        operation="extract",
        inputs=(liabilities_to_equity_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="liabilities_to_equity_series",
        description="trích hệ số nợ phải trả trên vốn chủ sở hữu của từng công ty trong năm",
    )
    winner = ReasoningNode(
        step_id="winner",
        operation="argmax",
        inputs=(StepOutputInput("extract_liabilities_to_equity", ValueKind.NUMERIC_SERIES_ENTITY),),
        output_kind=ValueKind.ENTITY_KEY,
        domain=reference_domain,
        params={},
        output="winner_key",
        description="chọn công ty có hệ số nợ phải trả trên vốn chủ sở hữu cao nhất trong nhóm",
    )
    extract_target = ReasoningNode(
        step_id="extract_target",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="target_series",
        description="trích hệ số khả năng thanh toán lãi vay của từng công ty trong năm",
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
        description="tra hệ số khả năng thanh toán lãi vay tại công ty đã chọn",
    )

    return ReasoningGraph(
        recipe_id=LEV_05_ID,
        domain=reference_domain,
        metric_roles=(liabilities_to_equity_role, target_role),
        nodes=(extract_lte, winner, extract_target, result),
        terminal_step_id="result",
    )


def build_wca10_graph(
    *,
    entities: tuple[str, ...],
    reference_period: str,
    report_scope: str,
    current_ratio_role: MetricRoleInput,
    target_role: MetricRoleInput,
) -> ReasoningGraph:
    reference_domain = Domain(entities=entities, periods=(reference_period,), report_scope=report_scope)
    entity_set_domain = Domain(entities=entities, periods=(), report_scope=report_scope)

    extract_cr = ReasoningNode(
        step_id="extract_current_ratio",
        operation="extract",
        inputs=(current_ratio_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="current_ratio_series",
        description="trích hệ số thanh toán hiện hành của từng công ty trong năm",
    )
    deficit_set = ReasoningNode(
        step_id="deficit_set",
        operation="filter",
        inputs=(StepOutputInput("extract_current_ratio", ValueKind.NUMERIC_SERIES_ENTITY),),
        output_kind=ValueKind.ENTITY_SET,
        domain=entity_set_domain,
        params={"operator": "<", "threshold_source": "convention", "threshold_value": WCA_10_DEFICIT_THRESHOLD},
        output="deficit_survivors",
        description="lọc công ty có hệ số thanh toán hiện hành dưới 1 (tài sản ngắn hạn thấp hơn nợ ngắn hạn)",
    )
    winner = ReasoningNode(
        step_id="winner",
        operation="argmin",
        inputs=(
            StepOutputInput("extract_current_ratio", ValueKind.NUMERIC_SERIES_ENTITY),
            StepOutputInput("deficit_set", ValueKind.ENTITY_SET),
        ),
        output_kind=ValueKind.ENTITY_KEY,
        domain=reference_domain,
        params={},
        output="winner_key",
        description="chọn công ty có hệ số thanh toán hiện hành thấp nhất trong nhóm thiếu hụt vốn lưu động",
    )
    extract_target = ReasoningNode(
        step_id="extract_target",
        operation="extract",
        inputs=(target_role,),
        output_kind=ValueKind.NUMERIC_SERIES_ENTITY,
        domain=reference_domain,
        params={},
        output="target_series",
        description="trích hệ số dòng tiền hoạt động trên nợ ngắn hạn của từng công ty trong năm",
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
        description="tra hệ số dòng tiền hoạt động trên nợ ngắn hạn tại công ty đã chọn",
    )

    return ReasoningGraph(
        recipe_id=WCA_10_ID,
        domain=reference_domain,
        metric_roles=(current_ratio_role, target_role),
        nodes=(extract_cr, deficit_set, winner, extract_target, result),
        terminal_step_id="result",
    )
