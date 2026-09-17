
from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.hard.recipe import operations  # noqa: F401
from vifinqa.generation.hard.recipe.audit.base import DependencyAuditor
from vifinqa.generation.hard.recipe.audit.cache import AuditCache
from vifinqa.generation.hard.recipe.audit.service import (
    AuditedValues,
    DependencyAuditRejected,
    audit_dependency_report,
    rebuild_validated_graph,
)
from vifinqa.generation.hard.recipe.base import (
    CandidateRejected,
    MetricRoleInput,
    MetricTerms,
    ReasoningGraph,
    ValueKind,
)
from vifinqa.generation.hard.recipe.bindings import (
    resolve_operating_accruals_terms,
    resolve_window_metric_terms,
)
from vifinqa.generation.hard.recipe.compiler import transformed_terminal_value_kind
from vifinqa.generation.hard.recipe.evaluator import EvaluationError, evaluate_graph
from vifinqa.generation.hard.recipe.finalize import (
    company_names,
    finalize_validated_candidate,
    unit_label,
)
from vifinqa.generation.hard.recipe.grounded.analytical_recipes import (
    build_derived_threshold_average,
    build_derived_threshold_share,
    build_dual_transform_panel_rank_lookup,
    build_dual_panel_predicate_count,
    build_dual_predicate_count,
    build_entity_filter_aggregate,
    build_extreme_lookup_difference,
    build_entity_filter_rank_lookup,
    build_transformed_filter_aggregate,
    build_filtered_dual_transform_lookup,
    build_filtered_ranked_cohort_gap,
    build_filtered_transform_rank_lookup,
    build_filter_share,
    build_dual_transform_count,
    build_dol_terminal,
    build_lower_median_positive_share,
    build_median_split_ratio,
    build_multi_predicate_terminal,
    build_multi_transform_max,
    build_nested_median_aggregate,
    build_panel_rank_lookup,
    build_panel_filter_average,
    build_period_filter_aggregate,
    build_period_filter_rank_lookup,
    build_rank_lookup,
    build_ranked_cohort_terminal,
    build_sign_cohort_gap,
    build_temporal_filter_aggregate,
    build_temporal_filter_rank_lookup,
    build_temporal_growth_average,
    build_temporal_growth_lookup,
    build_top_n_dual_predicate_count,
    build_transform_rank_share,
    build_transformed_filter_aggregate_terminal,
    build_transition_filter_rank_lookup,
)
from vifinqa.generation.hard.recipe.grounded.domains import (
    adjacent_period_domains,
    cross_entity_period_window_domains,
    industry_wide_same_period_domains,
    industry_wide_adjacent_period_domains,
    industry_wide_cross_entity_period_window_domains,
    nonfinancial_adjacent_period_domains,
    nonfinancial_cross_entity_period_window_domains,
    nonfinancial_same_period_domains,
    same_period_domains,
    single_entity_period_window_domains,
)
from vifinqa.generation.hard.recipe.grounded.recipes import EQ_01_TERMINAL_KEY
from vifinqa.generation.hard.recipe.planner import (
    PublicCohort,
    PublicPredicate,
    PublicSpec,
    PublicTerminalBranch,
    RecipeCandidate,
    table_ref_to_path_map,
)
from vifinqa.generation.hard.template_intents import CapabilityState, INTENTS_BY_ID
from vifinqa.generation.panel.base import Cube
from vifinqa.generation.panel.catalog import get_ratio

logger = logging.getLogger(__name__)

REPORT_SCOPE = "consolidated"
DomainKind = Literal[
    "same_period",
    "adjacent_period",
    "nonfinancial_same_period",
    "nonfinancial_adjacent_period",
    "industry_wide_same_period",
    "industry_wide_adjacent_period",
    "industry_wide_cross_entity_period_window",
    "cross_entity_period_window",
    "single_entity_period_window",
    "nonfinancial_cross_entity_period_window",
]
Topology = Literal[
    "sign_cohort_gap",
    "dual_panel_count",
    "temporal_growth_average",
    "temporal_growth_lookup",
    "panel_filter_average",
    "entity_filter_aggregate",
    "dual_predicate_count",
    "derived_threshold_share",
    "rank_lookup",
    "derived_threshold_average",
    "panel_rank_lookup",
    "temporal_filter_aggregate",
    "temporal_filter_rank_lookup",
    "period_filter_aggregate",
    "period_filter_rank_lookup",
    "dual_transform_panel_rank_lookup",
    "extreme_lookup_difference",
    "entity_filter_rank_lookup",
    "transformed_filter_aggregate",
    "median_split_ratio",
    "ranked_cohort_gap",
    "ranked_cohort_share",
    "ranked_cohort_average",
    "lower_median_positive_share",
    "dual_transform_count",
    "filtered_dual_transform_lookup",
    "multi_transform_max",
    "filtered_transform_rank_lookup",
    "filter_share",
    "transform_rank_share",
    "top_n_dual_predicate_count",
    "nested_median_aggregate",
    "multi_predicate_terminal",
    "dol_terminal",
    "transformed_filter_aggregate_terminal",
    "filtered_ranked_cohort_gap",
    "transition_filter_rank_lookup",
]

CASH_ROE_GAP = "AN_CASH_ROE_GAP"
GROWTH_MARGIN_SQUEEZE_COUNT = "AN_GROWTH_MARGIN_SQUEEZE_COUNT"
PERSISTENT_CASH_GROWTH_AVG = "AN_PERSISTENT_CASH_GROWTH_AVG"
PERSISTENT_CASH_GROWTH_ROA = "AN_PERSISTENT_CASH_GROWTH_ROA"
GROWTH_MARGIN_CHANGE_AVG = "AN_GROWTH_MARGIN_CHANGE_AVG"
WORKING_CAPITAL_CASH_COVERAGE_AVG = "AN_WORKING_CAPITAL_CASH_COVERAGE_AVG"
PROFITABLE_NEGATIVE_CASH_COUNT = "AN_PROFITABLE_NEGATIVE_CASH_COUNT"
PROFITABLE_ACCRUAL_AVG = "AN_PROFITABLE_ACCRUAL_AVG"
LIQUIDITY_INVENTORY_LOAD_AVG = "AN_LIQUIDITY_INVENTORY_LOAD_AVG"
LIQUIDITY_QUICK_RATIO_AVG = "AN_LIQUIDITY_QUICK_RATIO_AVG"
PROFITABLE_GROSS_NET_GAP_AVG = "AN_PROFITABLE_GROSS_NET_GAP_AVG"
LEVERAGE_INTEREST_SHARE = "AN_LEVERAGE_INTEREST_SHARE"
LEVERAGE_COVERAGE_LOOKUP = "AN_LEVERAGE_COVERAGE_LOOKUP"
HIGH_LEVERAGE_COVERAGE_AVG = "AN_HIGH_LEVERAGE_COVERAGE_AVG"
REVENUE_GROWTH_GROSS_MARGIN_PANEL_LOOKUP = "AN_REVENUE_GROWTH_GROSS_MARGIN_PANEL_LOOKUP"
PROFITABLE_NET_MARGIN_REVENUE_SUM = "AN_PROFITABLE_NET_MARGIN_REVENUE_SUM"
PERSISTENT_CASH_NET_MARGIN_MAX = "AN_PERSISTENT_CASH_NET_MARGIN_MAX"
PERSISTENT_PROFIT_LOW_LEVERAGE_QUICK_RATIO = (
    "AN_PERSISTENT_PROFIT_LOW_LEVERAGE_QUICK_RATIO"
)
PROFITABLE_YEARS_REVENUE_MIN = "AN_PROFITABLE_YEARS_REVENUE_MIN"
PROFITABLE_LOW_REVENUE_CASH_FLOW_RATIO = "AN_PROFITABLE_LOW_REVENUE_CASH_FLOW_RATIO"
CASH_POSITIVE_REVENUE_SUM = "AN_CASH_POSITIVE_REVENUE_SUM"
PERSISTENT_PROFIT_REVENUE_SUM = "AN_PERSISTENT_PROFIT_REVENUE_SUM"
REVENUE_GROWTH_ASSET_TURNOVER_PANEL_LOOKUP = (
    "AN_REVENUE_GROWTH_ASSET_TURNOVER_PANEL_LOOKUP"
)
INVENTORY_DAYS_GROSS_MARGIN_CHANGE_LEADER = (
    "AN_INVENTORY_DAYS_GROSS_MARGIN_CHANGE_LEADER"
)
REVENUE_GROWTH_SGA_GROWTH_LEADER = "AN_REVENUE_GROWTH_SGA_GROWTH_LEADER"
CFO_OPERATING_PROFIT_NET_MARGIN_LOOKUP = "AN_CFO_OPERATING_PROFIT_NET_MARGIN_LOOKUP"
TEMPLATE_A05 = "TPL_A05_CFO_NET_MARGIN_GAP_LEVERAGE"
TEMPLATE_C05 = "TPL_C05_CFO_NET_MARGIN_GAP_LEVERAGE"
TEMPLATE_A13 = "TPL_A13_OCF_RATIO_QUICK_RATIO"
TEMPLATE_C11 = "TPL_C11_OCF_RATIO_QUICK_RATIO"
TEMPLATE_C04 = "TPL_C04_ACCRUALS_OCF_RATIO"
TEMPLATE_B02 = "TPL_B02_INVENTORY_DAYS_CFO_MARGIN"
TEMPLATE_A16 = "TPL_A16_EQUITY_MULTIPLIER_COVERAGE"
TEMPLATE_A24 = "TPL_A24_LONG_TERM_ASSETS_TURNOVER"
TEMPLATE_D01 = "TPL_D01_ASSET_SCALE_MAX_ROA"
TEMPLATE_D05 = "TPL_D05_PROFIT_CFO_NPAT_QUICK_RATIO"
TEMPLATE_D09 = "TPL_D09_LOW_QUICK_MAX_OCF_RATIO"
TEMPLATE_D15 = "TPL_D15_REVENUE_QUANTILE_CASH_SHARE"
TEMPLATE_D16 = "TPL_D16_LEVERAGE_QUANTILE_INTEREST_SHARE"
TEMPLATE_D19 = "TPL_D19_LONG_TERM_ASSETS_TURNOVER_AVG"
TEMPLATE_A02 = "TPL_A02_INVENTORY_SHARE_MARGIN_COUNT"
TEMPLATE_C02 = "TPL_C02_INVENTORY_SHARE_MARGIN_COUNT"
TEMPLATE_A10 = "TPL_A10_SGA_OUTPACES_REVENUE_COUNT"
TEMPLATE_C08 = "TPL_C08_SGA_OUTPACES_REVENUE_COUNT"
TEMPLATE_A11 = "TPL_A11_REVENUE_DECLINE_COGS_MARGIN"
TEMPLATE_C09 = "TPL_C09_REVENUE_DECLINE_COGS_MARGIN"
TEMPLATE_A12 = "TPL_A12_SGA_INTENSITY_ROA_GAP"
TEMPLATE_A17 = "TPL_A17_LOW_LEVERAGE_NPAT_SHARE"
TEMPLATE_C13 = "TPL_C13_LOW_LEVERAGE_NPAT_SHARE"
TEMPLATE_A21 = "TPL_A21_ASSET_TURNOVER_ROE_GAP"
TEMPLATE_C16 = "TPL_C16_ASSET_TURNOVER_ROE_GAP"
TEMPLATE_B05 = "TPL_B05_LEVERAGE_INTEREST_RATIO"
TEMPLATE_C17 = "TPL_C17_LEVERAGE_INTEREST_RATIO"
TEMPLATE_B06 = "TPL_B06_CFO_MARGIN_ROA_QUANTILE_GAP"
TEMPLATE_C18 = "TPL_C18_CFO_MARGIN_ROA_QUANTILE_GAP"
TEMPLATE_C19 = "TPL_C19_GROSS_MARGIN_CASH_SHARE"
TEMPLATE_C20 = "TPL_C20_LEVERAGE_COVERAGE_QUANTILE_AVG"
TEMPLATE_D20 = "TPL_D20_LEVERAGE_COVERAGE_QUANTILE_AVG"
TEMPLATE_A23 = "TPL_A23_MARGIN_TURNOVER_ROE_MAX"
TEMPLATE_C15 = "TPL_C15_MARGIN_TURNOVER_ROE_TRIPLE_MAX"
TEMPLATE_A15 = "TPL_A15_NEGATIVE_NWC_LOW_LEVERAGE_ROA"
TEMPLATE_A07 = "TPL_A07_PERSISTENT_CFO_NPAT_ASSET_GROWTH"
TEMPLATE_A03 = "TPL_A03_INVENTORY_DECLINE_MAX_CFO_MARGIN"

AGGREGATE_OPERATIONS = frozenset({"sum", "average", "minimum", "maximum"})
RAW_MONEY_FLOW_METRICS = frozenset(
    {
        "kqkd:01",
        "kqkd:02",
        "kqkd:10",
        "kqkd:11",
        "kqkd:20",
        "kqkd:21",
        "kqkd:22",
        "kqkd:23",
        "kqkd:25",
        "kqkd:26",
        "kqkd:30",
        "kqkd:40",
        "kqkd:50",
        "kqkd:60",
        "lctt:20",
        "lctt:70",
    }
)


@dataclass(frozen=True, slots=True)
class AnalyticalFrame:
    frame_id: str
    story_family: str
    domain_kind: DomainKind
    topology: Topology
    metric_keys: tuple[str, ...]
    metric_labels: tuple[str, ...]
    metric_roles: tuple[tuple[Literal["filter", "rank", "target"], ...], ...]
    terminal_key: str
    meaning: str
    interpretation_limits: tuple[str, ...]
    template_id: str | None = None
    finance_rationale: str | None = None
    threshold_operator: str | None = None
    threshold_value: float | None = None
    selector_direction: str | None = None
    selector_transform: str | None = None
    target_transform: str | None = None
    transform_source_offset: int = 0
    required_period_count: int | None = None
    require_positive_selector_value: bool = False
    aggregate_operation: str | None = None
    target_unit_label: str | None = None
    public_metric_keys: tuple[str, ...] | None = None
    required_question_terms: tuple[str, ...] = ()
    require_explicit_period_span: bool = False
    predicate_operators: tuple[str, ...] = ()
    predicate_thresholds: tuple[float, ...] = ()
    predicate_transforms: tuple[str, ...] = ()
    topology_mode: str | None = None
    filter_transform: str | None = None
    filter_period_offset: int = -1
    rank_period_offset: int = -1
    target_period_offset: int = -1
    top_n: int | None = None
    quantile_percent: float | None = None
    secondary_threshold_value: float | None = None
    absolute_terminal: bool = False


FRAMES: tuple[AnalyticalFrame, ...] = (
    AnalyticalFrame(
        TEMPLATE_A03,
        "inventory_release_cash_margin",
        "industry_wide_adjacent_period",
        "transformed_filter_aggregate",
        ("cdkt:140", "cfo_margin"),
        ("Hàng tồn kho", "CFO margin"),
        (("filter",), ("target",)),
        "cfo_margin",
        "Trong cohort tồn kho giảm ít nhất 10%, hỏi CFO margin cao nhất.",
        ("Ngưỡng được khóa trước terminal values.",),
        template_id="A03",
        threshold_operator="<=",
        threshold_value=-0.10,
        aggregate_operation="maximum",
    ),
    AnalyticalFrame(
        TEMPLATE_A07,
        "persistent_cash_conversion_asset_growth",
        "industry_wide_adjacent_period",
        "temporal_growth_average",
        ("cfo_to_npat", "cdkt:200"),
        ("Tỷ lệ CFO trên lợi nhuận sau thuế", "Tăng trưởng tài sản dài hạn"),
        (("filter",), ("target",)),
        "persistent_cash_revenue_growth",
        "Trong cohort CFO/NPAT trên 1 ở cả hai kỳ, hỏi tăng trưởng tài sản dài hạn bình quân.",
        ("Ngưỡng 1 không phải chuẩn chất lượng phổ quát.",),
        template_id="A07",
        threshold_operator=">",
        threshold_value=1.0,
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        TEMPLATE_A15,
        "negative_working_capital_roa",
        "adjacent_period",
        "entity_filter_rank_lookup",
        ("net_working_capital", "liabilities_to_assets", "roa"),
        (
            "Vốn lưu động ròng",
            "Hệ số nợ phải trả trên tổng tài sản",
            "ROA theo tổng tài sản bình quân",
        ),
        (("filter",), ("rank",), ("target",)),
        "roa",
        "Trong cohort vốn lưu động ròng âm, chọn doanh nghiệp có liabilities-to-assets thấp nhất rồi hỏi ROA.",
        ("Không gọi liabilities-to-assets là debt-to-assets CFA.",),
        template_id="A15",
        threshold_operator="<",
        threshold_value=0.0,
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        TEMPLATE_A02,
        "inventory_margin_screen",
        "adjacent_period",
        "dual_transform_count",
        ("inventory_to_assets", "gross_margin"),
        ("Tỷ trọng hàng tồn kho trên tổng tài sản", "Biên lợi nhuận gộp"),
        (("filter",), ("filter",)),
        "growth_margin_squeeze_count",
        "Đếm doanh nghiệp đồng thời tăng tỷ trọng tồn kho và giảm biên lợi nhuận gộp.",
        ("Không giả định survivor duy nhất.",),
        template_id="A02",
        selector_transform="period_difference",
        target_transform="period_difference",
        topology_mode="positive_negative",
        target_unit_label="công ty",
    ),
    AnalyticalFrame(
        TEMPLATE_C02,
        "inventory_margin_screen",
        "adjacent_period",
        "dual_transform_count",
        ("inventory_to_assets", "gross_margin"),
        ("Tỷ trọng hàng tồn kho trên tổng tài sản", "Biên lợi nhuận gộp"),
        (("filter",), ("filter",)),
        "growth_margin_squeeze_count",
        "Đếm doanh nghiệp đồng thời tăng tỷ trọng tồn kho và giảm biên lợi nhuận gộp.",
        ("Terminal là count, không chọn winner.",),
        template_id="C02",
        selector_transform="period_difference",
        target_transform="period_difference",
        topology_mode="positive_negative",
        target_unit_label="công ty",
    ),
    AnalyticalFrame(
        TEMPLATE_A10,
        "sga_revenue_growth_screen",
        "adjacent_period",
        "dual_transform_count",
        ("sga_expense", "kqkd:10"),
        ("Tăng trưởng SG&A", "Tăng trưởng doanh thu thuần"),
        (("filter",), ("filter",)),
        "growth_margin_squeeze_count",
        "Đếm doanh nghiệp có SG&A tăng nhanh hơn doanh thu thuần.",
        ("Chuẩn hóa dấu chi phí trước khi tính tăng trưởng.",),
        template_id="A10",
        selector_transform="growth",
        target_transform="growth",
        topology_mode="first_above_second",
        target_unit_label="công ty",
    ),
    AnalyticalFrame(
        TEMPLATE_C08,
        "sga_revenue_growth_screen",
        "adjacent_period",
        "dual_transform_count",
        ("sga_expense", "kqkd:10"),
        ("Tăng trưởng SG&A", "Tăng trưởng doanh thu thuần"),
        (("filter",), ("filter",)),
        "growth_margin_squeeze_count",
        "Đếm doanh nghiệp có SG&A tăng nhanh hơn doanh thu thuần.",
        ("Chỉ so sánh hai tốc độ tăng trưởng.",),
        template_id="C08",
        selector_transform="growth",
        target_transform="growth",
        topology_mode="first_above_second",
        target_unit_label="công ty",
    ),
    AnalyticalFrame(
        TEMPLATE_A11,
        "revenue_contraction_cogs_margin",
        "industry_wide_adjacent_period",
        "filtered_dual_transform_lookup",
        ("kqkd:10", "kqkd:11", "gross_margin"),
        (
            "Tăng trưởng doanh thu thuần",
            "Tăng trưởng giá vốn hàng bán",
            "Thay đổi biên lợi nhuận gộp",
        ),
        (("filter",), ("rank",), ("target",)),
        "gro03_delta_gross_margin",
        "Trong nhóm giảm doanh thu, chọn doanh nghiệp giảm giá vốn mạnh nhất rồi hỏi thay đổi biên gộp.",
        ("Không khẳng định giá vốn là nguyên nhân duy nhất.",),
        template_id="A11",
        selector_direction="argmin",
        selector_transform="growth",
        target_transform="period_difference",
    ),
    AnalyticalFrame(
        TEMPLATE_C09,
        "revenue_contraction_cogs_margin",
        "industry_wide_adjacent_period",
        "filtered_dual_transform_lookup",
        ("kqkd:10", "kqkd:11", "gross_margin"),
        (
            "Tăng trưởng doanh thu thuần",
            "Tăng trưởng giá vốn hàng bán",
            "Thay đổi biên lợi nhuận gộp",
        ),
        (("filter",), ("rank",), ("target",)),
        "gro03_delta_gross_margin",
        "Trong nhóm giảm doanh thu, chọn doanh nghiệp giảm giá vốn mạnh nhất rồi hỏi thay đổi biên gộp.",
        ("Không dùng ngôn ngữ nhân quả.",),
        template_id="C09",
        selector_direction="argmin",
        selector_transform="growth",
        target_transform="period_difference",
    ),
    AnalyticalFrame(
        TEMPLATE_A12,
        "sga_intensity_roa_extremes",
        "adjacent_period",
        "extreme_lookup_difference",
        ("sga_intensity", "roa"),
        ("Cường độ SG&A trên doanh thu", "ROA theo tổng tài sản bình quân"),
        (("rank",), ("target",)),
        "cash_roe_cohort_gap",
        "Lấy ROA của cực SG&A intensity cao trừ cực thấp.",
        ("Tie bị reject trước terminal.",),
        template_id="A12",
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        TEMPLATE_A17,
        "low_leverage_profit_share",
        "industry_wide_same_period",
        "lower_median_positive_share",
        ("liabilities_to_equity", "kqkd:60"),
        ("Hệ số nợ phải trả trên vốn chủ sở hữu", "Lợi nhuận sau thuế dương"),
        (("filter",), ("target",)),
        "leverage_interest_expense_share",
        "Cohort dưới trung vị đòn bẩy đóng góp bao nhiêu vào tổng LNST dương.",
        ("Chỉ cộng LNST dương.",),
        template_id="A17",
    ),
    AnalyticalFrame(
        TEMPLATE_C13,
        "low_leverage_profit_share",
        "industry_wide_same_period",
        "lower_median_positive_share",
        ("liabilities_to_equity", "kqkd:60"),
        ("Hệ số nợ phải trả trên vốn chủ sở hữu", "Lợi nhuận sau thuế dương"),
        (("filter",), ("target",)),
        "leverage_interest_expense_share",
        "Cohort dưới trung vị đòn bẩy chiếm tỷ trọng tổng LNST dương.",
        ("Equality policy: bằng trung vị không thuộc cohort dưới.",),
        template_id="C13",
    ),
    AnalyticalFrame(
        TEMPLATE_A21,
        "asset_turnover_roe_extremes",
        "adjacent_period",
        "extreme_lookup_difference",
        ("asset_turnover_avg", "roe"),
        (
            "Vòng quay tổng tài sản theo tài sản bình quân",
            "ROE theo vốn chủ sở hữu bình quân",
        ),
        (("rank",), ("target",)),
        "cash_roe_cohort_gap",
        "Lấy ROE của doanh nghiệp vòng quay thấp nhất trừ doanh nghiệp cao nhất.",
        ("Không tự quy chênh lệch cho đòn bẩy.",),
        template_id="A21",
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        TEMPLATE_C16,
        "asset_turnover_roe_extremes",
        "adjacent_period",
        "extreme_lookup_difference",
        ("asset_turnover_avg", "roe"),
        (
            "Vòng quay tổng tài sản theo tài sản bình quân",
            "ROE theo vốn chủ sở hữu bình quân",
        ),
        (("rank",), ("target",)),
        "cash_roe_cohort_gap",
        "So chênh lệch ROE giữa hai cực vòng quay tài sản.",
        ("Tie policy reject.",),
        template_id="C16",
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        TEMPLATE_B05,
        "leverage_interest_cohort_ratio",
        "industry_wide_same_period",
        "median_split_ratio",
        ("liabilities_to_equity", "kqkd:23"),
        ("Hệ số nợ phải trả trên vốn chủ sở hữu", "Chi phí lãi vay"),
        (("filter",), ("target",)),
        "interest_coverage",
        "Tổng chi phí lãi vay cohort trên trung vị gấp bao nhiêu lần cohort còn lại.",
        ("Không gọi liabilities-to-equity là D/E nợ vay.",),
        template_id="B05",
    ),
    AnalyticalFrame(
        TEMPLATE_C17,
        "leverage_interest_cohort_ratio",
        "industry_wide_same_period",
        "median_split_ratio",
        ("liabilities_to_equity", "kqkd:23"),
        ("Hệ số nợ phải trả trên vốn chủ sở hữu", "Chi phí lãi vay"),
        (("filter",), ("target",)),
        "interest_coverage",
        "Tỷ số tổng chi phí lãi vay giữa hai cohort quanh trung vị đòn bẩy.",
        ("Hai tổng phải dương.",),
        template_id="C17",
    ),
    AnalyticalFrame(
        TEMPLATE_B06,
        "cfo_margin_roa_quantiles",
        "industry_wide_adjacent_period",
        "ranked_cohort_gap",
        ("cfo_margin", "roa"),
        ("CFO margin", "ROA theo tổng tài sản bình quân"),
        (("filter",), ("target",)),
        "cash_roe_cohort_gap",
        "ROA bình quân nhóm 25% CFO margin cao nhất trừ nhóm 25% thấp nhất.",
        ("Mỗi cohort có ít nhất ba doanh nghiệp; reject tie biên.",),
        template_id="B06",
        threshold_value=25.0,
    ),
    AnalyticalFrame(
        TEMPLATE_C18,
        "cfo_margin_roa_quantiles",
        "industry_wide_adjacent_period",
        "ranked_cohort_gap",
        ("cfo_margin", "roa"),
        ("CFO margin", "ROA theo tổng tài sản bình quân"),
        (("filter",), ("target",)),
        "cash_roe_cohort_gap",
        "Chênh lệch ROA bình quân giữa hai phân vị CFO margin.",
        ("Arithmetic mean of company ratios.",),
        template_id="C18",
        threshold_value=25.0,
    ),
    AnalyticalFrame(
        TEMPLATE_C19,
        "gross_margin_cash_quantile_share",
        "industry_wide_same_period",
        "ranked_cohort_share",
        ("gross_margin", "cdkt:110"),
        ("Biên lợi nhuận gộp", "Tiền và tương đương tiền"),
        (("filter",), ("target",)),
        "leverage_interest_expense_share",
        "Nhóm 25% biên gộp cao nhất nắm giữ bao nhiêu phần trăm tổng tiền ngành.",
        ("Tiền không âm; reject tie biên.",),
        template_id="C19",
        threshold_value=25.0,
    ),
    AnalyticalFrame(
        TEMPLATE_C20,
        "leverage_coverage_quantile",
        "industry_wide_same_period",
        "ranked_cohort_average",
        ("liabilities_to_assets", "interest_coverage"),
        ("Hệ số nợ phải trả trên tổng tài sản", "Hệ số khả năng thanh toán lãi vay"),
        (("filter",), ("target",)),
        "interest_coverage",
        "Bình quân interest coverage nhóm 25% liabilities-to-assets cao nhất.",
        ("Loại hệ số khả năng thanh toán lãi vay không hữu hạn.",),
        template_id="C20",
        threshold_value=25.0,
    ),
    AnalyticalFrame(
        TEMPLATE_D20,
        "leverage_coverage_quantile",
        "nonfinancial_same_period",
        "ranked_cohort_average",
        ("liabilities_to_assets", "interest_coverage"),
        ("Hệ số nợ phải trả trên tổng tài sản", "Hệ số khả năng thanh toán lãi vay"),
        (("filter",), ("target",)),
        "interest_coverage",
        "Bình quân interest coverage nhóm 25% liabilities-to-assets cao nhất trong corpus phi tài chính.",
        ("Không gọi selector là debt-to-assets CFA.",),
        template_id="D20",
        threshold_value=25.0,
    ),
    AnalyticalFrame(
        TEMPLATE_D15,
        "large_revenue_cash_concentration",
        "nonfinancial_same_period",
        "ranked_cohort_share",
        ("kqkd:10", "cdkt:110"),
        ("Doanh thu thuần", "Tiền và tương đương tiền"),
        (("rank",), ("target",)),
        "leverage_interest_expense_share",
        "Nhóm 25% doanh nghiệp phi tài chính có doanh thu cao nhất nắm giữ bao nhiêu phần trăm tổng tiền của universe.",
        ("Universe là corpus đủ coverage, không phải toàn thị trường.",),
        template_id="D15",
        threshold_value=25.0,
    ),
    AnalyticalFrame(
        TEMPLATE_D16,
        "high_liabilities_interest_burden",
        "nonfinancial_same_period",
        "ranked_cohort_share",
        ("liabilities_to_assets", "kqkd:23"),
        (
            "Hệ số nợ phải trả trên tổng tài sản",
            "Chi phí lãi vay",
        ),
        (("rank",), ("target",)),
        "leverage_interest_expense_share",
        "Nhóm 25% liabilities-to-assets cao nhất gánh bao nhiêu phần trăm tổng chi phí lãi vay của universe.",
        ("Không gọi liabilities-to-assets là debt-to-assets theo nghĩa CFA.",),
        template_id="D16",
        threshold_value=25.0,
    ),
    AnalyticalFrame(
        TEMPLATE_D19,
        "long_term_asset_intensity_turnover_cohort",
        "nonfinancial_adjacent_period",
        "ranked_cohort_average",
        ("long_term_assets_share", "asset_turnover_avg"),
        (
            "Tỷ trọng tài sản dài hạn trên tổng tài sản",
            "Vòng quay tổng tài sản theo tổng tài sản bình quân",
        ),
        (("rank",), ("target",)),
        "asset_turnover_avg",
        "Bình quân số học vòng quay tổng tài sản của nhóm 25% có tỷ trọng tài sản dài hạn cao nhất.",
        ("Terminal là arithmetic mean của tỷ số từng doanh nghiệp.",),
        template_id="D19",
        threshold_value=25.0,
    ),
    AnalyticalFrame(
        TEMPLATE_D05,
        "profitable_cash_conversion_liquidity",
        "nonfinancial_same_period",
        "entity_filter_rank_lookup",
        ("kqkd:60", "cfo_to_npat", "quick_ratio"),
        (
            "Lợi nhuận sau thuế",
            "Tỷ lệ CFO trên lợi nhuận sau thuế",
            "Hệ số thanh toán nhanh",
        ),
        (("filter",), ("rank",), ("target",)),
        "quick_ratio",
        "Trong universe LNST trên 100 tỷ đồng, chọn CFO/NPAT cao nhất rồi hỏi quick ratio.",
        ("CFO/NPAT cao không tự động chứng minh chất lượng lợi nhuận cao.",),
        template_id="D05",
        threshold_operator=">",
        threshold_value=100_000_000_000.0,
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        TEMPLATE_A23,
        "margin_turnover_roe_screen",
        "industry_wide_cross_entity_period_window",
        "multi_transform_max",
        ("gross_margin", "asset_turnover_avg", "roe"),
        (
            "Thay đổi biên lợi nhuận gộp",
            "Thay đổi vòng quay tổng tài sản",
            "ROE năm cuối",
        ),
        (("filter",), ("filter",), ("target",)),
        "roe",
        "Trong nhóm biên gộp giảm và vòng quay tăng, hỏi ROE cao nhất.",
        ("Không khẳng định đánh đổi nhân quả.",),
        template_id="A23",
        required_period_count=3,
        predicate_operators=("<", ">"),
    ),
    AnalyticalFrame(
        TEMPLATE_C15,
        "margin_turnover_roe_screen",
        "industry_wide_cross_entity_period_window",
        "multi_transform_max",
        ("net_margin", "asset_turnover_avg", "roe", "roe"),
        (
            "Thay đổi biên lợi nhuận ròng",
            "Thay đổi vòng quay tổng tài sản",
            "Thay đổi ROE",
            "ROE năm cuối",
        ),
        (("filter",), ("filter",), ("filter",), ("target",)),
        "roe",
        "Trong nhóm biên ròng giảm nhưng vòng quay và ROE cùng tăng, hỏi ROE cao nhất.",
        ("Không dùng ngôn ngữ nhân quả.",),
        template_id="C15",
        required_period_count=3,
        predicate_operators=("<", ">", ">"),
    ),
    AnalyticalFrame(
        TEMPLATE_A05,
        "cash_margin_gap_leverage",
        "same_period",
        "rank_lookup",
        ("cfo_minus_net_margin", "liabilities_to_equity"),
        (
            "Chênh lệch CFO margin trừ biên lợi nhuận ròng",
            "Hệ số nợ phải trả trên vốn chủ sở hữu",
        ),
        (("rank",), ("target",)),
        "liabilities_to_equity",
        "Chọn doanh nghiệp có chênh lệch CFO margin trừ biên lợi nhuận ròng lớn nhất rồi hỏi hệ số nợ phải trả trên vốn chủ sở hữu.",
        (
            "Không gán chênh lệch hai biên là bằng chứng gian lận hoặc chất lượng lợi nhuận tuyệt đối.",
        ),
        template_id="A05",
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        TEMPLATE_C05,
        "cash_margin_gap_leverage",
        "same_period",
        "rank_lookup",
        ("cfo_minus_net_margin", "liabilities_to_equity"),
        (
            "Chênh lệch CFO margin trừ biên lợi nhuận ròng",
            "Hệ số nợ phải trả trên vốn chủ sở hữu",
        ),
        (("rank",), ("target",)),
        "liabilities_to_equity",
        "Chọn doanh nghiệp có chênh lệch CFO margin trừ biên lợi nhuận ròng lớn nhất rồi hỏi hệ số nợ phải trả trên vốn chủ sở hữu.",
        ("Không gán chênh lệch hai biên là một nhãn chất lượng tuyệt đối.",),
        template_id="C05",
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        TEMPLATE_A13,
        "dynamic_cash_liquidity",
        "same_period",
        "rank_lookup",
        ("operating_cash_flow_ratio", "quick_ratio"),
        ("Hệ số dòng tiền hoạt động trên nợ ngắn hạn", "Hệ số thanh toán nhanh"),
        (("rank",), ("target",)),
        "quick_ratio",
        "Chọn doanh nghiệp có hệ số dòng tiền hoạt động trên nợ ngắn hạn thấp nhất rồi hỏi hệ số thanh toán nhanh.",
        ("Không gọi hệ số dòng tiền hoạt động là defensive interval ratio.",),
        template_id="A13",
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        TEMPLATE_C11,
        "dynamic_cash_liquidity",
        "same_period",
        "rank_lookup",
        ("operating_cash_flow_ratio", "quick_ratio"),
        ("Hệ số dòng tiền hoạt động trên nợ ngắn hạn", "Hệ số thanh toán nhanh"),
        (("rank",), ("target",)),
        "quick_ratio",
        "Chọn doanh nghiệp có hệ số dòng tiền hoạt động trên nợ ngắn hạn thấp nhất rồi hỏi hệ số thanh toán nhanh.",
        ("Không dùng tên defensive interval ratio.",),
        template_id="C11",
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        TEMPLATE_C04,
        "accruals_dynamic_liquidity",
        "adjacent_period",
        "rank_lookup",
        (EQ_01_TERMINAL_KEY, "operating_cash_flow_ratio"),
        (
            "Tỷ lệ dồn tích hoạt động trên tài sản bình quân",
            "Hệ số dòng tiền hoạt động trên nợ ngắn hạn",
        ),
        (("rank",), ("target",)),
        "operating_cash_flow_ratio",
        "Chọn doanh nghiệp có tỷ lệ dồn tích hoạt động trên tài sản bình quân cao nhất rồi hỏi hệ số dòng tiền hoạt động trên nợ ngắn hạn.",
        (
            "Chỉ coi tỷ lệ dồn tích là tín hiệu cần xem xét, không phải bằng chứng thao túng.",
        ),
        template_id="C04",
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        TEMPLATE_B02,
        "inventory_cycle_cash_margin",
        "adjacent_period",
        "rank_lookup",
        ("inventory_days", "cfo_margin"),
        ("Số ngày tồn kho", "Biên dòng tiền từ hoạt động kinh doanh trên doanh thu"),
        (("rank",), ("target",)),
        "cfo_margin",
        "Chọn doanh nghiệp có số ngày tồn kho dài nhất rồi hỏi CFO margin cùng năm.",
        ("Chỉ áp dụng cho ngành mà hàng tồn kho có ý nghĩa hoạt động.",),
        template_id="B02",
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        TEMPLATE_A16,
        "leverage_interest_coverage",
        "adjacent_period",
        "rank_lookup",
        ("equity_multiplier", "interest_coverage"),
        (
            "Hệ số nhân vốn chủ sở hữu theo số dư bình quân",
            "Hệ số khả năng thanh toán lãi vay",
        ),
        (("rank",), ("target",)),
        "interest_coverage",
        "Chọn doanh nghiệp có hệ số nhân vốn chủ sở hữu cao nhất rồi hỏi hệ số khả năng thanh toán lãi vay.",
        ("Interest coverage dùng proxy EBIT đã công bố của project.",),
        template_id="A16",
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        TEMPLATE_A24,
        "long_term_asset_intensity",
        "adjacent_period",
        "rank_lookup",
        ("long_term_assets_share", "asset_turnover_avg"),
        (
            "Tỷ trọng tài sản dài hạn trên tổng tài sản",
            "Vòng quay tổng tài sản theo tổng tài sản bình quân",
        ),
        (("rank",), ("target",)),
        "asset_turnover_avg",
        "Chọn doanh nghiệp có tỷ trọng tài sản dài hạn cao nhất rồi hỏi vòng quay tổng tài sản.",
        ("Không tự suy ra nguyên nhân mức độ thâm dụng tài sản.",),
        template_id="A24",
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        TEMPLATE_D01,
        "large_company_profitability",
        "nonfinancial_adjacent_period",
        "entity_filter_aggregate",
        ("cdkt:270", "roa"),
        ("Tổng tài sản", "ROA theo tổng tài sản bình quân"),
        (("filter",), ("target",)),
        "roa",
        "Trong universe phi tài chính vượt ngưỡng tổng tài sản cố định, hỏi ROA cao nhất.",
        (
            "Universe phải được mô tả là corpus có đủ coverage, không gọi là toàn thị trường.",
        ),
        template_id="D01",
        threshold_operator=">",
        threshold_value=100_000_000_000_000.0,
        aggregate_operation="maximum",
    ),
    AnalyticalFrame(
        TEMPLATE_D09,
        "low_quick_ratio_cash_coverage",
        "nonfinancial_same_period",
        "entity_filter_aggregate",
        ("quick_ratio", "operating_cash_flow_ratio"),
        ("Hệ số thanh toán nhanh", "Hệ số dòng tiền hoạt động trên nợ ngắn hạn"),
        (("filter",), ("target",)),
        "operating_cash_flow_ratio",
        "Trong các doanh nghiệp có hệ số thanh toán nhanh dưới 1 lần, hỏi hệ số dòng tiền hoạt động cao nhất.",
        ("Không kết luận chắc chắn về khả năng thanh toán từ hai tỷ số.",),
        template_id="D09",
        threshold_operator="<",
        threshold_value=1.0,
        aggregate_operation="maximum",
    ),
    AnalyticalFrame(
        "TPL_A04_CCC_MIN_ROA",
        "working_capital_cycle_profitability",
        "industry_wide_adjacent_period",
        "rank_lookup",
        ("cash_conversion_cycle", "roa"),
        ("Chu kỳ chuyển đổi tiền mặt", "ROA theo tổng tài sản bình quân"),
        (("rank",), ("target",)),
        "roa",
        "Chọn doanh nghiệp có CCC ngắn nhất trong ngành rồi hỏi ROA.",
        ("Không dùng COGS thay purchases nếu nguồn purchases không đáng tin cậy.",),
        template_id="A04",
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        "TPL_A06_PBT_OPERATING_SHARE_FUTURE_GROWTH",
        "operating_profit_mix_future_growth",
        "industry_wide_adjacent_period",
        "filtered_transform_rank_lookup",
        ("kqkd:50", "operating_profit_to_pbt", "kqkd:10"),
        (
            "Lợi nhuận trước thuế",
            "Lợi nhuận hoạt động trên LNTT",
            "Tăng trưởng doanh thu năm sau",
        ),
        (("filter",), ("rank",), ("target",)),
        "kqkd:10",
        "Trong cohort LNTT dương đủ lớn, chọn tỷ lệ operating profit/PBT thấp nhất rồi hỏi tăng trưởng doanh thu năm sau.",
        ("Tỷ lệ operating profit/PBT chỉ là tỷ lệ mô tả.",),
        template_id="A06",
        threshold_operator=">",
        threshold_value=100_000_000_000.0,
        selector_direction="argmin",
        selector_transform="identity",
        target_transform="growth",
        filter_transform="identity",
        filter_period_offset=-2,
        rank_period_offset=-2,
        target_period_offset=-1,
    ),
    AnalyticalFrame(
        "TPL_A08_DEBT_COST_NET_MARGIN",
        "recorded_debt_cost_profitability",
        "industry_wide_adjacent_period",
        "rank_lookup",
        ("recorded_debt_cost_proxy", "net_margin"),
        ("Proxy chi phí nợ ghi nhận", "Biên lợi nhuận ròng"),
        (("rank",), ("target",)),
        "net_margin",
        "Chọn doanh nghiệp có proxy chi phí nợ ghi nhận cao nhất rồi hỏi biên lợi nhuận ròng.",
        ("Proxy dùng nợ chịu lãi bình quân, không dùng tổng nợ phải trả.",),
        template_id="A08",
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        "TPL_A09_DOL_OPERATING_MARGIN",
        "operating_leverage_margin",
        "industry_wide_adjacent_period",
        "dol_terminal",
        ("kqkd:10", "kqkd:30", "operating_margin"),
        (
            "Tăng trưởng doanh thu thuần",
            "Tăng trưởng lợi nhuận thuần từ HĐKD",
            "Biên lợi nhuận hoạt động",
        ),
        (("filter", "rank"), ("rank",), ("target",)),
        "operating_margin",
        "Trong cohort tăng doanh thu, chọn DOL cao nhất rồi hỏi biên lợi nhuận hoạt động.",
        ("DOL dùng tăng trưởng operating profit chia tăng trưởng revenue.",),
        template_id="A09",
        threshold_value=0.0,
        selector_direction="argmax",
        required_question_terms=("Lợi nhuận thuần từ HĐKD dương ở cả hai kỳ",),
    ),
    AnalyticalFrame(
        "TPL_A14_DEBT_MATURITY_INTEREST_GROWTH",
        "debt_maturity_interest_burden",
        "industry_wide_adjacent_period",
        "dual_transform_panel_rank_lookup",
        ("long_term_debt_share", "kqkd:23"),
        ("Tỷ trọng nợ dài hạn", "Tăng trưởng chi phí lãi vay"),
        (("rank",), ("target",)),
        "kqkd:23",
        "Chọn doanh nghiệp tăng tỷ trọng nợ dài hạn mạnh nhất rồi hỏi tăng trưởng chi phí lãi vay.",
        ("Phân loại nợ chịu lãi phải nhất quán qua hai kỳ.",),
        template_id="A14",
        selector_direction="argmax",
        selector_transform="period_difference",
        target_transform="growth",
    ),
    AnalyticalFrame(
        "TPL_A18_PPE_CAPEX_SHARE",
        "investment_intensity_contribution",
        "industry_wide_adjacent_period",
        "transform_rank_share",
        ("gross_ppe", "cash_capex"),
        ("Mức tăng nguyên giá PPE", "CAPEX tiền mặt"),
        (("rank",), ("target",)),
        "leverage_interest_expense_share",
        "CAPEX tiền mặt của doanh nghiệp tăng nguyên giá PPE mạnh nhất chiếm bao nhiêu phần trăm tổng CAPEX ngành.",
        ("Không đồng nhất thay đổi nguyên giá PPE với CAPEX tiền mặt.",),
        template_id="A18",
        selector_direction="argmax",
        selector_transform="period_difference",
    ),
    AnalyticalFrame(
        "TPL_A19_REVENUE_INCREASE_SHARE",
        "revenue_growth_contribution",
        "industry_wide_adjacent_period",
        "transform_rank_share",
        ("kqkd:10",),
        ("Mức tăng tuyệt đối doanh thu thuần",),
        (("rank", "target"),),
        "leverage_interest_expense_share",
        "Doanh nghiệp tăng doanh thu tuyệt đối lớn nhất đóng góp bao nhiêu vào tổng mức tăng dương của ngành.",
        ("Mẫu số chỉ gồm các mức tăng dương, không net với doanh nghiệp giảm.",),
        template_id="A19",
        selector_direction="argmax",
        selector_transform="period_difference",
        topology_mode="positive_only",
    ),
    AnalyticalFrame(
        "TPL_A20_POSITIVE_CFO_CAPEX_SHARE",
        "cash_backed_investment_share",
        "industry_wide_same_period",
        "filter_share",
        ("lctt:20", "cash_capex"),
        ("Dòng tiền kinh doanh", "CAPEX tiền mặt"),
        (("filter",), ("target",)),
        "leverage_interest_expense_share",
        "Tổng CAPEX tiền mặt của cohort CFO dương chiếm bao nhiêu phần trăm tổng CAPEX ngành.",
        (
            "CAPEX là cash outflow mua tài sản dài hạn, không phải net investing cash flow.",
        ),
        template_id="A20",
        threshold_operator=">",
        threshold_value=0.0,
    ),
    AnalyticalFrame(
        "TPL_A22_NET_MARGIN_ROA_CHANGE",
        "margin_decline_asset_returns",
        "industry_wide_cross_entity_period_window",
        "dual_transform_panel_rank_lookup",
        ("net_margin", "roa"),
        ("Thay đổi biên lợi nhuận ròng", "Thay đổi ROA"),
        (("rank",), ("target",)),
        "cash_roe_cohort_gap",
        "Chọn doanh nghiệp giảm biên ròng mạnh nhất rồi hỏi thay đổi ROA cùng giai đoạn.",
        ("Không diễn giải suy giảm biên là nguyên nhân duy nhất của ROA.",),
        template_id="A22",
        selector_direction="argmin",
        selector_transform="period_difference",
        target_transform="period_difference",
        transform_source_offset=1,
        required_period_count=3,
    ),
    AnalyticalFrame(
        "TPL_B01_LARGE_REVENUE_MARGIN_DECLINE_ROA",
        "large_company_margin_pressure",
        "nonfinancial_cross_entity_period_window",
        "filtered_transform_rank_lookup",
        ("kqkd:10", "gross_margin", "roa"),
        ("Doanh thu thuần", "Thay đổi biên lợi nhuận gộp", "ROA"),
        (("filter",), ("rank",), ("target",)),
        "roa",
        "Trong universe doanh thu lớn, chọn doanh nghiệp giảm biên gộp mạnh nhất rồi hỏi ROA.",
        ("Universe là corpus phi tài chính đủ coverage.",),
        template_id="B01",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        selector_direction="argmin",
        filter_transform="identity",
        selector_transform="period_difference",
        target_transform="identity",
        required_period_count=3,
    ),
    AnalyticalFrame(
        "TPL_B03_LARGE_ASSETS_ACCRUALS_ROE",
        "large_company_accruals_profitability",
        "nonfinancial_adjacent_period",
        "entity_filter_rank_lookup",
        ("cdkt:270", "operating_accruals_ratio", "roe"),
        ("Tổng tài sản", "Tỷ lệ dồn tích hoạt động", "ROE"),
        (("filter",), ("rank",), ("target",)),
        "roe",
        "Trong cohort tổng tài sản lớn, chọn operating accruals ratio cao nhất rồi hỏi ROE.",
        ("Dồn tích chỉ là tín hiệu mô tả, không phải bằng chứng thao túng.",),
        template_id="B03",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        "TPL_B04_TOP_REVENUE_LIQUIDITY_LEVERAGE_COUNT",
        "large_revenue_liquidity_leverage_screen",
        "industry_wide_same_period",
        "top_n_dual_predicate_count",
        ("kqkd:10", "quick_ratio", "liabilities_to_equity"),
        (
            "Doanh thu thuần",
            "Hệ số thanh toán nhanh",
            "Nợ phải trả trên vốn chủ sở hữu",
        ),
        (("rank",), ("filter",), ("filter",)),
        "growth_margin_squeeze_count",
        "Trong top doanh thu, đếm doanh nghiệp đồng thời có quick ratio cao và liabilities-to-equity thấp.",
        ("Tie tại biên top-N bị reject.",),
        template_id="B04",
        top_n=5,
        predicate_operators=(">", "<"),
        predicate_thresholds=(1.0, 1.0),
        target_unit_label="công ty",
    ),
    AnalyticalFrame(
        "TPL_C01_SIZE_DOH_GPM_CHANGE",
        "inventory_cycle_margin_pressure",
        "industry_wide_cross_entity_period_window",
        "filtered_transform_rank_lookup",
        ("kqkd:10", "inventory_days", "gross_margin"),
        ("Doanh thu thuần", "Thay đổi số ngày tồn kho", "Thay đổi biên lợi nhuận gộp"),
        (("filter",), ("rank",), ("target",)),
        "cash_roe_cohort_gap",
        "Trong cohort doanh thu đủ lớn, chọn doanh nghiệp tăng số ngày tồn kho mạnh nhất rồi hỏi thay đổi biên gộp.",
        ("Không kết luận tồn kho tăng gây ra thay đổi biên gộp.",),
        template_id="C01",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        selector_direction="argmax",
        filter_transform="identity",
        selector_transform="period_difference",
        target_transform="period_difference",
        transform_source_offset=1,
        required_period_count=3,
    ),
    AnalyticalFrame(
        "TPL_C03_SIZE_CCC_ROA",
        "working_capital_cycle_profitability",
        "industry_wide_adjacent_period",
        "entity_filter_rank_lookup",
        ("kqkd:10", "cash_conversion_cycle", "roa"),
        ("Doanh thu thuần", "Chu kỳ chuyển đổi tiền mặt", "ROA"),
        (("filter",), ("rank",), ("target",)),
        "roa",
        "Trong cohort doanh thu đủ lớn, chọn CCC ngắn nhất rồi hỏi ROA.",
        ("CCC chỉ được dùng khi có purchases đáng tin cậy.",),
        template_id="C03",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        "TPL_C06_SIZE_OPERATING_SHARE_FUTURE_GROWTH",
        "operating_profit_mix_future_growth",
        "nonfinancial_adjacent_period",
        "filtered_transform_rank_lookup",
        ("kqkd:10", "operating_profit_to_pbt", "kqkd:10"),
        (
            "Doanh thu thuần",
            "Lợi nhuận hoạt động trên LNTT",
            "Tăng trưởng doanh thu năm sau",
        ),
        (("filter",), ("rank",), ("target",)),
        "kqkd:10",
        "Trong cohort doanh thu đủ lớn, chọn operating profit/PBT thấp nhất rồi hỏi tăng trưởng doanh thu năm sau.",
        ("Tỷ lệ operating profit/PBT chỉ là tỷ lệ mô tả.",),
        template_id="C06",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        selector_direction="argmin",
        filter_transform="identity",
        selector_transform="identity",
        target_transform="growth",
        filter_period_offset=-2,
        rank_period_offset=-2,
        target_period_offset=-1,
    ),
    AnalyticalFrame(
        "TPL_C07_POSITIVE_GROWTH_DOL_MAX",
        "operating_leverage_screen",
        "industry_wide_adjacent_period",
        "dol_terminal",
        ("kqkd:10", "kqkd:30"),
        ("Tăng trưởng doanh thu thuần", "Tăng trưởng lợi nhuận thuần từ HĐKD"),
        (("filter", "target"), ("target",)),
        "dol",
        "Trong cohort tăng doanh thu, hỏi DOL cao nhất.",
        ("DOL cực trị bị kiểm soát bằng ngưỡng mẫu số.",),
        template_id="C07",
        threshold_value=0.0,
        aggregate_operation="maximum",
        required_question_terms=("Lợi nhuận thuần từ HĐKD dương ở cả hai kỳ",),
    ),
    AnalyticalFrame(
        "TPL_C10_NEGATIVE_NWC_LOW_LEVERAGE_ROA",
        "working_capital_leverage_profitability",
        "industry_wide_adjacent_period",
        "nested_median_aggregate",
        ("net_working_capital", "liabilities_to_assets", "roa"),
        ("Vốn lưu động ròng", "Nợ phải trả trên tổng tài sản", "ROA"),
        (("filter",), ("filter",), ("target",)),
        "roa",
        "Trong cohort NWC âm, tiếp tục lọc liabilities-to-assets dưới median cohort rồi hỏi ROA cao nhất.",
        ("Median phải tính lại trong cohort NWC âm.",),
        template_id="C10",
        threshold_operator="<",
        threshold_value=0.0,
        predicate_operators=("<",),
        aggregate_operation="maximum",
    ),
    AnalyticalFrame(
        "TPL_C12_DEBT_MATURITY_COVERAGE",
        "debt_maturity_interest_coverage",
        "industry_wide_adjacent_period",
        "filtered_transform_rank_lookup",
        ("short_term_debt_share", "interest_coverage"),
        ("Thay đổi tỷ trọng nợ ngắn hạn", "Hệ số khả năng thanh toán lãi vay"),
        (("rank",), ("target",)),
        "interest_coverage",
        "Chọn doanh nghiệp giảm tỷ trọng nợ ngắn hạn mạnh nhất rồi hỏi interest coverage.",
        ("Tỷ trọng dùng tổng nợ chịu lãi phân kỳ.",),
        template_id="C12",
        selector_direction="argmin",
        selector_transform="period_difference",
        target_transform="identity",
    ),
    AnalyticalFrame(
        "TPL_C14_PPE_CAPEX_SHARE",
        "investment_intensity_contribution",
        "industry_wide_adjacent_period",
        "transform_rank_share",
        ("gross_ppe", "cash_capex"),
        ("Mức tăng nguyên giá PPE", "CAPEX tiền mặt"),
        (("rank",), ("target",)),
        "leverage_interest_expense_share",
        "Doanh nghiệp tăng nguyên giá PPE mạnh nhất chiếm bao nhiêu phần trăm tổng CAPEX tiền mặt ngành.",
        ("Không đồng nhất gross PPE change với cash CAPEX.",),
        template_id="C14",
        selector_direction="argmax",
        selector_transform="period_difference",
    ),
    AnalyticalFrame(
        "TPL_D02_GROWTH_ROE_NEGATIVE_CFO_COUNT",
        "growth_profitability_cash_screen",
        "nonfinancial_adjacent_period",
        "multi_predicate_terminal",
        ("kqkd:10", "roe", "lctt:20"),
        ("Tăng trưởng doanh thu thuần", "ROE", "Dòng tiền kinh doanh"),
        (("filter",), ("filter",), ("filter",)),
        "growth_margin_squeeze_count",
        "Đếm doanh nghiệp đồng thời tăng doanh thu, ROE cao và CFO âm.",
        ("Các ngưỡng được khóa trước terminal values.",),
        template_id="D02",
        predicate_transforms=("growth", "identity", "identity"),
        predicate_operators=(">", ">", "<"),
        predicate_thresholds=(0.10, 0.10, 0.0),
        aggregate_operation="set_count",
        target_unit_label="công ty",
    ),
    AnalyticalFrame(
        "TPL_D03_ASSET_GROWTH_TURNOVER_DECLINE",
        "asset_expansion_efficiency_decline",
        "nonfinancial_cross_entity_period_window",
        "transformed_filter_aggregate_terminal",
        ("cdkt:270", "asset_turnover_avg"),
        ("Tăng trưởng tổng tài sản", "Vòng quay tổng tài sản"),
        (("filter",), ("target",)),
        "asset_turnover_avg",
        "Trong cohort tăng tài sản mạnh, hỏi độ lớn suy giảm vòng quay tài sản lớn nhất.",
        ("Terminal báo độ lớn dương của minimum delta âm.",),
        template_id="D03",
        threshold_operator=">",
        threshold_value=0.10,
        filter_transform="growth",
        target_transform="period_difference",
        aggregate_operation="minimum",
        absolute_terminal=True,
        transform_source_offset=1,
        required_period_count=3,
    ),
    AnalyticalFrame(
        "TPL_D04_SIZE_ACCRUALS_NET_MARGIN",
        "large_company_accruals_margin",
        "nonfinancial_adjacent_period",
        "entity_filter_rank_lookup",
        ("kqkd:10", "operating_accruals_ratio", "net_margin"),
        ("Doanh thu thuần", "Tỷ lệ dồn tích hoạt động", "Biên lợi nhuận ròng"),
        (("filter",), ("rank",), ("target",)),
        "net_margin",
        "Trong cohort doanh thu đủ lớn, chọn operating accruals ratio thấp nhất rồi hỏi net margin.",
        ("Dồn tích âm không phải bằng chứng chất lượng cao.",),
        template_id="D04",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        "TPL_D06_PBT_OPERATING_SHARE_FUTURE_GROWTH",
        "operating_profit_mix_future_growth",
        "nonfinancial_adjacent_period",
        "filtered_transform_rank_lookup",
        ("kqkd:50", "operating_profit_to_pbt", "kqkd:10"),
        (
            "Lợi nhuận trước thuế",
            "Lợi nhuận hoạt động trên LNTT",
            "Tăng trưởng doanh thu năm sau",
        ),
        (("filter",), ("rank",), ("target",)),
        "kqkd:10",
        "Trong cohort LNTT đủ lớn, chọn operating profit/PBT thấp nhất rồi hỏi tăng trưởng doanh thu năm sau.",
        ("Tỷ lệ operating profit/PBT chỉ là tỷ lệ mô tả.",),
        template_id="D06",
        threshold_operator=">",
        threshold_value=100_000_000_000.0,
        selector_direction="argmin",
        filter_transform="identity",
        selector_transform="identity",
        target_transform="growth",
        filter_period_offset=-2,
        rank_period_offset=-2,
        target_period_offset=-1,
    ),
    AnalyticalFrame(
        "TPL_D07_NWC_TRANSITION_OCF_ROA",
        "working_capital_transition_cash_returns",
        "nonfinancial_adjacent_period",
        "transition_filter_rank_lookup",
        ("net_working_capital", "cdkt:310", "operating_cash_flow_ratio", "roa"),
        ("Vốn lưu động ròng", "Nợ ngắn hạn", "Hệ số dòng tiền hoạt động", "ROA"),
        (("filter",), ("filter",), ("rank",), ("target",)),
        "roa",
        "Trong cohort chuyển từ NWC không âm sang âm và nợ ngắn hạn đủ lớn, chọn OCF ratio cao nhất rồi hỏi ROA.",
        ("Không nói dòng tiền cả năm bù trực tiếp thiếu hụt NWC cuối kỳ.",),
        template_id="D07",
        threshold_value=1_000_000_000_000.0,
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        "TPL_D08_SIZE_CCC_MAX",
        "large_company_working_capital_cycle",
        "nonfinancial_adjacent_period",
        "entity_filter_aggregate",
        ("kqkd:10", "cash_conversion_cycle"),
        ("Doanh thu thuần", "Chu kỳ chuyển đổi tiền mặt"),
        (("filter",), ("target",)),
        "cash_conversion_cycle",
        "Trong cohort doanh thu đủ lớn, hỏi CCC dài nhất.",
        ("CCC cần purchases đáng tin cậy.",),
        template_id="D08",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        aggregate_operation="maximum",
    ),
    AnalyticalFrame(
        "TPL_D10_INTEREST_SCALE_COVERAGE_LEVERAGE",
        "interest_burden_coverage_leverage",
        "nonfinancial_same_period",
        "entity_filter_rank_lookup",
        ("kqkd:23", "interest_coverage", "liabilities_to_equity"),
        (
            "Chi phí lãi vay",
            "Hệ số khả năng thanh toán lãi vay",
            "Nợ phải trả trên vốn chủ sở hữu",
        ),
        (("filter",), ("rank",), ("target",)),
        "liabilities_to_equity",
        "Trong cohort chi phí lãi vay đủ lớn, chọn coverage cao nhất rồi hỏi liabilities-to-equity.",
        ("Không gọi liabilities-to-equity là D/E nợ vay.",),
        template_id="D10",
        threshold_operator=">",
        threshold_value=10_000_000_000.0,
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        "TPL_D11_DEBT_SCALE_MATURITY_INTEREST_GROWTH",
        "debt_maturity_interest_burden",
        "nonfinancial_adjacent_period",
        "filtered_transform_rank_lookup",
        ("total_interest_bearing_debt", "short_term_debt_share", "kqkd:23"),
        (
            "Tổng nợ chịu lãi",
            "Thay đổi tỷ trọng nợ ngắn hạn",
            "Tăng trưởng chi phí lãi vay",
        ),
        (("filter",), ("rank",), ("target",)),
        "kqkd:23",
        "Trong cohort tổng nợ chịu lãi đủ lớn, chọn giảm tỷ trọng nợ ngắn hạn mạnh nhất rồi hỏi tăng trưởng lãi vay.",
        ("Debt split phải nhất quán qua hai kỳ.",),
        template_id="D11",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        selector_direction="argmin",
        filter_transform="identity",
        selector_transform="period_difference",
        target_transform="growth",
    ),
    AnalyticalFrame(
        "TPL_D12_HIGH_ROE_LOW_ROA_EQUITY_MULTIPLIER",
        "dupont_leverage_screen",
        "nonfinancial_adjacent_period",
        "multi_predicate_terminal",
        ("roe", "roa", "equity_multiplier"),
        ("ROE", "ROA", "Hệ số nhân vốn chủ sở hữu"),
        (("filter",), ("filter",), ("target",)),
        "equity_multiplier",
        "Trong cohort ROE cao nhưng ROA thấp, hỏi equity multiplier bình quân.",
        ("Bình quân là arithmetic mean của ratio từng doanh nghiệp.",),
        template_id="D12",
        predicate_transforms=("identity", "identity"),
        predicate_operators=(">", "<"),
        predicate_thresholds=(0.15, 0.05),
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        "TPL_D13_SIZE_GROWTH_DOL_MAX",
        "controlled_operating_leverage",
        "nonfinancial_adjacent_period",
        "dol_terminal",
        ("kqkd:10", "kqkd:10", "kqkd:30"),
        (
            "Doanh thu thuần",
            "Tăng trưởng doanh thu thuần",
            "Đòn bẩy hoạt động (DOL)",
        ),
        (("filter",), ("filter",), ("target",)),
        "dol",
        "Trong cohort doanh thu đủ lớn và tăng trưởng doanh thu trên 10%, hỏi DOL cao nhất.",
        ("DOL dùng operating profit, không dùng gross profit.",),
        template_id="D13",
        threshold_value=1_000_000_000_000.0,
        secondary_threshold_value=0.10,
        topology_mode="with_size",
        aggregate_operation="maximum",
        public_metric_keys=("kqkd:10", "kqkd:10", "dol"),
        required_question_terms=(
            "Tăng trưởng lợi nhuận thuần từ HĐKD chia cho Tăng trưởng doanh thu thuần",
            "Lợi nhuận thuần từ HĐKD dương ở cả hai kỳ",
        ),
        require_explicit_period_span=True,
    ),
    AnalyticalFrame(
        "TPL_D14_REVENUE_DECLINE_SGA_MARGIN",
        "cost_discipline_in_contraction",
        "nonfinancial_adjacent_period",
        "filtered_transform_rank_lookup",
        ("kqkd:10", "sga_expense", "operating_margin"),
        ("Doanh thu thuần", "SG&A", "Biên lợi nhuận hoạt động"),
        (("filter",), ("rank",), ("target",)),
        "operating_margin",
        "Trong cohort doanh thu giảm mạnh, chọn doanh nghiệp giảm SG&A mạnh nhất rồi hỏi operating margin.",
        ("Không khẳng định giảm SG&A gây ra biên hoạt động.",),
        template_id="D14",
        threshold_operator="<",
        threshold_value=-0.10,
        selector_direction="argmin",
        filter_transform="growth",
        selector_transform="growth",
        target_transform="identity",
    ),
    AnalyticalFrame(
        "TPL_D17_SIZE_CFO_MARGIN_ROA_GAP",
        "large_company_cash_margin_return_gap",
        "nonfinancial_adjacent_period",
        "filtered_ranked_cohort_gap",
        ("kqkd:10", "cfo_margin", "roa"),
        ("Doanh thu thuần", "CFO margin", "ROA"),
        (("filter",), ("rank",), ("target",)),
        "cash_roe_cohort_gap",
        "Sau bộ lọc doanh thu, hỏi chênh lệch ROA bình quân giữa quantile CFO margin cao và thấp.",
        ("Mỗi cohort cần ít nhất ba doanh nghiệp và reject boundary tie.",),
        template_id="D17",
        threshold_operator=">",
        threshold_value=1_000_000_000_000.0,
        quantile_percent=25.0,
    ),
    AnalyticalFrame(
        "TPL_D18_MARGIN_DECLINE_TURNOVER_ROE",
        "margin_pressure_asset_efficiency",
        "nonfinancial_cross_entity_period_window",
        "filtered_transform_rank_lookup",
        ("gross_margin", "asset_turnover_avg", "roe"),
        ("Biên lợi nhuận gộp", "Vòng quay tổng tài sản", "ROE"),
        (("filter",), ("rank",), ("target",)),
        "roe",
        "Trong cohort biên gộp giảm mạnh, chọn doanh nghiệp tăng turnover mạnh nhất rồi hỏi ROE.",
        ("Không nói turnover cải thiện đã bù lại biên gộp.",),
        template_id="D18",
        threshold_operator="<",
        threshold_value=-0.05,
        selector_direction="argmax",
        filter_transform="period_difference",
        selector_transform="period_difference",
        target_transform="identity",
        transform_source_offset=1,
        required_period_count=3,
    ),
    AnalyticalFrame(
        CASH_ROE_GAP,
        "cash_profitability",
        "adjacent_period",
        "sign_cohort_gap",
        ("lctt:20", "roe"),
        (
            "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
            "ROE theo vốn chủ sở hữu bình quân",
        ),
        (("filter",), ("target",)),
        "cash_roe_cohort_gap",
        "Hỏi bình quân ROE theo vốn chủ sở hữu bình quân của cohort có dòng tiền kinh doanh dương cao hơn cohort có dòng tiền kinh doanh âm bao nhiêu điểm phần trăm.",
        (
            "ROE dùng lợi nhuận sau thuế năm cuối chia vốn chủ sở hữu bình quân của năm liền trước và năm cuối.",
            "Chỉ mô tả khác biệt giữa hai cohort, không kết luận dòng tiền gây ra ROE cao hay thấp.",
        ),
        threshold_operator=">",
        threshold_value=0.0,
        aggregate_operation="average",
        required_question_terms=("âm", "cao hơn", "bình quân"),
    ),
    AnalyticalFrame(
        GROWTH_MARGIN_SQUEEZE_COUNT,
        "revenue_growth_margin_pressure",
        "adjacent_period",
        "dual_panel_count",
        ("kqkd:10", "gross_margin"),
        ("Doanh thu thuần", "Biên lợi nhuận gộp"),
        (("filter",), ("target",)),
        "growth_margin_squeeze_count",
        "Đếm số công ty đồng thời có doanh thu thuần tăng và biên lợi nhuận gộp giảm giữa hai năm liền kề.",
        (
            "Đây là screen mô tả mở rộng doanh thu đi kèm thu hẹp biên gộp, không quy kết nguyên nhân.",
        ),
        target_unit_label="công ty",
        required_question_terms=("tăng", "giảm"),
    ),
    AnalyticalFrame(
        PERSISTENT_CASH_GROWTH_AVG,
        "cash_backed_revenue_growth",
        "adjacent_period",
        "temporal_growth_average",
        ("lctt:20", "kqkd:10"),
        (
            "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
            "Tăng trưởng doanh thu thuần",
        ),
        (("filter",), ("target",)),
        "persistent_cash_revenue_growth",
        "Trong cohort duy trì dòng tiền kinh doanh dương ở cả hai năm, hỏi tốc độ tăng trưởng doanh thu thuần bình quân giữa hai năm.",
        ("Không diễn giải dòng tiền dương là nguyên nhân tạo tăng trưởng doanh thu.",),
        threshold_operator=">",
        threshold_value=0.0,
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        PERSISTENT_CASH_GROWTH_ROA,
        "cash_backed_revenue_growth",
        "cross_entity_period_window",
        "temporal_filter_rank_lookup",
        ("lctt:20", "kqkd:10", "roa"),
        (
            "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
            "Tăng trưởng doanh thu thuần",
            "ROA theo tổng tài sản bình quân",
        ),
        (("filter",), ("rank",), ("target",)),
        "roa",
        "Trong cohort duy trì dòng tiền kinh doanh dương ở mọi năm của window, chọn công ty tăng trưởng doanh thu cao nhất tại năm cuối rồi hỏi ROA năm cuối theo tổng tài sản bình quân.",
        (
            "Điều kiện dòng tiền kinh doanh dương phải giữ ở mọi năm trong window.",
            "ROA dùng lợi nhuận sau thuế năm cuối chia tổng tài sản bình quân của năm liền trước và năm cuối.",
        ),
        threshold_operator=">",
        threshold_value=0.0,
        selector_direction="argmax",
        selector_transform="growth",
        required_question_terms=("bình quân",),
    ),
    AnalyticalFrame(
        GROWTH_MARGIN_CHANGE_AVG,
        "revenue_growth_margin_pressure",
        "adjacent_period",
        "panel_filter_average",
        ("kqkd:10", "gross_margin"),
        ("Tăng trưởng doanh thu thuần", "Thay đổi biên lợi nhuận gộp"),
        (("filter",), ("target",)),
        "growth_cohort_margin_change",
        "Trong cohort có doanh thu thuần tăng, hỏi mức thay đổi biên lợi nhuận gộp bình quân giữa hai năm.",
        (
            "Mức thay đổi có thể âm; không mô tả quan hệ nhân quả giữa doanh thu và biên gộp.",
        ),
        threshold_operator=">",
        threshold_value=0.0,
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        REVENUE_GROWTH_GROSS_MARGIN_PANEL_LOOKUP,
        "revenue_growth_margin_leader_quality",
        "cross_entity_period_window",
        "panel_rank_lookup",
        ("kqkd:10", "gross_margin"),
        ("Doanh thu thuần", "Biên lợi nhuận gộp"),
        (("rank",), ("target",)),
        "gross_margin",
        "Trong panel nhiều công ty và nhiều năm liên tục, chọn company-year có tăng trưởng doanh thu thuần cao nhất rồi hỏi biên lợi nhuận gộp tại đúng company-year đó.",
        (
            "Tăng trưởng chỉ được tính cho năm có năm liền trước trong cùng window.",
            "Biên lợi nhuận gộp được lookup tại đúng company-year do tăng trưởng doanh thu chọn, không phải tại năm cuối cố định.",
            "Không kết luận tăng trưởng doanh thu là nguyên nhân tạo biên lợi nhuận gộp cao hay thấp.",
        ),
        selector_direction="argmax",
        selector_transform="growth",
        required_question_terms=("công ty-năm", "tăng trưởng"),
    ),
    AnalyticalFrame(
        REVENUE_GROWTH_ASSET_TURNOVER_PANEL_LOOKUP,
        "revenue_growth_asset_efficiency_leader",
        "cross_entity_period_window",
        "panel_rank_lookup",
        ("kqkd:10", "asset_turnover"),
        ("Doanh thu thuần", "Vòng quay tổng tài sản"),
        (("rank",), ("target",)),
        "asset_turnover",
        "Trong panel nhiều công ty và nhiều năm liên tục, chọn company-year có tăng trưởng doanh thu thuần cao nhất rồi hỏi vòng quay tổng tài sản tại đúng company-year đó.",
        (
            "Tăng trưởng chỉ được tính cho năm có năm liền trước trong cùng window.",
            "Vòng quay tổng tài sản dùng doanh thu thuần chia tổng tài sản cuối kỳ theo ratio catalog hiện có.",
            "Không kết luận tăng trưởng doanh thu là nguyên nhân tạo hiệu quả sử dụng tài sản.",
        ),
        finance_rationale="Vòng quay tổng tài sản tại company-year dẫn đầu tăng trưởng doanh thu giúp kiểm tra liệu tăng trưởng đi kèm hiệu quả sử dụng tài sản hay chỉ mở rộng quy mô.",
        selector_direction="argmax",
        selector_transform="growth",
        required_question_terms=("công ty-năm", "tăng trưởng"),
    ),
    AnalyticalFrame(
        WORKING_CAPITAL_CASH_COVERAGE_AVG,
        "working_capital_liquidity",
        "same_period",
        "entity_filter_aggregate",
        ("current_ratio", "operating_cash_flow_ratio"),
        ("Hệ số thanh toán hiện hành", "Hệ số dòng tiền hoạt động trên nợ ngắn hạn"),
        (("filter",), ("target",)),
        "operating_cash_flow_ratio",
        "Trong cohort có tài sản ngắn hạn thấp hơn nợ ngắn hạn, hỏi hệ số dòng tiền hoạt động trên nợ ngắn hạn bình quân.",
        (
            "Không nói dòng tiền cả năm bù trực tiếp cho thiếu hụt vốn lưu động cuối kỳ.",
        ),
        threshold_operator="<",
        threshold_value=1.0,
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        PROFITABLE_NEGATIVE_CASH_COUNT,
        "earnings_cash_conversion",
        "same_period",
        "dual_predicate_count",
        ("kqkd:60", "lctt:20"),
        ("Lợi nhuận sau thuế", "Lưu chuyển tiền thuần từ hoạt động kinh doanh"),
        (("filter",), ("target",)),
        "profitable_negative_cash_count",
        "Đếm số công ty có lợi nhuận sau thuế dương nhưng dòng tiền thuần từ hoạt động kinh doanh âm trong cùng năm.",
        (
            "Đây là tín hiệu lệch pha lợi nhuận-tiền, không tự động đồng nghĩa lợi nhuận kém chất lượng hay gian lận.",
        ),
        target_unit_label="công ty",
        required_question_terms=("dương", "âm"),
    ),
    AnalyticalFrame(
        PROFITABLE_ACCRUAL_AVG,
        "earnings_cash_conversion",
        "adjacent_period",
        "entity_filter_aggregate",
        ("kqkd:60", EQ_01_TERMINAL_KEY),
        ("Lợi nhuận sau thuế", "Tỷ lệ dồn tích hoạt động trên tài sản bình quân"),
        (("filter",), ("target",)),
        EQ_01_TERMINAL_KEY,
        "Trong cohort có lợi nhuận sau thuế dương ở năm cuối, hỏi tỷ lệ dồn tích hoạt động trên tài sản bình quân trung bình.",
        (
            "Tỷ lệ dồn tích hoạt động trên tài sản bình quân được tính bằng lợi nhuận sau thuế trừ dòng tiền kinh doanh, rồi chia tài sản bình quân hai năm.",
            "Chỉ là proxy chuyển đổi lợi nhuận thành tiền, không gọi là Sloan accrual hay bằng chứng thao túng.",
        ),
        threshold_operator=">",
        threshold_value=0.0,
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        CASH_POSITIVE_REVENUE_SUM,
        "cash_positive_revenue_scale",
        "same_period",
        "entity_filter_aggregate",
        ("lctt:20", "kqkd:10"),
        ("Lưu chuyển tiền thuần từ hoạt động kinh doanh", "Doanh thu thuần"),
        (("filter",), ("target",)),
        "kqkd:10",
        "Trong cohort có dòng tiền kinh doanh dương, hỏi tổng doanh thu thuần của các công ty trong cohort đó.",
        (
            "Tổng chỉ cộng doanh thu thuần cùng năm, cùng report scope và cùng đơn vị tiền tệ.",
            "Dòng tiền kinh doanh dương là điều kiện phân nhóm, không chứng minh doanh thu có chất lượng cao.",
        ),
        finance_rationale="Tổng doanh thu của nhóm tạo tiền từ hoạt động kinh doanh cho biết quy mô doanh thu đang được hỗ trợ bởi dòng tiền hoạt động dương trong cùng kỳ.",
        threshold_operator=">",
        threshold_value=0.0,
        aggregate_operation="sum",
        required_question_terms=("dương", "tổng"),
    ),
    AnalyticalFrame(
        LIQUIDITY_INVENTORY_LOAD_AVG,
        "liquidity_composition",
        "same_period",
        "entity_filter_aggregate",
        ("current_ratio", "inventory_to_current_liabilities"),
        ("Hệ số thanh toán hiện hành", "Hàng tồn kho trên nợ ngắn hạn"),
        (("filter",), ("target",)),
        "inventory_to_current_liabilities",
        "Trong cohort có tài sản ngắn hạn không thấp hơn nợ ngắn hạn, hỏi hàng tồn kho trên nợ ngắn hạn bình quân.",
        (
            "Chỉ đo phần quy mô thanh khoản hiện hành nằm ở hàng tồn kho, không đo vòng quay tồn kho.",
        ),
        threshold_operator=">=",
        threshold_value=1.0,
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        LIQUIDITY_QUICK_RATIO_AVG,
        "liquidity_composition",
        "same_period",
        "entity_filter_aggregate",
        ("current_ratio", "quick_ratio"),
        ("Hệ số thanh toán hiện hành", "Hệ số thanh toán nhanh"),
        (("filter",), ("target",)),
        "quick_ratio",
        "Trong cohort có tài sản ngắn hạn không thấp hơn nợ ngắn hạn, hỏi hệ số thanh toán nhanh bình quân để đánh giá mức bao phủ sau khi loại hàng tồn kho.",
        (
            "Không kết luận khả năng thanh toán chắc chắn chỉ từ hai hệ số tại một thời điểm.",
        ),
        threshold_operator=">=",
        threshold_value=1.0,
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        PROFITABLE_GROSS_NET_GAP_AVG,
        "profitability_margin_bridge",
        "same_period",
        "entity_filter_aggregate",
        ("net_margin", "gross_to_net_margin_difference"),
        (
            "Biên lợi nhuận ròng",
            "Chênh lệch giữa biên lợi nhuận gộp và biên lợi nhuận ròng",
        ),
        (("filter",), ("target",)),
        "gross_to_net_margin_difference",
        "Trong cohort có biên lợi nhuận ròng dương, hỏi chênh lệch bình quân giữa biên lợi nhuận gộp và biên lợi nhuận ròng.",
        (
            "Không quy toàn bộ chênh lệch cho riêng chi phí bán hàng, quản lý, lãi vay hay thuế.",
        ),
        threshold_operator=">",
        threshold_value=0.0,
        aggregate_operation="average",
    ),
    AnalyticalFrame(
        PROFITABLE_NET_MARGIN_REVENUE_SUM,
        "profitable_revenue_scale",
        "same_period",
        "entity_filter_aggregate",
        ("net_margin", "kqkd:10"),
        ("Biên lợi nhuận ròng", "Doanh thu thuần"),
        (("filter",), ("target",)),
        "kqkd:10",
        "Trong cohort có biên lợi nhuận ròng trên 10%, hỏi tổng doanh thu thuần của các công ty trong cohort đó.",
        (
            "Tổng chỉ cộng doanh thu thuần cùng năm, cùng report scope và cùng đơn vị tiền tệ.",
            "Điều kiện 10% dùng representation nội bộ 0.10 và phải được diễn đạt công khai bằng phần trăm.",
        ),
        threshold_operator=">",
        threshold_value=0.10,
        aggregate_operation="sum",
        required_question_terms=("10%", "tổng"),
    ),
    AnalyticalFrame(
        PERSISTENT_CASH_NET_MARGIN_MAX,
        "persistent_cash_profitability_quality",
        "cross_entity_period_window",
        "temporal_filter_aggregate",
        ("lctt:20", "net_margin"),
        ("Lưu chuyển tiền thuần từ hoạt động kinh doanh", "Biên lợi nhuận ròng"),
        (("filter",), ("target",)),
        "net_margin",
        "Trong các công ty duy trì dòng tiền kinh doanh dương ở mọi năm của window, hỏi biên lợi nhuận ròng cao nhất tại năm cuối window.",
        (
            "Điều kiện dòng tiền kinh doanh dương phải giữ ở mọi năm trong window, không chỉ năm đầu hoặc năm cuối.",
            "Biên lợi nhuận ròng chỉ được aggregate tại năm tham chiếu cuối window, không aggregate trên toàn panel nhiều năm.",
            "Không diễn giải dòng tiền dương liên tục là nguyên nhân tạo biên lợi nhuận ròng cao.",
        ),
        threshold_operator=">",
        threshold_value=0.0,
        aggregate_operation="maximum",
        required_question_terms=("dương",),
    ),
    AnalyticalFrame(
        PERSISTENT_PROFIT_REVENUE_SUM,
        "persistent_profit_revenue_scale",
        "cross_entity_period_window",
        "temporal_filter_aggregate",
        ("net_margin", "kqkd:10"),
        ("Biên lợi nhuận ròng", "Doanh thu thuần"),
        (("filter",), ("target",)),
        "kqkd:10",
        "Trong các công ty duy trì biên lợi nhuận ròng dương ở mọi năm của window, hỏi tổng doanh thu thuần tại năm cuối window.",
        (
            "Điều kiện biên lợi nhuận ròng dương phải giữ ở mọi năm trong window.",
            "Doanh thu thuần chỉ được cộng tại năm tham chiếu cuối window, không cộng across-year.",
            "Không kết luận lợi nhuận ròng dương liên tục là nguyên nhân tạo quy mô doanh thu.",
        ),
        finance_rationale="Tổng doanh thu năm tham chiếu của các công ty có lợi nhuận ròng dương liên tục đo quy mô nhóm vừa duy trì khả năng sinh lời vừa còn hoạt động ở cùng kỳ.",
        threshold_operator=">",
        threshold_value=0.0,
        aggregate_operation="sum",
        required_question_terms=("dương", "tổng"),
    ),
    AnalyticalFrame(
        PERSISTENT_PROFIT_LOW_LEVERAGE_QUICK_RATIO,
        "persistent_profit_low_leverage_liquidity",
        "cross_entity_period_window",
        "temporal_filter_rank_lookup",
        ("net_margin", "liabilities_to_equity", "quick_ratio"),
        (
            "Biên lợi nhuận ròng",
            "Hệ số nợ phải trả trên vốn chủ sở hữu",
            "Hệ số thanh toán nhanh",
        ),
        (("filter",), ("rank",), ("target",)),
        "quick_ratio",
        "Trong các công ty duy trì biên lợi nhuận ròng dương ở mọi năm của window, chọn công ty có hệ số nợ phải trả trên vốn chủ sở hữu thấp nhất tại năm cuối rồi hỏi hệ số thanh toán nhanh tại cùng năm.",
        (
            "Điều kiện biên lợi nhuận ròng dương phải giữ ở mọi năm trong window.",
            "Selector dùng hệ số nợ phải trả trên vốn chủ sở hữu tại năm tham chiếu, không phải D/E nợ vay.",
            "Hệ số thanh toán nhanh chỉ là chỉ báo thanh khoản ngắn hạn tại cùng năm, không chứng minh khả năng trả nợ chắc chắn.",
        ),
        threshold_operator=">",
        threshold_value=0.0,
        selector_direction="argmin",
        selector_transform="identity",
    ),
    AnalyticalFrame(
        PROFITABLE_YEARS_REVENUE_MIN,
        "single_company_profitable_revenue_floor",
        "single_entity_period_window",
        "period_filter_aggregate",
        ("net_margin", "kqkd:10"),
        ("Biên lợi nhuận ròng", "Doanh thu thuần"),
        (("filter",), ("target",)),
        "kqkd:10",
        "Trong một công ty qua nhiều năm liên tục, xét các năm có biên lợi nhuận ròng trên 10% rồi hỏi doanh thu thuần thấp nhất trong các năm đó.",
        (
            "Điều kiện biên lợi nhuận ròng được đánh giá theo từng năm trong cùng một công ty.",
            "Doanh thu thuần thấp nhất là giá trị numeric của năm được lọc, không trả về năm.",
            "Không cộng gộp balance snapshot hoặc tỷ số qua nhiều năm.",
        ),
        threshold_operator=">",
        threshold_value=0.10,
        aggregate_operation="minimum",
        required_question_terms=("10%", "thấp nhất"),
    ),
    AnalyticalFrame(
        PROFITABLE_LOW_REVENUE_CASH_FLOW_RATIO,
        "single_company_profitable_revenue_cash_conversion",
        "single_entity_period_window",
        "period_filter_rank_lookup",
        ("net_margin", "kqkd:10", "operating_cash_flow_ratio"),
        (
            "Biên lợi nhuận ròng",
            "Doanh thu thuần",
            "Hệ số dòng tiền hoạt động trên nợ ngắn hạn",
        ),
        (("filter",), ("rank",), ("target",)),
        "operating_cash_flow_ratio",
        "Trong một công ty qua nhiều năm liên tục, xét các năm có biên lợi nhuận ròng trên 10%, chọn năm có doanh thu thuần thấp nhất rồi hỏi hệ số dòng tiền hoạt động trên nợ ngắn hạn tại chính năm đó.",
        (
            "Điều kiện biên lợi nhuận ròng được đánh giá theo từng năm trong cùng một công ty.",
            "Năm có doanh thu thấp nhất chỉ là key trung gian để tra hệ số dòng tiền hoạt động, không phải đáp án.",
            "Hệ số dòng tiền hoạt động trên nợ ngắn hạn là tỷ số dòng tiền cả năm trên nợ ngắn hạn cuối kỳ, không chứng minh thanh khoản chắc chắn.",
        ),
        threshold_operator=">",
        threshold_value=0.10,
        selector_direction="argmin",
        selector_transform="identity",
        required_question_terms=("10%", "thấp nhất"),
    ),
    AnalyticalFrame(
        INVENTORY_DAYS_GROSS_MARGIN_CHANGE_LEADER,
        "inventory_cycle_margin_pressure",
        "cross_entity_period_window",
        "dual_transform_panel_rank_lookup",
        ("inventory_days", "gross_margin"),
        ("Số ngày tồn kho", "Biên lợi nhuận gộp"),
        (("rank",), ("target",)),
        "inventory_days_gross_margin_change",
        "Trong peer group, chọn doanh nghiệp có số ngày tồn kho tăng mạnh nhất từ năm áp chót sang năm cuối rồi hỏi biên lợi nhuận gộp của doanh nghiệp đó thay đổi bao nhiêu điểm phần trăm trong cùng giai đoạn.",
        (
            "Số ngày tồn kho của mỗi năm dùng hàng tồn kho bình quân đầu-cuối năm chia giá vốn hàng bán rồi nhân 365.",
            "Window ba năm là cần thiết để tính số ngày tồn kho cho hai năm cuối; phép xếp hạng chỉ so mức thay đổi giữa hai năm cuối.",
            "Không kết luận tồn kho tăng là nguyên nhân làm biên lợi nhuận gộp thay đổi.",
        ),
        template_id="A01",
        finance_rationale="Đặt biến động chu kỳ tồn kho cạnh biến động biên gộp giúp nhận diện doanh nghiệp có dấu hiệu tích hàng trong khi hiệu quả gộp thay đổi.",
        selector_direction="argmax",
        selector_transform="period_difference",
        target_transform="period_difference",
        transform_source_offset=1,
        required_period_count=3,
        require_positive_selector_value=True,
        required_question_terms=("tăng", "thay đổi"),
    ),
    AnalyticalFrame(
        REVENUE_GROWTH_SGA_GROWTH_LEADER,
        "revenue_growth_cost_discipline",
        "adjacent_period",
        "dual_transform_panel_rank_lookup",
        ("kqkd:10", "sga_expense"),
        (
            "Tăng trưởng doanh thu thuần",
            "Tăng trưởng tổng chi phí bán hàng và chi phí quản lý doanh nghiệp",
        ),
        (("rank",), ("target",)),
        "sga_expense_growth",
        "Giữa hai năm liền kề trong peer group, chọn doanh nghiệp có doanh thu thuần tăng trưởng cao nhất rồi hỏi tổng chi phí bán hàng và chi phí quản lý doanh nghiệp của chính doanh nghiệp đó tăng trưởng bao nhiêu phần trăm.",
        (
            "Tổng chi phí được cộng từ chi phí bán hàng và chi phí quản lý doanh nghiệp trong từng năm trước khi tính tăng trưởng.",
            "Chỉ so tốc độ tăng của hai đại lượng, không gọi đây là đòn bẩy hoạt động và không phân loại biến phí hay định phí.",
        ),
        finance_rationale="So tăng trưởng SG&A tại doanh nghiệp dẫn đầu tăng trưởng doanh thu giúp kiểm tra chi phí bán hàng và quản lý có phình nhanh cùng quy mô hay không.",
        selector_direction="argmax",
        selector_transform="growth",
        target_transform="growth",
        required_period_count=2,
        required_question_terms=("tăng trưởng",),
    ),
    AnalyticalFrame(
        CFO_OPERATING_PROFIT_NET_MARGIN_LOOKUP,
        "cash_conversion_profitability",
        "same_period",
        "rank_lookup",
        ("cfo_to_operating_profit", "net_margin"),
        (
            "Tỷ lệ dòng tiền từ hoạt động kinh doanh trên lợi nhuận thuần từ hoạt động kinh doanh",
            "Biên lợi nhuận ròng",
        ),
        (("rank",), ("target",)),
        "net_margin",
        "Trong peer group, chọn công ty có tỷ lệ dòng tiền từ hoạt động kinh doanh trên lợi nhuận thuần từ hoạt động kinh doanh thấp nhất rồi hỏi biên lợi nhuận ròng của công ty đó.",
        (
            "Chỉ xét công ty có lợi nhuận thuần từ hoạt động kinh doanh dương để tỷ lệ có mẫu số có ý nghĩa.",
            "Lợi nhuận thuần từ hoạt động kinh doanh theo mã kqkd:30 không được gọi là EBIT.",
            "Tỷ lệ thấp là tín hiệu mô tả chuyển đổi lợi nhuận hoạt động thành tiền, không tự chứng minh chất lượng lợi nhuận kém.",
        ),
        finance_rationale="Đặt mức chuyển đổi lợi nhuận hoạt động thành dòng tiền cạnh biên lợi nhuận ròng giúp phân biệt khả năng sinh lời kế toán và khả năng tạo tiền.",
        selector_direction="argmin",
    ),
    AnalyticalFrame(
        LEVERAGE_INTEREST_SHARE,
        "leverage_interest_burden",
        "same_period",
        "derived_threshold_share",
        ("liabilities_to_equity", "kqkd:23"),
        ("Hệ số nợ phải trả trên vốn chủ sở hữu", "Chi phí lãi vay"),
        (("filter",), ("target",)),
        "leverage_interest_expense_share",
        "Các công ty có hệ số nợ phải trả trên vốn chủ sở hữu cao hơn trung vị peer group chiếm bao nhiêu phần trăm tổng chi phí lãi vay của cả nhóm.",
        (
            "Không gọi Nợ phải trả/Vốn chủ sở hữu là D/E nợ vay và không kết luận quan hệ nhân quả.",
        ),
        threshold_operator=">",
    ),
    AnalyticalFrame(
        LEVERAGE_COVERAGE_LOOKUP,
        "leverage_interest_burden",
        "same_period",
        "rank_lookup",
        ("liabilities_to_equity", "interest_coverage"),
        ("Hệ số nợ phải trả trên vốn chủ sở hữu", "Hệ số khả năng thanh toán lãi vay"),
        (("rank",), ("target",)),
        "interest_coverage",
        "Trong peer group, chọn công ty có hệ số nợ phải trả trên vốn chủ sở hữu cao nhất rồi hỏi hệ số khả năng thanh toán lãi vay.",
        ("Không gọi selector là D/E nợ vay và không suy diễn xác suất phá sản.",),
        selector_direction="argmax",
    ),
    AnalyticalFrame(
        HIGH_LEVERAGE_COVERAGE_AVG,
        "leverage_interest_burden",
        "same_period",
        "derived_threshold_average",
        ("liabilities_to_equity", "interest_coverage"),
        ("Hệ số nợ phải trả trên vốn chủ sở hữu", "Hệ số khả năng thanh toán lãi vay"),
        (("filter",), ("target",)),
        "interest_coverage",
        "Trong cohort có hệ số nợ phải trả trên vốn chủ sở hữu cao hơn trung vị peer group, hỏi hệ số khả năng thanh toán lãi vay bình quân.",
        ("Đây là mô tả cohort tương đối, không suy diễn xác suất phá sản.",),
        threshold_operator=">",
        aggregate_operation="average",
    ),
)

FRAMES_BY_ID: dict[str, AnalyticalFrame] = {frame.frame_id: frame for frame in FRAMES}
_MANUAL_SIGNATURE_PRIORITY = (
    CASH_ROE_GAP,
    GROWTH_MARGIN_SQUEEZE_COUNT,
    PERSISTENT_CASH_GROWTH_AVG,
    PERSISTENT_CASH_GROWTH_ROA,
    LEVERAGE_INTEREST_SHARE,
)
ENABLED_ANALYTICAL_FRAME_IDS: tuple[str, ...] = _MANUAL_SIGNATURE_PRIORITY + tuple(
    frame.frame_id
    for frame in FRAMES
    if frame.frame_id not in _MANUAL_SIGNATURE_PRIORITY
)
ENABLED_TEMPLATE_FRAME_IDS: tuple[str, ...] = tuple(
    frame.frame_id
    for frame in FRAMES
    if frame.template_id is not None
    and INTENTS_BY_ID[frame.template_id].implementation_state
    is CapabilityState.RUNNABLE
)


def _signature(*parts: object) -> str:
    return hashlib.sha256("|".join(repr(p) for p in parts).encode()).hexdigest()[:16]


def _valid_terms(
    cube: Cube, ticker: str, periods: tuple[str, ...], period: str, metric_key: str
) -> MetricTerms | None:
    terms = resolve_window_metric_terms(cube, ticker, periods, period, metric_key)
    if terms is None or terms.value is None:
        return None
    positive_denominator_key = {
        "liabilities_to_equity": "cdkt:400",
        "gross_margin": "kqkd:10",
        "net_margin": "kqkd:10",
        "gross_to_net_margin_difference": "kqkd:10",
        "current_ratio": "cdkt:310",
        "inventory_to_current_liabilities": "cdkt:310",
        "quick_ratio": "cdkt:310",
        "operating_cash_flow_ratio": "cdkt:310",
        "npat_to_ending_assets": "cdkt:270",
        "npat_to_ending_equity": "cdkt:400",
        "cfo_to_operating_profit": "kqkd:30",
        "cfo_margin": "kqkd:10",
        "cfo_minus_net_margin": "kqkd:10",
        "inventory_to_assets": "cdkt:270",
        "liabilities_to_assets": "cdkt:270",
        "sga_intensity": "kqkd:10",
        "long_term_assets_share": "cdkt:270",
        "cfo_to_npat": "kqkd:60",
        "operating_profit_to_pbt": "kqkd:50",
    }.get(metric_key)
    if positive_denominator_key:
        cell = cube.cell(ticker, period, positive_denominator_key)
        if cell is None or cell.value <= 0:
            return None
    if metric_key == "inventory_to_current_liabilities":
        inventory = cube.cell(ticker, period, "cdkt:140")
        if inventory is None or inventory.value < 0:
            return None
    if metric_key == "quick_ratio":
        current_assets = cube.cell(ticker, period, "cdkt:100")
        inventory = cube.cell(ticker, period, "cdkt:140")
        if (
            current_assets is None
            or inventory is None
            or current_assets.value < inventory.value
        ):
            return None
    return terms


def _role(
    role_id: str, metric_key: str, bindings: Mapping, kind: ValueKind
) -> MetricRoleInput:
    return MetricRoleInput(role_id, metric_key, kind, bindings)


@dataclass(frozen=True, slots=True)
class _TopologyBuildContext:
    frame: AnalyticalFrame
    entities: tuple[str, ...]
    periods: tuple[str, ...]
    current: str
    entity_role: Callable[[int], MetricRoleInput]
    entity_role_at: Callable[[int, str], MetricRoleInput]
    panel_role: Callable[[int, tuple[str, ...] | None], MetricRoleInput]
    period_role: Callable[[int], MetricRoleInput]


def _build_sign_cohort_gap(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_sign_cohort_gap(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        period=ctx.current,
        periods=ctx.periods,
        cohort_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
    )


def _build_dual_panel_count(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_dual_panel_predicate_count(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        prior=ctx.periods[0],
        current=ctx.current,
        growth_role=ctx.panel_role(0, ctx.periods),
        difference_role=ctx.panel_role(1, ctx.periods),
    )


def _build_temporal_growth_average(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_temporal_growth_average(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        prior=ctx.periods[0],
        current=ctx.current,
        temporal_role=ctx.panel_role(0, ctx.periods),
        growth_role=ctx.panel_role(1, ctx.periods),
        threshold=ctx.frame.threshold_value or 0.0,
    )


def _build_temporal_growth_lookup(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_temporal_growth_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        prior=ctx.periods[0],
        current=ctx.current,
        temporal_role=ctx.panel_role(0, ctx.periods),
        growth_role=ctx.panel_role(1, ctx.periods),
        target_role=ctx.entity_role(2),
    )


def _build_panel_filter_average(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_panel_filter_average(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        prior=ctx.periods[0],
        current=ctx.current,
        growth_role=ctx.panel_role(0, ctx.periods),
        difference_role=ctx.panel_role(1, ctx.periods),
    )


def _build_entity_filter_aggregate_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    assert (
        ctx.frame.threshold_operator is not None
        and ctx.frame.threshold_value is not None
    )
    aggregate_operation = ctx.frame.aggregate_operation
    if aggregate_operation is None:
        raise CandidateRejected(
            f"{ctx.frame.frame_id}: entity_filter_aggregate is missing aggregate_operation"
        )
    _validate_entity_filter_aggregate_frame(ctx.frame, len(ctx.periods))
    return build_entity_filter_aggregate(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        period=ctx.current,
        filter_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
        operator=ctx.frame.threshold_operator,
        threshold=ctx.frame.threshold_value,
        aggregate_operation=aggregate_operation,
    )


def _build_dual_predicate_count(ctx: _TopologyBuildContext) -> ReasoningGraph:
    first_operator, second_operator = (
        (">", "<")
        if ctx.frame.frame_id == PROFITABLE_NEGATIVE_CASH_COUNT
        else (">", "<")
    )
    return build_dual_predicate_count(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        period=ctx.current,
        first_role=ctx.entity_role(0),
        first_operator=first_operator,
        second_role=ctx.entity_role(1),
        second_operator=second_operator,
    )


def _build_derived_threshold_share(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_derived_threshold_share(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        period=ctx.current,
        selector_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
    )


def _build_rank_lookup_topology(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_rank_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        period=ctx.current,
        rank_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
        selector_direction=ctx.frame.selector_direction or "argmax",
    )


def _build_derived_threshold_average_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_derived_threshold_average(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        report_scope=REPORT_SCOPE,
        period=ctx.current,
        selector_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
    )


def _build_panel_rank_lookup_topology(ctx: _TopologyBuildContext) -> ReasoningGraph:
    candidate_periods = (
        ctx.periods[1:]
        if ctx.frame.selector_transform in {"growth", "period_difference"}
        else ctx.periods
    )
    return build_panel_rank_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        rank_role=ctx.panel_role(0, ctx.periods),
        target_role=ctx.panel_role(1, candidate_periods),
        transform=ctx.frame.selector_transform or "identity",
        selector_direction=ctx.frame.selector_direction or "argmax",
    )


def _build_dual_transform_panel_rank_lookup_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    source_periods = ctx.periods[ctx.frame.transform_source_offset :]
    return build_dual_transform_panel_rank_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        graph_periods=ctx.periods,
        source_periods=source_periods,
        report_scope=REPORT_SCOPE,
        rank_role=ctx.panel_role(0, source_periods),
        target_role=ctx.panel_role(1, source_periods),
        rank_transform=ctx.frame.selector_transform or "growth",
        target_transform=ctx.frame.target_transform or "growth",
        selector_direction=ctx.frame.selector_direction or "argmax",
        require_positive_selector_value=ctx.frame.require_positive_selector_value,
    )


def _build_extreme_lookup_difference_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_extreme_lookup_difference(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        rank_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
        first_selector=ctx.frame.selector_direction or "argmax",
    )


def _build_entity_filter_rank_lookup_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    if ctx.frame.threshold_operator is None or ctx.frame.threshold_value is None:
        raise CandidateRejected(
            f"{ctx.frame.frame_id}: missing threshold for entity_filter_rank_lookup"
        )
    return build_entity_filter_rank_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.entity_role(0),
        rank_role=ctx.entity_role(1),
        target_role=ctx.entity_role(2),
        filter_operator=ctx.frame.threshold_operator,
        threshold=ctx.frame.threshold_value,
        selector_direction=ctx.frame.selector_direction or "argmax",
    )


def _build_transformed_filter_aggregate_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    if (
        ctx.frame.threshold_operator is None
        or ctx.frame.threshold_value is None
        or ctx.frame.aggregate_operation is None
    ):
        raise CandidateRejected(
            f"{ctx.frame.frame_id}: missing transformed_filter_aggregate configuration"
        )
    return build_transformed_filter_aggregate(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.panel_role(0, ctx.periods),
        target_role=ctx.entity_role(1),
        operator=ctx.frame.threshold_operator,
        threshold=ctx.frame.threshold_value,
        aggregate_operation=ctx.frame.aggregate_operation,
    )


def _build_median_split_ratio_topology(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_median_split_ratio(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        cohort_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
    )


def _build_ranked_cohort_topology(ctx: _TopologyBuildContext) -> ReasoningGraph:
    terminal_by_topology = {
        "ranked_cohort_gap": "gap",
        "ranked_cohort_share": "share",
        "ranked_cohort_average": "average",
    }
    return build_ranked_cohort_terminal(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        cohort_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
        percent=ctx.frame.threshold_value or 25.0,
        terminal=terminal_by_topology[ctx.frame.topology],
    )


def _build_lower_median_positive_share_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_lower_median_positive_share(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        cohort_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
    )


def _build_dual_transform_count_topology(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_dual_transform_count(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        first_role=ctx.panel_role(0, ctx.periods),
        second_role=ctx.panel_role(1, ctx.periods),
        first_transform=ctx.frame.selector_transform or "period_difference",
        second_transform=ctx.frame.target_transform or "period_difference",
        mode=ctx.frame.topology_mode or "positive_negative",
    )


def _build_filtered_dual_transform_lookup_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_filtered_dual_transform_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.panel_role(0, ctx.periods),
        rank_role=ctx.panel_role(1, ctx.periods),
        target_role=ctx.panel_role(2, ctx.periods),
    )


def _build_multi_transform_max_topology(ctx: _TopologyBuildContext) -> ReasoningGraph:
    transform_periods = ctx.periods[-2:]
    predicate_roles = tuple(
        ctx.panel_role(index, transform_periods)
        for index in range(len(ctx.frame.metric_keys) - 1)
    )
    return build_multi_transform_max(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        predicate_roles=predicate_roles,
        operators=ctx.frame.predicate_operators,
        target_role=ctx.entity_role(len(ctx.frame.metric_keys) - 1),
    )


def _build_temporal_filter_aggregate_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    assert (
        ctx.frame.threshold_operator is not None
        and ctx.frame.threshold_value is not None
    )
    aggregate_operation = ctx.frame.aggregate_operation
    if aggregate_operation is None:
        raise CandidateRejected(
            f"{ctx.frame.frame_id}: temporal_filter_aggregate is missing aggregate_operation"
        )
    _validate_entity_filter_aggregate_frame(ctx.frame, 1)
    return build_temporal_filter_aggregate(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        reference_period=ctx.current,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.panel_role(0, ctx.periods),
        target_role=ctx.entity_role(1),
        operator=ctx.frame.threshold_operator,
        threshold=ctx.frame.threshold_value,
        aggregate_operation=aggregate_operation,
    )


def _build_temporal_filter_rank_lookup_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    assert (
        ctx.frame.threshold_operator is not None
        and ctx.frame.threshold_value is not None
    )
    transform = ctx.frame.selector_transform or "identity"
    if transform == "identity":
        rank_role = ctx.entity_role(1)
    else:
        rank_role = ctx.panel_role(1, ctx.periods)
    return build_temporal_filter_rank_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        reference_period=ctx.current,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.panel_role(0, ctx.periods),
        rank_role=rank_role,
        target_role=ctx.entity_role(2),
        operator=ctx.frame.threshold_operator,
        threshold=ctx.frame.threshold_value,
        transform=transform,
        selector_direction=ctx.frame.selector_direction or "argmax",
    )


def _build_period_filter_aggregate_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    assert (
        ctx.frame.threshold_operator is not None
        and ctx.frame.threshold_value is not None
    )
    aggregate_operation = ctx.frame.aggregate_operation
    if aggregate_operation is None:
        raise CandidateRejected(
            f"{ctx.frame.frame_id}: period_filter_aggregate is missing aggregate_operation"
        )
    _validate_period_filter_aggregate_frame(ctx.frame)
    return build_period_filter_aggregate(
        recipe_id=ctx.frame.frame_id,
        entity=ctx.entities[0],
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.period_role(0),
        target_role=ctx.period_role(1),
        operator=ctx.frame.threshold_operator,
        threshold=ctx.frame.threshold_value,
        aggregate_operation=aggregate_operation,
    )


def _build_period_filter_rank_lookup_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    assert (
        ctx.frame.threshold_operator is not None
        and ctx.frame.threshold_value is not None
    )
    return build_period_filter_rank_lookup(
        recipe_id=ctx.frame.frame_id,
        entity=ctx.entities[0],
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.period_role(0),
        rank_role=ctx.period_role(1),
        target_role=ctx.period_role(2),
        operator=ctx.frame.threshold_operator,
        threshold=ctx.frame.threshold_value,
        selector_direction=ctx.frame.selector_direction or "argmax",
    )


def _role_for_transform(
    ctx: _TopologyBuildContext, index: int, transform: str, period: str
) -> MetricRoleInput:
    return (
        ctx.entity_role_at(index, period)
        if transform == "identity"
        else ctx.panel_role(index, ctx.periods[ctx.frame.transform_source_offset :])
    )


def _build_filtered_transform_rank_lookup_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    has_filter = len(ctx.frame.metric_keys) == 3
    filter_index = 0 if has_filter else None
    rank_index = 1 if has_filter else 0
    target_index = 2 if has_filter else 1
    filter_transform = ctx.frame.filter_transform or "identity"
    rank_transform = ctx.frame.selector_transform or "identity"
    target_transform = ctx.frame.target_transform or "identity"
    filter_period = ctx.periods[ctx.frame.filter_period_offset]
    rank_period = ctx.periods[ctx.frame.rank_period_offset]
    target_period = ctx.periods[ctx.frame.target_period_offset]
    transform_periods = ctx.periods[ctx.frame.transform_source_offset :]
    return build_filtered_transform_rank_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        filter_role=(
            _role_for_transform(ctx, filter_index, filter_transform, filter_period)
            if filter_index is not None
            else None
        ),
        rank_role=_role_for_transform(ctx, rank_index, rank_transform, rank_period),
        target_role=_role_for_transform(
            ctx, target_index, target_transform, target_period
        ),
        filter_transform=filter_transform,
        rank_transform=rank_transform,
        target_transform=target_transform,
        filter_operator=ctx.frame.threshold_operator or ">",
        filter_threshold=ctx.frame.threshold_value or 0.0,
        selector_direction=ctx.frame.selector_direction or "argmax",
        filter_period=filter_period,
        rank_period=rank_period,
        target_period=target_period,
        transform_periods=transform_periods,
    )


def _build_filter_share_topology(ctx: _TopologyBuildContext) -> ReasoningGraph:
    return build_filter_share(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.entity_role(0),
        target_role=ctx.entity_role(1),
        operator=ctx.frame.threshold_operator or ">",
        threshold=ctx.frame.threshold_value or 0.0,
    )


def _build_transform_rank_share_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_transform_rank_share(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        rank_role=ctx.panel_role(0, ctx.periods),
        target_role=ctx.entity_role(1) if len(ctx.frame.metric_keys) > 1 else None,
        selector_direction=ctx.frame.selector_direction or "argmax",
        positive_only=ctx.frame.topology_mode == "positive_only",
    )


def _build_top_n_dual_predicate_count_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    thresholds = ctx.frame.predicate_thresholds or (1.0, 1.0)
    operators = ctx.frame.predicate_operators or (">", "<")
    return build_top_n_dual_predicate_count(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        rank_role=ctx.entity_role(0),
        first_role=ctx.entity_role(1),
        second_role=ctx.entity_role(2),
        n=ctx.frame.top_n or 3,
        first_operator=operators[0],
        first_threshold=thresholds[0],
        second_operator=operators[1],
        second_threshold=thresholds[1],
    )


def _build_nested_median_aggregate_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_nested_median_aggregate(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        first_role=ctx.entity_role(0),
        second_role=ctx.entity_role(1),
        target_role=ctx.entity_role(2),
        first_operator=ctx.frame.threshold_operator or "<",
        first_threshold=ctx.frame.threshold_value or 0.0,
        second_operator=(ctx.frame.predicate_operators or ("<",))[0],
        aggregate_operation=ctx.frame.aggregate_operation or "maximum",
    )


def _build_multi_predicate_terminal_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    predicate_count = len(ctx.frame.predicate_operators)
    transforms = ctx.frame.predicate_transforms or ("identity",) * predicate_count
    roles = tuple(
        _role_for_transform(ctx, index, transforms[index], ctx.current)
        for index in range(predicate_count)
    )
    has_target = len(ctx.frame.metric_keys) > predicate_count
    return build_multi_predicate_terminal(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        predicate_roles=roles,
        predicate_transforms=transforms,
        predicate_operators=ctx.frame.predicate_operators,
        predicate_thresholds=ctx.frame.predicate_thresholds,
        target_role=ctx.entity_role(predicate_count) if has_target else None,
        aggregate_operation=ctx.frame.aggregate_operation or "set_count",
    )


def _build_dol_terminal_topology(ctx: _TopologyBuildContext) -> ReasoningGraph:
    has_size = ctx.frame.topology_mode == "with_size"
    revenue_index = 1 if has_size else 0
    profit_index = revenue_index + 1
    target_index = profit_index + 1
    has_target = len(ctx.frame.metric_keys) > target_index
    return build_dol_terminal(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        size_role=ctx.entity_role(0) if has_size else None,
        size_threshold=ctx.frame.threshold_value if has_size else None,
        revenue_role=ctx.panel_role(revenue_index, ctx.periods),
        operating_profit_role=ctx.panel_role(profit_index, ctx.periods),
        revenue_growth_threshold=(
            ctx.frame.secondary_threshold_value
            if has_size
            else ctx.frame.threshold_value or 0.0
        ),
        target_role=ctx.entity_role(target_index) if has_target else None,
        selector_direction=ctx.frame.selector_direction or "argmax",
    )


def _build_transformed_filter_aggregate_terminal_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_transformed_filter_aggregate_terminal(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        filter_role=ctx.panel_role(0, ctx.periods[ctx.frame.transform_source_offset :]),
        target_role=ctx.panel_role(1, ctx.periods[ctx.frame.transform_source_offset :]),
        filter_transform=ctx.frame.filter_transform or "growth",
        target_transform=ctx.frame.target_transform or "period_difference",
        filter_operator=ctx.frame.threshold_operator or ">",
        filter_threshold=ctx.frame.threshold_value or 0.0,
        aggregate_operation=ctx.frame.aggregate_operation or "minimum",
        absolute_terminal=ctx.frame.absolute_terminal,
        transform_periods=ctx.periods[ctx.frame.transform_source_offset :],
    )


def _build_filtered_ranked_cohort_gap_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_filtered_ranked_cohort_gap(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        period=ctx.current,
        report_scope=REPORT_SCOPE,
        size_role=ctx.entity_role(0),
        cohort_role=ctx.entity_role(1),
        target_role=ctx.entity_role(2),
        size_threshold=ctx.frame.threshold_value or 0.0,
        percent=ctx.frame.quantile_percent or 25.0,
    )


def _build_transition_filter_rank_lookup_topology(
    ctx: _TopologyBuildContext,
) -> ReasoningGraph:
    return build_transition_filter_rank_lookup(
        recipe_id=ctx.frame.frame_id,
        entities=ctx.entities,
        periods=ctx.periods,
        report_scope=REPORT_SCOPE,
        transition_role=ctx.panel_role(0, ctx.periods),
        size_role=ctx.entity_role(1),
        rank_role=ctx.entity_role(2),
        target_role=ctx.entity_role(3),
        size_threshold=ctx.frame.threshold_value or 0.0,
        selector_direction=ctx.frame.selector_direction or "argmax",
    )


_TOPOLOGY_BUILDERS: dict[
    Topology, Callable[[_TopologyBuildContext], ReasoningGraph]
] = {
    "sign_cohort_gap": _build_sign_cohort_gap,
    "dual_panel_count": _build_dual_panel_count,
    "temporal_growth_average": _build_temporal_growth_average,
    "temporal_growth_lookup": _build_temporal_growth_lookup,
    "panel_filter_average": _build_panel_filter_average,
    "entity_filter_aggregate": _build_entity_filter_aggregate_topology,
    "dual_predicate_count": _build_dual_predicate_count,
    "derived_threshold_share": _build_derived_threshold_share,
    "rank_lookup": _build_rank_lookup_topology,
    "derived_threshold_average": _build_derived_threshold_average_topology,
    "panel_rank_lookup": _build_panel_rank_lookup_topology,
    "dual_transform_panel_rank_lookup": _build_dual_transform_panel_rank_lookup_topology,
    "extreme_lookup_difference": _build_extreme_lookup_difference_topology,
    "entity_filter_rank_lookup": _build_entity_filter_rank_lookup_topology,
    "transformed_filter_aggregate": _build_transformed_filter_aggregate_topology,
    "median_split_ratio": _build_median_split_ratio_topology,
    "ranked_cohort_gap": _build_ranked_cohort_topology,
    "ranked_cohort_share": _build_ranked_cohort_topology,
    "ranked_cohort_average": _build_ranked_cohort_topology,
    "lower_median_positive_share": _build_lower_median_positive_share_topology,
    "dual_transform_count": _build_dual_transform_count_topology,
    "filtered_dual_transform_lookup": _build_filtered_dual_transform_lookup_topology,
    "multi_transform_max": _build_multi_transform_max_topology,
    "temporal_filter_aggregate": _build_temporal_filter_aggregate_topology,
    "temporal_filter_rank_lookup": _build_temporal_filter_rank_lookup_topology,
    "period_filter_aggregate": _build_period_filter_aggregate_topology,
    "period_filter_rank_lookup": _build_period_filter_rank_lookup_topology,
    "filtered_transform_rank_lookup": _build_filtered_transform_rank_lookup_topology,
    "filter_share": _build_filter_share_topology,
    "transform_rank_share": _build_transform_rank_share_topology,
    "top_n_dual_predicate_count": _build_top_n_dual_predicate_count_topology,
    "nested_median_aggregate": _build_nested_median_aggregate_topology,
    "multi_predicate_terminal": _build_multi_predicate_terminal_topology,
    "dol_terminal": _build_dol_terminal_topology,
    "transformed_filter_aggregate_terminal": _build_transformed_filter_aggregate_terminal_topology,
    "filtered_ranked_cohort_gap": _build_filtered_ranked_cohort_gap_topology,
    "transition_filter_rank_lookup": _build_transition_filter_rank_lookup_topology,
}


def _validate_entity_filter_aggregate_frame(
    frame: AnalyticalFrame, period_count: int
) -> None:
    aggregate_operation = frame.aggregate_operation
    if aggregate_operation not in AGGREGATE_OPERATIONS:
        raise CandidateRejected(
            f"{frame.frame_id}: aggregate_operation must be one of {sorted(AGGREGATE_OPERATIONS)}, got {aggregate_operation!r}"
        )
    if aggregate_operation != "sum":
        return
    target_metric = frame.metric_keys[1]
    if period_count != 1:
        raise CandidateRejected(
            f"{frame.frame_id}: sum is valid only within one period; got {period_count} periods"
        )
    if get_ratio(target_metric) is not None or target_metric in {
        "roa",
        "roe",
        EQ_01_TERMINAL_KEY,
    }:
        raise CandidateRejected(
            f"{frame.frame_id}: ratio or percentage targets cannot be summed {target_metric!r}"
        )
    if target_metric not in RAW_MONEY_FLOW_METRICS:
        raise CandidateRejected(
            f"{frame.frame_id}: sum is valid only for raw monetary flows, not {target_metric!r}"
        )


def _validate_period_filter_aggregate_frame(frame: AnalyticalFrame) -> None:
    aggregate_operation = frame.aggregate_operation
    if aggregate_operation not in AGGREGATE_OPERATIONS:
        raise CandidateRejected(
            f"{frame.frame_id}: aggregate_operation must be one of {sorted(AGGREGATE_OPERATIONS)}, got {aggregate_operation!r}"
        )
    if aggregate_operation != "sum":
        return
    target_metric = frame.metric_keys[1]
    if get_ratio(target_metric) is not None or target_metric in {
        "roa",
        "roe",
        EQ_01_TERMINAL_KEY,
    }:
        raise CandidateRejected(
            f"{frame.frame_id}: ratio or percentage targets cannot be summed {target_metric!r}"
        )
    if target_metric not in RAW_MONEY_FLOW_METRICS:
        raise CandidateRejected(
            f"{frame.frame_id}: multi-year sum is valid only for raw monetary flows, not {target_metric!r}"
        )


def _build_attempt(
    cube: Cube,
    frame: AnalyticalFrame,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
) -> tuple[tuple[str, ...], ReasoningGraph] | None:
    current = periods[-1]
    if (
        frame.required_period_count is not None
        and len(periods) != frame.required_period_count
    ):
        return None

    eligible: list[str] = []
    cached: dict[tuple[str, str, str], MetricTerms] = {}
    if frame.topology == "filtered_transform_rank_lookup":
        has_filter = len(frame.metric_keys) == 3
        transforms = (
            (
                frame.filter_transform or "identity",
                frame.selector_transform or "identity",
                frame.target_transform or "identity",
            )
            if has_filter
            else (
                frame.selector_transform or "identity",
                frame.target_transform or "identity",
            )
        )
        offsets = (
            (
                frame.filter_period_offset,
                frame.rank_period_offset,
                frame.target_period_offset,
            )
            if has_filter
            else (frame.rank_period_offset, frame.target_period_offset)
        )
        for ticker in entities:
            valid = True
            for metric_key, transform, offset in zip(
                frame.metric_keys, transforms, offsets, strict=True
            ):
                role_periods = (
                    periods[frame.transform_source_offset :]
                    if transform != "identity"
                    else (periods[offset],)
                )
                for period in role_periods:
                    terms = _valid_terms(cube, ticker, periods, period, metric_key)
                    if terms is None:
                        valid = False
                        break
                    cached[(ticker, period, metric_key)] = terms
                if not valid:
                    break
            if valid:
                eligible.append(ticker)
    elif frame.topology == "transform_rank_share":
        for ticker in entities:
            valid = True
            for period in periods:
                terms = _valid_terms(
                    cube, ticker, periods, period, frame.metric_keys[0]
                )
                if terms is None:
                    valid = False
                    break
                cached[(ticker, period, frame.metric_keys[0])] = terms
            if valid and len(frame.metric_keys) > 1:
                terms = _valid_terms(
                    cube, ticker, periods, current, frame.metric_keys[1]
                )
                if terms is None:
                    valid = False
                else:
                    cached[(ticker, current, frame.metric_keys[1])] = terms
            if valid:
                eligible.append(ticker)
    elif frame.topology == "multi_predicate_terminal":
        predicate_count = len(frame.predicate_operators)
        transforms = frame.predicate_transforms or ("identity",) * predicate_count
        for ticker in entities:
            valid = True
            for index, transform in enumerate(transforms):
                metric_key = frame.metric_keys[index]
                for period in periods if transform != "identity" else (current,):
                    terms = _valid_terms(cube, ticker, periods, period, metric_key)
                    if terms is None:
                        valid = False
                        break
                    cached[(ticker, period, metric_key)] = terms
                if not valid:
                    break
            if valid and len(frame.metric_keys) > predicate_count:
                target_key = frame.metric_keys[predicate_count]
                terms = _valid_terms(cube, ticker, periods, current, target_key)
                if terms is None:
                    valid = False
                else:
                    cached[(ticker, current, target_key)] = terms
            if valid:
                eligible.append(ticker)
    elif frame.topology == "dol_terminal":
        has_size = frame.topology_mode == "with_size"
        revenue_index = 1 if has_size else 0
        profit_index = revenue_index + 1
        target_index = profit_index + 1
        for ticker in entities:
            valid = True
            if has_size:
                terms = _valid_terms(
                    cube, ticker, periods, current, frame.metric_keys[0]
                )
                if terms is None:
                    valid = False
                else:
                    cached[(ticker, current, frame.metric_keys[0])] = terms
            for index in (revenue_index, profit_index):
                if not valid:
                    break
                for period in periods:
                    terms = _valid_terms(
                        cube, ticker, periods, period, frame.metric_keys[index]
                    )
                    if terms is None:
                        valid = False
                        break
                    cached[(ticker, period, frame.metric_keys[index])] = terms
            if valid and len(frame.metric_keys) > target_index:
                terms = _valid_terms(
                    cube, ticker, periods, current, frame.metric_keys[target_index]
                )
                if terms is None:
                    valid = False
                else:
                    cached[(ticker, current, frame.metric_keys[target_index])] = terms
            if valid:
                eligible.append(ticker)
    elif frame.topology == "transformed_filter_aggregate_terminal":
        for ticker in entities:
            valid = True
            for metric_key in frame.metric_keys:
                for period in periods[frame.transform_source_offset :]:
                    terms = _valid_terms(cube, ticker, periods, period, metric_key)
                    if terms is None:
                        valid = False
                        break
                    cached[(ticker, period, metric_key)] = terms
                if not valid:
                    break
            if valid:
                eligible.append(ticker)
    elif frame.topology == "transition_filter_rank_lookup":
        for ticker in entities:
            valid = True
            for period in periods:
                terms = _valid_terms(
                    cube, ticker, periods, period, frame.metric_keys[0]
                )
                if terms is None:
                    valid = False
                    break
                cached[(ticker, period, frame.metric_keys[0])] = terms
            for metric_key in frame.metric_keys[1:]:
                if not valid:
                    break
                terms = _valid_terms(cube, ticker, periods, current, metric_key)
                if terms is None:
                    valid = False
                    break
                cached[(ticker, current, metric_key)] = terms
            if valid:
                eligible.append(ticker)
    elif frame.topology == "dual_transform_panel_rank_lookup":
        source_periods = periods[frame.transform_source_offset :]
        if len(source_periods) < 2:
            return None
        for ticker in entities:
            for period in source_periods:
                for metric_key in frame.metric_keys:
                    terms = _valid_terms(cube, ticker, periods, period, metric_key)
                    if terms is None:
                        return None
                    cached[(ticker, period, metric_key)] = terms
        eligible = list(entities)
    elif frame.topology == "panel_rank_lookup":
        if len(periods) < 2:
            return None
        transform = frame.selector_transform or "identity"
        candidate_periods = (
            periods[1:] if transform in {"growth", "period_difference"} else periods
        )
        for ticker in entities:
            for period in periods:
                terms = _valid_terms(
                    cube, ticker, periods, period, frame.metric_keys[0]
                )
                if terms is None:
                    return None
                cached[(ticker, period, frame.metric_keys[0])] = terms
            for period in candidate_periods:
                terms = _valid_terms(
                    cube, ticker, periods, period, frame.metric_keys[1]
                )
                if terms is None:
                    return None
                cached[(ticker, period, frame.metric_keys[1])] = terms
        eligible = list(entities)
    elif frame.topology == "temporal_filter_aggregate":
        _validate_entity_filter_aggregate_frame(frame, 1)
        for ticker in entities:
            for period in periods:
                terms = _valid_terms(
                    cube, ticker, periods, period, frame.metric_keys[0]
                )
                if terms is None:
                    return None
                cached[(ticker, period, frame.metric_keys[0])] = terms
            terms = _valid_terms(cube, ticker, periods, current, frame.metric_keys[1])
            if terms is None:
                return None
            cached[(ticker, current, frame.metric_keys[1])] = terms
        eligible = list(entities)
    elif frame.topology == "temporal_filter_rank_lookup":
        if len(periods) < 2:
            return None
        transform = frame.selector_transform or "identity"
        if transform not in {"identity", "growth", "period_difference"}:
            return None
        for ticker in entities:
            for period in periods:
                terms = _valid_terms(
                    cube, ticker, periods, period, frame.metric_keys[0]
                )
                if terms is None:
                    return None
                cached[(ticker, period, frame.metric_keys[0])] = terms
            if transform == "identity":
                terms = _valid_terms(
                    cube, ticker, periods, current, frame.metric_keys[1]
                )
                if terms is None:
                    return None
                cached[(ticker, current, frame.metric_keys[1])] = terms
            else:
                for period in periods:
                    terms = _valid_terms(
                        cube, ticker, periods, period, frame.metric_keys[1]
                    )
                    if terms is None:
                        return None
                    cached[(ticker, period, frame.metric_keys[1])] = terms
            terms = _valid_terms(cube, ticker, periods, current, frame.metric_keys[2])
            if terms is None:
                return None
            cached[(ticker, current, frame.metric_keys[2])] = terms
        eligible = list(entities)
    elif frame.topology == "period_filter_aggregate":
        if len(entities) != 1:
            return None
        _validate_period_filter_aggregate_frame(frame)
        ticker = entities[0]
        for period in periods:
            for metric_key in frame.metric_keys:
                terms = _valid_terms(cube, ticker, periods, period, metric_key)
                if terms is None:
                    return None
                cached[(ticker, period, metric_key)] = terms
        eligible = [ticker]
    elif frame.topology == "period_filter_rank_lookup":
        if len(entities) != 1:
            return None
        ticker = entities[0]
        for period in periods:
            for metric_key in frame.metric_keys:
                terms = _valid_terms(cube, ticker, periods, period, metric_key)
                if terms is None:
                    return None
                cached[(ticker, period, metric_key)] = terms
        eligible = [ticker]
    elif frame.topology in {"dual_transform_count", "filtered_dual_transform_lookup"}:
        if len(periods) < 2:
            return None
        for ticker in entities:
            valid = True
            for period in periods:
                for metric_key in frame.metric_keys:
                    terms = _valid_terms(cube, ticker, periods, period, metric_key)
                    if terms is None:
                        valid = False
                        break
                    cached[(ticker, period, metric_key)] = terms
                if not valid:
                    break
            if valid:
                eligible.append(ticker)
    elif frame.topology == "multi_transform_max":
        if len(periods) < 3 or not frame.predicate_operators:
            return None
        transform_periods = periods[-2:]
        target_key = frame.metric_keys[-1]
        for ticker in entities:
            valid = True
            for metric_key in frame.metric_keys[:-1]:
                for period in transform_periods:
                    terms = _valid_terms(cube, ticker, periods, period, metric_key)
                    if terms is None:
                        valid = False
                        break
                    cached[(ticker, period, metric_key)] = terms
                if not valid:
                    break
            if valid:
                terms = _valid_terms(cube, ticker, periods, current, target_key)
                if terms is None:
                    valid = False
                else:
                    cached[(ticker, current, target_key)] = terms
            if valid:
                eligible.append(ticker)
    elif frame.topology == "transformed_filter_aggregate":
        if len(periods) != 2:
            return None
        filter_key, target_key = frame.metric_keys
        for ticker in entities:
            valid = True
            for period in periods:
                terms = _valid_terms(cube, ticker, periods, period, filter_key)
                if terms is None:
                    valid = False
                    break
                cached[(ticker, period, filter_key)] = terms
            if valid:
                terms = _valid_terms(cube, ticker, periods, current, target_key)
                if terms is None:
                    valid = False
                else:
                    cached[(ticker, current, target_key)] = terms
            if valid:
                eligible.append(ticker)
    else:
        for ticker in entities:
            valid = True
            for metric_key in frame.metric_keys:
                if metric_key == EQ_01_TERMINAL_KEY:
                    if len(periods) != 2:
                        valid = False
                        break
                    terms = resolve_operating_accruals_terms(
                        cube, ticker, periods[0], current
                    )
                    if terms is None or terms.value is None:
                        valid = False
                        break
                    denominator = sum(
                        term.coefficient * term.cell.value for term in terms.denominator
                    )
                    if denominator <= 0:
                        valid = False
                        break
                    cached[(ticker, current, metric_key)] = terms
                    continue
                needs_panel = (
                    frame.topology
                    in {
                        "dual_panel_count",
                        "temporal_growth_average",
                        "temporal_growth_lookup",
                        "panel_filter_average",
                    }
                    and metric_key in frame.metric_keys[:2]
                )
                for period in periods if needs_panel else (current,):
                    terms = _valid_terms(cube, ticker, periods, period, metric_key)
                    if terms is None:
                        valid = False
                        break
                    cached[(ticker, period, metric_key)] = terms
                if not valid:
                    break
            if valid:
                eligible.append(ticker)
    selected = tuple(sorted(eligible))
    if frame.domain_kind != "single_entity_period_window" and len(selected) < 3:
        return None
    if frame.domain_kind == "single_entity_period_window" and len(selected) != 1:
        return None

    def entity_role(index: int) -> MetricRoleInput:
        key = frame.metric_keys[index]
        return _role(
            f"role_{index}",
            key,
            {t: cached[(t, current, key)] for t in selected},
            ValueKind.NUMERIC_SERIES_ENTITY,
        )

    def entity_role_at(index: int, period: str) -> MetricRoleInput:
        key = frame.metric_keys[index]
        return _role(
            f"role_{index}",
            key,
            {t: cached[(t, period, key)] for t in selected},
            ValueKind.NUMERIC_SERIES_ENTITY,
        )

    def panel_role(
        index: int, role_periods: tuple[str, ...] | None = None
    ) -> MetricRoleInput:
        key = frame.metric_keys[index]
        target_periods = role_periods or periods
        return _role(
            f"role_{index}",
            key,
            {(t, p): cached[(t, p, key)] for t in selected for p in target_periods},
            ValueKind.NUMERIC_SERIES_ENTITY_PERIOD,
        )

    def period_role(index: int) -> MetricRoleInput:
        key = frame.metric_keys[index]
        ticker = selected[0]
        return _role(
            f"role_{index}",
            key,
            {p: cached[(ticker, p, key)] for p in periods},
            ValueKind.NUMERIC_SERIES_PERIOD,
        )

    context = _TopologyBuildContext(
        frame=frame,
        entities=selected,
        periods=periods,
        current=current,
        entity_role=entity_role,
        entity_role_at=entity_role_at,
        panel_role=panel_role,
        period_role=period_role,
    )
    graph = _TOPOLOGY_BUILDERS[frame.topology](context)
    return selected, graph


def _public_spec(
    frame: AnalyticalFrame,
    *,
    candidate_id: str,
    entities: tuple[str, ...],
    periods: tuple[str, ...],
    company_meta: dict[str, CompanyInfo],
) -> PublicSpec:
    public_periods = periods[frame.transform_source_offset :] or periods
    labels = {chr(65 + i): label for i, label in enumerate(frame.metric_labels)}
    public_metric_keys = frame.public_metric_keys or frame.metric_keys
    keys = {chr(65 + i): key for i, key in enumerate(public_metric_keys)}
    roles = {chr(65 + i): role for i, role in enumerate(frame.metric_roles)}
    target_value_kind = transformed_terminal_value_kind(
        frame.terminal_key, frame.target_transform
    )
    target_unit = frame.target_unit_label or unit_label(target_value_kind)
    intent = INTENTS_BY_ID.get(frame.template_id or "")
    industry_name = None
    if entities:
        info = company_meta.get(entities[0])
        industry_name = info.industry_l3 if info is not None else None
    predicates, cohorts, terminal_branches = _template_public_contract(frame)
    required_question_terms = list(frame.required_question_terms)
    if frame.topology == "filtered_transform_rank_lookup":
        filter_period = periods[frame.filter_period_offset]
        rank_period = periods[frame.rank_period_offset]
        target_period = periods[frame.target_period_offset]
        if len({filter_period, rank_period, target_period}) > 1:
            has_filter = len(frame.metric_labels) == 3
            filter_index = 0 if has_filter else None
            rank_index = 1 if has_filter else 0
            if filter_index is not None:
                required_question_terms.append(
                    f"{frame.metric_labels[filter_index]} năm {filter_period}"
                )
            required_question_terms.append(
                f"{frame.metric_labels[rank_index]} năm {rank_period}"
            )
            if frame.target_transform in {"growth", "period_difference"}:
                target_index = periods.index(target_period)
                if target_index == 0:
                    raise ValueError(
                        f"{frame.frame_id}: transformed target lacks a prior period"
                    )
                required_question_terms.append(
                    f"năm {target_period} so với năm {periods[target_index - 1]}"
                )
    return PublicSpec(
        recipe_id=frame.frame_id,
        candidate_id=candidate_id,
        entities=entities,
        company_names=company_names(entities, company_meta),
        periods=public_periods,
        reference_period=(
            None
            if frame.require_explicit_period_span
            else public_periods[-1]
            if len(public_periods) == 1
            or frame.topology
            in {"temporal_filter_aggregate", "temporal_filter_rank_lookup"}
            or (
                frame.template_id is not None
                and frame.topology
                not in {
                    "dual_transform_count",
                    "temporal_growth_average",
                    "filtered_dual_transform_lookup",
                }
            )
            else None
        ),
        metric_labels=labels,
        metric_keys=keys,
        metric_roles=roles,
        threshold_operator=frame.threshold_operator,
        threshold_source=(
            "derived"
            if frame.topology
            in {"derived_threshold_share", "derived_threshold_average"}
            else "convention"
            if frame.threshold_operator is not None
            else None
        ),
        threshold_statistic=(
            "median"
            if frame.topology
            in {"derived_threshold_share", "derived_threshold_average"}
            else None
        ),
        threshold_convention_value=frame.threshold_value,
        selector_direction=frame.selector_direction,
        aggregate_operation=frame.aggregate_operation,
        target_value_kind=target_value_kind,
        target_unit_label=target_unit,
        intent_id=None,
        analytical_frame=frame.meaning,
        story_family=frame.story_family,
        filter_transform=(
            frame.filter_transform
            if frame.filter_transform is not None
            else "growth"
            if frame.topology == "transformed_filter_aggregate"
            else "period_difference"
            if frame.topology == "multi_transform_max"
            else "temporal_all"
            if frame.topology
            in {"temporal_filter_aggregate", "temporal_filter_rank_lookup"}
            else None
        ),
        rank_transform=frame.selector_transform,
        target_transform=frame.target_transform,
        terminal_measurement_name=frame.metric_labels[-1],
        interpretation_limits=frame.interpretation_limits,
        required_question_terms=tuple(dict.fromkeys(required_question_terms)),
        template_id=frame.template_id,
        universe_kind=intent.universe_kind.value if intent is not None else None,
        universe_description=(
            "tập doanh nghiệp phi tài chính có đủ dữ liệu trong corpus"
            if intent is not None
            and intent.universe_kind.value.startswith("nonfinancial")
            else f"toàn bộ doanh nghiệp phi tài chính ngành {industry_name} có đủ dữ liệu"
            if intent is not None and industry_name
            else None
        ),
        operation_sequence=intent.operator_sequence if intent is not None else (),
        predicates=predicates,
        cohorts=cohorts,
        terminal_branches=terminal_branches,
        tie_policy="reject_ties" if frame.selector_direction else None,
    )


def _template_public_contract(
    frame: AnalyticalFrame,
) -> tuple[
    tuple[PublicPredicate, ...],
    tuple[PublicCohort, ...],
    tuple[PublicTerminalBranch, ...],
]:
    if frame.template_id is None:
        return (), (), ()
    filter_roles = [
        chr(65 + index)
        for index, roles in enumerate(frame.metric_roles)
        if "filter" in roles
    ]
    predicates: list[PublicPredicate] = []
    cohorts: list[PublicCohort] = []

    if (
        frame.topology == "dual_transform_count"
        and frame.topology_mode == "first_above_second"
    ):
        predicates.append(
            PublicPredicate(
                "relation",
                "A",
                ">",
                "relation",
                related_metric_role="B",
                transform=(frame.predicate_transforms or (None,))[0],
            )
        )
        cohorts.append(PublicCohort("matched", "predicate", ("A", "B"), ("relation",)))
    elif frame.topology == "dual_transform_count":
        predicates.extend(
            (
                PublicPredicate(
                    "positive",
                    "A",
                    ">",
                    "zero",
                    transform=(frame.predicate_transforms or (None, None))[0],
                ),
                PublicPredicate(
                    "negative",
                    "B",
                    "<",
                    "zero",
                    transform=(frame.predicate_transforms or (None, None))[1],
                ),
            )
        )
        cohorts.append(
            PublicCohort(
                "matched", "intersection", ("A", "B"), ("positive", "negative")
            )
        )
    elif frame.topology == "multi_transform_max":
        for index, operator in enumerate(frame.predicate_operators):
            predicates.append(
                PublicPredicate(
                    f"predicate_{index}",
                    chr(65 + index),
                    operator,
                    "zero",
                    transform=(
                        frame.predicate_transforms[index]
                        if index < len(frame.predicate_transforms)
                        else None
                    ),
                )
            )
        cohorts.append(
            PublicCohort(
                "selected",
                "intersection",
                tuple(filter_roles),
                tuple(p.predicate_id for p in predicates),
            )
        )
    elif frame.topology == "median_split_ratio":
        predicates.extend(
            (
                PublicPredicate(
                    "above_median", "A", ">", "derived", statistic="median"
                ),
                PublicPredicate(
                    "at_or_below_median", "A", "<=", "derived", statistic="median"
                ),
            )
        )
        cohorts.extend(
            (
                PublicCohort(
                    "high",
                    "median_split",
                    ("A",),
                    ("above_median",),
                    equality_policy="strictly_above",
                ),
                PublicCohort(
                    "low",
                    "median_split",
                    ("A",),
                    ("at_or_below_median",),
                    equality_policy="at_or_below",
                ),
            )
        )
    elif frame.topology == "lower_median_positive_share":
        predicates.extend(
            (
                PublicPredicate(
                    "below_median", "A", "<", "derived", statistic="median"
                ),
                PublicPredicate("positive_target", "B", ">", "zero"),
            )
        )
        cohorts.extend(
            (
                PublicCohort(
                    "low",
                    "median_split",
                    ("A",),
                    ("below_median",),
                    equality_policy="strictly_below",
                ),
                PublicCohort("positive", "predicate", ("B",), ("positive_target",)),
                PublicCohort(
                    "selected",
                    "intersection",
                    ("A", "B"),
                    ("below_median", "positive_target"),
                ),
            )
        )
    elif frame.topology == "filtered_ranked_cohort_gap":
        operator = frame.threshold_operator or ">"
        predicates.append(
            PublicPredicate(
                "size_filter",
                "A",
                operator,
                "fixed",
                threshold_value=frame.threshold_value,
                transform=frame.filter_transform,
            )
        )
        cohorts.extend(
            (
                PublicCohort(
                    "top",
                    "top_quantile",
                    ("B",),
                    quantile_percent=frame.quantile_percent,
                    equality_policy="reject_boundary_ties",
                    rounding_policy="ceil",
                    minimum_size=3,
                ),
                PublicCohort(
                    "bottom",
                    "bottom_quantile",
                    ("B",),
                    quantile_percent=frame.quantile_percent,
                    equality_policy="reject_boundary_ties",
                    rounding_policy="ceil",
                    minimum_size=3,
                ),
            )
        )
    elif frame.topology.startswith("ranked_cohort_"):
        cohorts.append(
            PublicCohort(
                "top",
                "top_quantile",
                ("A",),
                quantile_percent=frame.threshold_value,
                equality_policy="reject_boundary_ties",
                rounding_policy="ceil",
                minimum_size=3,
            )
        )
        if frame.topology == "ranked_cohort_gap":
            cohorts.append(
                PublicCohort(
                    "bottom",
                    "bottom_quantile",
                    ("A",),
                    quantile_percent=frame.threshold_value,
                    equality_policy="reject_boundary_ties",
                    rounding_policy="ceil",
                    minimum_size=3,
                )
            )
    elif frame.topology == "top_n_dual_predicate_count":
        for index, role in enumerate(("B", "C")):
            predicates.append(
                PublicPredicate(
                    f"predicate_{index}",
                    role,
                    frame.predicate_operators[index],
                    "fixed",
                    threshold_value=frame.predicate_thresholds[index],
                    transform="identity",
                )
            )
        cohorts.extend(
            (
                PublicCohort(
                    "top",
                    "top_n",
                    ("A",),
                    size=frame.top_n,
                    equality_policy="reject_boundary_ties",
                ),
                PublicCohort(
                    "selected",
                    "intersection",
                    ("A", "B", "C"),
                    tuple(predicate.predicate_id for predicate in predicates),
                ),
            )
        )
    elif frame.topology == "dol_terminal" and frame.topology_mode == "with_size":
        predicates.extend(
            (
                PublicPredicate(
                    "size",
                    "A",
                    ">",
                    "fixed",
                    threshold_value=frame.threshold_value,
                    transform="identity",
                ),
                PublicPredicate(
                    "revenue_growth",
                    "B",
                    ">",
                    "fixed",
                    threshold_value=frame.secondary_threshold_value,
                    transform="growth",
                ),
            )
        )
        cohorts.append(
            PublicCohort(
                "selected",
                "intersection",
                ("A", "B"),
                ("size", "revenue_growth"),
            )
        )
    elif filter_roles:
        for index, role in enumerate(filter_roles):
            operator = (
                frame.predicate_operators[index]
                if index < len(frame.predicate_operators)
                else frame.threshold_operator
                or ("<" if frame.topology == "filtered_dual_transform_lookup" else ">")
            )
            threshold = (
                frame.predicate_thresholds[index]
                if index < len(frame.predicate_thresholds)
                else frame.threshold_value
            )
            transform = (
                frame.predicate_transforms[index]
                if index < len(frame.predicate_transforms)
                else frame.filter_transform
            )
            source = "zero" if threshold in (None, 0.0) else "fixed"
            predicates.append(
                PublicPredicate(
                    f"predicate_{index}",
                    role,
                    operator,
                    source,
                    threshold_value=threshold,
                    transform=transform,
                )
            )
        cohorts.append(
            PublicCohort(
                "selected",
                "intersection" if len(predicates) > 1 else "predicate",
                tuple(filter_roles),
                tuple(p.predicate_id for p in predicates),
            )
        )

    terminal_role = chr(64 + len(frame.metric_keys))
    terminal: list[PublicTerminalBranch]
    if frame.topology == "extreme_lookup_difference":
        terminal = [
            PublicTerminalBranch(
                "first_lookup",
                "lookup",
                metric_role=terminal_role,
                selector_step="first_key",
            ),
            PublicTerminalBranch(
                "second_lookup",
                "lookup",
                metric_role=terminal_role,
                selector_step="second_key",
            ),
            PublicTerminalBranch(
                "terminal", "difference", ("first_lookup", "second_lookup")
            ),
        ]
    elif frame.topology == "median_split_ratio":
        terminal = [
            PublicTerminalBranch(
                "high_sum",
                "restricted_sum",
                metric_role=terminal_role,
                cohort_id="high",
            ),
            PublicTerminalBranch(
                "low_sum", "restricted_sum", metric_role=terminal_role, cohort_id="low"
            ),
            PublicTerminalBranch("terminal", "ratio_of_sums", ("high_sum", "low_sum")),
        ]
    elif frame.topology in {"ranked_cohort_gap", "filtered_ranked_cohort_gap"}:
        terminal = [
            PublicTerminalBranch(
                "top_average", "average", metric_role=terminal_role, cohort_id="top"
            ),
            PublicTerminalBranch(
                "bottom_average",
                "average",
                metric_role=terminal_role,
                cohort_id="bottom",
            ),
            PublicTerminalBranch(
                "terminal", "difference", ("top_average", "bottom_average")
            ),
        ]
    elif frame.topology in {"ranked_cohort_share", "lower_median_positive_share"}:
        terminal = [
            PublicTerminalBranch(
                "part",
                "restricted_sum",
                metric_role=terminal_role,
                cohort_id="top"
                if frame.topology == "ranked_cohort_share"
                else "selected",
            ),
            PublicTerminalBranch(
                "whole",
                "sum",
                metric_role=terminal_role,
                cohort_id=None
                if frame.topology == "ranked_cohort_share"
                else "positive",
            ),
            PublicTerminalBranch("terminal", "share", ("part", "whole")),
        ]
    else:
        terminal = [
            PublicTerminalBranch(
                "terminal",
                INTENTS_BY_ID[frame.template_id].terminal_operation.value,
                metric_role=terminal_role,
                cohort_id="selected" if cohorts else None,
                selector_step="winner" if frame.selector_direction else None,
            )
        ]
    return tuple(predicates), tuple(cohorts), tuple(terminal)


@dataclass(frozen=True, slots=True)
class AnalyticalCandidateDraft:

    frame: AnalyticalFrame
    candidate_id: str
    entities: tuple[str, ...]
    periods: tuple[str, ...]
    graph: ReasoningGraph
    public_spec: PublicSpec


def iter_analytical_frame_drafts(
    frame_id: str,
    *,
    cube: Cube,
    company_meta: dict[str, CompanyInfo],
    seed: int | None,
    max_candidates: int,
) -> Iterator[AnalyticalCandidateDraft]:
    frame = FRAMES_BY_ID[frame_id]
    if frame.domain_kind == "same_period":
        raw_domains = same_period_domains(cube, company_meta)
    elif frame.domain_kind == "adjacent_period":
        raw_domains = adjacent_period_domains(cube, company_meta)
    elif frame.domain_kind == "nonfinancial_same_period":
        raw_domains = nonfinancial_same_period_domains(cube, company_meta)
    elif frame.domain_kind == "nonfinancial_adjacent_period":
        raw_domains = nonfinancial_adjacent_period_domains(cube, company_meta)
    elif frame.domain_kind == "industry_wide_same_period":
        raw_domains = industry_wide_same_period_domains(cube, company_meta)
    elif frame.domain_kind == "industry_wide_adjacent_period":
        raw_domains = industry_wide_adjacent_period_domains(cube, company_meta)
    elif frame.domain_kind == "industry_wide_cross_entity_period_window":
        raw_domains = industry_wide_cross_entity_period_window_domains(
            cube, company_meta
        )
    elif frame.domain_kind == "nonfinancial_cross_entity_period_window":
        raw_domains = nonfinancial_cross_entity_period_window_domains(
            cube, company_meta
        )
    elif frame.domain_kind == "single_entity_period_window":
        raw_domains = single_entity_period_window_domains(cube)
    else:
        raw_domains = cross_entity_period_window_domains(cube, company_meta)
    yielded = 0
    for raw in _seeded_domain_order(raw_domains, frame_id=frame_id, seed=seed):
        if yielded >= max_candidates:
            return
        if frame.domain_kind in {
            "same_period",
            "nonfinancial_same_period",
            "industry_wide_same_period",
        }:
            _industry, entities, period = raw
            periods = (period,)
        elif frame.domain_kind in {
            "adjacent_period",
            "nonfinancial_adjacent_period",
            "industry_wide_adjacent_period",
        }:
            _industry, entities, prior, current = raw
            periods = (prior, current)
        else:
            _industry, entities, periods = raw
        if (
            frame.topology
            in {
                "median_split_ratio",
                "lower_median_positive_share",
            }
            and len(entities) < 5
        ):
            # Keep period handling explicit and deterministic.
            continue
        attempt = _build_attempt(cube, frame, entities, periods)
        if attempt is None:
            continue
        selected, graph = attempt
        if (
            frame.topology
            in {
                "median_split_ratio",
                "lower_median_positive_share",
            }
            and len(selected) < 5
        ):
            continue
        # Keep LLM behavior within the declared contract.
        try:
            evaluate_graph(graph)
        except EvaluationError:
            continue
        candidate_id = _signature(frame_id, selected, periods)
        yield AnalyticalCandidateDraft(
            frame=frame,
            candidate_id=candidate_id,
            entities=selected,
            periods=periods,
            graph=graph,
            public_spec=_public_spec(
                frame,
                candidate_id=candidate_id,
                entities=selected,
                periods=periods,
                company_meta=company_meta,
            ),
        )
        yielded += 1


def _seeded_domain_order(
    raw_domains: list[tuple], *, frame_id: str, seed: int | None
) -> list[tuple]:
    """Deterministically diversify the first domains inspected by each frame.

    Domain builders intentionally return a canonical industry/year order.  Production used to
    discard ``seed`` here, so every frame repeatedly spent its first dependency-audit budget on
    the same oldest OCR windows.  A stable SHA-256 ordering keeps reproducibility without Python's
    process-randomized ``hash()`` and gives different frames independent domain permutations.
    ``seed=None`` preserves the canonical order used by capability probes and legacy callers.
    """
    if seed is None:
        return raw_domains
    return sorted(
        raw_domains,
        key=lambda raw: hashlib.sha256(
            f"{seed}|{frame_id}|{raw!r}".encode("utf-8")
        ).digest(),
    )


def _minimum_audited_universe_size(frame: AnalyticalFrame) -> int:
    if frame.domain_kind == "single_entity_period_window":
        return 1
    if frame.topology in {"median_split_ratio", "lower_median_positive_share"}:
        return 5
    return 3


def _term_dependency_id(role_id: str, term) -> str:
    cell = term.cell
    return f"{role_id}:{cell.table_ref}:{cell.row_idx}:{cell.col_idx}"


def _binding_entity(
    graph: ReasoningGraph,
    role: MetricRoleInput,
    domain_key,
) -> str:
    if role.expected_kind == ValueKind.NUMERIC_SERIES_ENTITY:
        return str(domain_key)
    if role.expected_kind == ValueKind.NUMERIC_SERIES_ENTITY_PERIOD:
        if not isinstance(domain_key, tuple) or len(domain_key) != 2:
            raise CandidateRejected(
                f"role {role.role_id}: invalid EntityPeriod binding key {domain_key!r}"
            )
        return domain_key[0]
    if role.expected_kind == ValueKind.NUMERIC_SERIES_PERIOD:
        if len(graph.domain.entities) != 1:
            raise CandidateRejected(
                f"role {role.role_id}: Period role requires exactly one entity in the graph domain"
            )
        return graph.domain.entities[0]
    raise CandidateRejected(
        f"role {role.role_id}: could not determine entity for kind {role.expected_kind.value}"
    )


def _dependency_entity_map(graph: ReasoningGraph) -> dict[str, frozenset[str]]:
    mapped: dict[str, set[str]] = {}
    for role in graph.metric_roles:
        for domain_key, terms in role.bindings.items():
            entity = _binding_entity(graph, role, domain_key)
            for term in (*terms.numerator, *terms.denominator):
                dependency_id = _term_dependency_id(role.role_id, term)
                mapped.setdefault(dependency_id, set()).add(entity)
    return {
        dependency_id: frozenset(entities) for dependency_id, entities in mapped.items()
    }


def _restrict_graph_entities(
    graph: ReasoningGraph,
    entities: tuple[str, ...],
) -> ReasoningGraph:
    """Restrict a provisional graph before candidate identity is locked.

    Every role remains full-coverage over the new eligible domain; no downstream survivor/winner is
    consulted. Node params contain no entity lists, while node/domain entity tuples are rebuilt.
    """
    allowed = set(entities)
    new_roles: list[MetricRoleInput] = []
    for role in graph.metric_roles:
        if role.expected_kind == ValueKind.NUMERIC_SERIES_ENTITY:
            bindings = {
                key: terms for key, terms in role.bindings.items() if key in allowed
            }
        elif role.expected_kind == ValueKind.NUMERIC_SERIES_ENTITY_PERIOD:
            bindings = {
                key: terms
                for key, terms in role.bindings.items()
                if isinstance(key, tuple) and key[0] in allowed
            }
        elif role.expected_kind == ValueKind.NUMERIC_SERIES_PERIOD:
            if len(entities) != 1:
                raise CandidateRejected(
                    f"role {role.role_id}: Period role requires exactly one entity after eligibility filtering"
                )
            bindings = dict(role.bindings)
        else:
            raise CandidateRejected(
                f"role {role.role_id}: unsupported source kind {role.expected_kind.value}"
            )
        new_roles.append(replace(role, bindings=bindings))
    role_map = {role.role_id: role for role in new_roles}
    new_nodes = tuple(
        replace(
            node,
            domain=replace(node.domain, entities=entities),
            inputs=tuple(
                role_map[item.role_id] if isinstance(item, MetricRoleInput) else item
                for item in node.inputs
            ),
        )
        for node in graph.nodes
    )
    return replace(
        graph,
        domain=replace(graph.domain, entities=entities),
        metric_roles=tuple(new_roles),
        nodes=new_nodes,
    )


def finalize_analytical_draft(
    draft: AnalyticalCandidateDraft,
    *,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
) -> RecipeCandidate:
    docs_by_name = {doc.doc_name: doc for doc in docs}
    ref_to_path: dict[str, Path] = table_ref_to_path_map(docs)
    report = audit_dependency_report(
        draft.graph,
        docs_by_name=docs_by_name,
        company_meta=company_meta,
        cache=audit_cache,
        auditor=auditor,
    )
    dependency_entities = _dependency_entity_map(draft.graph)
    unmapped_rejections = {
        dependency_id: reason
        for dependency_id, reason in report.rejection_reasons.items()
        if dependency_id not in dependency_entities
    }
    if unmapped_rejections:
        raise DependencyAuditRejected(unmapped_rejections)
    rejected_entities = {
        entity
        for dependency_id in report.rejection_reasons
        for entity in dependency_entities[dependency_id]
    }
    eligible_entities = tuple(
        entity for entity in draft.entities if entity not in rejected_entities
    )
    minimum_size = _minimum_audited_universe_size(draft.frame)
    if len(eligible_entities) < minimum_size:
        raise CandidateRejected(
            f"audited eligible universe has only {len(eligible_entities)}/{len(draft.entities)} "
            f"companies; requires >= {minimum_size}; rejected={sorted(rejected_entities)}"
        )
    if rejected_entities:
        logger.info(
            "Audited eligible universe frame=%s periods=%s retained=%d/%d excluded=%s",
            draft.frame.frame_id,
            draft.periods,
            len(eligible_entities),
            len(draft.entities),
            sorted(rejected_entities),
        )
    eligible_graph = _restrict_graph_entities(draft.graph, eligible_entities)
    validated_graph = rebuild_validated_graph(
        eligible_graph,
        AuditedValues(values=report.values, stats=report.stats),
    )
    graph, trace, compiled, answer, stats = finalize_validated_candidate(
        validated_graph=validated_graph,
        terminal_metric_key=draft.frame.terminal_key,
        terminal_transform=draft.frame.target_transform,
        table_ref_to_path=ref_to_path,
        audit_stats=report.stats,
        enforce_objective_gates=True,
    )
    candidate_id = _signature(draft.frame.frame_id, eligible_entities, draft.periods)
    return RecipeCandidate(
        candidate_id=candidate_id,
        recipe_id=draft.frame.frame_id,
        entities=eligible_entities,
        graph=graph,
        trace=trace,
        compiled=compiled,
        formatted_answer=answer,
        audit_stats=stats,
        public_spec=_public_spec(
            draft.frame,
            candidate_id=candidate_id,
            entities=eligible_entities,
            periods=draft.periods,
            company_meta=company_meta,
        ),
    )


def iter_analytical_frame_candidates(
    frame_id: str,
    *,
    cube: Cube,
    docs: list[DocumentRef],
    company_meta: dict[str, CompanyInfo],
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    max_candidates: int,
) -> Iterator[RecipeCandidate]:
    yielded = 0
    for draft in iter_analytical_frame_drafts(
        frame_id,
        cube=cube,
        company_meta=company_meta,
        seed=seed,
        max_candidates=max_candidates,
    ):
        if yielded >= max_candidates:
            return
        try:
            candidate = finalize_analytical_draft(
                draft,
                docs=docs,
                company_meta=company_meta,
                auditor=auditor,
                audit_cache=audit_cache,
            )
        except (DependencyAuditRejected, EvaluationError, CandidateRejected) as exc:
            logger.debug(
                "Analytical frame reject frame=%s periods=%s: %s",
                frame_id,
                draft.periods,
                exc,
            )
            continue
        yield candidate
        yielded += 1


def _iterator(frame_id: str):
    def iterate(**kwargs) -> Iterator[RecipeCandidate]:
        yield from iter_analytical_frame_candidates(frame_id, **kwargs)

    return iterate


ANALYTICAL_FRAME_ITERATORS: dict[str, object] = {
    frame_id: _iterator(frame_id) for frame_id in ENABLED_ANALYTICAL_FRAME_IDS
}
