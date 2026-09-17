"""Typed intent registry for the 70 reviewed Hard Cube templates.

The registry is deliberately data, not prompt examples.  A template keeps its own identity and
complete public operation grammar even when another template happens to use the same metrics.  A
compiler may only claim an intent after compiling that exact grammar; it must never substitute a
different intent under the original ``template_id``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal


class CapabilityState(str, Enum):
    RUNNABLE = "runnable"
    BLOCKED_BY_DATA = "blocked_by_data"
    BLOCKED_BY_METRIC = "blocked_by_metric"
    NOT_IMPLEMENTED = "not_implemented"


class UniverseKind(str, Enum):
    INDUSTRY = "industry"
    INDUSTRY_SIZE_FILTERED = "industry_size_filtered"
    NONFINANCIAL_CORPUS = "nonfinancial_corpus"
    NONFINANCIAL_SIZE_FILTERED = "nonfinancial_size_filtered"


class TerminalOperation(str, Enum):
    LOOKUP = "lookup"
    COUNT = "count"
    MAXIMUM = "maximum"
    MEAN = "mean"
    DIFFERENCE = "difference"
    SHARE = "share"
    RATIO_OF_SUMS = "ratio_of_sums"


@dataclass(frozen=True, slots=True)
class MetricContract:
    role: str
    metric_key: str
    formula_id: str


@dataclass(frozen=True, slots=True)
class DataRequirements:
    metric_keys: tuple[str, ...]
    period_offsets: tuple[int, ...]
    minimum_entities: int
    metadata_fields: tuple[str, ...] = ("industry_l3", "is_financial")
    requires_consolidated: bool = True
    special_sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class IntentSpec:
    template_id: str
    universe_kind: UniverseKind
    domain: Literal["entity", "entity_period"]
    metrics: tuple[MetricContract, ...]
    operator_sequence: tuple[str, ...]
    terminal_operation: TerminalOperation
    terminal_unit: Literal["count", "money", "number", "percentage", "percentage_point"]
    data_requirements: DataRequirements
    interpretation_limits: tuple[str, ...]
    deterministic_validity_gates: tuple[str, ...]
    implementation_state: CapabilityState = CapabilityState.NOT_IMPLEMENTED

    @property
    def operation_grammar(self) -> str:
        return " -> ".join(self.operator_sequence)

    @property
    def metric_family(self) -> str:
        families = {metric.formula_id for metric in self.metrics}
        return "+".join(sorted(families))


def _m(role: str, key: str, formula: str) -> MetricContract:
    return MetricContract(role, key, formula)


def _intent(
    template_id: str,
    universe: UniverseKind,
    metrics: tuple[MetricContract, ...],
    operations: tuple[str, ...],
    terminal: TerminalOperation,
    unit: Literal["count", "money", "number", "percentage", "percentage_point"],
    *,
    offsets: tuple[int, ...] = (0,),
    minimum_entities: int = 3,
    limits: tuple[str, ...] = (),
    gates: tuple[str, ...] = (),
    special_sources: tuple[str, ...] = (),
    state: CapabilityState = CapabilityState.NOT_IMPLEMENTED,
) -> IntentSpec:
    metric_keys = tuple(dict.fromkeys(metric.metric_key for metric in metrics))
    common_gates = (
        "universe_selected_from_metadata_and_coverage_before_terminal_evaluation",
        "terminal_is_one_finite_numeric_scalar",
        "all_runtime_dependencies_are_present_in_relevant_tables",
    )
    return IntentSpec(
        template_id=template_id,
        universe_kind=universe,
        domain="entity_period" if len(offsets) > 1 else "entity",
        metrics=metrics,
        operator_sequence=operations,
        terminal_operation=terminal,
        terminal_unit=unit,
        data_requirements=DataRequirements(
            metric_keys=metric_keys,
            period_offsets=offsets,
            minimum_entities=minimum_entities,
            special_sources=special_sources,
        ),
        interpretation_limits=limits,
        deterministic_validity_gates=common_gates + gates,
        implementation_state=state,
    )


# Formula/metric keys are stable compiler contracts.  Cross-period formula keys are resolved by
# ``recipe.bindings``; raw statement keys retain their existing namespaced form.
I = UniverseKind.INDUSTRY  # noqa: E741 - concise table alias local to this registry
IS = UniverseKind.INDUSTRY_SIZE_FILTERED
N = UniverseKind.NONFINANCIAL_CORPUS
NS = UniverseKind.NONFINANCIAL_SIZE_FILTERED


INTENT_REGISTRY: tuple[IntentSpec, ...] = (
    _intent(
        "A01",
        I,
        (_m("rank", "inventory_days", "F04"), _m("target", "gross_margin", "F03")),
        (
            "compute_two_period_inventory_days",
            "period_difference",
            "argmax",
            "lookup_same_entity",
            "period_difference",
        ),
        TerminalOperation.LOOKUP,
        "percentage_point",
        offsets=(-2, -1, 0),
        gates=(
            "inventory_nonnegative",
            "cogs_and_revenue_positive",
            "unique_selector_or_declared_tie_policy",
        ),
        limits=("do_not_infer_inventory_change_caused_margin_change",),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "A02",
        I,
        (
            _m("predicate_1", "inventory_to_assets", "inventory/total_assets"),
            _m("predicate_2", "gross_margin", "F03"),
        ),
        (
            "compute_two_period_ratios",
            "period_difference_each",
            "predicate_positive",
            "predicate_negative",
            "set_intersection",
            "count",
        ),
        TerminalOperation.COUNT,
        "count",
        offsets=(-1, 0),
        gates=("assets_and_revenue_positive",),
    ),
    _intent(
        "A03",
        I,
        (_m("filter", "cdkt:140", "F01"), _m("target", "cfo_margin", "F06")),
        ("growth", "filter_below_fixed_threshold", "restrict", "maximum"),
        TerminalOperation.MAXIMUM,
        "percentage",
        offsets=(-1, 0),
        gates=(
            "opening_inventory_positive",
            "revenue_positive",
            "threshold_fixed_before_terminal_values",
        ),
    ),
    _intent(
        "A04",
        I,
        (_m("rank", "cash_conversion_cycle", "F05"), _m("target", "roa", "F08")),
        ("compute_ccc", "argmin", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=(
            "receivables_payables_and_purchases_available",
            "no_implicit_cogs_proxy",
        ),
        special_sources=("reliable_purchases",),
    ),
    _intent(
        "A05",
        I,
        (
            _m("rank", "cfo_minus_net_margin", "F03+F06"),
            _m("target", "liabilities_to_equity", "F12"),
        ),
        (
            "compute_two_margins",
            "scalar_difference_by_entity",
            "argmax",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "number",
        gates=("revenue_and_equity_positive",),
        limits=("do_not_label_the_margin_gap_as_fraud",),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "A06",
        I,
        (
            _m("filter", "kqkd:50", "raw_pbt"),
            _m("rank", "operating_profit_to_pbt", "operating_profit/PBT"),
            _m("target", "kqkd:10", "F01"),
        ),
        (
            "filter_fixed_scale",
            "ratio",
            "argmin_within_survivors",
            "lookup_same_entity_next_period",
            "growth",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(0, 1),
        gates=("pbt_positive_and_not_near_zero", "next_period_coverage"),
        limits=("ratio_is_descriptive_not_a_standard_nonoperating_income_measure",),
    ),
    _intent(
        "A07",
        I,
        (_m("filter", "cfo_to_npat", "F19"), _m("target", "cdkt:200", "F01")),
        (
            "compute_ratio_two_periods",
            "temporal_all_above_fixed_threshold",
            "growth",
            "restrict",
            "arithmetic_mean",
        ),
        TerminalOperation.MEAN,
        "percentage",
        offsets=(-1, 0),
        gates=("npat_positive_both_periods", "survivor_set_nonempty"),
        limits=("threshold_is_not_a_universal_quality_standard",),
    ),
    _intent(
        "A08",
        I,
        (
            _m("rank", "recorded_debt_cost_proxy", "F14"),
            _m("target", "net_margin", "F03"),
        ),
        ("average_interest_bearing_debt", "ratio", "argmax", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=("gross_interest_expense_and_interest_bearing_debt_available",),
        limits=("proxy_is_backward_looking_not_a_marginal_rate",),
        special_sources=("interest_bearing_debt_split",),
    ),
    _intent(
        "A09",
        I,
        (
            _m("filter", "kqkd:10", "F01"),
            _m("rank", "dol", "F16"),
            _m("target", "operating_margin", "F03"),
        ),
        (
            "revenue_growth",
            "filter_positive",
            "operating_profit_growth",
            "divide_transformed_series",
            "argmax_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=(
            "operating_profit_positive_both_periods",
            "revenue_growth_not_near_zero",
            "dol_outlier_gate",
        ),
    ),
    _intent(
        "A10",
        I,
        (_m("predicate_1", "sga_expense", "F17"), _m("predicate_2", "kqkd:10", "F01")),
        ("normalize_cost_sign", "growth_each", "relational_predicate", "count"),
        TerminalOperation.COUNT,
        "count",
        offsets=(-1, 0),
        gates=("baseline_revenue_and_sga_positive",),
    ),
    _intent(
        "A11",
        I,
        (
            _m("filter", "kqkd:10", "F01"),
            _m("rank", "kqkd:11", "F01"),
            _m("target", "gross_margin", "F03"),
        ),
        (
            "revenue_growth",
            "filter_negative",
            "cogs_growth",
            "argmin_within_survivors",
            "gross_margin_difference",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage_point",
        offsets=(-1, 0),
        gates=("baseline_revenue_and_cogs_positive",),
        limits=("do_not_assert_cogs_change_is_the_only_margin_cause",),
    ),
    _intent(
        "A12",
        I,
        (_m("rank", "sga_intensity", "F17"), _m("target", "roa", "F08")),
        ("ratio", "argmax_and_argmin", "two_lookups", "scalar_difference"),
        TerminalOperation.DIFFERENCE,
        "percentage_point",
        offsets=(-1, 0),
        gates=("revenue_and_average_assets_positive", "declared_tie_policy"),
    ),
    _intent(
        "A13",
        I,
        (
            _m("rank", "operating_cash_flow_ratio", "F11"),
            _m("target", "quick_ratio", "F09"),
        ),
        ("ratio", "argmin", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "number",
        gates=("current_liabilities_positive",),
        limits=("do_not_call_this_defensive_interval_ratio",),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "A14",
        I,
        (_m("rank", "long_term_debt_share", "F13"), _m("target", "kqkd:23", "F01")),
        (
            "compute_debt_maturity_share_two_periods",
            "period_difference",
            "argmax",
            "interest_expense_growth",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=(
            "consistent_interest_bearing_debt_classification",
            "baseline_interest_expense_positive",
        ),
        special_sources=("interest_bearing_debt_split",),
    ),
    _intent(
        "A15",
        I,
        (
            _m("filter", "net_working_capital", "F10"),
            _m("rank", "liabilities_to_assets", "F12"),
            _m("target", "roa", "F08"),
        ),
        (
            "difference",
            "filter_negative",
            "ratio",
            "argmin_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=("total_and_average_assets_positive",),
        limits=("use_liabilities_to_assets_name_not_cfa_debt_to_assets",),
    ),
    _intent(
        "A16",
        I,
        (
            _m("rank", "equity_multiplier", "F08"),
            _m("target", "interest_coverage", "F15"),
        ),
        ("average_balances", "ratio", "argmax", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "number",
        offsets=(-1, 0),
        gates=("average_equity_and_interest_expense_positive",),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "A17",
        I,
        (
            _m("cohort", "liabilities_to_equity", "F12"),
            _m("target", "kqkd:60", "raw_npat"),
        ),
        (
            "median",
            "filter_below_median",
            "filter_positive_target",
            "restricted_sum",
            "total_sum",
            "scalar_share",
        ),
        TerminalOperation.SHARE,
        "percentage",
        gates=(
            "equity_positive",
            "positive_npat_denominator",
            "median_equality_policy_declared",
        ),
    ),
    _intent(
        "A18",
        I,
        (
            _m("rank", "gross_ppe", "gross_PPE_change"),
            _m("target", "cash_capex", "F18"),
        ),
        ("absolute_change", "argmax", "lookup", "sum_all", "scalar_share"),
        TerminalOperation.SHARE,
        "percentage",
        offsets=(-1, 0),
        gates=("gross_ppe_note_and_cash_capex_available",),
        special_sources=("ppe_note", "cash_capex_line"),
    ),
    _intent(
        "A19",
        I,
        (_m("rank", "kqkd:10", "absolute_change"),),
        (
            "absolute_change",
            "filter_positive",
            "argmax_within_survivors",
            "lookup",
            "restricted_sum",
            "scalar_share",
        ),
        TerminalOperation.SHARE,
        "percentage",
        offsets=(-1, 0),
        gates=("positive_increase_denominator",),
    ),
    _intent(
        "A20",
        I,
        (_m("filter", "lctt:20", "raw_cfo"), _m("target", "cash_capex", "F18")),
        ("filter_positive", "restricted_sum", "sum_all", "scalar_share"),
        TerminalOperation.SHARE,
        "percentage",
        gates=("cash_capex_line_available", "capex_denominator_positive"),
        special_sources=("cash_capex_line",),
    ),
    _intent(
        "A21",
        I,
        (_m("rank", "asset_turnover", "F08"), _m("target", "roe", "F08")),
        ("ratio", "argmin_and_argmax", "two_lookups", "scalar_difference"),
        TerminalOperation.DIFFERENCE,
        "percentage_point",
        offsets=(-1, 0),
        gates=("average_assets_and_equity_positive", "declared_tie_policy"),
    ),
    _intent(
        "A22",
        I,
        (_m("rank", "net_margin", "F03"), _m("target", "roa", "F08")),
        (
            "period_difference_rank_metric",
            "argmin",
            "period_difference_target",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage_point",
        offsets=(-2, -1, 0),
        gates=("revenue_and_average_assets_positive_both_periods",),
    ),
    _intent(
        "A23",
        I,
        (
            _m("predicate_1", "gross_margin", "F03"),
            _m("predicate_2", "asset_turnover", "F08"),
            _m("target", "roe", "F08"),
        ),
        (
            "period_difference_each",
            "predicate_negative",
            "predicate_positive",
            "set_intersection",
            "restrict",
            "maximum",
        ),
        TerminalOperation.MAXIMUM,
        "percentage",
        offsets=(-2, -1, 0),
        gates=("revenue_average_assets_and_equity_positive",),
        limits=("do_not_assert_a_causal_tradeoff",),
    ),
    _intent(
        "A24",
        I,
        (
            _m("rank", "long_term_assets_share", "F22"),
            _m("target", "asset_turnover", "F08"),
        ),
        ("ratio", "argmax", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "number",
        offsets=(-1, 0),
        gates=("total_and_average_assets_positive",),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "B01",
        NS,
        (
            _m("size", "kqkd:10", "raw_revenue"),
            _m("rank", "gross_margin", "F03"),
            _m("target", "roa", "F08"),
        ),
        (
            "filter_fixed_scale",
            "period_difference",
            "argmin_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-2, -1, 0),
        gates=(
            "revenue_and_average_assets_positive",
            "universe_description_matches_corpus",
        ),
    ),
    _intent(
        "B02",
        I,
        (_m("rank", "inventory_days", "F04"), _m("target", "cfo_margin", "F06")),
        ("compute_inventory_days", "argmax", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=("inventory_nonnegative", "cogs_and_revenue_positive"),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "B03",
        NS,
        (
            _m("size", "cdkt:270", "raw_assets"),
            _m("rank", "operating_accruals_ratio", "F07"),
            _m("target", "roe", "F08"),
        ),
        (
            "filter_fixed_scale",
            "ratio",
            "argmax_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=("average_assets_and_equity_positive",),
    ),
    _intent(
        "B04",
        IS,
        (
            _m("rank", "kqkd:10", "raw_revenue"),
            _m("predicate_1", "quick_ratio", "F09"),
            _m("predicate_2", "liabilities_to_equity", "F12"),
        ),
        (
            "top_n",
            "predicate_above_fixed_threshold",
            "predicate_below_fixed_threshold",
            "set_intersection",
            "count",
        ),
        TerminalOperation.COUNT,
        "count",
        gates=(
            "n_within_population",
            "tie_at_n_policy_declared",
            "current_liabilities_and_equity_positive",
        ),
    ),
    _intent(
        "B05",
        I,
        (
            _m("cohort", "liabilities_to_equity", "F12"),
            _m("target", "kqkd:23", "raw_interest_expense"),
        ),
        (
            "median",
            "split_above_and_at_or_below",
            "restricted_sum_each",
            "scalar_ratio",
        ),
        TerminalOperation.RATIO_OF_SUMS,
        "number",
        gates=("equity_positive", "both_interest_expense_sums_positive"),
        limits=("use_liabilities_to_equity_name_not_debt_to_equity",),
    ),
    _intent(
        "B06",
        N,
        (_m("cohort", "cfo_margin", "F06"), _m("target", "roa", "F08")),
        ("quantile_top_and_bottom", "cohort_average_each", "scalar_difference"),
        TerminalOperation.DIFFERENCE,
        "percentage_point",
        offsets=(-1, 0),
        minimum_entities=6,
        gates=(
            "at_least_three_entities_per_cohort",
            "quantile_and_tie_policy_declared",
            "revenue_and_average_assets_positive",
        ),
    ),
    _intent(
        "C01",
        IS,
        (
            _m("size", "kqkd:10", "raw_revenue"),
            _m("rank", "inventory_days", "F04"),
            _m("target", "gross_margin", "F03"),
        ),
        (
            "filter_fixed_scale",
            "compute_two_period_inventory_days",
            "period_difference",
            "argmax_within_survivors",
            "target_period_difference",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage_point",
        offsets=(-2, -1, 0),
        gates=("cogs_and_revenue_positive", "inventory_balance_coverage"),
    ),
    _intent(
        "C02",
        I,
        (
            _m("predicate_1", "inventory_to_assets", "inventory/total_assets"),
            _m("predicate_2", "gross_margin", "F03"),
        ),
        (
            "compute_two_period_ratios",
            "period_difference_each",
            "predicate_positive",
            "predicate_negative",
            "set_intersection",
            "count",
        ),
        TerminalOperation.COUNT,
        "count",
        offsets=(-1, 0),
        gates=("assets_and_revenue_positive",),
    ),
    _intent(
        "C03",
        IS,
        (
            _m("size", "kqkd:10", "raw_revenue"),
            _m("rank", "cash_conversion_cycle", "F05"),
            _m("target", "roa", "F08"),
        ),
        (
            "filter_fixed_scale",
            "compute_ccc",
            "argmin_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=(
            "receivables_payables_and_purchases_available",
            "no_implicit_cogs_proxy",
        ),
        special_sources=("reliable_purchases",),
    ),
    _intent(
        "C04",
        I,
        (
            _m("rank", "operating_accruals_ratio", "F07"),
            _m("target", "operating_cash_flow_ratio", "F11"),
        ),
        ("ratio", "argmax", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "number",
        offsets=(-1, 0),
        gates=("average_assets_and_current_liabilities_positive",),
        limits=("treat_as_a_review_signal_not_proof",),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "C05",
        I,
        (
            _m("rank", "cfo_minus_net_margin", "F03+F06"),
            _m("target", "liabilities_to_equity", "F12"),
        ),
        (
            "compute_two_margins",
            "scalar_difference_by_entity",
            "argmax",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "number",
        gates=("revenue_and_equity_positive",),
        limits=("do_not_assign_an_absolute_earnings_quality_label",),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "C06",
        NS,
        (
            _m("size", "kqkd:10", "raw_revenue"),
            _m("rank", "operating_profit_to_pbt", "operating_profit/PBT"),
            _m("target", "kqkd:10", "F01"),
        ),
        (
            "filter_fixed_scale",
            "ratio",
            "argmin_within_survivors",
            "lookup_same_entity_next_period",
            "growth",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(0, 1),
        gates=("pbt_positive_and_not_near_zero", "next_period_coverage"),
    ),
    _intent(
        "C07",
        I,
        (_m("filter", "kqkd:10", "F01"), _m("target", "dol", "F16")),
        (
            "revenue_growth",
            "filter_positive",
            "operating_profit_growth",
            "divide_transformed_series",
            "restrict",
            "maximum",
        ),
        TerminalOperation.MAXIMUM,
        "number",
        offsets=(-1, 0),
        gates=(
            "operating_profit_positive_both_periods",
            "revenue_growth_not_near_zero",
            "dol_outlier_gate",
        ),
    ),
    _intent(
        "C08",
        I,
        (_m("predicate_1", "sga_expense", "F17"), _m("predicate_2", "kqkd:10", "F01")),
        ("normalize_cost_sign", "growth_each", "relational_predicate", "count"),
        TerminalOperation.COUNT,
        "count",
        offsets=(-1, 0),
        gates=("baseline_revenue_and_sga_positive",),
    ),
    _intent(
        "C09",
        I,
        (
            _m("filter", "kqkd:10", "F01"),
            _m("rank", "kqkd:11", "F01"),
            _m("target", "gross_margin", "F03"),
        ),
        (
            "revenue_growth",
            "filter_negative",
            "cogs_growth",
            "argmin_within_survivors",
            "gross_margin_difference",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage_point",
        offsets=(-1, 0),
        gates=("baseline_revenue_and_cogs_positive",),
    ),
    _intent(
        "C10",
        I,
        (
            _m("filter_1", "net_working_capital", "F10"),
            _m("filter_2", "liabilities_to_assets", "F12"),
            _m("target", "roa", "F08"),
        ),
        (
            "difference",
            "filter_negative",
            "median_within_cohort",
            "filter_below_cohort_median",
            "restrict",
            "maximum",
        ),
        TerminalOperation.MAXIMUM,
        "percentage",
        offsets=(-1, 0),
        gates=("assets_positive", "each_intermediate_cohort_nonempty"),
    ),
    _intent(
        "C11",
        I,
        (
            _m("rank", "operating_cash_flow_ratio", "F11"),
            _m("target", "quick_ratio", "F09"),
        ),
        ("ratio", "argmin", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "number",
        gates=("current_liabilities_positive",),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "C12",
        I,
        (
            _m("rank", "short_term_debt_share", "F13"),
            _m("target", "interest_coverage", "F15"),
        ),
        (
            "compute_debt_maturity_share_two_periods",
            "period_difference",
            "argmin",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "number",
        offsets=(-1, 0),
        gates=("interest_bearing_debt_split_available", "interest_expense_positive"),
        special_sources=("interest_bearing_debt_split",),
    ),
    _intent(
        "C13",
        I,
        (
            _m("cohort", "liabilities_to_equity", "F12"),
            _m("target", "kqkd:60", "raw_npat"),
        ),
        (
            "median",
            "filter_below_median",
            "filter_positive_target",
            "restricted_sum",
            "total_sum",
            "scalar_share",
        ),
        TerminalOperation.SHARE,
        "percentage",
        gates=(
            "equity_and_positive_npat_denominator",
            "median_equality_policy_declared",
        ),
    ),
    _intent(
        "C14",
        I,
        (
            _m("rank", "gross_ppe", "gross_PPE_change"),
            _m("target", "cash_capex", "F18"),
        ),
        ("absolute_change", "argmax", "lookup", "sum_all", "scalar_share"),
        TerminalOperation.SHARE,
        "percentage",
        offsets=(-1, 0),
        gates=("gross_ppe_note_and_cash_capex_available",),
        special_sources=("ppe_note", "cash_capex_line"),
    ),
    _intent(
        "C15",
        I,
        (
            _m("predicate_1", "net_margin", "F03"),
            _m("predicate_2", "asset_turnover", "F08"),
            _m("predicate_3", "roe", "F08"),
            _m("target", "roe", "F08"),
        ),
        (
            "period_difference_each",
            "predicate_negative",
            "predicate_positive",
            "predicate_positive",
            "triple_intersection",
            "restrict",
            "maximum",
        ),
        TerminalOperation.MAXIMUM,
        "percentage",
        offsets=(-2, -1, 0),
        gates=("revenue_average_assets_and_equity_positive",),
        limits=("do_not_use_causal_language",),
    ),
    _intent(
        "C16",
        I,
        (_m("rank", "asset_turnover", "F08"), _m("target", "roe", "F08")),
        ("ratio", "argmin_and_argmax", "two_lookups", "scalar_difference"),
        TerminalOperation.DIFFERENCE,
        "percentage_point",
        offsets=(-1, 0),
        gates=("average_assets_and_equity_positive", "declared_tie_policy"),
    ),
    _intent(
        "C17",
        I,
        (
            _m("cohort", "liabilities_to_equity", "F12"),
            _m("target", "kqkd:23", "raw_interest_expense"),
        ),
        (
            "median",
            "split_above_and_at_or_below",
            "restricted_sum_each",
            "scalar_ratio",
        ),
        TerminalOperation.RATIO_OF_SUMS,
        "number",
        gates=("equity_and_denominator_sum_positive", "normalize_cost_sign"),
    ),
    _intent(
        "C18",
        I,
        (_m("cohort", "cfo_margin", "F06"), _m("target", "roa", "F08")),
        ("quantile_top_and_bottom", "cohort_average_each", "scalar_difference"),
        TerminalOperation.DIFFERENCE,
        "percentage_point",
        offsets=(-1, 0),
        minimum_entities=6,
        gates=("universe_large_enough", "revenue_and_average_assets_positive"),
    ),
    _intent(
        "C19",
        I,
        (_m("cohort", "gross_margin", "F03"), _m("target", "cdkt:110", "raw_cash")),
        ("top_quantile", "restricted_sum", "total_sum", "scalar_share"),
        TerminalOperation.SHARE,
        "percentage",
        minimum_entities=6,
        gates=("revenue_positive", "cash_nonnegative", "quantile_tie_policy_declared"),
    ),
    _intent(
        "C20",
        I,
        (
            _m("cohort", "liabilities_to_assets", "F12"),
            _m("target", "interest_coverage", "F15"),
        ),
        ("top_quantile", "restrict", "arithmetic_mean"),
        TerminalOperation.MEAN,
        "number",
        minimum_entities=6,
        gates=("interest_expense_positive", "finite_coverage_only"),
        limits=("use_liabilities_to_assets_name_not_cfa_debt_to_assets",),
    ),
    _intent(
        "D01",
        NS,
        (_m("filter", "cdkt:270", "raw_assets"), _m("target", "roa", "F08")),
        ("filter_fixed_scale", "restrict", "maximum"),
        TerminalOperation.MAXIMUM,
        "percentage",
        offsets=(-1, 0),
        gates=("average_assets_positive", "universe_description_matches_corpus"),
        state=CapabilityState.BLOCKED_BY_DATA,
    ),
    _intent(
        "D02",
        N,
        (
            _m("predicate_1", "kqkd:10", "F01"),
            _m("predicate_2", "roe", "F08"),
            _m("predicate_3", "lctt:20", "raw_cfo"),
        ),
        (
            "growth",
            "predicate_above_fixed_threshold",
            "predicate_above_fixed_threshold",
            "predicate_negative",
            "triple_intersection",
            "count",
        ),
        TerminalOperation.COUNT,
        "count",
        offsets=(-1, 0),
        gates=("thresholds_fixed_before_terminal_values", "average_equity_positive"),
    ),
    _intent(
        "D03",
        N,
        (_m("filter", "cdkt:270", "F01"), _m("target", "asset_turnover", "F08")),
        (
            "asset_growth",
            "filter_above_fixed_threshold",
            "target_period_difference",
            "restrict",
            "minimum",
            "absolute_magnitude",
        ),
        TerminalOperation.MAXIMUM,
        "number",
        offsets=(-2, -1, 0),
        gates=("wording_and_sign_alignment", "average_assets_positive"),
    ),
    _intent(
        "D04",
        NS,
        (
            _m("size", "kqkd:10", "raw_revenue"),
            _m("rank", "operating_accruals_ratio", "F07"),
            _m("target", "net_margin", "F03"),
        ),
        (
            "filter_fixed_scale",
            "ratio",
            "argmin_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=("average_assets_and_revenue_positive",),
        limits=("negative_accruals_are_not_proof_of_high_quality",),
    ),
    _intent(
        "D05",
        N,
        (
            _m("filter", "kqkd:60", "raw_npat"),
            _m("rank", "cfo_to_npat", "F19"),
            _m("target", "quick_ratio", "F09"),
        ),
        (
            "filter_fixed_scale",
            "ratio",
            "argmax_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "number",
        gates=("npat_and_current_liabilities_positive",),
    ),
    _intent(
        "D06",
        N,
        (
            _m("filter", "kqkd:50", "raw_pbt"),
            _m("rank", "operating_profit_to_pbt", "operating_profit/PBT"),
            _m("target", "kqkd:10", "F01"),
        ),
        (
            "filter_fixed_scale",
            "ratio",
            "argmin_within_survivors",
            "lookup_same_entity_next_period",
            "growth",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(0, 1),
        gates=("pbt_positive_and_not_near_zero", "next_period_coverage"),
    ),
    _intent(
        "D07",
        N,
        (
            _m("filter_1", "net_working_capital", "F10"),
            _m("filter_2", "cdkt:310", "raw_current_liabilities"),
            _m("rank", "operating_cash_flow_ratio", "F11"),
            _m("target", "roa", "F08"),
        ),
        (
            "compute_nwc_two_periods",
            "transition_nonnegative_to_negative",
            "filter_fixed_scale",
            "set_intersection",
            "argmax_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=("current_liabilities_and_average_assets_positive",),
    ),
    _intent(
        "D08",
        NS,
        (
            _m("size", "kqkd:10", "raw_revenue"),
            _m("target", "cash_conversion_cycle", "F05"),
        ),
        ("filter_fixed_scale", "compute_ccc", "restrict", "maximum"),
        TerminalOperation.MAXIMUM,
        "number",
        offsets=(-1, 0),
        gates=("receivables_payables_and_purchases_available",),
        special_sources=("reliable_purchases",),
    ),
    _intent(
        "D09",
        N,
        (
            _m("filter", "quick_ratio", "F09"),
            _m("target", "operating_cash_flow_ratio", "F11"),
        ),
        ("filter_below_fixed_threshold", "restrict", "maximum"),
        TerminalOperation.MAXIMUM,
        "number",
        gates=("current_liabilities_positive", "threshold_stated"),
        state=CapabilityState.RUNNABLE,
    ),
    _intent(
        "D10",
        N,
        (
            _m("filter", "kqkd:23", "raw_interest_expense"),
            _m("rank", "interest_coverage", "F15"),
            _m("target", "liabilities_to_equity", "F12"),
        ),
        ("filter_fixed_scale", "argmax_within_survivors", "lookup_same_entity"),
        TerminalOperation.LOOKUP,
        "number",
        gates=("equity_and_interest_expense_positive",),
    ),
    _intent(
        "D11",
        N,
        (
            _m("filter", "total_interest_bearing_debt", "F13"),
            _m("rank", "short_term_debt_share", "F13"),
            _m("target", "kqkd:23", "F01"),
        ),
        (
            "filter_fixed_scale",
            "compute_debt_share_two_periods",
            "period_difference",
            "argmin_within_survivors",
            "interest_expense_growth",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=("interest_bearing_debt_split_available",),
        special_sources=("interest_bearing_debt_split",),
    ),
    _intent(
        "D12",
        N,
        (
            _m("predicate_1", "roe", "F08"),
            _m("predicate_2", "roa", "F08"),
            _m("target", "equity_multiplier", "F08"),
        ),
        (
            "predicate_above_fixed_threshold",
            "predicate_below_fixed_threshold",
            "set_intersection",
            "restrict",
            "arithmetic_mean",
        ),
        TerminalOperation.MEAN,
        "number",
        offsets=(-1, 0),
        gates=("average_equity_and_assets_positive",),
    ),
    _intent(
        "D13",
        NS,
        (
            _m("filter_1", "kqkd:10", "raw_revenue"),
            _m("filter_2", "kqkd:10", "F01"),
            _m("target", "dol", "F16"),
        ),
        (
            "filter_fixed_scale",
            "growth",
            "filter_above_fixed_threshold",
            "set_intersection",
            "compute_dol",
            "restrict",
            "maximum",
        ),
        TerminalOperation.MAXIMUM,
        "number",
        offsets=(-1, 0),
        gates=("operating_profit_positive", "growth_denominator_control"),
    ),
    _intent(
        "D14",
        N,
        (
            _m("filter", "kqkd:10", "F01"),
            _m("rank", "sga_expense", "F17"),
            _m("target", "operating_margin", "F03"),
        ),
        (
            "revenue_growth",
            "filter_below_fixed_threshold",
            "sga_growth",
            "argmin_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-1, 0),
        gates=("baseline_revenue_and_sga_positive",),
    ),
    _intent(
        "D15",
        N,
        (_m("cohort", "kqkd:10", "raw_revenue"), _m("target", "cdkt:110", "raw_cash")),
        ("top_quantile", "restricted_sum", "total_sum", "scalar_share"),
        TerminalOperation.SHARE,
        "percentage",
        minimum_entities=6,
        gates=("cash_nonnegative", "universe_and_tie_policy_declared"),
    ),
    _intent(
        "D16",
        N,
        (
            _m("cohort", "liabilities_to_assets", "F12"),
            _m("target", "kqkd:23", "raw_interest_expense"),
        ),
        ("top_quantile", "restricted_sum", "total_sum", "scalar_share"),
        TerminalOperation.SHARE,
        "percentage",
        minimum_entities=6,
        gates=("assets_and_interest_expense_denominator_positive",),
    ),
    _intent(
        "D17",
        NS,
        (
            _m("size", "kqkd:10", "raw_revenue"),
            _m("cohort", "cfo_margin", "F06"),
            _m("target", "roa", "F08"),
        ),
        (
            "filter_fixed_scale",
            "top_and_bottom_quantiles_within_survivors",
            "cohort_average_each",
            "scalar_difference",
        ),
        TerminalOperation.DIFFERENCE,
        "percentage_point",
        offsets=(-1, 0),
        minimum_entities=6,
        gates=("at_least_three_entities_per_cohort",),
    ),
    _intent(
        "D18",
        N,
        (
            _m("filter", "gross_margin", "F03"),
            _m("rank", "asset_turnover", "F08"),
            _m("target", "roe", "F08"),
        ),
        (
            "gross_margin_difference",
            "filter_below_fixed_threshold",
            "asset_turnover_difference",
            "argmax_within_survivors",
            "lookup_same_entity",
        ),
        TerminalOperation.LOOKUP,
        "percentage",
        offsets=(-2, -1, 0),
        gates=("revenue_average_assets_and_equity_positive",),
        limits=("do_not_claim_the_turnover_change_offset_margin_decline",),
    ),
    _intent(
        "D19",
        N,
        (
            _m("cohort", "long_term_assets_share", "F22"),
            _m("target", "asset_turnover", "F08"),
        ),
        ("top_quantile", "restrict", "arithmetic_mean"),
        TerminalOperation.MEAN,
        "number",
        offsets=(-1, 0),
        minimum_entities=6,
        gates=("assets_positive", "mean_of_ratios_is_explicit"),
    ),
    _intent(
        "D20",
        N,
        (
            _m("cohort", "liabilities_to_assets", "F12"),
            _m("target", "interest_coverage", "F15"),
        ),
        ("top_quantile", "restrict", "arithmetic_mean"),
        TerminalOperation.MEAN,
        "number",
        minimum_entities=6,
        gates=(
            "interest_expense_positive",
            "finite_coverage_and_small_denominator_outlier_gate",
        ),
    ),
)


# Updated only after an exact compiler graph has produced at least one candidate that passes the
# deterministic dependency projection, evaluator, objective gates, compiler and CSV execution on
# the current corpus.  This is not a claim of real LLM dependency-audit acceptance; the capability
# report labels that verification layer separately.
_CORPUS_RUNNABLE_TEMPLATE_IDS = frozenset(
    {
        "A01",
        "A02",
        "A03",
        "A05",
        "A06",
        "A07",
        "A09",
        "A10",
        "A12",
        "A13",
        "A15",
        "A16",
        "A17",
        "A19",
        "A21",
        "A22",
        "A23",
        "A24",
        "B01",
        "B02",
        "B04",
        "B05",
        "B06",
        "C01",
        "C02",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C10",
        "C11",
        "C13",
        "C16",
        "C17",
        "C18",
        "C19",
        "C20",
        "D02",
        "D03",
        "D04",
        "D05",
        "D06",
        "D07",
        "D09",
        "D10",
        "D13",
        "D14",
        "D17",
        "D18",
        "D20",
    }
)
_CORPUS_BLOCKED_TEMPLATE_IDS = frozenset(
    {
        "A11",
        "B03",
        "C09",
        "C15",
        "D01",
        "D12",
        "D15",
        "D16",
        "D19",
    }
)
_METRIC_BLOCKED_TEMPLATE_IDS = frozenset(
    {"A04", "A08", "A14", "A18", "A20", "C03", "C12", "C14", "D08", "D11"}
)
INTENT_REGISTRY = tuple(
    replace(intent, implementation_state=CapabilityState.RUNNABLE)
    if intent.template_id in _CORPUS_RUNNABLE_TEMPLATE_IDS
    else replace(intent, implementation_state=CapabilityState.BLOCKED_BY_DATA)
    if intent.template_id in _CORPUS_BLOCKED_TEMPLATE_IDS
    else replace(intent, implementation_state=CapabilityState.BLOCKED_BY_METRIC)
    if intent.template_id in _METRIC_BLOCKED_TEMPLATE_IDS
    else intent
    for intent in INTENT_REGISTRY
)


INTENTS_BY_ID: dict[str, IntentSpec] = {
    intent.template_id: intent for intent in INTENT_REGISTRY
}


def expected_template_ids() -> tuple[str, ...]:
    return tuple(
        [
            *(f"A{i:02d}" for i in range(1, 25)),
            *(f"B{i:02d}" for i in range(1, 7)),
            *(f"C{i:02d}" for i in range(1, 21)),
            *(f"D{i:02d}" for i in range(1, 21)),
        ]
    )


def validate_intent_registry() -> None:
    ids = tuple(intent.template_id for intent in INTENT_REGISTRY)
    expected = expected_template_ids()
    if ids != expected:
        missing = sorted(set(expected) - set(ids))
        unexpected = sorted(set(ids) - set(expected))
        duplicates = sorted(
            {template_id for template_id in ids if ids.count(template_id) > 1}
        )
        raise ValueError(
            f"Hard template registry mismatch: missing={missing}, unexpected={unexpected}, "
            f"duplicates={duplicates}, ordered={ids == expected}"
        )
    for intent in INTENT_REGISTRY:
        if not intent.metrics or not intent.operator_sequence:
            raise ValueError(
                f"{intent.template_id}: metrics/operator_sequence must be non-empty"
            )
        if intent.operator_sequence[-1] not in {
            "lookup_same_entity",
            "lookup_same_entity_next_period",
            "period_difference",
            "growth",
            "count",
            "maximum",
            "arithmetic_mean",
            "scalar_difference",
            "scalar_share",
            "scalar_ratio",
            "lookup",
            "lookup_same_entity",
        }:
            # Some grammars intentionally include the terminal operation as a compound last token.
            terminal_tokens = {
                TerminalOperation.LOOKUP: ("lookup", "growth", "difference"),
                TerminalOperation.COUNT: ("count",),
                TerminalOperation.MAXIMUM: ("maximum", "magnitude"),
                TerminalOperation.MEAN: ("mean", "average"),
                TerminalOperation.DIFFERENCE: ("difference",),
                TerminalOperation.SHARE: ("share",),
                TerminalOperation.RATIO_OF_SUMS: ("ratio",),
            }[intent.terminal_operation]
            if not any(
                token in intent.operator_sequence[-1] for token in terminal_tokens
            ):
                raise ValueError(
                    f"{intent.template_id}: terminal={intent.terminal_operation.value} does not "
                    f"match final operator {intent.operator_sequence[-1]!r}"
                )


validate_intent_registry()
