
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vifinqa.common.corpus.company_meta import CompanyInfo

MetricValueKind = Literal["money", "percentage", "percentage_point", "number"]


_KQKD_METRIC_NAMES: dict[str, str] = {
    "01": "Doanh thu bán hàng và cung cấp dịch vụ",
    "02": "Các khoản giảm trừ doanh thu",
    "10": "Doanh thu thuần",
    "11": "Giá vốn hàng bán",
    "20": "Lợi nhuận gộp",
    "21": "Doanh thu hoạt động tài chính",
    "22": "Chi phí tài chính",
    "23": "Chi phí lãi vay",
    "25": "Chi phí bán hàng",
    "26": "Chi phí quản lý doanh nghiệp",
    "30": "Lợi nhuận thuần từ hoạt động kinh doanh",
    "40": "Lợi nhuận khác",
    "50": "Lợi nhuận trước thuế",
    "60": "Lợi nhuận sau thuế",
}
_CDKT_METRIC_NAMES: dict[str, str] = {
    "100": "Tài sản ngắn hạn",
    "110": "Tiền và các khoản tương đương tiền",
    "140": "Hàng tồn kho",
    "200": "Tài sản dài hạn",
    "270": "Tổng tài sản",
    "300": "Nợ phải trả",
    "310": "Nợ ngắn hạn",
    "400": "Vốn chủ sở hữu",
}
_LCTT_METRIC_NAMES: dict[str, str] = {
    "20": "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
    "70": "Tiền và tương đương tiền cuối kỳ",
}

_METRIC_NAMES_BY_KIND: dict[str, dict[str, str]] = {
    "kqkd": _KQKD_METRIC_NAMES,
    "cdkt": _CDKT_METRIC_NAMES,
    "lctt": _LCTT_METRIC_NAMES,
}


def metric_name(metric_key: str) -> str | None:
    kind, sep, ma_so = metric_key.partition(":")
    if not sep:
        return None
    return _METRIC_NAMES_BY_KIND.get(kind, {}).get(ma_so)


def is_known_metric(metric_key: str) -> bool:
    return metric_name(metric_key) is not None


def display_name(key: str) -> str | None:
    ratio = _RATIOS_BY_KEY.get(key)
    if ratio is not None:
        return ratio.name
    derived = GROUNDED_DERIVED_METRIC_NAMES.get(key)
    if derived is not None:
        return derived
    return metric_name(key)


# Keep period handling explicit and deterministic.
#
# Keep period handling explicit and deterministic.


@dataclass(frozen=True, slots=True)
class RatioDefinition:
    key: str
    name: str
    numerator: tuple[tuple[float, str], ...]
    denominator: tuple[tuple[float, str], ...]
    value_kind: MetricValueKind
    denominator_nonzero: bool = True

    @property
    def requires(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for _, metric_key in (*self.numerator, *self.denominator):
            seen.setdefault(metric_key, None)
        return tuple(seen)

    def compute(self, values: dict[str, float]) -> float | None:
        denominator = sum(
            coef * values[metric_key] for coef, metric_key in self.denominator
        )
        if self.denominator_nonzero and denominator == 0:
            return None
        numerator = sum(
            coef * values[metric_key] for coef, metric_key in self.numerator
        )
        return numerator / denominator


RATIOS: tuple[RatioDefinition, ...] = (
    RatioDefinition(
        key="gross_margin",
        name="Biên lợi nhuận gộp",
        numerator=((1.0, "kqkd:20"),),
        denominator=((1.0, "kqkd:10"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="net_margin",
        name="Biên lợi nhuận ròng",
        numerator=((1.0, "kqkd:60"),),
        denominator=((1.0, "kqkd:10"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="operating_margin",
        name="Biên lợi nhuận hoạt động",
        numerator=((1.0, "kqkd:30"),),
        denominator=((1.0, "kqkd:10"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="npat_to_ending_assets",
        name="Lợi nhuận sau thuế trên tổng tài sản cuối kỳ",
        numerator=((1.0, "kqkd:60"),),
        denominator=((1.0, "cdkt:270"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="npat_to_ending_equity",
        name="Lợi nhuận sau thuế trên vốn chủ sở hữu cuối kỳ",
        numerator=((1.0, "kqkd:60"),),
        denominator=((1.0, "cdkt:400"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="liabilities_to_equity",
        name="Hệ số nợ phải trả trên vốn chủ sở hữu",
        numerator=((1.0, "cdkt:300"),),
        denominator=((1.0, "cdkt:400"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="debt_to_assets",
        name="Hệ số nợ trên tổng tài sản",
        numerator=((1.0, "cdkt:300"),),
        denominator=((1.0, "cdkt:270"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="current_ratio",
        name="Hệ số thanh toán hiện hành",
        numerator=((1.0, "cdkt:100"),),
        denominator=((1.0, "cdkt:310"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="quick_ratio",
        name="Hệ số thanh toán nhanh",
        numerator=((1.0, "cdkt:100"), (-1.0, "cdkt:140")),
        denominator=((1.0, "cdkt:310"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="asset_turnover",
        name="Vòng quay tổng tài sản",
        numerator=((1.0, "kqkd:10"),),
        denominator=((1.0, "cdkt:270"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="interest_coverage",
        name="Hệ số khả năng thanh toán lãi vay",
        numerator=((1.0, "kqkd:50"), (1.0, "kqkd:23")),
        denominator=((1.0, "kqkd:23"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="inventory_to_current_liabilities",
        name="Hàng tồn kho trên nợ ngắn hạn",
        numerator=((1.0, "cdkt:140"),),
        denominator=((1.0, "cdkt:310"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="gross_to_net_margin_difference",
        name="Chênh lệch giữa biên lợi nhuận gộp và biên lợi nhuận ròng",
        numerator=((1.0, "kqkd:20"), (-1.0, "kqkd:60")),
        denominator=((1.0, "kqkd:10"),),
        value_kind="percentage_point",
    ),
    RatioDefinition(
        key="operating_cash_flow_ratio",
        name="Hệ số dòng tiền hoạt động trên nợ ngắn hạn",
        numerator=((1.0, "lctt:20"),),
        denominator=((1.0, "cdkt:310"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="cfo_margin",
        name="Biên dòng tiền từ hoạt động kinh doanh trên doanh thu",
        numerator=((1.0, "lctt:20"),),
        denominator=((1.0, "kqkd:10"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="cfo_minus_net_margin",
        name="Chênh lệch CFO margin trừ biên lợi nhuận ròng",
        numerator=((1.0, "lctt:20"), (-1.0, "kqkd:60")),
        denominator=((1.0, "kqkd:10"),),
        value_kind="percentage_point",
    ),
    RatioDefinition(
        key="inventory_to_assets",
        name="Tỷ trọng hàng tồn kho trên tổng tài sản",
        numerator=((1.0, "cdkt:140"),),
        denominator=((1.0, "cdkt:270"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="liabilities_to_assets",
        name="Hệ số nợ phải trả trên tổng tài sản",
        numerator=((1.0, "cdkt:300"),),
        denominator=((1.0, "cdkt:270"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="sga_intensity",
        name="Cường độ chi phí bán hàng và quản lý trên doanh thu",
        numerator=((1.0, "kqkd:25"), (1.0, "kqkd:26")),
        denominator=((1.0, "kqkd:10"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="long_term_assets_share",
        name="Tỷ trọng tài sản dài hạn trên tổng tài sản",
        numerator=((1.0, "cdkt:200"),),
        denominator=((1.0, "cdkt:270"),),
        value_kind="percentage",
    ),
    RatioDefinition(
        key="cfo_to_npat",
        name="Tỷ lệ CFO trên lợi nhuận sau thuế",
        numerator=((1.0, "lctt:20"),),
        denominator=((1.0, "kqkd:60"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="operating_profit_to_pbt",
        name="Tỷ lệ lợi nhuận thuần từ hoạt động kinh doanh trên lợi nhuận trước thuế",
        numerator=((1.0, "kqkd:30"),),
        denominator=((1.0, "kqkd:50"),),
        value_kind="number",
    ),
    RatioDefinition(
        key="cfo_to_operating_profit",
        name="Tỷ lệ dòng tiền từ hoạt động kinh doanh trên lợi nhuận thuần từ hoạt động kinh doanh",
        numerator=((1.0, "lctt:20"),),
        denominator=((1.0, "kqkd:30"),),
        value_kind="number",
    ),
)

_RATIOS_BY_KEY: dict[str, RatioDefinition] = {ratio.key: ratio for ratio in RATIOS}


def get_ratio(key: str) -> RatioDefinition | None:
    return _RATIOS_BY_KEY.get(key)


# Keep period handling explicit and deterministic.
GROUNDED_DERIVED_VALUE_KINDS: dict[str, MetricValueKind] = {
    "roa": "percentage",
    "roe": "percentage",
    "inventory_days": "number",
    "sga_expense": "money",
    "eq01_operating_accruals_ratio": "percentage",
    "gro03_delta_gross_margin": "percentage_point",
    "cash_roe_cohort_gap": "percentage_point",
    "growth_margin_squeeze_count": "number",
    "persistent_cash_revenue_growth": "percentage",
    "growth_cohort_margin_change": "percentage_point",
    "profitable_negative_cash_count": "number",
    "leverage_interest_expense_share": "percentage",
    "inventory_days_gross_margin_change": "percentage_point",
    "sga_expense_growth": "percentage",
    "asset_turnover_avg": "number",
    "equity_multiplier": "number",
    "net_working_capital": "money",
    "operating_accruals_ratio": "percentage",
    "cash_conversion_cycle": "number",
    "recorded_debt_cost_proxy": "percentage",
    "long_term_debt_share": "percentage",
    "short_term_debt_share": "percentage",
    "total_interest_bearing_debt": "money",
    "gross_ppe": "money",
    "cash_capex": "money",
    "dol": "number",
}

GROUNDED_DERIVED_METRIC_NAMES: dict[str, str] = {
    "roa": "Tỷ suất sinh lời trên tổng tài sản (ROA) theo tổng tài sản bình quân",
    "roe": "Tỷ suất sinh lời trên vốn chủ sở hữu (ROE) theo vốn chủ sở hữu bình quân",
    "inventory_days": "Số ngày tồn kho",
    "sga_expense": "Tổng chi phí bán hàng và chi phí quản lý doanh nghiệp",
    "asset_turnover_avg": "Vòng quay tổng tài sản theo tổng tài sản bình quân",
    "equity_multiplier": "Hệ số nhân vốn chủ sở hữu theo số dư bình quân",
    "net_working_capital": "Vốn lưu động ròng",
    "operating_accruals_ratio": "Tỷ lệ dồn tích hoạt động trên tổng tài sản bình quân",
    "cash_conversion_cycle": "Chu kỳ chuyển đổi tiền mặt",
    "recorded_debt_cost_proxy": "Proxy chi phí nợ ghi nhận",
    "long_term_debt_share": "Tỷ trọng nợ dài hạn trên tổng nợ chịu lãi",
    "short_term_debt_share": "Tỷ trọng nợ ngắn hạn trên tổng nợ chịu lãi",
    "total_interest_bearing_debt": "Tổng nợ chịu lãi",
    "gross_ppe": "Nguyên giá bất động sản, nhà xưởng và thiết bị",
    "cash_capex": "CAPEX tiền mặt",
    "dol": "Đòn bẩy hoạt động (DOL)",
}

GROUNDED_DERIVED_FORMULAS: dict[str, str] = {
    "roa": "Lợi nhuận sau thuế kỳ hiện tại / ((Tổng tài sản kỳ liền trước + Tổng tài sản kỳ hiện tại) / 2)",
    "roe": "Lợi nhuận sau thuế kỳ hiện tại / ((Vốn chủ sở hữu kỳ liền trước + Vốn chủ sở hữu kỳ hiện tại) / 2)",
    "inventory_days": "((Hàng tồn kho cuối kỳ liền trước + Hàng tồn kho cuối kỳ hiện tại) / 2) / Giá vốn hàng bán kỳ hiện tại * 365",
    "sga_expense": "Chi phí bán hàng + Chi phí quản lý doanh nghiệp",
    "asset_turnover_avg": "Doanh thu thuần / ((Tổng tài sản kỳ trước + Tổng tài sản kỳ hiện tại) / 2)",
    "equity_multiplier": "Tổng tài sản bình quân / Vốn chủ sở hữu bình quân",
    "net_working_capital": "Tài sản ngắn hạn - Nợ ngắn hạn",
    "operating_accruals_ratio": "(Lợi nhuận sau thuế - Dòng tiền từ hoạt động kinh doanh) / Tổng tài sản bình quân",
    "dol": "Tăng trưởng lợi nhuận thuần từ hoạt động kinh doanh / Tăng trưởng doanh thu thuần",
}

GROUNDED_DERIVED_UNIT_LABELS: dict[str, str] = {
    "roa": "%",
    "roe": "%",
    "inventory_days": "ngày",
    "sga_expense": "đồng",
    "asset_turnover_avg": "lần",
    "equity_multiplier": "lần",
    "net_working_capital": "đồng",
    "operating_accruals_ratio": "%",
    "cash_conversion_cycle": "ngày",
    "recorded_debt_cost_proxy": "%",
    "long_term_debt_share": "%",
    "short_term_debt_share": "%",
    "total_interest_bearing_debt": "đồng",
    "gross_ppe": "đồng",
    "cash_capex": "đồng",
    "dol": "lần",
}


# Keep period handling explicit and deterministic.
GROWTH_ELIGIBLE_METRICS: frozenset[str] = frozenset({"kqkd:10"})

COST_METRIC_KEYS: frozenset[str] = frozenset(
    f"kqkd:{code}" for code in ("11", "22", "23", "25", "26", "32", "51", "52")
)



_EXCLUDED_INDUSTRY_L2 = "Tổ chức tín dụng"


def is_excluded_industry(info: CompanyInfo) -> bool:
    return info.industry_l2 == _EXCLUDED_INDUSTRY_L2


def peer_tickers(ticker: str, company_meta: dict[str, CompanyInfo]) -> frozenset[str]:
    info = company_meta.get(ticker)
    if info is None or is_excluded_industry(info):
        return frozenset()

    same_l2 = {
        other
        for other, other_info in company_meta.items()
        if not is_excluded_industry(other_info)
        and other_info.industry_l2 == info.industry_l2
    }
    if (
        len(same_l2) >= 2
    ):
        return frozenset(same_l2)

    same_l1 = {
        other
        for other, other_info in company_meta.items()
        if not is_excluded_industry(other_info)
        and other_info.industry_l1 == info.industry_l1
    }
    return frozenset(same_l1)


def industry_l3_groups(
    company_meta: dict[str, CompanyInfo],
) -> dict[str, frozenset[str]]:
    groups: dict[str, set[str]] = {}
    for ticker, info in company_meta.items():
        if is_excluded_industry(info):
            continue
        groups.setdefault(info.industry_l3, set()).add(ticker)
    return {name: frozenset(tickers) for name, tickers in groups.items()}
