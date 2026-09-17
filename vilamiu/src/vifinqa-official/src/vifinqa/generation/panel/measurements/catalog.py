
from __future__ import annotations

from vifinqa.generation.panel.base import Cube
from vifinqa.generation.panel.catalog import get_ratio
from vifinqa.generation.panel.measurements.base import Measurement, MeasurementOutcome, MeasurementUnit, PeriodBasis

_REASON_MISSING_METRIC = "missing_metric"


def _cell_value(cube: Cube, ticker: str, period: str, metric_key: str) -> float | None:
    cell = cube.cell(ticker, period, metric_key)
    return cell.value if cell is not None else None


class OperatingAccrualsRatio:

    measurement_id = "EQ_01"
    name = "Tỷ lệ dồn tích hoạt động trên tài sản bình quân"
    period_basis: PeriodBasis = "adjacent_period"
    unit: MeasurementUnit = "percentage"
    invalid_reason_codes: tuple[str, ...] = (_REASON_MISSING_METRIC, "average_assets_not_positive")

    def evaluate(self, cube: Cube, ticker: str, periods: tuple[str, ...]) -> MeasurementOutcome:
        prior_period, period = periods
        npat = _cell_value(cube, ticker, period, "kqkd:60")
        cfo = _cell_value(cube, ticker, period, "lctt:20")
        assets_t = _cell_value(cube, ticker, period, "cdkt:270")
        assets_prior = _cell_value(cube, ticker, prior_period, "cdkt:270")
        if None in (npat, cfo, assets_t, assets_prior):
            return MeasurementOutcome(None, _REASON_MISSING_METRIC)
        average_assets = (assets_t + assets_prior) / 2
        if average_assets <= 0:
            return MeasurementOutcome(None, "average_assets_not_positive")
        return MeasurementOutcome((npat - cfo) / average_assets, None)


class InventoryToCurrentLiabilities:

    measurement_id = "LIQ_02"
    name = "Hàng tồn kho trên nợ ngắn hạn"
    period_basis: PeriodBasis = "same_period"
    unit: MeasurementUnit = "number"
    invalid_reason_codes: tuple[str, ...] = (
        _REASON_MISSING_METRIC,
        "current_liabilities_not_positive",
        "current_assets_negative",
        "inventory_negative",
    )

    def evaluate(self, cube: Cube, ticker: str, periods: tuple[str, ...]) -> MeasurementOutcome:
        (period,) = periods
        current_assets = _cell_value(cube, ticker, period, "cdkt:100")
        inventory = _cell_value(cube, ticker, period, "cdkt:140")
        current_liabilities = _cell_value(cube, ticker, period, "cdkt:310")
        if None in (current_assets, inventory, current_liabilities):
            return MeasurementOutcome(None, _REASON_MISSING_METRIC)
        if current_liabilities <= 0:
            return MeasurementOutcome(None, "current_liabilities_not_positive")
        if current_assets < 0:
            return MeasurementOutcome(None, "current_assets_negative")
        if inventory < 0:
            return MeasurementOutcome(None, "inventory_negative")
        ratio = get_ratio("inventory_to_current_liabilities")
        assert ratio is not None
        value = ratio.compute({"cdkt:140": inventory, "cdkt:310": current_liabilities})
        return MeasurementOutcome(value, None)


class GrossMarginChange:

    measurement_id = "GRO_03"
    name = "Thay đổi biên lợi nhuận gộp so với kỳ trước"
    period_basis: PeriodBasis = "adjacent_period"
    unit: MeasurementUnit = "percentage_point"
    invalid_reason_codes: tuple[str, ...] = (_REASON_MISSING_METRIC, "revenue_not_positive")

    def evaluate(self, cube: Cube, ticker: str, periods: tuple[str, ...]) -> MeasurementOutcome:
        prior_period, period = periods
        gross_t = _cell_value(cube, ticker, period, "kqkd:20")
        revenue_t = _cell_value(cube, ticker, period, "kqkd:10")
        gross_prior = _cell_value(cube, ticker, prior_period, "kqkd:20")
        revenue_prior = _cell_value(cube, ticker, prior_period, "kqkd:10")
        if None in (gross_t, revenue_t, gross_prior, revenue_prior):
            return MeasurementOutcome(None, _REASON_MISSING_METRIC)
        if revenue_t <= 0 or revenue_prior <= 0:
            return MeasurementOutcome(None, "revenue_not_positive")
        ratio = get_ratio("gross_margin")
        assert ratio is not None
        gpm_t = ratio.compute({"kqkd:20": gross_t, "kqkd:10": revenue_t})
        gpm_prior = ratio.compute({"kqkd:20": gross_prior, "kqkd:10": revenue_prior})
        assert gpm_t is not None and gpm_prior is not None
        return MeasurementOutcome(gpm_t - gpm_prior, None)


class GrossToNetMarginDifference:

    measurement_id = "PRO_04"
    name = "Chênh lệch giữa biên lợi nhuận gộp và biên lợi nhuận ròng"
    period_basis: PeriodBasis = "same_period"
    unit: MeasurementUnit = "percentage_point"
    invalid_reason_codes: tuple[str, ...] = (_REASON_MISSING_METRIC, "revenue_not_positive")

    def evaluate(self, cube: Cube, ticker: str, periods: tuple[str, ...]) -> MeasurementOutcome:
        (period,) = periods
        gross = _cell_value(cube, ticker, period, "kqkd:20")
        npat = _cell_value(cube, ticker, period, "kqkd:60")
        revenue = _cell_value(cube, ticker, period, "kqkd:10")
        if None in (gross, npat, revenue):
            return MeasurementOutcome(None, _REASON_MISSING_METRIC)
        if revenue <= 0:
            return MeasurementOutcome(None, "revenue_not_positive")
        ratio = get_ratio("gross_to_net_margin_difference")
        assert ratio is not None
        value = ratio.compute({"kqkd:20": gross, "kqkd:60": npat, "kqkd:10": revenue})
        return MeasurementOutcome(value, None)


class LeverageConditionedInterestCoverage:

    measurement_id = "LEV_05"
    name = "Hệ số khả năng thanh toán lãi vay (điều kiện theo hệ số nợ phải trả trên vốn chủ sở hữu)"
    period_basis: PeriodBasis = "same_period"
    unit: MeasurementUnit = "number"
    invalid_reason_codes: tuple[str, ...] = (
        _REASON_MISSING_METRIC,
        "equity_not_positive",
        "interest_expense_not_positive",
    )

    def evaluate(self, cube: Cube, ticker: str, periods: tuple[str, ...]) -> MeasurementOutcome:
        (period,) = periods
        pbt = _cell_value(cube, ticker, period, "kqkd:50")
        interest = _cell_value(cube, ticker, period, "kqkd:23")
        liabilities = _cell_value(cube, ticker, period, "cdkt:300")
        equity = _cell_value(cube, ticker, period, "cdkt:400")
        if None in (pbt, interest, liabilities, equity):
            return MeasurementOutcome(None, _REASON_MISSING_METRIC)
        if equity <= 0:
            return MeasurementOutcome(None, "equity_not_positive")
        if interest <= 0:
            return MeasurementOutcome(None, "interest_expense_not_positive")
        ratio = get_ratio("interest_coverage")
        assert ratio is not None
        value = ratio.compute({"kqkd:50": pbt, "kqkd:23": interest})
        return MeasurementOutcome(value, None)


class OcfRatioWorkingCapitalDeficit:

    measurement_id = "WCA_10"
    name = "Hệ số dòng tiền hoạt động trên nợ ngắn hạn (điều kiện thiếu hụt vốn lưu động)"
    period_basis: PeriodBasis = "same_period"
    unit: MeasurementUnit = "number"
    invalid_reason_codes: tuple[str, ...] = (
        _REASON_MISSING_METRIC,
        "current_liabilities_not_positive",
        "condition_not_met",
    )

    def evaluate(self, cube: Cube, ticker: str, periods: tuple[str, ...]) -> MeasurementOutcome:
        (period,) = periods
        current_assets = _cell_value(cube, ticker, period, "cdkt:100")
        current_liabilities = _cell_value(cube, ticker, period, "cdkt:310")
        cfo = _cell_value(cube, ticker, period, "lctt:20")
        if None in (current_assets, current_liabilities, cfo):
            return MeasurementOutcome(None, _REASON_MISSING_METRIC)
        if current_liabilities <= 0:
            return MeasurementOutcome(None, "current_liabilities_not_positive")
        if not (current_assets < current_liabilities):
            return MeasurementOutcome(None, "condition_not_met")
        ratio = get_ratio("operating_cash_flow_ratio")
        assert ratio is not None
        value = ratio.compute({"lctt:20": cfo, "cdkt:310": current_liabilities})
        return MeasurementOutcome(value, None)


EQ_01 = OperatingAccrualsRatio()
LIQ_02 = InventoryToCurrentLiabilities()
GRO_03 = GrossMarginChange()
PRO_04 = GrossToNetMarginDifference()
LEV_05 = LeverageConditionedInterestCoverage()
WCA_10 = OcfRatioWorkingCapitalDeficit()

MEASUREMENTS: tuple[Measurement, ...] = (EQ_01, LIQ_02, GRO_03, PRO_04, LEV_05, WCA_10)
