"""Deterministic L5 screening plans for ViFinQA q0539-q0577.

The module compiles multi-stage questions into a scalar pandas expression.
Every symbol in the expression is backed by one reviewed source-table cell.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from .l1_fact_layer import (
    FactRetriever,
    ManualOverride,
    ParsedQuestion,
    load_companies,
    normalize,
    pandas_query_for,
    write_source_table_csv,
)
from .set_reasoning import copy_base_submission


FIRST_ID = 456
LAST_ID = 577
L5_IDS = (*range(456, 495), *range(539, 578))


@dataclass(frozen=True)
class FactRequest:
    symbol: str
    ticker: str
    year: int
    metric: str
    scope: str = "consolidated"
    period: str = "end_or_flow"


@dataclass(frozen=True)
class L5Spec:
    question_id: int
    facts: tuple[FactRequest, ...]
    expression: str
    answer_type: str = "float"


@dataclass(frozen=True)
class L5Override:
    question_id: int
    symbol: str
    year: int
    document_id: str
    source_line_1: int
    row_index: int
    column_index: int
    raw_value: str
    review_note: str


class Builder:
    def __init__(self, question_id: int):
        self.question_id = question_id
        self.requests: list[FactRequest] = []
        self._symbols: dict[tuple[str, int, str, str, str], str] = {}

    def fact(self, ticker: str, year: int, metric: str,
             scope: str = "consolidated", period: str = "end_or_flow") -> str:
        key = (ticker, year, metric, scope, period)
        if key not in self._symbols:
            symbol = f"f{len(self.requests) + 1}"
            self._symbols[key] = symbol
            self.requests.append(FactRequest(symbol, ticker, year, metric, scope, period))
        return self._symbols[key]

    def spec(self, expression: str, answer_type: str = "float") -> L5Spec:
        return L5Spec(self.question_id, tuple(self.requests), expression, answer_type)


def series(expressions: list[str]) -> str:
    return f"pd.Series([{', '.join(expressions)}], dtype='float64')"


def ratio(a: str, b: str, multiplier: float = 1.0, absolute_b: bool = True) -> str:
    denominator = f"abs({b})" if absolute_b else b
    return f"(({a}) / ({denominator}) * {multiplier:.1f})"


def revenue(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "doanh thu thuan ve ban hang va cung cap dich vu")


def gross_profit(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "loi nhuan gop ve ban hang va cung cap dich vu")


def pat(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "loi nhuan sau thue thu nhap doanh nghiep")


def pbt(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "tong loi nhuan ke toan truoc thue")


def cfo(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "luu chuyen tien thuan tu hoat dong kinh doanh")


def current_assets(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "tai san ngan han")


def current_liabilities(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "no ngan han")


def inventory(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "hang ton kho")


def cogs(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "gia von hang ban")


def liabilities(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "no phai tra")


def equity(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "von chu so huu")


def total_assets(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "tong tai san")


def interest(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "chi phi lai vay")


def operating_profit(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "loi nhuan thuan tu hoat dong kinh doanh")


def selling_expense(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "chi phi ban hang")


def admin_expense(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "chi phi quan ly doanh nghiep")


def cash_equivalents(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "tien va cac khoan tuong duong tien")


def gross_margin(b: Builder, ticker: str, year: int) -> str:
    return ratio(gross_profit(b, ticker, year), revenue(b, ticker, year), 100.0)


def net_margin(b: Builder, ticker: str, year: int) -> str:
    return ratio(pat(b, ticker, year), revenue(b, ticker, year), 100.0)


def revenue_growth(b: Builder, ticker: str, previous: int, current: int) -> str:
    old, new = revenue(b, ticker, previous), revenue(b, ticker, current)
    return f"((({new}) - ({old})) / abs({old}) * 100.0)"


def gross_margin_change(b: Builder, ticker: str, previous: int, current: int) -> str:
    return f"(({gross_margin(b, ticker, current)}) - ({gross_margin(b, ticker, previous)}))"


def current_ratio(b: Builder, ticker: str, year: int) -> str:
    return ratio(current_assets(b, ticker, year), current_liabilities(b, ticker, year))


def cfo_current_liabilities(b: Builder, ticker: str, year: int) -> str:
    return ratio(cfo(b, ticker, year), current_liabilities(b, ticker, year))


def debt_equity(b: Builder, ticker: str, year: int) -> str:
    return ratio(liabilities(b, ticker, year), equity(b, ticker, year))


def interest_coverage(b: Builder, ticker: str, year: int) -> str:
    cost = interest(b, ticker, year)
    return f"((({pbt(b, ticker, year)}) + abs({cost})) / abs({cost}))"


def inventory_days(b: Builder, ticker: str, year: int) -> str:
    start, end = inventory(b, ticker, year - 1), inventory(b, ticker, year)
    return f"((((abs({start}) + abs({end})) / 2.0) * 365.0) / abs({cogs(b, ticker, year)}))"


def roa(b: Builder, ticker: str, year: int) -> str:
    start, end = total_assets(b, ticker, year - 1), total_assets(b, ticker, year)
    return f"(({pat(b, ticker, year)}) / ((abs({start}) + abs({end})) / 2.0) * 100.0)"


def accrual_ratio(b: Builder, ticker: str, year: int) -> str:
    start, end = total_assets(b, ticker, year - 1), total_assets(b, ticker, year)
    return (
        f"((({pat(b, ticker, year)}) - ({cfo(b, ticker, year)})) / "
        f"((abs({start}) + abs({end})) / 2.0) * 100.0)"
    )


def sga(b: Builder, ticker: str, year: int) -> str:
    return f"(abs({selling_expense(b, ticker, year)}) + abs({admin_expense(b, ticker, year)}))"


def select_answer(selector: list[str], answers: list[str], mode: str) -> str:
    method = "idxmax" if mode == "max" else "idxmin"
    return f"float({series(answers)}.iloc[int({series(selector)}.{method}())])"


def filter_mean(condition: list[str], answers: list[str]) -> str:
    return f"float({series(answers)}[{series(condition)} > 0.0].mean())"


def filter_sum(condition: list[str], answers: list[str]) -> str:
    return f"float({series(answers)}[{series(condition)} > 0.0].sum())"


def filter_extreme(condition: list[str], selector: list[str], answers: list[str],
                   mode: str) -> str:
    method = "idxmax" if mode == "max" else "idxmin"
    masked = f"{series(selector)}.where({series(condition)} > 0.0)"
    return f"float({series(answers)}.iloc[int({masked}.{method}())])"


def spec_debt_interest(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("BSR", "PLX", "PVT")
    return b.spec(select_answer(
        [debt_equity(b, ticker, 2019) for ticker in tickers],
        [interest_coverage(b, ticker, 2019) for ticker in tickers], "max",
    ))


def spec_margin_change(question_id: int, tickers: tuple[str, ...],
                       previous: int, current: int) -> L5Spec:
    b = Builder(question_id)
    conditions = [revenue_growth(b, ticker, previous, current) for ticker in tickers]
    answers = [gross_margin_change(b, ticker, previous, current) for ticker in tickers]
    return b.spec(filter_mean(conditions, answers))


def spec_min_revenue_margin(question_id: int, ticker: str, years: tuple[int, ...],
                            divisor: float = 1e12) -> L5Spec:
    b = Builder(question_id)
    margins = [f"(({net_margin(b, ticker, year)}) > 10.0)" for year in years]
    revenues = [revenue(b, ticker, year) for year in years]
    return b.spec(filter_extreme(
        margins, revenues, [f"(({value}) / {divisor:.1f})" for value in revenues], "min"
    ))


def spec_doh_margin(question_id: int, previous: int, current: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("HPG", "HSG", "MSR", "NKG")
    selectors = [
        f"(({inventory_days(b, ticker, current)}) - ({inventory_days(b, ticker, previous)}))"
        for ticker in tickers
    ]
    answers = [gross_margin_change(b, ticker, previous, current) for ticker in tickers]
    return b.spec(select_answer(selectors, answers, "max"))


def spec_dcm_cfo_ratio(question_id: int, years: tuple[int, ...]) -> L5Spec:
    b = Builder(question_id)
    conditions = [f"(({net_margin(b, 'DCM', year)}) > 10.0)" for year in years]
    selectors = [revenue(b, "DCM", year) for year in years]
    answers = [cfo_current_liabilities(b, "DCM", year) for year in years]
    return b.spec(filter_extreme(conditions, selectors, answers, "min"))


def spec_operating_filter(question_id: int, year: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("BSR", "PLX", "PVT")
    conditions = [f"(({operating_profit(b, ticker, year)}) > 0.0)" for ticker in tickers]
    selectors = [ratio(cfo(b, ticker, year), operating_profit(b, ticker, year)) for ticker in tickers]
    answers = [net_margin(b, ticker, year) for ticker in tickers]
    return b.spec(filter_extreme(conditions, selectors, answers, "min"))


def spec_current_cfo_mean(question_id: int, tickers: tuple[str, ...], year: int) -> L5Spec:
    b = Builder(question_id)
    conditions = [f"(({current_ratio(b, ticker, year)}) < 1.0)" for ticker in tickers]
    answers = [cfo_current_liabilities(b, ticker, year) for ticker in tickers]
    return b.spec(filter_mean(conditions, answers))


def spec_cfo_positive_max_margin(question_id: int, tickers: tuple[str, ...],
                                 years: tuple[int, ...], answer_year: int) -> L5Spec:
    b = Builder(question_id)
    conditions = [
        "(" + " and ".join(f"(({cfo(b, ticker, year)}) > 0.0)" for year in years) + ")"
        for ticker in tickers
    ]
    answers = [net_margin(b, ticker, answer_year) for ticker in tickers]
    masked = f"{series(answers)}.where({series(conditions)} > 0.0)"
    return b.spec(f"float({masked}.max())")


def spec_positive_margin_revenue_sum(question_id: int, tickers: tuple[str, ...],
                                     years: tuple[int, ...], answer_year: int) -> L5Spec:
    b = Builder(question_id)
    conditions = [
        "(" + " and ".join(f"(({net_margin(b, ticker, year)}) > 0.0)" for year in years) + ")"
        for ticker in tickers
    ]
    answers = [f"(({revenue(b, ticker, answer_year)}) / 1000000000000.0)" for ticker in tickers]
    return b.spec(filter_sum(conditions, answers))


def spec_cfo_positive_growth_mean(question_id: int, tickers: tuple[str, ...],
                                  previous: int, current: int) -> L5Spec:
    b = Builder(question_id)
    conditions = [
        f"((({cfo(b, ticker, previous)}) > 0.0) and (({cfo(b, ticker, current)}) > 0.0))"
        for ticker in tickers
    ]
    answers = [revenue_growth(b, ticker, previous, current) for ticker in tickers]
    return b.spec(filter_mean(conditions, answers))


def spec_accrual_mean(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("DLG", "HHV", "VSC")
    conditions = [f"(({pat(b, ticker, 2020)}) > 0.0)" for ticker in tickers]
    answers = [accrual_ratio(b, ticker, 2020) for ticker in tickers]
    return b.spec(filter_mean(conditions, answers))


def spec_revenue_sum_margin(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("AAA", "DCM", "DPM", "GVR")
    conditions = [f"(({net_margin(b, ticker, 2016)}) > 10.0)" for ticker in tickers]
    answers = [f"(({revenue(b, ticker, 2016)}) / 1000000000000.0)" for ticker in tickers]
    return b.spec(filter_sum(conditions, answers))


def spec_revenue_sga_growth(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("BSR", "PLX", "PVT")
    selectors = [revenue_growth(b, ticker, 2021, 2022) for ticker in tickers]
    answers = [f"((({sga(b, ticker, 2022)}) - ({sga(b, ticker, 2021)})) / abs({sga(b, ticker, 2021)}) * 100.0)" for ticker in tickers]
    return b.spec(select_answer(selectors, answers, "max"))


def spec_growth_asset_turnover(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("HPG", "HSG", "MSR", "NKG")
    selectors, answers = [], []
    for ticker in tickers:
        for previous, current in ((2021, 2022), (2022, 2023)):
            selectors.append(revenue_growth(b, ticker, previous, current))
            answers.append(ratio(revenue(b, ticker, current), total_assets(b, ticker, current)))
    return b.spec(select_answer(selectors, answers, "max"))


def spec_cfo_growth_roa(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("GEE", "GEX", "SAM")
    conditions = [
        "(" + " and ".join(f"(({cfo(b, ticker, year)}) > 0.0)" for year in (2022, 2023, 2024)) + ")"
        for ticker in tickers
    ]
    selectors = [revenue_growth(b, ticker, 2023, 2024) for ticker in tickers]
    answers = [roa(b, ticker, 2024) for ticker in tickers]
    return b.spec(filter_extreme(conditions, selectors, answers, "max"))


def spec_first_negative_cfo_next_margin(question_id: int) -> L5Spec:
    b = Builder(question_id)
    years = tuple(range(2016, 2021))
    conditions = [f"(({cfo(b, 'KBC', year)}) < 0.0)" for year in years]
    answers = [gross_margin(b, "KBC", year + 1) for year in years]
    return b.spec(select_answer(conditions, answers, "max"))


def spec_current_assets_cfo_count(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("HPX", "NVL", "SCR", "VIC", "VRE")
    conditions = [
        f"((({current_assets(b, ticker, 2024)}) < ({current_liabilities(b, ticker, 2024)})) "
        f"and (({cfo(b, ticker, 2024)}) > 0.0))"
        for ticker in tickers
    ]
    return b.spec(f"int(({series(conditions)} > 0.0).sum())", "int")


def spec_inventory_median_debt_share(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("CEO", "HPX", "KBC", "SNZ", "VIC", "VPI", "VRE")
    ratios = [ratio(inventory(b, ticker, 2022), current_liabilities(b, ticker, 2022))
              for ticker in tickers]
    debts = [current_liabilities(b, ticker, 2022) for ticker in tickers]
    ratio_values, debt_values = series(ratios), series(debts)
    return b.spec(
        f"float({debt_values}[{ratio_values} > {ratio_values}.median()].sum() / "
        f"{debt_values}.sum() * 100.0)"
    )


def spec_kbc_debt_interest(question_id: int) -> L5Spec:
    b = Builder(question_id)
    years = tuple(range(2016, 2021))
    return b.spec(select_answer(
        [debt_equity(b, "KBC", year) for year in years],
        [interest_coverage(b, "KBC", year) for year in years], "max",
    ))


def spec_median_debt_interest(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("DIG", "KBC", "NVL", "SCR", "VRE")
    ratios = [debt_equity(b, ticker, 2016) for ticker in tickers]
    costs = [f"abs({interest(b, ticker, 2016)})" for ticker in tickers]
    ratio_values, cost_values = series(ratios), series(costs)
    return b.spec(
        f"float({cost_values}[{ratio_values} > {ratio_values}.median()].sum() / "
        f"{cost_values}[{ratio_values} <= {ratio_values}.median()].sum())"
    )


def spec_accrual_margin_debt(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("BSR", "PLX", "PVT")
    selectors = [
        ratio(f"(({cfo(b, ticker, 2017)}) - ({pat(b, ticker, 2017)}))",
              revenue(b, ticker, 2017), 1.0)
        for ticker in tickers
    ]
    answers = [debt_equity(b, ticker, 2017) for ticker in tickers]
    return b.spec(select_answer(selectors, answers, "max"))


def spec_current_debt_roa(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("DLG", "HHV", "VSC")
    conditions = [
        f"(({current_assets(b, ticker, 2020)}) < ({current_liabilities(b, ticker, 2020)}))"
        for ticker in tickers
    ]
    selectors = [ratio(liabilities(b, ticker, 2020), total_assets(b, ticker, 2020))
                 for ticker in tickers]
    answers = [roa(b, ticker, 2020) for ticker in tickers]
    return b.spec(filter_extreme(conditions, selectors, answers, "min"))


def spec_sga_growth_count(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("BSR", "PLX", "PVT")
    conditions = []
    for ticker in tickers:
        sga_growth = (
            f"((({sga(b, ticker, 2022)}) - ({sga(b, ticker, 2021)})) / "
            f"abs({sga(b, ticker, 2021)}) * 100.0)"
        )
        conditions.append(f"(({sga_growth}) > ({revenue_growth(b, ticker, 2021, 2022)}))")
    return b.spec(f"int(({series(conditions)} > 0.0).sum())", "int")


Q464_TICKERS = (
    "AAA", "ASM", "CEO", "DBC", "DCM", "DIG", "DLG", "DPM", "DTK", "DXG",
    "FIT", "FOX", "FPT", "GAS", "GEX", "GVR", "HAG", "HBC", "HDG", "HHS",
    "HNG", "HPG", "HSG", "HUT", "IJC", "KBC", "MPC", "MSN", "MSR", "MWG",
    "NLG", "OGC", "PC1", "PLX", "PNJ", "PVT", "QNS", "SAM", "SCR", "TTF",
    "VGC", "VGT", "VIC", "VIF", "VJC", "VNM", "VRE", "VSC",
)


def spec_inventory_decline_cfo_margin(question_id: int) -> L5Spec:
    """q464 omits an entity list, so screen all non-financial firms with both reports."""

    b = Builder(question_id)
    conditions, answers = [], []
    for ticker in Q464_TICKERS:
        old, new = inventory(b, ticker, 2015), inventory(b, ticker, 2016)
        conditions.append(f"((((({new}) - ({old})) / abs({old}) * 100.0)) <= -10.0)")
        answers.append(ratio(cfo(b, ticker, 2016), revenue(b, ticker, 2016), 100.0))
    return b.spec(filter_extreme(conditions, answers, answers, "max"))


def spec_median_debt_positive_pat_share(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("ASM", "DBC", "MCH", "MML", "MPC", "MSN", "OGC", "QNS", "VNM", "VSF")
    ratios = [debt_equity(b, ticker, 2022) for ticker in tickers]
    profits = [pat(b, ticker, 2022) for ticker in tickers]
    ratio_values, profit_values = series(ratios), series(profits)
    condition = f"(({ratio_values} < {ratio_values}.median()) & ({profit_values} > 0.0))"
    return b.spec(
        f"float({profit_values}[{condition}].sum() / "
        f"{profit_values}[{profit_values} > 0.0].sum() * 100.0)"
    )


def spec_top3_margin_cash_share(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("CEO", "DIG", "IJC", "KBC", "NVL", "SCR", "VIC", "VRE")
    margins = [gross_margin(b, ticker, 2016) for ticker in tickers]
    cash = [cash_equivalents(b, ticker, 2016) for ticker in tickers]
    margin_values, cash_values = series(margins), series(cash)
    return b.spec(
        f"float({cash_values}.iloc[{margin_values}.nlargest(3).index].sum() / "
        f"{cash_values}.sum() * 100.0)"
    )


def spec_quick_ratio_mean(question_id: int) -> L5Spec:
    b = Builder(question_id)
    tickers = ("AAA", "DCM", "GVR", "PRT")
    conditions = [f"(({current_ratio(b, ticker, 2017)}) >= 1.0)" for ticker in tickers]
    answers = [
        ratio(f"(({current_assets(b, ticker, 2017)}) - ({inventory(b, ticker, 2017)}))",
              current_liabilities(b, ticker, 2017))
        for ticker in tickers
    ]
    return b.spec(filter_mean(conditions, answers))


def build_specs() -> dict[int, L5Spec]:
    specs: dict[int, L5Spec] = {
        456: spec_first_negative_cfo_next_margin(456),
        457: spec_current_assets_cfo_count(457),
        458: spec_inventory_median_debt_share(458),
        459: spec_kbc_debt_interest(459),
        460: spec_median_debt_interest(460),
        461: spec_accrual_margin_debt(461),
        462: spec_current_debt_roa(462),
        463: spec_sga_growth_count(463),
        464: spec_inventory_decline_cfo_margin(464),
        465: spec_debt_interest(465),
        466: spec_median_debt_positive_pat_share(466),
        467: spec_top3_margin_cash_share(467),
        468: spec_accrual_mean(468),
        469: None,  # filled below; same inventory/current-liability screen as q566
        470: spec_quick_ratio_mean(470),
        471: spec_revenue_sum_margin(471),
        472: spec_cfo_positive_max_margin(472, ("AAA", "DCM", "DPM", "GVR", "PRT"), (2020, 2021, 2022), 2022),
        473: spec_positive_margin_revenue_sum(473, ("HPG", "HSG", "MSR", "NKG"), (2020, 2021, 2022), 2022),
        474: spec_min_revenue_margin(474, "ASM", (2016, 2017, 2018), divisor=1e9),
        475: spec_doh_margin(475, 2021, 2022),
        476: spec_revenue_sga_growth(476),
        477: spec_operating_filter(477, 2017),
        478: spec_positive_margin_revenue_sum(478, ("HPG", "HSG", "MSR", "NKG"), (2021, 2022, 2023), 2023),
        479: spec_doh_margin(479, 2022, 2023),
        480: spec_margin_change(480, ("DCM", "DPM", "PRT"), 2019, 2020),
        481: spec_min_revenue_margin(481, "CEO", (2022, 2023, 2024), divisor=1e9),
        482: spec_margin_change(482, ("DCM", "DPM", "GVR", "PRT"), 2021, 2022),
        483: spec_cfo_positive_growth_mean(483, ("DCM", "DPM", "PRT"), 2019, 2020),
        484: spec_cfo_positive_growth_mean(484, ("DCM", "DPM", "GVR", "PRT"), 2020, 2021),
        485: spec_dcm_cfo_ratio(485, (2020, 2021, 2022)),
        486: spec_current_cfo_mean(486, ("DLG", "HHV", "VSC"), 2020),
        487: spec_accrual_mean(487),
        488: spec_revenue_sum_margin(488),
        489: spec_cfo_positive_max_margin(489, ("AAA", "DCM", "DPM", "GVR", "PRT"), (2020, 2021, 2022), 2022),
        490: spec_positive_margin_revenue_sum(490, ("HPG", "HSG", "MSR", "NKG"), (2020, 2021, 2022), 2022),
        491: spec_revenue_sga_growth(491),
        492: spec_operating_filter(492, 2017),
        493: spec_positive_margin_revenue_sum(493, ("HPG", "HSG", "MSR", "NKG"), (2021, 2022, 2023), 2023),
        494: spec_operating_filter(494, 2019),
        539: spec_debt_interest(539),
        540: spec_margin_change(540, ("DCM", "DPM", "PRT"), 2019, 2020),
        541: spec_min_revenue_margin(541, "CEO", (2022, 2023, 2024)),
        542: spec_margin_change(542, ("DCM", "DPM", "GVR", "PRT"), 2021, 2022),
        543: spec_doh_margin(543, 2024, 2025),
        544: spec_dcm_cfo_ratio(544, (2020, 2021, 2022)),
        545: spec_operating_filter(545, 2017),
        546: spec_current_cfo_mean(546, ("DLG", "HHV", "VSC"), 2020),
        547: spec_cfo_positive_max_margin(547, ("AAA", "DCM", "DPM", "GVR", "PRT"), (2020, 2021, 2022), 2022),
        548: spec_min_revenue_margin(548, "ASM", (2016, 2017, 2018)),
        549: spec_growth_asset_turnover(549),
        550: spec_current_cfo_mean(550, ("ACV", "DLG", "HHV"), 2024),
        551: spec_cfo_positive_max_margin(551, ("GEE", "GEX", "SAM"), (2022, 2023, 2024), 2024),
        552: spec_positive_margin_revenue_sum(552, ("HPG", "HSG", "MSR", "NKG"), (2021, 2022, 2023), 2023),
        553: spec_debt_interest(553),
        554: spec_margin_change(554, ("DCM", "DPM", "PRT"), 2019, 2020),
        555: spec_min_revenue_margin(555, "CEO", (2022, 2023, 2024)),
        556: spec_doh_margin(556, 2023, 2024),
        557: spec_margin_change(557, ("DCM", "DPM", "GVR", "PRT"), 2021, 2022),
        558: spec_doh_margin(558, 2024, 2025),
        559: spec_cfo_positive_growth_mean(559, ("DCM", "DPM", "PRT"), 2019, 2020),
        560: spec_cfo_positive_growth_mean(560, ("DCM", "DPM", "GVR", "PRT"), 2020, 2021),
        561: spec_dcm_cfo_ratio(561, (2020, 2021, 2022)),
        562: spec_dcm_cfo_ratio(562, (2022, 2023, 2024)),
        563: None,  # filled below
        564: spec_current_cfo_mean(564, ("DLG", "HHV", "VSC"), 2020),
        565: spec_accrual_mean(565),
        566: None,  # filled below
        567: spec_revenue_sum_margin(567),
        568: spec_positive_margin_revenue_sum(568, ("HPG", "HSG", "MSR", "NKG"), (2020, 2021, 2022), 2022),
        569: None,  # filled below
        570: spec_doh_margin(570, 2021, 2022),
        571: spec_revenue_sga_growth(571),
        572: spec_cfo_growth_roa(572),
        573: spec_growth_asset_turnover(573),
        574: spec_cfo_positive_max_margin(574, ("GEE", "GEX", "SAM"), (2022, 2023, 2024), 2024),
        575: spec_doh_margin(575, 2022, 2023),
        576: spec_margin_change(576, ("DCM", "DPM", "PRT"), 2019, 2020),
        577: spec_min_revenue_margin(577, "CEO", (2022, 2023, 2024)),
    }

    b = Builder(563)
    tickers = ("GEE", "GEX", "SAM")
    conditions = [
        f"((({revenue_growth(b, ticker, 2020, 2021)}) > 0.0) and "
        f"(({gross_margin_change(b, ticker, 2020, 2021)}) < 0.0))"
        for ticker in tickers
    ]
    specs[563] = b.spec(f"int(({series(conditions)} > 0.0).sum())", "int")

    b = Builder(469)
    tickers = ("AAA", "DCM", "GVR", "PRT")
    conditions = [f"(({current_ratio(b, ticker, 2017)}) >= 1.0)" for ticker in tickers]
    answers = [ratio(inventory(b, ticker, 2017), current_liabilities(b, ticker, 2017))
               for ticker in tickers]
    specs[469] = b.spec(filter_mean(conditions, answers))

    b = Builder(566)
    tickers = ("AAA", "DCM", "GVR", "PRT")
    conditions = [f"(({current_ratio(b, ticker, 2017)}) >= 1.0)" for ticker in tickers]
    answers = [ratio(inventory(b, ticker, 2017), current_liabilities(b, ticker, 2017)) for ticker in tickers]
    specs[566] = b.spec(filter_mean(conditions, answers))

    b = Builder(569)
    tickers = ("HPG", "HSG", "MSR", "NKG")
    conditions = [f"(({net_margin(b, ticker, 2022)}) > 0.0)" for ticker in tickers]
    answers = [f"(({gross_margin(b, ticker, 2022)}) - ({net_margin(b, ticker, 2022)}))" for ticker in tickers]
    specs[569] = b.spec(filter_mean(conditions, answers))

    if sorted(specs) != list(L5_IDS) or any(value is None for value in specs.values()):
        raise ValueError("Incomplete L5 spec registry")
    return specs


L5_SPECS = build_specs()


BALANCE_METRICS = {
    "tai san ngan han", "no ngan han", "hang ton kho",
    "no phai tra", "von chu so huu", "tong tai san",
    "tien va cac khoan tuong duong tien", "tai san dai han", "tai san co dinh",
}
INCOME_METRICS = {
    "doanh thu thuan ve ban hang va cung cap dich vu",
    "loi nhuan gop ve ban hang va cung cap dich vu",
    "loi nhuan sau thue thu nhap doanh nghiep",
    "tong loi nhuan ke toan truoc thue",
    "loi nhuan thuan tu hoat dong kinh doanh",
    "gia von hang ban", "chi phi ban hang", "chi phi quan ly doanh nghiep",
}


def rerank_statement_candidates(candidates: list, request: FactRequest) -> list:
    """Prefer exact scope and the canonical statement for L5 core metrics."""

    exact_scope = [candidate for candidate in candidates
                   if candidate.document_scope == request.scope]
    if exact_scope:
        candidates = exact_scope
    if request.metric in BALANCE_METRICS:
        period_candidates = [candidate for candidate in candidates if any(
            phrase in normalize(candidate.column_text)
            for phrase in ("so cuoi nam", "so du cuoi nam", "cuoi ky",
                           f"31 12 {request.year}", str(request.year))
        )]
    elif request.metric in INCOME_METRICS or request.metric in {
        "luu chuyen tien thuan tu hoat dong kinh doanh", "chi phi lai vay",
    }:
        period_candidates = [candidate for candidate in candidates if any(
            phrase in normalize(candidate.column_text)
            for phrase in ("nam nay", "ky nay", str(request.year))
        )]
    else:
        period_candidates = []
    if period_candidates:
        candidates = period_candidates
    if request.metric in BALANCE_METRICS:
        preferred = [candidate for candidate in candidates
                     if candidate.table_kind == "primary_balance_sheet"]
    elif request.metric in INCOME_METRICS:
        preferred = [candidate for candidate in candidates
                     if candidate.table_kind == "primary_income_statement"]
    elif request.metric == "luu chuyen tien thuan tu hoat dong kinh doanh":
        preferred = [candidate for candidate in candidates
                     if candidate.table_kind == "primary_cash_flow"]
    elif request.metric == "chi phi lai vay":
        preferred = [candidate for candidate in candidates if candidate.table_kind in {
            "primary_income_statement", "primary_cash_flow", "financial_notes",
        }]
    else:
        preferred = []
    if preferred:
        preferred_ids = {
            (candidate.table_id, candidate.row_index, candidate.column_index)
            for candidate in preferred
        }
        candidates = [*preferred, *[candidate for candidate in candidates
            if (candidate.table_id, candidate.row_index, candidate.column_index)
            not in preferred_ids]]
    return candidates


def load_questions(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if int(row["id"]) in L5_SPECS:
            rows[int(row["id"])] = row
    if sorted(rows) != sorted(L5_SPECS):
        raise ValueError("Question file does not cover q0539-q0577")
    return rows


def load_overrides(path: Optional[Path]) -> dict[tuple[int, str], L5Override]:
    if path is None or not path.is_file():
        return {}
    result = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            override = L5Override(
                int(row["question_id"]), row["symbol"], int(row["year"]),
                row["document_id"], int(row["source_line_1"]), int(row["row_index"]),
                int(row["column_index"]), row["raw_value"], row["review_note"],
            )
            result[(override.question_id, override.symbol)] = override
    return result


def run_candidates(questions_path: Path, database: Path, output_dir: Path,
                   overrides_path: Optional[Path]) -> None:
    questions = load_questions(questions_path)
    companies = load_companies(Path("ViFinQA/code_stock.csv"))
    bank_tickers = {company.ticker for company in companies if "ngan hang" in normalize(company.name)}
    overrides = load_overrides(overrides_path)
    retriever = FactRetriever(database, bank_tickers=bank_tickers)
    plans, candidates_output = [], []
    retrieval_cache: dict[tuple[str, int, str, str, str], list] = {}
    try:
        for offset, question_id in enumerate(sorted(L5_SPECS), 1):
            spec = L5_SPECS[question_id]
            facts = []
            for request in spec.facts:
                parsed = ParsedQuestion(
                    id=question_id, question=questions[question_id]["question"],
                    ticker=request.ticker, matched_company_alias=request.ticker.lower(),
                    year=request.year, scope=request.scope, period_kind=request.period,
                    target_unit="VND_1", metric_text=normalize(request.metric),
                    metric_tokens=tuple(dict.fromkeys(normalize(request.metric).split())),
                )
                override = overrides.get((question_id, request.symbol))
                cache_key = (request.ticker, request.year, request.metric,
                             request.scope, request.period)
                if override is None and cache_key in retrieval_cache:
                    candidates = retrieval_cache[cache_key]
                else:
                    candidates = rerank_statement_candidates(
                        retriever.retrieve(parsed, limit=20), request
                    )[:5]
                    if override is None:
                        retrieval_cache[cache_key] = candidates
                if override is not None:
                    reviewed = retriever.retrieve_reviewed(parsed, ManualOverride(
                        question_id, override.document_id, override.source_line_1,
                        override.row_index, override.column_index, override.raw_value,
                        override.review_note,
                    ))
                    candidates = [reviewed, *[candidate for candidate in candidates
                        if candidate.table_id != reviewed.table_id or candidate.row_index != reviewed.row_index
                        or candidate.column_index != reviewed.column_index][:4]]
                if not candidates:
                    raise ValueError(f"q{question_id} {request.symbol}: no candidate")
                records = []
                for rank, candidate in enumerate(candidates, 1):
                    record = {**asdict(candidate), "question_id": question_id,
                              "symbol": request.symbol, "metric": request.metric,
                              "candidate_rank": rank}
                    records.append(record)
                    candidates_output.append(record)
                facts.append({"request": asdict(request), "selected": records[0]})
            plans.append({
                "question": questions[question_id], "expression": spec.expression,
                "answer_type": spec.answer_type, "facts": facts,
            })
            if offset % 10 == 0:
                print(f"retrieved {offset}/{len(L5_SPECS)} L5 questions", flush=True)
    finally:
        retriever.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plans.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in plans), encoding="utf-8"
    )
    (output_dir / "candidates.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in candidates_output), encoding="utf-8"
    )
    print(json.dumps({"questions": len(plans), "facts": sum(len(p["facts"]) for p in plans),
                      "unique_retrievals": len(retrieval_cache),
                      "candidates": len(candidates_output)}, ensure_ascii=False))


def cell_query(selected: dict, variable: str) -> str:
    return pandas_query_for(
        selected["raw_value"], int(selected["row_index"]), int(selected["column_index"]),
        float(selected["source_scale"]), float(selected["target_scale"]),
    ).replace("df1", variable, 1)


def plan_query(plan: dict) -> str:
    expression = plan["expression"]
    replacements = {}
    for index, fact_row in enumerate(plan["facts"], 1):
        replacements[fact_row["request"]["symbol"]] = cell_query(fact_row["selected"], f"df{index}")
    for symbol in sorted(replacements, key=len, reverse=True):
        expression = re.sub(rf"\b{symbol}\b", f"({replacements[symbol]})", expression)
    return expression


def replay_query(query: str, frames: dict[str, object]):
    import pandas as pd
    return eval(  # noqa: S307 - expression registry is source-controlled
        query, {"__builtins__": {}, "pd": pd, "float": float, "int": int,
                "str": str, "abs": abs}, frames,
    )


def load_plans(path: Path) -> list[dict]:
    plans = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    ids = [int(plan["question"]["id"]) for plan in plans]
    if ids != sorted(set(ids)) or not ids or not set(ids) <= set(L5_SPECS):
        raise ValueError("Plan ids must be a sorted non-empty subset of the L5 registry")
    return plans


def build_submission(plans_path: Path, database: Path, base_submission: Path,
                     output_dir: Path) -> Path:
    plans = load_plans(plans_path)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    submission = copy_base_submission(base_submission, output_dir)
    connection = sqlite3.connect(str(database))
    try:
        for plan in plans:
            question_id = int(plan["question"]["id"])
            evidence, docs, tables, frames = [], [], [], {}
            import pandas as pd
            for index, fact_row in enumerate(plan["facts"], 1):
                selected = fact_row["selected"]
                grid = connection.execute(
                    "SELECT grid_json FROM tables WHERE table_id = ?", (int(selected["table_id"]),)
                ).fetchone()
                relative = f"data/q{question_id:04d}_{fact_row['request']['symbol']}.csv"
                write_source_table_csv(output_dir / relative, json.loads(grid[0]))
                variable = f"df{index}"
                evidence.append({"variable": variable, "csv_path": relative})
                frames[variable] = pd.read_csv(output_dir / relative)
                if selected["document_id"] not in docs:
                    docs.append(selected["document_id"])
                table = f"{selected['document_id']}|{selected['source_line_1']}"
                if table not in tables:
                    tables.append(table)
            query = plan_query(plan)
            answer = replay_query(query, frames)
            if not math.isfinite(float(answer)):
                raise ValueError(f"q{question_id}: non-finite answer")
            submission.append({
                "id": question_id, "question": plan["question"]["question"],
                "answer": int(answer) if plan["answer_type"] == "int" else float(answer),
                "relevant_docs": docs, "relevant_tables": tables,
                "evidence": evidence, "pandas_query": query,
            })
    finally:
        connection.close()
    submission.sort(key=lambda row: int(row["id"]))
    submission_path = output_dir / "submission.json"
    submission_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        for path in sorted((output_dir / "data").glob("*.csv")):
            archive.write(path, path.relative_to(output_dir).as_posix())
    print(json.dumps({"submission": str(submission_path), "zip": str(zip_path),
                      "items": len(submission), "new_items": len(plans)}, ensure_ascii=False))
    return zip_path


def validate_submission(submission_path: Path, plans_path: Path,
                        base_submission: Path, zip_path: Optional[Path] = None) -> dict:
    """Replay every L5 query and verify the combined package is lossless."""

    import pandas as pd

    items = json.loads(submission_path.read_text(encoding="utf-8"))
    base_items = json.loads(base_submission.read_text(encoding="utf-8"))
    plans = load_plans(plans_path)
    by_id = {int(item["id"]): item for item in items}
    base_by_id = {int(item["id"]): item for item in base_items}
    errors: list[str] = []
    plan_ids = {int(plan["question"]["id"]) for plan in plans}
    expected_ids = set(base_by_id) | plan_ids
    if set(by_id) != expected_ids or len(items) != len(expected_ids):
        errors.append("submission ids do not equal base + L5 ids")
    for question_id, base_item in base_by_id.items():
        if by_id.get(question_id) != base_item:
            errors.append(f"q{question_id}: base item changed")

    replayed = 0
    for plan in plans:
        question_id = int(plan["question"]["id"])
        item = by_id.get(question_id)
        if item is None:
            errors.append(f"q{question_id}: missing")
            continue
        expected_query = plan_query(plan)
        if item.get("pandas_query") != expected_query:
            errors.append(f"q{question_id}: query differs from benchmark_locked plan")
            continue
        frames = {}
        for evidence in item.get("evidence", []):
            relative = evidence["csv_path"]
            path = submission_path.parent / relative
            if not relative.startswith("data/") or ".." in Path(relative).parts:
                errors.append(f"q{question_id}: unsafe evidence path {relative!r}")
                continue
            if not path.is_file():
                errors.append(f"q{question_id}: missing {relative}")
                continue
            frames[evidence["variable"]] = pd.read_csv(path)
        try:
            actual = replay_query(expected_query, frames)
            if not math.isclose(float(actual), float(item["answer"]),
                                rel_tol=1e-12, abs_tol=1e-8):
                errors.append(f"q{question_id}: replay {actual} != {item['answer']}")
            else:
                replayed += 1
        except (KeyError, IndexError, TypeError, ValueError, NameError) as error:
            errors.append(f"q{question_id}: replay failed: {error}")

    referenced = {
        evidence["csv_path"] for item in items for evidence in item.get("evidence", [])
    }
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as archive:
            members = set(archive.namelist())
        if members != {"submission.json", *referenced}:
            errors.append("ZIP members differ from evidence references")
    result = {
        "valid": not errors,
        "items": len(items),
        "base_items": len(base_items),
        "l5_items": len(plans),
        "l5_replayed": replayed,
        "evidence_files": len(referenced),
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l5a-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l5_manual_overrides.csv"))
    submission = sub.add_parser("submission")
    submission.add_argument("--plans", type=Path, default=Path("outputs/l5a-facts/plans.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-l3-l4-submission-final/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-l3-l4-l5a-submission"))
    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--plans", type=Path, default=Path("outputs/l5a-facts/plans.jsonl"))
    validate.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-l3-l4-submission-final/submission.json"))
    validate.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args(argv)
    if args.command == "candidates":
        run_candidates(args.questions, args.database, args.output_dir, args.overrides)
    elif args.command == "submission":
        build_submission(args.plans, args.database, args.base_submission, args.output_dir)
    else:
        print(json.dumps(validate_submission(
            args.submission, args.plans, args.base_submission, args.zip_path
        ), ensure_ascii=False))


if __name__ == "__main__":
    main()
