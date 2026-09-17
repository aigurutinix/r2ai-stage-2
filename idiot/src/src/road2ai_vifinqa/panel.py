"""Canonical financial panel and grounded formula registry."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .paths import PANEL_PATH
from .text import fold_text


RAW_COLUMNS = {
    "net_revenue": "kqkd:10",
    "cogs": "kqkd:11",
    "gross_profit": "kqkd:20",
    "interest_expense": "kqkd:23",
    "selling_expense": "kqkd:25",
    "admin_expense": "kqkd:26",
    "operating_profit": "kqkd:30",
    "pbt": "kqkd:50",
    "npat": "kqkd:60",
    "current_assets": "cdkt:100",
    "cash": "cdkt:110",
    "inventory": "cdkt:140",
    "long_term_assets": "cdkt:200",
    "total_assets": "cdkt:270",
    "liabilities": "cdkt:300",
    "current_liabilities": "cdkt:310",
    "equity": "cdkt:400",
    "cfo": "lctt:20",
}

ALIASES: dict[str, str] = {
    "hoa phat": "HPG",
    "hoa sen": "HSG",
    "nam kim": "NKG",
    "masan high tech": "MSR",
    "masan consumer": "MCH",
    "masan meatlife": "MML",
    "masan": "MSN",
    "dai duong": "OGC",
    "vinamilk": "VNM",
    "vietjet": "VJC",
    "vincom retail": "VRE",
    "vingroup": "VIC",
    "dam phu my": "DPM",
    "dam ca mau": "DCM",
    "do thi kinh bac": "KBC",
    "kinh bac": "KBC",
    "the gioi di dong": "MWG",
    "binh son": "BSR",
    "pvtrans": "PVT",
    "xang dau": "PLX",
    "sao mai": "ASM",
    "dabaco": "DBC",
    "minh phu": "MPC",
    "bao viet": "BVH",
    "hoang anh gia lai": "HAG",
    "dat xanh": "DXG",
    "nam long": "NLG",
    "hai phat": "HPX",
    "van phu": "VPI",
    "c e o": "CEO",
    "fpt telecom": "FOX",
}


# The MSR consolidated statements from 2020 through 2025 declare ``Nghìn
# VND`` for the complete statement set.  The prebuilt panel snapshot retained
# the printed numbers but recorded scale=1, shrinking every money value by
# 1,000.  Keep the correction next to panel loading so both calculations and
# emitted provenance receive the same normalized VND value.
_PANEL_VALUE_MULTIPLIERS: dict[tuple[str, int, str], float] = {
    (
        "MSR",
        year,
        f"MSR_financial_statements_{year}_consolidated",
    ): 1_000.0
    for year in range(2020, 2026)
}


def _panel_value_multiplier(ticker: str, year: int, item: dict[str, object]) -> float:
    multiplier = _PANEL_VALUE_MULTIPLIERS.get(
        (ticker, int(year), str(item.get("doc_id", ""))),
        1.0,
    )
    # Panel v2 resolves the source unit during construction.  Apply the
    # historical repair only to legacy cells which still claim base VND.
    if multiplier != 1.0 and float(item.get("scale", 1.0)) != 1.0:
        return 1.0
    return multiplier


@dataclass(frozen=True, slots=True)
class PanelCell:
    value: float
    doc_id: str
    table_id: int
    row_idx: int
    col_idx: int
    label: str
    raw: str

    @property
    def table_ref(self) -> str:
        return f"{self.doc_id}|table_{self.table_id}"


def _safe_div(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return numerator / denominator.replace(0, np.nan)


def enrich_panel(frame: pd.DataFrame) -> pd.DataFrame:
    """Add every ratio from the organisers' grounded formula registry."""

    df = frame.copy().sort_values(["ticker", "year"], kind="stable")
    df["gross_margin"] = _safe_div(df.gross_profit, df.net_revenue) * 100
    df["net_margin"] = _safe_div(df.npat, df.net_revenue) * 100
    df["operating_margin"] = _safe_div(df.operating_profit, df.net_revenue) * 100
    df["liabilities_to_equity"] = _safe_div(df.liabilities, df.equity)
    df["liabilities_to_assets"] = _safe_div(df.liabilities, df.total_assets)
    df["current_ratio"] = _safe_div(df.current_assets, df.current_liabilities)
    df["quick_ratio"] = _safe_div(df.current_assets - df.inventory, df.current_liabilities)
    df["asset_turnover"] = _safe_div(df.net_revenue, df.total_assets)
    df["interest_coverage"] = _safe_div(df.pbt + df.interest_expense, df.interest_expense)
    df["inventory_to_current_liabilities"] = _safe_div(df.inventory, df.current_liabilities)
    df["gross_minus_net_margin"] = _safe_div(df.gross_profit - df.npat, df.net_revenue) * 100
    df["operating_cash_flow_ratio"] = _safe_div(df.cfo, df.current_liabilities)
    df["cfo_margin"] = _safe_div(df.cfo, df.net_revenue) * 100
    df["cfo_minus_net_margin"] = _safe_div(df.cfo - df.npat, df.net_revenue) * 100
    df["inventory_to_assets"] = _safe_div(df.inventory, df.total_assets) * 100
    df["sga_expense"] = df.selling_expense + df.admin_expense
    df["sga_intensity"] = _safe_div(df.sga_expense, df.net_revenue) * 100
    df["long_term_assets_share"] = _safe_div(df.long_term_assets, df.total_assets) * 100
    df["cfo_to_npat"] = _safe_div(df.cfo, df.npat)
    df["operating_profit_to_pbt"] = _safe_div(df.operating_profit, df.pbt)
    df["cfo_to_operating_profit"] = _safe_div(df.cfo, df.operating_profit)

    grouped = df.groupby("ticker", sort=False)
    # The previous row is not necessarily the previous financial year. A
    # missing 2023 report must never turn a 2024 YoY calculation into 2021→2024.
    # Explicit comparative cells can fill such gaps before enrichment; absent
    # those cells, leave the rolling metric unknown rather than invent a year.
    consecutive = df.year.sub(grouped.year.shift(1)).eq(1)
    prior_assets = grouped.total_assets.shift(1).where(consecutive)
    prior_equity = grouped.equity.shift(1).where(consecutive)
    prior_inventory = grouped.inventory.shift(1).where(consecutive)
    prior_revenue = grouped.net_revenue.shift(1).where(consecutive)
    prior_operating_profit = grouped.operating_profit.shift(1).where(consecutive)
    avg_assets = (prior_assets + df.total_assets) / 2
    avg_equity = (prior_equity + df.equity) / 2
    avg_inventory = (prior_inventory + df.inventory) / 2
    df["roa"] = _safe_div(df.npat, avg_assets) * 100
    df["roe"] = _safe_div(df.npat, avg_equity) * 100
    df["inventory_days"] = _safe_div(avg_inventory, df.cogs) * 365
    df["asset_turnover_avg"] = _safe_div(df.net_revenue, avg_assets)
    df["equity_multiplier"] = _safe_div(avg_assets, avg_equity)
    df["net_working_capital"] = df.current_assets - df.current_liabilities
    df["operating_accruals_ratio"] = _safe_div(df.npat - df.cfo, avg_assets) * 100
    df["revenue_growth"] = _safe_div(df.net_revenue - prior_revenue, prior_revenue) * 100
    df["gross_margin_change"] = grouped.gross_margin.diff().where(consecutive)
    op_growth = _safe_div(df.operating_profit - prior_operating_profit, prior_operating_profit)
    rev_growth = _safe_div(df.net_revenue - prior_revenue, prior_revenue)
    df["dol"] = _safe_div(op_growth, rev_growth)
    return df


class FinancialPanel:
    def __init__(self, path: Path = PANEL_PATH) -> None:
        self.raw: dict[str, dict[str, dict[str, dict[str, object]]]] = json.loads(
            path.read_text(encoding="utf-8")
        )
        records: list[dict[str, object]] = []
        for ticker, years in self.raw.items():
            for year, metrics in years.items():
                record: dict[str, object] = {"ticker": ticker, "year": int(year)}
                for column, key in RAW_COLUMNS.items():
                    cell = metrics.get(key)
                    record[column] = (
                        float(cell["value"])
                        * _panel_value_multiplier(ticker, int(year), cell)
                        if cell is not None
                        else math.nan
                    )
                records.append(record)
        self.frame = enrich_panel(pd.DataFrame.from_records(records))
        self.tickers = frozenset(self.raw)

    def cell(self, ticker: str, year: int, raw_column: str) -> PanelCell | None:
        key = RAW_COLUMNS.get(raw_column, raw_column)
        item = self.raw.get(ticker, {}).get(str(year), {}).get(key)
        if item is None:
            return None
        return PanelCell(
            value=float(item["value"])
            * _panel_value_multiplier(ticker, int(year), item),
            doc_id=str(item["doc_id"]),
            table_id=int(item["table_id"]),
            row_idx=int(item["row_idx"]),
            col_idx=int(item["col_idx"]),
            label=str(item["label"]),
            raw=str(item["raw"]),
        )

    def subset(self, tickers: Iterable[str], years: Iterable[int], *, include_prior: bool = True) -> pd.DataFrame:
        ticker_set = set(tickers)
        year_set = set(years)
        if include_prior:
            year_set.update(year - 1 for year in list(year_set))
        return self.frame[
            self.frame.ticker.isin(ticker_set) & self.frame.year.isin(year_set)
        ].copy()


def infer_panel_tickers(question: str, available: Iterable[str]) -> list[str]:
    available_set = set(available)
    found: list[tuple[int, str]] = []
    for match in re.finditer(r"(?<![A-Z0-9])([A-Z][A-Z0-9]{1,4})(?![A-Z0-9])", question):
        ticker = match.group(1)
        if ticker in available_set:
            found.append((match.start(), ticker))
    folded = fold_text(question)
    for alias, ticker in ALIASES.items():
        if ticker not in available_set:
            continue
        position = folded.find(alias)
        if position >= 0:
            found.append((position, ticker))
    found.sort()
    return list(dict.fromkeys(ticker for _, ticker in found))


def infer_panel_years(question: str) -> list[int]:
    """Extract listed years and expand explicit inclusive year ranges."""

    years: list[int] = []
    for match in re.finditer(r"\b(20\d{2})\s*[-–—]\s*(20\d{2})\b", question):
        start, end = map(int, match.groups())
        step = 1 if end >= start else -1
        for year in range(start, end + step, step):
            if year not in years:
                years.append(year)
    for raw in re.findall(r"\b20\d{2}\b", question):
        year = int(raw)
        if year not in years:
            years.append(year)
    return years


PANEL_COLUMN_GUIDE = """
Raw VND columns: net_revenue, cogs, gross_profit, interest_expense,
selling_expense, admin_expense, operating_profit, pbt, npat,
current_assets, cash, inventory, long_term_assets, total_assets,
liabilities, current_liabilities, equity, cfo. Derived full-precision columns:
gross_margin, net_margin, operating_margin, liabilities_to_equity,
liabilities_to_assets, current_ratio, quick_ratio, asset_turnover,
interest_coverage, inventory_to_current_liabilities,
gross_minus_net_margin, operating_cash_flow_ratio, cfo_margin,
cfo_minus_net_margin, inventory_to_assets, sga_expense, sga_intensity,
long_term_assets_share, cfo_to_npat, operating_profit_to_pbt,
cfo_to_operating_profit, roa, roe, inventory_days, asset_turnover_avg,
equity_multiplier, net_working_capital, operating_accruals_ratio,
revenue_growth, gross_margin_change, dol.
""".strip()
