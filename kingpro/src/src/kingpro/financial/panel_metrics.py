"""Auditable panel metrics computed from :mod:`statement_cube` values."""

from __future__ import annotations

from dataclasses import dataclass

from kingpro.financial.statement_cube import FinancialCube


RATIO_FORMULAS: dict[str, tuple[dict[str, float], dict[str, float], float]] = {
    "gross_margin_pct": ({"kqkd:20": 1}, {"kqkd:10": 1}, 100),
    "net_margin_pct": ({"kqkd:60": 1}, {"kqkd:10": 1}, 100),
    "operating_margin_pct": ({"kqkd:30": 1}, {"kqkd:10": 1}, 100),
    "liabilities_to_equity": ({"cdkt:300": 1}, {"cdkt:400": 1}, 1),
    "debt_to_assets_pct": ({"cdkt:300": 1}, {"cdkt:270": 1}, 100),
    "current_ratio": ({"cdkt:100": 1}, {"cdkt:310": 1}, 1),
    "quick_ratio": ({"cdkt:100": 1, "cdkt:140": -1}, {"cdkt:310": 1}, 1),
    "interest_coverage": ({"kqkd:50": 1, "kqkd:23": 1}, {"kqkd:23": 1}, 1),
    "inventory_to_current_liabilities": ({"cdkt:140": 1}, {"cdkt:310": 1}, 1),
    "operating_cash_flow_ratio": ({"lctt:20": 1}, {"cdkt:310": 1}, 1),
    "cfo_margin_pct": ({"lctt:20": 1}, {"kqkd:10": 1}, 100),
    "inventory_to_assets_pct": ({"cdkt:140": 1}, {"cdkt:270": 1}, 100),
    "liabilities_to_assets_pct": ({"cdkt:300": 1}, {"cdkt:270": 1}, 100),
    "sga_intensity_pct": ({"kqkd:25": 1, "kqkd:26": 1}, {"kqkd:10": 1}, 100),
    "long_term_assets_share_pct": ({"cdkt:200": 1}, {"cdkt:270": 1}, 100),
    "cfo_to_npat": ({"lctt:20": 1}, {"kqkd:60": 1}, 1),
    "operating_profit_to_pbt": ({"kqkd:30": 1}, {"kqkd:50": 1}, 1),
    "cfo_to_operating_profit": ({"lctt:20": 1}, {"kqkd:30": 1}, 1),
    "npat_to_assets_pct": ({"kqkd:60": 1}, {"cdkt:270": 1}, 100),
    "current_assets_to_assets_pct": ({"cdkt:100": 1}, {"cdkt:270": 1}, 100),
    "pbt_margin_pct": ({"kqkd:50": 1}, {"kqkd:10": 1}, 100),
    "admin_expense_to_revenue_pct": ({"kqkd:26": 1}, {"kqkd:10": 1}, 100),
    "selling_expense_to_revenue_pct": ({"kqkd:25": 1}, {"kqkd:10": 1}, 100),
    "short_term_borrowings_to_equity_pct": ({"cdkt:320": 1}, {"cdkt:400": 1}, 100),
    "current_liabilities_to_equity_pct": ({"cdkt:310": 1}, {"cdkt:400": 1}, 100),
    "current_liabilities_to_equity": ({"cdkt:310": 1}, {"cdkt:400": 1}, 1),
    "finance_expense_to_revenue_pct": ({"kqkd:22": 1}, {"kqkd:10": 1}, 100),
    "finance_revenue_to_expense_pct": ({"kqkd:21": 1}, {"kqkd:22": 1}, 100),
    "cogs_to_revenue_pct": ({"kqkd:11": 1}, {"kqkd:10": 1}, 100),
    "short_term_investments_to_cash_pct": ({"cdkt:120": 1}, {"cdkt:110": 1}, 100),
}

RAW_METRICS: dict[str, str] = {
    "revenue": "kqkd:10",
    "cogs": "kqkd:11",
    "gross_profit": "kqkd:20",
    "finance_revenue": "kqkd:21",
    "finance_expense": "kqkd:22",
    "interest_expense": "kqkd:23",
    "selling_expense": "kqkd:25",
    "admin_expense": "kqkd:26",
    "other_expense": "kqkd:32",
    "operating_profit": "kqkd:30",
    "pbt": "kqkd:50",
    "npat": "kqkd:60",
    "current_tax_expense": "kqkd:51",
    "eps": "kqkd:70",
    "cfo": "lctt:20",
    "current_assets": "cdkt:100",
    "cash": "cdkt:110",
    "inventory": "cdkt:140",
    "short_term_receivables": "cdkt:130",
    "long_term_receivables": "cdkt:210",
    "long_term_assets": "cdkt:200",
    "tangible_fixed_assets": "cdkt:221",
    "total_assets": "cdkt:270",
    "liabilities": "cdkt:300",
    "current_liabilities": "cdkt:310",
    "short_term_borrowings": "cdkt:320",
    "equity": "cdkt:400",
    "contributed_capital": "cdkt:411",
}


@dataclass(slots=True)
class PanelMetricEngine:
    cube: FinancialCube
    scope: str = "consolidated"

    def raw(self, ticker: str, year: str | int, metric_key: str) -> float | None:
        cell = self.cube.cell(ticker, year, metric_key, self.scope)
        return None if cell is None else cell.value

    def _linear(self, ticker: str, year: str | int, terms: dict[str, float]) -> float | None:
        total = 0.0
        for key, coefficient in terms.items():
            value = self.raw(ticker, year, key)
            if value is None:
                return None
            total += coefficient * value
        return total

    def value(self, ticker: str, year: str | int, metric: str) -> float | None:
        if metric in RAW_METRICS:
            return self.raw(ticker, year, RAW_METRICS[metric])
        if metric in RATIO_FORMULAS:
            numerator, denominator, multiplier = RATIO_FORMULAS[metric]
            top = self._linear(ticker, year, numerator)
            bottom = self._linear(ticker, year, denominator)
            if top is None or bottom in (None, 0):
                return None
            return top / bottom * multiplier

        current_year = int(year)
        previous_year = current_year - 1
        if metric == "revenue_growth_pct":
            current = self.raw(ticker, current_year, "kqkd:10")
            previous = self.raw(ticker, previous_year, "kqkd:10")
            if current is None or previous in (None, 0):
                return None
            return (current / previous - 1) * 100
        if metric == "gross_margin_change_pp":
            current = self.value(ticker, current_year, "gross_margin_pct")
            previous = self.value(ticker, previous_year, "gross_margin_pct")
            return None if current is None or previous is None else current - previous

        current_assets = self.raw(ticker, current_year, "cdkt:270")
        previous_assets = self.raw(ticker, previous_year, "cdkt:270")
        current_equity = self.raw(ticker, current_year, "cdkt:400")
        previous_equity = self.raw(ticker, previous_year, "cdkt:400")
        npat = self.raw(ticker, current_year, "kqkd:60")
        revenue = self.raw(ticker, current_year, "kqkd:10")
        if metric == "roa_pct":
            if npat is None or current_assets is None or previous_assets is None:
                return None
            average = (current_assets + previous_assets) / 2
            return None if average == 0 else npat / average * 100
        if metric == "roe_pct":
            if npat is None or current_equity is None or previous_equity is None:
                return None
            average = (current_equity + previous_equity) / 2
            return None if average == 0 else npat / average * 100
        if metric == "asset_turnover_avg":
            if revenue is None or current_assets is None or previous_assets is None:
                return None
            average = (current_assets + previous_assets) / 2
            return None if average == 0 else revenue / average
        if metric == "inventory_days":
            inventory = self.raw(ticker, current_year, "cdkt:140")
            previous_inventory = self.raw(ticker, previous_year, "cdkt:140")
            cogs = self.raw(ticker, current_year, "kqkd:11")
            if inventory is None or previous_inventory is None or cogs in (None, 0):
                return None
            return ((inventory + previous_inventory) / 2) / abs(cogs) * 365
        if metric == "net_working_capital":
            current = self.raw(ticker, current_year, "cdkt:100")
            liabilities = self.raw(ticker, current_year, "cdkt:310")
            return None if current is None or liabilities is None else current - liabilities
        if metric == "operating_accruals_ratio_pct":
            cfo = self.raw(ticker, current_year, "lctt:20")
            if npat is None or cfo is None or current_assets is None or previous_assets is None:
                return None
            average = (current_assets + previous_assets) / 2
            return None if average == 0 else (npat - cfo) / average * 100
        if metric == "sga_expense":
            selling = self.raw(ticker, current_year, "kqkd:25")
            admin = self.raw(ticker, current_year, "kqkd:26")
            return None if selling is None or admin is None else selling + admin
        if metric in {"fixed_assets_avg", "fixed_assets_turnover"}:
            current_fixed = self.raw(ticker, current_year, "cdkt:220")
            previous_fixed = self.raw(ticker, previous_year, "cdkt:220")
            if current_fixed is None or previous_fixed is None:
                return None
            average_fixed = (current_fixed + previous_fixed) / 2
            if metric == "fixed_assets_avg":
                return average_fixed
            return None if revenue is None or average_fixed == 0 else revenue / average_fixed
        if metric == "operating_leverage":
            operating_profit = self.raw(ticker, current_year, "kqkd:30")
            previous_operating_profit = self.raw(ticker, previous_year, "kqkd:30")
            growth = self.value(ticker, current_year, "revenue_growth_pct")
            if operating_profit is None or previous_operating_profit in (None, 0) or growth in (None, 0):
                return None
            operating_profit_growth_pct = (operating_profit / previous_operating_profit - 1) * 100
            return operating_profit_growth_pct / growth
        if metric == "equity_turnover_avg":
            if revenue is None or current_equity is None or previous_equity is None:
                return None
            average_equity = (current_equity + previous_equity) / 2
            return None if average_equity == 0 else revenue / average_equity
        return None

    def row(self, ticker: str, year: str | int) -> dict[str, float | str | None]:
        metrics = [
            *RAW_METRICS,
            *RATIO_FORMULAS,
            "revenue_growth_pct",
            "gross_margin_change_pp",
            "roa_pct",
            "roe_pct",
            "asset_turnover_avg",
            "inventory_days",
            "net_working_capital",
            "operating_accruals_ratio_pct",
            "sga_expense",
            "fixed_assets_avg",
            "fixed_assets_turnover",
            "operating_leverage",
            "equity_turnover_avg",
        ]
        return {"ticker": ticker, "year": str(year), **{key: self.value(ticker, year, key) for key in metrics}}
