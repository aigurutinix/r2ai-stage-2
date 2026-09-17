"""Deterministic answers for the generated panel rephrase block (451-494)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.financial.panel_metrics import PanelMetricEngine
from kingpro.financial.statement_cube import FinancialCube


QUESTIONS = {
    int(row["id"]): row["question"]
    for row in (
        json.loads(line)
        for line in (ROOT / "data/questions/questions.jsonl").read_text(encoding="utf-8").splitlines()
    )
}
ENGINE = PanelMetricEngine(FinancialCube.read_jsonl(ROOT / "build/statement_cube.jsonl"))


def panel(tickers: list[str], years: list[int]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        for year in years:
            row = ENGINE.row(ticker, year)
            row["year"] = year
            rows.append(row)
    return pd.DataFrame(rows)


def persistent(frame: pd.DataFrame, column: str, predicate) -> list[str]:
    valid = frame.dropna(subset=[column]).groupby("ticker").filter(
        lambda part: len(part) == frame["year"].nunique() and predicate(part[column]).all()
    )
    return sorted(valid["ticker"].unique())


def answer(qid: int) -> tuple[float, str]:
    if qid == 451:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2022, 2024])
        base = f[f.year == 2022]; keep = base[base.inventory_days > base.inventory_days.median()].ticker
        p = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="gross_margin_pct")
        return (p[2024] - p[2022]).mean(), "mean(gross_margin_2024 - gross_margin_2022) after inventory_days_2022 > median"
    if qid == 452:
        f = panel(["ACV", "DLG", "HHV", "VSC"], [2024]); keep = f.current_ratio < f.current_ratio.median()
        return f.loc[keep, "gross_margin_pct"].mean(), "mean gross margin where current ratio < group median"
    if qid == 453:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2022]); keep = f.quick_ratio < f.quick_ratio.median()
        return f.loc[keep, "net_margin_pct"].mean(), "mean net margin where quick ratio < group median"
    if qid == 454:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2022, 2023]); b = f[f.year == 2022]
        keep = b[b.quick_ratio < b.quick_ratio.median()].ticker
        p = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="gross_margin_pct")
        ticker = (p[2023] - p[2022]).idxmax()
        return f[(f.ticker == ticker) & (f.year == 2023)].interest_coverage.iloc[0], "interest coverage 2023 of filtered ticker with max gross-margin increase"
    if qid == 455:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2022, 2024]); b = f[f.year == 2022]
        keep = b[b.inventory_days > b.inventory_days.median()].ticker
        p = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="inventory_days")
        ticker = (p[2024] - p[2022]).idxmin()
        return f[(f.ticker == ticker) & (f.year == 2024)].gross_margin_pct.iloc[0], "gross margin 2024 of filtered ticker with largest inventory-days decrease"
    if qid == 456:
        f = panel(["KBC"], list(range(2016, 2022))); year = int(f[f.cfo < 0].year.min()) + 1
        return f[f.year == year].gross_margin_pct.iloc[0], "KBC gross margin in year after first negative CFO"
    if qid == 457:
        f = panel(["HPX", "NVL", "SCR", "VIC", "VRE"], [2024])
        return float(((f.current_assets < f.current_liabilities) & (f.cfo > 0)).sum()), "count current assets < current liabilities and CFO > 0"
    if qid == 458:
        f = panel(["CEO", "HPX", "KBC", "SNZ", "VIC", "VPI", "VRE"], [2022]); keep = f.inventory_to_current_liabilities > f.inventory_to_current_liabilities.median()
        return f.loc[keep, "current_liabilities"].sum() / f.current_liabilities.sum() * 100, "filtered current-liabilities share of full group"
    if qid == 459:
        f = panel(["KBC"], list(range(2016, 2021))); row = f.loc[f.liabilities_to_equity.idxmax()]
        return row.interest_coverage, "interest coverage in KBC max D/E year"
    if qid == 460:
        f = panel(["DIG", "KBC", "NVL", "SCR", "VRE"], [2016]); med = f.liabilities_to_equity.median()
        return f.loc[f.liabilities_to_equity > med, "interest_expense"].sum() / f.loc[f.liabilities_to_equity <= med, "interest_expense"].sum(), "interest-expense sum ratio above versus at-or-below median D/E"
    if qid == 461:
        f = panel(["BSR", "PLX", "PVT"], [2017]); score = (f.cfo - f.npat) / f.revenue
        return f.loc[score.idxmax(), "liabilities_to_equity"], "D/E of ticker with max (CFO-NPAT)/revenue"
    if qid == 462:
        f = panel(["DLG", "HHV", "VSC"], [2019, 2020]); now = f[(f.year == 2020) & (f.current_assets < f.current_liabilities)]
        ticker = now.loc[now.liabilities_to_assets_pct.idxmin(), "ticker"]; one = f[f.ticker == ticker].set_index("year")
        return one.loc[2020, "npat"] / ((one.loc[2019, "total_assets"] + one.loc[2020, "total_assets"]) / 2) * 100, "ROA using average assets for lowest debt/assets filtered ticker"
    if qid == 463:
        f = panel(["BSR", "PLX", "PVT"], [2021, 2022]); sga = f.pivot(index="ticker", columns="year", values="sga_expense"); rev = f.pivot(index="ticker", columns="year", values="revenue")
        return float((((sga[2022] / sga[2021]) - 1) > ((rev[2022] / rev[2021]) - 1)).sum()), "count SGA growth greater than revenue growth"
    if qid == 464:
        tickers = sorted(ENGINE.cube.data); f = panel(tickers, [2015, 2016]); inv = f.pivot(index="ticker", columns="year", values="inventory").dropna(); keep = inv[(inv[2016] / inv[2015] - 1) <= -0.10].index
        return f[(f.year == 2016) & f.ticker.isin(keep)].cfo_margin_pct.max(), "max 2016 CFO margin among companies with inventory decline at least 10%"
    if qid == 465:
        f = panel(["BSR", "PLX", "PVT"], [2019]); ticker = f.loc[f.liabilities_to_equity.idxmax(), "ticker"]
        return f[f.ticker == ticker].interest_coverage.iloc[0], "interest coverage of max D/E ticker"
    if qid == 466:
        f = panel(["ASM", "DBC", "MCH", "MML", "MPC", "MSN", "OGC", "QNS", "VNM", "VSF"], [2022]); med = f.liabilities_to_equity.median(); profitable = f.npat > 0
        return f.loc[(f.liabilities_to_equity < med) & profitable, "npat"].sum() / f.loc[profitable, "npat"].sum() * 100, "positive-NPAT contribution from below-median D/E group"
    if qid == 467:
        f = panel(["CEO", "DIG", "IJC", "KBC", "NVL", "SCR", "VIC", "VRE"], [2016]); top = f.nlargest(3, "gross_margin_pct")
        return top.cash.sum() / f.cash.sum() * 100, "cash share held by top-three gross-margin companies"
    if qid in (468, 487):
        f = panel(["DLG", "HHV", "VSC"], [2019, 2020]); p = f.pivot(index="ticker", columns="year"); positive = p["npat"][2020] > 0
        ratio = (p["npat"][2020] - p["cfo"][2020]) / ((p["total_assets"][2019] + p["total_assets"][2020]) / 2) * 100
        return ratio[positive].mean(), "mean (NPAT-CFO)/average-assets for positive-NPAT companies"
    if qid == 469:
        f = panel(["AAA", "DCM", "GVR", "PRT"], [2017])
        return f.loc[f.current_ratio >= 1, "inventory_to_current_liabilities"].mean(), "mean inventory/current-liabilities where current ratio >= 1"
    if qid == 470:
        f = panel(["AAA", "DCM", "GVR", "PRT"], [2017])
        return f.loc[f.current_ratio >= 1, "quick_ratio"].mean(), "mean quick ratio where current ratio >= 1"
    if qid in (471, 488):
        f = panel(["AAA", "DCM", "DPM", "GVR"], [2016])
        return f.loc[f.net_margin_pct > 10, "revenue"].sum() / 1e12, "revenue sum in trillion VND where net margin > 10%"
    if qid in (472, 489):
        f = panel(["AAA", "DCM", "DPM", "GVR", "PRT"], [2020, 2021, 2022]); keep = persistent(f, "cfo", lambda s: s > 0)
        return f[(f.year == 2022) & f.ticker.isin(keep)].net_margin_pct.max(), "max 2022 net margin among tickers with positive CFO all three years"
    if qid in (473, 490):
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2020, 2021, 2022]); keep = persistent(f, "net_margin_pct", lambda s: s > 0)
        return f[(f.year == 2022) & f.ticker.isin(keep)].revenue.sum() / 1e12, "2022 revenue sum for tickers with positive net margin all three years"
    if qid == 474:
        f = panel(["ASM"], [2016, 2017, 2018])
        return f.loc[f.net_margin_pct > 10, "revenue"].min() / 1e9, "minimum ASM revenue where net margin > 10%"
    if qid in (475, 479):
        start, end = ((2021, 2022) if qid == 475 else (2022, 2023)); f = panel(["HPG", "HSG", "MSR", "NKG"], [start, end]); inv = f.pivot(index="ticker", columns="year", values="inventory_days"); ticker = (inv[end] - inv[start]).idxmax(); gm = f[f.ticker == ticker].set_index("year").gross_margin_pct
        return gm[end] - gm[start], "gross-margin change for ticker with largest inventory-days increase"
    if qid in (476, 491):
        f = panel(["BSR", "PLX", "PVT"], [2021, 2022]); rev = f.pivot(index="ticker", columns="year", values="revenue"); ticker = (rev[2022] / rev[2021] - 1).idxmax(); sga = f[f.ticker == ticker].set_index("year").sga_expense
        return (sga[2022] / sga[2021] - 1) * 100, "SGA growth of ticker with highest revenue growth"
    if qid in (477, 492, 494):
        year = 2019 if qid == 494 else 2017; f = panel(["BSR", "PLX", "PVT"], [year]); eligible = f if qid == 477 else f[f.operating_profit > 0]; ticker = eligible.loc[(eligible.cfo / eligible.operating_profit).idxmin(), "ticker"]
        return f[f.ticker == ticker].net_margin_pct.iloc[0], "net margin of ticker with minimum CFO/operating-profit"
    if qid in (478, 493):
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2021, 2022, 2023]); keep = persistent(f, "net_margin_pct", lambda s: s > 0)
        return f[(f.year == 2023) & f.ticker.isin(keep)].revenue.sum() / 1e12, "2023 revenue sum for tickers with positive net margin all three years"
    if qid == 480:
        f = panel(["DCM", "DPM", "PRT"], [2019, 2020]); rev = f.pivot(index="ticker", columns="year", values="revenue"); keep = rev[rev[2020] > rev[2019]].index; gm = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="gross_margin_pct")
        return (gm[2020] - gm[2019]).mean(), "mean gross-margin change among revenue-growing tickers"
    if qid == 481:
        f = panel(["CEO"], [2022, 2023, 2024])
        return f.loc[f.net_margin_pct > 10, "revenue"].min() / 1e9, "minimum CEO revenue where net margin > 10%"
    if qid == 482:
        f = panel(["DCM", "DPM", "GVR", "PRT"], [2021, 2022]); rev = f.pivot(index="ticker", columns="year", values="revenue"); keep = rev[rev[2022] > rev[2021]].index; gm = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="gross_margin_pct")
        return (gm[2022] - gm[2021]).mean(), "mean gross-margin change among revenue-growing tickers"
    if qid in (483, 484):
        tickers = ["DCM", "DPM", "PRT"] if qid == 483 else ["DCM", "DPM", "GVR", "PRT"]; start, end = ((2019, 2020) if qid == 483 else (2020, 2021)); f = panel(tickers, [start, end]); keep = persistent(f, "cfo", lambda s: s > 0); rev = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="revenue")
        return ((rev[end] / rev[start] - 1) * 100).mean(), "mean revenue growth for tickers with positive CFO in both years"
    if qid == 485:
        f = panel(["DCM"], [2020, 2021, 2022]); row = f[f.net_margin_pct > 10].sort_values("revenue").iloc[0]
        return row.operating_cash_flow_ratio, "CFO/current-liabilities in lowest-revenue qualifying year"
    if qid == 486:
        f = panel(["DLG", "HHV", "VSC"], [2020])
        return f.loc[f.current_ratio < 1, "operating_cash_flow_ratio"].mean(), "mean CFO/current-liabilities where current ratio < 1"
    if qid == 497:
        f = panel(["VIC", "DXG", "SCR", "CEO"], [2017]); row = f.loc[f.current_tax_expense.idxmax()]
        return row.cogs / row.inventory, "COGS/inventory of maximum current-tax-expense company"
    if qid == 504:
        f = panel(["ASM"], [2022, 2024, 2025]); row = f.loc[f.short_term_borrowings.idxmax()]
        return row.cfo / row.revenue * 100, "ASM CFO/revenue in maximum short-term-borrowings year"
    if qid == 506:
        f = panel(["IJC", "DXG", "NVL", "NLG", "KBC"], [2024]); row = f.loc[f.tangible_fixed_assets.idxmax()]
        return row.revenue / 1e12, "revenue of maximum tangible-fixed-assets company, VND tn"
    if qid == 536:
        f = panel(["GVR", "DPM", "HT1", "NKG"], [2023]); row = f.loc[f.equity.idxmax()]
        return row.current_tax_expense / 1e9, "current tax expense of maximum-equity company, VND bn"
    if qid == 539:
        f = panel(["BSR", "PLX", "PVT"], [2019]); ticker = f.loc[f.liabilities_to_equity.idxmax(), "ticker"]
        return f[f.ticker == ticker].interest_coverage.iloc[0], "interest coverage of maximum D/E ticker"
    if qid in (540, 554, 576):
        f = panel(["DCM", "DPM", "PRT"], [2019, 2020]); rev = f.pivot(index="ticker", columns="year", values="revenue"); keep = rev[rev[2020] > rev[2019]].index; gm = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="gross_margin_pct")
        return (gm[2020] - gm[2019]).mean(), "mean gross-margin change among revenue-growing tickers"
    if qid in (541, 555, 577):
        f = panel(["CEO"], [2022, 2023, 2024])
        return f.loc[f.net_margin_pct > 10, "revenue"].min() / 1e12, "minimum CEO revenue above 10% net margin, VND tn"
    if qid in (542, 557):
        f = panel(["DCM", "DPM", "GVR", "PRT"], [2021, 2022]); rev = f.pivot(index="ticker", columns="year", values="revenue"); keep = rev[rev[2022] > rev[2021]].index; gm = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="gross_margin_pct")
        return (gm[2022] - gm[2021]).mean(), "mean gross-margin change among 2022 revenue growers"
    if qid in (543, 558):
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2024, 2025]); inv = f.pivot(index="ticker", columns="year", values="inventory_days"); ticker = (inv[2025] - inv[2024]).idxmax(); gm = f[f.ticker == ticker].set_index("year").gross_margin_pct
        return gm[2025] - gm[2024], "gross-margin change for max 2024-2025 inventory-days increase"
    if qid in (544, 561):
        f = panel(["DCM"], [2020, 2021, 2022]); row = f[f.net_margin_pct > 10].sort_values("revenue").iloc[0]
        return row.operating_cash_flow_ratio, "DCM CFO/current-liabilities in lowest-revenue qualifying year"
    if qid == 545:
        f = panel(["BSR", "PLX", "PVT"], [2017]); eligible = f[f.operating_profit > 0]; ticker = eligible.loc[(eligible.cfo / eligible.operating_profit).idxmin(), "ticker"]
        return f[f.ticker == ticker].net_margin_pct.iloc[0], "net margin of positive-operating-profit ticker with minimum CFO/operating-profit"
    if qid in (546, 564):
        f = panel(["DLG", "HHV", "VSC"], [2020])
        return f.loc[f.current_ratio < 1, "operating_cash_flow_ratio"].mean(), "mean CFO/current-liabilities where current ratio < 1"
    if qid == 547:
        f = panel(["AAA", "DCM", "DPM", "GVR", "PRT"], [2020, 2021, 2022]); keep = persistent(f, "cfo", lambda s: s > 0)
        return f[(f.year == 2022) & f.ticker.isin(keep)].net_margin_pct.max(), "max 2022 net margin among persistent positive-CFO tickers"
    if qid == 548:
        f = panel(["ASM"], [2016, 2017, 2018])
        return f.loc[f.net_margin_pct > 10, "revenue"].min() / 1e12, "minimum ASM revenue above 10% net margin, VND tn"
    if qid in (549, 573):
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2021, 2022, 2023]); row = f.loc[f.revenue_growth_pct.idxmax()]
        return row.revenue / row.total_assets, "revenue/ending-assets for maximum company-year revenue growth"
    if qid == 550:
        f = panel(["ACV", "DLG", "HHV"], [2024])
        return f.loc[f.current_ratio < 1, "operating_cash_flow_ratio"].mean(), "mean CFO/current-liabilities where current ratio < 1"
    if qid in (551, 574):
        f = panel(["GEE", "GEX", "SAM"], [2022, 2023, 2024]); keep = persistent(f, "cfo", lambda s: s > 0)
        return f[(f.year == 2024) & f.ticker.isin(keep)].net_margin_pct.max(), "max 2024 net margin among persistent positive-CFO tickers"
    if qid == 552:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2021, 2022, 2023]); keep = persistent(f, "net_margin_pct", lambda s: s > 0)
        return f[(f.year == 2023) & f.ticker.isin(keep)].revenue.sum() / 1e12, "2023 revenue sum for persistent positive-net-margin tickers"
    if qid == 553:
        f = panel(["BSR", "PLX", "PVT"], [2019]); ticker = f.loc[f.liabilities_to_equity.idxmax(), "ticker"]
        return f[f.ticker == ticker].interest_coverage.iloc[0], "interest coverage of maximum D/E ticker"
    if qid == 556:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2023, 2024]); inv = f.pivot(index="ticker", columns="year", values="inventory_days"); ticker = (inv[2024] - inv[2023]).idxmax(); gm = f[f.ticker == ticker].set_index("year").gross_margin_pct
        return gm[2024] - gm[2023], "gross-margin change for max 2023-2024 inventory-days increase"
    if qid == 559:
        f = panel(["DCM", "DPM", "PRT"], [2019, 2020]); keep = persistent(f, "cfo", lambda s: s > 0); rev = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="revenue")
        return ((rev[2020] / rev[2019] - 1) * 100).mean(), "mean revenue growth for persistent positive-CFO tickers"
    if qid == 560:
        f = panel(["DCM", "DPM", "GVR", "PRT"], [2020, 2021]); keep = persistent(f, "cfo", lambda s: s > 0); rev = f[f.ticker.isin(keep)].pivot(index="ticker", columns="year", values="revenue")
        return ((rev[2021] / rev[2020] - 1) * 100).mean(), "mean revenue growth for persistent positive-CFO tickers"
    if qid == 562:
        f = panel(["DCM"], [2022, 2023, 2024]); row = f[f.net_margin_pct > 10].sort_values("revenue").iloc[0]
        return row.operating_cash_flow_ratio, "DCM 2022-2024 CFO/current-liabilities in lowest-revenue qualifying year"
    if qid == 563:
        f = panel(["GEE", "GEX", "SAM"], [2020, 2021]); rev = f.pivot(index="ticker", columns="year", values="revenue"); gm = f.pivot(index="ticker", columns="year", values="gross_margin_pct")
        return float(((rev[2021] > rev[2020]) & (gm[2021] < gm[2020])).sum()), "count revenue-up and gross-margin-down tickers"
    if qid == 565:
        f = panel(["DLG", "HHV", "VSC"], [2019, 2020]); p = f.pivot(index="ticker", columns="year"); positive = p["npat"][2020] > 0; ratio = (p["npat"][2020] - p["cfo"][2020]) / ((p["total_assets"][2019] + p["total_assets"][2020]) / 2) * 100
        return ratio[positive].mean(), "mean accruals ratio for positive-NPAT tickers"
    if qid == 566:
        f = panel(["AAA", "DCM", "GVR", "PRT"], [2017])
        return f.loc[f.current_ratio >= 1, "inventory_to_current_liabilities"].mean(), "mean inventory/current-liabilities where current ratio >= 1"
    if qid == 567:
        f = panel(["AAA", "DCM", "DPM", "GVR"], [2016])
        return f.loc[f.net_margin_pct > 10, "revenue"].sum() / 1e12, "revenue sum where net margin > 10%, VND tn"
    if qid == 568:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2020, 2021, 2022]); keep = persistent(f, "net_margin_pct", lambda s: s > 0)
        return f[(f.year == 2022) & f.ticker.isin(keep)].revenue.sum() / 1e12, "2022 revenue sum for persistent positive-net-margin tickers"
    if qid == 569:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2022]); eligible = f[f.net_margin_pct > 0]
        return (eligible.gross_margin_pct - eligible.net_margin_pct).mean(), "mean gross-minus-net margin among positive-net-margin tickers"
    if qid == 570:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2021, 2022]); inv = f.pivot(index="ticker", columns="year", values="inventory_days"); ticker = (inv[2022] - inv[2021]).idxmax(); gm = f[f.ticker == ticker].set_index("year").gross_margin_pct
        return gm[2022] - gm[2021], "gross-margin change for max 2021-2022 inventory-days increase"
    if qid == 571:
        f = panel(["BSR", "PLX", "PVT"], [2021, 2022]); rev = f.pivot(index="ticker", columns="year", values="revenue"); ticker = (rev[2022] / rev[2021] - 1).idxmax(); sga = f[f.ticker == ticker].set_index("year").sga_expense
        return (sga[2022] / sga[2021] - 1) * 100, "SGA growth of maximum revenue-growth ticker"
    if qid == 572:
        f = panel(["GEE", "GEX", "SAM"], [2022, 2023, 2024]); keep = persistent(f, "cfo", lambda s: s > 0); eligible = f[(f.year == 2024) & f.ticker.isin(keep)]; ticker = eligible.loc[eligible.revenue_growth_pct.idxmax(), "ticker"]
        return eligible[eligible.ticker == ticker].roa_pct.iloc[0], "2024 ROA of persistent positive-CFO ticker with max revenue growth"
    if qid == 575:
        f = panel(["HPG", "HSG", "MSR", "NKG"], [2022, 2023]); inv = f.pivot(index="ticker", columns="year", values="inventory_days"); ticker = (inv[2023] - inv[2022]).idxmax(); gm = f[f.ticker == ticker].set_index("year").gross_margin_pct
        return gm[2023] - gm[2022], "gross-margin change for max 2022-2023 inventory-days increase"
    raise KeyError(qid)


def main() -> None:
    rows = []
    for qid in [*range(451, 495), 497, 504, 506, 536, 539, *range(540, 578)]:
        value, trace = answer(qid)
        rows.append({"id": qid, "ok": True, "question": QUESTIONS[qid], "answer": round(float(value), 2), "code": trace})
    out = ROOT / "build/panel_rephrases_deterministic.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    for row in rows:
        print(f"{row['id']}\t{row['answer']}\t{row['code']}")


if __name__ == "__main__":
    main()
