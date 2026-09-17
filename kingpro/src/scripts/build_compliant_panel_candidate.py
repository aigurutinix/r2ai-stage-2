"""Make the proven panel overrides execute entirely from original BTC CSVs."""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from kingpro.financial.panel_metrics import RAW_METRICS
from kingpro.financial.statement_cube import FinancialCube, StatementCell
import solve_panel_rephrases as rephrases


CUBE = FinancialCube.read_jsonl(ROOT / "build" / "statement_cube.jsonl")
COST_KEYS = frozenset(f"kqkd:{code}" for code in ("11", "22", "23", "25", "26", "32", "51", "52"))
HIGH_CONFIDENCE_IDS = frozenset({
    362, 363, 364, 365, 366, 368, 369, 370, 371, 372,
    375, 378, 379, 381, 382, 387, 388, 390, 394, 397,
    398, 404, 405, 407, 410, 411, 415, 416, 420,
})
MINIMAL_DEPENDENCY_FALLBACK_IDS = frozenset({429, 432, 506})
MINIMAL_SEMANTIC_COLUMNS = {
    # Revenue algebraically cancels from the stated constant-margin scenario,
    # but retaining the direct input keeps the trace faithful to the question
    # and preserves the metric-code audit without adding a new table family.
    435: ("revenue",),
}
AUDITED_PANEL_FIXES = {
    # Legacy single-company programs for these panel questions ignored most of
    # the named peer group.  Each repair below computes the selector and target
    # metric from every named ticker/year at grader runtime.
    367: (["MSN", "MCH", "DBC", "ASM", "OGC"], [2024, 2025], """cfo = df.pivot(index='ticker', columns='year', values='cfo').dropna()
revenue = df.pivot(index='ticker', columns='year', values='revenue').dropna()
common = cfo.index.intersection(revenue.index)
eligible = common[(cfo.loc[common].gt(0).all(axis=1)) & (revenue.loc[common, 2025] < revenue.loc[common, 2024])]
current = df[(df['year'] == 2025) & df['ticker'].isin(eligible)].dropna(subset=['gross_margin_pct', 'net_margin_pct'])
result = (current['gross_margin_pct'] - current['net_margin_pct']).mean()"""),
    374: (["HPG", "HSG", "MSR", "NKG"], [2022, 2024], """base = df[df['year'] == 2022].dropna(subset=['inventory_days'])
eligible = base[base['inventory_days'] > base['inventory_days'].median()]['ticker']
margin = df[df['ticker'].isin(eligible)].pivot(index='ticker', columns='year', values='gross_margin_pct').dropna()
result = (margin[2024] - margin[2022]).mean()"""),
    377: (["ASM", "DBC", "MPC", "MSN", "OGC", "QNS"], [2023, 2024], """current = df[(df['year'] == 2024) & (df['revenue_growth_pct'] > 0)].dropna(subset=['gross_margin_change_pp', 'cfo_margin_pct'])
selected = current.loc[current['gross_margin_change_pp'].idxmin()]
result = selected['cfo_margin_pct']"""),
    384: (["HPG", "HSG", "NKG"], [2023, 2024], """inventory_share = df.pivot(index='ticker', columns='year', values='inventory_to_assets_pct').dropna()
selected_ticker = (inventory_share[2024] - inventory_share[2023]).idxmax()
result = df[(df['ticker'] == selected_ticker) & (df['year'] == 2024)]['gross_margin_pct'].iloc[0]"""),
    385: (["DBC", "MPC", "MSN", "OGC", "QNS"], [2023, 2024], """eligible = df.dropna(subset=['npat', 'cfo']).groupby('ticker').filter(lambda part: len(part) == 2 and (part['npat'] > 0).all() and (part['cfo'] > 0).all())
result = eligible[eligible['year'] == 2024]['revenue_growth_pct'].mean()"""),
    391: (["VNM", "MCH", "QNS", "OGC"], [2023, 2024], """current = df[(df['year'] == 2024) & (df['revenue_growth_pct'] > 0)].dropna(subset=['sga_intensity_pct', 'net_margin_pct'])
selected = current.loc[current['sga_intensity_pct'].idxmax()]
result = selected['net_margin_pct']"""),
    393: (["HPG", "HSG", "NKG"], [2023, 2024], """current = df[(df['year'] == 2024) & (df['revenue_growth_pct'] > 0)].dropna(subset=['revenue_growth_pct', 'gross_margin_change_pp'])
selected = current.loc[current['revenue_growth_pct'].idxmax()]
result = selected['gross_margin_change_pp']"""),
    401: (["DBC", "MPC", "MSN", "OGC", "QNS"], [2023, 2024], """persistent = df.dropna(subset=['npat', 'cfo_to_npat']).groupby('ticker').filter(lambda part: len(part) == 2 and (part['npat'] > 0).all() and (part['cfo_to_npat'] > 0.5).all())
current = persistent[persistent['year'] == 2024].dropna(subset=['revenue_growth_pct', 'gross_margin_pct'])
selected = current.loc[current['revenue_growth_pct'].idxmax()]
result = selected['gross_margin_pct']"""),
    403: (["DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"], [2024], """filtered = df[df['current_ratio'] > 1].dropna(subset=['quick_ratio', 'operating_cash_flow_ratio'])
selected = filtered.loc[filtered['quick_ratio'].idxmin()]
result = selected['operating_cash_flow_ratio']"""),
    408: (["VNM", "DBC", "BAF"], [2023, 2024], """filtered = df[(df['year'] == 2024) & (df['revenue_growth_pct'] > 5)].dropna(subset=['gross_margin_pct'])
result = filtered['gross_margin_pct'].mean()"""),
    409: (["VIC", "KBC", "NLG", "DXG", "DIG"], [2024], """filtered = df.dropna(subset=['inventory_to_assets_pct', 'net_margin_pct'])
threshold = filtered['inventory_to_assets_pct'].median()
result = filtered[filtered['inventory_to_assets_pct'] > threshold]['net_margin_pct'].mean()"""),
    421: (["VRE", "VIC", "KBC", "VPI", "HPX"], [2023, 2024], """net_margin = df.pivot(index='ticker', columns='year', values='net_margin_pct').dropna()
selected_ticker = (net_margin[2024] - net_margin[2023]).idxmin()
roa = df[df['ticker'] == selected_ticker].set_index('year')['roa_pct']
result = roa.loc[2024] - roa.loc[2023]"""),
    424: (["DCM", "GVR", "HT1"], [2024], """filtered = df.dropna(subset=['inventory_days', 'cogs'])
median_days = filtered['inventory_days'].median()
selected = filtered.loc[(filtered['inventory_days'] - median_days).idxmax()]
result = abs(selected['cogs']) * (selected['inventory_days'] - median_days) / 365 / 1e9"""),
    425: (["FPT"], [2021, 2022, 2023, 2024], """filtered = df.dropna(subset=['roe_pct', 'eps'])
selected = filtered.loc[filtered['roe_pct'].idxmax()]
result = selected['eps'] / 1.1 / 1000"""),
    426: (["FPT"], [2021, 2022, 2023, 2024], """filtered = df[(df['npat'] > 0)].dropna(subset=['cfo_to_npat', 'npat', 'cfo'])
selected = filtered.loc[filtered['cfo_to_npat'].idxmin()]
result = (selected['npat'] - selected['cfo']) / 1e12"""),
    394: (["HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"], [2024], """filtered = df[(df['year'] == 2024) & (df['npat'] > 0)].dropna(subset=['npat', 'cfo'])
selected = filtered.loc[filtered['npat'].idxmax()]
result = selected['cfo'] / 1e12"""),
    380: (["DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"], [2024], """top_five = df.dropna(subset=['revenue', 'quick_ratio', 'liabilities_to_equity']).nlargest(5, 'revenue')
result = ((top_five['quick_ratio'] > 1) & (top_five['liabilities_to_equity'] < 1.5)).sum()"""),
    399: (["ASM", "DBC", "MPC", "MSN", "OGC", "QNS", "VNM"], [2023, 2024], """sga = df.pivot(index='ticker', columns='year', values='sga_expense').dropna()
revenue = df.pivot(index='ticker', columns='year', values='revenue').dropna()
common = sga.index.intersection(revenue.index)
sga_growth = sga.loc[common, 2024] / sga.loc[common, 2023] - 1
revenue_growth = revenue.loc[common, 2024] / revenue.loc[common, 2023] - 1
result = (sga_growth > revenue_growth).sum()"""),
    402: (["HPX", "KBC", "NVL", "VIC", "VPI", "VRE"], [2023, 2024], """inventory_share = df.pivot(index='ticker', columns='year', values='inventory_to_assets_pct').dropna()
gross_margin = df.pivot(index='ticker', columns='year', values='gross_margin_pct').dropna()
common = inventory_share.index.intersection(gross_margin.index)
result = ((inventory_share.loc[common, 2024] > inventory_share.loc[common, 2023]) & (gross_margin.loc[common, 2024] < gross_margin.loc[common, 2023])).sum()"""),
    413: (["MSN", "VNM", "MCH", "MPC", "DBC", "ASM", "QNS", "OGC"], [2024], """top_five = df.dropna(subset=['revenue', 'quick_ratio', 'liabilities_to_equity']).nlargest(5, 'revenue')
result = ((top_five['quick_ratio'] > 1) & (top_five['liabilities_to_equity'] < 1)).sum()"""),
    373: (["HPG", "HSG", "MSR", "NKG"], [2022, 2024], """inventory_days = df.pivot(index='ticker', columns='year', values='inventory_days').dropna()
eligible = inventory_days[inventory_days[2022] > inventory_days[2022].median()]
selected_ticker = (eligible[2024] - eligible[2022]).idxmin()
result = df[(df['ticker'] == selected_ticker) & (df['year'] == 2024)]['gross_margin_pct'].iloc[0]"""),
    389: (["NVL", "VIC", "VPI", "SCR", "KBC", "HPX", "VRE"], [2024], """median_debt_assets = df['liabilities_to_assets_pct'].median()
filtered = df[df['liabilities_to_assets_pct'] > median_debt_assets].dropna(subset=['operating_cash_flow_ratio', 'quick_ratio'])
selected = filtered.loc[filtered['operating_cash_flow_ratio'].idxmax()]
result = selected['quick_ratio']"""),
    390: (["HPG", "HSG", "NKG"], [2024], """filtered = df.dropna(subset=['quick_ratio', 'inventory'])
selected = filtered.loc[filtered['quick_ratio'].idxmin()]
result = selected['inventory'] / 1e12"""),
    397: (["DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"], [2024], """filtered = df[df['current_ratio'] > 1.5].dropna(subset=['quick_ratio', 'inventory'])
selected = filtered.loc[filtered['quick_ratio'].idxmin()]
result = selected['inventory'] / 1e12"""),
    # Question 412 sits inside the 2024 panel block and omits the year in its
    # wording.  Its peer questions and source bundle consistently use 2024.
    412: (["MSN", "OGC", "VNM"], [2024], """filtered = df[df['npat'] > 0].dropna(subset=['net_margin_pct', 'cfo_to_npat'])
selected = filtered.loc[filtered['net_margin_pct'].idxmax()]
result = selected['cfo_to_npat']"""),
    417: (["MSN", "DBC", "ASM", "MPC", "OGC"], [2024], """filtered = df.dropna(subset=['cfo_margin_pct', 'net_margin_pct', 'liabilities_to_equity']).copy()
filtered['cfo_minus_net_margin'] = filtered['cfo_margin_pct'] - filtered['net_margin_pct']
selected = filtered.loc[filtered['cfo_minus_net_margin'].idxmax()]
result = selected['liabilities_to_equity']"""),
    431: (["HPX", "KBC", "NVL", "VIC", "VPI", "VRE"], [2023, 2024], """current = df[df['year'] == 2024].dropna(subset=['gross_margin_change_pp', 'roe_pct'])
eligible_tickers = current[current['gross_margin_change_pp'] < -2]['ticker']
turnover = df[df['ticker'].isin(eligible_tickers)].pivot(index='ticker', columns='year', values='asset_turnover_avg').dropna()
selected_ticker = (turnover[2024] - turnover[2023]).idxmax()
result = current[current['ticker'] == selected_ticker]['roe_pct'].iloc[0]"""),
    376: (["HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"], [2024], """filtered = df[(df['year'] == 2024) & (df['current_ratio'] > 1.5)].dropna(subset=['quick_ratio', 'inventory_to_assets_pct'])
selected = filtered.loc[filtered['quick_ratio'].idxmin()]
result = selected['inventory_to_assets_pct']"""),
    383: (["MWG"], [2021, 2022, 2023, 2024], """filtered = df.sort_values('year').copy()
median_cfo_margin = filtered['cfo_margin_pct'].median()
result = ((filtered['gross_margin_pct'].diff() > 0) & (filtered['cfo_margin_pct'] > median_cfo_margin)).sum()"""),
    386: (["MSN"], [2020, 2021, 2022, 2023, 2024], """filtered = df[df['npat'] > 0].dropna(subset=['cfo_to_npat', 'quick_ratio'])
selected = filtered.loc[filtered['cfo_to_npat'].idxmin()]
result = selected['quick_ratio']"""),
    392: (["MCH", "QNS", "OGC"], [2024], """filtered = df.dropna(subset=['cfo_to_npat', 'quick_ratio'])
selected = filtered.loc[filtered['cfo_to_npat'].idxmax()]
result = selected['quick_ratio']"""),
    395: (["KBC"], [2022, 2023, 2024, 2025], """filtered = df.dropna(subset=['revenue_growth_pct', 'cfo_margin_pct'])
selected = filtered.loc[filtered['revenue_growth_pct'].idxmin()]
result = selected['cfo_margin_pct']"""),
    396: (["ASM", "DBC", "MSN", "OGC"], [2024], """filtered = df[(df['cfo'] > 0) & (df['npat'] > 0)].dropna(subset=['cfo_to_npat', 'quick_ratio'])
selected = filtered.loc[filtered['cfo_to_npat'].idxmax()]
result = selected['quick_ratio']"""),
    400: (["HPG"], [2020, 2021, 2022, 2023, 2024], """filtered = df[df['revenue_growth_pct'] > 0].dropna(subset=['revenue_growth_pct', 'cfo_margin_pct'])
selected = filtered.loc[filtered['revenue_growth_pct'].idxmax()]
result = selected['cfo_margin_pct']"""),
    406: (["DBC", "MSN", "OGC"], [2023, 2024], """current = df[(df['year'] == 2024) & (df['npat'] > 0) & (df['cfo_to_npat'] > 1)]
assets = df[df['ticker'].isin(current['ticker'])].pivot(index='ticker', columns='year', values='long_term_assets').dropna()
result = ((assets[2024] / assets[2023] - 1) * 100).mean()"""),
    414: (["HPG", "HSG", "NKG"], [2024], """filtered = df[(df['revenue_growth_pct'] > 3) & (df['operating_profit'] > 0) & (df['_prev_operating_profit'] > 0)].dropna(subset=['operating_leverage', 'operating_margin_pct'])
selected = filtered.loc[filtered['operating_leverage'].idxmax()]
result = selected['operating_margin_pct']"""),
    418: (["VIC", "NVL", "VRE", "KBC", "SCR", "VPI"], [2024], """filtered = df.dropna(subset=['sga_intensity_pct', 'roa_pct'])
high = filtered.loc[filtered['sga_intensity_pct'].idxmax()]
low = filtered.loc[filtered['sga_intensity_pct'].idxmin()]
result = high['roa_pct'] - low['roa_pct']"""),
    419: (["BSR", "PLX", "PVT"], [2024], """filtered = df[df['interest_coverage'] > 2].dropna(subset=['pbt', 'interest_expense', 'revenue'])
scenario_margin = (filtered['pbt'] - 0.2 * filtered['interest_expense']) / filtered['revenue'] * 100
result = scenario_margin.min()"""),
    422: (["DCM", "DPM", "GVR", "HPG", "HT1"], [2024], """median_growth = df['revenue_growth_pct'].median()
filtered = df[(df['revenue_growth_pct'] > median_growth)].dropna(subset=['npat', 'cfo', 'cfo_to_npat']).copy()
filtered['positive_gap'] = filtered['npat'] - filtered['cfo']
filtered = filtered[filtered['positive_gap'] > 0]
selected = filtered.loc[filtered['positive_gap'].idxmax()]
result = selected['cfo_to_npat']"""),
    423: (["GEX", "HBC", "PC1", "SAM", "VGC"], [2024], """median_quick = df['quick_ratio'].median()
filtered = df[df['quick_ratio'] < median_quick].dropna(subset=['pbt', 'interest_expense'])
scenario_coverage = (0.85 * (filtered['pbt'] + filtered['interest_expense'])) / filtered['interest_expense']
result = scenario_coverage.min()"""),
    429: (["ASM", "DBC", "MSN", "OGC", "VNM"], [2024], """median_turnover = df['fixed_assets_turnover'].median()
filtered = df[df['fixed_assets_turnover'] < median_turnover].dropna(subset=['fixed_assets_avg', 'fixed_assets_turnover'])
selected = filtered.loc[filtered['fixed_assets_avg'].idxmax()]
result = selected['fixed_assets_turnover']"""),
    430: (["DIG", "IJC", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"], [2023, 2024], """revenue = df.pivot(index='ticker', columns='year', values='revenue').dropna()
operating_margin = df.pivot(index='ticker', columns='year', values='operating_margin_pct').dropna()
sga_intensity = df.pivot(index='ticker', columns='year', values='sga_intensity_pct').dropna()
common = revenue.index.intersection(operating_margin.index).intersection(sga_intensity.index)
eligible = common[(revenue.loc[common, 2024] < revenue.loc[common, 2023]) & (operating_margin.loc[common, 2024] < operating_margin.loc[common, 2023])]
selected_ticker = (sga_intensity.loc[eligible, 2024] - sga_intensity.loc[eligible, 2023]).idxmax()
result = operating_margin.loc[selected_ticker, 2023] - operating_margin.loc[selected_ticker, 2024]"""),
    432: (["ASM", "DBC", "MML", "MPC", "MSN", "OGC", "QNS", "SAB", "VNM", "VSF"], [2023], """median_turnover = df['fixed_assets_turnover'].median()
filtered = df[df['fixed_assets_turnover'] < median_turnover].dropna(subset=['fixed_assets_avg', 'revenue'])
required_growth_pct = (median_turnover * filtered['fixed_assets_avg'] / filtered['revenue'] - 1) * 100
result = required_growth_pct.max()"""),
    433: (["CRE", "HPX", "KBC", "KHG", "NVL", "SNZ", "SSH", "VIC", "VPI", "VRE"], [2023], """median_debt_assets = df['liabilities_to_assets_pct'].median()
filtered = df[df['liabilities_to_assets_pct'] > median_debt_assets].dropna(subset=['total_assets', 'liabilities', 'short_term_receivables', 'long_term_receivables', 'inventory']).copy()
filtered['stressed_net_assets'] = filtered['total_assets'] - 0.3 * (filtered['short_term_receivables'] + filtered['long_term_receivables']) - 0.5 * filtered['inventory'] - filtered['liabilities']
negative = filtered[filtered['stressed_net_assets'] < 0]
result = negative['liabilities'].sum() / filtered['liabilities'].sum() * 100"""),
    434: (["DIG", "HPX", "SNZ", "SSH", "VRE"], [2023], """filtered = df[df['interest_coverage'] > 2].dropna(subset=['interest_coverage'])
selected = filtered.loc[filtered['interest_coverage'].idxmin()]
result = (selected['interest_coverage'] / 2 - 1) * 100"""),
    435: (["CRE", "DIG", "HPX", "KHG", "SNZ", "SSH", "VRE"], [2023], """filtered = df.dropna(subset=['pbt', 'interest_expense', 'gross_profit']).copy()
filtered['scenario_coverage'] = (filtered['pbt'] + filtered['interest_expense'] - 0.1 * filtered['gross_profit']) / filtered['interest_expense']
result = (filtered['scenario_coverage'] < 1.5).sum()"""),
    436: (["GEE", "GEX", "HHV", "SAM", "SJG", "VGC"], [2023], """filtered = df[df['operating_profit'] > 0].dropna(subset=['revenue', 'cogs', 'operating_profit']).copy()
filtered['required_price_increase_pct'] = 0.05 * filtered['cogs'] / (filtered['revenue'] - filtered['operating_profit']) * 100
result = filtered['required_price_increase_pct'].max()"""),
    437: (["VIC", "NVL", "VRE", "KBC", "SCR", "VPI", "HPX"], [2024], """median_debt_assets = df['liabilities_to_assets_pct'].median()
profitable = df[df['npat'] > 0]
low_debt_profitable = profitable[profitable['liabilities_to_assets_pct'] < median_debt_assets]
result = low_debt_profitable['npat'].sum() / profitable['npat'].sum() * 100"""),
    438: (["VIC", "NVL", "VRE", "KBC", "VPI", "HPX"], [2023, 2024], """eligible = df.dropna(subset=['npat', 'cfo_to_npat']).groupby('ticker').filter(lambda part: len(part) == 2 and (part['npat'] > 0).all() and (part['cfo_to_npat'] > 1).all())
result = eligible[eligible['year'] == 2024]['revenue_growth_pct'].mean()"""),
    439: (["HPG"], [2018, 2019, 2020, 2021, 2022, 2023, 2024], """median_gross_margin = df['gross_margin_pct'].median()
filtered = df[df['gross_margin_pct'] < median_gross_margin].dropna(subset=['cfo_margin_pct', 'roe_pct'])
selected = filtered.loc[filtered['cfo_margin_pct'].idxmax()]
result = selected['roe_pct']"""),
    440: (["DIG"], [2021, 2022, 2023, 2024], """filtered = df.dropna(subset=['liabilities_to_equity', 'interest_coverage'])
selected = filtered.loc[filtered['liabilities_to_equity'].idxmax()]
result = selected['interest_coverage']"""),
    441: (["HPG", "HSG", "MSR", "NKG"], [2024, 2025], """base = df[df['year'] == 2024]
eligible = base[base['liabilities_to_equity'] < base['liabilities_to_equity'].median()]['ticker']
candidates = df[(df['year'] == 2025) & df['ticker'].isin(eligible)].dropna(subset=['revenue_growth_pct', 'gross_margin_pct'])
selected = candidates.loc[candidates['revenue_growth_pct'].idxmax()]
result = selected['gross_margin_pct']"""),
    442: (["CEO", "DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"], [2024, 2025], """base = df[df['year'] == 2024]
eligible = base[base['liabilities_to_equity'] < base['liabilities_to_equity'].median()]['ticker']
candidates = df[(df['year'] == 2025) & df['ticker'].isin(eligible)].dropna(subset=['revenue_growth_pct', 'gross_margin_pct'])
selected = candidates.loc[candidates['revenue_growth_pct'].idxmax()]
result = selected['gross_margin_pct']"""),
    443: (["ASM", "DBC", "MCH", "MSN", "OGC", "VNM"], [2024, 2025], """base = df[df['year'] == 2024]
eligible = base[base['liabilities_to_equity'] < base['liabilities_to_equity'].median()]['ticker']
candidates = df[(df['year'] == 2025) & df['ticker'].isin(eligible)].dropna(subset=['revenue_growth_pct', 'gross_margin_pct'])
selected = candidates.loc[candidates['revenue_growth_pct'].idxmax()]
result = selected['gross_margin_pct']"""),
    444: (["ASM", "DBC", "MCH", "MPC", "MSN", "OGC", "QNS"], [2024], """filtered = df[df['sga_intensity_pct'] > df['sga_intensity_pct'].median()].dropna(subset=['operating_cash_flow_ratio', 'current_ratio'])
selected = filtered.loc[filtered['operating_cash_flow_ratio'].idxmin()]
result = selected['current_ratio']"""),
    445: (["DIG", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"], [2024], """filtered = df[df['sga_intensity_pct'] > df['sga_intensity_pct'].median()].dropna(subset=['operating_cash_flow_ratio', 'current_ratio'])
selected = filtered.loc[filtered['operating_cash_flow_ratio'].idxmin()]
result = selected['current_ratio']"""),
    446: (["DBC", "MCH", "MSN", "OGC", "QNS", "VNM"], [2024], """profitable = df[df['npat'] > 0]
median_de = profitable['liabilities_to_equity'].median()
result = profitable[profitable['liabilities_to_equity'] < median_de]['npat'].sum() / profitable['npat'].sum() * 100"""),
    447: (["ASM", "DBC", "MCH", "MSN", "OGC", "VNM"], [2024, 2025], """current = df[df['year'] == 2025].dropna(subset=['revenue_growth_pct', 'gross_margin_pct', 'interest_coverage'])
eligible = current[current['revenue_growth_pct'] > current['revenue_growth_pct'].median()]
selected = eligible.loc[eligible['gross_margin_pct'].idxmax()]
result = selected['interest_coverage']"""),
    448: (["BSR", "PLX", "PVT", "GAS"], [2024, 2025], """current = df[df['year'] == 2025].dropna(subset=['revenue_growth_pct', 'gross_margin_pct', 'interest_coverage'])
eligible = current[current['revenue_growth_pct'] > current['revenue_growth_pct'].median()]
selected = eligible.loc[eligible['gross_margin_pct'].idxmax()]
result = selected['interest_coverage']"""),
    449: (["MSN"], [2021, 2022, 2023, 2024, 2025], """filtered = df[df['cfo_margin_pct'] > df['cfo_margin_pct'].median()].dropna(subset=['revenue_growth_pct', 'roe_pct'])
selected = filtered.loc[filtered['revenue_growth_pct'].idxmax()]
result = selected['roe_pct']"""),
    450: (["HPG"], [2021, 2022, 2023, 2024], """median_accruals = df['operating_accruals_ratio_pct'].median()
filtered = df[df['operating_accruals_ratio_pct'] < median_accruals].dropna(subset=['revenue_growth_pct', 'gross_margin_pct'])
selected = filtered.loc[filtered['revenue_growth_pct'].idxmin()]
result = selected['gross_margin_pct']"""),
    511: (["DPM", "HT1", "HPG"], [2018], """filtered = df.dropna(subset=['eps', 'npat', 'equity'])
selected = filtered.loc[filtered['eps'].idxmax()]
result = selected['npat'] / selected['equity'] * 100"""),
}
REPHRASE_IDS = frozenset([*range(451, 495), 497, 504, 506, 536, 539, *range(540, 578)])


NUMBER_PARSER = """def _btc_number(x):
    s = str(x).strip().replace(' ', ' ')
    if ' ' in s:
        s = s.split()[0]
    if ')(' in s:
        s = s.split(')(')[0] + ')'
    negative = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace('%', '').replace('$', '')
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        tail = s.split(',')[-1]
        if len(tail) <= 2:
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    elif '.' in s:
        tail = s.split('.')[-1]
        if len(tail) == 3:
            s = s.replace('.', '')
    value = float(s)
    if negative:
        value = -abs(value)
    return value
"""


MINIMAL_NUMBER_PARSER = """def _btc_number(x, typed_factor=1):
    if not isinstance(x, str):
        return float(x) * float(typed_factor)
    s = str(x).strip().replace(' ', ' ')
    if s in ('', '-'):
        return 0.0
    if ' ' in s:
        s = s.split()[0]
    if ')(' in s:
        s = s.split(')(')[0] + ')'
    negative = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace('%', '').replace('$', '')
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        tail = s.split(',')[-1]
        if len(tail) <= 2:
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    elif '.' in s:
        tail = s.split('.')[-1]
        if len(tail) == 3:
            s = s.replace('.', '')
    value = float(s)
    if negative:
        value = -abs(value)
    return value
"""


PREVIOUS_METRICS = {
    "_prev_revenue": "kqkd:10",
    "_prev_gross_profit": "kqkd:20",
    "_prev_total_assets": "cdkt:270",
    "_prev_equity": "cdkt:400",
    "_prev_inventory": "cdkt:140",
    "_prev_tangible_fixed_assets": "cdkt:221",
    "_prev_operating_profit": "kqkd:30",
}


PANEL_DERIVATIONS = """
_numeric_columns = [c for c in df.columns if c not in ['ticker', 'year']]
for c in _numeric_columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
df['gross_margin_pct'] = df['gross_profit'] / df['revenue'] * 100
df['net_margin_pct'] = df['npat'] / df['revenue'] * 100
df['operating_margin_pct'] = df['operating_profit'] / df['revenue'] * 100
df['liabilities_to_equity'] = df['liabilities'] / df['equity']
df['debt_to_assets_pct'] = df['liabilities'] / df['total_assets'] * 100
df['liabilities_to_assets_pct'] = df['liabilities'] / df['total_assets'] * 100
df['current_ratio'] = df['current_assets'] / df['current_liabilities']
df['quick_ratio'] = (df['current_assets'] - df['inventory']) / df['current_liabilities']
df['interest_coverage'] = (df['pbt'] + df['interest_expense']) / df['interest_expense']
df['inventory_to_current_liabilities'] = df['inventory'] / df['current_liabilities']
df['operating_cash_flow_ratio'] = df['cfo'] / df['current_liabilities']
df['cfo_margin_pct'] = df['cfo'] / df['revenue'] * 100
df['inventory_to_assets_pct'] = df['inventory'] / df['total_assets'] * 100
df['sga_intensity_pct'] = (df['selling_expense'] + df['admin_expense']) / df['revenue'] * 100
df['long_term_assets_share_pct'] = df['long_term_assets'] / df['total_assets'] * 100
df['cfo_to_npat'] = df['cfo'] / df['npat']
df['operating_profit_to_pbt'] = df['operating_profit'] / df['pbt']
df['cfo_to_operating_profit'] = df['cfo'] / df['operating_profit']
df['revenue_growth_pct'] = (df['revenue'] / df['_prev_revenue'] - 1) * 100
df['gross_margin_change_pp'] = df['gross_margin_pct'] - (df['_prev_gross_profit'] / df['_prev_revenue'] * 100)
df['roa_pct'] = df['npat'] / ((df['_prev_total_assets'] + df['total_assets']) / 2) * 100
df['roe_pct'] = df['npat'] / ((df['_prev_equity'] + df['equity']) / 2) * 100
df['asset_turnover_avg'] = df['revenue'] / ((df['_prev_total_assets'] + df['total_assets']) / 2)
df['inventory_days'] = ((df['_prev_inventory'] + df['inventory']) / 2) / abs(df['cogs']) * 365
df['net_working_capital'] = df['current_assets'] - df['current_liabilities']
df['operating_accruals_ratio_pct'] = (df['npat'] - df['cfo']) / ((df['_prev_total_assets'] + df['total_assets']) / 2) * 100
df['sga_expense'] = df['selling_expense'] + df['admin_expense']
df['fixed_assets_avg'] = (df['_prev_tangible_fixed_assets'] + df['tangible_fixed_assets']) / 2
df['fixed_assets_turnover'] = df['revenue'] / df['fixed_assets_avg']
df['operating_leverage'] = ((df['operating_profit'] / df['_prev_operating_profit'] - 1) * 100) / df['revenue_growth_pct']
"""


NUMERIC_COERCION = """_numeric_columns = [c for c in df.columns if c not in ['ticker', 'year']]
for c in _numeric_columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
"""


def _df_column(node: ast.AST) -> str | None:
    """Return a literal ``df['column']`` reference from an AST node."""

    if not isinstance(node, ast.Subscript):
        return None
    if not isinstance(node.value, ast.Name) or node.value.id != "df":
        return None
    return node.slice.value if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str) else None


def _derivation_graph() -> tuple[dict[str, set[str]], dict[str, str], list[str]]:
    tree = ast.parse(PANEL_DERIVATIONS)
    dependencies: dict[str, set[str]] = {}
    statements: dict[str, str] = {}
    order: list[str] = []
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = _df_column(statement.targets[0])
        if target is None:
            continue
        referenced = {
            column
            for node in ast.walk(statement.value)
            if (column := _df_column(node)) is not None
        }
        dependencies[target] = referenced
        statements[target] = ast.unparse(statement)
        order.append(target)
    return dependencies, statements, order


DERIVATION_DEPENDENCIES, DERIVATION_STATEMENTS, DERIVATION_ORDER = _derivation_graph()
KNOWN_PANEL_COLUMNS = frozenset({
    "ticker", "year", *RAW_METRICS, *PREVIOUS_METRICS, *DERIVATION_DEPENDENCIES,
})


def required_panel_columns(
    analysis: str,
    extra_columns: tuple[str, ...] = (),
) -> tuple[list[str], list[str], list[str]]:
    """Find raw, previous-year and derived columns consumed by ``analysis``.

    Literal strings cover pandas APIs such as ``pivot(values=...)`` and
    ``dropna(subset=[...])``; attributes cover row-style access.  Derived
    metrics are recursively expanded to their original statement inputs.
    """

    tree = ast.parse(analysis)
    string_literals = {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    required = {value for value in string_literals if value in KNOWN_PANEL_COLUMNS}
    # pandas merge suffixes produce names such as revenue_growth_pct_2021.
    # Resolve the longest known prefix so the underlying metric is retained.
    for value in string_literals:
        matches = [column for column in KNOWN_PANEL_COLUMNS if value.startswith(f"{column}_")]
        if matches:
            required.add(max(matches, key=len))
    required.update(
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in KNOWN_PANEL_COLUMNS
    )
    required.update(column for column in extra_columns if column in KNOWN_PANEL_COLUMNS)
    pending = list(required)
    while pending:
        column = pending.pop()
        for dependency in DERIVATION_DEPENDENCIES.get(column, ()):
            if dependency not in required:
                required.add(dependency)
                pending.append(dependency)
    raw = [column for column in RAW_METRICS if column in required]
    previous = [column for column in PREVIOUS_METRICS if column in required]
    derived = [column for column in DERIVATION_ORDER if column in required]
    return raw, previous, derived


def minimal_derivations(columns: list[str]) -> str:
    lines = [NUMERIC_COERCION.rstrip()]
    lines.extend(DERIVATION_STATEMENTS[column] for column in columns)
    return "\n".join(lines) + "\n"


def typed_factor(raw: object) -> float:
    """Recover one Vietnamese thousands group after pandas dtype inference."""

    token = str(raw).strip().replace("\u00a0", " ")
    token = token.replace("(", "").replace(")", "").lstrip("+-")
    if "," in token or token.count(".") != 1:
        return 1.0
    head, tail = token.split(".", 1)
    return 1000.0 if head.isdigit() and tail.isdigit() and len(tail) == 3 else 1.0


def compute_audited_answer(tickers: list[str], years: list[int], analysis: str) -> float:
    frame = rephrases.panel(tickers, years)
    frame['_prev_operating_profit'] = [
        rephrases.ENGINE.value(str(row.ticker), int(row.year) - 1, 'operating_profit')
        for row in frame.itertuples()
    ]
    namespace = {"df": frame, "pd": pd}
    exec(analysis, {"pd": pd}, namespace)
    return round(float(namespace["result"]), 2)


def read_jsonl(path: Path) -> dict[int, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["id"])] = row
    return rows


def capture_context(qid: int) -> tuple[list[str], list[int]]:
    calls: list[tuple[list[str], list[int]]] = []
    original = rephrases.panel

    def tracking(tickers: list[str], years: list[int]) -> pd.DataFrame:
        calls.append((list(tickers), list(years)))
        return original(tickers, years)

    rephrases.panel = tracking
    try:
        try:
            rephrases.answer(qid)
        except Exception:
            # Context capture only needs the panel() calls.  Some historical
            # branches intentionally operate on sparse frames and can fail
            # later (for example idxmax on an all-NA metric) under newer
            # pandas versions; preserve the captured entity/year universe.
            if not calls:
                raise
    finally:
        rephrases.panel = original
    tickers = sorted({ticker for call, _ in calls for ticker in call})
    years = sorted({int(year) for _, call in calls for year in call})
    return tickers, years


class ReturnToResult(ast.NodeTransformer):
    def visit_Return(self, node: ast.Return) -> ast.AST:  # noqa: N802
        value = node.value
        if isinstance(value, ast.Tuple):
            value = value.elts[0]
        return ast.copy_location(ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=value), node)


def rephrase_analysis_code(qid: int) -> str:
    tree = ast.parse((ROOT / "scripts" / "solve_panel_rephrases.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "answer")
    branch = None
    for statement in function.body:
        if not isinstance(statement, ast.If):
            continue
        matches = bool(eval(compile(ast.Expression(statement.test), "<qid-test>", "eval"), {}, {"qid": qid}))
        if matches:
            branch = statement.body
            break
    if branch is None:
        raise KeyError(f"no deterministic branch for {qid}")
    module = ast.Module(body=[ast.Assign(targets=[ast.Name(id="qid", ctx=ast.Store())], value=ast.Constant(qid)), *branch], type_ignores=[])
    module = ReturnToResult().visit(module)
    ast.fix_missing_locations(module)
    code = ast.unparse(module)
    code = code.replace("tickers = sorted(ENGINE.cube.data)", "tickers = sorted(df['ticker'].unique())")
    helpers = """def panel(tickers, years):
    return df[df['ticker'].isin(tickers) & df['year'].isin(years)].copy()

def persistent(frame, column, predicate):
    valid = frame.dropna(subset=[column]).groupby('ticker').filter(lambda part: len(part) == frame['year'].nunique() and predicate(part[column]).all())
    return sorted(valid['ticker'].unique())
"""
    return helpers + "\n" + code


def source_expr(ticker: str, year: int, key: str) -> str:
    value = f"_source_value({ticker!r}, {year}, {key!r})"
    return f"abs({value})" if key in COST_KEYS else value


def make_panel_query(
    tickers: list[str],
    years: list[int],
    analysis: str,
    minimal_dependencies: bool = False,
    extra_columns: tuple[str, ...] = (),
) -> tuple[str, list[StatementCell]]:
    sources: list[StatementCell] = []
    row_codes = []
    if minimal_dependencies:
        raw_columns, previous_columns, derived_columns = required_panel_columns(analysis, extra_columns)
    else:
        raw_columns = list(RAW_METRICS)
        previous_columns = list(PREVIOUS_METRICS)
        derived_columns = list(DERIVATION_ORDER)
    for ticker in tickers:
        for year in years:
            values = [f"'ticker': {ticker!r}", f"'year': {year}"]
            for metric in raw_columns:
                key = RAW_METRICS[metric]
                cell = CUBE.cell(ticker, year, key, "consolidated")
                if cell is None:
                    values.append(f"{metric!r}: None")
                else:
                    sources.append(cell)
                    values.append(f"{metric!r}: {source_expr(ticker, year, key)}")
            for column in previous_columns:
                key = PREVIOUS_METRICS[column]
                cell = CUBE.cell(ticker, year - 1, key, "consolidated")
                if cell is None:
                    values.append(f"{column!r}: None")
                else:
                    sources.append(cell)
                    values.append(f"{column!r}: {source_expr(ticker, year - 1, key)}")
            row_codes.append("    {" + ", ".join(values) + "}")
    compact_loader = """
_src = list(dfs.values())[0].copy()
_src['year'] = pd.to_numeric(_src['year'], errors='coerce')
_src['scale'] = pd.to_numeric(_src['scale'], errors='coerce')
_src['value'] = _src['raw'].apply(_btc_number) * _src['scale']

def _source_value(ticker, year, metric_key):
    found = _src[(_src['ticker'] == ticker) & (_src['year'] == year) & (_src['metric_key'] == metric_key)]
    if found.empty:
        return None
    return float(found['value'].iloc[0])
"""
    if minimal_dependencies:
        compact_loader = """
_src = list(dfs.values())[0].copy()
_src['year'] = pd.to_numeric(_src['year'], errors='coerce')
_src['scale'] = pd.to_numeric(_src['scale'], errors='coerce')
_src['typed_factor'] = pd.to_numeric(_src['typed_factor'], errors='coerce').fillna(1)
_src['value'] = _src.apply(lambda row: _btc_number(row['raw'], row['typed_factor']) * row['scale'], axis=1)

def _source_value(ticker, year, metric_key):
    found = _src[(_src['ticker'] == ticker) & (_src['year'] == year) & (_src['metric_key'] == metric_key)]
    if found.empty:
        return None
    return float(found['value'].iloc[0])
"""
    parser = MINIMAL_NUMBER_PARSER if minimal_dependencies else NUMBER_PARSER
    derivations = minimal_derivations(derived_columns) if minimal_dependencies else PANEL_DERIVATIONS
    query = parser + compact_loader + "\n_rows = [\n" + ",\n".join(row_codes) + "\n]\ndf = pd.DataFrame(_rows)\n"
    query += derivations + "\n" + analysis.rstrip() + "\nresult = round(float(result), 2)"
    return query, sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "sub_compliant_standard")
    parser.add_argument("--out", type=Path, default=ROOT / "sub_compliant_all")
    parser.add_argument("--minimal-dependencies", action="store_true")
    args = parser.parse_args()

    v4 = read_jsonl(ROOT / "build" / "panel_answers_v4.jsonl")
    submission = json.loads((args.base / "submission.json").read_text(encoding="utf-8"))
    base_panel_audit_path = args.base / "panel_source_audit.json"
    base_panel_audit = {
        int(entry["id"]): entry
        for entry in (
            json.loads(base_panel_audit_path.read_text(encoding="utf-8"))
            if base_panel_audit_path.exists()
            else []
        )
    }
    args.out.mkdir(parents=True, exist_ok=True)
    data_dst = args.out / "data"
    if data_dst.exists():
        shutil.rmtree(data_dst)
    shutil.copytree(args.base / "data", data_dst)
    source_audit = args.base / "source_audit.json"
    if source_audit.exists():
        shutil.copy2(source_audit, args.out / source_audit.name)

    audit = []
    rewritten = HIGH_CONFIDENCE_IDS | REPHRASE_IDS | frozenset(AUDITED_PANEL_FIXES)
    for row in submission:
        qid = int(row["id"])
        if qid not in rewritten:
            continue
        if args.minimal_dependencies and qid in MINIMAL_DEPENDENCY_FALLBACK_IDS:
            fallback_audit = dict(base_panel_audit.get(qid, {"id": qid}))
            fallback_manifest = data_dst / f"q{qid}_source_cells.csv"
            if fallback_manifest.is_file():
                fallback_frame = pd.read_csv(
                    fallback_manifest,
                    encoding="utf-8-sig",
                    dtype=str,
                    keep_default_na=False,
                    index_col=None,
                )
                fallback_audit.update({
                    "source_cells": len(fallback_frame),
                    "source_tables": fallback_frame["source_table"].nunique(),
                    "evidence_files": 1,
                })
            fallback_audit.update({
                "dependency_mode": "base_fallback",
                "fallback_reason": "current statement cube cannot reproduce the stored fixed-asset series",
            })
            audit.append(fallback_audit)
            continue
        old_answer = row.get("answer")
        if qid in AUDITED_PANEL_FIXES:
            tickers, years, analysis = AUDITED_PANEL_FIXES[qid]
            if not args.minimal_dependencies:
                row["answer"] = compute_audited_answer(tickers, years, analysis)
        elif qid in HIGH_CONFIDENCE_IDS:
            generated = v4[qid]
            tickers = list(generated["tickers"])
            years = [int(year) for year in generated["years"]]
            analysis = generated["code"]
        else:
            tickers, years = capture_context(qid)
            analysis = rephrase_analysis_code(qid)
            if not args.minimal_dependencies:
                row["answer"] = round(float(rephrases.answer(qid)[0]), 2)
        semantic_columns = MINIMAL_SEMANTIC_COLUMNS.get(qid, ()) if args.minimal_dependencies else ()
        query, cells = make_panel_query(
            tickers,
            years,
            analysis,
            args.minimal_dependencies,
            semantic_columns,
        )
        unique_cells = []
        seen_cells = set()
        for cell in cells:
            identity = (cell.table_ref, cell.row_idx, cell.col_idx)
            if identity not in seen_cells:
                seen_cells.add(identity)
                unique_cells.append(cell)
        safe_name = f"q{qid}_source_cells.csv"
        destination = data_dst / safe_name
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            fieldnames = ["ticker", "year", "metric_key", "raw"]
            if args.minimal_dependencies:
                fieldnames.append("typed_factor")
            fieldnames.extend(("scale", "source_table", "source_csv", "row_idx", "col_idx"))
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for cell in unique_cells:
                payload = {
                    "ticker": cell.ticker, "year": cell.year, "metric_key": cell.metric_key,
                    "raw": cell.raw, "scale": cell.scale, "source_table": cell.table_ref,
                    "source_csv": Path(cell.csv_path).name, "row_idx": cell.row_idx, "col_idx": cell.col_idx,
                }
                if args.minimal_dependencies:
                    payload["typed_factor"] = typed_factor(cell.raw)
                writer.writerow(payload)
        evidence = [{"variable": "df1", "csv_path": f"data/{safe_name}"}]
        row["evidence"] = evidence
        row["pandas_query"] = query
        row["relevant_tables"] = list(dict.fromkeys(cell.table_ref for cell in unique_cells))
        row["relevant_docs"] = list(dict.fromkeys(cell.table_ref.split("|", 1)[0] for cell in unique_cells))
        audit_row = {"id": qid, "tickers": tickers, "years": years, "old_answer": old_answer, "answer": row.get("answer"), "source_tables": len({cell.table_ref for cell in unique_cells}), "source_cells": len(unique_cells), "evidence_files": 1}
        if args.minimal_dependencies:
            raw_columns, previous_columns, derived_columns = required_panel_columns(analysis, semantic_columns)
            audit_row.update({"dependency_mode": "minimal", "raw_columns": raw_columns, "previous_columns": previous_columns, "derived_columns": derived_columns})
        audit.append(audit_row)

    (args.out / "submission.json").write_text(json.dumps(submission, ensure_ascii=False), encoding="utf-8")
    (args.out / "panel_source_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.out), "rewritten": len(audit), "ids": [row["id"] for row in audit]}, indent=2))


if __name__ == "__main__":
    main()
