"""Deterministic ViFinQA plans for the advanced L5 block q0362-q0455.

This module deliberately keeps the numerical execution outside the LLM.  Each
question is compiled to the same audited pandas DSL used by ``l5_screening``;
the LLM is only used later to review ambiguous metric and selector choices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import l5_screening as engine
from .l5_screening import (
    Builder,
    L5Spec,
    accrual_ratio,
    admin_expense,
    cfo,
    cfo_current_liabilities,
    cogs,
    current_assets,
    current_liabilities,
    current_ratio,
    debt_equity,
    equity,
    filter_extreme,
    filter_mean,
    gross_margin,
    gross_margin_change,
    interest,
    interest_coverage,
    inventory,
    inventory_days,
    liabilities,
    net_margin,
    operating_profit,
    pat,
    pbt,
    ratio,
    revenue,
    revenue_growth,
    roa,
    select_answer,
    selling_expense,
    series,
    sga,
    total_assets,
)


ADVANCED_IDS = tuple(i for i in range(362, 456) if i not in {
    419, 423, 424, 425, 427, 428, 432, 433, 434, 435, 436,
})


def quick_ratio(b: Builder, ticker: str, year: int) -> str:
    return ratio(
        f"(({current_assets(b, ticker, year)}) - ({inventory(b, ticker, year)}))",
        current_liabilities(b, ticker, year),
    )


def cfo_margin(b: Builder, ticker: str, year: int) -> str:
    return ratio(cfo(b, ticker, year), revenue(b, ticker, year), 100.0)


def cfo_pat(b: Builder, ticker: str, year: int) -> str:
    return ratio(cfo(b, ticker, year), pat(b, ticker, year), absolute_b=False)


def debt_assets(b: Builder, ticker: str, year: int) -> str:
    return ratio(liabilities(b, ticker, year), total_assets(b, ticker, year))


def roe(b: Builder, ticker: str, year: int) -> str:
    start, end = equity(b, ticker, year - 1), equity(b, ticker, year)
    return f"(({pat(b, ticker, year)}) / ((abs({start}) + abs({end})) / 2.0) * 100.0)"


def asset_turnover(b: Builder, ticker: str, year: int) -> str:
    start, end = total_assets(b, ticker, year - 1), total_assets(b, ticker, year)
    return f"(({revenue(b, ticker, year)}) / ((abs({start}) + abs({end})) / 2.0))"


def operating_margin(b: Builder, ticker: str, year: int) -> str:
    return ratio(operating_profit(b, ticker, year), revenue(b, ticker, year), 100.0)


def sga_intensity(b: Builder, ticker: str, year: int) -> str:
    return ratio(sga(b, ticker, year), revenue(b, ticker, year), 100.0)


def noncurrent_assets(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "tai san dai han")


def fixed_assets(b: Builder, ticker: str, year: int) -> str:
    return b.fact(ticker, year, "tai san co dinh")


def fixed_asset_average(b: Builder, ticker: str, year: int) -> str:
    return f"((abs({fixed_assets(b, ticker, year - 1)}) + abs({fixed_assets(b, ticker, year)})) / 2.0)"


def fixed_asset_turnover(b: Builder, ticker: str, year: int) -> str:
    return f"(({revenue(b, ticker, year)}) / ({fixed_asset_average(b, ticker, year)}))"


def median_filter_extreme(values: list[str], selectors: list[str], answers: list[str],
                          side: str, mode: str) -> str:
    value_series = series(values)
    condition = f"({value_series} {side} {value_series}.median())"
    method = "idxmax" if mode == "max" else "idxmin"
    masked = f"{series(selectors)}.where({condition})"
    return f"float({series(answers)}.iloc[int({masked}.{method}())])"


def median_filter_mean(values: list[str], answers: list[str], side: str) -> str:
    value_series = series(values)
    return f"float({series(answers)}[{value_series} {side} {value_series}.median()].mean())"


def make_specs() -> dict[int, L5Spec]:
    specs: dict[int, L5Spec] = {}

    # q362-q366 have already-scored paraphrases in Batch A.
    specs[362] = engine.spec_inventory_median_debt_share(362)
    specs[363] = engine.spec_kbc_debt_interest(363)

    b = Builder(364)
    tickers = ("GVR", "DPM", "DCM", "PRT")
    cond = [f"((({cfo(b,t,2020)}) > 0.0) and (({cfo(b,t,2021)}) > 0.0))" for t in tickers]
    specs[364] = b.spec(filter_extreme(
        cond, [revenue_growth(b,t,2020,2021) for t in tickers],
        [accrual_ratio(b,t,2021) for t in tickers], "max"))
    specs[365] = engine.spec_first_negative_cfo_next_margin(365)
    specs[366] = engine.spec_current_assets_cfo_count(366)

    b = Builder(367); tickers = ("MSN", "MCH", "DBC", "ASM", "OGC")
    cond = [f"((({cfo(b,t,2024)}) > 0.0) and (({cfo(b,t,2025)}) > 0.0) and (({revenue_growth(b,t,2024,2025)}) < 0.0))" for t in tickers]
    specs[367] = b.spec(filter_mean(cond, [f"(({gross_margin(b,t,2025)}) - ({net_margin(b,t,2025)}))" for t in tickers]))

    b = Builder(368); tickers = ("HPG", "HSG", "MSR", "NKG")
    specs[368] = b.spec(median_filter_mean([quick_ratio(b,t,2022) for t in tickers], [net_margin(b,t,2022) for t in tickers], "<"))
    b = Builder(369); tickers = ("HPG", "HSG", "MSR", "NKG")
    specs[369] = b.spec(median_filter_extreme(
        [quick_ratio(b,t,2022) for t in tickers], [gross_margin_change(b,t,2022,2023) for t in tickers],
        [interest_coverage(b,t,2023) for t in tickers], "<", "max"))
    b = Builder(370); tickers = ("GEE", "GEX", "SAM")
    cond = ["(" + " and ".join(f"(({cfo(b,t,y)}) > 0.0)" for y in (2022,2023,2024)) + ")" for t in tickers]
    cagr = [f"(((abs({revenue(b,t,2024)}) / abs({revenue(b,t,2022)})) ** 0.5) - 1.0)" for t in tickers]
    specs[370] = b.spec(filter_extreme(cond, cagr, [net_margin(b,t,2024) for t in tickers], "max"))
    b = Builder(371); tickers = ("BSR", "PLX", "PVT")
    specs[371] = b.spec(filter_extreme([f"(({cfo(b,t,2024)}) > 0.0)" for t in tickers], [gross_margin(b,t,2024) for t in tickers], [interest_coverage(b,t,2024) for t in tickers], "max"))
    b = Builder(372); years = (2021,2022,2023,2024)
    specs[372] = b.spec(select_answer([quick_ratio(b,"VRE",y) for y in years], [cfo_current_liabilities(b,"VRE",y+1) for y in years], "min"))

    b = Builder(373); tickers = ("HPG", "HSG", "MSR", "NKG")
    specs[373] = b.spec(median_filter_extreme(
        [inventory_days(b,t,2022) for t in tickers], [f"(({inventory_days(b,t,2022)}) - ({inventory_days(b,t,2024)}))" for t in tickers],
        [gross_margin(b,t,2024) for t in tickers], ">", "max"))
    b = Builder(374); tickers = ("HPG", "HSG", "MSR", "NKG")
    specs[374] = b.spec(median_filter_mean([inventory_days(b,t,2022) for t in tickers], [gross_margin_change(b,t,2022,2024) for t in tickers], ">"))
    b = Builder(375); tickers = ("DCM", "DPM", "GVR", "PRT")
    values = series([debt_equity(b,t,2021) for t in tickers]); answers = series([interest_coverage(b,t,2021) for t in tickers])
    specs[375] = b.spec(f"float({answers}[{values} > {values}.median()].mean() - {answers}[{values} <= {values}.median()].mean())")
    b = Builder(376); tickers = ("HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE")
    specs[376] = b.spec(filter_extreme([f"(({current_ratio(b,t,2024)}) > 1.5)" for t in tickers], [quick_ratio(b,t,2024) for t in tickers], [ratio(inventory(b,t,2024),total_assets(b,t,2024),100.0) for t in tickers], "min"))
    b = Builder(377); tickers = ("ASM", "DBC", "MPC", "MSN", "OGC", "QNS")
    specs[377] = b.spec(filter_extreme([f"(({revenue_growth(b,t,2023,2024)}) > 0.0)" for t in tickers], [gross_margin_change(b,t,2023,2024) for t in tickers], [cfo_margin(b,t,2024) for t in tickers], "min"))
    b = Builder(378); years = tuple(range(2018,2025)); margins = [gross_margin(b,"HPG",y) for y in years]; mv = series(margins)
    specs[378] = b.spec(f"float({series([roe(b,'HPG',y) for y in years])}.iloc[int({series([cfo_margin(b,'HPG',y) for y in years])}.where({mv} < {mv}.median()).idxmax())])")
    b = Builder(379); tickers = ("ASM", "DBC", "MCH", "MSN", "OGC", "VNM"); growth = [revenue_growth(b,t,2024,2025) for t in tickers]; gv = series(growth)
    specs[379] = b.spec(f"float({series([interest_coverage(b,t,2025) for t in tickers])}.iloc[int({series([gross_margin(b,t,2025) for t in tickers])}.where({gv} > {gv}.median()).idxmax())])")
    b = Builder(380); tickers = ("DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"); revs = series([revenue(b,t,2024) for t in tickers]); eligible = [f"((({quick_ratio(b,t,2024)}) > 1.0) and (({debt_equity(b,t,2024)}) < 1.5))" for t in tickers]
    specs[380] = b.spec(f"int(({series(eligible)}.iloc[{revs}.nlargest(5).index] > 0.0).sum())", "int")

    b = Builder(381); years = tuple(range(2017,2024))
    specs[381] = b.spec(select_answer([cfo_pat(b,"HPG",y) for y in years], [ratio(inventory(b,"HPG",y),total_assets(b,"HPG",y),100.0) for y in years], "min"))
    b = Builder(382); tickers = ("DPM", "DCM")
    specs[382] = b.spec(select_answer([f"(({gross_margin(b,t,2022)}) - ({gross_margin(b,t,2023)}))" for t in tickers], [cfo_current_liabilities(b,t,2023) for t in tickers], "max"))
    b = Builder(383); years = (2021,2022,2023,2024); cm = [cfo_margin(b,"MWG",y) for y in years]; cv = series(cm)
    conditions = [f"((({gross_margin_change(b,'MWG',y-1,y)}) > 0.0) and (({cfo_margin(b,'MWG',y)}) > {cv}.median()))" for y in years[1:]]
    specs[383] = b.spec(f"int(({series(conditions)} > 0.0).sum())", "int")
    b = Builder(384); tickers = ("HPG", "HSG", "NKG")
    inv_change = [f"((({inventory(b,t,2024)}) / abs({total_assets(b,t,2024)})) - (({inventory(b,t,2023)}) / abs({total_assets(b,t,2023)})))" for t in tickers]
    specs[384] = b.spec(select_answer(inv_change, [gross_margin(b,t,2024) for t in tickers], "max"))
    b = Builder(385); tickers = ("DBC", "MPC", "MSN", "OGC", "QNS")
    cond = ["(" + " and ".join(f"((({pat(b,t,y)}) > 0.0) and (({cfo(b,t,y)}) > 0.0))" for y in (2023,2024)) + ")" for t in tickers]
    specs[385] = b.spec(filter_mean(cond, [revenue_growth(b,t,2023,2024) for t in tickers]))
    b = Builder(386); years = tuple(range(2020,2025)); cond = [f"(({pat(b,'MSN',y)}) > 0.0)" for y in years]
    specs[386] = b.spec(filter_extreme(cond, [cfo_pat(b,"MSN",y) for y in years], [quick_ratio(b,"MSN",y) for y in years], "min"))
    b = Builder(387); years = tuple(range(2021,2025)); cond = [f"(({pat(b,'HPG',y)}) > 0.0)" for y in years]
    specs[387] = b.spec(filter_extreme(cond, [cfo_pat(b,"HPG",y) for y in years], [interest_coverage(b,"HPG",y) for y in years], "min"))
    b = Builder(388); tickers = ("NVL", "KBC", "DIG", "IJC", "CEO", "CRE")
    specs[388] = b.spec(filter_mean([f"(({cfo_margin(b,t,2024)}) < 0.0)" for t in tickers], [gross_margin(b,t,2024) for t in tickers]))
    b = Builder(389); tickers = ("NVL", "VIC", "VPI", "SCR", "KBC", "HPX", "VRE")
    specs[389] = b.spec(median_filter_extreme([debt_assets(b,t,2024) for t in tickers], [cfo_current_liabilities(b,t,2024) for t in tickers], [quick_ratio(b,t,2024) for t in tickers], ">", "max"))
    b = Builder(390); tickers = ("HPG", "HSG", "NKG")
    specs[390] = b.spec(select_answer([quick_ratio(b,t,2024) for t in tickers], [f"(({inventory(b,t,2024)}) / 1000000000000.0)" for t in tickers], "min"))

    b = Builder(391); tickers = ("VNM", "MCH", "QNS", "OGC")
    specs[391] = b.spec(filter_extreme([f"(({revenue_growth(b,t,2023,2024)}) > 0.0)" for t in tickers], [sga_intensity(b,t,2024) for t in tickers], [net_margin(b,t,2024) for t in tickers], "max"))
    b = Builder(392); tickers = ("MCH", "QNS", "OGC")
    specs[392] = b.spec(select_answer([cfo_pat(b,t,2024) for t in tickers], [quick_ratio(b,t,2024) for t in tickers], "max"))
    b = Builder(393); tickers = ("HPG", "HSG", "NKG"); growth = [revenue_growth(b,t,2023,2024) for t in tickers]
    specs[393] = b.spec(filter_extreme([f"(({x}) > 0.0)" for x in growth], growth, [gross_margin_change(b,t,2023,2024) for t in tickers], "max"))
    b = Builder(394); tickers = ("HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE")
    specs[394] = b.spec(filter_extreme([f"(({pat(b,t,2024)}) > 0.0)" for t in tickers], [pat(b,t,2024) for t in tickers], [f"(({cfo(b,t,2024)}) / 1000000000000.0)" for t in tickers], "max"))
    b = Builder(395); years = (2022,2023,2024,2025)
    specs[395] = b.spec(select_answer([revenue_growth(b,"KBC",y-1,y) for y in years], [cfo_margin(b,"KBC",y) for y in years], "min"))
    b = Builder(396); tickers = ("ASM", "DBC", "MSN", "OGC")
    cond = [f"((({cfo(b,t,2024)}) > 0.0) and (({pat(b,t,2024)}) > 0.0))" for t in tickers]
    specs[396] = b.spec(filter_extreme(cond, [cfo_pat(b,t,2024) for t in tickers], [quick_ratio(b,t,2024) for t in tickers], "max"))
    b = Builder(397); tickers = ("DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE")
    specs[397] = b.spec(filter_extreme([f"(({current_ratio(b,t,2024)}) > 1.5)" for t in tickers], [quick_ratio(b,t,2024) for t in tickers], [f"(({inventory(b,t,2024)}) / 1000000000000.0)" for t in tickers], "min"))
    b = Builder(398); tickers = ("ACV", "HHV", "VSC")
    specs[398] = b.spec(select_answer([ratio(noncurrent_assets(b,t,2024),total_assets(b,t,2024)) for t in tickers], [asset_turnover(b,t,2024) for t in tickers], "max"))
    b = Builder(399); tickers = ("ASM", "DBC", "MPC", "MSN", "OGC", "QNS", "VNM")
    cond = [f"((((({sga(b,t,2024)}) - ({sga(b,t,2023)})) / abs({sga(b,t,2023)})) > ((({revenue(b,t,2024)}) - ({revenue(b,t,2023)})) / abs({revenue(b,t,2023)}))))" for t in tickers]
    specs[399] = b.spec(f"int(({series(cond)} > 0.0).sum())", "int")
    b = Builder(400); years = tuple(range(2020,2025)); growth = [revenue_growth(b,"HPG",y-1,y) for y in years]
    specs[400] = b.spec(filter_extreme([f"(({x}) > 0.0)" for x in growth], growth, [cfo_margin(b,"HPG",y) for y in years], "max"))
    b = Builder(401); tickers = ("DBC", "MPC", "MSN", "OGC", "QNS")
    cond = ["(" + " and ".join(f"((({pat(b,t,y)}) > 0.0) and (({cfo_pat(b,t,y)}) > 0.5))" for y in (2023,2024)) + ")" for t in tickers]
    specs[401] = b.spec(filter_extreme(cond, [revenue_growth(b,t,2023,2024) for t in tickers], [gross_margin(b,t,2024) for t in tickers], "max"))
    b = Builder(402); tickers = ("HPX", "KBC", "NVL", "VIC", "VPI", "VRE")
    cond = [f"((((({inventory(b,t,2024)}) / abs({total_assets(b,t,2024)})) > (({inventory(b,t,2023)}) / abs({total_assets(b,t,2023)}))) and (({gross_margin_change(b,t,2023,2024)}) < 0.0)))" for t in tickers]
    specs[402] = b.spec(f"int(({series(cond)} > 0.0).sum())", "int")
    b = Builder(403); tickers = ("DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE")
    specs[403] = b.spec(filter_extreme([f"(({current_ratio(b,t,2024)}) > 1.0)" for t in tickers], [quick_ratio(b,t,2024) for t in tickers], [cfo_current_liabilities(b,t,2024) for t in tickers], "min"))
    b = Builder(404); tickers = ("DCM", "DPM", "GVR"); dv = series([debt_equity(b,t,2024) for t in tickers]); costs = series([f"abs({interest(b,t,2024)})" for t in tickers])
    specs[404] = b.spec(f"float({costs}[{dv} > {dv}.median()].sum() / {costs}.sum() * 100.0)")
    b = Builder(405); years = (2022,2023,2024)
    specs[405] = b.spec(filter_extreme([f"(({revenue_growth(b,'VIC',y-1,y)}) > 0.0)" for y in years], [asset_turnover(b,"VIC",y) for y in years], [roe(b,"VIC",y) for y in years], "max"))
    b = Builder(406); tickers = ("DBC", "MSN", "OGC")
    cond = [f"((({pat(b,t,2024)}) > 0.0) and (({cfo_pat(b,t,2024)}) > 1.0))" for t in tickers]
    growth = [f"((({noncurrent_assets(b,t,2024)}) - ({noncurrent_assets(b,t,2023)})) / abs({noncurrent_assets(b,t,2023)}) * 100.0)" for t in tickers]
    specs[406] = b.spec(filter_mean(cond, growth))
    b = Builder(407); years = (2021,2022,2023); cond = [f"(({pat(b,'MWG',y)}) > 0.0)" for y in years]
    specs[407] = b.spec(filter_extreme(cond, [cfo_pat(b,"MWG",y) for y in years], [current_ratio(b,"MWG",y+1) for y in years], "min"))
    b = Builder(408); tickers = ("VNM", "DBC", "BAF")
    specs[408] = b.spec(filter_mean([f"(({revenue_growth(b,t,2023,2024)}) > 5.0)" for t in tickers], [gross_margin(b,t,2024) for t in tickers]))
    b = Builder(409); tickers = ("VIC", "KBC", "NLG", "DXG", "DIG")
    specs[409] = b.spec(median_filter_mean([ratio(inventory(b,t,2024),total_assets(b,t,2024)) for t in tickers], [net_margin(b,t,2024) for t in tickers], ">"))
    b = Builder(410); tickers = ("HPG", "HSG", "NKG"); de = [debt_equity(b,t,2024) for t in tickers]; dv = series(de)
    specs[410] = b.spec(f"float({series([roe(b,t,2024) for t in tickers])}[{dv} < {dv}.median()].max())")

    b = Builder(411); tickers = ("HPX", "KBC", "NVL", "PDR", "SCR")
    cond = [f"((({revenue_growth(b,t,2024,2025)}) > 0.0) and (({cfo_margin(b,t,2025)}) < 0.0))" for t in tickers]
    specs[411] = b.spec(f"int(({series(cond)} > 0.0).sum())", "int")
    b = Builder(412); tickers = ("MSN", "OGC", "VNM")
    specs[412] = b.spec(filter_extreme([f"(({pat(b,t,2024)}) > 0.0)" for t in tickers], [net_margin(b,t,2024) for t in tickers], [cfo_pat(b,t,2024) for t in tickers], "max"))
    b = Builder(413); tickers = ("MSN", "VNM", "MCH", "MPC", "DBC", "ASM", "QNS", "OGC"); revs = series([revenue(b,t,2024) for t in tickers]); ok = series([f"((({quick_ratio(b,t,2024)}) > 1.0) and (({debt_equity(b,t,2024)}) < 1.0))" for t in tickers])
    specs[413] = b.spec(f"int(({ok}.iloc[{revs}.nlargest(5).index] > 0.0).sum())", "int")
    b = Builder(414); tickers = ("HPG", "HSG", "NKG")
    cond = [f"((({revenue_growth(b,t,2023,2024)}) > 3.0) and (({operating_profit(b,t,2023)}) > 0.0) and (({operating_profit(b,t,2024)}) > 0.0))" for t in tickers]
    leverage = [f"((((({operating_profit(b,t,2024)}) - ({operating_profit(b,t,2023)})) / abs({operating_profit(b,t,2023)})) / ((({revenue(b,t,2024)}) - ({revenue(b,t,2023)})) / abs({revenue(b,t,2023)}))))" for t in tickers]
    specs[414] = b.spec(filter_extreme(cond, leverage, [operating_margin(b,t,2024) for t in tickers], "max"))
    b = Builder(415); years = tuple(range(2020,2025))
    specs[415] = b.spec(select_answer([revenue(b,"HPG",y) for y in years], [current_ratio(b,"HPG",y) for y in years], "max"))
    b = Builder(416); tickers = ("BSR", "PLX", "PVT")
    specs[416] = b.spec(select_answer([cfo_current_liabilities(b,t,2024) for t in tickers], [quick_ratio(b,t,2024) for t in tickers], "min"))
    b = Builder(417); tickers = ("MSN", "DBC", "ASM", "MPC", "OGC")
    specs[417] = b.spec(select_answer([f"(({cfo_margin(b,t,2024)}) - ({net_margin(b,t,2024)}))" for t in tickers], [debt_equity(b,t,2024) for t in tickers], "max"))
    b = Builder(418); tickers = ("VIC", "NVL", "VRE", "KBC", "SCR", "VPI"); si = [sga_intensity(b,t,2024) for t in tickers]; rv = [roa(b,t,2024) for t in tickers]
    specs[418] = b.spec(f"float(abs({series(rv)}.iloc[int({series(si)}.idxmax())] - {series(rv)}.iloc[int({series(si)}.idxmin())]))")

    b = Builder(420); tickers = ("VNM", "MSN", "DBC", "ASM", "MPC", "OGC")
    specs[420] = b.spec(filter_extreme([f"(({current_assets(b,t,2024)}) < ({current_liabilities(b,t,2024)}))" for t in tickers], [debt_assets(b,t,2024) for t in tickers], [roa(b,t,2024) for t in tickers], "min"))
    b = Builder(421); tickers = ("VIC", "VRE", "KBC", "VPI", "HPX"); nm_change = [f"(({net_margin(b,t,2024)}) - ({net_margin(b,t,2023)}))" for t in tickers]
    specs[421] = b.spec(select_answer(nm_change, [f"(({roa(b,t,2024)}) - ({roa(b,t,2023)}))" for t in tickers], "min"))
    b = Builder(422); tickers = ("DCM", "DPM", "GVR", "HPG", "HT1"); growth = [revenue_growth(b,t,2023,2024) for t in tickers]; gv = series(growth); gap = [f"(({pat(b,t,2024)}) - ({cfo(b,t,2024)}))" for t in tickers]; gapv = series(gap)
    raw = f"{series([cfo_pat(b,t,2024) for t in tickers])}.iloc[int({gapv}.where(({gv} > {gv}.median()) & ({gapv} > 0.0)).idxmax())]"
    specs[422] = b.spec(f"float(pd.Series([{raw}]).round(2).iloc[0])")
    b = Builder(426); years = (2021,2022,2023,2024); cond = [f"(({pat(b,'FPT',y)}) > 0.0)" for y in years]
    specs[426] = b.spec(filter_extreme(cond, [cfo_pat(b,"FPT",y) for y in years], [f"((({pat(b,'FPT',y)}) - ({cfo(b,'FPT',y)})) / 1000000000000.0)" for y in years], "min"))

    b = Builder(429); tickers = ("ASM", "DBC", "MSN", "OGC", "VNM"); turns = [fixed_asset_turnover(b,t,2024) for t in tickers]; tv = series(turns); avgs = series([fixed_asset_average(b,t,2024) for t in tickers])
    specs[429] = b.spec(f"float({tv}.iloc[int({avgs}.where({tv} < {tv}.median()).idxmax())])")
    b = Builder(430); tickers = ("DIG", "IJC", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE")
    cond = [f"((({revenue_growth(b,t,2023,2024)}) < 0.0) and (({operating_margin(b,t,2024)}) < ({operating_margin(b,t,2023)})))" for t in tickers]
    selectors = [f"(({sga_intensity(b,t,2024)}) - ({sga_intensity(b,t,2023)}))" for t in tickers]
    answers = [f"(({operating_margin(b,t,2023)}) - ({operating_margin(b,t,2024)}))" for t in tickers]
    specs[430] = b.spec(filter_extreme(cond, selectors, answers, "max"))
    b = Builder(431); tickers = ("HPX", "KBC", "NVL", "VIC", "VPI", "VRE")
    cond = [f"((({gross_margin(b,t,2023)}) - ({gross_margin(b,t,2024)})) > 2.0)" for t in tickers]
    selectors = [f"(({asset_turnover(b,t,2024)}) - ({asset_turnover(b,t,2023)}))" for t in tickers]
    specs[431] = b.spec(filter_extreme(cond, selectors, [roe(b,t,2024) for t in tickers], "max"))

    b = Builder(437); tickers = ("VIC", "NVL", "VRE", "KBC", "SCR", "VPI", "HPX"); da = [debt_assets(b,t,2024) for t in tickers]; dv = series(da); profits = series([pat(b,t,2024) for t in tickers])
    specs[437] = b.spec(f"float({profits}[({dv} < {dv}.median()) & ({profits} > 0.0)].sum() / {profits}[{profits} > 0.0].sum() * 100.0)")
    b = Builder(438); tickers = ("VIC", "NVL", "VRE", "KBC", "VPI", "HPX")
    cond = ["(" + " and ".join(f"((({pat(b,t,y)}) > 0.0) and (({cfo_pat(b,t,y)}) > 1.0))" for y in (2023,2024)) + ")" for t in tickers]
    specs[438] = b.spec(filter_mean(cond, [revenue_growth(b,t,2023,2024) for t in tickers]))
    b = Builder(439); years = tuple(range(2018,2025)); gm = [gross_margin(b,"HPG",y) for y in years]; gv = series(gm)
    specs[439] = b.spec(f"float({series([roe(b,'HPG',y) for y in years])}.iloc[int({series([cfo_margin(b,'HPG',y) for y in years])}.where({gv} < {gv}.median()).idxmax())])")
    b = Builder(440); years = (2021,2022,2023,2024)
    specs[440] = b.spec(select_answer([debt_equity(b,"DIG",y) for y in years], [interest_coverage(b,"DIG",y) for y in years], "max"))

    def low_de_growth_margin(qid: int, tickers: tuple[str, ...]) -> L5Spec:
        b = Builder(qid); de = [debt_equity(b,t,2024) for t in tickers]; dv = series(de)
        return b.spec(f"float({series([gross_margin(b,t,2025) for t in tickers])}.iloc[int({series([revenue_growth(b,t,2024,2025) for t in tickers])}.where({dv} < {dv}.median()).idxmax())])")
    specs[441] = low_de_growth_margin(441, ("HPG", "HSG", "MSR", "NKG"))
    specs[442] = low_de_growth_margin(442, ("CEO", "DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"))
    specs[443] = low_de_growth_margin(443, ("ASM", "DBC", "MCH", "MSN", "OGC", "VNM"))

    def high_sga_low_cfo_current(qid: int, tickers: tuple[str, ...]) -> L5Spec:
        b = Builder(qid); values = [sga_intensity(b,t,2024) for t in tickers]; vv = series(values)
        return b.spec(f"float({series([current_ratio(b,t,2024) for t in tickers])}.iloc[int({series([cfo_current_liabilities(b,t,2024) for t in tickers])}.where({vv} > {vv}.median()).idxmin())])")
    specs[444] = high_sga_low_cfo_current(444, ("ASM", "DBC", "MCH", "MPC", "MSN", "OGC", "QNS"))
    specs[445] = high_sga_low_cfo_current(445, ("DIG", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"))
    b = Builder(446); tickers = ("DBC", "MCH", "MSN", "OGC", "QNS", "VNM"); de = [debt_equity(b,t,2024) for t in tickers]; dv = series(de); pv = series([pat(b,t,2024) for t in tickers])
    specs[446] = b.spec(f"float({pv}[{dv} < {dv}.median()].sum() / {pv}.sum() * 100.0)")

    def growth_median_gm_interest(qid: int, tickers: tuple[str, ...]) -> L5Spec:
        b = Builder(qid); growth = [revenue_growth(b,t,2024,2025) for t in tickers]; gv = series(growth)
        return b.spec(f"float({series([interest_coverage(b,t,2025) for t in tickers])}.iloc[int({series([gross_margin(b,t,2025) for t in tickers])}.where({gv} > {gv}.median()).idxmax())])")
    specs[447] = growth_median_gm_interest(447, ("ASM", "DBC", "MCH", "MSN", "OGC", "VNM"))
    specs[448] = growth_median_gm_interest(448, ("BSR", "PLX", "PVT", "GAS"))
    b = Builder(449); years = (2021,2022,2023,2024,2025); cm = [cfo_margin(b,"MSN",y) for y in years]; cv = series(cm)
    specs[449] = b.spec(f"float({series([roe(b,'MSN',y) for y in years])}.iloc[int({series([revenue_growth(b,'MSN',y-1,y) for y in years])}.where({cv} > {cv}.median()).idxmax())])")
    b = Builder(450); years = (2021,2022,2023,2024); accr = [accrual_ratio(b,"HPG",y) for y in years]; av = series(accr)
    specs[450] = b.spec(f"float({series([gross_margin(b,'HPG',y) for y in years])}.iloc[int({series([revenue_growth(b,'HPG',y-1,y) for y in years])}.where({av} < {av}.median()).idxmin())])")
    b = Builder(451); tickers = ("HPG", "HSG", "MSR", "NKG")
    specs[451] = b.spec(median_filter_mean([inventory_days(b,t,2022) for t in tickers], [gross_margin_change(b,t,2022,2024) for t in tickers], ">"))
    b = Builder(452); tickers = ("ACV", "DLG", "HHV", "VSC")
    specs[452] = b.spec(median_filter_mean([current_ratio(b,t,2024) for t in tickers], [gross_margin(b,t,2024) for t in tickers], "<"))
    b = Builder(453); tickers = ("HPG", "HSG", "MSR", "NKG")
    specs[453] = b.spec(median_filter_mean([quick_ratio(b,t,2022) for t in tickers], [net_margin(b,t,2022) for t in tickers], "<"))
    b = Builder(454); tickers = ("HPG", "HSG", "MSR", "NKG")
    specs[454] = b.spec(median_filter_extreme([quick_ratio(b,t,2022) for t in tickers], [gross_margin_change(b,t,2022,2023) for t in tickers], [interest_coverage(b,t,2023) for t in tickers], "<", "max"))
    b = Builder(455); tickers = ("HPG", "HSG", "MSR", "NKG")
    specs[455] = b.spec(median_filter_extreme([inventory_days(b,t,2022) for t in tickers], [f"(({inventory_days(b,t,2022)}) - ({inventory_days(b,t,2024)}))" for t in tickers], [gross_margin(b,t,2024) for t in tickers], ">", "max"))

    if tuple(sorted(specs)) != ADVANCED_IDS:
        missing = sorted(set(ADVANCED_IDS) - set(specs))
        extra = sorted(set(specs) - set(ADVANCED_IDS))
        raise ValueError(f"Incomplete advanced L5 registry: missing={missing}, extra={extra}")
    return specs


ADVANCED_SPECS = make_specs()


def activate() -> None:
    """Point the shared retrieval/build engine at this source-controlled registry."""
    engine.L5_SPECS = ADVANCED_SPECS


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    candidates = sub.add_parser("candidates")
    candidates.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidates.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidates.add_argument("--output-dir", type=Path, default=Path("outputs/l5-advanced-facts"))
    candidates.add_argument("--overrides", type=Path, default=Path("analysis/l5_advanced_manual_overrides.csv"))
    submission = sub.add_parser("submission")
    submission.add_argument("--plans", type=Path, default=Path("outputs/l5-advanced-facts/plans.jsonl"))
    submission.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-l3-l4-l5-batch-a-submission-final/submission.json"))
    submission.add_argument("--output-dir", type=Path, default=Path("outputs/l1-l2-l3-l4-l5-advanced-submission"))
    validate = sub.add_parser("validate")
    validate.add_argument("--submission", type=Path, required=True)
    validate.add_argument("--plans", type=Path, default=Path("outputs/l5-advanced-facts/plans.jsonl"))
    validate.add_argument("--base-submission", type=Path, default=Path("outputs/l1-l2-l3-l4-l5-batch-a-submission-final/submission.json"))
    validate.add_argument("--zip", dest="zip_path", type=Path)
    args = parser.parse_args(argv)
    activate()
    if args.command == "candidates":
        engine.run_candidates(args.questions, args.database, args.output_dir, args.overrides)
    elif args.command == "submission":
        engine.build_submission(args.plans, args.database, args.base_submission, args.output_dir)
    else:
        print(json.dumps(engine.validate_submission(
            args.submission, args.plans, args.base_submission, args.zip_path
        ), ensure_ascii=False))


if __name__ == "__main__":
    main()
