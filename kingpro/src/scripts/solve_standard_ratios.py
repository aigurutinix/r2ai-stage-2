"""Solve high-confidence standard-statement questions in the medium block."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.financial.panel_metrics import PanelMetricEngine
from kingpro.financial.statement_cube import FinancialCube


CUBE = FinancialCube.read_jsonl(ROOT / "build/statement_cube.jsonl")
CON = PanelMetricEngine(CUBE, scope="consolidated")
SEP = PanelMetricEngine(CUBE, scope="separate")
QUESTIONS = {
    int(row["id"]): row["question"]
    for row in (
        json.loads(line)
        for line in (ROOT / "data/questions/questions.jsonl").read_text(encoding="utf-8").splitlines()
    )
}


def required(engine: PanelMetricEngine, ticker: str, year: int, metric: str) -> float:
    value = engine.value(ticker, year, metric)
    if value is None:
        raise ValueError(f"missing {engine.scope} {ticker} {year} {metric}")
    return value


def raw(engine: PanelMetricEngine, ticker: str, year: int, key: str) -> float:
    value = engine.raw(ticker, year, key)
    if value is None:
        raise ValueError(f"missing {engine.scope} {ticker} {year} {key}")
    return value


def solve(qid: int) -> tuple[float, str]:
    if qid == 655:
        return (required(CON, "VIC", 2021, "cash") / required(CON, "VIC", 2019, "cash") - 1) * 100, "VIC consolidated cash growth 2019-2021"
    if qid == 662:
        return required(CON, "CEO", 2016, "npat") / required(CON, "CEO", 2016, "total_assets") * 100, "CEO consolidated NPAT / ending assets"
    if qid == 666:
        return (required(SEP, "KBC", 2015, "finance_revenue") - required(SEP, "KBC", 2015, "finance_expense")) / 1e9, "KBC separate net finance result in VND bn"
    if qid == 673:
        return required(SEP, "SCR", 2025, "current_assets") / required(SEP, "SCR", 2025, "total_assets") * 100, "SCR separate current-assets share of total capital"
    if qid == 674:
        return required(CON, "HBC", 2016, "net_margin_pct"), "HBC consolidated net margin"
    if qid == 678:
        return required(SEP, "DNH", 2025, "current_liabilities") / required(SEP, "DNH", 2025, "equity"), "DNH separate current liabilities / equity"
    if qid == 679:
        return required(CON, "FOX", 2016, "gross_profit") / 1e9, "FOX consolidated gross profit in VND bn"
    if qid == 681:
        return required(SEP, "NLG", 2019, "gross_profit") / 1e9, "NLG separate gross profit in VND bn"
    if qid == 682:
        return raw(CON, "HSG", 2021, "kqkd:40") / 1e6, "HSG consolidated net other income in VND mn"
    if qid == 686:
        return required(CON, "VNM", 2023, "pbt") / required(CON, "VNM", 2023, "revenue") * 100, "VNM consolidated PBT margin"
    if qid == 694:
        return required(SEP, "SGB", 2015, "liabilities") / required(SEP, "SGB", 2015, "total_assets") * 100, "SGB separate liabilities / total assets"
    if qid == 695:
        equity_avg = (required(CON, "DCM", 2016, "equity") + required(CON, "DCM", 2017, "equity")) / 2
        return required(CON, "DCM", 2017, "npat") / equity_avg * 100, "DCM consolidated ROE using average equity"
    if qid == 696:
        equity_avg = (required(CON, "HSG", 2018, "equity") + required(CON, "HSG", 2019, "equity")) / 2
        return required(CON, "HSG", 2019, "revenue") / equity_avg, "HSG consolidated equity turnover using average equity"
    if qid == 698:
        return required(CON, "TTF", 2020, "admin_expense") / required(CON, "TTF", 2020, "revenue") * 100, "TTF consolidated admin expense / net revenue"
    if qid == 701:
        return required(SEP, "IJC", 2020, "selling_expense") / required(SEP, "IJC", 2020, "revenue") * 100, "IJC separate selling expense / net revenue"
    if qid == 702:
        return raw(SEP, "CEO", 2018, "kqkd:40") / 1e9, "CEO separate net other income in VND bn"
    if qid == 704:
        return (required(SEP, "NLG", 2015, "finance_revenue") - required(SEP, "NLG", 2015, "finance_expense")) / 1e9, "NLG separate net finance result in VND bn"
    if qid == 708:
        return required(CON, "MML", 2017, "net_margin_pct"), "MML consolidated net margin"
    if qid == 710:
        return required(SEP, "VNM", 2022, "current_liabilities") / required(SEP, "VNM", 2022, "equity") * 100, "VNM separate current liabilities / equity"
    if qid == 712:
        return required(CON, "GEG", 2025, "finance_expense") / required(CON, "GEG", 2025, "revenue") * 100, "GEG consolidated finance expense / net revenue"
    if qid == 713:
        return required(SEP, "HT1", 2020, "finance_revenue") / required(SEP, "HT1", 2020, "finance_expense") * 100, "HT1 separate finance revenue / finance expense"
    if qid == 714:
        return (required(CON, "HUT", 2024, "finance_revenue") - required(CON, "HUT", 2024, "finance_expense")) / 1e9, "HUT consolidated net finance result in VND bn"
    if qid == 718:
        return required(CON, "BMG", 2017, "cogs") / required(CON, "BMG", 2017, "revenue") * 100, "BMG consolidated COGS / net revenue"
    if qid == 719:
        return required(SEP, "NCB", 2016, "cfo") / required(SEP, "NCB", 2016, "pbt") * 100, "NCB separate CFO / PBT"
    if qid == 722:
        return raw(CON, "OGC", 2022, "kqkd:40") / 1e9, "OGC consolidated net other profit in VND bn"
    if qid == 725:
        return (required(SEP, "SAM", 2017, "finance_revenue") - required(SEP, "SAM", 2017, "finance_expense")) / 1e9, "SAM separate net finance result in VND bn"
    if qid == 726:
        return (required(CON, "GEX", 2024, "finance_revenue") - required(CON, "GEX", 2024, "finance_expense")) / 1e9, "GEX consolidated net finance result in VND bn"
    if qid == 728:
        sga = required(SEP, "DBC", 2018, "selling_expense") + required(SEP, "DBC", 2018, "admin_expense")
        return sga / required(SEP, "DBC", 2018, "revenue") * 100, "DBC separate SGA / net revenue"
    if qid == 730:
        return required(SEP, "DTK", 2017, "gross_profit") / 1e12, "DTK separate gross profit in trillion VND"
    raise KeyError(qid)


def main() -> None:
    rows = []
    failures = []
    for qid in (655, 662, 666, 673, 674, 678, 679, 681, 682, 686, 694, 695, 696, 698, 701, 702, 704, 708, 710, 712, 713, 714, 718, 719, 722, 725, 726, 728, 730):
        try:
            value, trace = solve(qid)
            rows.append({"id": qid, "ok": True, "question": QUESTIONS[qid], "answer": round(float(value), 2), "code": trace})
        except Exception as exc:
            failures.append({"id": qid, "error": str(exc)})
    out = ROOT / "build/standard_ratio_answers.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"answers": rows, "failures": failures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
