"""Curated direct lookups whose wording maps exactly to statement line codes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.financial.panel_metrics import PanelMetricEngine
from kingpro.financial.statement_cube import FinancialCube


ENGINE = PanelMetricEngine(FinancialCube.read_jsonl(ROOT / "build/statement_cube.jsonl"))
QUESTIONS = {
    int(row["id"]): row["question"]
    for row in (
        json.loads(line)
        for line in (ROOT / "data/questions/questions.jsonl").read_text(encoding="utf-8").splitlines()
    )
}

# id: (ticker, statement year, metric, divisor, audit note)
LOOKUPS = {
    18: ("FIT", 2015, "equity", 1e9, "ending equity, VND bn"),
    47: ("PLX", 2017, "selling_expense", 1e12, "selling expense, VND tn"),
    71: ("GEG", 2025, "revenue", 1e9, "net revenue, VND bn"),
    87: ("HPX", 2024, "cogs", 1e9, "total COGS, VND bn"),
    # Beginning balances are the previous report year's ending balances.
    124: ("DIG", 2023, "cash", 1e11, "beginning 2024 cash = ending 2023, VND 100bn"),
    126: ("SAB", 2022, "cash", 1e12, "beginning 2023 cash = ending 2022, VND tn"),
    171: ("NLG", 2021, "cfo", 1e11, "operating cash flow, VND 100bn"),
    172: ("FPT", 2022, "cash", 1e12, "ending cash and equivalents, VND tn"),
    177: ("FIT", 2018, "cash", 1e6, "ending cash and equivalents, VND mn"),
    237: ("GAS", 2016, "revenue", 1e11, "net revenue, VND 100bn"),
    256: ("MCH", 2019, "npat", 1e11, "consolidated NPAT, VND 100bn"),
    293: ("ACV", 2015, "cash", 1e11, "ending cash and equivalents, VND 100bn"),
    295: ("HSG", 2018, "cfo", 1e11, "operating cash flow, VND 100bn"),
    297: ("SJG", 2024, "total_assets", 1e11, "ending total assets, VND 100bn"),
}


def main() -> None:
    rows, failures = [], []
    for qid, (ticker, year, metric, divisor, note) in LOOKUPS.items():
        value = ENGINE.value(ticker, year, metric)
        if value is None:
            failures.append({"id": qid, "error": f"missing {ticker} {year} {metric}"})
            continue
        rows.append({
            "id": qid,
            "ok": True,
            "question": QUESTIONS[qid],
            "answer": round(float(value / divisor), 2),
            "code": f"{ticker} consolidated {year} {metric}: {note}",
        })
    out = ROOT / "build/standard_lookup_answers.jsonl"
    out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"answers": rows, "failures": failures}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
