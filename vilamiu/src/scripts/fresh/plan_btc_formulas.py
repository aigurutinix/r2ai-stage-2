"""Lock BTC-registered formulas to Mã số cells, then emit a two-/three-cell plan.

Ban tổ chức locks roles first (`intermediate_formulas/registry.py`,
`panel/catalog.py` RATIOS), then compiles pandas. This planner does the inverse
on the public exam: detect the named formula, bind each role to a verified
address-book cell, refuse when any role is missing.

ROA / ROE use the official average denominator (đầu kỳ + cuối kỳ) / 2 — not ending
only. Quick ratio uses (TS ngắn hạn − HTK) / Nợ ngắn hạn.

Usage:
  python scripts/fresh/plan_btc_formulas.py --show 12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_identities import IDENTITIES, REL_TOL  # noqa: E402
from plan_answers import PARENT_RE, YEAR_RE  # noqa: E402
from refine_codes import RELIABLE_LEN  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

# formula_id -> (op, roles). Each role is (kind, code, period) with period in
# {"current", "prior"}. Expression is evaluated later in build_submission.
FORMULAS = {
    "roa": {
        "op": "roa_avg_pct",
        "patterns": (r"\bROA\b", r"tỷ suất sinh lời trên tổng tài sản",
                     r"sinh lời trên (?:tổng )?tài sản"),
        "roles": {
            "net_income": ("kqkd", "60", "current"),
            "assets_begin": ("cdkt", "270", "prior"),
            "assets_end": ("cdkt", "270", "current"),
        },
    },
    "roe": {
        "op": "roe_avg_pct",
        "patterns": (r"\bROE\b", r"tỷ suất sinh lời trên vốn chủ",
                     r"sinh lời trên vốn chủ"),
        "roles": {
            "net_income": ("kqkd", "60", "current"),
            "equity_begin": ("cdkt", "400", "prior"),
            "equity_end": ("cdkt", "400", "current"),
        },
    },
    "quick_ratio": {
        "op": "quick_ratio",
        "patterns": (r"thanh toán nhanh", r"quick ratio",
                     r"khả năng thanh toán nhanh"),
        "roles": {
            "current_assets": ("cdkt", "100", "current"),
            "inventory": ("cdkt", "140", "current"),
            "current_liabilities": ("cdkt", "310", "current"),
        },
    },
    "current_ratio": {
        "op": "current_ratio",
        "patterns": (r"thanh toán hiện hành", r"current ratio",
                     r"khả năng thanh toán hiện hành"),
        "roles": {
            "current_assets": ("cdkt", "100", "current"),
            "current_liabilities": ("cdkt", "310", "current"),
        },
    },
    "gross_margin": {
        "op": "pct",
        "patterns": (r"biên lợi nhuận gộp",),
        "roles": {
            "num": ("kqkd", "20", "current"),
            "den": ("kqkd", "10", "current"),
        },
    },
    "net_margin": {
        "op": "pct",
        "patterns": (r"biên lợi nhuận (?:ròng|thuần)", r"\bROS\b"),
        "roles": {
            "num": ("kqkd", "60", "current"),
            "den": ("kqkd", "10", "current"),
        },
    },
    "operating_margin": {
        "op": "pct",
        "patterns": (r"biên lợi nhuận hoạt động",),
        "roles": {
            "num": ("kqkd", "30", "current"),
            "den": ("kqkd", "10", "current"),
        },
    },
    "debt_to_assets": {
        "op": "pct",
        "patterns": (r"tỷ lệ nợ trên tổng tài sản", r"hệ số nợ trên tổng tài sản"),
        "roles": {
            "num": ("cdkt", "300", "current"),
            "den": ("cdkt", "270", "current"),
        },
    },
    "liabilities_to_equity": {
        "op": "ratio",
        "patterns": (r"nợ phải trả trên vốn chủ", r"hệ số nợ phải trả trên vốn chủ"),
        "roles": {
            "num": ("cdkt", "300", "current"),
            "den": ("cdkt", "400", "current"),
        },
    },
}

FILTER_CLAUSE_RE = re.compile(
    r"giai đoạn|trong các năm|năm mà|năm nào|cao nhất|thấp nhất|"
    r"lớn nhất|nhỏ nhất|trung vị|liền trước|liền sau", re.I)


def verified(record: dict) -> bool:
    for kind, target, plus, minus in IDENTITIES:
        if record["kind"] != kind:
            continue
        for period in ("current", "prior"):
            cells = record[period]
            if not all(c in cells for c in (target,) + plus + minus):
                continue
            expected = cells[target][0]
            total = sum(cells[c][0] for c in plus) - sum(cells[c][0] for c in minus)
            if abs(expected - total) <= REL_TOL * max(abs(expected), abs(total), 1.0):
                return True
    return False


def detect(text: str) -> str | None:
    if FILTER_CLAUSE_RE.search(text):
        return None
    for formula_id, spec in FORMULAS.items():
        for pattern in spec["patterns"]:
            if re.search(pattern, text, re.I):
                return formula_id
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--out", default="artifacts/fresh/btc_formula_plan.jsonl")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    records = [json.loads(line) for line in
               (ROOT / args.index).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    cells: dict[tuple, dict] = defaultdict(dict)
    for record in records:
        if record.get("bank"):
            continue
        if not verified(record):
            continue
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        want = RELIABLE_LEN[record["kind"]]
        for period in ("current", "prior"):
            key = (record["ticker"], record["year"], scope, record["kind"], period)
            for code, cell in record[period].items():
                if len(code) != want:
                    continue
                cells[key].setdefault(code, {
                    "value": cell[0], "label": cell[1], "row": cell[2],
                    "col": cell[3], "raw": cell[4], "csv": record["csv"],
                    "doc": record["doc"], "table_id": record["table_id"],
                    "table_ref": record["table_ref"], "scale": record["scale"],
                    "kind": record["kind"], "code": code, "period": period,
                    "ticker": record["ticker"], "year": record["year"],
                    "scope": scope,
                })

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    counters: Counter[str] = Counter()
    plan = []
    for question in questions:
        text = question["question"]
        formula_id = detect(text)
        if formula_id is None:
            counters["khong phai cong thuc BTC"] += 1
            continue
        tickers = resolver.resolve(text)
        years = YEAR_RE.findall(text)
        if len(tickers) != 1 or len(years) != 1:
            counters["khong khoa duoc 1 ma + 1 nam"] += 1
            continue
        ticker, year = next(iter(tickers)), years[0]
        scope = "separate" if PARENT_RE.search(text) else "consolidated"
        spec = FORMULAS[formula_id]
        bound = {}
        missing = False
        for role, (kind, code, period) in spec["roles"].items():
            cell = cells.get((ticker, year, scope, kind, period), {}).get(code)
            if cell is None:
                missing = True
                break
            bound[role] = cell
        if missing:
            counters["thieu role"] += 1
            continue
        entry = {
            "id": question["id"],
            "source": "btc_formula",
            "formula_id": formula_id,
            "op": spec["op"],
            "roles": bound,
        }
        plan.append(entry)
        counters[f"ok:{formula_id}"] += 1

    out = ROOT / args.out
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in plan) +
                   ("\n" if plan else ""), encoding="utf-8")
    print(f"{len(plan)} ke hoach")
    for name, count in counters.most_common():
        print(f"  {name}: {count}")
    for entry in plan[: args.show]:
        print(f"  id={entry['id']} {entry['formula_id']} {entry['op']}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
