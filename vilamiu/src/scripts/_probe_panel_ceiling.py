"""Ceiling for a pure-deterministic panel path on the exam.

No LLM. For each question that looks like a cohort screen / ratio / count /
superlative over Circular-200 metrics, ask: does `metrics.parquet` contain
every (ticker, year, scope) cell the question would need?

If yes, a program that only does arithmetic over the panel can answer it
exactly — the same guarantee BTC's hard generator uses. If no, the question
needs OCR-table reading (or a thicker panel).

Scopes tried in order: consolidated, then separate. A question that names
"công ty mẹ" prefers separate.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _probe_exam_classes import classify  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402

# Phrase → metric column in metrics.parquet. Longer phrases first.
ALIASES: list[tuple[str, str]] = [
    ("lưu chuyển tiền thuần từ hoạt động kinh doanh", "cfo"),
    ("lưu chuyển tiền thuần từ hoạt động đầu tư", "cfi"),
    ("lưu chuyển tiền thuần từ hoạt động tài chính", "cff"),
    ("lợi nhuận thuần từ hoạt động kinh doanh", "operating_profit"),
    ("chi phí quản lý doanh nghiệp", "admin_expense"),
    ("chi phí bán hàng", "selling_expense"),
    ("chi phí lãi vay", "interest_expense"),
    ("chi phí tài chính", "financial_expense"),
    ("doanh thu hoạt động tài chính", "financial_income"),
    ("lợi nhuận trước thuế", "profit_before_tax"),
    ("lợi nhuận sau thuế", "net_profit"),
    ("lợi nhuận gộp", "gross_profit"),
    ("doanh thu thuần", "net_revenue"),
    ("doanh thu bán hàng", "revenue_gross"),
    ("giá vốn hàng bán", "cogs"),
    ("tài sản ngắn hạn", "current_assets"),
    ("tài sản dài hạn", "long_assets"),
    ("tổng cộng tài sản", "total_assets"),
    ("tổng tài sản", "total_assets"),
    ("nợ ngắn hạn", "liabilities_short"),
    ("nợ dài hạn", "liabilities_long"),
    ("nợ phải trả", "liabilities"),
    ("vốn chủ sở hữu", "equity"),
    ("hàng tồn kho", "inventory"),
    ("tiền và các khoản tương đương tiền", "cash"),
    ("các khoản phải thu ngắn hạn", "receivables_short"),
    ("cho vay khách hàng", "loans_to_customers"),
    ("tiền gửi của khách hàng", "customer_deposits"),
    ("vốn lưu động ròng", "_nwc"),  # derived: current_assets - liabilities_short
    ("roa", "net_profit"),  # needs total_assets too; flagged separately
    ("roe", "net_profit"),
    ("cfo", "cfo"),
]

DERIVED = {
    "_nwc": ("current_assets", "liabilities_short"),
    "roa": ("net_profit", "total_assets"),
    "roe": ("net_profit", "equity"),
}


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold())


def metrics_in(text: str) -> list[str]:
    folded = fold(text)
    found: list[str] = []
    for phrase, name in ALIASES:
        if phrase in folded and name not in found:
            found.append(name)
    # Expand derived
    expanded: list[str] = []
    for name in found:
        if name in DERIVED:
            expanded.extend(DERIVED[name])
        elif name.startswith("_"):
            continue
        else:
            expanded.append(name)
    # de-dup preserve order
    out: list[str] = []
    for m in expanded:
        if m not in out:
            out.append(m)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    panel = pd.read_parquet(ROOT / "artifacts" / "metrics.parquet")
    # index: (ticker, year, scope) -> row
    by_key: dict[tuple[str, str, str], pd.Series] = {}
    for _, row in panel.iterrows():
        by_key[(str(row["ticker"]), str(row["year"]), str(row["scope"]))] = row

    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "questions" / "questions.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    prize = {"cohort screen / rank", "count over a group", "ratio / percentage",
             "superlative", "two-cell arithmetic", "two-company arithmetic"}

    stats: collections.Counter[str] = collections.Counter()
    by_class: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    complete_examples: list[str] = []
    missing_examples: list[str] = []

    for i, row in enumerate(rows):
        text = row["question"]
        parsed = parse_question(i, text, roster)
        cls = classify(text, len(parsed.tickers), len(parsed.years))
        metrics = metrics_in(text)
        tickers = list(parsed.tickers)
        years = [str(y) for y in parsed.years]

        if cls not in prize:
            stats["out_of_prize"] += 1
            continue
        if not metrics:
            stats["no_metric_phrase"] += 1
            by_class[cls]["no_metric"] += 1
            continue
        if not tickers or not years:
            stats["no_entity"] += 1
            by_class[cls]["no_entity"] += 1
            continue

        prefer_sep = "công ty mẹ" in fold(text) or "riêng" in fold(text)
        scopes = ("separate", "consolidated") if prefer_sep else ("consolidated", "separate")

        missing = 0
        for t in tickers:
            for y in years:
                row_hit = None
                for scope in scopes:
                    cand = by_key.get((t, y, scope))
                    if cand is not None:
                        row_hit = cand
                        break
                if row_hit is None:
                    missing += 1
                    continue
                for m in metrics:
                    if m not in row_hit.index or pd.isna(row_hit[m]):
                        missing += 1

        if missing == 0:
            stats["COMPLETE"] += 1
            by_class[cls]["complete"] += 1
            if len(complete_examples) < 5:
                complete_examples.append(
                    f"[{cls}] {len(tickers)}co {len(years)}y {metrics}: {text[:110]}")
        else:
            stats["incomplete"] += 1
            by_class[cls]["incomplete"] += 1
            if len(missing_examples) < 5:
                missing_examples.append(
                    f"[{cls}] miss={missing} {metrics}: {text[:100]}")

    n = len(rows)
    print(f"panel cells: {len(by_key)} (ticker,year,scope)\n")
    print("overall on prize classes:")
    for k, v in stats.most_common():
        print(f"  {k:<22}{v:>5}")

    print(f"\n{'class':<26}{'complete':>10}{'incomplete':>12}{'no_metric':>10}")
    for cls in sorted(by_class):
        c = by_class[cls]
        print(f"{cls:<26}{c['complete']:>10}{c['incomplete']:>12}{c['no_metric']:>10}")

    print(f"\nCOMPLETE = {stats['COMPLETE']}/{n} = {stats['COMPLETE']/n:.1%} of exam")
    print("These are the questions a deterministic panel program can answer "
          "without inventing a formula or reading OCR.")
    print("\ncomplete examples:")
    for e in complete_examples:
        print(f"  {e}")
    print("\nincomplete examples:")
    for e in missing_examples:
        print(f"  {e}")


if __name__ == "__main__":
    main()
