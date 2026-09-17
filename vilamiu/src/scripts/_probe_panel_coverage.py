"""How much of the exam is answerable from Circular-200 / BTC panel vocabulary?

Top teams sit at ~0.67 EXEC. Our 0.39 is not a retrieval problem (TABLES_F2 is
already competitive). The gap is answering, and the organisers' own generation
code names the shapes: peer-group screens, ratios over Circular 200 codes,
superlatives, multi-period arithmetic.

This measures the prize for a *deterministic panel path* that does not need an
LLM to invent the formula — only to (optionally) parse the question into
(tickers, year, metric, operation).
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _probe_exam_classes import classify  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402

# Labels that appear in BTC panel catalog + our metrics.py. Matching is
# intentionally loose (substring on folded Vietnamese) so this is an upper bound
# on coverage, not a claim that every hit is currently answered.
PANEL_PHRASES = [
    "tài sản ngắn hạn", "tài sản dài hạn", "tổng tài sản", "tổng cộng tài sản",
    "nợ ngắn hạn", "nợ dài hạn", "nợ phải trả", "vốn chủ sở hữu",
    "hàng tồn kho", "tiền và các khoản tương đương tiền",
    "doanh thu thuần", "doanh thu bán hàng", "giá vốn",
    "lợi nhuận gộp", "lợi nhuận sau thuế", "lợi nhuận trước thuế",
    "lợi nhuận thuần từ hoạt động kinh doanh",
    "chi phí bán hàng", "chi phí quản lý", "chi phí lãi vay", "chi phí tài chính",
    "lưu chuyển tiền thuần từ hoạt động kinh doanh",
    "lưu chuyển tiền thuần từ hoạt động đầu tư",
    "lưu chuyển tiền thuần từ hoạt động tài chính",
    "cho vay khách hàng", "tiền gửi của khách hàng",
    "roa", "roe", "d/e", "cfo", "biên lợi nhuận",
    "vốn lưu động ròng", "khả năng thanh toán", "hệ số",
    "tỷ lệ", "tỷ trọng", "tỷ số",
]

# Operations that need more than reading one cell.
OPS = {
    "count": re.compile(r"bao nhiêu (?:doanh nghiệp|công ty|mã)", re.I),
    "median": re.compile(r"trung vị|bình quân|trung bình", re.I),
    "superlative": re.compile(r"cao nhất|thấp nhất|lớn nhất|nhỏ nhất", re.I),
    "growth": re.compile(r"tăng trưởng|tốc độ tăng|giảm bao nhiêu", re.I),
    "difference": re.compile(r"chênh lệch|trừ đi|so với", re.I),
    "share": re.compile(r"tỷ trọng|chiếm bao nhiêu", re.I),
    "ratio": re.compile(r"tỷ lệ|tỷ số|hệ số|biên |ROA|ROE|lần\b", re.I),
    "sum": re.compile(r"\btổng\b", re.I),
    "screen": re.compile(r"có .{0,40}(?:lớn hơn|nhỏ hơn|trên|dưới|dương|âm)", re.I),
}


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    rows = [
        json.loads(line)
        for line in (ROOT / "data" / "questions" / "questions.jsonl")
        .read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    class_counts: collections.Counter[str] = collections.Counter()
    panel_hit: collections.Counter[str] = collections.Counter()
    ops_hit: collections.Counter[str] = collections.Counter()
    multi_ticker_panel = 0
    examples: dict[str, list[str]] = collections.defaultdict(list)

    for i, row in enumerate(rows):
        text = row["question"]
        parsed = parse_question(i, text, roster)
        cls = classify(text, len(parsed.tickers), len(parsed.years))
        class_counts[cls] += 1

        folded = fold(text)
        hits = [p for p in PANEL_PHRASES if p in folded]
        if hits:
            panel_hit[cls] += 1
            if len(parsed.tickers) >= 2:
                multi_ticker_panel += 1
        for name, pattern in OPS.items():
            if pattern.search(text):
                ops_hit[name] += 1
                if len(examples[name]) < 2:
                    examples[name].append(text)

    n = len(rows)
    print(f"{n} exam questions\n")
    print(f"{'class':<26}{'n':>5}{'share':>8}{'panel vocab':>12}")
    for cls, count in class_counts.most_common():
        print(f"{cls:<26}{count:>5}{count/n:>8.1%}"
              f"{panel_hit[cls]/count if count else 0:>12.1%}")

    print(f"\npanel-vocab hits overall: {sum(panel_hit.values())}/{n} = "
          f"{sum(panel_hit.values())/n:.1%}")
    print(f"multi-ticker + panel vocab: {multi_ticker_panel}/{n} = "
          f"{multi_ticker_panel/n:.1%}")

    print("\noperation markers (overlapping):")
    for name, count in ops_hit.most_common():
        print(f"  {name:<14}{count:>5}{count/n:>8.1%}")

    # The cohort+ratio+count block is what BTC's hard/intermediate generators
    # produce and what a panel path is built for.
    prize = (class_counts["cohort screen / rank"]
             + class_counts["count over a group"]
             + class_counts["ratio / percentage"]
             + class_counts["superlative"]
             + class_counts["two-cell arithmetic"])
    print(f"\nprize classes (cohort+count+ratio+super+two-cell): "
          f"{prize}/{n} = {prize/n:.1%}")
    print(f"  at 0.67 target that is ~{0.67*n:.0f} correct;")
    print(f"  we have ~{0.3933*n:.0f}; gap ~{(0.67-0.3933)*n:.0f} questions.")
    print(f"  if we held single-cell at current rate (~0.55 of 370≈204) and "
          f"solved 70% of the {prize} prize class: "
          f"~{204 + 0.7*prize:.0f}/{n} = {(204 + 0.7*prize)/n:.1%}")

    print("\nexamples:")
    for name in ("screen", "count", "median", "superlative", "growth", "ratio"):
        for text in examples.get(name, []):
            print(f"  [{name}] {text[:130]}")


if __name__ == "__main__":
    main()
