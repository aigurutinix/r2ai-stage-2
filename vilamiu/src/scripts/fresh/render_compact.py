"""A compact block, so several companies fit in one prompt.

247 questions name more than one company and therefore get no prompt at all: a full
block is one company's statements at about 4,500 tokens, and four of those overflow
even a 16k window.

What a cohort question needs from each company is narrower than what a single-company
question needs. It asks for one comparable figure per company — revenue, profit, total
assets, a ratio's operands — and never for a note detail. So the compact rendering
keeps the lines a comparison is actually made on and drops the rest:

  every three-digit code of the balance sheet that is a SECTION total (ends in 0)
  every two-digit code of the income statement and the cash flow
  no note index at all

That is roughly 1,200 tokens per company, so four companies and the question fit
inside 16k with room to spare. A cohort question that needs a detail line is out of
reach either way, and pretending otherwise would only add wrong answers.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from refine_codes import RELIABLE_LEN  # noqa: E402
from render_block import KIND_NAMES, MAGNITUDE_CODES, Corpus  # noqa: E402


# Keeping every two-digit code left the cash flow at twenty-five lines and the block at
# 3,164 tokens, so four companies came to 12,657 — inside a 16k window but with nothing
# left for the question. A comparison is made on the headline lines, so those are named
# explicitly rather than inferred from the code's length.
KQKD_HEADLINE = frozenset({"01", "02", "10", "11", "20", "21", "22", "23", "24",
                           "25", "26", "30", "31", "32", "40", "50", "51", "60"})
LCTT_HEADLINE = frozenset({"20", "30", "40", "50", "60", "61", "70"})


def is_headline(kind: str, code: str) -> bool:
    """Lines a cross-company comparison is actually made on."""

    if kind == "cdkt":
        # Section totals: 100, 110, 120, …, 270, 300, 310, …, 440. The detail lines
        # under them (131, 132, 152) are never what a cohort question compares.
        return len(code) == 3 and code.endswith("0")
    if kind == "kqkd":
        return code in KQKD_HEADLINE
    return code in LCTT_HEADLINE


def compact(corpus: Corpus, ticker: str, year: str, scope: str) -> str:
    rows = corpus.rows.get((ticker, year, scope))
    if not rows:
        return ""
    lines = [f"--- {ticker} {year} "
             f"({'riêng' if scope == 'separate' else 'hợp nhất'}) — đơn vị: đồng"]
    for kind in ("cdkt", "kqkd", "lctt"):
        present = sorted((code for (k, code) in rows
                          if k == kind and is_headline(kind, code)),
                         key=lambda c: (len(c), c))
        if not present:
            continue
        lines.append(f"  [{KIND_NAMES[kind]}]")
        for code in present:
            slot = rows[(kind, code)]
            label = corpus.consensus.get(
                (kind, code),
                (slot.get("current") or slot.get("prior", {})).get("label", ""))
            label = " ".join(str(label).split())[:56]
            magnitude = code in MAGNITUDE_CODES.get(kind, set())

            def show(period: str) -> str:
                cell = slot.get(period)
                if not cell:
                    return "—"
                value = abs(cell["value"]) if magnitude else cell["value"]
                return f"{value:,.0f}".replace(",", ".")

            lines.append(f"    {code:>4s} {label:<56s} {show('current'):>20s} "
                         f"{show('prior'):>20s}")
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    corpus = Corpus(ROOT / "artifacts" / "fresh" / "statements2.jsonl",
                    ROOT / "artifacts" / "fresh" / "notes.jsonl")
    sample = ("HPG", "2023", "consolidated")
    full = corpus.block(*sample)
    small = compact(corpus, *sample)
    print(small)
    print(f"\n--- day du: {len(full)} ky tu (~{len(full) // 2.5:.0f} token)")
    print(f"--- gon   : {len(small)} ky tu (~{len(small) // 2.5:.0f} token)")
    print(f"--- 4 cong ty gon: ~{4 * len(small) // 2.5:.0f} token")


if __name__ == "__main__":
    main()
