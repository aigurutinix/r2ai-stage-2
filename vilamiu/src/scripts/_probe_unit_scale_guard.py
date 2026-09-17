"""How wide is the `unit_scale is None` guard in `compose.eligible`?

    if op != "growth" and question.unit_scale is None:
        return None

The comment says "a money answer needs a scale to convert into". True — but a
question answered in %, lần or vòng is not a money answer, so it never carries a
scale, and the guard rejects the whole family on a premise that does not apply to
it. The identical guard, copied into `resolve_screen_ratio`, was found and fixed
on 08/08; this one was never revisited.

Two numbers decide whether that is worth acting on:

  1. across all 1012 questions, how many are rejected by this guard *alone*;
  2. of those, how many are the shape where the operation genuinely yields a rate
     — a percent change of one metric between two years — versus shapes where the
     scale really is needed.

Also re-checks the "certainly wrong" premise of the 94-question pool. A growth
percentage may legitimately exceed 1000 (an eleven-fold rise over six years is
1000%), so the threshold that makes an answer *impossible* is higher than the one
that makes it suspicious.

Usage:  PYTHONPATH=src python scripts/_probe_unit_scale_guard.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import compose  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402

SUB = ROOT / "submissions" / "screen_ratio_gated.zip"
RATE_UNITS = ("phan_tram", "lan", "vong")


def axis_of(q) -> tuple[str, int] | None:
    if len(q.tickers) == 1 and len(q.years) >= 2:
        return "year", len(q.years)
    if len(q.tickers) >= 2 and q.years:
        return "ticker", len(q.tickers)
    return None


def blocked_only_by_scale(q) -> bool:
    """Would `compose.eligible` accept this question if the guard were lifted?"""

    op = compose.classify(q.question)
    if op is None or op == "growth":
        return False
    if q.unit_scale is not None:
        return False
    if compose.screen_shape(q) is not None:
        return False
    got = axis_of(q)
    if got is None:
        return False
    axis, operands = got
    if axis == "ticker" and compose.SCREEN_RE.search(q.question):
        return False
    if op in ("diff", "growth") and operands != 2:
        return False
    return True


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(SUB) as z:
        preds = {p["id"]: p for p in json.loads(z.read("submission.json"))}

    def answer(i: int) -> float:
        try:
            return float(preds[i].get("answer") or 0)
        except (TypeError, ValueError):
            return 0.0

    lines: list[str] = []
    p = lines.append

    rate = [q for q in parsed.values() if q.target_unit in RATE_UNITS]
    p(f"questions answered in %/lần/vòng: {len(rate)} of {len(parsed)}")
    for cut in (1e3, 1e4, 1e6, 1e9):
        n = sum(1 for q in rate if abs(answer(q.id)) > cut)
        p(f"  shipping |answer| > {cut:.0e}: {n}")
    p("  (a growth rate can exceed 1000%; above 1e6 nothing is a rate)")

    hit = [q for q in parsed.values() if blocked_only_by_scale(q)]
    p(f"\nrejected by the unit_scale guard ALONE: {len(hit)}")
    p(f"  of which answered in a rate unit: "
      f"{sum(1 for q in hit if q.target_unit in RATE_UNITS)}")
    p(f"  ops: {dict(Counter(compose.classify(q.question) for q in hit))}")
    p(f"  target units: {dict(Counter(q.target_unit for q in hit))}")
    p(f"  axis: {dict(Counter(axis_of(q)[0] for q in hit))}")

    # The sub-family where the operation itself yields a rate, so no scale can
    # ever be needed: percent change of one metric between exactly two years.
    pct_change = [
        q for q in hit
        if q.target_unit == "phan_tram" and axis_of(q) == ("year", 2)
        and compose.classify(q.question) == "diff"
    ]
    p(f"\npercent change, one company, exactly two years: {len(pct_change)}")
    for q in pct_change:
        p(f"  id={q.id:4d} ships {answer(q.id):.4g}")
        p(f"        {q.question[:150]}")

    rest = [q for q in hit if q not in pct_change]
    p(f"\nthe other {len(rest)} — scale may genuinely be needed, sample:")
    for q in rest[:25]:
        p(f"  id={q.id:4d} op={compose.classify(q.question):6s} unit={q.target_unit:9s} "
          f"axis={axis_of(q)[0]:6s} ships {answer(q.id):.4g}")
        p(f"        {q.question[:140]}")

    out = ROOT / "artifacts" / "_probe_unit_scale_guard.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
