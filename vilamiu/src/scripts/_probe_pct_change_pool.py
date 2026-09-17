"""Inside the 94 certainly-wrong ratio answers, how many are percent *change*?

`ratio.py` answers "X trên Y in one year". A different family is hiding in the
same pool: one metric compared across two years — "tăng bao nhiêu %", "tỷ lệ tăng
trưởng … từ cuối năm 2015 đến cuối năm 2021". `compose` already owns that shape
(`growth` -> `growth_pct`), so if these are shipping a đồng level instead of a
rate, something is declining them before the arithmetic runs.

This asks, for every one of the 94: which operation does `compose.classify` see,
does `compose.eligible` accept, and if not, which specific guard rejected it.

Usage:  PYTHONPATH=src python scripts/_probe_pct_change_pool.py
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


def why_declined(q) -> str:
    """The first guard in `compose.eligible` that rejects this question."""

    op = compose.classify(q.question)
    if op is None:
        return "no_op"
    if compose.screen_shape(q) is not None:
        return "screen_shape"
    if op != "growth" and q.unit_scale is None:
        return "no_unit_scale"
    if len(q.tickers) == 1 and len(q.years) >= 2:
        operands = len(q.years)
    elif len(q.tickers) >= 2 and q.years:
        if compose.SCREEN_RE.search(q.question):
            return "SCREEN_RE"
        operands = len(q.tickers)
    else:
        return f"axis_none(t={len(q.tickers)},y={len(q.years)})"
    if op in ("diff", "growth") and operands != 2:
        return f"{op}_needs_2_got_{operands}"
    return "ACCEPTED"


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(SUB) as z:
        preds = {p["id"]: p for p in json.loads(z.read("submission.json"))}

    bad = []
    for i, q in parsed.items():
        if q.target_unit not in ("phan_tram", "lan", "vong"):
            continue
        try:
            ans = float(preds[i].get("answer") or 0)
        except (TypeError, ValueError):
            continue
        if abs(ans) > 1000:
            bad.append((i, ans, q))

    lines: list[str] = []
    p = lines.append
    p(f"pool: {len(bad)} ratio-unit answers over 1000")

    ops = Counter(compose.classify(q.question) or "-" for _, _, q in bad)
    p(f"\ncompose.classify: {dict(ops)}")

    reasons: Counter[str] = Counter()
    by_reason: dict[str, list[int]] = {}
    for i, _, q in bad:
        r = why_declined(q)
        reasons[r] += 1
        by_reason.setdefault(r, []).append(i)
    p("\nwhy compose declines each of the 94:")
    for reason, count in reasons.most_common():
        p(f"  {reason:26s} {count:3d}  ids={by_reason[reason][:14]}")

    # The clean sub-family: one company, exactly two years, growth wording.
    clean = [
        (i, a, q) for i, a, q in bad
        if len(q.tickers) == 1 and len(q.years) == 2
        and compose.classify(q.question) in ("growth", "diff")
    ]
    p(f"\none company + exactly two years + growth/diff wording: {len(clean)}")
    for i, ans, q in clean:
        p(f"  id={i:4d} op={compose.classify(q.question):6s} why={why_declined(q):20s} "
          f"unit={q.target_unit:9s} ans={ans:.4g}")
        p(f"        {q.question[:150]}")

    # Same wording, but the year list is not exactly two — the `operands != 2`
    # guard. The question may still name a start and an end year explicitly.
    wide = [
        (i, a, q) for i, a, q in bad
        if len(q.tickers) == 1 and len(q.years) > 2
        and compose.classify(q.question) in ("growth", "diff")
    ]
    p(f"\none company + more than two years + growth/diff wording: {len(wide)}")
    for i, ans, q in wide[:20]:
        p(f"  id={i:4d} years={q.years} ans={ans:.4g}")
        p(f"        {q.question[:150]}")

    out = ROOT / "artifacts" / "_probe_pct_change_pool.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
