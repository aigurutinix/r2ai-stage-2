"""Do the 252 percent questions ship percents, or a mix of percents and fractions?

"Bao nhiêu phần trăm" wants 27.5, not 0.275. Both come out of the same pipeline —
a lookup that reads a cell already printed as a percent returns 27.5, while a
quotient computed from two đồng figures returns 0.275 unless something multiplies
by a hundred. If both conventions are shipping, one of them is wrong on every
question that uses it, and the split is visible without gold: a percent answer
below 1 in absolute value is a fraction that was never converted.

The same question applies to `lan` and `vong`, where the convention is the plain
multiple and no conversion applies — so an answer there below 0.01 is suspicious
for a different reason.

Usage:  PYTHONPATH=src python scripts/_probe_pct_convention.py
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402

SUB = ROOT / "submissions" / "screen_ratio_gated.zip"

BANDS = (
    ("zero", 0.0, 0.0),
    ("<0.01", 0.0, 0.01),
    ("0.01-1  (fraction?)", 0.01, 1.0),
    ("1-100   (percent)", 1.0, 100.0),
    ("100-1e3", 100.0, 1e3),
    ("1e3-1e6", 1e3, 1e6),
    (">1e6    (đồng)", 1e6, float("inf")),
)


def band(value: float) -> str:
    v = abs(value)
    for name, lo, hi in BANDS:
        if name == "zero":
            if v == 0.0:
                return name
            continue
        if lo < v <= hi or (lo == 0.0 and v <= hi):
            return name
    return ">1e6    (đồng)"


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

    def branch(i: int) -> str:
        c = (preds[i].get("pandas_query") or "").strip()
        if c in ("", "result = 0.0"):
            return "zero"
        if "def num(" in c or "find_row(" in c:
            return "llm"
        if c.count("df") >= 2:
            return "multi"
        if "iloc" in c:
            return "lookup"
        return "other"

    lines: list[str] = []
    p = lines.append

    for unit in ("phan_tram", "lan", "vong"):
        family = [q for q in parsed.values() if q.target_unit == unit]
        p(f"\n=== {unit}: {len(family)} questions ===")
        counts: Counter[str] = Counter()
        by_band: dict[str, list[int]] = {}
        for q in family:
            b = band(answer(q.id))
            counts[b] += 1
            by_band.setdefault(b, []).append(q.id)
        for name, *_ in BANDS:
            if counts[name]:
                p(f"  {name:22s} {counts[name]:4d}   "
                  f"branches={dict(Counter(branch(i) for i in by_band[name]))}")

        suspect = by_band.get("0.01-1  (fraction?)", [])
        p(f"  -- {len(suspect)} in the fraction band, first 20 --")
        for i in suspect[:20]:
            p(f"     id={i:4d} {branch(i):6s} ships {answer(i):.6g}")
            p(f"           {parsed[i].question[:135]}")

    out = ROOT / "artifacts" / "_probe_pct_convention.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
