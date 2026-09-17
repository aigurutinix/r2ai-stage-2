"""Split the cross-year disagreements into the bug that caused each one.

The score is explained by a single parameter: the probability p of reading one cell
correctly. 506 one-cell questions, 150 ratio questions needing two reads, 288 cohort
questions needing eight to sixteen — at p = 0.56 that predicts 342 correct against
the 348 measured. Reaching EXEC 0.60 needs p ≈ 0.85, and nothing else moves the
number, because every other question type is p raised to a power.

Cross-year agreement is the only offline measure of p, and it also says where p
leaks: half the disagreements read the SAME row label in both reports. A same-label
disagreement cannot be a localisation error — the locator found the right line item
twice. It is the column or the unit. This separates those two, which need opposite
fixes:

  ratio is a power of ten            -> `column_scale` disagreed between reports
  headers name different periods     -> `pick_column` took the wrong column
  neither                            -> the two reports genuinely differ (restated
                                        figures, or one is a subtotal)

Usage:  PYTHONPATH=src python scripts/_diagnose_reads.py --file artifacts/p_baseline.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]", "", text)


def power_of_ten(a: float, b: float) -> int | None:
    if not a or not b:
        return None
    ratio = max(abs(a), abs(b)) / min(abs(a), abs(b))
    if ratio < 1.5:
        return None
    power = round(math.log10(ratio))
    return power if abs(math.log10(ratio) - power) < 1e-6 else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default="artifacts/p_baseline.jsonl")
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    rows = [json.loads(line) for line in
            (ROOT / args.file).read_text(encoding="utf-8").splitlines() if line.strip()]
    bad = [r for r in rows if not r["agree"]]
    print(f"{len(rows)} cau doc duoc ca hai, {len(bad)} lech "
          f"({100 * len(bad) / max(1, len(rows)):.0f}%)")

    kinds: Counter[str] = Counter()
    examples: dict[str, list] = {}
    for row in bad:
        left, right = row["current"], row["comparative"]
        same_label = fold(left["label"]) == fold(right["label"])
        if not same_label:
            kind = "sai DINH VI (khac nhan dong)"
        else:
            power = power_of_ten(left["value"], right["value"])
            scales_differ = left.get("scale") != right.get("scale")
            years_left = set(YEAR_RE.findall(left.get("header", "")))
            years_right = set(YEAR_RE.findall(right.get("header", "")))
            if power is not None and scales_differ:
                kind = f"sai DON VI (he so khac, x10^{power})"
            elif power is not None:
                kind = f"lech LUY THUA 10 nhung he so BANG NHAU (x10^{power})"
            elif years_left and years_right and years_left != years_right:
                kind = "sai COT (tieu de nam khac nhau)"
            else:
                kind = "cung nhan, khong phai don vi hay nam"
        kinds[kind] += 1
        examples.setdefault(kind, []).append(row)

    print()
    for kind, count in kinds.most_common():
        print(f"  {kind}: {count} ({100 * count / len(bad):.0f}%)")

    for kind in kinds:
        print(f"\n--- {kind}")
        for row in examples[kind][:args.show]:
            left, right = row["current"], row["comparative"]
            print(f"  id={row['id']}")
            print(f"    Y   {left['value']:>20,.0f}  he so {left.get('scale')!s:>12}  "
                  f"cot{left.get('col')} '{left.get('header')}'  o={left.get('cell')}")
            print(f"        nhan '{str(left['label'])[:52]}'")
            print(f"    Y+1 {right['value']:>20,.0f}  he so {right.get('scale')!s:>12}  "
                  f"cot{right.get('col')} '{right.get('header')}'  o={right.get('cell')}")
            print(f"        nhan '{str(right['label'])[:52]}'")


if __name__ == "__main__":
    main()
