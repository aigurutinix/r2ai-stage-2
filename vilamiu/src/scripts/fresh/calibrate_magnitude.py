"""Calibrate the magnitude the question's unit implies, instead of assuming a band.

A flat band of [0.05, 1e6] flagged 21.4% of the shipped answers as absurd, and the
breakdown showed the band was the thing at fault: `tỷ đồng` and `nghìn tỷ đồng` sat 99%
inside it while `triệu đồng` sat 48% inside, and the "absurd" triệu answers were
figures like 3,505,000 triệu đồng — 3.5 trillion, which is exactly what a large bank's
sector loan book is. Vietnamese reports are routinely denominated in triệu đồng, so a
seven-figure answer in that unit is ordinary.

So the question is empirical: given the unit a question names, how wide is the range of
plausible answers? If it is two or three orders of magnitude, the unit excludes most of
a statement's lines before any label is compared. If it is six, the unit says nothing.

Measured on the shipped submission. It is 39% correct, so the distribution is
contaminated — but a wrong answer usually comes from the same statement as the right
one, so the SPREAD is a fair estimate even where the centre is not.

Usage:  python scripts/fresh/calibrate_magnitude.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402


def spread(values: list[float]) -> str:
    values = sorted(values)
    if len(values) < 5:
        return f"n={len(values)} (qua it)"
    def at(fraction: float) -> float:
        return values[min(len(values) - 1, int(fraction * len(values)))]
    low, mid, high = at(0.05), at(0.5), at(0.95)
    orders = math.log10(high / low) if low > 0 else float("inf")
    return (f"n={len(values):4d}  p5={low:>12,.4g}  p50={mid:>12,.4g}  "
            f"p95={high:>12,.4g}  rong {orders:.1f} bac")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default="submissions/aimed.zip")
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / args.zip) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))

    # In the stated unit, and converted to đồng — the second is what a candidate cell
    # would be compared against.
    in_unit: dict[str, list[float]] = defaultdict(list)
    in_dong: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if not (row.get("evidence") or []):
            continue
        name, unit = unit_of(row["question"])
        if not unit:
            continue
        try:
            value = abs(float(row["answer"]))
        except (TypeError, ValueError):
            continue
        if value <= 0:
            continue
        in_unit[name].append(value)
        in_dong[name].append(value * unit)

    print(f"{args.zip}\n")
    print("theo don vi cau hoi neu — do lon DAP AN (trong don vi do):")
    for name in sorted(in_unit, key=lambda n: -len(in_unit[n])):
        print(f"  {name:16s} {spread(in_unit[name])}")
    print("\ncung the, nhung quy ve DONG — day la thu so voi o ung vien:")
    for name in sorted(in_dong, key=lambda n: -len(in_dong[n])):
        print(f"  {name:16s} {spread(in_dong[name])}")

    every = [v for values in in_dong.values() for v in values]
    print(f"\ngop tat ca (dong): {spread(every)}")
    print("\ndoc ket qua: neu dai cua mot don vi HEP hon dai gop lai thi don vi la")
    print("tin hieu loc duoc; neu rong bang nhau thi no khong noi gi.")


if __name__ == "__main__":
    main()
