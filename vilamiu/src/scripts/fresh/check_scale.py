"""Check the unit scale across documents, which the identities cannot do.

Correcting an earlier claim: the accounting identities do NOT verify the unit. Every
term of `270 = 100 + 200` comes from the same table and therefore carries the same
scale, so the identity is scale-invariant. It verifies the code assignment and the
column choice, and nothing about the unit.

The unit needs two documents that declare theirs independently, and the corpus
supplies them. A balance sheet prints `Số cuối năm` and `Số đầu năm` side by side, so
the closing figure for year Y appears as `current` in the year-Y report and again as
`prior` in the year-(Y+1) report. Same company, same `Mã số`, two files, two separate
unit declarations.

Anchoring on a typed address rather than a row label is what makes this a clean test:
there is no question of having matched the wrong line, so a mismatch is a scale error
or an OCR error and nothing else. A ratio that is a power of ten separates the two.

This is the instrument that can judge a looser unit detector, so it is measured
before anything is loosened.

Usage:  python scripts/fresh/check_scale.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Codes worth comparing: the balance-sheet totals and their main components, which
# every report prints and which carry the largest figures.
CDKT_CODES = ("100", "110", "120", "130", "140", "150", "200", "270",
              "300", "310", "330", "400", "410", "440")
REL_TOL = 1e-4


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    merged: dict[tuple[str, str, str, str], dict[str, dict[str, float]]] = defaultdict(
        lambda: {"current": {}, "prior": {}})
    scales: dict[tuple[str, str, str, str], Counter[float]] = defaultdict(Counter)

    for line in (ROOT / "artifacts" / "fresh" / "statements.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        scope = ("separate" if "separate" in record["scope"]
                 else "consolidated" if "consolidated" in record["scope"]
                 else record["scope"])
        key = (record["ticker"], scope, record["kind"], record["year"])
        scales[key][record["scale"]] += 1
        for period in ("current", "prior"):
            for code, cell in record[period].items():
                # A statement split over pages arrives as several tables; the first
                # table to carry a code wins, matching the parser's own rule.
                merged[key][period].setdefault(code, cell[0])

    print(f"{len(merged)} (ma, pham vi, loai, nam)")
    mixed = sum(1 for counter in scales.values() if len(counter) > 1)
    print(f"  trong do khai NHIEU HON MOT he so don vi: {mixed}")

    counters: Counter[str] = Counter()
    powers: Counter[str] = Counter()
    examples: list[str] = []
    for (ticker, scope, kind, year), cells in sorted(merged.items()):
        if kind != "cdkt":
            continue
        try:
            later = merged.get((ticker, scope, kind, str(int(year) + 1)))
        except ValueError:
            continue
        if later is None:
            continue
        for code in CDKT_CODES:
            here = cells["current"].get(code)
            there = later["prior"].get(code)
            if here is None or there is None or not here or not there:
                continue
            counters["so sanh duoc"] += 1
            scale = max(abs(here), abs(there))
            if abs(here - there) <= REL_TOL * scale:
                counters["KHOP"] += 1
                continue
            counters["lech"] += 1
            ratio = max(abs(here), abs(there)) / min(abs(here), abs(there))
            power = round(math.log10(ratio))
            if abs(math.log10(ratio) - power) < 1e-6 and power != 0:
                powers[f"x10^{power}"] += 1
                if len(examples) < 8:
                    examples.append(
                        f"  {ticker} {scope[:4]} {year}->{int(year)+1} ma={code}  "
                        f"{here:,.0f}  vs  {there:,.0f}   (x10^{power})")
            else:
                powers["khong phai luy thua 10"] += 1

    have = counters["so sanh duoc"]
    print(f"\nso sanh current[nam Y] voi prior[nam Y+1] tren cung ma so:")
    print(f"  cap so sanh duoc : {have}")
    if have:
        print(f"  KHOP             : {counters['KHOP']} "
              f"({100 * counters['KHOP'] / have:.1f}%)")
        print(f"  lech             : {counters['lech']}")
    print("\nban chat cac ca lech:")
    for name, count in powers.most_common():
        print(f"  {name}: {count}")
    print("\nvi du lech dung luy thua 10 (loi he so don vi):")
    for line in examples:
        print(line)


if __name__ == "__main__":
    main()
