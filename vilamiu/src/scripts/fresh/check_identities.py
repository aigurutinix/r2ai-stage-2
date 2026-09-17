"""Verify the parse against the accounting identities the statements print themselves.

The row labels carry the statutory arithmetic: "TỔNG TÀI SẢN (270 = 100 + 200)",
"Lợi nhuận gộp (20 = 10 - 11)", "Lưu chuyển tiền thuần trong năm (50=20+30+40)",
"TỔNG CỘNG NGUỒN VỐN (440 = 300 + 400)". Thông tư 200 fixes them, so they hold for
every company in every year.

That makes the corpus self-checking, and it checks exactly the three things a read
can get wrong at once:

  * the code assignment — a wrong `Mã số` breaks the sum
  * the column — mixing this year's 270 with last year's 100 breaks the sum
  * the unit scale — a table read at 1e6 against one read at 1 breaks the sum

No gold, no model, no leaderboard. And unlike agreement between two guesses, an
identity cannot be satisfied by two errors that happen to coincide: the residual is
a signed number over five or six independent cells.

Reports, per statement kind, how often each identity holds — which is a direct
measure of parse accuracy on the real corpus.

Usage:  python scripts/fresh/check_identities.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# (kind, target, [addends], [subtrahends]) from Thông tư 200. Only identities whose
# every term is a code the parser reliably sees: three digits in `cdkt`, two in the
# other two.
IDENTITIES: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("cdkt", "270", ("100", "200"), ()),
    ("cdkt", "440", ("300", "400"), ()),
    ("cdkt", "300", ("310", "330"), ()),
    ("cdkt", "100", ("110", "120", "130", "140", "150"), ()),
    ("kqkd", "20", ("10",), ("11",)),
    ("kqkd", "10", ("01",), ("02",)),
    ("kqkd", "30", ("20", "21"), ("22", "24", "25", "26")),
    ("kqkd", "50", ("30", "40"), ()),
    ("kqkd", "60", ("50",), ("51", "52")),
    ("lctt", "50", ("20", "30", "40"), ()),
    ("lctt", "70", ("50", "60", "61"), ()),
)

# The corpus is OCR, so an exact match is the wrong test: one mis-scanned digit in a
# nine-digit figure moves the residual without meaning the parse is wrong. A tenth
# of a percent of the target is tight enough that a wrong code or column fails, and
# loose enough to survive a rounding difference in the printed statement.
REL_TOL = 1e-3


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = [json.loads(line) for line in
               (ROOT / "artifacts" / "fresh" / "statements.jsonl").read_text(
                   encoding="utf-8").splitlines() if line.strip()]
    print(f"{len(records)} bang bao cao chinh")

    results: dict[str, Counter[str]] = {}
    verified_docs: set[str] = set()
    checked_docs: set[str] = set()

    for kind, target, plus, minus in IDENTITIES:
        counter: Counter[str] = Counter()
        for record in records:
            if record["kind"] != kind:
                continue
            for period in ("current", "prior"):
                cells = record[period]
                needed = (target,) + plus + minus
                if not all(code in cells for code in needed):
                    counter["thieu ma"] += 1
                    continue
                expected = cells[target][0]
                total = sum(cells[c][0] for c in plus) - sum(cells[c][0] for c in minus)
                counter["du ma"] += 1
                scale = max(abs(expected), abs(total), 1.0)
                if abs(expected - total) <= REL_TOL * scale:
                    counter["DUNG"] += 1
                    verified_docs.add(record["doc"])
                else:
                    counter["sai"] += 1
                checked_docs.add(record["doc"])
        results[f"{kind}: {target} = " +
                " + ".join(plus) + ("" if not minus else " - " + " - ".join(minus))] = counter

    print("\ndang thuc                                  du ma    DUNG      sai   ty le")
    for name, counter in results.items():
        have = counter["du ma"]
        if not have:
            print(f"  {name:40s} {have:6d}  (khong bang nao du ma)")
            continue
        print(f"  {name:40s} {have:6d} {counter['DUNG']:7d} {counter['sai']:8d}"
              f"   {100 * counter['DUNG'] / have:5.1f}%")

    print(f"\ntai lieu co it nhat mot dang thuc DUNG: {len(verified_docs)}")
    print(f"tai lieu duoc kiem: {len(checked_docs)}")


if __name__ == "__main__":
    main()
