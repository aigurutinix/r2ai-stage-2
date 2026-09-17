"""Questions asked in millions of shares that ship a raw share count.

`UNIT_SCALE` maps only currency words, on the stated ground that share counts and
USD carry no VND conversion. That is right about the *rescale* pass, which infers
the unit a table is printed in — but it also means `unit_scale` is None for
"bao nhiêu triệu cổ phiếu", so nothing ever divides the answer by a million. The
figure ships in units of one share against a question asked in millions, which at
a 0.02% tolerance is simply wrong.

Same shape for "triệu USD". This measures how many questions are in each of those
families and how many of them ship a magnitude that proves the division was never
done.

Usage:  PYTHONPATH=src python scripts/_probe_count_units.py
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

# The magnitude above which the asked-for unit cannot be what was shipped. A
# company with a hundred million million shares does not exist; a hundred
# thousand million shares (1e11 raw) does not either.
IMPLAUSIBLE = {"trieu_co_phieu": 1e5, "trieu_usd": 1e5, "usd": 1e11}


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

    p(f"target units over all {len(parsed)} questions:")
    for unit, count in Counter(q.target_unit for q in parsed.values()).most_common():
        p(f"  {unit or '(none)':16s} {count:4d}")

    for unit, cut in IMPLAUSIBLE.items():
        family = [q for q in parsed.values() if q.target_unit == unit]
        over = [q for q in family if abs(answer(q.id)) > cut]
        p(f"\n{unit}: {len(family)} questions, "
          f"{len(over)} ship |answer| > {cut:.0e} (division never applied)")
        for q in family:
            mark = "  WRONG" if abs(answer(q.id)) > cut else "       "
            p(f" {mark} id={q.id:4d} ships {answer(q.id):.6g}")
            p(f"          {q.question[:145]}")

    out = ROOT / "artifacts" / "_probe_count_units.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    for line in lines:
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode("ascii", "replace").decode("ascii"))


if __name__ == "__main__":
    main()
