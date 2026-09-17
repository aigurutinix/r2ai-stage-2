"""Settle the unit convention from the corpus, without spending a submission.

Two readings are possible and they differ by a factor of a thousand or more.

  raw-cell      the answer is the parsed cell, and the unit named in the question
                merely describes the unit the table is already denominated in
  converted     the answer is the cell times the table's scale (đồng), divided by
                the unit the question names

The organisers' easy prompt supports the first: the query may use only `df`, must
parse the Vietnamese number itself, and is forbidden from multiplying or dividing by
any correction constant. Under that contract `result` is the raw cell, and `unit` is
a description rather than a target.

But 149 questions ask for "nghìn tỷ đồng" or "trăm tỷ đồng", and no table is ever
denominated in either, so those at least cannot be raw cells. The two readings may
therefore both be true, of different difficulty tiers — the hard generators compile
`_parse_vn_number(...) * binding.scale`, which is đồng.

The corpus can decide it. If the question's unit mirrors the source table's declared
scale, the raw-cell reading holds: questions saying "triệu đồng" will come from tables
scaled 1e6 far more often than chance. If the units are spread across every scale,
conversion is happening.

Usage:  python scripts/fresh/check_unit_convention.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402

SCALE_NAMES = {1.0: "bang dong", 1e3: "nghin dong", 1e6: "trieu dong",
               1e9: "ty dong"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    questions = {}
    for line in (ROOT / "data" / "questions" / "questions.jsonl").read_text(
            encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            questions[record["id"]] = record["question"]

    plan = [json.loads(line) for line in
            (ROOT / "artifacts" / "fresh" / "answer_plan.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]

    grid: dict[str, Counter[str]] = defaultdict(Counter)
    for entry in plan:
        name, _scale = unit_of(questions[entry["id"]])
        if not name:
            continue
        grid[name][SCALE_NAMES.get(entry["scale"], f"x{entry['scale']:g}")] += 1

    order = ["bang dong", "nghin dong", "trieu dong", "ty dong"]
    print("don vi CAU HOI neu  ->  he so cua BANG nguon")
    print(f"  {'cau hoi':16s} " + " ".join(f"{n:>12s}" for n in order))
    for name in ("đồng", "nghìn đồng", "triệu đồng", "tỷ đồng",
                 "trăm tỷ đồng", "nghìn tỷ đồng"):
        counter = grid.get(name)
        if not counter:
            continue
        print(f"  {name:16s} " + " ".join(f"{counter.get(n, 0):12d}" for n in order))

    print("\ndoc ket qua:")
    print("  duong cheo manh  -> don vi cau hoi = don vi bang  -> dap an la O THO")
    print("  khong duong cheo -> co quy doi     -> dap an = dong / don vi cau hoi")

    # The decisive slice: questions whose unit no table can carry.
    impossible = sum(count for name in ("trăm tỷ đồng", "nghìn tỷ đồng")
                     for count in grid.get(name, Counter()).values())
    print(f"\ncau hoi don vi 'tram ty'/'nghin ty' (khong bang nao co he so do): "
          f"{impossible}")
    print("  chung KHONG THE la o tho, nen neu duong cheo manh o cac don vi khac")
    print("  thi quy uoc KHAC NHAU giua cac tang do kho.")


if __name__ == "__main__":
    main()
