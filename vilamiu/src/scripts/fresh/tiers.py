"""Rank verified addresses by how much two independent readings agree.

A verified quote proves the model looked at the cell it names. It does not prove that
cell answers the question — the model can copy a real figure off the wrong row. So a
second reading of the same tables in the opposite order supplies the missing signal:
position bias cannot survive the reversal, and an address both orders choose was chosen
for what the row says.

Three tiers come out of it, and they are meant to be spliced in that order:

  same cell      both readings name the same table, row and column
  same figure    different cells, identical value — usually the same line printed twice,
                 in a statement and again in its note, so the figure is right either way
  one reading    only one order produced a verified address

The tiers are reported against the incumbent as well: a row where the incumbent's answer
is not any cell in the document is a row it cannot be right about, and that is where a
replacement costs nothing.

Usage:
  python scripts/fresh/tiers.py --a artifacts/fresh/tab_plan.jsonl \\
      --b artifacts/fresh/tab_plan_rev.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in (json.loads(line) for line
                                 in path.read_text(encoding="utf-8").splitlines()
                                 if line.strip())}


def value_of(entry: dict) -> float | None:
    import parse_statements as ps
    return ps.parse_vn_number(str(entry.get("quoted", "")))


def main() -> None:
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    parser = argparse.ArgumentParser()
    parser.add_argument("--a", default="artifacts/fresh/tab_plan.jsonl")
    parser.add_argument("--b", default="artifacts/fresh/tab_plan_rev.jsonl")
    parser.add_argument("--blocks", default="artifacts/fresh/blocks.json")
    parser.add_argument("--not-a-cell", default="artifacts/fresh/not_a_cell.json")
    parser.add_argument("--out", default="artifacts/fresh/tiers.json")
    args = parser.parse_args()

    first, second = load(ROOT / args.a), load(ROOT / args.b)
    blocks = json.loads((ROOT / args.blocks).read_text(encoding="utf-8"))
    single = set(blocks.get("tien — MOT O", [])) | set(blocks.get("khac — don gian", []))
    doubtful = set()
    path = ROOT / args.not_a_cell
    if path.exists():
        doubtful = set(json.loads(path.read_text(encoding="utf-8"))["doubtful"])

    tiers: dict[str, list[int]] = {"cung o": [], "cung con so": [], "mot lan doc": []}
    for qid, entry in first.items():
        other = second.get(qid)
        if other is None:
            tiers["mot lan doc"].append(qid)
        elif (entry["table_id"], entry["row"], entry["col"]) == (
                other["table_id"], other["row"], other["col"]):
            tiers["cung o"].append(qid)
        else:
            left, right = value_of(entry), value_of(other)
            if left is not None and right is not None and abs(left - right) <= 0.01:
                tiers["cung con so"].append(qid)
            else:
                tiers.setdefault("hai lan doc KHAC nhau", []).append(qid)

    counts = Counter({name: len(ids) for name, ids in tiers.items()})
    print(f"doc lan 1: {len(first)} dia chi | doc lan 2: {len(second)}\n")
    for name, _count in counts.most_common():
        ids = tiers[name]
        in_single = [i for i in ids if i in single]
        replaceable = [i for i in in_single if i in doubtful]
        print(f"  {len(ids):5d}  {name}")
        print(f"         trong khoi mot-o: {len(in_single)}"
              f" | trong so do incumbent KHONG the dung: {len(replaceable)}")

    (ROOT / args.out).write_text(json.dumps(
        {"tiers": tiers,
         "single_cell": sorted(single),
         "doubtful": sorted(doubtful)}, ensure_ascii=False), encoding="utf-8")
    print(f"\n-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
