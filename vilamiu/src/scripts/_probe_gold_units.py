"""Is the local gold set scoring the unit conversion backwards?

Read by hand, record [3] of `easy_full.jsonl` asks for a figure "tỷ đồng" and
gives 530.339.169.047 as the answer -- the raw VND cell, which is 530.34 tỷ. A
model that converts, as the rules require and as our own system prompt instructs,
is marked wrong; a model that echoes the cell is marked right.

Three of the four records read alongside it did not show the fault, because the
question asked in the unit the table already prints. So the defect only bites
where the two differ, and its size decides whether every A/B run against this set
today measured anything at all.

For each record this checks whether the gold answer is literally a cell of the
gold table, and whether the question asks for a unit that cell is not already in.

Usage:
  PYTHONPATH=src python scripts/_probe_gold_units.py --limit 200
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.store import TableKey, TableStore  # noqa: E402

# What the question asks to be answered in, and the VND each unit is worth.
ASKED = (
    ("nghìn tỷ", 1e12),
    ("tỷ", 1e9),
    ("triệu", 1e6),
    ("nghìn", 1e3),
    ("ngàn", 1e3),
)


def asked_unit(question: str):
    lowered = question.lower()
    for name, scale in ASKED:
        if re.search(rf"bao nhiêu[^?]*\b{re.escape(name)}\b", lowered):
            return name, scale
        if re.search(rf"\({re.escape(name)}\s*đồng\)", lowered):
            return name, scale
    return None, None


def parse_number(text):
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    cleaned = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if negative else value


def table_unit(grid, caption: str):
    """The unit the table prints in, read off its headers and caption."""

    text = " ".join(str(c) for c in grid[0]) + " " + str(caption)
    if len(grid) > 1:
        text += " " + " ".join(str(c) for c in grid[1])
    lowered = text.lower()
    if "nghìn tỷ" in lowered:
        return "nghìn tỷ", 1e12
    if "triệu" in lowered:
        return "triệu", 1e6
    if re.search(r"\btỷ\b", lowered):
        return "tỷ", 1e9
    return "vnd", 1.0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    records = [
        json.loads(l) for l in (ROOT / args.gold).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ][: args.limit]

    stat: collections.Counter[str] = collections.Counter()
    examples = []
    for record in records:
        answer = parse_number(record.get("answer"))
        refs = record.get("relevant_tables") or []
        if answer is None or not refs:
            stat["unusable"] += 1
            continue
        doc, tid = refs[0].rsplit("|table_", 1)
        grid = store.rows(TableKey(doc, int(tid)))
        if not grid:
            stat["no table"] += 1
            continue
        meta = store.meta(TableKey(doc, int(tid)))

        raw_cell = any(
            parse_number(cell) is not None and parse_number(cell) == answer
            for line in grid[1:] for cell in line
        )
        name, scale = asked_unit(record["question"])
        t_name, t_scale = table_unit(grid, getattr(meta, "caption", ""))

        if name is None:
            stat["question names no unit"] += 1
            continue
        if not raw_cell:
            stat["answer is derived, not a cell"] += 1
            continue
        if abs(scale - t_scale) < 1:
            stat["units agree — answer is right either way"] += 1
        else:
            stat["UNITS DIFFER — gold is the unscaled cell"] += 1
            if len(examples) < 4:
                examples.append((record["question"][:96], record["answer"],
                                 name, t_name, answer * t_scale / scale))

    total = sum(stat.values()) or 1
    print(f"{total} records\n")
    for name, count in stat.most_common():
        print(f"  {count:4d}  {100 * count / total:5.1f}%  {name}")
    print("\nwhere the units differ, what the answer should have been:")
    for question, gold, asked, printed, converted in examples:
        print(f"\n  Q: {question}")
        print(f"     asks in {asked}, table prints {printed}")
        print(f"     gold says {gold}  ->  correct is {converted:,.2f}")


if __name__ == "__main__":
    main()
