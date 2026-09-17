"""Can a directly-answered number be traced back to a cell, and does it agree?

Two numbers decide whether the direct path is worth a submission.

**Localisation.** EXECUTION scores the program, not the answer, so a bare number
is worthless on its own. It becomes a program if the value can be found in one of
the tables the model was shown: then `num(df_k, r, c)` reads a real frame and is
admissible. The search is principled rather than a scale hunt — a cell is a match
when `num(cell) * column_scale / asked_scale` lands on the model's number, which
is the same conversion our 42.8% branch performs.

**Agreement.** Correctness cannot be measured without gold, but the label matcher
is pinned at 42.8% on its own pool, so agreement with it says whether the direct
answer is in the same league or noise.

Usage:  PYTHONPATH=src python scripts/_probe_direct_locate.py artifacts/direct_probe.jsonl
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import UNIT_SCALE, parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "artifacts" / "direct_probe.jsonl"
BASE = ROOT / "submissions" / "sub15.zip"
# Loose enough to absorb the model's 2-decimal rounding, tight enough that a
# different cell does not pass.
REL_TOL = 5e-4


def cell_value(text: str) -> float | None:
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    if "," in raw:
        raw = raw.replace(".", "").replace(",", ".")
    else:
        raw = raw.replace(".", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= REL_TOL or abs(a - b) <= 0.01


def main() -> None:
    parsed = {q.id: q for q in parse_all(
        ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")}
    rows = [
        json.loads(line)
        for line in CACHE.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    with zipfile.ZipFile(BASE) as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    stats: Counter[str] = Counter()
    agree_located = agree_all = located_n = compared = 0

    for row in rows:
        qid = row["id"]
        value = row.get("value")
        if value is None:
            stats["no_number"] += 1
            continue
        question = parsed[qid]
        asked = UNIT_SCALE.get(question.target_unit) or 1.0

        found = None
        for name, (doc, tid) in zip(row["variables"], row["keys"]):
            key = TableKey(doc, int(tid))
            grid = store.rows(key)
            meta = store.meta(key)
            fallback = f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            for c in range(max(len(line) for line in grid)):
                scale = lookup_mod.column_scale(grid, c, fallback)
                factor = scale / asked
                for r, line in enumerate(grid[1:], start=1):
                    if c >= len(line):
                        continue
                    raw = cell_value(line[c])
                    if raw is None or raw == 0:
                        continue
                    if close(raw * factor, float(value)):
                        found = (name, r - 1, c, factor)
                        break
                if found:
                    break
            if found:
                break

        if found is None:
            stats["value_not_in_tables"] += 1
        else:
            stats["LOCATED"] += 1
            located_n += 1

        try:
            shipped = float(base[qid].get("answer") or 0)
        except (TypeError, ValueError):
            continue
        compared += 1
        if close(float(value), shipped):
            agree_all += 1
            if found:
                agree_located += 1

    total = sum(stats.values()) or 1
    print(f"{CACHE.name}: {len(rows)} replies\n")
    for key, count in stats.most_common():
        print(f"  {key:22s} {count:4d}  ({count / total:5.1%})")
    print(f"\n  agreement with the shipped answer: {agree_all}/{compared} = "
          f"{agree_all / max(compared, 1):.1%}")
    print(f"  among the located ones           : {agree_located}/{located_n} = "
          f"{agree_located / max(located_n, 1):.1%}")
    print("\n  LOCATED is the ceiling on how many of these can ship as programs.")
    print("  Agreement is a league check, not accuracy: the branch it is compared")
    print("  against scores 42.8% on its own pool.")


if __name__ == "__main__":
    main()
