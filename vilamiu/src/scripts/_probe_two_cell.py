"""Can the unlocatable answers be expressed as two cells instead of one?

205 questions have a direct answer that matches no single cell. Most are derived —
a difference, a sum, a ratio — so no cell can hold them, and that is the ceiling
of the one-cell search, not a failure of it. Those are also the half of the
question set we lose on.

The danger here is different from everything before. A single-cell search over
~2,900 cells either matches or does not; a *pair* search covers ~4 million
combinations, and at any workable tolerance some pair will hit almost any number
by coincidence. A coincidental pair produces a program that returns the right
figure for the wrong reason — it would ship, score, and teach nothing.

So this measures the thing that decides whether the idea is usable at all:
**how often the match is unique.** Restricting both cells to the same table cuts
the space eightfold and is what a real difference or ratio looks like anyway. If
most questions come back with one candidate pair, the search is identifying
structure; if they come back with dozens, it is fitting noise and must not ship.

Usage:  PYTHONPATH=src python scripts/_probe_two_cell.py [limit]
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

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
BASE = ROOT / "submissions" / "direct14b.zip"
DIRECT = ROOT / "artifacts" / "direct_all.jsonl"
RATE_UNITS = {"phan_tram", "lan", "vong"}
REL_TOL = 5e-4


def cell_value(text: str) -> float | None:
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    raw = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        value = float(raw)
    except ValueError:
        return None
    return -value if negative else value


def close(a: float, b: float) -> bool:
    if a == b:
        return True
    scale = max(abs(a), abs(b))
    return (scale > 0 and abs(a - b) / scale <= REL_TOL) or abs(a - b) <= 0.01


def main() -> None:
    parsed = {q.id: q for q in parse_all(
        ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")}
    with zipfile.ZipFile(BASE) as z:
        base = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    direct = {}
    for line in DIRECT.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("value") is not None:
                direct[row["id"]] = row
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    stats: Counter[str] = Counter()
    candidates_hist: Counter[int] = Counter()
    ops_hist: Counter[str] = Counter()
    examined = 0

    for qid, row in direct.items():
        question = parsed[qid]
        target = float(row["value"])
        if target == 0:
            continue
        asked = UNIT_SCALE.get(question.target_unit) or 1.0
        is_rate = question.target_unit in RATE_UNITS

        # Cells per frame, already converted to the asked unit for the additive
        # operations. Ratios cancel the scale, so they use the raw values.
        per_frame: dict[str, list[tuple[float, float, int, int]]] = {}
        for name, (doc, tid) in zip(row["variables"], row["keys"]):
            key = TableKey(doc, int(tid))
            grid = store.rows(key)
            meta = store.meta(key)
            fallback = f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            cells: list[tuple[float, float, int, int]] = []
            width = max((len(line) for line in grid), default=0)
            for c in range(width):
                scale = lookup_mod.column_scale(grid, c, fallback)
                for r, line in enumerate(grid[1:], start=1):
                    if c >= len(line):
                        continue
                    raw = cell_value(line[c])
                    if raw is None or raw == 0:
                        continue
                    cells.append((raw, raw * scale / asked, r - 1, c))
            if cells:
                per_frame[name] = cells

        # A single cell first: if one matches, this question was never in the
        # unlocatable set and should not be counted here.
        single = any(
            close(conv, target) for cells in per_frame.values() for _, conv, _, _ in cells)
        if single:
            stats["single_cell_already"] += 1
            continue

        examined += 1
        if LIMIT and examined > LIMIT:
            examined -= 1
            break

        found: list[str] = []
        for name, cells in per_frame.items():
            n = len(cells)
            if n > 400:  # keep the pair search bounded on very large tables
                cells = cells[:400]
                n = 400
            for i in range(n):
                raw_a, conv_a, ra, ca = cells[i]
                for j in range(n):
                    if i == j:
                        continue
                    raw_b, conv_b, rb, cb = cells[j]
                    if close(conv_a - conv_b, target):
                        found.append(f"diff:{name}:{ra},{ca}-{rb},{cb}")
                    elif close(conv_a + conv_b, target) and i < j:
                        found.append(f"sum:{name}:{ra},{ca}+{rb},{cb}")
                    elif is_rate and raw_b != 0:
                        quotient = raw_a / raw_b
                        if close(quotient, target) or close(quotient * 100.0, target):
                            found.append(f"ratio:{name}:{ra},{ca}/{rb},{cb}")
            if len(found) > 60:
                break

        candidates_hist[min(len(found), 10)] += 1
        if not found:
            stats["no_two_cell_match"] += 1
        else:
            stats["TWO_CELL_MATCH"] += 1
            ops_hist[found[0].split(":", 1)[0]] += 1
            if len(found) == 1:
                stats["unique_pair"] += 1

    print(f"examined {examined} questions with no single-cell match\n")
    for key, count in stats.most_common():
        print(f"  {key:24s} {count:5d}")
    print(f"\n  candidate pairs per question (10 = ten or more):")
    for k in sorted(candidates_hist):
        print(f"    {k:2d} candidates: {candidates_hist[k]:4d}")
    print(f"\n  first-match operation: {dict(ops_hist)}")
    matched = stats["TWO_CELL_MATCH"] or 1
    print(f"\n  unique among matched: {stats['unique_pair']}/{matched} = "
          f"{stats['unique_pair'] / matched:.1%}")
    print("\n  A high unique share means the pair search finds structure and the")
    print("  program can ship. A low one means many pairs hit the number by")
    print("  coincidence, and shipping any of them is guessing with extra steps.")


if __name__ == "__main__":
    main()
