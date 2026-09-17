"""Does the direct answer look like the 42.8% branch or like the 20% one?

The weak-branch pool is exhausted, so any further gain means displacing a branch
that is already earning points. That can lose, so it needs evidence first — and
there is a free measurement available.

Split the direct answers by the branch that produced the shipped answer and look
at agreement in each group. The branches have known accuracies, so the shape of
the split is readable:

  * high agreement with `lexical` (mostly the 42.8% label matcher) means the
    direct path reads tables to roughly that standard;
  * low agreement with `llm_14b` (measured 20-25% on the pools it displaced) means
    the two disagree — and if direct is the better reader, those disagreements are
    where the points are.

If instead agreement is uniformly high everywhere, direct adds nothing new and the
submission slot is better spent elsewhere.

Usage:  PYTHONPATH=src python scripts/_probe_direct_by_branch.py
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import UNIT_SCALE, parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

BASE = ROOT / "submissions" / "noconst.zip"
DIRECT = ROOT / "artifacts" / "direct_all.jsonl"
REL_TOL = 5e-4

KNOWN = {
    "lexical": "42.8% matcher + 5.9% fallback, mixed",
    "llm_14b": "~20-25%",
    "llm_8b": "27%",
    "locate_model": "13.3%",
    "locate_embed": "13.3%",
    "plan": "7.2%",
    "placeholder": "0%",
}


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

    caches: dict[str, dict[int, str]] = {}
    for label, name in (("locate_model", "located.jsonl"),
                        ("locate_embed", "embed_located.jsonl"),
                        ("plan", "planned.jsonl"),
                        ("llm_8b", "gen_helpers.jsonl"),
                        ("llm_14b", "gen14b_merged.jsonl")):
        path = ROOT / "artifacts" / name
        if not path.exists():
            continue
        out: dict[int, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("code"):
                    out[row["id"]] = row["code"].strip()
        caches[label] = out

    def branch_of(record) -> str:
        code = (record.get("pandas_query") or "").strip()
        if code in ("", "result = 0.0"):
            return "placeholder"
        for label, cache in caches.items():
            if cache.get(record["id"]) == code:
                return label
        return "lexical"

    total: Counter[str] = Counter()
    agree: Counter[str] = Counter()
    located: Counter[str] = Counter()
    located_agree: Counter[str] = Counter()

    for qid, row in direct.items():
        record = base.get(qid)
        if record is None:
            continue
        question = parsed[qid]
        branch = branch_of(record)
        try:
            shipped = float(record.get("answer") or 0)
        except (TypeError, ValueError):
            continue
        value = float(row["value"])
        asked = UNIT_SCALE.get(question.target_unit) or 1.0

        hit = False
        for name, (doc, tid) in zip(row["variables"], row["keys"]):
            key = TableKey(doc, int(tid))
            grid = store.rows(key)
            meta = store.meta(key)
            fallback = f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            for c in range(max((len(line) for line in grid), default=0)):
                factor = lookup_mod.column_scale(grid, c, fallback) / asked
                for line in grid[1:]:
                    if c >= len(line):
                        continue
                    raw = cell_value(line[c])
                    if raw is not None and raw != 0 and close(raw * factor, value):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                break

        total[branch] += 1
        agree[branch] += close(value, shipped)
        if hit:
            located[branch] += 1
            located_agree[branch] += close(value, shipped)

    print(f"{sum(total.values())} direct answers compared against {BASE.name}\n")
    print(f"  {'branch':14s} {'n':>4} {'agree':>7} {'located':>8} "
          f"{'agree|loc':>10}   known accuracy")
    for branch in sorted(total, key=lambda b: -total[b]):
        n = total[branch]
        loc = located[branch]
        print(f"  {branch:14s} {n:4d} {agree[branch] / n:6.1%} "
              f"{loc / n:7.1%} {located_agree[branch] / max(loc, 1):9.1%}   "
              f"{KNOWN.get(branch, '')}")
    print("\n  Read the `agree` column against the known accuracy beside it.")
    print("  Agreement well above a branch's accuracy means the two mostly")
    print("  coincide; well below means they disagree, and the disagreements are")
    print("  where a replacement can win or lose.")


if __name__ == "__main__":
    main()
