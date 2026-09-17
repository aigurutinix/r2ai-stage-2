"""Rescore today's runs against a gold set whose units match the questions.

`easy_full.jsonl` stores the raw cell as the answer. Where a question asks for
"tỷ đồng" and the table prints VND, that makes the correct answer wrong: 27% of
200 records read this way. Our system prompt instructs the model to convert, as
the rules require, so every A/B run today was penalising the behaviour it asked
for.

The model's own outputs were saved per question, so the correction needs no API
calls. This rebuilds the gold in the unit each question names and rescores every
`artifacts/_sc_*.jsonl` twice, side by side.

Usage:
  PYTHONPATH=src python scripts/_rescore_unit_fixed.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.store import TableKey, TableStore  # noqa: E402

ASKED = (("nghìn tỷ", 1e12), ("tỷ", 1e9), ("triệu", 1e6),
         ("nghìn", 1e3), ("ngàn", 1e3))


def asked_unit(question: str):
    lowered = question.lower()
    for name, scale in ASKED:
        if re.search(rf"bao nhiêu[^?]*\b{re.escape(name)}\b", lowered):
            return scale
        if re.search(rf"\({re.escape(name)}\s*đồng\)", lowered):
            return scale
    return None


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


def table_scale(grid, caption: str) -> float:
    text = " ".join(str(c) for c in grid[0]) + " " + str(caption)
    if len(grid) > 1:
        text += " " + " ".join(str(c) for c in grid[1])
    lowered = text.lower()
    if "nghìn tỷ" in lowered:
        return 1e12
    if "triệu" in lowered:
        return 1e6
    if re.search(r"\btỷ\b", lowered):
        return 1e9
    return 1.0


def close(a, b, tol: float = 2e-3) -> bool:
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    scale = max(abs(a), abs(b))
    return a == b or (scale > 0 and abs(a - b) / scale <= tol)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")

    corrected = {}
    raw_gold = {}
    for line in (ROOT / "artifacts" / "easy_full.jsonl").read_text(
            encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        answer = parse_number(record.get("answer"))
        refs = record.get("relevant_tables") or []
        rid = record.get("id")
        if answer is None or not refs or rid is None:
            continue
        raw_gold[rid] = answer
        wanted = asked_unit(record["question"])
        if wanted is None:
            corrected[rid] = answer
            continue
        doc, tid = refs[0].rsplit("|table_", 1)
        grid = store.rows(TableKey(doc, int(tid)))
        if not grid:
            corrected[rid] = answer
            continue
        # Only rescale when the stored answer really is a cell of that table: a
        # derived answer is already in whatever unit the generator chose.
        is_cell = any(
            parse_number(cell) == answer
            for row in grid[1:] for cell in row if parse_number(cell) is not None
        )
        if not is_cell:
            corrected[rid] = answer
            continue
        meta = store.meta(TableKey(doc, int(tid)))
        scale = table_scale(grid, getattr(meta, "caption", ""))
        corrected[rid] = answer * scale / wanted

    changed = sum(1 for k in corrected if not close(corrected[k], raw_gold[k]))
    print(f"gold rebuilt: {len(corrected)} records, {changed} rescaled "
          f"({100 * changed / max(len(corrected), 1):.1f}%)\n")

    print(f"{'run':34s} {'raw gold':>9s} {'unit-fixed':>11s} {'delta':>7s}")
    for path in sorted(glob.glob(str(ROOT / "artifacts" / "_sc_*.jsonl"))):
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        if not rows:
            continue
        old = new = usable = 0
        for row in rows:
            rid = row.get("id")
            values = row.get("values") or []
            if rid not in corrected:
                continue
            usable += 1
            if any(close(v, raw_gold[rid]) for v in values):
                old += 1
            if any(close(v, corrected[rid]) for v in values):
                new += 1
        if not usable:
            continue
        name = Path(path).stem.replace("_sc_", "")
        print(f"  {name:32s} {old / usable:8.1%} {new / usable:10.1%} "
              f"{(new - old) / usable:+6.1%}")


if __name__ == "__main__":
    main()
