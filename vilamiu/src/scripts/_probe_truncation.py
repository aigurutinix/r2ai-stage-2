"""Is the answer's row inside the part of the table we actually show?

Coverage was measured as "some shown table contains the answer" and came out at
89.3%. But `render_tables` truncates each table to 60 rows and 7000 characters, so
containing the answer and showing it are different things. Every conclusion drawn
from that 89.3% -- including the claim that coverage times reading is an invariant
0.40 -- rests on the difference being small.

This measures it: for each gold question, whether a holder table exists at all,
and whether the answer survives into the rendered text the model is given.

Usage:
  PYTHONPATH=src python scripts/_probe_truncation.py --limit 200
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rescore_unit_fixed import close, parse_number  # noqa: E402

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.generate import render_tables, variable_names  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


def holder_row(grid, answer, context: str):
    """Index of the first row holding the answer, raw or scaled, else None."""

    for index, row in enumerate(grid[1:]):
        for column, cell in enumerate(row):
            value = parse_number(cell)
            if value is None or value == 0:
                continue
            if close(value, answer, 5e-4):
                return index
            scaled = value * lookup_mod.column_scale(grid, column, context)
            if close(scaled, answer, 5e-4):
                return index
            for unit in (1e12, 1e9, 1e6, 1e3):
                if close(value / unit, answer, 5e-4) or close(scaled / unit, answer, 5e-4):
                    return index
    return None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--rank", default="artifacts/_gold_anchor_rank.jsonl")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--tables", type=int, default=8)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    ranking = {}
    for line in (ROOT / args.rank).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ranking[row["id"]] = [
                TableKey(r["doc_name"], int(r["table_id"])) for r in row.get("refs", [])
            ]

    records = [
        json.loads(l) for l in (ROOT / args.gold).read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]

    stat: collections.Counter[str] = collections.Counter()
    lost_rows = []
    for index, record in enumerate(records):
        if stat["checked"] >= args.limit:
            break
        answer = parse_number(record.get("answer"))
        keys = ranking.get(index, [])[: args.tables]
        if answer is None or not keys:
            continue
        stat["checked"] += 1

        names = variable_names(len(keys))
        grids = {n: store.rows(k) for n, k in zip(names, keys)}
        refs = {n: f"{k.doc_name}|table_{k.table_id}" for n, k in zip(names, keys)}
        notes = {}
        for n, k in zip(names, keys):
            meta = store.meta(k)
            notes[n] = f"{getattr(meta, 'unit_page', '')} {getattr(meta, 'unit_doc', '')} " \
                       f"{getattr(meta, 'caption', '')}"

        held = None
        for n in names:
            row_index = holder_row(grids[n], answer, notes[n])
            if row_index is not None:
                held = (n, row_index, len(grids[n]) - 1)
                break
        if held is None:
            continue
        stat["answer in some table"] += 1

        rendered = render_tables(grids, refs)
        # The row survives if the label of the holding row appears in the text the
        # model is given. Comparing the label rather than the number keeps this
        # honest when the same figure appears twice.
        name, row_index, total = held
        label = str(grids[name][row_index + 1][0]).strip()
        if label and label[:40] in rendered:
            stat["and it is shown"] += 1
        else:
            stat["TRUNCATED AWAY"] += 1
            if len(lost_rows) < 5:
                lost_rows.append((record["question"][:78], row_index, total, label[:44]))

    checked = stat["checked"] or 1
    inside = stat["answer in some table"] or 1
    print(f"{stat['checked']} questions, {args.tables} tables each\n")
    print(f"  answer in some retrieved table : {stat['answer in some table']:4d}  "
          f"{100 * stat['answer in some table'] / checked:5.1f}%")
    print(f"  ... and shown to the model     : {stat['and it is shown']:4d}  "
          f"{100 * stat['and it is shown'] / checked:5.1f}%")
    print(f"  ... TRUNCATED AWAY             : {stat['TRUNCATED AWAY']:4d}  "
          f"{100 * stat['TRUNCATED AWAY'] / checked:5.1f}%  "
          f"({100 * stat['TRUNCATED AWAY'] / inside:.1f}% of covered)")
    for question, row_index, total, label in lost_rows:
        print(f"\n  row {row_index} of {total}: {label!r}")
        print(f"    {question}")


if __name__ == "__main__":
    main()
