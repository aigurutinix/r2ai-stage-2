"""Check the unit arithmetic of every single-cell answer against the cell it read.

No gold answers are needed. Three facts are already known per question: the raw
cell the program read (from tracing `num`), the unit the table declares, and the
unit the question asks for. The answer is then determined arithmetic, and any
disagreement is a definite defect rather than a suspicion.

This is the check that the self-evident-defect audit cannot make. An answer of
316.31 for a question asking "nghìn tỷ" looks perfectly reasonable on its own; only
against the cell it came from (316.305.014.560, i.e. 0.32 nghìn tỷ) is it visibly a
thousand times out.

Usage:
  PYTHONPATH=src python scripts/audit_unit_math.py --base aimed.zip
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rescore_unit_fixed import parse_number  # noqa: E402
from trace_answer import logged_reads, read_grid  # noqa: E402

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


CSV_RE = re.compile(r"([A-Za-z0-9_.\-]+)_table_(\d+)\.csv$")


def unit_context(store, csv_path: str) -> str:
    """The text a table declares its unit in: page note, document note, caption.

    Passing the csv path instead — which is what this script did first — makes
    `column_scale` return 1.0 for every table, so a statement written in triệu
    đồng is read as đồng and a correct answer is reported as a million times out.
    id=174 was hand-verified correct and the audit called it wrong.
    """

    match = CSV_RE.search(csv_path)
    if match is None:
        return ""
    try:
        meta = store.meta(TableKey(match.group(1), int(match.group(2))))
    except Exception:  # noqa: BLE001 - a missing table simply has no context
        return ""
    return (f"{getattr(meta, 'unit_page', '')} {getattr(meta, 'unit_doc', '')} "
            f"{getattr(meta, 'caption', '')}")

POWERS = (1e-12, 1e-9, 1e-6, 1e-3, 1e3, 1e6, 1e9, 1e12)


def as_number(value):
    """A JSON answer field as a float, without Vietnamese digit-grouping rules."""

    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return parse_number(value)


def close(a: float, b: float, tolerance: float = 5e-3) -> bool:
    if a is None or b is None:
        return False
    if b == 0:
        return abs(a) < 1e-9
    return abs(a - b) / abs(b) <= tolerance


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--show", type=int, default=10)
    args = parser.parse_args()

    import pandas as pd

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    questions = {
        question.id: question
        for question in parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                                  ROOT / "data" / "code_stock.csv")
    }

    tally: collections.Counter[str] = collections.Counter()
    offenders = []

    with zipfile.ZipFile(ROOT / "submissions" / args.base) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
        rows = payload if isinstance(payload, list) else (
            payload.get("predictions") or list(payload.values())[0])

        for row in rows:
            question = questions.get(row["id"])
            # NOT `parse_number`: the answer field is already a JSON number, and
            # that parser reads "36871.716535" as a Vietnamese thousands-grouped
            # integer — 36,871,716,535. Every answer came out a million times too
            # large and the audit reported zero correct, which was the audit's bug
            # and not the pipeline's.
            answer = as_number(row.get("answer"))
            if question is None or answer is None:
                tally["bỏ qua"] += 1
                continue
            want_scale = question.unit_scale
            if not want_scale:
                # A ratio or a count has no money scale, so there is no arithmetic
                # to check here.
                tally["không có đơn vị tiền"] += 1
                continue

            frames, grids = {}, {}
            for item in row.get("evidence") or []:
                try:
                    grid = read_grid(archive, item["csv_path"])
                except KeyError:
                    continue
                if grid:
                    frames[item["variable"]] = pd.DataFrame(grid[1:], columns=grid[0])
                    grids[item["variable"]] = (grid, item["csv_path"])
            reads = logged_reads(row.get("pandas_query") or "", frames)
            # Only a single-cell answer is decidable this way: with several cells
            # the arithmetic between them is unknown and a mismatch says nothing.
            unique = {(v, r, c) for v, r, c, _ in reads}
            if len(unique) != 1:
                tally["không phải một ô"] += 1
                continue

            variable, row_index, column, cell = reads[0]
            grid, csv_path = grids.get(variable, (None, ""))
            if grid is None:
                tally["thiếu bảng"] += 1
                continue
            raw = parse_number(cell)
            if raw is None or raw == 0:
                tally["ô không phải số"] += 1
                continue

            column_scale = lookup_mod.column_scale(
                grid, column, unit_context(store, csv_path))
            in_dong = abs(raw) * column_scale
            expected = in_dong / want_scale
            if close(abs(answer), expected):
                tally["ĐÚNG số học"] += 1
                continue

            # Which power of ten would reconcile them, if any.
            factor = None
            for power in POWERS:
                if close(abs(answer), expected * power):
                    factor = power
                    break
            if factor is not None:
                tally[f"LỆCH ×{factor:g}"] += 1
                offenders.append((row["id"], factor, row.get("answer"), expected,
                                  cell, question.question[:76]))
            else:
                tally["lệch không phải bậc 10"] += 1

    total = sum(tally.values())
    print(f"{args.base}: {total} câu\n")
    for name, count in tally.most_common():
        print(f"  {name:26s} {count:5d}  {count / max(total, 1):5.1%}")
    print()
    for qid, factor, shipped, expected, cell, text in offenders[: args.show]:
        print(f"  id={qid:<5d} ×{factor:g}  nộp={shipped}  đúng≈{expected:,.2f}"
              f"  ô={cell!r}")
        print(f"        {text}")


if __name__ == "__main__":
    main()
