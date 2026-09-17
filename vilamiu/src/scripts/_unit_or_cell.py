"""Did the reader lose because it read the wrong cell, or because it scaled wrong?

The leaderboard just settled the outcome: swapping 285 one-cell answers for the
two-stage reader's cost 108 questions, so on the rows where the two disagree the
reader converts almost nothing. That number alone does not say why, and the two
causes need opposite fixes — a wrong cell needs better localisation, a wrong factor
needs one line of arithmetic.

The split is free to measure. Both sides read through `num(frame, r, c)`, so trace
both and compare the (table, row, column) each one touches:

  same cell, different answer  -> the factor is the whole error
  different cell               -> localisation is the error

Usage:
  PYTHONPATH=src python scripts/_unit_or_cell.py --other ts_cell.zip
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from vifin.answering.sandbox import frame_from_rows  # noqa: E402

spec = importlib.util.spec_from_file_location("tr", ROOT / "scripts" / "trace_answer.py")
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def traced(name: str) -> dict[int, dict]:
    """Per question: the answer, and the set of cells the program read."""

    out: dict[int, dict] = {}
    with zipfile.ZipFile(ROOT / "submissions" / name) as archive:
        rows = json.loads(archive.read("submission.json").decode("utf-8"))
        grids: dict[str, list] = {}
        for row in rows:
            frames, source = {}, {}
            for item in row.get("evidence") or []:
                path = item["csv_path"]
                if path not in grids:
                    try:
                        grids[path] = tr.read_grid(archive, path)
                    except KeyError:
                        continue
                if path in grids:
                    frames[item["variable"]] = frame_from_rows(grids[path])
                    source[item["variable"]] = path
            reads = tr.logged_reads(row.get("pandas_query") or "", frames) if frames else []
            out[row["id"]] = {
                "answer": row.get("answer"),
                "question": row["question"],
                "cells": {(source.get(v, "?"), r, c) for v, r, c, _ in reads},
                "raw": [cell for _, _, _, cell in reads],
            }
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="aimed.zip")
    parser.add_argument("--other", default="ts_cell.zip")
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()

    base = traced(args.base)
    other = traced(args.other)
    counts: Counter[str] = Counter()
    same_cell = []

    for qid, left in base.items():
        right = other.get(qid)
        if right is None:
            continue
        if str(left["answer"]) == str(right["answer"]):
            continue
        counts["khac dap an"] += 1
        if not left["cells"] or not right["cells"]:
            counts["mot ben khong truy duoc o"] += 1
            continue
        if left["cells"] == right["cells"]:
            counts["cung o, khac dap an"] += 1
            same_cell.append(qid)
        elif left["cells"] & right["cells"]:
            counts["giao nhau mot phan"] += 1
        else:
            counts["khac o hoan toan"] += 1

    for name, value in counts.most_common():
        print(f"  {name}: {value}")

    if same_cell:
        print(f"\nvi du {min(args.show, len(same_cell))} cau cung o khac dap an:")
        for qid in same_cell[:args.show]:
            ratio = ""
            try:
                a, b = float(base[qid]["answer"]), float(other[qid]["answer"])
                if a and b:
                    ratio = f"  ty le={b / a:.6g}"
            except (TypeError, ValueError):
                pass
            print(f"  id={qid} aimed={base[qid]['answer']} "
                  f"may doc={other[qid]['answer']}{ratio}")
            print(f"     o: {sorted(base[qid]['cells'])[:2]}  raw={base[qid]['raw'][:2]}")

    # Where the cells differ, is the reader's factor still the suspect? Compare the
    # ratio between the two answers against the powers of ten a unit error makes.
    factors: Counter[str] = Counter()
    for qid, left in base.items():
        right = other.get(qid)
        if right is None or str(left["answer"]) == str(right["answer"]):
            continue
        try:
            a, b = float(left["answer"]), float(right["answer"])
        except (TypeError, ValueError):
            continue
        if not a or not b:
            continue
        for power in (-6, -3, -2, 0, 2, 3, 6):
            if abs(b / a - 10.0 ** power) < 1e-6 * max(1.0, abs(10.0 ** power)):
                factors[f"x10^{power}"] += 1
                break
        else:
            factors["khong phai luy thua 10"] += 1
    print("\nty le dap an may doc / aimed:")
    for name, value in factors.most_common():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
