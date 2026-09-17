"""Recompile existing plans with the current compile_plan (after num() fix).

Extracts cell coordinates from generated code, reads pre-parsed values from the
corpus, and re-emits plans with safe_num() fallback. No LLM needed.

Usage:  python scripts/recompile_plans.py [--in planned.jsonl] [--out planned.jsonl]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.plan_cells import OPERATIONS, Cell, Plan, compile_plan  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# v0 = num(df, 9, 1) * 1000000.0
# v1 = num(df1, 3, 2) * 1.0
NUM_RE = re.compile(r"(\w+)\s*=\s*num\((\w+),\s*(-?\d+),\s*(-?\d+)\)\s*\*\s*([\d.eE+.-]+)")


def extract_cells(code: str, variables: list[str]) -> list[tuple[str, int, int, float]]:
    """Parse cells from generated code. Returns [(var, row-1, col, raw_scale), ...]."""
    cells = []
    for match in NUM_RE.finditer(code):
        var = match.group(1)
        row_offset = int(match.group(3))
        col = int(match.group(4))
        raw_scale = float(match.group(5))
        cells.append((var, row_offset, col, raw_scale))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="in_path", default="artifacts/planned.jsonl")
    parser.add_argument("--out", dest="out_path", default="artifacts/planned.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    in_path = Path(args.in_path)
    if not in_path.exists():
        raise SystemExit(f"{in_path} not found")

    rows = []
    for line in in_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("ok"):
                rows.append(row)

    print(f"{len(rows)} plans to recompile")
    if args.limit:
        rows = rows[: args.limit]

    store = TableStore.load(root / "artifacts" / "tables.parquet")

    stats: Counter = Counter()
    out_path = Path(args.out_path)
    out_handle = out_path.open("w", encoding="utf-8")

    for row in rows:
        keys: list[TableKey] = [TableKey(str(pair[0]), int(pair[1])) for pair in row["keys"]]
        variables: list[str] = list(row["variables"])
        code: str = row["code"]
        op: str = row.get("op", "")

        cells_parsed = extract_cells(code, variables)
        if not cells_parsed or not variables:
            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["skipped (no cells parsed)"] += 1
            continue

        # Reconstruct plan
        frame_to_table = {name: idx for idx, name in enumerate(variables)}
        plan_cells: list[Cell] = []
        scales: list[float] = []
        cell_values: list[float] = []
        used_tables: list[int] = []

        for var, row_offset, col, raw_scale in cells_parsed:
            table = frame_to_table.get(var, 0)
            cell = Cell(table, row_offset + 1, col)  # row in Plan is 1-indexed
            plan_cells.append(cell)
            if table not in used_tables:
                used_tables.append(table)

            # Pre-parse the actual cell value from the corpus
            key = keys[table]
            grid = store.rows(key)
            if cell.row >= len(grid) or cell.column >= len(grid[cell.row]):
                cell_values.append(0.0)
            else:
                parsed = lookup_mod._parse_cell(grid[cell.row][cell.column])
                cell_values.append(parsed if parsed is not None else 0.0)

            # Original scale = raw_scale (already includes column_scale in the
            # original plan). We need to recover the pure column_scale.
            # raw_scale = column_scale / out_scale
            # But we don't know the original out_scale. Simpler: re-derive from grid.
            meta = store.meta(key)
            col_scale = lookup_mod.column_scale(
                grid, col, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
            )
            scales.append(col_scale)

        if op not in OPERATIONS:
            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["skipped (bad op)"] += 1
            continue

        plan = Plan(op, tuple(plan_cells))
        if not plan.valid:
            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["skipped (invalid plan)"] += 1
            continue

        # Re-derive out_scale from the original code: look for result = round(.../X, 2)
        out_scale = 1.0  # default for ratios
        if op not in ("ratio", "ratio_pct", "growth_pct"):
            m = re.search(r"round\(.*\s*/\s*([\d.eE+.-]+)\s*[,)]", code)
            if m:
                out_scale = float(m.group(1))

        frame_names = ["df"] if len(used_tables) == 1 else [f"df{i + 1}" for i in range(len(used_tables))]

        new_code = compile_plan(plan, scales, out_scale, frame_names, cell_values)
        tables = {n: store.rows(keys[t]) for n, t in zip(frame_names, used_tables)}
        outcome = run_query(new_code, tables)

        result = dict(row)
        result["code"] = new_code
        result["ok"] = outcome.ok
        result["value"] = outcome.value
        if not outcome.ok:
            result["error"] = outcome.error[:200]

        out_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
        stats["ok" if outcome.ok else "crash"] += 1

    out_handle.close()
    print("\n" + "  ".join(f"{k}={v}" for k, v in stats.most_common()))


if __name__ == "__main__":
    main()
