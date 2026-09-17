"""Inspect one or more submission programs after faithful local execution.

The normal release gate only reports whether ``result`` matches the stored
answer.  That is not enough when reviewing selector-heavy financial QA: the
same scalar can be reached after silently dropping a company or choosing the
wrong row.  This tool executes the exact submitted program under the same
restricted builtin contract as ``grader_check.py`` and emits bounded snapshots
of the Pandas objects left in its namespace.

Examples::

    python scripts/inspect_query_runtime.py sub_candidate --ids 379,418
    python scripts/inspect_query_runtime.py sub_candidate --ids 425 --official

It is read-only.  No submission, evidence file, or review log is changed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from grader_check import _SAFE_BUILTINS, _read_csv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _scalar(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _frame_snapshot(frame: pd.DataFrame, max_rows: int, max_cols: int) -> dict[str, Any]:
    bounded = frame.iloc[:max_rows, :max_cols]
    return {
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "columns": [str(value) for value in bounded.columns],
        "rows": [
            {str(column): _scalar(value) for column, value in row.items()}
            for row in bounded.to_dict(orient="records")
        ],
        "truncated_rows": max(0, int(frame.shape[0]) - max_rows),
        "truncated_columns": max(0, int(frame.shape[1]) - max_cols),
    }


def _series_snapshot(series: pd.Series, max_rows: int) -> dict[str, Any]:
    bounded = series.iloc[:max_rows]
    return {
        "length": int(series.shape[0]),
        "name": _scalar(series.name),
        "values": [
            {"index": _scalar(index), "value": _scalar(value)}
            for index, value in bounded.items()
        ],
        "truncated_rows": max(0, int(series.shape[0]) - max_rows),
    }


def inspect_row(
    candidate: Path,
    row: dict[str, Any],
    *,
    official: bool,
    max_rows: int,
    max_cols: int,
) -> dict[str, Any]:
    evidence = row.get("evidence") or []
    csv_paths = {
        str(item["variable"]): (candidate / str(item["csv_path"])).resolve()
        for item in evidence
    }
    dfs = {
        variable: _read_csv(path, typed=official)
        for variable, path in csv_paths.items()
    }
    namespace: dict[str, Any] = {
        "pd": pd,
        "dfs": dfs,
        "__builtins__": _SAFE_BUILTINS,
    }
    if len(dfs) == 1:
        namespace["df"] = next(iter(dfs.values()))

    code = str(row.get("pandas_query") or "").strip()
    if not code:
        raise ValueError("empty pandas_query")
    exec(compile(code, f"<q{row.get('id')}_pandas_query>", "exec"), namespace)  # noqa: S102
    if "result" not in namespace:
        raise RuntimeError("no-result-var")

    frames: dict[str, Any] = {}
    series: dict[str, Any] = {}
    scalars: dict[str, Any] = {}
    for name, value in sorted(namespace.items()):
        if name.startswith("_") or name in {"pd", "dfs"}:
            continue
        if isinstance(value, pd.DataFrame):
            frames[name] = _frame_snapshot(value, max_rows=max_rows, max_cols=max_cols)
        elif isinstance(value, pd.Series):
            series[name] = _series_snapshot(value, max_rows=max_rows)
        elif name not in {"df"} and isinstance(value, (str, int, float, bool)):
            scalars[name] = _scalar(value)

    runtime_result = _scalar(namespace["result"])
    try:
        delta = float(runtime_result) - float(row.get("answer"))
    except (TypeError, ValueError):
        delta = None
    return {
        "id": int(row["id"]),
        "question": row.get("question"),
        "stored_answer": row.get("answer"),
        "runtime_result": runtime_result,
        "result_delta": delta,
        "mode": "official-typed-dfs" if official else "string-dfs",
        "evidence_count": len(evidence),
        "frames": frames,
        "series": series,
        "scalars": scalars,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--ids", required=True, help="comma-separated question IDs")
    parser.add_argument(
        "--official",
        action="store_true",
        help="load evidence with pandas dtype inference like the official grader",
    )
    parser.add_argument("--max-rows", type=int, default=30)
    parser.add_argument("--max-cols", type=int, default=40)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidate = args.candidate.resolve()
    selected = {int(value.strip()) for value in args.ids.split(",") if value.strip()}
    rows = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    missing = sorted(selected - set(by_id))
    if missing:
        raise KeyError(f"unknown question IDs: {missing}")

    report = {
        "candidate": candidate.name,
        "read_only": True,
        "questions": [
            inspect_row(
                candidate,
                by_id[question_id],
                official=args.official,
                max_rows=max(1, args.max_rows),
                max_cols=max(1, args.max_cols),
            )
            for question_id in sorted(selected)
        ],
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
        print(args.out.resolve())
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
