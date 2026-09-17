"""Stress-test explicit panel ``None`` cells with row-aware counterfactuals.

The static missing-operand audit is intentionally conservative: if a field is
used anywhere downstream, every missing cell for that field is reported.  In
practice a missing 2023 CFO cannot affect a query that filters to 2024, and a
missing EPS cannot affect a year that loses an ROE selector.  This audit
distinguishes those cases by replacing each literal ``None`` with several
values derived from the same field's observed panel distribution and executing
the original program again.

A sensitive result is a review lead, not evidence that any stress value is the
real financial-statement value.  Source verification remains mandatory before
changing a submission.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


_WHITELIST = (
    "abs round len min max sum sorted float int str bool list dict set "
    "range enumerate zip all any isinstance"
).split()
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _WHITELIST}


def _constant(node: ast.AST | None) -> Any:
    return node.value if isinstance(node, ast.Constant) else None


def _dict_items(node: ast.Dict) -> dict[str, ast.AST]:
    return {
        str(_constant(key)): value
        for key, value in zip(node.keys, node.values)
        if _constant(key) is not None
    }


def _panel(tree: ast.AST) -> ast.List | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "_rows":
            return node.value if isinstance(node.value, ast.List) else None
    return None


def _missing_cells(tree: ast.AST) -> list[tuple[str, int, str]]:
    panel = _panel(tree)
    if panel is None:
        return []
    cells: list[tuple[str, int, str]] = []
    for item in panel.elts:
        if not isinstance(item, ast.Dict):
            continue
        values = _dict_items(item)
        ticker = _constant(values.get("ticker"))
        year = _constant(values.get("year"))
        if ticker is None or year is None:
            continue
        for field, value in values.items():
            if field in {"ticker", "year"}:
                continue
            if isinstance(value, ast.Constant) and value.value is None:
                cells.append((str(ticker), int(year), field))
    return cells


class ReplaceNone(ast.NodeTransformer):
    """Replace selected ``_rows`` cells while leaving every other node intact."""

    def __init__(self, replacements: dict[tuple[str, int, str], float]) -> None:
        self.replacements = replacements

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node = self.generic_visit(node)
        if not (
            len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_rows"
            and isinstance(node.value, ast.List)
        ):
            return node
        for item in node.value.elts:
            if not isinstance(item, ast.Dict):
                continue
            values = _dict_items(item)
            ticker = _constant(values.get("ticker"))
            year = _constant(values.get("year"))
            if ticker is None or year is None:
                continue
            for index, (key_node, value_node) in enumerate(zip(item.keys, item.values)):
                field = _constant(key_node)
                key = (str(ticker), int(year), str(field))
                if (
                    key in self.replacements
                    and isinstance(value_node, ast.Constant)
                    and value_node.value is None
                ):
                    item.values[index] = ast.Constant(value=float(self.replacements[key]))
        return node


def _frames(submission_dir: Path, row: dict) -> dict[str, pd.DataFrame]:
    return {
        str(item["variable"]): pd.read_csv(
            submission_dir / item["csv_path"],
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            index_col=None,
        )
        for item in row.get("evidence", [])
    }


def _run(code: str, frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    namespace: dict[str, Any] = {
        "pd": pd,
        "dfs": {key: value.copy() for key, value in frames.items()},
        "__builtins__": _SAFE_BUILTINS,
    }
    if len(frames) == 1:
        namespace["df"] = next(iter(namespace["dfs"].values()))
    exec(compile(code, "<missing-operand-counterfactual>", "exec"), namespace)  # noqa: S102
    return namespace


def _same(left: Any, right: Any) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    except (TypeError, ValueError):
        return left == right


def _stress_values(series: pd.Series) -> list[float]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return [-1.0, 0.0, 1.0]
    low = float(numeric.min())
    median = float(numeric.median())
    high = float(numeric.max())
    magnitude = max(abs(low), abs(high), 1.0)
    values = [low, median, high, -2.0 * magnitude, 2.0 * magnitude]
    unique: list[float] = []
    for value in values:
        if math.isfinite(value) and not any(_same(value, old) for old in unique):
            unique.append(value)
    return unique


def _patched_code(tree: ast.Module, replacements: dict[tuple[str, int, str], float]) -> str:
    # Transform a fresh parse because NodeTransformer mutates its input tree.
    clone = ast.parse(ast.unparse(tree))
    patched = ReplaceNone(replacements).visit(clone)
    ast.fix_missing_locations(patched)
    return ast.unparse(patched)


def audit(submission_dir: Path) -> dict:
    submission_dir = submission_dir.resolve()
    rows = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict] = []
    probe_count = 0

    for row in rows:
        code = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        missing = _missing_cells(tree)
        if not missing:
            continue
        frames = _frames(submission_dir, row)
        try:
            baseline_ns = _run(code, frames)
            baseline = baseline_ns["result"]
            panel = baseline_ns.get("df")
            if not isinstance(panel, pd.DataFrame):
                raise TypeError("program did not retain panel dataframe as df")
        except Exception as exc:
            findings.append(
                {
                    "id": int(row["id"]),
                    "status": "baseline-error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "missing_cells": [list(cell) for cell in missing],
                }
            )
            continue

        # Very broad universe queries (notably q464) can contain hundreds of
        # omitted cells.  Probing every cell independently is quadratic-feeling
        # in practice because each probe rebuilds the full pandas panel.  First
        # run three global envelopes.  A sensitive broad panel can then be
        # narrowed in a dedicated follow-up instead of blocking the full audit.
        if len(missing) > 25:
            changed_outputs: list[dict] = []
            errors: list[str] = []
            for value_index, label in ((0, "low"), (1, "median"), (-1, "high")):
                replacements: dict[tuple[str, int, str], float] = {}
                for cell in missing:
                    field = cell[2]
                    values = (
                        _stress_values(panel[field])
                        if field in panel.columns
                        else [-1.0, 0.0, 1.0]
                    )
                    replacements[cell] = values[value_index]
                probe_count += 1
                try:
                    result = _run(_patched_code(tree, replacements), frames)["result"]
                    if not _same(baseline, result):
                        changed_outputs.append({"scenario": label, "result": result})
                except Exception as exc:
                    errors.append(f"{label}: {type(exc).__name__}: {exc}")
            findings.append(
                {
                    "id": int(row["id"]),
                    "question": row.get("question"),
                    "baseline_result": baseline,
                    "stored_answer": row.get("answer"),
                    "status": (
                        "broad-sensitive" if changed_outputs else "broad-stable-under-stress"
                    ),
                    "missing_cell_count": len(missing),
                    "global_changed_outputs": changed_outputs,
                    "probe_errors": errors[:20],
                }
            )
            continue

        cell_reports: list[dict] = []
        sensitive = False
        errors: list[str] = []
        for cell in missing:
            field = cell[2]
            values = _stress_values(panel[field]) if field in panel.columns else [-1.0, 0.0, 1.0]
            changed_outputs: list[dict] = []
            for value in values:
                probe_count += 1
                try:
                    result = _run(_patched_code(tree, {cell: value}), frames)["result"]
                    if not _same(baseline, result):
                        changed_outputs.append({"value": value, "result": result})
                except Exception as exc:
                    errors.append(f"{cell}: {type(exc).__name__}: {exc}")
            if changed_outputs:
                sensitive = True
            cell_reports.append(
                {
                    "ticker": cell[0],
                    "year": cell[1],
                    "field": field,
                    "stress_values": values,
                    "changed_outputs": changed_outputs,
                    "sensitive": bool(changed_outputs),
                }
            )

        # A row can require several missing operands together before it enters a
        # filter. Probe grouped low/median/high scenarios per ticker/year too.
        grouped: dict[tuple[str, int], list[tuple[str, int, str]]] = defaultdict(list)
        for cell in missing:
            grouped[cell[:2]].append(cell)
        group_reports: list[dict] = []
        for group, cells in grouped.items():
            scenarios: list[dict[tuple[str, int, str], float]] = []
            for value_index in (0, 1, -1):
                replacements: dict[tuple[str, int, str], float] = {}
                for cell in cells:
                    field = cell[2]
                    values = _stress_values(panel[field]) if field in panel.columns else [-1.0, 0.0, 1.0]
                    replacements[cell] = values[value_index]
                scenarios.append(replacements)
            changed_outputs = []
            for replacements in scenarios:
                probe_count += 1
                try:
                    result = _run(_patched_code(tree, replacements), frames)["result"]
                    if not _same(baseline, result):
                        changed_outputs.append(
                            {
                                "values": {cell[2]: value for cell, value in replacements.items()},
                                "result": result,
                            }
                        )
                except Exception as exc:
                    errors.append(f"{group}: {type(exc).__name__}: {exc}")
            if changed_outputs:
                sensitive = True
            group_reports.append(
                {
                    "ticker": group[0],
                    "year": group[1],
                    "fields": [cell[2] for cell in cells],
                    "changed_outputs": changed_outputs,
                    "sensitive": bool(changed_outputs),
                }
            )

        findings.append(
            {
                "id": int(row["id"]),
                "question": row.get("question"),
                "baseline_result": baseline,
                "stored_answer": row.get("answer"),
                "status": "sensitive" if sensitive else "stable-under-stress",
                "cell_reports": cell_reports,
                "group_reports": group_reports,
                "probe_errors": errors[:20],
            }
        )

    sensitive = [
        item
        for item in findings
        if item.get("status") in {"sensitive", "broad-sensitive"}
    ]
    stable = [
        item
        for item in findings
        if item.get("status") in {
            "stable-under-stress",
            "broad-stable-under-stress",
        }
    ]
    return {
        "submission": str(submission_dir),
        "questions_with_literal_none": len(findings),
        "probe_count": probe_count,
        "sensitive_count": len(sensitive),
        "sensitive_ids": [item["id"] for item in sensitive],
        "stable_count": len(stable),
        "stable_ids": [item["id"] for item in stable],
        "findings": findings,
        "claim_limit": (
            "Stability only covers the tested counterfactual range. Sensitivity is a "
            "review signal; verify the real source cell before editing an answer."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.submission)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
