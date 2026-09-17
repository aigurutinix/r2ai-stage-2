"""Recover explicit panel ``None`` operands and measure answer impact.

This tool joins three layers that were previously reviewed separately:

1. AST analysis identifies ``None`` fields inside a panel ``_rows`` literal.
2. The metric used by the same field in sibling rows is inferred from
   ``_source_value(ticker, year, metric_key)`` calls.
3. A source-derived statement cube supplies the omitted cell, after which the
   original program is executed again with the repaired evidence table.

The report is diagnostic only.  A changed result is a high-value review lead,
not permission to alter a submission without checking the original BTC CSV.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


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


def _source_metric(node: ast.AST) -> str | None:
    """Return the metric key from ``_source_value(..., metric)`` or ``abs(...)``."""
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id == "_source_value":
            if len(node.args) >= 3:
                metric = _constant(node.args[2])
                return str(metric) if metric is not None else None
        if isinstance(node.func, ast.Name) and node.func.id == "abs" and node.args:
            return _source_metric(node.args[0])
    return None


def _scope_for(row: dict, ticker: str, year: int) -> str | None:
    prefix = f"{ticker}_financial_statements_{year}_"
    matches = [str(doc) for doc in row.get("relevant_docs", []) if str(doc).startswith(prefix)]
    scopes = {
        "consolidated" if doc.endswith("_consolidated") else "separate"
        for doc in matches
        if doc.endswith(("_consolidated", "_separate"))
    }
    return next(iter(scopes)) if len(scopes) == 1 else None


class RepairNone(ast.NodeTransformer):
    def __init__(self, repairs: dict[tuple[str, int, str], str]) -> None:
        self.repairs = repairs

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
            for idx, (key_node, value_node) in enumerate(zip(item.keys, item.values)):
                field = _constant(key_node)
                key = (str(ticker), int(year), str(field))
                metric = self.repairs.get(key)
                if metric and isinstance(value_node, ast.Constant) and value_node.value is None:
                    item.values[idx] = ast.Call(
                        func=ast.Name(id="_source_value", ctx=ast.Load()),
                        args=[
                            ast.Constant(value=str(ticker)),
                            ast.Constant(value=int(year)),
                            ast.Constant(value=metric),
                        ],
                        keywords=[],
                    )
        return node


def _run(code: str, frames: dict[str, pd.DataFrame]) -> Any:
    namespace = {"pd": pd, "np": np, "dfs": frames}
    exec(compile(code, "<panel-query>", "exec"), namespace, namespace)
    return namespace.get("result")


def _same_result(left: Any, right: Any) -> bool:
    try:
        if pd.isna(left) and pd.isna(right):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(left, (int, float, np.number)) and isinstance(
        right, (int, float, np.number)
    ):
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-12)
    return left == right


def _cube_index(paths: list[Path], wanted: set[tuple[str, str, str, str | None]]) -> dict:
    candidates: dict[tuple[str, str, str, str | None], list[dict]] = defaultdict(list)
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                key = (
                    str(record.get("ticker")),
                    str(record.get("year")),
                    str(record.get("metric_key")),
                    str(record.get("scope")) if record.get("scope") is not None else None,
                )
                if key in wanted:
                    candidates[key].append(record)
    return candidates


def audit(submission_dir: Path, cube_paths: list[Path]) -> dict:
    submission = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    plans: list[dict] = []
    wanted: set[tuple[str, str, str, str | None]] = set()

    for row in submission:
        code = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        panel = _panel(tree)
        if panel is None:
            continue
        field_metrics: dict[str, set[str]] = defaultdict(set)
        items: list[tuple[str, int, str]] = []
        for item in panel.elts:
            if not isinstance(item, ast.Dict):
                continue
            values = _dict_items(item)
            ticker = _constant(values.get("ticker"))
            year = _constant(values.get("year"))
            if ticker is None or year is None:
                continue
            for field, value_node in values.items():
                if field in {"ticker", "year"}:
                    continue
                metric = _source_metric(value_node)
                if metric:
                    field_metrics[field].add(metric)
                elif isinstance(value_node, ast.Constant) and value_node.value is None:
                    items.append((str(ticker), int(year), field))
        repairs = []
        for ticker, year, field in items:
            metrics = sorted(field_metrics.get(field, set()))
            if len(metrics) != 1:
                repairs.append(
                    {"ticker": ticker, "year": year, "field": field, "status": "metric-ambiguous", "metrics": metrics}
                )
                continue
            metric = metrics[0]
            scope = _scope_for(row, ticker, year)
            key = (ticker, str(year), metric, scope)
            wanted.add(key)
            repairs.append(
                {"ticker": ticker, "year": year, "field": field, "metric_key": metric, "scope": scope, "cube_key": key}
            )
        if repairs:
            plans.append({"row": row, "tree": tree, "repairs": repairs})

    cube = _cube_index(cube_paths, wanted)
    findings = []
    for plan in plans:
        row = plan["row"]
        recoverable: dict[tuple[str, int, str], str] = {}
        source_rows = []
        repair_report = []
        for repair in plan["repairs"]:
            if "cube_key" not in repair:
                repair_report.append(repair)
                continue
            candidates = cube.get(tuple(repair["cube_key"]), [])
            values = {float(item["value"]) for item in candidates if item.get("value") is not None}
            if not candidates:
                repair_report.append({**repair, "status": "not-found"})
                continue
            if len(values) != 1:
                repair_report.append({**repair, "status": "value-ambiguous", "candidate_values": sorted(values)})
                continue
            source = candidates[0]
            recoverable[(repair["ticker"], repair["year"], repair["field"])] = repair["metric_key"]
            source_rows.append(
                {
                    "ticker": repair["ticker"],
                    "year": repair["year"],
                    "metric_key": repair["metric_key"],
                    "raw": source.get("raw"),
                    "typed_factor": 1.0,
                    "scale": source.get("scale", 1.0),
                    "source_table": source.get("table_ref"),
                    "source_csv": Path(str(source.get("csv_path", ""))).name,
                    "row_idx": source.get("row_idx"),
                    "col_idx": source.get("col_idx"),
                }
            )
            repair_report.append(
                {**repair, "status": "recovered", "raw": source.get("raw"), "value": source.get("value"), "table_ref": source.get("table_ref")}
            )

        before = row.get("answer")
        after = None
        error = None
        changed = False
        if recoverable and len(source_rows) == len(recoverable):
            try:
                evidence = row.get("evidence", [])
                if len(evidence) != 1:
                    raise ValueError("impact simulation currently requires exactly one evidence frame")
                csv_path = submission_dir / evidence[0]["csv_path"]
                frame = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
                frame = pd.concat([frame, pd.DataFrame(source_rows)], ignore_index=True)
                repaired_tree = RepairNone(recoverable).visit(ast.parse(str(row["pandas_query"])))
                ast.fix_missing_locations(repaired_tree)
                after = _run(ast.unparse(repaired_tree), {evidence[0]["variable"]: frame})
                changed = not _same_result(before, after)
            except Exception as exc:  # diagnostic: preserve the exception in the report
                error = f"{type(exc).__name__}: {exc}"
        findings.append(
            {
                "id": int(row["id"]),
                "question": row.get("question"),
                "stored_answer": before,
                "recomputed_answer": after,
                "answer_changed": changed,
                "simulation_error": error,
                "repairs": repair_report,
            }
        )

    return {
        "submission": str(submission_dir.resolve()),
        "cubes": [str(path.resolve()) for path in cube_paths],
        "questions_with_none": len(findings),
        "simulated_count": sum(item["recomputed_answer"] is not None for item in findings),
        "changed_count": sum(item["answer_changed"] for item in findings),
        "changed_ids": [item["id"] for item in findings if item["answer_changed"]],
        "simulation_error_count": sum(bool(item["simulation_error"]) for item in findings),
        "findings": findings,
        "claim_limit": "Verify every changed candidate against its original BTC CSV before building a submission.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--cube", action="append", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    cubes = args.cube or [Path("build/statement_cube.jsonl")]
    targeted = Path("build/statement_cube_missing_operands_targeted.jsonl")
    if args.cube is None and targeted.is_file():
        cubes.append(targeted)
    report = audit(args.submission, cubes)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
