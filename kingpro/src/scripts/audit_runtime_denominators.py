"""Audit the values that actually reach division denominators at runtime.

Static scans can see that a program divides by a data-derived expression, but
they cannot tell whether the executed path contains zero, NaN, or infinity.
This read-only audit instruments every division expression, executes the exact
submission program with the official typed-CSV contract, and reports only
denominators that are empty or contain non-finite/zero values.

The report is a triage queue, not a correctness oracle.  A vector denominator
may legitimately contain a missing value if the question explicitly excludes
that observation, so every finding still requires source reconciliation.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd

try:  # direct script execution
    from grader_check import _SAFE_BUILTINS, _read_csv
except ModuleNotFoundError:  # imported as ``scripts.audit_runtime_denominators``
    from scripts.grader_check import _SAFE_BUILTINS, _read_csv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class DenominatorInstrumenter(ast.NodeTransformer):
    """Wrap each denominator without evaluating it more than once."""

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
            return node
        expression = ast.unparse(node.right)
        wrapped = ast.Call(
            func=ast.Name(id="_audit_denominator", ctx=ast.Load()),
            args=[
                node.right,
                ast.Constant(value=int(getattr(node, "lineno", -1))),
                ast.Constant(value=expression),
            ],
            keywords=[],
        )
        node.right = ast.copy_location(wrapped, node.right)
        return node


def instrument(code: str) -> Any:
    tree = ast.parse(code)
    tree = DenominatorInstrumenter().visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, "<instrumented_pandas_query>", "exec")


def _numeric_values(value: Any) -> tuple[list[float], str]:
    if isinstance(value, pd.DataFrame):
        raw = value.to_numpy().reshape(-1).tolist()
        kind = "dataframe"
    elif isinstance(value, pd.Series):
        raw = value.tolist()
        kind = "series"
    elif isinstance(value, (list, tuple, set)):
        raw = list(value)
        kind = type(value).__name__
    else:
        raw = [value]
        kind = "scalar"

    numbers: list[float] = []
    for item in raw:
        try:
            numbers.append(float(item))
        except (TypeError, ValueError):
            continue
    return numbers, kind


def denominator_stats(value: Any, *, line: int, expression: str) -> dict[str, Any]:
    numbers, kind = _numeric_values(value)
    finite = [number for number in numbers if math.isfinite(number)]
    zeros = [number for number in finite if number == 0.0]
    nan_count = sum(math.isnan(number) for number in numbers)
    inf_count = sum(math.isinf(number) for number in numbers)
    return {
        "line": line,
        "expression": expression,
        "kind": kind,
        "value_count": len(numbers),
        "zero_count": len(zeros),
        "nan_count": nan_count,
        "infinite_count": inf_count,
        "empty": len(numbers) == 0,
        "minimum_finite": min(finite) if finite else None,
        "maximum_finite": max(finite) if finite else None,
    }


def run_row(candidate: Path, row: dict[str, Any]) -> dict[str, Any]:
    csv_paths = {
        str(item["variable"]): (candidate / str(item["csv_path"])).resolve()
        for item in row.get("evidence") or []
    }
    dfs = {variable: _read_csv(path, typed=True) for variable, path in csv_paths.items()}
    observations: list[dict[str, Any]] = []

    def audit_denominator(value: Any, line: int, expression: str) -> Any:
        observations.append(denominator_stats(value, line=line, expression=expression))
        return value

    namespace: dict[str, Any] = {
        "pd": pd,
        "dfs": dfs,
        "_audit_denominator": audit_denominator,
        "__builtins__": _SAFE_BUILTINS,
    }
    if len(dfs) == 1:
        namespace["df"] = next(iter(dfs.values()))

    code = str(row.get("pandas_query") or "").strip()
    exec(instrument(code), namespace)  # noqa: S102
    risky = [
        item
        for item in observations
        if item["empty"]
        or item["zero_count"]
        or item["nan_count"]
        or item["infinite_count"]
    ]
    return {
        "id": int(row["id"]),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "division_count_executed": len(observations),
        "risky_denominators": risky,
        "relevant_tables": row.get("relevant_tables", []),
    }


def audit(candidate: Path) -> dict[str, Any]:
    candidate = candidate.resolve()
    rows = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    rows_with_division = 0
    divisions_executed = 0

    for row in rows:
        code = str(row.get("pandas_query") or "").strip()
        if not code or "/" not in code:
            continue
        try:
            result = run_row(candidate, row)
        except Exception as error:  # keep a complete audit trail
            errors.append(
                {
                    "id": int(row["id"]),
                    "error": f"{type(error).__name__}: {str(error)[:300]}",
                }
            )
            continue
        if result["division_count_executed"]:
            rows_with_division += 1
            divisions_executed += result["division_count_executed"]
        if result["risky_denominators"]:
            findings.append(result)

    return {
        "kind": "runtime_denominator_audit",
        "candidate": str(candidate),
        "execution_contract": "official typed CSV + restricted grader builtins",
        "checked": len(rows),
        "rows_with_executed_division": rows_with_division,
        "divisions_executed": divisions_executed,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "execution_error_count": len(errors),
        "policy": "Read-only triage; reconcile findings with exact source cells before any answer change.",
        "findings": findings,
        "execution_errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = audit(args.candidate)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered, encoding="utf-8")
        print(args.out.resolve())
    else:
        print(rendered, end="")
    return 0 if not report["execution_errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
