"""Find executed max/min selections whose candidate values are tied.

Pandas ``idxmax``/``idxmin`` and Python ``max``/``min`` deterministically pick
one item, which can conceal an under-specified or incorrectly implemented
tie-break.  This read-only audit instruments extrema calls outside parser
helper functions, executes the official typed-CSV program, and emits the tied
candidate labels for source review.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from grader_check import _SAFE_BUILTINS, _read_csv
except ModuleNotFoundError:
    from scripts.grader_check import _SAFE_BUILTINS, _read_csv


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


EXTREMA_QUESTION_TOKENS = (
    "cao nhất",
    "lớn nhất",
    "nhiều nhất",
    "thấp nhất",
    "nhỏ nhất",
    "tối đa",
    "tối thiểu",
)


class ExtremaInstrumenter(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        # Generated parser helpers contain incidental min/max operations that
        # do not choose a financial period/entity.
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        node = self.generic_visit(node)
        line = int(getattr(node, "lineno", -1))
        expression = ast.unparse(node)
        if isinstance(node.func, ast.Attribute) and node.func.attr in {
            "idxmax",
            "idxmin",
            "max",
            "min",
        }:
            wrapped = ast.Call(
                func=ast.Name(id="_audit_method_extreme", ctx=ast.Load()),
                args=[
                    node.func.value,
                    ast.Constant(value=node.func.attr),
                    ast.Constant(value=line),
                    ast.Constant(value=expression),
                    *node.args,
                ],
                keywords=node.keywords,
            )
            return ast.copy_location(wrapped, node)
        if isinstance(node.func, ast.Name) and node.func.id in {"max", "min"}:
            wrapped = ast.Call(
                func=ast.Name(id="_audit_builtin_extreme", ctx=ast.Load()),
                args=[
                    ast.Constant(value=node.func.id),
                    ast.Constant(value=line),
                    ast.Constant(value=expression),
                    *node.args,
                ],
                keywords=node.keywords,
            )
            return ast.copy_location(wrapped, node)
        return node


def instrument(code: str) -> Any:
    tree = ExtremaInstrumenter().visit(ast.parse(code))
    ast.fix_missing_locations(tree)
    return compile(tree, "<instrumented_extrema_query>", "exec")


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _series_tie(series: pd.Series, method: str, result: Any) -> dict[str, Any] | None:
    clean = series.dropna()
    if clean.empty:
        return None
    is_max = method in {"max", "idxmax"}
    extreme = clean.max() if is_max else clean.min()
    try:
        mask = clean == extreme
        tied = clean[mask]
    except Exception:
        return None
    if len(tied) <= 1:
        return None
    return {
        "candidate_count": int(len(clean)),
        "tie_count": int(len(tied)),
        "extreme_value": _plain(extreme),
        "tied_labels": [_plain(label) for label in tied.index.tolist()[:30]],
        "returned": _plain(result),
    }


def _iterable_tie(values: tuple[Any, ...], method: str, result: Any) -> dict[str, Any] | None:
    if len(values) == 1 and not isinstance(values[0], (str, bytes)):
        try:
            candidates = list(values[0])
        except TypeError:
            candidates = list(values)
    else:
        candidates = list(values)
    if len(candidates) <= 1:
        return None
    try:
        tie_count = sum(value == result for value in candidates)
    except Exception:
        return None
    if tie_count <= 1:
        return None
    return {
        "candidate_count": len(candidates),
        "tie_count": int(tie_count),
        "extreme_value": _plain(result),
        "tied_labels": [index for index, value in enumerate(candidates) if value == result][:30],
        "returned": _plain(result),
    }


def run_row(candidate: Path, row: dict[str, Any]) -> dict[str, Any]:
    csv_paths = {
        str(item["variable"]): (candidate / str(item["csv_path"])).resolve()
        for item in row.get("evidence") or []
    }
    dfs = {variable: _read_csv(path, typed=True) for variable, path in csv_paths.items()}
    observations: list[dict[str, Any]] = []
    ties: list[dict[str, Any]] = []

    def method_extreme(obj: Any, method: str, line: int, expression: str, *args: Any, **kwargs: Any) -> Any:
        result = getattr(obj, method)(*args, **kwargs)
        observations.append({"line": line, "method": method, "expression": expression})
        tie = _series_tie(obj, method, result) if isinstance(obj, pd.Series) else None
        if tie:
            ties.append({"line": line, "method": method, "expression": expression, **tie})
        return result

    def builtin_extreme(method: str, line: int, expression: str, *values: Any, **kwargs: Any) -> Any:
        function = max if method == "max" else min
        result = function(*values, **kwargs)
        observations.append({"line": line, "method": method, "expression": expression})
        tie = _iterable_tie(values, method, result)
        if tie:
            ties.append({"line": line, "method": method, "expression": expression, **tie})
        return result

    namespace: dict[str, Any] = {
        "pd": pd,
        "dfs": dfs,
        "_audit_method_extreme": method_extreme,
        "_audit_builtin_extreme": builtin_extreme,
        "__builtins__": _SAFE_BUILTINS,
    }
    if len(dfs) == 1:
        namespace["df"] = next(iter(dfs.values()))
    code = str(row.get("pandas_query") or "").strip()
    exec(instrument(code), namespace)  # noqa: S102
    return {
        "id": int(row["id"]),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "extrema_calls_executed": len(observations),
        "ties": ties,
        "relevant_tables": row.get("relevant_tables", []),
    }


def audit(candidate: Path) -> dict[str, Any]:
    candidate = candidate.resolve()
    rows = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    semantic_rows = [
        row
        for row in rows
        if any(token in str(row.get("question") or "").casefold() for token in EXTREMA_QUESTION_TOKENS)
    ]
    findings: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    calls = 0
    for row in semantic_rows:
        try:
            result = run_row(candidate, row)
        except Exception as error:
            errors.append({"id": int(row["id"]), "error": f"{type(error).__name__}: {str(error)[:300]}"})
            continue
        calls += result["extrema_calls_executed"]
        if result["ties"]:
            findings.append(result)
    return {
        "kind": "runtime_extrema_tie_audit",
        "candidate": str(candidate),
        "execution_contract": "official typed CSV + restricted grader builtins",
        "checked": len(rows),
        "extrema_language_questions": len(semantic_rows),
        "extrema_calls_executed": calls,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "execution_error_count": len(errors),
        "policy": "A tie is a review trigger, not proof of an incorrect answer.",
        "findings": findings,
        "execution_errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
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
