"""Detect answer changes caused by rounding intermediate financial values.

The expected convention in the Stage-2 questions is to perform arithmetic at
source precision and round the final answer to two decimals.  This audit makes
a counterfactual program that removes every ``round`` except the outermost
round assigned directly to ``result``.  Both programs run with the official
typed-CSV contract; a finding is emitted only when the final two-decimal value
changes.

The transformation is diagnostic and never mutates a candidate directory.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
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


def _is_result_target(target: ast.AST) -> bool:
    return isinstance(target, ast.Name) and target.id == "result"


def _final_round_nodes(value: ast.AST | None) -> list[ast.Call]:
    """Return final-output round calls, including conditional assignments."""
    if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
        if value.func.id == "round":
            return [value]
        if value.func.id in {"float", "int"} and len(value.args) == 1:
            return _final_round_nodes(value.args[0])
    if isinstance(value, ast.IfExp):
        return _final_round_nodes(value.body) + _final_round_nodes(value.orelse)
    return []


class DeferredRoundingTransformer(ast.NodeTransformer):
    def __init__(self, protected: set[int]) -> None:
        self.protected = protected
        self.stripped: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:  # noqa: N802
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:  # noqa: N802
        return node

    def visit_If(self, node: ast.If) -> ast.AST:  # noqa: N802
        # Several hardened direct-answer programs compare ``round(result, 2)``
        # with an independent recall value and raise on disagreement.  Those
        # are validation guards, not financial intermediate rounding.
        return node

    def visit_Call(self, node: ast.Call) -> ast.AST:  # noqa: N802
        original_id = id(node)
        node = self.generic_visit(node)
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "round"
            and node.args
        ):
            return node
        if original_id in self.protected:
            return node
        self.stripped.append(
            {
                "line": int(getattr(node, "lineno", -1)),
                "expression": ast.unparse(node),
            }
        )
        return ast.copy_location(node.args[0], node)


def defer_intermediate_rounding(code: str) -> tuple[Any, list[dict[str, Any]]]:
    tree = ast.parse(code)
    protected: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(_is_result_target(target) for target in node.targets):
            protected.update(id(call) for call in _final_round_nodes(node.value))
        elif isinstance(node, ast.AnnAssign) and _is_result_target(node.target):
            protected.update(id(call) for call in _final_round_nodes(node.value))
    transformer = DeferredRoundingTransformer(protected)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return compile(tree, "<deferred_rounding_query>", "exec"), transformer.stripped


def _run(candidate: Path, row: dict[str, Any], compiled: Any | None = None) -> Any:
    csv_paths = {
        str(item["variable"]): (candidate / str(item["csv_path"])).resolve()
        for item in row.get("evidence") or []
    }
    dfs = {variable: _read_csv(path, typed=True) for variable, path in csv_paths.items()}
    namespace: dict[str, Any] = {"pd": pd, "dfs": dfs, "__builtins__": _SAFE_BUILTINS}
    if len(dfs) == 1:
        namespace["df"] = next(iter(dfs.values()))
    executable = compiled or compile(
        str(row.get("pandas_query") or ""),
        "<original_pandas_query>",
        "exec",
    )
    exec(executable, namespace)  # noqa: S102
    result = namespace["result"]
    if hasattr(result, "item"):
        try:
            result = result.item()
        except Exception:
            pass
    return result


def audit(candidate: Path) -> dict[str, Any]:
    candidate = candidate.resolve()
    rows = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict[str, Any]] = []
    sensitivities: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    transformed_rows = 0
    stripped_calls = 0

    for row in rows:
        code = str(row.get("pandas_query") or "").strip()
        if "round(" not in code.replace(" ", ""):
            continue
        try:
            compiled, stripped = defer_intermediate_rounding(code)
            if not stripped:
                continue
            transformed_rows += 1
            stripped_calls += len(stripped)
            original = float(_run(candidate, row))
            deferred = float(_run(candidate, row, compiled))
            if not (math.isfinite(original) and math.isfinite(deferred)):
                continue
            delta = deferred - original
            if not math.isclose(delta, 0.0, rel_tol=0.0, abs_tol=1e-12):
                item = {
                    "id": int(row["id"]),
                    "question": row.get("question"),
                    "answer": row.get("answer"),
                    "original_result": original,
                    "deferred_result": deferred,
                    "delta": delta,
                    "original_round2": round(original, 2),
                    "deferred_round2": round(deferred, 2),
                    "stripped_rounds": stripped,
                    "relevant_tables": row.get("relevant_tables", []),
                }
                sensitivities.append(item)
                if not math.isclose(
                    item["original_round2"],
                    item["deferred_round2"],
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    findings.append(item)
        except Exception as error:
            errors.append({"id": int(row["id"]), "error": f"{type(error).__name__}: {str(error)[:300]}"})

    return {
        "kind": "rounding_order_sensitivity_audit",
        "candidate": str(candidate),
        "execution_contract": "official typed CSV + restricted grader builtins",
        "checked": len(rows),
        "transformed_rows": transformed_rows,
        "intermediate_round_calls_removed": stripped_calls,
        "raw_sensitivity_count": len(sensitivities),
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "execution_error_count": len(errors),
        "policy": "Only a final-two-decimal change is a review finding; source proof is still required.",
        "findings": findings,
        "raw_sensitivities": sensitivities,
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
