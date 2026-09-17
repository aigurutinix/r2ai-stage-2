"""Locate value-dependent attribute calls that can fail in the grader.

Financial CSV cells may be delivered as strings, Python scalars or NumPy
scalars depending on dtype inference.  Calling ``cell.replace(...)`` or
``cell.strip()`` directly is therefore unsafe even when one local replay
happens to infer a string.  Likewise, Pandas reductions can return either a
NumPy scalar (which exposes ``.round``) or a Python scalar (which does not), so
``series.mean().round(2)`` is less portable than ``round(float(...), 2)``.
This audit is read-only and reports the exact receiver expression and source
line; it deliberately ignores attribute calls inside functions that first
normalize their argument with ``str(...)``.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


RISKY_METHODS = {
    "replace",
    "strip",
    "split",
    "startswith",
    "endswith",
    "rfind",
}

SCALAR_REDUCTIONS = {
    "mean",
    "median",
    "sum",
    "max",
    "min",
}


def _function_ancestors(tree: ast.AST) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[child] = parent
    return parents


def _inside_normalizing_helper(node: ast.AST, parents: dict[ast.AST, ast.AST]) -> bool:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # The standard compact-source parsers normalize x through str(x)
            # before any string methods.  Their internal ``s.replace`` calls
            # are not value-dependent hazards.
            for item in ast.walk(current):
                if (
                    isinstance(item, ast.Call)
                    and isinstance(item.func, ast.Name)
                    and item.func.id == "str"
                ):
                    return True
            return False
    return False


def _safe_receiver(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "str"
    ):
        return True
    # A chained call such as str(x).replace(...).replace(...) remains safe at
    # every outer link even though the immediate receiver is another Call.
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in RISKY_METHODS
    ):
        return _safe_receiver(node.func.value)
    # Pandas' vectorized string accessor performs its own dtype contract.
    if isinstance(node, ast.Attribute) and node.attr == "str":
        return True
    return False


def _scalar_round_hazard(node: ast.Call) -> bool:
    """Return whether ``node`` is ``reduction().round(...)``.

    DataFrame/Series ``.round`` is stable, but a reduction result is a scalar
    whose concrete Python/NumPy type can vary with Pandas and dtype inference.
    """
    if not (
        isinstance(node.func, ast.Attribute)
        and node.func.attr == "round"
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Attribute)
    ):
        return False
    return node.func.value.func.attr in SCALAR_REDUCTIONS


def audit(submission: Path) -> dict:
    rows = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict] = []
    for row in rows:
        code = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        parents = _function_ancestors(tree)
        hazards: list[dict] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _scalar_round_hazard(node):
                hazards.append(
                    {
                        "method": "round",
                        "receiver": ast.unparse(node.func.value),
                        "line": int(getattr(node, "lineno", 0)),
                        "source": ast.get_source_segment(code, node),
                        "reason": "scalar_reduction_round_portability",
                    }
                )
                continue
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in RISKY_METHODS
            ):
                continue
            receiver = node.func.value
            if _safe_receiver(receiver) or _inside_normalizing_helper(node, parents):
                continue
            hazards.append(
                {
                    "method": node.func.attr,
                    "receiver": ast.unparse(receiver),
                    "line": int(getattr(node, "lineno", 0)),
                    "source": ast.get_source_segment(code, node),
                }
            )
        if hazards:
            findings.append(
                {
                    "id": int(row["id"]),
                    "question": row.get("question", ""),
                    "answer": row.get("answer"),
                    "hazards": hazards,
                }
            )
    return {
        "submission": str(submission.resolve()),
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
        "claim_limit": (
            "A dynamic string-method receiver is a portability signal, not "
            "proof of failure. Harden only after value-preserving replay."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()
    report = audit(args.submission)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    if args.fail_on_findings and report["finding_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
