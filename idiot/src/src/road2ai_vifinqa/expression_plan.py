"""Compile a short acyclic calculation plan into one checked pandas expression."""
from __future__ import annotations

import ast
import copy
import re

from .submission import _validate_expression


_RESERVED = {"pd", "np", "float", "int", "abs", "round", "min", "max", "sum", "len"}
_ATTRIBUTES = {
    "DataFrame", "Series", "Index", "concat", "merge", "where", "nan",
    "loc", "iloc", "at", "iat", "columns", "index", "values", "shape", "size", "empty",
    "str", "contains", "startswith", "endswith", "split", "extract", "lower", "strip",
    "sum", "mean", "median", "min", "max", "idxmin", "idxmax", "count", "quantile",
    "nlargest", "nsmallest", "groupby", "pivot", "pivot_table", "unstack", "stack",
    "reset_index", "set_index", "sort_values", "sort_index", "assign", "drop", "dropna",
    "drop_duplicates", "rename", "reindex", "fillna", "astype", "isin", "notna", "isna",
    "abs", "round", "clip", "rank", "head", "tail", "nunique", "unique", "to_numpy",
}


def inline_plan(steps: list[dict[str, str]], expression: str, *, frames: set[str], columns: set[str]) -> str:
    """No execution, imports, IO, lambdas, rebinding or forward references."""
    if len(steps) > 20:
        raise ValueError("too many calculation steps")
    bindings: dict[str, ast.AST] = {}

    class Substitute(ast.NodeTransformer):
        def visit_Name(self, node):
            return copy.deepcopy(bindings[node.id]) if node.id in bindings else node

    def parse(source: str) -> ast.AST:
        tree = _validate_expression(source, frames | set(bindings))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr not in _ATTRIBUTES | columns:
                raise ValueError(f"unsupported attribute: {node.attr}")
            if isinstance(node, ast.keyword) and node.arg == "inplace":
                raise ValueError("in-place mutation is not allowed")
        body = Substitute().visit(tree.body)
        if sum(1 for _ in ast.walk(body)) > 50_000:
            raise ValueError("expanded expression too large")
        return body

    for step in steps:
        name = str(step["name"])
        if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", name) or name in frames | _RESERVED | set(bindings):
            raise ValueError("invalid or repeated calculation name")
        bindings[name] = parse(str(step["expression"]))
    output = ast.unparse(ast.fix_missing_locations(parse(expression)))
    if len(output) > 200_000:
        raise ValueError("expanded expression too long")
    _validate_expression(output, frames)
    return output
