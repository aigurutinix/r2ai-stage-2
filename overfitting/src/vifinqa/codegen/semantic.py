"""Static semantic guards for generated pandas programs.

Execution success alone is not enough: ``answer = 123`` is a valid numeric
program but is not grounded in the submitted evidence.  These checks are kept
deterministic and lightweight so they can run for every Kaggle generation.
"""
from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field

from ..config import YEAR_MAX, YEAR_MIN


_DF_NAME = re.compile(r"^df\d+$")


@dataclass
class SemanticCheck:
    ok: bool
    dataframe_refs: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "dataframe_refs": self.dataframe_refs,
            "errors": self.errors,
            "warnings": self.warnings,
        }


def answer_dataframe_refs(code: str) -> set[str]:
    """Return dataframes that influence the final expression/``answer``.

    For scripts, simple local-variable dependencies are followed.  This catches
    constants disguised by dead references, for example ``x = df1; answer = 0``.
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        tree = ast.parse(code, mode="exec")
    if isinstance(tree, ast.Expression):
        return _refs_from_node(tree.body, {})

    assignments: dict[str, ast.AST] = {}
    answer_node: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            value = node.value
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    assignments[target.id] = value
                    if target.id == "answer":
                        answer_node = value
    return _refs_from_node(answer_node, assignments) if answer_node is not None else set()


def all_dataframe_refs(code: str) -> set[str]:
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError:
        tree = ast.parse(code, mode="eval")
    return {node.id for node in ast.walk(tree)
            if isinstance(node, ast.Name) and _DF_NAME.fullmatch(node.id)}


def validate_generated_answer(code: str, available_vars, answer: float,
                              route: dict | None = None) -> SemanticCheck:
    errors, warnings = [], []
    available = set(available_vars)
    try:
        refs = answer_dataframe_refs(code)
        all_refs = all_dataframe_refs(code)
    except SyntaxError as exc:
        return SemanticCheck(False, errors=[f"invalid syntax: {exc.msg}"])

    if not refs:
        errors.append("answer is not derived from any dataframe")
    unknown = refs - available
    if unknown:
        errors.append(f"answer references unavailable dataframes: {sorted(unknown)}")
    dead_refs = all_refs - refs
    if dead_refs:
        warnings.append(f"dataframe references do not influence answer: {sorted(dead_refs)}")
    if not math.isfinite(float(answer)):
        errors.append("answer is NaN or infinite")

    route = route or {}
    output_type = route.get("output_type") or route.get("target_type")
    if output_type == "year":
        if abs(float(answer) - round(float(answer))) > 1e-6:
            errors.append("year answer is not an integer")
        elif not YEAR_MIN - 1 <= int(round(float(answer))) <= YEAR_MAX + 1:
            errors.append("year answer is outside the supported financial-report range")
    elif output_type == "count":
        if float(answer) < 0 or abs(float(answer) - round(float(answer))) > 1e-6:
            errors.append("count answer must be a non-negative integer")
    elif output_type in {"percent", "percentage_point"} and abs(float(answer)) > 10000:
        warnings.append("percentage magnitude exceeds 10,000; verify unit conversion")
    elif output_type == "ratio" and abs(float(answer)) > 1000:
        warnings.append("ratio magnitude exceeds 1,000; verify unit conversion")

    return SemanticCheck(not errors, sorted(refs), errors, warnings)


def _refs_from_node(node: ast.AST | None, assignments: dict[str, ast.AST],
                    seen: set[str] | None = None) -> set[str]:
    if node is None:
        return set()
    seen = set() if seen is None else seen
    refs: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Name):
            continue
        name = child.id
        if _DF_NAME.fullmatch(name):
            refs.add(name)
        elif name in assignments and name not in seen:
            seen.add(name)
            refs.update(_refs_from_node(assignments[name], assignments, seen))
    return refs
