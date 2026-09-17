"""Verify that year-returning answers derive their labels from evidence data."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
from pathlib import Path


YEAR_QUESTION = re.compile(r"năm nào", re.I)


class _PreviousValueInliner(ast.NodeTransformer):
    """Inline an earlier binding into a self-referential reassignment."""

    def __init__(self, name: str, previous: ast.AST) -> None:
        self.name = name
        self.previous = previous

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id == self.name:
            return copy.deepcopy(self.previous)
        return node


def _references_name(value: ast.AST, name: str) -> bool:
    return any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == name
        for node in ast.walk(value)
    )


def assignments(tree: ast.AST) -> dict[str, ast.AST]:
    """Return latest assignments while preserving prior-value dataflow.

    Submission programs commonly finish with ``result = float(result)``.
    A plain last-write mapping turns that into a self-cycle and hides the
    evidence dependency of the preceding assignment.  Inline only the prior
    value of the same target; other names remain symbolic for the normal
    dependency walk.
    """
    found: dict[str, ast.AST] = {}
    assignment_nodes = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assignment_nodes.sort(
        key=lambda node: (
            int(getattr(node, "lineno", 0)),
            int(getattr(node, "col_offset", 0)),
        )
    )
    for node in assignment_nodes:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                value = node.value
                if value is None:
                    continue
                previous = found.get(target.id)
                if previous is not None and _references_name(value, target.id):
                    value = _PreviousValueInliner(target.id, previous).visit(
                        copy.deepcopy(value)
                    )
                    ast.fix_missing_locations(value)
                found[target.id] = value
    return found


def dependency_slice(root: ast.AST, mapping: dict[str, ast.AST]) -> list[ast.AST]:
    queue = [root]
    visited_names: set[str] = set()
    nodes: list[ast.AST] = []
    while queue:
        node = queue.pop()
        nodes.append(node)
        for item in ast.walk(node):
            if not isinstance(item, ast.Name) or item.id in visited_names:
                continue
            visited_names.add(item.id)
            value = mapping.get(item.id)
            if value is not None:
                queue.append(value)
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    args = parser.parse_args()
    rows = json.loads(
        (args.submission_dir / "submission.json").read_text(encoding="utf-8")
    )
    issues = []
    checked = 0
    for row in rows:
        if not (
            YEAR_QUESTION.search(str(row.get("question", "")))
            and isinstance(row.get("answer"), (int, float))
            and 2000 <= float(row["answer"]) <= 2030
        ):
            continue
        checked += 1
        tree = ast.parse(row["pandas_query"])
        mapping = assignments(tree)
        result = mapping.get("result")
        if result is None:
            issues.append({"id": row["id"], "kind": "missing-result"})
            continue
        sliced = dependency_slice(result, mapping)
        year_literals = sorted(
            {
                int(item.value)
                for node in sliced
                for item in ast.walk(node)
                if isinstance(item, ast.Constant)
                and isinstance(item.value, int)
                and 2000 <= item.value <= 2030
            }
        )
        names = {
            item.id
            for node in sliced
            for item in ast.walk(node)
            if isinstance(item, ast.Name)
        }
        if year_literals:
            issues.append(
                {
                    "id": row["id"],
                    "kind": "literal-year-in-result-dataflow",
                    "years": year_literals,
                }
            )
        if not ({"df", "dfs"} & names):
            issues.append(
                {"id": row["id"], "kind": "year-result-without-data-dependency"}
            )
    report = {
        "entries": len(rows),
        "year_output_queries": checked,
        "issues": len(issues),
        "by_kind": {
            kind: sum(issue["kind"] == kind for issue in issues)
            for kind in sorted({issue["kind"] for issue in issues})
        },
        "samples": issues[:30],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
