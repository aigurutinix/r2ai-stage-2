"""Audit explicit ``None`` operands in source-backed panel programs.

The panel builders intentionally use ``None`` when a source cell has not yet
been reconciled. That is safe only when the operand cannot change filtering,
ranking, or the final result. This audit makes those omissions visible and
prioritises fields that are referenced after ``df = pd.DataFrame(_rows)``.

This is a review queue, not a correctness oracle. Every proposed repair must
still be traced to an original BTC table cell before changing a submission.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _dict_value(node: ast.Dict, key: str) -> ast.AST | None:
    for key_node, value_node in zip(node.keys, node.values):
        if isinstance(key_node, ast.Constant) and key_node.value == key:
            return value_node
    return None


def _constant(node: ast.AST | None):
    return node.value if isinstance(node, ast.Constant) else None


def _rows_assignment(tree: ast.AST) -> ast.List | None:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "_rows":
            return node.value if isinstance(node.value, ast.List) else None
    return None


def _downstream(code: str) -> str:
    marker = "df = pd.DataFrame(_rows)"
    return code.split(marker, 1)[1] if marker in code else code


def _string_slice(node: ast.Subscript) -> str | None:
    value = node.slice
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return value.value
    return None


def _dependencies(node: ast.AST, known_fields: set[str]) -> tuple[set[str], set[str]]:
    """Return variable and dataframe-column dependencies in *node*.

    This deliberately ignores method names such as ``mean`` and ``idxmax``.
    Attribute access is treated as a column only when the attribute matches a
    source or derived field already known to the program.
    """

    names: set[str] = set()
    columns: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            names.add(child.id)
        elif isinstance(child, ast.Subscript):
            column = _string_slice(child)
            if column in known_fields:
                columns.add(column)
        elif isinstance(child, ast.Attribute) and child.attr in known_fields:
            columns.add(child.attr)
        elif (
            isinstance(child, ast.Constant)
            and isinstance(child.value, str)
            and child.value in known_fields
        ):
            # Pandas APIs also carry column dependencies as string-valued
            # keywords, for example ``pivot(values='revenue')`` and
            # ``sort_values(by='score')``.
            columns.add(child.value)
        elif isinstance(child, ast.keyword) and child.arg == "subset":
            for value in ast.walk(child.value):
                if (
                    isinstance(value, ast.Constant)
                    and isinstance(value.value, str)
                    and value.value in known_fields
                ):
                    columns.add(value.value)
    return names, columns


def _target_column(node: ast.AST) -> str | None:
    if isinstance(node, ast.Subscript):
        return _string_slice(node)
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _live_source_fields(tree: ast.Module, source_fields: set[str]) -> set[str]:
    """Conservatively slice dataframe fields backwards from ``result``.

    Panel programs often materialise a broad metric frame and compute many
    convenience columns that the final question never uses. A lexical search
    therefore over-prioritises unrelated ``None`` values. This small dataflow
    graph keeps only fields that can reach the final ``result`` expression via
    scalar variables, filters, selectors, or derived dataframe columns.
    """

    known_fields = set(source_fields)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            column = _target_column(target)
            if column:
                known_fields.add(column)

    variable_deps: dict[str, tuple[set[str], set[str]]] = {}
    column_deps: dict[str, tuple[set[str], set[str]]] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value = statement.value
        if value is None:
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        deps = _dependencies(value, known_fields)
        for target in targets:
            if isinstance(target, ast.Name):
                if target.id in {"_rows", "df", "_src"}:
                    # Frame construction itself depends on every row literal,
                    # but liveness is determined by later column access. If we
                    # propagate ``_rows`` here, every source field becomes live.
                    variable_deps[target.id] = (set(), set())
                    continue
                names, columns = deps
                # Preserve the previous version for wrappers and in-place
                # narrowing such as ``result = round(float(result), 2)`` and
                # ``filtered = filtered[mask]``.
                if target.id in names and target.id in variable_deps:
                    previous_names, previous_columns = variable_deps[target.id]
                    names = (names - {target.id}) | previous_names
                    columns = columns | previous_columns
                variable_deps[target.id] = (set(names), set(columns))
            else:
                column = _target_column(target)
                if column:
                    column_deps[column] = deps

    live_fields: set[str] = set()
    seen_names: set[str] = set()
    seen_columns: set[str] = set()

    def visit_name(name: str) -> None:
        if name in seen_names:
            return
        seen_names.add(name)
        names, columns = variable_deps.get(name, (set(), set()))
        for dependency in names:
            visit_name(dependency)
        for dependency in columns:
            visit_column(dependency)

    def visit_column(column: str) -> None:
        if column in seen_columns:
            return
        seen_columns.add(column)
        if column in column_deps:
            names, columns = column_deps[column]
            for dependency in names:
                visit_name(dependency)
            for dependency in columns:
                visit_column(dependency)
        elif column in source_fields:
            live_fields.add(column)

    visit_name("result")
    return live_fields


def audit(submission_dir: Path) -> dict:
    rows = json.loads(
        (submission_dir / "submission.json").read_text(encoding="utf-8")
    )
    findings: list[dict] = []
    for row in rows:
        code = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        panel = _rows_assignment(tree)
        if panel is None:
            continue
        source_fields = {
            str(_constant(key_node))
            for item in panel.elts
            if isinstance(item, ast.Dict)
            for key_node in item.keys
            if _constant(key_node) not in {None, "ticker", "year"}
        }
        live_source_fields = _live_source_fields(tree, source_fields)
        missing: list[dict] = []
        for item in panel.elts:
            if not isinstance(item, ast.Dict):
                continue
            ticker = _constant(_dict_value(item, "ticker"))
            year = _constant(_dict_value(item, "year"))
            for key_node, value_node in zip(item.keys, item.values):
                key = _constant(key_node)
                if key in {"ticker", "year"}:
                    continue
                if not (isinstance(value_node, ast.Constant) and value_node.value is None):
                    continue
                field = str(key)
                referenced = field in live_source_fields
                missing.append(
                    {
                        "ticker": ticker,
                        "year": year,
                        "field": field,
                        "referenced_downstream": referenced,
                        "field_reaches_result": referenced,
                    }
                )
        if missing:
            findings.append(
                {
                    "id": int(row["id"]),
                    "question": row.get("question", ""),
                    "answer": row.get("answer"),
                    "missing_operands": missing,
                    "high_priority": any(
                        item["field_reaches_result"] for item in missing
                    ),
                    "live_source_fields": sorted(live_source_fields),
                }
            )

    return {
        "submission": str(submission_dir.resolve()),
        "finding_count": len(findings),
        "high_priority_count": sum(item["high_priority"] for item in findings),
        "high_priority_question_ids": [
            item["id"] for item in findings if item["high_priority"]
        ],
        "findings": findings,
        "claim_limit": (
            "Static backward slicing is field-aware but not row/value-aware. "
            "An explicit None remains a review signal; add an operand only after "
            "verifying its BTC source cell and recomputing the full query."
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
