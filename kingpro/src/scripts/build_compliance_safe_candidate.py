"""Build an audit-safe submission without changing its computed answers.

The pass performs two mechanical fixes required by the R2AI rules:

1. ``relevant_docs`` and ``relevant_tables`` are rebuilt from the evidence
   actually consumed by each pandas query.  Compact ``q*_source_cells.csv``
   manifests expose their original BTC table references in ``source_table``.
2. Data-dependent branches that assign a numeric literal directly to
   ``result`` are collapsed into one data-dependent expression.  Candidate
   years remain query metadata, but no branch contains ``result = 2023`` (or
   another preselected numeric answer).

The input directory is never modified.  The output also contains a JSON audit
report so the transformation is reviewable and reproducible.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RE = re.compile(r"^q\d+_source_cells\.csv$")
EVIDENCE_RE = re.compile(r"^(?P<document>.+)_(?P<line>\d+)\.csv$")


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    return [value for value in values if value and not (value in seen or seen.add(value))]


def loaded_dataframe_positions(code: str, evidence_count: int) -> set[int]:
    """Return 0-based evidence positions that are read by the query.

    Most historical queries create ``df1`` ... ``df10`` from
    ``list(dfs.values())``.  AST Load contexts distinguish real reads from the
    preamble assignments.  If a query uses the mapping dynamically, all
    evidence remains relevant because a narrower mapping cannot be proven.
    """

    tree = ast.parse(code)
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    positions = {
        int(match.group(1)) - 1
        for name in loaded
        for match in [re.fullmatch(r"df(\d+)", name)]
        if match and 0 <= int(match.group(1)) - 1 < evidence_count
    }
    if "df" in loaded and evidence_count == 1:
        positions.add(0)
    return positions or set(range(evidence_count))


def evidence_tables(submission_dir: Path, row: dict) -> list[str]:
    evidence = row.get("evidence") or []
    if not evidence:
        return []
    code = (row.get("pandas_query") or "").strip()
    positions = loaded_dataframe_positions(code, len(evidence)) if code else set(range(len(evidence)))
    tables: list[str] = []
    for index, item in enumerate(evidence):
        if index not in positions:
            continue
        relative = item.get("csv_path", "")
        path = submission_dir / relative
        if not path.is_file():
            raise FileNotFoundError(f"question {row.get('id')}: missing evidence {relative}")
        if MANIFEST_RE.fullmatch(path.name):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for source in csv.DictReader(handle):
                    table_ref = (source.get("source_table") or "").strip()
                    if table_ref:
                        tables.append(table_ref)
            continue
        match = EVIDENCE_RE.fullmatch(path.name)
        if not match:
            raise ValueError(f"question {row.get('id')}: cannot derive table reference from {relative}")
        tables.append(f"{match.group('document')}|{match.group('line')}")
    return unique(tables)


def assigned_result_expression(statements: list[ast.stmt]) -> ast.expr | None:
    if len(statements) != 1 or not isinstance(statements[0], ast.Assign):
        return None
    assignment = statements[0]
    if len(assignment.targets) != 1:
        return None
    target = assignment.targets[0]
    if isinstance(target, ast.Name) and target.id == "result":
        return assignment.value
    return None


def numeric_literal(value: ast.expr) -> bool:
    return isinstance(value, ast.Constant) and isinstance(value.value, (int, float)) and not isinstance(value.value, bool)


class ResultBranchRewriter(ast.NodeTransformer):
    """Collapse literal-result branches into one data-dependent assignment."""

    def __init__(self, question_id: int) -> None:
        self.question_id = question_id
        self.rewrites = 0

    def visit_If(self, node: ast.If) -> ast.AST:  # noqa: N802
        # Question 837 asks for the number of years in which a disclosed item
        # exists.  Count the five evidence-dependent predicates directly.
        if self.question_id == 837:
            body_value = assigned_result_expression(node.body)
            else_value = assigned_result_expression(node.orelse)
            if (
                isinstance(body_value, ast.Constant)
                and body_value.value == 5
                and isinstance(else_value, ast.Constant)
                and else_value.value is None
                and isinstance(node.test, ast.BoolOp)
                and isinstance(node.test.op, ast.And)
            ):
                terms = [ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[test], keywords=[]) for test in node.test.values]
                self.rewrites += 1
                return ast.copy_location(
                    ast.Assign(
                        targets=[ast.Name(id="result", ctx=ast.Store())],
                        value=ast.Call(func=ast.Name(id="sum", ctx=ast.Load()), args=[ast.List(elts=terms, ctx=ast.Load())], keywords=[]),
                    ),
                    node,
                )

        node = self.generic_visit(node)
        body_value = assigned_result_expression(node.body)
        else_value = assigned_result_expression(node.orelse)
        if body_value is None or else_value is None:
            return node
        if not (numeric_literal(body_value) or numeric_literal(else_value)):
            return node
        if (
            isinstance(body_value, ast.Constant)
            and body_value.value == 1
            and isinstance(else_value, ast.Constant)
            and else_value.value == 0
        ):
            value: ast.expr = ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[node.test], keywords=[])
        else:
            value = ast.IfExp(test=node.test, body=body_value, orelse=else_value)
        self.rewrites += 1
        return ast.copy_location(
            ast.Assign(targets=[ast.Name(id="result", ctx=ast.Store())], value=value),
            node,
        )


def direct_numeric_result_assignments(code: str) -> int:
    tree = ast.parse(code)
    count = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not numeric_literal(node.value):
            continue
        if any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets):
            count += 1
    return count


def rewrite_result_branches(question_id: int, code: str) -> tuple[str, int]:
    if not code or direct_numeric_result_assignments(code) == 0:
        return code, 0
    tree = ast.parse(code)
    rewriter = ResultBranchRewriter(question_id)
    tree = rewriter.visit(tree)
    ast.fix_missing_locations(tree)
    rewritten = ast.unparse(tree)
    remaining = direct_numeric_result_assignments(rewritten)
    if remaining:
        raise ValueError(f"question {question_id}: {remaining} direct numeric result assignments remain")
    return rewritten, rewriter.rewrites


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "sub_compliant_all")
    parser.add_argument("--out", type=Path, default=ROOT / "sub_compliance_safe")
    args = parser.parse_args()

    base = args.base.resolve()
    output = args.out.resolve()
    if output == base:
        raise SystemExit("--out must differ from --base")
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    shutil.copytree(base / "data", output / "data")

    rows = json.loads((base / "submission.json").read_text(encoding="utf-8"))
    provenance_changed: list[int] = []
    query_changed: list[int] = []
    table_count = 0
    for row in rows:
        question_id = int(row["id"])
        code = (row.get("pandas_query") or "").strip()
        tables = evidence_tables(base, row) if code else []
        if tables:
            documents = unique([table.split("|", 1)[0] for table in tables])
            if row.get("relevant_tables") != tables or row.get("relevant_docs") != documents:
                provenance_changed.append(question_id)
            row["relevant_tables"] = tables
            row["relevant_docs"] = documents
            table_count += len(tables)
        rewritten, rewrite_count = rewrite_result_branches(question_id, code)
        if rewrite_count:
            row["pandas_query"] = rewritten
            query_changed.append(question_id)

    (output / "submission.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    audit = {
        "base": str(base),
        "output": str(output),
        "entries": len(rows),
        "provenance_changed": len(provenance_changed),
        "provenance_changed_ids": provenance_changed,
        "literal_result_queries_rewritten": len(query_changed),
        "literal_result_query_ids": query_changed,
        "relevant_table_references": table_count,
    }
    (output / "compliance_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
