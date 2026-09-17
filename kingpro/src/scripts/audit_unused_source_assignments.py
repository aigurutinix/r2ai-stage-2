"""Find source-backed scalar assignments absent from the final result graph.

Financial QA programs often fetch a correct table cell into ``v7`` and then
accidentally use ``v6`` twice, or leave ``v7`` out of a total.  Existing source
audits prove that the cell exists; this read-only audit asks the complementary
question: does the fetched scalar have any dependency path to ``result``?

The analysis deliberately over-approximates branches and reassignments.  A
finding is only a review signal: diagnostics and source cross-checks may be
intentionally unused, so no answer may be changed without source verification
and a reproducible recomputation.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _source_calls(node: ast.AST) -> list[dict]:
    calls: list[dict] = []
    for child in ast.walk(node):
        if not (
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_source_value"
        ):
            continue
        values = [
            argument.value if isinstance(argument, ast.Constant) else None
            for argument in child.args[:3]
        ]
        calls.append(
            {
                "ticker": values[0] if len(values) > 0 else None,
                "year": values[1] if len(values) > 1 else None,
                "metric_key": values[2] if len(values) > 2 else None,
                "expression": ast.unparse(child),
            }
        )
    return calls


def _direct_scalar_source_calls(node: ast.AST) -> list[dict]:
    """Return calls only for a scalar fetch, never a panel/container build."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_source_value"
    ):
        return _source_calls(node)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"abs", "float", "int"}
        and len(node.args) == 1
    ):
        return _direct_scalar_source_calls(node.args[0])
    if isinstance(node, ast.UnaryOp):
        return _direct_scalar_source_calls(node.operand)
    return []


def _direct_scalar_evidence_reads(node: ast.AST) -> list[dict]:
    source_calls = _direct_scalar_source_calls(node)
    if source_calls:
        return [{"kind": "source_value", **call} for call in source_calls]
    if isinstance(node, (ast.List, ast.Tuple, ast.Dict, ast.Set, ast.ListComp, ast.DictComp)):
        return []
    reads: list[dict] = []
    seen: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Subscript):
            continue
        rendered = ast.unparse(child)
        if ".iloc[" not in rendered and ".values[" not in rendered:
            continue
        if rendered in seen:
            continue
        seen.add(rendered)
        reads.append({"kind": "dataframe_scalar", "expression": rendered})
    return reads


def _assignment_map(tree: ast.AST) -> tuple[dict[str, list[ast.AST]], list[ast.AST]]:
    assignments: dict[str, list[ast.AST]] = {}
    result_nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if not isinstance(target, ast.Name):
                continue
            assignments.setdefault(target.id, []).append(value)
            if target.id == "result":
                result_nodes.append(value)
    return assignments, result_nodes


def _reachable_names(
    roots: list[ast.AST], assignments: dict[str, list[ast.AST]]
) -> set[str]:
    reachable: set[str] = set()
    visiting: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            name = node.id
            reachable.add(name)
            if name in assignments and name not in visiting:
                visiting.add(name)
                for value in assignments[name]:
                    visit(value)
                visiting.remove(name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for root in roots:
        visit(root)
    return reachable


def audit(submission_dir: Path) -> dict:
    submission_dir = submission_dir.resolve()
    rows = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict] = []
    programs_with_scalar_sources = 0
    source_bindings = 0
    diagnostic_bindings_excluded = 0

    for row in rows:
        code = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        assignments, result_nodes = _assignment_map(tree)
        if not result_nodes:
            continue
        bound_sources = {
            name: reads
            for name, values in assignments.items()
            if name != "result"
            and (
                reads := [
                    read
                    for value in values
                    for read in _direct_scalar_evidence_reads(value)
                ]
            )
        }
        if not bound_sources:
            continue
        diagnostic_bindings_excluded += sum(
            name.startswith("_recall") for name in bound_sources
        )
        bound_sources = {
            name: reads
            for name, reads in bound_sources.items()
            if not name.startswith("_recall")
        }
        if not bound_sources:
            continue
        programs_with_scalar_sources += 1
        source_bindings += len(bound_sources)
        control_nodes = [
            node.test
            for node in ast.walk(tree)
            if isinstance(node, (ast.If, ast.While))
        ] + [
            node.test for node in ast.walk(tree) if isinstance(node, ast.Assert)
        ]
        # A reconciliation value can guard whether a result is emitted at all.
        # Treat control predicates conservatively as part of the result graph.
        reachable = _reachable_names(result_nodes + control_nodes, assignments)
        unused = [
            {"binding": name, "evidence_reads": reads}
            for name, reads in sorted(bound_sources.items())
            if name not in reachable
        ]
        if unused:
            findings.append(
                {
                    "id": int(row["id"]),
                    "question": row.get("question", ""),
                    "answer": row.get("answer"),
                    "unused_binding_count": len(unused),
                    "unused_bindings": unused,
                }
            )

    findings.sort(key=lambda item: (-item["unused_binding_count"], item["id"]))
    return {
        "submission": str(submission_dir),
        "programs_with_scalar_sources": programs_with_scalar_sources,
        "source_binding_count": source_bindings,
        "diagnostic_bindings_excluded": diagnostic_bindings_excluded,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
        "claim_limit": (
            "An unused source binding may be an intentional diagnostic or cross-check. "
            "Change an answer only after checking the original source and recomputing it."
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
