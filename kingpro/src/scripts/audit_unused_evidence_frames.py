"""Find declared evidence frames with no dependency path to ``result``.

This complements ``audit_unused_source_assignments.py``.  Legacy questions can
package several CSV frames, bind them positionally from ``list(dfs.values())``,
and then use only one bound frame.  Those dead frames increase payload size and
can preserve dangerous same-label collisions even when ``relevant_tables`` is
already minimal (q300).

The detector is intentionally fail-closed: it reports only frames whose
positional binding is explicit, or declared frame names that are directly
present in the program.  Dynamic iteration over all ``dfs`` values is skipped.
Findings authorize review/cleanup only; they never authorize answer changes.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def assignment_map(tree: ast.AST) -> tuple[dict[str, list[ast.AST]], list[ast.AST]]:
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


def reachable_names(roots: list[ast.AST], assignments: dict[str, list[ast.AST]]) -> set[str]:
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


def positional_index(node: ast.AST) -> int | None:
    """Return the explicit index in ``_dfvals[i]``/``list(dfs.values())[i]``."""

    if not isinstance(node, ast.Subscript):
        return None
    index = node.slice.value if isinstance(node.slice, ast.Index) else node.slice
    if not isinstance(index, ast.Constant) or not isinstance(index.value, int):
        return None
    base = node.value
    if isinstance(base, ast.Name) and base.id == "_dfvals":
        return index.value
    if (
        isinstance(base, ast.Call)
        and isinstance(base.func, ast.Name)
        and base.func.id == "list"
        and len(base.args) == 1
    ):
        values_call = base.args[0]
        if (
            isinstance(values_call, ast.Call)
            and isinstance(values_call.func, ast.Attribute)
            and isinstance(values_call.func.value, ast.Name)
            and values_call.func.value.id == "dfs"
            and values_call.func.attr == "values"
        ):
            return index.value
    return None


def explicit_bindings(
    assignments: dict[str, list[ast.AST]], evidence_count: int
) -> dict[str, int]:
    bindings: dict[str, int] = {}
    for name, values in assignments.items():
        indices = {index for value in values if (index := positional_index(value)) is not None}
        if len(indices) == 1:
            index = next(iter(indices))
            if 0 <= index < evidence_count:
                bindings[name] = index
    return bindings


def audit_row(row: dict) -> dict | None:
    evidence = row.get("evidence")
    if not isinstance(evidence, list) or len(evidence) < 2:
        return None
    code = str(row.get("pandas_query", ""))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    assignments, result_nodes = assignment_map(tree)
    if not result_nodes:
        return None
    control_nodes = [
        node.test for node in ast.walk(tree) if isinstance(node, (ast.If, ast.While, ast.Assert))
    ]
    reachable = reachable_names(result_nodes + control_nodes, assignments)
    bindings = explicit_bindings(assignments, len(evidence))
    if not bindings:
        return None

    used_indices = {index for name, index in bindings.items() if name in reachable}
    bound_indices = set(bindings.values())
    unused_indices = sorted(bound_indices - used_indices)
    if not used_indices or not unused_indices:
        return None

    return {
        "id": int(row["id"]),
        "question": row.get("question"),
        "answer": row.get("answer"),
        "evidence_count": len(evidence),
        "explicit_binding_count": len(bindings),
        "used_indices": sorted(used_indices),
        "unused_indices": unused_indices,
        "used_evidence": [evidence[index] for index in sorted(used_indices)],
        "unused_evidence": [evidence[index] for index in unused_indices],
        "bindings": bindings,
    }


def audit(submission_dir: Path) -> dict:
    submission_dir = submission_dir.resolve()
    rows = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    findings = [finding for row in rows if (finding := audit_row(row)) is not None]
    findings.sort(key=lambda item: (-len(item["unused_indices"]), item["id"]))
    return {
        "submission": str(submission_dir),
        "question_count": len(rows),
        "finding_count": len(findings),
        "unused_frame_count": sum(len(item["unused_indices"]) for item in findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
        "claim_limit": (
            "A finding authorizes evidence-cleanup review only. Preserve answers, "
            "declared retrieval and code unless independent source evidence proves a change."
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
