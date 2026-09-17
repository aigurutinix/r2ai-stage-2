"""Static compatibility gate for the Python 3.7 grader contract.

The project executes the dynamic grader checks with pandas 1.1.5.  This gate
adds the part that a newer local interpreter cannot prove: every submitted
query must still parse with the Python 3.7 grammar and must not reference
well-known builtins or standard-library conveniences introduced after 3.7.
"""

from __future__ import print_function

import argparse
import ast
import json
from pathlib import Path


POST_37_NAMES = {
    "aiter": "Python 3.10",
    "anext": "Python 3.10",
}

POST_37_ATTRIBUTES = {
    "bit_count": "Python 3.8",
    "removeprefix": "Python 3.9",
    "removesuffix": "Python 3.9",
}


class CompatibilityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.findings = []

    def visit_Name(self, node):  # noqa: N802 - ast visitor API
        if node.id in POST_37_NAMES:
            self.findings.append(
                {
                    "line": getattr(node, "lineno", None),
                    "kind": "post_37_name",
                    "token": node.id,
                    "introduced": POST_37_NAMES[node.id],
                }
            )
        self.generic_visit(node)

    def visit_Attribute(self, node):  # noqa: N802 - ast visitor API
        if node.attr in POST_37_ATTRIBUTES:
            self.findings.append(
                {
                    "line": getattr(node, "lineno", None),
                    "kind": "post_37_attribute",
                    "token": node.attr,
                    "introduced": POST_37_ATTRIBUTES[node.attr],
                }
            )
        self.generic_visit(node)


def parse_python37(source):
    """Use the oldest grammar exposed by the running CPython AST module."""

    try:
        return ast.parse(source, mode="exec", feature_version=7)
    except TypeError:
        # Python releases that accept the documented major/minor tuple.
        return ast.parse(source, mode="exec", feature_version=(3, 7))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()

    submission_path = args.submission_dir / "submission.json"
    rows = json.loads(submission_path.read_text(encoding="utf-8"))
    findings = []

    for row in rows:
        question_id = row.get("id")
        query = row.get("pandas_query") or ""
        try:
            tree = parse_python37(query)
        except SyntaxError as exc:
            findings.append(
                {
                    "id": question_id,
                    "kind": "python37_syntax",
                    "line": exc.lineno,
                    "message": exc.msg,
                }
            )
            continue

        visitor = CompatibilityVisitor()
        visitor.visit(tree)
        for finding in visitor.findings:
            finding["id"] = question_id
            findings.append(finding)

    report = {
        "submission": str(args.submission_dir),
        "queries_checked": len(rows),
        "python_grammar": "3.7",
        "finding_count": len(findings),
        "findings": findings,
    }

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.fail_on_findings and findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
