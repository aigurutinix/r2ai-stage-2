"""Audit threshold wording against business comparison operators.

Financial QA programs often differ by a single boundary operator: ``>`` vs
``>=`` or ``<`` vs ``<=``.  The numerical result may stay unchanged on the
development rows, so runtime replay alone cannot expose the semantic defect.

This audit is deliberately read-only.  It ignores comparisons inside helper
function definitions (number parsing, source lookup, and guards), recognizes
both Python operators and Pandas ``gt/ge/lt/le`` calls, and reports only a
clear strict/inclusive contradiction.  Missing or ambiguous comparisons are
kept in a separate unresolved queue rather than treated as defects.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


NUMBER = r"[-+]?\d+(?:[.,]\d+)?\s*%?"
BOUNDARY = rf"(?:{NUMBER}|mức\s+trung\s+vị|trung\s+vị|ngưỡng)"

RULES: tuple[tuple[str, str, re.Pattern[str]], ...] = (
    (
        "inclusive_lower",
        ">=",
        re.compile(
            rf"(?:(?:bằng\s+hoặc\s+)?(?:cao|lớn)\s+hơn\s+hoặc\s+bằng"
            rf"|bằng\s+hoặc\s+(?:cao|lớn)\s+hơn"
            rf"|ít\s+nhất|không\s+(?:thấp|nhỏ)\s+hơn)\s+{BOUNDARY}"
            rf"|(?:từ\s+{NUMBER}\s*(?:lần|%|phần\s+trăm)?\s+trở\s+lên)",
            re.IGNORECASE,
        ),
    ),
    (
        "inclusive_upper",
        "<=",
        re.compile(
            rf"(?:(?:bằng\s+hoặc\s+)?(?:thấp|nhỏ)\s+hơn\s+hoặc\s+bằng"
            rf"|bằng\s+hoặc\s+(?:thấp|nhỏ)\s+hơn"
            rf"|không\s+(?:cao|lớn)\s+hơn|không\s+vượt\s+quá|tối\s+đa)"
            rf"\s+{BOUNDARY}|(?:{NUMBER}\s*(?:lần|%|phần\s+trăm)?\s+trở\s+xuống)",
            re.IGNORECASE,
        ),
    ),
    (
        "strict_lower",
        ">",
        re.compile(
            rf"(?:lớn|cao)\s+hơn\s+(?:mức\s+)?{BOUNDARY}",
            re.IGNORECASE,
        ),
    ),
    (
        "strict_upper",
        "<",
        re.compile(
            rf"(?:nhỏ|thấp)\s+hơn\s+(?:mức\s+)?{BOUNDARY}",
            re.IGNORECASE,
        ),
    ),
)

AST_OPERATOR = {
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Lt: "<",
    ast.LtE: "<=",
}
METHOD_OPERATOR = {"gt": ">", "ge": ">=", "lt": "<", "le": "<="}
BOUNDARY_PAIR = {">": ">=", ">=": ">", "<": "<=", "<=": "<"}


class BusinessComparisonVisitor(ast.NodeVisitor):
    """Collect comparisons outside helper function/class definitions."""

    def __init__(self, source: str) -> None:
        self.source = source
        self.comparisons: list[dict[str, object]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        return

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:  # noqa: N802
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # noqa: N802
        return

    def visit_Compare(self, node: ast.Compare) -> None:  # noqa: N802
        symbols = [AST_OPERATOR[type(operator)] for operator in node.ops if type(operator) in AST_OPERATOR]
        if symbols:
            self.comparisons.append(
                {
                    "operators": symbols,
                    "expression": ast.get_source_segment(self.source, node) or ast.unparse(node),
                    "line": node.lineno,
                }
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        if isinstance(node.func, ast.Attribute) and node.func.attr in METHOD_OPERATOR:
            self.comparisons.append(
                {
                    "operators": [METHOD_OPERATOR[node.func.attr]],
                    "expression": ast.get_source_segment(self.source, node) or ast.unparse(node),
                    "line": node.lineno,
                }
            )
        self.generic_visit(node)


def _expectations(question: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    occupied: list[tuple[int, int]] = []
    # Inclusive rules run first so their embedded words (for example
    # ``không thấp hơn``) are not reclassified as strict comparisons.
    for name, expected, pattern in RULES:
        for match in pattern.finditer(question):
            span = match.span()
            if any(start < span[1] and span[0] < end for start, end in occupied):
                continue
            occupied.append(span)
            hits.append(
                {
                    "rule": name,
                    "expected_operator": expected,
                    "phrase": match.group(0),
                }
            )
    return hits


def _business_comparisons(code: str) -> tuple[list[dict[str, object]], str | None]:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return [], f"{exc.msg} at line {exc.lineno}"
    visitor = BusinessComparisonVisitor(code)
    visitor.visit(tree)
    return visitor.comparisons, None


def audit(submission_dir: Path) -> dict[str, object]:
    rows = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict[str, object]] = []
    unresolved: list[dict[str, object]] = []
    checked = 0
    parse_failures: list[dict[str, object]] = []

    for row in rows:
        question = str(row.get("question", ""))
        expectations = _expectations(question)
        if not expectations:
            continue
        checked += 1
        comparisons, error = _business_comparisons(str(row.get("pandas_query", "")))
        qid = int(row["id"])
        if error:
            parse_failures.append({"id": qid, "error": error})
            continue
        actual = {
            str(operator)
            for comparison in comparisons
            for operator in comparison["operators"]
        }
        if not actual:
            unresolved.append(
                {
                    "id": qid,
                    "question": question,
                    "expectations": expectations,
                    "reason": "no top-level business comparison found",
                }
            )
            continue

        contradictions = [
            expectation
            for expectation in expectations
            if expectation["expected_operator"] not in actual
            and BOUNDARY_PAIR[expectation["expected_operator"]] in actual
        ]
        if contradictions:
            findings.append(
                {
                    "id": qid,
                    "question": question,
                    "answer": row.get("answer"),
                    "contradictions": contradictions,
                    "actual_operators": sorted(actual),
                    "business_comparisons": comparisons,
                }
            )

    return {
        "kind": "threshold_comparison_semantics",
        "submission": str(submission_dir.resolve()),
        "questions_with_threshold_language": checked,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "unresolved_count": len(unresolved),
        "parse_failure_count": len(parse_failures),
        "findings": findings,
        "unresolved": unresolved,
        "parse_failures": parse_failures,
        "claim_limit": (
            "A finding proves a strict/inclusive operator contradiction, not the correct answer. "
            "Review source values and the intended cohort before changing a submission row."
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
