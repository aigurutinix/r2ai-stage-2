"""Audit negative answers to generic Vietnamese difference-magnitude questions.

The benchmark uses an absolute magnitude for prompts that ask how much one
value ``chênh lệch ... so với`` another.  This is distinct from explicitly
directional prompts such as growth, change, ``trừ đi`` or ``hiệu giữa``.

Read-only audit.  Findings still require source review before a submission is
changed; positive programs using ``abs`` are emitted as internal precedents.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _generic_difference(question: str) -> bool:
    text = question.casefold()
    return (
        "chênh lệch" in text
        and "so với" in text
        and "tỷ lệ chênh lệch" not in text
        and "xét các năm mà chênh lệch" not in text
    )


def _result_uses_abs(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets):
            continue
        if any(
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Name)
            and item.func.id == "abs"
            for item in ast.walk(node.value)
        ):
            return True
    return False


def _record(row: dict, submission: Path) -> dict:
    qid = int(row["id"])
    return {
        "id": qid,
        "answer": row.get("answer"),
        "question": row.get("question", ""),
        "result_uses_abs": _result_uses_abs(str(row.get("pandas_query", ""))),
        "source_operands": [
            {
                "ticker": item.get("ticker"),
                "year": item.get("year"),
                "raw": item.get("raw"),
                "metric_key": item.get("metric_key"),
                "source_table": item.get("source_table"),
            }
            for item in _manifest(submission / "data" / f"q{qid}_source_cells.csv")
            if not str(item.get("metric_key", "")).startswith("recall:")
        ],
    }


def audit(submission: Path) -> dict:
    rows = json.loads((submission / "submission.json").read_text(encoding="utf-8"))
    family = [row for row in rows if _generic_difference(str(row.get("question", "")))]
    negative = [row for row in family if float(row.get("answer", 0)) < 0]
    precedents = [
        row
        for row in family
        if float(row.get("answer", 0)) >= 0
        and _result_uses_abs(str(row.get("pandas_query", "")))
    ]
    findings = [_record(row, submission) for row in negative]
    return {
        "submission": str(submission.resolve()),
        "family_count": len(family),
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
        "positive_abs_precedents": [_record(row, submission) for row in precedents],
        "policy": (
            "Generic 'chênh lệch ... so với' asks for a non-negative magnitude. "
            "Directional growth/change/subtraction prompts retain their sign."
        ),
        "claim_limit": "Review signal only; exact source operands must be confirmed.",
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
