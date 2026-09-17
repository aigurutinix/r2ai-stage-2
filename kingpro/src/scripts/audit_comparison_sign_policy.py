"""Find multi-value comparisons with inconsistent accounting sign handling.

An argmax/argmin is unstable when some yearly/company operands are wrapped in
``abs`` and their peers are not. The inconsistency can be legitimate when the
source reports change presentation conventions, but it must be reconciled
against the row label and raw tokens instead of inherited accidentally.

Read-only audit: findings require source review before any submission change.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


VALUE_RE = re.compile(r"v\d+")


def _uses_abs(node: ast.AST) -> bool:
    return any(
        isinstance(item, ast.Call)
        and isinstance(item.func, ast.Name)
        and item.func.id == "abs"
        for item in ast.walk(node)
    )


def _manifest_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(submission_dir: Path) -> dict:
    rows = json.loads(
        (submission_dir / "submission.json").read_text(encoding="utf-8")
    )
    findings: list[dict] = []
    for row in rows:
        code = str(row.get("pandas_query", ""))
        if "max(" not in code and "min(" not in code and "idxmax(" not in code and "idxmin(" not in code:
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        assignments: list[dict] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or not VALUE_RE.fullmatch(target.id):
                continue
            assignments.append(
                {
                    "name": target.id,
                    "uses_abs": _uses_abs(node.value),
                    "expression": ast.unparse(node.value),
                }
            )
        if len(assignments) < 2:
            continue
        policies = {item["uses_abs"] for item in assignments}
        if len(policies) < 2:
            continue
        qid = int(row["id"])
        manifest = _manifest_rows(submission_dir / "data" / f"q{qid}_source_cells.csv")
        findings.append(
            {
                "id": qid,
                "question": row.get("question", ""),
                "answer": row.get("answer"),
                "assignments": sorted(assignments, key=lambda item: item["name"]),
                "source_operands": [
                    {
                        "ticker": item.get("ticker"),
                        "year": item.get("year"),
                        "metric_key": item.get("metric_key"),
                        "raw": item.get("raw"),
                        "source_table": item.get("source_table"),
                    }
                    for item in manifest
                ],
            }
        )
    return {
        "submission": str(submission_dir.resolve()),
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
        "claim_limit": (
            "Mixed abs policy is a review signal only. Preserve statement "
            "semantics and source presentation when deciding the sign policy."
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
