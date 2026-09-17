"""Find explicit arithmetic means whose divisor disagrees with source groups.

Adjacent financial-QA benchmarks repeatedly report aggregation errors where
the right cells are retrieved but the program divides by the wrong number of
entities/years.  This read-only audit compares the small constant divisors in
the final ``result`` expression with the number of distinct ticker-year groups
in each source-cell manifest.

Dynamic dataframe ``.mean()`` programs and questions without a source manifest
are skipped.  Findings are triage only and require source review.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _fold(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value).casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def _result_nodes(tree: ast.AST) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets):
            nodes.append(node.value)
    return nodes


def _constant_divisors(nodes: list[ast.AST]) -> list[float]:
    values: list[float] = []
    for root in nodes:
        for node in ast.walk(root):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            right = node.right
            if isinstance(right, ast.Constant) and isinstance(right.value, (int, float)):
                values.append(float(right.value))
    return values


def _has_dynamic_mean(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute) and node.func.attr == "mean":
            return True
        if isinstance(node.func, ast.Name) and node.func.id == "mean":
            return True
    return False


def _is_output_mean(question: str) -> bool:
    """Separate an output aggregation from an average-balance sub-formula."""

    if not re.search(r"\b(trung binh|binh quan)\b", question):
        return False
    # These questions select an extreme/difference/ratio whose *input metric*
    # happens to contain an average balance or an already disclosed average.
    if re.search(
        r"\b(chenh lech|su chenh lech|hieu |cao nhat|thap nhat|lon nhat|nho nhat|"
        r"muc thay doi|tai doanh nghiep|tai cong ty|tai nam)\b",
        question,
    ):
        return False
    if re.search(
        r"(tai san|von chu so huu|hang ton kho)\s+(?:thuan\s+)?(?:trung binh|binh quan)|"
        r"(?:trung binh|binh quan)\s+(?:tong\s+)?(?:tai san|hang ton kho)|"
        r"thu nhap binh quan (?:thang|nam)|ty le bieu quyet trung binh|"
        r"binh quan (?:dau|cuoi) (?:nam|ky)",
        question,
    ):
        return False
    return True


def _source_groups(path: Path) -> tuple[set[tuple[str, str]], int]:
    if not path.is_file():
        return set(), 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = {
        (str(row.get("ticker", "")).strip(), str(row.get("year", "")).strip())
        for row in rows
        if str(row.get("ticker", "")).strip() and str(row.get("year", "")).strip()
    }
    return groups, len(rows)


def audit(submission_dir: Path) -> dict:
    records = json.loads(
        (submission_dir / "submission.json").read_text(encoding="utf-8")
    )
    findings: list[dict] = []
    checked = 0
    skipped_dynamic = 0
    skipped_no_manifest = 0
    for record in records:
        question = _fold(record.get("question", ""))
        if not _is_output_mean(question):
            continue
        qid = int(record["id"])
        groups, source_cell_count = _source_groups(
            submission_dir / "data" / "q{}_source_cells.csv".format(qid)
        )
        if len(groups) < 2:
            skipped_no_manifest += 1
            continue
        try:
            tree = ast.parse(str(record.get("pandas_query", "")))
        except SyntaxError:
            continue
        result_nodes = _result_nodes(tree)
        if _has_dynamic_mean(tree):
            skipped_dynamic += 1
            continue
        checked += 1
        divisors = _constant_divisors(result_nodes)
        small_divisors = sorted({value for value in divisors if 2 <= value <= 20})
        expected = len(groups)
        if any(abs(value - expected) < 1e-9 for value in small_divisors):
            continue
        findings.append(
            {
                "id": qid,
                "question": record.get("question"),
                "answer": record.get("answer"),
                "source_group_count": expected,
                "source_groups": sorted([list(group) for group in groups]),
                "source_cell_count": source_cell_count,
                "small_constant_divisors": small_divisors,
                "all_constant_divisors": divisors,
                "reason": "final expression has no divisor matching distinct ticker-year source groups",
            }
        )
    return {
        "submission": str(submission_dir.resolve()),
        "question_count": len(records),
        "explicit_mean_programs_checked": checked,
        "dynamic_mean_programs_skipped": skipped_dynamic,
        "missing_or_single_group_manifests_skipped": skipped_no_manifest,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
        "claim_limit": (
            "A source group is a triage proxy for mean arity. Component sums, "
            "weighted averages, and nested period averages require source review."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = audit(args.submission.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
