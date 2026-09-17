"""Resolve every literal ``_source_value`` call against its evidence panel.

Panel programs use ``_source_value(ticker, year, metric_key)`` and return
``None`` when no matching manifest row exists.  Pandas may then silently drop
that company/year during ``dropna``, filtering, or ranking.  This audit reports
only calls that the program actually makes, avoiding the broad false positives
of comparing every metric with every cohort member.

Read-only audit: a missing call still needs an original BTC source cell before
it can be repaired.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def _literal(node: ast.AST):
    return node.value if isinstance(node, ast.Constant) else None


def _manifest_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.is_file():
        return set()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    keys: set[tuple[str, int, str]] = set()
    for row in rows:
        try:
            year = int(float(str(row.get("year", ""))))
        except ValueError:
            continue
        keys.add(
            (
                str(row.get("ticker", "")).strip().upper(),
                year,
                str(row.get("metric_key", "")).strip(),
            )
        )
    return keys


def audit(submission_dir: Path) -> dict:
    rows = json.loads(
        (submission_dir / "submission.json").read_text(encoding="utf-8")
    )
    findings: list[dict] = []
    total_calls = 0
    dynamic_calls = 0
    for row in rows:
        code = str(row.get("pandas_query", ""))
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        calls: list[tuple[str, int, str]] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "_source_value"
            ):
                continue
            total_calls += 1
            if len(node.args) < 3:
                dynamic_calls += 1
                continue
            ticker, year, metric = (_literal(arg) for arg in node.args[:3])
            if not isinstance(ticker, str) or not isinstance(year, int) or not isinstance(metric, str):
                dynamic_calls += 1
                continue
            calls.append((ticker.upper(), year, metric))
        if not calls:
            continue
        qid = int(row["id"])
        available = _manifest_keys(
            submission_dir / "data" / f"q{qid}_source_cells.csv"
        )
        missing_counts = Counter(call for call in calls if call not in available)
        if not missing_counts:
            continue
        findings.append(
            {
                "id": qid,
                "question": row.get("question", ""),
                "answer": row.get("answer"),
                "missing_calls": [
                    {
                        "ticker": key[0],
                        "year": key[1],
                        "metric_key": key[2],
                        "call_count": count,
                    }
                    for key, count in sorted(missing_counts.items())
                ],
                "missing_call_count": sum(missing_counts.values()),
                "unique_missing_count": len(missing_counts),
            }
        )
    findings.sort(
        key=lambda item: (-item["unique_missing_count"], item["id"])
    )
    return {
        "submission": str(submission_dir.resolve()),
        "literal_source_value_calls": total_calls - dynamic_calls,
        "dynamic_source_value_calls": dynamic_calls,
        "finding_count": len(findings),
        "question_ids": [item["id"] for item in findings],
        "findings": findings,
        "claim_limit": (
            "A missing runtime lookup is a review signal. Never synthesize its "
            "value; append only a verified BTC source cell."
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
