"""Find selector questions whose source panel covers a metric unevenly.

This is a research audit, not a correctness oracle.  It catches the failure mode
where a query asks for the company/year with the largest selector metric but one
candidate silently lacks that metric (q511 before v191).  Every finding still
requires checking the original financial statement before changing an answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


SELECTOR_RE = re.compile(
    r"\b(cao nhất|thấp nhất|lớn nhất|nhỏ nhất|tối đa|tối thiểu|nhiều nhất|ít nhất)\b",
    re.IGNORECASE,
)


def _read_panel(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _cohort_findings(
    *,
    qid: int,
    question: str,
    axis: str,
    cohorts: dict[str, list[tuple[str, str]]],
    metrics_by_group: dict[tuple[str, str], set[str]],
) -> list[dict]:
    findings: list[dict] = []
    for cohort_name, groups in sorted(cohorts.items()):
        groups = sorted(set(groups))
        if len(groups) < 2:
            continue
        coverage: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for group in groups:
            for metric in metrics_by_group[group]:
                if metric and not metric.startswith("label:"):
                    coverage[metric].append(group)
        for metric, present_groups in sorted(coverage.items()):
            if len(present_groups) == len(groups):
                continue
            missing_groups = sorted(set(groups) - set(present_groups))
            findings.append(
                {
                    "id": qid,
                    "question": question,
                    "axis": axis,
                    "cohort": cohort_name,
                    "metric_key": metric,
                    "present_groups": [list(group) for group in present_groups],
                    "missing_groups": [list(group) for group in missing_groups],
                    "coverage": len(present_groups),
                    "cohort_size": len(groups),
                    "coverage_ratio": round(len(present_groups) / len(groups), 4),
                    "review_priority": (
                        "high"
                        if len(present_groups) >= math.ceil(len(groups) / 2)
                        else "medium"
                    ),
                }
            )
    return findings


def audit(submission_dir: Path, *, include_cross_year: bool = False) -> dict:
    submission_dir = submission_dir.resolve()
    rows = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    findings: list[dict] = []
    selector_questions = 0
    panels_checked = 0

    for row in rows:
        question = str(row.get("question", ""))
        if not SELECTOR_RE.search(question):
            continue
        selector_questions += 1
        qid = int(row["id"])
        panel_path = submission_dir / "data" / f"q{qid}_source_cells.csv"
        if not panel_path.is_file():
            continue
        panel = _read_panel(panel_path)
        if not panel:
            continue
        panels_checked += 1

        metrics_by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
        for source in panel:
            ticker = str(source.get("ticker", "")).strip().upper()
            year = str(source.get("year", "")).strip()
            metric = str(source.get("metric_key", "")).strip()
            if ticker and year and metric:
                metrics_by_group[(ticker, year)].add(metric)

        by_year: dict[str, list[tuple[str, str]]] = defaultdict(list)
        by_ticker: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for group in metrics_by_group:
            by_year[group[1]].append(group)
            by_ticker[group[0]].append(group)

        findings.extend(
            _cohort_findings(
                qid=qid,
                question=question,
                axis="companies_within_year",
                cohorts=by_year,
                metrics_by_group=metrics_by_group,
            )
        )
        # Raw metric keys are not a stable schema across reporting years.
        # Prior-year balance inputs can be needed only for one terminal year,
        # note categories legitimately appear/disappear, and bespoke note keys
        # often encode the year.  Keep this noisy view available for research,
        # but do not mix it into the default actionable queue.
        if include_cross_year:
            findings.extend(
                _cohort_findings(
                    qid=qid,
                    question=question,
                    axis="years_within_company",
                    cohorts=by_ticker,
                    metrics_by_group=metrics_by_group,
                )
            )

    high = [finding for finding in findings if finding["review_priority"] == "high"]
    return {
        "submission": str(submission_dir),
        "selector_questions": selector_questions,
        "panels_checked": panels_checked,
        "include_cross_year": include_cross_year,
        "finding_count": len(findings),
        "high_priority_count": len(high),
        "high_priority_question_ids": sorted({finding["id"] for finding in high}),
        "findings": findings,
        "claim_limit": (
            "Coverage asymmetry is a review signal only. A source-backed repair is "
            "required before changing any submission row."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument(
        "--include-cross-year",
        action="store_true",
        help="include noisy raw metric-key comparisons across years",
    )
    args = parser.parse_args()
    report = audit(args.submission, include_cross_year=args.include_cross_year)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
