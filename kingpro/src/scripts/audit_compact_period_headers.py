"""Audit period headers for every compact source-cell manifest.

The older period audit is intentionally limited to questions represented in
``source_audit.json``.  Newer candidates carry exact physical coordinates in
``data/q*_source_cells.csv`` for a larger set of questions.  This audit opens
those original HTML tables and checks the closest unambiguous column-period
header against the operand year.

Only explicit years and ``Năm nay/Năm trước`` style headers are judged.  Bare
``Số đầu năm/Số cuối năm`` labels inside movement tables are excluded because
they frequently describe row sections rather than the column period.  A
finding is triage evidence, never an automatic answer repair.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

from audit_audited_period_columns import expected_column_year
from audit_period_columns import descriptor_explicit_year, fold
from audit_source_cell_semantics import parsed_table


def report_year_from_table(source_table: object) -> int | None:
    document = str(source_table).split("|", 1)[0]
    match = re.search(r"_financial_statements_((?:19|20)\d{2})(?:_|$)", document)
    return int(match.group(1)) if match else None


def is_explicit_period_descriptor(value: object) -> bool:
    descriptor = fold(value)
    if not descriptor:
        return False
    if re.match(
        r"^(?:(?:vnd|trieu dong|nghin vnd|nghin dong)\s+)?"
        r"(?:nam nay|nam truoc|ky nay|ky truoc)(?=\b|trieu|vnd|nghin)",
        descriptor,
    ):
        return True
    if any(token in descriptor for token in (
        "nam tai chinh ket thuc",
        "tai ngay 31",
        "vao ngay 31",
        "31/12/",
        "31 thang 12 nam",
        "01/01/",
        "1/1/",
    )):
        return descriptor_explicit_year(descriptor) is not None
    return bool(re.fullmatch(
        r"(?:nam\s+)?(?:19|20)\d{2}(?:\s*(?:vnd|trieu dong|nghin vnd|nghin dong))?",
        descriptor,
    ))


def nearest_period_descriptor(frame: Any, row: int, column: int) -> str:
    """Return the closest high-confidence column period above a cell."""

    if row <= 0 or column < 0 or column >= len(frame.columns):
        return ""
    for candidate in range(row - 1, max(-1, row - 20), -1):
        value = str(frame.iloc[candidate, column]).strip()
        if not is_explicit_period_descriptor(value):
            continue
        # Do not mistake an ordinary data row containing a year-like scalar
        # for a header.  A true period-header row should not also contain a
        # large accounting amount in another column.
        other_values = [
            str(frame.iloc[candidate, index]).strip()
            for index in range(len(frame.columns))
            if index != column
        ]
        if any(
            re.fullmatch(r"\(?\d+(?:[.,]\d+)*\)?", re.sub(r"\s+", "", item))
            and len(re.sub(r"\D", "", item)) >= 5
            for item in other_values
        ):
            continue
        return fold(value)
    return ""


def audit(submission_dir: Path) -> dict[str, Any]:
    submission_dir = submission_dir.resolve()
    rows = {
        int(item["id"]): item
        for item in json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    }
    checked_questions: set[int] = set()
    checked_cells = 0
    unresolved: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    for path in sorted((submission_dir / "data").glob("q*_source_cells.csv")):
        match = re.fullmatch(r"q(\d+)_source_cells", path.stem)
        if not match:
            continue
        qid = int(match.group(1))
        with path.open(encoding="utf-8-sig", newline="") as handle:
            sources = list(csv.DictReader(handle))
        for source_index, source in enumerate(sources):
            source_table = str(source.get("source_table", ""))
            try:
                operand_year = int(source["year"])
                row = int(source["row_idx"])
                column = int(source["col_idx"])
                frame = parsed_table(source_table)
                descriptor = nearest_period_descriptor(frame, row, column)
            except (KeyError, TypeError, ValueError, IndexError) as error:
                unresolved.append({
                    "id": qid,
                    "source_index": source_index,
                    "source_table": source_table,
                    "error": f"{type(error).__name__}: {error}",
                })
                continue
            if not descriptor:
                continue
            report_year = report_year_from_table(source_table)
            expected = expected_column_year(
                Path(source_table.split("|", 1)[0] + ".csv"),
                descriptor,
                str(source.get("metric_key", "")),
                report_year,
            )
            if expected is None:
                continue
            checked_questions.add(qid)
            checked_cells += 1
            column_year, rule = expected
            if operand_year == column_year:
                continue
            findings.append({
                "id": qid,
                "source_index": source_index,
                "operand_year": operand_year,
                "column_year": column_year,
                "report_year": report_year,
                "rule": rule,
                "descriptor": descriptor,
                "metric_key": source.get("metric_key"),
                "raw": source.get("raw"),
                "source_table": source_table,
                "row_idx": row,
                "col_idx": column,
                "question": rows.get(qid, {}).get("question"),
                "answer": rows.get(qid, {}).get("answer"),
            })

    return {
        "submission": str(submission_dir),
        "questions_with_explicit_period_headers": len(checked_questions),
        "period_cells_checked": checked_cells,
        "finding_count": len(findings),
        "question_ids": sorted({int(item["id"]) for item in findings}),
        "resolution_error_count": len(unresolved),
        "policy": "Read-only high-confidence period triage; inspect the full table before repair.",
        "findings": findings,
        "errors": unresolved,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()
    payload = audit(args.submission_dir)
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.fail_on_findings and payload["finding_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
