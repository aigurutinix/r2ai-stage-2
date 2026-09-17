"""Find direct queries that read an incompatible financial table family.

The legacy generator can produce executable programs that filter a plausible
row label from a semantically unrelated table.  A recurring example is a
question about debt securities reading the ending row of a loan-loss provision
roll-forward.  Runtime and source-coordinate checks cannot detect that error:
the cell exists and the arithmetic is valid.

This audit deliberately implements only high-precision rules.  It reports a
provision-family collision when the selected value column is explicitly headed
by both general and specific provision labels while the question contains no
provision/risk intent.  Ambiguous tables are left untouched for manual review.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from audit_direct_units import (
    audited_ids,
    direct_reads,
    fold,
    load_csv,
    selected_cell,
)


PROVISION_INTENT = re.compile(
    r"\b(?:du phong|rui ro|no xau|bao phu|trich lap|xu ly no)\b"
)


def column_context(
    fieldnames: list[str], rows: list[dict[str, str]], column: str, row_index: int
) -> str:
    """Return explicit table-family and value-column context before a read."""

    values = [column]
    if column in fieldnames:
        values.append(column)
    # A provision roll-forward usually puts ``Dự phòng chung`` and
    # ``Dự phòng cụ thể`` in adjacent columns.  Inspect the compact header
    # region across the table, then retain the selected column's pre-row
    # values for tables whose header spans more than one extracted row.
    for row in rows[: min(4, row_index)]:
        values.extend(str(row.get(field, "")) for field in fieldnames)
    values.extend(str(row.get(column, "")) for row in rows[:row_index])
    return " ".join(values)


def provision_family_collision(question: str, descriptor: str) -> bool:
    context = fold(descriptor)
    if not (
        re.search(r"\bdu phong chung\b", context)
        and re.search(r"\bdu phong cu the\b", context)
    ):
        return False
    return PROVISION_INTENT.search(fold(question)) is None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-audited", action="store_true")
    parser.add_argument("--fail-on-findings", action="store_true")
    args = parser.parse_args()

    submission = json.loads(
        (args.submission_dir / "submission.json").read_text(encoding="utf-8")
    )
    known = set() if args.include_audited else audited_ids(args.submission_dir)
    checked = 0
    findings: list[dict[str, object]] = []
    for item in submission:
        qid = int(item["id"])
        if qid in known:
            continue
        question = str(item.get("question", ""))
        evidence = {
            str(entry.get("variable")): args.submission_dir
            / str(entry.get("csv_path"))
            for entry in item.get("evidence", [])
            if entry.get("variable") and entry.get("csv_path")
        }
        seen: set[tuple[str, str, int, str]] = set()
        for read in direct_reads(str(item.get("pandas_query", ""))):
            path = evidence.get(read.dataframe)
            if path is None or not path.exists():
                continue
            fieldnames, rows = load_csv(path)
            selected = selected_cell(rows, read)
            if selected is None:
                continue
            row_index, raw = selected
            checked += 1
            descriptor = column_context(fieldnames, rows, read.column, row_index)
            if not provision_family_collision(question, descriptor):
                continue
            key = (read.dataframe, str(path), row_index, read.column)
            if key in seen:
                continue
            seen.add(key)
            findings.append(
                {
                    "id": qid,
                    "kind": "provision-table-family-collision",
                    "question": question,
                    "answer": item.get("answer"),
                    "variable": read.dataframe,
                    "csv": str(path),
                    "row": row_index,
                    "column": read.column,
                    "selected_label": rows[row_index].get("0", ""),
                    "raw": raw,
                    "column_context": descriptor,
                }
            )

    payload = {
        "submission": str(args.submission_dir),
        "audited_ids_skipped": 0 if args.include_audited else len(known),
        "checked_direct_reads": checked,
        "finding_count": len(findings),
        "findings": findings,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.fail_on_findings and findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
