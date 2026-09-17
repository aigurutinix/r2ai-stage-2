"""Build three source-backed panel-completeness repairs on locked v216.

The missing-operand and selector-coverage audits agree on three live cohort
gaps.  They happen not to change the final answers, but they exclude a named
company before the requested cohort/threshold logic and omit exact source
tables from retrieval labels:

* q409: DIG 2024 total assets;
* q412: VNM 2024 operating cash flow; and
* q430: IJC 2023 comparative income-statement fields.

All values are bound to physical cells.  The builder changes only the three
programs, the two genuinely missing table labels, their compact evidence CSVs,
and matching panel-audit counts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v216_q368_scalar_round_hardened"
OUTPUT = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"

PATCHES = {
    409: {
        "old": "'total_assets': None",
        "new": "'total_assets': _source_value('DIG', 2024, 'cdkt:270')",
        "table": "DIG_financial_statements_2024_consolidated|333",
        "cell": {
            "ticker": "DIG",
            "year": "2024",
            "metric_key": "cdkt:270",
            "raw": "18.539.323.581.176",
            "typed_factor": "1.0",
            "scale": "1.0",
            "source_table": "DIG_financial_statements_2024_consolidated|333",
            "source_csv": "table_4_line333.csv",
            "row_idx": "4",
            "col_idx": "3",
        },
        "expected_table_count": 15,
        "expected_cell_count": 20,
    },
    412: {
        "old": "'cfo': None",
        "new": "'cfo': _source_value('VNM', 2024, 'lctt:20')",
        "table": "VNM_financial_statements_2024_consolidated|318",
        "cell": {
            "ticker": "VNM",
            "year": "2024",
            "metric_key": "lctt:20",
            "raw": "9.685.937.539.346",
            "typed_factor": "1.0",
            "scale": "1.0",
            "source_table": "VNM_financial_statements_2024_consolidated|318",
            "source_csv": "table_11_line318.csv",
            "row_idx": "17",
            "col_idx": "3",
        },
        "expected_table_count": 6,
        "expected_cell_count": 9,
    },
    430: {
        "old": (
            "{'ticker': 'IJC', 'year': 2023, 'revenue': None, "
            "'selling_expense': None, 'admin_expense': None, "
            "'operating_profit': None}"
        ),
        "new": (
            "{'ticker': 'IJC', 'year': 2023, "
            "'revenue': _source_value('IJC', 2023, 'kqkd:10'), "
            "'selling_expense': abs(_source_value('IJC', 2023, 'kqkd:25')), "
            "'admin_expense': abs(_source_value('IJC', 2023, 'kqkd:26')), "
            "'operating_profit': _source_value('IJC', 2023, 'kqkd:30')}"
        ),
        "table": None,
        "cells": [
            {
                "ticker": "IJC",
                "year": "2023",
                "metric_key": metric,
                "raw": raw,
                "typed_factor": "1.0",
                "scale": "1.0",
                "source_table": "IJC_financial_statements_2024_consolidated|445",
                "source_csv": "table_8_line445.csv",
                "row_idx": str(row),
                "col_idx": "4",
            }
            for metric, raw, row in (
                ("kqkd:10", "1.494.344.009.912", 3),
                ("kqkd:25", "37.268.971.708", 10),
                ("kqkd:26", "63.545.736.704", 11),
                ("kqkd:30", "483.074.150.619", 12),
            )
        ],
        "expected_table_count": 15,
        "expected_cell_count": 64,
    },
}

FIELDS = [
    "ticker",
    "year",
    "metric_key",
    "raw",
    "typed_factor",
    "scale",
    "source_table",
    "source_csv",
    "row_idx",
    "col_idx",
]


def _digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _physical_raw(cell: dict[str, str]) -> str:
    document, _ = cell["source_table"].rsplit("|", 1)
    path = ROOT / "build" / "tables" / document / cell["source_csv"]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    # CSV row zero is the header consumed by pandas.read_csv.
    return rows[int(cell["row_idx"]) + 1][int(cell["col_idx"])]


def _append_cells(path: Path, additions: list[dict[str, str]]) -> None:
    # Existing compact panels may carry a UTF-8 BOM.  ``utf-8`` leaves it on
    # the first DictReader key (``\ufeffticker``), which can make a build stop
    # after copytree and leave a deceptively complete-looking partial output.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    existing = {
        (row["ticker"], row["year"], row["metric_key"], row["source_table"])
        for row in rows
    }
    for addition in additions:
        key = (
            addition["ticker"],
            addition["year"],
            addition["metric_key"],
            addition["source_table"],
        )
        if key in existing:
            raise AssertionError(f"duplicate compact source cell: {key}")
        if _physical_raw(addition) != addition["raw"]:
            raise AssertionError(f"physical source mismatch: {key}")
        rows.append(addition)
        existing.add(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    by_id = {int(row["id"]): row for row in rows}

    for qid, patch in PATCHES.items():
        row = by_id[qid]
        code = str(row.get("pandas_query") or "")
        if code.count(patch["old"]) != 1:
            raise AssertionError(f"q{qid}: unexpected missing-operand expression")
        row["pandas_query"] = code.replace(patch["old"], patch["new"], 1)
        table = patch["table"]
        if table is not None:
            if table in row["relevant_tables"]:
                raise AssertionError(f"q{qid}: table already present")
            row["relevant_tables"].append(table)

    changed = [int(row["id"]) for row in rows if row != source_by_id[int(row["id"])]]
    if changed != sorted(PATCHES):
        raise AssertionError(f"unexpected changed IDs: {changed}")
    expected_fields = {
        409: {"pandas_query", "relevant_tables"},
        412: {"pandas_query", "relevant_tables"},
        430: {"pandas_query"},
    }
    for qid in changed:
        fields = {
            key for key in by_id[qid]
            if by_id[qid].get(key) != source_by_id[qid].get(key)
        }
        if fields != expected_fields[qid]:
            raise AssertionError(f"q{qid}: unexpected changed fields {fields}")

    shutil.copytree(SOURCE, OUTPUT)
    _write_json(OUTPUT / "submission.json", rows)
    for qid, patch in PATCHES.items():
        additions = patch.get("cells") or [patch["cell"]]
        _append_cells(OUTPUT / "data" / f"q{qid}_source_cells.csv", additions)

    audits = json.loads((OUTPUT / "panel_source_audit.json").read_text(encoding="utf-8"))
    audit_by_id = {int(item["id"]): item for item in audits}
    for qid, patch in PATCHES.items():
        audit = audit_by_id[qid]
        audit["source_tables"] = patch["expected_table_count"]
        audit["source_cells"] = patch["expected_cell_count"]
        audit["dependency_mode"] = "complete_live_cohort"
    _write_json(OUTPUT / "panel_source_audit.json", audits)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": sorted(PATCHES),
        "answer_changes": [],
        "changed_fields": {
            str(qid): sorted(expected_fields[qid]) for qid in sorted(PATCHES)
        },
        "physical_cells_added": 6,
        "relevant_tables_added": {
            "409": [PATCHES[409]["table"]],
            "412": [PATCHES[412]["table"]],
            "430": [],
        },
        "source_submission_sha256": _digest(source_rows),
        "candidate_submission_sha256": _digest(rows),
        "claim_limit": (
            "Source-backed cohort-completeness batch; leaderboard effect is "
            "unknown until submitted."
        ),
    }
    _write_json(OUTPUT / "v217_missing_panel_operand_batch3_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
