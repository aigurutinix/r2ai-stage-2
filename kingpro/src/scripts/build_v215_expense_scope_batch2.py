"""Build two additional source-backed expense-scope repairs on v214.

Both prompts ask for an unqualified expense element.  The previous rows came
from narrower selling/administrative-expense disclosures even though each
report also contains the general ``chi phi SXKD theo yeu to`` disclosure.
Only q775 and q923 are changed; all earlier cumulative fixes remain intact.
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
SOURCE = ROOT / "sub_top123_candidate_v214_scope2_ablation"
OUTPUT = ROOT / "sub_top123_candidate_v215_expense_scope_batch2"

PATCHES = {
    775: {
        "old_answer": 16.82,
        "new_answer": 2148.40,
        "old_tables": [
            "SNZ_financial_statements_2022_consolidated|2030",
            "VPI_financial_statements_2022_consolidated|1479",
        ],
        "new_tables": [
            "SNZ_financial_statements_2022_consolidated|2034",
            "VPI_financial_statements_2022_consolidated|1498",
        ],
        "cells": [
            {
                "ticker": "SNZ",
                "year": "2022",
                "metric_key": "note:general_outsourced_service_expense",
                "raw": "2.529.098.642.906",
                "typed_factor": "1.0",
                "scale": "1.0",
                "source_table": "SNZ_financial_statements_2022_consolidated|2034",
                "source_csv": "SNZ_financial_statements_2022_consolidated_2034.csv",
                "row_idx": "5",
                "col_idx": "1",
            },
            {
                "ticker": "VPI",
                "year": "2022",
                "metric_key": "note:general_outsourced_service_expense",
                "raw": "380.697.368.367",
                "typed_factor": "1.0",
                "scale": "1.0",
                "source_table": "VPI_financial_statements_2022_consolidated|1498",
                "source_csv": "VPI_financial_statements_2022_consolidated_1498.csv",
                "row_idx": "5",
                "col_idx": "1",
            },
        ],
    },
    923: {
        "old_answer": 131.98,
        "new_answer": 1102.67,
        "old_tables": [
            "OGC_financial_statements_2016_consolidated|2064",
            "OGC_financial_statements_2017_consolidated|1992",
            "OGC_financial_statements_2018_consolidated|2212",
            "OGC_financial_statements_2019_consolidated|2037",
        ],
        "new_tables": [
            "OGC_financial_statements_2016_consolidated|2105",
            "OGC_financial_statements_2017_consolidated|2054",
            "OGC_financial_statements_2018_consolidated|2266",
            "OGC_financial_statements_2019_consolidated|2087",
        ],
        "cells": [
            {
                "ticker": "OGC",
                "year": str(year),
                "metric_key": "note:general_employee_cost",
                "raw": raw,
                "typed_factor": "1.0",
                "scale": "1.0",
                "source_table": f"OGC_financial_statements_{year}_consolidated|{line}",
                "source_csv": f"OGC_financial_statements_{year}_consolidated_{line}.csv",
                "row_idx": "3",
                "col_idx": "1",
            }
            for year, line, raw in (
                (2016, 2105, "239.080.080.571"),
                (2017, 2054, "261.585.083.957"),
                (2018, 2266, "287.946.510.328"),
                (2019, 2087, "314.060.717.471"),
            )
        ],
    },
}

RAW_TABLES = {
    "OGC_financial_statements_2016_consolidated_2105.csv": (
        "OGC_financial_statements_2016_consolidated",
        "table_57_line2105.csv",
    ),
    "OGC_financial_statements_2017_consolidated_2054.csv": (
        "OGC_financial_statements_2017_consolidated",
        "table_60_line2054.csv",
    ),
    "OGC_financial_statements_2018_consolidated_2266.csv": (
        "OGC_financial_statements_2018_consolidated",
        "table_60_line2266.csv",
    ),
    "OGC_financial_statements_2019_consolidated_2087.csv": (
        "OGC_financial_statements_2019_consolidated",
        "table_62_line2087.csv",
    ),
}


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_cells(path: Path, rows: list[dict[str, str]]) -> None:
    fields = [
        "ticker", "year", "metric_key", "raw", "typed_factor", "scale",
        "source_table", "source_csv", "row_idx", "col_idx",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
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
        if float(row["answer"]) != patch["old_answer"]:
            raise AssertionError(f"q{qid} unexpected old answer: {row['answer']}")
        if row["relevant_tables"] != patch["old_tables"]:
            raise AssertionError(f"q{qid} unexpected old tables: {row['relevant_tables']}")
        row["answer"] = patch["new_answer"]
        row["relevant_tables"] = patch["new_tables"]

    changed = [int(row["id"]) for row in rows if row != source_by_id[int(row["id"])]]
    if changed != sorted(PATCHES):
        raise AssertionError(f"unexpected changed IDs: {changed}")
    for qid in changed:
        fields = {
            key for key in by_id[qid]
            if by_id[qid].get(key) != source_by_id[qid].get(key)
        }
        if fields != {"answer", "relevant_tables"}:
            raise AssertionError(f"q{qid} unexpected changed fields: {fields}")

    shutil.copytree(SOURCE, OUTPUT)
    _write_json(OUTPUT / "submission.json", rows)
    for qid, patch in PATCHES.items():
        _write_cells(OUTPUT / "data" / f"q{qid}_source_cells.csv", patch["cells"])

    for target_name, (doc, file_name) in RAW_TABLES.items():
        source_path = ROOT / "build" / "tables" / doc / file_name
        target_path = OUTPUT / "data" / target_name
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if not target_path.exists():
            shutil.copy2(source_path, target_path)

    audits = json.loads((OUTPUT / "source_audit.json").read_text(encoding="utf-8"))
    audit_by_id = {int(item["id"]): item for item in audits}
    updated_audit_ids: list[int] = []
    for qid, patch in PATCHES.items():
        item = audit_by_id.get(qid)
        if item is None:
            continue
        if float(item["answer"]) != patch["old_answer"]:
            raise AssertionError(f"q{qid} unexpected source-audit answer")
        item["answer"] = patch["new_answer"]
        item["note"] = (
            "Unqualified expense element resolved to the general "
            f"SXKD-by-element disclosure(s): {patch['new_tables']}."
        )
        updated_audit_ids.append(qid)
    _write_json(OUTPUT / "source_audit.json", audits)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": sorted(PATCHES),
        "repairs": {str(qid): patch for qid, patch in PATCHES.items()},
        "source_audit_rows_updated": updated_audit_ids,
        "internal_controls": {
            "q360": "retained: prompt does not say owner contributed capital; current rows are disclosed capital-contribution transactions",
            "q519_q537": "retained: prompts quote exact narrow disclosure metrics",
            "q531": "retained: transport metric is specific to selling expense",
            "q646": "retained: no general SXKD-by-element table found",
            "q773": "retained: already uses general disclosure for the target value",
            "q789": "retained: prompt says total and intentionally sums selling plus admin",
        },
        "source_submission_sha256": _digest(source_rows),
        "candidate_submission_sha256": _digest(rows),
        "claim_limit": "Source-backed ablation; public effect unknown until submitted.",
    }
    _write_json(OUTPUT / "v215_expense_scope_batch2_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
