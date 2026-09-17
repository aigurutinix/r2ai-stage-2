"""Build three answer repairs for tables whose internal report scope was wrong.

The source corpus contains compound/swapped OCR documents where the directory
name says ``separate`` while the table masthead says consolidated, or vice
versa.  v217 trusted the directory name for q98, q714 and q764.  This builder
keeps the old retrieval labels (so document recall is not discarded), adds the
physically correct table, and replaces only the answer plus compact lineage.
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
SOURCE = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
OUTPUT = ROOT / "sub_top123_candidate_v219_internal_scope_batch3"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")

FIELDS = [
    "ticker", "year", "metric_key", "raw", "typed_factor", "scale",
    "source_table", "source_csv", "row_idx", "col_idx",
]


def cell(
    ticker: str,
    year: int,
    metric: str,
    raw: str,
    table: str,
    csv_name: str,
    row: int,
    column: int,
) -> dict[str, str]:
    return {
        "ticker": ticker,
        "year": str(year),
        "metric_key": metric,
        "raw": raw,
        "typed_factor": "1.0",
        "scale": "1.0",
        "source_table": table,
        "source_csv": csv_name,
        "row_idx": str(row),
        "col_idx": str(column),
    }


PATCHES = {
    98: {
        "answer": 146.47,
        "docs": ["HUT_financial_statements_2024_consolidated"],
        "tables": ["HUT_financial_statements_2024_consolidated|325"],
        "cells": [cell(
            "HUT", 2024, "cdkt:140", "146.469.679.444",
            "HUT_financial_statements_2024_consolidated|325",
            "table_0_line325.csv", 12, 4,
        )],
        "note": (
            "HUT parent inventory from the internally separate balance sheet "
            "stored in the swapped *_consolidated OCR container, VND billion"
        ),
    },
    714: {
        "answer": 168.74,
        "docs": ["HUT_financial_statements_2024_separate"],
        "tables": ["HUT_financial_statements_2024_separate|399"],
        "cells": [
            cell(
                "HUT", 2024, "kqkd:21", "874.739.630.652",
                "HUT_financial_statements_2024_separate|399",
                "table_3_line399.csv", 6, 4,
            ),
            cell(
                "HUT", 2024, "kqkd:22", "706.004.285.205",
                "HUT_financial_statements_2024_separate|399",
                "table_3_line399.csv", 7, 4,
            ),
        ],
        "note": (
            "HUT consolidated net finance result from the internally consolidated "
            "income statement stored in the swapped *_separate OCR container, VND billion"
        ),
    },
    764: {
        "answer": 2.94,
        "docs": [],
        "tables": ["DPM_financial_statements_2015_consolidated|1909"],
        "cells": [
            cell(
                "GVR", 2015, "cdkt:418", "6.437.295.628.830",
                "GVR_financial_statements_2015_consolidated|290",
                "table_7_line290.csv", 11, 3,
            ),
            cell(
                "DPM", 2015, "cdkt:418", "3.498.666.363.829",
                "DPM_financial_statements_2015_consolidated|1909",
                "table_63_line1909.csv", 28, 3,
            ),
        ],
        "note": (
            "absolute GVR/DPM consolidated development-investment-fund difference; "
            "the earlier DPM line 272 belongs to the separate report, VND trillion"
        ),
    },
}


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def physical_path(item: dict[str, str]) -> Path:
    document = item["source_table"].split("|", 1)[0]
    return ROOT / "build" / "tables" / document / item["source_csv"]


def physical_raw(item: dict[str, str]) -> str:
    with physical_path(item).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[int(item["row_idx"]) + 1][int(item["col_idx"])]


def write_manifest(path: Path, cells: list[dict[str, str]]) -> None:
    for item in cells:
        observed = physical_raw(item)
        if observed != item["raw"]:
            raise AssertionError(
                f"physical source mismatch for {item['source_table']}: "
                f"{observed!r} != {item['raw']!r}"
            )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)


def audit_source(item: dict[str, str]) -> dict[str, object]:
    labels = {
        "cdkt:140": "Hàng tồn kho",
        "cdkt:418": "Quỹ đầu tư phát triển",
        "kqkd:21": "Doanh thu hoạt động tài chính",
        "kqkd:22": "Chi phí tài chính",
    }
    return {
        "table_ref": item["source_table"],
        "csv": item["source_csv"],
        "row": int(item["row_idx"]),
        "column": int(item["col_idx"]),
        "metric": item["metric_key"],
        "label": labels[item["metric_key"]],
        "source_row_labels": [labels[item["metric_key"]]],
        "scale": 1.0,
        "typed_factor": 1.0,
        "raw": item["raw"],
    }


def append_unique(values: list[str], additions: list[str]) -> list[str]:
    return list(dict.fromkeys([*values, *additions]))


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    source_rows = load(SOURCE / "submission.json")
    if not isinstance(source_rows, list):
        raise AssertionError("source submission is not a list")
    rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    source_by_id = {int(row["id"]): row for row in source_rows}
    by_id = {int(row["id"]): row for row in rows}

    for qid, patch in PATCHES.items():
        row = by_id[qid]
        if not row.get("pandas_query"):
            raise AssertionError(f"q{qid}: expected source-cell replay program")
        row["answer"] = patch["answer"]
        row["relevant_docs"] = append_unique(row["relevant_docs"], patch["docs"])
        row["relevant_tables"] = append_unique(row["relevant_tables"], patch["tables"])

    changed = [int(row["id"]) for row in rows if row != source_by_id[int(row["id"])]]
    if changed != sorted(PATCHES):
        raise AssertionError(f"unexpected changed IDs: {changed}")
    allowed = {"answer", "relevant_docs", "relevant_tables"}
    changed_fields: dict[str, list[str]] = {}
    for qid in changed:
        fields = sorted(
            key for key in set(source_by_id[qid]) | set(by_id[qid])
            if source_by_id[qid].get(key) != by_id[qid].get(key)
        )
        if not set(fields) <= allowed or "answer" not in fields:
            raise AssertionError(f"q{qid}: unexpected changed fields {fields}")
        changed_fields[str(qid)] = fields

    shutil.copytree(SOURCE, BUILDING)
    write(BUILDING / "submission.json", rows)

    copied_tables: list[str] = []
    for qid, patch in PATCHES.items():
        write_manifest(BUILDING / "data" / f"q{qid}_source_cells.csv", patch["cells"])
        for item in patch["cells"]:
            if item["source_table"] not in patch["tables"]:
                continue
            destination = BUILDING / "data" / (item["source_table"].replace("|", "_") + ".csv")
            shutil.copy2(physical_path(item), destination)
            copied_tables.append(destination.name)

    source_audit_path = BUILDING / "source_audit.json"
    source_audit = load(source_audit_path)
    if not isinstance(source_audit, list):
        raise AssertionError("source_audit.json is not a list")
    audit_by_id = {int(record["id"]): record for record in source_audit}
    for qid, patch in PATCHES.items():
        record = audit_by_id[qid]
        record["old_answer"] = source_by_id[qid]["answer"]
        record["answer"] = patch["answer"]
        record["note"] = patch["note"]
        record["sources"] = [audit_source(item) for item in patch["cells"]]
    write(source_audit_path, source_audit)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": sorted(PATCHES),
        "answer_changes": {
            str(qid): {"from": source_by_id[qid]["answer"], "to": PATCHES[qid]["answer"]}
            for qid in sorted(PATCHES)
        },
        "changed_fields": changed_fields,
        "correct_source_tables_added": [
            table for qid in sorted(PATCHES) for table in PATCHES[qid]["tables"]
        ],
        "old_retrieval_labels_retained": True,
        "physical_table_files_copied": sorted(copied_tables),
        "source_submission_sha256": digest(SOURCE / "submission.json"),
        "claim_limit": "Three source-verified scope repairs; leaderboard effect is unmeasured.",
    }
    write(BUILDING / "v219_internal_scope_batch3_audit.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = digest(OUTPUT / "submission.json")
    write(OUTPUT / "v219_internal_scope_batch3_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
