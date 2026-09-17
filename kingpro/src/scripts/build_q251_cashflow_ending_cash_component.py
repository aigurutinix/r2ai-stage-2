"""Build an isolated q251 table-lineage repair over the locked v207 candidate.

The answer is unchanged.  The current balance-sheet cash row has the correct
amount, but the cash-flow statement contains the exact question phrase
``Tiền và tương đương tiền cuối năm`` with the same physical value.  This
component changes only q251's table/source binding and fails closed on every
other submission field.
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
SOURCE = ROOT / "sub_top123_candidate_v207_semantic_batch6_final"
OUTPUT = ROOT / "sub_top123_candidate_v208_q251_cashflow_component"
QID = 251

OLD_TABLE = "PRT_financial_statements_2019_separate|187"
NEW_TABLE = "PRT_financial_statements_2019_separate|289"
OLD_SOURCE_CSV = "table_4_line187.csv"
NEW_SOURCE_CSV = "table_8_line289.csv"
RAW = "38.738.403.096"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def patch_manifest(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise AssertionError(f"unexpected q251 source-row count: {len(rows)}")
    row = rows[0]
    expected = {
        "ticker": "PRT",
        "year": "2019",
        "metric_key": "cdkt:110",
        "raw": RAW,
        "source_table": OLD_TABLE,
        "source_csv": OLD_SOURCE_CSV,
        "row_idx": "3",
        "col_idx": "3",
    }
    for key, value in expected.items():
        if str(row.get(key)) != value:
            raise AssertionError(f"q251 manifest mismatch for {key}: {row}")
    row.update(
        {
            "metric_key": "lctt:70",
            "source_table": NEW_TABLE,
            "source_csv": NEW_SOURCE_CSV,
            "row_idx": "10",
            "col_idx": "3",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    return row


def patch_source_audit(path: Path) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    matches = [record for record in records if int(record["id"]) == QID]
    if len(matches) != 1:
        raise AssertionError("q251 source-audit record must be unique")
    record = matches[0]
    sources = record.get("sources", [])
    if len(sources) != 1:
        raise AssertionError("unexpected q251 source-audit arity")
    source = sources[0]
    if (
        source.get("table_ref") != OLD_TABLE
        or source.get("csv") != OLD_SOURCE_CSV
        or source.get("raw") != RAW
        or int(source.get("row")) != 3
        or int(source.get("column")) != 3
    ):
        raise AssertionError(f"unexpected q251 source-audit state: {source}")
    source.update(
        {
            "table_ref": NEW_TABLE,
            "csv": NEW_SOURCE_CSV,
            "row": 10,
            "column": 3,
            "metric": "lctt:70",
            "label": "Tiền và tương đương tiền cuối năm",
            "source_row_labels": ["Tiền và tương đương tiền cuối năm"],
        }
    )
    record["note"] = (
        "PRT parent cash and cash equivalents at end-2019, VND billion; "
        "bound to the exact cash-flow ending row"
    )
    write_json(path, records)


def verify_physical_source() -> None:
    path = (
        ROOT
        / "build"
        / "tables"
        / "PRT_financial_statements_2019_separate"
        / NEW_SOURCE_CSV
    )
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    # pandas row 10 follows the CSV column-name row at index zero.
    physical = rows[11]
    if physical[1] != "Tiền và tương đương tiền cuối năm" or physical[3] != RAW:
        raise AssertionError(f"unexpected q251 physical cash-flow row: {physical}")


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    verify_physical_source()
    shutil.copytree(SOURCE, OUTPUT)

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(item["id"]): item for item in source_rows}
    row = next(item for item in rows if int(item["id"]) == QID)
    before = source_by_id[QID]
    if row.get("relevant_tables") != [OLD_TABLE] or float(row.get("answer")) != 38.74:
        raise AssertionError("unexpected q251 source submission state")
    row["relevant_tables"] = [NEW_TABLE]
    write_json(OUTPUT / "submission.json", rows)

    corrected_source = patch_manifest(OUTPUT / "data" / "q251_source_cells.csv")
    patch_source_audit(OUTPUT / "source_audit.json")

    changed_ids = [
        int(item["id"])
        for item in rows
        if item != source_by_id[int(item["id"])]
    ]
    if changed_ids != [QID]:
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    changed_fields = {
        key for key in set(row) | set(before) if row.get(key) != before.get(key)
    }
    if changed_fields != {"relevant_tables"}:
        raise AssertionError(f"unexpected q251 changed fields: {sorted(changed_fields)}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_unchanged": 38.74,
        "old_relevant_tables": [OLD_TABLE],
        "new_relevant_tables": [NEW_TABLE],
        "source_correction": corrected_source,
        "invariants": {
            "only_q251_submission_row_changed": True,
            "only_relevant_tables_changed_in_submission_json": True,
            "answer_unchanged": True,
            "query_unchanged": True,
            "relevant_docs_unchanged": True,
            "evidence_path_unchanged": True,
            "exact_raw_operand_unchanged": True,
            "physical_source_row_verified": True,
        },
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Table-lineage component only; hold until the v208 batch gate passes.",
    }
    write_json(OUTPUT / "q251_cashflow_table_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
