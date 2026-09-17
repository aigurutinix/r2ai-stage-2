"""Build an isolated v207 component repairing q971 table lineage.

q971's answer and raw operands are correct, but v206 cites five summary cost-of-
brokerage tables.  The question explicitly asks for total brokerage commission;
each report has a nearby child table listing commission components whose exact
additive ``Cộng`` row equals the existing operand.  This builder replaces only
q971's relevant-table/source coordinates.  It is not an upload artifact and is
held for a later multi-question v207 bundle.
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
SOURCE = ROOT / "sub_top123_candidate_v206_semantic_batch11"
OUTPUT = ROOT / "sub_top123_candidate_v207_q971_semantic_child_table_component"
QID = 971

CORRECTIONS = {
    "2019": {
        "old_table": "KHG_financial_statements_2019_separate|693",
        "table": "KHG_financial_statements_2019_separate|697",
        "old_row": "2", "row": "7", "column": "1",
        "source_csv": "table_28_line697.csv", "raw": "89.144.410.598",
    },
    "2020": {
        "old_table": "KHG_financial_statements_2020_separate|667",
        "table": "KHG_financial_statements_2020_separate|671",
        "old_row": "2", "row": "6", "column": "1",
        "source_csv": "table_27_line671.csv", "raw": "159.868.248.240",
    },
    "2021": {
        "old_table": "KHG_financial_statements_2021_separate|910",
        "table": "KHG_financial_statements_2021_separate|914",
        "old_row": "2", "row": "6", "column": "1",
        "source_csv": "table_33_line914.csv", "raw": "310.228.563.963",
    },
    "2022": {
        "old_table": "KHG_financial_statements_2022_separate|876",
        "table": "KHG_financial_statements_2022_separate|880",
        "old_row": "2", "row": "5", "column": "1",
        "source_csv": "table_35_line880.csv", "raw": "560.911.129.505",
    },
    "2023": {
        "old_table": "KHG_financial_statements_2023_separate|916",
        "table": "KHG_financial_statements_2023_separate|931",
        "old_row": "2", "row": "5", "column": "1",
        "source_csv": "table_39_line931.csv", "raw": "42.745.665.072",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != len(CORRECTIONS):
        raise AssertionError(f"unexpected q971 source-row count: {len(rows)}")
    changed: list[dict[str, str]] = []
    for row in rows:
        correction = CORRECTIONS[str(row.get("year"))]
        expected = {
            "source_table": correction["old_table"],
            "row_idx": correction["old_row"],
            "col_idx": correction["column"],
            "raw": correction["raw"],
        }
        for key, value in expected.items():
            if str(row.get(key)) != value:
                raise AssertionError(f"q971 manifest mismatch for {row.get('year')} {key}: {row}")
        row.update({
            "source_table": correction["table"],
            "source_csv": correction["source_csv"],
            "row_idx": correction["row"],
            "col_idx": correction["column"],
        })
        changed.append(row.copy())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return changed


def patch_source_audit(path: Path) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    record = next(item for item in records if int(item["id"]) == QID)
    sources = list(record.get("sources", []))
    if len(sources) != len(CORRECTIONS):
        raise AssertionError("unexpected q971 source-audit arity")
    for year, source in zip(sorted(CORRECTIONS), sources):
        correction = CORRECTIONS[year]
        if source.get("table_ref") != correction["old_table"] or source.get("raw") != correction["raw"]:
            raise AssertionError(f"q971 source-audit mismatch for {year}: {source}")
        source.update({
            "table_ref": correction["table"],
            "csv": correction["source_csv"],
            "row": int(correction["row"]),
            "column": int(correction["column"]),
            "label": "Tổng chi phí hoa hồng môi giới bất động sản",
            "source_row_labels": ["Cộng"],
        })
    write_json(path, records)


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(item["id"]): item for item in source_rows}
    row = next(item for item in rows if int(item["id"]) == QID)
    old_tables = [CORRECTIONS[year]["old_table"] for year in sorted(CORRECTIONS)]
    new_tables = [CORRECTIONS[year]["table"] for year in sorted(CORRECTIONS)]
    if row.get("relevant_tables") != old_tables or float(row.get("answer")) != 2022.0:
        raise AssertionError("unexpected q971 source submission state")
    row["relevant_tables"] = new_tables
    write_json(OUTPUT / "submission.json", rows)

    corrected_sources = patch_manifest(OUTPUT / "data" / "q971_source_cells.csv")
    patch_source_audit(OUTPUT / "source_audit.json")

    changed_ids = [
        int(item["id"])
        for item in rows
        if item != source_by_id[int(item["id"])]
    ]
    if changed_ids != [QID]:
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    changed_fields = {
        key for key in set(row) | set(source_by_id[QID])
        if row.get(key) != source_by_id[QID].get(key)
    }
    if changed_fields != {"relevant_tables"}:
        raise AssertionError(f"unexpected q971 changed fields: {sorted(changed_fields)}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_unchanged": 2022.0,
        "old_relevant_tables": old_tables,
        "new_relevant_tables": new_tables,
        "source_corrections": corrected_sources,
        "invariants": {
            "only_q971_submission_row_changed": True,
            "only_relevant_tables_changed_in_submission_json": True,
            "answer_unchanged": True,
            "query_unchanged": True,
            "relevant_docs_unchanged": True,
            "evidence_path_unchanged": True,
            "exact_raw_operands_unchanged": True,
            "source_binding_rows_changed": 5,
        },
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Table-lineage component only; hold for a multi-question bundle and full release gate.",
    }
    write_json(OUTPUT / "q971_semantic_child_table_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
