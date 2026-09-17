"""Build v210 with the source-proven q769 asset-class repair.

The submitted v209 compares VSC intangible land-use rights with ACV investment
property land-use rights.  The question asks the same remaining-value concept
for both companies.  ACV's matching intangible-PPE table reports
36,648,986,835 VND; against VSC's 5,355,027,273 VND the difference is 31.29
billion VND.  v209 is copied and remains untouched.
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
SOURCE = ROOT / "sub_top123_candidate_v209_q15_board_role_r2"
OUTPUT = ROOT / "sub_top123_candidate_v210_q769_asset_class"
QID = 769
OLD_ANSWER = 26.89
NEW_ANSWER = 31.29
VSC_REF = "VSC_financial_statements_2015_consolidated|846"
OLD_ACV_REF = "ACV_financial_statements_2015_consolidated|1641"
ACV_REF = "ACV_financial_statements_2015_consolidated|1571"
ACV_CSV_NAME = "ACV_financial_statements_2015_consolidated_1571.csv"
ACV_SOURCE_CSV = (
    ROOT
    / "build"
    / "tables"
    / "ACV_financial_statements_2015_consolidated"
    / "table_22_line1571.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_manifest(path: Path) -> None:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2:
        raise AssertionError("q769 manifest must contain exactly two operands")
    acv = next(row for row in rows if row["ticker"] == "ACV")
    if acv["source_table"] != OLD_ACV_REF or acv["raw"] != "32.243.749.055":
        raise AssertionError("unexpected old q769 ACV binding")
    acv.update(
        {
            "raw": "36.648.986.835",
            "source_table": ACV_REF,
            "source_csv": ACV_CSV_NAME,
            "row_idx": "12",
            "col_idx": "1",
        }
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def patch_source_audit(path: Path) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    record = next(item for item in records if int(item["id"]) == QID)
    if float(record["answer"]) != OLD_ANSWER:
        raise AssertionError("unexpected old q769 source audit")
    record["answer"] = NEW_ANSWER
    record["note"] = (
        "Like-for-like VSC/ACV intangible-PPE ending land-use-rights NBV difference, VND billion"
    )
    acv = next(item for item in record["sources"] if item["table_ref"] == OLD_ACV_REF)
    acv.update(
        {
            "table_ref": ACV_REF,
            "csv": ACV_CSV_NAME,
            "row": 12,
            "column": 1,
            "label": "Intangible-PPE land-use-rights ending NBV",
            "source_row_labels": ["GIÁ TRỊ CÒN LẠI", "31/12/2015"],
            "raw": "36.648.986.835",
        }
    )
    write_json(path, records)


def build() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not SOURCE.is_dir() or not ACV_SOURCE_CSV.is_file():
        raise FileNotFoundError("v209 source or ACV intangible source table is missing")

    frame_lines = ACV_SOURCE_CSV.read_text(encoding="utf-8-sig").splitlines()
    if len(frame_lines) <= 13 or "36.648.986.835" not in frame_lines[13]:
        raise AssertionError("ACV q769 physical source token moved")

    shutil.copytree(SOURCE, OUTPUT)
    shutil.copy2(ACV_SOURCE_CSV, OUTPUT / "data" / ACV_CSV_NAME)

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(item["id"]): item for item in source_rows}
    row = next(item for item in rows if int(item["id"]) == QID)
    if float(row["answer"]) != OLD_ANSWER or row["relevant_tables"] != [VSC_REF, OLD_ACV_REF]:
        raise AssertionError("unexpected submitted v209 q769 state")
    row["answer"] = NEW_ANSWER
    row["relevant_tables"] = [VSC_REF, ACV_REF]
    write_json(OUTPUT / "submission.json", rows)

    patch_manifest(OUTPUT / "data" / "q769_source_cells.csv")
    patch_source_audit(OUTPUT / "source_audit.json")

    changed_ids = [
        int(item["id"])
        for item in rows
        if item != source_by_id[int(item["id"])]
    ]
    if changed_ids != [QID]:
        raise AssertionError(f"unexpected changed IDs: {changed_ids}")
    changed_fields = {
        key
        for key in set(row) | set(source_by_id[QID])
        if row.get(key) != source_by_id[QID].get(key)
    }
    if changed_fields != {"answer", "relevant_tables"}:
        raise AssertionError(f"unexpected q769 changed fields: {sorted(changed_fields)}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "old_answer": OLD_ANSWER,
        "new_answer": NEW_ANSWER,
        "old_semantics": "VSC intangible PPE versus ACV investment property",
        "new_semantics": "like-for-like intangible PPE land-use-rights ending NBV",
        "recomputation": {
            "vsc_vnd": 5_355_027_273,
            "acv_vnd": 36_648_986_835,
            "absolute_difference_billion": 31.293959562,
            "rounded_answer": NEW_ANSWER,
        },
        "source_tables": [VSC_REF, ACV_REF],
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Exact-source asset-class correction; public effect unknown until submitted.",
    }
    write_json(OUTPUT / "v210_q769_asset_class_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
