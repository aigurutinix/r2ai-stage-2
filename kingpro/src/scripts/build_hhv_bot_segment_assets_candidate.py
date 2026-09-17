"""Build v197: one source-proven q1007 BOT-segment coordinate repair.

The old 2021/2022 operands read the grand total of a geographic segment table
and mislabeled it as BOT assets.  Both reports contain an activity-segment
table whose first numeric column is explicitly ``Dự án BOT``.
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
SOURCE = ROOT / "sub_top123_candidate_v192_data_derived_table_order"
OUTPUT = ROOT / "sub_top123_candidate_v197_hhv_bot_segment_assets"
QID = 1007

REPAIRS = {
    2021: {
        "raw": "32.355.512.700.711",
        "source_table": "HHV_financial_statements_2021_consolidated|2563",
        "source_csv": "table_67_line2563.csv",
        "row_idx": "1",
        "col_idx": "1",
    },
    2022: {
        "raw": "33.657.835.517.377",
        "source_table": "HHV_financial_statements_2022_consolidated|2450",
        "source_csv": "table_76_line2450.csv",
        "row_idx": "1",
        "col_idx": "1",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _physical_cell(year: int, repair: dict[str, str]) -> str:
    table_dir = ROOT / "build" / "tables" / f"HHV_financial_statements_{year}_consolidated"
    table_path = table_dir / repair["source_csv"]
    with table_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))
    # CSV row 0 is the extracted header; manifest row_idx is a pandas index.
    return rows[int(repair["row_idx"]) + 1][int(repair["col_idx"])]


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    source_submission = SOURCE / "submission.json"
    output_submission = OUTPUT / "submission.json"
    old_rows = json.loads(source_submission.read_text(encoding="utf-8"))
    rows = json.loads(output_submission.read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in old_rows}
    row = next(item for item in rows if int(item["id"]) == QID)
    old_answer = row["answer"]
    old_tables = list(row["relevant_tables"])

    ratios = {
        2021: 32_355_512_700_711 / 33_963_489_243_390,
        2022: 33_657_835_517_377 / 35_653_232_484_507,
        2024: 35_317_671_994_443 / 38_906_360_732_239,
        2025: 35_890_505_367_786 / 40_752_075_682_457,
    }
    corrected_answer = round(sum(ratios.values()) / len(ratios) * 100, 2)
    if corrected_answer != 92.13:
        raise AssertionError(f"unexpected q1007 result {corrected_answer}")
    row["answer"] = corrected_answer
    replacements = {
        "HHV_financial_statements_2021_consolidated|2588": REPAIRS[2021]["source_table"],
        "HHV_financial_statements_2022_consolidated|2471": REPAIRS[2022]["source_table"],
    }
    row["relevant_tables"] = [replacements.get(value, value) for value in old_tables]

    manifest_path = OUTPUT / "data" / "q1007_source_cells.csv"
    with manifest_path.open(encoding="utf-8", newline="") as stream:
        manifest = list(csv.DictReader(stream))
        fieldnames = list(manifest[0])
    repaired_rows = 0
    for item in manifest:
        year = int(item["year"])
        if item["metric_key"] != "note:bot_segment_assets" or year not in REPAIRS:
            continue
        repair = REPAIRS[year]
        if _physical_cell(year, repair) != repair["raw"]:
            raise AssertionError(f"q1007 physical source mismatch for {year}")
        item.update(repair)
        repaired_rows += 1
    if repaired_rows != 2:
        raise AssertionError(f"expected two repaired manifest rows, got {repaired_rows}")
    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest)

    output_submission.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    allowed = {"answer", "relevant_tables"}
    for candidate_row in rows:
        qid = int(candidate_row["id"])
        old = old_by_id[qid]
        if qid != QID and candidate_row != old:
            raise AssertionError(f"q{qid} changed unexpectedly")
    for key in set(old_by_id[QID]) | set(row):
        if key not in allowed and old_by_id[QID].get(key) != row.get(key):
            raise AssertionError(f"q1007 field changed unexpectedly: {key}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids": [QID],
        "old_answer": old_answer,
        "new_answer": corrected_answer,
        "old_relevant_tables": old_tables,
        "new_relevant_tables": row["relevant_tables"],
        "yearly_ratios_percent": {year: value * 100 for year, value in ratios.items()},
        "repairs": REPAIRS,
        "invariants": {
            "only_q1007_submission_row_changed": True,
            "only_answer_and_relevant_tables_changed": True,
            "query_structure_unchanged": True,
            "two_manifest_coordinates_repaired": True,
            "leaderboard_feedback_not_used": True,
            "source_derived": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": "Source-proven candidate; BTC score is unknown until this exact artifact is evaluated.",
    }
    (OUTPUT / "q1007_hhv_bot_segment_assets_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
