"""Build v224 with the source-verified parent-company scope repair for q98."""

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
SOURCE = ROOT / "sub_top123_candidate_v223_nab_performing_loan_unit_batch7"
OUTPUT = ROOT / "sub_top123_candidate_v224_hut_parent_inventory_scope_batch8"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")
QID = 98
OLD_ANSWER = 146.47
NEW_ANSWER = 3177.37
TABLE = "HUT_financial_statements_2024_separate|328"
TABLE_CSV = "table_0_line328.csv"
RAW = "3.177.372.538.020"
ROW_IDX = 15
COL_IDX = 4


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    physical = ROOT / "build" / "tables" / TABLE.split("|", 1)[0] / TABLE_CSV
    with physical.open("r", encoding="utf-8-sig", newline="") as handle:
        physical_rows = list(csv.reader(handle))
    observed = physical_rows[ROW_IDX + 1][COL_IDX]
    if observed != RAW:
        raise AssertionError(f"physical q98 cell changed: {observed!r} != {RAW!r}")

    source_rows = load(SOURCE / "submission.json")
    rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    source_by_id = {int(row["id"]): row for row in source_rows}
    row = {int(item["id"]): item for item in rows}[QID]
    if row["answer"] != OLD_ANSWER:
        raise AssertionError(f"q{QID} old answer changed: {row['answer']!r}")
    row["answer"] = NEW_ANSWER
    row["relevant_docs"] = [TABLE.split("|", 1)[0]]
    row["relevant_tables"] = [TABLE]

    changed = [int(item["id"]) for item in rows if item != source_by_id[int(item["id"])] ]
    if changed != [QID]:
        raise AssertionError(f"unexpected changed IDs: {changed}")

    shutil.copytree(SOURCE, BUILDING)
    write(BUILDING / "submission.json", rows)
    shutil.copy2(physical, BUILDING / "data" / TABLE_CSV)

    fields = [
        "ticker", "year", "metric_key", "raw", "typed_factor", "scale",
        "source_table", "source_csv", "row_idx", "col_idx",
    ]
    evidence_row = {
        "ticker": "HUT", "year": "2024", "metric_key": "cdkt:140",
        "raw": RAW, "typed_factor": "1.0", "scale": "1.0",
        "source_table": TABLE, "source_csv": TABLE_CSV,
        "row_idx": str(ROW_IDX), "col_idx": str(COL_IDX),
    }
    with (BUILDING / "data" / "q98_source_cells.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(evidence_row)

    audits = load(SOURCE / "source_audit.json")
    audit_rows = json.loads(json.dumps(audits, ensure_ascii=False))
    matches = [item for item in audit_rows if int(item.get("id", -1)) == QID]
    if len(matches) != 1:
        raise AssertionError(f"expected one q{QID} source-audit row, got {len(matches)}")
    audit = matches[0]
    audit["old_answer"] = OLD_ANSWER
    audit["answer"] = NEW_ANSWER
    audit["note"] = (
        "HUT parent-company closing inventory (balance-sheet code 140), VND; "
        "the previous 146.47 billion value came from the consolidated report."
    )
    audit["sources"] = [{
        "table_ref": TABLE,
        "csv": TABLE_CSV,
        "row": ROW_IDX,
        "column": COL_IDX,
        "metric": "cdkt:140",
        "label": "Parent-company closing inventory, VND",
        "source_row_labels": ["IV.", "Hàng tồn kho"],
        "scale": 1.0,
        "typed_factor": 1.0,
        "raw": RAW,
    }]
    write(BUILDING / "source_audit.json", audit_rows)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_changes": {str(QID): {"from": OLD_ANSWER, "to": NEW_ANSWER}},
        "scope_repair": {
            "question_scope": "HUT parent company",
            "old_source": "HUT_financial_statements_2024_consolidated|325",
            "old_consolidated_inventory_vnd": 146_469_679_444,
            "new_source": TABLE,
            "parent_inventory_vnd": 3_177_372_538_020,
            "parent_inventory_billion_vnd": 3177.37253802,
        },
        "source_submission_sha256": digest(SOURCE / "submission.json"),
        "claim_limit": "One source-verified report-scope repair; leaderboard effect is unmeasured.",
    }
    write(BUILDING / "v224_hut_parent_inventory_scope_batch8_audit.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = digest(OUTPUT / "submission.json")
    write(OUTPUT / "v224_hut_parent_inventory_scope_batch8_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
