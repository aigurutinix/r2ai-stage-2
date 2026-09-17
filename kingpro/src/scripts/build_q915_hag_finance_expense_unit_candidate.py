"""Build v202: restore HAG 2016's thousand-VND unit for q915.

q915 sums parent-company finance expense for MPC, SAB, and HAG.  The MPC and
SAB statements are in VND, while HAG's exact 2016 separate statement labels
the income statement ``Ngàn VND``.  The compact manifest retained HAG's
printed token but lost that factor, understating the total by 1,432.43 billion
VND.  This builder layers the single source-confirmed correction on v201.
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
SOURCE = ROOT / "sub_top123_candidate_v201_cumulative_three_unit_fixes"
OUTPUT = ROOT / "sub_top123_candidate_v202_q915_hag_finance_expense_unit"
QID = 915
HAG_TABLE = "HAG_financial_statements_2016_separate|281"
OLD_ANSWER = 301.42
NEW_ANSWER = 1733.85


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


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
    if float(row["answer"]) != OLD_ANSWER:
        raise AssertionError(f"unexpected q915 old answer: {row['answer']}")

    evidence_path = OUTPUT / "data" / "q915_source_cells.csv"
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as handle:
        evidence_rows = list(csv.DictReader(handle))
        fieldnames = list(evidence_rows[0])
    matches = [item for item in evidence_rows if item["source_table"] == HAG_TABLE]
    if len(matches) != 1:
        raise AssertionError(f"expected one HAG source row, got {len(matches)}")
    hag = matches[0]
    if hag != {
        "ticker": "HAG",
        "year": "2016",
        "metric_key": "kqkd:22",
        "raw": "(1.433.862.140) (1.299.333.937)",
        "typed_factor": "1.0",
        "scale": "1.0",
        "source_table": HAG_TABLE,
        "source_csv": "table_3_line281.csv",
        "row_idx": "7",
        "col_idx": "3",
    }:
        raise AssertionError(f"unexpected HAG evidence row: {hag}")
    hag["scale"] = "1000.0"
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(evidence_rows)

    corrected = round((279_209_466_041 + 20_775_323_891 + 1_433_862_140_000) / 1e9, 2)
    if corrected != NEW_ANSWER:
        raise AssertionError(f"unexpected corrected answer: {corrected}")
    row["answer"] = corrected
    write_json(output_submission, rows)

    source_audit_path = OUTPUT / "source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    audit_matches = [item for item in source_audit if int(item["id"]) == QID]
    if len(audit_matches) != 1:
        raise AssertionError("expected one q915 source-audit row")
    audit = audit_matches[0]
    if float(audit["answer"]) != OLD_ANSWER:
        raise AssertionError("unexpected q915 source-audit answer")
    audit["answer"] = corrected
    audit["note"] = (
        "total 2016 parent finance expense for MPC, SAB and HAG, VND billion; "
        "HAG separate statement is explicitly stated in thousand VND"
    )
    audit_hag = [item for item in audit["sources"] if item["table_ref"] == HAG_TABLE]
    if len(audit_hag) != 1 or float(audit_hag[0]["scale"]) != 1.0:
        raise AssertionError("unexpected q915 HAG source-audit scale")
    audit_hag[0]["scale"] = 1000.0
    write_json(source_audit_path, source_audit)

    changed_ids = [
        int(candidate["id"])
        for candidate in rows
        if candidate != old_by_id[int(candidate["id"])]
    ]
    if changed_ids != [QID]:
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    for key in set(row) | set(old_by_id[QID]):
        if key != "answer" and row.get(key) != old_by_id[QID].get(key):
            raise AssertionError(f"q915 non-answer field changed: {key}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "old_answer": OLD_ANSWER,
        "new_answer": NEW_ANSWER,
        "hag_printed_thousand_vnd": 1_433_862_140,
        "hag_scale": 1000.0,
        "source_unit_evidence": [
            "data/financial_statements/HAG/2016/HAG_financial_statements_2016_separate/"
            "HAG_financial_statements_2016_separate_extracted.txt:234",
            "build/tables/HAG_financial_statements_2016_separate/table_3_line281.csv",
        ],
        "invariants": {
            "only_q915_answer_changed_relative_to_v201": True,
            "pandas_query_unchanged": True,
            "retrieval_fields_unchanged": True,
            "only_q915_hag_evidence_scale_changed": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": "Source-confirmed unit correction; leaderboard result unknown.",
    }
    write_json(OUTPUT / "q915_hag_finance_expense_unit_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
