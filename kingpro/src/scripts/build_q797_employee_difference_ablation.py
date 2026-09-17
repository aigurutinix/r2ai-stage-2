"""Build a one-question q797 magnitude ablation from locked v203.

The audited 2019 disclosures report ACV employee expense below Vietjet's.
Question 797 nevertheless asks how much ACV's amount is greater.  The ordered
subtraction introduced by the unmeasured v169 experiment therefore returns a
negative answer to a ``bao nhiêu`` comparison.  This ablation restores the
absolute-difference convention that was used by measured v106 while preserving
both exact source cells and every retrieval field.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v203_q24_q607_double_unit"
OUTPUT = ROOT / "sub_top123_candidate_v204_q797_employee_difference"
QUESTION_ID = 797
OLD_ANSWER = -1706.2
NEW_ANSWER = 1706.2
OLD_EXPRESSION = "result = round((v0 - v1) / 1e9, 2)"
NEW_EXPRESSION = "result = round(abs(v0 - v1) / 1e9, 2)"


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
    by_id = {int(row["id"]): row for row in rows}

    row = by_id[QUESTION_ID]
    if float(row["answer"]) != OLD_ANSWER:
        raise AssertionError(f"q797 unexpected old answer: {row['answer']}")
    query = str(row["pandas_query"])
    if query.count(OLD_EXPRESSION) != 1:
        raise AssertionError("q797 expected exactly one ordered result expression")
    row["answer"] = NEW_ANSWER
    row["pandas_query"] = query.replace(OLD_EXPRESSION, NEW_EXPRESSION)
    write_json(output_submission, rows)

    changed_ids = [
        int(candidate["id"])
        for candidate in rows
        if candidate != old_by_id[int(candidate["id"])]
    ]
    if changed_ids != [QUESTION_ID]:
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    changed_fields = sorted(
        key
        for key in set(row) | set(old_by_id[QUESTION_ID])
        if row.get(key) != old_by_id[QUESTION_ID].get(key)
    )
    if changed_fields != ["answer", "pandas_query"]:
        raise AssertionError(f"q797 unexpected changed fields: {changed_fields}")

    source_audit_path = OUTPUT / "source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    audit_by_id = {int(item["id"]): item for item in source_audit}
    audit = audit_by_id[QUESTION_ID]
    if float(audit["answer"]) != OLD_ANSWER:
        raise AssertionError(f"q797 unexpected source-audit answer: {audit['answer']}")
    audit["old_answer"] = OLD_ANSWER
    audit["answer"] = NEW_ANSWER
    audit["note"] = (
        "absolute difference between audited ACV and Vietjet employee/labor "
        "expense in 2019, VND billion; restores the measured-v106 magnitude "
        "interpretation because the question asks how much rather than asking "
        "for a signed ACV-minus-Vietjet balance"
    )
    write_json(source_audit_path, source_audit)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QUESTION_ID],
        "fix": {
            "question_id": QUESTION_ID,
            "old_answer": OLD_ANSWER,
            "new_answer": NEW_ANSWER,
            "old_expression": OLD_EXPRESSION,
            "new_expression": NEW_EXPRESSION,
            "source_tables": list(row["relevant_tables"]),
            "history": {
                "positive_magnitude_used_in_measured_v106": True,
                "negative_direction_variant_v169_v170_submitted": False,
            },
        },
        "invariants": {
            "only_q797_changed_relative_to_v203": True,
            "changed_submission_fields": changed_fields,
            "questions_unchanged": True,
            "retrieval_fields_unchanged": True,
            "evidence_and_source_cells_unchanged": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": "One-question semantic ablation; leaderboard result unknown.",
    }
    write_json(OUTPUT / "q797_employee_difference_ablation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
