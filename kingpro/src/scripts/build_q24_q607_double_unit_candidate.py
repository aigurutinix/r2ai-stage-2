"""Build v203 with two exact-source double-conversion fixes.

Both q24 and q607 ask for answers in thousand VND and cite tables explicitly
labelled ``Ngàn VND``.  Their programs divide by 1e3 at the result, so the
compact manifest must first restore the printed values to VND with scale=1000.
The old scale=1 caused both answers to be divided by one thousand twice.
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
SOURCE = ROOT / "sub_top123_candidate_v202_q915_hag_finance_expense_unit"
OUTPUT = ROOT / "sub_top123_candidate_v203_q24_q607_double_unit"

FIXES = {
    24: {
        "old_answer": 2_083_992.73,
        "new_answer": 2_083_992_733.0,
        "tables": ["HNG_financial_statements_2017_separate|995"],
        "evidence": [
            "data/financial_statements/HNG/2017/HNG_financial_statements_2017_separate/"
            "HNG_financial_statements_2017_separate_extracted.txt:983"
        ],
    },
    607: {
        "old_answer": 6_143.2,
        "new_answer": 6_143_196.0,
        "tables": [
            "HAG_financial_statements_2023_consolidated|334",
            "HAG_financial_statements_2017_consolidated|250",
        ],
        "evidence": [
            "data/financial_statements/HAG/2023/HAG_financial_statements_2023_consolidated/"
            "HAG_financial_statements_2023_consolidated_extracted.txt:332",
            "data/financial_statements/HAG/2017/HAG_financial_statements_2017_consolidated/"
            "HAG_financial_statements_2017_consolidated_extracted.txt:248",
        ],
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def update_manifest(question_id: int, expected_tables: list[str]) -> None:
    path = OUTPUT / "data" / f"q{question_id}_source_cells.csv"
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    found = []
    for row in rows:
        if row.get("source_table") not in expected_tables:
            continue
        if float(row.get("scale", 1.0)) != 1.0:
            raise AssertionError(f"q{question_id} unexpected old scale: {row}")
        row["scale"] = "1000.0"
        found.append(str(row["source_table"]))
    if sorted(found) != sorted(expected_tables):
        raise AssertionError(
            f"q{question_id} expected {expected_tables}, found {found}"
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


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

    for question_id, fix in FIXES.items():
        row = by_id[question_id]
        if float(row["answer"]) != float(fix["old_answer"]):
            raise AssertionError(
                f"q{question_id} unexpected old answer: {row['answer']}"
            )
        update_manifest(question_id, list(fix["tables"]))
        row["answer"] = fix["new_answer"]
    write_json(output_submission, rows)

    source_audit_path = OUTPUT / "source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    audit_by_id = {int(row["id"]): row for row in source_audit}
    for question_id, fix in FIXES.items():
        audit = audit_by_id[question_id]
        if float(audit["answer"]) != float(fix["old_answer"]):
            raise AssertionError(f"q{question_id} unexpected audit answer")
        audit["answer"] = fix["new_answer"]
        audit["note"] = (
            str(audit.get("note", ""))
            + "; exact cited source and requested output are both thousand VND"
        ).lstrip("; ")
        found = []
        for source in audit.get("sources", []):
            if source.get("table_ref") not in fix["tables"]:
                continue
            if float(source.get("scale", 1.0)) != 1.0:
                raise AssertionError(
                    f"q{question_id} unexpected audit source scale: {source}"
                )
            source["scale"] = 1000.0
            found.append(str(source["table_ref"]))
        if sorted(found) != sorted(fix["tables"]):
            raise AssertionError(f"q{question_id} audit sources missing: {found}")
    write_json(source_audit_path, source_audit)

    changed_ids = [
        int(row["id"])
        for row in rows
        if row != old_by_id[int(row["id"])]
    ]
    if changed_ids != sorted(FIXES):
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    for question_id in FIXES:
        for key in set(by_id[question_id]) | set(old_by_id[question_id]):
            if key != "answer" and by_id[question_id].get(key) != old_by_id[
                question_id
            ].get(key):
                raise AssertionError(f"q{question_id} non-answer field changed: {key}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": sorted(FIXES),
        "fixes": {
            str(question_id): {
                "old_answer": fix["old_answer"],
                "new_answer": fix["new_answer"],
                "source_tables_scaled_to_vnd": fix["tables"],
                "source_unit_evidence": fix["evidence"],
            }
            for question_id, fix in FIXES.items()
        },
        "invariants": {
            "only_q24_q607_answers_changed_relative_to_v202": True,
            "pandas_queries_unchanged": True,
            "retrieval_fields_unchanged": True,
            "only_confirmed_manifest_scales_changed": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": "Exact-source unit corrections; leaderboard result unknown.",
    }
    write_json(OUTPUT / "q24_q607_double_unit_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
