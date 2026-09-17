"""Build v223 with the source-verified unit repair for q530."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v222_total_fixed_assets_scope_batch6"
OUTPUT = ROOT / "sub_top123_candidate_v223_nab_performing_loan_unit_batch7"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")
QID = 530
OLD_ANSWER = 0.19
NEW_ANSWER = 190.76
OLD_EXPRESSION = ")[1] / 1e9, 2)"
NEW_EXPRESSION = ")[1] / 1e6, 2)"


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

    source_rows = load(SOURCE / "submission.json")
    rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    source_by_id = {int(row["id"]): row for row in source_rows}
    row = {int(item["id"]): item for item in rows}[QID]

    if row["answer"] != OLD_ANSWER:
        raise AssertionError(f"q{QID} old answer changed: {row['answer']!r}")
    if row["pandas_query"].count(OLD_EXPRESSION) != 1:
        raise AssertionError("q530 expected exactly one old conversion expression")
    row["answer"] = NEW_ANSWER
    row["pandas_query"] = row["pandas_query"].replace(OLD_EXPRESSION, NEW_EXPRESSION)

    changed = [int(item["id"]) for item in rows if item != source_by_id[int(item["id"])] ]
    if changed != [QID]:
        raise AssertionError(f"unexpected changed IDs: {changed}")

    shutil.copytree(SOURCE, BUILDING)
    write(BUILDING / "submission.json", rows)

    audits = load(SOURCE / "source_audit.json")
    audit_rows = json.loads(json.dumps(audits, ensure_ascii=False))
    matches = [item for item in audit_rows if int(item.get("id", -1)) == QID]
    if len(matches) != 1:
        raise AssertionError(f"expected one q{QID} source-audit row, got {len(matches)}")
    audit = matches[0]
    if audit.get("answer") != OLD_ANSWER:
        raise AssertionError(f"q{QID} source-audit answer changed: {audit.get('answer')!r}")
    audit["old_answer"] = OLD_ANSWER
    audit["answer"] = NEW_ANSWER
    audit["note"] = (
        "NAB ending performing customer loans in 2025, selected because 2025 has the "
        "maximum ending construction in progress. The physical loan table is in million "
        "VND, so 190,759,675 million VND / 1,000,000 = 190.76 trillion VND "
        "(Vietnamese: nghìn tỷ đồng)."
    )
    write(BUILDING / "source_audit.json", audit_rows)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_changes": {str(QID): {"from": OLD_ANSWER, "to": NEW_ANSWER}},
        "question_unit": "nghìn tỷ đồng (trillion VND)",
        "physical_source_unit": "triệu đồng (million VND)",
        "selector": {
            "maximum_construction_in_progress_year": 2025,
            "construction_in_progress_million_vnd": 602_113,
        },
        "selected_value": {
            "performing_customer_loans_million_vnd": 190_759_675,
            "performing_customer_loans_trillion_vnd": 190.759675,
            "rounded_answer": NEW_ANSWER,
        },
        "query_conversion": {
            "from": "raw million-VND value / 1e9",
            "to": "raw million-VND value / 1e6",
        },
        "physical_tables": [
            "NAB_financial_statements_2025_consolidated|1748",
            "NAB_financial_statements_2025_consolidated|1479",
        ],
        "source_submission_sha256": digest(SOURCE / "submission.json"),
        "claim_limit": "One source-verified unit repair; leaderboard effect is unmeasured.",
    }
    write(BUILDING / "v223_nab_performing_loan_unit_batch7_audit.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = digest(OUTPUT / "submission.json")
    write(OUTPUT / "v223_nab_performing_loan_unit_batch7_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
