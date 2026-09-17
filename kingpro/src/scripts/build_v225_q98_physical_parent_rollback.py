"""Build v225 by reverting q98 to the physically parent-only HUT statement.

The HUT 2024 OCR containers are swapped: the directory ending in
``_consolidated`` contains the statement headed ``BẢNG CÂN ĐỐI KẾ TOÁN
RIÊNG``, while the directory ending in ``_separate`` contains the statement
headed ``... HỢP NHẤT``.  v224 trusted the directory suffix and therefore
replaced the correct parent inventory with the consolidated inventory.  This
builder restores the v223 q98 payload and leaves every other v224 row intact.
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
SOURCE = ROOT / "sub_top123_candidate_v224_hut_parent_inventory_scope_batch8"
TRUSTED_Q98 = ROOT / "sub_top123_candidate_v223_nab_performing_loan_unit_batch7"
OUTPUT = ROOT / "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")
QID = 98

PARENT_TABLE = "HUT_financial_statements_2024_consolidated|325"
PARENT_RAW = "146.469.679.444"
PARENT_ANSWER = 146.47
CONSOLIDATED_TABLE = "HUT_financial_statements_2024_separate|328"
CONSOLIDATED_RAW = "3.177.372.538.020"


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def by_id(rows: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    return {int(row["id"]): row for row in rows}


def main() -> None:
    if not SOURCE.is_dir() or not TRUSTED_Q98.is_dir():
        raise FileNotFoundError("v224 source or trusted v223 rollback source is missing")
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    source_rows = load(SOURCE / "submission.json")
    trusted_rows = load(TRUSTED_Q98 / "submission.json")
    if not isinstance(source_rows, list) or not isinstance(trusted_rows, list):
        raise AssertionError("submission payload must be a list")
    source_map = by_id(source_rows)
    trusted_map = by_id(trusted_rows)
    if source_map[QID].get("answer") != 3177.37:
        raise AssertionError("unexpected v224 q98 answer")
    if trusted_map[QID].get("answer") != PARENT_ANSWER:
        raise AssertionError("unexpected trusted q98 answer")

    # Prove the correction from the physical OCR masthead and exact table cell,
    # not from the misleading document-directory suffix.
    parent_text = next(
        (ROOT / "data" / "financial_statements").rglob(
            "HUT_financial_statements_2024_consolidated_extracted.txt"
        )
    ).read_text(encoding="utf-8", errors="replace")
    consolidated_text = next(
        (ROOT / "data" / "financial_statements").rglob(
            "HUT_financial_statements_2024_separate_extracted.txt"
        )
    ).read_text(encoding="utf-8", errors="replace")
    if "BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG" not in parent_text:
        raise AssertionError("parent physical masthead is missing")
    if "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT" not in consolidated_text:
        raise AssertionError("consolidated physical masthead is missing")
    if PARENT_RAW not in parent_text or CONSOLIDATED_RAW not in consolidated_text:
        raise AssertionError("expected HUT inventory source cell is missing")

    rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    rows_map = by_id(rows)
    rows_map[QID].clear()
    rows_map[QID].update(
        json.loads(json.dumps(trusted_map[QID], ensure_ascii=False))
    )
    changed = [
        int(row["id"])
        for row in rows
        if row != source_map[int(row["id"])]
    ]
    if changed != [QID]:
        raise AssertionError(f"unexpected changed IDs: {changed}")

    shutil.copytree(SOURCE, BUILDING)
    write(BUILDING / "submission.json", rows)
    shutil.copy2(
        TRUSTED_Q98 / "data" / "q98_source_cells.csv",
        BUILDING / "data" / "q98_source_cells.csv",
    )

    source_audit = load(SOURCE / "source_audit.json")
    trusted_audit = load(TRUSTED_Q98 / "source_audit.json")
    if not isinstance(source_audit, list) or not isinstance(trusted_audit, list):
        raise AssertionError("source audit payload must be a list")
    audit_rows = json.loads(json.dumps(source_audit, ensure_ascii=False))
    audit_map = by_id(audit_rows)
    trusted_audit_map = by_id(trusted_audit)
    audit_map[QID].clear()
    audit_map[QID].update(
        json.loads(json.dumps(trusted_audit_map[QID], ensure_ascii=False))
    )
    audit_map[QID]["old_answer"] = 3177.37
    audit_map[QID]["note"] = (
        "Physical-masthead rollback: parent inventory comes from the OCR "
        "container named *_consolidated whose statement heading is RIÊNG; "
        "the 3,177.37b source is physically HỢP NHẤT."
    )
    write(BUILDING / "source_audit.json", audit_rows)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_changes": {"98": {"from": 3177.37, "to": PARENT_ANSWER}},
        "physical_scope_proof": {
            "question_scope": "công ty mẹ / separate",
            "selected_table": PARENT_TABLE,
            "selected_masthead": "BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG",
            "selected_raw_vnd": PARENT_RAW,
            "rejected_table": CONSOLIDATED_TABLE,
            "rejected_masthead": "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
            "rejected_raw_vnd": CONSOLIDATED_RAW,
        },
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "trusted_q98_submission_sha256": sha256(TRUSTED_Q98 / "submission.json"),
        "claim_limit": (
            "One physical-masthead scope rollback; leaderboard effect is unmeasured."
        ),
    }
    write(BUILDING / "v225_q98_physical_parent_rollback_batch9_audit.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = sha256(OUTPUT / "submission.json")
    write(OUTPUT / "v225_q98_physical_parent_rollback_batch9_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
