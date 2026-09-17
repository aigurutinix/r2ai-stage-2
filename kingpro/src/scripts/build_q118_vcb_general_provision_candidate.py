"""Build v205: fix VCB 2015 customer-loan general provision for q118.

The legacy row asks for ``dự phòng chung`` but executes the following movement
table for ``dự phòng cụ thể`` and therefore returns 5,875,693.  The immediately
preceding general-provision movement table reports an ending balance of
2,688,909 million VND.  This builder changes only q118 and adds an exact source
binding; v203 remains untouched.
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
OUTPUT = ROOT / "sub_top123_candidate_v205_q118_vcb_general_provision"
QID = 118
OLD_ANSWER = 5_875_693.0
NEW_ANSWER = 2_688_909.0
TABLE_REF = "VCB_financial_statements_2015_separate|1418"
CSV_NAME = "VCB_financial_statements_2015_separate_1418.csv"
SOURCE_CSV = (
    ROOT
    / "build"
    / "tables"
    / "VCB_financial_statements_2015_separate"
    / "table_31_line1418.csv"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def corrected_query() -> str:
    return """_dfvals = list(dfs.values())
if len(_dfvals) >= 1:
    df1 = _dfvals[0]
# Exact source: movement in general provision for customer loans.
_r = df1[df1['0'].astype(str).str.contains('Số dư cuối kỳ', case=False, na=False, regex=False)]
_v = str(_r['1'].values[0]).replace('(','-').replace(')','').replace('.','').replace(',','.')
result = round(float(_v), 2)"""


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(SOURCE_CSV)
    shutil.copytree(SOURCE, OUTPUT)
    copied_csv = OUTPUT / "data" / CSV_NAME
    shutil.copy2(SOURCE_CSV, copied_csv)

    old_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in old_rows}
    row = next(item for item in rows if int(item["id"]) == QID)
    if float(row["answer"]) != OLD_ANSWER:
        raise AssertionError(f"unexpected q118 old answer: {row['answer']}")

    original_evidence = list(row.get("evidence", []))
    row["answer"] = NEW_ANSWER
    row["relevant_tables"] = [TABLE_REF]
    row["evidence"] = [
        {"variable": "df1", "csv_path": f"data/{CSV_NAME}"},
        *[
            {"variable": f"df{index}", "csv_path": item["csv_path"]}
            for index, item in enumerate(original_evidence, 2)
        ],
    ]
    row["pandas_query"] = corrected_query()
    write_json(OUTPUT / "submission.json", rows)

    source_audit_path = OUTPUT / "source_audit.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if any(int(item["id"]) == QID for item in source_audit):
        raise AssertionError("q118 unexpectedly already exists in source_audit")
    source_audit.append(
        {
            "id": QID,
            "old_answer": OLD_ANSWER,
            "answer": NEW_ANSWER,
            "note": "VCB parent 2015 ending general provision for customer loans, VND million",
            "sources": [
                {
                    "table_ref": TABLE_REF,
                    "csv": "table_31_line1418.csv",
                    "row": 4,
                    "column": 1,
                    "metric": "note:customer_loan_general_provision_ending",
                    "label": "Số dư cuối kỳ dự phòng chung cho vay khách hàng",
                    "source_row_labels": ["Số dư cuối kỳ"],
                    "scale": 1.0,
                    "typed_factor": 1.0,
                    "raw": "2.688.909",
                }
            ],
        }
    )
    source_audit.sort(key=lambda item: int(item["id"]))
    write_json(source_audit_path, source_audit)

    changed_ids = [
        int(candidate["id"])
        for candidate in rows
        if candidate != old_by_id[int(candidate["id"])]
    ]
    if changed_ids != [QID]:
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    allowed = {"answer", "relevant_tables", "evidence", "pandas_query"}
    changed_fields = {
        key
        for key in set(row) | set(old_by_id[QID])
        if row.get(key) != old_by_id[QID].get(key)
    }
    if changed_fields != allowed:
        raise AssertionError(f"unexpected q118 changed fields: {sorted(changed_fields)}")
    if copied_csv.read_bytes() != SOURCE_CSV.read_bytes():
        raise AssertionError("copied q118 source CSV differs from catalog source")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "old_answer": OLD_ANSWER,
        "new_answer": NEW_ANSWER,
        "old_semantics": "ending specific provision for customer loans",
        "new_semantics": "ending general provision for customer loans",
        "source_table": TABLE_REF,
        "source_raw": "2.688.909 million VND",
        "source_evidence": [
            "data/financial_statements/VCB/2015/VCB_financial_statements_2015_separate/"
            "VCB_financial_statements_2015_separate_extracted.txt:1418",
            "build/tables/VCB_financial_statements_2015_separate/table_31_line1418.csv",
        ],
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Exact-source semantic correction; leaderboard result unknown.",
    }
    write_json(OUTPUT / "q118_vcb_general_provision_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
