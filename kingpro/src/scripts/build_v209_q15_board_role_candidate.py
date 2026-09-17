"""Build v209 from submitted v208 with one source-proven role repair.

q15 asks for HĐQT remuneration for Chu Thị Bình.  The submitted program reads
the same person's Ban Giám đốc income table.  The exact HĐQT table reports
150,000,000 VND, so this builder changes q15 to 150.00 million VND and binds
the runtime to that table.  The submitted v208 directory remains untouched.
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
SOURCE = ROOT / "sub_top123_candidate_v208_semantic_batch2"
OUTPUT = ROOT / "sub_top123_candidate_v209_q15_board_role_r2"
QID = 15
OLD_ANSWER = 1_150.85
NEW_ANSWER = 150.0
OLD_TABLE = "MPC_financial_statements_2021_separate|1309"
TABLE_REF = "MPC_financial_statements_2021_separate|1299"
CSV_NAME = "MPC_financial_statements_2021_separate_1299.csv"
SOURCE_CSV = (
    ROOT
    / "build"
    / "tables"
    / "MPC_financial_statements_2021_separate"
    / "table_60_line1299.csv"
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
# Exact person row inside the Hội đồng Quản trị remuneration table.
_r = df1[df1['0'].astype(str).str.strip().eq('Chu Thị Bình')]
if len(_r) != 1:
    raise ValueError('Expected exactly one HĐQT row for Chu Thị Bình')
_v = str(_r['1'].values[0]).replace('(','-').replace(')','').replace('.','').replace(',','.')
result = round(float(_v) * 1e-06, 2)"""


def verify_physical_source() -> None:
    lines = SOURCE_CSV.read_text(encoding="utf-8-sig").splitlines()
    expected = "Chu Thị Bình,150.000.000,150.000.000"
    if len(lines) <= 3 or lines[2] != "Hội đồng Quản trị,Hội đồng Quản trị,Hội đồng Quản trị":
        raise AssertionError("q15 source is not the expected HĐQT table")
    if lines[3] != expected:
        raise AssertionError(f"unexpected q15 source row: {lines[3] if len(lines) > 3 else None}")


def append_source_audit(path: Path) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    if any(int(record["id"]) == QID for record in records):
        raise AssertionError("q15 unexpectedly already exists in source_audit")
    records.append(
        {
            "id": QID,
            "old_answer": OLD_ANSWER,
            "answer": NEW_ANSWER,
            "note": "MPC parent 2021 HĐQT remuneration for Chu Thị Bình, VND million",
            "sources": [
                {
                    "table_ref": TABLE_REF,
                    "csv": "table_60_line1299.csv",
                    "row": 2,
                    "column": 1,
                    "metric": "note:board_member_remuneration",
                    "label": "Chu Thị Bình | Hội đồng Quản trị remuneration",
                    "source_row_labels": ["Hội đồng Quản trị", "Chu Thị Bình"],
                    "scale": 0.000001,
                    "typed_factor": 1.0,
                    "raw": "150.000.000",
                }
            ],
        }
    )
    records.sort(key=lambda record: int(record["id"]))
    write_json(path, records)


def build() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not SOURCE.is_dir() or not SOURCE_CSV.is_file():
        raise FileNotFoundError("v208 source candidate or q15 source CSV is missing")
    verify_physical_source()
    shutil.copytree(SOURCE, OUTPUT)

    copied_csv = OUTPUT / "data" / CSV_NAME
    if copied_csv.read_bytes() != SOURCE_CSV.read_bytes():
        raise AssertionError("v208 q15 HĐQT evidence differs from catalog source")

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(item["id"]): item for item in source_rows}
    row = next(item for item in rows if int(item["id"]) == QID)
    if float(row["answer"]) != OLD_ANSWER or row.get("relevant_tables") != [OLD_TABLE]:
        raise AssertionError("unexpected submitted v208 q15 state")

    row["answer"] = NEW_ANSWER
    row["relevant_tables"] = [TABLE_REF]
    row["evidence"] = [{"variable": "df1", "csv_path": f"data/{CSV_NAME}"}]
    row["pandas_query"] = corrected_query()
    write_json(OUTPUT / "submission.json", rows)
    append_source_audit(OUTPUT / "source_audit.json")

    changed_ids = [
        int(item["id"])
        for item in rows
        if item != source_by_id[int(item["id"])]
    ]
    if changed_ids != [QID]:
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    changed_fields = {
        key
        for key in set(row) | set(source_by_id[QID])
        if row.get(key) != source_by_id[QID].get(key)
    }
    expected_fields = {"answer", "relevant_tables", "evidence", "pandas_query"}
    if changed_fields != expected_fields:
        raise AssertionError(f"unexpected q15 changed fields: {sorted(changed_fields)}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "old_answer": OLD_ANSWER,
        "new_answer": NEW_ANSWER,
        "old_semantics": "same person's Ban Giám đốc income",
        "new_semantics": "HĐQT remuneration requested by the question",
        "old_table": OLD_TABLE,
        "source_table": TABLE_REF,
        "source_raw": "150.000.000 VND",
        "source_evidence": [
            "data/financial_statements/MPC/2021/MPC_financial_statements_2021_separate/"
            "MPC_financial_statements_2021_separate_extracted.txt:1299",
            "build/tables/MPC_financial_statements_2021_separate/table_60_line1299.csv",
            "build/v209_role_bound_label_collisions_v207.json",
        ],
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Exact-source role correction; leaderboard effect unknown until submitted.",
    }
    write_json(OUTPUT / "v209_q15_board_role_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
