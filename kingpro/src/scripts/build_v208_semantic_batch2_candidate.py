"""Build v208 from the locked v207 base with two source-backed repairs.

* q79 reads the explicit ``Tổng cộng`` geography column instead of the
  numerically-equal ``Trong nước`` bucket.
* q251 uses the exact cash-flow ``Tiền và tương đương tiền cuối năm`` row.

Neither answer changes.  The builder starts from the already isolated q251
component, patches q79, and then proves the only submission-row differences
from locked v207 are q79 and q251.
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
LOCKED = ROOT / "sub_top123_candidate_v207_semantic_batch6_final"
SOURCE = ROOT / "sub_top123_candidate_v208_q251_cashflow_component"
OUTPUT = ROOT / "sub_top123_candidate_v208_semantic_batch2"
Q79 = 79
Q251 = 251
Q79_TABLE = "STB_financial_statements_2022_separate|2132"
Q79_RAW = "25.820.307"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def corrected_q79_query() -> str:
    return """_dfvals = list(dfs.values())
if len(_dfvals) >= 1:
    df1 = _dfvals[0]
if len(_dfvals) >= 2:
    df2 = _dfvals[1]
if len(_dfvals) >= 3:
    df3 = _dfvals[2]
if len(_dfvals) >= 4:
    df4 = _dfvals[3]
if len(_dfvals) >= 5:
    df5 = _dfvals[4]
if len(_dfvals) >= 6:
    df6 = _dfvals[5]
# Exact row and explicit Total column; columns 1/2 are domestic/foreign.
_r = df5[df5['0'].str.contains('Phát hành giấy tờ có giá', case=False, na=False, regex=False)]
_v = str(_r['3'].values[0]).replace('(','-').replace(')','').replace('.','').replace(',','.')
result = round(float(_v), 2)"""


def verify_q79_physical_source() -> None:
    path = (
        ROOT
        / "build"
        / "tables"
        / "STB_financial_statements_2022_separate"
        / "table_101_line2132.csv"
    )
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    if lines[1] != ",Trong nướcTriệu đồng,Nước ngoàiTriệu đồng,Tổng cộngTriệu đồng":
        raise AssertionError(f"unexpected q79 header: {lines[1]}")
    if lines[13] != "Phát hành giấy tờ có giá,25.820.307,-,25.820.307":
        raise AssertionError(f"unexpected q79 row: {lines[13]}")


def append_q79_source_audit(path: Path) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    if any(int(record["id"]) == Q79 for record in records):
        raise AssertionError("q79 unexpectedly already has a source-audit record")
    records.append(
        {
            "id": Q79,
            "old_answer": 25_820_307.0,
            "answer": 25_820_307.0,
            "note": (
                "STB parent 2022 issued valuable papers, million VND; "
                "bound to the explicit Total column rather than the equal domestic bucket"
            ),
            "sources": [
                {
                    "table_ref": Q79_TABLE,
                    "csv": "table_101_line2132.csv",
                    "row": 12,
                    "column": 3,
                    "metric": "note:issued_valuable_papers_total_million",
                    "label": "Phát hành giấy tờ có giá | Tổng cộng",
                    "source_row_labels": ["Phát hành giấy tờ có giá"],
                    "scale": 1.0,
                    "typed_factor": 1.0,
                    "raw": Q79_RAW,
                }
            ],
        }
    )
    records.sort(key=lambda record: int(record["id"]))
    write_json(path, records)


def build() -> dict[str, object]:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not LOCKED.is_dir() or not SOURCE.is_dir():
        raise FileNotFoundError("locked v207 or isolated q251 component is missing")
    verify_q79_physical_source()
    shutil.copytree(SOURCE, OUTPUT)

    locked_rows = json.loads((LOCKED / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    locked_by_id = {int(row["id"]): row for row in locked_rows}
    row79 = next(row for row in rows if int(row["id"]) == Q79)
    before79 = locked_by_id[Q79]
    if (
        float(row79["answer"]) != 25_820_307.0
        or row79.get("relevant_tables") != [Q79_TABLE]
        or "_r['1'].values[0]" not in str(row79.get("pandas_query"))
    ):
        raise AssertionError("unexpected locked q79 state")
    row79["pandas_query"] = corrected_q79_query()
    write_json(OUTPUT / "submission.json", rows)
    append_q79_source_audit(OUTPUT / "source_audit.json")

    changed_ids = [
        int(row["id"])
        for row in rows
        if row != locked_by_id[int(row["id"])]
    ]
    if changed_ids != [Q79, Q251]:
        raise AssertionError(f"unexpected changed IDs from v207: {changed_ids}")
    changed_fields = {
        Q79: sorted(
            key
            for key in set(row79) | set(before79)
            if row79.get(key) != before79.get(key)
        ),
    }
    row251 = next(row for row in rows if int(row["id"]) == Q251)
    changed_fields[Q251] = sorted(
        key
        for key in set(row251) | set(locked_by_id[Q251])
        if row251.get(key) != locked_by_id[Q251].get(key)
    )
    if changed_fields != {Q79: ["pandas_query"], Q251: ["relevant_tables"]}:
        raise AssertionError(f"unexpected changed fields: {changed_fields}")

    report = {
        "candidate": OUTPUT.name,
        "locked_base": LOCKED.name,
        "changed_question_ids_relative_to_locked_v207": [Q79, Q251],
        "answer_changes": [],
        "repairs": {
            "79": "domestic bucket -> explicit total column; raw and answer unchanged",
            "251": "balance-sheet cash row -> exact cash-flow ending row; raw and answer unchanged",
        },
        "changed_fields": {str(key): value for key, value in changed_fields.items()},
        "locked_submission_sha256": sha256(LOCKED / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Provenance/structure batch; leaderboard effect unknown until submitted.",
    }
    write_json(OUTPUT / "v208_semantic_batch2_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
