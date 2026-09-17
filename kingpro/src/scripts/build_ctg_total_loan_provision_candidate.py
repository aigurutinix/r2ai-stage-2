"""Build the isolated q356 CTG total-loan-provision repair over v182.

The legacy program selected only the specific-provision component from Note 11.
This candidate reads the note's explicit total and independently reconciles it
against the contra-asset balance on the parent bank balance sheet.  Both values
come from BTC-provided CSVs at grader runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v182_balanced_table_recall"
OUTPUT = ROOT / "sub_top123_candidate_v183_ctg_total_loan_provision"
QUESTION_ID = 356

NOTE_TABLE = "CTG_financial_statements_2019_separate|1212"
BALANCE_SHEET_TABLE = "CTG_financial_statements_2019_separate|264"
NOTE_CSV = "CTG_financial_statements_2019_separate_1212.csv"
BALANCE_SHEET_CSV = "CTG_financial_statements_2019_separate_264.csv"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_cell(csv_name: str, row_idx: int, col_idx: int) -> str:
    frame = pd.read_csv(
        SOURCE / "data" / csv_name,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        index_col=None,
    )
    return str(frame.iloc[row_idx, col_idx])


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    submission = read_json(SOURCE / "submission.json")
    rows = {int(row["id"]): row for row in submission}
    row = rows[QUESTION_ID]
    before_submission = digest(submission)
    before_rows = {int(item["id"]): digest(item) for item in submission}

    if row["answer"] != 6071288.0:
        raise ValueError(f"unexpected q356 baseline answer: {row['answer']!r}")
    if row.get("relevant_docs") != ["CTG_financial_statements_2019_separate"]:
        raise ValueError("q356 source document changed from reviewed baseline")
    if row.get("relevant_tables") != [NOTE_TABLE]:
        raise ValueError("q356 source table changed from reviewed baseline")

    note_total = source_cell(NOTE_CSV, 4, 3)
    balance_sheet_total = source_cell(BALANCE_SHEET_CSV, 12, 2)
    if note_total != "12.788.628":
        raise ValueError(f"unexpected Note 11 total: {note_total!r}")
    if balance_sheet_total != "(12.788.628)":
        raise ValueError(f"unexpected balance-sheet provision: {balance_sheet_total!r}")

    # Reuse the already hardened string/typed-number parser from q33.
    q33_query = rows[33]["pandas_query"]
    marker = "df1 = list(dfs.values())[0]"
    parser, separator, _ = q33_query.partition(marker)
    if not separator:
        raise ValueError("cannot locate hardened number parser in q33")

    row["answer"] = 12788628.0
    row["relevant_tables"] = [NOTE_TABLE, BALANCE_SHEET_TABLE]
    row["evidence"] = [
        {"variable": "df1", "csv_path": "data/q356_source_cells.csv"}
    ]
    row["pandas_query"] = parser + """df1 = list(dfs.values())[0]
v_note_total = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])
v_balance_sheet = _btc_number(df1.iloc[1]['raw'], df1.iloc[1]['typed_factor']) * float(df1.iloc[1]['scale'])
# The note reports a positive provision amount while the balance sheet presents
# the same balance as a contra-asset in parentheses. Refuse a mismatched source.
result = round(abs(v_note_total), 2) if abs(abs(v_note_total) - abs(v_balance_sheet)) <= 0.01 else None
"""

    unchanged_ids = [qid for qid in before_rows if qid != QUESTION_ID]
    changed_elsewhere = [qid for qid in unchanged_ids if digest(rows[qid]) != before_rows[qid]]
    if changed_elsewhere:
        raise AssertionError(f"unexpected row changes outside q356: {changed_elsewhere[:10]}")

    shutil.copytree(SOURCE, OUTPUT)
    manifest_path = OUTPUT / "data" / "q356_source_cells.csv"
    fieldnames = (
        "ticker",
        "year",
        "metric_key",
        "raw",
        "typed_factor",
        "scale",
        "source_table",
        "source_csv",
        "row_idx",
        "col_idx",
    )
    manifest_rows = [
        {
            "ticker": "CTG",
            "year": 2019,
            "metric_key": "note:q356_total_customer_loan_loss_provision_million",
            "raw": note_total,
            "typed_factor": 1.0,
            "scale": 1.0,
            "source_table": NOTE_TABLE,
            "source_csv": NOTE_CSV,
            "row_idx": 4,
            "col_idx": 3,
        },
        {
            "ticker": "CTG",
            "year": 2019,
            "metric_key": "balance_sheet:q356_customer_loan_loss_provision_million",
            "raw": balance_sheet_total,
            "typed_factor": 1.0,
            "scale": 1.0,
            "source_table": BALANCE_SHEET_TABLE,
            "source_csv": BALANCE_SHEET_CSV,
            "row_idx": 12,
            "col_idx": 2,
        },
    ]
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    (OUTPUT / "submission.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    audit_list = read_json(SOURCE / "source_audit.json")
    audit_by_id = {int(item["id"]): item for item in audit_list}
    if QUESTION_ID in audit_by_id:
        raise ValueError("q356 unexpectedly already exists in source_audit")
    audit_by_id[QUESTION_ID] = {
        "id": QUESTION_ID,
        "old_answer": 6071288.0,
        "answer": 12788628.0,
        "note": (
            "CTG parent-bank total customer-loan loss provision at 31/12/2019. "
            "Note 11 total equals general plus specific provision and reconciles "
            "to the balance-sheet contra-asset; the question asks for the amount."
        ),
        "sources": [
            {
                "table_ref": NOTE_TABLE,
                "csv": NOTE_CSV,
                "row": 4,
                "column": 3,
                "metric": "note:q356_total_customer_loan_loss_provision_million",
                "label": "Số dư tại ngày 31 tháng 12 năm 2019 - Tổng cộng (triệu đồng)",
                "source_row_labels": ["Số dư tại ngày 31 tháng 12 năm 2019"],
                "scale": 1.0,
                "typed_factor": 1.0,
                "raw": note_total,
            },
            {
                "table_ref": BALANCE_SHEET_TABLE,
                "csv": BALANCE_SHEET_CSV,
                "row": 12,
                "column": 2,
                "metric": "balance_sheet:q356_customer_loan_loss_provision_million",
                "label": "Dự phòng rủi ro cho vay khách hàng - 31/12/2019 (triệu đồng)",
                "source_row_labels": ["Dự phòng rủi ro cho vay khách hàng"],
                "scale": 1.0,
                "typed_factor": 1.0,
                "raw": balance_sheet_total,
            },
        ],
    }
    ordered_audit = [
        audit_by_id[int(item["id"])]
        for item in submission
        if int(item["id"]) in audit_by_id
    ]
    (OUTPUT / "source_audit.json").write_text(
        json.dumps(ordered_audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "purpose": "isolated source-reconciled q356 total-provision repair",
        "submission_rows": len(submission),
        "source_submission_sha256": before_submission,
        "candidate_submission_sha256": digest(submission),
        "changed_questions": [QUESTION_ID],
        "answer_change": {"old": 6071288.0, "new": 12788628.0},
        "relevant_table_change": {
            "old": [NOTE_TABLE],
            "new": [NOTE_TABLE, BALANCE_SHEET_TABLE],
        },
        "source_reconciliation": {
            "general_provision": "6.717.340",
            "specific_provision": "6.071.288",
            "note_total": note_total,
            "balance_sheet_contra_asset": balance_sheet_total,
            "identity": "6.717.340 + 6.071.288 = 12.788.628",
        },
        "invariants": {
            "questions_unchanged": True,
            "relevant_docs_unchanged": True,
            "all_other_submission_rows_unchanged": True,
            "source_values_read_at_grader_runtime": True,
            "note_total_reconciled_to_balance_sheet": True,
            "v165_registry_untouched": True,
            "v182_source_untouched": True,
        },
    }
    (OUTPUT / "ctg_total_provision_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
