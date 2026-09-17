"""Build the isolated q671 MBB credit-provision ratio repair over v183.

The legacy program divided gross customer-loan provision expense by profit
before tax.  The question asks for the broader credit-risk provision ratio,
whose numerator is the income-statement ``Chi phí dự phòng rủi ro`` amount.
This candidate reads that net amount, reconciles it to the total in Note 35,
and divides it by profit before tax.  Every operand comes from BTC-provided
CSVs at grader runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v183_ctg_total_loan_provision"
OUTPUT = ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
QUESTION_ID = 671

NOTE_TABLE = "MBB_financial_statements_2020_consolidated|2005"
INCOME_TABLE = "MBB_financial_statements_2020_consolidated|409"
NOTE_CSV = "MBB_financial_statements_2020_consolidated_2005.csv"
INCOME_CSV = "MBB_financial_statements_2020_consolidated_409.csv"


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

    if row["answer"] != 57.44:
        raise ValueError(f"unexpected q671 baseline answer: {row['answer']!r}")
    if row.get("relevant_docs") != ["MBB_financial_statements_2020_consolidated"]:
        raise ValueError("q671 source document changed from reviewed baseline")
    if row.get("relevant_tables") != [NOTE_TABLE, INCOME_TABLE]:
        raise ValueError("q671 source tables changed from reviewed baseline")

    note_net_provision = source_cell(NOTE_CSV, 3, 1)
    statement_provision = source_cell(INCOME_CSV, 17, 2)
    profit_before_tax = source_cell(INCOME_CSV, 18, 2)
    if note_net_provision != "6.118.440":
        raise ValueError(f"unexpected Note 35 net provision: {note_net_provision!r}")
    if statement_provision != "(6.118.440)":
        raise ValueError(f"unexpected income-statement provision: {statement_provision!r}")
    if profit_before_tax != "10.688.276":
        raise ValueError(f"unexpected profit before tax: {profit_before_tax!r}")

    # Reuse the hardened string/typed-number parser from audited q33.
    q33_query = rows[33]["pandas_query"]
    marker = "df1 = list(dfs.values())[0]"
    parser, separator, _ = q33_query.partition(marker)
    if not separator:
        raise ValueError("cannot locate hardened number parser in q33")

    row["answer"] = 57.24
    row["evidence"] = [
        {"variable": "df1", "csv_path": "data/q671_source_cells.csv"}
    ]
    row["pandas_query"] = parser + """df1 = list(dfs.values())[0]
v_note_net = _btc_number(df1.iloc[0]['raw'], df1.iloc[0]['typed_factor']) * float(df1.iloc[0]['scale'])
v_statement_expense = _btc_number(df1.iloc[1]['raw'], df1.iloc[1]['typed_factor']) * float(df1.iloc[1]['scale'])
v_profit_before_tax = _btc_number(df1.iloc[2]['raw'], df1.iloc[2]['typed_factor']) * float(df1.iloc[2]['scale'])
# Note 35 nets customer-loan provisioning against reversal on other credit-risk
# assets.  The resulting 6,118,440 equals the income-statement credit-risk
# provision expense, so use that broader measure rather than the gross
# customer-loan-only line.  Refuse if the two disclosures stop reconciling.
if abs(abs(v_note_net) - abs(v_statement_expense)) <= 0.01 and v_profit_before_tax != 0:
    result = round(abs(v_statement_expense) / v_profit_before_tax * 100, 2)
else:
    result = None
"""

    changed_elsewhere = [
        qid
        for qid in before_rows
        if qid != QUESTION_ID and digest(rows[qid]) != before_rows[qid]
    ]
    if changed_elsewhere:
        raise AssertionError(f"unexpected row changes outside q671: {changed_elsewhere[:10]}")

    shutil.copytree(SOURCE, OUTPUT)
    manifest_path = OUTPUT / "data" / "q671_source_cells.csv"
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
            "ticker": "MBB",
            "year": 2020,
            "metric_key": "note:q671_net_credit_provision_expense_million",
            "raw": note_net_provision,
            "typed_factor": 1.0,
            "scale": 1.0,
            "source_table": NOTE_TABLE,
            "source_csv": NOTE_CSV,
            "row_idx": 3,
            "col_idx": 1,
        },
        {
            "ticker": "MBB",
            "year": 2020,
            "metric_key": "income:q671_credit_provision_expense_million",
            "raw": statement_provision,
            "typed_factor": 1.0,
            "scale": 1.0,
            "source_table": INCOME_TABLE,
            "source_csv": INCOME_CSV,
            "row_idx": 17,
            "col_idx": 2,
        },
        {
            "ticker": "MBB",
            "year": 2020,
            "metric_key": "income:q671_profit_before_tax_million",
            "raw": profit_before_tax,
            "typed_factor": 1.0,
            "scale": 1.0,
            "source_table": INCOME_TABLE,
            "source_csv": INCOME_CSV,
            "row_idx": 18,
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
        raise ValueError("q671 unexpectedly already exists in source_audit")
    audit_by_id[QUESTION_ID] = {
        "id": QUESTION_ID,
        "old_answer": 57.44,
        "answer": 57.24,
        "note": (
            "MBB consolidated 2020 net credit-risk provision expense divided "
            "by profit before tax. Note 35 total reconciles to the income-"
            "statement expense; the previous answer used only the gross "
            "customer-loan provision line before reversal on other assets."
        ),
        "sources": [
            {
                "table_ref": NOTE_TABLE,
                "csv": NOTE_CSV,
                "row": 3,
                "column": 1,
                "metric": "note:q671_net_credit_provision_expense_million",
                "label": "Tổng chi phí dự phòng rủi ro năm 2020 (triệu đồng)",
                "source_row_labels": [
                    "Trích lập dự phòng rủi ro cho vay khách hàng",
                    "Hoàn nhập dự phòng rủi ro cho các tài sản có khác",
                    "Tổng cộng",
                ],
                "scale": 1.0,
                "typed_factor": 1.0,
                "raw": note_net_provision,
            },
            {
                "table_ref": INCOME_TABLE,
                "csv": INCOME_CSV,
                "row": 17,
                "column": 2,
                "metric": "income:q671_credit_provision_expense_million",
                "label": "Chi phí dự phòng rủi ro năm 2020 (triệu đồng)",
                "source_row_labels": ["Chi phí dự phòng rủi ro"],
                "scale": 1.0,
                "typed_factor": 1.0,
                "raw": statement_provision,
            },
            {
                "table_ref": INCOME_TABLE,
                "csv": INCOME_CSV,
                "row": 18,
                "column": 2,
                "metric": "income:q671_profit_before_tax_million",
                "label": "Tổng lợi nhuận trước thuế năm 2020 (triệu đồng)",
                "source_row_labels": ["TỔNG LỢI NHUẬN TRƯỚC THUẾ"],
                "scale": 1.0,
                "typed_factor": 1.0,
                "raw": profit_before_tax,
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
        "purpose": "isolated source-reconciled q671 credit-provision ratio repair",
        "submission_rows": len(submission),
        "source_submission_sha256": before_submission,
        "candidate_submission_sha256": digest(submission),
        "changed_questions": [QUESTION_ID],
        "answer_change": {"old": 57.44, "new": 57.24},
        "relevant_tables_unchanged": [NOTE_TABLE, INCOME_TABLE],
        "source_reconciliation": {
            "gross_customer_loan_provision": "6.139.086",
            "other_credit_asset_reversal": "(20.646)",
            "note_net_provision": note_net_provision,
            "income_statement_provision": statement_provision,
            "profit_before_tax": profit_before_tax,
            "identity": "6.139.086 - 20.646 = 6.118.440",
            "ratio": "6.118.440 / 10.688.276 * 100 = 57,24%",
        },
        "invariants": {
            "questions_unchanged": True,
            "relevant_docs_unchanged": True,
            "relevant_tables_unchanged": True,
            "all_other_submission_rows_unchanged": True,
            "source_values_read_at_grader_runtime": True,
            "note_total_reconciled_to_income_statement": True,
            "v165_registry_untouched": True,
            "v183_source_untouched": True,
        },
    }
    (OUTPUT / "mbb_credit_provision_ratio_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
