"""Stage the source-proven q24 counterparty correction on top of v217.

The old q24 payload reads the grand total of two related-party lenders.  The
question names only Công ty Cổ phần Hoàng Anh Gia Lai, whose direct contract
row is 1,957,824,733 thousand VND.  This builder is fail-closed and never
modifies v217, v206 or v207 in place.
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
SOURCE = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
OUTPUT = ROOT / "sub_top123_candidate_v228_q24_counterparty_fix"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")
QID = 24
OLD_ANSWER = 2083992733.0
NEW_ANSWER = 1957824733.0
DOC = "HNG_financial_statements_2017_separate"
OLD_TABLE = f"{DOC}|995"
NEW_TABLE = f"{DOC}|987"
HEADER_TABLE = f"{DOC}|983"
RAW_COUNTERPARTY = "1.957.824.733"
RAW_OTHER = "126.168.000"
RAW_TOTAL = "2.083.992.733"


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def by_id(rows: list[dict]) -> dict[int, dict]:
    return {int(row["id"]): row for row in rows}


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    report_text = next(
        (ROOT / "data" / "financial_statements").rglob(
            "HNG_financial_statements_2017_separate_extracted.txt"
        )
    ).read_text(encoding="utf-8", errors="replace")
    required_fragments = (
        "Bên cho vay",
        "Công ty Cổ phần Hoàng Anh Gia Lai, công ty mẹ",
        RAW_COUNTERPARTY,
        "Công ty Cổ phần Thủy Điện Hoàng Anh Gia Lai",
        RAW_OTHER,
        RAW_TOTAL,
    )
    missing = [fragment for fragment in required_fragments if fragment not in report_text]
    if missing:
        raise AssertionError(f"q24 physical proof missing: {missing}")
    if int(RAW_COUNTERPARTY.replace(".", "")) + int(RAW_OTHER.replace(".", "")) != int(
        RAW_TOTAL.replace(".", "")
    ):
        raise AssertionError("q24 counterparty additivity proof failed")

    rows = load(SOURCE / "submission.json")
    audits = load(SOURCE / "source_audit.json")
    if not isinstance(rows, list) or not isinstance(audits, list):
        raise AssertionError("submission/source_audit must be lists")
    source_map = by_id(rows)
    audit_map = by_id(audits)
    if source_map[QID].get("answer") != OLD_ANSWER:
        raise AssertionError("unexpected v217 q24 answer")
    if source_map[QID].get("relevant_tables") != [OLD_TABLE]:
        raise AssertionError("unexpected v217 q24 table")

    shutil.copytree(SOURCE, BUILDING)
    staged_rows = json.loads(json.dumps(rows, ensure_ascii=False))
    staged = by_id(staged_rows)[QID]
    staged["answer"] = NEW_ANSWER
    staged["relevant_docs"] = [DOC]
    staged["relevant_tables"] = [NEW_TABLE]
    staged["evidence"] = [{"variable": "df1", "csv_path": "data/q24_source_cells.csv"}]
    changed = [
        int(row["id"])
        for row in staged_rows
        if row != source_map[int(row["id"])]
    ]
    if changed != [QID]:
        raise AssertionError(f"unexpected changed IDs: {changed}")
    write_json(BUILDING / "submission.json", staged_rows)

    evidence_path = BUILDING / "data" / "q24_source_cells.csv"
    with evidence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
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
            ]
        )
        writer.writerow(
            [
                "HNG",
                2017,
                "note:hag_long_term_loan_named_counterparty_ending",
                RAW_COUNTERPARTY,
                1.0,
                1000.0,
                NEW_TABLE,
                "table_38_line987.csv",
                0,
                1,
            ]
        )

    staged_audits = json.loads(json.dumps(audits, ensure_ascii=False))
    audit = by_id(staged_audits)[QID]
    audit.clear()
    audit.update(
        {
            "id": QID,
            "old_answer": OLD_ANSWER,
            "answer": NEW_ANSWER,
            "note": (
                "Counterparty repair: the old value was the grand total of HAGL "
                "and HAGL Hydropower. The question names only HAGL."
            ),
            "sources": [
                {
                    "table_ref": NEW_TABLE,
                    "header_table_ref": HEADER_TABLE,
                    "csv": "q24_source_cells.csv",
                    "row": 0,
                    "column": 1,
                    "metric": "note:hag_long_term_loan_named_counterparty_ending",
                    "label": "HAGL long-term loan contract, ending balance, thousand VND",
                    "source_row_labels": [
                        "Công ty Cổ phần Hoàng Anh Gia Lai, công ty mẹ",
                        "Hợp đồng vay số 10/HĐVHAGL-NNQT",
                    ],
                    "scale": 1000.0,
                    "typed_factor": 1.0,
                    "raw": RAW_COUNTERPARTY,
                }
            ],
        }
    )
    write_json(BUILDING / "source_audit.json", staged_audits)

    proof = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_changes": {str(QID): {"from": OLD_ANSWER, "to": NEW_ANSWER}},
        "table_changes": {str(QID): {"from": [OLD_TABLE], "to": [NEW_TABLE]}},
        "counterparty_proof_thousand_vnd": {
            "HAGL": int(RAW_COUNTERPARTY.replace(".", "")),
            "HAGL_Hydropower": int(RAW_OTHER.replace(".", "")),
            "grand_total": int(RAW_TOTAL.replace(".", "")),
        },
        "header_table": HEADER_TABLE,
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "claim_limit": "One source-proven q24 correction; leaderboard effect is unmeasured.",
    }
    write_json(BUILDING / "v228_q24_counterparty_fix_audit.json", proof)
    BUILDING.rename(OUTPUT)
    proof["candidate_submission_sha256"] = sha256(OUTPUT / "submission.json")
    proof["candidate_evidence_sha256"] = sha256(OUTPUT / "data" / "q24_source_cells.csv")
    write_json(OUTPUT / "v228_q24_counterparty_fix_audit.json", proof)
    print(json.dumps(proof, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
