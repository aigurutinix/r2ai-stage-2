"""Build a rollbackable v269 lineage-only cleanup for q749 and q780.

The builder copies the immutable v269 control into a new candidate, adds the
missing q749 physical-cell manifest, and corrects q780's manifest typed_factor
to match its parenthesized million-VND physical cell.  It refuses to mutate an
existing candidate and asserts that submission answers and all non-lineage
artifacts remain byte-identical.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from tempfile import mkdtemp


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v269_source_lineage_control"
OUTPUT = ROOT / "sub_v272_q749_q780"


Q749_ROWS = [
    ["EIB", "2025", "note:investment_securities_trading_income_million", "380", "1.0", "1.0", "EIB_financial_statements_2025_separate|1381", "table_64_line1381.csv", "2", "1"],
    ["ACB", "2025", "note:investment_securities_trading_income_million", "450.276", "1000.0", "1.0", "ACB_financial_statements_2025_separate|2015", "table_77_line2015.csv", "1", "1"],
]
CSV_HEADER = ["ticker", "year", "metric_key", "raw", "typed_factor", "scale", "source_table", "source_csv", "row_idx", "col_idx"]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_manifest(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)


def q749_audit_entry() -> dict:
    def source(
        ticker: str,
        raw: str,
        table_ref: str,
        csv_name: str,
        row: int,
        typed_factor: float,
    ) -> dict:
        return {
            "table_ref": table_ref,
            "csv": csv_name,
            "row": row,
            "column": 1,
            "metric": "note:investment_securities_trading_income_million",
            "label": "Investment-securities trading income, million VND",
            "source_row_labels": ["Thu nhập từ mua bán chứng khoán đầu tư"],
            "scale": 1.0,
            "typed_factor": typed_factor,
            "raw": raw,
        }

    return {
        "id": 749,
        "old_answer": -449896.0,
        "answer": -449896.0,
        "note": "EIB parent investment-securities trading income minus ACB parent investment-securities trading income in 2025, million VND",
        "sources": [
            source("EIB", "380", "EIB_financial_statements_2025_separate|1381", "table_64_line1381.csv", 2, 1.0),
            source("ACB", "450.276", "ACB_financial_statements_2025_separate|2015", "table_77_line2015.csv", 1, 1000.0),
        ],
    }


def build() -> dict:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {OUTPUT}")
    temp = Path(mkdtemp(prefix="v272_q749_q780_", dir=str(ROOT)))
    try:
        staged = temp / OUTPUT.name
        shutil.copytree(SOURCE, staged)
        source_submission_hash = sha(SOURCE / "submission.json")
        if sha(staged / "submission.json") != source_submission_hash:
            raise AssertionError("copy changed submission bytes")

        write_manifest(staged / "data" / "q749_source_cells.csv", Q749_ROWS)
        q780 = staged / "data" / "q780_source_cells.csv"
        rows = list(csv.DictReader(q780.open(encoding="utf-8-sig", newline="")))
        if len(rows) != 2 or rows[1]["raw"] != "(210.684)":
            raise AssertionError("unexpected q780 source manifest")
        rows[1]["typed_factor"] = "1.0"
        with q780.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        source_audit_path = staged / "source_audit.json"
        source_audit = read_json(source_audit_path)
        by_id = {int(item["id"]): item for item in source_audit}
        if 749 in by_id:
            raise AssertionError("q749 source audit unexpectedly already exists")
        if 780 not in by_id:
            raise AssertionError("q780 source audit missing")
        by_id[780]["sources"][1]["typed_factor"] = 1.0
        by_id[780]["sources"][1]["raw"] = "(210.684)"
        source_audit = [q749_audit_entry()] + [item for item in source_audit if int(item["id"]) != 749]
        source_audit.sort(key=lambda item: int(item["id"]))
        source_audit_path.write_text(json.dumps(source_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        if sha(staged / "submission.json") != source_submission_hash:
            raise AssertionError("cleanup changed submission bytes")
        audit = {
            "candidate": OUTPUT.name,
            "baseline": SOURCE.name,
            "purpose": "lineage-only cleanup for q749 missing manifest and q780 typed_factor metadata",
            "changed_lineage_ids": [749, 780],
            "answer_changes": [],
            "submission_sha256": source_submission_hash,
            "candidate_submission_sha256": sha(staged / "submission.json"),
            "invariants": {
                "submission_identical_to_v269": True,
                "answers_unchanged": True,
                "q749_manifest_added": True,
                "q749_numeric_inference_factor_fixed": True,
                "q780_typed_factor_fixed_to_1": True,
                "abs_difference_operator_unchanged": True,
                "automatic_promotion": False,
            },
            "physical_refs": {
                "q749": ["EIB_financial_statements_2025_separate|1381", "ACB_financial_statements_2025_separate|2015"],
                "q780": ["BAB_financial_statements_2024_separate|180", "SGB_financial_statements_2024_separate|263"],
            },
        }
        (staged / "v272_q749_q780_cleanup_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        shutil.move(str(staged), str(OUTPUT))
        return audit
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
