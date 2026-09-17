"""Build the isolated q411 PDR comparative-column repair over v189.

PDR's 2025 consolidated income statement contains both 2025 and comparative
2024 revenue.  The prior program set PDR's 2024 revenue to ``None`` and thus
excluded PDR from a conditional count.  This builder appends the missing
source cell to the existing grounded manifest and lets the unchanged panel
formula evaluate PDR normally.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v189_interest_coverage_difference"
OUTPUT = ROOT / "sub_top123_candidate_v190_pdr_growth_count"
QUESTION_ID = 411
SOURCE_TABLE = "PDR_financial_statements_2025_consolidated|286"
SOURCE_CSV = "table_6_line286.csv"
MANIFEST = "q411_source_cells.csv"

OLD_2024_ROW = (
    "{'ticker': 'PDR', 'year': 2024, 'revenue': None, 'cfo': None, "
    "'_prev_revenue': None}"
)
NEW_2024_ROW = (
    "{'ticker': 'PDR', 'year': 2024, 'revenue': _source_value('PDR', 2024, "
    "'kqkd:10'), 'cfo': None, '_prev_revenue': None}"
)
OLD_2025_ROW = (
    "{'ticker': 'PDR', 'year': 2025, 'revenue': _source_value('PDR', 2025, "
    "'kqkd:10'), 'cfo': _source_value('PDR', 2025, 'lctt:20'), "
    "'_prev_revenue': None}"
)
NEW_2025_ROW = (
    "{'ticker': 'PDR', 'year': 2025, 'revenue': _source_value('PDR', 2025, "
    "'kqkd:10'), 'cfo': _source_value('PDR', 2025, 'lctt:20'), "
    "'_prev_revenue': _source_value('PDR', 2024, 'kqkd:10')}"
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    submission = read_json(SOURCE / "submission.json")
    rows = {int(row["id"]): row for row in submission}
    row = rows[QUESTION_ID]
    before_rows = {int(item["id"]): digest(item) for item in submission}
    before_submission = digest(submission)

    if float(row["answer"]) != 2.0:
        raise ValueError(f"unexpected q411 baseline answer: {row['answer']!r}")
    for old in (OLD_2024_ROW, OLD_2025_ROW):
        if row["pandas_query"].count(old) != 1:
            raise ValueError(f"reviewed q411 row not found exactly once: {old}")

    source_frame = pd.read_csv(
        ROOT / "build" / "tables" / "PDR_financial_statements_2025_consolidated" / SOURCE_CSV,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        index_col=None,
    )
    comparative_revenue = str(source_frame.iloc[3, 4])
    if comparative_revenue != "821.690.538.691":
        raise ValueError(f"unexpected PDR comparative 2024 revenue: {comparative_revenue!r}")

    row["answer"] = 3.0
    row["pandas_query"] = (
        row["pandas_query"]
        .replace(OLD_2024_ROW, NEW_2024_ROW)
        .replace(OLD_2025_ROW, NEW_2025_ROW)
    )

    changed = [
        question_id
        for question_id, old_digest in before_rows.items()
        if digest(rows[question_id]) != old_digest
    ]
    if changed != [QUESTION_ID]:
        raise AssertionError(f"unexpected submission deltas: {changed[:20]}")

    shutil.copytree(SOURCE, OUTPUT)
    write_json(OUTPUT / "submission.json", submission)

    manifest_path = OUTPUT / "data" / MANIFEST
    with manifest_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        manifest_rows = list(reader)
    expected_fields = [
        "ticker", "year", "metric_key", "raw", "typed_factor", "scale",
        "source_table", "source_csv", "row_idx", "col_idx",
    ]
    if fieldnames != expected_fields:
        raise ValueError(f"unexpected q411 manifest schema: {fieldnames!r}")
    if any(
        item["ticker"] == "PDR" and item["year"] == "2024" and item["metric_key"] == "kqkd:10"
        for item in manifest_rows
    ):
        raise ValueError("PDR 2024 revenue already exists in q411 manifest")
    manifest_rows.append({
        "ticker": "PDR",
        "year": "2024",
        "metric_key": "kqkd:10",
        "raw": comparative_revenue,
        "typed_factor": "1.0",
        "scale": "1.0",
        "source_table": SOURCE_TABLE,
        "source_csv": SOURCE_CSV,
        "row_idx": "3",
        "col_idx": "4",
    })
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    panel_audit = read_json(SOURCE / "panel_source_audit.json")
    panel_row = next(item for item in panel_audit if int(item["id"]) == QUESTION_ID)
    if float(panel_row["answer"]) != 2.0 or int(panel_row["source_cells"]) != 22:
        raise ValueError("q411 panel audit baseline does not match v189")
    panel_row["answer"] = 3.0
    panel_row["source_cells"] = 23
    panel_row["formula_repair"] = {
        "kind": "missing_comparative_period_operand",
        "ticker": "PDR",
        "current_year": 2025,
        "comparative_year": 2024,
        "comparative_source_table": SOURCE_TABLE,
        "comparative_raw_revenue": comparative_revenue,
    }
    write_json(OUTPUT / "panel_source_audit.json", panel_audit)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "purpose": "isolated source-proven q411 missing comparative operand repair",
        "changed_questions": [QUESTION_ID],
        "source_submission_sha256": before_submission,
        "candidate_submission_sha256": digest(submission),
        "answer_change": {"old": 2.0, "new": 3.0},
        "pdr_calculation": {
            "revenue_2025": 1324974747132.0,
            "revenue_2024": 821690538691.0,
            "revenue_growth_pct": 61.25106155662531,
            "cfo_2025": -2975418118104.0,
            "cfo_margin_pct": -224.564137984457,
            "qualifies": True,
        },
        "qualifying_tickers": ["KBC", "PDR", "SCR"],
        "invariants": {
            "question_unchanged": True,
            "relevant_docs_unchanged": True,
            "relevant_tables_unchanged": True,
            "evidence_path_unchanged": True,
            "one_source_cell_appended": True,
            "all_other_submission_rows_unchanged": True,
            "v184_locked_artifact_untouched": True,
            "v189_source_untouched": True,
        },
        "claim_limit": (
            "Source-proven local correction; BTC score is unknown until the exact "
            "artifact is evaluated. This report is not leaderboard gold."
        ),
    }
    write_json(OUTPUT / "q411_pdr_growth_count_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
