"""Build the isolated q511 EPS-selection repair over the scored v190 champion.

The baseline compared DPM, HT1 and HPG but supplied EPS only for HT1.  The
source reports disclose EPS for all three companies; HPG has the largest 2018
basic EPS, so the requested NPAT/equity ratio must be calculated for HPG.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v190_pdr_growth_count"
OUTPUT = ROOT / "sub_top123_candidate_v191_eps_selection"
QUESTION_ID = 511
MANIFEST = "q511_source_cells.csv"

EPS_SOURCES = [
    {
        "ticker": "DPM",
        "year": "2018",
        "metric_key": "kqkd:70",
        "raw": "1.551",
        "typed_factor": "1000.0",
        "scale": "1.0",
        "source_table": "DPM_financial_statements_2018_consolidated|1135",
        "source_csv": "table_49_line1135.csv",
        "row_idx": "5",
        "col_idx": "1",
    },
    {
        "ticker": "HPG",
        "year": "2018",
        "metric_key": "kqkd:70",
        "raw": "4.037",
        "typed_factor": "1000.0",
        "scale": "1.0",
        "source_table": "HPG_financial_statements_2018_consolidated|246",
        "source_csv": "table_6_line246.csv",
        "row_idx": "6",
        "col_idx": "3",
    },
]

ROW_REPLACEMENTS = {
    (
        "{'ticker': 'DPM', 'year': 2018, 'npat': _source_value('DPM', 2018, "
        "'kqkd:60'), 'eps': None, 'equity': _source_value('DPM', 2018, 'cdkt:400')}"
    ): (
        "{'ticker': 'DPM', 'year': 2018, 'npat': _source_value('DPM', 2018, "
        "'kqkd:60'), 'eps': _source_value('DPM', 2018, 'kqkd:70'), "
        "'equity': _source_value('DPM', 2018, 'cdkt:400')}"
    ),
    (
        "{'ticker': 'HPG', 'year': 2018, 'npat': _source_value('HPG', 2018, "
        "'kqkd:60'), 'eps': None, 'equity': _source_value('HPG', 2018, 'cdkt:400')}"
    ): (
        "{'ticker': 'HPG', 'year': 2018, 'npat': _source_value('HPG', 2018, "
        "'kqkd:60'), 'eps': _source_value('HPG', 2018, 'kqkd:70'), "
        "'equity': _source_value('HPG', 2018, 'cdkt:400')}"
    ),
}


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


def source_cell(source: dict[str, str]) -> str:
    path = (
        ROOT
        / "build"
        / "tables"
        / source["source_table"].split("|", 1)[0]
        / source["source_csv"]
    )
    frame = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        index_col=None,
    )
    return str(frame.iloc[int(source["row_idx"]), int(source["col_idx"])])


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

    if float(row["answer"]) != 12.39:
        raise ValueError(f"unexpected q511 baseline answer: {row['answer']!r}")
    for old in ROW_REPLACEMENTS:
        if row["pandas_query"].count(old) != 1:
            raise ValueError(f"reviewed q511 row not found exactly once: {old}")
    for source in EPS_SOURCES:
        actual = source_cell(source)
        if actual != source["raw"]:
            raise ValueError(
                f"unexpected {source['ticker']} EPS source: {actual!r} != {source['raw']!r}"
            )

    for old, new in ROW_REPLACEMENTS.items():
        row["pandas_query"] = row["pandas_query"].replace(old, new)
    row["answer"] = 21.17
    expected_tables = [
        "DPM_financial_statements_2018_consolidated|305",
        "DPM_financial_statements_2018_consolidated|268",
        "HT1_financial_statements_2018_consolidated|306",
        "HT1_financial_statements_2018_consolidated|270",
        "HPG_financial_statements_2018_consolidated|229",
        "HPG_financial_statements_2018_consolidated|196",
    ]
    if row["relevant_tables"] != expected_tables:
        raise ValueError("q511 relevant_tables baseline does not match v190")
    row["relevant_tables"] = expected_tables + [
        source["source_table"] for source in EPS_SOURCES
    ]

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
        raise ValueError(f"unexpected q511 manifest schema: {fieldnames!r}")
    existing = {
        (item["ticker"], item["year"], item["metric_key"])
        for item in manifest_rows
    }
    for source in EPS_SOURCES:
        key = (source["ticker"], source["year"], source["metric_key"])
        if key in existing:
            raise ValueError(f"EPS source already exists in q511 manifest: {key}")
        manifest_rows.append(source)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(manifest_rows)

    panel_audit = read_json(SOURCE / "panel_source_audit.json")
    panel_row = next(item for item in panel_audit if int(item["id"]) == QUESTION_ID)
    if float(panel_row["answer"]) != 12.39 or int(panel_row["source_cells"]) != 7:
        raise ValueError("q511 panel audit baseline does not match v190")
    panel_row["answer"] = 21.17
    panel_row["source_cells"] = 9
    panel_row["formula_repair"] = {
        "kind": "missing_selection_metric_operands",
        "metric": "basic EPS",
        "added_tickers": ["DPM", "HPG"],
        "selected_ticker": "HPG",
    }
    write_json(OUTPUT / "panel_source_audit.json", panel_audit)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "purpose": "isolated source-proven q511 missing EPS selection operands repair",
        "changed_questions": [QUESTION_ID],
        "source_submission_sha256": before_submission,
        "candidate_submission_sha256": digest(submission),
        "answer_change": {"old": 12.39, "new": 21.17},
        "selection": {
            "DPM_eps": 1551.0,
            "HT1_eps": 1681.0,
            "HPG_eps": 4037.0,
            "selected_ticker": "HPG",
            "HPG_npat": 8600550706227.0,
            "HPG_equity": 40622949840810.0,
            "HPG_npat_to_equity_pct": 21.171654791023688,
        },
        "invariants": {
            "question_unchanged": True,
            "relevant_docs_unchanged": True,
            "relevant_tables_extended_with_two_verified_eps_tables": True,
            "evidence_path_unchanged": True,
            "two_source_cells_appended": True,
            "all_other_submission_rows_unchanged": True,
            "v184_locked_artifact_untouched": True,
            "v190_scored_champion_untouched": True,
        },
        "claim_limit": (
            "Source-proven local correction; BTC score is unknown until the exact "
            "artifact is evaluated. This report is not leaderboard gold."
        ),
    }
    write_json(OUTPUT / "q511_eps_selection_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
