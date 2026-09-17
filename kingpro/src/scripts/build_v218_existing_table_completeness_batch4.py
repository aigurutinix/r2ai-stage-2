"""Build four completeness repairs using already-retrieved source tables.

v217 proved on the public leaderboard that filling live-cohort operands can
recover execution and answer credit even when the local stored answer remains
unchanged.  This batch deliberately adds no relevant-table labels:

* q411: PDR 2024 CFO from the comparative column of its already retrieved
  2025 cash-flow table;
* q472/q489/q547: AAA 2020 NPAT from the already retrieved 2020 income table.

The output is assembled in a ``.building`` directory and renamed only after
all assertions and writes succeed, so a failed build cannot masquerade as a
submission artifact.
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
OUTPUT = ROOT / "sub_top123_candidate_v218_existing_table_completeness_batch4"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")

FIELDS = [
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

PDR_CFO = {
    "ticker": "PDR",
    "year": "2024",
    "metric_key": "lctt:20",
    "raw": "(4.616.202.196.414)",
    "typed_factor": "1.0",
    "scale": "1.0",
    "source_table": "PDR_financial_statements_2025_consolidated|323",
    "source_csv": "table_7_line323.csv",
    "row_idx": "18",
    "col_idx": "4",
}

AAA_NPAT = {
    "ticker": "AAA",
    "year": "2020",
    "metric_key": "kqkd:60",
    "raw": "283.172.810.679",
    "typed_factor": "1.0",
    "scale": "1.0",
    "source_table": "AAA_financial_statements_2020_consolidated|248",
    "source_csv": "table_7_line248.csv",
    "row_idx": "19",
    "col_idx": "3",
}

PATCHES = {
    411: {
        "old": "'cfo': None, '_prev_revenue': None",
        "new": (
            "'cfo': _source_value('PDR', 2024, 'lctt:20'), "
            "'_prev_revenue': None"
        ),
        "cell": PDR_CFO,
        "expected_cells": 24,
    },
    472: {
        "old": "'npat': None",
        "new": "'npat': _source_value('AAA', 2020, 'kqkd:60')",
        "cell": AAA_NPAT,
        "expected_cells": 45,
    },
    489: {
        "old": "'npat': None",
        "new": "'npat': _source_value('AAA', 2020, 'kqkd:60')",
        "cell": AAA_NPAT,
        "expected_cells": 45,
    },
    547: {
        "old": "'npat': None",
        "new": "'npat': _source_value('AAA', 2020, 'kqkd:60')",
        "cell": AAA_NPAT,
        "expected_cells": 45,
    },
}


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _physical_raw(cell: dict[str, str]) -> str:
    document, _ = cell["source_table"].rsplit("|", 1)
    path = ROOT / "build" / "tables" / document / cell["source_csv"]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[int(cell["row_idx"]) + 1][int(cell["col_idx"])]


def _append_cell(path: Path, addition: dict[str, str]) -> None:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    key = (
        addition["ticker"],
        addition["year"],
        addition["metric_key"],
        addition["source_table"],
    )
    existing = {
        (row["ticker"], row["year"], row["metric_key"], row["source_table"])
        for row in rows
    }
    if key in existing:
        raise AssertionError(f"duplicate compact source cell: {key}")
    if _physical_raw(addition) != addition["raw"]:
        raise AssertionError(f"physical source mismatch: {key}")
    rows.append(addition)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    by_id = {int(row["id"]): row for row in rows}

    for qid, patch in PATCHES.items():
        row = by_id[qid]
        code = str(row.get("pandas_query") or "")
        if code.count(patch["old"]) != 1:
            raise AssertionError(f"q{qid}: unexpected missing-operand expression")
        if patch["cell"]["source_table"] not in row["relevant_tables"]:
            raise AssertionError(f"q{qid}: repair table is not already retrieved")
        row["pandas_query"] = code.replace(patch["old"], patch["new"], 1)

    changed = [int(row["id"]) for row in rows if row != source_by_id[int(row["id"])]]
    if changed != sorted(PATCHES):
        raise AssertionError(f"unexpected changed IDs: {changed}")
    for qid in changed:
        fields = {
            key
            for key in by_id[qid]
            if by_id[qid].get(key) != source_by_id[qid].get(key)
        }
        if fields != {"pandas_query"}:
            raise AssertionError(f"q{qid}: unexpected changed fields {fields}")

    shutil.copytree(SOURCE, BUILDING)
    _write_json(BUILDING / "submission.json", rows)
    for qid, patch in PATCHES.items():
        _append_cell(BUILDING / "data" / f"q{qid}_source_cells.csv", patch["cell"])

    audits = json.loads((BUILDING / "panel_source_audit.json").read_text(encoding="utf-8"))
    audit_by_id = {int(item["id"]): item for item in audits}
    for qid, patch in PATCHES.items():
        audit = audit_by_id[qid]
        audit["source_cells"] = patch["expected_cells"]
        audit["dependency_mode"] = "complete_existing_table_operand"
    _write_json(BUILDING / "panel_source_audit.json", audits)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": sorted(PATCHES),
        "answer_changes": [],
        "changed_fields": {str(qid): ["pandas_query"] for qid in sorted(PATCHES)},
        "physical_cells_added": len(PATCHES),
        "relevant_tables_added": [],
        "source_submission_sha256": _digest(source_rows),
        "candidate_submission_sha256": _digest(rows),
        "claim_limit": "Leaderboard effect is unknown until submitted.",
    }
    _write_json(BUILDING / "v218_existing_table_completeness_batch4_audit.json", report)
    BUILDING.rename(OUTPUT)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
