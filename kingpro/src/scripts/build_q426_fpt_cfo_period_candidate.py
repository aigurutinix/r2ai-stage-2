"""Build a v207 component fixing q426's shifted FPT CFO period columns.

The FPT 2023 and 2024 cash-flow HTML tables expand merged cells such that the
current-year CFO amount is physically one column left of its year header.  The
old compact manifest selected the duplicated comparative amount.  This builder
updates only those two source bindings and q426's materialized answer; it is a
component for a later multi-question bundle, not an upload artifact.
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
SOURCE = ROOT / "sub_top123_candidate_v206_semantic_batch11"
OUTPUT = ROOT / "sub_top123_candidate_v207_q426_fpt_cfo_period_component"
QID = 426
OLD_ANSWER = 2.73
NEW_ANSWER = 1.44
CORRECTIONS = {
    "2023": {
        "old_raw": "5.053.831.756.700",
        "raw": "9.517.095.698.405",
        "old_col_idx": "3",
        "col_idx": "2",
        "source_table": "FPT_financial_statements_2023_consolidated|311",
    },
    "2024": {
        "old_raw": "9.517.095.698.405",
        "raw": "11.703.777.188.868",
        "old_col_idx": "3",
        "col_idx": "2",
        "source_table": "FPT_financial_statements_2024_consolidated|494",
    },
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError("empty q426 source manifest")
    changed: list[dict[str, str]] = []
    for row in rows:
        correction = CORRECTIONS.get(str(row.get("year")))
        if row.get("metric_key") != "lctt:20" or not correction:
            continue
        if row.get("source_table") != correction["source_table"]:
            raise AssertionError(f"unexpected q426 table for {row.get('year')}")
        if row.get("raw") != correction["old_raw"] or row.get("col_idx") != correction["old_col_idx"]:
            raise AssertionError(f"unexpected q426 old binding for {row.get('year')}: {row}")
        row["raw"] = correction["raw"]
        row["col_idx"] = correction["col_idx"]
        changed.append(row.copy())
    if sorted(row["year"] for row in changed) != ["2023", "2024"]:
        raise AssertionError(f"unexpected corrected q426 years: {changed}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return changed


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    old_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in old_rows}
    row = next(item for item in rows if int(item["id"]) == QID)
    if float(row["answer"]) != OLD_ANSWER:
        raise AssertionError(f"unexpected q426 old answer: {row['answer']}")
    row["answer"] = NEW_ANSWER
    write_json(OUTPUT / "submission.json", rows)

    corrected_sources = patch_manifest(OUTPUT / "data" / "q426_source_cells.csv")

    panel_path = OUTPUT / "panel_source_audit.json"
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    panel_row = next(item for item in panel if int(item["id"]) == QID)
    if float(panel_row["answer"]) != OLD_ANSWER:
        raise AssertionError("unexpected q426 panel audit answer")
    panel_row["answer"] = NEW_ANSWER
    panel_row["period_binding_fix"] = {
        "2023": "column 2 current period, replacing duplicated comparative column 3",
        "2024": "column 2 current period, replacing duplicated comparative column 3",
    }
    write_json(panel_path, panel)

    changed_ids = [
        int(candidate["id"])
        for candidate in rows
        if candidate != old_by_id[int(candidate["id"])]
    ]
    if changed_ids != [QID]:
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    changed_fields = {
        key
        for key in set(row) | set(old_by_id[QID])
        if row.get(key) != old_by_id[QID].get(key)
    }
    if changed_fields != {"answer"}:
        raise AssertionError(f"unexpected q426 changed fields: {sorted(changed_fields)}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "old_answer": OLD_ANSWER,
        "new_answer": NEW_ANSWER,
        "selected_year_after_repair": 2022,
        "source_corrections": corrected_sources,
        "recomputation": {
            "2022_npat": 6_491_343_454_469.0,
            "2022_cfo": 5_053_831_756_700.0,
            "npat_minus_cfo_trillion": 1.437511697769,
        },
        "invariants": {
            "only_q426_submission_row_changed": True,
            "only_materialized_answer_changed_in_submission_json": True,
            "query_unchanged": True,
            "relevant_tables_unchanged": True,
            "evidence_path_unchanged": True,
            "source_binding_rows_changed": 2,
        },
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Exact-source component fix; hold for a multi-question bundle and full release gate.",
    }
    write_json(OUTPUT / "q426_fpt_cfo_period_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
