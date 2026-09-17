"""Build v221 with the source-verified BOT segment repair for q1007."""

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
SOURCE = ROOT / "sub_top123_candidate_v220_physical_scope_batch4"
OUTPUT = ROOT / "sub_top123_candidate_v221_segment_scope_batch5"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")
QID = 1007
OLD_TABLES = {
    "HHV_financial_statements_2021_consolidated|2588",
    "HHV_financial_statements_2022_consolidated|2471",
}
REPAIRS = [
    {
        "year": "2021", "raw": "32.355.512.700.711",
        "source_table": "HHV_financial_statements_2021_consolidated|2563",
        "source_csv": "table_67_line2563.csv", "row_idx": "1", "col_idx": "1",
    },
    {
        "year": "2022", "raw": "33.657.835.517.377",
        "source_table": "HHV_financial_statements_2022_consolidated|2450",
        "source_csv": "table_76_line2450.csv", "row_idx": "1", "col_idx": "1",
    },
]
NEW_ANSWER = 92.13


def load(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def physical_csv(repair: dict[str, str]) -> Path:
    document = repair["source_table"].split("|", 1)[0]
    return ROOT / "build" / "tables" / document / repair["source_csv"]


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    source_rows = load(SOURCE / "submission.json")
    rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    source_by_id = {int(row["id"]): row for row in source_rows}
    row = {int(item["id"]): item for item in rows}[QID]
    row["answer"] = NEW_ANSWER
    kept = [table for table in row["relevant_tables"] if table not in OLD_TABLES]
    row["relevant_tables"] = list(dict.fromkeys([
        *kept, *(repair["source_table"] for repair in REPAIRS),
    ]))

    changed = [int(item["id"]) for item in rows if item != source_by_id[int(item["id"])]]
    if changed != [QID]:
        raise AssertionError(f"unexpected changed IDs: {changed}")

    shutil.copytree(SOURCE, BUILDING)
    write(BUILDING / "submission.json", rows)

    manifest_path = BUILDING / "data" / "q1007_source_cells.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        cells = list(csv.DictReader(handle))
        fields = list(cells[0])
    for index, repair in enumerate(REPAIRS):
        cells[index].update(repair)
        source = physical_csv(repair)
        if not source.is_file():
            raise FileNotFoundError(source)
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            physical = list(csv.reader(handle))
        observed = physical[int(repair["row_idx"]) + 1][int(repair["col_idx"])]
        if observed != repair["raw"]:
            raise AssertionError(f"physical mismatch: {observed!r} != {repair['raw']!r}")
        destination = BUILDING / "data" / (repair["source_table"].replace("|", "_") + ".csv")
        shutil.copy2(source, destination)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "answer_changes": {str(QID): {"from": source_by_id[QID]["answer"], "to": NEW_ANSWER}},
        "old_wrong_tables_removed": sorted(OLD_TABLES),
        "correct_tables_added": [repair["source_table"] for repair in REPAIRS],
        "ratios_percent": {
            "2021": 95.26557318314417,
            "2022": 94.40332102286352,
            "2024": 90.77608732799749,
            "2025": 88.07037375825297,
        },
        "source_submission_sha256": digest(SOURCE / "submission.json"),
        "claim_limit": "One source-verified segment-column repair; leaderboard effect is unmeasured.",
    }
    write(BUILDING / "v221_segment_scope_batch5_audit.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = digest(OUTPUT / "submission.json")
    write(OUTPUT / "v221_segment_scope_batch5_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
