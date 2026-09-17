"""Build a v207 component replacing BID domestic loan provision with totals.

q967 asks for the *total* ending customer-loan provision.  Every v206 source
binding points to the first geographic component (Vietnam) and omits the next
foreign-market component.  Each source table has an exact following blank
total row.  This builder updates those five bindings and the materialized
answer only.  It is a component for a later multi-question bundle, not an
upload artifact.
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
OUTPUT = ROOT / "sub_top123_candidate_v207_q967_bid_total_provision_component_v2"
QID = 967
OLD_ANSWER = 39_850_765.0
NEW_ANSWER = 40_469_060.0
CORRECTIONS = {
    "2017": {"old_raw": "10.833.513", "raw": "11.349.782", "old_row_idx": "1", "row_idx": "3"},
    "2021": {"old_raw": "28.451.297", "raw": "29.103.718", "old_row_idx": "2", "row_idx": "4"},
    "2023": {"old_raw": "39.850.765", "raw": "40.469.060", "old_row_idx": "2", "row_idx": "4"},
    "2024": {"old_raw": "37.423.555", "raw": "38.038.771", "old_row_idx": "1", "row_idx": "3"},
    "2025": {"old_raw": "34.220.631", "raw": "34.945.553", "old_row_idx": "2", "row_idx": "4"},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def patch_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 5:
        raise AssertionError(f"unexpected q967 source-row count: {len(rows)}")
    changed: list[dict[str, str]] = []
    for row in rows:
        year = str(row.get("year"))
        correction = CORRECTIONS.get(year)
        if correction is None:
            raise AssertionError(f"unexpected q967 year: {year}")
        if row.get("metric_key") != "note:domestic_customer_loan_provision_million":
            raise AssertionError(f"unexpected q967 metric key: {row}")
        if row.get("raw") != correction["old_raw"] or row.get("row_idx") != correction["old_row_idx"]:
            raise AssertionError(f"unexpected q967 old binding for {year}: {row}")
        row["metric_key"] = "note:total_customer_loan_provision_million"
        row["raw"] = correction["raw"]
        row["row_idx"] = correction["row_idx"]
        changed.append(row.copy())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return changed


def patch_source_audit(path: Path, manifest_rows: list[dict[str, str]]) -> None:
    records = json.loads(path.read_text(encoding="utf-8"))
    record = next(item for item in records if int(item["id"]) == QID)
    if float(record.get("answer")) != OLD_ANSWER:
        raise AssertionError("unexpected q967 source-audit answer")
    manifest_by_table = {row["source_table"]: row for row in manifest_rows}
    sources = list(record.get("sources", []))
    if len(sources) != len(CORRECTIONS):
        raise AssertionError("unexpected q967 source-audit arity")
    for source in sources:
        manifest = manifest_by_table.get(source.get("table_ref"))
        if manifest is None:
            raise AssertionError(f"q967 source-audit table mismatch: {source}")
        if (
            str(source.get("raw")) != CORRECTIONS[str(manifest["year"])]["old_raw"]
            or int(source.get("row"))
            != int(CORRECTIONS[str(manifest["year"])]["old_row_idx"])
        ):
            raise AssertionError(f"q967 source-audit binding mismatch: {source}")
        source.update({
            "row": int(manifest["row_idx"]),
            "metric": manifest["metric_key"],
            "label": "Ending total provision for customer loans, VND million",
            "source_row_labels": ["Total"],
            "raw": manifest["raw"],
        })
    record["old_answer"] = OLD_ANSWER
    record["answer"] = NEW_ANSWER
    record["note"] = (
        "Maximum ending total BID customer-loan provision across the five "
        "requested years, including domestic and foreign-market components."
    )
    write_json(path, records)


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((OUTPUT / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(item["id"]): item for item in source_rows}
    row = next(item for item in rows if int(item["id"]) == QID)
    if float(row["answer"]) != OLD_ANSWER:
        raise AssertionError(f"unexpected q967 old answer: {row['answer']}")
    row["answer"] = NEW_ANSWER
    write_json(OUTPUT / "submission.json", rows)

    corrected_sources = patch_manifest(OUTPUT / "data" / "q967_source_cells.csv")
    patch_source_audit(OUTPUT / "source_audit.json", corrected_sources)

    changed_ids = [
        int(item["id"])
        for item in rows
        if item != source_by_id[int(item["id"])]
    ]
    if changed_ids != [QID]:
        raise AssertionError(f"unexpected changed submission IDs: {changed_ids}")
    changed_fields = {
        key for key in set(row) | set(source_by_id[QID])
        if row.get(key) != source_by_id[QID].get(key)
    }
    if changed_fields != {"answer"}:
        raise AssertionError(f"unexpected q967 changed fields: {sorted(changed_fields)}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": [QID],
        "old_answer": OLD_ANSWER,
        "new_answer": NEW_ANSWER,
        "selected_year_after_repair": 2023,
        "source_corrections": corrected_sources,
        "year_totals_million": {
            "2017": 11_349_782,
            "2021": 29_103_718,
            "2023": 40_469_060,
            "2024": 38_038_771,
            "2025": 34_945_553,
        },
        "invariants": {
            "only_q967_submission_row_changed": True,
            "only_materialized_answer_changed_in_submission_json": True,
            "query_unchanged": True,
            "relevant_tables_unchanged": True,
            "evidence_path_unchanged": True,
            "source_binding_rows_changed": 5,
            "source_audit_synchronized": True,
        },
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Exact-source component fix; hold for a multi-question bundle and full release gate.",
    }
    write_json(OUTPUT / "q967_bid_total_provision_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
