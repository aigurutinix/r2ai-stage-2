"""Merge the six source-verified v207 repairs over the locked v206 champion.

Each repair was first built and audited in an isolated component directory.
This builder deliberately copies only the selected submission row, its compact
source manifest, and any matching audit record.  It fails closed if a component
changes another submission row or if the merged candidate differs outside the
six whitelisted question IDs.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v206_semantic_batch11"
OUTPUT = ROOT / "sub_top123_candidate_v207_semantic_batch6_final"

COMPONENTS = {
    426: ROOT / "sub_top123_candidate_v207_q426_fpt_cfo_period_component",
    617: ROOT / "sub_top123_candidate_v207_total_capital_semantics_component",
    673: ROOT / "sub_top123_candidate_v207_total_capital_semantics_component",
    853: ROOT / "sub_top123_candidate_v207_total_capital_semantics_component",
    967: ROOT / "sub_top123_candidate_v207_q967_bid_total_provision_component_v2",
    971: ROOT / "sub_top123_candidate_v207_q971_semantic_child_table_component",
}
ANSWER_CHANGED_IDS = [426, 617, 967]
PROVENANCE_ONLY_IDS = [673, 853, 971]
EXPECTED_FIELDS = {
    426: {"answer"},
    617: {"answer", "relevant_tables"},
    673: {"relevant_tables"},
    853: {"relevant_tables"},
    967: {"answer"},
    971: {"relevant_tables"},
}
EXPECTED_ANSWERS = {
    426: 1.44,
    617: 10.34,
    673: 39.19,
    853: 66.39,
    967: 40_469_060.0,
    971: 2022.0,
}


def load_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def rows_by_id(rows: list[dict]) -> dict[int, dict]:
    result = {int(row["id"]): row for row in rows}
    if len(result) != len(rows):
        raise AssertionError("duplicate question IDs")
    return result


def changed_fields(before: dict, after: dict) -> set[str]:
    return {
        key
        for key in set(before) | set(after)
        if before.get(key) != after.get(key)
    }


def replace_audit_record(path: Path, component_path: Path, qid: int) -> bool:
    """Replace qid in an audit list only when both artifacts contain it."""

    if not path.is_file() or not component_path.is_file():
        return False
    records = load_json(path)
    component_records = load_json(component_path)
    if not isinstance(records, list) or not isinstance(component_records, list):
        raise AssertionError(f"audit is not a list: {path.name}")
    positions = [index for index, row in enumerate(records) if int(row["id"]) == qid]
    component_matches = [row for row in component_records if int(row["id"]) == qid]
    if not positions and not component_matches:
        return False
    if len(positions) != 1 or len(component_matches) != 1:
        raise AssertionError(f"q{qid}: inconsistent {path.name} membership")
    records[positions[0]] = component_matches[0]
    write_json(path, records)
    return True


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    for component in set(COMPONENTS.values()):
        if not component.is_dir():
            raise FileNotFoundError(component)

    source_rows = load_json(SOURCE / "submission.json")
    if not isinstance(source_rows, list):
        raise AssertionError("source submission is not a list")
    source_by_id = rows_by_id(source_rows)
    merged_rows = json.loads(json.dumps(source_rows, ensure_ascii=False))
    merged_by_id = rows_by_id(merged_rows)

    component_cache: dict[Path, tuple[list[dict], dict[int, dict]]] = {}
    component_reports = {}
    for qid, component in COMPONENTS.items():
        if component not in component_cache:
            component_rows = load_json(component / "submission.json")
            if not isinstance(component_rows, list):
                raise AssertionError(f"component submission is not a list: {component}")
            component_cache[component] = (component_rows, rows_by_id(component_rows))
        component_rows, component_by_id = component_cache[component]

        component_changed = [
            int(row["id"])
            for row in component_rows
            if row != source_by_id[int(row["id"])]
        ]
        allowed_component_ids = sorted(
            candidate_qid
            for candidate_qid, candidate_component in COMPONENTS.items()
            if candidate_component == component
        )
        if not component_changed or not set(component_changed) <= set(allowed_component_ids):
            raise AssertionError(
                f"{component.name}: unexpected changed IDs {component_changed}"
            )

        fields = changed_fields(source_by_id[qid], component_by_id[qid])
        if fields != EXPECTED_FIELDS[qid]:
            raise AssertionError(f"q{qid}: unexpected component fields {sorted(fields)}")
        if float(component_by_id[qid]["answer"]) != EXPECTED_ANSWERS[qid]:
            raise AssertionError(f"q{qid}: unexpected component answer")
        merged_by_id[qid].clear()
        merged_by_id[qid].update(component_by_id[qid])

        manifest = component / "data" / f"q{qid}_source_cells.csv"
        if not manifest.is_file():
            raise FileNotFoundError(manifest)
        component_reports[str(qid)] = {
            "component": component.name,
            "component_submission_sha256": sha256(component / "submission.json"),
            "manifest_sha256": sha256(manifest),
            "changed_fields": sorted(fields),
        }

    shutil.copytree(SOURCE, OUTPUT)
    write_json(OUTPUT / "submission.json", merged_rows)

    synchronized_audits = {"source_audit.json": [], "panel_source_audit.json": []}
    for qid, component in COMPONENTS.items():
        shutil.copy2(
            component / "data" / f"q{qid}_source_cells.csv",
            OUTPUT / "data" / f"q{qid}_source_cells.csv",
        )
        for audit_name in synchronized_audits:
            if replace_audit_record(
                OUTPUT / audit_name,
                component / audit_name,
                qid,
            ):
                synchronized_audits[audit_name].append(qid)

    final_rows = load_json(OUTPUT / "submission.json")
    final_by_id = rows_by_id(final_rows)
    changed_ids = [
        int(row["id"])
        for row in final_rows
        if row != source_by_id[int(row["id"])]
    ]
    expected_ids = sorted(COMPONENTS)
    if changed_ids != expected_ids:
        raise AssertionError(
            f"unexpected merged submission IDs {changed_ids}; expected {expected_ids}"
        )
    for qid in expected_ids:
        fields = changed_fields(source_by_id[qid], final_by_id[qid])
        if fields != EXPECTED_FIELDS[qid]:
            raise AssertionError(f"q{qid}: unexpected merged fields {sorted(fields)}")
        if float(final_by_id[qid]["answer"]) != EXPECTED_ANSWERS[qid]:
            raise AssertionError(f"q{qid}: unexpected merged answer")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": expected_ids,
        "answer_changed_ids": ANSWER_CHANGED_IDS,
        "provenance_only_ids": PROVENANCE_ONLY_IDS,
        "components": component_reports,
        "synchronized_audits": synchronized_audits,
        "invariants": {
            "exactly_six_submission_rows_changed": True,
            "all_changed_fields_whitelisted": True,
            "all_six_manifests_copied_from_isolated_components": True,
            "all_other_submission_rows_identical_to_locked_v206": True,
            "questions_unchanged": True,
            "relevant_docs_unchanged": True,
            "evidence_paths_unchanged": True,
            "pandas_queries_unchanged": True,
        },
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "candidate_submission_sha256": sha256(OUTPUT / "submission.json"),
        "claim_limit": "Six source-verified changes; leaderboard effect remains unmeasured.",
    }
    write_json(OUTPUT / "v207_semantic_batch6_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
