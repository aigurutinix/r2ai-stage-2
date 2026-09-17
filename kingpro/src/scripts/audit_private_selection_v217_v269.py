"""Prove the measured public tie and source-lineage delta of v217/v269."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V217 = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
V269 = ROOT / "sub_top123_candidate_v269_source_lineage_control"
SCORES = ROOT / "build/leaderboard_scores_authoritative_full_after_v269.json"
RELEASE = ROOT / "build/release_gate/sub_top123_candidate_v269_source_lineage_control/release.json"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def keyed(rows: list[dict]) -> dict[int, dict]:
    return {int(row["id"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build/v271_private_selection_v217_v269.json",
    )
    args = parser.parse_args()
    v217 = keyed(load(V217 / "submission.json"))
    v269 = keyed(load(V269 / "submission.json"))
    if sorted(v217) != list(range(1, 1013)) or sorted(v269) != list(range(1, 1013)):
        raise ValueError("candidate IDs are not exactly 1..1012")
    row_changes = []
    for qid in sorted(v217):
        if v217[qid] == v269[qid]:
            continue
        fields = sorted(
            key
            for key in set(v217[qid]) | set(v269[qid])
            if v217[qid].get(key) != v269[qid].get(key)
        )
        row_changes.append(
            {
                "id": qid,
                "fields": fields,
                "answer_before": v217[qid].get("answer"),
                "answer_after": v269[qid].get("answer"),
                "tables_before": v217[qid].get("relevant_tables"),
                "tables_after": v269[qid].get("relevant_tables"),
            }
        )
    source217 = keyed(load(V217 / "source_audit.json"))
    source269 = keyed(load(V269 / "source_audit.json"))
    source_changes = sorted(qid for qid in source217 if source217[qid] != source269[qid])
    panel217 = keyed(load(V217 / "panel_source_audit.json"))
    panel269 = keyed(load(V269 / "panel_source_audit.json"))
    panel_changes = sorted(qid for qid in panel217 if panel217[qid] != panel269[qid])

    submissions = {int(item["id"]): item for item in load(SCORES)["submissions"]}
    score217 = submissions[3696]["scores"]
    score269 = submissions[3741]["scores"]
    score_fields = [key for key, value in score217.items() if value is not None]
    public_tie = all(score217[key] == score269[key] for key in score_fields)
    release = load(RELEASE)
    protected = {
        "v206": sha(ROOT / "sub_top123_candidate_v206_semantic_batch11.zip"),
        "v207": sha(ROOT / "sub_top123_candidate_v207_semantic_batch6_final.zip"),
    }
    report = {
        "kind": "private_final_selection_v217_v269",
        "public_measurement": {
            "v217_submission_id": 3696,
            "v269_submission_id": 3741,
            "score_fields": score_fields,
            "v217": score217,
            "v269": score269,
            "all_ten_metrics_equal": public_tie,
        },
        "candidate_diff": {
            "row_change_count": len(row_changes),
            "row_changes": row_changes,
            "answer_change_ids": [
                item["id"]
                for item in row_changes
                if item["answer_before"] != item["answer_after"]
            ],
            "source_audit_change_ids": source_changes,
            "panel_source_audit_change_ids": panel_changes,
        },
        "v269_release": {
            "status": release.get("status"),
            "archive": release.get("archive"),
            "submission_sha256": sha(V269 / "submission.json"),
        },
        "protected_artifacts": protected,
        "decision": {
            "public_selected": "v217",
            "private_final_preference": "v269",
            "reason": (
                "V269 is public-score equivalent, passes the complete release gate, "
                "and replaces five source-lineage records with narrower physical evidence."
            ),
            "rollback": "v217",
            "automatic_leaderboard_selection": False,
        },
        "claim_limits": [
            "Public equality does not reveal private score.",
            "The private preference is a source-robustness decision, not a private-score claim.",
            "q24 physical evidence favors v269, but aggregate public constraints show all tried q24 variants are public-inactive.",
        ],
    }
    required = {
        "public_tie": public_tie,
        "expected_row_changes": [item["id"] for item in row_changes]
        == [24, 709, 826, 861],
        "expected_answer_change": report["candidate_diff"]["answer_change_ids"] == [24],
        "expected_source_changes": source_changes == [24, 61, 709, 826, 861],
        "panel_unchanged": not panel_changes,
        "release_pass": release.get("status") == "PASS",
        "v269_archive_hash": (release.get("archive") or {}).get("sha256")
        == "C933B5D901836C1414DAB801DAAA77BBA8654BC64572F208A8D8730959B935A0",
        "v206_locked": protected["v206"]
        == "34B19414AA1610785163513F8D2F16CCD41A1F797A90F7121817657F553E4B5B",
        "v207_locked": protected["v207"]
        == "DD616AA408E4B601B921246DDFF4DA4BFD4E8C283FCEF22FF68680AAAB2ADDAF",
    }
    report["gates"] = required
    report["status"] = "PASS" if all(required.values()) else "FAIL"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "public_tie": public_tie,
                "row_changes": [item["id"] for item in row_changes],
                "source_changes": source_changes,
                "decision": report["decision"],
                "output": str(args.out.resolve().relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
