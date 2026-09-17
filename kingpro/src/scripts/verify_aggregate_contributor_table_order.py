"""Replay and verify the aggregate-contributor ordering candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_aggregate_table_order import audit as run_audit  # noqa: E402
from build_data_derived_table_order_candidate import _priority  # noqa: E402


DEFAULT_SOURCE = ROOT / "sub_top123_candidate_v194_attribute_metric_table_order"
DEFAULT_CANDIDATE = ROOT / "sub_top123_candidate_v195_aggregate_contributor_order"
DEFAULT_REPORT = ROOT / "build" / "v195_aggregate_contributor_order_verification.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _expected_order(row: dict[str, Any], finding: dict[str, Any]) -> list[str]:
    tickers = {str(value).upper() for value in finding["selected_tickers"]}
    years = {int(value) for value in finding["selected_years"]}
    original = list(row.get("relevant_tables") or [])
    return [
        table_ref
        for _index, table_ref in sorted(
            enumerate(original),
            key=lambda pair: (_priority(pair[1], tickers, years), pair[0]),
        )
    ]


def verify(source: Path, candidate: Path) -> dict[str, Any]:
    diagnostic = run_audit()
    eligible = {
        int(item["id"]): item
        for item in diagnostic["findings"]
        if item.get("eligible")
    }
    source_rows = json.loads((source / "submission.json").read_text(encoding="utf-8"))
    candidate_rows = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    if len(source_rows) != len(candidate_rows):
        raise AssertionError("submission row count changed")
    source_by_id = {int(row["id"]): row for row in source_rows}
    candidate_by_id = {int(row["id"]): row for row in candidate_rows}
    if source_by_id.keys() != candidate_by_id.keys():
        raise AssertionError("submission id set changed")

    expected_changed: set[int] = set()
    replay: list[dict[str, Any]] = []
    for qid, finding in eligible.items():
        source_row = source_by_id[qid]
        expected = _expected_order(source_row, finding)
        original = list(source_row.get("relevant_tables") or [])
        if expected == original:
            continue
        expected_changed.add(qid)
        tickers = {str(value).upper() for value in finding["selected_tickers"]}
        years = {int(value) for value in finding["selected_years"]}
        before = [
            index + 1
            for index, table_ref in enumerate(original)
            if _priority(table_ref, tickers, years) == 0
        ]
        after = [
            index + 1
            for index, table_ref in enumerate(expected)
            if _priority(table_ref, tickers, years) == 0
        ]
        if not before or tuple(after) >= tuple(before):
            raise AssertionError(f"q{qid}: contributor rank vector did not improve")
        replay.append(
            {
                "id": qid,
                "selected_tickers": sorted(tickers),
                "selected_years": sorted(years),
                "ranks_before": before,
                "ranks_after": after,
                "top5_before": sum(rank <= 5 for rank in before),
                "top5_after": sum(rank <= 5 for rank in after),
            }
        )

    actual_changed: set[int] = set()
    for qid, old in source_by_id.items():
        new = candidate_by_id[qid]
        if old == new:
            continue
        actual_changed.add(qid)
        old_without_order = dict(old)
        new_without_order = dict(new)
        old_tables = old_without_order.pop("relevant_tables", [])
        new_tables = new_without_order.pop("relevant_tables", [])
        if old_without_order != new_without_order:
            raise AssertionError(f"q{qid}: field other than relevant_tables changed")
        if len(old_tables) != len(new_tables) or set(old_tables) != set(new_tables):
            raise AssertionError(f"q{qid}: relevant-table membership/multiplicity changed")
        expected = _expected_order(old, eligible[qid])
        if list(new_tables) != expected:
            raise AssertionError(f"q{qid}: candidate order differs from replay")

    if actual_changed != expected_changed:
        raise AssertionError(
            f"changed ids do not match replay: rows={sorted(actual_changed)} "
            f"replay={sorted(expected_changed)}"
        )

    build_audit = json.loads(
        (candidate / "aggregate_contributor_order_audit.json").read_text(encoding="utf-8")
    )
    if set(int(value) for value in build_audit["changed_question_ids"]) != actual_changed:
        raise AssertionError("build audit changed ids do not match candidate")
    if build_audit["source_submission_sha256"] != _sha256(source / "submission.json"):
        raise AssertionError("source submission hash drifted")
    if build_audit["candidate_submission_sha256"] != _sha256(candidate / "submission.json"):
        raise AssertionError("candidate submission hash drifted")

    return {
        "source": source.name,
        "candidate": candidate.name,
        "source_submission_sha256": _sha256(source / "submission.json"),
        "candidate_submission_sha256": _sha256(candidate / "submission.json"),
        "changed_question_count": len(actual_changed),
        "changed_question_ids": sorted(actual_changed),
        "contributors_in_top5_before": sum(item["top5_before"] for item in replay),
        "contributors_in_top5_after": sum(item["top5_after"] for item in replay),
        "invariants": {
            "answers_unchanged": True,
            "queries_unchanged": True,
            "evidence_unchanged": True,
            "relevant_docs_unchanged": True,
            "relevant_table_membership_and_multiplicity_unchanged": True,
            "only_relevant_table_order_changed": True,
            "runtime_contributor_replay_matches_candidate": True,
            "all_changed_rank_vectors_improve": True,
        },
        "replay": replay,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = verify(args.source.resolve(), args.candidate.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in report.items() if key != "replay"}, ensure_ascii=False, indent=2))
    print(f"PASS: wrote {args.report.resolve()}")


if __name__ == "__main__":
    main()
