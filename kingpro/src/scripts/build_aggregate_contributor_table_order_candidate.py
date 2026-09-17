"""Build v195 by ranking source-derived aggregate contributors first.

The candidate starts from fully gated v194.  For unresolved mean/sum/count
panels, ``audit_aggregate_table_order.py`` evaluates the final aggregate
receiver and maps its contributing rows to ticker/year groups.  Only tables
belonging to those runtime-selected groups are moved forward.  Membership,
answers, queries, evidence and documents are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_aggregate_table_order import OUTPUT as AUDIT_OUTPUT  # noqa: E402
from audit_aggregate_table_order import audit as run_audit  # noqa: E402
from build_data_derived_table_order_candidate import _priority  # noqa: E402


SOURCE = ROOT / "sub_top123_candidate_v194_attribute_metric_table_order"
OUTPUT = ROOT / "sub_top123_candidate_v195_aggregate_contributor_order"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build() -> dict:
    if (OUTPUT / "aggregate_contributor_order_audit.json").exists():
        raise FileExistsError(f"refusing to overwrite completed candidate {OUTPUT}")

    diagnostic = run_audit()
    eligible = {
        int(item["id"]): item
        for item in diagnostic["findings"]
        if item.get("eligible")
    }
    if not eligible:
        raise AssertionError("no source-derived aggregate contributors found")

    # ``copytree`` may be interrupted by the Windows command runner while it
    # is copying ~2k evidence files.  A directory without submission.json is
    # an incomplete build owned by this builder and is safe to resume.
    shutil.copytree(SOURCE, OUTPUT, dirs_exist_ok=True)
    source_submission = SOURCE / "submission.json"
    output_submission = OUTPUT / "submission.json"
    source_rows = json.loads(source_submission.read_text(encoding="utf-8"))
    rows = json.loads(output_submission.read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    changes: list[dict] = []

    for row in rows:
        qid = int(row["id"])
        finding = eligible.get(qid)
        if finding is None:
            continue
        tickers = {str(value).upper() for value in finding["selected_tickers"]}
        years = {int(value) for value in finding["selected_years"]}
        if not tickers and not years:
            raise AssertionError(f"q{qid}: empty contributor group")

        original = list(row.get("relevant_tables") or [])
        selected_before = [
            index + 1
            for index, table_ref in enumerate(original)
            if _priority(table_ref, tickers, years) == 0
        ]
        if not selected_before:
            raise AssertionError(f"q{qid}: contributor group has no submitted table")
        ranked = sorted(
            enumerate(original),
            key=lambda pair: (_priority(pair[1], tickers, years), pair[0]),
        )
        reordered = [table_ref for _index, table_ref in ranked]
        if len(reordered) != len(original) or set(reordered) != set(original):
            raise AssertionError(f"q{qid}: relevant-table membership changed")
        if reordered == original:
            continue

        selected_after = [
            index + 1
            for index, table_ref in enumerate(reordered)
            if _priority(table_ref, tickers, years) == 0
        ]
        if tuple(selected_after) >= tuple(selected_before):
            raise AssertionError(f"q{qid}: contributor-rank vector did not improve")
        row["relevant_tables"] = reordered
        changes.append(
            {
                "id": qid,
                "aggregate": finding["aggregate"],
                "receiver": finding["receiver"],
                "selected_tickers": sorted(tickers),
                "selected_years": sorted(years),
                "source_names": finding["source_names"],
                "table_count": len(original),
                "first_contributor_rank_before": min(selected_before),
                "first_contributor_rank_after": min(selected_after),
                "contributor_ranks_before": selected_before,
                "contributor_ranks_after": selected_after,
                "contributors_in_top5_before": sum(rank <= 5 for rank in selected_before),
                "contributors_in_top5_after": sum(rank <= 5 for rank in selected_after),
                "membership_preserved": True,
            }
        )

    if not changes:
        raise AssertionError("aggregate contributors were already first")

    output_submission.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    changed_ids = {item["id"] for item in changes}
    for row in rows:
        qid = int(row["id"])
        old = source_by_id[qid]
        if qid not in changed_ids:
            if row != old:
                raise AssertionError(f"q{qid}: unexpected row change")
            continue
        old_without_order = dict(old)
        new_without_order = dict(row)
        old_tables = old_without_order.pop("relevant_tables", [])
        new_tables = new_without_order.pop("relevant_tables", [])
        if old_without_order != new_without_order:
            raise AssertionError(f"q{qid}: field other than relevant_tables changed")
        if len(old_tables) != len(new_tables) or set(old_tables) != set(new_tables):
            raise AssertionError(f"q{qid}: relevant-table membership changed")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "purpose": "source-derived aggregate contributor table ordering",
        "selection_rule": (
            "evaluate the receiver of the final mean/sum/count expression and "
            "map only its contributing Pandas rows to submitted ticker/year groups"
        ),
        "source_submission_sha256": _sha256(source_submission),
        "candidate_submission_sha256": _sha256(output_submission),
        "changed_question_count": len(changes),
        "changed_question_ids": sorted(changed_ids),
        "changes": changes,
        "invariants": {
            "answers_unchanged": True,
            "queries_unchanged": True,
            "evidence_unchanged": True,
            "relevant_docs_unchanged": True,
            "relevant_table_membership_unchanged": True,
            "only_relevant_table_order_changes": True,
            "question_ids_not_used_by_selection_rule": True,
            "leaderboard_feedback_not_used": True,
            "scalar_answer_reverse_matching_not_used": True,
            "v192_scored_champion_untouched": True,
        },
        "claim_limit": (
            "Source-grounded table-order hypothesis only. BTC metrics remain "
            "unknown until this exact artifact is evaluated."
        ),
    }
    (OUTPUT / "aggregate_contributor_order_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(AUDIT_OUTPUT, OUTPUT / "aggregate_table_order_diagnostic.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    os.chdir(ROOT)
    build()
