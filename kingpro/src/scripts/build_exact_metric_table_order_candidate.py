"""Build v193 by resolving exact answer-row groups left untouched in v192.

The builder consumes the diagnostic audit produced by
``audit_unresolved_table_order.py``.  Only scalar selection/max/min expressions
with one exact derived-metric ticker/year are eligible.  It preserves every
submission field and every relevant-table membership; only stable list order
may change.

No question id, hidden label, answer override, score or leaderboard delta is
used by the selection rule.
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

from audit_unresolved_table_order import OUTPUT as DIAGNOSTIC_OUTPUT  # noqa: E402
from audit_unresolved_table_order import audit as run_diagnostic_audit  # noqa: E402
from build_data_derived_table_order_candidate import _priority  # noqa: E402


SOURCE = ROOT / "sub_top123_candidate_v192_data_derived_table_order"
OUTPUT = ROOT / "sub_top123_candidate_v193_exact_metric_table_order"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    diagnostic = run_diagnostic_audit()
    eligible = {
        int(item["id"]): item
        for item in diagnostic["findings"]
        if item["exact_metric_group"] is not None
    }
    if not eligible:
        raise AssertionError("no exact metric groups found; refusing empty candidate")

    shutil.copytree(SOURCE, OUTPUT)
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
        group = finding["exact_metric_group"]
        tickers = {str(value).upper() for value in group["tickers"]}
        years = {int(value) for value in group["years"]}
        if not tickers and not years:
            raise AssertionError(f"q{qid}: empty exact metric group")

        original = list(row.get("relevant_tables") or [])
        ranked = sorted(
            enumerate(original),
            key=lambda pair: (_priority(pair[1], tickers, years), pair[0]),
        )
        reordered = [table_ref for _index, table_ref in ranked]
        if set(reordered) != set(original) or len(reordered) != len(original):
            raise AssertionError(f"q{qid}: table membership changed")
        if reordered == original:
            continue

        selected_before = [
            index + 1
            for index, table_ref in enumerate(original)
            if _priority(table_ref, tickers, years) == 0
        ]
        selected_after = [
            index + 1
            for index, table_ref in enumerate(reordered)
            if _priority(table_ref, tickers, years) == 0
        ]
        if not selected_before or not selected_after:
            raise AssertionError(f"q{qid}: exact metric group has no submitted table")
        if min(selected_after) >= min(selected_before):
            raise AssertionError(f"q{qid}: rerank did not improve first selected rank")

        row["relevant_tables"] = reordered
        changes.append(
            {
                "id": qid,
                "result": finding["result"],
                "result_expression": finding["expression"],
                "selected_tickers": sorted(tickers),
                "selected_years": sorted(years),
                "metric_matches": finding["metric_matches"],
                "table_count": len(original),
                "first_selected_rank_before": min(selected_before),
                "first_selected_rank_after": min(selected_after),
                "membership_preserved": True,
            }
        )

    if not changes:
        raise AssertionError("all eligible groups were already first; refusing empty candidate")

    output_submission.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    changed_ids = {item["id"] for item in changes}
    for row in rows:
        qid = int(row["id"])
        old = source_by_id[qid]
        if qid not in changed_ids:
            if row != old:
                raise AssertionError(f"q{qid}: unexpected change")
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
        "purpose": "exact derived-metric answer-row table ordering",
        "selection_rule": (
            "final result is scalar selection/max/min, rounded result matches one "
            "derived metric ticker/year, and no mean/sum/count reverse matching"
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
            "v192_scored_champion_untouched": True,
        },
        "claim_limit": (
            "Source-grounded table-order hypothesis only. BTC metrics are unknown "
            "unless this exact artifact is separately evaluated."
        ),
    }
    (OUTPUT / "exact_metric_table_order_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    # Keep the standalone diagnostic beside the build for reproducibility.
    shutil.copy2(DIAGNOSTIC_OUTPUT, OUTPUT / "unresolved_table_order_diagnostic.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    os.chdir(ROOT)
    build()
