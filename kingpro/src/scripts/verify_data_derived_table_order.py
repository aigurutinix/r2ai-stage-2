"""Verify a data-derived ``relevant_tables`` ordering candidate.

This verifier is intentionally independent from the builder.  It replays every
panel query, derives the selected ticker/year again, proves that candidate rows
only reorder the existing table references, and rejects extracted variables
that are not on the static dependency path to ``result``.

The reported MRR@5 value is a local proxy: a table matching the query-selected
ticker/year is treated as relevant.  It is useful for regression testing, but
it is not a claim about the hidden BTC labels.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter
import json
from pathlib import Path
import sys

try:
    from .build_data_derived_table_order_candidate import (
        ROOT,
        _load_namespace,
        _priority,
        _selected_groups,
    )
except ImportError:  # direct ``python scripts/...`` execution
    from build_data_derived_table_order_candidate import (
        ROOT,
        _load_namespace,
        _priority,
        _selected_groups,
    )


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


DEFAULT_SOURCE = ROOT / "sub_top123_candidate_v191_eps_selection"
DEFAULT_CANDIDATE = ROOT / "sub_top123_candidate_v192_data_derived_table_order"
DEFAULT_REPORT = ROOT / "build" / "v192_data_derived_table_order_verification.json"


def _assigned_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        names: set[str] = set()
        for item in target.elts:
            names.update(_assigned_names(item))
        return names
    return set()


def _loaded_names(node: ast.AST | None) -> set[str]:
    if node is None:
        return set()
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _result_dependency_names(query: str) -> set[str]:
    """Return an over-approximated backwards slice feeding ``result``."""

    dependencies: dict[str, set[str]] = {}
    tree = ast.parse(query)
    for node in ast.walk(tree):
        targets: set[str] = set()
        value: ast.AST | None = None
        if isinstance(node, ast.Assign):
            for target in node.targets:
                targets.update(_assigned_names(target))
            value = node.value
        elif isinstance(node, ast.AnnAssign):
            targets.update(_assigned_names(node.target))
            value = node.value
        elif isinstance(node, ast.AugAssign):
            targets.update(_assigned_names(node.target))
            value = node.value
        if not targets:
            continue
        loaded = _loaded_names(value)
        for target in targets:
            dependencies.setdefault(target, set()).update(loaded)

    reachable = {"result"}
    pending = ["result"]
    while pending:
        current = pending.pop()
        for dependency in dependencies.get(current, set()):
            if dependency not in reachable:
                reachable.add(dependency)
                pending.append(dependency)
    return reachable


def _mrr5(rank: int) -> float:
    return 1.0 / rank if 1 <= rank <= 5 else 0.0


def _matching_rank(tables: list[str], tickers: set[str], years: set[int]) -> int:
    return min(
        index + 1
        for index, table_ref in enumerate(tables)
        if _priority(table_ref, tickers, years) == 0
    )


def verify(source: Path, candidate: Path) -> dict:
    source_rows = json.loads((source / "submission.json").read_text(encoding="utf-8"))
    candidate_rows = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    candidate_by_id = {int(row["id"]): row for row in candidate_rows}
    if list(source_by_id) != list(candidate_by_id):
        raise AssertionError("question ids or order changed")

    changed_ids: list[int] = []
    for qid, old in source_by_id.items():
        new = candidate_by_id[qid]
        old_other = dict(old)
        new_other = dict(new)
        old_tables = list(old_other.pop("relevant_tables", []))
        new_tables = list(new_other.pop("relevant_tables", []))
        if old_other != new_other:
            raise AssertionError(f"q{qid}: a field other than relevant_tables changed")
        if Counter(old_tables) != Counter(new_tables):
            raise AssertionError(f"q{qid}: relevant_tables membership changed")
        if old_tables != new_tables:
            changed_ids.append(qid)

    panel_audit = json.loads((source / "panel_source_audit.json").read_text(encoding="utf-8"))
    resolved: list[dict] = []
    unresolved: list[dict] = []
    dependency_failures: list[dict] = []

    for audit in panel_audit:
        qid = int(audit["id"])
        old = source_by_id[qid]
        old_tables = list(old.get("relevant_tables") or [])
        if len(old_tables) < 2:
            continue
        namespace = _load_namespace(source, old)
        tickers, years, extraction = _selected_groups(namespace, audit)
        if not tickers and not years:
            unresolved.append({"id": qid, "reason": "no_unambiguous_data_selected_group"})
            continue

        dependency_names = _result_dependency_names(old["pandas_query"])
        extracted_names = {
            part
            for part in extraction.split("+")
            if part and part not in {"single_panel_ticker", "single_panel_year"}
        }
        missing_dependencies = sorted(extracted_names - dependency_names)
        if missing_dependencies:
            dependency_failures.append(
                {
                    "id": qid,
                    "extraction": extraction,
                    "not_on_result_dependency_path": missing_dependencies,
                }
            )

        new_tables = list(candidate_by_id[qid].get("relevant_tables") or [])
        before = _matching_rank(old_tables, tickers, years)
        after = _matching_rank(new_tables, tickers, years)
        resolved.append(
            {
                "id": qid,
                "extraction": extraction,
                "selected_tickers": sorted(tickers),
                "selected_years": sorted(years),
                "rank_before": before,
                "rank_after": after,
                "mrr5_before": _mrr5(before),
                "mrr5_after": _mrr5(after),
                "changed": old_tables != new_tables,
            }
        )

    if dependency_failures:
        raise AssertionError(
            "extracted selector variables outside result dependency path: "
            + json.dumps(dependency_failures, ensure_ascii=False)
        )

    regressions = [item for item in resolved if item["rank_after"] > item["rank_before"]]
    if regressions:
        raise AssertionError(
            "selected-table rank regressions: "
            + json.dumps(regressions, ensure_ascii=False)
        )

    audited_changed_ids = sorted(item["id"] for item in resolved if item["changed"])
    if audited_changed_ids != sorted(changed_ids):
        raise AssertionError(
            f"changed ids do not match replay audit: rows={sorted(changed_ids)} "
            f"replay={audited_changed_ids}"
        )

    before_mrr5 = sum(item["mrr5_before"] for item in resolved) / len(resolved)
    after_mrr5 = sum(item["mrr5_after"] for item in resolved) / len(resolved)
    report = {
        "source": source.name,
        "candidate": candidate.name,
        "question_count": len(source_rows),
        "panel_rows_replayed": len(panel_audit),
        "resolved_panel_rows": len(resolved),
        "unresolved_panel_rows": len(unresolved),
        "changed_question_count": len(changed_ids),
        "changed_question_ids": sorted(changed_ids),
        "result_dependency_failures": dependency_failures,
        "selected_table_rank_regressions": regressions,
        "selected_table_proxy_mrr5_before": round(before_mrr5, 6),
        "selected_table_proxy_mrr5_after": round(after_mrr5, 6),
        "selected_table_proxy_mrr5_delta": round(after_mrr5 - before_mrr5, 6),
        "invariants": {
            "question_ids_and_order_unchanged": True,
            "all_non_table_fields_unchanged": True,
            "table_membership_and_multiplicity_unchanged": True,
            "only_table_order_changed": True,
            "all_extracted_variables_feed_result": True,
            "no_selected_table_rank_regression": True,
        },
        "claim_limit": (
            "The MRR@5 values are a source-derived selected-table proxy, not hidden "
            "BTC retrieval labels or a promised leaderboard score."
        ),
        "resolved": resolved,
        "unresolved": unresolved,
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    report = verify(args.source.resolve(), args.candidate.resolve())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {key: value for key, value in report.items() if key not in {"resolved", "unresolved"}}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"PASS: wrote {args.report.resolve()}")


if __name__ == "__main__":
    main()
