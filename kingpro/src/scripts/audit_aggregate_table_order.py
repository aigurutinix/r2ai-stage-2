"""Find source-derived contributor groups for unresolved aggregate panels.

This audit complements ``audit_unresolved_table_order.py``.  It never infers a
winner from a scalar answer.  Instead, it evaluates the receiver of the final
Pandas aggregate (for example the Series before ``.mean()`` or ``.sum()``)
and maps the contributing indices back to submitted panel rows.  The output is
diagnostic only; it does not edit a candidate.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from audit_unresolved_table_order import _result_expression  # noqa: E402
from build_data_derived_table_order_candidate import _load_namespace  # noqa: E402


CANDIDATE = ROOT / "sub_top123_candidate_v194_attribute_metric_table_order"
DIAGNOSTIC = CANDIDATE / "unresolved_table_order_diagnostic.json"
OUTPUT = ROOT / "build" / "v194_aggregate_table_order_audit.json"
AGGREGATES = {"mean", "sum", "count", "nunique", "size"}


def _normalise_ticker(value: Any, valid: set[str]) -> str | None:
    ticker = str(value).upper()
    return ticker if ticker in valid else None


def _normalise_year(value: Any, valid: set[int]) -> int | None:
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    return year if year in valid else None


def _receiver(expression: ast.AST) -> tuple[ast.AST | None, str]:
    """Return the object whose rows contribute to the final aggregate."""

    for node in ast.walk(expression):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr in AGGREGATES:
            return node.func.value, node.func.attr

    # Count programs often finish as ``filtered.shape[0]``.
    if (
        isinstance(expression, ast.Subscript)
        and isinstance(expression.value, ast.Attribute)
        and expression.value.attr == "shape"
    ):
        return expression.value.value, "shape"
    return None, ""


def _selected_index(value: Any) -> tuple[pd.Index | None, str]:
    if isinstance(value, pd.DataFrame):
        return value.index, "dataframe"
    if isinstance(value, pd.Series):
        if pd.api.types.is_bool_dtype(value.dtype):
            return value[value.fillna(False)].index, "boolean_series"
        return value.dropna().index, "series"
    return None, type(value).__name__


def _groups_from_frame(
    frame: pd.DataFrame,
    selected_index: pd.Index,
    valid_tickers: set[str],
    valid_years: set[int],
) -> tuple[set[str], set[int]]:
    if not selected_index.isin(frame.index).all():
        return set(), set()
    selected = frame.loc[selected_index]
    if isinstance(selected, pd.Series):
        selected = selected.to_frame().T

    tickers: set[str] = set()
    years: set[int] = set()
    for column in selected.columns:
        label = str(column)
        if label.startswith("ticker"):
            tickers.update(
                ticker
                for ticker in (
                    _normalise_ticker(value, valid_tickers) for value in selected[column]
                )
                if ticker is not None
            )
        if label.startswith("year"):
            years.update(
                year
                for year in (
                    _normalise_year(value, valid_years) for value in selected[column]
                )
                if year is not None
            )

    if not tickers:
        tickers.update(
            ticker
            for ticker in (
                _normalise_ticker(value, valid_tickers) for value in selected.index
            )
            if ticker is not None
        )
    return tickers, years


def _groups_from_namespace(
    namespace: dict[str, Any],
    receiver_value: Any,
    valid_tickers: set[str],
    valid_years: set[int],
) -> tuple[set[str], set[int], list[str], str]:
    selected_index, value_kind = _selected_index(receiver_value)
    tickers: set[str] = set()
    years: set[int] = set()
    sources: list[str] = []

    if isinstance(receiver_value, pd.DataFrame) and selected_index is not None:
        frame_tickers, frame_years = _groups_from_frame(
            receiver_value, selected_index, valid_tickers, valid_years
        )
        tickers.update(frame_tickers)
        years.update(frame_years)
        if frame_tickers or frame_years:
            sources.append("receiver")

    if selected_index is not None:
        tickers.update(
            ticker
            for ticker in (
                _normalise_ticker(value, valid_tickers) for value in selected_index
            )
            if ticker is not None
        )
        for name, value in namespace.items():
            if name.startswith("_") or not isinstance(value, pd.DataFrame):
                continue
            frame_tickers, frame_years = _groups_from_frame(
                value, selected_index, valid_tickers, valid_years
            )
            if frame_tickers or frame_years:
                tickers.update(frame_tickers)
                years.update(frame_years)
                sources.append(name)

    # Named masks and entity collections preserve provenance even when a
    # derived Series has lost its original DataFrame columns.
    for name in ("keep", "common", "positive"):
        if tickers:
            break
        value = namespace.get(name)
        values: list[Any] = []
        if isinstance(value, pd.Series):
            if pd.api.types.is_bool_dtype(value.dtype):
                values = list(value[value.fillna(False)].index)
            else:
                values = list(value.dropna().index)
        elif isinstance(value, (pd.Index, list, tuple, set)):
            values = list(value)
        found = {
            ticker
            for ticker in (
                _normalise_ticker(item, valid_tickers) for item in values
            )
            if ticker is not None
        }
        if found:
            tickers.update(found)
            sources.append(name)

    return tickers, years, list(dict.fromkeys(sources)), value_kind


def audit() -> dict[str, Any]:
    rows = json.loads((CANDIDATE / "submission.json").read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    panels = json.loads((CANDIDATE / "panel_source_audit.json").read_text(encoding="utf-8"))
    panel_by_id = {int(item["id"]): item for item in panels}
    unresolved = json.loads(DIAGNOSTIC.read_text(encoding="utf-8"))["findings"]

    findings: list[dict[str, Any]] = []
    for item in unresolved:
        if item.get("exact_metric_group") is not None:
            continue
        qid = int(item["id"])
        row = by_id[qid]
        panel = panel_by_id[qid]
        valid_tickers = {str(value).upper() for value in panel.get("tickers", [])}
        valid_years = {int(value) for value in panel.get("years", [])}
        namespace = _load_namespace(CANDIDATE, row)
        expression = _result_expression(row["pandas_query"])
        receiver, aggregate = _receiver(expression)
        if receiver is None:
            findings.append(
                {
                    "id": qid,
                    "eligible": False,
                    "reason": "no_final_aggregate_receiver",
                    "expression": ast.unparse(expression),
                }
            )
            continue

        receiver_value = eval(  # noqa: S307 - submitted code already ran in same constrained namespace
            compile(ast.Expression(receiver), f"<q{qid}-aggregate>", "eval"),
            namespace,
        )
        tickers, years, sources, value_kind = _groups_from_namespace(
            namespace, receiver_value, valid_tickers, valid_years
        )
        literal_years = set(int(value) for value in item.get("literal_years", []))
        years.update(valid_years.intersection(literal_years))
        eligible = bool(tickers or years)
        findings.append(
            {
                "id": qid,
                "eligible": eligible,
                "aggregate": aggregate,
                "receiver": ast.unparse(receiver),
                "receiver_kind": value_kind,
                "selected_tickers": sorted(tickers),
                "selected_years": sorted(years),
                "source_names": sources,
                "valid_ticker_count": len(valid_tickers),
                "valid_year_count": len(valid_years),
                "table_count": len(row.get("relevant_tables") or []),
                "expression": ast.unparse(expression),
                "reason": "source_derived_contributors" if eligible else "contributors_not_mappable",
            }
        )

    report = {
        "candidate": CANDIDATE.name,
        "method": "runtime aggregate receiver mapped to submitted panel rows",
        "checked_count": len(findings),
        "eligible_count": sum(bool(item.get("eligible")) for item in findings),
        "eligible_ids": [item["id"] for item in findings if item.get("eligible")],
        "findings": findings,
        "guardrails": {
            "question_ids_not_used_for_selection": True,
            "leaderboard_feedback_not_used": True,
            "scalar_answer_reverse_matching_not_used": True,
            "diagnostic_only": True,
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "findings"}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    audit()
