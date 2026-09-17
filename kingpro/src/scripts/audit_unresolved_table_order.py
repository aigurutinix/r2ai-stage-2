"""Audit unresolved v192 panels without using question ids or score feedback.

The audit replays each submitted program and inspects only the expression that
produces ``result``.  It reports a selected ticker/year only when either:

* the final expression references a frame whose entity/year is uniform; or
* a scalar selection/max/min equals one derived metric cell in one entity/year.

Mean, sum and count-like outputs are deliberately not reverse-matched to a
single row.  This script is diagnostic: it never edits a submission.
"""

from __future__ import annotations

import ast
import json
import math
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

from build_data_derived_table_order_candidate import _load_namespace  # noqa: E402


CANDIDATE = ROOT / "sub_top123_candidate_v192_data_derived_table_order"
VERIFICATION = ROOT / "build" / "v192_data_derived_table_order_verification.json"
OUTPUT = ROOT / "build" / "v192_unresolved_table_order_audit.json"

_AGGREGATE_METHODS = {"mean", "sum", "count", "nunique", "size"}
_SELECT_METHODS = {"max", "min", "idxmax", "idxmin"}


def _result_expression(query: str) -> ast.AST:
    tree = ast.parse(query)
    assignments = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "result" for target in node.targets)
    ]
    if not assignments:
        raise ValueError("query has no result assignment")
    # Builders finish with result = round(float(result), 2).  The preceding
    # assignment contains the actual selection/aggregation expression.
    if len(assignments) > 1:
        return assignments[-2]
    return assignments[-1]


def _expression_features(expression: ast.AST) -> dict[str, Any]:
    names = sorted({node.id for node in ast.walk(expression) if isinstance(node, ast.Name)})
    methods = sorted(
        {
            node.attr
            for node in ast.walk(expression)
            if isinstance(node, ast.Attribute)
        }
    )
    string_selectors = sorted(
        {
            node.value
            for node in ast.walk(expression)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
    )
    integer_literals = sorted(
        {
            int(node.value)
            for node in ast.walk(expression)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        }
    )
    return {
        "names": names,
        "methods": methods,
        "string_selectors": string_selectors,
        "integer_literals": integer_literals,
        "aggregate": bool(_AGGREGATE_METHODS.intersection(methods)),
        "selection": bool(_SELECT_METHODS.intersection(methods))
        or "values" in methods
        or "loc" in methods
        or "iloc" in methods,
    }


def _normalise_ticker(value: Any, valid: set[str]) -> str | None:
    ticker = str(value).upper()
    return ticker if ticker in valid else None


def _normalise_year(value: Any, valid: set[int]) -> int | None:
    try:
        year = int(float(value))
    except (TypeError, ValueError):
        return None
    return year if year in valid else None


def _frame_uniform_groups(
    frame: pd.DataFrame,
    valid_tickers: set[str],
    valid_years: set[int],
) -> tuple[set[str], set[int]]:
    tickers: set[str] = set()
    years: set[int] = set()
    ticker_columns = [column for column in frame.columns if str(column).startswith("ticker")]
    year_columns = [column for column in frame.columns if str(column).startswith("year")]

    for column in ticker_columns:
        values = {
            ticker
            for ticker in (_normalise_ticker(value, valid_tickers) for value in frame[column])
            if ticker is not None
        }
        if len(values) == 1:
            tickers.update(values)
    for column in year_columns:
        values = {
            year
            for year in (_normalise_year(value, valid_years) for value in frame[column])
            if year is not None
        }
        if len(values) == 1:
            years.update(values)

    index_tickers = {
        ticker
        for ticker in (_normalise_ticker(value, valid_tickers) for value in frame.index)
        if ticker is not None
    }
    if len(index_tickers) == 1:
        tickers.update(index_tickers)
    return tickers, years


def _matching_metric_groups(
    namespace: dict[str, Any],
    features: dict[str, Any],
    valid_tickers: set[str],
    valid_years: set[int],
) -> list[dict[str, Any]]:
    if features["aggregate"] or not features["selection"]:
        return []
    try:
        target = round(float(namespace["result"]), 2)
    except (KeyError, TypeError, ValueError):
        return []
    if not math.isfinite(target):
        return []

    # Pandas permits both frame["metric"] and frame.metric.  Attribute names
    # are safe candidates here because a match still needs a numeric column,
    # an exact rounded result and one unambiguous ticker/year group.
    selectors = set(features["string_selectors"]) | set(features["methods"])
    matches: list[dict[str, Any]] = []
    for name, value in namespace.items():
        if not isinstance(value, pd.DataFrame) or name.startswith("_"):
            continue
        for column in value.columns:
            label = str(column)
            base_label = label.rsplit("_", 1)[0] if label.rsplit("_", 1)[-1].isdigit() else label
            if label not in selectors and base_label not in selectors:
                continue
            numeric = pd.to_numeric(value[column], errors="coerce")
            for index, cell in numeric.items():
                if pd.isna(cell) or round(float(cell), 2) != target:
                    continue
                row = value.loc[index]
                ticker: str | None = None
                year: int | None = None
                for ticker_column in [c for c in value.columns if str(c).startswith("ticker")]:
                    ticker = _normalise_ticker(row[ticker_column], valid_tickers) or ticker
                for year_column in [c for c in value.columns if str(c).startswith("year")]:
                    year = _normalise_year(row[year_column], valid_years) or year
                if ticker is None:
                    ticker = _normalise_ticker(index, valid_tickers)
                suffix = label.rsplit("_", 1)[-1]
                if suffix.isdigit():
                    year = _normalise_year(suffix, valid_years) or year
                matches.append(
                    {
                        "namespace": name,
                        "column": label,
                        "ticker": ticker,
                        "year": year,
                    }
                )
    unique = {
        (item["column"], item["ticker"], item["year"]): item
        for item in matches
        if item["ticker"] is not None or item["year"] is not None
    }
    return sorted(unique.values(), key=lambda item: (str(item["ticker"]), int(item["year"] or 0), item["column"]))


def audit() -> dict[str, Any]:
    rows = json.loads((CANDIDATE / "submission.json").read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    panels = json.loads((CANDIDATE / "panel_source_audit.json").read_text(encoding="utf-8"))
    audit_by_id = {int(item["id"]): item for item in panels}
    unresolved = json.loads(VERIFICATION.read_text(encoding="utf-8"))["unresolved"]
    findings: list[dict[str, Any]] = []

    for unresolved_item in unresolved:
        qid = int(unresolved_item["id"])
        row = by_id[qid]
        panel = audit_by_id[qid]
        valid_tickers = {str(value).upper() for value in panel.get("tickers", [])}
        valid_years = {int(value) for value in panel.get("years", [])}
        namespace = _load_namespace(CANDIDATE, row)
        expression = _result_expression(row["pandas_query"])
        features = _expression_features(expression)

        uniform_sources: list[dict[str, Any]] = []
        for name in features["names"]:
            value = namespace.get(name)
            if not isinstance(value, pd.DataFrame):
                continue
            tickers, years = _frame_uniform_groups(value, valid_tickers, valid_years)
            if tickers or years:
                uniform_sources.append(
                    {"namespace": name, "tickers": sorted(tickers), "years": sorted(years)}
                )

        literal_years = sorted(valid_years.intersection(features["integer_literals"]))
        metric_matches = _matching_metric_groups(
            namespace, features, valid_tickers, valid_years
        )
        matched_groups = {
            (item["ticker"], item["year"])
            for item in metric_matches
            if item["ticker"] is not None or item["year"] is not None
        }
        exact_group = None
        if len(matched_groups) == 1:
            ticker, year = next(iter(matched_groups))
            exact_group = {
                "tickers": [ticker] if ticker else [],
                "years": [year] if year else [],
            }

        findings.append(
            {
                "id": qid,
                "result": float(namespace["result"]),
                "expression": ast.unparse(expression),
                "features": features,
                "uniform_sources": uniform_sources,
                "literal_years": literal_years,
                "metric_matches": metric_matches,
                "exact_metric_group": exact_group,
                "candidate_for_conservative_rerank": exact_group is not None,
            }
        )

    report = {
        "candidate": CANDIDATE.name,
        "method": "final-expression runtime inspection; no question-id or score labels",
        "unresolved_count": len(findings),
        "exact_metric_group_count": sum(
            bool(item["exact_metric_group"]) for item in findings
        ),
        "exact_metric_group_ids": [
            item["id"] for item in findings if item["exact_metric_group"]
        ],
        "findings": findings,
        "claim_limit": (
            "Diagnostic only. A candidate may be built only after every proposed group "
            "is independently checked against the final expression and table membership."
        ),
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "findings"}, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    audit()
