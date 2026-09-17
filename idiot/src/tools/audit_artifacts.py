"""Audit the table index and financial panel before a solver run."""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from road2ai_vifinqa.text import parse_vn_number  # noqa: E402


COST_PANEL_KEYS = frozenset(f"kqkd:{code}" for code in ("11", "22", "23", "25", "26", "32", "51", "52"))


def audit(index_path: Path, panel_path: Path, *, integrity: bool = False) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    warnings: list[dict[str, object]] = []
    conn = sqlite3.connect(f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        counts = {
            name: int(conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0])
            for name in ("documents", "tables", "rows")
        }
        if integrity:
            result = str(conn.execute("PRAGMA integrity_check").fetchone()[0])
            if result != "ok":
                errors.append({"code": "index_integrity", "detail": result})
        table_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(tables)")}
        row_columns = {str(row[1]) for row in conn.execute("PRAGMA table_info(rows)")}
        semantic_schema = {
            "statement_kind", "unit_scale", "header_rows_json"
        }.issubset(table_columns) and {"row_label", "section"}.issubset(row_columns)
        panel: dict[str, dict[str, dict[str, dict[str, object]]]] = json.loads(
            panel_path.read_text(encoding="utf-8")
        )
        cells = 0
        period_roles: Counter[str] = Counter()
        low_margin = 0
        source_cache: dict[tuple[str, int], list[list[str]]] = {}
        doc_cache: dict[str, sqlite3.Row] = {}
        for ticker, years in panel.items():
            for raw_year, metrics in years.items():
                year = int(raw_year)
                for metric, item in metrics.items():
                    cells += 1
                    doc_id = str(item.get("doc_id", ""))
                    table_id = int(item.get("table_id", 0))
                    row_idx = int(item.get("row_idx", -1))
                    col_idx = int(item.get("col_idx", -1))
                    if doc_id not in doc_cache:
                        row = conn.execute(
                            "SELECT ticker, report_year FROM documents WHERE doc_id=?", (doc_id,)
                        ).fetchone()
                        if row is not None:
                            doc_cache[doc_id] = row
                    doc = doc_cache.get(doc_id)
                    if doc is None:
                        errors.append({"code": "missing_doc", "ticker": ticker, "year": year, "metric": metric})
                        continue
                    if str(doc["ticker"]) != ticker or int(doc["report_year"]) != year:
                        errors.append(
                            {
                                "code": "entity_period_mismatch",
                                "ticker": ticker,
                                "year": year,
                                "metric": metric,
                                "doc_id": doc_id,
                            }
                        )
                    source_key = (doc_id, table_id)
                    if source_key not in source_cache:
                        row = conn.execute(
                            "SELECT rows_json FROM tables WHERE doc_id=? AND table_id=?", source_key
                        ).fetchone()
                        if row is not None:
                            source_cache[source_key] = json.loads(row["rows_json"])
                    rows = source_cache.get(source_key)
                    if rows is None or row_idx < 0 or row_idx >= len(rows) or col_idx < 0 or col_idx >= len(rows[row_idx]):
                        errors.append(
                            {
                                "code": "invalid_coordinate",
                                "ticker": ticker,
                                "year": year,
                                "metric": metric,
                                "source": [doc_id, table_id, row_idx, col_idx],
                            }
                        )
                        continue
                    raw = rows[row_idx][col_idx]
                    if str(item.get("raw", "")) != raw:
                        errors.append(
                            {
                                "code": "raw_mismatch",
                                "ticker": ticker,
                                "year": year,
                                "metric": metric,
                                "stored": item.get("raw"),
                                "source": raw,
                            }
                        )
                    number = parse_vn_number(raw)
                    if number is None:
                        errors.append({"code": "non_numeric_source", "ticker": ticker, "year": year, "metric": metric})
                        continue
                    expected = number * float(item.get("scale", 1.0))
                    if metric in COST_PANEL_KEYS:
                        expected = abs(expected)
                    if not math.isclose(expected, float(item["value"]), rel_tol=1e-12, abs_tol=1e-6):
                        errors.append(
                            {
                                "code": "value_replay_mismatch",
                                "ticker": ticker,
                                "year": year,
                                "metric": metric,
                                "stored": item["value"],
                                "replayed": expected,
                            }
                        )
                    role = str(item.get("period_role", "legacy"))
                    period_roles[role] += 1
                    period_year = item.get("period_year")
                    if period_year is not None and int(period_year) != year:
                        warnings.append(
                            {
                                "code": "period_year_mismatch",
                                "ticker": ticker,
                                "year": year,
                                "metric": metric,
                                "period_year": period_year,
                            }
                        )
                    margin = item.get("selection_margin")
                    if isinstance(margin, (int, float)) and float(margin) < 1.0:
                        low_margin += 1
    finally:
        conn.close()
    return {
        "schema_version": 1,
        "ok": not errors,
        "index": {
            "path": str(index_path.resolve()),
            "counts": counts,
            "semantic_schema": semantic_schema,
        },
        "panel": {
            "path": str(panel_path.resolve()),
            "cells": cells,
            "period_roles": dict(period_roles),
            "low_margin_selections": low_margin,
        },
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--integrity", action="store_true")
    args = parser.parse_args()
    result = audit(args.index, args.panel, integrity=args.integrity)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": result["ok"],
                "errors": len(result["errors"]),
                "warnings": len(result["warnings"]),
                "index": result["index"]["counts"],
                "panel_cells": result["panel"]["cells"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
