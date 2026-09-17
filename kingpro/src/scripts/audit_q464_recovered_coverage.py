"""Audit q464 after recovering missing consolidated statement cells.

The script is deliberately read-only with respect to a submission.  It fills only
cells that are absent from v217, preserves every source candidate, and refuses to
choose among conflicting recovered values.  Its output is evidence for review,
not mutation authority.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V217 = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
BASE_CSV = V217 / "data" / "q464_source_cells.csv"
RECOVERED = ROOT / "build" / "v227_q464_missing_pairs_fullscan.jsonl"
OUTPUT = ROOT / "build" / "v227_q464_recovered_coverage_audit.json"

REQUIRED = ("cdkt:140", "kqkd:10", "lctt:20")
RECOVERY_KEYS = REQUIRED + ("kqkd:01",)
METRIC_NAME = {
    "cdkt:140": "inventory",
    "kqkd:10": "revenue",
    "lctt:20": "cfo",
}


def parse_vietnamese_number(raw: str) -> float | None:
    text = str(raw or "").strip()
    if not text or text in {"-", "--", "None", "nan"}:
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.replace(".", "").replace(",", ".").replace(" ", "")
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def load_base() -> tuple[dict[tuple[str, int, str], dict], list[dict]]:
    chosen: dict[tuple[str, int, str], dict] = {}
    duplicates: list[dict] = []
    with BASE_CSV.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["ticker"], int(row["year"]), row["metric_key"])
            record = {
                "ticker": key[0],
                "year": key[1],
                "metric_key": key[2],
                "metric": METRIC_NAME.get(key[2]),
                "value": parse_vietnamese_number(row.get("raw", "")),
                "raw": row.get("raw"),
                "label": None,
                "table_ref": row.get("source_table"),
                "csv_path": row.get("source_csv"),
                "row_idx": row.get("row_idx"),
                "col_idx": row.get("col_idx"),
                "scope": "consolidated",
                "origin": "v217",
                "typed_factor": float(row.get("typed_factor") or 1),
                "scale": float(row.get("scale") or 1),
            }
            if key in chosen:
                duplicates.append({"key": list(key), "records": [chosen[key], record]})
            else:
                chosen[key] = record
    return chosen, duplicates


def load_recovered() -> dict[tuple[str, int, str], list[dict]]:
    grouped: dict[tuple[str, int, str], list[dict]] = defaultdict(list)
    with RECOVERED.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("scope") != "consolidated" or row.get("metric_key") not in RECOVERY_KEYS:
                continue
            key = (str(row["ticker"]), int(row["year"]), str(row["metric_key"]))
            grouped[key].append(
                {
                    "ticker": key[0],
                    "year": key[1],
                    "metric_key": key[2],
                    "metric": METRIC_NAME.get(key[2], "revenue" if key[2] == "kqkd:01" else None),
                    "value": float(row["value"]) if finite(row.get("value")) else None,
                    "raw": row.get("raw"),
                    "label": row.get("label"),
                    "table_ref": row.get("table_ref"),
                    "csv_path": row.get("csv_path"),
                    "row_idx": row.get("row_idx"),
                    "col_idx": row.get("col_idx"),
                    "scope": row.get("scope"),
                    "origin": "recovered_fullscan",
                }
            )
    return grouped


def prior_year_cell(current: dict) -> dict | None:
    """Read the comparative column adjacent to a validated 2016 source cell."""
    source_table = str(current.get("table_ref") or "")
    source_csv = str(current.get("csv_path") or "")
    if not source_table or not source_csv:
        return None
    doc = source_table.split("|", 1)[0]
    path = Path(source_csv)
    if not path.is_absolute():
        path = ROOT / "build" / "tables" / doc / path.name
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    row_idx = int(current["row_idx"]) + 1
    col_idx = int(current["col_idx"])
    prior_col = col_idx + 1
    if not rows or row_idx >= len(rows) or prior_col >= len(rows[row_idx]):
        return None
    # Extracted tables retain numeric CSV column names on row 0; the first
    # actual table row usually carries the year headers.  Search the first
    # three physical rows instead of assuming a fixed layout.
    header_cells = None
    for candidate in rows[:3]:
        if prior_col >= len(candidate):
            continue
        joined = " ".join(candidate).lower()
        if (
            "2016" in joined
            or "năm nay" in joined
            or "năm trước" in joined
            or "cuối năm" in joined
            or "đầu năm" in joined
        ):
            header_cells = candidate
            break
    if header_cells is None:
        return None
    current_header = header_cells[col_idx].strip().lower()
    prior_header = header_cells[prior_col].strip().lower()
    header_ok = (
        "2016" in current_header
        or "năm nay" in current_header
        or "cuối năm" in current_header
        or "31/12" in current_header
    ) and (
        "2015" in prior_header
        or "năm trước" in prior_header
        or "đầu năm" in prior_header
        or "1/1/2016" in prior_header
        or "01/01/2016" in prior_header
    )
    if not header_ok:
        return None
    raw = rows[row_idx][prior_col]
    value = parse_vietnamese_number(raw)
    if value is None:
        return None
    value *= float(current.get("typed_factor") or 1) * float(current.get("scale") or 1)
    return {
        **current,
        "year": 2015,
        "value": value,
        "raw": raw,
        "col_idx": prior_col,
        "source_header": prior_header,
        "origin": "comparative_column_from_2016_report",
    }


def main() -> None:
    base, base_duplicates = load_base()
    recovered = load_recovered()
    merged = dict(base)
    filled: list[dict] = []
    conflicts: list[dict] = []
    corroborations: list[dict] = []

    # Some issuers use code 01 as net revenue when no deduction/code-02 row is
    # present.  Keep this alias explicit and source-visible.
    alias_fills: list[dict] = []
    for (ticker, year, metric_key), candidates in sorted(recovered.items()):
        if metric_key != "kqkd:01" or (ticker, year, "kqkd:10") in merged:
            continue
        valid = [row for row in candidates if finite(row.get("value"))]
        values = {float(row["value"]) for row in valid}
        if len(values) == 1:
            picked = {**valid[0], "metric_key": "kqkd:10", "metric": "revenue", "origin": "semantic_alias_kqkd01_no_deductions"}
            merged[(ticker, year, "kqkd:10")] = picked
            alias_fills.append({"key": [ticker, year, "kqkd:10"], "picked": picked, "all_candidates": valid})

    # A 2016 statement normally carries the 2015 comparative in the adjacent
    # column.  Recover it only when both headers make the year mapping explicit.
    comparative_fills: list[dict] = []
    for ticker, year, metric_key in sorted(list(merged)):
        if year != 2016 or (ticker, 2015, metric_key) in merged:
            continue
        prior = prior_year_cell(merged[(ticker, year, metric_key)])
        if prior:
            merged[(ticker, 2015, metric_key)] = prior
            comparative_fills.append({"key": [ticker, 2015, metric_key], "picked": prior})

    # TTF's extracted cash-flow table is structurally malformed: the code-20
    # row contains an extra blank cell.  The raw row was inspected directly.
    ttf_cfo_key = ("TTF", 2016, "lctt:20")
    if ttf_cfo_key not in merged:
        ttf_cfo = {
            "ticker": "TTF",
            "year": 2016,
            "metric_key": "lctt:20",
            "metric": "cfo",
            "value": -809_987_053_343.0,
            "raw": "(809.987.053.343)",
            "label": "Lưu chuyển tiền thuần sử dụng vào hoạt động kinh doanh",
            "table_ref": "TTF_financial_statements_2016_consolidated|300",
            "csv_path": str(ROOT / "build" / "tables" / "TTF_financial_statements_2016_consolidated" / "table_9_line300.csv"),
            "row_idx": 15,
            "col_idx": 3,
            "scope": "consolidated",
            "origin": "manual_structural_row_repair",
            "typed_factor": 1.0,
            "scale": 1.0,
        }
        merged[ttf_cfo_key] = ttf_cfo
        filled.append({"key": list(ttf_cfo_key), "picked": ttf_cfo, "all_candidates": []})

    for key, candidates in sorted(recovered.items()):
        if key[2] not in REQUIRED:
            continue
        valid = [row for row in candidates if finite(row.get("value"))]
        unique_values = sorted({float(row["value"]) for row in valid})
        if key in base:
            base_value = base[key].get("value")
            same = [row for row in valid if base_value is not None and math.isclose(row["value"], base_value)]
            if same:
                corroborations.append({"key": list(key), "base": base[key], "candidates": same})
            continue
        if len(unique_values) == 1:
            picked = valid[0]
            merged[key] = picked
            filled.append({"key": list(key), "picked": picked, "all_candidates": valid})
        elif valid:
            conflicts.append({"key": list(key), "values": unique_values, "candidates": valid})

    tickers = sorted({ticker for ticker, _, _ in merged})
    rows: list[dict] = []
    incomplete: list[dict] = []
    for ticker in tickers:
        record: dict[str, object] = {"ticker": ticker}
        missing: list[str] = []
        provenance: dict[str, dict] = {}
        for year in (2015, 2016):
            for metric_key in REQUIRED:
                metric = METRIC_NAME[metric_key]
                key = (ticker, year, metric_key)
                cell = merged.get(key)
                column = f"{metric}_{year}"
                record[column] = cell.get("value") if cell else None
                if cell:
                    provenance[column] = cell
                else:
                    missing.append(f"{year}:{metric_key}")
        complete_selector = finite(record["inventory_2015"]) and finite(record["inventory_2016"])
        eligible = False
        if complete_selector and float(record["inventory_2015"]) != 0:
            record["inventory_change_pct"] = (
                float(record["inventory_2016"]) / float(record["inventory_2015"]) - 1
            ) * 100
            eligible = float(record["inventory_2016"]) <= 0.9 * float(record["inventory_2015"])
        else:
            record["inventory_change_pct"] = None
        record["eligible"] = eligible
        if eligible and finite(record["cfo_2016"]) and finite(record["revenue_2016"]) and float(record["revenue_2016"]) != 0:
            record["cfo_margin_pct"] = float(record["cfo_2016"]) / float(record["revenue_2016"]) * 100
        else:
            record["cfo_margin_pct"] = None
        record["missing"] = missing
        record["provenance"] = provenance
        rows.append(record)
        if missing:
            incomplete.append({"ticker": ticker, "missing": missing, "eligible": eligible})

    eligible_rows = sorted(
        [row for row in rows if row["eligible"]],
        key=lambda row: (-math.inf if row["cfo_margin_pct"] is None else float(row["cfo_margin_pct"])),
        reverse=True,
    )
    ranked = [row for row in eligible_rows if finite(row.get("cfo_margin_pct"))]
    winner = ranked[0] if ranked else None
    unresolved_eligible = [row for row in eligible_rows if not finite(row.get("cfo_margin_pct"))]

    report = {
        "question_id": 464,
        "review_contract": {
            "report_scope": "consolidated (question does not request parent/separate statements)",
            "selector_table_and_row": "balance sheet, inventory, code 140, years 2015 and 2016",
            "numerator_table_and_row": "cash-flow statement, net CFO, code 20, year 2016",
            "denominator_table_and_row": "income statement, net revenue, code 10, year 2016",
            "unit_rule": "ratio is unit-invariant only when numerator and denominator share compatible scales",
            "formula": "max(CFO_2016 / net_revenue_2016 * 100) among inventory_2016 <= 90% * inventory_2015",
        },
        "baseline_answer": 35.42,
        "recomputed_answer": round(float(winner["cfo_margin_pct"]), 2) if winner else None,
        "winner": winner,
        "eligible_ranking": ranked,
        "unresolved_eligible": unresolved_eligible,
        "counts": {
            "base_cells": len(base),
            "base_duplicates": len(base_duplicates),
            "recovered_metric_keys": len(recovered),
            "filled_cells": len(filled),
            "conflicting_missing_cells": len(conflicts),
            "tickers_seen": len(tickers),
            "eligible": len(eligible_rows),
            "eligible_with_ratio": len(ranked),
            "incomplete_tickers": len(incomplete),
        },
        "filled": filled,
        "semantic_alias_fills": alias_fills,
        "comparative_fills": comparative_fills,
        "conflicts": conflicts,
        "base_duplicates": base_duplicates,
        "incomplete": incomplete,
        "semantic_warning": (
            "Codes 10/20/140 are accepted only as candidates. Labels and statement type still require "
            "manual review for the winner and any newly eligible/high-margin competitor, especially banks/insurers."
        ),
        "mutation_authority": False,
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT),
        "baseline": report["baseline_answer"],
        "recomputed": report["recomputed_answer"],
        "winner": winner["ticker"] if winner else None,
        "counts": report["counts"],
        "conflict_keys": [item["key"] for item in conflicts],
        "unresolved_eligible": [item["ticker"] for item in unresolved_eligible],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
