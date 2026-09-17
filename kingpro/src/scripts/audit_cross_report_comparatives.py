"""Cross-check primary-statement operands against the next audited report.

The current submission, its source manifest and its generated Pandas program
share ancestry.  They can therefore agree while all three selected the wrong
period or value.  Financial statements provide a genuinely separate check:
year Y is commonly repeated as the comparative column of report Y+1.

This audit never edits a submission.  It only compares cells with the same
primary-statement code (CDKT/KQKD/LCTT), same ticker/scope, and an explicit
comparative-year header in the following report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from audit_legacy_source_scope import fold
from audit_semantic_child_table_lineage import requested_year_matches
from audit_source_cell_semantics import header_path, physical_label


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "build" / "catalog_enriched.jsonl"


def accounting_number(value: object, typed_factor: object = 1) -> float | None:
    if value is None:
        return None
    if not isinstance(value, str):
        try:
            return float(value) * float(typed_factor or 1)
        except (TypeError, ValueError):
            return None
    text = value.strip().replace("\u00a0", " ")
    if text in {"", "-", "–", "—"}:
        return None
    if " " in text:
        text = text.split()[0]
    if ")(" in text:
        text = text.split(")(", 1)[0] + ")"
    negative = text.startswith("(") and text.endswith(")")
    text = text.replace("(", "").replace(")", "").replace("%", "").replace("$", "")
    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.split(",")[-1]
        text = text.replace(",", "." if len(tail) <= 2 else "")
    elif "." in text and len(text.split(".")[-1]) == 3:
        text = text.replace(".", "")
    try:
        result = float(text)
    except ValueError:
        return None
    return -abs(result) if negative else result


def unit_scale(text: object) -> float:
    value = fold(text)
    if re.search(r"\b(trieu|million)\b", value):
        return 1e6
    if re.search(r"\b(nghin|thousand)\b", value):
        return 1e3
    if re.search(r"\b(ty|billion)\b", value):
        return 1e9
    return 1.0


def report_year(report_id: str) -> int | None:
    match = re.search(r"_financial_statements_(\d{4})", report_id)
    return int(match.group(1)) if match else None


def report_scope(report_id: str) -> str:
    if "_consolidated" in report_id:
        return "consolidated"
    if "_separate" in report_id:
        return "separate"
    return "unknown"


def expected_family(metric_key: object) -> str:
    return str(metric_key or "").split(":", 1)[0].lower()


def table_has_family(meta: dict[str, Any], family: str) -> bool:
    text = fold(f"{meta.get('section_title', '')} | {meta.get('search_text', '')}")
    if family == "cdkt":
        return "can doi ke toan" in text or "tinh hinh tai chinh" in text
    if family == "kqkd":
        return "ket qua hoat dong kinh doanh" in text
    if family == "lctt":
        return "luu chuyen tien" in text or re.search(r"\blu\W*u\s+chuyen tien", text) is not None
    return False


@lru_cache(maxsize=None)
def compact_table(path_text: str) -> pd.DataFrame:
    return pd.read_csv(path_text, dtype=str, keep_default_na=False)


def metric_code(metric_key: object) -> str:
    value = str(metric_key or "")
    if ":" not in value:
        return ""
    prefix, code = value.split(":", 1)
    return code if prefix in {"cdkt", "kqkd", "lctt"} and code.isdigit() else ""


def candidate_coordinates(
    frame: pd.DataFrame,
    *,
    code: str,
    requested_year: int,
    table_context: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in range(len(frame)):
        code_columns = [
            column
            for column in range(len(frame.columns))
            if str(frame.iloc[row, column]).strip().lstrip("0") == code.lstrip("0")
        ]
        if not code_columns:
            continue
        first_code_column = min(code_columns)
        for column in range(first_code_column + 1, len(frame.columns)):
            raw = str(frame.iloc[row, column]).strip()
            number = accounting_number(raw)
            if number is None or not requested_year_matches(frame, row, column, requested_year):
                continue
            path = header_path(frame, column, row)
            # requested_year_matches intentionally tolerates weak table-wide
            # evidence for other audits.  Cross-report validation needs a
            # stricter contract: the selected column's own header lineage must
            # explicitly contain the comparative year.  This rejects note
            # numbers and retrospective-adjustment columns that happen to sit
            # in a table mentioning the year elsewhere.
            if str(requested_year) not in fold(" | ".join(path)):
                continue
            # The extracted CSV keeps the physical number exactly as printed.
            # Some catalog/search metadata repeats a presentation unit even
            # when the extractor has already expanded the cell to VND.  Applying
            # that unit here therefore double-scales otherwise identical facts.
            # Keep the parsed physical value as the comparison contract and
            # expose the detected unit only as review metadata.
            detected_unit_scale = unit_scale(" | ".join(path) + " | " + table_context)
            rows.append(
                {
                    "row": row,
                    "column": column,
                    "raw": raw,
                    "value": number,
                    "detected_unit_scale": detected_unit_scale,
                    "label": physical_label(frame, row, column),
                    "header_path": path,
                }
            )
    return rows


def close(a: float, b: float, relative_tolerance: float) -> bool:
    return abs(a - b) <= relative_tolerance * max(abs(a), abs(b), 1.0)


def load_catalog(path: Path) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int, str], list[dict[str, Any]]]]:
    catalog: dict[str, dict[str, Any]] = {}
    by_report: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        meta = json.loads(line)
        report = str(meta["report_id"])
        year = report_year(report)
        if year is None:
            continue
        catalog[str(meta["table_ref"])] = meta
        by_report[(str(meta["ticker"]), year, report_scope(report))].append(meta)
    return catalog, by_report


@lru_cache(maxsize=None)
def manifest_rows(candidate_dir_text: str, csv_rel: str) -> list[dict[str, str]]:
    path = Path(candidate_dir_text) / csv_rel
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def source_value(candidate_dir: Path, cell: dict[str, Any]) -> tuple[float | None, dict[str, str]]:
    rows = manifest_rows(str(candidate_dir), str(cell.get("csv", "")))
    index = int(cell.get("source_index", 0))
    if index >= len(rows):
        return None, {}
    source = rows[index]
    raw = source.get("raw", cell.get("raw_physical", ""))
    value = accounting_number(raw, source.get("typed_factor", 1))
    try:
        scale = float(source.get("scale", 1) or 1)
    except ValueError:
        scale = 1.0
    return (value * scale if value is not None else None), source


def audit(
    lineage_path: Path,
    candidate_dir: Path,
    catalog_path: Path,
    relative_tolerance: float,
) -> dict[str, Any]:
    lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    catalog, by_report = load_catalog(catalog_path)
    findings: list[dict[str, Any]] = []
    exact_matches = 0
    ambiguous = 0
    checked = 0
    errors: list[dict[str, Any]] = []

    for record in lineage.get("records", []):
        qid = int(record["id"])
        for cell_index, cell in enumerate(record.get("cells", [])):
            family = expected_family(cell.get("metric_key"))
            code = metric_code(cell.get("metric_key"))
            current_ref = str(cell.get("source_table", ""))
            current_meta = catalog.get(current_ref)
            if not code or family not in {"cdkt", "kqkd", "lctt"} or current_meta is None:
                continue
            year = int(cell.get("year"))
            current_report = str(current_meta["report_id"])
            # First pass deliberately uses only facts read from their original
            # year report; comparative backfills need a separate contract.
            if report_year(current_report) != year:
                continue
            current_value, source = source_value(candidate_dir, cell)
            if current_value is None:
                continue
            current_raw = source.get("raw", cell.get("raw_physical", ""))
            current_physical_value = accounting_number(current_raw)
            if current_physical_value is None:
                continue
            next_tables = [
                meta
                for meta in by_report.get((str(cell.get("ticker")), year + 1, report_scope(current_report)), [])
                if table_has_family(meta, family)
            ]
            candidates: list[dict[str, Any]] = []
            for meta in next_tables:
                try:
                    frame = compact_table(str(ROOT / "build" / "tables" / str(meta["csv_path"])))
                    for coordinate in candidate_coordinates(
                        frame,
                        code=code,
                        requested_year=year,
                        table_context=f"{meta.get('section_title', '')} | {meta.get('search_text', '')}",
                    ):
                        candidates.append({"table_ref": meta["table_ref"], **coordinate})
                except Exception as error:
                    errors.append({"id": qid, "table_ref": meta.get("table_ref"), "error": f"{type(error).__name__}: {error}"})
            if not candidates:
                continue
            checked += 1
            # Compare the two reports in their physical presentation space.
            # The candidate manifest may scale a raw "83.177.720" by 1e6 for
            # execution, while the following audited report prints the same
            # physical cell.  Canonical execution scale is useful for answers,
            # but would create false mismatches in this source-to-source audit.
            matching = [
                row for row in candidates
                if close(current_physical_value, float(row["value"]), relative_tolerance)
            ]
            if matching:
                exact_matches += 1
                continue
            unique_values: list[float] = []
            for row in candidates:
                value = float(row["value"])
                if not any(close(value, other, relative_tolerance) for other in unique_values):
                    unique_values.append(value)
            if len(unique_values) != 1:
                ambiguous += 1
                continue
            candidate = candidates[0]
            findings.append(
                {
                    "id": qid,
                    "cell_index": cell_index,
                    "question": record.get("question"),
                    "metric_key": cell.get("metric_key"),
                    "ticker": cell.get("ticker"),
                    "year": year,
                    "scope": report_scope(current_report),
                    "current_table": current_ref,
                    "current_label": cell.get("source_label"),
                    "current_raw": current_raw,
                    "current_value": current_physical_value,
                    "current_canonical_value": current_value,
                    "comparative_table": candidate["table_ref"],
                    "comparative_label": candidate["label"],
                    "comparative_raw": candidate["raw"],
                    "comparative_value": candidate["value"],
                    "comparative_header_path": candidate["header_path"],
                    "relative_difference": abs(float(candidate["value"]) - current_physical_value)
                    / max(abs(current_physical_value), 1.0),
                    "review_only": True,
                }
            )

    findings.sort(key=lambda row: (-float(row["relative_difference"]), int(row["id"]), int(row["cell_index"])))
    return {
        "kind": "cross_report_comparative_audit",
        "lineage": str(lineage_path),
        "policy": "Same ticker/scope/primary-statement code; original year Y versus explicit comparative Y column in report Y+1. Review only because later reports may restate values.",
        "checked_operand_count": checked,
        "exact_match_operand_count": exact_matches,
        "ambiguous_operand_count": ambiguous,
        "finding_count": len(findings),
        "question_count": len({int(row["id"]) for row in findings}),
        "question_ids": sorted({int(row["id"]) for row in findings}),
        "resolution_error_count": len(errors),
        "findings": findings,
        "errors": errors,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("lineage", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, default=CATALOG)
    parser.add_argument("--relative-tolerance", type=float, default=0.0001)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    payload = audit(args.lineage, args.candidate, args.catalog, args.relative_tolerance)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: payload[key] for key in (
        "checked_operand_count", "exact_match_operand_count", "ambiguous_operand_count",
        "finding_count", "question_count", "question_ids", "resolution_error_count",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
