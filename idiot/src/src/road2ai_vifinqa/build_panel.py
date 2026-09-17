"""Extract the canonical statement cube used by formula questions.

Version 2 treats panel construction as a grounded selection problem.  The old
builder accepted the first numeric column in the first matching table, which
silently preferred prior periods, segment notes and stale units.  This module
scores every viable source cell, resolves duplicates deterministically and
records enough diagnostics to audit each choice.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .paths import INDEX_PATH, PANEL_MANIFEST_PATH, PANEL_PATH
from .table_semantics import (
    CellSemantics,
    PeriodRole,
    TableAnalyzer,
    period_match_score,
    row_label,
)
from .text import fold_text, parse_vn_number


CODE_RE = re.compile(r"^\d{1,3}$")
KQKD_REQUIRED = frozenset({"01", "11", "20", "25", "26"})
COST_KEYS = frozenset(f"kqkd:{code}" for code in ("11", "22", "23", "25", "26", "32", "51", "52"))
KNOWN_CODES = {
    "kqkd": frozenset({"01", "02", "10", "11", "20", "21", "22", "23", "25", "26", "30", "31", "32", "40", "50", "51", "52", "60"}),
    "cdkt": frozenset({"100", "110", "120", "130", "140", "150", "200", "210", "220", "230", "240", "250", "260", "270", "300", "310", "320", "330", "400", "410", "420", "430", "440"}),
    "lctt": frozenset({f"{value:02d}" for value in range(1, 81)}),
}


@dataclass(frozen=True, slots=True)
class Classification:
    kind: str
    code_col: int
    confidence: float


@dataclass(frozen=True, slots=True)
class PanelCandidate:
    score: float
    key: str
    value: float
    raw: str
    label: str
    doc_id: str
    table_id: int
    row_idx: int
    col_idx: int
    scale: float
    scope: str
    period_role: str
    period_year: int | None
    column_header: str
    section: str


def _normalise_code(value: str, kind: str) -> str:
    return value if kind == "cdkt" else value.zfill(2)


def _code_columns(rows: list[list[str]]) -> list[tuple[float, int, set[str], int]]:
    """Rank consistent code columns instead of accepting any leading number."""

    width = min(max((len(row) for row in rows), default=0), 5)
    candidates: list[tuple[float, int, set[str], int]] = []
    for col_idx in range(width):
        codes: set[str] = set()
        populated = 0
        label_after = 0
        for row in rows:
            if col_idx >= len(row):
                continue
            raw = row[col_idx].strip()
            if raw:
                populated += 1
            if CODE_RE.fullmatch(raw):
                codes.add(raw)
                if col_idx + 1 < len(row):
                    neighbour = row[col_idx + 1].strip()
                    if neighbour and not CODE_RE.fullmatch(neighbour) and bool(re.search(r"[^\W\d_]", neighbour, re.UNICODE)):
                        label_after += 1
        three_digit = len({code for code in codes if len(code) == 3})
        normalised = {code.zfill(2) for code in codes}
        statement_codes = len(normalised & (KNOWN_CODES["kqkd"] | KNOWN_CODES["lctt"]))
        score = (
            2.5 * three_digit
            + statement_codes
            + 0.45 * label_after
            + 0.05 * populated
            - 0.15 * col_idx
        )
        if codes:
            candidates.append((score, col_idx, codes, label_after))
    return sorted(candidates, reverse=True)


def _legacy_kind(rows: list[list[str]]) -> str | None:
    """Fast, high-recall statement prefilter used by the original builder."""

    codes: set[str] = set()
    labels: list[str] = []
    for row in rows:
        found = [
            (idx, raw.strip())
            for idx, raw in enumerate(row[:3])
            if CODE_RE.fullmatch(raw.strip())
        ]
        if not found:
            continue
        for _, code in found:
            codes.add(code)
            if len(code) == 1:
                codes.add(code.zfill(2))
        labels.append(fold_text(_label(row, found[-1][0])))
    if len({code for code in codes if len(code) == 3}) >= 5:
        return "cdkt"
    if any("chuyen tien" in label for label in labels):
        return "lctt"
    if len(codes & KQKD_REQUIRED) >= 3:
        return "kqkd"
    return None


def _classify(
    rows: list[list[str]],
    context: str = "",
    *,
    preclassified_kind: str | None = None,
) -> Classification | None:
    legacy_kind = preclassified_kind or _legacy_kind(rows)
    if legacy_kind is None:
        return None
    columns = _code_columns(rows)
    if not columns:
        return None
    best: tuple[float, Classification] | None = None
    for _, col_idx, codes, label_after in columns:
        normalised = {code.zfill(2) for code in codes}
        valid = len({_normalise_code(code, legacy_kind) for code in codes} & KNOWN_CODES[legacy_kind])
        if valid == 0:
            continue
        confidence = 6.0 + valid * 0.35 + min(label_after, 12) * 0.15
        result = Classification(kind=legacy_kind, code_col=col_idx, confidence=confidence)
        rank = confidence - col_idx * 0.05
        if best is None or rank > best[0]:
            best = (rank, result)
    return best[1] if best else None


def _find_code(row: list[str], *, code_col: int | None = None) -> tuple[int, str] | None:
    indices = [code_col] if code_col is not None else list(range(min(3, len(row))))
    for idx in indices:
        if idx is None or idx >= len(row):
            continue
        value = row[idx].strip()
        if CODE_RE.fullmatch(value):
            return idx, value
    return None


def _label(row: list[str], code_idx: int) -> str:
    return row_label(row, code_idx=code_idx)


def _numeric_values(
    row: list[str], code_idx: int, *, analyzer: TableAnalyzer | None = None,
    row_idx: int | None = None,
) -> list[tuple[int, float, str]]:
    values: list[tuple[int, float, str]] = []
    for idx in range(code_idx + 1, len(row)):
        raw = row[idx].strip()
        # A small number under an explicit reporting-period heading is an
        # amount, not a note number. An explicit dash in the same column is
        # a reported nil amount; an empty OCR cell remains unknown.
        period_column = False
        scaled_amount_column = False
        if analyzer is not None and row_idx is not None:
            semantics = analyzer.cell(row_idx, idx, code_idx=code_idx)
            period = semantics.period
            period_column = bool(semantics.row_label) and period.explicit and period.role != PeriodRole.UNKNOWN
            # OCR occasionally shifts a note reference into an amount column.
            # Only relax the old small-number guard when a scaled currency
            # unit is explicit; base-VND one-digit values remain ambiguous.
            scaled_amount_column = period_column and semantics.unit_scale > 1
        value = parse_vn_number(raw)
        if value is None and raw in {"-", "–", "—"} and period_column:
            value = 0.0
        if value is None:
            continue
        compact = raw.replace(".", "").replace(",", "").replace(" ", "")
        unsigned = compact.strip("+-()")
        if not scaled_amount_column and unsigned.isdigit() and (len(unsigned) <= 3 or 1900 <= int(unsigned) <= 2100):
            continue
        values.append((idx, value, raw))
    return values


def _cell_score(
    *,
    semantics: CellSemantics,
    col_idx: int,
    code_idx: int,
    report_year: int,
) -> float:
    score = period_match_score(
        semantics.period,
        requested_year=report_year,
        report_year=report_year,
    )
    if semantics.period.role == PeriodRole.UNKNOWN and not semantics.period.explicit:
        score += max(0.0, 2.0 - 0.25 * max(0, col_idx - code_idx - 1))
    if semantics.period.role in (PeriodRole.PRIOR, PeriodRole.OPENING):
        score -= 2.0
    if semantics.column_header:
        score += 0.3
    return score


def _candidate_score(
    *,
    classification: Classification,
    semantics: CellSemantics,
    scope: str,
    table_id: int,
    context: str,
    col_score: float,
) -> float:
    score = classification.confidence + col_score
    if scope == "consolidated":
        score += 2.0
    elif scope == "unknown":
        score += 0.5
    folded = fold_text(context)
    if classification.kind == "cdkt" and any(marker in folded for marker in ("bang can doi", "tinh hinh tai chinh")):
        score += 2.0
    elif classification.kind == "kqkd" and "ket qua hoat dong kinh doanh" in folded:
        score += 2.0
    elif classification.kind == "lctt" and "luu chuyen tien te" in folded:
        score += 2.0
    if any(marker in folded for marker in ("bo phan", "phan khuc", "dieu chinh", "phan loai lai")):
        score -= 3.0
    if semantics.period.role in (PeriodRole.CURRENT, PeriodRole.CLOSING):
        score += 1.0
    if semantics.is_total:
        score += 0.25
    # Stable final tie-breaker: face statements generally occur before notes.
    score -= table_id * 1e-5
    return score


def _is_current_period_source(semantics: CellSemantics, report_year: int) -> bool:
    if semantics.period.year is not None and semantics.period.year != report_year:
        return False
    return semantics.period.role not in (PeriodRole.PRIOR, PeriodRole.OPENING)


def _atomic_json_write(path: Path, payload: object, *, pretty: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )
    tmp.write_text(text + ("\n" if pretty else ""), encoding="utf-8")
    tmp.replace(path)


def build_panel(
    *,
    force: bool = False,
    output_path: Path = PANEL_PATH,
    manifest_path: Path = PANEL_MANIFEST_PATH,
    index_path: Path = INDEX_PATH,
) -> dict[str, object]:
    if output_path.exists() and manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    started = time.time()
    conn = sqlite3.connect(f"file:{index_path.as_posix()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    candidates: dict[tuple[str, str, str], list[PanelCandidate]] = {}
    classified_buckets: set[tuple[str, str]] = set()
    table_counts = {"kqkd": 0, "cdkt": 0, "lctt": 0}
    query = """
        SELECT d.ticker, d.report_year, d.scope, t.doc_id, t.table_id,
               t.context, t.rows_json
        FROM tables t JOIN documents d ON d.doc_id=t.doc_id
        WHERE d.scope != 'parent'
        ORDER BY d.ticker, d.report_year,
                 CASE d.scope WHEN 'consolidated' THEN 0 ELSE 1 END,
                 t.doc_id, t.table_id
    """
    try:
        for item in conn.execute(query):
            rows: list[list[str]] = json.loads(item["rows_json"])
            legacy_kind = _legacy_kind(rows)
            if legacy_kind is None:
                continue
            classified_buckets.add((str(item["ticker"]), str(item["report_year"])))
            classification = _classify(
                rows,
                item["context"],
                preclassified_kind=legacy_kind,
            )
            if classification is None:
                continue
            kind = classification.kind
            table_counts[kind] += 1
            ticker = str(item["ticker"])
            year = str(item["report_year"])
            classified_buckets.add((ticker, year))
            report_year = int(item["report_year"])
            analyzer = TableAnalyzer(
                rows,
                context=item["context"],
                report_year=report_year,
            )
            for row_idx, row in enumerate(rows):
                found = _find_code(row, code_col=classification.code_col)
                if found is None:
                    continue
                code_idx, code = found
                code = _normalise_code(code, kind)
                if code not in KNOWN_CODES[kind]:
                    continue
                values = _numeric_values(row, code_idx, analyzer=analyzer, row_idx=row_idx)
                if not values:
                    continue
                evaluated: list[tuple[float, int, float, str, CellSemantics]] = []
                for col_idx, parsed, raw in values:
                    if raw.strip().endswith("%"):
                        continue
                    semantics = analyzer.cell(row_idx, col_idx, code_idx=code_idx)
                    evaluated.append(
                        (
                            _cell_score(
                                semantics=semantics,
                                col_idx=col_idx,
                                code_idx=code_idx,
                                report_year=report_year,
                            ),
                            col_idx,
                            parsed,
                            raw,
                            semantics,
                        )
                    )
                if not evaluated:
                    continue
                col_score, col_idx, parsed, raw, semantics = max(
                    evaluated, key=lambda value: (value[0], -value[1])
                )
                # A comparative/opening value is not a substitute for a
                # missing current-period operand.  Leaving the panel cell
                # absent makes downstream recipes fail closed instead of
                # silently calculating with the wrong year.
                if not _is_current_period_source(semantics, report_year):
                    continue
                key = f"{kind}:{code}"
                value = parsed * semantics.unit_scale
                if key in COST_KEYS:
                    value = abs(value)
                score = _candidate_score(
                    classification=classification,
                    semantics=semantics,
                    scope=item["scope"],
                    table_id=int(item["table_id"]),
                    context=item["context"],
                    col_score=col_score,
                )
                candidate = PanelCandidate(
                    score=score,
                    key=key,
                    value=value,
                    raw=raw,
                    label=semantics.row_label,
                    doc_id=item["doc_id"],
                    table_id=int(item["table_id"]),
                    row_idx=row_idx,
                    col_idx=col_idx,
                    scale=semantics.unit_scale,
                    scope=item["scope"],
                    period_role=semantics.period.role.value,
                    period_year=semantics.period.year,
                    column_header=semantics.column_header,
                    section=semantics.section,
                )
                candidates.setdefault((ticker, year, key), []).append(candidate)
    finally:
        conn.close()

    panel: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for ticker, year in sorted(classified_buckets):
        panel.setdefault(ticker, {}).setdefault(year, {})
    duplicate_count = 0
    ambiguous = 0
    margins: list[float] = []
    for (ticker, year, key), options in sorted(candidates.items()):
        ranked = sorted(
            options,
            key=lambda item: (-item.score, item.doc_id, item.table_id, item.row_idx, item.col_idx),
        )
        winner = ranked[0]
        duplicate_count += max(0, len(ranked) - 1)
        margin = winner.score - ranked[1].score if len(ranked) > 1 else 99.0
        margins.append(margin)
        if margin < 1.0:
            ambiguous += 1
        panel.setdefault(ticker, {}).setdefault(year, {})[key] = {
            "value": winner.value,
            "raw": winner.raw,
            "label": winner.label,
            "doc_id": winner.doc_id,
            "table_id": winner.table_id,
            "row_idx": winner.row_idx,
            "col_idx": winner.col_idx,
            "scale": winner.scale,
            "selection_score": round(winner.score, 6),
            "selection_margin": round(margin, 6),
            "candidate_count": len(ranked),
            "scope": winner.scope,
            "period_role": winner.period_role,
            "period_year": winner.period_year,
            "column_header": winner.column_header,
            "section": winner.section,
        }

    _atomic_json_write(output_path, panel)
    cells = sum(len(metrics) for years in panel.values() for metrics in years.values())
    manifest: dict[str, object] = {
        "format_version": 2,
        "tickers": len(panel),
        "ticker_years": sum(len(years) for years in panel.values()),
        "cells": cells,
        "statement_tables": table_counts,
        "duplicate_candidates": duplicate_count,
        "ambiguous_selections_margin_lt_1": ambiguous,
        "median_selection_margin": round(sorted(margins)[len(margins) // 2], 6) if margins else None,
        "elapsed_seconds": round(time.time() - started, 3),
        "output_path": str(output_path),
        "index_path": str(index_path),
    }
    _atomic_json_write(manifest_path, manifest, pretty=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=PANEL_PATH)
    parser.add_argument("--manifest", type=Path, default=PANEL_MANIFEST_PATH)
    parser.add_argument("--index", type=Path, default=INDEX_PATH)
    args = parser.parse_args()
    print(
        json.dumps(
            build_panel(
                force=args.force,
                output_path=args.output,
                manifest_path=args.manifest,
                index_path=args.index,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
