"""Deterministic single-table value extraction for direct questions."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Corpus
from .retrieval import RowHit, retrieve_rows
from .table_semantics import TableAnalyzer, cell_semantics, period_match_score
from .text import fold_text, parse_vn_number, requested_scale, source_scale


@dataclass(frozen=True, slots=True)
class DirectAnswer:
    answer: float
    hit: RowHit
    col_idx: int
    raw_value: str
    source_scale: float
    requested_scale: float
    confidence: float


def _column_year_score(
    hit: RowHit,
    col_idx: int,
    requested_year: int,
    *,
    analyzer: TableAnalyzer | None = None,
    semantic: bool = False,
) -> float:
    if not semantic:
        score = 0.0
        for row_idx, row in enumerate(hit.table.rows[:5]):
            if row_idx >= hit.row.row_idx or col_idx >= len(row):
                continue
            cell = fold_text(row[col_idx])
            if str(requested_year) in cell:
                distance = abs(hit.row.row_idx - row_idx)
                score = max(score, 10.0 / (1.0 + distance * 0.15))
            if requested_year == hit.document.report_year and any(
                marker in cell for marker in ("nam nay", "ky nay", "cuoi ky")
            ):
                distance = abs(hit.row.row_idx - row_idx)
                score = max(score, 9.0 / (1.0 + distance * 0.15))
            if requested_year == hit.document.report_year - 1 and any(
                marker in cell for marker in ("nam truoc", "ky truoc", "dau ky")
            ):
                distance = abs(hit.row.row_idx - row_idx)
                score = max(score, 9.0 / (1.0 + distance * 0.15))
        if requested_year == hit.document.report_year:
            score += max(0.0, 2.5 - 0.25 * col_idx)
        if requested_year == hit.document.report_year - 1:
            score += max(0.0, 1.5 - 0.15 * col_idx)
        return score

    semantics = (
        analyzer.cell(hit.row.row_idx, col_idx)
        if analyzer is not None
        else cell_semantics(
            hit.table.rows,
            hit.row.row_idx,
            col_idx,
            context=hit.table.context,
            report_year=hit.document.report_year,
        )
    )
    score = period_match_score(
        semantics.period,
        requested_year=requested_year,
        report_year=hit.document.report_year,
    )
    # Header-less statements normally put the current period first.  This is
    # only a weak tie-breaker now; an explicit conflicting header wins.
    if not semantics.period.explicit and semantics.period.role.value == "unknown":
        if requested_year == hit.document.report_year:
            score += max(0.0, 1.5 - 0.15 * col_idx)
        elif requested_year == hit.document.report_year - 1:
            score += max(0.0, 0.75 - 0.08 * col_idx)
    return score


def _source_scale_for_hit(hit: RowHit) -> float:
    rows = hit.table.rows
    header_text = " ".join(" ".join(row) for row in rows[: min(5, len(rows))])
    scale = source_scale(f"{hit.table.context} {header_text}")
    folded_header = fold_text(header_text)
    if "vnd" in folded_header and source_scale(header_text) == 1.0 and not any(
        unit in folded_header for unit in ("trieu vnd", "nghin vnd", "ngan vnd", "ty vnd")
    ):
        return 1.0
    return scale


def answer_direct(
    corpus: Corpus,
    question: str,
    *,
    limit: int = 30,
    semantic_columns: bool = False,
) -> DirectAnswer | None:
    years = corpus.infer_years(question)
    requested_year = years[-1] if years else 0
    hits = retrieve_rows(corpus, question, limit=limit, include_prior=False)
    candidates: list[DirectAnswer] = []
    target_scale = requested_scale(question)
    analyzers: dict[tuple[str, int], TableAnalyzer] = {}

    for rank, hit in enumerate(hits):
        key = (hit.table.doc_id, hit.table.table_id)
        analyzer = analyzers.get(key) if semantic_columns else None
        if semantic_columns and analyzer is None:
            analyzer = TableAnalyzer(
                hit.table.rows,
                context=hit.table.context,
                report_year=hit.document.report_year,
            )
            analyzers[key] = analyzer
        numeric: list[tuple[int, float, str]] = []
        for col_idx, raw in enumerate(hit.row.cells):
            value = parse_vn_number(raw)
            if value is None:
                continue
            # Ignore line-item codes and year labels when other values exist.
            stripped = raw.strip().replace(".", "").replace(",", "")
            if stripped.isdigit() and (len(stripped) <= 3 or 1900 <= int(stripped) <= 2100):
                continue
            numeric.append((col_idx, value, raw))
        if not numeric:
            continue

        all_col_scores = [
            _column_year_score(
                hit,
                col_idx,
                requested_year,
                analyzer=analyzer,
                semantic=semantic_columns,
            )
            for col_idx in range(max((len(row) for row in hit.table.rows), default=0))
        ]

        best_col, value, raw = max(
            numeric,
            key=lambda item: (
                _column_year_score(
                    hit,
                    item[0],
                    requested_year,
                    analyzer=analyzer,
                    semantic=semantic_columns,
                ),
                -item[0],
            ),
        )
        col_score = _column_year_score(
            hit,
            best_col,
            requested_year,
            analyzer=analyzer,
            semantic=semantic_columns,
        )
        if all_col_scores and col_score + 2.0 < max(all_col_scores):
            # Do not silently substitute the prior-year value when the target
            # period cell in an otherwise attractive row is blank.
            continue
        scale = (
            analyzer.cell(hit.row.row_idx, best_col).unit_scale
            if semantic_columns and analyzer is not None
            else _source_scale_for_hit(hit)
        )
        converted = value * scale / target_scale
        year_bonus = 1.5 if requested_year and hit.document.report_year == requested_year else 0.0
        confidence = hit.score + min(col_score, 3.0) * 0.5 + year_bonus - rank * 0.08
        candidates.append(
            DirectAnswer(converted, hit, best_col, raw, scale, target_scale, confidence)
        )

    if not candidates:
        return None
    candidates.sort(key=lambda item: -item.confidence)
    return candidates[0]
