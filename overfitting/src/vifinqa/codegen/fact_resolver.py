"""Resolve one Fact (ticker, year, doc_type, metric) to one concrete table cell.

This is the missing link between `router.decompose` (which says WHICH facts a
question needs) and `codegen.formulas` (which says HOW to combine them).
Composite questions — growth, difference, ratio, ranking — are ~50% of the test
set and scored 0.000 with the lookup-only rule engine.

A resolved fact carries full provenance so the generated pandas query is a
single expression addressing exactly the located row/column, and so the caller
can compute a confidence for arbitration against the LLM.
"""
from __future__ import annotations

import math
import json
import re
from dataclasses import dataclass

from ..finance.metrics import get_metric, metric_context_matches, metric_keys
from ..retrieval.shortlist import (
    _period_kind,
    build_shortlist,
    candidate_matches_requirement,
    requirement_linking_variants,
)
from ..retrieval.serialize import df_roundtrip
from ..utils.viet_num import parse_vn_number
from ..utils.viet_text import norm, tokens

# a column header explicitly naming the wanted year is the strongest evidence
_YEAR_STRONG = 3
_YEAR_WEAK = 1


@dataclass
class ResolvedFact:
    ticker: str
    year: int | None
    metric: str
    var: str                 # df variable holding the cell
    report_id: str
    table_pos: int
    row: int
    label: str
    code: str
    col: int
    col_name: str
    value: float             # raw cell value (NOT unit-converted)
    unit_scale: float        # multiply to get VND
    score: float             # label-match score of the chosen row
    year_evidence: int       # 3 = header names the year, 1 = positional guess
    value_column: str = "value"
    value_multiplier: float = 1.0

    @property
    def value_vnd(self) -> float:
        return float(self.value) * float(self.unit_scale)

    def expr(self) -> str:
        """Single pandas expression returning this cell's raw value."""
        label = re.sub(r"\s+", " ", str(self.label)).strip()
        label_filter = (
            f"{self.var}['label'].isna()" if not label else
            f"{self.var}['label'].str.strip().eq({label!r})"
        )
        if self.value_column == "code":
            return (f"(float({self.var}.loc[({self.var}['row'] == {self.row}) "
                    f"& {label_filter}, 'code'].iloc[0]) "
                    f"* {self.value_multiplier:g})")
        return (f"float({self.var}.loc[({self.var}['row'] == {self.row}) "
                f"& {label_filter} "
                f"& ({self.var}['col'] == {self.col}), 'value'].iloc[0])")

    def expr_vnd(self) -> str:
        return f"({self.expr()} * {self.unit_scale:g})"


@dataclass(frozen=True)
class MatrixRequest:
    """Typed row/column/block lookup over the original extracted grid.

    Each entry in ``table_term_groups`` is an OR-group; every group must be
    represented in the table. The remaining axes are exact by default so a
    repeated label can only be selected inside its requested block/column.
    """

    table_term_groups: tuple[tuple[str, ...], ...] = ()
    context_term_groups: tuple[tuple[str, ...], ...] = ()
    row_variants: tuple[str, ...] = ()
    row_codes: tuple[str, ...] = ()
    column_variants: tuple[str, ...] = ()
    block_variants: tuple[str, ...] = ()
    block_stop_variants: tuple[str, ...] = ()
    row_before_variants: tuple[str, ...] = ()
    column_index: int | None = None
    collect: str = "cell"       # cell | rows | block | last_total
    row_mode: str = "exact"     # exact | prefix | contains


def resolve_matrix_request(
        request: MatrixRequest, tables: list[dict], ticker: str, year: int,
        doc_type: str, metric: str) -> list[ResolvedFact]:
    """Resolve a matrix request, refusing conflicting table renderings.

    Unlike the regular row resolver, this path reads ``grid_json`` to retain
    multi-row headers and section boundaries. Returned facts still point at
    the tidy CSV cells, so the emitted pandas expression is executable by the
    competition grader.
    """
    scoped = [
        table for table in _tables_for_exact_year(
            tables, ticker.upper(), int(year), doc_type)
        if _report_year(str(table.get("report_id") or "")) == int(year)
    ]
    candidates: list[list[ResolvedFact]] = []
    for table in scoped:
        grid = _matrix_grid(table)
        if not grid or not _matrix_table_matches(grid, request.table_term_groups):
            continue
        if not _matrix_text_matches(
                str(table.get("context") or ""), request.context_term_groups):
            continue
        cells = _matrix_cells(grid, request)
        if not cells:
            continue
        resolved = []
        for row, col in cells:
            fact = _resolved_grid_cell(
                table, ticker.upper(), int(year), metric, row, col)
            if fact is None:
                resolved = []
                break
            resolved.append(fact)
        if resolved:
            candidates.append(resolved)
    if not candidates:
        return []

    # Duplicate OCR renderings are common. Keep one only when the full ordered
    # value vector agrees; a competing interpretation fails closed.
    best = candidates[0]
    signature = [fact.value_vnd for fact in best]
    for candidate in candidates[1:]:
        values = [fact.value_vnd for fact in candidate]
        if len(values) != len(signature) or any(
                not math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)
                for left, right in zip(signature, values)):
            return []
    return best


def _matrix_grid(table: dict) -> list[list[str]]:
    raw = table.get("grid_json")
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(str(raw or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _matrix_table_matches(
        grid: list[list[str]], groups: tuple[tuple[str, ...], ...]) -> bool:
    return _matrix_text_matches(
        " ".join(str(cell) for row in grid for cell in row), groups)


def _matrix_text_matches(
        value: str, groups: tuple[tuple[str, ...], ...]) -> bool:
    blob = norm(value)
    return all(any(norm(variant) in blob for variant in group) for group in groups)


def _matrix_cells(grid: list[list[str]], request: MatrixRequest) -> list[tuple[int, int]]:
    start = 0
    limit = len(grid)
    if request.row_before_variants:
        stops = [
            index for index, row in enumerate(grid)
            if _row_matches(row, request.row_before_variants, "contains")
        ]
        if stops:
            limit = min(stops)

    if request.collect == "block":
        starts = [
            index for index, row in enumerate(grid[:limit])
            if _row_matches(row, request.block_variants, "exact")
        ]
        if len(starts) != 1:
            return []
        start = starts[0]
        end = limit
        for index in range(start + 1, limit):
            if _row_matches(
                    grid[index], request.block_stop_variants, "exact"):
                end = index
                break
        col = (request.column_index if request.column_index is not None else
               _matrix_column(grid, request.column_variants, start))
        if col is None:
            return []
        return [
            (index, col) for index in range(start + 1, end)
            if grid[index] and norm(grid[index][0])
            and col < len(grid[index])
            and parse_vn_number(grid[index][col]) is not None
        ]

    if request.block_variants:
        starts = [
            index for index, row in enumerate(grid[:limit])
            if _row_matches(row, request.block_variants, "exact")
        ]
        if len(starts) != 1:
            return []
        start = starts[0] + 1
        if request.block_stop_variants:
            stops = [
                index for index in range(start, limit)
                if _row_matches(
                    grid[index], request.block_stop_variants, "exact")
            ]
            if stops:
                limit = min(stops)

    row_indexes = []
    if request.row_codes:
        wanted = {str(value).strip() for value in request.row_codes}
        row_indexes = [
            index for index, row in enumerate(grid[start:limit], start=start)
            if any(str(cell).strip() in wanted for cell in row[:4])
        ]
    elif request.collect == "last_total":
        row_indexes = [
            index for index, row in enumerate(grid[start:limit], start=start)
            if row and not norm(row[0])
        ]
    else:
        row_indexes = [
            index for index, row in enumerate(grid[start:limit], start=start)
            if _row_matches(row, request.row_variants, request.row_mode)
        ]
    if not row_indexes:
        return []

    header_limit = min(max(row_indexes) + 1, 6)
    col = (request.column_index if request.column_index is not None else
           _matrix_column(grid, request.column_variants, header_limit))
    if col is None:
        return []
    row_indexes = [
        index for index in row_indexes
        if col < len(grid[index])
        and parse_vn_number(grid[index][col]) is not None
    ]
    if request.collect == "last_total":
        row_indexes = row_indexes[-1:]
    elif request.collect == "cell" and len(row_indexes) != 1:
        return []
    return [(index, col) for index in row_indexes]


def _matrix_column(
        grid: list[list[str]], variants: tuple[str, ...], header_limit: int) -> int | None:
    wanted = tuple(norm(value) for value in variants if value)
    if not wanted:
        return None
    scan_limit = min(header_limit, len(grid))
    for row_index in range(1, scan_limit):
        if (any(parse_vn_number(cell) is not None for cell in grid[row_index])
                and not _period_header_row(grid[row_index])):
            scan_limit = row_index
            break
    scan_limit = max(1, scan_limit)
    width = max((len(row) for row in grid), default=0)
    # Variants are ordered from most specific to fallback. Returning on the
    # first unique match prevents a broad fallback such as "year end" or the
    # bare year from also selecting the comparative column.
    for target in wanted:
        columns = []
        for col in range(width):
            headers = [norm(grid[row][col]) for row in range(scan_limit)
                       if col < len(grid[row])]
            combined = " ".join(header for header in headers if header)
            if target and (any(target in header for header in headers)
                           or target in combined):
                columns.append(col)
        if len(columns) == 1:
            return columns[0]
    return None


def _period_header_row(row: list[str]) -> bool:
    values = [re.sub(r"\s+", "", norm(cell)) for cell in row if norm(cell)]
    periods = [
        value for value in values
        if re.fullmatch(r"(?:nam)?20\d{2}(?:vnd)?", value) is not None
    ]
    # Multi-row statement headers repeat descriptor columns ("Ma so",
    # "Thuyet minh") beside two period cells. Those descriptors must not make
    # the second header row look like the first numeric data row.
    return len(periods) >= 2 or (bool(values) and len(periods) == len(values))


def _row_matches(row: list[str], variants: tuple[str, ...], mode: str) -> bool:
    if not variants:
        return False
    wanted = tuple(norm(value) for value in variants if value)
    for cell in row[:4]:
        value = norm(cell)
        if mode == "prefix" and any(value.startswith(target) for target in wanted):
            return True
        if mode == "contains" and any(target in value for target in wanted):
            return True
        if mode == "exact" and value in wanted:
            return True
    return False


def _resolved_grid_cell(
        table: dict, ticker: str, year: int, metric: str,
        row: int, col: int) -> ResolvedFact | None:
    try:
        frame = df_roundtrip(str(table.get("csv_text") or ""))
    except Exception:
        return None
    found = frame.loc[(frame["row"] == row) & (frame["col"] == col)]
    if len(found) != 1:
        return None
    value = found.iloc[0]
    label = value.get("label")
    label = "" if not isinstance(label, str) else label
    code = value.get("code")
    code = "" if not isinstance(code, str) and code != code else str(code)
    return ResolvedFact(
        ticker=ticker, year=year, metric=metric,
        var=str(table["var"]), report_id=str(table["report_id"]),
        table_pos=int(table["table_pos"]), row=int(row), label=label,
        code=_clean_code(code), col=int(col),
        col_name=str(value.get("col_name") or ""),
        value=float(value["value"]), unit_scale=float(value["unit_scale"]),
        score=96.0, year_evidence=_YEAR_STRONG,
    )


def _tables_for(tables: list[dict], ticker: str, year: int | None) -> list[dict]:
    """Tables belonging to this fact's company (and, if possible, its year).

    report_id looks like TICKER_financial_statements_YYYY_{consolidated,separate}
    so the ticker/year filter is exact rather than fuzzy.
    """
    if not ticker:
        return tables
    same_ticker = [t for t in tables
                   if str(t["report_id"]).split("_")[0].upper() == ticker.upper()]
    if not same_ticker:
        return []
    if year is None:
        return same_ticker
    same_year = [t for t in same_ticker if f"_{year}_" in f"_{t['report_id']}_"
                 or str(t.get("report_year")) == str(year)]
    # the FY-Y figure also lives in the Y+1 report's prior-year column
    if not same_year:
        same_year = [t for t in same_ticker
                     if str(t.get("report_year")) == str(year + 1)]
    return same_year or same_ticker


def resolve_fact(fact, tables: list[dict], metric_variants: list[str],
                 encoder=None, min_score: float = 62.0,
                 question: str = "") -> ResolvedFact | None:
    """Locate the cell for one Fact. Returns None when nothing clears min_score."""
    scoped = _tables_for(tables, fact.ticker, fact.year)
    if not scoped:
        return None
    variants = [v for v in (metric_variants or [fact.metric]) if v]
    cands = build_shortlist(scoped, variants, [fact.year] if fact.year else [],
                            top_n=6, encoder=encoder, min_score=min_score,
                            question=question)
    if not cands:
        return None
    best = cands[0]
    year_ev = _year_evidence(best.col_name, fact.year, best.report_id)
    return ResolvedFact(
        ticker=fact.ticker, year=fact.year, metric=fact.metric,
        var=best.var, report_id=best.report_id, table_pos=best.table_pos,
        row=best.row, label=best.label, code=best.code, col=best.col,
        col_name=best.col_name,
        value=best.value, unit_scale=best.unit_scale, score=best.score,
        year_evidence=year_ev)


def resolve_requirement(requirement: dict, tables: list[dict], encoder=None,
                        min_score: float = 62.0, question: str = "",
                        ambiguity_gap: float = 3.0) -> ResolvedFact | None:
    """Resolve one canonical requirement, refusing fuzzy or ambiguous rows.

    This stricter path is intended for deterministic projections such as
    argmax(metric by year). It accepts only canonical row identity, strong
    period evidence and one unambiguous value for the requested report year.
    """
    ticker = str(requirement.get("ticker") or "").upper()
    raw_year = requirement.get("year")
    year = int(raw_year) if raw_year is not None else None
    metric_key = str(requirement.get("metric_key") or "")
    if not ticker or year is None or not metric_key:
        return None

    doc_type = str(requirement.get("doc_type") or "")
    scoped = _tables_for_exact_year(tables, ticker, year, doc_type)
    if not scoped:
        return None
    metric = get_metric(metric_key)
    expected_codes = set(metric.codes)
    coded, code_candidates_seen = _resolve_requirement_by_code(
        scoped, ticker, year, metric_key,
        str(requirement.get("metric_label") or metric_key), expected_codes,
        metric.statement,
    )
    if code_candidates_seen:
        return coded
    grid_coded, grid_code_seen = _resolve_requirement_by_grid_code(
        scoped, ticker, year, metric_key,
        str(requirement.get("metric_label") or metric_key), expected_codes,
    )
    if grid_code_seen:
        return grid_coded
    typed_column, typed_column_seen = _resolve_requirement_by_typed_column(
        scoped, ticker, year, metric_key,
        str(requirement.get("metric_label") or metric_key), metric,
    )
    if typed_column_seen:
        return typed_column
    if metric_key in {"basic_eps", "deferred_allocation_expense"}:
        stored_as_code = _resolve_numeric_value_stored_as_code(
            scoped, ticker, year, metric_key,
            str(requirement.get("metric_label") or metric_key))
        if stored_as_code is not None:
            return stored_as_code
    variants = requirement_linking_variants(requirement)
    if not variants:
        return None
    candidates = build_shortlist(
        scoped, variants, [year], top_n=16, encoder=encoder,
        min_score=min(35.0, min_score), question=question,
    )
    exact = [
        candidate for candidate in candidates
        if ((expected_codes
             and _clean_code(candidate.code) in expected_codes)
            or (candidate_matches_requirement(candidate, requirement)
                and _canonical_label_is_exact(
                    candidate.label, candidate.code, metric_key)))
        and _candidate_context_is_exact(candidate.var, scoped, metric_key)
        and _candidate_period_is_exact(candidate, scoped, year, metric)
    ]
    if not exact:
        return None

    if metric.column_phrases:
        exact = [candidate for candidate in exact
                 if _candidate_column_matches(
                     candidate, scoped, metric.column_phrases)]
        if not exact:
            return None

    coded = [
        candidate for candidate in exact
        if _clean_code(candidate.code) in expected_codes
    ]
    if coded:
        # A VAS line code is stronger row identity than an unnumbered note row
        # with the same wording. This avoids treating consolidated code 60 as
        # ambiguous with a parent-company profit row in an EPS note.
        exact = coded

    # Prefer the requested year's own filing. Use the following filing's
    # prior-year column only when the requested filing has no exact candidate.
    current = [candidate for candidate in exact
               if _report_year(candidate.report_id) == year]
    exact = current or [candidate for candidate in exact
                        if _report_year(candidate.report_id) == year + 1]
    if not exact:
        return None
    best = exact[0]

    # Duplicate renderings of the same cell/value are harmless. Competing
    # exact-looking rows with different labels or values are not.
    for candidate in exact[1:]:
        if candidate.score < best.score - ambiguity_gap:
            break
        candidate_code = _clean_code(candidate.code)
        best_code = _clean_code(best.code)
        same_code = (
            best_code not in {"", "nan", "none"}
            and candidate_code == best_code
        )
        both_expected_codes = (
            candidate_code in expected_codes and best_code in expected_codes
        )
        same_identity = same_code or (
            norm(candidate.label) == norm(best.label)
            and candidate_code == best_code
        )
        same_value = math.isclose(
            candidate.value * candidate.unit_scale,
            best.value * best.unit_scale,
            rel_tol=1e-9,
            abs_tol=1e-6,
        )
        if not ((same_identity or both_expected_codes) and same_value):
            return None

    return ResolvedFact(
        ticker=ticker, year=year,
        metric=str(requirement.get("metric_label") or metric_key),
        var=best.var, report_id=best.report_id, table_pos=best.table_pos,
        row=best.row, label=best.label, code=best.code, col=best.col,
        col_name=best.col_name, value=best.value, unit_scale=best.unit_scale,
        score=best.score, year_evidence=_YEAR_STRONG,
    )


def _resolve_requirement_by_code(
        tables: list[dict], ticker: str, year: int, metric_key: str,
        metric_label: str, expected_codes: set[str], statement: str,
        ) -> tuple[ResolvedFact | None, bool]:
    """Resolve exact VAS rows even when OCR destroyed the descriptive label."""
    if not expected_codes:
        return None, False

    candidates: list[ResolvedFact] = []
    for table in tables:
        context = str(table.get("context") or "")
        if not _coded_statement_matches(statement, context):
            continue
        if not _candidate_context_is_exact(
                str(table.get("var") or ""), tables, metric_key):
            continue
        try:
            frame = df_roundtrip(str(table.get("csv_text") or ""))
        except Exception:
            continue
        for value in frame.to_dict("records"):
            code = _clean_code(value.get("code"))
            if code not in expected_codes:
                continue
            label = value.get("label")
            label = "" if not isinstance(label, str) else label
            labelled_metrics = set(metric_keys([label], expand_derived=False))
            if labelled_metrics and metric_key not in labelled_metrics:
                continue
            col_name = str(value.get("col_name") or "")
            report_id = str(table.get("report_id") or "")
            if not (_coded_period_matches(col_name, year, report_id)
                    or _grid_period_matches(table, int(value["col"]), year,
                                            report_id)):
                continue
            raw_value = value.get("value")
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue
            candidates.append(ResolvedFact(
                ticker=ticker, year=year, metric=metric_label,
                var=str(table["var"]), report_id=report_id,
                table_pos=int(table["table_pos"]), row=int(value["row"]),
                label=label, code=code, col=int(value["col"]),
                col_name=col_name, value=numeric_value,
                unit_scale=float(value.get("unit_scale") or 1.0),
                score=99.0, year_evidence=_YEAR_STRONG,
            ))
    if not candidates:
        return None, False

    # Some OCR statement headers lose both year numbers and repeat a generic
    # "at 31 December" title for current/prior columns. Financial statements
    # put the current-period value first, so retain the leftmost cell for the
    # same coded row/header identity before checking cross-table conflicts.
    collapsed: dict[tuple, ResolvedFact] = {}
    for candidate in candidates:
        key = (
            candidate.report_id, candidate.table_pos, candidate.row,
            candidate.code, norm(candidate.col_name),
        )
        previous = collapsed.get(key)
        if previous is None or candidate.col < previous.col:
            collapsed[key] = candidate
    candidates = list(collapsed.values())

    # A dated closing-balance header is stronger than another same-year date
    # on the same coded row (for example 31/12/2018 versus a 01/06/2018
    # transition balance). Keep the latter only when no closing header exists.
    closing = [
        candidate for candidate in candidates
        if (_period_kind(candidate.col_name) == "current"
            and not norm(candidate.col_name).startswith("don vi tinh"))
    ]
    candidates = closing or candidates

    current = [candidate for candidate in candidates
               if _report_year(candidate.report_id) == year]
    candidates = current or [candidate for candidate in candidates
                             if _report_year(candidate.report_id) == year + 1]
    if not candidates:
        return None, True
    best = candidates[0]
    for candidate in candidates[1:]:
        if not math.isclose(
                candidate.value_vnd, best.value_vnd,
                rel_tol=1e-9, abs_tol=1e-6):
            return None, True
    return best, True


def _resolve_requirement_by_typed_column(
        tables: list[dict], ticker: str, year: int, metric_key: str,
        metric_label: str, metric) -> tuple[ResolvedFact | None, bool]:
    """Resolve note matrices whose metric identity spans row and column axes."""
    if not metric.column_phrases:
        return None, False

    candidates: list[ResolvedFact] = []
    for table in tables:
        if not _candidate_context_is_exact(
                str(table.get("var") or ""), tables, metric_key):
            continue
        try:
            frame = df_roundtrip(str(table.get("csv_text") or ""))
        except Exception:
            continue
        for value in frame.to_dict("records"):
            label = value.get("label")
            label = "" if not isinstance(label, str) else label
            code = _clean_code(value.get("code"))
            if not _canonical_label_is_exact(label, code, metric_key):
                continue
            raw_value = value.get("value")
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(numeric_value):
                continue
            candidate = ResolvedFact(
                ticker=ticker, year=year, metric=metric_label,
                var=str(table["var"]), report_id=str(table["report_id"]),
                table_pos=int(table["table_pos"]), row=int(value["row"]),
                label=label, code=code, col=int(value["col"]),
                col_name=str(value.get("col_name") or ""),
                value=numeric_value,
                unit_scale=float(value.get("unit_scale") or 1.0),
                score=99.0, year_evidence=_YEAR_STRONG,
            )
            if not _candidate_column_matches(
                    candidate, tables, metric.column_phrases):
                continue
            if not _candidate_period_is_exact(
                    candidate, tables, year, metric):
                continue
            candidates.append(candidate)
    if not candidates:
        return None, False

    current = [candidate for candidate in candidates
               if _report_year(candidate.report_id) == year]
    candidates = current or [candidate for candidate in candidates
                             if _report_year(candidate.report_id) == year + 1]
    if not candidates:
        return None, True
    best = candidates[0]
    for candidate in candidates[1:]:
        if not math.isclose(
                candidate.value_vnd, best.value_vnd,
                rel_tol=1e-9, abs_tol=1e-6):
            return None, True
    return best, True


def _resolve_requirement_by_grid_code(
        tables: list[dict], ticker: str, year: int, metric_key: str,
        metric_label: str, expected_codes: set[str],
        ) -> tuple[ResolvedFact | None, bool]:
    """Recover exact VAS rows when OCR damaged context or shifted the code.

    Some statements lose their title during OCR, while others retain a leading
    printed row number before the actual VAS code. The tidy serializer then
    stores that row number as ``code``. Raw-grid acceptance is deliberately
    strict: canonical label, expected VAS code, target-period header and the
    corresponding tidy CSV cell must all agree.
    """
    if not expected_codes:
        return None, False

    candidates: list[ResolvedFact] = []
    matching_rows_seen = False
    for table in tables:
        report_id = str(table.get("report_id") or "")
        try:
            grid = json.loads(str(table.get("grid_json") or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(grid, list):
            continue
        for row_index, row in enumerate(grid):
            if not isinstance(row, list):
                continue
            raw_codes = [
                _clean_code(cell) for cell in row[:4]
                if _clean_code(cell) in expected_codes
            ]
            raw_labels = [
                str(cell) for cell in row[:4]
                if _canonical_label_is_exact(str(cell), "", metric_key)
            ]
            if len(set(raw_codes)) != 1 or not raw_labels:
                continue
            matching_rows_seen = True
            raw_code = raw_codes[0]
            for col in range(len(row)):
                if not _grid_period_matches(table, col, year, report_id):
                    continue
                fact = _resolved_grid_cell(
                    table, ticker, year, metric_label, row_index, col)
                if fact is None:
                    continue
                candidates.append(ResolvedFact(
                    ticker=fact.ticker, year=fact.year, metric=fact.metric,
                    var=fact.var, report_id=fact.report_id,
                    table_pos=fact.table_pos, row=fact.row,
                    label=fact.label, code=raw_code, col=fact.col,
                    col_name=fact.col_name, value=fact.value,
                    unit_scale=fact.unit_scale, score=99.0,
                    year_evidence=_YEAR_STRONG,
                ))
    if not candidates:
        return None, matching_rows_seen

    current = [candidate for candidate in candidates
               if _report_year(candidate.report_id) == year]
    candidates = current or [candidate for candidate in candidates
                             if _report_year(candidate.report_id) == year + 1]
    if not candidates:
        return None, True
    best = candidates[0]
    if any(not math.isclose(
            candidate.value_vnd, best.value_vnd,
            rel_tol=1e-9, abs_tol=1e-6) for candidate in candidates[1:]):
        return None, True
    return best, True


def _resolve_numeric_value_stored_as_code(
        tables: list[dict], ticker: str, year: int, metric_key: str,
        metric_label: str) -> ResolvedFact | None:
    """Recover a current-period value swallowed by the legacy code heuristic.

    This is intentionally allowlisted to metrics observed in three-column
    tables where a value such as ``1.551`` is parsed as a row code while the
    prior value remains the only tidy value cell. The emitted expression reads
    that existing CSV ``code`` cell, so global serialization remains unchanged.
    """
    candidates: list[ResolvedFact] = []
    for table in tables:
        report_id = str(table.get("report_id") or "")
        if _report_year(report_id) != year:
            continue
        try:
            frame = df_roundtrip(str(table.get("csv_text") or ""))
            grid = json.loads(str(table.get("grid_json") or "[]"))
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
        if not isinstance(grid, list):
            continue
        for value in frame.to_dict("records"):
            label = value.get("label")
            label = "" if not isinstance(label, str) else label
            if not _canonical_label_is_exact(label, "", metric_key):
                continue
            try:
                row_index = int(value["row"])
                numeric_code = float(value.get("code"))
            except (TypeError, ValueError):
                continue
            if (not math.isfinite(numeric_code) or numeric_code == 0
                    or row_index >= len(grid)):
                continue
            raw_row = grid[row_index]
            numeric_columns = [
                col for col, cell in enumerate(raw_row)
                if parse_vn_number(str(cell)) is not None
            ]
            if not numeric_columns:
                continue
            current_col = numeric_columns[0]
            raw_cell = str(raw_row[current_col]).strip()
            raw_value = parse_vn_number(raw_cell)
            try:
                csv_value = float(raw_cell)
            except ValueError:
                csv_value = numeric_code
            if (raw_value is None or numeric_code == 0
                    or not math.isclose(
                        csv_value, numeric_code,
                        rel_tol=1e-12, abs_tol=1e-6)):
                continue
            multiplier = float(raw_value) / numeric_code
            header = " ".join(
                str(header_row[current_col]).strip()
                for header_row in grid[:3]
                if isinstance(header_row, list)
                and current_col < len(header_row)
                and str(header_row[current_col]).strip()
            )
            if not _coded_period_matches(header, year, report_id):
                continue
            candidates.append(ResolvedFact(
                ticker=ticker, year=year, metric=metric_label,
                var=str(table["var"]), report_id=report_id,
                table_pos=int(table["table_pos"]), row=row_index,
                label=label, code="", col=current_col, col_name=header,
                value=float(raw_value),
                unit_scale=float(value.get("unit_scale") or 1.0),
                score=99.0, year_evidence=_YEAR_STRONG,
                value_column="code", value_multiplier=multiplier,
            ))
    if not candidates:
        return None
    best = candidates[0]
    if any(not math.isclose(
            candidate.value_vnd, best.value_vnd,
            rel_tol=1e-9, abs_tol=1e-6) for candidate in candidates[1:]):
        return None
    return best


def _coded_statement_matches(statement: str, context: str) -> bool:
    value = norm(context)
    markers = {
        "balance_sheet": (
            "bang can doi ke toan", "bao cao tinh hinh tai chinh",
        ),
        "income_statement": (
            "bao cao ket qua hoat dong kinh doanh", "ket qua kinh doanh",
        ),
        "cash_flow": (
            "bao cao luu chuyen tien te", "luu chuyen tien te",
        ),
    }
    required = markers.get(str(statement or ""))
    return bool(required) and any(marker in value for marker in required)


def _grid_period_matches(table: dict, col: int, year: int,
                         report_id: str) -> bool:
    """Recover split multi-row headers such as ``Tại ngày``/``31.12.2024``."""
    try:
        grid = json.loads(str(table.get("grid_json") or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(grid, list):
        return False
    parts = [
        str(row[col]).strip() for row in grid[:3]
        if isinstance(row, list) and col < len(row) and str(row[col]).strip()
    ]
    return _coded_period_matches(" ".join(parts), year, report_id)


def _coded_period_matches(col_name: str, year: int, report_id: str) -> bool:
    """Require the target fiscal period, excluding opening-balance columns."""
    explicit_years = {
        int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", col_name)
    }
    if explicit_years and year not in explicit_years:
        return False
    if _year_evidence(col_name, year, report_id) != _YEAR_STRONG:
        return False
    report_year = _report_year(report_id)
    period_kind = _period_kind(col_name)
    if report_year == year:
        return period_kind != "prior"
    if report_year == year + 1:
        return (period_kind == "prior"
                or bool(re.search(rf"(?<!\d){year}(?!\d)", col_name)))
    return False


def _tables_for_exact_year(tables: list[dict], ticker: str, year: int,
                           doc_type: str = "") -> list[dict]:
    out = []
    for table in tables:
        report_id = str(table.get("report_id") or "")
        if report_id.split("_")[0].upper() != ticker:
            continue
        if doc_type in {"consolidated", "separate", "aggregated"}:
            if not re.search(rf"_{re.escape(doc_type)}(?:_|$)", report_id):
                continue
        report_year = _report_year(report_id)
        if report_year is None:
            raw = table.get("report_year")
            report_year = int(raw) if raw is not None else None
        if report_year in {year, year + 1}:
            out.append(table)
    return out


def _report_year(report_id: str) -> int | None:
    found = re.search(r"(?:financial_statements_|_)(20\d{2})(?:_|$)",
                      str(report_id))
    return int(found.group(1)) if found else None


def _clean_code(code: str) -> str:
    return re.sub(r"\.0$", "", str(code or "").strip())


def _canonical_label_is_exact(label: str, code: str, metric_key: str) -> bool:
    """Reject child/detail rows that merely contain a parent metric alias."""
    raw_label = str(label).strip()
    # OCR can fold an empty current-period cell into the label and leave the
    # prior-period value as the first numeric candidate.
    if re.search(r"(?:^|\s)[-\u2013\u2014]\s*$", raw_label):
        return False

    metric = get_metric(metric_key)
    clean_code = _clean_code(code)
    if clean_code.isdigit() and metric.codes:
        if clean_code in metric.codes:
            return True
        # A short number beside a three-digit balance-sheet line is commonly
        # a note reference, not a conflicting VAS code. Keep lexical identity
        # available for that narrow case.
        expected_are_vas = all(len(value) == 3 for value in metric.codes)
        if not (metric.statement == "balance_sheet"
                and expected_are_vas and len(clean_code) < 3):
            return False

    label_norm = norm(label)
    if any(value in label_norm for value in metric.forbidden_phrases):
        return False
    label_tokens = tokens(label_norm)
    while label_tokens and (
        label_tokens[0].isdigit()
        or len(label_tokens[0]) == 1
        or re.fullmatch(r"[ivxlcdm]+", label_tokens[0])
    ):
        label_tokens.pop(0)
    clean_label = " ".join(label_tokens)
    variants = list(dict.fromkeys([
        metric.label,
        *metric.row_aliases,
        *(value for value in metric.variants if len(tokens(value)) >= 3),
    ]))
    exact_only = {"tong cong", "tong", "so cuoi nam", "gia goc"}
    return any(
        clean_label == variant
        if variant in exact_only
        else (clean_label == variant or clean_label.startswith(f"{variant} "))
        for variant in variants
    )


def _candidate_context_is_exact(var: str, tables: list[dict],
                                metric_key: str) -> bool:
    table = next((table for table in tables if table.get("var") == var), None)
    return bool(table) and metric_context_matches(
        metric_key, str(table.get("context") or ""))


def _candidate_period_is_exact(candidate, tables: list[dict], year: int,
                               metric) -> bool:
    table = next((item for item in tables
                  if item.get("var") == candidate.var), None)
    if table is None:
        return False
    column_text = _candidate_column_text(candidate, table)
    if _period_kind(column_text) == "prior":
        return False
    if _year_evidence(
            candidate.col_name, year, candidate.report_id) == _YEAR_STRONG:
        return True
    if _grid_period_matches(
            table, int(candidate.col), year, candidate.report_id):
        return True
    if not metric.column_phrases or not _candidate_column_matches(
            candidate, tables, metric.column_phrases):
        return False
    context = norm(str(table.get("context") or ""))
    return (_report_year(candidate.report_id) == year
            and bool(re.search(rf"(?<!\d){year}(?!\d)", context)))


def _candidate_column_matches(candidate, tables: list[dict],
                              phrases: tuple[str, ...]) -> bool:
    table = next((item for item in tables
                  if item.get("var") == candidate.var), None)
    if table is None:
        return False
    column_text = _candidate_column_text(candidate, table)
    return any(phrase in column_text for phrase in phrases)


def _candidate_column_text(candidate, table: dict) -> str:
    parts = [str(candidate.col_name or "")]
    try:
        grid = json.loads(str(table.get("grid_json") or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        grid = []
    for row in grid[:3] if isinstance(grid, list) else []:
        if isinstance(row, list) and int(candidate.col) < len(row):
            parts.append(str(row[int(candidate.col)] or ""))
    return norm(" ".join(parts))


def _year_evidence(col_name: str, year: int | None,
                   report_id: str = "") -> int:
    if year is None:
        return _YEAR_WEAK
    cn = str(col_name)
    if (re.search(rf"31\s*[./-]\s*12\s*[./-]\s*{year}", cn)
            or re.search(rf"(?<!\d){year}(?!\d)", cn)):
        return _YEAR_STRONG
    found = re.search(r"(?:financial_statements_|_)(20\d{2})(?:_|$)",
                      str(report_id))
    report_year = int(found.group(1)) if found else None
    kind = _period_kind(cn)
    if report_year == year and kind == "current":
        return _YEAR_STRONG
    if report_year == year + 1 and kind == "prior":
        return _YEAR_STRONG
    return _YEAR_WEAK


def resolve_all(facts, tables: list[dict], metric_variants: list[str],
                encoder=None, min_score: float = 62.0,
                question: str = ""):
    """(resolved list, confidence 0..100). Confidence is driven by the WEAKEST
    fact: a composite answer is only as trustworthy as its worst operand."""
    out = []
    for f in facts:
        r = resolve_fact(f, tables, metric_variants, encoder, min_score,
                         question=question)
        if r is None:
            return out, 0.0
        out.append(r)
    if not out:
        return out, 0.0
    weakest = min(r.score for r in out)
    year_ok = all(r.year_evidence == _YEAR_STRONG for r in out)
    conf = min(99.0, weakest + (8.0 if year_ok else -10.0))
    # distinct facts must not collapse onto the same cell (a classic failure of
    # multi-entity questions where only one company's tables were retrieved)
    keys = {(r.report_id, r.table_pos, r.row, r.label, r.col) for r in out}
    if len(keys) < len(out):
        conf = min(conf, 35.0)
    return out, max(0.0, conf)


def distinct_cells(resolved: list[ResolvedFact]) -> bool:
    return len({(r.report_id, r.table_pos, r.row, norm(r.label), r.col)
                for r in resolved}) == len(resolved)
