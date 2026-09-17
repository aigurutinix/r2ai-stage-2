"""Metadata-gated BM25 retrieval over OCR table rows."""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
from pathlib import Path
import re
import statistics
import time
import unicodedata

from .tables import PAGE_RE, TABLE_RE, ReportIdentity, iter_report_paths, parse_report_identity, parse_table_rows


TOKEN_RE = re.compile(r"[a-z0-9%]{2,}")
UNICODE_TOKEN_RE = re.compile(r"[^\W_]{2,}", re.UNICODE)
STOPWORDS = {
    "bao", "nhieu", "la", "cua", "cho", "trong", "nam", "vao", "den", "ngay",
    "cong", "ty", "vnd", "dong", "tyle", "phan", "tram", "trieu", "nghin",
}
CONTEXT_STOPWORDS = STOPWORDS - {"vnd", "dong", "tyle", "phan", "tram", "trieu", "nghin"}
METRIC_STOPWORDS = CONTEXT_STOPWORDS | {
    "tinh", "hay", "biet", "gia", "tri", "muc", "do", "so", "voi", "giua",
    "tu", "den", "vao", "tai", "cuoi", "dau", "ky", "theo", "tren",
    "duoi", "tang", "giam", "truong", "ty", "le", "phan", "tram",
    "chenh", "lech", "binh", "quan", "lon", "nho", "cao", "thap",
}
ROLE_STOPWORDS = {
    "cac", "co", "ghi", "nhan", "so", "gia", "tri", "nao", "cao", "thap", "lon", "nho",
    "hon", "nhat", "me", "tap", "doan", "ctcp", "ma", "tinh", "va",
}
PERIOD_RE = re.compile(r"\b(?:19|20)\d{2}\b|\b\d{1,2}/\d{1,2}/(?:19|20)\d{2}\b")
# A financial figure: at least two digits, so account codes and list numbering
# do not make a prose block look like a data table.
NUMERIC_CELL_RE = re.compile(r"\d[\d.,]*\d")
TITLE_RE = re.compile(r"\b(?:bao cao|bang|thuyet minh)\b")
UNIT_PHRASES = ("vnd", "don vi", "trieu dong", "nghin dong", "ty dong", "million", "billion")
HEADER_TOKENS = {"ma", "so", "thuyet", "minh", "chi", "tieu", "don", "vi"}
RRF_OFFSET = 60
# How many rows of a table contribute to its score. Questions cite at most a
# handful of line items, so counting more rows would reward long tables for length.
SUPPORTING_ROWS = 3
# Equal-start weights for field-aware fusion; tune on dev only.
FIELD_WEIGHTS = {
    "row": 4.0,
    "title": 4.0,
    "header": 4.0,
    "unit": 1.0,
    "phrase": 4.0,
    "rrf": 4.0,
}
RANK_FUSION_FAMILIES = {
    "row": ("folded_row", "unicode_row"),
    "context": ("folded_context", "unicode_context"),
    "metadata": ("title", "header", "unit"),
}


@dataclass(frozen=True)
class Report:
    identity: ReportIdentity
    path: Path


@dataclass(frozen=True)
class Table:
    table_id: str
    report_id: str
    page: int | None
    start_line: int
    rows: tuple[tuple[str, ...], ...]
    title: str
    context: tuple[str, ...]
    headers: tuple[tuple[str, ...], ...]
    periods: tuple[str, ...]
    unit: str


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", text.lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")
    return TOKEN_RE.findall(text)


def unicode_tokenize(text: str) -> list[str]:
    """Tokenize NFC text without removing Vietnamese diacritics."""
    return UNICODE_TOKEN_RE.findall(unicodedata.normalize("NFC", text.lower()))


def load_reports(dataset_root: Path) -> list[Report]:
    return [Report(parse_report_identity(path, dataset_root), path) for path in iter_report_paths(dataset_root)]


def filter_reports(reports: list[Report], metadata: dict) -> tuple[list[Report], str]:
    tickers, years, scope = set(metadata.get("tickers", [])), set(metadata.get("years", [])), metadata.get("scope")
    stages = (
        ("ticker_year_scope", lambda report: (not tickers or report.identity.ticker in tickers) and (not years or report.identity.year in years) and (scope is None or report.identity.scope == scope)),
        ("ticker_year", lambda report: (not tickers or report.identity.ticker in tickers) and (not years or report.identity.year in years)),
        ("ticker", lambda report: not tickers or report.identity.ticker in tickers),
        ("year", lambda report: not years or report.identity.year in years),
        ("global", lambda report: True),
    )
    for stage, matches in stages:
        candidates = [report for report in reports if matches(report)]
        if candidates:
            return candidates, stage
    return [], "empty"


def table_context(lines: list[str]) -> tuple[str, tuple[str, ...]]:
    lines = [line for line in lines if line and not PAGE_RE.fullmatch(line)]
    title_lines = [line for line in lines if TITLE_RE.search(" ".join(tokenize(line)))]
    return (title_lines[-1] if title_lines else lines[-1] if lines else ""), tuple(lines[-8:])


def table_metadata(title: str, context: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...], str]:
    headers = tuple(
        row for row in rows[:3]
        if PERIOD_RE.search(" ".join(row)) or HEADER_TOKENS & set(tokenize(" ".join(row)))
    ) or rows[:1]
    source = " ".join((title, *context, *(" ".join(row) for row in headers)))
    periods = tuple(dict.fromkeys(PERIOD_RE.findall(source)))
    unit = " ".join(
        line for line in (title, *context, *(" ".join(row) for row in headers))
        if "%" in line or any(phrase in " ".join(tokenize(line)) for phrase in UNIT_PHRASES)
    )
    return headers, periods, unit


# Corpus contains more than 256 reports; retain parsed tables for one full run.
@lru_cache(maxsize=4096)
def report_tables(path_text: str, identity: ReportIdentity) -> tuple[Table, ...]:
    text = Path(path_text).read_text(encoding="utf-8")
    pages = list(PAGE_RE.finditer(text))
    page_index, page, cursor, line, context_cursor = 0, None, 0, 1, 0
    context_lines: list[str] = []
    tables = []
    for match in TABLE_RE.finditer(text):
        context_lines.extend(" ".join(value.split()) for value in text[context_cursor:match.start()].splitlines())
        context_lines = context_lines[-16:]
        line += text.count("\n", cursor, match.start())
        cursor = match.start()
        while page_index < len(pages) and pages[page_index].start() < match.start():
            page = int(pages[page_index].group(1))
            page_index += 1
        rows = tuple(tuple(row) for row in parse_table_rows(match.group(0)))
        title, context = table_context(context_lines)
        headers, periods, unit = table_metadata(title, context, rows)
        tables.append(Table(
            f"{identity.report_id}|{line}", identity.report_id, page, line, rows,
            title, context, headers, periods, unit,
        ))
        context_cursor = match.end()
    return tuple(tables)


def query_tokens(question: str, metadata: dict) -> list[str]:
    blocked = {str(year) for year in metadata.get("years", [])}
    blocked.update(ticker.lower() for ticker in metadata.get("tickers", []))
    return [token for token in tokenize(question) if token not in STOPWORDS and token not in blocked and not token.isdigit()]


def context_query_tokens(question: str, metadata: dict) -> list[str]:
    blocked = {ticker.lower() for ticker in metadata.get("tickers", [])}
    return [token for token in tokenize(question.replace("%", " percent ")) if token not in CONTEXT_STOPWORDS and token not in blocked]


def metric_query_tokens(question: str, metadata: dict, *, keep_years: bool = False) -> list[str]:
    """Keep line-item terms while removing arithmetic wording from a question."""
    blocked = {ticker.lower() for ticker in metadata.get("tickers", [])}
    if not keep_years:
        blocked.update(str(year) for year in metadata.get("years", []))
    return [
        token for token in tokenize(question.replace("%", " percent "))
        if token not in METRIC_STOPWORDS and token not in blocked and (keep_years or not token.isdigit())
    ]


def unicode_query_tokens(question: str, metadata: dict, *, keep_context: bool = False) -> list[str]:
    """Mirror folded query filtering while retaining NFC Vietnamese tokens."""
    blocked = set() if keep_context else {str(year) for year in metadata.get("years", [])}
    blocked.update(ticker.lower() for ticker in metadata.get("tickers", []))
    stopwords = CONTEXT_STOPWORDS if keep_context else STOPWORDS
    result = []
    for token in unicode_tokenize(question.replace("%", " percent ")):
        folded = tokenize(token)
        folded_token = folded[0] if folded else token
        if folded_token in stopwords or folded_token in blocked:
            continue
        if not keep_context and token.isdigit():
            continue
        result.append(token)
    return result


def contextual_prefix(table: Table) -> str:
    return " ".join((
        table.title,
        *table.context,
        *(" ".join(header) for header in table.headers),
        " ".join(table.periods),
        table.unit,
    )).replace("%", " percent ")


def contextual_row(table: Table, row: str) -> str:
    return f"{contextual_prefix(table)} {row}"


def score_bm25(terms: set[str], tokens: list[list[str]]) -> list[float]:
    """Score tokenized rows against the candidate slice.

    Corpus-wide document frequency was measured on the dev split and rejected:
    it lifted candidate recall@50 from 0.9052 to 0.9122 but dropped submitted F2
    from 0.5343 to 0.4923. Statistics local to the gated slice downweight terms
    that are boilerplate within the company's own reports, which is exactly the
    discrimination top-k ranking needs.
    """
    ordered_terms = sorted(terms)
    document_frequency = Counter(
        term for row in tokens for term in set(row) if term in terms
    )
    average_length = statistics.fmean(max(1, len(row)) for row in tokens)
    scores = []
    for row in tokens:
        counts = Counter(row)
        score, length = 0.0, max(1, len(row))
        # Floating-point addition is order-sensitive. Iterating a set made a
        # tied row depend on PYTHONHASHSEED, which changed the text shown to the
        # reranker even when the selected table was identical.
        for term in ordered_terms:
            frequency = counts[term]
            if frequency:
                seen = document_frequency[term]
                idf = math.log(1 + (len(tokens) - seen + 0.5) / (seen + 0.5))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average_length))
        scores.append(score)
    return scores


def bm25(query: list[str], rows: list[str]) -> list[float]:
    if not query or not rows:
        return [0.0] * len(rows)
    return score_bm25(set(query), [tokenize(row) for row in rows])


def unicode_bm25(query: list[str], rows: list[str]) -> list[float]:
    """BM25 variant preserving NFC Vietnamese distinctions."""
    if not query or not rows:
        return [0.0] * len(rows)
    return score_bm25(set(query), [unicode_tokenize(row) for row in rows])


def phrase_bonus(query: list[str], row: str) -> float:
    tokens = tokenize(row)
    bonus = 0.0
    for size, weight in ((3, 1.4), (2, 0.35)):
        query_phrases = {tuple(query[index:index + size]) for index in range(len(query) - size + 1)}
        row_phrases = {tuple(tokens[index:index + size]) for index in range(len(tokens) - size + 1)}
        bonus += len(query_phrases & row_phrases) * weight
    return bonus


def header_cells(table: Table) -> list[str]:
    """Merge every header row into one label per column for cell-level gating."""
    width = max((len(row) for row in table.headers), default=0)
    return [
        " ".join(row[column] for row in table.headers if column < len(row) and row[column])
        for column in range(width)
    ]


FULL_NUMBER_RE = re.compile(r"^\s*[([]?[+-]?\d[\d.,\s]*[%)]?\s*$")
ROW_METADATA_HEADERS = {"ma so", "thuyet minh", "stt", "so thu tu"}


def distinct_path(parts) -> tuple[str, ...]:
    """Keep a hierarchy path while removing empty and span-repeated labels."""
    result: list[str] = []
    seen: set[str] = set()
    for value in parts:
        value = " ".join(value.split())
        key = " ".join(tokenize(value))
        if not value or not key or key in seen:
            continue
        result.append(value)
        seen.add(key)
    return tuple(result)


def column_paths(table: Table) -> tuple[tuple[str, ...], ...]:
    """Return each column's complete top-to-bottom header path.

    Expanded colspans repeat their parent label across child columns. Keeping
    one copy per path preserves the parent-child meaning without adding OCR
    repetition to the retrieval text.
    """
    width = max((len(row) for row in (*table.headers, *table.rows)), default=0)
    return tuple(
        distinct_path(
            row[column] for row in table.headers
            if column < len(row)
        )
        for column in range(width)
    )


def row_path(table: Table, row_index: int) -> tuple[str, ...]:
    """Return the source-derived label path leading to one table row.

    OCR span expansion commonly places a parent group in the first column and
    its child label in the next. Numeric values, account codes, and note-number
    columns are coordinates rather than row labels, so they are excluded.
    """
    if not 0 <= row_index < len(table.rows):
        raise IndexError(row_index)
    paths = column_paths(table)
    labels = []
    for column, cell in enumerate(table.rows[row_index]):
        value = " ".join(cell.split())
        if not value or FULL_NUMBER_RE.fullmatch(value):
            continue
        header = " ".join(tokenize(" ".join(paths[column]))) if column < len(paths) else ""
        if header in ROW_METADATA_HEADERS:
            continue
        labels.append(value)
    return distinct_path(labels)


def row_paths(table: Table) -> tuple[tuple[str, ...], ...]:
    """Propagate explicit section rows to their following numeric children.

    A row with labels but no financial value is a structural heading. It stays
    active until the next such row. Header rows are excluded, and numeric
    account/note columns do not make a heading look like a value row.
    """
    header_rows = set(table.headers)
    columns = column_paths(table)
    active_section: tuple[str, ...] = ()
    paths = []
    for index, row in enumerate(table.rows):
        direct = row_path(table, index)
        if row in header_rows:
            paths.append(direct)
            active_section = ()
            continue
        has_value = any(
            FULL_NUMBER_RE.fullmatch(cell or "")
            and " ".join(tokenize(" ".join(columns[column]))) not in ROW_METADATA_HEADERS
            for column, cell in enumerate(row)
            if column < len(columns)
        )
        if direct and not has_value:
            active_section = direct
            paths.append(direct)
        elif active_section and direct:
            paths.append(distinct_path((*active_section, *direct)))
        else:
            paths.append(direct or active_section)
    return tuple(paths)


def ranked_tables(best: dict[str, tuple[float, Table, int]]) -> list[tuple[float, Table, int]]:
    return sorted(best.values(), key=lambda item: (-item[0], item[1].table_id))


def select_candidate_reports(
    reports: list[Report], metadata: dict, report_ids: list[str] | None,
) -> tuple[list[Report], str]:
    """Stage 1: preserve report-gate behavior for implicit and explicit IDs."""
    if report_ids is None:
        return filter_reports(reports, metadata)
    reports_by_id = {report.identity.report_id: report for report in reports}
    return [reports_by_id[report_id] for report_id in report_ids if report_id in reports_by_id], "report_ids"


def carries_figures(table: Table) -> bool:
    """Reject tables that cannot hold an answer: prose blocks and layout fragments.

    OCR turns headers, signature blocks, and page furniture into <table> elements.
    The corpus holds 8,901 tables of at most two rows and 1,435 with a single
    column, and unclassified fragments were 18.5% of what we submitted against
    7.5% of gold. A table with no numeric cell, or with nothing beside its label
    column, cannot be the evidence for a numeric question.
    """
    if max((len(row) for row in table.rows), default=0) < 2:
        return False
    if is_contents_page(table):
        return False
    return any(NUMERIC_CELL_RE.search(cell) for row in table.rows for cell in row[1:])


PAGE_RANGE_RE = re.compile(r"^\s*\d{1,3}\s*[-–]\s*\d{1,3}\s*$")


def is_contents_page(table: Table) -> bool:
    """Whether the table is the report's index rather than a statement.

    Every report opens with a contents page — "Báo cáo lưu chuyển tiền tệ hợp
    nhất | 10 - 11" — whose row labels are the names of the statements. That is
    the best lexical match a question about cash flow will ever find, so the
    retriever ranks it first, and the answer path then reads a page number as the
    figure. `carries_figures` passed it because page numbers are numeric cells.

    Measured on the shipped submission: 57 of 5,938 tables, across 32 questions,
    each one an answer that could not have been right.
    """
    values = [cell for row in table.rows for cell in row[1:] if cell.strip()]
    if not values:
        return False
    if any(tokenize(cell) == ["trang"] for row in table.rows for cell in row[1:]):
        return True
    # A page column is ranges and small integers; a statement column is not.
    return sum(1 for cell in values if PAGE_RANGE_RE.match(cell)) / len(values) >= 0.5


def materialize_candidate_rows(
    candidates: list[Report], *, hierarchy: bool = False,
) -> tuple[list[Table], list[tuple[Table, int, str]]]:
    """Stage 2: materialize immutable table and row candidates in report order."""
    tables = [
        table for report in candidates
        for table in report_tables(str(report.path), report.identity)
        if carries_figures(table)
    ]
    rows = []
    for table in tables:
        paths = row_paths(table) if hierarchy else ()
        for index, row in enumerate(table.rows):
            raw = " ".join(row)
            path = " > ".join(paths[index]) if hierarchy and paths[index] else ""
            text = f"{path} | {raw}" if path else raw
            rows.append((table, index, text))
    return tables, rows


def baseline_ranked_rows(
    query: list[str], context_query: list[str], rows: list[tuple[Table, int, str]],
    metric_query: list[str] | None = None,
    family_prior: bool = False,
) -> list[tuple[float, Table, int]]:
    """Stage 2 baseline ranker: reciprocal-rank fusion over row, context, and metric views.

    The raw question carries arithmetic wording ("chênh lệch", "tăng trưởng",
    "trung bình") that no statement row contains, and on derived questions that
    wording outweighs the line item being asked for. The metric view drops it.
    Swapping the query for that view outright was measured and rejected: it lifts
    intermediate (+0.0431) and hard (+0.0222) but costs easy (-0.0258), where the
    question already reads like a row label. Fusing it as a third ranking keeps
    both behaviours without a per-question switch.
    """
    scores = bm25(query, [row for _, _, row in rows])
    context_scores = bm25(context_query, [contextual_row(table, row) for table, _, row in rows])
    metric_scores = bm25(metric_query or [], [row for _, _, row in rows])
    family_by_table: dict[str, float] = {}
    if family_prior:
        from .table_families import family_match_score
        family_query = metric_query or query
        family_by_table = {
            table.table_id: family_match_score(family_query, table)
            for table, _, _ in rows
        }
    raw_best: dict[str, tuple[float, Table, int]] = {}
    context_best: dict[str, tuple[float, Table, int]] = {}
    metric_best: dict[str, tuple[float, Table, int]] = {}
    supporting: dict[str, list[float]] = {}
    for (table, row_index, row), raw_score, context_score, metric_score in zip(
        rows, scores, context_scores, metric_scores,
    ):
        raw_score += phrase_bonus(query, row)
        if metric_query:
            metric_score += phrase_bonus(metric_query, row)
        supporting.setdefault(table.table_id, []).append(max(raw_score, metric_score))
        for score, best in ((raw_score, raw_best), (context_score, context_best), (metric_score, metric_best)):
            previous = best.get(table.table_id)
            if previous is None or score > previous[0] or score == previous[0] and row_index < previous[2]:
                best[table.table_id] = score, table, row_index
    # A question naming several line items is answered by the one statement holding
    # all of them, but a table scored only by its best row cannot express that: a
    # note repeating one item ties with a balance sheet carrying three. Replacing
    # the raw view with this one was measured and rejected — it lifts intermediate
    # (+0.0166) and hard (+0.0262) while costing easy (-0.0227) and medium
    # (-0.0160), which name a single item. It earns its place as its own ranking.
    supporting_best = {
        table_id: (sum(sorted(supporting[table_id], reverse=True)[:SUPPORTING_ROWS]), table, row_index)
        for table_id, (_, table, row_index) in raw_best.items()
    }
    raw_ranked = ranked_tables(raw_best)
    context_ranked = ranked_tables(context_best)
    metric_ranked = ranked_tables(metric_best) if metric_query else []
    supporting_ranked = ranked_tables(supporting_best)
    family_best = {
        table_id: (score, table, row_index)
        for table_id, score in family_by_table.items()
        if score > 0
        for _, table, row_index in [raw_best[table_id]]
    }
    family_ranked = ranked_tables(family_best)
    raw_positive = [item for item in raw_ranked if item[0] > 0]
    context_positive = [item for item in context_ranked if item[0] > 0]
    metric_positive = [item for item in metric_ranked if item[0] > 0]
    supporting_positive = [item for item in supporting_ranked if item[0] > 0]
    family_positive = [item for item in family_ranked if item[0] > 0]
    if not context_positive and not metric_positive:
        return raw_ranked
    raw_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(raw_positive, 1)}
    context_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(context_positive, 1)}
    metric_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(metric_positive, 1)}
    supporting_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(supporting_positive, 1)}
    family_ranks = {table.table_id: rank for rank, (_, table, _) in enumerate(family_positive, 1)}
    covered = raw_ranks.keys() | context_ranks.keys() | metric_ranks.keys()
    selected = {
        table_id: (
            raw_best[table_id] if table_id in raw_ranks
            else metric_best[table_id] if table_id in metric_ranks
            else context_best[table_id]
        )
        for table_id in covered
    }
    ranked = []
    for table_id in covered:
        score = sum(
            1 / (RRF_OFFSET + ranks[table_id])
            for ranks in (raw_ranks, context_ranks, metric_ranks, supporting_ranks)
            if table_id in ranks
        )
        if table_id in family_ranks:
            score += 0.25 / (RRF_OFFSET + family_ranks[table_id])
        ranked.append((score, selected[table_id][1], selected[table_id][2]))
    ranked.sort(key=lambda item: (-item[0], item[1].table_id))
    selected_ids = {table.table_id for _, table, _ in ranked}
    ranked.extend(item for item in raw_ranked if item[1].table_id not in selected_ids)
    return ranked


def table_budget(report_count: int, setting: str | int = "auto") -> int:
    """How many tables to submit for one question.

    The ranker returns an ordered list and we have to cut it somewhere. Cutting
    early loses gold tables, cutting late dilutes precision, and F2 trades the
    two.

    The cut scales with the number of reports the gate selected, because a
    five-year comparison needs tables from five filings and a single-filing
    question does not. The multiplier is a measured value, not a derived one.
    """
    if isinstance(setting, str):
        # "auto" or "auto:n" for n tables per gated report; a bare integer fixes
        # the count regardless of how many reports there are.
        _, _, multiplier = setting.partition(":")
        return min(30, max(1, int(multiplier or 2) * report_count))
    return max(1, int(setting))


def select_report_coverage(
    ranked: list[tuple[float, Table, int]], candidates: list[Report], top_k: int,
) -> list[tuple[float, Table, int]]:
    """Reserve one relevant table per gated report before relevance-only fill.

    Round-robin interleaving by report was measured as an alternative, on the
    theory that one strong report starves the others: +0.0044 F2, CI [-0.0065,
    +0.0152], and it changed only 18 of 192 questions. Starvation is real but
    lives beyond the submitted budget, so reordering inside the budget cannot
    reach it. Rejected in favour of the simpler rule.
    """
    selected: list[tuple[float, Table, int]] = []
    selected_ids: set[str] = set()
    for report in candidates[:top_k]:
        item = next(
            (item for item in ranked if item[1].report_id == report.identity.report_id and item[1].table_id not in selected_ids),
            None,
        )
        if item is not None:
            selected.append(item)
            selected_ids.add(item[1].table_id)
    selected.extend(item for item in ranked if item[1].table_id not in selected_ids)
    return selected


def retrieve_rows(
    question: str,
    metadata: dict,
    reports: list[Report],
    top_k: int = 5,
    report_ids: list[str] | None = None,
    mode: str = "baseline",
    field_weights: dict[str, float] | None = None,
    hierarchy: bool = False,
    family_prior: bool = False,
) -> dict:
    """Rank the tables of the gated reports against one question.

    Two modes survive. `baseline` is the sparse ranking as it comes; the shipped
    `report-coverage` spends the budget across the reports the gate selected
    rather than letting one filing take every slot, which is what a question
    spanning several years or issuers needs.

    Every other mode this function once carried — metric and role and field
    views, rank fusion, evidence slots, a dense hybrid — was a rule of ours
    tuned against labels of ours, and both are quarantined. The dense retrieval
    that replaced the hybrid is a separate stage now: it runs off a corpus index
    and joins the candidate pool in export_rerank_pairs.py, where the reranker
    judges it rather than a weight of ours deciding in advance.
    """
    if mode not in {"baseline", "report-coverage"}:
        raise ValueError(f"Unknown retrieval mode: {mode}")
    started = time.perf_counter()
    candidates, stage = select_candidate_reports(reports, metadata, report_ids)
    tables, rows = materialize_candidate_rows(candidates, hierarchy=hierarchy)
    query = query_tokens(question, metadata)
    context_query = context_query_tokens(question, metadata)
    ranked = baseline_ranked_rows(
        query, context_query, rows,
        metric_query=metric_query_tokens(question, metadata),
        family_prior=family_prior,
    )
    baseline_rank = {table.table_id: rank
                     for rank, (_, table, _) in enumerate(ranked, 1)}
    if mode == "report-coverage":
        ranked = select_report_coverage(ranked, candidates, top_k)
    ranked = ranked[:top_k]
    return {
        "filter_stage": stage,
        "query_tokens": query,
        "context_query_tokens": context_query,
        "mode": mode,
        "hierarchy": hierarchy,
        "family_prior": family_prior,
        "candidate_report_count": len(candidates),
        "candidate_table_count": len(tables),
        "candidate_row_count": len(rows),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "tables": [
            {
                "table_id": table.table_id,
                "report_id": table.report_id,
                "page": table.page,
                "start_line": table.start_line,
                "score": round(score, 6),
                "row_index": row_index,
                "row_cells": list(table.rows[row_index]),
                "rows": [list(row) for row in table.rows],
                "header_cells": header_cells(table),
                "title": table.title,
                "periods": list(table.periods),
                "unit": table.unit,
                "pre_role_rank": baseline_rank.get(table.table_id),
            }
            for score, table, row_index in ranked
        ],
    }
