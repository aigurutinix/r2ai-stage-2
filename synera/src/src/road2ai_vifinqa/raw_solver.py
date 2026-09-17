"""Grounded fallback solver for arbitrary note-table questions.

The canonical financial panel covers common statement line items.  This module
handles the long tail (bank disclosures, segment notes, sensitivity tables,
etc.) without giving the model direct filesystem access.  It retrieves a small
set of numeric cells, normalises their units, and asks the local open model to
compile one execution-checked pandas expression over that finite candidate set.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

from .corpus import Corpus
from .direct import _column_year_score, _source_scale_for_hit
from .local_llm import chat, extract_json
from .panel_solver import execute_panel_query
from .retrieval import RowHit, retrieve_rows
from .text import fold_text, parse_vn_number


@dataclass(frozen=True, slots=True)
class NumericCandidate:
    candidate_id: str
    ticker: str
    report_year: int
    scope: str
    doc_id: str
    table_id: int
    row_idx: int
    col_idx: int
    row_label: str
    column_header: str
    table_context: str
    raw_value: str
    raw_number: float
    source_scale: float
    vnd_value: float
    retrieval_phrase: str
    retrieval_score: float

    @property
    def table_ref(self) -> str:
        return f"{self.doc_id}|table_{self.table_id}"


@dataclass(frozen=True, slots=True)
class RawSolution:
    answer: float | int
    pandas_query: str
    selected: tuple[NumericCandidate, ...]
    candidates: tuple[NumericCandidate, ...]
    lookup_phrases: tuple[str, ...]
    attempts: int
    note: str
    elapsed_seconds: float


PHRASE_SYSTEM = """You extract retrieval keys from a Vietnamese financial question.
Return one JSON object only:
{"lookup_phrases":["exact financial line item 1","exact financial line item 2"],"operation":"short description"}
Each phrase must be a short Vietnamese accounting row label, not a company, year, unit, filter, max/min,
or question wording. Include every distinct financial measure needed both for filtering/selection and for the final answer.
For "tỷ lệ giữa A và B", emit A and B as TWO separate phrases; never emit a phrase beginning with "tỷ lệ giữa".
For a clause such as "tại công ty/ngân hàng có X cao nhất/thấp nhất", X is a selector metric and MUST be an additional phrase.
Use at most 4 phrases. Preserve important qualifiers such as short-term/long-term, parent company, VND, principal,
interest, related party, beginning/end balance, or a named segment/currency. No markdown."""


CELL_SYSTEM = """You are a deterministic compiler for Vietnamese financial questions.
You receive a DataFrame `df` containing ONLY grounded numeric-cell candidates. Compile ONE pandas expression.
Return exactly one JSON object, no markdown:
{"pandas_query":"<single expression>","selected_ids":["c0001"],"note":"brief audit"}

Columns: candidate_id, ticker, report_year, scope, doc_id, table_id, row_idx, col_idx, row_label,
column_header, table_context, raw_value, raw_number, source_scale, vnd_value, retrieval_phrase.

Rules:
- Select candidates by exact candidate_id after checking company, scope, period, row label and column header.
- Monetary arithmetic uses `vnd_value`; percentages, ratios, counts and per-share figures normally use `raw_number`.
- Convert the FINAL monetary result to the unit asked: /1e3 thousand VND, /1e6 million VND,
  /1e9 billion VND, /1e11 hundred-billion VND, /1e12 trillion VND. Do not convert intermediate values.
- A growth rate is (new/old-1)*100. A percentage share/ratio is numerator/denominator*100.
- An unspecified difference/chênh lệch is abs(a-b). Explicit "A trừ B" preserves A-B.
- "Mức giảm", "mức bất lợi", "giảm bao nhiêu" asks for a positive magnitude: use `abs(...)`.
- If a sensitivity table's context already states the scenario (for example VND +/-5%) and its rows are the
  resulting profit impacts, use those row values directly. Never multiply an already-computed sensitivity result by 5% again.
- Apply every filter first, then max/min/mean/sum, then look up the requested target at the same company/year.
- Do not confuse selector metric with final target metric. Use full precision and no intermediate rounding.
- Return one numeric scalar. Do not return an index, Series, array, string, year label as text, or candidate value constant.
- The expression must visibly reference at least one candidate_id. No assignments, imports, eval, exec or file access.
- `selected_ids` lists every cell whose value affects the result, including selector/filter cells.
"""


def _short_label(hit: RowHit) -> str:
    labels = [cell.strip() for cell in hit.row.cells if cell.strip() and parse_vn_number(cell) is None]
    return " | ".join(labels[:2]) if labels else ""


def _header_for(hit: RowHit, col_idx: int) -> str:
    values: list[str] = []
    # Only structural header rows.  Including earlier data rows would turn
    # unrelated numeric values into a fake column label.
    for row_idx, row in enumerate(hit.table.rows[: min(3, hit.row.row_idx)]):
        if col_idx >= len(row):
            continue
        value = row[col_idx].strip()
        if value and value not in values:
            values.append(value)
    return " | ".join(values[-3:])

    


def plan_lookup_phrases(question: str) -> tuple[list[str], str]:
    completion = chat(system=PHRASE_SYSTEM, user=question, max_tokens=384, temperature=0.0)
    payload = extract_json(completion.content)
    phrases = [str(value).strip() for value in payload.get("lookup_phrases", []) if str(value).strip()]
    phrases = list(dict.fromkeys(phrases))[:4]
    if not phrases:
        raise ValueError("Phrase planner returned no lookup phrase")
    return phrases, str(payload.get("operation", ""))


def _scope_text(scope: str | None) -> str:
    if scope == "parent":
        return "công ty mẹ"
    if scope == "consolidated":
        return "hợp nhất"
    return ""


def build_numeric_candidates(
    corpus: Corpus,
    question: str,
    phrases: list[str],
    *,
    rows_per_document: int = 4,
    max_candidates: int = 60,
) -> list[NumericCandidate]:
    tickers = corpus.infer_tickers(question)
    years = corpus.infer_years(question)
    scope = corpus.infer_scope(question)
    if not tickers or not years:
        raise ValueError(f"Cannot resolve entity/year: tickers={tickers}, years={years}")
    # "versus previous year" needs the comparative column, which is retained
    # below; an explicit previous report is also useful when OCR headers are weak.
    include_prior_report = any(
        marker in fold_text(question)
        for marker in ("nam truoc", "ky truoc", "so voi cuoi nam truoc", "so voi nam lien truoc")
    )
    target_years = list(years)
    if include_prior_report:
        target_years.extend(year - 1 for year in years)
    raw_hits: list[tuple[str, str, int, RowHit]] = []
    table_cache: dict[tuple[str, int], object] = {}
    for phrase in phrases:
        for ticker in tickers:
            for year in dict.fromkeys(target_years):
                query = f"{phrase} của {ticker} năm {year} {_scope_text(scope)}"
                hits = retrieve_rows(corpus, query, limit=rows_per_document)
                raw_hits.extend((phrase, ticker, year, hit) for hit in hits)
                # Some disclosures (notably sensitivity/currency and segment
                # tables) put the financial concept only in the paragraph
                # immediately before the table; the body rows contain merely
                # currencies or segment names.  Add context-matched rows so
                # lexical row retrieval cannot make those tables invisible.
                documents = corpus.documents_for_question(query)
                phrase_folded = fold_text(phrase)
                phrase_tokens = set(phrase_folded.split())
                doc_by_id = {document.doc_id: document for document in documents}
                contextual: list[RowHit] = []
                for row in corpus.rows_for_documents(documents):
                    table_key = (row.doc_id, row.table_id)
                    table = table_cache.get(table_key)
                    if table is None:
                        table = corpus.table(*table_key)
                        table_cache[table_key] = table
                    context_folded = fold_text(table.context)
                    context_tokens = set(context_folded.split())
                    context_recall = len(phrase_tokens & context_tokens) / max(len(phrase_tokens), 1)
                    if context_recall < 0.34:
                        continue
                    row_folded = fold_text(" ".join(row.cells))
                    row_tokens = set(row_folded.split())
                    row_overlap = len(phrase_tokens & row_tokens) / max(len(phrase_tokens), 1)
                    sequence = SequenceMatcher(None, phrase_folded, row_folded).ratio()
                    score = 6.0 * context_recall + 3.0 * row_overlap + 3.0 * sequence
                    contextual.append(RowHit(score, row, table, doc_by_id[row.doc_id]))
                contextual.sort(key=lambda hit: -hit.score)
                raw_hits.extend(
                    (phrase, ticker, year, hit) for hit in contextual[:rows_per_document]
                )

    # Deduplicate identical retrieved rows while retaining the best score and
    # the phrase that surfaced them.
    best: dict[tuple[str, int, int], tuple[str, str, int, RowHit]] = {}
    for item in raw_hits:
        phrase, ticker, year, hit = item
        key = (hit.row.doc_id, hit.row.table_id, hit.row.row_idx)
        if key not in best or hit.score > best[key][3].score:
            best[key] = item

    pending: list[dict[str, object]] = []
    for phrase, ticker, requested_year, hit in sorted(
        best.values(), key=lambda item: (-item[3].score, item[1], item[2])
    ):
        numeric: list[tuple[float, int, str, float]] = []
        for col_idx, raw in enumerate(hit.row.cells):
            number = parse_vn_number(raw)
            if number is None:
                continue
            compact = raw.strip().replace(".", "").replace(",", "")
            if compact.lstrip("+-").isdigit():
                integer = abs(int(compact))
                if integer <= 999 or 1900 <= integer <= 2100:
                    continue
            score = _column_year_score(hit, col_idx, requested_year)
            numeric.append((score, col_idx, raw, float(number)))
        # Keep the likely current and comparative values, not every code/footnote.
        numeric.sort(key=lambda item: (-item[0], item[1]))
        scale = _source_scale_for_hit(hit)
        for column_score, col_idx, raw, number in numeric[:3]:
            pending.append(
                {
                    "ticker": ticker,
                    "report_year": hit.document.report_year,
                    "scope": hit.document.scope,
                    "doc_id": hit.row.doc_id,
                    "table_id": hit.row.table_id,
                    "row_idx": hit.row.row_idx,
                    "col_idx": col_idx,
                    "row_label": _short_label(hit),
                    "column_header": _header_for(hit, col_idx),
                    "table_context": hit.table.context[:140],
                    "raw_value": raw,
                    "raw_number": number,
                    "source_scale": scale,
                    "vnd_value": number * scale,
                    "retrieval_phrase": phrase,
                    "retrieval_score": float(
                        hit.score
                        + min(column_score, 4.0)
                        + 10.0
                        * len(
                            set(fold_text(question).split())
                            & set(fold_text(f"{hit.table.context} {' '.join(hit.row.cells)}").split())
                        )
                        / max(len(set(fold_text(question).split())), 1)
                    ),
                }
            )
    # Preserve entity/year/phrase diversity instead of allowing one very long
    # table to consume the global prompt budget.
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for item in pending:
        key = (str(item["retrieval_phrase"]), str(item["ticker"]), int(item["report_year"]))
        grouped.setdefault(key, []).append(item)
    for values in grouped.values():
        values.sort(key=lambda item: -float(item["retrieval_score"]))
        del values[10:]
    pending = []
    depth = 0
    while len(pending) < max_candidates:
        added = False
        for key in sorted(grouped):
            values = grouped[key]
            if depth < len(values):
                pending.append(values[depth])
                added = True
                if len(pending) >= max_candidates:
                    break
        if not added:
            break
        depth += 1
    result: list[NumericCandidate] = []
    for index, item in enumerate(pending[:max_candidates], 1):
        result.append(NumericCandidate(candidate_id=f"c{index:04d}", **item))
    return result


def candidate_frame(candidates: list[NumericCandidate]) -> pd.DataFrame:
    return pd.DataFrame.from_records(asdict(candidate) for candidate in candidates)


def _candidate_preview(candidates: list[NumericCandidate]) -> str:
    fields = (
        "candidate_id",
        "ticker",
        "report_year",
        "scope",
        "row_label",
        "column_header",
        "raw_value",
        "raw_number",
        "source_scale",
        "vnd_value",
        "retrieval_phrase",
        "table_context",
    )
    compact = [{field: getattr(candidate, field) for field in fields} for candidate in candidates]
    return json.dumps(compact, ensure_ascii=False, separators=(",", ":"))


def solve_raw_question(
    question: str,
    corpus: Corpus,
    *,
    max_attempts: int = 3,
    log_path: Path | None = None,
) -> RawSolution:
    started = time.time()
    phrases, operation = plan_lookup_phrases(question)
    candidates = build_numeric_candidates(corpus, question, phrases)
    if not candidates:
        raise ValueError("Retrieval produced no numeric candidates")
    frame = candidate_frame(candidates)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    prompt = (
        f"Vietnamese question: {question}\nPlanner operation hint: {operation}\n"
        f"Grounded candidates JSON:\n{_candidate_preview(candidates)}"
    )
    logs: list[dict[str, object]] = [
        {"lookup_phrases": phrases, "operation": operation, "candidate_count": len(candidates)}
    ]
    prior = ""
    error = ""
    for attempt in range(1, max_attempts + 1):
        user = prompt
        if attempt > 1:
            user += (
                f"\nPrevious expression: {prior}\nValidation error: {error}\n"
                "Repair the expression against the same candidates and return the JSON contract only."
            )
        completion = chat(system=CELL_SYSTEM, user=user, max_tokens=1024, temperature=0.0)
        entry: dict[str, object] = {"attempt": attempt, "response": completion.content, **asdict(completion)}
        try:
            payload = extract_json(completion.content)
            expression = str(payload["pandas_query"]).strip()
            # Small models occasionally omit only the final closing parenthesis
            # after otherwise valid, grounded code.  Repair that mechanical
            # truncation before invoking the strict AST validator.
            if expression.count("(") > expression.count(")"):
                expression += ")" * (expression.count("(") - expression.count(")"))
            prior = expression
            answer = execute_panel_query(expression, frame)
            folded_question = fold_text(question)
            asks_positive_magnitude = any(
                marker in folded_question
                for marker in ("muc giam", "muc bat loi", "giam bao nhieu", "chenh lech")
            ) and not any(marker in folded_question for marker in ("a tru b", "trai tru phai"))
            if asks_positive_magnitude and float(answer) < 0:
                expression = f"abs({expression})"
                answer = execute_panel_query(expression, frame)
            literal_ids = set(re.findall(r"c\d{4}", expression))
            if not literal_ids:
                raise ValueError("pandas_query must visibly filter exact candidate_id literals")
            mentioned = set(literal_ids)
            mentioned.update(str(value) for value in payload.get("selected_ids", []))
            selected = tuple(by_id[value] for value in sorted(mentioned) if value in by_id)
            if not selected:
                raise ValueError("Expression does not reference a grounded candidate_id")
            if not math.isfinite(float(answer)):
                raise ValueError("Non-finite result")
            entry["answer"] = answer
            entry["selected_ids"] = [value.candidate_id for value in selected]
            logs.append(entry)
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return RawSolution(
                answer=answer,
                pandas_query=expression,
                selected=selected,
                candidates=tuple(candidates),
                lookup_phrases=tuple(phrases),
                attempts=attempt,
                note=str(payload.get("note", "")),
                elapsed_seconds=time.time() - started,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            entry["error"] = error
            logs.append(entry)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError(f"Raw compilation failed after {max_attempts} attempts: {error}")
