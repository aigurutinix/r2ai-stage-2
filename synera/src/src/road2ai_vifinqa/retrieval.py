"""Entity-constrained lexical row and table retrieval."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher

from .corpus import Corpus, DocumentRef, RowAsset, TableAsset
from .table_semantics import (
    StatementKind,
    asks_total,
    question_statement_preference,
    row_label,
    statement_kind,
)
from .text import fold_text


STOPWORDS = frozenset(
    "cua cong ty me co phan tap doan ngan hang tmcp tong trong vao nam cuoi den ngay "
    "la bao nhieu trieu ty nghin tram bao cao cac va cho mot".split()
)
YEAR_RE = re.compile(r"\b20\d{2}\b")


@dataclass(frozen=True, slots=True)
class RowHit:
    score: float
    row: RowAsset
    table: TableAsset
    document: DocumentRef


@dataclass(frozen=True, slots=True)
class TableHit:
    score: float
    table: TableAsset
    document: DocumentRef
    best_row_score: float


def _lexical_row_score(
    row: RowAsset,
    *,
    phrase: str,
    qtokens: set[str],
    weights: dict[str, float],
) -> float | None:
    rtokens = set(row.folded_text.split())
    overlap = qtokens & rtokens
    if not overlap:
        return None
    weighted_recall = sum(weights[token] for token in overlap) / max(sum(weights.values()), 1e-9)
    precision = len(overlap) / max(len(rtokens & (qtokens | STOPWORDS)), 1)
    sequence = SequenceMatcher(None, phrase, row.folded_text).ratio()
    exact = 1.0 if phrase and phrase in row.folded_text else 0.0
    short_bonus = 1.0 / (1.0 + max(0, len(rtokens) - len(qtokens)) / 8.0)
    return 7.0 * weighted_recall + 1.5 * precision + 2.5 * sequence + 5.0 * exact + short_bonus


def _table_semantic_prior(table: TableAsset, question: str) -> float:
    try:
        kind = (
            statement_kind(table.context, table.rows)
            if table.statement_kind == "unknown"
            else StatementKind(table.statement_kind)
        )
    except ValueError:
        kind = statement_kind(table.context, table.rows)
    preferences = question_statement_preference(question)
    prior = 0.0
    if preferences:
        if kind == preferences[0]:
            prior += 2.8
        elif kind in preferences[1:]:
            prior += 0.8
        elif kind.value != "unknown":
            prior -= 1.2
    folded_q = fold_text(question)
    folded_context = fold_text(table.context)
    if any(marker in folded_q for marker in ("cuoi nam", "dau nam", "tai ngay")):
        if kind.value == "balance_sheet":
            prior += 2.0
        elif kind.value in {"income_statement", "cash_flow"}:
            prior -= 1.0
    if "cong ty me" in folded_q:
        prior += 0.7 if "cong ty me" in folded_context else 0.0
    if "hop nhat" in folded_q:
        prior += 0.7 if "hop nhat" in folded_context else 0.0
    return prior


def retrieve_tables(
    corpus: Corpus,
    question: str,
    *,
    limit: int = 20,
    include_prior: bool = False,
) -> list[TableHit]:
    """Retrieve tables before rows using entity and structural constraints."""

    documents = corpus.documents_for_question(question, include_prior=include_prior)
    rows = corpus.rows_for_documents(documents)
    phrase = metric_phrase(question, tickers=corpus.infer_tickers(question))
    qtokens = set(phrase.split()) or (set(fold_text(question).split()) - STOPWORDS)
    weights = _idf_weights(rows, qtokens)
    doc_by_id = {doc.doc_id: doc for doc in documents}
    grouped: dict[tuple[str, int], list[float]] = {}
    for row in rows:
        value = _lexical_row_score(row, phrase=phrase, qtokens=qtokens, weights=weights)
        if value is not None:
            grouped.setdefault((row.doc_id, row.table_id), []).append(value)

    hits: list[TableHit] = []
    for key, row_scores in grouped.items():
        table = corpus.table(*key)
        top_rows = sorted(row_scores, reverse=True)[:3]
        lexical = top_rows[0] + 0.20 * sum(top_rows[1:])
        context_tokens = set(fold_text(table.context).split())
        context = 2.0 * len(context_tokens & qtokens) / max(len(qtokens), 1)
        score = lexical + context + _table_semantic_prior(table, question)
        hits.append(TableHit(score, table, doc_by_id[key[0]], top_rows[0]))
    hits.sort(key=lambda item: (-item.score, item.document.report_year, item.table.table_id))
    return hits[:limit]


def metric_phrase(question: str, *, tickers: list[str] | None = None) -> str:
    folded = fold_text(question)
    folded = YEAR_RE.sub(" ", folded)
    entity_patterns = (
        r"\s+cua\s+cong\s+ty\s+me\b",
        r"\s+cua\s+(?:ctcp|ngan hang|tong cong ty|tap doan|cong ty co phan)\b",
        r"\s+tai\s+(?:cong ty me|ctcp|ngan hang|tong cong ty|tap doan)\b",
    )
    cut_positions = [m.start() for pattern in entity_patterns if (m := re.search(pattern, folded))]
    for ticker in tickers or []:
        match = re.search(rf"\s+cua\s+(?:cong\s+ty\s+me\s+)?{re.escape(ticker.casefold())}\b", folded)
        if match:
            cut_positions.append(match.start())
    if cut_positions:
        folded = folded[: min(cut_positions)]
    for separator in (" la bao nhieu", " bang bao nhieu"):
        if separator in folded:
            folded = folded.split(separator, 1)[0]
    folded = re.sub(r"\b(?:cuoi|dau|trong) nam\b", " ", folded)
    folded = re.sub(r"\b(?:bao nhieu )?(?:trieu|ty|nghin ty|tram ty) dong\b", " ", folded)
    return " ".join(token for token in folded.split() if token not in {"nam"} and not token.isdigit())


def _idf_weights(rows: list[RowAsset], query_tokens: set[str]) -> dict[str, float]:
    if not rows:
        return {token: 1.0 for token in query_tokens}
    counts: Counter[str] = Counter()
    for row in rows:
        present = set(row.folded_text.split()) & query_tokens
        counts.update(present)
    return {
        token: math.log((len(rows) + 1) / (counts[token] + 1)) + 1.0 for token in query_tokens
    }


def retrieve_rows(
    corpus: Corpus,
    question: str,
    *,
    limit: int = 20,
    include_prior: bool = False,
    semantic: bool = False,
    candidate_tables: frozenset[tuple[str, int]] | None = None,
) -> list[RowHit]:
    """Retrieve rows, retaining the calibrated legacy scorer by default.

    ``semantic=True`` is deliberately opt-in. Offline evaluation shows that
    statement and total priors help table retrieval but can overrule exact row
    matches, so callers must explicitly request that experimental behavior.
    """
    documents = corpus.documents_for_question(question, include_prior=include_prior)
    rows = corpus.rows_for_documents(documents)
    if candidate_tables is not None:
        rows = [row for row in rows if (row.doc_id, row.table_id) in candidate_tables]
    years = set(corpus.infer_years(question))
    folded_question = fold_text(question)
    phrase = metric_phrase(question, tickers=corpus.infer_tickers(question))
    qtokens = set(phrase.split())
    if not qtokens:
        qtokens = set(fold_text(question).split()) - STOPWORDS
    weights = _idf_weights(rows, qtokens)
    doc_by_id = {doc.doc_id: doc for doc in documents}
    table_cache: dict[tuple[str, int], TableAsset] = {}
    table_prior: dict[tuple[str, int], float] = {}
    scored: list[RowHit] = []

    for row in rows:
        score = _lexical_row_score(row, phrase=phrase, qtokens=qtokens, weights=weights)
        if score is None:
            continue
        key = (row.doc_id, row.table_id)
        if key not in table_cache:
            table_cache[key] = corpus.table(*key)
        table = table_cache[key]
        if semantic:
            if key not in table_prior:
                table_prior[key] = _table_semantic_prior(table, question)
            score += table_prior[key]
        context_tokens = set(fold_text(table.context).split())
        folded_context = fold_text(table.context)
        score += 2.2 * len(context_tokens & qtokens) / max(len(qtokens), 1)
        if any(marker in folded_question for marker in ("cuoi nam", "den ngay", "vao ngay")):
            if any(marker in folded_context for marker in ("bang can doi", "bao cao tinh hinh tai chinh")):
                score += 2.4
        if "nganh " in phrase:
            subtype = phrase.split("nganh ", 1)[1].strip()
            if subtype and subtype in row.folded_text:
                score += 6.0
        if hit_year := doc_by_id[row.doc_id].report_year:
            score += 1.5 if hit_year in years else (-0.5 if years else 0.0)
        if semantic:
            label = fold_text(row_label(row.cells))
            if asks_total(question):
                if any(
                    marker in label
                    for marker in ("tong cong", "tong so", "tong tai san", "tong nguon von")
                ):
                    score += 2.0
            elif label.startswith("tong cong"):
                score -= 0.4
        scored.append(RowHit(score, row, table, doc_by_id[row.doc_id]))

    scored.sort(key=lambda hit: (-hit.score, hit.document.report_year, hit.row.table_id, hit.row.row_idx))
    return scored[:limit]


def retrieve_rows_cascade(
    corpus: Corpus,
    question: str,
    *,
    table_limit: int = 20,
    row_limit: int = 20,
    include_prior: bool = False,
) -> list[RowHit]:
    """Run table-first retrieval, then the calibrated row scorer.

    Semantic priors determine only the table shortlist. Row ordering retains
    the historical scorer which performs better in source-audited evaluation.
    """

    table_hits = retrieve_tables(
        corpus,
        question,
        limit=table_limit,
        include_prior=include_prior,
    )
    keys = frozenset((hit.table.doc_id, hit.table.table_id) for hit in table_hits)
    if not keys:
        return []
    return retrieve_rows(
        corpus,
        question,
        limit=row_limit,
        include_prior=include_prior,
        candidate_tables=keys,
    )
