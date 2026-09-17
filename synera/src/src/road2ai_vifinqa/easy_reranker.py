"""Generator-faithful deterministic reranking for exhaustive Easy candidates.

The official Easy generator chose one exact source table before locking the
metric, period, time basis, unit and scope shown to the question writer.  This
module mirrors that information boundary: it ranks grounded numeric cells from
whole-table labels, the official neighbouring-page context, and deterministic
period/unit cues.  It never supplies or changes an answer and Qwen remains the
final semantic selector.

The checked-in weights are a full-data production refit.  Reported metrics in
the artifact are exclusively out-of-fold predictions from fold-specific fits.
"""

from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Protocol

from .paths import INDEX_PATH
from .retrieval import STOPWORDS, metric_phrase
from .text import clean_text, fold_text


MODEL_RESOURCE = "easy_reranker_v2.json"
_PAGE_RE = re.compile(r"===== PAGE\s+(\d+)\s+=====")
_TAG_RE = re.compile(r"<[^>]+>")


class EasyCandidateLike(Protocol):
    """Structural fields required by the reranker, avoiding solver cycles."""

    candidate_id: str
    ticker: str
    report_year: int
    doc_id: str
    table_id: int
    table_rows: int
    row_idx: int
    col_idx: int
    row_label: str
    section: str
    column_header: str
    table_context: str
    raw_value: str
    raw_number: float
    source_scale: float
    requested_scale: float
    retrieval_score: float


@dataclass(frozen=True, slots=True)
class _TableEvidence:
    page: int
    context: str
    rows: tuple[tuple[str, ...], ...]
    generator_pages: str


def _load_model() -> dict[str, object]:
    payload = json.loads(
        files(__package__).joinpath(MODEL_RESOURCE).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict) or payload.get("schema") != 2:
        raise RuntimeError(f"Invalid Easy reranker artifact: {MODEL_RESOURCE}")
    names = payload.get("feature_names")
    weights = payload.get("effective_raw_feature_weights")
    if not isinstance(names, list) or not isinstance(weights, list) or len(names) != len(weights):
        raise RuntimeError("Easy reranker feature/weight schema mismatch")
    if not names or any(not isinstance(name, str) for name in names):
        raise RuntimeError("Easy reranker feature names are invalid")
    numeric_weights = [float(value) for value in weights]
    if not all(math.isfinite(value) for value in numeric_weights):
        raise RuntimeError("Easy reranker contains a non-finite weight")
    payload["effective_raw_feature_weights"] = numeric_weights
    return payload


EASY_RERANKER_MODEL = _load_model()
EASY_RERANKER_NAME = str(EASY_RERANKER_MODEL["name"])
EASY_RERANKER_VALIDATION = EASY_RERANKER_MODEL["validation"]
_FEATURE_NAMES = tuple(str(value) for value in EASY_RERANKER_MODEL["feature_names"])
_WEIGHTS = tuple(float(value) for value in EASY_RERANKER_MODEL["effective_raw_feature_weights"])


def _tokens(text: str) -> list[str]:
    return [token for token in fold_text(text).split() if token and not token.isdigit()]


def _generator_phrase(question: str, tickers: list[str]) -> str:
    """Approximate the generator's locked ``metric_name`` field."""

    value = metric_phrase(question, tickers=tickers)
    value = re.sub(
        r"\b(?:so du|so tien|gia tri|khoan muc|tong gia tri|tong so du|tong so tien)\b",
        " ",
        value,
    )
    value = re.sub(
        r"\b(?:vao|tai|den|tinh den|cuoi|dau|trong|cua cong ty me)\b",
        " ",
        value,
    )
    replacements = {
        "tctd": "to chuc tin dung",
        "tndn": "thu nhap doanh nghiep",
        "tscd": "tai san co dinh",
        "bds": "bat dong san",
        "lai thuan": "loi nhuan thuan",
        "von co phan": "von gop cua chu so huu",
    }
    value = " ".join(value.split())
    for old, new in replacements.items():
        value = value.replace(old, new)
    return " ".join(value.split())


def _weighted_stats(
    query: set[str], document: str, idf: Mapping[str, float]
) -> tuple[float, float, float]:
    document_tokens = set(_tokens(document))
    overlap = query & document_tokens
    # Stable token order makes scores bitwise reproducible across Python hash
    # seeds, not merely deterministic inside one process.
    query_mass = sum(idf.get(token, 1.0) for token in sorted(query))
    document_mass = sum(idf.get(token, 1.0) for token in sorted(document_tokens))
    overlap_mass = sum(idf.get(token, 1.0) for token in sorted(overlap))
    recall = overlap_mass / max(query_mass, 1e-9)
    precision = overlap_mass / max(document_mass, 1e-9)
    cosine = overlap_mass / max(math.sqrt(query_mass * document_mass), 1e-9)
    return recall, precision, cosine


def _time_flags(question: str) -> dict[str, bool]:
    folded = fold_text(question)
    ending = any(
        marker in folded
        for marker in (
            "cuoi nam",
            "cuoi ky",
            "so cuoi nam",
            "den ngay 31",
            "vao ngay 31",
            "tinh den ngay 31",
        )
    )
    opening = any(
        marker in folded
        for marker in ("dau nam", "dau ky", "so dau nam", "ngay 01 01")
    )
    explicit_date = bool(
        re.search(r"\b(?:31|01)[ /-](?:12|01)[ /-]20\d{2}\b", folded)
    ) or "ngay 31 thang 12" in folded
    movement = any(
        marker in folded
        for marker in (
            "trich lap trong nam",
            "hoan nhap trong nam",
            "phat sinh trong nam",
            "trong nam",
        )
    )
    flow = movement or (
        any(
            marker in folded
            for marker in (
                "doanh thu",
                "chi phi",
                "loi nhuan",
                "lai thuan",
                "thu nhap",
                "luu chuyen tien",
                "dong tien",
            )
        )
        and not ending
        and not opening
    )
    point = ending or opening or explicit_date or folded.startswith("so du ")
    return {
        "ending": ending,
        "opening": opening,
        "date": explicit_date,
        "movement": movement,
        "flow": flow,
        "point": point,
    }


def _unit_phrase(question: str) -> str:
    folded = fold_text(question)
    for value in (
        "nghin ty dong",
        "tram ty dong",
        "ty dong",
        "trieu dong",
        "nghin dong",
        "trieu usd",
        "ty usd",
        "phan tram",
    ):
        if value in folded:
            return value
    if "%" in question or "ty le" in folded:
        return "phan tram"
    return "dong" if "dong" in folded else ""


def _bm25_scores(
    query: set[str], texts: Sequence[str], idf: Mapping[str, float]
) -> list[float]:
    lengths = [max(len(_tokens(text)), 1) for text in texts]
    average_length = sum(lengths) / max(len(lengths), 1)
    output: list[float] = []
    for text, length in zip(texts, lengths):
        counts = Counter(_tokens(text))
        score = 0.0
        for token in sorted(query):
            term_frequency = counts[token]
            if not term_frequency:
                continue
            denominator = term_frequency + 1.2 * (
                1.0 - 0.75 + 0.75 * length / max(average_length, 1e-9)
            )
            score += idf.get(token, 1.0) * term_frequency * 2.2 / denominator
        output.append(score)
    return output


def _read_pages(source_path: str) -> list[tuple[int, str]]:
    try:
        raw = Path(source_path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return []
    matches = list(_PAGE_RE.finditer(raw))
    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(raw)
        pages.append(
            (
                int(match.group(1)),
                clean_text(_TAG_RE.sub(" ", raw[match.end() : end])),
            )
        )
    return pages


@lru_cache(maxsize=16)
def _load_document_evidence(doc_id: str) -> dict[int, _TableEvidence]:
    """Load exact indexed table grids and official +/-1 page context."""

    if not INDEX_PATH.exists():
        return {}
    try:
        connection = sqlite3.connect(f"file:{INDEX_PATH.as_posix()}?mode=ro", uri=True)
        connection.row_factory = sqlite3.Row
        try:
            document = connection.execute(
                "SELECT source_path FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            rows = connection.execute(
                "SELECT table_id, page, context, rows_json FROM tables "
                "WHERE doc_id=? ORDER BY table_id",
                (doc_id,),
            ).fetchall()
        finally:
            connection.close()
    except sqlite3.Error:
        return {}
    pages = _read_pages(str(document["source_path"])) if document is not None else []
    page_position = {page_number: index for index, (page_number, _) in enumerate(pages)}
    output: dict[int, _TableEvidence] = {}
    for row in rows:
        page = int(row["page"])
        position = page_position.get(page)
        if position is None:
            page_text = str(row["context"])
        else:
            page_text = " ".join(
                text for _, text in pages[max(0, position - 1) : position + 2]
            )
        try:
            grid = tuple(
                tuple(str(cell) for cell in table_row)
                for table_row in json.loads(str(row["rows_json"]))
            )
        except (TypeError, ValueError):
            grid = ()
        output[int(row["table_id"])] = _TableEvidence(
            page=page,
            context=str(row["context"]),
            rows=grid,
            generator_pages=page_text,
        )
    return output


def _fallback_table_evidence(
    table_candidates: Sequence[EasyCandidateLike],
) -> _TableEvidence:
    """Conservative source-free fallback used by isolated unit tests."""

    by_row: dict[int, list[EasyCandidateLike]] = defaultdict(list)
    for candidate in table_candidates:
        by_row[candidate.row_idx].append(candidate)
    row_count = max(
        max((candidate.table_rows for candidate in table_candidates), default=0),
        max(by_row, default=-1) + 1,
    )
    rows: list[tuple[str, ...]] = [() for _ in range(row_count)]
    for row_idx in sorted(by_row):
        members = sorted(by_row[row_idx], key=lambda candidate: candidate.col_idx)
        values: list[str] = []
        for value in (
            members[0].section,
            members[0].row_label,
            *(candidate.column_header for candidate in members),
            *(candidate.raw_value for candidate in members),
        ):
            if value and value not in values:
                values.append(value)
        rows[row_idx] = tuple(values)
    first = table_candidates[0]
    return _TableEvidence(
        page=0,
        context=first.table_context,
        rows=tuple(rows),
        generator_pages=first.table_context,
    )


def generator_feature_vectors(
    question: str, candidates: Sequence[EasyCandidateLike]
) -> list[tuple[float, ...]]:
    """Return v2 features in candidate order.

    This public diagnostic hook makes runtime/training feature parity directly
    testable; production scoring simply takes its dot product with the artifact.
    """

    if not candidates:
        return []
    tickers = list(dict.fromkeys(candidate.ticker for candidate in candidates))
    phrase = _generator_phrase(question, tickers)
    query = set(_tokens(phrase)) - STOPWORDS
    if not query:
        query = set(_tokens(question)) - STOPWORDS
    flags = _time_flags(question)
    years = [int(value) for value in re.findall(r"\b20\d{2}\b", question)]
    target_year = years[-1] if years else 0
    unit = _unit_phrase(question)
    folded_question = fold_text(question)
    asks_percent = unit == "phan tram" or "%" in question
    asks_money = bool(unit and unit != "phan tram")
    asks_total = any(
        marker in folded_question for marker in ("tong ", "tong gia", "tong so", "cong ")
    )

    table_keys = list(dict.fromkeys((candidate.doc_id, candidate.table_id) for candidate in candidates))
    candidates_by_table: dict[tuple[str, int], list[EasyCandidateLike]] = defaultdict(list)
    for candidate in candidates:
        candidates_by_table[(candidate.doc_id, candidate.table_id)].append(candidate)
    evidence: dict[tuple[str, int], _TableEvidence] = {}
    for key in table_keys:
        document_tables = _load_document_evidence(key[0])
        evidence[key] = document_tables.get(key[1]) or _fallback_table_evidence(
            candidates_by_table[key]
        )

    table_labels: dict[tuple[str, int], str] = {}
    table_context: dict[tuple[str, int], str] = {}
    generator_pages: dict[tuple[str, int], str] = {}
    table_rows_text: dict[tuple[str, int], list[str]] = {}
    for key in table_keys:
        table = evidence[key]
        row_texts: list[str] = []
        labels: list[str] = []
        for row in table.rows:
            row_texts.append(" | ".join(cell for cell in row if cell.strip()))
            labels.extend(
                cell for cell in row if cell.strip() and not re.search(r"\d", cell)
            )
        table_rows_text[key] = row_texts
        table_labels[key] = " | ".join(labels)
        table_context[key] = table.context
        generator_pages[key] = table.generator_pages

    document_frequency: Counter[str] = Counter()
    for key in table_keys:
        present = set(_tokens(table_labels[key] + " " + table_context[key])) & query
        document_frequency.update(present)
    idf = {
        token: math.log((len(table_keys) + 1.0) / (document_frequency[token] + 0.5)) + 1.0
        for token in query
    }
    table_texts = [table_labels[key] + " " + table_context[key] for key in table_keys]
    bm25_values = _bm25_scores(query, table_texts, idf)
    table_bm25 = dict(zip(table_keys, bm25_values))
    bm25_max = max(bm25_values, default=0.0)
    bm25_order = sorted(table_keys, key=lambda key: (-table_bm25[key], key))
    bm25_rank = {key: index + 1 for index, key in enumerate(bm25_order)}
    page_bm25_values = _bm25_scores(
        query, [generator_pages[key] for key in table_keys], idf
    )
    page_bm25 = dict(zip(table_keys, page_bm25_values))
    page_bm25_max = max(page_bm25_values, default=0.0)
    page_bm25_order = sorted(table_keys, key=lambda key: (-page_bm25[key], key))
    page_bm25_rank = {key: index + 1 for index, key in enumerate(page_bm25_order)}

    table_max_current: dict[tuple[str, int], float] = defaultdict(lambda: -math.inf)
    for candidate in candidates:
        key = (candidate.doc_id, candidate.table_id)
        table_max_current[key] = max(table_max_current[key], candidate.retrieval_score)
    current_max = max(table_max_current.values(), default=0.0)
    current_order = sorted(table_keys, key=lambda key: (-table_max_current[key], key))
    current_rank = {key: index + 1 for index, key in enumerate(current_order)}

    table_static: dict[tuple[str, int], dict[str, object]] = {}
    unit_prefix = {key: " ".join(table_rows_text[key][:4]) for key in table_keys}
    for key in table_keys:
        labels = table_labels[key]
        context = table_context[key]
        whole = labels + " " + context
        table_recall, table_precision, table_cosine = _weighted_stats(query, whole, idf)
        context_recall, _, _ = _weighted_stats(query, context, idf)
        labels_recall, _, _ = _weighted_stats(query, labels, idf)
        pages = generator_pages[key]
        pages_recall, pages_precision, pages_cosine = _weighted_stats(query, pages, idf)
        row_stats = [
            _weighted_stats(query, row_text, idf)[0]
            for row_text in table_rows_text[key]
        ]
        row_sequences = [
            SequenceMatcher(None, phrase, fold_text(row_text)).ratio()
            for row_text in table_rows_text[key]
        ]
        folded_context = fold_text(context)
        table_static[key] = {
            "whole_fold": fold_text(whole),
            "context_fold": folded_context,
            "pages_fold": fold_text(pages),
            "table_recall": table_recall,
            "table_precision": table_precision,
            "table_cosine": table_cosine,
            "context_recall": context_recall,
            "labels_recall": labels_recall,
            "pages_recall": pages_recall,
            "pages_precision": pages_precision,
            "pages_cosine": pages_cosine,
            "best_row_recall": max(row_stats, default=0.0),
            "best_row_sequence": max(row_sequences, default=0.0),
            "point_table": any(
                marker in folded_context
                for marker in (
                    "bang can doi",
                    "bao cao tinh hinh tai chinh",
                    "tai ngay",
                    "vao ngay",
                )
            ),
            "flow_table": any(
                marker in folded_context
                for marker in (
                    "ket qua hoat dong",
                    "cho nam tai chinh",
                    "cho nam ket thuc",
                    "luu chuyen tien",
                )
            ),
            "movement_table": any(
                marker in folded_context
                for marker in ("bien dong", "trong nam nhu sau", "tang giam", "thay doi")
            ),
            "explicit_unit": bool(
                unit and unit in fold_text(context + " " + unit_prefix[key])
            ),
        }

    vectors: list[tuple[float, ...]] = []
    for candidate in candidates:
        key = (candidate.doc_id, candidate.table_id)
        static = table_static[key]
        row = candidate.row_label
        section = candidate.section
        header = candidate.column_header
        row_recall, row_precision, _ = _weighted_stats(query, row, idf)
        header_recall, _, _ = _weighted_stats(query, header, idf)
        section_recall, _, _ = _weighted_stats(query, section, idf)
        row_fold = fold_text(row)
        header_fold = fold_text(header)
        rare_tokens = {token for token in query if idf.get(token, 1.0) >= 1.8}
        scale_distance = abs(
            math.log10(
                max(candidate.source_scale, 1e-12) / max(candidate.requested_scale, 1e-12)
            )
        )
        raw_percent = "%" in candidate.raw_value or "ty le" in fold_text(row + " " + header)
        header_years = {int(value) for value in re.findall(r"\b20\d{2}\b", header_fold)}
        ending_header = any(
            marker in header_fold
            for marker in ("cuoi nam", "cuoi ky", "31 12", "nam nay", "ky nay")
        )
        opening_header = any(
            marker in header_fold
            for marker in ("dau nam", "dau ky", "01 01", "nam truoc", "ky truoc")
        )
        period_header = bool(target_year and target_year in header_years) or any(
            marker in header_fold for marker in ("nam nay", "ky nay")
        )
        date_header = bool(
            re.search(r"\b(?:31|01)[ ./-](?:12|01)[ ./-]20\d{2}\b", header_fold)
        )
        total_row = any(
            marker in row_fold for marker in ("tong cong", "cong", "tong gia", "tong so", "total")
        ) or row_fold.strip() == "tong"
        component_row = row_fold.startswith("-") or bool(
            re.match(r"^(?:[a-z]\)|\d+[.)])", row_fold)
        )
        raw_digits = re.sub(r"\D", "", candidate.raw_value)
        whole_fold = str(static["whole_fold"])
        context_fold = str(static["context_fold"])
        vector = (
            float(static["table_recall"]),
            float(static["table_precision"]),
            float(static["table_cosine"]),
            table_bm25[key],
            table_bm25[key] - bm25_max,
            -math.log1p(bm25_rank[key]),
            float(static["context_recall"]),
            float(static["labels_recall"]),
            float(static["pages_recall"]),
            float(static["pages_precision"]),
            float(static["pages_cosine"]),
            page_bm25[key],
            page_bm25[key] - page_bm25_max,
            -math.log1p(page_bm25_rank[key]),
            float(bool(phrase) and phrase in str(static["pages_fold"])),
            float(static["best_row_recall"]),
            float(static["best_row_sequence"]),
            float(bool(phrase) and phrase in whole_fold),
            float(bool(phrase) and phrase in context_fold),
            table_max_current[key],
            table_max_current[key] - current_max,
            -math.log1p(current_rank[key]),
            candidate.retrieval_score - table_max_current[key],
            row_recall,
            row_precision,
            SequenceMatcher(None, phrase, row_fold).ratio(),
            float(bool(phrase) and phrase in row_fold),
            header_recall,
            section_recall,
            float(len(rare_tokens & set(_tokens(row)))),
            float(len(rare_tokens & set(_tokens(whole_fold)))),
            1.0 / (1.0 + abs(len(_tokens(row)) - max(len(query), 1))),
            float(bool(set(_tokens(row)) - STOPWORDS)),
            float(scale_distance < 1e-8),
            -scale_distance,
            float(bool(static["explicit_unit"])),
            float(asks_percent and raw_percent),
            float(asks_percent and not raw_percent),
            float(asks_money and raw_percent),
            float(flags["ending"] and ending_header),
            float(flags["opening"] and opening_header),
            float(flags["flow"] and period_header),
            float(bool(target_year and target_year in header_years)),
            float(bool(target_year and target_year - 1 in header_years)),
            float(
                bool(
                    header_years
                    and target_year
                    and target_year not in header_years
                    and target_year - 1 not in header_years
                )
            ),
            float(flags["point"] and bool(static["point_table"])),
            float(flags["point"] and bool(static["flow_table"])),
            float(flags["flow"] and bool(static["flow_table"])),
            float(flags["flow"] and bool(static["point_table"])),
            float(flags["movement"] and bool(static["movement_table"])),
            float(flags["date"] and date_header),
            float(asks_total and total_row),
            float(asks_total and component_row and not total_row),
            float(not flags["opening"] and candidate.col_idx > 0),
            float(flags["opening"] and candidate.col_idx > 0),
            float(not raw_digits or len(raw_digits) > 4 or not raw_digits.startswith(("19", "20"))),
            float(abs(candidate.raw_number) > 1e-15),
            1.0 / math.sqrt(max(len(evidence[key].rows), 1)),
        )
        if len(vector) != len(_FEATURE_NAMES):
            raise RuntimeError("Easy reranker runtime feature schema mismatch")
        vectors.append(vector)
    return vectors


def score_easy_candidates(
    question: str,
    candidates: Sequence[EasyCandidateLike],
    bm25_row_scores: Mapping[tuple[str, int, int], float],
) -> dict[str, float]:
    """Return deterministic v2 scores keyed by stable candidate ID.

    ``bm25_row_scores`` remains in the public shortlist API for compatibility
    and audit logging.  V2 intentionally replaces (rather than stacks) the
    older row-BM25 model, so its 58 validated generator features do not consume
    those scores.
    """

    del bm25_row_scores
    if not candidates:
        return {}
    candidate_ids = [candidate.candidate_id for candidate in candidates]
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("Easy reranker requires unique candidate IDs")
    vectors = generator_feature_vectors(question, candidates)
    scores: dict[str, float] = {}
    for candidate, vector in zip(candidates, vectors):
        score = sum(feature * weight for feature, weight in zip(vector, _WEIGHTS))
        if not math.isfinite(score):
            raise ValueError(
                f"Easy reranker produced a non-finite score for {candidate.candidate_id}"
            )
        scores[candidate.candidate_id] = float(score)
    return scores


__all__ = [
    "EASY_RERANKER_MODEL",
    "EASY_RERANKER_NAME",
    "EASY_RERANKER_VALIDATION",
    "MODEL_RESOURCE",
    "generator_feature_vectors",
    "score_easy_candidates",
]
