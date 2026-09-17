"""Lightweight, source-grounded reranking for tables inside a known report.

The legacy catalog assumes that the first CSV column contains row labels.  OCR
tables frequently put a numeric ``Mã số`` column first and the actual financial
labels second.  BM25 then sees only numbers and cannot rank the correct primary
statement fragment.  This module inspects the already extracted CSV, chooses
the most text-like label column, and adds a conservative label-match score.

No question IDs, expected answers or leaderboard feedback are used here.
"""

from __future__ import annotations

import math
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd


_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"^[\s()+\-.,%0-9/]+$")
_PREFIX_RE = re.compile(r"^(?:[ivxlcdm]+|\d+(?:\.\d+)*)[.)\-:]?\s+", re.IGNORECASE)
_STOPWORDS = {
    "a", "ai", "bao", "bang", "bi", "cac", "cho", "co", "cong", "cua", "cuoi",
    "den", "do", "doanh", "dong", "duoc", "giai", "giua", "hoi", "la", "ma",
    "me", "mot", "nam", "nao", "ngan", "nhieu", "nhom", "nhu", "nhung", "phan",
    "qua", "sau", "so", "tai", "tap", "theo", "thi", "thoi", "tong", "trong",
    "truoc", "tu", "ty", "va", "vao", "ve", "voi",
}


def fold(text: object) -> str:
    value = unicodedata.normalize("NFD", str(text).casefold()).replace("đ", "d")
    return "".join(char for char in value if unicodedata.category(char) != "Mn")


def terms(text: object) -> list[str]:
    return [token for token in _WORD_RE.findall(fold(text)) if token not in _STOPWORDS]


def _is_label(value: object) -> bool:
    text = str(value).strip()
    if not text or text.casefold() == "nan" or _NUMBER_RE.fullmatch(text):
        return False
    return any(char.isalpha() for char in text)


def label_column(frame: pd.DataFrame) -> int | None:
    """Return the column most likely to contain financial row labels."""
    best = None
    for index in range(len(frame.columns)):
        values = [str(value).strip() for value in frame.iloc[:, index].tolist()]
        labels = [value for value in values if _is_label(value)]
        if not labels:
            continue
        unique = len({fold(value) for value in labels})
        average_length = sum(min(len(value), 120) for value in labels) / len(labels)
        # Count and uniqueness dominate. Average length breaks ties away from
        # short code/name columns without rewarding verbose OCR noise too much.
        score = (len(labels), unique, min(average_length, 40.0))
        if best is None or score > best[0]:
            best = (score, index)
    return None if best is None else best[1]


@lru_cache(maxsize=65536)
def labels_from_csv(csv_path: str) -> tuple[str, ...]:
    path = Path(csv_path)
    try:
        frame = pd.read_csv(
            path,
            encoding="utf-8-sig",
            dtype=str,
            keep_default_na=False,
            index_col=None,
        )
    except Exception:
        return ()
    index = label_column(frame)
    if index is None:
        return ()
    values = []
    seen = set()
    for raw in frame.iloc[:, index].tolist():
        value = str(raw).strip()
        normalized = fold(_PREFIX_RE.sub("", value)).strip()
        if not _is_label(value) or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        values.append(value)
    return tuple(values)


def label_match_score(question: str, labels: Iterable[str]) -> float:
    query_terms = set(terms(question))
    if not query_terms:
        return 0.0
    query_folded = " ".join(_WORD_RE.findall(fold(question)))
    best = 0.0
    for raw_label in labels:
        normalized_label = _PREFIX_RE.sub("", fold(raw_label)).strip()
        label_terms = set(terms(normalized_label))
        if not label_terms:
            continue
        shared = len(query_terms.intersection(label_terms))
        if not shared:
            continue
        label_recall = shared / len(label_terms)
        query_precision = shared / len(query_terms)
        f2 = 5.0 * label_recall * query_precision / (4.0 * label_recall + query_precision)
        exact = 1.0 if len(normalized_label) >= 7 and normalized_label in query_folded else 0.0
        # Reward two matched content terms; a single generic token is weak.
        multi = min(1.0, max(0, shared - 1) / 2.0)
        best = max(best, f2 + 0.75 * exact + 0.20 * multi)
    return best


def rerank_tables(
    question: str,
    hits: list[dict],
    catalog: dict[str, dict],
    tables_root: str | Path,
    *,
    label_weight: float = 2.0,
) -> list[dict]:
    """Return a stable reranking of existing BM25 hits.

    ``label_weight`` is intentionally bounded by callers during experiments;
    BM25 remains the base signal and stable input rank is the final tie-break.
    """
    root = Path(tables_root)
    scored = []
    finite_scores = [float(hit.get("score", 0.0)) for hit in hits if math.isfinite(float(hit.get("score", 0.0)))]
    floor = min(finite_scores) if finite_scores else 0.0
    for rank, hit in enumerate(hits):
        row = catalog.get(str(hit.get("table_ref")), {})
        raw_score = float(hit.get("score", floor))
        if not math.isfinite(raw_score):
            raw_score = floor
        csv_path = root / str(row.get("csv_path", ""))
        labels = labels_from_csv(str(csv_path.resolve())) if csv_path.is_file() else ()
        # Catalog v2 stores source-derived section/header/blank-total context.
        # Keep the CSV labels first for backwards compatibility and add only
        # non-empty structural fields; old catalogs therefore rank identically.
        context_labels = tuple(
            part.strip()
            for field in ("section_title", "header_text", "inferred_row_text")
            for part in str(row.get(field, "")).split("|")
            if part.strip()
        )
        scoring_labels = labels + context_labels
        match = label_match_score(question, scoring_labels)
        item = dict(hit)
        item["bm25_score"] = raw_score
        item["label_match_score"] = round(match, 6)
        item["rerank_score"] = raw_score + label_weight * match
        item["retrieval_label_text"] = " | ".join(scoring_labels[:80])
        scored.append((item["rerank_score"], raw_score, -rank, item))
    scored.sort(key=lambda value: (-value[0], -value[1], -value[2]))
    return [value[3] for value in scored]
