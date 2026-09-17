"""Canonical evidence requirements shared by retrieval and code generation."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from ..finance.metrics import (
    find_metrics,
    get_metric,
    metric_evidence_components,
    metric_keys,
)
from ..utils.viet_text import norm


@dataclass(frozen=True)
class EvidenceRequirement:
    requirement_id: str
    ticker: str
    year: int | None
    doc_type: str
    metric_key: str
    metric_label: str
    metric_variants: tuple[str, ...]
    statement: str

    def to_dict(self) -> dict:
        value = asdict(self)
        value["metric_variants"] = list(self.metric_variants)
        return value


def build_evidence_requirements(
    question: str,
    tickers: list[str],
    years: list[int],
    doc_type: str,
    metric_variants: list[str] | None = None,
    plan: dict | None = None,
) -> list[EvidenceRequirement]:
    """Expand a question into concrete (entity, period, operand) requirements.

    Derived canonical metrics are expanded to their atomic components. This is
    deliberately a cross product: count, temporal and nested-ranking questions
    need every named entity/period to have every operand available before they
    can be answered safely.
    """
    specific_sources = [str(value) for value in (metric_variants or []) if value]
    texts = [question, *specific_sources]
    for fact in (plan or {}).get("facts", []):
        fact_metric = str(fact.get("metric") or "")
        texts.append(fact_metric)
        if fact_metric:
            specific_sources.append(fact_metric)
    root_keys = metric_keys(texts, expand_derived=False)
    if not root_keys:
        return []

    operands: list[tuple[str, int]] = []
    for root_key in root_keys:
        for operand in metric_evidence_components(root_key):
            if operand not in operands:
                operands.append(operand)

    entities = list(dict.fromkeys(str(t).upper() for t in tickers if t)) or [""]
    periods = list(dict.fromkeys(int(y) for y in years if y is not None)) or [None]
    out = []
    seen = set()

    def append_requirement(ticker: str, root_key: str, key: str,
                           evidence_year: int | None) -> None:
        metric = get_metric(key)
        specific = [
            norm(source) for source in specific_sources
            if key in metric_keys([source], expand_derived=False)
        ]
        variants = tuple(dict.fromkeys(
            value for value in (*specific, *metric.variants) if value
        ))
        req_id = (f"{ticker or '*'}|"
                  f"{evidence_year if evidence_year is not None else '*'}|{key}")
        if req_id in seen:
            return
        seen.add(req_id)
        out.append(EvidenceRequirement(
            requirement_id=req_id,
            ticker=ticker,
            year=evidence_year,
            doc_type=doc_type,
            metric_key=key,
            metric_label=metric.label,
            metric_variants=variants,
            statement=metric.statement,
        ))

    for ticker in entities:
        for year in periods:
            for key, year_offset in operands:
                evidence_year = None if year is None else year + year_offset
                append_requirement(ticker, "", key, evidence_year)

    # Average-balance formulas need the opening stock from the preceding year
    # in addition to the closing stock already named by the question.
    average_balance_keys = _average_balance_keys(question)
    for ticker in entities:
        for year in periods:
            if year is None:
                continue
            for key in average_balance_keys:
                append_requirement(ticker, key, key, int(year) - 1)

    # Growth is evaluated against each candidate year's immediately preceding
    # period. The cross product already covers prior years inside a stated
    # range, so this mainly adds the opening boundary year.
    temporal_roots = _implied_previous_period_keys(question)
    for ticker in entities:
        for year in periods:
            if year is None:
                continue
            for root_key in temporal_roots:
                for key, year_offset in metric_evidence_components(root_key):
                    append_requirement(
                        ticker, root_key, key, int(year) - 1 + year_offset)

    # A projection at the year after the selected event can fall just outside
    # the candidate interval. Reserve exact target evidence for every possible
    # selected year; duplicate requirements are removed by append_requirement.
    if _asks_next_period(question):
        for ticker in entities:
            for year in periods:
                if year is None:
                    continue
                for root_key in root_keys:
                    for key, year_offset in metric_evidence_components(root_key):
                        append_requirement(
                            ticker, root_key, key, int(year) + 1 + year_offset)
    return out


def _average_balance_keys(question: str) -> set[str]:
    text = norm(question)
    average = r"(?:binh quan|trung binh)"
    aliases = {
        "total_assets": ("tong tai san",),
        "equity": ("von chu so huu",),
        "fixed_assets": ("tai san co dinh thuan", "tai san co dinh"),
    }
    keys = set()
    for key, values in aliases.items():
        if any(
            re.search(rf"\b{re.escape(value)}\s+{average}\b", text)
            or re.search(rf"\b{average}\s+{re.escape(value)}\b", text)
            for value in values
        ):
            keys.add(key)
    if re.search(
            r"\bvong quay tong tai san(?:\s+\([^)]*\))?\s+"
            r"(?:tinh\s+)?(?:theo\s+)?tai san\s+"
            r"(?:binh quan|trung binh)\b", text):
        keys.add("total_assets")
    return keys


def _implied_previous_period_keys(question: str) -> set[str]:
    text = norm(question)
    matches = find_metrics(question)
    keys = set()
    markers = (
        "tang truong", "toc do tang", "ty le tang", "phan tram tang",
        "muc tang tuong doi", "tang tuong doi",
    )
    pattern = "|".join(re.escape(marker) for marker in markers)
    for marker in re.finditer(pattern, text):
        following = [
            match for match in matches
            if match.start >= marker.end() and match.start - marker.end() <= 45
        ]
        if following:
            keys.add(min(following, key=lambda match: match.start).metric.key)
    return keys


def _asks_next_period(question: str) -> bool:
    text = norm(question)
    return any(value in text for value in (
        "nam ngay sau nam", "nam lien sau nam", "nam sau nam",
        "nam ke tiep", "nam lien ke", "cuoi nam ke tiep",
    ))


def evidence_coverage(requirements: list[dict], candidates: list[dict]) -> dict:
    """Summarize which requirements are covered by the selected table set."""
    required = [str(r.get("requirement_id") or "") for r in requirements]
    required = list(dict.fromkeys(r for r in required if r))
    covered = {
        str(req_id)
        for candidate in candidates
        for req_id in candidate.get("requirement_hits", [])
        if req_id
    }
    missing = [req_id for req_id in required if req_id not in covered]
    return {
        "required": len(required),
        "covered": len(required) - len(missing),
        "complete": bool(required) and not missing,
        "missing": missing,
    }
