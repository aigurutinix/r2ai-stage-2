"""Query-derived company/scope candidate selection before reranking.

No gold label, csv_path, answer or pandas_query is used here. Year handling intentionally stays
out of this module until it is implemented together with query decomposition.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.encoding.table_text import detect_report_scope
from vifinqa.retrieval.base import RetrievalHit, RetrievalIndex

METADATA_POLICIES = (
    "none",
    "claude_scope",
    "safe",
    "scope_or_unknown",
    "scope_hard",
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_COMPANY_LEGAL_PREFIXES = ("congtycophan", "ctcp", "congtytnhh", "tnhh")


def _ascii_compact(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    stripped = stripped.replace("đ", "d").replace("Đ", "D").lower()
    return _NON_ALNUM_RE.sub("", stripped)


def _ticker_mentioned(query: str, ticker: str) -> bool:
    return bool(
        re.search(
            rf"(?<![A-Z0-9]){re.escape(ticker.upper())}(?![A-Z0-9])", query.upper()
        )
    )


def _company_aliases(name: str) -> tuple[str, ...]:
    compact = _ascii_compact(name)
    aliases = [compact] if compact else []
    for prefix in _COMPANY_LEGAL_PREFIXES:
        if compact.startswith(prefix):
            suffix = compact[len(prefix) :]
            if len(suffix) >= 6:
                aliases.append(suffix)
            break
    return tuple(aliases)


@dataclass(frozen=True, slots=True)
class MetadataIntent:
    tickers: frozenset[str]
    explicit_scope: str | None


def parse_metadata_intent(
    query: str, companies: dict[str, CompanyInfo]
) -> MetadataIntent:
    compact = _ascii_compact(query)
    company_name_matches = {
        ticker
        for ticker, company in companies.items()
        if any(alias in compact for alias in _company_aliases(company.name))
    }
    ticker_matches = {
        ticker for ticker in companies if _ticker_mentioned(query, ticker)
    }
    shadowed_tickers = {
        ticker
        for ticker in ticker_matches
        if any(
            ticker.lower() in _ascii_compact(companies[company_ticker].name)
            for company_ticker in company_name_matches
        )
    }
    explicit_tickers = ticker_matches - shadowed_tickers
    tickers = explicit_tickers or company_name_matches or ticker_matches
    parent_markers = ("congtyme", "baocaorieng", "bctcrieng")
    consolidated_markers = ("hopnhat", "baocaohopnhat", "bctchn")
    parent = any(marker in compact for marker in parent_markers)
    consolidated = any(marker in compact for marker in consolidated_markers)
    explicit_scope = (
        "công ty mẹ"
        if parent and not consolidated
        else "hợp nhất"
        if consolidated and not parent
        else None
    )
    return MetadataIntent(tickers=frozenset(tickers), explicit_scope=explicit_scope)


def target_scope_for_policy(intent: MetadataIntent, policy: str) -> str | None:
    """Resolve scope target without looking at candidates or gold labels."""
    if policy in ("claude_scope", "scope_or_unknown", "scope_hard"):
        return intent.explicit_scope or "hợp nhất"
    return intent.explicit_scope


def hit_scope(hit: RetrievalHit) -> str | None:
    return detect_report_scope(hit.table.doc_name)


def hit_matches_entities(hit: RetrievalHit, intent: MetadataIntent) -> bool:
    return not intent.tickers or hit.table.ticker in intent.tickers


def partition_scope_candidates(
    hits: list[RetrievalHit],
    *,
    intent: MetadataIntent,
    target_scope: str,
) -> tuple[list[RetrievalHit], list[RetrievalHit]]:
    """Stable partition: matching entity + exact/unknown scope, then every fallback."""
    strict: list[RetrievalHit] = []
    fallback: list[RetrievalHit] = []
    for hit in hits:
        if hit_matches_entities(hit, intent) and hit_scope(hit) in (target_scope, None):
            strict.append(hit)
        else:
            fallback.append(hit)
    return strict, fallback


class CandidateMetadataRouter:
    """Over-fetch, select metadata-consistent candidates, then retain a global fallback quota."""

    def __init__(
        self,
        inner: RetrievalIndex,
        *,
        companies: dict[str, CompanyInfo],
        policy: str,
        fetch_n: int,
        global_fallback: int,
    ) -> None:
        if policy not in METADATA_POLICIES or policy == "none":
            raise ValueError(
                "CandidateMetadataRouter requires policy claude_scope|safe|scope_or_unknown|scope_hard; "
                f"got {policy!r}"
            )
        if fetch_n <= 0:
            raise ValueError("metadata fetch_n must be greater than 0")
        if global_fallback < 0:
            raise ValueError("metadata global_fallback must be non-negative")
        self._inner = inner
        self._companies = companies
        self._policy = policy
        self._fetch_n = fetch_n
        self._global_fallback = global_fallback

    @property
    def size(self) -> int:
        return self._inner.size

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        hits = self._inner.search(query, top_k=max(top_k, self._fetch_n))
        if not hits:
            return []
        intent = parse_metadata_intent(query, self._companies)
        target_scope = target_scope_for_policy(intent, self._policy)

        if self._policy in ("scope_or_unknown", "scope_hard"):
            assert target_scope is not None
            strict, fallback = partition_scope_candidates(
                hits,
                intent=intent,
                target_scope=target_scope,
            )
            if self._policy == "scope_hard":
                return strict[:top_k]
            fallback_count = (
                min(self._global_fallback, max(1, top_k // 10))
                if self._global_fallback
                else 0
            )
            strict_quota = top_k - fallback_count
            selected = strict[:strict_quota]
            selected.extend(fallback[:fallback_count])
            if len(selected) < top_k:
                used = {hit.table.table_ref for hit in selected}
                selected.extend(
                    hit
                    for hit in (*strict[strict_quota:], *fallback[fallback_count:])
                    if hit.table.table_ref not in used
                )
            return selected[:top_k]

        def passes(hit: RetrievalHit) -> bool:
            entity_ok = not intent.tickers or hit.table.ticker in intent.tickers
            scope = hit_scope(hit)
            scope_ok = target_scope is None or scope in (target_scope, None)
            return entity_ok and scope_ok

        def sort_key(item: tuple[int, RetrievalHit]) -> tuple[int, int, int]:
            rank, hit = item
            scope = hit_scope(hit)
            # Keep report-scope handling explicit.
            exact_scope_group = (
                0 if target_scope is not None and scope == target_scope else 1
            )
            mild_rank = (
                max(0, rank - 5)
                if self._policy == "safe"
                and target_scope is None
                and scope == "hợp nhất"
                else rank
            )
            return exact_scope_group, mild_rank, rank

        ranked = list(enumerate(hits))
        constrained = [(rank, hit) for rank, hit in ranked if passes(hit)]
        constrained.sort(key=sort_key)

        has_constraint = bool(intent.tickers or target_scope is not None)
        if not has_constraint:
            return [hit for _, hit in constrained[:top_k]]

        fallback_count = (
            min(self._global_fallback, max(1, top_k // 10))
            if self._global_fallback
            else 0
        )
        constrained_quota = top_k - fallback_count
        selected = constrained[:constrained_quota]
        global_candidates = [(rank, hit) for rank, hit in ranked if not passes(hit)]
        selected.extend(global_candidates[:fallback_count])
        return [hit for _, hit in selected[:top_k]]


class FinalScopeGuard:
    """Post-rerank stable guard that keeps known opposite scope at the tail."""

    def __init__(
        self, inner: RetrievalIndex, *, companies: dict[str, CompanyInfo]
    ) -> None:
        self._inner = inner
        self._companies = companies

    @property
    def size(self) -> int:
        return self._inner.size

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        hits = self._inner.search(query, top_k=top_k)
        intent = parse_metadata_intent(query, self._companies)
        target_scope = intent.explicit_scope or "hợp nhất"
        preferred = [hit for hit in hits if hit_scope(hit) in (target_scope, None)]
        opposite = [hit for hit in hits if hit_scope(hit) not in (target_scope, None)]
        return (preferred + opposite)[:top_k]


# Backward-compatible public name for existing callers and old A/B policies.
MetadataFilteredIndex = CandidateMetadataRouter
