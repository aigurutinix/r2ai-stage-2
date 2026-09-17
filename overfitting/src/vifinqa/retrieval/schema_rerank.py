"""Codegen-only tail reranker that preserves the submitted top-k evidence."""
from __future__ import annotations

from collections.abc import Callable

from ..utils.viet_text import norm, tokens


_STOPWORDS = {
    "bao", "bao nhieu", "cua", "cho", "cong", "cong ty", "co", "phan",
    "cuoi", "dau", "den", "gia", "giua", "la", "nam", "ngay", "nhieu",
    "so", "tai", "theo", "tinh", "tong", "trong", "tu", "va", "vao",
    "viet", "dong", "trieu", "ty", "nghin", "tram",
}


def rerank_codegen_tail(
        record: dict, blob_for: Callable[[dict], str], *,
        freeze_top: int = 5, model_top: int = 15) -> dict:
    """Keep top ``freeze_top`` byte-for-byte and improve only model-visible tail."""
    candidates = list(record.get("candidates") or [])
    if len(candidates) <= freeze_top or model_top <= freeze_top:
        return record
    head = candidates[:freeze_top]
    tail = candidates[freeze_top:]
    slots = min(model_top - freeze_top, len(tail))
    route = record.get("route") or {}
    requirements = route.get("evidence_requirements") or []
    required = {
        str(item.get("requirement_id") or "")
        for item in requirements if item.get("requirement_id")
    }
    covered = {
        str(hit) for candidate in head
        for hit in candidate.get("requirement_hits") or []
    }
    uncovered = required - covered

    ranked = sorted(
        tail,
        key=lambda candidate: _schema_key(candidate, route, blob_for(candidate)),
        reverse=True,
    )
    promoted, selected = [], set()
    while uncovered and len(promoted) < slots:
        choices = []
        for candidate in ranked:
            key = _candidate_key(candidate)
            if key in selected:
                continue
            hits = uncovered & set(candidate.get("requirement_hits") or [])
            if not hits:
                continue
            scores = candidate.get("requirement_scores") or {}
            choices.append((
                len(hits),
                sum(float(scores.get(hit, 0.0)) for hit in hits),
                _schema_key(candidate, route, blob_for(candidate)),
                candidate,
                hits,
            ))
        if not choices:
            break
        *_rank, candidate, hits = max(choices, key=lambda item: item[:3])
        promoted.append(candidate)
        selected.add(_candidate_key(candidate))
        uncovered -= hits

    for candidate in ranked:
        if len(promoted) >= slots:
            break
        key = _candidate_key(candidate)
        if key not in selected:
            promoted.append(candidate)
            selected.add(key)

    remainder = [
        candidate for candidate in tail
        if _candidate_key(candidate) not in selected
    ]
    out = dict(record)
    out["candidates"] = [*head, *promoted, *remainder]
    out["schema_rerank"] = {
        "freeze_top": freeze_top,
        "model_top": model_top,
        "requirements": len(required),
        "covered_top5": len(required - (required - covered)),
        "missing_after_model_top": sorted(uncovered),
    }
    return out


def _schema_key(candidate: dict, route: dict, blob: str) -> tuple:
    hits = candidate.get("requirement_hits") or []
    requirement_scores = candidate.get("requirement_scores") or {}
    req_max = max((float(value) for value in requirement_scores.values()), default=0.0)
    phrase_score = _phrase_coverage(route, blob)
    return (
        bool(hits), len(hits), req_max, phrase_score,
        float(candidate.get("row_score") or 0.0),
        float(candidate.get("label_match") or 0.0),
        float(candidate.get("score") or 0.0),
    )


def _phrase_coverage(route: dict, blob: str) -> float:
    blob_norm = norm(blob)
    if not blob_norm:
        return 0.0
    phrases = [
        str(route.get("metric_norm") or ""),
        *(str(value) for value in route.get("metric_variants") or []),
        *(str(fact.get("metric") or "")
          for fact in (route.get("plan") or {}).get("facts") or []),
    ]
    blob_tokens = set(tokens(blob_norm))
    best = 0.0
    for phrase in phrases:
        phrase_norm = norm(phrase)
        wanted = {
            token for token in tokens(phrase_norm)
            if len(token) > 2 and token not in _STOPWORDS and not token.isdigit()
        }
        if not wanted:
            continue
        coverage = len(wanted & blob_tokens) / len(wanted) * 100.0
        exact = 20.0 if len(phrase_norm) >= 8 and phrase_norm in blob_norm else 0.0
        best = max(best, coverage + exact)
    return best


def _candidate_key(candidate: dict) -> tuple[str, int]:
    return str(candidate.get("report_id") or ""), int(candidate.get("table_pos") or 0)
