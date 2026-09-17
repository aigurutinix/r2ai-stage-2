"""Retrieval = structured lookup (router) + BM25 inside the locked report(s).

Writes retrieval.jsonl: one record per question with the route and ranked
candidate tables.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

from ..config import RETRIEVE_DEPTH
from ..extraction.build_store import Store
from ..finance.metrics import expand_metric_variants, metric_context_matches
from ..router.entities import StockMap
from ..router.decompose import split_ratio_metric
from ..router.evidence import evidence_coverage
from ..router.router import route_question
from ..utils.io import read_jsonl, write_jsonl
from ..utils.viet_text import fuzz_token_set, norm, tokens
from .bm25 import BM25
from .serialize import table_doc_tokens, grid_of, tidy_csv_text
from .shortlist import (build_shortlist, candidate_matches_requirement,
                        requirement_linking_variants,
                        requirement_specificity_key)

_LABEL_BOOST = 1.5
_LABEL_THRESHOLD = 88.0
_ROW_SCORE_WEIGHT = 0.18
_REQUIREMENT_SCORE_CACHE: OrderedDict[tuple, float | None] = OrderedDict()
_REQUIREMENT_SCORE_CACHE_SIZE = 50000

def retrieve_for_route(route, store: Store, depth: int = RETRIEVE_DEPTH,
                       row_rerank: bool = False,
                       row_score_weight: float = _ROW_SCORE_WEIGHT) -> list[dict]:
    if not route.report_ids or not route.tickers:
        return []
    cands = []
    for ticker in route.tickers:
        tdf = store.tables_of(ticker, route.report_ids)
        if not len(tdf):
            continue
        metas = tdf.to_dict("records")
        grids = [grid_of(m) for m in metas]
        docs = [table_doc_tokens(m, g) for m, g in zip(metas, grids)]
        bm25 = BM25(docs)
        base_variants = _base_metric_variants(route)
        variants = _retrieval_metric_variants(route)
        expanded = bool(set(variants) - set(base_variants))
        rank_variants = variants if expanded else base_variants
        scores = _combined_bm25_scores(
            bm25, base_variants, variants, getattr(route, "years", []) or []
        )
        requirement_matches = _requirement_table_matches(route, metas)
        base_order = sorted(
            range(len(metas)), key=lambda i: -scores[i]
        )[: max(depth * 3, 60)]
        by_key = {
            (m["report_id"], int(m["table_pos"])): i
            for i, m in enumerate(metas)
        }
        targeted = sorted(
            (by_key[key] for key in requirement_matches if key in by_key),
            key=lambda i: -max(requirement_matches[
                (metas[i]["report_id"], int(metas[i]["table_pos"]))
            ].values()),
        )
        order = list(dict.fromkeys([*base_order, *targeted]))
        row_scores = _row_scores(route, metas, order, rank_variants) if row_rerank else {}
        for i in order:
            m, g = metas[i], grids[i]
            s = scores[i]
            # number-aware-lite boost: a row label closely matching the metric
            best_lab = 0.0
            for row in g[1:60]:
                for cell in row[:3]:
                    if cell and len(cell) > 3:
                        best_lab = max(
                            best_lab,
                            max((fuzz_token_set(cell, v) for v in rank_variants),
                                default=0.0),
                        )
                        break
            if best_lab >= _LABEL_THRESHOLD:
                s *= _LABEL_BOOST
            key = (m["report_id"], int(m["table_pos"]))
            row_score = row_scores.get(key, 0.0)
            req_scores = requirement_matches.get(key, {})
            bm25_score = float(s)
            final_score = bm25_score + row_score_weight * row_score
            cands.append({
                "report_id": m["report_id"], "ticker": m["ticker"],
                "table_pos": int(m["table_pos"]), "page": int(m["page"]),
                "unit_scale": (None if m["unit_scale"] is None or m["unit_scale"] != m["unit_scale"]
                               else float(m["unit_scale"])),
                "unit_source": m["unit_source"],
                "n_rows": int(m["n_rows"]), "score": round(float(final_score), 4),
                "bm25_score": round(bm25_score, 4),
                "row_score": round(float(row_score), 1),
                "label_match": round(best_lab, 1),
                "requirement_hits": sorted(req_scores),
                "requirement_scores": {
                    req_id: round(float(req_score), 1)
                    for req_id, req_score in sorted(req_scores.items())
                },
                "requirement_score": round(max(req_scores.values()), 1)
                if req_scores else 0.0,
            })
    cands.sort(key=lambda c: -c["score"])
    return _apply_quota(cands, route, depth)


def _base_metric_variants(route) -> list[str]:
    return _dedupe_variants(
        list(getattr(route, "metric_variants", None) or [getattr(route, "metric_norm", "")])
    )


def _retrieval_metric_variants(route) -> list[str]:
    """Metric phrases used only for table retrieval/rerank.

    Composite questions often ask one final metric after filtering/ranking on
    several other metrics. Pulling those atomic fact metrics and formula parts
    into retrieval improves table recall without changing the router's primary
    metric that downstream codegen sees.
    """
    out: list[str] = []
    for v in (getattr(route, "metric_variants", None) or [getattr(route, "metric_norm", "")]):
        out.append(str(v or ""))
    plan = getattr(route, "plan", None) or {}
    facts = plan.get("facts", []) if isinstance(plan, dict) else []
    for fact in facts:
        out.append(str(fact.get("metric") or ""))
    for v in list(out):
        num, den = split_ratio_metric(v)
        out.extend([num, den])
    return expand_metric_variants(
        _dedupe_variants(out),
        question=getattr(route, "question", "") or "",
    )


def _combined_bm25_scores(bm25: BM25, base_variants: list[str],
                          variants: list[str], years: list[int]) -> list[float]:
    """Legacy BM25 foundation, with component-metric recall as a bounded boost."""
    if not bm25.docs:
        return []
    year_tokens = [str(y) for y in years]
    base_tokens: list[str] = []
    for v in base_variants:
        base_tokens.extend(tokens(v))
    base_tokens.extend(year_tokens)
    base = bm25.scores(base_tokens)
    extras = [v for v in variants if v and v not in set(base_variants)]
    if not extras:
        return base
    if not base_tokens:
        return bm25.scores(year_tokens)
    best_extra = [0.0] * len(bm25.docs)
    for v in extras[:32]:
        q_tokens = tokens(v) + year_tokens
        scores = bm25.scores(q_tokens)
        for i, s in enumerate(scores):
            if s > best_extra[i]:
                best_extra[i] = s
    return [b + 0.35 * e for b, e in zip(base, best_extra)]


def _dedupe_variants(variants: list[str]) -> list[str]:
    seen, out = set(), []
    for v in variants:
        v = " ".join(norm(v).split())
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _row_scores(route, metas: list[dict], order: list[int],
                variants: list[str] | None = None) -> dict[tuple[str, int], float]:
    """Best row-level schema-linking score per table.

    BM25 ranks whole tables. For table retrieval we also care whether a table
    contains a concrete row/cell that looks like the asked metric and period.
    `build_shortlist` is already the row linker used by rule/codegen; reusing it
    here makes the submitted top-k closer to the evidence the answer path would
    actually use.
    """
    variants = variants or _retrieval_metric_variants(route)
    if not variants:
        return {}
    blocks = []
    for j, i in enumerate(order, start=1):
        m = metas[i]
        blocks.append({
            "var": f"df{j}",
            "report_id": m["report_id"],
            "table_pos": int(m["table_pos"]),
            "csv_text": tidy_csv_text(m),
        })
    # Ask for more rows than tables: several top rows can live in the same
    # table, and we only need the best row score per table.
    shortlist = build_shortlist(
        blocks, variants, getattr(route, "years", []) or [],
        top_n=min(80, max(12, len(blocks) * 2)),
        min_score=35.0,
        question=getattr(route, "question", "") or "",
    )
    out: dict[tuple[str, int], float] = {}
    for c in shortlist:
        key = (c.report_id, int(c.table_pos))
        out[key] = max(out.get(key, 0.0), float(c.score))
    return out


def _requirement_table_matches(
    route, metas: list[dict], per_requirement: int = 2,
) -> dict[tuple[str, int], dict[str, float]]:
    """Find exact table candidates independently for every formula operand."""
    requirements = getattr(route, "evidence_requirements", None) or []
    if not requirements or not metas:
        return {}

    out: dict[tuple[str, int], dict[str, float]] = {}
    ticker = str(metas[0].get("ticker") or "").upper()
    for requirement in requirements:
        if str(requirement.get("ticker") or "").upper() not in {"", ticker}:
            continue
        req_id = str(requirement.get("requirement_id") or "")
        variants = list(requirement.get("metric_variants") or [])
        year = requirement.get("year")
        if not req_id or not variants:
            continue

        scoped = [
            i for i, meta in enumerate(metas)
            if year is None or int(meta.get("year") or 0) in {int(year), int(year) + 1}
        ]
        if not scoped:
            scoped = list(range(len(metas)))
        scored_tables = []
        for i in scoped:
            meta = metas[i]
            score = _table_requirement_score(meta, requirement)
            if score is not None:
                scored_tables.append((float(score), meta))
        scored_tables.sort(key=lambda item: -item[0])
        for score, meta in scored_tables[:per_requirement]:
            key = (meta["report_id"], int(meta["table_pos"]))
            out.setdefault(key, {})[req_id] = score
    return out


def _table_requirement_score(meta: dict, requirement: dict) -> float | None:
    """Best exact row score for one stable (table, metric, period) key."""
    year = requirement.get("year")
    cache_key = (
        meta["report_id"],
        int(meta["table_pos"]),
        str(requirement.get("metric_key") or ""),
        int(year) if year is not None else None,
        requirement_specificity_key(requirement),
    )
    if cache_key in _REQUIREMENT_SCORE_CACHE:
        value = _REQUIREMENT_SCORE_CACHE.pop(cache_key)
        _REQUIREMENT_SCORE_CACHE[cache_key] = value
        return value

    metric_key = str(requirement.get("metric_key") or "")
    if metric_key == "operating_lease_commitments":
        value = _operating_lease_schedule_score(meta, requirement)
        _REQUIREMENT_SCORE_CACHE[cache_key] = value
        return value
    if not metric_context_matches(metric_key, str(meta.get("context") or "")):
        _REQUIREMENT_SCORE_CACHE[cache_key] = None
        return None

    block = {
        "var": "req",
        "report_id": meta["report_id"],
        "table_pos": int(meta["table_pos"]),
        "report_year": int(meta["year"]),
        "csv_text": tidy_csv_text(meta),
    }
    shortlist = build_shortlist(
        [block],
        requirement_linking_variants(requirement),
        [int(year)] if year is not None else [],
        top_n=40,
        min_score=62.0,
        question=str(requirement.get("metric_label") or ""),
    )
    exact = [
        candidate.score for candidate in shortlist
        if candidate_matches_requirement(candidate, requirement)
    ]
    value = float(max(exact)) if exact else None
    _REQUIREMENT_SCORE_CACHE[cache_key] = value
    if len(_REQUIREMENT_SCORE_CACHE) > _REQUIREMENT_SCORE_CACHE_SIZE:
        _REQUIREMENT_SCORE_CACHE.popitem(last=False)
    return value


def _operating_lease_schedule_score(
        meta: dict, requirement: dict) -> float | None:
    """Recognize a maturity schedule even though its total row is generic."""
    context = norm(str(meta.get("context") or ""))
    grid = norm(str(meta.get("grid_json") or ""))
    blob = f"{context} {grid}"
    short = any(value in blob for value in (
        "den 1 nam", "duoi 1 nam", "duoi mot nam", "1 nam tro xuong",
        "01 nam tro xuong",
    ))
    medium = any(value in blob for value in (
        "tren 1 den 5 nam", "tren 1 nam den 5 nam", "tren 1 5 nam",
        "tu 1 den 5 nam", "tu 1 5 nam", "tu mot den nam nam",
    ))
    if not short or not medium or "thue" not in context:
        return None

    variants = [norm(value) for value in requirement.get("metric_variants") or []]
    metric_label = norm(str(requirement.get("metric_label") or ""))
    try:
        source_variants = variants[:variants.index(metric_label)]
    except ValueError:
        source_variants = variants[:1]
    asked = " ".join(source_variants or variants[:1])
    wants_receivable = any(value in asked for value in (
        "cho thue", "phai thu", "phai nhan", "thu duoc",
    ))
    wants_payable = any(value in asked for value in (
        "phai tra", "ben di thue",
    ))
    context_receivable = any(value in context for value in (
        "cho thue", "ben cho thue", "phai thu", "thu duoc",
    ))
    context_payable = any(value in context for value in (
        "ben di thue", "phai tra", "cong ty thue", "tap doan thue",
        "nhom cong ty thue",
    ))
    if wants_receivable and not context_receivable:
        return None
    if wants_payable and not context_payable:
        return None
    return 96.0


def _apply_quota(cands: list[dict], route, depth: int) -> list[dict]:
    """Dynamic evidence allocation (P1.4).

    A flat top-k starves composite questions: for "chênh lệch giữa A và B" a
    pure score ranking can return 5 tables that all belong to A. Guarantee at
    least `per_doc` slots for every locked report, then fill by score.
    """
    if not cands:
        return []
    plan = getattr(route, "plan", None) or {}
    facts = plan.get("facts", [])
    requirements = getattr(route, "evidence_requirements", None) or []
    required = {
        str(requirement.get("requirement_id") or "")
        for requirement in requirements
        if requirement.get("requirement_id")
    }
    out, selected = [], set()

    # When the evidence budget can hold one table per operand, reserve the
    # strongest exact row match independently. A compact set-cover table can be
    # semantically wrong (for example a disposal note containing both "current
    # assets" and "current liabilities") while the two statement tables carry
    # the stable VAS codes requested by the formula.
    uncovered = set(required)
    if len(required) <= depth:
        for req_id in sorted(required):
            choices = [
                candidate for candidate in cands
                if req_id in candidate.get("requirement_hits", [])
            ]
            if not choices:
                continue
            candidate = max(
                choices,
                key=lambda item: (
                    float(item.get("requirement_scores", {}).get(req_id, 0.0)),
                    float(item.get("score", 0.0)),
                ),
            )
            key = (candidate["report_id"], int(candidate["table_pos"]))
            if key not in selected:
                out.append(candidate)
                selected.add(key)
            uncovered -= set(candidate.get("requirement_hits", []))

    # Greedy set cover: reserve as few tables as possible while giving every
    # operand an exact row-level table candidate.
    while uncovered and len(out) < depth:
        eligible = []
        for candidate in cands:
            key = (candidate["report_id"], int(candidate["table_pos"]))
            if key in selected:
                continue
            new_hits = uncovered & set(candidate.get("requirement_hits", []))
            if not new_hits:
                continue
            req_scores = candidate.get("requirement_scores", {})
            eligible.append((
                len(new_hits),
                sum(float(req_scores.get(req_id, 0.0)) for req_id in new_hits),
                float(candidate.get("score", 0.0)),
                candidate,
                new_hits,
            ))
        if not eligible:
            break
        _, _, _, candidate, new_hits = max(eligible, key=lambda item: item[:3])
        out.append(candidate)
        selected.add((candidate["report_id"], int(candidate["table_pos"])))
        uncovered -= new_hits

    if len(facts) <= 1 and len(requirements) <= 1:
        for candidate in cands:
            if len(out) >= depth:
                break
            key = (candidate["report_id"], int(candidate["table_pos"]))
            if key not in selected:
                out.append(candidate)
                selected.add(key)
        return out[:depth]

    docs = list(dict.fromkeys(c["report_id"] for c in cands))
    per_doc = max(1, depth // max(1, len(docs)))
    taken = {}
    for candidate in out:
        report_id = candidate["report_id"]
        taken[report_id] = taken.get(report_id, 0) + 1
    for c in cands:
        if len(out) >= depth:
            break
        key = (c["report_id"], int(c["table_pos"]))
        if key not in selected and taken.get(c["report_id"], 0) < per_doc:
            out.append(c)
            selected.add(key)
            taken[c["report_id"]] = taken.get(c["report_id"], 0) + 1
    for c in cands:                      # fill the remainder by pure score
        if len(out) >= depth:
            break
        key = (c["report_id"], int(c["table_pos"]))
        if key not in selected:
            out.append(c)
            selected.add(key)
    return out[:depth]


def run_retrieval(questions_path: Path, store_dir: Path, code_stock_csv: Path,
                  out_path: Path, depth: int = RETRIEVE_DEPTH, limit: int = 0,
                  row_rerank: bool = False,
                  row_score_weight: float = _ROW_SCORE_WEIGHT,
                  question_ids: set[int] | None = None,
                  base_path: Path | None = None) -> None:
    store = Store(store_dir)
    stock = StockMap(code_stock_csv)
    questions = read_jsonl(questions_path)
    if question_ids:
        questions = [q for q in questions if int(q["id"]) in question_ids]
    if limit:
        questions = questions[:limit]
    out = []
    try:
        from tqdm import tqdm
        it = tqdm(questions, desc="retrieve")
    except ImportError:
        it = questions
    for q in it:
        route = route_question(q["id"], q["question"], stock, store)
        cands = retrieve_for_route(
            route, store, depth, row_rerank=row_rerank,
            row_score_weight=row_score_weight,
        )
        route_dict = route.to_dict()
        out.append({
            "id": q["id"],
            "question": q["question"],
            "route": route_dict,
            "candidates": cands,
            "evidence": evidence_coverage(
                route_dict.get("evidence_requirements", []), cands
            ),
        })
    if base_path is not None:
        base = read_jsonl(base_path)
        replacements = {int(row["id"]): row for row in out}
        out = [replacements.get(int(row["id"]), row) for row in base]
        existing = {int(row["id"]) for row in base}
        out.extend(row for row_id, row in replacements.items()
                   if row_id not in existing)
    write_jsonl(out_path, out)
    n_empty = sum(1 for r in out if not r["candidates"])
    print(f"retrieval done: {len(out)} questions, {n_empty} with no candidates")
