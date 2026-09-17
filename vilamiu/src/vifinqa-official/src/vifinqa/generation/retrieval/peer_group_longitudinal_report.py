
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.embeddings.base import Embedder
from vifinqa.generation.table_index.store import NumpyTableIndexStore
from vifinqa.generation.retrieval.peer_group_longitudinal import (
    DEFAULT_MAX_TRIPLES_PER_GROUP,
    PeerLongitudinalGroup,
    enumerate_peer_longitudinal_groups,
    iter_ticker_triples,
    retrieve_peer_longitudinal_bundle,
    select_best_bundle,
)

DEFAULT_PROXY_CONCEPT = ("Lợi nhuận sau thuế",)


@dataclass(frozen=True, slots=True)
class GroupFeasibility:
    industry_l3: str
    report_scope: str
    periods: tuple[str, ...]
    eligible_ticker_count: int
    triples_tried: int
    best_tickers: tuple[str, ...] | None
    best_min_score: float | None
    best_avg_score: float | None


def measure_retrieval_feasibility(
    *,
    all_docs: list[DocumentRef],
    companies: dict[str, CompanyInfo],
    embedder: Embedder,
    cache_dir: Path,
    embedding_model: str,
    concept_names: tuple[str, ...] = DEFAULT_PROXY_CONCEPT,
    max_groups: int | None = None,
    max_triples_per_group: int = DEFAULT_MAX_TRIPLES_PER_GROUP,
    seed: int = 0,
) -> list[GroupFeasibility]:
    groups: list[PeerLongitudinalGroup] = enumerate_peer_longitudinal_groups(all_docs, companies)
    if max_groups is not None:
        groups = groups[:max_groups]

    index_store = NumpyTableIndexStore(embedder=embedder, cache_dir=cache_dir, embedding_model=embedding_model)
    rng = random.Random(seed)
    results: list[GroupFeasibility] = []
    for group in groups:
        triples = iter_ticker_triples(group.eligible_tickers, rng=rng, max_triples=max_triples_per_group)
        bundles = []
        for triple in triples:
            bundle = retrieve_peer_longitudinal_bundle(
                group=group,
                tickers=triple,
                concept_names=concept_names,
                index_store=index_store,
                companies=companies,
            )
            if bundle is not None:
                bundles.append(bundle)
        best = select_best_bundle(bundles)
        results.append(
            GroupFeasibility(
                industry_l3=group.industry_l3,
                report_scope=group.report_scope,
                periods=group.periods,
                eligible_ticker_count=len(group.eligible_tickers),
                triples_tried=len(triples),
                best_tickers=best.tickers if best else None,
                best_min_score=best.min_score if best else None,
                best_avg_score=best.avg_score if best else None,
            )
        )
    return results


def render_markdown(results: list[GroupFeasibility]) -> str:
    total = len(results)
    with_bundle = sum(1 for r in results if r.best_tickers is not None)
    lines = [
        "# CP2 — peer_group_longitudinal retrieval-only feasibility",
        "",
        f"Measured groups: {total}. Groups with >=1 complete 3x3 bundle: {with_bundle}.",
        "",
        "| industry_l3 | scope | periods | eligible | triples | best tickers | min | avg |",
        "|---|---|---|---:|---:|---|---:|---:|",
    ]
    for r in results:
        best = ", ".join(r.best_tickers) if r.best_tickers else "-"
        min_s = f"{r.best_min_score:.3f}" if r.best_min_score is not None else "-"
        avg_s = f"{r.best_avg_score:.3f}" if r.best_avg_score is not None else "-"
        lines.append(
            f"| {r.industry_l3} | {r.report_scope} | {list(r.periods)} | {r.eligible_ticker_count} | "
            f"{r.triples_tried} | {best} | {min_s} | {avg_s} |"
        )
    return "\n".join(lines) + "\n"


def write_report(results: list[GroupFeasibility], *, json_out: Path, md_out: Path) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "industry_l3": r.industry_l3,
            "report_scope": r.report_scope,
            "periods": list(r.periods),
            "eligible_ticker_count": r.eligible_ticker_count,
            "triples_tried": r.triples_tried,
            "best_tickers": list(r.best_tickers) if r.best_tickers else None,
            "best_min_score": r.best_min_score,
            "best_avg_score": r.best_avg_score,
        }
        for r in results
    ]
    json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    md_out.parent.mkdir(parents=True, exist_ok=True)
    md_out.write_text(render_markdown(results), encoding="utf-8")
