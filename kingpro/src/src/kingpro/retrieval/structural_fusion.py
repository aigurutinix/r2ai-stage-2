"""Guarded structural rescue over the canonical BM25 table order.

The canonical retriever stays primary.  A table from an experimental
structural index may replace at most one weak tail item only when the source
metadata independently proves all of the following:

* a section ancestor exists;
* a multi-level header path exists;
* an unlabeled total was proven by exact Decimal additivity;
* the question asks for a total and shares source-context terms.

No question IDs, expected tables, answers, or leaderboard outcomes are used at
runtime.  The deliberately narrow gate turns structural retrieval into a
backfill channel instead of a wholesale BM25 replacement.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import bm25s

from kingpro.retrieval.bm25_index import extract_all_facets, fold
from kingpro.retrieval.table_reranker import label_match_score, terms


@dataclass(frozen=True)
class StructuralEvidence:
    score: float
    strong: bool
    shared_terms: int
    has_section: bool
    has_multilevel_header: bool
    has_proven_total: bool
    question_requests_total: bool
    single_entity_direct: bool

    def to_dict(self) -> dict:
        return asdict(self)


class StructuralSidecarRetriever:
    """Explicit, opt-in structural BM25 sidecar.

    The sidecar owns separate catalog/index paths and therefore cannot mutate
    or silently replace the canonical BM25 globals.
    """

    def __init__(self, catalog_path: str | Path, index_dir: str | Path) -> None:
        self.catalog_path = Path(catalog_path).resolve()
        self.index_dir = Path(index_dir).resolve()
        self.catalog = {
            row["table_ref"]: row
            for row in (
                json.loads(line)
                for line in self.catalog_path.open(encoding="utf-8")
            )
        }
        self.retriever = bm25s.BM25.load(str(self.index_dir), load_corpus=False)
        self.meta = [
            json.loads(line)
            for line in (self.index_dir / "meta.jsonl").open(encoding="utf-8")
        ]
        if len(self.catalog) != len(self.meta):
            raise ValueError(
                "structural sidecar catalog/index length mismatch: "
                f"{len(self.catalog)} != {len(self.meta)}"
            )

    def tables_in_reports(
        self,
        question: str,
        report_ids: list[str],
        *,
        n: int = 40,
        pool: int = 2000,
    ) -> list[dict]:
        if not report_ids:
            return []
        from kingpro.retrieval.bm25_index import tokenize

        indexes, scores = self.retriever.retrieve(
            [tokenize(question)],
            k=min(pool, len(self.meta)),
            show_progress=False,
        )
        report_set = set(report_ids)
        by_report = {report_id: [] for report_id in report_ids}
        for index, score in zip(indexes[0].tolist(), scores[0].tolist()):
            metadata = self.meta[index]
            report_id = str(metadata["table_ref"]).split("|", 1)[0]
            if report_id in report_set:
                by_report[report_id].append(
                    {"table_ref": metadata["table_ref"], "score": score, **metadata}
                )
        per_report = max(1, n // max(1, len(report_ids)))
        picked = []
        seen = set()
        for report_id in report_ids:
            for row in sorted(by_report[report_id], key=lambda item: -item["score"])[
                :per_report
            ]:
                picked.append(row)
                seen.add(row["table_ref"])
        rest = sorted(
            (
                row
                for report_id in report_ids
                for row in by_report[report_id]
                if row["table_ref"] not in seen
            ),
            key=lambda item: -item["score"],
        )
        picked.extend(rest[: max(0, n - len(picked))])
        return sorted(picked, key=lambda item: -item["score"])[:n]


def _parts(value: object) -> list[str]:
    return [part.strip() for part in str(value or "").split("|") if part.strip()]


def _question_requests_total(question: str) -> bool:
    normalized = fold(question)
    return "tong" in normalized.split() or "so du" in normalized


def structural_evidence(
    question: str,
    row: dict,
    *,
    minimum_score: float = 0.52,
    minimum_shared_terms: int = 2,
) -> StructuralEvidence:
    section = str(row.get("section_title", "")).strip()
    header = str(row.get("header_text", "")).strip()
    inferred = str(row.get("inferred_row_text", "")).strip()
    context_parts = _parts(section) + _parts(header) + _parts(inferred)
    score = label_match_score(question, context_parts) if context_parts else 0.0
    question_terms = set(terms(question))
    # The total label itself is generic.  Bind using section + header terms so
    # a random additive row cannot qualify merely because the question says
    # "tổng".
    source_terms = set(terms(f"{section} | {header}"))
    shared = len(question_terms.intersection(source_terms))
    requests_total = _question_requests_total(question)
    facets = extract_all_facets(question)
    single_entity_direct = bool(
        len(facets.get("tickers", [])) == 1 and not facets.get("analytic")
    )
    has_section = bool(section)
    has_header = bool(header and ">" in header)
    has_total = bool(inferred and any(part.startswith("Tổng cộng") for part in _parts(inferred)))
    strong = bool(
        has_section
        and has_header
        and has_total
        and requests_total
        and single_entity_direct
        and shared >= minimum_shared_terms
        and score >= minimum_score
    )
    return StructuralEvidence(
        score=round(float(score), 6),
        strong=strong,
        shared_terms=shared,
        has_section=has_section,
        has_multilevel_header=has_header,
        has_proven_total=has_total,
        question_requests_total=requests_total,
        single_entity_direct=single_entity_direct,
    )


def guarded_structural_rescue(
    question: str,
    baseline_hits: list[dict],
    structural_hits: list[dict],
    structural_catalog: dict[str, dict],
    *,
    limit: int = 8,
    max_rescues: int = 1,
    minimum_score: float = 0.52,
    minimum_shared_terms: int = 2,
    minimum_margin: float = 0.12,
) -> tuple[list[dict], dict]:
    """Return baseline-first hits plus at most ``max_rescues`` strong backfills."""

    output = [dict(hit) for hit in baseline_hits[:limit]]
    existing = {str(hit.get("table_ref", "")) for hit in output}
    trace = {
        "policy": "baseline_primary_structural_tail_rescue",
        "limit": int(limit),
        "max_rescues": int(max_rescues),
        "rescues": [],
        "rejections": [],
    }
    rescued = 0
    for candidate in structural_hits:
        if rescued >= max_rescues:
            break
        table_ref = str(candidate.get("table_ref", ""))
        if not table_ref or table_ref in existing:
            continue
        row = structural_catalog.get(table_ref, {})
        evidence = structural_evidence(
            question,
            row,
            minimum_score=minimum_score,
            minimum_shared_terms=minimum_shared_terms,
        )
        if not evidence.strong:
            trace["rejections"].append(
                {"table_ref": table_ref, "reason": "weak_structural_evidence", **evidence.to_dict()}
            )
            continue
        if len(output) < limit:
            insertion = len(output)
            removed = None
        else:
            insertion = -1
            removed = None
            for index in range(len(output) - 1, -1, -1):
                baseline_ref = str(output[index].get("table_ref", ""))
                baseline_evidence = structural_evidence(
                    question,
                    structural_catalog.get(baseline_ref, {}),
                    minimum_score=minimum_score,
                    minimum_shared_terms=minimum_shared_terms,
                )
                if baseline_evidence.strong:
                    continue
                if evidence.score < baseline_evidence.score + minimum_margin:
                    continue
                insertion = index
                removed = output[index]
                break
            if insertion < 0:
                trace["rejections"].append(
                    {"table_ref": table_ref, "reason": "no_weak_tail_slot", **evidence.to_dict()}
                )
                continue
        promoted = dict(candidate)
        promoted["structural_rescue"] = True
        promoted["structural_evidence"] = evidence.to_dict()
        if insertion == len(output):
            output.append(promoted)
        else:
            output[insertion] = promoted
            if removed is not None:
                existing.discard(str(removed.get("table_ref", "")))
        existing.add(table_ref)
        rescued += 1
        trace["rescues"].append(
            {
                "table_ref": table_ref,
                "position": insertion + 1,
                "removed_table_ref": (
                    str(removed.get("table_ref", "")) if removed is not None else None
                ),
                **evidence.to_dict(),
            }
        )
    return output, trace
