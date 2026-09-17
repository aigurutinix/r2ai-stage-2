
from __future__ import annotations

import re
from dataclasses import dataclass

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.generation.table_index.base import TableIndexStore
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.generation.common import CandidateTable, candidate_table_from_doc, report_scope
from vifinqa.generation.retrieval.base import TableDescriptor, build_table_descriptor
from vifinqa.generation.retrieval.time_series import TimeSeriesWindow


@dataclass(frozen=True, slots=True)
class RoleRequest:
    metric_role: str
    concept_name: str
    concept_formula: str
    anchor_ref: str
    anchor_text: str = ""


RoleCandidates = dict[str, dict[str, list[CandidateTable]]]


def build_period_inventories(
    window: TimeSeriesWindow, companies: dict[str, CompanyInfo]
) -> dict[str, list[TableDescriptor]]:
    inventories: dict[str, list[TableDescriptor]] = {}
    for period in window.periods:
        descriptors: list[TableDescriptor] = []
        for doc in window.docs_by_period[period]:
            if doc.tables_dir is None or doc.text_path is None:
                continue
            scope = report_scope(doc.doc_name)
            if scope != window.report_scope:
                continue
            document = parse_document(doc.text_path)
            company = companies.get(doc.ticker)
            for table_id in doc.table_ids:
                table = load_table(
                    doc.table_csv_path(table_id),
                    ticker=doc.ticker,
                    year=doc.year,
                    doc_name=doc.doc_name,
                    table_id=table_id,
                )
                if not is_table_eligible(table):
                    continue
                descriptors.append(
                    build_table_descriptor(
                        doc=doc,
                        table=table,
                        document=document,
                        company_name=company.name if company else doc.ticker,
                        report_scope=scope,  # type: ignore[arg-type]
                    )
                )
        inventories[period] = descriptors
    return inventories


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_STOP_TOKENS = frozenset({"cột", "dòng", "năm", "nay", "trước", "giá", "trị", "khoản", "mục"})


def _tokens(text: str) -> frozenset[str]:
    return frozenset(token for token in _TOKEN_RE.findall(text.casefold()) if token not in _STOP_TOKENS)


def select_cross_period_anchor_inventory(
    inventories: dict[str, list[TableDescriptor]],
    *,
    anchor_period: str,
    max_tables: int,
) -> list[TableDescriptor]:
    anchor = inventories.get(anchor_period, [])
    other_periods = [period for period in inventories if period != anchor_period]
    token_cache = {
        descriptor.table_ref: _tokens(f"{descriptor.table_labels} {descriptor.anchor_context}")
        for descriptors in inventories.values()
        for descriptor in descriptors
    }

    def _coverage_score(descriptor: TableDescriptor) -> tuple[float, int, str]:
        source = token_cache[descriptor.table_ref]
        scores: list[float] = []
        for period in other_periods:
            best = 0.0
            for candidate in inventories.get(period, []):
                target = token_cache[candidate.table_ref]
                union = source | target
                score = len(source & target) / len(union) if union else 0.0
                best = max(best, score)
            scores.append(best)
        return (min(scores, default=0.0), len(source), descriptor.table_ref)

    return sorted(anchor, key=_coverage_score, reverse=True)[:max_tables]


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def retrieve_period_roles(
    *,
    window: TimeSeriesWindow,
    requests: list[RoleRequest],
    index_store: TableIndexStore,
    companies: dict[str, CompanyInfo],
    max_tables_per_role_period: int,
    semantic_pool_size: int = 50,
) -> RoleCandidates | None:
    result: RoleCandidates = {request.metric_role: {} for request in requests}
    for period in window.periods:
        docs = list(window.docs_by_period[period])
        doc_by_name: dict[str, DocumentRef] = {doc.doc_name: doc for doc in docs}
        index = index_store.get(
            ticker=window.ticker,
            report_scope=window.report_scope,
            period=period,
            docs=docs,
        )
        for request in requests:
            query = f"{request.concept_name}. {request.concept_formula}"
            if request.anchor_text:
                query += f". Bảng neo cùng loại: {request.anchor_text[:2500]}"
            hits = index.search(query, top_k=semantic_pool_size)
            normalized_concept = _normalize(request.concept_name)
            ranked = sorted(
                hits,
                key=lambda hit: (
                    normalized_concept in _normalize(hit[0].text),
                    hit[1],
                ),
                reverse=True,
            )
            candidates: list[CandidateTable] = []
            seen: set[str] = set()
            if period == window.anchor_period:
                doc_name, _, table_part = request.anchor_ref.partition("|table_")
                doc = doc_by_name.get(doc_name)
                if doc is not None and table_part.isdigit():
                    candidates.append(
                        candidate_table_from_doc(
                            doc,
                            int(table_part),
                            company_name=companies.get(doc.ticker).name if doc.ticker in companies else doc.ticker,
                            context_before=0,
                            context_after=0,
                        )
                    )
                    seen.add(request.anchor_ref)
            for indexed, _score in ranked:
                if len(candidates) >= max_tables_per_role_period:
                    break
                ref = f"{indexed.doc_name}|table_{indexed.table_id}"
                if ref in seen:
                    continue
                doc = doc_by_name.get(indexed.doc_name)
                if doc is None:
                    continue
                company = companies.get(doc.ticker)
                candidates.append(
                    candidate_table_from_doc(
                        doc,
                        indexed.table_id,
                        company_name=company.name if company else doc.ticker,
                        context_before=0,
                        context_after=0,
                    )
                )
                seen.add(ref)
            if not candidates:
                return None
            result[request.metric_role][period] = candidates
    return result
