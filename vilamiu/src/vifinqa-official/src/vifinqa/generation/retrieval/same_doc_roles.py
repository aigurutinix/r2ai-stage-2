
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.generation.table_index.base import TableIndexStore
from vifinqa.generation.common import CandidateTable, candidate_table_from_doc

DEFAULT_MAX_TABLES_PER_ROLE = 3
DEFAULT_SEMANTIC_POOL_SIZE = 30


@dataclass(frozen=True, slots=True)
class DocumentRoleRequest:
    role_id: str
    concept_names: tuple[str, ...]


DocumentRoleCandidates = dict[str, list[CandidateTable]]

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.casefold()))


def retrieve_document_roles(
    *,
    doc: DocumentRef,
    requests: list[DocumentRoleRequest],
    index_store: TableIndexStore,
    company_name: str,
    report_scope: Literal["consolidated", "parent"],
    max_tables_per_role: int = DEFAULT_MAX_TABLES_PER_ROLE,
    semantic_pool_size: int = DEFAULT_SEMANTIC_POOL_SIZE,
) -> DocumentRoleCandidates | None:
    if doc.tables_dir is None or doc.text_path is None or not doc.table_ids:
        return None

    index = index_store.get(ticker=doc.ticker, report_scope=report_scope, period=doc.year, docs=[doc])

    result: DocumentRoleCandidates = {}
    for request in requests:
        query = ". ".join(request.concept_names)
        hits = index.search(query, top_k=semantic_pool_size)
        if not hits:
            return None
        normalized_concepts = {_normalize(name) for name in request.concept_names}
        ranked = sorted(
            hits,
            key=lambda hit: (
                any(concept in _normalize(hit[0].text) for concept in normalized_concepts),
                hit[1],
            ),
            reverse=True,
        )
        candidates: list[CandidateTable] = []
        for indexed, _score in ranked[:max_tables_per_role]:
            candidates.append(
                candidate_table_from_doc(
                    doc,
                    indexed.table_id,
                    company_name=company_name,
                    context_before=0,
                    context_after=0,
                )
            )
        if not candidates:
            return None
        result[request.role_id] = candidates

    return result
