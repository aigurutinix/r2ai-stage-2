
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.document import Document
from vifinqa.common.corpus.table import TableAsset
from vifinqa.generation.common import CandidateTable, table_index_text

RetrievalResult = list[CandidateTable] | None


@dataclass(frozen=True, slots=True)
class TableDescriptor:

    table_ref: str
    ticker: str
    company_name: str
    period: str
    doc_name: str
    report_scope: Literal["consolidated", "parent", "unknown"]
    table_labels: str
    anchor_context: str
    csv_header: str
    unit_snippets: tuple[str, ...]


def build_table_descriptor(
    *,
    doc: DocumentRef,
    table: TableAsset,
    document: Document,
    company_name: str,
    report_scope: Literal["consolidated", "parent", "unknown"],
) -> TableDescriptor:
    table_ref = f"{doc.doc_name}|table_{table.table_id}"
    return TableDescriptor(
        table_ref=table_ref,
        ticker=doc.ticker,
        company_name=company_name,
        period=doc.year,
        doc_name=doc.doc_name,
        report_scope=report_scope,
        table_labels=table_index_text(table),
        anchor_context=document.table_anchor_context(table.table_id),
        csv_header=",".join(table.header),
        unit_snippets=document.table_unit_snippets(table.table_id),
    )
