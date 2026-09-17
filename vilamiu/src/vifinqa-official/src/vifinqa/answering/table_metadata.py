
from __future__ import annotations

from pathlib import Path

from vifinqa.answering.base import TablePromptMetadata
from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.document import Document, parse_document
from vifinqa.retrieval.base import parse_table_ref


def build_table_metadata_lookup(
    table_refs: list[str] | tuple[str, ...] | set[str],
    *,
    doc_name_lookup: dict[str, DocumentRef],
    company_names: dict[str, str],
) -> dict[str, TablePromptMetadata]:
    documents: dict[Path, Document] = {}
    result: dict[str, TablePromptMetadata] = {}
    for table_ref in table_refs:
        doc_name, table_id = parse_table_ref(table_ref)
        doc = doc_name_lookup.get(doc_name)
        if doc is None:
            continue

        anchor_context = ""
        if doc.text_path is not None:
            document = documents.get(doc.text_path)
            if document is None:
                document = parse_document(doc.text_path)
                documents[doc.text_path] = document
            anchor_context = document.table_anchor_context(table_id, lines_before=6, lines_after=2)

        result[table_ref] = TablePromptMetadata(
            table_ref=table_ref,
            ticker=doc.ticker,
            year=doc.year,
            doc_name=doc.doc_name,
            company_name=company_names.get(doc.ticker, ""),
            anchor_context=anchor_context,
        )
    return result


__all__ = ["build_table_metadata_lookup"]
