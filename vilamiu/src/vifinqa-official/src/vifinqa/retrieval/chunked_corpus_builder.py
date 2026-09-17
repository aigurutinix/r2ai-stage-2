"""Build the dense row-chunk corpus while preserving canonical parent table references."""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path

from vifinqa.constants import DEFAULT_SUMMARY_MAX_CHARS
from vifinqa.common.corpus.catalog import scan_catalog
from vifinqa.common.corpus.company_meta import get_company_meta
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.encoding.row_chunks import (
    ROW_CHUNK_VIEW,
    ChunkTruncation,
    EncodedChunk,
    RowChunkConfig,
    encode_table_views,
)
from vifinqa.common.filtering.table_filters import is_table_eligible

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ChunkedCorpusStats:
    parent_tables: int
    chunks: int
    safety_split_tables: int
    safety_split_chunks: int
    safety_split_count: int
    truncated_tables: int
    truncated_chunks: int
    pathological_truncations: int


@dataclass(frozen=True, slots=True)
class CollectedChunkedCorpus:
    chunks: list[EncodedChunk]
    source_paths: list[Path]
    stats: ChunkedCorpusStats


def _log_truncation(event: ChunkTruncation) -> None:
    logger.warning(
        "Pathological OCR cell truncated: table_ref=%s section=%s row=%s cell=%s chars=%d->%d",
        event.table_ref,
        event.section,
        event.row_index,
        event.cell_index,
        event.original_chars,
        event.kept_chars,
    )


def build_chunked_corpus(
    data_root: Path,
    *,
    company_meta_path: Path,
    config: RowChunkConfig,
    views: tuple[str, ...] = (ROW_CHUNK_VIEW,),
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    metadata_csv_layout: str = "head",
    log_truncations: bool = True,
) -> CollectedChunkedCorpus:
    company_meta = get_company_meta(company_meta_path)
    chunks: list[EncodedChunk] = []
    source_paths: list[Path] = [company_meta_path]
    warned_tickers: set[str] = set()
    split_tables: set[str] = set()
    truncated_tables: set[str] = set()
    safety_split_count = 0
    safety_split_chunks = 0
    truncated_chunks = 0
    pathological_truncations = 0
    parent_tables = 0

    for doc in scan_catalog(data_root):
        if doc.tables_dir is None or doc.text_path is None:
            continue
        document = parse_document(doc.text_path)
        source_paths.append(doc.text_path)
        info = company_meta.get(doc.ticker)
        if info is None and doc.ticker not in warned_tickers:
            logger.warning(
                "company_name not found for ticker=%s in %s",
                doc.ticker,
                company_meta_path,
            )
            warned_tickers.add(doc.ticker)
        company_name = info.name if info else ""

        for table_id in doc.table_ids:
            csv_path = doc.table_csv_path(table_id)
            table = load_table(
                csv_path,
                ticker=doc.ticker,
                year=doc.year,
                doc_name=doc.doc_name,
                table_id=table_id,
            )
            if not is_table_eligible(table):
                continue
            parent_tables += 1
            source_paths.append(csv_path)
            encoded = encode_table_views(
                table,
                document,
                company_name,
                views=views,
                row_config=config,
                summary_max_chars=summary_max_chars,
                metadata_csv_layout=metadata_csv_layout,
            )
            normalized_chunks = [
                replace(chunk, text=unicodedata.normalize("NFC", chunk.text))
                for chunk in encoded.chunks
            ]
            chunks.extend(normalized_chunks)
            safety_split_count += encoded.safety_split_count
            safety_split_chunks += encoded.safety_split_windows
            truncated_chunks += encoded.truncated_chunks
            if encoded.safety_split_count:
                split_tables.add(table.table_ref)
            if encoded.truncations:
                truncated_tables.add(table.table_ref)
                pathological_truncations += len(encoded.truncations)
                if log_truncations:
                    for event in encoded.truncations:
                        _log_truncation(event)

    return CollectedChunkedCorpus(
        chunks=chunks,
        source_paths=source_paths,
        stats=ChunkedCorpusStats(
            parent_tables=parent_tables,
            chunks=len(chunks),
            safety_split_tables=len(split_tables),
            safety_split_chunks=safety_split_chunks,
            safety_split_count=safety_split_count,
            truncated_tables=len(truncated_tables),
            truncated_chunks=truncated_chunks,
            pathological_truncations=pathological_truncations,
        ),
    )
