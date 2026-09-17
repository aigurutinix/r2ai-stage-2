
from __future__ import annotations

import hashlib
import logging
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from vifinqa.common.corpus.catalog import scan_catalog
from vifinqa.common.corpus.company_meta import get_company_meta
from vifinqa.common.corpus.document import parse_document
from vifinqa.common.corpus.table import load_table
from vifinqa.encoding.table_text import TABLE_ENCODERS
from vifinqa.common.filtering.table_filters import is_table_eligible
from vifinqa.retrieval.base import IndexedTable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CollectedCorpus:
    tables: list[IndexedTable]
    source_paths: list[Path]  # Cache behavior is part of the run contract.


def build_corpus(data_root: Path, *, table_encoding: str, company_meta_path: Path) -> CollectedCorpus:
    if table_encoding not in TABLE_ENCODERS:
        raise ValueError(f"Invalid table_encoding: {table_encoding!r} (choose from {sorted(TABLE_ENCODERS)})")
    encoder = TABLE_ENCODERS[table_encoding]
    company_meta = get_company_meta(company_meta_path)

    tables: list[IndexedTable] = []
    source_paths: list[Path] = [company_meta_path]
    warned_tickers: set[str] = set()
    for doc in scan_catalog(data_root):
        if doc.tables_dir is None or doc.text_path is None:
            continue
        document = parse_document(doc.text_path)
        source_paths.append(doc.text_path)
        info = company_meta.get(doc.ticker)
        if info is None and doc.ticker not in warned_tickers:
            logger.warning("company_name not found for ticker=%s in %s", doc.ticker, company_meta_path)
            warned_tickers.add(doc.ticker)
        company_name = info.name if info else ""
        for table_id in doc.table_ids:
            csv_path = doc.table_csv_path(table_id)
            table = load_table(csv_path, ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=table_id)
            if not is_table_eligible(table):
                continue
            source_paths.append(csv_path)
            text = unicodedata.normalize("NFC", encoder(table, document, company_name))
            tables.append(
                IndexedTable(
                    ticker=doc.ticker,
                    year=doc.year,
                    doc_name=doc.doc_name,
                    table_id=table_id,
                    text=text,
                )
            )
    return CollectedCorpus(tables=tables, source_paths=source_paths)


def fingerprint(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(f"{path}:missing")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()
