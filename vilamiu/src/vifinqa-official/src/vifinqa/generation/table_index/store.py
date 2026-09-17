
from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.document import Document, parse_document
from vifinqa.common.corpus.table import TableAsset, load_table
from vifinqa.embeddings.base import Embedder
from vifinqa.generation.embedding_index import IndexedTable
from vifinqa.generation.table_index.numpy import NumpyTableSearchIndex
from vifinqa.common.filtering.table_filters import is_numeric_cell, is_table_eligible

logger = logging.getLogger(__name__)

INDEX_FORMAT_VERSION = 2


def _safe_name(name: str) -> str:
    return name.replace("/", "__")


def _retrieval_text(
    table: TableAsset,
    anchor_context: str,
    *,
    max_rows: int = 80,
    max_context_chars: int = 2500,
) -> str:
    columns = " | ".join(h for h in table.header if h)
    row_labels = " | ".join(
        " / ".join(cell for cell in row[:3] if cell and not is_numeric_cell(cell))
        for row in table.rows[:max_rows]
        if any(cell and not is_numeric_cell(cell) for cell in row[:3])
    )
    context = " ".join(anchor_context.split())[:max_context_chars]
    return f"Cột: {columns}. Dòng: {row_labels}. Ngữ cảnh: {context}"


def _fingerprint(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(f"{path}:missing")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _CollectedEntries:
    indexed: list[IndexedTable]
    source_paths: list[Path]


def _collect_entries(docs: list[DocumentRef]) -> _CollectedEntries:
    entries: list[IndexedTable] = []
    source_paths: list[Path] = []
    document_cache: dict[str, Document] = {}
    for doc in sorted(docs, key=lambda d: d.doc_name):
        if doc.tables_dir is None or doc.text_path is None:
            continue
        document = document_cache.get(doc.doc_name)
        if document is None:
            document = parse_document(doc.text_path)
            document_cache[doc.doc_name] = document
        source_paths.append(doc.text_path)
        for table_id in doc.table_ids:
            csv_path = doc.table_csv_path(table_id)
            table = load_table(csv_path, ticker=doc.ticker, year=doc.year, doc_name=doc.doc_name, table_id=table_id)
            if not is_table_eligible(table):
                continue
            source_paths.append(csv_path)
            context = document.table_anchor_context(table_id)
            entries.append(
                IndexedTable(
                    ticker=doc.ticker,
                    year=doc.year,
                    doc_name=doc.doc_name,
                    table_id=table_id,
                    text=_retrieval_text(table, context),
                )
            )
    return _CollectedEntries(indexed=entries, source_paths=source_paths)


class NumpyTableIndexStore:

    def __init__(self, *, embedder: Embedder, cache_dir: Path, embedding_model: str) -> None:
        self._embedder = embedder
        self._cache_dir = cache_dir
        self._embedding_model = embedding_model
        self._memory: dict[tuple[str, str, str], NumpyTableSearchIndex] = {}
        self._lock = threading.Lock()
        self.last_cache_hit: bool | None = None  # Cache behavior is part of the run contract.

    def _partition_dir(self, ticker: str, report_scope: str, period: str) -> Path:
        return self._cache_dir / "table_indexes" / _safe_name(self._embedding_model) / ticker / report_scope / period

    def get(
        self,
        *,
        ticker: str,
        report_scope: str,
        period: str,
        docs: list[DocumentRef],
    ) -> NumpyTableSearchIndex:
        key = (ticker, report_scope, period)
        with self._lock:
            cached = self._memory.get(key)
            if cached is not None:
                self.last_cache_hit = True
                return cached
            index = self._load_or_build(ticker=ticker, report_scope=report_scope, period=period, docs=docs)
            self._memory[key] = index
            return index

    def _load_or_build(
        self, *, ticker: str, report_scope: str, period: str, docs: list[DocumentRef]
    ) -> NumpyTableSearchIndex:
        collected = _collect_entries(docs)
        fingerprint = _fingerprint(collected.source_paths)
        partition_dir = self._partition_dir(ticker, report_scope, period)
        manifest_path = partition_dir / "manifest.json"
        vectors_path = partition_dir / "vectors.npy"

        if manifest_path.exists() and vectors_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
            if (
                manifest is not None
                and manifest.get("index_format_version") == INDEX_FORMAT_VERSION
                and manifest.get("embedding_model") == self._embedding_model
                and manifest.get("fingerprint") == fingerprint
            ):
                vectors = np.load(vectors_path, mmap_mode="r")
                indexed = [
                    IndexedTable(
                        ticker=e["ticker"], year=e["year"], doc_name=e["doc_name"], table_id=e["table_id"], text=e["text"]
                    )
                    for e in manifest["entries"]
                ]
                self.last_cache_hit = True
                logger.debug("table_index cache HIT: partition=%s entries=%d", (ticker, report_scope, period), len(indexed))
                return NumpyTableSearchIndex(self._embedder, indexed, vectors)

        self.last_cache_hit = False
        logger.debug(
            "table_index cache MISS: partition=%s entries=%d — rebuilding", (ticker, report_scope, period), len(collected.indexed)
        )
        if not collected.indexed:
            return NumpyTableSearchIndex(self._embedder, [], np.zeros((0, 0), dtype=np.float32))

        vectors = np.asarray(self._embedder.embed([e.text for e in collected.indexed], is_query=False), dtype=np.float32)
        partition_dir.mkdir(parents=True, exist_ok=True)
        np.save(vectors_path, vectors)
        manifest = {
            "index_format_version": INDEX_FORMAT_VERSION,
            "embedding_model": self._embedding_model,
            "fingerprint": fingerprint,
            "vector_dtype": str(vectors.dtype),
            "vector_dim": int(vectors.shape[1]) if vectors.size else 0,
            "entries": [
                {"ticker": e.ticker, "year": e.year, "doc_name": e.doc_name, "table_id": e.table_id, "text": e.text}
                for e in collected.indexed
            ],
        }
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
        return NumpyTableSearchIndex(self._embedder, collected.indexed, vectors)
