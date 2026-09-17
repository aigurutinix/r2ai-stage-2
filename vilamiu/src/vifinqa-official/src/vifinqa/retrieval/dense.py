
from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import numpy as np

from vifinqa.constants import EMBEDDING_MAX_SEQ_LENGTH, INDEX_FORMAT_VERSION
from vifinqa.embeddings.base import Embedder
from vifinqa.retrieval.base import IndexedTable, RetrievalHit
from vifinqa.retrieval.corpus_builder import build_corpus, fingerprint

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    return name.replace("/", "__")


class DenseRetrievalIndex:
    def __init__(
        self, embedder: Embedder, entries: list[IndexedTable], vectors: np.ndarray
    ) -> None:
        self._embedder = embedder
        self._entries = entries
        self._vectors = vectors

    @property
    def size(self) -> int:
        return len(self._entries)

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if not self._entries:
            return []
        normalized_query = unicodedata.normalize("NFC", query)
        query_vec = np.asarray(
            self._embedder.embed([normalized_query], is_query=True)[0], dtype=np.float32
        )
        scores = np.asarray(self._vectors) @ query_vec

        top_k = min(top_k, len(scores))
        if top_k < len(scores):
            candidate_idx = np.argpartition(-scores, top_k - 1)[:top_k]
            order = candidate_idx[np.argsort(-scores[candidate_idx])]
        else:
            order = np.argsort(-scores)
        return [
            RetrievalHit(table=self._entries[i], score=float(scores[i])) for i in order
        ]


def _index_dir(
    cache_dir: Path,
    embedding_model: str,
    table_encoding: str,
    embedding_max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
) -> Path:
    base = cache_dir / "dense_index" / _safe_name(embedding_model) / table_encoding
    return (
        base
        if embedding_max_tokens == EMBEDDING_MAX_SEQ_LENGTH
        else base / f"tokens-{embedding_max_tokens}"
    )


def load_or_build_dense_index(
    *,
    embedder: Embedder,
    data_root: Path,
    cache_dir: Path,
    embedding_model: str,
    table_encoding: str,
    company_meta_path: Path,
    embedding_max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
    rebuild: bool = False,
) -> DenseRetrievalIndex:
    collected = build_corpus(
        data_root, table_encoding=table_encoding, company_meta_path=company_meta_path
    )
    fp = fingerprint(collected.source_paths)
    index_dir = _index_dir(
        cache_dir, embedding_model, table_encoding, embedding_max_tokens
    )
    manifest_path = index_dir / "manifest.json"
    vectors_path = index_dir / "vectors.npy"

    if not rebuild and manifest_path.exists() and vectors_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if (
            manifest is not None
            and manifest.get("index_format_version") == INDEX_FORMAT_VERSION
            and manifest.get("embedding_model") == embedding_model
            and manifest.get("table_encoding") == table_encoding
            and manifest.get("embedding_max_tokens", EMBEDDING_MAX_SEQ_LENGTH)
            == embedding_max_tokens
            and manifest.get("fingerprint") == fp
        ):
            vectors = np.load(vectors_path, mmap_mode="r")
            entries = [
                IndexedTable(
                    ticker=e["ticker"],
                    year=e["year"],
                    doc_name=e["doc_name"],
                    table_id=e["table_id"],
                    text=e["text"],
                )
                for e in manifest["entries"]
            ]
            logger.info("dense index cache HIT: %d entries", len(entries))
            return DenseRetrievalIndex(embedder, entries, np.asarray(vectors))

    logger.info(
        "Dense index cache miss; rebuilding from %d eligible tables", len(collected.tables)
    )
    if not collected.tables:
        return DenseRetrievalIndex(embedder, [], np.zeros((0, 0), dtype=np.float32))

    vectors = np.asarray(
        embedder.embed([t.text for t in collected.tables], is_query=False),
        dtype=np.float32,
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    manifest = {
        "index_format_version": INDEX_FORMAT_VERSION,
        "embedding_model": embedding_model,
        "table_encoding": table_encoding,
        "embedding_max_tokens": embedding_max_tokens,
        "fingerprint": fp,
        "vector_dim": int(vectors.shape[1]) if vectors.size else 0,
        "entries": [
            {
                "ticker": t.ticker,
                "year": t.year,
                "doc_name": t.doc_name,
                "table_id": t.table_id,
                "text": t.text,
            }
            for t in collected.tables
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return DenseRetrievalIndex(embedder, collected.tables, vectors)
