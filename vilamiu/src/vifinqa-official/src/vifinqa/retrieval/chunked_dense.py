"""Dense retrieval over row chunks with deterministic max aggregation to parent tables."""

from __future__ import annotations

import json
import logging
import unicodedata
from pathlib import Path

import numpy as np

from vifinqa.constants import (
    CHUNKED_INDEX_FORMAT_VERSION,
    DEFAULT_SUMMARY_MAX_CHARS,
    EMBEDDING_MAX_SEQ_LENGTH,
    MULTI_VIEW_INDEX_FORMAT_VERSION,
)
from vifinqa.embeddings.base import Embedder
from vifinqa.encoding.row_chunks import (
    CHUNKED_TABLE_ENCODING,
    METADATA_CSV_VIEW,
    MULTI_VIEW_TABLE_ENCODING,
    ROW_CHUNK_VIEW,
    SEMANTIC_LABELS_VIEW,
    TABLE_VIEWS,
    EncodedChunk,
    RowChunkConfig,
)
from vifinqa.retrieval.base import IndexedTable, RetrievalHit
from vifinqa.retrieval.chunked_corpus_builder import build_chunked_corpus
from vifinqa.retrieval.corpus_builder import fingerprint

logger = logging.getLogger(__name__)


def _safe_name(name: str) -> str:
    return name.replace("/", "__")


def quota_union_parent_order(view_orders: list[list[int]], *, top_k: int) -> list[int]:
    """Interleave per-view parent ranks, dedupe, and keep reading deeper until full."""
    if top_k <= 0 or not view_orders:
        return []
    cursors = [0] * len(view_orders)
    selected: list[int] = []
    seen: set[int] = set()
    while len(selected) < top_k:
        progressed = False
        for view_id, order in enumerate(view_orders):
            while cursors[view_id] < len(order):
                parent_id = order[cursors[view_id]]
                cursors[view_id] += 1
                progressed = True
                if parent_id not in seen:
                    seen.add(parent_id)
                    selected.append(parent_id)
                    break
            if len(selected) >= top_k:
                break
        if not progressed:
            break
    return selected


class ChunkedDenseRetrievalIndex:
    def __init__(
        self,
        embedder: Embedder,
        chunks: list[EncodedChunk],
        vectors: np.ndarray,
        *,
        active_views: tuple[str, ...] = (ROW_CHUNK_VIEW,),
        view_fusion: str = "max",
        rerank_chunks_per_table: int = 1,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and vector counts do not match")
        self._embedder = embedder
        self._chunks = chunks
        self._vectors = vectors
        if not active_views or set(active_views) - set(TABLE_VIEWS):
            raise ValueError(f"Invalid active_views: {active_views}")
        if view_fusion not in ("max", "rrf", "quota_union"):
            raise ValueError("view_fusion must be max|rrf|quota_union")
        if rerank_chunks_per_table <= 0:
            raise ValueError("rerank_chunks_per_table must be greater than 0")
        self._active_views = active_views
        self._view_fusion = view_fusion
        self._rerank_chunks_per_table = rerank_chunks_per_table

        parent_by_ref: dict[str, int] = {}
        parents: list[IndexedTable] = []
        chunk_parent_ids: list[int] = []
        for chunk in chunks:
            parent_id = parent_by_ref.get(chunk.table_ref)
            if parent_id is None:
                parent_id = len(parents)
                parent_by_ref[chunk.table_ref] = parent_id
                parents.append(
                    IndexedTable(
                        ticker=chunk.ticker,
                        year=chunk.year,
                        doc_name=chunk.doc_name,
                        table_id=chunk.table_id,
                        text="",
                    )
                )
            chunk_parent_ids.append(parent_id)
        self._parents = parents
        self._parent_refs = np.asarray(
            [parent.table_ref for parent in parents], dtype=object
        )
        self._chunk_parent_ids = np.asarray(chunk_parent_ids, dtype=np.int64)
        self._chunk_views = np.asarray([chunk.view for chunk in chunks], dtype=object)
        chunk_ids_by_parent: list[list[int]] = [[] for _ in parents]
        for chunk_id, parent_id in enumerate(chunk_parent_ids):
            chunk_ids_by_parent[parent_id].append(chunk_id)
        self._chunk_ids_by_parent = tuple(tuple(ids) for ids in chunk_ids_by_parent)

    @property
    def size(self) -> int:
        return len(self._parents)

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if not self._chunks or top_k <= 0:
            return []
        normalized_query = unicodedata.normalize("NFC", query)
        query_vec = np.asarray(
            self._embedder.embed([normalized_query], is_query=True)[0], dtype=np.float32
        )
        raw_chunk_scores = np.asarray(self._vectors) @ query_vec
        active_mask = np.isin(self._chunk_views, self._active_views)
        chunk_scores = np.where(active_mask, raw_chunk_scores, -np.inf)

        max_parent_scores = np.full(
            len(self._parents), -np.inf, dtype=chunk_scores.dtype
        )
        np.maximum.at(max_parent_scores, self._chunk_parent_ids, chunk_scores)

        quota_order: list[int] | None = None
        if self._view_fusion == "max" or len(self._active_views) == 1:
            parent_scores = max_parent_scores
        else:
            view_orders: list[list[int]] = []
            for view in self._active_views:
                view_chunk_mask = self._chunk_views == view
                view_scores = np.full(
                    len(self._parents), -np.inf, dtype=chunk_scores.dtype
                )
                np.maximum.at(
                    view_scores,
                    self._chunk_parent_ids[view_chunk_mask],
                    raw_chunk_scores[view_chunk_mask],
                )
                valid = np.flatnonzero(np.isfinite(view_scores))
                view_order = valid[
                    np.lexsort((self._parent_refs[valid], -view_scores[valid]))
                ]
                view_orders.append([int(parent_id) for parent_id in view_order])
            if self._view_fusion == "rrf":
                parent_scores = np.zeros(len(self._parents), dtype=np.float64)
                for view_order in view_orders:
                    parent_scores[view_order] += 1.0 / (
                        60.0 + np.arange(1, len(view_order) + 1)
                    )
            else:
                quota_order = quota_union_parent_order(
                    view_orders, top_k=min(top_k, len(self._parents))
                )
                parent_scores = np.zeros(len(self._parents), dtype=np.float64)
                for rank, parent_id in enumerate(quota_order, start=1):
                    parent_scores[parent_id] = 1.0 / rank

        # First chunk wins an exact score tie. Chunks are deterministic by corpus/window order.
        matching_best = chunk_scores == max_parent_scores[self._chunk_parent_ids]
        best_chunk_ids = np.full(len(self._parents), len(self._chunks), dtype=np.int64)
        np.minimum.at(
            best_chunk_ids,
            self._chunk_parent_ids[matching_best],
            np.flatnonzero(matching_best),
        )

        # Stable total order: descending dense score, then canonical table_ref ascending.
        eligible_parents = np.flatnonzero(np.isfinite(max_parent_scores))
        order = (
            np.asarray(quota_order, dtype=np.int64)
            if quota_order is not None
            else eligible_parents[
                np.lexsort(
                    (
                        self._parent_refs[eligible_parents],
                        -parent_scores[eligible_parents],
                    )
                )
            ][: min(top_k, len(eligible_parents))]
        )
        hits: list[RetrievalHit] = []
        for parent_id in order:
            chunk = self._chunks[int(best_chunk_ids[parent_id])]
            parent = self._parents[int(parent_id)]
            text_count = self._rerank_chunks_per_table
            candidate_ids = [
                chunk_id
                for chunk_id in self._chunk_ids_by_parent[int(parent_id)]
                if active_mask[chunk_id]
            ]
            candidate_ids.sort(
                key=lambda chunk_id: (-raw_chunk_scores[chunk_id], chunk_id)
            )
            detail_ids = [
                chunk_id
                for chunk_id in candidate_ids
                if self._chunks[chunk_id].view == ROW_CHUNK_VIEW
            ]
            summary_ids = [
                chunk_id
                for chunk_id in candidate_ids
                if self._chunks[chunk_id].view == METADATA_CSV_VIEW
            ]
            summary_ids.extend(
                chunk_id
                for chunk_id in candidate_ids
                if self._chunks[chunk_id].view == SEMANTIC_LABELS_VIEW
            )
            selected_ids = candidate_ids[:1]
            if text_count > 1:
                remaining_summaries = [
                    chunk_id for chunk_id in summary_ids if chunk_id not in selected_ids
                ]
                if remaining_summaries:
                    selected_ids.append(remaining_summaries[0])
                selected_ids.extend(
                    chunk_id
                    for chunk_id in candidate_ids
                    if chunk_id not in selected_ids
                )
            selected_ids = selected_ids[:text_count]
            hits.append(
                RetrievalHit(
                    table=IndexedTable(
                        ticker=parent.ticker,
                        year=parent.year,
                        doc_name=parent.doc_name,
                        table_id=parent.table_id,
                        text=chunk.text,
                        rerank_texts=tuple(
                            self._chunks[chunk_id].text for chunk_id in selected_ids
                        ),
                        summary_text=self._chunks[summary_ids[0]].text
                        if summary_ids
                        else None,
                        detail_texts=tuple(
                            self._chunks[chunk_id].text for chunk_id in detail_ids
                        ),
                    ),
                    score=float(parent_scores[parent_id]),
                )
            )
        return hits


def _index_dir(
    cache_dir: Path,
    embedding_model: str,
    config: RowChunkConfig,
    embedding_max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
) -> Path:
    base = (
        cache_dir
        / "chunked_dense_index"
        / _safe_name(embedding_model)
        / config.namespace
    )
    return (
        base
        if embedding_max_tokens == EMBEDDING_MAX_SEQ_LENGTH
        else base / f"tokens-{embedding_max_tokens}"
    )


def _multi_view_index_dir(
    cache_dir: Path,
    embedding_model: str,
    config: RowChunkConfig,
    views: tuple[str, ...],
    summary_max_chars: int,
    metadata_csv_layout: str = "head",
    embedding_max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
) -> Path:
    view_key = "+".join(views)
    base = (
        cache_dir
        / "multi_view_dense_index"
        / _safe_name(embedding_model)
        / view_key
        / f"summary-{summary_max_chars}"
        / config.namespace
    )
    if metadata_csv_layout != "head":
        base = base / f"csv-{metadata_csv_layout}"
    if embedding_max_tokens != EMBEDDING_MAX_SEQ_LENGTH:
        base = base / f"tokens-{embedding_max_tokens}"
    return base


def _chunk_to_dict(chunk: EncodedChunk) -> dict:
    return {
        "ticker": chunk.ticker,
        "year": chunk.year,
        "doc_name": chunk.doc_name,
        "table_id": chunk.table_id,
        "chunk_index": chunk.chunk_index,
        "row_start": chunk.row_start,
        "row_end": chunk.row_end,
        "text": chunk.text,
        "view": chunk.view,
    }


def _chunk_from_dict(data: dict) -> EncodedChunk:
    return EncodedChunk(
        ticker=data["ticker"],
        year=data["year"],
        doc_name=data["doc_name"],
        table_id=data["table_id"],
        chunk_index=data["chunk_index"],
        row_start=data["row_start"],
        row_end=data["row_end"],
        text=data["text"],
        view=data.get("view", ROW_CHUNK_VIEW),
    )


def load_or_build_chunked_dense_index(
    *,
    embedder: Embedder,
    data_root: Path,
    cache_dir: Path,
    embedding_model: str,
    company_meta_path: Path,
    config: RowChunkConfig,
    table_encoding: str = CHUNKED_TABLE_ENCODING,
    views: tuple[str, ...] = (ROW_CHUNK_VIEW,),
    active_views: tuple[str, ...] | None = None,
    summary_max_chars: int = DEFAULT_SUMMARY_MAX_CHARS,
    metadata_csv_layout: str = "head",
    view_fusion: str = "max",
    rerank_chunks_per_table: int = 1,
    embedding_max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
    rebuild: bool = False,
) -> ChunkedDenseRetrievalIndex:
    if table_encoding not in (CHUNKED_TABLE_ENCODING, MULTI_VIEW_TABLE_ENCODING):
        raise ValueError(f"Invalid chunked table_encoding: {table_encoding}")
    if table_encoding == CHUNKED_TABLE_ENCODING:
        views = (ROW_CHUNK_VIEW,)
        active_views = views
    else:
        unknown = set(views) - set(TABLE_VIEWS)
        if unknown or not views:
            raise ValueError(f"Invalid views: {views}")
        active_views = active_views or views
        if set(active_views) - set(views):
            raise ValueError("active_views must be a subset of the built views")

    collected = build_chunked_corpus(
        data_root,
        company_meta_path=company_meta_path,
        config=config,
        views=views,
        summary_max_chars=summary_max_chars,
        metadata_csv_layout=metadata_csv_layout,
    )
    fp = fingerprint(collected.source_paths)
    index_dir = (
        _index_dir(cache_dir, embedding_model, config, embedding_max_tokens)
        if table_encoding == CHUNKED_TABLE_ENCODING
        else _multi_view_index_dir(
            cache_dir,
            embedding_model,
            config,
            views,
            summary_max_chars,
            metadata_csv_layout,
            embedding_max_tokens,
        )
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
            and manifest.get("index_format_version")
            == (
                CHUNKED_INDEX_FORMAT_VERSION
                if table_encoding == CHUNKED_TABLE_ENCODING
                else MULTI_VIEW_INDEX_FORMAT_VERSION
            )
            and manifest.get("embedding_model") == embedding_model
            and manifest.get("table_encoding") == table_encoding
            and manifest.get("chunk_config") == config.as_dict()
            and tuple(manifest.get("views", (ROW_CHUNK_VIEW,))) == views
            and manifest.get("summary_max_chars", summary_max_chars)
            == summary_max_chars
            and manifest.get("metadata_csv_layout", "head") == metadata_csv_layout
            and manifest.get("embedding_max_tokens", EMBEDDING_MAX_SEQ_LENGTH)
            == embedding_max_tokens
            and manifest.get("fingerprint") == fp
        ):
            chunks = [_chunk_from_dict(item) for item in manifest["chunks"]]
            vectors = np.load(vectors_path, mmap_mode="r")
            logger.info(
                "chunked dense index cache HIT: %d tables, %d chunks",
                manifest["parent_tables"],
                len(chunks),
            )
            return ChunkedDenseRetrievalIndex(
                embedder,
                chunks,
                np.asarray(vectors),
                active_views=active_views,
                view_fusion=view_fusion,
                rerank_chunks_per_table=rerank_chunks_per_table,
            )

    logger.info(
        "chunked dense index cache MISS — rebuilding (%d tables, %d chunks)",
        collected.stats.parent_tables,
        collected.stats.chunks,
    )
    if not collected.chunks:
        return ChunkedDenseRetrievalIndex(
            embedder,
            [],
            np.zeros((0, 0), dtype=np.float32),
            active_views=active_views,
            view_fusion=view_fusion,
            rerank_chunks_per_table=rerank_chunks_per_table,
        )

    vectors = np.asarray(
        embedder.embed([chunk.text for chunk in collected.chunks], is_query=False),
        dtype=np.float32,
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(vectors_path, vectors)
    manifest = {
        "index_format_version": (
            CHUNKED_INDEX_FORMAT_VERSION
            if table_encoding == CHUNKED_TABLE_ENCODING
            else MULTI_VIEW_INDEX_FORMAT_VERSION
        ),
        "embedding_model": embedding_model,
        "table_encoding": table_encoding,
        "chunk_config": config.as_dict(),
        "views": list(views),
        "summary_max_chars": summary_max_chars,
        "metadata_csv_layout": metadata_csv_layout,
        "embedding_max_tokens": embedding_max_tokens,
        "fingerprint": fp,
        "vector_dim": int(vectors.shape[1]) if vectors.size else 0,
        "parent_tables": collected.stats.parent_tables,
        "chunk_stats": {
            "chunks": collected.stats.chunks,
            "safety_split_tables": collected.stats.safety_split_tables,
            "safety_split_chunks": collected.stats.safety_split_chunks,
            "safety_split_count": collected.stats.safety_split_count,
            "truncated_tables": collected.stats.truncated_tables,
            "truncated_chunks": collected.stats.truncated_chunks,
            "pathological_truncations": collected.stats.pathological_truncations,
        },
        "chunks": [_chunk_to_dict(chunk) for chunk in collected.chunks],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return ChunkedDenseRetrievalIndex(
        embedder,
        collected.chunks,
        vectors,
        active_views=active_views,
        view_fusion=view_fusion,
        rerank_chunks_per_table=rerank_chunks_per_table,
    )
