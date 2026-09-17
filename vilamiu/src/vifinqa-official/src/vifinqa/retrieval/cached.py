
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vifinqa.retrieval.base import IndexedTable, RetrievalHit, RetrievalIndex


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace(" ", "_")


def _namespace_dir_name(namespace: str) -> str:
    safe = _safe_name(namespace)
    if len(safe.encode("utf-8")) <= 200:
        return safe
    digest = hashlib.sha256(namespace.encode()).hexdigest()[:20]
    return f"{safe[:48]}__sha256-{digest}"


class CachedRetrievalIndex:
    def __init__(
        self, inner: RetrievalIndex, *, cache_dir: Path, namespace: str
    ) -> None:
        self._inner = inner
        self._dir = cache_dir / "search_cache" / _namespace_dir_name(namespace)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def size(self) -> int:
        return self._inner.size

    def _path(self, query: str) -> Path:
        key = hashlib.sha256(query.encode()).hexdigest()
        return self._dir / f"{key}.json"

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        path = self._path(query)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload["top_k"] >= top_k:
                return [_hit_from_dict(h) for h in payload["hits"][:top_k]]

        hits = self._inner.search(query, top_k=top_k)
        payload = {"top_k": top_k, "hits": [_hit_to_dict(h) for h in hits]}
        path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return hits


def _hit_to_dict(hit: RetrievalHit) -> dict:
    t = hit.table
    return {
        "ticker": t.ticker,
        "year": t.year,
        "doc_name": t.doc_name,
        "table_id": t.table_id,
        "text": t.text,
        "rerank_texts": list(t.rerank_texts),
        "summary_text": t.summary_text,
        "detail_texts": list(t.detail_texts),
        "score": float(hit.score),
    }


def _hit_from_dict(d: dict) -> RetrievalHit:
    return RetrievalHit(
        table=IndexedTable(
            ticker=d["ticker"],
            year=d["year"],
            doc_name=d["doc_name"],
            table_id=d["table_id"],
            text=d["text"],
            rerank_texts=tuple(d.get("rerank_texts", ())),
            summary_text=d.get("summary_text"),
            detail_texts=tuple(d.get("detail_texts", ())),
        ),
        score=d["score"],
    )
