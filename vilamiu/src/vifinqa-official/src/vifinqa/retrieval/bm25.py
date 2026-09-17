
from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path

import bm25s
from underthesea import word_tokenize

from vifinqa.constants import BM25_TOKENIZER_VERSION, INDEX_FORMAT_VERSION
from vifinqa.retrieval.base import IndexedTable, RetrievalHit
from vifinqa.retrieval.corpus_builder import build_corpus, fingerprint

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).replace("\u00a0", " ").casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def _tokenize(text: str) -> list[str]:
    return word_tokenize(_normalize_text(text))


class BM25RetrievalIndex:
    def __init__(self, model: bm25s.BM25 | None, entries: list[IndexedTable]) -> None:
        self._model = model
        self._entries = entries

    @property
    def size(self) -> int:
        return len(self._entries)

    def search(self, query: str, *, top_k: int) -> list[RetrievalHit]:
        if not self._entries or self._model is None:
            return []
        top_k = min(top_k, len(self._entries))
        query_tokens = [_tokenize(query)]
        results, scores = self._model.retrieve(query_tokens, k=top_k, show_progress=False)
        return [
            RetrievalHit(table=self._entries[int(idx)], score=float(score))
            for idx, score in zip(results[0], scores[0], strict=True)
        ]


def _index_dir(cache_dir: Path, table_encoding: str) -> Path:
    return cache_dir / "bm25_index" / table_encoding


def load_or_build_bm25_index(
    *, data_root: Path, cache_dir: Path, table_encoding: str, company_meta_path: Path, rebuild: bool = False
) -> BM25RetrievalIndex:
    collected = build_corpus(data_root, table_encoding=table_encoding, company_meta_path=company_meta_path)
    fp = fingerprint(collected.source_paths)
    index_dir = _index_dir(cache_dir, table_encoding)
    manifest_path = index_dir / "manifest.json"

    if not rebuild and manifest_path.exists() and (index_dir / "params.index.json").exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = None
        if (
            manifest is not None
            and manifest.get("index_format_version") == INDEX_FORMAT_VERSION
            and manifest.get("tokenizer_version") == BM25_TOKENIZER_VERSION
            and manifest.get("table_encoding") == table_encoding
            and manifest.get("fingerprint") == fp
        ):
            model = bm25s.BM25.load(str(index_dir), load_corpus=False)
            entries = [
                IndexedTable(
                    ticker=e["ticker"], year=e["year"], doc_name=e["doc_name"], table_id=e["table_id"], text=e["text"]
                )
                for e in manifest["entries"]
            ]
            logger.info("bm25 index cache HIT: %d entries", len(entries))
            return BM25RetrievalIndex(model, entries)

    logger.info("BM25 index cache miss; rebuilding from %d eligible tables", len(collected.tables))
    index_dir.mkdir(parents=True, exist_ok=True)
    if not collected.tables:
        return BM25RetrievalIndex(None, [])

    corpus_tokens = [_tokenize(t.text) for t in collected.tables]
    model = bm25s.BM25()
    model.index(corpus_tokens, show_progress=True)
    model.save(str(index_dir))

    manifest = {
        "index_format_version": INDEX_FORMAT_VERSION,
        "tokenizer_version": BM25_TOKENIZER_VERSION,
        "table_encoding": table_encoding,
        "fingerprint": fp,
        "entries": [
            {"ticker": t.ticker, "year": t.year, "doc_name": t.doc_name, "table_id": t.table_id, "text": t.text}
            for t in collected.tables
        ],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return BM25RetrievalIndex(model, collected.tables)
