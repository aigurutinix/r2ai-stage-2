"""Optional dense (BGE-M3) matching between a metric phrase and row labels.

Design constraints:
  * The whole pipeline must keep working with NO torch installed (local CPU box)
    -> `load_encoder` returns None and every caller falls back to lexical only.
  * Embedding the label vocabulary is a GPU job, but a CHEAP one (encode-only,
    no generation). Row labels repeat massively across reports, so we embed the
    DEDUPLICATED label vocabulary once and cache it to disk; at query time only
    the handful of metric phrases need encoding.

Vocabulary size check on this corpus: ~146k tables but only ~O(100k) distinct
labels -> a single ~10 min GPU pass, reusable by every later run.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DEFAULT_MODEL = "BAAI/bge-m3"


class LabelEncoder:
    """Cosine similarity between metric phrases and a cached label vocabulary."""

    def __init__(self, model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None,
                 device: str | None = None, batch_size: int = 64):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 64          # row labels are short
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.vocab: list[str] = []
        self.matrix: np.ndarray | None = None
        self._index: dict[str, int] = {}
        if self.cache_dir and (self.cache_dir / "labels.json").exists():
            self.load_cache(self.cache_dir)

    # ---------- vocabulary cache ----------

    def build_cache(self, labels, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        vocab = sorted({str(l).strip() for l in labels if str(l).strip()})
        emb = self.model.encode(vocab, batch_size=self.batch_size,
                                normalize_embeddings=True, show_progress_bar=True)
        np.save(cache_dir / "labels.npy", np.asarray(emb, dtype=np.float16))
        (cache_dir / "labels.json").write_text(
            json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
        self.vocab, self.matrix = vocab, np.asarray(emb, dtype=np.float32)
        self._index = {l: i for i, l in enumerate(vocab)}
        print(f"[dense] cached {len(vocab)} labels -> {cache_dir}")

    def load_cache(self, cache_dir: Path) -> None:
        cache_dir = Path(cache_dir)
        self.vocab = json.loads((cache_dir / "labels.json").read_text(encoding="utf-8"))
        self.matrix = np.load(cache_dir / "labels.npy").astype(np.float32)
        self._index = {l: i for i, l in enumerate(self.vocab)}
        print(f"[dense] loaded {len(self.vocab)} cached label vectors")

    # ---------- query ----------

    def similarity(self, queries: list[str], labels: list[str]) -> dict[str, float]:
        """label -> best cosine similarity against any query (0..1)."""
        if not queries or not labels:
            return {}
        q = self.model.encode(list(queries), batch_size=self.batch_size,
                              normalize_embeddings=True)
        q = np.asarray(q, dtype=np.float32)

        known = [l for l in labels if l in self._index]
        unknown = [l for l in labels if l not in self._index]
        out: dict[str, float] = {}
        if known and self.matrix is not None:
            m = self.matrix[[self._index[l] for l in known]]
            sims = (m @ q.T).max(axis=1)
            out.update({l: float(s) for l, s in zip(known, sims)})
        if unknown:
            e = np.asarray(self.model.encode(unknown, batch_size=self.batch_size,
                                             normalize_embeddings=True),
                           dtype=np.float32)
            sims = (e @ q.T).max(axis=1)
            out.update({l: float(s) for l, s in zip(unknown, sims)})
        return out


def load_encoder(model_name: str = DEFAULT_MODEL, cache_dir: Path | None = None,
                 device: str | None = None, quiet: bool = False):
    """Return a LabelEncoder, or None when the dependency/cache is unavailable.

    Callers MUST treat None as 'lexical only' — never a hard failure.
    """
    try:
        return LabelEncoder(model_name, cache_dir, device)
    except Exception as e:  # noqa: BLE001 - optional dependency by design
        if not quiet:
            print(f"[dense] disabled ({type(e).__name__}: {e}); using lexical matching")
        return None


def collect_labels(store, tickers=None, max_per_ticker: int = 0) -> list[str]:
    """Distinct row labels across the store — the vocabulary to embed."""
    from .serialize import grid_of
    import pandas as pd  # noqa: PLC0415

    tick = tickers or sorted({t for (t, _y, _d) in store.report_index})
    seen: set[str] = set()
    for t in tick:
        df = store._load("tables", t)
        if not len(df):
            continue
        rows = df.to_dict("records")
        if max_per_ticker:
            rows = rows[:max_per_ticker]
        for m in rows:
            for row in grid_of(m)[1:]:
                for c in row[:3]:
                    c = (c or "").strip()
                    if len(c) > 3 and not c.replace(".", "").replace(",", "").isdigit():
                        seen.add(c[:160])
                        break
    return sorted(seen)
