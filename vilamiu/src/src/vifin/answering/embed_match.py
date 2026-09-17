"""Match the question's metric phrase to a row label by embedding similarity.

A third, independent localiser. The regex matcher needs shared tokens, so it
misses "Lãi thuần từ hoạt động dịch vụ" against "Thu nhập thuần từ dịch vụ"; the
model localiser reads the whole table but its answers are so far unmeasured. An
embedding compares meaning without either an exact-token requirement or an API
call, which makes it useful twice over:

* it can fill gaps the regex matcher leaves, deterministically and offline;
* it is a third opinion on the 151 questions where regex and the model disagree
  and nothing currently says which is right.

`BAAI/bge-m3` — 568M, January 2024, multilingual with strong Vietnamese — is
within the contest limits (open weights, <= 14B, before 2026-06-01).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MODEL_ID = "BAAI/bge-m3"

# Row labels are short; anything longer is prose that crept into column 0.
MAX_LABEL_CHARS = 120


def clean_label(text: str) -> str:
    label = re.sub(r"\s+", " ", str(text)).strip()
    # Statement lines carry an ordinal prefix that carries no meaning for
    # similarity: "1. Doanh thu thuần", "C. NỢ PHẢI TRẢ", "I. Nợ ngắn hạn".
    label = re.sub(r"^(?:[0-9IVXivx]+\s*[.)\-]\s*)+", "", label).strip()
    return label[:MAX_LABEL_CHARS]


@dataclass
class LabelEmbedder:
    model: object
    cache: dict[str, object] = field(default_factory=dict)
    batch_size: int = 96

    @classmethod
    def load(cls, model_id: str = MODEL_ID, device: str | None = None, **kwargs):
        import torch
        from sentence_transformers import SentenceTransformer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_id, device=device)
        if device == "cuda":
            model = model.half()
        return cls(model=model, **kwargs)

    def encode(self, texts: list[str]):
        """Embed, reusing anything already seen — labels repeat heavily."""

        import numpy as np

        missing = [t for t in dict.fromkeys(texts) if t and t not in self.cache]
        if missing:
            vectors = self.model.encode(
                missing, batch_size=self.batch_size, normalize_embeddings=True,
                show_progress_bar=False, convert_to_numpy=True,
            )
            for text, vector in zip(missing, vectors):
                self.cache[text] = vector
        dim = self.model.get_sentence_embedding_dimension()
        return np.stack([
            self.cache.get(t, np.zeros(dim, dtype="float32")) for t in texts
        ])


@dataclass(frozen=True, slots=True)
class Match:
    table: int
    row: int
    label: str
    score: float


def best_label(
    embedder: LabelEmbedder,
    metric: str,
    grids: list[list[list[str]]],
    min_score: float,
) -> Match | None:
    """Highest-similarity row label across every candidate table."""

    import numpy as np

    metric = clean_label(metric)
    if not metric:
        return None

    labels: list[str] = []
    origins: list[tuple[int, int]] = []
    for table_index, grid in enumerate(grids):
        for row_index, row in enumerate(grid[1:], start=1):
            if not row:
                continue
            label = clean_label(row[0])
            if label:
                labels.append(label)
                origins.append((table_index, row_index))
    if not labels:
        return None

    vectors = embedder.encode([metric] + labels)
    scores = vectors[1:] @ vectors[0]
    best = int(np.argmax(scores))
    if float(scores[best]) < min_score:
        return None
    table_index, row_index = origins[best]
    return Match(table_index, row_index, labels[best], float(scores[best]))
