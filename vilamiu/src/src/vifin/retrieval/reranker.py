"""Cross-encoder reranking of BM25 table candidates.

BM25 counts shared tokens; it cannot tell that "Lãi tiền gửi" belongs in the
financial-income note rather than the cash-flow statement, where the same words
appear as an adjustment. Three attempts to fix the resulting wrong-table choices
by post-processing the answer all failed (see IMPLEMENTATION.md), which is what
finally pointed at the input rather than the output.

A cross-encoder reads the question against the table together, so it can make
that distinction. Reranking only the BM25 shortlist keeps this cheap: ~30 pairs
per question rather than an index over all 146,246 tables, which fits an 8 GB
laptop GPU in minutes.

`BAAI/bge-reranker-v2-m3` — 568M parameters, released March 2024 — satisfies the
contest limits (open weights, <= 14B, published before 2026-06-01).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

MODEL_ID = "BAAI/bge-reranker-v2-m3"

# The reranker truncates at 512 tokens, so the table has to be summarised rather
# than dumped. Caption plus first-column labels is what identifies a statement.
MAX_LABELS = 40
MAX_CHARS = 1400


def table_text(row) -> str:
    """A compact description of one table for the cross-encoder."""

    grid = json.loads(row.rows_json)
    parts = [str(row.caption)]
    if str(row.unit_line):
        parts.append(str(row.unit_line))
    if grid:
        parts.append(" | ".join(str(cell) for cell in grid[0]))
        labels = [str(line[0]) for line in grid[1:] if line and str(line[0]).strip()]
        parts.append(" ; ".join(labels[:MAX_LABELS]))
    return " \n".join(p for p in parts if p)[:MAX_CHARS]


@dataclass(slots=True)
class Reranker:
    model: object
    batch_size: int = 32

    @classmethod
    def load(cls, model_id: str = MODEL_ID, device: str | None = None, batch_size: int = 32):
        from sentence_transformers import CrossEncoder

        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        model = CrossEncoder(
            model_id,
            max_length=512,
            device=device,
            model_kwargs={"torch_dtype": torch.float16} if device == "cuda" else {},
        )
        return cls(model=model, batch_size=batch_size)

    def score(self, question: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[question, text] for text in texts]
        scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
        return [float(s) for s in scores]
