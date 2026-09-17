
from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

from vifinqa.constants import EMBEDDING_MAX_SEQ_LENGTH
from vifinqa.models.device import pick_device, pick_dtype


class HFLocalEmbedder:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        hf_token: str | None = None,
        batch_size: int | None = None,
        max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        self._device = device or pick_device()
        self._batch_size = batch_size
        self._model = SentenceTransformer(
            model_name,
            device=self._device,
            token=hf_token,
            model_kwargs={"torch_dtype": pick_dtype(self._device)},
        )
        # Preserve the default behavior for compatibility.
        # even for a small model. This cap does not apply to short queries.
        self._model.max_seq_length = max_tokens
        self._use_query_prompt = "qwen3-embedding" in model_name.lower()

    def embed(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        # Preserve the default behavior for compatibility.
        # INFO/DEBUG (not guaranteed in notebook environments) — force it for large batches
        kwargs: dict = {
            "normalize_embeddings": True,
            "show_progress_bar": len(texts) > 1,
        }
        if self._batch_size is not None:
            kwargs["batch_size"] = self._batch_size
        if is_query and self._use_query_prompt:
            kwargs["prompt_name"] = "query"
        return np.asarray(self._model.encode(texts, **kwargs))
