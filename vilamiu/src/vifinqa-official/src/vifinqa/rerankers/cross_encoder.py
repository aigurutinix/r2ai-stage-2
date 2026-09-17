
from __future__ import annotations

import torch
from sentence_transformers import CrossEncoder

from vifinqa.constants import EMBEDDING_MAX_SEQ_LENGTH
from vifinqa.models.device import pick_device, pick_dtype


QWEN3_FINANCIAL_TABLE_INSTRUCTION = (
    "Given a Vietnamese financial question, determine whether the candidate financial table contains any "
    "evidence needed to answer the question, including one component of a multi-table calculation. Pay close "
    "attention to the company, fiscal year, report scope (consolidated or parent company), accounting line "
    "items, units, and requested periods. Rank tables containing required evidence above merely topically "
    "similar tables."
)


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str,
        *,
        device: str | None = None,
        hf_token: str | None = None,
        instruction: str | None = None,
        max_tokens: int = EMBEDDING_MAX_SEQ_LENGTH,
    ) -> None:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be greater than 0")
        self._device = device or pick_device()
        # Preserve the default behavior for compatibility.
        # (751 questions x 500 pairs = 375500 forward passes) prohibitively slow. Apply the same
        model_kwargs: dict = {
            "device": self._device,
            "token": hf_token,
            "model_kwargs": {"torch_dtype": pick_dtype(self._device)},
            "max_length": max_tokens,
        }
        if instruction is not None:
            model_kwargs["prompts"] = {"financial_table": instruction}
            model_kwargs["default_prompt_name"] = "financial_table"
        self._model = CrossEncoder(model_name, **model_kwargs)
        self._use_sigmoid = "qwen3-reranker" in model_name.lower()

    def rerank(self, query: str, docs: list[str]) -> list[float]:
        if not docs:
            return []
        pairs = [(query, doc) for doc in docs]
        kwargs: dict = {"show_progress_bar": False}
        if self._use_sigmoid:
            kwargs["activation_fn"] = torch.nn.Sigmoid()
        scores = self._model.predict(pairs, **kwargs)
        return [float(s) for s in scores]
