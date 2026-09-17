"""Central backend wiring; pipelines consume only the protocols in ``base.py``."""

from __future__ import annotations

from typing import Any, Mapping

from vifinqa.answering.base import AnswerStrategy
from vifinqa.answering.direct_answer import DirectAnswerStrategy
from vifinqa.answering.pandas_query import PandasQueryStrategy
from vifinqa.config.loader import env_value
from vifinqa.embeddings.base import Embedder
from vifinqa.embeddings.hf_local import HFLocalEmbedder
from vifinqa.llm.base import ChatLLM
from vifinqa.llm.hf_local import HFLocalLLM
from vifinqa.llm.openai_compatible import OpenAICompatibleLLM
from vifinqa.rerankers.base import Reranker
from vifinqa.rerankers.cross_encoder import CrossEncoderReranker
from vifinqa.rerankers.cross_encoder import QWEN3_FINANCIAL_TABLE_INSTRUCTION
from vifinqa.rerankers.noop import NoOpReranker


def build_embedder(config: Mapping[str, Any], *, hf_token: str | None = None) -> Embedder:
    if config.get("backend", "sentence_transformers") != "sentence_transformers":
        raise ValueError(f"Unsupported embedding backend: {config.get('backend')!r}")
    model_id = str(config.get("model_id", "BAAI/bge-m3"))
    return HFLocalEmbedder(
        model_id,
        device=config.get("device"),
        hf_token=hf_token,
        batch_size=config.get("batch_size"),
        max_tokens=int(config.get("max_tokens", 1024)),
    )


def build_reranker(config: Mapping[str, Any], *, hf_token: str | None = None) -> Reranker:
    if not config.get("enabled", False) or config.get("backend") in (None, "none"):
        return NoOpReranker()
    if config.get("backend", "cross_encoder") != "cross_encoder":
        raise ValueError(f"Unsupported reranker backend: {config.get('backend')!r}")
    instruction = config.get("instruction")
    if instruction == "financial_table_v1":
        instruction = QWEN3_FINANCIAL_TABLE_INSTRUCTION
    return CrossEncoderReranker(
        str(config["model_id"]),
        device=config.get("device"),
        hf_token=hf_token,
        instruction=instruction,
        max_tokens=int(config.get("max_tokens", 1024)),
    )


def build_llm(config: Mapping[str, Any], *, hf_token: str | None = None) -> ChatLLM:
    backend = config.get("backend", "openai_compatible")
    if backend == "openai_compatible":
        return OpenAICompatibleLLM(
            base_url=env_value(config, "base_url", required=True) or "",
            api_key=env_value(config, "api_key", required=True) or "",
            model=str(config["model_id"]),
            temperature=float(config.get("temperature", 0)),
            max_completion_tokens=int(config.get("max_completion_tokens", 8192)),
            max_retries=int(config.get("max_retries", 0)),
            reasoning_effort=config.get("reasoning_effort"),
        )
    if backend == "hf_transformers":
        return HFLocalLLM(
            str(config["model_id"]),
            hf_token=hf_token,
            device=config.get("device", "auto"),
            dtype=config.get("dtype", "auto"),
            max_completion_tokens=int(config.get("max_completion_tokens", 8192)),
            temperature=float(config.get("temperature", 0)),
        )
    raise ValueError(f"Unsupported LLM backend: {backend!r}")


def build_answer_strategy(config: Mapping[str, Any]) -> AnswerStrategy:
    name = config.get("strategy", "pandas_query")
    max_chars = int(config.get("table_max_chars", 20_000))
    if name == "pandas_query":
        return PandasQueryStrategy(table_max_chars=max_chars)
    if name == "direct_answer":
        return DirectAnswerStrategy(table_max_chars=max_chars)
    raise ValueError(f"Unsupported answer strategy: {name!r}")
