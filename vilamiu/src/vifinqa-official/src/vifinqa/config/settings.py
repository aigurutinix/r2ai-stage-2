
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_url: str = ""
    openai_api_key: str = ""
    openai_model: str = ""

    openrouter_url: str = "https://openrouter.ai/api/v1"
    openrouter_api_key: str = ""
    openrouter_model: str = "deepseek/deepseek-v4-pro"
    openrouter_reasoning_effort: str = "high"

    hf_token: str | None = None
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    table_max_chars: int = 20_000

    data_root: Path = Path("data/ocr_filter")
    company_meta_path: Path = Path("data/file_filter.csv")
    questions_dir: Path = Path("data/questions")
    cache_dir: Path = Path(".cache")
    runs_dir: Path = Path("runs")
    hard_manual_root: Path = Path("data/generated/hard_manual")


@lru_cache
def get_settings() -> Settings:
    return Settings()
