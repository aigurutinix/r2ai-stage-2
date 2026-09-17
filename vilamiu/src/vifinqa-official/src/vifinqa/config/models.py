"""Typed experiment configuration shared by all public commands."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Difficulty = Literal["easy", "medium", "intermediate", "hard"]
DIFFICULTY_TO_INTERNAL: dict[str, str] = {
    "easy": "easy",
    "medium": "medium",
    "intermediate": "intermediate",
    "hard": "hard",
}


class Section(BaseModel):
    """A permissive typed section: stable common fields plus experimental extensions."""

    model_config = ConfigDict(extra="allow")

    backend: str | None = None
    model_id: str | None = None
    adapter: str | None = None
    enabled: bool | None = None


class ExperimentConfig(BaseModel):
    """Top-level YAML contract.

    Experimental sections intentionally allow extra keys so modes ported from the
    research repositories remain configurable without a closed model enum.
    """

    model_config = ConfigDict(extra="allow")

    command: str
    run: dict[str, Any] = Field(default_factory=dict)
    paths: dict[str, Any] = Field(default_factory=dict)
    generation: Section | None = None
    retrieval: Section | None = None
    embedding: Section | None = None
    reranker: Section | None = None
    answering: Section | None = None
    llm: Section | None = None
    prompt: Section | None = None
    evaluation: Section | None = None

    @model_validator(mode="after")
    def reject_invalid_answering_mode(self) -> "ExperimentConfig":
        if self.answering is None:
            return self
        context = getattr(self.answering, "context", None)
        strategy = getattr(self.answering, "strategy", None)
        if context == "none" and strategy == "pandas_query":
            raise ValueError("context=none is incompatible with strategy=pandas_query")
        return self


def external_to_internal_difficulty(value: str) -> str:
    try:
        return DIFFICULTY_TO_INTERNAL[value]
    except KeyError as exc:
        choices = ", ".join(DIFFICULTY_TO_INTERNAL)
        raise ValueError(f"Unknown public difficulty {value!r}; choose one of: {choices}") from exc
