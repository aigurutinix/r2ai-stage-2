"""Minimal interface for pandas-query and direct-answer strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from vifinqa.llm.base import ChatLLM


@dataclass(frozen=True, slots=True)
class TablePromptMetadata:
    table_ref: str
    ticker: str
    year: str
    doc_name: str
    company_name: str
    anchor_context: str


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    question: str
    csv_paths: dict[str, Path]  # table_ref -> path; empty without context, nonempty with tables.
    table_metadata: dict[str, TablePromptMetadata] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AnswerResult:
    actual: object  # Returned value, or None on failure; numeric coercion happens in evaluation.
    raw_output: str  # Raw LLM response retained for debugging.
    error: str = ""  # Empty when no error occurred.


class AnswerStrategy(Protocol):
    name: str

    def answer(self, req: AnswerRequest, llm: ChatLLM) -> AnswerResult: ...
