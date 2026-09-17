"""Direct-answer strategy: give CSV text to the LLM and return a value without executing code."""

from __future__ import annotations

from pathlib import Path

from vifinqa.answering.base import AnswerRequest, AnswerResult
from vifinqa.answering.prompts import (
    COMMON_USER_PROMPT,
    DIRECT_SYSTEM_PROMPT,
    build_direct_answer_prompt,
)
from vifinqa.answering.table_context import read_csv_contexts
from vifinqa.llm.base import ChatLLM


class DirectAnswerStrategy:
    name = "direct_answer"

    def __init__(
        self,
        *,
        table_max_chars: int,
        system_path: str | Path = DIRECT_SYSTEM_PROMPT,
        user_template_path: str | Path = COMMON_USER_PROMPT,
    ) -> None:
        self._table_max_chars = table_max_chars
        self._system_path = system_path
        self._user_template_path = user_template_path

    def answer(self, req: AnswerRequest, llm: ChatLLM) -> AnswerResult:
        csv_texts = read_csv_contexts(req.csv_paths, max_chars=self._table_max_chars)
        system, user = build_direct_answer_prompt(
            req.question,
            csv_texts,
            req.table_metadata,
            system_path=self._system_path,
            user_template_path=self._user_template_path,
        )
        raw = llm.complete(system=system, user=user)
        return AnswerResult(actual=raw.strip(), raw_output=raw)
