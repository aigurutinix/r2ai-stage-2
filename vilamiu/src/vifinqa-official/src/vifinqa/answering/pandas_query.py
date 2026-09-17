
from __future__ import annotations

import logging
from pathlib import Path

from vifinqa.answering.base import AnswerRequest, AnswerResult
from vifinqa.answering.prompts import (
    COMMON_USER_PROMPT,
    PROGRAM_SYSTEM_PROMPT,
    build_pandas_query_prompt,
)
from vifinqa.answering.sandbox import SandboxError, run_pandas_code
from vifinqa.answering.table_context import read_csv_contexts
from vifinqa.llm.base import ChatLLM

logger = logging.getLogger(__name__)


def _sanitize_generated_code(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    lines = [line for line in text.splitlines() if line.strip() != "import pandas as pd"]
    return "\n".join(lines).strip()


class PandasQueryStrategy:
    name = "pandas_query"

    def __init__(
        self,
        *,
        table_max_chars: int,
        system_path: str | Path = PROGRAM_SYSTEM_PROMPT,
        user_template_path: str | Path = COMMON_USER_PROMPT,
    ) -> None:
        self._table_max_chars = table_max_chars
        self._system_path = system_path
        self._user_template_path = user_template_path

    def answer(self, req: AnswerRequest, llm: ChatLLM) -> AnswerResult:
        csv_texts = read_csv_contexts(req.csv_paths, max_chars=self._table_max_chars)
        system, user = build_pandas_query_prompt(
            req.question,
            csv_texts,
            req.table_metadata,
            system_path=self._system_path,
            user_template_path=self._user_template_path,
        )
        raw = llm.complete(system=system, user=user)
        code = _sanitize_generated_code(raw)

        try:
            actual = run_pandas_code(code, dict(req.csv_paths))
        except SandboxError as exc:
            logger.warning("pandas_query strategy failed in the sandbox: %s", exc)
            return AnswerResult(actual=None, raw_output=raw, error=str(exc))
        return AnswerResult(actual=actual, raw_output=raw)
