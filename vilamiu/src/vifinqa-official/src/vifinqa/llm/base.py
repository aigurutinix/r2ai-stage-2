"""Minimal interface for interchangeable chat LLM implementations."""

from __future__ import annotations

from typing import Protocol


class LLMError(RuntimeError):
    """Classified LLM error that tells the runner whether the run can continue."""

    def __init__(self, message: str, *, fatal: bool = False) -> None:
        super().__init__(message)
        self.fatal = fatal


class ChatLLM(Protocol):
    def complete(self, *, system: str, user: str) -> str: ...
