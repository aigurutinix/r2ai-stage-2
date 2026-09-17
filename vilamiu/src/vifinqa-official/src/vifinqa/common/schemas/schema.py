
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vifinqa.common.schemas.table_ref import parse_table_ref


INTERNAL_TO_PUBLIC_DIFFICULTY = {
    "easy": "easy",
    "medium": "medium",
    "intermediate": "intermediate",
    "hard": "hard",
}
PUBLIC_DIFFICULTIES = frozenset(INTERNAL_TO_PUBLIC_DIFFICULTY.values())


@dataclass(frozen=True, slots=True)
class Question:
    id: int
    question: str
    answer: object
    relevant_tables: tuple[str, ...]  # gold retrieval labels ("doc_name|table_N")
    csv_paths: tuple[Path, ...]
    pandas_query: str
    difficulty: str
    source_file: str
    source_id: int

    def __post_init__(self) -> None:
        if self.id < 1 or self.source_id < 1:
            raise ValueError("question ids must be positive")
        if not self.question.strip():
            raise ValueError("question must not be empty")
        if (
            self.difficulty not in INTERNAL_TO_PUBLIC_DIFFICULTY
            and self.difficulty not in PUBLIC_DIFFICULTIES
        ):
            raise ValueError(f"unknown released difficulty: {self.difficulty!r}")
        for table_ref in self.relevant_tables:
            parse_table_ref(table_ref)

    @property
    def original(self) -> str:
        return f"{self.source_id}_{self.source_file}"

    @property
    def relevant_docs(self) -> tuple[str, ...]:
        """Gold report ids, deduplicated in first-seen order.

        The released ``relevant_docs`` field is exactly the set of document names carried by
        ``relevant_tables``, so it is derived rather than stored; the two can never desync.
        """

        return tuple(dict.fromkeys(parse_table_ref(ref)[0] for ref in self.relevant_tables))

    @property
    def public_difficulty(self) -> str:
        """Paper-facing name; ``difficulty`` retains the released dataset value."""

        return INTERNAL_TO_PUBLIC_DIFFICULTY.get(self.difficulty, self.difficulty)
