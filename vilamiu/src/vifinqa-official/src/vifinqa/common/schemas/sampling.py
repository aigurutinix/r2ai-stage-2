"""Select reproducible question samples balanced across source files."""

from __future__ import annotations

import random
from collections import defaultdict

from vifinqa.common.schemas.schema import Question


def sample_from_sources(
    questions: list[Question], *, source_files: list[str], size: int, seed: int
) -> list[Question]:
    """Select ``size`` questions while including at least one from every source."""
    if size <= 0:
        raise ValueError("sample_size must be greater than 0")
    if not source_files:
        raise ValueError("sample_size requires at least one source file")
    if len(source_files) != len(set(source_files)):
        raise ValueError("source_files must not contain duplicates")
    if size < len(source_files):
        raise ValueError("sample_size must be at least the number of source files")

    by_source: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        by_source[question.source_file].append(question)
    missing = [source_file for source_file in source_files if not by_source[source_file]]
    if missing:
        raise ValueError(f"Source files not found: {', '.join(missing)}")

    requested_sources = set(source_files)
    candidates = [question for question in questions if question.source_file in requested_sources]
    if len(candidates) < size:
        raise ValueError(f"Pool has only {len(candidates)} questions, fewer than sample_size={size}")

    rng = random.Random(seed)
    selected = [rng.choice(by_source[source_file]) for source_file in source_files]
    selected_ids = {question.id for question in selected}
    remaining = [question for question in candidates if question.id not in selected_ids]
    selected.extend(rng.sample(remaining, size - len(selected)))
    return sorted(selected, key=lambda question: question.id)


def sample_per_source(
    questions: list[Question], *, source_files: list[str], per_file: int, seed: int
) -> list[Question]:
    """Select ``per_file`` questions from each source with a fixed RNG seed.

    Results retain global ID order for stable runners and reports.
    """
    if per_file <= 0:
        raise ValueError("sample_per_file must be greater than 0")
    if not source_files:
        raise ValueError("sample_per_file requires at least one source file")
    if len(source_files) != len(set(source_files)):
        raise ValueError("source_files must not contain duplicates")

    by_source: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        by_source[question.source_file].append(question)

    rng = random.Random(seed)
    selected: list[Question] = []
    for source_file in source_files:
        candidates = by_source.get(source_file, [])
        if len(candidates) < per_file:
            raise ValueError(
                f"source_file={source_file!r} has only {len(candidates)} questions, "
                f"fewer than sample_per_file={per_file}"
            )
        selected.extend(rng.sample(candidates, per_file))

    return sorted(selected, key=lambda question: question.id)


__all__ = ["sample_from_sources", "sample_per_source"]
