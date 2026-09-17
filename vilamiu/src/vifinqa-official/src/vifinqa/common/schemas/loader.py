
from __future__ import annotations

import json
import logging
from pathlib import Path

from vifinqa.common.schemas.schema import Question

logger = logging.getLogger(__name__)


def discover_question_files(directory: Path) -> list[Path]:
    return sorted(directory.glob("*.jsonl"))


def _normalize_csv_paths(raw: str | list[str]) -> tuple[Path, ...]:
    if isinstance(raw, str):
        return (Path(raw),)
    return tuple(Path(p) for p in raw)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON at %s:%d: %s", path, line_no, exc)
    return rows


def load_question_files(paths: list[Path]) -> list[Question]:
    questions: list[Question] = []
    next_id = 1
    for path in sorted(paths):
        source_file = path.stem
        for row in _read_jsonl(path):
            relevant_tables = tuple(row["relevant_tables"])
            csv_paths = _normalize_csv_paths(row["csv_path"])
            if len(csv_paths) != len(relevant_tables):
                logger.warning(
                    "%s id=%s: csv_path count (%d) differs from relevant_tables count (%d)",
                    source_file,
                    row.get("id"),
                    len(csv_paths),
                    len(relevant_tables),
                )
            questions.append(
                Question(
                    id=next_id,
                    question=row["question"],
                    answer=row["answer"],
                    relevant_tables=relevant_tables,
                    csv_paths=csv_paths,
                    pandas_query=row["pandas_query"],
                    difficulty=row["difficulty"],
                    source_file=source_file,
                    source_id=row["id"],
                )
            )
            next_id += 1
    return questions


def load_questions_from_dir(directory: Path) -> list[Question]:
    return load_question_files(discover_question_files(directory))
