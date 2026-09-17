
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_TRUNCATED_NOTE = "\n...[TABLE CONTENT TRUNCATED IN PROMPT]"


def read_csv_contexts(csv_paths: dict[str, Path], *, max_chars: int) -> dict[str, str]:
    if max_chars <= len(_TRUNCATED_NOTE):
        raise ValueError(f"table_max_chars must be greater than {len(_TRUNCATED_NOTE)}")

    contexts: dict[str, str] = {}
    for ref, path in csv_paths.items():
        text = path.read_text(encoding="utf-8-sig")
        if len(text) > max_chars:
            logger.warning(
                "Table %s is %d characters long; truncating it to %d characters for the prompt",
                ref,
                len(text),
                max_chars,
            )
            text = text[: max_chars - len(_TRUNCATED_NOTE)] + _TRUNCATED_NOTE
        contexts[ref] = text
    return contexts
