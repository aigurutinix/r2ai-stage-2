"""Shared paper prompt builder for program and direct answer strategies."""

from __future__ import annotations

from pathlib import Path

from vifinqa.answering.base import TablePromptMetadata
from vifinqa.config.loader import resolve_asset_path


PROGRAM_SYSTEM_PROMPT = "prompts/answering/program_system.txt"
DIRECT_SYSTEM_PROMPT = "prompts/answering/direct_system.txt"
COMMON_USER_PROMPT = "prompts/answering/common_user.txt"


def _prompt(path: str | Path) -> str:
    return resolve_asset_path(path).read_text(encoding="utf-8").strip()

_NO_TABLE_NOTE = "(Không có bảng dữ liệu nào được cung cấp — trả lời dựa trên hiểu biết sẵn có nếu có thể.)"


def _format_table_metadata(metadata: TablePromptMetadata | None) -> str:
    if metadata is None:
        return ""
    return (
        "<metadata>\n"
        f"<ticker>{metadata.ticker}</ticker>\n"
        f"<year>{metadata.year}</year>\n"
        f"<doc_name>{metadata.doc_name}</doc_name>\n"
        f"<company_name>{metadata.company_name}</company_name>\n"
        f"<anchor_context>\n{metadata.anchor_context}\n</anchor_context>\n"
        "</metadata>\n"
    )


def _format_tables_xml(
    csv_texts: dict[str, str], table_metadata: dict[str, TablePromptMetadata]
) -> str:
    if not csv_texts:
        return _NO_TABLE_NOTE
    parts = [
        f'<table table_ref="{table_ref}">\n'
        f"{_format_table_metadata(table_metadata.get(table_ref))}"
        f"<csv>\n{text}\n</csv>\n</table>"
        for table_ref, text in csv_texts.items()
    ]
    return "\n\n".join(parts)


def _build_user_prompt(
    question: str,
    csv_texts: dict[str, str],
    table_metadata: dict[str, TablePromptMetadata] | None = None,
    *,
    user_template_path: str | Path = COMMON_USER_PROMPT,
) -> str:
    template = _prompt(user_template_path)
    var_hint = "df" if len(csv_texts) <= 1 else "dfs"
    return (
        template.replace("{{QUESTION}}", question)
        .replace("{{VAR_HINT}}", var_hint)
        .replace("{{TABLES}}", _format_tables_xml(csv_texts, table_metadata or {}))
    )


def build_pandas_query_prompt(
    question: str,
    csv_texts: dict[str, str],
    table_metadata: dict[str, TablePromptMetadata] | None = None,
    *,
    system_path: str | Path = PROGRAM_SYSTEM_PROMPT,
    user_template_path: str | Path = COMMON_USER_PROMPT,
) -> tuple[str, str]:
    return _prompt(system_path), _build_user_prompt(
        question, csv_texts, table_metadata, user_template_path=user_template_path
    )


def build_direct_answer_prompt(
    question: str,
    csv_texts: dict[str, str],
    table_metadata: dict[str, TablePromptMetadata] | None = None,
    *,
    system_path: str | Path = DIRECT_SYSTEM_PROMPT,
    user_template_path: str | Path = COMMON_USER_PROMPT,
) -> tuple[str, str]:
    return _prompt(system_path), _build_user_prompt(
        question, csv_texts, table_metadata, user_template_path=user_template_path
    )
