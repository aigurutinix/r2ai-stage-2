"""Canonical table reference shared by generation, retrieval, and evaluation."""

from __future__ import annotations


def make_table_ref(doc_name: str, table_id: int) -> str:
    if not doc_name or "|table_" in doc_name or table_id < 0:
        raise ValueError("invalid table reference components")
    return f"{doc_name}|table_{table_id}"


def parse_table_ref(table_ref: str) -> tuple[str, int]:
    doc_name, separator, raw_id = table_ref.rpartition("|table_")
    if not separator or not doc_name:
        raise ValueError(f"table_ref must have format 'doc_name|table_N': {table_ref!r}")
    try:
        table_id = int(raw_id)
    except ValueError as exc:
        raise ValueError(f"table_ref has a non-integer table id: {table_ref!r}") from exc
    if table_id < 0:
        raise ValueError(f"table_ref has a negative table id: {table_ref!r}")
    return doc_name, table_id
