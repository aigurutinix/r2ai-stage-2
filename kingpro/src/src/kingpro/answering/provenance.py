"""Bind generated Pandas programs to the real tables used by the answer.

The competition artifact cites table references. The live product needs the
same invariant: a citation may only leave the service when the generated code
actually references the corresponding DataFrame from the retrieval trace.
"""

from __future__ import annotations

import ast
import re

_DF_NAME = re.compile(r"^df(\d+)$")


def dataframe_dependencies(code: str, n_tables: int) -> list[int]:
    """Return sorted 1-based table indexes referenced by generated code.

    ``df`` means the only table; ``df3`` means table three. Direct access to
    ``dfs`` is conservatively treated as depending on every supplied table
    unless a literal key such as ``dfs['df2']`` identifies one table.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []

    used: set[int] = set()
    direct_dfs = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if node.id == "df" and n_tables == 1:
                used.add(1)
            match = _DF_NAME.match(node.id)
            if match:
                idx = int(match.group(1))
                if 1 <= idx <= n_tables:
                    used.add(idx)
            if node.id == "dfs" and not isinstance(getattr(node, "ctx", None), ast.Store):
                direct_dfs = True
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "dfs":
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                match = _DF_NAME.match(key.value)
                if match:
                    idx = int(match.group(1))
                    if 1 <= idx <= n_tables:
                        used.add(idx)
                    direct_dfs = False
    if direct_dfs:
        used.update(range(1, n_tables + 1))
    return sorted(used)


def bind_evidence(code: str, tables: list[dict]) -> list[dict]:
    """Create evidence only for real retrieved tables read by ``code``."""
    indexes = dataframe_dependencies(code, len(tables))
    return [
        {
            "variable": f"df{idx}",
            "csv_path": tables[idx - 1]["csv_path"],
            "table_ref": tables[idx - 1]["table_ref"],
        }
        for idx in indexes
    ]

