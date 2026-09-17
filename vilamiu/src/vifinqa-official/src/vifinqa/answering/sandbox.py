
from __future__ import annotations

import builtins
from pathlib import Path

import pandas as pd


class SandboxError(Exception):
    pass


def _to_native(value: object) -> object:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def _read_raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
        index_col=None,
    )


def run_pandas_code(code: str, csv_paths: dict[str, Path]) -> object:
    execution_builtins = vars(builtins).copy()
    dfs = {ref: _read_raw_csv(p) for ref, p in csv_paths.items()}
    namespace: dict = {
        "pd": pd,
        "dfs": dfs,
        "__builtins__": execution_builtins,
    }
    if len(dfs) == 1:
        namespace["df"] = next(iter(dfs.values()))

    try:
        exec(compile(code, "<pandas_query>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        raise SandboxError(f"Failed to execute pandas code: {exc}") from exc
    if "result" not in namespace:
        raise SandboxError("Code did not assign the `result` variable")
    return _to_native(namespace["result"])
