"""Execute an LLM-generated pandas query and verify that it matches the answer.

Conventions:
- For one table, the code receives ``df = pd.read_csv(csv_path)``.
- For two or more tables, it receives ``dfs = {table_ref: pd.read_csv(path)}``.
- CSVs always contain the original strings. The generated code must parse localized
  numbers such as ``1.234.567`` and ``(1.234)`` when necessary.
- The code must assign its final value to ``result``.
"""

from __future__ import annotations

import builtins
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

import pandas as pd

_ALLOWED_BUILTIN_NAMES = (
    "abs", "round", "len", "min", "max", "sum", "sorted", "float", "int",
    "str", "bool", "list", "dict", "set", "range", "enumerate", "zip",
    "all", "any",
)
_ALLOWED_BUILTINS = {name: getattr(builtins, name) for name in _ALLOWED_BUILTIN_NAMES}
_ALLOWED_BUILTINS.update({"True": True, "False": False, "None": None})


class PandasCheckError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class CheckResult:
    ok: bool
    actual: object
    detail: str = ""
    accessed_refs: frozenset[str] = frozenset()


ValueT = TypeVar("ValueT")


class TrackingDict(dict[str, ValueT]):
    """Mapping that records which tables the LLM-generated code actually reads."""

    def __init__(self, values: dict[str, ValueT]) -> None:
        super().__init__(values)
        self.accessed_refs: set[str] = set()

    def __getitem__(self, key: str) -> ValueT:
        value = super().__getitem__(key)
        self.accessed_refs.add(key)
        return value

    def get(self, key: str, default: ValueT | None = None) -> ValueT | None:
        if key in self:
            self.accessed_refs.add(key)
        return super().get(key, default)

    def items(self):  # type: ignore[no-untyped-def]
        self.accessed_refs.update(self.keys())
        return super().items()

    def values(self):  # type: ignore[no-untyped-def]
        self.accessed_refs.update(self.keys())
        return super().values()

    def __iter__(self):
        self.accessed_refs.update(self.keys())
        return super().__iter__()


def _run_pandas_code_tracked(code: str, csv_path: Path | dict[str, Path]) -> tuple[object, frozenset[str]]:
    tracked_dfs: TrackingDict[pd.DataFrame] | None = None
    if isinstance(csv_path, dict):
        tracked_dfs = TrackingDict({ref: pd.read_csv(p) for ref, p in csv_path.items()})
        namespace: dict = {"pd": pd, "dfs": tracked_dfs, "__builtins__": _ALLOWED_BUILTINS}
    else:
        namespace = {"pd": pd, "df": pd.read_csv(csv_path), "__builtins__": _ALLOWED_BUILTINS}
    try:
        exec(compile(code, "<pandas_query>", "exec"), namespace)  # noqa: S102
    except Exception as exc:
        raise PandasCheckError(f"Failed to execute pandas_query: {exc}") from exc
    if "result" not in namespace:
        raise PandasCheckError("pandas_query did not assign the `result` variable")
    accessed = frozenset(tracked_dfs.accessed_refs) if tracked_dfs is not None else frozenset()
    return namespace["result"], accessed


def run_pandas_code(code: str, csv_path: Path | dict[str, Path]) -> object:
    actual, _accessed = _run_pandas_code_tracked(code, csv_path)
    return actual


def _to_native(value: object) -> object:
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            return value
    return value


def values_match(expected: object, actual: object, *, abs_tol: float = 1e-2) -> bool:
    expected = _to_native(expected)
    actual = _to_native(actual)

    if isinstance(expected, bool) or isinstance(actual, bool):
        return bool(expected) == bool(actual)

    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return math.isclose(float(expected), float(actual), rel_tol=0.0, abs_tol=abs_tol)

    return str(expected).strip().casefold() == str(actual).strip().casefold()


def check_answer(code: str, csv_path: Path | dict[str, Path], expected_answer: object) -> CheckResult:
    execution = execute_query(code, csv_path)
    if not execution.ok:
        return execution

    ok = values_match(expected_answer, execution.actual)
    detail = "" if ok else f"expected={expected_answer!r} actual={execution.actual!r}"
    return CheckResult(
        ok=ok,
        actual=execution.actual,
        detail=detail,
        accessed_refs=execution.accessed_refs,
    )


def execute_query(code: str, csv_path: Path | dict[str, Path]) -> CheckResult:
    """Run the query on the original CSV and return ``result`` as the canonical value."""
    try:
        actual, accessed_refs = _run_pandas_code_tracked(code, csv_path)
    except PandasCheckError as exc:
        return CheckResult(ok=False, actual=None, detail=str(exc))
    return CheckResult(ok=True, actual=_to_native(actual), accessed_refs=accessed_refs)
