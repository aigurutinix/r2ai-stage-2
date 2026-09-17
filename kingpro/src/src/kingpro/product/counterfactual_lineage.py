"""Causal source-cell discovery for legacy Pandas programs.

Some older verified programs read full table CSVs and predate the compact
``q<ID>_source_cells.csv`` contract.  This analyzer perturbs numeric cells,
replays the exact program, and records a cell only when a valid counterfactual
changes the numeric result.  It then binds the evidence frame back to exactly
one ``relevant_tables`` CSV by full-frame equality.

The output is a product-side audit sidecar.  It never edits a competition
submission and never treats the stored answer value as a search key.
"""

from __future__ import annotations

import ast
import builtins
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from kingpro.answering.sandbox import validate_code
from kingpro.evaluation.metrics import coerce_number


_ACCOUNTING_RE = re.compile(r"^\(?-?\d+(?:[.,]\d+)*\)?%?$")
_VARIANTS = ("9999999999999999", "(9999999999999999)", "0")
_TEXT_VARIANT = "__KINGPRO_COUNTERFACTUAL__"
_WHITELIST = (
    "abs round len min max sum sorted float int str bool list dict set "
    "range enumerate zip all any isinstance"
).split()
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _WHITELIST}


def _clean(value: object) -> str:
    text = str(value).replace("\ufeff", "").replace("\u00a0", " ").strip()
    return "" if text.casefold() == "nan" else " ".join(text.split())


def _is_numeric_token(value: object) -> bool:
    return bool(_ACCOUNTING_RE.fullmatch(_clean(value).replace(" ", "")))


def _same_frame(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if left.shape != right.shape or list(map(str, left.columns)) != list(map(str, right.columns)):
        return False
    for row in range(len(left)):
        for column in range(len(left.columns)):
            if _clean(left.iloc[row, column]) != _clean(right.iloc[row, column]):
                return False
    return True


def _different(left: float, right: float) -> bool:
    return not math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8)


class CounterfactualLineageAnalyzer:
    """Discover result-sensitive cells while preserving grader semantics."""

    def __init__(self, root: str | Path, candidate: str | Path) -> None:
        self.root = Path(root).resolve()
        self.candidate = Path(candidate).resolve()
        self._catalog: dict[str, dict] | None = None

    def _load_catalog(self) -> dict[str, dict]:
        if self._catalog is None:
            catalog: dict[str, dict] = {}
            with (self.root / "build" / "catalog.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        catalog[str(row["table_ref"])] = row
            self._catalog = catalog
        return self._catalog

    @staticmethod
    def _read(path: Path) -> pd.DataFrame:
        return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")

    def _frames(self, row: dict[str, Any]) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
        paths = {
            str(item["variable"]): (self.candidate / str(item["csv_path"])).resolve()
            for item in row.get("evidence", [])
            if item.get("variable") and item.get("csv_path")
        }
        if not paths or any(path != self.candidate and self.candidate not in path.parents for path in paths.values()):
            raise ValueError("evidence path escapes candidate or evidence is empty")
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError("one or more evidence CSVs are missing")
        return {variable: self._read(path) for variable, path in paths.items()}, paths

    def _physical_frame(self, table_ref: str) -> tuple[Path, pd.DataFrame] | None:
        catalog = self._load_catalog().get(table_ref)
        if not catalog or not catalog.get("csv_path"):
            return None
        table_root = (self.root / "build" / "tables").resolve()
        path = (table_root / str(catalog["csv_path"])).resolve()
        if path != table_root and table_root not in path.parents:
            return None
        if not path.is_file():
            return None
        return path, self._read(path)

    def _bind_frames(
        self, frames: dict[str, pd.DataFrame], relevant_tables: list[str]
    ) -> dict[str, tuple[str, Path]]:
        physical: dict[str, tuple[Path, pd.DataFrame]] = {}
        for table_ref in dict.fromkeys(map(str, relevant_tables)):
            resolved = self._physical_frame(table_ref)
            if resolved is not None:
                physical[table_ref] = resolved
        bindings: dict[str, tuple[str, Path]] = {}
        for variable, frame in frames.items():
            matches = [
                (table_ref, path)
                for table_ref, (path, source) in physical.items()
                if _same_frame(frame, source)
            ]
            if len(matches) == 1:
                bindings[variable] = matches[0]
        return bindings

    @staticmethod
    def _execute_namespace(
        code: str, frames: dict[str, pd.DataFrame]
    ) -> tuple[float | None, dict[str, Any]]:
        namespace: dict[str, Any] = {
            "pd": pd,
            "dfs": frames,
            "__builtins__": _SAFE_BUILTINS,
        }
        if len(frames) == 1:
            namespace["df"] = next(iter(frames.values()))
        try:
            exec(compile(code, "<counterfactual-lineage>", "exec"), namespace)  # noqa: S102
        except Exception:
            return None, namespace
        value = coerce_number(namespace.get("result"))
        return (value if value is not None and math.isfinite(value) else None), namespace

    @classmethod
    def _execute(cls, code: str, frames: dict[str, pd.DataFrame]) -> float | None:
        return cls._execute_namespace(code, frames)[0]

    @staticmethod
    def _mutated(
        frames: dict[str, pd.DataFrame], variable: str, cells: list[tuple[int, int]], token: str
    ) -> dict[str, pd.DataFrame]:
        copied = dict(frames)
        copied[variable] = frames[variable].copy(deep=True)
        for row, column in cells:
            copied[variable].iat[row, column] = token
        return copied

    @staticmethod
    def _referenced_frames(code: str, variables: set[str]) -> set[str]:
        """Backward-slice simple assignments from ``result`` to df variables.

        Legacy programs commonly alias ``df3 = list(dfs.values())[2]`` and use
        only that frame.  The slice is an optimization only: if syntax or a
        dynamic construct is unclear, every evidence frame remains eligible.
        """

        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set(variables)
        dependencies: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            value = getattr(node, "value", None)
            if value is None:
                continue
            loaded = {
                item.id
                for item in ast.walk(value)
                if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load)
            }
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for item in ast.walk(target):
                    if isinstance(item, ast.Name):
                        dependencies.setdefault(item.id, set()).update(loaded)
        pending = ["result"]
        reached: set[str] = set()
        while pending:
            name = pending.pop()
            if name in reached:
                continue
            reached.add(name)
            pending.extend(dependencies.get(name, ()))
        direct = reached & variables
        return direct or set(variables)

    @staticmethod
    def _candidate_rows(
        frame: pd.DataFrame,
        namespace: dict[str, Any],
        source_frames: dict[str, pd.DataFrame],
    ) -> tuple[list[int], bool]:
        """Use exact filtered-frame remnants as a safe row prefilter."""

        candidates: set[int] = set()
        source_ids = {id(value) for value in source_frames.values()}
        for value in namespace.values():
            if (
                not isinstance(value, pd.DataFrame)
                or id(value) in source_ids
                or value.empty
            ):
                continue
            if len(value) >= len(frame) or list(map(str, value.columns)) != list(map(str, frame.columns)):
                continue
            for index in value.index:
                try:
                    row_index = int(index)
                except (TypeError, ValueError):
                    continue
                if row_index < 0 or row_index >= len(frame):
                    continue
                if all(
                    _clean(value.loc[index, column]) == _clean(frame.loc[index, column])
                    for column in frame.columns
                ):
                    candidates.add(row_index)
        return (
            (sorted(candidates), True)
            if candidates
            else (list(range(len(frame))), False)
        )

    def analyze(
        self,
        row: dict[str, Any],
        *,
        max_rows_per_frame: int = 500,
        max_numeric_cells: int = 20_000,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        code = str(row.get("pandas_query", ""))
        valid, error = validate_code(code)
        if not valid:
            return {"status": "skipped", "reason": f"unsafe_program:{error}", "cells": []}
        try:
            frames, _paths = self._frames(row)
        except Exception as exc:
            return {"status": "skipped", "reason": f"evidence:{type(exc).__name__}", "cells": []}
        baseline, baseline_namespace = self._execute_namespace(code, frames)
        expected = coerce_number(row.get("answer"))
        if baseline is None or expected is None or _different(baseline, expected):
            return {"status": "skipped", "reason": "baseline_replay_mismatch", "cells": []}
        bindings = self._bind_frames(frames, list(row.get("relevant_tables", [])))
        if not bindings:
            return {"status": "skipped", "reason": "no_unique_physical_binding", "cells": []}
        active_variables = self._referenced_frames(code, set(frames))

        numeric_count = sum(
            _is_numeric_token(frame.iat[row_index, column_index])
            for frame in frames.values()
            for row_index in range(min(len(frame), max_rows_per_frame))
            for column_index in range(len(frame.columns))
        )
        if numeric_count > max_numeric_cells:
            return {"status": "skipped", "reason": "numeric_cell_budget", "cells": []}

        dependencies: list[dict[str, Any]] = []
        executions = 1
        for variable, frame in frames.items():
            if (
                variable not in bindings
                or variable not in active_variables
                or len(frame) > max_rows_per_frame
            ):
                continue
            table_ref, physical_path = bindings[variable]
            candidate_rows, narrowed = self._candidate_rows(
                frame, baseline_namespace, frames
            )
            for row_index in candidate_rows:
                candidates = [
                    (row_index, column_index)
                    for column_index in range(len(frame.columns))
                    if _is_numeric_token(frame.iat[row_index, column_index])
                ]
                row_sensitive = False
                if candidates:
                    for token in _VARIANTS:
                        result = self._execute(
                            code, self._mutated(frames, variable, candidates, token)
                        )
                        executions += 1
                        if result is not None and _different(result, baseline):
                            row_sensitive = True
                            break
                if row_sensitive or narrowed:
                    for _row, column_index in candidates:
                        changed: list[dict[str, Any]] = []
                        for token in _VARIANTS:
                            result = self._execute(
                                code,
                                self._mutated(frames, variable, [(row_index, column_index)], token),
                            )
                            executions += 1
                            if result is not None and _different(result, baseline):
                                changed.append({"token": token, "result": result})
                        if changed:
                            dependencies.append(
                                {
                                    "question_id": int(row["id"]),
                                    "variable": variable,
                                    "table_ref": table_ref,
                                    "source_path": physical_path.relative_to(self.root).as_posix(),
                                    "row_idx": row_index,
                                    "col_idx": column_index,
                                    "raw": _clean(frame.iat[row_index, column_index]),
                                    "counterfactuals": changed,
                                    "verified": True,
                                    "verification": "counterfactual_result_dependency",
                                }
                            )

                # Count/existence programs can depend on a row label without
                # consuming any numeric value.  Accept a text cell only when a
                # valid label mutation changes the numeric result.
                for column_index in range(len(frame.columns)):
                    raw = _clean(frame.iat[row_index, column_index])
                    if not raw or _is_numeric_token(raw) or not any(char.isalpha() for char in raw):
                        continue
                    result = self._execute(
                        code,
                        self._mutated(
                            frames,
                            variable,
                            [(row_index, column_index)],
                            _TEXT_VARIANT,
                        ),
                    )
                    executions += 1
                    if result is None or not _different(result, baseline):
                        continue
                    dependencies.append(
                        {
                            "question_id": int(row["id"]),
                            "variable": variable,
                            "table_ref": table_ref,
                            "source_path": physical_path.relative_to(self.root).as_posix(),
                            "row_idx": row_index,
                            "col_idx": column_index,
                            "raw": raw,
                            "counterfactuals": [
                                {"token": _TEXT_VARIANT, "result": result}
                            ],
                            "verified": True,
                            "verification": "counterfactual_result_dependency",
                        }
                    )
        return {
            "status": "verified" if dependencies else "unresolved",
            "reason": "" if dependencies else "no_result_sensitive_numeric_cell",
            "question_id": int(row["id"]),
            "baseline": baseline,
            "cells": dependencies,
            "executions": executions,
            "active_variables": sorted(active_variables),
            "elapsed_ms": round((time.perf_counter() - started) * 1000),
        }
