"""Structural and full execution validation for a packaged submission ZIP."""

from __future__ import annotations

import argparse
import ast
import csv
import io
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from .experiments import SubmissionArchive


SAFE_NAMES = {"pd", "float", "int", "str", "abs", "round"}
SAFE_ATTRIBUTES = {
    "Series",
    "abs",
    "astype",
    "fillna",
    "idxmax",
    "idxmin",
    "iloc",
    "index",
    "max",
    "mean",
    "median",
    "min",
    "nlargest",
    "replace",
    "round",
    "strip",
    "str",
    "sum",
    "to_numeric",
    "where",
}
SAFE_NODE_TYPES = {
    ast.Expression,
    ast.Constant,
    ast.Call,
    ast.Attribute,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Pow,
    ast.Subscript,
    ast.Slice,
    ast.Tuple,
    ast.List,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.BoolOp,
    ast.And,
    ast.BitAnd,
    ast.UnaryOp,
    ast.USub,
    ast.keyword,
}
FRAME_RE = re.compile(r"^df\d+$")
RESERVED_DERIVED_COLUMNS = {
    "answer", "answers", "result", "results", "precomputed",
    "dap an", "ket qua",
}


def _is_literal_zero(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and float(node.value) == 0.0
    )


def validate_query_ast(query: str) -> ast.Expression:
    if "__" in query:
        raise ValueError("dunder access is forbidden")
    tree = ast.parse(query, mode="eval")
    frame_names = {
        node.id for node in ast.walk(tree)
        if isinstance(node, ast.Name) and FRAME_RE.fullmatch(node.id)
    }
    if not frame_names:
        raise ValueError("query must read at least one evidence dataframe")
    for node in ast.walk(tree):
        if type(node) not in SAFE_NODE_TYPES:
            raise ValueError(f"unsupported AST node {type(node).__name__}")
        if isinstance(node, ast.Name) and not (
            node.id in SAFE_NAMES or FRAME_RE.fullmatch(node.id)
        ):
            raise ValueError(f"unsupported name {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_ATTRIBUTES:
            raise ValueError(f"unsupported attribute {node.attr!r}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and (
            _is_literal_zero(node.left) or _is_literal_zero(node.right)
        ):
            raise ValueError("multiplication by literal zero breaks source data flow")
    return tree


def _normalized_column(value: object) -> str:
    return re.sub(r"[\s_]+", " ", str(value).strip().casefold())


def _padded_grid(grid: list[list[Any]]) -> list[list[str]]:
    width = max((len(row) for row in grid), default=0)
    return [
        [str(value) for value in row] + [""] * (width - len(row))
        for row in grid
    ]


def validate_source_provenance(
    archive: SubmissionArchive, database: Path
) -> dict[str, int]:
    """Prove every evidence CSV is an unchanged BTC source-table grid."""

    connection = sqlite3.connect(str(database))
    table_cache: dict[str, list[list[str]] | None] = {}
    bindings = 0
    try:
        for item in archive.items:
            source_grids = []
            for evidence_key in item["relevant_tables"]:
                if evidence_key not in table_cache:
                    row = connection.execute(
                        "SELECT grid_json FROM tables WHERE evidence_key = ?",
                        (evidence_key,),
                    ).fetchone()
                    table_cache[evidence_key] = (
                        _padded_grid(json.loads(row[0])) if row else None
                    )
                grid = table_cache[evidence_key]
                if grid is not None:
                    source_grids.append(grid)
            for evidence in item["evidence"]:
                csv_rows = list(csv.reader(io.StringIO(
                    archive.members[evidence["csv_path"]].decode("utf-8-sig")
                )))
                if not csv_rows:
                    raise ValueError(
                        f"q{item['id']}: empty evidence {evidence['csv_path']}"
                    )
                header, csv_grid = csv_rows[0], csv_rows[1:]
                expected_header = [f"col_{index}" for index in range(len(header))]
                if header != expected_header:
                    raise ValueError(
                        f"q{item['id']}: non-source schema in {evidence['csv_path']}"
                    )
                if any(
                    _normalized_column(column) in RESERVED_DERIVED_COLUMNS
                    for column in header
                ):
                    raise ValueError(
                        f"q{item['id']}: derived answer column in {evidence['csv_path']}"
                    )
                if csv_grid not in source_grids:
                    raise ValueError(
                        f"q{item['id']}: evidence is not an exact cited BTC table: "
                        f"{evidence['csv_path']}"
                    )
                bindings += 1
    finally:
        connection.close()
    return {"provenance_bindings": bindings, "source_tables": len(table_cache)}


def replay_archive(path: Path, database: Optional[Path] = None) -> dict[str, Any]:
    archive = SubmissionArchive.load(path)
    referenced = {
        evidence["csv_path"]
        for item in archive.items
        for evidence in item["evidence"]
    }
    if set(archive.members) != {"submission.json", *referenced}:
        raise ValueError("ZIP contains missing or unreferenced files")

    frame_cache: dict[str, pd.DataFrame] = {}
    errors = []
    replayed = 0
    for item in archive.items:
        question_id = int(item["id"])
        try:
            validate_query_ast(item["pandas_query"])
            frames = {}
            for evidence in item["evidence"]:
                csv_path = evidence["csv_path"]
                if csv_path not in frame_cache:
                    frame_cache[csv_path] = pd.read_csv(
                        io.BytesIO(archive.members[csv_path])
                    )
                frames[evidence["variable"]] = frame_cache[csv_path]
            actual = eval(  # noqa: S307 - AST and namespaces are allowlisted above
                item["pandas_query"],
                {
                    "__builtins__": {},
                    "pd": pd,
                    "float": float,
                    "int": int,
                    "str": str,
                    "abs": abs,
                    "round": round,
                },
                frames,
            )
            actual_value = float(actual)
            expected = float(item["answer"])
            if not math.isfinite(actual_value) or not math.isclose(
                actual_value, expected, rel_tol=1e-9, abs_tol=1e-7
            ):
                raise ValueError(f"answer mismatch: query={actual_value}, answer={expected}")
            replayed += 1
        except Exception as error:  # report all question-level failures together
            errors.append({"question_id": question_id, "error": str(error)[:500]})

    result = {
        "valid": not errors,
        "items": len(archive.items),
        "evidence_files": len(referenced),
        "replayed": replayed,
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    if database is not None:
        result.update(validate_source_provenance(archive, database))
    result["dataflow_gate"] = "PASS"
    result["zero_weight_gate"] = "PASS"
    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument(
        "--database", type=Path, default=Path("artifacts/vifinqa.db"),
        help="BTC warehouse used to prove every evidence CSV has source provenance",
    )
    args = parser.parse_args(argv)
    print(json.dumps(replay_archive(args.zip, args.database), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
