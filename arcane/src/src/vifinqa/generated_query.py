"""Validate and execute model-generated Pandas answer expressions."""

from __future__ import annotations

import ast
import builtins
import math
import re
from typing import Mapping

import pandas as pd

from .answers import (
    SKIP_HEADERS,
    cell_expression,
    fold,
    operator_text,
    question_years,
    requested_scale,
    says,
    source_scale,
)


SAFE_NAMES = {"result", "pd", "float", "int", "str", "len", "abs", "min", "max", "sum", "round"}
SAFE_ATTRIBUTES = {
    "agg", "all", "any", "astype", "between", "contains", "count", "dropna",
    "eq", "fillna", "ge", "groupby", "gt", "idxmax", "idxmin", "iloc", "index",
    "isin", "item", "le", "loc", "lower", "lt", "map", "mask", "max", "mean",
    "min", "nlargest", "nsmallest", "nunique", "replace", "reset_index", "round",
    "shape", "sort_values", "str", "strip", "sum", "to_numeric", "tolist",
    "transform", "unique", "values", "where",
}
SAFE_NODES = (
    ast.Module, ast.Assign, ast.Name, ast.Load, ast.Store, ast.Constant, ast.BinOp,
    ast.UnaryOp, ast.Call, ast.Attribute, ast.Subscript, ast.Slice, ast.List, ast.Tuple,
    ast.Dict, ast.Compare, ast.BoolOp, ast.IfExp, ast.keyword, ast.Add, ast.Sub, ast.Mult,
    ast.Div, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Eq, ast.NotEq, ast.Lt, ast.LtE,
    ast.Gt, ast.GtE, ast.And, ast.Or, ast.Not, ast.BitAnd, ast.BitOr, ast.In, ast.NotIn,
)


class GeneratedQueryError(ValueError):
    """The proposed query is unsafe, invalid, or not scalar."""


def is_direct_lookup_question(question: str) -> bool:
    """Route only unambiguous one-period questions to cell selection."""
    if len(question_years(question)) > 1:
        return False
    text = operator_text(question)
    if re.search(r"(?<!\w)(tong cua|tong cong cua|cong voi|cong lai)(?!\w)", text):
        return False
    return not says(
        text,
        "tang truong", "toc do tang", "tang giam", "chenh lech", "khac biet",
        "hieu", "tru", "cao hon", "thap hon", "lon hon", "trung binh",
        "binh quan", "ty trong", "chiem bao nhieu", "gap", "bao nhieu lan",
        "bien loi nhuan", "lon nhat", "cao nhat", "nhieu nhat", "nho nhat",
        "thap nhat", "it nhat", "nam nao", "cong ty nao", "doanh nghiep nao",
        "trong so", "neu", "vuot", "tren muc", "duoi muc",
    )


def numeric_consensus(candidates: list[dict]) -> list[dict]:
    """Return the largest organizer-tolerance answer cluster."""
    groups: list[list[dict]] = []
    for candidate in candidates:
        group = next((items for items in groups if math.isclose(
            float(items[0]["answer"]), float(candidate["answer"]), rel_tol=2e-4,
            abs_tol=max(0.02, abs(float(items[0]["answer"])) * 2e-4),
        )), None)
        if group is None:
            groups.append([candidate])
        else:
            group.append(candidate)
    return max(groups, key=len) if groups else []


def validate_generated_query(query: str, variables: set[str]) -> ast.Module:
    if len(query) > 4000 or "__" in query:
        raise GeneratedQueryError("query is too long or contains a dunder name")
    try:
        tree = ast.parse(query)
    except SyntaxError as error:
        raise GeneratedQueryError("query is not valid Python") from error
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assign):
        raise GeneratedQueryError("query must contain one assignment")
    assignment = tree.body[0]
    if len(assignment.targets) != 1 or not isinstance(assignment.targets[0], ast.Name) or assignment.targets[0].id != "result":
        raise GeneratedQueryError("query must assign only to result")
    allowed_names = SAFE_NAMES | variables
    for node in ast.walk(tree):
        if not isinstance(node, SAFE_NODES):
            raise GeneratedQueryError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise GeneratedQueryError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr not in SAFE_ATTRIBUTES:
            raise GeneratedQueryError(f"disallowed attribute: {node.attr}")
        if isinstance(node, ast.keyword) and node.arg == "inplace":
            raise GeneratedQueryError("inplace operations are disallowed")
    if not any(isinstance(node, ast.Name) and node.id in variables for node in ast.walk(tree)):
        raise GeneratedQueryError("query reads no evidence variable")
    return tree


def execute_generated_query(query: str, frames: Mapping[str, pd.DataFrame]) -> float:
    tree = validate_generated_query(query, set(frames))
    namespace = {"pd": pd, **frames}
    safe_builtins = {name: getattr(builtins, name) for name in SAFE_NAMES - {"result", "pd"}}
    local: dict = {}
    exec(compile(tree, "<generated-query>", "exec"), {"__builtins__": safe_builtins, **namespace}, local)
    try:
        result = float(local["result"])
    except (KeyError, TypeError, ValueError) as error:
        raise GeneratedQueryError("result is not a scalar number") from error
    if not math.isfinite(result):
        raise GeneratedQueryError("result is not finite")
    return result


def compile_lookup_query(
    question: str,
    variable: str,
    frame: pd.DataFrame,
    row: int,
    column: int,
    context: str = "",
) -> str:
    """Compile one model-selected cell into the canonical numeric expression."""
    if row < 0 or column < 0 or row >= len(frame.index) or column >= len(frame.columns):
        raise GeneratedQueryError("selected cell is outside the evidence frame")
    if any(marker in fold(str(frame.columns[column])) for marker in SKIP_HEADERS):
        raise GeneratedQueryError("selected cell is a code or note column")
    cell = str(frame.iloc[row, column])
    structured_context = " ".join(
        line for line in context.splitlines()
        if fold(line).startswith(("don vi", "cac chi tieu", "duong dan cot", "duong dan dong"))
    )
    metadata = " ".join((structured_context, str(frame.columns[column]), *map(str, frame.iloc[:2, column].tolist())))
    input_scale = source_scale(metadata, cell)
    multiplier = input_scale / requested_scale(question)
    grouped_comma = (
        input_scale > 1 and "." not in cell and cell.count(",") == 1
        and len(cell.rstrip("%) ").rsplit(",", 1)[-1]) == 3
    )
    if grouped_comma:
        value = f"str({variable}.iloc[{row}, {column}])"
        expression = f"float({value}.replace('(', '-').replace(')', '').replace('%', '').replace(',', ''))"
    else:
        expression = cell_expression(variable, row, column).removeprefix("result = ")
    return f"result = round(({expression} * {multiplier}), 2)"
