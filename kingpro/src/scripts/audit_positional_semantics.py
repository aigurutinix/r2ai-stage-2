"""Rank unaudited fixed-``iloc`` reads whose source row mismatches the question.

Legacy programs sometimes read a hard positional cell such as ``df1.iloc[11,
1]``.  Existing row-filter audits cannot inspect those programs.  This
read-only audit resolves each constant coordinate against its exact evidence
CSV and ranks low lexical overlap for manual source review.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd


STOPWORDS = {
    "bao", "bao nhieu", "bang", "cac", "cua", "cho", "cuoi", "dau",
    "den", "dong", "duoc", "gia", "giua", "hay", "la", "nam", "ngay",
    "phan", "tai", "theo", "thi", "trieu", "tren", "trong", "ty", "va",
    "vao", "vnd", "voi",
}


def fold(text: object) -> str:
    value = unicodedata.normalize("NFD", str(text).casefold().replace("đ", "d"))
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def tokens(text: object) -> set[str]:
    return {
        token for token in fold(text).split()
        if len(token) > 2 and token not in STOPWORDS and not token.isdigit()
    }


def audited_ids(root: Path) -> set[int]:
    result: set[int] = set()
    for name in ("source_audit.json", "panel_source_audit.json"):
        path = root / name
        if not path.exists():
            continue
        result.update(
            int(row["id"])
            for row in json.loads(path.read_text(encoding="utf-8"))
            if isinstance(row, dict) and "id" in row
        )
    return result


def constant_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -int(node.operand.value)
    return None


def _root_dataframe(node: ast.AST, aliases: dict[str, str]) -> str | None:
    """Resolve a filtered/copied expression back to its original df variable."""
    if isinstance(node, ast.Name):
        name = aliases.get(node.id, node.id)
        return name if re.fullmatch(r"df\d+", name) else None
    if isinstance(node, ast.Subscript):
        return _root_dataframe(node.value, aliases)
    if isinstance(node, ast.Attribute):
        return _root_dataframe(node.value, aliases)
    if isinstance(node, ast.Call):
        return _root_dataframe(node.func, aliases)
    return None


def _dataframe_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    # Statements are visited in source order so chains such as
    # ``subset = filtered.copy()`` resolve after ``filtered = df1[...]``.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        root = _root_dataframe(value, aliases) if value is not None else None
        if root is None:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                aliases[target.id] = root
    return aliases


_ALLOWED_PANDAS_METHODS = {
    "astype", "between", "contains", "copy", "dropna", "eq", "fillna",
    "ge", "gt", "isin", "le", "lt", "ne", "reset_index", "sort_values",
    "strip",
}


def _safe_dataframe_expression(node: ast.AST, names: set[str]) -> bool:
    """Allow only indexing/boolean expressions over already loaded frames."""
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id not in names:
            return False
        if isinstance(child, ast.Attribute) and child.attr.startswith("_"):
            return False
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Attribute) or child.func.attr not in _ALLOWED_PANDAS_METHODS:
                return False
        if isinstance(
            child,
            (
                ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
                ast.GeneratorExp, ast.Await, ast.Yield, ast.NamedExpr,
            ),
        ):
            return False
    return True


def _evaluate_dataframe_aliases(query: str, frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Evaluate safe dataframe-only assignments so alias ``iloc`` is inspected accurately."""
    try:
        tree = ast.parse(query)
    except SyntaxError:
        return dict(frames)
    env: dict[str, object] = dict(frames)
    aliases: dict[str, str] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        value_node = statement.value
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        root = _root_dataframe(value_node, aliases) if value_node is not None else None
        if root is None or value_node is None or not _safe_dataframe_expression(value_node, set(env)):
            continue
        try:
            value = eval(compile(ast.Expression(value_node), "<audit-expression>", "eval"), {"__builtins__": {}}, env)
        except Exception:
            continue
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                env[target.id] = value
                aliases[target.id] = root
    return env


def _resolve_derived_frame(expression: str, env: dict[str, object]) -> object | None:
    if expression in env:
        return env[expression]
    try:
        node = ast.parse(expression, mode="eval").body
    except SyntaxError:
        return None
    if not _safe_dataframe_expression(node, set(env)):
        return None
    try:
        value = eval(compile(ast.Expression(node), "<audit-expression>", "eval"), {"__builtins__": {}}, env)
    except Exception:
        return None
    return value if isinstance(value, (pd.DataFrame, pd.Series)) else None


def positional_reads(query: str) -> list[tuple[str, str, int, int]]:
    try:
        tree = ast.parse(query)
    except SyntaxError:
        return []
    aliases = _dataframe_aliases(tree)
    found: list[tuple[str, str, int, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        value = node.value
        if not (
            isinstance(value, ast.Attribute)
            and value.attr == "iloc"
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
        ):
            continue
        alias = value.value.id if isinstance(value.value, ast.Name) else ast.unparse(value.value)
        root = _root_dataframe(value.value, aliases)
        if root is None:
            continue
        row = constant_int(node.slice.elts[0])
        column = constant_int(node.slice.elts[1])
        if row is not None and column is not None:
            found.append((alias, root, row, column))
    return list(dict.fromkeys(found))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--include-audited", action="store_true")
    args = parser.parse_args()

    rows = json.loads((args.submission_dir / "submission.json").read_text(encoding="utf-8"))
    known = set() if args.include_audited else audited_ids(args.submission_dir)
    findings: list[dict] = []
    checked_coordinates = 0
    unresolved_coordinates = 0
    unresolved: list[dict] = []

    for item in rows:
        qid = int(item["id"])
        if qid in known:
            continue
        evidence = {
            str(entry.get("variable")): args.submission_dir / str(entry.get("csv_path"))
            for entry in item.get("evidence", [])
            if entry.get("variable") and entry.get("csv_path")
        }
        base_frames: dict[str, pd.DataFrame] = {}
        for variable, path in evidence.items():
            if path.exists():
                base_frames[variable] = pd.read_csv(
                    path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
                )
        derived_frames = _evaluate_dataframe_aliases(
            str(item.get("pandas_query", "")), base_frames
        )
        question_tokens = tokens(item.get("question", ""))
        for variable, evidence_variable, row_index, column_index in positional_reads(str(item.get("pandas_query", ""))):
            path = evidence.get(evidence_variable)
            if path is None or not path.exists():
                unresolved_coordinates += 1
                unresolved.append({
                    "id": qid, "variable": variable, "evidence_variable": evidence_variable, "row": row_index,
                    "column": column_index, "reason": "evidence variable/path not found",
                })
                continue
            derived = _resolve_derived_frame(variable, derived_frames)
            frame = derived if derived is not None else base_frames[evidence_variable]
            try:
                label = str(frame.iloc[row_index, 0])
                raw = str(frame.iloc[row_index, column_index])
            except IndexError:
                unresolved_coordinates += 1
                unresolved.append({
                    "id": qid, "variable": variable, "row": row_index,
                    "column": column_index, "reason": "coordinate outside evidence table",
                    "csv": str(path.relative_to(args.submission_dir)),
                })
                continue
            checked_coordinates += 1
            label_tokens = tokens(label)
            shared = question_tokens & label_tokens
            overlap = len(shared) / max(1, len(label_tokens))
            findings.append({
                "id": qid,
                "score": round(overlap, 4),
                "shared": sorted(shared),
                "variable": variable,
                "evidence_variable": evidence_variable,
                "csv": str(path.relative_to(args.submission_dir)),
                "row": row_index,
                "column": column_index,
                "label": label,
                "raw": raw,
                "answer": item.get("answer"),
                "question": item.get("question"),
            })

    findings.sort(key=lambda row: (row["score"], row["id"], row["row"], row["column"]))
    payload = {
        "submission": str(args.submission_dir),
        "audited_ids_skipped": 0 if args.include_audited else len(known),
        "checked_coordinates": checked_coordinates,
        "unresolved_coordinates": unresolved_coordinates,
        "unresolved": unresolved,
        "finding_count": len(findings),
        "findings": findings[: args.limit],
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
