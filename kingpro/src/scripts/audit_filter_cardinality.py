"""Find legacy row filters that match zero or multiple source-table rows.

The submission programs commonly filter a label and then take ``values[0]``.
This read-only audit replays literal ``str.contains`` predicates against the
exact CSV named by that row's evidence manifest.  Predicates joined by ``&`` in
the same DataFrame filter are evaluated as one conjunction; this prevents a
false alarm when each predicate is broad but their intersection is unique.
Cardinality other than one is a triage signal only; every candidate still
requires source/question review.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import pandas as pd


GENERIC_LABELS = {
    "tong cong",
    "tong",
    "cong",
    "tai san",
    "cong no",
    "vay dai han",
    "chi phi lai vay",
    "doanh thu",
}


def audited_ids(submission_dir: Path) -> set[int]:
    result: set[int] = set()
    for name in ("source_audit.json", "panel_source_audit.json"):
        path = submission_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        result.update(int(row["id"]) for row in payload if "id" in row)
    return result


def literal(node: ast.AST) -> object | None:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def evidence_name(node: ast.AST, available: set[str]) -> str | None:
    names = [item.id for item in ast.walk(node) if isinstance(item, ast.Name)]
    return next((name for name in names if name in available), None)


def series_column(node: ast.AST, df_name: str) -> tuple[str, object] | None:
    """Return (kind, column) for df['x'] or df.iloc[:, n] inside wrappers."""
    for item in ast.walk(node):
        if not isinstance(item, ast.Subscript):
            continue
        if isinstance(item.value, ast.Name) and item.value.id == df_name:
            column = literal(item.slice)
            if isinstance(column, (str, int)):
                return "label", column
        if (
            isinstance(item.value, ast.Attribute)
            and item.value.attr == "iloc"
            and isinstance(item.value.value, ast.Name)
            and item.value.value.id == df_name
            and isinstance(item.slice, ast.Tuple)
            and len(item.slice.elts) == 2
        ):
            column = literal(item.slice.elts[1])
            if isinstance(column, int):
                return "position", column
    return None


def normalize(text: object) -> str:
    return " ".join(str(text).casefold().split())


def contains_calls(node: ast.AST) -> list[ast.Call]:
    """Return literal ``.contains(...)`` calls below *node*."""

    return [
        item
        for item in ast.walk(node)
        if (
            isinstance(item, ast.Call)
            and isinstance(item.func, ast.Attribute)
            and item.func.attr == "contains"
            and item.args
            and isinstance(literal(item.args[0]), str)
        )
    ]


def is_and_only_filter(node: ast.AST) -> bool:
    """Whether a filter expression combines predicates with ``&`` but not ``|``."""

    operators = [
        item.op
        for item in ast.walk(node)
        if isinstance(item, ast.BinOp)
    ]
    return any(isinstance(operator, ast.BitAnd) for operator in operators) and not any(
        isinstance(operator, ast.BitOr) for operator in operators
    )


def evaluate_contains(
    call: ast.Call,
    *,
    evidence: dict[str, Path],
    frames: dict[str, pd.DataFrame],
) -> dict | None:
    """Resolve one literal contains call to its exact frame mask."""

    series_expr = call.func.value
    df_name = evidence_name(series_expr, set(evidence))
    column_spec = series_column(series_expr, df_name) if df_name else None
    if not df_name or not column_spec or not evidence[df_name].exists():
        return None
    if df_name not in frames:
        frames[df_name] = pd.read_csv(evidence[df_name], dtype=str, keep_default_na=False)
    frame = frames[df_name]
    kind, column = column_spec
    try:
        series = frame.iloc[:, int(column)] if kind == "position" else frame[str(column)]
    except (IndexError, KeyError, ValueError):
        return None

    kwargs = {item.arg: literal(item.value) for item in call.keywords if item.arg}
    pattern = str(literal(call.args[0]))
    case = bool(kwargs.get("case", True))
    regex = bool(kwargs.get("regex", True))
    na = bool(kwargs.get("na", False))
    try:
        mask = series.astype(str).str.contains(pattern, case=case, regex=regex, na=na)
    except Exception:
        return None
    return {
        "call": call,
        "df_name": df_name,
        "frame": frame,
        "mask": mask,
        "kind": kind,
        "column": column,
        "pattern": pattern,
        "generic": normalize(pattern) in GENERIC_LABELS,
    }


def conjunctive_evaluations(
    tree: ast.AST,
    *,
    evidence: dict[str, Path],
    frames: dict[str, pd.DataFrame],
) -> list[dict]:
    """Evaluate ``df[predicate_a & predicate_b]`` contains intersections."""

    groups: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript) or not is_and_only_filter(node.slice):
            continue
        calls = contains_calls(node.slice)
        if len(calls) < 2:
            continue
        resolved = [
            evaluate_contains(call, evidence=evidence, frames=frames)
            for call in calls
        ]
        if any(item is None for item in resolved):
            continue
        items = [item for item in resolved if item is not None]
        if len({item["df_name"] for item in items}) != 1:
            continue
        call_ids = tuple(sorted(id(item["call"]) for item in items))
        if call_ids in seen:
            continue
        seen.add(call_ids)
        mask = items[0]["mask"].copy()
        for item in items[1:]:
            mask &= item["mask"]
        groups.append({
            "items": items,
            "mask": mask,
            "frame": items[0]["frame"],
            "df_name": items[0]["df_name"],
            "call_ids": set(call_ids),
        })
    return groups


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-audited", action="store_true")
    args = parser.parse_args()

    rows = json.loads((args.submission_dir / "submission.json").read_text(encoding="utf-8"))
    known = audited_ids(args.submission_dir)
    findings: list[dict] = []
    checked_filters = 0
    checked_conjunctions = 0
    predicates_suppressed_by_unique_conjunction = 0
    unresolved_filters = 0

    for row in rows:
        qid = int(row["id"])
        if qid in known and not args.include_audited:
            continue
        evidence = {
            str(item["variable"]): args.submission_dir / str(item["csv_path"])
            for item in row.get("evidence", [])
            if item.get("variable") and item.get("csv_path")
        }
        if not evidence:
            continue
        try:
            tree = ast.parse(str(row.get("pandas_query", "")))
        except SyntaxError:
            continue
        frames: dict[str, pd.DataFrame] = {}

        resolved_cache: dict[int, dict | None] = {}
        for call in contains_calls(tree):
            resolved_cache[id(call)] = evaluate_contains(
                call,
                evidence=evidence,
                frames=frames,
            )

        grouped_call_ids: set[int] = set()
        for group in conjunctive_evaluations(tree, evidence=evidence, frames=frames):
            checked_conjunctions += 1
            grouped_call_ids.update(group["call_ids"])
            items = group["items"]
            mask = group["mask"]
            frame = group["frame"]
            count = int(mask.sum())
            checked_filters += len(items)
            if count == 1:
                predicates_suppressed_by_unique_conjunction += len(items)
                continue
            patterns = [item["pattern"] for item in items]
            generic = any(item["generic"] for item in items)
            matched = frame.loc[mask]
            findings.append({
                "id": qid,
                "risk": (10 if count == 0 else 5) + (2 if generic else 0),
                "scope": "conjunction",
                "patterns": patterns,
                "pattern": " & ".join(patterns),
                "generic": generic,
                "variable": group["df_name"],
                "csv": str(evidence[group["df_name"]]),
                "columns": [
                    {"kind": item["kind"], "column": item["column"]}
                    for item in items
                ],
                "match_count": count,
                "matched_labels": matched.iloc[:12, 0].astype(str).tolist(),
                "question": row.get("question"),
                "answer": row.get("answer"),
            })

        for call in contains_calls(tree):
            if id(call) in grouped_call_ids:
                continue
            resolved = resolved_cache.get(id(call))
            if resolved is None:
                unresolved_filters += 1
                continue
            df_name = resolved["df_name"]
            frame = resolved["frame"]
            mask = resolved["mask"]
            kind = resolved["kind"]
            column = resolved["column"]
            pattern = resolved["pattern"]
            checked_filters += 1
            matched = frame.loc[mask]
            count = int(mask.sum())
            generic = resolved["generic"]
            if count == 1 and not generic:
                continue
            findings.append({
                "id": qid,
                "risk": (10 if count == 0 else 5 if count > 1 else 0) + (2 if generic else 0),
                "scope": "single_predicate",
                "pattern": pattern,
                "generic": generic,
                "variable": df_name,
                "csv": str(row.get("evidence", [])[list(evidence).index(df_name)].get("csv_path", ""))
                if df_name in list(evidence) else str(evidence[df_name]),
                "column_kind": kind,
                "column": column,
                "match_count": count,
                "matched_labels": matched.iloc[:12, 0].astype(str).tolist(),
                "question": row.get("question"),
                "answer": row.get("answer"),
            })

    findings.sort(key=lambda item: (-item["risk"], -item["match_count"], item["id"], item["pattern"]))
    payload = {
        "submission": str(args.submission_dir),
        "audited_ids_skipped": 0 if args.include_audited else len(known),
        "checked_filters": checked_filters,
        "checked_conjunctions": checked_conjunctions,
        "predicates_suppressed_by_unique_conjunction": predicates_suppressed_by_unique_conjunction,
        "unresolved_filters": unresolved_filters,
        "finding_count": len(findings),
        "findings": findings,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
