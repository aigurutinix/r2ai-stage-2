"""Resolve legacy pandas-query reads back to their exact evidence cells.

The cumulative ``source_audit.json`` and ``panel_source_audit.json`` manifests
cover programs rebuilt by the two deterministic builders.  Older programs can
still be valid, but historically had no machine-readable record of the exact
row/column read by ``values[0]`` or ``iloc``.  This read-only audit closes that
observability gap without changing a submission.

For every row not present in either manifest, the audit:

* identifies the evidence tables actually loaded by the query;
* verifies those tables equal ``relevant_tables``;
* safely evaluates dataframe-only filter assignments;
* resolves constant ``.values[n]`` and two-dimensional ``.iloc[r, c]`` reads
  to the original CSV row, column, label and raw value; and
* reports queries whose terminal source reads remain unresolved.

No submitted program is executed by this script.  Only a restricted subset of
dataframe indexing/filter expressions is evaluated with empty builtins.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from audit_positional_semantics import (
    _safe_dataframe_expression,
    _dataframe_aliases,
    _evaluate_dataframe_aliases,
    _resolve_derived_frame,
    _root_dataframe,
    constant_int,
)
from build_compliance_safe_candidate import evidence_tables


@dataclass(frozen=True)
class TerminalRead:
    expression: str
    frame_expression: str
    evidence_variable: str
    row: int
    column_kind: str
    column: str | int
    access_kind: str


def audited_ids(submission_dir: Path) -> set[int]:
    result: set[int] = set()
    for name in ("source_audit.json", "panel_source_audit.json"):
        path = submission_dir / name
        if not path.exists():
            continue
        result.update(
            int(row["id"])
            for row in json.loads(path.read_text(encoding="utf-8"))
            if isinstance(row, dict) and "id" in row
        )
    return result


def literal_index(node: ast.AST) -> str | int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return node.value
    return constant_int(node)


def _column_from_series(node: ast.AST) -> tuple[ast.AST, str | int] | None:
    """Return the dataframe expression and column for a simple Series read."""
    if isinstance(node, ast.Subscript):
        column = literal_index(node.slice)
        if column is not None:
            return node.value, column
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "iloc"
        and isinstance(node.slice, ast.Tuple)
        and len(node.slice.elts) == 2
        and isinstance(node.slice.elts[0], ast.Slice)
    ):
        column = literal_index(node.slice.elts[1])
        if column is not None:
            return node.value.value, column
    return None


def evaluate_dataframe_aliases(query: str, frames: dict[str, pd.DataFrame]) -> dict[str, object]:
    """Also resolve safe dataframe assignments nested under guard branches."""
    env = _evaluate_dataframe_aliases(query, frames)
    # These names are allowed only as dtype arguments in dataframe expressions
    # such as ``series.astype(str)``.  Builtins remain empty during eval.
    env.update({"str": str, "float": float, "int": int, "bool": bool})
    try:
        tree = ast.parse(query)
    except SyntaxError:
        return env
    aliases = _dataframe_aliases(tree)
    assignments = [
        node for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    for _ in range(5):
        changed = False
        for statement in assignments:
            value_node = statement.value
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            root = _root_dataframe(value_node, aliases) if value_node is not None else None
            if root is None or value_node is None or not _safe_dataframe_expression(value_node, set(env)):
                continue
            try:
                value = eval(
                    compile(ast.Expression(value_node), "<legacy-audit-expression>", "eval"),
                    {"__builtins__": {}},
                    env,
                )
            except Exception:
                continue
            if not isinstance(value, (pd.DataFrame, pd.Series)):
                continue
            for target in targets:
                if isinstance(target, ast.Name) and env.get(target.id) is not value:
                    env[target.id] = value
                    changed = True
        if not changed:
            break
    return env


def _aggregate_series(node: ast.AST, aliases: dict[str, str]) -> tuple[ast.AST, str | int, str] | None:
    """Resolve ``filtered['1']....sum()`` to its source dataframe/column."""
    if not (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"sum", "mean", "min", "max"}
    ):
        return None
    for item in ast.walk(node.func.value):
        series = _column_from_series(item)
        if series is None:
            continue
        frame_node, column = series
        if _root_dataframe(frame_node, aliases) is not None:
            return frame_node, column, node.func.attr
    return None


def terminal_reads(query: str) -> list[TerminalRead]:
    """Extract constant scalar reads that feed legacy numeric programs."""
    try:
        tree = ast.parse(query)
    except SyntaxError:
        return []
    aliases = _dataframe_aliases(tree)
    found: list[TerminalRead] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue

        # filtered['1'].values[0]
        if isinstance(node.value, ast.Attribute) and node.value.attr == "values":
            row = constant_int(node.slice)
            series = _column_from_series(node.value.value)
            if row is None or series is None:
                continue
            frame_node, column = series
            root = _root_dataframe(frame_node, aliases)
            if root is None:
                continue
            found.append(TerminalRead(
                expression=ast.unparse(node),
                frame_expression=ast.unparse(frame_node),
                evidence_variable=root,
                row=row,
                column_kind="label" if isinstance(column, str) else "position",
                column=column,
                access_kind="values",
            ))
            continue

        # filtered.iloc[0]['2']
        if (
            isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Attribute)
            and node.value.value.attr == "iloc"
        ):
            row = constant_int(node.value.slice)
            column = literal_index(node.slice)
            frame_node = node.value.value.value
            root = _root_dataframe(frame_node, aliases)
            if row is not None and column is not None and root is not None:
                found.append(TerminalRead(
                    expression=ast.unparse(node),
                    frame_expression=ast.unparse(frame_node),
                    evidence_variable=root,
                    row=row,
                    column_kind="label" if isinstance(column, str) else "position",
                    column=column,
                    access_kind="iloc-then-column",
                ))
                continue

        # filtered.iloc[0, 2]
        if not (
            isinstance(node.value, ast.Attribute)
            and node.value.attr == "iloc"
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
        ):
            continue
        row = constant_int(node.slice.elts[0])
        column = literal_index(node.slice.elts[1])
        if row is None or not isinstance(column, int):
            continue
        frame_node = node.value.value
        root = _root_dataframe(frame_node, aliases)
        if root is None:
            continue
        found.append(TerminalRead(
            expression=ast.unparse(node),
            frame_expression=ast.unparse(frame_node),
            evidence_variable=root,
            row=row,
            column_kind="position",
            column=column,
            access_kind="iloc",
        ))

    for node in ast.walk(tree):
        aggregate = _aggregate_series(node, aliases)
        if aggregate is None:
            continue
        frame_node, column, kind = aggregate
        root = _root_dataframe(frame_node, aliases)
        assert root is not None
        found.append(TerminalRead(
            expression=ast.unparse(node),
            frame_expression=ast.unparse(frame_node),
            evidence_variable=root,
            row=0,
            column_kind="label" if isinstance(column, str) else "position",
            column=column,
            access_kind=f"aggregate-{kind}",
        ))

    if not found:
        empty_aliases = {
            node.value.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr == "empty"
            and isinstance(node.value, ast.Name)
        }
        for alias in sorted(empty_aliases):
            root = aliases.get(alias)
            if root:
                found.append(TerminalRead(
                    expression=f"{alias}.empty",
                    frame_expression=alias,
                    evidence_variable=root,
                    row=0,
                    column_kind="position",
                    column=0,
                    access_kind="selector-exists",
                ))

    # AST walks can encounter the same expression through repeated assignments.
    return list(dict.fromkeys(found))


def table_ref_from_csv(path: Path) -> str | None:
    match = re.fullmatch(r"(?P<document>.+)_(?P<line>\d+)\.csv", path.name)
    if not match:
        return None
    return f"{match.group('document')}|{match.group('line')}"


def resolve_read(
    read: TerminalRead,
    env: dict[str, object],
    base_frames: dict[str, pd.DataFrame],
) -> tuple[dict | None, str | None]:
    base = base_frames.get(read.evidence_variable)
    if base is None:
        return None, "evidence dataframe unavailable"
    frame = _resolve_derived_frame(read.frame_expression, env)
    if not isinstance(frame, pd.DataFrame):
        return None, "derived dataframe expression unresolved"
    try:
        if read.access_kind == "selector-exists":
            source_rows = [int(base.index.get_loc(index)) for index in frame.index if index in base.index]
            return {
                **asdict(read),
                "matched_rows": len(source_rows),
                "source_rows": source_rows,
                "source_labels": [str(base.iloc[index, 0]) for index in source_rows],
                "source_rows_values": [base.iloc[index].astype(str).tolist() for index in source_rows],
                "raw_values": [str(base.iloc[index, 0]) for index in source_rows],
            }, None
        positional_row = read.row if read.row >= 0 else len(frame) + read.row
        if read.access_kind.startswith("aggregate-"):
            if read.column_kind == "label":
                column_label = str(read.column)
                if column_label not in frame.columns:
                    return None, "column label absent from aggregate dataframe"
                column_position = int(frame.columns.get_loc(column_label))
            else:
                column_position = int(read.column)
                if column_position < 0:
                    column_position += len(frame.columns)
                if not 0 <= column_position < len(frame.columns):
                    return None, "column outside aggregate dataframe"
                column_label = str(frame.columns[column_position])
            source_rows = [int(base.index.get_loc(index)) for index in frame.index if index in base.index]
            return {
                **asdict(read),
                "source_column": column_position,
                "source_column_label": column_label,
                "source_rows": source_rows,
                "source_labels": [str(base.iloc[index, 0]) for index in source_rows],
                "source_rows_values": [base.iloc[index].astype(str).tolist() for index in source_rows],
                "raw_values": [str(base.iloc[index, column_position]) for index in source_rows],
            }, None
        if not 0 <= positional_row < len(frame):
            return None, "row outside derived dataframe"
        source_index = frame.index[positional_row]
        if source_index not in base.index:
            return None, "derived index does not map to source dataframe"
        if read.column_kind == "label":
            column_label = str(read.column)
            if column_label not in frame.columns:
                return None, "column label absent from derived dataframe"
            column_position = int(frame.columns.get_loc(column_label))
        else:
            column_position = int(read.column)
            if column_position < 0:
                column_position += len(frame.columns)
            if not 0 <= column_position < len(frame.columns):
                return None, "column outside derived dataframe"
            column_label = str(frame.columns[column_position])
        raw = str(frame.iloc[positional_row, column_position])
        source_row = int(base.index.get_loc(source_index))
        label = str(base.iloc[source_row, 0]) if len(base.columns) else ""
    except (IndexError, KeyError, TypeError, ValueError) as error:
        return None, f"{type(error).__name__}: {error}"

    return {
        **asdict(read),
        "source_row": source_row,
        "source_column": column_position,
        "source_column_label": column_label,
        "source_label": label,
        "source_row_values": base.iloc[source_row].astype(str).tolist(),
        "raw": raw,
    }, None


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-audited", action="store_true")
    args = parser.parse_args()

    submission_dir = args.submission_dir.resolve()
    rows = json.loads((submission_dir / "submission.json").read_text(encoding="utf-8"))
    known = set() if args.include_audited else audited_ids(submission_dir)
    records: list[dict] = []
    unresolved: list[dict] = []
    provenance_mismatches: list[dict] = []

    for row in rows:
        qid = int(row["id"])
        if qid in known:
            continue
        query = str(row.get("pandas_query", ""))
        evidence = {
            str(item.get("variable")): submission_dir / str(item.get("csv_path"))
            for item in row.get("evidence", [])
            if isinstance(item, dict) and item.get("variable") and item.get("csv_path")
        }
        base_frames: dict[str, pd.DataFrame] = {}
        for variable, path in evidence.items():
            if path.exists():
                base_frames[variable] = pd.read_csv(
                    path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
                )
        env = evaluate_dataframe_aliases(query, base_frames)
        reads = terminal_reads(query)
        resolved: list[dict] = []
        row_errors: list[dict] = []
        for read in reads:
            item, reason = resolve_read(read, env, base_frames)
            if item is None:
                row_errors.append({**asdict(read), "reason": reason})
                continue
            path = evidence[read.evidence_variable]
            item["csv"] = str(path.relative_to(submission_dir))
            item["source_table"] = table_ref_from_csv(path)
            resolved.append(item)

        declared = list(row.get("relevant_tables") or [])
        actual = evidence_tables(submission_dir, row) if query else []
        if declared != actual:
            provenance_mismatches.append({
                "id": qid,
                "declared": declared,
                "actual": actual,
                "question": row.get("question"),
            })
        if row_errors or not reads:
            unresolved.append({
                "id": qid,
                "terminal_reads_found": len(reads),
                "terminal_reads_resolved": len(resolved),
                "errors": row_errors,
                "question": row.get("question"),
                "pandas_query": query,
            })
        records.append({
            "id": qid,
            "question": row.get("question"),
            "answer": row.get("answer"),
            "relevant_tables": declared,
            "actual_tables": actual,
            "terminal_reads": resolved,
        })

    payload = {
        "submission": str(submission_dir),
        "audited_ids_skipped": 0 if args.include_audited else len(known),
        "legacy_rows_checked": len(records),
        "rows_with_resolved_terminal_reads": sum(bool(row["terminal_reads"]) for row in records),
        "resolved_terminal_reads": sum(len(row["terminal_reads"]) for row in records),
        "unresolved_row_count": len(unresolved),
        "unresolved": unresolved,
        "provenance_mismatch_count": len(provenance_mismatches),
        "provenance_mismatches": provenance_mismatches,
        "records": records,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(rendered + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in payload.items() if key not in {"records", "unresolved", "provenance_mismatches"}}, ensure_ascii=False, indent=2))
    if unresolved:
        print(json.dumps({"unresolved": unresolved}, ensure_ascii=False, indent=2))
    if provenance_mismatches:
        print(json.dumps({"provenance_mismatches": provenance_mismatches}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
