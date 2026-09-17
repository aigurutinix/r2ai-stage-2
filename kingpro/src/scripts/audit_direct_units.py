"""Audit explicit currency-unit conversions for direct one-cell programs.

The legacy submission contains many simple programs of the form ``_r =
df1[...]`` followed by ``_v = _r['1'].values[0]``.  This read-only audit
independently reads the selected source cell, detects a literal unit in the
CSV header, converts it to the unit requested by the question, and reports
power-of-ten disagreements.  Ambiguous tables without a literal source unit
are deliberately skipped.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
import sys
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path


def fold(value: object) -> str:
    text = str(value).casefold().replace("đ", "d")
    text = "".join(
        character
        for character in unicodedata.normalize("NFKD", text)
        if not unicodedata.combining(character)
    )
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def requested_currency_unit(question: str) -> int | None:
    text = fold(question)
    patterns = (
        (r"\bnghin ty (?:dong|vnd)\b", 1_000_000_000_000),
        (r"\btram ty (?:dong|vnd)\b", 100_000_000_000),
        (r"\bty (?:dong|vnd)\b", 1_000_000_000),
        (r"\btrieu (?:dong|vnd)\b", 1_000_000),
        (r"\bnghin (?:dong|vnd)\b", 1_000),
        (r"\b(?:dong|vnd)\b", 1),
    )
    # Questions may contain a threshold in one unit and request the answer in
    # another (for example "vượt 10.000 tỷ ... bao nhiêu triệu đồng").  The
    # requested unit is conventionally the final explicit unit phrase.
    matches: list[tuple[int, int, int]] = []
    for pattern, unit in patterns:
        for match in re.finditer(pattern, text):
            matches.append((match.end(), len(match.group(0)), unit))
    return max(matches)[2] if matches else None


def source_unit(descriptor: str) -> int | None:
    text = fold(descriptor)
    if re.search(r"\btrieu (?:dong|vnd)\b", text):
        return 1_000_000
    if re.search(r"\bnghin (?:dong|vnd)\b", text):
        return 1_000
    if re.search(r"\bty (?:dong|vnd)\b", text):
        return 1_000_000_000
    # Bare ``VND`` is not enough: banking tables commonly print ``VND`` in
    # the extracted header while the surrounding (non-tabular) note declares
    # that every value is already in million VND.  Treating it as raw dong
    # would manufacture a 1e6 false positive.
    return None


def parse_number(value: object) -> float | None:
    match = re.match(r"\s*(\(?-?\d[\d.,]*\)?)", str(value))
    if not match:
        return None
    token = match.group(1)
    negative = token.startswith("(") or token.startswith("-")
    token = token.replace("(", "").replace(")", "").replace("-", "")
    if "," in token and "." in token:
        token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        tail = token.rsplit(",", 1)[-1]
        token = token.replace(",", ".") if len(tail) <= 2 else token.replace(",", "")
    elif "." in token:
        parts = token.split(".")
        token = "".join(parts) if all(len(part) == 3 for part in parts[1:]) else token
    try:
        number = float(token)
    except ValueError:
        return None
    return -number if negative else number


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


def root_dataframe(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name) and re.fullmatch(r"df\d+", node.id):
        return node.id
    for child in ast.iter_child_nodes(node):
        root = root_dataframe(child)
        if root is not None:
            return root
    return None


@dataclass(frozen=True)
class DirectRead:
    dataframe: str
    alias: str
    pattern: str
    regex: bool
    column: str


def direct_reads(query: str) -> list[DirectRead]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(query)
    except SyntaxError:
        return []
    filters: dict[str, tuple[str, str, bool]] = {}
    for statement in tree.body:
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if not isinstance(target, ast.Name):
            continue
        contains = next(
            (
                node
                for node in ast.walk(statement.value)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "contains"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ),
            None,
        )
        root = root_dataframe(statement.value)
        if contains is None or root is None:
            continue
        regex = True
        for keyword in contains.keywords:
            if keyword.arg == "regex" and isinstance(keyword.value, ast.Constant):
                regex = bool(keyword.value.value)
        filters[target.id] = (root, str(contains.args[0].value), regex)

    reads: list[DirectRead] = []
    for alias, (root, pattern, regex) in filters.items():
        columns: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
                continue
            if node.value.id != alias or not isinstance(node.slice, ast.Constant):
                continue
            if isinstance(node.slice.value, (str, int)):
                column = str(node.slice.value)
                if column != "0":
                    columns.add(column)
        if len(columns) == 1:
            reads.append(DirectRead(root, alias, pattern, regex, next(iter(columns))))
    return reads


def load_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def selected_cell(
    rows: list[dict[str, str]], read: DirectRead,
) -> tuple[int, str] | None:
    for index, row in enumerate(rows):
        label = str(row.get("0", ""))
        try:
            matched = bool(re.search(read.pattern, label, flags=re.IGNORECASE)) if read.regex else read.pattern.casefold() in label.casefold()
        except re.error:
            matched = read.pattern.casefold() in label.casefold()
        if matched:
            return index, str(row.get(read.column, ""))
    return None


def header_descriptor(rows: list[dict[str, str]], column: str, selected_row: int) -> str:
    # Unit strings normally live in one of the first four extracted header
    # rows.  Do not include the selected data row: company/metric names can
    # contain unrelated unit-like words.
    header_rows = rows[: min(4, selected_row)]
    return " ".join(str(row.get(column, "")) for row in header_rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-audited", action="store_true")
    args = parser.parse_args()

    submission = json.loads((args.submission_dir / "submission.json").read_text(encoding="utf-8"))
    known = set() if args.include_audited else audited_ids(args.submission_dir)
    checked = 0
    findings: list[dict[str, object]] = []
    for item in submission:
        qid = int(item["id"])
        if qid in known:
            continue
        requested = requested_currency_unit(str(item.get("question", "")))
        if requested is None:
            continue
        evidence = {
            str(entry.get("variable")): args.submission_dir / str(entry.get("csv_path"))
            for entry in item.get("evidence", [])
            if entry.get("variable") and entry.get("csv_path")
        }
        for read in direct_reads(str(item.get("pandas_query", ""))):
            path = evidence.get(read.dataframe)
            if path is None or not path.exists():
                continue
            _, rows = load_csv(path)
            selected = selected_cell(rows, read)
            if selected is None:
                continue
            row_index, raw = selected
            source = source_unit(header_descriptor(rows, read.column, row_index))
            value = parse_number(raw)
            if source is None or value is None:
                continue
            checked += 1
            expected = round(value * source / requested, 2)
            try:
                answer = float(item.get("answer"))
            except (TypeError, ValueError):
                continue
            if expected == 0 or answer == 0:
                continue
            ratio = abs(answer / expected)
            exponent = round(math.log10(ratio))
            if exponent == 0 or not math.isclose(ratio, 10.0 ** exponent, rel_tol=0.02):
                continue
            findings.append({
                "id": qid,
                "question": item.get("question"),
                "answer": item.get("answer"),
                "expected_direct_value": expected,
                "factor": ratio,
                "source_unit": source,
                "requested_unit": requested,
                "variable": read.dataframe,
                "column": read.column,
                "pattern": read.pattern,
                "raw": raw,
                "csv": str(path),
            })

    payload = {
        "submission": str(args.submission_dir),
        "audited_ids_skipped": 0 if args.include_audited else len(known),
        "checked_direct_reads": checked,
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
