"""Flag legacy queries that read an opening/prior column for an ending-period question.

The audit follows simple assignment aliases (``_r = df1[...]`` then
``_v = _r['2']``) back to the evidence dataframe and compares the accessed CSV
column's header rows with the period requested by the question.  Findings are
triage only; the script never rewrites a submission.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import sys
import unicodedata
from pathlib import Path


def fold(value: str) -> str:
    value = value.casefold().replace("đ", "d")
    value = "".join(
        character
        for character in unicodedata.normalize("NFKD", value)
        if not unicodedata.combining(character)
    )
    return re.sub(r"\s+", " ", value).strip()


def audited_ids(submission_dir: Path) -> set[int]:
    ids: set[int] = set()
    for name in ("source_audit.json", "panel_source_audit.json"):
        path = submission_dir / name
        if not path.exists():
            continue
        ids.update(
            int(row["id"])
            for row in json.loads(path.read_text(encoding="utf-8"))
            if isinstance(row, dict) and "id" in row
        )
    return ids


def question_period(question: str) -> str | None:
    text = fold(question)
    # A comparative/growth question legitimately needs both an ending cell
    # and an opening/prior comparative cell.  This audit only has enough
    # information to judge single-period questions without false positives.
    if len(set(re.findall(r"\b(?:19|20)\d{2}\b", text))) >= 2:
        return None
    opening = any(token in text for token in (
        "dau nam", "so dau nam", "ngay 1/1", "ngay 01/01", "tai 1/1", "tai 01/01",
    ))
    ending = any(token in text for token in (
        "cuoi nam", "so cuoi nam", "den ngay 31/12", "tai ngay 31/12", "vao ngay 31/12",
    ))
    if opening == ending:
        return None
    return "opening" if opening else "ending"


def single_question_year(question: str) -> int | None:
    """Return the only calendar/fiscal year named by a question.

    A single-year question that reads a ``Năm trước`` comparative column from
    a report carrying that same year is a high-signal error even when the
    wording does not explicitly say ``cuối năm``.  Multi-year questions are
    intentionally excluded because both current and comparative columns can be
    legitimate operands.
    """

    years = {int(value) for value in re.findall(r"\b(?:19|20)\d{2}\b", fold(question))}
    return next(iter(years)) if len(years) == 1 else None


def report_year(path: Path) -> int | None:
    match = re.search(r"financial_statements_((?:19|20)\d{2})(?:_|\.)", path.name)
    return int(match.group(1)) if match else None


def descriptor_year_role(descriptor: str) -> str | None:
    """Classify a column as the report's current or comparative year."""

    current = any(token in descriptor for token in (
        "nam nay", "nam hien hanh", "ky nay", "ky hien hanh",
    ))
    prior = any(token in descriptor for token in (
        "nam truoc", "ky truoc",
    ))
    if current == prior:
        return None
    return "current" if current else "prior"


def descriptor_explicit_year(descriptor: str) -> int | None:
    """Return an unambiguous calendar year printed in a column header.

    Some reports label comparative columns with exact dates (for example
    ``31/12/2023``) instead of ``Năm trước``.  The report filename cannot tell
    which of those columns a legacy query accessed, so retain the explicit
    header year whenever the descriptor contains exactly one distinct year.
    """

    # Restrict extraction to period-like header forms.  A broad four-digit
    # search mistakes legal citations such as Thông tư 200/2014 for a data
    # period, which is especially common in Vietnamese statement templates.
    years = {
        int(value)
        for pattern in (
            r"(?:31/12|0?1/0?1)/((?:19|20)\d{2})",
            r"\bnam\s+((?:19|20)\d{2})\b",
            r"^((?:19|20)\d{2})(?=\b|vnd|trieu|nghin)",
        )
        for value in re.findall(pattern, descriptor)
    }
    return next(iter(years)) if len(years) == 1 else None


def year_role_mismatch(
    question: str, path: Path, descriptor: str,
) -> tuple[int, int, str] | None:
    """Return requested/report year and role when a comparative column is wrong."""

    requested = single_question_year(question)
    report = report_year(path)
    explicit = descriptor_explicit_year(descriptor)
    if requested is not None and report is not None and explicit is not None:
        if requested != explicit:
            return requested, report, f"explicit:{explicit}"
        return None
    role = descriptor_year_role(descriptor)
    if requested is None or report is None or role is None:
        return None
    expected = report if role == "current" else report - 1
    if requested == expected:
        return None
    return requested, report, role


def assignment_map(tree: ast.AST) -> dict[str, list[ast.AST]]:
    result: dict[str, list[ast.AST]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                result.setdefault(target.id, []).append(node.value)
    return result


def dataframe_roots(
    node: ast.AST, assignments: dict[str, list[ast.AST]], seen: set[str] | None = None,
) -> set[str]:
    seen = set() if seen is None else set(seen)
    if isinstance(node, ast.Name):
        if re.fullmatch(r"df\d+", node.id):
            return {node.id}
        if node.id in seen:
            return set()
        seen.add(node.id)
        roots: set[str] = set()
        for value in assignments.get(node.id, []):
            roots.update(dataframe_roots(value, assignments, seen))
        return roots
    roots: set[str] = set()
    for child in ast.iter_child_nodes(node):
        roots.update(dataframe_roots(child, assignments, seen))
    return roots


def literal_index(node: ast.AST) -> str | int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int)):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
    ):
        return -node.operand.value
    return None


def accessed_columns(tree: ast.AST, assignments: dict[str, list[ast.AST]]) -> set[tuple[str, str | int]]:
    accesses: set[tuple[str, str | int]] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Subscript):
            continue
        # df/alias['2']
        if not (isinstance(node.value, ast.Attribute) and node.value.attr in {"iloc", "loc"}):
            index = literal_index(node.slice)
            if index is None:
                continue
            for root in dataframe_roots(node.value, assignments):
                accesses.add((root, index))
            continue
        # df/alias.iloc[row, column]
        if not isinstance(node.slice, ast.Tuple) or len(node.slice.elts) < 2:
            continue
        index = literal_index(node.slice.elts[1])
        if index is None:
            continue
        for root in dataframe_roots(node.value.value, assignments):
            accesses.add((root, index))
    return accesses


def column_descriptors(path: Path) -> tuple[list[str], dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = []
        for index, row in enumerate(reader):
            if index >= 4:
                break
            rows.append(row)
    descriptors = {
        name: fold(" ".join(str(row.get(name, "")) for row in rows))
        for name in fieldnames
    }
    return fieldnames, descriptors


def descriptor_period(descriptor: str) -> str | None:
    opening = any(token in descriptor for token in (
        "so dau nam", "nam truoc", "01/01", "1/1/", "dau ky",
    ))
    ending = any(token in descriptor for token in (
        "so cuoi nam", "so du cuoi nam", "nam nay", "31/12", "cuoi ky",
    ))
    if opening == ending:
        return None
    return "opening" if opening else "ending"


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--include-audited", action="store_true")
    args = parser.parse_args()

    rows = json.loads((args.submission_dir / "submission.json").read_text(encoding="utf-8"))
    known = set() if args.include_audited else audited_ids(args.submission_dir)
    findings: list[dict] = []
    checked = 0
    for row in rows:
        qid = int(row["id"])
        if qid in known:
            continue
        question = str(row.get("question", ""))
        period = question_period(question)
        requested_year = single_question_year(question)
        if period is None and requested_year is None:
            continue
        try:
            query_text = str(row.get("pandas_query", ""))
            tree = ast.parse(query_text)
        except SyntaxError:
            continue
        # Some source tables encode an opening balance as an explicitly named
        # row (for example "cash and cash equivalents at beginning of year")
        # under a generic current-year column.  The row label is the stronger
        # period signal in that layout.
        query_folded = fold(query_text)
        if period == "opening" and any(token in query_folded for token in (
            "dau nam", "so dau nam", "opening balance", "beginning of year",
        )):
            continue
        checked += 1
        assignments = assignment_map(tree)
        evidence = {
            str(item.get("variable")): args.submission_dir / str(item.get("csv_path"))
            for item in row.get("evidence", [])
            if isinstance(item, dict) and item.get("variable") and item.get("csv_path")
        }
        for variable, index in sorted(accessed_columns(tree, assignments), key=lambda item: (item[0], str(item[1]))):
            path = evidence.get(variable)
            if path is None or not path.exists():
                continue
            fieldnames, descriptors = column_descriptors(path)
            if isinstance(index, int):
                resolved = index if index >= 0 else len(fieldnames) + index
                if not (0 <= resolved < len(fieldnames)):
                    continue
                column = fieldnames[resolved]
            else:
                column = index
            if column == "0" or column not in descriptors:
                continue
            descriptor = descriptors[column]
            source_period = descriptor_period(descriptor)
            mismatch = year_role_mismatch(question, path, descriptor)
            period_conflict = (
                period is not None
                and source_period is not None
                and source_period != period
            )
            if not period_conflict and mismatch is None:
                continue
            finding = {
                "id": qid,
                "question_period": period,
                "source_period": source_period,
                "variable": variable,
                "column": column,
                "descriptor": descriptor,
                "csv": str(path),
                "answer": row.get("answer"),
                "question": row.get("question"),
            }
            if mismatch is not None:
                requested, report, role = mismatch
                finding.update({
                    "rule": "single_year_current_prior_mismatch",
                    "requested_year": requested,
                    "report_year": report,
                    "column_year_role": role,
                })
            else:
                finding["rule"] = "opening_ending_mismatch"
            findings.append(finding)

    payload = {
        "submission": str(args.submission_dir),
        "audited_ids_skipped": 0 if args.include_audited else len(known),
        "checked_period_questions": checked,
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
