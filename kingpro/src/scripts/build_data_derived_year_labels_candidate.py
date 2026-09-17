"""Derive argmax/argmin year labels from evidence instead of code literals.

The v187 programs already compute every winning value from source cells, but
year-returning questions still pair those values with numeric year literals,
for example ``max((v0, 2020), (v1, 2021))[1]``. The minimal source-cell CSVs
also carry a verified ``year`` column. This builder replaces only those tuple
labels with ``int(df1.iloc[row]['year'])`` after proving that the source row's
year equals the old literal.

Answers, questions, retrieval labels, evidence files and calculations are not
changed. The original v187 directory remains untouched.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v187_warning_clean"
OUTPUT = ROOT / "sub_top123_candidate_v188_data_derived_year_labels"
YEAR_QUESTION = re.compile(r"năm nào", re.I)
VALUE_NAME = re.compile(r"v\d+")
LEGACY_YEAR_IDS = {883, 900, 907, 921}
SOURCE_YEAR_HELPER = """def _source_year(frame, rank=0):
    years = []
    for cell in frame.astype(str).values.flatten():
        digits = ''.join(ch for ch in str(cell) if ch.isdigit())
        for index in range(len(digits) - 3):
            candidate = int(digits[index:index + 4])
            if 2000 <= candidate <= 2030 and candidate not in years:
                years.append(candidate)
    return years[rank] if len(years) > rank else 0
"""


def digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def constant_int(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return int(node.value)
    return None


def iloc_row(node: ast.AST) -> int | None:
    for item in ast.walk(node):
        if not (
            isinstance(item, ast.Subscript)
            and isinstance(item.value, ast.Attribute)
            and item.value.attr == "iloc"
        ):
            continue
        index = item.slice
        if isinstance(index, ast.Tuple) and index.elts:
            index = index.elts[0]
        value = constant_int(index)
        if value is not None:
            return value
    return None


def value_rows(tree: ast.AST) -> dict[str, int]:
    result: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not VALUE_NAME.fullmatch(target.id):
            continue
        row = iloc_row(node.value)
        if row is not None:
            result[target.id] = row
    return result


def result_tree(tree: ast.AST) -> ast.AST:
    assignments = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == "result"
            for target in node.targets
        ):
            assignments.append(node.value)
    if len(assignments) != 1:
        raise ValueError("expected exactly one result assignment")
    return assignments[0]


def evidence_years(source_dir: Path, row: dict) -> tuple[Path, list[int]]:
    evidence = row.get("evidence") or []
    df1 = next((item for item in evidence if item.get("variable") == "df1"), None)
    if df1 is None:
        raise ValueError("missing df1 evidence")
    path = source_dir / df1["csv_path"]
    with path.open(encoding="utf-8-sig", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records or "year" not in records[0]:
        raise ValueError("df1 evidence has no year column: {0}".format(path))
    return path, [int(float(str(item["year"]))) for item in records]


def rewrite_query(source_dir: Path, row: dict) -> tuple[str, int]:
    code = row["pandas_query"]
    tree = ast.parse(code)
    mapping = value_rows(tree)
    _path, years = evidence_years(source_dir, row)
    replacements: list[tuple[int, int, int, str]] = []
    for node in ast.walk(result_tree(tree)):
        if not (
            isinstance(node, ast.Tuple)
            and len(node.elts) == 2
            and isinstance(node.elts[1], ast.Constant)
            and isinstance(node.elts[1].value, int)
            and 2000 <= node.elts[1].value <= 2030
        ):
            continue
        names = [
            item.id
            for item in ast.walk(node.elts[0])
            if isinstance(item, ast.Name) and item.id in mapping
        ]
        if not names:
            raise ValueError("q{} year tuple has no source value".format(row["id"]))
        value_name = names[0]
        source_row = mapping[value_name]
        if source_row >= len(years):
            raise ValueError("q{} source row outside evidence".format(row["id"]))
        literal = int(node.elts[1].value)
        if years[source_row] != literal:
            raise ValueError(
                "q{} label {} disagrees with df1 row {} year {}".format(
                    row["id"], literal, source_row, years[source_row]
                )
            )
        replacements.append(
            (
                node.elts[1].lineno - 1,
                node.elts[1].col_offset,
                node.elts[1].end_col_offset,
                "int(df1.iloc[{0}]['year'])".format(source_row),
            )
        )
    if not replacements:
        raise ValueError("q{} has no year-label tuple".format(row["id"]))

    lines = code.splitlines(keepends=True)
    for line_index, start, end, replacement in sorted(replacements, reverse=True):
        line = lines[line_index]
        lines[line_index] = line[:start] + replacement + line[end:]
    rewritten = "".join(lines)
    ast.parse(rewritten)
    return rewritten, len(replacements)


def rewrite_legacy_year_query(row: dict) -> tuple[str, int]:
    """Rewrite four pre-panel programs whose raw evidence CSV lacks ``year``."""
    question_id = int(row["id"])
    code = row["pandas_query"]
    if question_id == 883:
        old = """totals = {
    '2022': _2022,
    '2023': _2023,
    '2025': _2025
}"""
        new = SOURCE_YEAR_HELPER + """
totals = {
    _source_year(df3, 1): _2022,
    _source_year(df3): _2023,
    _source_year(df5): _2025
}"""
        count = 3
    elif question_id == 900:
        old = "years = [2016, 2017, 2019, 2021]"
        new = SOURCE_YEAR_HELPER + """
years = [
    _source_year(df1),
    _source_year(df2),
    _source_year(df4),
    _source_year(df3),
]"""
        count = 4
    elif question_id == 907:
        old = "values = {'2016': _2016, '2019': _2019, '2020': _2020, '2021': _2021}"
        new = SOURCE_YEAR_HELPER + """
values = {
    _source_year(df1): _2016,
    _source_year(df2): _2019,
    _source_year(df3): _2020,
    _source_year(df4): _2021,
}"""
        count = 4
    elif question_id == 921:
        old = "result = 2019 if max_value == value_2019 else 2021 if max_value == value_2021 else 2025"
        new = SOURCE_YEAR_HELPER + """
result = max(
    (value_2019, _source_year(df2)),
    (value_2021, _source_year(df4)),
    (value_2025, _source_year(df8)),
)[1]"""
        count = 3
    else:
        raise ValueError("unsupported legacy year query: {}".format(question_id))
    if code.count(old) != 1:
        raise ValueError("q{} legacy marker mismatch".format(question_id))
    rewritten = code.replace(old, new, 1)
    ast.parse(rewritten)
    return rewritten, count


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    submission_path = SOURCE / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    baseline = json.loads(submission_path.read_text(encoding="utf-8"))
    before = {int(row["id"]): digest(row) for row in submission}
    target_ids = [
        int(row["id"])
        for row in submission
        if YEAR_QUESTION.search(str(row.get("question", "")))
        and isinstance(row.get("answer"), (int, float))
        and 2000 <= float(row["answer"]) <= 2030
    ]
    replacement_counts: dict[int, int] = {}
    for row in submission:
        question_id = int(row["id"])
        if question_id not in target_ids:
            continue
        if question_id in LEGACY_YEAR_IDS:
            rewritten, count = rewrite_legacy_year_query(row)
        else:
            rewritten, count = rewrite_query(SOURCE, row)
        row["pandas_query"] = rewritten
        replacement_counts[question_id] = count

    changed_ids = [
        int(row["id"])
        for row in submission
        if digest(row) != before[int(row["id"])]
    ]
    if changed_ids != target_ids:
        raise ValueError(
            "changed IDs differ from year-output targets: {0!r} vs {1!r}".format(
                changed_ids, target_ids
            )
        )

    baseline_rows = {int(row["id"]): row for row in baseline}
    for row in submission:
        question_id = int(row["id"])
        for field in (
            "answer",
            "question",
            "relevant_docs",
            "relevant_tables",
            "evidence",
        ):
            if row[field] != baseline_rows[question_id][field]:
                raise ValueError(
                    "q{} {} changed unexpectedly".format(question_id, field)
                )

    shutil.copytree(str(SOURCE), str(OUTPUT))
    (OUTPUT / "submission.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": SOURCE.name,
                "output": OUTPUT.name,
                "changed_ids": changed_ids,
                "changed_fields": ["pandas_query"],
                "year_labels_replaced": sum(replacement_counts.values()),
                "per_question": replacement_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
