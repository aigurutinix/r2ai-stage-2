#!/usr/bin/env python3
"""Static, organizer-aware validator for ViFinQA submission packages."""

import argparse
import json
import math
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
import zipfile


VARIABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TABLE_ID_RE = re.compile(r"^(.+)\|([1-9]\d*)$")
CONSTANT_QUERY_RE = re.compile(r"^\s*(?:result\s*=\s*)?[-+]?(?:\d+(?:\.\d*)?|\.\d+)\s*$")
REQUIRED = {"id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"}


def load_ids(path: Path, field: str) -> set:
    return {json.loads(line)[field] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def validate(root: Path, expected_ids: set[int] | None = None, catalog_ids: set[str] | None = None) -> list[str]:
    json_files = list(root.glob("*.json"))
    if len(json_files) != 1:
        return [f"Expected exactly one top-level JSON file, found {len(json_files)}"]
    try:
        rows = json.loads(json_files[0].read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"Invalid JSON: {error}"]
    if not isinstance(rows, list):
        return ["Submission JSON must contain a list"]
    errors, seen_ids = [], set()
    for index, row in enumerate(rows):
        prefix = f"record {index}"
        if not isinstance(row, dict):
            errors.append(f"{prefix}: must be an object")
            continue
        missing = REQUIRED - row.keys()
        if missing:
            errors.append(f"{prefix}: missing keys {sorted(missing)}")
            continue
        identifier = row["id"]
        if not isinstance(identifier, int) or isinstance(identifier, bool) or identifier in seen_ids:
            errors.append(f"{prefix}: id must be a unique integer")
        seen_ids.add(identifier)
        answer = row["answer"]
        if not isinstance(answer, (int, float)) or isinstance(answer, bool) or not math.isfinite(answer):
            errors.append(f"{prefix}: answer must be a finite number")
        docs = row["relevant_docs"]
        if not isinstance(docs, list) or not all(isinstance(item, str) and item for item in docs):
            errors.append(f"{prefix}: relevant_docs must be a non-empty string list")
            docs = []
        tables = row["relevant_tables"]
        if not isinstance(tables, list) or not all(isinstance(item, str) for item in tables):
            errors.append(f"{prefix}: relevant_tables must be a string list")
            tables = []
        for table_id in tables:
            match = TABLE_ID_RE.fullmatch(table_id)
            if not match:
                errors.append(f"{prefix}: invalid table ID {table_id!r}; expected report_id|positive_start_line")
            elif match.group(1) not in docs:
                errors.append(f"{prefix}: table {table_id!r} has no matching relevant_doc")
            elif catalog_ids is not None and table_id not in catalog_ids:
                errors.append(f"{prefix}: table {table_id!r} is absent from the supplied catalog")
        evidence = row["evidence"]
        variables = set()
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{prefix}: evidence must contain at least one submitted table")
            evidence = []
        for item in evidence:
            if not isinstance(item, dict):
                errors.append(f"{prefix}: evidence entry must be an object")
                continue
            variable, csv_path = item.get("variable"), item.get("csv_path")
            if not isinstance(variable, str) or not VARIABLE_RE.fullmatch(variable) or variable in variables:
                errors.append(f"{prefix}: evidence variables must be unique Python identifiers")
            if isinstance(variable, str):
                variables.add(variable)
            if not isinstance(csv_path, str) or not csv_path.startswith("data/"):
                errors.append(f"{prefix}: csv_path must be a relative data/ path")
            elif not (root / csv_path).is_file():
                errors.append(f"{prefix}: missing evidence file {csv_path}")
        query = row["pandas_query"]
        if not isinstance(query, str) or not query.strip():
            errors.append(f"{prefix}: pandas_query must be non-empty text")
        elif CONSTANT_QUERY_RE.fullmatch(query):
            errors.append(f"{prefix}: pandas_query may not be a constant result")
        else:
            # The organizers require the query to compute its answer from the
            # submitted tables rather than return a constant, and in the private
            # phase they read the queries by hand. They do not require every
            # submitted table to be read: a question needing one figure from six
            # retrieved tables legitimately touches one. Demanding all of them
            # produced "+ 0 * df3.shape[0]" padding on 97% of rows — terms that
            # compute nothing, on queries a human is about to inspect.
            if not any(re.search(rf"\b{re.escape(variable)}\b", query) for variable in variables):
                errors.append(f"{prefix}: pandas_query reads none of its evidence variables")
    if expected_ids is not None and seen_ids != expected_ids:
        errors.append("Submission IDs do not exactly match the supplied question IDs")
    return errors


def safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    for member in archive.infolist():
        parts = PurePosixPath(member.filename).parts
        if member.filename.startswith("/") or ".." in parts:
            raise ValueError(f"Unsafe ZIP member: {member.filename}")
    archive.extractall(target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--catalog", type=Path)
    args = parser.parse_args()
    expected_ids = load_ids(args.questions, "id") if args.questions else None
    catalog_ids = load_ids(args.catalog, "submission_table_id") if args.catalog else None
    temporary = Path(tempfile.mkdtemp(prefix="vifinqa_submission_"))
    try:
        root = args.submission
        if args.submission.suffix.lower() == ".zip":
            with zipfile.ZipFile(args.submission) as archive:
                safe_extract(archive, temporary)
            root = temporary
        errors = validate(root, expected_ids, catalog_ids)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    if errors:
        print("INVALID")
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(1)
    print("VALID")


if __name__ == "__main__":
    main()
