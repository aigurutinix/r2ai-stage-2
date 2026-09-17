"""P7 — validate a built ZIP against the submission rules.

Deliberately independent of the packager: it re-opens the archive and checks it
the way the grader would, so a bug in `package.py` cannot validate itself.
"""

from __future__ import annotations

import json
import math
import re
import zipfile
from pathlib import Path

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TABLE_REF_RE = re.compile(r"^[^|]+\|(?:table_)?\d+$")

PYTHON_KEYWORDS = frozenset(
    "False None True and as assert async await break class continue def del elif else "
    "except finally for from global if import in is lambda nonlocal not or pass raise "
    "return try while with yield".split()
)


FRAME_NAME_RE = re.compile(r"^dfs?\d*$")


def undeclared_frames(code: str, declared: set[str]) -> set[str]:
    """Frame-looking names the program reads without the grader providing them.

    A textual scan is not enough: `for df in [df1, df2]:` binds `df` itself, and
    flagging that would reject a perfectly good program. Only names that are read
    without ever being assigned anywhere in the program can be undefined.
    """

    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()

    assigned: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load) and FRAME_NAME_RE.match(node.id):
                loaded.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assigned.add(node.name)
            assigned.update(a.arg for a in node.args.args)
    return loaded - assigned - declared


def reads_no_frame(code: str) -> bool:
    """True when the program never reads a provided DataFrame.

    The organisers review `pandas_query` by hand in the private round and reject
    a hard-coded answer such as `result = 12345.67`. A program that touches no
    frame is exactly that, whatever else it contains, so this is a hard gate
    rather than a style note.
    """

    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if FRAME_NAME_RE.match(node.id):
                return False
    return True


def validate_submission(zip_path: Path, expected_ids: set[int]) -> list[str]:
    """Return a list of problems; empty means the archive is submittable."""

    problems: list[str] = []
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if not n.endswith("/")]

        json_names = [n for n in names if n.lower().endswith(".json")]
        if len(json_names) != 1:
            problems.append(f"expected exactly 1 .json, found {len(json_names)}: {json_names}")
            return problems
        json_name = json_names[0]
        if "/" in json_name:
            problems.append(f"result .json must sit at the archive root, found {json_name!r}")

        csv_members = {n for n in names if n.startswith("data/")}
        for name in names:
            if name != json_name and name not in csv_members:
                problems.append(f"unexpected member outside data/: {name!r}")

        try:
            records = json.loads(archive.read(json_name).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            problems.append(f"{json_name} is not valid UTF-8 JSON: {exc}")
            return problems

        if not isinstance(records, list):
            problems.append("top-level JSON must be a list")
            return problems

        seen: set[int] = set()
        referenced: set[str] = set()

        for position, record in enumerate(records):
            where = f"record[{position}]"
            if not isinstance(record, dict):
                problems.append(f"{where} is not an object")
                continue
            qid = record.get("id")
            if not isinstance(qid, int) or isinstance(qid, bool):
                problems.append(f"{where} id must be an integer, got {qid!r}")
            else:
                where = f"id={qid}"
                if qid in seen:
                    problems.append(f"{where} duplicated")
                seen.add(qid)

            answer = record.get("answer")
            if isinstance(answer, bool) or not isinstance(answer, (int, float)):
                problems.append(f"{where} answer must be a number, got {answer!r}")
            elif not math.isfinite(float(answer)):
                problems.append(f"{where} answer is not finite: {answer!r}")

            if not isinstance(record.get("question"), str):
                problems.append(f"{where} question must be a string")
            if not isinstance(record.get("pandas_query"), str) or not record["pandas_query"].strip():
                problems.append(f"{where} pandas_query must be a non-empty string")

            docs = record.get("relevant_docs")
            if not isinstance(docs, list) or not all(isinstance(d, str) and d for d in docs):
                problems.append(f"{where} relevant_docs must be a list of strings")

            refs = record.get("relevant_tables")
            if not isinstance(refs, list):
                problems.append(f"{where} relevant_tables must be a list")
            else:
                for ref in refs:
                    if not isinstance(ref, str) or not TABLE_REF_RE.match(ref):
                        problems.append(f"{where} malformed table ref: {ref!r}")

            evidence = record.get("evidence")
            if not isinstance(evidence, list):
                problems.append(f"{where} evidence must be a list")
                continue
            variables: set[str] = set()
            for item in evidence:
                if not isinstance(item, dict):
                    problems.append(f"{where} evidence entry is not an object")
                    continue
                variable = item.get("variable")
                if not isinstance(variable, str) or not IDENTIFIER_RE.match(variable or ""):
                    problems.append(f"{where} invalid variable name: {variable!r}")
                elif variable in PYTHON_KEYWORDS:
                    problems.append(f"{where} variable shadows a Python keyword: {variable!r}")
                elif variable in variables:
                    problems.append(f"{where} duplicate variable: {variable!r}")
                else:
                    variables.add(variable)

                csv_path = item.get("csv_path")
                if not isinstance(csv_path, str) or not csv_path.startswith("data/"):
                    problems.append(f"{where} csv_path must start with 'data/': {csv_path!r}")
                elif csv_path not in csv_members:
                    problems.append(f"{where} csv_path missing from archive: {csv_path!r}")
                else:
                    referenced.add(csv_path)

            # The grader binds each CSV to its declared `variable` name. A
            # program that names something else raises NameError and scores zero
            # on execution however correct its answer is — this exact mismatch
            # ("df" in the code, "df1" in the evidence) cost a whole submission.
            query = record.get("pandas_query")
            if isinstance(query, str):
                if reads_no_frame(query):
                    problems.append(
                        f"{where} pandas_query reads no DataFrame — a hard-coded answer "
                        f"is rejected on manual review: {query.strip()[:60]!r}"
                    )
                undeclared = undeclared_frames(query, variables)
                if undeclared:
                    problems.append(
                        f"{where} pandas_query uses undeclared frame(s) {sorted(undeclared)}; "
                        f"evidence declares {sorted(variables)}"
                    )

        missing = expected_ids - seen
        if missing:
            sample = sorted(missing)[:10]
            problems.append(f"{len(missing)} question ids missing, e.g. {sample}")
        unexpected = seen - expected_ids
        if unexpected:
            problems.append(f"{len(unexpected)} unexpected ids, e.g. {sorted(unexpected)[:10]}")

        orphans = csv_members - referenced
        if orphans:
            problems.append(f"{len(orphans)} CSV files never referenced, e.g. {sorted(orphans)[:5]}")

    return problems
