"""Competition submission writer and strict local release validator."""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
import math
import re
import shutil
import sqlite3
import tempfile
import zipfile
from contextlib import closing
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class EvidenceFrame:
    variable: str
    frame: pd.DataFrame


@dataclass(frozen=True, slots=True)
class SubmissionSolution:
    id: int
    question: str
    answer: float | int
    relevant_docs: tuple[str, ...]
    relevant_tables: tuple[str, ...]
    evidence: tuple[EvidenceFrame, ...]
    pandas_query: str
    method: str
    confidence: float


@dataclass(frozen=True, slots=True)
class ValidationReport:
    rows: int
    csv_files: int
    replayed: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    sha256: str

    @property
    def ok(self) -> bool:
        return not self.errors


_IDENTIFIER = re.compile(r"^[A-Za-z_]\w*$")
_TABLE_REF = re.compile(r"^([^|]+)\|(?:table_)?([1-9][0-9]*)$")


def _dedupe(values: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


@lru_cache(maxsize=2048)
def _physical_table_lines(source_path: str, modified_ns: int, size: int) -> tuple[int, ...]:
    """One-based physical source lines, never expanded-row or ordinal indices."""
    del modified_ns, size  # These arguments invalidate cached source revisions.
    raw = Path(source_path).read_bytes()
    return tuple(
        raw.count(b"\n", 0, match.start()) + 1
        for match in re.finditer(br"<table\b[^>]*>.*?</table>", raw, re.I | re.S)
    )


def canonical_table_ref(value: str, *, index_path: Path | None = None) -> str:
    """Resolve internal ``document|table_N`` to ``document|source_line``.

    Internal table IDs are one-based ordinals. Released IDs are one-based
    physical line positions in the organizer's TXT file, as used by the
    release audit. Bare numeric IDs are already release IDs and are preserved;
    their meaning must not be guessed or converted a second time.
    """

    match = _TABLE_REF.fullmatch(value)
    if not match:
        raise ValueError(f"invalid table reference: {value!r}")
    doc_id, position = match.group(1), int(match.group(2))
    if not value.rsplit("|", 1)[1].startswith("table_"):
        return f"{doc_id}|{position}"
    if index_path is None:
        from .paths import INDEX_PATH
        index_path = INDEX_PATH
    with closing(sqlite3.connect(f"file:{index_path.resolve().as_posix()}?mode=ro", uri=True)) as connection:
        document = connection.execute(
            "SELECT source_path, table_count FROM documents WHERE doc_id=?", (doc_id,)
        ).fetchone()
    if document is None:
        raise ValueError(f"unknown source document: {doc_id}")
    source = Path(document[0])
    if not source.is_absolute():
        from .paths import PROJECT_ROOT
        source = PROJECT_ROOT / source
    stat = source.stat()
    lines = _physical_table_lines(str(source.resolve()), stat.st_mtime_ns, stat.st_size)
    if len(lines) != int(document[1]) or not 1 <= position <= len(lines):
        raise ValueError(f"source/index table mismatch: {value}")
    return f"{doc_id}|{lines[position - 1]}"


def _finite_scalar(value: object) -> float | int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"not a numeric scalar: {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite scalar: {value!r}")
    return value


def _validate_expression(expression: str, variable_names: set[str]) -> ast.Expression:
    tree = ast.parse(expression, mode="eval")
    forbidden = (
        ast.Import,
        ast.ImportFrom,
        ast.NamedExpr,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
    )
    local_names=set()
    for node in ast.walk(tree):
        if isinstance(node,ast.Lambda):
            local_names.update(a.arg for a in node.args.posonlyargs+node.args.args+node.args.kwonlyargs)
            if node.args.vararg:local_names.add(node.args.vararg.arg)
            if node.args.kwarg:local_names.add(node.args.kwarg.arg)
        if isinstance(node,ast.comprehension):
            local_names.update(n.id for n in ast.walk(node.target) if isinstance(n,ast.Name))
    allowed_names = variable_names | local_names | {
        "pd",
        "np",
        "float",
        "int",
        "abs",
        "round",
        "min",
        "max",
        "sum",
        "len",
    }
    for node in ast.walk(tree):
        if isinstance(node, forbidden):
            raise ValueError(f"forbidden syntax {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"unknown name {node.id!r}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise ValueError("dunder attribute is forbidden")
    return tree


def _numpy_internal_import(name, globals=None, locals=None, fromlist=(), level=0):
    # NumPy 2.5 ndarray reductions import their internal implementation from C.
    # Keep expression-level imports forbidden and permit only this exact module.
    if level or name not in {"numpy._core._methods", "numpy.core._methods"}:
        raise ImportError(f"query runtime import blocked: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def evaluate_expression(expression: str, frames: dict[str, pd.DataFrame]) -> float | int:
    tree = _validate_expression(expression, set(frames))
    safe_globals = {
        "__builtins__": {"__import__": _numpy_internal_import},
        "pd": pd,
        "np": np,
        "float": float,
        "int": int,
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
    }
    value = eval(compile(tree, "<submission_query>", "eval"), safe_globals, dict(frames))  # noqa: S307
    if isinstance(value, pd.Series):
        if len(value) != 1:
            raise TypeError(f"query returned Series length {len(value)}")
        value = value.iloc[0]
    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise TypeError(f"query returned ndarray size {value.size}")
        value = value.item()
    return _finite_scalar(value)


def write_submission(
    solutions: list[SubmissionSolution],
    build_dir: Path,
    zip_path: Path,
) -> dict[str, object]:
    """Write a deterministic ZIP with one root JSON and root ``data/``."""

    if build_dir.exists():
        shutil.rmtree(build_dir)
    data_dir = build_dir / "data"
    data_dir.mkdir(parents=True)
    payload: list[dict[str, object]] = []
    for solution in sorted(solutions, key=lambda value: value.id):
        answer = _finite_scalar(solution.answer)
        evidence_payload: list[dict[str, str]] = []
        seen_variables: set[str] = set()
        for item in solution.evidence:
            if not _IDENTIFIER.fullmatch(item.variable):
                raise ValueError(f"Invalid evidence variable {item.variable!r} for q{solution.id}")
            if item.variable in seen_variables:
                raise ValueError(f"Duplicate evidence variable {item.variable!r} for q{solution.id}")
            seen_variables.add(item.variable)
            filename = f"q{solution.id:04d}_{item.variable}.csv"
            item.frame.to_csv(data_dir / filename, index=False, encoding="utf-8", lineterminator="\n")
            evidence_payload.append({"variable": item.variable, "csv_path": f"data/{filename}"})
        payload.append(
            {
                "id": int(solution.id),
                "question": solution.question,
                "answer": answer,
                "relevant_docs": _dedupe(solution.relevant_docs),
                "relevant_tables": _dedupe(
                    tuple(canonical_table_ref(value) for value in solution.relevant_tables)
                ),
                "evidence": evidence_payload,
                "pandas_query": solution.pandas_query.strip(),
            }
        )
    json_path = build_dir / "submission.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if zip_path.exists():
        zip_path.unlink()
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    members = [json_path, *sorted(data_dir.glob("*.csv"))]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source in members:
            name = "submission.json" if source == json_path else f"data/{source.name}"
            info = zipfile.ZipInfo(name, date_time=(2026, 6, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    return {"rows": len(payload), "members": len(members), "sha256": digest, "size": zip_path.stat().st_size}


def validate_submission(zip_path: Path, canonical_questions: list[dict[str, object]]) -> ValidationReport:
    errors: list[str] = []
    warnings: list[str] = []
    digest = hashlib.sha256(zip_path.read_bytes()).hexdigest() if zip_path.exists() else ""
    if not zip_path.exists():
        return ValidationReport(0, 0, 0, (f"missing ZIP: {zip_path}",), (), digest)
    canonical = {int(row["id"]): str(row["question"]) for row in canonical_questions}
    replayed = 0
    csv_count = 0
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            errors.append("duplicate ZIP member")
        for info in infos:
            path = PurePosixPath(info.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in info.filename:
                errors.append(f"unsafe ZIP path: {info.filename}")
            if info.flag_bits & 0x1:
                errors.append(f"encrypted ZIP member: {info.filename}")
        if names.count("submission.json") != 1:
            errors.append("ZIP must contain exactly one root submission.json")
            return ValidationReport(0, 0, 0, tuple(errors), tuple(warnings), digest)
        unexpected = [name for name in names if name != "submission.json" and not re.fullmatch(r"data/[^/]+\.csv", name)]
        if unexpected:
            errors.append(f"unexpected ZIP members: {unexpected[:5]}")
        csv_count = sum(name.startswith("data/") and name.endswith(".csv") for name in names)
        try:
            payload = json.loads(archive.read("submission.json").decode("utf-8"))
        except Exception as exc:
            return ValidationReport(0, csv_count, 0, (f"invalid submission.json: {exc}",), (), digest)
        if not isinstance(payload, list):
            return ValidationReport(0, csv_count, 0, ("submission.json root is not an array",), (), digest)
        ids = [row.get("id") for row in payload if isinstance(row, dict)]
        if len(ids) != len(set(ids)):
            errors.append("duplicate question IDs")
        if set(ids) != set(canonical):
            errors.append(f"ID coverage mismatch: missing={sorted(set(canonical)-set(ids))[:10]} extra={sorted(set(ids)-set(canonical))[:10]}")
        with tempfile.TemporaryDirectory(prefix="vifinqa_validate_") as tmp:
            root = Path(tmp)
            archive.extractall(root)
            for index, row in enumerate(payload):
                prefix = f"row[{index}]"
                if not isinstance(row, dict):
                    errors.append(f"{prefix}: not an object")
                    continue
                required = {"id", "question", "answer", "relevant_docs", "relevant_tables", "evidence", "pandas_query"}
                if set(row) != required:
                    errors.append(f"{prefix}: keys mismatch")
                    continue
                qid = row["id"]
                if not isinstance(qid, int) or isinstance(qid, bool):
                    errors.append(f"{prefix}: id is not int")
                    continue
                if canonical.get(qid) != row["question"]:
                    errors.append(f"q{qid}: question text mismatch")
                try:
                    answer = _finite_scalar(row["answer"])
                except Exception as exc:
                    errors.append(f"q{qid}: invalid answer: {exc}")
                    continue
                docs = row["relevant_docs"]
                tables = row["relevant_tables"]
                if not isinstance(docs, list) or not all(isinstance(value, str) for value in docs):
                    errors.append(f"q{qid}: relevant_docs invalid")
                    continue
                if not isinstance(tables, list) or not all(
                    re.fullmatch(r"[^|]+\|table_[1-9]\d*", value) for value in tables
                ):
                    errors.append(f"q{qid}: relevant_tables invalid")
                    continue
                if len(docs) != len(set(docs)) or len(tables) != len(set(tables)):
                    errors.append(f"q{qid}: duplicate provenance")
                if any(value.rsplit("|", 1)[0] not in docs for value in tables):
                    errors.append(f"q{qid}: table document absent from relevant_docs")
                evidence = row["evidence"]
                if not isinstance(evidence, list) or not evidence:
                    errors.append(f"q{qid}: evidence must be a non-empty list")
                    continue
                frames: dict[str, pd.DataFrame] = {}
                for item in evidence:
                    if not isinstance(item, dict) or set(item) != {"variable", "csv_path"}:
                        errors.append(f"q{qid}: malformed evidence entry")
                        continue
                    variable = item["variable"]
                    csv_path = item["csv_path"]
                    if not isinstance(variable, str) or not _IDENTIFIER.fullmatch(variable):
                        errors.append(f"q{qid}: invalid variable {variable!r}")
                        continue
                    posix = PurePosixPath(csv_path) if isinstance(csv_path, str) else PurePosixPath("INVALID")
                    if not isinstance(csv_path, str) or posix.is_absolute() or ".." in posix.parts or not csv_path.startswith("data/"):
                        errors.append(f"q{qid}: unsafe csv_path {csv_path!r}")
                        continue
                    source = root.joinpath(*posix.parts)
                    if not source.exists():
                        errors.append(f"q{qid}: missing evidence CSV {csv_path}")
                        continue
                    if variable in frames:
                        errors.append(f"q{qid}: duplicate evidence variable {variable}")
                        continue
                    try:
                        frames[variable] = pd.read_csv(source)
                    except Exception as exc:
                        errors.append(f"q{qid}: cannot read {csv_path}: {exc}")
                expression = row["pandas_query"]
                if not isinstance(expression, str) or not expression.strip():
                    errors.append(f"q{qid}: empty pandas_query")
                    continue
                unused = [variable for variable in frames if not re.search(rf"\b{re.escape(variable)}\b", expression)]
                if unused:
                    errors.append(f"q{qid}: unused evidence variables {unused}")
                try:
                    first = evaluate_expression(expression, frames)
                    second = evaluate_expression(expression, frames)
                    if float(first) != float(second):
                        errors.append(f"q{qid}: non-deterministic replay")
                    if abs(float(first) - float(answer)) > 1e-9:
                        errors.append(f"q{qid}: replay {first!r} != answer {answer!r}")
                    replayed += 1
                except Exception as exc:
                    errors.append(f"q{qid}: replay failed: {type(exc).__name__}: {exc}")
    return ValidationReport(len(payload), csv_count, replayed, tuple(errors), tuple(warnings), digest)
