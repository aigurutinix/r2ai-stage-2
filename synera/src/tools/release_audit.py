"""Strict, read-only release audit for a ViFinQA ``submission.zip``.

This script intentionally does not extract the archive.  It validates the
complete 1,012-question competition contract, checks table references against
the local HTML-block index, and replays every Pandas expression from
freshly parsed CSV bytes.  It is independent of the solver run so it can be
used as a final gate immediately before uploading an archive.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
import platform
import re
import sqlite3
import stat
import subprocess
import sys
import zlib
import zipfile
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from road2ai_vifinqa.submission import _validate_expression, evaluate_expression  # noqa: E402


EXPECTED_ROWS = 1_012
REQUIRED_KEYS = {
    "id",
    "question",
    "answer",
    "relevant_docs",
    "relevant_tables",
    "evidence",
    "pandas_query",
}
IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
DOC_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
TABLE_REF_RE = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)\|(0|[1-9][0-9]*)$"
)
TABLE_BLOCK_RE = re.compile(br"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
ROW_TAG_RE = re.compile(br"<tr\b", re.IGNORECASE)
CSV_PATH_RE = re.compile(r"^data/[A-Za-z0-9][A-Za-z0-9_.-]*\.csv$")
CSV_MEMBER_RE = re.compile(r"^data/[A-Za-z0-9][A-Za-z0-9_.-]*\.csv$")
ANSWER_ABS_TOLERANCE = 1e-9


class DuplicateJsonKey(ValueError):
    """Raised when JSON text contains an object key more than once."""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateJsonKey(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON numeric constant: {value}")


def _load_json_bytes(data: bytes, *, label: str) -> Any:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{label} is not valid UTF-8: {exc}") from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, DuplicateJsonKey, ValueError) as exc:
        raise ValueError(f"invalid {label}: {exc}") from exc


def _load_canonical_questions(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    rows: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(raw.splitlines(), 1):
        if not raw_line.strip():
            continue
        row = _load_json_bytes(raw_line, label=f"canonical questions line {line_number}")
        if not isinstance(row, dict):
            raise ValueError(f"canonical questions line {line_number} is not an object")
        rows.append(row)
    return rows, raw


def _tree_hash(paths: Iterable[Path], *, base: Path) -> dict[str, Any]:
    files = sorted(
        {path.resolve() for path in paths if path.is_file()},
        key=lambda value: value.relative_to(base.resolve()).as_posix(),
    )
    digest = hashlib.sha256()
    total_size = 0
    for path in files:
        relative = path.relative_to(base.resolve()).as_posix()
        size = path.stat().st_size
        file_hash = _sha256_file(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        total_size += size
    return {
        "sha256": digest.hexdigest(),
        "files": len(files),
        "bytes": total_size,
    }


def _directory_hash(path: Path) -> dict[str, Any] | None:
    if not path.is_dir():
        return None
    return _tree_hash(path.rglob("*"), base=path)


def _git_metadata() -> dict[str, Any]:
    result: dict[str, Any] = {"commit": "", "dirty": None, "status": []}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = [line for line in status.stdout.splitlines() if line]
        result.update(commit=commit.stdout.strip(), dirty=bool(lines), status=lines)
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def _numeric_scalar(value: Any) -> float | int:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"not a numeric scalar: {type(value).__name__}")
    if not math.isfinite(float(value)):
        raise ValueError(f"non-finite scalar: {value!r}")
    return value


def _member_is_safe(info: zipfile.ZipInfo) -> str | None:
    name = info.filename
    if not name or "\x00" in name:
        return "empty or NUL-containing member name"
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        return "absolute or parent-traversing path"
    if "\\" in name or ":" in name:
        return "backslash or drive/stream separator in path"
    if name.endswith("/") or info.is_dir():
        return "directory member is not allowed"
    if info.flag_bits & 0x1:
        return "encrypted member is not allowed"
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    if info.create_system == 3 and unix_mode:
        if stat.S_ISLNK(unix_mode):
            return "symbolic-link member is not allowed"
        if not stat.S_ISREG(unix_mode):
            return "non-regular member is not allowed"
    return None


def _archive_member_hashes(
    archive: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
) -> tuple[str, dict[str, str]]:
    hashes: dict[str, str] = {}
    for info in infos:
        digest = hashlib.sha256()
        with archive.open(info, "r") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
        hashes[info.filename] = digest.hexdigest()
    tree = hashlib.sha256()
    for name in sorted(hashes):
        info = archive.getinfo(name)
        tree.update(name.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(info.file_size).encode("ascii"))
        tree.update(b"\0")
        tree.update(hashes[name].encode("ascii"))
        tree.update(b"\n")
    return tree.hexdigest(), hashes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", type=Path, required=True)
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "source" / "ViFinQA" / "questions" / "questions.jsonl",
    )
    parser.add_argument("--index", type=Path, default=PROJECT_ROOT / "artifacts" / "tables.sqlite3")
    parser.add_argument(
        "--panel",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "financial_panel.json",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "models" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf",
    )
    parser.add_argument("--run-dir", type=Path, help="run directory for checkpoint/log tree hashes")
    parser.add_argument("--report", type=Path, help="optional JSON report output")
    parser.add_argument(
        "--table-ref-mode",
        choices=("ordinal-zero", "expanded-zero", "one-based"),
        default="ordinal-zero",
        help="representation used after the document id in relevant_tables",
    )
    parser.add_argument("--replays", type=int, default=3, help="fresh CSV parses and query executions per row")
    parser.add_argument("--max-members", type=int, default=5_000)
    parser.add_argument("--max-member-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--max-total-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-compression-ratio", type=float, default=200.0)
    return parser


def audit(args: argparse.Namespace) -> dict[str, Any]:
    if args.replays < 1:
        raise ValueError("--replays must be positive")
    if args.max_members < EXPECTED_ROWS + 1:
        raise ValueError("--max-members is too small for a full release")
    if args.max_member_bytes < 1 or args.max_total_bytes < 1 or args.max_compression_ratio <= 0:
        raise ValueError("archive limits must be positive")

    zip_path = args.zip_path.resolve()
    questions_path = args.questions.resolve()
    index_path = args.index.resolve()
    panel_path = args.panel.resolve()
    model_path = args.model.resolve()
    run_dir = (args.run_dir or zip_path.parent).resolve()

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    def error(code: str, message: str, **context: Any) -> None:
        errors.append({"code": code, "message": message, **context})

    def warning(code: str, message: str, **context: Any) -> None:
        warnings.append({"code": code, "message": message, **context})

    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "target": str(zip_path),
        "limits": {
            "max_members": args.max_members,
            "max_member_bytes": args.max_member_bytes,
            "max_total_bytes": args.max_total_bytes,
            "max_compression_ratio": args.max_compression_ratio,
            "fresh_replays": args.replays,
            "answer_abs_tolerance": ANSWER_ABS_TOLERANCE,
            "table_ref_mode": args.table_ref_mode,
        },
        "archive": {},
        "canonical": {},
        "contract": {},
        "provenance": {},
        "replay": {},
        "hashes": {},
        "environment": {},
    }

    canonical_rows: list[dict[str, Any]] = []
    canonical_by_id: dict[int, str] = {}
    if not questions_path.is_file():
        error("canonical.missing", "canonical questions file is missing", path=str(questions_path))
    else:
        try:
            canonical_rows, canonical_raw = _load_canonical_questions(questions_path)
            canonical_ids: list[int] = []
            valid_canonical = True
            for position, row in enumerate(canonical_rows):
                qid = row.get("id")
                question = row.get("question")
                if not isinstance(qid, int) or isinstance(qid, bool):
                    error("canonical.id_type", "canonical ID is not an integer", position=position)
                    valid_canonical = False
                    continue
                if not isinstance(question, str) or not question:
                    error("canonical.question_type", "canonical question is empty or not a string", id=qid)
                    valid_canonical = False
                    continue
                canonical_ids.append(qid)
                if qid in canonical_by_id:
                    error("canonical.duplicate_id", "duplicate canonical question ID", id=qid)
                    valid_canonical = False
                canonical_by_id[qid] = question
            expected_ids = list(range(1, EXPECTED_ROWS + 1))
            if canonical_ids != expected_ids:
                error(
                    "canonical.coverage",
                    "canonical questions must be ordered IDs 1..1012",
                    rows=len(canonical_rows),
                )
                valid_canonical = False
            report["canonical"] = {
                "path": str(questions_path),
                "sha256": _sha256_bytes(canonical_raw),
                "rows": len(canonical_rows),
                "ordered_complete": valid_canonical and canonical_ids == expected_ids,
            }
        except Exception as exc:
            error("canonical.invalid", f"{type(exc).__name__}: {exc}", path=str(questions_path))

    input_hashes: dict[str, Any] = {}
    for label, path, required in (
        ("questions", questions_path, True),
        ("table_index", index_path, True),
        ("financial_panel", panel_path, True),
        ("model", model_path, True),
    ):
        if path.is_file():
            input_hashes[label] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
        elif required:
            error("input.missing", f"required {label} input is missing", path=str(path))
    source_files = list((PROJECT_ROOT / "src" / "road2ai_vifinqa").rglob("*.py"))
    source_files.extend(path for path in (PROJECT_ROOT / "pyproject.toml", PROJECT_ROOT / "README.md") if path.is_file())
    input_hashes["source_tree"] = _tree_hash(source_files, base=PROJECT_ROOT)
    input_hashes["release_auditor"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _sha256_file(Path(__file__).resolve()),
        "bytes": Path(__file__).stat().st_size,
    }
    for label in ("cache", "checkpoints", "llm"):
        tree = _directory_hash(run_dir / label)
        if tree is None:
            warning("run_artifact.missing", f"run {label} directory is missing", path=str(run_dir / label))
        else:
            input_hashes[f"run_{label}"] = tree | {"path": str(run_dir / label)}
    for label in ("manifest.json", "validation.json"):
        path = run_dir / label
        if path.is_file():
            input_hashes[f"run_{label.removesuffix('.json')}"] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
    report["hashes"] = input_hashes

    git = _git_metadata()
    if git.get("dirty"):
        warning("git.dirty", "working tree is dirty; source_tree hash is authoritative")
    model_source = os.environ.get("VIFINQA_MODEL_SOURCE", "")
    if not model_source and (run_dir / "manifest.json").is_file():
        try:
            run_manifest = _load_json_bytes(
                (run_dir / "manifest.json").read_bytes(),
                label="run manifest",
            )
            if isinstance(run_manifest, dict) and isinstance(run_manifest.get("model"), dict):
                model_source = str(run_manifest["model"].get("source", ""))
        except Exception as exc:
            warning("run_manifest.invalid", f"cannot read model source: {type(exc).__name__}: {exc}")
    report["environment"] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "sqlite": sqlite3.sqlite_version,
        "zlib_compile": zlib.ZLIB_VERSION,
        "zlib_runtime": zlib.ZLIB_RUNTIME_VERSION,
        "git": git,
        "model_source": model_source,
    }

    if not zip_path.is_file():
        error("archive.missing", "submission ZIP is missing", path=str(zip_path))
        report.update(errors=errors, warnings=warnings, ok=False)
        return report

    report["archive"]["path"] = str(zip_path)
    report["archive"]["bytes"] = zip_path.stat().st_size
    report["archive"]["sha256"] = _sha256_file(zip_path)

    payload: list[Any] = []
    csv_member_names: set[str] = set()
    referenced_csv_paths: list[str] = []
    replay_candidates: list[tuple[int, Any, list[dict[str, str]], str]] = []
    all_table_refs: list[tuple[int, str, int]] = []
    valid_schema_rows = 0
    finite_answers = 0
    exact_question_rows = 0
    exact_docs_rows = 0
    evidence_entries = 0

    try:
        with zipfile.ZipFile(zip_path, "r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            lowered = [name.casefold() for name in names]
            total_uncompressed = sum(info.file_size for info in infos)
            total_compressed = sum(info.compress_size for info in infos)
            ratios = [
                info.file_size / max(1, info.compress_size)
                for info in infos
                if info.file_size
            ]
            report["archive"].update(
                members=len(infos),
                total_uncompressed_bytes=total_uncompressed,
                total_compressed_bytes=total_compressed,
                maximum_compression_ratio=max(ratios, default=0.0),
                comment_bytes=len(archive.comment),
            )
            safe_to_scan = True
            if len(infos) > args.max_members:
                error("archive.member_limit", "ZIP contains too many members", actual=len(infos))
                safe_to_scan = False
            if total_uncompressed > args.max_total_bytes:
                error("archive.total_size_limit", "ZIP exceeds total uncompressed size limit", actual=total_uncompressed)
                safe_to_scan = False
            if archive.comment:
                error("archive.comment", "ZIP archive comment is not allowed")
            if len(names) != len(set(names)):
                error("archive.duplicate_member", "ZIP contains duplicate member names")
            if len(lowered) != len(set(lowered)):
                error("archive.case_collision", "ZIP contains case-insensitive member collisions")
            for info in infos:
                unsafe_reason = _member_is_safe(info)
                if unsafe_reason:
                    error("archive.unsafe_member", unsafe_reason, member=info.filename)
                    safe_to_scan = False
                if info.file_size > args.max_member_bytes:
                    error(
                        "archive.member_size_limit",
                        "ZIP member exceeds size limit",
                        member=info.filename,
                        bytes=info.file_size,
                    )
                    safe_to_scan = False
                ratio = info.file_size / max(1, info.compress_size) if info.file_size else 0.0
                if ratio > args.max_compression_ratio:
                    error(
                        "archive.compression_ratio",
                        "ZIP member exceeds compression-ratio limit",
                        member=info.filename,
                        ratio=ratio,
                    )
                    safe_to_scan = False
            if names.count("submission.json") != 1:
                error("archive.submission_json", "ZIP must contain exactly one root submission.json")
            unexpected = [
                name
                for name in names
                if name != "submission.json" and not CSV_MEMBER_RE.fullmatch(name)
            ]
            if unexpected:
                error("archive.unexpected_member", "ZIP contains unexpected members", members=unexpected[:20])
            csv_member_names = {name for name in names if CSV_MEMBER_RE.fullmatch(name)}
            report["archive"]["csv_members"] = len(csv_member_names)

            if safe_to_scan:
                try:
                    bad_crc = archive.testzip()
                    if bad_crc is not None:
                        error("archive.crc", "ZIP member failed CRC", member=bad_crc)
                    report["archive"]["crc_ok"] = bad_crc is None
                except Exception as exc:
                    error("archive.crc", f"CRC scan failed: {type(exc).__name__}: {exc}")
                    report["archive"]["crc_ok"] = False

                try:
                    member_tree_hash, member_hashes = _archive_member_hashes(archive, infos)
                    report["archive"]["member_tree_sha256"] = member_tree_hash
                    report["archive"]["submission_json_sha256"] = member_hashes.get("submission.json", "")
                except Exception as exc:
                    error("archive.member_hash", f"member hashing failed: {type(exc).__name__}: {exc}")
            else:
                report["archive"]["crc_ok"] = False
                report["archive"]["deep_scan_skipped"] = True

            if safe_to_scan and names.count("submission.json") == 1:
                try:
                    loaded = _load_json_bytes(archive.read("submission.json"), label="submission.json")
                    if not isinstance(loaded, list):
                        error("schema.root", "submission.json root must be an array")
                    else:
                        payload = loaded
                except Exception as exc:
                    error("schema.json", f"{type(exc).__name__}: {exc}")

            expected_ids = list(range(1, EXPECTED_ROWS + 1))
            observed_ids: list[int] = []
            for position, row in enumerate(payload):
                prefix = f"row[{position}]"
                if not isinstance(row, dict):
                    error("schema.row_type", "submission row is not an object", position=position)
                    continue
                if set(row) != REQUIRED_KEYS:
                    error(
                        "schema.keys",
                        "submission row keys do not match the required schema",
                        position=position,
                        missing=sorted(REQUIRED_KEYS - set(row)),
                        extra=sorted(set(row) - REQUIRED_KEYS),
                    )
                    continue
                qid = row["id"]
                if not isinstance(qid, int) or isinstance(qid, bool):
                    error("schema.id_type", "question ID is not an integer", position=position)
                    continue
                observed_ids.append(qid)
                question = row["question"]
                row_valid = True
                if not isinstance(question, str) or not question:
                    error("schema.question_type", "question is empty or not a string", id=qid)
                    row_valid = False
                elif canonical_by_id.get(qid) != question:
                    error("schema.question_mismatch", "question does not exactly match canonical text", id=qid)
                    row_valid = False
                else:
                    exact_question_rows += 1
                try:
                    answer = _numeric_scalar(row["answer"])
                    finite_answers += 1
                except Exception as exc:
                    error("schema.answer", f"{type(exc).__name__}: {exc}", id=qid)
                    row_valid = False
                    answer = None

                docs = row["relevant_docs"]
                tables = row["relevant_tables"]
                parsed_tables: list[tuple[str, int]] = []
                if (
                    not isinstance(docs, list)
                    or not docs
                    or not all(isinstance(value, str) and DOC_ID_RE.fullmatch(value) for value in docs)
                ):
                    error("provenance.docs", "relevant_docs must be a non-empty list of canonical document IDs", id=qid)
                    row_valid = False
                elif len(docs) != len(set(docs)):
                    error("provenance.duplicate_docs", "relevant_docs contains duplicates", id=qid)
                    row_valid = False
                if not isinstance(tables, list) or not tables:
                    error("provenance.tables", "relevant_tables must be a non-empty list", id=qid)
                    row_valid = False
                elif not all(isinstance(value, str) for value in tables):
                    error("provenance.tables", "relevant_tables contains a non-string value", id=qid)
                    row_valid = False
                else:
                    for value in tables:
                        match = TABLE_REF_RE.fullmatch(value)
                        if not match:
                            error("provenance.table_ref", "invalid relevant_tables reference", id=qid, value=value)
                            row_valid = False
                            continue
                        parsed_tables.append((match.group(1), int(match.group(2))))
                    if len(tables) != len(set(tables)):
                        error("provenance.duplicate_tables", "relevant_tables contains duplicates", id=qid)
                        row_valid = False
                if isinstance(docs, list) and parsed_tables and len(parsed_tables) == len(tables):
                    expected_docs = list(dict.fromkeys(doc_id for doc_id, _ in parsed_tables))
                    if docs != expected_docs:
                        error(
                            "provenance.docs_mismatch",
                            "relevant_docs must exactly equal documents derived from relevant_tables",
                            id=qid,
                        )
                        row_valid = False
                    else:
                        exact_docs_rows += 1
                    all_table_refs.extend((qid, doc_id, table_id) for doc_id, table_id in parsed_tables)

                evidence = row["evidence"]
                normalized_evidence: list[dict[str, str]] = []
                variables: list[str] = []
                if not isinstance(evidence, list) or not evidence:
                    error("evidence.list", "evidence must be a non-empty list", id=qid)
                    row_valid = False
                else:
                    for evidence_position, item in enumerate(evidence):
                        if not isinstance(item, dict) or set(item) != {"variable", "csv_path"}:
                            error(
                                "evidence.entry",
                                "evidence entry must contain exactly variable and csv_path",
                                id=qid,
                                position=evidence_position,
                            )
                            row_valid = False
                            continue
                        variable = item["variable"]
                        csv_path = item["csv_path"]
                        if not isinstance(variable, str) or not IDENTIFIER_RE.fullmatch(variable):
                            error("evidence.variable", "invalid evidence variable", id=qid, value=repr(variable))
                            row_valid = False
                            continue
                        if not isinstance(csv_path, str) or not CSV_PATH_RE.fullmatch(csv_path):
                            error("evidence.csv_path", "invalid evidence CSV path", id=qid, value=repr(csv_path))
                            row_valid = False
                            continue
                        if csv_path not in csv_member_names:
                            error("evidence.csv_missing", "evidence CSV is absent from ZIP", id=qid, csv_path=csv_path)
                            row_valid = False
                            continue
                        variables.append(variable)
                        referenced_csv_paths.append(csv_path)
                        normalized_evidence.append({"variable": variable, "csv_path": csv_path})
                        evidence_entries += 1
                    if len(variables) != len(set(variables)):
                        error("evidence.duplicate_variable", "evidence variables must be unique within a row", id=qid)
                        row_valid = False

                expression = row["pandas_query"]
                if not isinstance(expression, str) or not expression.strip():
                    error("query.empty", "pandas_query must be a non-empty string", id=qid)
                    row_valid = False
                elif isinstance(evidence, list) and len(normalized_evidence) == len(evidence):
                    try:
                        tree = _validate_expression(expression, set(variables))
                        loaded_names = {
                            node.id
                            for node in ast.walk(tree)
                            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
                        }
                        missing_names = sorted(set(variables) - loaded_names)
                        if missing_names:
                            raise ValueError(f"evidence variables not loaded by expression: {missing_names}")
                    except Exception as exc:
                        error("query.ast", f"{type(exc).__name__}: {exc}", id=qid)
                        row_valid = False

                if row_valid and answer is not None:
                    valid_schema_rows += 1
                    replay_candidates.append((qid, answer, normalized_evidence, expression))

            if len(payload) != EXPECTED_ROWS:
                error("schema.row_count", "full release must contain exactly 1,012 rows", actual=len(payload))
            if observed_ids != expected_ids:
                error(
                    "schema.id_order",
                    "full release IDs must be unique and ordered 1..1012",
                    observed=len(observed_ids),
                )

            csv_ref_counts = Counter(referenced_csv_paths)
            referenced_set = set(referenced_csv_paths)
            missing_csvs = sorted(referenced_set - csv_member_names)
            orphan_csvs = sorted(csv_member_names - referenced_set)
            reused_csvs = sorted(path for path, count in csv_ref_counts.items() if count != 1)
            if missing_csvs:
                error("evidence.csv_set_missing", "referenced CSVs are absent from archive", paths=missing_csvs[:20])
            if orphan_csvs:
                error("evidence.csv_set_orphan", "archive contains unreferenced CSVs", paths=orphan_csvs[:20])
            if reused_csvs:
                error("evidence.csv_reused", "each evidence CSV must be referenced exactly once", paths=reused_csvs[:20])
            if len(referenced_csv_paths) != len(csv_member_names):
                error(
                    "evidence.csv_cardinality",
                    "evidence references and archive CSV members are not one-to-one",
                    references=len(referenced_csv_paths),
                    members=len(csv_member_names),
                )

            report["contract"] = {
                "rows": len(payload),
                "valid_rows": valid_schema_rows,
                "ordered_complete_ids": observed_ids == expected_ids,
                "exact_questions": exact_question_rows,
                "finite_answers": finite_answers,
            }
            report["provenance"] = {
                "exact_docs_rows": exact_docs_rows,
                "table_references": len(all_table_refs),
                "unique_table_references": len({(doc, table) for _, doc, table in all_table_refs}),
                "evidence_entries": evidence_entries,
                "csv_members": len(csv_member_names),
                "unique_csv_references": len(referenced_set),
                "orphan_csvs": len(orphan_csvs),
                "missing_csvs": len(missing_csvs),
                "reused_csvs": len(reused_csvs),
            }

            existing_tables: set[tuple[str, int]] = set()
            if index_path.is_file():
                try:
                    uri = f"file:{index_path.as_posix()}?mode=ro"
                    with sqlite3.connect(uri, uri=True) as connection:
                        documents = connection.execute(
                            "SELECT doc_id, source_path, table_count FROM documents ORDER BY doc_id"
                        ).fetchall()
                    for doc_id, raw_path, expected_count in documents:
                        source = Path(str(raw_path))
                        if not source.is_absolute():
                            source = PROJECT_ROOT / source
                        raw = source.read_bytes()
                        blocks = list(TABLE_BLOCK_RE.finditer(raw))
                        if len(blocks) != int(expected_count):
                            raise ValueError(
                                f"{doc_id}: index has {expected_count} tables but source has {len(blocks)}"
                            )
                        expanded_line_delta_before = 0
                        for table_id, block in enumerate(blocks):
                            if args.table_ref_mode == "expanded-zero":
                                physical_zero_based = raw.count(b"\n", 0, block.start())
                                position = physical_zero_based + expanded_line_delta_before
                                row_count = len(ROW_TAG_RE.findall(block.group(0)))
                                physical_span = block.group(0).count(b"\n") + 1
                                expanded_span = max(1, row_count)
                                expanded_line_delta_before += expanded_span - physical_span
                            elif args.table_ref_mode == "one-based":
                                position = raw.count(b"\n", 0, block.start()) + 1
                            else:
                                position = table_id
                            existing_tables.add((str(doc_id), position))
                except Exception as exc:
                    error("provenance.index", f"cannot read table index: {type(exc).__name__}: {exc}")
            missing_table_refs = [
                {"id": qid, "table_ref": f"{doc_id}|{table_id}"}
                for qid, doc_id, table_id in all_table_refs
                if (doc_id, table_id) not in existing_tables
            ]
            if missing_table_refs:
                error(
                    "provenance.table_missing",
                    "relevant_tables contains references absent from the official table index",
                    examples=missing_table_refs[:20],
                    count=len(missing_table_refs),
                )
            report["provenance"].update(
                table_ref_mode=args.table_ref_mode,
                index_tables=len(existing_tables),
                missing_table_references=len(missing_table_refs),
            )

            replayed = 0
            deterministic_rows = 0
            answer_matches = 0
            max_abs_error = 0.0
            max_rel_error = 0.0
            # Expressions reach this point only after _validate_expression has
            # accepted their AST above.  Every pass builds entirely fresh
            # DataFrames from the member bytes; no query can reuse mutated state.
            for qid, answer, evidence, expression in replay_candidates:
                outputs: list[float | int] = []
                try:
                    for _ in range(args.replays):
                        frames = {
                            item["variable"]: pd.read_csv(
                                io.BytesIO(archive.read(item["csv_path"])),
                                encoding="utf-8",
                            )
                            for item in evidence
                        }
                        _validate_expression(expression, set(frames))
                        outputs.append(evaluate_expression(expression, frames))
                except Exception as exc:
                    error("replay.failed", f"{type(exc).__name__}: {exc}", id=qid)
                    continue
                replayed += 1
                output_values = [float(value) for value in outputs]
                if all(value == output_values[0] for value in output_values[1:]):
                    deterministic_rows += 1
                else:
                    error("replay.nondeterministic", "fresh replays produced different values", id=qid, values=output_values)
                difference = abs(output_values[0] - float(answer))
                relative = difference / max(abs(float(answer)), 1e-300)
                max_abs_error = max(max_abs_error, difference)
                max_rel_error = max(max_rel_error, relative)
                if difference <= ANSWER_ABS_TOLERANCE:
                    answer_matches += 1
                else:
                    error(
                        "replay.answer_mismatch",
                        "replayed query does not match submitted answer",
                        id=qid,
                        replayed=outputs[0],
                        answer=answer,
                        absolute_error=difference,
                    )
            report["replay"] = {
                "candidate_rows": len(replay_candidates),
                "fresh_runs_per_row": args.replays,
                "replayed_rows": replayed,
                "deterministic_rows": deterministic_rows,
                "answer_matches": answer_matches,
                "max_absolute_error": max_abs_error,
                "max_relative_error": max_rel_error,
            }
    except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, RuntimeError) as exc:
        error("archive.invalid", f"{type(exc).__name__}: {exc}")

    report.update(errors=errors, warnings=warnings, ok=not errors)
    return report


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        report = audit(args)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "generated_at": _utc_now(),
            "target": str(args.zip_path.resolve()),
            "ok": False,
            "errors": [
                {
                    "code": "audit.fatal",
                    "message": f"{type(exc).__name__}: {exc}",
                }
            ],
            "warnings": [],
        }
    if args.report:
        _write_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    raise SystemExit(0 if report.get("ok") else 1)


if __name__ == "__main__":
    main()
