"""Remap ViFinQA table ordinals to competition table-position references.

The public ViFinQA code emits one-based ``document|table_N`` references, while
the competition requires a numeric position in the report.  The modes in this
tool encode the tested interpretations of that position.  ``one-based`` is the
physical 1-based source line at which the ``<table>`` block starts.

This tool rewrites only ``relevant_tables``.  Answers, questions, documents,
evidence CSVs, and Pandas programs are copied byte-for-byte.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import tempfile
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath


TABLE_REF_RE = re.compile(r"^(?P<doc>[^|]+)\|(?:table_)?(?P<ordinal>[1-9]\d*)$")
LINE_REF_RE = re.compile(r"^(?P<doc>[^|]+)\|(?P<line>0|[1-9]\d*)$")
TABLE_BLOCK_RE = re.compile(br"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
ROW_TAG_RE = re.compile(br"<tr\b", re.IGNORECASE)
FIXED_ZIP_TIME = (2026, 6, 1, 0, 0, 0)


@dataclass(frozen=True, slots=True)
class RemapStats:
    rows: int
    input_references: int
    output_references: int
    documents_indexed: int
    tables_indexed: int
    mode: str
    sha256: str
    size: int


def _source_path(project_root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def build_position_map(
    database: Path,
    project_root: Path,
    mode: str,
) -> tuple[dict[tuple[str, int], int], dict[str, set[int]]]:
    """Return ``(doc, ordinal) -> requested position`` and valid positions."""

    mapping: dict[tuple[str, int], int] = {}
    valid_lines: dict[str, set[int]] = {}
    connection = sqlite3.connect(f"file:{database.as_posix()}?mode=ro", uri=True)
    try:
        documents = connection.execute(
            "SELECT doc_id, source_path, table_count FROM documents ORDER BY doc_id"
        ).fetchall()
    finally:
        connection.close()

    for doc_id, raw_path, expected_count in documents:
        source = _source_path(project_root, str(raw_path))
        raw = source.read_bytes()
        blocks = list(TABLE_BLOCK_RE.finditer(raw))
        if len(blocks) != int(expected_count):
            raise ValueError(
                f"{doc_id}: index has {expected_count} tables but source has {len(blocks)}"
            )
        positions: set[int] = set()
        expanded_line_delta_before = 0
        for ordinal, block in enumerate(blocks, start=1):
            physical_one_based = raw.count(b"\n", 0, block.start()) + 1
            if mode == "ordinal-zero":
                position = ordinal - 1
            elif mode == "expanded-zero":
                position = (physical_one_based - 1) + expanded_line_delta_before
            else:
                position = physical_one_based
            if position in positions:
                raise ValueError(f"{doc_id}: two tables map to position {position}")
            mapping[(str(doc_id), ordinal)] = position
            positions.add(position)
            row_count = len(ROW_TAG_RE.findall(block.group(0)))
            # Most source tables occupy one physical line, so expanding their
            # rows adds ``row_count - 1`` lines.  A few official source tables
            # already contain embedded newlines; account for their existing
            # physical span instead of adding those lines a second time.
            physical_span = block.group(0).count(b"\n") + 1
            expanded_span = max(1, row_count)
            expanded_line_delta_before += expanded_span - physical_span
        valid_lines[str(doc_id)] = positions
    return mapping, valid_lines


def _safe_members(archive: zipfile.ZipFile) -> dict[str, bytes]:
    members: dict[str, bytes] = {}
    for info in archive.infolist():
        name = info.filename
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or "\\" in name:
            raise ValueError(f"unsafe ZIP member: {name!r}")
        if name in members:
            raise ValueError(f"duplicate ZIP member: {name!r}")
        members[name] = archive.read(info)
    if "submission.json" not in members:
        raise ValueError("input ZIP has no root submission.json")
    json_members = [name for name in members if name.endswith(".json")]
    if json_members != ["submission.json"]:
        raise ValueError(f"input ZIP must contain one root JSON, got {json_members}")
    unexpected = [
        name
        for name in members
        if name != "submission.json" and not re.fullmatch(r"data/[^/]+\.csv", name)
    ]
    if unexpected:
        raise ValueError(f"unexpected ZIP members: {unexpected[:5]}")
    return members


def _remap_rows(
    rows: list[dict[str, object]],
    mapping: dict[tuple[str, int], int],
    mode: str,
) -> tuple[list[dict[str, object]], int, int]:
    rewritten: list[dict[str, object]] = []
    input_count = 0
    output_count = 0
    for row in rows:
        tables = row.get("relevant_tables")
        if not isinstance(tables, list) or not tables:
            raise ValueError(f"q{row.get('id')}: relevant_tables must be non-empty")
        output_tables: list[str] = []
        seen: set[str] = set()
        for raw_ref in tables:
            if not isinstance(raw_ref, str):
                raise TypeError(f"q{row.get('id')}: non-string table reference")
            match = TABLE_REF_RE.fullmatch(raw_ref)
            if not match:
                raise ValueError(f"q{row.get('id')}: invalid ordinal reference {raw_ref!r}")
            doc_id = match.group("doc")
            ordinal = int(match.group("ordinal"))
            try:
                one_based = mapping[(doc_id, ordinal)]
            except KeyError as exc:
                raise KeyError(f"q{row.get('id')}: table absent from source: {raw_ref}") from exc
            candidates = [f"{doc_id}|{one_based}"]
            if mode == "dual-base":
                zero_based = one_based - 1
                if zero_based < 1:
                    raise ValueError(f"q{row.get('id')}: cannot encode zero-based line {zero_based}")
                candidates.append(f"{doc_id}|{zero_based}")
            for candidate in candidates:
                if candidate not in seen:
                    seen.add(candidate)
                    output_tables.append(candidate)
            input_count += 1
        copied = dict(row)
        copied["relevant_tables"] = output_tables
        rewritten.append(copied)
        output_count += len(output_tables)
    return rewritten, input_count, output_count


def _validate_line_refs(
    rows: list[dict[str, object]],
    valid_lines: dict[str, set[int]],
    mode: str,
) -> None:
    for row in rows:
        qid = row.get("id")
        docs = row.get("relevant_docs")
        tables = row.get("relevant_tables")
        if not isinstance(docs, list) or not all(isinstance(value, str) for value in docs):
            raise ValueError(f"q{qid}: invalid relevant_docs")
        if not isinstance(tables, list) or not tables:
            raise ValueError(f"q{qid}: invalid relevant_tables")
        for index, value in enumerate(tables):
            if not isinstance(value, str):
                raise TypeError(f"q{qid}: non-string line reference")
            match = LINE_REF_RE.fullmatch(value)
            if not match:
                raise ValueError(f"q{qid}: invalid line reference {value!r}")
            doc_id = match.group("doc")
            line_number = int(match.group("line"))
            if doc_id not in docs:
                raise ValueError(f"q{qid}: table document absent from relevant_docs: {value}")
            is_source_line = line_number in valid_lines.get(doc_id, set())
            if mode in {"ordinal-zero", "one-based", "expanded-zero"} and not is_source_line:
                raise ValueError(f"q{qid}: reference is not a table-start line: {value}")
            if mode == "dual-base" and index % 2 == 0 and not is_source_line:
                raise ValueError(f"q{qid}: preferred reference is not a table-start line: {value}")


def _write_zip(output: Path, members: dict[str, bytes], payload: list[dict[str, object]]) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{output.stem}.", suffix=".tmp", dir=output.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            names = ["submission.json", *sorted(name for name in members if name != "submission.json")]
            for name in names:
                info = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o100644 << 16
                data = rendered if name == "submission.json" else members[name]
                archive.writestr(
                    info,
                    data,
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=9,
                )
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()


def remap(
    source_zip: Path,
    output_zip: Path,
    database: Path,
    project_root: Path,
    mode: str,
) -> RemapStats:
    mapping, valid_lines = build_position_map(database, project_root, mode)
    with zipfile.ZipFile(source_zip) as archive:
        members = _safe_members(archive)
    rows = json.loads(members["submission.json"].decode("utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("submission.json root must be an array of objects")
    rewritten, input_count, output_count = _remap_rows(rows, mapping, mode)
    _validate_line_refs(rewritten, valid_lines, mode)
    _write_zip(output_zip, members, rewritten)
    digest = hashlib.sha256(output_zip.read_bytes()).hexdigest()
    return RemapStats(
        rows=len(rewritten),
        input_references=input_count,
        output_references=output_count,
        documents_indexed=len(valid_lines),
        tables_indexed=len(mapping),
        mode=mode,
        sha256=digest,
        size=output_zip.stat().st_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source submission ZIP")
    parser.add_argument("--output", type=Path, required=True, help="rewritten submission ZIP")
    parser.add_argument("--database", type=Path, default=Path("artifacts/tables.sqlite3"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--mode",
        choices=("ordinal-zero", "expanded-zero", "one-based", "dual-base"),
        default="ordinal-zero",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    stats = remap(
        source_zip=args.input.resolve(),
        output_zip=args.output.resolve(),
        database=args.database.resolve(),
        project_root=args.project_root.resolve(),
        mode=args.mode,
    )
    print(json.dumps(asdict(stats), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
