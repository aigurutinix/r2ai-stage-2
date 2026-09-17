"""Build the durable raw-table and row index used by every solver pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import time
from pathlib import Path

from .html_tables import parse_html_table
from .paths import INDEX_MANIFEST_PATH, INDEX_PATH, REPORT_ROOT
from .table_semantics import TableAnalyzer, statement_kind
from .text import clean_text, fold_text, parse_vn_number


TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
PAGE_RE = re.compile(r"===== PAGE\s+(\d+)\s+=====")
TAG_RE = re.compile(r"<[^>]+>")


def report_scope(doc_id: str) -> str:
    folded = fold_text(doc_id).replace(" ", "")
    if any(marker in folded for marker in ("congtyme", "separate", "parent")):
        return "parent"
    if any(marker in folded for marker in ("hopnhat", "consol", "consolidated")):
        return "consolidated"
    return "unknown"


def _plain_context(raw: str) -> str:
    return clean_text(TAG_RE.sub(" ", raw))


def _page_at(markers: list[tuple[int, int]], offset: int) -> int:
    page = 0
    for position, number in markers:
        if position > offset:
            break
        page = number
    return page


def _init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        PRAGMA temp_store=MEMORY;
        CREATE TABLE documents (
            doc_id TEXT PRIMARY KEY,
            ticker TEXT NOT NULL,
            report_year INTEGER NOT NULL,
            scope TEXT NOT NULL,
            source_path TEXT NOT NULL,
            table_count INTEGER NOT NULL
        );
        CREATE TABLE tables (
            doc_id TEXT NOT NULL,
            table_id INTEGER NOT NULL,
            page INTEGER NOT NULL,
            context TEXT NOT NULL,
            folded_text TEXT NOT NULL,
            rows_json TEXT NOT NULL,
            n_rows INTEGER NOT NULL,
            n_cols INTEGER NOT NULL,
            statement_kind TEXT NOT NULL,
            unit_scale REAL NOT NULL,
            header_rows_json TEXT NOT NULL,
            PRIMARY KEY (doc_id, table_id)
        );
        CREATE TABLE rows (
            doc_id TEXT NOT NULL,
            table_id INTEGER NOT NULL,
            row_idx INTEGER NOT NULL,
            cells_json TEXT NOT NULL,
            folded_text TEXT NOT NULL,
            row_label TEXT NOT NULL,
            section TEXT NOT NULL,
            PRIMARY KEY (doc_id, table_id, row_idx)
        );
        CREATE INDEX idx_documents_entity ON documents(ticker, report_year, scope);
        CREATE INDEX idx_rows_doc ON rows(doc_id);
        CREATE INDEX idx_rows_table ON rows(doc_id, table_id);
        """
    )


def _source_fingerprint(paths: list[Path], report_root: Path = REPORT_ROOT) -> str:
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.relative_to(report_root)).encode("utf-8"))
        digest.update(f":{stat.st_size}:{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def build_index(
    *,
    force: bool = False,
    output_path: Path = INDEX_PATH,
    manifest_path: Path = INDEX_MANIFEST_PATH,
    expected_tables: int | None = 146_246,
    report_root: Path = REPORT_ROOT,
) -> dict[str, object]:
    paths = sorted(report_root.rglob("*_extracted.txt"))
    if not paths:
        raise FileNotFoundError(f"No source reports found under {report_root}")
    fingerprint = _source_fingerprint(paths, report_root)
    if output_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("source_fingerprint") == fingerprint and manifest.get("format_version") == 2:
            return manifest

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".building.sqlite3")
    if tmp_path.exists():
        tmp_path.unlink()
    conn = sqlite3.connect(tmp_path)
    _init_schema(conn)
    started = time.time()
    table_total = 0
    row_total = 0

    try:
        for doc_no, path in enumerate(paths, start=1):
            raw = path.read_text(encoding="utf-8")
            doc_id = path.parent.name
            try:
                ticker = path.parents[2].name
                report_year = int(path.parents[1].name)
            except (IndexError, ValueError) as exc:
                raise ValueError(f"Unexpected source layout: {path}") from exc
            matches = list(TABLE_RE.finditer(raw))
            markers = [(m.start(), int(m.group(1))) for m in PAGE_RE.finditer(raw)]
            conn.execute(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                (doc_id, ticker, report_year, report_scope(doc_id), str(path), len(matches)),
            )

            previous_end = 0
            for table_id, match in enumerate(matches, start=1):
                grid = parse_html_table(match.group(0))
                page = _page_at(markers, match.start())
                page_start = max((pos for pos, _ in markers if pos <= match.start()), default=0)
                before_start = max(previous_end, page_start)
                context_before = _plain_context(raw[before_start : match.start()])[-1200:]
                next_break = raw.find("\n\n", match.end())
                if next_break < 0:
                    next_break = min(len(raw), match.end() + 400)
                context_after = _plain_context(raw[match.end() : next_break])[:300]
                context = clean_text(f"{context_before} {context_after}")
                flat = clean_text(context + " " + " ".join(" ".join(row) for row in grid))
                folded = fold_text(flat)
                rows_json = json.dumps(grid, ensure_ascii=False, separators=(",", ":"))
                n_cols = max((len(row) for row in grid), default=0)
                analyzer = TableAnalyzer(grid, context=context, report_year=report_year)
                header_rows = list(analyzer.header_indices)
                kind = statement_kind(context, grid).value
                scale_probe_row = next(
                    (idx for idx, row in enumerate(grid) if any(parse is not None for parse in (parse_vn_number(cell) for cell in row))),
                    0,
                )
                scale_probe_col = next(
                    (
                        idx
                        for idx, cell in enumerate(grid[scale_probe_row] if grid else [])
                        if parse_vn_number(cell) is not None
                    ),
                    0,
                )
                table_scale = (
                    analyzer.cell(scale_probe_row, scale_probe_col).unit_scale if grid else 1.0
                )
                conn.execute(
                    "INSERT INTO tables VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        doc_id,
                        table_id,
                        page,
                        context,
                        folded,
                        rows_json,
                        len(grid),
                        n_cols,
                        kind,
                        table_scale,
                        json.dumps(header_rows, separators=(",", ":")),
                    ),
                )
                conn.executemany(
                    "INSERT INTO rows VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        (
                            doc_id,
                            table_id,
                            row_idx,
                            json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                            fold_text(" ".join(row)),
                            analyzer.labels[row_idx],
                            analyzer.sections[row_idx],
                        )
                        for row_idx, row in enumerate(grid)
                    ),
                )
                table_total += 1
                row_total += len(grid)
                previous_end = match.end()

            if doc_no % 25 == 0:
                conn.commit()
                elapsed = max(time.time() - started, 0.001)
                print(
                    f"indexed {doc_no}/{len(paths)} documents, {table_total} tables, "
                    f"{row_total} rows ({doc_no / elapsed:.1f} docs/s)",
                    flush=True,
                )
        conn.commit()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
        if integrity != "ok":
            raise RuntimeError(f"SQLite integrity check failed: {integrity}")
    finally:
        conn.close()

    if expected_tables is not None and table_total != expected_tables:
        raise RuntimeError(f"Expected {expected_tables} tables, extracted {table_total}")
    os.replace(tmp_path, output_path)
    manifest: dict[str, object] = {
        "format_version": 2,
        "source_fingerprint": fingerprint,
        "documents": len(paths),
        "tables": table_total,
        "rows": row_total,
        "elapsed_seconds": round(time.time() - started, 3),
        "index_bytes": output_path.stat().st_size,
        "table_id_base": 1,
        "semantic_columns": True,
        "output_path": str(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    manifest_tmp.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(manifest_tmp, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output", type=Path, default=INDEX_PATH)
    parser.add_argument("--manifest", type=Path, default=INDEX_MANIFEST_PATH)
    parser.add_argument("--expected-tables", type=int, default=146_246)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    print(
        json.dumps(
            build_index(
                force=args.force,
                output_path=args.output,
                manifest_path=args.manifest,
                expected_tables=args.expected_tables,
                report_root=args.report_root,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
