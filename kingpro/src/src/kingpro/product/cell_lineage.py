"""Fail-closed runtime lineage for audited financial source cells.

The competition artifact already carries compact ``q<ID>_source_cells.csv``
manifests.  They record the physical table, row and column used by an audited
answer.  This module exposes those coordinates to the product runtime, but
only after reading the original extracted table and proving that the manifest
raw token still matches the physical cell.

No coordinate is inferred from an answer value.  Missing, malformed or stale
manifests therefore produce an empty lineage instead of a persuasive but
incorrect highlight.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import threading
from pathlib import Path

import pandas as pd

from kingpro.corpus.table_context import header_row_indexes


def _clean(value: object) -> str:
    text = str(value).replace("\ufeff", "").replace("\u00a0", " ").strip()
    return "" if text.casefold() == "nan" else " ".join(text.split())


def _physical_label(frame: pd.DataFrame, row: int, column: int) -> str:
    """Return the nearest descriptive cell left of one audited value."""

    for index in range(min(column - 1, len(frame.columns) - 1), -1, -1):
        value = _clean(frame.iloc[row, index])
        if not value or value in {"-", "—"}:
            continue
        compact = re.sub(r"[().,%\s+\-]", "", value)
        if compact.isdigit() or re.fullmatch(r"\d+[a-z]?", value.casefold()):
            continue
        return value
    return ""


def _header_path(frame: pd.DataFrame, column: int) -> list[str]:
    values: list[str] = []
    for row in header_row_indexes(frame):
        value = _clean(frame.iloc[row, column])
        if value and value not in values:
            values.append(value)
    column_name = _clean(frame.columns[column])
    if (
        column_name
        and not column_name.isdigit()
        and not column_name.casefold().startswith("unnamed:")
        and column_name not in values
    ):
        values.insert(0, column_name)
    return values


class RuntimeCellLineageIndex:
    """Resolve audited question IDs to verified physical source cells."""

    def __init__(self, root: str | Path, artifact: str | Path) -> None:
        self.root = Path(root).resolve()
        self.artifact = Path(artifact).resolve()
        self._lock = threading.Lock()
        self._catalog: dict[str, dict] | None = None
        self._audit: dict[int, dict] | None = None
        self._counterfactual: dict[int, dict] | None = None
        self._coverage: dict[str, int] | None = None

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()

    def _load_catalog(self) -> dict[str, dict]:
        if self._catalog is None:
            with self._lock:
                if self._catalog is None:
                    catalog: dict[str, dict] = {}
                    path = self.root / "build" / "catalog.jsonl"
                    if path.is_file():
                        with path.open(encoding="utf-8") as handle:
                            for line in handle:
                                if line.strip():
                                    row = json.loads(line)
                                    catalog[str(row.get("table_ref", ""))] = row
                    self._catalog = catalog
        return self._catalog

    def _load_audit(self) -> dict[int, dict]:
        if self._audit is None:
            with self._lock:
                if self._audit is None:
                    indexed: dict[int, dict] = {}
                    path = self.artifact / "source_audit.json"
                    if path.is_file():
                        rows = json.loads(path.read_text(encoding="utf-8-sig"))
                        indexed = {
                            int(row["id"]): row
                            for row in rows
                            if row.get("id") is not None and row.get("sources")
                        }
                    self._audit = indexed
        return self._audit

    def _load_counterfactual(self) -> dict[int, dict]:
        if self._counterfactual is None:
            with self._lock:
                if self._counterfactual is None:
                    indexed: dict[int, dict] = {}
                    path = (
                        self.root
                        / "build"
                        / "runtime_lineage"
                        / f"{self.artifact.name}_counterfactual.json"
                    )
                    submission = self.artifact / "submission.json"
                    catalog = self.root / "build" / "catalog.jsonl"
                    analyzer = (
                        self.root
                        / "src"
                        / "kingpro"
                        / "product"
                        / "counterfactual_lineage.py"
                    )
                    if path.is_file() and all(item.is_file() for item in (submission, catalog, analyzer)):
                        payload = json.loads(path.read_text(encoding="utf-8"))
                        hashes = payload.get("input_hashes") or {}
                        valid = bool(
                            payload.get("schema_version") == "counterfactual-cell-lineage/v1"
                            and payload.get("complete") is True
                            and Path(str(payload.get("candidate", ""))).name == self.artifact.name
                            and hashes.get("submission_sha256") == self._sha256(submission)
                            and hashes.get("catalog_sha256") == self._sha256(catalog)
                            and hashes.get("analyzer_sha256") == self._sha256(analyzer)
                        )
                        if valid:
                            indexed = {
                                int(row["question_id"]): row
                                for row in payload.get("records", [])
                                if row.get("status") == "verified" and row.get("cells")
                            }
                    self._counterfactual = indexed
        return self._counterfactual

    def coverage(self) -> dict[str, int]:
        """Return measured artifact coverage without claiming runtime success."""

        if self._coverage is None:
            audit_ids = set(self._load_audit())
            counterfactual_ids = set(self._load_counterfactual())
            manifest_ids: set[int] = set()
            for path in (self.artifact / "data").glob("q*_source_cells.csv"):
                match = re.fullmatch(r"q(\d+)_source_cells", path.stem)
                if match:
                    manifest_ids.add(int(match.group(1)))
            total = 0
            submission = self.artifact / "submission.json"
            if submission.is_file():
                total = len(json.loads(submission.read_text(encoding="utf-8-sig")))
            self._coverage = {
                "manifest_questions": len(manifest_ids),
                "source_audit_questions": len(audit_ids),
                "audited_union_questions": len(manifest_ids | audit_ids),
                "counterfactual_questions": len(counterfactual_ids),
                "runtime_lineage_questions": len(manifest_ids | audit_ids | counterfactual_ids),
                "submission_questions": total,
            }
        return dict(self._coverage)

    def _manifest_sources(self, question_id: int) -> list[dict]:
        path = self.artifact / "data" / f"q{question_id}_source_cells.csv"
        if not path.is_file():
            return []
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        required = {"source_table", "row_idx", "col_idx", "raw"}
        if not rows or not required.issubset(rows[0]):
            return []
        return rows

    def _audit_sources(self, question_id: int) -> list[dict]:
        row = self._load_audit().get(question_id) or {}
        converted: list[dict] = []
        for source in row.get("sources", []):
            converted.append(
                {
                    "ticker": source.get("ticker", ""),
                    "year": source.get("year", ""),
                    "metric_key": source.get("metric", ""),
                    "raw": source.get("raw", ""),
                    "typed_factor": source.get("typed_factor", 1.0),
                    "scale": source.get("scale", 1.0),
                    "source_table": source.get("table_ref", ""),
                    "source_csv": source.get("csv", ""),
                    "row_idx": source.get("row"),
                    "col_idx": source.get("column"),
                    "audit_label": source.get("label", ""),
                }
            )
        return converted

    def _counterfactual_sources(self, question_id: int) -> list[dict]:
        record = self._load_counterfactual().get(question_id) or {}
        converted: list[dict] = []
        for source in record.get("cells", []):
            converted.append(
                {
                    "ticker": "",
                    "year": "",
                    "metric_key": "",
                    "raw": source.get("raw", ""),
                    "typed_factor": 1.0,
                    "scale": 1.0,
                    "source_table": source.get("table_ref", ""),
                    "source_path": source.get("source_path", ""),
                    "row_idx": source.get("row_idx"),
                    "col_idx": source.get("col_idx"),
                    "variable": source.get("variable", ""),
                    "counterfactuals": source.get("counterfactuals", []),
                    "proof": "counterfactual_result_dependency",
                }
            )
        return converted

    def _physical_path(self, table_ref: str) -> Path | None:
        row = self._load_catalog().get(table_ref)
        if not row or not row.get("csv_path"):
            return None
        table_root = (self.root / "build" / "tables").resolve()
        path = (table_root / str(row["csv_path"])).resolve()
        if path != table_root and table_root not in path.parents:
            return None
        return path if path.is_file() else None

    def for_question(self, question_id: int | str | None) -> list[dict]:
        """Return only cells whose coordinate and raw token can be reproved."""

        try:
            qid = int(question_id)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return []
        audit_sources = self._audit_sources(qid)
        audit_by_coordinate = {
            (
                _clean(item.get("source_table", "")),
                int(item["row_idx"]),
                int(item["col_idx"]),
            ): item
            for item in audit_sources
            if item.get("row_idx") is not None and item.get("col_idx") is not None
        }
        sources = self._manifest_sources(qid) or audit_sources or self._counterfactual_sources(qid)
        verified: list[dict] = []
        frames_by_path: dict[Path, pd.DataFrame] = {}
        for source_index, source in enumerate(sources):
            table_ref = _clean(source.get("source_table", ""))
            physical_path = self._physical_path(table_ref)
            if not table_ref or physical_path is None:
                continue
            try:
                row_idx = int(source["row_idx"])
                col_idx = int(source["col_idx"])
                frame = frames_by_path.get(physical_path)
                if frame is None:
                    frame = pd.read_csv(
                        physical_path,
                        dtype=str,
                        keep_default_na=False,
                        encoding="utf-8-sig",
                    )
                    frames_by_path[physical_path] = frame
                if row_idx < 0 or col_idx < 0:
                    continue
                raw_physical = _clean(frame.iloc[row_idx, col_idx])
                raw_manifest = _clean(source.get("raw", ""))
                if raw_physical != raw_manifest:
                    continue
                recorded_path = _clean(source.get("source_path", ""))
                relative = physical_path.relative_to(self.root).as_posix()
                if recorded_path and recorded_path.replace("\\", "/") != relative:
                    continue
                audited = audit_by_coordinate.get((table_ref, row_idx, col_idx), {})
                label = (
                    _clean(source.get("audit_label", ""))
                    or _clean(audited.get("audit_label", ""))
                    or _physical_label(
                    frame, row_idx, col_idx
                    )
                )
            except (IndexError, KeyError, TypeError, ValueError, OSError):
                continue
            proof = str(source.get("proof") or "coordinate_and_raw_match")
            verification = (
                "counterfactual_result_dependency_and_coordinate_raw_match"
                if proof == "counterfactual_result_dependency"
                else "coordinate_and_raw_match"
            )
            verified.append(
                {
                    "question_id": qid,
                    "source_index": source_index,
                    "table_ref": table_ref,
                    "source_path": relative,
                    "row_idx": row_idx,
                    "col_idx": col_idx,
                    "source_label": label,
                    "header_path": _header_path(frame, col_idx),
                    "raw_manifest": raw_manifest,
                    "raw_physical": raw_physical,
                    "ticker": _clean(source.get("ticker", "")),
                    "year": _clean(source.get("year", "")),
                    "metric_key": _clean(source.get("metric_key", "")),
                    "typed_factor": float(source.get("typed_factor", 1.0) or 1.0),
                    "scale": float(source.get("scale", 1.0) or 1.0),
                    "verified": True,
                    "verification": verification,
                    "variable": _clean(source.get("variable", "")),
                    "counterfactuals": source.get("counterfactuals", []),
                }
            )
        return verified
