
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from vifinqa.common.corpus.catalog import DocumentRef
from vifinqa.common.corpus.statement import StatementCell
from vifinqa.generation.panel.builder import DictCube, build_cube

logger = logging.getLogger(__name__)

CUBE_FORMAT_VERSION = 1


def _source_paths(docs: list[DocumentRef]) -> list[Path]:
    paths: list[Path] = []
    for doc in sorted(docs, key=lambda d: d.doc_name):
        if doc.text_path is not None:
            paths.append(doc.text_path)
        if doc.tables_dir is not None:
            for table_id in doc.table_ids:
                paths.append(doc.table_csv_path(table_id))
    return paths


def _fingerprint(paths: list[Path]) -> str:
    parts: list[str] = []
    for path in paths:
        try:
            stat = path.stat()
            parts.append(f"{path}:{stat.st_size}:{stat.st_mtime_ns}")
        except OSError:
            parts.append(f"{path}:missing")
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _cell_to_json(cell: StatementCell) -> dict:
    return {
        "ma_so": cell.ma_so,
        "label": cell.label,
        "value": cell.value,
        "raw": cell.raw,
        "table_ref": cell.table_ref,
        "row_idx": cell.row_idx,
        "col_idx": cell.col_idx,
        "scale": cell.scale,
    }


def _cell_from_json(data: dict) -> StatementCell:
    return StatementCell(
        ma_so=data["ma_so"],
        label=data["label"],
        value=data["value"],
        raw=data["raw"],
        table_ref=data["table_ref"],
        row_idx=data["row_idx"],
        col_idx=data["col_idx"],
        scale=data["scale"],
    )


class JsonCubeStore:

    def __init__(self, *, docs: list[DocumentRef], cache_dir: Path) -> None:
        self._docs = docs
        self._cache_dir = cache_dir

    def _manifest_path(self) -> Path:
        return self._cache_dir / "cube" / "manifest.json"

    def _cube_path(self) -> Path:
        return self._cache_dir / "cube" / "cube.json"

    def load_or_build(self) -> DictCube:
        fingerprint = _fingerprint(_source_paths(self._docs))
        manifest_path = self._manifest_path()
        cube_path = self._cube_path()

        if manifest_path.exists() and cube_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                manifest = None
            if (
                manifest is not None
                and manifest.get("cube_format_version") == CUBE_FORMAT_VERSION
                and manifest.get("fingerprint") == fingerprint
            ):
                logger.debug("cube cache HIT: fingerprint=%s", fingerprint)
                raw = json.loads(cube_path.read_text(encoding="utf-8"))
                data = {
                    ticker: {
                        year: {metric: _cell_from_json(cell) for metric, cell in metrics.items()}
                        for year, metrics in years.items()
                    }
                    for ticker, years in raw.items()
                }
                return DictCube(data=data)

        logger.debug("cube cache MISS: fingerprint=%s — rebuilding", fingerprint)
        cube = build_cube(self._docs)
        serializable = {
            ticker: {
                year: {metric: _cell_to_json(cell) for metric, cell in metrics.items()}
                for year, metrics in years.items()
            }
            for ticker, years in cube.data.items()
        }
        cube_path.parent.mkdir(parents=True, exist_ok=True)
        cube_path.write_text(json.dumps(serializable, ensure_ascii=False), encoding="utf-8")
        manifest_path.write_text(
            json.dumps(
                {"cube_format_version": CUBE_FORMAT_VERSION, "fingerprint": fingerprint},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return cube
