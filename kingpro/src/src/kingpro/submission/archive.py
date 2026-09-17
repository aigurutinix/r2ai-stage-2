"""Deterministic competition-archive helpers."""

from __future__ import annotations

import zipfile
from pathlib import Path


DETERMINISTIC_TIMESTAMP = (1980, 1, 1, 0, 0, 0)


def write_deterministic(
    archive: zipfile.ZipFile, path: Path, arcname: str
) -> None:
    """Write one regular file without host timestamps or platform metadata."""
    info = zipfile.ZipInfo(arcname, date_time=DETERMINISTIC_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(
        info,
        path.read_bytes(),
        compress_type=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    )
