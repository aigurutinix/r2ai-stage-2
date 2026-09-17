"""Resolve sensitive configuration from environment or mounted files."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_secret(name: str, default: str = "") -> str:
    direct = os.getenv(name)
    if direct:
        return direct
    file_path = os.getenv(f"{name}_FILE", "").strip()
    if not file_path:
        return default
    path = Path(file_path).resolve()
    if not path.is_file():
        raise RuntimeError(f"mounted sensitive-config file for {name} does not exist")
    value = path.read_text(encoding="utf-8").strip()
    if not value:
        raise RuntimeError(f"mounted sensitive-config file for {name} is empty")
    return value

