"""Recoverable TTL enforcement limited to explicit KINGPRO managed roots."""

from __future__ import annotations

import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


class RetentionManager:
    def __init__(self, managed_root: str | Path) -> None:
        self.root = Path(managed_root).resolve()
        self.trash = self.root / ".retention-trash"

    def plan(self, roots: Iterable[tuple[str, int]], *, now: float | None = None) -> list[dict]:
        current = time.time() if now is None else now
        actions: list[dict] = []
        for relative, ttl_days in roots:
            source = (self.root / relative).resolve()
            if source == self.root or self.root not in source.parents or not source.is_dir():
                continue
            cutoff = current - max(1, int(ttl_days)) * 86400
            for child in source.iterdir():
                if child.name.startswith(".") or child.stat().st_mtime >= cutoff:
                    continue
                actions.append(
                    {
                        "source": str(child),
                        "relative": str(child.relative_to(self.root)).replace("\\", "/"),
                        "ttl_days": int(ttl_days),
                        "mtime": datetime.fromtimestamp(
                            child.stat().st_mtime, timezone.utc
                        ).isoformat(),
                    }
                )
        return sorted(actions, key=lambda item: item["relative"])

    def execute(self, actions: list[dict]) -> list[dict]:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        results = []
        for action in actions:
            source = Path(action["source"]).resolve()
            if source == self.root or self.root not in source.parents or not source.exists():
                raise ValueError("retention target escaped managed root or disappeared")
            target = self.trash / stamp / action["relative"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(target)
            shutil.move(str(source), str(target))
            results.append({**action, "trash_path": str(target), "recoverable": True})
        return results


def default_retention_roots() -> list[tuple[str, int]]:
    return [
        ("product_batch_jobs/runs", int(os.getenv("KINGPRO_BATCH_RETENTION_DAYS", "30"))),
        ("product_batch_jobs/inputs", int(os.getenv("KINGPRO_BATCH_INPUT_RETENTION_DAYS", "30"))),
        ("uploads/quarantine", int(os.getenv("KINGPRO_QUARANTINE_RETENTION_DAYS", "7"))),
        ("uploads/approved", int(os.getenv("KINGPRO_UPLOAD_RETENTION_DAYS", "90"))),
    ]

