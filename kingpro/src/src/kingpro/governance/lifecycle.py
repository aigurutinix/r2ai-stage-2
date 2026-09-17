"""Version and supersession evidence for the local financial source catalog."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def build_catalog_snapshot(catalog_path: str | Path) -> dict[str, Any]:
    catalog = Path(catalog_path).resolve()
    reports: dict[str, dict[str, Any]] = {}
    table_count = 0
    with catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            report_id = str(row.get("report_id") or "")
            if not report_id:
                continue
            table_count += 1
            state = reports.setdefault(
                report_id,
                {
                    "report_id": report_id,
                    "ticker": str(row.get("ticker") or ""),
                    "year": str(row.get("year") or ""),
                    "scope": str(row.get("scope") or ""),
                    "tables": 0,
                    "fingerprint_input": [],
                },
            )
            state["tables"] += 1
            state["fingerprint_input"].append(
                (
                    str(row.get("table_ref") or ""),
                    str(row.get("csv_path") or ""),
                    int(row.get("n_rows") or 0),
                    int(row.get("n_cols") or 0),
                )
            )
    normalized: dict[str, dict[str, Any]] = {}
    for report_id, state in sorted(reports.items()):
        evidence = sorted(state.pop("fingerprint_input"))
        fingerprint = hashlib.sha256(
            json.dumps(evidence, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest().upper()
        normalized[report_id] = {**state, "fingerprint": fingerprint}
    return {
        "schema_version": "kingpro-source-lifecycle/v1",
        "generated_at": _utc_now(),
        "authority": os.getenv("KINGPRO_SOURCE_AUTHORITY", "BTC ViFinQA source package"),
        "catalog_path": str(catalog),
        "catalog_sha256": _sha256(catalog),
        "report_count": len(normalized),
        "table_count": table_count,
        "reports": normalized,
    }


def compare_snapshots(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any]:
    old = (previous or {}).get("reports") or {}
    new = current.get("reports") or {}
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))
    changed = sorted(
        report_id
        for report_id in set(old) & set(new)
        if old[report_id].get("fingerprint") != new[report_id].get("fingerprint")
    )
    return {
        "previous_catalog_sha256": (previous or {}).get("catalog_sha256"),
        "current_catalog_sha256": current.get("catalog_sha256"),
        "added_reports": added,
        "removed_reports": removed,
        "changed_reports": changed,
        "has_changes": bool(added or removed or changed),
        "review_required": bool(removed or changed),
        "policy": "changed-or-removed-authoritative-report-requires-review-before-promotion",
    }


def atomic_write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=target.parent, delete=False
    ) as handle:
        staging = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(target)


def read_lifecycle_status(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        return {"available": False, "review_required": None}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"available": False, "review_required": True, "error": "invalid_lifecycle_report"}
    comparison = payload.get("comparison") or {}
    return {
        "available": True,
        "generated_at": payload.get("generated_at"),
        "catalog_sha256": payload.get("catalog_sha256"),
        "report_count": payload.get("report_count"),
        "table_count": payload.get("table_count"),
        "review_required": comparison.get("review_required", False),
        "added_reports": len(comparison.get("added_reports") or []),
        "removed_reports": len(comparison.get("removed_reports") or []),
        "changed_reports": len(comparison.get("changed_reports") or []),
    }

