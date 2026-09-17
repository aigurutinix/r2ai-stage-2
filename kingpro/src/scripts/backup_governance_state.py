"""Create, verify and restore-drill a governance-state backup archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.governance.audit import get_audit_ledger  # noqa: E402
from kingpro.submission.archive import write_deterministic  # noqa: E402

SOURCES = (
    ROOT / "outputs" / "governance",
    ROOT / "outputs" / "feedback",
    ROOT / "outputs" / "uploads",
    ROOT / "build" / "source_lifecycle",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def entries() -> list[tuple[Path, str]]:
    rows = []
    for folder in SOURCES:
        if not folder.is_dir():
            continue
        for path in folder.rglob("*"):
            if path.is_file():
                rows.append((path, path.relative_to(ROOT).as_posix()))
    return sorted(rows, key=lambda item: item[1])


def create(output: Path) -> dict:
    rows = entries()
    manifest = {
        "schema_version": "kingpro-governance-backup/v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": [
            {"path": name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path, name in rows
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", delete=False
    ) as handle:
        manifest_path = Path(handle.name)
        json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", suffix=".zip", dir=output.parent, delete=False
    ) as handle:
        staging = Path(handle.name)
    try:
        with zipfile.ZipFile(staging, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            write_deterministic(archive, manifest_path, "MANIFEST.json")
            for path, name in rows:
                write_deterministic(archive, path, name)
        if zipfile.ZipFile(staging).testzip() is not None:
            raise RuntimeError("backup CRC failed")
        staging.replace(output)
    finally:
        manifest_path.unlink(missing_ok=True)
        staging.unlink(missing_ok=True)
    report = verify(output)
    get_audit_ledger().append(
        "backup.created",
        {"archive_sha256": report["archive_sha256"], "files": report["files"]},
    )
    return report


def verify(archive_path: Path) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("backup CRC failed")
        manifest = json.loads(archive.read("MANIFEST.json"))
        for row in manifest["files"]:
            payload = archive.read(row["path"])
            if hashlib.sha256(payload).hexdigest().upper() != row["sha256"]:
                raise RuntimeError(f"backup hash mismatch: {row['path']}")
    return {
        "status": "PASS",
        "archive": str(archive_path.resolve()),
        "archive_sha256": sha256(archive_path),
        "files": len(manifest["files"]),
    }


def restore_drill(archive_path: Path, destination: Path) -> dict:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        for row in manifest["files"]:
            relative = Path(row["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("unsafe backup path")
            target = (destination / relative).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError("backup path escaped restore destination")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(row["path"]))
            if sha256(target) != row["sha256"]:
                raise RuntimeError(f"restore hash mismatch: {row['path']}")
    report = {"status": "PASS", "destination": str(destination.resolve()), "files": len(manifest["files"])}
    get_audit_ledger().append("backup.restore_drill", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    create_parser = sub.add_parser("create")
    create_parser.add_argument("archive", type=Path)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("archive", type=Path)
    restore_parser = sub.add_parser("restore-drill")
    restore_parser.add_argument("archive", type=Path)
    restore_parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.command == "create":
        report = create(args.archive.resolve())
    elif args.command == "verify":
        report = verify(args.archive.resolve())
    else:
        report = restore_drill(args.archive.resolve(), args.destination.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

