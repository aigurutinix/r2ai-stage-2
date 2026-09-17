"""Tenant-scoped upload quarantine with type, payload and malware gates."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .crypto import encrypt_bytes, encryption_health


ALLOWED_TYPES = {
    ".csv": {"text/csv", "application/csv", "application/vnd.ms-excel", "text/plain"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".json": {"application/json", "text/json", "text/plain"},
}
_FORMULA = re.compile(r"^\s*(?:[=+@]|-\s*[A-Za-z(])")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "default"


def _safe_name(value: str) -> str:
    name = Path(value or "upload").name
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:120] or "upload"


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", dir=path.parent, delete=False
    ) as handle:
        staging = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)


class UploadRejected(ValueError):
    pass


class UploadQuarantine:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.max_bytes = int(os.getenv("KINGPRO_UPLOAD_MAX_BYTES", str(25 * 1024 * 1024)))
        self.require_scanner = os.getenv(
            "KINGPRO_UPLOAD_REQUIRE_MALWARE_SCANNER", "true"
        ).lower() in {"1", "true", "yes", "on"}
        self.encryption = encryption_health()

    def _scan_payload(self, payload: bytes, suffix: str) -> list[str]:
        issues: list[str] = []
        if payload.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")):
            issues.append("archive_payload_not_allowed")
        if b"\x00" in payload:
            issues.append("binary_nul_byte")
        try:
            text = payload.decode("utf-8-sig")
        except UnicodeDecodeError:
            issues.append("not_utf8_text")
            return issues
        if suffix == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError:
                issues.append("invalid_json")
        if suffix == ".csv":
            dangerous = 0
            for line in text.splitlines()[:10000]:
                for cell in line.split(",")[:500]:
                    if _FORMULA.search(cell.strip(' "')):
                        dangerous += 1
                        if dangerous >= 5:
                            break
                if dangerous >= 5:
                    break
            if dangerous:
                issues.append("csv_formula_injection")
        if any(len(line) > 1_000_000 for line in text.splitlines()):
            issues.append("pathological_line_length")
        return issues

    def _malware_scan(self, path: Path) -> dict[str, Any]:
        scanner = shutil.which("clamscan")
        if not scanner:
            return {
                "available": False,
                "clean": False if self.require_scanner else None,
                "status": "scanner_unavailable",
            }
        result = subprocess.run(
            [scanner, "--no-summary", str(path)],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        return {
            "available": True,
            "clean": result.returncode == 0,
            "status": "clean" if result.returncode == 0 else "infected_or_scan_error",
            "return_code": result.returncode,
            "detail": (result.stdout or result.stderr)[-500:],
        }

    def receive(
        self,
        *,
        tenant_id: str,
        actor: str,
        filename: str,
        content_type: str,
        payload: bytes,
    ) -> dict[str, Any]:
        if not payload:
            raise UploadRejected("empty_upload")
        if len(payload) > self.max_bytes:
            raise UploadRejected("upload_too_large")
        safe_name = _safe_name(filename)
        suffix = Path(safe_name).suffix.lower()
        if suffix not in ALLOWED_TYPES:
            raise UploadRejected("unsupported_file_extension")
        normalized_type = content_type.split(";", 1)[0].strip().lower()
        if normalized_type not in ALLOWED_TYPES[suffix]:
            raise UploadRejected("content_type_extension_mismatch")
        issues = self._scan_payload(payload, suffix)
        if issues:
            raise UploadRejected(",".join(issues))

        upload_id = "upl_" + uuid.uuid4().hex
        tenant = _slug(tenant_id)
        folder = self.root / "quarantine" / tenant / upload_id
        folder.mkdir(parents=True, exist_ok=False)
        data_path = folder / safe_name
        data_path.write_bytes(payload)
        malware = self._malware_scan(data_path)
        if self.encryption["required"] and not self.encryption["configured"]:
            data_path.unlink(missing_ok=True)
            folder.rmdir()
            raise UploadRejected("data_encryption_required_but_unavailable")
        stored_name = safe_name
        if self.encryption["configured"]:
            aad = f"{tenant_id}:{upload_id}:{safe_name}".encode("utf-8")
            encrypted_path = folder / f"{safe_name}.enc"
            encrypted_path.write_bytes(encrypt_bytes(payload, aad=aad))
            data_path.unlink()
            stored_name = encrypted_path.name
        status = "quarantined_clean" if malware.get("clean") is True else "quarantined"
        metadata = {
            "schema_version": "kingpro-upload-quarantine/v1",
            "upload_id": upload_id,
            "tenant_id": tenant_id,
            "actor": actor,
            "filename": safe_name,
            "stored_name": stored_name,
            "content_type": normalized_type,
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest().upper(),
            "created_at": _utc_now(),
            "status": status,
            "malware_scan": malware,
            "encryption": {
                "encrypted": bool(self.encryption["configured"]),
                "algorithm": self.encryption["algorithm"],
                "aad_policy": "tenant:upload-id:filename",
            },
            "approved": False,
            "usable_by_retrieval": False,
        }
        _atomic_json(folder / "metadata.json", metadata)
        return metadata

    def status(self, upload_id: str, tenant_id: str) -> dict[str, Any] | None:
        for state in ("quarantine", "approved"):
            path = self.root / state / _slug(tenant_id) / upload_id / "metadata.json"
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return None

    def approve(self, upload_id: str, tenant_id: str, actor: str) -> dict[str, Any]:
        tenant = _slug(tenant_id)
        source = self.root / "quarantine" / tenant / upload_id
        metadata_path = source / "metadata.json"
        if not metadata_path.is_file():
            raise KeyError(upload_id)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("malware_scan", {}).get("clean") is not True:
            raise UploadRejected("malware_scan_not_clean")
        target = self.root / "approved" / tenant / upload_id
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise UploadRejected("upload_already_approved")
        metadata.update(
            {
                "approved": True,
                "approved_by": actor,
                "approved_at": _utc_now(),
                "status": "approved",
                "usable_by_retrieval": False,
            }
        )
        _atomic_json(source / "metadata.json", metadata)
        source.replace(target)
        return metadata
