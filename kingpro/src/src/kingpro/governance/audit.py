"""Privacy-preserving append-only hash-chain audit ledger.

The ledger stores operational metadata, never raw prompts, API keys or table
contents.  A hash chain detects accidental/replayed edits.  Configure
``KINGPRO_AUDIT_HMAC_KEY`` in production to authenticate every record as well.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from .secure_config import resolve_secret


_SECRET_KEY = re.compile(
    r"(?:api[_-]?key|token|password|passwd|secret|authorization|cookie|session)",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:Bearer\s+|\bsk-|\bhf_|\brpa_)[A-Za-z0-9._-]{8,}", re.IGNORECASE
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _redact(value: Any, key: str = "") -> Any:
    if _SECRET_KEY.search(key):
        return "<redacted>"
    if isinstance(value, Mapping):
        return {str(child): _redact(item, str(child)) for child, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("<redacted>", value)[:1000]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)[:300]


class AuditLedger:
    """Thread-safe append-only JSONL ledger with optional keyed signatures."""

    def __init__(self, path: str | Path, *, hmac_key: str | bytes | None = None) -> None:
        self.path = Path(path).resolve()
        configured = hmac_key if hmac_key is not None else resolve_secret("KINGPRO_AUDIT_HMAC_KEY", "")
        self._key = configured.encode("utf-8") if isinstance(configured, str) else (configured or b"")
        self._lock = threading.Lock()

    @property
    def integrity_mode(self) -> str:
        return "hmac-sha256-chain" if self._key else "sha256-chain"

    def _tail(self) -> tuple[int, str]:
        if not self.path.is_file():
            return 0, "0" * 64
        last = ""
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    last = line
        if not last:
            return 0, "0" * 64
        try:
            record = json.loads(last)
            return int(record.get("seq", 0)), str(record.get("record_hash", "0" * 64))
        except (json.JSONDecodeError, TypeError, ValueError):
            raise RuntimeError("audit ledger tail is corrupt")

    def append(self, event: str, payload: Mapping[str, Any] | None = None) -> dict[str, Any]:
        safe_payload = _redact(dict(payload or {}))
        with self._lock:
            seq, previous = self._tail()
            unsigned: dict[str, Any] = {
                "seq": seq + 1,
                "timestamp": _utc_now(),
                "event": str(event)[:120],
                "payload": safe_payload,
                "prev_hash": previous,
                "integrity": self.integrity_mode,
            }
            digest = hashlib.sha256(_canonical(unsigned)).hexdigest()
            record = {**unsigned, "record_hash": digest}
            if self._key:
                record["signature"] = hmac.new(
                    self._key, digest.encode("ascii"), hashlib.sha256
                ).hexdigest()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            return record

    def verify(self) -> dict[str, Any]:
        previous = "0" * 64
        count = 0
        issues: list[str] = []
        if not self.path.is_file():
            return {"valid": True, "records": 0, "integrity": self.integrity_mode, "issues": []}
        with self.path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                count += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    issues.append(f"line_{line_number}:invalid_json")
                    break
                actual_hash = str(record.pop("record_hash", ""))
                signature = record.pop("signature", None)
                if record.get("prev_hash") != previous:
                    issues.append(f"line_{line_number}:previous_hash_mismatch")
                expected_hash = hashlib.sha256(_canonical(record)).hexdigest()
                if actual_hash != expected_hash:
                    issues.append(f"line_{line_number}:record_hash_mismatch")
                if self._key:
                    expected_sig = hmac.new(
                        self._key, actual_hash.encode("ascii"), hashlib.sha256
                    ).hexdigest()
                    if not signature or not hmac.compare_digest(str(signature), expected_sig):
                        issues.append(f"line_{line_number}:signature_mismatch")
                previous = actual_hash
        return {
            "valid": not issues,
            "records": count,
            "integrity": self.integrity_mode,
            "tail_hash": previous,
            "issues": issues,
        }


_LEDGER: AuditLedger | None = None
_LEDGER_LOCK = threading.Lock()


def get_audit_ledger() -> AuditLedger:
    global _LEDGER
    if _LEDGER is None:
        with _LEDGER_LOCK:
            if _LEDGER is None:
                path = os.getenv("KINGPRO_AUDIT_LOG", "outputs/governance/audit.jsonl")
                _LEDGER = AuditLedger(path)
    return _LEDGER
