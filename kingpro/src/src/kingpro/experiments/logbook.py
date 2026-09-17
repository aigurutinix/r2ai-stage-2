"""Vô Thượng-style durable memory, adapted for the KINGPRO repository.

The original local tool lives at ``~/.claude/tools/vothuong/log.py``.  This
version keeps the useful invariants (append-only deltas, explicit oracle trust,
duplicate gates and project attribution) while making the storage repository
local and adding structured review/run events.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


REAL_ORACLES = {
    "leaderboard",
    "do-luong",
    "measurement",
    "prod",
    "source",
    "manual_source",
    "source_packet",
    "raw_source",
    "raw extracted reports + internal question controls",
    "user",
}
BIASED_ORACLES = {"dev", "offline", "val", "clean_gold", "dev_gold", "heuristic"}
_SECRET_KEY = re.compile(r"(?:api[_-]?key|token|password|passwd|secret|authorization|cookie)", re.I)
_SECRET_VALUE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|Bearer\s+[A-Za-z0-9._~+/-]{12,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})",
    re.I,
)


def oracle_trust(oracle: str) -> str:
    """Classify an evidence source without pretending local validation is gold."""
    normalized = oracle.strip().casefold()
    if normalized in REAL_ORACLES:
        return "that"
    if normalized in BIASED_ORACLES:
        return "thien-lech"
    return "khong-ro"


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().casefold())


def _redact(value: Any, key: str = "") -> Any:
    """Best-effort guard: operational memory must never become a secret dump."""
    if _SECRET_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, Mapping):
        return {str(k): _redact(v, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        return _SECRET_VALUE.sub("[REDACTED]", value)
    return value


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


class ExperimentLogbook:
    """Repository-local, append-only operational memory.

    ``experiments.jsonl`` stores machine-readable events. ``playbook.md`` stores
    durable lessons. Old records are never edited; corrections point to the old
    record through ``supersedes``.
    """

    schema_version = 2

    def __init__(self, root: Path, project: str = "KINGPRO-R2AI-Stage2") -> None:
        self.root = Path(root)
        self.project = project
        self.events_path = self.root / "experiments.jsonl"
        self.playbook_path = self.root / "playbook.md"

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.events_path.exists():
            self.events_path.write_text("", encoding="utf-8")
        if not self.playbook_path.exists():
            self.playbook_path.write_text(
                "# Vô Thượng — KINGPRO playbook\n\n"
                "Append-only: cập nhật bằng delta mới và `supersedes`, không sửa lịch sử.\n\n",
                encoding="utf-8",
            )

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    def _events(self) -> list[dict[str, Any]]:
        self.ensure()
        rows: list[dict[str, Any]] = []
        for line in self.events_path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except (TypeError, ValueError):
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows

    def _next_id(self, prefix: str) -> str:
        largest = 0
        if prefix == "L":
            self.ensure()
            for match in re.finditer(r"\*\*L(\d+)\*\*", self.playbook_path.read_text(encoding="utf-8")):
                largest = max(largest, int(match.group(1)))
        else:
            for row in self._events():
                match = re.fullmatch(rf"{re.escape(prefix)}(\d+)", str(row.get("id", "")))
                if match:
                    largest = max(largest, int(match.group(1)))
        return f"{prefix}{largest + 1:04d}"

    @staticmethod
    def _fingerprint(payload: Mapping[str, Any]) -> str:
        stable = {k: v for k, v in payload.items() if k not in {"id", "event_id", "timestamp"}}
        blob = json.dumps(stable, ensure_ascii=False, sort_keys=True, default=_json_default)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:20]

    def append_event(
        self,
        kind: str,
        *,
        oracle: str,
        deduplicate: bool = False,
        **fields: Any,
    ) -> dict[str, Any]:
        """Append one structured event and return the stored record."""
        self.ensure()
        trust = oracle_trust(oracle)
        prefix = {"experiment": "E", "review": "R", "run": "X", "submission": "S"}.get(kind, "X")
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": kind,
            "project": self.project,
            "oracle": oracle,
            "oracle_trust": trust,
            **fields,
        }
        payload = _redact(payload)
        fingerprint = self._fingerprint(payload)
        if deduplicate:
            for prior in reversed(self._events()):
                if prior.get("fingerprint") == fingerprint:
                    return {**prior, "deduplicated": True}
        record = {
            "id": self._next_id(prefix),
            "event_id": str(uuid.uuid4()),
            "timestamp": self._timestamp(),
            **payload,
            "fingerprint": fingerprint,
        }
        with self.events_path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
        return record

    def add_lesson(
        self,
        text: str,
        *,
        confidence: str = "vua",
        tags: Sequence[str] = (),
        supersedes: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        """Append a durable lesson; block exact normalized duplicates by default."""
        if confidence not in {"cao", "vua", "thap"}:
            raise ValueError("confidence must be cao, vua or thap")
        self.ensure()
        body = self.playbook_path.read_text(encoding="utf-8")
        normalized = _normalize_text(text)
        for existing in re.findall(r"— (.+)$", body, flags=re.MULTILINE):
            if _normalize_text(existing) == normalized and not force:
                return {"status": "duplicate", "existing": existing}
        lesson_id = self._next_id("L")
        parts = [
            f"**{lesson_id}**",
            self._timestamp()[:10],
            f"conf:{confidence}",
            f"proj:{self.project}",
        ]
        clean_tags = [tag.strip() for tag in tags if tag.strip()]
        if clean_tags:
            parts.append(f"tags:{','.join(clean_tags)}")
        if supersedes:
            parts.append(f"supersedes:{supersedes}")
        safe_text = _redact(text)
        with self.playbook_path.open("a", encoding="utf-8", newline="") as stream:
            stream.write(f"- {' · '.join(parts)} — {safe_text.strip()}\n")
        return {"status": "appended", "id": lesson_id, "text": safe_text}

    def recent(
        self,
        limit: int = 20,
        *,
        kind: str = "",
        question_id: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        rows = self._events()
        if kind:
            rows = [row for row in rows if row.get("kind") == kind]
        if question_id is not None:
            rows = [
                row
                for row in rows
                if question_id in [int(item) for item in row.get("question_ids", [])]
            ]
        return rows[-max(0, limit) :]

    def run_and_log(
        self,
        command: Sequence[str],
        *,
        name: str,
        oracle: str = "do-luong",
        cwd: Optional[Path] = None,
        timeout: Optional[float] = None,
        note: str = "",
    ) -> subprocess.CompletedProcess[str]:
        """Run a command and always append exit code and duration."""
        started = datetime.now(timezone.utc)
        try:
            completed = subprocess.run(
                list(command), cwd=cwd, text=True, timeout=timeout, check=False
            )
            status = "thang" if completed.returncode == 0 else "thua"
            exit_code: Optional[int] = completed.returncode
            error = ""
        except subprocess.TimeoutExpired as exc:
            completed = subprocess.CompletedProcess(list(command), 124)
            status = "thua"
            exit_code = 124
            error = f"timeout after {exc.timeout}s"
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        self.append_event(
            "run",
            oracle=oracle,
            name=name,
            result=status,
            metric="exit_code",
            value=exit_code,
            duration_seconds=round(elapsed, 3),
            command=list(command),
            cwd=str(cwd) if cwd else "",
            note=note,
            error=error,
        )
        return completed


def parse_key_values(values: Iterable[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        result[key.strip()] = value.strip()
    return result
