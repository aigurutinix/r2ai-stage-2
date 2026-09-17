
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from vifinqa.generation.hard.recipe.audit.base import AuditVerdict, DependencyStatus, RejectionReason


def cache_key(
    *,
    source_content_hash: str,
    dependency_id: str,
    interpretation_id: str | None,
    prompt_version: str,
    model_id: str,
) -> str:
    raw = "|".join(
        [source_content_hash, dependency_id, interpretation_id or "", prompt_version, model_id]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class AuditCache:
    def __init__(self, *, cache_dir: Path) -> None:
        self._dir = cache_dir / "audit"
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self._dir / f"{key}.json"

    def get(self, key: str) -> AuditVerdict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return AuditVerdict(
            status=DependencyStatus(data["status"]),
            reason=RejectionReason(data["reason"]) if data.get("reason") else None,
            interpretation_id=data.get("interpretation_id"),
            detail=data.get("detail", ""),
        )

    def put(self, key: str, verdict: AuditVerdict) -> None:
        path = self._path(key)
        path.write_text(
            json.dumps(
                {
                    "status": verdict.status.value,
                    "reason": verdict.reason.value if verdict.reason else None,
                    "interpretation_id": verdict.interpretation_id,
                    "detail": verdict.detail,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
