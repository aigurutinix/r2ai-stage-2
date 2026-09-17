"""Tenant-scoped feedback and adjudication events; never mutates answers."""

from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Any

from .audit import AuditLedger


VERDICTS = {"correct", "incorrect", "uncertain"}
DECISIONS = {"accepted", "rejected", "needs_evidence"}


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:80] or "default"


class FeedbackStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def _ledger(self, tenant_id: str) -> AuditLedger:
        return AuditLedger(self.root / _slug(tenant_id) / "feedback.jsonl")

    def submit(
        self,
        *,
        tenant_id: str,
        actor: str,
        trace_id: str,
        verdict: str,
        reason_code: str = "",
        comment: str = "",
    ) -> dict[str, Any]:
        if verdict not in VERDICTS:
            raise ValueError("unsupported feedback verdict")
        if not re.fullmatch(r"[A-Fa-f0-9]{32}", trace_id):
            raise ValueError("trace_id must be 32 hexadecimal characters")
        feedback_id = "fb_" + uuid.uuid4().hex
        record = self._ledger(tenant_id).append(
            "feedback.submitted",
            {
                "feedback_id": feedback_id,
                "tenant_id": tenant_id,
                "actor": actor,
                "trace_id": trace_id,
                "verdict": verdict,
                "reason_code": re.sub(r"[^A-Za-z0-9_.-]+", "_", reason_code)[:80],
                "comment_sha256": hashlib.sha256(comment.encode("utf-8")).hexdigest(),
                "comment_chars": len(comment),
            },
        )
        return {"feedback_id": feedback_id, "accepted": True, "ledger_seq": record["seq"]}

    def adjudicate(
        self,
        *,
        tenant_id: str,
        actor: str,
        feedback_id: str,
        decision: str,
        evidence_refs: list[str],
    ) -> dict[str, Any]:
        if decision not in DECISIONS:
            raise ValueError("unsupported adjudication decision")
        if not re.fullmatch(r"fb_[A-Fa-f0-9]{32}", feedback_id):
            raise ValueError("invalid feedback_id")
        record = self._ledger(tenant_id).append(
            "feedback.adjudicated",
            {
                "feedback_id": feedback_id,
                "tenant_id": tenant_id,
                "actor": actor,
                "decision": decision,
                "evidence_refs": [str(item)[:240] for item in evidence_refs[:20]],
                "mutation_allowed": False,
                "next_step": "review-ledger-promotion-gate",
            },
        )
        return {"feedback_id": feedback_id, "decision": decision, "ledger_seq": record["seq"]}

