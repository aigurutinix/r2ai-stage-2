"""Mask user PII and credentials before any external model request."""

from __future__ import annotations

import re
from typing import Any


_PATTERNS = (
    ("email", re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")),
    ("phone", re.compile(r"(?<!\d)(?:\+?84|0)(?:3|5|7|8|9)(?:\d[ .-]?){7}\d(?!\d)")),
    (
        "labeled_personal_id",
        re.compile(r"(?i)\b(?:cccd|cmnd|can cuoc)\b([^\d\n]{0,20})(\d{9}|\d{12})"),
    ),
    (
        "labeled_account",
        re.compile(r"(?i)\b(?:so tai khoan|tai khoan|account)\b([^\d\n]{0,20})(\d{8,20})"),
    ),
    (
        "credential",
        re.compile(
            r"(?i)(?:Bearer\s+|\b(?:api[_-]?key|token|secret|password)\s*[:=]\s*|\bsk-|\bhf_)[A-Za-z0-9._-]{8,}"
        ),
    ),
)


def sanitize_egress(text: str) -> tuple[str, dict[str, Any]]:
    value = str(text or "")
    counts: dict[str, int] = {}
    for label, pattern in _PATTERNS:
        def replace(match: re.Match) -> str:
            counts[label] = counts.get(label, 0) + 1
            if label.startswith("labeled_"):
                whole = match.group(0)
                digits = match.group(2)
                return whole[: -len(digits)] + "<masked:" + digits[-3:] + ">"
            if label == "email":
                local, _, domain = match.group(0).partition("@")
                return (local[:1] or "x") + "***@" + domain
            if label == "phone":
                digits = re.sub(r"\D", "", match.group(0))
                return "<masked-phone:" + digits[-3:] + ">"
            return "<redacted-credential>"

        value = pattern.sub(replace, value)
    return value, {
        "sanitized": bool(counts),
        "finding_counts": counts,
        "policy": "mask-before-model-egress-no-raw-pii-in-audit",
    }

