"""Read/write authority contracts for product responses and future actions."""

from __future__ import annotations

from typing import Any


_WRITE_ACTIONS = {
    "export_submission",
    "send_email",
    "publish_report",
    "update_dashboard",
    "erp_write",
    "portfolio_action",
}


def approval_contract(action: str, *, approved: bool = False, actor: str | None = None) -> dict[str, Any]:
    action = str(action or "read_analysis")
    requires = action in _WRITE_ACTIONS
    return {
        "action": action,
        "risk": "write" if requires else "read",
        "requires_approval": requires,
        "approved": bool(approved) if requires else True,
        "actor": actor,
        "can_execute": (not requires) or bool(approved),
        "policy": "human-approval-before-side-effect",
    }


def annotate_read_response(
    response: dict[str, Any],
    *,
    authority: str,
    llm_role: str,
) -> dict[str, Any]:
    """Attach a stable authority declaration without changing answer semantics."""
    response["operation"] = "read"
    response["requires_approval"] = False
    response["authority"] = {
        "decision": authority,
        "llm_role": llm_role,
        "verified": bool(response.get("grounded")) and response.get("status") == "answered",
        "policy": "code-decides-llm-proposes-language-or-bounded-plan",
    }
    response["approval"] = approval_contract("read_analysis")
    return response

