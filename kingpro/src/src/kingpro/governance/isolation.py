"""Attest external OS-level isolation without pretending unavailable tools exist."""

from __future__ import annotations

import os
import shutil


def isolation_health() -> dict[str, object]:
    configured = os.getenv("KINGPRO_OS_SANDBOX_PROVIDER", "auto").strip().lower()
    required = os.getenv("KINGPRO_REQUIRE_OS_SANDBOX", "false").lower() in {
        "1", "true", "yes", "on"
    }
    attested = os.getenv("KINGPRO_OS_SANDBOX_ATTESTED", "false").lower() in {
        "1", "true", "yes", "on"
    }
    discovered = {
        name: shutil.which(name)
        for name in ("nsjail", "bwrap", "firejail", "runsc")
        if shutil.which(name)
    }
    if configured == "auto":
        provider = next(iter(discovered), None)
        ready = bool(provider)
    elif configured in discovered:
        provider = configured
        ready = True
    elif configured in {"gvisor-container", "kata-container", "microvm"}:
        provider = configured
        ready = attested
    else:
        provider = None
        ready = False
    return {
        "required": required,
        "provider": provider,
        "ready": ready,
        "attested": attested,
        "discovered_tools": sorted(discovered),
        "current_fallback": "ast-gate-plus-child-process-timeout",
        "dynamic_execution_allowed": ready or not required,
    }

