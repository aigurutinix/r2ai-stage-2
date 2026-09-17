from __future__ import annotations

import importlib.util
import socket
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "hanoi_cold_start_drill.py"
SPEC = importlib.util.spec_from_file_location("hanoi_cold_start_drill", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_port_guard_does_not_claim_an_owned_listener_is_free() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        listener.listen(1)
        assert not MODULE.port_available("127.0.0.1", port)


def test_report_requires_smoke_truth_and_cleanup() -> None:
    good = MODULE.build_report(
        ports_free_before=True,
        backend_ready=True,
        frontend_ready=True,
        health={"ok": True},
        smoke={"ok": True},
        cleanup={
            "backend": {"stopped": True},
            "frontend": {"stopped": True},
        },
        ports_free_after=True,
    )
    assert good["cold_start_ok"]

    unsafe = MODULE.build_report(
        ports_free_before=True,
        backend_ready=True,
        frontend_ready=True,
        health={"ok": True},
        smoke={"ok": True},
        cleanup={
            "backend": {"stopped": True},
            "frontend": {"stopped": False},
        },
        ports_free_after=False,
    )
    assert not unsafe["cold_start_ok"]
    assert not unsafe["gates"]["owned_process_cleanup"]
