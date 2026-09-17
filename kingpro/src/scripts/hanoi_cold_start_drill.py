"""Cold-start and recover the audited Hanoi demo on isolated local ports.

The drill never stops an existing service.  It first binds both requested
ports to prove they are free, launches backend and frontend as owned child
process groups, waits for real HTTP readiness, runs the seven Judge View
scenarios, validates champion/fallback truth, and then tears down only those
owned children.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, BinaryIO
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from hanoi_preflight import inspect_health  # noqa: E402
from smoke_demo_day import SCENARIOS, check_scenario  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def port_available(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            probe.bind((host, port))
    except OSError:
        return False
    return True


def _get_json(url: str, timeout: float = 2.0) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def wait_for_json(
    url: str,
    *,
    timeout: float,
    process: subprocess.Popen[bytes],
) -> dict[str, Any]:
    started = time.perf_counter()
    last_error = "not_attempted"
    while time.perf_counter() - started < timeout:
        if process.poll() is not None:
            raise RuntimeError("process_exited_{0}".format(process.returncode))
        try:
            return _get_json(url)
        except (
            urllib.error.URLError,
            TimeoutError,
            OSError,
            ValueError,
            json.JSONDecodeError,
        ) as exc:
            last_error = type(exc).__name__
        time.sleep(0.25)
    raise TimeoutError("startup_timeout:{0}:{1}".format(url, last_error))


def _popen_flags() -> dict[str, Any]:
    if os.name == "nt":
        return {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        }
    return {"start_new_session": True}


def start_process(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log: BinaryIO,
) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        command,
        cwd=str(cwd),
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        **_popen_flags(),
    )


def stop_owned_process(process: subprocess.Popen[bytes] | None) -> dict[str, Any]:
    if process is None:
        return {"started": False, "stopped": True, "pid": None, "returncode": None}
    pid = process.pid
    if process.poll() is None:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                try:
                    os.killpg(os.getpgid(pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
            process.wait(timeout=5)
    return {
        "started": True,
        "stopped": process.poll() is not None,
        "pid": pid,
        "returncode": process.returncode,
    }


def run_smoke(base_url: str, timeout: float, budget_seconds: float) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    transport_error: str | None = None
    started = time.perf_counter()
    try:
        for scenario in SCENARIOS:
            checks.append(check_scenario(base_url, scenario, timeout))
    except (
        urllib.error.URLError,
        TimeoutError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        transport_error = type(exc).__name__
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    passed = sum(bool(check.get("ok")) for check in checks)
    return {
        "passed": passed,
        "total": len(SCENARIOS),
        "elapsed_ms": elapsed_ms,
        "within_budget": elapsed_ms <= int(budget_seconds * 1000),
        "transport_error": transport_error,
        "checks": checks,
        "ok": bool(
            transport_error is None
            and len(checks) == len(SCENARIOS)
            and passed == len(SCENARIOS)
            and elapsed_ms <= int(budget_seconds * 1000)
        ),
    }


def build_report(
    *,
    ports_free_before: bool,
    backend_ready: bool,
    frontend_ready: bool,
    health: dict[str, Any],
    smoke: dict[str, Any],
    cleanup: dict[str, Any],
    ports_free_after: bool,
) -> dict[str, Any]:
    cleanup_ok = bool(
        cleanup.get("backend", {}).get("stopped")
        and cleanup.get("frontend", {}).get("stopped")
        and ports_free_after
    )
    return {
        "cold_start_ok": bool(
            ports_free_before
            and backend_ready
            and frontend_ready
            and health.get("ok")
            and smoke.get("ok")
            and cleanup_ok
        ),
        "gates": {
            "isolated_ports_free_before": ports_free_before,
            "backend_http_ready": backend_ready,
            "frontend_http_ready": frontend_ready,
            "champion_and_fallback_truth": bool(health.get("ok")),
            "judge_view_smoke": bool(smoke.get("ok")),
            "owned_process_cleanup": cleanup_ok,
        },
        "health": health,
        "smoke": smoke,
        "cleanup": cleanup,
        "isolated_ports_free_after": ports_free_after,
        "claim_limit": (
            "This proves a cold local replay/compiler recovery on this machine. "
            "It does not attest a remote dynamic model, private score, venue network, "
            "or manual eligibility evidence."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=18080)
    parser.add_argument("--frontend-port", type=int, default=13010)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--budget-seconds", type=float, default=120.0)
    parser.add_argument(
        "--output",
        default="build/demo_compliance/hanoi_cold_start_latest.json",
    )
    args = parser.parse_args()
    if args.backend_port == args.frontend_port:
        raise SystemExit("backend and frontend ports must differ")

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_dir = output.parent / "cold_start_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    backend_log_path = log_dir / "backend.log"
    frontend_log_path = log_dir / "frontend.log"

    ports_free_before = bool(
        port_available(args.host, args.backend_port)
        and port_available(args.host, args.frontend_port)
    )
    if not ports_free_before:
        report = build_report(
            ports_free_before=False,
            backend_ready=False,
            frontend_ready=False,
            health={"ok": False, "issues": ["isolated_port_in_use"]},
            smoke={"ok": False, "checks": []},
            cleanup={
                "backend": {"started": False, "stopped": True},
                "frontend": {"started": False, "stopped": True},
            },
            ports_free_after=False,
        )
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    backend: subprocess.Popen[bytes] | None = None
    frontend: subprocess.Popen[bytes] | None = None
    backend_ready = False
    frontend_ready = False
    backend_started_ms: int | None = None
    frontend_started_ms: int | None = None
    health: dict[str, Any] = {"ok": False, "issues": ["not_started"]}
    smoke: dict[str, Any] = {"ok": False, "checks": []}
    failure: str | None = None
    started = time.perf_counter()

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["KINGPRO_HOST"] = args.host
        env["KINGPRO_PORT"] = str(args.backend_port)
        env["KINGPRO_ALLOWED_ORIGIN"] = "http://{0}:{1}".format(
            args.host, args.frontend_port
        )
        with backend_log_path.open("wb") as backend_log:
            backend = start_process(
                [sys.executable, str(ROOT / "scripts" / "serve_product.py")],
                cwd=ROOT,
                env=env,
                log=backend_log,
            )
            wait_for_json(
                "http://{0}:{1}/health".format(args.host, args.backend_port),
                timeout=args.startup_timeout,
                process=backend,
            )
            backend_ready = True
            backend_started_ms = int((time.perf_counter() - started) * 1000)

            frontend_env = env.copy()
            frontend_env["KINGPRO_API_URL"] = "http://{0}:{1}".format(
                args.host, args.backend_port
            )
            npm = "npm.cmd" if os.name == "nt" else "npm"
            with frontend_log_path.open("wb") as frontend_log:
                frontend = start_process(
                    [
                        npm,
                        "run",
                        "start",
                        "--",
                        "--hostname",
                        args.host,
                        "--port",
                        str(args.frontend_port),
                    ],
                    cwd=ROOT / "frontend",
                    env=frontend_env,
                    log=frontend_log,
                )
                wait_for_json(
                    "http://{0}:{1}/api/health".format(
                        args.host, args.frontend_port
                    ),
                    timeout=args.startup_timeout,
                    process=frontend,
                )
                frontend_ready = True
                frontend_started_ms = int((time.perf_counter() - started) * 1000)
                health = inspect_health(
                    "http://{0}:{1}/health".format(args.host, args.backend_port),
                    timeout=5.0,
                )
                smoke = run_smoke(
                    "http://{0}:{1}".format(args.host, args.frontend_port),
                    args.request_timeout,
                    args.budget_seconds,
                )
    except Exception as exc:  # report and clean up every cold-start failure
        failure = "{0}:{1}".format(type(exc).__name__, str(exc))
    finally:
        cleanup = {
            "frontend": stop_owned_process(frontend),
            "backend": stop_owned_process(backend),
        }

    # Give Windows a brief interval to release listening sockets after the
    # child process tree has exited.
    deadline = time.perf_counter() + 5.0
    ports_free_after = False
    while time.perf_counter() < deadline:
        ports_free_after = bool(
            port_available(args.host, args.backend_port)
            and port_available(args.host, args.frontend_port)
        )
        if ports_free_after:
            break
        time.sleep(0.1)

    report = build_report(
        ports_free_before=ports_free_before,
        backend_ready=backend_ready,
        frontend_ready=frontend_ready,
        health=health,
        smoke=smoke,
        cleanup=cleanup,
        ports_free_after=ports_free_after,
    )
    report.update(
        {
            "generated_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(
                timespec="seconds"
            ),
            "host": args.host,
            "backend_port": args.backend_port,
            "frontend_port": args.frontend_port,
            "backend_ready_ms": backend_started_ms,
            "frontend_ready_ms": frontend_started_ms,
            "total_elapsed_ms": int((time.perf_counter() - started) * 1000),
            "failure": failure,
            "logs": {
                "backend": str(backend_log_path.relative_to(ROOT)).replace("\\", "/"),
                "frontend": str(frontend_log_path.relative_to(ROOT)).replace("\\", "/"),
                "backend_sha256": sha256_file(backend_log_path),
                "frontend_sha256": sha256_file(frontend_log_path),
            },
        }
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["cold_start_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
