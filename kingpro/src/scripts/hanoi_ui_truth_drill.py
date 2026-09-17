"""Boot an isolated full stack, audit Judge View truth, then clean up."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from hanoi_cold_start_drill import port_available, start_process, stop_owned_process, wait_for_json


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=18086)
    parser.add_argument("--frontend-port", type=int, default=13014)
    parser.add_argument("--chrome-port", type=int, default=9344)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument(
        "--ui-output",
        default="build/demo_compliance/demo_ui_truth_20260828_v269_v26.json",
    )
    parser.add_argument(
        "--screenshot",
        default="build/demo_compliance/demo_ui_truth_20260828_v269_v26.png",
    )
    parser.add_argument(
        "--output",
        default="build/demo_compliance/hanoi_ui_truth_drill_20260828_v269_v26.json",
    )
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    ports = [args.backend_port, args.frontend_port, args.chrome_port]
    report = {
        "ports_free_before": all(port_available(args.host, port) for port in ports),
        "backend_ready": False,
        "frontend_ready": False,
        "ui_audit": None,
        "ui_exit_code": None,
        "cleanup": None,
        "ports_free_after": False,
        "failure": None,
        "ok": False,
    }
    if not report["ports_free_before"]:
        report["failure"] = "isolated_port_in_use"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1
    backend = None
    frontend = None
    started = time.perf_counter()
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["KINGPRO_HOST"] = args.host
        env["KINGPRO_PORT"] = str(args.backend_port)
        env["KINGPRO_ALLOWED_ORIGIN"] = f"http://{args.host}:{args.frontend_port}"
        backend_log = output.parent / "ui_truth_backend.log"
        frontend_log = output.parent / "ui_truth_frontend.log"
        with backend_log.open("wb") as log:
            backend = start_process(
                [sys.executable, str(ROOT / "scripts/serve_product.py")],
                cwd=ROOT,
                env=env,
                log=log,
            )
            wait_for_json(
                f"http://{args.host}:{args.backend_port}/health",
                timeout=args.startup_timeout,
                process=backend,
            )
            report["backend_ready"] = True
            frontend_env = env.copy()
            frontend_env["KINGPRO_API_URL"] = f"http://{args.host}:{args.backend_port}"
            npm = "npm.cmd" if os.name == "nt" else "npm"
            with frontend_log.open("wb") as log:
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
                    log=log,
                )
                wait_for_json(
                    f"http://{args.host}:{args.frontend_port}/api/health",
                    timeout=args.startup_timeout,
                    process=frontend,
                )
                report["frontend_ready"] = True
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/audit_demo_ui_truth.py"),
                        "--url",
                        f"http://{args.host}:{args.frontend_port}/?view=demo",
                        "--port",
                        str(args.chrome_port),
                        "--output",
                        args.ui_output,
                        "--screenshot",
                        args.screenshot,
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=90,
                    check=False,
                )
                report["ui_exit_code"] = completed.returncode
                ui_path = (ROOT / args.ui_output).resolve()
                if ui_path.is_file():
                    report["ui_audit"] = json.loads(ui_path.read_text(encoding="utf-8"))
                elif completed.returncode:
                    report["failure"] = (completed.stderr or completed.stdout)[-2000:]
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}:{exc}"
    finally:
        report["cleanup"] = {
            "frontend": stop_owned_process(frontend),
            "backend": stop_owned_process(backend),
        }
    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        if all(port_available(args.host, port) for port in ports):
            report["ports_free_after"] = True
            break
        time.sleep(0.1)
    report["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    ui = report.get("ui_audit") or {}
    report["ok"] = bool(
        report["backend_ready"]
        and report["frontend_ready"]
        and report["ui_exit_code"] == 0
        and ui.get("passed") is True
        and report["cleanup"]["frontend"].get("stopped") is True
        and report["cleanup"]["backend"].get("stopped") is True
        and report["ports_free_after"]
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "backend_ready": report["backend_ready"],
                "frontend_ready": report["frontend_ready"],
                "ui_passed": ui.get("passed"),
                "ui_checks": ui.get("checks"),
                "cleanup": report["cleanup"],
                "ports_free_after": report["ports_free_after"],
                "elapsed_ms": report["elapsed_ms"],
                "failure": report["failure"],
                "output": str(output.relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
