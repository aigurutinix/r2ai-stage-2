"""Run Hanoi preflight against an owned, temporary live backend.

The ordinary cold-start drill intentionally cleans up before returning.  A
later preflight then correctly reports ``live_product_truth=false``.  This
orchestrator starts only the backend on an isolated port, runs preflight while
health is live, and finally stops the exact owned process tree and proves the
port was released.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from hanoi_cold_start_drill import (
    port_available,
    start_process,
    stop_owned_process,
    wait_for_json,
)


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--backend-port", type=int, default=18083)
    parser.add_argument("--startup-timeout", type=float, default=90.0)
    parser.add_argument(
        "--cold-start-report",
        default="build/demo_compliance/hanoi_cold_start_20260828_v217_v23.json",
    )
    parser.add_argument(
        "--evidence-bundle", default="build/demo_evidence_public_v22_final_a.zip"
    )
    parser.add_argument(
        "--preflight-output",
        default="build/demo_compliance/hanoi_preflight_20260828_v217_v24_live.json",
    )
    parser.add_argument(
        "--output",
        default="build/demo_compliance/hanoi_live_preflight_drill_20260828_v24.json",
    )
    args = parser.parse_args()

    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.parent / "live_preflight_backend.log"
    backend: subprocess.Popen[bytes] | None = None
    started = time.perf_counter()
    report: dict = {
        "generated_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(
            timespec="seconds"
        ),
        "host": args.host,
        "backend_port": args.backend_port,
        "port_free_before": port_available(args.host, args.backend_port),
        "backend_ready": False,
        "preflight_exit_code": None,
        "preflight": None,
        "cleanup": None,
        "port_free_after": False,
        "ok": False,
        "failure": None,
    }
    if not report["port_free_before"]:
        report["failure"] = "isolated_backend_port_in_use"
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 1

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT / "src")
        env["KINGPRO_HOST"] = args.host
        env["KINGPRO_PORT"] = str(args.backend_port)
        with log_path.open("wb") as log:
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
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/hanoi_preflight.py"),
                    "--health-url",
                    f"http://{args.host}:{args.backend_port}/health",
                    "--cold-start-report",
                    args.cold_start_report,
                    "--evidence-bundle",
                    args.evidence_bundle,
                    "--output",
                    args.preflight_output,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=60,
                check=False,
            )
            report["preflight_exit_code"] = completed.returncode
            preflight_path = (ROOT / args.preflight_output).resolve()
            if preflight_path.is_file():
                report["preflight"] = json.loads(preflight_path.read_text(encoding="utf-8"))
            if completed.returncode and report["preflight"] is None:
                report["failure"] = (completed.stderr or completed.stdout)[-2000:]
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}:{exc}"
    finally:
        report["cleanup"] = stop_owned_process(backend)

    deadline = time.perf_counter() + 5.0
    while time.perf_counter() < deadline:
        if port_available(args.host, args.backend_port):
            report["port_free_after"] = True
            break
        time.sleep(0.1)
    preflight = report.get("preflight") or {}
    report["elapsed_ms"] = int((time.perf_counter() - started) * 1000)
    report["ok"] = bool(
        report["backend_ready"]
        and report["preflight_exit_code"] == 0
        and preflight.get("stage_safe") is True
        and preflight.get("fallback_ready") is True
        and (report.get("cleanup") or {}).get("stopped") is True
        and report["port_free_after"]
    )
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "backend_ready": report["backend_ready"],
                "stage_safe": preflight.get("stage_safe"),
                "fallback_ready": preflight.get("fallback_ready"),
                "full_dynamic_ready": preflight.get("full_dynamic_ready"),
                "cleanup": report["cleanup"],
                "port_free_after": report["port_free_after"],
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
