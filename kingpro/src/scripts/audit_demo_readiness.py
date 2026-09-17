"""Fail-closed Demo Day readiness audit.

The existing compliance audit proves repository and submission properties. This
script composes that evidence with the real Judge View HTTP smoke test, the most
recent runtime attestation and human-supplied evidence. It deliberately reports
two different outcomes:

* ``stage_safe_local``: the audited replay/compiler demo can be shown safely;
* ``full_demo_ready``: every technical, operational and manual gate is complete.

The distinction prevents a healthy fallback demo from being misrepresented as
an operational free-form model runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import urllib.error
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from smoke_demo_day import SCENARIOS, check_scenario  # noqa: E402


MANUAL_EVIDENCE_KEYS = (
    "btc_model_size_confirmation",
    "runpod_checkpoint_screenshot",
    "exact_model_card_and_license",
    "team_and_submission_account_eligibility",
    "data_rights_inventory",
    "ip_and_development_tool_disclosure",
    "winning_artifact_handover_inventory",
    "timed_pitch_rehearsal",
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object: {0}".format(path))
    return payload


def _technical_report(
    artifact: Path, compiler_mutation_report: Path | None = None
) -> dict[str, Any]:
    # Release packaging keeps deterministic A/B names while the unpacked
    # registry retains the canonical candidate name. The lower-level audit
    # resolves a registry by ZIP stem, so provide a short-lived canonical ZIP
    # alias for that exact case and remove only the alias created here.
    audit_artifact = artifact
    created_alias: Path | None = None
    if not artifact.with_suffix("").is_dir() and artifact.stem.endswith("_a"):
        canonical_dir = artifact.with_name(artifact.stem[:-2])
        canonical_zip = canonical_dir.with_suffix(".zip")
        if canonical_dir.is_dir() and not canonical_zip.exists():
            with canonical_zip.open("xb") as target, artifact.open("rb") as source:
                shutil.copyfileobj(source, target)
            created_alias = canonical_zip
            audit_artifact = canonical_zip
    command = [
        sys.executable,
        str(ROOT / "scripts" / "audit_demo_compliance.py"),
        "--artifact",
        str(audit_artifact),
    ]
    if compiler_mutation_report is not None:
        command.extend(
            ["--compiler-mutation-report", str(compiler_mutation_report)]
        )
    try:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    finally:
        if created_alias is not None:
            created_alias.unlink(missing_ok=True)
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError("technical audit returned invalid JSON: {0}".format(detail)) from exc
    payload["process_exit_code"] = completed.returncode
    payload["requested_artifact"] = str(artifact.resolve())
    return payload


def run_stage_smoke(
    base_url: str,
    timeout: float,
    budget_seconds: float,
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    transport_error: str | None = None
    import time

    started = time.perf_counter()
    try:
        for scenario in SCENARIOS:
            checks.append(check_scenario(base_url, scenario, timeout))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        transport_error = type(exc).__name__
    passed = sum(bool(item.get("ok")) for item in checks)
    total_elapsed_ms = int((time.perf_counter() - started) * 1000)
    within_budget = total_elapsed_ms <= int(budget_seconds * 1000)
    return {
        "attempted": True,
        "ok": bool(
            transport_error is None
            and len(checks) == len(SCENARIOS)
            and passed == len(SCENARIOS)
        ),
        "base_url": base_url,
        "passed": passed,
        "total": len(SCENARIOS),
        "total_elapsed_ms": total_elapsed_ms,
        "budget_seconds": budget_seconds,
        "within_budget": within_budget,
        "checks": checks,
        "transport_error": transport_error,
    }


def runtime_summary(report: dict[str, Any]) -> dict[str, Any]:
    checks = report.get("checks") or {}
    configuration_ok = bool(checks.get("runpod_configuration_attested"))
    operational_ready = bool(
        report.get("control_plane_attested")
        and report.get("runtime_identity_attested")
        and report.get("dynamic_enable_recommended")
    )
    return {
        "configuration_ok": configuration_ok,
        "operational_ready": operational_ready,
        "control_plane_attested": bool(report.get("control_plane_attested")),
        "runtime_identity_attested": bool(report.get("runtime_identity_attested")),
        "dynamic_enable_recommended": bool(report.get("dynamic_enable_recommended")),
        "model": report.get("model"),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _resolve_evidence_path(value: str, root: Path) -> tuple[Path | None, str | None]:
    raw = Path(value)
    if raw.is_absolute():
        return None, "evidence_path_must_be_project_relative"
    resolved = (root / raw).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError:
        return None, "evidence_path_escapes_project"
    if not resolved.is_file():
        return None, "evidence_file_missing"
    if resolved.stat().st_size <= 0:
        return None, "evidence_file_empty"
    return resolved, None


def manual_evidence_summary(
    manifest: dict[str, Any], *, root: Path = ROOT
) -> dict[str, Any]:
    entries = manifest.get("evidence")
    if not isinstance(entries, dict):
        entries = {}
    missing: list[str] = []
    incomplete: list[str] = []
    normalized: dict[str, dict[str, Any]] = {}
    for key in MANUAL_EVIDENCE_KEYS:
        entry = entries.get(key)
        if not isinstance(entry, dict):
            missing.append(key)
            normalized[key] = {"confirmed": False, "evidence_path": ""}
            continue
        confirmed = entry.get("confirmed") is True
        evidence_path = str(entry.get("evidence_path") or "").strip()
        note = str(entry.get("note") or "").strip()
        declared_sha256 = str(entry.get("sha256") or "").strip().upper()
        resolved_path = None
        evidence_error = None
        actual_sha256 = None
        evidence_bytes = None
        if evidence_path:
            resolved_path, evidence_error = _resolve_evidence_path(evidence_path, root)
            if resolved_path is not None:
                actual_sha256 = _sha256(resolved_path)
                evidence_bytes = resolved_path.stat().st_size
                if declared_sha256 and declared_sha256 != actual_sha256:
                    evidence_error = "evidence_sha256_mismatch"
        if (
            not confirmed
            or not evidence_path
            or evidence_error is not None
            or not declared_sha256
        ):
            incomplete.append(key)
        normalized[key] = {
            "confirmed": confirmed,
            "evidence_path": evidence_path,
            "sha256": declared_sha256,
            "actual_sha256": actual_sha256,
            "bytes": evidence_bytes,
            "validation_error": evidence_error,
            "note": note,
        }
    return {
        "ok": not missing and not incomplete,
        "entries": normalized,
        "missing_keys": missing,
        "incomplete_keys": incomplete,
        "scope_note": (
            "These confirmations are supplied by the team. The audit validates "
            "manifest completeness, not the legal authenticity of external evidence."
        ),
    }


def build_readiness_report(
    *,
    technical: dict[str, Any],
    runtime: dict[str, Any],
    smoke: dict[str, Any],
    manual: dict[str, Any],
) -> dict[str, Any]:
    technical_ok = bool(
        technical.get("technical_pass") and technical.get("process_exit_code") == 0
    )
    model_attested = bool((technical.get("runtime") or {}).get("model_attested"))
    fail_closed_ok = bool(runtime["operational_ready"] or not model_attested)
    gates = {
        "technical_artifact": technical_ok,
        "judge_view_http_smoke": bool(smoke.get("ok")),
        "stage_request_budget": bool(smoke.get("within_budget")),
        "runtime_configuration": bool(runtime["configuration_ok"]),
        "runtime_operational": bool(runtime["operational_ready"]),
        "runtime_fail_closed_when_unattested": fail_closed_ok,
        "manual_evidence_complete": bool(manual["ok"]),
    }
    stage_safe_local = bool(
        gates["technical_artifact"]
        and gates["judge_view_http_smoke"]
        and gates["stage_request_budget"]
        and gates["runtime_configuration"]
        and gates["runtime_fail_closed_when_unattested"]
    )
    full_demo_ready = bool(stage_safe_local and all(gates.values()))
    return {
        "stage_safe_local": stage_safe_local,
        "full_demo_ready": full_demo_ready,
        "gates": gates,
        "technical": technical,
        "runtime": runtime,
        "stage_smoke": smoke,
        "manual_evidence": manual,
        "claims_allowed": {
            "audited_replay_and_compiler_demo": stage_safe_local,
            "operational_dynamic_model": gates["runtime_operational"],
            "fully_demo_ready": full_demo_ready,
        },
        "scope_note": (
            "A local PASS is not a BTC score or final eligibility decision. "
            "Never claim dynamic model availability unless runtime_operational is true."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default="sub_v297_scope2_a.zip",
    )
    parser.add_argument(
        "--runtime-report",
        default="build/demo_compliance/runtime_attestation_hanoi_readonly_20260826.json",
    )
    parser.add_argument(
        "--manual-evidence",
        default="docs/demo_manual_evidence.json",
    )
    parser.add_argument(
        "--compiler-mutation-report",
        default="build/demo_compliance/compiler_registry_mutations_hanoi_v297_v45.json",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:3010")
    parser.add_argument("--smoke-timeout", type=float, default=30.0)
    parser.add_argument(
        "--stage-request-budget",
        type=float,
        default=120.0,
        help="Cumulative seconds available to the scripted HTTP interactions.",
    )
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument(
        "--output",
        default="build/demo_compliance/demo_readiness_latest.json",
    )
    args = parser.parse_args()

    artifact = (ROOT / args.artifact).resolve()
    runtime_path = (ROOT / args.runtime_report).resolve()
    manual_path = (ROOT / args.manual_evidence).resolve()
    compiler_mutation_path = (ROOT / args.compiler_mutation_report).resolve()
    for label, path in (
        ("artifact", artifact),
        ("runtime report", runtime_path),
        ("manual evidence manifest", manual_path),
        ("compiler mutation report", compiler_mutation_path),
    ):
        if not path.is_file():
            raise SystemExit("Missing {0}: {1}".format(label, path))

    technical = _technical_report(artifact, compiler_mutation_path)
    runtime = runtime_summary(_load_json(runtime_path))
    manual = manual_evidence_summary(_load_json(manual_path), root=ROOT)
    smoke = (
        {
            "attempted": False,
            "ok": False,
            "base_url": args.base_url,
            "passed": 0,
            "total": len(SCENARIOS),
            "total_elapsed_ms": 0,
            "budget_seconds": args.stage_request_budget,
            "within_budget": False,
            "checks": [],
            "transport_error": None,
        }
        if args.skip_smoke
        else run_stage_smoke(
            args.base_url,
            args.smoke_timeout,
            args.stage_request_budget,
        )
    )
    report = build_readiness_report(
        technical=technical,
        runtime=runtime,
        smoke=smoke,
        manual=manual,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    output = (ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["full_demo_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
