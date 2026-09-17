"""Build a credential-safe attestation report for the live open-model runtime.

The report intentionally separates control-plane evidence from an inference
identity probe. RunPod endpoint/template metadata can prove what was configured;
only a successful OpenAI-compatible response can prove that the serving route is
currently reachable and reports the same model identifier. No credential,
endpoint ID, user ID or generated text is emitted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]

from _env import load_local_env  # noqa: E402


MODEL_EVIDENCE = {
    "Qwen/Qwen2.5-Coder-14B-Instruct": {
        "release_date": "2024-11-12",
        "parameters_billion": 14.7,
        "non_embedding_parameters_billion": 13.1,
        "license": "Apache-2.0",
        "model_card": "https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct",
        "release_announcement": "https://qwenlm.github.io/blog/qwen2.5-coder-family/",
    }
}

SAFE_MODEL_ENV_KEYS = {
    "MODEL_NAME",
    "MODEL_ID",
    "MODEL_REPO",
    "HF_MODEL_ID",
    "BASE_MODEL",
    "SERVED_MODEL_NAME",
    "QUANTIZATION",
    "DTYPE",
    "MAX_MODEL_LEN",
}


def _request_json(url, api_key, timeout, data=None, method=None):
    headers = {"Authorization": "Bearer " + api_key}
    if data is not None:
        headers["Content-Type"] = "application/json"
        body = json.dumps(data).encode("utf-8")
    else:
        body = None
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return getattr(response, "status", 200), json.loads(
            response.read().decode("utf-8")
        )


def _error_name(exc):
    if isinstance(exc, urllib.error.HTTPError):
        return "HTTPError:{0}".format(exc.code)
    if isinstance(exc, urllib.error.URLError):
        return "URLError:{0}".format(type(exc.reason).__name__)
    return type(exc).__name__


def parse_runpod_endpoint(base_url):
    """Return an endpoint ID internally; callers must never serialize it."""
    try:
        parsed = urlsplit(base_url)
    except (TypeError, ValueError):
        return None
    parts = [part for part in parsed.path.split("/") if part]
    if (
        parsed.scheme != "https"
        or parsed.hostname != "api.runpod.ai"
        or len(parts) < 4
        or parts[0] != "v2"
        or parts[2:4] != ["openai", "v1"]
    ):
        return None
    return parts[1]


def _safe_env(values):
    return {
        key: value
        for key, value in (values or {}).items()
        if key.upper() in SAFE_MODEL_ENV_KEYS
        and isinstance(value, (str, int, float, bool))
    }


def collect_runpod_control_plane(base_url, api_key, configured_model, timeout=20.0):
    endpoint_id = parse_runpod_endpoint(base_url)
    if not endpoint_id:
        return {
            "attempted": False,
            "ok": False,
            "error": "not_a_supported_runpod_openai_url",
            "secrets_exposed": False,
        }
    fingerprint = hashlib.sha256(endpoint_id.encode("utf-8")).hexdigest()[:12]
    try:
        _, endpoints = _request_json(
            "https://rest.runpod.io/v1/endpoints", api_key, timeout
        )
        endpoint = next(
            (row for row in endpoints if str(row.get("id", "")) == endpoint_id),
            None,
        )
        if endpoint is None:
            raise LookupError("configured_endpoint_not_found")
        template_id = str(endpoint.get("templateId") or "")
        if not template_id:
            raise LookupError("template_id_missing")
        _, template = _request_json(
            "https://rest.runpod.io/v1/templates/{0}?includeEndpointBoundTemplates=true"
            .format(template_id),
            api_key,
            timeout,
        )
        health_url = "https://api.runpod.ai/v2/{0}/health".format(endpoint_id)
        _, health = _request_json(health_url, api_key, timeout)
    except Exception as exc:  # credential-safe error classification only
        return {
            "attempted": True,
            "ok": False,
            "endpoint_fingerprint": fingerprint,
            "error": _error_name(exc),
            "secrets_exposed": False,
        }

    endpoint_env = _safe_env(endpoint.get("env"))
    template_env = _safe_env(template.get("env"))
    image = str(template.get("imageName") or "")
    image_tag = image.rsplit(":", 1)[1] if ":" in image else ""
    workers = health.get("workers", {}) if isinstance(health, dict) else {}
    model_matches = (
        endpoint_env.get("MODEL_NAME") == configured_model
        and template_env.get("MODEL_NAME") == configured_model
    )
    immutable_image_tag = bool(image_tag and image_tag.lower() != "latest")
    ready = int(workers.get("ready", 0) or 0)
    unhealthy = int(workers.get("unhealthy", 0) or 0)
    endpoint_workers = endpoint.get("workers") or []
    worker_versions = sorted(
        {
            int(worker.get("version"))
            for worker in endpoint_workers
            if isinstance(worker, dict)
            and isinstance(worker.get("version"), (int, float))
        }
    )
    worker_status_counts = {}
    for worker in endpoint_workers:
        if not isinstance(worker, dict):
            continue
        status = str(
            worker.get("desiredStatus") or worker.get("status") or "UNKNOWN"
        ).upper()
        worker_status_counts[status] = worker_status_counts.get(status, 0) + 1
    pod_inventory = {
        "attempted": True,
        "ok": False,
        "records": 0,
        "desired_status_counts": {},
        "endpoint_versions": [],
        "template_image_matches": 0,
        "error": None,
    }
    try:
        _pod_status, pods = _request_json(
            "https://rest.runpod.io/v1/pods?" + urlencode(
                {"includeWorkers": "true", "endpointId": endpoint_id}
            ),
            api_key,
            timeout,
        )
        scoped_pods = [
            pod for pod in pods
            if isinstance(pod, dict) and str(pod.get("endpointId", "")) == endpoint_id
        ]
        desired_status_counts = {}
        for pod in scoped_pods:
            desired = str(pod.get("desiredStatus") or "UNKNOWN").upper()
            desired_status_counts[desired] = desired_status_counts.get(desired, 0) + 1
        pod_inventory = {
            "attempted": True,
            "ok": True,
            "records": len(scoped_pods),
            "desired_status_counts": desired_status_counts,
            "endpoint_versions": sorted(
                {
                    int(pod["slsVersion"])
                    for pod in scoped_pods
                    if isinstance(pod.get("slsVersion"), (int, float))
                }
            ),
            "template_image_matches": sum(
                str(pod.get("image") or "") == image for pod in scoped_pods
            ),
            "error": None,
        }
    except Exception as exc:
        pod_inventory["error"] = _error_name(exc)
    configuration_ok = bool(model_matches and immutable_image_tag)
    running_pods = int(
        pod_inventory.get("desired_status_counts", {}).get("RUNNING", 0) or 0
    )
    endpoint_version = endpoint.get("version")
    inventory_versions = pod_inventory.get("endpoint_versions", [])
    inventory_current = bool(
        isinstance(endpoint_version, (int, float))
        and int(endpoint_version) in inventory_versions
    )
    worker_inventory_ready = bool(
        pod_inventory.get("ok") and running_pods > 0 and inventory_current
    )
    operational_ready = bool(
        ready > 0 and unhealthy == 0 and worker_inventory_ready
    )
    return {
        "attempted": True,
        "ok": bool(configuration_ok and operational_ready),
        "configuration_ok": configuration_ok,
        "operational_ready": operational_ready,
        "worker_inventory_ready": worker_inventory_ready,
        "endpoint_fingerprint": fingerprint,
        "endpoint_created_at": endpoint.get("createdAt"),
        "endpoint_version": endpoint.get("version"),
        "endpoint_name": endpoint.get("name"),
        "workers_min": endpoint.get("workersMin"),
        "workers_max": endpoint.get("workersMax"),
        "scaler_type": endpoint.get("scalerType"),
        "scaler_value": endpoint.get("scalerValue"),
        "idle_timeout_seconds": endpoint.get("idleTimeout"),
        "flashboot": endpoint.get("flashboot"),
        "gpu_count": endpoint.get("gpuCount"),
        "gpu_types": endpoint.get("gpuTypeIds") or [],
        "execution_timeout_ms": endpoint.get("executionTimeoutMs"),
        "endpoint_worker_records": len(endpoint_workers),
        "endpoint_worker_versions": worker_versions,
        "endpoint_worker_status_counts": worker_status_counts,
        "worker_pod_inventory": pod_inventory,
        "endpoint_model_environment": endpoint_env,
        "template_name": template.get("name"),
        "template_image": image,
        "template_model_environment": template_env,
        "model_environment_matches": model_matches,
        "template_image_has_immutable_tag": immutable_image_tag,
        "health": {
            "jobs": health.get("jobs", {}),
            "workers": workers,
        },
        "error": None,
        "secrets_exposed": False,
    }


def probe_openai_identity(base_url, api_key, configured_model, timeout=30.0):
    """Probe model identity without returning generated content."""
    started = time.monotonic()
    try:
        status, payload = _request_json(
            base_url.rstrip("/") + "/models", api_key, timeout
        )
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        reported = sorted(
            {
                str(row.get("id", "")).strip()
                for row in rows
                if isinstance(row, dict) and str(row.get("id", "")).strip()
            }
        )
        match = configured_model in reported
        return {
            "attempted": True,
            "ok": bool(status == 200 and match),
            "route": "models",
            "http_status": status,
            "configured_model_reported": match,
            "reported_model_ids": reported,
            "latency_seconds": round(time.monotonic() - started, 3),
            "generated_content_exposed": False,
            "error": None,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "route": "models",
            "http_status": None,
            "configured_model_reported": False,
            "reported_model_ids": [],
            "latency_seconds": round(time.monotonic() - started, 3),
            "generated_content_exposed": False,
            "error": _error_name(exc),
        }


def probe_native_worker(
    base_url, api_key, timeout=10.0, probe_window=30.0, poll_interval=2.0
):
    """Submit one minimal native job, expose no content, cancel if still pending."""
    endpoint_id = parse_runpod_endpoint(base_url)
    if not endpoint_id:
        return {
            "attempted": False,
            "ok": False,
            "error": "not_a_supported_runpod_openai_url",
            "generated_content_exposed": False,
            "job_id_exposed": False,
        }
    endpoint_root = "https://api.runpod.ai/v2/{0}".format(endpoint_id)
    started = time.monotonic()
    try:
        status, payload = _request_json(
            endpoint_root + "/run",
            api_key,
            timeout,
            data={
                "input": {
                    "messages": [{"role": "user", "content": "Reply OK."}],
                    "sampling_params": {"temperature": 0, "max_tokens": 1},
                }
            },
        )
        job_id = str(payload.get("id", "")).strip()
        initial_status = str(payload.get("status", "UNKNOWN")).upper()
        if status not in (200, 201) or not job_id:
            raise RuntimeError("native_job_id_missing")
        job_fingerprint = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:12]
        deadline = started + probe_window
        final_status = initial_status
        polls = 0
        while final_status not in {
            "COMPLETED", "FAILED", "ERROR", "TIMED_OUT", "CANCELLED"
        } and time.monotonic() < deadline:
            if poll_interval:
                time.sleep(poll_interval)
            _status_code, job = _request_json(
                endpoint_root + "/status/" + job_id, api_key, timeout
            )
            final_status = str(job.get("status", "UNKNOWN")).upper()
            polls += 1

        cancelled_after_probe = False
        cancel_status = None
        if final_status not in {
            "COMPLETED", "FAILED", "ERROR", "TIMED_OUT", "CANCELLED"
        }:
            _cancel_code, cancelled = _request_json(
                endpoint_root + "/cancel/" + job_id,
                api_key,
                timeout,
                method="POST",
            )
            cancel_status = str(cancelled.get("status", "UNKNOWN")).upper()
            cancelled_after_probe = cancel_status == "CANCELLED"
        return {
            "attempted": True,
            "ok": final_status == "COMPLETED",
            "initial_status": initial_status,
            "final_status": final_status,
            "polls": polls,
            "latency_seconds": round(time.monotonic() - started, 3),
            "job_fingerprint": job_fingerprint,
            "cancelled_after_probe": cancelled_after_probe,
            "cancel_status": cancel_status,
            "generated_content_exposed": False,
            "job_id_exposed": False,
            "error": None,
        }
    except Exception as exc:
        return {
            "attempted": True,
            "ok": False,
            "latency_seconds": round(time.monotonic() - started, 3),
            "generated_content_exposed": False,
            "job_id_exposed": False,
            "error": _error_name(exc),
        }


def build_report(base_url, api_key, model, allowed_models, size_limit_billion,
                 cutoff, timeout=20.0, probe_openai=False, probe_native=False,
                 native_probe_window=30.0):
    evidence = MODEL_EVIDENCE.get(model)
    control_plane = collect_runpod_control_plane(
        base_url, api_key, model, timeout=timeout
    )
    identity = (
        probe_openai_identity(base_url, api_key, model, timeout=timeout)
        if probe_openai
        else {
            "attempted": False,
            "ok": None,
            "configured_model_reported": None,
            "generated_content_exposed": False,
            "error": None,
        }
    )
    native_worker = (
        probe_native_worker(
            base_url,
            api_key,
            timeout=min(timeout, 15.0),
            probe_window=native_probe_window,
        )
        if probe_native
        else {
            "attempted": False,
            "ok": None,
            "generated_content_exposed": False,
            "job_id_exposed": False,
            "error": None,
        }
    )
    released_before_cutoff = bool(
        evidence and date.fromisoformat(evidence["release_date"]) < cutoff
    )
    within_limit = bool(
        evidence and evidence["parameters_billion"] <= size_limit_billion
    )
    checks = {
        "configured_model_allowlisted": bool(model and model in allowed_models),
        "public_model_evidence_known": bool(evidence),
        "released_before_cutoff": released_before_cutoff,
        "within_confirmed_parameter_limit": within_limit,
        "apache_2_license": bool(evidence and evidence["license"] == "Apache-2.0"),
        "runpod_configuration_attested": bool(
            control_plane.get("configuration_ok")
        ),
        "runpod_control_plane_attested": bool(control_plane.get("ok")),
    }
    control_plane_pass = all(checks.values())
    runtime_identity_pass = bool(identity.get("ok"))
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "rule_assumptions": {
            "release_cutoff_exclusive": cutoff.isoformat(),
            "parameter_limit_billion": size_limit_billion,
            "parameter_limit_basis": (
                "configured local threshold; BTC written confirmation must be "
                "attached through the manual-evidence gate"
            ),
        },
        "public_model_evidence": evidence,
        "checks": checks,
        "control_plane_attested": control_plane_pass,
        "runtime_identity_attested": runtime_identity_pass,
        "dynamic_enable_recommended": bool(
            control_plane_pass and runtime_identity_pass
        ),
        "runpod_control_plane": control_plane,
        "openai_identity_probe": identity,
        "native_worker_probe": native_worker,
        "secrets_exposed": False,
        "scope_note": (
            "Control-plane metadata proves the configured endpoint/template/model. "
            "It does not cryptographically prove loaded weight bytes. Keep dynamic "
            "generation fail-closed until the serving route reports the exact model "
            "and the operator retains the RunPod deployment screenshot/revision."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--probe-openai", action="store_true")
    parser.add_argument("--probe-native", action="store_true")
    parser.add_argument("--native-probe-window", type=float, default=30.0)
    parser.add_argument("--size-limit-billion", type=float, default=15.0)
    parser.add_argument("--release-cutoff", default="2026-06-01")
    args = parser.parse_args()

    load_local_env(ROOT)
    base_url = os.environ.get("KINGPRO_LLM_BASE_URL", "").strip()
    api_key = os.environ.get("KINGPRO_LLM_API_KEY", "").strip()
    model = os.environ.get("KINGPRO_LLM_MODEL", "").strip()
    allowed_models = {
        item.strip()
        for item in os.environ.get(
            "KINGPRO_ALLOWED_MODELS",
            "Qwen/Qwen2.5-Coder-14B-Instruct,Qwen/Qwen3-14B",
        ).split(",")
        if item.strip()
    }
    report = build_report(
        base_url,
        api_key,
        model,
        allowed_models,
        args.size_limit_billion,
        date.fromisoformat(args.release_cutoff),
        timeout=args.timeout,
        probe_openai=args.probe_openai,
        probe_native=args.probe_native,
        native_probe_window=args.native_probe_window,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = (ROOT / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["control_plane_attested"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
