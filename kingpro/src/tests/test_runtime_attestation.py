from __future__ import annotations

import json
import os
import sys
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from audit_runtime_attestation import (  # noqa: E402
    build_report,
    collect_runpod_control_plane,
    parse_runpod_endpoint,
    probe_native_worker,
    probe_openai_identity,
)


class _Response:
    status = 200

    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RuntimeAttestationTests(unittest.TestCase):
    base_url = "https://api.runpod.ai/v2/demo-endpoint/openai/v1"
    model = "Qwen/Qwen2.5-Coder-14B-Instruct"

    def test_runpod_url_parser_is_strict(self):
        self.assertEqual(parse_runpod_endpoint(self.base_url), "demo-endpoint")
        self.assertIsNone(parse_runpod_endpoint("https://api.openai.com/v1"))
        self.assertIsNone(parse_runpod_endpoint("http://api.runpod.ai/v2/x/openai/v1"))

    def test_control_plane_report_matches_endpoint_and_template(self):
        endpoints = [{
            "id": "demo-endpoint",
            "createdAt": "2026-08-01T00:00:00Z",
            "version": 2,
            "name": "vLLM",
            "templateId": "template-1",
            "workersMin": 0,
            "workersMax": 3,
            "gpuCount": 1,
            "gpuTypeIds": ["NVIDIA A40"],
            "executionTimeoutMs": 600000,
            "env": {"MODEL_NAME": self.model},
        }]
        template = {
            "name": "vLLM template",
            "imageName": "registry.example/vllm:immutable123",
            "env": {
                "MODEL_NAME": self.model,
                "DTYPE": "auto",
                "HF_TOKEN": "must-not-leak",
            },
        }
        health = {
            "jobs": {"completed": 10},
            "workers": {"ready": 1, "unhealthy": 0},
        }
        pods = [{
            "endpointId": "demo-endpoint",
            "desiredStatus": "RUNNING",
            "slsVersion": 2,
            "image": "registry.example/vllm:immutable123",
        }]
        with patch(
            "audit_runtime_attestation._request_json",
            side_effect=[
                (200, endpoints),
                (200, template),
                (200, health),
                (200, pods),
            ],
        ):
            report = collect_runpod_control_plane(
                self.base_url, "credential-value-123", self.model, timeout=0.01
            )
        self.assertTrue(report["ok"])
        self.assertTrue(report["configuration_ok"])
        self.assertTrue(report["operational_ready"])
        rendered = json.dumps(report)
        self.assertNotIn("credential-value-123", rendered)
        self.assertNotIn("must-not-leak", rendered)
        self.assertNotIn("demo-endpoint", rendered)

    def test_openai_probe_requires_exact_model_and_never_returns_content(self):
        payload = {"data": [{"id": self.model}], "content": "do not expose"}
        with patch(
            "audit_runtime_attestation._request_json",
            return_value=(200, payload),
        ):
            report = probe_openai_identity(
                self.base_url, "secret", self.model, timeout=0.01
            )
        self.assertTrue(report["ok"])
        self.assertNotIn("do not expose", json.dumps(report))

        with patch(
            "audit_runtime_attestation._request_json",
            side_effect=[
                (200, {"id": "private-job-id", "status": "IN_QUEUE"}),
                (200, {"id": "private-job-id", "status": "COMPLETED", "output": "do not expose"}),
            ],
        ):
            native = probe_native_worker(
                self.base_url,
                "secret",
                timeout=0.01,
                probe_window=1,
                poll_interval=0,
            )
        self.assertTrue(native["ok"])
        rendered = json.dumps(native)
        self.assertNotIn("private-job-id", rendered)
        self.assertNotIn("do not expose", rendered)

    def test_dynamic_enable_requires_control_plane_and_live_identity(self):
        control = {
            "ok": True,
            "configuration_ok": True,
            "secrets_exposed": False,
        }
        identity = {
            "attempted": True,
            "ok": False,
            "generated_content_exposed": False,
        }
        with patch(
            "audit_runtime_attestation.collect_runpod_control_plane",
            return_value=control,
        ), patch(
            "audit_runtime_attestation.probe_openai_identity",
            return_value=identity,
        ):
            report = build_report(
                self.base_url,
                "secret",
                self.model,
                {self.model},
                15.0,
                date(2026, 6, 1),
                timeout=0.01,
                probe_openai=True,
            )
        self.assertTrue(report["control_plane_attested"])
        self.assertFalse(report["runtime_identity_attested"])
        self.assertFalse(report["dynamic_enable_recommended"])
        self.assertIn(
            "must be attached",
            report["rule_assumptions"]["parameter_limit_basis"],
        )
        self.assertNotIn(
            "held by team",
            report["rule_assumptions"]["parameter_limit_basis"],
        )


if __name__ == "__main__":
    unittest.main()
