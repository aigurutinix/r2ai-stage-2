from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from audit_demo_readiness import (  # noqa: E402
    MANUAL_EVIDENCE_KEYS,
    build_readiness_report,
    manual_evidence_summary,
    runtime_summary,
)
from audit_demo_compliance import (  # noqa: E402
    inspect_compiler_mutation_report,
    registry_for_artifact,
    sha256,
)


class DemoReadinessTests(unittest.TestCase):
    def test_compiler_mutation_report_is_bound_to_current_inputs(self) -> None:
        registry = (
            ROOT
            / "sub_top123_candidate_v217_missing_panel_operand_batch3"
            / "submission.json"
        )
        payload = {
            "passed": True,
            "compiled_entries": 100,
            "baseline_replays_passed": 100,
            "mutated_replays_rejected": 100,
            "protected_operand_coordinates": 270,
            "failures": [],
            "input_hashes": {
                "registry_sha256": sha256(registry),
                "compiler_sha256": sha256(
                    ROOT
                    / "src"
                    / "kingpro"
                    / "product"
                    / "deterministic_compiler.py"
                ),
                "sandbox_sha256": sha256(
                    ROOT / "src" / "kingpro" / "answering" / "sandbox.py"
                ),
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "mutation.json"
            import json

            report_path.write_text(json.dumps(payload), encoding="utf-8")
            report = inspect_compiler_mutation_report(
                report_path, registry, root=ROOT
            )
            self.assertTrue(report["validation_passed"])
            payload["input_hashes"]["compiler_sha256"] = "0" * 64
            report_path.write_text(json.dumps(payload), encoding="utf-8")
            report = inspect_compiler_mutation_report(
                report_path, registry, root=ROOT
            )
            self.assertFalse(report["validation_passed"])

    def test_demo_compliance_registry_tracks_selected_artifact(self) -> None:
        artifact = ROOT / "sub_top123_candidate_v206_semantic_batch11.zip"
        registry = registry_for_artifact(artifact)
        self.assertEqual(
            registry,
            ROOT
            / "sub_top123_candidate_v206_semantic_batch11"
            / "submission.json",
        )

    def test_runtime_requires_all_operational_attestations(self) -> None:
        summary = runtime_summary(
            {
                "checks": {"runpod_configuration_attested": True},
                "control_plane_attested": True,
                "runtime_identity_attested": False,
                "dynamic_enable_recommended": True,
            }
        )
        self.assertTrue(summary["configuration_ok"])
        self.assertFalse(summary["operational_ready"])

    def test_manual_evidence_requires_confirmation_and_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = {}
            for key in MANUAL_EVIDENCE_KEYS:
                path = root / "evidence" / (key + ".txt")
                path.parent.mkdir(parents=True, exist_ok=True)
                content = ("proof:" + key).encode("utf-8")
                path.write_bytes(content)
                entries[key] = {
                    "confirmed": True,
                    "evidence_path": path.relative_to(root).as_posix(),
                    "sha256": hashlib.sha256(content).hexdigest().upper(),
                }
            self.assertTrue(
                manual_evidence_summary({"evidence": entries}, root=root)["ok"]
            )
            entries[MANUAL_EVIDENCE_KEYS[0]]["sha256"] = "0" * 64
            summary = manual_evidence_summary({"evidence": entries}, root=root)
            self.assertFalse(summary["ok"])
            self.assertIn(MANUAL_EVIDENCE_KEYS[0], summary["incomplete_keys"])
            self.assertEqual(
                summary["entries"][MANUAL_EVIDENCE_KEYS[0]]["validation_error"],
                "evidence_sha256_mismatch",
            )

    def test_manual_evidence_rejects_path_outside_project(self) -> None:
        entries = {
            key: {
                "confirmed": True,
                "evidence_path": "../outside.txt",
                "sha256": "0" * 64,
            }
            for key in MANUAL_EVIDENCE_KEYS
        }
        with tempfile.TemporaryDirectory() as directory:
            summary = manual_evidence_summary(
                {"evidence": entries}, root=Path(directory)
            )
        self.assertFalse(summary["ok"])
        self.assertTrue(
            all(
                entry["validation_error"] == "evidence_path_escapes_project"
                for entry in summary["entries"].values()
            )
        )

    def test_local_stage_can_be_safe_without_claiming_full_readiness(self) -> None:
        report = build_readiness_report(
            technical={
                "technical_pass": True,
                "process_exit_code": 0,
                "runtime": {"model_attested": False},
            },
            runtime={"configuration_ok": True, "operational_ready": False},
            smoke={"ok": True, "within_budget": True},
            manual={"ok": False},
        )
        self.assertTrue(report["stage_safe_local"])
        self.assertFalse(report["full_demo_ready"])
        self.assertFalse(report["claims_allowed"]["operational_dynamic_model"])

    def test_unattested_runtime_must_remain_fail_closed(self) -> None:
        report = build_readiness_report(
            technical={
                "technical_pass": True,
                "process_exit_code": 0,
                "runtime": {"model_attested": True},
            },
            runtime={"configuration_ok": True, "operational_ready": False},
            smoke={"ok": True, "within_budget": True},
            manual={"ok": True},
        )
        self.assertFalse(report["gates"]["runtime_fail_closed_when_unattested"])
        self.assertFalse(report["stage_safe_local"])

    def test_over_budget_smoke_is_not_stage_safe(self) -> None:
        report = build_readiness_report(
            technical={
                "technical_pass": True,
                "process_exit_code": 0,
                "runtime": {"model_attested": False},
            },
            runtime={"configuration_ok": True, "operational_ready": False},
            smoke={"ok": True, "within_budget": False},
            manual={"ok": False},
        )
        self.assertFalse(report["gates"]["stage_request_budget"])
        self.assertFalse(report["stage_safe_local"])


if __name__ == "__main__":
    unittest.main()
