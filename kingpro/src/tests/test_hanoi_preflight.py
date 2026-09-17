from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "scripts"))

from hanoi_preflight import (  # noqa: E402
    ARTIFACTS,
    build_report,
    inspect_artifact,
    inspect_evidence_bundle,
    inspect_health,
    inspect_cold_start,
)
from manage_hanoi_evidence import attach, status  # noqa: E402


def _hash_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest().upper()


class HanoiPreflightTests(unittest.TestCase):
    def test_artifact_requires_hash_crc_and_registry_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "candidate.zip"
            registry = [{"id": index} for index in range(3)]
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("submission.json", json.dumps(registry))
            spec = {
                "version": "test",
                "role": "test",
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper(),
                "registry_entries": 3,
            }
            self.assertTrue(inspect_artifact(spec, root=root)["ok"])
            spec["sha256"] = "0" * 64
            self.assertFalse(inspect_artifact(spec, root=root)["ok"])

    def test_health_requires_v297_truth_and_fail_closed_runtime(self) -> None:
        payload = {
            "status": "ok",
            "replay_artifact": "sub_v297_scope2",
            "replay_entries": 1012,
            "model_attested": False,
            "dynamic_generation_available": False,
            "public_champion": {
                "version": "v297",
                "submission_id": 3747,
                "artifact": "sub_v297_scope2",
                "artifact_sha256": "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC",
                "on_leaderboard": True,
                "release_gate": "PASS",
                "release_proof": {
                    "source_correct_answer_ids": [98, 224, 764, 966],
                    "excluded_answer_change_ids": [714],
                    "q98_measured_retrieval_union": {
                        "physical_answer_table": "HUT_financial_statements_2024_consolidated|325"
                    },
                    "q764_physical_answer_table": "DPM_financial_statements_2015_consolidated|1909",
                    "q224_measured_retrieval_union": {
                        "physical_answer_table": "HUT_financial_statements_2024_separate|1624"
                    },
                    "q966_physical_answer_table": "GEG_financial_statements_2025_consolidated|493",
                },
                "scores": {
                    "execution_accuracy": 0.7115,
                    "answer_accuracy": 0.7115,
                    "tables_f2_macro": 0.612,
                    "tables_precision": 0.5928,
                    "tables_recall": 0.6261,
                    "tables_mrr5": 0.6514,
                    "docs_f2_macro": 0.9618,
                    "docs_precision": 0.9587,
                    "docs_recall": 0.9678,
                    "docs_mrr5": 0.9806,
                },
            },
            "rollback_fallback": {
                "version": "v290",
                "artifact_sha256": "711A3493279387593ECA17C3C4130154E9F860CA891B91013083E7D76E5C04A4",
                "measured": True,
                "submission_id": 3745,
                "complete_score_vector": True,
                "fallback_priority": 1,
                "scores": {
                    "execution_accuracy": 0.7115,
                    "answer_accuracy": 0.7115,
                    "tables_f2_macro": 0.6114,
                    "docs_f2_macro": 0.9611,
                },
                "release_gate": "PASS",
            },
            "measured_rollback_fallback": {
                "version": "v276",
                "artifact_sha256": "86A52DA0FE9A9121C6BB08191FDC2FABFDD97620020B9F3F46DAB81B2258478E",
                "measured": True,
                "submission_id": 3742,
                "complete_score_vector": True,
                "fallback_priority": 2,
                "scores": {"execution_accuracy": 0.7115},
                "release_gate": "PASS"
            },
            "local_successor": {
                "version": "v218",
                "measured": True,
                "complete_score_vector": True,
                "scores": {"execution_accuracy": 0.7115},
            },
            "audited_candidate": {
                "version": "v225",
                "artifact_sha256": "9E452A3814CB5DA5C62493B5B902ABF5E42F78D4ABC745F270F7E00A6CDF8888",
                "measured": True,
                "submission_id": 3723,
                "complete_score_vector": True,
                "scores": {
                    "execution_accuracy": 0.7095,
                    "answer_accuracy": 0.7095,
                },
                "release_gate": "PASS",
                "individual_audit_questions": 1012,
            },
            "source_clean_fallback": {
                "version": "v269",
                "artifact_sha256": "C933B5D901836C1414DAB801DAAA77BBA8654BC64572F208A8D8730959B935A0",
                "measured": True,
                "submission_id": 3741,
                "complete_score_vector": True,
                "fallback_priority": 3,
                "scores": {
                    "execution_accuracy": 0.7115,
                    "answer_accuracy": 0.7115,
                    "tables_f2_macro": 0.6104,
                    "tables_precision": 0.5912,
                    "tables_recall": 0.6244,
                    "tables_mrr5": 0.6514,
                    "docs_f2_macro": 0.9611,
                    "docs_precision": 0.9580,
                    "docs_recall": 0.9672,
                    "docs_mrr5": 0.9806,
                },
                "release_gate": "PASS",
            },
            "document_retrieval": {
                "policy_version": "v22",
                "bounded_multi_entity_comparative_series_year_backfill": True,
                "ambiguous_long_series_max_reports_per_facet": 2,
                "implicit_ownership_dual_scope_minimum_ratio": 1.2,
                "multi_entity_count_scope_fallback": True,
                "catalog_universe_scan": True,
                "catalog_universe_cap": 200,
                "broad_sparse_series_backfill": False,
                "omitted_year_guessing": False,
                "source_bound_regression": {
                    "questions": 1012,
                    "macro_precision": 0.975283,
                    "macro_recall": 0.997908,
                    "macro_f2": 0.990582,
                    "missed_questions": 4,
                    "promotion_gate": "PASS",
                    "claim": "local_source_bound_not_btc_hidden_gold",
                },
            },
        }
        with patch("hanoi_preflight.get_json", return_value=payload):
            report = inspect_health("http://example.invalid/health", 1.0)
        self.assertTrue(report["ok"])
        self.assertTrue(report["fallback_fail_closed"])
        self.assertTrue(report["truth_checks"]["champion_version"])
        self.assertTrue(report["truth_checks"]["champion_submission"])
        self.assertTrue(report["truth_checks"]["champion_artifact"])
        self.assertTrue(report["truth_checks"]["champion_release"])
        self.assertTrue(report["truth_checks"]["champion_source_proof"])
        self.assertTrue(report["truth_checks"]["tables_precision_score"])
        self.assertTrue(report["truth_checks"]["docs_vector"])
        self.assertTrue(report["truth_checks"]["audited_candidate_measured"])
        self.assertTrue(report["truth_checks"]["rollback_fallback_measured"])
        self.assertTrue(report["truth_checks"]["rollback_fallback_release"])
        self.assertTrue(report["truth_checks"]["measured_rollback_fallback_measured"])
        self.assertTrue(report["truth_checks"]["measured_rollback_fallback_release"])
        self.assertTrue(report["truth_checks"]["source_clean_fallback_measured"])
        self.assertTrue(report["truth_checks"]["source_clean_fallback_release"])
        self.assertTrue(report["truth_checks"]["retrieval_policy_v22"])
        self.assertTrue(report["truth_checks"]["retrieval_bounded_v22_rules"])
        self.assertTrue(report["truth_checks"]["retrieval_promotion_pass"])
        payload["dynamic_generation_available"] = True
        with patch("hanoi_preflight.get_json", return_value=payload):
            report = inspect_health("http://example.invalid/health", 1.0)
        self.assertFalse(report["ok"])

    def test_public_bundle_verifies_every_manifest_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "evidence.zip"
            files = []
            payloads = {}
            for index in range(17):
                name = "reports/file_{0:02d}.json".format(index)
                content = json.dumps({"index": index}).encode("utf-8")
                payloads[name] = content
                files.append(
                    {"path": name, "sha256": _hash_bytes(content), "bytes": len(content)}
                )
            manifest = {
                "files": files,
                "competition_artifacts": [
                    {"name": item["name"], "sha256": item["sha256"]}
                    for item in ARTIFACTS
                ],
            }
            with zipfile.ZipFile(path, "w") as archive:
                for name, content in payloads.items():
                    archive.writestr(name, content)
                archive.writestr("manifest.json", json.dumps(manifest))
            report = inspect_evidence_bundle(path)
        self.assertTrue(report["ok"])
        self.assertEqual(report["manifest_files_verified"], 17)

    def test_stage_safe_does_not_imply_dynamic_ready(self) -> None:
        report = build_report(
            artifact_reports=[{"ok": True}],
            health={
                "ok": True,
                "fallback_fail_closed": True,
                "dynamic_generation_available": False,
                "truth_matches": True,
            },
            readiness={"ok": True, "full_demo_ready": False},
            cold_start={"ok": True},
            evidence_bundle={"ok": True},
        )
        self.assertTrue(report["stage_safe"])
        self.assertTrue(report["fallback_ready"])
        self.assertFalse(report["full_dynamic_ready"])

    def test_cold_start_requires_all_gates_and_released_ports(self) -> None:
        required = {
            "isolated_ports_free_before": True,
            "backend_http_ready": True,
            "frontend_http_ready": True,
            "champion_and_fallback_truth": True,
            "judge_view_smoke": True,
            "owned_process_cleanup": True,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cold_start.json"
            path.write_text(
                json.dumps(
                    {
                        "cold_start_ok": True,
                        "gates": required,
                        "isolated_ports_free_after": True,
                        "failure": None,
                        "smoke": {"elapsed_ms": 6000},
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(inspect_cold_start(path, 1.0)["ok"])
            required["owned_process_cleanup"] = False
            path.write_text(
                json.dumps(
                    {
                        "cold_start_ok": True,
                        "gates": required,
                        "isolated_ports_free_after": True,
                        "failure": None,
                    }
                ),
                encoding="utf-8",
            )
            self.assertFalse(inspect_cold_start(path, 1.0)["ok"])

    def test_evidence_attachment_is_hashed_and_explicitly_confirmed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "docs" / "manual.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps(
                    {
                        "evidence": {
                            key: {
                                "confirmed": False,
                                "evidence_path": "",
                                "note": "pending",
                            }
                            for key in (
                                "btc_model_size_confirmation",
                                "runpod_checkpoint_screenshot",
                                "exact_model_card_and_license",
                                "team_and_submission_account_eligibility",
                                "data_rights_inventory",
                                "ip_and_development_tool_disclosure",
                                "winning_artifact_handover_inventory",
                                "timed_pitch_rehearsal",
                            )
                        }
                    }
                ),
                encoding="utf-8",
            )
            source = root / "proof.txt"
            source.write_text("verified evidence", encoding="utf-8")
            result = attach(
                manifest_path=manifest,
                key="timed_pitch_rehearsal",
                source=source,
                note="operator reviewed",
                confirmed=True,
                root=root,
                private_root=root / "private_evidence" / "hanoi",
            )
            summary = status(manifest, root=root)
        self.assertTrue(result["confirmed"])
        self.assertEqual(result["sha256"], _hash_bytes(b"verified evidence"))
        self.assertNotIn("timed_pitch_rehearsal", summary["incomplete_keys"])


if __name__ == "__main__":
    unittest.main()
