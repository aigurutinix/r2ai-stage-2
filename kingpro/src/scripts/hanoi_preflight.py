"""One-command, fail-closed preflight for the Hanoi on-site demo.

This script validates the preserved competition artifacts, the running product,
the latest readiness report and the credential-safe public evidence bundle. It
returns success when the audited offline/replay stage path is safe. Full dynamic
readiness remains a separate, stricter field and is never inferred.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    {
        "version": "v206",
        "role": "preserved_predecessor",
        "name": "sub_top123_candidate_v206_semantic_batch11.zip",
        "sha256": "34B19414AA1610785163513F8D2F16CCD41A1F797A90F7121817657F553E4B5B",
        "registry_entries": 1012,
    },
    {
        "version": "v207",
        "role": "preserved_former_champion",
        "name": "sub_top123_candidate_v207_semantic_batch6_final.zip",
        "sha256": "DD616AA408E4B601B921246DDFF4DA4BFD4E8C283FCEF22FF68680AAAB2ADDAF",
        "registry_entries": 1012,
    },
    {
        "version": "v217",
        "role": "preserved_former_champion",
        "name": "sub_top123_candidate_v217_missing_panel_operand_batch3.zip",
        "sha256": "63BA7D1982815D8BA89983A86078D1F1B31FD6CC571B492609BA75244C46E9AC",
        "registry_entries": 1012,
    },
    {
        "version": "v297",
        "role": "public_champion",
        "name": "sub_v297_scope2_a.zip",
        "sha256": "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC",
        "registry_entries": 1012,
    },
    {
        "version": "v290",
        "role": "direct_measured_rollback_fallback",
        "name": "sub_v290_scope2_a.zip",
        "sha256": "711A3493279387593ECA17C3C4130154E9F860CA891B91013083E7D76E5C04A4",
        "registry_entries": 1012,
    },
    {
        "version": "v276",
        "role": "secondary_measured_rollback_fallback",
        "name": "sub_v276_q638_fix.zip",
        "sha256": "86A52DA0FE9A9121C6BB08191FDC2FABFDD97620020B9F3F46DAB81B2258478E",
        "registry_entries": 1012,
    },
    {
        "version": "v218",
        "role": "measured_successor_tie_complete_vector",
        "name": "sub_top123_candidate_v218_existing_table_completeness_batch4.zip",
        "sha256": "F0C7667D8CB222A120092834DEE553637582E12746BC25BA614E2F3BAA6F4C18",
        "registry_entries": 1012,
    },
    {
        "version": "v225",
        "role": "audited_measured_nonchampion_candidate",
        "name": "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9.zip",
        "sha256": "9E452A3814CB5DA5C62493B5B902ABF5E42F78D4ABC745F270F7E00A6CDF8888",
        "registry_entries": 1012,
    },
    {
        "version": "v265",
        "role": "rejected_measured_retrieval_ablation",
        "name": "sub_v265_q24_ret_ablation.zip",
        "sha256": "A4B96E0C9695DAB0BA12514FA067ACCCC2CA0237DA525E759330422A4335C511",
        "registry_entries": 1012,
    },
    {
        "version": "v269",
        "role": "measured_source_clean_fallback",
        "name": "sub_v269_lineage_control.zip",
        "sha256": "C933B5D901836C1414DAB801DAAA77BBA8654BC64572F208A8D8730959B935A0",
        "registry_entries": 1012,
    },
)
SENSITIVE_MARKERS = (
    ".env",
    "api_key",
    "apikey",
    "credential",
    "private_evidence",
    "secret",
    "token",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def inspect_artifact(spec: dict[str, Any], root: Path = ROOT) -> dict[str, Any]:
    path = root / spec["name"]
    report: dict[str, Any] = {
        "version": spec["version"],
        "role": spec["role"],
        "name": spec["name"],
        "exists": path.is_file(),
        "sha256": None,
        "hash_matches": False,
        "zip_readable": False,
        "zip_entries": 0,
        "registry_entries": 0,
        "registry_count_matches": False,
        "ok": False,
    }
    if not path.is_file():
        return report
    report["sha256"] = sha256_file(path)
    report["hash_matches"] = report["sha256"] == spec["sha256"]
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            report["zip_entries"] = len(names)
            bad_entry = archive.testzip()
            if bad_entry is not None or "submission.json" not in names:
                return report
            registry = json.loads(archive.read("submission.json"))
            report["zip_readable"] = isinstance(registry, list)
            report["registry_entries"] = len(registry) if isinstance(registry, list) else 0
            report["registry_count_matches"] = (
                report["registry_entries"] == spec["registry_entries"]
            )
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError):
        return report
    report["ok"] = bool(
        report["hash_matches"]
        and report["zip_readable"]
        and report["registry_count_matches"]
    )
    return report


def get_json(url: str, timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("expected JSON object")
    return payload


def inspect_health(health_url: str, timeout: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "url": health_url,
        "reachable": False,
        "truth_matches": False,
        "fallback_fail_closed": False,
        "issues": [],
        "ok": False,
    }
    try:
        health = get_json(health_url, timeout)
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError) as exc:
        report["issues"].append(type(exc).__name__)
        return report
    report["reachable"] = True
    champion = health.get("public_champion") or {}
    scores = champion.get("scores") or {}
    successor = health.get("local_successor") or {}
    audited = health.get("audited_candidate") or {}
    source_clean = health.get("source_clean_fallback") or {}
    rollback = health.get("rollback_fallback") or {}
    measured_rollback = health.get("measured_rollback_fallback") or {}
    retrieval = health.get("document_retrieval") or {}
    retrieval_regression = retrieval.get("source_bound_regression") or {}
    expected = {
        "status": health.get("status") == "ok",
        "replay_artifact": health.get("replay_artifact") == "sub_v297_scope2",
        "replay_entries": health.get("replay_entries") == 1012,
        "champion_version": champion.get("version") == "v297",
        "champion_submission": champion.get("submission_id") == 3747,
        "champion_artifact": (
            champion.get("artifact") == "sub_v297_scope2"
            and champion.get("artifact_sha256")
            == "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC"
        ),
        "champion_measured": champion.get("on_leaderboard") is True,
        "champion_release": champion.get("release_gate") == "PASS",
        "champion_source_proof": (
            (champion.get("release_proof") or {}).get("source_correct_answer_ids")
            == [98, 224, 764, 966]
            and (champion.get("release_proof") or {}).get(
                "excluded_answer_change_ids"
            )
            == [714]
            and (
                (champion.get("release_proof") or {}).get(
                    "q98_measured_retrieval_union"
                )
                or {}
            ).get("physical_answer_table")
            == "HUT_financial_statements_2024_consolidated|325"
            and (champion.get("release_proof") or {}).get(
                "q764_physical_answer_table"
            )
            == "DPM_financial_statements_2015_consolidated|1909"
            and (
                (champion.get("release_proof") or {}).get(
                    "q224_measured_retrieval_union"
                )
                or {}
            ).get("physical_answer_table")
            == "HUT_financial_statements_2024_separate|1624"
            and (champion.get("release_proof") or {}).get(
                "q966_physical_answer_table"
            )
            == "GEG_financial_statements_2025_consolidated|493"
        ),
        "execution_score": scores.get("execution_accuracy") == 0.7115,
        "answer_score": scores.get("answer_accuracy") == 0.7115,
        "tables_f2_score": scores.get("tables_f2_macro") == 0.6120,
        "tables_precision_score": scores.get("tables_precision") == 0.5928,
        "tables_recall_score": scores.get("tables_recall") == 0.6261,
        "tables_mrr5_score": scores.get("tables_mrr5") == 0.6514,
        "docs_vector": (
            scores.get("docs_f2_macro") == 0.9618
            and scores.get("docs_precision") == 0.9587
            and scores.get("docs_recall") == 0.9678
            and scores.get("docs_mrr5") == 0.9806
        ),
        "successor_version": successor.get("version") == "v218",
        "successor_measured": successor.get("measured") is True,
        "successor_complete_vector": successor.get("complete_score_vector") is True,
        "successor_execution_tie": (successor.get("scores") or {}).get(
            "execution_accuracy"
        ) == 0.7115,
        "audited_candidate_version": audited.get("version") == "v225",
        "audited_candidate_measured": (
            audited.get("measured") is True
            and audited.get("submission_id") == 3723
            and audited.get("complete_score_vector") is True
            and (audited.get("scores") or {}).get("execution_accuracy") == 0.7095
            and (audited.get("scores") or {}).get("answer_accuracy") == 0.7095
        ),
        "audited_candidate_release": audited.get("release_gate") == "PASS",
        "audited_candidate_coverage": audited.get("individual_audit_questions") == 1012,
        "audited_candidate_hash": audited.get("artifact_sha256")
        == "9E452A3814CB5DA5C62493B5B902ABF5E42F78D4ABC745F270F7E00A6CDF8888",
        "rollback_fallback_version": rollback.get("version") == "v290",
        "rollback_fallback_measured": (
            rollback.get("measured") is True
            and rollback.get("submission_id") == 3745
            and rollback.get("complete_score_vector") is True
            and rollback.get("fallback_priority") == 1
            and (rollback.get("scores") or {}).get("execution_accuracy") == 0.7115
            and (rollback.get("scores") or {}).get("answer_accuracy") == 0.7115
            and (rollback.get("scores") or {}).get("tables_f2_macro") == 0.6114
            and (rollback.get("scores") or {}).get("docs_f2_macro") == 0.9611
        ),
        "rollback_fallback_release": rollback.get("release_gate") == "PASS",
        "rollback_fallback_hash": rollback.get("artifact_sha256")
        == "711A3493279387593ECA17C3C4130154E9F860CA891B91013083E7D76E5C04A4",
        "measured_rollback_fallback_version": measured_rollback.get("version")
        == "v276",
        "measured_rollback_fallback_measured": (
            measured_rollback.get("measured") is True
            and measured_rollback.get("submission_id") == 3742
            and measured_rollback.get("fallback_priority") == 2
            and (measured_rollback.get("scores") or {}).get("execution_accuracy")
            == 0.7115
        ),
        "measured_rollback_fallback_release": measured_rollback.get("release_gate")
        == "PASS",
        "source_clean_fallback_version": source_clean.get("version") == "v269",
        "source_clean_fallback_measured": (
            source_clean.get("measured") is True
            and source_clean.get("submission_id") == 3741
            and source_clean.get("complete_score_vector") is True
            and source_clean.get("fallback_priority") == 3
            and (source_clean.get("scores") or {}).get("execution_accuracy") == 0.7115
            and (source_clean.get("scores") or {}).get("answer_accuracy") == 0.7115
            and (source_clean.get("scores") or {}).get("tables_f2_macro") == 0.6104
            and (source_clean.get("scores") or {}).get("tables_precision") == 0.5912
            and (source_clean.get("scores") or {}).get("tables_recall") == 0.6244
            and (source_clean.get("scores") or {}).get("tables_mrr5") == 0.6514
            and (source_clean.get("scores") or {}).get("docs_f2_macro") == 0.9611
            and (source_clean.get("scores") or {}).get("docs_precision") == 0.9580
            and (source_clean.get("scores") or {}).get("docs_recall") == 0.9672
            and (source_clean.get("scores") or {}).get("docs_mrr5") == 0.9806
        ),
        "source_clean_fallback_release": source_clean.get("release_gate") == "PASS",
        "source_clean_fallback_hash": source_clean.get("artifact_sha256")
        == "C933B5D901836C1414DAB801DAAA77BBA8654BC64572F208A8D8730959B935A0",
        "retrieval_policy_v22": retrieval.get("policy_version") == "v22",
        "retrieval_bounded_series_rule": retrieval.get(
            "bounded_multi_entity_comparative_series_year_backfill"
        ) is True,
        "retrieval_bounded_v22_rules": (
            retrieval.get("ambiguous_long_series_max_reports_per_facet") == 2
            and retrieval.get("implicit_ownership_dual_scope_minimum_ratio") == 1.2
            and retrieval.get("multi_entity_count_scope_fallback") is True
            and retrieval.get("catalog_universe_scan") is True
            and retrieval.get("catalog_universe_cap") == 200
            and retrieval.get("broad_sparse_series_backfill") is False
            and retrieval.get("omitted_year_guessing") is False
        ),
        "retrieval_promotion_pass": (
            retrieval_regression.get("promotion_gate") == "PASS"
            and retrieval_regression.get("questions") == 1012
            and retrieval_regression.get("macro_precision") == 0.975283
            and retrieval_regression.get("macro_recall") == 0.997908
            and retrieval_regression.get("macro_f2") == 0.990582
            and retrieval_regression.get("missed_questions") == 4
            and retrieval_regression.get("claim")
            == "local_source_bound_not_btc_hidden_gold"
        ),
    }
    report["truth_checks"] = expected
    report["truth_matches"] = all(expected.values())
    model_attested = health.get("model_attested") is True
    dynamic_available = health.get("dynamic_generation_available") is True
    report["fallback_fail_closed"] = bool(model_attested or not dynamic_available)
    report["dynamic_generation_available"] = dynamic_available
    report["model_attested"] = model_attested
    report["readiness"] = health.get("demo_readiness") or {}
    report["ok"] = bool(report["truth_matches"] and report["fallback_fail_closed"])
    return report


def inspect_readiness(path: Path, max_age_hours: float) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "exists": path.is_file(),
        "age_hours": None,
        "fresh": False,
        "stage_safe_local": False,
        "full_demo_ready": False,
        "smoke_passed": 0,
        "smoke_total": 0,
        "ok": False,
    }
    if not path.is_file():
        return report
    now = datetime.now().timestamp()
    report["age_hours"] = round((now - path.stat().st_mtime) / 3600, 3)
    report["fresh"] = report["age_hours"] <= max_age_hours
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return report
    smoke = payload.get("stage_smoke") or {}
    report["stage_safe_local"] = payload.get("stage_safe_local") is True
    report["full_demo_ready"] = payload.get("full_demo_ready") is True
    report["smoke_passed"] = int(smoke.get("passed", 0) or 0)
    report["smoke_total"] = int(smoke.get("total", 0) or 0)
    report["smoke_elapsed_ms"] = int(smoke.get("total_elapsed_ms", 0) or 0)
    report["within_budget"] = smoke.get("within_budget") is True
    report["manual_incomplete"] = list(
        (payload.get("manual_evidence") or {}).get("incomplete_keys") or []
    )
    report["runtime_operational"] = bool(
        (payload.get("runtime") or {}).get("operational_ready")
    )
    report["ok"] = bool(
        report["fresh"]
        and report["stage_safe_local"]
        and report["smoke_total"] == 7
        and report["smoke_passed"] == report["smoke_total"]
        and report["within_budget"]
    )
    return report


def inspect_cold_start(path: Path, max_age_hours: float) -> dict[str, Any]:
    """Validate a recent isolated boot/smoke/cleanup drill.

    A live health endpoint can remain green for days while the documented
    restart path silently rots.  This gate therefore consumes the report from
    ``hanoi_cold_start_drill.py`` and requires every boot, truth, smoke and
    cleanup invariant independently.
    """

    report: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "exists": path.is_file(),
        "age_hours": None,
        "fresh": False,
        "cold_start_ok": False,
        "gates": {},
        "ports_free_after": False,
        "failure": None,
        "ok": False,
    }
    if not path.is_file():
        return report
    now = datetime.now().timestamp()
    report["age_hours"] = round((now - path.stat().st_mtime) / 3600, 3)
    report["fresh"] = report["age_hours"] <= max_age_hours
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return report
    gates = payload.get("gates") or {}
    required_gates = (
        "isolated_ports_free_before",
        "backend_http_ready",
        "frontend_http_ready",
        "champion_and_fallback_truth",
        "judge_view_smoke",
        "owned_process_cleanup",
    )
    report["cold_start_ok"] = payload.get("cold_start_ok") is True
    report["gates"] = {key: gates.get(key) is True for key in required_gates}
    report["ports_free_after"] = payload.get("isolated_ports_free_after") is True
    report["failure"] = payload.get("failure")
    report["backend_ready_ms"] = payload.get("backend_ready_ms")
    report["frontend_ready_ms"] = payload.get("frontend_ready_ms")
    report["smoke_elapsed_ms"] = (payload.get("smoke") or {}).get("elapsed_ms")
    report["ok"] = bool(
        report["fresh"]
        and report["cold_start_ok"]
        and all(report["gates"].values())
        and report["ports_free_after"]
        and report["failure"] is None
    )
    return report


def inspect_evidence_bundle(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path),
        "exists": path.is_file(),
        "sha256": None,
        "entries": 0,
        "duplicate_entries": 0,
        "manifest_files_verified": 0,
        "manifest_files_total": 0,
        "sensitive_entries": [],
        "ok": False,
    }
    if not path.is_file():
        return report
    report["sha256"] = sha256_file(path)
    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            report["entries"] = len(names)
            report["duplicate_entries"] = len(names) - len(set(names))
            report["sensitive_entries"] = [
                name
                for name in names
                if any(marker in name.lower() for marker in SENSITIVE_MARKERS)
            ]
            if archive.testzip() is not None or "manifest.json" not in names:
                return report
            manifest = json.loads(archive.read("manifest.json"))
            files = manifest.get("files") or []
            report["manifest_files_total"] = len(files)
            verified = 0
            for entry in files:
                name = entry.get("path") if isinstance(entry, dict) else None
                if not isinstance(name, str) or name not in names:
                    continue
                content = archive.read(name)
                digest = hashlib.sha256(content).hexdigest().upper()
                if digest == entry.get("sha256") and len(content) == entry.get("bytes"):
                    verified += 1
            report["manifest_files_verified"] = verified
            expected_hashes = {
                item["name"]: item["sha256"] for item in ARTIFACTS
            }
            declared = {
                item.get("name"): item.get("sha256")
                for item in (manifest.get("competition_artifacts") or [])
                if isinstance(item, dict)
            }
            report["artifact_hashes_match"] = declared == expected_hashes
    except (OSError, zipfile.BadZipFile, json.JSONDecodeError, UnicodeDecodeError):
        return report
    report["ok"] = bool(
        report["entries"] == report["manifest_files_total"] + 1
        and report["duplicate_entries"] == 0
        and not report["sensitive_entries"]
        and report["manifest_files_total"] >= 17
        and report["manifest_files_verified"] == report["manifest_files_total"]
        and report.get("artifact_hashes_match")
    )
    return report


def build_report(
    *,
    artifact_reports: list[dict[str, Any]],
    health: dict[str, Any],
    readiness: dict[str, Any],
    cold_start: dict[str, Any],
    evidence_bundle: dict[str, Any],
) -> dict[str, Any]:
    truth_checks = health.get("truth_checks") or {}
    artifacts_ok = bool(artifact_reports and all(item["ok"] for item in artifact_reports))
    stage_safe = bool(
        artifacts_ok
        and health["ok"]
        and readiness["ok"]
        and cold_start["ok"]
        and evidence_bundle["ok"]
    )
    fallback_ready = bool(
        stage_safe
        and health["fallback_fail_closed"]
        and not health["dynamic_generation_available"]
    )
    full_ready = bool(stage_safe and readiness["full_demo_ready"])
    return {
        "generated_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(
            timespec="seconds"
        ),
        "venue_city": "Hà Nội",
        "stage_safe": stage_safe,
        "fallback_ready": fallback_ready,
        "full_dynamic_ready": full_ready,
        "gates": {
            "preserved_artifacts": artifacts_ok,
            "live_product_truth": bool(health["ok"]),
            "fresh_stage_readiness": bool(readiness["ok"]),
            "isolated_cold_start_recovery": bool(cold_start["ok"]),
            "public_evidence_bundle": bool(evidence_bundle["ok"]),
        },
        "artifacts": artifact_reports,
        "health": health,
        "readiness": readiness,
        "cold_start": cold_start,
        "evidence_bundle": evidence_bundle,
        "claims_allowed": {
            "public_v297_scores": bool(
                truth_checks.get("champion_version", health.get("truth_matches", False))
                and truth_checks.get("champion_submission", health.get("truth_matches", False))
                and truth_checks.get("tables_f2_score", health.get("truth_matches", False))
            ),
            "measured_v290_direct_rollback_fallback": bool(
                truth_checks.get(
                    "rollback_fallback_measured", health.get("truth_matches", False)
                )
                and truth_checks.get(
                    "rollback_fallback_release", health.get("truth_matches", False)
                )
            ),
            "measured_v276_rollback_fallback": bool(
                truth_checks.get(
                    "measured_rollback_fallback_measured", health.get("truth_matches", False)
                )
                and truth_checks.get(
                    "measured_rollback_fallback_release", health.get("truth_matches", False)
                )
            ),
            "measured_v225_nonchampion_and_local_audit": bool(
                truth_checks.get("audited_candidate_measured", health.get("truth_matches", False))
                and truth_checks.get("audited_candidate_release", health.get("truth_matches", False))
                and truth_checks.get("audited_candidate_coverage", health.get("truth_matches", False))
            ),
            "measured_v269_source_clean_fallback": bool(
                truth_checks.get("source_clean_fallback_measured", health.get("truth_matches", False))
                and truth_checks.get("source_clean_fallback_release", health.get("truth_matches", False))
            ),
            "audited_offline_demo": stage_safe,
            "offline_fallback_ready": fallback_ready,
            "operational_dynamic_model": full_ready,
        },
        "operator_action": (
            "Proceed with audited replay/compiler demo. Keep dynamic mode locked."
            if fallback_ready and not full_ready
            else "Resolve failed gates before presenting."
        ),
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--health-url", default="http://127.0.0.1:8080/health")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--max-readiness-age-hours", type=float, default=24.0)
    parser.add_argument("--max-cold-start-age-hours", type=float, default=24.0)
    parser.add_argument(
        "--readiness-report",
        default="build/demo_compliance/demo_readiness_hanoi_v297_v52.json",
    )
    parser.add_argument(
        "--evidence-bundle",
        default="build/demo_evidence_hanoi_20260827_v225_v2_a.zip",
    )
    parser.add_argument(
        "--cold-start-report",
        default="build/demo_compliance/hanoi_cold_start_20260827_v225_v2.json",
    )
    parser.add_argument(
        "--output", default="build/demo_compliance/hanoi_preflight_v297_latest.json"
    )
    args = parser.parse_args()

    readiness_path = (ROOT / args.readiness_report).resolve()
    evidence_path = (ROOT / args.evidence_bundle).resolve()
    cold_start_path = (ROOT / args.cold_start_report).resolve()
    output_path = (ROOT / args.output).resolve()
    for candidate in (readiness_path, evidence_path, cold_start_path, output_path):
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise SystemExit("all file paths must remain inside the project") from exc

    report = build_report(
        artifact_reports=[inspect_artifact(spec) for spec in ARTIFACTS],
        health=inspect_health(args.health_url, args.timeout),
        readiness=inspect_readiness(readiness_path, args.max_readiness_age_hours),
        cold_start=inspect_cold_start(
            cold_start_path, args.max_cold_start_age_hours
        ),
        evidence_bundle=inspect_evidence_bundle(evidence_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["stage_safe"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
