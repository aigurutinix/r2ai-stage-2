"""Create a credential-free technical compliance report for Demo Day.

This audit proves what the repository and submission archive can prove. Human
eligibility, BTC correspondence, remote model weights and third-party licenses
remain manual evidence and are reported separately rather than guessed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _env import load_local_env  # noqa: E402
from audit_document_retrieval_facets import audit as audit_document_retrieval_facets  # noqa: E402
from audit_compiler_adversarial import audit as audit_compiler_adversarial  # noqa: E402
from audit_compiler_schema_robustness import audit as audit_compiler_schema  # noqa: E402
from audit_deterministic_compiler import audit as audit_compiler  # noqa: E402
from audit_runtime_attestation import MODEL_EVIDENCE  # noqa: E402
from audit_retrieval_reranker import audit as audit_retrieval_reranker  # noqa: E402
from kingpro.answering.llm_client import endpoint_compliance  # noqa: E402


REQUIRED_DOSSIER = (
    "README.md",
    "docs/PRODUCT_PROFILE.md",
    "docs/COMPLIANCE_CHECKLIST.md",
    "docs/REPRODUCIBILITY.md",
    "docs/MODEL_CARD.md",
    "docs/RUNTIME_ATTESTATION.md",
    "docs/DATA_CARD.md",
    "docs/DEMO_DAY_COMPLIANCE.md",
    "docs/HANOI_DEMO_HANDOFF.md",
    "docs/HANOI_EVIDENCE_COLLECTION.md",
)
SENSITIVE_NAME_MARKERS = (
    ".env",
    "credential",
    "secret",
    ".pem",
    ".key",
)
SOURCE_SUFFIXES = (".py", ".js", ".ts", ".tsx")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def inspect_compiler_mutation_report(
    report_path: Path,
    registry: Path,
    *,
    root: Path = ROOT,
) -> dict:
    """Validate the expensive real-registry mutation audit without rerunning it.

    Hash binding prevents a stale green report from surviving a change to the
    compiler, sandbox or selected registry.  The full mutation audit remains a
    deliberate release action because it launches two isolated processes for
    each accepted question.
    """
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {
            "passed": False,
            "report_path": str(report_path),
            "validation_error": type(exc).__name__,
        }
    expected_hashes = {
        "registry_sha256": sha256(registry),
        "compiler_sha256": sha256(
            root / "src" / "kingpro" / "product" / "deterministic_compiler.py"
        ),
        "sandbox_sha256": sha256(
            root / "src" / "kingpro" / "answering" / "sandbox.py"
        ),
    }
    actual_hashes = payload.get("input_hashes", {})
    hash_matches = {
        key: actual_hashes.get(key) == value
        for key, value in expected_hashes.items()
    }
    valid = bool(
        payload.get("passed")
        and int(payload.get("compiled_entries", 0)) >= 100
        and payload.get("baseline_replays_passed")
        == payload.get("compiled_entries")
        and payload.get("mutated_replays_rejected")
        == payload.get("compiled_entries")
        and int(payload.get("protected_operand_coordinates", 0)) >= 100
        and not payload.get("failures")
        and all(hash_matches.values())
    )
    return {
        **payload,
        "report_path": str(report_path.resolve()),
        "expected_input_hashes": expected_hashes,
        "input_hash_matches": hash_matches,
        "validation_passed": valid,
    }


def archive_report(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
        archive.testzip()
    lowered = [name.lower() for name in names]
    sensitive = [
        name
        for name, folded in zip(names, lowered)
        if any(marker in folded for marker in SENSITIVE_NAME_MARKERS)
    ]
    source_files = [name for name in names if name.lower().endswith(SOURCE_SUFFIXES)]
    return {
        "path": str(path),
        "sha256": sha256(path),
        "entries": len(names),
        "has_submission_json": "submission.json" in names,
        "sensitive_entries": sensitive,
        "source_code_entries": source_files,
        "safe": bool(
            "submission.json" in names and not sensitive and not source_files
        ),
    }


def registry_for_artifact(artifact: Path) -> Path:
    """Resolve the unpacked registry that must correspond to an artifact ZIP."""
    registry = artifact.with_suffix("") / "submission.json"
    if not registry.is_file():
        raise FileNotFoundError(
            "submission registry matching artifact stem not found: {0}".format(
                registry
            )
        )
    return registry


def probe_model_endpoint(timeout: float = 150) -> dict:
    """Ask the OpenAI-compatible endpoint for model IDs without printing its key."""
    base_url = os.environ.get("KINGPRO_LLM_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("KINGPRO_LLM_API_KEY", "")
    configured_model = os.environ.get("KINGPRO_LLM_MODEL", "")
    request = urllib.request.Request(
        base_url + "/models",
        headers={"Authorization": "Bearer " + api_key},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            status = getattr(response, "status", 200)
        rows = payload.get("data", []) if isinstance(payload, dict) else []
        model_ids = sorted(
            {
                str(row.get("id", "")).strip()
                for row in rows
                if isinstance(row, dict) and str(row.get("id", "")).strip()
            }
        )
        return {
            "attempted": True,
            "ok": bool(status == 200 and configured_model in model_ids),
            "http_status": status,
            "reported_model_ids": model_ids,
            "configured_model_reported": configured_model in model_ids,
            "credential_exposed": False,
            "error": None,
        }
    except urllib.error.HTTPError as exc:
        error = "HTTPError:{0}".format(exc.code)
    except urllib.error.URLError as exc:
        error = "URLError:{0}".format(type(exc.reason).__name__)
    except TimeoutError:
        error = "TimeoutError"
    except OSError as exc:
        error = type(exc).__name__
    except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError) as exc:
        error = type(exc).__name__
    return {
        "attempted": True,
        "ok": False,
        "http_status": None,
        "reported_model_ids": [],
        "configured_model_reported": False,
        "credential_exposed": False,
        "error": error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact",
        default="sub_top123_candidate_v217_missing_panel_operand_batch3.zip",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--probe-model-endpoint",
        action="store_true",
        help="Call GET /models and require the configured model ID in the response.",
    )
    parser.add_argument("--probe-timeout", type=float, default=150.0)
    parser.add_argument(
        "--compiler-mutation-report",
        default=(
            "build/demo_compliance/"
            "compiler_registry_mutations_hanoi_v217_v2.json"
        ),
    )
    args = parser.parse_args()

    load_local_env(ROOT)
    artifact = (ROOT / args.artifact).resolve()
    if not artifact.is_file():
        raise SystemExit("Artifact not found: {0}".format(artifact))
    try:
        registry = registry_for_artifact(artifact)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))

    endpoint = endpoint_compliance()
    model = os.environ.get("KINGPRO_LLM_MODEL", "")
    model_attested = os.environ.get("KINGPRO_LLM_ATTESTED", "").strip().lower() in {
        "1", "true", "yes", "on"
    }
    allowed_models = {
        item.strip()
        for item in os.environ.get(
            "KINGPRO_ALLOWED_MODELS",
            "Qwen/Qwen2.5-Coder-14B-Instruct,Qwen/Qwen3-14B",
        ).split(",")
        if item.strip()
    }
    model_evidence = MODEL_EVIDENCE.get(model)
    size_limit_billion = float(
        os.environ.get("KINGPRO_MODEL_SIZE_LIMIT_BILLION", "15")
    )
    release_cutoff = os.environ.get("KINGPRO_MODEL_RELEASE_CUTOFF", "2026-06-01")
    dossier = {
        relative: (ROOT / relative).is_file() for relative in REQUIRED_DOSSIER
    }
    archive = archive_report(artifact)
    compiler_report = audit_compiler(registry, root=ROOT)
    compiler_adversarial_report = audit_compiler_adversarial(
        root=ROOT,
        registry=registry,
    )
    compiler_schema_report = audit_compiler_schema()
    compiler_mutation_report = inspect_compiler_mutation_report(
        (ROOT / args.compiler_mutation_report).resolve(), registry, root=ROOT
    )
    retrieval_reranker_report = audit_retrieval_reranker()
    document_retrieval_report = audit_document_retrieval_facets()
    technical_checks = {
        "model_identifier_allowlisted": bool(model and model in allowed_models),
        "model_release_before_cutoff": bool(
            model_evidence
            and model_evidence["release_date"] < release_cutoff
        ),
        "model_within_confirmed_parameter_limit": bool(
            model_evidence
            and model_evidence["parameters_billion"] <= size_limit_billion
        ),
        "model_license_allowlisted": bool(
            model_evidence and model_evidence["license"] == "Apache-2.0"
        ),
        "endpoint_host_allowlisted": endpoint["host_allowed"],
        "endpoint_transport_secure": endpoint["secure_transport"],
        "submission_archive_safe": archive["safe"],
        "dossier_present": all(dossier.values()),
        "deterministic_compiler_regression": bool(
            compiler_report["compiled_entries"] >= 99
            and compiler_report["precision_on_compiled_subset"] == 1.0
            and not compiler_report["mismatches"]
            and not compiler_report["provenance_mismatches"]
        ),
        "deterministic_compiler_adversarial": bool(
            compiler_adversarial_report["passed"]
            and not compiler_adversarial_report["positive_failures"]
            and not compiler_adversarial_report["negative_acceptances"]
        ),
        "deterministic_compiler_schema_robustness": bool(
            compiler_schema_report["passed"]
            and compiler_schema_report["case_count"] >= 10
            and compiler_schema_report["failed_count"] == 0
        ),
        "deterministic_compiler_registry_mutations": bool(
            compiler_mutation_report.get("validation_passed")
        ),
        "source_proven_table_reranker_regression": bool(
            retrieval_reranker_report["passed"]
        ),
        "source_bound_document_retrieval_regression": bool(
            document_retrieval_report["passed"]
        ),
    }
    model_probe = (
        probe_model_endpoint(args.probe_timeout)
        if args.probe_model_endpoint
        else {
            "attempted": False,
            "ok": None,
            "configured_model_reported": None,
            "credential_exposed": False,
        }
    )
    if args.probe_model_endpoint:
        technical_checks["configured_model_reported_by_endpoint"] = model_probe["ok"]
    report = {
        "technical_pass": all(technical_checks.values()),
        "technical_checks": technical_checks,
        "runtime": {
            "model": model,
            "endpoint_host": endpoint["host"],
            "endpoint_provider": endpoint["provider"],
            "endpoint_allowed": endpoint["allowed"],
            "model_attested": model_attested,
            "dynamic_generation_available": bool(
                model_attested
                and model in allowed_models
                and endpoint["allowed"]
            ),
            "public_model_evidence": model_evidence,
            "release_cutoff": release_cutoff,
            "confirmed_parameter_limit_billion": size_limit_billion,
            "api_key_present": bool(os.environ.get("KINGPRO_LLM_API_KEY")),
            "api_key_value_exposed": False,
        },
        "remote_model_probe": model_probe,
        "archive": archive,
        "submission_registry": {
            "path": str(registry),
            "artifact_stem_matches": registry.parent.name == artifact.stem,
        },
        "deterministic_compiler": compiler_report,
        "deterministic_compiler_adversarial": compiler_adversarial_report,
        "deterministic_compiler_schema_robustness": compiler_schema_report,
        "deterministic_compiler_registry_mutations": compiler_mutation_report,
        "table_reranker_regression": retrieval_reranker_report,
        "document_retrieval_regression": document_retrieval_report,
        "dossier": dossier,
        "manual_evidence_required": [
            "BTC written confirmation for the applicable model-size threshold",
            "RunPod deployment screenshot proving the loaded checkpoint/revision",
            "model card and license for the exact checkpoint",
            "team/account eligibility and one-approved-submission-account confirmation",
            "data-source rights/license inventory",
            "IP/development-tool disclosure if BTC requests it",
            "winning-artifact source/data handover inventory and secret-scan evidence",
            "one timed five-minute rehearsal including endpoint fallback and Q&A handoff",
        ],
        "scope_note": (
            "A passing client-side audit cannot attest remote model weights, "
            "human eligibility, private leaderboard results or BTC discretion."
        ),
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = (ROOT / args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["technical_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
