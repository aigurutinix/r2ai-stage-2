"""Build a deterministic, credential-safe public Demo Day evidence bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.submission.archive import write_deterministic  # noqa: E402


DOSSIER_FILES: tuple[tuple[str, str], ...] = (
    ("README.md", "README.md"),
    ("docs/PRODUCT_PROFILE.md", "dossier/PRODUCT_PROFILE.md"),
    ("docs/COMPLIANCE_CHECKLIST.md", "dossier/COMPLIANCE_CHECKLIST.md"),
    ("docs/REPRODUCIBILITY.md", "dossier/REPRODUCIBILITY.md"),
    ("docs/MODEL_CARD.md", "dossier/MODEL_CARD.md"),
    ("docs/RUNTIME_ATTESTATION.md", "dossier/RUNTIME_ATTESTATION.md"),
    ("docs/DATA_CARD.md", "dossier/DATA_CARD.md"),
    ("docs/DEMO_DAY_COMPLIANCE.md", "dossier/DEMO_DAY_COMPLIANCE.md"),
    ("docs/HANOI_DEMO_HANDOFF.md", "dossier/HANOI_DEMO_HANDOFF.md"),
    (
        "docs/HANOI_EVIDENCE_COLLECTION.md",
        "dossier/HANOI_EVIDENCE_COLLECTION.md",
    ),
    ("docs/V269_MEASURED_HANDOFF.md", "dossier/V269_MEASURED_HANDOFF.md"),
    ("docs/V297_SUBMISSION_HANDOFF.md", "dossier/V297_SUBMISSION_HANDOFF.md"),
    ("docs/SILENT_ERROR_ASSURANCE.md", "dossier/SILENT_ERROR_ASSURANCE.md"),
    (
        "docs/PRIVATE_FINAL_SELECTION_V297.md",
        "dossier/PRIVATE_FINAL_SELECTION_V297.md",
    ),
    ("docs/HANOI_HANDOVER_INVENTORY.md", "dossier/HANOI_HANDOVER_INVENTORY.md"),
    ("docs/HANOI_IP_TOOL_DISCLOSURE.md", "dossier/HANOI_IP_TOOL_DISCLOSURE.md"),
    ("docs/HANOI_DATA_RIGHTS_INVENTORY.md", "dossier/HANOI_DATA_RIGHTS_INVENTORY.md"),
    ("docs/HANOI_5_MINUTE_PITCH.md", "dossier/HANOI_5_MINUTE_PITCH.md"),
)

_BLOCKED_NAME_MARKERS = (".env", "credential", "secret", ".pem", ".key")
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("bearer_token", re.compile(r"\bBearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE)),
    ("provider_token", re.compile(r"\b(?:sk-|hf_|rpa_)[A-Za-z0-9._-]{16,}")),
    (
        "json_api_key_value",
        re.compile(r'"(?:api_key|access_token|secret_key)"\s*:\s*"[^"\r\n]{8,}"', re.IGNORECASE),
    ),
    (
        "environment_secret_value",
        re.compile(r"\b(?:[A-Z0-9_]*API_KEY|[A-Z0-9_]*TOKEN|[A-Z0-9_]*SECRET)\s*=\s*[^\s'\"]{8,}"),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def secret_hits(name: str, payload: bytes) -> list[str]:
    lowered = name.casefold()
    hits = [
        "blocked_filename"
        for marker in _BLOCKED_NAME_MARKERS
        if marker in lowered
    ]
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        return list(dict.fromkeys(hits))
    hits.extend(label for label, pattern in _SECRET_PATTERNS if pattern.search(text))
    return list(dict.fromkeys(hits))


def _resolve_file(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _safe_output_dir(value: str) -> Path:
    target = _resolve_file(value)
    build_root = (ROOT / "build").resolve()
    if target == build_root or build_root not in target.parents:
        raise ValueError("output directory must be a child of the project build directory")
    if target == ROOT or ROOT not in target.parents:
        raise ValueError("output directory must remain inside the project")
    return target


def _copy_entries(entries: Iterable[tuple[Path, str]], output: Path) -> list[dict]:
    manifest_rows: list[dict] = []
    for source, arcname in sorted(entries, key=lambda item: item[1]):
        if not source.is_file():
            raise FileNotFoundError("evidence file not found: {0}".format(source))
        payload = source.read_bytes()
        hits = secret_hits(arcname, payload)
        if hits:
            raise ValueError(
                "credential scan blocked {0}: {1}".format(arcname, ", ".join(hits))
            )
        destination = output / Path(arcname)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(source), str(destination))
        manifest_rows.append(
            {
                "path": arcname.replace("\\", "/"),
                "sha256": sha256(destination),
                "bytes": destination.stat().st_size,
            }
        )
    return manifest_rows


def _write_zip(directory: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            write_deterministic(archive, path, path.relative_to(directory).as_posix())


def build_bundle(
    *,
    output: Path,
    readiness_report: Path,
    runtime_report: Path,
    compiler_report: Path,
    compiler_adversarial_report: Path,
    compiler_schema_report: Path,
    compiler_mutation_report: Path,
    table_reranker_report: Path,
    document_retrieval_report: Path,
    private_proxy_report: Path,
    ui_truth_report: Path,
    cold_start_report: Path,
    release_report: Path,
    candidate_release_report: Path | None,
    public_leaderboard_report: Path,
    artifacts: tuple[Path, ...],
) -> dict:
    archive_path = output.with_suffix(".zip")
    if output.exists() or archive_path.exists():
        raise FileExistsError(
            "refusing to overwrite existing bundle target: {0} or {1}".format(
                output, archive_path
            )
        )
    output.mkdir(parents=True)
    try:
        evidence_entries = [
            *((ROOT / source, target) for source, target in DOSSIER_FILES),
            (readiness_report, "reports/demo_readiness.json"),
            (runtime_report, "reports/runtime_attestation.json"),
            (compiler_report, "reports/deterministic_compiler.json"),
            (compiler_adversarial_report, "reports/compiler_adversarial.json"),
            (compiler_schema_report, "reports/compiler_schema_robustness.json"),
            (compiler_mutation_report, "reports/compiler_registry_mutations.json"),
            (table_reranker_report, "reports/table_reranker_regression.json"),
            (document_retrieval_report, "reports/document_retrieval_regression.json"),
            (private_proxy_report, "reports/private_test_proxy_cohorts.json"),
            (ui_truth_report, "reports/demo_ui_truth.json"),
            (cold_start_report, "reports/hanoi_cold_start.json"),
            (release_report, "reports/submission_release_gate.json"),
            (public_leaderboard_report, "reports/public_leaderboard.json"),
        ]
        if candidate_release_report is not None:
            evidence_entries.append(
                (
                    candidate_release_report,
                    "reports/audited_candidate_release_gate.json",
                )
            )
        files = _copy_entries(evidence_entries, output)
        artifact_rows = []
        for artifact in artifacts:
            if not artifact.is_file():
                raise FileNotFoundError("artifact not found: {0}".format(artifact))
            artifact_rows.append(
                {
                    "name": artifact.name,
                    "sha256": sha256(artifact),
                    "bytes": artifact.stat().st_size,
                    "included_in_bundle": False,
                }
            )
        manifest = {
            "bundle": "KINGPRO R2AI Stage 2 public Demo Day evidence",
        "as_of": "2026-08-28",
            "files": files,
            "competition_artifacts": artifact_rows,
            "manual_evidence_included": False,
            "excluded_on_purpose": [
                "BTC correspondence and identity/account screenshots",
                "environment files, API keys, tokens and third-party credentials",
                "competition ZIP payloads (names and hashes only)",
            ],
            "scope_note": (
                "This bundle proves local technical evidence only. It does not replace "
                "BTC eligibility review, written rule confirmation or live runtime attestation."
            ),
        }
        manifest_path = output / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_hits = secret_hits("manifest.json", manifest_path.read_bytes())
        if manifest_hits:
            raise ValueError("generated manifest failed credential scan")
        _write_zip(output, archive_path)
        manifest["manifest_sha256"] = sha256(manifest_path)
        manifest["archive_sha256"] = sha256(archive_path)
        manifest["archive_bytes"] = archive_path.stat().st_size
        return manifest
    except Exception:
        # The directory was created by this invocation and the exact target was
        # validated inside build/. Clean only that incomplete output.
        if output.is_dir():
            shutil.rmtree(output)
        if archive_path.is_file():
            archive_path.unlink()
        raise


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build/demo_evidence_public")
    parser.add_argument(
        "--readiness-report",
        default="build/demo_compliance/demo_readiness_hanoi_v297_v52.json",
    )
    parser.add_argument(
        "--runtime-report",
        default="build/demo_compliance/runtime_attestation_hanoi_readonly_20260826.json",
    )
    parser.add_argument(
        "--compiler-report",
        default="build/demo_compliance/deterministic_compiler_hanoi_v297_v46.json",
    )
    parser.add_argument(
        "--compiler-adversarial-report",
        default="build/demo_compliance/compiler_adversarial_v21.json",
    )
    parser.add_argument(
        "--compiler-schema-report",
        default="build/demo_compliance/compiler_schema_robustness_v21.json",
    )
    parser.add_argument(
        "--compiler-mutation-report",
        default="build/demo_compliance/compiler_registry_mutations_hanoi_v297_v45.json",
    )
    parser.add_argument(
        "--table-reranker-report",
        default="build/demo_compliance/table_reranker_regression_semantic_v6.json",
    )
    parser.add_argument(
        "--document-retrieval-report",
        default="build/demo_compliance/document_retrieval_v22_promotion.json",
    )
    parser.add_argument(
        "--private-proxy-report",
        default="build/demo_compliance/private_proxy_cohorts_hanoi_v297_v51.json",
    )
    parser.add_argument(
        "--ui-truth-report",
        default="build/demo_compliance/demo_ui_truth_20260828_v297_v52.json",
    )
    parser.add_argument(
        "--cold-start-report",
        default="build/demo_compliance/hanoi_cold_start_20260828_v297_v52.json",
    )
    parser.add_argument(
        "--release-report",
        default=(
            "build/release_gate/sub_v297_scope2/release.json"
        ),
    )
    parser.add_argument(
        "--public-leaderboard-report",
        default="build/leaderboard_after_v297_selected.json",
    )
    parser.add_argument(
        "--candidate-release-report",
        default="build/v225_final_release_gate.json",
        help="local-only audited candidate gate; included separately from replay proof",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[
            "sub_top123_candidate_v206_semantic_batch11.zip",
            "sub_top123_candidate_v207_semantic_batch6_final.zip",
            "sub_top123_candidate_v217_missing_panel_operand_batch3.zip",
            "sub_top123_candidate_v218_existing_table_completeness_batch4.zip",
            "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9.zip",
            "sub_v265_q24_ret_ablation.zip",
            "sub_v269_lineage_control.zip",
            "sub_v276_q638_fix.zip",
            "sub_v290_scope2_a.zip",
            "sub_v297_scope2_a.zip",
        ],
        help="Artifact to hash into the manifest; payload is not copied.",
    )
    args = parser.parse_args()

    report = build_bundle(
        output=_safe_output_dir(args.out),
        readiness_report=_resolve_file(args.readiness_report),
        runtime_report=_resolve_file(args.runtime_report),
        compiler_report=_resolve_file(args.compiler_report),
        compiler_adversarial_report=_resolve_file(args.compiler_adversarial_report),
        compiler_schema_report=_resolve_file(args.compiler_schema_report),
        compiler_mutation_report=_resolve_file(args.compiler_mutation_report),
        table_reranker_report=_resolve_file(args.table_reranker_report),
        document_retrieval_report=_resolve_file(args.document_retrieval_report),
        private_proxy_report=_resolve_file(args.private_proxy_report),
        ui_truth_report=_resolve_file(args.ui_truth_report),
        cold_start_report=_resolve_file(args.cold_start_report),
        release_report=_resolve_file(args.release_report),
        candidate_release_report=(
            _resolve_file(args.candidate_release_report)
            if args.candidate_release_report
            else None
        ),
        public_leaderboard_report=_resolve_file(args.public_leaderboard_report),
        artifacts=tuple(_resolve_file(value) for value in args.artifact),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
