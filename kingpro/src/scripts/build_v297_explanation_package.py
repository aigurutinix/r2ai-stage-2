"""Build the single canonical V297 explanation/source/data/model package."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "deliverables" / "KINGPRO_V297_FINAL"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from build_demo_evidence_bundle import secret_hits  # noqa: E402
from kingpro.submission.archive import write_deterministic  # noqa: E402


V297_ZIP_SHA = "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC"
V297_JSON_SHA = "E19E4748DDAFFAD28112292EAE17E9296F85BBC99265C9D52EEB27E8F8539C85"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def safe_file(source: Path, arcname: str) -> dict:
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = source.read_bytes()
    hits = secret_hits(arcname, payload)
    if hits:
        raise ValueError(f"credential scan blocked {arcname}: {hits}")
    return {"source": source, "path": arcname.replace("\\", "/"), "sha256": sha256(source), "bytes": len(payload)}


def deterministic_zip(
    output: Path,
    entries: Iterable[tuple[Path, str]],
    *,
    metadata: dict,
    manifest_name: str = "MANIFEST.json",
) -> dict:
    rows = [safe_file(source, arcname) for source, arcname in entries]
    paths = [row["path"] for row in rows]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate archive path")
    manifest = {**metadata, "files": [{key: row[key] for key in ("path", "sha256", "bytes")} for row in sorted(rows, key=lambda item: item["path"])]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", delete=False) as handle:
        manifest_path = Path(handle.name)
        handle.write(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".zip",
        prefix=f".{output.name}.",
        dir=output.parent,
        delete=False,
    ) as archive_handle:
        staging_archive = Path(archive_handle.name)
    try:
        with zipfile.ZipFile(staging_archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            write_deterministic(archive, manifest_path, manifest_name)
            for row in sorted(rows, key=lambda item: item["path"]):
                write_deterministic(archive, row["source"], row["path"])
        with zipfile.ZipFile(staging_archive) as archive:
            names = archive.namelist()
            if archive.testzip() is not None or len(names) != len(set(names)):
                raise RuntimeError("archive CRC/duplicate verification failed")
        staging_archive.replace(output)
    finally:
        manifest_path.unlink(missing_ok=True)
        staging_archive.unlink(missing_ok=True)
    return {"path": output.name, "sha256": sha256(output), "bytes": output.stat().st_size, "entries": len(rows) + 1}


def source_entries() -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    excluded_test_fixtures = {
        "tests/test_demo_evidence_bundle.py",
        "tests/test_experiment_logbook.py",
        "tests/test_product_governance.py",
    }
    for folder, suffixes in (
        (ROOT / "src", {".py"}),
        (ROOT / "scripts", {".py", ".sh", ".ps1"}),
        (ROOT / "tests", {".py"}),
        (ROOT / "config", {".json", ".yaml", ".yml"}),
        (ROOT / "configs", {".json", ".yaml", ".yml"}),
        (ROOT / "docker", {".py", ".sh", ".dockerfile"}),
    ):
        for path in folder.rglob("*"):
            relative = path.relative_to(ROOT).as_posix() if path.is_file() else ""
            if (
                path.is_file()
                and path.suffix.casefold() in suffixes
                and "__pycache__" not in path.parts
                and relative not in excluded_test_fixtures
            ):
                entries.append((path, relative))
    frontend_roots = (
        ROOT / "frontend" / "app",
        ROOT / "frontend" / "components",
        ROOT / "frontend" / "lib",
        ROOT / "frontend" / "public",
    )
    for folder in frontend_roots:
        for path in folder.rglob("*"):
            if path.is_file():
                entries.append((path, path.relative_to(ROOT).as_posix()))
    for name in (
        "README.md",
        "requirements.txt",
        "requirements-llm.txt",
        "pytest.ini",
        ".env.example",
        ".env.docker.example",
        ".dockerignore",
        "compose.yaml",
        "frontend/package.json",
        "frontend/package-lock.json",
        "frontend/components.json",
        "frontend/next.config.ts",
        "frontend/tsconfig.json",
        "frontend/postcss.config.mjs",
    ):
        source = ROOT / name
        arcname = {
            ".env.example": "env.example",
            ".env.docker.example": "env.docker.example",
        }.get(name, name)
        entries.append((source, arcname))
    for name in (
        "DATA_CARD.md",
        "MODEL_CARD.md",
        "REPRODUCIBILITY.md",
        "COMPLIANCE_CHECKLIST.md",
        "DEMO_DAY_COMPLIANCE.md",
        "PRIVATE_TEST_FINAL_CHECKLIST_V297.md",
        "V297_SUBMISSION_HANDOFF.md",
        "RUNTIME_ATTESTATION.md",
        "PRODUCT_PROFILE.md",
        "AUDIT_FACTORY.md",
        "SILENT_ERROR_ASSURANCE.md",
        "DOCKER_DEPLOYMENT.md",
        "TOP3_STAGE1_REPO_LESSONS_20260828.md",
        "PRODUCT_READINESS_FROM_VAIC_DDAY_20260828.md",
        "SECURITY_AND_GOVERNANCE.md",
    ):
        entries.append((ROOT / "docs" / name, f"docs/{name}"))
    for name in ("SOURCE_AND_DEPLOYMENT.md", "SUBMISSION_PROVENANCE.md", "LINKS.md"):
        entries.append((FINAL_DIR / name, f"docs/{name}"))
    return entries


def build_components() -> list[dict]:
    doctor = json.loads((ROOT / "build" / "private_ready_doctor.json").read_text(encoding="utf-8"))
    if doctor.get("status") != "PASS" or not all(doctor.get("checks", {}).values()):
        raise RuntimeError("private-ready doctor is not fully PASS")
    if sha256(ROOT / "sub_v297_scope2_a.zip") != V297_ZIP_SHA:
        raise RuntimeError("V297 competition ZIP drift")
    if sha256(ROOT / "sub_v297_scope2" / "submission.json") != V297_JSON_SHA:
        raise RuntimeError("V297 submission.json drift")

    source_zip = FINAL_DIR / "KINGPRO_V297_SOURCE.zip"
    data_zip = FINAL_DIR / "KINGPRO_V297_DATA_PROVENANCE.zip"
    model_zip = FINAL_DIR / "KINGPRO_V297_MODEL_DOSSIER.zip"
    components = [
        deterministic_zip(
            source_zip,
            source_entries(),
            metadata={
                "schema_version": "kingpro-v297-source/v1",
                "canonical_version": "V297",
                "secret_scan": "PASS",
                "excluded": [
                    ".env",
                    "credentials",
                    "node_modules",
                    ".next",
                    "venv",
                    "raw corpus",
                    "historical candidate payloads",
                    "tests/test_demo_evidence_bundle.py (deliberate fake-token scanner fixture)",
                    "tests/test_experiment_logbook.py (deliberate fake-token scanner fixture)",
                    "tests/test_product_governance.py (deliberate fake-token and PII scanner fixture)",
                ],
            },
        ),
        deterministic_zip(
            data_zip,
            [
                (ROOT / "sub_v297_scope2_a.zip", "submission/sub_v297_scope2_a.zip"),
                (ROOT / "sub_v297_scope2" / "source_audit.json", "provenance/source_audit.json"),
                (ROOT / "build" / "runtime_lineage" / "sub_v297_scope2_counterfactual.json", "provenance/counterfactual_lineage.json"),
                (ROOT / "build" / "demo_compliance" / "runtime_cell_lineage_coverage.json", "provenance/runtime_cell_lineage_coverage.json"),
                (ROOT / "build" / "release_gate" / "sub_v297_scope2" / "release.json", "provenance/v297_release_gate.json"),
                (ROOT / "build" / "private_ready_doctor.json", "provenance/private_ready_doctor.json"),
                (ROOT / "data" / "questions" / "questions.jsonl", "sample/questions.jsonl"),
                (ROOT / "data" / "code_stock.csv", "sample/code_stock.csv"),
                (ROOT / "docs" / "DATA_CARD.md", "DATA_CARD.md"),
                (FINAL_DIR / "DATA_ACCESS.md", "DATA_ACCESS.md"),
                (FINAL_DIR / "SUBMISSION_PROVENANCE.md", "SUBMISSION_PROVENANCE.md"),
                (FINAL_DIR / "LINKS.md", "LINKS.md"),
            ],
            metadata={
                "schema_version": "kingpro-v297-data-provenance/v1",
                "canonical_version": "V297",
                "dataset": "AIGuruTinix/ViFinQA",
                "dataset_revision": "0450088ab22ec946f04f097586967ca405955b3b",
                "raw_corpus_included": False,
                "raw_corpus_access": "https://huggingface.co/datasets/AIGuruTinix/ViFinQA/tree/0450088ab22ec946f04f097586967ca405955b3b",
            },
        ),
        deterministic_zip(
            model_zip,
            [
                (ROOT / "docs" / "MODEL_CARD.md", "MODEL_CARD.md"),
                (ROOT / "docs" / "RUNTIME_ATTESTATION.md", "RUNTIME_ATTESTATION.md"),
                (FINAL_DIR / "MODEL_ACCESS.md", "MODEL_ACCESS.md"),
                (FINAL_DIR / "LINKS.md", "LINKS.md"),
                (ROOT / "requirements-llm.txt", "requirements-llm.txt"),
                (ROOT / ".env.example", "env.example"),
            ],
            metadata={
                "schema_version": "kingpro-v297-model/v1",
                "canonical_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
                "revision": "aedcc2d42b622764e023cf882b6652e646b95671",
                "license": "Apache-2.0",
                "checkpoint_weights_included": False,
                "checkpoint_access": "https://huggingface.co/Qwen/Qwen2.5-Coder-14B-Instruct/tree/aedcc2d42b622764e023cf882b6652e646b95671",
            },
        ),
    ]
    return components


def write_outer_manifest(components: list[dict]) -> Path:
    core_names = (
        "README.md",
        "LINKS.md",
        "DATA_ACCESS.md",
        "MODEL_ACCESS.md",
        "SOURCE_AND_DEPLOYMENT.md",
        "SUBMISSION_PROVENANCE.md",
        "KINGPRO_V297_THUYET_MINH_SAN_PHAM.md",
        "KINGPRO_V297_THUYET_MINH_SAN_PHAM.pdf",
    )
    files = []
    for name in core_names:
        path = FINAL_DIR / name
        files.append({"path": name, "sha256": sha256(path), "bytes": path.stat().st_size})
    files.extend({key: item[key] for key in ("path", "sha256", "bytes")} for item in components)
    manifest = {
        "schema_version": "kingpro-v297-explanation/v1",
        "canonical_version": "V297",
        "submission_id": 3747,
        "competition_file": {"name": "sub_v297_scope2_a.zip", "sha256": V297_ZIP_SHA, "included_in_outer_bundle": False},
        "files": sorted(files, key=lambda item: item["path"]),
        "operator_warning": "The explanation ZIP is documentation. Upload only sub_v297_scope2_a.zip to the scorer.",
    }
    path = FINAL_DIR / "MANIFEST.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def final_entries() -> list[tuple[Path, str]]:
    return [
        (path, path.relative_to(FINAL_DIR).as_posix())
        for path in FINAL_DIR.iterdir()
        if path.is_file()
    ]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--prepare-components", action="store_true")
    parser.add_argument("--copy-canonical", action="store_true")
    args = parser.parse_args()
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    if args.prepare_components:
        components = build_components()
        write_outer_manifest(components)
    else:
        required = [FINAL_DIR / name for name in ("KINGPRO_V297_SOURCE.zip", "KINGPRO_V297_DATA_PROVENANCE.zip", "KINGPRO_V297_MODEL_DOSSIER.zip", "MANIFEST.json")]
        if any(not path.is_file() for path in required):
            raise FileNotFoundError("run once with --prepare-components")
    result = deterministic_zip(
        args.archive.resolve(),
        final_entries(),
        metadata={
            "schema_version": "kingpro-v297-explanation-outer/v1",
            "canonical_version": "V297",
            "purpose": "BTC product explanation and acceptance dossier",
            "not_a_scorer_submission": True,
        },
        manifest_name="ARCHIVE_MANIFEST.json",
    )
    canonical = ROOT / "deliverables" / "KINGPRO_V297_THUYET_MINH_FINAL.zip"
    if args.copy_canonical:
        if canonical.exists():
            raise FileExistsError(canonical)
        shutil.copyfile(args.archive.resolve(), canonical)
        result["canonical_copy"] = {"path": str(canonical), "sha256": sha256(canonical)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
