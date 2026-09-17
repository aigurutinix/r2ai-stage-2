"""Independent acceptance audit for the canonical V297 explanation package."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from build_demo_evidence_bundle import secret_hits  # noqa: E402


V297_SHA = "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC"
REQUIRED_OUTER = {
    "ARCHIVE_MANIFEST.json",
    "MANIFEST.json",
    "README.md",
    "LINKS.md",
    "DATA_ACCESS.md",
    "MODEL_ACCESS.md",
    "SOURCE_AND_DEPLOYMENT.md",
    "SUBMISSION_PROVENANCE.md",
    "KINGPRO_V297_THUYET_MINH_SAN_PHAM.md",
    "KINGPRO_V297_THUYET_MINH_SAN_PHAM.pdf",
    "KINGPRO_V297_SOURCE.zip",
    "KINGPRO_V297_DATA_PROVENANCE.zip",
    "KINGPRO_V297_MODEL_DOSSIER.zip",
}


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def accessible(url: str) -> bool:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "KINGPRO-v297-audit"})
        with urllib.request.urlopen(request, timeout=20) as response:
            return 200 <= int(response.status) < 400
    except Exception:
        return False


def zip_state(payload: bytes) -> tuple[zipfile.ZipFile, list[str]]:
    archive = zipfile.ZipFile(io.BytesIO(payload))
    names = archive.namelist()
    if archive.testzip() is not None:
        raise RuntimeError("nested archive CRC failure")
    if len(names) != len(set(names)):
        raise RuntimeError("nested archive duplicate path")
    return archive, names


def audit(path: Path) -> dict:
    payload = path.read_bytes()
    outer, names = zip_state(payload)
    name_set = set(names)
    checks: dict[str, bool] = {
        "outer_crc_and_unique_paths": True,
        "required_outer_files": REQUIRED_OUTER.issubset(name_set),
        "outer_entry_count_exact": len(names) == len(REQUIRED_OUTER),
    }
    manifest = json.loads(outer.read("MANIFEST.json"))
    manifest_rows = {row["path"]: row for row in manifest.get("files", [])}
    manifest_ok = True
    for name, row in manifest_rows.items():
        if name not in name_set:
            manifest_ok = False
            continue
        data = outer.read(name)
        if digest(data) != row.get("sha256") or len(data) != int(row.get("bytes", -1)):
            manifest_ok = False
    checks["component_manifest_hashes"] = manifest_ok and len(manifest_rows) == 11

    placeholder_hits: list[str] = []
    secret_findings: list[dict] = []
    for name in names:
        data = outer.read(name)
        if name.endswith((".md", ".json", ".txt")):
            text = data.decode("utf-8", errors="replace")
            if re.search(r"\b(?:TODO|TBD|PENDING|FIXME)\b|CẦN DÁN LINK|example\.com", text, re.I):
                placeholder_hits.append(name)
            hits = secret_hits(name, data)
            if hits:
                secret_findings.append({"path": name, "hits": hits})
    checks["no_placeholders"] = not placeholder_hits
    checks["outer_secret_scan"] = not secret_findings

    source, source_names = zip_state(outer.read("KINGPRO_V297_SOURCE.zip"))
    data_zip, data_names = zip_state(outer.read("KINGPRO_V297_DATA_PROVENANCE.zip"))
    model, model_names = zip_state(outer.read("KINGPRO_V297_MODEL_DOSSIER.zip"))
    checks["source_bundle_has_core"] = all(
        required in source_names
        for required in (
            "MANIFEST.json",
            "README.md",
            "requirements.txt",
            "src/kingpro/product/service.py",
            "scripts/private_ready_doctor.py",
            "frontend/app/page.tsx",
            "frontend/components/ui/button.tsx",
            "compose.yaml",
            "docker/backend.Dockerfile",
            "docker/frontend.Dockerfile",
            "docker/backend_entrypoint.py",
            "scripts/docker_product.ps1",
            "docs/DOCKER_DEPLOYMENT.md",
            "docs/SECURITY_AND_GOVERNANCE.md",
        )
    )
    checks["source_bundle_excludes_runtime_junk"] = not any(
        marker in name.casefold()
        for name in source_names
        for marker in (".env", "node_modules", ".next/", "__pycache__", ".pem", ".key")
    )
    nested_secret_findings: list[dict] = []
    for name in source_names:
        if name.endswith((".py", ".md", ".json", ".ts", ".tsx", ".sh", ".ps1", ".txt", ".yaml", ".yml")):
            hits = secret_hits(name, source.read(name))
            if hits:
                nested_secret_findings.append({"path": name, "hits": hits})
    checks["source_bundle_secret_scan"] = not nested_secret_findings

    checks["data_bundle_has_v297"] = "submission/sub_v297_scope2_a.zip" in data_names
    if checks["data_bundle_has_v297"]:
        checks["nested_v297_hash_locked"] = digest(data_zip.read("submission/sub_v297_scope2_a.zip")) == V297_SHA
    else:
        checks["nested_v297_hash_locked"] = False
    data_manifest = json.loads(data_zip.read("MANIFEST.json"))
    checks["data_revision_locked"] = data_manifest.get("dataset_revision") == "0450088ab22ec946f04f097586967ca405955b3b"
    model_manifest = json.loads(model.read("MANIFEST.json"))
    checks["model_revision_locked"] = (
        model_manifest.get("canonical_model") == "Qwen/Qwen2.5-Coder-14B-Instruct"
        and model_manifest.get("revision") == "aedcc2d42b622764e023cf882b6652e646b95671"
        and model_manifest.get("license") == "Apache-2.0"
    )

    pdf_payload = outer.read("KINGPRO_V297_THUYET_MINH_SAN_PHAM.pdf")
    reader = PdfReader(io.BytesIO(pdf_payload))
    pdf_text = "\n".join(page.extract_text() or "" for page in reader.pages)
    required_pdf_text = (
        "Tóm tắt điều hành",
        "Dữ liệu",
        "Mô hình sử dụng",
        "Mã nguồn và dependencies",
        "Bài nộp và no-hardcode",
        "Reproduce từ máy sạch",
        "sub_v297_scope2_a.zip",
    )
    checks["pdf_valid_and_complete"] = len(reader.pages) >= 7 and all(value in pdf_text for value in required_pdf_text)

    links_text = outer.read("LINKS.md").decode("utf-8")
    urls = re.findall(r"https://[^\s|]+", links_text)
    link_results = {url: accessible(url) for url in urls}
    checks["all_declared_links_accessible"] = bool(link_results) and all(link_results.values())
    checks["outer_marks_not_scorer_submission"] = json.loads(outer.read("ARCHIVE_MANIFEST.json")).get("not_a_scorer_submission") is True
    outer.close()
    source.close()
    data_zip.close()
    model.close()
    return {
        "schema_version": "kingpro-v297-explanation-audit/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "archive": str(path),
        "archive_sha256": digest(payload),
        "archive_bytes": len(payload),
        "checks": checks,
        "placeholder_hits": placeholder_hits,
        "secret_findings": secret_findings,
        "nested_secret_findings": nested_secret_findings,
        "link_results": link_results,
        "pdf_pages": len(reader.pages),
        "operator_instruction": "Send this ZIP for product explanation only; upload sub_v297_scope2_a.zip to the scorer.",
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "archive",
        type=Path,
        nargs="?",
        default=ROOT / "deliverables" / "KINGPRO_V297_THUYET_MINH_FINAL.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "build" / "v297_explanation_package_audit.json",
    )
    args = parser.parse_args()
    report = audit(args.archive.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
