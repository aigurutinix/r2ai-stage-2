"""Build a hash-locked, credential-scanned Hanoi handover inventory."""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from build_demo_evidence_bundle import secret_hits, sha256


ROOT = Path(__file__).resolve().parents[1]
LOCKED = {
    "sub_top123_candidate_v206_semantic_batch11.zip": "34B19414AA1610785163513F8D2F16CCD41A1F797A90F7121817657F553E4B5B",
    "sub_top123_candidate_v207_semantic_batch6_final.zip": "DD616AA408E4B601B921246DDFF4DA4BFD4E8C283FCEF22FF68680AAAB2ADDAF",
    "sub_top123_candidate_v217_missing_panel_operand_batch3.zip": "63BA7D1982815D8BA89983A86078D1F1B31FD6CC571B492609BA75244C46E9AC",
    "sub_top123_candidate_v218_existing_table_completeness_batch4.zip": "F0C7667D8CB222A120092834DEE553637582E12746BC25BA614E2F3BAA6F4C18",
    "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9.zip": "9E452A3814CB5DA5C62493B5B902ABF5E42F78D4ABC745F270F7E00A6CDF8888",
    "sub_v265_q24_ret_ablation.zip": "A4B96E0C9695DAB0BA12514FA067ACCCC2CA0237DA525E759330422A4335C511",
    "sub_v269_lineage_control.zip": "C933B5D901836C1414DAB801DAAA77BBA8654BC64572F208A8D8730959B935A0",
    "sub_v276_q638_fix.zip": "86A52DA0FE9A9121C6BB08191FDC2FABFDD97620020B9F3F46DAB81B2258478E",
    "sub_v290_scope2_a.zip": "711A3493279387593ECA17C3C4130154E9F860CA891B91013083E7D76E5C04A4",
    "sub_v297_scope2_a.zip": "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC",
}


def scan_zip(path: Path) -> tuple[int, list[dict]]:
    hits = []
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        for name in names:
            if name.endswith("/"):
                continue
            labels = secret_hits(name, archive.read(name))
            if labels:
                hits.append({"archive": path.name, "path": name, "hits": labels})
    return len(names), hits


def source_files() -> list[Path]:
    paths = list((ROOT / "src").rglob("*.py"))
    paths.extend((ROOT / "scripts").glob("*.py"))
    paths.extend((ROOT / "docs").glob("*.md"))
    paths.extend(
        [
            ROOT / "README.md",
            ROOT / "pyproject.toml",
            ROOT / "frontend/package.json",
            ROOT / "frontend/package-lock.json",
        ]
    )
    generated_status = (ROOT / "docs/HANOI_MANUAL_EVIDENCE_STATUS.md").resolve()
    return sorted(
        {
            path.resolve()
            for path in paths
            if path.is_file() and path.resolve() != generated_status
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "build/demo_compliance/hanoi_handover_inventory_20260828.json",
    )
    args = parser.parse_args()
    artifacts = []
    secret_findings = []
    for name, expected in LOCKED.items():
        path = ROOT / name
        exists = path.is_file()
        actual = sha256(path) if exists else None
        entries = 0
        if exists:
            entries, hits = scan_zip(path)
            secret_findings.extend(hits)
        artifacts.append(
            {
                "path": name,
                "role": (
                    "selected_public_champion"
                    if name.startswith("sub_v297")
                    else "direct_measured_rollback"
                    if name.startswith("sub_v290")
                    else "second_measured_rollback"
                    if name.startswith("sub_v276")
                    else "rejected_measured_ablation"
                    if name.startswith("sub_v265")
                    else "measured_source_clean_tie"
                    if name.startswith("sub_v269")
                    else "locked_measured_or_rollback"
                ),
                "exists": exists,
                "bytes": path.stat().st_size if exists else None,
                "sha256": actual,
                "expected_sha256": expected,
                "hash_matches": actual == expected,
                "zip_entries": entries,
            }
        )

    scanned_sources = []
    for path in source_files():
        relative = path.relative_to(ROOT).as_posix()
        payload = path.read_bytes()
        labels = secret_hits(relative, payload)
        if labels:
            secret_findings.append({"path": relative, "hits": labels})
        scanned_sources.append(
            {"path": relative, "bytes": len(payload), "sha256": sha256(path)}
        )

    report = {
        "kind": "hanoi_source_data_artifact_handover_inventory",
        "artifacts": artifacts,
        "source_inventory": {
            "files": len(scanned_sources),
            "roots": ["src/", "scripts/", "docs/", "frontend lockfiles"],
            "manifest": scanned_sources,
        },
        "data_inventory": {
            "competition_corpus": "data/financial_statements (not copied into the public evidence bundle)",
            "derived_tables": "build/tables (rebuildable local derivative, excluded from handover public bundle)",
            "submission_evidence": "candidate data/ files included only when referenced by submission.json",
            "private_evidence": "private_evidence/hanoi (ignored; never public-bundled)",
        },
        "secret_scan": {
            "patterns": "build_demo_evidence_bundle.secret_hits",
            "source_files_scanned": len(scanned_sources),
            "archives_scanned": len(artifacts),
            "finding_count": len(secret_findings),
            "findings": secret_findings,
        },
        "manual_confirmation_required": True,
    }
    report["ok"] = bool(
        all(item["exists"] and item["hash_matches"] for item in artifacts)
        and not secret_findings
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "ok": report["ok"],
                "artifacts": len(artifacts),
                "source_files_scanned": len(scanned_sources),
                "secret_findings": len(secret_findings),
                "output": str(args.out.resolve().relative_to(ROOT)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
