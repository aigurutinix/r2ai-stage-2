"""Build a deterministic, payload-free V297 private/final handoff bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.submission.archive import write_deterministic  # noqa: E402

FILES = (
    ("docs/PRIVATE_TEST_FINAL_CHECKLIST_V297.md", "PRIVATE_TEST_FINAL_CHECKLIST_V297.md"),
    ("docs/V297_SUBMISSION_HANDOFF.md", "V297_SUBMISSION_HANDOFF.md"),
    ("docs/PRIVATE_FINAL_SELECTION_V297.md", "PRIVATE_FINAL_SELECTION_V297.md"),
    ("build/private_final_freeze_v297_local_units.json", "reports/private_final_freeze_v297.json"),
    ("build/private_ready_doctor.json", "reports/private_ready_doctor.json"),
    ("build/private_final_leaderboard_check_v297.json", "reports/leaderboard_selected_v297.json"),
    ("build/v299_v297_public_dominance.json", "reports/v297_public_dominance.json"),
    ("build/release_gate/sub_v297_scope2/release.json", "reports/v297_release_gate.json"),
    ("build/typed_plan_dual_execution_local_units_production.json", "reports/typed_plan_dual_execution.json"),
    ("build/compiler_adversarial_local_units_production.json", "reports/compiler_adversarial.json"),
    ("build/compiler_metamorphic_local_units_production.json", "reports/compiler_metamorphic.json"),
    ("build/demo_compliance/runtime_cell_lineage_coverage.json", "reports/runtime_cell_lineage.json"),
    ("build/demo_compliance/demo_ui_truth_private_ready.json", "reports/demo_ui_truth.json"),
    ("build/local_unit_cube_candidate_audit.json", "reports/local_unit_cube_candidate.json"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = args.out.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "kind": "v297_private_final_handoff",
        "selected": {
            "version": "v297",
            "submission_id": 3747,
            "competition_file": "sub_v297_scope2_a.zip",
            "sha256": "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC",
            "payload_included": False,
        },
        "operator_warning": "This handoff ZIP is documentation only. Upload sub_v297_scope2_a.zip to Codabench, never this bundle.",
        "files": [],
    }
    resolved = []
    for source_name, arcname in FILES:
        source = ROOT / source_name
        if not source.is_file():
            raise FileNotFoundError(source)
        resolved.append((source, arcname))
        manifest["files"].append({"path": arcname, "sha256": sha256(source), "bytes": source.stat().st_size})
    manifest_path = output.with_suffix(".manifest.tmp.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            write_deterministic(archive, manifest_path, "manifest.json")
            for source, arcname in resolved:
                write_deterministic(archive, source, arcname)
        with zipfile.ZipFile(output) as archive:
            bad = archive.testzip()
            if bad is not None or len(archive.namelist()) != len(resolved) + 1:
                raise RuntimeError(f"handoff archive verification failed: {bad}")
    finally:
        manifest_path.unlink(missing_ok=True)
    print(json.dumps({"archive": str(output), "sha256": sha256(output), "entries": len(resolved) + 1, "bytes": output.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
