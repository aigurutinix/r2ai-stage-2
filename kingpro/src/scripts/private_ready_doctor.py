"""One-command fail-closed doctor for the locked V297 private submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
V297_ZIP_SHA = "90F7CC5E678EF80EB2741C5B9F162853AAF72CEA5B0ECC94BF80E2B7EAC36AAC"
V297_JSON_SHA = "E19E4748DDAFFAD28112292EAE17E9296F85BBC99265C9D52EEB27E8F8539C85"
LOCAL_UNIT_CUBE_SHA = "EC80A93F159528180402A5B91E91575B05751E59BF96DD8CBAE252FAE0D22EBC"
LEGACY_CUBE_SHA = "3B509E2E2969C4481A48CF673E95138541A4235822F28691EB7C1BD0F733076B"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def live_json(url: str) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            payload = json.load(response)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=ROOT / "build" / "private_ready_doctor.json"
    )
    parser.add_argument("--require-live", action="store_true")
    args = parser.parse_args()

    paths = {
        "freeze": ROOT / "build" / "private_final_freeze_v297_local_units.json",
        "typed": ROOT / "build" / "typed_plan_dual_execution_local_units_production.json",
        "adversarial": ROOT / "build" / "compiler_adversarial_local_units_production.json",
        "metamorphic": ROOT / "build" / "compiler_metamorphic_local_units_production.json",
        "lineage": ROOT / "build" / "demo_compliance" / "runtime_cell_lineage_coverage.json",
        "ui": ROOT / "build" / "demo_compliance" / "demo_ui_truth_private_ready.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing doctor reports: {missing}")
    freeze = load(paths["freeze"])
    typed = load(paths["typed"])
    adversarial = load(paths["adversarial"])
    metamorphic = load(paths["metamorphic"])
    lineage = load(paths["lineage"])
    ui = load(paths["ui"])

    zip_a = ROOT / "sub_v297_scope2_a.zip"
    zip_b = ROOT / "sub_v297_scope2_b.zip"
    submission = ROOT / "sub_v297_scope2" / "submission.json"
    cube = ROOT / "build" / "statement_cube.jsonl"
    rollback_cube = ROOT / "build" / "statement_cube_legacy_pre_local_units.jsonl"
    catalog = ROOT / "build" / "catalog.jsonl"
    sidecar = ROOT / "build" / "runtime_lineage" / "sub_v297_scope2_counterfactual.json"
    current = {
        "v297_zip_a": sha256(zip_a),
        "v297_zip_b": sha256(zip_b),
        "submission": sha256(submission),
        "statement_cube": sha256(cube),
        "rollback_cube": sha256(rollback_cube),
        "catalog": sha256(catalog),
        "counterfactual_sidecar": sha256(sidecar),
        "compiler": sha256(ROOT / "src" / "kingpro" / "product" / "deterministic_compiler.py"),
        "counterfactual_analyzer": sha256(ROOT / "src" / "kingpro" / "product" / "counterfactual_lineage.py"),
        "lineage_verifier": sha256(ROOT / "src" / "kingpro" / "product" / "cell_lineage.py"),
        "source_unit_parser": sha256(ROOT / "src" / "kingpro" / "financial" / "source_units.py"),
    }
    typed_summary = typed.get("summary", {})
    typed_inputs = typed.get("inputs", {})
    lineage_inputs = lineage.get("input_hashes", {})
    adversarial_inputs = adversarial.get("input_hashes", {})
    meta_inputs = metamorphic.get("input_hashes", {})
    backend = live_json("http://127.0.0.1:8080/health") if args.require_live else None
    frontend = live_json("http://127.0.0.1:3000/api/health") if args.require_live else None

    checks = {
        "v297_zip_a_locked": current["v297_zip_a"] == V297_ZIP_SHA,
        "v297_zip_b_locked": current["v297_zip_b"] == V297_ZIP_SHA,
        "v297_submission_locked": current["submission"] == V297_JSON_SHA,
        "local_unit_cube_locked": current["statement_cube"] == LOCAL_UNIT_CUBE_SHA,
        "legacy_cube_rollback_locked": current["rollback_cube"] == LEGACY_CUBE_SHA,
        "freeze_pass": freeze.get("status") == "PASS" and all(freeze.get("checks", {}).values()),
        "typed_plan_106_of_106": (
            typed_summary.get("compiler_accepted") == 106
            and typed_summary.get("verified") == 106
            and typed_summary.get("disagreements_or_errors") == 0
            and str(typed_inputs.get("statement_cube", {}).get("sha256", "")).upper()
            == current["statement_cube"]
            and str(typed_inputs.get("questions", {}).get("sha256", "")).upper()
            == current["submission"]
        ),
        "adversarial_29_of_29": (
            adversarial.get("passed") is True
            and adversarial.get("adversarial_rejected") == 29
            and adversarial_inputs.get("statement_cube_sha256") == current["statement_cube"]
            and adversarial_inputs.get("registry_sha256") == current["submission"]
            and adversarial_inputs.get("compiler_sha256") == current["compiler"]
        ),
        "metamorphic_3_of_3": (
            metamorphic.get("passed") is True
            and not metamorphic.get("disagreement_ids")
            and metamorphic.get("status_counts", {}).get("invariant") == 3
            and meta_inputs.get("statement_cube_sha256") == current["statement_cube"]
            and meta_inputs.get("registry_sha256") == current["submission"]
            and meta_inputs.get("compiler_sha256") == current["compiler"]
        ),
        "cell_lineage_1012_of_1012": (
            lineage.get("status") == "PASS"
            and lineage.get("verified_questions") == 1012
            and lineage.get("verified_cells") == 7299
            and lineage.get("coverage") == 1.0
            and not lineage.get("missing_question_ids")
            and not lineage.get("citation_unbound_cells")
            and lineage_inputs.get("submission_sha256") == current["submission"]
            and lineage_inputs.get("catalog_sha256") == current["catalog"]
            and lineage_inputs.get("counterfactual_sidecar_sha256")
            == current["counterfactual_sidecar"]
            and lineage_inputs.get("counterfactual_analyzer_sha256")
            == current["counterfactual_analyzer"]
            and lineage_inputs.get("runtime_verifier_sha256")
            == current["lineage_verifier"]
        ),
        "canonical_ui_audit_pass": ui.get("passed") is True and all(ui.get("checks", {}).values()),
        "frontend_build_exists": (ROOT / "frontend" / ".next" / "BUILD_ID").is_file(),
        "no_unsigned_v312_archive": not any(ROOT.glob("sub_v312*.zip")),
    }
    if args.require_live:
        checks.update(
            {
                "backend_live": bool(
                    backend
                    and backend.get("status") == "ok"
                    and backend.get("cell_lineage", {}).get("runtime_lineage_questions") == 1012
                    and backend.get("public_champion", {}).get("version") == "v297"
                ),
                "frontend_live": bool(frontend and frontend.get("status") == "ok"),
                "dynamic_claim_locked": bool(
                    backend and backend.get("dynamic_generation_available") is False
                ),
            }
        )
    report = {
        "schema_version": "private-ready-doctor/v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "selected_submission": {
            "version": "v297",
            "submission_id": 3747,
            "upload_file": "sub_v297_scope2_a.zip",
            "sha256": V297_ZIP_SHA,
        },
        "checks": checks,
        "current_hashes": current,
        "reports": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        "live_required": args.require_live,
        "operator_instruction": (
            "Upload/select only sub_v297_scope2_a.zip. Do not upload a handoff bundle, "
            "runtime sidecar, local-unit cube, or unsigned successor."
        ),
        "claim_limit": (
            "PASS proves local artifact integrity, reproducibility and tested runtime gates; "
            "it does not reveal or guarantee the private score."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

