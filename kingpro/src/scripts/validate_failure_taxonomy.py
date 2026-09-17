"""Validate the persistent Financial-QA failure/fix registry.

This does not judge answer correctness.  It ensures that the project's memory
stays machine-readable, references real audit scripts, and only names question
IDs that exist in the current 1,012-row rollback artifact.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "knowledge" / "financial_qa_failure_taxonomy.json"
SUBMISSION = ROOT / "sub_top123_candidate_v195_aggregate_contributor_order" / "submission.json"


def main() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    rows = json.loads(SUBMISSION.read_text(encoding="utf-8"))
    valid_ids = {int(row["id"]) for row in rows}
    problems: list[str] = []
    seen_modes: set[str] = set()

    for mode in registry.get("failure_modes", []):
        mode_id = str(mode.get("id", "")).strip()
        if not mode_id:
            problems.append("failure mode without id")
            continue
        if mode_id in seen_modes:
            problems.append(f"duplicate failure mode: {mode_id}")
        seen_modes.add(mode_id)
        for field in ("signals", "safe_fix", "audits", "known_cases"):
            if field not in mode:
                problems.append(f"{mode_id}: missing {field}")
        for script_name in mode.get("audits", []):
            if not (ROOT / "scripts" / script_name).is_file():
                problems.append(f"{mode_id}: missing audit script {script_name}")
        for qid in mode.get("known_cases", []):
            if int(qid) not in valid_ids:
                problems.append(f"{mode_id}: unknown question id {qid}")

    urls = [str(item.get("url", "")) for item in registry.get("external_references", [])]
    if len(urls) != len(set(urls)):
        problems.append("duplicate external reference URL")

    result = {
        "registry": str(REGISTRY),
        "failure_modes": len(seen_modes),
        "known_case_links": sum(
            len(mode.get("known_cases", [])) for mode in registry.get("failure_modes", [])
        ),
        "external_references": len(urls),
        "problems": problems,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
