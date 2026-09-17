"""Build five source-reviewed absolute-difference repairs on hardened v211.

The five prompts ask the magnitude of a generic ``chênh lệch ... so với``.
Internal same-template precedents q781 and q799 return a positive magnitude
even when the first named operand is smaller than the second.  Explicitly
directional questions (growth, change, subtraction and ``hiệu giữa``) are not
modified.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v211_global_result_float_hardened"
OUTPUT = ROOT / "sub_top123_candidate_v212_difference_magnitude_batch5"

PATCHES = {
    418: {
        "old_answer": -4.39,
        "new_answer": 4.39,
        "old_code": "result = high['roa_pct'] - low['roa_pct']",
        "new_code": "result = abs(high['roa_pct'] - low['roa_pct'])",
        "reason": "absolute ROA percentage-point difference",
    },
    742: {
        "old_answer": -4691149.0,
        "new_answer": 4691149.0,
        "old_code": "result = round(v0 - v1, 2)",
        "new_code": "result = round(abs(v0 - v1), 2)",
        "reason": "absolute MSB/VCB provision difference, VND million",
    },
    758: {
        "old_answer": -608.58,
        "new_answer": 608.58,
        "old_code": "result = round((v0 - v1 * 1e6) / 1e9, 2)",
        "new_code": "result = round(abs(v0 - v1 * 1e6) / 1e9, 2)",
        "reason": "absolute KBC/VIC related-loan difference, VND billion",
    },
    790: {
        "old_answer": -11.46,
        "new_answer": 11.46,
        "old_code": "result = round((v0 - v1) / 100, 2)",
        "new_code": "result = round(abs(v0 - v1) / 100, 2)",
        "reason": "absolute VIB/BID manufacturing-loan share difference, pp",
    },
    793: {
        "old_answer": -536.4,
        "new_answer": 536.4,
        "old_code": "result = round((v0 - v1) / 1e6, 2)",
        "new_code": "result = round(abs(v0 - v1) / 1e6, 2)",
        "reason": "absolute GEE/GEX outstanding-share difference, million shares",
    },
}


def _digest(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    source_rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    source_by_id = {int(row["id"]): row for row in source_rows}
    by_id = {int(row["id"]): row for row in rows}

    for qid, patch in PATCHES.items():
        row = by_id[qid]
        if float(row["answer"]) != patch["old_answer"]:
            raise AssertionError(f"q{qid} unexpected old answer: {row['answer']}")
        code = str(row["pandas_query"])
        if code.count(patch["old_code"]) != 1:
            raise AssertionError(f"q{qid} old code not unique")
        row["answer"] = patch["new_answer"]
        row["pandas_query"] = code.replace(patch["old_code"], patch["new_code"])

    changed = [int(row["id"]) for row in rows if row != source_by_id[int(row["id"])]]
    expected = sorted(PATCHES)
    if changed != expected:
        raise AssertionError(f"unexpected changed IDs: {changed}")
    for qid in expected:
        fields = {
            key
            for key in by_id[qid]
            if by_id[qid].get(key) != source_by_id[qid].get(key)
        }
        if fields != {"answer", "pandas_query"}:
            raise AssertionError(f"q{qid} unexpected changed fields: {fields}")

    shutil.copytree(SOURCE, OUTPUT)
    _write_json(OUTPUT / "submission.json", rows)

    audits = json.loads((OUTPUT / "source_audit.json").read_text(encoding="utf-8"))
    audit_by_id = {int(item["id"]): item for item in audits}
    for qid in (742, 758, 790, 793):
        item = audit_by_id[qid]
        if float(item["answer"]) != PATCHES[qid]["old_answer"]:
            raise AssertionError(f"q{qid} unexpected source-audit answer")
        item["answer"] = PATCHES[qid]["new_answer"]
        item["note"] = PATCHES[qid]["reason"]
    _write_json(OUTPUT / "source_audit.json", audits)

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids_relative_to_source": expected,
        "changed_fields": ["answer", "pandas_query"],
        "repairs": {str(qid): PATCHES[qid] for qid in expected},
        "internal_precedents": {
            "q781": "PC1 < GEX, but the same prompt family uses abs and returns +1251.54",
            "q799": "HBC < SAM, but the same prompt family uses abs and returns +118.85",
        },
        "unchanged_directional_policy": (
            "Growth, change, explicit subtraction, 'hiệu giữa', and signed accounting "
            "ratios retain their original sign."
        ),
        "source_submission_sha256": _digest(source_rows),
        "candidate_submission_sha256": _digest(rows),
        "answers_changed": len(expected),
        "evidence_changed": 0,
        "tables_changed": 0,
        "runtime_hardening_inherited": "result = float(result) on all 1012 programs",
        "claim_limit": "Public effect unknown until submitted.",
    }
    _write_json(OUTPUT / "v212_difference_magnitude_batch5_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
