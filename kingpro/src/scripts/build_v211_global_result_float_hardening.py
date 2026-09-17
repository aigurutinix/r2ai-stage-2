"""Build a value-preserving scalar-shape hardening ablation from v210.

Every non-empty program already produces a numeric result matching its stored
answer in local string and typed-data replays.  Append a final ``float`` cast so
the scorer receives one stable Python scalar type regardless of whether the
program naturally returns ``int``, ``numpy`` scalar, or Python ``float``.

This does not claim to fix the undisclosed AttributeError.  It is a reversible
runtime ablation that changes no answer, evidence, document, or table choice.
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
SOURCE = ROOT / "sub_top123_candidate_v210_semantic_scope_batch2"
OUTPUT = ROOT / "sub_top123_candidate_v211_global_result_float_hardened"
MARKER = "\nresult = float(result)"


def digest(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")

    source_rows = json.loads(
        (SOURCE / "submission.json").read_text(encoding="utf-8")
    )
    rows = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    before = {int(row["id"]): digest(row) for row in rows}
    hardened_ids: list[int] = []

    for row in rows:
        code = str(row.get("pandas_query") or "").rstrip()
        if not code:
            continue
        if code.endswith(MARKER.strip()):
            continue
        row["pandas_query"] = code + MARKER
        hardened_ids.append(int(row["id"]))

    changed_ids = [
        int(row["id"]) for row in rows if digest(row) != before[int(row["id"])]
    ]
    if changed_ids != hardened_ids:
        raise AssertionError("hardening/change-set mismatch")
    if not changed_ids:
        raise AssertionError("no programs hardened")

    for old, new in zip(source_rows, rows):
        changed_fields = {key for key in new if new.get(key) != old.get(key)}
        if int(new["id"]) in changed_ids:
            if changed_fields != {"pandas_query"}:
                raise AssertionError(
                    f"q{new['id']} changed unexpected fields: {changed_fields}"
                )
        elif changed_fields:
            raise AssertionError(f"q{new['id']} unexpectedly changed")

    shutil.copytree(SOURCE, OUTPUT)
    write_json(OUTPUT / "submission.json", rows)
    write_json(
        OUTPUT / "v211_global_result_float_hardening_audit.json",
        {
            "candidate": OUTPUT.name,
            "source_candidate": SOURCE.name,
            "programs_hardened": len(changed_ids),
            "changed_question_ids_relative_to_source": changed_ids,
            "changed_fields": ["pandas_query"],
            "answers_changed": 0,
            "evidence_changed": 0,
            "tables_changed": 0,
            "hardening": "result = float(result)",
            "claim_limit": (
                "Value-preserving runtime ablation. The scorer does not disclose "
                "the current AttributeError question ID."
            ),
        },
    )
    print(
        json.dumps(
            {
                "output": OUTPUT.name,
                "programs_hardened": len(changed_ids),
                "answers_changed": 0,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
