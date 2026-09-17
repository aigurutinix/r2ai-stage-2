"""Build a q501-only runtime-shape ablation from v210.

The public scorer reports one ``AttributeError`` among an undisclosed subset of
506 questions.  q501 was initially suspected because it is an integer-shaped
year result, but the 506 gold IDs are not the first 506 predictions and later
history analysis did not identify the failing ID.  Keep this builder only as a
clearly labelled ablation; it is not a source-proven repair.
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
OUTPUT = ROOT / "sub_top123_candidate_v211_q501_execution_hardened"
QUESTION_ID = 501
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
    by_id = {int(row["id"]): row for row in rows}
    row = by_id[QUESTION_ID]

    if float(row["answer"]) != 2023.0:
        raise AssertionError("unexpected q501 answer")
    if MARKER in row["pandas_query"]:
        raise AssertionError("q501 is already hardened")
    row["pandas_query"] = row["pandas_query"].rstrip() + MARKER

    changed_ids = [
        int(item["id"]) for item in rows if digest(item) != before[int(item["id"])]
    ]
    if changed_ids != [QUESTION_ID]:
        raise AssertionError(f"unexpected changed IDs: {changed_ids}")

    source = {int(item["id"]): item for item in source_rows}[QUESTION_ID]
    changed_fields = {
        key for key in row if row.get(key) != source.get(key)
    }
    if changed_fields != {"pandas_query"}:
        raise AssertionError(f"unexpected q501 fields: {changed_fields}")

    shutil.copytree(SOURCE, OUTPUT)
    write_json(OUTPUT / "submission.json", rows)
    write_json(
        OUTPUT / "v211_q501_execution_hardening_audit.json",
        {
            "candidate": OUTPUT.name,
            "source_candidate": SOURCE.name,
            "changed_question_ids_relative_to_source": changed_ids,
            "changed_fields": ["pandas_query"],
            "answer_before": source["answer"],
            "answer_after": row["answer"],
            "claim_limit": (
                "Ablation only. Hidden gold IDs and history analysis invalidate "
                "the earlier first-506 attribution; the failing ID is undisclosed."
            ),
        },
    )
    print(
        json.dumps(
            {
                "output": OUTPUT.name,
                "changed_ids": changed_ids,
                "answer_unchanged": row["answer"],
                "hardening": "result = float(result)",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
