r"""Build a warning-clean successor without changing any calculation.

v186 removes the sole Pandas ``SettingWithCopyWarning``.  When all Python
warnings are promoted to errors, q170 and q222 still fail because their regex
patterns are ordinary string literals containing invalid ``\(``/``\.``
escapes.  Prefixing those two literals with ``r`` preserves the regex received
by Pandas while making the source portable across stricter Python parsers.
"""

from __future__ import print_function

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v186_q369_copy_safe"
OUTPUT = ROOT / "sub_top123_candidate_v187_warning_clean"
QUESTION_IDS = (170, 222)


def digest(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(SOURCE)
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)

    submission_path = SOURCE / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    baseline = json.loads(submission_path.read_text(encoding="utf-8"))
    before = {int(row["id"]): digest(row) for row in submission}
    rows = {int(row["id"]): row for row in submission}

    for question_id in QUESTION_IDS:
        query = rows[question_id]["pandas_query"]
        marker = ".str.contains('"
        if query.count(marker) != 1:
            raise ValueError(
                "q{} expected exactly one ordinary regex literal".format(question_id)
            )
        rows[question_id]["pandas_query"] = query.replace(
            marker, ".str.contains(r'", 1
        )

    changed_ids = [
        int(row["id"])
        for row in submission
        if digest(row) != before[int(row["id"])]
    ]
    if changed_ids != list(QUESTION_IDS):
        raise ValueError("unexpected changed rows: {!r}".format(changed_ids))

    baseline_rows = {int(row["id"]): row for row in baseline}
    for question_id in QUESTION_IDS:
        for field in (
            "answer",
            "question",
            "relevant_docs",
            "relevant_tables",
            "evidence",
        ):
            if rows[question_id][field] != baseline_rows[question_id][field]:
                raise ValueError(
                    "q{} {} changed unexpectedly".format(question_id, field)
                )

    shutil.copytree(str(SOURCE), str(OUTPUT))
    (OUTPUT / "submission.json").write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "source": SOURCE.name,
                "output": OUTPUT.name,
                "changed_ids": changed_ids,
                "changed_fields": ["pandas_query"],
                "purpose": "pass strict warnings-as-errors in both grader modes",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
