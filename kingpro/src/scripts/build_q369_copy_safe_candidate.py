"""Build a query-only q369 candidate without chained-assignment ambiguity.

The v184 query is numerically correct but emits the only
``SettingWithCopyWarning`` in both local pandas 1.1.5 grader modes.  The BTC
public metrics differ by exactly one question between answer and execution
accuracy.  This candidate does not claim that q369 is that question; it only
removes a real cross-environment execution risk while preserving every
answer, evidence file and retrieval label.
"""

from __future__ import print_function

import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v184_mbb_credit_provision_ratio"
OUTPUT = ROOT / "sub_top123_candidate_v186_q369_copy_safe"
QUESTION_ID = 369


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

    submission = json.loads((SOURCE / "submission.json").read_text(encoding="utf-8"))
    before_rows = {int(row["id"]): digest(row) for row in submission}
    rows = {int(row["id"]): row for row in submission}
    row = rows[QUESTION_ID]

    old_2023 = "filtered_2023 = df[df['ticker'].isin(filtered['ticker']) & (df['year'] == 2023)]"
    old_2022 = "filtered_2022 = df[df['ticker'].isin(filtered['ticker']) & (df['year'] == 2022)]"
    query = row["pandas_query"]
    if query.count(old_2023) != 1 or query.count(old_2022) != 1:
        raise ValueError("q369 baseline query no longer matches the reviewed form")

    row["pandas_query"] = query.replace(old_2023, old_2023 + ".copy()").replace(
        old_2022, old_2022 + ".copy()"
    )

    changed_ids = [
        int(item["id"])
        for item in submission
        if digest(item) != before_rows[int(item["id"])]
    ]
    if changed_ids != [QUESTION_ID]:
        raise ValueError("unexpected changed rows: {!r}".format(changed_ids))

    for field in ("answer", "question", "relevant_docs", "relevant_tables", "evidence"):
        source_row = next(
            item
            for item in json.loads(
                (SOURCE / "submission.json").read_text(encoding="utf-8")
            )
            if int(item["id"]) == QUESTION_ID
        )
        if row[field] != source_row[field]:
            raise ValueError("q369 {} changed unexpectedly".format(field))

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
                "answer_preserved": row["answer"],
                "purpose": "remove the only pandas SettingWithCopyWarning",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
