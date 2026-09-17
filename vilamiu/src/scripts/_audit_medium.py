"""Is the derived (medium) pool sound enough to train on at all?

Three defects are visible by eye in the first three records, and each would be
taught verbatim by an SFT run:

1. **Arity.** Record 3 sums *four* cells (two per frame), not two. A locator that
   assumes one cell per frame cannot express it, and forcing it to would attach
   the wrong cell reference to the right answer.
2. **Coverage.** Record 2 asks about ASM, NKG *and* VPH but cites two tables and
   sums two figures — its answer is that of record 1, to the digit. The question
   and the answer disagree, so the pair teaches arithmetic that is wrong for the
   question asked.
3. **Duplication.** Two records already share an answer. Near-duplicate prompts
   with one correct and one incorrect target are worse than either alone.

This counts all three so the decision to fix, filter, or drop rests on numbers.
Tickers come from the question's `(XXX)` parentheses and from the
`relevant_tables` prefixes; the roster is not needed to compare the two sets.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TICKER_IN_QUESTION = re.compile(r"\(([A-Z]{3})\)")
READ_CALL = re.compile(r"\.loc\[|\.iloc\[|\.at\[|\.values\[")


def load(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line.replace(": NaN", ": null")))
    return rows


def table_tickers(record: dict) -> set[str]:
    out = set()
    for ref in record.get("relevant_tables", []):
        head = str(ref).split("_", 1)[0]
        if re.fullmatch(r"[A-Z]{3}", head):
            out.add(head)
    return out


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = load(ROOT / "artifacts" / "sft_records.jsonl")
    medium = [r for r in rows if r.get("difficulty") == "medium"]
    print(f"{len(medium)} medium records\n")

    # 1. arity: how many cell reads does the gold program make?
    arity: collections.Counter[int] = collections.Counter()
    for record in medium:
        arity[len(READ_CALL.findall(str(record.get("pandas_query", ""))))] += 1
    print("cell reads per program:")
    for n, count in sorted(arity.items()):
        print(f"  {n:>2} reads  {count:>4}")

    # 2. coverage: does the question name a company the tables cannot answer for?
    uncovered = []
    for record in medium:
        asked = set(TICKER_IN_QUESTION.findall(record["question"]))
        have = table_tickers(record)
        if asked - have:
            uncovered.append((record, sorted(asked - have)))
    print(f"\nquestion names a ticker with no cited table: "
          f"{len(uncovered)}/{len(medium)} = {len(uncovered) / len(medium):.1%}")
    for record, missing in uncovered[:5]:
        print(f"  missing {missing}: {record['question'][:100]}")

    # 3. duplication: same answer, different question
    by_answer: dict[str, list[str]] = collections.defaultdict(list)
    for record in medium:
        by_answer[repr(record.get("answer"))].append(record["question"])
    shared = {a: qs for a, qs in by_answer.items() if len(qs) > 1}
    print(f"\nanswers shared by more than one question: {len(shared)} "
          f"covering {sum(len(q) for q in shared.values())} records")
    for answer, questions in list(shared.items())[:3]:
        print(f"  {answer} <- {len(questions)} questions")
        for q in questions[:3]:
            print(f"      {q[:95]}")

    # A record is only trainable if every company asked about is citable AND the
    # program's arity is something the target format can express.
    clean = [
        r for r in medium
        if not (set(TICKER_IN_QUESTION.findall(r["question"])) - table_tickers(r))
    ]
    print(f"\nmedium records passing the coverage check: {len(clean)}/{len(medium)}")
    print("  of those, cell reads:",
          dict(sorted(collections.Counter(
              len(READ_CALL.findall(str(r.get("pandas_query", ""))))
              for r in clean).items())))


if __name__ == "__main__":
    main()
