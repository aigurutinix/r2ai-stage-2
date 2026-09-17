"""A gold set of multi-step questions, in the shape `eval_gold.py` reads.

The automated A/B scores `easy_416`, which is entirely single-cell. That answers
"did the adapter damage the easy tier" and says nothing about the multi-step
shapes it was trained for — and those are the majority of the exam. This turns
the verified `shapes_v2` records into a scorable set for exactly that.

Two properties make it usable as a held-out measure rather than a memory test:

* every record was executed against its own tables and reproduced its own answer
  before `gen_shapes.py` wrote it, so a wrong label cannot be scored as right;
* `shapes_v2` was generated on a different seed and behind two gates the training
  pool never had, and any question whose text appears in the training set is
  dropped here anyway.

Usage:
  PYTHONPATH=src python scripts/build_multistep_gold.py \
      --records artifacts/shapes_v2.jsonl --train artifacts/sft_mixed.jsonl \
      --out artifacts/gold_multistep.jsonl --per-shape 40
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalise(text: str) -> str:
    return " ".join(str(text).split()).lower()


def training_questions(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        messages = row.get("messages")
        if messages and len(messages) > 1:
            user = messages[1].get("content", "")
            start = user.find("<question")
            end = user.find("</question>")
            if start != -1 and end != -1:
                seen.add(normalise(user[user.find(">", start) + 1:end]))
            continue
        if row.get("question"):
            seen.add(normalise(row["question"]))
    return seen


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="artifacts/shapes_v2.jsonl")
    parser.add_argument("--train", default="artifacts/sft_mixed.jsonl")
    parser.add_argument("--out", default="artifacts/gold_multistep.jsonl")
    parser.add_argument("--per-shape", type=int, default=40)
    parser.add_argument("--min-tables", type=int, default=2,
                        help="drop single-table records; the easy set covers those")
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()

    trained = training_questions(ROOT / args.train)
    print(f"{len(trained)} training questions to exclude")

    by_shape: dict[str, list[dict]] = collections.defaultdict(list)
    dropped_overlap = dropped_small = 0
    for line in (ROOT / args.records).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if len(row.get("relevant_tables") or []) < args.min_tables:
            dropped_small += 1
            continue
        if normalise(row["question"]) in trained:
            dropped_overlap += 1
            continue
        by_shape[row.get("shape", "?")].append(row)

    rng = random.Random(args.seed)
    out: list[dict] = []
    for shape in sorted(by_shape):
        rows = by_shape[shape]
        rng.shuffle(rows)
        for row in rows[: args.per_shape]:
            out.append({
                "id": len(out) + 1,
                "question": row["question"],
                "answer": row["answer"],
                "relevant_tables": row["relevant_tables"],
                "relevant_docs": sorted({r.split("|")[0] for r in row["relevant_tables"]}),
                "pandas_query": row["pandas_query"],
                "csv_path": "",
                "difficulty": row.get("shape", "?"),
            })

    path = ROOT / args.out
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in out),
                    encoding="utf-8")
    print(f"dropped {dropped_small} single-table, {dropped_overlap} seen in training")
    print(f"wrote {len(out)} records -> {path}")
    spread = collections.Counter(r["difficulty"] for r in out)
    for name, count in spread.most_common():
        print(f"  {count:4d}  {name}")
    tables = collections.Counter(len(r["relevant_tables"]) for r in out)
    print("  tables per question: " + ", ".join(
        f"{k}:{tables[k]}" for k in sorted(tables)))


if __name__ == "__main__":
    main()
