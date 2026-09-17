"""Merge the two-stage reader cache into the shipped 14B cache.

The two caches answer the same questions two different ways and agree on the value
only 26.9% of the time, so how they are merged decides the size of the diff — and
the diff-size law from the leaderboard says a targeted change wins where a wide
rebuild loses 4/4. Three modes, from narrowest to widest:

  fill      only ids the 14B cache cannot run at all (47 questions)
  confident fill, plus override where the note pinned a table whose heading really
            matches the question (spec score >= --min-score)
  replace   the two-stage cache alone, which is the head-on test of "does the
            reader beat the label matcher"

`fill` and `confident` keep the 14B record's shape verbatim, so `run_submit`
cannot tell the merged file from the original.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rows[record["id"]] = record
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="artifacts/gen14b_merged.jsonl")
    parser.add_argument("--reader", default="artifacts/two_stage.jsonl")
    parser.add_argument("--spec", default="artifacts/spec.jsonl")
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", choices=("fill", "confident", "replace"), default="fill")
    parser.add_argument("--min-score", type=float, default=0.7)
    args = parser.parse_args()

    base = load(Path(args.base))
    reader = load(Path(args.reader))
    scores = {}
    spec_path = Path(args.spec)
    if spec_path.exists():
        scores = {i: r.get("score", 0.0) for i, r in load(spec_path).items()}

    if args.mode == "replace":
        merged = dict(reader)
        added = len(reader)
        overridden = 0
    else:
        merged = dict(base)
        runnable = {i: r for i, r in reader.items() if r.get("ok")}
        added = 0
        overridden = 0
        for qid, record in runnable.items():
            if qid not in base or not base[qid].get("ok"):
                merged[qid] = record
                added += 1
            elif args.mode == "confident" and scores.get(qid, 0.0) >= args.min_score:
                merged[qid] = record
                overridden += 1

    out = Path(args.out)
    with out.open("w", encoding="utf-8") as handle:
        for qid in sorted(merged):
            handle.write(json.dumps(merged[qid], ensure_ascii=False) + "\n")
    print(
        f"{args.mode}: {len(merged)} ban ghi -> {out}  "
        f"(them {added}, ghi de {overridden})"
    )


if __name__ == "__main__":
    main()
