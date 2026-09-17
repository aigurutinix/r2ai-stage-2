"""Label every shipped program by the cache it came from, exactly.

Guessing the branch from a program's shape failed — 501 of the 1,012 programs
come from one generator shared by the 42.8% label matcher and the 5.9%
best-effort fallback. Replaying `Corroborator.choose` worked but needed both of
`run_submit`'s gates reconstructed by hand, and the first attempt over-selected
by a factor of four.

There is an exact test available for three of the branches. `located.jsonl`,
`planned.jsonl` and the generation cache each store the *program text* they
handed to the cascade, so a shipped `pandas_query` that is byte-identical to one
of them came from that branch and no other. That labels the model-backed
branches with no inference at all, and what is left over is the lexical side.

Measured accuracies, for deciding what is worth displacing with the 14B (which
the last two submissions put at ~15% on the hardest pool and ~20% on the
displaced branches):

    label matcher  42.8%   llm (8B)  27%   locate  13.3%   plan  7.2%   fallback  5.9%

Usage:  PYTHONPATH=src python scripts/_label_branches.py [submission.zip]
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUB = sys.argv[1] if len(sys.argv) > 1 else "weak14b.zip"

CACHES = {
    "locate_model": "located.jsonl",
    "locate_embed": "embed_located.jsonl",
    "plan": "planned.jsonl",
    "llm_8b": "gen_helpers.jsonl",
    "llm_14b": "gen14b_v2.jsonl",
}


def load(name: str) -> dict[int, str]:
    path = ROOT / "artifacts" / name
    if not path.exists():
        return {}
    out: dict[int, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("code"):
                out[row["id"]] = row["code"].strip()
    return out


def main() -> None:
    with zipfile.ZipFile(ROOT / "submissions" / SUB) as z:
        records = json.loads(z.read("submission.json"))
    caches = {name: load(file) for name, file in CACHES.items()}

    labels: Counter[str] = Counter()
    ids: dict[str, list[int]] = {}
    for record in records:
        code = (record.get("pandas_query") or "").strip()
        if code in ("", "result = 0.0"):
            label = "placeholder"
        else:
            label = "lexical"
            for name, cache in caches.items():
                if cache.get(record["id"]) == code:
                    label = name
                    break
        labels[label] += 1
        ids.setdefault(label, []).append(record["id"])

    print(f"{SUB}: {len(records)} programs\n")
    accuracy = {
        "lexical": "42.8% matcher + 5.9% fallback, mixed",
        "llm_8b": "27%", "llm_14b": "~15-20% (measured 10/08)",
        "locate_model": "13.3%", "locate_embed": "13.3%",
        "plan": "7.2%", "placeholder": "0%",
    }
    for label, count in labels.most_common():
        print(f"  {count:5d}  {label:14s}  {accuracy.get(label, '')}")

    # What is still served by a branch the 14B beats?
    beatable = [
        qid for label in ("locate_model", "locate_embed", "plan", "placeholder")
        for qid in ids.get(label, [])
    ]
    fresh = load("gen14b_v2.jsonl")
    have = [qid for qid in beatable if qid in fresh]
    print(f"\n  still served by a branch under 15%: {len(beatable)}")
    print(f"  of those, a 14B program exists      : {len(have)}")
    out = ROOT / "artifacts" / "_beatable_ids.json"
    out.write_text(json.dumps(sorted(have)), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
