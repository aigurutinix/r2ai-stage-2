"""Which questions are still worth another generation attempt, and why.

Three patches took EXECUTION from 0.3281 to 0.3636 by handing questions from the
5.9%/7.2%/13.3% branches to the 14B. That pool is nearly empty: 51 questions
remain on a sub-15% branch, and every one of them already has a 14B row that
failed to produce a usable program. Attempting them again is the only way to
reach them.

Two disjoint sets are worth the retry, for different reasons:

  * **still provably wrong** — a rate unit carrying a đồng figure, an impossible
    count, or a placeholder. These score zero for certain, so any working
    replacement is a free roll.
  * **still on a sub-15% branch** — the 14B beats those branches by the two
    measurements we now have (~20% and ~25%), so a working program is positive
    expected value even though it is not free.

The retry changes the two things that plausibly caused the failure: a wider table
budget, and more attempts at the program itself.

Usage:  PYTHONPATH=src python scripts/_retry_targets.py [submission.zip]
"""

from __future__ import annotations

import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from patch_submission import wrong_for_sure  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402

SUB = sys.argv[1] if len(sys.argv) > 1 else "sub15.zip"
CACHES = {
    "locate_model": "located.jsonl",
    "locate_embed": "embed_located.jsonl",
    "plan": "planned.jsonl",
    "llm_8b": "gen_helpers.jsonl",
    "llm_14b": "gen14b_v2.jsonl",
}
WEAK = {"locate_model", "locate_embed", "plan", "placeholder"}


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
    parsed = {
        q.id: q
        for q in parse_all(
            ROOT / "data/questions/questions.jsonl", ROOT / "data/code_stock.csv")
    }
    with zipfile.ZipFile(ROOT / "submissions" / SUB) as z:
        records = {r["id"]: r for r in json.loads(z.read("submission.json"))}
    caches = {name: load(file) for name, file in CACHES.items()}

    reasons: Counter[str] = Counter()
    targets: set[int] = set()
    for qid, record in records.items():
        code = (record.get("pandas_query") or "").strip()
        label = "placeholder" if code in ("", "result = 0.0") else "lexical"
        if label != "placeholder":
            for name, cache in caches.items():
                if cache.get(qid) == code:
                    label = name
                    break

        why = wrong_for_sure(parsed[qid], record)
        if why is not None:
            reasons[f"provably_wrong:{why}"] += 1
            targets.add(qid)
        elif label in WEAK:
            reasons[f"weak_branch:{label}"] += 1
            targets.add(qid)

    print(f"{SUB}: {len(records)} questions")
    for key, count in reasons.most_common():
        print(f"  {key:34s} {count:4d}")
    print(f"\n  retry targets: {len(targets)}")

    out = ROOT / "artifacts" / "_retry_ids.json"
    out.write_text(json.dumps(sorted(targets)), encoding="utf-8")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
