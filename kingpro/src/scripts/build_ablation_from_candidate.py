"""Build a source-grounded ablation by selecting question rows from a candidate.

The candidate directory is expected to contain a superset of the baseline data
files (as produced by the compliant standard/panel builders).  This utility is
used to isolate logical repair groups for leaderboard measurement without
changing either source artifact.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def parse_ids(spec: str) -> set[int]:
    ids: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = (int(value) for value in part.split("-", 1))
            ids.update(range(start, end + 1))
        else:
            ids.add(int(part))
    return ids


def load(path: Path) -> dict[int, dict]:
    rows = json.loads((path / "submission.json").read_text(encoding="utf-8"))
    return {int(row["id"]): row for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ids", required=True, help="comma-separated IDs and inclusive ranges")
    args = parser.parse_args()

    selected = parse_ids(args.ids)
    base = load(args.base)
    candidate = load(args.candidate)
    missing = sorted(selected - set(base) | selected - set(candidate))
    if missing:
        raise SystemExit(f"missing IDs: {missing}")

    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)
    shutil.copytree(args.candidate / "data", args.out / "data")

    output = []
    audit = []
    for qid in sorted(base):
        source = candidate[qid] if qid in selected else base[qid]
        output.append(source)
        if qid in selected:
            audit.append({
                "id": qid,
                "old_answer": base[qid].get("answer"),
                "answer": candidate[qid].get("answer"),
                "answer_changed": base[qid].get("answer") != candidate[qid].get("answer"),
            })

    (args.out / "submission.json").write_text(
        json.dumps(output, ensure_ascii=False), encoding="utf-8"
    )
    (args.out / "ablation_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "output": str(args.out),
        "selected": len(selected),
        "answer_changes": sum(row["answer_changed"] for row in audit),
        "ids": sorted(selected),
    }, indent=2))


if __name__ == "__main__":
    main()
