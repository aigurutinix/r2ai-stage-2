"""Emit an answer cache from the consensus of the channels, not from branch priority.

The calibration is measured: a value that N independent channels land on agrees with
the leaderboard-selected artifact 100% of the time at N=8, 72% at N=5, and 23% at
N=2. The label match score, which every branch here ranks by, has no separation at
all. So the selection rule is cluster size, and the cache written here carries, for
each question, the value the largest cluster agreed on.

A value alone is not shippable — the scorer re-runs `pandas_query`. So the record
comes from a channel INSIDE the winning cluster that has a program: the value is
chosen by consensus, the program that produces it by whichever member can express
it. Clusters made only of the cross-year reads are dropped, because that channel
compares two reports and has no single program.

Which member to take when several qualify is decided empirically rather than by
taste: channels are ranked by how often they appear in large clusters, and the best
such channel in the winning cluster supplies the record.

Usage:
  PYTHONPATH=src python scripts/build_consensus_cache.py --min-size 3 \
      --out artifacts/cons3.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402

CACHES = (
    ("lookup@8", "artifacts/phr_8.jsonl"),
    ("lookup@30", "artifacts/phr_30.jsonl"),
    ("plan", "artifacts/planned_v3.jsonl"),
    ("locate", "artifacts/located.jsonl"),
    ("embed", "artifacts/embed_located.jsonl"),
    ("llm14b", "artifacts/gen14b_merged.jsonl"),
    ("read_note", "artifacts/two_stage.jsonl"),
    ("read_rank1", "artifacts/rank1.jsonl"),
    ("ratio_pair", "artifacts/spec_ratio_fake.jsonl"),
)


def close(a, b) -> bool:
    if a is None or b is None:
        return False
    if abs(a - b) <= 0.01:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= 1e-4


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-size", type=int, default=3)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    questions = {q.id: q for q in parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv")}

    records: dict[str, dict[int, dict]] = {}
    for name, path in CACHES:
        file = ROOT / path
        if not file.exists():
            continue
        rows = {}
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("ok", True) and record.get("value") is not None \
                    and record.get("code"):
                rows[record["id"]] = record
        records[name] = rows

    readings: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for name, rows in records.items():
        for qid, record in rows.items():
            readings[qid].append((name, float(record["value"])))

    cross = ROOT / "artifacts" / "cross_year.jsonl"
    if cross.exists():
        for line in cross.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            question = questions.get(row["id"])
            if question is None:
                continue
            scale = question.unit_scale or 1.0
            readings[row["id"]].append(("year_Y", row["current"]["value"] / scale))
            readings[row["id"]].append(("year_Y1", row["comparative"]["value"] / scale))

    def clusters(values):
        groups = []
        for name, value in values:
            for index, (centre, members) in enumerate(groups):
                if close(centre, value):
                    groups[index] = (centre, members + [name])
                    break
            else:
                groups.append((value, [name]))
        groups.sort(key=lambda item: (-len(item[1]), item[0]))
        return groups

    # Rank the channels by how often they sit in a large cluster. A channel that
    # keeps company with agreement is the one to take the program from.
    weight: Counter[str] = Counter()
    for qid, values in readings.items():
        groups = clusters(values)
        if len(groups[0][1]) >= 4:
            for name in groups[0][1]:
                weight[name] += 1
    order = [name for name, _ in weight.most_common()]
    print("kenh xep theo so lan nam trong nhom dong thuan lon:")
    for name in order:
        print(f"  {name:12s} {weight[name]}")

    out, sizes = [], Counter()
    no_program = 0
    for qid in sorted(readings):
        groups = clusters(readings[qid])
        value, members = groups[0]
        if len(members) < args.min_size:
            continue
        pick = None
        for name in order + [m for m in members if m not in order]:
            if name in members and name in records and qid in records[name]:
                pick = records[name][qid]
                break
        if pick is None:
            no_program += 1
            continue
        sizes[len(members)] += 1
        out.append({**pick, "value": pick["value"], "cluster": len(members),
                    "members": members})

    (ROOT / args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
        encoding="utf-8")
    print(f"\nnguong >={args.min_size}: {len(out)} cau, "
          f"{no_program} cau nhom chi co kenh khong co chuong trinh")
    print("  phan bo do lon nhom: " +
          ", ".join(f"{s}:{n}" for s, n in sorted(sizes.items(), reverse=True)))
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
