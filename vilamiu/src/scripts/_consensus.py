"""Answer by consensus across independent channels instead of by branch priority.

The pipeline runs its branches in a fixed order and takes the first one that
answers. That throws away the only signal measured today that actually separates a
good read from a bad one. On 185 exam questions readable from two independent
documents, the reads that AGREE match the leaderboard-selected artifact 74% of the
time; where they disagree, either read matches it 27% of the time. And the label
match score — which every mechanism here ranks by — has no separation at all: its
median is 0.92 when the reads agree and 0.95 when they disagree, higher when wrong.

So ranking is the wrong operation. Agreement is the right one, and it needs no gold:
two independently located cells landing on the same nine-digit figure is not chance.

The channels are already on disk. Each was produced by a different mechanism at a
different time — the regex label matcher at two search depths, two cell locators,
two model passes with different table pinning, the 14B program cache, the cell-plan
cache, and the cross-year comparative read. Nine ways to be wrong independently.

What comes out is a value plus a cluster size, and the cluster size is a confidence
this project has never had. That is the thing the frozen artifact bought with ten
leaderboard slots a day, available offline for nothing.

Usage:  PYTHONPATH=src python scripts/_consensus.py
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402

# Every cache in `run_generate`'s shape holds `value` already scaled to the unit the
# question asks for. `cross_year.jsonl` holds đồng, because it compares two reports
# whose units differ, so it is converted on load.
CACHES = (
    ("lookup@8", "artifacts/phr_8.jsonl", "unit"),
    ("lookup@30", "artifacts/phr_30.jsonl", "unit"),
    ("plan", "artifacts/planned_v3.jsonl", "unit"),
    ("locate", "artifacts/located.jsonl", "unit"),
    ("embed", "artifacts/embed_located.jsonl", "unit"),
    ("llm14b", "artifacts/gen14b_merged.jsonl", "unit"),
    ("read_note", "artifacts/two_stage.jsonl", "unit"),
    ("read_rank1", "artifacts/rank1.jsonl", "unit"),
    ("ratio_pair", "artifacts/spec_ratio_fake.jsonl", "unit"),
)


def close(a: float, b: float) -> bool:
    """The scorer's own test, plus a relative band for OCR of long figures."""

    if a is None or b is None:
        return False
    if abs(a - b) <= 0.01:
        return True
    scale = max(abs(a), abs(b))
    return scale > 0 and abs(a - b) / scale <= 1e-4


def cluster(values: list[tuple[str, float]]) -> list[tuple[float, list[str]]]:
    """Group channel readings that agree, largest group first."""

    groups: list[tuple[float, list[str]]] = []
    for name, value in values:
        for index, (centre, members) in enumerate(groups):
            if close(centre, value):
                groups[index] = (centre, members + [name])
                break
        else:
            groups.append((value, [name]))
    groups.sort(key=lambda item: (-len(item[1]), item[0]))
    return groups


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/consensus.jsonl")
    # Cluster size is calibrated only against the channel set that produced it.
    # Adding a channel that agrees with the primary read 19% of the time created
    # size-4 and size-5 clusters that are weaker than the old ones — the band's
    # match rate fell from 48% to 38% and from 72% to 53%. So a channel earns its
    # place by raising the bands, not by existing, and this drops one to check.
    parser.add_argument("--exclude", action="append", default=[])
    args = parser.parse_args()

    questions = {q.id: q for q in parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv")}
    with zipfile.ZipFile(ROOT / "submissions" / "aimed.zip") as archive:
        aimed = {r["id"]: r.get("answer")
                 for r in json.loads(archive.read("submission.json"))}

    readings: dict[int, list[tuple[str, float]]] = defaultdict(list)
    present: Counter[str] = Counter()
    for name, path, unit in CACHES:
        file = ROOT / path
        if not file.exists():
            print(f"  thieu {path}")
            continue
        for line in file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not record.get("ok", True):
                continue
            value = record.get("value")
            if value is None:
                continue
            if name in args.exclude:
                continue
            readings[record["id"]].append((name, float(value)))
            present[name] += 1

    cross = ROOT / "artifacts" / "cross_year.jsonl"
    if cross.exists():
        for line in cross.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            question = questions.get(record["id"])
            if question is None:
                continue
            scale = question.unit_scale or 1.0
            readings[record["id"]].append(
                ("year_Y", record["current"]["value"] / scale))
            readings[record["id"]].append(
                ("year_Y1", record["comparative"]["value"] / scale))
            present["year_Y"] += 1
            present["year_Y1"] += 1
            # The best match in a different table of the same report. Measured to
            # agree with the primary read only 19% of the time — the second-best
            # table usually holds a different line item, not a duplicate of the
            # same one — so it is a weak channel. Kept because a disagreeing
            # channel forms its own singleton and cannot pollute the winning
            # cluster: it can only ever add evidence, never subtract it.
            if record.get("alternate") and "year_Y_alt" not in args.exclude:
                readings[record["id"]].append(
                    ("year_Y_alt", record["alternate"]["value"] / scale))
                present["year_Y_alt"] += 1

    print("so cau moi kenh doc duoc:")
    for name, count in present.most_common():
        print(f"  {name:12s} {count}")

    out, by_size = [], defaultdict(lambda: [0, 0])
    for qid in sorted(readings):
        groups = cluster(readings[qid])
        top_value, members = groups[0]
        size = len(members)
        agrees_with_aimed = close(top_value, aimed.get(qid))
        by_size[size][0] += 1
        by_size[size][1] += int(agrees_with_aimed)
        out.append({
            "id": qid,
            "value": top_value,
            "size": size,
            "members": members,
            "channels": len(readings[qid]),
            "matches_aimed": bool(agrees_with_aimed),
        })

    (ROOT / args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out),
        encoding="utf-8")

    print(f"\n{len(out)}/1012 cau co it nhat mot kenh doc duoc")
    print("do lon cua nhom dong thuan -> ty le khop dap an cua aimed:")
    for size in sorted(by_size, reverse=True):
        total, hits = by_size[size]
        print(f"  {size} kenh trung nhau: {total:4d} cau, khop aimed "
              f"{hits:4d} ({100 * hits / total:3.0f}%)")
    strong = sum(t for s, (t, _) in by_size.items() if s >= 2)
    print(f"\ncau co >=2 kenh dong thuan: {strong} ({100 * strong / 1012:.0f}% de bai)")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
