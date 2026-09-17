"""Did widening the unit scanner break what already worked?

The acceptance criterion set before the change was: keep agreement at or above 97.7% on
the documents whose scale is known from their statements. The change reported 97.2% —
but on 1,046 documents rather than 696, so the two rates describe different populations
and neither confirms nor refutes the criterion.

The honest test is per-population: run both variants over the same documents and compare
agreement on the set BOTH cover. If the old set holds and the newly covered documents
score lower, the change is a net win and the criterion passes as intended. If the old set
degraded, the wider marker window is admitting false positives and the change goes back.

  old rule   a unit declaration is a line of at most 60 characters carrying a marker
             and a đồng token; table headers are not read
  new rule   the length test applies to a 40-character window after the marker, and
             table headers are read too

Usage:  python scripts/fresh/compare_scale_scanners.py
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from document_scale import DONG_RE, MARKER_RE, SCALE_WORDS, WINDOW, fold  # noqa: E402

OLD_MAX_LINE = 60


def old_scale_of_line(line: str) -> float | None:
    if len(line) > OLD_MAX_LINE:
        return None
    flat = fold(line)
    if not MARKER_RE.search(flat) or not DONG_RE.search(flat):
        return None
    for pattern, scale in SCALE_WORDS:
        if re.search(pattern, flat):
            return scale
    return 1.0


def new_scale_of_line(line: str) -> float | None:
    flat = fold(line)
    match = MARKER_RE.search(flat)
    if match is None:
        return None
    span = flat[match.start():match.start() + WINDOW]
    if not DONG_RE.search(span):
        return None
    for pattern, scale in SCALE_WORDS:
        if re.search(pattern, span):
            return scale
    return 1.0


def modal(votes: Counter) -> float | None:
    return votes.most_common(1)[0][0] if votes else None


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="artifacts/fresh/statements2.jsonl")
    parser.add_argument("--tables", default="artifacts/fresh/all_tables.jsonl")
    args = parser.parse_args()

    truth: dict[str, Counter] = defaultdict(Counter)
    for line in (ROOT / args.index).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            truth[record["doc"]][record["scale"]] += 1
    known = {doc: modal(counter) for doc, counter in truth.items()}

    where: dict[str, tuple[str, str]] = {}
    for line in (ROOT / args.tables).read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            where.setdefault(record["doc"], (record["ticker"], record["year"]))

    # Only the documents that can be validated at all.
    docs = sorted(doc for doc in where if doc in known)
    print(f"{len(docs)} tai lieu co the kiem duoc", flush=True)

    stats: Counter[str] = Counter()
    for doc in docs:
        ticker, year = where[doc]
        base = ROOT / "data" / "official_corpus" / ticker / year / doc
        path = base / f"{doc}_extracted.txt"
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        headers = []
        table_dir = base / f"{doc}_extracted_tables"
        if table_dir.is_dir():
            for csv_path in sorted(table_dir.glob("table_*.csv")):
                try:
                    with csv_path.open(encoding="utf-8-sig", newline="") as file:
                        first = next(csv_mod.reader(file), None)
                except OSError:
                    continue
                if first:
                    headers.append(",".join(str(c) for c in first))

        old_votes: Counter[float] = Counter()
        for line in lines:
            scale = old_scale_of_line(line)
            if scale is not None:
                old_votes[scale] += 1
        new_votes: Counter[float] = Counter()
        for line in lines + headers:
            scale = new_scale_of_line(line)
            if scale is not None:
                new_votes[scale] += 1

        old_scale, new_scale = modal(old_votes), modal(new_votes)
        target = known[doc]
        if old_scale is not None and new_scale is not None:
            stats["ca hai phu"] += 1
            stats["  cu dung"] += abs(old_scale - target) < 1e-9
            stats["  moi dung"] += abs(new_scale - target) < 1e-9
        elif new_scale is not None:
            stats["chi MOI phu"] += 1
            stats["  moi dung (tren tap moi)"] += abs(new_scale - target) < 1e-9
        elif old_scale is not None:
            stats["chi CU phu"] += 1
            stats["  cu dung (moi mat)"] += abs(old_scale - target) < 1e-9
        else:
            stats["ca hai khong phu"] += 1

    both = stats["ca hai phu"]
    only_new = stats["chi MOI phu"]
    print()
    for name, count in stats.most_common():
        print(f"  {name}: {count}")
    if both:
        print(f"\ntren {both} tai lieu CA HAI cung phu:")
        print(f"  cu  : {100 * stats['  cu dung'] / both:.1f}%")
        print(f"  moi : {100 * stats['  moi dung'] / both:.1f}%")
    if only_new:
        print(f"\ntren {only_new} tai lieu CHI BAN MOI phu: "
              f"{100 * stats['  moi dung (tren tap moi)'] / only_new:.1f}% dung")
    print("\ntieu chi: ban moi khong duoc kem ban cu tren tap giao.")


if __name__ == "__main__":
    main()
