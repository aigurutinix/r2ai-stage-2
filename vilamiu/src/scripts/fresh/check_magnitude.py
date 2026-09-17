"""The unit in the question constrains the answer's magnitude. Nothing here uses it yet.

The reads are verified at 90.9% across documents and the answers are 21% correct, so
line choice is running near 23%. Both attacks on it — token overlap and a model
choosing from a clean list — stall around 52% on the easiest subset, and neither uses
a signal sitting in plain sight.

Whoever wrote the question chose the unit so the answer would read naturally. "Bao
nhiêu nghìn tỷ đồng" is not asked when the answer is 0.00001, and "bao nhiêu triệu
đồng" is not asked when it is 4,300,000. So the stated unit pins the answer to roughly
one to four orders of magnitude, which excludes most of a statement's two hundred lines
before any label is compared.

Two uses, and this measures both:

  as a detector   an answer outside the plausible band is almost certainly wrong, and
                  that is measurable on the submissions already scored
  as a filter     candidates outside the band can be dropped before the label match,
                  which is where line choice is losing

Compares the shipped submissions against the band to see how much of the error it
catches.

Usage:  python scripts/fresh/check_magnitude.py --zip submissions/fresh_v7.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_submission import unit_of  # noqa: E402

# A number a Vietnamese question would state in the given unit. Wider than it needs
# to be on purpose: the point is to catch answers that are absurd, not to second-guess
# a plausible one.
BAND = (0.05, 1e6)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", action="append", default=[])
    parser.add_argument("--show", type=int, default=8)
    args = parser.parse_args()
    if not args.zip:
        args.zip = ["submissions/fresh_v7.zip", "submissions/aimed.zip"]

    for name in args.zip:
        path = ROOT / name
        if not path.exists():
            print(f"{name}: khong co file")
            continue
        with zipfile.ZipFile(path) as archive:
            rows = json.loads(archive.read("submission.json").decode("utf-8"))

        counters: Counter[str] = Counter()
        by_unit: dict[str, Counter] = defaultdict(Counter)
        samples = []
        for row in rows:
            if not (row.get("evidence") or []):
                continue
            unit_name, unit = unit_of(row["question"])
            if not unit:
                counters["don vi khong phai tien"] += 1
                continue
            try:
                value = abs(float(row["answer"]))
            except (TypeError, ValueError):
                counters["dap an khong phai so"] += 1
                continue
            counters["do duoc"] += 1
            if value == 0:
                counters["  bang 0"] += 1
                by_unit[unit_name]["ngoai dai"] += 1
            elif BAND[0] <= value <= BAND[1]:
                counters["  TRONG dai hop ly"] += 1
                by_unit[unit_name]["trong dai"] += 1
            else:
                counters["  NGOAI dai — gan nhu chac sai"] += 1
                by_unit[unit_name]["ngoai dai"] += 1
                if len(samples) < args.show:
                    samples.append(
                        f"    id={row['id']:<5d} {value:,.4g} {unit_name}\n"
                        f"       {row['question'][:92]}")

        print(f"\n=== {name}")
        for key, count in counters.most_common():
            print(f"  {key}: {count}")
        measured = counters["do duoc"]
        if measured:
            bad = counters["  NGOAI dai — gan nhu chac sai"] + counters["  bang 0"]
            print(f"  => ty le gan nhu chac sai: {100 * bad / measured:.1f}%")
        print("  theo don vi:")
        for unit_name, counter in sorted(by_unit.items(),
                                         key=lambda item: -sum(item[1].values())):
            total = sum(counter.values())
            print(f"    {unit_name:16s} trong dai {counter['trong dai']:4d} / "
                  f"{total:4d} ({100 * counter['trong dai'] / total:3.0f}%)")
        if samples:
            print("  vi du ngoai dai:")
            for line in samples:
                print(line)


if __name__ == "__main__":
    main()
