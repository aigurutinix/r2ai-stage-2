"""Self-test for `ab_heldout.parse_target` before an A/B depends on it.

If the parser cannot read the gold targets it will silently score both arms zero
and the comparison will look like "the adapter changed nothing".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ab_heldout import parse_target  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = [
        json.loads(line)
        for line in (ROOT / "artifacts" / "sft_easy.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    failures: list[str] = []
    scales: dict[float, int] = {}
    for row in rows:
        target = row["messages"][-1]["content"]
        got = parse_target(target)
        if got is None:
            failures.append(target.replace("\n", " ; "))
            continue
        scales[got[3]] = scales.get(got[3], 0) + 1

    print(f"parsed {len(rows) - len(failures)}/{len(rows)} gold targets")
    for scale, count in sorted(scales.items()):
        print(f"  scale {scale:<10g} {count:>4}")
    for target in failures[:10]:
        print(f"  FAILED: {target[:120]}")

    # Free wins are not wins: if one scale covers almost everything, a model that
    # always emits it scores well on `scale` without understanding units.
    top = max(scales.values()) / len(rows)
    print(f"\nmajority-scale baseline: {top:.1%} — read the `scale` column against this")


if __name__ == "__main__":
    main()
