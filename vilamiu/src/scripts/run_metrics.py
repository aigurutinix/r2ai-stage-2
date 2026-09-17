"""Build the Circular 200 metric panel and check it against accounting identities.

Two identities are free ground truth: total assets must equal total resources,
and must equal liabilities plus equity. A panel that fails them is misreading
codes, columns, or units — and unlike the leaderboard, it says so immediately.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.corpus.metrics import build_panel  # noqa: E402
from vifin.store import TableStore  # noqa: E402

TOLERANCE = 0.02


def check(panel, left: str, right: tuple[str, ...]) -> tuple[int, int]:
    ok = bad = 0
    for values in panel.values():
        if left not in values or not all(r in values for r in right):
            continue
        total = sum(values[r] for r in right)
        if abs(values[left] - total) <= TOLERANCE * max(abs(values[left]), 1.0):
            ok += 1
        else:
            bad += 1
    return ok, bad


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    panel = build_panel(store.frame)

    print(f"(ticker, year, scope) with at least one metric: {len(panel)}")
    counts = Counter()
    for values in panel.values():
        counts.update(values.keys())
    for name, n in counts.most_common():
        print(f"  {n:5d}  {name}")

    print()
    for left, right in (("total_assets", ("liabilities", "equity")),
                        ("total_resources", ("liabilities", "equity")),
                        ("total_assets", ("current_assets", "long_assets"))):
        ok, bad = check(panel, left, right)
        total = ok + bad
        label = f"{left} = {' + '.join(right)}"
        print(f"  {label:52s} {ok:4d} ok / {total:4d}" + (f"  ({ok / total:.1%})" if total else ""))

    import pandas as pd

    rows = [
        {"ticker": t, "year": y, "scope": s, **values}
        for (t, y, s), values in sorted(panel.items())
    ]
    out = root / "artifacts" / "metrics.parquet"
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"\nwrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
