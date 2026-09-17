"""Do the new panel addresses round-trip through the program the model would run?

The generator is about to write thousands of pairs whose target is
`result = num(df, r, c)`. If `r` or `c` is off, every one of them teaches a wrong
cell while the answer still looks right, and nothing downstream can tell —
`build_sft.py` already lost 4 of 505 pairs to exactly this, off by a column.

So the check goes the long way round on purpose: build the DataFrame the way the
sandbox builds it, read the cell with the same `num` helper the model is given,
apply the recorded scale, and compare against the panel figure.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.corpus.metrics import build_panel  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

TOL = 1e-6


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, provenance = build_panel(store.frame, with_provenance=True)

    stats: collections.Counter[str] = collections.Counter()
    offenders: list[str] = []

    for key, sources in provenance.items():
        for metric, source in sources.items():
            expected = panel[key][metric]
            grid = store.rows(TableKey(source.doc_name, source.table_id))

            # The frame the sandbox builds drops grid row 0 as the header, so the
            # program addresses row-1. Run it through `num` exactly as the model
            # would, rather than indexing the grid directly, so a wrong offset
            # here shows up as a wrong number rather than being assumed away.
            outcome = run_query(
                f"{PRELUDE}\nresult = num(df, {source.row - 1}, {source.column})"
                f" * {source.scale!r}",
                {"df": grid},
            )
            if not outcome.ok or outcome.value is None:
                stats["unreadable"] += 1
                continue
            got = float(outcome.value)
            if abs(got - expected) <= TOL * max(abs(expected), 1.0):
                stats["matches"] += 1
            else:
                stats["MISMATCH"] += 1
                if len(offenders) < 8:
                    offenders.append(
                        f"{key} {metric}: panel {expected:,.2f} vs cell {got:,.2f} "
                        f"@ {source.doc_name}|{source.table_id} "
                        f"[{source.row},{source.column}] x{source.scale}")

    total = sum(stats.values())
    print(f"{total} (group, metric) addresses checked")
    for name, count in stats.most_common():
        print(f"  {name:<12} {count:>6}  {count / total:6.2%}")
    for line in offenders:
        print(f"  ! {line}")

    # Derived metrics carry no address by design; confirm the gap is only those.
    addressed = sum(len(s) for s in provenance.values())
    figures = sum(len(v) for v in panel.values())
    print(f"\npanel figures {figures}, addressed {addressed}, "
          f"derived-only {figures - addressed}")


if __name__ == "__main__":
    main()
