"""Which gate inside build_panel empties the 599 silent groups.

Their tables carry a median of six alias-matching row labels, so the labels are
found. That leaves two gates: `extract_located`, which must also locate a column
and parse a value, and the identity verifier, which deletes a group outright when
its balance figures contradict each other and nothing else survives.

Turning each off in turn says which, and costs three panel builds.

Usage:  PYTHONPATH=src python scripts/_panel_gate_ab.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.corpus.metrics import build_panel  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    frame = store.frame
    corpus = {(str(t), str(y), str(s))
              for t, y, s in zip(frame["ticker"], frame["year"], frame["scope"])}
    print(f"corpus groups: {len(corpus)}\n")

    for label, kwargs in (
        ("as shipped", {}),
        ("verify off", {"verify": False}),
        ("label-only values allowed", {"allow_label_only": True}),
        ("both off", {"verify": False, "allow_label_only": True}),
    ):
        panel = build_panel(frame, **kwargs)
        cells = sum(len(v) for v in panel.values())
        print(f"  {label:28s} {len(panel):5d} groups "
              f"({100 * len(panel) / len(corpus):5.1f}% of corpus), {cells} cells")


if __name__ == "__main__":
    main()
