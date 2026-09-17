"""Where the retrieval score is actually lost: the line, not the document.

The board reports both halves of the same declaration. DOCS PRECISION is 0.9705
and TABLES PRECISION is 0.3357 on identical refs — the document part of nearly
every ref we declare is a gold document, while only a third of the refs are gold
tables. The rest name the right report at the wrong line.

That is the largest headroom on the board and it needs no model: it is a ranking
problem inside a document whose identity is already solved. This prints the shape
of the declaration so the arithmetic behind that claim is checkable.

Usage:  python scripts/_probe_table_vs_doc.py [zip name]
"""

from __future__ import annotations

import json
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = sys.argv[1] if len(sys.argv) > 1 else "screen_ratio_gated.zip"

# From the public board, for the submission built by this configuration.
BOARD = {
    "TABLES_P": 0.3357, "TABLES_R": 0.7213, "TABLES_F2": 0.5641,
    "DOCS_P": 0.9705, "DOCS_R": 0.9647, "DOCS_F2": 0.9628,
}


def main() -> None:
    with zipfile.ZipFile(ROOT / "submissions" / NAME) as z:
        recs = json.loads(z.read("submission.json"))

    tables = [len(r["relevant_tables"]) for r in recs]
    docs = [len({t.split("|", 1)[0] for t in r["relevant_tables"]}) for r in recs]
    declared_docs = [len(r["relevant_docs"]) for r in recs]

    print(f"{NAME}: {len(recs)} questions")
    print(f"  table refs per question : mean {statistics.mean(tables):.2f} "
          f"median {statistics.median(tables)} max {max(tables)}")
    print(f"  distinct docs inside    : mean {statistics.mean(docs):.2f}")
    print(f"  relevant_docs declared  : mean {statistics.mean(declared_docs):.2f}")
    print(f"  refs per doc            : {statistics.mean(tables) / statistics.mean(docs):.2f}")
    print(f"  questions declaring 1 doc: {sum(1 for d in docs if d == 1)}")
    print(f"  ref count histogram     : "
          f"{dict(sorted(Counter(tables).items()))}")

    p_t, p_d = BOARD["TABLES_P"], BOARD["DOCS_P"]
    print()
    print(f"  precision on documents : {p_d:.4f}")
    print(f"  precision on lines     : {p_t:.4f}")
    print(f"  share of declared refs in a gold doc but at a non-gold line: "
          f"{1 - p_t / p_d:.1%}")
    print()
    print("  what closing that gap is worth, holding recall and k fixed:")
    # F2 = 5h / (4g + k). Recall r = h/g, so F2 = 5r / (4 + k/g).
    r, k_over_g = BOARD["TABLES_R"], None
    # Solve k/g from the observed F2 and recall.
    k_over_g = 5 * r / BOARD["TABLES_F2"] - 4
    print(f"    implied k/g from F2 and recall: {k_over_g:.2f}")
    for target_r in (0.75, 0.80, 0.85, 0.90):
        f2 = 5 * target_r / (4 + k_over_g)
        print(f"    recall {target_r:.2f} at the same k -> TABLES_F2 {f2:.4f}")
    for shrink in (0.8, 0.6, 0.5):
        f2 = 5 * r / (4 + k_over_g * shrink)
        print(f"    same recall, k x{shrink:.1f} -> TABLES_F2 {f2:.4f} "
              f"(macro +{(f2 - BOARD['TABLES_F2']) / 3:.4f})")


if __name__ == "__main__":
    main()
