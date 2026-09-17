"""Does the lexical pool index rank the right table, and how fast does it build?

The dense pool index was replaced because it re-embedded a few hundred tables per
seed on CPU. Speed alone is not a reason to swap: an index that builds instantly and
ranks badly turns a 12-day run into a fast way to generate rubbish. So this measures
both, on real tables, against the gold `relevant_tables` of the organisers' own
generator output.

Pool size matches `POOL_TABLE_CAP` (300) so the timing is the timing that matters.
"""

from __future__ import annotations

import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "vifinqa-official" / "src"))

from vifinqa.generation.embedding_index import (  # noqa: E402
    IndexedTable,
    LexicalTableIndex,
)

from vifin.store import TableKey, TableStore  # noqa: E402

POOL = 300
REF_RE = re.compile(r"^(?P<doc>.+)\|table_(?P<tid>\d+)$")


def table_text(store: TableStore, key: TableKey) -> str:
    """Roughly what `table_retrieval_text` builds: caption plus labels."""

    grid = store.rows(key)
    meta = store.meta(key)
    header = " ".join(str(c) for c in grid[0]) if grid else ""
    labels = " ".join(str(row[0]) for row in grid[1:40] if row)
    return f"{meta.caption} {header} {labels}"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    records = [
        json.loads(line)
        for line in (ROOT / "artifacts" / "easy_416.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]

    all_keys = [TableKey(d, int(t)) for d, t in
                store.frame[["doc_name", "table_id"]].itertuples(index=False)]
    rng = random.Random(0)

    build_times: list[float] = []
    search_times: list[float] = []
    hits = {1: 0, 5: 0, 10: 0}
    scored = 0

    for record in records[:60]:
        gold = []
        for ref in record.get("relevant_tables") or []:
            match = REF_RE.match(ref)
            if match:
                key = TableKey(match.group("doc"), int(match.group("tid")))
                if key in store:
                    gold.append(key)
        if not gold:
            continue

        # A realistic pool: the gold tables plus distractors, capped like the real one.
        pool = list(gold) + rng.sample(all_keys, POOL - len(gold))
        entries = [
            IndexedTable(
                ticker="", year="", doc_name=key.doc_name,
                table_id=key.table_id, text=table_text(store, key),
            )
            for key in pool
        ]

        start = time.perf_counter()
        index = LexicalTableIndex(entries)
        build_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        found = index.search(record["question"], top_k=10)
        search_times.append(time.perf_counter() - start)

        ranked = [(e.doc_name, e.table_id) for e, _ in found]
        wanted = {(k.doc_name, k.table_id) for k in gold}
        scored += 1
        for cutoff in (1, 5, 10):
            if wanted & set(ranked[:cutoff]):
                hits[cutoff] += 1

    n = scored or 1
    print(f"pool={POOL} tables, {scored} gold questions scored")
    print(f"  build  : {sum(build_times) / len(build_times) * 1000:.0f} ms per index")
    print(f"  search : {sum(search_times) / len(search_times) * 1000:.1f} ms per query")
    for cutoff in (1, 5, 10):
        print(f"  gold table in top-{cutoff:<2}: {hits[cutoff] / n:.1%}")
    per_seed = (sum(build_times) + sum(search_times)) / n
    print(f"\n  {per_seed * 1000:.0f} ms of index work per seed "
          f"-> {3600 / max(per_seed, 1e-9):,.0f} seeds/hour on this alone")


if __name__ == "__main__":
    main()
