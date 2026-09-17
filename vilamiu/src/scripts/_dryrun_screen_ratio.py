"""Dry-run resolve_screen_ratio vs helpers.zip — no zip build."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

from vifin.answering import compose
from vifin.answering.sandbox import run_query
from vifin.query.parse import parse_all
from vifin.retrieval.lexical import LexicalRetriever
from vifin.store import TableStore

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parsed = {
        p.id: p
        for p in parse_all(
            ROOT / "data/questions/questions.jsonl",
            ROOT / "data/code_stock.csv",
        )
    }
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    ret = LexicalRetriever(store.frame)
    with zipfile.ZipFile(ROOT / "submissions" / "helpers.zip") as z:
        raw = json.loads(z.read("submission.json"))
    preds = {p["id"]: p for p in (raw if isinstance(raw, list) else raw["predictions"])}

    hits = []
    for i, q in parsed.items():
        got = compose.resolve_screen_ratio(q, store, ret)
        if got is None:
            continue
        outcome = run_query(got.code, {"df": store.rows(got.key)})
        old = preds[i].get("answer")
        hits.append(
            {
                "id": i,
                "winner": got.winner,
                "filter": got.filter_metric[:50],
                "label": got.label[:50],
                "new": outcome.value if outcome.ok else None,
                "old": old,
                "same": outcome.ok and outcome.value == old,
                "q": q.question[:120],
            }
        )

    out = ROOT / "artifacts" / "_dryrun_screen_ratio.txt"
    lines = [f"resolve_screen_ratio hits: {len(hits)}"]
    for h in hits:
        lines.append(
            f"id={h['id']} winner={h['winner']} same={h['same']} "
            f"new={h['new']} old={h['old']}"
        )
        lines.append(f"  filter={h['filter']!r} label={h['label']!r}")
        lines.append(f"  {h['q']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out)
    print(f"hits={len(hits)} changed={sum(1 for h in hits if not h['same'])}")
    for h in hits:
        print(h["id"], h["winner"], "same" if h["same"] else "DIFF", h["new"], "<-", h["old"])


if __name__ == "__main__":
    main()
