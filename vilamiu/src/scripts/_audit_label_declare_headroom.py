"""Measure label-scan DECLARE headroom beyond the BM25/rerank shortlist."""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.answering.corroborate import Corroborator  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402


def load_reranked(path: Path) -> dict[int, list[TableKey]]:
    order: dict[int, list[TableKey]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        order[row["id"]] = [TableKey(doc, int(tid)) for doc, tid in row["keys"]]
    return order


def main() -> None:
    t0 = time.time()
    ts = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    parsed = parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv",
    )
    retr = LexicalRetriever(ts.frame)
    corr = Corroborator(ts)
    order = load_reranked(ROOT / "artifacts" / "reranked_metric.jsonl")

    n_docs_all: list[int] = []
    n_tables_all: list[int] = []
    for q in parsed:
        docs = retr.candidate_docs(q)
        n_docs_all.append(len(docs))
        n_tables_all.append(sum(len(retr._rows_by_doc.get(d, [])) for d in docs))
    print(
        f"candidate_docs ALL: mean_docs={statistics.mean(n_docs_all):.2f} "
        f"median={statistics.median(n_docs_all):.0f} "
        f"mean_tables={statistics.mean(n_tables_all):.1f} "
        f"median={statistics.median(n_tables_all):.0f} "
        f"p90={sorted(n_tables_all)[int(0.9 * len(n_tables_all))]} "
        f"max={max(n_tables_all)} total_scans={sum(n_tables_all)}"
    )

    t1 = time.time()
    for q in parsed[:40]:
        corr.keys_matching_label(q, retr.candidate_docs(q), limit=10)
    dt = time.time() - t1
    print(f"40 full label scans: {dt:.1f}s -> est all ~{dt / 40 * len(parsed):.0f}s")

    # Shortlist vs full-doc label hits
    both = short_only = doc_extra = neither = 0
    shortlist_missing_top = 0  # doc's #1 not in shortlist
    extra_examples: list[tuple] = []
    hit_counts: Counter[int] = Counter()
    doc_hit_counts: Counter[int] = Counter()
    unit_qs = 0
    single_qs = 0

    # Also: among questions where answer evidence key exists, is it in declare top-1?
    # Use shape vs reorder first-ref identity as proxy.

    for q in parsed:
        if not q.unit_scale:
            continue
        unit_qs += 1
        bm25 = [h.key for h in retr.search(q, top_k=10)]
        searched = (order.get(q.id) or bm25)[:10]
        short_hits: list[TableKey] = []
        for key in searched:
            found = lookup_mod.find(ts.rows(key), q)
            if found is not None and found.score >= lookup_mod.MIN_LABEL_SCORE:
                short_hits.append(key)
        hit_counts[len(short_hits)] += 1

        if not lookup_mod.is_single_lookup(q.question):
            continue
        single_qs += 1
        doc_hits = corr.keys_matching_label(q, retr.candidate_docs(q), limit=10)
        doc_hit_counts[len(doc_hits)] += 1
        if short_hits and doc_hits:
            both += 1
        elif short_hits:
            short_only += 1
        elif doc_hits:
            doc_extra += 1
            if len(extra_examples) < 6:
                extra_examples.append(
                    (q.id, [(k.doc_name, k.table_id) for k in doc_hits[:3]])
                )
        else:
            neither += 1

        if doc_hits and doc_hits[0] not in set(searched):
            shortlist_missing_top += 1

    print(f"unit_scale qs={unit_qs} single_lookup={single_qs}")
    print(
        f"single_lookup label presence: both={both} short_only={short_only} "
        f"doc_extra_only={doc_extra} neither={neither}"
    )
    print(f"doc top-1 label hit NOT in shortlist: {shortlist_missing_top}/{single_qs}")
    print(f"label hits in top-10 (unit qs): {dict(sorted(hit_counts.items()))}")
    print(f"label hits in full-doc top10 (single): {dict(sorted(doc_hit_counts.items()))}")
    print("doc_extra examples:", extra_examples)

    # Declare strategy simulation proxies WITHOUT gold:
    # A) evidence-only count from zips
    # B) how many shortlist slots are non-label after evidence
    # C) how many label fillers shortlist provides vs doc-scan
    import zipfile

    with zipfile.ZipFile(ROOT / "submissions" / "label_reorder.zip") as z:
        reorder = {r["id"]: r for r in json.loads(z.read("submission.json"))}

    # rebuild start_line -> key
    sl_to_key: dict[tuple[str, int], TableKey] = {}
    for row in ts.frame.itertuples(index=False):
        sl_to_key[(row.doc_name, int(row.start_line))] = TableKey(
            row.doc_name, int(row.table_id)
        )

    label_fill_short = 0
    label_fill_doc = 0
    pad_noise_short = 0
    n_compare = 0
    for q in parsed:
        if not q.unit_scale:
            continue
        rec = reorder[q.id]
        refs = []
        for ref in rec["relevant_tables"]:
            doc, line = ref.split("|", 1)
            key = sl_to_key.get((doc, int(line)))
            if key is not None:
                refs.append(key)
        if len(refs) < 2:
            continue
        evid = refs[:1]  # rough: first is usually evidence under our policy
        # Better: parse evidence csv back to keys
        evid_keys: list[TableKey] = []
        for e in rec.get("evidence") or []:
            name = e["csv_path"].split("/")[-1]
            if name.endswith(".csv") and "_table_" in name:
                base = name[: -len(".csv")]
                doc, tid = base.rsplit("_table_", 1)
                evid_keys.append(TableKey(doc, int(tid)))
        if not evid_keys:
            continue
        n_compare += 1
        fillers = [k for k in refs if k not in set(evid_keys)]
        # score fillers
        for k in fillers:
            found = lookup_mod.find(ts.rows(k), q)
            if found is not None and found.score >= lookup_mod.MIN_LABEL_SCORE:
                label_fill_short += 1
            else:
                pad_noise_short += 1
        # how many doc-scan labels not in declare
        doc_hits = corr.keys_matching_label(q, retr.candidate_docs(q), limit=5)
        for k in doc_hits:
            if k not in set(refs) and k not in set(evid_keys):
                label_fill_doc += 1
                break  # count questions with unused doc label

    print(
        f"among unit qs with evidence ({n_compare}): "
        f"filler slots labelish={label_fill_short} noise={pad_noise_short} "
        f"(noise_rate={pad_noise_short / max(1, label_fill_short + pad_noise_short):.1%})"
    )
    print(
        f"questions with unused full-doc label hit outside declare: {label_fill_doc}"
    )
    print(f"elapsed {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
