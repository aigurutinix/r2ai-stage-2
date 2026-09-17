"""Can a Vietnamese-tuned embedding resolve the panel cells the token match cannot?

The dead-zone panel fills 971 of 2,327 cells — 42%. The model's plans are read
correctly and then die because the figure they name cannot be found, so this is
the binding constraint, not reasoning.

Two things have to be true before an embedding is allowed to fill those holes,
and they are measured separately here:

*Precision.* On the cells the token match already solves — most at an exact label
match — does the embedding choose the same row? Agreement there is the only
evidence available offline that its answers elsewhere are worth having. This
project has measured the opposite case: embeddings used to *replace* the token
match cost 13 answers.

*Coverage.* On the cells the token match leaves empty, how many does it fill at
all, and at what similarity?

Usage:
  PYTHONPATH=src python scripts/probe_embed_fill.py --limit 400
  PYTHONPATH=src python scripts/probe_embed_fill.py --model BAAI/bge-m3
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod, panel2  # noqa: E402
from vifin.answering.embed_match import LabelEmbedder, best_label, clean_label  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402


def candidates(question, phrase, ticker, year, store, retriever, top_k):
    """The tables the panel builder would have looked in, same order."""

    probe = dataclasses.replace(question, question=phrase, tickers=[ticker], years=[year])
    keys = []
    for variant in lookup_mod.metric_variants(phrase):
        for hit in retriever.search(dataclasses.replace(probe, question=variant),
                                    top_k=top_k):
            meta = store.meta(hit.key)
            if str(meta.ticker) != str(ticker) or str(meta.year) != str(year):
                continue
            if hit.key not in keys:
                keys.append(hit.key)
    return keys, probe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=400)
    parser.add_argument("--model", default="AITeamVN/Vietnamese_Embedding")
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=8)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    parsed = {q.id: q for q in parse_all(
        root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")}
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)
    dead = json.loads((root / "artifacts" / "deadzone.json").read_text())

    print(f"loading {args.model} ...")
    embedder = LabelEmbedder.load(args.model)

    stats: collections.Counter = collections.Counter()
    scores_hit: list[float] = []
    scores_fill: list[float] = []
    disagreements: list[tuple] = []
    started = time.time()
    cells = 0

    for qid in dead:
        if cells >= args.limit:
            break
        question = parsed[qid]
        phrases = panel2.expand_formulas(panel2.metric_phrases(question))
        years = sorted(question.years)[-6:]
        for ticker in question.tickers:
            for year in years:
                for phrase in phrases:
                    if cells >= args.limit:
                        break
                    cells += 1
                    keys, probe = candidates(
                        question, phrase, ticker, year, store, retriever, args.top_k)
                    if not keys:
                        stats["khong co bang nao cua (ma, nam)"] += 1
                        continue
                    grids = [store.rows(k) for k in keys]

                    lexical = panel2._resolve(
                        question, phrase, ticker, year, store, retriever, args.top_k)
                    match = best_label(embedder, phrase, grids, args.min_score)

                    if lexical is not None:
                        stats["lexical giai duoc"] += 1
                        if match is None:
                            stats["  embed im lang"] += 1
                            continue
                        same = (lookup_mod._norm(clean_label(match.label))
                                == lookup_mod._norm(clean_label(lexical[1].label)))
                        stats["  embed CHON CUNG dong" if same
                              else "  embed chon KHAC dong"] += 1
                        scores_hit.append(match.score)
                        if not same and len(disagreements) < 8:
                            disagreements.append(
                                (qid, phrase[:30], lexical[1].label[:34],
                                 match.label[:34], round(match.score, 3)))
                    else:
                        stats["lexical IM LANG"] += 1
                        if match is None:
                            stats["  embed cung im lang"] += 1
                        else:
                            stats["  embed LAP DUOC"] += 1
                            scores_fill.append(match.score)

    print(f"\n{cells} o kiem tra, {time.time() - started:.0f}s\n")
    for key, count in sorted(stats.items()):
        print(f"  {count:5d}  {key}")

    def summary(name, values):
        if not values:
            print(f"\n{name}: khong co")
            return
        values = sorted(values)
        mid = values[len(values) // 2]
        print(f"\n{name}: n={len(values)} trung vi={mid:.3f} "
              f"p10={values[len(values) // 10]:.3f} p90={values[int(len(values) * 0.9)]:.3f}")

    summary("Diem tuong dong khi lexical cung giai duoc", scores_hit)
    summary("Diem tuong dong khi embed lap cho trong", scores_fill)

    if disagreements:
        print("\nVi du embed chon khac lexical:")
        for qid, phrase, lex, emb, score in disagreements:
            print(f"  id={qid:4d} {phrase!r}\n      lexical={lex!r}\n      embed  ={emb!r} ({score})")


if __name__ == "__main__":
    main()
