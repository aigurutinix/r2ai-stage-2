"""Rank tables for each question by BM25 over the surrounding prose.

Same BM25 constants, same tokenizer and the same metadata pre-filter as
`LexicalRetriever`, so the only thing that changes against the current ranking is
the text being scored: `context_index.jsonl` (company name, columns, first 20 row
labels, and 2,500 characters of the page's narrative) instead of caption plus
labels alone.

Only terms that occur in at least one of the 1,012 questions are indexed. Nothing
else can ever contribute to a score, and skipping the rest is what keeps the
posting lists small enough to hold in memory.

Usage:  PYTHONPATH=src python scripts/rank_context.py [top_n] [index.jsonl] [out.jsonl]
"""

from __future__ import annotations

import array
import json
import os
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import B, K1, LexicalRetriever, tokenize  # noqa: E402
from vifin.store import TableKey  # noqa: E402


def main() -> None:
    top_n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    index_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "artifacts" / "context_index.jsonl"
    out_path = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT / "artifacts" / "context_rank.jsonl"
    # A flat list collapses onto whichever company matched best, which is why
    # run_generate measured executable programs falling from 49.9% to 26.4% when
    # fed the reranked top-30. `balanced` reserves slots per (ticker, year) the
    # way LexicalRetriever.search_balanced does, and writes the {"keys": [...]}
    # shape the generation scripts already read.
    balanced = len(sys.argv) > 4 and sys.argv[4] == "balanced"
    per_group = int(os.environ.get('PER_GROUP', '2'))
    group_cap = int(os.environ.get('GROUP_CAP', '8'))

    # `QUESTIONS_FILE` lets this rank a different question set than the exam's.
    # It exists so the local gold set can be scored through the *same* ranking the
    # submission uses. Scoring it against `LexicalRetriever` instead measured a
    # system we do not ship, and comparing against that weaker baseline is how a
    # patch that looked like a gain cost 24 questions on 12/08.
    questions = parse_all(
        Path(os.environ.get("QUESTIONS_FILE",
                            ROOT / "data" / "questions" / "questions.jsonl")),
        ROOT / "data" / "code_stock.csv")
    frame = pd.read_parquet(
        ROOT / "artifacts" / "tables.parquet",
        columns=["doc_name", "ticker", "year", "scope", "table_id", "eligible"])
    retriever = LexicalRetriever(frame)

    wanted_docs: set[str] = set()
    query_tokens: dict[int, Counter] = {}
    for question in questions:
        wanted_docs.update(retriever.candidate_docs(question))
        query_tokens[question.id] = Counter(tokenize(question.question))
    vocabulary = {term for counts in query_tokens.values() for term in counts}
    print(f"{len(questions)} questions, {len(wanted_docs)} candidate documents, "
          f"{len(vocabulary)} distinct query terms")

    started = time.time()
    keys: list[TableKey] = []
    position_of: dict[tuple[str, int], int] = {}
    lengths: list[int] = []
    postings: dict[str, tuple[array.array, array.array]] = {
        term: (array.array("i"), array.array("i")) for term in vocabulary
    }

    with index_path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            record = json.loads(line)
            doc_name = record["doc_name"]
            if doc_name not in wanted_docs:
                continue
            position = len(keys)
            table_id = int(record["table_id"])
            keys.append(TableKey(doc_name, table_id))
            position_of[(doc_name, table_id)] = position
            tokens = tokenize(record["text"])
            lengths.append(len(tokens) or 1)
            for term, count in Counter(tokens).items():
                entry = postings.get(term)
                if entry is not None:
                    entry[0].append(position)
                    entry[1].append(count)
            if line_no % 20000 == 0 and line_no:
                print(f"  read {line_no} lines, kept {len(keys)}, {time.time() - started:.0f}s")

    total = len(keys)
    print(f"indexed {total} tables in {time.time() - started:.0f}s")
    doc_length = np.array(lengths, dtype=np.float32)
    posting_arrays = {
        term: (np.frombuffer(p, dtype=np.int32), np.frombuffer(t, dtype=np.int32).astype(np.float32))
        for term, (p, t) in postings.items() if len(p)
    }
    del postings

    positions_by_doc: dict[str, list[int]] = {}
    for (doc_name, _), position in position_of.items():
        positions_by_doc.setdefault(doc_name, []).append(position)

    group_of: dict[str, tuple[str, str]] = {
        str(r.doc_name): (str(r.ticker), str(r.year))
        for r in frame[["doc_name", "ticker", "year"]].drop_duplicates().itertuples(index=False)
    }

    ranked = 0
    empty = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            pool = np.fromiter(
                (p for d in retriever.candidate_docs(question) for p in positions_by_doc.get(d, ())),
                dtype=np.int64)
            if pool.size == 0:
                handle.write(json.dumps({"id": question.id, "refs": []}) + "\n")
                empty += 1
                continue

            in_pool = np.zeros(total, dtype=bool)
            in_pool[pool] = True
            average = float(doc_length[pool].mean())
            scores = np.zeros(total, dtype=np.float32)

            for term, weight in query_tokens[question.id].items():
                entry = posting_arrays.get(term)
                if entry is None:
                    continue
                where, frequency = entry
                selected = in_pool[where]
                hits = int(selected.sum())
                if not hits:
                    continue
                idf = math.log(1 + (pool.size - hits + 0.5) / (hits + 0.5))
                position = where[selected]
                freq = frequency[selected]
                denominator = freq + K1 * (1 - B + B * doc_length[position] / average)
                scores[position] += idf * weight * freq * (K1 + 1) / denominator

            order = pool[np.argsort(-scores[pool], kind="stable")]
            if balanced:
                groups: dict[tuple[str, str], list[int]] = {}
                for p in order:
                    if scores[p] <= 0:
                        continue
                    groups.setdefault(group_of[keys[p].doc_name], []).append(int(p))
                picked: list[int] = []
                for rank in range(per_group):
                    for members in groups.values():
                        if rank < len(members):
                            picked.append(members[rank])
                picked.sort(key=lambda p: -scores[p])
                best = picked[:group_cap]
            else:
                best = [int(p) for p in order[:top_n] if scores[p] > 0]

            if balanced:
                handle.write(json.dumps(
                    {"id": question.id,
                     "keys": [[keys[p].doc_name, keys[p].table_id] for p in best]}) + "\n")
            else:
                # The score travels with the key. Declaring a fixed top-k treats
                # rank 6 as worth as much on a question where it scores 0.4 of
                # the leader as on one where it scores 0.95, and the board says
                # that costs: 65% of declared refs name a gold document at a
                # non-gold line. A consumer can only cut relative to the leader
                # if it can see how far behind the tail sits.
                refs = [{"doc_name": keys[p].doc_name, "table_id": keys[p].table_id,
                         "score": round(float(scores[p]), 4)} for p in best]
                handle.write(json.dumps({"id": question.id, "refs": refs}) + "\n")
            ranked += 1

    print(f"ranked {ranked} questions ({empty} with no candidate table) "
          f"in {time.time() - started:.0f}s")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
