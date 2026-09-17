"""Run the organisers' BM25 retrieval exactly as their package defines it.

Their leaderboard row reports TABLES_F2 0.8934 where ours is 0.5494. That is a
60% relative gap, which is not a tuning difference — it is a different system.
This reproduces theirs rather than approximating it, so the number either
transfers or the row is an oracle and the question closes for good.

Three things differ from `LexicalRetriever`:

* `underthesea.word_tokenize` segments Vietnamese into words. Vietnamese is
  written one syllable at a time, so "lợi nhuận" is one word spelled as two
  tokens; our character-class regex plus hand-rolled bigrams only approximates
  that.
* `bm25s` rather than a per-question scorer.
* The index text is `table_retrieval_text` — metadata, columns, the first 20 row
  labels and 2,500 characters of the page — which `artifacts/context_index.jsonl`
  already holds, built with the same encoder.

Their pipeline also searches the whole corpus with no metadata pre-filter. That
is kept faithful here; `--filter` adds our filter back as a separate variant, so
the two effects stay separable.

Usage:
  PYTHONPATH=src python scripts/run_bm25_official.py --top-k 30
  PYTHONPATH=src python scripts/run_bm25_official.py --top-k 30 --filter
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import bm25s  # noqa: E402
from underthesea import word_tokenize  # noqa: E402

from vifin.query.parse import parse_all  # noqa: E402
from vifin.store import TableKey  # noqa: E402

WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    """`_normalize_text` from vifinqa/retrieval/bm25.py, verbatim in behaviour."""

    flat = unicodedata.normalize("NFC", text).replace(" ", " ").casefold()
    return WHITESPACE_RE.sub(" ", flat).strip()


def tokenize(text: str) -> list[str]:
    return word_tokenize(normalize(text))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--index", default="artifacts/context_index.jsonl")
    parser.add_argument("--out", default="artifacts/bm25_official_rank.jsonl")
    parser.add_argument("--filter", action="store_true",
                        help="restrict to our metadata-filtered documents")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    # Segmented tokens are cached per index text, since the two indexes
    # segment differently and 47 minutes is too long to pay twice.
    parser.add_argument("--token-cache", default="artifacts/bm25_tokens.tsv")
    args = parser.parse_args()

    started = time.time()
    keys: list[TableKey] = []
    texts: list[str] = []
    with (ROOT / args.index).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            keys.append(TableKey(row["doc_name"], int(row["table_id"])))
            texts.append(row["text"])
    print(f"{len(texts)} bảng trong chỉ mục, {time.time() - started:.0f}s")

    # Word segmentation runs at ~41 documents/second single-threaded, so the full
    # corpus costs 45 minutes. Pay it once, cache it, and fan it across cores —
    # tokens are joined with tabs because a segmented token contains spaces
    # ("Lợi nhuận" is one token).
    SEP = chr(9)
    cache_path = ROOT / args.token_cache
    if cache_path.exists():
        corpus = [line.split(SEP) for line in
                  cache_path.read_text(encoding="utf-8").splitlines()]
        print(f"nạp {len(corpus)} bảng đã tách từ khỏi cache, {time.time() - started:.0f}s")
        if len(corpus) != len(texts):
            raise SystemExit(f"cache lệch: {len(corpus)} so với {len(texts)} bảng")
    else:
        print("tách từ bằng underthesea (chậm; chạy song song, cache lại) ...")
        from concurrent.futures import ProcessPoolExecutor
        corpus = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for position, tokens in enumerate(
                    pool.map(tokenize, texts, chunksize=200), start=1):
                corpus.append(tokens)
                if position % 20000 == 0:
                    print(f"  {position}/{len(texts)}  {time.time() - started:.0f}s")
        cache_path.write_text(
            chr(10).join(SEP.join(t) for t in corpus), encoding="utf-8")
        print(f"tách từ xong và đã cache, {time.time() - started:.0f}s")

    model = bm25s.BM25()
    model.index(corpus, show_progress=False)
    print(f"đã dựng chỉ mục bm25s, {time.time() - started:.0f}s")

    questions = parse_all(
        ROOT / "data" / "questions" / "questions.jsonl", ROOT / "data" / "code_stock.csv")
    if args.limit:
        questions = questions[: args.limit]

    allowed: dict[int, set[str]] = {}
    if args.filter:
        import pandas as pd
        from vifin.retrieval.lexical import LexicalRetriever
        frame = pd.read_parquet(
            ROOT / "artifacts" / "tables.parquet",
            columns=["doc_name", "ticker", "year", "scope", "table_id", "eligible"])
        retriever = LexicalRetriever(frame)
        for question in questions:
            allowed[question.id] = set(retriever.candidate_docs(question))
        print("bộ lọc metadata BẬT")

    out_path = ROOT / args.out
    # Retrieve deeper when filtering, since most hits are dropped afterwards.
    depth = args.top_k * (20 if args.filter else 1)
    with out_path.open("w", encoding="utf-8") as handle:
        for position, question in enumerate(questions, start=1):
            hits, _ = model.retrieve(
                [tokenize(question.question)], k=min(depth, len(corpus)), show_progress=False)
            picked: list[TableKey] = []
            for index in hits[0]:
                key = keys[int(index)]
                if args.filter and key.doc_name not in allowed.get(question.id, set()):
                    continue
                picked.append(key)
                if len(picked) >= args.top_k:
                    break
            handle.write(json.dumps({
                "id": question.id,
                "refs": [{"doc_name": k.doc_name, "table_id": k.table_id} for k in picked],
            }) + "\n")
            if position % 100 == 0:
                print(f"  {position}/{len(questions)} câu, {time.time() - started:.0f}s")

    print(f"\nxong trong {time.time() - started:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
