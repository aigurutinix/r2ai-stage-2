"""Every table that any question could need, as one text per table, ready to embed.

The organisers measured their own retrieval: BM25 47.41% recall@10 against 80.80% for a
dense embedder with a reranker, and 34.7% of their end-to-end failures are the retriever
missing the table. Retrieval by word overlap — which is what this pipeline has — sits in
the BM25 band, so this is the largest unclaimed lever left.

Only the documents the questions can reach are included. Company, year and scope are
resolved from the question text and that resolution is reliable, so embedding all 146K
tables of the corpus would pay for candidates no question can use. Each table becomes one
short document: the heading printed above it in the report, then its row labels, then its
column headers — the three things a question actually names. Cell values are left out;
a question asks for a figure by its label, never by its digits.

Written before renting anything, because the paid window is minutes long and every
avoidable minute inside it is money.

Usage:
  python scripts/fresh/build_table_corpus.py --out artifacts/fresh/table_corpus.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_answers import YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
CONTEXT_CHARS = 300
MAX_ROWS = 40
TEXT_CHARS = 1200


def contexts_of(text_path: Path) -> dict[int, str]:
    try:
        body = text_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out, previous_end = {}, 0
    for match in ANCHOR_RE.finditer(body):
        lead = re.sub(r"\s+", " ", body[previous_end:match.start()].strip())
        out[int(match.group(1))] = lead[-CONTEXT_CHARS:]
        previous_end = match.end()
    return out


def table_text(context: str, grid: list[list[str]]) -> str:
    """Heading, column headers, row labels — what a question names, and nothing else."""

    pieces = []
    if context:
        pieces.append(context)
    if grid:
        header = " | ".join(str(c).strip() for c in grid[0] if str(c).strip())
        if header:
            pieces.append(f"Cột: {header}")
    labels = []
    for row in grid[1:MAX_ROWS + 1]:
        for cell in row[:2]:
            text = str(cell).strip()
            # A label is text; a figure is not. Digits alone say nothing about topic.
            if text and not re.fullmatch(r"[\d.,()%\-\s]+", text):
                labels.append(text)
                break
    if labels:
        pieces.append("Dòng: " + " ; ".join(labels))
    return " \n".join(pieces)[:TEXT_CHARS]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/fresh/table_corpus.jsonl")
    parser.add_argument("--limit-docs", type=int, default=0)
    args = parser.parse_args()

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]

    # Every (ticker, year) any question mentions — the metadata filter the organisers
    # apply before retrieval, and the reason this corpus is 135K tables and not 146K.
    wanted: set[tuple[str, str]] = set()
    for question in questions:
        text = question["question"]
        years = sorted({y for y in YEAR_RE.findall(text)})
        for ticker in resolver.resolve(text):
            for year in years:
                wanted.add((ticker, year))
                # "Cuối năm N" is also printed as the prior column of the N+1 report.
                wanted.add((ticker, str(int(year) + 1)))

    corpus = ROOT / "data" / "official_corpus"
    documents = []
    for ticker, year in sorted(wanted):
        base = corpus / ticker / year
        if not base.is_dir():
            continue
        for doc_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            documents.append((ticker, year, doc_dir))
    if args.limit_docs:
        documents = documents[:args.limit_docs]
    print(f"{len(wanted)} cap ma-nam, {len(documents)} tai lieu", flush=True)

    written = 0
    started = time.time()
    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for index, (ticker, year, doc_dir) in enumerate(documents, start=1):
            tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
            if not tables_dir.is_dir():
                continue
            anchors = contexts_of(doc_dir / f"{doc_dir.name}_extracted.txt")
            for path in sorted(tables_dir.glob("table_*.csv"),
                               key=lambda p: int(p.stem.split("_")[-1])):
                table_id = int(path.stem.split("_")[-1])
                try:
                    with path.open(encoding="utf-8-sig", newline="") as file:
                        grid = [row for _, row in zip(range(MAX_ROWS + 1),
                                                      csv_mod.reader(file))]
                except OSError:
                    continue
                text = table_text(anchors.get(table_id, ""), grid)
                if not text.strip():
                    continue
                handle.write(json.dumps({
                    "ref": f"{doc_dir.name}|table_{table_id}",
                    "doc": doc_dir.name, "ticker": ticker, "year": year,
                    "table_id": table_id, "text": text,
                }, ensure_ascii=False) + "\n")
                written += 1
            if index % 100 == 0:
                print(f"  {index}/{len(documents)} tai lieu, {written} bang, "
                      f"{time.time() - started:.0f}s", flush=True)

    print(f"\n{written} bang -> {ROOT / args.out}")


if __name__ == "__main__":
    main()
