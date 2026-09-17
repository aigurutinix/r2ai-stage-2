#!/usr/bin/env python3
"""Build an explicit structural BM25 sidecar without touching canonical BM25."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import bm25s


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.retrieval.bm25_index import tokenize  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    catalog = args.catalog.resolve()
    output = args.out.resolve()
    canonical = (ROOT / "build/bm25").resolve()
    if output == canonical or canonical in output.parents:
        raise ValueError("refusing to write structural sidecar inside canonical build/bm25")
    rows = [json.loads(line) for line in catalog.open(encoding="utf-8")]
    retriever = bm25s.BM25()
    retriever.index([tokenize(row["search_text"]) for row in rows])
    output.mkdir(parents=True, exist_ok=True)
    retriever.save(str(output))
    with (output / "meta.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        key: row[key]
                        for key in ("table_ref", "ticker", "year", "scope")
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
    print(json.dumps({"catalog": str(catalog), "out": str(output), "rows": len(rows)}))


if __name__ == "__main__":
    main()

