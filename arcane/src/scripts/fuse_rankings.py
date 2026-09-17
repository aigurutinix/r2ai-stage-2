#!/usr/bin/env python3
"""Fuse complete table rankings with equal reciprocal-rank contributions."""

import argparse
import json
from pathlib import Path

from vifinqa.fusion import fuse_orders
from vifinqa.jsonl import load_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranking", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--ids-from", type=Path,
        help="limit every ranking to question IDs from this JSONL evaluation set",
    )
    args = parser.parse_args()

    sources = [json.loads(path.read_text(encoding="utf-8")) for path in args.ranking]
    identifiers = (
        {str(record["id"]) for record in load_jsonl(args.ids_from)}
        if args.ids_from else set(sources[0])
    )
    if any(not identifiers <= set(source) for source in sources):
        raise ValueError("every ranking must contain all requested question IDs")
    fused = {
        identifier: fuse_orders([source[identifier] for source in sources])
        for identifier in sorted(identifiers, key=int)
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(fused, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        "questions": len(fused),
        "inputs": [str(path) for path in args.ranking],
        "output": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()
