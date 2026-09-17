"""Render compact, grep-friendly lines from a source review packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()
    records = json.loads(args.packet.read_text(encoding="utf-8"))["records"]
    for record in records[args.start : args.start + args.limit]:
        sources = " || ".join(
            f"{source.get('source_label', '')}={source.get('raw', '')} [{source.get('table_ref', '')}]"
            for source in record.get("sources", [])
        )
        print(
            f"q{record['id']} :: {record['question']} :: "
            f"ans={record['answer']} :: {sources}"
        )


if __name__ == "__main__":
    main()
