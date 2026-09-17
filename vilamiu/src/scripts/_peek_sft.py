"""Print one SFT pair so the shape of the target is visible, not inferred."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    lines = (ROOT / "artifacts" / "sft_easy.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    for message in json.loads(lines[idx])["messages"]:
        body = message["content"]
        print(f"===== {message['role']} ({len(body):,} chars) =====")
        print(body if len(body) <= 900 else f"{body[:700]}\n  ... snip ...\n{body[-200:]}")


if __name__ == "__main__":
    main()
