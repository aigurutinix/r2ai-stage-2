"""Inventory every generation cache: coverage of the 1,012 and executable rate."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    rows = []
    for path in sorted((ROOT / "artifacts").glob("*.jsonl")):
        ids, ok, values = set(), 0, 0
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "id" not in rec or "code" not in rec:
                    ids = set()
                    break
                ids.add(rec["id"])
                if rec.get("ok"):
                    ok += 1
                    if rec.get("value") is not None:
                        values += 1
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if ids:
            rows.append((path.name, len(ids), ok, values, path.stat().st_mtime))

    rows.sort(key=lambda r: -r[4])
    print(f"{'cache':<34}{'ids':>6}{'ok':>7}{'usable':>8}  {'rate':>6}")
    for name, n, ok, values, _ in rows:
        rate = f"{values / n:.0%}" if n else "-"
        print(f"{name:<34}{n:>6}{ok:>7}{values:>8}  {rate:>6}")


if __name__ == "__main__":
    main()
