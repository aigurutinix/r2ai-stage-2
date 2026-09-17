"""List model-audit approximate answers that materially disagree with a submission."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import zipfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    args = parser.parse_args()

    with zipfile.ZipFile(args.submission) as archive:
        json_name = next(name for name in archive.namelist() if name.endswith(".json"))
        baseline = {
            int(row["id"]): float(row["answer"])
            for row in json.loads(archive.read(json_name))
        }

    paths = glob.glob("runs/reasoning_selector/*_9b.json")
    paths += glob.glob("runs/qwen35_lowconf/*.json")
    pattern = re.compile(
        r'"approximate_answer"\s*:\s*"?(-?[0-9][0-9.eE+-]*)'
    )
    for path in paths:
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
        if isinstance(rows, dict):
            rows = list(rows.values())
        disagreements: list[tuple[int, float, float]] = []
        for row in rows:
            if not isinstance(row, dict) or "id" not in row:
                continue
            qid = int(row["id"])
            match = pattern.search(str(row.get("content", "")))
            if not match or qid not in baseline:
                continue
            proposed = float(match.group(1))
            current = baseline[qid]
            if not math.isclose(proposed, current, rel_tol=0.01, abs_tol=0.01):
                disagreements.append((qid, current, proposed))
        if disagreements:
            print(f"\n{Path(path).name} ({len(disagreements)})")
            for item in disagreements:
                print(item)


if __name__ == "__main__":
    main()
