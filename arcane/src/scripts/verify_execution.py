#!/usr/bin/env python3
"""Execute every submitted pandas query against its own CSVs and verify agreement.

The organizers recompute each answer from the packaged CSVs at scoring time, so a
submission whose stored `answer` disagrees with its `pandas_query` result fails
both ANSWER_ACCURACY and EXECUTION_ACCURACY for that question. This runs the same
check locally before anything ships:

    python scripts/verify_execution.py --package output/run/package

Exit code is 1 when any query raises or disagrees, so it gates a build.
"""

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True,
                        help="directory holding submission.json and data/tables/")
    parser.add_argument("--tolerance", type=float, default=2e-4,
                        help="relative tolerance matching the 0.02% scoring rule")
    args = parser.parse_args()

    import pandas as pd

    rows = json.loads((args.package / "submission.json").read_text("utf-8"))
    raised, disagreed, agreed, skipped = [], [], 0, 0
    for row in rows:
        query = row.get("pandas_query", "")
        if not query or query.startswith("result = 0 * df"):
            skipped += 1
            continue
        frame = {}
        for item in row.get("evidence", []):
            path = args.package / item["csv_path"]
            if not path.is_file():
                raised.append((row["id"], f"missing {item['csv_path']}"))
                break
            frame[item["variable"]] = pd.read_csv(path, dtype=str)
        else:
            try:
                local: dict = {}
                exec(f"import pandas as pd\n{query}", {"pd": pd, "dfs": frame, **frame}, local)
                result = local["result"]
            except Exception as error:
                raised.append((row["id"], repr(error)[:120]))
                continue
            expected = float(row["answer"])
            actual = float(result)
            denominator = max(1.0, abs(expected))
            if abs(actual - expected) / denominator > args.tolerance:
                disagreed.append((row["id"], expected, actual))
            else:
                agreed += 1

    report = {
        "questions": len(rows),
        "agreed": agreed,
        "disagreed": len(disagreed),
        "raised": len(raised),
        "skipped_no_query": skipped,
    }
    print(json.dumps(report, indent=2))
    for identifier, detail in raised[:10]:
        print(f"  raise id={identifier}: {detail}")
    for identifier, expected, actual in disagreed[:10]:
        print(f"  differ id={identifier}: answer={expected} query={actual}")
    if raised or disagreed:
        sys.exit(1)


if __name__ == "__main__":
    main()
