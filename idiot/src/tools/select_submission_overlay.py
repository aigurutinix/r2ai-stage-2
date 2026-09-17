"""Select a validated subset of rows and evidence CSVs from a submission build."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_ids(spec: str) -> set[int]:
    values: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            if start > end:
                raise ValueError(f"descending range: {part!r}")
            values.update(range(start, end + 1))
        else:
            values.add(int(part))
    if not values:
        raise ValueError("--ids selected no rows")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-build", type=Path, required=True)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--output-build", type=Path, required=True)
    args = parser.parse_args()

    source = args.source_build.resolve()
    payload = json.loads((source / "submission.json").read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError("source submission.json is not a list")
    by_id: dict[int, dict[str, Any]] = {}
    for row in payload:
        qid = int(row["id"])
        if qid in by_id:
            raise ValueError(f"duplicate source id: {qid}")
        by_id[qid] = row

    selected_ids = parse_ids(args.ids)
    missing = sorted(selected_ids - set(by_id))
    if missing:
        raise ValueError(f"source build is missing ids: {missing}")

    output = args.output_build.resolve()
    if output.exists():
        shutil.rmtree(output)
    (output / "data").mkdir(parents=True)

    selected: list[dict[str, Any]] = []
    copied: set[str] = set()
    for qid in sorted(selected_ids):
        row = by_id[qid]
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"q{qid}: missing evidence")
        selected.append(row)
        for item in evidence:
            rel = str(item["csv_path"]).replace("\\", "/")
            if not rel.startswith("data/") or rel in copied:
                raise ValueError(f"q{qid}: invalid or reused csv_path {rel!r}")
            src = source / Path(rel)
            if not src.is_file():
                raise FileNotFoundError(src)
            dst = output / Path(rel)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.add(rel)

    (output / "submission.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"selected {len(selected)} rows and {len(copied)} CSVs -> {output}")


if __name__ == "__main__":
    main()
