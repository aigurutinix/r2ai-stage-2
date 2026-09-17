#!/usr/bin/env python3
"""Merge the table-to-report maps the dense, graph, and note lanes each emit.

`run.py --extra-tables` takes one map, but three independent steps produce one
each. Merging them by hand is where a silent error lives: if two lanes disagree
about which report a table came from, the last file read would win and `run.py`
would materialize the table out of the wrong filing, which the validator cannot
catch because the row it writes is internally consistent.

So a conflict is an error here, not a preference. Identical duplicates are
expected — the same table is often reached by more than one lane — and pass.
"""

import argparse
import json
from pathlib import Path


def load_map(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"{path}: expected an object of table_id -> report_id")
    for table_id, report_id in value.items():
        if not isinstance(report_id, str) or not report_id:
            raise SystemExit(f"{path}: {table_id} has no report id")
    return value


def merge(sources: list[tuple[Path, dict[str, str]]]) -> tuple[dict[str, str], list[str]]:
    merged: dict[str, str] = {}
    origin: dict[str, Path] = {}
    conflicts = []
    for path, mapping in sources:
        for table_id, report_id in mapping.items():
            held = merged.get(table_id)
            if held is None:
                merged[table_id] = report_id
                origin[table_id] = path
                continue
            if held != report_id:
                conflicts.append(
                    f"{table_id}: {origin[table_id]} says {held}, {path} says {report_id}"
                )
    return merged, conflicts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", default=[], required=True,
                        help="a table_id -> report_id map; repeat once per lane")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = []
    for path in args.input:
        if not path.exists():
            raise SystemExit(f"missing sidecar: {path}")
        sources.append((path, load_map(path)))

    merged, conflicts = merge(sources)
    if conflicts:
        print(f"conflicting report ids for {len(conflicts)} tables:")
        for line in conflicts[:20]:
            print(f"  {line}")
        if len(conflicts) > 20:
            print(f"  ... and {len(conflicts) - 20} more")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(args.output)

    print(json.dumps({
        "inputs": {str(path): len(mapping) for path, mapping in sources},
        "tables": len(merged),
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
