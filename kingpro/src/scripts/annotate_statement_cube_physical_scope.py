"""Atomically annotate an existing cube with physical report-scope authority.

This is the fast migration path for the already parsed 98k-cell cube. Future
full builds write the same ``physical_scope`` field directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.financial.report_scope import physical_scope  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cube", type=Path, default=ROOT / "build" / "statement_cube.jsonl")
    parser.add_argument("--catalog", type=Path, default=ROOT / "build" / "catalog.jsonl")
    args = parser.parse_args()
    cube_path = args.cube.resolve()
    catalog_path = args.catalog.resolve()
    for path in (cube_path, catalog_path):
        path.relative_to(ROOT.resolve())
    with cube_path.open(encoding="utf-8") as handle:
        required_refs = {
            json.loads(line)["table_ref"] for line in handle if line.strip()
        }
    catalog = {
        entry["table_ref"]: entry
        for entry in (
            json.loads(line)
            for line in catalog_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if entry["table_ref"] in required_refs
    }
    scopes: dict[str, tuple[str, dict | None]] = {}
    for table_ref, entry in catalog.items():
        scopes[table_ref] = physical_scope(entry, ROOT)
    temporary = cube_path.with_suffix(cube_path.suffix + ".physical.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    cells = overrides = evidence = 0
    try:
        with cube_path.open(encoding="utf-8") as source, temporary.open(
            "w", encoding="utf-8", newline="\n"
        ) as target:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                resolved, proof = scopes.get(
                    row["table_ref"], (row.get("scope", "unknown"), None)
                )
                row["physical_scope"] = resolved
                cells += 1
                if proof is not None:
                    evidence += 1
                    if proof["overrides_container"]:
                        overrides += 1
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(cube_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    print(
        json.dumps(
            {
                "cube": str(cube_path),
                "cells": cells,
                "cells_with_physical_evidence": evidence,
                "cells_overriding_container": overrides,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
