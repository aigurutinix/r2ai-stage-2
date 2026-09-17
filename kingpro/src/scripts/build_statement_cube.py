"""Build the auditable local financial-statement cube."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.financial.statement_cube import build_statement_cube


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", default=str(ROOT / "build" / "catalog.jsonl"))
    parser.add_argument("--tables", default=str(ROOT / "build" / "tables"))
    parser.add_argument("--output", default=str(ROOT / "build" / "statement_cube.jsonl"))
    parser.add_argument(
        "--local-units-root",
        type=Path,
        help=(
            "Optional extracted-report root. Explicit markers immediately before each "
            "table override magnitude heuristics; omit to preserve legacy behavior."
        ),
    )
    parser.add_argument(
        "--max-page",
        type=int,
        default=30,
        help="Ignore late note-disclosure tables (default: 30; use 0 for no limit).",
    )
    parser.add_argument(
        "--scope",
        action="append",
        choices=("consolidated", "separate", "aggregated", "unknown"),
        help="Scope(s) to include. Repeat for multiple scopes; defaults to all.",
    )
    args = parser.parse_args()
    scopes = tuple(args.scope) if args.scope else ("consolidated", "separate", "aggregated", "unknown")
    cube = build_statement_cube(
        args.catalog,
        args.tables,
        scopes=scopes,
        max_page=args.max_page or None,
        local_units_root=args.local_units_root,
    )
    cube.write_jsonl(args.output)
    print(json.dumps({"output": args.output, **cube.stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
