"""Build a small all-page statement cube for selected ticker/year pairs.

The production cube intentionally limits discovery to early report pages.  This
read-only recovery helper keeps that cube untouched and filters the catalog
before asking the normal parser to scan every page for a small set of reports.
It is useful when a panel audit finds a required operand missing from the
production cube.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.financial.statement_cube import build_statement_cube


def parse_pair(value: str) -> tuple[str, int]:
    try:
        ticker, year = value.rsplit(":", 1)
        return ticker.upper(), int(year)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("pair must look like TICKER:YEAR") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", action="append", type=parse_pair, required=True)
    parser.add_argument(
        "--catalog", type=Path, default=ROOT / "build" / "catalog.jsonl"
    )
    parser.add_argument("--tables", type=Path, default=ROOT / "build" / "tables")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    wanted = set(args.pair)
    selected: list[str] = []
    report_ids: set[str] = set()
    with args.catalog.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row.get("ticker", "")).upper(), int(row.get("year", 0)))
            if key in wanted:
                selected.append(json.dumps(row, ensure_ascii=False))
                report_ids.add(str(row.get("report_id", "")))

    if not selected:
        raise SystemExit("no catalog rows matched the requested ticker/year pairs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".jsonl", delete=False
    ) as handle:
        filtered_catalog = Path(handle.name)
        handle.write("\n".join(selected) + "\n")

    try:
        cube = build_statement_cube(
            filtered_catalog,
            args.tables,
            scopes=("consolidated", "separate", "aggregated", "unknown"),
            max_page=None,
        )
        cube.write_jsonl(args.output)
    finally:
        filtered_catalog.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "requested_pairs": sorted(f"{ticker}:{year}" for ticker, year in wanted),
                "catalog_rows": len(selected),
                "reports": len(report_ids),
                **cube.stats,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
