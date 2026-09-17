"""Build a separate, audited missing-year backfill; never overwrite a panel."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road2ai_vifinqa.comparative_panel import fill_comparatives
from road2ai_vifinqa.corpus import Corpus


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Use new output/report paths; existing artifacts are protected")
    original = args.base.read_bytes()
    with Corpus(args.index) as corpus:
        panel, added = fill_comparatives(json.loads(original), corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(panel, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    report = {"schema": 1, "base": str(args.base.resolve()), "base_sha256": hashlib.sha256(original).hexdigest(),
              "output": str(args.output.resolve()), "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
              "added_cells": len(added), "added": added}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"added_cells": len(added), "ticker_years": len({(r['ticker'],r['year']) for r in added})}))


if __name__ == "__main__":
    main()
