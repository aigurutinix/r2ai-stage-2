"""Generate the deterministic formula-specific offline evaluation suite.

Never upload this synthetic suite to the official leaderboard.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.utils.io import setup_stdout
from vifinqa.validation.gen_formula_eval import generate


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--code-stock", default=str(config.CODE_STOCK_CSV))
    parser.add_argument("--out-dir", default=str(config.ART_DIR / "formula_eval"))
    parser.add_argument("--per-class", type=int, default=24)
    parser.add_argument("--max-tickers", type=int, default=0,
                        help="limit ticker scan for a quick smoke run")
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    generate(
        Path(args.store_dir), Path(args.code_stock), Path(args.out_dir),
        per_class=args.per_class, seed=args.seed,
        max_tickers=args.max_tickers,
    )


if __name__ == "__main__":
    main()
