"""Step 1 (local, CPU): parse the whole corpus into the parquet dual store.

  python scripts/01_build_store.py                    # full corpus (~15-30 min)
  python scripts/01_build_store.py --tickers VNM,VJC  # smoke test
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.extraction.build_store import build_store
from vifinqa.utils.io import setup_stdout


def main():
    setup_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs-dir", default=str(config.FS_DIR))
    ap.add_argument("--store-dir", default=str(config.STORE_DIR))
    ap.add_argument("--tickers", default="", help="comma-separated subset, e.g. VNM,VJC")
    ap.add_argument("--max-reports-per-ticker", type=int, default=0)
    ap.add_argument("--no-cells", action="store_true")
    args = ap.parse_args()

    tickers = [t for t in args.tickers.split(",") if t.strip()] or None
    reports = build_store(Path(args.fs_dir), Path(args.store_dir), tickers,
                          args.max_reports_per_ticker, with_cells=not args.no_cells)
    print(f"done: {len(reports)} reports parsed -> {args.store_dir}")


if __name__ == "__main__":
    main()
