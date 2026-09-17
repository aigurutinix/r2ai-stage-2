"""P1 driver: data/financial_statements -> artifacts/tables.parquet"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.corpus.extract import extract_corpus  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    workers = max(1, (os.cpu_count() or 4) - 2)
    started = time.time()
    n = extract_corpus(
        data_root=root / "data" / "financial_statements",
        out_path=root / "artifacts" / "tables.parquet",
        workers=workers,
    )
    print(f"{n} tables in {time.time() - started:.1f}s using {workers} workers")


if __name__ == "__main__":
    main()
