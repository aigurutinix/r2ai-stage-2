"""Validate the Reverse-ViFinQA loop on a couple of recipe questions."""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import rev_official  # noqa: E402

rev_official.ensure_vifinqa_importable()


def main() -> None:
    t0 = time.time()
    print("loading corpus...", flush=True)
    cube, docs = rev_official.load_cube()
    print(f"cube tickers: {len(cube.tickers())}  ({time.time()-t0:.0f}s)", flush=True)
    total_cells = sum(
        len(y) for t in cube.tickers() for y in cube.years(t)
    )
    print(f"ticker-years: {total_cells}", flush=True)

    from vifinqa.generation.hard.recipe.grounded.planner import (
        lev05_attempt,
        liq02_attempt,
        eq01_attempt,
    )

    # Question 539-like: peer group of oil industry, LEV_05 recipe, year 2019.
    entities = ("BSR", "PLX", "PVT")
    attempt = lev05_attempt(cube, "oil", entities, "2019")
    print(f"\nLEV_05 attempt: graph={attempt.graph is not None} eligible={attempt.entities}")

    if attempt.graph is not None:
        graph, trace, compiled, formatted_answer = rev_official.finalize(
            attempt.graph, attempt.terminal_metric_key, docs
        )
        print(f"  answer={formatted_answer}")
        print(f"  relevant_tables={compiled.relevant_tables}")
        print("  query:\n" + compiled.pandas_query[:800])

    # LIQ_02 on the real-estate group from question 397/403 style, 2024.
    entities2 = ("DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE")
    attempt2 = liq02_attempt(cube, "realestate", entities2, "2024")
    print(f"\nLIQ_02 attempt: graph={attempt2.graph is not None} eligible={attempt2.entities}")
    if attempt2.graph is not None:
        graph, trace, compiled, formatted_answer = rev_official.finalize(
            attempt2.graph, attempt2.terminal_metric_key, docs
        )
        print(f"  answer={formatted_answer}")
        print(f"  relevant_tables={compiled.relevant_tables}")

    # EQ_01 style on a small group, 2020-2021 window.
    attempt3 = eq01_attempt(cube, "steel", ("HPG", "HSG", "NKG"), "2020", "2021")
    print(f"\nEQ_01 attempt: graph={attempt3.graph is not None} eligible={attempt3.entities}")
    if attempt3.graph is not None:
        graph, trace, compiled, formatted_answer = rev_official.finalize(
            attempt3.graph, attempt3.terminal_metric_key, docs
        )
        print(f"  answer={formatted_answer}")
        print(f"  relevant_tables={compiled.relevant_tables}")


if __name__ == "__main__":
    main()
