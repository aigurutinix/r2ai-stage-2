"""Deterministic panel answers — no LLM.

Writes one JSONL row per solved question. Re-executes each program against the
packaged panel CSV before caching so a non-reproducing value never ships.

Usage:
  PYTHONPATH=src python scripts/run_panel_det.py
  PYTHONPATH=src python scripts/run_panel_det.py --limit 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.panel_det import load_panel, solve  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--cache", default="artifacts/panel_det.jsonl")
    parser.add_argument("--panel", default="artifacts/metrics.parquet")
    args = parser.parse_args()

    parsed = parse_all(
        ROOT / "data" / "questions" / "questions.jsonl",
        ROOT / "data" / "code_stock.csv",
    )
    if args.limit:
        parsed = parsed[: args.limit]

    panel = load_panel(ROOT / args.panel)
    print(f"panel keys: {len(panel)}  questions: {len(parsed)}")

    shapes: Counter[str] = Counter()
    ok = fail_exec = skipped = 0
    out = ROOT / args.cache
    with out.open("w", encoding="utf-8") as handle:
        for question in parsed:
            answer = solve(question, panel)
            if answer is None:
                skipped += 1
                continue
            outcome = run_query(answer.code, {"df": answer.panel_rows})
            if not outcome.ok:
                fail_exec += 1
                print(f"  exec fail q{question.id} [{answer.shape}]: {outcome.error}")
                continue
            if abs(outcome.value - answer.value) > 0.011:
                fail_exec += 1
                print(
                    f"  value mismatch q{question.id}: "
                    f"pack={answer.value} exec={outcome.value}"
                )
                continue
            row = {
                "id": question.id,
                "ok": True,
                "value": outcome.value,
                "code": answer.code,
                "panel_rows": answer.panel_rows,
                "shape": answer.shape,
                "metrics": answer.metrics,
                "keys": answer.keys,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            shapes[answer.shape] += 1
            ok += 1

    print(f"ok={ok}  exec_fail={fail_exec}  skipped={skipped}  -> {args.cache}")
    for name, count in shapes.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
