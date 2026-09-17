"""Run the offline hard-negative row-linking evaluation."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa.utils.io import write_json
from vifinqa.validation.row_linking_eval import (
    default_hard_negative_cases,
    evaluate_row_linking,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    parser.add_argument("--show-failures", type=int, default=12)
    args = parser.parse_args()

    report = evaluate_row_linking(default_hard_negative_cases())
    overall = report["overall"]
    print(f"row-linking n={overall['n']} top1={overall['top1']:.3f} "
          f"mrr={overall['mrr']:.3f} recall@5={overall['recall5']:.3f}")
    for category, metrics in report["per_category"].items():
        print(f"  {category:24} n={metrics['n']:3d} "
              f"top1={metrics['top1']:.3f} mrr={metrics['mrr']:.3f} "
              f"recall@5={metrics['recall5']:.3f}")
    for failure in report["failures"][:max(0, args.show_failures)]:
        print("  FAIL " + json.dumps(failure, ensure_ascii=False))
    if args.out:
        write_json(args.out, report)
        print(f"[row-linking-audit] report -> {args.out}")


if __name__ == "__main__":
    main()
