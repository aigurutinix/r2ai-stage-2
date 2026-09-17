"""Export a durable product batch result into a strict deterministic submission ZIP."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from kingpro.submission.export_batch import export_batch_results  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_result", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--fallback-artifact", default="sub_v297_scope2")
    args = parser.parse_args()
    archive = args.archive or args.out.parent / f"{args.out.name}.zip"
    report = export_batch_results(
        args.batch_result,
        root=ROOT,
        output_dir=args.out,
        archive_path=archive,
        fallback_artifact=args.fallback_artifact,
        expected_count=args.expected_count,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
