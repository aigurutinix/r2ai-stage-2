"""Re-run the strict release checks for an existing ViFinQA archive."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .corpus import load_questions
from .submission import validate_submission


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", type=Path, help="run directory containing submission.zip")
    target.add_argument("--zip", dest="zip_path", type=Path, help="submission ZIP to validate")
    parser.add_argument("--report", type=Path, help="optional JSON report path")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    zip_path = (args.zip_path or args.run / "submission.zip").resolve()
    report = validate_submission(zip_path, load_questions())
    payload = asdict(report) | {"ok": report.ok, "zip_path": str(zip_path)}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report.ok else 1)


if __name__ == "__main__":
    main()
