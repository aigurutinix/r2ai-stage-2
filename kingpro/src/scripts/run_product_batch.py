"""Run KINGPRO ProductService over JSON/JSONL with durable checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from _env import load_local_env  # noqa: E402

load_local_env(ROOT)

from kingpro.operations import BatchRunConfig, BatchRunner, load_fallback_records, read_questions  # noqa: E402
from kingpro.product import ProductService  # noqa: E402


def _progress(payload: dict) -> None:
    last = payload["last"]
    print(
        f"[{payload['completed']}/{payload['total']}] "
        f"id={last['id']} status={last['status']} elapsed={last['elapsed_ms']}ms",
        flush=True,
    )


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=ROOT / "data" / "questions" / "questions.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "product_batch")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--run-id")
    parser.add_argument("--resume", type=Path, help="Existing run directory containing manifest.json and running.json")
    parser.add_argument(
        "--fallback-submission",
        type=Path,
        default=ROOT / "sub_v297_scope2" / "submission.json",
        help="Exact-question fallback used only after an item error",
    )
    parser.add_argument("--no-fallback", action="store_true")
    parser.add_argument("--fallback-on-refusal", action="store_true")
    args = parser.parse_args()

    questions = read_questions(args.input, expected_count=args.expected_count)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be at least 1")
        questions = questions[: args.limit]

    service = ProductService(root=ROOT)
    fallback_path = None if args.no_fallback else args.fallback_submission
    fallback = load_fallback_records(fallback_path)
    fingerprints = [
        args.input,
        ROOT / "sub_v297_scope2" / "submission.json",
        ROOT / "build" / "catalog.jsonl",
        ROOT / "build" / "bm25" / "params.index.json",
        ROOT / "build" / "bm25" / "vocab.index.json",
        ROOT / "src" / "kingpro" / "product" / "service.py",
        ROOT / "src" / "kingpro" / "product" / "deterministic_compiler.py",
    ]
    config = BatchRunConfig(
        output_root=args.output_root,
        max_workers=args.workers,
        expected_count=len(questions),
        fallback_on_refusal=args.fallback_on_refusal,
        run_id=args.run_id,
        resume_run_dir=args.resume,
    )
    runner = BatchRunner(
        service.ask,
        root=ROOT,
        input_path=args.input,
        config=config,
        fallback_records=fallback,
        manifest_extra={
            "service_health": service.health(),
            "selected_submission": "v297",
            "fallback_submission": str(fallback_path.resolve()) if fallback_path else None,
            "limit": args.limit,
        },
        fingerprint_paths=fingerprints,
        progress_fn=_progress,
    )
    summary = runner.run(questions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["status"] in {"success", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
