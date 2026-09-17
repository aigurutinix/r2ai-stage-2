"""Build one evidence-first review queue from all current audit artifacts.

Example:
    python scripts/answer_forensics.py \
      sub_top123_candidate_v196_effective_tax_sign \
      --audit-glob "build/v196_*.json" \
      --ledger knowledge/answer_forensics_ledger.json \
      --out build/v196_answer_forensics.json \
      --markdown docs/ANSWER_FORENSICS_V196.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kingpro.forensics import AnswerForensicsEngine  # noqa: E402
from kingpro.forensics.engine import render_markdown  # noqa: E402
from kingpro.experiments import ExperimentLogbook  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--audit-glob", default="build/v196_*.json")
    parser.add_argument("--ledger", type=Path, default=ROOT / "knowledge" / "answer_forensics_ledger.json")
    parser.add_argument(
        "--review-log",
        type=Path,
        default=ROOT / "knowledge" / "vothuong" / "experiments.jsonl",
        help="trusted Vô Thượng review events to exclude from repeat triage",
    )
    parser.add_argument("--history-root", type=Path, default=ROOT)
    parser.add_argument("--history-min-version", type=int, default=161)
    parser.add_argument("--out", type=Path, default=ROOT / "build" / "answer_forensics.json")
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--top", type=int, default=100)
    parser.add_argument("--no-log", action="store_true", help="do not append this run to the Vô Thượng log")
    args = parser.parse_args()

    candidate = args.candidate if args.candidate.is_absolute() else ROOT / args.candidate
    output_path = args.out if args.out.is_absolute() else ROOT / args.out
    output_resolved = output_path.resolve()
    audit_paths = [
        path
        for path in sorted(ROOT.glob(args.audit_glob))
        if path.resolve() != output_resolved and "answer_forensics" not in path.stem.casefold()
    ]
    engine = AnswerForensicsEngine(
        candidate=candidate,
        audit_paths=audit_paths,
        ledger_path=args.ledger,
        history_root=args.history_root,
        history_min_version=args.history_min_version,
        review_log_path=args.review_log,
    )
    report = engine.run(top=args.top)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report), encoding="utf-8")
    log_record = None
    if not args.no_log:
        log_record = ExperimentLogbook(ROOT / "knowledge" / "vothuong").append_event(
            "run",
            oracle="heuristic",
            name="answer-forensics",
            result="thang",
            metric="prioritized_count",
            value=report["counts"].get("prioritized_unreviewed", len(report.get("prioritized", []))),
            candidate=str(candidate.relative_to(ROOT)) if candidate.is_relative_to(ROOT) else str(candidate),
            audit_glob=args.audit_glob,
            counts=report["counts"],
            top_ids=report["top_ids"][:20],
            artifacts=[str(output_path), str(args.markdown) if args.markdown else ""],
            note="Heuristic triage only; every proposed answer change still requires original-source verification.",
        )
    print(
        json.dumps(
            {
                "counts": report["counts"],
                "top_ids": report["top_ids"][:20],
                "output": str(output_path),
                "markdown": str(args.markdown) if args.markdown else None,
                "log_event": log_record.get("id") if log_record else None,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
