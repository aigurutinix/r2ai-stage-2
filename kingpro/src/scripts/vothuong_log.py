"""Repo-local Vô Thượng log: lessons, experiments, reviews and auto-logged runs.

Examples:
  python scripts/vothuong_log.py add "Bài học..." --conf cao --tags retrieval,unit
  python scripts/vothuong_log.py exp --name v196-sign --metric execution --value 0.6877 --result hoa --oracle leaderboard
  python scripts/vothuong_log.py review --question-ids 97 --verdict source_confirmed --summary "Raw source is already percent"
  python scripts/vothuong_log.py run --name forensic-v197 -- python scripts/answer_forensics.py candidate
  python scripts/vothuong_log.py recent 20
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

from kingpro.experiments import ExperimentLogbook  # noqa: E402
from kingpro.experiments.logbook import parse_key_values  # noqa: E402


def _csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _question_ids(value: str) -> list[int]:
    return [int(part) for part in _csv(value)]


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2))


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="KINGPRO Vô Thượng append-only log")
    parser.add_argument("--root", type=Path, default=ROOT / "knowledge" / "vothuong")
    sub = parser.add_subparsers(dest="command", required=True)

    add = sub.add_parser("add", help="append a durable lesson")
    add.add_argument("text")
    add.add_argument("--conf", default="vua", choices=["cao", "vua", "thap"])
    add.add_argument("--tags", default="")
    add.add_argument("--supersedes", default="")
    add.add_argument("--force", action="store_true")

    exp = sub.add_parser("exp", help="append a measured experiment")
    exp.add_argument("--name", required=True)
    exp.add_argument("--metric", required=True)
    exp.add_argument("--value", required=True)
    exp.add_argument("--result", required=True, choices=["thang", "thua", "hoa"])
    exp.add_argument("--oracle", required=True)
    exp.add_argument("--note", default="")
    exp.add_argument("--artifact", action="append", default=[])
    exp.add_argument("--question-ids", default="")
    exp.add_argument("--supersedes", default="")

    review = sub.add_parser("review", help="append a source/hypothesis review")
    review.add_argument("--question-ids", required=True)
    review.add_argument("--verdict", required=True)
    review.add_argument("--summary", required=True)
    review.add_argument("--oracle", default="source")
    review.add_argument("--evidence", action="append", default=[])
    review.add_argument("--detail", action="append", default=[], metavar="KEY=VALUE")
    review.add_argument("--supersedes", default="")

    recent = sub.add_parser("recent", help="show recent structured events")
    recent.add_argument("n", nargs="?", type=int, default=20)
    recent.add_argument("--kind", default="")
    recent.add_argument("--question-id", type=int)

    run = sub.add_parser("run", help="execute a command and auto-log outcome")
    run.add_argument("--name", required=True)
    run.add_argument("--oracle", default="do-luong")
    run.add_argument("--note", default="")
    run.add_argument("command_line", nargs=argparse.REMAINDER)

    args = parser.parse_args()
    logbook = ExperimentLogbook(args.root)
    if args.command == "add":
        _print(
            logbook.add_lesson(
                args.text,
                confidence=args.conf,
                tags=_csv(args.tags),
                supersedes=args.supersedes,
                force=args.force,
            )
        )
        return 0
    if args.command == "exp":
        _print(
            logbook.append_event(
                "experiment",
                oracle=args.oracle,
                name=args.name,
                metric=args.metric,
                value=args.value,
                result=args.result,
                note=args.note,
                artifacts=args.artifact,
                question_ids=_question_ids(args.question_ids),
                supersedes=args.supersedes,
            )
        )
        return 0
    if args.command == "review":
        _print(
            logbook.append_event(
                "review",
                oracle=args.oracle,
                question_ids=_question_ids(args.question_ids),
                verdict=args.verdict,
                summary=args.summary,
                evidence=args.evidence,
                details=parse_key_values(args.detail),
                supersedes=args.supersedes,
                deduplicate=True,
            )
        )
        return 0
    if args.command == "recent":
        _print(logbook.recent(args.n, kind=args.kind, question_id=args.question_id))
        return 0
    command_line = list(args.command_line)
    if command_line and command_line[0] == "--":
        command_line = command_line[1:]
    if not command_line:
        parser.error("run requires a command after --")
    completed = logbook.run_and_log(
        command_line,
        name=args.name,
        oracle=args.oracle,
        cwd=ROOT,
        note=args.note,
    )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
