"""Resumable end-to-end solver and submission release CLI.

The public question IDs form four stable solver families.  This command keeps
one independently verifiable checkpoint per question, so an interrupted local
LLM run resumes without repeating completed work.  A ZIP is emitted only after
all selected questions are solved and the strict submission validator replays
every pandas expression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import sys
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

from .corpus import Corpus, load_questions
from .panel import FinancialPanel
from .paths import INDEX_PATH, PANEL_PATH, PROJECT_ROOT, RUNS_ROOT
from .pipeline import (
    solve_easy_submission,
    solve_hard_submission,
    solve_note_submission,
    solve_template_submission,
)
from .submission import (
    SubmissionSolution,
    evaluate_expression,
    validate_submission,
    write_submission,
)
from .template_solver import TemplateSolver


CHECKPOINT_SCHEMA = 2
DIRECT_IDS = frozenset(range(1, 362))
HARD_IDS = frozenset((*range(362, 427), *range(440, 495), *range(539, 578)))
NOTE_IDS = frozenset((*range(427, 440), *range(495, 539)))
TEMPLATE_IDS = frozenset(range(578, 1013))
PUBLIC_IDS = DIRECT_IDS | HARD_IDS | NOTE_IDS | TEMPLATE_IDS
_TABLE_REF = re.compile(r"^[^|]+\|(?:table_)?[1-9]\d*$")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_pickle(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
    temporary.replace(path)


def route_for_id(question_id: int) -> str:
    if question_id in DIRECT_IDS:
        return "direct"
    if question_id in HARD_IDS:
        return "hard"
    if question_id in NOTE_IDS:
        return "note"
    if question_id in TEMPLATE_IDS:
        return "template"
    raise ValueError(f"Question ID {question_id} is outside the public 1--1012 set")


def parse_id_spec(spec: str | None, available: Iterable[int]) -> list[int]:
    available_set = set(available)
    if not spec:
        return sorted(available_set)
    selected: set[int] = set()
    for raw_part in spec.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            start, end = int(left), int(right)
            if end < start:
                raise ValueError(f"Descending ID range is not allowed: {part!r}")
            selected.update(range(start, end + 1))
        else:
            selected.add(int(part))
    unknown = selected - available_set
    if unknown:
        raise ValueError(f"Unknown question IDs: {sorted(unknown)[:20]}")
    if not selected:
        raise ValueError("ID selection is empty")
    return sorted(selected)


class _Resources:
    """Lazily construct the large read-only resources needed by each route."""

    def __init__(self, *, index_path: Path = INDEX_PATH, panel_path: Path = PANEL_PATH) -> None:
        self.index_path = index_path
        self.panel_path = panel_path
        self._corpus: Corpus | None = None
        self._panel: FinancialPanel | None = None
        self._template: TemplateSolver | None = None

    @property
    def corpus(self) -> Corpus:
        if self._corpus is None:
            self._corpus = Corpus(self.index_path)
        return self._corpus

    @property
    def panel(self) -> FinancialPanel:
        if self._panel is None:
            self._panel = FinancialPanel(self.panel_path)
        return self._panel

    @property
    def template(self) -> TemplateSolver:
        if self._template is None:
            self._template = TemplateSolver(self.corpus, self.panel)
        return self._template

    def close(self) -> None:
        if self._corpus is not None:
            self._corpus.close()


def _solve_one(
    question_id: int,
    question: str,
    route: str,
    resources: _Resources,
    run_dir: Path,
    max_attempts: int,
) -> SubmissionSolution:
    if route == "direct":
        return solve_easy_submission(
            question_id,
            question,
            resources.corpus,
            max_attempts=max_attempts,
            log_path=run_dir / "llm" / f"q{question_id:04d}.json",
        )
    if route == "hard":
        return solve_hard_submission(question_id, question, resources.panel)
    if route == "note":
        return solve_note_submission(
            question_id,
            question,
            resources.corpus,
            panel=resources.panel,
            max_attempts=max_attempts,
            log_path=run_dir / "llm" / f"q{question_id:04d}.json",
        )
    if route == "template":
        return solve_template_submission(question_id, question, resources.template)
    raise AssertionError(f"Unhandled route {route!r}")


def _verify_solution(solution: SubmissionSolution, question_id: int, question: str) -> None:
    if not isinstance(solution, SubmissionSolution):
        raise TypeError(f"checkpoint is {type(solution).__name__}, not SubmissionSolution")
    if solution.id != question_id or solution.question != question:
        raise ValueError("checkpoint question identity mismatch")
    if isinstance(solution.answer, bool) or not isinstance(solution.answer, (int, float)):
        raise TypeError("solution answer is not numeric")
    if not math.isfinite(float(solution.answer)):
        raise ValueError("solution answer is non-finite")
    if not solution.evidence:
        raise ValueError("solution has no evidence")
    variables = [item.variable for item in solution.evidence]
    if len(variables) != len(set(variables)):
        raise ValueError("solution contains duplicate evidence variables")
    frames = {item.variable: item.frame for item in solution.evidence}
    replayed = evaluate_expression(solution.pandas_query, frames)
    if abs(float(replayed) - float(solution.answer)) > 1e-9:
        raise ValueError(f"checkpoint replay {replayed!r} != {solution.answer!r}")
    if any(not _TABLE_REF.fullmatch(value) for value in solution.relevant_tables):
        raise ValueError("solution contains an invalid relevant_tables reference")
    docs = set(solution.relevant_docs)
    if any(value.rsplit("|", 1)[0] not in docs for value in solution.relevant_tables):
        raise ValueError("solution table provenance is absent from relevant_docs")


def _checkpoint_path(run_dir: Path, question_id: int) -> Path:
    return run_dir / "cache" / f"q{question_id:04d}.pkl"


def _load_checkpoint(
    run_dir: Path,
    question_id: int,
    question: str,
    route: str,
    resource_signature: str,
) -> SubmissionSolution | None:
    path = _checkpoint_path(run_dir, question_id)
    if not path.exists():
        return None
    # Checkpoints are trusted local artifacts created by this command.  They
    # are never accepted from the submission ZIP or another external source.
    payload = pickle.loads(path.read_bytes())
    if not isinstance(payload, dict) or payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint schema mismatch")
    if payload.get("route") != route:
        raise ValueError("checkpoint route mismatch")
    if payload.get("resource_signature") != resource_signature:
        raise ValueError("checkpoint artifact signature mismatch")
    solution = payload.get("solution")
    _verify_solution(solution, question_id, question)
    if route == "direct" and not str(solution.method).startswith("easy_llm:"):
        # Legacy lexical checkpoints (and transient direct fallbacks) must not
        # suppress the new semantic reranker on a resumed release run.
        raise ValueError(f"legacy/fallback easy checkpoint method: {solution.method}")
    return solution


def _solution_summary(solution: SubmissionSolution, route: str, cache_hit: bool) -> dict[str, object]:
    return {
        "id": solution.id,
        "route": route,
        "method": solution.method,
        "answer": solution.answer,
        "confidence": solution.confidence,
        "relevant_docs": list(solution.relevant_docs),
        "relevant_tables": list(solution.relevant_tables),
        "evidence_variables": [item.variable for item in solution.evidence],
        "cache_hit": cache_hit,
        "completed_at": _now(),
    }


def _save_checkpoint(
    run_dir: Path,
    solution: SubmissionSolution,
    route: str,
    resource_signature: str,
) -> None:
    _verify_solution(solution, solution.id, solution.question)
    _atomic_pickle(
        _checkpoint_path(run_dir, solution.id),
        {
            "schema": CHECKPOINT_SCHEMA,
            "route": route,
            "question": solution.question,
            "created_at": _now(),
            "resource_signature": resource_signature,
            "solution": solution,
        },
    )
    _atomic_json(
        run_dir / "checkpoints" / f"q{solution.id:04d}.json",
        _solution_summary(solution, route, cache_hit=False),
    )


def _model_manifest() -> dict[str, object]:
    raw_path = os.environ.get("VIFINQA_MODEL", "")
    path = Path(raw_path).expanduser() if raw_path else None
    result: dict[str, object] = {
        "source": os.environ.get("VIFINQA_MODEL_SOURCE", ""),
        "path": str(path.resolve()) if path and path.exists() else raw_path,
    }
    if path and path.exists():
        stat = path.stat()
        result.update(size=stat.st_size, modified_ns=stat.st_mtime_ns)
    return result


def _progress_payload(
    *,
    selected: list[int],
    solutions: dict[int, SubmissionSolution],
    failures: dict[int, str],
    cache_hits: int,
) -> dict[str, object]:
    return {
        "updated_at": _now(),
        "selected": len(selected),
        "solved": len(solutions),
        "failed": len(failures),
        "remaining": len(selected) - len(solutions) - len(failures),
        "cache_hits": cache_hits,
        "failed_ids": sorted(failures),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, default=1, help="run number used in runs/iteration_N")
    parser.add_argument("--run-dir", type=Path, help="override the run directory")
    parser.add_argument("--ids", help="optional comma/range selection, e.g. 1-10,362,578")
    parser.add_argument("--index", type=Path, default=INDEX_PATH, help="read-only table index")
    parser.add_argument("--panel", type=Path, default=PANEL_PATH, help="financial panel JSON")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="local-LLM repairs for easy and curated-note questions",
    )
    parser.add_argument("--no-resume", action="store_true", help="ignore existing per-question checkpoints")
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first solver failure")
    parser.add_argument("--publish", action="store_true", help="copy a validated full-set ZIP to --publish-path")
    parser.add_argument(
        "--publish-path",
        type=Path,
        default=PROJECT_ROOT / "submission.zip",
        help="final archive path used with --publish",
    )
    return parser


def _artifact_signature(*paths: Path) -> str:
    """Cheap cache key which changes whenever a generated artifact changes."""

    digest = hashlib.sha256()
    for path in paths:
        resolved = path.resolve()
        stat = resolved.stat()
        digest.update(str(resolved).encode("utf-8"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
    return digest.hexdigest()


def run(args: argparse.Namespace) -> int:
    if args.iteration < 1:
        raise ValueError("--iteration must be positive")
    if args.max_attempts < 1:
        raise ValueError("--max-attempts must be positive")

    canonical_questions = load_questions()
    question_by_id = {int(row["id"]): str(row["question"]) for row in canonical_questions}
    if set(question_by_id) != PUBLIC_IDS:
        missing = sorted(PUBLIC_IDS - set(question_by_id))
        extra = sorted(set(question_by_id) - PUBLIC_IDS)
        raise RuntimeError(f"Unexpected public question set: missing={missing[:10]} extra={extra[:10]}")
    selected = parse_id_spec(args.ids, question_by_id)
    run_dir = (args.run_dir or RUNS_ROOT / f"iteration_{args.iteration}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    solutions: dict[int, SubmissionSolution] = {}
    failures: dict[int, str] = {}
    cache_hits = 0
    index_path = args.index.resolve()
    panel_path = args.panel.resolve()
    resource_signature = _artifact_signature(index_path, panel_path)
    resources = _Resources(index_path=index_path, panel_path=panel_path)
    started_at = _now()
    print(f"ViFinQA iteration {args.iteration}: {len(selected)} questions -> {run_dir}", flush=True)
    try:
        for position, question_id in enumerate(selected, 1):
            question = question_by_id[question_id]
            route = route_for_id(question_id)
            solution: SubmissionSolution | None = None
            cache_hit = False
            if not args.no_resume:
                try:
                    solution = _load_checkpoint(
                        run_dir,
                        question_id,
                        question,
                        route,
                        resource_signature,
                    )
                    cache_hit = solution is not None
                except Exception as exc:
                    # A stale/corrupt checkpoint is never fatal; record why it
                    # was rejected and regenerate that one question.
                    _atomic_json(
                        run_dir / "checkpoint_rejections" / f"q{question_id:04d}.json",
                        {"id": question_id, "rejected_at": _now(), "error": f"{type(exc).__name__}: {exc}"},
                    )
            try:
                if solution is None:
                    solution = _solve_one(
                        question_id,
                        question,
                        route,
                        resources,
                        run_dir,
                        args.max_attempts,
                    )
                    _save_checkpoint(run_dir, solution, route, resource_signature)
                else:
                    cache_hits += 1
                solutions[question_id] = solution
                error_path = run_dir / "errors" / f"q{question_id:04d}.json"
                if error_path.exists():
                    error_path.unlink()
                marker = "cache" if cache_hit else solution.method
                print(f"[{position:04d}/{len(selected):04d}] q{question_id:04d} {route}: {marker}", flush=True)
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                failures[question_id] = message
                _atomic_json(
                    run_dir / "errors" / f"q{question_id:04d}.json",
                    {
                        "id": question_id,
                        "route": route,
                        "question": question,
                        "failed_at": _now(),
                        "error": message,
                        "traceback": traceback.format_exc(),
                    },
                )
                print(f"[{position:04d}/{len(selected):04d}] q{question_id:04d} {route}: FAILED {message}", flush=True)
                if args.fail_fast:
                    break
            _atomic_json(
                run_dir / "progress.json",
                _progress_payload(
                    selected=selected,
                    solutions=solutions,
                    failures=failures,
                    cache_hits=cache_hits,
                ),
            )
    finally:
        resources.close()

    route_counts = Counter(route_for_id(value) for value in solutions)
    method_counts = Counter(solution.method.split(":", 1)[0] for solution in solutions.values())
    base_manifest: dict[str, object] = {
        "schema": 1,
        "iteration": args.iteration,
        "started_at": started_at,
        "finished_at": _now(),
        "run_dir": str(run_dir),
        "selected_ids": selected,
        "full_public_set": set(selected) == set(question_by_id),
        "solved": len(solutions),
        "failed": len(failures),
        "failed_ids": sorted(failures),
        "failures": {str(key): value for key, value in sorted(failures.items())},
        "cache_hits": cache_hits,
        "route_counts": dict(sorted(route_counts.items())),
        "method_counts": dict(sorted(method_counts.items())),
        "confidence": {
            "minimum": min((value.confidence for value in solutions.values()), default=0.0),
            "mean": (
                sum(value.confidence for value in solutions.values()) / len(solutions)
                if solutions
                else 0.0
            ),
        },
        "model": _model_manifest(),
        "artifacts": {
            "index": str(index_path),
            "panel": str(panel_path),
            "signature": resource_signature,
        },
    }

    if failures or len(solutions) != len(selected):
        _atomic_json(run_dir / "manifest.json", base_manifest)
        print(
            f"Run incomplete: solved={len(solutions)} failed={len(failures)}. "
            "Fix failures and rerun the same command to resume.",
            file=sys.stderr,
            flush=True,
        )
        return 1

    ordered = [solutions[question_id] for question_id in selected]
    zip_path = run_dir / "submission.zip"
    build_stats = write_submission(ordered, run_dir / "build", zip_path)
    selected_canonical = [row for row in canonical_questions if int(row["id"]) in set(selected)]
    validation = validate_submission(zip_path, selected_canonical)
    validation_payload = asdict(validation) | {"ok": validation.ok}
    _atomic_json(run_dir / "validation.json", validation_payload)
    base_manifest.update(submission=build_stats, validation=validation_payload)
    _atomic_json(run_dir / "manifest.json", base_manifest)
    if not validation.ok:
        print(f"Submission validation failed with {len(validation.errors)} errors", file=sys.stderr, flush=True)
        return 2

    if args.publish:
        if set(selected) != set(question_by_id):
            print("Refusing --publish for a partial ID selection", file=sys.stderr, flush=True)
            return 3
        publish_path = args.publish_path.resolve()
        publish_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(zip_path, publish_path)
        base_manifest["published_to"] = str(publish_path)
        _atomic_json(run_dir / "manifest.json", base_manifest)
        print(f"Published {publish_path}", flush=True)

    print(
        f"Validated {validation.replayed}/{validation.rows} answers; "
        f"ZIP sha256={validation.sha256}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        code = run(args)
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
