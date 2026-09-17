"""Compare two financial panels without mutating release artifacts.

The report separates source-coordinate changes from numeric answer changes and
replays every deterministic Hard/Template question against both panels.  It is
intended as the mandatory gate before promoting a newly built panel.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from road2ai_vifinqa.corpus import Corpus, load_questions  # noqa: E402
from road2ai_vifinqa.hard_solver import solve_hard  # noqa: E402
from road2ai_vifinqa.panel import FinancialPanel  # noqa: E402
from road2ai_vifinqa.template_solver import TemplateSolver, _AUDITED_OVERRIDES  # noqa: E402


HARD_IDS = frozenset((*range(362, 427), *range(440, 495), *range(539, 578)))
TEMPLATE_IDS = frozenset(range(578, 1013))


def _same_number(left: object, right: object) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-12, abs_tol=1e-9)
    except (TypeError, ValueError):
        return left == right


def _panel_cells(panel: FinancialPanel) -> dict[tuple[str, int, str], dict[str, object]]:
    result: dict[tuple[str, int, str], dict[str, object]] = {}
    for ticker, years in panel.raw.items():
        for year, metrics in years.items():
            for key, cell in metrics.items():
                result[(ticker, int(year), key)] = cell
    return result


def _source_tuple(item: dict[str, object] | None) -> tuple[object, ...]:
    if item is None:
        return ()
    return (
        item.get("doc_id"),
        item.get("table_id"),
        item.get("row_idx"),
        item.get("col_idx"),
    )


def _solution_sources(solution: object, panel: FinancialPanel, *, hard: bool) -> list[tuple[object, ...]]:
    if not hard:
        return sorted(
            (
                source.ticker,
                int(source.year),
                source.doc_id,
                int(source.table_id),
                int(source.row_idx),
                int(source.col_idx),
            )
            for source in solution.sources
        )
    result: set[tuple[object, ...]] = set()
    slices = getattr(solution, "source_slices", ())
    domains = (
        [(item.tickers, item.years, item.raw_columns) for item in slices]
        if slices
        else [(solution.tickers, solution.years, solution.raw_columns)]
    )
    for tickers, years, columns in domains:
        for ticker in tickers:
            for year in years:
                for column in columns:
                    cell = panel.cell(ticker, int(year), column)
                    if cell is not None:
                        result.add(
                            (
                                ticker,
                                int(year),
                                column,
                                cell.doc_id,
                                cell.table_id,
                                cell.row_idx,
                                cell.col_idx,
                            )
                        )
    return sorted(result)


def benchmark(
    baseline_path: Path,
    candidate_path: Path,
    *,
    routes: frozenset[str] = frozenset({"hard", "template"}),
    qid_min: int | None = None,
    qid_max: int | None = None,
) -> dict[str, object]:
    baseline = FinancialPanel(baseline_path)
    candidate = FinancialPanel(candidate_path)
    left_cells = _panel_cells(baseline)
    right_cells = _panel_cells(candidate)
    keys = sorted(set(left_cells) | set(right_cells))
    cell_changes: list[dict[str, object]] = []
    metric_changes: Counter[str] = Counter()
    for key in keys:
        left = left_cells.get(key)
        right = right_cells.get(key)
        numeric_changed = left is None or right is None or not _same_number(left.get("value"), right.get("value"))
        source_changed = _source_tuple(left) != _source_tuple(right)
        if not (numeric_changed or source_changed):
            continue
        metric_changes[key[2]] += 1
        cell_changes.append(
            {
                "ticker": key[0],
                "year": key[1],
                "metric": key[2],
                "numeric_changed": numeric_changed,
                "source_changed": source_changed,
                "baseline_value": left.get("value") if left else None,
                "candidate_value": right.get("value") if right else None,
                "baseline_source": _source_tuple(left),
                "candidate_source": _source_tuple(right),
                "candidate_margin": right.get("selection_margin") if right else None,
            }
        )

    questions = {int(row["id"]): str(row["question"]) for row in load_questions()}
    answer_changes: list[dict[str, object]] = []
    source_changes: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    replay_started = time.perf_counter()
    with Corpus() as corpus:
        template_left = TemplateSolver(corpus, baseline)
        template_right = TemplateSolver(corpus, candidate)
        selected_ids: set[int] = set()
        if "hard" in routes:
            selected_ids.update(HARD_IDS)
        if "template" in routes:
            # Locked audited recipes load raw cells directly and are therefore
            # invariant to panel changes; avoid hundreds of redundant reads.
            selected_ids.update(TEMPLATE_IDS - set(_AUDITED_OVERRIDES))
        if qid_min is not None:
            selected_ids = {qid for qid in selected_ids if qid >= qid_min}
        if qid_max is not None:
            selected_ids = {qid for qid in selected_ids if qid <= qid_max}
        for position, qid in enumerate(sorted(selected_ids), 1):
            try:
                if qid in HARD_IDS:
                    old = solve_hard(questions[qid], qid, baseline)
                    new = solve_hard(questions[qid], qid, candidate)
                else:
                    old = template_left.solve(questions[qid], question_id=qid)
                    new = template_right.solve(questions[qid], question_id=qid)
                if old is None or new is None:
                    if old is not new:
                        errors.append({"id": qid, "error": "one solver abstained"})
                    continue
                old_sources = _solution_sources(old, baseline, hard=qid in HARD_IDS)
                new_sources = _solution_sources(new, candidate, hard=qid in HARD_IDS)
                if old_sources != new_sources:
                    source_changes.append(
                        {
                            "id": qid,
                            "answer_changed": not _same_number(old.answer, new.answer),
                            "baseline_sources": old_sources,
                            "candidate_sources": new_sources,
                        }
                    )
                if not _same_number(old.answer, new.answer):
                    answer_changes.append(
                        {
                            "id": qid,
                            "question": questions[qid],
                            "baseline": old.answer,
                            "candidate": new.answer,
                            "delta": float(new.answer) - float(old.answer),
                        }
                    )
            except Exception as exc:  # diagnostic must continue after one bad family
                errors.append({"id": qid, "error": f"{type(exc).__name__}: {exc}"})
            if position % 25 == 0:
                print(
                    f"replayed {position}/{len(selected_ids)} questions; "
                    f"answer_changes={len(answer_changes)} errors={len(errors)}",
                    flush=True,
                )

    replay_elapsed = time.perf_counter() - replay_started
    period_mismatches = [
        {"ticker": ticker, "year": year, "metric": metric, "period_year": item.get("period_year")}
        for (ticker, year, metric), item in right_cells.items()
        if item.get("period_year") not in (None, year)
    ]
    low_margin = [
        {"ticker": ticker, "year": year, "metric": metric, "margin": item.get("selection_margin")}
        for (ticker, year, metric), item in right_cells.items()
        if isinstance(item.get("selection_margin"), (int, float))
        and float(item["selection_margin"]) < 1.0
    ]
    return {
        "schema_version": 1,
        "baseline": str(baseline_path.resolve()),
        "candidate": str(candidate_path.resolve()),
        "panel": {
            "baseline_cells": len(left_cells),
            "candidate_cells": len(right_cells),
            "changed_cells": len(cell_changes),
            "numeric_changes": sum(bool(item["numeric_changed"]) for item in cell_changes),
            "source_changes": sum(bool(item["source_changed"]) for item in cell_changes),
            "changes_by_metric": dict(metric_changes.most_common()),
            "period_year_mismatches": period_mismatches,
            "low_margin_selections": low_margin,
            "changes": cell_changes,
        },
        "solver_replay": {
            "routes": sorted(routes),
            "questions": len(selected_ids),
            "elapsed_seconds": round(replay_elapsed, 3),
            "questions_per_second": (
                round(len(selected_ids) / replay_elapsed, 3) if replay_elapsed else None
            ),
            "answer_changes": answer_changes,
            "source_changes": source_changes,
            "errors": errors,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--routes",
        default="hard,template",
        help="comma-separated deterministic routes: hard,template",
    )
    parser.add_argument("--qid-min", type=int)
    parser.add_argument("--qid-max", type=int)
    args = parser.parse_args()
    routes = frozenset(value.strip() for value in args.routes.split(",") if value.strip())
    unknown = routes - {"hard", "template"}
    if unknown:
        parser.error(f"unknown routes: {sorted(unknown)}")
    report = benchmark(
        args.baseline,
        args.candidate,
        routes=routes,
        qid_min=args.qid_min,
        qid_max=args.qid_max,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "changed_cells": report["panel"]["changed_cells"],
                "answer_changes": len(report["solver_replay"]["answer_changes"]),
                "errors": len(report["solver_replay"]["errors"]),
                "report": str(args.report),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
