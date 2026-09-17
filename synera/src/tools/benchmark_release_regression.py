"""Compare deterministic solver answers with a known-good submission ZIP."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from road2ai_vifinqa.corpus import Corpus, load_questions  # noqa: E402
from road2ai_vifinqa.hard_solver import solve_hard  # noqa: E402
from road2ai_vifinqa.hard_note_solver import solve_note  # noqa: E402
from road2ai_vifinqa.panel import FinancialPanel  # noqa: E402
from road2ai_vifinqa.template_solver import TemplateSolver  # noqa: E402


HARD_IDS = frozenset((*range(362, 427), *range(440, 495), *range(539, 578)))
TEMPLATE_IDS = frozenset(range(578, 1013))
NOTE_IDS = frozenset((*range(427, 440), *range(495, 539)))


def _reference_answers(path: Path) -> dict[int, float]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if "/" not in name and name.endswith(".json")]
        if len(members) != 1:
            raise ValueError(f"expected one root JSON in {path}, found {members}")
        rows = json.loads(archive.read(members[0]).decode("utf-8"))
    return {int(row["id"]): float(row["answer"]) for row in rows}


def _same(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-9)


def benchmark(
    reference_zip: Path,
    index_path: Path,
    panel_path: Path,
    *,
    qid_min: int | None = None,
    qid_max: int | None = None,
    include_notes: bool = False,
    signed_temporal_changes: bool = False,
    ordered_comparisons: bool = False,
    temporal_contrasts: bool = False,
    literal_ratio_contracts: bool = False,
) -> dict[str, object]:
    reference = _reference_answers(reference_zip)
    questions = {int(row["id"]): str(row["question"]) for row in load_questions()}
    selected = sorted((HARD_IDS | TEMPLATE_IDS | (NOTE_IDS if include_notes else frozenset())) & reference.keys())
    if qid_min is not None:
        selected = [qid for qid in selected if qid >= qid_min]
    if qid_max is not None:
        selected = [qid for qid in selected if qid <= qid_max]
    changes: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    started = time.perf_counter()
    with Corpus(index_path) as corpus:
        panel = FinancialPanel(panel_path)
        template = TemplateSolver(corpus, panel, signed_temporal_changes=signed_temporal_changes,
                                  ordered_comparisons=ordered_comparisons,
                                  temporal_contrasts=temporal_contrasts,
                                  literal_ratio_contracts=literal_ratio_contracts)
        for position, qid in enumerate(selected, 1):
            try:
                solution = (
                    solve_hard(questions[qid], qid, panel)
                    if qid in HARD_IDS
                    else solve_note(questions[qid], qid, corpus, panel=panel)
                    if qid in NOTE_IDS
                    else template.solve(questions[qid], question_id=qid)
                )
                if solution is None:
                    errors.append({"id": qid, "error": "solver abstained"})
                elif not _same(float(solution.answer), reference[qid]):
                    changes.append(
                        {
                            "id": qid,
                            "question": questions[qid],
                            "reference": reference[qid],
                            "candidate": float(solution.answer),
                        }
                    )
            except Exception as exc:
                errors.append({"id": qid, "error": f"{type(exc).__name__}: {exc}"})
            if position % 25 == 0:
                print(
                    f"checked {position}/{len(selected)}; changes={len(changes)} errors={len(errors)}",
                    flush=True,
                )
    elapsed = time.perf_counter() - started
    return {
        "schema_version": 1,
        "reference_zip": str(reference_zip.resolve()),
        "index": str(index_path.resolve()),
        "panel": str(panel_path.resolve()),
        "signed_temporal_changes": signed_temporal_changes,
        "ordered_comparisons": ordered_comparisons,
        "temporal_contrasts": temporal_contrasts,
        "literal_ratio_contracts": literal_ratio_contracts,
        "questions": len(selected),
        "elapsed_seconds": round(elapsed, 3),
        "questions_per_second": round(len(selected) / elapsed, 3) if elapsed else None,
        "answer_changes": changes,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--qid-min", type=int)
    parser.add_argument("--qid-max", type=int)
    parser.add_argument("--include-notes", action="store_true", help="Also replay the 57 deterministic note/scenario recipes")
    parser.add_argument("--signed-temporal-changes", action="store_true")
    parser.add_argument("--ordered-comparisons", action="store_true")
    parser.add_argument("--temporal-contrasts", action="store_true")
    parser.add_argument("--literal-ratio-contracts", action="store_true")
    args = parser.parse_args()
    result = benchmark(
        args.reference,
        args.index,
        args.panel,
        qid_min=args.qid_min,
        qid_max=args.qid_max,
        include_notes=args.include_notes,
        signed_temporal_changes=args.signed_temporal_changes,
        ordered_comparisons=args.ordered_comparisons,
        temporal_contrasts=args.temporal_contrasts,
        literal_ratio_contracts=args.literal_ratio_contracts,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "questions": result["questions"],
                "answer_changes": len(result["answer_changes"]),
                "errors": len(result["errors"]),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )
    raise SystemExit(0 if not result["errors"] else 1)


if __name__ == "__main__":
    main()
