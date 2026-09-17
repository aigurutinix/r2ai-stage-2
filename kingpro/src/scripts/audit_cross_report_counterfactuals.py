"""Execute later-report comparative restatements as in-memory counterfactuals.

This audit never edits a submission.  It takes the review-only differences from
``audit_cross_report_comparatives.py``, substitutes every later comparative
value into the corresponding compact source-cell frame, then executes the
question's real ``pandas_query``.  The output distinguishes differences that
cannot affect the final answer from restatements that actually propagate.
"""

from __future__ import annotations

import argparse
import builtins
import copy
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd


_WHITELIST = (
    "abs round len min max sum sorted float int str bool list dict set "
    "range enumerate zip all any isinstance"
).split()
_SAFE_BUILTINS = {name: getattr(builtins, name) for name in _WHITELIST}


def _norm(path: str | Path) -> str:
    return str(path).replace("\\", "/").lower()


def _scalar(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _run(code: str, frames: dict[str, pd.DataFrame]) -> Any:
    runtime = {key: frame.copy(deep=True) for key, frame in frames.items()}
    namespace: dict[str, Any] = {
        "pd": pd,
        "dfs": runtime,
        "__builtins__": _SAFE_BUILTINS,
    }
    if len(runtime) == 1:
        namespace["df"] = next(iter(runtime.values()))
    exec(compile(code, "<cross_report_counterfactual>", "exec"), namespace)  # noqa: S102
    if "result" not in namespace:
        raise RuntimeError("no-result-var")
    return _scalar(namespace["result"])


def _equal(left: Any, right: Any, tolerance: float = 0.005) -> bool:
    try:
        return math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
    except (TypeError, ValueError):
        return str(left).strip() == str(right).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparatives", type=Path)
    parser.add_argument("lineage", type=Path)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    comparisons = json.loads(args.comparatives.read_text(encoding="utf-8"))
    lineage = json.loads(args.lineage.read_text(encoding="utf-8"))
    submission = json.loads((candidate / "submission.json").read_text(encoding="utf-8"))

    entries = {int(row["id"]): row for row in submission}
    lineage_by_id = {int(row["id"]): row for row in lineage["records"]}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for finding in comparisons.get("findings", []):
        grouped.setdefault(int(finding["id"]), []).append(finding)

    records: list[dict[str, Any]] = []
    for question_id in sorted(grouped):
        entry = entries[question_id]
        source_record = lineage_by_id[question_id]
        frames: dict[str, pd.DataFrame] = {}
        frame_paths: dict[str, str] = {}
        for evidence in entry.get("evidence", []):
            variable = str(evidence["variable"])
            relative = str(evidence["csv_path"])
            frames[variable] = pd.read_csv(
                candidate / relative,
                encoding="utf-8-sig",
                dtype=str,
                keep_default_na=False,
                index_col=None,
            )
            frame_paths[variable] = _norm(relative)

        record: dict[str, Any] = {
            "id": question_id,
            "question": entry.get("question", ""),
            "stored_answer": entry.get("answer"),
            "substitution_count": 0,
            "substitutions": [],
        }
        try:
            baseline = _run(entry["pandas_query"], frames)
            mutated = {key: value.copy(deep=True) for key, value in frames.items()}
            for finding in grouped[question_id]:
                cell_index = int(finding["cell_index"])
                cell = source_record["cells"][cell_index]
                csv_key = _norm(cell["csv"])
                candidates = [key for key, value in frame_paths.items() if value == csv_key]
                if len(candidates) != 1:
                    raise RuntimeError(f"frame-resolution:{cell['csv']}:{candidates}")
                variable = candidates[0]
                source_index = int(cell["source_index"])
                frame = mutated[variable]
                if source_index >= len(frame):
                    raise IndexError(f"source-index:{source_index}>={len(frame)}")
                if "raw" not in frame.columns:
                    raise KeyError("raw")
                old_raw = str(frame.iloc[source_index]["raw"])
                expected_raw = str(finding["current_raw"])
                if old_raw.strip() != expected_raw.strip():
                    raise RuntimeError(
                        f"raw-attestation:q{question_id}:row{source_index}:"
                        f"{old_raw!r}!={expected_raw!r}"
                    )
                raw_column = frame.columns.get_loc("raw")
                frame.iat[source_index, raw_column] = str(finding["comparative_raw"])
                record["substitutions"].append(
                    {
                        "cell_index": cell_index,
                        "source_index": source_index,
                        "ticker": finding["ticker"],
                        "year": finding["year"],
                        "metric_key": finding["metric_key"],
                        "old_raw": old_raw,
                        "new_raw": str(finding["comparative_raw"]),
                    }
                )
            counterfactual = _run(entry["pandas_query"], mutated)
            record.update(
                {
                    "baseline_result": baseline,
                    "baseline_matches_stored": _equal(baseline, entry.get("answer")),
                    "counterfactual_result": counterfactual,
                    "answer_changes": not _equal(baseline, counterfactual),
                    "substitution_count": len(record["substitutions"]),
                    "status": "ok",
                }
            )
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        records.append(record)

    changed = [row for row in records if row.get("answer_changes")]
    errors = [row for row in records if row["status"] == "error"]
    report = {
        "kind": "cross_report_comparative_counterfactual_audit",
        "candidate": str(candidate),
        "comparatives": str(args.comparatives.resolve()),
        "lineage": str(args.lineage.resolve()),
        "question_count": len(records),
        "successful_count": len(records) - len(errors),
        "changed_answer_count": len(changed),
        "changed_answer_ids": [row["id"] for row in changed],
        "error_count": len(errors),
        "error_ids": [row["id"] for row in errors],
        "records": records,
        "claim_limit": (
            "Counterfactual propagation only. A changed result is a review queue, "
            "not evidence that the later restatement is the intended gold source."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: report[key] for key in (
                "question_count",
                "successful_count",
                "changed_answer_count",
                "changed_answer_ids",
                "error_count",
                "error_ids",
            )},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
