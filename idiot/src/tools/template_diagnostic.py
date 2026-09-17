"""Produce a compact, source-backed review queue for template answers.

This is intentionally read-only.  It joins the final submission row, the
solver checkpoint, and the CSV evidence that will actually be shipped, so a
reviewer never has to infer an answer from stale LLM logs or an intermediate
candidate table.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from road2ai_vifinqa.template_solver import _AUDITED_OVERRIDES


def _read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("runs/iteration_4"))
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--include-audited", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/template_review_queue.json"))
    args = parser.parse_args()

    build = args.run_dir / "build"
    submission = _read_json(build / "submission.json")
    if not isinstance(submission, list):
        raise TypeError("submission.json must be a JSON list")
    by_id = {int(row["id"]): row for row in submission if isinstance(row, dict)}

    reviews: list[dict[str, object]] = []
    for checkpoint_path in sorted((args.run_dir / "checkpoints").glob("q*.json")):
        checkpoint = _read_json(checkpoint_path)
        if not isinstance(checkpoint, dict) or checkpoint.get("route") != "template":
            continue
        qid = int(checkpoint["id"])
        audited = qid in _AUDITED_OVERRIDES
        if audited and not args.include_audited:
            continue
        row = by_id[qid]
        evidence_rows: list[dict[str, str]] = []
        for evidence in row.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            csv_path = evidence.get("csv_path")
            if not isinstance(csv_path, str) or not csv_path.startswith("data/"):
                continue
            csv_file = build / csv_path
            with csv_file.open(encoding="utf-8-sig", newline="") as handle:
                evidence_rows.extend(csv.DictReader(handle))
        reviews.append(
            {
                "id": qid,
                "confidence": float(checkpoint.get("confidence", 0.0)),
                "method": checkpoint.get("method"),
                "audited_override": audited,
                "question": row.get("question"),
                "answer": row.get("answer"),
                "relevant_docs": row.get("relevant_docs"),
                "relevant_tables": row.get("relevant_tables"),
                "source_rows": [
                    {
                        key: source.get(key)
                        for key in (
                            "ticker",
                            "year",
                            "value",
                            "doc_id",
                            "table_id",
                            "row_idx",
                            "col_idx",
                            "raw_value",
                            "label",
                            "source_scale",
                            "computed_answer",
                        )
                    }
                    for source in evidence_rows
                ],
            }
        )

    reviews.sort(key=lambda item: (float(item["confidence"]), int(item["id"])))
    if args.limit > 0:
        reviews = reviews[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(reviews, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(reviews)} review items to {args.output}")
    for item in reviews:
        print(
            f"{item['id']:04d}\t{item['confidence']:.3f}\t{item['method']}\t"
            f"{item['answer']}\t{item['question']}"
        )


if __name__ == "__main__":
    main()
