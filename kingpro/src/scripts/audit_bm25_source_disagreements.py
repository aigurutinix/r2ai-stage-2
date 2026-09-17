"""Build a source-review queue from BM25/current-table disagreements.

This is a diagnostic, not a hidden-gold evaluator.  For each submission row it
restricts retrieval to the documents already declared by the candidate, then
asks the repository BM25 + CSV-label reranker which table it would choose.  A
different top table is evidence for manual inspection only; it is never used
to rewrite an answer automatically.

The useful cases are legacy, single-table rows where the independent table has
substantially better label support and the current table has not already been
confirmed by a trusted source review.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.retrieval.bm25_index import tables_in_reports  # noqa: E402
from kingpro.retrieval.table_reranker import (  # noqa: E402
    label_match_score,
    labels_from_csv,
    rerank_tables,
)


DEFAULT_CANDIDATE = ROOT / "sub_top123_candidate_v203_q24_q607_double_unit"
DEFAULT_OUTPUT = ROOT / "build" / "v203_bm25_source_disagreements.json"
TERMINAL_REVIEW_VERDICTS = {
    "source_confirmed",
    "corrected",
    "false_positive",
    "rejected",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def unique(values: Iterable[Any]) -> list[str]:
    return list(dict.fromkeys(str(value) for value in values if value))


def report_id(table_ref: str) -> str:
    return str(table_ref).split("|", 1)[0]


def target_tables(row: dict[str, Any], audit: dict[str, Any] | None) -> list[str]:
    """Prefer executed bindings, while excluding retrieval-only recall rows."""
    if audit:
        executed = unique(
            source.get("table_ref")
            for source in audit.get("sources", [])
            if not str(source.get("metric", "")).startswith("recall:")
        )
        if executed:
            return executed
    return unique(row.get("relevant_tables", []))


def trusted_review_ids(path: Path) -> set[int]:
    reviewed: set[int] = set()
    if not path.is_file():
        return reviewed
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except (TypeError, ValueError):
            continue
        if (
            event.get("kind") == "review"
            and event.get("oracle") == "source"
            and event.get("oracle_trust") == "that"
            and event.get("verdict") in TERMINAL_REVIEW_VERDICTS
        ):
            reviewed.update(int(value) for value in event.get("question_ids", []))
    return reviewed


def load_catalog(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows[str(row["table_ref"])] = row
    return rows


def table_labels(
    table_ref: str,
    catalog: dict[str, dict[str, Any]],
    tables_root: Path,
) -> tuple[str, ...]:
    meta = catalog.get(table_ref, {})
    csv_path = tables_root / str(meta.get("csv_path", ""))
    return labels_from_csv(str(csv_path.resolve())) if csv_path.is_file() else ()


def finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def priority_score(
    *,
    has_source_audit: bool,
    trusted_review: bool,
    target_count: int,
    target_rank: int | None,
    alternative_label_score: float,
    target_label_score: float,
) -> float:
    """Rank review value, not probability that the current answer is wrong."""
    if trusted_review:
        return 0.0
    score = 0.0
    score += 3.0 if not has_source_audit else 0.5
    score += 2.0 if target_count == 1 else 0.0
    score += 2.0 if target_rank is None else min(2.0, max(0, target_rank - 1) / 5.0)
    score += min(3.0, max(0.0, alternative_label_score - target_label_score) * 2.0)
    return round(score, 6)


def compact_hit(hit: dict[str, Any] | None) -> dict[str, Any] | None:
    if not hit:
        return None
    return {
        "table_ref": str(hit.get("table_ref", "")),
        "bm25_score": round(finite(hit.get("bm25_score", hit.get("score"))), 6),
        "label_match_score": round(finite(hit.get("label_match_score")), 6),
        "rerank_score": round(finite(hit.get("rerank_score", hit.get("score"))), 6),
        "retrieval_label_text": str(hit.get("retrieval_label_text", ""))[:1500],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--label-weight", type=float, default=4.0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    candidate = args.candidate.resolve()
    rows = read_json(candidate / "submission.json")
    if args.limit:
        rows = rows[: args.limit]
    audit_path = candidate / "source_audit.json"
    source_audit = (
        {int(row["id"]): row for row in read_json(audit_path)}
        if audit_path.is_file()
        else {}
    )
    panel_audit_path = candidate / "panel_source_audit.json"
    panel_audit = (
        {int(row["id"]): row for row in read_json(panel_audit_path)}
        if panel_audit_path.is_file()
        else {}
    )
    reviewed = trusted_review_ids(ROOT / "knowledge" / "vothuong" / "experiments.jsonl")
    catalog = load_catalog(ROOT / "build" / "catalog.jsonl")
    tables_root = ROOT / "build" / "tables"

    records: list[dict[str, Any]] = []
    agreement = 0
    target_missing = 0
    for position, row in enumerate(rows, 1):
        qid = int(row["id"])
        question = str(row.get("question", ""))
        audit = source_audit.get(qid)
        panel = panel_audit.get(qid)
        has_exact_audit = bool(audit and audit.get("sources")) or bool(
            panel and int(panel.get("source_cells", 0)) > 0
        )
        targets = target_tables(row, audit)
        if not targets:
            continue
        documents = unique(row.get("relevant_docs", [])) or unique(report_id(ref) for ref in targets)
        hits = tables_in_reports(question, documents, n=args.top_k)
        ranked = rerank_tables(
            question,
            hits,
            catalog,
            tables_root,
            label_weight=args.label_weight,
        )
        ranked_refs = unique(hit.get("table_ref") for hit in ranked)
        top = ranked[0] if ranked else None
        top_ref = str(top.get("table_ref", "")) if top else ""
        target_rank = next(
            (index for index, ref in enumerate(ranked_refs, 1) if ref in set(targets)),
            None,
        )
        if top_ref in set(targets):
            agreement += 1
            continue
        if target_rank is None:
            target_missing += 1

        target_details = []
        target_best_label = 0.0
        for ref in targets:
            labels = table_labels(ref, catalog, tables_root)
            match = label_match_score(question, labels)
            target_best_label = max(target_best_label, match)
            ranked_hit = next(
                (hit for hit in ranked if str(hit.get("table_ref", "")) == ref),
                None,
            )
            target_details.append(
                {
                    "table_ref": ref,
                    "label_match_score": round(match, 6),
                    "ranked_hit": compact_hit(ranked_hit),
                    "labels": list(labels[:80]),
                }
            )
        alternative_label = finite(top.get("label_match_score")) if top else 0.0
        priority = priority_score(
            has_source_audit=has_exact_audit,
            trusted_review=qid in reviewed,
            target_count=len(targets),
            target_rank=target_rank,
            alternative_label_score=alternative_label,
            target_label_score=target_best_label,
        )
        records.append(
            {
                "id": qid,
                "question": question,
                "answer": row.get("answer"),
                "classification": (
                    "trusted_review_disagreement"
                    if qid in reviewed
                    else "source_audited_disagreement"
                    if audit and audit.get("sources")
                    else "panel_source_audited_disagreement"
                    if panel and int(panel.get("source_cells", 0)) > 0
                    else "legacy_actionable_disagreement"
                ),
                "priority": priority,
                "trusted_source_review": qid in reviewed,
                "has_source_audit": bool(audit and audit.get("sources")),
                "has_panel_source_audit": bool(panel and int(panel.get("source_cells", 0)) > 0),
                "target_rank": target_rank,
                "target_docs": documents,
                "target_tables": targets,
                "target_details": target_details,
                "independent_top": compact_hit(top),
                "independent_top_8": [compact_hit(hit) for hit in ranked[:8]],
            }
        )
        if position % 100 == 0:
            print(f"audited {position}/{len(rows)}", flush=True)

    records.sort(key=lambda item: (-float(item["priority"]), int(item["id"])))
    actionable = [row for row in records if row["classification"] == "legacy_actionable_disagreement"]
    payload = {
        "kind": "source_review_queue_not_hidden_gold",
        "candidate": candidate.name,
        "configuration": {"top_k": args.top_k, "label_weight": args.label_weight},
        "summary": {
            "questions_scanned": len(rows),
            "top1_agreement": agreement,
            "top1_disagreement": len(records),
            "target_absent_top_k": target_missing,
            "legacy_actionable_disagreements": len(actionable),
            "legacy_actionable_positive_priority": sum(float(row["priority"]) > 0 for row in actionable),
        },
        "actionable_queue": actionable,
        "all_disagreements": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
