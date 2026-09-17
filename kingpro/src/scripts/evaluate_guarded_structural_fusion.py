#!/usr/bin/env python3
"""Offline proxy gate for guarded structural table rescue.

The script never writes the canonical catalog or BM25 index.  It accepts an
explicit structural catalog (normally under a temporary directory), builds its
index in memory, and compares baseline/candidate/fused top-eight tables against
the source-known ``relevant_tables`` of a submission.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import bm25s


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import kingpro.retrieval.bm25_index as bi  # noqa: E402
from kingpro.retrieval.structural_fusion import guarded_structural_rescue  # noqa: E402
from kingpro.retrieval.table_reranker import (  # noqa: E402
    labels_from_csv,
    rerank_tables,
)


def load_catalog(path: Path) -> tuple[list[dict], dict[str, dict]]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    return rows, {row["table_ref"]: row for row in rows}


def build_index(rows: list[dict]):
    retriever = bm25s.BM25()
    retriever.index([bi.tokenize(row["search_text"]) for row in rows], show_progress=False)
    meta = [
        {key: row[key] for key in ("table_ref", "ticker", "year", "scope")}
        for row in rows
    ]
    return retriever, meta


def selected_tables(question: str, targets: list[str], catalog: dict[str, dict], *, n: int) -> list[dict]:
    reports = sorted({ref.split("|", 1)[0] for ref in targets})
    raw = bi.tables_in_reports(question, reports, n=40, pool=2000)
    return rerank_tables(
        question,
        raw,
        catalog,
        ROOT / "build" / "tables",
        label_weight=4.0,
    )[:n]


def metric(selected: list[dict], targets: list[str]) -> dict:
    refs = [str(item["table_ref"]) for item in selected]
    target_set = set(targets)
    overlap = target_set.intersection(refs)
    reciprocal = next(
        (1.0 / rank for rank, ref in enumerate(refs[:5], 1) if ref in target_set),
        0.0,
    )
    return {
        "recall_at_8": len(overlap) / len(target_set) if target_set else 0.0,
        "mrr_at_5": reciprocal,
        "precision_proxy": len(overlap) / len(refs) if refs else 0.0,
        "selected": refs,
    }


def summarize(ids: list[int], records: dict[int, dict], variant: str) -> dict:
    rows = [records[qid][variant] for qid in ids]
    count = len(rows)
    return {
        "questions": count,
        "recall_at_8": sum(row["recall_at_8"] for row in rows) / count,
        "mrr_at_5": sum(row["mrr_at_5"] for row in rows) / count,
        "precision_proxy": sum(row["precision_proxy"] for row in rows) / count,
    }


def regressions(ids: list[int], records: dict[int, dict], variant: str) -> dict[str, list[int]]:
    output = {key: [] for key in ("recall_at_8", "mrr_at_5", "precision_proxy")}
    for qid in ids:
        for key in output:
            if records[qid][variant][key] < records[qid]["baseline"][key] - 1e-12:
                output[key].append(qid)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-catalog", type=Path, required=True)
    parser.add_argument(
        "--submission",
        type=Path,
        default=ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3/submission.json",
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=ROOT / "knowledge/vothuong/question_source_verdicts.json",
    )
    parser.add_argument("--ledger-first", type=int, default=55)
    parser.add_argument("--extra-ids", default="312,395")
    parser.add_argument(
        "--public-queue",
        type=Path,
        default=ROOT / "build/v227_residual_public100_fused_v217.json",
    )
    parser.add_argument(
        "--batch-queue",
        type=Path,
        default=ROOT / "build/v227_residual_batch150_fused_v217.json",
    )
    parser.add_argument("--skip-queues", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    submission = json.loads(args.submission.read_text(encoding="utf-8-sig"))
    by_id = {int(row["id"]): row for row in submission}
    ledger = json.loads(args.ledger.read_text(encoding="utf-8-sig"))
    first = [int(value) for value in list(ledger["verdicts"])[: args.ledger_first]]
    extras = [int(value) for value in args.extra_ids.split(",") if value.strip()]
    public = [] if args.skip_queues else [
        int(row["id"])
        for row in json.loads(args.public_queue.read_text(encoding="utf-8-sig"))["queue"]
    ]
    batch = [] if args.skip_queues else [
        int(row["id"])
        for row in json.loads(args.batch_queue.read_text(encoding="utf-8-sig"))["queue"]
    ]
    groups = {"first": first, "public100": public, "batch150": batch}
    ids = list(dict.fromkeys(first + public + batch + extras))

    baseline_rows, baseline_catalog = load_catalog(ROOT / "build/catalog.jsonl")
    candidate_rows, candidate_catalog = load_catalog(args.candidate_catalog)
    records: dict[int, dict] = {}

    bi._CACHE.clear()
    baseline_retriever, baseline_meta = bi._load_index()
    labels_from_csv.cache_clear()
    for qid in ids:
        row = by_id[qid]
        selected = selected_tables(
            row["question"], row["relevant_tables"], baseline_catalog, n=8
        )
        records[qid] = {
            "question": row["question"],
            "targets": row["relevant_tables"],
            "baseline_hits": selected,
            "baseline": metric(selected, row["relevant_tables"]),
        }
    del baseline_retriever, baseline_meta, baseline_rows
    bi._CACHE.clear()
    gc.collect()

    candidate_retriever, candidate_meta = build_index(candidate_rows)
    bi._CACHE.update({"r": candidate_retriever, "meta": candidate_meta})
    labels_from_csv.cache_clear()
    for qid in ids:
        row = by_id[qid]
        structural = selected_tables(
            row["question"], row["relevant_tables"], candidate_catalog, n=16
        )
        fused, trace = guarded_structural_rescue(
            row["question"],
            records[qid].pop("baseline_hits"),
            structural,
            candidate_catalog,
            limit=8,
        )
        records[qid]["candidate"] = metric(structural[:8], row["relevant_tables"])
        records[qid]["fused"] = metric(fused, row["relevant_tables"])
        records[qid]["fusion_trace"] = trace
    bi._CACHE.clear()
    labels_from_csv.cache_clear()

    report = {
        "schema": "guarded-structural-fusion-eval-v1",
        "candidate_catalog": str(args.candidate_catalog.resolve()),
        "first_ids": first,
        "extra_ids": extras,
        "summary": {
            variant: summarize(first, records, variant)
            for variant in ("baseline", "candidate", "fused")
        },
        "regressions": {
            variant: regressions(first, records, variant)
            for variant in ("candidate", "fused")
        },
        "q312": records.get(312),
        "q395": records.get(395),
        "cohorts": {
            name: {
                "summary": {
                    variant: summarize(group_ids, records, variant)
                    for variant in ("baseline", "candidate", "fused")
                },
                "regressions": {
                    variant: regressions(group_ids, records, variant)
                    for variant in ("candidate", "fused")
                },
            }
            for name, group_ids in groups.items()
            if group_ids
        },
        "records": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(args.out), **report["summary"], "regressions": report["regressions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
