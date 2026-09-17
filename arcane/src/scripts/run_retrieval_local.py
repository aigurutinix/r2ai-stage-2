#!/usr/bin/env python3
"""Run the frozen local retrieval stage and write a resumable candidate union."""

import argparse
import hashlib
import json
from pathlib import Path

from docs import load_companies, load_reports, parse_question, required_report_years, retrieve_docs
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import load_reports as load_table_reports, retrieve_rows


def plan_for(row: dict) -> dict:
    return row.get("plan") or row.get("verified_plan") or row.get("draft_plan") or {}


def row_id(row: dict) -> int:
    value = row.get("id")
    return int(value if value is not None else row["question_id"])


def fact_queries(row: dict) -> list[str]:
    facts = plan_for(row).get("facts") or row.get("facts") or []
    queries = []
    for fact in facts:
        if fact.get("retrieval_candidate", True) is False:
            continue
        query = fact.get("query") or fact.get("metric")
        if query:
            queries.append(query)
    return list(dict.fromkeys(queries))


def retrieve_one(row: dict, companies: dict, documents: dict, tables: list, report_meta: dict,
                 depth: int, fact_depth: int) -> dict:
    plan = plan_for(row)
    if not plan.get("facts"):
        raise ValueError(f"question {row_id(row)} has no plan facts")
    parsed = parse_question(row["question"], companies)
    docs, decisions = retrieve_docs(parsed, documents)
    metadata = {
        "tickers": parsed.tickers,
        "years": parsed.years,
        "slot_years": required_report_years(parsed),
        "scope": parsed.scope,
    }
    queries = list(dict.fromkeys([row["question"], *fact_queries(row)]))
    merged: dict[str, dict] = {}
    for index, query in enumerate(queries):
        result = retrieve_rows(
            query,
            metadata,
            tables,
            top_k=depth if index == 0 else fact_depth,
            report_ids=docs or None,
            mode="report-coverage",
            hierarchy=False,
        )
        for rank, table in enumerate(result["tables"], 1):
            item = merged.setdefault(table["table_id"], {**table, "retrieval_score": 0.0, "retrieval_queries": []})
            item["retrieval_score"] += 1 / (60 + rank)
            item["retrieval_queries"].append(query)
    candidates = sorted(merged.values(), key=lambda item: (-item["retrieval_score"], item["table_id"]))[:depth]
    for rank, table in enumerate(candidates, 1):
        identity = report_meta[table["report_id"]]
        table.update({"retrieval_rank": rank, "ticker": identity.ticker, "year": identity.year, "scope": identity.scope})
    return {
        "id": row_id(row),
        "question": row["question"],
        "metadata": metadata,
        "plan": plan,
        "facts": fact_queries(row),
        "docs": docs,
        "decisions": decisions,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/vifinqa"))
    parser.add_argument("--plans", type=Path, default=Path("output/final_stretch/fact_slots_v3_1.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/retrieval_local"))
    parser.add_argument("--depth", type=int, default=130)
    parser.add_argument("--fact-depth", type=int, default=10)
    parser.add_argument("--start", type=int, default=0, help="zero-based plan offset")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--refresh", action="store_true", help="recompute selected checkpoint rows")
    args = parser.parse_args()

    all_plans = load_jsonl(args.plans)
    end = args.start + args.limit if args.limit is not None else None
    plans = all_plans[args.start:end]
    companies = load_companies(args.data_root / "code_stock.csv")
    documents = load_reports(args.data_root / "financial_statements")
    tables = load_table_reports(args.data_root)
    report_meta = {report.identity.report_id: report.identity for report in tables}

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "candidates.jsonl"
    done = {str(item["id"]): item for item in load_jsonl(checkpoint)} if checkpoint.exists() else {}
    with checkpoint.open("a", encoding="utf-8") as handle:
        for number, row in enumerate(plans, 1):
            key = str(row_id(row))
            if key not in done or args.refresh:
                print(f"retrieving question {key}", flush=True)
                done[key] = retrieve_one(row, companies, documents, tables, report_meta, args.depth, args.fact_depth)
                handle.write(json.dumps(done[key], ensure_ascii=False) + "\n")
                handle.flush()
            if number % 25 == 0 or number == len(plans):
                print(f"retrieved {number}/{len(plans)}")

    missing = [row_id(row) for row in all_plans if str(row_id(row)) not in done]
    if missing and args.start == 0 and args.limit is None:
        raise RuntimeError(f"missing questions: {missing[:5]}")
    ordered = [done[str(row_id(row))] for row in all_plans if str(row_id(row)) in done]
    checkpoint.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8")
    manifest = {
        "questions": len(ordered),
        "depth": args.depth,
        "fact_depth": args.fact_depth,
        "candidate_pairs": sum(len(item["candidates"]) * len(item["facts"]) for item in ordered),
        "candidates_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    (args.output_dir / "retrieval_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
