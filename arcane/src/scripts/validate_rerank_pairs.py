#!/usr/bin/env python3
"""Validate the full CPU candidate pool before spending GPU time on it."""

import argparse
import hashlib
import json
from pathlib import Path

from export_rerank_pairs import dense_report_ids
from vifinqa.jsonl import load_jsonl
from vifinqa.retrieval import load_reports

ROOT = Path(__file__).resolve().parents[1]
SOURCE_FILES = (
    "scripts/export_rerank_pairs.py",
    "scripts/add_graph_tables.py",
    "src/vifinqa/retrieval.py",
    "src/vifinqa/rerank.py",
    "src/vifinqa/fusion.py",
    "src/vifinqa/graph.py",
    "src/vifinqa/notes.py",
    "src/vifinqa/slots.py",
    "data/derived/line_items.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_map(path: Path) -> dict[str, str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return {str(table_id): str(report_id) for table_id, report_id in value.items()}


def validate(records: list[dict], questions: list[dict], reports: dict,
             extra_tables: dict[str, str], traces: list[dict]) -> dict:
    question_ids = [str(record["id"]) for record in questions]
    pair_ids = [str(record["id"]) for record in records]
    if pair_ids != question_ids:
        raise ValueError("pair question ids or order differ from the released questions")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("duplicate question id in pair file")

    trace_by_id = {str(trace["id"]): trace for trace in traces}
    if len(trace_by_id) != len(traces) or set(trace_by_id) != set(pair_ids):
        raise ValueError("graph trace question ids do not match pair file")

    pairs = dense = dense_only = graph = 0
    for record in records:
        identifier = str(record["id"])
        candidates = record.get("candidates", [])
        table_ids = [candidate.get("table_id") for candidate in candidates]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError(f"question {identifier}: duplicate candidate")
        if any(not table_id or not candidate.get("text", "").strip()
               for table_id, candidate in zip(table_ids, candidates)):
            raise ValueError(f"question {identifier}: empty table id or candidate text")

        allowed = dense_report_ids(reports, record.get("selected_docs", []), "same-triple")
        traced = {item["table_id"] for item in trace_by_id[identifier].get("added", [])}
        candidate_graph = {
            candidate["table_id"] for candidate in candidates
            if candidate.get("graph_rank") is not None
        }
        if candidate_graph != traced:
            raise ValueError(f"question {identifier}: graph candidates and trace differ")

        for candidate in candidates:
            table_id = candidate["table_id"]
            report_id, separator, line = table_id.rpartition("|")
            if not separator or not line.isdigit() or report_id not in reports:
                raise ValueError(f"question {identifier}: unresolved table {table_id}")
            if candidate.get("dense_rank") is not None:
                dense += 1
                if candidate.get("sparse_rank") is None:
                    dense_only += 1
                if report_id not in allowed:
                    raise ValueError(
                        f"question {identifier}: dense table violates same-triple: {table_id}"
                    )
                if extra_tables.get(table_id) != report_id:
                    raise ValueError(f"question {identifier}: dense sidecar missing {table_id}")
            if candidate.get("graph_rank") is not None:
                graph += 1
                if extra_tables.get(table_id) != report_id:
                    raise ValueError(f"question {identifier}: graph sidecar missing {table_id}")
        pairs += len(candidates)

    return {
        "questions": len(records),
        "pairs": pairs,
        "dense_candidates": dense,
        "dense_only_candidates": dense_only,
        "graph_candidates": graph,
        "duplicate_candidates": 0,
        "same_triple_violations": 0,
        "untraced_graph_candidates": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--extra-tables", type=Path, required=True)
    parser.add_argument("--graph-trace", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=ROOT / "data" / "raw" / "vifinqa")
    parser.add_argument("--questions", type=Path,
                        default=ROOT / "data" / "raw" / "vifinqa" /
                        "questions" / "questions.jsonl")
    parser.add_argument("--dense-manifest", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    records = load_jsonl(args.pairs)
    questions = load_jsonl(args.questions)
    traces = load_jsonl(args.graph_trace)
    reports = {report.identity.report_id: report for report in load_reports(args.dataset_root)}
    summary = validate(records, questions, reports, load_map(args.extra_tables), traces)

    inputs = {
        "pairs": sha256(args.pairs),
        "extra_tables": sha256(args.extra_tables),
        "graph_trace": sha256(args.graph_trace),
        "questions": sha256(args.questions),
    }
    if args.dense_manifest:
        inputs["dense_manifest"] = sha256(args.dense_manifest)
    sources = {name: sha256(ROOT / name) for name in SOURCE_FILES}
    manifest = {"validation": summary, "sha256": inputs, "source_sha256": sources}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(args.manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
