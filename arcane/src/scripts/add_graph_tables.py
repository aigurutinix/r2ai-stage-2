#!/usr/bin/env python3
"""Union a base ranking with tables reached by the slot -> row -> table path.

Additive only. The incoming order is preserved as a prefix and graph tables are
appended behind it, so this step can add evidence but can never reorder a
ranking that already scores well.

Also writes the table-to-report map `run.py` needs to materialize a table the
lexical gate never ranked, under `<output>.graph_tables.json`.
"""

import argparse
import json
from pathlib import Path

from vifinqa.fusion import first_stage
from vifinqa.graph import NOTE_REF, evidence_slots, select_paths, table_paths
from vifinqa.jsonl import load_jsonl
from vifinqa.rerank import INVENTORY, table_representation
from vifinqa.retrieval import bm25, report_tables, tokenize
from vifinqa.slots import load_line_items, named_line_items, normalize
from vifinqa.tables import (
    cached_report_text,
    extract_rows_at_line,
    iter_report_paths,
    parse_report_identity,
)


def report_index(dataset_root: Path) -> dict[str, Path]:
    return {
        parse_report_identity(path, dataset_root).report_id: path
        for path in iter_report_paths(dataset_root)
    }


def split_table_id(table_id: str) -> tuple[str, int] | None:
    report_id, _, line = table_id.rpartition("|")
    return (report_id, int(line)) if report_id and line.isdigit() else None


def candidate_order(record: dict, arm: str) -> list[str]:
    scored = [c for c in record["candidates"] if first_stage(c, arm) > 0]
    scored.sort(key=lambda c: -first_stage(c, arm))
    return [c["table_id"] for c in scored]


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


def best_row(table, line_item: str) -> int:
    if not table.rows:
        return 0
    scores = bm25(tokenize(line_item), [" ".join(row) for row in table.rows])
    return max(range(len(scores)), key=scores.__getitem__)


def graph_candidate(path, rank: int, reports: dict[str, Path], dataset_root: Path,
                    inventory: int, hierarchy: bool) -> dict | None:
    report_id = path.edge.anchor.report_id
    report_path = reports.get(report_id)
    if report_path is None:
        return None
    identity = parse_report_identity(report_path, dataset_root)
    table = next(
        (table for table in report_tables(str(report_path), identity)
         if table.table_id == path.edge.table_id),
        None,
    )
    if table is None:
        return None
    return {
        "table_id": table.table_id,
        "sparse_rank": rank,
        "graph_rank": rank,
        "graph_kind": path.edge.kind,
        "graph_source": path.edge.anchor.table_id,
        "text": table_representation(
            table, best_row(table, path.slot.line_item), inventory=inventory,
            identity=identity, hierarchy=hierarchy,
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--base-ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=root / "data" / "raw" / "vifinqa")
    parser.add_argument("--line-items", type=Path,
                        default=root / "data" / "derived" / "line_items.json")
    parser.add_argument("--arm", choices=("sparse", "dense", "hybrid"), default="sparse")
    parser.add_argument("--depth", type=int, default=30,
                        help="candidate tables per question searched for row anchors")
    parser.add_argument("--max-added", type=int, default=3,
                        help="most graph tables appended to one question")
    parser.add_argument("--no-notes", action="store_true",
                        help="use SOURCE_TABLE edges only")
    parser.add_argument("--expanded-pairs", type=Path,
                        help="optional reranker pairs with graph tables appended")
    parser.add_argument("--hierarchy", action="store_true",
                        help="render appended candidates with row/column paths")
    parser.add_argument("--inventory", type=int, default=INVENTORY)
    args = parser.parse_args()

    labels = load_line_items(args.line_items)
    records = load_jsonl(args.pairs)
    base = json.loads(args.base_ranking.read_text(encoding="utf-8"))
    reports = report_index(args.dataset_root)

    rankings, traces, graph_reports, expanded_records = {}, [], {}, []
    kinds: dict[str, int] = {}
    for record in records:
        identifier = str(record["id"])
        order = list(base.get(identifier, []))
        items = named_line_items(record["question"], labels)

        paths = []
        if items:
            slots = evidence_slots(record.get("selected_docs", []), items)
            candidates = {candidate["table_id"]: candidate for candidate in record["candidates"]}
            covered = {
                slot for slot in slots
                if any(
                    table_id.rsplit("|", 1)[0] == slot.report_id
                    and f" {slot.line_item} "
                    in f" {normalize(candidates.get(table_id, {}).get('text', ''))} "
                    for table_id in order
                )
            }
            for table_id in candidate_order(record, args.arm)[: args.depth]:
                parts = split_table_id(table_id)
                if parts is None:
                    continue
                report_id, source_line = parts
                path = reports.get(report_id)
                if path is None:
                    continue
                text = cached_report_text(str(path))
                try:
                    grid = extract_rows_at_line(text, source_line)
                except KeyError:
                    continue
                for slot in slots:
                    found = table_paths(
                        slot, table_id, grid, text, source_line,
                        follow_notes=not args.no_notes,
                    )
                    if slot in covered:
                        found = [
                            path for path in found
                            if path.edge.kind == NOTE_REF and path.anchor.table_id in order
                        ]
                    paths.extend(found)

        added = select_paths(paths, set(order), args.max_added)
        for path in added:
            edge = path.edge
            graph_reports[edge.table_id] = edge.anchor.report_id
            kinds[edge.kind] = kinds.get(edge.kind, 0) + 1

        rankings[identifier] = (order + [path.edge.table_id for path in added])[:30]
        if args.expanded_pairs:
            expanded = {**record, "candidates": [dict(candidate) for candidate in record["candidates"]]}
            by_table = {candidate["table_id"]: candidate for candidate in expanded["candidates"]}
            next_rank = max(
                (candidate.get("sparse_rank") or 0 for candidate in expanded["candidates"]),
                default=0,
            )
            for graph_rank, path in enumerate(added, 1):
                existing = by_table.get(path.edge.table_id)
                if existing is not None:
                    existing["graph_rank"] = graph_rank
                    existing["graph_kind"] = path.edge.kind
                    existing["graph_source"] = path.edge.anchor.table_id
                    continue
                candidate = graph_candidate(
                    path, next_rank + graph_rank, reports, args.dataset_root,
                    args.inventory, args.hierarchy,
                )
                if candidate is not None:
                    candidate["graph_rank"] = graph_rank
                    expanded["candidates"].append(candidate)
                    by_table[candidate["table_id"]] = candidate
            expanded_records.append(expanded)
        traces.append({
            "id": record["id"], "line_items": items,
            "added": [
                {
                    "slot": {"report_id": path.slot.report_id,
                             "line_item": path.slot.line_item},
                    "table_id": edge.table_id, "kind": edge.kind,
                    "from_table": edge.anchor.table_id, "row": edge.anchor.row,
                    "row_label": edge.anchor.label, "line_item": edge.anchor.line_item,
                    "code": edge.anchor.code, "note": edge.anchor.note,
                }
                for path in added for edge in [path.edge]
            ],
            "selected": rankings[identifier],
        })

    write_json(args.output, rankings)
    graph_tables_path = args.output.with_suffix(".graph_tables.json")
    write_json(graph_tables_path, graph_reports)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.trace.with_suffix(args.trace.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    temporary.replace(args.trace)

    if args.expanded_pairs:
        args.expanded_pairs.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.expanded_pairs.with_suffix(args.expanded_pairs.suffix + ".tmp")
        temporary.write_text(
            "".join(json.dumps(record, ensure_ascii=False) + "\n"
                    for record in expanded_records),
            encoding="utf-8",
        )
        temporary.replace(args.expanded_pairs)

    print(json.dumps({
        "questions": len(records),
        "questions_with_added_tables": sum(bool(t["added"]) for t in traces),
        "tables_added": sum(len(t["added"]) for t in traces),
        "by_edge": kinds,
        "output": str(args.output),
        "graph_tables": str(graph_tables_path),
        "trace": str(args.trace),
        "expanded_pairs": str(args.expanded_pairs) if args.expanded_pairs else None,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
