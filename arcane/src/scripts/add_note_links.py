#!/usr/bin/env python3
"""Append each selected table's declared `Thuyết minh` note to a ranking.

Additive by design: the incoming order is preserved exactly and note tables are
appended behind it, so a ranking that already scores well cannot be reordered by
this step. Only the budget decides what survives.
"""

import argparse
import json
from pathlib import Path

from vifinqa.jsonl import load_jsonl
from vifinqa.notes import linked_note_lines
from vifinqa.slots import load_line_items, named_line_items
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


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    temporary.replace(path)


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
    parser.add_argument("--budget-multiplier", type=int, default=2)
    parser.add_argument("--max-links", type=int, default=3,
                        help="most note tables appended to one question")
    args = parser.parse_args()

    labels = load_line_items(args.line_items)
    records = load_jsonl(args.pairs)
    base = json.loads(args.base_ranking.read_text(encoding="utf-8"))
    reports = report_index(args.dataset_root)

    # A note table is one the BM25 gate never ranked, so `run.py` cannot rebuild
    # it from its own retrieval. Handing over the table-to-report map is what
    # lets it materialize one, exactly as it does for a promoted dense candidate.
    rankings, traces, link_reports = {}, [], {}
    added_total = 0
    for record in records:
        identifier = str(record["id"])
        order = list(base.get(identifier, []))
        items = named_line_items(record["question"], labels)
        budget = min(30, max(1, args.budget_multiplier * len(record.get("selected_docs", []))))

        # Only the prefix that survives the budget is actually submitted, so it
        # is the only place a link can start from, and the only membership test
        # that means anything. Seeding `seen` from the whole ranking suppressed
        # a note whose table sits below the budget — a table nothing submits —
        # and following links from below the budget proposed notes for rows the
        # package never carries.
        prefix = order[:budget]
        links, seen = [], set(prefix)
        for table_id in prefix:
            if len(links) >= args.max_links:
                break
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
            for note, line in linked_note_lines(text, grid, items, source_line):
                linked = f"{report_id}|{line}"
                if linked in seen:
                    continue
                seen.add(linked)
                link_reports[linked] = report_id
                links.append({"from": table_id, "note": note, "table_id": linked})
                if len(links) >= args.max_links:
                    break

        # The base ranking already arrives trimmed to its budget, so links have
        # to extend past it; truncating the concatenation would discard them all.
        # The package cap of 30 still binds.
        selected = (prefix + [link["table_id"] for link in links])[:30]
        rankings[identifier] = selected
        added_total += sum(link["table_id"] in selected for link in links)
        traces.append({
            "id": record["id"], "line_items": items, "budget": budget,
            "links": links, "selected": selected,
        })

    write_json(args.output, rankings)
    note_tables_path = args.output.with_suffix(".note_tables.json")
    write_json(note_tables_path, link_reports)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.trace.with_suffix(args.trace.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(trace, ensure_ascii=False) + "\n" for trace in traces),
        encoding="utf-8",
    )
    temporary.replace(args.trace)

    linked_questions = sum(bool(trace["links"]) for trace in traces)
    print(json.dumps({
        "questions": len(records),
        "questions_with_a_link": linked_questions,
        "links_found": sum(len(trace["links"]) for trace in traces),
        "links_inside_budget": added_total,
        "output": str(args.output),
        "note_tables": str(note_tables_path),
        "trace": str(args.trace),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
