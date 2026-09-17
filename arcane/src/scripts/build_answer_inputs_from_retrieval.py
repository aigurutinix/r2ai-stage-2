#!/usr/bin/env python3
"""Materialize answer-stage inputs from retrieval/reranking artifacts."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def ranked_ids(ranked: dict | None, candidate_ids: list[str], per_fact: int, max_tables: int) -> list[str]:
    if not ranked:
        return candidate_ids[:max_tables]
    ids = []
    for values in ranked.get("fact_rankings", {}).values():
        ids.extend(values[:per_fact])
    ids.extend(ranked.get("candidate_ids", []))
    return list(dict.fromkeys(table_id for table_id in ids if table_id in candidate_ids))[:max_tables]


def build(retrieval_run: Path, output_dir: Path, ranked_path: Path | None, tables_per_fact: int, max_tables: int) -> dict:
    candidates_path = retrieval_run / "candidates.jsonl"
    if not candidates_path.is_file():
        raise FileNotFoundError(candidates_path)
    items = load_jsonl(candidates_path)
    if len(items) != 1012 or len({int(item["id"]) for item in items}) != 1012:
        raise ValueError("retrieval output must contain 1,012 unique questions")

    ranked = {}
    if ranked_path:
        ranked = {str(item["id"]): item for item in load_jsonl(ranked_path)}
        if set(ranked) != {str(item["id"]) for item in items}:
            raise ValueError("ranked output question IDs do not match retrieval output")

    tables_dir = output_dir / "data" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    table_count = 0
    for item in items:
        by_id = {table["table_id"]: table for table in item.get("candidates", [])}
        candidate_ids = list(by_id)
        selected = ranked_ids(ranked.get(str(item["id"])), candidate_ids, tables_per_fact, max_tables)
        if not selected:
            raise ValueError(f"question {item['id']} has no selected tables")

        evidence = []
        for position, table_id in enumerate(selected):
            table = by_id[table_id]
            path = tables_dir / f"table_{item['id']}_{position}.csv"
            rows = table.get("rows") or []
            if not rows:
                raise ValueError(f"table {table_id} has no rows")
            width = max(len(row) for row in rows)
            lines = [",".join(f'"col_{index}"' for index in range(width))]
            for row in rows:
                values = [str(value).replace('"', '""') for value in row]
                values.extend([""] * (width - len(values)))
                lines.append(",".join(f'"{value}"' for value in values))
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            evidence.append({"variable": f"df{position}", "csv_path": f"data/tables/{path.name}"})
            table_count += 1

        selected_tables = [by_id[table_id] for table_id in selected]
        manifest_rows.append({
            "id": int(item["id"]),
            "question": item["question"],
            "metadata": item.get("metadata", {}),
            "relevant_docs": list(dict.fromkeys(table["report_id"] for table in selected_tables)),
            "relevant_tables": selected,
            "evidence": evidence,
        })

    manifest_path = output_dir / "retrieval_manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows),
        encoding="utf-8",
    )
    manifest = {
        "questions": len(manifest_rows),
        "tables": table_count,
        "candidates_sha256": sha256(candidates_path),
        "ranked_sha256": sha256(ranked_path) if ranked_path else None,
        "retrieval_manifest_sha256": sha256(manifest_path),
        "tables_per_fact": tables_per_fact,
        "max_tables": max_tables,
    }
    (output_dir / "answer_inputs_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--ranked", type=Path, help="reranked.jsonl or cascade output")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tables-per-fact", type=int, default=5)
    parser.add_argument("--max-tables", type=int, default=30)
    args = parser.parse_args()
    print(json.dumps(build(args.retrieval_run, args.output_dir, args.ranked, args.tables_per_fact, args.max_tables), indent=2))


if __name__ == "__main__":
    main()
