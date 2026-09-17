"""Promote formula/note schema evidence into positions 6-15 for codegen."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vifinqa import config
from vifinqa.extraction.build_store import Store
from vifinqa.retrieval.schema_rerank import rerank_codegen_tail
from vifinqa.utils.io import read_jsonl, setup_stdout, write_jsonl


def _read_ids(path: str) -> set[int] | None:
    if not path:
        return None
    text = Path(path).read_text(encoding="utf-8").strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = text.replace(",", " ").split()
    if isinstance(value, dict):
        value = value.get("ids", [])
    return {int(item) for item in value}


def main() -> None:
    setup_stdout()
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", required=True)
    parser.add_argument("--store-dir", default=str(config.STORE_DIR))
    parser.add_argument("--out", required=True)
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--freeze-top", type=int, default=5)
    parser.add_argument("--model-top", type=int, default=15)
    args = parser.parse_args()

    rows = read_jsonl(args.retrieval)
    target_ids = _read_ids(args.ids_file)
    store = Store(Path(args.store_dir), cache_size=120)
    blob_cache: dict[tuple[str, int], str] = {}
    changed, top5_changed = [], []
    output = []
    for row in rows:
        qid = int(row["id"])
        if target_ids is not None and qid not in target_ids:
            output.append(row)
            continue
        candidates = row.get("candidates") or []
        by_ticker: dict[str, set[str]] = {}
        for candidate in candidates:
            by_ticker.setdefault(str(candidate.get("ticker") or ""), set()).add(
                str(candidate.get("report_id") or ""))
        metadata = {}
        for ticker, report_ids in by_ticker.items():
            if not ticker:
                continue
            for item in store.tables_of(ticker, sorted(report_ids)).to_dict("records"):
                metadata[(str(item["report_id"]), int(item["table_pos"]))] = item

        def blob_for(candidate: dict) -> str:
            key = (str(candidate.get("report_id") or ""),
                   int(candidate.get("table_pos") or 0))
            if key not in blob_cache:
                item = metadata.get(key) or {}
                blob_cache[key] = (
                    f"{item.get('context') or ''} {item.get('grid_json') or ''}"
                )
            return blob_cache[key]

        reranked = rerank_codegen_tail(
            row, blob_for, freeze_top=args.freeze_top, model_top=args.model_top)
        before = [(c["report_id"], int(c["table_pos"])) for c in candidates]
        after = [(c["report_id"], int(c["table_pos"]))
                 for c in reranked.get("candidates") or []]
        if before != after:
            changed.append(qid)
        if before[:args.freeze_top] != after[:args.freeze_top]:
            top5_changed.append(qid)
        output.append(reranked)

    if top5_changed:
        raise ValueError(f"frozen top-{args.freeze_top} changed: {top5_changed[:10]}")
    write_jsonl(args.out, output)
    print(f"reranked {len(changed)}/{len(rows)} rows; frozen top-{args.freeze_top} unchanged")
    print(f"changed ids={changed}")
    print(f"retrieval -> {args.out}")


if __name__ == "__main__":
    main()
