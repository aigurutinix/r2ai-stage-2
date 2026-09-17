"""Diagnostic local-model source review, never a release builder or uploader.

Only questions and original source-table excerpts are shown to the model.
Submitted answers, expressions and retrieval rationales are withheld. A model
flag is a review lead, not a corrected label or a reason to submit an answer.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from road2ai_vifinqa import local_llm, reasoning_llm
from road2ai_vifinqa.corpus import Corpus
from road2ai_vifinqa.table_semantics import TableAnalyzer


def source_excerpt(corpus, record, archive, *, full_tables=False):
    groups = {}
    for evidence in record["evidence"]:
        frame = pd.read_csv(io.BytesIO(archive.read(evidence["csv_path"])))
        for _, row in frame.iterrows():
            if pd.isna(row.get("doc_id")) or pd.isna(row.get("table_id")):
                raise ValueError("Evidence lacks source coordinates")
            key = (str(row["doc_id"]), int(row["table_id"]))
            groups.setdefault(key, set()).add((int(row["row_idx"]), int(row["col_idx"])))
    result = []
    for (doc, tid), cells in sorted(groups.items()):
        table = corpus.table(doc, tid)
        analyzer = TableAnalyzer(table.rows, context=table.context)
        selected = []
        indices = set(range(len(table.rows) if full_tables else min(4, len(table.rows))))
        for ri, ci in sorted(cells):
            semantics = analyzer.cell(ri, ci)
            selected.append({"row": ri, "column": ci, "raw": table.rows[ri][ci],
                             "label": semantics.row_label, "section": semantics.section,
                             "header": semantics.column_header})
            indices.update(range(max(0, ri - 1), min(len(table.rows), ri + 2)))
        result.append({"document": doc, "table": tid, "context": table.context,
                       "selected_cells": selected,
                       "rows": [{"index": i, "cells": table.rows[i]} for i in sorted(indices)]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, default=ROOT / "submission_vn53.zip")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--port", type=int, default=8096)
    parser.add_argument("--full-tables", action="store_true", help="Supply complete selected tables, still subject to the input-size guard")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "artifacts/models/qwen35_9b_manifest.json").read_text(encoding="utf-8"))
    model = Path(manifest["path"])
    if not manifest.get("verified") or model.stat().st_size != manifest["size"]:
        raise RuntimeError("Verified model missing")
    base_url = f"http://127.0.0.1:{args.port}"
    if local_llm.server_ready(base_url):
        raise RuntimeError("Port occupied")
    signature = {"script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "model": manifest["sha256"], "budget": args.budget,
                 "zip": hashlib.sha256(args.zip.read_bytes()).hexdigest(),
                 "full_tables": args.full_tables}
    result = {"signature": signature, "rows": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result["signature"] != signature:
            raise RuntimeError("Configuration changed; use a fresh output file")
    command = [str(local_llm.RUNTIME), "-m", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "-ngl", "auto", "-fitt", "700", "-c", "16384", "-ctk", "q8_0", "-ctv", "q8_0",
               "--parallel", "1", "--jinja", "--reasoning-format", "deepseek", "--reasoning", "on",
               "--reasoning-budget", str(args.budget), "--reasoning-budget-message", "Return the final JSON now."]
    with Corpus() as corpus, zipfile.ZipFile(args.zip) as archive, args.output.with_suffix(".server.log").open("a", encoding="utf-8") as log:
        records = {int(r["id"]): r for r in json.loads(archive.read("submission.json"))}
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 300
            while not local_llm.server_ready(base_url):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Server failed to start")
                time.sleep(1)
            for qid in map(int, args.ids.split(",")):
                if str(qid) in result["rows"]:
                    continue
                record = records[qid]
                item = {"id": qid, "question": record["question"]}
                try:
                    payload = json.dumps(source_excerpt(corpus, record, archive, full_tables=args.full_tables), ensure_ascii=False)
                    if len(payload) > 42000:
                        raise ValueError("Complete source input too large; no silent truncation")
                    args.output.with_name(f"{args.output.stem}_q{qid:04d}_input.json").write_text(payload, encoding="utf-8")
                    response = reasoning_llm.chat(
                        system=("Audit a Vietnamese financial question against the original report excerpts. "
                                "Selected cells are a candidate solution's operands, NOT guaranteed correct. "
                                "Check scope (parent/consolidated), requested years and opening/closing, "
                                "row vs column meaning, units, signs, selector vs target, and total vs component. "
                                "Read the raw row and surrounding cells, not just generated header or label. "
                                "Identify a specific contradiction only when the excerpt proves it. "
                                "If a source is absent or ambiguous say so; never invent missing numbers. "
                                "Independently describe the calculation and compute its diagnostic result if possible. "
                                "Return JSON only: {issues: [{document, table, row, column, reason}], "
                                "missing_inputs: [], calculation: string, approximate_answer: number or null, "
                                "answer_unit: string}. Do not generate code."),
                        user=record["question"] + "\nORIGINAL SOURCE EXCERPTS:\n" + payload,
                        model=model.name, base_url=base_url, max_tokens=args.budget + 1800,
                        seed=20260827)
                    item.update(asdict(response))
                    item["review"] = local_llm.extract_json(response.content)
                except Exception as exc:
                    item["error"] = f"{type(exc).__name__}: {exc}"
                result["rows"][str(qid)] = item
                temporary = args.output.with_suffix(".tmp.json")
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(args.output)
                print(f"Q{qid}: issues={len(item.get('review', {}).get('issues', []))} error={item.get('error')}", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
