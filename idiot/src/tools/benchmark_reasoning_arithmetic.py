"""Independent local arithmetic check; never changes a release or submits it.

The model sees the question and input cells, but not the submitted answer,
expression or precomputed-answer columns. This audits computation only; a
matching result is not a claim of official answer accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from road2ai_vifinqa import local_llm, reasoning_llm
from road2ai_vifinqa.corpus import Corpus
from road2ai_vifinqa.expression_plan import inline_plan
from road2ai_vifinqa.submission import evaluate_expression
from road2ai_vifinqa.table_semantics import TableAnalyzer


def first_present(row, *names, default=None):
    """Mixed CSV row types share columns whose unused entries are NaN."""
    for name in names:
        value = row.get(name)
        if value is not None and not pd.isna(value) and value != "":
            return value
    return default


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default="427,429,430,431,432,433,434,435,436,437,438,439")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget", type=int, default=3072)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--baseline", type=Path, default=ROOT / "submission_vn53.zip")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "artifacts/models/qwen35_9b_manifest.json").read_text(encoding="utf-8"))
    model = Path(manifest["path"])
    if not manifest.get("verified") or model.stat().st_size != manifest["size"]:
        raise RuntimeError("Verified model missing")
    base_url = f"http://127.0.0.1:{args.port}"
    if local_llm.server_ready(base_url):
        raise RuntimeError("Port in use; refusing to share an unknown inference run")
    baseline_path = args.baseline.resolve(strict=True)
    signature = {"script": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 "model": manifest["sha256"], "budget": args.budget,
                 "baseline": str(baseline_path),
                 "baseline_sha256": hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
                 "compiler": hashlib.sha256((ROOT / "src/road2ai_vifinqa/expression_plan.py").read_bytes()).hexdigest(),
                 "client": hashlib.sha256((ROOT / "src/road2ai_vifinqa/reasoning_llm.py").read_bytes()).hexdigest()}
    result = {"schema": 1, "signature": signature, "rows": {}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result["signature"] != signature:
            raise RuntimeError("Configuration changed: use a fresh output")
    command = [str(local_llm.RUNTIME), "-m", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "-ngl", "auto", "-fitt", "700", "-c", "16384", "-ctk", "q8_0", "-ctv", "q8_0",
               "--parallel", "1", "--jinja", "--reasoning-format", "deepseek", "--reasoning", "on",
               "--reasoning-budget", str(args.budget), "--reasoning-budget-message", "Return the final JSON now."]
    with Corpus() as corpus, zipfile.ZipFile(baseline_path) as archive, args.output.with_suffix(".server.log").open("a", encoding="utf-8") as log:
        baseline = {int(r["id"]): r for r in json.loads(archive.read("submission.json"))}
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
                record = baseline[qid]
                rows = []
                for item in record["evidence"]:
                    original = pd.read_csv(io.BytesIO(archive.read(item["csv_path"])))
                    for index, row in original.iterrows():
                        source_id = str(first_present(row, "source_id", "candidate_id", default=f"s{len(rows)+1}"))
                        header = row.get("column_header", "")
                        if (
                            row.get("doc_id")
                            and pd.notna(row.get("table_id"))
                            and pd.notna(row.get("row_idx"))
                            and pd.notna(row.get("col_idx"))
                        ):
                            table = corpus.table(str(row["doc_id"]), int(row["table_id"]))
                            header = TableAnalyzer(table.rows, context=table.context).cell(int(row["row_idx"]), int(row["col_idx"])).column_header
                        rows.append({"source_id": source_id, "doc_id": row.get("doc_id", ""),
                                     "column_header": header,
                                     "label": first_present(row, "label", "row_label", default=""),
                                     "value": first_present(row, "value", "vnd_value", "raw_number"),
                                     "raw": first_present(row, "raw", "raw_value", default=""),
                                     "table_id": row.get("table_id"), "row_idx": row.get("row_idx"),
                                     "col_idx": row.get("col_idx")})
                frame = pd.DataFrame(rows)
                frame["ticker"] = frame["doc_id"].str.split("_").str[0]
                frame["year"] = pd.to_numeric(frame["doc_id"].str.extract(r"_(20\d\d)(?:_|$)", expand=False))
                payload = frame.to_json(orient="records", force_ascii=False)
                item = {"id": qid, "question": record["question"], "baseline": record["answer"]}
                if frame["value"].isna().any() or frame["source_id"].duplicated().any():
                    item["error"] = "incomplete_or_duplicate_source_cells"
                elif len(payload) > 33000:
                    item["error"] = "complete_input_too_large"
                else:
                    args.output.with_name(f"{args.output.stem}_q{qid:04d}_input.json").write_text(payload, encoding="utf-8")
                    try:
                        response = reasoning_llm.chat(
                            system=("Independently solve the financial question using the supplied source cells in DataFrame df. "
                                    "The numeric value column is normalized to VND for monetary inputs; raw preserves printed text. "
                                    "source_id uniquely names cells; ticker/year identify the reporting entity and year (a cell may be comparative). "
                                    "Check all filters, denominators, direction, ties, averages and requested units. "
                                    "If a needed measure is absent, explicitly list missing_inputs rather than invent it. "
                                    "Avoid repeating long expressions. Define short intermediate steps, in dependency order; "
                                    "later steps and the final expression can refer to earlier names. Each step is one expression, no assignment syntax. "
                                    "Return only JSON: {steps: [{name: string, expression: pandas expression}], expression: final numeric expression, "
                                    "used_source_ids: list, missing_inputs: list, explanation: brief calculation}. "
                                    "No imports, lambdas, assignments, file access or precomputed answer literals. "
                                    "Select input cells by source_id. Use bracket indexing for derived columns. "
                                    "Return a numeric scalar. Python/pandas compute all arithmetic; do not approximate numbers."),
                            user=record["question"] + "\nSOURCE CELLS:\n" + payload,
                            model=model.name, base_url=base_url, max_tokens=args.budget + 2800,
                            seed=20260827)
                        item.update(asdict(response))
                        selection = local_llm.extract_json(response.content)
                        item["selection"] = selection
                        if selection.get("missing_inputs"):
                            item["error"] = "missing_inputs"
                        else:
                            query = inline_plan(selection.get("steps", []), str(selection["expression"]), frames={"df"}, columns=set(frame.columns))
                            item["compiled_expression"] = query
                            ids = selection.get("used_source_ids", [])
                            if not ids or not set(ids).issubset(set(frame["source_id"])) or "df" not in query or "source_id" not in query:
                                raise ValueError("Ungrounded expression")
                            answer = evaluate_expression(query, {"df": frame})
                            item["answer"] = float(answer)
                            item["changed"] = abs(float(answer) - float(record["answer"])) > 1e-9 * max(1, abs(float(record["answer"])))
                    except Exception as exc:
                        item["error"] = f"{type(exc).__name__}: {exc}"
                result["rows"][str(qid)] = item
                temporary = args.output.with_suffix(".tmp.json")
                temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                temporary.replace(args.output)
                print(f"Q{qid}: changed={item.get('changed')} answer={item.get('answer')} error={item.get('error')}", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
