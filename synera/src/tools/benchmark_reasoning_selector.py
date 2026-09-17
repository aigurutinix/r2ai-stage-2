"""Read-only source-selection experiment with bounded local Qwen reasoning.

Uses only official question text and source tables as model input. Audit labels
are used afterwards for a local diagnostic, not placed into the prompt. This
does not modify any submission or connect to the competition server.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road2ai_vifinqa import local_llm, reasoning_llm  # noqa: E402
from road2ai_vifinqa.corpus import Corpus, load_questions  # noqa: E402
from road2ai_vifinqa.easy_solver import build_easy_candidates, shortlist_easy_candidates  # noqa: E402
from road2ai_vifinqa.retrieval import retrieve_rows  # noqa: E402


STOPWORDS = {
    "bao", "nhiêu", "của", "công", "ty", "năm", "là", "vào", "tại", "theo",
    "được", "cho", "và", "trong", "đồng", "triệu", "tỷ", "vnd", "cuối", "đầu",
}


def semantic_tokens(text: str) -> set[str]:
    return {
        token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in STOPWORDS and not token.isdigit()
    }


def few_shot_examples(qid: int, question: str, audits: dict[int, dict], count: int) -> list[dict[str, object]]:
    target = semantic_tokens(question)
    ranked = []
    for other_id, row in audits.items():
        if other_id == qid:
            continue
        tokens = semantic_tokens(str(row["question"]))
        union = target | tokens
        score = len(target & tokens) / len(union) if union else 0.0
        ranked.append((score, other_id, row))
    examples = []
    for _, other_id, row in sorted(ranked, key=lambda item: (-item[0], item[1]))[:count]:
        gold = row["gold"]
        examples.append({
            "question": row["question"],
            "correct_source": {
                "document": gold["coordinate"][0],
                "table": gold["coordinate"][1],
                "row_label": gold["row_label"],
                "table_context": gold["table_context"],
                "column_header": gold["column_header"],
                "raw_value": gold["raw_value"],
            },
        })
    return examples


def source_payload(corpus, candidates):
    tables = {}
    cells = []
    for cell in candidates:
        key = f"{cell.doc_id}|{cell.table_id}"
        table = corpus.table(cell.doc_id, cell.table_id)
        if key not in tables:
            tables[key] = {"context": table.context, "rows": {}}
        indices = set(range(min(4, len(table.rows)))) | set(range(max(0, cell.row_idx - 1), min(len(table.rows), cell.row_idx + 2)))
        for index in sorted(indices):
            tables[key]["rows"][str(index)] = table.rows[index]
        cells.append({
            "id": cell.candidate_id, "table": key, "row": cell.row_idx,
            "column": cell.col_idx, "label": cell.row_label,
            "section": cell.section, "header": cell.column_header,
            "raw": cell.raw_value, "normalized_answer": cell.answer_value,
            "source_scale": cell.source_scale, "requested_scale": cell.requested_scale,
        })
    return {"candidate_cells": cells, "source_tables": tables}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ids", default="17,120,221,124,229,240")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/reasoning_selector/controls.json")
    parser.add_argument("--model", choices=("8b", "14b", "qwen35-9b"), default="14b")
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--budget", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--few-shot", type=int, default=0)
    parser.add_argument("--max-new", type=int, default=0, help="Stop cleanly after this many new items (0=all)")
    args = parser.parse_args()
    model_dir = "Qwen3-8B-GGUF" if args.model == "8b" else "Qwen3-14B-GGUF-Q4"
    model_name = "Qwen3-8B-Q4_K_M.gguf" if args.model == "8b" else "Qwen3-14B-Q4_K_M.gguf"
    model = ROOT / "artifacts/models" / model_dir / model_name
    if args.model == "qwen35-9b":
        manifest = json.loads((ROOT / "artifacts/models/qwen35_9b_manifest.json").read_text(encoding="utf-8"))
        model = Path(manifest["path"])
        model_name = model.name
        if not manifest.get("verified") or not model.exists() or model.stat().st_size != manifest["size"]:
            raise RuntimeError("Verified Qwen3.5 model is not available")
    base_url = f"http://127.0.0.1:{args.port}"
    if local_llm.server_ready(base_url):
        raise RuntimeError("Audit port already in use; refusing to use an unknown server")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    signature = {
        "model": str(model), "model_size": model.stat().st_size,
        "model_mtime_ns": model.stat().st_mtime_ns, "seed": args.seed,
        "thinking_budget": args.budget,
        "few_shot": args.few_shot,
        "client_sha256": hashlib.sha256((ROOT / "src/road2ai_vifinqa/reasoning_llm.py").read_bytes()).hexdigest(),
        "candidates_sha256": hashlib.sha256((ROOT / "src/road2ai_vifinqa/easy_solver.py").read_bytes()).hexdigest(),
        "units_sha256": hashlib.sha256((ROOT / "src/road2ai_vifinqa/source_units.py").read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result = {"schema": 1, "model": str(model), "seed": args.seed, "thinking_budget": args.budget, "signature": signature, "rows": {}}
    if args.output.exists():
        result = json.loads(args.output.read_text(encoding="utf-8"))
        if result.get("signature") != signature:
            raise RuntimeError("Cached run has a different model/configuration; choose a fresh output path")
    questions = {int(row["id"]): str(row["question"]) for row in load_questions()}
    with zipfile.ZipFile(ROOT / "submission_vn53.zip") as archive:
        baseline = {int(row["id"]): row for row in json.loads(archive.read("submission.json"))}
    audit_path = ROOT / "runs/agent_easy_selector_audit/snapshot_diagnostic.json"
    audits = {int(row["id"]): row for row in json.loads(audit_path.read_text(encoding="utf-8"))["details"]}
    command = [str(local_llm.RUNTIME), "-m", str(model), "--host", "127.0.0.1", "--port", str(args.port),
               "-ngl", "auto", "-fitt", "700", "-c", "16384", "-ctk", "q8_0", "-ctv", "q8_0",
               "--parallel", "1", "--jinja", "--reasoning-format", "deepseek", "--reasoning", "on",
               "--reasoning-budget", str(args.budget), "--reasoning-budget-message", "Finish the analysis and give the final JSON now."]
    with args.output.with_suffix(".server.log").open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            deadline = time.monotonic() + 300
            while not local_llm.server_ready(base_url):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Local model startup failed; inspect server log")
                time.sleep(1)
            with Corpus() as corpus:
                completed = 0
                for qid in map(int, args.ids.split(",")):
                    if str(qid) in result["rows"]:
                        print(f"Q{qid} cached", flush=True)
                        continue
                    candidates = build_easy_candidates(corpus, questions[qid])
                    hits = retrieve_rows(corpus, questions[qid], limit=100_000, include_prior=False)
                    scores = {(hit.row.doc_id, hit.row.table_id, hit.row.row_idx): float(hit.score) for hit in hits}
                    shortlist = shortlist_easy_candidates(candidates, question=questions[qid], bm25_row_scores=scores, use_learned_reranker=True)
                    shortlist = shortlist[:48]
                    source = source_payload(corpus, shortlist)
                    while len(json.dumps(source, ensure_ascii=False)) > 22000 and len(shortlist) > 1:
                        shortlist.pop()
                        source = source_payload(corpus, shortlist)
                    prompt = "Question: " + questions[qid] + "\n" + json.dumps(source, ensure_ascii=False, separators=(",", ":"))
                    if args.few_shot:
                        examples = few_shot_examples(qid, questions[qid], audits, args.few_shot)
                        prompt = (
                            "Audited examples from other questions (learn the mapping pattern; never copy their values):\n"
                            + json.dumps(examples, ensure_ascii=False, separators=(",", ":"))
                            + "\nTarget " + prompt
                        )
                    # Save only source context and final answer; no hidden labels in inputs.
                    args.output.with_name(f"{args.output.stem}_q{qid:04d}_input.json").write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding="utf-8")
                    record = {"id": qid, "question": questions[qid], "baseline": baseline[qid]["answer"], "candidates": len(shortlist)}
                    if qid in audits:
                        audit_coordinate = tuple(audits[qid]["gold"]["coordinate"])
                        record["local_audit_in_prompt"] = any(
                            (cell.doc_id, cell.table_id, cell.row_idx, cell.col_idx) == audit_coordinate
                            for cell in shortlist
                        )
                    try:
                        completion = reasoning_llm.chat(
                            system=("Select the source cell answering this Vietnamese financial-report question. "
                                    "Check the full metric, entity, scope, period, column, total versus component and unit. "
                                    "Rows and columns use zero-based indices. Headers and surrounding rows are source evidence. "
                                    "Never assume that the first candidate is correct. Identify ambiguity if no cell matches exactly. "
                                    "Final response must be one JSON object with keys selected_id (candidate id or null), "
                                    "confidence (0..1), evidence (brief explanation of metric AND header match), ambiguous (boolean)."),
                            user=prompt, model=model_name, base_url=base_url, max_tokens=args.budget + 1024, seed=args.seed,
                        )
                        record.update(asdict(completion))
                        selection = local_llm.extract_json(completion.content)
                        record["selection"] = selection
                        chosen = next((c for c in shortlist if c.candidate_id == selection.get("selected_id")), None)
                        if chosen is not None:
                            record["selected"] = asdict(chosen)
                            record["changed"] = abs(float(chosen.answer_value) - float(record["baseline"])) > 1e-9 * max(1, abs(float(record["baseline"])))
                            if qid in audits:
                                record["local_audit_answer_ok"] = abs(float(chosen.answer_value) - float(audits[qid]["gold_answer"])) <= 1e-8 * max(1, abs(float(audits[qid]["gold_answer"])))
                    except Exception as exc:
                        record["error"] = f"{type(exc).__name__}: {exc}"
                    result["rows"][str(qid)] = record
                    temporary = args.output.with_suffix(".tmp.json")
                    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                    temporary.replace(args.output)
                    print(f"Q{qid}: changed={record.get('changed')} local_audit_ok={record.get('local_audit_answer_ok')} seconds={record.get('elapsed_seconds')} error={record.get('error')}", flush=True)
                    completed += 1
                    if args.max_new and completed >= args.max_new:
                        break
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


if __name__ == "__main__":
    main()
