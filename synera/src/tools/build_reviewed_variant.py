"""Build a surgical variant from a reviewed source-coordinate plan.

Plans contain source coordinates and calculations, never expected answers.
The baseline is hash-locked, untouched rows retain their CSV bytes, and all
outputs are exclusive-created. Run release_audit.py before any upload.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from road2ai_vifinqa.corpus import Corpus, load_questions
from road2ai_vifinqa.submission import canonical_table_ref, evaluate_expression
from road2ai_vifinqa.table_semantics import TableAnalyzer
from road2ai_vifinqa.text import parse_vn_number


def digest(data):
    return hashlib.sha256(data).hexdigest()


def build(plan_path, output, manifest):
    if output.exists() or manifest.exists():
        raise FileExistsError("output or manifest already exists")
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    baseline = ROOT / plan["baseline"]
    baseline_bytes = baseline.read_bytes()
    if digest(baseline_bytes) != plan["baseline_sha256"]:
        raise ValueError("baseline hash mismatch")
    with zipfile.ZipFile(io.BytesIO(baseline_bytes)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("duplicate ZIP members")
        members = {name: archive.read(name) for name in archive.namelist()}
    original = json.loads(members["submission.json"])
    rows = {row["id"]: row for row in original}
    questions = {row["id"]: row["question"] for row in load_questions()}
    if len(rows) != 1012 or set(rows) != set(questions):
        raise ValueError("question coverage mismatch")
    edits, csv_members, changes = {}, {}, []
    with Corpus() as corpus:
        for edit in plan["edits"]:
            qid = int(edit["id"])
            if qid in edits or qid not in rows or rows[qid]["question"] != questions[qid]:
                raise ValueError(f"duplicate/invalid question: {qid}")
            records = []
            for index, spec in enumerate(edit["sources"], 1):
                table = corpus.table(spec["doc_id"], spec["table_id"])
                document = corpus.document(spec["doc_id"])
                raw = table.rows[spec["row_idx"]][spec["col_idx"]]
                number = parse_vn_number(raw)
                if number is None or not math.isfinite(number):
                    raise ValueError(f"source is not a finite number: {qid}, {spec}")
                semantics = TableAnalyzer(table.rows, context=table.context, report_year=document.report_year).cell(spec["row_idx"], spec["col_idx"])
                records.append({
                    **spec, "source_id": f"s{index}", "ticker": document.ticker,
                    "report_year": document.report_year, "raw_value": raw,
                    "value": number * spec["source_scale"],
                    "label": semantics.row_label, "column_header": semantics.column_header,
                    "table_context": table.context,
                })
            frame = pd.DataFrame(records)
            csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            query = edit["pandas_query"]
            if "df" not in query or "computed_answer" in query:
                raise ValueError("query must calculate from sources")
            answer = evaluate_expression(query, {"df": pd.read_csv(io.BytesIO(csv_bytes))})
            if not math.isclose(answer, evaluate_expression(query, {"df": frame}), rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError("CSV round-trip altered the result")
            csv_path = f"data/q{qid:04d}_review.csv"
            csv_members[csv_path] = csv_bytes
            edits[qid] = {
                "id": qid, "question": questions[qid], "answer": answer,
                "relevant_docs": list(dict.fromkeys(s["doc_id"] for s in records)),
                "relevant_tables": list(dict.fromkeys(canonical_table_ref(f"{s['doc_id']}|table_{s['table_id']}") for s in records)),
                "evidence": [{"variable": "df", "csv_path": csv_path}],
                "pandas_query": query,
            }
            changes.append({"id": qid, "before": rows[qid]["answer"], "after": answer,
                            "reason": edit["reason"], "sources": records, "csv_sha256": digest(csv_bytes)})
    payload = [edits.get(row["id"], row) for row in original]
    references = {item["csv_path"] for row in payload for item in row["evidence"]}
    result_members = {name: data for name, data in members.items() if name in references}
    result_members.update(csv_members)
    if set(result_members) != references:
        raise ValueError("missing or orphan evidence")
    result_members["submission.json"] = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(result_members, key=lambda n: (n != "submission.json", n)):
            archive.writestr(name, result_members[name])
    binary = buffer.getvalue()
    report = {"release": plan["release"], "baseline": str(baseline.resolve()),
              "baseline_sha256": digest(baseline_bytes), "archive": str(output.resolve()),
              "sha256": digest(binary), "plan_sha256": digest(plan_bytes),
              "changed_rows": sorted(edits), "numeric_changes": [c["id"] for c in changes if not math.isclose(c["before"], c["after"], rel_tol=1e-12, abs_tol=1e-9)],
              "changes": changes, "preserved_rows": 1012 - len(edits), "official_score": None}
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(binary)
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({key: value for key, value in report.items() if key != "changes"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    build(args.plan, args.output, args.manifest)
