"""Reproduce VN67 from the protected scored baseline and audited source repairs.

No hidden answer labels or model guesses are used. Only the allowlisted rows
are recomputed. All other JSON rows and CSV bytes are retained unchanged.
Outputs are exclusive-create: neither a release nor its manifest is overwritten.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
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
from road2ai_vifinqa.hard_note_solver import solve_note
from road2ai_vifinqa.panel import FinancialPanel
from road2ai_vifinqa.submission import canonical_table_ref, evaluate_expression
from road2ai_vifinqa.template_solver import TemplateSolver

BASE_SHA256 = "f30b3890ac00af812e39eef1cfe234ca4146f083020592bd8f5047524e67cd67"
NOTE_IDS = (495, 501, 502, 506, 521, 526)
PATCH_IDS = frozenset((*NOTE_IDS, 904))
NUMERIC_CHANGE_IDS = frozenset((495, 521, 904))
REASONS = {
    495: "Use related-party subset rather than all other short-term receivables in VGT 2021.",
    501: "Use gross overdue principal instead of allowance in QNS 2021; calculate tie-break from CSV.",
    502: "Use PLX 2021 fund bank-account balance rather than net fund balance.",
    506: "Use tangible fixed assets code 221 rather than all fixed assets code 220.",
    521: "Replace mislabeled HDG 2023 cash with explicit comparative in genuine 2024 statement.",
    526: "Use current-year 2023 EPS rather than its 2022 comparative.",
    904: "Replace mislabeled HDG 2023 financial income with explicit 2023 comparative.",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build(baseline: Path, output: Path, manifest: Path, panel_path: Path) -> dict:
    if output.exists() or manifest.exists():
        raise FileExistsError("release or manifest already exists; choose new paths")
    base_bytes = baseline.read_bytes()
    if sha256(base_bytes) != BASE_SHA256:
        raise ValueError("protected baseline hash mismatch")
    with zipfile.ZipFile(io.BytesIO(base_bytes)) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError("duplicate baseline members")
        members = {name: archive.read(name) for name in archive.namelist()}
    rows = json.loads(members["submission.json"])
    old_rows = {row["id"]: row for row in rows}
    questions = {row["id"]: row["question"] for row in load_questions()}
    if len(rows) != 1012 or set(old_rows) != set(questions):
        raise ValueError("baseline question coverage mismatch")
    replacements, new_members, changes, numeric_changes = {}, {}, [], []
    with Corpus() as corpus:
        panel = FinancialPanel(panel_path)
        template = TemplateSolver(corpus, panel)
        for qid in sorted(PATCH_IDS):
            old = old_rows[qid]
            if old["question"] != questions[qid]:
                raise ValueError(f"question mismatch: {qid}")
            result = (
                solve_note(questions[qid], qid, corpus, panel=panel)
                if qid in NOTE_IDS else template.solve(questions[qid], question_id=qid)
            )
            if result is None:
                raise ValueError(f"solver abstained: {qid}")
            records = [asdict(source) for source in result.sources]
            for record in records:
                table = corpus.table(record["doc_id"], record["table_id"])
                raw = table.rows[record["row_idx"]][record["col_idx"]]
                if record["raw_value"] != raw:
                    raise ValueError(f"raw source mismatch: {qid}")
            expression = (
                result.pandas_query if qid in NOTE_IDS
                else "float(df.set_index('year')['value'].idxmax())"
            )
            frame = pd.DataFrame(records)
            if any("computed_answer" in column for column in frame.columns):
                raise ValueError("new evidence must not store a precomputed answer")
            csv_bytes = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
            replayed = evaluate_expression(expression, {"df": pd.read_csv(io.BytesIO(csv_bytes))})
            if not math.isclose(replayed, result.answer, rel_tol=1e-12, abs_tol=1e-9):
                raise ValueError(f"fresh CSV execution mismatch: {qid}")
            csv_path = f"data/q{qid:04d}_df.csv"
            new_members[csv_path] = csv_bytes
            new = {
                "id": qid, "question": questions[qid], "answer": result.answer,
                "relevant_docs": list(dict.fromkeys(s["doc_id"] for s in records)),
                "relevant_tables": list(dict.fromkeys(
                    canonical_table_ref(f"{s['doc_id']}|table_{s['table_id']}") for s in records
                )),
                "evidence": [{"variable": "df", "csv_path": csv_path}],
                "pandas_query": expression,
            }
            changed_number = not math.isclose(old["answer"], new["answer"], rel_tol=1e-12, abs_tol=1e-9)
            if changed_number:
                numeric_changes.append(qid)
            replacements[qid] = new
            changes.append({
                "id": qid, "before": old["answer"], "after": new["answer"],
                "numeric_change": changed_number, "reason": REASONS[qid],
                "query": expression, "sources": records, "csv_sha256": sha256(csv_bytes),
            })
    if set(numeric_changes) != NUMERIC_CHANGE_IDS:
        raise ValueError(f"unexpected answer-change set: {numeric_changes}")
    payload = [replacements.get(row["id"], row) for row in rows]
    for row in payload:
        if row["id"] not in PATCH_IDS and row != old_rows[row["id"]]:
            raise ValueError("unapproved baseline row changed")
    referenced = {item["csv_path"] for row in payload for item in row["evidence"]}
    # Omit only evidence replaced in this new ZIP; the original ZIP is intact.
    release_members = {name: data for name, data in members.items() if name in referenced}
    release_members.update(new_members)
    if set(release_members) != referenced:
        raise ValueError("missing or unreferenced evidence")
    release_members["submission.json"] = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(release_members, key=lambda n: (n != "submission.json", n)):
            info = zipfile.ZipInfo(name, date_time=(2026, 8, 27, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, release_members[name])
    release_bytes = memory.getvalue()
    report = {
        "release": "VN67", "created_utc": datetime.now(timezone.utc).isoformat(),
        "baseline": str(baseline.resolve()), "baseline_sha256": BASE_SHA256,
        "archive": str(output.resolve()), "sha256": sha256(release_bytes),
        "panel": str(panel_path.resolve()), "panel_sha256": sha256(panel_path.read_bytes()),
        "rows": len(payload), "changed_rows": sorted(PATCH_IDS),
        "numeric_changes": numeric_changes, "preserved_rows": len(rows) - len(PATCH_IDS),
        "changes": changes, "official_score": None,
        "qualification": "Source and execution validation are not hidden-answer accuracy measurements.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(release_bytes)
    with manifest.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return {key: value for key, value in report.items() if key != "changes"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=ROOT / "submission_vn53.zip")
    parser.add_argument("--output", type=Path, default=ROOT / "submission_vn67.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/reasoning_selector/vn67_manifest.json")
    parser.add_argument("--panel", type=Path, default=ROOT / "artifacts/financial_panel_comparatives_v6.json")
    args = parser.parse_args()
    print(json.dumps(build(args.baseline, args.output, args.manifest, args.panel), ensure_ascii=False, indent=2))
