"""Build rollbackable V290: measured-union q98 plus physical q764 on V276."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from tempfile import mkdtemp

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_v276_q638_fix"
Q98_PAYLOAD = ROOT / "sub_top123_candidate_v225_q98_physical_parent_rollback_batch9"
Q764_PAYLOAD = ROOT / "sub_v287_scope3"
OUTPUT = ROOT / "sub_v290_scope2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def keyed(path: Path) -> dict[int, dict]:
    return {
        int(row["id"]): row
        for row in json.loads(path.read_text(encoding="utf-8-sig"))
    }


def verify_physical(table_ref: str, row: int, column: int, raw: str, masthead: str) -> dict:
    catalog = {
        item["table_ref"]: item
        for item in (
            json.loads(line)
            for line in (ROOT / "build" / "catalog.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    entry = catalog[table_ref]
    csv_path = ROOT / "build" / "tables" / entry["csv_path"]
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    actual = str(frame.iloc[row, column]).strip()
    if actual != raw:
        raise AssertionError(f"physical raw mismatch {table_ref}: {actual!r}")
    report_path = (
        ROOT / "data" / "financial_statements" / entry["ticker"]
        / str(entry["year"]) / entry["report_id"]
        / f"{entry['report_id']}_extracted.txt"
    )
    lines = report_path.read_text(encoding="utf-8").splitlines()
    context = " | ".join(
        line.strip()
        for line in lines[max(0, int(entry["line"]) - 16) : int(entry["line"])]
        if line.strip()
    )
    if masthead.casefold() not in context.casefold():
        raise AssertionError(f"physical masthead mismatch {table_ref}")
    return {
        "table_ref": table_ref,
        "catalog_scope": entry["scope"],
        "physical_masthead": masthead,
        "csv_path": str(csv_path.relative_to(ROOT)).replace("\\", "/"),
        "csv_sha256": sha(csv_path),
        "report_path": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "report_sha256": sha(report_path),
        "row": row,
        "column": column,
        "raw": actual,
    }


def build() -> dict:
    if not SOURCE.is_dir() or not Q98_PAYLOAD.is_dir() or not Q764_PAYLOAD.is_dir():
        raise FileNotFoundError("missing source or trusted payload candidate")
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite existing candidate: {OUTPUT}")
    physical = {
        "q98_selected_parent": verify_physical(
            "HUT_financial_statements_2024_consolidated|325", 12, 4,
            "146.469.679.444", "BẢNG CÂN ĐỐI KẾ TOÁN RIÊNG",
        ),
        "q98_retained_hard_negative": verify_physical(
            "HUT_financial_statements_2024_separate|328", 15, 4,
            "3.177.372.538.020", "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
        ),
        "q764_gvr_consolidated": verify_physical(
            "GVR_financial_statements_2015_consolidated|290", 11, 3,
            "6.437.295.628.830", "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
        ),
        "q764_dpm_consolidated": verify_physical(
            "DPM_financial_statements_2015_consolidated|1909", 28, 3,
            "3.498.666.363.829", "BẢNG CÂN ĐỐI KẾ TOÁN HỢP NHẤT",
        ),
    }
    temp = Path(mkdtemp(prefix="v290_scope2_", dir=str(ROOT)))
    try:
        staged = temp / OUTPUT.name
        shutil.copytree(SOURCE, staged)
        base_rows = keyed(SOURCE / "submission.json")
        staged_rows = keyed(staged / "submission.json")
        q98_rows = keyed(Q98_PAYLOAD / "submission.json")
        q764_rows = keyed(Q764_PAYLOAD / "submission.json")
        baseline_q714 = json.loads(json.dumps(base_rows[714], ensure_ascii=False))

        # Copy only the scored fields from the measured V225 q98 union. Query
        # and evidence remain the V276 bytes/semantics.
        for field in ("answer", "relevant_docs", "relevant_tables"):
            staged_rows[98][field] = json.loads(
                json.dumps(q98_rows[98][field], ensure_ascii=False)
            )
        # Copy only source-correct q764 score/table payload from V287. Docs are
        # explicitly preserved from V276.
        staged_rows[764]["answer"] = q764_rows[764]["answer"]
        staged_rows[764]["relevant_tables"] = json.loads(
            json.dumps(q764_rows[764]["relevant_tables"], ensure_ascii=False)
        )
        if staged_rows[764]["relevant_docs"] != base_rows[764]["relevant_docs"]:
            raise AssertionError("q764 relevant docs changed")
        for question_id in (98, 764):
            if staged_rows[question_id]["pandas_query"] != base_rows[question_id]["pandas_query"]:
                raise AssertionError(f"q{question_id} query changed")
            if staged_rows[question_id]["evidence"] != base_rows[question_id]["evidence"]:
                raise AssertionError(f"q{question_id} evidence binding changed")
        if staged_rows[714] != baseline_q714:
            raise AssertionError("q714 changed")
        (staged / "submission.json").write_text(
            json.dumps(list(staged_rows.values()), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        # Reuse the already measured/source-adjudicated compact payloads byte
        # for byte instead of reconstructing them.
        shutil.copy2(Q98_PAYLOAD / "data" / "q98_source_cells.csv", staged / "data" / "q98_source_cells.csv")
        shutil.copy2(Q764_PAYLOAD / "data" / "q764_source_cells.csv", staged / "data" / "q764_source_cells.csv")
        dpm_evidence = Q764_PAYLOAD / "data" / "DPM_financial_statements_2015_consolidated_1909.csv"
        shutil.copy2(dpm_evidence, staged / "data" / dpm_evidence.name)

        staged_audit = keyed(staged / "source_audit.json")
        base_audit = keyed(SOURCE / "source_audit.json")
        q98_audit = keyed(Q98_PAYLOAD / "source_audit.json")
        q764_audit = keyed(Q764_PAYLOAD / "source_audit.json")
        staged_audit[98] = json.loads(json.dumps(q98_audit[98], ensure_ascii=False))
        staged_audit[764] = json.loads(json.dumps(q764_audit[764], ensure_ascii=False))
        if staged_audit[714] != base_audit[714]:
            raise AssertionError("q714 source audit changed")
        (staged / "source_audit.json").write_text(
            json.dumps(
                sorted(staged_audit.values(), key=lambda item: int(item["id"])),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        candidate_audit = {
            "schema_version": "v290-scope2-candidate/v1",
            "candidate": OUTPUT.name,
            "baseline": SOURCE.name,
            "payload_sources": {
                "q98": Q98_PAYLOAD.name,
                "q764": Q764_PAYLOAD.name,
            },
            "answer_changes": [
                {"id": 98, "from": 3177.37, "to": 146.47},
                {"id": 764, "from": 2.99, "to": 2.94},
            ],
            "changed_submission_fields": {
                "98": ["answer", "relevant_docs", "relevant_tables"],
                "764": ["answer", "relevant_tables"],
            },
            "q98_measured_union": {
                "relevant_docs": staged_rows[98]["relevant_docs"],
                "relevant_tables": staged_rows[98]["relevant_tables"],
            },
            "q714_exactly_preserved": True,
            "queries_and_evidence_bindings_preserved": True,
            "q764_docs_preserved": True,
            "compact_manifest_payloads_reused": [98, 764],
            "physical_proof": physical,
            "packaged": False,
            "submitted": False,
            "automatic_promotion": False,
            "baseline_submission_sha256": sha(SOURCE / "submission.json"),
            "candidate_submission_sha256": sha(staged / "submission.json"),
        }
        (staged / "v290_scope2_audit.json").write_text(
            json.dumps(candidate_audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        shutil.move(str(staged), str(OUTPUT))
        temp.rmdir()
        return candidate_audit
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(build(), ensure_ascii=False, indent=2))
