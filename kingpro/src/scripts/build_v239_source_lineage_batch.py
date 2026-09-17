"""Build v239: q24 answer fix plus four source-lineage corrections."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v228_q24_counterparty_fix"
BASELINE = ROOT / "sub_top123_candidate_v217_missing_panel_operand_batch3"
OUTPUT = ROOT / "sub_top123_candidate_v239_source_lineage_batch5"
BUILDING = OUTPUT.with_name(OUTPUT.name + ".building")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def by_id(rows: list[dict]) -> dict[int, dict]:
    return {int(row["id"]): row for row in rows}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def replace_table(values: list[str], old: str, new: str) -> list[str]:
    if old not in values:
        raise AssertionError(f"missing table {old}")
    return [new if value == old else value for value in values]


def assert_physical(document: str, line: int, raw: str) -> str:
    source_dir = ROOT / "build/tables" / document
    paths = list(source_dir.glob(f"*line{line}.csv"))
    if len(paths) != 1 or raw not in paths[0].read_text(encoding="utf-8-sig"):
        raise AssertionError(f"physical proof failed: {document}|{line} {raw}")
    return paths[0].name


def audit_sources_from_manifest(rows: list[dict[str, str]]) -> list[dict]:
    return [
        {
            "table_ref": row["source_table"],
            "csv": row["source_csv"],
            "row": int(row["row_idx"]),
            "column": int(row["col_idx"]),
            "metric": row["metric_key"],
            "scale": float(row["scale"]),
            "typed_factor": float(row["typed_factor"]),
            "raw": row["raw"],
        }
        for row in rows
    ]


def main() -> None:
    if not SOURCE.is_dir() or not BASELINE.is_dir():
        raise FileNotFoundError("v228 source or v217 baseline missing")
    if OUTPUT.exists() or BUILDING.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT} or {BUILDING}")

    rows = load(SOURCE / "submission.json")
    baseline_rows = load(BASELINE / "submission.json")
    audits = load(SOURCE / "source_audit.json")
    if not all(isinstance(value, list) for value in (rows, baseline_rows, audits)):
        raise AssertionError("submission/source_audit payload must be lists")
    staged_rows = json.loads(json.dumps(rows, ensure_ascii=False))
    staged = by_id(staged_rows)

    proofs = {
        61: assert_physical("DXS_financial_statements_2023_consolidated", 389, "1.997.404.377.548"),
        826: assert_physical("KBC_financial_statements_2016_consolidated", 1351, "790.694.785.484"),
        861: [
            assert_physical("VNM_financial_statements_2015_consolidated", 1371, "1.200.139.398"),
            assert_physical("HNG_financial_statements_2015_consolidated", 1551, "708.143.895"),
        ],
        709: assert_physical("MSR_financial_statements_2022_consolidated", 357, "1.194.553.796"),
    }

    staged[826]["relevant_tables"] = replace_table(
        staged[826]["relevant_tables"],
        "KBC_financial_statements_2016_consolidated|1357",
        "KBC_financial_statements_2016_consolidated|1351",
    )
    staged[861]["relevant_tables"] = replace_table(
        staged[861]["relevant_tables"],
        "VNM_financial_statements_2015_consolidated|1377",
        "VNM_financial_statements_2015_consolidated|1371",
    )
    staged[861]["relevant_tables"] = replace_table(
        staged[861]["relevant_tables"],
        "HNG_financial_statements_2015_consolidated|269",
        "HNG_financial_statements_2015_consolidated|1551",
    )
    if "v2 / 10" not in staged[861]["pandas_query"]:
        raise AssertionError("unexpected q861 query")
    staged[861]["pandas_query"] = staged[861]["pandas_query"].replace("v2 / 10", "v2")
    staged[709]["relevant_tables"] = replace_table(
        staged[709]["relevant_tables"],
        "MSR_financial_statements_2022_consolidated|399",
        "MSR_financial_statements_2022_consolidated|357",
    )

    shutil.copytree(SOURCE, BUILDING)
    write_json(BUILDING / "submission.json", staged_rows)

    manifests: dict[int, list[dict[str, str]]] = {}
    for qid in (61, 826, 861, 709):
        path = BUILDING / "data" / f"q{qid}_source_cells.csv"
        fields, source_rows = read_csv(path)
        manifests[qid] = source_rows
        if qid == 61:
            source_rows[0].update(
                metric_key="kqkd:01",
                source_csv="table_7_line389.csv",
                row_idx="1",
                col_idx="3",
            )
        elif qid == 826:
            source_rows[0].update(
                raw="790.694.785.484",
                source_table="KBC_financial_statements_2016_consolidated|1351",
                source_csv="table_45_line1351.csv",
                row_idx="2",
                col_idx="1",
            )
        elif qid == 861:
            source_rows[1].update(
                metric_key="note:ending_common_shares_outstanding",
                source_table="VNM_financial_statements_2015_consolidated|1371",
                source_csv="table_50_line1371.csv",
                row_idx="8",
                col_idx="1",
            )
            source_rows[2].update(
                metric_key="note:ending_common_shares_outstanding",
                raw="708.143.895",
                source_table="HNG_financial_statements_2015_consolidated|1551",
                source_csv="table_53_line1551.csv",
                row_idx="7",
                col_idx="1",
            )
        elif qid == 709:
            source_rows[0].update(
                metric_key="kqkd:23",
                source_table="MSR_financial_statements_2022_consolidated|357",
                source_csv="table_3_line357.csv",
                row_idx="8",
                col_idx="3",
            )
        write_csv(path, fields, source_rows)

    staged_audits = json.loads(json.dumps(audits, ensure_ascii=False))
    audit_map = by_id(staged_audits)
    notes = {
        61: "Intent/source repair: gross-sales code01 replaces answer-neutral net code10 lineage.",
        826: "Source correction: 2016 leasing COGS component now comes from the same COGS decomposition as its denominator.",
        861: "Direct-metric repair: VNM/HNG outstanding-share rows replace indirect balance/par-value inference.",
        709: "Primary-statement repair: KQKD code23 replaces equal-valued cash-flow adjustment lineage.",
    }
    for qid, source_rows in manifests.items():
        audit = audit_map[qid]
        audit["old_answer"] = audit.get("answer")
        audit["answer"] = staged[qid]["answer"]
        audit["note"] = notes[qid]
        audit["sources"] = audit_sources_from_manifest(source_rows)
    write_json(BUILDING / "source_audit.json", staged_audits)

    baseline_map = by_id(baseline_rows)
    changed_submission_ids = [
        int(row["id"])
        for row in staged_rows
        if row != baseline_map[int(row["id"])]
    ]
    if changed_submission_ids != [24, 709, 826, 861]:
        raise AssertionError(f"unexpected submission changes: {changed_submission_ids}")
    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_submission_ids_relative_to_v217": changed_submission_ids,
        "changed_evidence_ids_relative_to_v217": [24, 61, 709, 826, 861],
        "answer_changes": {"24": {"from": 2083992733.0, "to": 1957824733.0}},
        "answer_neutral_source_fixes": [61, 709, 826, 861],
        "physical_proofs": proofs,
        "source_submission_sha256": sha256(SOURCE / "submission.json"),
        "baseline_submission_sha256": sha256(BASELINE / "submission.json"),
        "claim_limit": "One answer fix plus four source-lineage fixes; leaderboard effect unmeasured.",
    }
    write_json(BUILDING / "v239_source_lineage_batch5_audit.json", report)
    BUILDING.rename(OUTPUT)
    report["candidate_submission_sha256"] = sha256(OUTPUT / "submission.json")
    write_json(OUTPUT / "v239_source_lineage_batch5_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
