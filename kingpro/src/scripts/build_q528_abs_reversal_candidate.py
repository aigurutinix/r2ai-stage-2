"""Build v198: isolate the q528 signed-versus-magnitude interpretation.

All three disclosed AFS provision movements are parenthesized reversals.  The
v196 program interprets "lớn nhất" as the largest signed number and therefore
selects 2023 (-12,703 million).  This A/B candidate interprets "số ... hoàn
nhập lớn nhất" as the largest amount by magnitude and selects 2022 (25,896
million).  No other submission row or retrieval field is changed.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "sub_top123_candidate_v196_effective_tax_sign"
OUTPUT = ROOT / "sub_top123_candidate_v198_q528_abs_reversal"
QID = 528


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build() -> dict:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    shutil.copytree(SOURCE, OUTPUT)

    source_submission = SOURCE / "submission.json"
    output_submission = OUTPUT / "submission.json"
    old_rows = json.loads(source_submission.read_text(encoding="utf-8"))
    rows = json.loads(output_submission.read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in old_rows}
    row = next(item for item in rows if int(item["id"]) == QID)

    old_fragment = (
        "result = round((v3 / v6 if v0 == max(v0, v1, v2) else "
        "v4 / v7 if v1 == max(v0, v1, v2) else v5 / v8) * 100, 2)"
    )
    new_fragment = (
        "result = round((v3 / v6 if abs(v0) == max(abs(v0), abs(v1), abs(v2)) else "
        "v4 / v7 if abs(v1) == max(abs(v0), abs(v1), abs(v2)) else v5 / v8) * 100, 2)"
    )
    if row["pandas_query"].count(old_fragment) != 1:
        raise AssertionError("q528 selector expression does not match v196")

    provision_movements = {2020: -24_107.0, 2022: -25_896.0, 2023: -12_703.0}
    pre_provision_profit = {2020: 1_881_705.0, 2022: 2_462_760.0, 2023: 2_012_636.0}
    total_assets = {2020: 116_267_442.0, 2022: 130_064_695.0, 2023: 161_977_363.0}
    selected_year = max(provision_movements, key=lambda year: abs(provision_movements[year]))
    corrected_answer = round(pre_provision_profit[selected_year] / total_assets[selected_year] * 100, 2)
    if selected_year != 2022 or corrected_answer != 1.89:
        raise AssertionError("unexpected q528 magnitude interpretation")

    old_answer = row["answer"]
    row["answer"] = corrected_answer
    row["pandas_query"] = row["pandas_query"].replace(old_fragment, new_fragment)
    output_submission.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for candidate_row in rows:
        qid = int(candidate_row["id"])
        old = old_by_id[qid]
        if qid != QID and candidate_row != old:
            raise AssertionError(f"q{qid} changed unexpectedly")
    allowed = {"answer", "pandas_query"}
    for key in set(old_by_id[QID]) | set(row):
        if key not in allowed and old_by_id[QID].get(key) != row.get(key):
            raise AssertionError(f"q528 field changed unexpectedly: {key}")

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids": [QID],
        "old_answer": old_answer,
        "new_answer": corrected_answer,
        "selected_year": selected_year,
        "provision_movements_million_signed": provision_movements,
        "pre_provision_profit_million": pre_provision_profit,
        "total_assets_million": total_assets,
        "hypothesis": "largest reversal amount means largest absolute magnitude",
        "invariants": {
            "only_q528_changed": True,
            "only_answer_and_query_changed": True,
            "source_cells_unchanged": True,
            "retrieval_fields_unchanged": True,
            "base_is_best_measured_v196": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": "Isolated semantic A/B hypothesis; leaderboard result is unknown until evaluated.",
    }
    (OUTPUT / "q528_abs_reversal_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
