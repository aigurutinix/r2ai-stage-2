"""Build v196 with the source-proven q993 deferred-tax sign repair.

The 2022 SJG income statement reports current tax expense of
201,952,237,413 VND and deferred tax expense of (8,056,358,883) VND.  The
reported profit-after-tax row reconciles only when the parenthesized deferred
amount keeps its negative sign.  v195 applied ``abs`` and overstated both tax
expense and the three-company mean effective tax rate.
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
SOURCE = ROOT / "sub_top123_candidate_v195_aggregate_contributor_order"
OUTPUT = ROOT / "sub_top123_candidate_v196_effective_tax_sign"
QID = 993


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def build() -> dict:
    marker = OUTPUT / "q993_effective_tax_rate_audit.json"
    if marker.exists():
        raise FileExistsError("refusing to overwrite completed candidate {}".format(OUTPUT))
    shutil.copytree(SOURCE, OUTPUT, dirs_exist_ok=False)

    source_submission = SOURCE / "submission.json"
    output_submission = OUTPUT / "submission.json"
    source_rows = json.loads(source_submission.read_text(encoding="utf-8"))
    rows = json.loads(output_submission.read_text(encoding="utf-8"))
    old_by_id = {int(row["id"]): row for row in source_rows}
    row = next(item for item in rows if int(item["id"]) == QID)

    old_fragment = (
        "v7 = abs(_btc_number(df1.iloc[7]['raw'], "
        "df1.iloc[7]['typed_factor']) * float(df1.iloc[7]['scale']))"
    )
    new_fragment = (
        "v7 = _btc_number(df1.iloc[7]['raw'], "
        "df1.iloc[7]['typed_factor']) * float(df1.iloc[7]['scale'])"
    )
    if row["pandas_query"].count(old_fragment) != 1:
        raise AssertionError("q993 deferred-tax expression does not match v195")
    row["pandas_query"] = row["pandas_query"].replace(old_fragment, new_fragment)

    source_values = {
        "gee_profit_before_tax": 912_546_851_205.0,
        "gee_total_tax_expense": 0.0,
        "vgc_profit_before_tax": 1_709_898_127_440.0,
        "vgc_current_tax_expense": 314_011_179_543.0,
        "vgc_deferred_tax_income": 2_116_228_119.0,
        "sjg_profit_before_tax": 1_414_526_135_994.0,
        "sjg_current_tax_expense": 201_952_237_413.0,
        "sjg_deferred_tax_expense_signed": -8_056_358_883.0,
        "sjg_profit_after_tax": 1_220_630_257_464.0,
    }
    sjg_reconciled_pat = (
        source_values["sjg_profit_before_tax"]
        - source_values["sjg_current_tax_expense"]
        - source_values["sjg_deferred_tax_expense_signed"]
    )
    if sjg_reconciled_pat != source_values["sjg_profit_after_tax"]:
        raise AssertionError("SJG tax sign does not reconcile to reported PAT")

    rates = {
        "GEE": 0.0,
        "VGC": (
            source_values["vgc_current_tax_expense"]
            - source_values["vgc_deferred_tax_income"]
        ) / source_values["vgc_profit_before_tax"],
        "SJG": (
            source_values["sjg_current_tax_expense"]
            + source_values["sjg_deferred_tax_expense_signed"]
        ) / source_values["sjg_profit_before_tax"],
    }
    corrected_answer = round(sum(rates.values()) / len(rates) * 100, 2)
    if corrected_answer != 10.65:
        raise AssertionError("unexpected corrected q993 answer {}".format(corrected_answer))
    old_answer = row["answer"]
    row["answer"] = corrected_answer
    output_submission.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    for candidate_row in rows:
        qid = int(candidate_row["id"])
        old = old_by_id[qid]
        if qid != QID and candidate_row != old:
            raise AssertionError("q{} changed unexpectedly".format(qid))
    old_q = old_by_id[QID]
    allowed = {"answer", "pandas_query"}
    for key in set(old_q) | set(row):
        if key not in allowed and old_q.get(key) != row.get(key):
            raise AssertionError("q993 field changed unexpectedly: {}".format(key))

    report = {
        "candidate": OUTPUT.name,
        "source_candidate": SOURCE.name,
        "changed_question_ids": [QID],
        "old_answer": old_answer,
        "new_answer": corrected_answer,
        "source_tables": {
            "GEE": "GEE_financial_statements_2022_separate|408",
            "VGC": "VGC_financial_statements_2022_separate|375",
            "SJG": "SJG_financial_statements_2022_separate|437",
        },
        "source_values": source_values,
        "rates_before_percent": rates,
        "sjg_pat_reconciliation": sjg_reconciled_pat,
        "repair": "preserve the signed SJG deferred-tax expense instead of applying abs()",
        "invariants": {
            "only_q993_changed": True,
            "only_answer_and_query_changed": True,
            "source_cells_unchanged": True,
            "relevant_docs_unchanged": True,
            "relevant_tables_unchanged": True,
            "leaderboard_feedback_not_used": True,
            "source_derived": True,
        },
        "source_submission_sha256": sha256(source_submission),
        "candidate_submission_sha256": sha256(output_submission),
        "claim_limit": "Source-derived hypothesis; BTC score is unknown until this exact artifact is evaluated.",
    }
    marker.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    build()
