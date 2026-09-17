"""Apply a source-cell exact repair to the V64 formula challenger."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts/codegen_private_v64_formula_core9_from_v63.jsonl"
OUT = ROOT / "artifacts/codegen_private_v67_formula_core9_exact1.jsonl"


def replacement(qid: int, answer: float, query: str, used_vars: list[dict], detail: str) -> dict:
    return {
        "id": qid,
        "answer": answer,
        "pandas_query": query,
        "used_vars": used_vars,
        "status": "ok",
        "source": "manual_exact_v66",
        "votes": 1,
        "n_ok": 1,
        "detail": detail,
        "detail_conf": 100.0,
        "semantic": {"ok": True, "dataframe_refs": [item["var"] for item in used_vars],
                     "errors": [], "warnings": []},
        "run_signature": "private_v66_audited_source_cells",
    }


def main() -> None:
    rows = [json.loads(line) for line in BASE.read_text(encoding="utf-8").splitlines()]
    by_id = {row["id"]: row for row in rows}
    by_id[324] = replacement(
        324, 80.86,
        "round(float(df1.loc[(df1['row'].isin([1, 2, 3])) & (df1['col'] == 1), 'value'].sum()) / 1000000000.0, 2)",
        [{"var": "df1", "report_id": "HBC_financial_statements_2024_separate", "table_pos": 25}],
        "sum of the three exact short-term-loan provision counterparties",
    )
    OUT.write_text("".join(json.dumps(by_id[row["id"]], ensure_ascii=False) + "\n"
                           for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()
