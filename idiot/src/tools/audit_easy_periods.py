"""Flag explicit period conflicts in submitted direct-value source cells."""
from __future__ import annotations
import argparse
import io
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from road2ai_vifinqa.corpus import Corpus
from road2ai_vifinqa.table_semantics import TableAnalyzer
from road2ai_vifinqa.text import fold_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    flags = []
    with Corpus() as corpus, zipfile.ZipFile(ROOT / "submission_vn53.zip") as archive:
        for record in json.loads(archive.read("submission.json")):
            if record["id"] > 361:
                continue
            question = str(record["question"])
            years = corpus.infer_years(question)
            if len(years) != 1:
                continue
            year = years[0]
            opening = any(t in fold_text(question) for t in ("dau nam", "dau ky"))
            for evidence in record["evidence"]:
                frame = pd.read_csv(io.BytesIO(archive.read(evidence["csv_path"])))
                if not {"doc_id", "table_id", "row_idx", "col_idx"} <= set(frame):
                    continue
                for _, row in frame.iterrows():
                    if pd.isna(row["doc_id"]):
                        continue
                    doc = corpus._doc_by_id[str(row["doc_id"])]
                    table = corpus.table(doc.doc_id, int(row["table_id"]))
                    analyzer = TableAnalyzer(table.rows, context=table.context, report_year=doc.report_year)
                    sem = analyzer.cell(int(row["row_idx"]), int(row["col_idx"]))
                    wrong = sem.period.explicit and (
                        (sem.period.year is not None and sem.period.year != year and not opening)
                        or (sem.period.role.value == "prior" and year == doc.report_year and not opening)
                        or (sem.period.role.value == "current" and year == doc.report_year and opening)
                    )
                    if wrong:
                        flags.append({"id": record["id"], "question": question, "answer": record["answer"],
                                      "coordinate": [doc.doc_id, table.table_id, int(row["row_idx"]), int(row["col_idx"])],
                                      "header": sem.column_header, "period_year": sem.period.year, "period_role": sem.period.role.value,
                                      "rows": table.rows[:5], "selected_row": table.rows[int(row["row_idx"])]})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"flags": flags}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"flags": len(flags), "ids": sorted({r['id'] for r in flags})}))


if __name__ == "__main__":
    main()
