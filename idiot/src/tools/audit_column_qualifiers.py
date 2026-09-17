"""List submitted source cells whose questions contain column qualifiers.

This is a read-only diagnostic.  It resolves each evidence coordinate against
the original table so row labels, hierarchical headers, and sections can be
reviewed together instead of trusting the flattened submission CSV.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from road2ai_vifinqa.corpus import Corpus
from road2ai_vifinqa.table_semantics import TableAnalyzer
from road2ai_vifinqa.text import fold_text


QUALIFIERS = (
    "trong han",
    "qua han",
    "bang vnd",
    "bang ngoai te",
    "gia tri hop ly",
    "gia goc",
    "tong cong",
    "chua phai lap du phong",
    "da lap du phong",
    "duoi 1 thang",
    "tren 1 nam",
    "cuoi ky",
    "dau ky",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--ids", help="Optional comma-separated question IDs")
    args = parser.parse_args()
    wanted = {int(value) for value in args.ids.split(",")} if args.ids else None

    with Corpus() as corpus, zipfile.ZipFile(args.archive) as archive:
        records = json.loads(archive.read("submission.json"))
        for record in records:
            qid = int(record["id"])
            folded = fold_text(record["question"])
            if qid < 578 or (wanted is not None and qid not in wanted):
                continue
            matched = [term for term in QUALIFIERS if term in folded]
            if wanted is None and not matched:
                continue
            print(f"Q{qid} [{', '.join(matched)}] {record['question']}")
            for evidence in record["evidence"]:
                frame = pd.read_csv(io.BytesIO(archive.read(evidence["csv_path"])))
                for _, row in frame.iterrows():
                    if pd.isna(row.get("doc_id")) or pd.isna(row.get("table_id")):
                        continue
                    doc_id = str(row["doc_id"])
                    table_id = int(row["table_id"])
                    row_idx = int(row["row_idx"])
                    col_idx = int(row["col_idx"])
                    table = corpus.table(doc_id, table_id)
                    semantic = TableAnalyzer(table.rows, context=table.context).cell(row_idx, col_idx)
                    print(
                        "  "
                        f"{doc_id}|{table_id} r{row_idx} c{col_idx} "
                        f"raw={table.rows[row_idx][col_idx]!r} "
                        f"row={semantic.row_label!r} header={semantic.column_header!r} "
                        f"section={semantic.section!r}"
                    )


if __name__ == "__main__":
    main()
