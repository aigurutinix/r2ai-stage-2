"""Report selected Easy source cells whose printed sign disagrees with the answer."""
from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with zipfile.ZipFile(args.archive) as archive:
        rows = {int(row["id"]): row for row in json.loads(archive.read("submission.json"))}
        findings = []
        for qid in range(1, 362):
            record = rows[qid]
            match = re.search(r"candidate_id[^=]*==\s*'([^']+)'", record["pandas_query"])
            if match is None:
                continue
            frame = pd.read_csv(io.BytesIO(archive.read(record["evidence"][0]["csv_path"])))
            selected = frame.loc[frame["candidate_id"] == match.group(1)]
            if len(selected) != 1:
                continue
            source = selected.iloc[0]
            raw = str(source.get("raw_value", ""))
            raw_number = source.get("raw_number")
            answer = float(record["answer"])
            source_negative = (
                ("(" in raw and ")" in raw)
                or (pd.notna(raw_number) and float(raw_number) < 0)
            )
            if source_negative != (answer < 0):
                findings.append({
                    "id": qid,
                    "question": record["question"],
                    "answer": answer,
                    "candidate_id": match.group(1),
                    "raw_value": raw,
                    "raw_number": None if pd.isna(raw_number) else float(raw_number),
                    "doc_id": source.get("doc_id"),
                    "table_id": int(source["table_id"]),
                    "row_idx": int(source["row_idx"]),
                    "col_idx": int(source["col_idx"]),
                })
    print(json.dumps({"count": len(findings), "findings": findings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
