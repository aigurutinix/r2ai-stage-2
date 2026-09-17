"""Read-only unit audit against explicit source-table declarations."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from road2ai_vifinqa.corpus import Corpus
from road2ai_vifinqa.source_units import continuation_scale, declared_scale
from road2ai_vifinqa.submission import evaluate_expression
from road2ai_vifinqa.text import fold_text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ROOT / "submission_vn53.zip")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    changes, errors = [], []
    checked = 0
    with Corpus() as corpus, zipfile.ZipFile(args.zip) as archive:
        for record in json.loads(archive.read("submission.json")):
            if int(record["id"]) > 361:
                continue
            question = fold_text(record["question"])
            if not any(unit in question for unit in ("dong", "vnd")):
                continue
            frames, modifications = {}, []
            for item in record["evidence"]:
                frame = pd.read_csv(io.BytesIO(archive.read(item["csv_path"])))
                frames[item["variable"]] = frame
                if not {"doc_id", "table_id", "raw_number", "source_scale", "requested_scale", "answer_value"} <= set(frame.columns):
                    continue
                for index, row in frame.iterrows():
                    if pd.isna(row["doc_id"]) or pd.isna(row["table_id"]):
                        continue
                    checked += 1
                    table = corpus.table(str(row["doc_id"]), int(row["table_id"]))
                    scale = declared_scale(table.context)
                    if scale is None:
                        scale = continuation_scale(corpus, table)
                    if scale is None or scale == float(row["source_scale"]):
                        continue
                    # This is a flag for source review, not an automatic patch:
                    # mixed-unit and percentage columns must still be reviewed.
                    old = float(row["answer_value"])
                    new = float(row["raw_number"]) * scale / float(row["requested_scale"])
                    frame.loc[index, "answer_value"] = new
                    frame.loc[index, "source_scale"] = scale
                    modifications.append({"row": int(index), "old": old, "new": new, "scale": scale,
                                          "coordinate": [str(row["doc_id"]), int(row["table_id"]), int(row["row_idx"]), int(row["col_idx"])],
                                          "label": row.get("row_label", ""), "unit_context": table.context[-450:]})
            if not modifications:
                continue
            try:
                value = float(evaluate_expression(record["pandas_query"], frames))
                if abs(value - float(record["answer"])) > 1e-9 * max(1, abs(float(record["answer"]))):
                    changes.append({"id": record["id"], "question": record["question"], "old": record["answer"], "new": value, "flags": modifications})
            except Exception as exc:
                errors.append({"id": record["id"], "error": str(exc)})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps({"checked_cells": checked, "changes": changes, "errors": errors}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"checked_cells": checked, "flags_affecting_answers": len(changes), "errors": len(errors)}))


if __name__ == "__main__":
    main()
