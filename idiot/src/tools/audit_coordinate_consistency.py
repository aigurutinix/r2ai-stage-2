"""Find inconsistent base-unit values for identical source coordinates.

Question-specific answer units are excluded. Expense-sign conventions are
reported separately from magnitude/scale conflicts. This is read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import io
import json
import math
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    grouped = defaultdict(list)
    with zipfile.ZipFile(args.zip) as archive:
        records = json.loads(archive.read("submission.json"))
        for record in records:
            for evidence in record["evidence"]:
                for row in csv.DictReader(io.StringIO(archive.read(evidence["csv_path"]).decode("utf-8-sig"))):
                    if not all(row.get(key) not in (None, "") for key in ("doc_id", "table_id", "row_idx", "col_idx")):
                        continue
                    coordinate = (row["doc_id"], int(row["table_id"]), int(row["row_idx"]), int(row["col_idx"]))
                    base = row.get("vnd_value") or row.get("value")
                    if base in (None, "") and row.get("raw_number") not in (None, "") and row.get("source_scale") not in (None, ""):
                        base = float(row["raw_number"]) * float(row["source_scale"])
                    if base in (None, ""):
                        continue
                    value = float(base)
                    if not math.isfinite(value):
                        continue
                    grouped[coordinate].append({"qid": record["id"], "question": record["question"], "value": value,
                        "raw": row.get("raw_value", row.get("raw")), "source_scale": row.get("source_scale"),
                        "label": row.get("label") or row.get("row_label")})
    conflicts = []
    for coordinate, items in grouped.items():
        if len({row["qid"] for row in items}) < 2:
            continue
        magnitudes = sorted({abs(row["value"]) for row in items})
        if not math.isclose(magnitudes[0], magnitudes[-1], rel_tol=1e-10, abs_tol=1e-6):
            conflicts.append({"coordinate": coordinate, "items": items})
    report = {"archive": str(args.zip), "coordinates": len(grouped), "conflicts": conflicts}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"coordinates": len(grouped), "conflicts": len(conflicts)}))
    for conflict in conflicts:
        print(conflict["coordinate"], [(item["qid"], item["value"], item["source_scale"]) for item in conflict["items"]])


if __name__ == "__main__":
    main()
