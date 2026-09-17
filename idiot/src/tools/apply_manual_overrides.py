"""Apply audited manual answer/evidence overrides to a complete submission build."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-build", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--output-build", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()

    base_build = args.base_build.resolve()
    output_build = args.output_build.resolve()
    if output_build.exists():
        shutil.rmtree(output_build)
    shutil.copytree(base_build, output_build)

    submission_path = output_build / "submission.json"
    rows: list[dict[str, Any]] = json.loads(submission_path.read_text(encoding="utf-8"))
    by_id = {int(row["id"]): row for row in rows}
    overrides: list[dict[str, Any]] = json.loads(args.manual.read_text(encoding="utf-8"))

    changed: list[int] = []
    for override in overrides:
        qid = int(override["id"])
        if qid not in by_id:
            raise ValueError(f"override id is absent from base: {qid}")
        row = by_id[qid]
        variable = str(override.get("variable", "df1"))
        csv_path = f"data/q{qid:04d}_{variable}.csv"
        csv_rows = override.get("csv_rows")
        if not isinstance(csv_rows, list) or not csv_rows:
            raise ValueError(f"q{qid}: csv_rows must be a non-empty list")

        fieldnames: list[str] = []
        for csv_row in csv_rows:
            for key in csv_row:
                if key not in fieldnames:
                    fieldnames.append(key)
        evidence_path = output_build / csv_path
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        with evidence_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        row["answer"] = float(override["answer"])
        if "relevant_tables" in override:
            row["relevant_tables"] = list(override["relevant_tables"])
        if "relevant_docs" in override:
            row["relevant_docs"] = list(override["relevant_docs"])
        row["evidence"] = [{"variable": variable, "csv_path": csv_path}]
        row["pandas_query"] = str(override["pandas_query"])
        changed.append(qid)

    submission_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    output_zip = args.output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        for evidence_path in sorted((output_build / "data").rglob("*.csv")):
            archive.write(evidence_path, evidence_path.relative_to(output_build).as_posix())

    print(f"overrode ids={sorted(changed)} -> {output_zip}")


if __name__ == "__main__":
    main()
