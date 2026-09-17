"""Merge independently audited ViFinQA build directories into one ZIP.

The first build supplies the complete public set.  Later builds replace rows
with the same question id, including their referenced evidence CSV files.
"""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any


def load_build(path: Path) -> tuple[list[dict[str, Any]], Path]:
    submission_path = path / "submission.json"
    data_path = path / "data"
    if not submission_path.is_file() or not data_path.is_dir():
        raise FileNotFoundError(f"invalid build directory: {path}")
    payload = json.loads(submission_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"submission is not a list: {submission_path}")
    return payload, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, action="append", default=[])
    parser.add_argument("--output-build", type=Path, required=True)
    parser.add_argument("--output-zip", type=Path, required=True)
    args = parser.parse_args()

    rows, base_source = load_build(args.base.resolve())
    chosen: dict[int, tuple[dict[str, Any], Path]] = {}
    for row in rows:
        qid = int(row["id"])
        if qid in chosen:
            raise ValueError(f"duplicate base id: {qid}")
        chosen[qid] = (row, base_source)

    replaced: list[int] = []
    for overlay_arg in args.overlay:
        overlay_rows, overlay_source = load_build(overlay_arg.resolve())
        for row in overlay_rows:
            qid = int(row["id"])
            if qid not in chosen:
                raise ValueError(f"overlay id is absent from base: {qid}")
            chosen[qid] = (row, overlay_source)
            replaced.append(qid)

    expected = list(range(1, 1013))
    actual = sorted(chosen)
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"base is not the full public set; missing={missing}, extra={extra}")

    output_build = args.output_build.resolve()
    if output_build.exists():
        shutil.rmtree(output_build)
    (output_build / "data").mkdir(parents=True)

    merged: list[dict[str, Any]] = []
    copied: set[str] = set()
    for qid in expected:
        row, source = chosen[qid]
        merged.append(row)
        evidence = row.get("evidence")
        if not isinstance(evidence, list) or not evidence:
            raise ValueError(f"q{qid}: missing evidence")
        for item in evidence:
            rel = str(item["csv_path"]).replace("\\", "/")
            if not rel.startswith("data/") or rel in {"data/", "data"}:
                raise ValueError(f"q{qid}: invalid csv_path {rel!r}")
            src = source / Path(rel)
            dst = output_build / Path(rel)
            if not src.is_file():
                raise FileNotFoundError(f"q{qid}: missing source CSV {src}")
            if rel in copied:
                raise ValueError(f"evidence path reused by multiple rows: {rel}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.add(rel)

    submission_path = output_build / "submission.json"
    submission_path.write_text(
        json.dumps(merged, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    output_zip = args.output_zip.resolve()
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        for rel in sorted(copied):
            archive.write(output_build / Path(rel), rel)

    print(
        f"merged {len(merged)} rows and {len(copied)} CSVs; "
        f"overlaid {len(set(replaced))} ids -> {output_zip}"
    )


if __name__ == "__main__":
    main()
