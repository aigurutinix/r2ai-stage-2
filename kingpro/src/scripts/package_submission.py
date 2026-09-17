"""Create a competition ZIP with submission.json and data at archive root."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.submission.archive import write_deterministic  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--referenced-only", action="store_true", help="include only evidence CSVs referenced by submission.json")
    args = parser.parse_args()
    root = args.submission_dir.resolve()
    output = args.out.resolve() if args.out else root.with_suffix(".zip")
    submission = root / "submission.json"
    data = root / "data"
    if not submission.is_file() or not data.is_dir():
        raise SystemExit("expected submission.json and data/ in submission directory")
    if args.referenced_only:
        rows = json.loads(submission.read_text(encoding="utf-8"))
        relative_files = sorted({
            item["csv_path"]
            for row in rows
            for item in row.get("evidence", [])
            if isinstance(item, dict) and isinstance(item.get("csv_path"), str)
        })
        missing = [name for name in relative_files if not (root / name).is_file()]
        if missing:
            raise SystemExit(f"missing referenced evidence files: {missing[:10]}")
        paths = [root / name for name in relative_files]
    else:
        paths = [path for path in sorted(data.rglob("*")) if path.is_file()]
    entries = 1
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        write_deterministic(archive, submission, "submission.json")
        for path in paths:
            write_deterministic(archive, path, path.relative_to(root).as_posix())
            entries += 1
    with zipfile.ZipFile(output) as archive:
        names = archive.namelist()
        if names.count("submission.json") != 1 or not any(name.startswith("data/") for name in names):
            raise SystemExit("invalid archive layout")
    print(json.dumps({"archive": str(output), "entries": entries, "bytes": output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
