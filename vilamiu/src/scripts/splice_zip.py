"""Move whole answers from one built submission into another, for the listed ids only.

A rebuild has lost every time it was tried — four submissions, each differing from the
incumbent on more than four hundred rows, each scoring below it. A narrow splice has
won: the diffs that gained were between forty and sixty rows. So a new mechanism ships
as a diff against the best zip rather than as a build of its own, and the diff size is
an argument of the tool rather than a consequence of the mechanism.

Only what belongs to the answer moves: `answer`, `pandas_query`, `evidence` and the csv
the new evidence cites. `relevant_tables` and `relevant_docs` stay as the base has them,
which holds TABLES_F2 fixed and makes any change in EXECUTION_ACCURACY attributable to
the spliced rows alone.

Usage:
  python scripts/splice_zip.py --base submissions/aimed.zip \
      --from submissions/fresh_tab.zip --ids-file artifacts/fresh/tab_ids.json \
      --out submissions/tab_splice.zip
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOVED = ("answer", "pandas_query", "evidence")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--from", dest="source", required=True)
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0,
                        help="cap the diff at this many rows, lowest id first")
    args = parser.parse_args()

    with zipfile.ZipFile(ROOT / args.base) as archive:
        base_rows = json.loads(archive.read("submission.json"))
        base_files = {name: archive.read(name) for name in archive.namelist()
                      if name != "submission.json"}
    with zipfile.ZipFile(ROOT / args.source) as archive:
        source_rows = {r["id"]: r for r in
                       json.loads(archive.read("submission.json"))}
        source_files = {name: archive.read(name) for name in archive.namelist()
                        if name != "submission.json"}

    wanted = None
    if args.ids_file:
        wanted = set(json.loads(
            (ROOT / args.ids_file).read_text(encoding="utf-8")))

    candidates = []
    for row in base_rows:
        new = source_rows.get(row["id"])
        if new is None:
            continue
        if wanted is not None and row["id"] not in wanted:
            continue
        if all(new.get(field) == row.get(field) for field in MOVED):
            continue
        candidates.append(row["id"])
    if args.limit:
        candidates = sorted(candidates)[:args.limit]
    moving = set(candidates)

    out_rows, cited = [], set()
    for row in base_rows:
        if row["id"] in moving:
            new = source_rows[row["id"]]
            merged = dict(row)
            for field in MOVED:
                if field in new:
                    merged[field] = new[field]
            out_rows.append(merged)
            # An `evidence` entry is a dict of {variable, csv_path}, not a bare
            # filename. Treating it as a string copied nothing, and the zip shipped
            # with 34 evidence rows pointing at csv files that were not in it.
            evidence = new.get("evidence")
            if isinstance(evidence, str):
                evidence = [evidence]
            for item in evidence or []:
                if isinstance(item, str):
                    cited.add(item)
                elif isinstance(item, dict) and item.get("csv_path"):
                    cited.add(item["csv_path"])
        else:
            out_rows.append(dict(row))

    files = dict(base_files)
    for name in cited:
        for key in (name, f"data/{name}"):
            if key in source_files:
                files[key] = source_files[key]
                break

    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json",
                         json.dumps(out_rows, ensure_ascii=False, indent=1))
        for name, blob in sorted(files.items()):
            archive.writestr(name, blob)

    print(f"co so   : {args.base}")
    print(f"nguon   : {args.source}")
    print(f"doi     : {len(moving)} dong")
    print(f"-> {target}")


if __name__ == "__main__":
    main()
