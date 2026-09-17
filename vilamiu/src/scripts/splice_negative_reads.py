"""Replace answers that rest on a failed row lookup with ones that do not.

`find_row` returns -1 when no label matches; a program that uses it anyway makes
`num(frame, -1, c)` read the table's LAST row, which in a balance sheet is the
total. 75 of 1012 answers in `direct14b_repair.zip` read at least one negative
index, and twelve of them came back as exactly 100.0 — total assets divided by
itself.

This differs from `splice_rows.py` in the criterion, and the criterion is the whole
point. That script swapped answers that *looked* impossible, and hit 6 of 44: an
impossible answer replaced by a plausible one is still usually wrong. This one
swaps answers whose *mechanism* is proven broken, and only accepts a replacement
whose mechanism is not.

Usage:
  PYTHONPATH=src python scripts/splice_negative_reads.py \
      --base direct14b_repair.zip --fallback tags_clean.zip --out aimed.zip
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from trace_answer import read_grid  # noqa: E402

from vifin.answering.reads import negative_reads  # noqa: E402

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("sp", ROOT / "scripts" / "splice_rows.py")
_sp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sp)


def load(path: Path):
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
    rows = payload if isinstance(payload, list) else (
        payload.get("predictions") or list(payload.values())[0])
    return payload, rows


def frames_for(archive: zipfile.ZipFile, row) -> dict:
    import pandas as pd

    frames = {}
    for item in row.get("evidence") or []:
        try:
            grid = read_grid(archive, item["csv_path"])
        except KeyError:
            continue
        if grid:
            frames[item["variable"]] = pd.DataFrame(grid[1:], columns=grid[0])
    return frames


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--fallback", action="append", required=True,
                        help="may be repeated; tried in order")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    base_path = ROOT / "submissions" / args.base
    payload, rows = load(base_path)

    broken = []
    with zipfile.ZipFile(base_path) as archive:
        for index, row in enumerate(rows):
            frames = frames_for(archive, row)
            if frames and negative_reads(row.get("pandas_query") or "", frames):
                broken.append(index)
    print(f"{args.base}: {len(broken)} câu đọc chỉ số âm")

    swapped, needed, from_where = 0, set(), {}
    for name in args.fallback:
        other_path = ROOT / "submissions" / name
        if not other_path.exists():
            print(f"  bỏ qua {name}: không có")
            continue
        _, other_rows = load(other_path)
        other = {r["question"]: r for r in other_rows}
        with zipfile.ZipFile(other_path) as archive:
            for index in list(broken):
                candidate = other.get(rows[index]["question"])
                if candidate is None:
                    continue
                frames = frames_for(archive, candidate)
                # Two conditions, not one. The mechanism must be sound — no
                # negative read — and the answer must not contradict its own
                # question. Requiring only the first traded 70 broken answers for
                # 13 fresh self-evident defects, because the donor builds are
                # weaker overall.
                if not frames or negative_reads(candidate.get("pandas_query") or "",
                                                frames):
                    continue
                if _sp.impossible(candidate["question"], candidate.get("answer")):
                    continue
                rows[index] = candidate
                from_where[candidate["id"]] = name
                for item in candidate.get("evidence") or []:
                    needed.add((name, item["csv_path"]))
                broken.remove(index)
                swapped += 1
    print(f"  đổi {swapped} dòng, còn {len(broken)} câu không có bản thay thế sạch")

    if isinstance(payload, list):
        payload = rows
    elif "predictions" in payload:
        payload["predictions"] = rows
    else:
        payload[list(payload.keys())[0]] = rows

    out_path = ROOT / "submissions" / args.out
    with zipfile.ZipFile(base_path) as src:
        present = set(src.namelist())
        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as dst:
            for name in src.namelist():
                if name != "submission.json":
                    dst.writestr(name, src.read(name))
            added = 0
            for source, csv_path in sorted(needed):
                if csv_path in present:
                    continue
                with zipfile.ZipFile(ROOT / "submissions" / source) as extra:
                    try:
                        dst.writestr(csv_path, extra.read(csv_path))
                    except KeyError:
                        continue
                present.add(csv_path)
                added += 1
            dst.writestr("submission.json",
                         json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"  thêm {added} csv -> {args.out}")


if __name__ == "__main__":
    main()
