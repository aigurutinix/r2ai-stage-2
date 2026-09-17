"""Show which cell a shipped answer actually came from, so an error has a cause.

`verify_against_reports.py` searches the report for rows whose label resembles the
question and prints their values. It adjudicated two of six sampled answers: the
label matcher pulls in subsidiary names, and the questions in this set deliberately
avoid the table's own wording — the organisers' generator rejects any question that
copies a row label verbatim.

This works the other way round and needs no matching. The submission already names
the frames its program binds, and the answer is a number. So find the cell in those
frames whose value equals the answer — raw, column-scaled, or scaled by a power of
ten — and print its row label and column header. That is what the system read.

The judgement is still the reader's, but the question becomes answerable at a
glance: is this row the line item the question names, and is this column its
period? A wrong row and a wrong column look completely different, and knowing
which one it is decides where the fix goes.

Usage:
  PYTHONPATH=src python scripts/trace_answer.py --base direct14b_repair.zip \
      --branch lookup --n 8 --seed 11
"""

from __future__ import annotations

import argparse
import io
import json
import random
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _rescore_unit_fixed import close, parse_number  # noqa: E402

from vifin.answering import lookup as lookup_mod  # noqa: E402

SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)


def read_grid(archive: zipfile.ZipFile, path: str):
    """The csv as a list of rows, exactly as the scorer would bind it."""

    import csv

    text = archive.read(path).decode("utf-8", "replace")
    return [row for row in csv.reader(io.StringIO(text))]


def find_cells(grid, target: float, context: str):
    """Every cell whose value could be the answer, with how it was scaled."""

    hits = []
    for index, row in enumerate(grid):
        for column, cell in enumerate(row):
            value = parse_number(cell)
            if value is None or value == 0:
                continue
            column_scale = lookup_mod.column_scale(grid, column, context)
            for scale in SCALES:
                for candidate, how in (
                    (value / scale, f"/{scale:g}"),
                    (value * column_scale / scale, f"×cột/{scale:g}"),
                ):
                    if close(candidate, target, 2e-3):
                        hits.append((index, column, str(cell), how))
                        break
                else:
                    continue
                break
    return hits


def logged_reads(query: str, frames: dict):
    """Every (variable, row, col, cell) the program reads through `num`.

    Searching the frames for a cell equal to the answer found it in one of six
    sampled questions: most answers are rounded, rescaled, or combined, so the
    number never appears verbatim. Watching the reads instead is exact, and it
    also shows the reads of a program whose arithmetic is wrong — which is the
    case the value search can never see.

    Every program in this corpus reads through the `num(frame, r, c)` helper the
    prelude defines, so wrapping that one function covers them.
    """

    import pandas as pd

    reads: list[tuple[str, int, int, str]] = []
    names = {id(frame): name for name, frame in frames.items()}

    def tracer(frame, event, arg):
        # Rebinding `num` in the namespace does not work: the program defines it
        # itself, so the definition overwrites any wrapper injected beforehand,
        # and injecting afterwards means the body has already run. Tracing needs
        # no source surgery and survives whatever the model wrote.
        if event != "call" or frame.f_code.co_name != "num":
            return None
        args = frame.f_locals
        table = args.get("frame")
        row, column = args.get("r"), args.get("c")
        if table is None or row is None or column is None:
            return None
        try:
            cell = str(table.iloc[int(row), int(column)])
        except Exception:  # noqa: BLE001
            cell = "<ngoài bảng>"
        reads.append((names.get(id(table), "?"), int(row), int(column), cell))
        return None

    namespace: dict = {"pd": pd}
    namespace.update(frames)
    sys.settrace(tracer)
    try:
        exec(query, namespace, namespace)  # noqa: S102 - our own generated code
    except Exception:  # noqa: BLE001 - a broken program still has readable reads
        pass
    finally:
        sys.settrace(None)
    return reads


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--branch", default="")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--ids", default="", help="comma-separated ids instead of a sample")
    # 823 of 1012 questions need one company and hold ~45% of the answers right.
    # That pool is where the score is, so it needs to be sampled on its own.
    parser.add_argument("--single", action="store_true",
                        help="only questions naming at most one company")
    args = parser.parse_args()

    path = ROOT / "submissions" / args.base
    with zipfile.ZipFile(path) as archive:
        payload = json.loads(archive.read("submission.json").decode("utf-8"))
        rows = payload if isinstance(payload, list) else (
            payload.get("predictions") or list(payload.values())[0])

        if args.single:
            from vifin.query.companies import CompanyRoster
            from vifin.query.parse import parse_question
            roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
            rows = [row for row in rows
                    if len(set(parse_question(0, row["question"], roster).tickers)) <= 1]

        pool = rows
        if args.ids:
            keep = {int(part) for part in args.ids.split(",") if part.strip()}
            pool = [row for row in rows if row["id"] in keep]
        elif args.branch:
            branches = {int(k): v for k, v in json.loads(
                (ROOT / "artifacts" / "branches.json").read_text(encoding="utf-8")).items()}
            pool = [row for row in rows if branches.get(row["id"]) == args.branch]
            random.Random(args.seed).shuffle(pool)
        else:
            pool = list(rows)
            random.Random(args.seed).shuffle(pool)

        print(f"{len(pool)} câu trong phạm vi; đọc {min(args.n, len(pool))}\n")
        for row in pool[: args.n]:
            answer = parse_number(row.get("answer"))
            print("=" * 100)
            print(f"id={row['id']}  {row['question'][:150]}")
            print(f"NỘP : {row.get('answer')}")
            if answer is None:
                print("      đáp án không phải số")
                continue
            # Watch the program run first: it names the cells exactly. Fall back
            # to searching for the value only when nothing was read through `num`.
            import pandas as pd
            frames = {}
            for item in row.get("evidence") or []:
                try:
                    grid = read_grid(archive, item["csv_path"])
                except KeyError:
                    continue
                if grid:
                    frames[item["variable"]] = pd.DataFrame(grid[1:], columns=grid[0])
            traced = logged_reads(row.get("pandas_query") or "", frames) if frames else []
            if traced:
                seen_reads = set()
                for variable, r_i, c_i, cell in traced:
                    if (variable, r_i, c_i) in seen_reads:
                        continue
                    seen_reads.add((variable, r_i, c_i))
                    frame = frames.get(variable)
                    label = header = ""
                    if frame is not None:
                        try:
                            label = str(frame.iloc[r_i, 0])[:56]
                        except Exception:
                            label = "<?>"
                        cols = list(frame.columns)
                        header = str(cols[c_i])[:34] if c_i < len(cols) else "<?>"
                    print(f"  ĐỌC {variable} r{r_i} c{c_i}  ô={cell!r}")
                    print(f"      dòng  : {label!r}")
                    print(f"      cột   : {header!r}")
                    # A blank label is not judgeable on its own: Vietnamese
                    # statements leave it empty on the total line, and the row
                    # above tells you whether this is that total or one item of a
                    # list. 21.4% of gold cells sit on such a row.
                    if frame is not None and not label.strip():
                        for near in (r_i - 2, r_i - 1, r_i + 1):
                            if 0 <= near < len(frame):
                                try:
                                    text = str(frame.iloc[near, 0])[:56]
                                except Exception:
                                    continue
                                print(f"        lân cận r{near}: {text!r}")
                continue

            found = False
            for item in row.get("evidence") or []:
                csv_path = item["csv_path"]
                try:
                    grid = read_grid(archive, csv_path)
                except KeyError:
                    print(f"      thiếu {csv_path}")
                    continue
                hits = find_cells(grid, answer, csv_path)
                if not hits:
                    continue
                found = True
                doc = csv_path.split("/")[-1].replace(".csv", "")
                for index, column, cell in ((h[0], h[1], h[2]) for h in hits[:3]):
                    label = str(grid[index][0])[:56] if grid[index] else ""
                    header = str(grid[0][column])[:34] if grid and column < len(grid[0]) else ""
                    print(f"  ĐỌC Ô r{index} c{column}  ô={cell!r}")
                    print(f"      dòng  : {label!r}")
                    print(f"      cột   : {header!r}")
                    print(f"      bảng  : {doc[:74]}")
            if not found:
                # An answer that appears in none of its own frames is either
                # computed from several cells or not in the tables at all. Both
                # are worth knowing; the second is a fabrication.
                print("      KHÔNG tìm thấy ô nào khớp trong các bảng đã khai "
                      "(tính từ nhiều ô, hoặc không có trong bảng)")


if __name__ == "__main__":
    main()
