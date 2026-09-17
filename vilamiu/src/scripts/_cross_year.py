"""Measure the locator against the corpus's own redundancy, with no gold at all.

Every instrument this project has used to judge an answer has been unusable. The
local gold set is self-generated: it lies about retrieval and stores raw cells in
28% of records, and measuring on it has now cost three submissions. The leaderboard
is honest but gives one scalar per submission, ten a day.

The reports carry a third source of truth that needs neither. A balance sheet
prints two columns — the closing balance of this year and the opening balance,
which IS the closing balance of last year. So the figure a question asks about for
end-of-2022 appears twice in the corpus: as "Số cuối năm" in the 2022 report, and
as "Số đầu năm" in the 2023 report. Two different documents, two different
retrievals, two different row matches, one number.

If our locator reads both and they agree to the scorer's own tolerance, it almost
certainly read the right row in both: two wrong rows agreeing to 0.01 absolute on
a nine-digit figure does not happen by chance. If they disagree, at least one read
is wrong and we know it — per question, offline, for free.

The corpus supports this almost everywhere: 215 (company, scope) pairs with a
median of 11 years each, 1,677 consecutive-year pairs, 91% of companies having at
least one.

What this buys is the thing the frozen artifact got by burning ten leaderboard
slots a day: per-question selection between mechanisms. Offline.

Usage:
  PYTHONPATH=src python scripts/_cross_year.py --n 300
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import column as column_mod  # noqa: E402
from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402

# Set from --strict so the old and new column rules can be compared on the same
# instrument without a second copy of the read path.
STRICT = [False]
MODE = ["both"]
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402

# The scorer's own tolerance is 0.01 absolute on the reported figure. Comparing in
# đồng instead, a relative band is the honest test: OCR of a nine-digit number and
# a differing rounding convention between two reports both move the last digits.
REL_TOL = 1e-4


def locate(question, year_for_search: int, year_for_column: int,
           store: TableStore, retriever, metric: str, shortlist: int = 8):
    """Best (value in đồng, label, table) for `metric` in that year's reports.

    Two years go in because they play different roles. `year_for_search` chooses
    which report to read; `year_for_column` chooses which column inside it. Asking
    the 2023 report for the column headed 2022 is what reads the comparative.
    """

    probe = dataclasses.replace(question, years=[year_for_search])
    column_probe = dataclasses.replace(question, years=[year_for_column])
    keys = [hit.key for hit in retriever.search_balanced(
        probe, per_group=shortlist, cap=shortlist)]

    found: list[tuple] = []
    for key in keys:
        if str(store.meta(key).year) != str(year_for_search):
            continue
        grid = store.rows(key)
        if len(grid) < 2:
            continue
        label_col = lookup_mod.label_column(grid)
        match = lookup_mod.match_row(grid, metric, label_col, store.meta(key).caption)
        if match is None:
            continue
        row, score, label = match
        if score < lookup_mod.MIN_LABEL_SCORE:
            continue
        # The positional fallback now runs INSIDE `pick`, over the columns its
        # exclusions left, so a movement or share-count column cannot be selected
        # by position. Passing it here rather than overriding afterwards is the
        # difference between testing the rules and testing nothing.
        position = 0 if year_for_column == year_for_search else 1
        column = column_mod.pick(grid, column_probe, label_col,
                                 strict=STRICT[0], position=position,
                                 exclusions=MODE[0] in ("excl", "both"),
                                 period_pref=MODE[0] in ("pref", "both"))
        if column is None:
            continue
        header = " ".join(str(grid[r][column]) for r in range(min(2, len(grid)))
                          if column < len(grid[r]))
        if not STRICT[0] and str(year_for_column) not in header:
            # `pick_column` falls back to the FIRST value column when no header
            # names the year, and for the comparative read that fallback returns
            # this year's figure, which agrees with nothing. Requiring a literal
            # year in the header is correct but only 26% of reads survive it —
            # most statements head their columns "Số cuối năm" / "Số đầu năm" with
            # no year at all. The convention those follow is positional: current
            # period first, prior period second, which is what `value_columns`
            # already returns left to right. So when the year is absent, take the
            # position the convention assigns rather than dropping the read.
            columns = lookup_mod.value_columns(grid, label_col)
            wanted = 0 if year_for_column == year_for_search else 1
            if len(columns) <= wanted:
                continue
            column = columns[wanted]
        raw = lookup_mod._parse_cell(grid[row][column]) \
            if column < len(grid[row]) else None
        if raw is None:
            continue
        meta = store.meta(key)
        scale = lookup_mod.column_scale(
            grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}")
        value = abs(raw) * scale
        if not value:
            continue
        header = " | ".join(
            str(grid[r][column]) for r in range(min(2, len(grid)))
            if column < len(grid[r]))
        found.append((score, value, label, f"{key.doc_name}|t{key.table_id}",
                      column, header[:60], str(grid[row][column])[:24], scale))

    if not found:
        return None
    found.sort(key=lambda item: -item[0])
    best = found[0]
    # A statement prints the same figure twice — once in the balance sheet, once in
    # the note that breaks it down — so the best match in a DIFFERENT table is a
    # second reading that is independent in the way that matters here: it can only
    # agree with the first if both landed on the real line item. Returned alongside
    # the winner rather than discarded, because independence is the currency of the
    # consensus objective, not coverage.
    alternate = next((item for item in found[1:] if item[3] != best[3]), None)
    return best, alternate


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--out", default="artifacts/cross_year.jsonl")
    parser.add_argument("--show", type=int, default=8)
    parser.add_argument("--scope", choices=("calibrated", "wide"),
                        default="calibrated")
    parser.add_argument("--strict", action="store_true",
                        help="use the exclusion-first column rules")
    parser.add_argument("--mode", choices=("excl", "pref", "both"), default="both")
    args = parser.parse_args()
    STRICT[0] = args.strict
    MODE[0] = args.mode

    questions = parse_all(ROOT / "data" / "questions" / "questions.jsonl",
                          ROOT / "data" / "code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    # `is_single_lookup` plus exactly one year covered 300 questions and produced
    # the calibration. For consensus the scope should be every question whose
    # ANSWER is a single cell, whatever branch normally serves it — a reading is
    # worth having even where another mechanism currently wins, because the value
    # of a channel here is its independence, not its coverage of the branch.
    import importlib.util

    shape_spec = importlib.util.spec_from_file_location(
        "qs", ROOT / "scripts" / "_question_shape.py")
    shape_mod = importlib.util.module_from_spec(shape_spec)
    shape_spec.loader.exec_module(shape_mod)

    # Two scopes, and the narrow one is the measurement. `is_single_lookup` plus a
    # single year is the population the calibration was built on; widening to every
    # one-cell question raised coverage from 185 to 298 reads but DILUTED the
    # consensus bands (the 4-channel band's match rate fell 48% -> 39%), because
    # tables without two period columns get a positional fallback that is
    # meaningless. So `calibrated` is the number to track across changes, and
    # `wide` exists only to feed extra channels.
    if args.scope == "calibrated":
        scope = [q for q in questions
                 if len(q.tickers) == 1 and len(q.years) == 1
                 and lookup_mod.is_single_lookup(q.question)][:args.n]
    else:
        scope = [q for q in questions
                 if len(q.tickers) == 1 and q.years
                 and shape_mod.shape(q.question) == "mot o"][:args.n]
    print(f"{len(scope)} cau (pham vi {args.scope})", flush=True)

    both = agree = 0
    only_current = only_next = 0
    records, disagreements = [], []
    started = time.time()
    for index, question in enumerate(scope, start=1):
        year = question.years[0]
        metric = lookup_mod.extract_metric(question.question)
        try:
            first = locate(question, year, year, store, retriever, metric)
            second = locate(question, year + 1, year, store, retriever, metric)
        except Exception:  # noqa: BLE001
            continue
        current, alternate = first if first else (None, None)
        comparative, _ = second if second else (None, None)
        if current and not comparative:
            only_current += 1
        elif comparative and not current:
            only_next += 1
        if not (current and comparative):
            continue
        both += 1
        left, right = current[1], comparative[1]
        close = abs(left - right) <= REL_TOL * max(abs(left), abs(right))
        if close:
            agree += 1
        else:
            disagreements.append((question.question, current, comparative))
        def detail(read):
            return {"value": read[1], "label": read[2], "table": read[3],
                    "score": read[0], "col": read[4], "header": read[5],
                    "cell": read[6], "scale": read[7]}

        records.append({
            "id": question.id,
            "agree": bool(close),
            "current": detail(current),
            "comparative": detail(comparative),
        })
        if alternate is not None:
            records[-1]["alternate"] = detail(alternate)
        if index % 50 == 0:
            print(f"  {index}/{len(scope)}  ca hai={both} trung={agree}  "
                  f"{time.time() - started:.0f}s", flush=True)

    (ROOT / args.out).write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
        encoding="utf-8")

    print(f"\nchi doc duoc bao cao nam Y : {only_current}")
    print(f"chi doc duoc bao cao Y+1  : {only_next}")
    print(f"doc duoc CA HAI           : {both}/{len(scope)} "
          f"({100 * both / max(1, len(scope)):.0f}%)")
    if both:
        print(f"HAI DUONG DOC LAP TRUNG NHAU: {agree}/{both} "
              f"({100 * agree / both:.0f}%)")
    print(f"\n{min(args.show, len(disagreements))} vi du lech nhau:")
    for text, current, comparative in disagreements[:args.show]:
        print(f"  {text[:96]}")
        print(f"    nam Y  {current[1]:>22,.0f}  {str(current[2])[:44]}  ({current[3]})")
        print(f"    Y+1    {comparative[1]:>22,.0f}  {str(comparative[2])[:44]}  "
              f"({comparative[3]})")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
