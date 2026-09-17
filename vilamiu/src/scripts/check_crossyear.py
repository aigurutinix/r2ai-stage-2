"""Estimate lookup accuracy without gold answers, using the corpus itself.

Every annual report restates the previous year beside the current one. A figure
we extract from report(ticker, Y) should therefore reappear in the prior-year
column of report(ticker, Y+1). Agreement is independent evidence that the label,
the column, and the unit were all read correctly; disagreement localises which.

This is not a substitute for the leaderboard — it only covers questions whose
label also matches in the neighbouring year — but it turns "accuracy unknown"
into a number we can move between submissions.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableStore  # noqa: E402

SEARCH_K = 10
TOLERANCE = 2e-4


def prior_year_value(grid: list[list[str]], label: str, target_year: int) -> float | None:
    """Read `label`'s figure from the column standing for `target_year`."""

    matched = lookup_mod.match_row(grid, label)
    if matched is None:
        return None
    row, _, _ = matched
    columns = lookup_mod.value_columns(grid)
    if not columns:
        return None

    chosen = None
    for column in columns:
        header = " ".join(str(r[column]) for r in grid[:2] if column < len(r))
        if str(target_year) in header:
            chosen = column
            break
    if chosen is None:
        # No year in the header: the restated prior year is the second figure
        # column ("Năm nay" then "Năm trước", "Số cuối năm" then "Số đầu năm").
        if len(columns) < 2:
            return None
        chosen = columns[1]
    if chosen >= len(grid[row]):
        return None
    raw = lookup_mod._parse_cell(grid[row][chosen])
    if raw is None:
        return None
    return raw * lookup_mod.column_scale(grid, chosen, "")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parsed = parse_all(root / "data" / "questions" / "questions.jsonl", root / "data" / "code_stock.csv")
    store = TableStore.load(root / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    by_key: dict[tuple[str, str, str], list[str]] = {}
    for doc_name, ticker, year, scope in (
        store.frame[["doc_name", "ticker", "year", "scope"]].drop_duplicates().itertuples(index=False)
    ):
        by_key.setdefault((ticker, year, scope), []).append(doc_name)

    outcome = Counter()
    disagreements = []

    for question in parsed:
        if question.unit_scale is None or not lookup_mod.is_single_lookup(question.question):
            continue
        if not question.tickers or len(question.years) != 1:
            continue

        found = None
        source = None
        for hit in retriever.search(question, top_k=SEARCH_K):
            grid = store.rows(hit.key)
            found = lookup_mod.find(grid, question)
            if found is not None:
                source = (hit.key, grid)
                break
        if found is None:
            outcome["no_answer"] += 1
            continue

        key, grid = source
        meta = store.meta(key)
        ours = found.value * lookup_mod.column_scale(
            grid, found.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
        )

        year = question.years[0]
        neighbours = by_key.get((question.tickers[0], str(year + 1), question.scope), [])
        theirs = None
        for doc_name in neighbours:
            for table_id in store.frame.loc[store.frame.doc_name == doc_name, "table_id"]:
                candidate = prior_year_value(
                    store.rows(type(key)(doc_name, int(table_id))), found.label, year
                )
                if candidate is not None:
                    theirs = candidate
                    break
            if theirs is not None:
                break

        if theirs is None:
            outcome["unverifiable"] += 1
            continue
        if abs(ours - theirs) <= TOLERANCE * max(abs(ours), abs(theirs), 1.0):
            outcome["agree"] += 1
        else:
            outcome["disagree"] += 1
            ratio = theirs / ours if ours else float("inf")
            disagreements.append((question.id, found.label[:44], ours, theirs, ratio))

    checked = outcome["agree"] + outcome["disagree"]
    print(f"answered by lookup      {checked + outcome['unverifiable']}")
    print(f"  verifiable next year  {checked}")
    print(f"  agree                 {outcome['agree']}")
    print(f"  disagree              {outcome['disagree']}")
    if checked:
        print(f"\nestimated extraction accuracy: {outcome['agree'] / checked:.1%} (n={checked})")

    scales = Counter()
    for _, _, ours, theirs, ratio in disagreements:
        for name, factor in (("1000x", 1000), ("1e6", 1e6), ("sign", -1)):
            if abs(abs(ratio) - factor) < 0.02 * factor or (factor == -1 and ratio < 0):
                scales[name] += 1
                break
        else:
            scales["other"] += 1
    print(f"disagreement shape: {dict(scales)}")
    print("\nsample disagreements:")
    for qid, label, ours, theirs, ratio in disagreements[:12]:
        print(f"  id={qid:4d} x{ratio:>10,.2f}  ours={ours:>20,.0f} next-year={theirs:>20,.0f}  {label}")


if __name__ == "__main__":
    main()
