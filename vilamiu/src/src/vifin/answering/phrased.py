"""Find a row using the model's phrasings of the metric, judged by the old matcher.

Measured on 400 in-scope single-figure questions: across all eight shortlisted
tables the label matcher finds **no** row at all on 18% of them, and where it does
find one the median score is 0.89. There is no grey zone in between — nothing sits
below the 0.55 threshold. So the failure is not a threshold to loosen, it is that
the question's words and the statement's words are different words, and
`metric_variants` generates a median of ONE phrasing to try.

"Vốn cổ phần" is printed "Vốn góp của chủ sở hữu". "Tiền mặt" is printed "Tiền".
Token overlap cannot bridge either. Naming the same thing a different way is a
language task, and it is the one place in this pipeline where a model has been
measured to have room:

  picking a cell out of a table it was handed   cost 0.38 questions per row (20/08)
  choosing among the matcher's candidates       little room: most near-ties are the
                                                same row printed differently
  rephrasing the metric                         18% of questions match nothing

The safety property is what makes it worth shipping. Every phrasing the model
offers is run through `lookup.match_row` — the same scorer, the same threshold, the
same cross-table comparison — and the best-scoring row wins. A phrasing that names
nothing real matches nothing and the answer is what it was before. There is no path
by which a bad suggestion ships a wrong number, which is exactly what the cell-
picking experiment could not say for itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from vifin.answering import lookup as lookup_mod
from vifin.query.parse import ParsedQuestion
from vifin.store import TableKey, TableStore


@dataclass(slots=True)
class Found:
    key: TableKey
    row: int
    column: int
    score: float
    label: str
    phrase: str


def best_row(
    question: ParsedQuestion,
    phrases: list[str],
    store: TableStore,
    keys: list[TableKey],
) -> Found | None:
    """The highest-scoring (table, row) any of the phrasings reaches.

    `phrases` is the model's list plus whatever the rules already produce; order
    does not matter because the score decides. Ties keep the earlier phrase, which
    puts the rule-based phrasing ahead of the model's on equal evidence.
    """

    seen: set[str] = set()
    candidates: list[str] = []
    for phrase in phrases:
        text = (phrase or "").strip()
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        candidates.append(text)
        for variant in lookup_mod.metric_variants(text):
            if variant.casefold() not in seen:
                seen.add(variant.casefold())
                candidates.append(variant)
    if not candidates:
        return None

    best: Found | None = None
    for key in keys:
        grid = store.rows(key)
        if len(grid) < 2:
            continue
        label_col = lookup_mod.label_column(grid)
        column = lookup_mod.pick_column(grid, question, label_col)
        if column is None:
            continue
        caption = store.meta(key).caption
        for phrase in candidates:
            match = lookup_mod.match_row(grid, phrase, label_col, caption)
            if match is None:
                continue
            row, score, label = match
            if score < lookup_mod.MIN_LABEL_SCORE:
                continue
            if best is None or score > best.score:
                best = Found(key=key, row=row, column=column, score=score,
                             label=label, phrase=phrase)
    return best
