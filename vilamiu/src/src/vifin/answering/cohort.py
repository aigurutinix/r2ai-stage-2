"""Answer a question naming several companies by reading each one separately.

The measurement that forced this branch: of 1012 exam questions, 288 name more than
one ticker, and **one** of them reaches the high-confidence band of the consensus
objective. Meanwhile all 198 confident answers are single-cell questions. The
arithmetic then closes the argument — EXEC 0.3439 is about 348 correct, roughly 290
of them among the 506 single-cell questions, so the derived half is running at ~11%.
Answering every single-cell question perfectly would still land under 0.57. There is
no route to 0.60 that does not go through these 288.

And the pipeline is not attempting them: 160 of the 288 ship with a single table in
`evidence`, meaning a question about three companies was answered by reading one.

What makes them tractable is their shape rather than their difficulty. 82% name only
two to four companies, and 64% carry no ranking step at all — they are "read the same
line item for each of these companies and combine". That is N single-cell reads,
which is the one operation this project does well, plus one arithmetic step. Nothing
here asks a model to look at a table; the per-company read is the same
`match_row` / `pick_column` / `column_scale` path the shipped lookup branch uses, and
the only new thing is the loop and the operation.

Deliberately narrow. A read that fails for ANY of the named companies fails the whole
question, so the branch declines rather than answering from a subset — an average
over two of four companies is a confident wrong number, and the day's evidence is
that confident wrong numbers cost 0.3 to 0.4 questions each.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from vifin.answering import lookup as lookup_mod
from vifin.answering.plan_cells import Cell, Plan, compile_plan
from vifin.query.parse import ParsedQuestion
from vifin.store import TableKey, TableStore

# The operation, read off the question's own words. Order matters: "tổng chênh
# lệch" is a difference, and "chênh lệch" must therefore be tested before "tổng".
# Ranking questions ("cao nhất", "trung vị") are NOT here — those need a filter
# step over the cohort and are left to a later branch rather than answered wrongly.
OPS: tuple[tuple[str, str], ...] = (
    (r"ch[êe]nh l[ệe]ch|kh[áa]c bi[ệe]t|nhi[ềe]u h[ơo]n|[íi]t h[ơo]n", "diff"),
    (r"trung b[ìi]nh|b[ìi]nh qu[âa]n", "avg_of"),
    (r"t[ổo]ng|c[ộo]ng l[ạa]i|g[ộo]p", "sum"),
)

# A ranking or filtering step means the answer is one company's figure chosen by a
# comparison, not a combination of all of them. 36% of the cohort questions carry
# one, and this branch must not silently treat them as sums.
RANKING_RE = re.compile(
    r"trung v[ịi]|cao nh[ấa]t|th[ấa]p nh[ấa]t|l[ớo]n nh[ấa]t|nh[ỏo] nh[ấa]t|"
    r"v[ượuơ]t|tr[êe]n trung|d[ướuơ]i trung|x[ếe]p h[ạa]ng|th[ứu] nh[ấa]t",
    re.I)


@dataclass(slots=True)
class CohortAnswer:
    code: str
    keys: list[TableKey]
    variables: list[str]
    labels: list[str]
    op: str
    score: float
    value: float


def operation(question: str) -> str | None:
    """The combining operation, or None when the question ranks instead."""

    if RANKING_RE.search(question):
        return None
    for pattern, name in OPS:
        if re.search(pattern, question, re.I):
            return name
    return None


def _read_one(question: ParsedQuestion, ticker: str, metric: str,
              store: TableStore, retriever, shortlist: int):
    """The single-cell read for one company, exactly as the lookup branch does it.

    The probe keeps the question's own wording and swaps only the ticker, so the
    year, the scope and the metric extraction are unchanged — the point is to reuse
    the path that works, not to invent a second one.
    """

    probe = dataclasses.replace(question, tickers=[ticker])
    groups = max(1, len(probe.years))
    per_group = max(2, -(-shortlist // groups))
    keys = [hit.key for hit in retriever.search_balanced(
        probe, per_group=per_group, cap=shortlist)]

    best = None
    for key in keys:
        grid = store.rows(key)
        if len(grid) < 2:
            continue
        label_col = lookup_mod.label_column(grid)
        match = lookup_mod.match_row(grid, metric, label_col,
                                     store.meta(key).caption)
        if match is None:
            continue
        row, score, label = match
        if score < lookup_mod.MIN_LABEL_SCORE:
            continue
        column = lookup_mod.pick_column(grid, probe, label_col)
        if column is None:
            continue
        if lookup_mod._parse_cell(grid[row][column]) is None \
                if column < len(grid[row]) else True:
            continue
        if best is None or score > best[0]:
            meta = store.meta(key)
            scale = lookup_mod.column_scale(
                grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}")
            best = (score, key, row, column, label, scale)
    return best


def resolve(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    shortlist: int = 8,
) -> CohortAnswer | None:
    """One cell per named company, combined by the question's own operation."""

    if len(question.tickers) < 2 or not question.years:
        return None
    op = operation(question.question)
    if op is None:
        return None
    if op == "diff" and len(question.tickers) != 2:
        # `diff` is binary. Three companies and a "chênh lệch" is asking something
        # this shape cannot express.
        return None

    metric = lookup_mod.extract_metric(question.question)
    if not metric:
        return None

    reads = []
    for ticker in question.tickers:
        found = _read_one(question, ticker, metric, store, retriever, shortlist)
        if found is None:
            # Declining beats answering from a subset: an average over two of four
            # companies is a confident wrong number.
            return None
        reads.append(found)

    cells, scales, labels, keys = [], [], [], []
    for score, key, row, column, label, scale in reads:
        if key not in keys:
            keys.append(key)
        cells.append(Cell(keys.index(key), row, column))
        scales.append(scale)
        labels.append(label)

    names = ["df"] if len(keys) == 1 else [f"df{i + 1}" for i in range(len(keys))]
    plan = Plan(op, tuple(cells))
    if not plan.valid:
        return None
    code = compile_plan(plan, scales, question.unit_scale or 1.0, names, None,
                        magnitude=True)
    return CohortAnswer(
        code=code,
        keys=keys,
        variables=names,
        labels=labels,
        op=op,
        score=min(read[0] for read in reads),
        value=0.0,
    )
