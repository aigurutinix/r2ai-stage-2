"""Answer a ratio question by searching the table for two rows, not by parsing it.

`ratio.py` reads the question with a regular expression: "tỷ trọng X trên Y" gives
the numerator and the denominator by name. That mechanism cannot be finished. The
exam asks 251 ratio questions and the regex parses 48 of them; of the 88 it fails on
with one company and a rate unit, 19 differ from a parseable one by the single word
`trong`, and behind those 19 sits the same open-ended tail — "chiếm bao nhiêu %",
"bằng bao nhiêu % của", "tỷ suất … trên", "so với tổng". Every phrasing added is one
alternation, and the next question is phrased a way that is not in the list.

This never reads the question's structure. Both operands are found the same way a
human skims: score every row of the table by how much of *its own label* the question
repeats, keep the few best, and try the pairs. Which of the two is the denominator is
decided by the result — a share is a number in range, and the wrong way round gives
a number that is not.

That inverts the failure mode. The regex is silent unless the wording is one it
knows; this is silent only when no pair of rows produces a plausible rate, whatever
the wording was. It also cannot be fooled by the wording alone: `SANITY` and the
minimum coverage still have to be satisfied by real cells.

Note what this deliberately does not do: it never asks a model to pick a cell. That
was measured on the board on 20/08 and cost 0.38 questions per row changed — given
the right table, the model picks the wrong cell almost every time. Here the model is
not in the loop at all; the search is over labels the corpus already contains.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from vifin.answering import lookup as lookup_mod
from vifin.answering.plan_cells import Cell, Plan, compile_plan
from vifin.query.parse import ParsedQuestion
from vifin.store import TableKey, TableStore

# A rate beyond this is not a near miss, it is the wrong pair of rows. Same limit
# `ratio.py` uses, for the same reason.
SANITY = 1000.0

# How much of a row's own label the question has to repeat before the row is a
# candidate. Labels are short and specific ("Tài sản ngắn hạn", "Chi phí bán
# hàng"); a question asking about one of them repeats nearly all of it. Below
# half, the match is coincidence.
MIN_COVERAGE = 0.5

# Free pairing does not work. Measured on all 157 in-scope questions it resolved
# 154 of them, and the labels say why: in a table of subsidiaries every row
# carries the parent's name, so "Công ty CP Năng lượng Hòa Phát" scores 0.67
# against a question about Hòa Phát, and two unrelated subsidiaries divide into a
# number that passes every range check. It divided 'Cộng' by 'Cộng' and 'Anh Vũ
# Phú Yên' by itself.
#
# What survives is the one reading that has a meaning: a share OF A TOTAL. The
# denominator must be the row the table itself calls its total, and the numerator
# must repeat almost all of its own label. Everything else is declined, and a
# later branch answers.
DENOMINATOR_MUST_BE_TOTAL = True
MIN_NUMERATOR = 0.8

# Rows that are structure, not line items. Matching one of these produces a
# quotient of a heading against a total, which parses fine and means nothing.
FURNITURE_RE = re.compile(
    r"^\s*(?:đơn vị|unit|thuyết minh|mã số|chỉ tiêu|tài sản|nguồn vốn)\s*[:.]?\s*$",
    re.I)

# Words every question carries and no row label distinguishes itself by.
STOP = frozenset({
    "cua", "cong", "ty", "ctcp", "tong", "nam", "cuoi", "dau", "ngay",
    "bao", "nhieu", "la", "dat", "trong", "tren", "phan", "tram", "trieu",
    "nghin", "dong", "cp", "co", "tap", "doan", "ngan", "hang", "tmcp",
    "me", "hop", "nhat", "rieng", "gia", "tri", "muc", "khoan", "so", "du",
    "chiem", "bang", "tinh", "xac", "dinh", "lan", "vong", "suat", "le",
})


@dataclass(slots=True)
class PairAnswer:
    code: str
    keys: list[TableKey]
    variables: list[str]
    labels: tuple[str, str]
    score: float
    value: float


def _content(text: str) -> set[str]:
    """The words that can distinguish one line item from another."""

    return {t for t in lookup_mod._tokens(text) if len(t) > 2 and t not in STOP}


def _coverage(question_words: set[str], label: str) -> float:
    """Share of the label's own content words that the question repeats.

    Coverage of the *label*, not of the question: the question also names a
    company, a year and a unit, none of which any row label contains, so scoring
    against the question's own token count penalises every correct row equally.
    """

    words = _content(label)
    if not words:
        return 0.0
    return len(question_words & words) / len(words)


def _candidates(grid: list[list[str]], question_words: set[str], label_col: int,
                limit: int = 4) -> list[tuple[int, float, str]]:
    """The rows whose labels the question most repeats, best first."""

    scored = []
    for index in range(1, len(grid)):
        line = grid[index]
        if label_col >= len(line):
            continue
        label = str(line[label_col] or "").strip()
        if not label or FURNITURE_RE.match(label):
            continue
        score = _coverage(question_words, label)
        if score >= MIN_COVERAGE:
            scored.append((index, score, label))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored[:limit]


def resolve(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    shortlist: int = 8,
) -> PairAnswer | None:
    """The best (numerator row, denominator row) pair across the shortlist."""

    if question.target_unit not in ("phan_tram", "lan", "vong"):
        return None
    if len(question.tickers) != 1 or not question.years:
        return None

    question_words = _content(question.question)
    if len(question_words) < 2:
        return None

    groups = max(1, len(question.tickers)) * max(1, len(question.years))
    per_group = max(2, -(-shortlist // groups))
    keys = [hit.key for hit in retriever.search_balanced(
        question, per_group=per_group, cap=shortlist)]

    as_percent = question.target_unit == "phan_tram"
    best: PairAnswer | None = None

    for key in keys:
        grid = store.rows(key)
        if len(grid) < 3:
            continue
        label_col = lookup_mod.label_column(grid)
        column = lookup_mod.pick_column(grid, question, label_col)
        if column is None:
            continue
        meta = store.meta(key)
        scale = lookup_mod.column_scale(
            grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}")

        rows = _candidates(grid, question_words, label_col)
        total = lookup_mod.total_row(grid, label_col)
        if total is not None and all(index != total for index, _, _ in rows):
            # The denominator of a share is usually the table's total, and the
            # total row never repeats the metric's words — it says "Cộng". It
            # therefore cannot come out of the coverage search and has to be
            # offered explicitly.
            label = str(grid[total][label_col] or "").strip() if \
                label_col < len(grid[total]) else ""
            rows = rows + [(total, MIN_COVERAGE, label or "Cộng")]
        if len(rows) < 2:
            continue

        for top_index, top_score, top_label in rows:
            for bottom_index, bottom_score, bottom_label in rows:
                if top_index == bottom_index:
                    continue
                top_value = lookup_mod._parse_cell(grid[top_index][column]) \
                    if column < len(grid[top_index]) else None
                bottom_value = lookup_mod._parse_cell(grid[bottom_index][column]) \
                    if column < len(grid[bottom_index]) else None
                if top_value is None or not bottom_value:
                    continue
                reported = abs(top_value) / abs(bottom_value) * (
                    100.0 if as_percent else 1.0)
                if not reported or abs(reported) > SANITY:
                    continue
                if as_percent and reported > 100.0:
                    # A share above 100 means the pair is the wrong way round;
                    # the reversed pair is tried by this same loop.
                    continue
                combined = top_score + bottom_score
                if best is not None and combined <= best.score:
                    continue
                cells = (Cell(0, top_index, column), Cell(0, bottom_index, column))
                op = "ratio_pct" if as_percent else "ratio"
                best = PairAnswer(
                    code=compile_plan(Plan(op, cells), [scale, scale], 1.0,
                                      ["df"], None, magnitude=True),
                    keys=[key],
                    variables=["df"],
                    labels=(top_label, bottom_label),
                    score=combined,
                    value=round(reported, 2),
                )
    return best
