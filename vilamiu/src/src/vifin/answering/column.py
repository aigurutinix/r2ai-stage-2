"""Choose the column by what the question asks for, and refuse the rest.

`lookup.pick_column` has two rules: take a column whose header contains the year,
otherwise take the first value column. Both are wrong on note tables, and the
cross-year instrument shows the cost. Of 79 disagreements between two independent
reads of the same figure, 42 are localisation errors and **35 are column errors** —
the locator found the right row label in both reports and then read different
columns.

Two shapes cause almost all of them.

A movement note lays out `Số đầu năm | Số phát sinh trong năm | Đã nộp | Số cuối
năm`. None of those is a reporting period, so "header contains the year" latches
onto `1/1/2020` — the OPENING balance — and the positional fallback takes
`Số phát sinh`, the flow. Measured case: a question about corporate income tax read
222,811 from one report and 1,601,262 from the other, same row label, both columns
wrong.

A capital note lays out `Số cổ phiếu | VND`. The share count parses as a number and
sits left of the value, so the fallback takes it: a question asking for nghìn tỷ
đồng read 500,000,000 shares in one report and 5,000,000,000,000 đồng in the other.

So the rules here are exclusions first, preference second. Every experiment on
20/08 that widened a candidate set and picked by score lost points — model cell
choice, note-title pinning, search depth 8 to 30, full-document scan. Narrowing is
the operation that has worked, and a column this refuses is a column the caller can
decline on rather than guess at.
"""

from __future__ import annotations

import re

from vifin.answering import lookup as lookup_mod
from vifin.query.parse import ParsedQuestion

# Headers naming a flow through the period rather than a balance at its edge.
# Reading one of these for a "số dư cuối năm" question is a category error, not a
# near miss.
MOVEMENT_RE = re.compile(
    r"s[ốo] ph[áa]t sinh|ph[áa]t sinh trong|trong n[ăa]m|trong k[ỳy]|"
    r"t[ăa]ng trong|gi[ảa]m trong|[đd][ãa] n[ộo]p|[đd][ãa] tr[ảa]|[đd][ãa] thu|"
    r"[đd][ãa] chi|[đd][ãa] s[ửu] d[ụu]ng|ho[àa]n nh[ậa]p|tr[íi]ch l[ậa]p|"
    r"gi[áa] tr[ịi] giao d[ịi]ch|s[ốo] [đd][ãa] tr[ảa]",
    re.I)

# Headers naming something that is not an amount of money. These parse as numbers
# and sit left of the value column, so the positional fallback prefers them.
NON_MONEY_RE = re.compile(
    r"s[ốo] c[ổo] phi[ếe]u|s[ốo] l[ượuơ]ng|t[ỷy] l[ệe]|t[ỉi] l[ệe]|%|"
    r"s[ởo] h[ữu]u|bi[ểe]u quy[ếe]t|l[ợo]i [íi]ch|ph[ầa]n v[ốo]n|"
    r"m[ãa] s[ốo]|thuy[ếe]t minh|s[ốo] ng[àa]y|k[ỳy] h[ạa]n",
    re.I)

CLOSING_RE = re.compile(
    r"s[ốo] cu[ốo]i|cu[ốo]i n[ăa]m|cu[ốo]i k[ỳy]|cu[ốo]i th[áa]ng|31[/.]12|"
    r"t[ạa]i ng[àa]y",
    re.I)
OPENING_RE = re.compile(
    r"s[ốo] [đd][ầa]u|[đd][ầa]u n[ăa]m|[đd][ầa]u k[ỳy]|1[/.]1[/.]|01[/.]01",
    re.I)

# What the question itself asks for.
Q_CLOSING_RE = re.compile(
    r"cu[ốo]i n[ăa]m|cu[ốo]i k[ỳy]|[đd][ếe]n ng[àa]y|t[ạa]i ng[àa]y|"
    r"31[/.]12|v[àa]o cu[ốo]i",
    re.I)
Q_OPENING_RE = re.compile(r"[đd][ầa]u n[ăa]m|[đd][ầa]u k[ỳy]|1[/.]1[/.]", re.I)


def header_text(grid: list[list[str]], column: int, depth: int = 2) -> str:
    return " ".join(
        str(grid[row][column]) for row in range(min(depth, len(grid)))
        if column < len(grid[row]))


def pick(
    grid: list[list[str]],
    question: ParsedQuestion,
    label_col: int = 0,
    strict: bool = True,
    position: int | None = None,
    exclusions: bool = True,
    period_pref: bool = True,
) -> int | None:
    """The column to read, or None when no column can be justified.

    `strict=False` reproduces `lookup.pick_column` exactly, so the two can be
    compared on the cross-year instrument without a second code path.

    `position` is the positional fallback for callers that know which period they
    want by convention rather than by header text — the comparative read of a Y+1
    report wants the second value column when no header names year Y. It indexes
    the ALLOWED columns, not the raw ones: indexing the raw list is what let the
    first A/B of these rules silently override every exclusion, and the comparison
    came out a no-op because of it.
    """

    columns = lookup_mod.value_columns(grid, label_col)
    if not columns:
        return None
    if not strict:
        return lookup_mod.pick_column(grid, question, label_col)

    headers = {column: header_text(grid, column) for column in columns}
    wants_money = bool(question.unit_scale)

    allowed = []
    for column in columns:
        text = headers[column]
        if not exclusions:
            allowed.append(column)
            continue
        if wants_money and NON_MONEY_RE.search(text):
            continue
        # A movement column is only excluded when the question asks for a balance.
        # "Số tiền đã nộp trong năm" legitimately wants the flow.
        if Q_CLOSING_RE.search(question.question) and MOVEMENT_RE.search(text) \
                and not CLOSING_RE.search(text):
            continue
        allowed.append(column)
    if not allowed:
        # Refusing beats guessing: the caller can fall through to another branch,
        # and a wrong column ships a confident wrong number.
        return None

    year = str(max(question.years)) if question.years else None
    if year:
        named = [c for c in allowed if year in headers[c]]
        if named and not period_pref:
            return named[0]
        if named:
            # Among columns naming the year, honour the period the question asks
            # for. A movement note names the year on its opening column too.
            if Q_CLOSING_RE.search(question.question):
                closing = [c for c in named if CLOSING_RE.search(headers[c])
                           and not OPENING_RE.search(headers[c])]
                if closing:
                    return closing[0]
                not_opening = [c for c in named
                               if not OPENING_RE.search(headers[c])]
                if not_opening:
                    return not_opening[0]
            if Q_OPENING_RE.search(question.question):
                opening = [c for c in named if OPENING_RE.search(headers[c])]
                if opening:
                    return opening[0]
                return named[0]
            if not Q_CLOSING_RE.search(question.question):
                return named[0]
            # A closing-balance question whose only year-bearing column is the
            # OPENING one: a movement note names the year on `1/1/2020` and
            # nowhere else. Returning it reads the wrong edge of the period, which
            # is what the instrument caught on the corporate-income-tax questions.
            # Fall through to the header-text rules instead of honouring the year.

    # No column names the year. Statements put the current period first, which is
    # the convention `value_columns` already returns.
    if period_pref and Q_OPENING_RE.search(question.question):
        opening = [c for c in allowed if OPENING_RE.search(headers[c])]
        if opening:
            return opening[0]
    if period_pref and Q_CLOSING_RE.search(question.question):
        closing = [c for c in allowed
                   if CLOSING_RE.search(headers[c])
                   and not OPENING_RE.search(headers[c])]
        if closing:
            return closing[0]
    if position is not None:
        return allowed[position] if position < len(allowed) else None
    return allowed[0]
