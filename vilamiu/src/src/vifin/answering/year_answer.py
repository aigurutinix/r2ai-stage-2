"""Repair answers to "năm nào" questions that came back as a value.

A question of the form "Năm nào X đạt mức cao nhất trong các năm A, B, C?" wants a
year. Twenty-two of the fifty-three such questions in the best submission answered
with the figure instead — 212.164.523.100 where 2023 was meant. Those are wrong
with certainty, which is what makes the repair safe: any plausible year is at
least a candidate, while the figure cannot be right.

The repair is deterministic and produces a real program. The reported value is
located in the cited table, the year columns of that same row are read off the
header, and a comparison chain over those columns is emitted that returns the
winning year. That is the shape the generator already uses for this question type,
and it reads the DataFrame rather than asserting a constant.
"""

from __future__ import annotations

import re

YEAR_QUESTION_RE = re.compile(
    r"(năm nào|vào năm nào|thời điểm nào|năm bao nhiêu)", re.I)
YEAR_IN_HEADER_RE = re.compile(r"(19|20)\d{2}")
MIN_YEAR, MAX_YEAR = 1990, 2100


def asks_for_a_year(question: str) -> bool:
    return bool(YEAR_QUESTION_RE.search(question))


def looks_like_a_year(value) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return number.is_integer() and MIN_YEAR <= number <= MAX_YEAR


def _cell_number(text):
    # A float arriving here must not go through the Vietnamese reader: stripping
    # "." as a thousands separator turns 218605226759.0 into 2186052267590.
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        return float(text)
    raw = str(text).strip()
    if not raw or raw in ("-", "--"):
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    raw = raw.strip("()").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"-?[\d.,]+", raw):
        return None
    cleaned = raw.replace(".", "").replace(",", ".") if "," in raw else raw.replace(".", "")
    try:
        number = float(cleaned)
    except ValueError:
        return None
    return -number if negative else number


def year_columns(grid) -> dict:
    """Column index -> year, for every column whose header names one."""

    if not grid:
        return {}
    found = {}
    for index, header in enumerate(grid[0]):
        match = YEAR_IN_HEADER_RE.search(str(header))
        if match:
            year = int(match.group(0))
            if MIN_YEAR <= year <= MAX_YEAR:
                found[index] = year
    return found


def locate(grid, value) -> int | None:
    """Row holding the reported figure, comparing as numbers."""

    target = _cell_number(value)
    if target is None:
        return None
    for index, row in enumerate(grid[1:]):
        for cell in row:
            number = _cell_number(cell)
            if number is not None and number == target:
                return index
    return None


def repair(grid, value, want_max: bool = True):
    """A program returning the winning year, and that year, or None.

    Returns `(code, year)`. `None` means the question cannot be repaired from this
    table — no year columns, or the figure is not in it — and the caller should
    leave the original answer alone rather than invent one.
    """

    columns = year_columns(grid)
    if len(columns) < 2:
        return None
    row = locate(grid, value)
    if row is None:
        return None

    ordered = sorted(columns.items())
    readable = []
    for column, year in ordered:
        if column < len(grid[row + 1]) and _cell_number(grid[row + 1][column]) is not None:
            readable.append((column, year))
    if len(readable) < 2:
        return None

    lines = []
    for position, (column, _) in enumerate(readable):
        lines.append(f"v{position} = num(df, {row}, {column})")
    lines.append("best = v0")
    lines.append(f"best_year = {readable[0][1]}")
    comparison = ">" if want_max else "<"
    for position, (_, year) in enumerate(readable[1:], start=1):
        lines.append(f"if v{position} {comparison} best:")
        lines.append(f"    best = v{position}")
        lines.append(f"    best_year = {year}")
    lines.append("result = float(best_year)")

    values = [_cell_number(grid[row + 1][column]) for column, _ in readable]
    best_index = (values.index(max(values)) if want_max
                  else values.index(min(values)))
    return "\n".join(lines), float(readable[best_index][1])


DOC_YEAR_RE = re.compile(r"_(19|20)(\d{2})_")


def year_of_document(doc_name: str) -> int | None:
    match = DOC_YEAR_RE.search(str(doc_name))
    if not match:
        return None
    year = int(match.group(1) + match.group(2))
    return year if MIN_YEAR <= year <= MAX_YEAR else None


def _row_of(grid, label: str) -> int | None:
    wanted = " ".join(str(label).split()).lower()
    if not wanted:
        return None
    for index, row in enumerate(grid[1:]):
        if " ".join(str(row[0]).split()).lower() == wanted:
            return index
    return None


def repair_across_documents(frames, value, want_max: bool = True):
    """Compare one line item across annual reports and return the winning year.

    Most "năm nào" questions name years that live in separate documents, not in
    separate columns of one table: "trong các năm 2016, 2018 và 2020" is three
    annual reports. The single-table repair declined 19 of 31 such questions for
    exactly that reason, because their headers read "Năm nay"/"Năm trước".

    `frames` is a list of `(year, grid)` in the order the program will bind them.
    The row is identified by the label of the cell that produced the reported
    figure, and the same label is then read in every other year.
    """

    if len(frames) < 2:
        return None

    anchor = None
    for position, (year, grid) in enumerate(frames):
        row = locate(grid, value)
        if row is not None:
            anchor = (position, year, grid, row)
            break
    if anchor is None:
        return None
    _, _, anchor_grid, anchor_row = anchor
    label = str(anchor_grid[anchor_row + 1][0]).strip()
    column = None
    target = _cell_number(value)
    for index, cell in enumerate(anchor_grid[anchor_row + 1]):
        if _cell_number(cell) == target:
            column = index
            break
    if column is None:
        return None

    readable = []
    for position, (year, grid) in enumerate(frames):
        row = _row_of(grid, label)
        if row is None or column >= len(grid[row + 1]):
            continue
        if _cell_number(grid[row + 1][column]) is None:
            continue
        readable.append((position, year, row, _cell_number(grid[row + 1][column])))
    if len(readable) < 2:
        return None

    names = [f"df{index + 1}" for index, _ in enumerate(readable)]
    lines = []
    for index, (_, _, row, _) in enumerate(readable):
        lines.append(f"v{index} = num({names[index]}, {row}, {column})")
    lines.append("best = v0")
    lines.append(f"best_year = {readable[0][1]}")
    comparison = ">" if want_max else "<"
    for index, (_, year, _, _) in enumerate(readable[1:], start=1):
        lines.append(f"if v{index} {comparison} best:")
        lines.append(f"    best = v{index}")
        lines.append(f"    best_year = {year}")
    lines.append("result = float(best_year)")

    values = [entry[3] for entry in readable]
    winner = values.index(max(values)) if want_max else values.index(min(values))
    return "\n".join(lines), float(readable[winner][1]), [e[0] for e in readable]


SINGLE_FRAME_RE = re.compile(r"\bdf\d*\b")


def replay_across_years(code: str, frames, want_max: bool = True):
    """Run one extraction against each year's table and return the winning year.

    Tracing the reported figure back to a cell fails whenever it is a total
    rather than a cell, which is most of these questions: "tổng giá gốc nợ phải
    thu quá hạn" is a sum of rows and appears nowhere as a number. But the program
    the model already wrote knows how to extract that quantity, so the year
    comparison can reuse it rather than rediscover it.

    `frames` is `[(year, grid)]`. The program is re-emitted once per year with its
    frame renamed, and a comparison chain picks the winner. Returns
    `(code, year, positions)` or None.
    """

    names = set(SINGLE_FRAME_RE.findall(code or ""))
    if len(names) != 1 or len(frames) < 2:
        return None
    original = names.pop()

    blocks = []
    for index, _ in enumerate(frames):
        renamed = re.sub(rf"\b{re.escape(original)}\b", f"df{index + 1}", code)
        renamed = re.sub(r"\bresult\b", f"v{index}", renamed)
        blocks.append(renamed)

    lines = list(blocks)
    lines.append("best = v0")
    lines.append(f"best_year = {frames[0][0]}")
    comparison = ">" if want_max else "<"
    for index, (year, _) in enumerate(frames[1:], start=1):
        lines.append(f"if v{index} {comparison} best:")
        lines.append(f"    best = v{index}")
        lines.append(f"    best_year = {year}")
    lines.append("result = float(best_year)")
    return "\n".join(lines), None, list(range(len(frames)))
