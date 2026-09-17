"""The one link a financial statement declares about itself.

A balance sheet or income statement carries a `Thuyết minh` column beside its
`Mã số` column. The number in it points at the note that details that exact
line. The report states the link, so following it needs no model and no
vocabulary matching, which is what makes it a different kind of evidence from
BM25: its mistakes are not BM25's mistakes.

Only one hop, only inside the same report.
"""

from __future__ import annotations

from collections import Counter
import re

from .retrieval import tokenize


CODE = re.compile(r"^\d{2,3}[A-Za-z]?$")
# "6", "12", "5.1", "V.01" — the shapes a note reference takes in these reports.
NOTE = re.compile(r"^(?:[IVX]+\.)?\d{1,2}(?:\.\d{1,2})?$")
TABLE_TAG = re.compile(r"<table")
NUMBER = re.compile(r"^\(?-?\d[\d.,\s]*%?\)?$")
NOTE_HEADING = re.compile(
    r"^\s*(?:[IVX]+\.)?\d{1,2}(?:\.\d{1,2})?[.)]?\s+\S[^\n]{5,70}$",
    re.M,
)


def normalize(text: str) -> str:
    return " ".join(tokenize(text))


def code_column(grid: list[list[str]], probe_rows: int = 30) -> int | None:
    """The `Mã số` column: the one carrying repeated two/three digit codes.

    Detected rather than assumed, because the label column is sometimes split by
    OCR and the code lands in a different position from report to report. Three
    is enough to distinguish a code column from a stray number in a data cell.
    """
    for row in grid[:3]:
        for index, cell in enumerate(row):
            if normalize(cell) == "ma so":
                return index

    counts: Counter[int] = Counter()
    for row in grid[:probe_rows]:
        for index, cell in enumerate(row):
            if CODE.match(cell.strip()):
                counts[index] += 1
    return next((index for index in sorted(counts) if counts[index] >= 3), None)


def note_column(grid: list[list[str]], code_index: int | None) -> int | None:
    """The declared note column, falling back to the cell right of `Mã số`."""
    for row in grid[:3]:
        for index, cell in enumerate(row):
            if normalize(cell) == "thuyet minh":
                return index
    return code_index + 1 if code_index is not None else None


def row_label(row: list[str], code_index: int | None, note_index: int | None) -> str:
    """Join textual label cells while excluding coordinates and values."""
    blocked = {index for index in (code_index, note_index) if index is not None}
    return " ".join(
        cell.strip() for index, cell in enumerate(row)
        if index not in blocked and cell.strip() and not NUMBER.fullmatch(cell.strip())
    )


def row_note(
    row: list[str], code_index: int, note_index: int | None = None,
) -> str | None:
    """The note reference in the declared or adjacent note column."""
    index = code_index + 1 if note_index is None else note_index
    if index >= len(row):
        return None
    cell = row[index].strip()
    return cell if NOTE.match(cell) else None


def normalize_note(note: str) -> str:
    """Normalize `V.04`, `04`, and `05.01` without losing sub-note levels."""
    parts = note.split(".")
    if parts and re.fullmatch(r"[IVX]+", parts[0]):
        parts = parts[1:]
    return ".".join(str(int(part)) for part in parts if part.isdigit())


def note_table_line(report_text: str, note: str) -> int | None:
    """Source line of the table under the heading a note reference names.

    The heading is matched on its own line — `26. Doanh thu thuần` — and the
    table taken is the first one after it. Bounding the heading length keeps a
    numbered sentence in a paragraph of prose from being read as a note title.
    """
    if not note:
        return None
    number = normalize_note(note)
    if not number:
        return None
    heading = re.search(
        rf"^\s*{re.escape(number)}[.)]?\s+\S[^\n]{{5,70}}$", report_text, re.M,
    )
    if not heading:
        return None
    following = report_text[heading.end():]
    table = TABLE_TAG.search(following)
    if not table:
        return None
    next_heading = NOTE_HEADING.search(following)
    if next_heading and next_heading.start() < table.start():
        return None
    return report_text.count("\n", 0, heading.end() + table.start()) + 1


def linked_note_lines(
    report_text: str, grid: list[list[str]], line_items: list[str], source_line: int,
) -> list[tuple[str, int]]:
    """`(note reference, source line)` for every row a question's items name.

    Linking from every row would pull in most of the notes in the report; the
    question's own line items are what make one row the relevant one. A note
    resolving back to the table it came from is dropped, since it adds nothing.
    """
    code_index = code_column(grid)
    if code_index is None or not line_items:
        return []
    note_index = note_column(grid, code_index)
    found: dict[str, int] = {}
    for row in grid:
        if not row:
            continue
        label = normalize(row_label(row, code_index, note_index))
        if not label or not any(f" {item} " in f" {label} " for item in line_items):
            continue
        note = row_note(row, code_index, note_index)
        if not note or note in found:
            continue
        line = note_table_line(report_text, note)
        if line is not None and line != source_line:
            found[note] = line
    return sorted(found.items(), key=lambda pair: pair[1])
