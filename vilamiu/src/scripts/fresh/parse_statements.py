"""Read the OCR reports the way the organisers' own generator reads them.

Written from `vifinqa-official/src/vifinqa/common/corpus/statement.py`, which is the
provenance of the gold answers for every question about the three primary
statements. That parser does something none of the obvious approaches do: it never
matches the question's words against a row label. It keys on `Mã số`.

The contract, verbatim from their code:

  * a row counts only if it holds a `Mã số` — an integer of one to three digits,
    taken from the column whose header normalises to "mãsố", else from one of the
    first three columns
  * value cells are the row's remaining cells that parse as Vietnamese numbers and
    are NOT bare integers (that exclusion is what drops note references)
  * `current` is the FIRST value cell in the row and `prior` is the SECOND —
    positional, with no column header consulted at all
  * the table's unit scale comes from its header, else from an "Đơn vị tính" line
    near it, and the stored value is raw × scale, i.e. đồng
  * a table is a statement only if: five or more three-digit codes (cân đối kế
    toán), or a label containing "chuyển tiền" (lưu chuyển tiền tệ), or three or
    more of {01, 11, 20, 25, 26} (kết quả kinh doanh). Anything else is discarded.

Why this matters: `Mã số` is a one-to-three digit integer fixed by Thông tư 200, so
the address of a figure is external knowledge rather than something to be recovered
from OCR'd Vietnamese. Row labels are the noisy part of these documents; the code
beside them is not.

This module only reads and reports. It makes no attempt to answer anything yet.
"""

from __future__ import annotations

import csv
import io
import re
import unicodedata
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

PAGE_RE = re.compile(r"^=====\s*PAGE\s+(\d+)\s*=====\s*$", re.M)
TABLE_RE = re.compile(r"<table>.*?</table>", re.S | re.I)
MA_SO_RE = re.compile(r"^\d{1,3}$")
BARE_INT_RE = re.compile(r"^\d+$")
VN_NUMBER_RE = re.compile(r"^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$")
HEADER_MA_SO_HINT = "mãsố"
UNIT_PREFIXES = ("Đơn vị tính", "Đơn vị tiền tệ", "Đơn vị", "ĐVT")

KQKD_REQUIRED = frozenset({"01", "11", "20", "25", "26"})
MIN_KQKD_MATCHES = 3
MIN_CDKT_3DIGIT = 5


class _TableReader(HTMLParser):
    """Rows of cell strings from one `<table>` blob."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag):
        if tag in ("td", "th") and self._row is not None and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self.rows.append(self._row)
            self._row = None

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)


def parse_vn_number(raw: str) -> float | None:
    text = str(raw).strip()
    if not text or text == "-":
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    if text.startswith("-"):
        negative = True
        text = text[1:].strip()
    if text.endswith("%"):
        text = text[:-1].strip()
    if not VN_NUMBER_RE.match(text):
        return None
    # OCR leaves unbalanced parentheses — `(1.564.203` matches the regex, since the
    # closing bracket is optional there, but survives into `float()`. The
    # organisers' parser catches ValueError around the same conversion, so a cell
    # like that is simply not a value; treating it as one would invent a figure.
    try:
        value = float(text.replace(".", "").replace(",", "."))
    except ValueError:
        return None
    return -value if negative else value


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()


# OCR variants of VND. A single misread character discards a whole statement: the
# corpus contains "Đơn vị tính: VNO", and that table is a textbook balance sheet.
VND_RE = r"\bvn[dgo0]\b|\bvnd\b|\bwnd\b"


def scale_from_unit_text(text: str, *, strict: bool = True) -> float | None:
    """The multiplier a unit line implies, or None when the text names no unit.

    `strict` reproduces the organisers' rule: both a đồng token and a unit marker
    must be present. That rule discards 3,631 tables whose headers are textbook
    statement headers — `Mã số | CHỈ TIÊU | Thuyết minh | Năm nay | Năm trước` —
    for the sole reason that the page carries no `Đơn vị tính` line, because a
    statement runs over several pages and the unit is declared once at its start.
    """

    flat = fold(text)
    has_dong = re.search(r"\bdong\b", flat) or re.search(VND_RE, flat)
    if not has_dong:
        return None
    if strict:
        if not re.search(r"\bdon\s*vi\b|\bdvt\b", flat) and not re.search(VND_RE, flat):
            return None
    if re.search(r"\btrieu\b", flat):
        return 1e6
    if re.search(r"\bty\b", flat):
        return 1e9
    if re.search(r"\bnghin\b", flat):
        return 1e3
    return 1.0


@dataclass(frozen=True, slots=True)
class Cell:
    ma_so: str
    label: str
    value: float
    raw: str
    row_idx: int
    col_idx: int
    scale: float


@dataclass(frozen=True, slots=True)
class Statement:
    doc_name: str
    table_id: int
    kind: str
    scale: float
    current: dict[str, Cell]
    prior: dict[str, Cell]
    # False when no unit line was found and `scale` is a placeholder 1.0. The values
    # are then in whatever the table was printed in, not in đồng. Defaults to True so
    # existing construction sites keep their meaning.
    scale_known: bool = True


@dataclass(frozen=True, slots=True)
class Table:
    doc_name: str
    table_id: int
    page_no: int
    header: list[str]
    rows: list[list[str]]
    unit_snippets: tuple[str, ...]


def read_document(path: Path) -> list[Table]:
    """Every `<table>` in the file, with the page it sat on and nearby unit lines."""

    text = path.read_text(encoding="utf-8", errors="replace")
    doc_name = path.parent.name
    pages: list[tuple[int, str]] = []
    marks = list(PAGE_RE.finditer(text))
    for index, mark in enumerate(marks):
        end = marks[index + 1].start() if index + 1 < len(marks) else len(text)
        pages.append((int(mark.group(1)), text[mark.end():end]))
    if not pages:
        pages = [(1, text)]

    tables: list[Table] = []
    table_id = 0
    # MEASURED AND REJECTED: carrying the previous page's unit line forward, so a
    # statement's later pages inherit the declaration made at its start. It added
    # 786 statement tables and made the reads worse — cross-document agreement on
    # typed addresses fell 90.9% -> 89.3%, scale errors went 153 -> 310, and
    # `kqkd 20 = 10 - 11` fell 76.5% -> 71.6%. The inherited unit belongs to a
    # different section as often as it belongs to this table: it produced figures
    # like 3,978,192,126,938,999,947,264 by applying 1e9 to đồng. The organisers'
    # strictness here is protective, not an oversight.
    for page_no, body in pages:
        snippets = tuple(
            line.strip() for line in body.splitlines()
            if any(line.strip().casefold().startswith(p.casefold())
                   for p in UNIT_PREFIXES))
        for blob in TABLE_RE.findall(body):
            reader = _TableReader()
            reader.feed(blob)
            rows = [r for r in reader.rows if any(c.strip() for c in r)]
            if not rows:
                continue
            table_id += 1
            tables.append(Table(doc_name=doc_name, table_id=table_id, page_no=page_no,
                                header=rows[0], rows=rows[1:], unit_snippets=snippets))
    return tables


def _header_ma_so_index(header: list[str]) -> int | None:
    for index, cell in enumerate(header):
        if HEADER_MA_SO_HINT in "".join(str(cell).split()).casefold():
            return index
    return None


def _find_ma_so_index(row: list[str], hint: int | None, scan_width: int = 3) -> int | None:
    order: list[int] = []
    if hint is not None:
        order.append(hint)
    order += [i for i in range(min(scan_width, len(row))) if i not in order]
    for index in order:
        if index < len(row) and MA_SO_RE.match(str(row[index]).strip()):
            return index
    return None


def _value_cells(row: list[str], exclude: int) -> list[tuple[int, float, str]]:
    out = []
    for index, cell in enumerate(row):
        if index == exclude:
            continue
        raw = str(cell).strip()
        if not raw or BARE_INT_RE.match(raw):
            continue
        value = parse_vn_number(raw)
        if value is None:
            continue
        out.append((index, value, raw))
    return out


def _row_label(row: list[str], ma_so_idx: int, value_indices: set[int]) -> str:
    candidates = [str(c).strip() for i, c in enumerate(row)
                  if i != ma_so_idx and i not in value_indices and str(c).strip()]
    return max(candidates, key=len) if candidates else ""


def _classify(codes: set[str], labels: list[str]) -> str | None:
    if len({c for c in codes if len(c) == 3}) >= MIN_CDKT_3DIGIT:
        return "cdkt"
    if any("chuyen tien" in fold(label) for label in labels):
        return "lctt"
    if len(codes & KQKD_REQUIRED) >= MIN_KQKD_MATCHES:
        return "kqkd"
    return None


def parse_statement(table: Table) -> Statement | None:
    hint = _header_ma_so_index(table.header)
    parsed = []
    for row_idx, row in enumerate(table.rows):
        ma_so_idx = _find_ma_so_index(row, hint)
        if ma_so_idx is None:
            continue
        values = _value_cells(row, exclude=ma_so_idx)
        if not values:
            continue
        parsed.append((str(row[ma_so_idx]).strip(), row_idx,
                       _row_label(row, ma_so_idx, {i for i, _, _ in values}), values))
    if not parsed:
        return None

    kind = _classify({p[0] for p in parsed}, [p[2] for p in parsed])
    if kind is None:
        return None

    scale = scale_from_unit_text(",".join(table.header))
    if scale is None:
        for snippet in table.unit_snippets:
            scale = scale_from_unit_text(snippet)
            if scale is not None:
                break

    # A missing unit used to discard the whole statement. Measured on 250 documents,
    # that threw away 18% of the corpus — balance sheets and income statements that had
    # parsed perfectly, and whose printed identities confirm they were read correctly:
    # 48 of them satisfy two identities, one income statement satisfies five. The scale
    # is an attribute of the table, not a precondition for the table existing, and it
    # can still be settled afterwards from the document-wide scan, the neighbouring
    # year's column, or the note that ties back to a line.
    #
    # `scale_known` says whether the figures below are in đồng or in whatever the table
    # was printed in. A consumer that needs đồng MUST check it; treating an unknown
    # scale as 1.0 silently claims the table was denominated in đồng, and that error
    # cost a submission (`vote7`, 0.3913 against 0.4150).
    scale_known = scale is not None
    factor = scale if scale_known else 1.0

    current: dict[str, Cell] = {}
    prior: dict[str, Cell] = {}
    for ma_so, row_idx, label, values in parsed:
        if ma_so in current:
            continue
        col, raw_value, raw = values[0]
        current[ma_so] = Cell(ma_so, label, raw_value * factor, raw, row_idx, col,
                              factor)
        if len(values) >= 2:
            col2, value2, raw2 = values[1]
            prior[ma_so] = Cell(ma_so, label, value2 * factor, raw2, row_idx, col2,
                                factor)
    return Statement(table.doc_name, table.table_id, kind, factor, current, prior,
                     scale_known)


def statements_in(path: Path) -> list[Statement]:
    out = []
    for table in read_document(path):
        statement = parse_statement(table)
        if statement is not None:
            out.append(statement)
    return out
