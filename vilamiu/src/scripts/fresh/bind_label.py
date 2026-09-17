"""Bind an arbitrary indicator name to a cell, so multi-hop can leave the Mã số world.

`hard_hop` solves the dependent-hop archetypes and it works: 53 spliced rows moved the
board from 0.4150 to 0.4289. What stops it going further is not its arithmetic — measured
on the 558 multi-cell questions it refuses, **423 of them (76%) name an indicator it cannot
bind at all**, because it reads only Circular-200 coded rows and those questions ask about
note-level items: `số dư nợ đủ tiêu chuẩn`, `chi phí xây dựng cơ bản dở dang`, `trạng thái
tiền tệ nội bảng`.

Locating those rows is what the navigation work does: the statement's own `Thuyết minh`
pointer resolves to the detailing note 92% of the time and its total ties back to the
statement line 93% of the time, against 11% when deliberately pointed at the wrong note.

So this is the join. It takes the label variants a model produced from the question — one
short call, already run for all 1012 — and returns the cell, in đồng, for a given company,
year and scope. `hard_hop`'s seven solvers then work unchanged.

Two rules that come from having read the failures by hand:

  the scale must be known   a value whose unit is a guess cannot be compared across years,
                            and an argmax over mixed units is worse than no answer
  the label is read with    a row called `Tổ chức kinh tế` means deposits or loans
  its section heading       depending on the heading above it
"""

from __future__ import annotations

import csv as csv_mod
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

import parse_statements as ps  # noqa: E402
from find_statements import locate_columns  # noqa: E402
from label_match import score  # noqa: E402
from measure_pointers import POINTER_RE, contexts_of, heading_numbers  # noqa: E402
from table_reading import (_NUMERIC as _NUMERIC_CELL, money_columns, row_label,
                           scale_from_headers, section_label)  # noqa: E402

CODE_RE = re.compile(r"^\d{1,3}$")
MIN_SCORE = 2000.0

# A balance is asked at a point in time; a flow is asked over a period. The wording
# is unambiguous in this exam and the column headers follow the same convention.
_ASKS_BALANCE = re.compile(
    r"s\u1ed1 d\u01b0|\u0111\u1ebfn ng\u00e0y|t\u1ea1i ng\u00e0y|cu\u1ed1i n\u0103m|\u0111\u1ea7u n\u0103m|cu\u1ed1i k\u1ef3|\u0111\u1ea7u k\u1ef3|"
    r"t\u1ea1i th\u1eddi \u0111i\u1ec3m|v\u00e0o ng\u00e0y|31/12|01/01", re.I)
# `L\u00e3i vay ... n\u0103m 2022` names a year with no `trong`, and it is still a flow. Balance
# wording is tested first, so `cu\u1ed1i n\u0103m 2016` never reaches this.
_ASKS_FLOW = re.compile(
    r"trong n\u0103m|trong k\u1ef3|c\u1ea3 n\u0103m|l\u0169y k\u1ebf|ph\u00e1t sinh trong|n\u0103m\s+20[0-2]\d", re.I)
# `31/12/2022`, `1/1/2022` — a column headed by a date holds a balance.
_DATE_HEADER = re.compile(r"\d{1,2}\s*[/.-]\s*\d{1,2}\s*[/.-]\s*20[0-2]\d")


# A row marker rather than a name: `I.`, `3`, `II`, `a)`.
_MARKER = re.compile(r"^[\dIVXivx]{1,4}[.)]?$")
# `Tổng cộng toàn doanh nghiệp` — a column that holds the total, not one segment.
_TOTAL_HEADER = re.compile(
    r"tổng cộng|tổng số|toàn doanh nghiệp|cộng|^\s*tổng\s*$", re.I)
# `Chênh lệch giữa A và B` — two companies, so one bound cell is the wrong answer.
_TWO_PARTIES = re.compile(r"chênh lệch giữa|so với|cao hơn|thấp hơn|"
                          r"gấp bao nhiêu|khác biệt giữa", re.I)


# `Quá hạn`, `Đến 1 tháng`, `Từ 1 đến 3 tháng` — a row is split across these, so one
# column is a fragment of the figure and not the figure.
_BUCKET_HEADER = re.compile(
    r"quá hạn|trong hạn|đến 1 tháng|đến một tháng|từ 1 đến|từ 1 tháng|"
    r"từ 3 đến|trên 5 năm|không chịu lãi|đúng hạn", re.I)
_ASKS_MATURITY = re.compile(
    r"quá hạn|trong hạn|kỳ hạn|đáo hạn|đến hạn|thanh khoản|không chịu lãi", re.I)
# `USD được quy đổi`, `EUR`, `Các ngoại tệ khác` — the columns split a figure by currency,
# so one of them is a fragment in the same way a maturity bucket is.
_CURRENCY_HEADER = re.compile(
    r"được quy đổi|ngoại tệ khác|\busd\b|\beur\b|\bjpy\b|nguyên tệ", re.I)
# `Bên liên quan`, `Mối quan hệ`, `Nội dung nghiệp vụ` — rows keyed by counterparty.
_PARTY_HEADER = re.compile(
    r"bên liên quan|mối quan hệ|nội dung nghiệp vụ|bên có liên quan", re.I)
_ASKS_PARTY = re.compile(
    r"bên liên quan|bên có liên quan|công ty con|công ty liên kết|"
    r"giao dịch với", re.I)
_ASKS_CURRENCY = re.compile(
    r"ngoại tệ|usd|eur|jpy|đô la|quy đổi|nguyên tệ", re.I)
# An organisation the question names: a form word followed by a capitalised name. The
# capital matters — `công ty mẹ` is a scope, `Công ty Cổ phần Hoàng Anh Gia Lai` is a party.
_ORG_NAME = re.compile(
    r"(?:Ngân hàng|Tổng Công ty|Công ty|Tập đoàn|CTCP|Quỹ)"
    # The name runs on through its own lowercase words (`Xăng dầu`, `Chứng khoán`)
    # and stops at the word that joins it back to the sentence.
    r"(?:\s+(?!của\b|vào\b|năm\b|là\b|bao\b|đến\b|cuối\b|đầu\b|trong\b"
    r"|tại\b|và\b|với\b|số\b|mẹ\b|riêng\b|hợp\b|theo\b)"
    r"[^\s,.;?()]+)+")
# Words that carry no meaning for matching a label against a question: the scaffolding
# every question is built from, plus the unit words the answer is asked in.
_STOP = {"của", "cho", "các", "và", "từ", "tại", "trong", "là", "bao", "nhiêu",
         "năm", "cuối", "đầu", "số", "dư", "đến", "ngày", "vào", "công", "ty",
         "mẹ", "ctcp", "cổ", "phần", "tổng", "giá", "trị", "đồng", "triệu",
         "tỷ", "nghìn", "trăm", "kỳ", "này", "đó", "với", "theo", "về", "một"}


def _content_words(text: str) -> set[str]:
    words = re.findall(r"[0-9A-Za-zÀ-ỹ]+", str(text).casefold())
    return {w for w in words if len(w) > 1 and w not in _STOP}


def exact_variant(label: str, names: list[str]) -> bool:
    """Is the row's label one of the asked names, word for word?

    NKG id=99 asks about `Tiền` and the row called `Tiền` scored 3004 while `Tiền mặt`
    scored 4008 — the scorer rewards a longer overlap, so a narrower item outranks the
    exact one. Equality is the strongest evidence available and cannot be outscored.
    """

    flat = " ".join(_content_words(label))
    return any(flat == " ".join(_content_words(name)) for name in names)


def unsaid_words(label: str, question: str) -> int:
    """How many of the label's content words the question never mentions.

    Zero means the question names this row. A large count means the row is a different
    indicator that happens to share a word, which is how a question about borrowings was
    answered from `NỢ PHẢI TRẢ` and one about cash from `Tiền mặt tại quỹ`.
    """

    return len(_content_words(label) - _content_words(question))


def shares_words(label: str, question: str) -> bool:
    """Does the matched label name something the question actually mentions?

    A label the question never alludes to means the variant list, not the table, produced
    the match — the model invented a name and the table happened to print it.
    """

    label_words = _content_words(label)
    if not label_words:
        return False
    return bool(label_words & _content_words(question))


def bucket_table(grid: list[list[str]]) -> bool:
    """Are the value columns maturity buckets rather than periods?"""

    header = " ".join(str(cell) for row in grid[:2] for cell in row)
    return bool(_BUCKET_HEADER.search(header))


def party_table(grid: list[list[str]]) -> bool:
    """Does the table key its rows by counterparty?

    HHV id=122 asks the company's short-term trade receivables and was answered with the
    subtotal of its RELATED-PARTY receivables, because that note prints the same words as a
    merged heading. A question that names no party is not asking about this table.
    """

    header = " ".join(str(cell) for row in grid[:2] for cell in row)
    return bool(_PARTY_HEADER.search(header))


def currency_table(grid: list[list[str]]) -> bool:
    """Are the value columns a split by currency rather than by period?"""

    header = " ".join(str(cell) for row in grid[:2] for cell in row)
    return bool(_CURRENCY_HEADER.search(header))


def counterparties(question: str, subject: str, resolve) -> list[str]:
    """Organisation names in the question that are not the subject company.

    `resolve` maps a piece of text to the tickers it names. A phrase that resolves to the
    subject is the subject being named; a phrase that resolves to anything else, or to
    nothing while still being a proper name, narrows the indicator to a slice of itself.
    """

    out = []
    for match in _ORG_NAME.finditer(question):
        phrase = match.group(0).strip()
        if subject in (resolve(phrase) or set()):
            continue
        out.append(phrase)
    return out


def refuses_binding(question: str, subject: str | None = None,
                    resolve=None) -> bool:
    """Question shapes a single bound cell cannot answer."""

    if _TWO_PARTIES.search(question):
        return True
    if subject and resolve is not None \
            and counterparties(question, subject, resolve):
        return True
    return False


def own_label(row: list[str], skip: set[int]) -> str:
    """The row's name, taken from its first text column rather than its longest.

    A note that keys rows by two things — counterparty and transaction — prints the key
    first. Reading the longest cell instead picks whichever of the two happens to be
    wordier, which is how a revenue question was answered from a related-party table.
    """

    for column, cell in enumerate(row):
        if column in skip:
            continue
        text = str(cell).strip()
        if not text or _NUMERIC_CELL.match(text):
            continue
        if _MARKER.match(text):
            # `I.` or `3` is a marker; the name is in the next column.
            continue
        return text
    return ""


def total_column(grid: list[list[str]], allowed: set[int] | None) -> int | None:
    """The column a segment note marks as the total, if it marks one."""

    width = max((len(row) for row in grid), default=0)
    for column in range(width - 1, -1, -1):
        if allowed is not None and column not in allowed:
            continue
        header = " ".join(str(row[column]) for row in grid[:2]
                          if column < len(row))
        if _TOTAL_HEADER.search(header):
            return column
    return None


def question_shape(question: str) -> str | None:
    """`balance`, `flow`, or None when the wording does not say."""

    if _ASKS_BALANCE.search(question):
        return "balance"
    if _ASKS_FLOW.search(question):
        return "flow"
    return None


def column_shape(grid: list[list[str]], column: int) -> str | None:
    """What the column header says it holds, or None when it says nothing."""

    header = " ".join(str(row[column]) for row in grid[:2]
                      if column < len(row))
    if _DATE_HEADER.search(header):
        return "balance"
    flat = header.casefold()
    if "n\u0103m nay" in flat or "n\u0103m tr\u01b0\u1edbc" in flat or "k\u1ef3 n\u00e0y" in flat:
        return "flow"
    return None


def _scale_of_column(grid: list[list[str]], column: int) -> float | None:
    """The unit printed above this column, which beats anything table-wide."""

    from apply_picks import scale_of_column

    return scale_of_column(grid, column)


class LabelBinder:
    """Caches, per company-year-scope, the rows a set of label variants can reach."""

    def __init__(self, doc_scale: dict[str, float] | None = None) -> None:
        self.doc_scale = doc_scale or {}
        self._tables: dict[tuple[str, str, str], list[tuple]] = {}

    def _load(self, ticker: str, year: str, scope: str) -> list[tuple]:
        key = (ticker, year, scope)
        if key in self._tables:
            return self._tables[key]
        out: list[tuple] = []
        root = ROOT / "data" / "official_corpus" / ticker / year
        if root.is_dir():
            for doc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
                doc = doc_dir.name
                if scope == "consolidated" and "consolidated" not in doc \
                        and "aggregated" not in doc:
                    continue
                if scope == "separate" and "separate" not in doc:
                    continue
                tables_dir = doc_dir / f"{doc}_extracted_tables"
                if not tables_dir.is_dir():
                    continue
                grids: dict[int, list[list[str]]] = {}
                for csv_path in sorted(tables_dir.glob("table_*.csv")):
                    try:
                        with csv_path.open(encoding="utf-8-sig", newline="") as file:
                            grids[int(csv_path.stem.split("_")[-1])] = list(
                                csv_mod.reader(file))
                    except OSError:
                        continue
                anchors = contexts_of(doc_dir / f"{doc}_extracted.txt")
                by_number: dict[str, list[int]] = {}
                for table_id in grids:
                    for number in heading_numbers(anchors.get(table_id, "")):
                        by_number.setdefault(number, []).append(table_id)
                # Which tables a statement line points at, so a note row can be
                # reached the way the report says to reach it.
                pointed: set[int] = set()
                for table_id, grid in grids.items():
                    code_col, pointer_col = locate_columns(grid)
                    if code_col is None or pointer_col is None:
                        continue
                    for row in grid[1:]:
                        if len(row) <= pointer_col:
                            continue
                        candidate = str(row[pointer_col]).strip()
                        if POINTER_RE.match(candidate):
                            pointed.update(by_number.get(
                                re.sub(r"[\s.]", "", candidate).upper(), []))
                for table_id, grid in grids.items():
                    if not grid:
                        continue
                    out.append((doc, table_id, grid, table_id in pointed))
        self._tables[key] = out
        return out

    def bind(self, ticker: str, year: str, scope: str, names: list[str],
             slot: int = 0, min_score: float = MIN_SCORE,
             shape: str | None = None,
             question: str | None = None) -> tuple[float, dict] | None:
        """The best-matching row's figure in đồng, or None when nothing is certain.

        `slot` picks the period: a statement prints the current period first and the
        prior one second, so slot 0 is `cuối năm`/`năm nay` and slot 1 is `đầu
        năm`/`năm trước`. `min_score` is the label-match floor; read by hand, correct
        picks sit above 3000 and wrong ones below 2100.
        """

        if not names:
            return None
        candidates: list[tuple[float, float, dict]] = []
        for doc, table_id, grid, pointed in self._load(ticker, year, scope):
            code_col, pointer_col = locate_columns(grid)
            skip = set()
            if code_col is not None:
                skip.add(code_col)
            if pointer_col is not None:
                skip.add(pointer_col)
            if bucket_table(grid) and not _ASKS_MATURITY.search(question or ""):
                continue
            if currency_table(grid) and not _ASKS_CURRENCY.search(question or ""):
                continue
            if party_table(grid) and not _ASKS_PARTY.search(question or ""):
                continue
            allowed = money_columns(grid)
            for index, row in enumerate(grid[1:]):
                if not row:
                    continue
                label = own_label(row, skip)
                if not label:
                    continue
                if question and not shares_words(label, question):
                    continue
                unsaid = unsaid_words(label, question) if question else 0
                if unsaid >= 3:
                    continue
                value = score(names, section_label(grid, index, skip))
                # A statement prints sub-lines as `- Trong đó: ...`, which the scorer reads
                # as a loose match. Admit them lower; the vote below decides.
                exact = exact_variant(own_label(row, skip), names)
                floor = min_score - 1000.0 if code_col is not None else min_score
                if value < max(floor, MIN_SCORE):
                    # Equality does not admit a row the floor rejects; it only decides
                    # between rows already admitted. A model-written variant can name a
                    # narrower item exactly, and that is not evidence of anything.
                    continue
                if row_label(row, skip) != own_label(row, skip):
                    # The longest text cell is not the row's key: this is a two-key note
                    # and the match landed on the second key.
                    continue
                slots = []
                for column, cell in enumerate(row):
                    if column in skip:
                        continue
                    if allowed is not None and column not in allowed:
                        continue
                    raw = str(cell).strip()
                    if not raw or CODE_RE.match(raw):
                        continue
                    parsed = ps.parse_vn_number(raw)
                    if parsed is not None:
                        slots.append((column, parsed, raw))
                if not slots:
                    continue
                column, parsed, raw = slots[min(slot, len(slots) - 1)]
                if len(slots) > 2:
                    marked = total_column(grid, allowed)
                    if marked is not None:
                        for candidate, candidate_value, candidate_raw in slots:
                            if candidate == marked:
                                column = candidate
                                parsed = candidate_value
                                raw = candidate_raw
                                break
                if shape is not None:
                    said = column_shape(grid, column)
                    if said is not None and said != shape:
                        # The question wants a balance and the column holds a flow, or
                        # the reverse. The label matched; the cell is still the wrong one.
                        continue
                scale = _scale_of_column(grid, column)
                if scale is None:
                    scale = scale_from_headers(grid)
                if scale is None:
                    scale = self.doc_scale.get(doc)
                if scale is None:
                    # An unknown unit cannot be compared across years; refusing keeps
                    # an argmax from ranking đồng against triệu đồng.
                    continue
                # A row the report itself points at outranks one merely matched.
                # A Mã số column marks one of the three statements, whose line is the
                # reported total; a note repeating the same label gives a component of it.
                rank = (value + (500.0 if pointed else 0.0)
                        + (1500.0 if code_col is not None else 0.0)
                        + (3000.0 if exact else 0.0))
                candidates.append((rank, parsed * scale, {
                    "doc": doc, "table_id": table_id, "row": index,
                    "col": column, "scale": scale, "raw": raw,
                    "label": row_label(row, skip), "score": value,
                    "pointed": pointed,
                    # Admitted below the caller's floor on the strength of being a
                    # statement line, so it may join a vote but never win alone.
                    "below_floor": value < min_score and not exact,
                    "exact": exact, "unsaid": unsaid,
                }))
        return self._decide(candidates)

    @staticmethod
    def _decide(candidates: list[tuple[float, float, dict]]
                ) -> tuple[float, dict] | None:
        """The figure the most separate tables agree on, ties broken by match quality."""

        if not candidates:
            return None
        # Grouped on magnitude: a statement prints an expense as a deduction, `(243.685...)`,
        # and the note detailing it prints the same figure unsigned. Treating those as
        # different numbers is what stopped them corroborating each other.
        tables: dict[float, set[tuple[str, int]]] = {}
        for _rank, value, meta in candidates:
            key = round(abs(value), 2)
            tables.setdefault(key, set()).add((meta["doc"], meta["table_id"]))
        best = None
        for rank, value, meta in candidates:
            if meta.get("below_floor"):
                # Admitted only to corroborate; it never speaks for itself.
                continue
            votes = len(tables[round(abs(value), 2)])
            # Within one magnitude, prefer the unsigned reading. Measured on the corpus,
            # only 9.8% of the tables the organisers' generator can draw a fact from are
            # statements — the rest are notes, which print a figure without the
            # parentheses a statement puts around a deduction. So the positive reading is
            # the likelier gold by roughly nine to one.
            key = (votes, -meta.get("unsaid", 0), value > 0, rank)
            if best is None or key > best[0]:
                best = (key, votes, rank, value, meta)
        if best is None:
            return None
        # A tie at the top between different figures is not an answer. Two rows that
        # match equally well and disagree mean the question does not identify one cell,
        # and PC1 id=49 had three such rows in three separate cost notes.
        rivals = {round(abs(value), 2) for _rank, value, meta in candidates
                  if not meta.get("below_floor")
                  and (len(tables[round(abs(value), 2)]),
                       -meta.get("unsaid", 0), value > 0, _rank) == best[0]}
        if len(rivals) > 1:
            return None
        meta = dict(best[4])
        meta["votes"] = best[1]
        return best[3], meta
