"""Extract statutory line items by their Circular 200 statement code.

Vietnamese statutory statements carry a "Mã số" column whose numbers are fixed
by law: 270 is always total assets, 10 is always net revenue. Matching on that
code is far steadier than matching OCR'd Vietnamese text, and 35% of the test
questions ask for one of these headline figures.

Two traps make a naive code lookup wrong:

* The same number means different things in different statements. Code 20 is
  gross profit in the income statement and net operating cash flow in the cash
  flow statement. Every metric therefore carries its statement, and a row is
  only accepted when the code *and* the label agree.
* OCR splits the balance sheet across pages, so assets (…270) and resources
  (300…440) usually land in separate tables. Each table is read independently
  rather than requiring all codes to co-occur.

Approach credit: the idea of keying on Circular 200 codes comes from the public
repository github.com/Kouzira/fin-graphrag. No code is reused from it — that
repository carries no licence — and the code list itself is public law.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from vifin.answering.lookup import (  # noqa: E402
    _parse_cell,
    column_scale,
    label_column,
    value_columns,
)
from vifin.query.companies import strip_tones

BALANCE, INCOME, CASHFLOW = "balance", "income", "cashflow"
# Credit institutions file on the SBV template, not Circular 200: no "Mã số"
# column and a different chart of accounts. They are 39% of the groups the
# Circular 200 catalogue misses, so they get their own statements.
BANK_BALANCE, BANK_INCOME = "bank_balance", "bank_income"


@dataclass(frozen=True, slots=True)
class Metric:
    name: str
    code: str
    statement: str
    aliases: tuple[str, ...]


METRICS: tuple[Metric, ...] = (
    Metric("cash", "110", BALANCE, ("tiền và các khoản tương đương tiền",)),
    Metric("receivables_short", "130", BALANCE, ("các khoản phải thu ngắn hạn",)),
    Metric("inventory", "140", BALANCE, ("hàng tồn kho",)),
    Metric("current_assets", "100", BALANCE, ("tài sản ngắn hạn",)),
    Metric("long_assets", "200", BALANCE, ("tài sản dài hạn",)),
    Metric("total_assets", "270", BALANCE, ("tổng cộng tài sản", "tổng tài sản")),
    Metric("liabilities", "300", BALANCE, ("nợ phải trả",)),
    Metric("liabilities_short", "310", BALANCE, ("nợ ngắn hạn",)),
    Metric("liabilities_long", "330", BALANCE, ("nợ dài hạn",)),
    Metric("equity", "400", BALANCE, ("vốn chủ sở hữu",)),
    Metric("total_resources", "440", BALANCE, ("tổng cộng nguồn vốn", "tổng nguồn vốn")),
    Metric("revenue_gross", "01", INCOME, ("doanh thu bán hàng và cung cấp dịch vụ",)),
    Metric("net_revenue", "10", INCOME, ("doanh thu thuần",)),
    Metric("cogs", "11", INCOME, ("giá vốn hàng bán",)),
    Metric("gross_profit", "20", INCOME, ("lợi nhuận gộp",)),
    Metric("financial_income", "21", INCOME, ("doanh thu hoạt động tài chính",)),
    Metric("financial_expense", "22", INCOME, ("chi phí tài chính",)),
    Metric("interest_expense", "23", INCOME, ("chi phí lãi vay",)),
    Metric("selling_expense", "25", INCOME, ("chi phí bán hàng",)),
    Metric("admin_expense", "26", INCOME, ("chi phí quản lý doanh nghiệp",)),
    Metric("operating_profit", "30", INCOME, ("lợi nhuận thuần từ hoạt động kinh doanh",)),
    Metric("profit_before_tax", "50", INCOME, ("tổng lợi nhuận kế toán trước thuế", "lợi nhuận trước thuế")),
    Metric("net_profit", "60", INCOME, ("lợi nhuận sau thuế",)),
    Metric("eps", "70", INCOME, ("lãi cơ bản trên cổ phiếu",)),
    Metric("cfo", "20", CASHFLOW, ("lưu chuyển tiền thuần từ hoạt động kinh doanh",)),
    Metric("cfi", "30", CASHFLOW, ("lưu chuyển tiền thuần từ hoạt động đầu tư",)),
    Metric("cff", "40", CASHFLOW, ("lưu chuyển tiền thuần từ hoạt động tài chính",)),

    # SBV template. No codes exist, so the empty string keeps the dataclass
    # uniform and matching falls to the anchored label alone.
    Metric("cash", "", BANK_BALANCE, ("tiền mặt",)),
    Metric("deposits_at_sbv", "", BANK_BALANCE, ("tiền gửi tại ngân hàng nhà nước",)),
    Metric("loans_to_customers", "", BANK_BALANCE, ("cho vay khách hàng",)),
    Metric("customer_deposits", "", BANK_BALANCE, ("tiền gửi của khách hàng",)),
    Metric("total_assets", "", BANK_BALANCE, ("tổng tài sản", "tổng cộng tài sản")),
    Metric("liabilities", "", BANK_BALANCE, ("tổng nợ phải trả",)),
    Metric("equity", "", BANK_BALANCE, ("tổng vốn chủ sở hữu", "vốn chủ sở hữu")),
    Metric("net_interest_income", "", BANK_INCOME, ("thu nhập lãi thuần",)),
    Metric("net_fee_income", "", BANK_INCOME, ("lãi thuần từ hoạt động dịch vụ",)),
    Metric("operating_income", "", BANK_INCOME, ("tổng thu nhập hoạt động",)),
    Metric("operating_expense", "", BANK_INCOME, ("chi phí hoạt động",)),
    Metric("provision_expense", "", BANK_INCOME, ("chi phí dự phòng rủi ro tín dụng",)),
    Metric("profit_before_tax", "", BANK_INCOME, ("tổng lợi nhuận trước thuế", "lợi nhuận trước thuế")),
    Metric("net_profit", "", BANK_INCOME, ("lợi nhuận sau thuế",)),
)

BANK_STATEMENTS = frozenset({BANK_BALANCE, BANK_INCOME})

# Two indexes, because bank metrics have no code. Keying everything by code
# collapsed all thirteen SBV line items onto the empty string, leaving one
# survivor per statement and an empty extraction for every bank balance sheet.
BY_STATEMENT: dict[str, list[Metric]] = {}
CODE_INDEX: dict[str, dict[str, Metric]] = {}
for metric in METRICS:
    BY_STATEMENT.setdefault(metric.statement, []).append(metric)
    if metric.code:
        CODE_INDEX.setdefault(metric.statement, {})[metric.code] = metric

MA_SO_RE = re.compile(r"m[aã]\s*s[oố]", re.I)
CODE_RE = re.compile(r"^\d{1,3}$")

# EPS is đồng per share and never carries the table's thousands/millions scale.
UNSCALED = frozenset({"eps"})

# How many catalogue lines a table must carry before its labels are trusted
# without a code column. Below this it is a note or a breakdown, not a statement.
MIN_STATEMENT_LINES = 4
# `VIFIN_FLOW_ONLY=0` disables the narrow non-balance relaxation.
FLOW_ONLY = os.environ.get("VIFIN_FLOW_ONLY", "1") != "0"

# `VIFIN_POOL_ANCHORS=1` counts anchored statement lines across a document's
# fragments rather than within one table. Off until the identity agreement and
# the question count both say it helps.
POOL_ANCHORS = os.environ.get("VIFIN_POOL_ANCHORS") == "1"


# After tone-strip, Vietnamese still keeps ă/â/ê/ô/ơ/ư/đ. OCR often writes the
# plain Latin twin ("NGÁN" for "ngắn"), so catalogue aliases miss unless these
# collapse to ASCII too.
_ASCII_VOWELS = str.maketrans({
    "ă": "a", "â": "a", "ê": "e", "ô": "o", "ơ": "o", "ư": "u", "đ": "d",
})


def _fold(text: str) -> str:
    folded = strip_tones(str(text)).casefold().translate(_ASCII_VOWELS)
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", folded)).strip()


# Circular-200 headline codes are unique within a statement type. When OCR puts
# the name on a previous spanning row and the code row is only "A - (100=…)",
# the label check below would discard a correct figure. Trust these codes alone.
_HEADLINE_CODES = frozenset({
    "100", "200", "270", "300", "310", "330", "400", "440",
    "10", "20", "30", "40", "50", "60",
})


# Statement lines are numbered and lettered: "C. NỢ PHẢI TRẢ", "10. Doanh thu
# thuần", "I. Nợ ngắn hạn". Strip that ordinal prefix before anchoring.
_ORDINAL_PREFIX_RE = re.compile(r"^(?:[0-9ivxIVX]+\s*[.)\-]\s*)+")


def _matches_label(label: str, alias: str) -> bool:
    """True when the row's caption starts with the alias.

    Anchoring matters: "lợi nhuận thuần từ hoạt động kinh doanh trước chi phí
    dự phòng" is a different figure from "chi phí dự phòng", and a substring
    test happily confuses the two.
    """

    stripped = _ORDINAL_PREFIX_RE.sub("", label).strip()
    folded = _fold(alias)
    return stripped.startswith(folded)


# Probes must be folded the same way the labels are. `strip_tones` removes the
# tone but keeps the letter: "thuần" folds to "thuân", not "thuan". Comparing
# against plain ASCII silently matched nothing and left every income and cash
# flow statement unclassified.
_CASHFLOW_PROBES = ("lưu chuyển tiền",)
_INCOME_PROBES = ("doanh thu thuần", "giá vốn hàng bán", "lợi nhuận gộp")
_BALANCE_PROBES = (
    "tổng cộng tài sản", "nợ phải trả", "vốn chủ sở hữu",
    "tài sản ngắn hạn", "tài sản dài hạn",
)


def _probe(labels: str, probes: tuple[str, ...]) -> bool:
    return any(_fold(p) in labels for p in probes)


_BANK_BALANCE_PROBES = ("tiền gửi tại ngân hàng nhà nước", "cho vay khách hàng", "tài sản có khác")
_BANK_INCOME_PROBES = ("thu nhập lãi thuần", "tổng thu nhập hoạt động", "chi phí dự phòng rủi ro tín dụng")

# A bank's liquidity and maturity notes repeat the balance sheet's captions —
# "Tổng tài sản" appears in both — but spread the figures across time buckets
# ("Quá hạn", "Đến 1 tháng", …). A real statement shows two periods, so more
# than three value columns means we are looking at a note.
MAX_STATEMENT_VALUE_COLUMNS = 3


def find_code_column(grid: list[list[str]]) -> int | None:
    """The "Mã số" column: named in the header, or holding mostly short codes."""

    if not grid or not grid[0]:
        return None
    width = len(grid[0])
    for column in range(min(width, 5)):
        header = " ".join(str(row[column]) for row in grid[:2] if column < len(row))
        if MA_SO_RE.search(header):
            return column
    best, best_score = None, 0
    # Column 0 is a legal code column: Circular 200 templates put "Mã số" first
    # and the line label second (VIC, MWG, HSG…). Searching from 1 silently
    # left those statements undetected — `detect_statement` then read the codes
    # as if they were labels and returned None for every such table.
    for column in range(0, min(width, 4)):
        values = [str(row[column]).strip() for row in grid[1:] if column < len(row)]
        score = sum(1 for v in values if CODE_RE.match(v))
        if score > best_score and score >= max(3, len(values) // 3):
            best, best_score = column, score
    return best


def detect_statement(grid: list[list[str]]) -> str | None:
    """Which statement a table belongs to, judged from its own row labels."""

    label_col = label_column(grid)
    labels = " | ".join(
        _fold(row[label_col]) for row in grid[1:]
        if row and label_col < len(row)
    )
    if _probe(labels, _CASHFLOW_PROBES):
        return CASHFLOW
    if _probe(labels, _BANK_INCOME_PROBES):
        return BANK_INCOME
    if _probe(labels, _BANK_BALANCE_PROBES):
        return BANK_BALANCE
    if _probe(labels, _INCOME_PROBES):
        return INCOME
    if _probe(labels, _BALANCE_PROBES):
        return BALANCE
    return None


@dataclass(frozen=True, slots=True)
class Located:
    """A metric together with the cell it was read from.

    The cell is what makes a generated training pair possible: a question whose
    answer comes from the panel can only be taught as a *program* if we can say
    which cell to read, and the panel alone gives the figure without the address.

    `row` indexes the grid, where row 0 is the header. `frame_from_rows` consumes
    that header, so the DataFrame row is `row - 1`; columns are not shifted.
    Conflating the two puts every generated program one row off, silently.
    """

    value: float
    row: int
    column: int
    scale: float


@dataclass(frozen=True, slots=True)
class Source:
    """Where a panel figure came from: which table, and which cell of it.

    Unpacks as `(doc_name, table_id)` so the call sites that only ever wanted the
    table keep working unchanged; the cell is read by name.
    """

    doc_name: str
    table_id: int
    row: int
    column: int
    scale: float

    def __iter__(self):
        yield self.doc_name
        yield self.table_id

    def __getitem__(self, index: int):
        return (self.doc_name, self.table_id)[index]


def extract_located(
    grid: list[list[str]], allow_label_only: bool = False,
    pooled_anchors: int = 0,
) -> dict[str, Located]:
    """Metrics readable from one table's current-period column, with addresses."""

    if len(grid) < 3:
        return {}
    statement = detect_statement(grid)
    if statement is None:
        return {}
    code_column = find_code_column(grid)
    label_col = label_column(grid)
    columns = [
        c for c in value_columns(grid, label_col=label_col)
        if c != code_column and c != label_col
    ]
    if not columns:
        return {}
    if statement in BANK_STATEMENTS and len(columns) > MAX_STATEMENT_VALUE_COLUMNS:
        return {}
    value_column = columns[0]
    scale = column_scale(grid, value_column, "")

    catalogue = BY_STATEMENT[statement]
    codes = CODE_INDEX.get(statement, {})

    # Label matching is safe inside a real statement and reckless outside one.
    # Matching labels anywhere lifted coverage but sank the assets identity from
    # 97.8% to 73.2%; requiring the table to already read like a statement — a
    # handful of catalogue lines anchored at the start of their row — keeps the
    # precision while still reaching statements whose "Mã số" column was lost.
    anchored = 0
    for row in grid[1:]:
        if not row or label_col >= len(row):
            continue
        folded = _fold(row[label_col])
        if any(_matches_label(folded, a) for m in catalogue for a in m.aliases):
            anchored += 1
    # `pooled_anchors` is the same count summed over every table in this document
    # that detection assigned to this same statement. It exists because 83.3% of
    # the 599 company-years the panel cannot serve do have their statement
    # detected and do fail this threshold, with a median of one anchored line per
    # table but six across the document: OCR split one balance sheet into
    # fragments, and each fragment is judged as if it were the whole statement.
    # Pooling is narrower than `allow_label_only`, which admits any table at all —
    # a fragment still has to be typed as this statement to contribute.
    label_ok = (allow_label_only or statement in BANK_STATEMENTS
                or anchored >= MIN_STATEMENT_LINES
                or pooled_anchors >= MIN_STATEMENT_LINES)
    # A narrower relaxation than `allow_label_only`, which admitted any table and
    # sank the assets identity from 97.8% to 73.2%. The cohort-screen questions
    # need per-company `cfo`, `net_profit` and the like, and those are exactly the
    # metrics the panel is thinnest on. None of them appears in the balance-sheet
    # identities, so admitting them on an exact label match cannot make the
    # verifier's job harder — a wrong `cfo` is never caught by it either way, and
    # it was previously simply absent. Balance-sheet lines stay behind the
    # anchored-lines gate.
    exact_only = FLOW_ONLY and not label_ok

    out: dict[str, Located] = {}
    for row_index, row in enumerate(grid[1:], start=1):
        if value_column >= len(row) or label_col >= len(row):
            continue
        label = _fold(row[label_col])
        if not label:
            continue

        metric = None
        if code_column is not None and code_column < len(row):
            raw_code = str(row[code_column]).strip()
            code_key = raw_code.lstrip("0") or "0"
            metric = codes.get(code_key) or codes.get(raw_code)
            # The code alone is not enough: OCR drops digits and note numbers
            # share the column. Require the label to agree before trusting it,
            # except for statutory headline totals unique within the statement.
            if metric is not None and not any(_fold(a) in label for a in metric.aliases):
                if code_key not in _HEADLINE_CODES and raw_code not in _HEADLINE_CODES:
                    metric = None

        if metric is None and exact_only:
            # Exact, whole-label match only: no substring, no fuzzy. "Lưu chuyển
            # tiền thuần từ hoạt động kinh doanh" qualifies; "Tiền" does not.
            for candidate in catalogue:
                if candidate.name in BALANCE_METRICS:
                    continue
                if any(_fold(alias) == label for alias in candidate.aliases):
                    metric = candidate
                    break
        if metric is None and label_ok:
            # Tried and rejected as a default: matching on the label alone lifted
            # coverage from 601 to 706 (ticker, year, scope) groups but dropped
            # the assets identity from 97.8% to 73.2%, i.e. nearly every group it
            # added was wrong. A wrong figure is worse than a missing one here,
            # because it feeds the answering step as if it were solid.
            for candidate in catalogue:
                if any(_matches_label(label, alias) for alias in candidate.aliases):
                    metric = candidate
                    break
        if metric is None:
            continue

        parsed = _parse_cell(row[value_column])
        if parsed is None:
            continue
        unscaled = metric.name in UNSCALED
        out.setdefault(metric.name, Located(
            value=parsed if unscaled else parsed * scale,
            row=row_index,
            column=value_column,
            scale=1.0 if unscaled else scale,
        ))
    return out


def extract_table(grid: list[list[str]], allow_label_only: bool = False) -> dict[str, float]:
    """Metrics readable from one table's current-period column.

    Delegates so the addressed and unaddressed readings cannot disagree: a second
    copy of the matching rules would drift the moment one is edited.
    """

    return {
        name: found.value
        for name, found in extract_located(grid, allow_label_only).items()
    }


# Identities every correctly-read balance sheet must satisfy. They cost nothing
# and are the only ground truth available without the organisers' answer key.
IDENTITIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("total_assets", ("current_assets", "long_assets")),
    ("total_resources", ("liabilities", "equity")),
    ("total_assets", ("liabilities", "equity")),
)

IDENTITY_TOLERANCE = 0.02

# Metrics vouched for by the balance sheet identities. Income and cash flow
# lines have no such check, so they stand or fall with the code match alone.
BALANCE_METRICS = frozenset(m.name for m in METRICS if m.statement == BALANCE)


def verify_group(values: dict[str, float]) -> bool:
    """Whether a group's balance-sheet figures are mutually consistent.

    Loosening extraction to reach statements without a code column lifts group
    coverage from 730 to 1001 but drops the resources identity from 100% to 27%
    — the added groups are mostly misreads. Rather than pick a threshold that
    trades one against the other, take the wide net and discard whatever fails
    arithmetic that must hold.
    """

    for left, right in IDENTITIES:
        if left not in values or not all(r in values for r in right):
            continue
        total = sum(values[r] for r in right)
        if abs(values[left] - total) > IDENTITY_TOLERANCE * max(abs(values[left]), 1.0):
            return False
    # Nothing checkable means nothing disproved.
    return True


# Filled in only when both operands are present and the group has already passed
# verification, so a derived figure never props up an identity that vouches for
# it. `liabilities` alone was the missing metric in 66 otherwise-answerable
# questions, and it is a subtraction away from figures we already trust.
DERIVATIONS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("liabilities", "+", ("liabilities_short", "liabilities_long")),
    ("liabilities", "-", ("total_resources", "equity")),
    ("equity", "-", ("total_resources", "liabilities")),
    ("total_assets", "+", ("current_assets", "long_assets")),
    # Inverse of the assets identity: OCR often names long assets / totals but
    # drops the "A. Tài sản ngắn hạn" caption on the code-100 row.
    ("current_assets", "-", ("total_assets", "long_assets")),
    ("liabilities_short", "-", ("liabilities", "liabilities_long")),
    ("long_assets", "-", ("total_assets", "current_assets")),
)


def derive(values: dict[str, float]) -> None:
    for target, op, operands in DERIVATIONS:
        if target in values or not all(o in values for o in operands):
            continue
        if op == "+":
            values[target] = sum(values[o] for o in operands)
        else:
            values[target] = values[operands[0]] - values[operands[1]]


def build_panel(frame, allow_label_only: bool = False, verify: bool = True,
                with_provenance: bool = False) -> dict[tuple[str, str, str], dict[str, float]]:
    """Metric panel keyed by (ticker, year, scope)."""

    panel: dict[tuple[str, str, str], dict[str, float]] = {}
    provenance: dict[tuple[str, str, str], dict[str, Source]] = {}

    # First pass, only when pooling is on: how many catalogue lines each document
    # anchors for each statement, across all of its fragments. Off by default —
    # this widens what counts as a statement, and the last change that widened it
    # cost the assets identity 24 points, so it ships only once measured.
    pooled: dict[tuple[str, str], int] = {}
    if POOL_ANCHORS:
        for row in frame.itertuples():
            if not row.eligible:
                continue
            grid = json.loads(row.rows_json)
            if len(grid) < 3:
                continue
            statement = detect_statement(grid)
            if statement is None:
                continue
            catalogue = BY_STATEMENT[statement]
            count = 0
            for line in grid[1:]:
                if not line:
                    continue
                folded = _fold(line[0])
                if any(_matches_label(folded, a) for m in catalogue for a in m.aliases):
                    count += 1
            key = (str(row.doc_name), str(statement))
            pooled[key] = pooled.get(key, 0) + count

    for row in frame.itertuples():
        if not row.eligible:
            continue
        grid = json.loads(row.rows_json)
        anchors = 0
        if POOL_ANCHORS:
            statement = detect_statement(grid) if len(grid) >= 3 else None
            if statement is not None:
                anchors = pooled.get((str(row.doc_name), str(statement)), 0)
        found = extract_located(grid, allow_label_only=allow_label_only,
                                pooled_anchors=anchors)
        if not found:
            continue
        key = (row.ticker, row.year, row.scope)
        bucket = panel.setdefault(key, {})
        source = provenance.setdefault(key, {})
        for name, located in found.items():
            if name not in bucket:
                bucket[name] = located.value
                # Remember which table each figure came from so the submission
                # can cite real evidence instead of a synthesised frame alone —
                # and which cell, so a generated question can be taught as a
                # program rather than as a bare number.
                source[name] = Source(
                    doc_name=row.doc_name,
                    table_id=int(row.table_id),
                    row=located.row,
                    column=located.column,
                    scale=located.scale,
                )

    if verify:
        for key, values in list(panel.items()):
            if verify_group(values):
                continue
            # Keep what the identities cannot speak to; drop the balance sheet
            # figures they contradict.
            kept = {k: v for k, v in values.items() if k not in BALANCE_METRICS}
            if kept:
                panel[key] = kept
                provenance[key] = {k: v for k, v in provenance[key].items() if k in kept}
            else:
                del panel[key]
                provenance.pop(key, None)
    # Derivation runs after the first verification so that pass judges only what
    # was actually read. It then completes triples that were previously too
    # sparse to check, so verify once more: a derived value can only ever make
    # its own identity trivially true, never falsely fail one.
    for values in panel.values():
        derive(values)
    if verify:
        for key, values in list(panel.items()):
            if verify_group(values):
                continue
            kept = {k: v for k, v in values.items() if k not in BALANCE_METRICS}
            if kept:
                panel[key] = kept
                provenance[key] = {k: v for k, v in provenance.get(key, {}).items() if k in kept}
            else:
                del panel[key]
                provenance.pop(key, None)
    return (panel, provenance) if with_provenance else panel
