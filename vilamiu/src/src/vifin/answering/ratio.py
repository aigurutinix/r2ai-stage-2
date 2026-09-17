"""Answer ratio questions by dividing two looked-up figures.

298 questions ask for a proportion, and 120 of the answers currently shipped for
them are impossible by inspection — a percentage reported in the billions, which
happens when a đồng amount is handed back as a rate. The `USE_RATIO` experiment
tried replacing those with a row's share of its column total and measured *worse*
on the leaderboard: plausible is not correct.

The figure being asked for is a quotient, so the fix is to compute the quotient.
150 questions name both operands outright — "tỷ lệ lợi nhuận sau thuế **trên**
doanh thu thuần" — which makes this the same composition as `compose`, along a
third axis: two different metrics in one period rather than one metric across
years or companies.

Named ratios are handled by rewriting rather than by special-casing: ROA becomes
"lợi nhuận sau thuế" over "tổng tài sản" and then follows the identical path. The
table below is deliberately small; every entry earns its place by appearing in the
question set, and anything not in it simply declines.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass

from vifin.answering import lookup as lookup_mod
from vifin.answering.plan_cells import Cell, Plan, compile_plan
from vifin.query.parse import ParsedQuestion
from vifin.store import TableKey, TableStore

# "tỷ lệ X trên Y". The tail is cut at whatever starts the owner clause or the
# question's closing words, since neither belongs to the denominator's name.
RATIO_RE = re.compile(
    r"\b(?:tỷ trọng|tỷ lệ|tỉ lệ|tỷ số|tỉ số|hệ số)\s+(.{3,70}?)\s+"
    r"(?:trên|chia cho|so với|/)\s+(.{3,70}?)"
    r"(?=\s+(?:của|tại|năm|vào|đến|là|đạt)\b|[,?]|$)",
    re.I | re.S,
)

# Named ratios, rewritten into the same (numerator, denominator) shape. Kept to
# what the question set actually asks for — each of these appears at least twice.
FORMULAS: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"biên lợi nhuận gộp", re.I), "lợi nhuận gộp", "doanh thu thuần"),
    (re.compile(r"biên lợi nhuận (?:ròng|thuần)|\bROS\b", re.I),
     "lợi nhuận sau thuế", "doanh thu thuần"),
    (re.compile(r"(?:hệ số|tỷ số|tỉ số) thanh toán hiện hành", re.I),
     "tài sản ngắn hạn", "nợ ngắn hạn"),
    (re.compile(r"hệ số nợ|tỷ lệ nợ trên tổng tài sản", re.I),
     "nợ phải trả", "tổng cộng tài sản"),
)

# Ratios a single quotient cannot express. ROA and ROE were previously listed
# above as "lợi nhuận sau thuế / tổng cộng tài sản" — that divides by the closing
# balance, and the standard definition divides by the average of the opening and
# closing balance. At a 0.02% answer tolerance that is not an approximation, it
# is a wrong answer on every ROA and ROE question.
#
# `avg_den` reads the denominator's row at two columns of the same table: the
# first value column is the closing balance, the second the opening one, which is
# the convention `pick_column` already follows.
COMPOUND: tuple[tuple[re.Pattern[str], str, tuple[str, ...]], ...] = (
    (re.compile(r"(?:hệ số|tỷ số|tỉ số) thanh toán nhanh", re.I),
     "diff_ratio", ("tài sản ngắn hạn", "hàng tồn kho", "nợ ngắn hạn")),
    (re.compile(r"\bROA\b|sinh lời trên (?:tổng )?tài sản", re.I),
     "avg_den", ("lợi nhuận sau thuế", "tổng cộng tài sản")),
    (re.compile(r"\bROE\b|sinh lời trên vốn chủ sở hữu", re.I),
     "avg_den", ("lợi nhuận sau thuế", "vốn chủ sở hữu")),
)


def compound_shape(question: ParsedQuestion) -> tuple[str, tuple[str, ...]] | None:
    for pattern, kind, operands in COMPOUND:
        if pattern.search(question.question):
            return kind, operands
    return None


# Beyond this, the quotient is not a rate but a mis-resolution. Financial ratios
# reach a few hundred percent at the extreme; tens of thousands never.
SANITY_LIMIT = 1000.0


@dataclass(frozen=True, slots=True)
class Ratio:
    numerator: str
    denominator: str
    code: str
    keys: list[TableKey]
    variables: list[str]
    labels: tuple[str, str]
    score: float
    value: float


def shape(question: ParsedQuestion) -> tuple[str, str] | None:
    """`(numerator phrase, denominator phrase)`, from wording or a formula."""

    for pattern, numerator, denominator in FORMULAS:
        if pattern.search(question.question):
            return numerator, denominator
    match = RATIO_RE.search(question.question)
    if match is None:
        return None
    numerator = match.group(1).strip(" ,.;:")
    denominator = match.group(2).strip(" ,.;:")
    if not numerator or not denominator:
        return None
    return numerator, denominator


def eligible(question: ParsedQuestion) -> tuple[str, str] | None:
    """In scope for one company; the period is the latest year the question names.

    Ratios inside a screen ("công ty có ROA cao nhất") need the operand loop in
    `compose`, and this module deliberately does not duplicate it. One company is
    required so both operands come from the same report, which is what makes the
    quotient meaningful.
    """

    if question.target_unit not in ("phan_tram", "lan", "vong"):
        return None
    if len(question.tickers) != 1 or not question.years:
        return None
    return shape(question)


def _lookup(question, metric, store, retriever, top_k):
    """The best (key, Lookup) for one operand, within this company and year."""

    # The period both operands must come from: a quotient across two years is
    # not the figure asked for. Questions naming several years are asking about
    # the latest, the same convention `pick_column` follows inside a table.
    year = max(question.years)
    probe = dataclasses.replace(question, question=metric, years=[year])
    best = None
    for variant in lookup_mod.metric_variants(metric):
        for hit in retriever.search(dataclasses.replace(probe, question=variant),
                                    top_k=top_k):
            meta = store.meta(hit.key)
            if str(meta.year) != str(year):
                continue
            found = lookup_mod.find(store.rows(hit.key), probe)
            if found is not None and (best is None or found.score > best[1].score):
                best = (hit.key, found)
    return best


def resolve_compound(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    top_k: int = 10,
) -> Ratio | None:
    """Named ratios that need three operands rather than one quotient."""

    scoped = compound_shape(question)
    if scoped is None:
        return None
    kind, operands = scoped

    found = [_lookup(question, name, store, retriever, top_k) for name in operands]
    if any(hit is None for hit in found):
        return None

    used: list[TableKey] = []
    cells, scales, values = [], [], []

    def add(key: TableKey, row: int, column: int) -> None:
        if key not in used:
            used.append(key)
        grid = store.rows(key)
        meta = store.meta(key)
        scales.append(lookup_mod.column_scale(
            grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"))
        values.append(lookup_mod._parse_cell(grid[row][column]) or 0.0)
        cells.append(Cell(used.index(key), row, column))

    if kind == "diff_ratio":
        for key, hit in found:
            add(key, hit.row, hit.column)
        op = "diff_ratio"
    else:
        top_key, top_hit = found[0]
        add(top_key, top_hit.row, top_hit.column)
        den_key, den_hit = found[1]
        columns = lookup_mod.value_columns(store.rows(den_key))
        if len(columns) < 2:
            # No opening balance beside the closing one; averaging is impossible
            # and dividing by the closing balance alone is the bug being fixed.
            return None
        add(den_key, den_hit.row, columns[0])
        add(den_key, den_hit.row, columns[1])
        op = ("ratio_pct_avg_den" if question.target_unit == "phan_tram"
              else "ratio_avg_den")

    names = ["df"] if len(used) == 1 else [f"df{i + 1}" for i in range(len(used))]
    code = compile_plan(Plan(op, tuple(cells)), scales, 1.0, names, None, magnitude=True)

    amounts = [abs(v) * s for v, s in zip(values, scales)]
    if op == "diff_ratio":
        if amounts[2] == 0:
            return None
        reported = (amounts[0] - amounts[1]) / amounts[2]
    else:
        average = (amounts[1] + amounts[2]) / 2.0
        if average == 0:
            return None
        reported = amounts[0] / average * (100.0 if op.endswith("pct_avg_den") else 1.0)
    if abs(reported) > SANITY_LIMIT:
        return None

    return Ratio(
        numerator=operands[0],
        denominator="/".join(operands[1:]),
        code=code,
        keys=used,
        variables=names,
        labels=(found[0][1].label, found[1][1].label),
        score=min(hit.score for _, hit in found),
        value=round(reported, 2),
    )


def resolve(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    top_k: int = 10,
) -> Ratio | None:
    """Divide two looked-up figures, in the unit the question asks for."""

    if question.target_unit in ("phan_tram", "lan", "vong") and \
            len(question.tickers) == 1 and question.years:
        compound = resolve_compound(question, store, retriever, top_k)
        if compound is not None:
            return compound

    scoped = eligible(question)
    if scoped is None:
        return None
    numerator, denominator = scoped
    return resolve_pair(question, numerator, denominator, store, retriever, top_k)


def resolve_pair(
    question: ParsedQuestion,
    numerator: str,
    denominator: str,
    store: TableStore,
    retriever,
    top_k: int = 10,
) -> Ratio | None:
    """The quotient of two named metrics, with the operand phrases supplied.

    Split out of `resolve` so the phrases can come from somewhere other than
    `RATIO_RE`. The regex parses 63 of the 157 in-scope ratio questions and the
    remainder differ only in wording — "chiếm bao nhiêu %", "bằng bao nhiêu % của",
    "tỷ trọng X trong Y" — so the wording is the part worth handing to a model,
    while everything below this line stays deterministic: the label matcher finds
    the rows, `column_scale` fixes the units, and `SANITY_LIMIT` withholds an
    impossible rate. A model asked to pick the cell itself was measured on the
    board on 20/08 at 0.38 questions lost per row changed; asked only to name the
    two metrics, it is doing the job regexes are bad at and tables are not
    involved.
    """

    top = _lookup(question, numerator, store, retriever, top_k)
    bottom = _lookup(question, denominator, store, retriever, top_k)
    if top is None or bottom is None:
        return None
    if top[0] == bottom[0] and top[1].row == bottom[1].row:
        # Both operands matched the same cell: the split produced two readings of
        # one phrase, and the quotient would be a meaningless 1.0.
        return None

    cells, scales, values, used = [], [], [], []
    for key, found in (top, bottom):
        if key not in used:
            used.append(key)
        meta = store.meta(key)
        scales.append(lookup_mod.column_scale(
            store.rows(key), found.column,
            f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
        ))
        values.append(found.value)
        cells.append(Cell(used.index(key), found.row, found.column))

    names = ["df"] if len(used) == 1 else [f"df{i + 1}" for i in range(len(used))]
    op = "ratio_pct" if question.target_unit == "phan_tram" else "ratio"
    # Magnitudes, for the same reason `compose` uses them: a cost printed in
    # parentheses parses negative, and a ratio of a negative to a positive flips
    # sign for a reason that has nothing to do with the business.
    code = compile_plan(Plan(op, tuple(cells)), scales, 1.0, names, None, magnitude=True)

    bottom_amount = abs(values[1]) * scales[1]
    if bottom_amount == 0:
        return None
    computed = abs(values[0]) * scales[0] / bottom_amount
    reported = computed * (100.0 if op == "ratio_pct" else 1.0)
    # A rate of 29,460% is not a near miss; one of the two operands is the wrong
    # row or carries the wrong unit. Declining lets a later branch answer, which
    # is the whole difference from `USE_RATIO`: that one *substituted* a plausible
    # number, this one withholds an impossible one.
    if abs(reported) > SANITY_LIMIT:
        return None
    return Ratio(
        numerator=numerator,
        denominator=denominator,
        code=code,
        keys=used,
        variables=names,
        labels=(top[1].label, bottom[1].label),
        score=min(top[1].score, bottom[1].score),
        value=round(reported, 2),
    )
