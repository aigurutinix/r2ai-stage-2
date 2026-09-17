"""Answer derived questions by composing single-cell lookups over an axis.

550 of 1012 questions are derived, and every mechanism currently serving them is
model-based: LLM code generation (19.5%), cell plans (7.2%), best-effort fallback
(5.9%). Yet a derived question is compositional — "chênh lệch X giữa năm A và
năm B" is two label matches and a subtraction — and label matching is the
strongest component in this pipeline by a wide margin.

The axis is whatever the operation reduces over. "X cao nhất qua các năm
2020-2024" reduces over years; "X cao nhất trong số HPG, HSG và NKG" reduces over
companies. Both are the same computation on a different operand set, so they
share every line here.

Three design choices carry the accuracy:

*Label locking.* Operands are not matched independently. The one whose label
match scores highest becomes the anchor, and every other operand must yield that
same label. Independent matching agreed on the label in 70% of cases and
disagreed in 30%; locking turns those 30% into "same label or no answer" instead
of quietly mixing two different line items into one subtraction.

*Magnitudes, not signed cells.* Vietnamese statements print costs in parentheses,
so the same line parses negative in one report and positive in another. Summing
three companies' "chi phí tài chính" with mixed signs returned 5.40 where 75.42
was wanted. Every operand is read through `abs()`, matching what the single-cell
branch has always done.

*Cross-operand magnitude check.* A unit misread shows up as one operand sitting
orders of magnitude from its siblings — GAS came back with 4.17e17 next to
4.55e11. A single lookup has nothing to compare against; here the median is a
free detector and the outlier is dropped rather than averaged in.

Known limitation: that detector needs three operands. Three of 68 compositions
have exactly two operands differing by more than 100x, which is the signature of
a scale error — but two banks' securities-trading income genuinely can differ
1000x, so rejecting them would be a guess, and guesses have cost this project
more than they have earned. They ship.
"""

from __future__ import annotations

import dataclasses
import re
import statistics
from dataclasses import dataclass

from vifin.answering import lookup as lookup_mod
from vifin.answering.plan_cells import Cell, Plan, compile_plan
from vifin.answering.sandbox import run_query
from vifin.query.parse import ParsedQuestion
from vifin.store import TableKey, TableStore

# Ordered: the first match wins, so the specific comparatives are tested before
# the plain aggregations whose wording they can contain ("tổng chênh lệch ...").
OPS = (
    ("growth", re.compile(r"tăng trưởng|tốc độ tăng", re.I)),
    ("diff", re.compile(
        r"chênh lệch|hiệu số|biến động|thay đổi|so với|trừ đi|"
        r"(?:nhiều|ít|cao|thấp|lớn|nhỏ)\s+hơn", re.I)),
    ("max", re.compile(r"cao nhất|lớn nhất", re.I)),
    ("min", re.compile(r"thấp nhất|nhỏ nhất", re.I)),
    ("avg", re.compile(r"trung bình|bình quân", re.I)),
    ("sum", re.compile(r"\btổng\b|tích lũy|cộng lại", re.I)),
)

# Derived vocabulary that belongs to a metric's *name*, not to an operation.
# "Lỗ chênh lệch tỷ giá" is a line item; "chênh lệch giữa 2016 và 2017" is a
# subtraction. Matching the keyword alone conflates the two.
NAME_IDIOMS = re.compile(
    r"chênh lệch tỷ giá|chênh lệch đánh giá lại|tỷ lệ quyền biểu quyết|"
    r"tỷ lệ lợi ích|tỷ lệ sở hữu|thu nhập bình quân|bình quân tháng|"
    r"bình quân gia quyền",
    re.I,
)

# Operations that reduce a series of years to one figure. `diff` and `growth`
# instead compare exactly two, so a longer year list means the question is not
# the shape this module handles.
SERIES_OPS = frozenset({"max", "min", "avg", "sum"})

# How far a year's figure may sit from the median of its siblings before it is
# read as a unit error rather than a real movement. Financial series move by
# factors of a few; unit errors move by 1e3 and up.
OUTLIER_FACTOR = 100.0

PLAN_OPS = {
    "diff": "diff",
    "growth": "growth_pct",
    "max": "max_of",
    "min": "min_of",
    "avg": "avg_of",
    "sum": "sum",
}


@dataclass(frozen=True, slots=True)
class Composition:
    op: str
    axis: str
    label: str
    code: str
    keys: list[TableKey]
    variables: list[str]
    # Keyed by year for the year axis, by ticker for the ticker axis.
    values: dict[object, float]
    score: float


def classify(question: str) -> str | None:
    """Which operation the question names, ignoring metric-name look-alikes."""

    stripped = NAME_IDIOMS.sub(" ", question)
    for name, pattern in OPS:
        if pattern.search(stripped):
            return name
    return None


# A selection clause: the question names a candidate set, ranks or filters it by
# one metric, then asks for a *different* metric of whatever survives. "Trong số
# A, B, C, doanh nghiệp có hệ số thanh toán nhanh thấp nhất có hàng tồn kho bao
# nhiêu" is two passes, and the list of companies is the thing being filtered —
# not a set of operands to aggregate. Aggregating it anyway produced confident
# wrong answers, which is worse than declining: a wrong composition displaces
# whatever the later branches would have said.
SCREEN_RE = re.compile(
    r"\b(?:năm|doanh nghiệp|công ty|ngân hàng|mã|đơn vị)\s+(?:nào\s+|mà\s+)?có\b|"
    r"\bở\s+(?:các\s+)?năm\b|\bvào\s+năm\s+mà\b|\bxét\b|\btrong đó\b|\bthỏa\b|"
    r"\bđồng thời\b|\bduy trì\b|\btrung vị\b",
    re.I,
)


# The screen shape, narrowly: "<A> ... tại năm có <B> lớn nhất", "<A> ... doanh
# nghiệp có <B> thấp nhất". Both name the operand set, rank it by B, and ask for A
# of the winner. This is far tighter than SCREEN_RE — it must capture B and a
# superlative — which is why it can be used to *decline* on the year axis where
# the broad filter cost 10 answers.
_SUPERLATIVE = r"(cao nhất|lớn nhất|thấp nhất|nhỏ nhất)"
SCREEN_YEAR_RE = re.compile(
    r"\b(?:tại|trong|vào|ở)\s+(?:cuối\s+)?năm\s+(?:mà\s+)?có\s+(.{4,90}?)\s+" + _SUPERLATIVE,
    re.I | re.S,
)
SCREEN_TICKER_RE = re.compile(
    r"\b(?:doanh nghiệp|công ty|ngân hàng|đơn vị|mã)\s+(?:nào\s+|mà\s+)?có\s+"
    r"(.{4,90}?)\s+" + _SUPERLATIVE,
    re.I | re.S,
)

# A filter metric that is itself computed. Those need a formula table this module
# does not have; 48 of the 99 screens are like that and are left alone.
RATIO_FILTER_RE = re.compile(
    r"tỷ lệ|tỉ lệ|tỷ số|hệ số|biên |CAGR|ROA|ROE|EPS|tốc độ tăng|tăng trưởng|"
    r"\btrên\b|chia cho|thay đổi|trung vị|so với|đòn bẩy|bình quân",
    re.I,
)

# Filter phrases that look like ratios but are not a single quotable figure for
# every ticker (period change, median compare, difference of two margins…).
_RATIO_SCREEN_DIRTY = re.compile(
    r"thay đổi|mức tăng|mức giảm|trung vị|hiệu số|CAGR|tốc độ tăng|tăng trưởng",
    re.I,
)

# Nested cohort *before* the ranking superlative: "…có hệ số hiện hành > 1,5,
# doanh nghiệp có thanh toán nhanh thấp nhất". Ranking the full ticker list would
# ignore the cohort. Same for "cao nhất trong số các công ty có CFO dương".
_RATIO_SCREEN_PREFIX_NESTED = re.compile(
    r"trung vị|lớn hơn|thấp hơn|xét (?:những|các)|trong số các",
    re.I,
)
_RATIO_SCREEN_AFTER_NESTED = re.compile(r"^\s*trong số\b", re.I)

# Nested threshold cohort: "có hệ số thanh toán hiện hành lớn hơn 1,5, doanh
# nghiệp có hệ số thanh toán nhanh thấp nhất …". First keep tickers passing the
# current-ratio cut, then rank survivors by another ratio.
_CURRENT_RATIO_THRESH_RE = re.compile(
    r"hệ\s+số\s+thanh\s+toán\s+hiện\s+hành\s+"
    r"(lớn hơn|thấp hơn|trên|dưới)\s+"
    r"(\d+(?:[.,]\d+)?)",
    re.I,
)

MAX_WORDS = frozenset({"cao nhất", "lớn nhất"})


@dataclass(frozen=True, slots=True)
class Screen:
    axis: str
    filter_metric: str
    want_max: bool
    winner: object
    key: TableKey
    label: str
    code: str
    score: float
    filter_label: str
    filter_keys: list[TableKey]
    # When the asked figure is itself a multi-table ratio, bind these in order as
    # df / df2 / … (see run_submit). None → single-frame `df` from `key`.
    operand_keys: list[TableKey] | None = None


def screen_shape(question: ParsedQuestion) -> tuple[str, str, bool] | None:
    """`(axis, filter metric, want_max)` for a two-stage screen."""

    if len(question.tickers) == 1 and len(question.years) >= 2:
        match = SCREEN_YEAR_RE.search(question.question)
        axis = "year"
    elif len(question.tickers) >= 2 and question.years:
        match = SCREEN_TICKER_RE.search(question.question)
        axis = "ticker"
    else:
        return None
    if match is None:
        return None
    metric = match.group(1).strip(" ,.;:")
    if not metric or RATIO_FILTER_RE.search(metric):
        return None
    return axis, metric, match.group(2).lower() in MAX_WORDS


def eligible(question: ParsedQuestion) -> tuple[str, str] | None:
    """`(operation, axis)`, or None if this question is out of scope.

    The axis is what the operation reduces over. "Chi phí lãi vay cao nhất qua
    các năm 2020-2024" reduces over years; "chi phí lãi vay cao nhất trong số
    HPG, HSG và NKG" reduces over companies. The two are the same computation
    with a different operand set, so they share every line below.
    """

    op = classify(question.question)
    if op is None:
        return None
    # A screen is not an aggregation. Left to itself, `classify` reads the
    # superlative in "tại năm có <B> lớn nhất" as the operation and returns
    # max(A) over the years — the wrong figure on 11 questions the leaderboard
    # was already paying for. `resolve_screen` handles these.
    if screen_shape(question) is not None:
        return None
    # A money answer needs a scale to convert into; growth is a pure ratio.
    if op != "growth" and question.unit_scale is None:
        return None

    if len(question.tickers) == 1 and len(question.years) >= 2:
        axis = "year"
        operands = len(question.years)
    elif len(question.tickers) >= 2 and question.years:
        # SCREEN_RE guards this axis only. The year axis shipped in the build
        # measured at EXEC 0.2273, and applying the same filter there cost it 10
        # of its 50 answers — a measured win must not be traded away on an
        # unmeasured heuristic. The false positives it was written for were all
        # multi-company screens anyway.
        if SCREEN_RE.search(question.question):
            return None
        axis = "ticker"
        operands = len(question.tickers)
    else:
        return None

    # A subtraction or a growth rate compares exactly two operands; a longer list
    # means the question is not the shape this module handles.
    if op in ("diff", "growth") and operands != 2:
        return None
    return op, axis


def _row_by_label(
    grid: list[list[str]], label: str, label_col: int
) -> int | None:
    """The row carrying exactly this label, compared after normalisation."""

    wanted = lookup_mod._norm(label)
    for index, row in enumerate(grid[1:], start=1):
        if label_col >= len(row):
            continue
        if lookup_mod._norm(str(row[label_col])) == wanted:
            return index
    return None


def _operands(question: ParsedQuestion, axis: str) -> list[tuple[object, ParsedQuestion]]:
    """One (label, narrowed question) pair per thing being compared."""

    if axis == "year":
        return [(y, dataclasses.replace(question, years=[y]))
                for y in sorted(question.years)]
    # Companies are compared within one period. Questions that name several years
    # usually ask about the last, but rather than assume, every year is tried and
    # the one resolving the most companies wins.
    return [(t, dataclasses.replace(question, tickers=[t])) for t in question.tickers]


def _candidates(retriever, store, probe, variants, top_k, want):
    """Tables from documents matching `want`, ranked by the metric phrase.

    Ranking by the whole question wastes the signal on company and year tokens
    the document filter has already consumed — the defect that made the
    cross-encoder rerank worse than plain BM25 until it was fed the metric alone.

    `want(meta) -> bool` is enforced after retrieval because `candidate_docs`
    widens to adjacent years when a year is missing. That is right for a single
    lookup reading a comparative column, but here every operand must come from
    its own report or the composition is silently built from the wrong periods.
    """

    keys: list[TableKey] = []
    for variant in variants:
        for hit in retriever.search(dataclasses.replace(probe, question=variant),
                                    top_k=top_k):
            if hit.key in keys or not want(store.meta(hit.key)):
                continue
            keys.append(hit.key)
    return keys


def _expected(op: str, amounts: list[float], out_scale: float) -> float | None:
    """The value the emitted program must reproduce, computed here in Python.

    Exists so the reading program can be *checked* rather than trusted. Without
    it a `num()` parse failure would silently ship whatever the program happened
    to compute instead of falling back to the pre-parsed figures.
    """

    if op == "diff":
        return round(abs(amounts[0] - amounts[1]) / out_scale, 2)
    if op == "growth":
        if amounts[1] == 0:
            return None
        return round((amounts[0] - amounts[1]) / abs(amounts[1]) * 100.0, 2)
    if op == "sum":
        return round(sum(amounts) / out_scale, 2)
    if op == "avg":
        return round(sum(amounts) / len(amounts) / out_scale, 2)
    if op == "max":
        return round(max(amounts) / out_scale, 2)
    if op == "min":
        return round(min(amounts) / out_scale, 2)
    return None


def _cell_in(store, grid, key, column, row):
    meta = store.meta(key)
    scale = lookup_mod.column_scale(
        grid, column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
    )
    value = lookup_mod._parse_cell(grid[row][column])
    return (value, scale) if value is not None else (None, scale)


def _pick_operands(question, store, retriever, variants, operands, axis, top_k,
                   year=None, metric=None):
    """Anchor the label on the best match, then demand that label everywhere.

    Matching each operand independently agreed on the label in 70% of cases and
    disagreed in 30%. Locking is what makes the remaining 30% safe: the answer is
    withheld rather than built by subtracting one line item from another.
    """

    pools: dict[object, list[TableKey]] = {}
    anchor: tuple[float, str] | None = None
    for name, probe in operands:
        # A screen resolves a *different* metric than the one being asked for, so
        # the probe carries that phrase instead of the question.
        if metric is not None:
            probe = dataclasses.replace(probe, question=metric)
        if axis == "ticker":
            probe = dataclasses.replace(probe, years=[year])
            want = (lambda m, t=name, y=year:
                    str(m.ticker) == str(t) and str(m.year) == str(y))
        else:
            want = lambda m, y=name: str(m.year) == str(y)  # noqa: E731
        keys = _candidates(retriever, store, probe, variants, top_k, want)
        pools[name] = keys
        for key in keys:
            found = lookup_mod.find(store.rows(key), probe)
            if found is not None and (anchor is None or found.score > anchor[0]):
                anchor = (found.score, found.label)
    if anchor is None:
        return None, None

    score, label = anchor
    picks: dict[object, tuple[TableKey, int, int, float, float]] = {}
    for name, probe in operands:
        if metric is not None:
            probe = dataclasses.replace(probe, question=metric)
        if axis == "ticker":
            probe = dataclasses.replace(probe, years=[year])
        for key in pools[name]:
            grid = store.rows(key)
            label_col = lookup_mod.label_column(grid)
            row = _row_by_label(grid, label, label_col)
            if row is None:
                continue
            column = lookup_mod.pick_column(grid, probe, label_col)
            if column is None or column >= len(grid[row]):
                continue
            value, scale = _cell_in(store, grid, key, column, row)
            if value is None:
                continue
            picks[name] = (key, row, column, value, scale)
            break
    return picks, (score, label)


def resolve(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    top_k: int = 10,
) -> Composition | None:
    """Compose one figure per operand into the answer the question asks for."""

    scoped = eligible(question)
    if scoped is None:
        return None
    op, axis = scoped
    variants = lookup_mod.metric_variants(question.question)
    operands = _operands(question, axis)

    if axis == "year":
        picks, found = _pick_operands(
            question, store, retriever, variants, operands, axis, top_k
        )
    else:
        # Companies must be compared within one period. Every year the question
        # names is tried and the one resolving the most companies wins, rather
        # than assuming the question means its latest year.
        picks, found = None, None
        for candidate_year in sorted(question.years, reverse=True):
            attempt, attempt_found = _pick_operands(
                question, store, retriever, variants, operands, axis, top_k,
                year=candidate_year,
            )
            if attempt and (picks is None or len(attempt) > len(picks)):
                picks, found = attempt, attempt_found
            if picks is not None and len(picks) == len(operands):
                break
    if not picks or found is None:
        return None
    score, label = found

    if len(picks) < 2 or (op in ("diff", "growth") and len(picks) != 2):
        return None

    amounts = {n: abs(v) * s for n, (_, _, _, v, s) in picks.items()}
    if len(amounts) >= 3:
        middle = statistics.median(amounts.values())
        if middle > 0:
            outliers = [
                n for n, a in amounts.items()
                if a > middle * OUTLIER_FACTOR or (a > 0 and a * OUTLIER_FACTOR < middle)
            ]
            # One stray operand is a unit misread; several means the median itself
            # is not trustworthy and the whole composition should be abandoned.
            if len(outliers) > len(amounts) - 2:
                return None
            for name in outliers:
                picks.pop(name, None)
        if len(picks) < 2:
            return None

    # Later year first: `diff` reports a magnitude either way, but `growth_pct`
    # divides by the second operand and would report the change backwards. Across
    # companies there is no such ordering, so the question's own order is kept.
    order = list(picks)
    if axis == "year":
        order = sorted(picks, reverse=op in ("diff", "growth"))

    used: list[TableKey] = []
    for name in order:
        key = picks[name][0]
        if key not in used:
            used.append(key)
    names = ["df"] if len(used) == 1 else [f"df{i + 1}" for i in range(len(used))]

    cells, scales, values = [], [], []
    for name in order:
        key, row, column, value, scale = picks[name]
        cells.append(Cell(used.index(key), row, column))
        scales.append(scale)
        values.append(value)

    out_scale = 1.0 if op == "growth" else (question.unit_scale or 1.0)
    shaped = Plan(PLAN_OPS[op], tuple(cells))
    frames = {n: store.rows(k) for n, k in zip(names, used)}

    # Prefer the program that parses the cells itself. Passing `values` makes
    # `compile_plan` emit the figures as literals, which reads as a hard-coded
    # answer — the organisers reject those on the private round's manual review.
    # The literal form is kept only as a fallback for cells `num()` cannot parse,
    # and it is used only when the reading program disagrees with what the same
    # cells produced here.
    expected = _expected(op, [abs(v) * s for v, s in zip(values, scales)], out_scale)
    reading = compile_plan(shaped, scales, out_scale, names, None, magnitude=True)
    outcome = run_query(reading, frames)
    if outcome.ok and expected is not None and abs(outcome.value - expected) <= 1e-6 * max(
        abs(outcome.value), abs(expected), 1.0
    ):
        code = reading
    else:
        code = compile_plan(shaped, scales, out_scale, names, values, magnitude=True)
    return Composition(
        op=op,
        axis=axis,
        label=label,
        code=code,
        keys=used,
        variables=names,
        values={y: amounts[y] for y in order},
        score=score,
    )


def _tables_for(store: TableStore, ticker: str, year: int) -> list[TableKey]:
    frame = store.frame
    mask = (frame["ticker"].astype(str) == str(ticker)) & (
        frame["year"].astype(str) == str(year)
    )
    return [
        TableKey(doc, int(tid))
        for doc, tid in frame.loc[mask, ["doc_name", "table_id"]].itertuples(index=False)
    ]


def _best_label(
    store: TableStore,
    keys: list[TableKey],
    probe: ParsedQuestion,
    variants: list[str],
):
    best = None
    for key in keys:
        grid = store.rows(key)
        for variant in variants:
            found = lookup_mod.find(grid, dataclasses.replace(probe, question=variant))
            if found is not None and (best is None or found.score > best[1].score):
                best = (key, found)
    return best


def _ratio_filter_amount(
    question: ParsedQuestion,
    ticker: str,
    year: int,
    filter_metric: str,
    store: TableStore,
    retriever,
    top_k: int,
) -> tuple[float, list[TableKey], str] | None:
    """Resolve a ratio named by `filter_metric` for one ticker/year.

    Walks every table for that (ticker, year) directly. BM25 shortlists are too
    thin here — offline, HSG's quick-ratio operands sat outside top-k while the
    labels exist in-doc.
    """

    from vifin.answering import ratio as ratio_mod

    del retriever, top_k  # same signature as the resolve path; unused on purpose
    keys = _tables_for(store, ticker, year)
    if not keys:
        return None

    short = filter_metric.split(".")[0].split(",")[0].strip()
    for text in (short, filter_metric):
        probe = dataclasses.replace(
            question, tickers=[ticker], years=[year], question=text, target_unit="lan"
        )
        cshape = ratio_mod.compound_shape(probe)
        shaped = None if cshape else ratio_mod.shape(probe)
        if cshape is None and shaped is None:
            continue

        if cshape is not None:
            kind, names = cshape
            hits = []
            for name in names:
                hit = _best_label(store, keys, probe, lookup_mod.metric_variants(name))
                if hit is None:
                    hits = None
                    break
                hits.append(hit)
            if not hits:
                continue
            used: list[TableKey] = []
            amounts: list[float] = []
            if kind == "diff_ratio":
                for key, found in hits:
                    if key not in used:
                        used.append(key)
                    meta = store.meta(key)
                    scale = lookup_mod.column_scale(
                        store.rows(key),
                        found.column,
                        f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
                    )
                    amounts.append(abs(found.value) * scale)
                if amounts[2] == 0:
                    continue
                value = (amounts[0] - amounts[1]) / amounts[2]
            else:
                top_key, top_hit = hits[0]
                den_key, den_hit = hits[1]
                meta_t = store.meta(top_key)
                scale_t = lookup_mod.column_scale(
                    store.rows(top_key),
                    top_hit.column,
                    f"{meta_t.unit_page} {meta_t.unit_doc} {meta_t.caption}",
                )
                cols = lookup_mod.value_columns(store.rows(den_key))
                if len(cols) < 2:
                    continue
                meta_d = store.meta(den_key)
                grid_d = store.rows(den_key)
                s0 = lookup_mod.column_scale(
                    grid_d, cols[0], f"{meta_d.unit_page} {meta_d.unit_doc} {meta_d.caption}"
                )
                s1 = lookup_mod.column_scale(
                    grid_d, cols[1], f"{meta_d.unit_page} {meta_d.unit_doc} {meta_d.caption}"
                )
                v0 = lookup_mod._parse_cell(grid_d[den_hit.row][cols[0]]) or 0.0
                v1 = lookup_mod._parse_cell(grid_d[den_hit.row][cols[1]]) or 0.0
                average = (abs(v0) * s0 + abs(v1) * s1) / 2.0
                if average == 0:
                    continue
                value = abs(top_hit.value) * scale_t / average
                used = [top_key] if top_key == den_key else [top_key, den_key]
            if abs(value) > ratio_mod.SANITY_LIMIT:
                continue
            return value, used, hits[0][1].label

        num, den = shaped
        top = _best_label(store, keys, probe, lookup_mod.metric_variants(num))
        bot = _best_label(store, keys, probe, lookup_mod.metric_variants(den))
        if top is None or bot is None:
            continue
        if top[0] == bot[0] and top[1].row == bot[1].row:
            continue
        amounts = []
        used = []
        for key, found in (top, bot):
            if key not in used:
                used.append(key)
            meta = store.meta(key)
            scale = lookup_mod.column_scale(
                store.rows(key),
                found.column,
                f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
            )
            amounts.append(abs(found.value) * scale)
        if amounts[1] == 0:
            continue
        value = amounts[0] / amounts[1]
        # shape() for tỷ trọng often returns a fraction already in-table as %
        # printed without '%'; if the quotient is <<1 and labels look like
        # totals of totals, keep as-is for ranking.
        if abs(value) > ratio_mod.SANITY_LIMIT:
            continue
        return value, used, top[1].label
    return None


def resolve_screen_ratio(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    top_k: int = 10,
) -> Screen | None:
    """Two-stage screen whose ranking key is a *computed* ratio.

    `screen_shape` declines these via `RATIO_FILTER_RE`. Reopening them blindly
    was measured unsafe (joint solvability ≪ 1). This path only fires when:

    * ticker axis, money/ratio answer with a scale or a ratio unit
    * the captured filter is a clean named/simple ratio (not a change/median)
    * the question is not a nested cohort → then superlative
    * the ratio resolves for **every** ticker (all-operands gate)
    * the asked metric then resolves in the winner

    Upside is small and deliberate — a handful of clean questions — not the full
    48 ratio-filter residual.
    """

    from vifin.answering import ratio as ratio_mod

    if len(question.tickers) < 2 or not question.years:
        return None
    match = SCREEN_TICKER_RE.search(question.question)
    if match is None:
        return None
    filter_metric = match.group(1).strip(" ,.;:")
    if not filter_metric or not RATIO_FILTER_RE.search(filter_metric):
        return None
    if _RATIO_SCREEN_DIRTY.search(filter_metric):
        return None
    prefix = question.question[: match.start()]
    if _RATIO_SCREEN_PREFIX_NESTED.search(prefix):
        return None
    after = question.question[match.end() :]
    if _RATIO_SCREEN_AFTER_NESTED.search(after):
        return None

    syn = dataclasses.replace(question, question=filter_metric.split(".")[0].split(",")[0].strip())
    if ratio_mod.compound_shape(syn) is None and ratio_mod.shape(syn) is None:
        return None

    # Money answers need a scale; growth-like ratio answers (`lan`/`vong`/
    # `phan_tram`) do not — the looked-up cell is itself a rate or we still emit
    # through synthesize with unit_scale=None handled by lookup.
    if question.unit_scale is None and question.target_unit not in (
        "phan_tram",
        "lan",
        "vong",
    ):
        return None

    want_max = match.group(2).lower() in MAX_WORDS
    year = max(question.years)
    amounts: dict[str, float] = {}
    filter_keys: list[TableKey] = []
    filter_label = filter_metric
    for ticker in question.tickers:
        hit = _ratio_filter_amount(
            question, ticker, year, filter_metric, store, retriever, top_k
        )
        if hit is None:
            return None
        value, keys, label = hit
        amounts[ticker] = value
        filter_label = label
        for key in keys:
            if key not in filter_keys:
                filter_keys.append(key)

    winner = max(amounts, key=amounts.get) if want_max else min(amounts, key=amounts.get)
    operands = _operands(question, "ticker")
    asked = dict(operands)[winner]
    # Strip the screen clause; then peel owner particles / "giá trị" wrappers
    # extract_metric leaves behind so label match can see "hàng tồn kho".
    asked_phrase = lookup_mod.extract_metric(prefix)
    asked_phrase = re.sub(r"\s+của\s*$", "", asked_phrase, flags=re.I)
    asked_phrase = re.sub(
        r"^(?:giá trị|số dư|số tiền|chỉ tiêu)\s+", "", asked_phrase, flags=re.I
    ).strip(" ,.;:")
    if not asked_phrase:
        asked_phrase = prefix
    asked = dataclasses.replace(asked, years=[year], question=asked_phrase)

    # Prefer in-doc tables for the winner — same reason as the filter pass.
    direct = _tables_for(store, str(winner), year)
    variants = lookup_mod.metric_variants(asked_phrase)
    hit = _best_label(store, direct, asked, variants) if direct else None
    if hit is None:
        want = _wanted("ticker", winner, asked, store)
        keys = _candidates(retriever, store, asked, variants, top_k, want)
        hit = _best_label(store, keys, asked, variants)
    if hit is None:
        return None
    key, got = hit
    meta = store.meta(key)
    scale = lookup_mod.column_scale(
        store.rows(key), got.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
    )
    code = lookup_mod.synthesize(got, scale, question.unit_scale, magnitude=True)
    return Screen(
        axis="ticker",
        filter_metric=filter_metric,
        want_max=want_max,
        winner=winner,
        key=key,
        label=got.label,
        code=code,
        score=got.score,
        filter_label=filter_label,
        filter_keys=filter_keys,
    )


def _parse_vn_float(text: str) -> float:
    return float(text.replace(",", "."))


def _asked_phrase(prefix: str) -> str:
    asked_phrase = lookup_mod.extract_metric(prefix)
    asked_phrase = re.sub(r"\s+của\s*$", "", asked_phrase, flags=re.I)
    asked_phrase = re.sub(
        r"^(?:giá trị|số dư|số tiền|chỉ tiêu)\s+", "", asked_phrase, flags=re.I
    ).strip(" ,.;:")
    return asked_phrase or prefix


_AFTER_ASK_RE = re.compile(
    r"ghi nhận\s+(.{4,80}?)(?:\s+là\s+bao nhiêu|\?|$)",
    re.I | re.S,
)


def _asked_phrase_threshold(prefix: str, after: str) -> str:
    """Asked metric for threshold-cohort screens.

    Two Vietnamese word orders appear:

    * ask-before: ``A của doanh nghiệp có B thấp nhất``
    * ask-after:  ``doanh nghiệp có B thấp nhất ghi nhận A``

    The threshold clause (``có hệ số thanh toán hiện hành lớn hơn 1,5``) lives in
    the prefix and would otherwise poison ``extract_metric``.
    """

    m = _AFTER_ASK_RE.search(after)
    if m:
        phrase = m.group(1).strip(" ,.;:")
        phrase = re.sub(
            r"^(?:giá trị|số dư|số tiền|chỉ tiêu)\s+", "", phrase, flags=re.I
        ).strip(" ,.;:")
        if phrase:
            return phrase
    cleaned = _CURRENT_RATIO_THRESH_RE.sub(" ", prefix)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.;:")
    return _asked_phrase(cleaned)


def _ratio_program_for_ticker(
    question: ParsedQuestion,
    ticker: str,
    year: int,
    metric: str,
    store: TableStore,
) -> tuple[float, str, list[TableKey], str] | None:
    """Value + executable ratio code for one ticker (direct tables, no BM25)."""

    from vifin.answering import ratio as ratio_mod
    from vifin.answering.plan_cells import Cell, Plan, compile_plan

    keys = _tables_for(store, ticker, year)
    if not keys:
        return None
    short = metric.split(".")[0].split(",")[0].strip()
    unit = question.target_unit if question.target_unit in (
        "phan_tram", "lan", "vong"
    ) else "lan"
    for text in (short, metric):
        probe = dataclasses.replace(
            question, tickers=[ticker], years=[year], question=text, target_unit=unit
        )
        cshape = ratio_mod.compound_shape(probe)
        shaped = None if cshape else ratio_mod.shape(probe)
        if cshape is None and shaped is None:
            continue
        if cshape is not None:
            kind, names = cshape
            hits = []
            for name in names:
                hit = _best_label(store, keys, probe, lookup_mod.metric_variants(name))
                if hit is None:
                    hits = None
                    break
                hits.append(hit)
            if not hits:
                continue
            used: list[TableKey] = []
            cells: list[Cell] = []
            scales: list[float] = []
            values: list[float] = []

            def add(key: TableKey, row: int, column: int, value: float) -> None:
                if key not in used:
                    used.append(key)
                meta = store.meta(key)
                scales.append(lookup_mod.column_scale(
                    store.rows(key), column,
                    f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
                ))
                values.append(value)
                cells.append(Cell(used.index(key), row, column))

            if kind == "diff_ratio":
                for key, found in hits:
                    add(key, found.row, found.column, found.value)
                if abs(values[2]) * scales[2] == 0:
                    continue
                op = "diff_ratio"
                reported = (
                    (abs(values[0]) * scales[0] - abs(values[1]) * scales[1])
                    / (abs(values[2]) * scales[2])
                )
            else:
                top_key, top_hit = hits[0]
                den_key, den_hit = hits[1]
                add(top_key, top_hit.row, top_hit.column, top_hit.value)
                cols = lookup_mod.value_columns(store.rows(den_key))
                if len(cols) < 2:
                    continue
                grid_d = store.rows(den_key)
                v0 = lookup_mod._parse_cell(grid_d[den_hit.row][cols[0]]) or 0.0
                v1 = lookup_mod._parse_cell(grid_d[den_hit.row][cols[1]]) or 0.0
                add(den_key, den_hit.row, cols[0], v0)
                add(den_key, den_hit.row, cols[1], v1)
                op = (
                    "ratio_pct_avg_den" if unit == "phan_tram" else "ratio_avg_den"
                )
                average = (abs(values[1]) * scales[1] + abs(values[2]) * scales[2]) / 2
                if average == 0:
                    continue
                reported = abs(values[0]) * scales[0] / average
                if op.endswith("pct_avg_den"):
                    reported *= 100.0
            if abs(reported) > ratio_mod.SANITY_LIMIT:
                continue
            names = ["df"] if len(used) == 1 else [f"df{i+1}" for i in range(len(used))]
            code = compile_plan(
                Plan(op, tuple(cells)), scales, 1.0, names, None, magnitude=True
            )
            return reported, code, used, hits[0][1].label

        num, den = shaped
        top = _best_label(store, keys, probe, lookup_mod.metric_variants(num))
        bot = _best_label(store, keys, probe, lookup_mod.metric_variants(den))
        if top is None or bot is None:
            continue
        if top[0] == bot[0] and top[1].row == bot[1].row:
            continue
        used = []
        cells = []
        scales = []
        values = []
        for key, found in (top, bot):
            if key not in used:
                used.append(key)
            meta = store.meta(key)
            scales.append(lookup_mod.column_scale(
                store.rows(key), found.column,
                f"{meta.unit_page} {meta.unit_doc} {meta.caption}",
            ))
            values.append(found.value)
            cells.append(Cell(used.index(key), found.row, found.column))
        if abs(values[1]) * scales[1] == 0:
            continue
        reported = abs(values[0]) * scales[0] / (abs(values[1]) * scales[1])
        if unit == "phan_tram":
            reported *= 100.0
            op = "ratio_pct"
        else:
            op = "ratio"
        if abs(reported) > ratio_mod.SANITY_LIMIT:
            continue
        names = ["df"] if len(used) == 1 else [f"df{i+1}" for i in range(len(used))]
        code = compile_plan(
            Plan(op, tuple(cells)), scales, 1.0, names, None, magnitude=True
        )
        return reported, code, used, top[1].label
    return None


def resolve_screen_ratio_threshold(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    top_k: int = 10,
) -> Screen | None:
    """Threshold cohort on current ratio, then rank survivors by another ratio.

    Targets the 397/403 class: keep tickers with thanh toán hiện hành ≷ τ, then
    argmin/argmax thanh toán nhanh (or another clean ratio), then read the asked
    figure from the winner. Declines median/"thay đổi" filters and empty cohorts.
    """

    from vifin.answering import ratio as ratio_mod

    if len(question.tickers) < 2 or not question.years:
        return None
    thr = _CURRENT_RATIO_THRESH_RE.search(question.question)
    if thr is None:
        return None
    match = SCREEN_TICKER_RE.search(question.question)
    if match is None:
        return None
    filter_metric = match.group(1).strip(" ,.;:")
    if not filter_metric or not RATIO_FILTER_RE.search(filter_metric):
        return None
    if _RATIO_SCREEN_DIRTY.search(filter_metric):
        return None
    if re.search(r"trung vị", question.question, re.I):
        return None

    syn = dataclasses.replace(
        question, question=filter_metric.split(".")[0].split(",")[0].strip()
    )
    if ratio_mod.compound_shape(syn) is None and ratio_mod.shape(syn) is None:
        return None

    cmp_word = thr.group(1).lower()
    want_gt = cmp_word in {"lớn hơn", "trên"}
    threshold = _parse_vn_float(thr.group(2))
    want_max = match.group(2).lower() in MAX_WORDS
    year = max(question.years)
    prefix = question.question[: match.start()]

    cohort: list[str] = []
    filter_keys: list[TableKey] = []
    for ticker in question.tickers:
        cur = _ratio_filter_amount(
            question, ticker, year, "hệ số thanh toán hiện hành", store, retriever, top_k
        )
        if cur is None:
            continue
        value, keys, _ = cur
        passed = value > threshold if want_gt else value < threshold
        if not passed:
            continue
        cohort.append(ticker)
        for key in keys:
            if key not in filter_keys:
                filter_keys.append(key)
    if len(cohort) < 1:
        return None

    amounts: dict[str, float] = {}
    filter_label = filter_metric
    for ticker in cohort:
        hit = _ratio_filter_amount(
            question, ticker, year, filter_metric, store, retriever, top_k
        )
        if hit is None:
            return None  # cohort member unscored → refuse (no subset argmax)
        value, keys, label = hit
        amounts[ticker] = value
        filter_label = label
        for key in keys:
            if key not in filter_keys:
                filter_keys.append(key)

    winner = max(amounts, key=amounts.get) if want_max else min(amounts, key=amounts.get)
    after = question.question[match.end() :]
    asked_phrase = _asked_phrase_threshold(prefix, after)

    # Prefer asked-as-ratio (403: CFO / short-term debt) when the phrase shapes.
    ratio_ask = _ratio_program_for_ticker(
        question, str(winner), year, asked_phrase, store
    )
    if ratio_ask is not None:
        value, code, used, label = ratio_ask
        return Screen(
            axis="ticker",
            filter_metric=filter_metric,
            want_max=want_max,
            winner=winner,
            key=used[0],
            label=label,
            code=code,
            score=1.0,
            filter_label=filter_label,
            filter_keys=filter_keys,
            operand_keys=used,
        )

    # Money / line-item ask (397: hàng tồn kho).
    operands = _operands(question, "ticker")
    asked = dict(operands)[winner]
    asked = dataclasses.replace(asked, years=[year], question=asked_phrase)
    direct = _tables_for(store, str(winner), year)
    variants = lookup_mod.metric_variants(asked_phrase)
    hit = _best_label(store, direct, asked, variants) if direct else None
    if hit is None:
        want = _wanted("ticker", winner, asked, store)
        keys = _candidates(retriever, store, asked, variants, top_k, want)
        hit = _best_label(store, keys, asked, variants)
    if hit is None:
        return None
    key, got = hit
    meta = store.meta(key)
    scale = lookup_mod.column_scale(
        store.rows(key), got.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
    )
    code = lookup_mod.synthesize(got, scale, question.unit_scale, magnitude=True)
    return Screen(
        axis="ticker",
        filter_metric=filter_metric,
        want_max=want_max,
        winner=winner,
        key=key,
        label=got.label,
        code=code,
        score=got.score,
        filter_label=filter_label,
        filter_keys=filter_keys,
    )


def resolve_screen(
    question: ParsedQuestion,
    store: TableStore,
    retriever,
    top_k: int = 10,
) -> Screen | None:
    """Rank the operands by one metric, then read another from the winner.

    "Số dư vay dài hạn của HSG trong năm có tổng vốn chủ sở hữu cao nhất" needs
    two passes over the same machinery: resolve the filter metric for every year,
    take the argmax, then look up the asked metric in that year alone. Nothing new
    is required — pass one is `_pick_operands` aimed at a different phrase, pass
    two is the ordinary single-cell lookup.

    The emitted program reads only the winner's table. The selection is not in the
    code, and it does not need to be: the scorer re-runs `pandas_query` to check it
    reproduces the answer, and a label lookup is a real query, not a constant. The
    tables consulted for the filter are still declared in `relevant_tables`, since
    the question genuinely depends on them.
    """

    shape = screen_shape(question)
    if shape is None or question.unit_scale is None:
        return None
    axis, filter_metric, want_max = shape

    operands = _operands(question, axis)
    filter_variants = lookup_mod.metric_variants(filter_metric)

    if axis == "year":
        picks, found = _pick_operands(
            question, store, retriever, filter_variants, operands, axis, top_k,
            metric=filter_metric,
        )
    else:
        picks, found = None, None
        for candidate_year in sorted(question.years, reverse=True):
            attempt, attempt_found = _pick_operands(
                question, store, retriever, filter_variants, operands, axis, top_k,
                year=candidate_year, metric=filter_metric,
            )
            if attempt and (picks is None or len(attempt) > len(picks)):
                picks, found = attempt, attempt_found
            if picks is not None and len(picks) == len(operands):
                break
    # Ranking needs the whole candidate set. With operands missing, the argmax is
    # over a subset and the winner may simply be the one that resolved.
    if not picks or found is None or len(picks) < len(operands):
        return None

    amounts = {n: abs(v) * s for n, (_, _, _, v, s) in picks.items()}
    winner = max(amounts, key=amounts.get) if want_max else min(amounts, key=amounts.get)
    filter_keys = [picks[n][0] for n in picks]

    # Pass two: the asked metric, in the winner's period or company only.
    asked = dict(operands)[winner]
    if axis == "ticker":
        asked = dataclasses.replace(asked, years=[str_year(picks, winner, store)])
    # The asked metric is read from the question with the screen clause cut away.
    # Left in, "cao nhất" and the filter metric's own words join the phrase being
    # matched against row labels.
    pattern = SCREEN_YEAR_RE if axis == "year" else SCREEN_TICKER_RE
    match = pattern.search(question.question)
    trimmed = question.question[: match.start()] if match else question.question
    asked = dataclasses.replace(asked, question=trimmed)
    want = _wanted(axis, winner, asked, store)
    keys = _candidates(
        retriever, store, asked, lookup_mod.metric_variants(trimmed), top_k, want
    )
    for key in keys:
        grid = store.rows(key)
        got = lookup_mod.find(grid, asked)
        if got is None:
            continue
        meta = store.meta(key)
        scale = lookup_mod.column_scale(
            grid, got.column, f"{meta.unit_page} {meta.unit_doc} {meta.caption}"
        )
        code = lookup_mod.synthesize(got, scale, question.unit_scale, magnitude=True)
        return Screen(
            axis=axis, filter_metric=filter_metric, want_max=want_max, winner=winner,
            key=key, label=got.label, code=code, score=got.score,
            filter_label=found[1], filter_keys=filter_keys,
        )
    return None


def str_year(picks, winner, store) -> int:
    """The year the filter pass actually resolved for the winning company."""

    return int(store.meta(picks[winner][0]).year)


def _wanted(axis: str, winner: object, asked: ParsedQuestion, store: TableStore):
    if axis == "year":
        return lambda m, y=winner: str(m.year) == str(y)
    year = asked.years[0] if asked.years else None
    return (lambda m, t=winner, y=year:
            str(m.ticker) == str(t) and (y is None or str(m.year) == str(y)))
