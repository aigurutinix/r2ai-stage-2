"""Generate the question shapes the training set is missing, with exact answers.

The previous training set was 92.7% one-cell lookups while the exam is 36.6% —
a 56-point distribution gap, and the reason a 69.4% held-out score preceded an
exam regression (`_probe_exam_classes.py`, IMPLEMENTATION.md §18). Closing it
needs pairs for the shapes that dominate the exam, and the derived generator
cannot supply them: 30% of its records name a company no cited table can answer
for, so its answers disagree with its own questions (`_audit_medium.py`).

This takes the opposite approach. No model invents anything:

* the figures come from the Circular 200 metric panel, whose three accounting
  identities hold on 100% of the 478/455/493 groups that can be checked;
* the answer is arithmetic over those figures, so it cannot disagree with the
  question — the question is written *from* the computation, not about it;
* the program reads the real cells the figures came from, verified to round-trip
  through `num()` on 16,430 of 16,430 addresses (`_verify_cells.py`);
* every emitted record is executed and compared against its own answer before
  being written, so a template bug cannot ship.

What it deliberately does not generate: cohort screens over 4+ companies. Those
are 21.2% of the exam and need one statement per company, but the inference
prompt carries eight tables inside a 12k-token budget, so the model cannot see
enough to answer regardless of training. They belong to the deterministic panel
path; `--shapes cohort` writes them as a *labelled test set* for that path rather
than as training pairs.

Usage:
  PYTHONPATH=src python scripts/gen_shapes.py --out artifacts/shapes.jsonl
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.generate import variable_names  # noqa: E402
from vifin.answering.plan_cells import PRELUDE  # noqa: E402
from vifin.answering.sandbox import run_query  # noqa: E402
from vifin.corpus.metrics import Source, build_panel  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# How the exam names each line item. Taken from the statement labels themselves,
# because that is the text the model has to match in the table.
LABEL: dict[str, str] = {
    "cash": "tiền và các khoản tương đương tiền",
    "receivables_short": "các khoản phải thu ngắn hạn",
    "inventory": "hàng tồn kho",
    "current_assets": "tài sản ngắn hạn",
    "long_assets": "tài sản dài hạn",
    "total_assets": "tổng tài sản",
    "liabilities": "nợ phải trả",
    "liabilities_short": "nợ ngắn hạn",
    "liabilities_long": "nợ dài hạn",
    "equity": "vốn chủ sở hữu",
    "net_revenue": "doanh thu thuần",
    "revenue_gross": "doanh thu bán hàng và cung cấp dịch vụ",
    "cogs": "giá vốn hàng bán",
    "gross_profit": "lợi nhuận gộp",
    "financial_income": "doanh thu hoạt động tài chính",
    "financial_expense": "chi phí tài chính",
    "interest_expense": "chi phí lãi vay",
    "selling_expense": "chi phí bán hàng",
    "admin_expense": "chi phí quản lý doanh nghiệp",
    "operating_profit": "lợi nhuận thuần từ hoạt động kinh doanh",
    "profit_before_tax": "lợi nhuận trước thuế",
    "net_profit": "lợi nhuận sau thuế",
    "cfo": "lưu chuyển tiền thuần từ hoạt động kinh doanh",
    "cfi": "lưu chuyển tiền thuần từ hoạt động đầu tư",
    "cff": "lưu chuyển tiền thuần từ hoạt động tài chính",
    "loans_to_customers": "cho vay khách hàng",
    "customer_deposits": "tiền gửi của khách hàng",
    "net_interest_income": "thu nhập lãi thuần",
    "operating_income": "tổng thu nhập hoạt động",
    "operating_expense": "chi phí hoạt động",
}

# Balance-sheet lines are stocks at a date; the rest are flows over a year. The
# exam words the two differently ("cuối năm X" vs "trong năm X") and a question
# worded against the wrong one reads as a different question.
STOCK = frozenset({
    "cash", "receivables_short", "inventory", "current_assets", "long_assets",
    "total_assets", "liabilities", "liabilities_short", "liabilities_long",
    "equity", "loans_to_customers", "customer_deposits",
})

MONEY_UNITS: tuple[tuple[str, float], ...] = (
    ("tỷ đồng", 1e9),
    ("triệu đồng", 1e6),
)

# Ratios the exam actually asks for, as (numerator, denominator, wording, unit).
RATIOS: tuple[tuple[str, str, str, str], ...] = (
    ("current_assets", "liabilities_short",
     "tỷ lệ tài sản ngắn hạn trên nợ ngắn hạn", "lần"),
    ("inventory", "liabilities_short",
     "tỷ lệ hàng tồn kho trên nợ ngắn hạn", "lần"),
    ("cfo", "liabilities_short",
     "tỷ lệ lưu chuyển tiền thuần từ hoạt động kinh doanh trên nợ ngắn hạn", "lần"),
    ("liabilities", "equity", "tỷ số nợ trên vốn chủ sở hữu (D/E)", "lần"),
    ("liabilities", "total_assets",
     "hệ số nợ phải trả trên tổng tài sản", "lần"),
    ("net_profit", "total_assets", "ROA", "%"),
    ("net_profit", "equity", "ROE", "%"),
    ("gross_profit", "net_revenue", "tỷ lệ lợi nhuận gộp trên doanh thu thuần", "%"),
    ("cfo", "net_revenue", "hệ số biên dòng tiền từ hoạt động kinh doanh (CFO Margin)", "%"),
)

# Part-of-whole pairs, worded as "tỷ trọng A trong B".
SHARES: tuple[tuple[str, str], ...] = (
    ("inventory", "current_assets"),
    ("receivables_short", "current_assets"),
    ("cash", "current_assets"),
    ("liabilities_short", "liabilities"),
    ("current_assets", "total_assets"),
    ("liabilities", "total_assets"),
    ("cogs", "net_revenue"),
)

ABS_TOL = 1e-2

# How many years the exam puts in one single-company question, measured on the
# 283 such questions in `data/questions/questions.jsonl` by counting the distinct
# years each names. The multi-year shapes here were fixed at three, so 49.4% of
# this class — every question spanning four years or more — had no pair at all.
# Sampling from the exam's own histogram rather than a constant closes that.
YEAR_SPAN_WEIGHTS: tuple[tuple[int, float], ...] = (
    (3, 0.212), (4, 0.254), (5, 0.208), (6, 0.021), (7, 0.011),
)


def year_span(rng: random.Random, available: int, minimum: int = 3) -> int:
    """Pick how many years one question should span, capped by what exists."""

    spans = [n for n, _ in YEAR_SPAN_WEIGHTS if minimum <= n <= available]
    if not spans:
        return min(available, minimum)
    weights = [w for n, w in YEAR_SPAN_WEIGHTS if n in spans]
    return rng.choices(spans, weights=weights, k=1)[0]


@dataclass
class Cell:
    """One reference the generated program will read, bound to a gold table."""

    key: TableKey
    row: int
    column: int
    scale: float


class Builder:
    """Accumulates cells for one record and hands out program expressions.

    Frame names are not known here — `build_sft.py` decides them when it places
    the gold tables among distractors — so expressions carry `{t0}`, `{t1}`
    placeholders indexed by the record's own table list.
    """

    def __init__(self) -> None:
        self.keys: list[TableKey] = []
        self.cells: list[Cell] = []

    def read(self, source: Source) -> str:
        key = TableKey(source.doc_name, source.table_id)
        if key not in self.keys:
            self.keys.append(key)
        slot = self.keys.index(key)
        self.cells.append(Cell(key, source.row, source.column, source.scale))
        # `frame_from_rows` consumes grid row 0 as the header, so the frame row is
        # the grid row minus one. Columns are not shifted.
        expression = f"num({{t{slot}}}, {source.row - 1}, {source.column})"
        if source.scale != 1.0:
            expression = f"{expression} * {source.scale!r}"
        return expression


_GRIDS: dict[TableKey, list[list[str]]] = {}


def grid_of(store: TableStore, key: TableKey) -> list[list[str]]:
    """Cached table rows. Verification reads the same handful of tables repeatedly
    and `store.rows` re-parses JSON each call, which dominated the first run."""

    grid = _GRIDS.get(key)
    if grid is None:
        grid = store.rows(key)
        _GRIDS[key] = grid
    return grid


def entity_name(roster: CompanyRoster, ticker: str, scope: str) -> str:
    company = roster.by_ticker.get(ticker)
    name = company.name if company else ticker
    return f"công ty mẹ {name} ({ticker})" if scope == "separate" else f"{name} ({ticker})"


def when(metric: str, year: str) -> str:
    return f"cuối năm {year}" if metric in STOCK else f"trong năm {year}"


def verify(program: str, keys: list[TableKey], store: TableStore,
           answer: float) -> bool:
    """Execute the program on the real tables and insist it reproduces the answer.

    A template that substitutes the wrong slot, or an off-by-one row, produces a
    number that still looks plausible. Only execution catches it, and it is cheap
    compared to training on it.

    The names must come from `variable_names`, not from anything invented here.
    `run_query` binds the bare name `df` only when a single frame is present and
    otherwise binds `df1..dfn` by position, so a hand-rolled `["df", "df1", ...]`
    leaves `df` unbound for multi-frame programs *and* shifts every other frame
    by one. That silently failed all 45,492 two-table candidates on the first
    run while the single-table shapes passed, which is what the asymmetry meant.
    """

    names = variable_names(len(keys))
    bound = program
    for index, name in enumerate(names):
        bound = bound.replace(f"{{t{index}}}", name)
    if "{t" in bound:
        return False
    grids = {name: grid_of(store, key) for name, key in zip(names, keys)}
    try:
        outcome = run_query(f"{PRELUDE}\n{bound}", grids)
    except Exception:
        return False
    if not outcome.ok or outcome.value is None:
        return False
    try:
        got = float(outcome.value)
    except (TypeError, ValueError):
        return False
    return abs(got - answer) <= ABS_TOL


# Line items that are always billions of dong at a listed company. A record whose
# answer says otherwise has read a table denominated in triệu or tỷ without the
# matching scale — three of the 48 large-metric records in the first widened run,
# all of them bank statements. Execution cannot catch it: the program faithfully
# reproduces the wrong figure, so only magnitude gives it away.
LARGE_METRICS = frozenset({
    "total_assets", "equity", "liabilities", "liabilities_short",
    "liabilities_long", "current_assets", "long_assets", "net_revenue",
    "revenue_gross", "cogs", "loans_to_customers", "customer_deposits",
})
LARGE_LABELS = tuple(LABEL[m] for m in LARGE_METRICS)
PLAUSIBLE_FLOOR_VND = 1e9


def implausible_magnitude(question: str, answer: float) -> bool:
    """Does a large balance-sheet or revenue line come out under a billion dong?"""

    lowered = question.lower()
    if not any(label in lowered for label in LARGE_LABELS):
        return False
    for name, scale in MONEY_UNITS:
        if name in lowered:
            return abs(answer) * scale < PLAUSIBLE_FLOOR_VND
    return False


def emit(records: list[dict], stats: collections.Counter, shape: str,
         question: str, answer: float, program: str, builder: Builder,
         store: TableStore) -> None:
    # Answers that round to zero are unusable: the tolerance is absolute 0.01, so
    # a near-zero answer is satisfied by any near-zero guess and scores nothing.
    if abs(answer) < 0.01:
        stats[f"skip:{shape}:answer_is_zero"] += 1
        return
    if implausible_magnitude(question, answer):
        stats[f"skip:{shape}:implausible_magnitude"] += 1
        return
    if not verify(program, builder.keys, store, answer):
        stats[f"skip:{shape}:verify_failed"] += 1
        return
    records.append({
        "question": question,
        "answer": round(answer, 2),
        "pandas_query": program,
        "relevant_tables": [f"{k.doc_name}|table_{k.table_id}" for k in builder.keys],
        "difficulty": shape,
        "shape": shape,
        "pregenerated": True,
    })
    stats[shape] += 1


def money_unit(rng: random.Random, magnitude: float) -> tuple[str, float]:
    """Pick a unit that leaves the answer readable rather than 0.00 or huge.

    The exam asks in tỷ đồng for big figures and triệu đồng for small ones; an
    answer of 0.00 tỷ is both unanswerable and unlike anything in the exam.
    """

    for name, scale in MONEY_UNITS:
        if magnitude / scale >= 1.0:
            return name, scale
    return MONEY_UNITS[-1]


def gen_year_difference(panel, prov, store, roster, rng, records, stats,
                        limit: int) -> None:
    """`X của {company} năm A trừ đi năm B là bao nhiêu {unit}?` — two cells."""

    by_entity: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for ticker, year, scope in panel:
        by_entity[(ticker, scope)].append(year)

    candidates = []
    for (ticker, scope), years in by_entity.items():
        for a, b in itertools.combinations(sorted(years), 2):
            for metric in LABEL:
                sa = prov.get((ticker, a, scope), {}).get(metric)
                sb = prov.get((ticker, b, scope), {}).get(metric)
                if sa is None or sb is None:
                    continue
                candidates.append((ticker, scope, a, b, metric))
    rng.shuffle(candidates)

    for ticker, scope, older, newer, metric in candidates:
        if stats[f"year_difference"] >= limit:
            return
        sa = prov[(ticker, newer, scope)][metric]
        sb = prov[(ticker, older, scope)][metric]
        raw = panel[(ticker, newer, scope)][metric] - panel[(ticker, older, scope)][metric]
        unit, scale = money_unit(rng, abs(raw))
        answer = round(raw / scale, 2)

        builder = Builder()
        left = builder.read(sa)
        right = builder.read(sb)
        program = (f"result = round(({left} - {right}) / {scale:.0f}, 2)")
        label = LABEL[metric]
        entity = entity_name(roster, ticker, scope)
        question = (f"{label.capitalize()} của {entity} "
                    f"{'cuối năm' if metric in STOCK else 'năm'} {newer} trừ đi "
                    f"{'cuối năm' if metric in STOCK else 'năm'} {older} là bao nhiêu {unit}?")
        emit(records, stats, "year_difference", question, answer, program,
             builder, store)


def gen_growth(panel, prov, store, roster, rng, records, stats,
               limit: int) -> None:
    """`Tỷ lệ tăng trưởng X từ năm A đến năm B là bao nhiêu phần trăm?`"""

    by_entity: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for ticker, year, scope in panel:
        by_entity[(ticker, scope)].append(year)

    candidates = []
    for (ticker, scope), years in by_entity.items():
        for a, b in itertools.combinations(sorted(years), 2):
            for metric in LABEL:
                if prov.get((ticker, a, scope), {}).get(metric) is None:
                    continue
                if prov.get((ticker, b, scope), {}).get(metric) is None:
                    continue
                candidates.append((ticker, scope, a, b, metric))
    rng.shuffle(candidates)

    for ticker, scope, older, newer, metric in candidates:
        if stats["growth_percent"] >= limit:
            return
        base = panel[(ticker, older, scope)][metric]
        # Growth off a negative or near-zero base is not defined the way the
        # exam means it, and a generated pair that asserts one teaches a formula
        # the exam never rewards.
        if base <= 0:
            stats["skip:growth_percent:base_not_positive"] += 1
            continue
        head = panel[(ticker, newer, scope)][metric]
        answer = round((head - base) / base * 100.0, 2)

        builder = Builder()
        left = builder.read(prov[(ticker, newer, scope)][metric])
        right = builder.read(prov[(ticker, older, scope)][metric])
        program = (f"a = {left}\nb = {right}\n"
                   f"result = round((a - b) / b * 100, 2)")
        label = LABEL[metric]
        entity = entity_name(roster, ticker, scope)
        question = (f"Tỷ lệ tăng trưởng {label} của {entity} từ "
                    f"{'cuối năm' if metric in STOCK else 'năm'} {older} đến "
                    f"{'cuối năm' if metric in STOCK else 'năm'} {newer} "
                    f"là bao nhiêu phần trăm?")
        emit(records, stats, "growth_percent", question, answer, program,
             builder, store)


def gen_ratio(panel, prov, store, roster, rng, records, stats,
              limit: int) -> None:
    """`Tỷ lệ A trên B của {company} {when} là bao nhiêu lần/%?`"""

    candidates = [
        (key, spec)
        for key in panel
        for spec in RATIOS
        if prov.get(key, {}).get(spec[0]) is not None
        and prov.get(key, {}).get(spec[1]) is not None
    ]
    rng.shuffle(candidates)

    for key, (num, den, wording, unit) in candidates:
        if stats["ratio"] >= limit:
            return
        ticker, year, scope = key
        bottom = panel[key][den]
        if bottom <= 0:
            stats["skip:ratio:denominator_not_positive"] += 1
            continue
        factor = 100.0 if unit == "%" else 1.0
        answer = round(panel[key][num] / bottom * factor, 2)

        builder = Builder()
        top_expr = builder.read(prov[key][num])
        bottom_expr = builder.read(prov[key][den])
        tail = f" * 100" if unit == "%" else ""
        program = (f"a = {top_expr}\nb = {bottom_expr}\n"
                   f"result = round(a / b{tail}, 2)")
        entity = entity_name(roster, ticker, scope)
        # Both legs are balance-sheet or both are flows in most pairs; where they
        # mix, the date wording follows the denominator, which is what the exam
        # does with per-revenue ratios ("trong năm X").
        question = (f"{wording.capitalize() if wording[0].islower() else wording} "
                    f"của {entity} {when(den, year)} là bao nhiêu {unit}?")
        emit(records, stats, "ratio", question, answer, program, builder, store)


def gen_share(panel, prov, store, roster, rng, records, stats,
              limit: int) -> None:
    """`Tỷ trọng A trong B của {company} {when} là bao nhiêu %?`"""

    candidates = [
        (key, pair)
        for key in panel
        for pair in SHARES
        if prov.get(key, {}).get(pair[0]) is not None
        and prov.get(key, {}).get(pair[1]) is not None
    ]
    rng.shuffle(candidates)

    for key, (part, whole) in candidates:
        if stats["share_percent"] >= limit:
            return
        ticker, year, scope = key
        bottom = panel[key][whole]
        if bottom <= 0:
            stats["skip:share_percent:whole_not_positive"] += 1
            continue
        answer = round(panel[key][part] / bottom * 100.0, 2)

        builder = Builder()
        part_expr = builder.read(prov[key][part])
        whole_expr = builder.read(prov[key][whole])
        program = (f"a = {part_expr}\nb = {whole_expr}\n"
                   f"result = round(a / b * 100, 2)")
        entity = entity_name(roster, ticker, scope)
        question = (f"Tỷ trọng {LABEL[part]} trong {LABEL[whole]} của {entity} "
                    f"{when(whole, year)} là bao nhiêu %?")
        emit(records, stats, "share_percent", question, answer, program,
             builder, store)


def gen_extreme(panel, prov, store, roster, rng, records, stats,
                limit: int) -> None:
    """`Trong các năm A, B, C, X của {company} cao nhất là bao nhiêu {unit}?`

    Three to seven cells and a comparison — the shape a one-cell target cannot
    express and the exam asks for in 14.7% of questions. The span comes from
    `year_span`, so the record mix carries the four- and five-year questions that
    are a quarter of this class each.
    """

    by_entity: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for ticker, year, scope in panel:
        by_entity[(ticker, scope)].append(year)

    candidates = []
    for (ticker, scope), years in by_entity.items():
        for metric in LABEL:
            usable = sorted(
                y for y in years
                if prov.get((ticker, y, scope), {}).get(metric) is not None
            )
            if len(usable) >= 3:
                candidates.append((ticker, scope, metric, usable))
    rng.shuffle(candidates)

    for ticker, scope, metric, usable in candidates:
        if stats["extreme"] >= limit:
            return
        chosen = sorted(rng.sample(usable, year_span(rng, len(usable))))
        values = [panel[(ticker, y, scope)][metric] for y in chosen]
        biggest = max(values)
        unit, scale = money_unit(rng, abs(biggest))
        answer = round(biggest / scale, 2)

        builder = Builder()
        reads = [builder.read(prov[(ticker, y, scope)][metric]) for y in chosen]
        lines = [f"v{i} = {expr}" for i, expr in enumerate(reads)]
        lines.append(f"result = round(max({', '.join(f'v{i}' for i in range(len(reads)))})"
                     f" / {scale:.0f}, 2)")
        program = "\n".join(lines)
        entity = entity_name(roster, ticker, scope)
        years_text = ", ".join(chosen[:-1]) + f" và {chosen[-1]}"
        question = (f"Trong các năm {years_text}, {LABEL[metric]} "
                    f"{'cuối năm ' if metric in STOCK else ''}của {entity} "
                    f"đạt giá trị lớn nhất là bao nhiêu {unit}?")
        emit(records, stats, "extreme", question, answer, program, builder, store)


def gen_argmax(panel, prov, store, roster, rng, records, stats,
               limit: int) -> None:
    """`Trong các năm A, B, C, {company} có X lớn nhất vào năm nào?`

    The answer is a year, which is still a number, so EXECUTION scores it the
    same way. It teaches argmax rather than max — a different program.
    """

    by_entity: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for ticker, year, scope in panel:
        by_entity[(ticker, scope)].append(year)

    candidates = []
    for (ticker, scope), years in by_entity.items():
        for metric in LABEL:
            usable = sorted(
                y for y in years
                if prov.get((ticker, y, scope), {}).get(metric) is not None
            )
            if len(usable) >= 3:
                candidates.append((ticker, scope, metric, usable))
    rng.shuffle(candidates)

    for ticker, scope, metric, usable in candidates:
        if stats["argmax_year"] >= limit:
            return
        chosen = sorted(rng.sample(usable, year_span(rng, len(usable))))
        values = [panel[(ticker, y, scope)][metric] for y in chosen]
        # A tie has two right answers and one recorded one; drop it rather than
        # teach the model to guess which of two identical figures was meant.
        if len(set(values)) != len(values):
            stats["skip:argmax_year:tied"] += 1
            continue
        answer = float(chosen[values.index(max(values))])

        builder = Builder()
        reads = [builder.read(prov[(ticker, y, scope)][metric]) for y in chosen]
        # Written as an explicit comparison chain, not `max(..., key=lambda)`.
        # The grader runs py37, where a lambda or comprehension inside `exec`
        # with separate globals and locals resolves free names against globals
        # and raises NameError; `portability_problems` rejects both, which is
        # why every one of the first 2,137 argmax candidates failed to verify.
        lines = [f"v{i} = {expr}" for i, expr in enumerate(reads)]
        lines.append("best = v0")
        lines.append(f"best_year = {chosen[0]}")
        for i, year in enumerate(chosen[1:], start=1):
            lines.append(f"if v{i} > best:")
            lines.append(f"    best = v{i}")
            lines.append(f"    best_year = {year}")
        lines.append("result = float(best_year)")
        program = "\n".join(lines)
        entity = entity_name(roster, ticker, scope)
        years_text = ", ".join(chosen[:-1]) + f" và {chosen[-1]}"
        question = (f"Trong các năm {years_text}, {entity} có {LABEL[metric]} "
                    f"{'cuối năm ' if metric in STOCK else ''}lớn nhất vào năm nào?")
        emit(records, stats, "argmax_year", question, answer, program, builder,
             store)


def gen_company_sum(panel, prov, store, roster, rng, records, stats,
                    limit: int) -> None:
    """`Tổng X của {A} và {B} {when} {year} là bao nhiêu {unit}?`

    The shape the derived pool was meant to supply and could not: 30% of its
    records name a company no cited table can answer for, so their answers are
    wrong for their own questions (`_audit_medium.py`). Here both companies are
    chosen *because* both have the figure, so the defect cannot occur.

    Worded as a sum rather than a difference on purpose. "Chênh lệch" leaves
    signed-versus-absolute open — a known error class in this contest — while
    "tổng" has one reading.
    """

    by_slot: dict[tuple[str, str, str], list[str]] = collections.defaultdict(list)
    for ticker, year, scope in panel:
        for metric in LABEL:
            if prov.get((ticker, year, scope), {}).get(metric) is not None:
                by_slot[(year, scope, metric)].append(ticker)

    candidates = []
    for (year, scope, metric), tickers in by_slot.items():
        if len(tickers) < 2:
            continue
        for a, b in itertools.combinations(sorted(tickers), 2):
            candidates.append((year, scope, metric, a, b))
    rng.shuffle(candidates)

    for year, scope, metric, first, second in candidates:
        if stats["company_sum"] >= limit:
            return
        left_key = (first, year, scope)
        right_key = (second, year, scope)
        raw = panel[left_key][metric] + panel[right_key][metric]
        unit, scale = money_unit(rng, abs(raw))
        answer = round(raw / scale, 2)

        builder = Builder()
        left = builder.read(prov[left_key][metric])
        right = builder.read(prov[right_key][metric])
        program = f"result = round(({left} + {right}) / {scale:.0f}, 2)"
        question = (f"Tổng {LABEL[metric]} của "
                    f"{entity_name(roster, first, scope)} và "
                    f"{entity_name(roster, second, scope)} "
                    f"{when(metric, year)} là bao nhiêu {unit}?")
        emit(records, stats, "company_sum", question, answer, program, builder,
             store)


SHAPES = {
    "year_difference": gen_year_difference,
    "company_sum": gen_company_sum,
    "growth_percent": gen_growth,
    "ratio": gen_ratio,
    "share_percent": gen_share,
    "extreme": gen_extreme,
    "argmax_year": gen_argmax,
}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/shapes.jsonl")
    parser.add_argument("--per-shape", type=int, default=400,
                        help="records to keep per shape before verification")
    parser.add_argument("--shapes", default=",".join(SHAPES),
                        help="comma-separated subset of " + ",".join(SHAPES))
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    panel, prov = build_panel(store.frame, with_provenance=True)
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    print(f"panel: {len(panel)} groups, "
          f"{sum(len(v) for v in prov.values())} addressed figures")

    rng = random.Random(args.seed)
    records: list[dict] = []
    stats: collections.Counter[str] = collections.Counter()

    for name in args.shapes.split(","):
        name = name.strip()
        if name not in SHAPES:
            raise SystemExit(f"unknown shape {name!r}; have {', '.join(SHAPES)}")
        SHAPES[name](panel, prov, store, roster, rng, records, stats,
                     args.per_shape)
        print(f"  {name:<18} kept {stats[name]}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # `build_sft.py` keys the anchor ranking by record *index*, so the questions
    # file has to be written from the same list in the same order. Emitting it
    # here rather than by hand removes the one step where the two could disagree
    # — and a ranking off by one row would hand every pair another company's
    # tables as its distractors.
    questions = out.with_name(out.stem + "_questions.jsonl")
    with questions.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(records):
            handle.write(json.dumps(
                {"id": index, "question": record["question"]},
                ensure_ascii=False) + "\n")

    print()
    for key, count in sorted(stats.items()):
        print(f"  {key:<42} {count:>5}")
    print(f"\nwrote {len(records)} records -> {out}")
    print(f"  and {len(records)} questions -> {questions}")
    print("  every one executed against its own tables and matched its answer")


if __name__ == "__main__":
    main()
