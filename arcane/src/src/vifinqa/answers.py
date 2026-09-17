"""Conservative numeric extraction for retrieved OCR evidence."""

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata


NUMBER_RE = re.compile(r"[+-]?\d[\d.,]*")
GROUPED_RE = re.compile(r"\d{1,3}([.,]\d{3})+(?![\d])")
SKIP_HEADERS = ("ma so", "thuyet minh", "stt", "ghi chu")


def parse_ocr_number(value: str) -> float | None:
    """Parse a Vietnamese-formatted numeric OCR cell without guessing text values."""
    match = NUMBER_RE.search(value.replace("(", "-").replace(")", ""))
    if not match:
        return None
    number = match.group(0)
    if "," in number and "." in number:
        number = number.replace(".", "").replace(",", ".")
    elif number.count(".") > 1 or ("." in number and len(number.rsplit(".", 1)[1]) == 3):
        number = number.replace(".", "")
    elif number.count(",") > 1 or ("," in number and len(number.rsplit(",", 1)[1]) == 3):
        number = number.replace(",", "")
    else:
        number = number.replace(",", ".")
    try:
        return float(number)
    except ValueError:
        return None


STRICT_CELL_RE = re.compile(r"^\s*\(?[+-]?\d[\d.,]*\)?%?\s*$")


def strict_number(value: str) -> float | None:
    """Parse only cells that are entirely a number, never a number inside text.

    `parse_ocr_number` reads the leading digits of any cell, so "062 Chi phí quản
    lý doanh nghiệp" binds as 62. The emitted query re-reads the raw string at
    execution time and float() raises on the whole cell — an answer that cannot
    execute is scored as no answer. Binding through here keeps the two agree.
    """
    if not value or not STRICT_CELL_RE.match(value):
        return None
    # Percentage tables use a comma as a decimal separator even when there are
    # three digits after it (99,999%). Currency figures use that same shape as
    # a thousands separator, so the percent sign is the required disambiguator.
    if "%" in value:
        match = NUMBER_RE.search(value)
        if match is None:
            return None
        try:
            return float(match.group(0).replace(".", "").replace(",", "."))
        except ValueError:
            return None
    return parse_ocr_number(value)


def fold(text: str) -> str:
    """Strip Vietnamese diacritics and casefold, matching retrieval-side folding."""
    text = unicodedata.normalize("NFD", text.casefold())
    return "".join(char for char in text if unicodedata.category(char) != "Mn").replace("đ", "d")


YEAR_IN_HEADER_RE = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
# A balance sheet carries "Số cuối năm" beside "Số đầu năm"; both hold real
# figures for the same line item, so only a question cue can tell them apart.
PERIOD_POSITION_CUES = (
    ("cuoi nam", "cuoi ky", "ngay cuoi"), ("dau nam", "dau ky", "ngay dau"),
)


def header_years(header: str) -> set[int]:
    """Fiscal years a column label claims, e.g. '01/01/2023' or '2022'."""
    return {int(value) for value in YEAR_IN_HEADER_RE.findall(header)}


def question_years(question: str) -> list[int]:
    """Years the question names, earliest first."""
    return sorted({int(value) for value in YEAR_IN_HEADER_RE.findall(question)})


def asks_period_position(question: str, position: int) -> bool:
    """Whether the question cues the opening or closing column of a statement.

    `position` 0 means the opening period ("Số đầu năm"), 1 the closing one.
    Questions without either cue accept both, which keeps this from narrowing
    anything that was never ambiguous.
    """
    text = fold(question)
    cues = PERIOD_POSITION_CUES[position]
    if any(f" {cue} " in f" {text} " for cue in cues):
        return True
    return False


def first_numeric_cell(cells: list[str], headers: list[str] | None = None) -> tuple[int, float] | None:
    """Return the value cell after a row label, skipping code/note columns."""
    tiers: list[list[tuple[int, float]]] = [[], [], [], []]
    for column, cell in enumerate(cells[1:], 1):
        if headers and column < len(headers):
            folded = fold(headers[column])
            if any(marker in folded for marker in SKIP_HEADERS):
                continue
        value = parse_ocr_number(cell)
        if value is None:
            continue
        if GROUPED_RE.search(cell):
            tiers[0].append((column, value))
        elif abs(value) >= 1000:
            tiers[1].append((column, value))
        elif "%" in cell or value != int(value):
            tiers[2].append((column, value))
        else:
            tiers[3].append((column, value))
    for tier in tiers:
        if tier:
            return tier[0]
    return None


# Cells are raw VND; 86.3% of questions ask for a scaled unit, and the answer
# tolerance is 0.02%, so an unconverted answer is wrong by six to nine orders of
# magnitude. Longest phrases first: "nghìn tỷ" must win over "tỷ".
OUTPUT_UNITS = (
    ("nghin ty dong", 1e12), ("nghin ty", 1e12),
    ("tram ty dong", 1e11), ("tram ty", 1e11),
    ("ty dong", 1e9), ("ty vnd", 1e9),
    ("trieu dong", 1e6), ("trieu vnd", 1e6),
    ("nghin dong", 1e3), ("nghin vnd", 1e3),
)
PERCENT_CUES = ("phan tram", "%", "diem phan tram")
SOURCE_UNITS = (
    ("nghin ty", 1e12), ("ngan ty", 1e12),
    ("ty dong", 1e9), ("ty vnd", 1e9),
    ("trieu dong", 1e6), ("trieu vnd", 1e6), ("million", 1e6),
    ("nghin dong", 1e3), ("nghin vnd", 1e3), ("ngan dong", 1e3), ("ngan vnd", 1e3),
)


def source_scale(metadata: str, cell: str = "") -> float:
    """Convert a table's stated currency unit to VND without scaling percents."""
    if "%" in cell or "phan tram" in fold(metadata):
        return 1.0
    text = fold(metadata)
    for phrase, scale in SOURCE_UNITS:
        if phrase in text:
            return scale
    return 1.0


def requested_scale(question: str) -> float:
    """The divisor that converts a raw VND figure into the unit the question asks for."""
    text = fold(question)
    for phrase, scale in OUTPUT_UNITS:
        if phrase in text:
            return scale
    return 1.0


def asks_percentage(question: str) -> bool:
    text = fold(question)
    return any(cue in text for cue in PERCENT_CUES)


def asks_for_year(question: str) -> bool:
    text = f" {fold(question)} "
    return any(f" {cue} " in text for cue in ("nam nao", "vao nam nao", "tai nam nao"))


def present(value: float, question: str, *, already_relative: bool = False) -> float:
    """Scale a computed figure into the requested unit and round as the spec asks.

    The organizers round only the final result, to two decimals, and return
    percentages as percentage points. Ratios and growth rates are already
    relative, so they are never divided by a currency scale.
    """
    if not already_relative:
        value = value / requested_scale(question)
    return round(value, 2)


def cell_expression(variable: str, row: int, column: int) -> str:
    """Create a pandas-compatible expression matching common Vietnamese number formatting."""
    value = f"str({variable}.iloc[{row}, {column}])"
    normalized = f"{value}.replace('(', '-').replace(')', '').replace('%', '').replace('.', '').replace(',', '.')"
    return f"result = float({normalized})"


@dataclass(frozen=True)
class EvidenceValue:
    variable: str
    row: int
    column: int
    value: float
    report_id: str
    year: int | None = None
    source_scale: float = 1.0
    label: str = ""


def year_matched_cells(
    cells: list[str], headers: list[str] | None, years: list[int],
) -> list[tuple[int, float, int]]:
    """Numeric cells whose column header names a year the question asks for.

    A growth or multi-year sum is answered from several columns of one row; the
    first-numeric-cell rule can only ever read one. Matching headers to the
    question's years turns a single bound row into every operand the question
    names. Columns matching no requested year are skipped — an unmatched column
    is exactly the wrong-period figure this exists to avoid.
    """
    if not years:
        return []
    wanted = set(years)
    found: dict[int, tuple[int, float, int]] = {}
    for column, cell in enumerate(cells[1:], 1):
        if headers and column < len(headers):
            matches = header_years(headers[column]) & wanted
        else:
            matches = set()
        if not matches:
            continue
        value = parse_ocr_number(cell)
        if value is None:
            continue
        for year in sorted(matches):
            found.setdefault(column, (column, value, year))
    return [found[column] for column in sorted(found)]


def numeric_expression(value: EvidenceValue) -> str:
    """Read a bound cell, in the row space the evaluator's DataFrame will have.

    Our row indices count the parsed grid, whose first row is the table's header.
    `pandas.read_csv` consumes that line as column names, so the same figure sits
    one row earlier in the DataFrame. Emitting the grid index made every query
    read the row below the intended one — which is why answer accuracy rose to
    0.1186 while execution accuracy stayed at 0.004.
    """
    expression = f"str({value.variable}.iloc[{max(0, value.row - 1)}, {value.column}])"
    normalized = (
        f"{expression}.replace('(', '-').replace(')', '').replace('--', '-')"
        f".replace('%', '').replace('.', '').replace(',', '.' if '%' in {expression} else '')"
    )
    parsed = f"float({normalized})"
    return parsed if value.source_scale == 1.0 else f"({parsed} * {value.source_scale:g})"


def select_cell(
    rows: list[list[str]], question_tokens: set[str], headers: list[str] | None = None,
) -> tuple[int, int, float] | None:
    """Find the row whose label best matches the question, then read its value.

    Retrieval binds a row while ranking a whole table, so in principle the bound
    row is the one that made the table look relevant rather than the one asked
    for. Production still reads the retrieval binding; this helper remains a
    deterministic fallback for callers that need row search.

    Ties go to the earlier row, so statements keep their natural order.
    """
    best: tuple[float, int, int, float] | None = None
    for index, row in enumerate(rows):
        if not row:
            continue
        label = " ".join(fold(cell) for cell in row[:2])
        overlap = sum(1 for token in question_tokens if token in label)
        if not overlap:
            continue
        numeric = first_numeric_cell(list(row), headers)
        if numeric is None:
            continue
        column, value = numeric
        score = overlap / max(1, len(question_tokens))
        if best is None or score > best[0]:
            best = (score, index, column, value)
    return (best[1], best[2], best[3]) if best else None


def bound_cells(
    rows: list[list[str]],
    headers: list[str] | None,
    tokens: set[str],
    years: list[int],
) -> list[tuple[int, int]]:
    """Every (row, column) cell a multi-operand plan could need from one table.

    Retrieval binds the single row that made a table look relevant, but a
    formula usually spans several line items — a three-year sum reads one row,
    while a margin reads revenue beside cost. Matching each row's label against
    the question's line-item tokens binds all of them at once; year-matched
    columns then turn each bound row into every period the question names.
    Measured on gold-150: questions with every operand bound rose from 0.5467
    (retrieval binding alone) to 0.68.
    """
    if not tokens:
        return []

    bound: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    def add(row_index: int, column: int) -> None:
        key = (row_index, column)
        if key not in seen:
            seen.add(key)
            bound.append(key)

    for index, row in enumerate(rows):
        if index == 0 or not row:
            continue
        label = " ".join(fold(cell) for cell in row[:2])
        label_words = set(label.split())
        matched = {
            token for token in tokens
            if any(word.startswith(token) or token.startswith(word) for word in label_words)
        }
        required_matches = min(2, len(tokens))
        if len(matched) < required_matches:
            continue
        matched = year_matched_cells(list(row), headers, years)
        for column, _, _ in matched:
            add(index, column)
        numeric = first_numeric_cell(list(row), headers)
        if numeric is not None:
            add(index, numeric[0])
    return bound


def says(text: str, *phrases: str) -> bool:
    """Whole-word phrase match.

    Folding removes the diacritics that separate "hiệu" (difference) from "nhiêu"
    in "bao nhiêu", which ends nearly every question, so a substring test routes
    almost everything to subtraction.
    """
    return any(re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text) for phrase in phrases)


def operator_text(question: str) -> str:
    """The question with its line-item names removed, leaving the arithmetic wording.

    "Tổng phải thu ngắn hạn khác của OGC" is a lookup, not a sum: "tổng" belongs to
    the line item. Matching operator keywords against the raw question routed 106
    such lookups into sums. Stripping the corpus's own row labels first leaves only
    the words that describe an operation.
    """
    text = f" {fold(question)} "
    for label in line_item_phrases():
        if label in text:
            text = text.replace(label, " ")
    # Company and scope boilerplate carries operator homographs of its own:
    # "công ty" holds "cộng"'s folded spelling and "tổng công ty" holds "tổng",
    # so a question naming a parent company read as an arithmetic instruction
    # even after every line item was stripped.
    return without_boilerplate(text)


def without_boilerplate(text: str) -> str:
    """Remove company/report wording that can look like an arithmetic cue."""
    for phrase in sorted(BOILERPLATE_PHRASES, key=len, reverse=True):
        text = text.replace(phrase, " ")
    return text


BOILERPLATE_PHRASES = (
    "cua cong ty me", "cua tong cong ty", "cua ngan hang", "cua cong ty",
    "cong ty me", "cong ty con", "cong ty co phan", "cong ty tnhh",
    "tong cong ty", "cong ty", "ngan hang thuong mai",
    "ngan hang tmcp", "ngan hang", "ctcp", "tnhh", "tmcp",
    "bao cao tai chinh", "bao cao rieng", "bao cao hop nhat",
    "nam tai chinh", "ke toi", "ke tu ngay", "den ngay", "ky ke toan",
)


@lru_cache(maxsize=1)
def line_item_phrases() -> tuple[str, ...]:
    """Corpus row labels, longest first, from scripts/build_line_item_lexicon.py."""
    path = Path(__file__).resolve().parents[2] / "data" / "derived" / "line_items.json"
    if not path.exists():
        return ()
    labels = json.loads(path.read_text(encoding="utf-8"))
    return tuple(sorted(labels, key=lambda label: (-len(label), label)))


def answer_plan(question: str, values: list[EvidenceValue]) -> tuple[float, str] | None:
    """The answer and a Pandas expression that reproduces it.

    Both halves are scored, separately: `answer` against the organizers' figure,
    and the query's own result against it again through execution. So the two
    have to agree, and they did not. `present` rounds to two decimals because the
    scorer compares within 0.02%, while the expression was emitted unrounded — on
    a figure of 1.42 the query returned 1.4229, which is 0.20% out and fails by
    ten times the tolerance. It cost 231 of 1,012 questions: executing the
    shipped queries against their own CSVs reproduced the submitted answer 738
    times, and 967 with the rounding applied.

    Wrapping here rather than in each branch keeps it true of every operation,
    including any added later.
    """
    plan = _unrounded_plan(question, values)
    if plan is None:
        return None
    value, expression = plan
    return value, f"round({expression}, 2)"


def _question_ordered_operands(question: str, values: list[EvidenceValue]) -> list[EvidenceValue]:
    """Order labeled operands by the metric phrases in the question."""
    requested = _question_metric_mentions(question, values)
    if len(requested) < 2:
        return values
    ordered: list[EvidenceValue] = []
    used: set[int] = set()
    for metric in requested:
        matches = [
            (index, value) for index, value in enumerate(values)
            if index not in used and _metric_match(value, metric)
        ]
        if matches:
            index, value = matches[0]
            used.add(index)
            ordered.append(value)
    ordered.extend(value for index, value in enumerate(values) if index not in used)
    return ordered


def _metric_fold(text: str) -> str:
    """Fold common financial abbreviations before matching row labels."""
    return fold(text).replace("tai san co dinh", "tscd")


def _question_metric_mentions(question: str, values: list[EvidenceValue] = ()) -> list[str]:
    """Line-item phrases in question order, dropping nested duplicates."""
    text = _metric_fold(question)
    candidates = {_metric_fold(label) for label in line_item_phrases()}
    # Some generated benchmark metrics are absent from the static corpus
    # lexicon; bound row labels are a safe local vocabulary for those questions.
    candidates.update(_metric_fold(value.label) for value in values if value.label)
    found = sorted(
        ((text.find(label), -len(label), label)
         for label in candidates
         if label and label in text and label not in {"tong cong", "tong cong tai san"}),
        key=lambda item: (item[0], item[1]),
    )
    mentions: list[tuple[int, int, str]] = []
    for start, neg_length, label in found:
        end = start - neg_length
        if any(start >= old_start and end <= old_end
               for old_start, old_end, _ in mentions):
            continue
        mentions.append((start, end, label))
    return [label for _, _, label in mentions]


def _metric_match(value: EvidenceValue, phrase: str) -> bool:
    phrase_tokens = set(phrase.split())
    label_tokens = set(_metric_fold(value.label).split())
    return bool(phrase_tokens) and phrase_tokens <= label_tokens


def _metric_score(value: EvidenceValue, phrase: str) -> tuple[int, int, int]:
    """Prefer a compact row label and earlier evidence for duplicate tables."""
    label = _metric_fold(value.label)
    exact = int(label == phrase)
    # Segment rows are frequent distractors when the question asks total revenue.
    segment_penalty = int("giua cac bo phan" in label or "noi bo" in label)
    return exact, -segment_penalty, -value.row


def _unrounded_plan(question: str, values: list[EvidenceValue]) -> tuple[float, str] | None:
    """Plan only common arithmetic when every operand is source-bound evidence.

    The returned figure is in the unit the question asks for and rounded to two
    decimals; the expression is not, and `answer_plan` rounds it.
    """
    if not values:
        return None
    text = operator_text(question)
    expressions = [numeric_expression(value) for value in values]
    scale = requested_scale(question)

    # Several columns of one bound row can carry the years the question names;
    # when they do, the operands live inside a single table and the year order —
    # not the table's rank order — decides which value is old and which is new.
    grouped: dict[tuple[str, int], list[EvidenceValue]] = {}
    for value in values:
        if value.year is not None:
            grouped.setdefault((value.variable, value.row), []).append(value)
    year_groups = [
        sorted(members, key=lambda value: (value.year, value.column))
        for members in grouped.values() if len({value.year for value in members}) > 1
    ]
    named_years = set(question_years(question))
    # A line item can itself start with an operator word, e.g. "Tổng doanh
    # thu". The lookup safeguard removes that phrase, but for an explicit
    # multi-year question doing so also hides the instruction to aggregate.
    # In that narrow case, recover arithmetic wording after removing only
    # company/report boilerplate; a single-period line-item lookup stays a
    # lookup.
    if len(named_years) > 1:
        text = without_boilerplate(f" {fold(question)} ")
    complete = [
        members for members in year_groups
        if len(named_years) <= 1 or {value.year for value in members} >= named_years
    ]
    if says(text, "tang truong", "toc do tang") and complete:
        members = max(complete, key=lambda members: len(members))
        first, last = members[0], members[-1]
        if first.value == 0:
            return None
        growth = (last.value - first.value) / first.value * 100
        return present(growth, question, already_relative=True), (
            f"(({numeric_expression(last)} - {numeric_expression(first)}) / {numeric_expression(first)} * 100)"
        )
    if says(text, "trung binh", "binh quan") and year_groups:
        members = max(complete or year_groups, key=lambda members: len(members))
        mean = sum(value.value for value in members) / len(members)
        terms = "+".join(numeric_expression(value) for value in members)
        return present(mean, question), f"((({terms}) / {len(members)}) / {scale})"
    if says(text, "tong", "cong", "tong cong") and year_groups and named_years:
        spanning = [members for members in year_groups
                    if {value.year for value in members} == named_years]
        if spanning:
            members = max(spanning, key=lambda members: len(members))
            terms = "+".join(numeric_expression(value) for value in members)
            return present(sum(value.value for value in members), question), (
                f"(({terms}) / {scale})"
            )
    if says(text, "chenh lech", "hieu", "tru", "cao hon", "thap hon", "lon hon") and complete:
        members = max(complete, key=lambda members: len(members))
        # A year-ordered span reads newest minus oldest; the legacy branch below
        # keeps first-minus-rest because its rank order carries no chronology.
        difference = (
            members[-1].value - sum(value.value for value in members[:-1])
            if len(members) > 1 else members[0].value
        )
        head, rest = numeric_expression(members[-1]), [numeric_expression(value) for value in members[:-1]]
        return present(difference, question), (
            f"(({head} - ({' + '.join(rest)})) / {scale})" if rest else f"({head} / {scale})"
        )

    # The year-aware branches above consume every cell they match. What follows
    # is the one-figure-per-row arithmetic, so collapse each bound row to its
    # first value: without this, a row holding three year columns would feed its
    # whole span into a plain sum meant to run across reports.
    singles, seen_rows = [], set()
    for value in values:
        key = (value.variable, value.row)
        if key not in seen_rows:
            seen_rows.add(key)
            singles.append(value)
    values = singles
    values = _question_ordered_operands(question, values)
    expressions = [numeric_expression(value) for value in values]

    mentions = _question_metric_mentions(question, values)
    superlative = says(text, "lon nhat", "cao nhat", "nhieu nhat",
                       "nho nhat", "thap nhat", "it nhat")
    named_years = set(question_years(question))
    explicit_year_selector = says(text, "nam ma", "tai nam ma", "vao nam ma")

    # A common multi-hop form first selects a year by one metric, then asks for
    # another metric in that winning year. Keep the selection metric scoped; the
    # old global max mixed revenue, assets, and ratios into one expression.
    if explicit_year_selector and superlative and len(named_years) > 1 and len(mentions) >= 2:
        primary = [value for value in values
                   if value.year in named_years and _metric_match(value, mentions[0])]
        target = [value for value in values
                  if value.year in named_years and _metric_match(value, mentions[1])]
        years = {value.year for value in primary}
        if len(years) >= 2 and target:
            per_year = {
                year: max((value for value in primary if value.year == year),
                          key=lambda value: _metric_score(value, mentions[0]))
                for year in years
            }
            chooser = max if says(text, "lon nhat", "cao nhat", "nhieu nhat") else min
            winning_year = chooser(per_year, key=lambda year: (per_year[year].value, year))
            target_values = [value for value in target if value.year == winning_year]
            if target_values:
                chosen = max(target_values, key=lambda value: _metric_score(value, mentions[1]))
                return present(chosen.value, question), f"({numeric_expression(chosen)} / {scale})"

    # If the question asks for the largest/smallest amount (not which year),
    # select only the requested metric and the named periods before taking the
    # extremum. This fixes multi-year amount questions without changing the
    # existing year-returning branch.
    if superlative and len(named_years) > 1 and len(mentions) == 1 and not asks_for_year(question):
        candidates = [value for value in values
                      if value.year in named_years and _metric_match(value, mentions[0])]
        if len({value.year for value in candidates}) >= 2:
            chooser = max if says(text, "lon nhat", "cao nhat", "nhieu nhat") else min
            chosen = chooser(candidates, key=lambda value: (value.value, value.year))
            return present(chosen.value, question), f"({numeric_expression(chosen)} / {scale})"

    if asks_for_year(question) and any(value.year is not None for value in values):
        dated = [value for value in values if value.year is not None]
        if says(text, "lon nhat", "cao nhat", "nhieu nhat"):
            chosen = max(dated, key=lambda value: (value.value, value.year))
            expression_op = "max"
        elif says(text, "nho nhat", "thap nhat", "it nhat"):
            chosen = min(dated, key=lambda value: (value.value, value.year))
            expression_op = "min"
        else:
            chosen = None
        if chosen is not None:
            terms = ", ".join(
                f"({numeric_expression(value)}, {value.year})" for value in dated
            )
            return float(chosen.year), f"{expression_op}([{terms}])[1]"

    if len(values) == 1:
        return present(values[0].value, question), f"({expressions[0]} / {scale})"
    if says(text, "tang truong", "toc do tang"):
        if values[0].value == 0:
            return None
        growth = (values[-1].value - values[0].value) / values[0].value * 100
        return present(growth, question, already_relative=True), (
            f"(({expressions[-1]} - {expressions[0]}) / {expressions[0]} * 100)"
        )
    if says(text, "trung binh", "binh quan"):
        mean = sum(value.value for value in values) / len(values)
        return present(mean, question), f"((({' + '.join(expressions)}) / {len(values)}) / {scale})"
    if says(text, "tong", "cong", "tong cong"):
        return present(sum(value.value for value in values), question), (
            f"(({' + '.join(expressions)}) / {scale})"
        )
    if says(text, "chenh lech", "hieu", "tru", "cao hon", "thap hon", "lon hon"):
        difference = values[0].value - sum(value.value for value in values[1:])
        return present(difference, question), (
            f"(({expressions[0]} - ({' + '.join(expressions[1:])})) / {scale})"
        )
    if says(text, "ty le", "ty trong", "gap", "bien loi nhuan", "tren"):
        if values[1].value == 0:
            return None
        multiplier = 100 if asks_percentage(question) else 1
        ratio = values[0].value / values[1].value * multiplier
        return present(ratio, question, already_relative=True), (
            f"({expressions[0]} / {expressions[1]} * {multiplier})"
        )
    if says(text, "lon nhat", "cao nhat", "nhieu nhat"):
        return present(max(value.value for value in values), question), (
            f"(max({', '.join(expressions)}) / {scale})"
        )
    if says(text, "nho nhat", "thap nhat", "it nhat"):
        return present(min(value.value for value in values), question), (
            f"(min({', '.join(expressions)}) / {scale})"
        )
    return None
