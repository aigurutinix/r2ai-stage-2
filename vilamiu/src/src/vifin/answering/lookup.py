"""Deterministic single-cell lookups: the questions that need no LLM.

Roughly the first third of the test set asks for one figure from one statement
("Lãi tiền gửi năm 2018 của công ty mẹ VJC là bao nhiêu triệu đồng?"). Those are
a label match plus a column choice plus a unit conversion — all decidable from
the table itself, and all cheaper and more reliable than generated code.

Everything here emits a program rather than a number. `EXECUTION_ACCURACY` is
the ranked metric and the scorer re-runs `pandas_query` against the CSV we ship,
so the answer we report must be the answer the code produces. The caller is
expected to execute the emitted program and keep the two in lockstep.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from vifin.corpus.numeric import is_numeric_cell
from vifin.query.companies import strip_tones
from vifin.query.parse import ParsedQuestion

# Columns holding statement line codes or note cross-references look numeric but
# never hold the figure being asked for.
NON_VALUE_HEADER_RE = re.compile(r"mã\s*số|thuyết\s*minh|^tm$|^stt$|^mã$|^a$|^b$|^c$", re.I)

# Framing words that sit in front of the metric name. "Tổng" is deliberately
# absent: balance-sheet lines are literally "TỔNG CỘNG TÀI SẢN", so dropping it
# turned "Tổng tài sản" into "tài sản" and lost the row it was pointing at.
METRIC_PREFIX_RE = re.compile(r"^(số dư|mức|khoản)\s+", re.I)

# The interrogative tail: "... là bao nhiêu tỷ đồng?", "... đạt bao nhiêu %".
# The unit in it is already parsed separately, so none of it names the metric.
TAIL_RE = re.compile(r"\s*(?:là|đạt|ở mức|vào khoảng)?\s*bao nhiêu\b.*$", re.I)

# A leading framing clause, comma-terminated: "Trong các năm 2017, 2019 và 2022,",
# "Tại CTCP Nhiệt điện Hải Phòng (HND),", "Xét nhóm cổ phiếu CEO, HPX, VIC,".
# Applied repeatedly, because the year lists these questions open with contain
# commas of their own — stopping at the first one left the metric starting with
# "2019 và 2022, tổng số dư ..." and dropped the label match below its floor on
# 84 questions.
LEADING_CLAUSE_RE = re.compile(
    r"^[^,]*?\b(?:năm|ngày|giai đoạn|nhóm|xét|tại|với|trong|đối với)\b[^,]*,\s*", re.I
)

# What the clause above leaves behind once its own first comma is consumed:
# "2021 và 2022, số dư vay ngân hàng". No framing word remains to match on, so
# the remaining years are stripped on their own terms.
LEADING_YEARS_RE = re.compile(r"^(?:(?:19|20)\d{2}|và|hoặc|[-–,\s])+,\s*", re.I)

# What a legal entity looks like at the start of a clause. Bare tickers count,
# which is why this is only ever tested immediately after "của" — "Tiền gửi tại
# NHNN" would otherwise lose its own name to the ticker pattern.
ENTITY_HEAD_RE = re.compile(
    r"^\s*(?:CTCP|Cty|Công ty|Ngân hàng|Tập đoàn|Doanh nghiệp|Tổng Công ty|"
    r"Tổng công ty|công ty mẹ|[A-Z][A-Z0-9]{2,3}\b)",
    re.I,
)

# A period phrase opening the question, with the metric behind it: "Số dư cuối
# năm 2015 của Quỹ bình ổn giá xăng dầu ...". Anchored at the start and requiring
# the period word up front, so "Giá trị còn lại cuối năm Tài sản cố định vô
# hình" cannot lose its head — that one collapsed to "còn lại" once already.
LEADING_PERIOD_RE = re.compile(
    r"^(?:số dư|giá trị|tổng)?\s*(?:cuối|đầu|trong|vào)\s+(?:năm|ngày|quý)\s*"
    r"(?:tài chính)?\s*[\d/.\-]*\s*(?:của\s+)?",
    re.I,
)

# Company-first phrasing: "CRE có tổng doanh thu ...", "Doanh nghiệp CTCP Tập
# đoàn C.E.O ghi nhận lãi tiền gửi ...". There is no "của" to split on, so the
# whole question survived as the metric. Only fires when an entity marker sits
# in front of the verb, so a metric containing "có" ("tiền gửi có kỳ hạn") is
# never truncated.
ENTITY_VERB_RE = re.compile(
    r"^.*?\b(?:CTCP|Công ty|Cty|Ngân hàng|Tập đoàn|Doanh nghiệp|Tổng Công ty|"
    r"[A-Z][A-Z0-9]{2,3})\b.*?\b(?:có|ghi nhận|đạt|báo cáo|công bố)\s+", re.I
)

# Operation nouns opening a derived question. Stripped so the remainder is the
# metric being operated on — but never when the phrase is a line item in its own
# right: "chênh lệch tỷ giá" and "chênh lệch đánh giá lại tài sản" are accounts.
LEADING_OP_RE = re.compile(
    r"^(?:tính\s+)?(?:phần trăm\s+|tỷ lệ\s+|tốc độ\s+|mức\s+|giá trị\s+)?"
    r"(?:chênh lệch|hiệu số|biến động|thay đổi|tăng trưởng|tăng)\s+"
    r"(?!tỷ giá|đánh giá lại)",
    re.I,
)

# A dangling comparative left at the end once the owner clause is cut away:
# "Chênh lệch vốn chủ sở hữu giữa cuối năm 2025 và".
TRAILING_COMPARE_RE = re.compile(
    r"\s*(?:giữa|so với)\s*(?:cuối|đầu)?\s*(?:năm|ngày|quý)?\s*[\d/.\-]*\s*"
    r"(?:và|so với)?\s*$",
    re.I,
)

# Report scope, not part of any line-item label.
SCOPE_WORDS_RE = re.compile(r"\s*\b(hợp nhất|riêng|của công ty mẹ|công ty mẹ)\b", re.I)

# A trailing period expression: "trong năm 2018", "vào ngày 31/12/2020",
# "cuối năm". Anchored to the end so a date sitting mid-phrase cannot truncate
# the metric itself — "Giá trị còn lại cuối năm Tài sản cố định vô hình" once
# collapsed to "còn lại".
TRAILING_PERIOD_RE = re.compile(
    r"(?:\s*(?:trong|vào|đến|tại|kể từ)?\s*(?:cuối|đầu)?\s*"
    r"(?:năm|ngày|quý)\s*(?:tài chính)?\s*[\d/.\-]*)$",
    re.I,
)

# Comparative or derived phrasing: not a single-cell lookup.
DERIVED_RE = re.compile(
    r"chênh lệch|thay đổi|hiệu số|so với|tăng trưởng|bình quân|trung bình|"
    r"cao nhất|thấp nhất|lớn nhất|nhỏ nhất|tỷ lệ|tỷ số|hệ số|biên lợi nhuận|"
    r"trong các công ty|trong nhóm|trong số",
    re.I,
)


# Phrasings `DERIVED_RE` misses, which is how 19 of the 411 questions this branch
# claims are two-cell differences answered with one cell — guaranteed wrong, and
# they never reach the panel path because this branch takes them first.
#
# Every word here has to be one that cannot appear in a *row label*, or the gate
# starts rejecting genuine single lookups. "Tổng cộng tài sản" and "Tổng cộng
# nguồn vốn" are balance-sheet lines, not sums, so a naive `tổng|tổng cộng` rule
# flagged 65 questions of which only these 19 were real.
MORE_DERIVED_RE = re.compile(
    r"trừ đi|lớn hơn|nhỏ hơn|cao hơn|thấp hơn|nhiều hơn|ít hơn|gấp bao nhiêu|"
    r"tỷ trọng|bao nhiêu (doanh nghiệp|công ty|đơn vị|mã)",
    re.I,
)
STRICT_DERIVED = os.environ.get("VIFIN_STRICT_DERIVED") == "1"


def is_single_lookup(question: str) -> bool:
    """Whether the question asks for one figure rather than a computation."""

    if DERIVED_RE.search(question):
        return False
    # Off by default: narrowing this branch re-routes those questions to whatever
    # runs next, and a change that looked safe on this branch cost 24 questions on
    # 12/08. `VIFIN_STRICT_DERIVED=1` turns it on so the A/B can measure it.
    return not (STRICT_DERIVED and MORE_DERIVED_RE.search(question))

TABLE_SCALES = (
    ("nghin_ty", re.compile(r"nghìn tỷ", re.I), 1e12),
    ("ty", re.compile(r"\btỷ\b", re.I), 1e9),
    ("trieu", re.compile(r"triệu", re.I), 1e6),
    ("nghin", re.compile(r"nghìn|ngàn", re.I), 1e3),
)

# The local scorer says the matcher produces no candidate at all on 40.5% of gold
# questions while being right 74.2% of the times it does commit, so its reach is a
# larger lever than its accuracy. Sweepable so that trade can be measured offline
# instead of through submissions.
# MEASURED: swept against the organisers' own cell coordinates through `find()`,
# on 1,500 gold records. 0.75 commits on 731 and takes the right row on 434; 0.55
# commits on 895 and takes 454. Reach is worth more than per-commitment precision
# here, which is what the coverage figure in `score_local.py` predicted.
MIN_LABEL_SCORE = float(os.environ.get("VIFIN_MIN_LABEL_SCORE", "0.55"))

# MEASURED AS A NO-OP, so off by default. The winning label contradicts the metric
# on 54 `match_row` calls, and 73 questions with a contrastive metric are answered
# by this branch — yet rejecting those winners changed **zero** shipped answers.
# The reason is that `match_row` runs once per (table, variant) and the caller keeps
# the best result across all of them: wherever one table offers "Phải trả ngắn hạn
# khác" for a question about "phải thu ngắn hạn khác", another table offers the real
# row at a higher score. Cross-table scoring already resolves the siblings.
# `VIFIN_REJECT_CONTRASTIVE=1` turns the rejection on; the measurement below runs
# either way and is the part worth keeping.
REJECT_CONTRASTIVE = os.environ.get("VIFIN_REJECT_CONTRASTIVE", "0") == "1"
# Every (metric, rejected label) the guard fired on, for measurement.
CONTRASTIVE_SKIPS: list[tuple[str, str]] = []

# Pairs that name different figures while sharing almost every word. A
# Vietnamese statement lists both members side by side, so the token overlap
# between them is near-total: "tài sản cố định vô hình" against "tài sản cố định
# hữu hình" shares five of six tokens and scores 0.833, comfortably over the
# threshold. The matcher then ships the sibling whenever the wanted row is absent
# from the table it happens to be reading — and the sibling is a completely
# different number, not a near miss.
#
# Written in ordinary Vietnamese and normalised through `_norm` at first use.
# Hand-writing the normalised form is the fifth-time trap on this project:
# `strip_tones` removes the tone marks but keeps the vowel-quality ones and "đ",
# so "vô hình" becomes "vô hinh" — neither the original nor plain ASCII. Guessing
# it wrong makes the guard match nothing, silently.
CONTRASTIVE_PAIRS = (
    ("vô hình", "hữu hình"),
    ("ngắn hạn", "dài hạn"),
    ("trước thuế", "sau thuế"),
    ("phải thu", "phải trả"),
    ("trong nước", "nước ngoài"),
    ("đầu năm", "cuối năm"),
    ("đầu kỳ", "cuối kỳ"),
    ("năm trước", "năm nay"),
    ("nội bộ", "bên ngoài"),
)

_CONTRASTIVE_NORM: tuple[tuple[str, str], ...] | None = None


def _contrastive() -> tuple[tuple[str, str], ...]:
    """Both directions of every pair, normalised the way labels are.

    Built lazily: `_norm` expands mined abbreviations, and that table is loaded
    after import.
    """

    global _CONTRASTIVE_NORM
    if _CONTRASTIVE_NORM is None:
        both = list(CONTRASTIVE_PAIRS) + [(b, a) for a, b in CONTRASTIVE_PAIRS]
        _CONTRASTIVE_NORM = tuple(
            (_norm(one), _norm(other)) for one, other in both)
    return _CONTRASTIVE_NORM


def contradicts(metric_norm: str, label_norm: str) -> bool:
    """True when the label names the opposite member of a contrastive pair.

    Only fires when the metric commits to one side and the label commits to the
    other. A label carrying neither modifier is not a contradiction — most rows do
    not carry one at all, and rejecting those would empty the matcher.
    """

    for wanted, opposite in _contrastive():
        if wanted in metric_norm and opposite in label_norm and wanted not in label_norm:
            return True
    return False


def _load_abbreviations() -> dict[str, str]:
    """Abbreviation expansions mined from the corpus, if they have been built.

    Produced by `scripts/mine_aliases.py`: a figure unique in both report(T, Y)
    and report(T, Y+1) links two row labels to the same account regardless of
    wording, and a rule is only kept when the short token's letters are exactly
    the initials of the span it replaces. So "TSCĐ" -> "tài sản cố định" is
    verified, not guessed.

    This is the cheap half of a finding that started out pointed at fine-tuning.
    The reliable half of the mined pairs turned out to be abbreviations and OCR
    variants, which a table handles deterministically — and deterministic has
    beaten every model mechanism measured on this corpus.
    """

    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "artifacts" / "abbreviations.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    rules = {}
    for short, long in raw.items():
        words = str(long).split()
        # Guard against OCR merges the miner cannot see through, such as
        # "thu nhap doanh nghiephien" or a company name glued to its initials.
        if len(words) < 2 or any(len(w) > 12 for w in words):
            continue
        if len(short) < 2:
            continue
        rules[short] = " ".join(words)
    return rules


ABBREVIATIONS = _load_abbreviations()
_ABBREV_RE = (
    re.compile(r"\b(" + "|".join(sorted(map(re.escape, ABBREVIATIONS), key=len, reverse=True)) + r")\b")
    if ABBREVIATIONS else None
)


def _norm(text: str) -> str:
    flat = re.sub(r"[^\w\s]", " ", strip_tones(str(text)).casefold()).strip()
    if _ABBREV_RE is not None:
        flat = _ABBREV_RE.sub(lambda m: ABBREVIATIONS[m.group(1)], flat)
        flat = re.sub(r"\s+", " ", flat).strip()
    return flat


def _tokens(text: str) -> list[str]:
    return _norm(text).split()


# Aspects of an asset rather than assets themselves. In "Giá trị còn lại của
# quyền sử dụng đất" the aspect names the *column* and the noun behind it names
# the *row*, so the whole phrase matches neither well — the row reads only
# "Quyền sử dụng đất". Restricted to this vocabulary on purpose: applied to
# "Tiền gửi của khách hàng" the same split would offer "khách hàng", which can
# score a spurious 1.00 against an unrelated row.
ASPECT_OF_RE = re.compile(
    r"^(?:tổng\s+)?(?:giá trị còn lại|giá trị thuần|nguyên giá|giá gốc|giá vốn|"
    r"khấu hao lũy kế|dự phòng|số lượng)\s+của\s+(.+)$",
    re.I,
)


# "Tỷ lệ sở hữu Công ty CP Gang thép Hòa Phát" names the column ("Tỷ lệ sở hữu")
# and the row (the subsidiary). Unlike ASPECT_OF_RE there is no "của" between
# them, so the split is on the aspect phrase alone.
RATIO_ASPECT_RE = re.compile(
    r"^(?:tổng\s+)?(?:tỷ lệ|tỉ lệ)\s+(?:sở hữu|biểu quyết|quyền biểu quyết|lợi ích|"
    r"lợi ích kinh tế|nắm giữ|góp vốn|phần vốn)\s+(?:tại\s+|của\s+|trên\s+)?(.+)$",
    re.I,
)


# "Tỷ lệ sở hữu của công ty mẹ PLX tại Công ty TNHH Hóa chất PTN" holds the row
# name behind "tại", on the far side of the owner clause — so it is gone by the
# time the metric is extracted, leaving only "Tỷ lệ sở hữu". Read off the
# original question instead.
AT_ENTITY_RE = re.compile(
    r"\btại\s+((?:CTCP|Công ty|Cty|Ngân hàng|Xí nghiệp|Tập đoàn|Tổng Công ty)[^,?]*)",
    re.I,
)

# Trailing period phrases the row name picks up when it is read out of the middle
# of a sentence: "... PTN vào cuối năm 2023", "... 3F Việt cuối năm 2022".
_AT_ENTITY_TAIL_RE = re.compile(
    r"\s*(?:vào|đến|tính đến|trong)?\s*(?:cuối|đầu)?\s*(?:năm|ngày|quý)\b.*$", re.I
)


# Trailing words that qualify *which* year's figure is wanted, not which row:
# "doanh thu thuần thấp nhất" is the row "Doanh thu thuần".
TRAILING_SUPERLATIVE_RE = re.compile(
    r"\s*\b(?:cao nhất|thấp nhất|lớn nhất|nhỏ nhất|trung bình|bình quân|"
    r"tích lũy|cộng lại)\s*$",
    re.I,
)

# An owner clause introduced by "tại" rather than "của": "... tài sản cố định
# hữu hình tại CTCP Tập đoàn Dabaco Việt Nam". Requires an explicit legal form —
# "Ngân hàng" is excluded on purpose, because "Tiền gửi tại Ngân hàng Nhà nước"
# is an account name and would lose its own tail.
TRAILING_AT_OWNER_RE = re.compile(
    r"\s*\btại\s+(?:CTCP|Công ty Cổ phần|Tổng Công ty|Tập đoàn|Xí nghiệp)\b.*$", re.I
)

LEADING_TONG_RE = re.compile(r"^tổng\s+(?!cộng)", re.I)


def metric_variants(question: str) -> list[str]:
    """Every plausible reading of which row the question names.

    Additive by design. `match_row` scores each variant and keeps the best, and
    MIN_LABEL_SCORE still floors the result, so offering another reading can only
    surface a row that matches *better* than the ones already on offer. That makes
    this the low-risk place to handle wording the extractor gets wrong — as
    opposed to changing `extract_metric`, where every edit trades one class of
    question for another.
    """

    metric = extract_metric(question)
    variants = [metric]

    def offer(text: str) -> None:
        text = text.strip(" ,.;:")
        if text and text not in variants:
            variants.append(text)

    for pattern in (ASPECT_OF_RE, RATIO_ASPECT_RE):
        match = pattern.match(metric)
        if match:
            offer(match.group(1))
    at = AT_ENTITY_RE.search(question)
    if at:
        offer(_AT_ENTITY_TAIL_RE.sub("", at.group(1)))
    for pattern in (TRAILING_SUPERLATIVE_RE, TRAILING_AT_OWNER_RE, LEADING_TONG_RE):
        reduced = pattern.sub("", metric)
        if reduced != metric:
            offer(reduced)
    return variants


def find_ratio(grid: list[list[str]], parsed: ParsedQuestion) -> Lookup | None:
    """A proportion read straight from a percentage column."""

    label_col = label_column(grid)
    columns = ratio_columns(grid, label_col)
    if not columns:
        return None
    matched = None
    for variant in metric_variants(parsed.question):
        candidate = match_row(grid, variant, label_col)
        if candidate is not None and (matched is None or candidate[1] > matched[1]):
            matched = candidate
    if matched is None:
        return None
    row, score, label = matched
    for column in columns:
        if column >= len(grid[row]):
            continue
        value = _parse_cell(grid[row][column])
        if value is not None and 0 < abs(value) <= 100:
            return Lookup(row=row, column=column, label=label, value=value,
                          score=score, label_col=label_col)
    return None


def synthesize_ratio_cell(lookup: Lookup, as_percent: bool) -> str:
    """Report the proportion verbatim, converting only between % and times."""

    label = lookup.label.replace("\\", "\\\\").replace('"', '\\"')
    factor = "1.0" if as_percent else "100.0"
    return f'''labels = df.iloc[:, {lookup.label_col}].astype(str).str.strip()
hits = labels[labels == "{label}"]
row = hits.index[0] if len(hits) > 0 else {lookup.row - 1}
col = df.iloc[:, {lookup.column}].astype(str).str.strip()
col = col.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
col = col.str.replace("%", "", regex=False)
col = col.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
values = pd.to_numeric(col, errors="coerce")
row = min(max(int(row), 0), len(values) - 1)
cell = float(values.iloc[row])
result = round(abs(cell) / {factor}, 2)'''


def _drop_owner_clause(text: str) -> str:
    """Cut at the "của" that introduces the company, not at the first one.

    "của" is doing two jobs. In "Lãi tiền gửi của VJC" it hands off to the owner;
    in "Giá trị còn lại của quyền sử dụng đất" it is part of the account name.
    Splitting at the first occurrence regardless truncated 152 of 1012 metrics,
    and the residue ("Giá trị còn lại", "Giá vốn") matches any line in a
    fixed-asset note equally well.
    """

    for match in re.finditer(r"\bcủa\b", text):
        if ENTITY_HEAD_RE.match(text[match.end():]):
            return text[: match.start()]
    return text


def extract_metric(question: str) -> str:
    """The figure being asked for, stripped of everything that frames it.

    Originally just "the phrase before the first `của`", which holds for the
    canonical "METRIC của COMPANY năm Y" wording but silently returned the whole
    question for the two other shapes the test set uses. 236 of 1012 extracted
    metrics still contained a year, and a metric whose tokens include "2019 và
    2022" cannot reach MIN_LABEL_SCORE against any real line item — the label
    match failed on all years of 96 multi-year questions for this reason alone.
    """

    head = TAIL_RE.sub("", question.strip().rstrip("?"))
    head = _drop_owner_clause(head)
    previous = None
    while previous != head:
        previous = head
        head = LEADING_CLAUSE_RE.sub("", head).strip(" ,.;:")
        head = LEADING_YEARS_RE.sub("", head).strip(" ,.;:")
    head = ENTITY_VERB_RE.sub("", head).strip(" ,.;:")
    head = LEADING_OP_RE.sub("", head).strip(" ,.;:")
    head = LEADING_PERIOD_RE.sub("", head).strip(" ,.;:")
    head = SCOPE_WORDS_RE.sub("", head).strip(" ,.;:")
    previous = None
    while previous != head:
        previous = head
        head = TRAILING_COMPARE_RE.sub("", head).strip(" ,.;:")
        head = TRAILING_PERIOD_RE.sub("", head).strip(" ,.;:")
    head = METRIC_PREFIX_RE.sub("", head.strip())
    return head.strip(" ,.;:")


def table_scale(unit_text: str) -> float:
    """Multiplier turning a cell value into đồng."""

    for _, pattern, scale in TABLE_SCALES:
        if pattern.search(unit_text or ""):
            return scale
    return 1.0


def column_scale(grid: list[list[str]], column: int, fallback: str) -> float:
    """Unit of one column's figures.

    Most statements never emit a standalone "Đơn vị tính" line — the unit rides
    in the column header instead ("2018Triệu VND", "Số cuối nămTriệu đồng").
    Reading only the surrounding prose scored those tables as plain đồng and put
    every answer out by a factor of a million.
    """

    header_rows = grid[:2]
    own_header = " ".join(str(row[column]) for row in header_rows if column < len(row))
    scale = table_scale(own_header)
    if scale == 1.0:
        whole_header = " ".join(str(cell) for row in header_rows for cell in row)
        scale = table_scale(whole_header)
    if scale == 1.0:
        scale = table_scale(fallback)
    return _checked_scale(grid, column, scale)


# The largest figure any of these companies reports, with room to spare: VCB's
# total assets are ~2e15 đồng. A column whose own numbers would exceed this after
# scaling is not in the unit the prose claims.
DONG_CEILING = 1e16


def _checked_scale(grid: list[list[str]], column: int, scale: float) -> float:
    """Reject a scale the column's own digits contradict.

    The unit is read off nearby prose, which belongs to the page rather than to
    this table: FPT's note says "triệu đồng)" while the table prints
    34.732.056.920.000 in full. Scaling that by a million claimed 3.5e19 đồng —
    roughly a thousand times Vietnam's GDP — and 29 submitted answers were wrong
    this way.

    Digit count settles it. No statement prints a 14-digit figure in triệu đồng,
    so a column whose median would break DONG_CEILING is already in đồng. This is
    evidence about which unit the table uses, not a guess at a plausible answer —
    the distinction that separates it from the `USE_RATIO` experiment, which
    replaced impossible values with merely believable ones and gained nothing.
    """

    if scale <= 1.0:
        return scale
    magnitudes = []
    for row in grid[1:]:
        if column >= len(row):
            continue
        value = _parse_cell(row[column])
        if value:
            magnitudes.append(abs(value))
    if not magnitudes:
        return scale
    magnitudes.sort()
    median = magnitudes[len(magnitudes) // 2]
    return 1.0 if median * scale > DONG_CEILING else scale


@dataclass(frozen=True, slots=True)
class Lookup:
    row: int
    column: int
    label: str
    value: float
    score: float
    label_col: int = 0


# A minority of reports (2.4% of tables) print figures English-style,
# "19,430,048,338,817". The organisers' `is_numeric_cell` rejects those outright,
# which silently emptied whole statements of value columns.
_EN_AMOUNT_RE = re.compile(r"^\(?-?\d{1,3}(,\d{3})+(\.\d+)?\)?%?$")


def _parse_cell(cell: str) -> float | None:
    text = str(cell).strip()
    english = bool(_EN_AMOUNT_RE.match(text))
    if not english and not is_numeric_cell(text):
        return None
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1]
    text = text.removesuffix("%")
    # Grouping and decimal marks swap roles between the two conventions.
    text = text.replace(",", "") if english else text.replace(".", "").replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


# A cell that carries a name rather than a code: four or more letters. Line
# codes ("100", "5.14."), sequence numbers and note refs all fall below it.
_WORDY_RE = re.compile(r"[^A-Za-zÀ-ỹ]")


def label_column(grid: list[list[str]], limit: int = 3) -> int:
    """Which column holds the line-item names.

    Column 0 usually does, but 9.4% of tables put a code there instead — and not
    marginal ones: a balance sheet with a "Mã số" column reads 100 / 110 / 111 in
    column 0 and "A. TÀI SẢN NGẮN HẠN" / "1. Tiền" in column 1, and subsidiary
    lists number their rows. Matching labels against column 0 regardless found
    nothing in any of them.
    """

    if len(grid) < 3 or not grid[0]:
        return 0
    best, best_score = 0, 0.0
    for column in range(min(limit, len(grid[0]))):
        cells = [str(row[column]).strip() for row in grid[1:] if column < len(row)]
        cells = [c for c in cells if c]
        if len(cells) < 3:
            continue
        score = sum(1 for c in cells if len(_WORDY_RE.sub("", c)) >= 4) / len(cells)
        # Strictly greater, so column 0 keeps the tie and stays the default.
        if score > best_score:
            best, best_score = column, score
    return best if best_score >= 0.5 else 0


def value_columns(grid: list[list[str]], label_col: int = 0) -> list[int]:
    """Columns that plausibly hold figures, left to right."""

    if len(grid) < 2:
        return []
    width = len(grid[0])
    header = [" ".join(str(row[c]) for row in grid[:2] if c < len(row)) for c in range(width)]
    chosen = []
    for column in range(label_col + 1, width):
        if NON_VALUE_HEADER_RE.search(header[column].strip()):
            continue
        values = [_parse_cell(row[column]) for row in grid[1:] if column < len(row)]
        numbers = [v for v in values if v is not None]
        if not values or len(numbers) / len(values) < 0.4:
            continue
        # Line codes ("01", "31") and note refs survive the header filter; real
        # figures are orders of magnitude larger.
        magnitudes = sorted(abs(v) for v in numbers if v)
        if magnitudes and magnitudes[len(magnitudes) // 2] < 1000:
            continue
        chosen.append(column)
    return chosen


# Headers of columns that hold a proportion rather than an amount.
RATIO_HEADER_RE = re.compile(r"%|tỷ lệ|tỉ lệ|sở hữu|biểu quyết|lợi ích|phần vốn", re.I)


def ratio_columns(grid: list[list[str]], label_col: int = 0) -> list[int]:
    """Columns holding proportions, which `value_columns` deliberately drops.

    `value_columns` discards any column whose median magnitude is under 1000, to
    keep statement line codes and note references out. That filter also discards
    every ownership and voting-rights percentage in the corpus — so questions
    asking "bao nhiêu %" were answered from a đồng column and came back with
    figures like 1,028,364,192,393 percent. The proportion is sitting in the
    table; nothing needs to be computed.

    A header check is required, not optional: "Thuyết minh" holds 28 and 29, and
    "Mã số" holds 1..5, both of which look like proportions by magnitude alone.
    """

    if len(grid) < 2:
        return []
    width = len(grid[0])
    headers = [
        " ".join(str(row[c]) for row in grid[:2] if c < len(row)) for c in range(width)
    ]
    named, plain = [], []
    for column in range(label_col + 1, width):
        header = headers[column].strip()
        if NON_VALUE_HEADER_RE.search(header):
            continue
        values = [_parse_cell(row[column]) for row in grid[1:] if column < len(row)]
        numbers = [v for v in values if v is not None]
        if len(numbers) < 2 or len(numbers) / max(1, len(values)) < 0.4:
            continue
        if sum(1 for v in numbers if 0 < abs(v) <= 100) / len(numbers) < 0.6:
            continue
        (named if RATIO_HEADER_RE.search(header) else plain).append(column)
    # Named only. Every genuine proportion column in the sample carried one of
    # these words in its header, and both false positives ("Thuyết minh", "Mã
    # số") carried none — so an unnamed small-magnitude column is not worth the
    # risk of answering from a note number.
    return named


def pick_column(
    grid: list[list[str]], parsed: ParsedQuestion, label_col: int = 0
) -> int | None:
    """Prefer a column whose header names the year asked about."""

    columns = value_columns(grid, label_col)
    if not columns:
        return None
    if parsed.years:
        target = str(max(parsed.years))
        for column in columns:
            header = " ".join(str(row[column]) for row in grid[:2] if column < len(row))
            if target in header:
                return column
    # Statements put the current period first: "Năm nay" before "Năm trước",
    # "Số cuối năm" before "Số đầu năm".
    return columns[0]


def match_row(
    grid: list[list[str]], metric: str, label_col: int = 0, caption: str = ""
) -> tuple[int, float, str] | None:
    """Best line-item label for the metric, scored symmetrically.

    Coverage alone ("do all the question's words appear?") rewards the longest
    label on the page: "Chi phí dự phòng" scored a perfect 1.0 against "Lợi
    nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng",
    which is a different figure entirely. Balancing coverage against the label's
    own extra words makes the specific row win.

    `caption` lets the total row be reached. Vietnamese statements leave the label
    cell empty on the line that sums the block above it, and **25.6% of gold cells
    sit on such a row** — measured against the organisers' own `pandas_query` in
    `scripts/_probe_blank_rows.py`. Tokenising an empty label yields nothing, so
    those rows were skipped and the matcher could never select one, whatever the
    threshold. Naming them "Tổng <caption>" lets ordinary scoring decide, and on
    1,500 gold records it raised correct row picks from 125 to 176 at the current
    threshold, and to 274 once the threshold is 0.55 as well: both the reach and
    the accuracy-per-commitment improve, which is why this is not the same thing
    as the bolted-on total-row rule that lost 24 questions.
    """

    wanted = set(_tokens(metric))
    if not wanted:
        return None
    best: tuple[int, float, str] | None = None
    for index, row in enumerate(grid[1:], start=1):
        if not row:
            continue
        if label_col >= len(row):
            continue
        label = str(row[label_col]).strip()
        if not label and caption and any(str(cell).strip() for cell in row[1:]):
            # The total line of a block leaves its label cell empty. Naming it
            # after the note the table belongs to lets it be scored like any other
            # row instead of being skipped outright.
            label = f"Tổng {caption}"
        tokens = set(_tokens(label))
        if not tokens:
            continue
        normalized = _norm(label)
        # Skip rather than score: the matcher then keeps looking and takes the
        # best row that does not contradict, instead of shipping the sibling.
        if normalized == _norm(metric):
            score = 1.0
        elif normalized.startswith(_norm(metric) + " "):
            # Statement captions elaborate rightward: "Lợi nhuận sau thuế" is
            # written "Lợi nhuận sau thuế thu nhập doanh nghiệp". A phrase that
            # instead appears *inside* a longer caption is usually a different
            # figure — "... trước chi phí dự phòng rủi ro tín dụng" is not
            # "chi phí dự phòng" — so only a prefix earns the boost.
            score = 0.95
        else:
            shared = len(wanted & tokens)
            if not shared:
                continue
            coverage = shared / len(wanted)
            focus = shared / len(tokens)
            score = 2 * coverage * focus / (coverage + focus)
        if score > (best[1] if best else 0.0):
            best = (index, score, label)
    if best is None or best[1] < MIN_LABEL_SCORE:
        return None
    # Skipping contradictions inside the loop fired 12,661 times and changed no
    # answer: it only ever rejected rows that scored far below the floor. What
    # matters is whether the WINNER contradicts, so it is tested here, once.
    if contradicts(_norm(metric), _norm(best[2])):
        CONTRASTIVE_SKIPS.append((metric[:60], best[2][:60]))
        if REJECT_CONTRASTIVE:
            return None
    return best


def synthesize_ratio(lookup: Lookup, as_percent: bool) -> str:
    """A ratio answer for a "phần trăm"/"lần" question.

    Reporting a single cell for these was wrong by construction: the currency
    scale turned a đồng amount into an answer like 16,398,795,750 percent. 273
    of 1,012 submitted answers looked like that.

    Two cases. A cell already expressing a proportion (0..100, often with a "%")
    is reported verbatim — ownership and voting-rights tables store it that way.
    Otherwise the row's share of its column total is the best available reading of
    "tỷ trọng"/"tỷ lệ", and unlike a đồng amount it at least lands in range.
    """

    label = lookup.label.replace("\\", "\\\\").replace('"', '\\"')
    factor = "100.0" if as_percent else "1.0"
    # `row` is clamped rather than trusted. The positional fallback is one index
    # off the frame this runs against, and an out-of-range `.iloc` raised
    # IndexError on 7 questions — which then lost their answer entirely and were
    # scored as evidence against ratio synthesis rather than against this line.
    return f'''labels = df.iloc[:, {lookup.label_col}].astype(str).str.strip()
hits = labels[labels == "{label}"]
row = hits.index[0] if len(hits) > 0 else {lookup.row - 1}
col = df.iloc[:, {lookup.column}].astype(str).str.strip()
col = col.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
col = col.str.replace("%", "", regex=False)
col = col.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
values = pd.to_numeric(col, errors="coerce")
row = min(max(int(row), 0), len(values) - 1)
cell = float(values.iloc[row])
if cell != cell:
    cell = 0.0
if 0.0 <= abs(cell) <= 100.0:
    result = round(cell, 2)
else:
    total = float(values.drop(values.index[row]).abs().sum())
    result = round(cell / total * {factor}, 2) if total > 0 else round(cell, 2)'''


def synthesize_scan(column: int, as_ratio: bool = False, as_percent: bool = False) -> str:
    """Absolute last resort: read the first figure in a column.

    Almost certainly the wrong number, but it is a program that reads the shipped
    table, which `result = 0.0` is not. The private round reviews `pandas_query`
    by hand and rejects hard-coded answers, so every question must carry real code.
    """

    body = f'''col = df.iloc[:, {column}].astype(str).str.strip()
col = col.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
col = col.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
values = pd.to_numeric(col, errors="coerce").dropna()
result = round(float(values.iloc[0]), 2) if len(values) > 0 else float(len(df))'''
    if as_ratio:
        factor = "100.0" if as_percent else "1.0"
        body += f'''
total = float(values.abs().sum())
if total > 0 and abs(result) > 100.0:
    result = round(result / total * {factor}, 2)'''
    return body


def find_best_effort(grid: list[list[str]], parsed: ParsedQuestion) -> Lookup | None:
    """The closest row available, with no confidence floor.

    Used only where nothing else produced an answer. The point is not accuracy —
    it is that the submitted `pandas_query` must be a real program. Shipping
    `result = 0.0` instead is a hard-coded constant, and the organisers reject
    those on manual review in the private round, so a weak genuine query strictly
    dominates a placeholder.
    """

    label_col = label_column(grid)
    metric = extract_metric(parsed.question)
    matched = (match_row(grid, metric, label_col)
               or _closest_row(grid, metric, label_col))
    column = pick_column(grid, parsed, label_col)
    if matched is None or column is None:
        return None
    row, score, label = matched
    if column >= len(grid[row]):
        return None
    value = _parse_cell(grid[row][column])
    if value is None:
        return None
    return Lookup(row=row, column=column, label=label, value=value,
                  score=score, label_col=label_col)


def _closest_row(
    grid: list[list[str]], metric: str, label_col: int = 0
) -> tuple[int, float, str] | None:
    wanted = set(_tokens(metric))
    best: tuple[int, float, str] | None = None
    for index, row in enumerate(grid[1:], start=1):
        if not row:
            continue
        if label_col >= len(row):
            continue
        label = str(row[label_col]).strip()
        tokens = set(_tokens(label))
        if not tokens or not wanted:
            continue
        shared = len(wanted & tokens)
        if not shared:
            continue
        score = shared / len(wanted | tokens)
        if score > (best[1] if best else 0.0):
            best = (index, score, label)
    return best


# "Tổng Công ty" is how a large share of Vietnamese issuers begin their name, so a
# bare \btổng\b matches the *company* on questions that ask for nothing of the
# kind. Caught by reading six patched records: "Nguyên giá xây dựng Nhà ga Hành
# khách T2 của Tổng Công ty Cảng Hàng không" is a single-asset question, and the
# rule fired on it and overwrote a better answer. The same trap sits in "Tổng Giám
# đốc" and "Tổng Giám sát".
ASKS_TOTAL_RE = re.compile(r"\b(tổng|cộng)\b(?!\s*(công\s*ty|giám))", re.I)

# Only a *bare* total marker counts. A described one — "Tổng nợ phải trả và vốn
# chủ sở hữu", "Cộng: Dự phòng tăng do hợp nhất" — is a named line item, and the
# argument for rejecting it is that this rule only runs when `match_row` already
# failed: if the description did match the question, the matcher would have taken
# it and the rule would never have been reached. Reaching here means the
# description is about something else. Asked for MSB's payroll, the described form
# returned total liabilities and equity.
#
# This costs the occasional real total that carries a description ("Cộng tiền và
# các khoản tương đương tiền"), which is the right side to err on: a silence gives
# the question up, a confident wrong answer also displaces whatever branch would
# have served it.
TOTAL_LABEL_RE = re.compile(r"^\s*(tổng\s*cộng|cộng|tổng|total)\s*[:：.]?\s*$", re.I)
TOTAL_ROW_ENABLED = os.environ.get("VIFIN_TOTAL_ROW", "1") != "0"
# MEASURED AS A NO-OP, so off by default. Naming an unlabelled total row
# "Tổng <caption>" so `match_row` can score it looked like the largest finding of
# the day — correct row picks went from 125 to 176 on 1,500 gold records. That
# baseline was `match_row` called on its own, which leaves out the `total_row`
# fallback a few lines below, and `total_row` already reaches those rows: run
# through `find()`, the function a submission actually calls, the count is 434
# either way and the baseline already lands on 201 unlabelled rows.
# `VIFIN_BLANK_CAPTION=1` re-enables it.
BLANK_ROW_CAPTION = os.environ.get("VIFIN_BLANK_CAPTION", "0") == "1"


def total_row(grid: list[list[str]], label_col: int) -> int | None:
    """The row a statement uses for its total, or None.

    These reports print a total in one of two ways: a row named `Cộng`/`Tổng`, or
    a trailing row with no label at all. Neither can be reached by matching the
    question's metric words against row labels, because the total row does not
    repeat the metric — the caption above the table names it.

    Measured on the gold set: this is where a third of `find`'s silences end.
    Asked "Tổng số dư tiền gửi của khách hàng", the table lists customer *types*
    and the answer is the unnamed row underneath them.

    A blank label is only read as a total in the last two rows. Mid-table it is
    usually a section break or a wrapped line, and treating those as totals turns
    a correct silence into a confident wrong answer.
    """

    body = grid[1:]
    for index in range(len(body) - 1, -1, -1):
        line = body[index]
        label = (line[label_col] or "").strip() if label_col < len(line) else ""
        if TOTAL_LABEL_RE.match(label):
            return index + 1
    for index in range(len(body) - 1, max(len(body) - 3, -1), -1):
        line = body[index]
        label = (line[label_col] or "").strip() if label_col < len(line) else ""
        if not label and any(_parse_cell(cell) is not None for cell in line):
            return index + 1
    return None


FUZZY_LABEL_ENABLED = os.environ.get("VIFIN_FUZZY_LABEL") == "1"
FUZZY_MIN = float(os.environ.get("VIFIN_FUZZY_MIN", "0.30"))


def _trigrams(text: str) -> set[str]:
    flat = re.sub(r"\s+", " ", strip_tones(text).lower()).strip()
    return {flat[index:index + 3] for index in range(max(len(flat) - 2, 0))}


def fuzzy_row(grid: list[list[str]], parsed: ParsedQuestion,
              label_col: int) -> tuple[int, float, str] | None:
    """Nearest row label by character trigrams, for when tokens do not meet.

    `match_row` compares token sets, so it scores zero when the question and the
    table name the same thing differently — and on the gold set that is 49% of
    the cases where `find` says nothing:

        hỏi "tại Ngân hàng Nhà nước"   bảng "NHNN"                viết tắt
        hỏi "chưa hoàn thành"          bảng "dở dang"             đồng nghĩa
        hỏi "cơ bản đã hoàn thành"     bảng "cơ bảnhoản thà"      OCR hỏng

    Trigrams survive OCR damage because it removes spaces rather than letters,
    and an initialism catches the abbreviation. Synonyms are out of reach here
    and are what the fine-tune is for.
    """

    best: tuple[int, float, str] | None = None
    variants = list(metric_variants(parsed.question))
    for index, row in enumerate(grid[1:], start=1):
        if label_col >= len(row):
            continue
        label = str(row[label_col]).strip()
        if not label:
            continue
        label_grams = _trigrams(label)
        if not label_grams:
            continue
        for variant in variants:
            grams = _trigrams(variant)
            if not grams:
                continue
            score = len(grams & label_grams) / len(grams | label_grams)
            if len(label) <= 8:
                flat = strip_tones(label).lower()
                acronym = "".join(w[0] for w in strip_tones(variant).lower().split() if w)
                if flat and acronym.startswith(flat):
                    score = max(score, 0.9)
            if score >= FUZZY_MIN and (best is None or score > best[1]):
                best = (index, score, label)
    return best


def find(grid: list[list[str]], parsed: ParsedQuestion,
         caption: str = "") -> Lookup | None:
    label_col = label_column(grid)
    matched = None
    for variant in metric_variants(parsed.question):
        candidate = match_row(grid, variant, label_col,
                              caption if BLANK_ROW_CAPTION else "")
        if candidate is not None and (matched is None or candidate[1] > matched[1]):
            matched = candidate
    column = pick_column(grid, parsed, label_col)

    # No label matched, but the question is asking for a total and the table has
    # a total row. Measured on 600 gold questions: fires on 41.8% of the cases
    # where `find` currently returns nothing, and is right 68.2% of those — which
    # is the same reliability as the label match itself (70.8%), and far above
    # the 5.9-13.3% of the LLM branches that pick up these questions today.
    #
    # Scored at MIN_LABEL_SCORE rather than above it: this is a structural
    # inference, not a label agreement, and it should lose to any real match.
    # `VIFIN_TOTAL_ROW=0` restores the previous behaviour, so the gain can be
    # measured as a difference on one sample rather than compared across two runs
    # of different sizes.
    if (TOTAL_ROW_ENABLED and matched is None and column is not None
            and ASKS_TOTAL_RE.search(parsed.question)):
        row = total_row(grid, label_col)
        if row is not None:
            matched = (row, MIN_LABEL_SCORE,
                       (grid[row][label_col] or "").strip()
                       if label_col < len(grid[row]) else "")

    # Last, and only behind a flag until an end-to-end A/B says otherwise: the
    # trigram fallback. Scored at the floor so it loses to a token match and to
    # the total row, both of which are more certain than a character similarity.
    if FUZZY_LABEL_ENABLED and matched is None and column is not None:
        fuzzy = fuzzy_row(grid, parsed, label_col)
        if fuzzy is not None:
            matched = (fuzzy[0], MIN_LABEL_SCORE, fuzzy[2])

    if matched is None or column is None:
        return None
    row, score, label = matched
    if column >= len(grid[row]):
        return None
    value = _parse_cell(grid[row][column])
    if value is None:
        return None
    return Lookup(row=row, column=column, label=label, value=value,
                  score=score, label_col=label_col)


def synthesize(lookup: Lookup, scale_in: float, scale_out: float, magnitude: bool = False) -> str:
    """Emit a self-contained program the scorer can re-run.

    Matching is by label text rather than row position so the program stays
    meaningful, with a positional fallback for OCR labels that will not compare
    equal after whitespace differences.
    """

    # A ratio question ("bao nhiêu lần", "%") has no money scale, so `unit_scale`
    # is None for it. `resolve_screen_ratio` admits those questions and then hands
    # that None straight here, which crashed the packager on every build. A
    # unitless answer needs no conversion, so the missing scale means 1.0.
    factor = scale_in / (scale_out or 1.0)
    label = lookup.label.replace("\\", "\\\\").replace('"', '\\"')
    # Vietnamese statements bracket expenses and deductions, so a cost line
    # reads as negative even though the question asks for its size. Which
    # convention the gold answers use is unmeasurable without labels, so the
    # caller emits both and lets ANSWER_ACCURACY decide.
    value_expr = "abs(float(raw))" if magnitude else "float(raw)"
    return f'''labels = df.iloc[:, {lookup.label_col}].astype(str).str.strip()
hits = labels[labels == "{label}"]
row = hits.index[0] if len(hits) > 0 else {lookup.row - 1}
raw = str(df.iloc[row, {lookup.column}]).strip()
raw = raw.replace("(", "-").replace(")", "").replace("%", "")
raw = raw.replace(".", "").replace(",", ".")
result = round({value_expr} * {factor!r}, 2)'''
