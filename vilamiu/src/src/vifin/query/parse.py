"""P2 — turn a Vietnamese financial question into hard retrieval filters.

Company, year, and scope are recovered deterministically before any embedding
runs. That narrows 1,973 documents to a handful, which is where most of the
retrieval F2 comes from: precision is otherwise capped by how many near-identical
tables the corpus repeats across 100 tickers and 11 years.
"""

from __future__ import annotations

import os as _os

import re
from dataclasses import dataclass, field
from pathlib import Path

from vifin.query.companies import CompanyRoster

# "công ty mẹ" and "riêng" both denote the parent-only statement; 360 questions
# hinge on this single distinction.
SEPARATE_RE = re.compile(r"công ty mẹ|báo cáo tài chính riêng|\briêng\b", re.I)

YEAR_RE = re.compile(r"(?<!\d)(20[0-2]\d)(?!\d)")
# "giai đoạn 2018-2024", "giai đoạn 2018–2024", "từ 2019 đến 2023"
RANGE_RE = re.compile(r"(20[0-2]\d)\s*(?:[-–—]|đến|tới)\s*(20[0-2]\d)")

# Longest-first: "nghìn tỷ đồng" must beat "tỷ đồng".
UNIT_PATTERNS = (
    ("nghin_ty", re.compile(r"nghìn tỷ (?:đồng|vnd)", re.I)),
    ("tram_ty", re.compile(r"trăm tỷ (?:đồng|vnd)", re.I)),
    ("phan_tram", re.compile(r"phần trăm|%", re.I)),
    ("lan", re.compile(r"bao nhiêu lần", re.I)),
    ("vong", re.compile(r"bao nhiêu vòng", re.I)),
    ("trieu_usd", re.compile(r"triệu usd", re.I)),
    ("usd", re.compile(r"\busd\b", re.I)),
    ("trieu_co_phieu", re.compile(r"triệu (?:cổ phiếu|cổ phần)", re.I)),
    ("co_phieu", re.compile(r"bao nhiêu (?:cổ phiếu|cổ phần)", re.I)),
    ("ty", re.compile(r"tỷ (?:đồng|vnd)", re.I)),
    ("trieu", re.compile(r"triệu (?:đồng|vnd)", re.I)),
    ("nghin", re.compile(r"(?:nghìn|ngàn) (?:đồng|vnd)", re.I)),
    ("dong", re.compile(r"(?:bao nhiêu|bằng|là) (?:đồng|vnd|vnđ)", re.I)),
)

# Only currency scales may be rescaled by P6. Share counts, USD amounts, ratios,
# and turnover figures carry no VND conversion and must be left alone.
UNIT_SCALE = {
    "dong": 1.0,
    "nghin": 1e3,
    "trieu": 1e6,
    "ty": 1e9,
    "tram_ty": 1e11,
    "nghin_ty": 1e12,
}

# Balance-sheet questions ask for a stock at a date; P&L questions ask for a flow.
POINT_IN_TIME_RE = re.compile(r"cuối năm|đến ngày|đến cuối|tại ngày|31/12|31 tháng 12|số dư", re.I)


# Scope assumed when the question names neither "hợp nhất" nor "công ty mẹ".
DEFAULT_SCOPE = _os.environ.get("DEFAULT_SCOPE", "consolidated")


@dataclass(slots=True)
class ParsedQuestion:
    id: int
    question: str
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    scope: str = "consolidated"
    target_unit: str = ""
    point_in_time: bool = False
    multi_entity: bool = False
    multi_year: bool = False
    unresolved: bool = False

    @property
    def unit_scale(self) -> float | None:
        return UNIT_SCALE.get(self.target_unit)


def _years(question: str) -> tuple[list[int], bool]:
    years = {int(y) for y in YEAR_RE.findall(question)}
    spanned = False
    for lo, hi in RANGE_RE.findall(question):
        lo, hi = int(lo), int(hi)
        if 0 < hi - lo <= 12:
            years.update(range(lo, hi + 1))
            spanned = True
    return sorted(years), spanned


def _target_unit(question: str) -> str:
    for name, pattern in UNIT_PATTERNS:
        if pattern.search(question):
            return name
    return ""


def parse_question(qid: int, question: str, roster: CompanyRoster) -> ParsedQuestion:
    by_name = roster.match_names(question)
    by_symbol = roster.match_tickers(question)

    # A symbol hit whose core nests inside a matched company's core is an
    # artefact of the brand name, not a separate company: "Chứng khoán FPT"
    # (FTS) contains the literal token "FPT".
    name_cores = [roster.by_ticker[t].core for t in by_name]
    symbol_only = [
        t
        for t in by_symbol
        if t not in by_name
        and not any(roster.by_ticker[t].core in core for core in name_cores)
    ]
    tickers = by_name + symbol_only
    for ticker in roster.match_aliases(question):
        if ticker not in tickers:
            tickers.append(ticker)
    if not tickers:
        tickers = roster.match_partial_names(question)

    years, spanned = _years(question)
    return ParsedQuestion(
        id=qid,
        question=question,
        tickers=tickers,
        years=years,
        # When the question names neither scope, one of the two reports has to
        # be assumed — and 588 questions are in that position with BOTH reports
        # available, so this single default decides them deterministically. It
        # is the largest untested assumption in the pipeline; DEFAULT_SCOPE
        # exists to put a number on it rather than keep assuming.
        scope="separate" if SEPARATE_RE.search(question) else DEFAULT_SCOPE,
        target_unit=_target_unit(question),
        point_in_time=bool(POINT_IN_TIME_RE.search(question)),
        multi_entity=len(tickers) > 1,
        multi_year=spanned or len(years) > 1,
        unresolved=not tickers,
    )


def parse_all(questions_path: Path, roster_path: Path) -> list[ParsedQuestion]:
    import json

    roster = CompanyRoster.load(roster_path)
    parsed = []
    with questions_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                parsed.append(parse_question(row["id"], row["question"], roster))
    return parsed
