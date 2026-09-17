"""Entity extraction from Vietnamese financial questions:
ticker, years, consolidated/separate, requested unit, metric phrase."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

from ..finance.metrics import extract_metric_qualifiers, metric_keys
from ..config import YEAR_MIN, YEAR_MAX
from .metric_phrase import extract_count_metrics, extract_metric
from ..utils.viet_text import (
    fuzz_token_set,
    norm,
    strip_company_prefixes,
    strip_diacritics,
)

_BARE_TICKER = re.compile(r"(?<![a-z0-9])([a-z][a-z0-9]{1,4})(?![a-z0-9])")
_PAREN_TICKER = re.compile(r"\(([a-z][a-z0-9]{1,4})\)", re.IGNORECASE)
_YEAR = re.compile(r"(?<!\d)(20[0-2]\d)(?!\d)")
_YEAR_RANGE = re.compile(
    r"(?<!\d)(20[0-2]\d)\s*(?:[-–—]|\bden\b|\btoi\b)\s*"
    r"(?:nam\s+)?(20[0-2]\d)(?!\d)"
)

SEPARATE_HINTS = (
    "cong ty me", "bao cao tai chinh rieng", "bao cao rieng", "rieng le",
    "don le", "bctc rieng",
)
GROWTH_HINTS = (
    "tang truong", "so voi", "thay doi", "bien dong", "tang giam",
    "chenh lech", "tang bao nhieu %",
)
PERCENT_HINTS = (
    "phan tram", "ty le", "ty suat", "ty trong", "bien loi nhuan",
    "bien gop", "roe", "roa", "ros",
)
_RATIO_OUTPUT = re.compile(r"\bbao\s+nhieu\s+lan\b|\bgap\s+(?:bao\s+nhieu\s+)?lan\b")
_TURN_OUTPUT = re.compile(r"\bbao\s+nhieu\s+vong\b")
_YEAR_OUTPUT = re.compile(r"\bnam\s+nao\b")
_PERCENTAGE_POINT_OUTPUT = re.compile(r"\bdiem\s+phan\s+tram\b")
_COUNT_OUTPUT = re.compile(
    r"\bbao\s+nhieu\s+(?:(?:trong\s+so)(?:\s+cac)?(?:\s+|\b)|"
    r"(?:cong\s+ty|doanh\s+nghiep|ngan\s+hang|nam|ma|truong\s+hop)\b)"
    r"|\bco\s+bao\s+nhieu\s+don\s+vi\b"
    r"|\b(?:co\s+)?so\s+nam\b.*\b(?:la\s+)?bao\s+nhieu\b"
)

_UNIT_Q = [
    (re.compile(r"\btrieu\s+(?:co\s+phieu|co\s+phan)\b"), 1e6, "triệu cổ phiếu"),
    (re.compile(r"\b(?:nghin|ngan)\s+(?:co\s+phieu|co\s+phan)\b"), 1e3, "nghìn cổ phiếu"),
    (re.compile(r"\b(?:co\s+phieu|co\s+phan)\b"), 1.0, "cổ phiếu"),
    (re.compile(r"\btrieu\s+usd\b"), 1e6, "triệu USD"),
    (re.compile(r"(nghin|ngan)\s*ty\s*(dong)?"), 1e12, "nghìn tỷ đồng"),
    (re.compile(r"\btram\s*ty\s*(dong|vnd)?\b"), 1e11, "trăm tỷ đồng"),
    (re.compile(r"\bty\s*(dong|vnd)\b"), 1e9, "tỷ đồng"),
    (re.compile(r"bao\s*nhieu\s*ty\b"), 1e9, "tỷ đồng"),
    (re.compile(r"\btrieu\s*(dong|vnd)?\b"), 1e6, "triệu đồng"),
    (re.compile(r"(nghin|ngan)\s*(dong|vnd)\b"), 1e3, "nghìn đồng"),
    (re.compile(r"\b(dong|vnd)\b"), 1.0, "đồng"),
]

STOP_TOKENS = {
    "la", "bao", "nhieu", "cua", "trong", "vao", "tai", "ngay", "den",
    "cuoi", "dau", "nam", "va", "the", "nao", "gi", "hoi", "biet", "bang",
}
_STRIP_PHRASES = re.compile(
    r"\bla bao nhieu\b.*$|\b(cong ty me|hop nhat|rieng le|bao cao rieng|bctc"
    r"|cong ty co phan|cong ty cp|tong cong ty|ngan hang thuong mai co phan"
    r"|ngan hang tmcp|tap doan)\b|\b31\s*/\s*12(\s*/\s*\d{2,4})?\b")

_SPECIAL_ALIASES: dict[str, tuple[str, ...]] = {
    # Questions often use trade names / short forms instead of the legal names
    # from code_stock.csv. Keep this list conservative: distinctive brand names
    # only, normalized later before matching.
    "VNM": ("Vinamilk",),
    "MSN": ("Masan", "Tập đoàn Masan"),
    "MCH": ("Masan Consumer", "Hàng tiêu dùng Masan"),
    "MML": ("Masan MeatLife",),
    "MSR": ("Masan High-Tech Materials",),
    "DBC": ("Dabaco",),
    "MPC": ("Minh Phú", "Thủy sản Minh Phú"),
    "ASM": ("Sao Mai", "Tập đoàn Sao Mai"),
    "OGC": ("Đại Dương", "Tập đoàn Đại Dương"),
    "QNS": ("Đường Quảng Ngãi",),
    "DPM": ("Đạm Phú Mỹ", "Phân bón và Hóa chất Dầu khí"),
    "DCM": ("Đạm Cà Mau", "Dầu khí Cà Mau", "Phân bón Dầu khí Cà Mau"),
    "HT1": ("Vicem Hà Tiên", "Xi măng Vicem Hà Tiên"),
    "BSR": ("Bình Sơn", "Lọc hóa Dầu Bình Sơn"),
    "PVT": ("PVTrans", "Vận tải Dầu khí"),
    "GEX": ("GELEX", "Tập đoàn GELEX"),
    "GEE": ("Điện lực GELEX",),
    "NKG": ("Nam Kim", "Thép Nam Kim"),
    "PRT": ("Sản xuất Kinh doanh Xuất nhập khẩu Dịch vụ và Đầu tư Tân Bình",),
    "VIC": ("Vingroup", "Tập đoàn Vingroup"),
    "VRE": ("Vincom Retail",),
    "KBC": ("Đô thị Kinh Bắc", "Kinh Bắc"),
    "VPI": ("Văn Phú Invest", "Văn Phú"),
    "HPX": ("Hải Phát", "Đầu tư Hải Phát"),
    "MWG": ("Thế Giới Di Động",),
    "PNJ": ("Phú Nhuận", "Vàng bạc Đá quý Phú Nhuận"),
    "HAG": ("Hoàng Anh Gia Lai",),
    "HNG": ("Nông nghiệp Quốc tế Hoàng Anh Gia Lai",),
    "MBB": ("MBBank", "MB Bank", "Ngân hàng Quân đội"),
    "VCB": ("Vietcombank", "Ngoại thương Việt Nam"),
    "CTG": ("VietinBank", "Công Thương Việt Nam"),
    "BID": ("BIDV", "Đầu tư và Phát triển Việt Nam"),
    "TCB": ("Techcombank", "Kỹ thương Việt Nam"),
    "VPB": ("VPBank", "Việt Nam Thịnh Vượng"),
    "HDB": ("HDBank",),
    "EIB": ("Eximbank", "Xuất nhập khẩu Việt Nam"),
    "KLB": ("Kienlongbank", "Kiên Long"),
    "NAB": ("Nam A Bank", "Nam Á"),
    "NVB": ("NCB", "Quốc Dân"),
    "SGB": ("Saigonbank", "Sài Gòn Công Thương"),
    "MSB": ("Maritime Bank", "Hàng hải Việt Nam"),
}

_COUNTRY_SUFFIXES = (" viet nam", " vn")
_CORP_SUFFIXES = (
    " ctcp", " cong ty co phan", " cong ty cp", " group", " holdings",
    " corporation", " corp", " jsc",
)


def _core_company_aliases(name_norm: str) -> list[str]:
    """Generate a few safe, legal-word-stripped aliases from code_stock names."""
    out: list[str] = []
    candidates = [norm(name_norm), strip_company_prefixes(norm(name_norm))]
    for raw in candidates:
        base = raw
        changed = True
        while changed:
            changed = False
            for suffix in _CORP_SUFFIXES:
                if base.endswith(suffix):
                    base = base[:-len(suffix)].strip()
                    changed = True
        base = strip_company_prefixes(base)
        if base and base not in out:
            out.append(base)
        for suffix in _COUNTRY_SUFFIXES:
            if base.endswith(suffix):
                short = base[:-len(suffix)].strip()
                if _is_distinctive_company_alias(short) and short not in out:
                    out.append(short)
    return out


def _is_distinctive_company_alias(alias: str) -> bool:
    toks = alias.split()
    if not toks:
        return False
    if len(alias) >= 7:
        return True
    return len(toks) == 1 and len(toks[0]) >= 6


@dataclass
class Parsed:
    tickers: list[str] = field(default_factory=list)
    ticker_source: str = "none"        # explicit | fuzzy | none
    ticker_score: float = 0.0
    years: list[int] = field(default_factory=list)
    doc_type: str = "consolidated"
    unit_scale: float = 1.0
    unit_name: str = "đồng"
    is_percent: bool = False
    output_type: str = "number"       # number | percent | percentage_point | ratio | year | count
    growth: bool = False
    metric_norm: str = ""
    metric_wide: str = ""
    metric_variants: list[str] = field(default_factory=list)
    metric_keys: list[str] = field(default_factory=list)
    metric_qualifiers: dict[str, str] = field(default_factory=dict)


class StockMap:
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]
        tick_col = df.columns[0]
        name_col = df.columns[1]
        self.ticker2name: dict[str, str] = {
            str(r[tick_col]).strip().upper(): str(r[name_col]).strip()
            for _, r in df.iterrows()
        }
        self.tickers = set(self.ticker2name)
        # normalized aliases for fuzzy matching / name removal
        self.aliases: list[tuple[str, str]] = []
        self.aliases_of: dict[str, list[str]] = {}
        self.short_aliases: set[tuple[str, str]] = set()
        for t, name in self.ticker2name.items():
            n = norm(name)
            variants = [n]
            stripped = strip_company_prefixes(n)
            if stripped and stripped != n:
                variants.append(stripped)
            # code_stock names commonly end in "- CTCP" while questions omit
            # that legal suffix (e.g. "Tổng Công ty Khí Việt Nam").
            for base in list(variants):
                for suffix in (" ctcp", " cong ty co phan"):
                    if base.endswith(suffix):
                        short = base[:-len(suffix)].strip()
                        if short:
                            variants.append(short)
            for base in list(variants):
                variants.extend(_core_company_aliases(base))
            for alias in _SPECIAL_ALIASES.get(t, ()):
                a = norm(alias)
                if a:
                    variants.append(a)
                    self.short_aliases.add((a, t))
            variants = list(dict.fromkeys(variants))
            for v in variants:
                self.aliases.append((v, t))
            self.aliases_of[t] = sorted(variants, key=len, reverse=True)

    def find_tickers(self, question: str) -> tuple[list[str], str, float]:
        qn = norm(question)
        raw_ascii = strip_diacritics(question)

        # Dataset questions often introduce the subject as "Company name
        # (TICKER)" and may also mention counterparties in the same sentence.
        # Parenthetical exchange symbols are the strongest available signal.
        parenthetical = []
        for match in _PAREN_TICKER.finditer(raw_ascii):
            ticker = match.group(1).upper()
            if ticker in self.tickers and ticker not in parenthetical:
                parenthetical.append(ticker)
        if parenthetical:
            return parenthetical, "explicit", 100.0

        # Match all non-overlapping company aliases, longest-first at a given
        # position. Mask them before looking for ticker tokens so the "FPT" in
        # "CTCP Chứng khoán FPT" maps only to FTS, not both FTS and FPT.
        candidates = []
        for alias, ticker in self.aliases:
            if len(alias) < 7 and (alias, ticker) not in self.short_aliases:
                continue
            rex = re.compile(rf"(?<![0-9a-z]){re.escape(alias)}(?![0-9a-z])")
            for match in rex.finditer(qn):
                candidates.append((match.start(), match.end(), alias, ticker))
        candidates.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[3]))

        aliases = []
        for start, end, alias, ticker in candidates:
            if any(start < chosen_end and end > chosen_start
                   for chosen_start, chosen_end, _alias, _ticker in aliases):
                continue
            aliases.append((start, end, alias, ticker))
        aliases.sort(key=lambda item: item[0])

        masked = list(qn)
        mentions: list[tuple[int, str, str]] = []
        for start, end, _alias, ticker in aliases:
            mentions.append((start, ticker, "name"))
            masked[start:end] = " " * (end - start)

        masked_qn = "".join(masked)
        lowercase_mentions: list[tuple[int, str, str]] = []
        for match in _BARE_TICKER.finditer(masked_qn):
            ticker = match.group(1).upper()
            if ticker in self.tickers:
                raw_matches = re.finditer(
                    rf"(?<![0-9A-Za-z]){re.escape(ticker)}(?![0-9A-Za-z])",
                    raw_ascii,
                    re.IGNORECASE,
                )
                has_explicit_case = any(
                    raw.group(0).isupper() or any(ch.isdigit() for ch in raw.group(0))
                    for raw in raw_matches
                )
                if has_explicit_case:
                    mentions.append((match.start(), ticker, "ticker"))
                else:
                    lowercase_mentions.append((match.start(), ticker, "ticker"))

        # Lowercase symbols are common in comparison lists ("vnm, gas") but
        # a single lowercase token can simply be a Vietnamese word after
        # diacritic removal ("sam" after removing accents). Accept a list of
        # at least two, or a single symbol only when its role is explicit.
        lowercase_tickers = {ticker for _pos, ticker, _kind in lowercase_mentions}
        if len(lowercase_tickers) >= 2:
            mentions.extend(lowercase_mentions)
        else:
            for mention in lowercase_mentions:
                ticker_lower = mention[1].lower()
                explicit_context = re.search(
                    rf"\b(?:ma(?:\s+co\s+phieu)?|co\s+phieu)\s+{re.escape(ticker_lower)}\b"
                    rf"|\b{re.escape(ticker_lower)}\s+(?:nam|trong\s+nam|giai\s+doan|tu\s+nam)\b",
                    qn,
                )
                if explicit_context:
                    mentions.append(mention)

        # When the question explicitly asks for the parent company, aliases
        # before that phrase describe subsidiaries or counterparties.
        parent_anchor = qn.rfind("cua cong ty me")
        # A single anchor distinguishes the target from counterparties named in
        # the note. Repeated anchors instead introduce multiple peer targets;
        # scoping to the final one would silently drop the earlier companies.
        if parent_anchor >= 0 and qn.count("cua cong ty me") == 1:
            scoped_mentions = [item for item in mentions if item[0] >= parent_anchor]
            if scoped_mentions:
                mentions = scoped_mentions

        if mentions:
            mentions.sort(key=lambda item: item[0])
            found = []
            for _position, ticker, _kind in mentions:
                if ticker not in found:
                    found.append(ticker)
            has_ticker_token = any(kind == "ticker" for _pos, _ticker, kind in mentions)
            return found, ("explicit" if has_ticker_token else "explicit_name"), (
                100.0 if has_ticker_token else 99.0
            )

        # Last resort: one fuzzy company-name match. Returning several fuzzy
        # matches would fan out retrieval on weak evidence.
        best_t, best_s = None, 0.0
        for alias, t in self.aliases:
            if len(alias) < 6:
                continue
            s = fuzz_token_set(alias, qn)
            if s > best_s:
                best_t, best_s = t, s
        if best_t and best_s >= 82:
            return [best_t], "fuzzy", best_s
        return ([best_t] if best_t else []), "low_conf_fuzzy", best_s


def parse_question(question: str, stock: StockMap) -> Parsed:
    p = Parsed()
    qn = norm(question)
    answer_tail = qn.rsplit("bao nhieu", 1)[-1] if "bao nhieu" in qn else qn

    p.tickers, p.ticker_source, p.ticker_score = stock.find_tickers(question)

    p.years = _extract_years(question)

    if any(h in qn for h in SEPARATE_HINTS):
        p.doc_type = "separate"
    p.growth = any(h in qn for h in GROWTH_HINTS)
    marker = _first_output_marker(answer_tail)
    if _YEAR_OUTPUT.search(qn):
        p.output_type, p.unit_name = "year", "năm"
    elif _COUNT_OUTPUT.search(qn):
        p.output_type, p.unit_name = "count", "số lượng"
    elif _TURN_OUTPUT.search(qn):
        p.output_type, p.unit_name = "ratio", "vòng"
    elif marker == "ratio" or _RATIO_OUTPUT.search(qn):
        p.output_type, p.unit_name = "ratio", "lần"
    elif marker == "percentage_point":
        p.output_type, p.unit_name = "percentage_point", "điểm phần trăm"
    elif marker == "percent":
        p.output_type, p.unit_name = "percent", "%"
    elif marker == "number":
        # An explicit requested money unit wins over ratios/percentages used
        # only as filtering conditions earlier in a complex question.
        p.output_type = "number"
    elif any(h in qn for h in PERCENT_HINTS):
        p.output_type, p.unit_name = "percent", "%"
    p.is_percent = p.output_type == "percent"

    if p.output_type == "number":
        for scope in (answer_tail, qn):
            match = next(((scale, name) for rex, scale, name in _UNIT_Q
                          if rex.search(scope)), None)
            if match:
                p.unit_scale, p.unit_name = match
                break

    # --- metric phrase ---------------------------------------------------
    # EXTRACTIVE (cut the question down to the metric span). The old
    # SUBTRACTIVE version (question minus stopwords) left noise like
    # "so du"/"tinh dong" in the phrase and pushed correct rows below the
    # match threshold - see P1_STRATEGY_REVIEW.md for the measurements.
    aliases: list[str] = []
    for t in p.tickers:
        aliases.extend(stock.aliases_of.get(t, []))
    mp = extract_metric(question, aliases, p.tickers)
    p.metric_norm = mp.core
    p.metric_wide = mp.wide
    p.metric_variants = mp.variants()
    if p.output_type == "count":
        count_variants = extract_count_metrics(question, aliases, p.tickers)
        if count_variants:
            p.metric_norm = count_variants[0]
            p.metric_variants = _dedupe_metric_variants(
                count_variants + p.metric_variants)

    # legacy subtractive phrase kept as a last-resort variant
    m = _STRIP_PHRASES.sub(" ", qn)
    for t in p.tickers:
        for alias in stock.aliases_of.get(t, []):
            m = m.replace(alias, " ")
        m = re.sub(rf"(?<![0-9a-z]){t.lower()}(?![0-9a-z])", " ", m)
    legacy = " ".join(tok for tok in m.split()
                      if tok not in STOP_TOKENS and not _YEAR.fullmatch(tok))
    if legacy and legacy not in p.metric_variants:
        p.metric_variants.append(legacy)
    if not p.metric_norm:
        p.metric_norm = legacy
    p.metric_keys = metric_keys(
        [p.metric_norm, p.metric_wide, *p.metric_variants],
        expand_derived=False,
    )
    p.metric_qualifiers = extract_metric_qualifiers(
        question, p.metric_keys).to_dict()
    return p


def _dedupe_metric_variants(variants: list[str]) -> list[str]:
    seen, out = set(), []
    for v in variants:
        v = " ".join(str(v or "").split())
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _first_output_marker(answer_tail: str) -> str | None:
    """Choose the first explicit unit after the final ``bao nhiêu`` phrase."""
    markers: list[tuple[int, int, str]] = []
    patterns = [
        (_PERCENTAGE_POINT_OUTPUT, 0, "percentage_point"),
        (re.compile(r"\bphan\s+tram\b|%"), 1, "percent"),
        (re.compile(r"\blan\b"), 2, "ratio"),
    ]
    for rex, priority, output_type in patterns:
        match = rex.search(answer_tail)
        if match:
            markers.append((match.start(), priority, output_type))
    for rex, _scale, _name in _UNIT_Q:
        match = rex.search(answer_tail)
        if match:
            markers.append((match.start(), 3, "number"))
    return min(markers)[2] if markers else None


def _extract_years(question: str) -> list[int]:
    """Extract explicit years, expanding only genuine range syntax."""
    text = strip_diacritics(question).lower()
    range_spans: list[tuple[int, int]] = []
    events: list[tuple[int, list[int]]] = []
    for match in _YEAR_RANGE.finditer(text):
        first, last = int(match.group(1)), int(match.group(2))
        if not (YEAR_MIN <= first <= YEAR_MAX and YEAR_MIN <= last <= YEAR_MAX):
            continue
        step = 1 if last >= first else -1
        events.append((match.start(), list(range(first, last + step, step))))
        range_spans.append(match.span())

    for match in _YEAR.finditer(text):
        if any(start <= match.start() and match.end() <= end
               for start, end in range_spans):
            continue
        year = int(match.group(1))
        if YEAR_MIN <= year <= YEAR_MAX:
            events.append((match.start(), [year]))

    years = []
    for _position, values in sorted(events, key=lambda item: item[0]):
        for year in values:
            if year not in years:
                years.append(year)
    return years
