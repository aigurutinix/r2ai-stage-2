"""Build auditable single-fact candidates and an executable L1 submission.

The layer is intentionally conservative: entity/year/scope parsing is
deterministic, every proposed value points to one source table cell, and low
confidence cases remain visible in the audit rather than being hidden.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import unicodedata
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

try:
    from rapidfuzz.fuzz import ratio as fuzzy_ratio
    from rapidfuzz.fuzz import token_set_ratio
except ImportError:  # pragma: no cover - small stdlib fallback
    from difflib import SequenceMatcher

    def fuzzy_ratio(left: str, right: str) -> float:
        return 100.0 * SequenceMatcher(None, left, right).ratio()

    def token_set_ratio(left: str, right: str) -> float:
        left_tokens, right_tokens = set(left.split()), set(right.split())
        if not left_tokens or not right_tokens:
            return 0.0
        return 100.0 * len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


L1_LAST_ID = 361
YEAR_RE = re.compile(r"(?<!\d)(20\d{2}|19\d{2})(?!\d)")

TARGET_UNIT_SCALE = {
    "VND_1": 1.0,
    "VND_1e3": 1e3,
    "VND_1e6": 1e6,
    "VND_1e8": 1e8,
    "VND_1e9": 1e9,
    "VND_1e11": 1e11,
    "VND_1e12": 1e12,
    "USD_1": 1.0,
    "USD_1e6": 1e6,
    "percent": 1.0,
    "shares": 1.0,
    "VND_per_share": 1.0,
}

SOURCE_UNIT_SCALE = {
    "VND_1": 1.0,
    "VND_1e3": 1e3,
    "VND_1e6": 1e6,
    "VND_1e9": 1e9,
    "VND_1e11": 1e11,
    "VND_1e12": 1e12,
    "USD": 1.0,
    "USD_1": 1.0,
    "USD_1e3": 1e3,
    "USD_1e6": 1e6,
    "percent": 1.0,
    "shares": 1.0,
    "VND_per_share": 1.0,
}

STOPWORDS = {
    "la", "bang", "nhieu", "may", "tinh", "den", "ngay", "vao", "voi",
    "trong", "nam", "cuoi", "ky", "cua", "theo",
    "duoc", "ghi",
    "ctcp", "tap", "doan", "me", "tmcp", "ket", "thuc",
    "cao", "rieng", "moc", "thoi",
    "diem", "duoi", "dang", "lieu", "ma", "tren",
}

# Tokens that are common in questions but still carry weak financial meaning.
WEAK_TOKENS = {
    "tong", "tong cong", "so du", "gia tri", "cuoi", "dau", "chi tieu",
    "khoan", "muc", "trong nam", "den ngay",
}


@dataclass(frozen=True)
class Company:
    ticker: str
    name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class ParsedQuestion:
    id: int
    question: str
    ticker: str
    matched_company_alias: str
    year: int
    scope: str
    period_kind: str
    target_unit: str
    metric_text: str
    metric_tokens: tuple[str, ...]


@dataclass(frozen=True)
class FactCandidate:
    question_id: int
    ticker: str
    year: int
    requested_scope: str
    document_scope: str
    document_id: str
    evidence_key: str
    table_id: int
    table_ordinal: int
    page_number: Optional[int]
    source_line_0: int
    source_line_1: int
    table_kind: str
    table_title: str
    table_unit_code: str
    row_index: int
    column_index: int
    row_text: str
    column_text: str
    raw_value: str
    parsed_value: float
    source_unit: str
    source_scale: float
    target_unit: str
    target_scale: float
    value_base: float
    answer: float
    row_score: float
    period_score: float
    column_role_score: float
    schema_link_score: float
    hierarchy_score: float
    scope_score: float
    unit_score: float
    semantic_conflict_score: float
    total_score: float
    runner_up_gap: float
    confidence: str
    selection_source: str
    review_note: str


@dataclass(frozen=True)
class ManualOverride:
    question_id: int
    document_id: str
    source_line_1: int
    row_index: int
    column_index: int
    raw_value: str
    review_note: str


def normalize(text: str) -> str:
    # OCR occasionally removes spaces at a lowercase/uppercase boundary
    # (``HoàngAnh``) or between a year and its unit (``2023Nghìn``).
    # Restore those boundaries before lowercasing.
    separated = re.sub(r"(?<=[a-z])(?=[A-ZĐ])", " ", str(text))
    # Restore a boundary after an OCR-joined acronym (``VNDNăm``). Without
    # this, ``Ngàn VNDNăm trước`` becomes ``ngan vndnam truoc`` and loses its
    # printed thousand-VND scale.
    separated = re.sub(r"(?<=VND)(?=[A-ZĐ])", " ", separated)
    separated = re.sub(r"(?<=[A-Z])(?=[A-ZĐ][a-z])", " ", separated)
    separated = re.sub(r"(?<=\d)(?=[A-Za-zĐ])", " ", separated)
    value = unicodedata.normalize("NFD", separated.lower())
    value = "".join(ch for ch in value if unicodedata.category(ch) != "Mn")
    value = value.replace("đ", "d")
    return re.sub(r"[^a-z0-9%]+", " ", value).strip()


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", normalize(text)).strip()


def company_aliases(name: str, ticker: str) -> tuple[str, ...]:
    base = normalize(name)
    aliases = {base, ticker.lower()}
    prefixes = (
        "cong ty co phan ", "ctcp ", "tong cong ty co phan ",
        "tong cong ty ", "tap doan ", "ngan hang tmcp ",
        "cong ty tai chinh tong hop co phan ",
    )
    suffixes = (" cong ty co phan", " ctcp")
    changed = True
    while changed:
        changed = False
        for alias in list(aliases):
            for prefix in prefixes:
                if alias.startswith(prefix):
                    candidate = alias[len(prefix) :].strip()
                    if candidate and candidate not in aliases:
                        aliases.add(candidate)
                        changed = True
            for suffix in suffixes:
                if alias.endswith(suffix):
                    candidate = alias[: -len(suffix)].strip()
                    if candidate and candidate not in aliases:
                        aliases.add(candidate)
                        changed = True
    return tuple(sorted(aliases, key=lambda value: (-len(value), value)))


def load_companies(path: Path) -> list[Company]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        Company(
            ticker=row["Mã CK"].strip(),
            name=row["Tên công ty"].strip(),
            aliases=company_aliases(row["Tên công ty"], row["Mã CK"]),
        )
        for row in rows
    ]


def resolve_company(question: str, companies: list[Company]) -> tuple[Company, str]:
    parenthetical = re.findall(r"\(([A-Z][A-Z0-9]{1,4})\)", question)
    for ticker in reversed(parenthetical):
        for company in companies:
            if company.ticker == ticker:
                return company, ticker.lower()

    original_tokens = set(
        re.findall(r"(?<![A-Za-z0-9])[A-Z][A-Z0-9]{1,4}(?![A-Za-z0-9])", question)
    )
    explicit = [company for company in companies if company.ticker in original_tokens]

    normalized_question = normalize(question)
    matches: list[tuple[int, Company, str]] = []
    for company in companies:
        for alias in company.aliases:
            if len(alias) >= 6 and re.search(rf"(?<!\w){re.escape(alias)}(?!\w)", normalized_question):
                matches.append((len(alias), company, alias))
    if len(explicit) == 1:
        # An explicit ticker is normally the strongest signal. The exception is
        # a full legal name containing another ticker as a brand, e.g.
        # ``CTCP Chứng khoán FPT`` is ticker FTS, not FPT.
        branded_matches = [
            item
            for item in matches
            if item[1].ticker != explicit[0].ticker
            and explicit[0].ticker.lower() in item[2].split()
            and len(item[2].split()) >= 3
        ]
        if branded_matches:
            _, company, alias = max(branded_matches, key=lambda item: item[0])
            return company, alias
        return explicit[0], explicit[0].ticker.lower()
    if len(explicit) > 1:
        raise ValueError(f"Multiple explicit tickers {sorted(c.ticker for c in explicit)}")

    if matches:
        matches.sort(key=lambda item: item[0], reverse=True)
        best_length = matches[0][0]
        best = [item for item in matches if item[0] == best_length]
        tickers = {item[1].ticker for item in best}
        if len(tickers) == 1:
            _, company, alias = best[0]
            return company, alias

    raise ValueError("No company alias found")


def detect_target_unit(question: str) -> str:
    value = normalize(question)
    ordered = (
        ("nghin ty dong", "VND_1e12"),
        ("ngan ty dong", "VND_1e12"),
        ("tram ty dong", "VND_1e11"),
        ("tram trieu dong", "VND_1e8"),
        ("trieu usd", "USD_1e6"),
        ("trieu do la", "USD_1e6"),
        ("ty dong", "VND_1e9"),
        ("trieu dong", "VND_1e6"),
        ("nghin dong", "VND_1e3"),
        ("ngan dong", "VND_1e3"),
        ("dong co phieu", "VND_per_share"),
        ("phan tram", "percent"),
    )
    for phrase, unit in ordered:
        if phrase in value:
            return unit
    if "%" in question:
        return "percent"
    if "cổ phiếu" in question.lower() or "cổ phần" in question.lower():
        return "shares"
    if re.search(r"\bdong\b", value):
        return "VND_1"
    raise ValueError("Cannot determine requested unit")


def metric_surface(
    question: str, company: Company, matched_alias: str, year: int
) -> tuple[str, tuple[str, ...]]:
    value = normalize(question)
    value = re.split(r"\bbao nhieu\b", value, maxsplit=1)[0]
    # Remove one owner-company mention. Replacing every alias occurrence can
    # erase a target entity that contains the owner's name, e.g. owner
    # ``Điện Gia Lai`` versus investee ``Thủy điện Gia Lai``.
    owner_spans = []
    for alias in company.aliases:
        if len(alias) < 6 or alias == company.ticker.lower():
            continue
        owner_spans.extend(
            (match.start(), match.end(), alias)
            for match in re.finditer(rf"(?<!\w){re.escape(alias)}(?!\w)", value)
        )
    if owner_spans:
        start, end, _ = max(owner_spans, key=lambda item: (item[1], item[1] - item[0]))
        value = f"{value[:start]} {value[end:]}"
    else:
        value = value.replace(matched_alias, " ")
    value = re.sub(rf"\b{re.escape(company.ticker.lower())}\b", " ", value)
    value = re.sub(r"\bngay \d{1,2} thang \d{1,2}(?: nam \d{4})?\b", " ", value)
    value = re.sub(r"\bnam tai chinh ket thuc ngay\b", " ", value)
    value = re.sub(r"\b(?:cua )?cong ty me\b", " ", value)
    value = re.sub(
        r"\b(?:tong )?cong ty (?:co phan|cp|tnhh)(?: mot thanh vien| mtv)?\b",
        " ",
        value,
    )
    value = re.sub(r"\bduoi dang du lieu\b", " ", value)
    value = re.sub(r"\bdau (?:nam|ky)\b", " ", value)
    value = re.sub(r"\bcuoi (?:nam|ky)\b", " ", value)
    value = re.sub(rf"\b{year}\b", " ", value)
    value = re.sub(r"\b(?:31|30|01|1|12|9)\b", " ", value)
    tokens = [token for token in value.split() if token not in STOPWORDS and not token.isdigit()]
    # Preserve order, but remove duplicate boilerplate tokens.
    deduplicated = tuple(dict.fromkeys(tokens))
    if not deduplicated:
        raise ValueError("No metric tokens remain after parsing")
    return " ".join(deduplicated), deduplicated


def parse_question(row: dict, companies: list[Company]) -> ParsedQuestion:
    question = row["question"]
    company, alias = resolve_company(question, companies)
    years = [int(value) for value in YEAR_RE.findall(question)]
    if len(set(years)) != 1:
        raise ValueError(f"Expected one year, found {sorted(set(years))}")
    year = years[0]
    normalized = normalize(question)
    if "cong ty me" in normalized or "bao cao rieng" in normalized:
        scope = "separate"
    elif "hop nhat" in normalized or "bao cao hop nhat" in normalized:
        scope = "consolidated"
    else:
        scope = "consolidated"
    period_kind = (
        "start"
        if (
            "dau nam" in normalized
            or re.search(rf"\b(?:1|01) (?:1|01) {year}\b", normalized)
        )
        else "end_or_flow"
    )
    metric_text, tokens = metric_surface(question, company, alias, year)
    return ParsedQuestion(
        id=int(row["id"]),
        question=question,
        ticker=company.ticker,
        matched_company_alias=alias,
        year=year,
        scope=scope,
        period_kind=period_kind,
        target_unit=detect_target_unit(question),
        metric_text=metric_text,
        metric_tokens=tokens,
    )


def parse_number(raw: str) -> Optional[float]:
    value = str(raw).strip().replace("\xa0", " ")
    if not value or value in {"-", "–", "—"}:
        return None
    if re.search(r"[A-Za-zÀ-ỹ]", value):
        return None
    if "/" in value or ":" in value:
        return None
    negative = value.startswith("(") and value.endswith(")")
    is_percent = "%" in value
    value = value.strip("() ").replace("%", "").replace(" ", "")
    if not re.fullmatch(r"[+\-]?[0-9][0-9.,]*", value):
        return None
    if value.count(".") and value.count(","):
        decimal = "," if value.rfind(",") > value.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        value = value.replace(thousands, "")
        value = value.replace(decimal, ".")
    elif "," in value:
        parts = value.split(",")
        if len(parts) == 2 and (is_percent or len(parts[-1]) in {1, 2}):
            value = ".".join(parts)
        elif all(len(part) == 3 for part in parts[1:]):
            value = "".join(parts)
        else:
            return None
    elif "." in value:
        parts = value.split(".")
        if is_percent and len(parts) == 2:
            value = ".".join(parts)
        elif all(len(part) == 3 for part in parts[1:]):
            value = "".join(parts)
        elif len(parts) == 2 and len(parts[-1]) in {1, 2}:
            value = ".".join(parts)
        else:
            return None
    try:
        number = float(value)
    except ValueError:
        return None
    return -number if negative else number


def row_label(row: list[str]) -> str:
    parts = []
    for value in row:
        if value.strip() and parse_number(value) is None:
            parts.append(value.strip())
    return " ".join(dict.fromkeys(parts))


def contextual_row_label(grid: list[list[str]], row_index: int) -> str:
    label = row_label(grid[row_index])
    if not label:
        return ""
    for prior_row in reversed(grid[max(0, row_index - 10) : row_index]):
        if any(parse_number(value) is not None for value in prior_row):
            continue
        section = row_label(prior_row)
        unique = {normalize(value) for value in prior_row if value.strip()}
        section_norm = normalize(section)
        is_semantic_section = any(
            phrase in section_norm
            for phrase in (
                "gia tri con lai",
                "nguyen gia",
                "gia tri hao mon",
                "khau hao luy ke",
                "tien gui tiet kiem",
                "vay dai han",
                "huong lai suat co dinh",
            )
        )
        if section and is_semantic_section and len(unique) <= 2 and len(section) <= 120:
            if normalize(section) not in normalize(label):
                return f"{section} {label}"
            break
    return label


def weighted_row_score(parsed: ParsedQuestion, label: str, title: str, context: str) -> float:
    label_norm = normalize(label)
    title_norm = normalize(title)
    context_norm = normalize(context[-900:])
    query_tokens = set(parsed.metric_tokens)
    label_tokens = set(label_norm.split())
    title_tokens = set(title_norm.split())
    context_tokens = set(context_norm.split())
    if not query_tokens:
        return 0.0
    label_coverage = len(query_tokens & label_tokens) / len(query_tokens)
    context_coverage = len(query_tokens & (title_tokens | context_tokens)) / len(query_tokens)
    precision = len(query_tokens & label_tokens) / max(1, len(label_tokens))
    fuzzy = max(
        fuzzy_ratio(parsed.metric_text, label_norm),
        token_set_ratio(parsed.metric_text, label_norm),
    ) / 100.0
    substring = 1.0 if parsed.metric_text in label_norm or label_norm in parsed.metric_text else 0.0
    query_sequence = parsed.metric_text.split()
    label_sequence = label_norm.split()
    query_ngrams = {
        tuple(query_sequence[index : index + size])
        for size in (2, 3)
        for index in range(len(query_sequence) - size + 1)
    }
    label_ngrams = {
        tuple(label_sequence[index : index + size])
        for size in (2, 3)
        for index in range(len(label_sequence) - size + 1)
    }
    ngram_coverage = len(query_ngrams & label_ngrams) / max(1, len(query_ngrams))
    score = (
        48.0 * label_coverage
        + 12.0 * fuzzy
        + 8.0 * min(1.0, precision)
        + 7.0 * context_coverage
        + 8.0 * substring
        + 20.0 * ngram_coverage
    )
    metric = parsed.metric_text
    semantic_rules = (
        ("tong", ("tong", "cong"), 9.0),
        ("gia tri con lai", ("gia tri con lai", "con lai"), 18.0),
        ("nguyen gia", ("nguyen gia",), 16.0),
        ("du phong", ("du phong",), 12.0),
        ("vo hinh", ("vo hinh",), 14.0),
        ("cong ty con", ("cong ty con",), 12.0),
    )
    for requested, row_phrases, weight in semantic_rules:
        if requested in metric:
            score += weight if any(phrase in label_norm for phrase in row_phrases) else -weight
    if "von co phan" in metric and "thang du" in label_norm:
        score -= 22.0
    if "tien gui khach hang" in metric and "lai tien gui" in label_norm:
        score -= 24.0
    if "chi phi hoat dong" in metric and "ngoai hoi" in label_norm:
        score -= 20.0
    if "von chu so huu" in metric and "no phai tra" in label_norm:
        score -= 28.0
    if (
        "ben lien quan" in title_norm
        and "cong ty" not in metric
        and len(label_tokens - query_tokens) >= 4
    ):
        score -= 12.0
    if "khong kiem soat" in label_norm and "khong kiem soat" not in metric:
        score -= 30.0

    # Canonical primary-statement metrics need stronger phrase identity than
    # generic token overlap.  Without this guard, rows such as "thuế TNDN đã
    # nộp" can outrank "lợi nhuận sau thuế TNDN" merely because both contain
    # the tokens ``thuế`` and ``thu nhập doanh nghiệp``.  These rules are
    # intentionally limited to the small statement vocabulary reused by the
    # multi-stage L5 screeners.
    primary_metric_rules = (
        ("doanh thu thuan ve ban hang va cung cap dich vu", ("doanh thu thuan",),
         ("loi nhuan gop", "gia von", "doanh thu hoat dong tai chinh"), 42.0),
        ("gia von hang ban", ("gia von hang ban",),
         ("doanh thu", "giam gia hang ban"), 46.0),
        ("loi nhuan sau thue", ("loi nhuan sau thue", "loi nhuan thuan sau thue",
                                "sau thue tndn"),
         ("chua phan phoi", "thue thu nhap doanh nghiep da nop",
          "chi phi thue", "loi ich thue", "co dong cong ty me",
          "thuoc ve co dong"), 46.0),
        ("loi nhuan gop", ("loi nhuan gop",),
         ("doanh thu thuan", "gia von hang ban"), 46.0),
        ("tong loi nhuan ke toan truoc thue", ("tong loi nhuan ke toan truoc thue",),
         ("thue thu nhap", "sau thue"), 38.0),
        ("loi nhuan thuan tu hoat dong kinh doanh",
         ("loi nhuan thuan tu hoat dong kinh doanh",),
         ("luu chuyen tien", "truoc thay doi von luu dong"), 42.0),
        ("luu chuyen tien thuan tu hoat dong kinh doanh",
         ("luu chuyen tien thuan tu hoat dong kinh doanh",),
         ("loi nhuan tu hoat dong kinh doanh",), 44.0),
        ("tai san ngan han", ("tai san ngan han",),
         ("tai san dai han", "no ngan han", "tai san ngan han khac"), 46.0),
        ("no ngan han", ("no ngan han",),
         ("no dai han", "tai san ngan han", "vay va no thue tai chinh ngan han"), 40.0),
        ("no phai tra", ("no phai tra",),
         ("phai tra nguoi ban", "phai tra khac"), 34.0),
        ("von chu so huu", ("von chu so huu",),
         ("no phai tra", "von gop chu so huu"), 34.0),
        ("tien va cac khoan tuong duong tien", ("tien va cac khoan tuong duong tien",),
         ("luu chuyen tien", "tien gui co ky han", "tien gui ngan hang"), 40.0),
        ("chi phi lai vay", ("chi phi lai vay",),
         ("lai vay phai tra", "chi phi phai tra"), 38.0),
        ("chi phi ban hang", ("chi phi ban hang",),
         ("chi phi quan ly doanh nghiep",), 34.0),
        ("chi phi quan ly doanh nghiep", ("chi phi quan ly doanh nghiep",),
         ("chi phi ban hang",), 34.0),
    )
    for requested, required_phrases, conflicts, weight in primary_metric_rules:
        if requested not in metric:
            continue
        has_conflict = any(phrase in label_norm for phrase in conflicts)
        if any(phrase in label_norm for phrase in required_phrases) and not has_conflict:
            score += weight
        else:
            score -= weight
        if has_conflict:
            score -= weight
    if parsed.period_kind == "start":
        if "dau nam" in label_norm or "dau ky" in label_norm:
            score += 16.0
        if "cuoi nam" in label_norm or "cuoi ky" in label_norm:
            score -= 18.0
    else:
        if "cuoi nam" in label_norm or "cuoi ky" in label_norm:
            score += 12.0
        if "dau nam" in label_norm or "dau ky" in label_norm:
            score -= 18.0
    score -= min(8.0, 0.45 * len(label_tokens - query_tokens))
    return round(score, 4)


# Qualifiers whose presence changes the financial meaning of an otherwise
# similar row.  A generic token-overlap penalty is too weak for distinctions
# such as ``vay`` versus ``phải thu về cho vay`` or a gross balance versus its
# provision.  These are deliberately phrases, not single common words.
SEMANTIC_QUALIFIERS = (
    "du phong",
    "phai thu",
    "phai tra",
    "gia goc",
    "gia tri thuan",
    "gia tri con lai",
    "gia tri hao mon",
    "nguyen gia",
    "binh quan",
    "trich lap",
    "chi phi lai",
    "khong kiem soat",
    "thang du von co phan",
)


def semantic_conflict_score(parsed: ParsedQuestion, label: str) -> float:
    """Penalize a meaning-changing row qualifier absent from the question."""

    metric = parsed.metric_text
    value = normalize(label)
    score = 0.0
    for phrase in SEMANTIC_QUALIFIERS:
        if phrase in value and phrase not in metric:
            score -= 24.0

    # The opposite maturity is stronger evidence of a mismatch than merely
    # adding an unrequested qualifier.
    if "ngan han" in metric and "dai han" in value:
        score -= 22.0
    if "dai han" in metric and "ngan han" in value:
        score -= 22.0
    return score


def column_header(
    grid: list[list[str]], row_index: int, column_index: int, header_row_count: int
) -> str:
    values = []
    # HTML ``th`` detection sometimes under-counts a multi-row header.  Extend
    # through leading non-numeric rows, but stop before the first data row so a
    # prior period/value cannot leak into every later column header.
    header_rows = []
    for index, row in enumerate(grid[: min(row_index, 5)]):
        declared_header = index < header_row_count
        contains_number = any(
            parse_number(value) is not None
            and not YEAR_RE.fullmatch(value.strip())
            for value in row
        )
        if contains_number and not declared_header:
            break
        header_rows.append(row)
    for row in header_rows:
        if column_index < len(row):
            value = row[column_index].strip()
            if value and value not in values:
                values.append(value)
    return " ".join(values)


def is_plain_small_integer(raw: str, number: float) -> bool:
    value = raw.strip().strip("() ")
    return abs(number) <= 999 and bool(re.fullmatch(r"[+\-]?\d{1,3}", value))


def column_role_score(
    parsed: ParsedQuestion,
    header: str,
    raw: str,
    number: float,
    column_index: int,
    numeric_cells: list[tuple[int, str, float]],
) -> float:
    value = normalize(header)
    score = 0.0
    if any(phrase in value for phrase in ("ma so", "thuyet minh", "stt", "ghi chu")):
        score -= 35.0
    if any(phrase in value for phrase in ("nam nay", "ky nay", str(parsed.year), "cuoi ky")):
        score += 5.0
    header_is_total = bool(re.search(r"\btong(?: cong)?\b", value))
    if header_is_total:
        # A broad metric normally refers to the aggregate column unless the
        # question names a component carried by another header.
        score += 10.0
        if "tong" in parsed.metric_tokens:
            score += 10.0
    if "tong" in parsed.metric_tokens and len(numeric_cells) >= 4:
        if column_index == numeric_cells[-1][0]:
            score += 12.0
    if parsed.target_unit == "percent":
        score += 12.0 if "%" in raw or any(
            phrase in value for phrase in ("ty le", "phan tram", "%")
        ) else -30.0
    elif is_plain_small_integer(raw, number):
        later_large_value = any(
            other_column > column_index and abs(other_number) >= 10_000
            for other_column, _, other_number in numeric_cells
        )
        if later_large_value:
            score -= 45.0
    return score


def schema_link_score(parsed: ParsedQuestion, label: str, header: str) -> float:
    label_tokens = set(normalize(label).split())
    header_tokens = set(normalize(header).split())
    missing_from_label = set(parsed.metric_tokens) - label_tokens
    linked = missing_from_label & header_tokens
    return min(24.0, 6.0 * len(linked))


def lexical_cell_coverage(
    parsed: ParsedQuestion,
    label: str,
    header: str,
    title: str,
    context: str,
) -> float:
    """Coverage used to guard unit inference on otherwise unitless tables."""

    query_tokens = set(parsed.metric_tokens)
    if not query_tokens:
        return 0.0
    evidence_tokens = set(
        normalize(" ".join((label, header, title, context[-500:]))).split()
    )
    return len(query_tokens & evidence_tokens) / len(query_tokens)


def period_score(
    parsed: ParsedQuestion,
    header: str,
    value_column_rank: int,
    label: str = "",
) -> float:
    value = normalize(header)
    row_value = normalize(label)
    score = max(0.0, 4.0 - value_column_rank)
    is_current_year_start = bool(
        re.search(rf"(?<!\d)(?:1|01) (?:1|01) {parsed.year}(?!\d)", value)
    )
    if parsed.period_kind == "start":
        if (
            "dau nam" in value
            or "dau ky" in value
            or re.search(r"\b(?:1|01) thang (?:1|01)\b", value)
        ):
            score += 15.0
        if (
            ("dau nam" in row_value or "dau ky" in row_value)
            and re.search(rf"(?<!\d){parsed.year}(?!\d)", value)
        ):
            score += 15.0
        elif re.search(rf"(?<!\d){parsed.year - 1}(?!\d)", value):
            score += 12.0
        if is_current_year_start:
            score += 15.0
        if (
            "cuoi nam" in value
            or "cuoi ky" in value
            or "31 thang 12" in value
        ):
            score -= 8.0
    else:
        if str(parsed.year) in value:
            score += 15.0
        if (
            "nam nay" in value
            or "ky nay" in value
            or "cuoi nam" in value
            or "cuoi ky" in value
            or "31 thang 12" in value
        ):
            score += 10.0
        if (
            str(parsed.year - 1) in value
            or "nam truoc" in value
            or "ky truoc" in value
            or "dau nam" in value
            or "1 thang 1" in value
            or "01 thang 01" in value
        ):
            score -= 8.0
        if is_current_year_start:
            score -= 18.0
    return score


def table_period_score(parsed: ParsedQuestion, title: str) -> float:
    """Prefer an explicitly current-period table over a comparative table.

    OCR reports commonly place otherwise identical current- and prior-year
    movement tables next to each other. Column headers such as ``Tại ngày cuối
    năm`` do not disambiguate them, so the table title must participate in the
    temporal score. Titles without a year remain neutral.
    """

    normalized_title = normalize(title)
    # Keep this deliberately narrow. Generic titles such as ``Năm 2023`` or
    # ``tại ngày 31/12/2023`` occur on many unrelated statement fragments and
    # are not strong enough to move a candidate. The duplicated movement-table
    # pattern is the audited failure mode this feature is meant to resolve.
    if "bien dong" not in normalized_title or "trong nam" not in normalized_title:
        return 0.0
    years = {int(year) for year in YEAR_RE.findall(str(title))}
    if not years:
        return 0.0
    if parsed.year in years:
        return 8.0
    return -8.0


def infer_cell_source_unit(
    table_unit: str,
    unit_text: str,
    header: str,
    raw: str,
    label: str,
    target_unit: str,
    document_default_unit: str = "unknown",
) -> str:
    def mentions_plain_vnd(value: str) -> bool:
        # ``đồng`` also means a contract/agreement (for example ``hợp đồng``).
        # Treat it as the currency only when it appears as an explicit unit.
        return (
            "vnd" in value
            or value.strip() == "dong"
            or bool(re.search(r"\b(?:bang|don vi(?: tinh)?|dvt)[: ]+dong\b", value))
        )

    local_value = normalize(" ".join((header, raw)))
    value = normalize(" ".join((unit_text, header, raw)))
    label_value = normalize(label)
    header_and_label = normalize(" ".join((header, label)))
    if any(
        phrase in header_and_label
        for phrase in (
            "so co phieu",
            "so co phan",
            "so luong co phieu",
            "so luong co phan",
        )
    ):
        return "shares"
    if target_unit == "VND_per_share" or any(
        phrase in header_and_label
        for phrase in ("vnd co phieu", "dong co phieu", "lai tren moi co phieu")
    ):
        return "VND_1"
    if "%" in raw or "phan tram" in value:
        return "percent"
    # Percentage schedules often put the unit only in a title outside the
    # extracted grid.  Once the question explicitly requests a percentage,
    # treat an otherwise plain numeric cell as percent.
    if target_unit == "percent":
        return "percent"
    if target_unit == "shares" and (
        "so luong" in normalize(header)
        or "so luong" in label_value
        or (table_unit == "shares" and "vnd" not in normalize(header))
    ):
        return "shares"
    # Currency-position tables name columns USD/EUR even when the statement
    # explicitly says every amount was converted to VND.  Vietnamese bank
    # statements report those converted figures in the document's default
    # million-VND scale.
    if (
        target_unit.startswith("VND")
        and document_default_unit.startswith("VND")
        and ("quy doi sang vnd" in value or table_unit == "USD")
    ):
        return document_default_unit
    # Prefer the unit printed in the selected column. OCR context can contain
    # a unit carried over from a neighboring table.
    if "trieu usd" in local_value:
        return "USD_1e6"
    if "nghin usd" in local_value or "ngan usd" in local_value:
        return "USD_1e3"
    if "usd" in local_value or "do la my" in local_value:
        return "USD_1"
    if "nghin ty" in local_value or "ngan ty" in local_value:
        return "VND_1e12"
    if "tram ty" in local_value:
        return "VND_1e11"
    if re.search(r"\bty (?:vnd|dong)\b", local_value):
        return "VND_1e9"
    if "trieu" in local_value and ("vnd" in local_value or "dong" in local_value):
        return "VND_1e6"
    if "vnd million" in local_value or "million vnd" in local_value:
        return "VND_1e6"
    if re.search(r"\b(?:nghin|ngan) (?:vnd|dong)\b", local_value):
        return "VND_1e3"
    if "vnd thousand" in local_value or "thousand vnd" in local_value:
        return "VND_1e3"
    if mentions_plain_vnd(local_value):
        return "VND_1"
    if "trieu usd" in value:
        return "USD_1e6"
    if "nghin usd" in value or "ngan usd" in value:
        return "USD_1e3"
    if "usd" in value or "do la my" in value:
        return "USD_1"
    if "nghin ty" in value or "ngan ty" in value:
        return "VND_1e12"
    if "tram ty" in value:
        return "VND_1e11"
    if re.search(r"\bty (?:vnd|dong)\b", value):
        return "VND_1e9"
    if "trieu" in value and ("vnd" in value or "dong" in value):
        return "VND_1e6"
    if "vnd million" in value or "million vnd" in value:
        return "VND_1e6"
    if re.search(r"\b(?:nghin|ngan) (?:vnd|dong)\b", value):
        return "VND_1e3"
    if "vnd thousand" in value or "thousand vnd" in value:
        return "VND_1e3"
    # Vietnamese bank statements conventionally report in million VND.  A
    # neighboring OCR table can leak a bare ``VND`` into unit_text; do not let
    # that weak signal override the document-level bank convention.
    if document_default_unit == "VND_1e6" and table_unit in {"unknown", "VND_1"}:
        return "VND_1e6"
    if mentions_plain_vnd(value):
        return "VND_1"
    if "co phieu" in value or "co phan" in value:
        return "shares"
    separators = raw.count(".") + raw.count(",")
    if not unit_text.strip() and separators >= 2 and table_unit.startswith("VND"):
        return "VND_1"
    # A sizeable minority of OCR tables omit the printed unit even though the
    # cells are VND-formatted amounts.  Only use this fallback for a monetary
    # question, after excluding explicit share/percent/USD headers above.
    if (
        table_unit == "unknown"
        and target_unit.startswith("VND")
        and separators >= 2
    ):
        if document_default_unit.startswith("VND"):
            return document_default_unit
        return "VND_1"
    # A cash-flow table may contain a share-issuance row and therefore be
    # classified globally as ``shares``.  That table-level guess must not
    # override a reviewed monetary row such as net financing cash flow.
    if (
        table_unit == "shares"
        and target_unit.startswith("VND")
        and "co phieu" not in header_and_label
        and "co phan" not in header_and_label
        and separators >= 2
    ):
        return (
            document_default_unit
            if document_default_unit.startswith("VND")
            else "VND_1"
        )
    if table_unit == "USD":
        return "USD_1"
    return table_unit


def units_compatible(source: str, target: str) -> bool:
    if source.startswith("VND") and target.startswith("VND"):
        return True
    if source.startswith("USD") and target.startswith("USD"):
        return True
    if source == target and source in {"percent", "shares"}:
        return True
    if source.startswith("VND") and target == "VND_per_share":
        return True
    return False


STANDARD_PRIMARY_METRICS = (
    "tong tai san",
    "tong no phai tra",
    "von chu so huu",
    "loi nhuan sau thue",
    "loi nhuan truoc thue",
    "doanh thu thuan",
    "tien va cac khoan tuong duong tien",
    "luu chuyen tien thuan",
    "chi phi quan ly",
    "chi phi ban hang",
    "chi phi hoat dong",
)


class FactRetriever:
    def __init__(self, database: Path, bank_tickers: Optional[set[str]] = None) -> None:
        self.connection = sqlite3.connect(str(database))
        self.connection.row_factory = sqlite3.Row
        self.cache: dict[tuple[str, int], list[sqlite3.Row]] = {}
        self.bank_tickers = bank_tickers or set()

    def close(self) -> None:
        self.connection.close()

    def tables_for(self, ticker: str, year: int) -> list[sqlite3.Row]:
        key = (ticker, year)
        if key not in self.cache:
            self.cache[key] = list(
                self.connection.execute(
                    """
                    SELECT t.*, d.report_scope
                    FROM tables t JOIN documents d ON d.document_id = t.document_id
                    WHERE d.ticker = ? AND d.year = ?
                    ORDER BY t.document_id, t.table_ordinal
                    """,
                    key,
                )
            )
        return self.cache[key]

    def retrieve_reviewed(
        self, parsed: ParsedQuestion, override: ManualOverride
    ) -> FactCandidate:
        """Materialize one human-reviewed cell without lexical-score gating."""

        if override.question_id != parsed.id:
            raise ValueError(
                f"Reviewed q{override.question_id} cannot be applied to q{parsed.id}"
            )
        table = next(
            (
                row
                for row in self.tables_for(parsed.ticker, parsed.year)
                if row["document_id"] == override.document_id
                and row["source_line_1"] == override.source_line_1
            ),
            None,
        )
        if table is None:
            raise ValueError(
                f"q{parsed.id}: reviewed table {override.document_id} "
                f"line {override.source_line_1} was not found"
            )
        grid = json.loads(table["grid_json"])
        try:
            raw = str(grid[override.row_index][override.column_index]).strip()
        except IndexError as error:
            raise ValueError(f"q{parsed.id}: reviewed cell is outside the table grid") from error
        if raw != override.raw_value:
            raise ValueError(
                f"q{parsed.id}: reviewed raw value changed: "
                f"expected {override.raw_value!r}, found {raw!r}"
            )
        number = parse_number(raw)
        # A dash or an empty current-period cell is the source report's
        # representation of no amount.  Only an explicit, human-reviewed
        # override may promote such a cell to zero; automatic retrieval still
        # ignores it because parse_number deliberately returns None.
        reviewed_missing_zero = raw in {"", "-", "–", "—"}
        if number is None and not reviewed_missing_zero:
            raise ValueError(f"q{parsed.id}: reviewed cell is not numeric: {raw!r}")
        if reviewed_missing_zero:
            number = 0.0

        label = contextual_row_label(grid, override.row_index)
        header = column_header(
            grid,
            override.row_index,
            override.column_index,
            int(table["header_row_count"]),
        )
        numeric_cells = [
            (column_index, value, parsed_number)
            for column_index, value in enumerate(grid[override.row_index])
            if (parsed_number := parse_number(value)) is not None
        ]
        if reviewed_missing_zero:
            numeric_cells.append((override.column_index, raw, 0.0))
            numeric_cells.sort(key=lambda cell: cell[0])
        value_rank = next(
            index
            for index, (column_index, _value, _number) in enumerate(numeric_cells)
            if column_index == override.column_index
        )
        document_scope = table["report_scope"]
        if document_scope == parsed.scope:
            scope_score = 12.0
        elif document_scope in {"unspecified", "aggregated"}:
            scope_score = 2.0
        else:
            scope_score = -10.0
        row_score = weighted_row_score(
            parsed, label, table["title_hint"], table["context_before"]
        )
        temporal_score = period_score(parsed, header, value_rank, label)
        temporal_score += table_period_score(parsed, table["title_hint"])
        role_score = column_role_score(
            parsed, header, raw, number, override.column_index, numeric_cells
        )
        link_score = schema_link_score(parsed, label, header)
        hierarchy_score = 20.0 * lexical_cell_coverage(
            parsed,
            label,
            header,
            table["title_hint"],
            table["context_before"],
        )
        conflict_score = semantic_conflict_score(parsed, label)
        source_unit = infer_cell_source_unit(
            table["unit_code"],
            table["unit_text"],
            header,
            raw,
            label,
            parsed.target_unit,
            "VND_1e6" if parsed.ticker in self.bank_tickers else "VND_1",
        )
        if source_unit not in SOURCE_UNIT_SCALE or not units_compatible(
            source_unit, parsed.target_unit
        ):
            raise ValueError(
                f"q{parsed.id}: reviewed cell unit {source_unit!r} is incompatible "
                f"with {parsed.target_unit!r}"
            )
        unit_score = 8.0
        kind_score = {
            "primary_balance_sheet": 3.0,
            "primary_income_statement": 3.0,
            "primary_cash_flow": 3.0,
            "financial_notes": 2.0,
            "other": 0.0,
            "governance_or_company_info": -1.0,
            "contents": -20.0,
            "audit_report": -20.0,
        }.get(table["table_kind"], 0.0)
        if any(metric in parsed.metric_text for metric in STANDARD_PRIMARY_METRICS):
            if table["table_kind"].startswith("primary_"):
                kind_score += 12.0
            elif table["table_kind"] == "financial_notes":
                kind_score -= 3.0
        total = (
            row_score
            + temporal_score
            + role_score
            + link_score
            + hierarchy_score
            + scope_score
            + unit_score
            + kind_score
            + conflict_score
        )
        source_scale = SOURCE_UNIT_SCALE[source_unit]
        target_scale = TARGET_UNIT_SCALE[parsed.target_unit]
        value_base = number * source_scale
        return FactCandidate(
            question_id=parsed.id,
            ticker=parsed.ticker,
            year=parsed.year,
            requested_scope=parsed.scope,
            document_scope=document_scope,
            document_id=table["document_id"],
            evidence_key=table["evidence_key"],
            table_id=table["table_id"],
            table_ordinal=table["table_ordinal"],
            page_number=table["page_number"],
            source_line_0=table["source_line_0"],
            source_line_1=table["source_line_1"],
            table_kind=table["table_kind"],
            table_title=table["title_hint"],
            table_unit_code=table["unit_code"],
            row_index=override.row_index,
            column_index=override.column_index,
            row_text=label,
            column_text=header,
            raw_value=raw,
            parsed_value=number,
            source_unit=source_unit,
            source_scale=source_scale,
            target_unit=parsed.target_unit,
            target_scale=target_scale,
            value_base=value_base,
            answer=value_base / target_scale,
            row_score=row_score,
            period_score=temporal_score,
            column_role_score=role_score,
            schema_link_score=link_score,
            hierarchy_score=round(hierarchy_score, 4),
            scope_score=scope_score,
            unit_score=unit_score,
            semantic_conflict_score=conflict_score,
            total_score=round(total, 4),
            runner_up_gap=0.0,
            confidence="high",
            selection_source="manual",
            review_note=override.review_note,
        )

    def retrieve(self, parsed: ParsedQuestion, limit: int = 5) -> list[FactCandidate]:
        scored: list[dict] = []
        for table in self.tables_for(parsed.ticker, parsed.year):
            document_scope = table["report_scope"]
            if document_scope == parsed.scope:
                scope_score = 12.0
            elif document_scope in {"unspecified", "aggregated"}:
                scope_score = 2.0
            else:
                scope_score = -10.0
            try:
                grid = json.loads(table["grid_json"])
            except json.JSONDecodeError:
                continue
            for row_index, row in enumerate(grid):
                label = contextual_row_label(grid, row_index)
                if not label:
                    continue
                row_score = weighted_row_score(
                    parsed, label, table["title_hint"], table["context_before"]
                )
                conflict_score = semantic_conflict_score(parsed, label)
                # Generic total rows often carry their metric in the table title
                # and their measure (for example ``Giá gốc``) in the column.
                # Keep a small lexical floor and let schema-link scoring decide.
                if row_score < 8.0:
                    continue
                numeric_cells = []
                for column_index, raw in enumerate(row):
                    number = parse_number(raw)
                    if number is not None:
                        numeric_cells.append((column_index, raw, number))
                for value_rank, (column_index, raw, number) in enumerate(numeric_cells):
                    header = column_header(
                        grid, row_index, column_index, int(table["header_row_count"])
                    )
                    temporal_score = period_score(parsed, header, value_rank, label)
                    temporal_score += table_period_score(parsed, table["title_hint"])
                    role_score = column_role_score(
                        parsed, header, raw, number, column_index, numeric_cells
                    )
                    link_score = schema_link_score(parsed, label, header)
                    hierarchy_score = 20.0 * lexical_cell_coverage(
                        parsed,
                        label,
                        header,
                        table["title_hint"],
                        table["context_before"],
                    )
                    source_unit = infer_cell_source_unit(
                        table["unit_code"], table["unit_text"], header, raw,
                        label, parsed.target_unit,
                        "VND_1e6" if parsed.ticker in self.bank_tickers else "VND_1",
                    )
                    if (
                        table["unit_code"] == "unknown"
                        and source_unit.startswith("VND")
                        and lexical_cell_coverage(
                            parsed,
                            label,
                            header,
                            table["title_hint"],
                            table["context_before"],
                        ) < 0.6
                    ):
                        continue
                    compatible = units_compatible(source_unit, parsed.target_unit)
                    unit_score = 8.0 if compatible else -25.0
                    if source_unit not in SOURCE_UNIT_SCALE:
                        unit_score = -30.0
                    kind_score = {
                        "primary_balance_sheet": 3.0,
                        "primary_income_statement": 3.0,
                        "primary_cash_flow": 3.0,
                        "financial_notes": 2.0,
                        "other": 0.0,
                        "governance_or_company_info": -1.0,
                        "contents": -20.0,
                        "audit_report": -20.0,
                    }.get(table["table_kind"], 0.0)
                    if any(metric in parsed.metric_text for metric in STANDARD_PRIMARY_METRICS):
                        if table["table_kind"].startswith("primary_"):
                            kind_score += 12.0
                        elif table["table_kind"] == "financial_notes":
                            kind_score -= 3.0
                    total = (
                        row_score
                        + temporal_score
                        + role_score
                        + link_score
                        + hierarchy_score
                        + scope_score
                        + unit_score
                        + kind_score
                        + conflict_score
                    )
                    if source_unit not in SOURCE_UNIT_SCALE or not compatible:
                        continue
                    source_scale = SOURCE_UNIT_SCALE[source_unit]
                    target_scale = TARGET_UNIT_SCALE[parsed.target_unit]
                    value_base = number * source_scale
                    answer = value_base / target_scale
                    scored.append(
                        {
                            "question_id": parsed.id,
                            "ticker": parsed.ticker,
                            "year": parsed.year,
                            "requested_scope": parsed.scope,
                            "document_scope": document_scope,
                            "document_id": table["document_id"],
                            "evidence_key": table["evidence_key"],
                            "table_id": table["table_id"],
                            "table_ordinal": table["table_ordinal"],
                            "page_number": table["page_number"],
                            "source_line_0": table["source_line_0"],
                            "source_line_1": table["source_line_1"],
                            "table_kind": table["table_kind"],
                            "table_title": table["title_hint"],
                            "table_unit_code": table["unit_code"],
                            "row_index": row_index,
                            "column_index": column_index,
                            "row_text": label,
                            "column_text": header,
                            "raw_value": raw,
                            "parsed_value": number,
                            "source_unit": source_unit,
                            "source_scale": source_scale,
                            "target_unit": parsed.target_unit,
                            "target_scale": target_scale,
                            "value_base": value_base,
                            "answer": answer,
                            "row_score": row_score,
                            "period_score": temporal_score,
                            "column_role_score": role_score,
                            "schema_link_score": link_score,
                            "hierarchy_score": round(hierarchy_score, 4),
                            "scope_score": scope_score,
                            "unit_score": unit_score,
                            "semantic_conflict_score": conflict_score,
                            "total_score": round(total, 4),
                        }
                    )
        scored.sort(key=lambda candidate: candidate["total_score"], reverse=True)
        results = []
        for index, candidate in enumerate(scored[:limit]):
            runner = scored[index + 1]["total_score"] if index + 1 < len(scored) else 0.0
            gap = candidate["total_score"] - runner
            if candidate["row_score"] >= 72 and candidate["period_score"] >= 8 and gap >= 5:
                confidence = "high"
            elif candidate["row_score"] >= 55 and candidate["period_score"] >= 0:
                confidence = "medium"
            else:
                confidence = "low"
            candidate["runner_up_gap"] = round(gap, 4)
            candidate["confidence"] = confidence
            candidate["selection_source"] = "automatic"
            candidate["review_note"] = ""
            results.append(FactCandidate(**candidate))
        return results


def load_l1_questions(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if int(row["id"]) <= L1_LAST_ID:
                rows.append(row)
    if [row["id"] for row in rows] != list(range(1, L1_LAST_ID + 1)):
        raise ValueError("Expected the complete L1 ID range 1..361")
    return rows


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manual_overrides(path: Optional[Path]) -> dict[int, ManualOverride]:
    if path is None or not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "question_id", "document_id", "source_line_1", "row_index",
        "column_index", "raw_value", "review_note",
    }
    if rows and set(rows[0]) != required:
        raise ValueError(f"Manual override columns must be {sorted(required)}")
    overrides = {}
    for row in rows:
        override = ManualOverride(
            question_id=int(row["question_id"]),
            document_id=row["document_id"],
            source_line_1=int(row["source_line_1"]),
            row_index=int(row["row_index"]),
            column_index=int(row["column_index"]),
            raw_value=row["raw_value"],
            review_note=row["review_note"],
        )
        if override.question_id in overrides:
            raise ValueError(f"Duplicate manual override q{override.question_id}")
        overrides[override.question_id] = override
    return overrides


FACT_EXPORT_FIELDS = (
    "question_id",
    "candidate_rank",
    "confidence",
    "selection_source",
    "review_note",
    "ticker",
    "year",
    "requested_scope",
    "document_scope",
    "document_id",
    "evidence_key",
    "table_id",
    "table_ordinal",
    "page_number",
    "source_line_0",
    "source_line_1",
    "table_kind",
    "table_title",
    "row_index",
    "column_index",
    "row_text",
    "column_text",
    "raw_value",
    "parsed_value",
    "source_unit",
    "source_scale",
    "target_unit",
    "target_scale",
    "value_base",
    "answer",
    "row_score",
    "period_score",
    "column_role_score",
    "schema_link_score",
    "hierarchy_score",
    "scope_score",
    "unit_score",
    "semantic_conflict_score",
    "total_score",
    "runner_up_gap",
    "metric_text",
    "question",
)


def write_fact_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FACT_EXPORT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in FACT_EXPORT_FIELDS})


def write_fact_database(
    path: Path,
    questions_path: Path,
    parsed_rows: list[ParsedQuestion],
    candidate_rows: list[dict],
    parse_errors: list[dict],
) -> None:
    if path.exists():
        path.unlink()
    connection = sqlite3.connect(str(path))
    try:
        connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE questions (
                question_id INTEGER PRIMARY KEY,
                question TEXT NOT NULL,
                ticker TEXT,
                year INTEGER,
                requested_scope TEXT,
                period_kind TEXT,
                target_unit TEXT,
                metric_text TEXT,
                parse_status TEXT NOT NULL,
                parse_error TEXT,
                parsed_json TEXT
            );
            CREATE TABLE fact_candidates (
                question_id INTEGER NOT NULL REFERENCES questions(question_id),
                candidate_rank INTEGER NOT NULL,
                confidence TEXT NOT NULL,
                selection_source TEXT NOT NULL,
                review_note TEXT NOT NULL,
                document_id TEXT NOT NULL,
                evidence_key TEXT NOT NULL,
                table_id INTEGER NOT NULL,
                page_number INTEGER,
                source_line_0 INTEGER NOT NULL,
                source_line_1 INTEGER NOT NULL,
                row_index INTEGER NOT NULL,
                column_index INTEGER NOT NULL,
                row_text TEXT NOT NULL,
                column_text TEXT NOT NULL,
                raw_value TEXT NOT NULL,
                source_unit TEXT NOT NULL,
                value_base REAL NOT NULL,
                target_unit TEXT NOT NULL,
                answer REAL NOT NULL,
                total_score REAL NOT NULL,
                candidate_json TEXT NOT NULL,
                PRIMARY KEY (question_id, candidate_rank)
            );
            CREATE INDEX idx_fact_document ON fact_candidates(document_id, table_id);
            CREATE INDEX idx_fact_confidence ON fact_candidates(confidence, candidate_rank);
            """
        )
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            (
                ("layer", "L1_single_fact"),
                ("question_id_range", "1..361"),
                (
                    "selection",
                    "candidate_rank=1 is automatic unless selection_source=manual; not a gold label",
                ),
                ("provenance", "evidence_key=document_id|source_line_1"),
            ),
        )
        parsed_by_id = {row.id: row for row in parsed_rows}
        errors_by_id = {int(row["id"]): row["error"] for row in parse_errors}
        for question in load_l1_questions(questions_path):
            question_id = int(question["id"])
            parsed = parsed_by_id.get(question_id)
            connection.execute(
                """
                INSERT INTO questions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    question_id,
                    question["question"],
                    parsed.ticker if parsed else None,
                    parsed.year if parsed else None,
                    parsed.scope if parsed else None,
                    parsed.period_kind if parsed else None,
                    parsed.target_unit if parsed else None,
                    parsed.metric_text if parsed else None,
                    "parsed" if parsed else "error",
                    errors_by_id.get(question_id),
                    json.dumps(asdict(parsed), ensure_ascii=False) if parsed else None,
                ),
            )
        connection.executemany(
            """
            INSERT INTO fact_candidates VALUES (
                :question_id, :candidate_rank, :confidence,
                :selection_source, :review_note, :document_id,
                :evidence_key, :table_id, :page_number, :source_line_0,
                :source_line_1, :row_index, :column_index, :row_text,
                :column_text, :raw_value, :source_unit, :value_base,
                :target_unit, :answer, :total_score, :candidate_json
            )
            """,
            (
                {
                    **row,
                    "candidate_json": json.dumps(row, ensure_ascii=False, separators=(",", ":")),
                }
                for row in candidate_rows
            ),
        )
        connection.commit()
    finally:
        connection.close()


def run_candidates(
    questions_path: Path,
    companies_path: Path,
    database: Path,
    output_dir: Path,
    overrides_path: Optional[Path] = None,
) -> None:
    companies = load_companies(companies_path)
    parsed_rows = []
    parse_errors = []
    for row in load_l1_questions(questions_path):
        try:
            parsed_rows.append(parse_question(row, companies))
        except ValueError as error:
            parse_errors.append({"id": row["id"], "question": row["question"], "error": str(error)})
    write_jsonl(output_dir / "parse_errors.jsonl", parse_errors)
    write_jsonl(output_dir / "parsed_questions.jsonl", (asdict(row) for row in parsed_rows))

    bank_tickers = {
        company.ticker
        for company in companies
        if "ngan hang" in normalize(company.name)
    }
    manual_overrides = load_manual_overrides(overrides_path)
    retriever = FactRetriever(database, bank_tickers=bank_tickers)
    candidate_rows = []
    top_rows = []
    try:
        for index, parsed in enumerate(parsed_rows, 1):
            override = manual_overrides.get(parsed.id)
            candidates = retriever.retrieve(parsed, limit=5)
            if override:
                reviewed = retriever.retrieve_reviewed(parsed, override)
                alternatives = [
                    candidate
                    for candidate in candidates
                    if not (
                        candidate.document_id == reviewed.document_id
                        and candidate.source_line_1 == reviewed.source_line_1
                        and candidate.row_index == reviewed.row_index
                        and candidate.column_index == reviewed.column_index
                    )
                ]
                candidates = [reviewed, *alternatives[:4]]
            for rank, candidate in enumerate(candidates, 1):
                record = asdict(candidate)
                record["candidate_rank"] = rank
                record["question"] = parsed.question
                record["metric_text"] = parsed.metric_text
                candidate_rows.append(record)
            if candidates:
                top_rows.append(candidate_rows[-len(candidates)])
            if index % 50 == 0:
                print(f"retrieved {index}/{len(parsed_rows)} questions", flush=True)
    finally:
        retriever.close()
    write_jsonl(output_dir / "candidates.jsonl", candidate_rows)
    write_jsonl(output_dir / "top_facts.jsonl", top_rows)
    write_fact_csv(output_dir / "top_facts.csv", top_rows)
    write_fact_database(
        output_dir / "l1_facts.sqlite",
        questions_path,
        parsed_rows,
        candidate_rows,
        parse_errors,
    )
    summary = {
        "questions": L1_LAST_ID,
        "parsed": len(parsed_rows),
        "parse_errors": len(parse_errors),
        "with_candidate": len(top_rows),
        "without_candidate": len(parsed_rows) - len(top_rows),
        "confidence": dict(Counter(row["confidence"] for row in top_rows)),
        "manual_overrides": sum(
            row["selection_source"] == "manual" for row in top_rows
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


def pandas_query_for(
    raw: str,
    row_index: int,
    column_index: int,
    source_scale: float,
    target_scale: float,
) -> str:
    value = str(raw).strip()
    if value in {"", "-", "–", "—"}:
        # This remains data-dependent: changing the cited CSV cell to a number
        # changes the result.  Missing/dash values are merely normalized to the
        # accounting value zero at execution time.
        expression = (
            f"pd.to_numeric(pd.Series([df1.iloc[{row_index}, {column_index}]]), "
            "errors=\"coerce\").fillna(0.0).iloc[0]"
        )
        return f"float({expression}) * {source_scale:.1f} / {target_scale:.1f}"
    is_percent = "%" in value
    negative = value.startswith("(") and value.endswith(")")
    expression = f'str(df1.iloc[{row_index}, {column_index}]).strip().strip("()%")'
    expression += '.replace(" ", "")'
    clean = value.strip("() %").replace(" ", "")
    if "." in clean and "," in clean:
        decimal = "," if clean.rfind(",") > clean.rfind(".") else "."
        thousands = "." if decimal == "," else ","
        expression += f'.replace("{thousands}", "")'
        if decimal != ".":
            expression += f'.replace("{decimal}", ".")'
    elif "," in clean:
        parts = clean.split(",")
        if len(parts) == 2 and (is_percent or len(parts[-1]) in {1, 2}):
            expression += '.replace(",", ".")'
        else:
            expression += '.replace(",", "")'
    elif "." in clean:
        parts = clean.split(".")
        if not (is_percent and len(parts) == 2) and all(
            len(part) == 3 for part in parts[1:]
        ):
            expression += '.replace(".", "")'
    sign = "-" if negative else ""
    return (
        f"{sign}float({expression}) * {source_scale:.1f} / {target_scale:.1f}"
    )


def write_source_table_csv(path: Path, grid: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = max((len(row) for row in grid), default=0)
    if width == 0:
        raise ValueError("Cannot export an empty evidence table")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow([f"col_{index}" for index in range(width)])
        for row in grid:
            writer.writerow([*row, *([""] * (width - len(row)))])


def replay_generated_query(query: str, frames: dict[str, object]) -> float:
    cell_match = re.search(r"(df\d+)\.iloc\[(\d+), (\d+)\]", query)
    scale_match = re.search(r" \* ([0-9.]+) / ([0-9.]+)$", query)
    if not cell_match or not scale_match:
        raise ValueError(f"Unsupported generated query: {query}")
    variable, row_text, column_text = cell_match.groups()
    row_index, column_index = int(row_text), int(column_text)
    source_scale, target_scale = map(float, scale_match.groups())
    frame = frames[variable]
    raw = str(frame.iloc[row_index, column_index])
    expected_query = pandas_query_for(
        raw, row_index, column_index, source_scale, target_scale
    )
    if query != expected_query:
        raise ValueError("Query is not the canonical safe numeric-cell expression")
    number = parse_number(raw)
    if number is None:
        raise ValueError(f"Selected cell is not numeric: {raw!r}")
    # The string must equal the canonical expression above before evaluation,
    # so only numeric casts and string replacements produced by this module run.
    return float(
        eval(  # noqa: S307 - guarded by exact canonical-query equality
            query,
            {"__builtins__": {}, "float": float, "str": str},
            frames,
        )
    )


def validate_submission_directory(submission_path: Path, zip_path: Optional[Path] = None) -> dict:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - declared project dependency
        raise RuntimeError("pandas is required to replay submission evidence") from error

    items = json.loads(submission_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("submission.json must contain a JSON array")
    required = {
        "id", "question", "answer", "relevant_docs", "relevant_tables",
        "evidence", "pandas_query",
    }
    errors: list[str] = []
    replayed = 0
    seen_ids: set[int] = set()
    referenced_paths: set[str] = set()
    for item in items:
        question_id = int(item.get("id", -1))
        if set(item) != required:
            errors.append(f"q{question_id}: fields do not match the submission schema")
            continue
        if question_id in seen_ids:
            errors.append(f"q{question_id}: duplicate id")
        seen_ids.add(question_id)
        if not item["relevant_docs"] or not item["relevant_tables"]:
            errors.append(f"q{question_id}: empty retrieval provenance")
        frames = {}
        for evidence in item["evidence"]:
            relative = evidence["csv_path"]
            if not relative.startswith("data/") or ".." in Path(relative).parts:
                errors.append(f"q{question_id}: unsafe evidence path {relative!r}")
                continue
            evidence_path = submission_path.parent / relative
            if not evidence_path.is_file():
                errors.append(f"q{question_id}: missing {relative}")
                continue
            referenced_paths.add(relative)
            frames[evidence["variable"]] = pd.read_csv(evidence_path)
        try:
            actual = replay_generated_query(item["pandas_query"], frames)
            expected = float(item["answer"])
            if not math.isfinite(actual) or not math.isclose(
                actual, expected, rel_tol=1e-12, abs_tol=1e-9
            ):
                errors.append(f"q{question_id}: replay {actual} != answer {expected}")
            else:
                replayed += 1
        except (KeyError, IndexError, TypeError, ValueError) as error:
            errors.append(f"q{question_id}: replay failed: {error}")
    if zip_path is not None:
        with zipfile.ZipFile(zip_path) as archive:
            actual_members = set(archive.namelist())
        expected_members = {"submission.json", *referenced_paths}
        if actual_members != expected_members:
            errors.append("ZIP members differ from submission.json evidence references")
    result = {
        "valid": not errors,
        "items": len(items),
        "unique_ids": len(seen_ids),
        "replayed": replayed,
        "evidence_files": len(referenced_paths),
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(result, ensure_ascii=False))
    return result


def build_submission(
    top_facts: Path,
    output_dir: Path,
    confidence_levels: set[str],
    line_base: int,
    database: Path,
) -> Path:
    with top_facts.open(encoding="utf-8") as handle:
        facts = [json.loads(line) for line in handle if line.strip()]
    selected = [fact for fact in facts if fact["confidence"] in confidence_levels]
    submission = []
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in data_dir.glob("q*_evidence.csv"):
        stale_path.unlink()
    connection = sqlite3.connect(str(database))
    try:
        for fact in selected:
            question_id = int(fact["question_id"])
            relative_csv = f"data/q{question_id:04d}_evidence.csv"
            table = connection.execute(
                "SELECT grid_json FROM tables WHERE table_id = ?", (int(fact["table_id"]),)
            ).fetchone()
            if table is None:
                raise ValueError(f"q{question_id}: table_id {fact['table_id']} not found")
            grid = json.loads(table[0])
            write_source_table_csv(output_dir / relative_csv, grid)
            source_line = int(
                fact["source_line_0"] if line_base == 0 else fact["source_line_1"]
            )
            query = pandas_query_for(
                fact["raw_value"],
                int(fact["row_index"]),
                int(fact["column_index"]),
                float(fact["source_scale"]),
                float(fact["target_scale"]),
            )
            submission.append(
                {
                    "id": question_id,
                    "question": fact["question"],
                    "answer": float(fact["answer"]),
                    "relevant_docs": [fact["document_id"]],
                    "relevant_tables": [f"{fact['document_id']}|{source_line}"],
                    "evidence": [{"variable": "df1", "csv_path": relative_csv}],
                    "pandas_query": query,
                }
            )
    finally:
        connection.close()
    output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = output_dir / "submission.json"
    submission_path.write_text(
        json.dumps(submission, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation = validate_submission_directory(submission_path)
    zip_path = output_dir.with_suffix(".zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, "submission.json")
        for path in sorted(data_dir.glob("*.csv")):
            archive.write(path, path.relative_to(output_dir).as_posix())
    validation = validate_submission_directory(submission_path, zip_path)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "submission": str(submission_path),
                "zip": str(zip_path),
                "items": len(submission),
                "confidence_levels": sorted(confidence_levels),
                "line_base": line_base,
                "validation": validation,
            },
            ensure_ascii=False,
        )
    )
    return zip_path


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    candidate_parser = subparsers.add_parser("candidates")
    candidate_parser.add_argument("--questions", type=Path, default=Path("ViFinQA/questions/questions.jsonl"))
    candidate_parser.add_argument("--companies", type=Path, default=Path("ViFinQA/code_stock.csv"))
    candidate_parser.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    candidate_parser.add_argument("--output-dir", type=Path, default=Path("outputs/l1-facts"))
    candidate_parser.add_argument(
        "--overrides",
        type=Path,
        default=Path("analysis/l1_manual_overrides.csv"),
    )

    submission_parser = subparsers.add_parser("submission")
    submission_parser.add_argument("--top-facts", type=Path, default=Path("outputs/l1-facts/top_facts.jsonl"))
    submission_parser.add_argument("--database", type=Path, default=Path("artifacts/vifinqa.db"))
    submission_parser.add_argument("--output-dir", type=Path, default=Path("outputs/l1-submission"))
    submission_parser.add_argument(
        "--confidence", choices=("high", "medium", "low", "all"), default="all"
    )
    submission_parser.add_argument("--line-base", choices=(0, 1), type=int, default=1)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument(
        "--submission", type=Path, default=Path("outputs/l1-submission/submission.json")
    )
    validate_parser.add_argument("--zip", dest="zip_path", type=Path)

    args = parser.parse_args(argv)
    if args.command == "candidates":
        run_candidates(
            args.questions,
            args.companies,
            args.database,
            args.output_dir,
            args.overrides,
        )
    elif args.command == "submission":
        levels = {"high", "medium", "low"} if args.confidence == "all" else {args.confidence}
        build_submission(
            args.top_facts, args.output_dir, levels, args.line_base, args.database
        )
    elif args.command == "validate":
        print(
            json.dumps(
                validate_submission_directory(args.submission, args.zip_path),
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
