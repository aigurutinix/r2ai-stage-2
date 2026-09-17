"""Extractive metric-phrase extraction.

WHY (measured on submission #6, see P1_STRATEGY_REVIEW.md):
`metric_norm` used to be built SUBTRACTIVELY (question minus stopwords), which
left noise words in the phrase and dragged the label-match score below the rule
threshold even when the correct row was present:

    id=179  metric_norm "so du tra truoc cho nguoi ban tinh dong"
            gold row    "Trả trước cho người bán"            -> score 63 (< 78)

This module instead CUTS the question down to the metric span:
  1. drop the interrogative tail  ("... là bao nhiêu tỷ đồng?")
  2. drop the entity tail         ("... của CTCP ABC (XYZ) năm 2023")
  3. drop leading question verbs  ("Cho biết", "Tính", "Tổng số tiền")
  4. drop leading quantity nouns  ("số dư", "giá trị ghi sổ", "tổng số tiền")
Everything is diacritic-free lowercase (see utils.viet_text.norm).

`extract_metric` returns BOTH the core phrase and a wider phrase; the wider one
keeps qualifiers such as "ngan han"/"dai han" which really do disambiguate rows
("Trả trước cho người bán" vs "Trả trước cho người bán ngắn hạn").
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.viet_text import norm

# --- tail cutters (applied on the normalized string) ---
_TAIL_PATTERNS = [
    r"\bla bao nhieu\b.*$",
    r"\bbang bao nhieu\b.*$",
    r"\blalbao nhieu\b.*$",
    r"\bdat bao nhieu\b.*$",
    r"\bbao nhieu\b.*$",
    r"\bchiem bao nhieu\b.*$",
    r"\?\s*$",
]
_TAIL_RE = re.compile("|".join(_TAIL_PATTERNS))

# entity tail: everything from "cua <company>" / "nam YYYY" / "cuoi nam" onward
_ENTITY_TAIL = re.compile(
    r"\s+(?:cua|tai|trong|vao|den|tinh den|ket thuc)\s+(?:cong ty|ctcp|ngan hang|"
    r"tong cong ty|tap doan|cty|ngan hang tmcp|cong ty me|ma|doanh nghiep)\b.*$"
)
_YEAR_TAIL = re.compile(
    r"\s+(?:trong\s+|vao\s+|cua\s+|tai\s+)?(?:nam|ngay|cuoi nam|dau nam|quy)\s*"
    r"(?:\d{1,2}\s*/\s*\d{1,2}\s*/\s*)?\d{4}\b.*$")
_DATE_TAIL = re.compile(r"\s+\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}\b.*$")
_UNIT_TAIL = re.compile(
    r"\s+(?:tinh\s+)?(?:bang|theo|don vi)\s+(?:dong|vnd|trieu|ty|nghin|phan tram|%)\b.*$")

# a LEADING time clause ("Năm 2015, doanh thu thuần của ...") is not a tail, so
# the tail cutters never removed it and "2015" polluted the metric
_LEAD_TIME = re.compile(
    r"^(?:vao\s+|trong\s+|tai\s+|ket thuc\s+)?"
    r"(?:nam|cuoi nam|dau nam|ngay|quy)\s+[\d/\s]+[,]?\s+")
# operation verbs belong to the PLAN, never to the metric. Leaving "chenh lech"
# in the phrase pulled every `difference` question below the match threshold.
_OP_WORDS = re.compile(
    r"\b(?:chenh lech|tang truong|tang giam|thay doi|bien dong|so voi|"
    r"cao hon|thap hon|nhieu hon|it hon|gap|tang|giam)\b")

# ranking questions enumerate the entities first:
#   "Trong các công ty A (X), B (Y), công ty có doanh thu thuần lớn nhất ..."
# everything up to the pivot "cong ty/doanh nghiep [nao] co" is entity noise.
_LEAD_ENUM = re.compile(
    r"^.*?\b(?:cong ty|doanh nghiep|ma co phieu|ngan hang)\s+(?:nao\s+)?co\s+")
# the superlative belongs to the OPERATION, not to the metric
_SUPERLATIVE = re.compile(
    r"\b(lon nhat|nho nhat|cao nhat|thap nhat|dung dau|dan dau|nhieu nhat|it nhat)\b")

# leading fillers
_LEAD_VERBS = re.compile(
    r"^(?:cho biet|hay cho biet|xac dinh|tinh toan|tinh|hoi|vui long|"
    r"trong cac|trong so cac|trong nhung|"
    r"theo bao cao|theo bctc|dua tren|can cu)\s+")
_LEAD_QUANTITY = re.compile(
    r"^(?:tong so tien|tong gia tri|tong muc|tong cong|tong|so du|so tien|"
    r"gia tri ghi so|gia tri|muc|khoan|so luong|so)\s+(?=\S)")
# qualifiers worth keeping in the wide form but dropping in the core form
_QUALIFIER = re.compile(r"\b(ngan han|dai han|hop nhat|rieng|thuan|rong|gop)\b")

# NOTE: "doanh" must NOT be here — trimming it breaks "doanh thu".
# Multi-word company noise is removed by _STOP_PHRASES instead.
_STOP_LEFTOVER = {
    "cua", "trong", "tai", "vao", "den", "nam", "ngay", "cuoi", "dau", "la",
    "bao", "nhieu", "va", "the", "nao", "gi", "cho", "biet", "tinh", "bang",
    "theo", "don", "vi", "dong", "vnd", "cong", "ty", "co", "phan", "ctcp",
    "tmcp", "tap", "doan", "me", "hop", "nhat", "rieng", "le", "bctc", "ma",
}
_STOP_PHRASES = re.compile(
    r"\b(?:doanh nghiep|cong ty me|cong ty co phan|ngan hang tmcp|"
    r"ngan hang thuong mai co phan|tong cong ty|tap doan)\b")
_COUNT_CONDITION_CUE = re.compile(
    r"\b(?:dong thoi|vua(?:\s+co)?|ghi nhan|phat sinh|"
    r"co\s+(?=(?:gia tri|dong|luu|lai|he|so\s+(?:du|luong)|"
    r"chi|doanh|tai|no|von|thue|cam|chenh|ty|bien|tong|khoan))|"
    r"dat|vuot|tang|giam)\b"
)
_COUNT_THRESHOLD_TAIL = re.compile(
    r"\b(?:lon hon|nho hon|duoi|vuot|cao hon|thap hon|it hon|nhieu hon|"
    r"toi thieu|toi da|khong qua|tren muc|duoi muc|tren)\s+"
    r"[-+]?\d[\d\s.,]*(?:ty|trieu|nghin|ngan|dong|vnd|lan|vong|phan tram|"
    r"co phieu|co phan|%)?.*$"
)
_COUNT_NOISE = re.compile(
    r"\b(?:dong thoi|vua|ghi nhan|phat sinh|dat|muc|gia tri|so du|"
    r"cuoi ky|cuoi nam|dau nam|trong nam|vao cuoi nam|tai cuoi nam|"
    r"duong|am|cai thien|tang truong|toc do|tang|giam|lon hon|nho hon|duoi|vuot|cao hon|"
    r"thap hon|so voi|hon|trung vi|ca giai doan|sau nam dau tien|"
    r"trong so|cac|nhom|thuoc nhom|nganh)\b"
)
_COUNT_SPLIT = re.compile(r"\b(?:dong thoi|vua|va|hoac)\b|[,;]")
_NUMERIC_AND_UNITS = re.compile(
    r"\b[-+]?\d[\d\s.,]*(?:ty|trieu|nghin|ngan|dong|vnd|lan|vong|phan tram|"
    r"co phieu|co phan|%)?\b"
)


@dataclass
class MetricPhrase:
    core: str          # tightest span, best for fuzzy matching row labels
    wide: str          # core + qualifiers, disambiguates near-duplicate rows
    raw: str           # normalized question with only the tail removed

    def variants(self) -> list[str]:
        seen, out = set(), []
        for v in (self.core, self.wide, self.raw):
            v = v.strip()
            if v and v not in seen:
                seen.add(v)
                out.append(v)
        return out


def extract_metric(question: str, company_aliases: list[str] | None = None,
                   tickers: list[str] | None = None) -> MetricPhrase:
    """Cut a Vietnamese financial question down to its metric phrase."""
    q = norm(question)

    # remove company names / tickers first: they are the noisiest part
    for alias in sorted(company_aliases or [], key=len, reverse=True):
        if len(alias) >= 5:
            q = q.replace(alias, " ")
    for t in tickers or []:
        q = re.sub(rf"(?<![0-9a-z]){t.lower()}(?![0-9a-z])", " ", q)

    q = _TAIL_RE.sub(" ", q)
    q = _STOP_PHRASES.sub(" ", q)
    raw = _squeeze(q)

    cut = _squeeze(_LEAD_TIME.sub("", raw))
    for rex in (_UNIT_TAIL, _ENTITY_TAIL, _YEAR_TAIL, _DATE_TAIL):
        cut = rex.sub(" ", cut)
    cut = _squeeze(_OP_WORDS.sub(" ", cut))

    if _LEAD_ENUM.match(cut):
        cut = _squeeze(_LEAD_ENUM.sub("", cut))
    cut = _squeeze(_SUPERLATIVE.sub(" ", cut))

    prev = None
    while prev != cut:
        prev = cut
        cut = _squeeze(_LEAD_VERBS.sub("", cut))
    wide = cut

    core = _squeeze(_LEAD_QUANTITY.sub("", wide))
    if len(core.split()) < 2:      # over-trimmed -> keep the wider span
        core = wide

    # strip dangling stopwords at both ends
    core = _trim_stopwords(core)
    wide = _trim_stopwords(wide)
    if not core:
        core = wide or raw
    return MetricPhrase(core=core, wide=wide, raw=_trim_stopwords(raw))


def extract_count_metrics(question: str, company_aliases: list[str] | None = None,
                          tickers: list[str] | None = None) -> list[str]:
    """Return condition metrics for count questions.

    Generic extraction cuts everything after the first ``bao nhieu``.  That is
    correct for "doanh thu la bao nhieu", but wrong for population-count
    questions: in "co bao nhieu cong ty ... co dong tien thuan duong", the row
    to retrieve is the condition after the population, not the phrase
    "co bao nhieu".
    """
    q = norm(question)
    for alias in sorted(company_aliases or [], key=len, reverse=True):
        if len(alias) >= 5:
            q = q.replace(alias, " ")
    for t in tickers or []:
        q = re.sub(rf"(?<![0-9a-z]){t.lower()}(?![0-9a-z])", " ", q)
    q = _squeeze(_STOP_PHRASES.sub(" ", q))

    if "bao nhieu" in q:
        idx = q.find("bao nhieu")
        # Leading "co bao nhieu ..." asks for a population count; the metric is
        # after that phrase.  Late "... trong bao nhieu nam" asks about a metric
        # already stated before it.
        before = q[:idx]
        after = q[idx + len("bao nhieu"):]
        asks_population = (
            idx < len(q) * 0.35
            or re.search(r"\bco\s*$", before) is not None
            or re.match(r"\s*(?:trong\s+so\s+)?(?:cac\s+)?"
                        r"(?:cong ty|doanh nghiep|ngan hang|nam|ma|don vi)\b",
                        after) is not None
        )
        seed = after if asks_population else before
    else:
        seed = q
    seed = _squeeze(seed)

    match = _COUNT_CONDITION_CUE.search(seed)
    segment = seed[match.start():] if match else seed
    segment = _squeeze(_LEAD_TIME.sub("", segment))

    raw_variants = [segment]
    raw_variants.extend(_COUNT_SPLIT.split(segment))
    if " tren " in segment:
        raw_variants.extend(re.split(r"\btren\b", segment))

    out, seen = [], set()
    for raw in raw_variants:
        cleaned = _clean_count_metric(raw)
        if not cleaned:
            continue
        toks = cleaned.split()
        if len(toks) < 2:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def _clean_count_metric(text: str) -> str:
    s = _squeeze(text)
    for rex in (_UNIT_TAIL, _YEAR_TAIL, _DATE_TAIL):
        s = rex.sub(" ", s)
    s = _COUNT_THRESHOLD_TAIL.sub(" ", s)
    s = _NUMERIC_AND_UNITS.sub(" ", s)
    s = re.sub(r"^co\s+(?=\S)", " ", s)
    s = _COUNT_NOISE.sub(" ", s)
    s = _squeeze(s)
    s = re.sub(r"^co\s+(?=\S)", " ", s)
    s = _squeeze(s)
    # Keep "so luong co phieu" intact; it is a real report row label, unlike
    # the leading "so du/gia tri" wrappers handled by extract_metric().
    return _trim_count_stopwords(s)


def _trim_count_stopwords(s: str) -> str:
    stop = set(_STOP_LEFTOVER) - {"co", "dong"}
    toks = s.split()
    while toks and toks[0] in stop:
        toks.pop(0)
    while toks and toks[-1] in stop:
        toks.pop()
    return " ".join(toks)


def _squeeze(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _trim_stopwords(s: str) -> str:
    toks = s.split()
    while toks and toks[0] in _STOP_LEFTOVER:
        toks.pop(0)
    while toks and toks[-1] in _STOP_LEFTOVER:
        toks.pop()
    return " ".join(toks)


def has_qualifier(phrase: str) -> bool:
    return bool(_QUALIFIER.search(phrase))
