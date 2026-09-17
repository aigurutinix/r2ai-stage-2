"""BM25 retrieval trên catalog 146k bảng. Xương sống truy hồi (đặt sàn Tables F2).

Bê `bm25s` (thư viện sẵn). Tokenizer VN đơn giản: NFC + lowercase + tách token,
GIỮ NGUYÊN mã số/hậu tố (vd '421a' không bị xé). underthesea để nâng cấp sau.

Chạy:
  python src/kingpro/retrieval/bm25_index.py build
  python src/kingpro/retrieval/bm25_index.py query "Doanh thu thuần của VNM năm 2023 là bao nhiêu?"
"""

from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import bm25s

CATALOG = Path("build/catalog.jsonl")
INDEX_DIR = Path("build/bm25")

_TOKEN_RE = re.compile(r"[0-9a-zà-ỹ]+", re.IGNORECASE)

# Chuẩn hoá vị trí dấu thanh oa/oe/uy (chính tả cũ<->mới) — đòn +0.083 recall (hung&phong).
# Đưa về MỘT dạng, áp cho CẢ query lẫn corpus (đối xứng).
_TONE_PAIRS = {
    "òa": "oà", "óa": "oá", "ỏa": "oả", "õa": "oã", "ọa": "oạ",
    "òe": "oè", "óe": "oé", "ỏe": "oẻ", "õe": "oẽ", "ọe": "oẹ",
    "ùy": "uỳ", "úy": "uý", "ủy": "uỷ", "ũy": "uỹ", "ụy": "uỵ",
}
_TONE_RE = re.compile("|".join(_TONE_PAIRS))


def canon_tones(text: str) -> str:
    return _TONE_RE.sub(lambda m: _TONE_PAIRS[m.group()], text)


def tokenize(text: str) -> list[str]:
    text = canon_tones(unicodedata.normalize("NFC", str(text)).lower())
    return _TOKEN_RE.findall(text)


def fold(s: str) -> str:
    """Bỏ dấu + đ->d + lowercase (để so tên công ty không lệ thuộc dấu)."""
    s = unicodedata.normalize("NFD", str(s).lower()).replace("đ", "d")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


_SEPARATE_SCOPE_RE = re.compile(
    r"\b(?:cong ty|ngan hang)\s+me\b|"
    r"\bbao cao(?: tai chinh)?\s+rieng\b|"
    r"\bbctc\s+rieng\b|"
    r"\brieng le\b",
    re.IGNORECASE,
)
_CONSOLIDATED_SCOPE_RE = re.compile(r"\bhop\s+nhat\b", re.IGNORECASE)
_OWNERSHIP_DISCLOSURE_RE = re.compile(r"\bty\s+le\s+so\s+huu\b", re.IGNORECASE)
_MULTI_ENTITY_COUNT_RE = re.compile(r"\bco\s+bao\s+nhieu\b", re.IGNORECASE)
_CATALOG_UNIVERSE_RE = re.compile(
    r"\btrong\s+(?:so\s+)?cac\s+(?:cong\s+ty|doanh\s+nghiep)\s+co\b",
    re.IGNORECASE,
)
_MULTI_STAGE_SERIES_RE = re.compile(
    r"\btai\s+(?:cuoi\s+)?nam\s+co\b", re.IGNORECASE
)


def infer_scope(question: str) -> str:
    """Map explicit separate-report phrases to the catalog scope.

    A bare ``riêng`` is deliberately insufficient because it often means
    "consider only" rather than a separate financial statement.
    """
    return "công ty mẹ" if _SEPARATE_SCOPE_RE.search(fold(question)) else "hợp nhất"


def has_explicit_scope_cue(question: str) -> bool:
    """Return whether the question itself explicitly requests a report scope."""
    question_folded = fold(question)
    return bool(
        _SEPARATE_SCOPE_RE.search(question_folded)
        or _CONSOLIDATED_SCOPE_RE.search(question_folded)
    )


def is_competitive_implicit_scope(
    question: str,
    preferred_score: float,
    opposite_score: float,
    *,
    minimum_ratio: float,
) -> bool:
    """Gate retention of an opposite-scope filing by query evidence.

    The default scope remains selected.  This function only allows an
    additional filing when the question did not specify scope and the opposite
    filing has materially stronger BM25 evidence.
    """
    return bool(
        minimum_ratio > 0
        and not has_explicit_scope_cue(question)
        and preferred_score > 0
        and opposite_score >= preferred_score * minimum_ratio
    )


def is_competitive_ownership_scope(
    question: str,
    facets: dict,
    preferred_score: float,
    opposite_score: float,
    *,
    minimum_ratio: float,
) -> bool:
    """Gate dual-scope evidence for an implicit ownership disclosure lookup."""
    return bool(
        len(facets.get("tickers", [])) == 1
        and not bool(facets.get("analytic"))
        and _OWNERSHIP_DISCLOSURE_RE.search(fold(question))
        and is_competitive_implicit_scope(
            question,
            preferred_score,
            opposite_score,
            minimum_ratio=minimum_ratio,
        )
    )


def needs_count_scope_fallback(question: str, facets: dict) -> bool:
    """Detect a one-period multi-entity count needing evidence for every entity."""
    return bool(
        not has_explicit_scope_cue(question)
        and len(facets.get("tickers", [])) >= 2
        and len(facets.get("years", [])) == 1
        and _MULTI_ENTITY_COUNT_RE.search(fold(question))
    )


def needs_catalog_universe_scan(question: str, facets: dict) -> bool:
    """Detect an explicitly broad cross-company scan with no named entities."""
    return bool(
        not facets.get("tickers")
        and facets.get("years")
        and bool(facets.get("analytic"))
        and _CATALOG_UNIVERSE_RE.search(fold(question))
    )


_ORG_FORMS = [
    (re.compile(r"\bcong ty co phan\b"), "ctcp"),
    (re.compile(r"\bcong ty cp\b"), "ctcp"),          # "Công ty CP ..." (viết tắt CP)
    (re.compile(r"\btong ct co phan\b"), "tctcp"),
    (re.compile(r"\bcong ty tnhh\b"), "ctnhh"),
    (re.compile(r"\btong cong ty\b"), "tct"),
    (re.compile(r"\bcong ty\b"), "ct"),
]


def canon_org(s: str) -> str:
    """Gộp biến thể dạng pháp nhân (câu hỏi ghi 'công ty cổ phần', code_stock ghi 'ctcp')
    về MỘT token, để so tên công ty không lệch vì cách viết. s đã fold."""
    for pat, rep in _ORG_FORMS:
        s = pat.sub(rep, s)
    return re.sub(r"\s+", " ", s).strip()


_NAME2TICKER: dict[str, str] | None = None
_TICKERS: set[str] | None = None


def load_name_to_ticker() -> dict[str, str]:
    """{tên công ty đã fold -> mã CK}, sắp để so khớp tên dài trước."""
    global _NAME2TICKER
    if _NAME2TICKER is None:
        import csv

        m: dict[str, str] = {}
        cs = Path("data/code_stock.csv")
        if cs.exists():
            with open(cs, encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    ma = (row.get("Mã CK") or "").strip()
                    ten = (row.get("Tên công ty") or "").strip()
                    if ma and ten:
                        m[canon_org(fold(ten))] = ma
        _NAME2TICKER = m
    return _NAME2TICKER


def load_tickers() -> set[str]:
    """Tập mã CK hợp lệ (để validate mã trần in hoa trong câu, tránh dính từ hoa lung tung)."""
    global _TICKERS
    if _TICKERS is None:
        _TICKERS = set(load_name_to_ticker().values())
    return _TICKERS


# Token dạng pháp nhân để lột ra LÕI tên (phần đặc thù) — cắt ở hai đầu.
_LEGAL_TOK = {"ctcp", "tct", "tong", "cong", "ty", "co", "phan", "tap", "doan",
              "ngan", "hang", "tmcp", "tnhh", "mtv", "nha", "nuoc"}
# Từ tiếng Việt phổ biến — KHÔNG làm alias 1-token dù hiếm trong tên công ty (dính bừa).
_GENERIC_WORD = {"khong", "luong", "ngoai", "phuong", "nhiet", "duong", "thanh", "hoa",
                 "phu", "tin", "minh", "dong", "cao", "dau", "dien", "xuat", "thuong",
                 "quoc", "viet", "trung", "dai", "tien", "hung", "cuong", "thang", "song"}
# Alias thương hiệu KHÔNG nằm trong tên chính thức (phải bổ sung tay). fold-canon -> mã.
_BRAND_ALIAS = {
    "vinamilk": "VNM", "dam ca mau": "DCM", "dam phu my": "DPM", "ocean group": "OGC",
    "petrolimex": "PLX", "sabeco": "SAB", "habeco": "BHN", "vietjet": "VJC",
    "techcombank": "TCB", "vietcombank": "VCB", "vietinbank": "CTG", "sacombank": "STB",
    "vpbank": "VPB", "mbbank": "MBB", "acb": "ACB", "bidv": "BID", "hdbank": "HDB",
    "vinhomes": "VHM", "novaland": "NVL", "the gioi di dong": "MWG", "fpt retail": "FRT",
    "hoang anh gia lai": "HAG", "pv gas": "GAS", "pvgas": "GAS", "hoa phat": "HPG",
    "binh son": "BSR", "pvtrans": "PVT", "nam kim": "NKG",
    "do thi kinh bac": "KBC", "kinh bac": "KBC",
    "minh phu": "MPC", "thuy san minh phu": "MPC",
    "vicem ha tien": "HT1", "van phu invest": "VPI", "van phu": "VPI",
    "hai phat dau tu": "HPX", "hai phat": "HPX",
    "dau khi ca mau": "DCM", "phan bon dau khi ca mau": "DCM",
    "dien luc gelex": "GEE", "tap doan gelex": "GEX", "deo ca": "HHV",
    "song da": "SJG", "viglacera": "VGC", "dai duong": "OGC",
    "sao mai": "ASM", "duong quang ngai": "QNS", "dabaco": "DBC",
    "dat xanh": "DXG",
    "ngan hang tmcp sai gon thuong tin": "STB",
    "bia ruou nuoc giai khat sai gon": "SAB",
    "san xuat kinh doanh xuat nhap khau dich vu va dau tu tan binh": "PRT",
    "saigonbank": "SGB", "nong nghiep quoc te hagl": "HNG",
    "eximbank": "EIB", "dich vu hoang huy": "HHS", "tkv": "DTK",
}

_ALIASES: dict[str, str] | None = None


def load_aliases() -> dict[str, str]:
    """{chuỗi alias fold-canon -> mã}: LÕI tên tự động (bỏ token pháp nhân) + brand tay.

    Bắt câu ghi tên NGẮN/thương hiệu ('Hoà Phát', 'Masan', 'Vinamilk', 'Phân bón Dầu khí Cà Mau').
    """
    global _ALIASES
    if _ALIASES is None:
        import collections
        cores: dict[str, list[str]] = {}
        for name_f, ma in load_name_to_ticker().items():
            toks = name_f.replace("-", " ").split()
            i, j = 0, len(toks)
            while i < j and toks[i] in _LEGAL_TOK:
                i += 1
            while j > i and toks[j - 1] in _LEGAL_TOK:
                j -= 1
            if i < j:
                cores[ma] = toks[i:j]
        # tần suất token qua các công ty -> token generic (nhiều công ty) KHÔNG làm alias 1-từ
        freq = collections.Counter(t for core in cores.values() for t in set(core))
        al: dict[str, str] = {}
        for ma, core in cores.items():
            full = " ".join(core)
            if len(full) >= 5:
                al.setdefault(full, ma)                    # lõi đầy đủ (đa từ luôn an toàn)
            if len(core[0]) >= 5 and freq[core[0]] <= 2 and core[0] not in _GENERIC_WORD:
                al.setdefault(core[0], ma)                 # token đầu ĐẶC THÙ (Masan/Vingroup/Dabaco)
        al.update(_BRAND_ALIAS)                            # brand tay ghi đè
        _ALIASES = al
    return _ALIASES


def load_catalog() -> list[dict]:
    rows = []
    with open(CATALOG, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def extract_facets(question: str) -> dict:
    """Bóc mã CK, năm, phạm vi từ câu hỏi để lọc-trước."""
    q = question
    year = None
    ys = re.findall(r"\b(20\d{2})\b", q)
    if ys:
        year = ys[-1]  # năm cuối cùng nhắc tới thường là năm hỏi
    # mã CK theo THỨ TỰ ưu tiên độ-cụ-thể (tránh mã trần chộp nhầm mã lồng trong tên dài:
    # "CTCP Chứng khoán FPT" là FTS, KHÔNG phải FPT):
    #   (1) mã trong ngoặc (VNM) — hiển ngôn;
    #   (2) khớp TÊN công ty dài nhất (đã fold+chuẩn hoá pháp nhân) — cụ thể hơn mã trần;
    #   (3) mã TRẦN in hoa validate với danh sách mã thật (SCR, VSC, HT1...) — fallback.
    ticker = None
    m = re.search(r"\(([A-Z]{3})\)", q)
    if m:
        ticker = m.group(1)
    if ticker is None:
        qf = canon_org(fold(q))
        best = ""
        for name_f, ma in load_name_to_ticker().items():
            if name_f and name_f in qf and len(name_f) > len(best):
                best, ticker = name_f, ma
    if ticker is None:
        valid = load_tickers()
        # token in hoa 2-4 ký tự (có thể kèm 1 chữ số cuối như HT1), lấy cái KHỚP mã thật
        for tok in re.findall(r"\b([A-Z]{2,4}\d?|[A-Z]{2,3}\d)\b", q):
            if tok in valid:
                ticker = tok
                break
    scope = infer_scope(q)
    return {"ticker": ticker, "year": year, "scope": scope}


_RANGE_RE = re.compile(
    r"(20\d{2})\s*(?:-|–|—|đến|tới|sang)\s*(?:năm\s*)?(20\d{2})",
    re.IGNORECASE,
)
_FOLDED_RANGE_RE = re.compile(
    r"(20\d{2})\s*(?:-|–|—|den|toi|sang)\s*(?:nam\s*)?(20\d{2})",
    re.IGNORECASE,
)
_SCOPE_CUE_RE = re.compile(
    r"\bbao cao(?: tai chinh)?\s+(?:cong ty me|rieng|hop nhat)\b|"
    r"\bbctc\s+(?:rieng|hop nhat)\b",
    re.IGNORECASE,
)
_ANALYTIC_RE = re.compile(r"giai đoạn|cao nhất|thấp nhất|lớn nhất|nhỏ nhất|so với|công ty nào|"
                          r"doanh nghiệp nào|trong nhóm|trong số|trung bình|mỗi năm|từng năm|"
                          r"tăng trưởng|xếp hạng|đứng đầu", re.IGNORECASE)
_OPENING_PERIOD_RE = re.compile(r"\bdau\s+(?:nam|ky)\b", re.IGNORECASE)
_OPENING_DATE_RE = re.compile(
    r"\b01\s*[/.-]\s*01(?:\s*[/.-]\s*\d{2,4})?\b", re.IGNORECASE
)
_AVERAGE_BALANCE_RE = re.compile(r"\b(?:binh quan|trung binh)\b", re.IGNORECASE)
_OPEN_TO_CLOSE_RE = re.compile(
    r"\btu\s+dau\s+nam\b.*\b(?:den|toi)\b.*\bcuoi\s+nam\b", re.IGNORECASE
)
_PRIOR_BALANCE_METRIC_RE = re.compile(
    r"\b(?:roe|roa)\b|\bty\s+suat\s+sinh\s+loi\s+tren\s+von\s+chu\s+so\s+huu\b|"
    r"\bvong\s+quay\s+(?:tong\s+tai\s+san|tai\s+san\s+co\s+dinh|von\s+chu\s+so\s+huu)\b|"
    r"\bso\s+ngay\s+ton\s+kho\b|"
    r"\b(?:tong\s+tai\s+san|tai\s+san\s+co\s+dinh|von\s+chu\s+so\s+huu|hang\s+ton\s+kho)\s+binh\s+quan\b|"
    r"\bbinh\s+quan\s+(?:tong\s+tai\s+san|tai\s+san\s+co\s+dinh|von\s+chu\s+so\s+huu|hang\s+ton\s+kho)\b",
    re.IGNORECASE,
)
_EXPLICIT_BALANCE_PAIR_RE = re.compile(
    r"(?:tong\s+tai\s+san|tai\s+san\s+co\s+dinh|von\s+chu\s+so\s+huu|hang\s+ton\s+kho)"
    r"\s+binh\s+quan(?:\s+cuoi)?\s+nam\s+20\d{2}"
    r"\s+va(?:\s+cuoi)?(?:\s+nam)?\s+20\d{2}",
    re.IGNORECASE,
)
_AVERAGE_ACROSS_YEARS_RE = re.compile(
    r"\bty\s+le\b.*\bbinh\s+quan\b.*\bqua\s+cac\s+nam\b", re.IGNORECASE
)
_REVENUE_GROWTH_SCAN_RE = re.compile(
    r"\b(?:tang\s+truong\s+doanh\s+thu|toc\s+do\s+tang\s+doanh\s+thu|"
    r"muc\s+tang\s+tuong\s+doi\s+cua\s+doanh\s+thu|doanh\s+thu.*sut\s+giam)\b",
    re.IGNORECASE,
)
_EXTREME_RE = re.compile(
    r"\b(?:cao\s+nhat|thap\s+nhat|lon\s+nhat|sau\s+nhat|manh\s+nhat)\b",
    re.IGNORECASE,
)
_COMPARATIVE_SELECTION_RE = re.compile(
    r"\b(?:tang\s+truong|toc\s+do\s+tang|ty\s+le\s+tang|"
    r"muc\s+thay\s+doi|sut\s+giam)\b|"
    r"\b(?:doanh\s+thu|bien\s+loi\s+nhuan)\b.{0,100}"
    r"\b(?:tang|giam)\b.{0,80}\b(?:so\s+voi|ky\s+so\s+sanh)\b",
    re.IGNORECASE,
)


def explicit_opening_previous_years(question: str, years: list[str]) -> list[str]:
    """Return at most one prior report year for an explicit opening balance.

    Vietnamese financial statements normally expose the opening balance of
    the first requested period as the prior report's closing balance.  We add
    only ``min(years) - 1`` and only for explicit opening/date language.  This
    bounded rule avoids the broad implicit-year expansion that reduced source
    precision in the full regression.
    """
    question_folded = fold(question)
    opening_mentions = _OPENING_PERIOD_RE.findall(question_folded)
    needs_prior_source = bool(_OPENING_DATE_RE.search(question_folded)) or (
        bool(opening_mentions)
        and (
            bool(_AVERAGE_BALANCE_RE.search(question_folded))
            or bool(_OPEN_TO_CLOSE_RE.search(question_folded))
            or len(opening_mentions) >= 2
        )
    )
    if not years or not needs_prior_source:
        return []
    mentioned_opening_years = [
        int(year)
        for year in re.findall(r"\bdau\s+nam\s+(20\d{2})\b", question_folded)
    ]
    if len(mentioned_opening_years) >= 2:
        return sorted({str(year - 1) for year in mentioned_opening_years if year > 2000})
    numeric_years = [int(year) for year in years if re.fullmatch(r"20\d{2}", year)]
    if not numeric_years:
        return []
    previous = min(numeric_years) - 1
    return [str(previous)] if previous >= 2000 else []


def financial_formula_previous_years(question: str, years: list[str]) -> list[str]:
    """Return one prior year for metrics defined on an average balance.

    This is an accounting dependency, not a generic analytic expansion: ROA,
    ROE, selected turnover metrics and inventory days need the opening balance
    of the earliest evaluated year.  Other ratios remain untouched.
    """
    question_folded = fold(question)
    if (
        not years
        or not _PRIOR_BALANCE_METRIC_RE.search(question_folded)
        or _EXPLICIT_BALANCE_PAIR_RE.search(question_folded)
        or _AVERAGE_ACROSS_YEARS_RE.search(question_folded)
    ):
        return []
    numeric_years = [int(year) for year in years if re.fullmatch(r"20\d{2}", year)]
    if not numeric_years:
        return []
    previous = min(numeric_years) - 1
    return [str(previous)] if previous >= 2000 else []


def growth_scan_previous_years(question: str, years: list[str]) -> list[str]:
    """Add the baseline year for an extreme revenue-growth scan over a period."""
    question_folded = fold(question)
    if (
        not years
        or "giai doan" not in question_folded
        or not _REVENUE_GROWTH_SCAN_RE.search(question_folded)
        or not _EXTREME_RE.search(question_folded)
    ):
        return []
    numeric_years = [int(year) for year in years if re.fullmatch(r"20\d{2}", year)]
    if not numeric_years:
        return []
    previous = min(numeric_years) - 1
    return [str(previous)] if previous >= 2000 else []


def comparative_selection_previous_years(
    question: str,
    years: list[str],
    *,
    entity_count: int,
) -> list[str]:
    """Add one baseline year for a cross-entity comparative selection.

    A multi-company filter/ranking that names only the evaluation year but
    uses growth or a change metric needs the immediately preceding report.
    When two or more years are already explicit, they are the comparison
    inputs and no further year is inferred.  This keeps ordinary cross-company
    differences and complete two-year comparisons untouched.
    """
    if entity_count < 2 or len(years) != 1:
        return []
    question_folded = fold(question)
    if not _COMPARATIVE_SELECTION_RE.search(question_folded):
        return []
    numeric_years = [int(year) for year in years if re.fullmatch(r"20\d{2}", year)]
    if not numeric_years:
        return []
    previous = min(numeric_years) - 1
    return [str(previous)] if previous >= 2000 else []


def comparative_series_previous_years(
    question: str,
    years: list[str],
    *,
    entity_count: int,
) -> list[str]:
    """Add one source-report dependency for multi-entity comparative series.

    A question may explicitly name two or more observation years while the
    executable source lineage also needs the report immediately before the
    earliest year.  This occurs when the earliest observation is read from a
    comparative/opening column before entities are filtered or ranked.  The
    rule is deliberately narrower than generic implicit-year expansion: it
    requires multiple named entities, multiple explicit years and comparative
    selection language, and adds exactly one year.
    """
    if entity_count < 2 or len(years) < 2:
        return []
    if not _COMPARATIVE_SELECTION_RE.search(fold(question)):
        return []
    numeric_years = [int(year) for year in years if re.fullmatch(r"20\d{2}", year)]
    if not numeric_years:
        return []
    previous = min(numeric_years) - 1
    return [str(previous)] if previous >= 2000 else []


def sparse_series_predecessor_years(
    years: list[str],
    *,
    entity_count: int,
    analytic: bool,
) -> list[str]:
    """Fill only internal one-year predecessors in a sparse single-entity series.

    Later financial statements frequently expose the preceding observation in
    a comparative column.  For a sparse list of at least three observation
    years, return missing ``year - 1`` dependencies that remain inside the
    stated series span.  The year before the earliest observation is excluded.
    """
    if entity_count != 1 or not analytic or len(years) < 3:
        return []
    numeric = sorted(
        {int(year) for year in years if re.fullmatch(r"20\d{2}", str(year))}
    )
    if len(numeric) < 3:
        return []
    present = set(numeric)
    earliest = numeric[0]
    return [
        str(year - 1)
        for year in numeric[1:]
        if year - 1 >= earliest and year - 1 not in present
    ]


def expand_retrieval_years(
    question: str,
    years: list[str],
    *,
    opening: bool = False,
    formula: bool = False,
    growth_scan: bool = False,
    comparative_selection_entities: int = 0,
    comparative_series_entities: int = 0,
    sparse_series_entities: int = 0,
    sparse_series_analytic: bool = False,
) -> list[str]:
    """Union independent semantic expansions derived from the original years."""
    base_years = list(years)
    expanded = set(base_years)
    if opening:
        expanded.update(explicit_opening_previous_years(question, base_years))
    if formula:
        expanded.update(financial_formula_previous_years(question, base_years))
    if growth_scan:
        expanded.update(growth_scan_previous_years(question, base_years))
    if comparative_selection_entities:
        expanded.update(
            comparative_selection_previous_years(
                question,
                base_years,
                entity_count=comparative_selection_entities,
            )
        )
    if comparative_series_entities:
        expanded.update(
            comparative_series_previous_years(
                question,
                base_years,
                entity_count=comparative_series_entities,
            )
        )
    if sparse_series_entities:
        expanded.update(
            sparse_series_predecessor_years(
                base_years,
                entity_count=sparse_series_entities,
                analytic=sparse_series_analytic,
            )
        )
    return sorted(expanded)


def extract_year_scope_overrides(question: str) -> dict[str, str]:
    """Extract per-year scope only when both scopes are explicitly segmented.

    Example: ``báo cáo công ty mẹ các năm 2022, 2023 và báo cáo hợp nhất
    các năm 2024, 2025``.  A single cue continues to use ``infer_scope``;
    ambiguous text therefore cannot create a mixed-scope guess.
    """
    question_folded = fold(question)
    cues = list(_SCOPE_CUE_RE.finditer(question_folded))
    cue_scopes = ["hợp nhất" if "hop nhat" in cue.group() else "công ty mẹ" for cue in cues]
    if len(cues) < 2 or len(set(cue_scopes)) < 2:
        return {}
    overrides: dict[str, str] = {}
    for index, cue in enumerate(cues):
        end = cues[index + 1].start() if index + 1 < len(cues) else len(question_folded)
        segment = question_folded[cue.end():end]
        years = set(re.findall(r"\b(20\d{2})\b", segment))
        for first, last in _FOLDED_RANGE_RE.findall(segment):
            lower, upper = sorted((int(first), int(last)))
            if upper - lower <= 12:
                years.update(str(year) for year in range(lower, upper + 1))
        for year in years:
            overrides[year] = cue_scopes[index]
    return overrides


def extract_all_facets(question: str) -> dict:
    """Bóc TẤT CẢ mã CK + TẤT CẢ năm (giãn 'giai đoạn 2016-2020' -> [2016..2020]) cho câu phân tích.

    Nguồn ý tưởng: Question Decomposition for RAG (arXiv 2507.00355) — tách câu đa-thực-thể
    thành nhiều truy vấn con (mã × năm) rồi union. Router phân nhánh theo Adaptive-RAG (NAACL 2024).
    """
    q = question
    valid = load_tickers()
    tickers: list[str] = []
    for mm in re.findall(r"\(([A-Z]{2,4}\d?)\)", q):          # (1) trong ngoặc — hiển ngôn
        if mm in valid and mm not in tickers:
            tickers.append(mm)
    qf = canon_org(fold(q))                                    # (2) khớp TÊN công ty (cụ thể hơn mã trần)
    matched_names: list[str] = []
    for name_f, ma in sorted(load_name_to_ticker().items(), key=lambda x: -len(x[0])):
        if name_f and name_f in qf:
            matched_names.append(name_f)
            if ma not in tickers:
                tickers.append(ma)
    for tok in re.findall(r"\b([A-Z]{2,4}\d?)\b", q):         # (3) mã trần HOA — BỎ nếu nằm trong tên đã khớp
        if tok in valid and tok not in tickers and not any(fold(tok) in nm for nm in matched_names):
            tickers.append(tok)
    # (4) mã trần THƯỜNG trong danh sách ("hpx,kbc,nvl,vic") — validate mã thật
    for tok in re.findall(r"\b([a-z]{2,4}\d?)\b", q):
        up = tok.upper()
        if up in valid and up not in tickers:
            tickers.append(up)
    # (5) alias lõi tên / thương hiệu (Hoà Phát, Masan, Vinamilk, Phân bón Dầu khí Cà Mau)
    # Company lists often use en/em dashes around legal-name fragments.  Alias
    # keys are lexical phrases, so collapse punctuation only for this matching
    # pass; raw ticker and full legal-name extraction above remain unchanged.
    alias_qf = re.sub(r"[^a-z0-9]+", " ", qf).strip()
    occupied_name_spans: list[tuple[int, int]] = []
    for matched_name in matched_names:
        matched_key = re.sub(r"[^a-z0-9]+", " ", matched_name).strip()
        if not matched_key:
            continue
        occupied_name_spans.extend(
            (match.start(), match.end())
            for match in re.finditer(re.escape(matched_key), alias_qf)
        )
    for alias, ma in sorted(load_aliases().items(), key=lambda x: -len(x[0])):
        alias_key = re.sub(r"[^a-z0-9]+", " ", alias).strip()
        alias_matches = list(
            re.finditer(rf"(?<![a-z]){re.escape(alias_key)}(?![a-z])", alias_qf)
        )
        has_unshadowed_match = any(
            not any(start <= match.start() and match.end() <= end for start, end in occupied_name_spans)
            for match in alias_matches
        )
        if ma not in tickers and has_unshadowed_match:
            tickers.append(ma)
    # năm: từng năm + giãn khoảng
    explicit_years = set(re.findall(r"\b(20\d{2})\b", q))
    years = set(explicit_years)
    for a, b in _RANGE_RE.findall(q):
        lo, hi = sorted((int(a), int(b)))
        if hi - lo <= 12:
            years |= {str(y) for y in range(lo, hi + 1)}
    scope = infer_scope(q)
    is_analytic = len(tickers) >= 2 or bool(_RANGE_RE.search(q)) or bool(_ANALYTIC_RE.search(q))
    return {"tickers": tickers, "years": sorted(years), "scope": scope, "analytic": is_analytic}


def _reports_for_facet(meta: list[dict], ticker: str, year: str, scope: str) -> list[dict]:
    """Return one catalog representative for every physical report in a facet."""
    if "facet_report_index" not in _CACHE:
        grouped: dict[tuple[str, str, str], dict[str, dict]] = {}
        for row in meta:
            key = (str(row["ticker"]), str(row["year"]), str(row["scope"]))
            report_id = str(row["table_ref"]).split("|", 1)[0]
            grouped.setdefault(key, {}).setdefault(report_id, row)
        _CACHE["facet_report_index"] = grouped
    reports = _CACHE["facet_report_index"].get((ticker, year, scope), {})
    return [dict(reports[report_id]) for report_id in sorted(reports)]


def catalog_universe_reports(
    meta: list[dict],
    years: list[str],
    scope: str,
) -> list[dict]:
    """Return one deterministic representative per report for a year/scope universe."""
    wanted_years = {str(year) for year in years}
    reports: dict[str, dict] = {}
    for row in meta:
        if str(row["year"]) not in wanted_years or str(row["scope"]) != scope:
            continue
        report_id = str(row["table_ref"]).split("|", 1)[0]
        reports.setdefault(report_id, row)
    return [dict(reports[report_id]) for report_id in sorted(reports)]


def select_common_evidence_year(
    ranked_rows: list[dict],
    tickers: list[str],
    preferred_scope: str,
) -> str | None:
    """Select the year with strong report evidence for every named entity.

    This is used only when a multi-entity question omitted its year.  Candidate
    years must have a ranked preferred-scope report for every entity.  The
    weakest entity score dominates, then total evidence and recency break ties.
    """
    wanted = {str(ticker) for ticker in tickers}
    if len(wanted) < 2:
        return None
    scores: dict[str, dict[str, float]] = {}
    for row in ranked_rows:
        ticker = str(row.get("ticker", ""))
        year = str(row.get("year", ""))
        if (
            ticker not in wanted
            or str(row.get("scope", "")) != preferred_scope
            or not re.fullmatch(r"20\d{2}", year)
        ):
            continue
        value = float(row.get("score", 0.0))
        scores.setdefault(year, {})[ticker] = max(
            scores.setdefault(year, {}).get(ticker, float("-inf")), value
        )
    complete = {
        year: by_ticker
        for year, by_ticker in scores.items()
        if wanted.issubset(by_ticker)
    }
    if not complete:
        return None
    return max(
        complete,
        key=lambda year: (
            min(complete[year][ticker] for ticker in wanted),
            sum(complete[year][ticker] for ticker in wanted),
            int(year),
        ),
    )


def _unique_report_for_facet(meta: list[dict], ticker: str, year: str, scope: str) -> dict | None:
    """Return a representative table when one report uniquely owns a facet.

    The BM25 top-k can omit every table from the requested scope even though
    the catalog contains exactly one matching report.  This lookup uses only
    public catalog metadata (ticker/year/scope), never an answer or question
    identifier.  Ambiguous facet groups deliberately fail closed.
    """
    reports = _reports_for_facet(meta, ticker, year, scope)
    if len(reports) != 1:
        return None
    return reports[0]


def series_report_variant_limit(
    facets: dict,
    requested_per_pair: int,
    *,
    question: str = "",
    retain_ambiguous_series_reports: bool = False,
) -> int:
    """Return a bounded report-variant limit for long single-entity series.

    Some entity/year/scope catalog facets contain two physical filings.  A
    multi-year selection may need a metric that exists only in the lower BM25
    variant, so collapsing the facet to one report loses source coverage.  We
    retain at most two variants only for analytic single-entity series with at
    least four explicit years.  Ordinary lookups and multi-entity questions
    keep the caller's original precision-oriented limit.
    """
    limit = max(1, int(requested_per_pair))
    if (
        retain_ambiguous_series_reports
        and len(facets.get("tickers", [])) == 1
        and len(facets.get("years", [])) >= 4
        and bool(facets.get("analytic"))
        and bool(_MULTI_STAGE_SERIES_RE.search(fold(question)))
    ):
        return max(limit, 2)
    return limit


def retrieve_decomposed(
    question: str,
    per_pair: int = 1,
    pool: int = 1000,
    cap: int = 40,
    *,
    backfill_unique_preferred_scope: bool = False,
    backfill_unique_scope_neutral: bool = False,
    backfill_explicit_opening_year: bool = False,
    backfill_financial_formula_year: bool = False,
    backfill_growth_scan_year: bool = False,
    backfill_comparative_selection_year: bool = False,
    backfill_comparative_series_year: bool = False,
    backfill_sparse_series_year: bool = False,
    retain_ambiguous_series_reports: bool = False,
    implicit_scope_minimum_ratio: float = 0.0,
    implicit_ownership_scope_minimum_ratio: float = 0.0,
    retain_count_scope_fallback: bool = False,
    expand_catalog_universe: bool = False,
    universe_cap: int = 200,
    infer_common_evidence_year: bool = False,
    evidence_year_pool: int = 5000,
) -> list[dict]:
    """Phân rã (mã × năm) -> đảm bảo PHỦ mọi cặp -> union. Đẩy DOCS recall cho câu phân tích.

    Với mỗi cặp (mã, năm) lấy `per_pair` báo cáo tốt nhất MỖI scope. Không có mã/năm thì lùi về BM25 thường.
    """
    retriever, meta = _load_index()
    fac = extract_all_facets(question)
    tickers = fac["tickers"]
    scope_overrides = extract_year_scope_overrides(question)
    if infer_common_evidence_year and tickers and not fac["years"]:
        evidence_idx, evidence_scores = retriever.retrieve(
            [tokenize(question)],
            k=min(max(pool, int(evidence_year_pool)), len(meta)),
            show_progress=False,
        )
        evidence_rows = [
            {"score": float(score), **meta[index]}
            for index, score in zip(
                evidence_idx[0].tolist(), evidence_scores[0].tolist()
            )
        ]
        inferred_year = select_common_evidence_year(
            evidence_rows,
            tickers,
            fac["scope"],
        )
        if inferred_year is not None:
            fac = {**fac, "years": [inferred_year], "inferred_year": inferred_year}
    years = expand_retrieval_years(
        question,
        fac["years"],
        opening=backfill_explicit_opening_year,
        formula=backfill_financial_formula_year,
        growth_scan=backfill_growth_scan_year,
        comparative_selection_entities=(
            len(tickers) if backfill_comparative_selection_year else 0
        ),
        comparative_series_entities=(
            len(tickers) if backfill_comparative_series_year else 0
        ),
        sparse_series_entities=(len(tickers) if backfill_sparse_series_year else 0),
        sparse_series_analytic=(bool(fac["analytic"]) if backfill_sparse_series_year else False),
    )
    if expand_catalog_universe and needs_catalog_universe_scan(question, fac):
        q_tokens = tokenize(question)
        idx, scores = retriever.retrieve(
            [q_tokens], k=min(pool, len(meta)), show_progress=False
        )
        score_by_report: dict[str, float] = {}
        for index, score in zip(idx[0].tolist(), scores[0].tolist()):
            report_id = str(meta[index]["table_ref"]).split("|", 1)[0]
            score_by_report[report_id] = max(
                score_by_report.get(report_id, float("-inf")), float(score)
            )
        reports = catalog_universe_reports(meta, years, fac["scope"])
        enriched = []
        for row in reports:
            report_id = str(row["table_ref"]).split("|", 1)[0]
            enriched.append(
                {
                    "score": score_by_report.get(report_id, 0.0),
                    "retrieval_reason": "catalog_universe_scan",
                    **row,
                }
            )
        enriched.sort(
            key=lambda row: (
                -float(row["score"]),
                str(row["table_ref"]).split("|", 1)[0],
            )
        )
        return enriched[: max(cap, int(universe_cap))]
    if not tickers or not years:
        return retrieve(question, k=cap, doc_diverse=True)
    q_tokens = tokenize(question)
    idx, scores = retriever.retrieve([q_tokens], k=min(pool, len(meta)), show_progress=False)
    idx, scores = idx[0], scores[0]
    tset, yset = set(tickers), set(years)
    report_variant_limit = series_report_variant_limit(
        fac,
        per_pair,
        question=question,
        retain_ambiguous_series_reports=retain_ambiguous_series_reports,
    )
    # gom ứng viên theo (mã, năm, scope) -> giữ bảng điểm cao nhất mỗi báo cáo
    by_report: dict[str, dict] = {}
    for i, s in zip(idx.tolist(), scores.tolist()):
        m = meta[i]
        if m["ticker"] in tset and m["year"] in yset:
            rid = m["table_ref"].split("|")[0]
            if rid not in by_report or s > by_report[rid]["score"]:
                by_report[rid] = {"table_ref": m["table_ref"], "score": s, **m}
    # Mỗi cặp (mã, năm) CHỈ lấy 1 scope = scope phát hiện được (gold thường 1 scope);
    # scope kia chỉ dùng dự phòng khi scope chính không có báo cáo -> precision cao mà giữ recall.
    picked: dict[str, dict] = {}
    for tk in tickers:
        for yr in years:
            pref = scope_overrides.get(yr, fac["scope"])
            other = "công ty mẹ" if pref == "hợp nhất" else "hợp nhất"
            scope_order = (pref, "không rõ", other) if backfill_unique_scope_neutral else (pref, other)
            selected_scope = None
            for sc in scope_order:
                # A scope-neutral report has no reliable consolidated/separate
                # label in the public catalog.  It is safer than selecting the
                # explicitly opposite scope, but only when ticker/year maps to
                # exactly one neutral report.  Ambiguous neutral groups fail
                # closed instead of being ranked into a guess.
                if sc == "không rõ":
                    representative = _unique_report_for_facet(meta, tk, yr, sc)
                    cand = []
                    if representative is not None:
                        pair_scores = [
                            float(row["score"])
                            for row in by_report.values()
                            if row["ticker"] == tk and row["year"] == yr
                        ]
                        proxy_score = max(pair_scores) if pair_scores else float(scores[-1])
                        cand = [{"score": proxy_score, **representative}]
                else:
                    cand = sorted([r for r in by_report.values()
                                   if r["ticker"] == tk and r["year"] == yr and r["scope"] == sc],
                                  key=lambda x: -x["score"])[:report_variant_limit]
                    if (
                        retain_ambiguous_series_reports
                        and report_variant_limit > per_pair
                        and len(cand) < report_variant_limit
                    ):
                        present = {str(row["table_ref"]).split("|", 1)[0] for row in cand}
                        pair_scores = [
                            float(row["score"])
                            for row in by_report.values()
                            if row["ticker"] == tk and row["year"] == yr
                        ]
                        proxy_score = min(pair_scores) if pair_scores else float(scores[-1])
                        for representative in _reports_for_facet(meta, tk, yr, sc):
                            report_id = str(representative["table_ref"]).split("|", 1)[0]
                            if report_id in present:
                                continue
                            cand.append({"score": proxy_score, **representative})
                            present.add(report_id)
                            if len(cand) >= report_variant_limit:
                                break
                if not cand and sc == pref and backfill_unique_preferred_scope:
                    representative = _unique_report_for_facet(meta, tk, yr, sc)
                    if representative is not None:
                        pair_scores = [
                            float(row["score"])
                            for row in by_report.values()
                            if row["ticker"] == tk and row["year"] == yr
                        ]
                        proxy_score = max(pair_scores) if pair_scores else float(scores[-1])
                        cand = [{"score": proxy_score, **representative}]
                if cand:
                    for r in cand:
                        picked[r["table_ref"].split("|")[0]] = r
                    selected_scope = sc
                    break                                 # đã có scope chính -> bỏ scope kia
            if selected_scope == pref and (
                implicit_scope_minimum_ratio > 0
                or implicit_ownership_scope_minimum_ratio > 0
                or retain_count_scope_fallback
            ):
                preferred_candidates = sorted(
                    [
                        row
                        for row in by_report.values()
                        if row["ticker"] == tk
                        and row["year"] == yr
                        and row["scope"] == pref
                    ],
                    key=lambda row: -row["score"],
                )
                opposite_candidates = sorted(
                    [
                        row
                        for row in by_report.values()
                        if row["ticker"] == tk
                        and row["year"] == yr
                        and row["scope"] == other
                    ],
                    key=lambda row: -row["score"],
                )
                global_competition = bool(
                    preferred_candidates
                    and opposite_candidates
                    and is_competitive_implicit_scope(
                        question,
                        float(preferred_candidates[0]["score"]),
                        float(opposite_candidates[0]["score"]),
                        minimum_ratio=implicit_scope_minimum_ratio,
                    )
                )
                ownership_competition = bool(
                    preferred_candidates
                    and opposite_candidates
                    and is_competitive_ownership_scope(
                        question,
                        fac,
                        float(preferred_candidates[0]["score"]),
                        float(opposite_candidates[0]["score"]),
                        minimum_ratio=implicit_ownership_scope_minimum_ratio,
                    )
                )
                count_fallback = bool(
                    retain_count_scope_fallback
                    and not preferred_candidates
                    and opposite_candidates
                    and needs_count_scope_fallback(question, fac)
                )
                if global_competition or ownership_competition or count_fallback:
                    for row in opposite_candidates[:per_pair]:
                        picked[str(row["table_ref"]).split("|", 1)[0]] = row
    out = sorted(picked.values(), key=lambda x: -x["score"])[:cap]
    return out if out else retrieve(question, k=cap, doc_diverse=True)


def tables_in_reports(question: str, report_ids: list[str], n: int = 8, pool: int = 2000) -> list[dict]:
    """Top-n BẢNG liên quan nhất TRONG các báo cáo cho trước (1 báo cáo ~74 bảng!).

    Dùng để FEED answering: đưa nhiều bảng ứng viên trong đúng báo cáo -> mô hình tự tìm số
    (RSL-SQL: thừa bảng vô hại, thiếu bảng mới chết). report_ids = các doc đã truy hồi tight.
    """
    if not report_ids:
        return []
    retriever, meta = _load_index()
    q_tokens = tokenize(question)
    idx, scores = retriever.retrieve([q_tokens], k=min(pool, len(meta)), show_progress=False)
    rset = set(report_ids)
    by_rep: dict[str, list] = {r: [] for r in report_ids}
    for i, s in zip(idx[0].tolist(), scores[0].tolist()):
        m = meta[i]
        rid = m["table_ref"].split("|")[0]
        if rid in rset:
            by_rep[rid].append({"table_ref": m["table_ref"], "score": s, **m})
    # CÂN BẰNG: mỗi báo cáo lấy top-k_per (phủ đủ công ty), rồi bù thêm theo điểm toàn cục.
    k_per = max(1, n // max(1, len(report_ids)))
    picked, seen = [], set()
    for rid in report_ids:
        for r in sorted(by_rep[rid], key=lambda x: -x["score"])[:k_per]:
            picked.append(r); seen.add(r["table_ref"])
    rest = sorted((r for rid in report_ids for r in by_rep[rid] if r["table_ref"] not in seen),
                  key=lambda x: -x["score"])
    picked += rest[: max(0, n - len(picked))]
    return sorted(picked, key=lambda x: -x["score"])[:n]


def build() -> None:
    rows = load_catalog()
    print(f"nạp {len(rows):,} bảng, tokenize...", flush=True)
    corpus_tokens = [tokenize(r["search_text"]) for r in rows]
    print("index BM25...", flush=True)
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    retriever.save(str(INDEX_DIR))
    # lưu meta song song (table_ref/ticker/year/scope) theo đúng thứ tự corpus
    with open(INDEX_DIR / "meta.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in ("table_ref", "ticker", "year", "scope")}, ensure_ascii=False) + "\n")
    print(f"DONE -> {INDEX_DIR}")


_CACHE = {}


def _load_index():
    if "r" not in _CACHE:
        _CACHE["r"] = bm25s.BM25.load(str(INDEX_DIR), load_corpus=False)
        _CACHE["meta"] = [json.loads(l) for l in open(INDEX_DIR / "meta.jsonl", encoding="utf-8")]
    return _CACHE["r"], _CACHE["meta"]


def retrieve(question: str, k: int = 10, pool: int = 300, doc_diverse: bool = False,
             year_window: int = 0) -> list[dict]:
    """doc_diverse=True: mỗi BÁO CÁO (doc) chỉ giữ bảng tốt nhất, rồi lấy top-k báo cáo.
    year_window>0: cho phép năm lân cận (BCTC hiện 2 năm: số năm Y nằm ở cả báo cáo Y và Y±1)."""
    retriever, meta = _load_index()
    fac = extract_facets(question)
    q_tokens = tokenize(question)
    idx, scores = retriever.retrieve([q_tokens], k=min(pool, len(meta)), show_progress=False)
    idx, scores = idx[0], scores[0]
    cands = []
    for i, s in zip(idx.tolist(), scores.tolist()):
        m = meta[i]
        # lọc-trước theo mã CK + năm nếu bóc được; scope ưu tiên nhưng không loại cứng
        if fac["ticker"] and m["ticker"] != fac["ticker"]:
            continue
        if fac["year"] and m["year"]:
            try:
                if abs(int(m["year"]) - int(fac["year"])) > year_window:
                    continue
            except ValueError:
                if m["year"] != fac["year"]:
                    continue
        elif fac["year"] and m["year"] != fac["year"]:
            continue
        bonus = 0.5 if m["scope"] == fac["scope"] else 0.0
        # phạt nhẹ năm lệch để năm ĐÚNG vẫn đứng trước khi doc_diverse dedup
        if fac["year"] and m["year"] and m["year"] != fac["year"]:
            bonus -= 0.25
        cands.append({"table_ref": m["table_ref"], "score": s + bonus, **m})
    cands.sort(key=lambda x: -x["score"])
    if doc_diverse:                                    # 1 bảng/1 báo cáo (giữ điểm cao nhất)
        seen: dict[str, dict] = {}
        for c in cands:
            rid = c["table_ref"].split("|")[0]
            if rid not in seen:
                seen[rid] = c
        cands = list(seen.values())
    return cands[:k] if cands else [{"table_ref": meta[i]["table_ref"], "score": s, **meta[i]} for i, s in zip(idx.tolist()[:k], scores.tolist()[:k])]


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    if cmd == "build":
        build()
    elif cmd == "query":
        q = sys.argv[2]
        print("facets:", extract_facets(q))
        for r in retrieve(q, k=8):
            print(f"  {r['score']:.2f}  {r['table_ref']}  [{r['scope']}]")
