"""Resolve Vietnamese company mentions to tickers.

378 of the 1,012 questions name a company without its ticker, so `code_stock.csv`
matching is not optional. Legal-form wording varies between the roster and the
questions ("Tổng Công ty" vs "Tổng công ty", "cổ phần" vs "CP", a trailing
"- CTCP"), so both sides are reduced to a distinctive core before matching.

Matching is deliberately whole-core, never partial: "Ngân hàng TMCP Sài Gòn
Tài Lộc" (a company that does not exist — question id 3 is a hallucination trap)
shares the prefix "Sài Gòn" with SHB, and a substring match would silently
answer the trap with SHB's numbers.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Ordered longest-first so "Công ty Cổ phần" is consumed before "Công ty".
_LEGAL_FORMS = (
    "ngân hàng thương mại cổ phần",
    "công ty tài chính tổng hợp cổ phần",
    "tổng công ty cổ phần",
    "công ty cổ phần",
    "ngân hàng tmcp",
    "tổng công ty",
    "công ty cp",
    "tập đoàn",
    "công ty",
    "ctcp",
    "tct",
)

_NOISE_RE = re.compile(r"[^\w\s]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Vietnamese tone marks, as combining characters after NFD.
_TONES = "̣̀́̃̉"

# Both tone placements are correct Vietnamese orthography and NFC does not unify
# them: the roster writes "Hòa Phát" while 36 questions write "Hoà Phát". Letter
# distinctions (ă â ê ô ơ ư đ) are preserved; only the tone is dropped.
_STRIP_TONES = str.maketrans("", "", _TONES)


def strip_tones(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", decomposed.translate(_STRIP_TONES))


# Tones are stripped before legal forms are matched, so the form list has to be
# toneless too or the stripping silently stops working.
_TONELESS_LEGAL_FORMS = tuple(strip_tones(form) for form in _LEGAL_FORMS)


def normalize_core(name: str) -> str:
    """Strip legal form, punctuation, casing, and tone marks down to the core."""

    text = strip_tones(unicodedata.normalize("NFC", name)).casefold()
    text = _NOISE_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    changed = True
    while changed:
        changed = False
        for form in _TONELESS_LEGAL_FORMS:
            if text.startswith(form + " "):
                text, changed = text[len(form) + 1:].strip(), True
            if text.endswith(" " + form):
                text, changed = text[: -len(form) - 1].strip(), True
    return _WS_RE.sub(" ", text).strip()


# Trading brands that never appear in the registered name. Each target was
# checked against the roster; Sacombank is deliberately absent because the
# roster assigns STB to "Ngân hàng TMCP Sài Gòn Tài Lộc" and the roster wins.
BRAND_ALIASES = {
    "vietcombank": "VCB",
    "vietinbank": "CTG",
    "bidv": "BID",
    "mbbank": "MBB",
    "mb bank": "MBB",
    "eximbank": "EIB",
    "petrolimex": "PLX",
    "saigonbank": "SGB",
    "viglacera": "VGC",
    "vinatex": "VGT",
    "novaland": "NVL",
    "sabeco": "SAB",
    "vinamilk": "VNM",
    # Trading short names used instead of the roster legal core.
    "nam kim": "NKG",
    "hoa sen": "HSG",
}


@dataclass(frozen=True, slots=True)
class Company:
    ticker: str
    name: str
    core: str


class CompanyRoster:
    def __init__(self, companies: list[Company]) -> None:
        self.companies = companies
        self.by_ticker = {c.ticker: c for c in companies}
        # Longest core first so "sài gòn hà nội" wins over any shorter overlap.
        self._ordered = sorted(companies, key=lambda c: -len(c.core))

    @classmethod
    def load(cls, csv_path: Path) -> "CompanyRoster":
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))[1:]
        return cls([Company(t.strip(), n.strip(), normalize_core(n)) for t, n in rows if t.strip()])

    def match_names(self, question: str) -> list[str]:
        """Tickers whose full normalized core appears in the question.

        12 roster cores nest inside another ("điện lực" ⊂ "điện lực tkv",
        "masan" ⊂ "masan meatlife"), so a matched span is masked out before
        shorter cores are tried. Longest-first ordering makes the specific
        company win; the generic one only matches where it stands alone.
        """

        haystack = normalize_core(question)
        hits: list[str] = []
        for company in self._ordered:
            if not company.core:
                continue
            pattern = re.compile(rf"(?<!\w){re.escape(company.core)}(?!\w)")
            match = pattern.search(haystack)
            if match is None:
                continue
            hits.append(company.ticker)
            haystack = pattern.sub(lambda m: "\x00" * len(m.group(0)), haystack)
        return hits

    def match_aliases(self, question: str) -> list[str]:
        haystack = normalize_core(question)
        hits: list[str] = []
        for alias, ticker in BRAND_ALIASES.items():
            if ticker in self.by_ticker and ticker not in hits:
                if re.search(rf"(?<!\w){re.escape(strip_tones(alias))}(?!\w)", haystack):
                    hits.append(ticker)
        return hits

    def match_partial_names(self, question: str) -> list[str]:
        """Fallback for shortened brand names, e.g. "Đô thị Kinh Bắc" for KBC.

        Only the leading tokens of a core may be dropped, the remainder must stay
        distinctive (2+ tokens, 10+ chars), and the tail must identify exactly one
        company — otherwise the mention stays unresolved rather than guessed.
        """

        haystack = normalize_core(question)
        hits: list[str] = []
        for company in self._ordered:
            tokens = company.core.split()
            for start in range(1, max(1, len(tokens) - 1)):
                tail = " ".join(tokens[start:])
                if len(tail) < 10 or len(tokens) - start < 2:
                    break
                if not re.search(rf"(?<!\w){re.escape(tail)}(?!\w)", haystack):
                    continue
                if sum(1 for c in self.companies if tail in c.core) == 1:
                    hits.append(company.ticker)
                break
        return hits

    def match_tickers(self, question: str) -> list[str]:
        """Explicit ticker symbols, whether parenthesised or standalone.

        Two roster symbols carry a digit (PC1, HT1). Lowercase symbols are only
        honoured inside an explicit enumeration — questions write "(gồm các công
        ty hpx,kbc,nvl)" — because bare lowercase roster codes such as "gas",
        "ceo", or "fit" collide with ordinary words.
        """

        found = []
        for token in re.findall(r"(?<![\wÀ-ỹ])([A-Z][A-Z0-9]{2,3})(?![\wÀ-ỹ])", question):
            if token in self.by_ticker and token not in found:
                found.append(token)
        for enumeration in re.findall(r"(?:gồm|bao gồm|các công ty)([^)?.]{0,160})", question, re.I):
            if "," not in enumeration:
                continue
            for token in re.findall(r"(?<![\wÀ-ỹ])([a-z][a-z0-9]{2,3})(?![\wÀ-ỹ])", enumeration):
                upper = token.upper()
                if upper in self.by_ticker and upper not in found:
                    found.append(upper)
        return found
