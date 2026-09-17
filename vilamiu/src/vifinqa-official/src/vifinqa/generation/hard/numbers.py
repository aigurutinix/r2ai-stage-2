
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

_NUMERIC_RE = re.compile(r"^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$")


class VnNumberError(ValueError):
    pass


def parse_vn_number(raw: str) -> float:
    s = str(raw).strip()
    if not s or s == "-":
        raise VnNumberError(f"Cell is empty or non-numeric: {raw!r}")
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    if s.startswith("-"):
        negative = True
        s = s[1:].strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    if not _NUMERIC_RE.match(s):
        raise VnNumberError(f"Invalid Vietnamese number format: {raw!r}")
    s = s.replace(".", "").replace(",", ".")
    value = float(s)
    return -value if negative else value


VN_NUMBER_PARSER_SOURCE = '''\
def _parse_vn_number(raw):
    s = str(raw).strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    if s.startswith("-"):
        negative = True
        s = s[1:].strip()
    if s.endswith("%"):
        s = s[:-1].strip()
    s = s.replace(".", "").replace(",", ".")
    value = float(s)
    return -value if negative else value
'''


# --- Unit contract ---------------------------------------------------------------

UnitKind = Literal[
    "vnd",
    "thousand_vnd",
    "million_vnd",
    "billion_vnd",
    "percentage",
    "number",
    "unknown",
]
UnitSource = Literal["csv_header", "unit_snippet", "cell", "none"]
UnitValueKind = Literal["money", "percentage", "number"]

# Keep unit and scale handling explicit.
UNIT_MULTIPLIERS: dict[str, tuple[float, str]] = {
    "vnd": (1.0, "đồng"),
    "thousand_vnd": (1e3, "đồng"),
    "million_vnd": (1e6, "đồng"),
    "billion_vnd": (1e9, "đồng"),
    "percentage": (1.0, "%"),
    "number": (1.0, "đơn vị"),
}

_MONEY_KINDS = frozenset({"vnd", "thousand_vnd", "million_vnd", "billion_vnd"})

_MILLION_TOKENS = (r"\btriệu\b", r"\bmillion\b")
_BILLION_TOKENS = (r"\btỷ\b", r"\bbillion\b")
_THOUSAND_TOKENS = (r"\bnghìn\b", r"\bthousand\b")
_DONG_TOKENS = (r"\bđồng\b", r"\bvnđ\b", r"\bvnd\b", r"\bdong\b")
_UNIT_MARKER_TOKENS = (r"\bđơn\s*vị\b", r"\bđvt\b", r"\bvnđ\b", r"\bvnd\b")

# Keep unit and scale handling explicit.
_MILLION_TOKENS_FOLDED = (r"\btrieu\b",)
_BILLION_TOKENS_FOLDED = (r"\bty\s+(?:dong|vnd)\b",)
_THOUSAND_TOKENS_FOLDED = (r"\bnghin\b",)
_UNIT_MARKER_TOKENS_FOLDED = (r"\bdon\s*vi\b", r"\bdvt\b")

_KIND_REQUIRED_TOKENS: dict[str, tuple[str, ...]] = {
    "million_vnd": _MILLION_TOKENS,
    "billion_vnd": _BILLION_TOKENS,
    "thousand_vnd": _THOUSAND_TOKENS,
    "vnd": _DONG_TOKENS,
}
_KIND_REQUIRED_TOKENS_FOLDED: dict[str, tuple[str, ...]] = {
    "million_vnd": _MILLION_TOKENS_FOLDED,
    "billion_vnd": _BILLION_TOKENS_FOLDED,
    "thousand_vnd": _THOUSAND_TOKENS_FOLDED,
}
_KIND_FORBIDDEN_TOKENS: dict[str, tuple[str, ...]] = {
    "vnd": _MILLION_TOKENS + _BILLION_TOKENS + _THOUSAND_TOKENS,
    "million_vnd": _BILLION_TOKENS + _THOUSAND_TOKENS,
    "billion_vnd": _MILLION_TOKENS + _THOUSAND_TOKENS,
    "thousand_vnd": _MILLION_TOKENS + _BILLION_TOKENS,
}
_KIND_FORBIDDEN_TOKENS_FOLDED: dict[str, tuple[str, ...]] = {
    "vnd": _MILLION_TOKENS_FOLDED + _BILLION_TOKENS_FOLDED + _THOUSAND_TOKENS_FOLDED,
    "million_vnd": _BILLION_TOKENS_FOLDED + _THOUSAND_TOKENS_FOLDED,
    "billion_vnd": _MILLION_TOKENS_FOLDED + _THOUSAND_TOKENS_FOLDED,
    "thousand_vnd": _MILLION_TOKENS_FOLDED + _BILLION_TOKENS_FOLDED,
}


def _fold_diacritics(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")


def _normalize_for_match(text: str) -> str:
    # Keep unit and scale handling explicit.
    normalized = " ".join(text.strip().casefold().replace("_", " ").split())
    # Keep unit and scale handling explicit.
    # Keep period handling explicit and deterministic.
    return re.sub(r"(?<=\d)(?=[^\W\d_])", " ", normalized)


def _has_any_token(
    text: str, patterns: tuple[str, ...], *, folded_patterns: tuple[str, ...] = ()
) -> bool:
    norm = _normalize_for_match(text)
    if any(re.search(pattern, norm) for pattern in patterns):
        return True
    if not folded_patterns:
        return False
    folded_norm = _normalize_for_match(_fold_diacritics(text))
    return any(re.search(pattern, folded_norm) for pattern in folded_patterns)


def _evidence_matches_kind(kind: str, evidence: str) -> bool:
    # Keep unit and scale handling explicit.
    if not _has_any_token(evidence, _DONG_TOKENS):
        return False
    has_unit_marker = _has_any_token(
        evidence,
        (r"\bđơn\s*vị\b", r"\bđvt\b", r"\bvnđ\b", r"\bvnd\b", r"\btiền\s*tệ\b"),
        folded_patterns=_UNIT_MARKER_TOKENS_FOLDED,
    )
    if kind == "vnd" and not has_unit_marker:
        return False
    required = _KIND_REQUIRED_TOKENS.get(kind, ())
    required_folded = _KIND_REQUIRED_TOKENS_FOLDED.get(kind, ())
    if required and not _has_any_token(evidence, required, folded_patterns=required_folded):
        return False
    forbidden = _KIND_FORBIDDEN_TOKENS.get(kind, ())
    forbidden_folded = _KIND_FORBIDDEN_TOKENS_FOLDED.get(kind, ())
    if forbidden and _has_any_token(evidence, forbidden, folded_patterns=forbidden_folded):
        return False
    return True


class UnitClaimProtocol:

    kind: str
    evidence: str
    source: str


@dataclass(frozen=True, slots=True)
class ResolvedUnit:
    scale: float
    normalized_unit: str
    raw_evidence: str


def scale_from_unit_text(text: str) -> tuple[float, str] | None:
    for kind in ("billion_vnd", "million_vnd", "thousand_vnd", "vnd"):
        if _evidence_matches_kind(kind, text):
            return UNIT_MULTIPLIERS[kind]
    return None


def resolve_unit_claim(
    claim: UnitClaimProtocol,
    *,
    csv_header: str,
    unit_snippets: list[str],
    value_kind: UnitValueKind,
) -> ResolvedUnit | None:
    kind = claim.kind
    evidence = claim.evidence.strip()

    if value_kind == "percentage":
        if kind != "percentage":
            return None
        scale, normalized_unit = UNIT_MULTIPLIERS["percentage"]
        return ResolvedUnit(scale=scale, normalized_unit=normalized_unit, raw_evidence=evidence)

    if value_kind == "number":
        if kind not in ("number", "unknown"):
            return None
        scale, normalized_unit = UNIT_MULTIPLIERS["number"]
        return ResolvedUnit(scale=scale, normalized_unit=normalized_unit, raw_evidence=evidence)

    # value_kind == "money"
    if kind not in _MONEY_KINDS:
        return None
    if not evidence:
        return None
    if claim.source == "csv_header":
        source_text = csv_header
    elif claim.source == "unit_snippet":
        source_text = "\n".join(unit_snippets)
    else:
        return None
    if evidence not in source_text:
        return None
    if not _evidence_matches_kind(kind, evidence):
        return None
    scale, normalized_unit = UNIT_MULTIPLIERS[kind]
    return ResolvedUnit(scale=scale, normalized_unit=normalized_unit, raw_evidence=evidence)
