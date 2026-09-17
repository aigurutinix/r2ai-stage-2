"""Row-level schema linking: turn N wide tables into a short list of candidate
cells, so the LLM picks from ~8 options instead of searching ~40 rows x 4 tables.

This is the text-to-SQL "schema linking" step (CHESS / DIN-SQL) applied to
pandas: narrowing BEFORE generation is what makes small models reliable.

Scoring per row label = max over metric variants of:
    lexical  : label_metric_score (token coverage + order-insensitive ratio)
    semantic : cosine(BGE-M3 label, BGE-M3 metric)   [optional, see dense.py]
plus small bonuses for a matching VAS code and for the qualifier
(ngan han / dai han / hop nhat) agreeing with the question.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from functools import lru_cache

from ..finance.metrics import (
    METRICS,
    code_expectation,
    expand_metric_variants,
    extract_metric_qualifiers,
    get_metric,
    metric_keys,
    metric_schema_score,
)
from ..router.metric_phrase import has_qualifier
from ..utils.viet_text import label_metric_score, norm, tokens
from .serialize import df_roundtrip

_QUALIFIERS = ("ngan han", "dai han", "hop nhat", "rieng")


@dataclass
class Candidate:
    var: str
    report_id: str
    table_pos: int
    row: int
    label: str
    code: str
    col: int
    col_name: str
    value: float
    unit_scale: float
    score: float
    lexical: float
    semantic: float

    def to_dict(self) -> dict:
        return asdict(self)


def candidate_matches_metric(candidate: Candidate, metric_key: str) -> bool:
    """Require canonical label identity or an exact VAS line code."""
    return metric_identity_matches(candidate.label, candidate.code, metric_key)


def candidate_matches_requirement(candidate: Candidate, requirement: dict) -> bool:
    """Canonical identity plus any counterparty/detail terms in the requirement."""
    metric_key = str(requirement.get("metric_key") or "")
    if metric_key and not candidate_matches_metric(candidate, metric_key):
        return False
    variants = list(requirement.get("metric_variants") or [])
    keys = [metric_key] if metric_key else metric_keys(variants, expand_derived=False)
    terms = _specific_terms(variants, keys)
    return not terms or _specific_term_coverage(candidate.label, terms) >= 0.8


def requirement_specificity_key(requirement: dict) -> tuple[str, ...]:
    """Small cache fingerprint; generic wording must not destroy cache reuse."""
    metric_key = str(requirement.get("metric_key") or "")
    variants = list(requirement.get("metric_variants") or [])
    keys = [metric_key] if metric_key else metric_keys(variants, expand_derived=False)
    return _specific_terms(variants, keys)


def requirement_linking_variants(requirement: dict) -> list[str]:
    """Use source wording only when it identifies a named detail row."""
    variants = list(requirement.get("metric_variants") or [])
    if requirement_specificity_key(requirement):
        return variants
    metric_key = str(requirement.get("metric_key") or "")
    try:
        return list(get_metric(metric_key).row_variants)
    except KeyError:
        return variants


def metric_identity_matches(label: str, code: str, metric_key: str) -> bool:
    """Canonical identity check usable before a Candidate is materialized."""
    if metric_key in metric_keys([label], expand_derived=False):
        return True
    try:
        metric = get_metric(metric_key)
    except KeyError:
        return False
    label_norm = norm(label)
    if any(
        label_norm == variant or label_norm.startswith(f"{variant} ")
        for variant in metric.row_variants
    ):
        return True
    code = re.sub(r"\.0$", "", str(code or "").strip())
    return bool(code and code in metric.codes)


def build_shortlist(tables: list[dict], metric_variants: list[str],
                    years: list[int] | None = None, top_n: int = 8,
                    encoder=None, min_score: float = 35.0,
                    question: str = "") -> list[Candidate]:
    """tables: bundle tables ({var, report_id, table_pos, csv_text, ...}).

    Returns the best `top_n` (label, column) cells across all tables, sorted by
    descending score. One entry per (table, label, chosen column).
    """
    original_variants = [m for m in metric_variants if m]
    asked_keys = metric_keys(original_variants, expand_derived=False)
    specific_terms = _specific_terms(original_variants, asked_keys)
    requested = extract_metric_qualifiers(
        " ".join((*original_variants, question)), asked_keys)
    metric_variants = expand_metric_variants(original_variants, question=question)
    if not metric_variants:
        return []
    want_qual = _qualifiers_in(metric_variants[0])

    sem_lookup = {}
    if encoder is not None:
        all_labels = []
        for t in tables:
            all_labels.extend(
                label for label, _ in _cached_label_groups(t["csv_text"])
                if len(label) > 3
            )
        if all_labels:
            sem_lookup = encoder.similarity(metric_variants, sorted(set(all_labels)))

    out: list[Candidate] = []
    for t in tables:
        year_hints = _cached_column_year_hints(t["csv_text"])
        grid_hints = _cached_grid_column_hints(str(t.get("grid_json") or ""))
        for label, sub in _cached_label_groups(t["csv_text"]):
            if len(label) <= 3:
                continue
            lex = max(label_metric_score(label, m) for m in metric_variants)
            sem = float(sem_lookup.get(label, 0.0)) * 100.0
            score = max(lex, sem) + 0.25 * min(lex, sem)
            score += _qualifier_bonus(label, want_qual)
            score += metric_schema_score(original_variants, label, question)
            score += _specificity_adjustment(label, specific_terms)
            if score < min_score:
                continue
            # Prefer the TIGHTEST label covering the metric: extra qualifier
            # tokens change the meaning ("Lợi nhuận sau thuế" vs "... chưa phân
            # phối"), yet token-coverage alone scores both 100.
            score -= _extra_token_penalty(label, metric_variants)

            pick = _pick_column(
                sub, years, t.get("report_year"), requested.period,
                year_hints=year_hints, grid_hints=grid_hints)
            if pick is None:
                continue
            row_i, col, col_name, value, unit_scale, code = pick
            # A column header naming the asked year is hard evidence. A header
            # naming a DIFFERENT year is evidence of the WRONG period, which
            # silently corrupts multi-company comparisons (one company read at
            # 2015, another at 2014) - so it is punished much harder than a
            # header with no year at all.
            ys = _year_status(col_name, years)
            score += {"match": 10.0, "none": -6.0, "other": -30.0}[ys]
            # VAS line code is far more stable than OCR'd Vietnamese text:
            # it separates "Lợi nhuận sau thuế" (code 60, income statement) from
            # "Lợi nhuận sau thuế chưa phân phối" (code 421, balance sheet).
            score += _code_bonus(code, metric_variants, label)
            if score < min_score:      # re-check: the adjustments can sink a row
                continue
            out.append(Candidate(
                var=t["var"], report_id=t["report_id"], table_pos=int(t["table_pos"]),
                row=int(row_i), label=str(label), code=str(code), col=int(col),
                col_name=str(col_name), value=float(value),
                unit_scale=float(unit_scale), score=round(float(score), 1),
                lexical=round(float(lex), 1), semantic=round(float(sem), 1)))
    out.sort(key=lambda c: -c.score)
    return _dedupe(out)[:top_n]


@lru_cache(maxsize=512)
def _cached_df_roundtrip(csv_text: str):
    """Avoid reparsing the same table once per formula operand."""
    return df_roundtrip(csv_text)


@lru_cache(maxsize=512)
def _cached_column_year_hints(csv_text: str) -> dict[int, int]:
    """Recover years OCR placed in a separate header row for each column."""
    df = _cached_df_roundtrip(csv_text)
    if not len(df) or not {"row", "col", "value"} <= set(df.columns):
        return {}
    header = df[df["row"] <= 2]
    out: dict[int, int] = {}
    for _row, group in header.groupby("row", sort=False):
        pairs = []
        for item in group.itertuples():
            year = _numeric_header_year(float(item.value))
            if year is not None:
                pairs.append((int(item.col), year))
        # Two adjacent fiscal years on one early row are strong header evidence;
        # a lone year-like line-item value is not.
        if len(pairs) >= 2 and len({year for _col, year in pairs}) >= 2:
            out.update(pairs)
    return out


@lru_cache(maxsize=512)
def _cached_grid_column_hints(grid_json: str) -> dict[int, str]:
    """Recover full dates when tidy OCR columns only retain a shared prefix."""
    try:
        grid = json.loads(grid_json)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(grid, list):
        return {}
    width = max((len(row) for row in grid[:3] if isinstance(row, list)), default=0)
    out = {}
    for col in range(width):
        parts = [str(row[col]).strip() for row in grid[:3]
                 if isinstance(row, list) and col < len(row) and str(row[col]).strip()]
        if parts:
            out[col] = " ".join(parts)
    return out


def _numeric_header_year(value: float) -> int | None:
    """Read either YYYY or OCR-flattened D/M/YYYY header dates."""
    if not math.isfinite(value):
        return None
    integer = int(value)
    if value != integer or integer < 0:
        return None
    if 1900 <= integer <= 2100:
        return integer
    digits = str(integer)
    if len(digits) not in {7, 8}:
        return None
    year = int(digits[-4:])
    return year if 1900 <= year <= 2100 else None


@lru_cache(maxsize=512)
def _cached_label_groups(csv_text: str):
    """Avoid rebuilding a boolean DataFrame slice once per row label."""
    df = _cached_df_roundtrip(csv_text)
    if not len(df):
        return ()
    return tuple(
        (str(label), group)
        for label, group in df.groupby("label", sort=False, dropna=True)
        if isinstance(label, str)
    )


def _dedupe(cands: list[Candidate]) -> list[Candidate]:
    seen, out = set(), []
    for c in cands:
        key = (c.report_id, c.table_pos, norm(c.label), c.col)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def _extra_token_penalty(label: str, metric_variants: list[str],
                         per_token: float = 3.0, cap: float = 18.0) -> float:
    from ..utils.viet_text import tokens
    lt = set(tokens(label))
    best = min((len(lt - set(tokens(m))) for m in metric_variants), default=0)
    return min(cap, per_token * max(0, best - 1))


_SPECIFIC_STOPWORDS = {
    "bao", "cac", "cao", "cho", "chinh", "con", "cong", "co", "cua", "cuoi",
    "dau", "den", "doi", "gia", "han", "ky", "la", "lai", "lon", "me",
    "nam", "nhat", "nhieu", "phan", "so", "tai", "thap", "theo", "tong",
    "trong", "tri", "ty", "va", "voi",
}


def _specific_terms(phrases: list[str], asked_keys: list[str]) -> tuple[str, ...]:
    """Return named detail terms left after removing canonical metric language."""
    if not asked_keys:
        return ()
    base_tokens = {
        token
        for key in asked_keys if key in METRICS
        for variant in METRICS[key].variants
        for token in tokens(variant)
    }
    best: tuple[str, ...] = ()
    for phrase in phrases:
        phrase_norm = norm(phrase)
        phrase_tokens = tokens(phrase_norm)
        if len(phrase_tokens) > 14 or not _has_named_detail_link(phrase_tokens):
            continue
        extra = tuple(
            token for token in phrase_tokens
            if token not in base_tokens
            and token not in _SPECIFIC_STOPWORDS
            and not token.isdigit()
        )
        if sum(len(token) for token in extra) >= 3 and len(extra) > len(best):
            best = extra
    return best


def _has_named_detail_link(phrase_tokens: list[str]) -> bool:
    if "ben lien quan" in " ".join(phrase_tokens):
        return False
    for index, token in enumerate(phrase_tokens):
        previous = phrase_tokens[index - 1] if index else ""
        if token == "voi" and previous not in {"so", "doi"}:
            return True
    return False


def _specific_term_coverage(label: str, terms: tuple[str, ...]) -> float:
    if not terms:
        return 1.0
    label_norm = norm(label)
    compact = "".join(terms)
    if len(compact) >= 3 and compact in label_norm.replace(" ", ""):
        return 1.0
    have = set(tokens(label_norm))
    return sum(term in have for term in terms) / len(terms)


def _specificity_adjustment(label: str, terms: tuple[str, ...]) -> float:
    if not terms:
        return 0.0
    coverage = _specific_term_coverage(label, terms)
    if coverage >= 0.8:
        return 36.0
    if coverage >= 0.5:
        return 10.0
    return -28.0


def _year_status(col_name: str, years: list[int] | None) -> str:
    """'match' | 'other' (header names a different year) | 'none'."""
    if not years:
        return "match"
    cn = str(col_name)
    if any(re.search(rf"(?<!\d){y}(?!\d)", cn) for y in years):
        return "match"
    other = re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", cn)
    return "other" if other else "none"


# Thong tu 200 line codes: the OCR-stable identity of a financial line item.
def _code_bonus(code: str, metric_variants: list[str], label: str = "") -> float:
    expected, known_mismatch = code_expectation(metric_variants, label)
    if known_mismatch:
        return -12.0
    if not expected:
        return 0.0
    c = str(code or "").strip().rstrip(".0") or str(code or "").strip()
    c = re.sub(r"\.0$", "", str(code or "").strip())
    if not c:
        return 0.0
    if c in expected:
        return 12.0
    return -10.0 if c.isdigit() else 0.0


def _qualifiers_in(metric: str) -> set[str]:
    return {q for q in _QUALIFIERS if q in metric}


def _qualifier_bonus(label: str, want: set[str]) -> float:
    """A wrong maturity qualifier means a semantically WRONG row, so the penalty
    must outweigh the lexical gain those extra shared tokens give
    ("...ngắn hạn" vs "...dài hạn" differ by one token but are different lines).
    """
    have = _qualifiers_in(norm(label))
    if not want and not have:
        return 0.0
    if want & have:
        return 8.0
    if want and have:
        return -25.0       # "ngan han" asked but row says "dai han"
    return -3.0 if want else 0.0


_CURRENT_PERIOD_HEADERS = (
    "nam nay", "ky nay", "so cuoi nam", "cuoi nam", "cuoi ky",
    "tai ngay 31 thang 12 nam", "gia goc/so co kha nang tra no",
    "don vi tinh",
    "tai ngay ket thuc", "nam ket thuc ngay", "nam tai chinh ket thuc ngay",
    "tong cong", "current year", "current period", "ending balance", "total",
)
_PRIOR_PERIOD_HEADERS = (
    "nam truoc", "ky truoc", "so dau nam", "dau nam", "dau ky",
    "prior year", "previous year", "beginning balance",
)
_METADATA_HEADERS = (
    "thuyet minh", "ma so", "chi tieu", "noi dung", "dien giai", "stt",
    "thuyetminh", "maso", "chitieu", "item", "code", "note",
)


def _period_kind(col_name: str) -> str:
    """Classify an original table header as current/prior/metadata/unknown."""
    cn = norm(str(col_name or ""))
    if not cn or cn == "nan":
        return "metadata"
    date_sep = r"(?:\s*[./-]\s*|\s+)"
    if re.search(
            rf"(?<!\d)(?:0?1){date_sep}(?:0?1){date_sep}20\d{{2}}(?!\d)",
            cn):
        return "prior"
    if re.search(rf"(?<!\d)31{date_sep}12{date_sep}20\d{{2}}(?!\d)", cn):
        return "current"
    if any(marker in cn for marker in _PRIOR_PERIOD_HEADERS):
        return "prior"
    if any(marker in cn for marker in _CURRENT_PERIOD_HEADERS):
        return "current"
    if any(marker == cn or marker in cn for marker in _METADATA_HEADERS):
        return "metadata"
    return "unknown"


def _column_score(col_name: str, years: list[int] | None,
                  report_year: int | None,
                  requested_period: str = "") -> int:
    """Rank value columns using explicit years and financial-period headers."""
    cn = str(col_name or "")
    requested = {int(y) for y in (years or [])}
    kind = _period_kind(cn)
    period_adjust = 0
    if requested_period == "opening":
        if any(re.search(rf"(?:1|01)\s*/\s*(?:1|01)\s*/\s*{y}", cn)
               for y in requested):
            period_adjust = 45
        elif kind == "prior":
            period_adjust = 35
        elif kind == "current":
            period_adjust = -45
    elif requested_period == "closing":
        if any(re.search(rf"31\s*/\s*12\s*/\s*{y}", cn) for y in requested):
            period_adjust = 45
        elif kind == "current":
            period_adjust = 35
        elif kind == "prior":
            period_adjust = -45

    # A 1/1/Y balance is the closing balance of Y-1. Handle that accounting
    # convention before the generic "different literal year" rejection.
    if (kind == "prior" and report_year is not None
            and int(report_year) - 1 in requested
            and re.search(
                rf"(?<!\d)(?:0?1)\s*/\s*(?:0?1)\s*/\s*{int(report_year)}(?!\d)",
                cn,
            )):
        return 100 + period_adjust

    # A literal requested year beats every inferred current/prior convention.
    for y in requested:
        if re.search(rf"31\s*/\s*12\s*/\s*{y}", cn):
            return 100 + period_adjust
        if re.search(rf"(?<!\d){y}(?!\d)", cn):
            return 95 + period_adjust
    if re.search(r"(?<!\d)(19|20)\d{2}(?!\d)", cn):
        return -40

    if kind == "metadata":
        return -100

    current_wanted = not requested or report_year is None or report_year in requested
    prior_wanted = (report_year is not None and
                    any(int(report_year) == y + 1 for y in requested))
    if kind == "current":
        base = 80 if current_wanted and not prior_wanted else 20
        return base + period_adjust
    if kind == "prior":
        base = 80 if prior_wanted else 10
        return base + period_adjust
    return 30 + period_adjust


def _pick_column(sub, years: list[int] | None,
                 report_year: int | None = None,
                 requested_period: str = "",
                 year_hints: dict[int, int] | None = None,
                 grid_hints: dict[int, str] | None = None):
    """Choose the requested financial period, excluding code/note columns."""
    best = None
    for r in sub.itertuples():
        cn = str(r.col_name)
        grid_hint = (grid_hints or {}).get(int(r.col), "")
        if grid_hint and norm(grid_hint) not in norm(cn):
            cn = f"{cn} {grid_hint}".strip()
        hinted_year = (year_hints or {}).get(int(r.col))
        if hinted_year is not None and not re.search(
                rf"(?<!\d){hinted_year}(?!\d)", cn):
            cn = f"{cn} {hinted_year}".strip()
        cs = _column_score(cn, years, report_year, requested_period)
        # For equally informative headers, retain the original left-to-right
        # convention among actual value columns.
        rank = (cs, -int(r.col))
        if best is None or rank > best[0]:
            best = (rank, r.Index, int(r.col), cn, float(r.value),
                    float(r.unit_scale), str(getattr(r, "code", "")))
    if best is None:
        return None
    _rank, idx, col, cn, val, us, code = best
    row_i = int(sub.loc[idx, "row"]) if "row" in sub.columns else 0
    return row_i, col, cn, val, us, code


def render_shortlist(cands: list[Candidate], unit_name: str = "") -> str:
    """Compact text block for the prompt."""
    if not cands:
        return "(no candidate row matched the metric — search the tables yourself)"
    lines = ["idx | var | label | code | col | col_name | value | unit_scale"]
    for i, c in enumerate(cands, 1):
        lines.append(f"{i} | {c.var} | {c.label[:60]} | {c.code or '-'} | {c.col} | "
                     f"{c.col_name[:22] or '-'} | {c.value:g} | {c.unit_scale:g}")
    return "\n".join(lines)
