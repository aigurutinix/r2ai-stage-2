"""Decompose a question into atomic FACT REQUIREMENTS + a combining operation.

Measured motivation (P1_STRATEGY_REVIEW.md): 507/1012 questions involve >=2
tickers, >=2 years, or an aggregate/ranking verb, and 85% of them came back
empty. A single (ticker, year, metric) route cannot serve them: they need one
evidence set PER ENTITY/PERIOD and a formula to combine the facts.

    "Chênh lệch ... giữa CTCP A và CTCP B cuối năm 2025"
        -> facts  [(A, 2025, metric), (B, 2025, metric)]
        -> op     difference

The output drives:
  * dynamic evidence budget in retrieval (quota per fact, not a flat k)
  * the formula hint given to the codegen prompt
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from ..utils.viet_text import norm

# operation detection, ordered by specificity (first match wins)
_OPS = [
    ("cagr", r"\bcagr\b|tang truong kep|binh quan nam giai doan"),
    ("growth_pct", r"\btang truong\b|\btang bao nhieu phan tram\b|\bty le tang\b|"
                   r"\btang bao nhieu\s*%|\bthay doi bao nhieu phan tram\b|"
                   r"\btang giam bao nhieu phan tram\b|"
                   r"\bty le bien dong\b|"
                   r"\bbien dong\b.*\bbao nhieu\s*(?:phan tram|%)|"
                   r"\b(?:tang|giam)\b.*\bbao nhieu\s*(?:phan tram|%)"),
    ("count", r"\bbao nhieu (cong ty|doanh nghiep|ma|nam)\b|"
              r"\bso luong (cong ty|doanh nghiep)\b|"
              r"\bco bao nhieu (don vi|truong hop)\b|"
              r"\bso nam\b.*\b(?:la )?bao nhieu\b"),
    ("ranking", r"\blon nhat\b|\bnho nhat\b|\bcao nhat\b|\bthap nhat\b|\bdung dau\b|"
                r"\bxep hang\b|\b(?:xep|dung) thu (hai|ba|tu|nam)\b|\bdan dau\b"),
    # Explicit aggregation over a list must win over metric words such as
    # "lai/lo chenh lech ty gia" which otherwise look like a difference op.
    ("sum", r"\btinh tong\b|\btong (?:gia tri|chi phi|so du|du no|so tien)\b.*"
            r"\b(?:cua cac|cho cac nam|trong cac nam)\b"),
    ("difference", r"\bchenh lech\b|\bhieu (?:so|giua)\b|\bcao hon\b|\bthap hon\b|"
                   r"\b(?:nhieu|it|lon|nho|be|kem) hon\b|\bhon kem\b|\btru di\b|"
                   r"\bhon\b.*\b(?:bao nhieu|may) (?:dong|trieu|ty)\b|"
                   r"\bso voi\b.*\bbao nhieu (dong|trieu|ty)\b|"
                   r"\bbien dong\b|"
                   r"\bthay doi\b.*\bbao nhieu (?:diem phan tram|dong|trieu|ty)\b|"
                   r"\b(?:tang|giam)\b.*\b(?:la )?bao nhieu (?:dong|trieu|ty)\b"),
    ("average", r"\btrung binh\b|\bbinh quan\b"),
    # "tong cong ty" is a company-name prefix, NOT an aggregation verb
    ("sum", r"\btong cong\b(?!\s*ty)|\btong cua cac\b|\btong so .* cua cac\b|"
            r"\bcong lai\b|\btong gia tri cua cac\b"),
    ("ratio", r"\bty le\b.*\btren\b|\bchiem bao nhieu\b|\bty trong\b|\bso voi tong\b"),
    ("margin", r"\bbien loi nhuan\b|\bty suat loi nhuan\b|\bros\b|\broe\b|\broa\b"),
    ("hypothetical", r"\bgia su\b|\bneu \b.*\bthi\b"),
]
_OP_RE = [(name, re.compile(rx)) for name, rx in _OPS]

# how many facts each op needs beyond the entity/period expansion
OP_ARITY = {
    "growth_pct": 2, "difference": 2, "cagr": 2, "ratio": 2, "margin": 2,
    "average": 0, "sum": 0, "ranking": 0, "count": 0, "hypothetical": 1,
    "lookup": 1,
}


@dataclass
class Fact:
    ticker: str
    year: int | None
    doc_type: str
    metric: str
    role: str = "value"        # value | numerator | denominator | base | end

    def key(self) -> tuple:
        return (self.ticker, self.year, self.doc_type, self.metric)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Plan:
    op: str = "lookup"
    facts: list[Fact] = field(default_factory=list)
    n_entities: int = 1
    n_periods: int = 1
    dimension: str = ""        # entity | year
    projection: str = ""       # value | year
    direction: str = ""        # max | min
    notes: list[str] = field(default_factory=list)

    @property
    def is_composite(self) -> bool:
        return self.op != "lookup" or len(self.facts) > 1

    def to_dict(self) -> dict:
        return {"op": self.op, "facts": [f.to_dict() for f in self.facts],
                "n_entities": self.n_entities, "n_periods": self.n_periods,
                "dimension": self.dimension, "projection": self.projection,
                "direction": self.direction,
                "notes": self.notes}


_RATIO_SPLIT = re.compile(r"\s+(?:tren|chia cho|so voi tong|tren tong)\s+")
_RATIO_LEAD = re.compile(r"^(?:ty le|ty suat|ty trong|bien loi nhuan|bien)\s+")


def split_ratio_metric(metric: str) -> tuple[str, str]:
    """"ty le loi nhuan sau thue tren doanh thu thuan"
        -> ("loi nhuan sau thue", "doanh thu thuan")

    Returns ("", "") when the phrase has no explicit denominator.
    """
    m = _RATIO_LEAD.sub("", (metric or "").strip())
    parts = _RATIO_SPLIT.split(m, maxsplit=1)
    if len(parts) != 2:
        return "", ""
    num, den = parts[0].strip(), parts[1].strip()
    if len(num) < 3 or len(den) < 3:
        return "", ""
    return num, den


def detect_op(question: str) -> str:
    q = norm(question)
    for name, rex in _OP_RE:
        if rex.search(q):
            return name
    return "lookup"


def _ranking_direction(question: str) -> str:
    text = norm(question)
    if any(value in text for value in (
        "nho nhat", "thap nhat", "it nhat", "be nhat", "toi thieu",
    )):
        return "min"
    return "max"


def build_plan(question: str, tickers: list[str], years: list[int],
               doc_type: str, metric: str,
               output_type: str = "number") -> Plan:
    """Cross entities x periods into concrete fact requirements."""
    op = detect_op(question)
    tickers = list(dict.fromkeys(tickers)) or [""]
    years = list(dict.fromkeys(years)) or [None]
    if (len(tickers) >= 2 and op not in {"ranking", "count"}
            and re.search(
                r"\btong (?:gia tri|chi phi|so du|du no|so tien|doanh thu)\b",
                norm(question),
            )):
        op = "sum"

    # growth/CAGR asked with a single stated year -> the prior year is implied
    if op in ("growth_pct", "cagr", "difference") and len(years) == 1 \
            and years[0] is not None and len(tickers) == 1:
        years = [years[0], years[0] - 1]

    facts: list[Fact] = []
    # ratio/margin need TWO DIFFERENT metrics ("A trên B"), not the same metric
    # twice: splitting here is what makes `ratio` solvable at all.
    num, den = split_ratio_metric(metric) if op in ("ratio", "margin") else (None, None)
    for t in tickers:
        for y in years:
            if num and den:
                facts.append(Fact(t, y, doc_type, num, role="numerator"))
                facts.append(Fact(t, y, doc_type, den, role="denominator"))
            else:
                facts.append(Fact(ticker=t, year=y, doc_type=doc_type, metric=metric))

    plan = Plan(op=op, facts=facts, n_entities=len(tickers), n_periods=len(years))
    if op == "ranking":
        plan.direction = _ranking_direction(question)
        if output_type == "year" and len(tickers) == 1 and len(years) >= 2:
            plan.dimension = "year"
            plan.projection = "year"
        elif len(tickers) >= 2:
            plan.dimension = "entity"
            plan.projection = "value"
    if op in ("ratio", "margin") and not (num and den):
        plan.op = "lookup"          # cannot split -> treat as a plain lookup
        plan.notes.append("ratio without a splittable 'A trên B' -> lookup")
    if op == "ranking" and not plan.dimension:
        plan.notes.append("ranking with <2 entities - entity list may be implicit")
    return plan


def evidence_budget(plan: Plan, base_k: int = 4, cap: int = 12) -> int:
    """Dynamic evidence count (P1.4): a flat k starves composite questions.

    lookup 1 entity/1 period  -> base_k
    each extra (entity, period) pair adds ~half of base_k, capped.
    """
    pairs = max(1, len(plan.facts))
    if pairs == 1:
        return base_k
    return int(min(cap, base_k + (pairs - 1) * max(2, base_k // 2)))


def per_fact_quota(plan: Plan, total: int) -> int:
    """At least one table per fact; spread the rest evenly."""
    return max(1, total // max(1, len(plan.facts)))
