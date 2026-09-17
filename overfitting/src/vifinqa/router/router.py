"""Structured lookup router: lock (ticker, year, doc_type) -> report_ids."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

from ..extraction.build_store import Store
from .entities import StockMap, parse_question, Parsed
from .decompose import build_plan, evidence_budget
from .evidence import build_evidence_requirements


@dataclass
class Route:
    qid: int
    question: str
    tickers: list[str] = field(default_factory=list)
    years: list[int] = field(default_factory=list)
    doc_type: str = "consolidated"
    unit_scale: float = 1.0
    unit_name: str = "đồng"
    is_percent: bool = False
    output_type: str = "number"
    growth: bool = False
    metric_norm: str = ""
    metric_variants: list[str] = field(default_factory=list)
    metric_keys: list[str] = field(default_factory=list)
    metric_qualifiers: dict[str, str] = field(default_factory=dict)
    plan: dict = field(default_factory=dict)     # decompose.Plan.to_dict()
    evidence_requirements: list[dict] = field(default_factory=list)
    evidence_budget: int = 0                     # dynamic k for this question
    report_ids: list[str] = field(default_factory=list)
    confidence: str = "high"           # high | medium | low
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def route_question(qid: int, question: str, stock: StockMap, store: Store) -> Route:
    p: Parsed = parse_question(question, stock)
    r = Route(qid=qid, question=question, tickers=p.tickers, years=list(p.years),
              doc_type=p.doc_type, unit_scale=p.unit_scale, unit_name=p.unit_name,
              is_percent=p.is_percent, output_type=p.output_type,
              growth=p.growth, metric_norm=p.metric_norm,
              metric_variants=list(p.metric_variants or [p.metric_norm]),
              metric_keys=list(p.metric_keys),
              metric_qualifiers=dict(p.metric_qualifiers))

    if not p.tickers:
        r.confidence = "low"
        r.notes.append("no ticker found")
        return r
    if p.ticker_source == "low_conf_fuzzy":
        r.confidence = "low"
        r.notes.append(f"fuzzy ticker score={p.ticker_score:.0f}")
    elif p.ticker_source == "fuzzy":
        r.confidence = "medium"

    # years: default = most recent available; growth -> include previous year
    if not r.years:
        avail = store.years_of(r.tickers[0])
        if avail:
            r.years = [avail[-1]]
            r.notes.append("no year in question -> latest available")
            r.confidence = "low"
    # A prior year is implied ONLY when a single company is compared over time.
    # "Doanh thu của A chênh lệch bao nhiêu so với B" (2 tickers, 1 year) must
    # NOT gain a second year: that produced 4 facts instead of 2 and made every
    # `difference` question unresolvable.
    if r.growth and len(r.years) == 1 and len(r.tickers) <= 1:
        r.years.append(r.years[0] - 1)
        r.notes.append("growth question -> added prior year")

    for t in r.tickers:
        for y in r.years:
            exact = store.find_reports(t, y, r.doc_type, allow_fallback=False)
            matches = exact or store.find_reports(t, y, r.doc_type)
            if not matches:
                r.notes.append(f"missing report {t}/{y}/{r.doc_type}")
                # try adjacent year (a FY-{y} figure often sits in the {y+1} report's
                # prior-year column)
                alt = store.find_reports(t, y + 1, r.doc_type)
                if alt:
                    r.report_ids.extend(alt)
                    r.notes.append(f"fallback to {t}/{y+1} (prior-year column)")
                continue
            r.report_ids.extend(matches)
            if not exact:
                r.notes.append(f"doc_type fallback for {t}/{y}")
    # dedupe, keep order
    seen = set()
    r.report_ids = [x for x in r.report_ids if not (x in seen or seen.add(x))]
    if not r.report_ids:
        r.confidence = "low"
        r.notes.append("no report locked")

    # --- decomposition + dynamic evidence budget (P1.1 / P1.4) -------------
    plan = build_plan(
        r.question, r.tickers, r.years, r.doc_type, r.metric_norm,
        output_type=r.output_type,
    )
    r.plan = plan.to_dict()
    r.evidence_budget = evidence_budget(plan)
    r.evidence_requirements = [
        requirement.to_dict()
        for requirement in build_evidence_requirements(
            r.question,
            r.tickers,
            r.years,
            r.doc_type,
            metric_variants=r.metric_variants,
            plan=r.plan,
        )
    ]
    # Some typed formulas require an adjacent fiscal period even when that year
    # is not written explicitly (for example average opening/closing inventory).
    # Lock those reports from the canonical evidence contract before retrieval.
    for requirement in r.evidence_requirements:
        ticker = str(requirement.get("ticker") or "")
        year = requirement.get("year")
        if not ticker or year is None:
            continue
        for report_id in store.find_reports(ticker, int(year), r.doc_type):
            if report_id not in r.report_ids:
                r.report_ids.append(report_id)
                r.notes.append(f"formula operand report {ticker}/{year}")
    if r.evidence_requirements:
        r.evidence_budget = max(
            r.evidence_budget,
            min(20, len(r.evidence_requirements)),
        )
    if plan.is_composite:
        r.notes.append(f"composite op={plan.op} facts={len(plan.facts)} "
                       f"budget={r.evidence_budget}")
        # a composite question needs the reports of EVERY (entity, period) pair;
        # blanket expansion failed on the leaderboard, targeted expansion is the
        # supported alternative (see P1_STRATEGY_REVIEW.md §2)
        for f in plan.facts:
            if not f.ticker or f.year is None:
                continue
            for rid in store.find_reports(f.ticker, f.year, f.doc_type):
                if rid not in r.report_ids:
                    r.report_ids.append(rid)
    return r
