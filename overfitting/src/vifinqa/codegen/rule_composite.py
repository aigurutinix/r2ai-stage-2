"""Deterministic solver for COMPOSITE questions (growth / difference / ratio /
margin / ranking / sum / average).

Why this exists: composite questions are ~50% of the test set and scored 0.000
with a lookup-only rule engine, while the 7B LLM converted only ~26% of the
questions it saw. Once `fact_resolver` has located each operand, combining them
is pure arithmetic — deterministic, unit-correct and crash-free.

Every emitted `pandas_query` is ONE expression (the grader evaluates it), and
every answer is already in the unit the question asks for (percent -> 90, not
0.9), matching the organizers' confirmation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .fact_resolver import ResolvedFact, resolve_all
from .units import check_answer_unit

# ops this module handles; everything else stays with the lookup rule / LLM
SUPPORTED = {"growth_pct", "difference", "ratio", "margin", "ratio_times",
             "sum", "average", "ranking", "cagr"}


@dataclass
class CompositeAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    confidence: float = 0.0
    detail: str = ""
    resolved: list = None

    def vars_used(self) -> list[str]:
        return sorted({r.var for r in (self.resolved or [])})


def try_composite_answer(route: dict, tables: list[dict], encoder=None,
                         min_score: float = 62.0) -> CompositeAnswer:
    plan = route.get("plan") or {}
    op = plan.get("op", "lookup")
    if op not in SUPPORTED:
        return CompositeAnswer(ok=False, detail=f"op={op} not handled here")

    facts = [_FactView(f) for f in plan.get("facts", [])]
    if len(facts) < 2 and op not in ("sum", "average", "ranking"):
        return CompositeAnswer(ok=False, detail=f"op={op} needs >=2 facts")

    variants = route.get("metric_variants") or [route.get("metric_norm", "")]
    # a fact carrying its own metric (ratio numerator/denominator) must be matched
    # against THAT metric, not against the whole question's phrase
    resolved, conf = _resolve_with_own_metrics(
        facts, tables, variants, encoder, min_score,
        question=route.get("question", ""))
    if len(resolved) != len(facts) or conf <= 0:
        return CompositeAnswer(ok=False,
                               detail=f"resolved {len(resolved)}/{len(facts)} facts")

    q_scale = float(route.get("unit_scale", 1.0))
    output_type = route.get("output_type", "number")

    try:
        answer, query = _combine(op, resolved, q_scale, output_type, route)
    except (ZeroDivisionError, ValueError, IndexError) as e:
        return CompositeAnswer(ok=False, detail=f"{type(e).__name__}: {e}")
    if answer is None:
        return CompositeAnswer(ok=False, detail=f"op={op} could not be combined")

    warn = check_answer_unit(answer, output_type)
    if warn and "outside plausible range" in warn:
        conf = min(conf, 40.0)
    return CompositeAnswer(ok=True, answer=round(float(answer), 2),
                           pandas_query=query, confidence=conf,
                           detail=f"op={op} facts={len(resolved)}"
                                  + (f" | UNIT-WARN: {warn}" if warn else ""),
                           resolved=resolved)


def _resolve_with_own_metrics(facts, tables, variants, encoder, min_score,
                              question=""):
    """Resolve each fact against its own metric when it has a specific one."""
    from .fact_resolver import resolve_fact
    out = []
    for f in facts:
        own = [f.metric] if f.role in ("numerator", "denominator") and f.metric else variants
        r = resolve_fact(f, tables, own, encoder, min_score,
                         question=question)
        if r is None:
            return out, 0.0
        out.append(r)
    if not out:
        return out, 0.0
    weakest = min(r.score for r in out)
    keys = {(r.report_id, r.table_pos, r.label, r.col) for r in out}
    conf = min(99.0, weakest)
    if len(keys) < len(out):
        conf = min(conf, 35.0)
    return out, max(0.0, conf)


class _FactView:
    """dict -> attribute access expected by fact_resolver."""

    __slots__ = ("ticker", "year", "doc_type", "metric", "role")

    def __init__(self, d: dict):
        self.ticker = d.get("ticker", "")
        self.year = d.get("year")
        self.doc_type = d.get("doc_type", "consolidated")
        self.metric = d.get("metric", "")
        self.role = d.get("role", "value")


def _combine(op: str, rs: list[ResolvedFact], q_scale: float,
             output_type: str, route: dict):
    """Return (answer, single-expression pandas query)."""
    if op in ("growth_pct", "cagr"):
        end, base = _order_by_year(rs, newest_first=True)
        if base.value_vnd == 0:
            raise ZeroDivisionError("growth base is zero")
        if op == "growth_pct":
            asks_decrease = _asks_decrease_magnitude(route.get("question", ""))
            first, last = (base, end) if asks_decrease else (end, base)
            ans = (first.value_vnd - last.value_vnd) / abs(base.value_vnd) * 100.0
            q = (f"round(({first.expr_vnd()} - {last.expr_vnd()}) "
                 f"/ abs({base.expr_vnd()}) * 100, 2)")
        else:
            n = abs((end.year or 0) - (base.year or 0)) or 1
            if base.value_vnd <= 0:
                raise ValueError("CAGR needs a positive base")
            ans = ((end.value_vnd / base.value_vnd) ** (1.0 / n) - 1.0) * 100.0
            q = (f"round((({end.expr_vnd()} / {base.expr_vnd()}) ** (1/{n}) - 1) "
                 f"* 100, 2)")
        return ans, q

    if op == "difference":
        a, b = _order_for_difference(rs, route)
        delta = a.value_vnd - b.value_vnd
        delta_expr = f"({a.expr_vnd()} - {b.expr_vnd()})"
        if _uses_absolute_gap(route.get("question", "")):
            delta = abs(delta)
            delta_expr = f"abs({delta_expr})"
        if output_type == "percent":
            ans = delta * 100.0
            q = f"round({delta_expr} * 100, 2)"
        else:
            ans = delta / q_scale
            q = f"round({delta_expr} / {q_scale:g}, 2)"
        return ans, q

    if op in ("ratio", "margin", "ratio_times"):
        num, den = rs[0], rs[1]
        if den.value_vnd == 0:
            raise ZeroDivisionError("ratio denominator is zero")
        if op == "ratio_times" or output_type == "ratio":
            ans = num.value_vnd / den.value_vnd
            q = f"round({num.expr_vnd()} / {den.expr_vnd()}, 2)"
        else:
            ans = num.value_vnd / den.value_vnd * 100.0
            q = f"round({num.expr_vnd()} / {den.expr_vnd()} * 100, 2)"
        return ans, q

    if op in ("sum", "average", "ranking"):
        vals = [r.value_vnd for r in rs]
        exprs = [r.expr_vnd() for r in rs]
        if op == "sum":
            ans, inner = sum(vals) / q_scale, " + ".join(exprs)
            return ans, f"round(({inner}) / {q_scale:g}, 2)"
        if op == "average":
            ans = (sum(vals) / len(vals)) / q_scale
            return ans, f"round(({' + '.join(exprs)}) / {len(exprs)} / {q_scale:g}, 2)"
        # ranking: return the extreme value the question asks for
        want_min = _wants_min(route.get("question", ""))
        fn = "min" if want_min else "max"
        ans = (min(vals) if want_min else max(vals)) / q_scale
        return ans, f"round({fn}({', '.join(exprs)}) / {q_scale:g}, 2)"

    return None, ""


def _order_by_year(rs: list[ResolvedFact], newest_first: bool):
    known = [r for r in rs if r.year is not None]
    if len(known) >= 2:
        known.sort(key=lambda r: r.year, reverse=newest_first)
        return known[0], known[1]
    return rs[0], rs[1]


def _order_for_difference(rs: list[ResolvedFact], route: dict):
    """"A chênh lệch bao nhiêu so với B" -> A - B.

    Facts are generated ticker-major in decompose.build_plan, so the first
    mentioned entity is rs[0]. When both facts share a ticker (two periods), the
    later year is the subject.
    """
    if len({r.ticker for r in rs}) == 1 and all(r.year is not None for r in rs[:2]):
        return _order_by_year(rs, newest_first=True)
    return rs[0], rs[1]


_MIN_WORDS = ("nho nhat", "thap nhat", "it nhat", "thap hon ca")


def _asks_decrease_magnitude(question: str) -> bool:
    from ..utils.viet_text import norm
    q = norm(question)
    return bool(re.search(r"\bgiam\s+bao nhieu\s*(?:%|phan tram\b)", q))


def _uses_absolute_gap(question: str) -> bool:
    from ..utils.viet_text import norm
    q = norm(question)
    directional_change = any(marker in q for marker in (
        "muc thay doi", "thay doi tu", "bien dong tu", "tang tu", "giam tu",
    ))
    return "chenh lech" in q and not directional_change


def _wants_min(question: str) -> bool:
    from ..utils.viet_text import norm
    q = norm(question)
    return any(w in q for w in _MIN_WORDS)
