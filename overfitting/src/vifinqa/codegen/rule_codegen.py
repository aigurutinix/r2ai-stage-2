"""Deterministic rule-based Text-to-Pandas fallback / fast path.

Good for single-fact lookups (the vast majority of ViFinQA questions):
fuzzy-match the metric phrase against row labels of the retrieved tables,
pick the year column, emit a reproducible one-line pandas expression.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..finance.metrics import metric_uses_absolute_value
from ..utils.viet_text import label_metric_score, norm
from ..retrieval.serialize import df_roundtrip
from ..retrieval.shortlist import build_shortlist
from .units import percent_from_cell, cell_is_already_percent, check_answer_unit


@dataclass
class RuleAnswer:
    ok: bool
    answer: float = 0.0
    pandas_query: str = ""
    var: str = ""
    confidence: float = 0.0
    detail: str = ""


def _year_col_score(col_name: str, year: int) -> int:
    cn = str(col_name)
    if re.search(rf"31\s*/\s*12\s*/\s*{year}", cn):
        return 3
    if str(year) in cn:
        return 2
    if norm(cn) in ("nam nay", "cuoi nam") and year:  # weak
        return 1
    return 0


AMBIGUOUS_MARGIN = 8.0     # best-vs-runner-up gap below which we distrust the pick


def try_rule_answer(route: dict, tables: list[dict], min_label: float = 62.0,
                    unambiguous_margin: float = 0.0) -> RuleAnswer:
    """tables: [{var, report_id, table_pos, report_year, csv_text, unit_scale}]

    Threshold history: the original 78.0 was mis-calibrated. Measured on the
    real corpus, CORRECT rows commonly score 62-64 against the extracted metric
    phrase (e.g. "tra truoc cho nguoi ban" vs "2. Trả trước cho người bán ngắn
    hạn"), so 78 silently rejected them.

    Calibration on the offline eval suite (lookup class, coverage x accuracy):
        threshold 78, no margin   -> 0.680
        threshold 62, no margin   -> 0.720   <-- default
        threshold 62, margin 8    -> 0.440
        threshold 62, margin 15   -> 0.320
    So the rule now ANSWERS whenever it clears 62 (refusing costs more than the
    occasional wrong pick). Ambiguity is instead expressed through `confidence`:
    a near-tie is capped below the `--rule-first` cut-off, so those questions
    still go to the LLM, which can overwrite them using the shortlist.
    Pass `unambiguous_margin > 0` only to force refusal (used for ablations).
    """
    metric = route.get("metric_norm", "")
    variants = [v for v in (route.get("metric_variants") or [metric]) if v]
    years = route.get("years") or []
    q_scale = float(route.get("unit_scale", 1.0))
    output_type = route.get("output_type") or (
        "percent" if route.get("is_percent") else "number"
    )
    plan_op = (route.get("plan") or {}).get("op", "lookup")
    if output_type not in ("number", "percent") or route.get("growth") or not variants:
        return RuleAnswer(ok=False, detail="needs multi-step reasoning -> LLM")
    if plan_op != "lookup":
        return RuleAnswer(ok=False, detail=f"composite op={plan_op} -> LLM")
    # ONE scoring path. This used to re-implement its own label/column scoring,
    # which disagreed with build_shortlist: a row the shortlist rated 78 (thanks
    # to year + VAS-code evidence) could score ~56 here and be refused, leaving
    # the question empty. Reusing the shortlist removes that whole class of
    # silent misses and keeps rule + prompt looking at the same candidates.
    cands = build_shortlist(
        tables, variants, years, top_n=6, min_score=min_label,
        question=route.get("question", ""))
    # magnitude sanity: a money figure asked in triệu/tỷ đồng is essentially
    # never < 1e6 VND -> such a candidate points at a wrong-unit table
    if q_scale >= 1e6:
        keep = [c for c in cands if abs(c.value * c.unit_scale) >= 1e6]
        cands = keep or cands
    if not cands:
        return RuleAnswer(ok=False, detail="no confident label/column match")

    best_c = cands[0]
    runner_up = cands[1].score if len(cands) > 1 else 0.0
    best = (best_c.score, best_c.var, best_c.row, best_c.label, best_c.col,
            best_c.value, best_c.unit_scale, best_c.lexical, best_c.col_name)

    _score, var, row, label, col, value, us, s_lab = best[:8]
    col_name = best[8] if len(best) > 8 else ""
    gap = _score - runner_up if runner_up > 0 else _score
    ambiguous = gap < AMBIGUOUS_MARGIN
    if unambiguous_margin and gap < unambiguous_margin:
        return RuleAnswer(ok=False, confidence=_score,
                          detail=f"ambiguous: best {_score:.0f} vs runner-up "
                                 f"{runner_up:.0f} -> LLM picks from shortlist")

    if output_type == "percent":
        # ORGANIZER-CONFIRMED: percent answers are 90, not 0.9
        answer = round(percent_from_cell(value, label, col_name), 2)
        scale_expr = ("" if cell_is_already_percent(label, col_name, value)
                      else " * 100")
        cell = (f"float({var}.loc[({var}['row'] == {row}) "
                f"& {var}['label'].str.strip().eq({label!r}) "
                f"& ({var}['col'] == {col}), 'value'].iloc[0])")
        query = f"round({cell}{scale_expr}, 2)"
    else:
        absolute = metric_uses_absolute_value(
            " ".join((route.get("question", ""), metric)),
            route.get("metric_keys") or ())
        raw_value = abs(value) if absolute else value
        answer = round(raw_value * us / q_scale, 2)
        cell = (f"float({var}.loc[({var}['row'] == {row}) "
                f"& {var}['label'].str.strip().eq({label!r}) "
                f"& ({var}['col'] == {col}), 'value'].iloc[0])")
        if absolute:
            cell = f"abs({cell})"
        query = f"round({cell} * {us:g} / {q_scale:g}, 2)"
    warn = check_answer_unit(answer, output_type)
    # a near-tie stays answerable but must NOT be trusted enough for --rule-first
    # to skip the LLM (the shortlist gives the model a real chance to do better)
    conf = min(99.0, _score)
    if ambiguous:
        conf = min(conf, 60.0)
    return RuleAnswer(ok=True, answer=answer, pandas_query=query, var=var,
                      confidence=conf,
                      detail=f"label='{label}' fuzz={s_lab:.0f} gap={gap:.0f}"
                             + (" AMBIGUOUS" if ambiguous else "")
                             + (f" | UNIT-WARN: {warn}" if warn else ""))
