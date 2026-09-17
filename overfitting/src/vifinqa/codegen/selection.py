"""Structured selection: the LLM PICKS cells, we WRITE the pandas.

WHY (audited on submission #12, Qwen 7B, 183 generated queries):
    35%  no column filter at all -> .iloc[0] grabs the wrong period
    15%  forgot to divide by ANSWER_SCALE -> raw VND for a "tỷ đồng" question
    90%  str.contains without regex=False -> parentheses become regex groups
    only 35% used the ['col'] == N condition the shortlist already handed them
Those 175 new answers moved the leaderboard by ~2 questions (~2% accuracy).

The shortlist ALREADY knows var/label/code/col/col_name/value/unit_scale — the
cell is located before the model is even called. Asking the model to re-derive
that addressing in pandas is where it fails. So instead it emits a tiny JSON:

    {"op": "difference", "operands": [1, 3]}
    {"op": "lookup", "operands": [2]}
    {"op": "growth_pct", "operands": [4, 5]}     # [end, base]

`operands` are 1-based indices into the rendered shortlist. We synthesise the
expression with the correct column filter and unit conversion, reusing the same
arithmetic as the deterministic engine. This makes three whole error classes
structurally impossible, and shrinks the generated text from ~256 tokens to ~30.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..finance.metrics import metric_keys, metric_uses_absolute_value
from ..retrieval.shortlist import candidate_matches_requirement
from ..utils.viet_text import strip_diacritics
from .units import check_answer_unit, percent_from_cell, cell_is_already_percent

# ops the synthesiser can execute from selected cells
SELECT_OPS = {"lookup", "sum", "average", "difference", "growth_pct",
              "ratio", "margin", "ratio_times", "ranking_max", "ranking_min",
              "percentage_point", "count"}

ARITY = {"lookup": 1, "difference": 2, "growth_pct": 2, "ratio": 2,
         "margin": 2, "ratio_times": 2, "percentage_point": 2}

_JSON_OBJ = re.compile(r"\{[^{}]*\}", re.S)


@dataclass
class Selection:
    op: str
    operands: list[int]          # 1-based indices into the shortlist
    note: str = ""

    def valid_for(self, n_candidates: int) -> str | None:
        if self.op not in SELECT_OPS:
            return f"unknown op {self.op!r}"
        if not self.operands:
            return "no operands"
        need = ARITY.get(self.op)
        if need and len(self.operands) < need:
            return f"op {self.op} needs {need} operands, got {len(self.operands)}"
        bad = [i for i in self.operands if not (1 <= i <= n_candidates)]
        if bad:
            return f"operand index out of range: {bad}"
        return None


def parse_selection(text: str) -> Selection | None:
    """Tolerant JSON extraction: models wrap output in prose or code fences."""
    if not text:
        return None
    blocks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S) or [text]
    for block in blocks:
        for m in [block] + _JSON_OBJ.findall(block):
            try:
                obj = json.loads(m.strip())
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(obj, dict):
                continue
            op = str(obj.get("op", "")).strip().lower()
            raw = obj.get("operands", obj.get("idx", obj.get("rows")))
            if isinstance(raw, (int, float)):
                raw = [raw]
            if not isinstance(raw, list):
                continue
            ops = []
            for x in raw:
                try:
                    ops.append(int(x))
                except (TypeError, ValueError):
                    continue
            if op and ops:
                return Selection(op=op, operands=ops,
                                 note=str(obj.get("note", ""))[:120])
    return None


def requirement_coverage(sel: Selection, candidates, requirements: list[dict]) -> dict:
    """Measure whether the selected cells prove every routed formula operand."""
    required = {
        str(req.get("requirement_id") or ""): req
        for req in requirements
        if req.get("requirement_id")
    }
    if not required:
        return {"required": 0, "covered": 0, "complete": True, "missing": []}
    picks = [candidates[i - 1] for i in sel.operands
             if 1 <= i <= len(candidates)]
    covered = set()
    for requirement_id, requirement in required.items():
        if any(_candidate_proves_requirement(candidate, requirement)
               for candidate in picks):
            covered.add(requirement_id)
    missing = sorted(set(required) - covered)
    return {
        "required": len(required), "covered": len(covered),
        "complete": not missing, "missing": missing,
    }


def selection_matches_route(sel: Selection, route: dict,
                            requirements: list[dict]) -> bool:
    """Whether one selection expression can represent the routed calculation."""
    plan_op = str((route.get("plan") or {}).get("op") or "lookup")
    output_type = str(route.get("output_type") or "number")
    question = _plain(route.get("question", ""))
    unique_operands = len(set(sel.operands))
    required = {
        str(req.get("requirement_id") or ""): req
        for req in requirements if req.get("requirement_id")
    }
    n_required = len(required)

    if plan_op == "ranking":
        if _looks_nested_selector(question):
            return False
        metrics = {str(req.get("metric_key") or "") for req in required.values()}
        return (
            sel.op in {"ranking_max", "ranking_min"}
            and (not n_required or len(metrics) == 1)
            and (not n_required or unique_operands == n_required)
        )
    if plan_op == "average":
        return sel.op == "average" and (not n_required or unique_operands == n_required)
    if plan_op == "growth_pct":
        return sel.op == "growth_pct" and unique_operands == 2
    if plan_op == "difference":
        if output_type == "count":
            return False
        return sel.op in {"difference", "percentage_point"} and unique_operands == 2
    if plan_op in {"ratio", "margin", "ratio_times"}:
        return sel.op in {"ratio", "margin", "ratio_times"} and unique_operands == 2
    if plan_op == "count" or output_type == "count":
        # Selection's count expression only proves that picked cells exist; it
        # cannot replay thresholds or prove why non-picked entities failed.
        return False
    if plan_op == "lookup":
        if n_required == 1:
            return sel.op == "lookup" and unique_operands == 1
        if n_required == 2 and output_type in {"percent", "ratio"}:
            return sel.op in {"ratio", "margin", "ratio_times"} and unique_operands == 2
        return not n_required and sel.op == "lookup" and unique_operands == 1
    return False


def _looks_nested_selector(text: str) -> bool:
    entities = r"(?:cong ty|doanh nghiep|ngan hang|ma|to chuc)"
    return bool(
        re.search(rf"\b(?:cua|cho|tai)\s+{entities}\s+co(?!\s+phan\b)\b", text)
        or re.search(rf"\b{entities}\s+co(?!\s+phan\b)\b", text)
        or re.search(r"\bnam\s+(?:ma|ghi nhan)\b", text)
        or re.search(r"\b(?:tai|vao)\s+nam\s+co\b", text)
    )


def _plain(text: str) -> str:
    return re.sub(r"\s+", " ", strip_diacritics(str(text or "")).lower()
                  .replace("đ", "d")).strip()


def _candidate_proves_requirement(candidate, requirement: dict) -> bool:
    ticker = str(requirement.get("ticker") or "").upper()
    report_id = str(getattr(candidate, "report_id", ""))
    parts = report_id.split("_")
    if ticker and (not parts or parts[0].upper() != ticker):
        return False

    doc_type = str(requirement.get("doc_type") or "")
    if doc_type and report_id.endswith(("_consolidated", "_separate")):
        if not report_id.endswith(f"_{doc_type}"):
            return False

    if not candidate_matches_requirement(candidate, requirement):
        return False

    year = requirement.get("year")
    if year is None:
        return True
    year = int(year)
    named_years = {int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)",
                                                       str(candidate.col_name))}
    if named_years:
        return year in named_years
    report_year = next((int(part) for part in parts[1:] if re.fullmatch(r"20\d{2}", part)),
                       None)
    return report_year in {year, year + 1}


def synthesize(sel: Selection, candidates, route: dict):
    """(answer, pandas_query, error). candidates = shortlist Candidate objects."""
    err = sel.valid_for(len(candidates))
    if err:
        return None, "", err
    picks = [candidates[i - 1] for i in sel.operands]
    q_scale = float(route.get("unit_scale", 1.0) or 1.0)
    output_type = route.get("output_type", "number")

    def vnd(c):
        value = float(c.value) * float(c.unit_scale)
        keys = metric_keys([c.label], expand_derived=False)
        return abs(value) if metric_uses_absolute_value(c.label, keys) else value

    def expr(c):
        # The shortlist already selected one concrete row; preserve that exact
        # identity instead of reintroducing a parent/child substring collision.
        label = re.sub(r"\s+", " ", str(c.label)).strip()
        return (f"float({c.var}.loc[({c.var}['row'] == {c.row}) "
                f"& {c.var}['label'].str.strip().eq({label!r}) "
                f"& ({c.var}['col'] == {c.col}), "
                f"'value'].iloc[0])")

    def expr_vnd(c):
        value_expr = f"{expr(c)} * {float(c.unit_scale):g}"
        keys = metric_keys([c.label], expand_derived=False)
        return (f"abs({value_expr})"
                if metric_uses_absolute_value(c.label, keys) else value_expr)

    try:
        answer, query = _apply(sel.op, picks, q_scale, output_type,
                               vnd, expr, expr_vnd, route)
    except ZeroDivisionError:
        return None, "", "division by zero"
    except (ValueError, IndexError) as e:
        return None, "", f"{type(e).__name__}: {e}"
    if answer is None:
        return None, "", f"op {sel.op} not synthesisable"
    if answer != answer or abs(answer) == float("inf"):
        return None, "", "non-finite answer"
    return round(float(answer), 2), query, None


def _apply(op, picks, q_scale, output_type, vnd, expr, expr_vnd, route):
    if op == "lookup":
        c = picks[0]
        if output_type == "percent":
            ans = percent_from_cell(c.value, c.label, c.col_name)
            mul = "" if cell_is_already_percent(c.label, c.col_name, c.value) else " * 100"
            return ans, f"round({expr(c)}{mul}, 2)"
        return vnd(c) / q_scale, f"round({expr_vnd(c)} / {q_scale:g}, 2)"

    if op == "difference":
        a, b = picks[0], picks[1]
        if output_type in ("percent", "percentage_point"):
            return (vnd(a) - vnd(b)), f"round({expr_vnd(a)} - {expr_vnd(b)}, 2)"
        return ((vnd(a) - vnd(b)) / q_scale,
                f"round(({expr_vnd(a)} - {expr_vnd(b)}) / {q_scale:g}, 2)")

    if op == "percentage_point":
        a, b = picks[0], picks[1]
        return (float(a.value) - float(b.value),
                f"round({expr(a)} - {expr(b)}, 2)")

    if op == "growth_pct":
        end, base = picks[0], picks[1]
        if vnd(base) == 0:
            raise ZeroDivisionError
        return ((vnd(end) - vnd(base)) / abs(vnd(base)) * 100.0,
                f"round(({expr_vnd(end)} - {expr_vnd(base)}) "
                f"/ abs({expr_vnd(base)}) * 100, 2)")

    if op in ("ratio", "margin", "ratio_times"):
        num, den = picks[0], picks[1]
        if vnd(den) == 0:
            raise ZeroDivisionError
        if op == "ratio_times" or output_type == "ratio":
            return vnd(num) / vnd(den), f"round({expr_vnd(num)} / {expr_vnd(den)}, 2)"
        return (vnd(num) / vnd(den) * 100.0,
                f"round({expr_vnd(num)} / {expr_vnd(den)} * 100, 2)")

    if op in ("sum", "average", "ranking_max", "ranking_min"):
        vals = [vnd(c) for c in picks]
        exprs = [expr_vnd(c) for c in picks]
        if op == "sum":
            return sum(vals) / q_scale, f"round(({' + '.join(exprs)}) / {q_scale:g}, 2)"
        if op == "average":
            return (sum(vals) / len(vals) / q_scale,
                    f"round(({' + '.join(exprs)}) / {len(exprs)} / {q_scale:g}, 2)")
        fn = "max" if op == "ranking_max" else "min"
        chosen = max(vals) if op == "ranking_max" else min(vals)
        return chosen / q_scale, f"round({fn}({', '.join(exprs)}) / {q_scale:g}, 2)"

    if op == "count":
        exprs = [expr(c) for c in picks]
        # Make the count dataframe-grounded for the semantic guard: each picked
        # proof cell contributes one boolean term when it exists and is finite.
        terms = [f"({e} == {e})" for e in exprs]
        return float(len(picks)), f"round(float({' + '.join(terms)}), 2)"

    return None, ""


def confidence(sel: Selection, candidates, answer: float, route: dict) -> float:
    """Confidence for arbitration: driven by the picked cells' match scores."""
    picks = [candidates[i - 1] for i in sel.operands
             if 1 <= i <= len(candidates)]
    if not picks:
        return 0.0
    base = min(float(c.score) for c in picks)
    if check_answer_unit(answer, route.get("output_type", "number")):
        base -= 20.0
    return max(0.0, min(95.0, base))
