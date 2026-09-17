"""Arbitration between the deterministic rule engine and the LLM.

Measured context: on the leaderboard the rule engine BEAT the 7B LLM on every
axis (#8 rule-only ANSWER .1285 / EXEC .1285 vs #6 Qwen .1047 / .0613), and the
LLM produced usable code for only ~26% of the questions it saw. So the LLM must
be treated as a *second opinion*, not as an automatic override.

Policy (deliberately simple and auditable):
  1. Only one side answered      -> take it.
  2. Both agree (within tol)     -> take it, confidence boosted (independent
                                    derivations agreeing is real evidence).
  3. Disagree, rule confident    -> keep the rule (it is grounded by construction
                                    and its query never crashes).
  4. Disagree, rule unconfident  -> take the LLM (the rule is guessing).
Every decision is recorded in `arbitration` so a submission can be audited.
"""
from __future__ import annotations

from dataclasses import dataclass

# a rule answer at/above this confidence is not overridden by a disagreeing LLM
RULE_TRUST = 78.0
# relative tolerance for calling two answers "the same"
AGREE_REL_TOL = 1e-4
AGREE_ABS_TOL = 0.011


@dataclass
class Verdict:
    answer: float
    pandas_query: str
    source: str            # rule | rule_composite | llm | llm_debug | agree
    confidence: float
    reason: str
    used: str              # "rule" | "llm"


def agree(a: float, b: float) -> bool:
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if a != a or b != b:
        return False
    return abs(a - b) <= max(AGREE_ABS_TOL, AGREE_REL_TOL * max(abs(a), abs(b)))


def arbitrate(rule: dict | None, llm: dict | None,
              rule_trust: float = RULE_TRUST) -> Verdict | None:
    """rule/llm: {"answer", "pandas_query", "confidence", "source"} or None."""
    if rule is None and llm is None:
        return None
    if llm is None:
        return Verdict(rule["answer"], rule["pandas_query"], rule.get("source", "rule"),
                       rule.get("confidence", 0.0), "llm produced nothing", "rule")
    if rule is None:
        return Verdict(llm["answer"], llm["pandas_query"], llm.get("source", "llm"),
                       llm.get("confidence", 50.0), "rule produced nothing", "llm")

    if agree(rule["answer"], llm["answer"]):
        # keep the rule's query: it is a single expression by construction
        return Verdict(rule["answer"], rule["pandas_query"],
                       rule.get("source", "rule"),
                       min(99.0, max(rule.get("confidence", 0.0), 85.0)),
                       "rule and llm agree", "rule")

    r_conf = float(rule.get("confidence", 0.0))
    if r_conf >= rule_trust:
        return Verdict(rule["answer"], rule["pandas_query"], rule.get("source", "rule"),
                       r_conf, f"disagree; rule confident ({r_conf:.0f})", "rule")
    return Verdict(llm["answer"], llm["pandas_query"], llm.get("source", "llm"),
                   float(llm.get("confidence", 50.0)),
                   f"disagree; rule weak ({r_conf:.0f}) -> llm", "llm")


def summarize(records) -> dict:
    """Counter over the `arbitration.reason` field of finished records."""
    from collections import Counter
    return dict(Counter((r.get("arbitration") or {}).get("reason", "n/a")
                        for r in records))
