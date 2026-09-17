"""Replaceable interface for the post-builder Hard-question critic.

The critic sees only the locked ``PublicSpec`` and the rendered question. It does not see raw
tables, runtime values, winners, or answers, and therefore cannot change candidate semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Protocol

from vifinqa.generation.hard.recipe.planner import PublicSpec

QuestionQualityDecision = Literal["accept", "rewrite"]


@dataclass(frozen=True, slots=True)
class QuestionQualityAssessment:
    candidate_id: str
    decision: QuestionQualityDecision
    feedback: str


class QuestionCritic(Protocol):
    def critique(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        questions: Mapping[str, str],
        feedback: str | None = None,
    ) -> tuple[QuestionQualityAssessment, ...]:
        """Assess every spec exactly once; malformed/incomplete responses must fail closed."""
        ...


class QuestionQualityResponseError(ValueError):
    """Critic response violates the complete, unique candidate-id contract."""


class FakeQuestionCritic:
    """Deterministic test double; never used as a production fallback."""

    def __init__(
        self,
        decisions: Mapping[str, QuestionQualityDecision] | None = None,
        *,
        rewrite_feedback: str = "Rewrite the question to be more concise and natural.",
    ) -> None:
        self._decisions = dict(decisions or {})
        self._rewrite_feedback = rewrite_feedback

    def critique(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        questions: Mapping[str, str],
        feedback: str | None = None,
    ) -> tuple[QuestionQualityAssessment, ...]:
        del questions, feedback
        return tuple(
            QuestionQualityAssessment(
                candidate_id=spec.candidate_id,
                decision=self._decisions.get(spec.candidate_id, "accept"),
                feedback=(
                    self._rewrite_feedback
                    if self._decisions.get(spec.candidate_id) == "rewrite"
                    else "The question is concise, natural, and has a clear financial purpose."
                ),
            )
            for spec in specs
        )
