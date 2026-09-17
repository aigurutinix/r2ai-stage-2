
from __future__ import annotations

import json
import logging
import math
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

from vifinqa.config import Settings
from vifinqa.common.corpus.catalog import scan_catalog
from vifinqa.common.corpus.company_meta import load_company_meta
from vifinqa.generation.budget import GenerationBudgetExceeded, LLMCallBudget
from vifinqa.generation.hard.recipe import operations  # noqa: F401 — side-effect: register operations
from vifinqa.generation.hard.recipe.audit.base import (
    DependencyAuditor,
    DependencyReviewItem,
    DependencyReviewResult,
)
from vifinqa.generation.hard.recipe.audit.cache import AuditCache
from vifinqa.generation.hard.recipe.audit.llm import LLMDependencyAuditor
from vifinqa.generation.hard.recipe.audit.service import (
    AuditStats,
    DependencyAuditRejected,
)
from vifinqa.generation.hard.recipe.archetype import reasoning_archetype
from vifinqa.generation.hard.recipe.base import ReasoningGraph
from vifinqa.generation.hard.recipe.base import CandidateRejected
from vifinqa.generation.hard.recipe.evaluator import EvaluationError
from vifinqa.generation.hard.recipe.gates import (
    compute_hardness,
    deep_hardness_gate_error,
)
from vifinqa.generation.hard.recipe.grounded.analytical_planner import (
    AnalyticalCandidateDraft,
    ENABLED_ANALYTICAL_FRAME_IDS,
    ENABLED_TEMPLATE_FRAME_IDS,
    finalize_analytical_draft,
    iter_analytical_frame_drafts,
)
from vifinqa.generation.hard.recipe.planner import PublicSpec, RecipeCandidate
from vifinqa.generation.hard.recipe.question import (
    MAX_QUESTION_BATCH_SIZE,
    LLMQuestionBuilder,
    QuestionBuilder,
    question_answer_leak_error,
    question_naturalness_error,
    question_round_trip_error,
)
from vifinqa.generation.hard.recipe.question_quality import (
    LLMQuestionCritic,
    QuestionCritic,
    QuestionQualityAssessment,
    QuestionQualityResponseError,
)
from vifinqa.generation.hard.recipe.semantic_auditor import (
    MAX_SEMANTIC_AUDIT_BATCH_SIZE,
    LLMSemanticAuditor,
    SemanticAssessment,
    SemanticAuditor,
    SemanticAuditResponseError,
)
from vifinqa.generation.hard.recipe.signature import operator_signature
from vifinqa.generation.hard.deep_shard import (
    _normalized_question as _deep_normalized_question,
    _query_fingerprint as _deep_query_fingerprint,
    _sha256 as _deep_sha256,
)
from vifinqa.generation.hard.template_intents import INTENTS_BY_ID
from vifinqa.generation.schemas import QARecord
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.output.writer import JsonlWriter
from vifinqa.generation.panel.store import JsonCubeStore

logger = logging.getLogger(__name__)

INVENTORY_PER_RECIPE_ATTEMPTS = (
    400
)
DEFAULT_SAMPLE_COUNT = 21
MAX_PER_OPERATOR_SIGNATURE = 3
MAX_PER_STORY_FAMILY = 3
MAX_PER_FRAME = 2
MAX_SEMANTIC_AUDIT_ATTEMPTS = (
    2
)
MAX_QUESTION_CRITIC_ATTEMPTS = 2
DEPENDENCY_RESERVED_CALLS = (
    MAX_SEMANTIC_AUDIT_ATTEMPTS
    + 1  # question build
    + MAX_QUESTION_CRITIC_ATTEMPTS
    + 1  # optional locked-spec rewrite
    + MAX_QUESTION_CRITIC_ATTEMPTS
)  # worst-case semantic/build/critic/rewrite/re-critic tail for one batch


class HardCubeCapacityError(RuntimeError):
    """The requested Hard Cube batch cannot be satisfied without relaxing its selected policy."""

    def __init__(self, report: dict[str, object], report_path: Path) -> None:
        self.report = report
        self.report_path = report_path
        available = report.get(
            "available_candidates", report["available_distinct_templates"]
        )
        super().__init__(
            f"Hard Cube capacity {available}/"
            f"{report['requested_count']}; report={report_path}"
        )


class BudgetedDependencyAuditor:

    def __init__(
        self,
        inner: DependencyAuditor,
        budget: LLMCallBudget,
        *,
        reserved_calls: int = DEPENDENCY_RESERVED_CALLS,
    ) -> None:
        self._inner = inner
        self._budget = budget
        self._reserved_calls = reserved_calls
        self.prompt_version = inner.prompt_version
        self.model_id = inner.model_id

    def audit(
        self, items: tuple[DependencyReviewItem, ...]
    ) -> tuple[DependencyReviewResult, ...]:
        if not items:
            return ()
        snapshot = self._budget.snapshot()
        if snapshot.max_calls is not None and snapshot.calls >= max(
            0, snapshot.max_calls - self._reserved_calls
        ):
            raise GenerationBudgetExceeded(
                candidate="dependency_audit",
                stage="dependency_reserved_budget",
                snapshot=snapshot,
            )
        self._budget.before_call(candidate="dependency_audit", stage="audit")
        return self._inner.audit(items)


@dataclass
class RunStats:
    candidates_per_recipe: dict[str, int] = field(default_factory=dict)
    candidates_per_story_family: dict[str, int] = field(default_factory=dict)
    rejections_by_recipe: dict[str, int] = field(default_factory=dict)
    audit_stats_total: AuditStats = field(default_factory=AuditStats)
    semantic_audit_calls: int = 0
    semantic_audit_response_failures: int = 0
    semantic_accepts: int = 0
    semantic_rejects: int = 0
    semantic_needs_review: int = 0
    semantic_accepted_per_recipe: dict[str, int] = field(default_factory=dict)
    question_calls: int = 0
    question_parse_failures: int = 0
    question_critic_calls: int = 0
    question_critic_response_failures: int = 0
    question_critic_rewrite_requests: int = 0
    question_critic_final_rejects: int = 0
    question_gate_rejects: int = 0
    dedup_rejects: int = 0
    exact_question_duplicate_rejects: int = 0
    runtime_semantic_duplicate_rejects: int = 0
    external_query_duplicate_rejects: int = 0
    external_question_duplicate_rejects: int = 0
    non_deep_hard_rejects: int = 0
    accepted_per_operator_signature: dict[str, int] = field(default_factory=dict)
    accepted_per_reasoning_archetype: dict[str, int] = field(default_factory=dict)
    accepted_per_story_family: dict[str, int] = field(default_factory=dict)
    accepted_per_frame: dict[str, int] = field(default_factory=dict)
    accepted_per_template_id: dict[str, int] = field(default_factory=dict)
    operator_quota_skips: int = 0
    archetype_quota_skips: int = 0
    story_family_quota_skips: int = 0
    frame_quota_skips: int = 0
    template_quota_skips: int = 0
    written: int = 0

    def add_audit_stats(self, stats: AuditStats) -> None:
        self.audit_stats_total.dependencies_total += stats.dependencies_total
        self.audit_stats_total.cache_hits += stats.cache_hits
        self.audit_stats_total.deterministic_valid += stats.deterministic_valid
        self.audit_stats_total.deterministic_rejected += stats.deterministic_rejected
        self.audit_stats_total.review_sent_to_llm += stats.review_sent_to_llm
        self.audit_stats_total.llm_calls += stats.llm_calls
        self.audit_stats_total.llm_confirmed += stats.llm_confirmed
        self.audit_stats_total.llm_rejected += stats.llm_rejected
        self.audit_stats_total.period_deterministic_valid += (
            stats.period_deterministic_valid
        )
        self.audit_stats_total.period_deterministic_rejected += (
            stats.period_deterministic_rejected
        )
        self.audit_stats_total.period_review_sent += stats.period_review_sent
        self.audit_stats_total.period_cache_hits += stats.period_cache_hits
        for reason, n in stats.rejection_reasons.items():
            self.audit_stats_total.rejection_reasons[reason] = (
                self.audit_stats_total.rejection_reasons.get(reason, 0) + n
            )


def _chunk(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


class _DraftRoundRobin:

    def __init__(
        self,
        *,
        frame_ids: tuple[str, ...],
        cube,
        company_meta,
        seed: int | None,
    ) -> None:
        self._order = list(frame_ids)
        if self._order and seed is not None:
            offset = seed % len(self._order)
            self._order = self._order[offset:] + self._order[:offset]
        self._iterators: dict[str, Iterator[AnalyticalCandidateDraft]] = {
            frame_id: iter_analytical_frame_drafts(
                frame_id,
                cube=cube,
                company_meta=company_meta,
                seed=seed,
                max_candidates=INVENTORY_PER_RECIPE_ATTEMPTS,
            )
            for frame_id in frame_ids
        }
        self._cursor = 0
        self._deferred: list[AnalyticalCandidateDraft] = []

    @property
    def exhausted(self) -> bool:
        return not self._order and not self._deferred

    def _next(self) -> AnalyticalCandidateDraft | None:
        while self._order:
            if self._cursor >= len(self._order):
                self._cursor = 0
            frame_id = self._order[self._cursor]
            iterator = self._iterators[frame_id]
            try:
                draft = next(iterator)
            except StopIteration:
                self._order.pop(self._cursor)
                self._iterators.pop(frame_id, None)
                continue
            self._cursor = (self._cursor + 1) % len(self._order)
            return draft
        return None

    def take(
        self,
        *,
        limit: int,
        quota_state: "_QuotaState",
    ) -> list[AnalyticalCandidateDraft]:
        result: list[AnalyticalCandidateDraft] = []
        planned = _QuotaState(quota_state.limits)

        still_deferred: list[AnalyticalCandidateDraft] = []
        for draft in self._deferred:
            key = _draft_quota_key(draft)
            if not quota_state.can_accept_key(key):
                continue
            if len(result) < limit and quota_state.can_accept_key(key, extra=planned):
                result.append(draft)
                planned.accept_key(key)
            else:
                still_deferred.append(draft)
        self._deferred = still_deferred

        while len(result) < limit:
            draft = self._next()
            if draft is None:
                break
            key = _draft_quota_key(draft)
            if not quota_state.can_accept_key(key):
                continue
            if not quota_state.can_accept_key(key, extra=planned):
                self._deferred.append(draft)
                continue
            result.append(draft)
            planned.accept_key(key)
        return result


@dataclass(frozen=True, slots=True)
class _QuotaLimits:
    operator_signature: int
    reasoning_archetype: int
    story_family: int
    frame: int
    # Reviewed-template contract is always one candidate per template until the registry has been
    # exhausted.  Legacy candidates receive a candidate-scoped synthetic key and are unaffected.
    template_id: int = 1


@dataclass(frozen=True, slots=True)
class _QuotaKey:
    operator_signature: str
    reasoning_archetype: str
    story_family: str
    frame: str
    template_id: str


def _quota_limits(count: int) -> _QuotaLimits:
    if count <= 5:
        return _QuotaLimits(
            operator_signature=1, reasoning_archetype=1, story_family=1, frame=1
        )
    return _QuotaLimits(
        operator_signature=MAX_PER_OPERATOR_SIGNATURE,
        reasoning_archetype=max(2, math.ceil(count / 8)),
        story_family=MAX_PER_STORY_FAMILY,
        frame=MAX_PER_FRAME,
    )


class _QuotaState:
    def __init__(self, limits: _QuotaLimits, *, stats: RunStats | None = None) -> None:
        self.limits = limits
        self.operator_counts: Counter[str] = Counter()
        self.archetype_counts: Counter[str] = Counter()
        self.story_counts: Counter[str] = Counter()
        self.frame_counts: Counter[str] = Counter()
        self.template_counts: Counter[str] = Counter()
        self._stats = stats

    def can_accept_key(
        self, key: _QuotaKey, *, extra: "_QuotaState | None" = None
    ) -> bool:
        extra_operator = (
            extra.operator_counts[key.operator_signature] if extra is not None else 0
        )
        extra_archetype = (
            extra.archetype_counts[key.reasoning_archetype] if extra is not None else 0
        )
        extra_story = extra.story_counts[key.story_family] if extra is not None else 0
        extra_frame = extra.frame_counts[key.frame] if extra is not None else 0
        extra_template = (
            extra.template_counts[key.template_id] if extra is not None else 0
        )
        if not key.template_id.startswith("__legacy__:"):
            return (
                self.template_counts[key.template_id] + extra_template
                < self.limits.template_id
            )
        return (
            self.operator_counts[key.operator_signature] + extra_operator
            < self.limits.operator_signature
            and self.archetype_counts[key.reasoning_archetype] + extra_archetype
            < self.limits.reasoning_archetype
            and self.story_counts[key.story_family] + extra_story
            < self.limits.story_family
            and self.frame_counts[key.frame] + extra_frame < self.limits.frame
            and self.template_counts[key.template_id] + extra_template
            < self.limits.template_id
        )

    def can_accept(self, candidate: RecipeCandidate) -> bool:
        return self.can_accept_key(_candidate_quota_key(candidate))

    def accept_key(self, key: _QuotaKey) -> None:
        self.operator_counts[key.operator_signature] += 1
        self.archetype_counts[key.reasoning_archetype] += 1
        self.story_counts[key.story_family] += 1
        self.frame_counts[key.frame] += 1
        self.template_counts[key.template_id] += 1

    def accept(self, candidate: RecipeCandidate) -> None:
        key = _candidate_quota_key(candidate)
        self.accept_key(key)
        if self._stats is not None:
            self._stats.accepted_per_operator_signature[key.operator_signature] = (
                self.operator_counts[key.operator_signature]
            )
            self._stats.accepted_per_reasoning_archetype[key.reasoning_archetype] = (
                self.archetype_counts[key.reasoning_archetype]
            )
            self._stats.accepted_per_story_family[key.story_family] = self.story_counts[
                key.story_family
            ]
            self._stats.accepted_per_frame[key.frame] = self.frame_counts[key.frame]
            if not key.template_id.startswith("__legacy__:"):
                self._stats.accepted_per_template_id[key.template_id] = (
                    self.template_counts[key.template_id]
                )

    def record_skip(self, candidate: RecipeCandidate) -> None:
        if self._stats is None:
            return
        key = _candidate_quota_key(candidate)
        if self.template_counts[key.template_id] >= self.limits.template_id:
            self._stats.template_quota_skips += 1
        elif (
            self.operator_counts[key.operator_signature]
            >= self.limits.operator_signature
        ):
            self._stats.operator_quota_skips += 1
        elif (
            self.archetype_counts[key.reasoning_archetype]
            >= self.limits.reasoning_archetype
        ):
            self._stats.archetype_quota_skips += 1
        elif self.story_counts[key.story_family] >= self.limits.story_family:
            self._stats.story_family_quota_skips += 1
        elif self.frame_counts[key.frame] >= self.limits.frame:
            self._stats.frame_quota_skips += 1


def _record_valid_candidate(candidate: RecipeCandidate, *, stats: RunStats) -> None:
    recipe_id = candidate.recipe_id
    stats.candidates_per_recipe[recipe_id] = (
        stats.candidates_per_recipe.get(recipe_id, 0) + 1
    )
    family = candidate.public_spec.story_family
    if family:
        stats.candidates_per_story_family[family] = (
            stats.candidates_per_story_family.get(family, 0) + 1
        )
    stats.add_audit_stats(candidate.audit_stats)


def _finalize_drafts(
    drafts: list[AnalyticalCandidateDraft],
    *,
    docs,
    company_meta,
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    stats: RunStats,
    require_deep_hard: bool = False,
) -> tuple[list[RecipeCandidate], bool]:
    validated: list[RecipeCandidate] = []
    for draft in drafts:
        if require_deep_hard:
            hardness_error = deep_hardness_gate_error(compute_hardness(draft.graph))
            if hardness_error:
                stats.non_deep_hard_rejects += 1
                stats.rejections_by_recipe[draft.frame.frame_id] = (
                    stats.rejections_by_recipe.get(draft.frame.frame_id, 0) + 1
                )
                logger.debug(
                    "Cube draft reject non-Deep frame=%s periods=%s: %s",
                    draft.frame.frame_id,
                    draft.periods,
                    hardness_error,
                )
                continue
        try:
            candidate = finalize_analytical_draft(
                draft,
                docs=docs,
                company_meta=company_meta,
                auditor=auditor,
                audit_cache=audit_cache,
            )
        except GenerationBudgetExceeded:
            logger.warning("Stopping dependency audit to preserve budget for semantic and question stages.")
            return validated, True
        except (DependencyAuditRejected, CandidateRejected, EvaluationError) as exc:
            stats.rejections_by_recipe[draft.frame.frame_id] = (
                stats.rejections_by_recipe.get(draft.frame.frame_id, 0) + 1
            )
            logger.debug(
                "Cube draft reject frame=%s periods=%s: %s",
                draft.frame.frame_id,
                draft.periods,
                exc,
            )
            continue
        if require_deep_hard:
            hardness_error = deep_hardness_gate_error(compute_hardness(candidate.graph))
            if hardness_error:
                stats.non_deep_hard_rejects += 1
                stats.rejections_by_recipe[draft.frame.frame_id] = (
                    stats.rejections_by_recipe.get(draft.frame.frame_id, 0) + 1
                )
                logger.error(
                    "Cube rebuilt graph no longer Deep Hard frame=%s candidate=%s: %s",
                    draft.frame.frame_id,
                    draft.candidate_id,
                    hardness_error,
                )
                continue
        validated.append(candidate)
        _record_valid_candidate(candidate, stats=stats)
    return validated, False


def _build_inventory(
    *,
    cube,
    docs,
    company_meta,
    auditor: DependencyAuditor,
    audit_cache: AuditCache,
    seed: int | None,
    stats: RunStats,
    requested_count: int = DEFAULT_SAMPLE_COUNT,
) -> dict[str, list[RecipeCandidate]]:
    inventory: dict[str, list[RecipeCandidate]] = {
        frame_id: [] for frame_id in ENABLED_ANALYTICAL_FRAME_IDS
    }
    pool = _DraftRoundRobin(
        frame_ids=ENABLED_ANALYTICAL_FRAME_IDS,
        cube=cube,
        company_meta=company_meta,
        seed=seed,
    )
    quota_state = _QuotaState(_quota_limits(requested_count))
    validated_total = 0
    while validated_total < requested_count and not pool.exhausted:
        drafts = pool.take(
            limit=min(MAX_SEMANTIC_AUDIT_BATCH_SIZE, requested_count - validated_total),
            quota_state=quota_state,
        )
        if not drafts:
            break
        validated, budget_exhausted = _finalize_drafts(
            drafts,
            docs=docs,
            company_meta=company_meta,
            auditor=auditor,
            audit_cache=audit_cache,
            stats=stats,
        )
        for candidate in validated:
            inventory[candidate.recipe_id].append(candidate)
            if quota_state.can_accept(candidate):
                quota_state.accept(candidate)
        validated_total += len(validated)
        if budget_exhausted:
            break
    return inventory


def _audit_batch_with_retry(
    specs: tuple[PublicSpec, ...],
    *,
    auditor: SemanticAuditor,
    budget: LLMCallBudget,
    recipe_id: str,
    stats: RunStats,
) -> tuple[SemanticAssessment, ...]:
    feedback: str | None = None
    last_error: SemanticAuditResponseError | None = None
    for attempt in range(1, MAX_SEMANTIC_AUDIT_ATTEMPTS + 1):
        budget.before_call(candidate=recipe_id, stage="semantic_audit")
        stats.semantic_audit_calls += 1
        try:
            return auditor.audit(specs, feedback=feedback)
        except SemanticAuditResponseError as exc:
            last_error = exc
            stats.semantic_audit_response_failures += 1
            logger.warning(
                "Invalid SemanticAuditor response (attempt %d/%d) recipe=%s: %s",
                attempt,
                MAX_SEMANTIC_AUDIT_ATTEMPTS,
                recipe_id,
                exc,
            )
            feedback = f"Response trước không hợp lệ: {exc}. Hãy audit lại toàn bộ batch và trả đúng schema."
    assert last_error is not None
    raise last_error


def _semantic_audit_per_recipe(
    inventory: dict[str, list[RecipeCandidate]],
    *,
    auditor: SemanticAuditor,
    budget: LLMCallBudget,
    stats: RunStats,
) -> dict[str, list[RecipeCandidate]]:
    accepted: dict[str, list[RecipeCandidate]] = {}
    for recipe_id, candidates in inventory.items():
        if not candidates:
            accepted[recipe_id] = []
            continue
        accepted_ids: set[str] = set()
        for batch in _chunk(candidates, MAX_SEMANTIC_AUDIT_BATCH_SIZE):
            specs = tuple(c.public_spec for c in batch)
            try:
                assessments = _audit_batch_with_retry(
                    specs,
                    auditor=auditor,
                    budget=budget,
                    recipe_id=recipe_id,
                    stats=stats,
                )
            except GenerationBudgetExceeded:
                logger.warning(
                    "LLM call budget exhausted during semantic audit; stopping with %d audited recipes.",
                    len(accepted),
                )
                accepted[recipe_id] = [
                    c for c in candidates if c.candidate_id in accepted_ids
                ]
                raise
            for assessment in assessments:
                if assessment.decision == "accept":
                    accepted_ids.add(assessment.candidate_id)
                    stats.semantic_accepts += 1
                elif assessment.decision == "reject":
                    stats.semantic_rejects += 1
                    logger.info(
                        "Semantic audit reject candidate=%s recipe=%s: %s",
                        assessment.candidate_id,
                        recipe_id,
                        assessment.rationale,
                    )
                else:
                    stats.semantic_needs_review += 1
                    logger.info(
                        "Semantic audit needs_review candidate=%s recipe=%s: %s",
                        assessment.candidate_id,
                        recipe_id,
                        assessment.rationale,
                    )
        accepted[recipe_id] = [c for c in candidates if c.candidate_id in accepted_ids]
        stats.semantic_accepted_per_recipe[recipe_id] = len(accepted[recipe_id])
    return accepted


def _candidate_operator_signature(candidate: RecipeCandidate) -> str:
    # Keep unit and scale handling explicit.
    if candidate.graph is None:
        return candidate.recipe_id
    return operator_signature(candidate.graph).digest


def _candidate_story_family(candidate: RecipeCandidate) -> str:
    return candidate.public_spec.story_family or candidate.recipe_id


def _graph_reasoning_archetype(graph: object, *, fallback: str) -> str:
    if isinstance(graph, ReasoningGraph):
        return reasoning_archetype(graph)
    return fallback


def _candidate_quota_key(candidate: RecipeCandidate) -> _QuotaKey:
    template_id = getattr(candidate.public_spec, "template_id", None)
    return _QuotaKey(
        operator_signature=_candidate_operator_signature(candidate),
        reasoning_archetype=_graph_reasoning_archetype(
            candidate.graph, fallback=candidate.recipe_id
        ),
        story_family=_candidate_story_family(candidate),
        frame=candidate.recipe_id,
        template_id=template_id or f"__legacy__:{candidate.candidate_id}",
    )


def _draft_quota_key(draft: AnalyticalCandidateDraft) -> _QuotaKey:
    template_id = getattr(draft.public_spec, "template_id", None)
    legacy_id = getattr(draft, "candidate_id", draft.frame.frame_id)
    return _QuotaKey(
        operator_signature=operator_signature(draft.graph).digest,
        reasoning_archetype=_graph_reasoning_archetype(
            draft.graph, fallback=draft.frame.frame_id
        ),
        story_family=draft.public_spec.story_family or draft.frame.frame_id,
        frame=draft.frame.frame_id,
        template_id=template_id or f"__legacy__:{legacy_id}",
    )


def _candidate_priority_order(
    selected: dict[str, list[RecipeCandidate]],
) -> list[RecipeCandidate]:
    """Stable round-robin by coarse archetype, interleaving frames inside each archetype."""
    nested: dict[str, dict[str, list[RecipeCandidate]]] = {}
    for frame_id, candidates in selected.items():
        for candidate in candidates:
            archetype = _candidate_quota_key(candidate).reasoning_archetype
            nested.setdefault(archetype, {}).setdefault(frame_id, []).append(candidate)

    queues: dict[str, list[RecipeCandidate]] = {}
    for archetype, frame_queues in nested.items():
        queue: list[RecipeCandidate] = []
        progressed = True
        while progressed:
            progressed = False
            for items in frame_queues.values():
                if items:
                    queue.append(items.pop(0))
                    progressed = True
        queues[archetype] = queue

    result: list[RecipeCandidate] = []
    seen_ids: set[str] = set()
    archetypes = list(queues)
    progressed = True
    while progressed:
        progressed = False
        for archetype in archetypes:
            queue = queues[archetype]
            if not queue:
                continue
            candidate = queue.pop(0)
            progressed = True
            if candidate.candidate_id in seen_ids:
                continue
            result.append(candidate)
            seen_ids.add(candidate.candidate_id)
    return result


def _candidate_runtime_semantic_key(candidate: RecipeCandidate) -> tuple:
    """Identity of the runtime path that actually produces the terminal value.

    Period windows are deliberately not included: extending a window without changing survivors,
    selected keys, terminal contributors, or answer does not create a new reasoning instance.
    """
    trace = candidate.trace
    graph = candidate.graph
    if trace is None or graph is None:
        return (candidate.recipe_id, candidate.candidate_id)
    terminal_keys = trace.contributing_keys.get(graph.terminal_step_id, frozenset())
    selected = tuple(
        sorted((key, repr(value)) for key, value in trace.selected_keys.items())
    )
    survivors = tuple(
        sorted(
            (key, tuple(sorted(map(repr, values))))
            for key, values in trace.survivor_sets.items()
        )
    )
    return (
        candidate.recipe_id,
        candidate.entities,
        tuple(sorted(map(repr, terminal_keys))),
        selected,
        survivors,
        candidate.formatted_answer,
    )


def _ordered_candidates(
    selected: dict[str, list[RecipeCandidate]],
    *,
    count: int,
    limit: int | None,
    stats: RunStats | None = None,
) -> list[RecipeCandidate]:
    """Apply the CP7 quota contract to the stable candidate priority order."""
    result: list[RecipeCandidate] = []
    quota_state = _QuotaState(_quota_limits(count), stats=stats)
    seen_runtime_semantics: set[tuple] = set()

    def _has_room() -> bool:
        return limit is None or len(result) < limit

    for candidate in _candidate_priority_order(selected):
        if not _has_room():
            break
        if not quota_state.can_accept(candidate):
            quota_state.record_skip(candidate)
            continue
        runtime_key = _candidate_runtime_semantic_key(candidate)
        if runtime_key in seen_runtime_semantics:
            if stats is not None:
                stats.runtime_semantic_duplicate_rejects += 1
            continue
        result.append(candidate)
        seen_runtime_semantics.add(runtime_key)
        quota_state.accept(candidate)

    return result


def _diversity_capacity_error(
    selected: dict[str, list[RecipeCandidate]], *, count: int
) -> str:
    ordered = _ordered_candidates(selected, count=count, limit=None)
    if len(ordered) < count:
        limits = _quota_limits(count)
        signatures = {_candidate_operator_signature(candidate) for candidate in ordered}
        archetypes = {
            _candidate_quota_key(candidate).reasoning_archetype for candidate in ordered
        }
        stories = {_candidate_story_family(candidate) for candidate in ordered}
        frames = {candidate.recipe_id for candidate in ordered}
        template_ids = {
            candidate.public_spec.template_id
            for candidate in ordered
            if candidate.public_spec.template_id is not None
        }
        return (
            f"template capacity provides only {len(ordered)} candidates; {count} are required; "
            f"limits={limits.operator_signature}/{limits.reasoning_archetype}/"
            f"{limits.story_family}/{limits.frame}/{limits.template_id}, "
            f"template_id={len(template_ids)}, operator={len(signatures)}, "
            f"archetype={len(archetypes)}, story={len(stories)}, frame={len(frames)}"
        )
    return ""


def _template_capacity_report(
    accepted: list[tuple[RecipeCandidate, str]], *, requested_count: int
) -> dict[str, object]:
    accepted_ids = tuple(
        candidate.public_spec.template_id
        for candidate, _question in accepted
        if candidate.public_spec.template_id is not None
    )
    distinct_ids = tuple(dict.fromkeys(accepted_ids))
    states = Counter(
        intent.implementation_state.value for intent in INTENTS_BY_ID.values()
    )
    return {
        "requested_count": requested_count,
        "available_candidates": len(accepted),
        "available_distinct_templates": len(distinct_ids),
        "accepted_template_ids": list(distinct_ids),
        "duplicate_template_ids": sorted(
            template_id for template_id, n in Counter(accepted_ids).items() if n > 1
        ),
        "missing_template_count": max(0, requested_count - len(distinct_ids)),
        "registry_size": len(INTENTS_BY_ID),
        "registry_status_distribution": dict(sorted(states.items())),
        "templates": [
            {
                "template_id": intent.template_id,
                "status": intent.implementation_state.value,
                "operation_grammar": intent.operation_grammar,
                "universe_kind": intent.universe_kind.value,
                "terminal_operation": intent.terminal_operation.value,
                "metric_family": intent.metric_family,
                "accepted": intent.template_id in distinct_ids,
            }
            for intent in INTENTS_BY_ID.values()
        ],
    }


def _distribution_report(
    accepted: list[tuple[RecipeCandidate, str]],
) -> dict[str, object]:
    template_ids = [
        candidate.public_spec.template_id
        for candidate, _question in accepted
        if candidate.public_spec.template_id is not None
    ]
    intents = [INTENTS_BY_ID[template_id] for template_id in template_ids]
    hardness = [
        compute_hardness(graph)
        for candidate, _question in accepted
        if isinstance((graph := getattr(candidate, "graph", None)), ReasoningGraph)
    ]
    return {
        "count": len(accepted),
        "distinct_template_ids": len(set(template_ids)),
        "template_id": dict(sorted(Counter(template_ids).items())),
        "operation_grammar": dict(
            sorted(Counter(i.operation_grammar for i in intents).items())
        ),
        "universe_kind": dict(
            sorted(Counter(i.universe_kind.value for i in intents).items())
        ),
        "terminal_operation": dict(
            sorted(Counter(i.terminal_operation.value for i in intents).items())
        ),
        "metric_family": dict(
            sorted(Counter(i.metric_family for i in intents).items())
        ),
        "deep_hard": {
            "required_reasoning_depth": 3,
            "required_adaptive_edges": 2,
            "verified_count": len(hardness),
            "pass_count": sum(
                not deep_hardness_gate_error(metrics) for metrics in hardness
            ),
            "reasoning_depth": dict(
                sorted(Counter(item.reasoning_depth for item in hardness).items())
            ),
            "adaptive_edges": dict(
                sorted(Counter(item.adaptive_edges for item in hardness).items())
            ),
        },
    }


def _apply_quota_and_dedup(
    selected: dict[str, list[RecipeCandidate]], *, count: int
) -> list[RecipeCandidate]:
    """CP7 diversity quota: operator/story/frame = 1/1/1 for small batches, 3/3/2 otherwise."""
    return _ordered_candidates(selected, count=count, limit=count)


def _normalize_question_for_exact_dup(question: str) -> str:
    return " ".join(question.casefold().split())


def _build_questions(
    candidates: list[RecipeCandidate],
    *,
    builder: QuestionBuilder,
    budget: LLMCallBudget,
    stats: RunStats,
) -> list[tuple[RecipeCandidate, str]]:
    by_id = {c.candidate_id: c for c in candidates}
    result: list[tuple[RecipeCandidate, str]] = []
    for batch in _chunk(candidates, MAX_QUESTION_BATCH_SIZE):
        specs = tuple(c.public_spec for c in batch)
        try:
            budget.before_call(candidate="question_batch", stage="question_build")
        except GenerationBudgetExceeded:
            logger.warning(
                "LLM call budget exhausted during question building; stopping with %d questions.",
                len(result),
            )
            raise
        stats.question_calls += 1
        items = builder.build(specs)
        if not items and specs:
            stats.question_parse_failures += 1
        for item in items:
            candidate = by_id.get(item.candidate_id)
            if candidate is None:
                continue
            result.append((candidate, item.question))
    return result


def _critique_questions(
    items: list[tuple[RecipeCandidate, str]],
    *,
    critic: QuestionCritic,
    budget: LLMCallBudget,
    stats: RunStats,
    stage: str,
) -> dict[str, QuestionQualityAssessment]:
    """Run the independent question critic with one contract retry and no implicit fallback.

    The orchestration re-validates complete/unique candidate IDs even though the production LLM
    implementation also does so; custom implementations cannot accidentally turn a missing
    assessment into an accept.
    """
    if not items:
        return {}
    candidates_by_id = {candidate.candidate_id: candidate for candidate, _ in items}
    questions = {candidate.candidate_id: question for candidate, question in items}
    if len(candidates_by_id) != len(items):
        raise QuestionQualityResponseError(
            "Question critic input contains duplicate candidate_id values"
        )
    specs = tuple(candidate.public_spec for candidate, _ in items)
    last_error: QuestionQualityResponseError | None = None
    for attempt in range(1, MAX_QUESTION_CRITIC_ATTEMPTS + 1):
        budget.before_call(candidate="question_critic_batch", stage=stage)
        stats.question_critic_calls += 1
        try:
            assessments = critic.critique(
                specs,
                questions=questions,
                feedback=str(last_error) if last_error else None,
            )
            decisions: dict[str, QuestionQualityAssessment] = {}
            for assessment in assessments:
                if assessment.candidate_id not in candidates_by_id:
                    raise QuestionQualityResponseError(
                        f"candidate_id is outside the critic batch: {assessment.candidate_id!r}"
                    )
                if assessment.candidate_id in decisions:
                    raise QuestionQualityResponseError(
                        f"Duplicate candidate_id in critic batch: {assessment.candidate_id!r}"
                    )
                decisions[assessment.candidate_id] = assessment
            missing = set(candidates_by_id) - set(decisions)
            if missing:
                raise QuestionQualityResponseError(
                    f"Critic response is missing candidate_id: {sorted(missing)}"
                )
            return decisions
        except QuestionQualityResponseError as exc:
            last_error = exc
            stats.question_critic_response_failures += 1
            logger.warning(
                "Invalid QuestionCritic response (attempt %d/%d, stage=%s): %s",
                attempt,
                MAX_QUESTION_CRITIC_ATTEMPTS,
                stage,
                exc,
            )
    assert last_error is not None
    raise last_error


def _question_gate_ok(
    candidate: RecipeCandidate,
    question: str,
    *,
    seen_normalized: set[str],
    stats: RunStats,
    rejection_feedback: dict[str, str] | None = None,
) -> bool:
    spec: PublicSpec = candidate.public_spec
    round_trip_error = question_round_trip_error(question, spec)
    if round_trip_error:
        stats.question_gate_rejects += 1
        if rejection_feedback is not None:
            rejection_feedback[candidate.candidate_id] = round_trip_error
        logger.info(
            "Cube question reject (round-trip) candidate=%s: %s",
            candidate.candidate_id,
            round_trip_error,
        )
        logger.debug(
            "Cube rejected question text candidate=%s question=%r",
            candidate.candidate_id,
            question,
        )
        return False
    leak_error = question_answer_leak_error(question, candidate.formatted_answer)
    if leak_error:
        stats.question_gate_rejects += 1
        if rejection_feedback is not None:
            rejection_feedback[candidate.candidate_id] = leak_error
        logger.info(
            "Cube question reject (answer leak) candidate=%s: %s",
            candidate.candidate_id,
            leak_error,
        )
        logger.debug(
            "Cube rejected question text candidate=%s question=%r",
            candidate.candidate_id,
            question,
        )
        return False
    naturalness_error = question_naturalness_error(question)
    if naturalness_error:
        stats.question_gate_rejects += 1
        if rejection_feedback is not None:
            rejection_feedback[candidate.candidate_id] = naturalness_error
        logger.info(
            "Cube question reject (naturalness) candidate=%s: %s",
            candidate.candidate_id,
            naturalness_error,
        )
        logger.debug(
            "Cube rejected question text candidate=%s question=%r",
            candidate.candidate_id,
            question,
        )
        return False
    normalized = _normalize_question_for_exact_dup(question)
    if normalized in seen_normalized:
        stats.dedup_rejects += 1
        stats.exact_question_duplicate_rejects += 1
        if rejection_feedback is not None:
            rejection_feedback[candidate.candidate_id] = (
                "câu hỏi trùng nguyên văn với câu đã nhận"
            )
        logger.info(
            "Cube question reject (exact text duplicate) candidate=%s",
            candidate.candidate_id,
        )
        logger.debug(
            "Cube rejected question text candidate=%s question=%r",
            candidate.candidate_id,
            question,
        )
        return False
    seen_normalized.add(normalized)
    return True


def _revise_questions(
    candidates: list[RecipeCandidate],
    *,
    rejection_feedback: dict[str, str],
    builder: QuestionBuilder,
    budget: LLMCallBudget,
    stats: RunStats,
) -> list[tuple[RecipeCandidate, str]]:
    revise = getattr(builder, "revise", None)
    if not candidates or not callable(revise):
        return []
    budget.before_call(candidate="question_revision_batch", stage="question_build")
    stats.question_calls += 1
    specs = tuple(candidate.public_spec for candidate in candidates)
    items = revise(specs, rejection_feedback=rejection_feedback)
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    return [
        (by_id[item.candidate_id], item.question)
        for item in items
        if item.candidate_id in by_id
    ]


def _gate_and_dedup_questions(
    items: list[tuple[RecipeCandidate, str]], *, stats: RunStats
) -> list[tuple[RecipeCandidate, str]]:
    accepted: list[tuple[RecipeCandidate, str]] = []
    seen_normalized: set[str] = set()
    for candidate, question in items:
        if _question_gate_ok(
            candidate, question, seen_normalized=seen_normalized, stats=stats
        ):
            accepted.append((candidate, question))
    return accepted


def _select_and_gate_questions(
    selected: dict[str, list[RecipeCandidate]],
    *,
    count: int,
    builder: QuestionBuilder,
    budget: LLMCallBudget,
    stats: RunStats,
) -> list[tuple[RecipeCandidate, str]]:
    ordered = _candidate_priority_order(selected)
    accepted: list[tuple[RecipeCandidate, str]] = []
    quota_state = _QuotaState(_quota_limits(count), stats=stats)
    seen_normalized: set[str] = set()
    seen_runtime_semantics: set[tuple] = set()
    idx = 0
    while len(accepted) < count and idx < len(ordered):
        batch = ordered[idx : idx + MAX_QUESTION_BATCH_SIZE]
        idx += len(batch)
        items = _build_questions(batch, builder=builder, budget=budget, stats=stats)
        for candidate, question in items:
            if len(accepted) >= count:
                break
            if not quota_state.can_accept(candidate):
                quota_state.record_skip(candidate)
                continue
            runtime_key = _candidate_runtime_semantic_key(candidate)
            if runtime_key in seen_runtime_semantics:
                stats.runtime_semantic_duplicate_rejects += 1
                continue
            if _question_gate_ok(
                candidate, question, seen_normalized=seen_normalized, stats=stats
            ):
                accepted.append((candidate, question))
                seen_runtime_semantics.add(runtime_key)
                quota_state.accept(candidate)
    return accepted


def _semantic_audit_candidates(
    candidates: list[RecipeCandidate],
    *,
    auditor: SemanticAuditor,
    budget: LLMCallBudget,
    stats: RunStats,
) -> list[RecipeCandidate]:
    if not candidates:
        return []
    assessments = _audit_batch_with_retry(
        tuple(candidate.public_spec for candidate in candidates),
        auditor=auditor,
        budget=budget,
        recipe_id="cube_mixed_batch",
        stats=stats,
    )
    decisions = {assessment.candidate_id: assessment for assessment in assessments}
    accepted: list[RecipeCandidate] = []
    for candidate in candidates:
        assessment = decisions.get(candidate.candidate_id)
        if assessment is None:
            stats.semantic_rejects += 1
            logger.info(
                "Semantic audit omitted candidate=%s; rejecting candidate.",
                candidate.candidate_id,
            )
            continue
        if assessment.decision == "accept":
            accepted.append(candidate)
            stats.semantic_accepts += 1
            stats.semantic_accepted_per_recipe[candidate.recipe_id] = (
                stats.semantic_accepted_per_recipe.get(candidate.recipe_id, 0) + 1
            )
        elif assessment.decision == "reject":
            stats.semantic_rejects += 1
            logger.info(
                "Semantic audit reject candidate=%s recipe=%s: %s",
                candidate.candidate_id,
                candidate.recipe_id,
                assessment.rationale,
            )
        else:
            stats.semantic_needs_review += 1
            logger.info(
                "Semantic audit needs_review candidate=%s recipe=%s: %s",
                candidate.candidate_id,
                candidate.recipe_id,
                assessment.rationale,
            )
    return accepted


def _run_lazy_pipeline(
    *,
    cube,
    docs,
    company_meta,
    dependency_auditor: DependencyAuditor,
    audit_cache: AuditCache,
    semantic_auditor: SemanticAuditor,
    question_builder: QuestionBuilder,
    question_critic: QuestionCritic,
    budget: LLMCallBudget,
    count: int,
    seed: int | None,
    frame_ids: tuple[str, ...],
    stats: RunStats,
    require_deep_hard: bool = False,
    external_query_fingerprints: set[str] | None = None,
    external_question_fingerprints: set[str] | None = None,
) -> list[tuple[RecipeCandidate, str]]:
    """Dependency -> semantic -> build -> critic/rewrite/re-critic -> gates, until count."""
    pool = _DraftRoundRobin(
        frame_ids=frame_ids,
        cube=cube,
        company_meta=company_meta,
        seed=seed,
    )
    accepted: list[tuple[RecipeCandidate, str]] = []
    quota_state = _QuotaState(_quota_limits(count), stats=stats)
    seen_normalized: set[str] = set()
    seen_runtime_semantics: set[tuple] = set()
    dedup_queries_enabled = external_query_fingerprints is not None
    seen_query_fingerprints = set(external_query_fingerprints or ())
    external_question_fingerprints = set(external_question_fingerprints or ())

    while len(accepted) < count and not pool.exhausted:
        drafts = pool.take(
            limit=MAX_SEMANTIC_AUDIT_BATCH_SIZE,
            quota_state=quota_state,
        )
        if not drafts:
            break
        validated, dependency_budget_exhausted = _finalize_drafts(
            drafts,
            docs=docs,
            company_meta=company_meta,
            auditor=dependency_auditor,
            audit_cache=audit_cache,
            stats=stats,
            require_deep_hard=require_deep_hard,
        )
        if dedup_queries_enabled:
            fresh_validated: list[RecipeCandidate] = []
            for candidate in validated:
                query_fingerprint = _deep_query_fingerprint(
                    candidate.compiled.pandas_query
                )
                if query_fingerprint in seen_query_fingerprints:
                    stats.external_query_duplicate_rejects += 1
                    logger.debug(
                        "Cube reject query fingerprint already present candidate=%s",
                        candidate.candidate_id,
                    )
                    continue
                seen_query_fingerprints.add(query_fingerprint)
                fresh_validated.append(candidate)
            validated = fresh_validated
        if validated:
            try:
                semantic_accepted = _semantic_audit_candidates(
                    validated,
                    auditor=semantic_auditor,
                    budget=budget,
                    stats=stats,
                )
                question_items = _build_questions(
                    semantic_accepted,
                    builder=question_builder,
                    budget=budget,
                    stats=stats,
                )
                fresh_question_items: list[tuple[RecipeCandidate, str]] = []
                for candidate, question in question_items:
                    fingerprint = _deep_sha256(_deep_normalized_question(question))
                    if fingerprint in external_question_fingerprints:
                        stats.external_question_duplicate_rejects += 1
                        logger.debug(
                            "Cube reject question fingerprint already present candidate=%s",
                            candidate.candidate_id,
                        )
                        continue
                    fresh_question_items.append((candidate, question))
                question_items = fresh_question_items
                if not question_items:
                    continue
                critic_decisions = _critique_questions(
                    question_items,
                    critic=question_critic,
                    budget=budget,
                    stats=stats,
                    stage="question_critic",
                )
            except GenerationBudgetExceeded as exc:
                logger.warning("Cube stopped at %s because the budget was exhausted: %s", exc.stage, exc)
                break
            rejected_for_revision: list[RecipeCandidate] = []
            rejection_feedback: dict[str, str] = {}

            def _accept_question(candidate: RecipeCandidate, question: str) -> bool:
                if len(accepted) >= count:
                    return False
                if not quota_state.can_accept(candidate):
                    quota_state.record_skip(candidate)
                    return False
                runtime_key = _candidate_runtime_semantic_key(candidate)
                if runtime_key in seen_runtime_semantics:
                    stats.runtime_semantic_duplicate_rejects += 1
                    return False
                if _question_gate_ok(
                    candidate,
                    question,
                    seen_normalized=seen_normalized,
                    stats=stats,
                    rejection_feedback=rejection_feedback,
                ):
                    accepted.append((candidate, question))
                    seen_runtime_semantics.add(runtime_key)
                    quota_state.accept(candidate)
                    return True
                return False

            for candidate, question in question_items:
                assessment = critic_decisions[candidate.candidate_id]
                if assessment.decision == "rewrite":
                    stats.question_critic_rewrite_requests += 1
                    rejection_feedback[candidate.candidate_id] = assessment.feedback
                    rejected_for_revision.append(candidate)
                    logger.info(
                        "QuestionCritic requested a rewrite for candidate=%s: %s",
                        candidate.candidate_id,
                        assessment.feedback,
                    )
                    continue
                if (
                    not _accept_question(candidate, question)
                    and candidate.candidate_id in rejection_feedback
                ):
                    rejected_for_revision.append(candidate)

            if rejected_for_revision and len(accepted) < count:
                try:
                    revised_items = _revise_questions(
                        rejected_for_revision,
                        rejection_feedback=rejection_feedback,
                        builder=question_builder,
                        budget=budget,
                        stats=stats,
                    )
                    revised_critic_decisions = _critique_questions(
                        revised_items,
                        critic=question_critic,
                        budget=budget,
                        stats=stats,
                        stage="question_recritic",
                    )
                except GenerationBudgetExceeded as exc:
                    logger.warning(
                        "Cube stopped at %s because the rewrite budget was exhausted: %s",
                        exc.stage,
                        exc,
                    )
                    break
                for candidate, question in revised_items:
                    if len(accepted) >= count:
                        break
                    assessment = revised_critic_decisions[candidate.candidate_id]
                    if assessment.decision != "accept":
                        stats.question_critic_rewrite_requests += 1
                        stats.question_critic_final_rejects += 1
                        logger.info(
                            "QuestionCritic reject sau locked-spec rewrite candidate=%s: %s",
                            candidate.candidate_id,
                            assessment.feedback,
                        )
                        continue
                    _accept_question(candidate, question)
        if dependency_budget_exhausted:
            break
    return accepted


def generate_hard_cube(
    *,
    settings: Settings,
    llm: ChatLLM,
    count: int = DEFAULT_SAMPLE_COUNT,
    out_path: Path,
    seed: int | None = None,
    max_llm_calls: int | None = None,
    frame_id: str | None = None,
    dedup_against_paths: tuple[Path, ...] = (),
    require_deep_hard: bool = True,
) -> int:
    docs = scan_catalog(settings.data_root)
    company_meta = load_company_meta(settings.company_meta_path)
    cube = JsonCubeStore(docs=docs, cache_dir=settings.cache_dir).load_or_build()

    budget = LLMCallBudget(max_llm_calls)
    audit_cache = AuditCache(cache_dir=settings.cache_dir)
    auditor = BudgetedDependencyAuditor(
        LLMDependencyAuditor(llm, model_id=settings.openai_model), budget
    )
    semantic_auditor: SemanticAuditor = LLMSemanticAuditor(llm)
    question_builder: QuestionBuilder = LLMQuestionBuilder(llm)
    question_critic: QuestionCritic = LLMQuestionCritic(llm)

    stats = RunStats()
    writer = JsonlWriter(out_path)

    external_query_fingerprints: set[str] = set()
    external_question_fingerprints: set[str] = set()
    for dedup_against_path in dedup_against_paths:
        for line in dedup_against_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            external_query_fingerprints.add(
                _deep_query_fingerprint(str(record.get("pandas_query", "")))
            )
            external_question_fingerprints.add(
                _deep_sha256(
                    _deep_normalized_question(str(record.get("question", "")))
                )
            )

    if frame_id is not None and frame_id not in ENABLED_ANALYTICAL_FRAME_IDS:
        raise ValueError(f"Cube frame does not exist or is not enabled: {frame_id}")
    frame_ids = (frame_id,) if frame_id is not None else ENABLED_TEMPLATE_FRAME_IDS
    accepted = _run_lazy_pipeline(
        cube=cube,
        docs=docs,
        company_meta=company_meta,
        dependency_auditor=auditor,
        audit_cache=audit_cache,
        semantic_auditor=semantic_auditor,
        question_builder=question_builder,
        question_critic=question_critic,
        budget=budget,
        count=count,
        seed=seed,
        frame_ids=frame_ids,
        stats=stats,
        require_deep_hard=require_deep_hard,
        external_query_fingerprints=(
            external_query_fingerprints if dedup_against_paths else None
        ),
        external_question_fingerprints=(
            external_question_fingerprints if dedup_against_paths else None
        ),
    )

    capacity_report = _template_capacity_report(accepted, requested_count=count)
    capacity_key = (
        "available_distinct_templates" if require_deep_hard else "available_candidates"
    )
    if capacity_report[capacity_key] != count:
        report_path = out_path.with_suffix(".capacity.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(capacity_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        if require_deep_hard:
            raise HardCubeCapacityError(capacity_report, report_path)
        logger.warning(
            "Hard Cube standard produced only %d/%d candidates; writing partial output and "
            "retaining the capacity report at %s",
            len(accepted),
            count,
            report_path,
        )

    records = [
        QARecord(
            id=0,
            question=question,
            answer=candidate.formatted_answer,
            relevant_docs=sorted(
                {ref.split("|", 1)[0] for ref in candidate.compiled.relevant_tables}
            ),
            relevant_tables=list(candidate.compiled.relevant_tables),
            pandas_query=candidate.compiled.pandas_query,
            csv_path=[
                str(candidate.compiled.csv_path[ref])
                for ref in candidate.compiled.relevant_tables
            ],
            difficulty="hard",
            template_id=candidate.public_spec.template_id,
        )
        for candidate, question in accepted[:count]
    ]
    writer.replace(records)
    stats.written = len(records)

    distribution_path = out_path.with_suffix(".distribution.json")
    distribution_path.write_text(
        json.dumps(_distribution_report(accepted[:count]), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )

    budget_snapshot = budget.snapshot()
    logger.info(
        "Cube run summary: written=%d/%d candidates_per_recipe=%s audit=%s "
        "candidates_per_story_family=%s "
        "semantic_audit_calls=%d semantic_audit_response_failures=%d semantic_accepts=%d "
        "semantic_rejects=%d semantic_needs_review=%d semantic_accepted_per_recipe=%s "
        "question_calls=%d question_parse_failures=%d question_critic_calls=%d "
        "question_critic_response_failures=%d question_critic_rewrite_requests=%d "
        "question_critic_final_rejects=%d question_gate_rejects=%d dedup_rejects=%d "
        "accepted_per_operator_signature=%s accepted_per_reasoning_archetype=%s "
        "accepted_per_story_family=%s accepted_per_frame=%s operator_quota_skips=%d "
        "archetype_quota_skips=%d story_family_quota_skips=%d frame_quota_skips=%d "
        "exact_question_duplicate_rejects=%d runtime_semantic_duplicate_rejects=%d "
        "external_query_duplicate_rejects=%d external_question_duplicate_rejects=%d "
        "non_deep_hard_rejects=%d "
        "llm_calls_total=%d max_llm_calls=%s",
        stats.written,
        count,
        stats.candidates_per_recipe,
        stats.audit_stats_total,
        stats.candidates_per_story_family,
        stats.semantic_audit_calls,
        stats.semantic_audit_response_failures,
        stats.semantic_accepts,
        stats.semantic_rejects,
        stats.semantic_needs_review,
        stats.semantic_accepted_per_recipe,
        stats.question_calls,
        stats.question_parse_failures,
        stats.question_critic_calls,
        stats.question_critic_response_failures,
        stats.question_critic_rewrite_requests,
        stats.question_critic_final_rejects,
        stats.question_gate_rejects,
        stats.dedup_rejects,
        stats.accepted_per_operator_signature,
        stats.accepted_per_reasoning_archetype,
        stats.accepted_per_story_family,
        stats.accepted_per_frame,
        stats.operator_quota_skips,
        stats.archetype_quota_skips,
        stats.story_family_quota_skips,
        stats.frame_quota_skips,
        stats.exact_question_duplicate_rejects,
        stats.runtime_semantic_duplicate_rejects,
        stats.external_query_duplicate_rejects,
        stats.external_question_duplicate_rejects,
        stats.non_deep_hard_rejects,
        budget_snapshot.calls,
        budget_snapshot.max_calls,
    )
    return stats.written
