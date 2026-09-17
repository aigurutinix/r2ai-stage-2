"""Capability matrix for the reviewed 70-template Hard Cube registry."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.hard.recipe.grounded.analytical_planner import (
    FRAMES,
    iter_analytical_frame_drafts,
)
from vifinqa.generation.hard.template_intents import (
    CapabilityState,
    INTENT_REGISTRY,
    IntentSpec,
)
from vifinqa.generation.panel.base import Cube


@dataclass(frozen=True, slots=True)
class TemplateCapability:
    template_id: str
    status: CapabilityState
    operation_grammar: str
    universe_kind: str
    terminal_operation: str
    metric_family: str
    compiler_frame_id: str | None
    corpus_draft_count: int
    blocker: str | None


@dataclass(frozen=True, slots=True)
class TemplateCapabilityMatrix:
    templates: tuple[TemplateCapability, ...]
    verification_level: str = "deterministic_graph_build_on_current_corpus"

    @property
    def runnable_count(self) -> int:
        return sum(item.status is CapabilityState.RUNNABLE for item in self.templates)

    def to_payload(self) -> dict[str, object]:
        states = Counter(item.status.value for item in self.templates)
        return {
            "summary": {
                "registry_size": len(self.templates),
                "implemented_count": sum(
                    item.compiler_frame_id is not None for item in self.templates
                ),
                "runnable_count": self.runnable_count,
                "status_distribution": dict(sorted(states.items())),
                "verification_level": self.verification_level,
                "real_dependency_audit_verified": False,
            },
            "templates": [
                {**asdict(item), "status": item.status.value} for item in self.templates
            ],
        }


def _frame_by_template_id() -> dict[str, str]:
    return {
        frame.template_id: frame.frame_id
        for frame in FRAMES
        if frame.template_id is not None
    }


def _capability(
    intent: IntentSpec,
    *,
    frame_id: str | None,
    draft_count: int,
) -> TemplateCapability:
    status = intent.implementation_state
    blocker: str | None = None
    if status is CapabilityState.RUNNABLE and draft_count == 0:
        status = CapabilityState.BLOCKED_BY_DATA
        blocker = "compiler exists but no deterministic graph candidate passes current corpus coverage/gates"
    elif status is CapabilityState.NOT_IMPLEMENTED:
        blocker = "exact operation grammar has no compiler topology"
    elif status is CapabilityState.BLOCKED_BY_METRIC:
        blocker = "required financial metric/formula is unavailable"
    elif status is CapabilityState.BLOCKED_BY_DATA:
        blocker = "required source coverage is unavailable on the current corpus"
    return TemplateCapability(
        template_id=intent.template_id,
        status=status,
        operation_grammar=intent.operation_grammar,
        universe_kind=intent.universe_kind.value,
        terminal_operation=intent.terminal_operation.value,
        metric_family=intent.metric_family,
        compiler_frame_id=frame_id,
        corpus_draft_count=draft_count,
        blocker=blocker,
    )


def build_template_capability_matrix(
    *,
    cube: Cube,
    company_meta: dict[str, CompanyInfo],
    max_drafts_per_template: int = 100,
) -> TemplateCapabilityMatrix:
    frame_ids = _frame_by_template_id()
    rows: list[TemplateCapability] = []
    for intent in INTENT_REGISTRY:
        frame_id = frame_ids.get(intent.template_id)
        draft_count = 0
        if frame_id is not None:
            draft_count = sum(
                1
                for _draft in iter_analytical_frame_drafts(
                    frame_id,
                    cube=cube,
                    company_meta=company_meta,
                    seed=None,
                    max_candidates=max_drafts_per_template,
                )
            )
        rows.append(_capability(intent, frame_id=frame_id, draft_count=draft_count))
    return TemplateCapabilityMatrix(tuple(rows))


def render_template_capability_markdown(matrix: TemplateCapabilityMatrix) -> str:
    payload = matrix.to_payload()
    summary = payload["summary"]
    assert isinstance(summary, dict)
    lines = [
        "# Hard Cube reviewed-template capability matrix",
        "",
        f"- Registry: {summary['registry_size']}/70 template IDs",
        f"- Compiler implemented: {summary['implemented_count']}",
        f"- Deterministic candidates on current corpus: {summary['runnable_count']}",
        f"- Verification: {summary['verification_level']}",
        "- Real dependency audit verified: no",
        "",
        "| template_id | status | operation grammar | universe | terminal | drafts | blocker |",
        "|---|---|---|---|---|---:|---|",
    ]
    for row in matrix.templates:
        lines.append(
            f"| {row.template_id} | {row.status.value} | {row.operation_grammar} | "
            f"{row.universe_kind} | {row.terminal_operation} | {row.corpus_draft_count} | "
            f"{row.blocker or ''} |"
        )
    return "\n".join(lines) + "\n"


def write_template_capability_matrix(
    matrix: TemplateCapabilityMatrix, *, json_out: Path, markdown_out: Path
) -> None:
    json_out.parent.mkdir(parents=True, exist_ok=True)
    markdown_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(
        json.dumps(matrix.to_payload(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_out.write_text(
        render_template_capability_markdown(matrix), encoding="utf-8"
    )
