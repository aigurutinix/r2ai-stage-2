
from __future__ import annotations

from vifinqa.generation.hard.schemas import (
    BaseHardPlan,
    HardP3Plan,
    HardPlan,
    MetricBinding,
    MetricRoleInput,
    ReasoningStep,
    StepOutputInput,
)
from vifinqa.generation.hard.scenarios import HardScenarioSpec


def graph_gate_error(plan: BaseHardPlan) -> str:
    steps = plan.reasoning_steps
    if not steps:
        return "Plan has no reasoning_steps."

    step_ids = [s.step_id for s in steps]
    if len(step_ids) != len(set(step_ids)):
        return f"Duplicate step_id values: {step_ids}"

    outputs = [s.output for s in steps]
    if len(outputs) != len(set(outputs)):
        return f"Duplicate outputs across steps: {outputs}"

    non_extract = [s for s in steps if s.operation != "extract"]
    if len(non_extract) < 2:
        return f"At least two non-extract steps are required; got {len(non_extract)}."

    lookup_without_key = [
        s.step_id
        for s in steps
        if s.operation == "lookup" and not any(isinstance(inp, StepOutputInput) for inp in s.inputs)
    ]
    if lookup_without_key:
        return f"Lookup steps must receive a lookup key from a previous step's output: {lookup_without_key}"

    step_by_id: dict[str, ReasoningStep] = {s.step_id: s for s in steps}
    depends_on: dict[str, set[str]] = {s.step_id: set() for s in steps}
    for step in steps:
        for inp in step.inputs:
            if isinstance(inp, StepOutputInput):
                if inp.step_id not in step_by_id:
                    return f"Step {step.step_id} references a nonexistent step_id: {inp.step_id}"
                depends_on[step.step_id].add(inp.step_id)

    cycle_error = _find_cycle(depends_on)
    if cycle_error:
        return f"Dependency graph contains a cycle: {cycle_error}"

    last = steps[-1]
    if not any(isinstance(inp, StepOutputInput) for inp in last.inputs):
        return "The final step must reference at least one previous step output (StepOutputInput)."

    return ""


def _find_cycle(depends_on: dict[str, set[str]]) -> str:
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(depends_on, WHITE)
    path: list[str] = []

    def visit(node: str) -> str:
        color[node] = GRAY
        path.append(node)
        for dep in depends_on[node]:
            if color[dep] == GRAY:
                cycle = " -> ".join([*path, dep])
                return cycle
            if color[dep] == WHITE:
                result = visit(dep)
                if result:
                    return result
        path.pop()
        color[node] = BLACK
        return ""

    for node in depends_on:
        if color[node] == WHITE:
            result = visit(node)
            if result:
                return result
    return ""


def p1_dependency_error(plan: HardPlan, scenario: HardScenarioSpec) -> str:
    steps = plan.reasoning_steps
    step_by_id = {s.step_id: s for s in steps}

    filter_steps = [s for s in steps if s.operation == "filter"]
    if len(filter_steps) != 1:
        return f"P1 requires exactly one filter step; got {len(filter_steps)}."
    filter_step = filter_steps[0]

    filter_deps = [
        step_by_id[inp.step_id]
        for inp in filter_step.inputs
        if isinstance(inp, StepOutputInput) and inp.step_id in step_by_id
    ]
    if not filter_deps or not all(d.operation in ("extract", "lookup") for d in filter_deps):
        return "The filter step must receive inputs directly from extract/lookup steps."

    final_step = steps[-1]
    if final_step.operation != plan.draft.final_operation:
        return (
            f"The final step operation must match the locked final_operation "
            f"({plan.draft.final_operation}), got {final_step.operation}."
        )
    if final_step.operation not in scenario.allowed_final_operations:
        return f"Scenario {scenario.name}: final_operation={final_step.operation} is not allowed."

    final_refs_filter = any(
        isinstance(inp, StepOutputInput) and inp.step_id == filter_step.step_id
        for inp in final_step.inputs
    )
    if not final_refs_filter:
        return "The final step must consume the filter step's output, not another step's output."

    return ""


def p1_coverage_error(bindings: list[MetricBinding], selected_table_refs: list[str]) -> str:
    binding_refs = [b.table_ref for b in bindings]
    if len(binding_refs) != len(set(binding_refs)):
        return f"Tables bound more than once: {binding_refs}"
    missing = sorted(set(selected_table_refs) - set(binding_refs))
    if missing:
        return f"Missing bindings for selected tables: {missing}"
    extra = sorted(set(binding_refs) - set(selected_table_refs))
    if extra:
        return f"Bindings reference tables outside the selected set: {extra}"
    tickers = [b.ticker for b in bindings]
    if len(tickers) != len(set(tickers)):
        return f"Companies bound more than once: {tickers}"
    return ""


def p3_dependency_error(plan: HardP3Plan, scenario: HardScenarioSpec) -> str:
    steps = plan.reasoning_steps
    step_by_id = {s.step_id: s for s in steps}
    draft = plan.draft

    selector_steps = [s for s in steps if s.operation in ("argmax", "argmin")]
    if len(selector_steps) != 1:
        return f"P3 requires exactly one argmax/argmin period-selection step; got {len(selector_steps)}."
    selector_step = selector_steps[0]
    if selector_step.operation != draft.selector_operation:
        return (
            f"The period-selection step operation must match the locked selector_operation "
            f"({draft.selector_operation}), got {selector_step.operation}."
        )

    selector_deps = [
        step_by_id[inp.step_id]
        for inp in selector_step.inputs
        if isinstance(inp, StepOutputInput) and inp.step_id in step_by_id
    ]
    if len(selector_deps) < scenario.min_periods:
        return (
            f"The period-selection step requires at least {scenario.min_periods} selector_metric_role "
            f"extract inputs; got {len(selector_deps)}."
        )
    if not all(d.operation == "extract" for d in selector_deps):
        return "The period-selection step must receive inputs directly from selector_metric_role extract steps."
    if not all(
        any(isinstance(inp, MetricRoleInput) and inp.metric_role == draft.selector_metric_role for inp in d.inputs)
        for d in selector_deps
    ):
        return "All extract steps feeding period selection must read selector_metric_role."

    final_step = steps[-1]
    if final_step.operation != "lookup":
        return f"The final P3 step must be a lookup of answer_metric_role for the selected period; got {final_step.operation}."
    if final_step.operation not in scenario.allowed_final_operations:
        return f"Scenario {scenario.name}: final operation={final_step.operation} is not allowed."

    final_refs_selector = any(
        isinstance(inp, StepOutputInput) and inp.step_id == selector_step.step_id for inp in final_step.inputs
    )
    if not final_refs_selector:
        return "The final lookup must use the period-selection step's output as its lookup key; the period cannot be hardcoded."

    answer_extract_deps = [
        step_by_id[inp.step_id]
        for inp in final_step.inputs
        if isinstance(inp, StepOutputInput) and inp.step_id in step_by_id and inp.step_id != selector_step.step_id
    ]
    if len(answer_extract_deps) < scenario.min_periods:
        return (
            f"The final lookup requires at least {scenario.min_periods} per-period answer_metric_role "
            f"extract inputs; got {len(answer_extract_deps)}."
        )
    if not all(d.operation == "extract" for d in answer_extract_deps):
        return "The final lookup's answer candidates must be extract steps."
    if not all(
        any(isinstance(inp, MetricRoleInput) and inp.metric_role == draft.answer_metric_role for inp in d.inputs)
        for d in answer_extract_deps
    ):
        return "All extract steps feeding the final lookup must read answer_metric_role."

    return ""


def p3_coverage_error(
    selector_bindings: list[MetricBinding],
    answer_bindings: list[MetricBinding],
    periods: list[str],
) -> str:
    required = set(periods)

    def _coverage_error(bindings: list[MetricBinding], label: str) -> str:
        binding_periods = [b.period for b in bindings]
        if len(binding_periods) != len(set(binding_periods)):
            return f"Periods with duplicate {label} bindings: {binding_periods}"
        missing = sorted(required - set(binding_periods))
        if missing:
            return f"Missing {label} bindings for periods: {missing}"
        extra = sorted(set(binding_periods) - required)
        if extra:
            return f"{label} bindings reference periods outside the selected window: {extra}"
        return ""

    selector_error = _coverage_error(selector_bindings, "selector")
    if selector_error:
        return selector_error
    answer_error = _coverage_error(answer_bindings, "answer")
    if answer_error:
        return answer_error

    tickers = {b.ticker for b in selector_bindings} | {b.ticker for b in answer_bindings}
    if len(tickers) != 1:
        return f"P3 must use exactly one company across all periods; got: {sorted(tickers)}"
    return ""
