
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from vifinqa.common.corpus.statement import StatementCell
from vifinqa.generation.panel.catalog import COST_METRIC_KEYS


class ValueKind(str, Enum):
    NUMERIC_SCALAR = "NumericScalar"
    NUMERIC_SERIES_ENTITY = "NumericSeriesEntity"
    NUMERIC_SERIES_PERIOD = "NumericSeriesPeriod"
    NUMERIC_SERIES_ENTITY_PERIOD = "NumericSeriesEntityPeriod"
    THRESHOLD = "Threshold"
    ENTITY_SET = "EntitySet"
    PERIOD_SET = "PeriodSet"
    ENTITY_KEY = "EntityKey"
    PERIOD_KEY = "PeriodKey"
    ENTITY_PERIOD_KEY = "EntityPeriodKey"


ADAPTIVE_VALUE_KINDS: frozenset[ValueKind] = frozenset(
    {
        ValueKind.THRESHOLD,
        ValueKind.ENTITY_SET,
        ValueKind.PERIOD_SET,
        ValueKind.ENTITY_KEY,
        ValueKind.PERIOD_KEY,
        ValueKind.ENTITY_PERIOD_KEY,
    }
)

# Keep period handling explicit and deterministic.
DomainKey = str | tuple[str, str]


class GraphError(ValueError):
    pass


class OperationError(ValueError):
    pass


class CandidateRejected(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Domain:
    entities: tuple[str, ...]
    periods: tuple[str, ...]
    report_scope: str


@dataclass(frozen=True, slots=True)
class BindingRef:

    table_ref: str
    row_idx: int
    col_idx: int
    metric_key: str
    scale: float


@dataclass(frozen=True, slots=True)
class MetricTerm:

    coefficient: float
    metric_key: str
    cell: StatementCell

    @property
    def binding_ref(self) -> BindingRef:
        return BindingRef(
            table_ref=self.cell.table_ref,
            row_idx=self.cell.row_idx,
            col_idx=self.cell.col_idx,
            metric_key=self.metric_key,
            scale=self.cell.scale,
        )


@dataclass(frozen=True, slots=True)
class MetricTerms:

    numerator: tuple[MetricTerm, ...]
    denominator: tuple[MetricTerm, ...]
    requires_positive_denominator: bool = False

    @property
    def value(self) -> float | None:
        numerator_value = sum(
            term.coefficient * _signed(term.metric_key, term.cell.value) for term in self.numerator
        )
        if not self.denominator:
            return numerator_value
        denominator_value = sum(
            term.coefficient * _signed(term.metric_key, term.cell.value) for term in self.denominator
        )
        if denominator_value == 0 or (self.requires_positive_denominator and denominator_value <= 0):
            return None
        return numerator_value / denominator_value

    @property
    def binding_refs(self) -> tuple[BindingRef, ...]:
        return tuple(term.binding_ref for term in (*self.numerator, *self.denominator))


def _signed(metric_key: str, value: float) -> float:
    return abs(value) if metric_key in COST_METRIC_KEYS else value


@dataclass(frozen=True, slots=True)
class MetricRoleInput:

    role_id: str
    metric_key: str
    expected_kind: ValueKind
    bindings: Mapping[DomainKey, MetricTerms]


@dataclass(frozen=True, slots=True)
class StepOutputInput:

    source_step_id: str
    expected_kind: ValueKind


NodeInput = MetricRoleInput | StepOutputInput


@dataclass(frozen=True, slots=True)
class ReasoningNode:
    step_id: str
    operation: str
    inputs: tuple[NodeInput, ...]
    output_kind: ValueKind
    domain: Domain
    params: Mapping[str, object]
    output: str
    description: str


@dataclass(frozen=True, slots=True)
class ReasoningGraph:
    recipe_id: str
    domain: Domain
    metric_roles: tuple[MetricRoleInput, ...]
    nodes: tuple[ReasoningNode, ...]
    terminal_step_id: str


_VAR_SANITIZE_RE = re.compile(r"[^0-9a-zA-Z_]")


@dataclass
class EmitContext:

    graph: ReasoningGraph
    _var_names: dict[str, str] = field(default_factory=dict)
    used_table_refs: set[str] = field(default_factory=set)

    def var_for(self, step_id: str) -> str:
        if step_id not in self._var_names:
            safe = _VAR_SANITIZE_RE.sub("_", step_id)
            self._var_names[step_id] = f"_v_{safe}"
        return self._var_names[step_id]

    def node_for(self, step_id: str) -> ReasoningNode:
        for node in self.graph.nodes:
            if node.step_id == step_id:
                return node
        raise GraphError(f"step_id was not found in the graph: {step_id}")

    def register_table(self, table_ref: str) -> None:
        self.used_table_refs.add(table_ref)

    def cell_expr(self, cell: StatementCell, metric_key: str) -> str:
        self.register_table(cell.table_ref)
        is_cost = metric_key in COST_METRIC_KEYS
        return f"_cell({cell.table_ref!r}, {cell.row_idx}, {cell.col_idx}, {cell.scale!r}, {is_cost!r})"


@runtime_checkable
class Operation(Protocol):

    name: str

    def validate(self, node: ReasoningNode, graph: ReasoningGraph) -> None:
        ...

    def evaluate(self, node: ReasoningNode, inputs: tuple[object, ...]) -> object:
        ...

    def emit(self, node: ReasoningNode, inputs: tuple[str, ...], context: EmitContext) -> list[str]:
        ...
