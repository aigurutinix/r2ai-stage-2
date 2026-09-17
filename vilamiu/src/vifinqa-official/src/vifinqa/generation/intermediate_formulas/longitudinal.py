
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

from vifinqa.generation.intermediate_formulas.base import ResolvedCell
from vifinqa.generation.schemas import AnswerType

MetricKind = Literal["money", "percentage", "number"]


@dataclass(frozen=True, slots=True)
class LongitudinalInputMetric:
    metric_id: str
    label_vi: str
    concept_names: tuple[str, ...]
    kind: MetricKind
    enabled: bool


INPUT_METRICS: dict[str, LongitudinalInputMetric] = {
    "net_revenue": LongitudinalInputMetric(
        metric_id="net_revenue",
        label_vi="Doanh thu thuần",
        concept_names=("Doanh thu thuần", "Doanh thu thuần về bán hàng và cung cấp dịch vụ"),
        kind="money",
        enabled=True,
    ),
    "net_income": LongitudinalInputMetric(
        metric_id="net_income",
        label_vi="Lợi nhuận sau thuế",
        concept_names=("Lợi nhuận sau thuế", "Lợi nhuận sau thuế thu nhập doanh nghiệp"),
        kind="money",
        enabled=True,
    ),
    "total_assets": LongitudinalInputMetric(
        metric_id="total_assets",
        label_vi="Tổng cộng tài sản",
        concept_names=("Tổng cộng tài sản", "Tổng tài sản"),
        kind="money",
        enabled=True,
    ),
}


def get_input_metric(metric_id: str) -> LongitudinalInputMetric:
    return INPUT_METRICS[metric_id]


def enabled_input_metrics() -> list[LongitudinalInputMetric]:
    return [m for m in INPUT_METRICS.values() if m.enabled]


@dataclass(frozen=True, slots=True)
class TransformDefinition:
    transform_id: str
    name_vi: str
    formula_text: str
    enabled: bool


TRANSFORMS: dict[str, TransformDefinition] = {
    "aagr_two_yoy_rates": TransformDefinition(
        transform_id="aagr_two_yoy_rates",
        name_vi="Tăng trưởng bình quân từ hai tỷ lệ YoY (AAGR)",
        formula_text="((x2/x1 - 1) + (x3/x2 - 1)) / 2",
        enabled=True,
    ),
    "signed_growth_acceleration": TransformDefinition(
        transform_id="signed_growth_acceleration",
        name_vi="Gia tốc tăng trưởng (có dấu)",
        formula_text="(x3/x2 - 1) - (x2/x1 - 1)",
        enabled=True,
    ),
    "coefficient_of_variation_three_levels": TransformDefinition(
        transform_id="coefficient_of_variation_three_levels",
        name_vi="Hệ số biến thiên (CoV) qua ba mức",
        formula_text="độ lệch chuẩn (quy ước population, chia cho 3) / trung bình của x1, x2, x3",
        enabled=True,
    ),
}

TRANSFORM_EXPRESSIONS: dict[str, str] = {
    "aagr_two_yoy_rates": "((x2 / x1 - 1) + (x3 / x2 - 1)) / 2",
    "signed_growth_acceleration": "(x3 / x2 - 1) - (x2 / x1 - 1)",
    "coefficient_of_variation_three_levels": (
        "((((x1 - (x1 + x2 + x3) / 3) ** 2 + (x2 - (x1 + x2 + x3) / 3) ** 2 + "
        "(x3 - (x1 + x2 + x3) / 3) ** 2) / 3) ** 0.5) / ((x1 + x2 + x3) / 3)"
    ),
}

_PERIOD_TOKEN_RE = {token: re.compile(rf"\b{token}\b") for token in ("x1", "x2", "x3")}


def transform_missing_period_tokens(transform_id: str) -> tuple[str, ...]:
    expression = TRANSFORM_EXPRESSIONS[transform_id]
    return tuple(token for token, pattern in _PERIOD_TOKEN_RE.items() if not pattern.search(expression))


def get_transform(transform_id: str) -> TransformDefinition:
    return TRANSFORMS[transform_id]


def enabled_transforms() -> list[TransformDefinition]:
    return [t for t in TRANSFORMS.values() if t.enabled]


def compute_transform(transform_id: str, x1: float, x2: float, x3: float) -> float:
    expression = TRANSFORM_EXPRESSIONS[transform_id]
    return eval(expression, {"__builtins__": {}}, {"x1": x1, "x2": x2, "x3": x3})  # noqa: S307


REDUCER_EXPRESSIONS: dict[str, str] = {
    "average": "sum(_values) / len(_values)",
    "median": "sorted(_values)[1]",
    "minimum": "min(_values)",
    "maximum": "max(_values)",
    "range": "max(_values) - min(_values)",
}


def get_reducer_ids() -> tuple[str, ...]:
    return tuple(REDUCER_EXPRESSIONS)


_REDUCER_BUILTINS = {"sum": sum, "min": min, "max": max, "sorted": sorted, "len": len}


def compute_reducer(reducer_id: str, values: tuple[float, float, float]) -> float:
    expression = REDUCER_EXPRESSIONS[reducer_id]
    return eval(expression, {"__builtins__": _REDUCER_BUILTINS}, {"_values": list(values)})  # noqa: S307


ReportScope = Literal["consolidated", "parent"]


@dataclass(frozen=True, slots=True)
class LongitudinalScenarioPlan:

    scenario: Literal["peer_group_longitudinal"]
    metric_family: str
    input_metric_kind: MetricKind
    entities: tuple[str, str, str]
    entity_names: Mapping[str, str]
    periods: tuple[str, str, str]
    report_scope: ReportScope
    per_entity_transform: str
    terminal_reducer: str
    time_basis: str
    measurement_basis: str
    unit: str
    answer_type: AnswerType
    # Keep period handling explicit and deterministic.
    bindings: Mapping[tuple[str, str], ResolvedCell] = field(default_factory=dict)

    def relevant_tables(self) -> list[str]:
        seen: dict[str, None] = {}
        for cell in self.bindings.values():
            seen.setdefault(cell.table_ref, None)
        return list(seen)
