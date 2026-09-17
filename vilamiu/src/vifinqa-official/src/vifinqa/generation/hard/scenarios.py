
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from vifinqa.generation.hard.schemas import StepOperation

HardScenarioName = Literal["p1_filter_aggregate", "p3_period_selector"]


@dataclass(frozen=True, slots=True)
class HardScenarioSpec:
    name: HardScenarioName
    min_entities: int
    allowed_final_operations: frozenset[StepOperation]
    min_periods: int = 1


HARD_SCENARIOS: dict[HardScenarioName, HardScenarioSpec] = {
    "p1_filter_aggregate": HardScenarioSpec(
        name="p1_filter_aggregate",
        min_entities=3,
        allowed_final_operations=frozenset({"count", "sum"}),
    ),
    "p3_period_selector": HardScenarioSpec(
        name="p3_period_selector",
        min_entities=1,
        allowed_final_operations=frozenset({"lookup"}),
        min_periods=3,
    ),
}


def get_hard_scenario(name: HardScenarioName) -> HardScenarioSpec:
    return HARD_SCENARIOS[name]
