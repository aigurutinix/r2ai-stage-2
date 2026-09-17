
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from vifinqa.generation.panel.base import Cube

PeriodBasis = Literal["same_period", "adjacent_period"]
MeasurementUnit = Literal["percentage", "percentage_point", "number"]


@dataclass(frozen=True, slots=True)
class MeasurementOutcome:

    value: float | None
    invalid_reason: str | None


class Measurement(Protocol):

    measurement_id: str
    name: str
    period_basis: PeriodBasis
    unit: MeasurementUnit
    invalid_reason_codes: tuple[str, ...]

    def evaluate(self, cube: Cube, ticker: str, periods: tuple[str, ...]) -> MeasurementOutcome:
        ...
