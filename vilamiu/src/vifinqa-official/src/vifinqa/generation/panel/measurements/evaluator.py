
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.panel.base import Cube
from vifinqa.generation.panel.catalog import is_excluded_industry
from vifinqa.generation.panel.measurements.base import Measurement, MeasurementUnit, PeriodBasis


@dataclass(frozen=True, slots=True)
class MeasurementObservation:

    ticker: str
    period_key: str
    value: float


@dataclass(frozen=True, slots=True)
class Distribution:

    min: float | None
    p25: float | None
    median: float | None
    p75: float | None
    max: float | None


@dataclass(frozen=True, slots=True)
class MeasurementFeasibilityReport:
    measurement_id: str
    name: str
    period_basis: PeriodBasis
    unit: MeasurementUnit
    candidate_observations: int
    valid_observations: int
    coverage_pct: float
    invalid_reasons: dict[str, int]
    distribution: Distribution
    observations: tuple[MeasurementObservation, ...]


def _eligible_tickers(cube: Cube, company_meta: dict[str, CompanyInfo]) -> tuple[str, ...]:
    return tuple(
        sorted(
            ticker
            for ticker in cube.tickers()
            if ticker in company_meta and not is_excluded_industry(company_meta[ticker])
        )
    )


def _adjacent_year_pairs(years: tuple[str, ...]) -> list[tuple[str, str]]:
    ordered = sorted(years, key=int)
    return [(prior, current) for prior, current in zip(ordered, ordered[1:]) if int(current) - int(prior) == 1]


def _candidate_periods(cube: Cube, ticker: str, period_basis: PeriodBasis) -> list[tuple[str, ...]]:
    years = cube.years(ticker)
    if period_basis == "same_period":
        return [(year,) for year in sorted(years)]
    return [pair for pair in _adjacent_year_pairs(years)]


def _period_key(periods: tuple[str, ...]) -> str:
    return periods[0] if len(periods) == 1 else f"{periods[0]}-{periods[1]}"


def _distribution(values: list[float]) -> Distribution:
    finite = sorted(v for v in values if math.isfinite(v))
    if not finite:
        return Distribution(min=None, p25=None, median=None, p75=None, max=None)
    if len(finite) >= 2:
        p25, _, p75 = statistics.quantiles(finite, n=4, method="inclusive")
    else:
        p25 = p75 = finite[0]
    return Distribution(min=finite[0], p25=p25, median=statistics.median(finite), p75=p75, max=finite[-1])


def audit_measurement(
    measurement: Measurement, cube: Cube, company_meta: dict[str, CompanyInfo]
) -> MeasurementFeasibilityReport:
    invalid_reasons: dict[str, int] = {code: 0 for code in measurement.invalid_reason_codes}
    observations: list[MeasurementObservation] = []
    candidate_count = 0

    for ticker in _eligible_tickers(cube, company_meta):
        for periods in _candidate_periods(cube, ticker, measurement.period_basis):
            candidate_count += 1
            outcome = measurement.evaluate(cube, ticker, periods)
            if outcome.invalid_reason is not None:
                invalid_reasons[outcome.invalid_reason] = invalid_reasons.get(outcome.invalid_reason, 0) + 1
                continue
            assert outcome.value is not None
            observations.append(
                MeasurementObservation(ticker=ticker, period_key=_period_key(periods), value=outcome.value)
            )

    observations.sort(key=lambda obs: (obs.ticker, obs.period_key))
    valid_count = len(observations)
    coverage_pct = (valid_count / candidate_count * 100) if candidate_count else 0.0

    return MeasurementFeasibilityReport(
        measurement_id=measurement.measurement_id,
        name=measurement.name,
        period_basis=measurement.period_basis,
        unit=measurement.unit,
        candidate_observations=candidate_count,
        valid_observations=valid_count,
        coverage_pct=coverage_pct,
        invalid_reasons=invalid_reasons,
        distribution=_distribution([obs.value for obs in observations]),
        observations=tuple(observations),
    )


def audit_all(
    measurements: tuple[Measurement, ...], cube: Cube, company_meta: dict[str, CompanyInfo]
) -> tuple[MeasurementFeasibilityReport, ...]:
    return tuple(audit_measurement(measurement, cube, company_meta) for measurement in measurements)
