
from __future__ import annotations

from vifinqa.generation.panel.measurements.evaluator import Distribution, MeasurementFeasibilityReport


def _distribution_json(distribution: Distribution) -> dict[str, float | None]:
    return {
        "min": distribution.min,
        "p25": distribution.p25,
        "median": distribution.median,
        "p75": distribution.p75,
        "max": distribution.max,
    }


def build_report_payload(reports: tuple[MeasurementFeasibilityReport, ...]) -> dict:
    return {
        "measurements": [
            {
                "measurement_id": report.measurement_id,
                "name": report.name,
                "period_basis": report.period_basis,
                "unit": report.unit,
                "candidate_observations": report.candidate_observations,
                "valid_observations": report.valid_observations,
                "coverage_pct": round(report.coverage_pct, 4),
                "invalid_reasons": dict(report.invalid_reasons),
                "distribution": _distribution_json(report.distribution),
                "observations": [
                    {"ticker": obs.ticker, "period": obs.period_key, "value": obs.value}
                    for obs in report.observations
                ],
            }
            for report in reports
        ]
    }


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.4f}"


def render_markdown(reports: tuple[MeasurementFeasibilityReport, ...]) -> str:
    lines = [
        "# Hard Cube — Measurement Feasibility Audit",
        "",
        "Observed coverage and distributions for measurement contracts audited in "
        "`data/catalog_audit02.md`. This report applies no good/bad threshold and removes no "
        "outliers; it supports later Hard-recipe selection rather than predetermining it.",
        "",
        "| Measurement | Unit | Period basis | Candidates | Valid | Coverage | Median | Min | Max |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for report in reports:
        d = report.distribution
        lines.append(
            f"| {report.measurement_id} ({report.name}) | {report.unit} | {report.period_basis} "
            f"| {report.candidate_observations} | {report.valid_observations} "
            f"| {report.coverage_pct:.1f}% | {_fmt(d.median)} | {_fmt(d.min)} | {_fmt(d.max)} |"
        )
    lines.append("")

    for report in reports:
        d = report.distribution
        lines.append(f"## {report.measurement_id} — {report.name}")
        lines.append("")
        lines.append(f"- Unit: `{report.unit}`; period basis: `{report.period_basis}`")
        lines.append(
            f"- Candidate: {report.candidate_observations}; Valid: {report.valid_observations} "
            f"({report.coverage_pct:.1f}%)"
        )
        reasons = ", ".join(f"{code}={count}" for code, count in report.invalid_reasons.items())
        lines.append(f"- Invalid reasons: {reasons if reasons else '(none)'}")
        lines.append(
            f"- Distribution (valid, finite): min={_fmt(d.min)}, p25={_fmt(d.p25)}, "
            f"median={_fmt(d.median)}, p75={_fmt(d.p75)}, max={_fmt(d.max)}"
        )
        lines.append("")

    return "\n".join(lines)
