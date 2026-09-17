
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from vifinqa.generation.hard.recipe.base import MetricTerm, MetricTerms
from vifinqa.generation.panel.base import Cube
from vifinqa.generation.panel.catalog import get_ratio


def longest_contiguous_run(years: set[str]) -> list[str]:
    ints = sorted(int(y) for y in years)
    best: list[int] = []
    current: list[int] = []
    for year in ints:
        if current and year == current[-1] + 1:
            current.append(year)
        else:
            current = [year]
        if len(current) > len(best):
            best = current
    return [str(y) for y in best]


def resolve_metric_terms(
    cube: Cube, ticker: str, period: str, metric_key: str
) -> MetricTerms | None:
    ratio = get_ratio(metric_key)
    if ratio is None:
        cell = cube.cell(ticker, period, metric_key)
        if cell is None:
            return None
        return MetricTerms(
            numerator=(MetricTerm(1.0, metric_key, cell),), denominator=()
        )

    numerator: list[MetricTerm] = []
    for coefficient, key in ratio.numerator:
        cell = cube.cell(ticker, period, key)
        if cell is None:
            return None
        numerator.append(MetricTerm(coefficient, key, cell))

    denominator: list[MetricTerm] = []
    for coefficient, key in ratio.denominator:
        cell = cube.cell(ticker, period, key)
        if cell is None:
            return None
        denominator.append(MetricTerm(coefficient, key, cell))

    return MetricTerms(numerator=tuple(numerator), denominator=tuple(denominator))


@dataclass(frozen=True, slots=True)
class MetricResolverContext:
    cube: Cube
    ticker: str
    periods: tuple[str, ...]
    current_period: str


WindowMetricResolver = Callable[[MetricResolverContext], MetricTerms | None]


def _adjacent_prior_current(
    periods: tuple[str, ...], current_period: str
) -> tuple[str, str] | None:
    if current_period not in periods:
        return None
    current_index = periods.index(current_period)
    if current_index == 0:
        return None
    prior = periods[current_index - 1]
    try:
        if int(current_period) - int(prior) != 1:
            return None
    except ValueError:
        return None
    return prior, current_period


def _resolve_average_balance_ratio_terms(
    context: MetricResolverContext, *, denominator_metric_key: str
) -> MetricTerms | None:
    window = _adjacent_prior_current(context.periods, context.current_period)
    if window is None:
        return None
    prior, current = window
    npat_cell = context.cube.cell(context.ticker, current, "kqkd:60")
    denominator_current_cell = context.cube.cell(
        context.ticker, current, denominator_metric_key
    )
    denominator_prior_cell = context.cube.cell(
        context.ticker, prior, denominator_metric_key
    )
    if None in (npat_cell, denominator_current_cell, denominator_prior_cell):
        return None
    assert (
        npat_cell is not None
        and denominator_current_cell is not None
        and denominator_prior_cell is not None
    )
    terms = MetricTerms(
        numerator=(MetricTerm(1.0, "kqkd:60", npat_cell),),
        denominator=(
            MetricTerm(0.5, denominator_metric_key, denominator_prior_cell),
            MetricTerm(0.5, denominator_metric_key, denominator_current_cell),
        ),
        requires_positive_denominator=True,
    )
    denominator_value = sum(
        term.coefficient * term.cell.value for term in terms.denominator
    )
    if denominator_value <= 0:
        return None
    return terms


def resolve_roa_terms(context: MetricResolverContext) -> MetricTerms | None:
    return _resolve_average_balance_ratio_terms(
        context, denominator_metric_key="cdkt:270"
    )


def resolve_roe_terms(context: MetricResolverContext) -> MetricTerms | None:
    return _resolve_average_balance_ratio_terms(
        context, denominator_metric_key="cdkt:400"
    )


def resolve_inventory_days_terms(context: MetricResolverContext) -> MetricTerms | None:
    window = _adjacent_prior_current(context.periods, context.current_period)
    if window is None:
        return None
    prior, current = window
    inventory_prior = context.cube.cell(context.ticker, prior, "cdkt:140")
    inventory_current = context.cube.cell(context.ticker, current, "cdkt:140")
    cost_of_goods_sold = context.cube.cell(context.ticker, current, "kqkd:11")
    if None in (inventory_prior, inventory_current, cost_of_goods_sold):
        return None
    assert inventory_prior is not None and inventory_current is not None
    assert cost_of_goods_sold is not None
    if (
        inventory_prior.value < 0
        or inventory_current.value < 0
        or abs(cost_of_goods_sold.value) <= 0
    ):
        return None
    return MetricTerms(
        numerator=(
            MetricTerm(182.5, "cdkt:140", inventory_prior),
            MetricTerm(182.5, "cdkt:140", inventory_current),
        ),
        denominator=(MetricTerm(1.0, "kqkd:11", cost_of_goods_sold),),
        requires_positive_denominator=True,
    )


def resolve_sga_expense_terms(context: MetricResolverContext) -> MetricTerms | None:
    selling = context.cube.cell(context.ticker, context.current_period, "kqkd:25")
    administration = context.cube.cell(
        context.ticker, context.current_period, "kqkd:26"
    )
    if selling is None or administration is None:
        return None
    return MetricTerms(
        numerator=(
            MetricTerm(1.0, "kqkd:25", selling),
            MetricTerm(1.0, "kqkd:26", administration),
        ),
        denominator=(),
    )


def resolve_asset_turnover_avg_terms(
    context: MetricResolverContext,
) -> MetricTerms | None:
    window = _adjacent_prior_current(context.periods, context.current_period)
    if window is None:
        return None
    prior, current = window
    revenue = context.cube.cell(context.ticker, current, "kqkd:10")
    assets_prior = context.cube.cell(context.ticker, prior, "cdkt:270")
    assets_current = context.cube.cell(context.ticker, current, "cdkt:270")
    if revenue is None or assets_prior is None or assets_current is None:
        return None
    terms = MetricTerms(
        numerator=(MetricTerm(1.0, "kqkd:10", revenue),),
        denominator=(
            MetricTerm(0.5, "cdkt:270", assets_prior),
            MetricTerm(0.5, "cdkt:270", assets_current),
        ),
        requires_positive_denominator=True,
    )
    return terms if terms.value is not None else None


def resolve_equity_multiplier_terms(
    context: MetricResolverContext,
) -> MetricTerms | None:
    window = _adjacent_prior_current(context.periods, context.current_period)
    if window is None:
        return None
    prior, current = window
    assets_prior = context.cube.cell(context.ticker, prior, "cdkt:270")
    assets_current = context.cube.cell(context.ticker, current, "cdkt:270")
    equity_prior = context.cube.cell(context.ticker, prior, "cdkt:400")
    equity_current = context.cube.cell(context.ticker, current, "cdkt:400")
    if None in (assets_prior, assets_current, equity_prior, equity_current):
        return None
    assert assets_prior is not None and assets_current is not None
    assert equity_prior is not None and equity_current is not None
    terms = MetricTerms(
        numerator=(
            MetricTerm(0.5, "cdkt:270", assets_prior),
            MetricTerm(0.5, "cdkt:270", assets_current),
        ),
        denominator=(
            MetricTerm(0.5, "cdkt:400", equity_prior),
            MetricTerm(0.5, "cdkt:400", equity_current),
        ),
        requires_positive_denominator=True,
    )
    return terms if terms.value is not None else None


def resolve_net_working_capital_terms(
    context: MetricResolverContext,
) -> MetricTerms | None:
    current_assets = context.cube.cell(
        context.ticker, context.current_period, "cdkt:100"
    )
    current_liabilities = context.cube.cell(
        context.ticker, context.current_period, "cdkt:310"
    )
    if current_assets is None or current_liabilities is None:
        return None
    return MetricTerms(
        numerator=(
            MetricTerm(1.0, "cdkt:100", current_assets),
            MetricTerm(-1.0, "cdkt:310", current_liabilities),
        ),
        denominator=(),
    )


WINDOW_METRIC_RESOLVERS: dict[str, WindowMetricResolver] = {
    "roa": resolve_roa_terms,
    "roe": resolve_roe_terms,
    "inventory_days": resolve_inventory_days_terms,
    "sga_expense": resolve_sga_expense_terms,
    "asset_turnover_avg": resolve_asset_turnover_avg_terms,
    "equity_multiplier": resolve_equity_multiplier_terms,
    "net_working_capital": resolve_net_working_capital_terms,
    "operating_accruals_ratio": lambda context: resolve_operating_accruals_terms(
        context.cube,
        context.ticker,
        context.periods[context.periods.index(context.current_period) - 1]
        if context.current_period in context.periods
        and context.periods.index(context.current_period) > 0
        else "",
        context.current_period,
    ),
}


def resolve_window_metric_terms(
    cube: Cube,
    ticker: str,
    periods: tuple[str, ...],
    current_period: str,
    metric_key: str,
) -> MetricTerms | None:
    resolver = WINDOW_METRIC_RESOLVERS.get(metric_key)
    if resolver is not None:
        return resolver(
            MetricResolverContext(
                cube=cube, ticker=ticker, periods=periods, current_period=current_period
            )
        )
    return resolve_metric_terms(cube, ticker, current_period, metric_key)


def resolve_operating_accruals_terms(
    cube: Cube, ticker: str, period_prior: str, period: str
) -> MetricTerms | None:
    npat_cell = cube.cell(ticker, period, "kqkd:60")
    cfo_cell = cube.cell(ticker, period, "lctt:20")
    assets_t_cell = cube.cell(ticker, period, "cdkt:270")
    assets_prior_cell = cube.cell(ticker, period_prior, "cdkt:270")
    if None in (npat_cell, cfo_cell, assets_t_cell, assets_prior_cell):
        return None
    assert npat_cell is not None and cfo_cell is not None
    assert assets_t_cell is not None and assets_prior_cell is not None
    numerator = (
        MetricTerm(1.0, "kqkd:60", npat_cell),
        MetricTerm(-1.0, "lctt:20", cfo_cell),
    )
    denominator = (
        MetricTerm(0.5, "cdkt:270", assets_t_cell),
        MetricTerm(0.5, "cdkt:270", assets_prior_cell),
    )
    return MetricTerms(numerator=numerator, denominator=denominator)
