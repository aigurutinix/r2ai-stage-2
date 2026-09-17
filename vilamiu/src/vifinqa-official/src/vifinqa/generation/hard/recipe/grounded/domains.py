
from __future__ import annotations

from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.panel.base import Cube
from vifinqa.generation.panel.catalog import industry_l3_groups, is_excluded_industry

MIN_PEER_ENTITIES = 3
MAX_PEER_ENTITIES = 5
MIN_CROSS_ENTITY_WINDOW_YEARS = 3
MAX_CROSS_ENTITY_WINDOW_YEARS = 4
MIN_SINGLE_ENTITY_WINDOW_YEARS = 3
MAX_SINGLE_ENTITY_WINDOW_YEARS = 5


def _adjacent_year_pairs(years: set[str]) -> list[tuple[str, str]]:
    ordered = sorted(years, key=int)
    return [
        (prior, current)
        for prior, current in zip(ordered, ordered[1:])
        if int(current) - int(prior) == 1
    ]


def _contiguous_windows(
    years: set[str], *, min_len: int, max_len: int
) -> list[tuple[str, ...]]:
    ordered = sorted(years, key=int)
    runs: list[list[str]] = []
    current: list[str] = []
    for year in ordered:
        if not current or int(year) - int(current[-1]) == 1:
            current.append(year)
        else:
            if current:
                runs.append(current)
            current = [year]
    if current:
        runs.append(current)

    windows: list[tuple[str, ...]] = []
    for run in runs:
        upper = min(max_len, len(run))
        for length in range(min_len, upper + 1):
            for start in range(0, len(run) - length + 1):
                windows.append(tuple(run[start : start + length]))
    return windows


def industry_groups_in_cube(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...]]]:
    groups = industry_l3_groups(company_meta)
    cube_tickers = set(cube.tickers())
    result: list[tuple[str, tuple[str, ...]]] = []
    for industry, tickers in sorted(groups.items()):
        present = sorted(tickers & cube_tickers)
        if MIN_PEER_ENTITIES <= len(present) <= MAX_PEER_ENTITIES:
            result.append((industry, tuple(present)))
    return result


def industry_wide_groups_in_cube(
    cube: Cube, company_meta: dict[str, CompanyInfo], *, minimum_entities: int = 3
) -> list[tuple[str, tuple[str, ...]]]:
    """Full industry_l3 groups with no upper-size truncation."""
    groups = industry_l3_groups(company_meta)
    cube_tickers = set(cube.tickers())
    return [
        (industry, tuple(present))
        for industry, tickers in sorted(groups.items())
        if len(present := sorted(tickers & cube_tickers)) >= minimum_entities
    ]


def industry_wide_same_period_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], str]]:
    result: list[tuple[str, tuple[str, ...], str]] = []
    for industry, entities in industry_wide_groups_in_cube(cube, company_meta):
        years = sorted(
            {year for ticker in entities for year in cube.years(ticker)}, key=int
        )
        result.extend((industry, entities, year) for year in years)
    return result


def industry_wide_adjacent_period_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], str, str]]:
    result: list[tuple[str, tuple[str, ...], str, str]] = []
    for industry, entities in industry_wide_groups_in_cube(cube, company_meta):
        years = {year for ticker in entities for year in cube.years(ticker)}
        result.extend(
            (industry, entities, prior, current)
            for prior, current in _adjacent_year_pairs(years)
        )
    return result


def industry_wide_cross_entity_period_window_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    result: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for industry, entities in industry_wide_groups_in_cube(cube, company_meta):
        years = {year for ticker in entities for year in cube.years(ticker)}
        for periods in _contiguous_windows(
            years,
            min_len=MIN_CROSS_ENTITY_WINDOW_YEARS,
            max_len=MAX_CROSS_ENTITY_WINDOW_YEARS,
        ):
            result.append((industry, entities, periods))
    return result


def same_period_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], str]]:
    result: list[tuple[str, tuple[str, ...], str]] = []
    for industry, entities in industry_groups_in_cube(cube, company_meta):
        years: set[str] = set()
        for ticker in entities:
            years.update(cube.years(ticker))
        for year in sorted(years, key=int):
            result.append((industry, entities, year))
    return result


def nonfinancial_same_period_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], str]]:
    """Full non-financial corpus universe by year; coverage filtering happens in the planner."""
    cube_tickers = set(cube.tickers())
    entities = tuple(
        sorted(
            ticker
            for ticker, info in company_meta.items()
            if ticker in cube_tickers and not is_excluded_industry(info)
        )
    )
    years = sorted(
        {year for ticker in entities for year in cube.years(ticker)}, key=int
    )
    return [("nonfinancial_corpus", entities, year) for year in years]


def adjacent_period_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], str, str]]:
    result: list[tuple[str, tuple[str, ...], str, str]] = []
    for industry, entities in industry_groups_in_cube(cube, company_meta):
        years: set[str] = set()
        for ticker in entities:
            years.update(cube.years(ticker))
        for prior, current in _adjacent_year_pairs(years):
            result.append((industry, entities, prior, current))
    return result


def nonfinancial_adjacent_period_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], str, str]]:
    cube_tickers = set(cube.tickers())
    entities = tuple(
        sorted(
            ticker
            for ticker, info in company_meta.items()
            if ticker in cube_tickers and not is_excluded_industry(info)
        )
    )
    years = {year for ticker in entities for year in cube.years(ticker)}
    return [
        ("nonfinancial_corpus", entities, prior, current)
        for prior, current in _adjacent_year_pairs(years)
    ]


def nonfinancial_cross_entity_period_window_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    """Full non-financial corpus universe over deterministic contiguous windows."""
    cube_tickers = set(cube.tickers())
    entities = tuple(
        sorted(
            ticker
            for ticker, info in company_meta.items()
            if ticker in cube_tickers and not is_excluded_industry(info)
        )
    )
    years = {year for ticker in entities for year in cube.years(ticker)}
    return [
        ("nonfinancial_corpus", entities, periods)
        for periods in _contiguous_windows(
            years,
            min_len=MIN_CROSS_ENTITY_WINDOW_YEARS,
            max_len=MAX_CROSS_ENTITY_WINDOW_YEARS,
        )
    ]


def cross_entity_period_window_domains(
    cube: Cube, company_meta: dict[str, CompanyInfo]
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    result: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for industry, entities in industry_groups_in_cube(cube, company_meta):
        years: set[str] = set()
        for ticker in entities:
            years.update(cube.years(ticker))
        for periods in _contiguous_windows(
            years,
            min_len=MIN_CROSS_ENTITY_WINDOW_YEARS,
            max_len=MAX_CROSS_ENTITY_WINDOW_YEARS,
        ):
            result.append((industry, entities, periods))
    return result


def single_entity_period_window_domains(
    cube: Cube,
) -> list[tuple[str, tuple[str, ...], tuple[str, ...]]]:
    result: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
    for ticker in sorted(cube.tickers()):
        for periods in _contiguous_windows(
            set(cube.years(ticker)),
            min_len=MIN_SINGLE_ENTITY_WINDOW_YEARS,
            max_len=MAX_SINGLE_ENTITY_WINDOW_YEARS,
        ):
            result.append((ticker, (ticker,), periods))
    return result
