"""Constrained, source-bound compiler for common financial-statement queries.

This is a resilience layer for live demos, not a replacement for the open
model.  It recognizes a deliberately small grammar, reads normalized primary
statement cells, and emits Pandas that re-reads the exact source coordinates.
Unknown, ambiguous or multi-entity questions return ``None`` and continue to
the model/refusal path.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from kingpro.answering.pandas_answer import requested_unit
from kingpro.financial.panel_metrics import RATIO_FORMULAS, RAW_METRICS
from kingpro.financial.statement_cube import StatementCell
from kingpro.retrieval.bm25_index import fold


_ALIASES: tuple[tuple[str, str], ...] = (
    (
        "ty le luu chuyen tien thuan tu hoat dong kinh doanh tren loi nhuan thuan tu hoat dong kinh doanh",
        "cfo_to_operating_profit",
    ),
    (
        "ty so luu chuyen tien thuan tu hoat dong kinh doanh tren loi nhuan thuan tu hoat dong kinh doanh",
        "cfo_to_operating_profit",
    ),
    ("he so kha nang thanh toan lai vay", "interest_coverage"),
    ("muc thay doi bien loi nhuan gop", "gross_margin_change_pp"),
    ("chenh lech bien loi nhuan gop", "gross_margin_change_pp"),
    ("ty suat sinh loi tren von chu so huu", "roe_pct"),
    ("loi nhuan sau thue tren von chu so huu binh quan", "roe_pct"),
    ("ty suat sinh loi tren tong tai san", "roa_pct"),
    ("loi nhuan sau thue tren tong tai san binh quan", "roa_pct"),
    ("luu chuyen tien thuan tu hoat dong kinh doanh", "cfo"),
    ("dong tien thuan tu hoat dong kinh doanh", "cfo"),
    ("ty le no phai tra tren von chu so huu", "liabilities_to_equity"),
    ("he so no tren von chu so huu", "liabilities_to_equity"),
    ("ty le no phai tra tren tong tai san", "liabilities_to_assets_pct"),
    ("ty le no tren tong tai san", "debt_to_assets_pct"),
    ("ty le hang ton kho tren tong tai san", "inventory_to_assets_pct"),
    ("he so thanh toan hien hanh", "current_ratio"),
    ("he so thanh toan nhanh", "quick_ratio"),
    ("bien loi nhuan tu hoat dong kinh doanh", "operating_margin_pct"),
    ("bien loi nhuan hoat dong", "operating_margin_pct"),
    ("bien loi nhuan sau thue", "net_margin_pct"),
    ("bien loi nhuan rong", "net_margin_pct"),
    ("ty le loi nhuan sau thue tren doanh thu thuan", "net_margin_pct"),
    ("ty so loi nhuan sau thue tren doanh thu thuan", "net_margin_pct"),
    ("bien loi nhuan gop", "gross_margin_pct"),
    ("ty suat loi nhuan gop", "gross_margin_pct"),
    ("loi nhuan gop tren doanh thu thuan", "gross_margin_pct"),
    ("loi nhuan sau thue tren tong tai san cuoi nam", "npat_to_assets_pct"),
    ("loi nhuan sau thue tren tong tai san", "npat_to_assets_pct"),
    ("ty trong tai san ngan han trong tong nguon von", "current_assets_to_assets_pct"),
    ("ty trong tai san ngan han tren tong nguon von", "current_assets_to_assets_pct"),
    ("ty suat loi nhuan truoc thue", "pbt_margin_pct"),
    ("vong quay von chu so huu", "equity_turnover_avg"),
    ("ty le chi phi quan ly doanh nghiep tren doanh thu thuan", "admin_expense_to_revenue_pct"),
    ("ty le chi phi ban hang tren doanh thu thuan", "selling_expense_to_revenue_pct"),
    ("ty le chi phi ban hang va quan ly doanh nghiep tren doanh thu thuan", "sga_intensity_pct"),
    ("chi phi ban hang va quan ly doanh nghiep tren doanh thu thuan", "sga_intensity_pct"),
    ("ty le vay ngan han tren von chu so huu", "short_term_borrowings_to_equity_pct"),
    ("ty le no ngan han tren von chu so huu", "current_liabilities_to_equity_pct"),
    ("ty so no ngan han tren von chu so huu", "current_liabilities_to_equity"),
    ("ty trong chi phi tai chinh tren doanh thu thuan", "finance_expense_to_revenue_pct"),
    ("ti trong chi phi tai chinh tren doanh thu thuan", "finance_expense_to_revenue_pct"),
    ("ty le chi phi tai chinh tren doanh thu thuan", "finance_expense_to_revenue_pct"),
    ("ty le doanh thu hoat dong tai chinh tren chi phi tai chinh", "finance_revenue_to_expense_pct"),
    ("ty le gia von hang ban tren doanh thu thuan", "cogs_to_revenue_pct"),
    ("ty le dau tu ngan han tren tien va tuong duong tien", "short_term_investments_to_cash_pct"),
    ("ty suat loi nhuan rong", "net_margin_pct"),
    ("tang truong doanh thu thuan", "revenue_growth_pct"),
    ("toc do tang truong doanh thu", "revenue_growth_pct"),
    ("von luu dong rong", "net_working_capital"),
    ("chi phi ban hang va quan ly", "sga_expense"),
    ("chi phi quan ly doanh nghiep", "admin_expense"),
    ("chi phi ban hang", "selling_expense"),
    ("chi phi lai vay", "interest_expense"),
    ("chi phi thue thu nhap hien hanh", "current_tax_expense"),
    ("chi phi thue tndn hien hanh", "current_tax_expense"),
    ("ket qua hoat dong tai chinh rong", "net_finance_result"),
    ("ket qua thuan tu hoat dong tai chinh", "net_finance_result"),
    ("loi nhuan thuan tu hoat dong tai chinh", "net_finance_result"),
    ("lai thuan hoat dong tai chinh", "net_finance_result"),
    ("lai rong tu hoat dong tai chinh", "net_finance_result"),
    ("thu nhap thuan tu hoat dong khac", "net_other_result"),
    ("thu nhap khac thuan", "net_other_result"),
    ("loi nhuan khac thuan", "net_other_result"),
    ("chi phi tai chinh", "finance_expense"),
    ("doanh thu hoat dong tai chinh", "finance_revenue"),
    ("loi nhuan thuan tu hoat dong kinh doanh", "operating_profit"),
    ("loi nhuan truoc thue", "pbt"),
    ("loi nhuan ke toan sau thue tndn", "npat"),
    ("loi nhuan thuan sau thue", "npat"),
    ("loi nhuan sau thue", "npat"),
    ("loi nhuan gop", "gross_profit"),
    ("gia von hang ban", "cogs"),
    ("gia von ban hang", "cogs"),
    ("doanh thu thuan", "revenue"),
    ("tong cong tai san", "total_assets"),
    ("tong tai san", "total_assets"),
    ("tong cong nguon von", "total_assets"),
    ("tong nguon von", "total_assets"),
    ("tai san ngan han", "current_assets"),
    ("tai san dai han", "long_term_assets"),
    ("cac khoan phai thu ngan han", "short_term_receivables"),
    ("cac khoan phai thu dai han", "long_term_receivables"),
    ("hang ton kho", "inventory"),
    ("tien va cac khoan tuong duong tien", "cash"),
    ("tien va tuong duong tien", "cash"),
    ("tong no phai tra", "liabilities"),
    ("no phai tra", "liabilities"),
    ("no ngan han", "current_liabilities"),
    ("vay va no thue tai chinh ngan han", "short_term_borrowings"),
    ("so du vay ngan han", "short_term_borrowings"),
    ("von chu so huu", "equity"),
    ("von gop cua chu so huu", "contributed_capital"),
    ("von co phan", "contributed_capital"),
    ("roe", "roe_pct"),
    ("roa", "roa_pct"),
)

_MONETARY_DERIVED = {
    "net_working_capital",
    "sga_expense",
    "net_finance_result",
    "net_other_result",
}
_COST_KEYS = {"kqkd:11", "kqkd:22", "kqkd:23", "kqkd:25", "kqkd:26", "kqkd:32", "kqkd:51"}
_BLOCKED_INTENTS = re.compile(r"\b(?:cao nhat|thap nhat|lon nhat|nho nhat|trung binh|binh quan|so sanh|so voi|giua)\b")
_QUALIFIER_BLOCKS = re.compile(
    r"\b(?:dau nam|dau ky|so dau nam|thuoc|neu|bang do nhay|"
    r"gia su|kich ban|nam sau|nam ke tiep|tiep theo|co dong|theo nganh|"
    r"bo phan|du an|ben lien quan|cua ong|cua ba|"
    r"thue thu nhap hoan lai|chua phan phoi)\b"
)
_BEGINNING_DATE_BLOCKS = re.compile(
    r"(?:^|\D)0?1[/-]0?1(?:\D|$)|\bngay\s+0?1\s+thang\s+0?1\b"
)

# A separate, deliberately tiny grammar for questions of the form “at the
# company/year with selector X highest/lowest, what is target Y?”.  Keep these
# aliases separate from the direct compiler: a phrase being safe as a selector
# in this exact topology does not make it safe for every one-cell question.
_EXTREME_PANEL_ALIASES: tuple[tuple[str, str], ...] = (
    (
        "ty le tong cua loi nhuan truoc thue va chi phi lai vay tren chi phi lai vay",
        "interest_coverage",
    ),
    (
        "tong loi nhuan truoc thue va chi phi lai vay gap bao nhieu lan chi phi lai vay",
        "interest_coverage",
    ),
    ("loi nhuan truoc lai vay va thue trong nam do gap bao nhieu lan chi phi lai vay", "interest_coverage"),
    ("he so kha nang thanh toan lai vay", "interest_coverage"),
    (
        "hang ton kho cuoi nam chiem bao nhieu phan tram tong tai san cuoi nam do",
        "inventory_to_assets_pct",
    ),
    ("ty trong tai san dai han tren tong tai san", "long_term_assets_share_pct"),
    ("vong quay tong tai san", "asset_turnover_avg"),
    ("ty le cfo tren loi nhuan sau thue", "cfo_to_npat"),
    ("ty le cfo tren lnst", "cfo_to_npat"),
    ("ty le no phai tra chia cho von chu so huu", "liabilities_to_equity"),
    ("ty le no phai tra tren von chu so huu", "liabilities_to_equity"),
    ("he so no phai tra tren von chu so huu", "liabilities_to_equity"),
    ("ty so d/e", "liabilities_to_equity"),
    ("he so thanh toan hien hanh", "current_ratio"),
    ("he so thanh toan nhanh", "quick_ratio"),
    ("doanh thu thuan", "revenue"),
)
_EXTREME_RE = re.compile(r"\b(?:cao nhat|thap nhat|lon nhat|nho nhat)\b")
_EXTREME_PANEL_BLOCKS = re.compile(
    r"\b(?:trung vi|binh quan|trung binh|cao hon|thap hon|lon hon|nho hon|"
    r"duong|am|nam sau|nam ke tiep|tiep theo|gia su|kich ban|cagr|"
    r"toc do tang|tang truong|muc thay doi|chenh lech|top\s*\d+|"
    r"bao nhieu doanh nghiep|dong thoi|trong\s+\d+\s+doanh nghiep|"
    r"tren\s+\d|duoi\s+\d)\b"
)
_SIGNED_FILTER_EXTREME_BLOCKS = re.compile(
    r"\b(?:trung\s+vi|binh\s+quan|trung\s+binh|cao\s+hon|thap\s+hon|"
    r"lon\s+hon|nho\s+hon|gia\s+su|kich\s+ban|cagr|top\s*\d+|"
    r"bao\s+nhieu\s+doanh\s+nghiep|dong\s+thoi|tren\s+\d|duoi\s+\d)\b",
    re.IGNORECASE,
)
_UNIVERSE_INVENTORY_DECLINE_CFO_MARGIN_RE = re.compile(
    r"\btrong\s+cac\s+cong\s+ty\b.*\bhang\s+ton\s+kho\b.*"
    r"\bgiam\s+it\s+nhat\s+(?P<threshold>\d+(?:[.,]\d+)?)\s*"
    r"(?:%(?=\s|$)|phan\s+tram\b).*"
    r"\b(?:ty\s+le|ty\s+so)\s+cfo\s*(?:/|tren)\s*"
    r"doanh\s+thu\s+thuan\b.*\bcao\s+nhat\b",
    re.IGNORECASE | re.DOTALL,
)
_REQUIRED_KEYS = (
    set(RAW_METRICS.values())
    | {key for top, bottom, _multiplier in RATIO_FORMULAS.values() for key in (*top, *bottom)}
    | {"kqkd:10", "kqkd:20", "kqkd:25", "kqkd:26", "kqkd:31", "kqkd:60", "cdkt:100", "cdkt:270", "cdkt:310", "cdkt:400"}
)


@dataclass(frozen=True)
class CompiledFinancialQuery:
    metric: str
    answer: float
    unit: str
    pandas_query: str
    csv_paths: dict[str, str]
    table_refs: list[str]
    source_cells: list[dict]

    def submission_evidence(self) -> list[dict[str, str]]:
        """Map each generated ``df`` variable to its exact source table.

        ``csv_paths`` and ``table_refs`` are both de-duplicated in first-read
        order while compiling, so their positions form a stable one-to-one
        binding.  Refuse an inconsistent object instead of guessing provenance.
        """
        variables = list(self.csv_paths)
        if len(variables) != len(self.table_refs):
            raise ValueError("compiler source-path/table-ref cardinality mismatch")
        return [
            {"variable": variable, "table_ref": table_ref}
            for variable, table_ref in zip(variables, self.table_refs)
        ]


class DeterministicFinancialCompiler:
    """Compile one-company/one-year standard metrics from verified cells."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.cube_path = self.root / "build" / "statement_cube.jsonl"
        self.catalog_path = self.root / "build" / "catalog.jsonl"
        self._cells: dict[tuple[str, str, str, str], StatementCell] | None = None

    @property
    def available(self) -> bool:
        return self.cube_path.is_file() and self.catalog_path.is_file()

    def _load(self) -> dict[tuple[str, str, str, str], StatementCell]:
        if self._cells is None:
            cells: dict[tuple[str, str, str, str], StatementCell] = {}
            with self.cube_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("metric_key") not in _REQUIRED_KEYS:
                        continue
                    cell = StatementCell(**row)
                    cells[(cell.ticker, cell.year, cell.scope, cell.metric_key)] = cell
                    if cell.physical_scope in {"consolidated", "separate"}:
                        cells[
                            (
                                cell.ticker,
                                cell.year,
                                f"physical:{cell.physical_scope}",
                                cell.metric_key,
                            )
                        ] = cell
            self._cells = cells
        return self._cells

    @staticmethod
    def _scope(value: str) -> str:
        normalized = fold(value)
        # Explicit parent/separate intent follows the physical report masthead.
        # Unqualified questions preserve the catalog/default consolidated index,
        # matching the measured benchmark convention when containers are swapped.
        return (
            "physical:separate"
            if "cong ty me" in normalized or "rieng" in normalized
            else "consolidated"
        )

    @staticmethod
    def _detect_metric(question: str) -> str | None:
        normalized = fold(question)
        matches: list[tuple[int, str]] = []
        for phrase, metric in _ALIASES:
            if re.search(rf"\b{re.escape(phrase)}\b", normalized):
                matches.append((len(phrase), metric))
        if not matches:
            return None
        longest = max(length for length, _metric in matches)
        winners = {metric for length, metric in matches if length == longest}
        return next(iter(winners)) if len(winners) == 1 else None

    @staticmethod
    def supported_metrics() -> list[str]:
        return sorted({metric for _phrase, metric in _ALIASES})

    @staticmethod
    def _cell_value(cell: StatementCell) -> float:
        value = float(cell.value)
        return abs(value) if cell.metric_key in _COST_KEYS else value

    def _cell(
        self,
        cells: dict[tuple[str, str, str, str], StatementCell],
        ticker: str,
        year: int,
        key: str,
        scope: str,
    ) -> StatementCell | None:
        return cells.get((str(ticker).upper(), str(year), scope, key))

    def _operands(
        self,
        source_index: dict[tuple[str, str, str, str], StatementCell],
        ticker: str,
        year: int,
        scope: str,
        metric: str,
    ) -> tuple[list[StatementCell], str, float, str] | None:
        cells: list[StatementCell] = []

        def add(target_year: int, key: str) -> str | None:
            cell = self._cell(source_index, ticker, target_year, key, scope)
            if cell is None:
                return None
            cells.append(cell)
            return f"v{len(cells)}"

        unit_name, unit_multiplier = requested_unit(metric)
        if metric in RAW_METRICS and metric != "eps":
            variable = add(year, RAW_METRICS[metric])
            if variable is None:
                return None
            return cells, f"{variable} / output_scale", float(unit_multiplier), "currency"

        if metric in RATIO_FORMULAS:
            numerator, denominator, multiplier = RATIO_FORMULAS[metric]
            top_terms: list[str] = []
            bottom_terms: list[str] = []
            for key, coefficient in numerator.items():
                variable = add(year, key)
                if variable is None:
                    return None
                top_terms.append(f"({coefficient:g} * {variable})")
            for key, coefficient in denominator.items():
                variable = add(year, key)
                if variable is None:
                    return None
                bottom_terms.append(f"({coefficient:g} * {variable})")
            expression = f"(({'+'.join(top_terms)}) / ({'+'.join(bottom_terms)}) * {multiplier:g})"
            unit = "phần trăm" if multiplier == 100 else "lần"
            return cells, expression, 1.0, unit

        if metric == "revenue_growth_pct":
            current = add(year, "kqkd:10")
            previous = add(year - 1, "kqkd:10")
            if current is None or previous is None:
                return None
            return cells, f"(({current} / {previous}) - 1) * 100", 1.0, "phần trăm"

        if metric == "gross_margin_change_pp":
            current_gross = add(year, "kqkd:20")
            current_revenue = add(year, "kqkd:10")
            previous_gross = add(year - 1, "kqkd:20")
            previous_revenue = add(year - 1, "kqkd:10")
            if None in (current_gross, current_revenue, previous_gross, previous_revenue):
                return None
            expression = (
                f"({current_gross} / {current_revenue} * 100) - "
                f"({previous_gross} / {previous_revenue} * 100)"
            )
            return cells, expression, 1.0, "điểm phần trăm"

        if metric in {"roe_pct", "roa_pct"}:
            profit = add(year, "kqkd:60")
            balance_key = "cdkt:400" if metric == "roe_pct" else "cdkt:270"
            current = add(year, balance_key)
            previous = add(year - 1, balance_key)
            if None in (profit, current, previous):
                return None
            return cells, f"{profit} / (({current} + {previous}) / 2) * 100", 1.0, "phần trăm"

        if metric == "equity_turnover_avg":
            revenue = add(year, "kqkd:10")
            current = add(year, "cdkt:400")
            previous = add(year - 1, "cdkt:400")
            if revenue is None or current is None or previous is None:
                return None
            return cells, f"{revenue} / (({current} + {previous}) / 2)", 1.0, "lần"

        if metric == "asset_turnover_avg":
            revenue = add(year, "kqkd:10")
            current = add(year, "cdkt:270")
            previous = add(year - 1, "cdkt:270")
            if revenue is None or current is None or previous is None:
                return None
            return cells, f"{revenue} / (({current} + {previous}) / 2)", 1.0, "lần"

        if metric == "net_working_capital":
            assets = add(year, "cdkt:100")
            liabilities = add(year, "cdkt:310")
            if assets is None or liabilities is None:
                return None
            return cells, f"({assets} - {liabilities}) / output_scale", float(unit_multiplier), "currency"

        if metric == "sga_expense":
            selling = add(year, "kqkd:25")
            admin = add(year, "kqkd:26")
            if selling is None or admin is None:
                return None
            return cells, f"({selling} + {admin}) / output_scale", float(unit_multiplier), "currency"
        if metric == "net_finance_result":
            revenue = add(year, "kqkd:21")
            expense = add(year, "kqkd:22")
            if revenue is None or expense is None:
                return None
            return cells, f"({revenue} - {expense}) / output_scale", float(unit_multiplier), "currency"
        if metric == "net_other_result":
            income = add(year, "kqkd:31")
            expense = add(year, "kqkd:32")
            if income is None or expense is None:
                return None
            return cells, f"({income} - {expense}) / output_scale", float(unit_multiplier), "currency"
        return None

    @staticmethod
    def _extreme_panel_metrics(normalized: str) -> tuple[str, str, bool] | None:
        """Return selector, target and direction for the strict panel grammar."""
        extremes = list(_EXTREME_RE.finditer(normalized))
        # “total assets average” is the denominator definition of asset
        # turnover, not a request to average candidate outputs.
        qualifier_text = normalized.replace(
            "tinh theo tong tai san binh quan", ""
        ).replace("theo tong tai san binh quan", "")
        if len(extremes) != 1 or _EXTREME_PANEL_BLOCKS.search(qualifier_text):
            return None
        extreme = extremes[0]
        occurrences: list[tuple[int, int, str, int]] = []
        for phrase, metric in _EXTREME_PANEL_ALIASES:
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", normalized):
                occurrences.append((match.start(), match.end(), metric, len(phrase)))
        if not occurrences:
            return None

        # Collapse overlapping aliases for the same metric by keeping the most
        # specific phrase. This prevents a long interest-coverage definition
        # from being mistaken for several separate metrics.
        collapsed: list[tuple[int, int, str, int]] = []
        for occurrence in sorted(occurrences, key=lambda item: (-item[3], item[0])):
            start, end, metric, _length = occurrence
            if any(
                prior_metric == metric and not (end <= prior_start or start >= prior_end)
                for prior_start, prior_end, prior_metric, _prior_length in collapsed
            ):
                continue
            collapsed.append(occurrence)

        before = [item for item in collapsed if item[1] <= extreme.start()]
        if not before:
            return None
        # The selector is the financial phrase immediately governing the single
        # extreme token. All other detected phrases must name one target metric.
        selector_occurrence = max(before, key=lambda item: (item[1], item[3]))
        selector = selector_occurrence[2]
        targets = {item[2] for item in collapsed if item[2] != selector}
        if len(targets) != 1:
            return None
        target = next(iter(targets))
        return selector, target, "cao" in extreme.group(0) or "lon" in extreme.group(0)

    @staticmethod
    def _signed_filter_extreme_metrics(
        normalized: str,
    ) -> tuple[list[tuple[str, int]], str, str, bool] | None:
        """Parse ``metric positive/negative -> extreme selector -> target``.

        This intentionally excludes medians, numeric thresholds, scenarios and
        multi-condition predicates.  Those require a richer grammar and must
        continue to the model instead of being approximated here.
        """
        extremes = list(_EXTREME_RE.finditer(normalized))
        if len(extremes) != 1 or _SIGNED_FILTER_EXTREME_BLOCKS.search(normalized):
            return None
        extreme = extremes[0]
        occurrences: list[tuple[int, int, str, int]] = []
        for phrase, metric in _ALIASES:
            for match in re.finditer(rf"\b{re.escape(phrase)}\b", normalized):
                occurrences.append((match.start(), match.end(), metric, len(phrase)))
        # Prefer the longest alias at every textual span.  Unlike the ordinary
        # direct compiler, a long ratio must also suppress nested raw metrics:
        # they are operands of the ratio, not independent query roles.
        collapsed: list[tuple[int, int, str, int]] = []
        for occurrence in sorted(occurrences, key=lambda item: (-item[3], item[0])):
            start, end, _metric, _length = occurrence
            if any(
                not (end <= prior_start or start >= prior_end)
                for prior_start, prior_end, _prior_metric, _prior_length in collapsed
            ):
                continue
            collapsed.append(occurrence)
        before = [item for item in collapsed if item[1] <= extreme.start()]
        if not before:
            return None
        selector_occurrence = max(before, key=lambda item: (item[1], item[3]))
        selector_metric = selector_occurrence[2]

        filters: list[tuple[str, int]] = []
        for start, end, metric, _length in collapsed:
            if (start, end) == selector_occurrence[:2]:
                continue
            nearby = normalized[end : end + 55]
            sign_match = re.search(r"\b(duong|am)\b", nearby)
            if sign_match is None:
                continue
            signed = (metric, 1 if sign_match.group(1) == "duong" else -1)
            if signed not in filters:
                filters.append(signed)
        if len(filters) != 1:
            return None

        excluded = {selector_metric, filters[0][0]}
        targets = {item[2] for item in collapsed if item[2] not in excluded}
        if len(targets) != 1:
            return None
        target_metric = next(iter(targets))
        descending = "cao" in extreme.group(0) or "lon" in extreme.group(0)
        return filters, selector_metric, target_metric, descending

    def _compile_extreme_panel(
        self,
        question: str,
        facets: dict,
    ) -> CompiledFinancialQuery | None:
        tickers = [str(value).upper() for value in facets.get("tickers", [])]
        years = [int(value) for value in facets.get("years", [])]
        one_company_many_years = len(tickers) == 1 and len(years) >= 2
        many_companies_one_year = len(tickers) >= 2 and len(years) == 1
        if not (one_company_many_years or many_companies_one_year):
            return None
        normalized = fold(question)
        filter_specs: list[tuple[str, int]] = []
        detected = self._extreme_panel_metrics(normalized)
        if detected is None and many_companies_one_year:
            filtered = self._signed_filter_extreme_metrics(normalized)
            if filtered is not None:
                filter_specs, selector_metric, target_metric, descending = filtered
            else:
                return None
        elif detected is not None:
            selector_metric, target_metric, descending = detected
        else:
            return None
        scope = self._scope(str(facets.get("scope", "")))
        source_index = self._load()
        candidates = (
            [(tickers[0], year) for year in years]
            if one_company_many_years
            else [(ticker, years[0]) for ticker in tickers]
        )

        unit_name, unit_multiplier = requested_unit(question)
        selector_expressions: list[str] = []
        target_expressions: list[str] = []
        filter_expressions: list[list[str]] = []
        all_cells: list[StatementCell] = []
        selector_values: list[float] = []
        target_values: list[float] = []
        filter_values: list[list[float]] = []

        def append_expression(
            built: tuple[list[StatementCell], str, float, str],
            *,
            output_scale: float,
        ) -> tuple[str, float]:
            cells, expression, _placeholder, _kind = built
            offset = len(all_cells)
            shifted = expression.replace("output_scale", f"{output_scale:g}")
            for local_index in reversed(range(1, len(cells) + 1)):
                shifted = re.sub(
                    rf"\bv{local_index}\b",
                    f"v{offset + local_index}",
                    shifted,
                )
            all_cells.extend(cells)
            namespace = {
                f"v{offset + index}": self._cell_value(cell)
                for index, cell in enumerate(cells, start=1)
            }
            try:
                value = float(eval(shifted, {"__builtins__": {}}, namespace))
            except (ArithmeticError, ValueError, TypeError, SyntaxError, NameError):
                raise ValueError("panel operand evaluation failed")
            if not math.isfinite(value):
                raise ValueError("non-finite panel operand")
            return shifted, value

        if target_metric in RAW_METRICS or target_metric in _MONETARY_DERIVED:
            if unit_name in {"phần trăm", "điểm phần trăm", "lần", "năm", "cổ phiếu"}:
                return None
            if float(unit_multiplier) == 1_000:
                return None
            target_scale = float(unit_multiplier)
            response_unit = unit_name
        else:
            target_scale = 1.0
            response_unit = (
                "phần trăm"
                if target_metric.endswith("_pct")
                else "điểm phần trăm"
                if target_metric.endswith("_pp")
                else "lần"
            )

        try:
            for ticker, year in candidates:
                candidate_filter_expressions: list[str] = []
                candidate_filter_values: list[float] = []
                for filter_metric, _sign in filter_specs:
                    filter_built = self._operands(
                        source_index, ticker, year, scope, filter_metric
                    )
                    if filter_built is None:
                        return None
                    filter_expression, filter_value = append_expression(
                        filter_built, output_scale=1.0
                    )
                    candidate_filter_expressions.append(filter_expression)
                    candidate_filter_values.append(filter_value)
                selector_built = self._operands(
                    source_index, ticker, year, scope, selector_metric
                )
                target_built = self._operands(
                    source_index, ticker, year, scope, target_metric
                )
                if selector_built is None or target_built is None:
                    return None
                selector_expression, selector_value = append_expression(
                    selector_built, output_scale=1.0
                )
                target_expression, target_value = append_expression(
                    target_built, output_scale=target_scale
                )
                selector_expressions.append(selector_expression)
                target_expressions.append(target_expression)
                filter_expressions.append(candidate_filter_expressions)
                selector_values.append(selector_value)
                target_values.append(target_value)
                filter_values.append(candidate_filter_values)
        except ValueError:
            return None

        eligible_indices = [
            index
            for index, values in enumerate(filter_values)
            if all(
                value > 0 if sign > 0 else value < 0
                for value, (_metric, sign) in zip(values, filter_specs)
            )
        ]
        if not eligible_indices:
            return None
        optimum = (max if descending else min)(
            selector_values[index] for index in eligible_indices
        )
        selected = [
            index
            for index in eligible_indices
            if math.isclose(
                selector_values[index], optimum, rel_tol=1e-12, abs_tol=1e-12
            )
        ]
        if len(selected) != 1:
            return None
        expected = target_values[selected[0]]

        unique_paths: dict[str, tuple[str, str]] = {}
        cell_expressions: list[str] = []
        source_cells: list[dict] = []
        source_cell_keys: set[tuple[str, str, str, str, int, int]] = set()
        for cell in all_cells:
            path_obj = Path(cell.csv_path)
            if not path_obj.is_file():
                normalized_parts = [part.casefold() for part in path_obj.parts]
                try:
                    build_index = normalized_parts.index("build")
                    if normalized_parts[build_index + 1] != "tables":
                        return None
                    path_obj = self.root.joinpath(*path_obj.parts[build_index:])
                except (ValueError, IndexError):
                    return None
            if not path_obj.is_file():
                return None
            path = str(path_obj.resolve())
            if path not in unique_paths:
                unique_paths[path] = (f"df{len(unique_paths) + 1}", cell.table_ref)
            variable, table_ref = unique_paths[path]
            if table_ref != cell.table_ref:
                return None
            source = (
                f"(_checked_number(dfs[{variable!r}], {cell.row_idx}, {cell.col_idx}, "
                f"{cell.label!r}, {cell.raw!r}) * {cell.scale:g})"
            )
            if cell.metric_key in _COST_KEYS:
                source = f"abs({source})"
            cell_expressions.append(source)
            source_key = (
                cell.ticker,
                cell.year,
                cell.metric_key,
                cell.table_ref,
                cell.row_idx,
                cell.col_idx,
            )
            if source_key not in source_cell_keys:
                source_cell_keys.add(source_key)
                source_cells.append(
                    {
                        "ticker": cell.ticker,
                        "year": cell.year,
                        "metric_key": cell.metric_key,
                        "label": cell.label,
                        "table_ref": cell.table_ref,
                        "row_idx": cell.row_idx,
                        "col_idx": cell.col_idx,
                        "raw": cell.raw,
                    }
                )

        def resolve_sources(expression: str) -> str:
            resolved = expression
            for index, source in reversed(list(enumerate(cell_expressions, start=1))):
                # ``source`` embeds repr() of labels copied from OCR tables.
                # Passing it as a replacement template makes ``re.sub`` consume
                # backslashes such as ``\\(``, yielding an invalid Python escape
                # in the emitted program.  A callable replacement is literal.
                resolved = re.sub(
                    rf"\bv{index}\b",
                    lambda _match, replacement=source: replacement,
                    resolved,
                )
            return resolved

        resolved_selectors = [resolve_sources(value) for value in selector_expressions]
        resolved_targets = [resolve_sources(value) for value in target_expressions]
        resolved_filters = [
            [resolve_sources(value) for value in expressions]
            for expressions in filter_expressions
        ]
        prelude = """def _number(value):
    text = str(value).strip()
    negative = '(' in text or text.startswith('-')
    token = ''
    started = False
    for char in text:
        if char in '0123456789.,':
            token += char
            started = True
        elif started:
            break
    number = float(token.replace('.', '').replace(',', '.'))
    return -number if negative else number

def _checked_number(frame, row, column, expected_label, expected_raw):
    if row < 0 or row >= len(frame) or column < 0 or column >= len(frame.columns):
        return None
    row_text = ' | '.join(
        str(frame.iloc[row, index]).strip().casefold()
        for index in range(len(frame.columns))
    )
    if str(expected_label).strip().casefold() not in row_text:
        return None
    parsed = _number(frame.iloc[row, column])
    if parsed != _number(expected_raw):
        return None
    return parsed
"""
        reducer = "max" if descending else "min"
        if filter_specs:
            rendered_filters = ", ".join(
                "[" + ", ".join(values) + "]" for values in resolved_filters
            )
            pandas_query = (
                f"{prelude}\nfilter_values = [{rendered_filters}]\n"
                f"filter_signs = {[sign for _metric, sign in filter_specs]!r}\n"
                f"selector_values = [{', '.join(resolved_selectors)}]\n"
                f"target_values = [{', '.join(resolved_targets)}]\n"
                "eligible_indices = [index for index, values in enumerate(filter_values) "
                "if all((value > 0 if sign > 0 else value < 0) "
                "for value, sign in zip(values, filter_signs))]\n"
                f"selected_index = {reducer}(eligible_indices, "
                "key=lambda index: selector_values[index])\n"
                "result = target_values[selected_index]"
            )
        else:
            pandas_query = (
                f"{prelude}\nselector_values = [{', '.join(resolved_selectors)}]\n"
                f"target_values = [{', '.join(resolved_targets)}]\n"
                f"selected_index = selector_values.index({reducer}(selector_values))\n"
                "result = target_values[selected_index]"
            )
        csv_paths = {
            variable: path for path, (variable, _table_ref) in unique_paths.items()
        }
        table_refs = [
            table_ref for _path, (_variable, table_ref) in unique_paths.items()
        ]
        return CompiledFinancialQuery(
            metric=(
                f"signed_filter_extreme:{filter_specs[0][0]}->{selector_metric}->{target_metric}"
                if filter_specs
                else f"extreme:{selector_metric}->{target_metric}"
            ),
            answer=expected,
            unit=response_unit,
            pandas_query=pandas_query,
            csv_paths=csv_paths,
            table_refs=table_refs,
            source_cells=source_cells,
        )

    def _compile_universe_inventory_decline_cfo_margin(
        self,
        question: str,
        facets: dict,
    ) -> CompiledFinancialQuery | None:
        """Compile a strict all-company filter/reduce without an LLM context.

        The question names no finite ticker list, so document retrieval cannot
        be made complete under the normal cap.  This grammar instead scans the
        normalized statement cube, keeps companies whose inventory declined by
        at least the requested threshold, and returns the maximum CFO margin.
        Every emitted operand remains bound to its exact CSV cell and raw text.
        """
        if facets.get("tickers"):
            return None
        years = sorted(
            int(value)
            for value in facets.get("years", [])
            if re.fullmatch(r"20\d{2}", str(value))
        )
        if len(years) != 2 or years[0] >= years[1]:
            return None
        matched = _UNIVERSE_INVENTORY_DECLINE_CFO_MARGIN_RE.search(fold(question))
        if matched is None:
            return None
        threshold = float(matched.group("threshold").replace(",", "."))
        if not 0 < threshold <= 100:
            return None

        previous_year, current_year = years
        scope = self._scope(str(facets.get("scope", "")))
        source_index = self._load()
        tickers = sorted(
            {
                ticker
                for ticker, year, candidate_scope, _metric_key in source_index
                if year in {str(previous_year), str(current_year)}
                and candidate_scope == scope
            }
        )
        if not tickers:
            return None

        all_cells: list[StatementCell] = []

        def append_expression(
            built: tuple[list[StatementCell], str, float, str],
        ) -> tuple[str, float]:
            cells, expression, _placeholder, _kind = built
            offset = len(all_cells)
            shifted = expression.replace("output_scale", "1")
            for local_index in reversed(range(1, len(cells) + 1)):
                shifted = re.sub(
                    rf"\bv{local_index}\b",
                    f"v{offset + local_index}",
                    shifted,
                )
            all_cells.extend(cells)
            namespace = {
                f"v{offset + index}": self._cell_value(cell)
                for index, cell in enumerate(cells, start=1)
            }
            try:
                value = float(eval(shifted, {"__builtins__": {}}, namespace))
            except (ArithmeticError, ValueError, TypeError, SyntaxError, NameError):
                raise ValueError("universe operand evaluation failed")
            if not math.isfinite(value):
                raise ValueError("non-finite universe operand")
            return shifted, value

        rows: list[tuple[str, str, str, str]] = []
        values: list[tuple[str, float, float]] = []
        for ticker in tickers:
            previous = self._operands(
                source_index, ticker, previous_year, scope, "inventory"
            )
            current = self._operands(
                source_index, ticker, current_year, scope, "inventory"
            )
            target = self._operands(
                source_index, ticker, current_year, scope, "cfo_margin_pct"
            )
            if previous is None or current is None or target is None:
                continue
            before_count = len(all_cells)
            try:
                previous_expression, previous_value = append_expression(previous)
                current_expression, current_value = append_expression(current)
                target_expression, target_value = append_expression(target)
            except ValueError:
                del all_cells[before_count:]
                continue
            if previous_value == 0:
                del all_cells[before_count:]
                continue
            change = (current_value / previous_value - 1) * 100
            rows.append(
                (
                    ticker,
                    previous_expression,
                    current_expression,
                    target_expression,
                )
            )
            values.append((ticker, change, target_value))

        eligible = [item for item in values if item[1] <= -threshold]
        if not eligible:
            return None
        optimum = max(item[2] for item in eligible)
        selected = [
            item
            for item in eligible
            if math.isclose(item[2], optimum, rel_tol=1e-12, abs_tol=1e-12)
        ]
        if len(selected) != 1:
            return None

        unique_paths: dict[str, tuple[str, str]] = {}
        cell_expressions: list[str] = []
        source_cells: list[dict] = []
        source_cell_keys: set[tuple[str, str, str, str, int, int]] = set()
        for cell in all_cells:
            path_obj = Path(cell.csv_path)
            if not path_obj.is_file():
                normalized_parts = [part.casefold() for part in path_obj.parts]
                try:
                    build_index = normalized_parts.index("build")
                    if normalized_parts[build_index + 1] != "tables":
                        return None
                    path_obj = self.root.joinpath(*path_obj.parts[build_index:])
                except (ValueError, IndexError):
                    return None
            if not path_obj.is_file():
                return None
            path = str(path_obj.resolve())
            if path not in unique_paths:
                unique_paths[path] = (f"df{len(unique_paths) + 1}", cell.table_ref)
            variable, table_ref = unique_paths[path]
            if table_ref != cell.table_ref:
                return None
            source = (
                f"(_checked_number(dfs[{variable!r}], {cell.row_idx}, {cell.col_idx}, "
                f"{cell.label!r}, {cell.raw!r}) * {cell.scale:g})"
            )
            cell_expressions.append(source)
            source_key = (
                cell.ticker,
                cell.year,
                cell.metric_key,
                cell.table_ref,
                cell.row_idx,
                cell.col_idx,
            )
            if source_key not in source_cell_keys:
                source_cell_keys.add(source_key)
                source_cells.append(
                    {
                        "ticker": cell.ticker,
                        "year": cell.year,
                        "metric_key": cell.metric_key,
                        "label": cell.label,
                        "table_ref": cell.table_ref,
                        "row_idx": cell.row_idx,
                        "col_idx": cell.col_idx,
                        "raw": cell.raw,
                    }
                )

        def resolve_sources(expression: str) -> str:
            resolved = expression
            for index, source in reversed(list(enumerate(cell_expressions, start=1))):
                resolved = re.sub(
                    rf"\bv{index}\b",
                    lambda _match, replacement=source: replacement,
                    resolved,
                )
            return resolved

        resolved_rows = [
            (
                ticker,
                resolve_sources(previous_expression),
                resolve_sources(current_expression),
                resolve_sources(target_expression),
            )
            for ticker, previous_expression, current_expression, target_expression in rows
        ]
        rendered_rows = ",\n    ".join(
            "{'ticker': %r, 'inventory_previous': %s, "
            "'inventory_current': %s, 'cfo_margin_pct': %s}"
            % row
            for row in resolved_rows
        )
        prelude = """def _number(value):
    text = str(value).strip()
    negative = '(' in text or text.startswith('-')
    token = ''
    started = False
    for char in text:
        if char in '0123456789.,':
            token += char
            started = True
        elif started:
            break
    number = float(token.replace('.', '').replace(',', '.'))
    return -number if negative else number

def _checked_number(frame, row, column, expected_label, expected_raw):
    if row < 0 or row >= len(frame) or column < 0 or column >= len(frame.columns):
        return None
    row_text = ' | '.join(
        str(frame.iloc[row, index]).strip().casefold()
        for index in range(len(frame.columns))
    )
    if str(expected_label).strip().casefold() not in row_text:
        return None
    parsed = _number(frame.iloc[row, column])
    if parsed != _number(expected_raw):
        return None
    return parsed
"""
        pandas_query = (
            f"{prelude}\n_universe_rows = [\n    {rendered_rows}\n]\n"
            "panel = pd.DataFrame(_universe_rows)\n"
            "panel['inventory_change_pct'] = "
            "(panel['inventory_current'] / panel['inventory_previous'] - 1) * 100\n"
            f"eligible = panel[panel['inventory_change_pct'] <= {-threshold:g}]\n"
            "result = eligible['cfo_margin_pct'].max()"
        )
        return CompiledFinancialQuery(
            metric="universe:inventory_decline->cfo_margin_pct",
            answer=optimum,
            unit="pháº§n trÄƒm",
            pandas_query=pandas_query,
            csv_paths={
                variable: path
                for path, (variable, _table_ref) in unique_paths.items()
            },
            table_refs=[
                table_ref
                for _path, (_variable, table_ref) in unique_paths.items()
            ],
            source_cells=source_cells,
        )

    def compile(self, question: str, facets: dict) -> CompiledFinancialQuery | None:
        if not self.available:
            return None
        universe = self._compile_universe_inventory_decline_cfo_margin(
            question, facets
        )
        if universe is not None:
            return universe
        panel = self._compile_extreme_panel(question, facets)
        if panel is not None:
            return panel
        if len(facets.get("tickers", [])) != 1 or len(facets.get("years", [])) != 1:
            return None
        normalized = fold(question)
        if (
            _BLOCKED_INTENTS.search(normalized)
            or _QUALIFIER_BLOCKS.search(normalized)
            or _BEGINNING_DATE_BLOCKS.search(normalized)
        ):
            return None
        metric = self._detect_metric(question)
        if metric is None:
            return None
        if metric in RAW_METRICS and re.search(
            r"\b(?:ty so|ti so|ty le|ti le|ty trong|ti trong|ty suat|ti suat|he so)\b",
            normalized,
        ):
            return None
        if metric == "interest_expense" and re.search(
            r"\bchi phi lai vay\s+\w.{0,70}\bcua\b", normalized
        ):
            return None
        if metric == "npat" and re.search(r"\bloi nhuan sau thue tndn\b", normalized):
            return None

        ticker = str(facets["tickers"][0]).upper()
        year = int(facets["years"][0])
        scope = self._scope(str(facets.get("scope", "")))
        source_index = self._load()

        unit_name, unit_multiplier = requested_unit(question)
        built = self._operands(source_index, ticker, year, scope, metric)
        if built is None:
            return None
        cells, expression, _placeholder_scale, kind = built
        if kind == "currency":
            if unit_name in {"phần trăm", "điểm phần trăm", "lần", "năm", "cổ phiếu"}:
                return None
            output_scale = float(unit_multiplier)
            # Compact OCR tables can omit whether a value is expressed in VND
            # or thousand VND.  Million/billion requests remain distinguishable
            # by magnitude and headers; direct thousand-VND compilation does not.
            if output_scale == 1_000:
                return None
            response_unit = unit_name
        else:
            output_scale = 1.0
            response_unit = kind

        unique_paths: dict[str, tuple[str, str]] = {}
        cell_expressions: list[str] = []
        source_cells: list[dict] = []
        for cell in cells:
            path_obj = Path(cell.csv_path)
            if not path_obj.is_file():
                normalized_parts = [part.casefold() for part in path_obj.parts]
                try:
                    build_index = normalized_parts.index("build")
                    if normalized_parts[build_index + 1] != "tables":
                        return None
                    path_obj = self.root.joinpath(*path_obj.parts[build_index:])
                except (ValueError, IndexError):
                    return None
            if not path_obj.is_file():
                return None
            path = str(path_obj.resolve())
            if path not in unique_paths:
                unique_paths[path] = (f"df{len(unique_paths) + 1}", cell.table_ref)
            variable, bound_table_ref = unique_paths[path]
            if bound_table_ref != cell.table_ref:
                # One evidence file must not silently claim two table IDs.
                return None
            source = (
                f"(_checked_number(dfs[{variable!r}], {cell.row_idx}, {cell.col_idx}, "
                f"{cell.label!r}, {cell.raw!r}) * {cell.scale:g})"
            )
            if cell.metric_key in _COST_KEYS:
                source = f"abs({source})"
            cell_expressions.append(source)
            source_cells.append(
                {
                    "ticker": cell.ticker,
                    "year": cell.year,
                    "metric_key": cell.metric_key,
                    "label": cell.label,
                    "table_ref": cell.table_ref,
                    "row_idx": cell.row_idx,
                    "col_idx": cell.col_idx,
                    "raw": cell.raw,
                }
            )

        resolved_expression = expression
        for index, source in reversed(list(enumerate(cell_expressions, start=1))):
            resolved_expression = re.sub(
                rf"\bv{index}\b",
                lambda _match, replacement=source: replacement,
                resolved_expression,
            )
        prelude = """def _number(value):
    text = str(value).strip()
    negative = '(' in text or text.startswith('-')
    token = ''
    started = False
    for char in text:
        if char in '0123456789.,':
            token += char
            started = True
        elif started:
            break
    number = float(token.replace('.', '').replace(',', '.'))
    return -number if negative else number

def _checked_number(frame, row, column, expected_label, expected_raw):
    if row < 0 or row >= len(frame) or column < 0 or column >= len(frame.columns):
        return None
    row_text = ' | '.join(
        str(frame.iloc[row, index]).strip().casefold()
        for index in range(len(frame.columns))
    )
    if str(expected_label).strip().casefold() not in row_text:
        return None
    parsed = _number(frame.iloc[row, column])
    if parsed != _number(expected_raw):
        return None
    return parsed
"""
        pandas_query = f"{prelude}\noutput_scale = {output_scale:g}\nresult = {resolved_expression}"
        csv_paths = {
            variable: path for path, (variable, _table_ref) in unique_paths.items()
        }
        table_refs = [
            table_ref for _path, (_variable, table_ref) in unique_paths.items()
        ]

        values = [self._cell_value(cell) for cell in cells]
        namespace = {f"v{index}": value for index, value in enumerate(values, start=1)}
        namespace["output_scale"] = output_scale
        try:
            expected = float(eval(expression, {"__builtins__": {}}, namespace))
        except (ArithmeticError, ValueError, TypeError, SyntaxError, NameError):
            return None
        if not math.isfinite(expected):
            return None

        return CompiledFinancialQuery(
            metric=metric,
            answer=expected,
            unit=response_unit,
            pandas_query=pandas_query,
            csv_paths=csv_paths,
            table_refs=table_refs,
            source_cells=source_cells,
        )
