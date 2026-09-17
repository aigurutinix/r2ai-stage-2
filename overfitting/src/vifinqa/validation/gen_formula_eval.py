"""Deterministic end-to-end evaluation set for financial formula solving.

Gold facts come from exact VAS codes, canonical labels, explicit year columns,
and known table units.  The generated questions exercise router, retrieval,
fact resolution, formula composition, submission replay, and unit handling.
"""
from __future__ import annotations

import itertools
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from ..extraction.build_store import Store
from ..router.entities import StockMap
from ..utils.io import write_json, write_jsonl
from ..utils.viet_text import fuzz_token_set


@dataclass(frozen=True)
class MetricDef:
    key: str
    code: str
    labels: tuple[str, ...]


METRICS = (
    MetricDef("net_revenue", "10", (
        "Doanh thu thuần về bán hàng và cung cấp dịch vụ",
    )),
    MetricDef("gross_profit", "20", (
        "Lợi nhuận gộp về bán hàng và cung cấp dịch vụ",
    )),
    MetricDef("operating_cash_flow", "20", (
        "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
    )),
    MetricDef("current_assets", "100", ("Tài sản ngắn hạn",)),
    MetricDef("inventory", "140", ("Hàng tồn kho",)),
    MetricDef("total_assets", "270", ("Tổng cộng tài sản", "Tổng tài sản")),
    MetricDef("liabilities", "300", ("Nợ phải trả",)),
    MetricDef("current_liabilities", "310", (
        "Nợ ngắn hạn", "Nợ phải trả ngắn hạn",
    )),
    MetricDef("equity", "400", ("Vốn chủ sở hữu",)),
)
_BY_CODE: dict[str, list[MetricDef]] = defaultdict(list)
for _metric_def in METRICS:
    _BY_CODE[_metric_def.code].append(_metric_def)

_STOCK_METRICS = {
    "current_assets", "inventory", "total_assets", "liabilities",
    "current_liabilities", "equity",
}


@dataclass(frozen=True)
class Cell:
    ticker: str
    year: int
    doc_type: str
    metric: str
    report_id: str
    table_pos: int
    line_no: int
    row: int
    col: int
    label: str
    value_vnd: float


FactMap = dict[tuple[str, int, str], Cell]


def _clean_code(value) -> str:
    return re.sub(r"\.0$", "", str(value or "").strip())


def _has_exact_year(col_name: str, year: int) -> bool:
    return bool(re.search(
        rf"(?<!\d){year}(?!\d)|31\s*/\s*12\s*/\s*{year}",
        str(col_name),
    ))


def _is_nonclosing_date(col_name: str, year: int) -> bool:
    """Reject opening/interim stock columns that happen to name the same year."""
    text = str(col_name)
    dated = re.search(rf"\d{{1,2}}\s*/\s*\d{{1,2}}\s*/\s*{year}", text)
    closing = re.search(rf"31\s*/\s*12\s*/\s*{year}", text)
    return bool(dated and not closing)


def _collect(store: Store, max_tickers: int = 0) -> FactMap:
    """Collect independent, unambiguous facts suitable for synthetic gold."""
    raw: dict[tuple[str, int, str], Cell | None] = {}
    tickers = sorted({ticker for ticker, _year, _doc in store.report_index})
    if max_tickers:
        tickers = tickers[:max_tickers]

    for ticker in tickers:
        cells = store.cells_of(ticker)
        if not len(cells):
            continue
        cells = cells[(cells.doc_type == "consolidated") & cells.unit_known]
        for row in cells.itertuples():
            code = _clean_code(row.row_code)
            specs = _BY_CODE.get(code, ())
            if not specs or not _has_exact_year(row.col_name, int(row.year)):
                continue
            value_vnd = float(row.value) * float(row.unit_scale)
            if not math.isfinite(value_vnd) or abs(value_vnd) < 1.0:
                continue
            label = str(row.label)
            for spec in specs:
                if spec.key in _STOCK_METRICS and _is_nonclosing_date(
                        row.col_name, int(row.year)):
                    continue
                if max(fuzz_token_set(label, expected)
                       for expected in spec.labels) < 85:
                    continue
                key = (str(row.ticker), int(row.year), spec.key)
                cell = Cell(
                    ticker=str(row.ticker), year=int(row.year),
                    doc_type=str(row.doc_type), metric=spec.key,
                    report_id=str(row.report_id), table_pos=int(row.table_pos),
                    line_no=store.line_no_of(str(row.report_id), int(row.table_pos)),
                    row=int(row.row), col=int(row.col), label=label,
                    value_vnd=value_vnd,
                )
                if key in raw:
                    raw[key] = None
                else:
                    raw[key] = cell
    return {key: cell for key, cell in raw.items() if cell is not None}


def _cell(facts: FactMap, ticker: str, year: int, metric: str) -> Cell | None:
    return facts.get((ticker, year, metric))


def _has(facts: FactMap, ticker: str, year: int,
         metrics: tuple[str, ...]) -> bool:
    return all(_cell(facts, ticker, year, metric) is not None for metric in metrics)


def _cells(facts: FactMap, ticker: str, year: int,
           metrics: tuple[str, ...]) -> list[Cell]:
    return [_cell(facts, ticker, year, metric) for metric in metrics]


def _name(names: dict[str, str], ticker: str) -> str:
    return f"{names.get(ticker, ticker)} ({ticker})"


def _name_list(names: dict[str, str], tickers: tuple[str, ...]) -> str:
    return ", ".join(_name(names, ticker) for ticker in tickers)


def _vn_number(value: float) -> str:
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return text.replace(".", ",")


def _entity_samples(facts: FactMap, metrics: tuple[str, ...], size: int,
                    rng: random.Random, limit: int) -> list[tuple[int, tuple[str, ...]]]:
    grouped: dict[int, list[str]] = defaultdict(list)
    for ticker, year, _metric in facts:
        if ticker not in grouped[year] and _has(facts, ticker, year, metrics):
            grouped[year].append(ticker)
    years = [year for year, tickers in grouped.items() if len(tickers) >= size]
    seen, out = set(), []
    attempts = max(200, limit * 80)
    for _ in range(attempts):
        if not years or len(out) >= limit:
            break
        year = rng.choice(years)
        picks = tuple(rng.sample(sorted(grouped[year]), size))
        key = (year, tuple(sorted(picks)))
        if key in seen:
            continue
        seen.add(key)
        out.append((year, picks))
    return out


def _temporal_samples(facts: FactMap, metrics: tuple[str, ...], size: int,
                      rng: random.Random,
                      limit: int) -> list[tuple[int, int, tuple[str, ...]]]:
    years_by_ticker: dict[str, set[int]] = defaultdict(set)
    for ticker, year, _metric in facts:
        if _has(facts, ticker, year, metrics):
            years_by_ticker[ticker].add(year)
    grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
    for ticker, years in years_by_ticker.items():
        for end in years:
            if end - 1 in years:
                grouped[(end - 1, end)].append(ticker)
    periods = [period for period, tickers in grouped.items()
               if len(tickers) >= size]
    seen, out = set(), []
    attempts = max(200, limit * 80)
    for _ in range(attempts):
        if not periods or len(out) >= limit:
            break
        start, end = rng.choice(periods)
        picks = tuple(rng.sample(sorted(grouped[(start, end)]), size))
        key = (start, end, tuple(sorted(picks)))
        if key in seen:
            continue
        seen.add(key)
        out.append((start, end, picks))
    return out


def build_cases(facts: FactMap, names: dict[str, str], per_class: int = 24,
                seed: int = 23) -> tuple[list[dict], dict[str, dict]]:
    """Build balanced formula cases from an already verified fact map."""
    rng = random.Random(seed)
    questions: list[dict] = []
    gold: dict[str, dict] = {}
    counts: Counter = Counter()
    seen_questions: set[str] = set()

    def emit(klass: str, question: str, answer: float, cells: list[Cell],
             output_type: str, unit: str) -> bool:
        if counts[klass] >= per_class or question in seen_questions:
            return False
        if not math.isfinite(float(answer)) or not cells:
            return False
        qid = len(questions) + 1
        seen_questions.add(question)
        counts[klass] += 1
        docs, tables, operands = [], [], []
        for cell in cells:
            if cell.report_id not in docs:
                docs.append(cell.report_id)
            table_ref = f"{cell.report_id}|{cell.line_no}"
            if table_ref not in tables:
                tables.append(table_ref)
            operands.append({
                "ticker": cell.ticker, "year": cell.year,
                "metric": cell.metric, "report_id": cell.report_id,
                "table_pos": cell.table_pos, "line_no": cell.line_no,
                "row": cell.row, "col": cell.col, "label": cell.label,
                "value_vnd": cell.value_vnd,
            })
        questions.append({"id": qid, "question": question})
        gold[str(qid)] = {
            "answer": round(float(answer), 2), "klass": klass,
            "formula": klass, "output_type": output_type, "unit": unit,
            "relevant_docs": docs, "relevant_tables": tables,
            "operands": operands,
        }
        return True

    # 1. Growth of one exact metric over consecutive years.
    growth = []
    for ticker, year, metric in facts:
        if metric != "net_revenue":
            continue
        end, base = _cell(facts, ticker, year, metric), _cell(
            facts, ticker, year - 1, metric)
        if base and end and base.value_vnd > 0:
            growth.append((ticker, year - 1, year, base, end))
    rng.shuffle(growth)
    for ticker, start, end_year, base, end in growth:
        answer = (end.value_vnd - base.value_vnd) / abs(base.value_vnd) * 100.0
        emit("growth_pct", (
            f"Doanh thu thuần của {_name(names, ticker)} năm {end_year} tăng trưởng "
            f"bao nhiêu phần trăm so với năm {start}?"
        ), answer, [end, base], "percent", "%")

    # 2-3. Direct margin and balance-sheet ratio.
    direct = sorted({(ticker, year) for ticker, year, _metric in facts})
    rng.shuffle(direct)
    for ticker, year in direct:
        if _has(facts, ticker, year, ("gross_profit", "net_revenue")):
            gross, revenue = _cells(
                facts, ticker, year, ("gross_profit", "net_revenue"))
            if revenue.value_vnd > 0 and abs(gross.value_vnd / revenue.value_vnd) <= 5:
                emit("gross_margin", (
                    f"Biên lợi nhuận gộp của {_name(names, ticker)} năm {year} "
                    "là bao nhiêu phần trăm?"
                ), gross.value_vnd / revenue.value_vnd * 100.0,
                     [gross, revenue], "percent", "%")
        if _has(facts, ticker, year, ("liabilities", "equity")):
            debt, equity = _cells(facts, ticker, year, ("liabilities", "equity"))
            ratio = debt.value_vnd / equity.value_vnd if equity.value_vnd > 0 else -1
            if 0 <= ratio <= 20:
                emit("debt_equity", (
                    "Hệ số nợ phải trả trên vốn chủ sở hữu của "
                    f"{_name(names, ticker)} năm {year} là bao nhiêu lần?"
                ), ratio, [debt, equity], "ratio", "lần")

    # 4. Difference between margins of two companies.
    for year, tickers in _entity_samples(
            facts, ("gross_profit", "net_revenue"), 2, rng, per_class * 5):
        t1, t2 = tickers
        c1 = _cells(facts, t1, year, ("gross_profit", "net_revenue"))
        c2 = _cells(facts, t2, year, ("gross_profit", "net_revenue"))
        if c1[1].value_vnd <= 0 or c2[1].value_vnd <= 0:
            continue
        m1 = c1[0].value_vnd / c1[1].value_vnd * 100.0
        m2 = c2[0].value_vnd / c2[1].value_vnd * 100.0
        emit("margin_difference", (
            f"Biên lợi nhuận gộp năm {year} của {_name(names, t1)} chênh lệch "
            f"bao nhiêu điểm phần trăm so với {_name(names, t2)}?"
        ), m1 - m2, c1 + c2, "percentage_point", "điểm phần trăm")

    # 5. Change in one company's margin between two periods.
    for start, end, (ticker,) in _temporal_samples(
            facts, ("gross_profit", "net_revenue"), 1, rng, per_class * 5):
        before = _cells(facts, ticker, start, ("gross_profit", "net_revenue"))
        after = _cells(facts, ticker, end, ("gross_profit", "net_revenue"))
        if before[1].value_vnd <= 0 or after[1].value_vnd <= 0:
            continue
        delta = (after[0].value_vnd / after[1].value_vnd
                 - before[0].value_vnd / before[1].value_vnd) * 100.0
        emit("margin_change", (
            f"Mức thay đổi từ năm {start} đến năm {end} của biên lợi nhuận gộp "
            f"tại {_name(names, ticker)} là bao nhiêu điểm phần trăm?"
        ), delta, after + before, "percentage_point", "điểm phần trăm")

    # 6. Count entities above a money threshold.
    for year, tickers in _entity_samples(
            facts, ("total_assets",), 4, rng, per_class * 8):
        cells = [_cell(facts, ticker, year, "total_assets") for ticker in tickers]
        values = sorted(cell.value_vnd / 1e9 for cell in cells)
        if values[0] <= 0:
            continue
        threshold_billion = round((values[1] + values[2]) / 2.0, 2)
        answer = sum(cell.value_vnd > threshold_billion * 1e9 for cell in cells)
        if not 0 < answer < len(cells):
            continue
        emit("count_threshold", (
            f"Trong các công ty {_name_list(names, tickers)}, có bao nhiêu công ty "
            f"có tổng tài sản lớn hơn {_vn_number(threshold_billion)} tỷ đồng "
            f"năm {year}?"
        ), answer, cells, "count", "số lượng")

    # 7. Count years of one company above a threshold.
    by_ticker: dict[str, list[int]] = defaultdict(list)
    for ticker, year, metric in facts:
        if metric == "total_assets":
            by_ticker[ticker].append(year)
    year_cases = []
    for ticker, years in by_ticker.items():
        for picked in itertools.combinations(sorted(set(years)), 3):
            year_cases.append((ticker, picked))
    rng.shuffle(year_cases)
    for ticker, years in year_cases:
        cells = [_cell(facts, ticker, year, "total_assets") for year in years]
        values = sorted(cell.value_vnd / 1e9 for cell in cells)
        if values[0] <= 0:
            continue
        threshold_billion = round((values[0] + values[-1]) / 2.0, 2)
        answer = sum(cell.value_vnd > threshold_billion * 1e9 for cell in cells)
        if not 0 < answer < len(cells):
            continue
        years_text = ", ".join(str(year) for year in years[:-1]) + f" và {years[-1]}"
        emit("count_years", (
            f"Trong các năm {years_text}, {_name(names, ticker)} có bao nhiêu năm "
            f"ghi nhận tổng tài sản lớn hơn {_vn_number(threshold_billion)} tỷ đồng?"
        ), answer, cells, "count", "số lượng")

    # 8. Count entities satisfying two independent statements/formulas.
    multi_metrics = ("current_assets", "current_liabilities", "operating_cash_flow")
    for year, tickers in _entity_samples(facts, multi_metrics, 3, rng, per_class * 12):
        all_cells, passed = [], []
        for ticker in tickers:
            current, liabilities, cash_flow = _cells(facts, ticker, year, multi_metrics)
            all_cells.extend((current, liabilities, cash_flow))
            passed.append(current.value_vnd - liabilities.value_vnd < 0
                          and cash_flow.value_vnd > 0)
        answer = sum(passed)
        if not 0 < answer < len(tickers):
            continue
        emit("count_multi_condition", (
            f"Năm {year}, trong các công ty {_name_list(names, tickers)}, có bao "
            "nhiêu doanh nghiệp đồng thời ghi nhận vốn lưu động ròng âm và lưu "
            "chuyển tiền thuần từ hoạt động kinh doanh dương?"
        ), answer, all_cells, "count", "số lượng")

    # 9. Rank companies by a derived ratio.
    for year, tickers in _entity_samples(
            facts, ("liabilities", "equity"), 3, rng, per_class * 8):
        all_cells, ratios = [], []
        for ticker in tickers:
            debt, equity = _cells(facts, ticker, year, ("liabilities", "equity"))
            if equity.value_vnd <= 0:
                break
            all_cells.extend((debt, equity))
            ratios.append(debt.value_vnd / equity.value_vnd)
        if len(ratios) != len(tickers) or max(ratios) > 20:
            continue
        emit("ranking_ratio", (
            f"Trong các công ty {_name_list(names, tickers)}, hệ số nợ phải trả "
            f"trên vốn chủ sở hữu lớn nhất năm {year} là bao nhiêu lần?"
        ), max(ratios), all_cells, "ratio", "lần")

    # 10. Nested ranking: select by D/E, answer the winner's current ratio.
    nested_metrics = ("liabilities", "equity", "current_assets", "current_liabilities")
    for year, tickers in _entity_samples(facts, nested_metrics, 3, rng, per_class * 12):
        all_cells, candidates = [], []
        for ticker in tickers:
            debt, equity, current, short_debt = _cells(
                facts, ticker, year, nested_metrics)
            if equity.value_vnd <= 0 or short_debt.value_vnd <= 0:
                break
            all_cells.extend((debt, equity, current, short_debt))
            candidates.append((debt.value_vnd / equity.value_vnd,
                               current.value_vnd / short_debt.value_vnd))
        if len(candidates) != len(tickers) or max(x[0] for x in candidates) > 20:
            continue
        winner = max(candidates, key=lambda item: item[0])
        emit("nested_ranking", (
            f"Năm {year}, trong các công ty {_name_list(names, tickers)}, hệ số "
            "thanh toán hiện hành của doanh nghiệp có hệ số nợ phải trả trên vốn "
            "chủ sở hữu cao nhất là bao nhiêu lần?"
        ), winner[1], all_cells, "ratio", "lần")

    # 11. Count temporal, multi-formula conditions.
    temporal_metrics = ("inventory", "total_assets", "gross_profit", "net_revenue")
    for start, end, tickers in _temporal_samples(
            facts, temporal_metrics, 2, rng, per_class * 16):
        all_cells, passed = [], []
        for ticker in tickers:
            before = _cells(facts, ticker, start, temporal_metrics)
            after = _cells(facts, ticker, end, temporal_metrics)
            if before[1].value_vnd <= 0 or before[3].value_vnd <= 0 \
                    or after[1].value_vnd <= 0 or after[3].value_vnd <= 0:
                break
            all_cells.extend(after + before)
            inv_up = after[0].value_vnd / after[1].value_vnd > \
                before[0].value_vnd / before[1].value_vnd
            margin_down = after[2].value_vnd / after[3].value_vnd < \
                before[2].value_vnd / before[3].value_vnd
            passed.append(inv_up and margin_down)
        answer = sum(passed)
        if len(passed) != len(tickers) or not 0 < answer < len(tickers):
            continue
        emit("temporal_count", (
            f"Từ {start} sang {end}, trong các công ty {_name_list(names, tickers)}, "
            "có bao nhiêu doanh nghiệp đồng thời tăng tỷ trọng hàng tồn kho trên "
            "tổng tài sản và giảm biên lợi nhuận gộp?"
        ), answer, all_cells, "count", "số lượng")

    # 12. Filter entities by revenue growth, then average margin changes.
    avg_metrics = ("gross_profit", "net_revenue")
    for start, end, tickers in _temporal_samples(
            facts, avg_metrics, 3, rng, per_class * 12):
        all_cells, changes = [], []
        for ticker in tickers:
            before = _cells(facts, ticker, start, avg_metrics)
            after = _cells(facts, ticker, end, avg_metrics)
            if before[1].value_vnd <= 0 or after[1].value_vnd <= 0:
                break
            all_cells.extend(after + before)
            if after[1].value_vnd > before[1].value_vnd:
                changes.append((after[0].value_vnd / after[1].value_vnd
                                - before[0].value_vnd / before[1].value_vnd) * 100.0)
        if not changes:
            continue
        emit("average_margin_change", (
            f"Mức thay đổi trung bình từ năm {start} đến năm {end} của biên lợi "
            f"nhuận gộp tại các công ty {_name_list(names, tickers)} có doanh thu "
            f"thuần năm {end} tăng so với năm {start} là bao nhiêu điểm phần trăm?"
        ), sum(changes) / len(changes), all_cells,
             "percentage_point", "điểm phần trăm")

    return questions, gold


def generate(store_dir: Path, code_stock_csv: Path, out_dir: Path,
             per_class: int = 24, seed: int = 23,
             max_tickers: int = 0) -> dict[str, int]:
    store = Store(store_dir, cache_size=120)
    stock = StockMap(code_stock_csv)
    facts = _collect(store, max_tickers=max_tickers)
    questions, gold = build_cases(
        facts, stock.ticker2name, per_class=per_class, seed=seed)
    out_dir = Path(out_dir)
    write_jsonl(out_dir / "formula_questions.jsonl", questions)
    write_json(out_dir / "formula_gold.json", gold)
    distribution = dict(sorted(Counter(
        item["klass"] for item in gold.values()).items()))
    write_json(out_dir / "formula_manifest.json", {
        "schema": 1, "seed": seed, "per_class_target": per_class,
        "verified_facts": len(facts), "questions": len(questions),
        "distribution": distribution,
    })
    print(f"[formula-eval] verified facts: {len(facts)}")
    print(f"[formula-eval] {len(questions)} questions -> {out_dir}")
    print(f"[formula-eval] per class: {distribution}")
    return distribution
