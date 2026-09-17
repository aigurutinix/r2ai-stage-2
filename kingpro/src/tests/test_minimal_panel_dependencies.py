from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_compliant_panel_candidate.py"


def load_builder():
    spec = importlib.util.spec_from_file_location("minimal_panel_builder", MODULE_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def execute_query(module, query: str, cells, typed: bool) -> float:
    rows = []
    for cell in cells:
        row = {
            "ticker": cell.ticker,
            "year": cell.year,
            "metric_key": cell.metric_key,
            "raw": cell.raw,
            "scale": cell.scale,
            "source_table": cell.table_ref,
            "source_csv": Path(cell.csv_path).name,
            "row_idx": cell.row_idx,
            "col_idx": cell.col_idx,
        }
        if typed:
            row["typed_factor"] = module.typed_factor(cell.raw)
        rows.append(row)
    namespace = {"dfs": {"df1": pd.DataFrame(rows)}, "pd": pd}
    exec(query, namespace)
    return float(namespace["result"])


def test_q390_only_loads_quick_ratio_and_inventory_dependencies() -> None:
    module = load_builder()
    tickers, years, analysis = module.AUDITED_PANEL_FIXES[390]
    raw, previous, derived = module.required_panel_columns(analysis)

    assert raw == ["current_assets", "inventory", "current_liabilities"]
    assert previous == []
    assert derived == ["quick_ratio"]

    full_query, full_cells = module.make_panel_query(tickers, years, analysis)
    lean_query, lean_cells = module.make_panel_query(
        tickers, years, analysis, minimal_dependencies=True
    )
    assert len(lean_cells) < len(full_cells)
    assert execute_query(module, lean_query, lean_cells, typed=True) == execute_query(
        module, full_query, full_cells, typed=False
    )


def test_nested_operating_leverage_dependencies_expand_to_original_cells() -> None:
    module = load_builder()
    _, _, analysis = module.AUDITED_PANEL_FIXES[414]
    raw, previous, derived = module.required_panel_columns(analysis)

    assert raw == ["revenue", "operating_profit"]
    assert previous == ["_prev_revenue", "_prev_operating_profit"]
    assert derived == [
        "operating_margin_pct",
        "revenue_growth_pct",
        "operating_leverage",
    ]


def test_direct_and_derived_columns_are_combined_without_unrelated_metrics() -> None:
    module = load_builder()
    _, _, analysis = module.AUDITED_PANEL_FIXES[433]
    raw, previous, derived = module.required_panel_columns(analysis)

    assert set(raw) == {
        "inventory",
        "short_term_receivables",
        "long_term_receivables",
        "total_assets",
        "liabilities",
    }
    assert previous == []
    assert derived == ["liabilities_to_assets_pct"]


def test_merge_suffix_columns_retain_their_underlying_metrics() -> None:
    module = load_builder()
    generated = module.read_jsonl(ROOT / "build" / "panel_answers_v4.jsonl")[364]
    raw, previous, derived = module.required_panel_columns(generated["code"])

    assert set(raw) == {"revenue", "npat", "cfo", "total_assets"}
    assert previous == ["_prev_revenue", "_prev_total_assets"]
    assert derived == ["revenue_growth_pct", "operating_accruals_ratio_pct"]


def test_semantic_input_can_be_retained_when_it_algebraically_cancels() -> None:
    module = load_builder()
    _, _, analysis = module.AUDITED_PANEL_FIXES[435]
    raw, previous, derived = module.required_panel_columns(
        analysis, module.MINIMAL_SEMANTIC_COLUMNS[435]
    )

    assert raw == ["revenue", "gross_profit", "interest_expense", "pbt"]
    assert previous == []
    assert derived == []
