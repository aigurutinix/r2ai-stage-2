from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_legacy_query_sources.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("audit_legacy_query_sources", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _resolve(query: str, frame: pd.DataFrame):
    env = MODULE.evaluate_dataframe_aliases(query, {"df1": frame})
    reads = MODULE.terminal_reads(query)
    return reads, [MODULE.resolve_read(read, env, {"df1": frame}) for read in reads]


def test_resolves_filtered_values_read_to_original_coordinate() -> None:
    frame = pd.DataFrame({"0": ["Khác", "Doanh thu", "Doanh thu phụ"], "1": ["1", "200", "3"]})
    query = """
_r = df1[df1['0'].str.contains('Doanh thu', regex=False)]
result = float(_r['1'].values[0])
"""
    reads, resolved = _resolve(query, frame)
    assert len(reads) == 1
    item, reason = resolved[0]
    assert reason is None
    assert item["source_row"] == 1
    assert item["source_column"] == 1
    assert item["source_label"] == "Doanh thu"
    assert item["raw"] == "200"


def test_resolves_filtered_iloc_read_and_negative_column() -> None:
    frame = pd.DataFrame({"0": ["2019", "2025"], "1": ["10", "20"], "2": ["100", "250"]})
    query = """
filtered = df1[df1.iloc[:, 0] == '2025']
result = float(filtered.iloc[0, -1])
"""
    reads, resolved = _resolve(query, frame)
    assert len(reads) == 1
    item, reason = resolved[0]
    assert reason is None
    assert item["source_row"] == 1
    assert item["source_column"] == 2
    assert item["raw"] == "250"


def test_two_predicate_filter_maps_to_single_source_row() -> None:
    frame = pd.DataFrame({
        "0": ["MNS Meat Hà Nam", "MNS Meat Hà Nam", "MEATDeli"],
        "1": ["Bán hàng hóa", "Mua hàng hóa", "Mua hàng hóa"],
        "2": ["1", "52", "3"],
    })
    query = """
filtered = df1[(df1.iloc[:, 0].str.contains('MNS Meat Hà Nam', regex=False)) & (df1.iloc[:, 1].str.contains('Mua hàng hóa', regex=False))]
result = float(filtered.iloc[0, 2])
"""
    reads, resolved = _resolve(query, frame)
    assert len(reads) == 1
    item, reason = resolved[0]
    assert reason is None
    assert item["source_row"] == 1
    assert item["raw"] == "52"


def test_resolves_iloc_then_column_label() -> None:
    frame = pd.DataFrame({"0": ["Cen Vĩnh Phúc"], "1": ["100%"], "2": ["51%"]})
    query = """
filtered = df1[df1['0'].str.contains('Cen Vĩnh Phúc', regex=False)]
result = float(filtered.iloc[0]['2'].replace('%', ''))
"""
    reads, resolved = _resolve(query, frame)
    assert len(reads) == 1
    item, reason = resolved[0]
    assert reason is None
    assert item["source_column_label"] == "2"
    assert item["raw"] == "51%"


def test_resolves_filtered_aggregate_source_cells() -> None:
    frame = pd.DataFrame({"0": ["Tiền mặt", "Tiền gửi", "Khác"], "1": ["10", "20", "99"]})
    query = """
filtered = df1[df1['0'].isin(['Tiền mặt', 'Tiền gửi'])]
result = filtered['1'].astype(float).sum()
"""
    reads, resolved = _resolve(query, frame)
    assert len(reads) == 1
    item, reason = resolved[0]
    assert reason is None
    assert item["source_rows"] == [0, 1]
    assert item["raw_values"] == ["10", "20"]


def test_resolves_selector_existence_reads() -> None:
    frame = pd.DataFrame({"0": ["Doanh thu nhà số 2", "Khác"], "1": ["1", "2"]})
    query = """
filtered = df1[df1['0'].str.contains('nhà số 2', regex=False)]
result = int(not filtered.empty)
"""
    reads, resolved = _resolve(query, frame)
    assert len(reads) == 1
    item, reason = resolved[0]
    assert reason is None
    assert item["matched_rows"] == 1
    assert item["source_rows"] == [0]
