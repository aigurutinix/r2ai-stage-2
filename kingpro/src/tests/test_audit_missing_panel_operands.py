from __future__ import annotations

import ast

from scripts.audit_missing_panel_operands import _live_source_fields


def _live(code: str, fields: set[str]) -> set[str]:
    return _live_source_fields(ast.parse(code), fields)


def test_backward_slice_ignores_materialised_but_unused_metrics() -> None:
    code = """
df = pd.DataFrame(_rows)
df['cfo_margin'] = df['cfo'] / df['revenue']
df['fixed_assets_avg'] = (df['_prev_fixed_assets'] + df['fixed_assets']) / 2
df['fixed_assets_turnover'] = df['revenue'] / df['fixed_assets_avg']
filtered = df[df['fixed_assets_turnover'] < df['fixed_assets_turnover'].median()]
result = filtered['fixed_assets_turnover'].max()
"""
    fields = {'cfo', 'revenue', '_prev_fixed_assets', 'fixed_assets'}
    assert _live(code, fields) == {'revenue', '_prev_fixed_assets', 'fixed_assets'}


def test_backward_slice_keeps_selector_and_answer_dependencies() -> None:
    code = """
df = pd.DataFrame(_rows)
df['net_margin'] = df['npat'] / df['revenue']
df['cfo_to_npat'] = df['cfo'] / df['npat']
filtered = df[df['npat'] > 0].dropna(subset=['net_margin', 'cfo_to_npat'])
selected = filtered.loc[filtered['net_margin'].idxmax()]
result = selected['cfo_to_npat']
result = round(float(result), 2)
"""
    fields = {'revenue', 'npat', 'cfo', 'eps'}
    assert _live(code, fields) == {'revenue', 'npat', 'cfo'}


def test_backward_slice_tracks_attribute_selector_access() -> None:
    code = """
df = pd.DataFrame(_rows)
row = df.loc[df.short_term_borrowings.idxmax()]
result = row.cfo / row.revenue * 100
"""
    fields = {'short_term_borrowings', 'cfo', 'revenue', 'inventory'}
    assert _live(code, fields) == {'short_term_borrowings', 'cfo', 'revenue'}


def test_backward_slice_tracks_pivot_value_keyword() -> None:
    code = """
df = pd.DataFrame(_rows)
df['margin'] = df['profit'] / df['revenue']
margin = df.pivot(index='ticker', columns='year', values='margin').dropna()
result = margin.loc['AAA', 2023] - margin.loc['AAA', 2024]
"""
    fields = {'profit', 'revenue', 'unused'}
    assert _live(code, fields) == {'profit', 'revenue'}
