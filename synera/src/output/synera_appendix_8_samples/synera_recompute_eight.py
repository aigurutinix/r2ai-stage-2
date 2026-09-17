"""Post-review reconstruction, NOT the pandas_query originally submitted.

Run: python synera_recompute_eight.py --package PATH_TO_UNPACKED_PACKAGE
Requires pandas. Reads only ticker/year/raw_column/value to calculate results;
submitted answers are loaded separately for comparison after calculation.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import pandas as pd

IDS = (363, 364, 367, 368, 369, 370, 371, 372)


def calculate(qid: int, evidence: pd.DataFrame):
    # This explicit allowlist excludes computed_answer, source_id and all labels.
    raw = evidence.loc[:, ['ticker', 'year', 'raw_column', 'value']].copy()
    if raw.duplicated(['ticker', 'year', 'raw_column']).any():
        raise ValueError('Duplicate source keys')
    p = raw.pivot(index=['ticker', 'year'], columns='raw_column', values='value')
    def at(year):
        return p.xs(year, level='year').copy()
    def quick(d):
        return (d.current_assets - d.inventory) / d.current_liabilities
    def cover(d):
        return (d.pbt + d.interest_expense) / d.interest_expense
    if qid == 363:
        d = p.xs('KBC', level='ticker').loc[2016:2020].copy()
        d['D/E'] = d.liabilities / d.equity
        d['coverage'] = cover(d)
        winner = int(d['D/E'].idxmax())
        trace = d[['D/E', 'coverage']].reset_index()
        result = d.loc[winner, 'coverage']
        choice = f'KBC, {winner}'
    elif qid == 364:
        a, b = at(2020), at(2021)
        d = pd.DataFrame({'CFO_2020': a.cfo, 'CFO_2021': b.cfo})
        d['eligible'] = (d.CFO_2020 > 0) & (d.CFO_2021 > 0)
        d['growth_pct'] = (b.net_revenue / a.net_revenue - 1) * 100
        d['accrual_pct'] = (b.npat - b.cfo) / ((a.total_assets + b.total_assets) / 2) * 100
        winner = d.loc[d.eligible, 'growth_pct'].idxmax()
        result = d.loc[winner, 'accrual_pct']
        choice = f'{winner}, 2021'
        trace = d.reset_index()
    elif qid == 367:
        a, b = at(2024), at(2025)
        d = pd.DataFrame({'CFO_positive_both': (a.cfo > 0) & (b.cfo > 0),
                          'revenue_declines': b.net_revenue < a.net_revenue})
        d['gross_margin_pct'] = b.gross_profit / b.net_revenue * 100
        d['net_margin_pct'] = b.npat / b.net_revenue * 100
        d['gap_pp'] = (b.gross_profit - b.npat) / b.net_revenue * 100
        keep = d.CFO_positive_both & d.revenue_declines
        result = d.loc[keep, 'gap_pp'].mean()
        choice = ', '.join(d.index[keep]) + ', 2025'
        trace = d.reset_index()
    elif qid == 368:
        d = at(2022)
        t = pd.DataFrame({'quick_ratio': quick(d), 'net_margin_pct': d.npat / d.net_revenue * 100})
        median = t.quick_ratio.median()
        t['eligible'] = t.quick_ratio < median
        result = t.loc[t.eligible, 'net_margin_pct'].mean()
        choice = ', '.join(t.index[t.eligible]) + f'; median={median:.15g}'
        trace = t.reset_index()
    elif qid == 369:
        a, b = at(2022), at(2023)
        d = pd.DataFrame({'quick_2022': quick(a),
                          'gross_2022_pct': a.gross_profit / a.net_revenue * 100,
                          'gross_2023_pct': b.gross_profit / b.net_revenue * 100})
        d['eligible'] = d.quick_2022 < d.quick_2022.median()
        d['change_pp'] = d.gross_2023_pct - d.gross_2022_pct
        d['coverage_2023'] = cover(b)
        winner = d.loc[d.eligible, 'change_pp'].idxmax()
        result = d.loc[winner, 'coverage_2023']
        choice = f'{winner}, 2023'
        trace = d.reset_index()
    elif qid == 370:
        a, b, c = at(2022), at(2023), at(2024)
        d = pd.DataFrame({'CFO_positive_3y': (a.cfo > 0) & (b.cfo > 0) & (c.cfo > 0)})
        d['CAGR_pct'] = ((c.net_revenue / a.net_revenue) ** 0.5 - 1) * 100
        d['net_margin_2024_pct'] = c.npat / c.net_revenue * 100
        winner = d.loc[d.CFO_positive_3y, 'CAGR_pct'].idxmax()
        result = d.loc[winner, 'net_margin_2024_pct']
        choice = f'{winner}, 2024'
        trace = d.reset_index()
    elif qid == 371:
        d = at(2024)
        t = pd.DataFrame({'CFO_positive': d.cfo > 0,
                          'gross_margin_pct': d.gross_profit / d.net_revenue * 100,
                          'coverage': cover(d)})
        winner = t.loc[t.CFO_positive, 'gross_margin_pct'].idxmax()
        result = t.loc[winner, 'coverage']
        choice = f'{winner}, 2024'
        trace = t.reset_index()
    elif qid == 372:
        d = p.xs('VRE', level='ticker').copy()
        d['quick_ratio'] = quick(d)
        d['CFO_to_short_debt'] = d.cfo / d.current_liabilities
        selected = int(d.loc[2021:2024, 'quick_ratio'].idxmin())
        result = d.loc[selected + 1, 'CFO_to_short_debt']
        choice = f'min quick ratio: {selected}; result year: {selected + 1}'
        trace = d[['quick_ratio', 'CFO_to_short_debt']].reset_index()
    else:
        raise ValueError(qid)
    result = float(result)
    if not math.isfinite(result):
        raise ValueError(f'Non-finite result for {qid}')
    return result, choice, trace


def audit_package(root: Path):
    submitted = {r['id']: r for r in json.loads((root / 'submitted_samples.json').read_text(encoding='utf-8'))}
    reports = []
    for qid in IDS:
        row = submitted[qid]
        evidence = pd.read_csv(root / 'evidence_original' / Path(row['evidence'][0]['csv_path']).name)
        result, choice, trace = calculate(qid, evidence)
        no_answer = evidence.drop(columns=['computed_answer'], errors='ignore')
        without, _, _ = calculate(qid, no_answer)
        poisoned = evidence.copy()
        poisoned['computed_answer'] = -987654321.0
        poison_result, _, _ = calculate(qid, poisoned)
        altered = evidence.copy()
        # Alter a metric that affects the output, leaving selection fixed.
        metric = 'npat' if qid in (364, 367, 368, 370) else 'cfo' if qid == 372 else 'pbt'
        altered.loc[altered.raw_column == metric, 'value'] *= 1.1
        changed_result, _, _ = calculate(qid, altered)
        expected = float(row['answer'])  # Comparison only, after calculation.
        reports.append({'id': qid, 'question': row['question'], 'submitted': expected,
                        'recomputed': result, 'absolute_error': abs(result - expected),
                        'matches': math.isclose(result, expected, rel_tol=1e-12, abs_tol=1e-10),
                        'selection': choice, 'shape': list(evidence.shape),
                        'answer_column_removed_same': without == result,
                        'answer_column_poisoned_same': poison_result == result,
                        'sensitivity_metric': metric, 'sensitivity_result': changed_result,
                        'source_value_change_affects_result': changed_result != result,
                        'trace': json.loads(trace.to_json(orient='records', double_precision=15))})
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    reports = audit_package(args.package)
    print(json.dumps(reports, ensure_ascii=False, indent=2, allow_nan=False))
    if not all(r['matches'] and r['answer_column_removed_same'] and r['answer_column_poisoned_same']
               and r['source_value_change_affects_result'] for r in reports):
        raise SystemExit(1)
