"""Reconstruct source numbers at query runtime; never trust preweighted value cells."""
from __future__ import annotations
import pandas as pd

def source_value_expression(name,columns):
    if not {'value','raw_value','source_scale'}<=set(columns):return None
    if 'raw_number' in columns:
        reconstructed=f"({name}['raw_number'] * {name}['source_scale'])"
        if 'raw' in columns:
            # Concatenated CSVs also contain original panel rows with no fields
            # from the raw_number schema. Their normalized source values remain
            # valid; the reconstruction mask must operate row by row.
            panel=f"({name}['source_scale'].isna() & {name}['raw_value'].isna() & {name}['raw_number'].isna() & {name}['raw'].notna())"
            return f"({reconstructed}.where(~{panel}, {name}['value']))"
        return reconstructed
    original=f"{name}['raw_value'].fillna('').astype('string').str.strip()"
    # OCR sometimes joins the current and comparative amount in a single cell.
    # Use the leading complete amount, preserving its sign and decimal marks.
    leading=(f"{original}.str.replace(r'^(\\(?[-−]?\\d[\\d.,]*[.,]\\d[\\d.,]*\\)?%?)(?:\\s+\\(?[-−]?\\d|(?<=\\))\\().*$', r'\\1', regex=True)")
    text=(f"{leading}.str.replace('−', '-', regex=False)"
          ".str.replace(r'^\\((.*)\\)$', r'-\\1', regex=True)"
          ".str.replace(r'\\s+', '', regex=True).str.replace('%', '', regex=False)"
          ".str.replace(r'^[–—-]$', '0', regex=True)")
    english=f"({text}.str.contains(r'^-?\\d{{1,3}}(?:,\\d{{3}})+(?:\\.\\d+)?$', regex=True) & ~{original}.str.contains('%', regex=False))"
    vietnamese=f"({text}.str.contains(r'^-?\\d{{1,3}}(?:\\.\\d{{3}})+(?:,\\d+)?$', regex=True) & ~{original}.str.contains('%', regex=False))"
    decimal=f"{text}.str.replace(',', '.', regex=False)"
    grouped_vn=f"{text}.str.replace('.', '', regex=False).str.replace(',', '.', regex=False)"
    grouped_en=f"{text}.str.replace(',', '', regex=False)"
    cleaned=f"({decimal}.where(~({vietnamese}), {grouped_vn}).where(~({english}), {grouped_en}))"
    return f"(pd.to_numeric({cleaned}, errors='coerce') * pd.to_numeric({name}['source_scale'], errors='coerce'))"

def _metadata_assignments(name,columns):
    assignments=[]
    if 'doc_id' in columns:
        doc=f"{name}['doc_id'].astype('string')"
        if 'ticker' in columns:assignments.append(f"ticker={name}['ticker'].fillna({doc}.str.split('_').str[0])")
        if 'report_year' in columns:
            assignments.append(f"report_year={name}['report_year'].fillna(pd.to_numeric({doc}.str.extract(r'financial_statements_(\\d{{4}})',expand=False),errors='coerce'))")
    if {'row_label','label'}<=set(columns):
        assignments.append(f"row_label={name}['row_label'].fillna({name}['label'])")
        assignments.append(f"label={name}['label'].fillna({name}['row_label'])")
    return assignments

def normalize_view(name,frame):
    expression=source_value_expression(name,frame.columns)
    assignments=_metadata_assignments(name,frame.columns)
    if expression is not None:assignments.insert(0,f'value={expression}')
    if not assignments:return frame.copy()
    return eval(f"{name}.assign({','.join(assignments)})",{'__builtins__':{},'pd':pd},{name:frame})

def protected_view_expression(name,columns):
    base=f'{name}[{list(columns)!r}]'
    expression=source_value_expression(name,columns)
    assignments=_metadata_assignments(name,columns)
    if expression is not None:assignments.insert(0,f'value={expression}')
    return f"({base}.assign({','.join(assignments)}))" if assignments else base
