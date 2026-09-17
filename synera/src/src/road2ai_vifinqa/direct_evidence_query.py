"""Conservative semantic fast path: one raw source cell, explicit currency unit.

This is a generic retrieval/conversion operation, not a question/formula registry.
No prior query, answer or question identifier is accepted.
"""
import math
import re
import unicodedata
from .compliant_query import output_contract

def direct_plan(question,frames):
    if len(frames)!=1:return None
    name,frame=next(iter(frames.items()))
    if 'source_scale' not in frame:return None
    candidate=frame
    selector=''
    if len(candidate)!=1:
        candidate,selector=_unique_semantic_row(question,name,frame)
        if candidate is None:return None
    if 'raw_number' in candidate:
        source=f"({name}.loc[{selector}, 'raw_number'].iloc[0] * {name}.loc[{selector}, 'source_scale'].iloc[0])" if selector else f"{name}['raw_number'].iloc[0] * {name}['source_scale'].iloc[0]"
        number=float(candidate['raw_number'].iloc[0])
    elif {'raw_value','value'}<=set(candidate):
        # The compiler reconstructs this value from raw_value and source_scale.
        from .source_views import normalize_view
        number=float(normalize_view(name,candidate)['value'].iloc[0])
        source=f"{name}.loc[{selector}, 'value'].iloc[0]" if selector else f"{name}['value'].iloc[0]"
    else:return None
    q=''.join(c for c in unicodedata.normalize('NFD',question.lower().replace('đ','d')) if unicodedata.category(c)!='Mn')
    # Words such as "và" / "giảm" inside an exactly named source metric are
    # not arithmetic instructions (e.g. "tiền và các khoản tương đương tiền").
    if 'row_label' in candidate:
        label=''.join(c for c in unicodedata.normalize('NFD',str(candidate['row_label'].iloc[0]).lower().replace('đ','d')) if unicodedata.category(c)!='Mn')
        if len(label.split())>=3 and label in q:q=q.replace(label,' source_metric ',1)
    # Reject every comparison, arithmetic instruction, ratio, multi-entity or
    # selection/ranking request; these go through the general model instead.
    if re.search(r'\b(tang|giam|chenh|so voi|so sanh|ty le|ty so|he so|lan|phan tram|trung binh|binh quan|trung vi|cao nhat|thap nhat|lon nhat|nho nhat|tong cua|tong so|cong voi|tru di|nhan voi|chia|va|hon|kem|usd|do la)\b|%',q):return None
    if not re.search(r'\b(bao nhieu|may)\b',q):return None
    scale=float(candidate['source_scale'].iloc[0])
    if not math.isfinite(number) or not math.isfinite(scale) or scale<=0:return None
    # Currency questions are computed in VND then converted by the compiler.
    # For a non-currency question the source number is returned in its declared
    # physical scale (for example shares or a count); no prior answer is used.
    contract=output_contract(question)
    return {'steps':[],
        'expression':f"float({source})",
        'expression_unit':'VND' if contract else 'requested','missing_inputs':[],
        'explanation':'Generic single-source-cell lookup from the declared source scale; currency display conversion is applied only when explicitly requested.'}

def _fold(text):
    return ''.join(c for c in unicodedata.normalize('NFD',str(text).lower().replace('đ','d')) if unicodedata.category(c)!='Mn')

def selector_ratio_plan(question,frames):
    """Generic: select an entity by an extremal metric, then calculate A/B.

    Metric roles are inferred from question text versus source labels/context.  The
    function never receives a question id or an answer and keeps all arithmetic in
    the emitted Pandas program.
    """
    if len(frames) != 1:
        return None
    name,d = next(iter(frames.items()))
    needed={'ticker','source_id','vnd_value'}
    if not needed <= set(d) or d['ticker'].nunique(dropna=True) < 2:
        return None
    q=_fold(question)
    ratio=re.search(r'ty le giua (.+?) va (.+?) cua ',q)
    selector=re.search(r'co (.+?) (lon nhat|cao nhat|nho nhat|thap nhat)',q)
    if not ratio or not selector or 'trong so' not in q:
        return None
    numerator,denominator=ratio.group(1),ratio.group(2)
    selector_phrase=selector.group(1)
    direction='idxmin' if selector.group(2) in {'nho nhat','thap nhat'} else 'idxmax'
    text_cols=[c for c in ('row_label','label','table_context','column_header') if c in d]
    if not text_cols:
        return None
    stop={'gia','tri','tong','cong','cuoi','nam','cua','doanh','nghiep','cong','ty','hien','hanh','giua','phan','tap','doan'}
    def tokens(text):
        return {x for x in re.findall(r'[a-z0-9]+',_fold(text)) if len(x)>=3 and x not in stop and not x.isdigit()}
    def scores(phrase):
        want=tokens(phrase)
        return d.apply(lambda row: sum(len(t) for t in want & tokens(' '.join(str(row[c]) for c in text_cols))),axis=1)
    selector_scores=scores(selector_phrase)
    numerator_scores=scores(numerator)
    denominator_scores=scores(denominator)
    # Exactly one strongest semantic selector row per entity; target roles may
    # have rows only for the eventual winner, which is resolved inside Pandas.
    selector_ids=[]
    for _,group in d.assign(_score=selector_scores).groupby('ticker'):
        best=group['_score'].max()
        if best>0:
            selector_ids.append(str(group.loc[group['_score']==best,'source_id'].iloc[0]))
    if len(selector_ids)<2:
        return None
    ns=numerator_scores.max(); ds=denominator_scores.max()
    if ns<=0 or ds<=0:
        return None
    numerator_ids=[str(x) for x in d.loc[numerator_scores==ns,'source_id']]
    denominator_ids=[str(x) for x in d.loc[denominator_scores==ds,'source_id']]
    if set(numerator_ids) & set(denominator_ids):
        return None
    steps=[
        {'name':'p0','expression':f"{name}[{name}['source_id'].isin({selector_ids!r})]"},
        {'name':'p1','expression':"p0.groupby('ticker')['vnd_value'].max()"},
        {'name':'p2','expression':f"p1.{direction}()"},
        {'name':'p3','expression':f"{name}.loc[({name}['ticker'] == p2) & ({name}['source_id'].isin({numerator_ids!r})), 'vnd_value'].iloc[0]"},
        {'name':'p4','expression':f"{name}.loc[({name}['ticker'] == p2) & ({name}['source_id'].isin({denominator_ids!r})), 'vnd_value'].iloc[0]"},
    ]
    return {'steps':steps,'expression':'float(p3 / p4)','expression_unit':'requested','missing_inputs':[],
            'explanation':'Generic semantic selector-then-ratio program over source evidence; entity selection and arithmetic both execute in Pandas.'}

def _unique_semantic_row(question,name,frame):
    """Conservative generic matching of a question's metric to a unique source row.

    This never sees an ID or answer. It only applies when the same distinctive
    source-label tokens occur in the question and leave exactly one row.
    """
    label_col=next((c for c in ('row_label','label') if c in frame),None)
    if not label_col:return None,''
    stop={'cua','cong','ty','co','phan','tap','doan','nam','tai','ngay','den','la','bao','nhieu','theo','trong','voi','va','cac','khoan','cho','nguoi','mot','so','tong','cuoi','dau','dong','vnd','dong','trieu','ty','nghin'}
    qtokens={x for x in re.findall(r'[a-z0-9]+',_fold(question)) if len(x)>=3 and x not in stop and not x.isdigit()}
    if len(qtokens)<3:return None,''
    scores=[]
    for idx,value in frame[label_col].items():
        tokens={x for x in re.findall(r'[a-z0-9]+',_fold(value)) if len(x)>=3 and x not in stop and not x.isdigit()}
        scores.append((len(qtokens & tokens),idx))
    scores.sort(reverse=True)
    if not scores or scores[0][0]<3 or (len(scores)>1 and scores[0][0]-scores[1][0]<2):return None,''
    row=frame.loc[[scores[0][1]]]
    clauses=[f"{name}[{label_col!r}] == {str(row[label_col].iloc[0])!r}"]
    years=[int(x) for x in re.findall(r'(?<!\d)(?:19|20)\d{2}(?!\d)',question)]
    if len(set(years))==1 and 'year' in frame:
        clauses.append(f"{name}['year'] == {years[0]}")
        row=row.loc[row['year']==years[0]]
    if len(row)!=1:return None,''
    return row,' & '.join(f'({c})' for c in clauses)

def temporal_pair_plan(question,frames):
    """Two years of the same source metric; only explicit directional comparisons."""
    if len(frames)!=1:return None
    name,d=next(iter(frames.items()))
    if len(d)!=2 or not {'year','value','source_scale','label'}<=set(d):return None
    labels=d['label'].astype(str).map(_fold).str.replace(r'\s*\([*\d\s]+\)\s*',' ',regex=True).str.replace(r'\s+',' ',regex=True).str.strip()
    if labels.nunique(dropna=False)!=1 or ('ticker' in d and d['ticker'].nunique(dropna=False)!=1):return None
    if d['year'].nunique()!=2 or not all(math.isfinite(float(v)) for v in d['value']):return None
    q=''.join(c for c in unicodedata.normalize('NFD',question.lower().replace('đ','d')) if unicodedata.category(c)!='Mn')
    if re.search(r'\b(trung binh|binh quan|trung vi|nhat|ty so|he so|bien loi nhuan|roe|roa|cagr|lan)\b',q):return None
    years=list(dict.fromkeys(int(y) for y in re.findall(r'\b(?:19|20)\d{2}\b',q)))
    if len(years)!=2 or set(years)!=set(int(v) for v in d['year']):return None
    percentage=('%' in q or 'phan tram' in q) and 'diem phan tram' not in q
    contract=output_contract(question)
    if not percentage and contract is None:return None
    first,second=years
    # Scope is deliberately narrow. Ambiguous 'chênh lệch' goes to the model.
    if 'hieu so giua' in q:
        if percentage:return None
        left,right=first,second
    elif re.search(r'\b(be hon|nho hon|thap hon|it hon)\b',q):
        if percentage:return None
        left,right=second,first
    elif re.search(r'\b(lon hon|cao hon|nhieu hon)\b',q):
        if percentage:return None
        left,right=first,second
    elif 'so voi' in q and re.search(r'\b(tang|giam|thay doi)\b',q):
        left,right=first,second
        if left<=right:return None
        if re.search(r'\bgiam\b',q):left,right=right,left
    elif re.search(r'\btu\b.*\b(?:den|toi)\b',q) and re.search(r'\b(tang|giam|thay doi)\b',q):
        left,right=max(years),min(years)
        if re.search(r'\bgiam\b',q):left,right=right,left
    else:return None
    a=f"{name}.loc[{name}['year']=={left}, 'value'].iloc[0]"
    b=f"{name}.loc[{name}['year']=={right}, 'value'].iloc[0]"
    if percentage:
        # Both positive and negative changes use the earlier period as denominator.
        base=f"{name}.loc[{name}['year']=={min(years)}, 'value'].iloc[0]"
        expression=f'float(({a} - {b}) / {base} * 100)'
    else:expression=f'float({a} - {b})'
    return {'steps':[],'expression':expression,'expression_unit':'VND' if contract else 'requested',
        'missing_inputs':[], 'explanation':'Generic explicit temporal comparison of the same source metric; years and direction parsed from question text, no question ID or answer registry.'}

def direct_percentage_plan(question,frames):
    """Read a single source cell explicitly printed as a percentage, not a new ratio."""
    if len(frames)!=1 or not ('%' in question or 'phần trăm' in question.lower()):return None
    name,d=next(iter(frames.items()))
    if len(d)!=1 or 'raw_number' not in d:return None
    q=''.join(c for c in unicodedata.normalize('NFD',question.lower().replace('đ','d')) if unicodedata.category(c)!='Mn')
    if re.search(r'\b(tang|giam|chenh|so voi|so sanh|trung binh|binh quan|trung vi|nhat|va|hon|kem|thay doi)\b',q):return None
    source_text=' '.join(str(d[c].iloc[0]) for c in ['raw_value','column_header','row_label'] if c in d)
    if '%' not in source_text or not math.isfinite(float(d['raw_number'].iloc[0])):return None
    return {'steps':[],'expression':f"float({name}['raw_number'].iloc[0])",'expression_unit':'requested',
            'missing_inputs':[], 'explanation':'Generic lookup of a raw source percentage already printed with a percent sign; no ratio is precomputed.'}
