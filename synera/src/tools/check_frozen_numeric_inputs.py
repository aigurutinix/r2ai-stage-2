"""Check normalized numeric cells against preserved raw text, without changing evidence."""
import argparse,io,json,math,sys,zipfile
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.text import parse_vn_number

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args();counts=Counter();issues=[]
    with zipfile.ZipFile(args.baseline) as z:
        for n in z.namelist():
            if not n.endswith('.csv'):continue
            d=pd.read_csv(io.BytesIO(z.read(n)),dtype=str,keep_default_na=False)
            raw_col=next((c for c in ['raw_value','raw'] if c in d),None)
            if raw_col is None:counts['files_without_raw_text_column']+=1;continue
            for i,row in d.iterrows():
                raw=parse_vn_number(row[raw_col])
                for col in ['raw_number','value','vnd_value']:
                    if col not in d:continue
                    try:value=float(row[col])
                    except ValueError:counts['unparseable_normalized']+=1;continue
                    if raw is None:counts['raw_text_not_parsed']+=1;continue
                    scale=float(row['source_scale']) if row.get('source_scale') else None
                    if col=='raw_number':expected=raw
                    elif scale is not None:expected=raw*scale
                    else:expected=None
                    if expected is not None and math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-7):
                        counts['matches_raw_and_scale']+=1;continue
                    if raw==0 and value==0:counts['zero_matches']+=1;continue
                    ratio=value/raw if raw else None
                    if ratio is not None and any(math.isclose(ratio,10.**k,rel_tol=1e-10) for k in range(-12,13)):
                        counts['power_of_ten_scale_needs_context']+=1;continue
                    issues.append({'csv_path':n,'row':int(i),'column':col,'raw_text':row[raw_col],
                        'normalized':value,'source_scale':scale,'ratio':ratio,
                        'doc_id':row.get('doc_id'),'table_id':row.get('table_id'),'row_idx':row.get('row_idx')})
                    counts['requires_review']+=1
    args.report.write_text(json.dumps({'counts':dict(counts),'issues':issues,
        'note':'This checks normalization, not whether the original source cell answers the question. No cells were changed.'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(counts)))

if __name__=='__main__':main()
