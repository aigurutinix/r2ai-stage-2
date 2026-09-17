"""Independent immutable-evidence/replay audit; never uses historical answers as labels."""
from __future__ import annotations
import argparse
import ast
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.compliant_query import ALLOWED_COLUMNS,make_request,compile_plan
from road2ai_vifinqa.submission import evaluate_expression

def sha(b):return hashlib.sha256(b).hexdigest()
def members(path):
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip():
            raise ValueError('Invalid ZIP members/CRC')
        return {n:z.read(n) for n in z.namelist()}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--report',type=Path,required=True)
    args=ap.parse_args()
    before=members(args.baseline);after=members(args.candidate)
    assert set(before)==set(after),'Archive members changed'
    immutable={n:sha(b) for n,b in before.items() if n!='submission.json'}
    assert immutable=={n:sha(b) for n,b in after.items() if n!='submission.json'},'Evidence bytes changed'
    old=json.loads(before['submission.json']);new=json.loads(after['submission.json'])
    assert len(old)==len(new)==1012
    assert len({r['id'] for r in new})==1012
    old_by_id={r['id']:r for r in old}
    results=[];errors=[];warnings=[]
    for row in new:
        try:
            original=old_by_id[row['id']]
            assert {k:v for k,v in original.items() if k not in {'answer','pandas_query'}}=={
                k:v for k,v in row.items() if k not in {'answer','pandas_query'}},'Non-query metadata changed'
            frames={e['variable']:pd.read_csv(io.BytesIO(after[e['csv_path']])) for e in row['evidence']}
            prompt,key=make_request(row['question'],frames)
            cached=json.loads((args.run/'generations'/(key+'.json')).read_text(encoding='utf-8'))
            assert cached['ok'] and cached['prompt']==prompt and cached['key']==key
            query,answer=compile_plan(cached['plan'],frames,row['question'])
            assert query==row['pandas_query'] and answer==row['answer'],'Generation replay mismatch'
            assert math.isfinite(answer)
            for _ in range(2):
                fresh={e['variable']:pd.read_csv(io.BytesIO(after[e['csv_path']])) for e in row['evidence']}
                assert evaluate_expression(query,fresh)==answer,'Fresh CSV replay mismatch'
            # Counterfactual is diagnostic, not a gold-label test: counts/rankings can
            # legitimately remain stable under these changes and must be reviewed.
            changed={n:d.copy() for n,d in frames.items()}
            edited=0
            for d in changed.values():
                if {'value','raw_value','source_scale'} <= set(d) and 'raw_number' not in d:
                    # Released queries reconstruct value from the raw source text.
                    # Changing only the old value column would be a false-negative.
                    d['raw_value']=pd.Series([str(173.21+i*19.37) for i in range(len(d))],index=d.index)
                    edited+=1
                for c in {'raw_number','value','vnd_value','sector_loans','total_industry_loans'} & set(d):
                    if pd.api.types.is_numeric_dtype(d[c]):
                        d[c]=d[c]*pd.Series([1.13+i*.017 for i in range(len(d))],index=d.index)+7.31
                        edited+=1
            try:
                alt=float(evaluate_expression(query,changed)) if edited else answer
                sensitive=not math.isclose(alt,answer,rel_tol=1e-10,abs_tol=1e-12)
            except Exception:
                sensitive=None
            if sensitive is not True:warnings.append({'id':row['id'],'reason':'Review source counterfactual (rank/count/filter may be invariant)','sensitive':sensitive})
            results.append({'id':row['id'],'key':key,'fresh_replays':3,
                'hidden_columns_removed_and_poisoned_pass':True,'source_counterfactual_changed':sensitive})
        except Exception as exc:
            errors.append({'id':row['id'],'error':f'{type(exc).__name__}: {exc}'})
    report={'baseline_sha256':sha(args.baseline.read_bytes()),'candidate_sha256':sha(args.candidate.read_bytes()),
        'immutable_members':immutable,'rows':len(new),'passed':len(results),'errors':errors,'warnings':warnings,
        'results':results,'official_accuracy':None,
        'limitations':['Execution consistency is not proof of semantic accuracy.',
            'Organizer decides final eligibility; this audit does not guarantee acceptance.',
            'Existing evidence including unused computed columns was retained byte-for-byte as requested.']}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({'passed':len(results),'errors':len(errors),'warnings':len(warnings),'report':str(args.report)},ensure_ascii=False))
    if errors:raise SystemExit(1)

if __name__=='__main__':main()
