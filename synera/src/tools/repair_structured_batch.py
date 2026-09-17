"""Local schema-constrained program repair; no prior submission answers or ID recipes."""
import argparse,hashlib,io,json,sys,time,urllib.request,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.compliant_query import SYSTEM,make_request,compile_plan,visible_frames,output_contract
from road2ai_vifinqa.plan_normalization import normalized_json,vectorize_assign,normalize_display_conversion
from road2ai_vifinqa.expression_plan import inline_plan

NAMES=[f'p{i}' for i in range(20)]
SCHEMA={'type':'object','properties':{
    'steps':{'type':'array','minItems':1,'maxItems':20,'items':{'type':'object','properties':{
        'name':{'type':'string','enum':NAMES},'expression':{'type':'string'}},
        'required':['name','expression'],'additionalProperties':False}},
    'result_step':{'type':'string','enum':NAMES},
    'expression_unit':{'type':'string','enum':['VND','requested']},
    'missing_inputs':{'type':'array','items':{'type':'string'}}},
    'required':['steps','result_step','expression_unit','missing_inputs'],'additionalProperties':False}
FORMAT='''Use the following output format instead of the earlier final-expression format:
{"steps":[{"name":"p0","expression":"df1['value'].sum()"}],"result_step":"p0","expression_unit":"VND","missing_inputs":[]}
The example is format only. Derive the program from the actual question.
Each step must be executable Pandas code, never the evaluated numerical result.
The result_step must name an existing step containing the requested scalar calculation.
Do not compute or paste the numerical answer in any expression. Read all numerical
facts from DataFrames; years, unit conversions and mathematical constants are allowed.
Keep the program short. Select scalar cells with .iloc[0] on each side of arithmetic
when rows represent different inputs. For multi-company panels pivot once, align
by ticker/year, apply the question's filters, then aggregate. Do not paste source
amounts as literals. Parenthesize each Boolean comparison separately before & or |.
For explicit currency output_contract, compute VND in the result step: conversion
to the requested display unit is performed by the compiler. No explanation field.
If a required fact is absent, declare it in missing_inputs instead of inventing it.
Evidence rows have already been retrieved for the question. Semantically equivalent
Vietnamese labels may differ across companies; use ticker/source order/retrieval_phrase
instead of demanding identical label text. A comparative pair from one annual report
may share report_year: use column_header and col_idx, where the current-period column
normally precedes the prior-period comparative column. Total borrowings means short-term
plus long-term borrowings. Operating leverage for two periods is the relative change in
operating profit divided by the relative change in net revenue. Interest coverage is
(pbt + positive interest expense) / positive interest expense.
When selector candidates cover many entities/years but target-metric rows exist
only for one entity/year, those target rows belong to the selector winner; do not
declare them missing for losing candidates. For a current/prior pair from the same
report with no separate year label, the smaller numeric col_idx is the current
period and the larger col_idx is the prior comparative period. To select a year by
debt/equity, pivot index=year and columns=raw_column, calculate liabilities/equity,
take idxmax, then read pbt and interest_expense at that year. If a selector measure
has multiple source rows per year, groupby year and sum before idxmax/idxmin.
For an "among companies, choose the company with the largest/smallest selector,
then calculate A/B" question: first filter only selector rows, use idxmax/idxmin
to obtain the winning ticker, then filter target A and B rows for that ticker and
divide their normalized numeric values. Rows for A and B may use the same generic
row label such as TỔNG CỘNG; distinguish them by table_context, source order or
retrieval_phrase. Never repeatedly filter the same row set.
'''

def program_input(question,frames):
    """Show computational inputs once; all original DataFrame columns still exist."""
    views=visible_frames(frames);data={}
    wanted={'ticker','year','report_year','raw_column','row_label','label','row_idx','col_idx','column_header',
            'retrieval_phrase','source_id','raw_number','source_scale','value','vnd_value','sector_loans','total_industry_loans'}
    for name,d in views.items():
        cols=[c for c in d if c in wanted]
        if 'label' in cols and 'row_label' in cols and d['label'].fillna('').equals(d['row_label'].fillna('')):
            cols.remove('label')
        if 'ticker' not in cols or 'year' not in cols and 'report_year' not in cols:
            if 'doc_id' in d:cols.append('doc_id')
        if len(d)<=4 and 'column_header' in d:cols.append('column_header')
        view=d[cols]
        common={c:json.loads(view[[c]].iloc[:1].to_json(orient='records',force_ascii=False,double_precision=15))[0][c]
                for c in view if len(view)>1 and view[c].nunique(dropna=False)==1}
        varying=view.drop(columns=list(common))
        if len(varying)>100:
            compact_cols=[c for c in ('ticker','year','report_year','raw_column','value','raw_number','source_scale') if c in varying]
            if 'raw_column' not in compact_cols:
                compact_cols += [c for c in ('row_label','label','col_idx') if c in varying and c not in compact_cols]
            data[name]={'columns':list(d.columns),'constant_columns':common,'numeric_view_columns':compact_cols,
                        'rows':json.loads(varying[compact_cols].to_json(orient='values',force_ascii=False,double_precision=15))}
        else:
            data[name]={'columns':list(d.columns),'constant_columns':common,
                        'rows':json.loads(varying.to_json(orient='records',force_ascii=False,double_precision=15))}
    payload={'question':question,'dataframes':data,
        'note':'Shown columns are a compact view. Every column listed in columns exists. Constant columns apply to EVERY row. Source values have not been computed into an answer. Use exact labels and DataFrame reads.'}
    contract=output_contract(question)
    if contract:payload['output_contract']=contract
    return json.dumps(payload,ensure_ascii=False,separators=(',',':'))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=4)
    ap.add_argument('--attempts',type=int,default=2)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--max-tokens',type=int,default=2048)
    ap.add_argument('--include-unseen',action='store_true')
    ap.add_argument('--thinking',action='store_true')
    ap.add_argument('--fresh',action='store_true',help='Do not replay the previous failed program')
    ap.add_argument('--seed',type=int,default=20260907)
    ap.add_argument('--temperature',type=float,default=0.0)
    args=ap.parse_args()
    history=args.run/'prior_failures';history.mkdir(exist_ok=True)
    jobs=[]
    with zipfile.ZipFile(args.baseline) as archive:
        for row in json.loads(archive.read('submission.json')):
            frames={e['variable']:pd.read_csv(io.BytesIO(archive.read(e['csv_path']))) for e in row['evidence']}
            prompt,key=make_request(row['question'],frames);path=args.run/'generations'/(key+'.json')
            if path.exists():
                saved=path.read_bytes();cached=json.loads(saved)
                if cached.get('ok') and '.merge(' in cached.get('query',''):
                    try:
                        inline_plan(cached['plan'].get('steps',[]),cached['plan']['expression'],frames=set(frames),
                                    columns={c for d in frames.values() for c in d})
                    except ValueError as exc:
                        preserved=history/(key+'_'+hashlib.sha256(saved).hexdigest()[:12]+'.json')
                        if not preserved.exists():preserved.write_bytes(saved)
                        cached['ok']=False
                        cached['logs'].append({'phase':'static_merge_revalidation','attempt':len(cached['logs']),'error':str(exc)})
                        path.write_text(json.dumps(cached,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            if (path.exists() and not json.loads(path.read_text(encoding='utf-8')).get('ok')) or (args.include_unseen and not path.exists()):
                jobs.append((key,row,frames,prompt,path))
    jobs.sort(key=lambda item:item[0]);jobs=jobs[:args.limit] if args.limit else jobs
    def repair(item):
        key,row,frames,prompt,path=item
        if path.exists():
            original=path.read_bytes();record=json.loads(original)
        else:
            record={'ok':False,'key':key,'prompt':prompt,'logs':[]}
            original=json.dumps(record,ensure_ascii=False).encode()
        if record.get('ok'):return True
        saved=history/(key+'_'+hashlib.sha256(original).hexdigest()[:12]+'.json')
        if not saved.exists():saved.write_bytes(original)
        schema_hints={name:{c:d[c].dropna().unique().tolist() for c in ('ticker','year','report_year','raw_column','row_label','label')
            if c in d and d[c].nunique()<=60} for name,d in visible_frames(frames).items()}
        compact=program_input(row['question'],frames)
        messages=[{'role':'system','content':SYSTEM+'\n'+FORMAT},
                  {'role':'user','content':compact+'\nExact available categories (never invent a label or column): '+json.dumps(schema_hints,ensure_ascii=False,default=str)}]
        if record['logs'] and not args.fresh:
            previous=record['logs'][-1]
            if previous.get('phase')=='structured_source_repair' and previous.get('model_input')==compact:
                choice=previous.get('response',{}).get('choices',[{}])[0]
                content=choice.get('message',{}).get('content')
                if content and choice.get('finish_reason')=='stop':
                    messages.append({'role':'assistant','content':content})
                    messages.append({'role':'user','content':'This prior program failed: '+str(previous.get('error'))+'. Correct its specific error while retaining a source-based calculation.'})
        for attempt in range(args.attempts):
            started=time.monotonic()
            body={'model':'Qwen3.5-9B','messages':messages,'temperature':args.temperature,'seed':args.seed,
                  'max_tokens':args.max_tokens,'stream':False,'chat_template_kwargs':{'enable_thinking':args.thinking},
                  'response_format':{'type':'json_schema','json_schema':{'name':'source_program','strict':True,'schema':SCHEMA}}}
            log={'phase':'structured_source_repair','attempt':len(record['logs']),
                 'format_instruction':FORMAT,'schema':SCHEMA,'max_tokens':args.max_tokens,'thinking':args.thinking,
                 'model_input':compact,'schema_hints':schema_hints}
            content=''
            try:
                request=urllib.request.Request('http://127.0.0.1:8095/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=600) as response:raw=json.load(response)
                log['response']=raw;choice=raw['choices'][0];content=choice['message'].get('content') or ''
                if choice.get('finish_reason')!='stop':raise ValueError('Truncated structured program')
                generated=normalized_json(content)
                plan={**generated,'expression':generated['result_step']}
                plan=normalize_display_conversion(vectorize_assign(plan),row['question'],frames)
                query,answer=compile_plan(plan,frames,row['question'])
                record.update(ok=True,plan=plan,query=query,answer=answer,method='structured_local_source_repair')
            except Exception as exc:
                log['error']=f'{type(exc).__name__}: {exc}'
                if content:messages.append({'role':'assistant','content':content})
                messages.append({'role':'user','content':'The program failed: '+log['error']+'. Generate a corrected complete program using the given source data. Never substitute a literal answer.'})
            log['elapsed']=time.monotonic()-started;record['logs'].append(log)
            path.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            if record.get('ok'):break
        return bool(record.get('ok'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for done,future in enumerate(as_completed([pool.submit(repair,item) for item in jobs]),1):
            print(json.dumps({'done':done,'total':len(jobs),'ok':future.result()}),flush=True)

if __name__=='__main__':main()
