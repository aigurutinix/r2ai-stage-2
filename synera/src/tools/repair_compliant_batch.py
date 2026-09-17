"""Generic local-model repair using source evidence and execution errors only."""
import argparse,hashlib,io,json,sys,time,urllib.request,zipfile
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.compliant_query import SYSTEM,make_request,compile_plan
from road2ai_vifinqa.plan_normalization import normalized_json,vectorize_assign

GUIDANCE='''Repair the calculation generically from the question and source data.
You must not return a prepared numerical answer or read any answer column.
Use at most 12 short steps named p0, p1, p2, etc.; put EACH in its own JSON object.
For long-format financial panels use one pivot, NOT dozens of separate filters.
Generic pattern (replace YEAR and metric names with actual question/evidence):
p0 = df.pivot_table(index='ticker', columns=['year','raw_column'], values='value', aggfunc='first')
p1 = p0[(YEAR,'npat')] / p0[(YEAR,'net_revenue')]
p2 = p0[(YEAR,'cfo')] > 0
final expression = float(p1.loc[p2].mean()*100)
This is only a schema pattern, not the formula for the current question.
If raw_column is absent, use existing labels to select the required source rows.
For a difference of two source cells, select each exact metric/year/ticker and
use .iloc[0] on EACH side. Do not subtract two differently indexed Series.
For a group mean keep the entire company-indexed ratio Series, apply all question
conditions, THEN call .mean(). For two-year panels use year/metric MultiIndex
columns to align companies. Do not compare a DataFrame to itself accidentally.
No lambda, apply, comprehensions or statements. .assign needs explicit Series.
Use normalized numeric inputs, not raw display text. If output_contract says
VND, include "expression_unit":"VND" and do NOT divide by the display divisor.
If required data is missing, state missing_inputs truthfully. Return complete JSON.
'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=1)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--attempts',type=int,default=2)
    ap.add_argument('--base-url',default='http://127.0.0.1:8095')
    args=ap.parse_args()
    history=args.run/'prior_failures';history.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.baseline) as z:
        members={n:z.read(n) for n in z.namelist()}
    jobs=[]
    for r in json.loads(members['submission.json']):
        frames={e['variable']:pd.read_csv(io.BytesIO(members[e['csv_path']])) for e in r['evidence']}
        prompt,key=make_request(r['question'],frames);path=args.run/'generations'/(key+'.json')
        if path.exists() and not json.loads(path.read_text(encoding='utf-8')).get('ok'):
            jobs.append((key,r,frames,prompt,path))
    jobs.sort(key=lambda j:j[0]);jobs=jobs[:args.limit] if args.limit else jobs
    def repair(job):
        key,r,frames,prompt,path=job
        original=path.read_bytes();result=json.loads(original)
        if result.get('ok'):return r['id'],True
        oldpath=history/(key+'_'+hashlib.sha256(original).hexdigest()[:12]+'.json')
        if not oldpath.exists():oldpath.write_bytes(original)
        last=result['logs'][-1]
        content=last.get('response',{}).get('choices',[{'message':{}}])[0]['message'].get('content','') or ''
        messages=[{'role':'system','content':SYSTEM+'\n'+GUIDANCE},{'role':'user','content':prompt}]
        if content:messages.append({'role':'assistant','content':content})
        messages.append({'role':'user','content':'This attempt failed: '+str(last.get('error'))+'. Correct the program, not the evidence. Return all required JSON fields.'})
        for a in range(args.attempts):
            log={'attempt':len(result['logs']),'phase':'generic_error_repair','guidance':GUIDANCE}
            started=time.monotonic()
            try:
                body={'model':'Qwen3.5-9B','messages':messages,'temperature':0.0,'seed':20260907,
                      'max_tokens':4096,'stream':False,'chat_template_kwargs':{'enable_thinking':True},'cache_prompt':True}
                req=urllib.request.Request(args.base_url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(req,timeout=600) as response:raw=json.load(response)
                log['response']=raw
                choice=raw['choices'][0];content=choice['message'].get('content') or ''
                if choice.get('finish_reason')!='stop':raise ValueError('Truncated completion')
                plan=vectorize_assign(normalized_json(content))
                query,answer=compile_plan(plan,frames,r['question'])
                result.update(ok=True,plan=plan,query=query,answer=answer,method='local_model_error_repair')
            except Exception as exc:
                log['error']=f'{type(exc).__name__}: {exc}'
                if content:messages.append({'role':'assistant','content':content})
                messages.append({'role':'user','content':'Execution/validation error: '+log['error']+'. Fix the program with the same source evidence. Do not invent any missing inputs.'})
            log['elapsed']=time.monotonic()-started;result['logs'].append(log)
            path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            if result.get('ok'):break
        return r['id'],bool(result.get('ok'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(repair,j) for j in jobs]):
            qid,ok=f.result();print(json.dumps({'repair_audit_id':qid,'ok':ok}),flush=True)

if __name__=='__main__':main()
