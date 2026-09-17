"""Ask the local model to return a source-reading final expression for its own plan."""
import argparse, hashlib, io, json, sys, time, urllib.request, zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.compliant_query import make_request, compile_plan
from road2ai_vifinqa.plan_normalization import normalized_json, vectorize_assign

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    history=args.run/'prior_failures';history.mkdir(exist_ok=True)
    jobs=[]
    with zipfile.ZipFile(args.baseline) as archive:
        for row in json.loads(archive.read('submission.json')):
            frames={e['variable']:pd.read_csv(io.BytesIO(archive.read(e['csv_path']))) for e in row['evidence']}
            prompt,key=make_request(row['question'],frames)
            path=args.run/'generations'/(key+'.json')
            if not path.exists():continue
            record=json.loads(path.read_text(encoding='utf-8'))
            if record.get('ok') or record['logs'][-1].get('error')!='ValueError: Expression does not read evidence':continue
            try:
                plan=normalized_json(record['logs'][-1]['response']['choices'][0]['message']['content'])
                if not plan.get('steps'):continue
            except Exception:continue
            jobs.append((key,row,frames,prompt,path,plan))
    def job(item):
        key,row,frames,prompt,path,plan=item
        original=path.read_bytes();record=json.loads(original)
        if record.get('ok'):return True
        model_input=prompt+'\nThe model already generated these intermediate steps: '+json.dumps(plan['steps'],ensure_ascii=False)
        instruction=('Return JSON with a single field "expression": a short executable Pandas expression that gives '
            'the requested scalar by referring to the supplied intermediate step names or source DataFrames. '
            'You must use the program and source evidence to calculate the answer at execution time. '
            'Never return a numerical answer literal. Do not repeat the steps. If the intermediate '
            'steps do not suffice, refer directly to the source DataFrames. Respect output_contract units.')
        body={'model':'Qwen3.5-9B','messages':[{'role':'system','content':instruction},{'role':'user','content':model_input}],
              'temperature':0,'seed':20260907,'max_tokens':256,'stream':False,
              'chat_template_kwargs':{'enable_thinking':False},
              'response_format':{'type':'json_object'}}
        log={'phase':'local_output_expression_repair','attempt':len(record['logs']),
             'instruction':instruction,'input_steps':plan['steps'],'settings':{k:v for k,v in body.items() if k!='messages'}}
        start=time.monotonic()
        try:
            request=urllib.request.Request('http://127.0.0.1:8095/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=300) as response:raw=json.load(response)
            log['response']=raw
            if raw['choices'][0].get('finish_reason')!='stop':raise ValueError('Truncated expression')
            patch=json.loads(raw['choices'][0]['message']['content'])
            revised=vectorize_assign({**plan,'expression':patch['expression']})
            query,answer=compile_plan(revised,frames,row['question'])
            record.update(ok=True,plan=revised,query=query,answer=answer,method='local_output_expression_repair')
        except Exception as exc:log['error']=f'{type(exc).__name__}: {exc}'
        log['elapsed']=time.monotonic()-start
        archived=history/(key+'_'+hashlib.sha256(original).hexdigest()[:12]+'.json')
        if not archived.exists():archived.write_bytes(original)
        record['logs'].append(log)
        path.write_text(json.dumps(record,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        return bool(record.get('ok'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,future in enumerate(as_completed([pool.submit(job,j) for j in jobs]),1):
            print(json.dumps({'done':i,'total':len(jobs),'ok':future.result()}),flush=True)

if __name__=='__main__':main()
