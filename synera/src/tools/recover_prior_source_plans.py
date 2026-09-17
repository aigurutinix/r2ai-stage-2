"""Recover alternate model-generated source plans from earlier compliant runs.

Only plans are reused.  Every candidate is recompiled and executed against the
current immutable baseline evidence; stored answers and released queries are
never read as results.
"""
import argparse,hashlib,io,json,sys,zipfile
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.compliant_query import make_request,compile_plan
from road2ai_vifinqa.plan_normalization import normalized_json,vectorize_assign,normalize_display_conversion

def prompt_question(record):
    try:return json.loads(record.get('prompt','')).get('question')
    except Exception:return None

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--search-root',type=Path,required=True)
    args=ap.parse_args()
    candidates=defaultdict(list)
    for path in args.search_root.glob('*/generations/*.json'):
        if args.run.resolve() in path.resolve().parents:continue
        try:record=json.loads(path.read_text(encoding='utf-8'))
        except Exception:continue
        question=prompt_question(record)
        if question and record.get('ok') and isinstance(record.get('plan'),dict):
            candidates[question].append((path,record['plan']))
        if question:
            base_plan=None
            for log in record.get('logs',[]):
                try:
                    choice=log['response']['choices'][0]
                    if choice.get('finish_reason')!='stop':continue
                    plan=normalized_json(choice.get('message',{}).get('content') or '')
                    if log.get('phase')=='structured_source_repair' or ('expression' not in plan and 'result_step' in plan):
                        plan={**plan,'expression':plan['result_step']}
                    if log.get('phase')=='local_output_expression_repair':
                        if base_plan is None:continue
                        plan={**base_plan,'steps':log['input_steps'],'expression':plan['expression']}
                    else:base_plan=plan
                    candidates[question].append((path,plan))
                except Exception:continue
    recovered=[]
    history=args.run/'prior_failures';history.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.baseline) as archive:
        for row in json.loads(archive.read('submission.json')):
            frames={e['variable']:pd.read_csv(io.BytesIO(archive.read(e['csv_path']))) for e in row['evidence']}
            _,key=make_request(row['question'],frames)
            target=args.run/'generations'/(key+'.json')
            if not target.exists():continue
            current=json.loads(target.read_text(encoding='utf-8'))
            if current.get('ok'):continue
            for source,raw_plan in candidates.get(row['question'],[]):
                try:
                    plan=normalize_display_conversion(vectorize_assign(raw_plan),row['question'],frames)
                    query,answer=compile_plan(plan,frames,row['question'])
                except Exception:continue
                original=target.read_bytes()
                saved=history/(key+'_'+hashlib.sha256(original).hexdigest()[:12]+'.json')
                if not saved.exists():saved.write_bytes(original)
                current.update(ok=True,plan=plan,query=query,answer=answer,
                    method='recompiled_prior_source_plan',prior_source=str(source),
                    prior_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
                target.write_text(json.dumps(current,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
                recovered.append(row['id']);break
    print(json.dumps({'recovered':len(recovered),'ids_for_audit_only':recovered}))

if __name__=='__main__':main()
