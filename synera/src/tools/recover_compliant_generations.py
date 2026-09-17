"""Revalidate model-written attempts after generic compiler support updates.

No new formulas are authored here. Existing model attempts are checked in order,
and original failed records are preserved for a complete provenance trail.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.compliant_query import make_request,extract_json,compile_plan
from road2ai_vifinqa.plan_normalization import normalized_json,vectorize_assign,normalize_display_conversion

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args()
    previous=args.run/'prior_failures';previous.mkdir(exist_ok=True)
    recovered=[]
    with zipfile.ZipFile(args.baseline) as z:
        for r in json.loads(z.read('submission.json')):
            frames={e['variable']:pd.read_csv(io.BytesIO(z.read(e['csv_path']))) for e in r['evidence']}
            prompt,key=make_request(r['question'],frames)
            path=args.run/'generations'/(key+'.json')
            if not path.exists():continue
            original=path.read_bytes();result=json.loads(original)
            if result.get('ok'):continue
            base_plan=None
            for log in result['logs']:
                try:
                    choice=log['response']['choices'][0]
                    if choice.get('finish_reason')!='stop':continue
                    plan=normalized_json(choice['message'].get('content') or '')
                    if log.get('phase')=='structured_source_repair':
                        plan={**plan,'expression':plan['result_step']}
                    elif 'expression' not in plan and 'result_step' in plan:
                        plan={**plan,'expression':plan['result_step']}
                    if log.get('phase')=='local_output_expression_repair':
                        if base_plan is None:continue
                        plan={**base_plan,'steps':log['input_steps'],'expression':plan['expression']}
                    else:base_plan=plan
                    plan=normalize_display_conversion(vectorize_assign(plan),r['question'],frames)
                    query,answer=compile_plan(plan,frames,r['question'])
                except Exception:continue
                preserved=previous/(key+'_'+hashlib.sha256(original).hexdigest()[:12]+'.json')
                if not preserved.exists():preserved.write_bytes(original)
                result.update(ok=True,plan=plan,query=query,answer=answer,
                    recovered_validation={'attempt':log['attempt'],'original_record_sha256':hashlib.sha256(original).hexdigest(),
                    'normalizer_sha256':hashlib.sha256((ROOT/'src/road2ai_vifinqa/plan_normalization.py').read_bytes()).hexdigest(),
                    'expression_plan_sha256':hashlib.sha256((ROOT/'src/road2ai_vifinqa/expression_plan.py').read_bytes()).hexdigest()})
                path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
                recovered.append(r['id']);break
    print(json.dumps({'recovered':len(recovered),'ids_for_audit_only':recovered}))

if __name__=='__main__':main()
