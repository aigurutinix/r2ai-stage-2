"""Apply the generic direct-source fast path, retaining all previous attempts."""
import argparse,hashlib,io,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.direct_evidence_query import direct_plan,temporal_pair_plan,direct_percentage_plan,selector_ratio_plan
from road2ai_vifinqa.compliant_query import make_request,compile_plan

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    args=ap.parse_args();count=0
    previous=args.run/'prior_failures';previous.mkdir(exist_ok=True)
    with zipfile.ZipFile(args.baseline) as z:
        for row in json.loads(z.read('submission.json')):
            frames={e['variable']:pd.read_csv(io.BytesIO(z.read(e['csv_path']))) for e in row['evidence']}
            plan=(direct_plan(row['question'],frames) or temporal_pair_plan(row['question'],frames)
                  or direct_percentage_plan(row['question'],frames) or selector_ratio_plan(row['question'],frames))
            if plan is None:continue
            prompt,key=make_request(row['question'],frames);path=args.run/'generations'/(key+'.json')
            if path.exists():
                original=path.read_bytes();prior=json.loads(original)
                if prior.get('ok'):continue
                preserved=previous/(key+'_'+hashlib.sha256(original).hexdigest()[:12]+'.json')
                if not preserved.exists():preserved.write_bytes(original)
            query,answer=compile_plan(plan,frames,row['question'])
            result={'ok':True,'key':key,'prompt':prompt,'logs':[],'plan':plan,'query':query,'answer':answer,
                'method':'generic_source_semantic_rule',
                'method_sha256':hashlib.sha256((ROOT/'src/road2ai_vifinqa/direct_evidence_query.py').read_bytes()).hexdigest()}
            path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8');count+=1
    print(json.dumps({'direct_source_queries_added':count}))

if __name__=='__main__':main()
