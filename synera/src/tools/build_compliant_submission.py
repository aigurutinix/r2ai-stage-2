"""Generate a complete submission with immutable evidence and fresh local LLM queries.
The orchestrator knows IDs only for serialization; the generator never receives them.
No legacy solver, manual override or previous query/answer is consulted for generation.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import pandas as pd
from road2ai_vifinqa.compliant_query import generate,make_request,compile_plan,SYSTEM,VERSION

def sha(b):return hashlib.sha256(b).hexdigest()
def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--limit',type=int)
    ap.add_argument('--workers',type=int,default=2)
    ap.add_argument('--attempts',type=int,default=3)
    ap.add_argument('--base-url',default='http://127.0.0.1:8095')
    ap.add_argument('--thinking',action='store_true')
    ap.add_argument('--package-only',action='store_true')
    ap.add_argument('--seed-run',type=Path,help='Reuse only validated machine generations with identical prompts and baseline')
    ap.add_argument('--skip-failed',action='store_true',help='Finish unseen questions before a separate error-repair pass')
    ap.add_argument('--guided',action='store_true')
    ap.add_argument('--inherit-failures',action='store_true')
    args=ap.parse_args()
    guidance=''
    if args.guided:
        from repair_compliant_batch import GUIDANCE
        guidance=GUIDANCE
    binary=args.baseline.read_bytes()
    with zipfile.ZipFile(io.BytesIO(binary)) as z:
        if z.testzip():raise ValueError('Corrupt baseline')
        members={n:z.read(n) for n in z.namelist()}
    records=json.loads(members['submission.json'])
    args.run.mkdir(parents=True,exist_ok=True)
    cache=args.run/'generations';cache.mkdir(exist_ok=True)
    lock={'baseline':str(args.baseline.resolve()),'sha256':sha(binary),'version':VERSION,
          'system_sha256':sha(SYSTEM.encode()),'generator_sha256':sha((ROOT/'src/road2ai_vifinqa/compliant_query.py').read_bytes()),
          'thinking':args.thinking,'evidence_sha256':{n:sha(b) for n,b in members.items() if n.endswith('.csv')}}
    if guidance:lock['guidance_sha256']=sha(guidance.encode())
    lockpath=args.run/'lock.json'
    if lockpath.exists():
        if json.loads(lockpath.read_text(encoding='utf-8'))!=lock:raise ValueError('Run lock changed; create a fresh run')
    else:write(lockpath,lock)
    snapshot=args.run/'generator_snapshot.py'
    if not snapshot.exists():snapshot.write_bytes((ROOT/'src/road2ai_vifinqa/compliant_query.py').read_bytes())
    def prepare(r):
        frames={e['variable']:pd.read_csv(io.BytesIO(members[e['csv_path']])) for e in r['evidence']}
        _,key=make_request(r['question'],frames)
        return frames,key
    if args.seed_run:
        parent_lock=json.loads((args.seed_run/'lock.json').read_text(encoding='utf-8'))
        # A validated source-only plan can be replayed under a different inference
        # setting. It is always recompiled against the current frozen evidence;
        # this is cache provenance, not reuse of an old answer/query.
        assert all(parent_lock[k]==lock[k] for k in ['sha256','version','system_sha256','evidence_sha256'])
        reused=[]
        for r in records:
            frames,key=prepare(r);origin=args.seed_run/'generations'/(key+'.json');target=cache/(key+'.json')
            if not origin.exists() or target.exists():continue
            result=json.loads(origin.read_text(encoding='utf-8'))
            try:
                if result['prompt']!=make_request(r['question'],frames)[0]:continue
                if not result.get('ok'):
                    if args.inherit_failures:target.write_bytes(origin.read_bytes());reused.append(key)
                    continue
                query,answer=compile_plan(result['plan'],frames,r['question'])
                if query!=result['query'] or answer!=result['answer']:
                    result={**result,'query':query,'answer':answer,'recompiled_from_sha256':sha(origin.read_bytes())}
            except Exception:continue
            write(target,result);reused.append(key)
        write(args.run/'cache_origin.json',{'parent_run':str(args.seed_run.resolve()),'parent_lock':parent_lock,
            'inference_setting_changed':parent_lock.get('thinking')!=lock.get('thinking'),'reused':reused})
    def job(r):
        frames,key=prepare(r)
        path=cache/(key+'.json')
        if path.exists():
            cached=json.loads(path.read_text(encoding='utf-8'))
            if cached.get('ok') or args.skip_failed:return r['id'],cached,True
            history=args.run/'prior_failures';history.mkdir(exist_ok=True)
            old_bytes=path.read_bytes();preserved=history/(key+'_'+sha(old_bytes)[:12]+'.json')
            if not preserved.exists():preserved.write_bytes(old_bytes)
        result=generate(r['question'],frames,args.base_url,thinking=args.thinking,attempts=args.attempts,guidance=guidance)
        write(path,result)
        return r['id'],result,False
    ordered=sorted(records,key=lambda r:sha(r['question'].encode()))
    if args.limit:ordered=ordered[:args.limit]
    if not args.package_only:
        started=time.monotonic();done=0;ok=0
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for fut in as_completed([pool.submit(job,r) for r in ordered]):
                qid,result,cached=fut.result();done+=1;ok+=bool(result['ok'])
                print(json.dumps({'done':done,'total':len(ordered),'id':qid,'ok':result['ok'],'cached':cached,'elapsed':round(time.monotonic()-started,1)},ensure_ascii=False),flush=True)
    if args.output:
        if args.output.exists():raise FileExistsError(args.output)
        rebuilt=[];summary=[];failures=[]
        for r in records:
            frames,key=prepare(r);path=cache/(key+'.json')
            if not path.exists():failures.append({'id':r['id'],'reason':'not generated'});continue
            result=json.loads(path.read_text(encoding='utf-8'))
            if not result.get('ok'):failures.append({'id':r['id'],'reason':'generation failed'});continue
            query,answer=compile_plan(result['plan'],frames,r['question'])
            if query!=result['query'] or answer!=result['answer']:raise ValueError('Replay mismatch')
            rebuilt.append({**r,'pandas_query':query,'answer':answer})
            summary.append({'id':r['id'],'content_key':key,'query_chars':len(query),'hidden_columns_inaccessible':True})
        write(args.run/'packaging_audit.json',{'rows':len(rebuilt),'failures':failures,'items':summary})
        if failures:raise RuntimeError(f'Incomplete generation: {len(failures)}; no ZIP written')
        output=dict(members)
        output['submission.json']=(json.dumps(rebuilt,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
            for n,b in output.items():z.writestr(n,b)
        with zipfile.ZipFile(args.output) as z:
            actual={n:sha(z.read(n)) for n in z.namelist() if n.endswith('.csv')}
            assert actual==lock['evidence_sha256']
        write(args.run/'release.json',{'archive':str(args.output.resolve()),'sha256':sha(args.output.read_bytes()),'rows':len(rebuilt),'all_evidence_bytes_unchanged':True,'official_score':None})
        print('READY '+str(args.output),flush=True)

if __name__=='__main__':main()
