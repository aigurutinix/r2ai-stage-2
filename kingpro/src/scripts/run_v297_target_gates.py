from __future__ import annotations
import hashlib,json,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];C=ROOT/'sub_v297_scope2';OUT=ROOT/'build/v297_target_gates.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest().upper()
def run(a):
 r=subprocess.run([sys.executable,*a],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
 if r.returncode:raise RuntimeError(r.stdout+r.stderr)
 return r.stdout
def js(a):return json.loads(run(a))
def main():
 t=run(['-m','pytest','scripts/test_v297_scope2.py','-q']);assert re.search(r'4 passed',t)
 runtime={}
 for n,f in [('string',[]),('typed',['--typed-dfs']),('official',['--official'])]:
  x=js(['scripts/grader_check.py',str(C),'--ids','224,966',*f]);assert x['match_stored_answer']==2 and not x['errors'];runtime[n]={k:x[k] for k in ['entries','ran_no_exception','numeric_result','match_stored_answer','mode','errors','mismatch_ids']}
 target=js(['scripts/verify_source_audit.py',str(C),'--ids','224,966']);full=js(['scripts/verify_source_audit.py',str(C)]);comp=js(['scripts/check_submission_compliance.py',str(C),'--allow-relevant-table-order','--provenance-allowlist','config/v297_provenance_allowlist.json'])
 assert target['source_cells_checked']==4 and target['issues']==0 and full['issues']==0
 assert comp['status']=='PASS' and comp['raw_issue_count']==comp['accepted_issue_count']==4 and comp['issues']==comp['unused_allowlist_count']==0
 base={int(x['id']):x for x in json.loads((ROOT/'sub_v290_scope2/submission.json').read_text(encoding='utf-8'))};cand={int(x['id']):x for x in json.loads((C/'submission.json').read_text(encoding='utf-8'))};diff=[]
 for i in base:
  k=sorted(x for x in set(base[i])|set(cand[i]) if base[i].get(x)!=cand[i].get(x))
  if k:diff.append({'id':i,'fields':k})
 expected=[{'id':224,'fields':['answer','evidence','pandas_query','relevant_docs','relevant_tables']},{'id':966,'fields':['relevant_docs','relevant_tables']}];assert diff==expected
 report={'schema_version':'v297-target-gates/v1','candidate':C.name,'baseline':'sub_v290_scope2','submission_sha256':sha(C/'submission.json'),'semantic_diff':diff,'tests':{'status':'PASS','passed':4},'target_runtime':runtime,'target_source_audit':{'status':'PASS','cells':4,'issues':0},'full_source_audit':{'status':'PASS','cells':full['source_cells_checked'],'issues':0},'compliance':{'status':'PASS','raw_issue_count':4,'accepted_issue_count':4,'issues':0,'unused_allowlist_count':0,'allowlist_sha256':sha(ROOT/'config/v297_provenance_allowlist.json')},'packaged':False,'submitted':False,'overall':'PASS'}
 OUT.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps({'output':str(OUT),'sha256':sha(OUT),'overall':'PASS'},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
