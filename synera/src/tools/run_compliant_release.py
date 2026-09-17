"""Unattended local build/review gate. Never submits externally or guesses missing answers."""
import argparse,ctypes,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,default=ROOT/'submission_vn79.zip')
    ap.add_argument('--output',type=Path,default=ROOT/'submission_vn80.zip')
    args=ap.parse_args();args.run=args.run.resolve();args.run.mkdir(parents=True,exist_ok=True)
    state={'started':time.time(),'status':'running','stages':[]}
    def save():
        (args.run/'driver_status.json').write_text(json.dumps(state,indent=2)+'\n',encoding='utf-8')
    def command(script,*options):
        cmd=[sys.executable,'-X','utf8',str(ROOT/'tools'/script),*map(str,options)]
        state['current_command']=cmd;save();print('STAGE '+script,flush=True)
        p=subprocess.run(cmd,cwd=ROOT)
        state['stages'].append({'command':cmd,'returncode':p.returncode,'ended':time.time()});save()
        if p.returncode:raise RuntimeError(f'{script}: exit {p.returncode}')
    def counts():
        rs=[json.loads(p.read_text(encoding='utf-8')) for p in (args.run/'generations').glob('*.json')]
        return {'cached':len(rs),'passed':sum(bool(r.get('ok')) for r in rs),'failed':sum(not r.get('ok') for r in rs)}
    keep_awake=False
    try:
        if sys.platform=='win32':ctypes.windll.kernel32.SetThreadExecutionState(0x80000001);keep_awake=True
        command('build_compliant_submission.py','--baseline',args.baseline,'--run',args.run,'--workers',4,'--thinking','--guided','--skip-failed','--attempts',1)
        for round_no in range(2):
            command('prefill_compliant_direct.py','--baseline',args.baseline,'--run',args.run)
            command('recover_compliant_generations.py','--baseline',args.baseline,'--run',args.run)
            before=counts();state['counts']=before;save()
            if before['passed']==1012:break
            command('repair_compliant_batch.py','--baseline',args.baseline,'--run',args.run,'--workers',4,'--attempts',2)
        command('recover_compliant_generations.py','--baseline',args.baseline,'--run',args.run)
        state['counts']=counts();save()
        if state['counts']['passed']!=1012:raise RuntimeError('Some queries remain unresolved. No ZIP released; inspect logs, do not replace with old answers.')
        command('build_compliant_submission.py','--baseline',args.baseline,'--run',args.run,'--thinking','--guided','--package-only','--output',args.output)
        audit=args.run/'independent_audit.json'
        command('audit_compliant_submission.py','--baseline',args.baseline,'--candidate',args.output,'--run',args.run,'--report',audit)
        command('package_compliant_support.py','--baseline',args.baseline,'--run',args.run,'--audit',audit,'--output',args.output.with_name(args.output.stem+'_technical.zip'))
        state['status']='audit_passed_pending_human_review';state['archive']=str(args.output);save()
    except Exception as exc:
        state.update(status='needs_attention',error=str(exc),counts=counts());save();raise
    finally:
        if keep_awake:ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)

if __name__=='__main__':main()
