"""Package the exact new pipeline and its audit trail only after a release passes."""
import argparse,hashlib,json,platform,sys,zipfile
from pathlib import Path
import pandas as pd
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

README='''# Ho so ky thuat - luot nop bo sung dau tien

## Nguyen tac

- Baseline la submission_vn79.zip da duoc doi xac nhan.
- Tat ca thanh vien archive ngoai submission.json duoc giu nguyen byte.
- Trong JSON chi thay pandas_query va answer. Khong thay evidence, ID cau hoi,
  noi dung cau hoi, relevant_docs hay relevant_tables.
- question_id chi duoc dung de ghep ket qua/log. Bo sinh cong thuc khong nhan ID.
- Khong su dung hard_solver.py, bang cong thuc theo ID, dap an cu, query cu,
  computed_answer hay answer_value de sinh dap an moi.
- Cac cot dap an cu van nam trong CSV vi BTC yeu cau giu nguyen evidence;
  query moi chi thay phep chieu cac cot nguon duoc phep. Neu co raw_value va
  source_scale, value duoc tai tao tu so lieu tho ngay trong query.

## Luong moi

1. Quy tac ngu nghia tong quat xu ly mot o nguon, ty le da in san trong bao cao,
   va so sanh hai nam co huong duoc hoi ro rang. Khong co danh sach ID/cong thuc.
2. Cac truong hop khac duoc Qwen3.5-9B local sinh ke hoach tinh toan tu cau hoi
   va cac cot nguon. Chi dung model cong khai trong manifest di kem.
3. Bo bien dich mo rong ke hoach thanh bieu thuc Pandas, chan IO, cot dap an,
   literal so lieu nguon, va ap dung doi don vi duoc phan tich tu cau hoi.
4. Sua loi bang phan hoi thuc thi cho model; khong chinh cong thuc thu cong theo ID.
   Chuan hoa JSON/lambda assign neu can chi la bien doi cu phap tuong duong.
5. Chay lai query tu CSV goc. Loai bo va lam nhieu cot an de kiem tra doc lap.
   So sanh hash evidence va audit ZIP doc lap truoc khi phat hanh.

## Tai lap

Can Python va cac phien ban pandas/numpy ghi trong environment.json. Khoi dong
llama-server voi model/path trong manifest, localhost:8095, thinking budget 1024,
4 slots, context 65536, Q8 KV. Khong can API model dong.

    python tools/build_compliant_submission.py --baseline input/submission_vn79.zip --run runs/reproduction --thinking --guided --package-only
    python tools/prefill_compliant_direct.py --baseline input/submission_vn79.zip --run runs/reproduction
    python tools/build_compliant_submission.py --baseline input/submission_vn79.zip --run runs/reproduction --thinking --guided --workers 4 --attempts 1 --skip-failed
    python tools/recover_compliant_generations.py --baseline input/submission_vn79.zip --run runs/reproduction
    python tools/repair_compliant_batch.py --baseline input/submission_vn79.zip --run runs/reproduction --workers 4

Chi dong ZIP khi tat ca ban ghi da qua kiem tra; khong fallback sang dap an/query cu.
Artifact runs/ cung cap prompt, response va query thuc te de replay. Tai sinh LLM
tren phan cung/ban llama khac co the khong trung tung byte; replay query da luu
tren evidence da khoa moi la kiem tra tinh toan chinh xac.

## Gioi han

Kiem tra query chay va trung answer KHONG chung minh answer khop dap an BTC.
Khong co diem chinh thuc moi trong ho so nay. BTC quyet dinh tinh hop le cuoi cung.
Ho so nay mo ta luong moi, khong khang dinh cac bai nop cu da tuan thu quy dinh.
'''

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    report=json.loads(args.audit.read_text(encoding='utf-8'))
    assert not report['errors'] and report['passed']==1012
    files={}
    modules=['__init__','compliant_query','direct_evidence_query','plan_normalization','expression_plan','source_views','submission']
    for name in modules:
        f=ROOT/'src/road2ai_vifinqa'/(name+'.py');files['src/road2ai_vifinqa/'+f.name]=f.read_bytes()
    for name in ['build_compliant_submission','prefill_compliant_direct','recover_compliant_generations','repair_compliant_batch','audit_compliant_submission','package_compliant_support']:
        f=ROOT/'tools'/(name+'.py');files['tools/'+f.name]=f.read_bytes()
    for f in args.run.rglob('*.json'):files['runs/release/'+f.relative_to(args.run).as_posix()]=f.read_bytes()
    files['input/submission_vn79.zip']=args.baseline.read_bytes()
    files['audit.json']=args.audit.read_bytes()
    files['model_manifest.json']=(ROOT/'artifacts/models/qwen35_9b_manifest.json').read_bytes()
    files['README.md']=README.encode('utf-8')
    files['THUYET_MINH_LUONG_MOI.md']=(ROOT/'runs/compliant_vn80/THUYET_MINH_LUONG_MOI.md').read_bytes()
    files['environment.json']=json.dumps({'python':sys.version,'platform':platform.platform(),'pandas':pd.__version__,'numpy':np.__version__},indent=2).encode()
    files['sha256_manifest.json']=json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in files.items()},indent=2).encode()
    with zipfile.ZipFile(args.output,'x',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for n,b in files.items():z.writestr(n,b)
    print(args.output)

if __name__=='__main__':main()
