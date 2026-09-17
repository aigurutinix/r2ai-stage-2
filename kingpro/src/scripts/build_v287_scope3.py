"""Build rollbackable v287 with the physically consolidated q764 source."""
from __future__ import annotations
import csv, hashlib, json, shutil
from pathlib import Path
from tempfile import mkdtemp

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sub_v283_hut_scope'
OUTPUT=ROOT/'sub_v287_scope3'
HEADER=['ticker','year','metric_key','raw','typed_factor','scale','source_table','source_csv','row_idx','col_idx']

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def write_csv(p:Path, rows:list[list[str]])->None:
    with p.open('w',encoding='utf-8',newline='') as h:
        w=csv.writer(h,lineterminator='\n');w.writerow(HEADER);w.writerows(rows)
def build()->dict:
    if OUTPUT.exists():raise FileExistsError(f'refusing to overwrite {OUTPUT}')
    temp=Path(mkdtemp(prefix='v287_scope3_',dir=str(ROOT)))
    try:
        staged=temp/OUTPUT.name;shutil.copytree(SOURCE,staged)
        original=json.loads((staged/'submission.json').read_text(encoding='utf-8-sig'))
        by_id={int(x['id']):x for x in original}
        if by_id[764]['answer']!=2.99:raise AssertionError('unexpected q764 baseline')
        by_id[764]['answer']=2.94
        by_id[764]['relevant_tables']=['GVR_financial_statements_2015_consolidated|290','DPM_financial_statements_2015_consolidated|1909']
        (staged/'submission.json').write_text(json.dumps(list(by_id.values()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        physical=ROOT/'build/tables/DPM_financial_statements_2015_consolidated/table_63_line1909.csv'
        if not physical.is_file():raise FileNotFoundError(physical)
        shutil.copy2(physical,staged/'data/DPM_financial_statements_2015_consolidated_1909.csv')
        write_csv(staged/'data/q764_source_cells.csv',[
            ['GVR','2015','cdkt:418','6.437.295.628.830','1.0','1.0','GVR_financial_statements_2015_consolidated|290','table_7_line290.csv','11','3'],
            ['DPM','2015','cdkt:418','3.498.666.363.829','1.0','1.0','DPM_financial_statements_2015_consolidated|1909','DPM_financial_statements_2015_consolidated_1909.csv','28','3']])
        ap=staged/'source_audit.json';audit=json.loads(ap.read_text(encoding='utf-8-sig'));a={int(x['id']):x for x in audit}
        q=a[764];q['answer']=2.94
        q['sources'][1].update({'table_ref':'DPM_financial_statements_2015_consolidated|1909','csv':'DPM_financial_statements_2015_consolidated_1909.csv','row':28,'column':3,'raw':'3.498.666.363.829','source_row_labels':['6. Quỹ đầu tư phát triển']})
        ap.write_text(json.dumps(sorted(a.values(),key=lambda x:int(x['id'])),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        (staged/'v287_scope3_audit.json').write_text(json.dumps({'candidate':OUTPUT.name,'baseline':SOURCE.name,'answer_changes':[{'id':764,'from':2.99,'to':2.94}],'relevant_table_changes':[{'id':764,'from':'DPM_financial_statements_2015_consolidated|272','to':'DPM_financial_statements_2015_consolidated|1909'}],'query_preserved':True,'automatic_promotion':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        shutil.move(str(staged),str(OUTPUT))
        return {'candidate':OUTPUT.name,'baseline':SOURCE.name,'answer_changes':[{'id':764,'from':2.99,'to':2.94}],'submission_sha256':sha(OUTPUT/'submission.json')}
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise
if __name__=='__main__':print(json.dumps(build(),ensure_ascii=False,indent=2))
