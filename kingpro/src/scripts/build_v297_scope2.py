"""Build rollbackable V297 with source-proven q224/q966 scope repairs."""
from __future__ import annotations
import hashlib,json,shutil,sys
from pathlib import Path
from tempfile import mkdtemp

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'sub_v290_scope2'
Q224=ROOT/'sub_top123_candidate_v225_q98_physical_parent_rollback_batch9'
OUTPUT=ROOT/'sub_v297_scope2'

def sha(p:Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest().upper()
def keyed(p:Path)->dict[int,dict]:return {int(x['id']):x for x in json.loads(p.read_text(encoding='utf-8-sig'))}
def build()->dict:
    if OUTPUT.exists():raise FileExistsError(f'refusing to overwrite {OUTPUT}')
    temp=Path(mkdtemp(prefix='v297_scope2_',dir=str(ROOT)))
    try:
        staged=temp/OUTPUT.name;shutil.copytree(SOURCE,staged)
        base=keyed(SOURCE/'submission.json');rows=keyed(staged/'submission.json');trusted=keyed(Q224/'submission.json')
        rows[224]=json.loads(json.dumps(trusted[224],ensure_ascii=False))
        if rows[224]['relevant_docs']!=['HUT_financial_statements_2024_consolidated','HUT_financial_statements_2024_separate']:raise AssertionError('q224 union docs drift')
        if rows[224]['relevant_tables']!=['HUT_financial_statements_2024_consolidated|1243','HUT_financial_statements_2024_separate|1624']:raise AssertionError('q224 union tables drift')
        rows[966]['answer']=2.0
        rows[966]['relevant_docs']=['HDG_financial_statements_2025_consolidated','GEG_financial_statements_2025_consolidated','DNH_financial_statements_2025_consolidated']
        rows[966]['relevant_tables']=['HDG_financial_statements_2025_consolidated|327','GEG_financial_statements_2025_consolidated|493','DNH_financial_statements_2025_consolidated|343']
        if rows[966]['pandas_query']!=base[966]['pandas_query'] or rows[966]['evidence']!=base[966]['evidence']:raise AssertionError('q966 formula/binding drift')
        (staged/'submission.json').write_text(json.dumps(list(rows.values()),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        shutil.copy2(Q224/'data/q224_source_cells.csv',staged/'data/q224_source_cells.csv')
        (staged/'data/q966_source_cells.csv').write_text(
            'ticker,year,metric_key,raw,typed_factor,scale,source_table,source_csv,row_idx,col_idx\n'
            'HDG,2025,lctt:20,1.243.479.705.955,1.0,1.0,HDG_financial_statements_2025_consolidated|327,table_9_line327.csv,17,3\n'
            'GEG,2025,lctt:20,932.670.196.340,1.0,1.0,GEG_financial_statements_2025_consolidated|493,table_11_line493.csv,16,3\n'
            'DNH,2025,lctt:20,1.432.290.911.421,1.0,1.0,DNH_financial_statements_2025_consolidated|343,table_6_line343.csv,17,3\n',encoding='utf-8')
        audits=keyed(staged/'source_audit.json')
        audits[224]={'id':224,'old_answer':181.54,'answer':1200.5,'note':'HUT consolidated third-party trade payables; measured union retained, physical HOP NHAT source is separate-container |1624','sources':[{'table_ref':'HUT_financial_statements_2024_separate|1624','csv':'table_37_line1624.csv','row':2,'column':1,'metric':'note:trade_payables_third_parties','label':'Phải trả người bán là bên thứ ba','source_row_labels':['Phải trả người bán là bên thứ ba'],'scale':1.0,'typed_factor':1.0,'raw':'1.200.498.290.074'}]}
        q=audits[966];q['answer']=2.0;q['sources'][1].update({'table_ref':'GEG_financial_statements_2025_consolidated|493','csv':'table_11_line493.csv','row':16,'column':3,'raw':'932.670.196.340'})
        (staged/'source_audit.json').write_text(json.dumps(sorted(audits.values(),key=lambda x:int(x['id'])),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        report={'candidate':OUTPUT.name,'baseline':SOURCE.name,'answer_changes':[{'id':224,'from':181.54,'to':1200.5}],'lineage_only_ids':[966],'q224_measured_union':{'docs':rows[224]['relevant_docs'],'tables':rows[224]['relevant_tables'],'physical_answer_table':'HUT_financial_statements_2024_separate|1624'},'q966':{'answer':2.0,'physical_geg_table':'GEG_financial_statements_2025_consolidated|493','physical_geg_raw':'932.670.196.340','threshold':'strictly > 1e12','qualifying':['HDG','DNH']},'automatic_promotion':False,'packaged':False,'submitted':False}
        (staged/'v297_scope2_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        shutil.move(str(staged),str(OUTPUT));temp.rmdir();return {**report,'submission_sha256':sha(OUTPUT/'submission.json')}
    except Exception:
        shutil.rmtree(temp,ignore_errors=True);raise
if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    print(json.dumps(build(),ensure_ascii=False,indent=2))
