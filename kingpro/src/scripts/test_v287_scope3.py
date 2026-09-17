from __future__ import annotations
import csv, hashlib, json, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import build_v287_scope3 as b

class V287Tests(unittest.TestCase):
    def test_overwrite_guard(self):
        with self.assertRaises(FileExistsError): b.build()
    def test_exact_submission_diff(self):
        a={int(x['id']):x for x in json.loads((b.SOURCE/'submission.json').read_text(encoding='utf-8-sig'))};c={int(x['id']):x for x in json.loads((b.OUTPUT/'submission.json').read_text(encoding='utf-8-sig'))}
        d=[]
        for i in a:
            keys=sorted(k for k in set(a[i])|set(c[i]) if a[i].get(k)!=c[i].get(k))
            if keys:d.append((i,keys))
        self.assertEqual(d,[(764,['answer','relevant_tables'])]);self.assertEqual(c[764]['answer'],2.94)
    def test_physical_consolidated_source(self):
        p=b.OUTPUT/'data/DPM_financial_statements_2015_consolidated_1909.csv';self.assertTrue(p.is_file())
        with p.open(encoding='utf-8-sig',newline='') as h: rows=list(csv.reader(h))
        self.assertIn('3.498.666.363.829',sum(rows,[]))
        catalog=Path('build/catalog_enriched.jsonl').read_text(encoding='utf-8');self.assertIn('DPM_financial_statements_2015_consolidated|1909',catalog)
        with (b.OUTPUT/'data/q764_source_cells.csv').open(encoding='utf-8-sig',newline='') as h:r=list(csv.DictReader(h))
        self.assertEqual(r[1]['source_table'],'DPM_financial_statements_2015_consolidated|1909');self.assertEqual(r[1]['row_idx'],'28')
    def test_non_target_bytes_identical(self):
        allowed={Path('submission.json'),Path('source_audit.json'),Path('data/q764_source_cells.csv'),Path('data/DPM_financial_statements_2015_consolidated_1909.csv'),Path('v287_scope3_audit.json')}
        a={p.relative_to(b.SOURCE) for p in b.SOURCE.rglob('*') if p.is_file() and p.relative_to(b.SOURCE) not in allowed};c={p.relative_to(b.OUTPUT) for p in b.OUTPUT.rglob('*') if p.is_file() and p.relative_to(b.OUTPUT) not in allowed};self.assertEqual(a,c)
        for p in a:self.assertEqual(hashlib.sha256((b.SOURCE/p).read_bytes()).digest(),hashlib.sha256((b.OUTPUT/p).read_bytes()).digest(),str(p))

if __name__=='__main__':unittest.main()
