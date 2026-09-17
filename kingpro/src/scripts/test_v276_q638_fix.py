"""Tests for v276's narrowly scoped q638/q613/q627 changes."""
from __future__ import annotations
import csv, hashlib, json, sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_v276_q638_fix as b

class V276Tests(unittest.TestCase):
    def test_refuses_overwrite(self):
        with self.assertRaises(FileExistsError): b.build()

    def test_submission_semantic_diff_is_exact(self):
        base={int(x['id']):x for x in json.loads((b.SOURCE/'submission.json').read_text(encoding='utf-8-sig'))}
        cand={int(x['id']):x for x in json.loads((b.OUTPUT/'submission.json').read_text(encoding='utf-8-sig'))}
        diffs=[]
        for qid in base:
            keys=sorted(k for k in set(base[qid])|set(cand[qid]) if base[qid].get(k)!=cand[qid].get(k))
            if keys: diffs.append((qid,keys))
        self.assertEqual(diffs, [(638,['answer','relevant_tables'])])
        self.assertEqual(cand[638]['answer'],5.74)
        self.assertEqual(cand[638]['relevant_tables'][1],'GEX_financial_statements_2022_separate|1417')

    def test_manifests_exact(self):
        for qid, expected in {
            613:[('IJC','2022','(115.274.660.827)','8','3'),('IJC','2018','(301.534.111.657)','8','3')],
            627:[('PVT','2018','41.676.677.160','3','1'),('PVT','2016','13.838.474.530','3','1')],
        }.items():
            with (b.OUTPUT/'data'/f'q{qid}_source_cells.csv').open(encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
            self.assertEqual([(r['ticker'],r['year'],r['raw'],r['row_idx'],r['col_idx']) for r in rows],expected)
        with (b.OUTPUT/'data/q638_source_cells.csv').open(encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
        self.assertEqual(rows[1]['source_table'],'GEX_financial_statements_2022_separate|1417')
        self.assertEqual(rows[1]['raw'],'25.779.332.206')

    def test_source_audit_entries(self):
        a={int(x['id']):x for x in json.loads((b.OUTPUT/'source_audit.json').read_text(encoding='utf-8-sig'))}
        self.assertEqual(a[638]['answer'],5.74)
        self.assertEqual(a[638]['sources'][1]['table_ref'],'GEX_financial_statements_2022_separate|1417')
        self.assertEqual(a[613]['sources'][0]['row'],8)
        self.assertEqual(a[627]['sources'][1]['column'],1)

    def test_non_target_files_identical(self):
        allowed={Path('submission.json'),Path('source_audit.json'),Path('data/q638_source_cells.csv'),Path('data/q613_source_cells.csv'),Path('data/q627_source_cells.csv'),Path('v276_q638_fix_audit.json')}
        base={p.relative_to(b.SOURCE) for p in b.SOURCE.rglob('*') if p.is_file() and p.relative_to(b.SOURCE) not in allowed}
        cand={p.relative_to(b.OUTPUT) for p in b.OUTPUT.rglob('*') if p.is_file() and p.relative_to(b.OUTPUT) not in allowed}
        self.assertEqual(base,cand)
        for p in base: self.assertEqual(hashlib.sha256((b.SOURCE/p).read_bytes()).digest(),hashlib.sha256((b.OUTPUT/p).read_bytes()).digest(),str(p))

if __name__=='__main__': unittest.main()
