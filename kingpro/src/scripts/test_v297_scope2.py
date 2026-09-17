from __future__ import annotations
import hashlib,json,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent));import build_v297_scope2 as b
def keyed(p):return {int(x['id']):x for x in json.loads(p.read_text(encoding='utf-8-sig'))}
class Tests(unittest.TestCase):
 def test_guard(self):
  with self.assertRaises(FileExistsError):b.build()
 def test_exact_diff(self):
  a=keyed(b.SOURCE/'submission.json');c=keyed(b.OUTPUT/'submission.json');d=[]
  for i in a:
   k=sorted(x for x in set(a[i])|set(c[i]) if a[i].get(x)!=c[i].get(x))
   if k:d.append((i,k))
  self.assertEqual(d,[(224,['answer','evidence','pandas_query','relevant_docs','relevant_tables']),(966,['relevant_docs','relevant_tables'])])
 def test_q224_exact_v225_and_q966_formula(self):
  c=keyed(b.OUTPUT/'submission.json');t=keyed(b.Q224/'submission.json');a=keyed(b.SOURCE/'submission.json')
  self.assertEqual(c[224],t[224]);self.assertEqual(c[966]['pandas_query'],a[966]['pandas_query']);self.assertEqual(c[966]['answer'],2.0)
 def test_non_target_bytes(self):
  allowed={Path('submission.json'),Path('source_audit.json'),Path('data/q224_source_cells.csv'),Path('data/q966_source_cells.csv'),Path('v297_scope2_audit.json')}
  aa={p.relative_to(b.SOURCE) for p in b.SOURCE.rglob('*') if p.is_file() and p.relative_to(b.SOURCE) not in allowed};cc={p.relative_to(b.OUTPUT) for p in b.OUTPUT.rglob('*') if p.is_file() and p.relative_to(b.OUTPUT) not in allowed};self.assertEqual(aa,cc)
  for p in aa:self.assertEqual(hashlib.sha256((b.SOURCE/p).read_bytes()).digest(),hashlib.sha256((b.OUTPUT/p).read_bytes()).digest(),str(p))
if __name__=='__main__':unittest.main()
