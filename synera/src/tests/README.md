# Test strategy

Chạy:

```powershell
$env:PYTHONPATH = (Resolve-Path "src").Path
python -m unittest discover -s tests -v
python -m compileall -q src tools tests
```

Test suite bao phủ:

- HTML table expansion và table semantics;
- source unit, period, opening date và comparative-year guards;
- retrieval/table/row/cell ranking;
- Easy reranker schema, reproducibility và shortlist behavior;
- panel year gaps, note injection và deterministic formulas;
- expression compilation, sign/ratio contracts và mutation sensitivity;
- submission table references, evidence CSV và regression registry.

Một test pass chỉ chứng minh invariant cục bộ. Trước release còn phải replay
đúng solver family và chạy `tools/release_audit.py` cho đủ 1.012 câu. Kết quả
leaderboard aggregate không được chuyển thành unit-test gold theo từng câu.
