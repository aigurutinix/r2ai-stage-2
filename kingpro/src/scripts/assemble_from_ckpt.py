"""Dựng submission.json + data/ từ _ckpt.jsonl (KHÔNG gọi LLM) — cứu khi build crash giữa chừng.
Chạy: python scripts/assemble_from_ckpt.py sub_program
"""
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "src")
try:
    from kingpro.evaluation.metrics import coerce_number
except Exception:
    def coerce_number(x):
        try:
            return float(x)
        except Exception:
            return None

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "sub_program")
ckpt = OUT / "_ckpt.jsonl"
data = OUT / "data"
data.mkdir(parents=True, exist_ok=True)


def safe(tref):
    return re.sub(r"[^0-9A-Za-z_]+", "_", tref) + ".csv"


copied, rows, seen = {}, [], set()
for l in open(ckpt, encoding="utf-8"):
    try:
        c = json.loads(l)
    except Exception:
        continue
    q, res = c["q"], c["res"]
    if q["id"] in seen:
        continue
    seen.add(q["id"])
    evidence = []
    for ev in res.get("evidence", []):
        tref = ev.get("table_ref")
        src = ev.get("csv_path")
        if not tref or not src:
            continue
        if tref not in copied:
            dst = data / safe(tref)
            try:
                shutil.copyfile(src, dst)
                copied[tref] = f"data/{dst.name}"
            except Exception:
                continue
        evidence.append({"variable": ev["variable"], "csv_path": copied[tref]})
    av = coerce_number(res.get("answer")) if res.get("ok") else None
    rows.append({
        "id": q["id"],
        "question": q["question"],
        "answer": float(av) if av is not None else 0.0,
        "relevant_docs": c.get("rel_docs", []),
        "relevant_tables": c.get("rel_tables", []),
        "evidence": evidence,
        "pandas_query": res.get("pandas_query") or "",
    })

json.dump(rows, open(OUT / "submission.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(f"assembled {len(rows)} rows | có evidence: {sum(1 for r in rows if r['evidence'])} | CSV copied: {len(copied)}")
