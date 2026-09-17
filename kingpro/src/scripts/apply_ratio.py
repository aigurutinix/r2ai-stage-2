"""Post-process TẤT ĐỊNH: áp ratio_answer (biên gộp/ròng + ROE/ROA bình quân) lên submission có sẵn.
Cho câu tỷ số: re-retrieve bảng trong ĐÚNG báo cáo (relevant_docs) -> ratio_answer -> override
answer/pandas/evidence. FREE (không LLM). Dùng thêm ROE/ROA vào bản voting (chỉ có margins).

  python scripts/apply_ratio.py sub_fpt_voting sub_fpt_final
"""
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import tables_in_reports
from kingpro.answering.pandas_answer import ratio_answer, _RATIO_MASO, _RATIO_AVG

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}
_RE = [v[0] for v in list(_RATIO_MASO.values()) + list(_RATIO_AVG.values())]


def csv_full(tr):
    r = CAT.get(tr)
    return "build/tables/" + r["csv_path"] if r else None


def safe_name(tr):
    return re.sub(r"[^0-9A-Za-z_]+", "_", tr) + ".csv"


def is_ratio(q):
    return any(rx.search(q) for rx in _RE)


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    data_dir = dst / "data"
    data_dir.mkdir(exist_ok=True)
    copied = {c.name for c in data_dir.glob("*.csv")}
    rows = json.loads((src / "submission.json").read_text(encoding="utf-8"))

    n_ratio = n_over = 0
    tables_in_reports("khởi động index", [], n=1)          # pre-warm
    for r in rows:
        if not is_ratio(r["question"]):
            continue
        n_ratio += 1
        docs = r.get("relevant_docs") or []
        if not docs:
            continue
        tabs = tables_in_reports(r["question"], docs, n=10)
        ans_tables = [{"table_ref": t["table_ref"], "csv_path": csv_full(t["table_ref"])}
                      for t in tabs if csv_full(t["table_ref"])]
        try:
            rr = ratio_answer(r["question"], ans_tables)
        except Exception:
            rr = None
        if not rr:
            continue
        ev = []
        for e in rr["evidence"]:
            nm = safe_name(e["table_ref"])
            if nm not in copied:
                try:
                    shutil.copyfile(e["csv_path"], data_dir / nm)
                    copied.add(nm)
                except Exception:
                    continue
            ev.append({"variable": e["variable"], "csv_path": f"data/{nm}"})
        r["answer"] = float(rr["answer"])
        r["pandas_query"] = rr["pandas_query"]
        r["evidence"] = ev
        n_over += 1

    (dst / "submission.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    zp = dst.parent / f"{dst.name}.zip"
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(dst / "submission.json", "submission.json")
        for c in data_dir.glob("*.csv"):
            z.write(c, f"data/{c.name}")
    print(f"câu tỷ số: {n_ratio} | override tất định: {n_over} -> {zp}")


if __name__ == "__main__":
    main()
