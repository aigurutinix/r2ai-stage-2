"""ĐÁNH GIÁ bộ SFT: (1) phân bố vs đề thật, (2) đa dạng, (3) re-validate qua grader,
(4) KIỂM ĐÚNG-SAI ĐỘC LẬP bằng đẳng thức kế toán (Tổng TS=TSNH+TSDH...) — vì grader chỉ tự-nhất-quán.
Chạy: python scripts/eval_sft.py build/sft_eval.jsonl
"""
import json, sys, random, collections
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import pandas as pd
from gen_sft_data import _cell, _maso_col_idx, mult_dong, STD_CODES, csv_full, build_index, _extract_dong
from grader_check import run_one
from kingpro.answering.pandas_answer import _with_preamble
from kingpro.answering.ma_so_tt200 import _norm

CAT = {x["table_ref"]: "build/tables/" + x["csv_path"]
       for x in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def main():
    rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
    tot = len(rows)
    c = collections.Counter(r["type"] for r in rows)
    print(f"=== BỘ DATA: {tot} mẫu ===")
    for k, v in c.most_common():
        print(f"  {k:14} {v:5} ({v*100//tot}%)")

    # 2) đa dạng câu hỏi
    qs = [next((l for l in r["user"].split("\n") if l.startswith("Câu hỏi")), "")[:45] for r in rows]
    print(f"\nĐA DẠNG: {len(set(qs))}/{tot} câu (đầu-45-ký-tự) khác nhau")

    # 3) re-validate ĐỘC LẬP qua grader
    random.seed(0); ok = err = 0
    for r in random.sample(rows, min(25, tot)):
        trs = r.get("table_refs") or r["meta"]["table_refs"]
        try:
            paths = {f"df{i+1}": CAT[tr] for i, tr in enumerate(trs)}
            v = float(run_one(_with_preamble(r["assistant"], len(trs)), paths))
            if abs(v - float(r["answer"])) <= 0.01: ok += 1
        except Exception:
            err += 1
    print(f"RE-VALIDATE grader: {ok}/25 khớp answer ({err} lỗi)")

    # 4) KIỂM ĐÚNG-SAI ĐỘC LẬP: đẳng thức kế toán trên bảng CDKT (extraction có đúng không)
    print("\n=== KIỂM ĐỘC LẬP (đẳng thức kế toán, đo extraction) ===")
    idx = build_index([json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8")], 3000)
    cdkt = [e for e in idx if e["has_maso"] and e["rtype"] == "CDKT"]
    checks = {"270=100+200": ("270", "100", "200"), "440=300+400": ("440", "300", "400")}
    for name, (whole, a1, a2) in checks.items():
        hold = bad = 0
        for e in cdkt[:200]:
            if not ({whole, a1, a2} <= e["codes"]): continue
            lw, la, lb = _cell(e, whole), _cell(e, a1), _cell(e, a2)
            if not (lw and la and lb): continue
            try:
                df = e["df"]
                def val(l, code):
                    s = str(df[df[l[0]].astype(str).str.strip() == code][l[1]].values[0])
                    s = s.replace("(", "-").replace(")", "").replace(".", "").replace(",", ".").replace("%", "").strip()
                    return float(s) if s and s not in "-" else None
                vw, va, vb = val(lw, whole), val(la, a1), val(lb, a2)
                if None in (vw, va, vb): continue
                if abs(vw - (va + vb)) <= abs(vw) * 0.01 + 1: hold += 1
                else: bad += 1
            except Exception:
                continue
        tot_c = hold + bad
        print(f"  {name}: {hold}/{tot_c} đẳng thức ĐÚNG "
              f"({'extraction TIN CẬY' if tot_c and hold >= tot_c*0.9 else 'CÓ LỖI extraction!' if tot_c else 'không đủ mẫu'})")

    # 5) mẫu mỗi loại
    print("\n=== MẪU mỗi loại ===")
    seen = set()
    for r in rows:
        if r["type"] in seen: continue
        seen.add(r["type"])
        q = next((l for l in r["user"].split("\n") if l.startswith("Câu hỏi")), "")
        print(f"  [{r['type']}] {q[:92]} -> {r['answer']}")


if __name__ == "__main__":
    main()
