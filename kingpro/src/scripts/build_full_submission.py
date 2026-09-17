"""Đóng gói submission.zip ĐẦY ĐỦ, đúng đặc tả: submission.json + data/*.csv.

Mỗi câu: retrieve -> answer (ensemble/1 model) -> answer, pandas_query, evidence(df1/df2 + csv trong data/).
Câu fail vẫn ghi entry (answer=0.0) + relevant_tables (để 3.1 vẫn chấm), KHÔNG bỏ câu (bỏ = sai định dạng).

Chạy:
  python scripts/build_full_submission.py --n 5 --k-ret 3 --k-ans 2 --out sub_full --model qwen3
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

sys.path.insert(0, "src")
from kingpro.retrieval.bm25_index import retrieve, retrieve_decomposed, extract_all_facets, tables_in_reports
from kingpro.answering.pandas_answer import answer_question, maso_rescue
from kingpro.answering.ensemble import smart_answer, self_consistent, mpr_answer
from kingpro.answering.llm_client import chat
from kingpro.evaluation.metrics import doc_of, coerce_number
from kingpro.product.deterministic_compiler import DeterministicFinancialCompiler
from kingpro.retrieval.table_reranker import rerank_tables
from kingpro.submission.archive import write_deterministic

CAT = {r["table_ref"]: r for r in (json.loads(l) for l in open("build/catalog.jsonl", encoding="utf-8"))}


def csv_full(table_ref):
    r = CAT.get(table_ref)
    return "build/tables/" + r["csv_path"] if r else None


def safe_name(table_ref):
    return re.sub(r"[^0-9A-Za-z_]+", "_", table_ref) + ".csv"


# --- SMART TABLE: chọn ĐÚNG bảng chứa mã số câu hỏi (precision cao, feed 1 bảng cho fine-tune) ---
import pandas as _pd
from kingpro.answering.ma_so_tt200 import maso_of

_CSV_CACHE: dict = {}


def _read_cached(tref):
    if tref not in _CSV_CACHE:
        p = csv_full(tref)
        try:
            _CSV_CACHE[tref] = _pd.read_csv(p, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        except Exception:
            _CSV_CACHE[tref] = None
    return _CSV_CACHE[tref]


_STD_MASO = set("01 10 11 20 21 22 23 25 26 30 31 32 40 50 51 52 60 70 100 110 120 130 140 "
                "200 220 230 250 270 300 310 320 330 338 400 410 411 420 440".split())


def _maso_col(df):
    """Trả (chỉ số cột mã số, số mã chuẩn trong cột) — cột chứa NHIỀU mã chuẩn = statement chính."""
    best_i, best_n = -1, 0
    for i, col in enumerate(list(df.columns)[:3]):
        vals = set(df[col].astype(str).str.strip())
        n = len(vals & _STD_MASO)
        if n > best_n:
            best_i, best_n = i, n
    return best_i, best_n


def select_maso_table(question, rel_docs, n_cand=12):
    """Chọn bảng CÓ mã số câu hỏi Ở CỘT MÃ SỐ, ưu tiên statement CHÍNH (cột mã có nhiều mã chuẩn)."""
    cands = tables_in_reports(question, rel_docs, n=n_cand)
    mm = maso_of(question)                       # (code, ...) hoặc None
    if mm and cands:
        target = str(mm[0]).strip()
        scored = []
        for c in cands:
            df = _read_cached(c["table_ref"])
            if df is None or df.shape[1] == 0:
                continue
            ci, cn = _maso_col(df)               # cột mã số + độ "statement chính"
            if ci >= 0 and (df[df.columns[ci]].astype(str).str.strip() == target).any():
                scored.append((cn, c))           # cn cao = statement chính chứa mã
        if scored:
            scored.sort(key=lambda x: -x[0])
            return [scored[0][1]]
    return cands[:1]


def product_retrieval(question, *, candidate_limit=40, label_weight=4.0):
    """Run the source-grounded retrieval stack used by the product service.

    The mode is opt-in in the submission builder so historical artifacts keep
    their exact behavior. Document hits are kept separate from table hits: a
    representative table that locates a report is not necessarily the best
    table for answering the question.
    """
    facets = extract_all_facets(question)
    doc_hits = retrieve_decomposed(
        question,
        per_pair=1,
        cap=40,
        backfill_unique_preferred_scope=True,
        backfill_unique_scope_neutral=True,
        backfill_explicit_opening_year=True,
        backfill_financial_formula_year=True,
        backfill_growth_scan_year=True,
        backfill_comparative_selection_year=True,
        backfill_comparative_series_year=True,
        retain_ambiguous_series_reports=True,
        implicit_ownership_scope_minimum_ratio=1.2,
        retain_count_scope_fallback=True,
        expand_catalog_universe=True,
        universe_cap=200,
    )
    report_ids = list(dict.fromkeys(doc_of(hit["table_ref"]) for hit in doc_hits))
    raw_tables = tables_in_reports(
        question,
        report_ids,
        n=max(1, int(candidate_limit)),
    )
    try:
        table_hits = rerank_tables(
            question,
            raw_tables,
            CAT,
            Path("build/tables"),
            label_weight=float(label_weight),
        )
    except Exception as exc:
        print(
            f"  [retrieval] CSV-label rerank fallback: {type(exc).__name__}",
            flush=True,
        )
        table_hits = raw_tables
    return facets, doc_hits, table_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--k-ret", type=int, default=3)   # số bảng cho relevant_tables (F2 recall)
    ap.add_argument("--k-ans", type=int, default=2)   # số bảng đưa vào answering (df1,df2)
    ap.add_argument("--out", type=Path, default=Path("sub_full"))
    ap.add_argument("--model", default="qwen3")       # coder | qwen3
    execution_group = ap.add_mutually_exclusive_group()
    execution_group.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Chỉ truy hồi (answer=0.0), KHÔNG gọi LLM — probe format + đo Tables F2 miễn phí",
    )
    execution_group.add_argument(
        "--compiler-only",
        action="store_true",
        help=(
            "Chỉ chạy grounded compiler, không gọi LLM; fail toàn lượt nếu có câu "
            "nằm ngoài grammar (dùng cho pilot/CI, không dùng để lấp 0 private set)"
        ),
    )
    ap.add_argument("--doc-diverse", action="store_true",
                    help="Mỗi báo cáo 1 bảng tốt nhất -> phủ nhiều doc gold (cả 2 scope) -> đẩy DOCS recall")
    ap.add_argument("--year-window", type=int, default=0, help="Cho phép năm lân cận ±N")
    ap.add_argument("--engine", default="sc", choices=["single", "sc", "mpr", "smart", "program"],
                    help="single=1 lần; sc=self-consistency n bản; mpr=phân rã+trích+synth+vote; smart=ensemble; program=kỷ luật vendor (CSV thô+đơn vị tường minh, đòn bẩy chính)")
    ap.add_argument("--n-vote", type=int, default=5, help="Số bản sinh để vote (sc). n=5 rẻ lấy phần lớn lợi ích; n=10 tối ưu EA (Innovation-LLM) nếu dư budget")
    ap.add_argument("--decompose", action="store_true",
                    help="Câu phân tích (đa mã/khoảng năm) -> phân rã (mã×năm) union, phủ nhiều doc gold")
    ap.add_argument(
        "--product-retrieval",
        action="store_true",
        help=(
            "Dùng retrieval source-grounded của ProductService: 5 facet "
            "backfill và rerank nhãn dòng CSV. Chế độ này là opt-in."
        ),
    )
    ap.add_argument(
        "--table-rerank-pool",
        type=int,
        default=40,
        help="Số bảng BM25 ứng viên trước khi rerank (product default: 40)",
    )
    ap.add_argument(
        "--table-label-weight",
        type=float,
        default=4.0,
        help="Trọng số khớp nhãn dòng CSV (product default: 4.0)",
    )
    ap.add_argument("--workers", type=int, default=1, help="Số luồng answering song song (vLLM batch; ~24 để no batch 3 GPU)")
    ap.add_argument("--maso", action="store_true", help="Option C: Mã số TT200 verify/cứu đáp án LLM (pandas thật)")
    ap.add_argument("--rel-k", type=int, default=0, help="Nới relevant_tables = top-k bảng trong ĐÚNG báo cáo (ăn recall F2 3.1). 0 = giữ hits gốc")
    ap.add_argument("--smart-table", action="store_true", help="Chọn ĐÚNG bảng chứa mã số câu hỏi (feed 1 bảng precision cao cho fine-tune)")
    ap.add_argument("--max-fix", type=int, default=2, help="Số lần self-debug (engine program). 2 = cân bằng chi phí/độ phủ")
    ap.add_argument("--resume", action="store_true", help="Tiếp tục từ _ckpt.jsonl (KHÔNG xoá out) — an toàn khi hết tiền GPU giữa chừng")
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Cho phép xóa đúng thư mục --out đã kiểm tra và ghi đè ZIP cùng tên",
    )
    ap.add_argument("--questions", default="data/questions/questions.jsonl", help="File câu hỏi (override để pilot subset)")
    compiler_group = ap.add_mutually_exclusive_group()
    compiler_group.add_argument(
        "--grounded-compiler",
        dest="grounded_compiler",
        action="store_true",
        default=True,
        help=(
            "Ưu tiên grammar source-bound cho chỉ tiêu một công ty/một năm và "
            "argmax/argmin panel bảo thủ (mặc định bật)"
        ),
    )
    compiler_group.add_argument(
        "--no-grounded-compiler",
        dest="grounded_compiler",
        action="store_false",
        help="Tắt compiler để chạy ablation model/retrieval thuần",
    )
    a = ap.parse_args()
    if a.table_rerank_pool < 1:
        ap.error("--table-rerank-pool phải >= 1")
    if a.table_label_weight < 0:
        ap.error("--table-label-weight phải >= 0")

    root = Path.cwd().resolve()
    out = a.out.resolve()
    if out == root or root not in out.parents:
        ap.error("--out phải là một thư mục con cụ thể bên trong project")
    a.out = out
    zip_path = a.out.parent / f"{a.out.name}.zip"
    if a.resume and a.overwrite:
        ap.error("--resume và --overwrite không được dùng cùng nhau")
    if a.out.exists() and not a.resume and not a.overwrite:
        ap.error("--out đã tồn tại; dùng --resume hoặc --overwrite một cách tường minh")
    if zip_path.exists() and not a.resume and not a.overwrite:
        ap.error("ZIP đầu ra đã tồn tại; dùng --overwrite một cách tường minh")

    grounded_compiler = None
    if a.grounded_compiler and not a.retrieval_only:
        candidate_compiler = DeterministicFinancialCompiler(Path.cwd())
        if candidate_compiler.available:
            grounded_compiler = candidate_compiler
        else:
            print("[compiler] statement cube/catalog unavailable -> model fallback", flush=True)
    if a.compiler_only and grounded_compiler is None:
        ap.error("--compiler-only cần statement cube/catalog và grounded compiler đang bật")

    key, models, single_llm = None, None, None
    if not a.retrieval_only and not a.compiler_only:      # probe offline không cần LLM/env
        key = os.environ["KINGPRO_LLM_API_KEY"]
        base_coder, model_coder = os.environ["BASE_CODER"], os.environ["MODEL_CODER"]
        if a.engine in ("single", "program"):
            _temp = 0.6 if a.n_vote > 1 else 0            # vote cần temp>0 để các mẫu khác nhau
            single_llm = lambda s, u: chat(s, u, base_url=base_coder, api_key=key, model=model_coder, temperature=_temp, max_tokens=1500, timeout=300)
        elif a.engine == "smart":                         # cần Qwen3 (đã xoá) — chỉ còn để tương thích
            models = [("coder", base_coder, model_coder),
                      ("qwen3", os.environ.get("BASE_QWEN3", base_coder), os.environ.get("MODEL_QWEN3", model_coder))]

    import threading
    data_dir = a.out / "data"
    ckpt = a.out / "_ckpt.jsonl"
    if a.out.exists() and a.overwrite:
        shutil.rmtree(a.out)
    data_dir.mkdir(parents=True, exist_ok=True)
    _ck_lock = threading.Lock()
    done_ids = set()
    if a.resume and ckpt.exists():
        for _l in open(ckpt, encoding="utf-8"):
            try:
                done_ids.add(json.loads(_l)["q"]["id"])
            except Exception:
                pass
        print(f"[resume] đã có {len(done_ids)} câu trong checkpoint -> bỏ qua", flush=True)

    qs = [json.loads(l) for l in open(a.questions, encoding="utf-8")][: a.n]
    copied: dict[str, str] = {}       # table_ref -> "data/<name>.csv"
    rows = []

    def process_one(iq):
        """Pha TÍNH (thread-safe): retrieve + answer. KHÔNG động vào data/ (để pha copy tuần tự)."""
        i, q = iq
        if grounded_compiler is not None:
            try:
                compiled = grounded_compiler.compile(
                    q["question"], extract_all_facets(q["question"])
                )
            except Exception as exc:
                print(
                    f"  [compiler] Q{q['id']} fallback: {type(exc).__name__}",
                    flush=True,
                )
                compiled = None
            if compiled is not None:
                rel_tables = list(compiled.table_refs)
                rel_docs = list(dict.fromkeys(doc_of(t) for t in rel_tables))
                res = {
                    "ok": True,
                    "answer": compiled.answer,
                    "pandas_query": compiled.pandas_query,
                    "evidence": compiled.submission_evidence(),
                    "attempts": 0,
                    "mode": "deterministic_compiler",
                }
                print(
                    f"[{i}/{len(qs)}] Q{q['id']} ok=True "
                    f"ans={compiled.answer} mode=deterministic_compiler",
                    flush=True,
                )
                rec = {
                    "q": q,
                    "rel_tables": rel_tables,
                    "rel_docs": rel_docs,
                    "res": res,
                }
                with _ck_lock:
                    with open(ckpt, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                return rec
        if a.compiler_only:
            raise RuntimeError(
                f"Q{q['id']} nằm ngoài grounded compiler; compiler-only từ chối tạo artifact thiếu"
            )
        product_table_hits = None
        if a.product_retrieval:
            facets, doc_hits, product_table_hits = product_retrieval(
                q["question"],
                candidate_limit=max(
                    a.table_rerank_pool,
                    a.k_ret,
                    a.k_ans,
                    a.rel_k,
                ),
                label_weight=a.table_label_weight,
            )
            analytic = facets["analytic"]
            hits = product_table_hits[: max(1, a.k_ret)]
            rel_docs = list(
                dict.fromkeys(doc_of(hit["table_ref"]) for hit in doc_hits)
            )
        elif a.decompose:
            analytic = extract_all_facets(q["question"])["analytic"]
            hits = retrieve_decomposed(q["question"])       # câu đơn -> 1 doc (scope đúng); phân tích -> N doc
        else:
            hits = retrieve(q["question"], k=a.k_ret, doc_diverse=a.doc_diverse, year_window=a.year_window)
            analytic = False
        rel_tables = [h["table_ref"] for h in hits]
        if not a.product_retrieval:
            rel_docs = list(dict.fromkeys(doc_of(t) for t in rel_tables))
        if a.rel_k:                                        # nới TABLES trong đúng báo cáo -> tăng recall (F2 3.1)
            expanded = (
                product_table_hits[: a.rel_k]
                if product_table_hits is not None
                else tables_in_reports(q["question"], rel_docs, n=a.rel_k)
            )
            if expanded:
                rel_tables = list(dict.fromkeys([t["table_ref"] for t in expanded] + rel_tables))
        # ans_tables: feed NHIỀU bảng TRONG đúng báo cáo (1 report ~74 bảng -> feed 1 bảng dễ trượt)
        if a.smart_table:
            atabs = select_maso_table(q["question"], rel_docs, n_cand=12)   # 1 bảng đúng mã số
        elif product_table_hits is not None:
            if a.engine == "program":
                n_tab = 6 if analytic else 4
            elif a.decompose:
                n_tab = 10 if analytic else 6
            else:
                n_tab = a.k_ans
            atabs = product_table_hits[: max(1, n_tab)]
        elif a.engine == "program":
            n_tab = 6 if analytic else 4                                     # CSV thô -> ít bảng hơn (context)
            atabs = tables_in_reports(q["question"], rel_docs, n=n_tab)
        elif a.decompose:
            n_tab = 10 if analytic else 6
            atabs = tables_in_reports(q["question"], rel_docs, n=n_tab)
        else:
            # NHỒI NHIỀU BẢNG (synera-style): top-k bảng TRONG đúng báo cáo, không chỉ top-2 hit toàn cục.
            # Execution chạy trên bảng GOLD -> model phải THẤY đúng bảng mới viết code đúng (TABLES_RECALL 0.44 -> nhồi thêm).
            atabs = tables_in_reports(q["question"], rel_docs, n=a.k_ans) or hits[: a.k_ans]
        ans_tables = [{"table_ref": h["table_ref"], "csv_path": csv_full(h["table_ref"])}
                      for h in atabs if csv_full(h["table_ref"])]
        try:
            if a.retrieval_only:
                res = {"ok": False, "answer": None, "evidence": [], "pandas_query": ""}
            elif a.engine == "single":
                res = answer_question(q["question"], ans_tables, single_llm, max_fix=3)
            elif a.engine == "program":                   # đòn bẩy chính: kỷ luật vendor + CSV thô + đơn vị + vote
                from kingpro.answering.program_engine import run_program
                res = run_program(q["question"], ans_tables, single_llm, max_fix=a.max_fix, n_vote=a.n_vote)
                res["ok"] = res.get("answer") is not None
            elif a.engine == "sc":                        # 1 model + self-consistency n bản (công thức thắng)
                res = self_consistent(q["question"], ans_tables, base_coder, key, model_coder,
                                      n=a.n_vote, temp=0.6, max_fix=2)
                res["ok"] = res.get("answer") is not None
            elif a.engine == "mpr":                       # phân rã+trích+synth (câu phân tích), câu đơn -> sc
                res = mpr_answer(q["question"], ans_tables, base_coder, key, model_coder,
                                 n_vote=a.n_vote, max_fix=2, analytic=analytic)
                res["ok"] = res.get("answer") is not None
            else:
                res = smart_answer(q["question"], ans_tables, models, key, max_fix=2)
                res["ok"] = res.get("answer") is not None
        except Exception as e:                            # 1 câu lỗi mạng/timeout không giết cả lượt
            print(f"  [!] Q{q['id']} lỗi: {type(e).__name__}: {str(e)[:80]}", flush=True)
            res = {"ok": False, "answer": None, "evidence": [], "pandas_query": ""}
        if a.maso and not a.retrieval_only:               # Option C: Mã số verify/cứu (pandas thật)
            try:
                res = maso_rescue(q["question"], ans_tables, res)
            except Exception:
                pass
        print(f"[{i}/{len(qs)}] Q{q['id']} ok={res.get('ok')} ans={res.get('answer')}", flush=True)
        rec = {"q": q, "rel_tables": rel_tables, "rel_docs": rel_docs, "res": res}
        with _ck_lock:                                    # CHECKPOINT: ghi ngay -> hết tiền GPU cũng không mất
            with open(ckpt, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec

    if not a.compiler_only:
        retrieve("khởi động index", k=1)                 # pre-warm _CACHE trước khi vào pool (tránh race)
    items = [(i, q) for i, q in enumerate(qs, 1) if q["id"] not in done_ids]
    if items and not a.retrieval_only and not a.compiler_only and a.workers > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(process_one, items))
    else:
        for it in items:
            process_one(it)
    # Dựng submission từ TOÀN BỘ checkpoint (câu cũ + mới), khử trùng theo id (giữ bản mới nhất)
    _by_id = {}
    if ckpt.exists():
        for _l in open(ckpt, encoding="utf-8"):
            try:
                _r = json.loads(_l)
                _by_id[_r["q"]["id"]] = _r
            except Exception:
                pass
    computed = list(_by_id.values())
    expected_ids = {q["id"] for q in qs}
    computed_ids = set(_by_id)
    if computed_ids != expected_ids:
        missing = sorted(expected_ids - computed_ids)
        extra = sorted(computed_ids - expected_ids)
        raise RuntimeError(
            f"checkpoint không khớp questions: missing={missing[:10]} extra={extra[:10]}"
        )

    for c in computed:                                    # pha COPY tuần tự (an toàn)
        q, rel_tables, rel_docs, res = c["q"], c["rel_tables"], c["rel_docs"], c["res"]
        evidence = []
        for ev in res.get("evidence", []):
            tref = ev["table_ref"]
            if tref not in copied:
                dst = data_dir / safe_name(tref)
                shutil.copyfile(csv_full(tref), dst)
                copied[tref] = f"data/{dst.name}"
            evidence.append({"variable": ev["variable"], "csv_path": copied[tref]})

        _av = coerce_number(res.get("answer")) if res.get("ok") else None   # ép an toàn, Series/chuỗi -> None
        rows.append({
            "id": q["id"],
            "question": q["question"],
            "answer": float(_av) if _av is not None else 0.0,
            "relevant_docs": rel_docs,
            "relevant_tables": rel_tables,
            "evidence": evidence,
            "pandas_query": res.get("pandas_query") or "",
        })

    (a.out / "submission.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    # đảm bảo data/ LUÔN tồn tại đúng hình dạng đặc tả (kể cả probe retrieval-only không có evidence)
    if not any(data_dir.glob("*.csv")):
        (data_dir / "_placeholder.csv").write_text("0\n0\n", encoding="utf-8")
    with zipfile.ZipFile(
        zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as z:
        write_deterministic(z, a.out / "submission.json", "submission.json")
        for csv in sorted(data_dir.glob("*.csv")):
            write_deterministic(z, csv, f"data/{csv.name}")
    archive_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest().upper()
    print(
        f"\nZIP -> {zip_path} | {len(rows)} câu | {len(copied)} CSV trong data/ "
        f"| SHA-256 {archive_sha256}"
    )


if __name__ == "__main__":
    main()
