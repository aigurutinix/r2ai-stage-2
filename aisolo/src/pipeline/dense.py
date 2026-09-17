"""Dense table retrieval bằng BGE-M3 (GPU) + cache đĩa. Mirror `table_retrieval_text` của repo ViFinQA.
Chỉ phục vụ relevant_tables (retrieval) — KHÔNG đụng answer/execution. RRF kết hợp với BM25 (rrf_k=60)."""
import os, re, json
import numpy as np

_MODEL = None
_CACHE = os.path.join(os.path.dirname(__file__), "emb_cache")
_MEM = {}   # docname -> (lines:list[int], mat:np.ndarray) trong RAM
_MODEL_ID = "BAAI/bge-m3"

def _model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        kw = {"torch_dtype": torch.float16} if dev == "cuda" else {}   # fp16 trên GPU: 2× tốc độ, nửa VRAM
        _MODEL = SentenceTransformer(_MODEL_ID, device=dev, model_kwargs=kw)
        _MODEL.max_seq_length = 1024
    return _MODEL

def _clean(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", s)).strip()

def table_docs(txt, company, ticker, year):
    """Mỗi <table> → {line, text} theo format table_retrieval_text (metadata + ngữ cảnh caption + row-labels)."""
    out = []
    for m in re.finditer(r"<table>(.*?)</table>", txt, re.S):
        line = txt.count("\n", 0, m.start()) + 1
        body = m.group(1)
        # row-labels: ô chữ đầu tiên (không phải số) mỗi <tr>, tối đa 40 dòng
        labels = []
        for tr in re.findall(r"<tr>(.*?)</tr>", body, re.S)[:40]:
            cells = [_clean(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)]
            lab = next((c for c in cells if c and re.search(r"[A-Za-zÀ-ỹ]{3}", c) and not re.match(r"^[\d.,%()\-\s]+$", c)), "")
            if lab: labels.append(lab)
        ctx = _clean(txt[max(0, m.start() - 500):m.start()])[-400:]   # caption/heading ngay trước bảng
        text = (f"Công ty: {company} (mã {ticker}), năm {year}. "
                f"Ngữ cảnh: {ctx}. Dòng: {' / '.join(labels)}")
        out.append({"line": line, "text": text[:2000]})
    return out

def embed(texts, batch_size=16):
    if not texts: return np.zeros((0, 1024), dtype=np.float32)
    return np.asarray(_model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                      show_progress_bar=False), dtype=np.float32)

def doc_matrix(docname, txt, company, ticker, year):
    """Embedding tất cả bảng của 1 report (cache RAM + đĩa) → (lines, matrix[N,1024])."""
    if docname in _MEM: return _MEM[docname]
    os.makedirs(_CACHE, exist_ok=True)
    npy = os.path.join(_CACHE, docname + ".npy"); js = os.path.join(_CACHE, docname + ".lines.json")
    if os.path.exists(npy) and os.path.exists(js):
        lines = json.load(open(js)); mat = np.load(npy)
        _MEM[docname] = (lines, mat); return lines, mat
    tds = table_docs(txt, company, ticker, year)
    lines = [t["line"] for t in tds]; mat = embed([t["text"] for t in tds])
    np.save(npy, mat); json.dump(lines, open(js, "w"))
    _MEM[docname] = (lines, mat); return lines, mat

_QCACHE = {}
def query_vec(q):
    if q not in _QCACHE: _QCACHE[q] = embed([q])[0]
    return _QCACHE[q]

def dense_rank(docname, txt, company, ticker, year, q):
    """Xếp hạng bảng theo cosine(query, table) → dict{line: rank} (0 = tốt nhất)."""
    lines, mat = doc_matrix(docname, txt, company, ticker, year)
    if len(lines) == 0: return {}
    sims = mat @ query_vec(q)
    order = sorted(range(len(lines)), key=lambda i: -sims[i])
    return {lines[i]: r for r, i in enumerate(order)}

def rrf(bm_lines_ranked, dense_rank_map, k=60, top=3):
    """RRF kết hợp rank BM25 (list line theo thứ tự) + rank dense (dict) → top line."""
    bm_rank = {ln: i for i, ln in enumerate(bm_lines_ranked)}
    allln = set(bm_rank) | set(dense_rank_map)
    score = {ln: 1.0 / (k + bm_rank.get(ln, 10**9)) + 1.0 / (k + dense_rank_map.get(ln, 10**9)) for ln in allln}
    return sorted(allln, key=lambda ln: -score[ln])[:top]
