"""A/B embedding-hybrid: so anchor 'token-only' vs 'hybrid' (rerank top-K lexical bang embedding)
tren cac ca moderate label-branch ma token-overlap tung thua. Eyeball xem hybrid co chon dung metric hon."""
import pipeline as P
import re, json, math, urllib.request

EMB = "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"
INSTRUCT = "Instruct: Cho tên một chỉ tiêu tài chính, tìm dòng chỉ tiêu khớp nhất trong bảng báo cáo.\nQuery:"
def _embed(texts):
    req = urllib.request.Request("http://localhost:11434/api/embed",
        data=json.dumps({"model": EMB, "input": texts}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req))["embeddings"]
def _cos(a, b):
    d = sum(x*y for x, y in zip(a, b)); na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(x*x for x in b))
    return d / (na*nb + 1e-9)
def embedder(target, labels):        # batch <=16 (query + <=15 candidate) -> khong loi 400
    qv = _embed([INSTRUCT + target])[0]; lvs = _embed(labels)
    return [_cos(qv, lv) for lv in lvs]

byid = {q["id"]: q["question"] for q in P.Q}
def anchor_of(q, use_embed):
    tk = P.resolve(q); yrs = sorted(set(re.findall(r"\b(20\d{2})\b", q)))
    fr = P.find_report(tk, yrs[-1] if yrs else None, P.doctype(q))
    if not fr: return None
    rows = P.ingest(fr[0].read_text(encoding="utf-8", errors="replace"))[0]
    target = P.clean_metric(P.target_label(q))
    return P.locate(rows, target, None, embedder if use_embed else None)

IDS = [637, 601, 645, 578, 624, 612, 606, 655, 588, 583, 629, 793, 795]
print(f"{'id':>5} | {'metric (clean)':30} | TOKEN-only  vs  HYBRID (≠=doi)")
changed = 0
for qid in IDS:
    q = byid[qid]; m = P.clean_metric(P.target_label(q))
    try:
        a = anchor_of(q, False); b = anchor_of(q, True)
    except Exception as e:
        print(f"{qid:>5} ERR {e}"); continue
    if not a or not b:
        print(f"{qid:>5} skip (no anchor)"); continue
    diff = a["label"] != b["label"]
    if diff: changed += 1
    print(f"{qid:>5} | {m[:30]:30} | {a['label'][:28]:28} {'≠' if diff else '='} {b['label'][:32]}")
print(f"\nHybrid đổi anchor ở {changed}/{len(IDS)} ca (eyeball xem có đúng hơn không).")
