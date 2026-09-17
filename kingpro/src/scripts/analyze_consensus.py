"""Ý của Duy: phân tích AI TRẢ LỜI THẾ NÀO trên 1012 câu để biết nó VẤP chỗ nào, NGON chỗ nào.
Tín hiệu KHÔNG thiên lệch = mức đồng thuận MBR (agree/attempts) trên MỌI câu (không cần gold).
5/5 = tự tin (nhiều khả năng đúng); 1/5 loạn = vấp (nhiều khả năng sai).

Chạy: PYTHONUTF8=1 python scripts/analyze_consensus.py [cache_file]
"""
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, "src")
QS = {q["id"]: q["question"] for q in (json.loads(l) for l in open("data/questions/questions.jsonl", encoding="utf-8"))}
CACHE = sys.argv[1] if len(sys.argv) > 1 else "build/llm_mbr_cache.jsonl"
cache = {r["id"]: r for r in (json.loads(l) for l in open(CACHE, encoding="utf-8"))}

YEARS = re.compile(r"20\d{2}")
SEL = re.compile(r"cao nhất|thấp nhất|lớn nhất|nhỏ nhất|nhiều nhất|ít nhất|trung vị", re.I)
GROUP = re.compile(r"nhóm|các công ty|các doanh nghiệp|các ngân hàng", re.I)
LIST = re.compile(r"các năm|những năm|giai đoạn", re.I)
PCT = re.compile(r"phần trăm|%|tỷ lệ|tỉ lệ|tỷ trọng|biên|tỷ suất|ROE|ROA", re.I)
LAN = re.compile(r"bao nhiêu lần|số lần|hệ số|vòng quay|D/E", re.I)
GROW = re.compile(r"tăng trưởng|tốc độ tăng", re.I)
DIFF = re.compile(r"chênh lệch|hiệu|thay đổi|biến động", re.I)
MEP = re.compile(r"công ty mẹ", re.I)


def bucket(q):
    yrs = set(YEARS.findall(q))
    if GROUP.search(q) and SEL.search(q):
        return "H cross-company"
    if len(yrs) >= 2 and SEL.search(q) and (LIST.search(q) or len(yrs) >= 3):
        return "M year-selector"
    if LAN.search(q):
        return "R hệ số/lần"
    if GROW.search(q):
        return "G tăng trưởng"
    if DIFF.search(q):
        return "D chênh lệch"
    if PCT.search(q):
        return "F/S tỷ số/%"
    return "L lookup"


def report(rows, title):
    """rows = list of (agree, attempts). In độ đồng thuận."""
    n = len(rows)
    if not n:
        return
    ok = sum(1 for a, t in rows if t > 0)
    strong = sum(1 for a, t in rows if t >= 4 and a >= t)         # 4/4 hoặc 5/5
    weak = sum(1 for a, t in rows if t >= 2 and a * 2 <= t)       # <= một nửa
    mean_ratio = sum((a / t) for a, t in rows if t > 0) / max(ok, 1)
    print(f"  {title:<20} n={n:>4}  đồng thuận TB={mean_ratio:.0%}  mạnh(5/5)={100*strong//n:>3}%  vấp(<=half)={100*weak//n:>3}%")


def main():
    have = [qid for qid in cache if qid in QS]
    print(f"Phân tích {len(have)}/{len(QS)} câu đã chạy (cache={CACHE})")
    print("=" * 78)
    allrows = [(cache[q].get("agree", 0), cache[q].get("attempts", 0)) for q in have]
    report(allrows, "TỔNG")
    print("-" * 78)
    from collections import defaultdict
    byb = defaultdict(list)
    for q in have:
        byb[bucket(QS[q])].append((cache[q].get("agree", 0), cache[q].get("attempts", 0)))
    # xếp theo tỷ lệ vấp giảm dần (chỗ vấp nhất lên đầu)
    def weakrate(rows):
        return sum(1 for a, t in rows if t >= 2 and a * 2 <= t) / max(len(rows), 1)
    print("THEO BUCKET (xếp chỗ VẤP nhất lên đầu):")
    for b in sorted(byb, key=lambda x: -weakrate(byb[x])):
        report(byb[b], b)
    print("-" * 78)
    # thêm chiều: công ty mẹ vs hợp nhất; đơn/đa thực thể
    me = [(cache[q].get("agree", 0), cache[q].get("attempts", 0)) for q in have if MEP.search(QS[q])]
    hn = [(cache[q].get("agree", 0), cache[q].get("attempts", 0)) for q in have if not MEP.search(QS[q])]
    print("THEO SCOPE:")
    report(me, "công ty mẹ")
    report(hn, "hợp nhất/khác")
    print("=" * 78)
    print("Đọc: bucket 'vấp' cao = AI loạn ở đó (cần đòn riêng). 'mạnh' cao = AI tự tin (giữ).")
    print("LƯU Ý: đồng thuận != đúng (có thể tự tin mà sai). Nhưng vấp cao GẦN CHẮC sai.")


if __name__ == "__main__":
    main()
