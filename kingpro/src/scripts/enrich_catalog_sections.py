"""LEVER A (đòn kma, EV#1): enrich catalog với SECTION_TITLE = tiêu đề mục/thuyết minh ngay TRƯỚC mỗi <table>.
25/32 nhãn-dòng gold xuất hiện ≥2 mảnh cùng report -> chọn nhầm occurrence. Heading ('29. Doanh thu hoạt động
tài chính' / 'BẢNG CÂN ĐỐI...') là tín hiệu phân biệt mà catalog ĐANG BỎ. Thêm nó vào search_text -> BM25 ranker
chọn ĐÚNG mảnh (đánh trực diện Tables-precision 0.20 -> mục tiêu ~0.38 như kma).

Chạy: PYTHONUTF8=1 python scripts/enrich_catalog_sections.py   (ghi build/catalog.jsonl mới, backup .bak)
"""
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kingpro.corpus.table_context import section_ancestors  # noqa: E402

CAT = "build/catalog.jsonl"
FS = "data/financial_statements"
RE_TABLE = re.compile(r"<table\b", re.IGNORECASE)

# nhiễu cần bỏ khi tìm heading
_NOISE = re.compile(
    r"^\s*=+\s*PAGE|đơn vị\s*:|mẫu\s+b|^\s*\d+\s*$|lô\s+cn|^\s*(công ty|tổng công ty|ngân hàng|ctcp)\b|"
    r"tỉnh|thành phố|quận|huyện|thị trấn|đường|^\s*số\s+\d|cho năm tài chính|tại ngày|^\s*$", re.IGNORECASE)
# heading đáng giá: note-number, hoặc tên báo cáo, hoặc dòng CHỮ đủ dài
_NOTE = re.compile(r"^\s*(\d{1,3}|[IVXLC]{1,5})\s*[.)]\s+\S")
_STMT = re.compile(r"BẢNG CÂN ĐỐI|BÁO CÁO KẾT QUẢ|LƯU CHUYỂN TIỀN|THUYẾT MINH|KẾT QUẢ (HOẠT ĐỘNG|KINH DOANH)", re.IGNORECASE)


def report_txt_path(report_id, ticker, year):
    d = os.path.join(FS, ticker, str(year), report_id)
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".txt"):
                return os.path.join(d, f)
    # fallback: dò
    base = os.path.join(FS, ticker, str(year))
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith(".txt") and report_id in f:
                    return os.path.join(root, f)
    return None


def heading_before(lines, table_line):
    """Compatibility wrapper around the canonical production helper."""

    return section_ancestors(lines, table_line)


def main():
    rows = [json.loads(l) for l in open(CAT, encoding="utf-8")]
    by_rep = defaultdict(list)
    for r in rows:
        by_rep[r["report_id"]].append(r)
    n_reports = len(by_rep)
    n_enriched = 0
    n_txt_miss = 0
    for i, (rep, entries) in enumerate(by_rep.items(), 1):
        e0 = entries[0]
        p = report_txt_path(rep, e0.get("ticker", ""), e0.get("year", ""))
        if not p or not os.path.exists(p):
            n_txt_miss += 1
            continue
        try:
            lines = open(p, encoding="utf-8", errors="ignore").read().split("\n")
        except Exception:
            continue
        for e in entries:
            ln = e.get("line", 0)
            h = heading_before(lines, ln)
            if h:
                e["section_title"] = h
                # prepend vào search_text để BM25 ranker weight heading
                st = e.get("search_text", "")
                if h not in st:
                    e["search_text"] = f"Mục: {h}. " + st
                n_enriched += 1
        if i % 300 == 0:
            print(f"  ...{i}/{n_reports} report, enriched {n_enriched} bảng", flush=True)
    print(f"Report: {n_reports}, thiếu txt: {n_txt_miss}, bảng có section_title: {n_enriched}/{len(rows)}", flush=True)
    # backup + ghi
    if not os.path.exists(CAT + ".bak"):
        shutil.copyfile(CAT, CAT + ".bak")
    with open(CAT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Đã ghi {CAT} (backup .bak). Nhớ rebuild BM25 index nếu nó cache search_text.", flush=True)


if __name__ == "__main__":
    main()
