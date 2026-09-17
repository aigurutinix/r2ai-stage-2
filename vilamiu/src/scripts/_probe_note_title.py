"""Ask the model to name the note a question's figure lives in, and nothing else.

Table selection is the weakest measured step in the pipeline: given that the gold
table is in our eight-table shortlist, our machinery picks it 38.1% of the time. Hand
the selector the gold table's own caption and it picks it 97.6% of the time. The
question's own wording recovers only 39.4% of that.

So the whole gap is one translation: from how a question is phrased to how the report
titles the note holding the figure. Vietnamese statements are organised into numbered
notes — "9. CHO VAY KHÁCH HÀNG", "19. PHÁT HÀNH GIẤY TỜ CÓ GIÁ" — and the question
usually names a line *inside* one of them, not the note itself:

  question: "Dư nợ cho vay các tổ chức kinh tế, cá nhân trong nước … cuối năm 2019"
  note    : "9. CHO VAY KHÁCH HÀNG"

That is a language task, not a navigation task, and it is the one thing the model has
never been asked to do. No tables go into the prompt, so it costs almost nothing to
run: the input is a question and the output is a heading.

Scored against the caption of the table the generator actually read, by the same token
overlap the selector would use. The baseline to beat is 39.4% — what the question's
own metric phrase already achieves with no model at all.

Usage:
  PYTHONPATH=src python scripts/_probe_note_title.py \
      --model Qwen/Qwen3-14B --local-url http://127.0.0.1:18000/v1 --limit 300
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.llm.client import ChatClient  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

SYSTEM = """Bạn đọc câu hỏi về báo cáo tài chính Việt Nam và cho biết con số đó nằm
trong THUYẾT MINH nào.

Báo cáo tài chính Việt Nam chia thành các thuyết minh có tiêu đề in hoa, ví dụ:
  "TIỀN VÀ CÁC KHOẢN TƯƠNG ĐƯƠNG TIỀN"
  "CHO VAY KHÁCH HÀNG"
  "PHÁT HÀNH GIẤY TỜ CÓ GIÁ"
  "TÀI SẢN CỐ ĐỊNH HỮU HÌNH"
  "CÁC KHOẢN PHẢI THU NGẮN HẠN"
  "CHI PHÍ SẢN XUẤT KINH DOANH THEO YẾU TỐ"
  "THÔNG TIN VỀ CÁC BÊN LIÊN QUAN"
  "BÁO CÁO BỘ PHẬN"

Câu hỏi thường gọi tên một DÒNG bên trong thuyết minh, không phải tên thuyết minh.
Việc của bạn là suy ra tên thuyết minh MẸ.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"note": "<tiêu đề thuyết minh>", "row": "<tên dòng trong thuyết minh đó>"}

Quy tắc:
- `note` viết theo cách báo cáo đặt tiêu đề: cụm danh từ, không số thứ tự, không tên
  công ty, không năm.
- `row` là tên khoản mục mà câu hỏi hỏi tới, viết theo lời báo cáo.
- Giữ đúng các cặp phân biệt: hữu hình/vô hình, ngắn hạn/dài hạn, trước thuế/sau
  thuế, phải thu/phải trả.
- Nếu con số nằm ngay trên báo cáo chính, dùng "BẢNG CÂN ĐỐI KẾ TOÁN",
  "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH" hoặc "BÁO CÁO LƯU CHUYỂN TIỀN TỆ"."""

USER = """Câu hỏi: {question}

Con số này nằm trong thuyết minh nào, và ở dòng nào?"""

JSON_RE = re.compile(r"\{[^{}]*\}")
THINK_RE = re.compile(r"<think>.*?</think>", re.S)
BOILERPLATE = ("thuyet minh nay la bo phan", "doc dong thoi voi", "ban hanh theo",
               "thong tu so", "mau b", "mau so b")


def fold(text: str) -> str:
    text = unicodedata.normalize("NFD", str(text).lower())
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text)).strip()


def overlap(a: str, b: str) -> float:
    left = {t for t in fold(a).split() if len(t) > 2}
    right = {t for t in fold(b).split() if len(t) > 2}
    if not left or not right:
        return 0.0
    shared = len(left & right)
    if not shared:
        return 0.0
    coverage = shared / len(left)
    focus = shared / len(right)
    return 2 * coverage * focus / (coverage + focus)


def useful(caption: str) -> bool:
    flat = fold(caption)
    return bool(flat) and not any(mark in flat for mark in BOILERPLATE)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="artifacts/easy_full_units.jsonl")
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--shortlist", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--model", default="Qwen/Qwen3-14B")
    parser.add_argument("--local-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--out", default="artifacts/_note_title.jsonl")
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    retriever = LexicalRetriever(store.frame)
    client = (ChatClient.local(args.model, args.local_url) if args.local_url
              else ChatClient.from_env(ROOT, model=args.model))

    prose: dict[tuple[str, int], str] = {}
    prose_path = ROOT / "artifacts" / "table_prose.jsonl"
    if prose_path.exists():
        for line in prose_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prose[(row["doc"], int(row["table_id"]))] = row["prose"][:160]

    work = []
    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or len(work) >= args.limit:
            continue
        record = json.loads(line)
        refs = record.get("relevant_tables") or []
        if len(refs) != 1:
            continue
        doc, _, table_id = refs[0].rpartition("|table_")
        if not table_id.isdigit():
            continue
        gold_key = TableKey(doc, int(table_id))
        parsed = parse_question(record.get("id", 0), record["question"], roster)
        groups = max(1, len(parsed.tickers)) * max(1, len(parsed.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            parsed, per_group=per_group, cap=args.shortlist)]
        if not keys or gold_key not in keys:
            continue
        work.append((record, parsed, keys, gold_key))

    print(f"{len(work)} câu có bảng gold trong shortlist, model={args.model}",
          flush=True)

    def caption_of(key: TableKey) -> str:
        text = str(getattr(store.meta(key), "caption", ""))
        if not useful(text):
            text = prose.get((key.doc_name, key.table_id), text)
        return text

    tally = collections.Counter()
    lock = threading.Lock()
    rows_out = []
    started = time.time()

    def run_one(item) -> None:
        record, parsed, keys, gold_key = item
        try:
            reply = client.complete(SYSTEM, USER.format(question=record["question"]))
        except RuntimeError:
            with lock:
                tally["lỗi truyền"] += 1
            return
        match = JSON_RE.search(THINK_RE.sub("", reply or ""))
        note = row_hint = ""
        if match:
            try:
                data = json.loads(match.group(0))
                note = str(data.get("note", "")).strip()
                row_hint = str(data.get("row", "")).strip()
            except (ValueError, TypeError):
                note = ""

        captions = {key: caption_of(key) for key in keys}
        metric = lookup_mod.extract_metric(record["question"])

        def pick(text: str):
            best, best_score = None, -1.0
            for key in keys:
                if not useful(captions[key]):
                    continue
                value = overlap(text, captions[key])
                if value > best_score:
                    best, best_score = key, value
            return best, best_score

        by_metric, _ = pick(metric)
        by_note, note_score = pick(note) if note else (None, 0.0)
        # The note names the section and the row names the line; together they are
        # what a caption would contain, so the pair is scored as well.
        by_both, _ = pick(f"{note} {row_hint}") if note else (None, 0.0)

        with lock:
            tally["n"] += 1
            tally["nền: chỉ tiêu -> caption"] += int(by_metric == gold_key)
            tally["MODEL: note -> caption"] += int(by_note == gold_key)
            tally["MODEL: note+row -> caption"] += int(by_both == gold_key)
            if not note:
                tally["model không trả note"] += 1
            rows_out.append({"id": record.get("id"), "note": note, "row": row_hint,
                             "gold_caption": captions[gold_key][:90],
                             "hit_note": by_note == gold_key,
                             "hit_metric": by_metric == gold_key})
            if tally["n"] % 50 == 0:
                n = tally["n"]
                print(f"  {n}/{len(work)}  nền={tally['nền: chỉ tiêu -> caption'] / n:.1%}"
                      f"  note={tally['MODEL: note -> caption'] / n:.1%}"
                      f"  note+row={tally['MODEL: note+row -> caption'] / n:.1%}"
                      f"  {time.time() - started:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run_one, work))

    n = max(tally["n"], 1)
    print(f"\n== {tally['n']} câu")
    for name in ("nền: chỉ tiêu -> caption", "MODEL: note -> caption",
                 "MODEL: note+row -> caption", "model không trả note", "lỗi truyền"):
        count = tally.get(name, 0)
        print(f"  {name:30s} {count:5d}  {count / n:6.1%}")
    print("\n  (nền không model đo trước đây: 39,4% | biết caption gold: 97,6%)")
    (ROOT / args.out).write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows_out),
        encoding="utf-8")


if __name__ == "__main__":
    main()
