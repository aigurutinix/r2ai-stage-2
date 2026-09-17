"""Prompts that hand the model RAW report text, not a structured index.

Two teams on the board score EXEC 0.6067 and 0.6601 with TABLES_F2 exactly 0.0.
Declaring a table costs nothing once you have located one, so a flat zero means they
never locate one — which fits reading the report directly rather than indexing it.

That is the opposite of what I built. Converting a report into (statement kind, Mã số,
period) destroys what makes a figure findable: the "Đơn vị tính" line above the table,
the note heading before it, the column header, the row label. In raw text those sit
within a few lines of each other; in my index each had to be reconstructed by a separate
mechanism and a separate measurement.

Two decisions that would each have cost a GPU pass to discover:

  the source   `official_corpus/*_extracted.txt` holds `[table_N](….csv)` anchors, so the
               table CONTENT lives in separate csv files and the text carries no figures
               at all. `data/financial_statements/**_extracted.txt` inlines the tables as
               `<table>` html — that is the version a reader needs.
  the contract the model reads html text, and its position there cannot be mapped to the
               official csv numbering. So it returns the figure AS PRINTED plus the row
               label, and code then finds the cell that holds that figure in that row.
               A hallucinated number matches no cell and the question is dropped rather
               than answered — the same guarantee every other path here keeps.

Windows are selected by metric-word overlap and capped so the prompt fits an 8k context.

Usage:
  python scripts/fresh/build_prompts_raw.py --out artifacts/fresh/prompts_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_answers import PARENT_RE, STRIP_CHUNKS, YEAR_RE  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

# Two windows of 6,000 characters is about 4,800 tokens of Vietnamese with grouped
# numbers, which leaves room for the question and the instructions inside 8k.
WINDOW_CHARS = 6000
WINDOWS = 2
STRIDE_LINES = 12
BLOCK_LINES = 46
STOP = frozenset({"cua", "cong", "ty", "nam", "cuoi", "dau", "bao", "nhieu", "la",
                  "trong", "cho", "tai", "theo", "phan", "tram", "ctcp", "tong",
                  "gia", "tri", "muc", "khoan", "so", "du", "dong", "trieu", "ty"})

SYSTEM = """Bạn đọc một đoạn báo cáo tài chính Việt Nam (văn bản OCR, bảng ở dạng HTML)
và tìm ĐÚNG MỘT con số mà câu hỏi cần.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"so": "<con số y như in trong bảng>", "nhan_dong": "<nhãn dòng chứa nó>", "cot": "<tiêu đề cột>"}

Quy tắc:
- `so` phải COPY Y NGUYÊN từ văn bản, giữ dấu chấm phân cách và dấu ngoặc nếu có.
  Đừng làm tròn, đừng đổi đơn vị, đừng tự tính.
- `nhan_dong` copy y nguyên nhãn dòng, để hệ thống tìm lại được ô đó.
- Chú ý dòng "Đơn vị tính" phía trên bảng: nó cho biết bảng đang tính bằng đồng, nghìn,
  triệu hay tỷ. Nhưng bạn vẫn copy con số như in, KHÔNG quy đổi.
- Chú ý tiêu đề cột: "Số cuối năm" / "Số đầu năm" / "Năm nay" / "Năm trước" là các kỳ
  khác nhau. Câu hỏi "cuối năm N" của báo cáo năm N là cột đầu; "đầu năm N" là cột sau.
- Phân biệt các cặp dễ lẫn: "trước thuế" khác "sau thuế"; "ngắn hạn" khác "dài hạn";
  "nguyên giá" khác "giá trị còn lại"; "phải thu" khác "phải trả".
- Nếu đoạn văn bản không chứa con số câu hỏi cần, trả {"so": "", "nhan_dong": "", "cot": ""}."""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

<đoạn_báo_cáo>
{window}
</đoạn_báo_cáo>

Con số nào là đáp án?"""


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", flat)).strip()


def metric_tokens(text: str, ticker: str, name: str) -> set[str]:
    probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", text)
    probe = re.sub(r"\([^)]*\)", " ", probe)
    probe = YEAR_RE.sub(" ", probe)
    for chunk in (name,) + STRIP_CHUNKS:
        probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
    return {t for t in fold(probe).split() if len(t) > 2 and t not in STOP}


def best_windows(lines: list[str], wanted: set[str], *, count: int = WINDOWS,
                 width: int = WINDOW_CHARS) -> list[str]:
    """The blocks of text whose words most overlap the question's metric words."""

    scored = []
    for start in range(0, max(1, len(lines) - BLOCK_LINES + 1), STRIDE_LINES):
        block = lines[start:start + BLOCK_LINES]
        words = set(fold(" ".join(block)).split())
        shared = len(wanted & words)
        if shared:
            scored.append((shared / len(wanted), start, block))
    if not scored:
        return []
    scored.sort(key=lambda item: -item[0])

    chosen, used = [], []
    for score, start, block in scored:
        # Skip a block that overlaps one already taken; two copies of the same page
        # waste the window.
        if any(abs(start - other) < BLOCK_LINES for other in used):
            continue
        text = "\n".join(block)
        if len(text) > width:
            text = text[:width]
        chosen.append(text)
        used.append(start)
        if len(chosen) >= count:
            break
    return chosen


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/fresh/prompts_raw.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    # A wide pass exists because the narrow one abstains: the model answers only when
    # the figure is inside the window it was given, so an abstention is as likely to
    # mean "wrong window" as "not in the report". Re-running only the abstainers with
    # twice the text costs a fraction of a full pass.
    parser.add_argument("--windows", type=int, default=WINDOWS)
    parser.add_argument("--window-chars", type=int, default=WINDOW_CHARS)
    parser.add_argument("--ids-file", default="",
                        help="json list of ids to build, for a re-run")
    args = parser.parse_args()

    only = None
    if args.ids_file:
        only = set(json.loads(
            (ROOT / args.ids_file).read_text(encoding="utf-8")))

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    root = ROOT / "data" / "financial_statements"
    cache: dict[Path, list[str]] = {}
    written = 0
    skipped = {"khong nhan ra ma": 0, "khong co nam": 0, "khong co tai lieu": 0,
               "khong con tu chi tieu": 0, "khong tim duoc cua so": 0}
    sizes = []
    started = time.time()

    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for number, question in enumerate(questions, start=1):
            if only is not None and question["id"] not in only:
                continue
            text = question["question"]
            found = sorted(resolver.resolve(text))
            years = YEAR_RE.findall(text)
            if not found:
                skipped["khong nhan ra ma"] += 1
                continue
            if not years:
                skipped["khong co nam"] += 1
                continue
            ticker, year = found[0], max(years)
            want_separate = bool(PARENT_RE.search(text))
            base = root / ticker / year
            if not base.is_dir():
                skipped["khong co tai lieu"] += 1
                continue
            wanted = metric_tokens(text, ticker, resolver.tickers.get(ticker, ""))
            if not wanted:
                skipped["khong con tu chi tieu"] += 1
                continue

            docs = sorted(p for p in base.iterdir() if p.is_dir())
            ordered = ([d for d in docs if ("separate" in d.name) == want_separate]
                       + [d for d in docs if ("separate" in d.name) != want_separate])
            windows, doc_used = [], None
            for doc_dir in ordered:
                path = doc_dir / f"{doc_dir.name}_extracted.txt"
                if not path.exists():
                    continue
                if path not in cache:
                    cache[path] = [line for line in path.read_text(
                        encoding="utf-8", errors="replace").splitlines()]
                windows = best_windows(cache[path], wanted,
                                       count=args.windows,
                                       width=args.window_chars)
                if windows:
                    doc_used = doc_dir.name
                    break
            if not windows:
                skipped["khong tim duoc cua so"] += 1
                continue

            body = "\n\n[…]\n\n".join(windows)
            handle.write(json.dumps({
                "id": question["id"],
                "system": SYSTEM,
                "user": USER.format(question=text, window=body),
                "meta": {"question": text, "ticker": ticker, "year": year,
                         "doc": doc_used,
                         "scope": "separate" if want_separate else "consolidated"},
            }, ensure_ascii=False) + "\n")
            written += 1
            sizes.append(len(body))
            if number % 200 == 0:
                print(f"  {number}/{len(questions)}  {time.time() - started:.0f}s",
                      flush=True)

    print(f"\n{written} prompt")
    for name, count in skipped.items():
        if count:
            print(f"  bo qua — {name}: {count}")
    if sizes:
        sizes.sort()
        print(f"  kich thuoc: p50={sizes[len(sizes) // 2]} max={sizes[-1]} ky tu")
        print(f"  uoc token: p50={sizes[len(sizes) // 2] // 2.5:.0f} "
              f"max={sizes[-1] // 2.5:.0f}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
