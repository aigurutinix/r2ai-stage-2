"""Candidate TABLES with their headings, addressed exactly, for the model to read.

Three things have to be true at once for an answer to land, and every mechanism here so
far got one of them by breaking another:

  localisation  a report holds a hundred-odd tables, and the earlier model passes chose
                a row out of a block MY retrieval had already narrowed — so when the
                retrieval missed, the model could not win. Retrieval here is at TABLE
                level, which is coarse enough to be reliable and small enough that five
                candidates fit in an 8k window.
  the address   raw report text cannot be mapped back to the organisers' csv numbering,
                so a reader working on it needs its figure relocated afterwards. These
                candidates are rendered FROM the official csv files, with the row and
                column index printed beside every cell, so the reply IS the address.
  the context   the unit line and the note heading sit in the prose above the table in
                the source text, and are cut off by the csv boundary. The prose
                preceding each anchor is carried along with it.

Printing `c0 c1 c2` above the row and `r12` beside it is not decoration: an earlier pass
that left the model to infer column numbers was wrong by one in 49 of 58 cases, and
naming the columns explicitly moved that read from 50.0% to 69.1%.

Usage:
  python scripts/fresh/build_prompts_tables.py --out artifacts/fresh/prompts_tab.jsonl
"""

from __future__ import annotations

import argparse
import csv as csv_mod
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

ANCHOR_RE = re.compile(r"\[table_(\d+)\]\([^)]*\)")
CANDIDATES = 6
CONTEXT_CHARS = 420
MAX_ROWS = 56
MAX_COLS = 7
# The label column is not truncated like the rest: cutting "LỢI NHUẬN KẾ TOÁN SAU
# THUẾ" at 26 characters removes the word the model is told to distinguish from
# "trước thuế", which turns a solvable read into a coin flip.
LABEL_CHARS = 62
CELL_CHARS = 24
BUDGET = 11000

STOP = frozenset({"cua", "cong", "ty", "nam", "cuoi", "dau", "bao", "nhieu", "la",
                  "trong", "cho", "tai", "theo", "phan", "tram", "ctcp", "tong",
                  "gia", "tri", "muc", "khoan", "so", "du", "dong", "trieu", "ty"})

SYSTEM = """Bạn đọc các bảng trích từ báo cáo tài chính Việt Nam và chỉ ra ĐÚNG MỘT ô \
chứa con số câu hỏi cần.

Mỗi bảng được đánh số `[bảng N]`, kèm đoạn văn ngay phía trên bảng đó trong báo cáo
(thường có tiêu đề thuyết minh và dòng "Đơn vị tính"). Trong bảng, dòng được đánh `r0`,
`r1`, ... và cột được đánh `c0`, `c1`, ... Dòng đầu tiên in tên các cột.

Trả lời bằng ĐÚNG một đối tượng JSON, không kèm gì khác:
{"bang": <N>, "dong": <r>, "cot": <c>, "so": "<ô đó y như in>"}

Quy tắc:
- `so` phải COPY Y NGUYÊN nội dung ô, giữ dấu chấm phân cách và dấu ngoặc nếu có.
- Chọn cột theo kỳ mà câu hỏi hỏi: "Số cuối năm"/"Năm nay" là kỳ hiện tại, "Số đầu
  năm"/"Năm trước" là kỳ trước. Câu hỏi "cuối năm N" trong báo cáo năm N là cột hiện tại.
- Phân biệt các cặp dễ lẫn: "trước thuế" khác "sau thuế"; "ngắn hạn" khác "dài hạn";
  "nguyên giá" khác "giá trị còn lại"; "phải thu" khác "phải trả"; hợp nhất khác công
  ty mẹ.
- Nếu không bảng nào chứa con số cần, trả {"bang": -1, "dong": -1, "cot": -1, "so": ""}.
"""

USER = """<câu_hỏi>
{question}
</câu_hỏi>

{tables}

Ô nào là đáp án?"""


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


def contexts_of(text_path: Path) -> dict[int, str]:
    """The prose immediately above each table anchor — heading and unit line."""

    try:
        body = text_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out, previous_end = {}, 0
    for match in ANCHOR_RE.finditer(body):
        lead = body[previous_end:match.start()].strip()
        lead = re.sub(r"\s+", " ", lead)
        out[int(match.group(1))] = lead[-CONTEXT_CHARS:]
        previous_end = match.end()
    return out


def render(table_id: int, context: str, grid: list[list[str]]) -> str:
    """The table as the model sees it, with every row and column named."""

    lines = [f"[bảng {table_id}]"]
    if context:
        lines.append(f"(văn bản phía trên: …{context})")
    width = min(MAX_COLS, max((len(r) for r in grid), default=0))
    header = grid[0] if grid else []
    lines.append("  ".join(
        f"c{i}={str(header[i]).strip()[:LABEL_CHARS if i == 0 else CELL_CHARS]}"
        if i < len(header) else f"c{i}="
        for i in range(width)))
    kept = 0
    for index, row in enumerate(grid[1:]):
        if kept >= MAX_ROWS:
            lines.append("  … (còn dòng)")
            break
        if not any(str(c).strip() for c in row):
            continue
        cells = " | ".join(
            str(row[i]).strip()[:LABEL_CHARS if i == 0 else CELL_CHARS]
            if i < len(row) else "" for i in range(width))
        lines.append(f"r{index} | {cells}")
        kept += 1
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/fresh/prompts_tab.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidates", type=int, default=CANDIDATES)
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--ids-file", default="")
    # A second reading of the SAME tables in the opposite order. Temperature zero
    # makes a repeat run identical, so the only way to get an independent second
    # opinion is to change where each candidate sits: an address both orders agree on
    # is not an artefact of the model preferring whatever came first.
    parser.add_argument("--reverse", action="store_true")
    args = parser.parse_args()

    only = None
    if args.ids_file:
        only = set(json.loads((ROOT / args.ids_file).read_text(encoding="utf-8")))

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / "data" / "questions" / "questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    corpus = ROOT / "data" / "official_corpus"
    # A few documents at a time. A report is a few hundred csv files and holding every
    # one of them parsed at once exhausted memory and killed the first run after 309
    # questions, so questions are visited in document order and the cache is dropped
    # when it fills.
    grids: dict[Path, list[tuple[int, list[list[str]]]]] = {}
    contexts: dict[Path, dict[int, str]] = {}
    cache_limit = 4
    written, sizes = 0, []
    skipped = {"khong nhan ra ma": 0, "khong co nam": 0, "khong co tai lieu": 0,
               "khong con tu chi tieu": 0, "khong bang nao trung tu": 0}
    started = time.time()

    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        def document_key(item: dict) -> tuple[str, str]:
            found = sorted(resolver.resolve(item["question"]))
            years = YEAR_RE.findall(item["question"])
            return (found[0] if found else "", max(years) if years else "")

        for number, question in enumerate(sorted(questions, key=document_key),
                                          start=1):
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
            base = corpus / ticker / year
            if not base.is_dir():
                skipped["khong co tai lieu"] += 1
                continue
            wanted = metric_tokens(text, ticker, resolver.tickers.get(ticker, ""))
            if not wanted:
                skipped["khong con tu chi tieu"] += 1
                continue

            want_separate = bool(PARENT_RE.search(text))
            docs = sorted(p for p in base.iterdir() if p.is_dir())
            ordered = ([d for d in docs if ("separate" in d.name) == want_separate]
                       + [d for d in docs if ("separate" in d.name) != want_separate])

            scored = []
            for doc_dir in ordered[:2]:
                tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
                if not tables_dir.is_dir():
                    continue
                if tables_dir not in grids:
                    if len(grids) >= cache_limit:
                        grids.clear()
                        contexts.clear()
                    loaded = []
                    for path in sorted(tables_dir.glob("table_*.csv"),
                                       key=lambda p: int(p.stem.split("_")[-1])):
                        try:
                            with path.open(encoding="utf-8-sig", newline="") as file:
                                loaded.append((int(path.stem.split("_")[-1]),
                                               list(csv_mod.reader(file))))
                        except OSError:
                            continue
                    grids[tables_dir] = loaded
                    contexts[tables_dir] = contexts_of(
                        doc_dir / f"{doc_dir.name}_extracted.txt")
                for table_id, grid in grids[tables_dir]:
                    if not grid:
                        continue
                    context = contexts[tables_dir].get(table_id, "")
                    words = set(fold(" ".join(
                        " ".join(str(c) for c in row) for row in grid[:40])).split())
                    inside = len(wanted & words)
                    around = len(wanted & set(fold(context).split()))
                    # A note heading names the metric once, while a table repeats
                    # generic words, so a hit in the prose above counts for more.
                    score = inside + 1.5 * around
                    if score:
                        scored.append((score, doc_dir.name, table_id, context, grid))
                if scored:
                    break
            if not scored:
                skipped["khong bang nao trung tu"] += 1
                continue

            scored.sort(key=lambda item: -item[0])
            blocks, used, spent = [], [], 0
            for _score, doc_name, table_id, context, grid in scored:
                block = render(table_id, context, grid)
                if spent + len(block) > args.budget and blocks:
                    continue
                blocks.append(block)
                used.append({"doc": doc_name, "table_id": table_id})
                spent += len(block)
                if len(blocks) >= args.candidates:
                    break

            body = "\n\n".join(reversed(blocks) if args.reverse else blocks)
            handle.write(json.dumps({
                "id": question["id"],
                "system": SYSTEM,
                "user": USER.format(question=text, tables=body),
                "meta": {"question": text, "ticker": ticker, "year": year,
                         "doc": used[0]["doc"], "tables": used,
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
        print(f"  ky tu: p50={sizes[len(sizes) // 2]} max={sizes[-1]}")
        print(f"  uoc token: p50={sizes[len(sizes) // 2] / 2.5:.0f} "
              f"max={sizes[-1] / 2.5:.0f}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
