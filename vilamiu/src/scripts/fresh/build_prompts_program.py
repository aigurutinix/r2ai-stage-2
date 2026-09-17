"""Tables from every company and year a question names, for the model to write code over.

Picking one cell caps the score at roughly half the exam. Measured on the actual
questions: 372 ask for a single figure, while 437 ask for a difference, an average over
several companies, a ratio, a count, or which year was highest — and a one-cell answer to
those is wrong however well the cell is read. Nine of ten verified addresses from the
single-cell reader landed on exactly those questions.

So this pass hands the model tables from EVERY company and year the question names and
asks for a program, which is the organisers' own answering shape: their
`prompts/answering/program_system.txt` is used verbatim, because it carries three rules
worth more than anything I would write in its place —

  each table's unit is determined separately, then the final value is converted to the
  unit the question asks for;
  a difference with no direction stated is a non-negative magnitude;
  only the final value is rounded, to two decimals.

Retrieval widens with the number of companies rather than the context: eight companies
get one table each, one company gets three, so the prompt stays inside the window
whatever shape the question has.

Usage:
  python scripts/fresh/build_prompts_program.py --out artifacts/fresh/prompts_prog.jsonl
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
CONTEXT_CHARS = 260
MAX_ROWS = 26
MAX_COLS = 6
LABEL_CHARS = 58
CELL_CHARS = 22
BUDGET = 15000
MAX_TABLES = 9
# Named rather than written inline, because writing "\n\n" through a shell heredoc has
# twice put a real newline into this file and broken it.
BLOCK_GAP = "\n\n"

STOP = frozenset({"cua", "cong", "ty", "nam", "cuoi", "dau", "bao", "nhieu", "la",
                  "trong", "cho", "tai", "theo", "phan", "tram", "ctcp", "tong",
                  "gia", "tri", "muc", "khoan", "so", "du", "dong", "trieu", "ty"})

# The organisers' prompt, plus the one rule it does not need and this does. Asked
# without it, Qwen3-14B answered `result = 2807566671231 / 100000000000` — it read the
# figure off the rendered table and wrote it into the code as a literal. That is a
# constant assignment, which the private round rejects on manual review, and it also
# throws away the only self-check a program gives: code that reads the frame either
# addresses a real cell or fails.
NO_CONSTANTS = """
<no_constants>
- Mọi con số phải được ĐỌC TỪ DataFrame. Tuyệt đối không viết vào mã một con số copy
  từ bảng: `result = 2807566671231 / 1e9` là SAI và sẽ bị loại bỏ.
- Mỗi giá trị phải đến từ một biểu thức đọc ô, ví dụ `dfs["<table_ref>"].iloc[12, 3]`,
  rồi mới parse chuỗi đó thành số.
- Hằng số duy nhất được phép là hệ số quy đổi đơn vị (1000, 1e6, 1e9) và 100 cho phần
  trăm.
- Mã không tham chiếu `dfs[...]` sẽ bị loại bỏ hoàn toàn.
</no_constants>
"""

SYSTEM = ((ROOT / "vifinqa-official" / "prompts" / "answering" /
           "program_system.txt").read_text(encoding="utf-8").rstrip()
          + "\n" + NO_CONSTANTS)

USER = """<câu_hỏi>
{question}
</câu_hỏi>

Các bảng có sẵn trong `dfs` (khóa là chuỗi in trong `table_ref`). Dòng được đánh `r0`,
`r1`, ... tương ứng `df.iloc[0]`, `df.iloc[1]`, ...; cột được đánh `c0`, `c1`, ...
tương ứng `df.iloc[:, 0]`, `df.iloc[:, 1]`, ...

{tables}

Viết chương trình gán `result`."""


def fold(text: str) -> str:
    text = str(text).replace("đ", "d").replace("Đ", "D")
    flat = "".join(c for c in unicodedata.normalize("NFD", text)
                   if unicodedata.category(c) != "Mn").casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", flat)).strip()


def metric_tokens(text: str, tickers: set[str], names: list[str]) -> set[str]:
    probe = text
    for ticker in tickers:
        probe = re.sub(rf"\b{re.escape(ticker)}\b", " ", probe)
    probe = re.sub(r"\([^)]*\)", " ", probe)
    probe = YEAR_RE.sub(" ", probe)
    for chunk in tuple(names) + STRIP_CHUNKS:
        if chunk:
            probe = re.sub(re.escape(chunk), " ", probe, flags=re.I)
    return {t for t in fold(probe).split() if len(t) > 2 and t not in STOP}


def contexts_of(text_path: Path) -> dict[int, str]:
    try:
        body = text_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out, previous_end = {}, 0
    for match in ANCHOR_RE.finditer(body):
        lead = re.sub(r"\s+", " ", body[previous_end:match.start()].strip())
        out[int(match.group(1))] = lead[-CONTEXT_CHARS:]
        previous_end = match.end()
    return out


def render(ref: str, context: str, grid: list[list[str]]) -> str:
    lines = [f'table_ref: "{ref}"']
    if context:
        lines.append(f"(văn bản phía trên: …{context})")
    width = min(MAX_COLS, max((len(r) for r in grid), default=0))
    header = grid[0] if grid else []
    lines.append("  ".join(
        f"c{i}={str(header[i]).strip()[:LABEL_CHARS if i == 0 else CELL_CHARS]}"
        if i < len(header) else f"c{i}=" for i in range(width)))
    kept = 0
    for index, row in enumerate(grid[1:]):
        if kept >= MAX_ROWS:
            lines.append("  … (còn dòng)")
            break
        if not any(str(c).strip() for c in row):
            continue
        lines.append(f"r{index} | " + " | ".join(
            str(row[i]).strip()[:LABEL_CHARS if i == 0 else CELL_CHARS]
            if i < len(row) else "" for i in range(width)))
        kept += 1
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/fresh/prompts_prog.jsonl")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids-file", default="")
    parser.add_argument("--budget", type=int, default=BUDGET)
    # Keep one slot per company-year for a main statement. Measured on the candidates
    # this retrieval produces: of 57 single-year ratio questions, only 15 were offered
    # any main statement at all, and a ratio of revenue to profit cannot be computed
    # from note tables. Word overlap alone does not find them because a statement
    # repeats generic words while a note names the metric outright.
    parser.add_argument("--reserve-statement", action="store_true")
    # Candidates chosen elsewhere. When a dense retriever has already ranked the tables,
    # this file replaces the word-overlap scoring entirely and the builder only renders
    # what it is given — which is the point: the organisers measure 47% recall@10 for
    # lexical matching against 81% for embeddings plus a reranker.
    parser.add_argument("--topk-file", default="")
    # How many candidates to offer. Measured: 9 tables gives 53% on the offline set, 5
    # gives 50%, 3 gives 46% — monotonic, so the losses from a distractor are smaller
    # than the losses from a missing table. The organisers' 23-point oracle-vs-k=10 gap
    # does not transfer, because their oracle hands over the gold table while a shorter
    # shortlist here still has to find it.
    parser.add_argument("--max-tables", type=int, default=MAX_TABLES)
    # Any file in the exam's `{id, question}` shape. Lets the same renderer build prompts
    # for synthetic atomic questions — one (company, year, indicator) each — so a
    # multi-company question can be answered as several single-cell reads composed by code.
    parser.add_argument("--questions", default="data/questions/questions.jsonl")
    args = parser.parse_args()

    only = None
    if args.ids_file:
        only = set(json.loads((ROOT / args.ids_file).read_text(encoding="utf-8")))

    given: dict[int, list[dict]] = {}
    if args.topk_file:
        given = {int(k): v for k, v in json.loads(
            (ROOT / args.topk_file).read_text(encoding="utf-8")).items()}

    resolver = TickerResolver()
    questions = [json.loads(line) for line in
                 (ROOT / args.questions).read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    if args.limit:
        questions = questions[:args.limit]

    corpus = ROOT / "data" / "official_corpus"
    # Only what scoring needs is cached: one folded word set and one context string per
    # table. A question about eight companies touches eight documents at once, so
    # caching parsed csv content instead meant every document was re-read for every
    # such question and the build crawled. Full rows are read for the handful of tables
    # that survive scoring.
    summaries: dict[Path, list[tuple[int, set[str], str, Path]]] = {}
    doc_contexts: dict[str, dict[int, str]] = {}
    written, sizes, table_counts = 0, [], []
    skipped = {"khong nhan ra ma": 0, "khong co nam": 0, "khong co tai lieu": 0,
               "khong con tu chi tieu": 0, "khong bang nao trung tu": 0}
    started = time.time()

    def sort_key(item: dict) -> tuple[str, str]:
        found = sorted(resolver.resolve(item["question"]))
        years = YEAR_RE.findall(item["question"])
        return (found[0] if found else "", max(years) if years else "")

    with (ROOT / args.out).open("w", encoding="utf-8") as handle:
        for number, question in enumerate(sorted(questions, key=sort_key), start=1):
            if only is not None and question["id"] not in only:
                continue
            text = question["question"]
            tickers = resolver.resolve(text)
            years = sorted(set(YEAR_RE.findall(text)))
            if not tickers:
                skipped["khong nhan ra ma"] += 1
                continue
            if not years:
                skipped["khong co nam"] += 1
                continue
            names = [resolver.tickers.get(t, "") for t in tickers]
            wanted = metric_tokens(text, tickers, names)
            if not wanted:
                skipped["khong con tu chi tieu"] += 1
                continue

            want_separate = bool(PARENT_RE.search(text))

            if given:
                blocks, refs, spent = [], [], 0
                for ref in given.get(question["id"], []):
                    path = (corpus / ref["ticker"] / ref["year"] / ref["doc"] /
                            f"{ref['doc']}_extracted_tables"
                            / f"table_{ref['table_id']}.csv")
                    try:
                        with path.open(encoding="utf-8-sig", newline="") as file:
                            grid = list(csv_mod.reader(file))
                    except OSError:
                        continue
                    if not grid:
                        continue
                    if doc_contexts.get(ref["doc"]) is None:
                        doc_contexts[ref["doc"]] = contexts_of(
                            corpus / ref["ticker"] / ref["year"] / ref["doc"] /
                            f"{ref['doc']}_extracted.txt")
                    context = doc_contexts[ref["doc"]].get(ref["table_id"], "")
                    block = render(ref["ref"], context, grid)
                    if spent + len(block) > args.budget and blocks:
                        break
                    blocks.append(block)
                    refs.append({"ref": ref["ref"], "doc": ref["doc"],
                                 "table_id": ref["table_id"],
                                 "ticker": ref["ticker"], "year": ref["year"]})
                    spent += len(block)
                if not blocks:
                    skipped["khong bang nao trung tu"] += 1
                    continue
                handle.write(json.dumps({
                    "id": question["id"], "system": SYSTEM,
                    "user": USER.format(question=text,
                                        tables=BLOCK_GAP.join(blocks)),
                    "meta": {"question": text, "tickers": sorted(tickers),
                             "years": years, "refs": refs,
                             "scope": ("separate" if want_separate
                                       else "consolidated")},
                }, ensure_ascii=False) + "\n")
                written += 1
                sizes.append(spent)
                table_counts.append(len(blocks))
                continue

            # Share the table budget across the pairs the question names, so a question
            # about eight companies still fits and one about a single company gets depth.
            pairs = [(t, y) for t in sorted(tickers) for y in years]
            per_pair = max(1, args.max_tables // max(1, len(pairs)))

            blocks, refs, spent = [], [], 0
            for ticker, year in pairs:
                base = corpus / ticker / year
                if not base.is_dir():
                    continue
                docs = sorted(p for p in base.iterdir() if p.is_dir())
                ordered = ([d for d in docs
                            if ("separate" in d.name) == want_separate]
                           + [d for d in docs
                              if ("separate" in d.name) != want_separate])
                scored, statements = [], []
                for doc_dir in ordered[:1]:
                    tables_dir = doc_dir / f"{doc_dir.name}_extracted_tables"
                    if not tables_dir.is_dir():
                        continue
                    if tables_dir not in summaries:
                        if len(summaries) >= 120:
                            summaries.clear()
                        anchors = contexts_of(
                            doc_dir / f"{doc_dir.name}_extracted.txt")
                        entries = []
                        for path in sorted(tables_dir.glob("table_*.csv"),
                                           key=lambda p: int(p.stem.split("_")[-1])):
                            table_id = int(path.stem.split("_")[-1])
                            try:
                                with path.open(encoding="utf-8-sig",
                                               newline="") as file:
                                    head = [row for _, row in
                                            zip(range(40), csv_mod.reader(file))]
                            except OSError:
                                continue
                            words = set(fold(" ".join(
                                " ".join(str(c) for c in row)
                                for row in head)).split())
                            entries.append((table_id, words,
                                            anchors.get(table_id, ""), path))
                        summaries[tables_dir] = entries
                    for table_id, words, context, path in summaries[tables_dir]:
                        score = len(wanted & words) + 1.5 * len(
                            wanted & set(fold(context).split()))
                        # "CHỈ TIÊU" beside "Mã số" is the signature of a balance
                        # sheet, an income statement or a cash-flow statement.
                        if {"ma", "so", "chi", "tieu"} <= words:
                            statements.append((score, doc_dir.name, table_id,
                                               context, path))
                        if score:
                            scored.append((score, doc_dir.name, table_id,
                                           context, path))
                scored.sort(key=lambda item: -item[0])
                picks = scored[:per_pair]
                if args.reserve_statement and statements:
                    statements.sort(key=lambda item: -item[0])
                    if not any(p[2] == statements[0][2] and p[1] == statements[0][1]
                               for p in picks):
                        picks = (picks[:per_pair - 1] if per_pair > 1 else [])                             + [statements[0]]
                for _score, doc_name, table_id, context, path in picks:
                    try:
                        with path.open(encoding="utf-8-sig", newline="") as file:
                            grid = list(csv_mod.reader(file))
                    except OSError:
                        continue
                    ref = f"{doc_name}|table_{table_id}"
                    block = render(ref, context, grid)
                    if spent + len(block) > args.budget and blocks:
                        break
                    blocks.append(block)
                    refs.append({"ref": ref, "doc": doc_name,
                                 "table_id": table_id, "ticker": ticker,
                                 "year": year})
                    spent += len(block)
                if len(blocks) >= args.max_tables:
                    break
            if not blocks:
                skipped["khong bang nao trung tu"] += 1
                continue

            handle.write(json.dumps({
                "id": question["id"],
                "system": SYSTEM,
                "user": USER.format(question=text, tables="\n\n".join(blocks)),
                "meta": {"question": text, "tickers": sorted(tickers),
                         "years": years, "refs": refs,
                         "scope": "separate" if want_separate else "consolidated"},
            }, ensure_ascii=False) + "\n")
            written += 1
            sizes.append(spent)
            table_counts.append(len(blocks))
            if number % 200 == 0:
                print(f"  {number}/{len(questions)}  {time.time() - started:.0f}s",
                      flush=True)

    print(f"\n{written} prompt")
    for name, count in skipped.items():
        if count:
            print(f"  bo qua — {name}: {count}")
    if sizes:
        sizes.sort()
        table_counts.sort()
        print(f"  ky tu: p50={sizes[len(sizes) // 2]} p90="
              f"{sizes[int(len(sizes) * 0.9)]} max={sizes[-1]}")
        print(f"  uoc token: p50={sizes[len(sizes) // 2] / 2.5:.0f} "
              f"max={sizes[-1] / 2.5:.0f}")
        print(f"  so bang: p50={table_counts[len(table_counts) // 2]} "
              f"max={table_counts[-1]}")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()
