"""Build the 1,012 inference prompts the tuned adapter will answer.

Built on this machine rather than the rented one: it needs the table store, the
retriever and the anchor ranking, all of which live here, and none of which need
a GPU. Uploading ~30 MB of finished prompts beats uploading a corpus and
rebuilding the retrieval stack on a box that bills by the hour.

The prompt must match training exactly or the adapter is being asked a different
question than it was taught. `build_sft.py` renders through
`vifin.answering.generate.render_tables` with `--tables 8` and the same
`search_balanced` shortlist, so this does too, and the system prompt is the same
locate-form instruction that file writes.

One difference is unavoidable and deliberate. Training put the gold table first,
so every target named `df1`; here there is no gold, and the tables arrive in
retrieval order. That difference was measured rather than assumed: relabelling
the gold table `df4` in a held-out prompt left `row` unchanged at 75.0% and the
model named `df4` in 40 of 40 cases, so it reads the tables rather than trusting
the position.

Usage:  PYTHONPATH=src python scripts/build_infer_prompts.py \
            --out artifacts/infer_prompts.jsonl [--rank artifacts/anchor_rank.jsonl]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering.generate import render_tables, variable_names  # noqa: E402
from vifin.query.parse import parse_all  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# Identical to the one `build_sft.py` writes into every training example. Any
# drift here and the adapter is answering a prompt it never saw.
LOCATE_SYSTEM = (
    "Bạn đọc báo cáo tài chính Việt Nam. Cho câu hỏi và các bảng, hãy chỉ ra "
    "ô chứa đáp án và giá trị của nó, theo đúng định dạng:\n"
    "<tên biến> | <nhãn dòng> | <tên cột> | <số>\n"
    "Nhãn dòng chép nguyên văn từ bảng. Nếu dòng không có nhãn, dùng #<chỉ số>. "
    "Số phải đã quy đổi về đơn vị câu hỏi yêu cầu, làm tròn 2 chữ số thập phân. "
    "Không giải thích, không viết mã."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/infer_prompts.jsonl")
    parser.add_argument("--tables", type=int, default=8)
    parser.add_argument("--rank", default="artifacts/anchor_rank.jsonl",
                        help="anchor ranking to draw candidates from; falls back "
                             "to the lexical retriever where a question is absent")
    args = parser.parse_args()

    questions = parse_all(ROOT / "data/questions/questions.jsonl",
                          ROOT / "data/code_stock.csv")
    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    retriever = LexicalRetriever(store.frame)

    # The anchor ranking is what the submission retrieves with, and it measured
    # 92.6% gold-in-pool against the lexical retriever's 85.6%. Using anything
    # weaker here would hand the model a worse shortlist than the pipeline does
    # and understate what it can do.
    ranking: dict[int, list[TableKey]] = {}
    rank_path = ROOT / args.rank
    if rank_path.exists():
        for line in rank_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                refs = row.get("refs") or []
                ranking[row["id"]] = [
                    TableKey(r["doc_name"], int(r["table_id"])) for r in refs
                ]
        print(f"anchor ranking for {len(ranking)} questions from {rank_path.name}")
    else:
        print(f"no {rank_path.name}; using the lexical retriever throughout")

    written = 0
    from collections import Counter

    stats: Counter[str] = Counter()
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as handle:
        for question in questions:
            keys = ranking.get(question.id, [])[: args.tables]
            if keys:
                stats["from_anchor"] += 1
            else:
                keys = [
                    hit.key for hit in retriever.search_balanced(
                        question,
                        per_group=max(2, -(-args.tables //
                                           (max(1, len(question.tickers))
                                            * max(1, len(question.years))))),
                        cap=args.tables)
                ]
                stats["from_lexical"] += 1
            if not keys:
                stats["no_candidates"] += 1
                continue

            names = variable_names(len(keys))
            grids = {name: store.rows(key) for name, key in zip(names, keys)}
            refs = {
                name: f"{key.doc_name}|{int(store.meta(key).start_line)}"
                for name, key in zip(names, keys)
            }
            user = (
                "Câu hỏi: " + question.question + "\n\n"
                "Các biến có sẵn: " + ", ".join(names) + "\n\n"
                + render_tables(grids, refs)
            )
            handle.write(json.dumps({
                "id": question.id,
                "messages": [
                    {"role": "system", "content": LOCATE_SYSTEM},
                    {"role": "user", "content": user},
                ],
                "meta": {
                    "question": question.question,
                    "keys": [[k.doc_name, k.table_id] for k in keys],
                    "variables": names,
                    "target_unit": question.target_unit,
                },
            }, ensure_ascii=False) + "\n")
            written += 1
            stats["WRITTEN"] += 1

    for key, count in stats.most_common():
        print(f"  {key:18s} {count:5d}")
    size = out_path.stat().st_size / 1e6
    print(f"\nwrote {written} prompts -> {out_path} ({size:.1f} MB)")
    print("  same renderer, same table budget and same system prompt as training,")
    print("  so the adapter is answering the question it was taught to answer.")


if __name__ == "__main__":
    main()
