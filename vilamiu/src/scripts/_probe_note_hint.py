"""If the note title were known, how much would table selection improve?

Table selection is the weakest measured step: the gold table is in the eight-table
shortlist 72.5% of the time and our machinery then picks it 37.9% of those, 27.5% end
to end. Everything downstream inherits that.

Vietnamese statements are organised into numbered notes -- "9. CHO VAY KHACH HANG",
"19. PHAT HANH GIAY TO CO GIA" -- and the store keeps that heading as each table's
caption. So a two-stage design where the first stage reads the question and names the
note it belongs to would turn table selection from a similarity contest into a lookup.

Four conditions, and the gap between the last two is the whole question:

  now      the metric against row labels, which is what the pipeline does
  oracle   the gold table's own caption matched against candidate captions. An
           upper bound WITH LEAKAGE: the gold caption matches itself, so this says
           only that captions discriminate, not that they can be predicted.
  question the question's own metric against candidate captions -- no model, no
           oracle, exactly what a submission already has.
  chained  caption to choose the table, then the row matcher inside it.

Usage:
  PYTHONPATH=src python scripts/_probe_note_hint.py --limit 800 --shortlist 8
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from vifin.answering import lookup as lookup_mod  # noqa: E402
from vifin.query.companies import CompanyRoster  # noqa: E402
from vifin.query.parse import parse_question  # noqa: E402
from vifin.retrieval.lexical import LexicalRetriever  # noqa: E402
from vifin.store import TableKey, TableStore  # noqa: E402

# Page furniture that carries no note identity. Written folded, because it is only
# ever tested against `fold()` output.
BOILERPLATE = (
    "thuyet minh nay la bo phan", "doc dong thoi voi", "ban hanh theo",
    "thong tu so", "mau b", "mau so b",
)


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
    parser.add_argument("--limit", type=int, default=800)
    parser.add_argument("--shortlist", type=int, default=8)
    args = parser.parse_args()

    store = TableStore.load(ROOT / "artifacts" / "tables.parquet")
    roster = CompanyRoster.load(ROOT / "data" / "code_stock.csv")
    retriever = LexicalRetriever(store.frame)

    # `caption` is a note heading on only 22.5% of tables; the rest are form codes,
    # company names, addresses and date lines. `build_note_titles.py` recovers the
    # real heading from the released text for 95.5% of tables, so it replaces the
    # caption entirely here rather than only filling in where the caption is empty.
    prose: dict[tuple[str, int], str] = {}
    notes_path = ROOT / "artifacts" / "table_notes.jsonl"
    if notes_path.exists():
        for line in notes_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                prose[(row["doc"], int(row["table_id"]))] = row["note"]
        print(f"{len(prose)} bảng có tiêu đề thuyết minh (table_notes)")
    else:
        prose_path = ROOT / "artifacts" / "table_prose.jsonl"
        if prose_path.exists():
            for line in prose_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    prose[(row["doc"], int(row["table_id"]))] = row["prose"][:160]
        print(f"{len(prose)} bảng có văn xuôi (dự phòng)")

    tally: collections.Counter[str] = collections.Counter()
    seen = 0

    for line in (ROOT / args.gold).read_text(encoding="utf-8").splitlines():
        if not line.strip() or seen >= args.limit:
            continue
        record = json.loads(line)
        refs = record.get("relevant_tables") or []
        if len(refs) != 1:
            continue
        doc, _, table_id = refs[0].rpartition("|table_")
        if not table_id.isdigit():
            continue
        gold_key = TableKey(doc, int(table_id))
        gold_caption = str(getattr(store.meta(gold_key), "caption", ""))
        parsed = parse_question(record.get("id", 0), record["question"], roster)
        groups = max(1, len(parsed.tickers)) * max(1, len(parsed.years))
        per_group = max(2, -(-args.shortlist // groups))
        keys = [hit.key for hit in retriever.search_balanced(
            parsed, per_group=per_group, cap=args.shortlist)]
        if not keys or gold_key not in keys:
            continue
        seen += 1
        if not useful(gold_caption):
            tally["gold caption is boilerplate"] += 1

        metric = lookup_mod.extract_metric(record["question"])
        # 24.1% of captions are the page footer rather than the note heading. The
        # prose index built by `scripts/build_prose_index.py` reads the heading out
        # of the released text and drops that furniture, so where the caption is
        # useless the prose stands in for it.
        captions = {}
        for key in keys:
            heading = prose.get((key.doc_name, key.table_id))
            if not heading:
                heading = str(getattr(store.meta(key), "caption", ""))
            captions[key] = heading
        gold_caption = prose.get((gold_key.doc_name, gold_key.table_id),
                                 gold_caption)

        best_now, best_score = None, -1.0
        for key in keys:
            grid = store.rows(key)
            if not grid:
                continue
            found = lookup_mod.find(grid, parsed, captions[key][:60])
            if found is not None and found.score > best_score:
                best_now, best_score = key, found.score
        tally["now: right table"] += int(best_now == gold_key)

        best_cap, best_cap_score = None, -1.0
        for key in keys:
            value = overlap(gold_caption, captions[key])
            if value > best_cap_score:
                best_cap, best_cap_score = key, value
        tally["oracle caption: right table"] += int(best_cap == gold_key)

        best_q, best_q_score = None, -1.0
        for key in keys:
            if not useful(captions[key]):
                continue
            value = overlap(metric, captions[key])
            if value > best_q_score:
                best_q, best_q_score = key, value
        tally["question->caption: right table"] += int(best_q == gold_key)
        if best_q_score <= 0:
            tally["  no caption matched the metric"] += 1

        grid = store.rows(best_q) if best_q else None
        if grid is not None:
            found = lookup_mod.find(grid, parsed, captions[best_q][:60])
            tally["chained: candidate found"] += int(found is not None)

    print(f"{seen} questions with the gold table inside shortlist {args.shortlist}\n")
    for name in ("now: right table", "oracle caption: right table",
                 "question->caption: right table",
                 "  no caption matched the metric",
                 "chained: candidate found", "gold caption is boilerplate"):
        count = tally.get(name, 0)
        print(f"  {name:34s} {count:5d}  {count / max(seen, 1):6.1%}")


if __name__ == "__main__":
    main()
