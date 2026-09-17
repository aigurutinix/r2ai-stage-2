"""Flag filename years contradicted by multiple face-statement headings.

This reports evidence only. It does not relabel source documents or change
answers. Legal enactment dates and audit-signature dates are not period votes.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from road2ai_vifinqa.corpus import Corpus
from road2ai_vifinqa.text import fold_text

HEADING = re.compile(r"bang can doi ke toan|bao cao tinh hinh tai chinh|bao cao ket qua hoat dong kinh doanh|bao cao luu chuyen tien te")
DATE = re.compile(r"(?:ngay )?\b(?:0?[1-9]|[12][0-9]|3[01]) thang (?:0?[1-9]|1[0-2]) nam (20\d{2})\b|(?:ngay )?\b(?:[12][0-9]|3[01]) (?:0?[1-9]|1[0-2]) (20\d{2})\b")


def heading_year(context: str) -> int | None:
    text = fold_text(context)
    headings = list(HEADING.finditer(text))
    if not headings:
        return None
    tail = text[headings[-1].end():]
    tail = re.split(r"mau (?:b|so)|ban hanh theo|thong tu|quyet dinh", tail, maxsplit=1)[0]
    dates = list(DATE.finditer(tail))
    if not dates:
        return None
    years = {int(match.group(1) or match.group(2)) for match in dates}
    return next(iter(years)) if len(years) == 1 else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zip", type=Path, default=ROOT / "submission_vn53.zip")
    args = parser.parse_args()
    votes = {}
    with Corpus() as corpus, zipfile.ZipFile(args.zip) as archive:
        records = json.loads(archive.read("submission.json"))
        used = {d for r in records for d in r["relevant_docs"]}
        for row in corpus.conn.execute("SELECT doc_id,table_id,context FROM tables ORDER BY doc_id,table_id"):
            if row["doc_id"] not in used:
                continue
            year = heading_year(row["context"])
            if year is not None:
                votes.setdefault(row["doc_id"], []).append({"table": row["table_id"], "year": year, "context": row["context"]})
        flags = []
        for doc, statements in sorted(votes.items()):
            counts = Counter(s["year"] for s in statements)
            declared = corpus._doc_by_id[doc].report_year
            year, count = counts.most_common(1)[0]
            if year != declared and count >= 2 and count / len(statements) >= .8:
                flags.append({"document": doc, "filename_year": declared, "statement_year": year,
                              "votes": dict(counts), "headings": statements,
                              "question_ids": [r["id"] for r in records if doc in r["relevant_docs"]]})
        output = {"documents_with_period_headings": len(votes), "flags": flags}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        for f in flags:
            print(f["document"], "printed year", f["statement_year"], "questions", f["question_ids"])


if __name__ == "__main__":
    main()
