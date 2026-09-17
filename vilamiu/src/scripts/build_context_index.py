"""Index the prose around each table, not just the table's own labels.

Our retriever indexes caption, unit line, header row and first-column labels
(`lexical.py:_table_tokens`). The organisers' `table_retrieval_text` indexes the
company's full name and 2,500 characters of the page the table sits on.

Their generation code says why that matters. `judge_and_maybe_rewrite` rejects
any generated question that copies row or column labels verbatim, so the wording
a question uses is, by construction, wording that does *not* appear in the
table. It appears in the narrative around it. We have been indexing the one part
of the document the questions were written to avoid.

The public corpus embeds tables inline as `<table>...</table>` rather than as the
`[table_N](path)` anchors the organisers' `parse_document` expects, so the page
split is reimplemented here. `table_id` is the ordinal of `<table>` across the
document; that was checked against the stored `page_no` before this was written.

Usage:  PYTHONPATH=src python scripts/build_context_index.py
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MAX_CONTEXT_CHARS = 2500
MAX_LABEL_ROWS = 20
MAX_HEADER_CHARS = 1200

PAGE_RE = re.compile(r"=====\s*PAGE (\d+)\s*=====")
TABLE_BLOCK_RE = re.compile(r"<table.*?</table>", re.S)
TABLE_OPEN_RE = re.compile(r"<table")
NUMERIC_RE = re.compile(r"^\(?-?\d{1,3}(\.\d{3})*(,\d+)?\)?%?$")

# Report scope, as `detect_report_scope` derives it from the document name.
NORM_RE = re.compile(r"[^a-z0-9]")
HOP_NHAT = ("hopnhat", "consol", "bctchn")
CONG_TY_ME = ("congtyme", "rieng", "separate")


def normalize_ascii(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    stripped = stripped.replace("đ", "d").replace("Đ", "D").lower()
    return NORM_RE.sub("", stripped)


def detect_report_scope(doc_name: str) -> str | None:
    flat = normalize_ascii(doc_name)
    consolidated = any(m in flat for m in HOP_NHAT)
    separate = any(m in flat for m in CONG_TY_ME)
    if consolidated and not separate:
        return "hợp nhất"
    if separate and not consolidated:
        return "công ty mẹ"
    return None


def is_numeric_cell(cell: str) -> bool:
    cell = cell.strip()
    if not cell or cell == "-":
        return False
    return bool(NUMERIC_RE.match(cell))


def label_text(grid: list[list[str]]) -> str:
    """Columns and row labels in the organisers' shape: first three cells, 20 rows."""

    header = grid[0] if grid else []
    columns = " | ".join(str(h) for h in header if str(h).strip())
    rows = []
    for row in grid[:MAX_LABEL_ROWS]:
        cells = [str(c) for c in row[:3] if str(c).strip() and not is_numeric_cell(str(c))]
        if cells:
            rows.append(" / ".join(cells))
    return f"Cột: {columns}. Dòng: {' | '.join(rows)}"[:MAX_HEADER_CHARS]


def page_context(text_path: Path) -> dict[int, str]:
    """Narrative text of the page holding each table, keyed by table ordinal.

    The table's own markup is dropped: in the organisers' corpus the page carries
    only an anchor where the table sits, so keeping the HTML here would index a
    different thing than they index — and would drown the prose in digits.
    """

    text = text_path.read_text(encoding="utf-8")
    marks = list(PAGE_RE.finditer(text))
    context: dict[int, str] = {}
    ordinal = 0
    for i, mark in enumerate(marks):
        start = mark.end()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        content = text[start:end]
        count = len(TABLE_OPEN_RE.findall(content))
        if not count:
            continue
        prose = " ".join(TABLE_BLOCK_RE.sub(" ", content).split())[:MAX_CONTEXT_CHARS]
        for _ in range(count):
            context[ordinal] = prose
            ordinal += 1
    return context


def main() -> None:
    frame = pd.read_parquet(ROOT / "artifacts" / "tables.parquet")
    stock = pd.read_csv(ROOT / "data" / "code_stock.csv")
    company = {
        str(row[0]).strip().upper(): str(row[1]).strip()
        for row in stock.itertuples(index=False)
    }
    print(f"{len(company)} company names from code_stock.csv")

    eligible = frame[frame.eligible]
    print(f"{eligible.doc_name.nunique()} documents, {len(eligible)} eligible tables")

    out_path = ROOT / "artifacts" / "context_index.jsonl"
    started = time.time()
    written = docs_done = missing = empty = 0

    with out_path.open("w", encoding="utf-8") as handle:
        for doc_name, group in eligible.groupby("doc_name", sort=False):
            first = group.iloc[0]
            ticker, year = str(first.ticker), str(first.year)
            doc_dir = ROOT / "data" / "financial_statements" / ticker / year / doc_name
            text_path = next(iter(doc_dir.glob("*_extracted.txt")), None)
            if text_path is None:
                missing += len(group)
                continue
            context = page_context(text_path)
            docs_done += 1

            name = company.get(ticker.upper(), "")
            head = f"Công ty: {name} (mã {ticker})" if name else f"Mã: {ticker}"
            scope = detect_report_scope(doc_name)
            prefix = f"{head}, năm {year}" + (f". Phạm vi báo cáo: {scope}" if scope else "") + "."

            for row in group.itertuples(index=False):
                table_id = int(row.table_id)
                prose = context.get(table_id, "")
                if not prose:
                    empty += 1
                grid = json.loads(row.rows_json)
                text = unicodedata.normalize(
                    "NFC", f"{prefix} {label_text(grid)}. Ngữ cảnh: {prose}")
                handle.write(json.dumps(
                    {"doc_name": doc_name, "table_id": table_id, "text": text},
                    ensure_ascii=False) + "\n")
                written += 1

            if docs_done % 250 == 0:
                print(f"  {docs_done} docs, {written} tables, {time.time() - started:.0f}s")

    print(f"\n{written} tables from {docs_done} documents in {time.time() - started:.0f}s")
    print(f"documents with no extracted text: {missing} tables")
    print(f"tables whose page yielded no prose: {empty}")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()
