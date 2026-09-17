"""Dựng CATALOG toàn kho ViFinQA: mỗi bảng HTML -> 1 CSV + 1 dòng chỉ mục JSONL.

Ráp từ bộ bóc bảng (table_extract) + map mã CK->tên (code_stock.csv). Mỗi bảng có:
report_id, ticker, year, scope (hợp nhất/công ty mẹ), line (số dòng thẻ <table>),
page, csv_path, và `search_text` kiểu SIÊU-DỮ-LIỆU-TRƯỚC để BM25/dense truy hồi.

Chạy:
  python src/kingpro/corpus/build_catalog.py --limit-companies 1        # test nhanh 1 công ty
  python src/kingpro/corpus/build_catalog.py                            # cả kho
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from kingpro.corpus.table_extract import extract_tables  # noqa: E402
from kingpro.corpus.table_context import (  # noqa: E402
    row_labels as structural_row_labels,
    section_ancestors,
    table_context,
)

DATA = Path("data/financial_statements")
CODE_STOCK = Path("data/code_stock.csv")


def load_company_names() -> dict[str, str]:
    m: dict[str, str] = {}
    if CODE_STOCK.exists():
        with open(CODE_STOCK, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                ma = (row.get("Mã CK") or row.get("Ma CK") or "").strip()
                ten = (row.get("Tên công ty") or row.get("Ten cong ty") or "").strip()
                if ma:
                    m[ma] = ten
    return m


def scope_of(doc_dir_name: str) -> str:
    n = doc_dir_name.lower()
    if "consolidated" in n or "hopnhat" in n:
        return "hợp nhất"
    if "separate" in n or "congtyme" in n or "rieng" in n:
        return "công ty mẹ"
    return "không rõ"


def row_labels(df, k: int = 40) -> str:
    """Semantic row labels from the best physical label column."""
    if df is None or df.shape[1] == 0:
        return ""
    return " | ".join(structural_row_labels(df, limit=k))


def columns_text(df) -> str:
    if df is None or df.shape[1] == 0:
        return ""
    return " | ".join(str(c) for c in df.columns.tolist())


def _compact_path_atoms(value: str) -> str:
    """Deduplicate path atoms so BM25 length normalization stays stable."""

    atoms = []
    seen = set()
    for path in str(value).split("|"):
        for raw in path.split(">"):
            atom = " ".join(raw.split())
            key = atom.casefold()
            if atom and key not in seen:
                seen.add(key)
                atoms.append(atom)
    return " | ".join(atoms)


def _compact_total_labels(value: str) -> str:
    labels = []
    seen = set()
    for raw in str(value).split("|"):
        label = " ".join(raw.split(">", 1)[0].split())
        key = label.casefold()
        if label and key not in seen:
            seen.add(key)
            labels.append(label)
    return " | ".join(labels)


def build_search_text(
    ticker,
    name,
    year,
    scope,
    df,
    *,
    section_title: str = "",
    context: dict[str, str] | None = None,
) -> str:
    """Build source-only retrieval text with bounded structural ancestry."""

    context = context or table_context(df)
    parts = []
    if section_title:
        parts.append(f"Mục: {section_title}")
    if context.get("header_text"):
        parts.append(f"Header: {_compact_path_atoms(context['header_text'])}")
    labels = context.get("row_label_text") or row_labels(df)
    if labels:
        parts.append(f"Chỉ tiêu: {labels}")
    if context.get("inferred_row_text"):
        parts.append(
            f"Dòng tổng suy luận: {_compact_total_labels(context['inferred_row_text'])}"
        )
    parts.append(f"Công ty: {name} (mã {ticker}), năm {year}, phạm vi {scope}")
    parts.append(f"Cột: {columns_text(df)}")
    return ". ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("build"))
    ap.add_argument("--limit-companies", type=int, default=0)
    args = ap.parse_args()

    names = load_company_names()
    tables_dir = args.out / "tables"
    args.out.mkdir(parents=True, exist_ok=True)

    companies = sorted(p.name for p in DATA.iterdir() if p.is_dir())
    if args.limit_companies:
        companies = companies[: args.limit_companies]

    n_docs = n_tables = 0
    cat_path = args.out / "catalog.jsonl"
    with open(cat_path, "w", encoding="utf-8") as cat:
        for ticker in companies:
            name = names.get(ticker, ticker)
            for txt in sorted((DATA / ticker).glob("*/*/*_extracted.txt")):
                year = txt.parents[1].name
                scope = scope_of(txt.parent.name)
                tables = extract_tables(txt, out_dir=tables_dir)
                source_lines = txt.read_text(encoding="utf-8", errors="replace").splitlines()
                n_docs += 1
                for t in tables:
                    n_tables += 1
                    section_title = section_ancestors(source_lines, t.line)
                    context = table_context(t.df)
                    cat.write(
                        json.dumps(
                            {
                                "table_ref": t.table_ref,      # report_id|dòng
                                "report_id": t.report_id,
                                "ticker": ticker,
                                "year": year,
                                "scope": scope,
                                "line": t.line,
                                "page": t.page,
                                "n_rows": t.n_rows,
                                "n_cols": t.n_cols,
                                "csv_path": t.csv_path,
                                "search_text": build_search_text(
                                    ticker,
                                    name,
                                    year,
                                    scope,
                                    t.df,
                                    section_title=section_title,
                                    context=context,
                                ),
                                "section_title": section_title,
                                **context,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            print(f"  {ticker}: xong", flush=True)

    print(f"DONE: {n_docs} báo cáo, {n_tables} bảng -> {cat_path}")


if __name__ == "__main__":
    main()
