"""Harness to run the official BTC compiler against our parquet-derived corpus.

Reuses the official `vifinqa` submodules (cube builder, recipes, finalize) by
stubbing the top-level package so `vifinqa/__init__.py` (which pulls the full
CLI stack) never executes. Table CSVs are materialized from our
`artifacts/tables.parquet` into a mirrored tree that the official
`DocumentRef`/`load_table`/`parse_statement_table` stack can read.
"""

from __future__ import annotations

import csv
import io
import json
import re
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_SRC = ROOT / "vifinqa-official" / "src"
DATA_ROOT = ROOT / "data" / "financial_statements"
PARQUET = ROOT / "artifacts" / "tables.parquet"
CORPUS_DIR = ROOT / "artifacts" / "official_corpus"

_sys_path_ready = False
_vifinqa_ready = False


def ensure_vifinqa_importable() -> None:
    global _sys_path_ready, _vifinqa_ready
    if _vifinqa_ready:
        return
    if not _sys_path_ready:
        if str(OFFICIAL_SRC) not in sys.path:
            sys.path.insert(0, str(OFFICIAL_SRC))
        _sys_path_ready = True
    if "vifinqa" not in sys.modules:
        stub = types.ModuleType("vifinqa")
        stub.__path__ = [str(OFFICIAL_SRC / "vifinqa")]
        stub.__package__ = "vifinqa"
        sys.modules["vifinqa"] = stub
    _vifinqa_ready = True


_MARKER = CORPUS_DIR / "done.marker"


def corpus_is_materialized() -> bool:
    return _MARKER.exists()


def materialize_corpus(force: bool = False) -> dict[str, dict]:
    """Write one CSV per table row in parquet; return {doc_name: info}."""
    if not force and corpus_is_materialized():
        return build_doc_index_from_disk()
    import pandas as pd

    frame = pd.read_parquet(PARQUET)
    docs: dict[str, dict] = {}
    for row in frame.itertuples(index=False):
        doc_name = row.doc_name
        info = docs.get(doc_name)
        if info is None:
            ticker, year = row.ticker, str(row.year)
            text_path = DATA_ROOT / ticker / year / doc_name / f"{doc_name}_extracted.txt"
            info = {
                "ticker": ticker,
                "year": year,
                "text_path": text_path,
                "tables_dir": CORPUS_DIR / doc_name / "tables",
                "table_ids": [],
                "table_file": {},
            }
            docs[doc_name] = info
        table_id = int(row.table_id)
        if table_id not in info["table_file"]:
            csv_path = info["tables_dir"] / f"table_{table_id}.csv"
            if not csv_path.exists():
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                with csv_path.open("w", encoding="utf-8", newline="") as f:
                    writer = csv.writer(f, lineterminator="\n")
                    writer.writerows(json.loads(row.rows_json))
            info["table_file"][table_id] = csv_path
            info["table_ids"].append(table_id)
    for info in docs.values():
        info["table_ids"] = sorted(info["table_ids"])
    _MARKER.parent.mkdir(parents=True, exist_ok=True)
    _MARKER.write_text("done", encoding="utf-8")
    return docs


def build_doc_index_from_disk() -> dict[str, dict]:
    """Rebuild the {doc_name: info} index from the materialized corpus dir."""
    docs: dict[str, dict] = {}
    for doc_dir in sorted(CORPUS_DIR.iterdir()):
        if not doc_dir.is_dir():
            continue
        tables_dir = doc_dir / "tables"
        if not tables_dir.is_dir():
            continue
        table_ids = sorted(
            int(p.stem.split("_")[-1])
            for p in tables_dir.glob("table_*.csv")
            if p.stem.startswith("table_") and p.stem.split("_")[-1].isdigit()
        )
        if not table_ids:
            continue
        doc_name = doc_dir.name
        match = re.match(r"^(.+)_financial_statements_(\d{4})_", doc_name)
        ticker, year = (match.group(1), match.group(2)) if match else ("", "")
        text_path = None
        if match:
            text_path = (
                DATA_ROOT / ticker / year / doc_name / f"{doc_name}_extracted.txt"
            )
        docs[doc_name] = {
            "ticker": ticker,
            "year": year,
            "text_path": text_path,
            "tables_dir": tables_dir,
            "table_ids": table_ids,
            "table_file": {
                tid: tables_dir / f"table_{tid}.csv" for tid in table_ids
            },
        }
    return docs


def build_docs() -> list:
    """Return list of official DocumentRef built from the materialized corpus."""
    ensure_vifinqa_importable()
    from vifinqa.common.corpus.catalog import DocumentRef

    docs_by_name = materialize_corpus()
    docs: list[DocumentRef] = []
    for doc_name, info in docs_by_name.items():
        docs.append(
            DocumentRef(
                ticker=info["ticker"],
                year=info["year"],
                doc_name=doc_name,
                doc_dir=CORPUS_DIR / doc_name,
                text_path=info["text_path"],
                tables_dir=info["tables_dir"],
                table_ids=tuple(info["table_ids"]),
            )
        )
    return docs


def build_docs_by_name(docs) -> dict[str, object]:
    return {d.doc_name: d for d in docs}


def load_cube():
    """Build the official DictCube from our materialized corpus."""
    ensure_vifinqa_importable()
    docs = build_docs()
    from vifinqa.generation.panel.builder import build_cube

    cube = build_cube(docs)
    return cube, docs


def load_company_meta():
    ensure_vifinqa_importable()
    from vifinqa.common.corpus.company_meta import load_company_meta

    path = ROOT / "data" / "code_stock.csv"
    meta = load_company_meta(path)
    # Our code_stock.csv carries no industry columns; leave them blank.
    return meta


def finalize(graph, terminal_metric_key, docs):
    """Run the official deterministic finalize (no LLM audit) on a graph."""
    ensure_vifinqa_importable()
    from vifinqa.generation.hard.recipe.finalize import finalize_validated_candidate

    table_ref_to_path = {
        f"{d.doc_name}|table_{tid}": d.table_csv_path(tid)
        for d in docs
        for tid in d.table_ids
    }
    graph, trace, compiled, formatted_answer, audit_stats = finalize_validated_candidate(
        validated_graph=graph,
        terminal_metric_key=terminal_metric_key,
        table_ref_to_path=table_ref_to_path,
        audit_stats=None,
    )
    return graph, trace, compiled, formatted_answer
