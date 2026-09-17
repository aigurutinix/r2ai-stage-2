"""P7 — build the submission ZIP.

Layout mandated by the rules:

    submission.zip
    ├── submission.json      exactly one .json, at the archive root
    └── data/<table>.csv     every csv_path referenced by evidence

Two details of `relevant_tables` are still unverified against the scorer, so both
are parameters rather than assumptions. The organisers' own code builds refs as
``f"{doc}|table_{id}"`` while the published rules show ``AAA_..._2015_consolidated|350``,
and nothing states whether their ids start at 0 or 1. `TableRefStyle` lets P9
submit the same predictions under each combination and read the answer off the
retrieval F2.
"""

from __future__ import annotations

import csv
import io
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from vifin.store import TableKey, TableStore


@dataclass(frozen=True, slots=True)
class TableRefStyle:
    """How to render `<doc>|<table position>`.

    The organisers confirmed the position is the **1-based line number where the
    table starts in the OCR .txt**, written bare: `VNM_..._2023_consolidated|350`.
    Our per-document table ordinal — the reading the organisers' own repo encodes
    as `doc|table_N` — scored TABLES_F2 = 0 under every prefix/offset combination.
    The knobs stay so a regression can be re-measured, but the default is settled.
    """

    prefixed: bool = False
    offset: int = 0

    def render(self, doc_name: str, position: int) -> str:
        position += self.offset
        return f"{doc_name}|table_{position}" if self.prefixed else f"{doc_name}|{position}"

    @property
    def label(self) -> str:
        return f"{'prefixed' if self.prefixed else 'line'}{self.offset:+d}"


@dataclass(slots=True)
class Prediction:
    id: int
    question: str
    answer: float
    pandas_query: str
    tables: list[TableKey] = field(default_factory=list)
    # `relevant_tables` may be broader than the evidence actually executed.
    ref_tables: list[TableKey] | None = None
    # `relevant_docs` is scored as its own list, so it need not mirror
    # `relevant_tables`. Table precision collapses with depth (TABLES_F2 0.3817
    # at 5 refs vs 0.3457 at 10) while document recall keeps improving
    # (DOCS_F2 0.8782 vs 0.9324). Declaring shallow tables and deep documents
    # takes the better number on both.
    ref_docs: list[str] | None = None
    # Evidence that is not a corpus table: the metric panel is assembled from
    # several statements, so it ships as its own CSV while `relevant_tables`
    # still cites the source tables the figures came from.
    inline_tables: list[tuple[str, list[list[str]]]] = field(default_factory=list)

    @property
    def declared(self) -> list[TableKey]:
        return self.tables if self.ref_tables is None else self.ref_tables


def _csv_name(key: TableKey) -> str:
    return f"{key.doc_name}_table_{key.table_id}.csv"


def _render_csv(rows: list[list[str]]) -> str:
    buffer = io.StringIO(newline="")
    # The scorer reads these back with `read_csv`, which infers a dtype per
    # column — see `sandbox.frame_from_rows`. CRLF would survive into cells.
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerows(rows)
    return buffer.getvalue()


def build_submission(
    predictions: list[Prediction],
    store: TableStore,
    out_zip: Path,
    style: TableRefStyle = TableRefStyle(),
) -> dict:
    """Write the ZIP and return a summary of what went into it."""

    records = []
    csv_payloads: dict[str, str] = {}

    for prediction in predictions:
        # The scorer binds each CSV to the name declared here, while the
        # organisers' own prompt tells the model to expect `df` for a single
        # table. Naming a lone table `df` satisfies both readings; anything else
        # left the program calling an undefined name and scored zero on
        # execution even when the answer itself was right.
        evidence = []
        single = len(prediction.tables) + len(prediction.inline_tables) == 1
        for position, (name, rows) in enumerate(prediction.inline_tables, start=1):
            if name not in csv_payloads:
                csv_payloads[name] = _render_csv(rows)
            variable = "df" if single else f"df{position}"
            evidence.append({"variable": variable, "csv_path": f"data/{name}"})
        offset = len(prediction.inline_tables)
        for position, key in enumerate(prediction.tables, start=offset + 1):
            name = _csv_name(key)
            if name not in csv_payloads:
                csv_payloads[name] = _render_csv(store.rows(key))
            variable = "df" if single else f"df{position}"
            evidence.append({"variable": variable, "csv_path": f"data/{name}"})

        docs: list[str] = []
        refs: list[str] = []
        for key in prediction.declared:
            if key.doc_name not in docs:
                docs.append(key.doc_name)
            ref = style.render(key.doc_name, int(store.meta(key).start_line))
            if ref not in refs:
                refs.append(ref)
        if prediction.ref_docs is not None:
            for doc_name in prediction.ref_docs:
                if doc_name not in docs:
                    docs.append(doc_name)

        records.append(
            {
                "id": int(prediction.id),
                "question": prediction.question,
                "answer": float(prediction.answer),
                "relevant_docs": docs,
                "relevant_tables": refs,
                "evidence": evidence,
                "pandas_query": prediction.pandas_query,
            }
        )

    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("submission.json", json.dumps(records, ensure_ascii=False, indent=1))
        for name, payload in csv_payloads.items():
            archive.writestr(f"data/{name}", payload)

    return {
        "zip": str(out_zip),
        "questions": len(records),
        "csv_files": len(csv_payloads),
        "style": style.label,
        "bytes": out_zip.stat().st_size,
    }
