"""Read-only access to the durable table index."""

from __future__ import annotations

import csv
import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .paths import COMPANIES_PATH, INDEX_PATH, QUESTIONS_PATH
from .text import fold_text


YEAR_RE = re.compile(r"\b(20\d{2})\b")
CODE_RE = re.compile(r"(?<![A-Z0-9])([A-Z][A-Z0-9]{1,4})(?![A-Z0-9])")
PAREN_CODE_RE = re.compile(r"\(([A-Z][A-Z0-9]{1,4})\)")
QUESTION_ALIASES = {
    "tap doan c e o": "CEO",
    "cong ty co phan tap doan c e o": "CEO",
    "hang khong vietjet": "VJC",
    "chung khoan fpt": "FTS",
    "the gioi di dong": "MWG",
    "do thi kinh bac": "KBC",
    "hoa phat": "HPG",
    "hoa sen": "HSG",
    "nam kim": "NKG",
    "masan high tech": "MSR",
    "masan consumer": "MCH",
    "masan meatlife": "MML",
    "dai duong": "OGC",
    "dau tu dich vu hoang huy": "HHS",
    "dien luc gelex": "GEE",
    "vinamilk": "VNM",
    "vincom retail": "VRE",
    "vingroup": "VIC",
    "dam phu my": "DPM",
    "dam ca mau": "DCM",
    "binh son": "BSR",
    "pvtrans": "PVT",
    "sao mai": "ASM",
    "dabaco": "DBC",
    "minh phu": "MPC",
    "dat xanh": "DXG",
    "nam long": "NLG",
    "hai phat": "HPX",
    "van phu": "VPI",
}


@dataclass(frozen=True, slots=True)
class DocumentRef:
    doc_id: str
    ticker: str
    report_year: int
    scope: str
    source_path: str
    table_count: int


@dataclass(frozen=True, slots=True)
class TableAsset:
    doc_id: str
    table_id: int
    page: int
    context: str
    rows: list[list[str]]
    statement_kind: str = "unknown"
    unit_scale: float = 1.0
    header_rows: tuple[int, ...] = ()

    @property
    def competition_ref(self) -> str:
        return f"{self.doc_id}|table_{self.table_id}"


@dataclass(frozen=True, slots=True)
class RowAsset:
    doc_id: str
    table_id: int
    row_idx: int
    cells: list[str]
    folded_text: str
    row_label: str = ""
    section: str = ""


def load_questions() -> list[dict[str, object]]:
    return [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()]


class Corpus:
    def __init__(self, path: Path = INDEX_PATH) -> None:
        self.path = path
        self.conn = sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self._documents = [DocumentRef(**dict(row)) for row in self.conn.execute("SELECT * FROM documents")]
        self._doc_by_id = {doc.doc_id: doc for doc in self._documents}
        self._by_ticker: dict[str, list[DocumentRef]] = {}
        for doc in self._documents:
            self._by_ticker.setdefault(doc.ticker, []).append(doc)
        self.tickers = frozenset(self._by_ticker)
        self.company_names = self._load_company_names()
        table_columns = {
            str(row[1]) for row in self.conn.execute("PRAGMA table_info(tables)")
        }
        row_columns = {
            str(row[1]) for row in self.conn.execute("PRAGMA table_info(rows)")
        }
        self._semantic_tables = {
            "statement_kind", "unit_scale", "header_rows_json"
        }.issubset(table_columns)
        self._semantic_rows = {"row_label", "section"}.issubset(row_columns)
        self._table_cache: dict[tuple[str, int], TableAsset] = {}

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Corpus":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _load_company_names() -> dict[str, str]:
        names: dict[str, str] = {}
        with COMPANIES_PATH.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    names[row[0].strip()] = row[1].strip()
        return names

    def infer_tickers(self, question: str) -> list[str]:
        folded_q = fold_text(question)
        parenthesized: list[tuple[int, str]] = []
        for match in PAREN_CODE_RE.finditer(question):
            code = match.group(1)
            if code in self.tickers:
                parenthesized.append((match.start(), code))
        aliased: list[tuple[int, str]] = []
        for alias, ticker in QUESTION_ALIASES.items():
            position = folded_q.find(alias)
            if position >= 0 and ticker in self.tickers:
                aliased.append((position, ticker))

        # Full official names are stronger than uppercase brands embedded in
        # those names ("CTCP Chứng khoán FPT" is FTS, not ticker FPT).
        contained = [
            (folded_q.find(fold_text(name)), ticker)
            for ticker, name in self.company_names.items()
            if len(fold_text(name)) >= 10 and fold_text(name) in folded_q
        ]
        strong = [*parenthesized, *contained] if contained else [*parenthesized, *aliased]
        if strong:
            strong.sort()
            return list(dict.fromkeys(ticker for _, ticker in strong))

        explicit: list[str] = []
        for code in CODE_RE.findall(question):
            if code in self.tickers and code not in explicit:
                explicit.append(code)
        if explicit:
            return explicit

        scored: list[tuple[float, str]] = []
        for ticker, name in self.company_names.items():
            folded_name = fold_text(name)
            if folded_name and folded_name in folded_q:
                score = 1.0
            else:
                score = SequenceMatcher(None, folded_name, folded_q).ratio()
            scored.append((score, ticker))
        scored.sort(reverse=True)
        return [scored[0][1]] if scored and scored[0][0] >= 0.42 else []

    @staticmethod
    def infer_years(question: str) -> list[int]:
        years: list[int] = []
        for value in YEAR_RE.findall(question):
            year = int(value)
            if year not in years:
                years.append(year)
        return years

    @staticmethod
    def infer_scope(question: str) -> str | None:
        folded = fold_text(question)
        if any(marker in folded for marker in ("cong ty me", "bao cao rieng", "co so cong ty me")):
            return "parent"
        if any(marker in folded for marker in ("hop nhat", "bao cao tai chinh hop nhat")):
            return "consolidated"
        return None

    def documents_for_question(self, question: str, *, include_prior: bool = False) -> list[DocumentRef]:
        tickers = self.infer_tickers(question)
        years = self.infer_years(question)
        scope = self.infer_scope(question)
        target_years = set(years)
        if include_prior:
            target_years.update(year - 1 for year in years)
        selected: list[DocumentRef] = []
        for ticker in tickers:
            candidates = [
                doc for doc in self._by_ticker.get(ticker, []) if not target_years or doc.report_year in target_years
            ]
            # Scope preference is resolved independently for each year. A
            # consolidated prior-year report must not hide an unknown-scope
            # current-year report (a real pattern in the 2025 snapshot).
            for year in sorted({doc.report_year for doc in candidates}):
                year_docs = [doc for doc in candidates if doc.report_year == year]
                if scope:
                    scoped = [doc for doc in year_docs if doc.scope == scope]
                    if scoped:
                        year_docs = scoped
                else:
                    consolidated = [doc for doc in year_docs if doc.scope == "consolidated"]
                    unknown = [doc for doc in year_docs if doc.scope == "unknown"]
                    if consolidated:
                        year_docs = consolidated
                    elif unknown:
                        year_docs = unknown
                selected.extend(year_docs)
        return sorted(selected, key=lambda doc: (doc.ticker, doc.report_year, doc.scope, doc.doc_id))

    def rows_for_documents(self, documents: list[DocumentRef]) -> list[RowAsset]:
        if not documents:
            return []
        ids = [doc.doc_id for doc in documents]
        placeholders = ",".join("?" for _ in ids)
        semantic = ", row_label, section" if self._semantic_rows else ""
        query = (
            f"SELECT doc_id, table_id, row_idx, cells_json, folded_text{semantic} FROM rows "
            f"WHERE doc_id IN ({placeholders})"
        )
        return [
            RowAsset(
                doc_id=row["doc_id"],
                table_id=row["table_id"],
                row_idx=row["row_idx"],
                cells=json.loads(row["cells_json"]),
                folded_text=row["folded_text"],
                row_label=row["row_label"] if self._semantic_rows else "",
                section=row["section"] if self._semantic_rows else "",
            )
            for row in self.conn.execute(query, ids)
        ]

    def table(self, doc_id: str, table_id: int) -> TableAsset:
        key = (doc_id, table_id)
        if key in self._table_cache:
            return self._table_cache[key]
        semantic = ", statement_kind, unit_scale, header_rows_json" if self._semantic_tables else ""
        row = self.conn.execute(
            f"SELECT doc_id, table_id, page, context, rows_json{semantic} "
            "FROM tables WHERE doc_id=? AND table_id=?",
            (doc_id, table_id),
        ).fetchone()
        if row is None:
            raise KeyError((doc_id, table_id))
        asset = TableAsset(
            doc_id=row["doc_id"],
            table_id=row["table_id"],
            page=row["page"],
            context=row["context"],
            rows=json.loads(row["rows_json"]),
            statement_kind=row["statement_kind"] if self._semantic_tables else "unknown",
            unit_scale=float(row["unit_scale"]) if self._semantic_tables else 1.0,
            header_rows=tuple(json.loads(row["header_rows_json"])) if self._semantic_tables else (),
        )
        self._table_cache[key] = asset
        return asset

    def document(self, doc_id: str) -> DocumentRef:
        return self._doc_by_id[doc_id]
