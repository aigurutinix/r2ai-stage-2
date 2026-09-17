"""Unified method pipeline — one router for all 1012 questions.

Methodology (for private-round presentation):

  question
    → shape router (blocks.py)          # same typology for every item
    → specialist for that shape         # catalog / DAG / address book
    → gates: answer_gate + reexec       # never ship unverified numbers
    → else keep vote3 incumbent         # ensemble floor already measured

Specialists (no per-question hand edits):

  catalog_ratio   BTC panel/catalog.RATIOS — match official ratio *name*,
                  bind Circular-200 cells via CellBook, emit pandas
  cellbook_dag    Multi-company / multi-year DAG over the same metric
                  vocabulary (argmin, median split, year extreme, …)
  dual_tab        Forward∩reverse table agree + table_norm unit
  maso_plan       Circular-200 / statement address + table_norm
  note_plan       Thuyết minh / disclosure rows (bank, thù lao, …)
  address_book    Wide maso zip (opt-in only; scale risk)

Usage:
  python scripts/fresh/method_pipeline.py
  python scripts/fresh/validate_submission.py --zip submissions/method_v1.zip
  python scripts/fresh/audit_method_reason.py
"""

from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fresh"))

from answer_gate import verdict  # noqa: E402
from blocks import block_of  # noqa: E402
from build_notcell_zip import reexec_ok  # noqa: E402
from build_dualread_zip import build_tab_patch, load_jsonl  # noqa: E402
from build_submission import MAGNITUDE_CODES, PROGRAM, unit_of as unit_of_sub  # noqa: E402
from cell_verify import verify_patch  # noqa: E402
from validate_bctc import in_document, reread_from_corpus, resolve_corpus_csv  # noqa: E402
import hard_hop as hh  # noqa: E402
from identity_check import prefer_identity_binding  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402
import parse_statements as ps  # noqa: E402
import table_norm as tn  # noqa: E402

# BTC panel/catalog.RATIOS — keyed by official `name` (Vietnamese).
# Source of truth: vifinqa-official/.../panel/catalog.py (loaded by name, not hand rules).
BTC_RATIO_NAMES: tuple[tuple[str, str], ...] = (
    ("gross_margin", "Biên lợi nhuận gộp"),
    ("net_margin", "Biên lợi nhuận ròng"),
    ("operating_margin", "Biên lợi nhuận hoạt động"),
    ("npat_to_ending_assets", "Lợi nhuận sau thuế trên tổng tài sản cuối kỳ"),
    ("npat_to_ending_equity", "Lợi nhuận sau thuế trên vốn chủ sở hữu cuối kỳ"),
    ("liabilities_to_equity", "Hệ số nợ phải trả trên vốn chủ sở hữu"),
    ("debt_to_assets", "Hệ số nợ trên tổng tài sản"),
    ("current_ratio", "Hệ số thanh toán hiện hành"),
    ("quick_ratio", "Hệ số thanh toán nhanh"),
    ("asset_turnover", "Vòng quay tổng tài sản"),
    ("interest_coverage", "Hệ số khả năng thanh toán lãi vay"),
    ("inventory_to_current_liabilities", "Hàng tồn kho trên nợ ngắn hạn"),
    ("operating_cash_flow_ratio", "Hệ số dòng tiền hoạt động trên nợ ngắn hạn"),
    ("cfo_margin", "Biên dòng tiền từ hoạt động kinh doanh trên doanh thu"),
    ("inventory_to_assets", "Tỷ trọng hàng tồn kho trên tổng tài sản"),
    ("liabilities_to_assets", "Hệ số nợ phải trả trên tổng tài sản"),
    ("sga_intensity", "Cường độ chi phí bán hàng và quản lý trên doanh thu"),
    ("long_term_assets_share", "Tỷ trọng tài sản dài hạn trên tổng tài sản"),
    ("cfo_to_npat", "Tỷ lệ CFO trên lợi nhuận sau thuế"),
)

# Map BTC ratio keys → CellBook metric keys (same Circular-200 codes).
CATALOG_TO_CELLBOOK = {
    "gross_margin": "gross_margin",
    "net_margin": "net_margin",
    "operating_margin": "operating_margin",
    "npat_to_ending_assets": "roa_avg",   # BTC intermediate: average assets
    "npat_to_ending_equity": "roe_avg",
    "liabilities_to_equity": "debt_to_equity",
    "debt_to_assets": "debt_to_assets",
    "liabilities_to_assets": "debt_to_assets",
    "current_ratio": "current_ratio",
    "quick_ratio": "quick_ratio",
    "asset_turnover": "asset_turnover",
    "interest_coverage": "interest_coverage",
    "inventory_to_current_liabilities": "inv_to_cl",
    "operating_cash_flow_ratio": "cfo_to_cl",
    "cfo_margin": "cfo_margin",
    "inventory_to_assets": "inventory_to_assets",
    "sga_intensity": "sga_intensity",
    "long_term_assets_share": "lt_assets_share",
    "cfo_to_npat": "cfo_to_ni",
}

SHAPE = {
    "single": ("tien — MOT O", "khac — don gian"),
    "ratio": ("ty le — mot nam", "so lan"),
    "multi_year": ("tien — nhieu nam", "ty le — nhieu nam", "khac — nhieu nam", "nam nao"),
    "screen": (
        "nhieu cong ty", "tong hop nhieu o",
        "tien — sang loc", "ty le — sang loc", "khac — sang loc",
    ),
}


@dataclass
class Patch:
    layer: str
    answer: float
    pandas_query: str
    evidence: list
    relevant_docs: list
    relevant_tables: list
    files: dict[str, bytes]
    meta: dict = field(default_factory=dict)


def shape_of(block: str) -> str:
    for name, members in SHAPE.items():
        if block in members:
            return name
    return "other"


def load_btc_catalog() -> list[tuple[str, str, str]]:
    """(btc_key, display_name, cellbook_metric) for every mapped ratio."""

    out = []
    for key, name in BTC_RATIO_NAMES:
        metric = CATALOG_TO_CELLBOOK.get(key)
        if metric is None:
            continue
        out.append((key, name, metric))
    return out


def match_catalog_ratio(question: str, catalog: list[tuple[str, str, str]]) -> tuple[str, str] | None:
    """Longest official ratio name that appears in the question."""

    folded = question.casefold()
    hits = [(key, metric, name) for key, name, metric in catalog
            if name.casefold() in folded]
    if not hits:
        # soft aliases used in the exam but not exact catalog strings
        aliases = (
            (r"\bROA\b|sinh lời trên tổng tài sản|tỷ suất sinh lời trên tổng tài sản",
             "npat_to_ending_assets", "roa_avg"),
            (r"\bROE\b|sinh lời trên vốn chủ|tỷ suất sinh lời trên vốn chủ",
             "npat_to_ending_equity", "roe_avg"),
            (r"\bROS\b|biên lợi nhuận thuần", "net_margin", "net_margin"),
            (r"\bD/E\b|tỷ số D/E", "liabilities_to_equity", "debt_to_equity"),
            (r"quick ratio|thanh toán nhanh", "quick_ratio", "quick_ratio"),
            (r"current ratio|thanh toán hiện hành", "current_ratio", "current_ratio"),
            (r"vòng quay tổng tài sản", "asset_turnover", "asset_turnover"),
        )
        for pat, key, metric in aliases:
            if re.search(pat, question, re.I):
                return key, metric
        return None
    hits.sort(key=lambda item: -len(item[2]))
    return hits[0][0], hits[0][1]


def patch_from_hop(hit: dict, layer: str) -> Patch:
    files = {f"data/{name}": text.encode("utf-8")
             for name, text in hit["csv_payloads"].items()}
    return Patch(
        layer=layer,
        answer=float(hit["answer"]),
        pandas_query=hit["pandas_query"],
        evidence=hit["evidence"],
        relevant_docs=hit["relevant_docs"],
        relevant_tables=hit["relevant_tables"],
        files=files,
        meta={"op": hit.get("op"), "filter": hit.get("filter"),
              "target": hit.get("target")},
    )


def try_catalog_ratio(
    book: hh.CellBook,
    question: str,
    resolver: TickerResolver,
    catalog: list[tuple[str, str, str]],
) -> Patch | None:
    matched = match_catalog_ratio(question, catalog)
    if matched is None:
        return None
    btc_key, metric = matched
    tickers = hh.resolve_cohort(question, resolver)
    years = hh.years_of(question)
    if len(tickers) != 1 or len(years) != 1:
        return None
    # Screening questions must not be answered by a bare ratio of one firm.
    if re.search(
        r"trung vị|cao nhất|thấp nhất|trong nhóm|có bao nhiêu doanh nghiệp|"
        r"bình quân của các|các doanh nghiệp có",
        question, re.I,
    ):
        return None
    ticker, year = tickers[0], years[0]
    scope = "separate" if hh.PARENT_RE.search(question) else "consolidated"
    unit = hh.unit_of(question)
    # Ratios / % : no money scale; money-like catalog entries rare here
    got = hh.compute(book, ticker, year, scope, metric)
    if got is None:
        return None
    value, cells = got
    # Catalog percentages already *100 inside CellBook pct ratios
    if unit is not None and metric in hh.ATOM:
        value = value / unit
    packed = hh._pack_rate(value, cells, filter_cells=[])
    hit = {
        "op": "catalog_ratio", "filter": btc_key, "target": metric,
        "winner": ticker, "year": year,
        **packed,
    }
    if not hh.accept_hit(question, hit):
        return None
    return patch_from_hop(hit, "catalog_ratio")


def try_cellbook_dag(
    book: hh.CellBook,
    question: str,
    resolver: TickerResolver,
) -> Patch | None:
    try:
        hit = hh.solve_one(book, question, resolver)
    except Exception:
        return None
    if hit is None or not hh.confidence_hit(question, hit):
        return None
    return patch_from_hop(hit, "cellbook_dag")


def try_address_book(qid: int, address_rows: dict[int, dict],
                     blobs: dict[str, bytes]) -> Patch | None:
    row = address_rows.get(qid)
    if row is None:
        return None
    evidence = row.get("evidence") or []
    if not evidence or not row.get("pandas_query"):
        return None
    files = {ev["csv_path"]: blobs[ev["csv_path"]]
             for ev in evidence if ev["csv_path"] in blobs}
    if len(files) != len(evidence):
        return None
    return Patch(
        layer="address_book",
        answer=float(row["answer"]),
        pandas_query=row["pandas_query"],
        evidence=evidence,
        relevant_docs=row.get("relevant_docs") or [],
        relevant_tables=row.get("relevant_tables") or [],
        files=files,
        meta={"source": "wide_maso"},
    )


def load_dual_ids() -> set[int]:
    path = ROOT / "artifacts/fresh/tiers.json"
    if not path.exists():
        return set()
    data = json.loads(path.read_text(encoding="utf-8"))
    tiers = data.get("tiers") or {}
    return set(tiers.get("cung o", [])) | set(tiers.get("cung con so", []))


def _iloc_wrap(code: str, kind: str) -> str:
    k = (kind or "").lower()
    if k in ("note", "thuyet minh"):
        return "abs"
    if str(code).lstrip("t") in MAGNITUDE_CODES.get(k, set()):
        return "abs"
    return ""


def _exec_iloc(blob: bytes, row: int, col: int, code: str, kind: str,
               scale: float, unit: float) -> tuple[float, str] | None:
    program = PROGRAM.format(
        row=row, col=col, code=code or "?", kind=kind or "bang",
        wrap=_iloc_wrap(code, kind), scale=scale, unit=unit,
    )
    frame = pd.read_csv(io.BytesIO(blob), dtype=str, keep_default_na=False)
    ns: dict = {"pd": pd, "df": frame}
    try:
        exec(program, ns, ns)  # noqa: S102
        return float(ns["result"]), program
    except Exception:
        return None


def try_dual_tab(qid: int, question: str, tab: dict[int, dict],
                 dual_ids: set[int]) -> Patch | None:
    """Only splice when forward+reverse agree (same cell or same figure)."""

    if qid not in dual_ids:
        return None
    entry = tab.get(qid)
    if entry is None:
        return None
    if not tn.scope_matches(question, str(entry.get("doc") or "")):
        return None
    built = build_tab_patch(entry, question)
    if built is None:
        return None
    record, files = built
    if record.get("answer") is None:
        return None
    blob = next(iter(files.values()))
    name_u, q_unit = unit_of_sub(question)
    if not q_unit:
        return None
    eff = tn.effective_divide_unit(question, blob, float(entry.get("scale") or 1.0))
    if not eff:
        return None
    ran = _exec_iloc(
        blob, int(entry["row"]), int(entry["col"]),
        str(entry.get("label") or entry.get("code") or "?"),
        str(entry.get("kind") or "bang"),
        float(entry.get("scale") or 1.0), eff,
    )
    if ran is None:
        return None
    answer, program = ran
    quoted = str(entry.get("quoted") or "")
    if abs(answer) < 1e-9 and re.search(r"[1-9]", quoted):
        return None
    return Patch(
        layer="dual_tab",
        answer=answer,
        pandas_query=program,
        evidence=record["evidence"],
        relevant_docs=record["relevant_docs"],
        relevant_tables=record["relevant_tables"],
        files=files,
        meta={"tier": "dual_agree", "table_ref": entry.get("table_ref"),
              "unit_name": name_u, "eff_unit": eff,
              "table_unit": tn.detect_table_unit_vnd(blob)},
    )


def maso_semantic_ok(question: str, entry: dict) -> bool:
    """Reject plans that pick the wrong statement family for the ask."""

    q = question.casefold()
    kind = (entry.get("kind") or "").lower()
    code = str(entry.get("code") or "").lstrip("t")
    if re.search(r"giá trị hợp lý|tài sản tài chính|số dư|tổng số dư", q):
        if kind == "kqkd" or code in {"21", "22", "23", "25", "26"}:
            return False
    if not tn.scope_matches(question, str(entry.get("doc") or "")):
        return False
    return True


def _doc_scope(doc: str) -> str:
    d = doc.casefold()
    if "separate" in d or "riêng" in d:
        return "separate"
    if "consolidated" in d or "hợp nhất" in d:
        return "consolidated"
    return "unknown"


def _doc_ticker_year(doc: str) -> tuple[str | None, str | None]:
    match = re.match(r"^([A-Z0-9]+)_financial_statements_(\d{4})_", doc)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def _resolve_topk_ref(ref: dict) -> Path | None:
    ticker = ref.get("ticker")
    year = str(ref.get("year") or "")
    doc = ref.get("doc")
    tid = ref.get("table_id")
    if not (ticker and year and doc and tid is not None):
        return None
    path = (
        ROOT / "data" / "official_corpus" / ticker / year / doc
        / f"{doc}_extracted_tables" / f"table_{tid}.csv"
    )
    return path if path.is_file() else None


def identity_topk_pool(
    question: str,
    greedy_doc: str,
    greedy_path: Path,
    topk_refs: list[dict] | None,
) -> list[Path]:
    """Retrieval tables for identity rebinding (same ticker/year/scope).

    Cross cons↔sep only when the question explicitly demands the other scope.
    """

    if not topk_refs:
        return []
    gt, gy = _doc_ticker_year(greedy_doc)
    gs = _doc_scope(greedy_doc)
    want_sep = tn.wants_separate(question)
    want_cons = tn.wants_consolidated(question)
    out: list[Path] = []
    seen: set[Path] = {greedy_path.resolve()} if greedy_path.is_file() else set()

    for ref in topk_refs:
        path = _resolve_topk_ref(ref)
        if path is None:
            continue
        doc = path.parent.parent.name
        ct, cy = _doc_ticker_year(doc)
        if gt and ct and ct != gt:
            continue
        if gy and cy and cy != gy:
            continue
        cs = _doc_scope(doc)
        if cs != gs:
            if want_sep and cs == "separate":
                pass
            elif want_cons and cs == "consolidated":
                pass
            else:
                continue
        for candidate in (path, *sorted(path.parent.glob("table_*.csv"))):
            key = candidate.resolve()
            if candidate.is_file() and key not in seen:
                seen.add(key)
                out.append(candidate)
    return out


def try_maso_plan(
    qid: int,
    question: str,
    plans: dict[int, dict],
    topk: dict[int, list[dict]] | None = None,
) -> Patch | None:
    """Circular-200 / statement address; prefer identity-winning CDKT table."""

    entry = plans.get(qid)
    if entry is None:
        return None
    if entry.get("source") not in ("statement", "maso") and entry.get("kind") not in (
            "cdkt", "kqkd", "lctt"):
        if entry.get("kind") not in ("cdkt", "kqkd", "lctt"):
            return None
    lex_score = float(entry.get("score") or 0)
    if lex_score < 0.35:
        return None
    if not maso_semantic_ok(question, entry):
        return None
    name, q_unit = unit_of_sub(question)
    if not q_unit:
        return None
    source = ROOT / entry["csv"] if not Path(entry["csv"]).is_file() else Path(entry["csv"])
    if not source.exists():
        source = tn.resolve_csv(entry["csv"], ROOT)
    if source is None or not source.exists():
        return None

    kind = str(entry.get("kind") or "cdkt")
    code = str(entry.get("code") or entry.get("label") or "?")
    row = int(entry["row"])
    col = int(entry["col"])
    scale = float(entry.get("scale") or 1.0)
    table_id = entry.get("table_id")
    id_meta: dict = {}
    greedy_doc = str(entry.get("doc") or source.parent.parent.name)
    extra = identity_topk_pool(
        question, greedy_doc, source, (topk or {}).get(qid))

    # Prefer identity-clean CDKT fragment. Never overwrite greedy row/col unless
    # we actually flip tables — stmt.current is always end-of-year and would
    # destroy "đầu năm" addresses (seen: DIG 124 23.07 → 7.85).
    preferred = prefer_identity_binding(
        source, code, kind=kind, extra_paths=extra or None)
    if preferred is not None:
        if preferred.get("doc_ok") is False and not preferred.get("flipped"):
            return None
        if preferred["flipped"]:
            if tn.wants_beginning(question):
                import csv as _csv
                alt = preferred["path"]
                rows_alt = list(_csv.reader(alt.open(encoding="utf-8-sig")))
                rebound = False
                if len(rows_alt) >= 2:
                    sc = ps.scale_from_unit_text(
                        ",".join(c for r in rows_alt[:3] for c in r), strict=False)
                    table = ps.Table(
                        doc_name=alt.parent.parent.name,
                        table_id=int(alt.stem.split("_")[-1]),
                        page_no=0, header=rows_alt[0], rows=rows_alt[1:],
                        unit_snippets=("Đơn vị tính: Đồng",) if sc is None else (),
                    )
                    stmt = ps.parse_statement(table)
                    if stmt is not None and code in stmt.prior:
                        cell = stmt.prior[code]
                        source = alt
                        row, col = cell.row_idx, cell.col_idx
                        scale = float(cell.scale or scale)
                        table_id = preferred["table_id"]
                        rebound = True
                if not rebound:
                    pass  # keep greedy address
            else:
                source = preferred["path"]
                row = int(preferred["row"])
                col = int(preferred["col"])
                scale = float(preferred["scale"] or scale)
                table_id = preferred["table_id"]
        id_meta = {
            "identity_flipped": preferred["flipped"] and (
                Path(source).resolve() != Path(entry["csv"]).resolve()
                and Path(source).resolve() != (ROOT / entry["csv"]).resolve()
            ),
            "identity_score": preferred["score"],
            "identity_n_codes": preferred["n_codes"],
            "identity_n_cand": preferred["n_cand"],
            "identity_doc_ok": preferred.get("doc_ok"),
            "identity_topk": bool(extra),
        }
        if lex_score < 0.55 and not id_meta["identity_flipped"]:
            return None
        if id_meta["identity_flipped"] and float(preferred["score"]) < 1.0:
            return None
    elif kind.lower() == "cdkt":
        from identity_check import doc_union_rollup_ok
        dok = doc_union_rollup_ok(source.parent)
        if dok is False:
            return None
        if lex_score < 0.55:
            return None
        id_meta = {
            "identity_doc_ok": dok,
            "identity_flipped": False,
            "identity_topk": bool(extra),
        }
    elif lex_score < 0.55:
        return None

    bound_doc = source.parent.parent.name
    blob = source.read_bytes()
    eff = tn.effective_divide_unit(question, blob, scale)
    if not eff:
        return None
    csv_name = f"{bound_doc}_table_{table_id}.csv"
    ran = _exec_iloc(blob, row, col, code, kind, scale, eff)
    if ran is None:
        return None
    answer, program = ran
    return Patch(
        layer="maso_plan",
        answer=answer,
        pandas_query=program,
        evidence=[{"variable": "df", "csv_path": f"data/{csv_name}"}],
        relevant_docs=[bound_doc],
        relevant_tables=[
            entry.get("table_ref") or f"{bound_doc}|table_{table_id}"],
        files={f"data/{csv_name}": blob},
        meta={"code": entry.get("code"), "kind": kind, "score": entry.get("score"),
              "source": entry.get("source"), "eff_unit": eff,
              "table_unit": tn.detect_table_unit_vnd(blob), **id_meta},
    )


def try_panel_det(qid: int, question: str, cache: dict[int, dict],
                  incumbent: float) -> Patch | None:
    """Deterministic panel answers from metrics.parquet (COMPLETE cohort)."""

    rec = cache.get(qid)
    if not rec or not rec.get("ok"):
        return None
    code = rec.get("code") or ""
    panel_rows = rec.get("panel_rows")
    if not code or not panel_rows:
        return None
    import csv as _csv

    buf = io.StringIO(newline="")
    _csv.writer(buf, lineterminator="\n").writerows(panel_rows)
    blob = buf.getvalue().encode("utf-8")
    name = f"metric_panel_q{qid}.csv"
    files = {f"data/{name}": blob}
    ns: dict = {"pd": pd}
    try:
        ns["df"] = pd.read_csv(io.BytesIO(blob), dtype=str, keep_default_na=False)
        exec(code, ns, ns)  # noqa: S102
        answer = float(ns["result"])
    except Exception:
        return None
    gate = verdict(question, answer)
    if gate:
        return None
    return Patch(
        layer="panel_det",
        answer=answer,
        pandas_query=code,
        evidence=[{"variable": "df", "csv_path": f"data/{name}"}],
        relevant_docs=[],
        relevant_tables=[],
        files=files,
        meta={"shape": rec.get("shape"), "metrics": rec.get("metrics")},
    )


def load_panel_det_cache(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out: dict[int, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("ok"):
                out[int(rec["id"])] = rec
    return out


def merge_note_plans(*paths: Path) -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for path in paths:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            qid = int(rec["id"])
            prev = merged.get(qid)
            if prev is None or float(rec.get("row_score") or 0) > float(
                    prev.get("row_score") or 0):
                merged[qid] = rec
    return merged


def try_note_plan(qid: int, question: str, plans: dict[int, dict]) -> Patch | None:
    """Thuyết minh / disclosure rows (bank ngành, thù lao, lãi tiền gửi, …)."""

    if not re.search(
        r"thuyết minh|thù lao|lãi tiền gửi|cho vay.{0,20}ngành|"
        r"cổ tức|cổ đông|phạt|dự phòng|dự phòng trích|"
        r"doanh thu từ|chi phí khác|lãi vay phải trả|trả trước|"
        r"tiền gửi|phải thu|phải trả",
        question, re.I,
    ):
        return None
    entry = plans.get(qid)
    if entry is None:
        return None
    row_score = float(entry.get("row_score") or entry.get("score") or 0)
    if row_score < 0.45:
        return None
    label = str(entry.get("row_label") or entry.get("label") or "")
    if not tn.label_overlaps_question(label, question, min_token=2):
        return None
    tokens = [t for t in re.split(r"[^0-9a-zà-ỹ]+", label.casefold()) if len(t) >= 3]
    if tokens:
        hits = sum(1 for t in tokens if t in question.casefold())
        if hits / len(tokens) < 0.4:
            return None
    if not tn.scope_matches(question, str(entry.get("doc") or "")):
        return None
    name, q_unit = unit_of_sub(question)
    if not q_unit:
        return None
    source = tn.resolve_csv(entry["csv"], ROOT)
    if source is None:
        return None
    blob = source.read_bytes()
    eff = tn.effective_divide_unit(question, blob, float(entry.get("scale") or 1.0))
    if not eff:
        return None
    csv_name = f"{entry['doc']}_table_{entry['table_id']}.csv"
    ran = _exec_iloc(
        blob, int(entry["row"]), int(entry["col"]),
        label or "note", "note",
        float(entry.get("scale") or 1.0), eff,
    )
    if ran is None:
        return None
    answer, program = ran
    if abs(answer) < 1e-9:
        return None
    return Patch(
        layer="note_plan",
        answer=answer,
        pandas_query=program,
        evidence=[{"variable": "df", "csv_path": f"data/{csv_name}"}],
        relevant_docs=[entry["doc"]],
        relevant_tables=[entry.get("table_ref") or f"{entry['doc']}|table_{entry['table_id']}"],
        files={f"data/{csv_name}": blob},
        meta={"row_label": label, "row_score": row_score, "row": int(entry["row"]),
              "col": int(entry["col"]), "eff_unit": eff,
              "table_unit": tn.detect_table_unit_vnd(blob)},
    )


def magnitude_ok(layer: str, answer: float, incumbent: float) -> bool:
    """Reject address splices that explode vs vote3 (wrong cell / unit)."""

    if layer not in ("maso_plan", "address_book", "dual_tab", "note_plan", "panel_det"):
        return True
    if abs(incumbent) < 1e-6:
        return abs(answer) < 1e6
    ratio = abs(answer / incumbent)
    return 0.02 <= ratio <= 50.0


def semantic_reject(question: str, patch: Patch) -> str | None:
    """Human-audited failure modes that pass reexec but answer the wrong ask."""

    q = question.casefold()
    meta = patch.meta or {}
    layer = patch.layer
    if layer == "maso_plan":
        code = str(meta.get("code") or "")
        if "tương đương đồng ngoại tệ" in q and code == "110":
            return "fx_equiv_vs_cash_110"
        if re.search(r"phải thu.{0,30}khách hàng", question, re.I) and code == "130":
            return "recv_customers_vs_130"
        if re.search(r"vay ngắn hạn từ ngân hàng", question, re.I) and code == "320":
            return "bank_loan_vs_all_320"
    if layer == "cellbook_dag":
        op = meta.get("op")
        tgt = meta.get("target")
        if op == "subgroup_share" and tgt == "inventory":
            if re.search(
                r"tỷ trọng tổng nợ ngắn hạn|phần trăm tổng nợ ngắn hạn|"
                r"phần trăm tổng nợ phải trả",
                question, re.I,
            ):
                return "share_debt_vs_inventory_target"
        if op == "year_argmax" and re.search(r"\bROE\b", question) and tgt == "gross_margin":
            return "roe_asked_gm_target"
        if op in ("year_argmin", "year_argmax") and re.search(
                r"năm kế tiếp|năm liền sau|vào năm sau năm", question, re.I,
        ):
            return "next_year_shift_unhandled"
    return None


def accept(patch: Patch | None, question: str, incumbent: float) -> tuple[Patch | None, str]:
    if patch is None:
        return None, "no_candidate"
    bad = semantic_reject(question, patch)
    if bad:
        return None, f"semantic:{bad}"
    gate = verdict(question, patch.answer)
    if gate:
        return None, f"gate:{gate}"
    if abs(patch.answer - incumbent) <= 0.01:
        return None, "same"
    if not magnitude_ok(patch.layer, patch.answer, incumbent):
        return None, "magnitude"
    record = {
        "answer": patch.answer,
        "pandas_query": patch.pandas_query,
        "evidence": patch.evidence,
    }
    if not reexec_ok(record, patch.files):
        return None, "reexec_fail"
    # BCTC grounding: answer = independent re-read from corpus / compute
    if patch.layer in ("maso_plan", "note_plan", "dual_tab"):
        reread, _err, source = reread_from_corpus(record)
        if reread is None or abs(reread - patch.answer) > 0.011:
            return None, "bctc_reread_fail"
        if source != "disk":
            return None, "bctc_not_corpus_disk"
        path = (
            resolve_corpus_csv(patch.evidence[0].get("csv_path", ""))
            if patch.evidence else None
        )
        if path and path.is_file():
            vfail = verify_patch(
                patch.layer, path.read_bytes(), path, patch.meta or {}, question)
            if vfail in ("cdkt_rollup_fail", "identity_doc_fail", "cong_mismatch"):
                return None, f"verify:{vfail}"
    elif patch.layer in ("panel_det", "cellbook_dag", "catalog_ratio"):
        reread, _err, _src = reread_from_corpus(record, patch.files)
        if reread is None or abs(reread - patch.answer) > 0.011:
            return None, "bctc_compute_fail"
    return patch, "ok"


def load_address_zip(path: Path) -> tuple[dict[int, dict], dict[str, bytes]]:
    if not path.exists():
        return {}, {}
    with zipfile.ZipFile(path) as z:
        rows = {r["id"]: r for r in json.loads(z.read("submission.json"))}
        blobs = {n: z.read(n) for n in z.namelist() if n.startswith("data/")}
    return rows, blobs


def write_method_doc(
    dest: Path,
    stats: Counter,
    by_layer: Counter,
    by_shape: Counter,
    catalog_n: int,
    consistency: Counter | None = None,
    consistency_n: int = 0,
) -> None:
    lines = [
        "# Phương pháp — ViFinQA (method_v1)",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"Submission: `{dest.name}`",
        "",
        "## Một câu cho giám khảo",
        "",
        "Mọi câu hỏi đi **cùng một pipeline**: phân hình → chuyên gia theo hình "
        "(catalog BTC / DAG CellBook / dual-tab / Mã số / thuyết minh) → cổng "
        "`answer_gate` + `table_norm` + re-exec → **consistency** "
        "(program ≡ answer bằng tập toán đóng abs / ×÷10ᵏ) → nếu fail thì giữ "
        "**vote3** (majority ba nguồn độc lập, EXEC 0.415).",
        "",
        "Model **không invent số**. Số chỉ đến từ ô Circular-200 / công thức "
        "BTC catalog; consistency chỉ chỉnh *đơn vị / dấu* khi program và "
        "`answer` lệch đúng khoảng đơn vị tiếng Việt chuẩn.",
        "",
        "## Vì sao consistency (không sửa tay từng ID)",
        "",
        "Public rank = **EXECUTION_ACCURACY**: scorer chạy lại `pandas_query`. "
        "ANSWER thường cao hơn EXEC vì field `answer` đúng nhưng program thiếu "
        "bước ÷Triệu/Tỷ hoặc `|·|` trên dòng chi phí. Lớp consistency đóng "
        "khoảng đó bằng algebra cố định — cùng ý với `table_norm` — "
        "không chọn số mới, không whitelist ID.",
        "",
        "## Sơ đồ",
        "",
        "```",
        "questions.jsonl",
        "      │",
        "      ▼",
        " shape router              single | ratio | multi_year | screen",
        "      │",
        "      ├── catalog_ratio    ← BTC RATIOS (+ ROA/ROE avg)",
        "      ├── panel_det          ← metrics.parquet COMPLETE cohort",
        "      ├── cellbook_dag     ← DAG trên vocabulary metric",
        "      ├── note_plan        ← thuyết minh / disclosure (+ all_tables)",
        "      ├── dual_tab         ← forward∩reverse + table_norm",
        "      ├── maso_plan        ← Mã số + identity CDKT (+ retrieval top-k) + table_norm",
        "      └── incumbent        ← vote3",
        "      │",
        "      ▼",
        " gates: answer_gate + magnitude + verify(cell) + reexec(pandas on CSV)",
        "      │",
        "      ▼",
        " consistency_repair        abs | ÷10³|⁶|⁹|¹² | ×10³|⁶",
        "      │                     (chỉ khi reexec(program) → answer)",
        "      ▼",
        f" {dest.name}",
        "```",
        "",
        "## Vì sao không “thêm rule tay từng câu”",
        "",
        "Vocabulary metric = **catalog BTC** (cùng bộ sinh đề). "
        "DAG screen = tổ hợp op đóng trên vocabulary đó. "
        "`table_norm` + identity CDKT chọn bảng/cột trước khi đọc. "
        "Consistency = hậu xử lý đơn vị/dấu đóng. "
        "Mỗi thay đổi **reproduce** bằng `pandas_query`.",
        "",
        "## Thống kê lần chạy này",
        "",
        f"- Catalog ratios mapped: **{catalog_n}**",
        f"- Đổi so với vote3 (specialists): **{sum(by_layer.values())}** / 1012",
        f"- Consistency repairs: **{consistency_n}**",
        "",
        "### Theo lớp specialist",
        "",
    ]
    for name, count in by_layer.most_common():
        lines.append(f"- `{name}`: {count}")
    lines.extend(["", "### Theo shape", ""])
    for name, count in by_shape.most_common():
        lines.append(f"- `{name}`: {count}")
    if consistency:
        lines.extend(["", "### Consistency ops", ""])
        for name, count in consistency.most_common():
            if name.startswith("repair_"):
                lines.append(f"- `{name}`: {count}")
    lines.extend([
        "",
        "### Cổng từ chối",
        "",
    ])
    for name, count in stats.most_common():
        if name.startswith("reject_") or name.startswith("skip_"):
            lines.append(f"- `{name}`: {count}")
    lines.extend([
        "",
        "## Reproduce",
        "",
        "```bash",
        "python scripts/fresh/method_pipeline.py",
        "# hoặc chỉ consistency trên zip đã có:",
        "python scripts/fresh/consistency_repair.py "
        "--src submissions/method_v1.zip --dest submissions/method_v2.zip",
        "python scripts/fresh/validate_submission.py --zip submissions/method_v2.zip",
        "```",
        "",
        "## File",
        "",
        "- `artifacts/fresh/METHOD.md` — tài liệu này",
        "- `artifacts/fresh/method_manifest.jsonl` — đổi lớp specialist",
        "- `artifacts/fresh/consistency_manifest.jsonl` — repair đơn vị/dấu",
        f"- `submissions/{dest.name}` — bài nộp",
        "",
    ])
    path = ROOT / "artifacts/fresh/METHOD.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"-> {path}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="submissions/vote3.zip")
    parser.add_argument("--dest", default="submissions/method_v3.zip")
    parser.add_argument("--address", default="submissions/_wide_maso.zip")
    parser.add_argument("--consistency", action="store_true", default=False,
                        help="Run consistency/sanitize repair pass")
    parser.add_argument("--no-consistency", action="store_false", dest="consistency",
                        help="Skip consistency repair (default)")
    parser.add_argument("--ids", default="",
                        help="Comma-separated question ids only (smoke mode)")
    parser.add_argument("--use-address", action="store_true",
                        help="Enable address_book specialist (off by default: "
                             "wide_maso has known scale-replace risk on public)")
    args = parser.parse_args()

    qs = {
        json.loads(line)["id"]: json.loads(line)["question"]
        for line in (ROOT / "data/questions/questions.jsonl").read_text(
            encoding="utf-8").splitlines()
        if line.strip()
    }
    # company counts for block_of
    resolver = TickerResolver()
    book = hh.CellBook()
    catalog = load_btc_catalog()

    base_path = ROOT / args.base
    with zipfile.ZipFile(base_path) as z:
        rows = {r["id"]: r for r in json.loads(z.read("submission.json"))}
        files = {n: z.read(n) for n in z.namelist() if n != "submission.json"}

    address_rows, address_blobs = load_address_zip(ROOT / args.address)
    dual_ids = load_dual_ids()
    tab = load_jsonl(ROOT / "artifacts/fresh/tab_plan.jsonl")
    maso_plans = load_jsonl(ROOT / "artifacts/fresh/greedy_plan.jsonl")
    note_plans = merge_note_plans(
        ROOT / "artifacts/fresh/note_plan.jsonl",
        ROOT / "artifacts/fresh/note_plan_all.jsonl",
    )
    panel_cache = load_panel_det_cache(ROOT / "artifacts/panel_det.jsonl")
    topk_path = ROOT / "artifacts/fresh/dense_topk.json"
    topk: dict[int, list[dict]] = {}
    if topk_path.exists():
        topk = {
            int(k): v[:9]
            for k, v in json.loads(topk_path.read_text(encoding="utf-8")).items()
        }
    print(f"catalog={len(catalog)} dual_tab_ids={len(dual_ids)} "
          f"maso_plans={len(maso_plans)} note_plans={len(note_plans)} "
          f"panel_det={len(panel_cache)} tab={len(tab)} identity_topk={len(topk)}",
          flush=True)

    stats: Counter[str] = Counter()
    manifest: list[dict] = []
    by_layer: Counter[str] = Counter()
    by_shape: Counter[str] = Counter()

    id_filter: set[int] | None = None
    if args.ids:
        id_filter = {int(x) for x in args.ids.split(",") if x.strip()}
        print(f"smoke mode: {len(id_filter)} ids", flush=True)

    for qid in sorted(qs):
        if id_filter is not None and qid not in id_filter:
            continue
        question = qs[qid]
        n_co = len(hh.resolve_cohort(question, resolver))
        block = block_of(question, companies=max(1, n_co))
        shape = shape_of(block)
        incumbent = float(rows[qid].get("answer") or 0)

        candidates: list[tuple[str, Patch | None]] = []
        # Verified specialists first; wide address only with --use-address
        if shape == "screen":
            order = ("panel_det", "cellbook_dag", "catalog_ratio")
        elif shape == "ratio":
            order = ("panel_det", "catalog_ratio", "cellbook_dag", "maso_plan")
        elif shape == "multi_year":
            order = ("panel_det", "cellbook_dag", "catalog_ratio")
        elif shape == "single":
            order = ("dual_tab", "maso_plan", "note_plan", "panel_det", "catalog_ratio")
        else:
            order = ("panel_det", "cellbook_dag", "catalog_ratio", "maso_plan", "note_plan")
        if args.use_address:
            order = order + ("address_book",)

        for name in order:
            if name == "catalog_ratio":
                candidates.append(
                    ("catalog_ratio", try_catalog_ratio(book, question, resolver, catalog)))
            elif name == "cellbook_dag":
                candidates.append(
                    ("cellbook_dag", try_cellbook_dag(book, question, resolver)))
            elif name == "dual_tab":
                candidates.append(
                    ("dual_tab", try_dual_tab(qid, question, tab, dual_ids)))
            elif name == "maso_plan":
                candidates.append(
                    ("maso_plan", try_maso_plan(qid, question, maso_plans, topk)))
            elif name == "note_plan":
                candidates.append(
                    ("note_plan", try_note_plan(qid, question, note_plans)))
            elif name == "panel_det":
                candidates.append(
                    ("panel_det", try_panel_det(qid, question, panel_cache, incumbent)))
            elif name == "address_book":
                candidates.append(
                    ("address_book", try_address_book(qid, address_rows, address_blobs)))

        chosen: Patch | None = None
        for name, cand in candidates:
            patch, status = accept(cand, question, incumbent)
            if status == "ok" and patch is not None:
                chosen = patch
                break
            stats[f"reject_{name}_{status}"] += 1

        if chosen is None:
            stats["keep_incumbent"] += 1
            continue

        rows[qid].update({
            "answer": chosen.answer,
            "pandas_query": chosen.pandas_query,
            "evidence": chosen.evidence,
            "relevant_docs": chosen.relevant_docs,
            "relevant_tables": chosen.relevant_tables,
        })
        files.update(chosen.files)
        by_layer[chosen.layer] += 1
        by_shape[shape] += 1
        stats[f"accept_{chosen.layer}"] += 1
        manifest.append({
            "id": qid,
            "block": block,
            "shape": shape,
            "layer": chosen.layer,
            "old_answer": incumbent,
            "new_answer": chosen.answer,
            "meta": chosen.meta,
        })
        print(f"{qid} [{shape}] {chosen.layer} {incumbent} -> {chosen.answer}",
              flush=True)

    # Intermediate: specialists only (kept for ablation / consistency --src)
    mid = ROOT / "submissions/method_v1.zip"
    with zipfile.ZipFile(mid, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "submission.json",
            json.dumps([rows[i] for i in sorted(rows)], ensure_ascii=False, indent=1),
        )
        for name, blob in files.items():
            archive.writestr(name, blob)

    man_path = ROOT / "artifacts/fresh/method_manifest.jsonl"
    man_path.write_text(
        "".join(json.dumps(m, ensure_ascii=False) + "\n" for m in manifest),
        encoding="utf-8",
    )

    dest = ROOT / args.dest
    consistency_stats: Counter[str] = Counter()
    consistency_n = 0
    if args.consistency:
        import consistency_repair as cr

        repair_dest = dest if args.consistency else ROOT / "submissions/method_v2.zip"
        consistency_stats, c_manifest = cr.repair_zip(mid, repair_dest, qs)
        consistency_n = sum(
            v for k, v in consistency_stats.items() if k.startswith("repair_"))
        c_path = ROOT / "artifacts/fresh/consistency_manifest.jsonl"
        c_path.write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in c_manifest) + "\n",
            encoding="utf-8",
        )
        print(f"consistency repaired {consistency_n}  stats={dict(consistency_stats)}")
        print(f"consistency manifest -> {c_path}")
        dest = repair_dest
    else:
        if dest.resolve() != mid.resolve():
            import shutil
            shutil.copy2(mid, dest)

    write_method_doc(
        dest, stats, by_layer, by_shape, len(catalog),
        consistency=consistency_stats, consistency_n=consistency_n,
    )
    print(f"changed {len(manifest)} / 1012  layers={dict(by_layer)}  -> {dest}")
    print(f"manifest -> {man_path}")


if __name__ == "__main__":
    main()
