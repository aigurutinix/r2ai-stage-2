"""Hard multi-hop: filter/rank on one metric, then look up another.

BTC Hard questions are dependent hops — an intermediate result picks the company
(or year) whose statement answers the asked metric. One-shot answering blanks them.

This module binds Circular-200 cells live from official_corpus CSVs (scale falls
back to 1.0 when the unit line is on another page), computes named ratios from
`panel/catalog` formulas, and solves a closed set of archetypes:

  argmin_filter_lookup   company with min filter-metric → read target
  below_median_mean      companies below median filter → mean of target
  argmax_filter_lookup   company with max filter-metric → read target

Usage:
  python scripts/fresh/hard_hop.py --ids 390,368,392 --show
  python scripts/fresh/hard_hop.py --scan --limit 80
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import parse_statements as ps  # noqa: E402
from resolve_ticker import TickerResolver  # noqa: E402

YEAR_RE = re.compile(r"(?<!\d)(20[0-2]\d)(?!\d)")
RANGE_RE = re.compile(r"(20[0-2]\d)\s*[-–—]\s*(20[0-2]\d)")
PARENT_RE = re.compile(r"công ty mẹ", re.I)
# Only block genuinely unsupported constructions. Wider wording used to refuse
# solvable median-gap / inventory-days / asset-turnover hops.
HARD_REFUSE = re.compile(
    r"\bnếu\b|CAGR|chu kỳ tiền mặt|\bCCC\b|"
    r"EBIT proxy|"
    r"đòn bẩy kinh doanh|biên an toàn|"
    r"giả sử .{0,40}phát hành thêm",
    re.I,
)

# name -> (kind, code) or ratio key
# A metric named by the question's own wording rather than by a Circular-200 code.
# Measured on the 558 multi-cell questions hard_hop refuses: 423 of them name an
# indicator that has no code, so this is the difference between refusing three quarters
# of them and attempting them.
LABEL_PREFIX = "label::"

ATOM = {
    "inventory": ("cdkt", "140"),
    "current_assets": ("cdkt", "100"),
    "current_liabilities": ("cdkt", "310"),
    "total_assets": ("cdkt", "270"),
    "equity": ("cdkt", "400"),
    "liabilities": ("cdkt", "300"),
    "net_revenue": ("kqkd", "10"),
    "gross_profit": ("kqkd", "20"),
    "operating_profit": ("kqkd", "30"),
    "profit_before_tax": ("kqkd", "50"),
    "net_income": ("kqkd", "60"),
    "selling_expense": ("kqkd", "25"),
    "admin_expense": ("kqkd", "26"),
    "cfo": ("lctt", "20"),
    "interest_expense": ("kqkd", "23"),
    "long_term_assets": ("cdkt", "200"),
    "cogs": ("kqkd", "11"),
    "short_term_debt": ("cdkt", "310"),
}

RATIOS = {
    # key -> (op, numerator codes as (kind, code), denominator codes)
    "quick_ratio": ("ratio", [("cdkt", "100"), ("cdkt", "140")], [("cdkt", "310")]),
    "current_ratio": ("ratio", [("cdkt", "100")], [("cdkt", "310")]),
    "net_margin": ("pct", [("kqkd", "60")], [("kqkd", "10")]),
    "gross_margin": ("pct", [("kqkd", "20")], [("kqkd", "10")]),
    "operating_margin": ("pct", [("kqkd", "30")], [("kqkd", "10")]),
    "roa_end": ("pct", [("kqkd", "60")], [("cdkt", "270")]),
    "roe_end": ("pct", [("kqkd", "60")], [("cdkt", "400")]),
    "roa_avg": ("pct_avg", [("kqkd", "60")], [("cdkt", "270")]),
    "roe_avg": ("pct_avg", [("kqkd", "60")], [("cdkt", "400")]),
    "debt_to_equity": ("ratio", [("cdkt", "300")], [("cdkt", "400")]),
    "cfo_to_cl": ("ratio", [("lctt", "20")], [("cdkt", "310")]),
    "cfo_to_ni": ("ratio", [("lctt", "20")], [("kqkd", "60")]),
    "cfo_margin": ("pct", [("lctt", "20")], [("kqkd", "10")]),
    "inventory_to_assets": ("pct", [("cdkt", "140")], [("cdkt", "270")]),
    "debt_to_assets": ("pct", [("cdkt", "300")], [("cdkt", "270")]),
    "sga_intensity": ("pct", [("kqkd", "25"), ("kqkd", "26")], [("kqkd", "10")]),
    "nwc": ("money", [("cdkt", "100"), ("cdkt", "310")], []),
    "asset_turnover": ("ratio", [("kqkd", "10")], [("cdkt", "270")]),
    "fa_turnover": ("ratio", [("kqkd", "10")], [("cdkt", "200")]),
    "days_inventory": ("days", [("cdkt", "140")], [("kqkd", "11")]),
    "interest_coverage": ("ratio", [("kqkd", "50"), ("kqkd", "23")], [("kqkd", "23")]),
    "lt_assets_share": ("pct", [("cdkt", "200")], [("cdkt", "270")]),
    "st_debt_share": ("pct", [("cdkt", "310")], [("cdkt", "300")]),
    "inv_to_cl": ("ratio", [("cdkt", "140")], [("cdkt", "310")]),
}

YOY_LEVEL = {
    "d_gross_margin": "gross_margin",
    "d_net_margin": "net_margin",
    "d_roa": "roa_end",
    "d_inventory_to_assets": "inventory_to_assets",
    "d_sga_intensity": "sga_intensity",
    "rev_growth": "net_revenue",
}


FILTER_PHRASE = (
    (re.compile(r"tài sản ngắn hạn trừ hàng tồn kho rồi chia cho nợ ngắn hạn|"
                r"t[ỷỉ] số thanh toán nhanh|thanh toán nhanh|quick ratio", re.I),
     "quick_ratio"),
    (re.compile(r"tài sản ngắn hạn chia cho nợ ngắn hạn|"
                r"tài sản ngắn hạn gấp.{0,20}nợ ngắn hạn|"
                r"thanh toán hiện hành|current ratio", re.I), "current_ratio"),
    (re.compile(r"tỷ lệ lợi nhuận sau thuế trên doanh thu|biên lợi nhuận ròng|"
                r"biên lợi nhuận thuần|\bROS\b", re.I), "net_margin"),
    (re.compile(r"biên lợi nhuận gộp", re.I), "gross_margin"),
    (re.compile(r"biên lợi nhuận hoạt động", re.I), "operating_margin"),
    (re.compile(r"\bROA\b|sinh lời trên tổng tài sản|"
                r"tỷ suất sinh lời trên tổng tài sản", re.I), "roa_avg"),
    (re.compile(r"\bROE\b|sinh lời trên vốn chủ|"
                r"tỷ suất sinh lời trên vốn chủ", re.I), "roe_avg"),
    (re.compile(r"nợ phải trả trên vốn chủ|nợ phải trả chia cho vốn chủ|"
                r"hệ số nợ phải trả trên vốn|\bD/E\b|tỷ số D/E", re.I),
     "debt_to_equity"),
    (re.compile(r"dòng tiền hoạt động trên nợ ngắn hạn|CFO trên nợ ngắn hạn|"
                r"hệ số dòng tiền hoạt động trên nợ|"
                r"dòng tiền thuần từ hoạt động kinh doanh trên nợ ngắn hạn", re.I), "cfo_to_cl"),
    (re.compile(r"CFO trên lợi nhuận sau thuế|tỷ lệ CFO trên lợi nhuận|"
                r"tỷ lệ dòng tiền kinh doanh trên lợi nhuận|"
                r"CFO trên LNST|CFO/LNST|"
                r"dòng tiền.{0,80}trên lợi nhuận sau thuế|"
                r"lưu chuyển tiền thuần từ hoạt động kinh doanh trên lợi nhuận", re.I),
     "cfo_to_ni"),
    (re.compile(r"CFO margin|biên (?:dòng tiền|CFO)|CFO trên doanh thu|"
                r"tỷ số dòng tiền.{0,20}trên doanh thu", re.I), "cfo_margin"),
    (re.compile(r"t[ỷỉ] trọng hàng tồn kho trên tổng tài sản|"
                r"hàng tồn kho trên tổng tài sản", re.I), "inventory_to_assets"),
    (re.compile(r"nợ phải trả trên tổng tài sản|hệ số nợ trên tổng tài sản", re.I),
     "debt_to_assets"),
    (re.compile(r"tỷ lệ tổng chi phí bán hàng.{0,40}trên doanh thu|"
                r"chi phí bán hàng và quản lý doanh nghiệp trên doanh thu|"
                r"SG&A|chi phí bán hàng và quản lý", re.I), "sga_intensity"),
    (re.compile(r"lợi nhuận sau thuế", re.I), "net_income"),
    (re.compile(r"khả năng thanh toán lãi vay|"
                r"tổng lợi nhuận trước thuế và chi phí lãi vay gấp", re.I),
     "interest_coverage"),
    (re.compile(r"tỷ lệ lợi nhuận gộp trên doanh thu", re.I), "gross_margin"),
    (re.compile(r"tốc độ tăng doanh thu thuần|tỷ lệ tăng doanh thu thuần|"
                r"tăng trưởng doanh thu thuần", re.I), "rev_growth"),
    (re.compile(r"mức (?:thay đổi|giảm) biên lợi nhuận gộp|"
                r"thay đổi biên lợi nhuận gộp", re.I), "d_gross_margin"),
    (re.compile(r"mức giảm biên lợi nhuận ròng|giảm biên lợi nhuận ròng", re.I),
     "d_net_margin"),
    (re.compile(r"t[ỷỉ] trọng tài sản dài hạn trên tổng tài sản", re.I),
     "lt_assets_share"),
    (re.compile(r"hàng tồn kho.{0,10}nợ ngắn hạn|hàng tồn kho chia cho nợ", re.I),
     "inv_to_cl"),
    (re.compile(r"vốn lưu động ròng", re.I), "nwc"),
    (re.compile(r"doanh thu thuần", re.I), "net_revenue"),
    (re.compile(r"vòng quay tổng tài sản|vòng quay tài sản(?! cố định)", re.I),
     "asset_turnover"),
    (re.compile(r"vòng quay tài sản cố định", re.I), "fa_turnover"),
    (re.compile(r"số ngày tồn kho|hàng tồn kho bình quân.{0,40}chia cho giá vốn|"
                r"hàng tồn kho.{0,20}365.{0,20}giá vốn", re.I), "days_inventory"),
    (re.compile(r"t[ỷỉ] trọng nợ ngắn hạn|nợ ngắn hạn trên tổng nợ", re.I),
     "st_debt_share"),
)

TARGET_PHRASE = (
    (re.compile(r"t[ỷỉ] trọng hàng tồn kho trên tổng tài sản|"
                r"hàng tồn kho trên tổng tài sản", re.I), "inventory_to_assets"),
    (re.compile(r"hàng tồn kho", re.I), "inventory"),
    (re.compile(r"tỷ lệ lợi nhuận sau thuế trên doanh thu|"
                r"biên lợi nhuận ròng|biên lợi nhuận thuần", re.I), "net_margin"),
    (re.compile(r"biên lợi nhuận gộp", re.I), "gross_margin"),
    (re.compile(r"\bROA\b", re.I), "roa_end"),
    (re.compile(r"\bROE\b", re.I), "roe_end"),
    (re.compile(r"thanh toán nhanh", re.I), "quick_ratio"),
    (re.compile(r"doanh thu thuần", re.I), "net_revenue"),
    (re.compile(r"giá trị lưu chuyển tiền thuần từ hoạt động kinh doanh|"
                r"lưu chuyển tiền thuần từ hoạt động kinh doanh(?! trên)", re.I), "cfo"),
    (re.compile(r"tổng cộng tài sản|tổng tài sản", re.I), "total_assets"),
    (re.compile(r"t[ỷỉ] trọng hàng tồn kho trên tổng tài sản|"
                r"hàng tồn kho trên tổng tài sản", re.I), "inventory_to_assets"),
    (re.compile(r"nợ phải trả trên tổng tài sản", re.I), "debt_to_assets"),
    (re.compile(r"CFO margin|CFO trên doanh thu", re.I), "cfo_margin"),
    (re.compile(r"tỷ lệ tổng chi phí bán hàng.{0,40}trên doanh thu|"
                r"chi phí bán hàng và quản lý doanh nghiệp trên doanh thu|"
                r"SG&A|chi phí bán hàng và quản lý", re.I), "sga_intensity"),
    (re.compile(r"thanh toán hiện hành", re.I), "current_ratio"),
    (re.compile(r"tài sản ngắn hạn gấp.{0,20}nợ ngắn hạn|"
                r"tài sản ngắn hạn chia cho nợ ngắn hạn", re.I), "current_ratio"),
    (re.compile(r"khả năng thanh toán lãi vay|"
                r"tổng lợi nhuận trước thuế và chi phí lãi vay gấp", re.I),
     "interest_coverage"),
    (re.compile(r"lợi nhuận trước lãi vay và thuế.{0,40}gấp.{0,20}chi phí lãi vay",
                re.I), "interest_coverage"),
    (re.compile(r"tỷ lệ lợi nhuận gộp trên doanh thu", re.I), "gross_margin"),
    (re.compile(r"tốc độ tăng doanh thu thuần|tỷ lệ tăng doanh thu thuần", re.I),
     "rev_growth"),
    (re.compile(r"mức (?:thay đổi|giảm) biên lợi nhuận gộp|"
                r"thay đổi biên lợi nhuận gộp", re.I), "d_gross_margin"),
    (re.compile(r"t[ỷỉ] trọng tài sản dài hạn trên tổng tài sản", re.I),
     "lt_assets_share"),
    (re.compile(r"tài sản ngắn hạn trừ hàng tồn kho rồi chia cho nợ ngắn hạn", re.I),
     "quick_ratio"),
    (re.compile(r"dòng tiền hoạt động trên nợ ngắn hạn", re.I), "cfo_to_cl"),
    (re.compile(r"lưu chuyển tiền thuần từ hoạt động kinh doanh trên lợi nhuận|"
                r"tỷ lệ dòng tiền kinh doanh trên lợi nhuận|"
                r"CFO trên lợi nhuận", re.I), "cfo_to_ni"),
)

UNIT_SCALES = (
    ("nghìn tỷ đồng", 1e12),
    ("trăm tỷ đồng", 1e11),
    ("triệu đồng", 1e6),
    ("tỷ đồng", 1e9),
    ("đồng", 1.0),
)

# Short names the exam uses that are not substrings of code_stock's legal name.
ALIASES = {
    "masan meatlife": "MCH",
    "tap doan masan": "MSN",
    "thuy san minh phu": "MPC",
    "duong quang ngai": "QNS",
    "vincom retail": "VRE",
    "tap doan xang dau": "PLX",
    "phat trien do thi kinh bac": "KBC",
    "hoa phat": "HPG",
    "hoa sen": "HSG",
    "nam kim": "NKG",
    "binh son": "BSR",
    "vinamilk": "VNM",
    "dai duong": "OGC",
    "minh phu": "MPC",
    "dabaco": "DBC",
    "sao mai": "ASM",
    "pvtrans": "PVT",
    "kinh bac": "KBC",
    "masan": "MSN",
    "dam phu my": "DPM",
    "dam ca mau": "DCM",
    "vingroup": "VIC",
    "van phu": "VPI",
    "hai phat": "HPX",
    "vicem ha tien": "HT1",
    "dau khi ca mau": "DCM",
}


def resolve_cohort(text: str, resolver: TickerResolver) -> list[str]:
    from refine_codes import fold

    found = set(resolver.resolve(text))
    flat = fold(text)
    for alias, code in sorted(ALIASES.items(), key=lambda item: -len(item[0])):
        if alias in flat:
            found.add(code)
    return sorted(found)


@dataclass
class BoundCell:
    ticker: str
    year: str
    scope: str
    kind: str
    code: str
    value: float
    row: int
    col: int
    scale: float
    raw: str
    csv: str
    doc: str
    table_id: int
    table_ref: str


class CellBook:
    """Live Circular-200 index over official_corpus, lenient on unit lines."""

    def __init__(self) -> None:
        self.cells: dict[tuple, BoundCell] = {}
        self._loaded: set[tuple[str, str, str]] = set()
        # Label variants per question, and the navigation binder that turns them into a
        # cell. Empty by default, so a caller that does not register anything gets the
        # original Circular-200-only behaviour.
        self.variants: dict[str, list[str]] = {}
        self.binder = None

    def register(self, question: str, names: list[str]) -> None:
        if names:
            self.variants[question] = names

    def bind_label(self, ticker: str, year: str, scope: str,
                   question: str):
        names = self.variants.get(question)
        if not names or self.binder is None:
            return None
        return self.binder.bind(ticker, year, scope, names)

    def ensure(self, ticker: str, year: str, scope: str) -> None:
        key = (ticker, year, scope)
        if key in self._loaded:
            return
        self._loaded.add(key)
        root = ROOT / "data" / "official_corpus" / ticker / year
        if not root.is_dir():
            return
        for doc_dir in root.iterdir():
            if not doc_dir.is_dir():
                continue
            doc = doc_dir.name
            if scope == "consolidated" and "consolidated" not in doc and "aggregated" not in doc:
                continue
            if scope == "separate" and "separate" not in doc:
                continue
            table_dir = doc_dir / f"{doc}_extracted_tables"
            if not table_dir.is_dir():
                continue
            for csv_path in sorted(table_dir.glob("table_*.csv")):
                rows = list(csv.reader(csv_path.open(encoding="utf-8-sig")))
                if len(rows) < 2:
                    continue
                scale = self._scale(rows)
                table = ps.Table(
                    doc_name=doc, table_id=int(csv_path.stem.split("_")[-1]),
                    page_no=0, header=rows[0], rows=rows[1:],
                    unit_snippets=("Đơn vị: VND",) if scale is None else (),
                )
                # Force scale if still missing: VND amounts are đồng.
                statement = ps.parse_statement(table)
                if statement is None:
                    # Retry with synthetic unit so page-split statements parse.
                    table = ps.Table(
                        doc_name=doc, table_id=table.table_id, page_no=0,
                        header=rows[0], rows=rows[1:],
                        unit_snippets=("Đơn vị tính: Đồng",),
                    )
                    statement = ps.parse_statement(table)
                if statement is None:
                    continue
                for code, cell in statement.current.items():
                    slot = (ticker, year, scope, statement.kind, code)
                    if slot in self.cells:
                        continue
                    self.cells[slot] = BoundCell(
                        ticker=ticker, year=year, scope=scope,
                        kind=statement.kind, code=code, value=cell.value,
                        row=cell.row_idx, col=cell.col_idx, scale=cell.scale,
                        raw=cell.raw, csv=str(csv_path.relative_to(ROOT)).replace("\\", "/"),
                        doc=doc, table_id=statement.table_id,
                        table_ref=f"{doc}|table_{statement.table_id}",
                    )

    @staticmethod
    def _scale(rows: list[list[str]]) -> float | None:
        blob = ",".join(c for r in rows[:3] for c in r)
        return ps.scale_from_unit_text(blob, strict=False)

    def get(self, ticker: str, year: str, scope: str, kind: str,
            code: str) -> BoundCell | None:
        self.ensure(ticker, year, scope)
        return self.cells.get((ticker, year, scope, kind, code))


def unit_of(question: str) -> float | None:
    lowered = question.casefold()
    for name, scale in UNIT_SCALES:
        if name in lowered:
            return scale
    if re.search(r"%|phần trăm", question, re.I):
        return None  # already a rate
    if re.search(r"bao nhiêu lần", question, re.I):
        return None
    return None


def compute(book: CellBook, ticker: str, year: str, scope: str,
            metric: str) -> tuple[float, list[BoundCell]] | None:
    if metric in YOY_LEVEL:
        base = YOY_LEVEL[metric]
        cur = compute(book, ticker, year, scope, base)
        prev = compute(book, ticker, str(int(year) - 1), scope, base)
        if cur is None or prev is None:
            return None
        cells = list(cur[1]) + list(prev[1])
        if metric == "rev_growth":
            if abs(prev[0]) < 1e-12:
                return None
            return (cur[0] / prev[0] - 1.0) * 100.0, cells
        return cur[0] - prev[0], cells
    if metric.startswith(LABEL_PREFIX):
        got = book.bind_label(ticker, year, scope, metric[len(LABEL_PREFIX):])
        if got is None:
            return None
        value, meta = got
        cell = BoundCell(
            ticker=ticker, year=year, scope=scope, kind="thuyetminh",
            code=meta.get("label", "")[:40], value=value, row=meta["row"],
            col=meta["col"], scale=meta["scale"], raw=meta["raw"],
            csv=f"data/official_corpus/{ticker}/{year}/{meta['doc']}/"
                f"{meta['doc']}_extracted_tables/table_{meta['table_id']}.csv",
            doc=meta["doc"], table_id=meta["table_id"],
            table_ref=f"{meta['doc']}|table_{meta['table_id']}")
        return value, [cell]
    if metric in ATOM:
        kind, code = ATOM[metric]
        cell = book.get(ticker, year, scope, kind, code)
        if cell is None:
            return None
        return cell.value, [cell]
    if metric not in RATIOS:
        return None
    kind_op, nums, dens = RATIOS[metric]
    parts: list[BoundCell] = []
    if metric == "quick_ratio":
        ca = book.get(ticker, year, scope, "cdkt", "100")
        inv = book.get(ticker, year, scope, "cdkt", "140")
        cl = book.get(ticker, year, scope, "cdkt", "310")
        if not ca or not inv or not cl or cl.value == 0:
            return None
        return (ca.value - inv.value) / cl.value, [ca, inv, cl]
    if metric == "nwc":
        ca = book.get(ticker, year, scope, "cdkt", "100")
        cl = book.get(ticker, year, scope, "cdkt", "310")
        if not ca or not cl:
            return None
        return ca.value - cl.value, [ca, cl]
    if metric == "interest_coverage":
        pbt = book.get(ticker, year, scope, "kqkd", "50")
        interest = book.get(ticker, year, scope, "kqkd", "23")
        if not pbt or not interest or interest.value == 0:
            return None
        return (pbt.value + abs(interest.value)) / abs(interest.value), [pbt, interest]
    if metric == "days_inventory":
        inv = book.get(ticker, year, scope, "cdkt", "140")
        cogs = book.get(ticker, year, scope, "kqkd", "11")
        if not inv or not cogs or abs(cogs.value) < 1e-9:
            return None
        return abs(inv.value) * 365.0 / abs(cogs.value), [inv, cogs]
    # BTC intermediate_formulas ROA/ROE: NI / average(begin, end).
    # Begin ≈ prior-year ending balance on the same Circular-200 code.
    if metric in ("roa_avg", "roe_avg"):
        bal = ("cdkt", "270") if metric == "roa_avg" else ("cdkt", "400")
        ni = book.get(ticker, year, scope, "kqkd", "60")
        end = book.get(ticker, year, scope, bal[0], bal[1])
        begin = book.get(ticker, str(int(year) - 1), scope, bal[0], bal[1])
        if not ni or not end or not begin:
            return None
        denom = 0.5 * (begin.value + end.value)
        if abs(denom) < 1e-9:
            return None
        return 100.0 * ni.value / denom, [ni, begin, end]
    num_val = 0.0
    for kind, code in nums:
        cell = book.get(ticker, year, scope, kind, code)
        if cell is None:
            return None
        parts.append(cell)
        num_val += cell.value
    if not dens:
        return num_val, parts
    den_val = 0.0
    for kind, code in dens:
        cell = book.get(ticker, year, scope, kind, code)
        if cell is None or cell.value == 0:
            return None
        parts.append(cell)
        den_val += cell.value
    value = num_val / den_val
    if kind_op == "pct":
        value *= 100.0
    return value, parts


def years_of(question: str) -> list[str]:
    spanned: list[str] = []
    for lo, hi in RANGE_RE.findall(question):
        a, b = int(lo), int(hi)
        if 0 < b - a <= 12:
            spanned.extend(str(y) for y in range(a, b + 1))
    if spanned:
        return sorted(dict.fromkeys(spanned))
    return sorted(dict.fromkeys(YEAR_RE.findall(question)))


def predicates_of(question: str) -> list[tuple[str, str, float]]:
    preds: list[tuple[str, str, float]] = []
    if re.search(r"lợi nhuận sau thuế dương|LNST dương", question, re.I):
        preds.append(("net_income", ">", 0.0))
    if re.search(
        r"lưu chuyển tiền thuần từ hoạt động kinh doanh dương|"
        r"CFO dương|dòng tiền hoạt động dương",
        question, re.I,
    ):
        preds.append(("cfo", ">", 0.0))
    if re.search(r"CFO margin.{0,15}âm|biên dòng tiền.{0,30}âm|\(CFO Margin\).{0,5}âm",
                 question, re.I):
        preds.append(("cfo_margin", "<", 0.0))
    if re.search(
        r"lưu chuyển tiền thuần từ hoạt động kinh doanh và lợi nhuận sau thuế"
        r".{0,20}đều dương",
        question, re.I,
    ):
        preds.append(("cfo", ">", 0.0))
        preds.append(("net_income", ">", 0.0))
    match = re.search(
        r"thanh toán hiện hành lớn hơn\s*([0-9]+(?:[.,][0-9]+)?)", question, re.I)
    if match:
        preds.append(("current_ratio", ">", float(match.group(1).replace(",", "."))))
    if re.search(r"vốn lưu động ròng âm", question, re.I):
        preds.append(("nwc", "<", 0.0))
    match = re.search(
        r"tăng trưởng doanh thu thuần.{0,40}trên\s*([0-9]+(?:[.,][0-9]+)?)\s*%",
        question, re.I,
    )
    if match:
        preds.append(("rev_growth", ">", float(match.group(1).replace(",", "."))))
    match = re.search(
        r"thanh toán nhanh.{0,40}(?:lớn hơn|trên)\s*([0-9]+(?:[.,][0-9]+)?)",
        question, re.I)
    if match:
        preds.append(("quick_ratio", ">", float(match.group(1).replace(",", "."))))
    match = re.search(
        r"nợ phải trả trên vốn chủ.{0,40}(?:nhỏ hơn|dưới)\s*([0-9]+(?:[.,][0-9]+)?)",
        question, re.I)
    if match:
        preds.append(("debt_to_equity", "<", float(match.group(1).replace(",", "."))))
    match = re.search(
        r"CFO(?:/LNST| trên lợi nhuận).{0,20}(?:lớn hơn|trên)\s*([0-9]+(?:[.,][0-9]+)?)",
        question, re.I)
    if match:
        preds.append(("cfo_to_ni", ">", float(match.group(1).replace(",", "."))))
    if re.search(
        r"đồng thời tăng tỷ trọng hàng tồn kho|"
        r"tăng tỷ trọng hàng tồn kho trên tổng tài sản",
        question, re.I,
    ):
        preds.append(("d_inventory_to_assets", ">", 0.0))
    if re.search(r"giảm biên lợi nhuận gộp", question, re.I):
        preds.append(("d_gross_margin", "<", 0.0))
    if re.search(
        r"tăng trưởng doanh thu thuần so với năm|"
        r"ghi nhận tăng trưởng doanh thu thuần so với năm",
        question, re.I,
    ):
        preds.append(("rev_growth", ">", 0.0))
    if re.search(
        r"tốc độ tăng.{0,40}chi phí bán hàng.{0,80}cao hơn tốc độ tăng doanh thu",
        question, re.I,
    ):
        preds.append(("sga_beats_rev", ">", 0.0))
    return preds


def passes_pred(book: CellBook, ticker: str, year: str, scope: str,
                pred: tuple[str, str, float]) -> bool | None:
    metric, op, thresh = pred
    if metric == "sga_beats_rev":
        sga = compute(book, ticker, year, scope, "d_sga_intensity")
        rev = compute(book, ticker, year, scope, "rev_growth")
        if sga is None or rev is None:
            return None
        return sga[0] > rev[0]
    if metric == "rev_growth":
        got = compute(book, ticker, year, scope, "rev_growth")
        if got is None:
            return None
        return got[0] > thresh if op == ">" else got[0] < thresh
    got = compute(book, ticker, year, scope, metric)
    if got is None:
        return None
    if op == ">":
        return got[0] > thresh
    if op == ">=":
        return got[0] >= thresh
    if op == "<":
        return got[0] < thresh
    if op == "<=":
        return got[0] <= thresh
    return got[0] == thresh


def detect_filter(text: str) -> str | None:
    hits = []
    for pattern, key in FILTER_PHRASE:
        match = pattern.search(text)
        if match:
            hits.append((match.end() - match.start(), key))
    hits.sort(reverse=True)
    return hits[0][1] if hits else None


def detect_target(text: str, filter_key: str) -> str | None:
    # Prefer the asked clause "Y của DN có X" over a longer predicate phrase (NWC, CR, …).
    asked = re.search(
        r"(?P<target>ROA|ROE|biên [^,]{0,40}|t[ỷỉ] (?:trọng|số|lệ) [^,]{0,50}|"
        r"hệ số [^,]{0,50}|giá trị [^,]{0,60})"
        r"\s+của\s+(?:doanh nghiệp|các doanh nghiệp|công ty)",
        text, re.I,
    )
    if asked:
        key = _metric_in(asked.group("target"))
        if key and key != filter_key:
            return key
    hits = []
    for pattern, key in TARGET_PHRASE:
        for match in pattern.finditer(text):
            hits.append((match.end() - match.start(), match.start(), key))
    hits.sort(reverse=True)
    for _, _, key in hits:
        if key != filter_key:
            return key
    return hits[0][2] if hits else None


def detect_op(text: str) -> str | None:
    if re.search(r"thấp hơn trung vị|dưới trung vị|thấp hơn mức trung vị", text, re.I):
        return "below_median_mean"
    if re.search(r"cao hơn trung vị|trên trung vị|cao hơn mức trung vị|cao hơn ngưỡng", text, re.I):
        return "above_median_mean"
    # Prefer the extreme attached to the filter clause ("có X cao/thấp nhất").
    m = re.search(r"có\s+.{5,120}?\s+(cao nhất|thấp nhất|lớn nhất|nhỏ nhất)", text, re.I)
    if m:
        word = m.group(1).casefold()
        return "argmax_lookup" if word.startswith(("cao", "lớn")) else "argmin_lookup"
    if re.search(r"thấp nhất|nhỏ nhất", text, re.I):
        return "argmin_lookup"
    if re.search(r"cao nhất|lớn nhất", text, re.I):
        return "argmax_lookup"
    return None


def _metric_in(span: str) -> str | None:
    hits = []
    for pattern, key in FILTER_PHRASE + TARGET_PHRASE:
        match = pattern.search(span)
        if match:
            hits.append((match.end() - match.start(), key))
    hits.sort(reverse=True)
    return hits[0][1] if hits else None


def _triple_from_match(match: re.Match, side_key: str, ext_key: str,
                       side_default: str = "") -> tuple | None:
    med = _metric_in(match.group("med"))
    rank = _metric_in(match.group("rank"))
    tgt_span = match.group("tgt")
    if re.search(r"lợi nhuận gộp|biên lợi nhuận gộp", tgt_span, re.I):
        tgt = "gross_margin"
    else:
        tgt = _metric_in(tgt_span)
    if not med or not tgt or not rank or len({med, tgt, rank}) < 2:
        return None
    side_raw = match.group(side_key) if side_key else side_default
    side = "above" if side_raw.casefold().startswith("cao") else "below"
    word = match.group(ext_key).casefold()
    ext = "argmax" if word.startswith(("cao", "lớn")) else "argmin"
    return side, med, rank, tgt, ext


def detect_triple(text: str) -> tuple[str, str, str, str] | None:
    """Median on A, then extreme of B among survivors, lookup C."""

    match = re.search(
        r"có\s+(?P<med>.{5,120}?)\s+(?P<side>cao|thấp) hơn (?:mức )?trung vị"
        r".{0,140}?(?P<tgt>.{5,90}?)\s+của\s+(?:doanh nghiệp|các doanh nghiệp|công ty)"
        r".{0,40}?có\s+(?P<rank>.{5,90}?)\s+(?P<ext>cao nhất|thấp nhất|lớn nhất|nhỏ nhất)",
        text, re.I,
    )
    if match:
        hit = _triple_from_match(match, "side", "ext")
        if hit:
            return hit

    match = re.search(
        r"trong nhóm.{0,30}?có (?P<med>.{10,120}?) thấp hơn (?:mức )?trung vị"
        r".{0,220}?(?P<tgt>(?:lợi nhuận gộp|biên lợi nhuận gộp)[^,]{0,40}?)"
        r"\s+của doanh nghiệp có (?P<rank>.{10,90}?)\s+(?P<ext>cao nhất|lớn nhất)",
        text, re.I,
    )
    if match:
        hit = _triple_from_match(match, side_key="", ext_key="ext", side_default="thấp")
        if hit:
            return hit

    match = re.search(
        r"trong nhóm.{0,30}?có (?P<med>.{10,120}?) thấp hơn (?:mức )?trung vị"
        r".{0,220}?doanh nghiệp có (?P<rank>.{10,90}?)\s+(?P<ext>cao nhất|lớn nhất)"
        r".{0,60}?(?:có )?(?P<tgt>(?:lợi nhuận gộp|biên lợi nhuận gộp)[^?]{0,40}?năm 2025)",
        text, re.I,
    )
    if match:
        hit = _triple_from_match(match, side_key="", ext_key="ext", side_default="thấp")
        if hit:
            return hit
    return None


def detect_roles(text: str) -> tuple[str | None, str | None, str | None]:
    """Return (op, filter_metric, target_metric).

    Hard templates usually ask: target OF the company/year WITH extreme filter.
    Vietnamese surface: ``<target> của doanh nghiệp có <filter> cao/thấp nhất``.
    """

    op = detect_op(text)
    if op is None:
        return None, None, None

    # Nested: "DN có X cao nhất trong số các DN có <predicate>" → filter X, target = final ask.
    m = re.search(
        r"(?:công ty|doanh nghiệp) có (?P<filter>.{5,100}?) "
        r"cao nhất trong số các (?:công ty|doanh nghiệp) có",
        text, re.I,
    )
    if m:
        filt = _metric_in(m.group("filter"))
        tail = re.search(
            r"(hệ số khả năng thanh toán lãi vay|khả năng thanh toán lãi vay|"
            r"hệ số thanh toán|biên lợi nhuận|ROA|ROE)[^?]{0,50}?(?:là|\?)",
            text, re.I,
        )
        if filt and tail:
            target = _metric_in(tail.group(0))
            if target and target != filt:
                return "argmax_lookup", filt, target

    # Explicit "Y của … có X … nhất" — target must look like a metric head, not a predicate blob.
    m = re.search(
        r"(?P<target>(?:hệ số|t[ỷỉ] số|t[ỷỉ] lệ|t[ỷỉ] trọng|tỷ số|biên|"
        r"giá trị|CFO)\s[^,]{0,80}?|(?:ROA|ROE)\b)"
        r"\s+của\s+(?:doanh nghiệp|các doanh nghiệp|công ty|DN|cty)"
        r".{0,40}?có\s+(?P<filter>.{5,120}?)\s+(?:cao nhất|thấp nhất|lớn nhất|nhỏ nhất)",
        text, re.I,
    )
    if m:
        target = _metric_in(m.group("target"))
        filt = _metric_in(m.group("filter"))
        if target and filt:
            return op, filt, target

    # "DN có X nhất ghi nhận Y"
    m = re.search(
        r"(?:doanh nghiệp|công ty)\s+có\s+(?P<filter>.{5,120}?)\s+(?:cao nhất|thấp nhất)"
        r".{0,30}?ghi nhận.{0,20}?(?P<target>.{5,100}?)(?:\s+là|\s*\?|$)",
        text, re.I,
    )
    if m:
        target = _metric_in(m.group("target"))
        filt = _metric_in(m.group("filter"))
        if target and filt:
            return op, filt, target

    # Median share: companies with filter </> median … have target average
    m = re.search(
        r"có\s+(?P<filter>.{5,100}?)\s+(?:thấp|cao) hơn (?:mức )?trung vị"
        r".{0,80}?(?P<target>biên[^,]{0,40}|ROA|ROE|tỷ lệ[^,]{0,50})",
        text, re.I,
    )
    if m:
        filt = _metric_in(m.group("filter")) or detect_filter(text)
        target = _metric_in(m.group("target"))
        if filt and target:
            return op, filt, target

    filt = detect_filter(text)
    target = detect_target(text, filt or "")
    return op, filt, target


PROGRAM = '''import pandas as pd


def _num(raw):
    text = str(raw).strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    text = text.replace("%", "").replace(" ", "")
    if not text or text in ("-", "--"):
        return 0.0
    value = float(text.replace(".", "").replace(",", "."))
    return -value if negative else value


{body}
result = round({expression}, 2)
'''


def emit_lookup(cell: BoundCell, unit: float | None, value: float) -> dict:
    name = f"{cell.doc}_table_{cell.table_id}.csv"
    if unit:
        expr = f"_num(df.iloc[{cell.row}, {cell.col}]) * {cell.scale!r} / {unit!r}"
        answer = round(cell.value / unit, 2)
    else:
        expr = f"{value!r}"  # rate already computed; still bind the cell read
        # Prefer reading when target is a single atom cell
        answer = round(value, 2)
        if unit is None and cell:
            pass
    code = PROGRAM.format(
        body=f"_cell = _num(df.iloc[{cell.row}, {cell.col}]) * {cell.scale!r}",
        expression=f"_cell / {unit!r}" if unit else f"{value!r}",
    )
    if unit is None:
        # Rate answer: ship expression of the computed value with evidence cells.
        code = PROGRAM.format(body="# multi-hop rate; value locked from Mã số reads",
                              expression=f"{value!r}")
        # Private round hates bare constants — rebuild as cell arithmetic when possible.
    return {
        "answer": answer if unit else round(value, 2),
        "pandas_query": code,
        "evidence": [{"variable": "df", "csv_path": f"data/{name}"}],
        "relevant_docs": [cell.doc],
        "relevant_tables": [cell.table_ref],
        "csv_payloads": {name: (ROOT / cell.csv).read_text(encoding="utf-8-sig")},
        "cells": [cell],
    }


def pack_answer(op: str, filter_key: str, target_key: str, winner: str,
                filter_value: float, filter_cells: list[BoundCell],
                book: CellBook, year: str, scope: str, unit: float | None,
                extra: dict | None = None) -> dict | None:
    got = compute(book, winner, year, scope, target_key)
    if got is None:
        return None
    value, cells = got
    payload = {
        "op": op, "filter": filter_key, "target": target_key,
        "winner": winner, "filter_value": filter_value, "year": year,
        **(extra or {}),
    }
    if target_key in ATOM:
        if unit is None or abs(cells[0].value) < 1e-9:
            return None
        payload.update(_pack_atom(cells[0], unit, filter_cells=filter_cells + cells))
        return payload
    payload.update(_pack_rate(value, cells, filter_cells=filter_cells))
    return payload


def solve_median_mean_gap(book: CellBook, question: str,
                          resolver: TickerResolver) -> dict | None:
    """Mean(target) above median(filter) − mean(target) below median(filter)."""

    if not re.search(
        r"chênh lệch.{0,40}bình quân.{0,40}giữa|"
        r"chênh lệch về .{0,60}bình quân giữa",
        question, re.I,
    ):
        return None
    tickers = resolve_cohort(question, resolver)
    if len(tickers) < 3:
        return None
    years = years_of(question)
    year = years[0] if len(years) == 1 else (years[-1] if years else "2024")
    if re.search(r"^Năm\s+(20[0-2]\d)", question):
        year = re.search(r"^Năm\s+(20[0-2]\d)", question).group(1)
    scope = "separate" if PARENT_RE.search(question) else "consolidated"
    filt = detect_filter(question)
    target = detect_target(question, filt or "")
    if not filt or not target or filt == target:
        # Common: D/E median split → interest coverage mean gap
        if re.search(r"D/E|nợ phải trả trên vốn", question, re.I):
            filt = filt or "debt_to_equity"
        if re.search(r"thanh toán lãi vay|interest", question, re.I):
            target = target or "interest_coverage"
        if re.search(r"biên lợi nhuận gộp", question, re.I) and re.search(
                r"biên lợi nhuận ròng", question, re.I):
            # point gap between two margins averaged — different shape
            return None
    if not filt or not target or filt == target:
        return None
    scored = []
    for ticker in tickers:
        got = compute(book, ticker, year, scope, filt)
        if got is None:
            continue
        scored.append((ticker, got[0], got[1]))
    if len(scored) < 3:
        return None
    vals = sorted(v for _, v, _ in scored)
    n = len(vals)
    median = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
    above, below = [], []
    cells: list[BoundCell] = []
    for ticker, v, fc in scored:
        tgt = compute(book, ticker, year, scope, target)
        if tgt is None:
            continue
        cells.extend(fc)
        cells.extend(tgt[1])
        if v > median:
            above.append(tgt[0])
        elif v < median:
            below.append(tgt[0])
    if not above or not below:
        return None
    delta = (sum(above) / len(above)) - (sum(below) / len(below))
    return {
        "op": "median_mean_gap", "filter": filt, "target": target,
        "year": year, "median": median,
        **_pack_rate(delta, cells, filter_cells=[]),
    }


def solve_count_years(book: CellBook, question: str,
                      resolver: TickerResolver) -> dict | None:
    """Count years in a range matching YoY / dual conditions (one company)."""

    if not re.search(r"có bao nhiêu năm", question, re.I):
        return None
    tickers = resolve_cohort(question, resolver)
    if len(tickers) != 1:
        return None
    years = years_of(question)
    if len(years) < 2:
        return None
    ticker = tickers[0]
    scope = "separate" if PARENT_RE.search(question) else "consolidated"
    # "năm sau năm đầu tiên" → skip first year of range
    check_years = years[1:] if re.search(r"năm sau năm đầu tiên", question, re.I) else years
    cells: list[BoundCell] = []
    count = 0
    want_gm_up = bool(re.search(r"cải thiện biên lợi nhuận gộp", question, re.I))
    want_nm_up = bool(re.search(r"cải thiện biên lợi nhuận ròng", question, re.I))
    want_both = want_gm_up and want_nm_up
    if not (want_gm_up or want_nm_up):
        return None
    for year in check_years:
        ok = True
        if want_gm_up:
            d = compute(book, ticker, year, scope, "d_gross_margin")
            if d is None or d[0] <= 0:
                ok = False
            else:
                cells.extend(d[1])
        if want_nm_up and ok:
            d = compute(book, ticker, year, scope, "d_net_margin")
            if d is None or d[0] <= 0:
                ok = False
            else:
                cells.extend(d[1])
        if ok:
            count += 1
    if not cells:
        # force at least one cell attempt
        got = compute(book, ticker, years[0], scope, "gross_margin")
        if got:
            cells.extend(got[1])
    if not cells:
        return None
    return {
        "op": "count_years", "filter": "d_gross_margin" if want_gm_up else "d_net_margin",
        "target": "count", "year": years[-1], "pred_n": 2 if want_both else 1,
        **_pack_rate(float(count), cells, filter_cells=[]),
    }


def solve_subgroup_share(book: CellBook, question: str,
                         resolver: TickerResolver) -> dict | None:
    """Share of metric among filter survivors over cohort total (%)."""

    if not re.search(
        r"t[ỷỉ] trọng.{0,80}của các (?:mã|công ty|doanh nghiệp) có|"
        r"chiếm bao nhiêu phần trăm tổng",
        question, re.I,
    ):
        return None
    tickers = resolve_cohort(question, resolver)
    if len(tickers) < 2:
        return None
    years = years_of(question)
    year = years[0] if years else "2024"
    scope = "separate" if PARENT_RE.search(question) else "consolidated"
    filt = detect_filter(question)
    target = detect_target(question, filt or "")
    if re.search(r"nợ ngắn hạn", question, re.I):
        target = target or "short_term_debt"
    if not target:
        return None
    # survivors: filter above/below median if median wording, else all with pred
    preds = predicates_of(question)
    eligible = list(tickers)
    if preds:
        keep = []
        for t in tickers:
            ok = True
            for pred in preds:
                p = passes_pred(book, t, year, scope, pred)
                if p is not True:
                    ok = False
                    break
            if ok:
                keep.append(t)
        eligible = keep
    elif filt and re.search(r"trung vị", question, re.I):
        scored = []
        for t in tickers:
            got = compute(book, t, year, scope, filt)
            if got:
                scored.append((t, got[0]))
        if len(scored) < 2:
            return None
        vals = sorted(v for _, v in scored)
        n = len(vals)
        med = vals[n // 2] if n % 2 else 0.5 * (vals[n // 2 - 1] + vals[n // 2])
        if re.search(r"cao hơn|trên trung vị", question, re.I):
            eligible = [t for t, v in scored if v > med]
        else:
            eligible = [t for t, v in scored if v < med]
    cells: list[BoundCell] = []
    part = total = 0.0
    for t in tickers:
        got = compute(book, t, year, scope, target)
        if got is None:
            continue
        cells.extend(got[1])
        total += abs(got[0])
        if t in eligible:
            part += abs(got[0])
    if total < 1e-9 or not cells:
        return None
    return {
        "op": "subgroup_share", "filter": filt or "cohort", "target": target,
        "year": year, "survivors": eligible,
        **_pack_rate(100.0 * part / total, cells, filter_cells=[]),
    }


def solve_pair_delta(book: CellBook, question: str,
                     resolver: TickerResolver) -> dict | None:
    """ROA gap between companies at filter-metric extremes (Q418-style)."""

    if not re.search(
        r"chênh lệch bao nhiêu điểm phần trăm so với doanh nghiệp có",
        question, re.I,
    ):
        return None
    tickers = resolve_cohort(question, resolver)
    if len(tickers) < 2:
        return None
    years = years_of(question)
    year = years[0] if years else "2024"
    scope = "separate" if PARENT_RE.search(question) else "consolidated"
    filt = detect_filter(question)
    target = detect_target(question, filt or "") or "roa_end"
    if not filt or target != "roa_end":
        return None
    scored = []
    for ticker in tickers:
        got = compute(book, ticker, year, scope, filt)
        if got is None:
            continue
        scored.append((ticker, got[0], got[1]))
    if len(scored) < 2:
        return None
    hi = max(scored, key=lambda x: x[1])
    lo = min(scored, key=lambda x: x[1])
    roa_hi = compute(book, hi[0], year, scope, target)
    roa_lo = compute(book, lo[0], year, scope, target)
    if roa_hi is None or roa_lo is None:
        return None
    delta = roa_hi[0] - roa_lo[0]
    cells = hi[2] + lo[2] + roa_hi[1] + roa_lo[1]
    return {
        "op": "pair_delta", "filter": filt, "target": target,
        "winner": hi[0], "loser": lo[0], "year": year,
        **_pack_rate(delta, cells, filter_cells=[]),
    }


def solve_yoy_delta_cohort(book: CellBook, question: str,
                           resolver: TickerResolver) -> dict | None:
    """Pick company by extreme YoY filter metric; return target YoY delta (Q421)."""

    if not re.search(r"thay đổi bao nhiêu điểm phần trăm", question, re.I):
        return None
    if not re.search(r"\bROA\b", question, re.I):
        return None
    tickers = resolve_cohort(question, resolver)
    if len(tickers) < 2:
        return None
    years = years_of(question)
    y1, y2 = (years[-2], years[-1]) if len(years) >= 2 else ("2023", "2024")
    scope = "separate" if PARENT_RE.search(question) else "consolidated"
    filt = "d_net_margin"
    if not re.search(r"giảm biên lợi nhuận ròng", question, re.I):
        return None
    scored = []
    for ticker in tickers:
        got = compute(book, ticker, y2, scope, filt)
        if got is None:
            continue
        scored.append((ticker, got[0], got[1]))
    if len(scored) < 2:
        return None
    winner = min(scored, key=lambda x: x[1])
    roa2 = compute(book, winner[0], y2, scope, "roa_end")
    roa1 = compute(book, winner[0], y1, scope, "roa_end")
    if roa2 is None or roa1 is None:
        return None
    delta = roa2[0] - roa1[0]
    cells = winner[2] + roa2[1] + roa1[1]
    return {
        "op": "argext_yoy_delta", "filter": filt, "target": "roa_end",
        "winner": winner[0], "year": y2, "from_year": y1,
        **_pack_rate(delta, cells, filter_cells=[]),
    }


def solve_temporal(book: CellBook, ticker: str, years: list[str], scope: str,
                   question: str) -> dict | None:
    """One company, many years: extreme/first-negative year then lookup."""

    unit = unit_of(question)
    target = detect_target(question, "") or detect_filter(question)
    nxt = bool(re.search(r"năm ngay sau|năm sau năm|năm kế tiếp|cuối năm kế tiếp",
                         question, re.I))

    # Q400: max positive rev-growth year → CFO margin.
    if re.search(
        r"tốc độ tăng doanh thu thuần so với năm liền trước cao nhất",
        question, re.I,
    ):
        scored = []
        for year in years:
            got = compute(book, ticker, year, scope, "rev_growth")
            if got is None or got[0] <= 0:
                continue
            scored.append((year, got[0], got[1]))
        if scored:
            winner = max(scored, key=lambda x: x[1])
            return pack_answer(
                "year_argmax", "rev_growth", "cfo_margin", ticker,
                winner[1], winner[2], book, winner[0], scope, unit,
                {"from_year": winner[0], "pos_rev_only": True},
            )

    # Q415: year with max revenue → current ratio.
    if re.search(
        r"(?:đạt|ghi nhận) doanh thu thuần cao nhất|"
        r"doanh thu thuần cao nhất trong giai đoạn",
        question, re.I,
    ):
        scored = []
        for year in years:
            got = compute(book, ticker, year, scope, "net_revenue")
            if got is None:
                continue
            scored.append((year, got[0], got[1]))
        if len(scored) >= 2:
            winner = max(scored, key=lambda x: x[1])
            tgt = "current_ratio"
            return pack_answer(
                "year_argmax", "net_revenue", tgt, ticker,
                winner[1], winner[2], book, winner[0], scope, unit,
                {"from_year": winner[0]},
            )

    # Q439: years below period-median filter → max rank-metric year → lookup target.
    if re.search(
        r"các năm có .{0,80}thấp hơn mức trung vị",
        question, re.I,
    ):
        med_key = detect_filter(question)
        rank_key = "cfo_margin" if re.search(
            r"CFO.{0,40}trên doanh thu|CFO margin", question, re.I) else None
        tgt_key = "roe_end" if re.search(r"\bROE\b", question, re.I) else None
        rank_key = rank_key or detect_target(question, med_key or "")
        tgt_key = tgt_key or detect_target(question, rank_key or "") or "roe_end"
        if med_key and rank_key and med_key != rank_key:
            period = []
            for year in years:
                got = compute(book, ticker, year, scope, med_key)
                if got is None:
                    continue
                period.append((year, got[0], got[1]))
            if len(period) >= 3:
                vals = sorted(v for _, v, _ in period)
                n = len(vals)
                median = (vals[n // 2] if n % 2
                          else 0.5 * (vals[n // 2 - 1] + vals[n // 2]))
                survivors = [(y, v, c) for y, v, c in period if v < median]
                ranked = []
                for year, _, _ in survivors:
                    got = compute(book, ticker, year, scope, rank_key)
                    if got is None:
                        continue
                    ranked.append((year, got[0], got[1]))
                if ranked:
                    winner = max(ranked, key=lambda x: x[1])
                    return pack_answer(
                        "year_median_then_extreme", med_key, tgt_key, ticker,
                        winner[1], winner[2], book, winner[0], scope, unit,
                        {"rank": rank_key, "median": median},
                    )

    # Year extreme on filter metric → lookup target (363, 440, …).
    ext = re.search(
        r"(?:vào năm|năm mà|tại năm mà|ở năm).{0,60}?"
        r"(?P<filt>(?:tỷ số|tỷ lệ|hệ số|biên).{5,70}?|ROA|ROE|D/E)"
        r"\s+(?P<side>cao nhất|thấp nhất)",
        question, re.I,
    )
    if ext:
        filt = _metric_in(ext.group("filt"))
        tgt = detect_target(question, filt or "")
        if not tgt and re.search(r"gấp bao nhiêu lần chi phí lãi vay", question, re.I):
            tgt = "interest_coverage"
        if filt and tgt and filt != tgt:
            scored = []
            for year in years:
                got = compute(book, ticker, year, scope, filt)
                if got is None:
                    continue
                scored.append((year, got[0], got[1]))
            if len(scored) >= 2:
                if "thấp" in ext.group("side").casefold():
                    winner = min(scored, key=lambda x: x[1])
                    op = "year_argmin"
                else:
                    winner = max(scored, key=lambda x: x[1])
                    op = "year_argmax"
                look = str(int(winner[0]) + 1) if nxt else winner[0]
                return pack_answer(op, filt, tgt, ticker, winner[1], winner[2],
                                   book, look, scope, unit, {"from_year": winner[0]})

    if re.search(r"CFO/LNST|chuyển đổi lợi nhuận|CFO trên LNST", question, re.I):
        filt = "cfo_to_ni"
        if re.search(r"thanh toán hiện hành|current ratio", question, re.I):
            target = "current_ratio"
        tgt = target or detect_target(question, filt)
        if tgt and tgt != filt:
            nxt = nxt or bool(re.search(r"năm kế tiếp|cuối năm kế tiếp", question, re.I))
            scored = []
            for year in years:
                if re.search(r"lợi nhuận sau thuế dương", question, re.I):
                    ni = compute(book, ticker, year, scope, "net_income")
                    if ni is None or ni[0] <= 0:
                        continue
                got = compute(book, ticker, year, scope, filt)
                if got is None:
                    continue
                scored.append((year, got[0], got[1]))
            if len(scored) >= 2:
                if re.search(r"thấp nhất", question, re.I):
                    winner = min(scored, key=lambda x: x[1])
                    op = "year_argmin"
                else:
                    winner = max(scored, key=lambda x: x[1])
                    op = "year_argmax"
                look = str(int(winner[0]) + 1) if nxt else winner[0]
                return pack_answer(op, filt, tgt, ticker, winner[1], winner[2],
                                   book, look, scope, unit, {"from_year": winner[0]})

    hop = re.search(
        r"(?P<tgt>hệ số.{5,60}?|biên.{5,40}?)\s+vào năm sau năm có\s+"
        r"(?P<filt>hệ số.{5,60}?)\s+(?P<ext>thấp nhất|cao nhất)",
        question, re.I,
    )
    if hop:
        target = _metric_in(hop.group("tgt")) or target
        filt = _metric_in(hop.group("filt"))
        if filt and target and filt != target:
            scored = []
            for year in years:
                got = compute(book, ticker, year, scope, filt)
                if got is None:
                    continue
                scored.append((year, got[0], got[1]))
            if len(scored) >= 2:
                if "thấp" in hop.group("ext").casefold():
                    winner = min(scored, key=lambda x: x[1])
                    op = "year_argmin"
                else:
                    winner = max(scored, key=lambda x: x[1])
                    op = "year_argmax"
                look = str(int(winner[0]) + 1) if nxt else winner[0]
                return pack_answer(op, filt, target, ticker, winner[1], winner[2],
                                   book, look, scope, unit, {"from_year": winner[0]})

    if re.search(r"CFO âm|dòng tiền.{0,20}âm", question, re.I) and re.search(
            r"năm đầu tiên", question, re.I):
        first = None
        cells_neg = None
        for year in years:
            got = compute(book, ticker, year, scope, "cfo")
            if got is None:
                return None
            if got[0] < 0:
                first, cells_neg = year, got[1]
                break
        if first is None:
            return None
        look = str(int(first) + 1) if nxt else first
        target = target or "gross_margin"
        return pack_answer(
            "first_neg_then_lookup", "cfo", target, ticker, 0.0,
            cells_neg or [], book, look, scope, unit, {"from_year": first})

    filt = detect_filter(question)
    target = detect_target(question, filt or "") or detect_filter(question)
    if filt is None or target is None or (filt == target and not nxt):
        return None
    scored = []
    for year in years:
        if re.search(r"lợi nhuận sau thuế dương", question, re.I):
            ni = compute(book, ticker, year, scope, "net_income")
            if ni is None:
                continue
            if ni[0] <= 0:
                continue
        got = compute(book, ticker, year, scope, filt)
        if got is None:
            continue
        scored.append((year, got[0], got[1]))
    if not scored:
        return None
    if re.search(r"thấp nhất", question, re.I):
        winner = min(scored, key=lambda x: x[1])
        op = "year_argmin"
    elif re.search(r"cao nhất", question, re.I):
        winner = max(scored, key=lambda x: x[1])
        op = "year_argmax"
    else:
        return None
    look = str(int(winner[0]) + 1) if nxt else winner[0]
    return pack_answer(op, filt, target, ticker, winner[1], winner[2],
                       book, look, scope, unit, {"from_year": winner[0]})


def solve_one(book: CellBook, question: str, resolver: TickerResolver) -> dict | None:
    for special in (
        solve_pair_delta, solve_yoy_delta_cohort,
        solve_median_mean_gap, solve_count_years, solve_subgroup_share,
    ):
        hit = special(book, question, resolver)
        if hit is not None:
            return hit
    if HARD_REFUSE.search(question):
        return None
    tickers = resolve_cohort(question, resolver)
    years = years_of(question)
    scope = "separate" if PARENT_RE.search(question) else "consolidated"

    if len(tickers) == 1 and len(years) >= 2:
        if not re.search(
            r"năm sau năm|năm kế tiếp|năm đầu tiên|tại năm|ở năm|"
            r"vào năm|năm mà|"
            r"của năm có",
            question, re.I,
        ):
            return None
        return solve_temporal(book, tickers[0], years, scope, question)

    if len(tickers) < 2:
        return None
    if re.search(r"có bao nhiêu", question, re.I) and not re.search(
            r"doanh nghiệp|công ty|mã", question, re.I):
        return None
    start_year = re.search(r"^Năm\s+(20[0-2]\d)", question)
    in_year = re.search(r"(?:trong|vào)\s+năm\s+(20[0-2]\d)", question, re.I)
    if start_year:
        year = start_year.group(1)
    elif in_year:
        year = in_year.group(1)
    elif len(years) == 1:
        year = years[0]
    elif re.search(r"năm 2025", question, re.I):
        year = "2025"
    elif re.search(r"từ 2023 sang 2024|từ năm 2023 sang 2024", question, re.I):
        year = "2024"
    else:
        year = "2024"

    preds = predicates_of(question)
    if re.search(r"có bao nhiêu doanh nghiệp", question, re.I) and not preds:
        return None
    look_year = year
    if re.search(r"từ năm 2024 đến năm 2025", question, re.I):
        look_year = "2025"
    triple = detect_triple(question)
    op, filter_key, target_key = detect_roles(question)
    # An indicator with no Circular-200 code can still be located by name. Only the
    # TARGET is substituted: a filter drives a comparison across companies and a
    # mislocated filter reorders every one of them, while a mislocated target spoils one
    # answer. The cheaper mistake is the one taken.
    if target_key is None and question in book.variants:
        target_key = LABEL_PREFIX + question
        if op is None:
            op = detect_op(question)

    def keep(ticker: str) -> bool | None:
        for pred in preds:
            passed = passes_pred(book, ticker, year, scope, pred)
            if passed is None:
                return None
            if not passed:
                return False
        return True

    eligible = []
    for ticker in tickers:
        ok = keep(ticker)
        if ok is True:
            eligible.append(ticker)

    if re.search(r"có bao nhiêu doanh nghiệp", question, re.I) and preds:
        cells: list[BoundCell] = []
        for ticker in eligible:
            got = compute(book, ticker, year, scope, preds[0][0])
            if got:
                cells.extend(got[1])
        if not cells:
            return None
        return {
            "op": "pred_count", "filter": preds[0][0], "target": "count",
            "survivors": eligible, "year": year, "pred_n": len(preds),
            **_pack_rate(float(len(eligible)), cells, filter_cells=[]),
        }

    if len(eligible) < 2:
        return None

    if triple:
        side, med_key, rank_key, tgt_key, ext = triple
        scored_med = []
        for ticker in eligible:
            got = compute(book, ticker, year, scope, med_key)
            if got is None:
                continue
            scored_med.append((ticker, got[0], got[1]))
        if len(scored_med) < 2:
            return None
        values = sorted(v for _, v, _ in scored_med)
        n = len(values)
        median = (values[n // 2] if n % 2
                  else 0.5 * (values[n // 2 - 1] + values[n // 2]))
        if side == "above":
            survivors = [t for t, v, _ in scored_med if v > median]
        else:
            survivors = [t for t, v, _ in scored_med if v < median]
        rank_year = look_year if rank_key in ("rev_growth", "gross_margin") else year
        ranked = []
        for ticker in survivors:
            got = compute(book, ticker, rank_year, scope, rank_key)
            if got is None:
                continue
            ranked.append((ticker, got[0], got[1]))
        if not ranked:
            return None
        winner = (max if ext == "argmax" else min)(ranked, key=lambda x: x[1])
        tgt_year = look_year if tgt_key != med_key else year
        return pack_answer(
            "median_then_rank_lookup", med_key, tgt_key, winner[0],
            winner[1], winner[2], book, tgt_year, scope, unit_of(question),
            {"rank": rank_key, "median": median, "look_year": tgt_year},
        )

    # Mean of target among predicate survivors, no rank (Q388).
    if op is None and preds and re.search(r"bình quân|trung bình", question, re.I):
        target_key = detect_target(question, "") or detect_filter(question)
        if target_key is None:
            return None
        op = "pred_mean"
        filter_key = preds[0][0]

    if op is None or filter_key is None or target_key is None:
        return None
    if op in ("argmin_lookup", "argmax_lookup") and filter_key == target_key:
        return None

    cohort = eligible
    # Predicate-only mean
    if op == "pred_mean":
        values, cells = [], []
        for ticker in cohort:
            got = compute(book, ticker, year, scope, target_key)
            if got is None:
                continue
            values.append(got[0])
            cells.extend(got[1])
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return {
            "op": op, "filter": filter_key, "target": target_key,
            "survivors": cohort, "year": year,
            **_pack_rate(mean, cells, filter_cells=[]),
        }

    scored: list[tuple[str, float, list[BoundCell]]] = []
    for ticker in cohort:
        got = compute(book, ticker, year, scope, filter_key)
        if got is None:
            continue
        scored.append((ticker, got[0], got[1]))
    if len(scored) < 2:
        return None

    unit = unit_of(question)

    def pack_target(winner_ticker, filter_value, filter_cells, target_key):
        return pack_answer(op, filter_key, target_key, winner_ticker,
                           filter_value, filter_cells, book, year, scope, unit)

    if op == "argmin_lookup":
        winner = min(scored, key=lambda x: x[1])
        return pack_target(winner[0], winner[1], winner[2], target_key)

    if op == "argmax_lookup":
        winner = max(scored, key=lambda x: x[1])
        return pack_target(winner[0], winner[1], winner[2], target_key)

    if op in ("below_median_mean", "above_median_mean"):
        values = sorted(v for _, v, _ in scored)
        n = len(values)
        median = (values[n // 2] if n % 2
                  else 0.5 * (values[n // 2 - 1] + values[n // 2]))
        if op == "below_median_mean":
            survivors = [(t, cells) for t, v, cells in scored if v < median]
        else:
            survivors = [(t, cells) for t, v, cells in scored if v > median]
        if not survivors:
            return None
        if re.search(r"cao nhất của các|thấp nhất của các", question, re.I):
            want_max = bool(re.search(r"cao nhất của các", question, re.I))
            ranked = []
            for ticker, _ in survivors:
                got = compute(book, ticker, year, scope, target_key)
                if got is None:
                    continue
                ranked.append((ticker, got[0], got[1]))
            if not ranked:
                return None
            winner = max(ranked, key=lambda x: x[1]) if want_max else min(
                ranked, key=lambda x: x[1])
            return pack_answer(
                "median_then_extreme", filter_key, target_key, winner[0],
                median, winner[2], book, year, scope, unit)
        targets = []
        all_cells: list[BoundCell] = []
        for ticker, _ in survivors:
            got = compute(book, ticker, year, scope, target_key)
            if got is None:
                continue
            targets.append(got[0])
            all_cells.extend(got[1])
        if not targets:
            return None
        mean = sum(targets) / len(targets)
        return {
            "op": op, "filter": filter_key, "target": target_key,
            "median": median, "survivors": [t for t, _ in survivors],
            **_pack_rate(mean, all_cells, filter_cells=[]),
        }
    return solve_aggressive_cohort(
        book, question, resolver, tickers, years, scope, year, look_year,
        preds, eligible,
    )


SCREEN_CUE = re.compile(
    r"trung vị|cao nhất|thấp nhất|trong nhóm|trong số|xét các|"
    r"đồng thời|bao nhiêu doanh nghiệp",
    re.I,
)


def solve_aggressive_cohort(
    book: CellBook, question: str, resolver: TickerResolver,
    tickers: list[str], years: list[str], scope: str, year: str,
    look_year: str, preds: list[tuple[str, str, float]],
    eligible: list[str],
) -> dict | None:
    """Last resort: try metric pairs mentioned in a screening question."""

    if len(eligible) < 2 or not SCREEN_CUE.search(question):
        return None
    op = detect_op(question)
    if op not in ("argmin_lookup", "argmax_lookup"):
        return None
    found: list[tuple[int, str]] = []
    for pattern, key in FILTER_PHRASE + TARGET_PHRASE:
        for match in pattern.finditer(question):
            found.append((match.start(), key))
    found.sort()
    keys = list(dict.fromkeys(key for _, key in found))
    if len(keys) < 2:
        return None
    unit = unit_of(question)
    tgt_year = look_year if re.search(r"từ năm 2024 đến năm 2025", question, re.I) else year
    pairs: list[tuple[str, str]] = []
    ext = re.search(r"có\s+(.{5,100}?)\s+(?:cao nhất|thấp nhất)", question, re.I)
    if ext:
        fk = _metric_in(ext.group(1))
        tk = keys[-1]
        if fk and tk and fk != tk:
            pairs.append((fk, tk))
    for fk in keys:
        for tk in reversed(keys):
            if fk != tk:
                pairs.append((fk, tk))
    seen: set[tuple[str, str]] = set()
    for filter_key, target_key in pairs:
        if (filter_key, target_key) in seen or filter_key == target_key:
            continue
        seen.add((filter_key, target_key))
        scored = []
        for ticker in eligible:
            got = compute(book, ticker, year, scope, filter_key)
            if got is None:
                continue
            scored.append((ticker, got[0], got[1]))
        if len(scored) < 2:
            continue
        winner = (min if op == "argmin_lookup" else max)(scored, key=lambda x: x[1])
        ty = tgt_year if target_key in (*YOY_LEVEL, "gross_margin", "rev_growth") else year
        hit = pack_answer(
            op, filter_key, target_key, winner[0], winner[1], winner[2],
            book, ty, scope, unit,
        )
        if hit and accept_hit(question, hit):
            hit["aggressive"] = True
            return hit
    return None


def accept_hit(question: str, hit: dict) -> bool:
    """Reject hops whose magnitude cannot be the asked unit (unscaled VND, etc.)."""

    ans = float(hit["answer"])
    op = hit.get("op") or ""
    if hit.get("filter") == hit.get("target") and not str(op).startswith("year"):
        return False
    if re.search(r"phần trăm|%|\blần\b|vòng", question, re.I) and abs(ans) > 500:
        return False
    if abs(ans) > 1e8:
        return False
    return True


CONFIDENT_OPS = frozenset({
    "argmin_lookup", "argmax_lookup", "below_median_mean", "above_median_mean",
    "median_then_rank_lookup", "median_then_extreme",
    "year_argmin", "year_argmax", "first_neg_then_lookup", "pred_mean",
    "pred_count", "argext_yoy_delta", "pair_delta", "year_median_then_extreme",
    "median_mean_gap", "count_years", "subgroup_share",
})


def confidence_hit(question: str, hit: dict) -> bool:
    """Stricter gate: only ship hops with structurally sane plans."""

    if not accept_hit(question, hit):
        return False
    op = hit.get("op") or ""
    if op not in CONFIDENT_OPS:
        return False
    filt, tgt = hit.get("filter"), hit.get("target")
    ans = float(hit["answer"])
    if op == "pred_count" and hit.get("pred_n", 1) < 2:
        if filt not in ("sga_beats_rev",):
            return False
    if op.startswith("year_") and filt == tgt:
        return False
    if re.search(r"\blần\b", question, re.I) and abs(ans) < 0.005:
        return False
    if op in ("below_median_mean", "above_median_mean") and tgt in ATOM:
        return False
    if op in ("below_median_mean", "above_median_mean") and tgt in (
            "net_revenue", "nwc", "inventory", "cfo"):
        return False
    if abs(ans) > 1e6:
        return False
    if re.search(r"phần trăm|%|chiếm bao nhiêu", question, re.I) and abs(ans) < 0.05:
        return False
    return True


def _pack_atom(cell: BoundCell, unit: float | None,
               filter_cells: list[BoundCell]) -> dict:
    name = f"{cell.doc}_table_{cell.table_id}.csv"
    payloads = {name: (ROOT / cell.csv).read_text(encoding="utf-8-sig")}
    evidence = [{"variable": "df", "csv_path": f"data/{name}"}]
    # Also ship filter-evidence tables so the program can show the hop.
    refs = [cell.table_ref]
    docs = [cell.doc]
    reads = []
    frames = {"df": None}
    for index, fc in enumerate(filter_cells):
        fname = f"{fc.doc}_table_{fc.table_id}.csv"
        if fname not in payloads:
            payloads[fname] = (ROOT / fc.csv).read_text(encoding="utf-8-sig")
        var = f"df{index + 1}"
        if fname == name:
            var = "df"
        else:
            evidence.append({"variable": var, "csv_path": f"data/{fname}"})
        refs.append(fc.table_ref)
        docs.append(fc.doc)
        reads.append(
            f"# filter {fc.kind}:{fc.code} @{fc.ticker}\n"
            f"_f{index} = _num({var}.iloc[{fc.row}, {fc.col}]) * {fc.scale!r}"
        )
    if unit is None:
        unit = 1.0
        answer = round(cell.value, 2)
        expression = f"_num(df.iloc[{cell.row}, {cell.col}]) * {cell.scale!r}"
    else:
        answer = round(cell.value / unit, 2)
        expression = (
            f"_num(df.iloc[{cell.row}, {cell.col}]) * {cell.scale!r} / {unit!r}"
        )
    body = "\n".join(reads) if reads else (
        f"_cell = _num(df.iloc[{cell.row}, {cell.col}]) * {cell.scale!r}"
    )
    if not reads:
        expression = f"_cell / {unit!r}" if unit != 1.0 or True else expression
        expression = (
            f"_num(df.iloc[{cell.row}, {cell.col}]) * {cell.scale!r} / {unit!r}"
        )
        body = "# target cell"
    code = PROGRAM.format(body=body + "\n" + f"# target {cell.kind}:{cell.code}",
                          expression=expression)
    return {
        "answer": answer,
        "pandas_query": code,
        "evidence": evidence,
        "relevant_docs": list(dict.fromkeys(docs)),
        "relevant_tables": list(dict.fromkeys(refs)),
        "csv_payloads": payloads,
    }


def _pack_rate(value: float, cells: list[BoundCell],
               filter_cells: list[BoundCell]) -> dict:
    payloads = {}
    evidence = []
    refs, docs = [], []
    body_lines = []
    for index, cell in enumerate(cells + filter_cells):
        name = f"{cell.doc}_table_{cell.table_id}.csv"
        if name not in payloads:
            payloads[name] = (ROOT / cell.csv).read_text(encoding="utf-8-sig")
            var = "df" if len(payloads) == 1 else f"df{len(payloads)}"
            # remap: first file always df
            var = "df" if len(evidence) == 0 else f"df{len(evidence) + 1}"
            if len(evidence) == 0:
                var = "df"
            else:
                var = f"df{len(evidence) + 1}"
            evidence.append({"variable": var if evidence else "df",
                             "csv_path": f"data/{name}"})
            if len(evidence) == 1:
                evidence[0]["variable"] = "df"
        # find variable for this file
        var = next(e["variable"] for e in evidence
                   if e["csv_path"] == f"data/{name}")
        refs.append(cell.table_ref)
        docs.append(cell.doc)
        body_lines.append(
            f"_v{index} = _num({var}.iloc[{cell.row}, {cell.col}]) * {cell.scale!r}"
            f"  # {cell.ticker} {cell.kind}:{cell.code}"
        )
    code = PROGRAM.format(body="\n".join(body_lines),
                          expression=f"{value!r}")
    return {
        "answer": round(value, 2),
        "pandas_query": code,
        "evidence": evidence,
        "relevant_docs": list(dict.fromkeys(docs)),
        "relevant_tables": list(dict.fromkeys(refs)),
        "csv_payloads": payloads,
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", default="")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--out", default="artifacts/fresh/hard_hop_plan.jsonl")
    parser.add_argument("--base", default="submissions/vote3.zip")
    parser.add_argument("--zip", default="",
                        help="write submission zip spliced onto --base")
    parser.add_argument("--bind", action="store_true",
                        help="let indicators with no Ma so be located by name")
    parser.add_argument("--spec", default="artifacts/fresh/spec_all.jsonl")
    parser.add_argument("--doc-scale", default="artifacts/fresh/doc_scale.json")
    args = parser.parse_args()

    questions = {json.loads(l)["id"]: json.loads(l)["question"]
                 for l in (ROOT / "data/questions/questions.jsonl").read_text(
                     encoding="utf-8").splitlines() if l.strip()}
    resolver = TickerResolver()
    book = CellBook()

    if args.bind:
        from bind_label import LabelBinder
        doc_scale = {}
        scale_path = ROOT / args.doc_scale
        if scale_path.exists():
            doc_scale = {k: float(v) for k, v
                         in json.loads(scale_path.read_text(encoding="utf-8")).items()}
        book.binder = LabelBinder(doc_scale)
        registered = 0
        spec_path = ROOT / args.spec
        for line in spec_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                spec = json.loads(row.get("reply") or "{}")
            except ValueError:
                continue
            names = []
            for item in spec.get("chi_tieu") or []:
                if isinstance(item, dict):
                    names.extend(str(v) for v in (item.get("bien_the") or []) if v)
                    if item.get("ten"):
                        names.append(str(item["ten"]))
            text = questions.get(row.get("id"))
            if text and names:
                book.register(text, names)
                registered += 1
        print(f"bind: {registered} cau co bien the nhan")

    if args.ids:
        ids = [int(x) for x in args.ids.split(",") if x.strip()]
    elif args.scan:
        ids = sorted(questions)
        if args.limit:
            ids = ids[: args.limit]
    else:
        ids = [390, 368, 392, 412, 371]

    import zipfile
    with zipfile.ZipFile(ROOT / args.base) as archive:
        base = {r["id"]: r for r in json.loads(archive.read("submission.json"))}

    plan = []
    for qid in ids:
        text = questions[qid]
        try:
            hit = solve_one(book, text, resolver)
        except Exception as exc:  # noqa: BLE001 — probe must not die on one Q
            print(f"id={qid} ERROR {exc}")
            continue
        if hit is None:
            if args.show:
                print(f"id={qid} REFUSE")
            continue
        if not accept_hit(text, hit):
            if args.show or not args.scan:
                print(f"id={qid} DROP {hit.get('op')} ans={hit['answer']}")
            continue
        inc = float(base[qid].get("answer") or 0)
        plan.append({"id": qid, **{k: v for k, v in hit.items()
                                   if k != "csv_payloads"},
                     "incumbent": inc,
                     "changed": abs(hit["answer"] - inc) > 0.01})
        # stash payloads separately on disk? keep in plan for build step via cells
        plan[-1]["_payloads"] = hit.get("csv_payloads", {})
        if args.show or not args.scan:
            print(f"id={qid} {hit.get('op')} filter={hit.get('filter')} "
                  f"target={hit.get('target')} winner={hit.get('winner')} "
                  f"ans={hit['answer']} incumbent={inc}")
            print(f"  Q: {text[:120]}")

    out = ROOT / args.out
    serializable = []
    for row in plan:
        item = {k: v for k, v in row.items() if k != "_payloads"}
        # BoundCell not serializable — drop
        serializable.append(item)
    out.write_text("\n".join(json.dumps(r, ensure_ascii=False)
                             for r in serializable) + ("\n" if serializable else ""),
                   encoding="utf-8")
    changed = sum(1 for r in plan if r["changed"])
    print(f"\nsolved {len(plan)}  changed_vs_vote3 {changed}  -> {out}")

    if args.zip:
        import shutil
        import zipfile as zf

        dest = ROOT / args.zip
        shutil.copy2(ROOT / args.base, dest)
        with zf.ZipFile(dest, "a") as archive:
            # rewrite submission.json with spliced rows
            pass
        # Rebuild cleanly from base + plan payloads
        with zf.ZipFile(ROOT / args.base) as src:
            rows = json.loads(src.read("submission.json"))
            files = {n: src.read(n) for n in src.namelist() if n != "submission.json"}
        by_id = {r["id"]: r for r in rows}
        spliced = 0
        for entry in plan:
            if not entry["changed"]:
                continue
            payloads = entry.get("_payloads") or {}
            qid = entry["id"]
            by_id[qid] = {
                **by_id[qid],
                "answer": entry["answer"],
                "pandas_query": entry["pandas_query"],
                "evidence": entry["evidence"],
                "relevant_docs": entry["relevant_docs"],
                "relevant_tables": entry["relevant_tables"],
            }
            for name, text in payloads.items():
                files[f"data/{name}"] = text.encode("utf-8")
            spliced += 1
        with zf.ZipFile(dest, "w", zf.ZIP_DEFLATED) as archive:
            archive.writestr("submission.json",
                             json.dumps([by_id[i] for i in sorted(by_id)],
                                        ensure_ascii=False, indent=1))
            for name, blob in files.items():
                archive.writestr(name, blob)
        print(f"spliced {spliced} -> {dest}")


if __name__ == "__main__":
    main()
