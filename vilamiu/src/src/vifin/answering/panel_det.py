"""Deterministic answers from the Circular-200 metric panel.

No LLM. The question is parsed into a closed set of operations over figures that
`metrics.parquet` already holds. The program shipped with each answer reads a
packaged panel CSV — the same contract `run_submit.py` already supports for
`USE_PANEL` — so EXECUTION scores the arithmetic, not OCR parsing.

Conservative by design: a pattern that is not recognised returns None, and the
caller leaves the existing submission row alone. A wrong parse that displaces a
correct answer costs more than a refusal.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import pandas as pd

from vifin.query.parse import ParsedQuestion, UNIT_SCALE

# Phrase → panel column. Longer phrases first so "tài sản ngắn hạn" wins over
# "tài sản". Derived names are computed at read time.
ALIASES: list[tuple[str, str]] = [
    ("lưu chuyển tiền thuần từ hoạt động kinh doanh", "cfo"),
    ("lưu chuyển tiền thuần từ hoạt động đầu tư", "cfi"),
    ("lưu chuyển tiền thuần từ hoạt động tài chính", "cff"),
    ("lợi nhuận thuần từ hoạt động kinh doanh", "operating_profit"),
    ("chi phí quản lý doanh nghiệp", "admin_expense"),
    ("chi phí bán hàng", "selling_expense"),
    ("chi phí lãi vay", "interest_expense"),
    ("chi phí tài chính", "financial_expense"),
    ("doanh thu hoạt động tài chính", "financial_income"),
    ("lợi nhuận trước thuế", "profit_before_tax"),
    ("lợi nhuận sau thuế", "net_profit"),
    ("lợi nhuận gộp", "gross_profit"),
    ("doanh thu thuần", "net_revenue"),
    ("doanh thu bán hàng", "revenue_gross"),
    ("giá vốn hàng bán", "cogs"),
    ("tài sản ngắn hạn", "current_assets"),
    ("tài sản dài hạn", "long_assets"),
    ("tổng cộng tài sản", "total_assets"),
    ("tổng tài sản", "total_assets"),
    ("nợ ngắn hạn", "liabilities_short"),
    ("nợ dài hạn", "liabilities_long"),
    ("nợ phải trả", "liabilities"),
    ("vốn chủ sở hữu", "equity"),
    ("hàng tồn kho", "inventory"),
    ("tiền và các khoản tương đương tiền", "cash"),
    ("các khoản phải thu ngắn hạn", "receivables_short"),
    ("cho vay khách hàng", "loans_to_customers"),
    ("tiền gửi của khách hàng", "customer_deposits"),
    ("vốn lưu động ròng", "_nwc"),
    ("vốn lưu động", "_nwc"),
]

# Named ratios → (numerator, denominator, multiply_by_100?).
NAMED_RATIOS: list[tuple[re.Pattern[str], str, str, bool]] = [
    (re.compile(r"biên lợi nhuận gộp", re.I), "gross_profit", "net_revenue", True),
    (re.compile(r"biên lợi nhuận (?:ròng|thuần)|\bROS\b", re.I),
     "net_profit", "net_revenue", True),
    (re.compile(r"\bROA\b|sinh lời trên (?:tổng )?tài sản", re.I),
     "net_profit", "total_assets", True),
    (re.compile(r"\bROE\b|sinh lời trên vốn chủ sở hữu", re.I),
     "net_profit", "equity", True),
    (re.compile(r"(?:hệ số|tỷ số|tỉ số) thanh toán hiện hành", re.I),
     "current_assets", "liabilities_short", False),
    (re.compile(r"(?:hệ số|tỷ số|tỉ số) thanh toán nhanh", re.I),
     "_quick", "liabilities_short", False),
    (re.compile(r"(?:hệ số|tỷ số) nợ phải trả trên vốn chủ sở hữu|\bD/E\b", re.I),
     "liabilities", "equity", False),
    (re.compile(r"hệ số nợ phải trả trên tổng tài sản|tỷ lệ nợ trên tổng tài sản", re.I),
     "liabilities", "total_assets", True),
    (re.compile(r"CFO [Mm]argin|biên dòng tiền từ hoạt động kinh doanh", re.I),
     "cfo", "net_revenue", True),
    (re.compile(r"tỷ lệ lưu chuyển tiền thuần từ hoạt động kinh doanh trên lợi nhuận", re.I),
     "cfo", "net_profit", False),
    (re.compile(r"tỷ lệ lưu chuyển tiền thuần từ hoạt động kinh doanh trên nợ ngắn hạn|"
                r"hệ số dòng tiền hoạt động trên nợ ngắn hạn", re.I),
     "cfo", "liabilities_short", False),
    (re.compile(r"tỷ lệ hàng tồn kho trên nợ ngắn hạn", re.I),
     "inventory", "liabilities_short", False),
    (re.compile(r"tỷ trọng hàng tồn kho trên tổng tài sản|"
                r"tỷ lệ hàng tồn kho trên tổng tài sản", re.I),
     "inventory", "total_assets", True),
    (re.compile(r"tỷ lệ tài sản ngắn hạn trên nợ ngắn hạn", re.I),
     "current_assets", "liabilities_short", False),
    (re.compile(r"vòng quay tổng tài sản", re.I),
     "net_revenue", "total_assets", False),
    (re.compile(r"khả năng thanh toán lãi vay", re.I),
     "_interest_cover_num", "interest_expense", False),
]

PANEL_HEADER = ["ma", "nam", "chi_tieu", "nhan_trong_bao_cao", "gia_tri_dong"]

COUNT_RE = re.compile(r"bao nhiêu (?:doanh nghiệp|công ty|mã|ngân hàng)", re.I)
MAX_RE = re.compile(r"cao nhất|lớn nhất|nhiều nhất", re.I)
MIN_RE = re.compile(r"thấp nhất|nhỏ nhất|ít nhất", re.I)
GROWTH_RE = re.compile(r"tăng trưởng|tốc độ tăng|giảm bao nhiêu\s*%", re.I)
DIFF_RE = re.compile(r"chênh lệch|trừ đi", re.I)
SUM_RE = re.compile(r"\btổng\b", re.I)
POSITIVE_RE = re.compile(r"(?:CFO|lưu chuyển tiền thuần từ hoạt động kinh doanh)\s+dương|"
                         r"dòng tiền hoạt động dương", re.I)
NEGATIVE_NWC_RE = re.compile(r"vốn lưu động ròng âm", re.I)
MEDIAN_RE = re.compile(r"trung vị", re.I)
YOY_UP_RE = re.compile(r"\btăng\b(?!\s*trưởng)", re.I)
YOY_DOWN_RE = re.compile(r"\bgiảm\b", re.I)
REV_GROWTH_RE = re.compile(
    r"tăng trưởng doanh thu thuần|doanh thu thuần\s+(?:tăng|tăng trưởng)", re.I)
CFO_MARGIN_NEG_RE = re.compile(
    r"(?:CFO\s*[Mm]argin|biên dòng tiền từ hoạt động kinh doanh).{0,40}âm|"
    r"âm.{0,40}(?:CFO\s*[Mm]argin|biên dòng tiền từ hoạt động kinh doanh)",
    re.I | re.S,
)


@dataclass(frozen=True, slots=True)
class PanelAnswer:
    value: float
    code: str
    panel_rows: list[list[str]]
    keys: list[list]  # provenance [[doc, table_id], ...] — may be empty
    shape: str
    metrics: list[str]


def fold(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold())


def metrics_mentioned(text: str) -> list[str]:
    folded = fold(text)
    found: list[str] = []
    for phrase, name in ALIASES:
        if phrase in folded and name not in found:
            found.append(name)
    return found


def _cell(panel: dict, ticker: str, year: str, scope: str, metric: str) -> float | None:
    """One figure, trying preferred scope then the other."""

    order = (scope, "separate" if scope == "consolidated" else "consolidated",
             "unspecified")
    for sc in order:
        row = panel.get((ticker, year, sc))
        if row is None:
            continue
        if metric == "_nwc":
            ca, ls = row.get("current_assets"), row.get("liabilities_short")
            if ca is None or ls is None or (isinstance(ca, float) and math.isnan(ca)) \
                    or (isinstance(ls, float) and math.isnan(ls)):
                continue
            return float(ca) - float(ls)
        if metric == "_quick":
            ca, inv = row.get("current_assets"), row.get("inventory")
            if ca is None or inv is None:
                continue
            if isinstance(ca, float) and math.isnan(ca):
                continue
            if isinstance(inv, float) and math.isnan(inv):
                continue
            return float(ca) - float(inv)
        if metric == "_interest_cover_num":
            pbt, ie = row.get("profit_before_tax"), row.get("interest_expense")
            if pbt is None or ie is None:
                continue
            if isinstance(pbt, float) and math.isnan(pbt):
                continue
            if isinstance(ie, float) and math.isnan(ie):
                continue
            return float(pbt) + float(ie)
        value = row.get(metric)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue
        return float(value)
    return None


def load_panel(path) -> dict[tuple[str, str, str], dict]:
    frame = pd.read_parquet(path)
    out: dict[tuple[str, str, str], dict] = {}
    for _, row in frame.iterrows():
        key = (str(row["ticker"]), str(row["year"]), str(row["scope"]))
        out[key] = {c: row[c] for c in frame.columns
                    if c not in ("ticker", "year", "scope")}
    return out


def _scale(question: ParsedQuestion) -> float:
    if question.target_unit in ("phan_tram", "lan", "vong", ""):
        return 1.0
    return UNIT_SCALE.get(question.target_unit, 1.0) or 1.0


def _pack(rows_body: list[list], code: str, value: float, shape: str,
          metrics: list[str]) -> PanelAnswer:
    panel_rows = [list(PANEL_HEADER)] + [
        [str(r[0]), str(r[1]), str(r[2]), str(r[3]), repr(float(r[4]))]
        for r in rows_body
    ]
    return PanelAnswer(
        value=round(float(value), 2),
        code=code,
        panel_rows=panel_rows,
        keys=[],
        shape=shape,
        metrics=metrics,
    )


def _emit_ratio(num_name: str, den_name: str, as_pct: bool, scale: float) -> str:
    pct = " * 100.0" if as_pct else ""
    return (
        f"num = float(df[df['chi_tieu'] == {num_name!r}].iloc[0]['gia_tri_dong'])\n"
        f"den = float(df[df['chi_tieu'] == {den_name!r}].iloc[0]['gia_tri_dong'])\n"
        f"result = round(num / den{pct} / {scale!r}, 2)"
    )


def _emit_series_op(op: str, metric: str, scale: float, axis: str = "nam") -> str:
    col = "nam" if axis == "nam" else "ma"
    # Growth keeps sign; money extremes / sums / diffs use magnitude — the
    # contest treats undirected money differences as absolute values, and
    # statement lines flip sign across issuers.
    use_abs = op != "growth"
    lines = [
        f"sub = df[df['chi_tieu'] == {metric!r}]",
        "vals = {}",
        "for i in range(len(sub)):",
        f"    k = str(sub.iloc[i][{col!r}])",
        "    if k in vals:",
        "        continue",
        ("    vals[k] = abs(float(sub.iloc[i]['gia_tri_dong']))" if use_abs
         else "    vals[k] = float(sub.iloc[i]['gia_tri_dong'])"),
        "keys = sorted(vals)",
    ]
    if op == "max":
        lines += ["win = keys[0]", "for k in keys:",
                  "    if vals[k] > vals[win]:", "        win = k",
                  "amount = vals[win]"]
    elif op == "min":
        lines += ["win = keys[0]", "for k in keys:",
                  "    if vals[k] < vals[win]:", "        win = k",
                  "amount = vals[win]"]
    elif op == "sum":
        lines += ["amount = 0.0", "for k in keys:", "    amount = amount + vals[k]"]
    elif op == "diff":
        lines.append("amount = abs(vals[keys[-1]] - vals[keys[0]])")
    elif op == "growth":
        lines.append(
            "amount = (vals[keys[-1]] - vals[keys[0]]) / abs(vals[keys[0]]) * 100.0")
    else:
        lines.append("amount = vals[keys[-1]]")
    if op == "growth":
        lines.append("result = round(amount, 2)")
    else:
        lines.append(f"result = round(amount / {scale!r}, 2)")
    return "\n".join(lines)


def _unique_tickers_code() -> str:
    """py37-safe unique ticker list — no set/genexp (sandbox forbids both)."""

    return (
        "tickers = []\n"
        "for x in df['ma'].tolist():\n"
        "    s = str(x)\n"
        "    if s in tickers:\n"
        "        continue\n"
        "    tickers.append(s)\n"
        "tickers = sorted(tickers)"
    )


def named_ratio(text: str) -> tuple[str, str, bool] | None:
    for pattern, num, den, pct in NAMED_RATIOS:
        if pattern.search(text):
            return num, den, pct
    return None


def _ratio_cell(panel: dict, ticker: str, year: str, scope: str,
                num: str, den: str) -> float | None:
    top = _cell(panel, ticker, year, scope, num)
    bottom = _cell(panel, ticker, year, scope, den)
    if top is None or bottom is None or bottom == 0:
        return None
    return top / bottom


def solve(question: ParsedQuestion, panel: dict) -> PanelAnswer | None:
    """High-confidence shapes only. Returns None when unsure."""

    text = question.question
    tickers = list(question.tickers)
    years = [str(y) for y in sorted(question.years)]
    if not tickers or not years:
        return None
    # Nested median / multi-hop filters are not yet encoded; declining them is
    # cheaper than answering the wrong subset.
    if MEDIAN_RE.search(text) and (MAX_RE.search(text) or MIN_RE.search(text)):
        if "trung vị" in fold(text) and ("có" in fold(text) or "xét" in fold(text)):
            # Allow simple "cao hơn trung vị" screens below; block "trung vị rồi
            # cao nhất rồi hỏi metric khác" for now.
            if COUNT_RE.search(text):
                pass  # count-with-median handled carefully below
            elif named_ratio(text) and not COUNT_RE.search(text):
                # e.g. screen by median then report a ratio of the winner — hard.
                if MAX_RE.search(text) or MIN_RE.search(text):
                    return None

    scale = _scale(question)
    scope = question.scope or "consolidated"

    # --- 1. Single-company named ratio in one year --------------------------------
    ratio = named_ratio(text)
    if (ratio and len(tickers) == 1 and len(years) == 1
            and not GROWTH_RE.search(text) and not DIFF_RE.search(text)
            and not COUNT_RE.search(text)
            and not (MAX_RE.search(text) and len(years) > 1)):
        num, den, as_pct = ratio
        # Quick ratio needs current_assets and inventory as intermediates.
        if num == "_quick":
            ca = _cell(panel, tickers[0], years[0], scope, "current_assets")
            inv = _cell(panel, tickers[0], years[0], scope, "inventory")
            ls = _cell(panel, tickers[0], years[0], scope, "liabilities_short")
            if None in (ca, inv, ls) or ls == 0:
                return None
            top, bottom = ca - inv, ls
            body = [
                [tickers[0], years[0], "current_assets", "TSNH", ca],
                [tickers[0], years[0], "inventory", "HTK", inv],
                [tickers[0], years[0], "liabilities_short", "NNH", ls],
            ]
            code = (
                "ca = float(df[df['chi_tieu'] == 'current_assets'].iloc[0]['gia_tri_dong'])\n"
                "inv = float(df[df['chi_tieu'] == 'inventory'].iloc[0]['gia_tri_dong'])\n"
                "ls = float(df[df['chi_tieu'] == 'liabilities_short'].iloc[0]['gia_tri_dong'])\n"
                "result = round((ca - inv) / ls, 2)"
            )
            return _pack(body, code, top / bottom, "ratio_quick",
                         ["current_assets", "inventory", "liabilities_short"])
        if num == "_interest_cover_num":
            top = _cell(panel, tickers[0], years[0], scope, "_interest_cover_num")
            bottom = _cell(panel, tickers[0], years[0], scope, "interest_expense")
            pbt = _cell(panel, tickers[0], years[0], scope, "profit_before_tax")
            ie = _cell(panel, tickers[0], years[0], scope, "interest_expense")
            if None in (top, bottom, pbt, ie) or bottom == 0:
                return None
            body = [
                [tickers[0], years[0], "profit_before_tax", "LNTT", pbt],
                [tickers[0], years[0], "interest_expense", "Lãi vay", ie],
            ]
            code = (
                "pbt = float(df[df['chi_tieu'] == 'profit_before_tax'].iloc[0]['gia_tri_dong'])\n"
                "ie = float(df[df['chi_tieu'] == 'interest_expense'].iloc[0]['gia_tri_dong'])\n"
                "result = round((pbt + ie) / ie, 2)"
            )
            return _pack(body, code, top / bottom, "ratio_interest_cover",
                         ["profit_before_tax", "interest_expense"])
        top = _cell(panel, tickers[0], years[0], scope, num)
        bottom = _cell(panel, tickers[0], years[0], scope, den)
        if top is None or bottom is None or bottom == 0:
            return None
        value = top / bottom * (100.0 if as_pct else 1.0)
        # Unit: ratio questions ask % or lần — never scale by tỷ/triệu.
        body = [
            [tickers[0], years[0], num, num, top],
            [tickers[0], years[0], den, den, bottom],
        ]
        return _pack(body, _emit_ratio(num, den, as_pct, 1.0), value,
                     "ratio_named", [num, den])

    # --- 2. Growth / diff for one company, two years, one metric ------------------
    mets = [m for m in metrics_mentioned(text) if not m.startswith("_")]
    if (len(tickers) == 1 and len(years) == 2 and len(mets) == 1
            and (GROWTH_RE.search(text) or DIFF_RE.search(text))
            and not COUNT_RE.search(text) and not MAX_RE.search(text)):
        metric = mets[0]
        a = _cell(panel, tickers[0], years[0], scope, metric)
        b = _cell(panel, tickers[0], years[1], scope, metric)
        if a is None or b is None:
            return None
        body = [
            [tickers[0], years[0], metric, metric, a],
            [tickers[0], years[1], metric, metric, b],
        ]
        if GROWTH_RE.search(text):
            if a == 0:
                return None
            value = (b - a) / abs(a) * 100.0
            return _pack(body, _emit_series_op("growth", metric, 1.0), value,
                         "growth", [metric])
        value = abs(b - a) / scale
        return _pack(body, _emit_series_op("diff", metric, scale), value,
                     "diff", [metric])

    # --- 2b. "Năm nào" with one Circular-200 metric → return the winning year ----
    # The extreme_* branch above deliberately skips these (answer type is a year,
    # not the figure). Without this, the zip keeps answering with the max value.
    if (len(tickers) == 1 and len(years) >= 2 and len(mets) == 1
            and "năm nào" in fold(text)
            and (MAX_RE.search(text) or MIN_RE.search(text))
            and not COUNT_RE.search(text)
            and not named_ratio(text)
            and "tỷ lệ" not in fold(text) and "tỉ lệ" not in fold(text)
            and "tỷ trọng" not in fold(text) and "%" not in text
            and "phần trăm" not in fold(text)
            and not re.search(
                r"(?:doanh nghiệp|công ty|ngân hàng)\s+(?:nào\s+|mà\s+)?có\b",
                text, re.I)):
        metric = mets[0]
        series: dict[str, float] = {}
        body = []
        for year in years:
            value = _cell(panel, tickers[0], year, scope, metric)
            if value is None:
                return None
            series[year] = abs(value)
            body.append([tickers[0], year, metric, metric, value])
        want_max = bool(MAX_RE.search(text))
        winner = (max if want_max else min)(series, key=series.get)
        cmp = ">" if want_max else "<"
        code = (
            f"sub = df[df['chi_tieu'] == {metric!r}]\n"
            "vals = {}\n"
            "for i in range(len(sub)):\n"
            "    y = str(sub.iloc[i]['nam'])\n"
            "    if y in vals:\n"
            "        continue\n"
            "    vals[y] = abs(float(sub.iloc[i]['gia_tri_dong']))\n"
            "keys = sorted(vals)\n"
            "win = keys[0]\n"
            "for y in keys:\n"
            f"    if vals[y] {cmp} vals[win]:\n"
            "        win = y\n"
            "result = float(win)"
        )
        return _pack(body, code, float(winner), "year_extreme", [metric])

    # --- 3. Max / min over years, one company, one metric -------------------------
    # Two-hop ("… trong năm có X cao nhất") must NOT land here — that returns
    # max(X) when the question asks for Y of the winning year.
    _twohop = re.compile(
        r"(?:tại|trong|vào|ở)\s+(?:cuối\s+)?năm\s+(?:mà\s+)?có\b|"
        r"\bnăm\s+có\b.{4,90}?(?:cao nhất|lớn nhất|thấp nhất|nhỏ nhất)|"
        r"doanh nghiệp|công ty|ngân hàng|mã\s+(?:nào\s+|mà\s+)?có\b",
        re.I | re.S,
    )
    if (len(tickers) == 1 and len(years) >= 3 and len(mets) == 1
            and (MAX_RE.search(text) or MIN_RE.search(text))
            and not COUNT_RE.search(text)
            and "vào năm" not in fold(text)  # argmax-year → different answer type
            and "năm nào" not in fold(text)
            and not named_ratio(text)
            and not _twohop.search(text)
            and "tỷ lệ" not in fold(text) and "tỉ lệ" not in fold(text)
            and "tỷ trọng" not in fold(text) and "%" not in text
            and "phần trăm" not in fold(text)):
        metric = mets[0]
        series = {}
        body = []
        for year in years:
            value = _cell(panel, tickers[0], year, scope, metric)
            if value is None:
                return None
            series[year] = abs(value)
            body.append([tickers[0], year, metric, metric, value])
        op = "max" if MAX_RE.search(text) else "min"
        amount = max(series.values()) if op == "max" else min(series.values())
        return _pack(body, _emit_series_op(op, metric, scale), amount / scale,
                     f"extreme_{op}", [metric])

    # --- 4. Sum across companies, one year, one metric ----------------------------
    if (len(tickers) >= 2 and len(years) == 1 and len(mets) == 1
            and SUM_RE.search(text) and not COUNT_RE.search(text)
            and not MAX_RE.search(text) and not MIN_RE.search(text)
            and not named_ratio(text) and not DIFF_RE.search(text)
            and "trung bình" not in fold(text) and "bình quân" not in fold(text)
            and "hiệu số" not in fold(text)
            and not re.search(r"lớn hơn|nhỏ hơn|cao hơn|thấp hơn|nhiều hơn|ít hơn",
                              text, re.I)
            and "tổng số công ty" not in fold(text)
            and "tổng số doanh nghiệp" not in fold(text)):
        metric = mets[0]
        body = []
        total = 0.0
        for ticker in tickers:
            value = _cell(panel, ticker, years[0], scope, metric)
            if value is None:
                return None
            total += abs(value)
            body.append([ticker, years[0], metric, metric, value])
        return _pack(body, _emit_series_op("sum", metric, scale, axis="ma"),
                     total / scale, "company_sum", [metric])

    # --- 5a. Count: dual YoY ratio moves (inventory weight ↑ + gross margin ↓) ---
    if (COUNT_RE.search(text) and len(tickers) >= 2 and len(years) >= 2
            and "đồng thời" in fold(text)):
        y0, y1 = years[-2], years[-1]
        # 402-style: inventory/assets up AND gross margin down.
        inv_up = bool(re.search(
            r"tăng\s+tỷ trọng hàng tồn kho trên tổng tài sản", text, re.I))
        gm_down = bool(re.search(r"giảm\s+biên lợi nhuận gộp", text, re.I))
        if inv_up and gm_down:
            body = []
            count = 0
            incomplete = False
            for ticker in tickers:
                i0 = _ratio_cell(panel, ticker, y0, scope, "inventory", "total_assets")
                i1 = _ratio_cell(panel, ticker, y1, scope, "inventory", "total_assets")
                g0 = _ratio_cell(panel, ticker, y0, scope, "gross_profit", "net_revenue")
                g1 = _ratio_cell(panel, ticker, y1, scope, "gross_profit", "net_revenue")
                if None in (i0, i1, g0, g1):
                    incomplete = True
                    break
                for y, met, lab, val in (
                    (y0, "inventory", "HTK", _cell(panel, ticker, y0, scope, "inventory")),
                    (y0, "total_assets", "TTS", _cell(panel, ticker, y0, scope, "total_assets")),
                    (y1, "inventory", "HTK", _cell(panel, ticker, y1, scope, "inventory")),
                    (y1, "total_assets", "TTS", _cell(panel, ticker, y1, scope, "total_assets")),
                    (y0, "gross_profit", "LNGop", _cell(panel, ticker, y0, scope, "gross_profit")),
                    (y0, "net_revenue", "DTT", _cell(panel, ticker, y0, scope, "net_revenue")),
                    (y1, "gross_profit", "LNGop", _cell(panel, ticker, y1, scope, "gross_profit")),
                    (y1, "net_revenue", "DTT", _cell(panel, ticker, y1, scope, "net_revenue")),
                ):
                    if val is not None:
                        body.append([ticker, y, met, lab, val])
                if i1 > i0 and g1 < g0:
                    count += 1
            if not incomplete and body:
                code = "\n".join([
                    _unique_tickers_code(),
                    "n = 0",
                    "for t in tickers:",
                    "    ok = True",
                    f"    i0n = df[(df['ma'] == t) & (df['nam'].map(str) == {y0!r}) & "
                    "(df['chi_tieu'] == 'inventory')]",
                    f"    i0d = df[(df['ma'] == t) & (df['nam'].map(str) == {y0!r}) & "
                    "(df['chi_tieu'] == 'total_assets')]",
                    f"    i1n = df[(df['ma'] == t) & (df['nam'].map(str) == {y1!r}) & "
                    "(df['chi_tieu'] == 'inventory')]",
                    f"    i1d = df[(df['ma'] == t) & (df['nam'].map(str) == {y1!r}) & "
                    "(df['chi_tieu'] == 'total_assets')]",
                    f"    g0n = df[(df['ma'] == t) & (df['nam'].map(str) == {y0!r}) & "
                    "(df['chi_tieu'] == 'gross_profit')]",
                    f"    g0d = df[(df['ma'] == t) & (df['nam'].map(str) == {y0!r}) & "
                    "(df['chi_tieu'] == 'net_revenue')]",
                    f"    g1n = df[(df['ma'] == t) & (df['nam'].map(str) == {y1!r}) & "
                    "(df['chi_tieu'] == 'gross_profit')]",
                    f"    g1d = df[(df['ma'] == t) & (df['nam'].map(str) == {y1!r}) & "
                    "(df['chi_tieu'] == 'net_revenue')]",
                    "    if (len(i0n) == 0 or len(i0d) == 0 or len(i1n) == 0 or "
                    "len(i1d) == 0 or len(g0n) == 0 or len(g0d) == 0 or "
                    "len(g1n) == 0 or len(g1d) == 0):",
                    "        ok = False",
                    "    if ok:",
                    "        i0 = float(i0n.iloc[0]['gia_tri_dong']) / "
                    "float(i0d.iloc[0]['gia_tri_dong'])",
                    "        i1 = float(i1n.iloc[0]['gia_tri_dong']) / "
                    "float(i1d.iloc[0]['gia_tri_dong'])",
                    "        g0 = float(g0n.iloc[0]['gia_tri_dong']) / "
                    "float(g0d.iloc[0]['gia_tri_dong'])",
                    "        g1 = float(g1n.iloc[0]['gia_tri_dong']) / "
                    "float(g1d.iloc[0]['gia_tri_dong'])",
                    "        if i1 > i0 and g1 < g0:",
                    "            n = n + 1",
                    "result = float(n)",
                ])
                return _pack(body, code, float(count), "count_yoy_dual",
                             ["inventory", "total_assets", "gross_profit", "net_revenue"])

        # 411-style: net revenue up YoY AND CFO margin negative in the later year.
        if REV_GROWTH_RE.search(text) and CFO_MARGIN_NEG_RE.search(text):
            body = []
            count = 0
            incomplete = False
            for ticker in tickers:
                r0 = _cell(panel, ticker, y0, scope, "net_revenue")
                r1 = _cell(panel, ticker, y1, scope, "net_revenue")
                cfo = _cell(panel, ticker, y1, scope, "cfo")
                if None in (r0, r1, cfo):
                    incomplete = True
                    break
                body.append([ticker, y0, "net_revenue", "DTT", r0])
                body.append([ticker, y1, "net_revenue", "DTT", r1])
                body.append([ticker, y1, "cfo", "CFO", cfo])
                if r1 > r0 and cfo < 0:
                    count += 1
            if not incomplete and body:
                code = "\n".join([
                    _unique_tickers_code(),
                    "n = 0",
                    "for t in tickers:",
                    f"    r0 = df[(df['ma'] == t) & (df['nam'].map(str) == {y0!r}) & "
                    "(df['chi_tieu'] == 'net_revenue')]",
                    f"    r1 = df[(df['ma'] == t) & (df['nam'].map(str) == {y1!r}) & "
                    "(df['chi_tieu'] == 'net_revenue')]",
                    f"    cfo = df[(df['ma'] == t) & (df['nam'].map(str) == {y1!r}) & "
                    "(df['chi_tieu'] == 'cfo')]",
                    "    if len(r0) == 0 or len(r1) == 0 or len(cfo) == 0:",
                    "        continue",
                    "    if (float(r1.iloc[0]['gia_tri_dong']) > "
                    "float(r0.iloc[0]['gia_tri_dong']) and "
                    "float(cfo.iloc[0]['gia_tri_dong']) < 0):",
                    "        n = n + 1",
                    "result = float(n)",
                ])
                return _pack(body, code, float(count), "count_rev_cfo_mgn",
                             ["net_revenue", "cfo"])

    # --- 5. Count: companies with CFO dương (+ optional NWC âm) -------------------
    if COUNT_RE.search(text) and len(tickers) >= 2 and len(years) >= 1:
        year = years[-1]
        need_cfo_pos = bool(POSITIVE_RE.search(text))
        need_nwc_neg = bool(NEGATIVE_NWC_RE.search(text))
        if not need_cfo_pos and not need_nwc_neg:
            return None
        # Two-year CFO positive: "dương trong cả hai năm"
        both_years = "cả hai năm" in fold(text) or "cả năm" in fold(text)
        use_years = years[-2:] if both_years and len(years) >= 2 else [year]
        body = []
        count = 0
        for ticker in tickers:
            ok = True
            if need_cfo_pos:
                for y in use_years:
                    cfo = _cell(panel, ticker, y, scope, "cfo")
                    if cfo is None:
                        ok = False
                        break
                    body.append([ticker, y, "cfo", "CFO", cfo])
                    if cfo <= 0:
                        ok = False
                        break
            if ok and need_nwc_neg:
                nwc = _cell(panel, ticker, year, scope, "_nwc")
                ca = _cell(panel, ticker, year, scope, "current_assets")
                ls = _cell(panel, ticker, year, scope, "liabilities_short")
                if None in (nwc, ca, ls):
                    ok = False
                else:
                    body.append([ticker, year, "current_assets", "TSNH", ca])
                    body.append([ticker, year, "liabilities_short", "NNH", ls])
                    if nwc >= 0:
                        ok = False
            if ok:
                count += 1
        if not body:
            return None
        # Program: count tickers satisfying the same predicates.
        # Compare nam as str: pandas.read_csv may promote years to int.
        code_lines = [
            _unique_tickers_code(),
            "n = 0",
            "for t in tickers:",
            "    ok = True",
        ]
        if need_cfo_pos:
            for y in use_years:
                code_lines += [
                    f"    sub = df[(df['ma'] == t) & (df['nam'].map(str) == {y!r}) & "
                    f"(df['chi_tieu'] == 'cfo')]",
                    "    if len(sub) == 0 or float(sub.iloc[0]['gia_tri_dong']) <= 0:",
                    "        ok = False",
                ]
        if need_nwc_neg:
            code_lines += [
                f"    ca = df[(df['ma'] == t) & (df['nam'].map(str) == {year!r}) & "
                f"(df['chi_tieu'] == 'current_assets')]",
                f"    ls = df[(df['ma'] == t) & (df['nam'].map(str) == {year!r}) & "
                f"(df['chi_tieu'] == 'liabilities_short')]",
                "    if len(ca) == 0 or len(ls) == 0:",
                "        ok = False",
                "    elif float(ca.iloc[0]['gia_tri_dong']) - "
                "float(ls.iloc[0]['gia_tri_dong']) >= 0:",
                "        ok = False",
            ]
        code_lines += ["    if ok:", "        n = n + 1", "result = float(n)"]
        return _pack(body, "\n".join(code_lines), float(count), "count_screen",
                     ["cfo", "current_assets", "liabilities_short"])

    # --- 6. Screen: among companies with CFO > 0, pick max/min named ratio ---------
    # Only when that ratio is itself the answer. "…biên gộp cao nhất… có hệ số
    # lãi vay là bao nhiêu" is a third hop — refuse rather than return the margin.
    if (len(tickers) >= 2 and len(years) == 1 and POSITIVE_RE.search(text)
            and (MAX_RE.search(text) or MIN_RE.search(text))
            and not COUNT_RE.search(text) and not MEDIAN_RE.search(text)):
        year = years[0]
        ratio = named_ratio(text)
        if ratio is None:
            return None
        num, den, as_pct = ratio
        if num.startswith("_") or den.startswith("_"):
            return None
        super_m = MAX_RE.search(text) or MIN_RE.search(text)
        after = text[super_m.end():] if super_m else ""
        if named_ratio(after) or metrics_mentioned(after):
            return None
        want_max = bool(MAX_RE.search(text))
        body = []
        scores: dict[str, float] = {}
        for ticker in tickers:
            cfo = _cell(panel, ticker, year, scope, "cfo")
            top = _cell(panel, ticker, year, scope, num)
            bottom = _cell(panel, ticker, year, scope, den)
            if None in (cfo, top, bottom) or bottom == 0 or cfo <= 0:
                continue
            scores[ticker] = top / bottom * (100.0 if as_pct else 1.0)
            body.append([ticker, year, "cfo", "CFO", cfo])
            body.append([ticker, year, num, num, top])
            body.append([ticker, year, den, den, bottom])
        if len(scores) < 1:
            return None
        winner = (max if want_max else min)(scores, key=scores.get)
        cmp = ">" if want_max else "<"
        code = (
            f"{_unique_tickers_code()}\n"
            f"best = None\n"
            f"best_val = None\n"
            f"for t in tickers:\n"
            f"    cfo = df[(df['ma'] == t) & (df['chi_tieu'] == 'cfo')]\n"
            f"    top = df[(df['ma'] == t) & (df['chi_tieu'] == {num!r})]\n"
            f"    bot = df[(df['ma'] == t) & (df['chi_tieu'] == {den!r})]\n"
            f"    if len(cfo) == 0 or len(top) == 0 or len(bot) == 0:\n"
            f"        continue\n"
            f"    if float(cfo.iloc[0]['gia_tri_dong']) <= 0:\n"
            f"        continue\n"
            f"    den_v = float(bot.iloc[0]['gia_tri_dong'])\n"
            f"    if den_v == 0:\n"
            f"        continue\n"
            f"    val = float(top.iloc[0]['gia_tri_dong']) / den_v"
            f"{' * 100.0' if as_pct else ''}\n"
            f"    if best is None or val {cmp} best_val:\n"
            f"        best = t\n"
            f"        best_val = val\n"
            f"result = round(best_val, 2)"
        )
        return _pack(body, code, scores[winner], "screen_cfo_ratio",
                     ["cfo", num, den])

    return None
