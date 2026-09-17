"""Hand the model a tidy metric frame instead of six OCR tables.

The generated-program branch answers 19.5% of the questions it takes, against
42.8% for deterministic lookup, and its failures are overwhelmingly long
programs: the ones that fail average 2,800 characters against 1,301 for the ones
that work. Most of that length is the model re-deriving, in every program, how
to find a line item and parse a Vietnamese number across several statements.

When every figure a question needs is already in the verified Circular 200
panel, none of that work is necessary. The model receives one small frame —
one row per company-year, one column per metric, values in đồng — and writes
arithmetic over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from vifin.query.parse import ParsedQuestion
from vifin.store import TableKey

# Ordered so the frame reads like a statement rather than a hash dump.
COLUMN_ORDER = (
    "net_revenue", "revenue_gross", "cogs", "gross_profit",
    "financial_income", "financial_expense", "interest_expense",
    "selling_expense", "admin_expense", "operating_profit",
    "profit_before_tax", "net_profit", "eps",
    "total_assets", "current_assets", "long_assets", "cash",
    "receivables_short", "inventory",
    "liabilities", "liabilities_short", "liabilities_long",
    "equity", "total_resources",
    "cfo", "cfi", "cff",
)


@dataclass(slots=True)
class PanelContext:
    rows: list[list[str]]
    tables: list[TableKey] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.missing and len(self.rows) > 1


def combos(question: ParsedQuestion) -> list[tuple[str, str, str]]:
    return [
        (ticker, str(year), question.scope)
        for ticker in question.tickers
        for year in sorted(question.years)
    ]


def build_context(
    question: ParsedQuestion,
    panel: dict[tuple[str, str, str], dict[str, float]],
    provenance: dict[tuple[str, str, str], dict[str, tuple[str, int]]],
    metrics: list[str],
) -> PanelContext:
    """A CSV grid of the requested metrics for every company-year in scope."""

    wanted = [m for m in COLUMN_ORDER if m in metrics]
    if not wanted:
        return PanelContext(rows=[], missing=list(metrics))

    header = ["ticker", "year", "scope"] + wanted
    rows: list[list[str]] = [header]
    sources: list[TableKey] = []
    missing: list[str] = []

    for key in combos(question):
        values = panel.get(key)
        if values is None:
            missing.append(f"{key[0]}/{key[1]}")
            continue
        line = [key[0], key[1], key[2]]
        for metric in wanted:
            value = values.get(metric)
            if value is None:
                missing.append(f"{key[0]}/{key[1]}:{metric}")
                line.append("")
                continue
            # Plain integers: the model must not have to parse anything.
            line.append(f"{value:.0f}")
            source = provenance.get(key, {}).get(metric)
            if source is not None:
                candidate = TableKey(source[0], source[1])
                if candidate not in sources:
                    sources.append(candidate)
        rows.append(line)

    return PanelContext(rows=rows, tables=sources, missing=missing)


# Ratios name themselves, not their operands: "biên lợi nhuận gộp" mentions
# gross_profit but never net_revenue, so metric extraction handed the model a
# numerator with no denominator and it could not compute anything. Each entry
# maps a phrase to every metric the formula needs.
#
# This is where the impossible answers came from. 273 of 1,012 submitted answers
# were raw đồng amounts reported against a "phần trăm" or "lần" question, because
# the fallback had no ratio to compute and fell back to a single cell. Those are
# wrong by construction; a ratio at least has a chance.
RATIO_OPERANDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("biên lợi nhuận gộp", ("gross_profit", "net_revenue")),
    ("biên lợi nhuận ròng", ("net_profit", "net_revenue")),
    ("biên lợi nhuận", ("gross_profit", "net_revenue")),
    ("roe", ("net_profit", "equity")),
    ("roa", ("net_profit", "total_assets")),
    ("nợ phải trả trên vốn chủ sở hữu", ("liabilities", "equity")),
    ("nợ trên vốn chủ sở hữu", ("liabilities", "equity")),
    ("d/e", ("liabilities", "equity")),
    ("thanh toán hiện hành", ("current_assets", "liabilities_short")),
    ("thanh toán nhanh", ("current_assets", "inventory", "liabilities_short")),
    ("khả năng thanh toán lãi vay", ("operating_profit", "interest_expense")),
    ("dòng tiền hoạt động trên doanh thu", ("cfo", "net_revenue")),
    ("cfo trên doanh thu", ("cfo", "net_revenue")),
    ("cfo trên lợi nhuận sau thuế", ("cfo", "net_profit")),
    ("cfo trên nợ ngắn hạn", ("cfo", "liabilities_short")),
    ("dồn tích", ("net_profit", "cfo", "total_assets")),
    ("vòng quay hàng tồn kho", ("cogs", "inventory")),
    ("vòng quay khoản phải thu", ("net_revenue", "receivables_short")),
    ("vòng quay tổng tài sản", ("net_revenue", "total_assets")),
    ("tăng trưởng doanh thu", ("net_revenue",)),
    ("tăng trưởng lợi nhuận", ("net_profit",)),
    ("tỷ trọng hàng tồn kho", ("inventory", "total_assets")),
    ("tỷ trọng nợ", ("liabilities", "total_assets")),
    ("vốn lưu động", ("current_assets", "liabilities_short")),
)


def expand_operands(question: str, metrics: list[str]) -> list[str]:
    """Add the metrics a named ratio needs but does not mention."""

    folded = _fold_simple(question)
    out = list(metrics)
    for phrase, operands in RATIO_OPERANDS:
        if phrase in folded:
            for operand in operands:
                if operand not in out:
                    out.append(operand)
    return out


# Content words that mark a note-level detail the panel cannot possibly hold.
_OUT_OF_SCOPE = (
    "giá gốc", "nguyên liệu", "vật liệu", "khách hàng mua", "đồng tiền",
    "ngoại tệ", "usd", "quỹ", "cổ đông", "công ty con", "liên kết",
    "thuê", "khấu hao", "dự phòng", "thuyết minh", "bộ phận", "chi nhánh",
)


def is_panel_question(question: str, metrics: list[str], aliases: dict[str, str]) -> bool:
    """Whether the panel can honestly answer, not merely produce a number.

    Alias matching alone routed "giá gốc nguyên liệu, vật liệu" and "số dư khách
    hàng mua căn hộ" to the panel because the sentence happened to mention a
    catalogue metric elsewhere. The model then answered confidently from data
    that had nothing to do with the question, and displaced the OCR branch that
    might have got it right.
    """

    if not metrics:
        return False
    folded = _fold_simple(question)
    # A whitelist of permitted words was tried first and rejected every one of
    # 287 candidates: real questions always contain vocabulary no hand-written
    # list anticipates. Excluding the note-level subjects the panel provably
    # cannot hold is the workable direction.
    return not any(marker in folded for marker in _OUT_OF_SCOPE)


def _fold_simple(text: str) -> str:
    import re

    return re.sub(r"[^\w\s]", " ", text.casefold())


PANEL_CLAUSE = """
<panel>
- `df` is NOT an OCR table. It is a clean panel: one row per company-year, with
  columns `ticker`, `year`, `scope` and one column per financial metric.
- Every metric value is already a plain number in đồng (VND). Do not strip dots
  or commas, and do not rescale unless the question asks for a different unit.
- An empty cell means the figure is unavailable; do not invent it.
- If the question asks "phần trăm"/"%", the answer is a percentage number:
  multiply the ratio by 100 (0.2134 -> 21.34), never report the ratio itself and
  never report a đồng amount.
- If it asks "bao nhiêu lần", report the plain ratio without multiplying.
- Select rows with ordinary comparisons, e.g.
      row = df[(df["ticker"] == "HPG") & (df["year"] == "2023")]
      value = float(row["net_revenue"].iloc[0])
</panel>
"""
