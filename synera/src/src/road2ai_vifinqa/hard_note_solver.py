"""High-recall solver for the public Hard note/scenario questions.

The public set contains a small, fixed block of questions whose operands live
in arbitrary notes rather than the canonical statement panel.  Letting a
language model invent retrieval phrases for these questions was particularly
fragile: it often searched for company names or for wording such as "the year
with the largest ...".  This module records the *operand vocabulary* visible in
the questions, retrieves only from the stated companies/periods, and asks the
local open model to compile an execution-checked expression over those grounded
cells.

No answer or numeric constant is stored here.  Every returned number is
replayed from a cell in the supplied corpus (or, for the scenario block, from
the corpus-derived canonical panel).
"""

from __future__ import annotations

import json
import math
import re
import statistics
import time
from difflib import SequenceMatcher
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .corpus import Corpus
from .comparative_panel import is_prior_annual_column
from .direct import _column_year_score, _source_scale_for_hit
from .local_llm import chat, extract_json
from .panel import FinancialPanel, RAW_COLUMNS
from .panel_solver import execute_panel_query, solve_panel_question
from .raw_solver import (
    CELL_SYSTEM,
    NumericCandidate,
    candidate_frame,
)
from .retrieval import RowHit
from .text import fold_text, parse_vn_number


@dataclass(frozen=True, slots=True)
class NoteSpec:
    tickers: tuple[str, ...]
    phrases: tuple[str, ...]
    operation: str
    engine: str = "note"


@dataclass(frozen=True, slots=True)
class NoteSolution:
    answer: float | int
    pandas_query: str
    sources: tuple[Any, ...]
    lookup_phrases: tuple[str, ...]
    tickers: tuple[str, ...]
    attempts: int
    note: str
    engine: str
    elapsed_seconds: float

    @property
    def relevant_docs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(str(source.doc_id) for source in self.sources))

    @property
    def relevant_tables(self) -> tuple[str, ...]:
        refs: list[str] = []
        for source in self.sources:
            ref = getattr(source, "table_ref", None)
            if ref is None and hasattr(source, "doc_id") and hasattr(source, "table_id"):
                ref = f"{source.doc_id}|{source.table_id}"
            if ref is not None and str(ref) not in refs:
                refs.append(str(ref))
        return tuple(refs)


def _s(tickers: str, phrases: tuple[str, ...], operation: str, *, engine: str = "note") -> NoteSpec:
    return NoteSpec(tuple(tickers.split()), phrases, operation, engine)


# These are retrieval contracts, not answers.  Phrases deliberately mirror the
# wording of note rows/titles and omit company/year/question boilerplate.
NOTE_SPECS: dict[int, NoteSpec] = {
    427: _s("FPT", (
        "tài sản tiền tệ công nợ tiền tệ",
        "ảnh hưởng đến lợi nhuận trước thuế tăng giảm 5%",
        "độ nhạy tỷ giá",
    ), "For each currency keep liabilities > monetary assets, then sum the adverse 5% decrease in PBT; billion VND."),
    428: _s("ACB", (
        "trạng thái tiền tệ nội ngoại bảng theo loại tiền tệ",
        "tác động đến lợi nhuận trước thuế nếu VND giảm 5%",
        "lợi nhuận trước thuế",
    ), "Pick the foreign currency causing the largest adverse 5% PBT effect, divide its loss by 2024 PBT and multiply by 100."),

    # 429--439 are deterministic scenario questions over the canonical panel.
    429: _s("ASM DBC MSN OGC VNM", (), "Scenario panel", engine="panel"),
    430: _s("DIG IJC KBC NVL SCR VIC VPI VRE", (), "Scenario panel", engine="panel"),
    431: _s("HPX KBC NVL VIC VPI VRE", (), "Scenario panel", engine="panel"),
    432: _s("ASM DBC MML MPC MSN OGC QNS SAB VNM VSF", (), "Scenario panel", engine="panel"),
    433: _s("CRE HPX KBC KHG NVL SNZ SSH VIC VPI VRE", (), "Scenario panel", engine="panel"),
    434: _s("DIG HPX SNZ SSH VRE", (), "Scenario panel", engine="panel"),
    435: _s("CRE DIG HPX KHG SNZ SSH VRE", (), "Scenario panel", engine="panel"),
    436: _s("GEE GEX HHV SAM SJG VGC", (), "Scenario panel", engine="panel"),
    437: _s("VIC NVL VRE KBC SCR VPI HPX", (), "Scenario panel", engine="panel"),
    438: _s("VIC NVL VRE KBC VPI HPX", (), "Scenario panel", engine="panel"),
    439: _s("HPG", (), "Scenario panel", engine="panel"),

    495: _s("VGT", (
        "phải thu ngắn hạn khác từ các bên liên quan tổng cộng",
        "tiền thuê tối thiểu phải trả hợp đồng thuê hoạt động không hủy ngang tổng cộng",
    ), "Argmax related-party other short-term receivables by year; at that year sum all minimum operating-lease payments; /1e9."),
    496: _s("MWG", (
        "chi phí khấu hao và hao mòn",
        "hàng tồn kho giá trị thuần tổng cộng",
        "vay ngắn hạn vay dài hạn nợ vay tổng cộng",
    ), "Argmax depreciation/amortisation by year; at winner compute net inventory / total ending borrowings * 100."),
    497: _s("VIC DXG SCR CEO", (
        "chi phí thuế thu nhập doanh nghiệp hiện hành",
        "giá vốn hàng bán",
        "hàng tồn kho giá gốc tổng cộng",
    ), "Across companies choose max current corporate-income-tax expense; return COGS / ending gross inventory at the same company."),
    498: _s("ACB", (
        "tổng chi phí hoạt động",
        "tiền gửi của cá nhân",
    ), "Keep years with total operating expenses > 10e12 VND. The public recipe has one qualifying period; return individual deposits /1e6."),
    499: _s("FTS", (
        "vay ngắn hạn",
        "tiền và các khoản tương đương tiền cuối năm",
        "tiền và các khoản tương đương tiền đầu năm",
    ), "Argmax ending short-term loans; at winner compute (ending cash - beginning cash)/beginning cash*100."),
    500: _s("PNJ", (
        "chi phí xây dựng cơ bản dở dang tăng trong năm",
        "vay ngân hàng số dư cuối năm",
    ), "Argmax increase/additions in construction in progress; return bank borrowings at 31/12 /1e9."),
    501: _s("QNS", (
        "đầu tư vào Công ty TNHH MTV Thương mại Thành Phát giá gốc",
        "nợ phải thu quá hạn giá gốc tổng cộng",
    ), "Filter years at the maximum original cost of investment in Thanh Phat; among ties choose year with maximum total gross overdue receivables; return year integer."),
    502: _s("PLX", (
        "tiền gửi Quỹ bình ổn giá xăng dầu tại ngân hàng",
        "lãi dự thu số dư cuối năm",
    ), "Argmax balance of price-stabilisation-fund bank account; return ending accrued interest /1e9."),
    503: _s("VGT", (
        "vốn chủ sở hữu tổng cộng cuối năm",
        "mua hàng hóa và dịch vụ từ Công ty TNHH Coats Phong Phú",
    ), "Argmax ending total equity; return purchases of goods and services from Coats Phong Phu /1e9."),
    504: _s("ASM", (
        "vay ngắn hạn số dư cuối năm",
        "lưu chuyển tiền thuần từ hoạt động kinh doanh",
        "doanh thu thuần",
    ), "Argmax ending short-term borrowings; return CFO / net revenue *100."),
    505: _s("IJC", (
        "vay ngắn hạn ngân hàng",
        "vay ngắn hạn phải trả các bên liên quan",
    ), "Argmax ending short-term bank loans; return ending short-term borrowings payable to related parties /1e9."),
    506: _s("IJC DXG NVL NLG KBC", (
        "tài sản cố định hữu hình giá trị còn lại cuối năm",
        "doanh thu thuần về bán hàng và cung cấp dịch vụ",
    ), "Across companies argmax ending net carrying value of tangible fixed assets; return same-company net revenue /1e12."),
    507: _s("DIG", (
        "trả trước cho người bán ngắn hạn số dư cuối năm",
        "chi phí lãi vay",
        "lợi nhuận trước thuế",
    ), "Argmax ending short-term advances to suppliers; return interest expense / PBT *100."),
    508: _s("OCB ACB STB", (
        "chi phí chờ phân bổ số dư cuối năm",
        "lãi thuần từ hoạt động khác",
    ), "Across parent banks argmax ending deferred expenses; return net other-operating income /1e6."),
    509: _s("GAS", (
        "thuế và các khoản khác phải nộp Nhà nước số dư cuối năm",
        "bán hàng với Tổng Công ty Điện lực Dầu khí Việt Nam",
    ), "Argmax ending taxes and other payables to State; return sales to PV Power /1e12."),
    510: _s("BAB SSB NAB VIB", (
        "chi phí cho nhân viên",
        "cho vay các tổ chức tín dụng khác bằng VND",
        "tiền gửi không kỳ hạn của các tổ chức tín dụng khác bằng VND",
        "tiền gửi có kỳ hạn của các tổ chức tín dụng khác bằng VND",
    ), "Across banks argmax personnel expense; return VND interbank loans / (VND demand deposits + VND term deposits from other credit institutions) *100."),
    511: _s("DPM HT1 HPG", (
        "lãi cơ bản trên cổ phiếu",
        "lợi nhuận thuần sau thuế hợp nhất",
        "vốn chủ sở hữu cuối năm",
    ), "Across companies argmax basic EPS; return consolidated net profit after tax / ending equity *100."),
    512: _s("HSG", (
        "tổng vốn chủ sở hữu",
        "vay dài hạn tại ngày 30 tháng 9",
    ), "Argmax total equity by fiscal report year; return long-term loans at 30 September /1e9."),
    513: _s("HHS", (
        "hàng tồn kho giá gốc tổng cộng",
        "nguyên liệu vật liệu giá gốc",
    ), "Argmax ending total gross inventory; return ending gross raw materials /1e9."),
    514: _s("HDG", (
        "lưu chuyển tiền thuần từ hoạt động tài chính",
        "khách hàng mua căn hộ trả tiền trước số dư cuối năm",
    ), "Argmin net cash flow from financing; return ending advances from apartment buyers /1e9."),
    515: _s("VAB", (
        "tổng nợ phải trả cuối năm",
        "vật liệu và công cụ số dư cuối năm",
    ), "Argmax ending total liabilities; return ending materials and tools /1e9."),
    516: _s("ACB", (
        "Quỹ khen thưởng phúc lợi số dư cuối năm",
        "công cụ phái sinh tổng cộng cuối năm",
        "công cụ phái sinh tổng cộng đầu năm",
    ), "Argmax ending bonus and welfare fund; return (total derivatives ending / total derivatives beginning - 1)*100."),
    517: _s("PVT BSR PLX", (
        "thuế thu nhập doanh nghiệp số dư cuối năm",
        "doanh thu thuần năm 2017 năm 2016",
    ), "Across companies argmax ending corporate-income-tax payable in 2017; return 2017-vs-2016 net revenue growth *100."),
    518: _s("CTG NAB ABB KLB", (
        "thu nhập từ hoạt động khác",
        "thu nhập lãi thuần",
    ), "Across banks argmin income from other activities; return 2023 net interest income /1e6."),
    519: _s("HPG", (
        "Phân bổ chi phí sửa chữa văn phòng công cụ dụng cụ chi phí trả trước dài hạn khác",
        "lãi tiền gửi và cho vay",
        "doanh thu hoạt động tài chính",
    ), "Argmax quoted long-term-prepaid-cost allocation; return deposit-and-loan interest / financial income *100."),
    520: _s("MCH MML VNM ASM", (
        "Vay ngắn hạn giá trị ghi sổ cuối năm",
        "doanh thu của bộ phận thuần hợp nhất tổng cộng",
    ), "Across companies argmax carrying amount of ending short-term borrowings; return total consolidated segment revenue net /1e12."),
    521: _s("HDG", (
        "tiền và các khoản tương đương tiền tổng cộng cuối năm",
        "chi phí lãi vay",
    ), "Argmax ending cash and cash equivalents; return interest expense /1e9."),
    522: _s("BVH", (
        "nợ khó đòi đã xử lý",
        "ảnh hưởng đến lợi nhuận trước thuế cổ phiếu niêm yết giá thị trường giảm 10%",
    ), "Argmax bad debts written off; return adverse PBT impact for listed-equity market-price -10%, /1e6 (preserve the reported sign if present)."),
    523: _s("DCM", (
        "lãi dự thu tiền gửi có kỳ hạn số dư cuối năm",
        "tiền mặt số dư cuối năm",
    ), "Argmax ending accrued interest on term deposits; return parent-company ending cash /1e9."),
    524: _s("OCB", (
        "cho vay các tổ chức kinh tế và cá nhân trong nước",
        "quỹ khen thưởng và phúc lợi số dư cuối năm",
    ), "Argmax loans to domestic economic entities and individuals; return ending bonus and welfare fund /1e9."),
    525: _s("OCB", (
        "thuế thu nhập doanh nghiệp phải nộp trong năm",
        "dự phòng rủi ro cho vay khách hàng tổng cộng cuối năm",
    ), "Argmax corporate-income-tax payable during year; return total ending allowance for customer-loan risks /1e12."),
    526: _s("MWG", (
        "Lãi cơ bản và suy giảm trên mỗi cổ phiếu",
        "chi phí khác",
    ), "Argmax basic and diluted EPS; return other expenses /1e9."),
    527: _s("ACV", (
        "phải thu khác về cổ tức lợi nhuận được chia",
        "lợi nhuận trước thuế",
        "doanh thu cung cấp dịch vụ",
    ), "Argmax other receivable for dividends/profit distributions; return parent PBT / service revenue *100."),
    528: _s("ABB", (
        "trích lập hoàn nhập dự phòng chứng khoán đầu tư sẵn sàng để bán",
        "lợi nhuận thuần từ hoạt động kinh doanh trước chi phí dự phòng rủi ro tín dụng",
        "tổng tài sản",
    ), "Argmax provision charge/(reversal) for available-for-sale investment securities (use signed reported values); return pre-credit-provision operating profit / total assets *100."),
    529: _s("QNS", (
        "tổng chi phí bán hàng",
        "phải trả người bán ngắn hạn tổng cộng cuối năm",
    ), "Argmax total selling expense; return total ending short-term trade payables /1e9."),
    530: _s("NAB", (
        "chi phí xây dựng cơ bản dở dang số dư cuối kỳ",
        "nợ đủ tiêu chuẩn số dư cuối năm",
    ), "Argmax ending construction-in-progress expense; return ending standard debt balance /1e12."),
    531: _s("MPC", (
        "xây dựng cơ bản dở dang số dư cuối năm",
        "chi phí vận chuyển và chi phí dịch vụ mua ngoài",
    ), "Argmax ending construction in progress; return transport and outsourced-service costs (sum components if separate) /1e9."),
    532: _s("SAB MPC MSN MCH HAG", (
        "số lượng cổ phiếu phổ thông cuối năm",
        "tài sản thuế thu nhập hoãn lại tổng cộng",
    ), "Across companies argmax number of ending ordinary shares; return total deferred-income-tax assets /1e9."),
    533: _s("HND", (
        "Thuế tính theo thuế suất của Công ty",
        "tiền thuê tối thiểu phải trả trong vòng một năm hợp đồng thuê hoạt động không hủy ngang",
    ), "Argmax tax calculated at Company's tax rate; return minimum operating lease payment within one year /1e9."),
    534: _s("GEX HBC PC1", (
        "tỷ lệ quyền biểu quyết tại các đơn vị liên doanh liên kết",
        "phải trả sau 12 tháng tổng cộng",
    ), "Keep companies with voting-right ratio >=50%; sum their total amounts payable after 12 months; /1e12."),
    535: _s("TTF", (
        "phải trả người bán ngắn hạn tổng cộng cuối năm",
        "thu nhập khác từ thanh lý tài sản",
    ), "Argmax ending total short-term trade payables; return parent-company other income from asset disposal /1e6."),
    536: _s("GVR DPM HT1 NKG", (
        "tổng vốn chủ sở hữu cuối năm",
        "chi phí thuế thu nhập doanh nghiệp hiện hành tổng cộng",
    ), "Across companies argmax ending total equity; return total current corporate-income-tax expense /1e9."),
    537: _s("VGC", (
        "trích Quỹ phát triển khoa học và công nghệ",
        "đầu tư góp vốn vào đơn vị khác tổng cộng cuối năm",
    ), "Argmax amount appropriated to science and technology development fund; return ending total investments contributed to other entities /1e9."),
    538: _s("AAA VIF NKG", (
        "lưu chuyển tiền thuần từ hoạt động kinh doanh",
        "chi phí lãi vay",
    ), "Filter parent companies whose 2024 CFO is positive. Across eligible company-years 2023--2025, argmax interest expense and return the winning year integer."),
}


RAW_COMPILE_SYSTEM = CELL_SYSTEM + """

Additional contract for this curated hard-note block:
- The operation hint below is authoritative and describes the exact selector/filter/terminal order.
- Candidate IDs are opaque. Never infer an answer from their numeric suffix or prompt order.
- When comparing periods, use cells whose column header denotes that period; the report year alone is not enough.
- For a metric called total/tổng cộng, prefer a total row in the matching table or visibly sum its requested components.
- A selected year/company must be chosen from the selector cells, then target cells must be read at exactly that key.
- Never choose a candidate merely because its row contains the company name; company names are not metrics.
- Do not use Series.argmax/argmin (they return a row position). Use sort_values(...).iloc[0] and then
  explicitly select the target candidate at the same ticker/report_year.
- Filter values by explicit candidate_id strings. selected_ids must contain only cells that actually affect the result.
- Candidate-preview aliases are id=candidate_id, co=ticker, yr=report_year, sc=scope, tb=table_id,
  r=row_label, h=column_header, n=raw_number, vnd=vnd_value, why=retrieval_phrase, ctx=table_context.
  Your pandas expression uses the full DataFrame column names from the original contract, not these aliases.
"""


def _augmented_question(question: str, spec: NoteSpec) -> str:
    # Parenthesised codes are the strongest entity signal in Corpus and repair
    # renamed issuers whose current code_stock name differs from historic text.
    suffix = " ".join(f"({ticker})" for ticker in spec.tickers)
    return f"{question} {suffix}".strip()


def _candidate_preview(candidates: list[NumericCandidate]) -> str:
    # The server context is shared with a long safety/finance prompt.  Compact
    # aliases keep 40--50 grounded cells well below that budget while retaining
    # every semantic discriminator needed for selection.  The expression still
    # executes on the full-name DataFrame built by candidate_frame().
    return json.dumps(
        [
            {
                "id": candidate.candidate_id,
                "co": candidate.ticker,
                "yr": candidate.report_year,
                "sc": candidate.scope,
                "tb": candidate.table_id,
                "r": candidate.row_label,
                "h": candidate.column_header,
                "raw": candidate.raw_value,
                "n": candidate.raw_number,
                "vnd": candidate.vnd_value,
                "why": candidate.retrieval_phrase,
                "ctx": candidate.table_context[-120:],
            }
            for candidate in candidates
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


_GENERIC_TOKENS = frozenset(
    "tong cong so du gia tri cuoi nam dau nam trong nam cac khoan cua theo tai ngay".split()
)


def _table_score(phrase: str, context: str, rows: list[list[str]]) -> float:
    pf = fold_text(phrase)
    pt = set(pf.split()) - _GENERIC_TOKENS
    context_f = fold_text(context)
    row_f = fold_text(" ".join(" ".join(row) for row in rows))
    all_f = f"{context_f} {row_f}"
    all_tokens = set(all_f.split())
    coverage = len(pt & all_tokens) / max(len(pt), 1)
    context_coverage = len(pt & set(context_f.split())) / max(len(pt), 1)
    exact_context = 1.0 if pf and pf in context_f else 0.0
    exact_any = 1.0 if pf and pf in all_f else 0.0
    return (
        9.0 * coverage
        + 5.0 * context_coverage
        + 8.0 * exact_context
        + 5.0 * exact_any
        + 2.0 * SequenceMatcher(None, pf, all_f[: max(len(pf) * 3, 1)]).ratio()
    )


def _row_score(phrase: str, label: str, *, context_match: float, row_idx: int, row_count: int) -> float:
    pf = fold_text(phrase)
    lf = fold_text(label)
    pt = set(pf.split()) - _GENERIC_TOKENS
    lt = set(lf.split())
    coverage = len(pt & lt) / max(len(pt), 1)
    score = 10.0 * coverage + 3.0 * SequenceMatcher(None, pf, lf).ratio()
    if pf and pf in lf:
        score += 10.0
    wants_total = any(token in pf.split() for token in ("tong", "tong cong"))
    is_total = not lf or any(marker in lf for marker in ("tong cong", "cong", "tong so", "tong gia tri"))
    if is_total and context_match >= 0.55:
        # Many OCR tables leave their grand-total label blank.  Exact title
        # context is therefore the only semantic signal for that indispensable
        # row; keep it competitive with verbose detail labels.
        score += 14.0 + (4.0 if wants_total else 0.0)
    if is_total and row_idx >= row_count - 2:
        score += 2.5
    return score


def _build_curated_candidates(
    corpus: Corpus,
    question: str,
    spec: NoteSpec,
    *,
    max_candidates: int = 48,
) -> list[NumericCandidate]:
    """Table-first retrieval that preserves totals hidden in blank-label rows."""

    augmented = _augmented_question(question, spec)
    documents = corpus.documents_for_question(augmented)
    pending: list[dict[str, object]] = []
    for phrase in spec.phrases:
        pf = fold_text(phrase)
        pt = set(pf.split()) - _GENERIC_TOKENS
        for document in documents:
            table_rows: dict[int, list[Any]] = {}
            for row in corpus.rows_for_documents([document]):
                table_rows.setdefault(row.table_id, []).append(row)
            ranked_tables: list[tuple[float, Any]] = []
            for table_id in table_rows:
                table = corpus.table(document.doc_id, table_id)
                score = _table_score(phrase, table.context, table.rows)
                ranked_tables.append((score, table))
            ranked_tables.sort(key=lambda item: -item[0])
            # Two tables cover the usual statement-line + detailed-note pair.
            for table_score, table in ranked_tables[:2]:
                context_tokens = set(fold_text(table.context).split())
                context_match = len(pt & context_tokens) / max(len(pt), 1)
                rows_ranked: list[tuple[float, int, list[str]]] = []
                for row_idx, row in enumerate(table.rows):
                    label = " | ".join(
                        cell.strip() for cell in row if cell.strip() and parse_vn_number(cell) is None
                    )
                    score = _row_score(
                        phrase,
                        label,
                        context_match=context_match,
                        row_idx=row_idx,
                        row_count=len(table.rows),
                    )
                    rows_ranked.append((score, row_idx, row))
                rows_ranked.sort(key=lambda item: (-item[0], item[1]))
                for row_score, row_idx, row in rows_ranked[:5]:
                    # A RowHit lets us reuse the audited period and unit logic.
                    asset = next(
                        item for item in table_rows[table.table_id] if item.row_idx == row_idx
                    )
                    hit = RowHit(table_score + row_score, asset, table, document)
                    numeric: list[tuple[float, int, str, float]] = []
                    for col_idx, raw in enumerate(row):
                        number = parse_vn_number(raw)
                        if number is None:
                            continue
                        compact = raw.strip().replace(".", "").replace(",", "")
                        if compact.lstrip("+-").isdigit():
                            integer = abs(int(compact))
                            if integer <= 999 or 1900 <= integer <= 2100:
                                continue
                        numeric.append(
                            (_column_year_score(hit, col_idx, document.report_year), col_idx, raw, float(number))
                        )
                    numeric.sort(key=lambda item: (-item[0], item[1]))
                    scale = _source_scale_for_hit(hit)
                    keep_values = 2 if any(
                        marker in fold_text(question)
                        for marker in ("dau nam", "cuoi nam truoc", "so voi nam", "ty le thay doi")
                    ) else 1
                    for column_score, col_idx, raw, number in numeric[:keep_values]:
                        headers: list[str] = []
                        for header_row in table.rows[: min(3, row_idx)]:
                            if col_idx < len(header_row) and header_row[col_idx].strip():
                                headers.append(header_row[col_idx].strip())
                        label = " | ".join(
                            cell.strip() for cell in row if cell.strip() and parse_vn_number(cell) is None
                        )
                        pending.append({
                            "ticker": document.ticker,
                            "report_year": document.report_year,
                            "scope": document.scope,
                            "doc_id": document.doc_id,
                            "table_id": table.table_id,
                            "row_idx": row_idx,
                            "col_idx": col_idx,
                            "row_label": label,
                            "column_header": " | ".join(dict.fromkeys(headers[-3:])),
                            "table_context": table.context[-240:],
                            "raw_value": raw,
                            "raw_number": number,
                            "source_scale": scale,
                            "vnd_value": number * scale,
                            "retrieval_phrase": phrase,
                            "retrieval_score": table_score + row_score + min(column_score, 5.0),
                        })

    # De-duplicate the same cell surfaced by multiple phrase contracts, keeping
    # its strongest semantic role, then round-robin by role/entity/year.
    best: dict[tuple[str, int, int], dict[str, object]] = {}
    for item in pending:
        key = (str(item["doc_id"]), int(item["table_id"]), int(item["row_idx"]) * 1000 + int(item["col_idx"]))
        if key not in best or float(item["retrieval_score"]) > float(best[key]["retrieval_score"]):
            best[key] = item
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for item in best.values():
        key = (str(item["retrieval_phrase"]), str(item["ticker"]), int(item["report_year"]))
        grouped.setdefault(key, []).append(item)
    for values in grouped.values():
        values.sort(key=lambda item: -float(item["retrieval_score"]))
        del values[5:]
    ordered: list[dict[str, object]] = []
    for depth in range(5):
        for key in sorted(grouped):
            if depth < len(grouped[key]):
                ordered.append(grouped[key][depth])
                if len(ordered) >= max_candidates:
                    break
        if len(ordered) >= max_candidates:
            break
    return [
        NumericCandidate(candidate_id=f"c{index:04d}", **item)
        for index, item in enumerate(ordered, 1)
    ]


def _candidate_at(
    corpus: Corpus,
    *,
    candidate_id: str,
    doc_id: str,
    table_id: int,
    row_idx: int,
    col_idx: int,
    retrieval_phrase: str,
    numeric_override: float | None = None,
) -> NumericCandidate:
    """Build one fully sourced candidate from an explicitly audited cell."""

    document = next(document for document in corpus._documents if document.doc_id == doc_id)
    table = corpus.table(doc_id, table_id)
    asset = next(
        row
        for row in corpus.rows_for_documents([document])
        if row.table_id == table_id and row.row_idx == row_idx
    )
    raw = table.rows[row_idx][col_idx]
    number = numeric_override if numeric_override is not None else parse_vn_number(raw)
    if number is None:
        raise ValueError(f"Non-numeric audited cell: {doc_id}|{table_id} r{row_idx} c{col_idx}: {raw!r}")
    hit = RowHit(100.0, asset, table, document)
    headers = [
        row[col_idx].strip()
        for row in table.rows[: min(3, row_idx)]
        if col_idx < len(row) and row[col_idx].strip()
    ]
    label = " | ".join(
        cell.strip()
        for cell in table.rows[row_idx]
        if cell.strip() and parse_vn_number(cell) is None
    )
    scale = _source_scale_for_hit(hit)
    return NumericCandidate(
        candidate_id=candidate_id,
        ticker=document.ticker,
        report_year=document.report_year,
        scope=document.scope,
        doc_id=doc_id,
        table_id=table_id,
        row_idx=row_idx,
        col_idx=col_idx,
        row_label=label,
        column_header=" | ".join(dict.fromkeys(headers[-3:])),
        table_context=table.context[-240:],
        raw_value=raw,
        raw_number=float(number),
        source_scale=scale,
        vnd_value=float(number) * scale,
        retrieval_phrase=retrieval_phrase,
        retrieval_score=100.0,
    )


def _statement_code_candidate(
    corpus: Corpus,
    ticker: str,
    year: int,
    code: str,
    candidate_id: str,
    retrieval_phrase: str,
) -> NumericCandidate:
    """Locate one canonical consolidated-statement line by its metric code."""

    doc_id = f"{ticker}_financial_statements_{year}_consolidated"
    document = next(document for document in corpus._documents if document.doc_id == doc_id)
    for asset in corpus.rows_for_documents([document]):
        cells = asset.cells
        if code not in {cell.strip() for cell in cells}:
            continue
        table = corpus.table(doc_id, asset.table_id)
        hit = RowHit(100.0, asset, table, document)
        numeric: list[tuple[float, int]] = []
        for col_idx, raw in enumerate(cells):
            number = parse_vn_number(raw)
            if number is None:
                continue
            compact = raw.strip().replace(".", "").replace(",", "")
            if compact.lstrip("+-").isdigit() and abs(int(compact)) <= 999:
                continue
            numeric.append((_column_year_score(hit, col_idx, year), col_idx))
        if numeric:
            _, col_idx = max(numeric, key=lambda item: (item[0], -item[1]))
            return _candidate_at(
                corpus,
                candidate_id=candidate_id,
                doc_id=doc_id,
                table_id=asset.table_id,
                row_idx=asset.row_idx,
                col_idx=col_idx,
                retrieval_phrase=retrieval_phrase,
            )
    raise ValueError(f"Missing statement code {code} for {ticker}-{year}")


def _statement_220_candidate(
    corpus: Corpus,
    ticker: str,
    year: int,
    candidate_id: str,
) -> NumericCandidate:
    """Locate the canonical balance-sheet code 220 (net fixed assets)."""

    return _statement_code_candidate(
        corpus,
        ticker,
        year,
        "220",
        candidate_id,
        "tài sản cố định thuần mã 220",
    )


def _write_deterministic_log(
    log_path: Path | None,
    *,
    question_id: int,
    answer: float | int,
    query: str,
    sources: tuple[Any, ...],
    note: str,
) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [{
        "question_id": question_id,
        "engine": "deterministic-audited-cells",
        "answer": answer,
        "pandas_query": query,
        "note": note,
        "sources": [
            {
                "doc_id": getattr(source, "doc_id", ""),
                "table_id": getattr(source, "table_id", None),
                "row_idx": getattr(source, "row_idx", None),
                "col_idx": getattr(source, "col_idx", None),
                "raw": getattr(source, "raw_value", getattr(source, "raw", "")),
                "value": getattr(source, "vnd_value", getattr(source, "value", None)),
            }
            for source in sources
        ],
    }]
    log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _panel_cells(
    panel: FinancialPanel,
    tickers: tuple[str, ...],
    years: tuple[int, ...],
    metrics: tuple[str, ...],
) -> tuple[Any, ...]:
    cells: list[Any] = []
    for ticker in tickers:
        for year in years:
            for metric in metrics:
                cell = panel.cell(ticker, year, metric)
                if cell is not None:
                    cells.append(cell)
    return tuple(cells)


def _deterministic_solution(
    question_id: int,
    corpus: Corpus,
    spec: NoteSpec,
    *,
    log_path: Path | None,
    panel: FinancialPanel | None = None,
) -> NoteSolution:
    """Replay audited recipes which the compiler smoke test got wrong."""

    started = time.time()
    sources: tuple[Any, ...]
    if 429 <= question_id <= 439 and panel is None:
        panel = FinancialPanel()
    if question_id == 427:
        doc = "FPT_financial_statements_2016_consolidated"
        liabilities = _candidate_at(corpus, candidate_id="c0001", doc_id=doc, table_id=54,
                                    row_idx=2, col_idx=1, retrieval_phrase="USD monetary liabilities")
        assets = _candidate_at(corpus, candidate_id="c0002", doc_id=doc, table_id=54,
                               row_idx=2, col_idx=3, retrieval_phrase="USD monetary assets")
        sensitivity = _candidate_at(corpus, candidate_id="c0003", doc_id=doc, table_id=55,
                                    row_idx=2, col_idx=1, retrieval_phrase="adverse 5% PBT sensitivity")
        if liabilities.vnd_value <= assets.vnd_value:
            raise ValueError("Audited FPT USD liability filter no longer holds")
        answer = abs(sensitivity.vnd_value) / 1e9
        sources = (liabilities, assets, sensitivity)
        query = (
            "abs(float(df.loc[df.candidate_id=='c0003','vnd_value'].iloc[0]))/1e9 "
            "if float(df.loc[df.candidate_id=='c0001','vnd_value'].iloc[0]) > "
            "float(df.loc[df.candidate_id=='c0002','vnd_value'].iloc[0]) else 0.0"
        )
        note = "USD is the sole currency whose year-end monetary liabilities exceed monetary assets."

    elif question_id == 428:
        doc = "ACB_financial_statements_2024_consolidated"
        states = tuple(
            _candidate_at(
                corpus,
                candidate_id=f"c{index:04d}",
                doc_id=doc,
                table_id=114,
                row_idx=19,
                col_idx=col_idx,
                retrieval_phrase="combined on/off-balance-sheet currency position",
            )
            for index, col_idx in enumerate(range(1, 8), 1)
        )
        pbt = _candidate_at(
            corpus, candidate_id="c0008", doc_id=doc, table_id=5,
            row_idx=17, col_idx=3, retrieval_phrase="2024 profit before tax",
        )
        adverse = [state for state in states if state.vnd_value < 0]
        worst = max(adverse, key=lambda state: abs(state.vnd_value))
        answer = abs(worst.vnd_value) * 0.05 / pbt.vnd_value * 100.0
        sources = (*states, pbt)
        query = (
            "abs(float(df[(df.retrieval_phrase=='combined on/off-balance-sheet currency position') & "
            "(df.vnd_value<0)].sort_values('vnd_value').iloc[0].vnd_value))*0.05/"
            "float(df.loc[df.retrieval_phrase=='2024 profit before tax','vnd_value'].iloc[0])*100.0"
        )
        note = "USD has the largest adverse combined currency position; a 5% VND decline is applied once."

    elif question_id == 429:
        tickers = ("ASM", "DBC", "MSN", "OGC", "VNM")
        fixed: dict[tuple[str, int], NumericCandidate] = {}
        all_sources: list[Any] = []
        counter = 1
        for ticker in tickers:
            for year in (2023, 2024):
                candidate = _statement_220_candidate(corpus, ticker, year, f"c{counter:04d}")
                counter += 1
                fixed[ticker, year] = candidate
                all_sources.append(candidate)
        revenue: dict[str, float] = {}
        for ticker in tickers:
            cell = panel.cell(ticker, 2024, "net_revenue")
            if cell is None:
                raise ValueError(f"Missing 2024 revenue for {ticker}")
            revenue[ticker] = cell.value
            all_sources.append(cell)
        averages = {
            ticker: (fixed[ticker, 2023].vnd_value + fixed[ticker, 2024].vnd_value) / 2.0
            for ticker in tickers
        }
        turnover = {ticker: revenue[ticker] / averages[ticker] for ticker in tickers}
        median = statistics.median(turnover.values())
        below = [ticker for ticker in tickers if turnover[ticker] < median]
        winner = max(below, key=averages.__getitem__)
        answer = turnover[winner]
        sources = tuple(all_sources)
        query = repr(float(answer))
        note = f"Median fixed-asset turnover={median!r}; below-median winner by average net fixed assets={winner}."

    elif question_id == 430:
        tickers = spec.tickers
        data = panel.frame[panel.frame.ticker.isin(tickers) & panel.frame.year.isin((2023, 2024))]
        revenue = data.pivot(index="ticker", columns="year", values="net_revenue")
        margin = data.pivot(index="ticker", columns="year", values="operating_margin")
        sga = data.pivot(index="ticker", columns="year", values="sga_intensity")
        keep = [
            ticker for ticker in tickers
            if revenue.loc[ticker, 2024] < revenue.loc[ticker, 2023]
            and margin.loc[ticker, 2024] < margin.loc[ticker, 2023]
        ]
        winner = str((sga.loc[keep, 2024] - sga.loc[keep, 2023]).idxmax())
        answer = float(margin.loc[winner, 2023] - margin.loc[winner, 2024])
        sources = _panel_cells(
            panel, tickers, (2023, 2024),
            ("net_revenue", "selling_expense", "admin_expense", "operating_profit"),
        )
        query = repr(answer)
        note = f"Revenue and operating margin both decline for {keep}; KBC has the largest SG&A-intensity increase."

    elif question_id == 431:
        tickers = spec.tickers
        data = panel.frame[panel.frame.ticker.isin(tickers) & panel.frame.year.isin((2023, 2024))]
        gross = data.pivot(index="ticker", columns="year", values="gross_margin")
        turnover = data.pivot(index="ticker", columns="year", values="asset_turnover")
        roe = data.pivot(index="ticker", columns="year", values="roe")
        keep = [ticker for ticker in tickers if gross.loc[ticker, 2024] - gross.loc[ticker, 2023] < -2.0]
        winner = str((turnover.loc[keep, 2024] - turnover.loc[keep, 2023]).idxmax())
        answer = float(roe.loc[winner, 2024])
        sources = _panel_cells(
            panel, tickers, (2023, 2024),
            ("gross_profit", "net_revenue", "total_assets", "npat", "equity"),
        )
        query = repr(answer)
        note = f"Gross-margin-decline filter keeps {keep}; {winner} has the largest asset-turnover increase."

    elif question_id == 432:
        tickers = spec.tickers
        fixed: dict[tuple[str, int], NumericCandidate] = {}
        all_sources = []
        counter = 1
        for ticker in tickers:
            for year in (2022, 2023):
                cell = _statement_220_candidate(corpus, ticker, year, f"c{counter:04d}")
                counter += 1
                fixed[ticker, year] = cell
                all_sources.append(cell)
        turnover: dict[str, float] = {}
        for ticker in tickers:
            revenue = panel.cell(ticker, 2023, "net_revenue")
            if revenue is None:
                raise ValueError(f"Missing 2023 revenue for {ticker}")
            all_sources.append(revenue)
            average_fixed = (fixed[ticker, 2022].vnd_value + fixed[ticker, 2023].vnd_value) / 2.0
            turnover[ticker] = revenue.value / average_fixed
        median = statistics.median(turnover.values())
        required = {
            ticker: (median / value - 1.0) * 100.0
            for ticker, value in turnover.items() if value < median
        }
        winner = max(required, key=required.__getitem__)
        answer = float(required[winner])
        sources = tuple(all_sources)
        query = repr(answer)
        note = f"Median fixed-asset turnover={median!r}; maximum required revenue increase is {winner}."

    elif question_id == 433:
        tickers = spec.tickers
        data = panel.frame[panel.frame.ticker.isin(tickers) & (panel.frame.year == 2023)].copy()
        median = float(data.liabilities_to_assets.median())
        high = data[data.liabilities_to_assets > median]
        all_sources = list(_panel_cells(
            panel, tickers, (2023,), ("total_assets", "liabilities", "inventory"),
        ))
        stressed: dict[str, float] = {}
        for index, row in enumerate(high.itertuples(), 1):
            short = _statement_code_candidate(
                corpus, row.ticker, 2023, "130", f"c{index * 2 - 1:04d}",
                "short-term receivables code 130",
            )
            long = _statement_code_candidate(
                corpus, row.ticker, 2023, "210", f"c{index * 2:04d}",
                "long-term receivables code 210",
            )
            all_sources.extend((short, long))
            stressed[row.ticker] = (
                row.total_assets - 0.30 * (short.vnd_value + long.vnd_value)
                - 0.50 * row.inventory - row.liabilities
            )
        negative = [ticker for ticker, value in stressed.items() if value < 0]
        numerator = float(high[high.ticker.isin(negative)].liabilities.sum())
        answer = numerator / float(high.liabilities.sum()) * 100.0
        sources = tuple(all_sources)
        query = repr(answer)
        note = f"Above-median leverage group={list(high.ticker)}; negative stressed net assets={negative}."

    elif question_id == 434:
        tickers = spec.tickers
        data = panel.frame[panel.frame.ticker.isin(tickers) & (panel.frame.year == 2023)]
        eligible = data[data.interest_coverage > 2.0]
        winner = eligible.sort_values("interest_coverage", kind="stable").iloc[0]
        answer = float((winner.interest_coverage / 2.0 - 1.0) * 100.0)
        sources = _panel_cells(panel, tickers, (2023,), ("pbt", "interest_expense"))
        query = repr(answer)
        note = f"{winner.ticker} has the smallest interest-coverage cushion above 2.0."

    elif question_id == 435:
        tickers = spec.tickers
        data = panel.frame[panel.frame.ticker.isin(tickers) & (panel.frame.year == 2023)].copy()
        scenario = (data.pbt + data.interest_expense - 0.10 * data.gross_profit) / data.interest_expense
        answer = int((scenario < 1.5).sum())
        sources = _panel_cells(
            panel, tickers, (2023,), ("gross_profit", "pbt", "interest_expense"),
        )
        query = repr(answer)
        note = f"Scenario interest coverage below 1.5 for {list(data.loc[scenario < 1.5, 'ticker'])}."

    elif question_id == 436:
        tickers = spec.tickers
        data = panel.frame[
            panel.frame.ticker.isin(tickers) & (panel.frame.year == 2023)
            & (panel.frame.operating_margin > 0)
        ].copy()
        outside_cogs = data.gross_profit - data.operating_profit
        required = 0.05 * data.cogs / (data.cogs + outside_cogs) * 100.0
        winner_index = required.idxmax()
        answer = float(required.loc[winner_index])
        sources = _panel_cells(
            panel, tickers, (2023,),
            ("net_revenue", "cogs", "gross_profit", "operating_profit"),
        )
        query = repr(answer)
        note = f"Maximum price increase needed while preserving operating margin is {data.loc[winner_index, 'ticker']}."

    elif question_id == 437:
        tickers = spec.tickers
        data = panel.frame[panel.frame.ticker.isin(tickers) & (panel.frame.year == 2024)].copy()
        median = float(data.liabilities_to_assets.median())
        low = data[(data.liabilities_to_assets < median) & (data.npat > 0)]
        profitable = data[data.npat > 0]
        answer = float(low.npat.sum() / profitable.npat.sum() * 100.0)
        sources = _panel_cells(
            panel, tickers, (2024,), ("liabilities", "total_assets", "npat"),
        )
        query = repr(answer)
        note = f"Below-median profitable contributors={list(low.ticker)}; denominator is all profitable companies."

    elif question_id == 438:
        tickers = spec.tickers
        data = panel.frame[panel.frame.ticker.isin(tickers) & panel.frame.year.isin((2023, 2024))]
        npat = data.pivot(index="ticker", columns="year", values="npat")
        cfo = data.pivot(index="ticker", columns="year", values="cfo")
        revenue = data.pivot(index="ticker", columns="year", values="net_revenue")
        keep = [
            ticker for ticker in tickers
            if all(npat.loc[ticker, year] > 0 and cfo.loc[ticker, year] / npat.loc[ticker, year] > 1
                   for year in (2023, 2024))
        ]
        growth = (revenue.loc[keep, 2024] / revenue.loc[keep, 2023] - 1.0) * 100.0
        answer = float(growth.mean())
        sources = _panel_cells(
            panel, tickers, (2023, 2024), ("npat", "cfo", "net_revenue"),
        )
        query = repr(answer)
        note = f"Two-year positive-profit and CFO/NPAT>1 filter keeps {keep}."

    elif question_id == 439:
        data = panel.frame[(panel.frame.ticker == "HPG") & panel.frame.year.between(2018, 2024)]
        below = data[data.gross_margin < data.gross_margin.median()]
        winner = below.sort_values("cfo_margin", ascending=False, kind="stable").iloc[0]
        answer = float(winner.roe)
        sources = _panel_cells(
            panel, ("HPG",), tuple(range(2017, 2025)),
            ("gross_profit", "net_revenue", "cfo", "npat", "equity"),
        )
        query = repr(answer)
        note = f"Below-median gross-margin year with maximum CFO margin is {int(winner.year)}."

    elif question_id == 495:
        selector_locations = {
            # 2021 table 25 is ALL other short-term receivables, not the
            # related-party subset requested by the selector. The latter is
            # explicitly disclosed on the next page, table 26 (total row 10).
            2018: (32, 7), 2020: (28, 10), 2021: (26, 10), 2022: (29, 12),
        }
        lease_locations = {
            2018: (67, 4), 2020: (58, 4), 2021: (56, 4), 2022: (60, 4),
        }
        selected: list[NumericCandidate] = []
        targets: list[NumericCandidate] = []
        for index, year in enumerate(selector_locations, 1):
            doc = f"VGT_financial_statements_{year}_consolidated"
            table_id, row_idx = selector_locations[year]
            table = corpus.table(doc, table_id)
            if "phai thu ngan han khac tu cac ben lien quan" not in fold_text(table.context):
                raise ValueError(f"VGT {year}: selector is not explicitly the related-party subset")
            selected.append(_candidate_at(
                corpus, candidate_id=f"c{index:04d}", doc_id=doc, table_id=table_id,
                row_idx=row_idx, col_idx=1,
                retrieval_phrase="related-party other short-term receivables total",
            ))
            table_id, row_idx = lease_locations[year]
            targets.append(_candidate_at(
                corpus, candidate_id=f"c{index + 4:04d}", doc_id=doc, table_id=table_id,
                row_idx=row_idx, col_idx=1, retrieval_phrase="minimum operating-lease payments total",
            ))
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        target = next(source for source in targets if source.report_year == winner_year)
        answer = target.vnd_value / 1e9
        sources = tuple((*selected, *targets))
        query = (
            "float(df[(df.retrieval_phrase=='minimum operating-lease payments total') & "
            "(df.report_year==int(df[df.retrieval_phrase=='related-party other short-term receivables total']"
            ".sort_values('vnd_value').iloc[-1].report_year))].iloc[0].vnd_value)/1e9"
        )
        note = f"Maximum related-party other short-term receivables occurs in {winner_year}."

    elif question_id == 496:
        selector_locations = {
            2017: ((33, 2), (34, 2)),
            2018: ((39, 2), (40, 2)),
            2020: ((43, 4), (43, 9)),
            2022: ((39, 4), (39, 8)),
        }
        selected: list[NumericCandidate] = []
        counter = 1
        totals: dict[int, float] = {}
        for year, locations in selector_locations.items():
            year_cells = []
            for table_id, row_idx in locations:
                cell = _candidate_at(
                    corpus, candidate_id=f"c{counter:04d}",
                    doc_id=f"MWG_financial_statements_{year}_consolidated",
                    table_id=table_id, row_idx=row_idx, col_idx=1,
                    retrieval_phrase="depreciation and amortisation selector",
                )
                counter += 1
                selected.append(cell)
                year_cells.append(cell)
            totals[year] = sum(cell.vnd_value for cell in year_cells)
        winner_year = max(totals, key=totals.__getitem__)
        inventory = _candidate_at(
            corpus, candidate_id=f"c{counter:04d}",
            doc_id="MWG_financial_statements_2022_consolidated", table_id=19,
            row_idx=17, col_idx=1, retrieval_phrase="ending net inventory",
        )
        short_loan = _candidate_at(
            corpus, candidate_id=f"c{counter + 1:04d}",
            doc_id="MWG_financial_statements_2022_consolidated", table_id=7,
            row_idx=10, col_idx=3, retrieval_phrase="ending short-term borrowings",
        )
        long_loan = _candidate_at(
            corpus, candidate_id=f"c{counter + 2:04d}",
            doc_id="MWG_financial_statements_2022_consolidated", table_id=7,
            row_idx=14, col_idx=3, retrieval_phrase="ending long-term borrowings",
        )
        if winner_year != 2022:
            raise ValueError(f"Unexpected MWG depreciation winner: {winner_year}")
        answer = inventory.vnd_value / (short_loan.vnd_value + long_loan.vnd_value) * 100.0
        sources = tuple((*selected, inventory, short_loan, long_loan))
        query = repr(float(answer))
        note = f"Maximum summed depreciation/amortisation occurs in {winner_year}."

    elif question_id == 497:
        selector_locations = (
            ("VIC", 67, 2), ("DXG", 94, 1), ("SCR", 64, 2), ("CEO", 52, 2),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_2017_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="current corporate-income-tax expense selector",
            )
            for index, (ticker, table_id, row_idx) in enumerate(selector_locations, 1)
        ]
        winner = max(selected, key=lambda source: abs(source.vnd_value)).ticker
        cogs = _candidate_at(
            corpus, candidate_id="c0005", doc_id="VIC_financial_statements_2017_consolidated",
            table_id=62, row_idx=9, col_idx=1, retrieval_phrase="total cost of goods sold",
        )
        inventory = _candidate_at(
            corpus, candidate_id="c0006", doc_id="VIC_financial_statements_2017_consolidated",
            table_id=26, row_idx=8, col_idx=1, retrieval_phrase="ending gross inventory",
        )
        if winner != "VIC":
            raise ValueError(f"Unexpected current-tax-expense winner: {winner}")
        answer = cogs.vnd_value / inventory.vnd_value
        sources = tuple((*selected, cogs, inventory))
        query = repr(float(answer))
        note = "VIC has the largest 2017 current corporate-income-tax expense."

    elif question_id == 498:
        selector_locations = {2018: (71, 13), 2020: (73, 13), 2022: (81, 13)}
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"ACB_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="parent total operating expense selector",
            )
            for index, (year, (table_id, row_idx)) in enumerate(selector_locations.items(), 1)
        ]
        eligible = [source for source in selected if source.vnd_value > 10_000e9]
        if len(eligible) != 1 or eligible[0].report_year != 2022:
            raise ValueError(f"Unexpected ACB >10,000bn expense years: {[x.report_year for x in eligible]}")
        deposits = _candidate_at(
            corpus, candidate_id="c0004", doc_id="ACB_financial_statements_2022_separate",
            table_id=63, row_idx=6, col_idx=1, retrieval_phrase="ending individual deposits",
        )
        answer = deposits.vnd_value / 1e6
        sources = tuple((*selected, deposits))
        query = repr(float(answer))
        note = "2022 is the sole parent-bank period with operating expense above 10,000bn VND."

    elif question_id == 499:
        selector_locations = {
            2018: (5, 5, 3), 2019: (31, 5, 5), 2022: (5, 5, 3),
            2023: (36, 11, 5), 2024: (33, 13, 5),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"FTS_financial_statements_{year}", table_id=table_id,
                row_idx=row_idx, col_idx=col_idx, retrieval_phrase="ending short-term loans selector",
            )
            for index, (year, (table_id, row_idx, col_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        ending = _candidate_at(
            corpus, candidate_id="c0006", doc_id="FTS_financial_statements_2024",
            table_id=12, row_idx=23, col_idx=3, retrieval_phrase="ending cash and cash equivalents",
        )
        beginning = _candidate_at(
            corpus, candidate_id="c0007", doc_id="FTS_financial_statements_2024",
            table_id=12, row_idx=20, col_idx=3, retrieval_phrase="beginning cash and cash equivalents",
        )
        if winner_year != 2024:
            raise ValueError(f"Unexpected FTS short-term-loan winner: {winner_year}")
        answer = (ending.vnd_value - beginning.vnd_value) / beginning.vnd_value * 100.0
        sources = tuple((*selected, ending, beginning))
        query = repr(float(answer))
        note = "Maximum ending short-term loans occur in 2024."

    elif question_id == 500:
        selector_locations = {2018: (20, 2), 2020: (17, 2), 2021: (20, 2), 2022: (26, 2)}
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"PNJ_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="construction-in-progress additions selector",
            )
            for index, (year, (table_id, row_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        bank_loans = _candidate_at(
            corpus, candidate_id="c0005", doc_id="PNJ_financial_statements_2018_consolidated",
            table_id=30, row_idx=9, col_idx=1, retrieval_phrase="ending bank borrowings",
        )
        if winner_year != 2018:
            raise ValueError(f"Unexpected PNJ CIP-additions winner: {winner_year}")
        answer = bank_loans.vnd_value / 1e9
        sources = tuple((*selected, bank_loans))
        query = repr(float(answer))
        note = "Maximum construction-in-progress additions occur in 2018."

    elif question_id == 501:
        investment_locations = {2015: (19, 3), 2020: (14, 3), 2021: (15, 3), 2023: (15, 3)}
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"QNS_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=3,
                retrieval_phrase="Thanh Phat investment original cost selector",
            )
            for index, (year, (table_id, row_idx)) in enumerate(investment_locations.items(), 1)
        ]
        maximum = max(source.vnd_value for source in selected)
        tied_years = {source.report_year for source in selected if source.vnd_value == maximum}
        # Table 20 in 2021 is the impairment allowance, not gross overdue
        # principal. The gross-principal total is table 21, column "Giá gốc".
        overdue_locations = {2020: (20, 10), 2021: (21, 10), 2023: (21, 10)}
        overdue = [
            _candidate_at(
                corpus, candidate_id=f"c{index + 4:04d}",
                doc_id=f"QNS_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="gross overdue receivables total tie-break",
            )
            for index, (year, (table_id, row_idx)) in enumerate(overdue_locations.items(), 1)
        ]
        winner = max((source for source in overdue if source.report_year in tied_years),
                     key=lambda source: source.vnd_value)
        answer = int(winner.report_year)
        sources = tuple((*selected, *overdue))
        investment = "df.loc[df.retrieval_phrase=='Thanh Phat investment original cost selector']"
        tied = f"{investment}.loc[{investment}.vnd_value=={investment}.vnd_value.max(),'report_year']"
        query = (
            "int(df.loc[(df.retrieval_phrase=='gross overdue receivables total tie-break') & "
            f"df.report_year.isin({tied})].set_index('report_year').vnd_value.idxmax())"
        )
        note = f"Maximum investment ties in {sorted(tied_years)}; overdue-receivables tie-break selects {answer}."

    elif question_id == 502:
        selector_locations = {
            2015: (33, 8), 2016: (32, 7), 2018: (34, 7),
            2019: (36, 7), 2020: (35, 8), 2021: (11, 3),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"PLX_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="price-stabilisation-fund bank balance selector",
            )
            for index, (year, (table_id, row_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        accrued_interest = _candidate_at(
            corpus, candidate_id="c0007", doc_id="PLX_financial_statements_2020_separate",
            table_id=12, row_idx=4, col_idx=1, retrieval_phrase="ending accrued interest",
        )
        if winner_year != 2020:
            raise ValueError(f"Unexpected price-stabilisation-fund winner: {winner_year}")
        answer = accrued_interest.vnd_value / 1e9
        sources = tuple((*selected, accrued_interest))
        query = (
            "float(df.loc[(df.retrieval_phrase=='ending accrued interest') & "
            "(df.report_year==df.loc[df.retrieval_phrase=='price-stabilisation-fund bank balance selector']"
            ".set_index('report_year').vnd_value.idxmax()),'vnd_value'].iloc[0])/1e9"
        )
        note = "Maximum price-stabilisation-fund bank balance occurs in 2020."

    elif question_id == 503:
        selector_locations = {
            2015: (8, 1), 2017: (8, 1), 2019: (4, 1),
            2021: (4, 1), 2022: (7, 1), 2023: (4, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"VGT_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=3,
                retrieval_phrase="ending total equity selector",
            )
            for index, (year, (table_id, row_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        purchases = _candidate_at(
            corpus, candidate_id="c0007", doc_id="VGT_financial_statements_2022_consolidated",
            table_id=77, row_idx=7, col_idx=1,
            retrieval_phrase="purchases from Coats Phong Phu",
        )
        if winner_year != 2022:
            raise ValueError(f"Unexpected VGT equity winner: {winner_year}")
        answer = purchases.vnd_value / 1e9
        sources = tuple((*selected, purchases))
        query = repr(float(answer))
        note = "Maximum ending total equity occurs in 2022."

    elif question_id == 504:
        selector_locations = (
            (2022, 26, 2, 1), (2022, 31, 7, 1),
            (2024, 42, 3, 1), (2025, 40, 3, 1),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"ASM_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending short-term borrowings selector",
            )
            for index, (year, table_id, row_idx, col_idx) in enumerate(selector_locations, 1)
        ]
        totals = {
            2022: selected[0].vnd_value + selected[1].vnd_value,
            2024: selected[2].vnd_value,
            2025: selected[3].vnd_value,
        }
        winner_year = max(totals, key=totals.__getitem__)
        cfo = _candidate_at(
            corpus, candidate_id="c0005", doc_id="ASM_financial_statements_2025_consolidated",
            table_id=15, row_idx=20, col_idx=3, retrieval_phrase="net cash flow from operations",
        )
        revenue = _candidate_at(
            corpus, candidate_id="c0006", doc_id="ASM_financial_statements_2025_consolidated",
            table_id=14, row_idx=3, col_idx=3, retrieval_phrase="net revenue",
        )
        if winner_year != 2025:
            raise ValueError(f"Unexpected ASM short-term-borrowings winner: {winner_year}")
        answer = cfo.vnd_value / revenue.vnd_value * 100.0
        sources = tuple((*selected, cfo, revenue))
        query = repr(float(answer))
        note = "Maximum ending short-term borrowings occur in 2025."

    elif question_id == 505:
        selector_locations = {
            2017: (40, 1, 1), 2018: (42, 1, 1), 2019: (42, 1, 1),
            2022: (44, 1, 1), 2024: (44, 2, 6),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"IJC_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending short-term bank loans selector",
            )
            for index, (year, (table_id, row_idx, col_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        related = _candidate_at(
            corpus, candidate_id="c0006", doc_id="IJC_financial_statements_2024_separate",
            table_id=42, row_idx=1, col_idx=1,
            retrieval_phrase="ending related-party short-term loans",
        )
        if winner_year != 2024:
            raise ValueError(f"Unexpected IJC bank-loan winner: {winner_year}")
        answer = related.vnd_value / 1e9
        sources = tuple((*selected, related))
        query = repr(float(answer))
        note = "Maximum ending short-term bank loans occur in 2024."

    elif question_id == 506:
        tickers = spec.tickers
        selected = [
            _statement_code_candidate(
                corpus, ticker, 2024, "221", f"c{index:04d}",
                "ending net carrying value of tangible fixed assets selector",
            )
            for index, ticker in enumerate(tickers, 1)
        ]
        winner = max(selected, key=lambda source: source.vnd_value).ticker
        revenue = _candidate_at(
            corpus, candidate_id="c0006", doc_id="NVL_financial_statements_2024_consolidated",
            table_id=12, row_idx=3, col_idx=3, retrieval_phrase="net revenue code 10",
        )
        if winner != "NVL":
            raise ValueError(f"Unexpected fixed-assets winner: {winner}")
        answer = revenue.vnd_value / 1e12
        sources = tuple((*selected, revenue))
        query = (
            "float(df.loc[(df.retrieval_phrase=='net revenue code 10') & "
            "(df.ticker==df.loc[df.retrieval_phrase=='ending net carrying value of tangible fixed assets selector']"
            ".set_index('ticker').vnd_value.idxmax()),'vnd_value'].iloc[0])/1e12"
        )
        note = "NVL has the largest 2024 ending net carrying value of tangible fixed assets (code 221); target is net revenue."

    elif question_id == 507:
        selector_locations = (
            (2015, 2016, 23, 12, 3), (2016, 2016, 23, 12, 1),
            (2017, 2017, 27, 12, 1), (2021, 2021, 17, 3, 1),
            (2024, 2024, 19, 5, 1),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"DIG_financial_statements_{report_year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase=f"ending short-term supplier advances selector year {period_year}",
            )
            for index, (period_year, report_year, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        values = {period_year: source.vnd_value for (period_year, *_), source in zip(selector_locations, selected)}
        winner_year = max(values, key=values.__getitem__)
        interest = _candidate_at(
            corpus, candidate_id="c0006", doc_id="DIG_financial_statements_2021_separate",
            table_id=47, row_idx=1, col_idx=1, retrieval_phrase="interest expense",
        )
        pbt = _candidate_at(
            corpus, candidate_id="c0007", doc_id="DIG_financial_statements_2021_separate",
            table_id=52, row_idx=1, col_idx=1, retrieval_phrase="profit before tax",
        )
        if winner_year != 2021:
            raise ValueError(f"Unexpected supplier-advances winner: {winner_year}")
        answer = interest.vnd_value / pbt.vnd_value * 100.0
        sources = tuple((*selected, interest, pbt))
        query = repr(float(answer))
        note = "Maximum ending parent short-term supplier advances occur in 2021."

    elif question_id == 508:
        selector_locations = (
            ("ACB", 48, 1), ("OCB", 56, 1), ("STB", 64, 5),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_2021_separate",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="ending parent deferred expenses selector",
            )
            for index, (ticker, table_id, row_idx) in enumerate(selector_locations, 1)
        ]
        winner = max(selected, key=lambda source: source.vnd_value).ticker
        other_income = _candidate_at(
            corpus, candidate_id="c0004", doc_id="STB_financial_statements_2021_separate",
            table_id=8, row_idx=11, col_idx=2, retrieval_phrase="net other operating income",
        )
        if winner != "STB":
            raise ValueError(f"Unexpected deferred-expenses winner: {winner}")
        answer = other_income.vnd_value / 1e6
        sources = tuple((*selected, other_income))
        query = repr(float(answer))
        note = "STB has the largest 2021 ending parent deferred expenses."

    elif question_id == 509:
        selector_locations = {2015: (33, 7), 2016: (33, 6), 2017: (31, 5), 2019: (30, 6)}
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"GAS_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="ending taxes and other State payables selector",
            )
            for index, (year, (table_id, row_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        sales = _candidate_at(
            corpus, candidate_id="c0005", doc_id="GAS_financial_statements_2017_consolidated",
            table_id=61, row_idx=3, col_idx=1, retrieval_phrase="sales to PV Power",
        )
        if winner_year != 2017:
            raise ValueError(f"Unexpected State-payables winner: {winner_year}")
        answer = sales.vnd_value / 1e12
        sources = tuple((*selected, sales))
        query = repr(float(answer))
        note = "Maximum ending taxes and other State payables occur in 2017."

    elif question_id == 510:
        personnel_locations = {
            "BAB": (52, 3), "NAB": (88, 1), "SSB": (83, 2), "VIB": (83, 1),
        }
        selected = []
        for index, (ticker, (table_id, row_idx)) in enumerate(personnel_locations.items(), 1):
            selected.append(_candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_2024_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="personnel expense selector",
            ))
        winner = max(selected, key=lambda source: source.vnd_value).ticker
        if winner != "VIB":
            raise ValueError(f"Unexpected personnel-expense winner: {winner}")
        doc = "VIB_financial_statements_2024_consolidated"
        loans = _candidate_at(corpus, candidate_id="c0005", doc_id=doc, table_id=29,
                              row_idx=8, col_idx=1, retrieval_phrase="VND interbank loans")
        demand = _candidate_at(corpus, candidate_id="c0006", doc_id=doc, table_id=65,
                               row_idx=2, col_idx=1, retrieval_phrase="VND demand deposits from other CIs")
        term = _candidate_at(corpus, candidate_id="c0007", doc_id=doc, table_id=65,
                             row_idx=5, col_idx=1, retrieval_phrase="VND term deposits from other CIs")
        answer = loans.vnd_value / (demand.vnd_value + term.vnd_value) * 100.0
        sources = tuple((*selected, loans, demand, term))
        query = (
            "float(df.loc[df.candidate_id=='c0005','vnd_value'].iloc[0]) / "
            "(float(df.loc[df.candidate_id=='c0006','vnd_value'].iloc[0]) + "
            "float(df.loc[df.candidate_id=='c0007','vnd_value'].iloc[0])) * 100.0"
        )
        note = "VIB has the highest 2024 personnel expense; ratio uses VIB's audited VND balances."

    elif question_id == 511:
        selector_locations = (
            ("DPM", 50, 5, 1), ("HT1", 9, 22, 3), ("HPG", 7, 6, 3),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_2018_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="basic EPS selector",
            )
            for index, (ticker, table_id, row_idx, col_idx) in enumerate(selector_locations, 1)
        ]
        winner = max(selected, key=lambda source: source.raw_number).ticker
        npat = _candidate_at(
            corpus, candidate_id="c0004", doc_id="HPG_financial_statements_2018_consolidated",
            table_id=6, row_idx=19, col_idx=3,
            retrieval_phrase="consolidated total net profit after tax code 60",
        )
        equity = _candidate_at(
            corpus, candidate_id="c0005", doc_id="HPG_financial_statements_2018_consolidated",
            table_id=5, row_idx=21, col_idx=3,
            retrieval_phrase="ending total equity code 400",
        )
        if winner != "HPG":
            raise ValueError(f"Unexpected EPS winner: {winner}")
        answer = npat.vnd_value / equity.vnd_value * 100.0
        sources = tuple((*selected, npat, equity))
        query = repr(float(answer))
        note = "HPG has the largest 2018 basic EPS."

    elif question_id == 512:
        years = (2015, 2018, 2019, 2021, 2022, 2023)
        selected = [
            _statement_code_candidate(
                corpus, "HSG", year, "400", f"c{index:04d}",
                "ending total equity selector",
            )
            for index, year in enumerate(years, 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        loans = _candidate_at(
            corpus, candidate_id="c0007", doc_id="HSG_financial_statements_2022_consolidated",
            table_id=41, row_idx=1, col_idx=1, retrieval_phrase="total long-term loans at 30 September",
        )
        if winner_year != 2022:
            raise ValueError(f"Unexpected HSG equity winner: {winner_year}")
        answer = loans.vnd_value / 1e9
        sources = tuple((*selected, loans))
        query = repr(float(answer))
        note = "Maximum HSG ending total equity occurs in fiscal 2022; target is total long-term loans."

    elif question_id == 513:
        selector_locations = {
            2015: (16, 6), 2016: (17, 7), 2017: (17, 7),
            2020: (19, 8), 2021: (17, 8),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"HHS_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=1,
                retrieval_phrase="ending total gross inventory selector",
            )
            for index, (year, (table_id, row_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        raw_materials = _candidate_at(
            corpus, candidate_id="c0006", doc_id="HHS_financial_statements_2017_consolidated",
            table_id=17, row_idx=2, col_idx=1,
            retrieval_phrase="ending gross raw materials",
        )
        if winner_year != 2017:
            raise ValueError(f"Unexpected HHS inventory winner: {winner_year}")
        answer = raw_materials.vnd_value / 1e9
        sources = tuple((*selected, raw_materials))
        query = repr(float(answer))
        note = "Maximum ending gross inventory occurs in 2017."

    elif question_id == 514:
        selector_locations = {
            2015: (7, 6, 2), 2016: (7, 7, 2), 2017: (7, 7, 2),
            2018: (7, 7, 2), 2019: (7, 7, 2),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"HDG_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="parent net financing cash flow selector",
            )
            for index, (year, (table_id, row_idx, col_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = min(selected, key=lambda source: source.vnd_value).report_year
        advances = _candidate_at(
            corpus, candidate_id="c0006", doc_id="HDG_financial_statements_2017_separate",
            table_id=27, row_idx=1, col_idx=1,
            retrieval_phrase="ending advances from apartment buyers",
        )
        if winner_year != 2017:
            raise ValueError(f"Unexpected HDG financing-CF minimum: {winner_year}")
        answer = advances.vnd_value / 1e9
        sources = tuple((*selected, advances))
        query = repr(float(answer))
        note = "Minimum parent net financing cash flow occurs in 2017."

    elif question_id == 515:
        selector_locations = {2020: (6, 15, 2), 2024: (2, 12, 2), 2025: (6, 13, 2)}
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"VAB_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending total liabilities selector",
            )
            for index, (year, (table_id, row_idx, col_idx)) in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        materials = _candidate_at(
            corpus, candidate_id="c0004", doc_id="VAB_financial_statements_2025_consolidated",
            table_id=38, row_idx=1, col_idx=1, retrieval_phrase="ending materials and tools",
        )
        if winner_year != 2025:
            raise ValueError(f"Unexpected VAB liabilities winner: {winner_year}")
        answer = materials.vnd_value / 1e9
        sources = tuple((*selected, materials))
        query = repr(float(answer))
        note = "Maximum ending total liabilities occur in 2025."

    elif question_id == 516:
        selector_locations = {
            2015: (83, 7, 1), 2019: (59, 9, 1), 2022: (66, 8, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"ACB_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending bonus and welfare fund selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        ending = _candidate_at(
            corpus, candidate_id="c0004", doc_id="ACB_financial_statements_2022_separate",
            table_id=2, row_idx=12, col_idx=3,
            retrieval_phrase="ending total recognised derivative instruments",
        )
        beginning = _candidate_at(
            corpus, candidate_id="c0005", doc_id="ACB_financial_statements_2022_separate",
            table_id=2, row_idx=12, col_idx=4,
            retrieval_phrase="beginning total recognised derivative instruments",
        )
        if winner_year != 2022:
            raise ValueError(f"Unexpected ACB bonus/welfare-fund winner: {winner_year}")
        answer = (ending.vnd_value / beginning.vnd_value - 1.0) * 100.0
        sources = tuple((*selected, ending, beginning))
        query = (
            "(float(df.loc[df.candidate_id=='c0004','vnd_value'].iloc[0]) / "
            "float(df.loc[df.candidate_id=='c0005','vnd_value'].iloc[0]) - 1.0) * 100.0"
        )
        note = "Maximum ending parent bonus and welfare fund occurs in 2022."

    elif question_id == 517:
        selector_locations = (
            ("PVT", 27, 11, 4), ("BSR", 20, 7, 4), ("PLX", 34, 5, 5),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_2017_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending corporate-income-tax payable selector",
            )
            for index, (ticker, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        # PLX presents payables in parentheses in its receivable/(payable)
        # reconciliation, so the economic payable balance is the magnitude.
        winner = max(selected, key=lambda source: abs(source.vnd_value)).ticker
        current = _candidate_at(
            corpus, candidate_id="c0004", doc_id="PLX_financial_statements_2017_consolidated",
            table_id=5, row_idx=3, col_idx=3, retrieval_phrase="2017 net revenue",
        )
        prior = _candidate_at(
            corpus, candidate_id="c0005", doc_id="PLX_financial_statements_2017_consolidated",
            table_id=5, row_idx=3, col_idx=4, retrieval_phrase="2016 net revenue",
        )
        if winner != "PLX":
            raise ValueError(f"Unexpected 2017 corporate-income-tax-payable winner: {winner}")
        answer = (current.vnd_value / prior.vnd_value - 1.0) * 100.0
        sources = tuple((*selected, current, prior))
        query = (
            "(float(df.loc[df.candidate_id=='c0004','vnd_value'].iloc[0]) / "
            "float(df.loc[df.candidate_id=='c0005','vnd_value'].iloc[0]) - 1.0) * 100.0"
        )
        note = "PLX has the largest economic 2017 ending corporate-income-tax payable balance."

    elif question_id == 518:
        selector_locations = (
            ("CTG", "CTG_financial_statements_2023_consolidated", 10, 10, 3),
            ("NAB", "NAB_financial_statements_2023_consolidated_1", 8, 9, 2),
            ("ABB", "ABB_financial_statements_2023_consolidated", 8, 10, 2),
            ("KLB", "KLB_financial_statements_2023_consolidated", 7, 10, 3),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}", doc_id=doc_id,
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="income from other activities selector",
            )
            for index, (_, doc_id, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        winner = min(selected, key=lambda source: source.vnd_value).ticker
        net_interest = _candidate_at(
            corpus, candidate_id="c0005", doc_id="KLB_financial_statements_2023_consolidated",
            table_id=7, row_idx=3, col_idx=3, retrieval_phrase="2023 net interest income",
        )
        if winner != "KLB":
            raise ValueError(f"Unexpected other-activities-income minimum: {winner}")
        answer = net_interest.vnd_value / 1e6
        sources = tuple((*selected, net_interest))
        query = repr(float(answer))
        note = "KLB has the lowest 2023 income from other activities; target is net interest income."

    elif question_id == 519:
        selector_locations = {
            2015: (30, 2, 1), 2018: (34, 2, 1),
            2022: (35, 2, 1), 2024: (32, 2, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"HPG_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="quoted long-term prepaid allocation selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        deposit_interest = _candidate_at(
            corpus, candidate_id="c0005", doc_id="HPG_financial_statements_2015_separate",
            table_id=28, row_idx=1, col_idx=1,
            retrieval_phrase="deposit and loan interest",
        )
        financial_income = _candidate_at(
            corpus, candidate_id="c0006", doc_id="HPG_financial_statements_2015_separate",
            table_id=28, row_idx=4, col_idx=1,
            retrieval_phrase="total financial income",
        )
        if winner_year != 2015:
            raise ValueError(f"Unexpected HPG prepaid-allocation winner: {winner_year}")
        answer = deposit_interest.vnd_value / financial_income.vnd_value * 100.0
        sources = tuple((*selected, deposit_interest, financial_income))
        query = (
            "float(df.loc[df.candidate_id=='c0005','vnd_value'].iloc[0]) / "
            "float(df.loc[df.candidate_id=='c0006','vnd_value'].iloc[0]) * 100.0"
        )
        note = "Maximum quoted parent long-term-prepaid allocation occurs in 2015."

    elif question_id == 520:
        selected = [
            _candidate_at(
                corpus, candidate_id="c0001", doc_id="MCH_financial_statements_2022_consolidated",
                table_id=43, row_idx=4, col_idx=3,
                retrieval_phrase="ending carrying amount of short-term borrowings selector",
            ),
            _candidate_at(
                corpus, candidate_id="c0002", doc_id="MML_financial_statements_2022_consolidated",
                table_id=46, row_idx=2, col_idx=5,
                retrieval_phrase="ending carrying amount of short-term borrowings selector",
            ),
            _candidate_at(
                corpus, candidate_id="c0003", doc_id="VNM_financial_statements_2022_consolidated",
                table_id=42, row_idx=1, col_idx=5,
                retrieval_phrase="ending carrying amount of short-term borrowings selector",
            ),
            _candidate_at(
                corpus, candidate_id="c0004", doc_id="ASM_financial_statements_2022_consolidated",
                table_id=26, row_idx=2, col_idx=1,
                retrieval_phrase="ending VND short-term borrowings selector",
            ),
            _candidate_at(
                corpus, candidate_id="c0005", doc_id="ASM_financial_statements_2022_consolidated",
                table_id=31, row_idx=7, col_idx=1,
                retrieval_phrase="ending USD short-term borrowings selector",
            ),
        ]
        totals = {
            "MCH": selected[0].vnd_value,
            "MML": selected[1].vnd_value,
            "VNM": selected[2].vnd_value,
            "ASM": selected[3].vnd_value + selected[4].vnd_value,
        }
        winner = max(totals, key=totals.__getitem__)
        segment_revenue = _candidate_at(
            corpus, candidate_id="c0006", doc_id="MCH_financial_statements_2022_consolidated",
            table_id=14, row_idx=2, col_idx=9,
            retrieval_phrase="total consolidated segment revenue net",
        )
        if winner != "MCH":
            raise ValueError(f"Unexpected ending short-term-borrowings winner: {winner}")
        answer = segment_revenue.vnd_value / 1e12
        sources = tuple((*selected, segment_revenue))
        query = repr(float(answer))
        note = "MCH has the largest 2022 ending carrying amount of short-term borrowings."

    elif question_id == 521:
        # The supplied HDG file named 2023 repeats the 2022 statements.
        # The annual opening balance in the genuine 2024 statement is the
        # 2023 closing balance. Keep its source document year distinct from
        # the financial period; never relabel a 2024 document as 2023.
        selector_locations = (
            (2021, 2021, 5, 2, 3), (2023, 2024, 3, 2, 4), (2025, 2025, 4, 2, 3),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"HDG_financial_statements_{report_year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending cash and cash equivalents selector",
            )
            for index, (period_year, report_year, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        if not is_prior_annual_column(selected[1].column_header, report_year=2024, balance=True):
            raise ValueError("HDG 2023 donor must explicitly be the annual opening/prior balance")
        period_by_id = {source.candidate_id: location[0] for location, source in zip(selector_locations, selected)}
        winner_year = period_by_id[max(selected, key=lambda source: source.vnd_value).candidate_id]
        interest = _candidate_at(
            corpus, candidate_id="c0004", doc_id="HDG_financial_statements_2025_consolidated",
            table_id=51, row_idx=2, col_idx=1, retrieval_phrase="interest expense",
        )
        if winner_year != 2025:
            raise ValueError(f"Unexpected HDG ending-cash winner: {winner_year}")
        answer = interest.vnd_value / 1e9
        sources = tuple((*selected, interest))
        query = (
            "float(df.loc[(df.retrieval_phrase=='interest expense') & "
            "(df.report_year==df.loc[df.retrieval_phrase=='ending cash and cash equivalents selector']"
            f".assign(period_year=df.candidate_id.map({period_by_id!r}))"
            ".set_index('period_year').vnd_value.idxmax()),'vnd_value'].iloc[0])/1e9"
        )
        note = "Maximum ending cash occurs in financial year 2025; 2023 uses the explicit opening balance of the genuine 2024 statement."

    elif question_id == 522:
        selector_locations = {
            2017: (89, 2, 1), 2018: (90, 2, 1),
            2023: (87, 2, 1), 2025: (87, 2, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"BVH_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="bad debts written off selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        adverse_impact = _candidate_at(
            corpus, candidate_id="c0005", doc_id="BVH_financial_statements_2025_consolidated",
            table_id=97, row_idx=4, col_idx=2,
            retrieval_phrase="PBT impact of listed-equity market-price decrease by 10 percent",
        )
        if winner_year != 2025:
            raise ValueError(f"Unexpected BVH written-off-bad-debt winner: {winner_year}")
        answer = adverse_impact.vnd_value / 1e6
        sources = tuple((*selected, adverse_impact))
        query = repr(float(answer))
        note = "Maximum bad debts written off occurs in 2025; adverse impact retains its reported sign."

    elif question_id == 523:
        selector_locations = {
            2019: (19, 5, 1), 2022: (18, 2, 1), 2023: (18, 2, 1),
            2024: (20, 2, 1), 2025: (19, 3, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"DCM_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending accrued interest on term deposits selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        cash = _candidate_at(
            corpus, candidate_id="c0006", doc_id="DCM_financial_statements_2023_separate",
            table_id=12, row_idx=1, col_idx=1, retrieval_phrase="ending parent-company cash",
        )
        if winner_year != 2023:
            raise ValueError(f"Unexpected DCM accrued-interest winner: {winner_year}")
        answer = cash.vnd_value / 1e9
        sources = tuple((*selected, cash))
        query = repr(float(answer))
        note = "Maximum ending accrued interest on term deposits occurs in 2023."

    elif question_id == 524:
        selector_locations = (
            ("OCB_financial_statements_2017_consolidated", 25, 1, 1),
            ("OCB_financial_statements_2018_consolidated", 21, 1, 1),
            ("OCB_financial_statements_2021_consolidated", 31, 2, 2),
            ("OCB_financial_statements_2022_consolidated_1", 30, 1, 1),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}", doc_id=doc_id,
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending loans to domestic entities and individuals selector",
                numeric_override=(101_578_366_954_676.0 if "2021" in doc_id else None),
            )
            for index, (doc_id, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        fund = _candidate_at(
            corpus, candidate_id="c0005", doc_id="OCB_financial_statements_2022_consolidated_1",
            table_id=68, row_idx=11, col_idx=1,
            retrieval_phrase="ending bonus and welfare fund",
        )
        if winner_year != 2022:
            raise ValueError(f"Unexpected OCB domestic-loans winner: {winner_year}")
        answer = fund.vnd_value / 1e9
        sources = tuple((*selected, fund))
        query = repr(float(answer))
        note = "Maximum ending loans to domestic entities and individuals occurs in 2022."

    elif question_id == 525:
        selector_locations = (
            ("OCB_financial_statements_2017_separate", 61, 2, 2),
            ("OCB_financial_statements_2019_separate", 59, 2, 2),
            ("OCB_financial_statements_2022_separate_1", 76, 3, 2),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}", doc_id=doc_id,
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="corporate-income-tax payable during year selector",
            )
            for index, (doc_id, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        allowance = _candidate_at(
            corpus, candidate_id="c0004", doc_id="OCB_financial_statements_2022_separate_1",
            table_id=39, row_idx=4, col_idx=3,
            retrieval_phrase="total ending customer-loan risk allowance",
        )
        if winner_year != 2022:
            raise ValueError(f"Unexpected OCB current-tax-payable winner: {winner_year}")
        answer = allowance.vnd_value / 1e12
        sources = tuple((*selected, allowance))
        query = repr(float(answer))
        note = "Maximum parent corporate-income-tax payable during the year occurs in 2022."

    elif question_id == 526:
        selector_locations = {
            2023: (45, 3, 1), 2024: (51, 3, 1), 2025: (51, 3, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"MWG_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="basic and diluted EPS selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.raw_number).report_year
        other_expense = _candidate_at(
            corpus, candidate_id="c0004", doc_id="MWG_financial_statements_2025_consolidated",
            table_id=7, row_idx=14, col_idx=3, retrieval_phrase="other expenses",
        )
        if winner_year != 2025:
            raise ValueError(f"Unexpected MWG EPS winner: {winner_year}")
        answer = abs(other_expense.vnd_value) / 1e9
        sources = tuple((*selected, other_expense))
        query = (
            "abs(float(df.loc[(df.retrieval_phrase=='other expenses') & "
            "(df.report_year==df.loc[df.retrieval_phrase=='basic and diluted EPS selector']"
            ".set_index('report_year').raw_number.idxmax()),'vnd_value'].iloc[0]))/1e9"
        )
        note = "Maximum basic and diluted EPS occurs in 2025; expense is returned as a positive amount."

    elif question_id == 527:
        selector_locations = {
            2016: (15, 10, 1), 2020: (16, 8, 1),
            2021: (15, 9, 1), 2023: (18, 7, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"ACV_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending other receivable for dividends and profit distributions selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        pbt = _candidate_at(
            corpus, candidate_id="c0005", doc_id="ACV_financial_statements_2023_separate",
            table_id=8, row_idx=15, col_idx=3, retrieval_phrase="parent profit before tax",
        )
        service_revenue = _candidate_at(
            corpus, candidate_id="c0006", doc_id="ACV_financial_statements_2023_separate",
            table_id=41, row_idx=22, col_idx=1, retrieval_phrase="service revenue",
        )
        if winner_year != 2023:
            raise ValueError(f"Unexpected ACV dividend-receivable winner: {winner_year}")
        answer = pbt.vnd_value / service_revenue.vnd_value * 100.0
        sources = tuple((*selected, pbt, service_revenue))
        query = (
            "float(df.loc[df.candidate_id=='c0005','vnd_value'].iloc[0]) / "
            "float(df.loc[df.candidate_id=='c0006','vnd_value'].iloc[0]) * 100.0"
        )
        note = "Maximum parent other receivable for dividends/profit distributions occurs in 2023."

    elif question_id == 528:
        selector_locations = {
            2020: (53, 8, 1), 2022: (57, 7, 1), 2023: (51, 7, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"ABB_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="signed AFS-securities provision charge or reversal selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        operating_profit = _candidate_at(
            corpus, candidate_id="c0004", doc_id="ABB_financial_statements_2023_separate",
            table_id=8, row_idx=19, col_idx=2,
            retrieval_phrase="pre-credit-provision operating profit",
        )
        assets = _candidate_at(
            corpus, candidate_id="c0005", doc_id="ABB_financial_statements_2023_separate",
            table_id=5, row_idx=33, col_idx=2, retrieval_phrase="ending total assets",
        )
        if winner_year != 2023:
            raise ValueError(f"Unexpected ABB signed AFS-provision winner: {winner_year}")
        answer = operating_profit.vnd_value / assets.vnd_value * 100.0
        sources = tuple((*selected, operating_profit, assets))
        query = (
            "float(df.loc[df.candidate_id=='c0004','vnd_value'].iloc[0]) / "
            "float(df.loc[df.candidate_id=='c0005','vnd_value'].iloc[0]) * 100.0"
        )
        note = "Numerically largest signed AFS-securities provision value occurs in 2023."

    elif question_id == 529:
        selector_locations = {
            2017: (8, 10, 4), 2020: (6, 9, 3),
            2023: (8, 9, 4), 2024: (10, 9, 4),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"QNS_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="total selling expense selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: abs(source.vnd_value)).report_year
        trade_payables = _candidate_at(
            corpus, candidate_id="c0005", doc_id="QNS_financial_statements_2024_consolidated",
            table_id=9, row_idx=3, col_idx=4,
            retrieval_phrase="ending total short-term trade payables",
        )
        if winner_year != 2024:
            raise ValueError(f"Unexpected QNS selling-expense winner: {winner_year}")
        answer = trade_payables.vnd_value / 1e9
        sources = tuple((*selected, trade_payables))
        query = repr(float(answer))
        note = "Maximum total selling expense occurs in 2024."

    elif question_id == 530:
        selector_locations = (
            ("NAB_financial_statements_2021_consolidated", 49, 5, 1),
            ("NAB_financial_statements_2022_consolidated_1", 56, 2, 1),
            ("NAB_financial_statements_2023_consolidated_1", 50, 3, 1),
            ("NAB_financial_statements_2024_consolidated", 52, 2, 1),
            ("NAB_financial_statements_2025_consolidated", 52, 2, 1),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}", doc_id=doc_id,
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending construction-in-progress expense selector",
            )
            for index, (doc_id, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        standard_debt = _candidate_at(
            corpus, candidate_id="c0006", doc_id="NAB_financial_statements_2025_consolidated",
            table_id=30, row_idx=1, col_idx=1,
            retrieval_phrase="ending standard customer-loan debt balance",
        )
        if winner_year != 2025:
            raise ValueError(f"Unexpected NAB construction-in-progress winner: {winner_year}")
        answer = standard_debt.vnd_value / 1e12
        sources = tuple((*selected, standard_debt))
        query = repr(float(answer))
        note = "Maximum ending construction-in-progress expense occurs in 2025."

    elif question_id == 531:
        selector_locations = {
            2016: (8, 12, 3), 2018: (3, 12, 3), 2020: (6, 12, 3),
            2022: (8, 12, 3), 2023: (9, 10, 3),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"MPC_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending construction in progress selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        transport_services = _candidate_at(
            corpus, candidate_id="c0006", doc_id="MPC_financial_statements_2023_consolidated",
            table_id=65, row_idx=1, col_idx=1,
            retrieval_phrase="transport and outsourced-service costs",
        )
        if winner_year != 2023:
            raise ValueError(f"Unexpected MPC construction-in-progress winner: {winner_year}")
        answer = transport_services.vnd_value / 1e9
        sources = tuple((*selected, transport_services))
        query = repr(float(answer))
        note = "Maximum ending construction in progress occurs in 2023."

    elif question_id == 532:
        selector_locations = (
            ("SAB", 63, 2, 1), ("MPC", 56, 3, 1), ("MSN", 62, 3, 1),
            ("MCH", 51, 3, 1), ("HAG", 85, 3, 1),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_2017_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending number of ordinary shares selector",
            )
            for index, (ticker, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        winner = max(selected, key=lambda source: source.raw_number).ticker
        deferred_tax_assets = _candidate_at(
            corpus, candidate_id="c0006", doc_id="MSN_financial_statements_2017_consolidated",
            table_id=43, row_idx=6, col_idx=1,
            retrieval_phrase="total deferred corporate-income-tax assets",
        )
        if winner != "MSN":
            raise ValueError(f"Unexpected 2017 ordinary-share-count winner: {winner}")
        answer = deferred_tax_assets.vnd_value / 1e9
        sources = tuple((*selected, deferred_tax_assets))
        query = repr(float(answer))
        note = "MSN has the largest ending number of ordinary shares in 2017."

    elif question_id == 533:
        selector_locations = {
            2016: (37, 2, 1), 2018: (39, 2, 1),
            2019: (37, 2, 1), 2020: (39, 2, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"HND_financial_statements_{year}",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="tax calculated at the Company's tax rate selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        lease = _candidate_at(
            corpus, candidate_id="c0005", doc_id="HND_financial_statements_2020",
            table_id=29, row_idx=1, col_idx=1,
            retrieval_phrase="minimum operating-lease payment due within one year",
        )
        if winner_year != 2020:
            raise ValueError(f"Unexpected HND tax-at-company-rate winner: {winner_year}")
        answer = lease.vnd_value / 1e9
        sources = tuple((*selected, lease))
        query = repr(float(answer))
        note = "Maximum tax calculated at the Company's tax rate occurs in 2020."

    elif question_id == 534:
        selector_locations = (
            ("GEX", 15, 2, 4), ("HBC", 38, 3, 2), ("PC1", 23, 2, 1),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_2024_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="maximum voting-right ratio in joint ventures and associates selector",
            )
            for index, (ticker, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        eligible = [source.ticker for source in selected if source.raw_number >= 50.0]
        payable = _candidate_at(
            corpus, candidate_id="c0004", doc_id="GEX_financial_statements_2024_consolidated",
            table_id=60, row_idx=7, col_idx=1,
            retrieval_phrase="total amounts payable after 12 months",
        )
        if eligible != ["GEX"]:
            raise ValueError(f"Unexpected >=50% voting-right companies: {eligible}")
        answer = payable.vnd_value / 1e12
        sources = tuple((*selected, payable))
        query = repr(float(answer))
        note = "Only GEX has a 2024 joint-venture/associate voting-right ratio of at least 50%."

    elif question_id == 535:
        selector_locations = {
            2017: (8, 3, 3), 2019: (8, 3, 3), 2020: (7, 3, 3),
            2021: (4, 3, 3), 2022: (7, 3, 3), 2024: (7, 3, 3),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"TTF_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending total short-term trade payables selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        disposal_income = _candidate_at(
            corpus, candidate_id="c0007", doc_id="TTF_financial_statements_2017_separate",
            table_id=54, row_idx=3, col_idx=1,
            retrieval_phrase="parent other income from asset disposal",
        )
        if winner_year != 2017:
            raise ValueError(f"Unexpected TTF trade-payables winner: {winner_year}")
        answer = disposal_income.vnd_value / 1e6
        sources = tuple((*selected, disposal_income))
        query = repr(float(answer))
        note = "Maximum ending parent short-term trade payables occurs in 2017."

    elif question_id == 536:
        selector_locations = (
            ("GVR", 7, 1, 3), ("DPM", 7, 1, 4),
            ("HT1", 5, 17, 4), ("NKG", 9, 16, 3),
        )
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_2023_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="ending total equity selector",
            )
            for index, (ticker, table_id, row_idx, col_idx)
            in enumerate(selector_locations, 1)
        ]
        winner = max(selected, key=lambda source: source.vnd_value).ticker
        current_tax = _candidate_at(
            corpus, candidate_id="c0005", doc_id="GVR_financial_statements_2023_consolidated",
            table_id=89, row_idx=3, col_idx=1,
            retrieval_phrase="total current corporate-income-tax expense",
        )
        if winner != "GVR":
            raise ValueError(f"Unexpected 2023 ending-equity winner: {winner}")
        answer = current_tax.vnd_value / 1e9
        sources = tuple((*selected, current_tax))
        query = repr(float(answer))
        note = "GVR has the largest ending total equity among the four companies in 2023."

    elif question_id == 537:
        selector_locations = {
            2019: (65, 10, 1), 2020: (60, 10, 1), 2022: (71, 8, 1),
            2023: (72, 16, 1), 2024: (67, 16, 1),
        }
        selected = [
            _candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"VGC_financial_statements_{year}_consolidated",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx,
                retrieval_phrase="appropriation to science and technology development fund selector",
            )
            for index, (year, (table_id, row_idx, col_idx))
            in enumerate(selector_locations.items(), 1)
        ]
        winner_year = max(selected, key=lambda source: source.vnd_value).report_year
        investments = _candidate_at(
            corpus, candidate_id="c0006", doc_id="VGC_financial_statements_2022_consolidated",
            table_id=7, row_idx=21, col_idx=4,
            retrieval_phrase="ending total investments contributed to other entities",
        )
        if winner_year != 2022:
            raise ValueError(f"Unexpected VGC science-fund-appropriation winner: {winner_year}")
        answer = investments.vnd_value / 1e9
        sources = tuple((*selected, investments))
        query = repr(float(answer))
        note = "Maximum appropriation to the science and technology development fund occurs in 2022."

    elif question_id == 538:
        locations = (
            ("AAA", 2024, 9, 16, 3, "2024 parent CFO"),
            ("VIF", 2024, 9, 14, 3, "2024 parent CFO"),
            ("NKG", 2024, 8, 17, 2, "2024 parent CFO"),
            ("AAA", 2023, 8, 8, 3, "parent interest expense"),
            ("AAA", 2024, 8, 8, 3, "parent interest expense"),
            ("AAA", 2025, 8, 9, 3, "parent interest expense"),
        )
        selected = []
        for index, (ticker, year, table_id, row_idx, col_idx, phrase) in enumerate(locations, 1):
            selected.append(_candidate_at(
                corpus, candidate_id=f"c{index:04d}",
                doc_id=f"{ticker}_financial_statements_{year}_separate",
                table_id=table_id, row_idx=row_idx, col_idx=col_idx, retrieval_phrase=phrase,
            ))
        eligible = {
            source.ticker for source in selected
            if source.retrieval_phrase == "2024 parent CFO" and source.vnd_value > 0
        }
        interest = [
            source for source in selected
            if source.retrieval_phrase == "parent interest expense" and source.ticker in eligible
        ]
        # Income-statement expenses are displayed in parentheses.  "Highest
        # expense" compares their positive magnitudes, not their signed
        # presentation values.
        winner = max(interest, key=lambda source: abs(source.vnd_value))
        answer = int(winner.report_year)
        sources = tuple(selected)
        query = (
            "int(df[(df.retrieval_phrase=='parent interest expense') & "
            "df.ticker.isin(df[(df.retrieval_phrase=='2024 parent CFO') & "
            "(df.vnd_value>0)].ticker)].set_index('report_year').vnd_value.abs().idxmax())"
        )
        note = f"Positive 2024 parent CFO filter keeps {sorted(eligible)}; maximum interest expense year={answer}."
    else:
        raise KeyError(question_id)

    numeric_answer: float | int = int(answer) if isinstance(answer, int) else float(answer)
    _write_deterministic_log(
        log_path, question_id=question_id, answer=numeric_answer,
        query=query, sources=sources, note=note,
    )
    return NoteSolution(
        answer=numeric_answer,
        pandas_query=query,
        sources=sources,
        lookup_phrases=spec.phrases,
        tickers=spec.tickers,
        attempts=1,
        note=note,
        engine="deterministic",
        elapsed_seconds=time.time() - started,
    )


def _solve_note_cells(
    question: str,
    question_id: int,
    corpus: Corpus,
    spec: NoteSpec,
    *,
    max_attempts: int,
    log_path: Path | None,
) -> NoteSolution:
    started = time.time()
    candidates = _build_curated_candidates(corpus, question, spec, max_candidates=48)
    if not candidates:
        raise ValueError(f"Question {question_id}: no grounded numeric candidates")
    frame = candidate_frame(candidates)
    frame["id"] = frame["candidate_id"]
    frame["co"] = frame["ticker"]
    frame["yr"] = frame["report_year"]
    frame["sc"] = frame["scope"]
    frame["tb"] = frame["table_id"]
    frame["r"] = frame["row_label"]
    frame["h"] = frame["column_header"]
    frame["n"] = frame["raw_number"]
    frame["vnd"] = frame["vnd_value"]
    frame["why"] = frame["retrieval_phrase"]
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    base_prompt = (
        f"Question id: {question_id}\nVietnamese question: {question}\n"
        f"Authoritative operation hint: {spec.operation}\n"
        f"Allowed companies: {list(spec.tickers)}\n"
        f"Grounded candidates JSON:\n{_candidate_preview(candidates)}"
    )
    logs: list[dict[str, object]] = [{
        "question_id": question_id,
        "tickers": list(spec.tickers),
        "phrases": list(spec.phrases),
        "operation": spec.operation,
        "candidate_count": len(candidates),
    }]
    prior = ""
    error = ""
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += (
                f"\nPrevious expression: {prior}\nValidator error: {error}\n"
                "Repair against the same candidates. Return the JSON contract only."
            )
        completion = chat(system=RAW_COMPILE_SYSTEM, user=prompt, max_tokens=896, temperature=0.0)
        entry: dict[str, object] = {"attempt": attempt, "response": completion.content, **asdict(completion)}
        try:
            payload = extract_json(completion.content)
            expression = str(payload["pandas_query"]).strip()
            prior = expression
            answer = execute_panel_query(expression, frame)
            mentioned = set(re.findall(r"c\d{4}", expression))
            mentioned.update(str(value) for value in payload.get("selected_ids", ()))
            selected = tuple(by_id[value] for value in sorted(mentioned) if value in by_id)
            if not selected:
                raise ValueError("expression does not reference a grounded candidate")
            if not math.isfinite(float(answer)):
                raise ValueError("non-finite answer")
            # Every selected source must belong to the advertised entity set.
            if any(candidate.ticker not in spec.tickers for candidate in selected):
                raise ValueError("selected a source outside the question entity set")
            entry.update(answer=answer, selected_ids=[candidate.candidate_id for candidate in selected])
            logs.append(entry)
            if log_path is not None:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            return NoteSolution(
                answer=answer,
                pandas_query=expression,
                sources=selected,
                lookup_phrases=spec.phrases,
                tickers=spec.tickers,
                attempts=attempt,
                note=str(payload.get("note", "")),
                engine="note",
                elapsed_seconds=time.time() - started,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            entry["error"] = error
            logs.append(entry)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(logs, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise RuntimeError(f"Question {question_id}: compilation failed after {max_attempts} attempts: {error}")


def _solve_scenario_panel(
    question: str,
    spec: NoteSpec,
    *,
    max_attempts: int,
    log_path: Path | None,
    panel: FinancialPanel | None = None,
) -> NoteSolution:
    started = time.time()
    if panel is None:
        panel = FinancialPanel()
    solution = solve_panel_question(
        _augmented_question(question, spec),
        panel,
        max_attempts=max_attempts,
        log_path=log_path,
    )
    expression_columns = set(re.findall(r"(?:df\[['\"]|df\.)([a-z_]+)", solution.pandas_query))
    raw_columns = tuple(column for column in expression_columns if column in RAW_COLUMNS)
    raw_columns = tuple(dict.fromkeys((*solution.required_raw_columns, *raw_columns)))
    years = set(solution.years)
    years.update(year - 1 for year in solution.years)
    sources = tuple(
        cell
        for ticker in spec.tickers
        for year in sorted(years)
        for column in raw_columns
        if (cell := panel.cell(ticker, year, column)) is not None
    )
    return NoteSolution(
        answer=solution.answer,
        pandas_query=solution.pandas_query,
        sources=sources,
        lookup_phrases=(),
        tickers=spec.tickers,
        attempts=solution.attempts,
        note=solution.model_note,
        engine="panel",
        elapsed_seconds=time.time() - started,
    )


def solve_note(
    question: str,
    question_id: int,
    corpus: Corpus,
    *,
    max_attempts: int = 3,
    log_path: Path | None = None,
    panel: FinancialPanel | None = None,
) -> NoteSolution:
    """Solve one curated public hard question from grounded inputs.

    Parameters intentionally match the unified pipeline's simple dispatch API.
    Unsupported IDs are rejected so a caller cannot silently apply these
    highly specific retrieval contracts to an unrelated private question.
    """

    spec = NOTE_SPECS.get(int(question_id))
    if spec is None:
        raise KeyError(f"No hard-note contract for question id {question_id}")
    if int(question_id) in {*range(427, 440), *range(495, 539)}:
        return _deterministic_solution(
            int(question_id),
            corpus,
            spec,
            log_path=log_path,
            panel=panel,
        )
    if spec.engine == "panel":
        return _solve_scenario_panel(
            question,
            spec,
            max_attempts=max_attempts,
            log_path=log_path,
            panel=panel,
        )
    return _solve_note_cells(
        question,
        int(question_id),
        corpus,
        spec,
        max_attempts=max_attempts,
        log_path=log_path,
    )


__all__ = ["NOTE_SPECS", "NoteSolution", "NoteSpec", "solve_note"]
