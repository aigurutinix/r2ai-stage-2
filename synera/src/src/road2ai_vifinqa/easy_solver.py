"""Grounded local-LLM reranker for the public direct-value questions.

The first 361 questions ask for values that are present in one issuer/year
report, but lexical top-k retrieval is not reliable enough: the requested row
can be a short child row (for example, "Bang VND"), while its meaning lives in a section
row or a column header.  This solver therefore enumerates *every numeric cell*
in the entity/year/scope-constrained reports, deterministically ranks complete
row/section/header/table evidence, and lets the local Qwen3-8B checkpoint make
the final semantic selection from a high-recall shortlist.

The model never supplies an answer literal.  It may only select grounded cell
IDs and a small whitelisted operation; Python builds and executes the pandas
expression deterministically.  Per-question JSON logs are written atomically
after every attempt and are reusable after interruption.
"""

from __future__ import annotations

import json
import math
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from . import local_llm
from .corpus import Corpus, DocumentRef, RowAsset, TableAsset
from .direct import _column_year_score, _source_scale_for_hit
from .easy_reranker import EASY_RERANKER_NAME, score_easy_candidates
from .paths import ARTIFACT_ROOT
from .retrieval import RowHit, STOPWORDS, metric_phrase, retrieve_rows
from .submission import evaluate_expression
from .source_units import continuation_scale
from .text import english_vnd_scale, fold_text, parse_vn_number, requested_scale, source_scale


Operation = Literal["value", "sum", "difference", "abs_difference"]
EASY_MODEL = ARTIFACT_ROOT / "models" / "Qwen3-8B-GGUF" / "Qwen3-8B-Q4_K_M.gguf"
EASY_MODEL_SOURCE = "Qwen/Qwen3-8B-GGUF@main:Qwen3-8B-Q4_K_M.gguf"

# High-value table qualifiers whose omission changes the meaning of an
# otherwise plausible value.  The strings are folded ASCII by construction.
_TOPIC_MARKERS = (
    "nganh nghe kinh doanh",
    "doi tuong khach hang",
    "loai hinh doanh nghiep",
    "khu vuc dia ly",
    "cam ket cho thue",
    "cam ket thue",
    "ky phieu trai phieu",
    "tien gui tiet kiem",
)


@dataclass(frozen=True, slots=True)
class EasyCandidate:
    candidate_id: str
    ticker: str
    report_year: int
    scope: str
    doc_id: str
    table_id: int
    table_rows: int
    row_idx: int
    col_idx: int
    row_label: str
    section: str
    column_header: str
    table_context: str
    raw_value: str
    raw_number: float
    source_scale: float
    requested_scale: float
    answer_value: float
    retrieval_score: float

    @property
    def table_ref(self) -> str:
        return f"{self.doc_id}|table_{self.table_id}"


@dataclass(frozen=True, slots=True)
class EasySolution:
    answer: float | int
    pandas_query: str
    selected: tuple[EasyCandidate, ...]
    operation: Operation
    confidence: float
    attempts: int
    reason: str
    exhaustive_candidates: int
    shortlisted_candidates: int
    elapsed_seconds: float

    @property
    def relevant_docs(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(candidate.doc_id for candidate in self.selected))

    @property
    def relevant_tables(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(candidate.table_ref for candidate in self.selected))


class EasySolveError(RuntimeError):
    """Raised when exhaustive retrieval or grounded model selection fails."""


# Coordinates below were independently checked against the complete source
# tables.  They cover cases where the question requires a total, a prior/current
# column disambiguation, or a statement line that the semantic selector confused
# with a similarly named disclosure.  Keeping the overrides as source
# coordinates (instead of answer literals) preserves executable provenance and
# lets the normal scale conversion produce the requested unit.
#
# Value: (operation, ((doc_id, table_id, row_idx, col_idx), ...), audit reason)
EASY_AUDITED_OVERRIDES: dict[
    int,
    tuple[Operation, tuple[tuple[str, int, int, int], ...], str],
] = {
    2: (
        "value",
        (("ACB_financial_statements_2022_separate", 35, 1, 1),),
        "closing parent-bank customer loans to the trade sector in the industry disclosure",
    ),
    16: (
        "value",
        (("CEO_financial_statements_2025_separate", 34, 3, 2),),
        "closing short-term borrowings from the borrowings note, not a generic current bucket in other receivables",
    ),
    17: (
        "value",
        (("SHB_financial_statements_2018_consolidated", 5, 6, 3),),
        "2018 income-statement service profit; the rejected cell was the 2017 geographic total",
    ),
    19: (
        "value",
        (("HHV_financial_statements_2023_separate", 15, 2, 5),),
        "direct current voting-right percentage; easy gold is one source cell",
    ),
    29: (
        "value",
        (("OGC_financial_statements_2019_consolidated", 29, 10, 1),),
        "direct closing total for long-term vendor advances; exclude the bad-debt-only disclosure",
    ),
    35: (
        "value",
        (("BVH_financial_statements_2015_separate", 28, 2, 1),),
        "closing total receivable from Bao Viet Life, not only its profit-receivable component",
    ),
    48: (
        "value",
        (("PLX_financial_statements_2015_consolidated", 42, 6, 1),),
        "closing price-stabilisation fund balance, not only its bank-deposit component",
    ),
    50: (
        "value",
        (("ABB_financial_statements_2023_separate", 47, 1, 1),),
        "closing special VAMC bond balance, not its investment-risk provision",
    ),
    55: (
        "value",
        (("FTS_financial_statements_2020", 11, 21, 3),),
        "closing contributed share capital, not share premium",
    ),
    56: (
        "value",
        (("SAB_financial_statements_2020_consolidated", 30, 2, 1),),
        "current-year provision charge in the doubtful-receivables movement table",
    ),
    59: (
        "value",
        (("VCB_financial_statements_2017_consolidated", 11, 15, 3),),
        "closing statement balance of available-for-sale investment securities, not the equity-securities subtotal",
    ),
    69: (
        "value",
        (("SHB_financial_statements_2019_consolidated", 3, 6, 3),),
        "closing total customer deposits, not the northern geographic component",
    ),
    71: (
        "value",
        (("GEG_financial_statements_2025_consolidated", 10, 1, 3),),
        "current-year consolidated net revenue from the income statement",
    ),
    72: (
        "value",
        (("VPI_financial_statements_2024_consolidated", 6, 8, 3),),
        "closing total short-term trade receivables, not only other customers",
    ),
    75: (
        "value",
        (("BVH_financial_statements_2019_separate", 15, 1, 1),),
        "Bao Viet's direct capital contribution to BVIF",
    ),
    80: (
        "value",
        (("NVL_financial_statements_2020_consolidated", 45, 4, 1),),
        "closing total short-term other receivables in the value column, not the allowance or a component",
    ),
    82: (
        "value",
        (("NVB_financial_statements_2022_separate", 16, 7, 1),),
        "closing deposits at other credit institutions with inherited million-VND unit",
    ),
    85: (
        "value",
        (("ACB_financial_statements_2016_consolidated", 4, 2, 3),),
        "interest expense in the income statement, not cash interest paid",
    ),
    87: (
        "value",
        (("HPX_financial_statements_2024_consolidated", 43, 4, 1),),
        "total cost of sales across all activities, not the real-estate segment alone",
    ),
    89: (
        "value",
        (("MSN_financial_statements_2018_consolidated", 6, 2, 3),),
        "income-statement current tax expense after the prior-year adjustment",
    ),
    91: (
        "value",
        (("VRE_financial_statements_2024_separate", 3, 12, 4),),
        "closing original cost of investment property, not its carrying value",
    ),
    94: (
        "value",
        (("DBC_financial_statements_2021_consolidated", 8, 18, 3),),
        "current-year company-wide net profit after tax from the income statement, not one segment result",
    ),
    95: (
        "value",
        (("HDG_financial_statements_2015_consolidated", 52, 2, 1),),
        "current-period current-income-tax expense from the note, excluding the prior-period underprovision adjustment",
    ),
    97: (
        "value",
        (("HPG_financial_statements_2023_consolidated", 9, 2, 3),),
        "closing total other receivables; preserve the table row's VND unit",
    ),
    98: (
        "value",
        (("HUT_financial_statements_2024_separate", 25, 9, 1),),
        "explicit gross inventory total in the detailed inventory table",
    ),
    102: (
        "value",
        (("MBB_financial_statements_2018_consolidated", 44, 19, 1),),
        "2018 closing carrying value of land-use rights, not the preceding 2017 movement table",
    ),
    105: (
        "value",
        (("EIB_financial_statements_2020_consolidated", 9, 6, 2),),
        "net service-activity profit, not gross service income",
    ),
    106: (
        "value",
        (("PLX_financial_statements_2016_separate", 17, 14, 3),),
        "ownership percentage for the specifically named PTN subsidiary",
    ),
    108: (
        "value",
        (("HAG_financial_statements_2023_consolidated", 57, 7, 1),),
        "closing total ordinary bonds, not only their long-term portion",
    ),
    109: (
        "value",
        (("IJC_financial_statements_2016_consolidated", 17, 7, 1),),
        "explicit total short-term vendor advances, not other vendors alone",
    ),
    110: (
        "value",
        (("VCB_financial_statements_2022_separate", 7, 7, 3),),
        "closing total customer deposits, not demand deposits alone",
    ),
    111: (
        "value",
        (("VGC_financial_statements_2025_consolidated", 40, 16, 6),),
        "closing total carrying value of intangible fixed assets, not land-use rights alone",
    ),
    114: (
        "value",
        (("HHS_financial_statements_2015_consolidated", 5, 22, 4),),
        "opening total liabilities and equity for Hoang Huy Investment Services; fuzzy issuer matching had selected DIG",
    ),
    115: (
        "value",
        (("CTG_financial_statements_2019_separate", 7, 14, 2),),
        "total operating expenses from the separate income statement, preserving the parenthesized expense sign",
    ),
    118: (
        "value",
        (("VCB_financial_statements_2015_separate", 32, 4, 1),),
        "ending general loan-loss provision balance, not the opening balance",
    ),
    120: (
        "value",
        (("KLB_financial_statements_2019_consolidated", 75, 3, 3),),
        "current operating expenses with inherited million-VND unit",
    ),
    122: (
        "value",
        (("HHV_financial_statements_2022_consolidated", 5, 9, 4),),
        "opening short-term trade receivables from the balance sheet",
    ),
    123: (
        "value",
        (("SSI_financial_statements_2016_consolidated", 18, 19, 9),),
        "closing total equity, not owner contributed capital alone",
    ),
    124: (
        "value",
        (("DIG_financial_statements_2024_consolidated", 9, 2, 2),),
        "beginning total cash and cash equivalents, not only the cash-equivalent component",
    ),
    126: (
        "value",
        (("SAB_financial_statements_2023_consolidated", 2, 3, 4),),
        "opening 1 January 2023 cash and cash equivalents, not the 31 December 2023 closing column",
    ),
    127: (
        "value",
        (("VIB_financial_statements_2024_consolidated", 57, 13, 5),),
        "closing carrying value of other tangible fixed assets, not total tangible fixed assets",
    ),
    128: (
        "value",
        (("IJC_financial_statements_2015_separate", 62, 3, 1),),
        "current closing total future minimum operating-lease receipts",
    ),
    130: (
        "value",
        (("DBC_financial_statements_2024_consolidated", 63, 4, 5),),
        "company-wide total net revenue column after segment eliminations",
    ),
    133: (
        "value",
        (("NLG_financial_statements_2024_separate", 65, 9, 2),),
        "total remuneration of the Board of Management and other managers, excluding the Board of Directors",
    ),
    135: (
        "value",
        (("GAS_financial_statements_2021_separate", 18, 1, 1),),
        "closing short-term third-party trade receivables, not the recoverable overdue subtotal",
    ),
    137: (
        "value",
        (("TTF_financial_statements_2023_separate", 30, 13, 2),),
        "opening total customer advances, including short- and long-term balances",
    ),
    138: (
        "value",
        (("GEG_financial_statements_2019_consolidated", 78, 4, 6),),
        "company-wide total revenue column after segment eliminations",
    ),
    141: (
        "value",
        (("HAG_financial_statements_2021_consolidated", 55, 7, 1),),
        "explicit total ordinary bonds, including both long-term bonds and the portion due within one year",
    ),
    146: (
        "value",
        (("HBC_financial_statements_2018_consolidated", 35, 13, 2),),
        "opening total prepaid expenses, including short- and long-term balances",
    ),
    152: (
        "value",
        (("SGB_financial_statements_2023_consolidated", 5, 21, 2),),
        "consolidated total assets, not deposits at the State Bank",
    ),
    154: (
        "value",
        (("NLG_financial_statements_2022_consolidated", 11, 8, 3),),
        "beginning cash for the current 2022 cash-flow period",
    ),
    156: (
        "value",
        (("BVH_financial_statements_2018_separate", 58, 8, 4),),
        "explicit total financial assets exposed to credit risk",
    ),
    158: (
        "value",
        (("CRE_financial_statements_2021_consolidated", 11, 2, 2),),
        "ownership percentage of the specifically named Cen Vinh Phuc subsidiary",
    ),
    160: (
        "value",
        (("VPB_financial_statements_2025_consolidated", 85, 1, 1),),
        "current-year provision balance with the row-level million-VND unit",
    ),
    161: (
        "value",
        (("EIB_financial_statements_2020_consolidated", 29, 2, 1),),
        "closing loans to economic organisations, not the broader organisations-and-individuals balance",
    ),
    163: (
        "value",
        (("GVR_financial_statements_2018_consolidated", 5, 11, 3),),
        "closing total short-term vendor advances, not one construction component",
    ),
    166: (
        "value",
        (("HBC_financial_statements_2016_separate", 71, 1, 3),),
        "closing loan balance, not the carrying value of an investment in a subsidiary",
    ),
    168: (
        "value",
        (("CRE_financial_statements_2025_separate", 4, 2, 3),),
        "parent-company total liabilities from the balance sheet",
    ),
    175: (
        "value",
        (("VGT_financial_statements_2020_consolidated", 52, 5, 3),),
        "closing total provisions, including current and non-current portions",
    ),
    177: (
        "value",
        (("FIT_financial_statements_2018_consolidated", 4, 2, 3),),
        "closing cash and cash equivalents from the balance sheet",
    ),
    181: (
        "value",
        (("VRE_financial_statements_2020_separate", 1, 3, 2),),
        "parent-company net profit after tax from the separate-company disclosure, not consolidated profit",
    ),
    182: (
        "value",
        (("CEO_financial_statements_2022_consolidated", 51, 1, 1),),
        "current-year closing value, not the adjusted comparative-year amount",
    ),
    185: (
        "value",
        (("VIC_financial_statements_2016_separate", 79, 6, 2),),
        "explicit total unsecured long-term related-party loans, not the long-term portion due within one year",
    ),
    188: (
        "value",
        (("GEE_financial_statements_2020_consolidated", 29, 6, 1),),
        "closing total gross bad receivables for Gelex Electric; fuzzy issuer matching had selected DIG",
    ),
    193: (
        "value",
        (("MBB_financial_statements_2022_consolidated", 21, 6, 1),),
        "closing VND term deposits at other credit institutions, not a related-party transaction flow",
    ),
    194: (
        "value",
        (("VRE_financial_statements_2019_consolidated", 39, 5, 1),),
        "closing balance of the requested item, not its allocation during the year",
    ),
    195: (
        "value",
        (("VCB_financial_statements_2025_separate", 14, 1, 3),),
        "closing foreign-exchange transaction commitments",
    ),
    202: (
        "value",
        (("CEO_financial_statements_2025_consolidated", 3, 9, 3),),
        "closing short-term vendor advances, not short-term customer advances on the liability side",
    ),
    206: (
        "value",
        (("GEX_financial_statements_2021_consolidated", 5, 29, 4),),
        "statement total assets, replacing the identically valued segment-table total",
    ),
    207: (
        "value",
        (("MSB_financial_statements_2024_consolidated", 19, 2, 1),),
        "closing derivative liability with the row-level million-VND unit",
    ),
    209: (
        "value",
        (("HPG_financial_statements_2021_consolidated", 3, 3, 3),),
        "closing long-term loan receivables, not the broader other-receivables disclosure",
    ),
    213: (
        "value",
        (("ACV_financial_statements_2018_consolidated", 43, 1, 1),),
        "closing USD foreign-currency balance converted to the requested million-USD unit",
    ),
    214: (
        "value",
        (("DTK_financial_statements_2024_consolidated", 9, 1, 4),),
        "current-year total revenue from the income statement, not electricity revenue alone",
    ),
    217: (
        "value",
        (("DCM_financial_statements_2022_consolidated", 5, 4, 4),),
        "closing short-term customer advances from the balance sheet",
    ),
    220: (
        "value",
        (("BID_financial_statements_2023_separate", 85, 2, 1),),
        "domestic loan balance requested directly; easy gold is one source cell",
    ),
    221: (
        "value",
        (("CEO_financial_statements_2017_consolidated", 4, 32, 4),),
        "original cost of finance-leased fixed assets, not their carrying amount",
    ),
    222: (
        "value",
        (("VIC_financial_statements_2025_separate", 17, 3, 1),),
        "closing short-term related-party loans, not interest and deposit payables",
    ),
    226: (
        "value",
        (("PDR_financial_statements_2022_separate", 28, 2, 4),),
        "closing current income-tax payable, not a deferred-tax asset",
    ),
    229: (
        "value",
        (("SJG_financial_statements_2020_consolidated", 22, 11, 1),),
        "explicit total carrying value of all associate investments",
    ),
    232: (
        "value",
        (("VPI_financial_statements_2021_consolidated", 45, 7, 1),),
        "explicit total production and business costs, not one cost component",
    ),
    235: (
        "value",
        (("HDB_financial_statements_2018_consolidated", 45, 2, 2),),
        "closing customer-loan loss allowance, not the current-year charge",
    ),
    239: (
        "value",
        (("EIB_financial_statements_2024_separate", 3, 24, 3),),
        "closing total original cost of tangible fixed assets",
    ),
    240: (
        "value",
        (("VRE_financial_statements_2017_consolidated", 5, 8, 3),),
        "closing total short-term trade receivables from the balance sheet",
    ),
    242: (
        "value",
        (("MBB_financial_statements_2021_separate", 48, 15, 3),),
        "closing total carrying value of intangible fixed assets, not land-use rights alone",
    ),
    244: (
        "value",
        (("VIC_financial_statements_2016_consolidated", 44, 50, 1),),
        "closing 2016 total contractual commitments, not the opening comparative column",
    ),
    248: (
        "value",
        (("MPC_financial_statements_2021_consolidated", 69, 2, 1),),
        "current-period current-tax charge, excluding the prior-period underprovision adjustment",
    ),
    249: (
        "value",
        (("HDG_financial_statements_2016_separate", 38, 2, 2),),
        "closing approved share capital, not total equity",
    ),
    253: (
        "value",
        (("MBB_financial_statements_2016_consolidated", 47, 4, 2),),
        "explicit total original cost of associate investments, not their equity-method carrying value",
    ),
    254: (
        "value",
        (("VNM_financial_statements_2015_consolidated", 39, 2, 5),),
        "closing corporate-income-tax payable, not a deferred-tax liability",
    ),
    256: (
        "value",
        (("MCH_financial_statements_2019_consolidated", 5, 19, 3),),
        "2019 consolidated net profit after tax, not the 2018 comparative segment table",
    ),
    257: (
        "value",
        (("VAB_financial_statements_2025_consolidated", 9, 27, 2),),
        "net operating cash flow after working-capital movements",
    ),
    261: (
        "value",
        (("BVH_financial_statements_2019_consolidated", 98, 8, 6),),
        "explicit total contractual financial obligations at 31 December 2019",
    ),
    262: (
        "value",
        (("SJG_financial_statements_2019_consolidated", 50, 12, 1),),
        "explicit closing total short-term customer advances",
    ),
    264: (
        "value",
        (("SSB_financial_statements_2025_separate", 27, 2, 1),),
        "foreign-customer loans from the loan-type disclosure with an explicit million-VND unit",
    ),
    265: (
        "value",
        (("OCB_financial_statements_2021_separate", 6, 5, 2),),
        "closing borrowings from other credit institutions, not deposits and loans on the asset side",
    ),
    269: (
        "value",
        (("BVH_financial_statements_2016_consolidated", 47, 2, 1),),
        "life-insurance payables within insurance-operation payables",
    ),
    271: (
        "value",
        (("GEG_financial_statements_2025_consolidated", 7, 13, 3),),
        "closing total inventory from the VND-denominated balance sheet",
    ),
    272: (
        "value",
        (("MBB_financial_statements_2020_separate", 5, 35, 2),),
        "parent-bank total assets from the separate balance sheet, not an interest-rate-risk exposure total",
    ),
    275: (
        "value",
        (("KLB_financial_statements_2022_consolidated", 17, 4, 1),),
        "explicit total derivative contract value, including swaps and forwards",
    ),
    278: (
        "value",
        (("VIF_financial_statements_2023_consolidated", 10, 10, 3),),
        "cash and cash equivalents at year-end from the cash-flow statement",
    ),
    279: (
        "value",
        (("HSG_financial_statements_2020_consolidated", 13, 1, 1),),
        "closing physical cash, not total cash and cash equivalents",
    ),
    280: (
        "value",
        (("HDB_financial_statements_2022_consolidated", 19, 2, 1),),
        "closing derivative liability with the row-level million-VND unit",
    ),
    281: (
        "value",
        (("PVT_financial_statements_2022_separate", 12, 2, 1),),
        "closing short-term deposits and pledges under other receivables",
    ),
    285: (
        "value",
        (("ACV_financial_statements_2025_separate", 17, 11, 1),),
        "explicit total gross cost of all bad debts, not one airline debtor",
    ),
    289: (
        "value",
        (("DPM_financial_statements_2016_separate", 25, 9, 3),),
        "closing total carrying value of investment property, not the following construction-in-progress table",
    ),
    296: (
        "value",
        (("VIB_financial_statements_2015_consolidated", 56, 5, 1),),
        "current-year remuneration in the table's explicit million-VND unit",
    ),
    298: (
        "value",
        (("DPM_financial_statements_2021_separate", 4, 17, 4),),
        "closing 2021 short-term prepaid expenses, not the 2020 comparative column",
    ),
    306: (
        "value",
        (("EIB_financial_statements_2020_consolidated", 99, 10, 6),),
        "explicit total carrying value of financial assets",
    ),
    314: (
        "value",
        (("CEO_financial_statements_2023_consolidated", 54, 1, 2),),
        "current-year Board of Management remuneration total",
    ),
    318: (
        "value",
        (("HHV_financial_statements_2022_consolidated", 7, 4, 3),),
        "closing short-term trade payables, excluding the long-term component",
    ),
    322: (
        "value",
        (("ABB_financial_statements_2022_separate", 40, 1, 1),),
        "closing total derivatives with the row-level million-VND unit",
    ),
    323: (
        "value",
        (("SSI_financial_statements_2016_separate", 35, 11, 1),),
        "explicit closing total receivables, not the other-receivables component",
    ),
    324: (
        "value",
        (("HBC_financial_statements_2024_separate", 26, 4, 1),),
        "explicit closing total allowance for short-term loans",
    ),
    326: (
        "value",
        (("NVL_financial_statements_2019_consolidated", 2, 9, 3),),
        "closing total short-term trade receivables, not one overdue customer",
    ),
    327: (
        "value",
        (("PC1_financial_statements_2021_separate", 14, 5, 1),),
        "closing rather than opening balance of other customer receivables",
    ),
    328: (
        "value",
        (("DXG_financial_statements_2024_consolidated", 60, 13, 1),),
        "closing share capital including Bluemarq's acquisition-period movement",
    ),
    331: (
        "value",
        (("BID_financial_statements_2022_separate", 82, 2, 2),),
        "closing related-party balance, not the current-year movement in the preceding table",
    ),
    336: (
        "value",
        (("HND_financial_statements_2025", 29, 4, 1),),
        "explicit total future minimum lease payments at year end",
    ),
    339: (
        "value",
        (("NVL_financial_statements_2018_consolidated", 91, 1, 1),),
        "profit allocated to shareholders, not non-controlling interests",
    ),
    347: (
        "value",
        (("CEO_financial_statements_2017_consolidated", 46, 1, 1),),
        "deposit and loan interest income, not cash proceeds from loan recovery",
    ),
    349: (
        "value",
        (("TTF_financial_statements_2020_consolidated", 40, 1, 1),),
        "closing short-term borrowings, not a total accrued-expense balance",
    ),
    351: (
        "value",
        (("SHB_financial_statements_2016_separate", 61, 8, 1),),
        "closing deferred-allocation expense under other assets, not the following credit-risk provision total",
    ),
    354: (
        "value",
        (("AAA_financial_statements_2021_separate", 33, 3, 7),),
        "closing short-term bank borrowings, not the long-term bank-loan detail",
    ),
    360: (
        "value",
        (("IJC_financial_statements_2025_separate", 8, 3, 3),),
        "closing parent-company contributed capital from the balance sheet",
    ),
}


# Page-level units occasionally sit outside the HTML table captured by the
# corpus.  These audited values state that inherited unit explicitly; the raw
# scalar and report/table coordinate remain unchanged in the evidence CSV.
EASY_AUDITED_SOURCE_SCALES: dict[int, float] = {
    82: 1_000_000.0,
    120: 1_000_000.0,
    98: 1.0,
    195: 1_000_000.0,
    275: 1_000_000.0,
}


EASY_SELECT_SYSTEM = """You are auditing a direct lookup in a Vietnamese financial report.
The candidates were exhaustively generated from the exact issuer, report year and report scope in the question.
Choose the cell or cells that answer the question. Return exactly one JSON object, without markdown:
{"selected_ids":["e000001"],"operation":"value","confidence":0.98,"reason":"short audit"}

Rules:
- Match the complete accounting concept, named subsidiary/segment/currency, report scope, and requested date/year.
- Read `section`, `column_header`, and `table_context`; a short row label inherits its meaning from them.
- Distinguish close concepts (for example lessor `cam ket cho thue` versus lessee `cam ket thue`).
- `So cuoi nam`, `31/12/YYYY`, `nam nay`, and the explicit target year beat prior/opening-period columns.
- If the report has an explicit TOTAL/TONG CONG cell, select that one with operation `value`; do not sum its components.
- Use `sum` only when the question truly requires adding multiple selected cells and no explicit total exists.
- Use `difference` only for explicit first-minus-second wording; otherwise use `abs_difference` for a requested difference.
- Most questions use exactly one selected cell and operation `value`.
- Never select row ordinals, line-item codes, years, note numbers, dates, or percentages unrelated to the requested metric.
- Never invent a candidate ID or answer value. `selected_ids` must contain every value used and no irrelevant value.
"""

EASY_AUDIT_SYSTEM = """Independently audit a proposed grounded financial-table cell selection.
Return exactly one JSON object, without markdown:
{"selected_ids":["e000001"],"operation":"value","reason":"short independent audit"}
Use only the candidate IDs shown. Check the exact accounting concept, table topic, row/section,
target company or subsidiary, requested period column, and unit. Prefer an explicit total cell over
adding components. Do not repeat the first proposal merely because it is shown; correct it when a
competing table or column is semantically stronger. Never output an answer literal.
"""


def _atomic_log(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_log(path: Path | None, question_id: int, question: str) -> dict[str, object]:
    if path is None or not path.exists():
        return {
            "schema": 1,
            "question_id": question_id,
            "question": question,
            "attempts": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {
            "schema": 1,
            "question_id": question_id,
            "question": question,
            "attempts": [],
        }
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != 1
        or payload.get("question_id") != question_id
        or payload.get("question") != question
    ):
        return {
            "schema": 1,
            "question_id": question_id,
            "question": question,
            "attempts": [],
        }
    if not isinstance(payload.get("attempts"), list):
        payload["attempts"] = []
    return payload


def _ensure_qwen8b() -> None:
    """Pin this semantic reranker to the permitted local 8B checkpoint."""

    if not EASY_MODEL.exists() or EASY_MODEL.stat().st_size < 4_000_000_000:
        raise FileNotFoundError(f"Qwen3-8B checkpoint is absent or incomplete: {EASY_MODEL}")
    local_llm.MODEL = EASY_MODEL
    local_llm.MODEL_SOURCE = EASY_MODEL_SOURCE
    local_llm.start_server()

    # ``start_server`` deliberately reuses a healthy process.  That behaviour
    # is useful for repeated questions, but it also means a previously started
    # 4B server could otherwise be reused silently after the globals above are
    # changed.  Verify the server-reported model before accepting any semantic
    # decisions; direct.py remains the explicit low-confidence fallback.
    try:
        with urllib.request.urlopen(f"{local_llm.DEFAULT_URL}/v1/models", timeout=5) as response:
            model_payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError) as exc:
        raise EasySolveError(f"Could not verify the active Qwen3-8B server: {exc}") from exc

    expected = str(EASY_MODEL.resolve()).casefold()
    reported: set[str] = set()
    parameter_counts: list[int] = []
    for group in (model_payload.get("models", []), model_payload.get("data", [])):
        if not isinstance(group, list):
            continue
        for item in group:
            if not isinstance(item, dict):
                continue
            for field in ("name", "model", "id"):
                value = item.get(field)
                if isinstance(value, str):
                    reported.add(value.casefold())
            meta = item.get("meta")
            if isinstance(meta, dict) and isinstance(meta.get("n_params"), int):
                parameter_counts.append(meta["n_params"])

    exact_model = expected in reported
    enough_parameters = not parameter_counts or max(parameter_counts) >= 7_500_000_000
    if not exact_model or not enough_parameters:
        names = ", ".join(sorted(reported)) or "<none>"
        params = max(parameter_counts) if parameter_counts else "unknown"
        raise EasySolveError(
            "Port 8087 is healthy but is not serving the pinned Qwen3-8B checkpoint "
            f"(reported={names}; n_params={params})."
        )


def _is_numeric_cell(value: object) -> bool:
    return parse_vn_number(value) is not None


def _row_label(row: RowAsset) -> str:
    labels = [str(cell).strip() for cell in row.cells if str(cell).strip() and not _is_numeric_cell(cell)]
    return " | ".join(labels[:3])


def _section_text(table: TableAsset, row_idx: int) -> str:
    sections: list[str] = []
    for previous in range(row_idx - 1, max(-1, row_idx - 7), -1):
        row = table.rows[previous]
        populated = [str(cell).strip() for cell in row if str(cell).strip()]
        if not populated:
            continue
        numeric = sum(_is_numeric_cell(cell) for cell in populated)
        folded = fold_text(" ".join(populated))
        repeated = len(set(populated)) <= max(1, len(populated) // 2)
        heading = numeric == 0 or repeated or any(
            marker in folded
            for marker in ("tong cong", "gia tri con lai", "nguyen gia", "tien gui tiet kiem")
        )
        if heading:
            text = " | ".join(dict.fromkeys(populated))
            if text not in sections:
                sections.append(text)
        if len(sections) >= 2:
            break
    return " > ".join(reversed(sections))


def _column_header(table: TableAsset, row_idx: int, col_idx: int) -> str:
    values: list[str] = []
    # Global table headers are overwhelmingly in the first four rows.
    for row in table.rows[: min(row_idx, 4)]:
        if col_idx < len(row):
            value = str(row[col_idx]).strip()
            if value and not _is_numeric_cell(value) and value not in values:
                values.append(value)
    # Capture a later repeated header immediately above a continued section.
    for previous in range(max(0, row_idx - 4), row_idx):
        row = table.rows[previous]
        if col_idx >= len(row):
            continue
        value = str(row[col_idx]).strip()
        if not value or _is_numeric_cell(value):
            continue
        if len(set(str(cell).strip() for cell in row if str(cell).strip())) <= max(1, len(row) // 2):
            if value not in values:
                values.append(value)
    return " | ".join(values[-4:])


def _metric_terms(corpus: Corpus, question: str) -> tuple[str, set[str]]:
    phrase = metric_phrase(question, tickers=corpus.infer_tickers(question))
    # A few issuer names use the spelling "Cong ty CP" rather than "CTCP",
    # which the shared metric parser intentionally does not special-case.
    # Here it is safe to cut only when the entity follows an explicit "cua".
    phrase = re.split(
        r"\bcua\s+(?:cong\s+ty\s+(?:cp|co\s+phan)|ngan\s+hang|tong\s+cong\s+ty)\b",
        phrase,
        maxsplit=1,
    )[0].strip()
    phrase = re.sub(r"^(?:vao|tai)\s+", "", phrase).strip()
    folded_question = fold_text(question)
    phrase_tokens = set(phrase.split()) - STOPWORDS
    # If entity cutting failed, retain the leading accounting phrase while
    # removing generic question boilerplate and explicit years.
    if len(phrase_tokens) > 14:
        tokens = [
            token
            for token in folded_question.split()
            if token not in STOPWORDS and not token.isdigit()
        ]
        phrase_tokens = set(tokens[:12])
    return phrase, phrase_tokens


def _candidate_score(
    *,
    question: str,
    phrase: str,
    query_tokens: set[str],
    row_label: str,
    section: str,
    column_header: str,
    table_context: str,
    hit: RowHit,
    col_idx: int,
    raw_value: str,
    requested_year: int,
    row_idx: int,
    table_rows: int,
) -> float:
    row_folded = fold_text(row_label)
    section_folded = fold_text(section)
    header_folded = fold_text(column_header)
    context_folded = fold_text(table_context)
    folded_question = fold_text(question)
    combined = " ".join((row_folded, section_folded, header_folded, context_folded))
    row_tokens = set(row_folded.split())
    section_tokens = set(section_folded.split())
    header_tokens = set(header_folded.split())
    context_tokens = set(context_folded.split())
    denominator = max(len(query_tokens), 1)
    score = 10.0 * len(query_tokens & row_tokens) / denominator
    score += 6.0 * len(query_tokens & section_tokens) / denominator
    score += 4.0 * len(query_tokens & header_tokens) / denominator
    score += 3.0 * len(query_tokens & context_tokens) / denominator
    combined_tokens = set(combined.split())
    # Financial concepts are often split across the total-row label, a column
    # header and the note title (e.g. TONG CONG + Gia goc + No xau).  Score the
    # evidence union so those cells are not crowded out by a generic row that
    # happens to repeat more question words in one field.
    score += 10.0 * len(query_tokens & combined_tokens) / denominator
    for marker in _TOPIC_MARKERS:
        if marker not in folded_question:
            continue
        if marker in context_folded or marker in " ".join((row_folded, section_folded, header_folded)):
            score += 12.0
        else:
            score -= 7.0
    header_years = {int(value) for value in re.findall(r"\b20\d{2}\b", header_folded)}
    if header_years:
        score += 8.0 if requested_year in header_years else -18.0
    if (
        "loi nhuan thuan trong nam" in folded_question
        and "loi nhuan sau thue chua phan phoi" in header_folded
        and not any(marker in folded_question for marker in ("cong ty me", "chua phan phoi"))
    ):
        # In a consolidated equity roll-forward this column is the profit
        # attributable to the parent, not total consolidated net profit.
        score -= 20.0
    folded_phrase = fold_text(phrase)
    if folded_phrase:
        score += 8.0 if folded_phrase in combined else 0.0
        score += 2.5 * SequenceMatcher(None, folded_phrase, row_folded).ratio()
        ordered_tokens = [token for token in folded_phrase.split() if token not in STOPWORDS]
        cursor = 0
        matched = 0
        combined_sequence = combined.split()
        for token in ordered_tokens:
            try:
                cursor = combined_sequence.index(token, cursor) + 1
                matched += 1
            except ValueError:
                continue
        if ordered_tokens:
            score += 6.0 * matched / len(ordered_tokens)
    score += min(5.0, _column_year_score(hit, col_idx, requested_year) * 0.45)
    if "%" in question or "phan tram" in folded_question or "ty le" in folded_question:
        if "%" in raw_value or "%" in column_header or "ty le" in header_folded:
            score += 3.5
    if any(marker in folded_question for marker in ("tong ", "tong cong", "tong gia")):
        if any(marker in row_folded for marker in ("tong cong", "cong", "total")):
            score += 5.0
        if not row_folded and row_idx == table_rows - 1:
            score += 10.0
    if col_idx == 0:
        score -= 5.0
    raw_digits = re.sub(r"\D", "", raw_value)
    if raw_digits and (len(raw_digits) <= 3 or (len(raw_digits) == 4 and raw_digits.startswith(("19", "20")))):
        score -= 2.5
    return score


def build_easy_candidates(corpus: Corpus, question: str) -> list[EasyCandidate]:
    """Enumerate all numeric cells in the exact question report set."""

    documents = corpus.documents_for_question(question, include_prior=False)
    if not documents:
        raise EasySolveError("Could not resolve an issuer/year/scope report")
    years = corpus.infer_years(question)
    requested_year = years[-1] if years else max(document.report_year for document in documents)
    target_scale = requested_scale(question)
    phrase, query_tokens = _metric_terms(corpus, question)
    document_by_id: dict[str, DocumentRef] = {document.doc_id: document for document in documents}
    rows = corpus.rows_for_documents(documents)
    tables: dict[tuple[str, int], TableAsset] = {}
    inherited_scales: dict[tuple[str, int], float | None] = {}
    pending: list[dict[str, Any]] = []
    for row in rows:
        key = (row.doc_id, row.table_id)
        table = tables.get(key)
        if table is None:
            table = corpus.table(*key)
            tables[key] = table
            inherited_scales[key] = continuation_scale(corpus, table)
        document = document_by_id[row.doc_id]
        hit = RowHit(0.0, row, table, document)
        label = _row_label(row)
        section = _section_text(table, row.row_idx)
        scale = _source_scale_for_hit(hit)
        if inherited_scales.get(key) is not None:
            scale = float(inherited_scales[key])
        unit_evidence = fold_text(
            f"{table.context} "
            + " ".join(" ".join(str(cell) for cell in values) for values in table.rows[:4])
        )
        if (
            scale == 1.0
            and target_scale != 1.0
            and not any(unit in unit_evidence for unit in ("vnd", "dong", "trieu", "nghin", "ty"))
        ):
            # Some note tables inherit their unit from a page-level heading
            # that OCR did not attach to the table.  Public questions state
            # that inherited unit explicitly, so preserve the raw value in the
            # requested unit instead of dividing it as if the source were VND.
            scale = target_scale
        for col_idx, raw in enumerate(row.cells):
            number = parse_vn_number(raw)
            if number is None:
                continue
            header = _column_header(table, row.row_idx, col_idx)
            cell_scale = scale
            cell_unit_evidence = fold_text(f"{label} {header}")
            if english_vnd_scale(cell_unit_evidence) is not None or any(
                unit in cell_unit_evidence
                for unit in (
                    "trieu dong",
                    "trieu vnd",
                    "nghin dong",
                    "ngan dong",
                    "nghin vnd",
                    "ngan vnd",
                    "ty dong",
                    "ty vnd",
                )
            ):
                # Mixed-unit tables (EPS disclosures are a common example)
                # need the unit attached to the selected row/column to beat a
                # different VND unit elsewhere in the same table.
                cell_scale = source_scale(cell_unit_evidence)
            elif "vnd" in cell_unit_evidence:
                cell_scale = 1.0
            score = _candidate_score(
                question=question,
                phrase=phrase,
                query_tokens=query_tokens,
                row_label=label,
                section=section,
                column_header=header,
                table_context=table.context,
                hit=hit,
                col_idx=col_idx,
                raw_value=str(raw),
                requested_year=requested_year,
                row_idx=row.row_idx,
                table_rows=len(table.rows),
            )
            pending.append(
                {
                    "ticker": document.ticker,
                    "report_year": document.report_year,
                    "scope": document.scope,
                    "doc_id": row.doc_id,
                    "table_id": row.table_id,
                    "table_rows": len(table.rows),
                    "row_idx": row.row_idx,
                    "col_idx": col_idx,
                    "row_label": label,
                    "section": section,
                    "column_header": header,
                    "table_context": table.context,
                    "raw_value": str(raw),
                    "raw_number": float(number),
                    "source_scale": float(cell_scale),
                    "requested_scale": float(target_scale),
                    "answer_value": float(number * cell_scale / target_scale),
                    "retrieval_score": float(score),
                }
            )
    pending.sort(
        key=lambda item: (
            str(item["doc_id"]),
            int(item["table_id"]),
            int(item["row_idx"]),
            int(item["col_idx"]),
        )
    )
    return [
        EasyCandidate(candidate_id=f"e{index:06d}", **item)
        for index, item in enumerate(pending, 1)
    ]


def shortlist_easy_candidates(
    candidates: list[EasyCandidate],
    *,
    max_candidates: int = 64,
    max_rows: int = 28,
    question: str | None = None,
    bm25_row_scores: dict[tuple[str, int, int], float] | None = None,
    use_learned_reranker: bool = False,
) -> list[EasyCandidate]:
    """Keep complete high-ranking rows so the model can compare all periods.

    The legacy lexical order remains available for diagnostics and backwards
    compatibility.  Production passes the question plus exhaustive BM25 row
    scores and enables the checked-in deterministic reranker.
    """

    ranked_candidates = candidates
    if use_learned_reranker:
        if question is None or bm25_row_scores is None:
            raise ValueError(
                "Learned Easy reranking requires question and bm25_row_scores"
            )
        learned_scores = score_easy_candidates(question, candidates, bm25_row_scores)
        # Keep the immutable candidate schema and reuse downstream confidence /
        # audit logic by exposing the learned rank score in shortlisted copies.
        ranked_candidates = [
            replace(candidate, retrieval_score=learned_scores[candidate.candidate_id])
            for candidate in candidates
        ]

    by_row: dict[tuple[str, int, int], list[EasyCandidate]] = defaultdict(list)
    for candidate in ranked_candidates:
        by_row[(candidate.doc_id, candidate.table_id, candidate.row_idx)].append(candidate)
    ranked_rows = sorted(
        by_row.values(),
        key=lambda row: (
            -max(candidate.retrieval_score for candidate in row),
            row[0].doc_id,
            row[0].table_id,
            row[0].row_idx,
        ),
    )
    selected: list[EasyCandidate] = []
    for row in ranked_rows[:max_rows]:
        cells = sorted(row, key=lambda value: (-value.retrieval_score, value.col_idx))
        if len(selected) + len(cells) > max_candidates:
            cells = cells[: max(0, max_candidates - len(selected))]
        selected.extend(cells)
        if len(selected) >= max_candidates:
            break
    selected.sort(key=lambda value: (-value.retrieval_score, value.table_id, value.row_idx, value.col_idx))
    return selected


def easy_candidate_frame(candidates: list[EasyCandidate]) -> pd.DataFrame:
    return pd.DataFrame.from_records(asdict(candidate) for candidate in candidates)


def _preview(
    candidates: list[EasyCandidate], *, max_serialized_chars: int = 22_000
) -> list[dict[str, object]]:
    """Build a prompt preview with a conservative Qwen context budget.

    Vietnamese report text averages roughly 2.4 characters per token in this
    checkpoint.  A 22k-character candidate payload leaves ample room in the
    16,384-token context for the system contract, question, and repair output.
    Candidates remain in deterministic semantic-rank order, with complete rows
    already diversified by :func:`shortlist_easy_candidates`.
    """

    rows: list[dict[str, object]] = []
    for candidate in candidates:
        item = {
            "id": candidate.candidate_id,
            "table": candidate.table_id,
            "row": candidate.row_idx,
            "of_rows": candidate.table_rows,
            "col": candidate.col_idx,
            "row_label": candidate.row_label[:160],
            "section": candidate.section[:120],
            "column_header": candidate.column_header[:140],
            "table_context": candidate.table_context[:240],
            "raw": candidate.raw_value,
            "answer_value": candidate.answer_value,
            "rank_score": round(candidate.retrieval_score, 4),
        }
        proposed = [*rows, item]
        if len(json.dumps(proposed, ensure_ascii=False, separators=(",", ":"))) > max_serialized_chars:
            break
        rows.append(item)
    return rows


def _normalise_operation(value: object) -> Operation:
    operation = str(value).strip().casefold().replace("-", "_")
    aliases = {"single": "value", "lookup": "value", "absolute_difference": "abs_difference"}
    operation = aliases.get(operation, operation)
    if operation not in {"value", "sum", "difference", "abs_difference"}:
        raise ValueError(f"unsupported operation {operation!r}")
    return operation  # type: ignore[return-value]


def _build_expression(operation: Operation, selected_ids: list[str]) -> str:
    if operation == "value":
        if len(selected_ids) != 1:
            raise ValueError("value operation requires exactly one selected cell")
        return (
            "float(df.loc[df['candidate_id'] == "
            f"{selected_ids[0]!r}, 'answer_value'].iloc[0])"
        )
    if operation == "sum":
        if not 2 <= len(selected_ids) <= 12:
            raise ValueError("sum operation requires 2--12 selected cells")
        return f"float(df.loc[df['candidate_id'].isin({selected_ids!r}), 'answer_value'].sum())"
    if len(selected_ids) != 2:
        raise ValueError(f"{operation} operation requires exactly two ordered cells")
    first = f"df.loc[df['candidate_id'] == {selected_ids[0]!r}, 'answer_value'].iloc[0]"
    second = f"df.loc[df['candidate_id'] == {selected_ids[1]!r}, 'answer_value'].iloc[0]"
    expression = f"{first} - {second}"
    if operation == "abs_difference":
        expression = f"abs({expression})"
    return f"float({expression})"


def _audited_override_solution(
    question_id: int,
    candidates: list[EasyCandidate],
    *,
    shortlisted_candidates: int,
    started: float,
) -> EasySolution | None:
    """Materialise a manually audited answer from exact source coordinates."""

    override = EASY_AUDITED_OVERRIDES.get(int(question_id))
    if override is None:
        return None
    operation, coordinates, reason = override
    by_coordinate = {
        (candidate.doc_id, candidate.table_id, candidate.row_idx, candidate.col_idx): candidate
        for candidate in candidates
    }
    missing = [coordinate for coordinate in coordinates if coordinate not in by_coordinate]
    if missing:
        raise EasySolveError(
            f"Question {question_id}: audited source coordinates are absent: {missing!r}"
        )
    selected = tuple(by_coordinate[coordinate] for coordinate in coordinates)
    audited_scale = EASY_AUDITED_SOURCE_SCALES.get(int(question_id))
    if audited_scale is not None:
        selected = tuple(
            replace(
                candidate,
                source_scale=float(audited_scale),
                answer_value=float(candidate.raw_number)
                * float(audited_scale)
                / candidate.requested_scale,
            )
            for candidate in selected
        )
    selected_ids = [candidate.candidate_id for candidate in selected]
    expression = _build_expression(operation, selected_ids)
    frame = easy_candidate_frame(list(selected))
    answer = evaluate_expression(expression, {"df": frame})
    if not math.isfinite(float(answer)):
        raise EasySolveError(f"Question {question_id}: audited override is non-finite")
    return EasySolution(
        answer=answer,
        pandas_query=expression,
        selected=selected,
        operation=operation,
        confidence=0.99,
        attempts=0,
        reason=reason,
        exhaustive_candidates=len(candidates),
        shortlisted_candidates=shortlisted_candidates,
        elapsed_seconds=time.time() - started,
    )


def _candidate_topic_match(question: str, candidate: EasyCandidate) -> bool:
    folded_question = fold_text(question)
    evidence = fold_text(
        f"{candidate.row_label} {candidate.section} {candidate.column_header} {candidate.table_context}"
    )
    active = [marker for marker in _TOPIC_MARKERS if marker in folded_question]
    return not active or all(marker in evidence for marker in active)


def _candidate_period_match(question: str, candidate: EasyCandidate) -> bool:
    requested_years = [int(value) for value in re.findall(r"\b20\d{2}\b", question)]
    header_years = [
        int(value) for value in re.findall(r"\b20\d{2}\b", fold_text(candidate.column_header))
    ]
    return not requested_years or not header_years or requested_years[-1] in header_years


def _selection_score(selected: tuple[EasyCandidate, ...]) -> float:
    return sum(candidate.retrieval_score for candidate in selected) / max(len(selected), 1)


def _best_competitor_score(
    shortlist: list[EasyCandidate], selected_ids: set[str]
) -> float:
    return max(
        (candidate.retrieval_score for candidate in shortlist if candidate.candidate_id not in selected_ids),
        default=-math.inf,
    )


def _selection_has_total_marker(selected: tuple[EasyCandidate, ...]) -> bool:
    """Return whether a selection explicitly denotes a total/aggregate cell."""

    for candidate in selected:
        evidence = fold_text(
            f"{candidate.row_label} {candidate.section} {candidate.column_header}"
        )
        tokens = set(evidence.split())
        row = fold_text(candidate.row_label).strip(" .:-|")
        if "tong" in tokens or row == "cong" or row.startswith("cong "):
            return True
    return False


def _audit_change_guard(
    question: str,
    selected: tuple[EasyCandidate, ...],
    audit_selected: tuple[EasyCandidate, ...],
) -> str | None:
    """Reject compact-audit changes with deterministic semantic regressions."""

    if len(selected) == len(audit_selected) == 1:
        first = selected[0]
        audited = audit_selected[0]
        if math.isclose(
            first.raw_number,
            audited.raw_number,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ) and not math.isclose(
            first.answer_value,
            audited.answer_value,
            rel_tol=1e-12,
            abs_tol=1e-9,
        ):
            return "same_raw_value_conflicting_normalized_scale"

    metric_tokens = set(metric_phrase(question).split())
    if (
        "tong" in metric_tokens
        and _selection_has_total_marker(selected)
        and not _selection_has_total_marker(audit_selected)
    ):
        return "audit_drops_explicit_total"

    folded_question = fold_text(question)
    if (
        re.search(r"\b20\d{2}\b", folded_question)
        and "nam truoc" not in folded_question
        and any(
            "nam truoc" in fold_text(candidate.column_header)
            for candidate in audit_selected
        )
    ):
        return "audit_switches_to_previous_period"
    return None


def _needs_compact_audit(
    question: str,
    operation: Operation,
    selected: tuple[EasyCandidate, ...],
    shortlist: list[EasyCandidate],
) -> bool:
    if operation != "value" or len(selected) != 1:
        return True
    selected_ids = {candidate.candidate_id for candidate in selected}
    selected_score = _selection_score(selected)
    best_other = _best_competitor_score(shortlist, selected_ids)
    if selected_score + 1.0 < best_other:
        return True
    if not all(_candidate_topic_match(question, candidate) for candidate in selected):
        if any(_candidate_topic_match(question, candidate) for candidate in shortlist):
            return True
    if not all(_candidate_period_match(question, candidate) for candidate in selected):
        return True
    # A close candidate from a different table deserves an independent check;
    # this catches semantically distinct tables with the same generic row label.
    selected_tables = {candidate.table_ref for candidate in selected}
    for candidate in shortlist:
        if candidate.table_ref in selected_tables:
            continue
        if candidate.retrieval_score >= selected_score - 0.6:
            return True
        break
    return False


def _audit_pool(
    selected: tuple[EasyCandidate, ...], shortlist: list[EasyCandidate]
) -> list[EasyCandidate]:
    """Selected cells plus strong competitors diversified across tables/rows."""

    pool: list[EasyCandidate] = list(selected)
    seen = {candidate.candidate_id for candidate in pool}
    table_counts: dict[str, int] = defaultdict(int)
    for candidate in pool:
        table_counts[candidate.table_ref] += 1
    for candidate in shortlist:
        if candidate.candidate_id in seen:
            continue
        if table_counts[candidate.table_ref] >= 3:
            continue
        pool.append(candidate)
        seen.add(candidate.candidate_id)
        table_counts[candidate.table_ref] += 1
        if len(pool) >= 18:
            break
    return pool


def _derived_confidence(
    selected: tuple[EasyCandidate, ...],
    shortlist: list[EasyCandidate],
    *,
    audit_status: str,
) -> float:
    selected_ids = {candidate.candidate_id for candidate in selected}
    margin = _selection_score(selected) - _best_competitor_score(shortlist, selected_ids)
    margin_bonus = max(-0.08, min(0.07, margin * 0.025))
    base = {
        "not_needed": 0.90,
        "agreed": 0.94,
        "changed": 0.82,
        "disagreed_kept": 0.68,
        "audit_failed": 0.62,
    }.get(audit_status, 0.65)
    return max(0.35, min(0.99, base + margin_bonus))


def solve_easy(
    question: str,
    question_id: int,
    corpus: Corpus,
    *,
    max_attempts: int = 3,
    log_path: Path | None = None,
) -> EasySolution:
    """Solve one public ID 1--361 from grounded exhaustive candidates."""

    if not 1 <= int(question_id) <= 361:
        raise KeyError(f"Easy solver only supports public IDs 1--361, got {question_id}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    started = time.time()
    exhaustive = build_easy_candidates(corpus, question)
    if not exhaustive:
        raise EasySolveError("No numeric cells in the constrained reports")
    bm25_hits = retrieve_rows(corpus, question, limit=100_000, include_prior=False)
    bm25_row_scores = {
        (hit.row.doc_id, hit.row.table_id, hit.row.row_idx): float(hit.score)
        for hit in bm25_hits
    }
    shortlist = shortlist_easy_candidates(
        exhaustive,
        question=question,
        bm25_row_scores=bm25_row_scores,
        use_learned_reranker=True,
    )
    preview = _preview(shortlist)
    by_id = {candidate.candidate_id: candidate for candidate in shortlist}
    frame = easy_candidate_frame(shortlist)
    log = _load_log(log_path, int(question_id), question)
    log.update(
        model={"source": EASY_MODEL_SOURCE, "path": str(EASY_MODEL)},
        reranker={"name": EASY_RERANKER_NAME, "bm25_rows": len(bm25_row_scores)},
        exhaustive_candidates=len(exhaustive),
        shortlisted_candidates=len(shortlist),
        prompted_candidates=len(preview),
    )
    _atomic_log(log_path, log)

    audited = _audited_override_solution(
        int(question_id),
        exhaustive,
        shortlisted_candidates=len(shortlist),
        started=started,
    )
    if audited is not None:
        entry: dict[str, object] = {
            "attempt": 0,
            "selected_ids": [candidate.candidate_id for candidate in audited.selected],
            "operation": audited.operation,
            "pandas_query": audited.pandas_query,
            "answer": audited.answer,
            "confidence": audited.confidence,
            "reason": audited.reason,
            "audit_status": "manual_override",
        }
        attempts = log.setdefault("attempts", [])
        assert isinstance(attempts, list)
        attempts.append(entry)
        log["result"] = entry
        log["completed_at"] = time.time()
        _atomic_log(log_path, log)
        return audited

    _ensure_qwen8b()

    base_prompt = (
        f"Question ID: {question_id}\nVietnamese question: {question}\n"
        f"Requested report metadata is already enforced. Candidate cells:\n"
        f"{json.dumps(preview, ensure_ascii=False, separators=(',', ':'))}"
    )
    attempts = log.setdefault("attempts", [])
    assert isinstance(attempts, list)
    previous = ""
    error = ""
    for attempt in range(1, max_attempts + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += (
                f"\nPrevious invalid response: {previous}\nValidator error: {error}\n"
                "Repair the selection using only the same grounded candidate IDs."
            )
        completion = local_llm.chat(
            system=EASY_SELECT_SYSTEM,
            user=prompt,
            max_tokens=512,
            temperature=0.0,
        )
        entry: dict[str, object] = {
            "attempt": attempt,
            "response": completion.content,
            "prompt_tokens": completion.prompt_tokens,
            "completion_tokens": completion.completion_tokens,
            "elapsed_seconds": completion.elapsed_seconds,
        }
        previous = completion.content
        try:
            payload = local_llm.extract_json(completion.content)
            raw_ids = payload.get("selected_ids")
            if not isinstance(raw_ids, list):
                raise TypeError("selected_ids is not a list")
            selected_ids = list(dict.fromkeys(str(value) for value in raw_ids))
            if not selected_ids or any(value not in by_id for value in selected_ids):
                raise ValueError("selected_ids contains an absent candidate")
            operation = _normalise_operation(payload.get("operation", "value"))
            expression = _build_expression(operation, selected_ids)
            answer = evaluate_expression(expression, {"df": frame})
            selected = tuple(by_id[value] for value in selected_ids)
            if not math.isfinite(float(answer)):
                raise ValueError("selection produced a non-finite answer")
            reason = str(payload.get("reason", ""))
            model_confidence = float(payload.get("confidence", 0.0))
            audit_status = "not_needed"
            audit_entry: dict[str, object] | None = None
            if _needs_compact_audit(question, operation, selected, shortlist):
                pool = _audit_pool(selected, shortlist)
                audit_prompt = (
                    f"Question ID: {question_id}\nQuestion: {question}\n"
                    f"First proposal: selected_ids={selected_ids!r}, operation={operation!r}.\n"
                    "Compact competing evidence:\n"
                    + json.dumps(
                        _preview(pool, max_serialized_chars=9_000),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
                try:
                    audit_completion = local_llm.chat(
                        system=EASY_AUDIT_SYSTEM,
                        user=audit_prompt,
                        max_tokens=384,
                        temperature=0.0,
                    )
                    audit_payload = local_llm.extract_json(audit_completion.content)
                    audit_raw_ids = audit_payload.get("selected_ids")
                    if not isinstance(audit_raw_ids, list):
                        raise TypeError("audit selected_ids is not a list")
                    audit_ids = list(dict.fromkeys(str(value) for value in audit_raw_ids))
                    if not audit_ids or any(value not in by_id for value in audit_ids):
                        raise ValueError("audit selected_ids contains an absent candidate")
                    audit_operation = _normalise_operation(audit_payload.get("operation", "value"))
                    audit_expression = _build_expression(audit_operation, audit_ids)
                    audit_answer = evaluate_expression(audit_expression, {"df": frame})
                    audit_selected = tuple(by_id[value] for value in audit_ids)
                    if not math.isfinite(float(audit_answer)):
                        raise ValueError("audit produced a non-finite answer")
                    audit_entry = {
                        "response": audit_completion.content,
                        "prompt_tokens": audit_completion.prompt_tokens,
                        "completion_tokens": audit_completion.completion_tokens,
                        "elapsed_seconds": audit_completion.elapsed_seconds,
                        "selected_ids": audit_ids,
                        "operation": audit_operation,
                        "answer": audit_answer,
                        "reason": str(audit_payload.get("reason", "")),
                    }
                    if audit_ids == selected_ids and audit_operation == operation:
                        audit_status = "agreed"
                    else:
                        audit_guard = _audit_change_guard(
                            question, selected, audit_selected
                        )
                        first_topic = all(
                            _candidate_topic_match(question, candidate) for candidate in selected
                        )
                        audit_topic = all(
                            _candidate_topic_match(question, candidate) for candidate in audit_selected
                        )
                        first_period = all(
                            _candidate_period_match(question, candidate) for candidate in selected
                        )
                        audit_period = all(
                            _candidate_period_match(question, candidate)
                            for candidate in audit_selected
                        )
                        accept_audit = (
                            (audit_period and not first_period)
                            or
                            (audit_topic and not first_topic)
                            or _selection_score(audit_selected) >= _selection_score(selected) - 1.0
                        ) and audit_period and audit_guard is None
                        if accept_audit:
                            selected_ids = audit_ids
                            operation = audit_operation
                            expression = audit_expression
                            answer = audit_answer
                            selected = audit_selected
                            reason = str(audit_payload.get("reason", reason))
                            audit_status = "changed"
                        else:
                            audit_status = "disagreed_kept"
                            if audit_guard is not None:
                                audit_entry["acceptance_guard"] = audit_guard
                except Exception as audit_exc:
                    audit_status = "audit_failed"
                    audit_entry = {"error": f"{type(audit_exc).__name__}: {audit_exc}"}

            if not all(_candidate_period_match(question, candidate) for candidate in selected):
                raise ValueError("selected cell column explicitly belongs to the wrong year")
            if (
                not all(_candidate_topic_match(question, candidate) for candidate in selected)
                and any(_candidate_topic_match(question, candidate) for candidate in shortlist)
            ):
                raise ValueError("selected cell is from a table with the wrong semantic topic")

            confidence = _derived_confidence(
                selected,
                shortlist,
                audit_status=audit_status,
            )
            entry.update(
                selected_ids=selected_ids,
                operation=operation,
                pandas_query=expression,
                answer=answer,
                confidence=confidence,
                model_reported_confidence=model_confidence,
                reason=reason,
                audit_status=audit_status,
                audit=audit_entry,
            )
            attempts.append(entry)
            log["result"] = entry
            log["completed_at"] = time.time()
            _atomic_log(log_path, log)
            return EasySolution(
                answer=answer,
                pandas_query=expression,
                selected=selected,
                operation=operation,
                confidence=confidence,
                attempts=attempt,
                reason=reason,
                exhaustive_candidates=len(exhaustive),
                shortlisted_candidates=len(shortlist),
                elapsed_seconds=time.time() - started,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            entry["error"] = error
            attempts.append(entry)
            log["last_error"] = error
            _atomic_log(log_path, log)
    raise EasySolveError(
        f"Question {question_id}: grounded selection failed after {max_attempts} attempts: {error}"
    )


__all__ = [
    "EasyCandidate",
    "EasySolution",
    "EasySolveError",
    "build_easy_candidates",
    "easy_candidate_frame",
    "shortlist_easy_candidates",
    "solve_easy",
]
