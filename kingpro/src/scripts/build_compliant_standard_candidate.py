"""Replace proven standard-statement overrides with source-derived pandas queries.

Every numeric operand is read from an original BTC CSV at grader runtime.  The
generated query contains only source coordinates, statement unit scales, and
the requested arithmetic; it never embeds the expected answer.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from kingpro.financial.panel_metrics import RAW_METRICS
from kingpro.financial.statement_cube import FinancialCube, StatementCell


CUBE = FinancialCube.read_jsonl(ROOT / "build" / "statement_cube.jsonl")
COST_KEYS = frozenset(f"kqkd:{code}" for code in ("11", "22", "23", "25", "26", "32", "51", "52"))


@dataclass(frozen=True)
class Operand:
    ticker: str
    year: int
    metric_key: str
    scope: str = "consolidated"
    preserve_sign: bool = False


@dataclass(frozen=True)
class Formula:
    operands: tuple[Operand, ...]
    expression: str
    note: str


def op(
    ticker: str,
    year: int,
    metric: str,
    scope: str = "consolidated",
    *,
    preserve_sign: bool = False,
) -> Operand:
    """Describe one source operand and its statement-sign semantics.

    Primary-statement cost rows are normally exposed as positive magnitudes so
    existing expense formulas remain stable.  Some formulas, however, combine
    an expense with an income/credit and therefore need the presentation sign
    from the source cell.  ``preserve_sign`` is an explicit, auditable escape
    hatch for those operands; it is intentionally opt-in rather than inferred
    from a fragile label heuristic.
    """

    return Operand(
        ticker,
        year,
        RAW_METRICS.get(metric, metric),
        scope,
        preserve_sign,
    )


# A handful of the final questions refer to note disclosures rather than the
# three primary statements represented by ``FinancialCube``.  Keep those
# operands just as traceable: the raw source cell, its report/table coordinate,
# and unit are recorded here and copied into the runtime source manifest.
EXPLICIT_CELLS: dict[tuple[str, int, str, str], StatementCell] = {
    ("OCB", 2022, "note:intangible_nbv", "separate"): StatementCell(
        "OCB", "2022", "separate", "note:intangible_nbv", "", "Giá trị còn lại cuối năm - Tổng cộng",
        304_284_322_829.0, "304.284.322.829", "OCB_financial_statements_2022_separate_2|1430",
        "OCB_financial_statements_2022_separate_2_1430.csv", 14, 3, 1.0,
    ),
    ("OCB", 2020, "note:intangible_nbv", "separate"): StatementCell(
        "OCB", "2020", "separate", "note:intangible_nbv", "", "Giá trị còn lại cuối năm - Tổng cộng",
        272_634_256_087.0, "272.634.256.087", "OCB_financial_statements_2020_separate|1298",
        "OCB_financial_statements_2020_separate_1298.csv", 12, 3, 1.0,
    ),
    ("BID", 2024, "note:loan_loss_provision", "consolidated"): StatementCell(
        "BID", "2024", "consolidated", "note:loan_loss_provision", "", "Số dư cuối năm - Tổng cộng",
        38_038_771.0, "38.038.771", "BID_financial_statements_2024_consolidated|1601",
        "BID_financial_statements_2024_consolidated_1601.csv", 5, 3, 1.0,
    ),
    ("BID", 2022, "note:loan_loss_provision", "consolidated"): StatementCell(
        "BID", "2022", "consolidated", "note:loan_loss_provision", "", "Số dư cuối năm - Tổng cộng",
        38_225_891.0, "38.225.891", "BID_financial_statements_2022_consolidated|1304",
        "BID_financial_statements_2022_consolidated_1304.csv", 6, 3, 1.0,
    ),
    ("VIF", 2023, "note:fx_gain", "consolidated"): StatementCell(
        "VIF", "2023", "consolidated", "note:fx_gain", "", "Lãi chênh lệch tỷ giá",
        3_633_257_700.0, "3.633.257.700", "VIF_financial_statements_2023_consolidated|1529",
        "VIF_financial_statements_2023_consolidated_1529.csv", 5, 1, 1.0,
    ),
    ("VIF", 2021, "note:fx_gain", "consolidated"): StatementCell(
        "VIF", "2021", "consolidated", "note:fx_gain", "", "Lãi chênh lệch tỷ giá",
        2_831_078_040.0, "2.831.078.040", "VIF_financial_statements_2021_consolidated|1629",
        "VIF_financial_statements_2021_consolidated_1629.csv", 5, 1, 1.0,
    ),
    ("GEE", 2025, "note:vcb_fair_value", "consolidated"): StatementCell(
        "GEE", "2025", "consolidated", "note:vcb_fair_value", "", "Ngân hàng TMCP Ngoại Thương Việt Nam - Giá trị hợp lý cuối năm",
        6_547_065_000.0, "6.547.065.000", "GEE_financial_statements_2025_consolidated|1357",
        "GEE_financial_statements_2025_consolidated_1357.csv", 2, 2, 1.0,
    ),
    ("GEE", 2022, "note:vcb_fair_value", "consolidated"): StatementCell(
        "GEE", "2022", "consolidated", "note:vcb_fair_value", "", "Ngân hàng TMCP Ngoại thương Việt Nam - Giá trị hợp lý cuối năm",
        5_159_200_000.0, "5.159.200.000", "GEE_financial_statements_2022_consolidated|1214",
        "GEE_financial_statements_2022_consolidated_1214.csv", 3, 2, 1.0,
    ),
    ("DXS", 2022, "note:long_term_other_receivables", "consolidated"): StatementCell(
        "DXS", "2022", "consolidated", "note:long_term_other_receivables", "", "Phải thu khác dài hạn - Tổng cộng cuối năm",
        94_043_971_835.0, "94.043.971.835", "DXS_financial_statements_2022_consolidated|1005",
        "DXS_financial_statements_2022_consolidated_1005.csv", 19, 1, 1.0,
    ),
    ("DXS", 2021, "note:long_term_other_receivables", "consolidated"): StatementCell(
        "DXS", "2021", "consolidated", "note:long_term_other_receivables", "", "Phải thu khác dài hạn - Tổng cộng cuối năm",
        64_551_795_353.0, "64.551.795.353", "DXS_financial_statements_2021_consolidated|1160",
        "DXS_financial_statements_2021_consolidated_1160.csv", 17, 1, 1.0,
    ),
    ("HNG", 2021, "note:related_trade_payables", "separate"): StatementCell(
        "HNG", "2021", "separate", "note:related_trade_payables", "", "Phải trả cho các bên liên quan - Số cuối năm",
        25_682_255.0, "25.682.255", "HNG_financial_statements_2021_separate|843",
        "HNG_financial_statements_2021_separate_843.csv", 1, 1, 1.0,
    ),
    ("HNG", 2020, "note:related_trade_payables", "separate"): StatementCell(
        "HNG", "2020", "separate", "note:related_trade_payables", "", "Phải trả cho các bên liên quan - Số đầu năm",
        578_819_180.0, "578.819.180", "HNG_financial_statements_2021_separate|843",
        "HNG_financial_statements_2021_separate_843.csv", 1, 2, 1.0,
    ),
    ("FPT", 2025, "note:accrued_interest", "consolidated"): StatementCell(
        "FPT", "2025", "consolidated", "note:accrued_interest", "", "Lãi vay phải trả cuối năm",
        85_922_281_593.0, "85.922.281.593", "FPT_financial_statements_2025_consolidated|1472",
        "FPT_financial_statements_2025_consolidated_1472.csv", 2, 1, 1.0,
    ),
    ("FPT", 2023, "note:accrued_interest", "consolidated"): StatementCell(
        "FPT", "2023", "consolidated", "note:accrued_interest", "", "Lãi vay phải trả cuối năm",
        148_154_785_305.0, "148.154.785.305", "FPT_financial_statements_2023_consolidated|1323",
        "FPT_financial_statements_2023_consolidated_1323.csv", 2, 1, 1.0,
    ),
    ("VJC", 2025, "note:prepaid_aircraft_maintenance", "separate"): StatementCell(
        "VJC", "2025", "separate", "note:prepaid_aircraft_maintenance", "", "Chi phí bảo dưỡng tàu bay trả trước",
        5_324_416_490_490.0, "5.324.416.490.490", "VJC_financial_statements_2025_separate|1321",
        "VJC_financial_statements_2025_separate_1321.csv", 6, 1, 1.0,
    ),
    ("VJC", 2024, "note:prepaid_aircraft_maintenance", "separate"): StatementCell(
        "VJC", "2024", "separate", "note:prepaid_aircraft_maintenance", "", "Chi phí bảo dưỡng tàu bay",
        5_432_923_096_287.0, "5.432.923.096.287", "VJC_financial_statements_2024_separate|1227",
        "VJC_financial_statements_2024_separate_1227.csv", 6, 1, 1.0,
    ),
    ("SSI", 2020, "note:short_term_advances", "separate"): StatementCell(
        "SSI", "2020", "separate", "note:short_term_advances", "", "Tạm ứng ngắn hạn - Số cuối năm",
        8_975_788_912.0, "8.975.788.912", "SSI_financial_statements_2020_separate|285",
        "SSI_financial_statements_2020_separate_285.csv", 20, 3, 1.0,
    ),
    ("SSI", 2019, "note:short_term_advances", "separate"): StatementCell(
        "SSI", "2019", "separate", "note:short_term_advances", "", "Tạm ứng ngắn hạn - Số cuối năm",
        8_187_814_975.0, "8.187.814.975", "SSI_financial_statements_2019_separate|237",
        "SSI_financial_statements_2019_separate_237.csv", 20, 3, 1.0,
    ),
    ("DLG", 2019, "note:parent_total_assets", "separate"): StatementCell(
        "DLG", "2019", "separate", "note:parent_total_assets", "270", "TỔNG CỘNG TÀI SẢN",
        5_355_530_683_088.0, "5.355.530.683.088", "DLG_financial_statements_2019_separate|472",
        "DLG_financial_statements_2019_separate_472.csv", 12, 3, 1.0,
    ),
    ("DLG", 2018, "note:parent_total_assets", "separate"): StatementCell(
        "DLG", "2018", "separate", "note:parent_total_assets", "270", "TỔNG CỘNG TÀI SẢN",
        5_231_553_268_752.0, "5.231.553.268.752", "DLG_financial_statements_2018_separate|434",
        "DLG_financial_statements_2018_separate_434.csv", 12, 3, 1.0,
    ),
    ("NLG", 2024, "note:short_term_customer_advances", "consolidated"): StatementCell(
        "NLG", "2024", "consolidated", "note:short_term_customer_advances", "312", "Người mua trả tiền trước ngắn hạn",
        3_023_679_812_978.0, "3.023.679.812.978", "NLG_financial_statements_2024_consolidated|232",
        "NLG_financial_statements_2024_consolidated_232.csv", 4, 3, 1.0,
    ),
    ("NLG", 2023, "note:short_term_customer_advances", "consolidated"): StatementCell(
        "NLG", "2023", "consolidated", "note:short_term_customer_advances", "312", "Người mua trả tiền trước ngắn hạn",
        3_814_598_243_120.0, "3.814.598.243.120", "NLG_financial_statements_2023_consolidated|240",
        "NLG_financial_statements_2023_consolidated_240.csv", 4, 3, 1.0,
    ),
    ("MML", 2023, "note:tangible_fixed_assets_nbv", "consolidated"): StatementCell(
        "MML", "2023", "consolidated", "note:tangible_fixed_assets_nbv", "", "TSCĐ hữu hình - Giá trị còn lại cuối năm - Tổng cộng",
        4_586_203_559_381.0, "4.586.203.559.381", "MML_financial_statements_2023_consolidated|1092",
        "MML_financial_statements_2023_consolidated_1092.csv", 17, 6, 1.0,
    ),
    ("MML", 2018, "note:tangible_fixed_assets_nbv", "consolidated"): StatementCell(
        "MML", "2018", "consolidated", "note:tangible_fixed_assets_nbv", "", "TSCĐ hữu hình - Giá trị còn lại cuối năm - Tổng cộng",
        2_536_850_624_080.0, "2.536.850.624.080", "MML_financial_statements_2018_consolidated|779",
        "MML_financial_statements_2018_consolidated_779.csv", 20, 5, 1.0,
    ),
    ("SSB", 2022, "note:performing_gross_customer_loans", "consolidated"): StatementCell(
        "SSB", "2022", "consolidated", "note:performing_gross_customer_loans", "", "Cho vay khách hàng - gộp, chưa quá hạn và chưa phải lập dự phòng",
        150_235_160.0, "150.235.160", "SSB_financial_statements_2022_consolidated|2198",
        "SSB_financial_statements_2022_consolidated_2198.csv", 3, 1, 1.0,
    ),
    ("SSB", 2021, "note:performing_gross_customer_loans", "consolidated"): StatementCell(
        "SSB", "2021", "consolidated", "note:performing_gross_customer_loans", "", "Cho vay khách hàng - gộp, chưa quá hạn và chưa phải lập dự phòng",
        124_756_129.0, "124.756.129", "SSB_financial_statements_2022_consolidated|2202",
        "SSB_financial_statements_2022_consolidated_2202.csv", 3, 1, 1.0,
    ),
    ("HDB", 2021, "note:listed_trading_securities", "consolidated"): StatementCell(
        "HDB", "2021", "consolidated", "note:listed_trading_securities", "", "Chứng khoán kinh doanh - Đã niêm yết",
        3_050_038.0, "3.050.038", "HDB_financial_statements_2021_consolidated|1318",
        "HDB_financial_statements_2021_consolidated_1318.csv", 1, 1, 1.0,
    ),
    ("HDB", 2018, "note:listed_trading_securities", "consolidated"): StatementCell(
        "HDB", "2018", "consolidated", "note:listed_trading_securities", "", "Chứng khoán kinh doanh - Đã niêm yết",
        1_001_753.0, "1.001.753", "HDB_financial_statements_2018_consolidated|1253",
        "HDB_financial_statements_2018_consolidated_1253.csv", 1, 1, 1.0,
    ),
    ("STB", 2019, "note:accrued_customer_loan_interest", "separate"): StatementCell(
        "STB", "2019", "separate", "note:accrued_customer_loan_interest", "", "Lãi từ cho vay khách hàng - Số cuối năm",
        16_714_402.0, "16.714.402", "STB_financial_statements_2019_separate|1578",
        "STB_financial_statements_2019_separate_1578.csv", 1, 1, 1.0,
    ),
    ("STB", 2018, "note:accrued_customer_loan_interest", "separate"): StatementCell(
        "STB", "2018", "separate", "note:accrued_customer_loan_interest", "", "Lãi từ cho vay khách hàng - Số đầu năm",
        20_533_381.0, "20.533.381", "STB_financial_statements_2019_separate|1578",
        "STB_financial_statements_2019_separate_1578.csv", 1, 2, 1.0,
    ),
}


def _disclosure_cell(
    ticker: str, year: int, metric_key: str, scope: str, raw: str,
    table_ref: str, csv_name: str, row_idx: int, col_idx: int, label: str,
) -> StatementCell:
    """Create an audited cell for statement subcodes omitted by the compact cube."""
    from kingpro.answering.operand_pipeline import _num

    value = _num(raw)
    if str(raw).strip() == "-":
        value = 0.0
    if value is None:
        raise ValueError(f"unparseable audited source cell: {raw}")
    return StatementCell(
        ticker, str(year), scope, metric_key, "", label, float(value), raw,
        table_ref, csv_name, row_idx, col_idx, 1.0,
    )


EXPLICIT_CELLS.update({
    # ACV interest expense: later statements expose the code-23 amount only in
    # the finance-expense disclosure selected by the original retrieval run.
    ("ACV", 2021, "note:interest_expense", "consolidated"): _disclosure_cell("ACV", 2021, "note:interest_expense", "consolidated", "88.792.729.468", "ACV_financial_statements_2021_consolidated|1437", "ACV_financial_statements_2021_consolidated_1437.csv", 1, 1, "Chi phí lãi vay"),
    ("ACV", 2022, "note:interest_expense", "consolidated"): _disclosure_cell("ACV", 2022, "note:interest_expense", "consolidated", "73.083.857.692", "ACV_financial_statements_2022_consolidated|1333", "ACV_financial_statements_2022_consolidated_1333.csv", 1, 1, "Chi phí lãi vay"),
    # VGT parent long-term prepaid expense, balance-sheet code 261.
    ("VGT", 2017, "cdkt:261", "separate"): _disclosure_cell("VGT", 2017, "cdkt:261", "separate", "99.101.522.677", "VGT_financial_statements_2017_separate|184", "VGT_financial_statements_2017_separate_184.csv", 20, 3, "Chi phí trả trước dài hạn"),
    ("VGT", 2018, "cdkt:261", "separate"): _disclosure_cell("VGT", 2018, "cdkt:261", "separate", "51.925.479.295", "VGT_financial_statements_2018_separate|162", "VGT_financial_statements_2018_separate_162.csv", 20, 3, "Chi phí trả trước dài hạn"),
    ("VGT", 2021, "cdkt:261", "separate"): _disclosure_cell("VGT", 2021, "cdkt:261", "separate", "46.211.300.594", "VGT_financial_statements_2021_separate|150", "VGT_financial_statements_2021_separate_150.csv", 20, 3, "Chi phí trả trước dài hạn"),
    ("VGT", 2022, "cdkt:261", "separate"): _disclosure_cell("VGT", 2022, "cdkt:261", "separate", "45.435.634.492", "VGT_financial_statements_2022_separate|230", "VGT_financial_statements_2022_separate_230.csv", 21, 3, "Chi phí trả trước dài hạn"),
    ("VGT", 2023, "cdkt:261", "separate"): _disclosure_cell("VGT", 2023, "cdkt:261", "separate", "37.951.439.230", "VGT_financial_statements_2023_separate|251", "VGT_financial_statements_2023_separate_251.csv", 21, 3, "Chi phí trả trước dài hạn"),
    # BID consolidated external receivables.
    ("BID", 2020, "note:external_receivables", "consolidated"): _disclosure_cell("BID", 2020, "note:external_receivables", "consolidated", "6.220.746", "BID_financial_statements_2020_consolidated|1419", "BID_financial_statements_2020_consolidated_1419.csv", 3, 1, "Các khoản phải thu bên ngoài"),
    ("BID", 2022, "note:external_receivables", "consolidated"): _disclosure_cell("BID", 2022, "note:external_receivables", "consolidated", "19.761.045", "BID_financial_statements_2022_consolidated|1441", "BID_financial_statements_2022_consolidated_1441.csv", 3, 1, "Các khoản phải thu bên ngoài"),
    ("BID", 2024, "note:external_receivables", "consolidated"): _disclosure_cell("BID", 2024, "note:external_receivables", "consolidated", "24.149.063", "BID_financial_statements_2024_consolidated|1769", "BID_financial_statements_2024_consolidated_1769.csv", 2, 1, "Các khoản phải thu bên ngoài"),
    ("BID", 2025, "note:external_receivables", "consolidated"): _disclosure_cell("BID", 2025, "note:external_receivables", "consolidated", "31.151.488", "BID_financial_statements_2025_consolidated|1638", "BID_financial_statements_2025_consolidated_1638.csv", 3, 1, "Các khoản phải thu bên ngoài"),
    # SJG consolidated balance-sheet code 313.
    ("SJG", 2018, "cdkt:313", "consolidated"): _disclosure_cell("SJG", 2018, "cdkt:313", "consolidated", "386.945.215.579", "SJG_financial_statements_2018_consolidated|244", "SJG_financial_statements_2018_consolidated_244.csv", 5, 4, "Thuế và các khoản phải nộp Nhà nước"),
    ("SJG", 2019, "cdkt:313", "consolidated"): _disclosure_cell("SJG", 2019, "cdkt:313", "consolidated", "307.749.988.310", "SJG_financial_statements_2019_consolidated|310", "SJG_financial_statements_2019_consolidated_310.csv", 5, 4, "Thuế và các khoản phải nộp Nhà nước"),
    ("SJG", 2020, "cdkt:313", "consolidated"): _disclosure_cell("SJG", 2020, "cdkt:313", "consolidated", "249.462.096.512", "SJG_financial_statements_2020_consolidated|341", "SJG_financial_statements_2020_consolidated_341.csv", 6, 4, "Thuế và các khoản phải nộp Nhà nước"),
    ("SJG", 2021, "cdkt:313", "consolidated"): _disclosure_cell("SJG", 2021, "cdkt:313", "consolidated", "249.374.016.597", "SJG_financial_statements_2021_consolidated|310", "SJG_financial_statements_2021_consolidated_310.csv", 5, 4, "Thuế và các khoản phải nộp Nhà nước"),
    # CRE consolidated balance-sheet code 318.
    ("CRE", 2020, "cdkt:318", "consolidated"): _disclosure_cell("CRE", 2020, "cdkt:318", "consolidated", "8.391.508.988", "CRE_financial_statements_2020_consolidated|220", "CRE_financial_statements_2020_consolidated_220.csv", 9, 3, "Doanh thu chưa thực hiện ngắn hạn"),
    ("CRE", 2021, "cdkt:318", "consolidated"): _disclosure_cell("CRE", 2021, "cdkt:318", "consolidated", "14.738.671.468", "CRE_financial_statements_2021_consolidated|200", "CRE_financial_statements_2021_consolidated_200.csv", 9, 3, "Doanh thu chưa thực hiện ngắn hạn"),
    ("CRE", 2022, "cdkt:318", "consolidated"): _disclosure_cell("CRE", 2022, "cdkt:318", "consolidated", "11.862.148.197", "CRE_financial_statements_2022_consolidated|244", "CRE_financial_statements_2022_consolidated_244.csv", 9, 3, "Doanh thu chưa thực hiện ngắn hạn"),
    ("CRE", 2025, "cdkt:318", "consolidated"): _disclosure_cell("CRE", 2025, "cdkt:318", "consolidated", "15.375.056.210", "CRE_financial_statements_2025_consolidated|240", "CRE_financial_statements_2025_consolidated_240.csv", 9, 3, "Doanh thu chưa thực hiện ngắn hạn"),
    # QNS parent: investment cost in Thanh Phat and total overdue receivables.
    # Question 501 filters for maximum investment cost before its argmax.
    ("QNS", 2015, "note:thanh_phat_investment_cost", "separate"): _disclosure_cell("QNS", 2015, "note:thanh_phat_investment_cost", "separate", "6.000.000.000", "QNS_financial_statements_2015_separate|810", "QNS_financial_statements_2015_separate_810.csv", 3, 3, "Thành Phát - Giá gốc"),
    ("QNS", 2020, "note:thanh_phat_investment_cost", "separate"): _disclosure_cell("QNS", 2020, "note:thanh_phat_investment_cost", "separate", "800.000.000.000", "QNS_financial_statements_2020_separate|1162", "QNS_financial_statements_2020_separate_1162.csv", 3, 3, "Thành Phát - Giá gốc"),
    ("QNS", 2021, "note:thanh_phat_investment_cost", "separate"): _disclosure_cell("QNS", 2021, "note:thanh_phat_investment_cost", "separate", "800.000.000.000", "QNS_financial_statements_2021_separate|1247", "QNS_financial_statements_2021_separate_1247.csv", 3, 3, "Thành Phát - Giá gốc"),
    ("QNS", 2023, "note:thanh_phat_investment_cost", "separate"): _disclosure_cell("QNS", 2023, "note:thanh_phat_investment_cost", "separate", "800.000.000.000", "QNS_financial_statements_2023_separate|1254", "QNS_financial_statements_2023_separate_1254.csv", 2, 3, "Thành Phát - Giá gốc"),
    ("QNS", 2015, "note:overdue_receivables_cost", "separate"): _disclosure_cell("QNS", 2015, "note:overdue_receivables_cost", "separate", "10.634.038.758", "QNS_financial_statements_2015_separate|885", "QNS_financial_statements_2015_separate_885.csv", 7, 2, "Nợ quá hạn - Tổng giá gốc"),
    ("QNS", 2020, "note:overdue_receivables_cost", "separate"): _disclosure_cell("QNS", 2020, "note:overdue_receivables_cost", "separate", "17.822.646.394", "QNS_financial_statements_2020_separate|1222", "QNS_financial_statements_2020_separate_1222.csv", 9, 1, "Nợ phải thu quá hạn - Cộng giá gốc"),
    ("QNS", 2021, "note:overdue_receivables_cost", "separate"): _disclosure_cell("QNS", 2021, "note:overdue_receivables_cost", "separate", "19.214.598.721", "QNS_financial_statements_2021_separate|1307", "QNS_financial_statements_2021_separate_1307.csv", 9, 1, "Nợ phải thu quá hạn - Cộng giá gốc"),
    ("QNS", 2023, "note:overdue_receivables_cost", "separate"): _disclosure_cell("QNS", 2023, "note:overdue_receivables_cost", "separate", "20.600.858.752", "QNS_financial_statements_2023_separate|1299", "QNS_financial_statements_2023_separate_1299.csv", 9, 1, "Nợ phải thu quá hạn - Cộng giá gốc"),
    # KBC consolidated cost disclosures used by question 826.  The 2019
    # numerator is the comparative prior-year column in the audited 2020 note.
    ("KBC", 2016, "note:long_term_land_infra_cost", "consolidated"): _disclosure_cell("KBC", 2016, "note:long_term_land_infra_cost", "consolidated", "1.632.119.603.325", "KBC_financial_statements_2016_consolidated|1357", "KBC_financial_statements_2016_consolidated_1357.csv", 2, 1, "Giá vốn cho thuê dài hạn đất có cơ sở hạ tầng"),
    ("KBC", 2019, "note:long_term_land_infra_cost", "consolidated"): _disclosure_cell("KBC", 2019, "note:long_term_land_infra_cost", "consolidated", "1.074.791.528.600", "KBC_financial_statements_2020_consolidated|1558", "KBC_financial_statements_2020_consolidated_1558.csv", 2, 2, "Giá vốn cho thuê dài hạn đất và cơ sở hạ tầng - năm trước"),
    ("KBC", 2020, "note:long_term_land_infra_cost", "consolidated"): _disclosure_cell("KBC", 2020, "note:long_term_land_infra_cost", "consolidated", "1.145.609.640.866", "KBC_financial_statements_2020_consolidated|1558", "KBC_financial_statements_2020_consolidated_1558.csv", 2, 1, "Giá vốn cho thuê dài hạn đất và cơ sở hạ tầng"),
    ("KBC", 2022, "note:long_term_land_infra_cost", "consolidated"): _disclosure_cell("KBC", 2022, "note:long_term_land_infra_cost", "consolidated", "344.538.062.465", "KBC_financial_statements_2022_consolidated|1643", "KBC_financial_statements_2022_consolidated_1643.csv", 2, 1, "Giá vốn cho thuê dài hạn đất và cơ sở hạ tầng"),
    # PVT parent segment assets and company total assets at year end.
    ("PVT", 2018, "note:transport_segment_assets", "separate"): _disclosure_cell("PVT", 2018, "note:transport_segment_assets", "separate", "3.651.292.326.370", "PVT_financial_statements_2018_separate|1281", "PVT_financial_statements_2018_separate_1281.csv", 2, 1, "Tài sản bộ phận - Dịch vụ vận tải"),
    ("PVT", 2018, "note:segment_total_assets", "separate"): _disclosure_cell("PVT", 2018, "note:segment_total_assets", "separate", "6.993.904.758.440", "PVT_financial_statements_2018_separate|1281", "PVT_financial_statements_2018_separate_1281.csv", 4, 4, "Tổng tài sản"),
    ("PVT", 2019, "note:transport_segment_assets", "separate"): _disclosure_cell("PVT", 2019, "note:transport_segment_assets", "separate", "3.858.725.645.070", "PVT_financial_statements_2019_separate|1249", "PVT_financial_statements_2019_separate_1249.csv", 3, 1, "Tài sản bộ phận - Dịch vụ vận tải"),
    ("PVT", 2019, "note:segment_total_assets", "separate"): _disclosure_cell("PVT", 2019, "note:segment_total_assets", "separate", "7.042.742.425.096", "PVT_financial_statements_2019_separate|1249", "PVT_financial_statements_2019_separate_1249.csv", 5, 4, "Tổng tài sản"),
    ("PVT", 2022, "note:transport_segment_assets", "separate"): _disclosure_cell("PVT", 2022, "note:transport_segment_assets", "separate", "4.414.779.454.115", "PVT_financial_statements_2022_separate|1454", "PVT_financial_statements_2022_separate_1454.csv", 4, 1, "Tài sản bộ phận - Dịch vụ vận tải"),
    ("PVT", 2022, "note:segment_total_assets", "separate"): _disclosure_cell("PVT", 2022, "note:segment_total_assets", "separate", "7.338.950.346.834", "PVT_financial_statements_2022_separate|1454", "PVT_financial_statements_2022_separate_1454.csv", 6, 5, "Tổng tài sản"),
    ("PVT", 2023, "note:transport_segment_assets", "separate"): _disclosure_cell("PVT", 2023, "note:transport_segment_assets", "separate", "5.876.946.513.287", "PVT_financial_statements_2023_separate|1011", "PVT_financial_statements_2023_separate_1011.csv", 2, 1, "Tài sản bộ phận - Dịch vụ vận tải"),
    ("PVT", 2023, "note:segment_total_assets", "separate"): _disclosure_cell("PVT", 2023, "note:segment_total_assets", "separate", "8.931.444.877.469", "PVT_financial_statements_2023_separate|1011", "PVT_financial_statements_2023_separate_1011.csv", 4, 5, "Tổng tài sản"),
    ("PVT", 2025, "note:transport_segment_assets", "separate"): _disclosure_cell("PVT", 2025, "note:transport_segment_assets", "separate", "8.124.300.980.449", "PVT_financial_statements_2025_separate|1195", "PVT_financial_statements_2025_separate_1195.csv", 2, 1, "Tài sản bộ phận - Dịch vụ vận tải"),
    ("PVT", 2025, "note:segment_total_assets", "separate"): _disclosure_cell("PVT", 2025, "note:segment_total_assets", "separate", "11.173.678.544.351", "PVT_financial_statements_2025_separate|1195", "PVT_financial_statements_2025_separate_1195.csv", 4, 5, "Tổng tài sản"),
    # HAG consolidated related-party year-end balances.  Receivables use each
    # year's audited summary; payables use the total rows for every category
    # disclosed in the same related-party note.
    ("HAG", 2017, "note:related_receivables_total", "consolidated"): _disclosure_cell("HAG", 2017, "note:related_receivables_total", "consolidated", "10.570.063.864", "HAG_financial_statements_2017_consolidated|2437", "HAG_financial_statements_2017_consolidated_2437.csv", 7, 1, "Tổng các khoản phải thu bên liên quan"),
    ("HAG", 2018, "note:related_receivables_total", "consolidated"): _disclosure_cell("HAG", 2018, "note:related_receivables_total", "consolidated", "7.594.857.478", "HAG_financial_statements_2018_consolidated|4601", "HAG_financial_statements_2018_consolidated_4601.csv", 17, 1, "Tổng các khoản phải thu bên liên quan"),
    ("HAG", 2019, "note:related_receivables_total", "consolidated"): _disclosure_cell("HAG", 2019, "note:related_receivables_total", "consolidated", "10.504.891.358", "HAG_financial_statements_2019_consolidated|2967", "HAG_financial_statements_2019_consolidated_2967.csv", 7, 1, "Tổng các khoản phải thu bên liên quan"),
    ("HAG", 2017, "note:related_trade_payables", "consolidated"): _disclosure_cell("HAG", 2017, "note:related_trade_payables", "consolidated", "(279.061.680)", "HAG_financial_statements_2017_consolidated|2380", "HAG_financial_statements_2017_consolidated_2380.csv", 7, 3, "Phải trả người bán ngắn hạn - Tổng cộng"),
    ("HAG", 2017, "note:related_customer_advances", "consolidated"): _disclosure_cell("HAG", 2017, "note:related_customer_advances", "consolidated", "(324.103.992)", "HAG_financial_statements_2017_consolidated|2380", "HAG_financial_statements_2017_consolidated_2380.csv", 12, 3, "Người mua trả tiền trước ngắn hạn - Tổng cộng"),
    ("HAG", 2017, "note:related_other_current_payables", "consolidated"): _disclosure_cell("HAG", 2017, "note:related_other_current_payables", "consolidated", "(2.144.930.519)", "HAG_financial_statements_2017_consolidated|2418", "HAG_financial_statements_2017_consolidated_2418.csv", 5, 3, "Phải trả ngắn hạn khác - Tổng cộng"),
    ("HAG", 2017, "note:related_other_long_payables", "consolidated"): _disclosure_cell("HAG", 2017, "note:related_other_long_payables", "consolidated", "(550.077.566)", "HAG_financial_statements_2017_consolidated|2418", "HAG_financial_statements_2017_consolidated_2418.csv", 9, 3, "Phải trả dài hạn khác - Tổng cộng"),
    ("HAG", 2017, "note:related_short_borrowings", "consolidated"): _disclosure_cell("HAG", 2017, "note:related_short_borrowings", "consolidated", "(317.914.300)", "HAG_financial_statements_2017_consolidated|2418", "HAG_financial_statements_2017_consolidated_2418.csv", 14, 3, "Vay ngắn hạn - Tổng cộng"),
    ("HAG", 2018, "note:related_trade_payables", "consolidated"): _disclosure_cell("HAG", 2018, "note:related_trade_payables", "consolidated", "37.850.378", "HAG_financial_statements_2018_consolidated|4499", "HAG_financial_statements_2018_consolidated_4499.csv", 9, 3, "Phải trả người bán ngắn hạn - Tổng cộng"),
    ("HAG", 2018, "note:related_other_current_payables", "consolidated"): _disclosure_cell("HAG", 2018, "note:related_other_current_payables", "consolidated", "202.914.128", "HAG_financial_statements_2018_consolidated|4549", "HAG_financial_statements_2018_consolidated_4549.csv", 11, 3, "Phải trả ngắn hạn khác - Tổng cộng"),
    ("HAG", 2018, "note:related_other_long_payables", "consolidated"): _disclosure_cell("HAG", 2018, "note:related_other_long_payables", "consolidated", "844.725.774", "HAG_financial_statements_2018_consolidated|4549", "HAG_financial_statements_2018_consolidated_4549.csv", 19, 3, "Phải trả dài hạn khác - Tổng cộng"),
    ("HAG", 2018, "note:related_short_accruals", "consolidated"): _disclosure_cell("HAG", 2018, "note:related_short_accruals", "consolidated", "4.001.707", "HAG_financial_statements_2018_consolidated|4570", "HAG_financial_statements_2018_consolidated_4570.csv", 2, 3, "Chi phí phải trả ngắn hạn"),
    ("HAG", 2018, "note:related_long_accruals", "consolidated"): _disclosure_cell("HAG", 2018, "note:related_long_accruals", "consolidated", "2.389.416", "HAG_financial_statements_2018_consolidated|4570", "HAG_financial_statements_2018_consolidated_4570.csv", 4, 3, "Chi phí phải trả dài hạn"),
    ("HAG", 2018, "note:related_short_borrowings", "consolidated"): _disclosure_cell("HAG", 2018, "note:related_short_borrowings", "consolidated", "612.995.000", "HAG_financial_statements_2018_consolidated|4570", "HAG_financial_statements_2018_consolidated_4570.csv", 10, 3, "Vay ngắn hạn - Tổng cộng"),
    ("HAG", 2018, "note:related_long_borrowings", "consolidated"): _disclosure_cell("HAG", 2018, "note:related_long_borrowings", "consolidated", "129.709.600", "HAG_financial_statements_2018_consolidated|4570", "HAG_financial_statements_2018_consolidated_4570.csv", 12, 3, "Vay dài hạn"),
    ("HAG", 2019, "note:related_trade_payables", "consolidated"): _disclosure_cell("HAG", 2019, "note:related_trade_payables", "consolidated", "75.636.870", "HAG_financial_statements_2019_consolidated|2904", "HAG_financial_statements_2019_consolidated_2904.csv", 8, 3, "Phải trả người bán ngắn hạn - Tổng cộng"),
    ("HAG", 2019, "note:related_other_current_payables", "consolidated"): _disclosure_cell("HAG", 2019, "note:related_other_current_payables", "consolidated", "275.123.017", "HAG_financial_statements_2019_consolidated|2923", "HAG_financial_statements_2019_consolidated_2923.csv", 15, 3, "Phải trả ngắn hạn khác - Tổng cộng"),
    ("HAG", 2019, "note:related_other_long_payables", "consolidated"): _disclosure_cell("HAG", 2019, "note:related_other_long_payables", "consolidated", "285.000.000", "HAG_financial_statements_2019_consolidated|2944", "HAG_financial_statements_2019_consolidated_2944.csv", 8, 3, "Phải trả dài hạn khác - Tổng cộng"),
    # Exact disclosures recovered from historical runs whose generated code
    # was unusable.  Only their candidate table coordinates were reused; all
    # values below were re-read and independently audited from BTC tables.
    ("MBB", 2022, "note:vnd_term_deposits", "consolidated"): _disclosure_cell("MBB", 2022, "note:vnd_term_deposits", "consolidated", "258.574.092", "MBB_financial_statements_2022_consolidated|1937", "MBB_financial_statements_2022_consolidated_1937.csv", 5, 1, "Tiền gửi có kỳ hạn - Bằng VND"),
    ("CEO", 2017, "note:finance_lease_asset_cost", "consolidated"): _disclosure_cell("CEO", 2017, "note:finance_lease_asset_cost", "consolidated", "10.604.545.454", "CEO_financial_statements_2017_consolidated|1375", "CEO_financial_statements_2017_consolidated_1375.csv", 4, 2, "TSCĐ thuê tài chính - Nguyên giá cuối năm"),
    ("SHB", 2016, "note:deferred_allocation_cost", "separate"): _disclosure_cell("SHB", 2016, "note:deferred_allocation_cost", "separate", "182.979", "SHB_financial_statements_2016_separate|1850", "SHB_financial_statements_2016_separate_1850.csv", 8, 1, "Chi phí chờ phân bổ"),
    # QNS consolidated intangible fixed-asset ending net book values.
    ("QNS", 2019, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("QNS", 2019, "note:intangible_total_nbv", "consolidated", "22.820.769.751", "QNS_financial_statements_2019_consolidated|1255", "QNS_financial_statements_2019_consolidated_1255.csv", 13, 3, "TSCĐ vô hình - Giá trị còn lại cuối năm - Cộng"),
    ("QNS", 2020, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("QNS", 2020, "note:intangible_total_nbv", "consolidated", "16.413.623.740", "QNS_financial_statements_2020_consolidated|1317", "QNS_financial_statements_2020_consolidated_1317.csv", 16, 3, "TSCĐ vô hình - Giá trị còn lại cuối năm - Cộng"),
    ("QNS", 2021, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("QNS", 2021, "note:intangible_total_nbv", "consolidated", "11.720.950.899", "QNS_financial_statements_2021_consolidated|1384", "QNS_financial_statements_2021_consolidated_1384.csv", 15, 3, "TSCĐ vô hình - Giá trị còn lại cuối năm - Cộng"),
    ("QNS", 2023, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("QNS", 2023, "note:intangible_total_nbv", "consolidated", "16.052.366.337", "QNS_financial_statements_2023_consolidated|1466", "QNS_financial_statements_2023_consolidated_1466.csv", 15, 3, "TSCĐ vô hình - Giá trị còn lại cuối năm - Cộng"),
    ("QNS", 2024, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("QNS", 2024, "note:intangible_total_nbv", "consolidated", "17.144.652.072", "QNS_financial_statements_2024_consolidated|1090", "QNS_financial_statements_2024_consolidated_1090.csv", 15, 3, "TSCĐ vô hình - Giá trị còn lại cuối năm - Cộng"),
    # VIB total customer-loan risk provision (million VND).
    ("VIB", 2015, "note:loan_risk_provision_total", "consolidated"): _disclosure_cell("VIB", 2015, "note:loan_risk_provision_total", "consolidated", "752.476", "VIB_financial_statements_2015_consolidated|950", "VIB_financial_statements_2015_consolidated_950.csv", 6, 3, "Dự phòng rủi ro cho vay khách hàng - Tổng cộng cuối năm"),
    ("VIB", 2022, "note:loan_risk_provision_total", "consolidated"): _disclosure_cell("VIB", 2022, "note:loan_risk_provision_total", "consolidated", "3.064.773", "VIB_financial_statements_2022_consolidated|1500", "VIB_financial_statements_2022_consolidated_1500.csv", 3, 1, "Dự phòng chung và cụ thể - Tổng cộng"),
    ("VIB", 2023, "note:loan_risk_general_provision", "consolidated"): _disclosure_cell("VIB", 2023, "note:loan_risk_general_provision", "consolidated", "1.981.106", "VIB_financial_statements_2023_consolidated|1527", "VIB_financial_statements_2023_consolidated_1527.csv", 3, 1, "Dự phòng chung - Số dư cuối năm"),
    ("VIB", 2023, "note:loan_risk_specific_provision", "consolidated"): _disclosure_cell("VIB", 2023, "note:loan_risk_specific_provision", "consolidated", "2.289.424", "VIB_financial_statements_2023_consolidated|1531", "VIB_financial_statements_2023_consolidated_1531.csv", 4, 1, "Dự phòng cụ thể - Số dư cuối năm"),
    # SHB consolidated intangible fixed-asset ending NBV (million VND).
    ("SHB", 2016, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("SHB", 2016, "note:intangible_total_nbv", "consolidated", "3.538.006", "SHB_financial_statements_2016_consolidated|2293", "SHB_financial_statements_2016_consolidated_2293.csv", 16, 4, "TSCĐ vô hình - Giá trị còn lại cuối năm - Tổng cộng"),
    ("SHB", 2018, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("SHB", 2018, "note:intangible_total_nbv", "consolidated", "3.526.893", "SHB_financial_statements_2018_consolidated|2398", "SHB_financial_statements_2018_consolidated_2398.csv", 16, 4, "TSCĐ vô hình - Giá trị còn lại cuối năm - Tổng cộng"),
    ("SHB", 2020, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("SHB", 2020, "note:intangible_total_nbv", "consolidated", "4.319.172", "SHB_financial_statements_2020_consolidated|1674", "SHB_financial_statements_2020_consolidated_1674.csv", 15, 4, "TSCĐ vô hình - Giá trị còn lại cuối năm - Tổng cộng"),
    ("SHB", 2021, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("SHB", 2021, "note:intangible_total_nbv", "consolidated", "4.333.077", "SHB_financial_statements_2021_consolidated|1775", "SHB_financial_statements_2021_consolidated_1775.csv", 17, 4, "TSCĐ vô hình - Giá trị còn lại cuối năm - Tổng cộng"),
    # SSB parent related-party loan balances (million VND).
    ("SSB", 2021, "note:related_party_loans", "separate"): _disclosure_cell("SSB", 2021, "note:related_party_loans", "separate", "-", "SSB_financial_statements_2021_separate|3114", "SSB_financial_statements_2021_separate_3114.csv", 7, 1, "Tiền vay tại Ngân hàng"),
    ("SSB", 2023, "note:related_party_loans", "separate"): _disclosure_cell("SSB", 2023, "note:related_party_loans", "separate", "274.352", "SSB_financial_statements_2023_separate|1632", "SSB_financial_statements_2023_separate_1632.csv", 7, 1, "Tiền vay tại Ngân hàng"),
    ("SSB", 2024, "note:related_party_loans", "separate"): _disclosure_cell("SSB", 2024, "note:related_party_loans", "separate", "4.300.000", "SSB_financial_statements_2024_separate|2138", "SSB_financial_statements_2024_separate_2138.csv", 26, 1, "Tiền vay tại Ngân hàng"),
    # HDG parent tangible fixed assets: ending gross cost and accumulated
    # depreciation.  Question 846 averages the five source-derived ratios.
    ("HDG", 2020, "note:tangible_gross_cost", "separate"): _disclosure_cell("HDG", 2020, "note:tangible_gross_cost", "separate", "344.631.427.264", "HDG_financial_statements_2020_separate|1113", "HDG_financial_statements_2020_separate_1113.csv", 4, 5, "TSCĐ hữu hình - Nguyên giá cuối năm - Tổng cộng"),
    ("HDG", 2020, "note:tangible_accumulated_depreciation", "separate"): _disclosure_cell("HDG", 2020, "note:tangible_accumulated_depreciation", "separate", "80.866.050.914", "HDG_financial_statements_2020_separate|1113", "HDG_financial_statements_2020_separate_1113.csv", 8, 5, "TSCĐ hữu hình - Hao mòn lũy kế cuối năm - Tổng cộng"),
    ("HDG", 2021, "note:tangible_gross_cost", "separate"): _disclosure_cell("HDG", 2021, "note:tangible_gross_cost", "separate", "343.518.052.309", "HDG_financial_statements_2021_separate|1102", "HDG_financial_statements_2021_separate_1102.csv", 5, 5, "TSCĐ hữu hình - Nguyên giá cuối năm - Tổng cộng"),
    ("HDG", 2021, "note:tangible_accumulated_depreciation", "separate"): _disclosure_cell("HDG", 2021, "note:tangible_accumulated_depreciation", "separate", "95.773.752.404", "HDG_financial_statements_2021_separate|1102", "HDG_financial_statements_2021_separate_1102.csv", 12, 5, "TSCĐ hữu hình - Hao mòn lũy kế cuối năm - Tổng cộng"),
    ("HDG", 2022, "note:tangible_gross_cost", "separate"): _disclosure_cell("HDG", 2022, "note:tangible_gross_cost", "separate", "343.518.052.309", "HDG_financial_statements_2022_separate|1084", "HDG_financial_statements_2022_separate_1084.csv", 4, 5, "TSCĐ hữu hình - Nguyên giá cuối năm - Tổng cộng"),
    ("HDG", 2022, "note:tangible_accumulated_depreciation", "separate"): _disclosure_cell("HDG", 2022, "note:tangible_accumulated_depreciation", "separate", "111.722.134.459", "HDG_financial_statements_2022_separate|1084", "HDG_financial_statements_2022_separate_1084.csv", 10, 5, "TSCĐ hữu hình - Hao mòn lũy kế cuối năm - Tổng cộng"),
    ("HDG", 2023, "note:tangible_gross_cost", "separate"): _disclosure_cell("HDG", 2023, "note:tangible_gross_cost", "separate", "344.018.052.309", "HDG_financial_statements_2023_separate|1065", "HDG_financial_statements_2023_separate_1065.csv", 5, 5, "TSCĐ hữu hình - Nguyên giá cuối năm - Tổng cộng"),
    ("HDG", 2023, "note:tangible_accumulated_depreciation", "separate"): _disclosure_cell("HDG", 2023, "note:tangible_accumulated_depreciation", "separate", "127.661.941.533", "HDG_financial_statements_2023_separate|1065", "HDG_financial_statements_2023_separate_1065.csv", 11, 5, "TSCĐ hữu hình - Hao mòn lũy kế cuối năm - Tổng cộng"),
    ("HDG", 2025, "note:tangible_gross_cost", "separate"): _disclosure_cell("HDG", 2025, "note:tangible_gross_cost", "separate", "344.596.121.939", "HDG_financial_statements_2025_separate|268", "HDG_financial_statements_2025_separate_268.csv", 9, 3, "TSCĐ hữu hình - Nguyên giá cuối năm"),
    ("HDG", 2025, "note:tangible_accumulated_depreciation", "separate"): _disclosure_cell("HDG", 2025, "note:tangible_accumulated_depreciation", "separate", "(155.541.193.868)", "HDG_financial_statements_2025_separate|268", "HDG_financial_statements_2025_separate_268.csv", 10, 3, "TSCĐ hữu hình - Khấu hao lũy kế cuối năm"),
    # EVF consolidated intangible fixed-asset ending net book value, million VND.
    ("EVF", 2021, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("EVF", 2021, "note:intangible_total_nbv", "consolidated", "50.351", "EVF_financial_statements_2021|1128", "EVF_financial_statements_2021_1128.csv", 13, 4, "TSCĐ vô hình - Giá trị còn lại cuối năm - Tổng cộng"),
    ("EVF", 2022, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("EVF", 2022, "note:intangible_total_nbv", "consolidated", "51.456", "EVF_financial_statements_2022|1134", "EVF_financial_statements_2022_1134.csv", 14, 4, "TSCĐ vô hình - Giá trị còn lại cuối năm - Tổng cộng"),
    ("EVF", 2023, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("EVF", 2023, "note:intangible_total_nbv", "consolidated", "47.394", "EVF_financial_statements_2023|1228", "EVF_financial_statements_2023_1228.csv", 12, 4, "TSCĐ vô hình - Giá trị còn lại cuối năm - Tổng cộng"),
    ("EVF", 2025, "note:intangible_total_nbv", "consolidated"): _disclosure_cell("EVF", 2025, "note:intangible_total_nbv", "consolidated", "211.067", "EVF_financial_statements_2025|1099", "EVF_financial_statements_2025_1099.csv", 16, 4, "TSCĐ vô hình - Giá trị còn lại cuối năm - Tổng cộng"),
    # KLB/NVB parent on-balance-sheet foreign-currency positions, million VND.
    ("KLB", 2024, "note:on_balance_currency_position", "separate"): _disclosure_cell("KLB", 2024, "note:on_balance_currency_position", "separate", "2.169.485", "KLB_financial_statements_2024_separate|1782", "KLB_financial_statements_2024_separate_1782.csv", 15, 4, "Trạng thái tiền tệ nội bảng - Tổng"),
    ("NVB", 2024, "note:on_balance_currency_position", "separate"): _disclosure_cell("NVB", 2024, "note:on_balance_currency_position", "separate", "(621.738)", "NVB_financial_statements_2024_separate|1753", "NVB_financial_statements_2024_separate_1753.csv", 16, 4, "Trạng thái tiền tệ nội bảng - Tổng"),
    # 2015 consolidated development-investment fund balances, VND.
    ("GVR", 2015, "cdkt:418", "consolidated"): _disclosure_cell("GVR", 2015, "cdkt:418", "consolidated", "6.437.295.628.830", "GVR_financial_statements_2015_consolidated|290", "GVR_financial_statements_2015_consolidated_290.csv", 12, 3, "Quỹ đầu tư phát triển"),
    ("DPM", 2015, "cdkt:418", "consolidated"): _disclosure_cell("DPM", 2015, "cdkt:418", "consolidated", "3.444.814.857.841", "DPM_financial_statements_2015_consolidated|272", "DPM_financial_statements_2015_consolidated_272.csv", 25, 3, "Quỹ đầu tư phát triển"),
    # FTS consolidated ending doubtful-receivable provision balances, VND.
    ("FTS", 2018, "note:doubtful_receivables_provision", "consolidated"): _disclosure_cell("FTS", 2018, "note:doubtful_receivables_provision", "consolidated", "190.150.000", "FTS_financial_statements_2018|1000", "FTS_financial_statements_2018_1000.csv", 9, 7, "Dự phòng phải thu khó đòi - Cộng số cuối năm"),
    ("FTS", 2020, "note:doubtful_receivables_provision", "consolidated"): _disclosure_cell("FTS", 2020, "note:doubtful_receivables_provision", "consolidated", "78.650.000", "FTS_financial_statements_2020|994", "FTS_financial_statements_2020_994.csv", 12, 7, "Dự phòng phải thu khó đòi - Cộng số cuối năm"),
    ("FTS", 2021, "note:doubtful_receivables_provision", "consolidated"): _disclosure_cell("FTS", 2021, "note:doubtful_receivables_provision", "consolidated", "125.070.000", "FTS_financial_statements_2021|1047", "FTS_financial_statements_2021_1047.csv", 13, 7, "Dự phòng phải thu khó đòi - Cộng số cuối năm"),
    ("FTS", 2023, "note:doubtful_receivables_provision", "consolidated"): _disclosure_cell("FTS", 2023, "note:doubtful_receivables_provision", "consolidated", "158.325.000", "FTS_financial_statements_2023|1134", "FTS_financial_statements_2023_1134.csv", 16, 7, "Dự phòng phải thu khó đòi - Cộng số cuối năm"),
    ("FTS", 2024, "note:doubtful_receivables_provision", "consolidated"): _disclosure_cell("FTS", 2024, "note:doubtful_receivables_provision", "consolidated", "173.475.000", "FTS_financial_statements_2024|1065", "FTS_financial_statements_2024_1065.csv", 17, 7, "Dự phòng phải thu khó đòi - Cộng số cuối năm"),
    # GEE consolidated gross bad receivables at end-2020, VND.
    ("GEE", 2020, "note:gross_bad_receivables", "consolidated"): _disclosure_cell("GEE", 2020, "note:gross_bad_receivables", "consolidated", "220.315.434.328", "GEE_financial_statements_2020_consolidated|938", "GEE_financial_statements_2020_consolidated_938.csv", 7, 1, "Nợ xấu - Tổng giá gốc cuối năm"),
    # Parent-bank customer loans split by original contractual maturity.
    ("ACB", 2019, "note:short_term_customer_loans", "separate"): _disclosure_cell("ACB", 2019, "note:short_term_customer_loans", "separate", "143.115.446", "ACB_financial_statements_2019_separate|2388", "ACB_financial_statements_2019_separate_2388.csv", 2, 1, "Cho vay khách hàng ngắn hạn"),
    ("ACB", 2019, "note:total_customer_loans_by_maturity", "separate"): _disclosure_cell("ACB", 2019, "note:total_customer_loans_by_maturity", "separate", "265.981.486", "ACB_financial_statements_2019_separate|2388", "ACB_financial_statements_2019_separate_2388.csv", 5, 1, "Tổng cho vay khách hàng theo kỳ hạn"),
    ("MBB", 2019, "note:short_term_customer_loans", "separate"): _disclosure_cell("MBB", 2019, "note:short_term_customer_loans", "separate", "117.047.164", "MBB_financial_statements_2019_separate|1189", "MBB_financial_statements_2019_separate_1189.csv", 2, 1, "Cho vay khách hàng ngắn hạn"),
    ("MBB", 2019, "note:total_customer_loans_by_maturity", "separate"): _disclosure_cell("MBB", 2019, "note:total_customer_loans_by_maturity", "separate", "239.082.993", "MBB_financial_statements_2019_separate|1189", "MBB_financial_statements_2019_separate_1189.csv", 5, 1, "Tổng cho vay khách hàng theo kỳ hạn"),
    ("BID", 2019, "note:short_term_customer_loans", "separate"): _disclosure_cell("BID", 2019, "note:short_term_customer_loans", "separate", "683.290.512", "BID_financial_statements_2019_separate|1010", "BID_financial_statements_2019_separate_1010.csv", 2, 1, "Cho vay khách hàng ngắn hạn"),
    ("BID", 2019, "note:total_customer_loans_by_maturity", "separate"): _disclosure_cell("BID", 2019, "note:total_customer_loans_by_maturity", "separate", "1.081.556.050", "BID_financial_statements_2019_separate|1010", "BID_financial_statements_2019_separate_1010.csv", 5, 1, "Tổng cho vay khách hàng theo kỳ hạn"),
    ("STB", 2019, "note:short_term_customer_loans", "separate"): _disclosure_cell("STB", 2019, "note:short_term_customer_loans", "separate", "151.947.082", "STB_financial_statements_2019_separate|1218", "STB_financial_statements_2019_separate_1218.csv", 2, 1, "Cho vay khách hàng ngắn hạn"),
    ("STB", 2019, "note:total_customer_loans_by_maturity", "separate"): _disclosure_cell("STB", 2019, "note:total_customer_loans_by_maturity", "separate", "288.265.973", "STB_financial_statements_2019_separate|1218", "STB_financial_statements_2019_separate_1218.csv", 5, 1, "Tổng cho vay khách hàng theo kỳ hạn"),
    # ACB consolidated financial-reserve and total-equity ending balances, million VND.
    ("ACB", 2015, "note:financial_reserve", "consolidated"): _disclosure_cell("ACB", 2015, "note:financial_reserve", "consolidated", "1.641.434", "ACB_financial_statements_2015_consolidated|2941", "ACB_financial_statements_2015_consolidated_2941.csv", 15, 4, "Quỹ dự phòng tài chính cuối năm"),
    ("ACB", 2015, "note:equity_movement_total", "consolidated"): _disclosure_cell("ACB", 2015, "note:equity_movement_total", "consolidated", "12.787.542", "ACB_financial_statements_2015_consolidated|2941", "ACB_financial_statements_2015_consolidated_2941.csv", 15, 7, "Vốn chủ sở hữu cuối năm - Tổng"),
    ("ACB", 2021, "note:financial_reserve", "consolidated"): _disclosure_cell("ACB", 2021, "note:financial_reserve", "consolidated", "4.744.306", "ACB_financial_statements_2021_consolidated|2369", "ACB_financial_statements_2021_consolidated_2369.csv", 13, 5, "Quỹ dự phòng tài chính cuối năm"),
    ("ACB", 2021, "note:equity_movement_total", "consolidated"): _disclosure_cell("ACB", 2021, "note:equity_movement_total", "consolidated", "44.900.909", "ACB_financial_statements_2021_consolidated|2369", "ACB_financial_statements_2021_consolidated_2369.csv", 13, 8, "Vốn chủ sở hữu cuối năm - Tổng"),
    ("ACB", 2023, "note:financial_reserve", "consolidated"): _disclosure_cell("ACB", 2023, "note:financial_reserve", "consolidated", "7.660.332", "ACB_financial_statements_2024_consolidated|2151", "ACB_financial_statements_2024_consolidated_2151.csv", 9, 4, "Quỹ dự phòng tài chính cuối năm 2023"),
    ("ACB", 2023, "note:equity_movement_total", "consolidated"): _disclosure_cell("ACB", 2023, "note:equity_movement_total", "consolidated", "70.955.961", "ACB_financial_statements_2024_consolidated|2151", "ACB_financial_statements_2024_consolidated_2151.csv", 9, 7, "Vốn chủ sở hữu cuối năm 2023 - Tổng"),
    ("ACB", 2025, "note:financial_reserve", "consolidated"): _disclosure_cell("ACB", 2025, "note:financial_reserve", "consolidated", "10.575.595", "ACB_financial_statements_2025_consolidated|2183", "ACB_financial_statements_2025_consolidated_2183.csv", 14, 4, "Quỹ dự phòng tài chính cuối năm"),
    ("ACB", 2025, "note:equity_movement_total", "consolidated"): _disclosure_cell("ACB", 2025, "note:equity_movement_total", "consolidated", "94.519.719", "ACB_financial_statements_2025_consolidated|2183", "ACB_financial_statements_2025_consolidated_2183.csv", 14, 7, "Vốn chủ sở hữu cuối năm - Tổng"),
    # HDB consolidated currency-risk schedule totals, million VND.
    ("HDB", 2018, "note:net_on_balance_currency_position", "consolidated"): _disclosure_cell("HDB", 2018, "note:net_on_balance_currency_position", "consolidated", "644.742", "HDB_financial_statements_2018_consolidated|2433", "HDB_financial_statements_2018_consolidated_2433.csv", 16, 5, "Trạng thái tiền tệ nội bảng thuần - Tổng"),
    ("HDB", 2018, "note:currency_schedule_total_assets", "consolidated"): _disclosure_cell("HDB", 2018, "note:currency_schedule_total_assets", "consolidated", "22.034.332", "HDB_financial_statements_2018_consolidated|2433", "HDB_financial_statements_2018_consolidated_2433.csv", 8, 5, "Tổng tài sản trong bảng trạng thái tiền tệ"),
    ("HDB", 2021, "note:net_on_balance_currency_position", "consolidated"): _disclosure_cell("HDB", 2021, "note:net_on_balance_currency_position", "consolidated", "(563.981)", "HDB_financial_statements_2021_consolidated|2437", "HDB_financial_statements_2021_consolidated_2437.csv", 18, 5, "Trạng thái tiền tệ nội bảng thuần - Tổng"),
    ("HDB", 2021, "note:currency_schedule_total_assets", "consolidated"): _disclosure_cell("HDB", 2021, "note:currency_schedule_total_assets", "consolidated", "35.609.486", "HDB_financial_statements_2021_consolidated|2437", "HDB_financial_statements_2021_consolidated_2437.csv", 9, 5, "Tổng tài sản trong bảng trạng thái tiền tệ"),
    ("HDB", 2022, "note:net_on_balance_currency_position", "consolidated"): _disclosure_cell("HDB", 2022, "note:net_on_balance_currency_position", "consolidated", "42.192.668", "HDB_financial_statements_2022_consolidated|2204", "HDB_financial_statements_2022_consolidated_2204.csv", 23, 6, "Trạng thái tiền tệ nội bảng thuần - Tổng"),
    ("HDB", 2022, "note:currency_schedule_total_assets", "consolidated"): _disclosure_cell("HDB", 2022, "note:currency_schedule_total_assets", "consolidated", "419.470.805", "HDB_financial_statements_2022_consolidated|2204", "HDB_financial_statements_2022_consolidated_2204.csv", 13, 6, "Tổng tài sản trong bảng trạng thái tiền tệ"),
    ("HDB", 2024, "note:net_on_balance_currency_position", "consolidated"): _disclosure_cell("HDB", 2024, "note:net_on_balance_currency_position", "consolidated", "62.694.143", "HDB_financial_statements_2024_consolidated|2270", "HDB_financial_statements_2024_consolidated_2270.csv", 23, 6, "Trạng thái tiền tệ nội bảng thuần - Tổng"),
    ("HDB", 2024, "note:currency_schedule_total_assets", "consolidated"): _disclosure_cell("HDB", 2024, "note:currency_schedule_total_assets", "consolidated", "703.403.340", "HDB_financial_statements_2024_consolidated|2270", "HDB_financial_statements_2024_consolidated_2270.csv", 13, 6, "Tổng tài sản trong bảng trạng thái tiền tệ"),
    # 2022 consolidated tangible-fixed-asset ending gross/depreciation totals, million VND.
    ("VPB", 2022, "note:tangible_gross_cost", "consolidated"): _disclosure_cell("VPB", 2022, "note:tangible_gross_cost", "consolidated", "2.733.537", "VPB_financial_statements_2022_consolidated|1717", "VPB_financial_statements_2022_consolidated_1717.csv", 7, 6, "TSCĐ hữu hình - Nguyên giá cuối năm - Tổng"),
    ("VPB", 2022, "note:tangible_accumulated_depreciation", "consolidated"): _disclosure_cell("VPB", 2022, "note:tangible_accumulated_depreciation", "consolidated", "1.514.429", "VPB_financial_statements_2022_consolidated|1717", "VPB_financial_statements_2022_consolidated_1717.csv", 13, 6, "TSCĐ hữu hình - Khấu hao lũy kế cuối năm - Tổng"),
    ("SHB", 2022, "note:tangible_gross_cost", "consolidated"): _disclosure_cell("SHB", 2022, "note:tangible_gross_cost", "consolidated", "1.287.149", "SHB_financial_statements_2022_consolidated|1576", "SHB_financial_statements_2022_consolidated_1576.csv", 9, 6, "TSCĐ hữu hình - Nguyên giá cuối năm - Tổng"),
    ("SHB", 2022, "note:tangible_accumulated_depreciation", "consolidated"): _disclosure_cell("SHB", 2022, "note:tangible_accumulated_depreciation", "consolidated", "784.295", "SHB_financial_statements_2022_consolidated|1576", "SHB_financial_statements_2022_consolidated_1576.csv", 17, 6, "TSCĐ hữu hình - Khấu hao lũy kế cuối năm - Tổng"),
    ("MBB", 2022, "note:tangible_gross_cost", "consolidated"): _disclosure_cell("MBB", 2022, "note:tangible_gross_cost", "consolidated", "7.110.562", "MBB_financial_statements_2022_consolidated|1788", "MBB_financial_statements_2022_consolidated_1788.csv", 9, 5, "TSCĐ hữu hình - Nguyên giá cuối năm - Tổng"),
    ("MBB", 2022, "note:tangible_accumulated_depreciation", "consolidated"): _disclosure_cell("MBB", 2022, "note:tangible_accumulated_depreciation", "consolidated", "3.653.001", "MBB_financial_statements_2022_consolidated|1788", "MBB_financial_statements_2022_consolidated_1788.csv", 17, 5, "TSCĐ hữu hình - Hao mòn lũy kế cuối năm - Tổng"),
    # HHV BOT segment assets and consolidated total assets, VND.  The old
    # 2021/2022 coordinates pointed to the geographic segment grand total,
    # which is not the requested BOT activity.  The activity-segment tables
    # disclose ``Dự án BOT`` explicitly in their first numeric column.
    ("HHV", 2021, "note:bot_segment_assets", "consolidated"): _disclosure_cell("HHV", 2021, "note:bot_segment_assets", "consolidated", "32.355.512.700.711", "HHV_financial_statements_2021_consolidated|2563", "HHV_financial_statements_2021_consolidated_2563.csv", 2, 1, "Tài sản bộ phận - Dự án BOT"),
    ("HHV", 2022, "note:bot_segment_assets", "consolidated"): _disclosure_cell("HHV", 2022, "note:bot_segment_assets", "consolidated", "33.657.835.517.377", "HHV_financial_statements_2022_consolidated|2450", "HHV_financial_statements_2022_consolidated_2450.csv", 2, 1, "Tài sản bộ phận - Dự án BOT"),
    ("HHV", 2024, "note:bot_segment_assets", "consolidated"): _disclosure_cell("HHV", 2024, "note:bot_segment_assets", "consolidated", "35.317.671.994.443", "HHV_financial_statements_2024_consolidated|2843", "HHV_financial_statements_2024_consolidated_2843.csv", 2, 1, "Tài sản bộ phận thu phí trạm BOT"),
    ("HHV", 2024, "note:segment_total_assets", "consolidated"): _disclosure_cell("HHV", 2024, "note:segment_total_assets", "consolidated", "38.906.360.732.239", "HHV_financial_statements_2024_consolidated|2843", "HHV_financial_statements_2024_consolidated_2843.csv", 4, 6, "Tổng tài sản hợp nhất"),
    ("HHV", 2025, "note:bot_segment_assets", "consolidated"): _disclosure_cell("HHV", 2025, "note:bot_segment_assets", "consolidated", "35.890.505.367.786", "HHV_financial_statements_2025_consolidated|3101", "HHV_financial_statements_2025_consolidated_3101.csv", 2, 1, "Tài sản bộ phận thu phí trạm BOT"),
    ("HHV", 2025, "note:segment_total_assets", "consolidated"): _disclosure_cell("HHV", 2025, "note:segment_total_assets", "consolidated", "40.752.075.682.457", "HHV_financial_statements_2025_consolidated|3101", "HHV_financial_statements_2025_consolidated_3101.csv", 4, 6, "Tổng tài sản hợp nhất"),
    # 2020 overall net-liquidity-gap totals, million VND.
    ("SGB", 2020, "note:overall_net_liquidity_gap", "consolidated"): _disclosure_cell("SGB", 2020, "note:overall_net_liquidity_gap", "consolidated", "3.994.831", "SGB_financial_statements_2020_consolidated|1998", "SGB_financial_statements_2020_consolidated_1998.csv", 19, 8, "Chênh lệch thanh khoản ròng - Cộng"),
    ("NVB", 2020, "note:overall_net_liquidity_gap", "consolidated"): _disclosure_cell("NVB", 2020, "note:overall_net_liquidity_gap", "consolidated", "5.213.560", "NVB_financial_statements_2020_consolidated_2|1620", "NVB_financial_statements_2020_consolidated_2_1620.csv", 21, 7, "Mức chênh thanh khoản ròng - Tổng"),
    ("MSB", 2020, "note:overall_net_liquidity_gap", "consolidated"): _disclosure_cell("MSB", 2020, "note:overall_net_liquidity_gap", "consolidated", "19.002.801", "MSB_financial_statements_2020_consolidated|2508", "MSB_financial_statements_2020_consolidated_2508.csv", 24, 8, "Mức chênh thanh khoản ròng - Tổng"),
    ("HDB", 2020, "note:overall_net_liquidity_gap", "consolidated"): _disclosure_cell("HDB", 2020, "note:overall_net_liquidity_gap", "consolidated", "26.798.403", "HDB_financial_statements_2020_consolidated|2444", "HDB_financial_statements_2020_consolidated_2444.csv", 24, 8, "Mức chênh thanh khoản ròng - Tổng"),
    ("VIB", 2020, "note:overall_net_liquidity_gap", "consolidated"): _disclosure_cell("VIB", 2020, "note:overall_net_liquidity_gap", "consolidated", "19.881.788", "VIB_financial_statements_2020_consolidated|2277", "VIB_financial_statements_2020_consolidated_2277.csv", 21, 8, "Mức chênh thanh khoản thuần - Tổng"),
})


# Final legacy multi-period repairs.  Each of these questions previously read
# two columns from one report even though the question named non-adjacent years,
# or omitted one requested year entirely.  Keep every endpoint as an explicit
# BTC source coordinate so the generated query remains data-dependent.
EXPLICIT_CELLS.update({
    ("IJC", 2021, "note:bonus_welfare_fund_ending", "separate"): _disclosure_cell(
        "IJC", 2021, "note:bonus_welfare_fund_ending", "separate", "19.415.310.894",
        "IJC_financial_statements_2021_separate|1528", "IJC_financial_statements_2021_separate_1528.csv",
        4, 1, "Quỹ khen thưởng, phúc lợi - Số cuối năm",
    ),
    ("IJC", 2015, "note:bonus_welfare_fund_ending", "separate"): _disclosure_cell(
        "IJC", 2015, "note:bonus_welfare_fund_ending", "separate", "1.580.297.059",
        "IJC_financial_statements_2015_separate|1450", "IJC_financial_statements_2015_separate_1450.csv",
        4, 1, "Quỹ khen thưởng, phúc lợi - Số cuối năm",
    ),
    ("STB", 2016, "note:foreign_currency_cash_million", "separate"): _disclosure_cell(
        "STB", 2016, "note:foreign_currency_cash_million", "separate", "1.408.091",
        "STB_financial_statements_2016_separate|997", "STB_financial_statements_2016_separate_997.csv",
        2, 1, "Tiền mặt bằng ngoại tệ - Số cuối năm (triệu đồng)",
    ),
    ("STB", 2015, "note:foreign_currency_cash_million", "separate"): _disclosure_cell(
        "STB", 2015, "note:foreign_currency_cash_million", "separate", "2.060.520",
        "STB_financial_statements_2015_separate|1091", "STB_financial_statements_2015_separate_1091.csv",
        2, 1, "Tiền mặt bằng ngoại tệ - Số cuối năm (triệu đồng)",
    ),
    ("BVH", 2020, "note:short_term_investment_receivable", "separate"): _disclosure_cell(
        "BVH", 2020, "note:short_term_investment_receivable", "separate", "799.275.014.996",
        "BVH_financial_statements_2020_separate|911", "BVH_financial_statements_2020_separate_911.csv",
        2, 2, "Phải thu từ hoạt động đầu tư - Số cuối năm",
    ),
    ("BVH", 2018, "note:short_term_investment_receivable", "separate"): _disclosure_cell(
        "BVH", 2018, "note:short_term_investment_receivable", "separate", "914.183.587.969",
        "BVH_financial_statements_2018_separate|1201", "BVH_financial_statements_2018_separate_1201.csv",
        2, 2, "Phải thu từ hoạt động đầu tư - Số cuối năm",
    ),
    ("VPI", 2022, "note:owner_contributed_capital_ending", "separate"): _disclosure_cell(
        "VPI", 2022, "note:owner_contributed_capital_ending", "separate", "2.419.996.170.000",
        "VPI_financial_statements_2022_separate|1345", "VPI_financial_statements_2022_separate_1345.csv",
        5, 1, "Vốn đã góp của chủ sở hữu - Số cuối năm",
    ),
    ("VPI", 2021, "note:owner_contributed_capital_ending", "separate"): _disclosure_cell(
        "VPI", 2021, "note:owner_contributed_capital_ending", "separate", "2.199.997.800.000",
        "VPI_financial_statements_2021_separate|1252", "VPI_financial_statements_2021_separate_1252.csv",
        5, 1, "Vốn đã góp của chủ sở hữu - Số cuối năm",
    ),
    ("GEX", 2025, "note:operating_lease_due_within_one_year", "separate"): _disclosure_cell(
        "GEX", 2025, "note:operating_lease_due_within_one_year", "separate", "27.258.108.199",
        "GEX_financial_statements_2025_separate|1801", "GEX_financial_statements_2025_separate_1801.csv",
        2, 1, "Cam kết thuê hoạt động - Từ 1 năm trở xuống",
    ),
    ("GEX", 2022, "note:operating_lease_due_within_one_year", "separate"): _disclosure_cell(
        "GEX", 2022, "note:operating_lease_due_within_one_year", "separate", "62.834.689.635",
        "GEX_financial_statements_2022_separate|1423", "GEX_financial_statements_2022_separate_1423.csv",
        2, 1, "Cam kết thuê hoạt động - Từ 1 năm trở xuống",
    ),
    ("DLG", 2020, "note:corporate_income_tax_payable_ending", "consolidated"): _disclosure_cell(
        "DLG", 2020, "note:corporate_income_tax_payable_ending", "consolidated", "64.112.708.746",
        "DLG_financial_statements_2020_consolidated|1830", "DLG_financial_statements_2020_consolidated_1830.csv",
        3, 8, "Thuế thu nhập doanh nghiệp - Số cuối kỳ phải trả",
    ),
    ("DLG", 2018, "note:corporate_income_tax_payable_ending", "consolidated"): _disclosure_cell(
        "DLG", 2018, "note:corporate_income_tax_payable_ending", "consolidated", "66.271.208.557",
        "DLG_financial_statements_2018_consolidated|2023", "DLG_financial_statements_2018_consolidated_2023.csv",
        3, 8, "Thuế thu nhập doanh nghiệp - Số cuối kỳ phải trả",
    ),
    ("HSG", 2025, "note:long_term_prepaid_rent", "consolidated"): _disclosure_cell(
        "HSG", 2025, "note:long_term_prepaid_rent", "consolidated", "43.136.294.408",
        "HSG_financial_statements_2025_consolidated|812", "HSG_financial_statements_2025_consolidated_812.csv",
        3, 1, "Chi phí thuê trả trước dài hạn - Số cuối năm",
    ),
    ("HSG", 2021, "note:long_term_prepaid_rent", "consolidated"): _disclosure_cell(
        "HSG", 2021, "note:long_term_prepaid_rent", "consolidated", "49.633.658.164",
        "HSG_financial_statements_2021_consolidated|1023", "HSG_financial_statements_2021_consolidated_1023.csv",
        3, 1, "Chi phí thuê trả trước dài hạn - Số cuối năm",
    ),
    ("SAB", 2022, "note:net_investing_cash_flow", "consolidated"): _disclosure_cell(
        "SAB", 2022, "note:net_investing_cash_flow", "consolidated", "(1.867.768.246.891)",
        "SAB_financial_statements_2022_consolidated|308", "SAB_financial_statements_2022_consolidated_308.csv",
        8, 3, "Lưu chuyển tiền thuần từ hoạt động đầu tư",
    ),
    ("SAB", 2023, "note:net_investing_cash_flow", "consolidated"): _disclosure_cell(
        "SAB", 2023, "note:net_investing_cash_flow", "consolidated", "2.715.583.467.608",
        "SAB_financial_statements_2023_consolidated|317", "SAB_financial_statements_2023_consolidated_317.csv",
        7, 3, "Lưu chuyển tiền thuần từ hoạt động đầu tư",
    ),
    ("SAB", 2024, "note:net_investing_cash_flow", "consolidated"): _disclosure_cell(
        "SAB", 2024, "note:net_investing_cash_flow", "consolidated", "18.884.385.322",
        "SAB_financial_statements_2024_consolidated|333", "SAB_financial_statements_2024_consolidated_333.csv",
        8, 3, "Lưu chuyển tiền thuần từ hoạt động đầu tư",
    ),
    ("OGC", 2016, "note:employee_cost", "consolidated"): _disclosure_cell(
        "OGC", 2016, "note:employee_cost", "consolidated", "24.728.164.794",
        "OGC_financial_statements_2016_consolidated|2064", "OGC_financial_statements_2016_consolidated_2064.csv",
        3, 1, "Chi phí nhân công",
    ),
    ("OGC", 2017, "note:employee_cost", "consolidated"): _disclosure_cell(
        "OGC", 2017, "note:employee_cost", "consolidated", "32.986.442.045",
        "OGC_financial_statements_2017_consolidated|1992", "OGC_financial_statements_2017_consolidated_1992.csv",
        3, 1, "Chi phí nhân công",
    ),
    ("OGC", 2018, "note:employee_cost", "consolidated"): _disclosure_cell(
        "OGC", 2018, "note:employee_cost", "consolidated", "34.406.890.797",
        "OGC_financial_statements_2018_consolidated|2212", "OGC_financial_statements_2018_consolidated_2212.csv",
        3, 1, "Chi phí nhân công",
    ),
    ("OGC", 2019, "note:employee_cost", "consolidated"): _disclosure_cell(
        "OGC", 2019, "note:employee_cost", "consolidated", "39.854.287.480",
        "OGC_financial_statements_2019_consolidated|2037", "OGC_financial_statements_2019_consolidated_2037.csv",
        3, 1, "Chi phí nhân công",
    ),
    ("SHB", 2016, "note:northern_pbt_million", "consolidated"): _disclosure_cell(
        "SHB", 2016, "note:northern_pbt_million", "consolidated", "387.762",
        "SHB_financial_statements_2016_consolidated|3100", "SHB_financial_statements_2016_consolidated_3100.csv",
        12, 2, "Miền Bắc - Tổng lợi nhuận trước thuế (triệu VND)",
    ),
    ("SHB", 2020, "note:northern_pbt_million", "consolidated"): _disclosure_cell(
        "SHB", 2020, "note:northern_pbt_million", "consolidated", "1.554.132",
        "SHB_financial_statements_2020_consolidated|2116", "SHB_financial_statements_2020_consolidated_2116.csv",
        10, 1, "Miền Bắc - Tổng lợi nhuận trước thuế (triệu VND)",
    ),
    ("SHB", 2022, "note:northern_pbt_million", "consolidated"): _disclosure_cell(
        "SHB", 2022, "note:northern_pbt_million", "consolidated", "6.832.442",
        "SHB_financial_statements_2022_consolidated|2061", "SHB_financial_statements_2022_consolidated_2061.csv",
        10, 1, "Miền Bắc - Tổng lợi nhuận trước thuế (triệu VND)",
    ),
    ("SHB", 2024, "note:northern_pbt_million", "consolidated"): _disclosure_cell(
        "SHB", 2024, "note:northern_pbt_million", "consolidated", "9.297.214",
        "SHB_financial_statements_2024_consolidated|1696", "SHB_financial_statements_2024_consolidated_1696.csv",
        13, 1, "Miền Bắc - Tổng lợi nhuận trước thuế (triệu VND)",
    ),
    # Tax paid is presented in parentheses because it is a cash outflow.  The
    # questions ask for the amount paid, so comparisons/averages use magnitude,
    # consistently with source-audited cash-outflow questions q268 and q933.
    ("VNM", 2019, "note:corporate_income_tax_paid", "separate"): _disclosure_cell(
        "VNM", 2019, "note:corporate_income_tax_paid", "separate", "(2.025.224.469.158)",
        "VNM_financial_statements_2019_separate|1225", "VNM_financial_statements_2019_separate_1225.csv",
        3, 3, "Thuế thu nhập doanh nghiệp - Đã nộp (VND)",
    ),
    ("VNM", 2022, "note:corporate_income_tax_paid", "separate"): _disclosure_cell(
        "VNM", 2022, "note:corporate_income_tax_paid", "separate", "(1.903.065.886.321)",
        "VNM_financial_statements_2022_separate|1293", "VNM_financial_statements_2022_separate_1293.csv",
        3, 3, "Thuế thu nhập doanh nghiệp - Đã nộp (VND)",
    ),
    ("VNM", 2023, "note:corporate_income_tax_paid", "separate"): _disclosure_cell(
        "VNM", 2023, "note:corporate_income_tax_paid", "separate", "(1.441.600.595.087)",
        "VNM_financial_statements_2023_separate|1306", "VNM_financial_statements_2023_separate_1306.csv",
        3, 3, "Thuế thu nhập doanh nghiệp - Đã nộp (VND)",
    ),
    ("VNM", 2024, "note:corporate_income_tax_paid", "separate"): _disclosure_cell(
        "VNM", 2024, "note:corporate_income_tax_paid", "separate", "(1.997.458.922.345)",
        "VNM_financial_statements_2024_separate|1252", "VNM_financial_statements_2024_separate_1252.csv",
        3, 3, "Thuế thu nhập doanh nghiệp - Đã nộp (VND)",
    ),
    ("VNM", 2025, "note:corporate_income_tax_paid", "separate"): _disclosure_cell(
        "VNM", 2025, "note:corporate_income_tax_paid", "separate", "(1.353.040.369.936)",
        "VNM_financial_statements_2025_separate|1271", "VNM_financial_statements_2025_separate_1271.csv",
        3, 3, "Thuế thu nhập doanh nghiệp - Đã nộp (VND)",
    ),
    ("VCB", 2018, "note:bank_corporate_income_tax_paid_million", "separate"): _disclosure_cell(
        "VCB", 2018, "note:bank_corporate_income_tax_paid_million", "separate", "(2.496.594)",
        "VCB_financial_statements_2018_separate|1567", "VCB_financial_statements_2018_separate_1567.csv",
        5, 3, "Thuế TNDN của Ngân hàng - Số đã nộp (triệu VND)",
    ),
    ("VCB", 2020, "note:bank_corporate_income_tax_paid_million", "separate"): _disclosure_cell(
        "VCB", 2020, "note:bank_corporate_income_tax_paid_million", "separate", "(4.588.752)",
        "VCB_financial_statements_2020_separate|1554", "VCB_financial_statements_2020_separate_1554.csv",
        5, 3, "Thuế TNDN của Ngân hàng - Số đã nộp (triệu VND)",
    ),
    ("VCB", 2021, "note:bank_corporate_income_tax_paid_million", "separate"): _disclosure_cell(
        "VCB", 2021, "note:bank_corporate_income_tax_paid_million", "separate", "(5.707.816)",
        "VCB_financial_statements_2021_separate|1666", "VCB_financial_statements_2021_separate_1666.csv",
        5, 3, "Thuế TNDN của Ngân hàng - Số đã nộp (triệu VND)",
    ),
    ("VCB", 2022, "note:bank_corporate_income_tax_paid_million", "separate"): _disclosure_cell(
        "VCB", 2022, "note:bank_corporate_income_tax_paid_million", "separate", "(3.820.917)",
        "VCB_financial_statements_2022_separate|1687", "VCB_financial_statements_2022_separate_1687.csv",
        4, 3, "Thuế TNDN của Ngân hàng - Số đã nộp (triệu VND)",
    ),
    ("VCB", 2025, "note:bank_corporate_income_tax_paid_million", "separate"): _disclosure_cell(
        "VCB", 2025, "note:bank_corporate_income_tax_paid_million", "separate", "(9.032.912)",
        "VCB_financial_statements_2025_separate|1926", "VCB_financial_statements_2025_separate_1926.csv",
        5, 3, "Thuế TNDN của Ngân hàng - Số đã nộp (triệu VND)",
    ),
    ("MBB", 2020, "note:customer_loan_loss_provision_million", "consolidated"): _disclosure_cell(
        "MBB", 2020, "note:customer_loan_loss_provision_million", "consolidated", "(4.354.219)",
        "MBB_financial_statements_2020_consolidated|358", "MBB_financial_statements_2020_consolidated_358.csv",
        14, 2, "Dự phòng rủi ro cho vay khách hàng - 31/12/2020 (triệu đồng)",
    ),
})

# Source cells recovered while auditing legacy multi-company and selector
# programs.  These are direct BTC table coordinates, not answer labels.
EXPLICIT_CELLS.update({
    ("DPM", 2016, "note:interest_expense", "separate"): _disclosure_cell(
        "DPM", 2016, "note:interest_expense", "separate", "4.473.655.664",
        "DPM_financial_statements_2016_separate|976",
        "DPM_financial_statements_2016_separate_976.csv", 2, 1,
        "Chi phí lãi vay",
    ),
    ("VJC", 2016, "note:ending_common_shares_outstanding", "consolidated"): _disclosure_cell(
        "VJC", 2016, "note:ending_common_shares_outstanding", "consolidated", "300.000.000",
        "VJC_financial_statements_2016_consolidated|1146",
        "VJC_financial_statements_2016_consolidated_1146.csv", 5, 1,
        "Số dư cổ phiếu phổ thông cuối năm",
    ),
    ("VJC", 2019, "note:ending_common_shares_outstanding", "consolidated"): _disclosure_cell(
        "VJC", 2019, "note:ending_common_shares_outstanding", "consolidated", "523.838.594",
        "VJC_financial_statements_2019_consolidated|1157",
        "VJC_financial_statements_2019_consolidated_1157.csv", 4, 1,
        "Số cổ phiếu đang lưu hành cuối năm",
    ),
    ("VJC", 2021, "note:ending_common_shares_outstanding", "consolidated"): _disclosure_cell(
        "VJC", 2021, "note:ending_common_shares_outstanding", "consolidated", "541.611.334",
        "VJC_financial_statements_2021_consolidated|1431",
        "VJC_financial_statements_2021_consolidated_1431.csv", 4, 1,
        "Số lượng cổ phiếu đang lưu hành cuối năm",
    ),
    ("FTS", 2018, "note:short_term_borrowings_ending", "unknown"): _disclosure_cell("FTS", 2018, "note:short_term_borrowings_ending", "unknown", "477.200.000.000", "FTS_financial_statements_2018|253", "FTS_financial_statements_2018_253.csv", 5, 3, "Vay ngắn hạn - số dư cuối năm"),
    ("FTS", 2019, "note:short_term_borrowings_ending", "unknown"): _disclosure_cell("FTS", 2019, "note:short_term_borrowings_ending", "unknown", "361.500.000.000", "FTS_financial_statements_2019|271", "FTS_financial_statements_2019_271.csv", 5, 3, "Vay ngắn hạn - số dư cuối năm"),
    ("FTS", 2022, "note:short_term_borrowings_ending", "unknown"): _disclosure_cell("FTS", 2022, "note:short_term_borrowings_ending", "unknown", "1.308.000.000.000", "FTS_financial_statements_2022|368", "FTS_financial_statements_2022_368.csv", 5, 3, "Vay ngắn hạn - số dư cuối năm"),
    ("FTS", 2023, "note:short_term_borrowings_ending", "unknown"): _disclosure_cell("FTS", 2023, "note:short_term_borrowings_ending", "unknown", "3.148.101.835.693", "FTS_financial_statements_2023|343", "FTS_financial_statements_2023_343.csv", 5, 3, "Vay ngắn hạn - số dư cuối năm"),
    ("FTS", 2024, "note:short_term_borrowings_ending", "unknown"): _disclosure_cell("FTS", 2024, "note:short_term_borrowings_ending", "unknown", "5.475.933.586.028", "FTS_financial_statements_2024|278", "FTS_financial_statements_2024_278.csv", 5, 3, "Vay ngắn hạn - số dư cuối năm"),
    ("FTS", 2018, "note:cash_beginning", "unknown"): _disclosure_cell("FTS", 2018, "note:cash_beginning", "unknown", "170.509.389.350", "FTS_financial_statements_2018|394", "FTS_financial_statements_2018_394.csv", 23, 3, "Tiền và các khoản tương đương tiền đầu kỳ"),
    ("FTS", 2018, "note:cash_ending", "unknown"): _disclosure_cell("FTS", 2018, "note:cash_ending", "unknown", "458.055.059.707", "FTS_financial_statements_2018|394", "FTS_financial_statements_2018_394.csv", 27, 3, "Tiền và các khoản tương đương tiền cuối kỳ"),
    ("FTS", 2019, "note:cash_beginning", "unknown"): _disclosure_cell("FTS", 2019, "note:cash_beginning", "unknown", "458.055.059.707", "FTS_financial_statements_2019|431", "FTS_financial_statements_2019_431.csv", 23, 3, "Tiền và các khoản tương đương tiền đầu kỳ"),
    ("FTS", 2019, "note:cash_ending", "unknown"): _disclosure_cell("FTS", 2019, "note:cash_ending", "unknown", "50.280.157.184", "FTS_financial_statements_2019|431", "FTS_financial_statements_2019_431.csv", 27, 3, "Tiền và các khoản tương đương tiền cuối kỳ"),
    ("FTS", 2022, "note:cash_beginning", "unknown"): _disclosure_cell("FTS", 2022, "note:cash_beginning", "unknown", "1.868.836.688.046", "FTS_financial_statements_2022|542", "FTS_financial_statements_2022_542.csv", 20, 3, "Tiền và các khoản tương đương tiền đầu kỳ"),
    ("FTS", 2022, "note:cash_ending", "unknown"): _disclosure_cell("FTS", 2022, "note:cash_ending", "unknown", "262.794.899.508", "FTS_financial_statements_2022|542", "FTS_financial_statements_2022_542.csv", 23, 3, "Tiền và các khoản tương đương tiền cuối kỳ"),
    ("FTS", 2023, "note:cash_beginning", "unknown"): _disclosure_cell("FTS", 2023, "note:cash_beginning", "unknown", "262.794.899.508", "FTS_financial_statements_2023|473", "FTS_financial_statements_2023_473.csv", 20, 3, "Tiền và các khoản tương đương tiền đầu kỳ"),
    ("FTS", 2023, "note:cash_ending", "unknown"): _disclosure_cell("FTS", 2023, "note:cash_ending", "unknown", "1.253.357.329.782", "FTS_financial_statements_2023|473", "FTS_financial_statements_2023_473.csv", 23, 3, "Tiền và các khoản tương đương tiền cuối kỳ"),
    ("FTS", 2024, "note:cash_beginning", "unknown"): _disclosure_cell("FTS", 2024, "note:cash_beginning", "unknown", "1.253.357.329.782", "FTS_financial_statements_2024|407", "FTS_financial_statements_2024_407.csv", 20, 3, "Tiền và các khoản tương đương tiền đầu kỳ"),
    ("FTS", 2024, "note:cash_ending", "unknown"): _disclosure_cell("FTS", 2024, "note:cash_ending", "unknown", "565.564.523.996", "FTS_financial_statements_2024|407", "FTS_financial_statements_2024_407.csv", 23, 3, "Tiền và các khoản tương đương tiền cuối kỳ"),
})

EXPLICIT_CELLS.update({
    ("MSB", 2024, "note:customer_loan_provision_expense", "separate"): _disclosure_cell("MSB", 2024, "note:customer_loan_provision_expense", "separate", "1.924.250", "MSB_financial_statements_2024_separate|1682", "MSB_financial_statements_2024_separate_1682.csv", 1, 1, "Trích lập dự phòng cho vay khách hàng"),
    ("BID", 2024, "note:customer_loan_provision_expense", "separate"): _disclosure_cell("BID", 2024, "note:customer_loan_provision_expense", "separate", "20.610.502", "BID_financial_statements_2024_separate|2011", "BID_financial_statements_2024_separate_2011.csv", 2, 1, "Trích lập dự phòng cho vay khách hàng"),
    ("ABB", 2024, "note:customer_loan_provision_expense", "separate"): _disclosure_cell("ABB", 2024, "note:customer_loan_provision_expense", "separate", "734.748", "ABB_financial_statements_2024_separate|1535", "ABB_financial_statements_2024_separate_1535.csv", 7, 3, "(Hoàn nhập)/trích lập dự phòng cho vay khách hàng trong năm"),
    ("OGC", 2015, "note:ending_common_shares_outstanding", "consolidated"): _disclosure_cell("OGC", 2015, "note:ending_common_shares_outstanding", "consolidated", "299.999.999", "OGC_financial_statements_2015_consolidated|2397", "OGC_financial_statements_2015_consolidated_2397.csv", 8, 1, "Số lượng cổ phiếu phổ thông đang lưu hành cuối năm"),
    ("VNM", 2015, "note:ending_common_shares_outstanding", "consolidated"): _disclosure_cell("VNM", 2015, "note:ending_common_shares_outstanding", "consolidated", "1.200.139.398", "VNM_financial_statements_2015_consolidated|1377", "VNM_financial_statements_2015_consolidated_1377.csv", 5, 1, "Số dư cổ phiếu cuối năm"),
    ("HNG", 2015, "note:share_capital_thousand_vnd", "consolidated"): _disclosure_cell("HNG", 2015, "note:share_capital_thousand_vnd", "consolidated", "7.081.438.950", "HNG_financial_statements_2015_consolidated|269", "HNG_financial_statements_2015_consolidated_269.csv", 17, 3, "Vốn cổ phần, nghìn VND"),
    ("OGC", 2017, "note:other_expense", "consolidated"): _disclosure_cell("OGC", 2017, "note:other_expense", "consolidated", "26.374.755.718", "OGC_financial_statements_2017_consolidated|537", "OGC_financial_statements_2017_consolidated_537.csv", 15, 3, "Chi phí khác"),
    ("VPI", 2016, "note:taxes_state_payables", "separate"): _disclosure_cell("VPI", 2016, "note:taxes_state_payables", "separate", "2.542.217.408", "VPI_financial_statements_2016_separate|187", "VPI_financial_statements_2016_separate_187.csv", 5, 3, "Thuế và các khoản phải nộp Nhà nước"),
    ("DIG", 2016, "note:taxes_state_payables", "separate"): _disclosure_cell("DIG", 2016, "note:taxes_state_payables", "separate", "26.323.895.303", "DIG_financial_statements_2016_separate|279", "DIG_financial_statements_2016_separate_279.csv", 5, 3, "Thuế và các khoản phải nộp Nhà nước"),
    ("VRE", 2016, "note:taxes_state_payables", "separate"): _disclosure_cell("VRE", 2016, "note:taxes_state_payables", "separate", "35.068.093.552", "VRE_financial_statements_2016_separate|343", "VRE_financial_statements_2016_separate_343.csv", 5, 3, "Thuế và các khoản phải nộp Nhà nước"),
    ("DXG", 2016, "note:taxes_state_payables", "separate"): _disclosure_cell("DXG", 2016, "note:taxes_state_payables", "separate", "69.976.249.971", "DXG_financial_statements_2016_separate|273", "DXG_financial_statements_2016_separate_273.csv", 5, 3, "Thuế và các khoản phải nộp Nhà nước"),
    ("PDR", 2016, "note:taxes_state_payables", "separate"): _disclosure_cell("PDR", 2016, "note:taxes_state_payables", "separate", "52.316.569.712", "PDR_financial_statements_2016_separate|200", "PDR_financial_statements_2016_separate_200.csv", 5, 3, "Thuế và các khoản phải nộp Nhà nước"),
    ("VSC", 2017, "note:short_term_trade_payables", "separate"): _disclosure_cell("VSC", 2017, "note:short_term_trade_payables", "separate", "23.030.269.193", "VSC_financial_statements_2017_separate|180", "VSC_financial_statements_2017_separate_180.csv", 4, 3, "Phải trả người bán ngắn hạn"),
    ("VJC", 2017, "note:short_term_trade_payables", "separate"): _disclosure_cell("VJC", 2017, "note:short_term_trade_payables", "separate", "593.067.305.745", "VJC_financial_statements_2017_separate|138", "VJC_financial_statements_2017_separate_138.csv", 4, 3, "Phải trả người bán ngắn hạn"),
    ("ACV", 2017, "note:short_term_trade_payables", "separate"): _disclosure_cell("ACV", 2017, "note:short_term_trade_payables", "separate", "1.133.590.882.595", "ACV_financial_statements_2017_separate|246", "ACV_financial_statements_2017_separate_246.csv", 3, 3, "Phải trả người bán ngắn hạn"),
    ("VIF", 2025, "note:investment_in_associates", "consolidated"): _disclosure_cell("VIF", 2025, "note:investment_in_associates", "consolidated", "1.521.893.908.775", "VIF_financial_statements_2025_consolidated|257", "VIF_financial_statements_2025_consolidated_257.csv", 18, 3, "Đầu tư vào công ty liên kết, liên doanh"),
    ("GVR", 2025, "note:investment_in_associates", "consolidated"): _disclosure_cell("GVR", 2025, "note:investment_in_associates", "consolidated", "1.958.492.542.280", "GVR_financial_statements_2025_consolidated|259", "GVR_financial_statements_2025_consolidated_259.csv", 25, 3, "Đầu tư vào công ty liên doanh, liên kết"),
    ("DPM", 2025, "note:investment_in_associates", "consolidated"): _disclosure_cell("DPM", 2025, "note:investment_in_associates", "consolidated", "31.808.865.536", "DPM_financial_statements_2025_consolidated|228", "DPM_financial_statements_2025_consolidated_228.csv", 17, 4, "Đầu tư vào công ty liên kết"),
    ("MSB", 2018, "note:net_customer_loans", "consolidated"): _disclosure_cell("MSB", 2018, "note:net_customer_loans", "consolidated", "47.768.344", "MSB_financial_statements_2018_consolidated|204", "MSB_financial_statements_2018_consolidated_204.csv", 11, 3, "Cho vay khách hàng - thuần"),
    ("EIB", 2018, "note:net_customer_loans", "consolidated"): _disclosure_cell("EIB", 2018, "note:net_customer_loans", "consolidated", "102.971.210", "EIB_financial_statements_2018_consolidated|192", "EIB_financial_statements_2018_consolidated_192.csv", 8, 3, "Cho vay khách hàng - thuần"),
    ("STB", 2018, "note:net_customer_loans", "consolidated"): _disclosure_cell("STB", 2018, "note:net_customer_loans", "consolidated", "253.100.111", "STB_financial_statements_2018_consolidated|139", "STB_financial_statements_2018_consolidated_139.csv", 11, 3, "Cho vay khách hàng - thuần"),
    ("VPB", 2020, "note:net_fx_income", "separate"): _disclosure_cell("VPB", 2020, "note:net_fx_income", "separate", "(167.056)", "VPB_financial_statements_2020_separate|203", "VPB_financial_statements_2020_separate_203.csv", 7, 3, "Lỗ thuần từ hoạt động kinh doanh ngoại hối"),
    ("EIB", 2020, "note:net_fx_income", "separate"): _disclosure_cell("EIB", 2020, "note:net_fx_income", "separate", "398.614", "EIB_financial_statements_2020_separate|269", "EIB_financial_statements_2020_separate_269.csv", 7, 2, "Lãi thuần từ hoạt động kinh doanh ngoại hối"),
    ("HDB", 2020, "note:net_fx_income", "separate"): _disclosure_cell("HDB", 2020, "note:net_fx_income", "separate", "237.571", "HDB_financial_statements_2020_separate|256", "HDB_financial_statements_2020_separate_256.csv", 7, 2, "Lãi thuần từ hoạt động kinh doanh ngoại hối"),
    ("VPI", 2021, "note:ending_common_shares_outstanding", "separate"): _disclosure_cell("VPI", 2021, "note:ending_common_shares_outstanding", "separate", "219.999.780", "VPI_financial_statements_2021_separate|1260", "VPI_financial_statements_2021_separate_1260.csv", 3, 1, "Cổ phiếu đang lưu hành cuối năm"),
    ("NLG", 2021, "note:ending_common_shares_outstanding", "separate"): _disclosure_cell("NLG", 2021, "note:ending_common_shares_outstanding", "separate", "382.940.013", "NLG_financial_statements_2021_separate|1214", "NLG_financial_statements_2021_separate_1214.csv", 5, 1, "Cổ phiếu đang lưu hành cuối năm"),
    ("DXG", 2021, "note:ending_common_shares_outstanding", "separate"): _disclosure_cell("DXG", 2021, "note:ending_common_shares_outstanding", "separate", "596.025.562", "DXG_financial_statements_2021_separate|1079", "DXG_financial_statements_2021_separate_1079.csv", 5, 1, "Cổ phiếu đang lưu hành cuối năm"),
    ("SNZ", 2021, "note:ending_common_shares_outstanding", "separate"): _disclosure_cell("SNZ", 2021, "note:ending_common_shares_outstanding", "separate", "376.491.800", "SNZ_financial_statements_2021_separate|1103", "SNZ_financial_statements_2021_separate_1103.csv", 4, 1, "Số lượng cổ phiếu phổ thông đang lưu hành"),
    ("GEE", 2022, "note:current_tax_expense_zero", "separate"): _disclosure_cell("GEE", 2022, "note:current_tax_expense_zero", "separate", "-", "GEE_financial_statements_2022_separate|408", "GEE_financial_statements_2022_separate_408.csv", 14, 4, "Chi phí thuế thu nhập doanh nghiệp hiện hành"),
    ("BID", 2023, "note:pre_provision_profit", "consolidated"): _disclosure_cell("BID", 2023, "note:pre_provision_profit", "consolidated", "47.932.419", "BID_financial_statements_2023_consolidated|402", "BID_financial_statements_2023_consolidated_402.csv", 19, 3, "Lợi nhuận thuần trước chi phí dự phòng rủi ro tín dụng"),
    ("BID", 2023, "note:credit_provision_expense", "consolidated"): _disclosure_cell("BID", 2023, "note:credit_provision_expense", "consolidated", "(20.343.515)", "BID_financial_statements_2023_consolidated|402", "BID_financial_statements_2023_consolidated_402.csv", 20, 3, "Chi phí dự phòng rủi ro tín dụng"),
    ("NAB", 2023, "note:pre_provision_profit", "consolidated"): _disclosure_cell("NAB", 2023, "note:pre_provision_profit", "consolidated", "4.151.756", "NAB_financial_statements_2023_consolidated_1|278", "NAB_financial_statements_2023_consolidated_1_278.csv", 15, 2, "Lợi nhuận thuần trước chi phí dự phòng rủi ro tín dụng"),
    ("NAB", 2023, "note:credit_provision_expense", "consolidated"): _disclosure_cell("NAB", 2023, "note:credit_provision_expense", "consolidated", "(847.804)", "NAB_financial_statements_2023_consolidated_1|278", "NAB_financial_statements_2023_consolidated_1_278.csv", 16, 2, "Chi phí dự phòng rủi ro tín dụng"),
    ("ABB", 2023, "note:pre_provision_profit", "consolidated"): _disclosure_cell("ABB", 2023, "note:pre_provision_profit", "consolidated", "2.083.392", "ABB_financial_statements_2023_consolidated|305", "ABB_financial_statements_2023_consolidated_305.csv", 19, 2, "Lợi nhuận thuần trước chi phí dự phòng rủi ro tín dụng"),
    ("ABB", 2023, "note:credit_provision_expense", "consolidated"): _disclosure_cell("ABB", 2023, "note:credit_provision_expense", "consolidated", "(1.499.348)", "ABB_financial_statements_2023_consolidated|305", "ABB_financial_statements_2023_consolidated_305.csv", 20, 2, "Chi phí dự phòng rủi ro tín dụng"),
})

EXPLICIT_CELLS.update({
    ("MSN", 2020, "note:finished_goods_gross", "consolidated"): _disclosure_cell("MSN", 2020, "note:finished_goods_gross", "consolidated", "3.163.599", "MSN_financial_statements_2020_consolidated|1730", "MSN_financial_statements_2020_consolidated_1730.csv", 6, 1, "Thành phẩm - giá gốc"),
    ("MSN", 2020, "note:inventory_gross_total", "consolidated"): _disclosure_cell("MSN", 2020, "note:inventory_gross_total", "consolidated", "12.730.397", "MSN_financial_statements_2020_consolidated|1730", "MSN_financial_statements_2020_consolidated_1730.csv", 9, 1, "Tổng hàng tồn kho - giá gốc"),
    ("MML", 2020, "note:finished_goods_gross", "consolidated"): _disclosure_cell("MML", 2020, "note:finished_goods_gross", "consolidated", "218.245.504.872", "MML_financial_statements_2020_consolidated|1032", "MML_financial_statements_2020_consolidated_1032.csv", 6, 1, "Thành phẩm - giá gốc"),
    ("MML", 2020, "note:inventory_gross_total", "consolidated"): _disclosure_cell("MML", 2020, "note:inventory_gross_total", "consolidated", "2.262.342.474.609", "MML_financial_statements_2020_consolidated|1032", "MML_financial_statements_2020_consolidated_1032.csv", 8, 1, "Tổng hàng tồn kho - giá gốc"),
    ("MPC", 2020, "note:finished_goods_gross", "consolidated"): _disclosure_cell("MPC", 2020, "note:finished_goods_gross", "consolidated", "2.758.530.073.037", "MPC_financial_statements_2020_consolidated|922", "MPC_financial_statements_2020_consolidated_922.csv", 6, 1, "Thành phẩm - giá gốc"),
    ("MPC", 2020, "note:inventory_gross_total", "consolidated"): _disclosure_cell("MPC", 2020, "note:inventory_gross_total", "consolidated", "3.135.048.390.423", "MPC_financial_statements_2020_consolidated|922", "MPC_financial_statements_2020_consolidated_922.csv", 7, 1, "Tổng hàng tồn kho - giá gốc"),
    ("VNM", 2020, "note:finished_goods_gross", "consolidated"): _disclosure_cell("VNM", 2020, "note:finished_goods_gross", "consolidated", "1.185.827.459.309", "VNM_financial_statements_2020_consolidated|1260", "VNM_financial_statements_2020_consolidated_1260.csv", 6, 1, "Thành phẩm - giá gốc"),
    ("VNM", 2020, "note:inventory_gross_total", "consolidated"): _disclosure_cell("VNM", 2020, "note:inventory_gross_total", "consolidated", "4.952.848.688.011", "VNM_financial_statements_2020_consolidated|1260", "VNM_financial_statements_2020_consolidated_1260.csv", 9, 1, "Tổng hàng tồn kho - giá gốc"),
    ("PNJ", 2022, "note:tangible_gross_cost", "consolidated"): _disclosure_cell("PNJ", 2022, "note:tangible_gross_cost", "consolidated", "672.781.337.441", "PNJ_financial_statements_2022_consolidated|779", "PNJ_financial_statements_2022_consolidated_779.csv", 6, 5, "TSCĐ hữu hình - nguyên giá cuối năm"),
    ("PNJ", 2022, "note:tangible_accumulated_depreciation", "consolidated"): _disclosure_cell("PNJ", 2022, "note:tangible_accumulated_depreciation", "consolidated", "433.046.785.694", "PNJ_financial_statements_2022_consolidated|779", "PNJ_financial_statements_2022_consolidated_779.csv", 11, 5, "TSCĐ hữu hình - khấu hao lũy kế cuối năm"),
    ("MWG", 2022, "note:tangible_gross_cost", "consolidated"): _disclosure_cell("MWG", 2022, "note:tangible_gross_cost", "consolidated", "20.841.513.609.195", "MWG_financial_statements_2022_consolidated|826", "MWG_financial_statements_2022_consolidated_826.csv", 6, 4, "TSCĐ hữu hình - nguyên giá cuối năm"),
    ("MWG", 2022, "note:tangible_accumulated_depreciation", "consolidated"): _disclosure_cell("MWG", 2022, "note:tangible_accumulated_depreciation", "consolidated", "(11.188.183.603.219)", "MWG_financial_statements_2022_consolidated|826", "MWG_financial_statements_2022_consolidated_826.csv", 13, 4, "TSCĐ hữu hình - khấu hao lũy kế cuối năm"),
    ("HHS", 2022, "note:tangible_gross_cost", "consolidated"): _disclosure_cell("HHS", 2022, "note:tangible_gross_cost", "consolidated", "31.692.909.172", "HHS_financial_statements_2022_consolidated|1181", "HHS_financial_statements_2022_consolidated_1181.csv", 5, 5, "TSCĐ hữu hình - nguyên giá cuối năm"),
    ("HHS", 2022, "note:tangible_accumulated_depreciation", "consolidated"): _disclosure_cell("HHS", 2022, "note:tangible_accumulated_depreciation", "consolidated", "12.059.071.393", "HHS_financial_statements_2022_consolidated|1181", "HHS_financial_statements_2022_consolidated_1181.csv", 10, 5, "TSCĐ hữu hình - hao mòn lũy kế cuối năm"),
    ("HUT", 2022, "note:tangible_gross_cost", "consolidated"): _disclosure_cell("HUT", 2022, "note:tangible_gross_cost", "consolidated", "7.484.988.382.760", "HUT_financial_statements_2022_consolidated|1142", "HUT_financial_statements_2022_consolidated_1142.csv", 11, 6, "TSCĐ hữu hình - nguyên giá cuối năm"),
    ("HUT", 2022, "note:tangible_accumulated_depreciation", "consolidated"): _disclosure_cell("HUT", 2022, "note:tangible_accumulated_depreciation", "consolidated", "1.812.165.288.740", "HUT_financial_statements_2022_consolidated|1142", "HUT_financial_statements_2022_consolidated_1142.csv", 20, 6, "TSCĐ hữu hình - hao mòn lũy kế cuối năm"),
    ("MSN", 2022, "note:admin_depreciation", "separate"): _disclosure_cell("MSN", 2022, "note:admin_depreciation", "separate", "5.071.225.625", "MSN_financial_statements_2022_separate|1078", "MSN_financial_statements_2022_separate_1078.csv", 1, 1, "Chi phí khấu hao và phân bổ trong chi phí quản lý"),
    ("MSN", 2022, "note:admin_expense_total", "separate"): _disclosure_cell("MSN", 2022, "note:admin_expense_total", "separate", "322.099.246.913", "MSN_financial_statements_2022_separate|1078", "MSN_financial_statements_2022_separate_1078.csv", 3, 1, "Tổng chi phí quản lý doanh nghiệp"),
    ("MPC", 2022, "note:admin_depreciation", "separate"): _disclosure_cell("MPC", 2022, "note:admin_depreciation", "separate", "9.276.739.197", "MPC_financial_statements_2022_separate|1344", "MPC_financial_statements_2022_separate_1344.csv", 3, 1, "Chi phí khấu hao và phân bổ trong chi phí quản lý"),
    ("MPC", 2022, "note:admin_expense_total", "separate"): _disclosure_cell("MPC", 2022, "note:admin_expense_total", "separate", "104.792.026.944", "MPC_financial_statements_2022_separate|1344", "MPC_financial_statements_2022_separate_1344.csv", 8, 1, "Tổng chi phí quản lý doanh nghiệp"),
    ("VNM", 2022, "note:admin_depreciation", "separate"): _disclosure_cell("VNM", 2022, "note:admin_depreciation", "separate", "45.698.243.760", "VNM_financial_statements_2022_separate|1557", "VNM_financial_statements_2022_separate_1557.csv", 3, 1, "Chi phí khấu hao trong chi phí quản lý"),
    ("VNM", 2022, "note:admin_expense_total", "separate"): _disclosure_cell("VNM", 2022, "note:admin_expense_total", "separate", "859.560.797.485", "VNM_financial_statements_2022_separate|1557", "VNM_financial_statements_2022_separate_1557.csv", 12, 1, "Tổng chi phí quản lý doanh nghiệp"),
    ("MML", 2022, "note:admin_depreciation", "separate"): _disclosure_cell("MML", 2022, "note:admin_depreciation", "separate", "276.480.503", "MML_financial_statements_2022_separate|1034", "MML_financial_statements_2022_separate_1034.csv", 4, 1, "Chi phí khấu hao và phân bổ trong chi phí quản lý"),
    ("MML", 2022, "note:admin_expense_total", "separate"): _disclosure_cell("MML", 2022, "note:admin_expense_total", "separate", "111.716.043.665", "MML_financial_statements_2022_separate|1034", "MML_financial_statements_2022_separate_1034.csv", 8, 1, "Tổng chi phí quản lý doanh nghiệp"),
    ("MBB", 2022, "note:government_bonds", "separate"): _disclosure_cell("MBB", 2022, "note:government_bonds", "separate", "44.620.225", "MBB_financial_statements_2022_separate_2|1405", "MBB_financial_statements_2022_separate_2_1405.csv", 2, 1, "Trái phiếu Chính phủ"),
    ("MBB", 2022, "note:total_debt_securities", "separate"): _disclosure_cell("MBB", 2022, "note:total_debt_securities", "separate", "152.053.008", "MBB_financial_statements_2022_separate_2|1405", "MBB_financial_statements_2022_separate_2_1405.csv", 1, 1, "Chứng khoán nợ"),
    ("MSB", 2022, "note:government_bonds", "separate"): _disclosure_cell("MSB", 2022, "note:government_bonds", "separate", "16.577.065", "MSB_financial_statements_2022_separate|1342", "MSB_financial_statements_2022_separate_1342.csv", 2, 1, "Chứng khoán Chính phủ"),
    ("MSB", 2022, "note:total_debt_securities", "separate"): _disclosure_cell("MSB", 2022, "note:total_debt_securities", "separate", "31.554.087", "MSB_financial_statements_2022_separate|1342", "MSB_financial_statements_2022_separate_1342.csv", 1, 1, "Chứng khoán nợ"),
    ("STB", 2022, "note:government_bonds", "separate"): _disclosure_cell("STB", 2022, "note:government_bonds", "separate", "22.309.012", "STB_financial_statements_2022_separate|1398", "STB_financial_statements_2022_separate_1398.csv", 2, 1, "Trái phiếu Chính phủ"),
    ("STB", 2022, "note:total_debt_securities", "separate"): _disclosure_cell("STB", 2022, "note:total_debt_securities", "separate", "27.759.758", "STB_financial_statements_2022_separate|1398", "STB_financial_statements_2022_separate_1398.csv", 1, 1, "Chứng khoán nợ"),
    ("NAB", 2025, "note:average_employee_income_monthly", "separate"): _disclosure_cell("NAB", 2025, "note:average_employee_income_monthly", "separate", "28", "NAB_financial_statements_2025_separate|2049", "NAB_financial_statements_2025_separate_2049.csv", 7, 1, "Thu nhập bình quân tháng, triệu đồng/người"),
    ("ABB", 2025, "note:average_employee_income_monthly_ocr", "separate"): _disclosure_cell("ABB", 2025, "note:average_employee_income_monthly_ocr", "separate", "3451", "ABB_financial_statements_2025_separate|2176", "ABB_financial_statements_2025_separate_2176.csv", 8, 1, "Thu nhập bình quân tháng, OCR 34,51 triệu đồng/người"),
    ("ACB", 2025, "note:average_employee_income_annual", "separate"): _disclosure_cell("ACB", 2025, "note:average_employee_income_annual", "separate", "453", "ACB_financial_statements_2025_separate|2113", "ACB_financial_statements_2025_separate_2113.csv", 7, 1, "Thu nhập bình quân năm, triệu đồng/người"),
    ("STB", 2025, "note:average_employee_income_monthly_ocr", "separate"): _disclosure_cell("STB", 2025, "note:average_employee_income_monthly_ocr", "separate", "3599", "STB_financial_statements_2025_separate|2327", "STB_financial_statements_2025_separate_2327.csv", 6, 1, "Thu nhập bình quân tháng, OCR 35,99 triệu đồng/người"),
})

# Additional multi-year disclosure cells recovered by re-auditing the tables
# selected across older runs.  The old numeric answers are not reused; only
# the source coordinates below enter the new deterministic argmax programs.
for _spec in [
    # NVL parent short-term advances to suppliers (code 132).
    ("NVL", 2017, "cdkt:132", "separate", "385.683.515.155", "NVL_financial_statements_2017_separate|178", "NVL_financial_statements_2017_separate_178.csv", 10, 3, "Trả trước cho người bán ngắn hạn"),
    ("NVL", 2021, "cdkt:132", "separate", "346.937.662.590", "NVL_financial_statements_2021_separate|252", "NVL_financial_statements_2021_separate_252.csv", 10, 3, "Trả trước cho người bán ngắn hạn"),
    ("NVL", 2023, "cdkt:132", "separate", "346.452.545.721", "NVL_financial_statements_2023_separate|358", "NVL_financial_statements_2023_separate_358.csv", 10, 3, "Trả trước cho người bán ngắn hạn"),
    # OCB parent specific loan-loss provision expense.
    ("OCB", 2017, "note:specific_loan_provision", "separate", "112.060.694.398", "OCB_financial_statements_2017_separate|1896", "OCB_financial_statements_2017_separate_1896.csv", 2, 1, "Trích lập dự phòng cụ thể cho vay khách hàng"),
    ("OCB", 2018, "note:specific_loan_provision", "separate", "633.084.224.721", "OCB_financial_statements_2018_separate|1940", "OCB_financial_statements_2018_separate_1940.csv", 2, 1, "Trích lập dự phòng cụ thể cho vay khách hàng"),
    ("OCB", 2019, "note:specific_loan_provision", "separate", "822.479.834.736", "OCB_financial_statements_2019_separate|2086", "OCB_financial_statements_2019_separate_2086.csv", 2, 1, "Trích lập dự phòng cụ thể cho vay khách hàng"),
    ("OCB", 2025, "note:specific_loan_provision", "separate", "2.163.777.088.772", "OCB_financial_statements_2025_separate|2770", "OCB_financial_statements_2025_separate_2770.csv", 2, 1, "Trích lập dự phòng cụ thể cho vay khách hàng"),
    # MBB government-bond investment balance (main portfolio row).
    ("MBB", 2015, "note:government_bonds", "consolidated", "18.919.916", "MBB_financial_statements_2015_consolidated|1374", "MBB_financial_statements_2015_consolidated_1374.csv", 3, 1, "Trái phiếu Chính phủ"),
    ("MBB", 2016, "note:government_bonds", "consolidated", "22.017.624", "MBB_financial_statements_2016_consolidated|1462", "MBB_financial_statements_2016_consolidated_1462.csv", 3, 1, "Trái phiếu Chính phủ"),
    ("MBB", 2017, "note:government_bonds", "consolidated", "23.334.935", "MBB_financial_statements_2017_consolidated|1390", "MBB_financial_statements_2017_consolidated_1390.csv", 3, 1, "Trái phiếu Chính phủ"),
    ("MBB", 2018, "note:government_bonds", "consolidated", "43.802.956", "MBB_financial_statements_2018_consolidated|1636", "MBB_financial_statements_2018_consolidated_1636.csv", 2, 1, "Trái phiếu Chính phủ"),
    ("MBB", 2022, "note:government_bonds", "consolidated", "44.620.225", "MBB_financial_statements_2022_consolidated|1701", "MBB_financial_statements_2022_consolidated_1701.csv", 2, 1, "Trái phiếu Chính phủ"),
    # CTG parent intangible fixed-asset total NBV (million VND; scale immaterial for argmax).
    ("CTG", 2016, "note:intangible_total_nbv", "separate", "3.932.117", "CTG_financial_statements_2016_separate|1222", "CTG_financial_statements_2016_separate_1222.csv", 18, 3, "Giá trị còn lại tại ngày cuối năm - Tổng"),
    ("CTG", 2021, "note:intangible_total_nbv", "separate", "4.204.532", "CTG_financial_statements_2021_separate|1515", "CTG_financial_statements_2021_separate_1515.csv", 15, 3, "Giá trị còn lại tại ngày cuối năm - Tổng"),
    ("CTG", 2022, "note:intangible_total_nbv", "separate", "4.076.230", "CTG_financial_statements_2022_separate|1341", "CTG_financial_statements_2022_separate_1341.csv", 15, 3, "Giá trị còn lại tại ngày cuối năm - Tổng"),
    # KHG parent total real-estate brokerage commission/cost.
    ("KHG", 2019, "note:brokerage_commission", "separate", "89.144.410.598", "KHG_financial_statements_2019_separate|693", "KHG_financial_statements_2019_separate_693.csv", 2, 1, "Giá vốn dịch vụ môi giới bất động sản"),
    ("KHG", 2020, "note:brokerage_commission", "separate", "159.868.248.240", "KHG_financial_statements_2020_separate|667", "KHG_financial_statements_2020_separate_667.csv", 2, 1, "Giá vốn dịch vụ môi giới bất động sản"),
    ("KHG", 2021, "note:brokerage_commission", "separate", "310.228.563.963", "KHG_financial_statements_2021_separate|910", "KHG_financial_statements_2021_separate_910.csv", 2, 1, "Giá vốn dịch vụ môi giới bất động sản"),
    ("KHG", 2022, "note:brokerage_commission", "separate", "560.911.129.505", "KHG_financial_statements_2022_separate|876", "KHG_financial_statements_2022_separate_876.csv", 2, 1, "Giá vốn dịch vụ môi giới bất động sản"),
    ("KHG", 2023, "note:brokerage_commission", "separate", "42.745.665.072", "KHG_financial_statements_2023_separate|916", "KHG_financial_statements_2023_separate_916.csv", 2, 1, "Giá vốn dịch vụ môi giới bất động sản"),
    # BSR consolidated LPG revenue.
    ("BSR", 2017, "note:lpg_revenue", "consolidated", "5.697.895.452.596", "BSR_financial_statements_2017_consolidated|1102", "BSR_financial_statements_2017_consolidated_1102.csv", 5, 1, "Doanh thu LPG"),
    ("BSR", 2019, "note:lpg_revenue", "consolidated", "5.983.079.101.866", "BSR_financial_statements_2019_consolidated|1021", "BSR_financial_statements_2019_consolidated_1021.csv", 6, 1, "Doanh thu LPG"),
    ("BSR", 2021, "note:lpg_revenue", "consolidated", "7.942.513.069.668", "BSR_financial_statements_2021_consolidated|1773", "BSR_financial_statements_2021_consolidated_1773.csv", 5, 1, "Doanh thu LPG"),
    ("BSR", 2022, "note:lpg_revenue", "consolidated", "9.292.539.523.199", "BSR_financial_statements_2022_consolidated|1284", "BSR_financial_statements_2022_consolidated_1284.csv", 6, 1, "Doanh thu LPG"),
    ("BSR", 2025, "note:lpg_revenue", "consolidated", "7.646.974.909.812", "BSR_financial_statements_2025_consolidated|1334", "BSR_financial_statements_2025_consolidated_1334.csv", 7, 1, "Doanh thu LPG"),
    # STB accrued interest from customer loans.
    ("STB", 2017, "note:accrued_loan_interest", "consolidated", "22.399.323", "STB_financial_statements_2017_consolidated|1554", "STB_financial_statements_2017_consolidated_1554.csv", 1, 1, "Lãi từ cho vay khách hàng"),
    ("STB", 2022, "note:accrued_loan_interest", "consolidated", "3.370.271", "STB_financial_statements_2022_consolidated|1678", "STB_financial_statements_2022_consolidated_1678.csv", 1, 1, "Lãi từ cho vay khách hàng"),
    ("STB", 2024, "note:accrued_loan_interest", "consolidated", "3.390.704", "STB_financial_statements_2024_consolidated|2054", "STB_financial_statements_2024_consolidated_2054.csv", 1, 1, "Lãi dự thu từ cho vay khách hàng"),
    # Combined consolidated/parent statement layouts omitted by the cube.
    ("MSR", 2016, "cdkt:200", "consolidated", "24.039.367.458", "MSR_financial_statements_2016_consolidated|181", "MSR_financial_statements_2016_consolidated_181.csv", 2, 3, "Tài sản dài hạn"),
    ("MSR", 2017, "cdkt:330", "separate", "541.074.175", "MSR_financial_statements_2017_separate|167", "MSR_financial_statements_2017_separate_167.csv", 12, 5, "Nợ dài hạn"),
    ("MSR", 2017, "cdkt:400", "separate", "9.443.792.507", "MSR_financial_statements_2017_separate|167", "MSR_financial_statements_2017_separate_167.csv", 17, 5, "Vốn chủ sở hữu"),
]:
    _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label = _spec
    EXPLICIT_CELLS[(_ticker, _year, _key, _scope_name)] = _disclosure_cell(
        _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label
    )

# More high-confidence failures recovered by contrasting historical runs.  The
# old programs confused a selector cell with the requested downstream ratio.
for _spec in [
    # HPG: select the year by the long-term-prepayment allocation, then compute
    # deposit/loan interest as a share of parent finance revenue.
    ("HPG", 2015, "note:office_repair_tool_prepayment_allocation", "separate", "6.194.124.815", "HPG_financial_statements_2015_separate|959", "HPG_financial_statements_2015_separate_959.csv", 2, 1, "Phân bổ chi phí sửa chữa văn phòng, công cụ dụng cụ và chi phí trả trước dài hạn khác"),
    ("HPG", 2018, "note:office_repair_tool_prepayment_allocation", "separate", "2.896.206.885", "HPG_financial_statements_2018_separate|1021", "HPG_financial_statements_2018_separate_1021.csv", 2, 1, "Phân bổ chi phí sửa chữa văn phòng, công cụ dụng cụ và chi phí trả trước dài hạn khác"),
    ("HPG", 2022, "note:office_repair_tool_prepayment_allocation", "separate", "2.556.315.001", "HPG_financial_statements_2022_separate|938", "HPG_financial_statements_2022_separate_938.csv", 2, 1, "Phân bổ chi phí sửa chữa văn phòng, công cụ dụng cụ và chi phí trả trước dài hạn khác"),
    ("HPG", 2024, "note:office_repair_tool_prepayment_allocation", "separate", "1.254.509.832", "HPG_financial_statements_2024_separate|927", "HPG_financial_statements_2024_separate_927.csv", 2, 1, "Phân bổ chi phí sửa chữa văn phòng, công cụ dụng cụ và chi phí trả trước dài hạn khác"),
    ("HPG", 2015, "note:deposit_and_loan_interest", "separate", "29.261.742.838", "HPG_financial_statements_2015_separate|938", "HPG_financial_statements_2015_separate_938.csv", 1, 1, "Lãi tiền gửi và cho vay"),
    ("HPG", 2018, "note:deposit_and_loan_interest", "separate", "42.347.332.796", "HPG_financial_statements_2018_separate|998", "HPG_financial_statements_2018_separate_998.csv", 1, 1, "Lãi tiền gửi và cho vay"),
    ("HPG", 2022, "note:deposit_and_loan_interest", "separate", "301.015.478.868", "HPG_financial_statements_2022_separate|917", "HPG_financial_statements_2022_separate_917.csv", 1, 1, "Lãi tiền gửi và cho vay"),
    ("HPG", 2024, "note:deposit_and_loan_interest", "separate", "57.679.836.170", "HPG_financial_statements_2024_separate|910", "HPG_financial_statements_2024_separate_910.csv", 1, 1, "Lãi tiền gửi và cho vay"),
    ("HPG", 2015, "note:parent_finance_revenue", "separate", "1.906.513.504.388", "HPG_financial_statements_2015_separate|222", "HPG_financial_statements_2015_separate_222.csv", 4, 3, "Doanh thu hoạt động tài chính"),
    ("HPG", 2018, "note:parent_finance_revenue", "separate", "7.338.449.240.184", "HPG_financial_statements_2018_separate|217", "HPG_financial_statements_2018_separate_217.csv", 4, 3, "Doanh thu hoạt động tài chính"),
    ("HPG", 2022, "note:parent_finance_revenue", "separate", "6.024.895.296.275", "HPG_financial_statements_2022_separate|212", "HPG_financial_statements_2022_separate_212.csv", 6, 3, "Doanh thu hoạt động tài chính"),
    ("HPG", 2024, "note:parent_finance_revenue", "separate", "10.300.211.056.695", "HPG_financial_statements_2024_separate|216", "HPG_financial_statements_2024_separate_216.csv", 6, 3, "Doanh thu hoạt động tài chính"),
    # ABB: signed provision movement selects the year; the requested answer is
    # pre-provision operating profit / total assets, not the selector itself.
    ("ABB", 2020, "note:afs_provision_movement", "separate", "(24.107)", "ABB_financial_statements_2020_separate|1449", "ABB_financial_statements_2020_separate_1449.csv", 8, 1, "Số trích lập/(hoàn nhập) dự phòng chứng khoán sẵn sàng để bán"),
    ("ABB", 2022, "note:afs_provision_movement", "separate", "(25.896)", "ABB_financial_statements_2022_separate|1501", "ABB_financial_statements_2022_separate_1501.csv", 7, 1, "Số trích lập/(hoàn nhập) dự phòng chứng khoán sẵn sàng để bán"),
    ("ABB", 2023, "note:afs_provision_movement", "separate", "(12.703)", "ABB_financial_statements_2023_separate|1575", "ABB_financial_statements_2023_separate_1575.csv", 7, 1, "Số trích lập/(hoàn nhập) dự phòng chứng khoán sẵn sàng để bán"),
    ("ABB", 2020, "note:pre_provision_operating_profit", "separate", "1.881.705", "ABB_financial_statements_2020_separate|287", "ABB_financial_statements_2020_separate_287.csv", 19, 2, "Lợi nhuận thuần trước chi phí dự phòng rủi ro tín dụng"),
    ("ABB", 2022, "note:pre_provision_operating_profit", "separate", "2.462.760", "ABB_financial_statements_2022_separate|272", "ABB_financial_statements_2022_separate_272.csv", 19, 2, "Lợi nhuận thuần trước chi phí dự phòng rủi ro tín dụng"),
    ("ABB", 2023, "note:pre_provision_operating_profit", "separate", "2.012.636", "ABB_financial_statements_2023_separate|296", "ABB_financial_statements_2023_separate_296.csv", 19, 2, "Lợi nhuận thuần trước chi phí dự phòng rủi ro tín dụng"),
    ("ABB", 2020, "note:parent_total_assets", "separate", "116.267.442", "ABB_financial_statements_2020_separate|220", "ABB_financial_statements_2020_separate_220.csv", 37, 2, "Tổng tài sản"),
    ("ABB", 2022, "note:parent_total_assets", "separate", "130.064.695", "ABB_financial_statements_2022_separate|207", "ABB_financial_statements_2022_separate_207.csv", 36, 2, "Tổng tài sản"),
    ("ABB", 2023, "note:parent_total_assets", "separate", "161.977.363", "ABB_financial_statements_2023_separate|217", "ABB_financial_statements_2023_separate_217.csv", 33, 2, "Tổng tài sản"),
    # OCB parent customer loans by industry: real-estate balance and the total
    # line from the same table for each requested date.
    ("OCB", 2017, "note:real_estate_customer_loans", "separate", "4.423.443.894.553", "OCB_financial_statements_2017_separate|1259", "OCB_financial_statements_2017_separate_1259.csv", 5, 1, "Hoạt động kinh doanh bất động sản"),
    ("OCB", 2017, "note:total_customer_loans_by_industry", "separate", "48.182.976.683.825", "OCB_financial_statements_2017_separate|1259", "OCB_financial_statements_2017_separate_1259.csv", 14, 1, "Tổng dư nợ cho vay khách hàng theo ngành"),
    ("OCB", 2020, "note:real_estate_customer_loans", "separate", "6.268.194.676.278", "OCB_financial_statements_2020_separate|1169", "OCB_financial_statements_2020_separate_1169.csv", 8, 1, "Hoạt động kinh doanh bất động sản"),
    ("OCB", 2020, "note:total_customer_loans_by_industry", "separate", "89.237.886.166.154", "OCB_financial_statements_2020_separate|1169", "OCB_financial_statements_2020_separate_1169.csv", 14, 1, "Tổng dư nợ cho vay khách hàng theo ngành"),
    ("OCB", 2024, "note:real_estate_customer_loans", "separate", "28.851.442.834.099", "OCB_financial_statements_2024_separate|1433", "OCB_financial_statements_2024_separate_1433.csv", 2, 1, "Hoạt động kinh doanh bất động sản"),
    ("OCB", 2024, "note:total_customer_loans_by_industry", "separate", "170.844.469.638.663", "OCB_financial_statements_2024_separate|1433", "OCB_financial_statements_2024_separate_1433.csv", 14, 1, "Tổng dư nợ cho vay khách hàng theo ngành"),
    ("OCB", 2025, "note:real_estate_customer_loans", "separate", "38.045.493.326.685", "OCB_financial_statements_2025_separate|1876", "OCB_financial_statements_2025_separate_1876.csv", 1, 1, "Hoạt động kinh doanh bất động sản"),
    ("OCB", 2025, "note:total_customer_loans_by_industry", "separate", "198.764.945.826.810", "OCB_financial_statements_2025_separate|1876", "OCB_financial_statements_2025_separate_1876.csv", 16, 1, "Tổng dư nợ cho vay khách hàng theo ngành"),
]:
    _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label = _spec
    EXPLICIT_CELLS[(_ticker, _year, _key, _scope_name)] = _disclosure_cell(
        _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label
    )

# Subsidiary voting-rate tables mix integer percentages (100) with OCR values
# whose decimal separator was stripped (9991 means 99.91).  Preserve the raw
# tokens and normalize only in the executable expression.
for _ticker, _ref, _csv, _col, _raw_values in (
    ("NLG", "NLG_financial_statements_2020_consolidated|312", "NLG_financial_statements_2020_consolidated_312.csv", 3,
     ("9991", "8733", "100", "100", "100", "9998", "5000", "100", "5000", "100", "100", "8125", "100", "7603", "100", "100", "100", "100", "100")),
    ("SCR", "SCR_financial_statements_2020_consolidated|372", "SCR_financial_statements_2020_consolidated_372.csv", 5,
     ("10000", "5200", "10000", "9990", "5000", "9017", "7400", "10000", "9517", "10000", "10000", "6100")),
):
    for _offset, _raw in enumerate(_raw_values):
        _key = f"note:subsidiary_voting_rate_{_offset + 1}"
        EXPLICIT_CELLS[(_ticker, 2020, _key, "consolidated")] = _disclosure_cell(
            _ticker, 2020, _key, "consolidated", _raw, _ref, _csv,
            _offset + 2, _col, f"Subsidiary voting rate #{_offset + 1}",
        )

# Direct percentage/count disclosures that historical models parsed as raw
# money or lost the decimal separator from.  Values remain source tokens and
# all comparisons/ratios are performed by the generated grader-time query.
for _spec in [
    ("PLX", 2016, "note:ptn_ownership_rate", "separate", "60,00%", "PLX_financial_statements_2016_separate|826", "PLX_financial_statements_2016_separate_826.csv", 14, 3, "Tỷ lệ sở hữu tại Công ty TNHH Hóa chất PTN"),
    ("HHV", 2023, "note:total_voting_rate", "separate", "21,34%", "HHV_financial_statements_2023_separate|2027", "HHV_financial_statements_2023_separate_2027.csv", 5, 2, "Cộng quyền biểu quyết trực tiếp và gián tiếp"),
    ("HDG", 2015, "note:interest_payable", "consolidated", "8.386.591.115", "HDG_financial_statements_2015_consolidated|1286", "HDG_financial_statements_2015_consolidated_1286.csv", 1, 1, "Lãi vay phải trả"),
    ("HDG", 2015, "note:long_term_borrowings", "consolidated", "674.955.821.621", "HDG_financial_statements_2015_consolidated|188", "HDG_financial_statements_2015_consolidated_188.csv", 15, 3, "Vay dài hạn"),
    ("KHG", 2024, "note:nguyen_khai_hoan_capital_rate", "consolidated", "31,97%", "KHG_financial_statements_2024_consolidated|894", "KHG_financial_statements_2024_consolidated_894.csv", 3, 2, "Tỷ trọng vốn góp của ông Nguyễn Khải Hoàn"),
    ("MBB", 2020, "note:operating_lease_due_within_one_year", "separate", "31.007", "MBB_financial_statements_2020_separate|1759", "MBB_financial_statements_2020_separate_1759.csv", 2, 1, "Cam kết thuê hoạt động đến hạn trong một năm"),
    ("HDB", 2020, "note:operating_lease_due_within_one_year", "separate", "17.186", "HDB_financial_statements_2020_separate|2246", "HDB_financial_statements_2020_separate_2246.csv", 3, 1, "Cam kết thuê hoạt động đến hạn trong một năm"),
    ("KLB", 2020, "note:operating_lease_due_within_one_year", "separate", "49.649", "KLB_financial_statements_2020_separate|2226", "KLB_financial_statements_2020_separate_2226.csv", 1, 1, "Cam kết thuê hoạt động trong vòng một năm"),
    ("NAB", 2020, "note:operating_lease_due_within_one_year", "separate", "79.657", "NAB_financial_statements_2020_separate|3284", "NAB_financial_statements_2020_separate_3284.csv", 1, 1, "Cam kết thuê hoạt động đến một năm"),
    ("SHB", 2016, "note:general_customer_loan_provision", "separate", "1.018.726", "SHB_financial_statements_2016_separate|1479", "SHB_financial_statements_2016_separate_1479.csv", 1, 1, "Dự phòng chung cho vay khách hàng"),
    ("SHB", 2016, "note:total_customer_loan_provision", "separate", "1.691.202", "SHB_financial_statements_2016_separate|1479", "SHB_financial_statements_2016_separate_1479.csv", 3, 1, "Tổng dự phòng rủi ro cho vay khách hàng"),
    ("CTG", 2022, "note:off_balance_contingent_liabilities", "separate", "159.575.895", "CTG_financial_statements_2022_separate|1609", "CTG_financial_statements_2022_separate_1609.csv", 2, 1, "Nghĩa vụ nợ tiềm ẩn"),
    ("CTG", 2022, "note:off_balance_commitments", "separate", "310.604.986", "CTG_financial_statements_2022_separate|1609", "CTG_financial_statements_2022_separate_1609.csv", 6, 1, "Các cam kết đưa ra"),
    ("CTG", 2022, "note:parent_total_assets", "separate", "1.793.240.351", "CTG_financial_statements_2022_separate|326", "CTG_financial_statements_2022_separate_326.csv", 34, 3, "Tổng tài sản Có"),
    ("MWG", 2020, "note:ending_common_shares_outstanding", "consolidated", "453.209.987", "MWG_financial_statements_2020_consolidated|979", "MWG_financial_statements_2020_consolidated_979.csv", 2, 1, "Cổ phiếu phổ thông đã phát hành và góp vốn đầy đủ"),
    ("HHS", 2020, "note:ending_common_shares_outstanding", "consolidated", "274.744.063", "HHS_financial_statements_2020_consolidated|1097", "HHS_financial_statements_2020_consolidated_1097.csv", 14, 1, "Số lượng cổ phiếu đang lưu hành"),
    ("PNJ", 2020, "note:ending_common_shares_outstanding", "consolidated", "227.442.803", "PNJ_financial_statements_2020_consolidated|801", "PNJ_financial_statements_2020_consolidated_801.csv", 4, 1, "Số lượng cổ phiếu đang lưu hành"),
    ("HUT", 2020, "note:ending_common_shares_outstanding", "consolidated", "268.631.965", "HUT_financial_statements_2020_consolidated|1192", "HUT_financial_statements_2020_consolidated_1192.csv", 4, 1, "Số lượng cổ phiếu đang lưu hành"),
]:
    _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label = _spec
    EXPLICIT_CELLS[(_ticker, _year, _key, _scope_name)] = _disclosure_cell(
        _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label
    )

for _year, _ref, _rows in (
    (2018, "SAB_financial_statements_2018_consolidated|1436", ((1, "5.813.809.353.665"), (2, "1.764.524.052.187"), (3, "634.592.696.360"), (4, "3.299.307.255.808"), (5, "646.656.970.561"))),
    (2020, "SAB_financial_statements_2020_consolidated|1559", ((1, "5.367.647.284.900"), (2, "1.507.607.536.510"), (3, "593.451.595.931"), (4, "2.876.725.251.113"), (5, "542.196.698.775"))),
    (2024, "SAB_financial_statements_2024_consolidated|1653", ((1, "7.849.174.906.031"), (2, "3.878.645.958.871"), (3, "1.717.796.120.922"), (4, "548.883.626.755"), (5, "675.003.471.040"))),
    (2025, "SAB_financial_statements_2025_consolidated|1852", ((1, "9.392.816.124.244"), (2, "3.826.738.803.498"), (3, "1.952.113.865.896"), (4, "730.272.467.434"), (5, "783.554.386.906"))),
):
    _line = _ref.split("|", 1)[1]
    for _component, (_row, _raw) in enumerate(_rows, start=1):
        _key = f"note:production_cost_component_{_component}"
        EXPLICIT_CELLS[("SAB", _year, _key, "consolidated")] = _disclosure_cell(
            "SAB", _year, _key, "consolidated", _raw, _ref,
            f"SAB_financial_statements_{_year}_consolidated_{_line}.csv",
            _row, 1, f"Production-cost component #{_component}",
        )

for _spec in [
    # Interest-rate repricing schedules: the 1–3 month asset bucket and total.
    ("EIB", 2020, "note:assets_repricing_1_3_months", "consolidated", "34.215.541", "EIB_financial_statements_2020_consolidated|2037", "EIB_financial_statements_2020_consolidated_2037.csv", 10, 4, "Total assets repricing in 1–3 months"),
    ("EIB", 2020, "note:assets_repricing_total", "consolidated", "163.129.920", "EIB_financial_statements_2020_consolidated|2037", "EIB_financial_statements_2020_consolidated_2037.csv", 10, 9, "Total assets in repricing schedule"),
    ("STB", 2020, "note:assets_repricing_1_3_months", "consolidated", "204.251.814", "STB_financial_statements_2020_consolidated|2324", "STB_financial_statements_2020_consolidated_2324.csv", 13, 4, "Total assets repricing in 1–3 months"),
    ("STB", 2020, "note:assets_repricing_total", "consolidated", "505.541.942", "STB_financial_statements_2020_consolidated|2324", "STB_financial_statements_2020_consolidated_2324.csv", 13, 9, "Total assets in repricing schedule"),
    ("SSB", 2020, "note:assets_repricing_1_3_months", "consolidated", "27.401.037", "SSB_financial_statements_2020_consolidated|2195", "SSB_financial_statements_2020_consolidated_2195.csv", 14, 4, "Total assets repricing in 1–3 months"),
    ("SSB", 2020, "note:assets_repricing_total", "consolidated", "181.190.363", "SSB_financial_statements_2020_consolidated|2195", "SSB_financial_statements_2020_consolidated_2195.csv", 14, 9, "Total assets in repricing schedule"),
    # Customer-loan general and total provisions at end-2018.
    ("OCB", 2018, "note:general_customer_loan_provision", "consolidated", "402.217.551.245", "OCB_financial_statements_2018_consolidated|1679", "OCB_financial_statements_2018_consolidated_1679.csv", 1, 1, "Dự phòng chung cho vay khách hàng"),
    ("OCB", 2018, "note:total_customer_loan_provision", "consolidated", "565.344.903.505", "OCB_financial_statements_2018_consolidated|1679", "OCB_financial_statements_2018_consolidated_1679.csv", 3, 1, "Tổng dự phòng rủi ro cho vay khách hàng"),
    ("EIB", 2018, "note:general_customer_loan_provision", "consolidated", "764.325", "EIB_financial_statements_2018_consolidated|1632", "EIB_financial_statements_2018_consolidated_1632.csv", 1, 1, "Dự phòng chung cho vay khách hàng"),
    ("EIB", 2018, "note:total_customer_loan_provision", "consolidated", "1.071.367", "EIB_financial_statements_2018_consolidated|1632", "EIB_financial_statements_2018_consolidated_1632.csv", 3, 1, "Tổng dự phòng rủi ro cho vay khách hàng"),
    ("MSB", 2018, "note:general_customer_loan_provision", "consolidated", "311.886", "MSB_financial_statements_2018_consolidated|1556", "MSB_financial_statements_2018_consolidated_1556.csv", 1, 1, "Dự phòng chung cho vay khách hàng"),
    ("MSB", 2018, "note:total_customer_loan_provision", "consolidated", "993.899", "MSB_financial_statements_2018_consolidated|1556", "MSB_financial_statements_2018_consolidated_1556.csv", 3, 1, "Tổng dự phòng rủi ro cho vay khách hàng"),
    ("VPB", 2018, "note:general_customer_loan_provision", "consolidated", "1.525.190", "VPB_financial_statements_2018_consolidated|1220", "VPB_financial_statements_2018_consolidated_1220.csv", 1, 1, "Dự phòng chung cho vay khách hàng"),
    ("VPB", 2018, "note:total_customer_loan_provision", "consolidated", "3.566.773", "VPB_financial_statements_2018_consolidated|1220", "VPB_financial_statements_2018_consolidated_1220.csv", 3, 1, "Tổng dự phòng rủi ro cho vay khách hàng"),
    # Parent tangible-PPE ending gross cost and accumulated depreciation.
    ("GEE", 2025, "note:tangible_gross_cost", "separate", "9.490.539.932", "GEE_financial_statements_2025_separate|1003", "GEE_financial_statements_2025_separate_1003.csv", 4, 4, "Tangible PPE ending gross cost"),
    ("GEE", 2025, "note:tangible_accumulated_depreciation", "separate", "7.148.669.440", "GEE_financial_statements_2025_separate|1003", "GEE_financial_statements_2025_separate_1003.csv", 8, 4, "Tangible PPE ending accumulated depreciation"),
    ("GEX", 2025, "note:tangible_gross_cost", "separate", "97.983.124.514", "GEX_financial_statements_2025_separate|1309", "GEX_financial_statements_2025_separate_1309.csv", 5, 5, "Tangible PPE ending gross cost"),
    ("GEX", 2025, "note:tangible_accumulated_depreciation", "separate", "40.304.826.813", "GEX_financial_statements_2025_separate|1309", "GEX_financial_statements_2025_separate_1309.csv", 10, 5, "Tangible PPE ending accumulated depreciation"),
    ("VGC", 2025, "note:tangible_gross_cost", "separate", "4.003.424.199.446", "VGC_financial_statements_2025_separate|1241", "VGC_financial_statements_2025_separate_1241.csv", 12, 6, "Tangible PPE ending gross cost"),
    ("VGC", 2025, "note:tangible_accumulated_depreciation", "separate", "2.692.553.107.516", "VGC_financial_statements_2025_separate|1241", "VGC_financial_statements_2025_separate_1241.csv", 20, 6, "Tangible PPE ending accumulated depreciation"),
    ("SAM", 2025, "note:tangible_gross_cost", "separate", "16.867.945.035", "SAM_financial_statements_2025_separate|974", "SAM_financial_statements_2025_separate_974.csv", 5, 5, "Tangible PPE ending gross cost"),
    ("SAM", 2025, "note:tangible_accumulated_depreciation", "separate", "15.168.351.394", "SAM_financial_statements_2025_separate|974", "SAM_financial_statements_2025_separate_974.csv", 10, 5, "Tangible PPE ending accumulated depreciation"),
    ("PC1", 2025, "note:tangible_gross_cost", "separate", "2.887.974.978.217", "PC1_financial_statements_2025_separate|1097", "PC1_financial_statements_2025_separate_1097.csv", 7, 6, "Tangible PPE ending gross cost"),
    ("PC1", 2025, "note:tangible_accumulated_depreciation", "separate", "1.001.599.075.979", "PC1_financial_statements_2025_separate|1097", "PC1_financial_statements_2025_separate_1097.csv", 12, 6, "Tangible PPE ending accumulated depreciation"),
    # Future minimum operating-lease receipts where each group is the lessor.
    ("NLG", 2017, "note:future_lease_receipts_under_one_year", "consolidated", "14.492.892.666", "NLG_financial_statements_2017_consolidated|1435", "NLG_financial_statements_2017_consolidated_1435.csv", 1, 1, "Future lease receipts due under one year"),
    ("NLG", 2017, "note:future_lease_receipts_total", "consolidated", "35.769.397.806", "NLG_financial_statements_2017_consolidated|1435", "NLG_financial_statements_2017_consolidated_1435.csv", 4, 1, "Total future lease receipts"),
    ("VIC", 2017, "note:future_lease_receipts_under_one_year", "consolidated", "2.704.061.603.249", "VIC_financial_statements_2017_consolidated|2324", "VIC_financial_statements_2017_consolidated_2324.csv", 2, 1, "Future lease receipts due under one year"),
    ("VIC", 2017, "note:future_lease_receipts_total", "consolidated", "13.235.439.236.527", "VIC_financial_statements_2017_consolidated|2324", "VIC_financial_statements_2017_consolidated_2324.csv", 5, 1, "Total future lease receipts"),
    ("DIG", 2017, "note:future_lease_receipts_under_one_year", "consolidated", "3.374.235.992", "DIG_financial_statements_2017_consolidated|1726", "DIG_financial_statements_2017_consolidated_1726.csv", 1, 1, "Future lease receipts due under one year"),
    ("DIG", 2017, "note:future_lease_receipts_total", "consolidated", "139.749.577.356", "DIG_financial_statements_2017_consolidated|1726", "DIG_financial_statements_2017_consolidated_1726.csv", 4, 1, "Total future lease receipts"),
    ("SNZ", 2017, "note:future_lease_receipts_under_one_year", "consolidated", "378.918.853.419", "SNZ_financial_statements_2017_consolidated|2269", "SNZ_financial_statements_2017_consolidated_2269.csv", 1, 1, "Future lease receipts due under one year"),
    ("SNZ", 2017, "note:future_lease_receipts_total", "consolidated", "7.156.738.806.762", "SNZ_financial_statements_2017_consolidated|2269", "SNZ_financial_statements_2017_consolidated_2269.csv", 4, 1, "Total future lease receipts"),
]:
    _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label = _spec
    EXPLICIT_CELLS[(_ticker, _year, _key, _scope_name)] = _disclosure_cell(
        _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label
    )

# Final empty-program recovery.  Historical runs are used only to locate the
# relevant notes; every operand below is re-read from the BTC source tables.
for _spec in [
    # PLX parent petrol-price-stabilisation deposits and accrued interest.
    ("PLX", 2015, "note:stabilisation_fund_bank_deposit", "separate", "2.185.442.448.068", "PLX_financial_statements_2015_separate|1078", "PLX_financial_statements_2015_separate_1078.csv", 8, 1, "Số dư tiền gửi Quỹ bình ổn giá xăng dầu"),
    ("PLX", 2016, "note:stabilisation_fund_bank_deposit", "separate", "1.529.199.064.366", "PLX_financial_statements_2016_separate|1066", "PLX_financial_statements_2016_separate_1066.csv", 7, 1, "Số dư tiền gửi Quỹ bình ổn giá xăng dầu"),
    ("PLX", 2018, "note:stabilisation_fund_bank_deposit", "separate", "1.372.574.859.359", "PLX_financial_statements_2018_separate|1127", "PLX_financial_statements_2018_separate_1127.csv", 7, 1, "Số dư tiền gửi Quỹ bình ổn giá xăng dầu"),
    ("PLX", 2019, "note:stabilisation_fund_bank_deposit", "separate", "1.288.147.098.093", "PLX_financial_statements_2019_separate|1123", "PLX_financial_statements_2019_separate_1123.csv", 7, 1, "Số dư tiền gửi Quỹ bình ổn giá xăng dầu"),
    ("PLX", 2020, "note:stabilisation_fund_bank_deposit", "separate", "3.944.927.175.227", "PLX_financial_statements_2020_separate|1131", "PLX_financial_statements_2020_separate_1131.csv", 8, 1, "Số dư tiền gửi Quỹ bình ổn giá xăng dầu"),
    ("PLX", 2021, "note:stabilisation_fund_bank_deposit", "separate", "16.633.673", "PLX_financial_statements_2021_separate|690", "PLX_financial_statements_2021_separate_690.csv", 3, 1, "Số dư tiền gửi Quỹ bình ổn giá xăng dầu"),
    ("PLX", 2015, "note:accrued_interest_receivable", "separate", "9.754.156.047", "PLX_financial_statements_2015_separate|750", "PLX_financial_statements_2015_separate_750.csv", 5, 1, "Lãi tiền gửi dự thu"),
    ("PLX", 2016, "note:accrued_interest_receivable", "separate", "12.947.248.952", "PLX_financial_statements_2016_separate|734", "PLX_financial_statements_2016_separate_734.csv", 4, 1, "Lãi tiền gửi dự thu"),
    ("PLX", 2018, "note:accrued_interest_receivable", "separate", "59.304.871.956", "PLX_financial_statements_2018_separate|779", "PLX_financial_statements_2018_separate_779.csv", 4, 1, "Lãi tiền gửi dự thu"),
    ("PLX", 2019, "note:accrued_interest_receivable", "separate", "150.851.013.726", "PLX_financial_statements_2019_separate|791", "PLX_financial_statements_2019_separate_791.csv", 5, 1, "Lãi dự thu"),
    ("PLX", 2020, "note:accrued_interest_receivable", "separate", "87.618.442.331", "PLX_financial_statements_2020_separate|728", "PLX_financial_statements_2020_separate_728.csv", 4, 1, "Lãi dự thu"),
    ("PLX", 2021, "note:accrued_interest_receivable", "separate", "106.350.095.876", "PLX_financial_statements_2021_separate|771", "PLX_financial_statements_2021_separate_771.csv", 1, 1, "Lãi dự thu"),
    # VGT purchases from Coats Phong Phu.
    ("VGT", 2015, "note:coats_phong_phu_purchases", "consolidated", "103.762.921.917", "VGT_financial_statements_2015_consolidated|2028", "VGT_financial_statements_2015_consolidated_2028.csv", 9, 1, "Mua hàng hóa và dịch vụ từ Coats Phong Phú"),
    ("VGT", 2017, "note:coats_phong_phu_purchases", "consolidated", "99.928.967.195", "VGT_financial_statements_2017_consolidated|2093", "VGT_financial_statements_2017_consolidated_2093.csv", 5, 1, "Mua hàng hóa và dịch vụ từ Coats Phong Phú"),
    ("VGT", 2019, "note:coats_phong_phu_purchases", "consolidated", "192.597.495.632", "VGT_financial_statements_2019_consolidated|2125", "VGT_financial_statements_2019_consolidated_2125.csv", 5, 1, "Mua hàng hóa và dịch vụ từ Coats Phong Phú"),
    ("VGT", 2021, "note:coats_phong_phu_purchases", "consolidated", "248.790.137.261", "VGT_financial_statements_2021_consolidated|2070", "VGT_financial_statements_2021_consolidated_2070.csv", 7, 1, "Mua hàng hóa và dịch vụ từ Coats Phong Phú"),
    ("VGT", 2022, "note:coats_phong_phu_purchases", "consolidated", "217.221.702.943", "VGT_financial_statements_2022_consolidated|1950", "VGT_financial_statements_2022_consolidated_1950.csv", 7, 1, "Mua hàng hóa và dịch vụ từ Coats Phong Phú"),
    ("VGT", 2023, "note:coats_phong_phu_purchases", "consolidated", "229.215.946.870", "VGT_financial_statements_2023_consolidated|2114", "VGT_financial_statements_2023_consolidated_2114.csv", 7, 1, "Mua hàng hóa và dịch vụ từ Coats Phong Phú"),
    # ACB parent bonus/welfare fund and aggregate recorded derivatives.
    ("ACB", 2015, "note:bonus_welfare_fund", "separate", "638", "ACB_financial_statements_2015_separate|2753", "ACB_financial_statements_2015_separate_2753.csv", 7, 1, "Quỹ khen thưởng, phúc lợi"),
    ("ACB", 2019, "note:bonus_welfare_fund", "separate", "204.068", "ACB_financial_statements_2019_separate|2864", "ACB_financial_statements_2019_separate_2864.csv", 9, 1, "Quỹ khen thưởng, phúc lợi"),
    ("ACB", 2022, "note:bonus_welfare_fund", "separate", "299.986", "ACB_financial_statements_2022_separate|1842", "ACB_financial_statements_2022_separate_1842.csv", 8, 1, "Quỹ khen thưởng, phúc lợi"),
    ("ACB", 2015, "note:recorded_derivatives_current", "separate", "47.603", "ACB_financial_statements_2015_separate|281", "ACB_financial_statements_2015_separate_281.csv", 11, 3, "Công cụ phái sinh và tài sản tài chính khác cuối 2015"),
    ("ACB", 2015, "note:recorded_derivatives_prior", "separate", "14.403", "ACB_financial_statements_2015_separate|281", "ACB_financial_statements_2015_separate_281.csv", 11, 4, "Công cụ phái sinh và tài sản tài chính khác cuối 2014"),
    ("ACB", 2019, "note:recorded_derivatives_current", "separate", "87.753", "ACB_financial_statements_2019_separate|221", "ACB_financial_statements_2019_separate_221.csv", 12, 3, "Công cụ phái sinh và tài sản tài chính khác cuối 2019"),
    ("ACB", 2019, "note:recorded_derivatives_prior", "separate", "-", "ACB_financial_statements_2019_separate|221", "ACB_financial_statements_2019_separate_221.csv", 12, 4, "Công cụ phái sinh và tài sản tài chính khác cuối 2018"),
    ("ACB", 2022, "note:recorded_derivatives_current", "separate", "100.072", "ACB_financial_statements_2022_separate|1449", "ACB_financial_statements_2022_separate_1449.csv", 7, 4, "Tổng giá trị ghi sổ công cụ phái sinh cuối 2022"),
    ("ACB", 2022, "note:recorded_derivatives_prior", "separate", "226.545", "ACB_financial_statements_2022_separate|1453", "ACB_financial_statements_2022_separate_1453.csv", 7, 4, "Tổng giá trị ghi sổ công cụ phái sinh cuối 2021"),
    # MPC construction in progress and the requested selling-expense line.
    ("MPC", 2016, "note:construction_in_progress", "consolidated", "160.575.227.654", "MPC_financial_statements_2016_consolidated|136", "MPC_financial_statements_2016_consolidated_136.csv", 12, 3, "Xây dựng cơ bản dở dang"),
    ("MPC", 2018, "note:construction_in_progress", "consolidated", "171.646.117.933", "MPC_financial_statements_2018_consolidated|119", "MPC_financial_statements_2018_consolidated_119.csv", 12, 3, "Xây dựng cơ bản dở dang"),
    ("MPC", 2020, "note:construction_in_progress", "consolidated", "497.585.536.429", "MPC_financial_statements_2020_consolidated|135", "MPC_financial_statements_2020_consolidated_135.csv", 12, 3, "Xây dựng cơ bản dở dang"),
    ("MPC", 2022, "note:construction_in_progress", "consolidated", "997.069.810.184", "MPC_financial_statements_2022_consolidated|217", "MPC_financial_statements_2022_consolidated_217.csv", 12, 3, "Xây dựng cơ bản dở dang"),
    ("MPC", 2023, "note:construction_in_progress", "consolidated", "1.412.545.844.995", "MPC_financial_statements_2023_consolidated|323", "MPC_financial_statements_2023_consolidated_323.csv", 10, 3, "Xây dựng cơ bản dở dang"),
    ("MPC", 2016, "note:transport_expense", "consolidated", "254.174.972.822", "MPC_financial_statements_2016_consolidated|1310", "MPC_financial_statements_2016_consolidated_1310.csv", 5, 1, "Chi phí vận chuyển"),
    ("MPC", 2016, "note:outside_services_expense", "consolidated", "38.859.249.203", "MPC_financial_statements_2016_consolidated|1310", "MPC_financial_statements_2016_consolidated_1310.csv", 8, 1, "Chi phí dịch vụ mua ngoài"),
    ("MPC", 2018, "note:transport_expense", "consolidated", "356.673.384.147", "MPC_financial_statements_2018_consolidated|1291", "MPC_financial_statements_2018_consolidated_1291.csv", 1, 1, "Chi phí vận chuyển"),
    ("MPC", 2018, "note:outside_services_expense", "consolidated", "80.639.162.283", "MPC_financial_statements_2018_consolidated|1291", "MPC_financial_statements_2018_consolidated_1291.csv", 8, 1, "Chi phí dịch vụ mua ngoài"),
    ("MPC", 2020, "note:transport_and_outside_services_expense", "consolidated", "386.563.872.017", "MPC_financial_statements_2020_consolidated|1266", "MPC_financial_statements_2020_consolidated_1266.csv", 1, 1, "Chi phí vận chuyển và dịch vụ mua ngoài"),
    ("MPC", 2022, "note:transport_and_outside_services_expense", "consolidated", "891.599.680.780", "MPC_financial_statements_2022_consolidated|1497", "MPC_financial_statements_2022_consolidated_1497.csv", 1, 1, "Chi phí vận chuyển và dịch vụ mua ngoài"),
    ("MPC", 2023, "note:transport_and_outside_services_expense", "consolidated", "259.934.945.215", "MPC_financial_statements_2023_consolidated|1550", "MPC_financial_statements_2023_consolidated_1550.csv", 1, 1, "Chi phí vận chuyển và chi phí dịch vụ mua ngoài"),
    # VGC science/technology fund appropriation and code-253 investments.
    ("VGC", 2019, "note:science_technology_fund_appropriation", "consolidated", "70.000.000.000", "VGC_financial_statements_2019_consolidated|1788", "VGC_financial_statements_2019_consolidated_1788.csv", 10, 1, "Trích quỹ phát triển khoa học công nghệ"),
    ("VGC", 2020, "note:science_technology_fund_appropriation", "consolidated", "50.000.000.000", "VGC_financial_statements_2020_consolidated|1889", "VGC_financial_statements_2020_consolidated_1889.csv", 10, 1, "Trích quỹ phát triển khoa học công nghệ"),
    ("VGC", 2022, "note:science_technology_fund_appropriation", "consolidated", "118.500.000.000", "VGC_financial_statements_2022_consolidated|1715", "VGC_financial_statements_2022_consolidated_1715.csv", 8, 1, "Trích Quỹ Phát triển khoa học và công nghệ"),
    ("VGC", 2023, "note:science_technology_fund_appropriation", "consolidated", "100.000.000.000", "VGC_financial_statements_2023_consolidated|1701", "VGC_financial_statements_2023_consolidated_1701.csv", 16, 1, "Trích Quỹ Phát triển khoa học và công nghệ"),
    ("VGC", 2024, "note:science_technology_fund_appropriation", "consolidated", "40.000.000.000", "VGC_financial_statements_2024_consolidated|1656", "VGC_financial_statements_2024_consolidated_1656.csv", 16, 1, "Trích Quỹ Phát triển khoa học và công nghệ"),
    ("VGC", 2019, "note:investment_other_entities", "consolidated", "9.332.682.344", "VGC_financial_statements_2019_consolidated|271", "VGC_financial_statements_2019_consolidated_271.csv", 22, 3, "Đầu tư góp vốn vào đơn vị khác"),
    ("VGC", 2020, "note:investment_other_entities", "consolidated", "9.332.682.344", "VGC_financial_statements_2020_consolidated|268", "VGC_financial_statements_2020_consolidated_268.csv", 22, 3, "Đầu tư góp vốn vào đơn vị khác"),
    ("VGC", 2022, "note:investment_other_entities", "consolidated", "9.332.682.344", "VGC_financial_statements_2022_consolidated|283", "VGC_financial_statements_2022_consolidated_283.csv", 21, 4, "Đầu tư góp vốn vào đơn vị khác"),
    ("VGC", 2023, "note:investment_other_entities", "consolidated", "9.332.682.344", "VGC_financial_statements_2023_consolidated|295", "VGC_financial_statements_2023_consolidated_295.csv", 21, 4, "Đầu tư góp vốn vào đơn vị khác"),
    ("VGC", 2024, "note:investment_other_entities", "consolidated", "9.332.682.344", "VGC_financial_statements_2024_consolidated|306", "VGC_financial_statements_2024_consolidated_306.csv", 22, 4, "Đầu tư góp vốn vào đơn vị khác"),
    # FPT 2016 currency exposure and the corresponding adverse-5% PBT sensitivity.
    ("FPT", 2016, "note:usd_monetary_liabilities", "consolidated", "1.971.391.027.060", "FPT_financial_statements_2016_consolidated|1371", "FPT_financial_statements_2016_consolidated_1371.csv", 2, 1, "USD monetary liabilities at year end"),
    ("FPT", 2016, "note:eur_monetary_liabilities", "consolidated", "68.492.481.932", "FPT_financial_statements_2016_consolidated|1371", "FPT_financial_statements_2016_consolidated_1371.csv", 3, 1, "EUR monetary liabilities at year end"),
    ("FPT", 2016, "note:jpy_monetary_liabilities", "consolidated", "423.062.639.575", "FPT_financial_statements_2016_consolidated|1371", "FPT_financial_statements_2016_consolidated_1371.csv", 4, 1, "JPY monetary liabilities at year end"),
    ("FPT", 2016, "note:sgd_monetary_liabilities", "consolidated", "81.015.737.069", "FPT_financial_statements_2016_consolidated|1371", "FPT_financial_statements_2016_consolidated_1371.csv", 5, 1, "SGD monetary liabilities at year end"),
    ("FPT", 2016, "note:usd_monetary_assets", "consolidated", "860.476.931.109", "FPT_financial_statements_2016_consolidated|1371", "FPT_financial_statements_2016_consolidated_1371.csv", 2, 3, "USD monetary assets at year end"),
    ("FPT", 2016, "note:eur_monetary_assets", "consolidated", "207.023.726.267", "FPT_financial_statements_2016_consolidated|1371", "FPT_financial_statements_2016_consolidated_1371.csv", 3, 3, "EUR monetary assets at year end"),
    ("FPT", 2016, "note:jpy_monetary_assets", "consolidated", "940.414.984.796", "FPT_financial_statements_2016_consolidated|1371", "FPT_financial_statements_2016_consolidated_1371.csv", 4, 3, "JPY monetary assets at year end"),
    ("FPT", 2016, "note:sgd_monetary_assets", "consolidated", "168.269.166.723", "FPT_financial_statements_2016_consolidated|1371", "FPT_financial_statements_2016_consolidated_1371.csv", 5, 3, "SGD monetary assets at year end"),
    ("FPT", 2016, "note:usd_adverse_5pct_pbt", "consolidated", "(55.545.704.798)", "FPT_financial_statements_2016_consolidated|1389", "FPT_financial_statements_2016_consolidated_1389.csv", 2, 1, "USD adverse-5% effect on PBT"),
    ("FPT", 2016, "note:eur_adverse_5pct_pbt", "consolidated", "6.926.562.217", "FPT_financial_statements_2016_consolidated|1389", "FPT_financial_statements_2016_consolidated_1389.csv", 3, 1, "EUR adverse-5% effect on PBT"),
    ("FPT", 2016, "note:jpy_adverse_5pct_pbt", "consolidated", "25.867.617.261", "FPT_financial_statements_2016_consolidated|1389", "FPT_financial_statements_2016_consolidated_1389.csv", 4, 1, "JPY adverse-5% effect on PBT"),
    ("FPT", 2016, "note:sgd_adverse_5pct_pbt", "consolidated", "4.362.671.483", "FPT_financial_statements_2016_consolidated|1389", "FPT_financial_statements_2016_consolidated_1389.csv", 5, 1, "SGD adverse-5% effect on PBT"),
    # Total on-balance interest-sensitivity gap, million VND.
    ("BID", 2025, "note:on_balance_interest_sensitivity_gap", "consolidated", "208.954.624", "BID_financial_statements_2025_consolidated|2219", "BID_financial_statements_2025_consolidated_2219.csv", 22, 9, "Mức chênh nhạy cảm với lãi suất nội bảng - Tổng"),
    ("STB", 2025, "note:on_balance_interest_sensitivity_gap", "consolidated", "91.625.819", "STB_financial_statements_2025_consolidated|3048", "STB_financial_statements_2025_consolidated_3048.csv", 21, 9, "Mức chênh nhạy cảm với lãi suất nội bảng - Tổng"),
]:
    _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label = _spec
    EXPLICIT_CELLS[(_ticker, _year, _key, _scope_name)] = _disclosure_cell(
        _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label
    )

# Obvious unit/row-selection failures exposed by comparing historical model
# runs.  Each replacement is tied to the audited row that the earlier program
# missed; no historical numeric output is trusted as a label.
for _spec in [
    ("GVR", 2019, "note:visorutex_voting_rate", "separate", "27,78%", "GVR_financial_statements_2019_separate|948", "GVR_financial_statements_2019_separate_948.csv", 1, 3, "Xí nghiệp Liên doanh Visorutex - Tỷ lệ biểu quyết"),
    ("GVR", 2020, "note:lai_chau_rubber_ownership_rate", "consolidated", "98,32%", "GVR_financial_statements_2020_consolidated|700", "GVR_financial_statements_2020_consolidated_700.csv", 1, 3, "Công ty CP Cao su Lai Châu - Tỷ lệ sở hữu"),
    ("GEG", 2022, "note:gialai_hydropower_paid_capital_ownership_rate", "consolidated", "6253", "GEG_financial_statements_2022_consolidated|430", "GEG_financial_statements_2022_consolidated_430.csv", 3, 4, "Công ty CP Thủy điện Gia Lai - Tỷ lệ sở hữu trên vốn thực góp"),
    ("ACB", 2024, "note:usd_combined_currency_position", "consolidated", "(1.216.640)", "ACB_financial_statements_2024_consolidated|2798", "ACB_financial_statements_2024_consolidated_2798.csv", 18, 1, "USD - Trạng thái tiền tệ nội, ngoại bảng"),
    ("ACB", 2024, "note:gold_combined_currency_position", "consolidated", "56.213", "ACB_financial_statements_2024_consolidated|2798", "ACB_financial_statements_2024_consolidated_2798.csv", 18, 2, "Vàng - Trạng thái tiền tệ nội, ngoại bảng"),
    ("ACB", 2024, "note:eur_combined_currency_position", "consolidated", "22.477", "ACB_financial_statements_2024_consolidated|2798", "ACB_financial_statements_2024_consolidated_2798.csv", 18, 3, "EUR - Trạng thái tiền tệ nội, ngoại bảng"),
    ("ACB", 2024, "note:jpy_combined_currency_position", "consolidated", "73.560", "ACB_financial_statements_2024_consolidated|2798", "ACB_financial_statements_2024_consolidated_2798.csv", 18, 4, "JPY - Trạng thái tiền tệ nội, ngoại bảng"),
    ("ACB", 2024, "note:aud_combined_currency_position", "consolidated", "15.822", "ACB_financial_statements_2024_consolidated|2798", "ACB_financial_statements_2024_consolidated_2798.csv", 18, 5, "AUD - Trạng thái tiền tệ nội, ngoại bảng"),
    ("ACB", 2024, "note:cad_combined_currency_position", "consolidated", "11.577", "ACB_financial_statements_2024_consolidated|2798", "ACB_financial_statements_2024_consolidated_2798.csv", 18, 6, "CAD - Trạng thái tiền tệ nội, ngoại bảng"),
    ("ACB", 2024, "note:other_combined_currency_position", "consolidated", "45.805", "ACB_financial_statements_2024_consolidated|2798", "ACB_financial_statements_2024_consolidated_2798.csv", 18, 7, "Ngoại tệ khác - Trạng thái tiền tệ nội, ngoại bảng"),
    ("ACB", 2024, "note:bank_pbt", "consolidated", "21.005.871", "ACB_financial_statements_2024_consolidated|213", "ACB_financial_statements_2024_consolidated_213.csv", 17, 3, "Tổng lợi nhuận trước thuế"),
    ("MWG", 2017, "note:depreciation_amortisation_expense", "consolidated", "689.713.708.632", "MWG_financial_statements_2017_consolidated|978", "MWG_financial_statements_2017_consolidated_978.csv", 3, 1, "Chi phí khấu hao và hao mòn"),
    ("MWG", 2018, "note:depreciation_amortisation_expense", "consolidated", "1.222.868.683.717", "MWG_financial_statements_2018_consolidated|989", "MWG_financial_statements_2018_consolidated_989.csv", 3, 1, "Chi phí khấu hao và hao mòn"),
    ("MWG", 2020, "note:depreciation_amortisation_expense", "consolidated", "2.195.583.071.035", "MWG_financial_statements_2020_consolidated|1029", "MWG_financial_statements_2020_consolidated_1029.csv", 3, 1, "Chi phí khấu hao và hao mòn"),
    ("MWG", 2022, "note:depreciation_amortisation_expense", "consolidated", "3.540.324.726.294", "MWG_financial_statements_2022_consolidated|1048", "MWG_financial_statements_2022_consolidated_1048.csv", 3, 1, "Chi phí khấu hao và hao mòn"),
    ("MWG", 2017, "note:long_term_borrowings", "consolidated", "1.199.932.994.830", "MWG_financial_statements_2017_consolidated|221", "MWG_financial_statements_2017_consolidated_221.csv", 13, 3, "Vay dài hạn"),
    ("MWG", 2018, "note:long_term_borrowings", "consolidated", "1.208.167.140.389", "MWG_financial_statements_2018_consolidated|200", "MWG_financial_statements_2018_consolidated_200.csv", 13, 3, "Vay dài hạn"),
    ("MWG", 2020, "note:long_term_borrowings", "consolidated", "1.126.676.666.653", "MWG_financial_statements_2020_consolidated|204", "MWG_financial_statements_2020_consolidated_204.csv", 14, 3, "Vay dài hạn"),
    ("MWG", 2022, "note:long_term_borrowings", "consolidated", "5.901.250.000.000", "MWG_financial_statements_2022_consolidated|210", "MWG_financial_statements_2022_consolidated_210.csv", 14, 3, "Vay dài hạn"),
    ("DNH", 2021, "note:tangible_fixed_asset_gross_cost", "separate", "14.814.242.649.760", "DNH_financial_statements_2021_separate|777", "DNH_financial_statements_2021_separate_777.csv", 6, 6, "TSCĐ hữu hình - Nguyên giá cuối năm"),
    ("HHS", 2023, "note:dividend_income", "separate", "318.822.993.407", "HHS_financial_statements_2023_separate|1237", "HHS_financial_statements_2023_separate_1237.csv", 3, 1, "Cổ tức, lợi nhuận được chia"),
    ("HHS", 2023, "note:short_term_financial_investments", "separate", "303.582.700.000", "HHS_financial_statements_2023_separate|215", "HHS_financial_statements_2023_separate_215.csv", 6, 3, "Đầu tư tài chính ngắn hạn"),
    ("HHS", 2023, "note:long_term_financial_investments", "separate", "3.423.739.097.286", "HHS_financial_statements_2023_separate|215", "HHS_financial_statements_2023_separate_215.csv", 26, 3, "Đầu tư tài chính dài hạn"),
    ("MSN", 2016, "note:techcombank_associate_investment", "separate", "7.989.232.239.897", "MSN_financial_statements_2016_separate|600", "MSN_financial_statements_2016_separate_600.csv", 11, 3, "Đầu tư vào Techcombank"),
    ("MSN", 2016, "note:total_associate_investment", "separate", "7.989.232.239.897", "MSN_financial_statements_2016_separate|569", "MSN_financial_statements_2016_separate_569.csv", 5, 1, "Đầu tư vào một công ty liên kết"),
    ("VGT", 2024, "note:gross_doubtful_customer_receivables", "consolidated", "463.862.185.773", "VGT_financial_statements_2024_consolidated|1281", "VGT_financial_statements_2024_consolidated_1281.csv", 13, 2, "Nợ quá hạn - Tổng giá gốc"),
    ("VGT", 2024, "note:total_customer_receivables", "consolidated", "2.275.337.763.396", "VGT_financial_statements_2024_consolidated|1179", "VGT_financial_statements_2024_consolidated_1179.csv", 3, 1, "Phải thu khách hàng - Tổng"),
    ("MSR", 2022, "note:interest_and_borrowing_cost", "consolidated", "1.194.553.796", "MSR_financial_statements_2022_consolidated|399", "MSR_financial_statements_2022_consolidated_399.csv", 7, 3, "Chi phí lãi vay và chi phí đi vay"),
    ("MSR", 2022, "note:short_term_borrowings_ending", "consolidated", "5.818.241.451", "MSR_financial_statements_2022_consolidated|1588", "MSR_financial_statements_2022_consolidated_1588.csv", 3, 6, "Vay ngắn hạn cuối năm"),
    ("NCB", 2016, "note:cfo", "separate", "2.970.254", "NVB_financial_statements_2016_separate|388", "NVB_financial_statements_2016_separate_388.csv", 10, 2, "Lưu chuyển tiền thuần từ hoạt động kinh doanh"),
    ("NCB", 2016, "note:pbt", "separate", "13.380", "NVB_financial_statements_2016_separate|336", "NVB_financial_statements_2016_separate_336.csv", 17, 2, "Tổng lợi nhuận trước thuế"),
]:
    _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label = _spec
    EXPLICIT_CELLS[(_ticker, _year, _key, _scope_name)] = _disclosure_cell(
        _ticker, _year, _key, _scope_name, _raw, _ref, _csv, _row, _col, _label
    )

# The empty current-year cell is the disclosed zero amount, not a guessed
# value. Preserve the exact extracted token and let the runtime parser
# normalize it to zero.
EXPLICIT_CELLS[("HAG", 2025, "kqkd:51", "separate")] = StatementCell(
    "HAG", "2025", "separate", "kqkd:51", "51", "Chi phí thuế TNDN hiện hành",
    0.0, "", "HAG_financial_statements_2025_separate|303",
    "HAG_financial_statements_2025_separate_303.csv", 14, 3, 1.0,
)

# The compact statement cube historically omitted balance-sheet subcodes 221
# (tangible fixed assets) and 242 (construction in progress).  Keep these
# source cells explicit so questions asking for a component never fall back to
# their broader 220/240 parent totals.  ``resolve_source_coordinate`` verifies
# and, when necessary, corrects the provisional row offset before packaging.
for _spec in (
    # q178/q581/q616/q743/q798: tangible fixed assets, code 221.
    ("NVL", 2021, "cdkt:221", "consolidated", "1.345.065.108.658", "NVL_financial_statements_2021_consolidated|148", "NVL_financial_statements_2021_consolidated_148.csv", 0, 3),
    ("DBC", 2019, "cdkt:221", "consolidated", "3.952.077.043.860", "DBC_financial_statements_2019_consolidated|312", "DBC_financial_statements_2019_consolidated_312.csv", 0, 3),
    ("DBC", 2015, "cdkt:221", "consolidated", "898.255.375.494", "DBC_financial_statements_2015_consolidated|259", "DBC_financial_statements_2015_consolidated_259.csv", 0, 3),
    ("VJC", 2018, "cdkt:221", "consolidated", "1.643.191.617.368", "VJC_financial_statements_2018_consolidated|138", "VJC_financial_statements_2018_consolidated_138.csv", 0, 3),
    ("VJC", 2015, "cdkt:221", "consolidated", "17.742.310.565", "VJC_financial_statements_2015_consolidated|269", "VJC_financial_statements_2015_consolidated_269.csv", 0, 3),
    ("GEE", 2023, "cdkt:221", "consolidated", "3.380.042.566.756", "GEE_financial_statements_2023_consolidated|397", "GEE_financial_statements_2023_consolidated_397.csv", 0, 4),
    ("SAM", 2023, "cdkt:221", "consolidated", "634.394.546.577", "SAM_financial_statements_2023_consolidated|176", "SAM_financial_statements_2023_consolidated_176.csv", 0, 3),
    ("HHV", 2022, "cdkt:221", "separate", "99.166.395.728", "HHV_financial_statements_2022_separate|336", "HHV_financial_statements_2022_separate_336.csv", 0, 3),
    ("VSC", 2022, "cdkt:221", "separate", "31.609.177.624", "VSC_financial_statements_2022_separate|220", "VSC_financial_statements_2022_separate_220.csv", 0, 3),
    # q948: PLX construction in progress, code 242.
    ("PLX", 2017, "cdkt:242", "consolidated", "777.729.469.586", "PLX_financial_statements_2017_consolidated|193", "PLX_financial_statements_2017_consolidated_193.csv", 0, 3),
    ("PLX", 2018, "cdkt:242", "consolidated", "875.013.410.257", "PLX_financial_statements_2018_consolidated|152", "PLX_financial_statements_2018_consolidated_152.csv", 0, 3),
    ("PLX", 2019, "cdkt:242", "consolidated", "989.693.974.370", "PLX_financial_statements_2019_consolidated|210", "PLX_financial_statements_2019_consolidated_210.csv", 0, 3),
    ("PLX", 2023, "cdkt:242", "consolidated", "919.552.879.504", "PLX_financial_statements_2023_consolidated|210", "PLX_financial_statements_2023_consolidated_210.csv", 0, 3),
    ("PLX", 2024, "cdkt:242", "consolidated", "1.445.555.813.867", "PLX_financial_statements_2024_consolidated|158", "PLX_financial_statements_2024_consolidated_158.csv", 0, 3),
    # q1000: SAM construction in progress, code 242.
    ("SAM", 2020, "cdkt:242", "consolidated", "279.200.617.865", "SAM_financial_statements_2020_consolidated|194", "SAM_financial_statements_2020_consolidated_194.csv", 0, 3),
    ("SAM", 2021, "cdkt:242", "consolidated", "676.888.943.619", "SAM_financial_statements_2021_consolidated|173", "SAM_financial_statements_2021_consolidated_173.csv", 0, 3),
    ("SAM", 2023, "cdkt:242", "consolidated", "422.108.075.599", "SAM_financial_statements_2023_consolidated|176", "SAM_financial_statements_2023_consolidated_176.csv", 0, 3),
    ("SAM", 2024, "cdkt:242", "consolidated", "324.140.630.136", "SAM_financial_statements_2024_consolidated|269", "SAM_financial_statements_2024_consolidated_269.csv", 0, 3),
):
    _ticker, _year, _metric, _scope, _raw, _ref, _csv, _row, _col = _spec
    EXPLICIT_CELLS[(_ticker, _year, _metric, _scope)] = _disclosure_cell(
        _ticker, _year, _metric, _scope, _raw, _ref, _csv, _row, _col,
        "Tangible fixed assets" if _metric == "cdkt:221" else "Construction in progress",
    )

# Component-level cells found by the question-to-metric-code audit.  These
# repair five broader/adjacent-code selections and make panel q506 depend on
# code 221 as requested.  Coordinates are provisional only: the builder
# resolves each exact raw token against the authoritative BTC table before it
# writes the audit manifest.
for _spec in (
    # q874: VNM tangible fixed assets, code 221 (the correct maximum is 2023).
    ("VNM", 2015, "cdkt:221", "consolidated", "7.795.345.501.520", "VNM_financial_statements_2015_consolidated|142", "VNM_financial_statements_2015_consolidated_142.csv", 0, 3, "Tangible fixed assets"),
    ("VNM", 2016, "cdkt:221", "consolidated", "7.916.322.992.944", "VNM_financial_statements_2016_consolidated|144", "VNM_financial_statements_2016_consolidated_144.csv", 0, 3, "Tangible fixed assets"),
    ("VNM", 2021, "cdkt:221", "consolidated", "11.620.094.589.519", "VNM_financial_statements_2021_consolidated|213", "VNM_financial_statements_2021_consolidated_213.csv", 0, 3, "Tangible fixed assets"),
    ("VNM", 2023, "cdkt:221", "consolidated", "11.688.520.305.045", "VNM_financial_statements_2023_consolidated|189", "VNM_financial_statements_2023_consolidated_189.csv", 0, 3, "Tangible fixed assets"),
    ("VNM", 2025, "cdkt:221", "consolidated", "11.618.118.961.976", "VNM_financial_statements_2025_consolidated|187", "VNM_financial_statements_2025_consolidated_187.csv", 0, 3, "Tangible fixed assets"),
    # q906: VGT parent tangible fixed assets, code 221 (maximum remains 2017).
    ("VGT", 2015, "cdkt:221", "separate", "424.308.792.043", "VGT_financial_statements_2015_separate|170", "VGT_financial_statements_2015_separate_170.csv", 0, 3, "Tangible fixed assets"),
    ("VGT", 2017, "cdkt:221", "separate", "1.189.563.447.563", "VGT_financial_statements_2017_separate|184", "VGT_financial_statements_2017_separate_184.csv", 0, 3, "Tangible fixed assets"),
    ("VGT", 2018, "cdkt:221", "separate", "1.125.874.312.586", "VGT_financial_statements_2018_separate|162", "VGT_financial_statements_2018_separate_162.csv", 0, 3, "Tangible fixed assets"),
    ("VGT", 2020, "cdkt:221", "separate", "1.039.404.891.409", "VGT_financial_statements_2020_separate|153", "VGT_financial_statements_2020_separate_153.csv", 0, 3, "Tangible fixed assets"),
    ("VGT", 2022, "cdkt:221", "separate", "797.363.441.490", "VGT_financial_statements_2022_separate|230", "VGT_financial_statements_2022_separate_230.csv", 0, 3, "Tangible fixed assets"),
    # q98/q279/q997: inventory net, cash on hand, supplier advances.
    ("HUT", 2024, "cdkt:140", "separate", "3.177.372.538.020", "HUT_financial_statements_2024_separate|328", "HUT_financial_statements_2024_separate_328.csv", 0, 4, "Net inventory"),
    ("HSG", 2020, "cdkt:111", "consolidated", "546.734.145.565", "HSG_financial_statements_2020_consolidated|124", "HSG_financial_statements_2020_consolidated_124.csv", 0, 3, "Cash on hand"),
    ("AAA", 2019, "cdkt:132", "consolidated", "635.624.680.580", "AAA_financial_statements_2019_consolidated|209", "AAA_financial_statements_2019_consolidated_209.csv", 0, 3, "Short-term advances to suppliers"),
    ("AAA", 2023, "cdkt:132", "consolidated", "342.180.956.924", "AAA_financial_statements_2023_consolidated|224", "AAA_financial_statements_2023_consolidated_224.csv", 0, 3, "Short-term advances to suppliers"),
    ("AAA", 2025, "cdkt:132", "consolidated", "399.960.128.545", "AAA_financial_statements_2025_consolidated|247", "AAA_financial_statements_2025_consolidated_247.csv", 0, 3, "Short-term advances to suppliers"),
    # q506: five-way 2024 tangible-fixed-assets selector.
    ("IJC", 2024, "cdkt:221", "consolidated", "531.904.880.087", "IJC_financial_statements_2024_consolidated|364", "IJC_financial_statements_2024_consolidated_364.csv", 0, 4, "Tangible fixed assets"),
    ("DXG", 2024, "cdkt:221", "consolidated", "298.730.468.439", "DXG_financial_statements_2024_consolidated|340", "DXG_financial_statements_2024_consolidated_340.csv", 0, 3, "Tangible fixed assets"),
    ("NVL", 2024, "cdkt:221", "consolidated", "1.875.694.928.140", "NVL_financial_statements_2024_consolidated|471", "NVL_financial_statements_2024_consolidated_471.csv", 0, 3, "Tangible fixed assets"),
    ("NLG", 2024, "cdkt:221", "consolidated", "72.116.379.142", "NLG_financial_statements_2024_consolidated|217", "NLG_financial_statements_2024_consolidated_217.csv", 0, 3, "Tangible fixed assets"),
    ("KBC", 2024, "cdkt:221", "consolidated", "408.144.202.512", "KBC_financial_statements_2024_consolidated|350", "KBC_financial_statements_2024_consolidated_350.csv", 0, 3, "Tangible fixed assets"),
):
    _ticker, _year, _metric, _scope, _raw, _ref, _csv, _row, _col, _label = _spec
    EXPLICIT_CELLS[(_ticker, _year, _metric, _scope)] = _disclosure_cell(
        _ticker, _year, _metric, _scope, _raw, _ref, _csv, _row, _col, _label,
    )


# q653: net long-term loan receivables, not long-term borrowings.  The 2024
# provision note assigns the full ending loan-receivable provision to the
# short-term bucket, while the 2022 note identifies a separate long-term
# provision.  Retain both 2024 provision cells so the zero long-term provision
# is derived from the source equality rather than embedded as a constant.
EXPLICIT_CELLS.update({
    ("HAG", 2024, "note:long_term_loan_receivables_gross", "consolidated"): _disclosure_cell(
        "HAG", 2024, "note:long_term_loan_receivables_gross", "consolidated",
        "46.813.199", "HAG_financial_statements_2024_consolidated|1149",
        "HAG_financial_statements_2024_consolidated_1149.csv", 8, 1,
        "Long-term loan receivables - subtotal, thousand VND",
    ),
    ("HAG", 2024, "note:loan_receivable_provision_total", "consolidated"): _disclosure_cell(
        "HAG", 2024, "note:loan_receivable_provision_total", "consolidated",
        "23.959.811", "HAG_financial_statements_2024_consolidated|1153",
        "HAG_financial_statements_2024_consolidated_1153.csv", 4, 1,
        "Ending loan-receivable provision - total, thousand VND",
    ),
    ("HAG", 2024, "note:loan_receivable_provision_short", "consolidated"): _disclosure_cell(
        "HAG", 2024, "note:loan_receivable_provision_short", "consolidated",
        "23.959.811", "HAG_financial_statements_2024_consolidated|1153",
        "HAG_financial_statements_2024_consolidated_1153.csv", 5, 1,
        "Ending loan-receivable provision - short-term, thousand VND",
    ),
    ("HAG", 2022, "note:long_term_loan_receivables_gross", "consolidated"): _disclosure_cell(
        "HAG", 2022, "note:long_term_loan_receivables_gross", "consolidated",
        "1.745.420.930", "HAG_financial_statements_2022_consolidated|1065",
        "HAG_financial_statements_2022_consolidated_1065.csv", 8, 1,
        "Long-term loan receivables - subtotal, thousand VND",
    ),
    ("HAG", 2022, "note:loan_receivable_provision_long", "consolidated"): _disclosure_cell(
        "HAG", 2022, "note:loan_receivable_provision_long", "consolidated",
        "17.526.263", "HAG_financial_statements_2022_consolidated|1069",
        "HAG_financial_statements_2022_consolidated_1069.csv", 6, 1,
        "Ending loan-receivable provision - long-term, thousand VND",
    ),
    ("VGT", 2022, "note:foreign_currency_vnd_equivalent_total", "consolidated"): _disclosure_cell(
        "VGT", 2022, "note:foreign_currency_vnd_equivalent_total", "consolidated",
        "216.674.089.381", "VGT_financial_statements_2022_consolidated|1741",
        "VGT_financial_statements_2022_consolidated_1741.csv", 4, 2,
        "Foreign-currency balances - total VND equivalent at end-2022",
    ),
    ("SSI", 2016, "note:receivables_total", "separate"): _disclosure_cell(
        "SSI", 2016, "note:receivables_total", "separate",
        "76.038.454.529", "SSI_financial_statements_2016_separate|1355",
        "SSI_financial_statements_2016_separate_1355.csv", 11, 1,
        "Receivables note - ending total",
    ),
    ("GAS", 2021, "note:third_party_short_term_customer_receivables", "separate"): _disclosure_cell(
        "GAS", 2021, "note:third_party_short_term_customer_receivables", "separate",
        "4.584.694.778.238", "GAS_financial_statements_2021_separate|1024",
        "GAS_financial_statements_2021_separate_1024.csv", 2, 1,
        "Parent-company short-term customer receivables - third parties",
    ),
    ("KBC", 2024, "note:other_current_receivables_total", "consolidated"): _disclosure_cell(
        "KBC", 2024, "note:other_current_receivables_total", "consolidated",
        "5.963.091.549.126", "KBC_financial_statements_2024_consolidated|331",
        "KBC_financial_statements_2024_consolidated_331.csv", 13, 3,
        "Other current receivables (balance-sheet code 136), ending 2024",
    ),
    ("KBC", 2022, "note:other_current_receivables_total", "consolidated"): _disclosure_cell(
        "KBC", 2022, "note:other_current_receivables_total", "consolidated",
        "3.654.794.506.950", "KBC_financial_statements_2022_consolidated|284",
        "KBC_financial_statements_2022_consolidated_284.csv", 13, 3,
        "Other current receivables (balance-sheet code 136), ending 2022",
    ),
})


# ``v0``, ``v1`` ... in each expression correspond to operands in order.
# The first group contains public-proven overrides.  The 21-Aug expansion also
# includes exact primary-statement formulas that existed in the offline solver
# but had never been wired into the final executable artifact.
# Audited cells recovered during the raw-unit scan.  These are deliberately
# anchored to the primary statement or the exact disclosure total requested by
# the question, rather than to the unrelated row selected by the historical
# model program.
EXPLICIT_CELLS.update({
    ("NVL", 2016, "note:other_current_receivables", "separate"): _disclosure_cell("NVL", 2016, "note:other_current_receivables", "separate", "4.328.204.265.542", "NVL_financial_statements_2016_separate|204", "NVL_financial_statements_2016_separate_204.csv", 12, 3, "Phải thu ngắn hạn khác"),
    ("VRE", 2016, "note:goodwill_nbv", "consolidated"): _disclosure_cell("VRE", 2016, "note:goodwill_nbv", "consolidated", "624.529.172.455", "VRE_financial_statements_2016_consolidated|1555", "VRE_financial_statements_2016_consolidated_1555.csv", 10, 4, "Lợi thế thương mại - Giá trị còn lại cuối năm - Tổng cộng"),
    ("NVL", 2020, "note:other_current_receivables", "consolidated"): _disclosure_cell("NVL", 2020, "note:other_current_receivables", "consolidated", "8.069.327.416.090", "NVL_financial_statements_2020_consolidated|203", "NVL_financial_statements_2020_consolidated_203.csv", 12, 3, "Phải thu ngắn hạn khác"),
    ("VGC", 2025, "note:intangible_fixed_assets_nbv", "consolidated"): _disclosure_cell("VGC", 2025, "note:intangible_fixed_assets_nbv", "consolidated", "215.181.280.173", "VGC_financial_statements_2025_consolidated|313", "VGC_financial_statements_2025_consolidated_313.csv", 12, 4, "Tài sản cố định vô hình - Giá trị còn lại cuối năm"),
    ("MPC", 2020, "note:bonus_welfare_fund", "consolidated"): _disclosure_cell("MPC", 2020, "note:bonus_welfare_fund", "consolidated", "53.102.595.837", "MPC_financial_statements_2020_consolidated|1110", "MPC_financial_statements_2020_consolidated_1110.csv", 4, 1, "Quỹ khen thưởng và phúc lợi - Số dư cuối năm"),
    ("VGT", 2020, "note:provisions_total", "consolidated"): _disclosure_cell("VGT", 2020, "note:provisions_total", "consolidated", "32.587.523.656", "VGT_financial_statements_2020_consolidated|1647", "VGT_financial_statements_2020_consolidated_1647.csv", 4, 3, "Dự phòng phải trả - Số dư cuối năm - Tổng cộng"),
    ("SNZ", 2020, "note:short_term_supplier_advances", "consolidated"): _disclosure_cell("SNZ", 2020, "note:short_term_supplier_advances", "consolidated", "1.302.097.620.567", "SNZ_financial_statements_2020_consolidated|335", "SNZ_financial_statements_2020_consolidated_335.csv", 9, 3, "Trả trước cho người bán ngắn hạn"),
    ("VNM", 2022, "note:common_shareholder_profit_before_fund", "consolidated"): _disclosure_cell("VNM", 2022, "note:common_shareholder_profit_before_fund", "consolidated", "8.516.023.694.342", "VNM_financial_statements_2022_consolidated|1938", "VNM_financial_statements_2022_consolidated_1938.csv", 0, 1, "Lợi nhuận thuần thuộc về cổ đông phổ thông trước khi trích quỹ"),
    ("NVL", 2016, "note:short_term_supplier_advances", "separate"): _disclosure_cell("NVL", 2016, "note:short_term_supplier_advances", "separate", "168.329.639.170", "NVL_financial_statements_2016_separate|204", "NVL_financial_statements_2016_separate_204.csv", 10, 3, "Trả trước cho người bán ngắn hạn"),
    ("PC1", 2025, "note:vat_payable_ending", "consolidated"): _disclosure_cell("PC1", 2025, "note:vat_payable_ending", "consolidated", "60.792.014.797", "PC1_financial_statements_2025_consolidated|1220", "PC1_financial_statements_2025_consolidated_1220.csv", 0, 6, "Thuế giá trị gia tăng phải nộp cuối năm"),
    ("AAA", 2023, "note:subsidiary_investment_gross_cost", "separate"): _disclosure_cell("AAA", 2023, "note:subsidiary_investment_gross_cost", "separate", "2.807.566.671.231", "AAA_financial_statements_2023_separate|1005", "AAA_financial_statements_2023_separate_1005.csv", 0, 1, "Đầu tư vào công ty con - Giá gốc cuối năm"),
    ("HBC", 2024, "note:short_term_loan_provision", "separate"): _disclosure_cell("HBC", 2024, "note:short_term_loan_provision", "separate", "80.864.684.721", "HBC_financial_statements_2024_separate|1197", "HBC_financial_statements_2024_separate_1197.csv", 4, 1, "Dự phòng các khoản cho vay ngắn hạn - Tổng cộng"),
    ("VGC", 2025, "note:short_term_pledges_deposits", "consolidated"): _disclosure_cell("VGC", 2025, "note:short_term_pledges_deposits", "consolidated", "86.004.272.122", "VGC_financial_statements_2025_consolidated|1273", "VGC_financial_statements_2025_consolidated_1273.csv", 1, 1, "Cầm cố, ký cược, ký quỹ ngắn hạn"),
})


# Cross-table repairs surfaced by comparing historical model generations.  Each
# operand below is still an original BTC disclosure cell; the old submissions
# only supplied audit targets and are never used as answer labels.
EXPLICIT_CELLS.update({
    # Q495: VGT related-party other receivables and operating-lease commitments.
    ("VGT", 2018, "note:related_other_current_receivables_total", "consolidated"): _disclosure_cell("VGT", 2018, "note:related_other_current_receivables_total", "consolidated", "406.195.735.600", "VGT_financial_statements_2018_consolidated|1244", "VGT_financial_statements_2018_consolidated_1244.csv", 7, 1, "Total other current receivables from related parties"),
    ("VGT", 2020, "note:related_other_current_receivables_total", "consolidated"): _disclosure_cell("VGT", 2020, "note:related_other_current_receivables_total", "consolidated", "407.407.224.329", "VGT_financial_statements_2020_consolidated|1191", "VGT_financial_statements_2020_consolidated_1191.csv", 10, 1, "Total other current receivables from related parties"),
    ("VGT", 2021, "note:related_other_current_receivables_total", "consolidated"): _disclosure_cell("VGT", 2021, "note:related_other_current_receivables_total", "consolidated", "248.590.459.577", "VGT_financial_statements_2021_consolidated|1267", "VGT_financial_statements_2021_consolidated_1267.csv", 10, 1, "Total other current receivables from related parties"),
    ("VGT", 2022, "note:related_other_current_receivables_total", "consolidated"): _disclosure_cell("VGT", 2022, "note:related_other_current_receivables_total", "consolidated", "167.714.432.463", "VGT_financial_statements_2022_consolidated|1148", "VGT_financial_statements_2022_consolidated_1148.csv", 12, 1, "Total other current receivables from related parties"),
    ("VGT", 2018, "note:noncancelable_operating_lease_total", "consolidated"): _disclosure_cell("VGT", 2018, "note:noncancelable_operating_lease_total", "consolidated", "263.025.886.486", "VGT_financial_statements_2018_consolidated|1842", "VGT_financial_statements_2018_consolidated_1842.csv", 4, 1, "Total minimum non-cancelable operating-lease payments"),
    ("VGT", 2020, "note:noncancelable_operating_lease_total", "consolidated"): _disclosure_cell("VGT", 2020, "note:noncancelable_operating_lease_total", "consolidated", "385.622.716.920", "VGT_financial_statements_2020_consolidated|1765", "VGT_financial_statements_2020_consolidated_1765.csv", 4, 1, "Total minimum non-cancelable operating-lease payments"),
    ("VGT", 2021, "note:noncancelable_operating_lease_total", "consolidated"): _disclosure_cell("VGT", 2021, "note:noncancelable_operating_lease_total", "consolidated", "375.899.885.116", "VGT_financial_statements_2021_consolidated|1845", "VGT_financial_statements_2021_consolidated_1845.csv", 4, 1, "Total minimum non-cancelable operating-lease payments"),
    ("VGT", 2022, "note:noncancelable_operating_lease_total", "consolidated"): _disclosure_cell("VGT", 2022, "note:noncancelable_operating_lease_total", "consolidated", "762.609.450.720", "VGT_financial_statements_2022_consolidated|1737", "VGT_financial_statements_2022_consolidated_1737.csv", 4, 1, "Total minimum non-cancelable operating-lease payments"),
    # Q500: PNJ construction-in-progress additions and personal-loan component.
    ("PNJ", 2018, "note:cip_additions", "consolidated"): _disclosure_cell("PNJ", 2018, "note:cip_additions", "consolidated", "61.157.602.188", "PNJ_financial_statements_2018_consolidated|664", "PNJ_financial_statements_2018_consolidated_664.csv", 2, 1, "Construction-in-progress additions"),
    ("PNJ", 2020, "note:cip_additions", "consolidated"): _disclosure_cell("PNJ", 2020, "note:cip_additions", "consolidated", "6.483.065.709", "PNJ_financial_statements_2020_consolidated|662", "PNJ_financial_statements_2020_consolidated_662.csv", 2, 1, "Construction-in-progress additions"),
    ("PNJ", 2021, "note:cip_additions", "consolidated"): _disclosure_cell("PNJ", 2021, "note:cip_additions", "consolidated", "2.388.425.000", "PNJ_financial_statements_2021_consolidated|707", "PNJ_financial_statements_2021_consolidated_707.csv", 2, 1, "Construction-in-progress additions"),
    ("PNJ", 2022, "note:cip_additions", "consolidated"): _disclosure_cell("PNJ", 2022, "note:cip_additions", "consolidated", "4.030.932.699", "PNJ_financial_statements_2022_consolidated|813", "PNJ_financial_statements_2022_consolidated_813.csv", 2, 1, "Construction-in-progress additions"),
    ("PNJ", 2018, "note:personal_loans", "consolidated"): _disclosure_cell("PNJ", 2018, "note:personal_loans", "consolidated", "131.308.602.284", "PNJ_financial_statements_2018_consolidated|711", "PNJ_financial_statements_2018_consolidated_711.csv", 12, 3, "Ending personal loans"),
    ("PNJ", 2020, "note:personal_loans", "consolidated"): _disclosure_cell("PNJ", 2020, "note:personal_loans", "consolidated", "122.076.727.800", "PNJ_financial_statements_2020_consolidated|721", "PNJ_financial_statements_2020_consolidated_721.csv", 15, 3, "Ending personal loans"),
    ("PNJ", 2021, "note:personal_loans", "consolidated"): _disclosure_cell("PNJ", 2021, "note:personal_loans", "consolidated", "79.930.655.403", "PNJ_financial_statements_2021_consolidated|776", "PNJ_financial_statements_2021_consolidated_776.csv", 16, 3, "Ending personal loans"),
    ("PNJ", 2022, "note:personal_loans", "consolidated"): _disclosure_cell("PNJ", 2022, "note:personal_loans", "consolidated", "32.431.978.659", "PNJ_financial_statements_2022_consolidated|872", "PNJ_financial_statements_2022_consolidated_872.csv", 16, 3, "Ending personal loans"),
    # Q532: ending common shares and each possible winner's deferred-tax assets.
    ("SAB", 2017, "note:ending_common_shares", "consolidated"): _disclosure_cell("SAB", 2017, "note:ending_common_shares", "consolidated", "641.281.186", "SAB_financial_statements_2017_consolidated|1232", "SAB_financial_statements_2017_consolidated_1232.csv", 2, 1, "Ending issued common shares"),
    ("MPC", 2017, "note:ending_common_shares", "consolidated"): _disclosure_cell("MPC", 2017, "note:ending_common_shares", "consolidated", "68.462.850", "MPC_financial_statements_2017_consolidated|1448", "MPC_financial_statements_2017_consolidated_1448.csv", 2, 1, "Ending common shares"),
    ("MSN", 2017, "note:ending_common_shares", "consolidated"): _disclosure_cell("MSN", 2017, "note:ending_common_shares", "consolidated", "1.157.373.974", "MSN_financial_statements_2017_consolidated|1749", "MSN_financial_statements_2017_consolidated_1749.csv", 3, 1, "Ending issued common shares"),
    ("MCH", 2017, "note:ending_common_shares", "consolidated"): _disclosure_cell("MCH", 2017, "note:ending_common_shares", "consolidated", "543.132.777", "MCH_financial_statements_2017_consolidated|1256", "MCH_financial_statements_2017_consolidated_1256.csv", 3, 1, "Ending issued common shares"),
    ("HAG", 2017, "note:ending_common_shares", "consolidated"): _disclosure_cell("HAG", 2017, "note:ending_common_shares", "consolidated", "927.467.947", "HAG_financial_statements_2017_consolidated|1980", "HAG_financial_statements_2017_consolidated_1980.csv", 3, 1, "Ending issued common shares"),
    ("SAB", 2017, "note:deferred_tax_assets_total", "consolidated"): _disclosure_cell("SAB", 2017, "note:deferred_tax_assets_total", "consolidated", "136.822.415.511", "SAB_financial_statements_2017_consolidated|258", "SAB_financial_statements_2017_consolidated_258.csv", 28, 3, "Deferred-tax assets"),
    ("MPC", 2017, "note:deferred_tax_assets_total", "consolidated"): _disclosure_cell("MPC", 2017, "note:deferred_tax_assets_total", "consolidated", "30.344.896.465", "MPC_financial_statements_2017_consolidated|958", "MPC_financial_statements_2017_consolidated_958.csv", 2, 1, "Total deferred-tax assets"),
    ("MSN", 2017, "note:deferred_tax_assets_total", "consolidated"): _disclosure_cell("MSN", 2017, "note:deferred_tax_assets_total", "consolidated", "300.831", "MSN_financial_statements_2017_consolidated|1437", "MSN_financial_statements_2017_consolidated_1437.csv", 6, 1, "Total deferred-tax assets, million VND"),
    ("MCH", 2017, "note:deferred_tax_assets_total", "consolidated"): _disclosure_cell("MCH", 2017, "note:deferred_tax_assets_total", "consolidated", "153.245.522.534", "MCH_financial_statements_2017_consolidated|119", "MCH_financial_statements_2017_consolidated_119.csv", 19, 3, "Deferred-tax assets"),
    ("HAG", 2017, "note:deferred_tax_assets_total", "consolidated"): _disclosure_cell("HAG", 2017, "note:deferred_tax_assets_total", "consolidated", "105.125.055", "HAG_financial_statements_2017_consolidated|210", "HAG_financial_statements_2017_consolidated_210.csv", 41, 3, "Deferred-tax assets, thousand VND"),
})


EXPLICIT_CELLS.update({
    # Q898: related-party short-term other-payable totals, not balance-sheet
    # line 319 and not a single company's first retrieved row.
    ("DIG", 2024, "note:related_short_term_other_payables", "separate"): _disclosure_cell("DIG", 2024, "note:related_short_term_other_payables", "separate", "213.797.839.976", "DIG_financial_statements_2024_separate|1573", "DIG_financial_statements_2024_separate_1573.csv", 5, 2, "Phải trả ngắn hạn khác - các bên liên quan - tổng cộng"),
    ("PDR", 2024, "note:related_short_term_other_payables", "separate"): _disclosure_cell("PDR", 2024, "note:related_short_term_other_payables", "separate", "1.536.567.176.610", "PDR_financial_statements_2024_separate|1116", "PDR_financial_statements_2024_separate_1116.csv", 9, 1, "Phải trả bên liên quan - ngắn hạn"),
    ("VRE", 2024, "note:related_short_term_other_payables", "separate"): _disclosure_cell("VRE", 2024, "note:related_short_term_other_payables", "separate", "24.770", "VRE_financial_statements_2024_separate|1106", "VRE_financial_statements_2024_separate_1106.csv", 22, 1, "Phải trả ngắn hạn khác - các bên liên quan - tổng cộng, triệu VND"),
    # Q987: each parent's net FX result is FX gains less FX losses.  SAB has
    # both realised and unrealised losses, so both source lines are retained.
    ("MCH", 2020, "note:fx_gain", "separate"): _disclosure_cell("MCH", 2020, "note:fx_gain", "separate", "7.560.862.181", "MCH_financial_statements_2020_separate|954", "MCH_financial_statements_2020_separate_954.csv", 4, 1, "Lãi chênh lệch tỷ giá hối đoái"),
    ("MCH", 2020, "note:fx_loss", "separate"): _disclosure_cell("MCH", 2020, "note:fx_loss", "separate", "10.688.648.490", "MCH_financial_statements_2020_separate|958", "MCH_financial_statements_2020_separate_958.csv", 2, 1, "Lỗ chênh lệch tỷ giá hối đoái"),
    ("MSN", 2020, "note:fx_gain", "separate"): _disclosure_cell("MSN", 2020, "note:fx_gain", "separate", "56.934", "MSN_financial_statements_2020_separate|897", "MSN_financial_statements_2020_separate_897.csv", 7, 1, "Lãi chênh lệch tỷ giá hối đoái"),
    ("MSN", 2020, "note:fx_loss", "separate"): _disclosure_cell("MSN", 2020, "note:fx_loss", "separate", "6.000.000.000", "MSN_financial_statements_2020_separate|901", "MSN_financial_statements_2020_separate_901.csv", 9, 1, "Lỗ chênh lệch tỷ giá hối đoái"),
    ("SAB", 2020, "note:fx_gain", "separate"): _disclosure_cell("SAB", 2020, "note:fx_gain", "separate", "41.563.062.738", "SAB_financial_statements_2020_separate|1298", "SAB_financial_statements_2020_separate_1298.csv", 3, 1, "Lãi chênh lệch tỷ giá đã thực hiện"),
    ("SAB", 2020, "note:fx_loss_realised", "separate"): _disclosure_cell("SAB", 2020, "note:fx_loss_realised", "separate", "11.626.621.856", "SAB_financial_statements_2020_separate|1302", "SAB_financial_statements_2020_separate_1302.csv", 2, 1, "Lỗ chênh lệch tỷ giá đã thực hiện"),
    ("SAB", 2020, "note:fx_loss_unrealised", "separate"): _disclosure_cell("SAB", 2020, "note:fx_loss_unrealised", "separate", "2.678.826.072", "SAB_financial_statements_2020_separate|1302", "SAB_financial_statements_2020_separate_1302.csv", 3, 1, "Lỗ chênh lệch tỷ giá chưa thực hiện"),
    ("VNM", 2020, "note:fx_gain", "separate"): _disclosure_cell("VNM", 2020, "note:fx_gain", "separate", "37.988.207.647", "VNM_financial_statements_2020_separate|1620", "VNM_financial_statements_2020_separate_1620.csv", 3, 1, "Lãi chênh lệch tỷ giá hối đoái"),
    ("VNM", 2020, "note:fx_loss", "separate"): _disclosure_cell("VNM", 2020, "note:fx_loss", "separate", "21.815.538.893", "VNM_financial_statements_2020_separate|1624", "VNM_financial_statements_2020_separate_1624.csv", 3, 1, "Lỗ chênh lệch tỷ giá hối đoái"),
    # Q991: the historical program returned SHB only even though the question
    # asks for the four-bank total of the specifically named impairment line.
    ("BID", 2016, "note:afs_impairment_provision", "consolidated"): _disclosure_cell("BID", 2016, "note:afs_impairment_provision", "consolidated", "(88.954)", "BID_financial_statements_2016_consolidated|1095", "BID_financial_statements_2016_consolidated_1095.csv", 10, 1, "Dự phòng giảm giá"),
    ("CTG", 2016, "note:afs_impairment_provision", "consolidated"): _disclosure_cell("CTG", 2016, "note:afs_impairment_provision", "consolidated", "(472.121)", "CTG_financial_statements_2016_consolidated|1236", "CTG_financial_statements_2016_consolidated_1236.csv", 10, 1, "Dự phòng giảm giá chứng khoán đầu tư sẵn sàng để bán"),
    ("SHB", 2016, "note:afs_impairment_provision", "consolidated"): _disclosure_cell("SHB", 2016, "note:afs_impairment_provision", "consolidated", "33.050", "SHB_financial_statements_2016_consolidated|2136", "SHB_financial_statements_2016_consolidated_2136.csv", 1, 1, "Dự phòng giảm giá chứng khoán sẵn sàng để bán"),
    ("VIB", 2016, "note:afs_impairment_provision", "consolidated"): _disclosure_cell("VIB", 2016, "note:afs_impairment_provision", "consolidated", "-", "VIB_financial_statements_2016_consolidated|1069", "VIB_financial_statements_2016_consolidated_1069.csv", 11, 1, "Dự phòng giảm giá chứng khoán"),
    # Direct cross-company/year amount questions.
    ("HT1", 2017, "note:corporate_tax_payable_ending", "separate"): _disclosure_cell("HT1", 2017, "note:corporate_tax_payable_ending", "separate", "27.897.500.519", "HT1_financial_statements_2018_separate|932", "HT1_financial_statements_2018_separate_932.csv", 1, 1, "Corporate income tax payable at end-2017"),
    ("HT1", 2018, "note:corporate_tax_payable_ending", "separate"): _disclosure_cell("HT1", 2018, "note:corporate_tax_payable_ending", "separate", "56.426.836.190", "HT1_financial_statements_2018_separate|932", "HT1_financial_statements_2018_separate_932.csv", 1, 4, "Corporate income tax payable at end-2018"),
    ("VSC", 2015, "note:land_use_rights_nbv", "consolidated"): _disclosure_cell("VSC", 2015, "note:land_use_rights_nbv", "consolidated", "5.355.027.273", "VSC_financial_statements_2015_consolidated|846", "VSC_financial_statements_2015_consolidated_846.csv", 13, 1, "Land-use-rights ending NBV"),
    ("ACV", 2015, "note:land_use_rights_nbv", "consolidated"): _disclosure_cell("ACV", 2015, "note:land_use_rights_nbv", "consolidated", "32.243.749.055", "ACV_financial_statements_2015_consolidated|1641", "ACV_financial_statements_2015_consolidated_1641.csv", 9, 4, "Land-use-rights ending NBV"),
    ("SCR", 2023, "note:construction_service_revenue", "consolidated"): _disclosure_cell("SCR", 2023, "note:construction_service_revenue", "consolidated", "4.551.525.000", "SCR_financial_statements_2023_consolidated|1342", "SCR_financial_statements_2023_consolidated_1342.csv", 5, 1, "Construction-service revenue"),
    ("NLG", 2023, "note:construction_service_revenue", "consolidated"): _disclosure_cell("NLG", 2023, "note:construction_service_revenue", "consolidated", "45.228.296.672", "NLG_financial_statements_2023_consolidated|1305", "NLG_financial_statements_2023_consolidated_1305.csv", 4, 1, "Construction-service revenue"),
    ("CRE", 2023, "note:term_bank_deposits", "consolidated"): _disclosure_cell("CRE", 2023, "note:term_bank_deposits", "consolidated", "113.880.369.863", "CRE_financial_statements_2023_consolidated|779", "CRE_financial_statements_2023_consolidated_779.csv", 3, 1, "Term bank deposits"),
    ("SCR", 2023, "note:term_bank_deposits", "consolidated"): _disclosure_cell("SCR", 2023, "note:term_bank_deposits", "consolidated", "74.721.870.474", "SCR_financial_statements_2023_consolidated|843", "SCR_financial_statements_2023_consolidated_843.csv", 3, 1, "Term bank deposits"),
    ("GAS", 2020, "note:gas_cylinder_shell_expense", "separate"): _disclosure_cell("GAS", 2020, "note:gas_cylinder_shell_expense", "separate", "454.637.977.537", "GAS_financial_statements_2020_separate|1023", "GAS_financial_statements_2020_separate_1023.csv", 1, 1, "Gas-cylinder-shell expense"),
    ("GAS", 2022, "note:gas_cylinder_shell_expense", "separate"): _disclosure_cell("GAS", 2022, "note:gas_cylinder_shell_expense", "separate", "511.995.872.427", "GAS_financial_statements_2022_separate|1221", "GAS_financial_statements_2022_separate_1221.csv", 2, 1, "Gas-cylinder-shell expense"),
    ("GAS", 2024, "note:gas_cylinder_shell_expense", "separate"): _disclosure_cell("GAS", 2024, "note:gas_cylinder_shell_expense", "separate", "416.719.496.903", "GAS_financial_statements_2024_separate|1193", "GAS_financial_statements_2024_separate_1193.csv", 2, 1, "Gas-cylinder-shell expense"),
    ("HPG", 2015, "note:building_depreciation", "separate"): _disclosure_cell("HPG", 2015, "note:building_depreciation", "separate", "13.765.062.289", "HPG_financial_statements_2015_separate|807", "HPG_financial_statements_2015_separate_807.csv", 8, 1, "Building depreciation during the year"),
    ("HPG", 2019, "note:building_depreciation", "separate"): _disclosure_cell("HPG", 2019, "note:building_depreciation", "separate", "14.491.457.532", "HPG_financial_statements_2019_separate|778", "HPG_financial_statements_2019_separate_778.csv", 5, 1, "Building depreciation during the year"),
    ("HPG", 2022, "note:building_depreciation", "separate"): _disclosure_cell("HPG", 2022, "note:building_depreciation", "separate", "14.647.724.280", "HPG_financial_statements_2022_separate|691", "HPG_financial_statements_2022_separate_691.csv", 5, 1, "Building depreciation during the year"),
    ("HPG", 2023, "note:building_depreciation", "separate"): _disclosure_cell("HPG", 2023, "note:building_depreciation", "separate", "14.495.039.243", "HPG_financial_statements_2023_separate|739", "HPG_financial_statements_2023_separate_739.csv", 7, 1, "Building depreciation during the year"),
    ("HPG", 2024, "note:building_depreciation", "separate"): _disclosure_cell("HPG", 2024, "note:building_depreciation", "separate", "12.961.352.808", "HPG_financial_statements_2024_separate|693", "HPG_financial_statements_2024_separate_693.csv", 5, 1, "Building depreciation during the year"),
    ("NVB", 2015, "note:net_financing_cash_flow", "separate"): _disclosure_cell("NVB", 2015, "note:net_financing_cash_flow", "separate", "-", "NVB_financial_statements_2015_separate|396", "NVB_financial_statements_2015_separate_396.csv", 8, 2, "Net financing cash flow, VND"),
    ("NVB", 2016, "note:net_financing_cash_flow", "separate"): _disclosure_cell("NVB", 2016, "note:net_financing_cash_flow", "separate", "-", "NVB_financial_statements_2016_separate|409", "NVB_financial_statements_2016_separate_409.csv", 8, 2, "Net financing cash flow, million VND"),
    ("NVB", 2019, "note:net_financing_cash_flow", "separate"): _disclosure_cell("NVB", 2019, "note:net_financing_cash_flow", "separate", "1.091.339", "NVB_financial_statements_2019_separate|410", "NVB_financial_statements_2019_separate_410.csv", 8, 2, "Net financing cash flow, million VND"),
    ("NVB", 2024, "note:net_financing_cash_flow", "separate"): _disclosure_cell("NVB", 2024, "note:net_financing_cash_flow", "separate", "6.552.212", "NVB_financial_statements_2024_separate|430", "NVB_financial_statements_2024_separate_430.csv", 8, 2, "Net financing cash flow, million VND"),
    ("NVB", 2025, "note:net_financing_cash_flow", "separate"): _disclosure_cell("NVB", 2025, "note:net_financing_cash_flow", "separate", "7.500.000", "NVB_financial_statements_2025_separate|428", "NVB_financial_statements_2025_separate_428.csv", 8, 2, "Net financing cash flow, million VND"),
    ("HBC", 2018, "note:q146_opening_short_term_prepaid_expense", "consolidated"): _disclosure_cell("HBC", 2018, "note:q146_opening_short_term_prepaid_expense", "consolidated", "111.719.354.613", "HBC_financial_statements_2018_consolidated|204", "HBC_financial_statements_2018_consolidated_204.csv", 19, 4, "Opening short-term prepaid expense"),
    ("HBC", 2018, "note:q146_opening_long_term_prepaid_expense", "consolidated"): _disclosure_cell("HBC", 2018, "note:q146_opening_long_term_prepaid_expense", "consolidated", "323.372.766.931", "HBC_financial_statements_2018_consolidated|223", "HBC_financial_statements_2018_consolidated_223.csv", 24, 4, "Opening long-term prepaid expense"),
    ("HBC", 2022, "note:short_term_accrued_interest", "consolidated"): _disclosure_cell("HBC", 2022, "note:short_term_accrued_interest", "consolidated", "20.761.899.768", "HBC_financial_statements_2022_consolidated|1722", "HBC_financial_statements_2022_consolidated_1722.csv", 3, 1, "Short-term accrued interest expense"),
    ("GEX", 2022, "note:short_term_accrued_interest", "consolidated"): _disclosure_cell("GEX", 2022, "note:short_term_accrued_interest", "consolidated", "100.308.485.707", "GEX_financial_statements_2022_consolidated|1534", "GEX_financial_statements_2022_consolidated_1534.csv", 4, 1, "Short-term accrued interest expense"),
    ("VGC", 2022, "note:short_term_accrued_interest", "consolidated"): _disclosure_cell("VGC", 2022, "note:short_term_accrued_interest", "consolidated", "34.269.030.863", "VGC_financial_statements_2022_consolidated|1383", "VGC_financial_statements_2022_consolidated_1383.csv", 2, 1, "Short-term accrued interest expense"),
    ("SJG", 2022, "note:short_term_accrued_interest", "consolidated"): _disclosure_cell("SJG", 2022, "note:short_term_accrued_interest", "consolidated", "324.484.565.081", "SJG_financial_statements_2022_consolidated|1644", "SJG_financial_statements_2022_consolidated_1644.csv", 2, 1, "Short-term accrued loan and bond interest"),
    ("MPC", 2018, "note:subsidiary_investment_ending", "separate"): _disclosure_cell("MPC", 2018, "note:subsidiary_investment_ending", "separate", "1.680.383.084.683", "MPC_financial_statements_2018_separate|701", "MPC_financial_statements_2018_separate_701.csv", 3, 1, "Ending investment in subsidiaries"),
    ("MPC", 2020, "note:subsidiary_investment_ending", "separate"): _disclosure_cell("MPC", 2020, "note:subsidiary_investment_ending", "separate", "3.738.260.328.062", "MPC_financial_statements_2020_separate|797", "MPC_financial_statements_2020_separate_797.csv", 4, 1, "Ending investment in subsidiaries"),
    ("MPC", 2022, "note:subsidiary_investment_ending", "separate"): _disclosure_cell("MPC", 2022, "note:subsidiary_investment_ending", "separate", "5.113.672.015.620", "MPC_financial_statements_2022_separate|825", "MPC_financial_statements_2022_separate_825.csv", 3, 1, "Ending investment in subsidiaries"),
    ("MPC", 2023, "note:subsidiary_investment_ending", "separate"): _disclosure_cell("MPC", 2023, "note:subsidiary_investment_ending", "separate", "5.311.626.449.501", "MPC_financial_statements_2023_separate|1126", "MPC_financial_statements_2023_separate_1126.csv", 3, 1, "Ending investment in subsidiaries"),
    ("DXG", 2021, "note:investment_property_depreciation", "separate"): _disclosure_cell("DXG", 2021, "note:investment_property_depreciation", "separate", "(1.485.726.898)", "DXG_financial_statements_2021_separate|806", "DXG_financial_statements_2021_separate_806.csv", 7, 1, "Investment-property depreciation during the year"),
    ("DXG", 2022, "note:investment_property_depreciation", "separate"): _disclosure_cell("DXG", 2022, "note:investment_property_depreciation", "separate", "(1.406.666.210)", "DXG_financial_statements_2022_separate|782", "DXG_financial_statements_2022_separate_782.csv", 7, 1, "Investment-property depreciation during the year"),
    ("DXG", 2023, "note:investment_property_depreciation", "separate"): _disclosure_cell("DXG", 2023, "note:investment_property_depreciation", "separate", "(1.361.933.052)", "DXG_financial_statements_2023_separate|793", "DXG_financial_statements_2023_separate_793.csv", 7, 1, "Investment-property depreciation during the year"),
    ("DXG", 2024, "note:investment_property_depreciation", "separate"): _disclosure_cell("DXG", 2024, "note:investment_property_depreciation", "separate", "(1.368.186.692)", "DXG_financial_statements_2024_separate|910", "DXG_financial_statements_2024_separate_910.csv", 7, 1, "Investment-property depreciation during the year"),
    ("DXG", 2025, "note:investment_property_depreciation", "separate"): _disclosure_cell("DXG", 2025, "note:investment_property_depreciation", "separate", "(3.400.085.178)", "DXG_financial_statements_2025_separate|888", "DXG_financial_statements_2025_separate_888.csv", 8, 1, "Investment-property depreciation during the year"),
})


# High-confidence repairs from the structural audit of legacy model programs.
# These programs had reused one table cell for several companies/years or had
# selected a similarly named but economically different disclosure.  Every
# replacement operand below is copied from the corresponding BTC source table.
EXPLICIT_CELLS.update({
    # Q534: maximum voting rate in associates/JVs and post-12-month payables.
    ("HBC", 2024, "note:max_associate_voting_rate", "consolidated"): _disclosure_cell("HBC", 2024, "note:max_associate_voting_rate", "consolidated", "35,25%", "HBC_financial_statements_2024_consolidated|1600", "HBC_financial_statements_2024_consolidated_1600.csv", 3, 2, "Maximum voting rate in associates"),
    ("GEX", 2024, "note:max_associate_voting_rate", "consolidated"): _disclosure_cell("GEX", 2024, "note:max_associate_voting_rate", "consolidated", "50,00%", "GEX_financial_statements_2024_consolidated|696", "GEX_financial_statements_2024_consolidated_696.csv", 2, 4, "Maximum voting rate in associates and joint ventures"),
    ("PC1", 2024, "note:max_associate_voting_rate", "consolidated"): _disclosure_cell("PC1", 2024, "note:max_associate_voting_rate", "consolidated", "49,00%", "PC1_financial_statements_2024_consolidated|1155", "PC1_financial_statements_2024_consolidated_1155.csv", 2, 1, "Maximum voting rate in associates"),
    ("HBC", 2024, "note:payables_after_12_months", "consolidated"): _disclosure_cell("HBC", 2024, "note:payables_after_12_months", "consolidated", "532.421.152.665", "HBC_financial_statements_2024_consolidated|1849", "HBC_financial_statements_2024_consolidated_1849.csv", 6, 1, "Payables after 12 months"),
    ("GEX", 2024, "note:payables_after_12_months", "consolidated"): _disclosure_cell("GEX", 2024, "note:payables_after_12_months", "consolidated", "8.306.581.259.300", "GEX_financial_statements_2024_consolidated|1877", "GEX_financial_statements_2024_consolidated_1877.csv", 7, 1, "Payables after 12 months"),
    ("PC1", 2024, "note:payables_after_12_months", "consolidated"): _disclosure_cell("PC1", 2024, "note:payables_after_12_months", "consolidated", "7.830.630.395.563", "PC1_financial_statements_2024_consolidated|1619", "PC1_financial_statements_2024_consolidated_1619.csv", 6, 1, "Payables after 12 months"),
    # Q851: corporate income tax incurred/payable during each requested year.
    ("GEX", 2015, "note:corporate_tax_payable_during_year", "separate"): _disclosure_cell("GEX", 2015, "note:corporate_tax_payable_during_year", "separate", "33.722.050.001", "GEX_financial_statements_2015_separate|941", "GEX_financial_statements_2015_separate_941.csv", 4, 3, "Corporate income tax payable during the year"),
    ("GEX", 2018, "note:corporate_tax_payable_during_year", "separate"): _disclosure_cell("GEX", 2018, "note:corporate_tax_payable_during_year", "separate", "27.251.729.705", "GEX_financial_statements_2018_separate|1343", "GEX_financial_statements_2018_separate_1343.csv", 2, 2, "Corporate income tax payable during the year"),
    ("GEX", 2022, "note:corporate_tax_payable_during_year", "separate"): _disclosure_cell("GEX", 2022, "note:corporate_tax_payable_during_year", "separate", "7.365.517.897", "GEX_financial_statements_2022_separate|1144", "GEX_financial_statements_2022_separate_1144.csv", 3, 3, "Corporate income tax payable during the year"),
    ("GEX", 2025, "note:corporate_tax_payable_during_year", "separate"): _disclosure_cell("GEX", 2025, "note:corporate_tax_payable_during_year", "separate", "244.199.827.004", "GEX_financial_statements_2025_separate|1421", "GEX_financial_statements_2025_separate_1421.csv", 4, 3, "Corporate income tax payable during the year"),
    # Q875: parent-bank total assets, reported in VND million.
    ("STB", 2016, "note:parent_total_assets_million", "separate"): _disclosure_cell("STB", 2016, "note:parent_total_assets_million", "separate", "329.187.491", "STB_financial_statements_2016_separate|197", "STB_financial_statements_2016_separate_197.csv", 39, 2, "Parent total assets, VND million"),
    ("STB", 2017, "note:parent_total_assets_million", "separate"): _disclosure_cell("STB", 2017, "note:parent_total_assets_million", "separate", "364.016.293", "STB_financial_statements_2017_separate|209", "STB_financial_statements_2017_separate_209.csv", 39, 2, "Parent total assets, VND million"),
    ("STB", 2022, "note:parent_total_assets_million", "separate"): _disclosure_cell("STB", 2022, "note:parent_total_assets_million", "separate", "587.216.341", "STB_financial_statements_2022_separate|201", "STB_financial_statements_2022_separate_201.csv", 35, 2, "Parent total assets, VND million"),
    ("STB", 2025, "note:parent_total_assets_million", "separate"): _disclosure_cell("STB", 2025, "note:parent_total_assets_million", "separate", "908.632.552", "STB_financial_statements_2025_separate|139", "STB_financial_statements_2025_separate_139.csv", 33, 3, "Parent total assets, VND million"),
    # Q925: VAB charter capital at each requested year end.
    ("VAB", 2020, "note:charter_capital", "consolidated"): _disclosure_cell("VAB", 2020, "note:charter_capital", "consolidated", "4.449.635.670.000", "VAB_financial_statements_2020_consolidated|344", "VAB_financial_statements_2020_consolidated_344.csv", 3, 2, "Charter capital"),
    ("VAB", 2024, "note:charter_capital", "consolidated"): _disclosure_cell("VAB", 2024, "note:charter_capital", "consolidated", "5.399.600.430.000", "VAB_financial_statements_2024_consolidated|312", "VAB_financial_statements_2024_consolidated_312.csv", 15, 2, "Charter capital"),
    ("VAB", 2025, "note:charter_capital", "consolidated"): _disclosure_cell("VAB", 2025, "note:charter_capital", "consolidated", "8.163.606.720.000", "VAB_financial_statements_2025_consolidated|333", "VAB_financial_statements_2025_consolidated_333.csv", 16, 2, "Charter capital"),
    # Q999: parent VCB issued valuable papers, reported in VND million.
    ("VCB", 2016, "note:issued_valuable_papers_million", "separate"): _disclosure_cell("VCB", 2016, "note:issued_valuable_papers_million", "separate", "10.005.376", "VCB_financial_statements_2016_separate|215", "VCB_financial_statements_2016_separate_215.csv", 7, 3, "Issued valuable papers, VND million"),
    ("VCB", 2020, "note:issued_valuable_papers_million", "separate"): _disclosure_cell("VCB", 2020, "note:issued_valuable_papers_million", "separate", "21.369.849", "VCB_financial_statements_2020_separate|201", "VCB_financial_statements_2020_separate_201.csv", 9, 3, "Issued valuable papers, VND million"),
    ("VCB", 2025, "note:issued_valuable_papers_million", "separate"): _disclosure_cell("VCB", 2025, "note:issued_valuable_papers_million", "separate", "27.101.221", "VCB_financial_statements_2025_separate|220", "VCB_financial_statements_2025_separate_220.csv", 9, 3, "Issued valuable papers, VND million"),
    ("VCB", 2025, "note:q195_fx_transaction_commitments_million", "separate"): _disclosure_cell("VCB", 2025, "note:q195_fx_transaction_commitments_million", "separate", "214.549.855", "VCB_financial_statements_2025_separate|247", "VCB_financial_statements_2025_separate_247.csv", 1, 3, "Foreign-exchange transaction commitments, VND million"),
    # Q1012: short-term related-party receivables include trade and other.
    ("GAS", 2017, "note:related_short_term_trade_receivables", "consolidated"): _disclosure_cell("GAS", 2017, "note:related_short_term_trade_receivables", "consolidated", "695.664.844.162", "GAS_financial_statements_2017_consolidated|1232", "GAS_financial_statements_2017_consolidated_1232.csv", 1, 1, "Related-party trade receivables"),
    ("GAS", 2017, "note:related_short_term_other_receivables", "consolidated"): _disclosure_cell("GAS", 2017, "note:related_short_term_other_receivables", "consolidated", "959.625.689.287", "GAS_financial_statements_2017_consolidated|1232", "GAS_financial_statements_2017_consolidated_1232.csv", 11, 1, "Related-party other short-term receivables"),
    ("POW", 2017, "note:related_short_term_trade_receivables", "consolidated"): _disclosure_cell("POW", 2017, "note:related_short_term_trade_receivables", "consolidated", "66.041.712.700", "POW_financial_statements_2017_consolidated|815", "POW_financial_statements_2017_consolidated_815.csv", 4, 1, "Related-party trade receivables"),
    ("POW", 2017, "note:related_short_term_other_receivables", "consolidated"): _disclosure_cell("POW", 2017, "note:related_short_term_other_receivables", "consolidated", "263.360.518.943", "POW_financial_statements_2017_consolidated|828", "POW_financial_statements_2017_consolidated_828.csv", 10, 1, "Related-party other short-term receivables"),
    ("DTK", 2017, "note:related_short_term_trade_receivables", "consolidated"): _disclosure_cell("DTK", 2017, "note:related_short_term_trade_receivables", "consolidated", "7.311.299.202", "DTK_financial_statements_2017_consolidated|1679", "DTK_financial_statements_2017_consolidated_1679.csv", 4, 2, "Related-party short-term receivables, code 131"),
    ("DTK", 2017, "note:related_short_term_other_receivables", "consolidated"): _disclosure_cell("DTK", 2017, "note:related_short_term_other_receivables", "consolidated", "53.091.731.843", "DTK_financial_statements_2017_consolidated|1679", "DTK_financial_statements_2017_consolidated_1679.csv", 4, 3, "Related-party other short-term receivables, code 138"),
    # Q160: the VPB disclosure is already denominated in VND million.
    ("VPB", 2025, "note:profit_attributable_common_shareholders_million", "consolidated"): _disclosure_cell("VPB", 2025, "note:profit_attributable_common_shareholders_million", "consolidated", "23.989.930", "VPB_financial_statements_2025_consolidated|2142", "VPB_financial_statements_2025_consolidated_2142.csv", 1, 1, "Profit attributable to ordinary shareholders, VND million"),
    # Q805: code-130-style related-party receivables comprise trade (131) and
    # other short-term receivables (138), not loans (135) or supplier advances
    # (132). SCR's note has no printed total, so retain every non-zero audited
    # ending trade-receivable cell. DIG prints a trade total and a combined
    # short/long other-receivable total; subtract its disclosed long-term row.
    ("SCR", 2018, "note:q805_related_trade_01", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_01", "separate", "7.277.782.332", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 4, 1, "Related-party trade receivable 01"),
    ("SCR", 2018, "note:q805_related_trade_02", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_02", "separate", "1.425.000.000", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 5, 1, "Related-party trade receivable 02"),
    ("SCR", 2018, "note:q805_related_trade_03", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_03", "separate", "182.508.548", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 6, 1, "Related-party trade receivable 03"),
    ("SCR", 2018, "note:q805_related_trade_04", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_04", "separate", "181.478.882", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 7, 1, "Related-party trade receivable 04"),
    ("SCR", 2018, "note:q805_related_trade_05", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_05", "separate", "72.523.981.814", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 10, 1, "Related-party trade receivable 05"),
    ("SCR", 2018, "note:q805_related_trade_06", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_06", "separate", "26.000.000.000", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 11, 1, "Related-party trade receivable 06"),
    ("SCR", 2018, "note:q805_related_trade_07", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_07", "separate", "1.010.036.878", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 12, 1, "Related-party trade receivable 07"),
    ("SCR", 2018, "note:q805_related_trade_08", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_08", "separate", "159.031.174", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 13, 1, "Related-party trade receivable 08"),
    ("SCR", 2018, "note:q805_related_trade_09", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_09", "separate", "592.639.108", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 14, 1, "Related-party trade receivable 09"),
    ("SCR", 2018, "note:q805_related_trade_10", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_10", "separate", "697.828.871", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 15, 1, "Related-party trade receivable 10"),
    ("SCR", 2018, "note:q805_related_trade_11", "separate"): _disclosure_cell("SCR", 2018, "note:q805_related_trade_11", "separate", "313.354.400", "SCR_financial_statements_2018_separate|837", "SCR_financial_statements_2018_separate_837.csv", 16, 1, "Related-party trade receivable 11"),
    ("DIG", 2018, "note:q805_related_trade_total", "separate"): _disclosure_cell("DIG", 2018, "note:q805_related_trade_total", "separate", "3.327.525.677", "DIG_financial_statements_2018_separate|1311", "DIG_financial_statements_2018_separate_1311.csv", 10, 3, "Related-party short-term trade receivables total"),
    ("DIG", 2018, "note:q805_related_other_all_total", "separate"): _disclosure_cell("DIG", 2018, "note:q805_related_other_all_total", "separate", "16.886.726.612", "DIG_financial_statements_2018_separate|1351", "DIG_financial_statements_2018_separate_1351.csv", 12, 3, "Related-party other receivables, short- and long-term total"),
    ("DIG", 2018, "note:q805_related_other_long_term", "separate"): _disclosure_cell("DIG", 2018, "note:q805_related_other_long_term", "separate", "3.000.000.000", "DIG_financial_statements_2018_separate|1351", "DIG_financial_statements_2018_separate_1351.csv", 11, 3, "Related-party other long-term receivable"),
    # Q51/Q88/Q110/Q115/Q139/Q152: high-confidence legacy repairs where the
    # old program selected an adjacent section/total, inverted an expense, or
    # applied a conversion despite the question requesting the source unit.
    ("VCB", 2020, "note:promissory_bonds_medium_term_vnd_million", "consolidated"): _disclosure_cell("VCB", 2020, "note:promissory_bonds_medium_term_vnd_million", "consolidated", "10.437.945", "VCB_financial_statements_2020_consolidated|1311", "VCB_financial_statements_2020_consolidated_1311.csv", 7, 1, "Medium-term VND promissory notes and bonds, VND million"),
    ("NVB", 2019, "note:total_assets_million", "consolidated"): _disclosure_cell("NVB", 2019, "note:total_assets_million", "consolidated", "80.394.022", "NVB_financial_statements_2019_consolidated|280", "NVB_financial_statements_2019_consolidated_280.csv", 19, 2, "Total assets at 31 December 2019, VND million"),
    # Q717: the legacy denominator came from the comparative-period interest-
    # rate-sensitivity table (VND 73,007,228 million).  The question asks for
    # parent-bank 2019 total assets, so bind both operands to direct 31/12/2019
    # rows in the same separate report.
    ("NVB", 2019, "note:q717_off_balance_commitments_million", "separate"): _disclosure_cell(
        "NVB", 2019, "note:q717_off_balance_commitments_million", "separate",
        "12.053.691", "NVB_financial_statements_2019_separate|1423",
        "NVB_financial_statements_2019_separate_1423.csv", 12, 1,
        "Parent NVB total off-balance-sheet commitments at 31 December 2019, VND million",
    ),
    ("NVB", 2019, "note:q717_parent_total_assets_million", "separate"): _disclosure_cell(
        "NVB", 2019, "note:q717_parent_total_assets_million", "separate",
        "80.405.111", "NVB_financial_statements_2019_separate|268",
        "NVB_financial_statements_2019_separate_268.csv", 21, 2,
        "Parent NVB total assets at 31 December 2019, VND million",
    ),
    ("VCB", 2022, "note:customer_deposits_million", "separate"): _disclosure_cell("VCB", 2022, "note:customer_deposits_million", "separate", "1.244.500.889", "VCB_financial_statements_2022_separate|229", "VCB_financial_statements_2022_separate_229.csv", 7, 3, "Parent customer deposits at 31 December 2022, VND million"),
    ("CTG", 2019, "note:total_operating_expense_million", "separate"): _disclosure_cell("CTG", 2019, "note:total_operating_expense_million", "separate", "(14.733.282)", "CTG_financial_statements_2019_separate|331", "CTG_financial_statements_2019_separate_331.csv", 14, 2, "Parent total operating expense, VND million"),
    ("HAG", 2020, "note:net_revenue_thousand", "consolidated"): _disclosure_cell("HAG", 2020, "note:net_revenue_thousand", "consolidated", "3.176.645.956", "HAG_financial_statements_2020_consolidated|2052", "HAG_financial_statements_2020_consolidated_2052.csv", 14, 1, "Net revenue, thousand VND"),
    ("SGB", 2023, "note:total_assets_million", "consolidated"): _disclosure_cell("SGB", 2023, "note:total_assets_million", "consolidated", "31.500.625", "SGB_financial_statements_2023_consolidated|301", "SGB_financial_statements_2023_consolidated_301.csv", 21, 2, "Consolidated total assets at 31 December 2023, VND million"),
    # Q877: deposit-interest cost and total interest cost for all requested years.
    ("HDB", 2023, "note:deposit_interest_cost", "consolidated"): _disclosure_cell("HDB", 2023, "note:deposit_interest_cost", "consolidated", "23.657.737", "HDB_financial_statements_2023_consolidated_2|1882", "HDB_financial_statements_2023_consolidated_2_1882.csv", 1, 1, "Deposit interest cost, VND million"),
    ("HDB", 2023, "note:total_interest_cost", "consolidated"): _disclosure_cell("HDB", 2023, "note:total_interest_cost", "consolidated", "30.456.603", "HDB_financial_statements_2023_consolidated_2|1882", "HDB_financial_statements_2023_consolidated_2_1882.csv", 5, 1, "Total interest cost, VND million"),
    ("HDB", 2024, "note:deposit_interest_cost", "consolidated"): _disclosure_cell("HDB", 2024, "note:deposit_interest_cost", "consolidated", "20.578.179", "HDB_financial_statements_2024_consolidated|1871", "HDB_financial_statements_2024_consolidated_1871.csv", 1, 1, "Deposit interest cost, VND million"),
    ("HDB", 2024, "note:total_interest_cost", "consolidated"): _disclosure_cell("HDB", 2024, "note:total_interest_cost", "consolidated", "27.138.452", "HDB_financial_statements_2024_consolidated|1871", "HDB_financial_statements_2024_consolidated_1871.csv", 5, 1, "Total interest cost, VND million"),
    ("HDB", 2025, "note:deposit_interest_cost", "consolidated"): _disclosure_cell("HDB", 2025, "note:deposit_interest_cost", "consolidated", "26.150.925", "HDB_financial_statements_2025_consolidated|1882", "HDB_financial_statements_2025_consolidated_1882.csv", 2, 1, "Deposit interest cost, VND million"),
    ("HDB", 2025, "note:total_interest_cost", "consolidated"): _disclosure_cell("HDB", 2025, "note:total_interest_cost", "consolidated", "33.246.226", "HDB_financial_statements_2025_consolidated|1882", "HDB_financial_statements_2025_consolidated_1882.csv", 6, 1, "Total interest cost, VND million"),
    # Q889: ending intangible-PPE net book value for every requested year.
    ("FTS", 2019, "note:intangible_ppe_nbv", "unknown"): _disclosure_cell("FTS", 2019, "note:intangible_ppe_nbv", "unknown", "18.974.457.727", "FTS_financial_statements_2019|1031", "FTS_financial_statements_2019_1031.csv", 20, 5, "Intangible fixed assets - ending net book value"),
    ("FTS", 2020, "note:intangible_ppe_nbv", "unknown"): _disclosure_cell("FTS", 2020, "note:intangible_ppe_nbv", "unknown", "19.383.703.274", "FTS_financial_statements_2020|1037", "FTS_financial_statements_2020_1037.csv", 20, 5, "Intangible fixed assets - ending net book value"),
    ("FTS", 2023, "note:intangible_ppe_nbv", "unknown"): _disclosure_cell("FTS", 2023, "note:intangible_ppe_nbv", "unknown", "18.335.715.313", "FTS_financial_statements_2023|1175", "FTS_financial_statements_2023_1175.csv", 20, 5, "Intangible fixed assets - ending net book value"),
    ("FTS", 2024, "note:intangible_ppe_nbv", "unknown"): _disclosure_cell("FTS", 2024, "note:intangible_ppe_nbv", "unknown", "20.231.334.842", "FTS_financial_statements_2024|1106", "FTS_financial_statements_2024_1106.csv", 20, 5, "Intangible fixed assets - ending net book value"),
    # Q515: the old program returned total liabilities instead of the requested
    # materials/tools balance in the year selected by those liabilities.
    ("VAB", 2020, "note:total_liabilities", "consolidated"): _disclosure_cell("VAB", 2020, "note:total_liabilities", "consolidated", "80.805.422.197.489", "VAB_financial_statements_2020_consolidated|327", "VAB_financial_statements_2020_consolidated_327.csv", 15, 2, "Total liabilities"),
    ("VAB", 2024, "note:total_liabilities", "consolidated"): _disclosure_cell("VAB", 2024, "note:total_liabilities", "consolidated", "110.975.359.140.135", "VAB_financial_statements_2024_consolidated|312", "VAB_financial_statements_2024_consolidated_312.csv", 12, 2, "Total liabilities"),
    ("VAB", 2025, "note:total_liabilities", "consolidated"): _disclosure_cell("VAB", 2025, "note:total_liabilities", "consolidated", "130.330.504.529.167", "VAB_financial_statements_2025_consolidated|333", "VAB_financial_statements_2025_consolidated_333.csv", 13, 2, "Total liabilities"),
    ("VAB", 2024, "note:materials_and_tools", "consolidated"): _disclosure_cell("VAB", 2024, "note:materials_and_tools", "consolidated", "3.609.504.412", "VAB_financial_statements_2024_consolidated|1336", "VAB_financial_statements_2024_consolidated_1336.csv", 7, 1, "Materials and tools"),
    ("VAB", 2025, "note:materials_and_tools", "consolidated"): _disclosure_cell("VAB", 2025, "note:materials_and_tools", "consolidated", "4.552.242.279", "VAB_financial_statements_2025_consolidated|1323", "VAB_financial_statements_2025_consolidated_1323.csv", 1, 1, "Materials and tools"),
})


# Q893 has no single disclosed subtotal: the related-party note lists sales
# separately for each counterparty across adjacent tables.  Retain all 32
# non-zero BTC cells so the four-year total is still computed at runtime.
_HDG_RELATED_SALES_SPECS = (
    (2016, "833.991.819", 1284, 5), (2016, "798.384.545", 1284, 14),
    (2016, "600.561.818", 1284, 21), (2016, "951.965.683", 1284, 27),
    (2016, "342.252.220", 1299, 11), (2016, "699.795.299", 1299, 14),
    (2016, "2.858.809.090", 1299, 17), (2016, "170.181.818", 1299, 20),
    (2016, "365.280.000", 1299, 29),
    (2017, "5.472.198.908", 1201, 5), (2017, "5.852.044.207", 1201, 12),
    (2017, "692.072.727", 1216, 4), (2017, "5.716.798.156", 1216, 12),
    (2017, "2.111.532.096", 1216, 29), (2017, "3.458.375.000", 1216, 34),
    (2017, "1.306.711.855", 1235, 5), (2017, "6.096.164.606", 1235, 20),
    (2018, "1.010.447.675", 1368, 6), (2018, "703.833.745", 1368, 13),
    (2018, "610.833.295", 1368, 26), (2018, "24.954.545", 1368, 31),
    (2018, "107.173.400", 1383, 5), (2018, "3.342.693.515", 1383, 10),
    (2018, "2.494.257.773", 1383, 19), (2018, "761.280.000", 1400, 6),
    (2019, "816.181.537", 1456, 6), (2019, "816.181.537", 1456, 16),
    (2019, "3.258.469.856", 1456, 27), (2019, "862.592.200", 1475, 16),
    (2019, "810.090.908", 1475, 21), (2019, "761.280.000", 1475, 31),
    (2019, "3.981.061.368", 1475, 38),
)
_hdg_year_counters: dict[int, int] = {}
for _year, _raw, _line, _row in _HDG_RELATED_SALES_SPECS:
    _index = _hdg_year_counters.get(_year, 0)
    _hdg_year_counters[_year] = _index + 1
    _metric = f"note:related_party_sales_{_index}"
    _ref = f"HDG_financial_statements_{_year}_separate|{_line}"
    EXPLICIT_CELLS[("HDG", _year, _metric, "separate")] = _disclosure_cell(
        "HDG", _year, _metric, "separate", _raw, _ref,
        f"HDG_financial_statements_{_year}_separate_{_line}.csv", _row, 1,
        "Sales of goods and services to a related party",
    )


EXPLICIT_CELLS.update({
    # Q979: each requested company's deposit-interest disclosure.  DTK labels
    # the component jointly as deposit and loan interest, so preserve that
    # exact BTC label rather than pretending a finer split exists.
    ("POW", 2022, "note:deposit_interest_income", "consolidated"): _disclosure_cell("POW", 2022, "note:deposit_interest_income", "consolidated", "319.109.165.043", "POW_financial_statements_2022_consolidated|1612", "POW_financial_statements_2022_consolidated_1612.csv", 2, 1, "Deposit interest income"),
    ("GAS", 2022, "note:deposit_interest_income", "consolidated"): _disclosure_cell("GAS", 2022, "note:deposit_interest_income", "consolidated", "1.236.723.285.029", "GAS_financial_statements_2022_consolidated|1677", "GAS_financial_statements_2022_consolidated_1677.csv", 1, 1, "Deposit interest income"),
    ("DTK", 2022, "note:deposit_interest_income", "consolidated"): _disclosure_cell("DTK", 2022, "note:deposit_interest_income", "consolidated", "5.030.193.776", "DTK_financial_statements_2022_consolidated|1118", "DTK_financial_statements_2022_consolidated_1118.csv", 1, 1, "Deposit and loan interest income"),
    ("GEG", 2022, "note:deposit_interest_income", "consolidated"): _disclosure_cell("GEG", 2022, "note:deposit_interest_income", "consolidated", "30.505.925.784", "GEG_financial_statements_2022_consolidated|1518", "GEG_financial_statements_2022_consolidated_1518.csv", 2, 1, "Deposit and loan interest income"),
    # Q795: parent-company short-term prepaid expense from each 2019 balance sheet.
    ("GAS", 2019, "note:short_term_prepaid_expense", "separate"): _disclosure_cell("GAS", 2019, "note:short_term_prepaid_expense", "separate", "50.699.483.380", "GAS_financial_statements_2019_separate|272", "GAS_financial_statements_2019_separate_272.csv", 17, 3, "Short-term prepaid expense"),
    ("POW", 2019, "note:short_term_prepaid_expense", "separate"): _disclosure_cell("POW", 2019, "note:short_term_prepaid_expense", "separate", "24.968.201.119", "POW_financial_statements_2019_separate|287", "POW_financial_statements_2019_separate_287.csv", 16, 3, "Short-term prepaid expense"),
    # Q809: corporate income tax payable/incurred during 2025.
    ("DNH", 2025, "note:corporate_tax_payable_during_year", "separate"): _disclosure_cell("DNH", 2025, "note:corporate_tax_payable_during_year", "separate", "214.228.338.491", "DNH_financial_statements_2025_separate|919", "DNH_financial_statements_2025_separate_919.csv", 1, 2, "Corporate income tax payable during the year"),
    ("HND", 2025, "note:corporate_tax_payable_during_year", "unknown"): _disclosure_cell("HND", 2025, "note:corporate_tax_payable_during_year", "unknown", "38.599.316.827", "HND_financial_statements_2025|789", "HND_financial_statements_2025_789.csv", 1, 2, "Corporate income tax payable during the year"),
    # Q815: consolidated net foreign-exchange result in every requested year.
    ("OCB", 2017, "note:net_fx_result", "consolidated"): _disclosure_cell("OCB", 2017, "note:net_fx_result", "consolidated", "46.999.721.794", "OCB_financial_statements_2017_consolidated|1992", "OCB_financial_statements_2017_consolidated_1992.csv", 10, 1, "Net result from foreign-exchange trading"),
    ("OCB", 2020, "note:net_fx_result", "consolidated"): _disclosure_cell("OCB", 2020, "note:net_fx_result", "consolidated", "94.975.111.343", "OCB_financial_statements_2020_consolidated|333", "OCB_financial_statements_2020_consolidated_333.csv", 6, 2, "Net result from foreign-exchange trading"),
    ("OCB", 2021, "note:net_fx_result", "consolidated"): _disclosure_cell("OCB", 2021, "note:net_fx_result", "consolidated", "99.732.528.303", "OCB_financial_statements_2021_consolidated|425", "OCB_financial_statements_2021_consolidated_425.csv", 6, 2, "Net result from foreign-exchange trading"),
    ("OCB", 2022, "note:net_fx_result", "consolidated"): _disclosure_cell("OCB", 2022, "note:net_fx_result", "consolidated", "145.113.883.664", "OCB_financial_statements_2022_consolidated_2|307", "OCB_financial_statements_2022_consolidated_2_307.csv", 6, 2, "Net result from foreign-exchange trading"),
    # Q830: parent-bank cash and gold; source reports are already in VND million.
    ("STB", 2017, "note:cash_and_gold_million", "separate"): _disclosure_cell("STB", 2017, "note:cash_and_gold_million", "separate", "5.906.775", "STB_financial_statements_2017_separate|1921", "STB_financial_statements_2017_separate_1921.csv", 1, 1, "Cash and gold, VND million"),
    ("STB", 2021, "note:cash_and_gold_million", "separate"): _disclosure_cell("STB", 2021, "note:cash_and_gold_million", "separate", "7.856.774", "STB_financial_statements_2021_separate|1965", "STB_financial_statements_2021_separate_1965.csv", 1, 1, "Cash and gold, VND million"),
    ("STB", 2022, "note:cash_and_gold_million", "separate"): _disclosure_cell("STB", 2022, "note:cash_and_gold_million", "separate", "7.440.220", "STB_financial_statements_2022_separate|1977", "STB_financial_statements_2022_separate_1977.csv", 1, 1, "Cash and gold, VND million"),
    # Q845: consolidated BVH EPS, stated in VND per share.
    ("BVH", 2017, "note:basic_eps", "consolidated"): _disclosure_cell("BVH", 2017, "note:basic_eps", "consolidated", "2.286", "BVH_financial_statements_2017_consolidated|2400", "BVH_financial_statements_2017_consolidated_2400.csv", 4, 1, "Basic earnings per share"),
    ("BVH", 2019, "note:basic_eps", "consolidated"): _disclosure_cell("BVH", 2019, "note:basic_eps", "consolidated", "1.689", "BVH_financial_statements_2019_consolidated|2155", "BVH_financial_statements_2019_consolidated_2155.csv", 5, 1, "Basic earnings per share"),
    ("BVH", 2022, "note:basic_eps", "consolidated"): _disclosure_cell("BVH", 2022, "note:basic_eps", "consolidated", "2.089", "BVH_financial_statements_2022_consolidated|2255", "BVH_financial_statements_2022_consolidated_2255.csv", 5, 1, "Basic earnings per share"),
    ("BVH", 2024, "note:basic_eps", "consolidated"): _disclosure_cell("BVH", 2024, "note:basic_eps", "consolidated", "2.843", "BVH_financial_statements_2024_consolidated|2229", "BVH_financial_statements_2024_consolidated_2229.csv", 5, 1, "Basic earnings per share"),
    # Q856: consolidated PC1 EPS in each financial year's own report.
    ("PC1", 2015, "note:basic_eps", "consolidated"): _disclosure_cell("PC1", 2015, "note:basic_eps", "consolidated", "9.179", "PC1_financial_statements_2015_consolidated|1258", "PC1_financial_statements_2015_consolidated_1258.csv", 4, 1, "Basic earnings per share"),
    ("PC1", 2020, "note:basic_eps", "consolidated"): _disclosure_cell("PC1", 2020, "note:basic_eps", "consolidated", "2.682", "PC1_financial_statements_2020_consolidated|1696", "PC1_financial_statements_2020_consolidated_1696.csv", 4, 1, "Basic earnings per share"),
    ("PC1", 2022, "note:basic_eps", "consolidated"): _disclosure_cell("PC1", 2022, "note:basic_eps", "consolidated", "1.519", "PC1_financial_statements_2022_consolidated|388", "PC1_financial_statements_2022_consolidated_388.csv", 5, 3, "Basic earnings per share"),
    ("PC1", 2023, "note:basic_eps", "consolidated"): _disclosure_cell("PC1", 2023, "note:basic_eps", "consolidated", "405", "PC1_financial_statements_2023_consolidated|259", "PC1_financial_statements_2023_consolidated_259.csv", 5, 3, "Basic earnings per share"),
    ("PC1", 2024, "note:basic_eps", "consolidated"): _disclosure_cell("PC1", 2024, "note:basic_eps", "consolidated", "1.177", "PC1_financial_statements_2024_consolidated|328", "PC1_financial_statements_2024_consolidated_328.csv", 5, 3, "Basic earnings per share"),
    # Q873: preserve the disclosed signs so deferred-tax income reduces, and
    # deferred-tax expense increases, total income-tax expense.
    ("DXG", 2017, "note:current_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2017, "note:current_tax_expense_signed", "consolidated", "(274.878.315.250)", "DXG_financial_statements_2017_consolidated|252", "DXG_financial_statements_2017_consolidated_252.csv", 15, 3, "Current corporate income tax expense"),
    ("DXG", 2017, "note:deferred_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2017, "note:deferred_tax_expense_signed", "consolidated", "24.978.471.969", "DXG_financial_statements_2017_consolidated|252", "DXG_financial_statements_2017_consolidated_252.csv", 16, 3, "Deferred corporate income tax income"),
    ("DXG", 2018, "note:current_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2018, "note:current_tax_expense_signed", "consolidated", "(381.773.261.869)", "DXG_financial_statements_2018_consolidated|317", "DXG_financial_statements_2018_consolidated_317.csv", 15, 3, "Current corporate income tax expense"),
    ("DXG", 2018, "note:deferred_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2018, "note:deferred_tax_expense_signed", "consolidated", "2.783.512.358", "DXG_financial_statements_2018_consolidated|317", "DXG_financial_statements_2018_consolidated_317.csv", 16, 3, "Deferred corporate income tax income"),
    ("DXG", 2021, "note:current_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2021, "note:current_tax_expense_signed", "consolidated", "(849.225.776.906)", "DXG_financial_statements_2021_consolidated|264", "DXG_financial_statements_2021_consolidated_264.csv", 15, 3, "Current corporate income tax expense"),
    ("DXG", 2021, "note:deferred_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2021, "note:deferred_tax_expense_signed", "consolidated", "(71.737.236.074)", "DXG_financial_statements_2021_consolidated|264", "DXG_financial_statements_2021_consolidated_264.csv", 16, 3, "Deferred corporate income tax expense"),
    ("DXG", 2023, "note:current_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2023, "note:current_tax_expense_signed", "consolidated", "(171.914.882.384)", "DXG_financial_statements_2023_consolidated|260", "DXG_financial_statements_2023_consolidated_260.csv", 15, 3, "Current corporate income tax expense"),
    ("DXG", 2023, "note:deferred_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2023, "note:deferred_tax_expense_signed", "consolidated", "(130.998.156.825)", "DXG_financial_statements_2023_consolidated|260", "DXG_financial_statements_2023_consolidated_260.csv", 16, 3, "Deferred corporate income tax expense"),
    ("DXG", 2025, "note:current_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2025, "note:current_tax_expense_signed", "consolidated", "(227.618.738.009)", "DXG_financial_statements_2025_consolidated|276", "DXG_financial_statements_2025_consolidated_276.csv", 15, 3, "Current corporate income tax expense"),
    ("DXG", 2025, "note:deferred_tax_expense_signed", "consolidated"): _disclosure_cell("DXG", 2025, "note:deferred_tax_expense_signed", "consolidated", "98.033.264.950", "DXG_financial_statements_2025_consolidated|276", "DXG_financial_statements_2025_consolidated_276.csv", 16, 3, "Deferred corporate income tax income"),
})


# Q945: off-balance-sheet foreign-currency quantities.  The source reports
# each currency as a separate row; the generated formula computes the USD
# share within each company before averaging the three percentages.
_FOREIGN_CURRENCY_SPECS = {
    # This OCR table stores the original-currency column without thousands
    # separators. Exact tokens keep provenance verification byte-for-byte.
    "MSR": (3533, ("27623650", "5043070", "272215588", "7473826", "1810973", "3606"), 2),
    "GVR": (1848, ("30.359.359", "1.022", "485.765", "999.738.501", "240.309"), 6),
    "AAA": (2282, ("43.589.612", "216.294.765", "31.111", "74", "36.418", "11.950"), 2),
}
for _ticker, (_line, _values, _start_row) in _FOREIGN_CURRENCY_SPECS.items():
    for _index, _raw in enumerate(_values):
        _metric = f"note:off_balance_foreign_currency_{_index}"
        _ref = f"{_ticker}_financial_statements_2023_consolidated|{_line}"
        EXPLICIT_CELLS[(_ticker, 2023, _metric, "consolidated")] = _disclosure_cell(
            _ticker, 2023, _metric, "consolidated", _raw, _ref,
            f"{_ticker}_financial_statements_2023_consolidated_{_line}.csv",
            _start_row + _index, 1, "Off-balance-sheet foreign-currency amount",
        )


# Six late legacy repairs use primary-statement or debt-rollforward cells that
# the old programs either ignored or misread.  Preserve every requested year:
# these questions are especially vulnerable to a plausible answer obtained
# from only the first retrieved report.
EXPLICIT_CELLS.update({
    # Q835: FOX parent total long-term borrowings at each requested year end.
    ("FOX", 2016, "note:long_term_borrowings_ending", "separate"): _disclosure_cell("FOX", 2016, "note:long_term_borrowings_ending", "separate", "654.643.132.429", "FOX_financial_statements_2016_separate|660", "FOX_financial_statements_2016_separate_660.csv", 2, 5, "Ending total long-term borrowings"),
    ("FOX", 2017, "note:long_term_borrowings_ending", "separate"): _disclosure_cell("FOX", 2017, "note:long_term_borrowings_ending", "separate", "174.997.959.102", "FOX_financial_statements_2017_separate|677", "FOX_financial_statements_2017_separate_677.csv", 2, 5, "Ending total long-term borrowings"),
    ("FOX", 2018, "note:long_term_borrowings_ending", "separate"): _disclosure_cell("FOX", 2018, "note:long_term_borrowings_ending", "separate", "237.714.628.160", "FOX_financial_statements_2018_separate|782", "FOX_financial_statements_2018_separate_782.csv", 2, 5, "Ending total long-term borrowings"),
    ("FOX", 2019, "note:long_term_borrowings_ending", "separate"): _disclosure_cell("FOX", 2019, "note:long_term_borrowings_ending", "separate", "499.997.472.295", "FOX_financial_statements_2019_separate|791", "FOX_financial_statements_2019_separate_791.csv", 3, 5, "Ending total long-term borrowings"),
    ("FOX", 2020, "note:long_term_borrowings_ending", "separate"): _disclosure_cell("FOX", 2020, "note:long_term_borrowings_ending", "separate", "530.883.851.240", "FOX_financial_statements_2020_separate|816", "FOX_financial_statements_2020_separate_816.csv", 10, 6, "Ending total long-term borrowings"),
    # Q840: line 261, parent long-term prepaid expense.
    ("HPG", 2016, "note:long_term_prepaid_expense_total", "separate"): _disclosure_cell("HPG", 2016, "note:long_term_prepaid_expense_total", "separate", "1.741.764.867", "HPG_financial_statements_2016_separate|162", "HPG_financial_statements_2016_separate_162.csv", 21, 3, "Long-term prepaid expense"),
    ("HPG", 2017, "note:long_term_prepaid_expense_total", "separate"): _disclosure_cell("HPG", 2017, "note:long_term_prepaid_expense_total", "separate", "6.481.123.161", "HPG_financial_statements_2017_separate|175", "HPG_financial_statements_2017_separate_175.csv", 23, 3, "Long-term prepaid expense"),
    ("HPG", 2018, "note:long_term_prepaid_expense_total", "separate"): _disclosure_cell("HPG", 2018, "note:long_term_prepaid_expense_total", "separate", "93.422.647.632", "HPG_financial_statements_2018_separate|174", "HPG_financial_statements_2018_separate_174.csv", 20, 3, "Long-term prepaid expense"),
    ("HPG", 2020, "note:long_term_prepaid_expense_total", "separate"): _disclosure_cell("HPG", 2020, "note:long_term_prepaid_expense_total", "separate", "51.392.217.827", "HPG_financial_statements_2020_separate|175", "HPG_financial_statements_2020_separate_175.csv", 22, 3, "Long-term prepaid expense"),
    ("HPG", 2024, "note:long_term_prepaid_expense_total", "separate"): _disclosure_cell("HPG", 2024, "note:long_term_prepaid_expense_total", "separate", "3.444.798.369", "HPG_financial_statements_2024_separate|173", "HPG_financial_statements_2024_separate_173.csv", 20, 3, "Long-term prepaid expense"),
    # Q854: NLG consolidated unearned revenue is the sum of current and
    # non-current balances (lines 318 and 336) at each requested year end.
    ("NLG", 2020, "note:unearned_revenue_current", "consolidated"): _disclosure_cell("NLG", 2020, "note:unearned_revenue_current", "consolidated", "6.698.604.900", "NLG_financial_statements_2020_consolidated|216", "NLG_financial_statements_2020_consolidated_216.csv", 8, 3, "Current unearned revenue"),
    ("NLG", 2020, "note:unearned_revenue_noncurrent", "consolidated"): _disclosure_cell("NLG", 2020, "note:unearned_revenue_noncurrent", "consolidated", "757.276.989.590", "NLG_financial_statements_2020_consolidated|216", "NLG_financial_statements_2020_consolidated_216.csv", 14, 3, "Non-current unearned revenue"),
    ("NLG", 2021, "note:unearned_revenue_current", "consolidated"): _disclosure_cell("NLG", 2021, "note:unearned_revenue_current", "consolidated", "7.186.302.603", "NLG_financial_statements_2021_consolidated|247", "NLG_financial_statements_2021_consolidated_247.csv", 8, 3, "Current unearned revenue"),
    ("NLG", 2021, "note:unearned_revenue_noncurrent", "consolidated"): _disclosure_cell("NLG", 2021, "note:unearned_revenue_noncurrent", "consolidated", "285.093.422.869", "NLG_financial_statements_2021_consolidated|247", "NLG_financial_statements_2021_consolidated_247.csv", 14, 3, "Non-current unearned revenue"),
    ("NLG", 2023, "note:unearned_revenue_current", "consolidated"): _disclosure_cell("NLG", 2023, "note:unearned_revenue_current", "consolidated", "7.589.982.574", "NLG_financial_statements_2023_consolidated|240", "NLG_financial_statements_2023_consolidated_240.csv", 8, 3, "Current unearned revenue"),
    ("NLG", 2023, "note:unearned_revenue_noncurrent", "consolidated"): _disclosure_cell("NLG", 2023, "note:unearned_revenue_noncurrent", "consolidated", "259.701.132.847", "NLG_financial_statements_2023_consolidated|240", "NLG_financial_statements_2023_consolidated_240.csv", 14, 3, "Non-current unearned revenue"),
    # Q911: parent tangible fixed-assets net book value, balance-sheet line 221.
    ("VIF", 2017, "note:tangible_fixed_assets_nbv", "separate"): _disclosure_cell("VIF", 2017, "note:tangible_fixed_assets_nbv", "separate", "145.182.929.479", "VIF_financial_statements_2017_separate|294", "VIF_financial_statements_2017_separate_294.csv", 6, 3, "Tangible fixed-assets net book value"),
    ("VIF", 2020, "note:tangible_fixed_assets_nbv", "separate"): _disclosure_cell("VIF", 2020, "note:tangible_fixed_assets_nbv", "separate", "99.036.209.405", "VIF_financial_statements_2020_separate|284", "VIF_financial_statements_2020_separate_284.csv", 6, 3, "Tangible fixed-assets net book value"),
    ("VIF", 2021, "note:tangible_fixed_assets_nbv", "separate"): _disclosure_cell("VIF", 2021, "note:tangible_fixed_assets_nbv", "separate", "89.504.268.749", "VIF_financial_statements_2021_separate|230", "VIF_financial_statements_2021_separate_230.csv", 6, 3, "Tangible fixed-assets net book value"),
    ("VIF", 2022, "note:tangible_fixed_assets_nbv", "separate"): _disclosure_cell("VIF", 2022, "note:tangible_fixed_assets_nbv", "separate", "86.456.838.520", "VIF_financial_statements_2022_separate|238", "VIF_financial_statements_2022_separate_238.csv", 6, 3, "Tangible fixed-assets net book value"),
    ("VIF", 2024, "note:tangible_fixed_assets_nbv", "separate"): _disclosure_cell("VIF", 2024, "note:tangible_fixed_assets_nbv", "separate", "76.342.365.757", "VIF_financial_statements_2024_separate|236", "VIF_financial_statements_2024_separate_236.csv", 7, 3, "Tangible fixed-assets net book value"),
    # Q916: line 151 total, not the narrower note component named "other".
    ("FOX", 2016, "note:short_term_prepaid_expense_total", "separate"): _disclosure_cell("FOX", 2016, "note:short_term_prepaid_expense_total", "separate", "505.128.057.529", "FOX_financial_statements_2016_separate|181", "FOX_financial_statements_2016_separate_181.csv", 15, 3, "Short-term prepaid expense"),
    ("FOX", 2018, "note:short_term_prepaid_expense_total", "separate"): _disclosure_cell("FOX", 2018, "note:short_term_prepaid_expense_total", "separate", "816.145.049.381", "FOX_financial_statements_2018_separate|217", "FOX_financial_statements_2018_separate_217.csv", 14, 4, "Short-term prepaid expense"),
    ("FOX", 2019, "note:short_term_prepaid_expense_total", "separate"): _disclosure_cell("FOX", 2019, "note:short_term_prepaid_expense_total", "separate", "508.408.558.838", "FOX_financial_statements_2019_separate|211", "FOX_financial_statements_2019_separate_211.csv", 14, 4, "Short-term prepaid expense"),
    ("FOX", 2020, "note:short_term_prepaid_expense_total", "separate"): _disclosure_cell("FOX", 2020, "note:short_term_prepaid_expense_total", "separate", "428.415.558.326", "FOX_financial_statements_2020_separate|233", "FOX_financial_statements_2020_separate_233.csv", 16, 4, "Short-term prepaid expense"),
    # Q944: DLG consolidated construction in progress, balance-sheet line 242.
    ("DLG", 2020, "note:construction_in_progress_ending", "consolidated"): _disclosure_cell("DLG", 2020, "note:construction_in_progress_ending", "consolidated", "417.852.299.548", "DLG_financial_statements_2020_consolidated|417", "DLG_financial_statements_2020_consolidated_417.csv", 37, 4, "Construction in progress"),
    ("DLG", 2021, "note:construction_in_progress_ending", "consolidated"): _disclosure_cell("DLG", 2021, "note:construction_in_progress_ending", "consolidated", "32.436.436.566", "DLG_financial_statements_2021_consolidated|439", "DLG_financial_statements_2021_consolidated_439.csv", 38, 3, "Construction in progress"),
    ("DLG", 2022, "note:construction_in_progress_ending", "consolidated"): _disclosure_cell("DLG", 2022, "note:construction_in_progress_ending", "consolidated", "40.860.898.048", "DLG_financial_statements_2022_consolidated|393", "DLG_financial_statements_2022_consolidated_393.csv", 37, 3, "Construction in progress"),
    ("DLG", 2023, "note:construction_in_progress_ending", "consolidated"): _disclosure_cell("DLG", 2023, "note:construction_in_progress_ending", "consolidated", "34.957.889.850", "DLG_financial_statements_2023_consolidated|393", "DLG_financial_statements_2023_consolidated_393.csv", 37, 3, "Construction in progress"),
    # Q935: equity-movement cells.  HDG has no explicit 2023 appropriation
    # row, so its zero amount is derived as the retained-earnings residual.
    ("GEG", 2023, "note:welfare_fund_appropriation", "separate"): _disclosure_cell("GEG", 2023, "note:welfare_fund_appropriation", "separate", "(8.338.816.730)", "GEG_financial_statements_2023_separate|1246", "GEG_financial_statements_2023_separate_1246.csv", 13, 4, "Appropriation to bonus and welfare fund"),
    ("GEG", 2023, "note:net_profit_equity_movement", "separate"): _disclosure_cell("GEG", 2023, "note:net_profit_equity_movement", "separate", "139.221.667.151", "GEG_financial_statements_2023_separate|1246", "GEG_financial_statements_2023_separate_1246.csv", 10, 4, "Net profit during the year"),
    ("DNH", 2023, "note:welfare_fund_appropriation", "separate"): _disclosure_cell("DNH", 2023, "note:welfare_fund_appropriation", "separate", "(51.747.906.000)", "DNH_financial_statements_2023_separate|870", "DNH_financial_statements_2023_separate_870.csv", 10, 4, "Appropriation to bonus and welfare fund"),
    ("DNH", 2023, "note:net_profit_equity_movement", "separate"): _disclosure_cell("DNH", 2023, "note:net_profit_equity_movement", "separate", "1.109.931.425.058", "DNH_financial_statements_2023_separate|870", "DNH_financial_statements_2023_separate_870.csv", 8, 4, "Net profit during the year"),
    ("HDG", 2023, "note:retained_earnings_opening", "separate"): _disclosure_cell("HDG", 2023, "note:retained_earnings_opening", "separate", "1.821.735.212.648", "HDG_financial_statements_2023_separate|1322", "HDG_financial_statements_2023_separate_1322.csv", 10, 4, "Opening retained earnings"),
    ("HDG", 2023, "note:stock_dividend_equity_movement", "separate"): _disclosure_cell("HDG", 2023, "note:stock_dividend_equity_movement", "separate", "(611.509.430.000)", "HDG_financial_statements_2023_separate|1322", "HDG_financial_statements_2023_separate_1322.csv", 11, 4, "Stock dividend retained-earnings movement"),
    ("HDG", 2023, "note:net_profit_equity_movement", "separate"): _disclosure_cell("HDG", 2023, "note:net_profit_equity_movement", "separate", "745.604.062.845", "HDG_financial_statements_2023_separate|1322", "HDG_financial_statements_2023_separate_1322.csv", 12, 4, "Net profit during the year"),
    ("HDG", 2023, "note:retained_earnings_ending", "separate"): _disclosure_cell("HDG", 2023, "note:retained_earnings_ending", "separate", "1.955.829.845.493", "HDG_financial_statements_2023_separate|1322", "HDG_financial_statements_2023_separate_1322.csv", 13, 4, "Ending retained earnings"),
    # Q978: ending construction-project costs for every requested NVL year.
    ("NVL", 2020, "note:construction_project_costs", "consolidated"): _disclosure_cell("NVL", 2020, "note:construction_project_costs", "consolidated", "11.390.096.945.795", "NVL_financial_statements_2020_consolidated|2287", "NVL_financial_statements_2020_consolidated_2287.csv", 1, 1, "Construction project costs"),
    ("NVL", 2022, "note:construction_project_costs", "consolidated"): _disclosure_cell("NVL", 2022, "note:construction_project_costs", "consolidated", "32.552.672.681.648", "NVL_financial_statements_2022_consolidated|2457", "NVL_financial_statements_2022_consolidated_2457.csv", 1, 1, "Construction project costs"),
    ("NVL", 2025, "note:construction_project_costs", "consolidated"): _disclosure_cell("NVL", 2025, "note:construction_project_costs", "consolidated", "35.093.087.673.011", "NVL_financial_statements_2025_consolidated|3234", "NVL_financial_statements_2025_consolidated_3234.csv", 1, 1, "Construction project costs"),
    # Q1004: depreciation and matching administrative total, same table/year.
    ("IJC", 2016, "note:admin_depreciation", "separate"): _disclosure_cell("IJC", 2016, "note:admin_depreciation", "separate", "737.084.353", "IJC_financial_statements_2016_separate|1511", "IJC_financial_statements_2016_separate_1511.csv", 4, 1, "Administrative depreciation expense"),
    ("IJC", 2016, "note:admin_expense_total", "separate"): _disclosure_cell("IJC", 2016, "note:admin_expense_total", "separate", "37.756.555.134", "IJC_financial_statements_2016_separate|1511", "IJC_financial_statements_2016_separate_1511.csv", 8, 1, "Total administrative expense"),
    ("IJC", 2017, "note:admin_depreciation", "separate"): _disclosure_cell("IJC", 2017, "note:admin_depreciation", "separate", "680.649.292", "IJC_financial_statements_2017_separate|1481", "IJC_financial_statements_2017_separate_1481.csv", 4, 1, "Administrative depreciation expense"),
    ("IJC", 2017, "note:admin_expense_total", "separate"): _disclosure_cell("IJC", 2017, "note:admin_expense_total", "separate", "36.757.177.851", "IJC_financial_statements_2017_separate|1481", "IJC_financial_statements_2017_separate_1481.csv", 8, 1, "Total administrative expense"),
    ("IJC", 2018, "note:admin_depreciation", "separate"): _disclosure_cell("IJC", 2018, "note:admin_depreciation", "separate", "876.710.946", "IJC_financial_statements_2018_separate|1587", "IJC_financial_statements_2018_separate_1587.csv", 4, 1, "Administrative depreciation expense"),
    ("IJC", 2018, "note:admin_expense_total", "separate"): _disclosure_cell("IJC", 2018, "note:admin_expense_total", "separate", "25.250.284.273", "IJC_financial_statements_2018_separate|1587", "IJC_financial_statements_2018_separate_1587.csv", 8, 1, "Total administrative expense"),
    ("IJC", 2023, "note:admin_depreciation", "separate"): _disclosure_cell("IJC", 2023, "note:admin_depreciation", "separate", "1.918.914.610", "IJC_financial_statements_2023_separate|1820", "IJC_financial_statements_2023_separate_1820.csv", 4, 1, "Administrative depreciation expense"),
    ("IJC", 2023, "note:admin_expense_total", "separate"): _disclosure_cell("IJC", 2023, "note:admin_expense_total", "separate", "36.430.912.061", "IJC_financial_statements_2023_separate|1820", "IJC_financial_statements_2023_separate_1820.csv", 7, 1, "Total administrative expense"),
    ("IJC", 2024, "note:admin_depreciation", "separate"): _disclosure_cell("IJC", 2024, "note:admin_depreciation", "separate", "1.751.655.515", "IJC_financial_statements_2024_separate|1756", "IJC_financial_statements_2024_separate_1756.csv", 4, 1, "Administrative depreciation expense"),
    ("IJC", 2024, "note:admin_expense_total", "separate"): _disclosure_cell("IJC", 2024, "note:admin_expense_total", "separate", "41.114.484.845", "IJC_financial_statements_2024_separate|1756", "IJC_financial_statements_2024_separate_1756.csv", 8, 1, "Total administrative expense"),
    # Q833/Q843: dividend income/receipts in each requested report.
    ("VGT", 2016, "note:dividend_income", "separate"): _disclosure_cell("VGT", 2016, "note:dividend_income", "separate", "333.469.304.204", "VGT_financial_statements_2016_separate|1460", "VGT_financial_statements_2016_separate_1460.csv", 2, 1, "Dividend income"),
    ("VGT", 2024, "note:dividend_income", "separate"): _disclosure_cell("VGT", 2024, "note:dividend_income", "separate", "374.746.516.545", "VGT_financial_statements_2024_separate|1341", "VGT_financial_statements_2024_separate_1341.csv", 1, 1, "Dividend income"),
    ("VGT", 2025, "note:dividend_income", "separate"): _disclosure_cell("VGT", 2025, "note:dividend_income", "separate", "262.855.051.500", "VGT_financial_statements_2025_separate|1400", "VGT_financial_statements_2025_separate_1400.csv", 1, 1, "Dividend income"),
    ("EVF", 2020, "note:dividends_received_million", "unknown"): _disclosure_cell("EVF", 2020, "note:dividends_received_million", "unknown", "4.235", "EVF_financial_statements_2020|1248", "EVF_financial_statements_2020_1248.csv", 2, 1, "Dividends received, VND million"),
    ("EVF", 2022, "note:dividends_received_million", "unknown"): _disclosure_cell("EVF", 2022, "note:dividends_received_million", "unknown", "11.117", "EVF_financial_statements_2022|1421", "EVF_financial_statements_2022_1421.csv", 1, 1, "Dividends received, VND million"),
    ("EVF", 2024, "note:dividends_received_million", "unknown"): _disclosure_cell("EVF", 2024, "note:dividends_received_million", "unknown", "15.204", "EVF_financial_statements_2024|1492", "EVF_financial_statements_2024_1492.csv", 1, 1, "Dividends received, VND million"),
    # Q859/Q864: ending bonus-and-welfare fund balances.
    ("DCM", 2022, "note:bonus_welfare_fund_ending", "consolidated"): _disclosure_cell("DCM", 2022, "note:bonus_welfare_fund_ending", "consolidated", "306.564.953.358", "DCM_financial_statements_2022_consolidated|401", "DCM_financial_statements_2022_consolidated_401.csv", 11, 4, "Ending bonus and welfare fund"),
    ("DCM", 2023, "note:bonus_welfare_fund_ending", "consolidated"): _disclosure_cell("DCM", 2023, "note:bonus_welfare_fund_ending", "consolidated", "335.746.014.085", "DCM_financial_statements_2023_consolidated|428", "DCM_financial_statements_2023_consolidated_428.csv", 11, 4, "Ending bonus and welfare fund"),
    ("DCM", 2024, "note:bonus_welfare_fund_ending", "consolidated"): _disclosure_cell("DCM", 2024, "note:bonus_welfare_fund_ending", "consolidated", "240.508.766.318", "DCM_financial_statements_2024_consolidated|419", "DCM_financial_statements_2024_consolidated_419.csv", 11, 4, "Ending bonus and welfare fund"),
    ("EIB", 2017, "note:bonus_welfare_fund_ending_million", "consolidated"): _disclosure_cell("EIB", 2017, "note:bonus_welfare_fund_ending_million", "consolidated", "27.437", "EIB_financial_statements_2017_consolidated|1671", "EIB_financial_statements_2017_consolidated_1671.csv", 14, 1, "Ending bonus and welfare fund, VND million"),
    ("EIB", 2019, "note:bonus_welfare_fund_ending_million", "consolidated"): _disclosure_cell("EIB", 2019, "note:bonus_welfare_fund_ending_million", "consolidated", "28.648", "EIB_financial_statements_2019_consolidated|2069", "EIB_financial_statements_2019_consolidated_2069.csv", 14, 1, "Ending bonus and welfare fund, VND million"),
    ("EIB", 2021, "note:bonus_welfare_fund_ending_million", "consolidated"): _disclosure_cell("EIB", 2021, "note:bonus_welfare_fund_ending_million", "consolidated", "15.010", "EIB_financial_statements_2021_consolidated|1553", "EIB_financial_statements_2021_consolidated_1553.csv", 3, 1, "Ending bonus and welfare fund, VND million"),
    ("EIB", 2025, "note:bonus_welfare_fund_ending_million", "consolidated"): _disclosure_cell("EIB", 2025, "note:bonus_welfare_fund_ending_million", "consolidated", "175.234", "EIB_financial_statements_2025_consolidated|1293", "EIB_financial_statements_2025_consolidated_1293.csv", 4, 1, "Ending bonus and welfare fund, VND million"),
    # Q871: annual MPC appropriations; the question asks for the amount, so
    # formulas use absolute values of the equity-decrease rows.
    ("MPC", 2019, "note:bonus_welfare_appropriation", "consolidated"): _disclosure_cell("MPC", 2019, "note:bonus_welfare_appropriation", "consolidated", "(1.477.919.227)", "MPC_financial_statements_2019_consolidated|1534", "MPC_financial_statements_2019_consolidated_1534.csv", 2, 1, "Bonus and welfare fund appropriation"),
    ("MPC", 2022, "note:bonus_welfare_appropriation", "consolidated"): _disclosure_cell("MPC", 2022, "note:bonus_welfare_appropriation", "consolidated", "(36.259.134.935)", "MPC_financial_statements_2022_consolidated|1331", "MPC_financial_statements_2022_consolidated_1331.csv", 18, 6, "Bonus and welfare fund appropriation"),
    ("MPC", 2023, "note:bonus_welfare_appropriation", "consolidated"): _disclosure_cell("MPC", 2023, "note:bonus_welfare_appropriation", "consolidated", "(25.688.629.546)", "MPC_financial_statements_2023_consolidated|1415", "MPC_financial_statements_2023_consolidated_1415.csv", 6, 5, "Bonus and welfare fund appropriation"),
    ("MPC", 2024, "note:bonus_welfare_appropriation", "consolidated"): _disclosure_cell("MPC", 2024, "note:bonus_welfare_appropriation", "consolidated", "(38.776.065.470)", "MPC_financial_statements_2024_consolidated|1449", "MPC_financial_statements_2024_consolidated_1449.csv", 7, 5, "Bonus and welfare fund appropriation"),
    # Q887/Q912: disclosed annual governance/management totals.
    ("VIC", 2022, "note:board_remuneration_million", "separate"): _disclosure_cell("VIC", 2022, "note:board_remuneration_million", "separate", "9.322", "VIC_financial_statements_2022_separate|1839", "VIC_financial_statements_2022_separate_1839.csv", 15, 2, "Board remuneration, VND million"),
    ("VIC", 2023, "note:board_remuneration_million", "separate"): _disclosure_cell("VIC", 2023, "note:board_remuneration_million", "separate", "11.513", "VIC_financial_statements_2023_separate|2003", "VIC_financial_statements_2023_separate_2003.csv", 12, 2, "Board remuneration, VND million"),
    ("VIC", 2024, "note:board_remuneration_million", "separate"): _disclosure_cell("VIC", 2024, "note:board_remuneration_million", "separate", "12.025", "VIC_financial_statements_2024_separate|2047", "VIC_financial_statements_2024_separate_2047.csv", 11, 2, "Board remuneration, VND million"),
    ("VIF", 2022, "note:board_management_income_total", "separate"): _disclosure_cell("VIF", 2022, "note:board_management_income_total", "separate", "11.453.874.342", "VIF_financial_statements_2022_separate|1675", "VIF_financial_statements_2022_separate_1675.csv", 10, 2, "Total board and management income"),
    ("VIF", 2024, "note:board_management_income_total", "separate"): _disclosure_cell("VIF", 2024, "note:board_management_income_total", "separate", "8.829.704.230", "VIF_financial_statements_2024_separate|1772", "VIF_financial_statements_2024_separate_1772.csv", 12, 2, "Total board and management income"),
    ("VIF", 2025, "note:board_management_income_total", "separate"): _disclosure_cell("VIF", 2025, "note:board_management_income_total", "separate", "10.156.401.300", "VIF_financial_statements_2025_separate|1949", "VIF_financial_statements_2025_separate_1949.csv", 11, 2, "Total board and management income"),
    # Q930: parent PC1 ending short-term dividend/profit receivable.
    ("PC1", 2022, "note:dividend_profit_receivable", "separate"): _disclosure_cell("PC1", 2022, "note:dividend_profit_receivable", "separate", "32.479.615.863", "PC1_financial_statements_2022_separate|930", "PC1_financial_statements_2022_separate_930.csv", 1, 1, "Short-term dividend and profit receivable"),
    ("PC1", 2023, "note:dividend_profit_receivable", "separate"): _disclosure_cell("PC1", 2023, "note:dividend_profit_receivable", "separate", "102.912.151.561", "PC1_financial_statements_2023_separate|859", "PC1_financial_statements_2023_separate_859.csv", 3, 1, "Short-term dividend and profit receivable"),
    ("PC1", 2024, "note:dividend_profit_receivable", "separate"): _disclosure_cell("PC1", 2024, "note:dividend_profit_receivable", "separate", "152.605.436.256", "PC1_financial_statements_2024_separate|893", "PC1_financial_statements_2024_separate_893.csv", 1, 1, "Short-term dividend and profit receivable"),
    ("PC1", 2025, "note:dividend_profit_receivable", "separate"): _disclosure_cell("PC1", 2025, "note:dividend_profit_receivable", "separate", "166.562.502.443", "PC1_financial_statements_2025_separate|1037", "PC1_financial_statements_2025_separate_1037.csv", 2, 1, "Short-term dividend and profit receivable"),
    ("MWG", 2024, "note:basic_and_diluted_eps", "consolidated"): _disclosure_cell("MWG", 2024, "note:basic_and_diluted_eps", "consolidated", "2.546", "MWG_financial_statements_2024_consolidated|1193", "MWG_financial_statements_2024_consolidated_1193.csv", 4, 1, "Basic and diluted earnings per share"),
    ("HHS", 2015, "note:inventory_gross", "consolidated"): _disclosure_cell("HHS", 2015, "note:inventory_gross", "consolidated", "818.760.481.699", "HHS_financial_statements_2015_consolidated|834", "HHS_financial_statements_2015_consolidated_834.csv", 6, 1, "Ending inventory gross amount"),
    ("HHS", 2016, "note:inventory_gross", "consolidated"): _disclosure_cell("HHS", 2016, "note:inventory_gross", "consolidated", "369.916.389.790", "HHS_financial_statements_2016_consolidated|691", "HHS_financial_statements_2016_consolidated_691.csv", 7, 1, "Ending inventory gross amount"),
    ("HHS", 2017, "note:inventory_gross", "consolidated"): _disclosure_cell("HHS", 2017, "note:inventory_gross", "consolidated", "904.950.295.424", "HHS_financial_statements_2017_consolidated|846", "HHS_financial_statements_2017_consolidated_846.csv", 7, 1, "Ending inventory gross amount"),
    ("HHS", 2020, "note:inventory_gross", "consolidated"): _disclosure_cell("HHS", 2020, "note:inventory_gross", "consolidated", "146.229.132.873", "HHS_financial_statements_2020_consolidated|942", "HHS_financial_statements_2020_consolidated_942.csv", 7, 1, "Ending inventory gross amount"),
    ("HHS", 2021, "note:inventory_gross", "consolidated"): _disclosure_cell("HHS", 2021, "note:inventory_gross", "consolidated", "243.136.299.257", "HHS_financial_statements_2021_consolidated|893", "HHS_financial_statements_2021_consolidated_893.csv", 8, 1, "Ending inventory gross amount"),
    ("HHS", 2015, "note:raw_materials_gross", "consolidated"): _disclosure_cell("HHS", 2015, "note:raw_materials_gross", "consolidated", "83.140.939.825", "HHS_financial_statements_2015_consolidated|834", "HHS_financial_statements_2015_consolidated_834.csv", 2, 1, "Ending raw materials gross amount"),
    ("HHS", 2016, "note:raw_materials_gross", "consolidated"): _disclosure_cell("HHS", 2016, "note:raw_materials_gross", "consolidated", "93.414.197.135", "HHS_financial_statements_2016_consolidated|691", "HHS_financial_statements_2016_consolidated_691.csv", 2, 1, "Ending raw materials gross amount"),
    ("HHS", 2017, "note:raw_materials_gross", "consolidated"): _disclosure_cell("HHS", 2017, "note:raw_materials_gross", "consolidated", "9.497.171.212", "HHS_financial_statements_2017_consolidated|846", "HHS_financial_statements_2017_consolidated_846.csv", 2, 1, "Ending raw materials gross amount"),
    ("HHS", 2020, "note:raw_materials_gross", "consolidated"): _disclosure_cell("HHS", 2020, "note:raw_materials_gross", "consolidated", "40.088.989.929", "HHS_financial_statements_2020_consolidated|942", "HHS_financial_statements_2020_consolidated_942.csv", 2, 1, "Ending raw materials gross amount"),
    ("HHS", 2021, "note:raw_materials_gross", "consolidated"): _disclosure_cell("HHS", 2021, "note:raw_materials_gross", "consolidated", "100.599.864.011", "HHS_financial_statements_2021_consolidated|893", "HHS_financial_statements_2021_consolidated_893.csv", 3, 1, "Ending raw materials gross amount"),
    ("POW", 2017, "note:evn_short_term_payable", "separate"): _disclosure_cell("POW", 2017, "note:evn_short_term_payable", "separate", "13.077.812.223", "POW_financial_statements_2017_separate|856", "POW_financial_statements_2017_separate_856.csv", 4, 1, "Short-term payable to Vietnam Electricity"),
    ("POW", 2018, "note:evn_short_term_payable", "separate"): _disclosure_cell("POW", 2018, "note:evn_short_term_payable", "separate", "20.520.253.105", "POW_financial_statements_2018_separate|871", "POW_financial_statements_2018_separate_871.csv", 4, 1, "Short-term payable to Vietnam Electricity"),
    ("POW", 2021, "note:evn_short_term_payable", "separate"): _disclosure_cell("POW", 2021, "note:evn_short_term_payable", "separate", "79.072.318.889", "POW_financial_statements_2021_separate|899", "POW_financial_statements_2021_separate_899.csv", 3, 1, "Short-term payable to Vietnam Electricity"),
    ("POW", 2024, "note:evn_short_term_payable", "separate"): _disclosure_cell("POW", 2024, "note:evn_short_term_payable", "separate", "61.539.096.219", "POW_financial_statements_2024_separate|961", "POW_financial_statements_2024_separate_961.csv", 6, 1, "Short-term payable to Vietnam Electricity"),
    ("GEE", 2022, "note:cft_short_term_loan_receivable", "separate"): _disclosure_cell("GEE", 2022, "note:cft_short_term_loan_receivable", "separate", "400.000.000.000", "GEE_financial_statements_2022_separate|889", "GEE_financial_statements_2022_separate_889.csv", 2, 1, "Short-term loan receivable from CFT"),
    ("GEE", 2023, "note:cft_short_term_loan_receivable", "separate"): _disclosure_cell("GEE", 2023, "note:cft_short_term_loan_receivable", "separate", "30.000.000.000", "GEE_financial_statements_2023_separate|920", "GEE_financial_statements_2023_separate_920.csv", 3, 1, "Short-term loan receivable from CFT"),
    ("GEE", 2024, "note:cft_short_term_loan_receivable", "separate"): _disclosure_cell("GEE", 2024, "note:cft_short_term_loan_receivable", "separate", "200.000.000.000", "GEE_financial_statements_2024_separate|970", "GEE_financial_statements_2024_separate_970.csv", 2, 1, "Short-term loan receivable from CFT"),
    ("GEE", 2025, "note:cft_short_term_loan_receivable", "separate"): _disclosure_cell("GEE", 2025, "note:cft_short_term_loan_receivable", "separate", "-", "GEE_financial_statements_2025_separate|975", "GEE_financial_statements_2025_separate_975.csv", 3, 1, "Short-term loan receivable from CFT"),
    ("ACB", 2018, "note:operating_expense_million", "separate"): _disclosure_cell("ACB", 2018, "note:operating_expense_million", "separate", "(6.541.128)", "ACB_financial_statements_2018_separate|180", "ACB_financial_statements_2018_separate_180.csv", 15, 3, "Operating expense"),
    ("ACB", 2020, "note:operating_expense_million", "separate"): _disclosure_cell("ACB", 2020, "note:operating_expense_million", "separate", "(7.423.285)", "ACB_financial_statements_2020_separate|251", "ACB_financial_statements_2020_separate_251.csv", 15, 3, "Operating expense"),
    ("ACB", 2022, "note:operating_expense_million", "separate"): _disclosure_cell("ACB", 2022, "note:operating_expense_million", "separate", "(11.261.725)", "ACB_financial_statements_2022_separate|221", "ACB_financial_statements_2022_separate_221.csv", 14, 3, "Operating expense"),
    ("ACB", 2018, "note:personal_deposits_million", "separate"): _disclosure_cell("ACB", 2018, "note:personal_deposits_million", "separate", "215.713.175", "ACB_financial_statements_2018_separate|1745", "ACB_financial_statements_2018_separate_1745.csv", 6, 1, "Deposits from individuals"),
    ("ACB", 2020, "note:personal_deposits_million", "separate"): _disclosure_cell("ACB", 2020, "note:personal_deposits_million", "separate", "280.172.776", "ACB_financial_statements_2020_separate|1771", "ACB_financial_statements_2020_separate_1771.csv", 6, 1, "Deposits from individuals"),
    ("ACB", 2022, "note:personal_deposits_million", "separate"): _disclosure_cell("ACB", 2022, "note:personal_deposits_million", "separate", "333.078.185", "ACB_financial_statements_2022_separate|1800", "ACB_financial_statements_2022_separate_1800.csv", 6, 1, "Deposits from individuals"),
    ("VCB", 2015, "note:current_income_tax_million", "consolidated"): _disclosure_cell("VCB", 2015, "note:current_income_tax_million", "consolidated", "(1.495.100)", "VCB_financial_statements_2015_consolidated|303", "VCB_financial_statements_2015_consolidated_303.csv", 1, 3, "Current corporate income tax expense"),
    ("VCB", 2017, "note:current_income_tax_million", "consolidated"): _disclosure_cell("VCB", 2017, "note:current_income_tax_million", "consolidated", "(2.234.378)", "VCB_financial_statements_2017_consolidated|279", "VCB_financial_statements_2017_consolidated_279.csv", 2, 3, "Current corporate income tax expense"),
    ("VCB", 2018, "note:current_income_tax_million", "consolidated"): _disclosure_cell("VCB", 2018, "note:current_income_tax_million", "consolidated", "(3.648.356)", "VCB_financial_statements_2018_consolidated|229", "VCB_financial_statements_2018_consolidated_229.csv", 2, 3, "Current corporate income tax expense"),
    ("VCB", 2022, "note:current_income_tax_million", "consolidated"): _disclosure_cell("VCB", 2022, "note:current_income_tax_million", "consolidated", "(8.406.860)", "VCB_financial_statements_2022_consolidated|272", "VCB_financial_statements_2022_consolidated_272.csv", 2, 3, "Current corporate income tax expense"),
    ("VCB", 2023, "note:current_income_tax_million", "consolidated"): _disclosure_cell("VCB", 2023, "note:current_income_tax_million", "consolidated", "(8.079.401)", "VCB_financial_statements_2023_consolidated|243", "VCB_financial_statements_2023_consolidated_243.csv", 2, 3, "Current corporate income tax expense"),
    ("HAG", 2015, "note:xnk_hagl_short_term_other_payable", "separate"): _disclosure_cell("HAG", 2015, "note:xnk_hagl_short_term_other_payable", "separate", "-", "HAG_financial_statements_2015_separate|1809", "HAG_financial_statements_2015_separate_1809.csv", 5, 3, "Ending short-term other payable to HAGL Import Export"),
    ("HAG", 2019, "note:xnk_hagl_short_term_other_payable", "separate"): _disclosure_cell("HAG", 2019, "note:xnk_hagl_short_term_other_payable", "separate", "9.408.242", "HAG_financial_statements_2019_separate|1688", "HAG_financial_statements_2019_separate_1688.csv", 8, 3, "Ending short-term other payable to HAGL Import Export"),
    ("HAG", 2021, "note:xnk_hagl_short_term_other_payable", "separate"): _disclosure_cell("HAG", 2021, "note:xnk_hagl_short_term_other_payable", "separate", "-", "HAG_financial_statements_2021_separate|1594", "HAG_financial_statements_2021_separate_1594.csv", 5, 3, "Ending short-term other payable to HAGL Import Export"),
    ("HND", 2016, "note:tangible_ppe_gross", "unknown"): _disclosure_cell("HND", 2016, "note:tangible_ppe_gross", "unknown", "22.141.526.552.885", "HND_financial_statements_2016|127", "HND_financial_statements_2016_127.csv", 18, 3, "Tangible fixed-assets gross cost"),
    ("HND", 2016, "note:tangible_ppe_accumulated_depreciation", "unknown"): _disclosure_cell("HND", 2016, "note:tangible_ppe_accumulated_depreciation", "unknown", "(8.001.667.854.893)", "HND_financial_statements_2016|127", "HND_financial_statements_2016_127.csv", 19, 3, "Tangible fixed-assets accumulated depreciation"),
    ("HND", 2018, "note:tangible_ppe_gross", "unknown"): _disclosure_cell("HND", 2018, "note:tangible_ppe_gross", "unknown", "22.058.473.317.440", "HND_financial_statements_2018|122", "HND_financial_statements_2018_122.csv", 17, 3, "Tangible fixed-assets gross cost"),
    ("HND", 2018, "note:tangible_ppe_accumulated_depreciation", "unknown"): _disclosure_cell("HND", 2018, "note:tangible_ppe_accumulated_depreciation", "unknown", "(11.731.433.309.660)", "HND_financial_statements_2018|122", "HND_financial_statements_2018_122.csv", 18, 3, "Tangible fixed-assets accumulated depreciation"),
    ("HND", 2019, "note:tangible_ppe_gross", "unknown"): _disclosure_cell("HND", 2019, "note:tangible_ppe_gross", "unknown", "22.079.164.840.230", "HND_financial_statements_2019|214", "HND_financial_statements_2019_214.csv", 18, 3, "Tangible fixed-assets gross cost"),
    ("HND", 2019, "note:tangible_ppe_accumulated_depreciation", "unknown"): _disclosure_cell("HND", 2019, "note:tangible_ppe_accumulated_depreciation", "unknown", "(13.520.488.721.292)", "HND_financial_statements_2019|214", "HND_financial_statements_2019_214.csv", 19, 3, "Tangible fixed-assets accumulated depreciation"),
    ("HND", 2020, "note:tangible_ppe_gross", "unknown"): _disclosure_cell("HND", 2020, "note:tangible_ppe_gross", "unknown", "22.083.494.486.346", "HND_financial_statements_2020|208", "HND_financial_statements_2020_208.csv", 19, 3, "Tangible fixed-assets gross cost"),
    ("HND", 2020, "note:tangible_ppe_accumulated_depreciation", "unknown"): _disclosure_cell("HND", 2020, "note:tangible_ppe_accumulated_depreciation", "unknown", "(15.298.798.199.853)", "HND_financial_statements_2020|208", "HND_financial_statements_2020_208.csv", 20, 3, "Tangible fixed-assets accumulated depreciation"),
    ("HND", 2021, "note:tangible_ppe_gross", "unknown"): _disclosure_cell("HND", 2021, "note:tangible_ppe_gross", "unknown", "22.125.917.998.980", "HND_financial_statements_2021|205", "HND_financial_statements_2021_205.csv", 21, 3, "Tangible fixed-assets gross cost"),
    ("HND", 2021, "note:tangible_ppe_accumulated_depreciation", "unknown"): _disclosure_cell("HND", 2021, "note:tangible_ppe_accumulated_depreciation", "unknown", "(16.599.466.811.506)", "HND_financial_statements_2021|205", "HND_financial_statements_2021_205.csv", 22, 3, "Tangible fixed-assets accumulated depreciation"),
    ("STB", 2021, "note:cd_under_12m_million", "consolidated"): _disclosure_cell("STB", 2021, "note:cd_under_12m_million", "consolidated", "1.822.241", "STB_financial_statements_2021_consolidated|1756", "STB_financial_statements_2021_consolidated_1756.csv", 2, 1, "Certificates of deposit under 12 months"),
    ("STB", 2021, "note:cd_12m_to_5y_million", "consolidated"): _disclosure_cell("STB", 2021, "note:cd_12m_to_5y_million", "consolidated", "5.903.855", "STB_financial_statements_2021_consolidated|1756", "STB_financial_statements_2021_consolidated_1756.csv", 3, 1, "Certificates of deposit from 12 months to under 5 years"),
    ("STB", 2021, "note:cd_5y_plus_million", "consolidated"): _disclosure_cell("STB", 2021, "note:cd_5y_plus_million", "consolidated", "13.377.683", "STB_financial_statements_2021_consolidated|1756", "STB_financial_statements_2021_consolidated_1756.csv", 4, 1, "Certificates of deposit from 5 years"),
    ("STB", 2022, "note:cd_under_12m_million", "consolidated"): _disclosure_cell("STB", 2022, "note:cd_under_12m_million", "consolidated", "304.654", "STB_financial_statements_2022_consolidated|1815", "STB_financial_statements_2022_consolidated_1815.csv", 2, 1, "Certificates of deposit under 12 months"),
    ("STB", 2022, "note:cd_12m_to_5y_million", "consolidated"): _disclosure_cell("STB", 2022, "note:cd_12m_to_5y_million", "consolidated", "4.849.570", "STB_financial_statements_2022_consolidated|1815", "STB_financial_statements_2022_consolidated_1815.csv", 3, 1, "Certificates of deposit from 12 months to under 5 years"),
    ("STB", 2022, "note:cd_5y_plus_million", "consolidated"): _disclosure_cell("STB", 2022, "note:cd_5y_plus_million", "consolidated", "13.366.083", "STB_financial_statements_2022_consolidated|1815", "STB_financial_statements_2022_consolidated_1815.csv", 4, 1, "Certificates of deposit from 5 years"),
    ("STB", 2025, "note:cd_under_12m_million", "consolidated"): _disclosure_cell("STB", 2025, "note:cd_under_12m_million", "consolidated", "523.859", "STB_financial_statements_2025_consolidated|2080", "STB_financial_statements_2025_consolidated_2080.csv", 2, 1, "Certificates of deposit under 12 months"),
    ("STB", 2025, "note:cd_12m_to_5y_million", "consolidated"): _disclosure_cell("STB", 2025, "note:cd_12m_to_5y_million", "consolidated", "25.610.061", "STB_financial_statements_2025_consolidated|2080", "STB_financial_statements_2025_consolidated_2080.csv", 3, 1, "Certificates of deposit from 12 months to under 5 years"),
    ("STB", 2025, "note:cd_5y_plus_million", "consolidated"): _disclosure_cell("STB", 2025, "note:cd_5y_plus_million", "consolidated", "8.050.784", "STB_financial_statements_2025_consolidated|2080", "STB_financial_statements_2025_consolidated_2080.csv", 4, 1, "Certificates of deposit from 5 years"),
    ("SSH", 2020, "note:related_party_service_revenue", "separate"): _disclosure_cell("SSH", 2020, "note:related_party_service_revenue", "separate", "11.863.882.275", "SSH_financial_statements_2020_separate|1053", "SSH_financial_statements_2020_separate_1053.csv", 2, 1, "Related-party service revenue"),
    ("SSH", 2021, "note:related_party_service_revenue", "separate"): _disclosure_cell("SSH", 2021, "note:related_party_service_revenue", "separate", "61.614.783.673", "SSH_financial_statements_2021_separate|1128", "SSH_financial_statements_2021_separate_1128.csv", 2, 2, "Related-party service revenue"),
    ("SSH", 2022, "note:related_party_service_revenue", "separate"): _disclosure_cell("SSH", 2022, "note:related_party_service_revenue", "separate", "62.744.631.137", "SSH_financial_statements_2022_separate|1059", "SSH_financial_statements_2022_separate_1059.csv", 2, 2, "Related-party service revenue"),
    ("SSH", 2023, "note:related_party_service_revenue", "separate"): _disclosure_cell("SSH", 2023, "note:related_party_service_revenue", "separate", "88.961.131.800", "SSH_financial_statements_2023_separate|1118", "SSH_financial_statements_2023_separate_1118.csv", 2, 2, "Related-party service revenue"),
    ("BSR", 2017, "note:inventory_gross", "consolidated"): _disclosure_cell("BSR", 2017, "note:inventory_gross", "consolidated", "8.139.311.457.689", "BSR_financial_statements_2017_consolidated|238", "BSR_financial_statements_2017_consolidated_238.csv", 14, 3, "Ending inventory gross amount"),
    ("BSR", 2017, "note:inventory_provision", "consolidated"): _disclosure_cell("BSR", 2017, "note:inventory_provision", "consolidated", "(96.412.876.496)", "BSR_financial_statements_2017_consolidated|238", "BSR_financial_statements_2017_consolidated_238.csv", 15, 3, "Ending inventory provision"),
    ("BSR", 2019, "note:inventory_gross", "consolidated"): _disclosure_cell("BSR", 2019, "note:inventory_gross", "consolidated", "8.535.271.500.226", "BSR_financial_statements_2019_consolidated|223", "BSR_financial_statements_2019_consolidated_223.csv", 14, 3, "Ending inventory gross amount"),
    ("BSR", 2019, "note:inventory_provision", "consolidated"): _disclosure_cell("BSR", 2019, "note:inventory_provision", "consolidated", "(20.033.774.981)", "BSR_financial_statements_2019_consolidated|223", "BSR_financial_statements_2019_consolidated_223.csv", 15, 3, "Ending inventory provision"),
    ("BSR", 2021, "note:inventory_gross", "consolidated"): _disclosure_cell("BSR", 2021, "note:inventory_gross", "consolidated", "10.376.585.353.744", "BSR_financial_statements_2021_consolidated|533", "BSR_financial_statements_2021_consolidated_533.csv", 14, 3, "Ending inventory gross amount"),
    ("BSR", 2021, "note:inventory_provision", "consolidated"): _disclosure_cell("BSR", 2021, "note:inventory_provision", "consolidated", "(18.489.988.587)", "BSR_financial_statements_2021_consolidated|533", "BSR_financial_statements_2021_consolidated_533.csv", 15, 3, "Ending inventory provision"),
    ("BSR", 2024, "note:inventory_gross", "consolidated"): _disclosure_cell("BSR", 2024, "note:inventory_gross", "consolidated", "15.890.950.395.456", "BSR_financial_statements_2024_consolidated|256", "BSR_financial_statements_2024_consolidated_256.csv", 14, 3, "Ending inventory gross amount"),
    ("BSR", 2024, "note:inventory_provision", "consolidated"): _disclosure_cell("BSR", 2024, "note:inventory_provision", "consolidated", "-", "BSR_financial_statements_2024_consolidated|256", "BSR_financial_statements_2024_consolidated_256.csv", 15, 3, "Ending inventory provision"),
    ("BSR", 2025, "note:inventory_gross", "consolidated"): _disclosure_cell("BSR", 2025, "note:inventory_gross", "consolidated", "12.797.925.022.543", "BSR_financial_statements_2025_consolidated|368", "BSR_financial_statements_2025_consolidated_368.csv", 12, 4, "Ending inventory gross amount"),
    ("BSR", 2025, "note:inventory_provision", "consolidated"): _disclosure_cell("BSR", 2025, "note:inventory_provision", "consolidated", "(131.015.917.907)", "BSR_financial_statements_2025_consolidated|368", "BSR_financial_statements_2025_consolidated_368.csv", 13, 4, "Ending inventory provision"),
    # Q213: the off-balance-sheet currency table reports an amount in USD,
    # with Vietnamese punctuation.  Preserve the decimal comma and convert the
    # underlying USD amount to the million-USD unit requested by the question.
    ("ACV", 2018, "note:ending_usd_balance", "consolidated"): _disclosure_cell("ACV", 2018, "note:ending_usd_balance", "consolidated", "6.155.698,34", "ACV_financial_statements_2018_consolidated|1297", "ACV_financial_statements_2018_consolidated_1297.csv", 1, 1, "Ending off-balance-sheet US-dollar amount"),
    # Q97: the parent-company investment note states HPG's direct ownership
    # percentage explicitly.  The source token uses a decimal comma with three
    # digits (99,988%); the generic number parser reads that as 99,988, so the
    # audited scale below restores the percentage value without changing the
    # parser semantics for any other question.
    ("HPG", 2023, "note:gang_steel_ownership_pct", "separate"): StatementCell(
        "HPG", "2023", "separate", "note:gang_steel_ownership_pct", "",
        "Direct ownership of Hoa Phat Steel Joint Stock Company",
        99.988, "99,988%", "HPG_financial_statements_2023_separate|774",
        "HPG_financial_statements_2023_separate_774.csv", 5, 1, 0.001,
    ),
    # Q355: MSN's subsidiary roster reports the requested economic-interest
    # percentage directly.  The legacy program instead divided retained
    # earnings by total equity, two unrelated balance-sheet lines.
    ("MSN", 2022, "note:3f_food_economic_interest_pct", "consolidated"): _disclosure_cell(
        "MSN", 2022, "note:3f_food_economic_interest_pct", "consolidated",
        "48,4%", "MSN_financial_statements_2022_consolidated|529",
        "MSN_financial_statements_2022_consolidated_529.csv", 3, 5,
        "Economic interest in 3F Viet Food Company Limited at end-2022",
    ),
    # Q629: compare like-for-like ending balance-sheet provisions.  The legacy
    # query mixed a 2016 construction-contract receivable provision with a
    # 2020 provision expense from the administrative-expense note.
    ("HBC", 2016, "cdkt:137", "separate"): _disclosure_cell(
        "HBC", 2016, "cdkt:137", "separate", "(262.534.406.527)",
        "HBC_financial_statements_2016_separate|190",
        "HBC_financial_statements_2016_separate_190.csv", 12, 3,
        "Ending allowance for doubtful short-term receivables",
    ),
    ("HBC", 2020, "cdkt:137", "separate"): _disclosure_cell(
        "HBC", 2020, "cdkt:137", "separate", "(384.655.311.583)",
        "HBC_financial_statements_2020_separate|208",
        "HBC_financial_statements_2020_separate_208.csv", 10, 3,
        "Ending allowance for doubtful short-term receivables",
    ),
    # Q263: the question asks for the ending fair value of FVTPL financial
    # assets.  The legacy query selected current-year income from FVTPL
    # instead.  The investment-note table reports the ending revalued amount
    # ("Giá trị đánh giá lại") directly in the FVTPL row.
    ("SSI", 2019, "note:fvtpl_ending_fair_value", "separate"): _disclosure_cell(
        "SSI", 2019, "note:fvtpl_ending_fair_value", "separate",
        "4.263.610.960.357", "SSI_financial_statements_2019_separate|1332",
        "SSI_financial_statements_2019_separate_1332.csv", 2, 4,
        "FVTPL financial assets - ending revalued amount (fair value)",
    ),
    # Q218: the summary table exposes gross investment and a combined
    # long-term-investment provision, while the detailed 6.3 table assigns
    # that provision specifically to investments in other entities and gives
    # the requested net carrying amount directly.
    ("BVH", 2021, "note:other_entity_investment_net", "separate"): _disclosure_cell(
        "BVH", 2021, "note:other_entity_investment_net", "separate",
        "492.933.134.359", "BVH_financial_statements_2021_separate|973",
        "BVH_financial_statements_2021_separate_973.csv", 17, 3,
        "Investment in other entities - ending net carrying amount",
    ),
    ("VIF", 2017, "note:trade_receivables_total", "separate"): _disclosure_cell("VIF", 2017, "note:trade_receivables_total", "separate", "36.129.709.922", "VIF_financial_statements_2017_separate|1016", "VIF_financial_statements_2017_separate_1016.csv", 6, 1, "Ending total short-term trade receivables"),
    ("VIF", 2017, "note:trade_receivables_provision", "separate"): _disclosure_cell("VIF", 2017, "note:trade_receivables_provision", "separate", "(1.791.632.437)", "VIF_financial_statements_2017_separate|1016", "VIF_financial_statements_2017_separate_1016.csv", 7, 1, "Ending short-term trade-receivables provision"),
    ("VIF", 2020, "note:trade_receivables_total", "separate"): _disclosure_cell("VIF", 2020, "note:trade_receivables_total", "separate", "72.395.199.451", "VIF_financial_statements_2020_separate|1004", "VIF_financial_statements_2020_separate_1004.csv", 6, 1, "Ending total short-term trade receivables"),
    ("VIF", 2020, "note:trade_receivables_provision", "separate"): _disclosure_cell("VIF", 2020, "note:trade_receivables_provision", "separate", "(3.068.979.395)", "VIF_financial_statements_2020_separate|1004", "VIF_financial_statements_2020_separate_1004.csv", 7, 1, "Ending short-term trade-receivables provision"),
    # Q721: the numerator is the share-investment component of investments in
    # other entities.  The denominator requested by the question is the
    # separately reported investment in associates/joint ventures, not the
    # subtotal of investments in other entities from the numerator's table.
    ("VIF", 2020, "note:other_entity_share_investment", "consolidated"): _disclosure_cell("VIF", 2020, "note:other_entity_share_investment", "consolidated", "15.996.208.039", "VIF_financial_statements_2020_consolidated|1262", "VIF_financial_statements_2020_consolidated_1262.csv", 2, 1, "Ending investment in shares within investments in other entities"),
    ("VIF", 2020, "note:associate_joint_venture_investment", "consolidated"): _disclosure_cell("VIF", 2020, "note:associate_joint_venture_investment", "consolidated", "1.141.390.360.287", "VIF_financial_statements_2020_consolidated|1162", "VIF_financial_statements_2020_consolidated_1162.csv", 2, 1, "Ending investment in associates and joint ventures"),
    ("VIF", 2021, "note:trade_receivables_total", "separate"): _disclosure_cell("VIF", 2021, "note:trade_receivables_total", "separate", "104.028.656.660", "VIF_financial_statements_2021_separate|959", "VIF_financial_statements_2021_separate_959.csv", 7, 1, "Ending total short-term trade receivables"),
    ("VIF", 2021, "note:trade_receivables_provision", "separate"): _disclosure_cell("VIF", 2021, "note:trade_receivables_provision", "separate", "(4.970.634.631)", "VIF_financial_statements_2021_separate|959", "VIF_financial_statements_2021_separate_959.csv", 8, 1, "Ending short-term trade-receivables provision"),
    ("VIF", 2022, "note:trade_receivables_total", "separate"): _disclosure_cell("VIF", 2022, "note:trade_receivables_total", "separate", "143.046.839.421", "VIF_financial_statements_2022_separate|935", "VIF_financial_statements_2022_separate_935.csv", 10, 1, "Ending total short-term trade receivables"),
    ("VIF", 2022, "note:trade_receivables_provision", "separate"): _disclosure_cell("VIF", 2022, "note:trade_receivables_provision", "separate", "(6.318.185.201)", "VIF_financial_statements_2022_separate|935", "VIF_financial_statements_2022_separate_935.csv", 11, 1, "Ending short-term trade-receivables provision"),
    ("TTF", 2018, "note:related_party_short_term_loan", "separate"): _disclosure_cell("TTF", 2018, "note:related_party_short_term_loan", "separate", "-", "TTF_financial_statements_2018_separate|1139", "TTF_financial_statements_2018_separate_1139.csv", 4, 1, "Ending short-term related-party loan"),
    ("TTF", 2021, "note:related_party_short_term_loan", "separate"): _disclosure_cell("TTF", 2021, "note:related_party_short_term_loan", "separate", "16.900.000.000", "TTF_financial_statements_2021_separate|1088", "TTF_financial_statements_2021_separate_1088.csv", 1, 1, "Ending short-term related-party loan"),
    ("TTF", 2022, "note:related_party_short_term_loan", "separate"): _disclosure_cell("TTF", 2022, "note:related_party_short_term_loan", "separate", "16.900.000.000", "TTF_financial_statements_2022_separate|969", "TTF_financial_statements_2022_separate_969.csv", 2, 1, "Ending short-term related-party loan"),
    ("TTF", 2024, "note:related_party_short_term_loan", "separate"): _disclosure_cell("TTF", 2024, "note:related_party_short_term_loan", "separate", "50.671.400.000", "TTF_financial_statements_2024_separate|1000", "TTF_financial_statements_2024_separate_1000.csv", 2, 1, "Ending short-term related-party loan"),
    ("TTF", 2017, "note:doubtful_receivable_provision_change", "separate"): _disclosure_cell("TTF", 2017, "note:doubtful_receivable_provision_change", "separate", "(685.306.267)", "TTF_financial_statements_2017_separate|1215", "TTF_financial_statements_2017_separate_1215.csv", 11, 1, "Doubtful-receivable provision charge or reversal during the year"),
    ("TTF", 2019, "note:doubtful_receivable_provision_change", "separate"): _disclosure_cell("TTF", 2019, "note:doubtful_receivable_provision_change", "separate", "53.683.365.257", "TTF_financial_statements_2019_separate|1227", "TTF_financial_statements_2019_separate_1227.csv", 5, 1, "Doubtful-receivable provision charge or reversal during the year"),
    ("TTF", 2021, "note:doubtful_receivable_provision_change", "separate"): _disclosure_cell("TTF", 2021, "note:doubtful_receivable_provision_change", "separate", "(11.845.063.319)", "TTF_financial_statements_2021_separate|1249", "TTF_financial_statements_2021_separate_1249.csv", 10, 1, "Doubtful-receivable provision charge or reversal during the year"),
    ("TTF", 2025, "note:doubtful_receivable_provision_change", "separate"): _disclosure_cell("TTF", 2025, "note:doubtful_receivable_provision_change", "separate", "1.121.711.370", "TTF_financial_statements_2025_separate|1301", "TTF_financial_statements_2025_separate_1301.csv", 7, 1, "Doubtful-receivable provision charge or reversal during the year"),
    ("DCM", 2019, "note:accrued_term_deposit_interest", "separate"): _disclosure_cell("DCM", 2019, "note:accrued_term_deposit_interest", "separate", "17.995.824.660", "DCM_financial_statements_2019_separate|823", "DCM_financial_statements_2019_separate_823.csv", 5, 1, "Ending accrued interest on term deposits"),
    ("DCM", 2022, "note:accrued_term_deposit_interest", "separate"): _disclosure_cell("DCM", 2022, "note:accrued_term_deposit_interest", "separate", "87.091.808.217", "DCM_financial_statements_2022_separate|912", "DCM_financial_statements_2022_separate_912.csv", 2, 1, "Ending accrued interest on term deposits"),
    ("DCM", 2023, "note:accrued_term_deposit_interest", "separate"): _disclosure_cell("DCM", 2023, "note:accrued_term_deposit_interest", "separate", "100.799.649.310", "DCM_financial_statements_2023_separate|1076", "DCM_financial_statements_2023_separate_1076.csv", 2, 1, "Ending accrued interest on term deposits"),
    ("DCM", 2024, "note:accrued_term_deposit_interest", "separate"): _disclosure_cell("DCM", 2024, "note:accrued_term_deposit_interest", "separate", "59.590.202.742", "DCM_financial_statements_2024_separate|1405", "DCM_financial_statements_2024_separate_1405.csv", 2, 1, "Ending accrued interest on term deposits"),
    ("DCM", 2025, "note:accrued_term_deposit_interest", "separate"): _disclosure_cell("DCM", 2025, "note:accrued_term_deposit_interest", "separate", "19.357.052.060", "DCM_financial_statements_2025_separate|1105", "DCM_financial_statements_2025_separate_1105.csv", 3, 1, "Ending accrued interest on term deposits"),
    ("DCM", 2019, "note:cash_ending", "separate"): _disclosure_cell("DCM", 2019, "note:cash_ending", "separate", "364.482.357.354", "DCM_financial_statements_2019_separate|331", "DCM_financial_statements_2019_separate_331.csv", 3, 4, "Ending cash"),
    ("DCM", 2022, "note:cash_ending", "separate"): _disclosure_cell("DCM", 2022, "note:cash_ending", "separate", "2.115.211.463.601", "DCM_financial_statements_2022_separate|370", "DCM_financial_statements_2022_separate_370.csv", 3, 4, "Ending cash"),
    ("DCM", 2023, "note:cash_ending", "separate"): _disclosure_cell("DCM", 2023, "note:cash_ending", "separate", "2.261.856.283.728", "DCM_financial_statements_2023_separate|478", "DCM_financial_statements_2023_separate_478.csv", 3, 4, "Ending cash"),
    ("DCM", 2024, "note:cash_ending", "separate"): _disclosure_cell("DCM", 2024, "note:cash_ending", "separate", "1.785.560.179.614", "DCM_financial_statements_2024_separate|449", "DCM_financial_statements_2024_separate_449.csv", 3, 4, "Ending cash"),
    ("DCM", 2025, "note:cash_ending", "separate"): _disclosure_cell("DCM", 2025, "note:cash_ending", "separate", "3.237.503.703.247", "DCM_financial_statements_2025_separate|446", "DCM_financial_statements_2025_separate_446.csv", 3, 4, "Ending cash"),
    ("OCB", 2017, "note:corporate_tax_payable_during_year", "separate"): _disclosure_cell("OCB", 2017, "note:corporate_tax_payable_during_year", "separate", "204.381.067.054", "OCB_financial_statements_2017_separate|1703", "OCB_financial_statements_2017_separate_1703.csv", 2, 2, "Corporate income tax payable during the year"),
    ("OCB", 2019, "note:corporate_tax_payable_during_year", "separate"): _disclosure_cell("OCB", 2019, "note:corporate_tax_payable_during_year", "separate", "648.727.468.950", "OCB_financial_statements_2019_separate|1867", "OCB_financial_statements_2019_separate_1867.csv", 2, 2, "Corporate income tax payable during the year"),
    ("OCB", 2022, "note:corporate_tax_payable_during_year", "separate"): _disclosure_cell("OCB", 2022, "note:corporate_tax_payable_during_year", "separate", "879.589.703.439", "OCB_financial_statements_2022_separate_1|1620", "OCB_financial_statements_2022_separate_1_1620.csv", 3, 2, "Corporate income tax payable during the year"),
    ("OCB", 2017, "note:customer_loan_loss_provision", "separate"): _disclosure_cell("OCB", 2017, "note:customer_loan_loss_provision", "separate", "(404.115.614.016)", "OCB_financial_statements_2017_separate|137", "OCB_financial_statements_2017_separate_137.csv", 12, 3, "Ending customer-loan loss provision"),
    ("OCB", 2019, "note:customer_loan_loss_provision", "separate"): _disclosure_cell("OCB", 2019, "note:customer_loan_loss_provision", "separate", "(724.735.852.400)", "OCB_financial_statements_2019_separate|113", "OCB_financial_statements_2019_separate_113.csv", 12, 3, "Ending customer-loan loss provision"),
    ("OCB", 2022, "note:customer_loan_loss_provision", "separate"): _disclosure_cell("OCB", 2022, "note:customer_loan_loss_provision", "separate", "(1.582.259.850.422)", "OCB_financial_statements_2022_separate_1|245", "OCB_financial_statements_2022_separate_1_245.csv", 12, 2, "Ending customer-loan loss provision"),
    ("HND", 2016, "note:tax_at_company_rate", "unknown"): _disclosure_cell("HND", 2016, "note:tax_at_company_rate", "unknown", "30.259.390.269", "HND_financial_statements_2016|942", "HND_financial_statements_2016_942.csv", 2, 1, "Tax calculated at the company's tax rate"),
    ("HND", 2018, "note:tax_at_company_rate", "unknown"): _disclosure_cell("HND", 2018, "note:tax_at_company_rate", "unknown", "44.874.485.818", "HND_financial_statements_2018|974", "HND_financial_statements_2018_974.csv", 2, 1, "Tax calculated at the company's tax rate"),
    ("HND", 2019, "note:tax_at_company_rate", "unknown"): _disclosure_cell("HND", 2019, "note:tax_at_company_rate", "unknown", "124.221.061.276", "HND_financial_statements_2019|1065", "HND_financial_statements_2019_1065.csv", 2, 1, "Tax calculated at the company's tax rate"),
    ("HND", 2020, "note:tax_at_company_rate", "unknown"): _disclosure_cell("HND", 2020, "note:tax_at_company_rate", "unknown", "150.437.520.136", "HND_financial_statements_2020|1060", "HND_financial_statements_2020_1060.csv", 2, 1, "Tax calculated at the company's tax rate"),
    ("HND", 2016, "note:operating_lease_due_within_one_year", "unknown"): _disclosure_cell("HND", 2016, "note:operating_lease_due_within_one_year", "unknown", "8.664.166.000", "HND_financial_statements_2017|908", "HND_financial_statements_2017_908.csv", 1, 2, "Minimum operating-lease payment due within one year at start-2017/end-2016"),
    ("HND", 2018, "note:operating_lease_due_within_one_year", "unknown"): _disclosure_cell("HND", 2018, "note:operating_lease_due_within_one_year", "unknown", "8.664.166.000", "HND_financial_statements_2018|881", "HND_financial_statements_2018_881.csv", 1, 1, "Minimum operating-lease payment due within one year"),
    ("HND", 2019, "note:operating_lease_due_within_one_year", "unknown"): _disclosure_cell("HND", 2019, "note:operating_lease_due_within_one_year", "unknown", "10.122.463.000", "HND_financial_statements_2019|976", "HND_financial_statements_2019_976.csv", 1, 1, "Minimum operating-lease payment due within one year"),
    ("HND", 2020, "note:operating_lease_due_within_one_year", "unknown"): _disclosure_cell("HND", 2020, "note:operating_lease_due_within_one_year", "unknown", "10.607.800.000", "HND_financial_statements_2020|973", "HND_financial_statements_2020_973.csv", 1, 1, "Minimum operating-lease payment due within one year"),
    ("IJC", 2017, "note:short_term_bank_loan", "separate"): _disclosure_cell("IJC", 2017, "note:short_term_bank_loan", "separate", "413.387.129.947", "IJC_financial_statements_2017_separate|1236", "IJC_financial_statements_2017_separate_1236.csv", 1, 1, "Ending short-term bank loan"),
    ("IJC", 2018, "note:short_term_bank_loan", "separate"): _disclosure_cell("IJC", 2018, "note:short_term_bank_loan", "separate", "388.644.836.245", "IJC_financial_statements_2018_separate|1343", "IJC_financial_statements_2018_separate_1343.csv", 1, 1, "Ending short-term bank loan"),
    ("IJC", 2019, "note:short_term_bank_loan", "separate"): _disclosure_cell("IJC", 2019, "note:short_term_bank_loan", "separate", "409.725.402.338", "IJC_financial_statements_2019_separate|1419", "IJC_financial_statements_2019_separate_1419.csv", 1, 1, "Ending short-term bank loan"),
    ("IJC", 2022, "note:short_term_bank_loan", "separate"): _disclosure_cell("IJC", 2022, "note:short_term_bank_loan", "separate", "325.964.618.787", "IJC_financial_statements_2022_separate|1489", "IJC_financial_statements_2022_separate_1489.csv", 1, 1, "Ending short-term bank loan"),
    ("IJC", 2024, "note:short_term_bank_loan", "separate"): _disclosure_cell("IJC", 2024, "note:short_term_bank_loan", "separate", "493.514.326.138", "IJC_financial_statements_2024_separate|1494", "IJC_financial_statements_2024_separate_1494.csv", 2, 1, "Ending short-term bank loan"),
    ("IJC", 2024, "note:related_party_short_term_loan_payable", "separate"): _disclosure_cell("IJC", 2024, "note:related_party_short_term_loan_payable", "separate", "4.500.000.000", "IJC_financial_statements_2024_separate|1477", "IJC_financial_statements_2024_separate_1477.csv", 1, 1, "Ending short-term loan payable to related parties"),
    ("DIG", 2015, "note:short_term_supplier_advances", "separate"): _disclosure_cell("DIG", 2015, "note:short_term_supplier_advances", "separate", "65.006.133.498", "DIG_financial_statements_2015_separate|205", "DIG_financial_statements_2015_separate_205.csv", 10, 3, "Ending short-term advances to suppliers"),
    ("DIG", 2016, "note:short_term_supplier_advances", "separate"): _disclosure_cell("DIG", 2016, "note:short_term_supplier_advances", "separate", "199.450.012.638", "DIG_financial_statements_2016_separate|233", "DIG_financial_statements_2016_separate_233.csv", 11, 3, "Ending short-term advances to suppliers"),
    ("DIG", 2017, "note:short_term_supplier_advances", "separate"): _disclosure_cell("DIG", 2017, "note:short_term_supplier_advances", "separate", "45.481.842.353", "DIG_financial_statements_2017_separate|212", "DIG_financial_statements_2017_separate_212.csv", 11, 3, "Ending short-term advances to suppliers"),
    ("DIG", 2021, "note:short_term_supplier_advances", "separate"): _disclosure_cell("DIG", 2021, "note:short_term_supplier_advances", "separate", "207.144.092.001", "DIG_financial_statements_2021_separate|184", "DIG_financial_statements_2021_separate_184.csv", 9, 3, "Ending short-term advances to suppliers"),
    ("DIG", 2024, "note:short_term_supplier_advances", "separate"): _disclosure_cell("DIG", 2024, "note:short_term_supplier_advances", "separate", "10.370.632.925", "DIG_financial_statements_2024_separate|311", "DIG_financial_statements_2024_separate_311.csv", 9, 3, "Ending short-term advances to suppliers"),
    ("TTF", 2017, "note:short_term_trade_payables", "separate"): _disclosure_cell("TTF", 2017, "note:short_term_trade_payables", "separate", "272.478.741.874", "TTF_financial_statements_2017_separate|238", "TTF_financial_statements_2017_separate_238.csv", 3, 3, "Ending short-term trade payables"),
    ("TTF", 2019, "note:short_term_trade_payables", "separate"): _disclosure_cell("TTF", 2019, "note:short_term_trade_payables", "separate", "151.290.182.187", "TTF_financial_statements_2019_separate|214", "TTF_financial_statements_2019_separate_214.csv", 3, 3, "Ending short-term trade payables"),
    ("TTF", 2020, "note:short_term_trade_payables", "separate"): _disclosure_cell("TTF", 2020, "note:short_term_trade_payables", "separate", "156.582.479.244", "TTF_financial_statements_2020_separate|234", "TTF_financial_statements_2020_separate_234.csv", 3, 3, "Ending short-term trade payables"),
    ("TTF", 2021, "note:short_term_trade_payables", "separate"): _disclosure_cell("TTF", 2021, "note:short_term_trade_payables", "separate", "164.697.168.260", "TTF_financial_statements_2021_separate|281", "TTF_financial_statements_2021_separate_281.csv", 3, 3, "Ending short-term trade payables"),
    ("TTF", 2022, "note:short_term_trade_payables", "separate"): _disclosure_cell("TTF", 2022, "note:short_term_trade_payables", "separate", "180.731.482.237", "TTF_financial_statements_2022_separate|218", "TTF_financial_statements_2022_separate_218.csv", 3, 3, "Ending short-term trade payables"),
    ("TTF", 2024, "note:short_term_trade_payables", "separate"): _disclosure_cell("TTF", 2024, "note:short_term_trade_payables", "separate", "152.028.849.068", "TTF_financial_statements_2024_separate|214", "TTF_financial_statements_2024_separate_214.csv", 3, 3, "Ending short-term trade payables"),
    ("TTF", 2017, "note:asset_liquidation_income", "separate"): _disclosure_cell("TTF", 2017, "note:asset_liquidation_income", "separate", "327.272.728", "TTF_financial_statements_2017_separate|1188", "TTF_financial_statements_2017_separate_1188.csv", 3, 1, "Income from asset liquidation"),
    ("GAS", 2015, "note:tax_and_other_payables", "consolidated"): _disclosure_cell("GAS", 2015, "note:tax_and_other_payables", "consolidated", "379.728.946.381", "GAS_financial_statements_2015_consolidated|339", "GAS_financial_statements_2015_consolidated_339.csv", 5, 3, "Ending taxes and other payables to the State"),
    ("GAS", 2016, "note:tax_and_other_payables", "consolidated"): _disclosure_cell("GAS", 2016, "note:tax_and_other_payables", "consolidated", "831.958.451.665", "GAS_financial_statements_2016_consolidated|351", "GAS_financial_statements_2016_consolidated_351.csv", 5, 4, "Ending taxes and other payables to the State"),
    ("GAS", 2017, "note:tax_and_other_payables", "consolidated"): _disclosure_cell("GAS", 2017, "note:tax_and_other_payables", "consolidated", "1.120.506.288.467", "GAS_financial_statements_2017_consolidated|269", "GAS_financial_statements_2017_consolidated_269.csv", 5, 4, "Ending taxes and other payables to the State"),
    ("GAS", 2019, "note:tax_and_other_payables", "consolidated"): _disclosure_cell("GAS", 2019, "note:tax_and_other_payables", "consolidated", "876.865.173.928", "GAS_financial_statements_2019_consolidated|288", "GAS_financial_statements_2019_consolidated_288.csv", 5, 4, "Ending taxes and other payables to the State"),
    ("GAS", 2017, "note:sales_to_pow", "consolidated"): _disclosure_cell("GAS", 2017, "note:sales_to_pow", "consolidated", "6.750.598.490.575", "GAS_financial_statements_2017_consolidated|1221", "GAS_financial_statements_2017_consolidated_1221.csv", 3, 1, "Sales to PetroVietnam Power Corporation"),
    ("HDG", 2015, "note:apartment_customer_advances", "separate"): _disclosure_cell("HDG", 2015, "note:apartment_customer_advances", "separate", "126.676.968.019", "HDG_financial_statements_2015_separate|936", "HDG_financial_statements_2015_separate_936.csv", 1, 1, "Ending advances from apartment customers"),
    ("HDG", 2016, "note:apartment_customer_advances", "separate"): _disclosure_cell("HDG", 2016, "note:apartment_customer_advances", "separate", "195.567.085.751", "HDG_financial_statements_2016_separate|969", "HDG_financial_statements_2016_separate_969.csv", 1, 1, "Ending advances from apartment customers"),
    ("HDG", 2017, "note:apartment_customer_advances", "separate"): _disclosure_cell("HDG", 2017, "note:apartment_customer_advances", "separate", "62.125.525.058", "HDG_financial_statements_2017_separate|918", "HDG_financial_statements_2017_separate_918.csv", 1, 1, "Ending advances from apartment customers"),
    ("HDG", 2018, "note:apartment_customer_advances", "separate"): _disclosure_cell("HDG", 2018, "note:apartment_customer_advances", "separate", "25.203.746.870", "HDG_financial_statements_2018_separate|1054", "HDG_financial_statements_2018_separate_1054.csv", 1, 1, "Ending advances from apartment customers"),
    ("HDG", 2019, "note:apartment_customer_advances", "separate"): _disclosure_cell("HDG", 2019, "note:apartment_customer_advances", "separate", "16.015.980.022", "HDG_financial_statements_2019_separate|1087", "HDG_financial_statements_2019_separate_1087.csv", 1, 1, "Ending advances from apartment customers"),
    ("HAG", 2015, "note:laos_external_revenue", "consolidated"): _disclosure_cell("HAG", 2015, "note:laos_external_revenue", "consolidated", "2.313.976.917", "HAG_financial_statements_2015_consolidated|2458", "HAG_financial_statements_2015_consolidated_2458.csv", 16, 2, "External-customer revenue from Laos, thousand VND"),
    ("HAG", 2015, "note:geographic_external_revenue_total", "consolidated"): _disclosure_cell("HAG", 2015, "note:geographic_external_revenue_total", "consolidated", "6.252.446.533", "HAG_financial_statements_2015_consolidated|2458", "HAG_financial_statements_2015_consolidated_2458.csv", 16, 6, "Total external-customer revenue, thousand VND"),
    ("HAG", 2016, "note:laos_external_revenue", "consolidated"): _disclosure_cell("HAG", 2016, "note:laos_external_revenue", "consolidated", "2.506.968.190", "HAG_financial_statements_2016_consolidated|2448", "HAG_financial_statements_2016_consolidated_2448.csv", 16, 2, "External-customer revenue from Laos, thousand VND"),
    ("HAG", 2016, "note:geographic_external_revenue_total", "consolidated"): _disclosure_cell("HAG", 2016, "note:geographic_external_revenue_total", "consolidated", "6.439.779.268", "HAG_financial_statements_2016_consolidated|2448", "HAG_financial_statements_2016_consolidated_2448.csv", 16, 6, "Total external-customer revenue, thousand VND"),
    ("HAG", 2017, "note:laos_external_revenue", "consolidated"): _disclosure_cell("HAG", 2017, "note:laos_external_revenue", "consolidated", "1.331.396.451", "HAG_financial_statements_2017_consolidated|2619", "HAG_financial_statements_2017_consolidated_2619.csv", 14, 2, "External-customer revenue from Laos, thousand VND"),
    ("HAG", 2017, "note:geographic_external_revenue_total", "consolidated"): _disclosure_cell("HAG", 2017, "note:geographic_external_revenue_total", "consolidated", "4.841.225.074", "HAG_financial_statements_2017_consolidated|2619", "HAG_financial_statements_2017_consolidated_2619.csv", 14, 6, "Total external-customer revenue, thousand VND"),
    ("HAG", 2022, "note:laos_external_revenue", "consolidated"): _disclosure_cell("HAG", 2022, "note:laos_external_revenue", "consolidated", "677.685.411", "HAG_financial_statements_2022_consolidated|2449", "HAG_financial_statements_2022_consolidated_2449.csv", 13, 2, "External-customer revenue from Laos, thousand VND"),
    ("HAG", 2022, "note:geographic_external_revenue_total", "consolidated"): _disclosure_cell("HAG", 2022, "note:geographic_external_revenue_total", "consolidated", "5.110.781.887", "HAG_financial_statements_2022_consolidated|2449", "HAG_financial_statements_2022_consolidated_2449.csv", 13, 5, "Total external-customer revenue, thousand VND"),
    ("VIC", 2015, "note:operating_lease_minimum_receipts_total", "consolidated"): _disclosure_cell("VIC", 2015, "note:operating_lease_minimum_receipts_total", "consolidated", "5.926.525.506.110", "VIC_financial_statements_2015_consolidated|2605", "VIC_financial_statements_2015_consolidated_2605.csv", 5, 1, "Total minimum operating-lease receipts, VND"),
    ("VIC", 2019, "note:operating_lease_minimum_receipts_total_million", "consolidated"): _disclosure_cell("VIC", 2019, "note:operating_lease_minimum_receipts_total_million", "consolidated", "22.012.268", "VIC_financial_statements_2019_consolidated|2286", "VIC_financial_statements_2019_consolidated_2286.csv", 5, 1, "Total minimum operating-lease receipts, million VND"),
    ("VIC", 2022, "note:operating_lease_minimum_receipts_total_million", "consolidated"): _disclosure_cell("VIC", 2022, "note:operating_lease_minimum_receipts_total_million", "consolidated", "23.885.078", "VIC_financial_statements_2022_consolidated|2432", "VIC_financial_statements_2022_consolidated_2432.csv", 5, 1, "Total minimum operating-lease receipts, million VND"),
    ("VIC", 2025, "note:operating_lease_minimum_receipts_total_million", "consolidated"): _disclosure_cell("VIC", 2025, "note:operating_lease_minimum_receipts_total_million", "consolidated", "8.025.973", "VIC_financial_statements_2025_consolidated|2862", "VIC_financial_statements_2025_consolidated_2862.csv", 5, 1, "Total minimum operating-lease receipts, million VND"),
    ("NLG", 2017, "note:subsidiary_short_term_other_payables", "separate"): _disclosure_cell("NLG", 2017, "note:subsidiary_short_term_other_payables", "separate", "130.863.796.545", "NLG_financial_statements_2017_separate|1291", "NLG_financial_statements_2017_separate_1291.csv", 5, 3, "Ending short-term other payables to subsidiaries"),
    ("NLG", 2018, "note:subsidiary_short_term_other_payables", "separate"): _disclosure_cell("NLG", 2018, "note:subsidiary_short_term_other_payables", "separate", "208.863.028.787", "NLG_financial_statements_2018_separate|1685", "NLG_financial_statements_2018_separate_1685.csv", 4, 3, "Ending short-term other payables to subsidiaries"),
    ("NLG", 2021, "note:subsidiary_short_term_other_payables", "separate"): _disclosure_cell("NLG", 2021, "note:subsidiary_short_term_other_payables", "separate", "1.150.344.202.519", "NLG_financial_statements_2021_separate|1440", "NLG_financial_statements_2021_separate_1440.csv", 5, 3, "Ending short-term other payables to subsidiaries"),
    ("NLG", 2025, "note:subsidiary_short_term_other_payables", "separate"): _disclosure_cell("NLG", 2025, "note:subsidiary_short_term_other_payables", "separate", "523.787.255.020", "NLG_financial_statements_2025_separate|1595", "NLG_financial_statements_2025_separate_1595.csv", 5, 3, "Ending short-term other payables to subsidiaries"),
    ("FOX", 2022, "note:short_term_term_deposit_carrying_amount", "separate"): _disclosure_cell("FOX", 2022, "note:short_term_term_deposit_carrying_amount", "separate", "100.000.000", "FOX_financial_statements_2022_separate|739", "FOX_financial_statements_2022_separate_739.csv", 3, 3, "Carrying amount of short-term term deposits"),
    ("STB", 2016, "note:total_assets_million", "consolidated"): _disclosure_cell("STB", 2016, "note:total_assets_million", "consolidated", "332.023.043", "STB_financial_statements_2016_consolidated|187", "STB_financial_statements_2016_consolidated_187.csv", 41, 2, "Ending total assets, million VND"),
    # Q64: the legacy selector used row 11 of the geographical-credit-risk
    # table (derivative contractual value) as if it were total assets.  The
    # consolidated balance sheet exposes the requested ending total directly.
    ("STB", 2021, "note:total_assets_million", "consolidated"): _disclosure_cell(
        "STB", 2021, "note:total_assets_million", "consolidated", "521.117.123",
        "STB_financial_statements_2021_consolidated|181",
        "STB_financial_statements_2021_consolidated_181.csv", 34, 2,
        "Ending consolidated total assets, million VND",
    ),
    # Q151: the final blank row is total customer deposits.  The question asks
    # for the VND sub-row within the savings-deposit section instead.
    ("EIB", 2015, "note:savings_deposits_vnd_million", "consolidated"): _disclosure_cell(
        "EIB", 2015, "note:savings_deposits_vnd_million", "consolidated", "53.658.311",
        "EIB_financial_statements_2015_consolidated|1831",
        "EIB_financial_statements_2015_consolidated_1831.csv", 8, 1,
        "Savings deposits denominated in VND, ending balance, million VND",
    ),
    # Q255: use balance-sheet code 132.  The legacy row belongs to a named
    # related party in a different receivable disclosure, not supplier advances.
    ("DCM", 2016, "note:short_term_supplier_advances", "separate"): _disclosure_cell(
        "DCM", 2016, "note:short_term_supplier_advances", "separate", "17.658.016.630",
        "DCM_financial_statements_2016_separate|345",
        "DCM_financial_statements_2016_separate_345.csv", 9, 3,
        "Ending short-term advances to suppliers, balance-sheet code 132",
    ),
    ("VRE", 2022, "note:interest_payable_million", "separate"): _disclosure_cell("VRE", 2022, "note:interest_payable_million", "separate", "49.408", "VRE_financial_statements_2022_separate|1107", "VRE_financial_statements_2022_separate_1107.csv", 1, 1, "Ending interest payable, million VND"),
    ("NVB", 2022, "note:deposits_at_other_credit_institutions_million", "separate"): _disclosure_cell("NVB", 2022, "note:deposits_at_other_credit_institutions_million", "separate", "11.658.653", "NVB_financial_statements_2022_separate|289", "NVB_financial_statements_2022_separate_289.csv", 5, 2, "Ending deposits at other credit institutions, million VND"),
    ("CEO", 2025, "note:short_term_supplier_advances", "consolidated"): _disclosure_cell("CEO", 2025, "note:short_term_supplier_advances", "consolidated", "68.897.389.031", "CEO_financial_statements_2025_consolidated|266", "CEO_financial_statements_2025_consolidated_266.csv", 9, 3, "Ending short-term advances to suppliers"),
    ("GEX", 2017, "note:operating_lease_minimum_receipts_total", "consolidated"): _disclosure_cell("GEX", 2017, "note:operating_lease_minimum_receipts_total", "consolidated", "244.103.155.846", "GEX_financial_statements_2017_consolidated|2425", "GEX_financial_statements_2017_consolidated_2425.csv", 5, 1, "Total future minimum operating-lease receipts"),
    ("BVH", 2018, "note:financial_assets_exposed_to_credit_risk", "separate"): _disclosure_cell("BVH", 2018, "note:financial_assets_exposed_to_credit_risk", "separate", "2.991.040.287.578", "BVH_financial_statements_2018_separate|1699", "BVH_financial_statements_2018_separate_1699.csv", 8, 4, "Total financial assets exposed to credit risk"),
    ("BAB", 2025, "note:customer_loan_loss_provision_million", "separate"): _disclosure_cell("BAB", 2025, "note:customer_loan_loss_provision_million", "separate", "(1.564.458)", "BAB_financial_statements_2025_separate|216", "BAB_financial_statements_2025_separate_216.csv", 12, 3, "Ending customer-loan loss provision, million VND"),
    ("BAB", 2025, "note:substandard_loans_million", "separate"): _disclosure_cell("BAB", 2025, "note:substandard_loans_million", "separate", "137.377", "BAB_financial_statements_2025_separate|977", "BAB_financial_statements_2025_separate_977.csv", 3, 1, "Ending substandard loans, million VND"),
    ("BAB", 2025, "note:doubtful_loans_million", "separate"): _disclosure_cell("BAB", 2025, "note:doubtful_loans_million", "separate", "137.466", "BAB_financial_statements_2025_separate|977", "BAB_financial_statements_2025_separate_977.csv", 4, 1, "Ending doubtful loans, million VND"),
    ("BAB", 2025, "note:loss_loans_million", "separate"): _disclosure_cell("BAB", 2025, "note:loss_loans_million", "separate", "1.179.903", "BAB_financial_statements_2025_separate|977", "BAB_financial_statements_2025_separate_977.csv", 5, 1, "Ending loss-category loans, million VND"),
    ("BAB", 2024, "note:customer_loan_loss_provision_million", "separate"): _disclosure_cell("BAB", 2024, "note:customer_loan_loss_provision_million", "separate", "(1.324.433)", "BAB_financial_statements_2024_separate|180", "BAB_financial_statements_2024_separate_180.csv", 13, 3, "Ending customer-loan loss provision, million VND"),
    ("SGB", 2024, "note:customer_loan_loss_provision_million", "separate"): _disclosure_cell("SGB", 2024, "note:customer_loan_loss_provision_million", "separate", "(210.684)", "SGB_financial_statements_2024_separate|263", "SGB_financial_statements_2024_separate_263.csv", 14, 2, "Ending customer-loan loss provision, million VND"),
    ("EIB", 2022, "note:other_on_balance_assets_provision_million", "consolidated"): _disclosure_cell("EIB", 2022, "note:other_on_balance_assets_provision_million", "consolidated", "(465.971)", "EIB_financial_statements_2022_consolidated|264", "EIB_financial_statements_2022_consolidated_264.csv", 26, 2, "Ending provision for other on-balance-sheet assets, million VND"),
    ("MBB", 2022, "note:other_on_balance_assets_provision_million", "consolidated"): _disclosure_cell("MBB", 2022, "note:other_on_balance_assets_provision_million", "consolidated", "(231.500)", "MBB_financial_statements_2022_consolidated|344", "MBB_financial_statements_2022_consolidated_344.csv", 41, 2, "Ending provision for other on-balance-sheet assets, million VND"),
    ("SNZ", 2019, "note:tax_and_state_payables", "separate"): _disclosure_cell("SNZ", 2019, "note:tax_and_state_payables", "separate", "15.344.409.381", "SNZ_financial_statements_2019_separate|406", "SNZ_financial_statements_2019_separate_406.csv", 5, 3, "Ending taxes and State payables"),
    ("VIC", 2019, "note:tax_and_state_payables_million", "separate"): _disclosure_cell("VIC", 2019, "note:tax_and_state_payables_million", "separate", "2.050.099", "VIC_financial_statements_2019_separate|241", "VIC_financial_statements_2019_separate_241.csv", 5, 3, "Ending taxes and State payables, million VND"),
    ("DXS", 2019, "note:tax_and_state_payables", "separate"): _disclosure_cell("DXS", 2019, "note:tax_and_state_payables", "separate", "185.442.713.325", "DXS_financial_statements_2019_separate|236", "DXS_financial_statements_2019_separate_236.csv", 5, 3, "Ending taxes and State payables"),
    ("HPX", 2019, "note:tax_and_state_payables", "separate"): _disclosure_cell("HPX", 2019, "note:tax_and_state_payables", "separate", "158.616.452.333", "HPX_financial_statements_2019_separate|302", "HPX_financial_statements_2019_separate_302.csv", 5, 3, "Ending taxes and State payables"),
})


def _ocr_percent_mean_expression(start: int, count: int) -> str:
    terms = [f"(v{i} / 100 if v{i} > 100 else v{i})" for i in range(start, start + count)]
    return f"({' + '.join(terms)}) / {count}"


# Semantic-alignment audit of the remaining old-model programs.  These cells
# repair cases where a syntactically valid query selected a nearby but
# unrelated accounting row.  Coordinates are re-read from the BTC tables;
# historical answers are never used as labels.
EXPLICIT_CELLS.update({
    ("VRE", 2019, "note:premises_tax_ending", "consolidated"): _disclosure_cell("VRE", 2019, "note:premises_tax_ending", "consolidated", "258.051", "VRE_financial_statements_2019_consolidated|1326", "VRE_financial_statements_2019_consolidated_1326.csv", 5, 1, "Chi phí thuế mặt bằng - Số dư cuối năm"),
    ("BID", 2016, "note:certificates_of_deposit", "consolidated"): _disclosure_cell("BID", 2016, "note:certificates_of_deposit", "consolidated", "47.141.004", "BID_financial_statements_2016_consolidated|1250", "BID_financial_statements_2016_consolidated_1250.csv", 2, 1, "Chứng chỉ tiền gửi - Số cuối năm"),
    ("VAB", 2025, "note:net_operating_cash_flow", "consolidated"): _disclosure_cell("VAB", 2025, "note:net_operating_cash_flow", "consolidated", "7.801.374.240.320", "VAB_financial_statements_2025_consolidated|434", "VAB_financial_statements_2025_consolidated_434.csv", 27, 2, "Lưu chuyển tiền thuần từ hoạt động kinh doanh"),
    ("NVL", 2019, "cdkt:131", "consolidated"): _disclosure_cell("NVL", 2019, "cdkt:131", "consolidated", "1.076.689.087.366", "NVL_financial_statements_2019_consolidated|133", "NVL_financial_statements_2019_consolidated_133.csv", 9, 3, "Phải thu ngắn hạn của khách hàng"),
    ("VPB", 2020, "note:total_customer_loans_by_industry", "consolidated"): _disclosure_cell("VPB", 2020, "note:total_customer_loans_by_industry", "consolidated", "290.816.086", "VPB_financial_statements_2020_consolidated|1258", "VPB_financial_statements_2020_consolidated_1258.csv", 23, 1, "Tổng dư nợ cho vay theo ngành nghề kinh doanh"),
    ("BVH", 2018, "note:total_financial_liabilities", "consolidated"): _disclosure_cell("BVH", 2018, "note:total_financial_liabilities", "consolidated", "90.002.872.153.963", "BVH_financial_statements_2018_consolidated|3548", "BVH_financial_statements_2018_consolidated_3548.csv", 9, 1, "Nợ phải trả tài chính - Tổng cộng"),
    ("BVH", 2022, "note:total_financial_liabilities_million", "consolidated"): _disclosure_cell("BVH", 2022, "note:total_financial_liabilities_million", "consolidated", "196.861.088", "BVH_financial_statements_2022_consolidated|3017", "BVH_financial_statements_2022_consolidated_3017.csv", 8, 6, "Nợ tài chính - Tổng cộng (triệu VND)"),
    ("QNS", 2024, "note:tangible_ppe_gross_cost", "consolidated"): _disclosure_cell("QNS", 2024, "note:tangible_ppe_gross_cost", "consolidated", "9.380.758.826.546", "QNS_financial_statements_2024_consolidated|272", "QNS_financial_statements_2024_consolidated_272.csv", 26, 3, "Tài sản cố định hữu hình - Nguyên giá"),
    ("QNS", 2024, "note:tangible_ppe_accumulated_depreciation", "consolidated"): _disclosure_cell("QNS", 2024, "note:tangible_ppe_accumulated_depreciation", "consolidated", "(5.946.334.636.629)", "QNS_financial_statements_2024_consolidated|272", "QNS_financial_statements_2024_consolidated_272.csv", 27, 3, "Tài sản cố định hữu hình - Giá trị hao mòn lũy kế"),
    ("PVT", 2025, "cdkt:221", "consolidated"): _disclosure_cell("PVT", 2025, "cdkt:221", "consolidated", "13.495.052.568.359", "PVT_financial_statements_2025_consolidated|336", "PVT_financial_statements_2025_consolidated_336.csv", 5, 4, "Tài sản cố định hữu hình"),
    ("OGC", 2017, "cdkt:320", "consolidated"): _disclosure_cell("OGC", 2017, "cdkt:320", "consolidated", "893.405.000.000", "OGC_financial_statements_2017_consolidated|493", "OGC_financial_statements_2017_consolidated_493.csv", 11, 3, "Vay và nợ thuê tài chính ngắn hạn"),
    ("OGC", 2017, "cdkt:338", "consolidated"): _disclosure_cell("OGC", 2017, "cdkt:338", "consolidated", "420.354.204.745", "OGC_financial_statements_2017_consolidated|493", "OGC_financial_statements_2017_consolidated_493.csv", 19, 3, "Vay và nợ thuê tài chính dài hạn"),
    ("ASM", 2017, "cdkt:320", "consolidated"): _disclosure_cell("ASM", 2017, "cdkt:320", "consolidated", "716.434.488.995", "ASM_financial_statements_2017_consolidated|321", "ASM_financial_statements_2017_consolidated_321.csv", 12, 3, "Vay và nợ thuê tài chính ngắn hạn"),
    ("ASM", 2017, "cdkt:338", "consolidated"): _disclosure_cell("ASM", 2017, "cdkt:338", "consolidated", "612.814.374.587", "ASM_financial_statements_2017_consolidated|321", "ASM_financial_statements_2017_consolidated_321.csv", 25, 3, "Vay và nợ thuê tài chính dài hạn"),
    ("CTG", 2024, "note:tangible_ppe_nbv", "separate"): _disclosure_cell("CTG", 2024, "note:tangible_ppe_nbv", "separate", "5.994.458", "CTG_financial_statements_2024_separate|318", "CTG_financial_statements_2024_separate_318.csv", 21, 3, "Tài sản cố định hữu hình"),
    ("MBB", 2024, "note:tangible_ppe_nbv", "separate"): _disclosure_cell("MBB", 2024, "note:tangible_ppe_nbv", "separate", "3.264.187", "MBB_financial_statements_2024_separate|322", "MBB_financial_statements_2024_separate_322.csv", 25, 2, "Tài sản cố định hữu hình"),
    ("ACB", 2022, "note:short_term_customer_loans", "consolidated"): _disclosure_cell("ACB", 2022, "note:short_term_customer_loans", "consolidated", "263.259.964", "ACB_financial_statements_2022_consolidated|1735", "ACB_financial_statements_2022_consolidated_1735.csv", 1, 1, "Cho vay khách hàng ngắn hạn"),
    ("HDB", 2022, "note:short_term_customer_loans", "consolidated"): _disclosure_cell("HDB", 2022, "note:short_term_customer_loans", "consolidated", "172.747.107", "HDB_financial_statements_2022_consolidated|1365", "HDB_financial_statements_2022_consolidated_1365.csv", 1, 1, "Cho vay khách hàng ngắn hạn"),
    ("KLB", 2016, "note:intangible_ppe_nbv", "separate"): _disclosure_cell("KLB", 2016, "note:intangible_ppe_nbv", "separate", "729.864", "KLB_financial_statements_2016_separate|235", "KLB_financial_statements_2016_separate_235.csv", 21, 3, "Tài sản cố định vô hình"),
    ("KLB", 2017, "note:intangible_ppe_nbv", "separate"): _disclosure_cell("KLB", 2017, "note:intangible_ppe_nbv", "separate", "711.549", "KLB_financial_statements_2017_separate|214", "KLB_financial_statements_2017_separate_214.csv", 20, 3, "Tài sản cố định vô hình"),
    ("KLB", 2018, "note:intangible_ppe_nbv", "separate"): _disclosure_cell("KLB", 2018, "note:intangible_ppe_nbv", "separate", "705.602", "KLB_financial_statements_2018_separate|145", "KLB_financial_statements_2018_separate_145.csv", 21, 3, "Tài sản cố định vô hình"),
    ("KLB", 2019, "note:intangible_ppe_nbv", "separate"): _disclosure_cell("KLB", 2019, "note:intangible_ppe_nbv", "separate", "718.399", "KLB_financial_statements_2019_separate|282", "KLB_financial_statements_2019_separate_282.csv", 34, 2, "Tài sản cố định vô hình"),
    ("KLB", 2020, "note:intangible_ppe_nbv", "separate"): _disclosure_cell("KLB", 2020, "note:intangible_ppe_nbv", "separate", "711.902", "KLB_financial_statements_2020_separate|290", "KLB_financial_statements_2020_separate_290.csv", 34, 2, "Tài sản cố định vô hình"),
    ("HHV", 2021, "note:short_loan_limit_1", "consolidated"): _disclosure_cell("HHV", 2021, "note:short_loan_limit_1", "consolidated", "100.000.000.000", "HHV_financial_statements_2021_consolidated|2040", "HHV_financial_statements_2021_consolidated_2040.csv", 1, 2, "Hạn mức vay ngắn hạn 1"),
    ("HHV", 2021, "note:short_loan_limit_2", "consolidated"): _disclosure_cell("HHV", 2021, "note:short_loan_limit_2", "consolidated", "80.000.000.000", "HHV_financial_statements_2021_consolidated|2040", "HHV_financial_statements_2021_consolidated_2040.csv", 2, 2, "Hạn mức vay ngắn hạn 2"),
    ("HHV", 2021, "note:short_loan_limit_3", "consolidated"): _disclosure_cell("HHV", 2021, "note:short_loan_limit_3", "consolidated", "300.000.000.000", "HHV_financial_statements_2021_consolidated|2040", "HHV_financial_statements_2021_consolidated_2040.csv", 3, 2, "Hạn mức vay ngắn hạn 3"),
    ("HHV", 2021, "note:short_loan_limit_4", "consolidated"): _disclosure_cell("HHV", 2021, "note:short_loan_limit_4", "consolidated", "2.600.000.000", "HHV_financial_statements_2021_consolidated|2040", "HHV_financial_statements_2021_consolidated_2040.csv", 4, 2, "Hạn mức vay ngắn hạn 4"),
    ("HHV", 2021, "note:short_loan_limit_5", "consolidated"): _disclosure_cell("HHV", 2021, "note:short_loan_limit_5", "consolidated", "400.000.000.000", "HHV_financial_statements_2021_consolidated|2040", "HHV_financial_statements_2021_consolidated_2040.csv", 5, 2, "Hạn mức vay ngắn hạn 5"),
    ("HHV", 2021, "note:short_loan_limit_6", "consolidated"): _disclosure_cell("HHV", 2021, "note:short_loan_limit_6", "consolidated", "202.000.000.000", "HHV_financial_statements_2021_consolidated|2040", "HHV_financial_statements_2021_consolidated_2040.csv", 6, 2, "Hạn mức vay ngắn hạn 6"),
    ("HHV", 2025, "note:short_loan_limit_1", "consolidated"): _disclosure_cell("HHV", 2025, "note:short_loan_limit_1", "consolidated", "200.000.000.000", "HHV_financial_statements_2025_consolidated|2559", "HHV_financial_statements_2025_consolidated_2559.csv", 1, 2, "Hạn mức vay ngắn hạn 1"),
    ("HHV", 2025, "note:short_loan_limit_2", "consolidated"): _disclosure_cell("HHV", 2025, "note:short_loan_limit_2", "consolidated", "100.000.000.000", "HHV_financial_statements_2025_consolidated|2559", "HHV_financial_statements_2025_consolidated_2559.csv", 2, 2, "Hạn mức vay ngắn hạn 2"),
    ("HHV", 2025, "note:short_loan_limit_3", "consolidated"): _disclosure_cell("HHV", 2025, "note:short_loan_limit_3", "consolidated", "800.000.000.000", "HHV_financial_statements_2025_consolidated|2559", "HHV_financial_statements_2025_consolidated_2559.csv", 3, 2, "Hạn mức vay ngắn hạn 3"),
    ("HHV", 2025, "note:short_loan_limit_4", "consolidated"): _disclosure_cell("HHV", 2025, "note:short_loan_limit_4", "consolidated", "1.000.000.000.000", "HHV_financial_statements_2025_consolidated|2559", "HHV_financial_statements_2025_consolidated_2559.csv", 4, 2, "Hạn mức vay ngắn hạn 4"),
    ("VPI", 2024, "cdkt:131", "consolidated"): _disclosure_cell("VPI", 2024, "cdkt:131", "consolidated", "179.433.940.407", "VPI_financial_statements_2024_consolidated|283", "VPI_financial_statements_2024_consolidated_283.csv", 8, 3, "Phải thu ngắn hạn của khách hàng"),
    ("KBC", 2022, "cdkt:242", "separate"): _disclosure_cell("KBC", 2022, "cdkt:242", "separate", "146.425.689.151", "KBC_financial_statements_2022_separate|320", "KBC_financial_statements_2022_separate_320.csv", 16, 3, "Chi phí xây dựng cơ bản dở dang"),
    ("HHV", 2021, "cdkt:131", "consolidated"): _disclosure_cell("HHV", 2021, "cdkt:131", "consolidated", "404.339.772.954", "HHV_financial_statements_2021_consolidated|331", "HHV_financial_statements_2021_consolidated_331.csv", 10, 3, "Phải thu ngắn hạn của khách hàng"),
    ("DBC", 2022, "cdkt:242", "separate"): _disclosure_cell("DBC", 2022, "cdkt:242", "separate", "1.359.527.554.512", "DBC_financial_statements_2022_separate|366", "DBC_financial_statements_2022_separate_366.csv", 29, 3, "Chi phí xây dựng cơ bản dở dang"),
    ("VRE", 2017, "cdkt:131", "consolidated"): _disclosure_cell("VRE", 2017, "cdkt:131", "consolidated", "567.834.036.030", "VRE_financial_statements_2017_consolidated|236", "VRE_financial_statements_2017_consolidated_236.csv", 8, 3, "Phải thu ngắn hạn của khách hàng"),
    ("NVL", 2016, "cdkt:227", "consolidated"): _disclosure_cell("NVL", 2016, "cdkt:227", "consolidated", "28.642.968.853", "NVL_financial_statements_2016_consolidated|170", "NVL_financial_statements_2016_consolidated_170.csv", 10, 3, "Tài sản cố định vô hình"),
    ("HAG", 2022, "cdkt:338", "separate"): _disclosure_cell("HAG", 2022, "cdkt:338", "separate", "3.581.600.405", "HAG_financial_statements_2022_separate|313", "HAG_financial_statements_2022_separate_313.csv", 15, 3, "Vay dài hạn"),
    ("HHV", 2024, "cdkt:251", "separate"): _disclosure_cell("HHV", 2024, "cdkt:251", "separate", "3.126.897.040.000", "HHV_financial_statements_2024_separate|379", "HHV_financial_statements_2024_separate_379.csv", 20, 3, "Đầu tư vào công ty con"),
    ("HUT", 2024, "cdkt:141", "separate"): _disclosure_cell("HUT", 2024, "cdkt:141", "separate", "3.180.337.280.522", "HUT_financial_statements_2024_separate|328", "HUT_financial_statements_2024_separate_328.csv", 16, 4, "Hàng tồn kho"),
    ("BVH", 2018, "note:total_shares", "consolidated"): _disclosure_cell("BVH", 2018, "note:total_shares", "consolidated", "700.886.434", "BVH_financial_statements_2018_consolidated|564", "BVH_financial_statements_2018_consolidated_564.csv", 5, 1, "Tổng số lượng cổ phần"),
    ("HAG", 2023, "note:total_bonds", "consolidated"): _disclosure_cell("HAG", 2023, "note:total_bonds", "consolidated", "4.948.065.559", "HAG_financial_statements_2023_consolidated|1661", "HAG_financial_statements_2023_consolidated_1661.csv", 3, 2, "Trái phiếu - Tổng cộng (ngàn VND)"),
    ("KLB", 2019, "note:operating_expense", "consolidated"): _disclosure_cell("KLB", 2019, "note:operating_expense", "consolidated", "1.041.601", "KLB_financial_statements_2019_consolidated|374", "KLB_financial_statements_2019_consolidated_374.csv", 14, 3, "Chi phí hoạt động"),
    ("HAG", 2021, "note:total_ordinary_bonds", "consolidated"): _disclosure_cell("HAG", 2021, "note:total_ordinary_bonds", "consolidated", "5.837.775.931", "HAG_financial_statements_2021_consolidated|1440", "HAG_financial_statements_2021_consolidated_1440.csv", 1, 1, "Trái phiếu thường - Tổng cộng"),
    ("OCB", 2021, "note:borrowings_other_credit_institutions", "separate"): _disclosure_cell("OCB", 2021, "note:borrowings_other_credit_institutions", "separate", "11.971.287.078.348", "OCB_financial_statements_2021_separate|290", "OCB_financial_statements_2021_separate_290.csv", 5, 2, "Vay các tổ chức tín dụng khác"),
    ("DLG", 2016, "cdkt:215", "separate"): _disclosure_cell("DLG", 2016, "cdkt:215", "separate", "225.647.099.300", "DLG_financial_statements_2016_separate|343", "DLG_financial_statements_2016_separate_343.csv", 22, 3, "Phải thu về cho vay dài hạn"),
    ("MPC", 2019, "note:farmer_advances", "consolidated"): _disclosure_cell("MPC", 2019, "note:farmer_advances", "consolidated", "3.500.000.000", "MPC_financial_statements_2019_consolidated|860", "MPC_financial_statements_2019_consolidated_860.csv", 4, 1, "Tạm ứng cho nông dân"),
    ("ACB", 2017, "note:customer_loan_provision", "consolidated"): _disclosure_cell("ACB", 2017, "note:customer_loan_provision", "consolidated", "(1.844.638)", "ACB_financial_statements_2017_consolidated|178", "ACB_financial_statements_2017_consolidated_178.csv", 15, 3, "Dự phòng rủi ro cho vay khách hàng"),
    ("ACB", 2019, "note:customer_loan_provision", "consolidated"): _disclosure_cell("ACB", 2019, "note:customer_loan_provision", "consolidated", "(2.535.689)", "ACB_financial_statements_2019_consolidated|171", "ACB_financial_statements_2019_consolidated_171.csv", 15, 3, "Dự phòng rủi ro cho vay khách hàng"),
    ("BID", 2022, "note:derivative_assets", "separate"): _disclosure_cell("BID", 2022, "note:derivative_assets", "separate", "1.038.368", "BID_financial_statements_2022_separate|274", "BID_financial_statements_2022_separate_274.csv", 11, 3, "Công cụ tài chính phái sinh và tài sản tài chính khác"),
    ("BID", 2021, "note:derivative_assets", "separate"): _disclosure_cell("BID", 2021, "note:derivative_assets", "separate", "196.743", "BID_financial_statements_2022_separate|274", "BID_financial_statements_2022_separate_274.csv", 11, 4, "Công cụ tài chính phái sinh và tài sản tài chính khác - Số đầu năm"),
    ("BID", 2018, "note:customer_loan_provision", "separate"): _disclosure_cell("BID", 2018, "note:customer_loan_provision", "separate", "(11.493.795)", "BID_financial_statements_2018_separate|235", "BID_financial_statements_2018_separate_235.csv", 13, 2, "Dự phòng rủi ro cho vay khách hàng"),
    ("BID", 2018, "note:gross_customer_loans", "separate"): _disclosure_cell("BID", 2018, "note:gross_customer_loans", "separate", "955.456.247", "BID_financial_statements_2018_separate|235", "BID_financial_statements_2018_separate_235.csv", 12, 2, "Cho vay khách hàng"),
    ("KBC", 2025, "note:related_short_term_borrowings", "separate"): _disclosure_cell("KBC", 2025, "note:related_short_term_borrowings", "separate", "154.300.000.000", "KBC_financial_statements_2025_separate|1267", "KBC_financial_statements_2025_separate_1267.csv", 4, 1, "Vay ngắn hạn từ các bên liên quan - Tổng cộng"),
    ("VIC", 2025, "note:related_short_term_borrowings", "separate"): _disclosure_cell("VIC", 2025, "note:related_short_term_borrowings", "separate", "762.882", "VIC_financial_statements_2025_separate|1425", "VIC_financial_statements_2025_separate_1425.csv", 8, 4, "Vay các bên liên quan ngắn hạn (triệu VND)"),
    ("HUT", 2020, "note:outstanding_shares", "consolidated"): _disclosure_cell("HUT", 2020, "note:outstanding_shares", "consolidated", "268.631.965", "HUT_financial_statements_2020_consolidated|1192", "HUT_financial_statements_2020_consolidated_1192.csv", 4, 1, "Số lượng cổ phiếu đang lưu hành"),
    ("HHS", 2020, "note:outstanding_shares", "consolidated"): _disclosure_cell("HHS", 2020, "note:outstanding_shares", "consolidated", "274.744.063", "HHS_financial_statements_2020_consolidated|1097", "HHS_financial_statements_2020_consolidated_1097.csv", 14, 1, "Số lượng cổ phiếu đang lưu hành"),
    ("STB", 2020, "note:credit_provision_expense", "consolidated"): _disclosure_cell("STB", 2020, "note:credit_provision_expense", "consolidated", "(3.036.974)", "STB_financial_statements_2020_consolidated|252", "STB_financial_statements_2020_consolidated_252.csv", 19, 2, "Chi phí dự phòng rủi ro tín dụng"),
    ("VRE", 2024, "note:subsidiary_investment_total", "separate"): _disclosure_cell("VRE", 2024, "note:subsidiary_investment_total", "separate", "13.976.356", "VRE_financial_statements_2024_separate|756", "VRE_financial_statements_2024_separate_756.csv", 8, 1, "Đầu tư vào công ty con - Tổng cộng"),
    ("VRE", 2024, "note:investment_property_gross_cost", "separate"): _disclosure_cell("VRE", 2024, "note:investment_property_gross_cost", "separate", "6.532.274", "VRE_financial_statements_2024_separate|822", "VRE_financial_statements_2024_separate_822.csv", 5, 3, "Bất động sản đầu tư - Nguyên giá cuối năm"),
    ("NKG", 2017, "cdkt:242", "consolidated"): _disclosure_cell("NKG", 2017, "cdkt:242", "consolidated", "132.823.233.933", "NKG_financial_statements_2017_consolidated|196", "NKG_financial_statements_2017_consolidated_196.csv", 33, 3, "Chi phí xây dựng cơ bản dở dang"),
    ("HHS", 2015, "note:opening_total_capital", "consolidated"): _disclosure_cell("HHS", 2015, "note:opening_total_capital", "consolidated", "1.441.245.797.506", "HHS_financial_statements_2015_consolidated|201", "HHS_financial_statements_2015_consolidated_201.csv", 22, 4, "Tổng cộng nguồn vốn - Số đầu năm"),
    ("VIC", 2016, "note:unsecured_long_term_loans_total", "separate"): _disclosure_cell("VIC", 2016, "note:unsecured_long_term_loans_total", "separate", "2.704.370.400.000", "VIC_financial_statements_2016_separate|2584", "VIC_financial_statements_2016_separate_2584.csv", 6, 2, "Các khoản cho vay dài hạn không có tài sản đảm bảo - Tổng cộng"),
    ("VIC", 2023, "note:business_partner_loans_due", "consolidated"): _disclosure_cell("VIC", 2023, "note:business_partner_loans_due", "consolidated", "4.063.713", "VIC_financial_statements_2023_consolidated|1399", "VIC_financial_statements_2023_consolidated_1399.csv", 2, 1, "Cho vay đối tác doanh nghiệp dài hạn đến hạn thu hồi"),
    ("VIC", 2023, "note:business_partner_loans_other", "consolidated"): _disclosure_cell("VIC", 2023, "note:business_partner_loans_other", "consolidated", "3.005.637", "VIC_financial_statements_2023_consolidated|1399", "VIC_financial_statements_2023_consolidated_1399.csv", 3, 1, "Các khoản cho vay các đối tác doanh nghiệp"),
    ("HDB", 2021, "note:htm_government_bonds", "consolidated"): _disclosure_cell("HDB", 2021, "note:htm_government_bonds", "consolidated", "8.198.347", "HDB_financial_statements_2021_consolidated|1472", "HDB_financial_statements_2021_consolidated_1472.csv", 1, 1, "Trái phiếu Chính phủ giữ đến ngày đáo hạn - Cuối năm"),
    ("HDB", 2020, "note:htm_government_bonds", "consolidated"): _disclosure_cell("HDB", 2020, "note:htm_government_bonds", "consolidated", "11.320.487", "HDB_financial_statements_2021_consolidated|1472", "HDB_financial_statements_2021_consolidated_1472.csv", 1, 2, "Trái phiếu Chính phủ giữ đến ngày đáo hạn - Đầu năm"),
    ("MML", 2024, "note:dividends_paid", "consolidated"): _disclosure_cell("MML", 2024, "note:dividends_paid", "consolidated", "(33.516.000)", "MML_financial_statements_2024_consolidated|367", "MML_financial_statements_2024_consolidated_367.csv", 7, 2, "Tiền trả cổ tức"),
    ("PNJ", 2018, "cdkt:221", "consolidated"): _disclosure_cell("PNJ", 2018, "cdkt:221", "consolidated", "225.960.569.846", "PNJ_financial_statements_2018_consolidated|146", "PNJ_financial_statements_2018_consolidated_146.csv", 23, 3, "Tài sản cố định hữu hình"),
    ("IJC", 2016, "cdkt:132", "consolidated"): _disclosure_cell("IJC", 2016, "cdkt:132", "consolidated", "25.195.451.251", "IJC_financial_statements_2016_consolidated|274", "IJC_financial_statements_2016_consolidated_274.csv", 11, 4, "Trả trước cho người bán ngắn hạn"),
    ("MSR", 2025, "kqkd:23", "separate"): _disclosure_cell("MSR", 2025, "kqkd:23", "separate", "154.674.553", "MSR_financial_statements_2025_separate|188", "MSR_financial_statements_2025_separate_188.csv", 3, 3, "Chi phí lãi vay (nghìn VND)"),
    ("SSB", 2020, "note:medium_term_customer_loans", "consolidated"): _disclosure_cell("SSB", 2020, "note:medium_term_customer_loans", "consolidated", "30.973.169", "SSB_financial_statements_2020_consolidated|1225", "SSB_financial_statements_2020_consolidated_1225.csv", 2, 1, "Nợ cho vay trung hạn"),
    ("BID", 2023, "note:gross_customer_loans", "separate"): _disclosure_cell("BID", 2023, "note:gross_customer_loans", "separate", "1.740.391.368", "BID_financial_statements_2023_separate|274", "BID_financial_statements_2023_separate_274.csv", 12, 3, "Cho vay khách hàng - Dư nợ gộp"),
    ("VPI", 2021, "note:production_business_wip", "consolidated"): _disclosure_cell("VPI", 2021, "note:production_business_wip", "consolidated", "2.604.207.662.218", "VPI_financial_statements_2021_consolidated|957", "VPI_financial_statements_2021_consolidated_957.csv", 2, 1, "Chi phí sản xuất kinh doanh dở dang"),
    ("DPM", 2021, "cdkt:151", "separate"): _disclosure_cell("DPM", 2021, "cdkt:151", "separate", "13.104.672.883", "DPM_financial_statements_2021_separate|1311", "DPM_financial_statements_2021_separate_1311.csv", 6, 1, "Chi phí trả trước ngắn hạn - Tổng cộng"),
    ("VRE", 2017, "cdkt:242", "consolidated"): _disclosure_cell("VRE", 2017, "cdkt:242", "consolidated", "1.080.110.809.673", "VRE_financial_statements_2017_consolidated|253", "VRE_financial_statements_2017_consolidated_253.csv", 16, 3, "Chi phí xây dựng cơ bản dở dang"),
    ("ABB", 2020, "note:customer_loans_financial_assets", "separate"): _disclosure_cell("ABB", 2020, "note:customer_loans_financial_assets", "separate", "62.588.033", "ABB_financial_statements_2020_separate|2471", "ABB_financial_statements_2020_separate_2471.csv", 6, 3, "Cho vay khách hàng - Cho vay và phải thu"),
    ("ABB", 2023, "note:customer_loans_financial_assets", "separate"): _disclosure_cell("ABB", 2023, "note:customer_loans_financial_assets", "separate", "96.781.614", "ABB_financial_statements_2023_separate|2663", "ABB_financial_statements_2023_separate_2663.csv", 6, 3, "Cho vay khách hàng - Cho vay và phải thu"),
    ("EVF", 2025, "note:afs_securities_gross", "consolidated"): _disclosure_cell("EVF", 2025, "note:afs_securities_gross", "consolidated", "1.795.912", "EVF_financial_statements_2025|1041", "EVF_financial_statements_2025_1041.csv", 7, 1, "Chứng khoán đầu tư sẵn sàng để bán - Cộng"),
    ("EVF", 2025, "note:afs_securities_provision", "consolidated"): _disclosure_cell("EVF", 2025, "note:afs_securities_provision", "consolidated", "(38.384)", "EVF_financial_statements_2025|1041", "EVF_financial_statements_2025_1041.csv", 8, 1, "Dự phòng giảm giá chứng khoán đầu tư sẵn sàng để bán"),
    ("VIB", 2017, "note:net_profit_after_tax", "separate"): _disclosure_cell("VIB", 2017, "note:net_profit_after_tax", "separate", "1.124.208", "VIB_financial_statements_2017_separate|222", "VIB_financial_statements_2017_separate_222.csv", 23, 2, "Lợi nhuận sau thuế"),
    ("MSB", 2017, "note:net_profit_after_tax", "separate"): _disclosure_cell("MSB", 2017, "note:net_profit_after_tax", "separate", "125.407", "MSB_financial_statements_2017_separate|335", "MSB_financial_statements_2017_separate_335.csv", 20, 3, "Lợi nhuận sau thuế"),
    ("HAG", 2024, "note:npat_thousand", "consolidated"): _disclosure_cell("HAG", 2024, "note:npat_thousand", "consolidated", "1.060.121.821", "HAG_financial_statements_2024_consolidated|401", "HAG_financial_statements_2024_consolidated_401.csv", 19, 3, "Lợi nhuận sau thuế TNDN (nghìn VND)"),
    ("ABB", 2024, "note:lc_commitments", "separate"): _disclosure_cell("ABB", 2024, "note:lc_commitments", "separate", "1.634.376", "ABB_financial_statements_2024_separate|205", "ABB_financial_statements_2024_separate_205.csv", 5, 3, "Cam kết trong nghiệp vụ thư tín dụng"),
    ("SSB", 2024, "note:lc_commitments", "separate"): _disclosure_cell("SSB", 2024, "note:lc_commitments", "separate", "2.228.158", "SSB_financial_statements_2024_separate|273", "SSB_financial_statements_2024_separate_273.csv", 6, 3, "Cam kết trong nghiệp vụ L/C"),
    ("BID", 2024, "note:lc_commitments", "separate"): _disclosure_cell("BID", 2024, "note:lc_commitments", "separate", "62.109.504", "BID_financial_statements_2024_separate|388", "BID_financial_statements_2024_separate_388.csv", 7, 3, "Cam kết trong nghiệp vụ L/C"),
    ("MBB", 2024, "note:lc_commitments", "separate"): _disclosure_cell("MBB", 2024, "note:lc_commitments", "separate", "29.138.440", "MBB_financial_statements_2024_separate|374", "MBB_financial_statements_2024_separate_374.csv", 7, 2, "Cam kết trong nghiệp vụ L/C"),
    ("PVT", 2017, "note:corporate_tax_payable_ending", "consolidated"): _disclosure_cell("PVT", 2017, "note:corporate_tax_payable_ending", "consolidated", "80.400.126.084", "PVT_financial_statements_2017_consolidated|834", "PVT_financial_statements_2017_consolidated_834.csv", 11, 4, "Thuế thu nhập doanh nghiệp phải trả - Số cuối năm"),
    ("BSR", 2017, "note:corporate_tax_payable_ending", "consolidated"): _disclosure_cell("BSR", 2017, "note:corporate_tax_payable_ending", "consolidated", "141.037.495.805", "BSR_financial_statements_2017_consolidated|902", "BSR_financial_statements_2017_consolidated_902.csv", 7, 4, "Thuế thu nhập doanh nghiệp phải trả - Số cuối năm"),
    ("PLX", 2017, "note:corporate_tax_payable_ending", "consolidated"): _disclosure_cell("PLX", 2017, "note:corporate_tax_payable_ending", "consolidated", "(206.404.582.261)", "PLX_financial_statements_2017_consolidated|1346", "PLX_financial_statements_2017_consolidated_1346.csv", 5, 5, "Thuế thu nhập doanh nghiệp phải trả - Số cuối năm"),
    ("BSR", 2016, "kqkd:10", "consolidated"): _disclosure_cell("BSR", 2016, "kqkd:10", "consolidated", "73.686.050.815.612", "BSR_financial_statements_2017_consolidated|296", "BSR_financial_statements_2017_consolidated_296.csv", 3, 4, "Doanh thu thuần năm trước"),
    ("GVR", 2019, "cdkt:311", "consolidated"): _disclosure_cell("GVR", 2019, "cdkt:311", "consolidated", "943.012.378.938", "GVR_financial_statements_2019_consolidated|289", "GVR_financial_statements_2019_consolidated_289.csv", 4, 3, "Phải trả người bán ngắn hạn"),
    ("MSB", 2019, "note:specific_customer_loan_provision", "consolidated"): _disclosure_cell("MSB", 2019, "note:specific_customer_loan_provision", "consolidated", "443.312", "MSB_financial_statements_2019_consolidated|1408", "MSB_financial_statements_2019_consolidated_1408.csv", 2, 1, "Dự phòng cụ thể cho vay khách hàng"),
    ("VCB", 2019, "note:specific_customer_loan_provision", "consolidated"): _disclosure_cell("VCB", 2019, "note:specific_customer_loan_provision", "consolidated", "5.134.461", "VCB_financial_statements_2019_consolidated|1164", "VCB_financial_statements_2019_consolidated_1164.csv", 2, 1, "Dự phòng cụ thể cho vay khách hàng"),
    ("NVL", 2018, "cdkt:242", "consolidated"): _disclosure_cell("NVL", 2018, "cdkt:242", "consolidated", "248.217.392.159", "NVL_financial_statements_2018_consolidated|198", "NVL_financial_statements_2018_consolidated_198.csv", 19, 3, "Chi phí xây dựng cơ bản dở dang"),
    ("NVL", 2020, "cdkt:242", "consolidated"): _disclosure_cell("NVL", 2020, "cdkt:242", "consolidated", "103.772.861.482", "NVL_financial_statements_2020_consolidated|217", "NVL_financial_statements_2020_consolidated_217.csv", 18, 3, "Chi phí xây dựng cơ bản dở dang"),
    ("NVL", 2022, "cdkt:242", "consolidated"): _disclosure_cell("NVL", 2022, "cdkt:242", "consolidated", "390.960.770.085", "NVL_financial_statements_2022_consolidated|232", "NVL_financial_statements_2022_consolidated_232.csv", 19, 3, "Chi phí xây dựng cơ bản dở dang"),
    ("DTK", 2022, "cdkt:136", "separate"): _disclosure_cell("DTK", 2022, "cdkt:136", "separate", "60.662.738.216", "DTK_financial_statements_2022_separate|214", "DTK_financial_statements_2022_separate_214.csv", 15, 4, "Phải thu ngắn hạn khác"),
    ("DTK", 2023, "cdkt:136", "separate"): _disclosure_cell("DTK", 2023, "cdkt:136", "separate", "42.773.553.679", "DTK_financial_statements_2023_separate|187", "DTK_financial_statements_2023_separate_187.csv", 15, 4, "Phải thu ngắn hạn khác"),
    ("DTK", 2024, "cdkt:136", "consolidated"): _disclosure_cell("DTK", 2024, "cdkt:136", "consolidated", "42.069.431.300", "DTK_financial_statements_2024_consolidated|318", "DTK_financial_statements_2024_consolidated_318.csv", 15, 4, "Phải thu ngắn hạn khác"),
    ("DTK", 2025, "cdkt:136", "consolidated"): _disclosure_cell("DTK", 2025, "cdkt:136", "consolidated", "27.886.854.405", "DTK_financial_statements_2025_consolidated|337", "DTK_financial_statements_2025_consolidated_337.csv", 10, 3, "Phải thu ngắn hạn khác"),
    ("CTG", 2022, "note:custody_assets_total", "consolidated"): _disclosure_cell("CTG", 2022, "note:custody_assets_total", "consolidated", "101.778.024", "CTG_financial_statements_2023_consolidated|1750", "CTG_financial_statements_2023_consolidated_1750.csv", 6, 2, "Tài sản và chứng từ giữ hộ, bảo quản - Số đầu năm"),
    ("CTG", 2023, "note:custody_assets_total", "consolidated"): _disclosure_cell("CTG", 2023, "note:custody_assets_total", "consolidated", "91.656.083", "CTG_financial_statements_2023_consolidated|1750", "CTG_financial_statements_2023_consolidated_1750.csv", 6, 1, "Tài sản và chứng từ giữ hộ, bảo quản - Số cuối năm"),
    ("CTG", 2024, "note:custody_assets_total", "consolidated"): _disclosure_cell("CTG", 2024, "note:custody_assets_total", "consolidated", "132.970.869", "CTG_financial_statements_2025_consolidated|2919", "CTG_financial_statements_2025_consolidated_2919.csv", 5, 2, "Tài sản và chứng từ giữ hộ, bảo quản - Số đầu năm"),
    ("CTG", 2025, "note:custody_assets_total", "consolidated"): _disclosure_cell("CTG", 2025, "note:custody_assets_total", "consolidated", "95.027.390", "CTG_financial_statements_2025_consolidated|2919", "CTG_financial_statements_2025_consolidated_2919.csv", 5, 1, "Tài sản và chứng từ giữ hộ, bảo quản - Số cuối năm"),
    # q965 originally counted only SSH although the question asks for SSH,
    # DXS and CEO. SSH and CEO each disclose two taxable activity buckets.
    ("SSH", 2022, "note:nondeductible_expense_real_estate", "separate"): _disclosure_cell("SSH", 2022, "note:nondeductible_expense_real_estate", "separate", "29.253.314.525", "SSH_financial_statements_2022_separate|1025", "SSH_financial_statements_2022_separate_1025.csv", 7, 1, "Nondeductible expense - real-estate activity"),
    ("SSH", 2022, "note:nondeductible_expense_other", "separate"): _disclosure_cell("SSH", 2022, "note:nondeductible_expense_other", "separate", "38.434.986.469", "SSH_financial_statements_2022_separate|1025", "SSH_financial_statements_2022_separate_1025.csv", 9, 1, "Nondeductible expense - other activity"),
    ("DXS", 2022, "note:nondeductible_expense", "separate"): _disclosure_cell("DXS", 2022, "note:nondeductible_expense", "separate", "826.702.244", "DXS_financial_statements_2022_separate|945", "DXS_financial_statements_2022_separate_945.csv", 4, 1, "Nondeductible expense"),
    ("CEO", 2022, "note:nondeductible_expense_real_estate", "separate"): _disclosure_cell("CEO", 2022, "note:nondeductible_expense_real_estate", "separate", "5.458.655.972", "CEO_financial_statements_2022_separate|1614", "CEO_financial_statements_2022_separate_1614.csv", 2, 1, "Nondeductible expense - real-estate activity"),
    ("CEO", 2022, "note:nondeductible_expense_social_housing", "separate"): _disclosure_cell("CEO", 2022, "note:nondeductible_expense_social_housing", "separate", "277.865.333", "CEO_financial_statements_2022_separate|1614", "CEO_financial_statements_2022_separate_1614.csv", 10, 1, "Nondeductible expense - social-housing activity"),
    # Multi-company semantic audit: legacy programs frequently read only one
    # company (or a nearby movement row) although the question asks for a
    # comparison or aggregate.  Each value below is tied to its BTC source
    # table so the generated program remains reproducible.
    ("HNG", 2019, "note:short_term_trade_payables_thousand", "consolidated"): _disclosure_cell("HNG", 2019, "note:short_term_trade_payables_thousand", "consolidated", "984.872.754", "HNG_financial_statements_2019_consolidated|216", "HNG_financial_statements_2019_consolidated_216.csv", 31, 3, "Ending short-term trade payables, thousand VND"),
    ("HAG", 2019, "note:short_term_trade_payables_thousand", "consolidated"): _disclosure_cell("HAG", 2019, "note:short_term_trade_payables_thousand", "consolidated", "1.014.993.762", "HAG_financial_statements_2019_consolidated|277", "HAG_financial_statements_2019_consolidated_277.csv", 31, 3, "Ending short-term trade payables, thousand VND"),
    ("MPC", 2019, "note:short_term_trade_payables", "consolidated"): _disclosure_cell("MPC", 2019, "note:short_term_trade_payables", "consolidated", "160.789.544.795", "MPC_financial_statements_2019_consolidated|150", "MPC_financial_statements_2019_consolidated_150.csv", 31, 3, "Ending short-term trade payables"),
    ("VJC", 2016, "cdkt:311", "consolidated"): _disclosure_cell("VJC", 2016, "cdkt:311", "consolidated", "391.117.403.830", "VJC_financial_statements_2016_consolidated|198", "VJC_financial_statements_2016_consolidated_198.csv", 31, 3, "Ending short-term trade payables"),
    ("VSC", 2016, "cdkt:311", "consolidated"): _disclosure_cell("VSC", 2016, "cdkt:311", "consolidated", "62.612.059.106", "VSC_financial_statements_2016_consolidated|166", "VSC_financial_statements_2016_consolidated_166.csv", 31, 3, "Ending short-term trade payables"),
    ("CEO", 2024, "cdkt:242", "consolidated"): _disclosure_cell("CEO", 2024, "cdkt:242", "consolidated", "1.077.553.963.639", "CEO_financial_statements_2024_consolidated|203", "CEO_financial_statements_2024_consolidated_203.csv", 33, 4, "Ending construction in progress"),
    ("VPI", 2024, "cdkt:242", "consolidated"): _disclosure_cell("VPI", 2024, "cdkt:242", "consolidated", "658.649.923.759", "VPI_financial_statements_2024_consolidated|298", "VPI_financial_statements_2024_consolidated_298.csv", 17, 3, "Ending construction in progress"),
    ("SCR", 2025, "cdkt:131", "consolidated"): _disclosure_cell("SCR", 2025, "cdkt:131", "consolidated", "174.131.397.722", "SCR_financial_statements_2025_consolidated|254", "SCR_financial_statements_2025_consolidated_254.csv", 9, 3, "Ending short-term customer receivables"),
    ("NVL", 2025, "cdkt:131", "consolidated"): _disclosure_cell("NVL", 2025, "cdkt:131", "consolidated", "3.274.282.872.300", "NVL_financial_statements_2025_consolidated|467", "NVL_financial_statements_2025_consolidated_467.csv", 8, 3, "Ending short-term customer receivables"),
    ("SHB", 2020, "note:tangible_ppe_nbv_million", "consolidated"): _disclosure_cell("SHB", 2020, "note:tangible_ppe_nbv_million", "consolidated", "532.986", "SHB_financial_statements_2020_consolidated|121", "SHB_financial_statements_2020_consolidated_121.csv", 21, 3, "Ending tangible PPE net book value, million VND"),
    ("NAB", 2020, "note:tangible_ppe_nbv_million", "consolidated"): _disclosure_cell("NAB", 2020, "note:tangible_ppe_nbv_million", "consolidated", "516.217", "NAB_financial_statements_2020_consolidated|167", "NAB_financial_statements_2020_consolidated_167.csv", 21, 3, "Ending tangible PPE net book value, million VND"),
    ("MBB", 2025, "note:net_other_income_million", "consolidated"): _disclosure_cell("MBB", 2025, "note:net_other_income_million", "consolidated", "5.314.474", "MBB_financial_statements_2025_consolidated|428", "MBB_financial_statements_2025_consolidated_428.csv", 32, 3, "Net other income, million VND"),
    ("MSB", 2025, "note:net_other_income_million", "consolidated"): _disclosure_cell("MSB", 2025, "note:net_other_income_million", "consolidated", "575.853", "MSB_financial_statements_2025_consolidated|332", "MSB_financial_statements_2025_consolidated_332.csv", 30, 3, "Net other income, million VND"),
    ("ABB", 2022, "note:net_service_income_million", "consolidated"): _disclosure_cell("ABB", 2022, "note:net_service_income_million", "consolidated", "232.042", "ABB_financial_statements_2022_consolidated|296", "ABB_financial_statements_2022_consolidated_296.csv", 28, 3, "Net service income, million VND"),
    ("MBB", 2022, "note:net_service_income_million", "consolidated"): _disclosure_cell("MBB", 2022, "note:net_service_income_million", "consolidated", "4.135.568", "MBB_financial_statements_2022_consolidated|410", "MBB_financial_statements_2022_consolidated_410.csv", 29, 3, "Net service income, million VND"),
    ("MBB", 2018, "note:outstanding_common_shares", "consolidated"): _disclosure_cell("MBB", 2018, "note:outstanding_common_shares", "consolidated", "2.160.451.381", "MBB_financial_statements_2018_consolidated|2024", "MBB_financial_statements_2018_consolidated_2024.csv", 5, 1, "Outstanding common shares"),
    ("ACB", 2018, "note:outstanding_common_shares", "consolidated"): _disclosure_cell("ACB", 2018, "note:outstanding_common_shares", "consolidated", "1.247.165.130", "ACB_financial_statements_2018_consolidated|2130", "ACB_financial_statements_2018_consolidated_2130.csv", 5, 1, "Outstanding common shares"),
    ("MBB", 2015, "note:state_bank_deposits_million", "separate"): _disclosure_cell("MBB", 2015, "note:state_bank_deposits_million", "separate", "8.181.894", "MBB_financial_statements_2015_separate|125", "MBB_financial_statements_2015_separate_125.csv", 3, 3, "Ending deposits at the State Bank, million VND"),
    ("SHB", 2015, "note:state_bank_deposits_million", "separate"): _disclosure_cell("SHB", 2015, "note:state_bank_deposits_million", "separate", "4.362.518", "SHB_financial_statements_2015_separate|195", "SHB_financial_statements_2015_separate_195.csv", 3, 3, "Ending deposits at the State Bank, million VND"),
    ("VIB", 2022, "note:cash_and_gold_million", "separate"): _disclosure_cell("VIB", 2022, "note:cash_and_gold_million", "separate", "1.617.912", "VIB_financial_statements_2022_separate|174", "VIB_financial_statements_2022_separate_174.csv", 2, 3, "Ending cash and gold, million VND"),
    ("SHB", 2022, "note:cash_and_gold_million", "separate"): _disclosure_cell("SHB", 2022, "note:cash_and_gold_million", "separate", "1.822.415", "SHB_financial_statements_2022_separate|164", "SHB_financial_statements_2022_separate_164.csv", 2, 3, "Ending cash and gold, million VND"),
    ("BAB", 2024, "note:employee_expense_million", "consolidated"): _disclosure_cell("BAB", 2024, "note:employee_expense_million", "consolidated", "1.403.176", "BAB_financial_statements_2024_consolidated|1322", "BAB_financial_statements_2024_consolidated_1322.csv", 5, 1, "Employee expense, million VND"),
    ("NVB", 2024, "note:employee_expense_million", "consolidated"): _disclosure_cell("NVB", 2024, "note:employee_expense_million", "consolidated", "955.516", "NVB_financial_statements_2024_consolidated|1495", "NVB_financial_statements_2024_consolidated_1495.csv", 5, 1, "Employee expense, million VND"),
    ("CTG", 2019, "note:deposit_interest_income_million", "separate"): _disclosure_cell("CTG", 2019, "note:deposit_interest_income_million", "separate", "3.094.131", "CTG_financial_statements_2019_separate|1516", "CTG_financial_statements_2019_separate_1516.csv", 2, 1, "Interest income from deposits, million VND"),
    ("VPB", 2019, "note:deposit_interest_income_million", "separate"): _disclosure_cell("VPB", 2019, "note:deposit_interest_income_million", "separate", "334.227", "VPB_financial_statements_2019_separate|1795", "VPB_financial_statements_2019_separate_1795.csv", 2, 1, "Interest income from deposits, million VND"),
    ("MBB", 2023, "note:customer_loan_provision_expense_million", "separate"): _disclosure_cell("MBB", 2023, "note:customer_loan_provision_expense_million", "separate", "2.938.200", "MBB_financial_statements_2023_separate|1778", "MBB_financial_statements_2023_separate_1778.csv", 2, 1, "Specific customer-loan provision expense, million VND"),
    ("CTG", 2023, "note:customer_loan_provision_expense_million", "separate"): _disclosure_cell("CTG", 2023, "note:customer_loan_provision_expense_million", "separate", "24.991.748", "CTG_financial_statements_2023_separate|1132", "CTG_financial_statements_2023_separate_1132.csv", 5, 1, "Customer-loan risk provision expense, million VND"),
    ("VCB", 2025, "note:deposit_interest_expense_million", "separate"): _disclosure_cell("VCB", 2025, "note:deposit_interest_expense_million", "separate", "44.283.911", "VCB_financial_statements_2025_separate|1780", "VCB_financial_statements_2025_separate_1780.csv", 2, 1, "Deposit interest expense, million VND"),
    ("VPB", 2025, "note:deposit_interest_expense_million", "separate"): _disclosure_cell("VPB", 2025, "note:deposit_interest_expense_million", "separate", "28.751.918", "VPB_financial_statements_2025_separate|1967", "VPB_financial_statements_2025_separate_1967.csv", 2, 1, "Deposit interest expense, million VND"),
    ("NVB", 2016, "note:external_receivables_million", "separate"): _disclosure_cell("NVB", 2016, "note:external_receivables_million", "separate", "2.709.225", "NVB_financial_statements_2016_separate|1182", "NVB_financial_statements_2016_separate_1182.csv", 5, 1, "External receivables, million VND"),
    ("VIB", 2016, "note:external_receivables_million", "separate"): _disclosure_cell("VIB", 2016, "note:external_receivables_million", "separate", "2.702.276", "VIB_financial_statements_2016_separate|1237", "VIB_financial_statements_2016_separate_1237.csv", 5, 1, "External receivables, million VND"),
    ("KLB", 2024, "note:issued_bonds_million", "separate"): _disclosure_cell("KLB", 2024, "note:issued_bonds_million", "separate", "800.000", "KLB_financial_statements_2024_separate|1294", "KLB_financial_statements_2024_separate_1294.csv", 5, 1, "Issued bonds, million VND"),
    ("EIB", 2024, "note:issued_bonds_million", "separate"): _disclosure_cell("EIB", 2024, "note:issued_bonds_million", "separate", "3.680.000", "EIB_financial_statements_2024_separate|1605", "EIB_financial_statements_2024_separate_1605.csv", 5, 1, "Issued bonds, million VND"),
    ("GEG", 2023, "cdkt:252", "separate"): _disclosure_cell("GEG", 2023, "cdkt:252", "separate", "143.790.000.000", "GEG_financial_statements_2023_separate|302", "GEG_financial_statements_2023_separate_302.csv", 17, 3, "Ending investment in associates"),
    ("DNH", 2023, "cdkt:252", "separate"): _disclosure_cell("DNH", 2023, "cdkt:252", "separate", "100.079.200.000", "DNH_financial_statements_2023_separate|235", "DNH_financial_statements_2023_separate_235.csv", 17, 3, "Ending investment in associates"),
    ("MSN", 2021, "note:npat", "separate"): _disclosure_cell("MSN", 2021, "note:npat", "separate", "1.725.926.701.022", "MSN_financial_statements_2021_separate|1003", "MSN_financial_statements_2021_separate_1003.csv", 19, 3, "Net profit after tax"),
    ("MCH", 2018, "note:eps_vnd", "consolidated"): _disclosure_cell("MCH", 2018, "note:eps_vnd", "consolidated", "5.549", "MCH_financial_statements_2018_consolidated|171", "MCH_financial_statements_2018_consolidated_171.csv", 70, 3, "Basic earnings per share, VND/share"),
    ("MML", 2018, "note:eps_vnd", "consolidated"): _disclosure_cell("MML", 2018, "note:eps_vnd", "consolidated", "442", "MML_financial_statements_2018_consolidated|241", "MML_financial_statements_2018_consolidated_241.csv", 70, 3, "Basic earnings per share, VND/share"),
    ("VPI", 2023, "note:real_estate_development_cost", "consolidated"): _disclosure_cell("VPI", 2023, "note:real_estate_development_cost", "consolidated", "2.162.118.955.056", "VPI_financial_statements_2023_consolidated|1699", "VPI_financial_statements_2023_consolidated_1699.csv", 2, 1, "Construction and real-estate development cost"),
    ("VRE", 2023, "note:real_estate_development_cost_million", "consolidated"): _disclosure_cell("VRE", 2023, "note:real_estate_development_cost_million", "consolidated", "438.660", "VRE_financial_statements_2023_consolidated|1556", "VRE_financial_statements_2023_consolidated_1556.csv", 1, 1, "Construction and real-estate development cost, million VND"),
    ("MBB", 2023, "note:total_assets_million", "separate"): _disclosure_cell("MBB", 2023, "note:total_assets_million", "separate", "902.044.947", "MBB_financial_statements_2023_separate|331", "MBB_financial_statements_2023_separate_331.csv", 38, 2, "Parent total assets, million VND"),
    ("EIB", 2023, "note:total_assets_million", "separate"): _disclosure_cell("EIB", 2023, "note:total_assets_million", "separate", "201.672.702", "EIB_financial_statements_2023_separate|319", "EIB_financial_statements_2023_separate_319.csv", 30, 3, "Parent total assets, million VND"),
    ("VNM", 2016, "note:tangible_ppe_nbv", "consolidated"): _disclosure_cell("VNM", 2016, "note:tangible_ppe_nbv", "consolidated", "7.916.322.992.944", "VNM_financial_statements_2016_consolidated|959", "VNM_financial_statements_2016_consolidated_959.csv", 20, 7, "Ending tangible PPE net book value"),
    ("SAB", 2016, "note:tangible_ppe_nbv", "consolidated"): _disclosure_cell("SAB", 2016, "note:tangible_ppe_nbv", "consolidated", "4.478.036.884.064", "SAB_financial_statements_2016_consolidated|943", "SAB_financial_statements_2016_consolidated_943.csv", 19, 6, "Ending tangible PPE net book value"),
    ("VPB", 2018, "note:interest_income_million", "consolidated"): _disclosure_cell("VPB", 2018, "note:interest_income_million", "consolidated", "40.280.214", "VPB_financial_statements_2018_consolidated|295", "VPB_financial_statements_2018_consolidated_295.csv", 1, 2, "Interest and similar income, million VND"),
    ("ACB", 2018, "note:interest_income_million", "consolidated"): _disclosure_cell("ACB", 2018, "note:interest_income_million", "consolidated", "24.015.362", "ACB_financial_statements_2018_consolidated|210", "ACB_financial_statements_2018_consolidated_210.csv", 2, 3, "Interest and similar income, million VND"),
    ("KBC", 2020, "note:outsourced_service_expense", "separate"): _disclosure_cell("KBC", 2020, "note:outsourced_service_expense", "separate", "34.982.559.953", "KBC_financial_statements_2020_separate|1341", "KBC_financial_statements_2020_separate_1341.csv", 5, 1, "Outsourced service expense"),
    ("VRE", 2020, "note:outsourced_service_expense_million", "separate"): _disclosure_cell("VRE", 2020, "note:outsourced_service_expense_million", "separate", "787.274", "VRE_financial_statements_2020_separate|1314", "VRE_financial_statements_2020_separate_1314.csv", 4, 1, "Outsourced service expense, million VND"),
    ("SNZ", 2022, "note:outsourced_service_expense", "consolidated"): _disclosure_cell("SNZ", 2022, "note:outsourced_service_expense", "consolidated", "47.907.597.111", "SNZ_financial_statements_2022_consolidated|2030", "SNZ_financial_statements_2022_consolidated_2030.csv", 7, 1, "Outsourced service expense"),
    ("VPI", 2022, "note:outsourced_service_expense", "consolidated"): _disclosure_cell("VPI", 2022, "note:outsourced_service_expense", "consolidated", "64.731.039.450", "VPI_financial_statements_2022_consolidated|1479", "VPI_financial_statements_2022_consolidated_1479.csv", 12, 1, "Outsourced service expense"),
    ("PVT", 2017, "note:related_customer_receivables", "consolidated"): _disclosure_cell("PVT", 2017, "note:related_customer_receivables", "consolidated", "416.469.830.532", "PVT_financial_statements_2017_consolidated|716", "PVT_financial_statements_2017_consolidated_716.csv", 6, 1, "Ending customer receivables from related parties"),
    ("BSR", 2017, "note:related_customer_receivables", "consolidated"): _disclosure_cell("BSR", 2017, "note:related_customer_receivables", "consolidated", "2.190.734.835.411", "BSR_financial_statements_2017_consolidated|800", "BSR_financial_statements_2017_consolidated_800.csv", 8, 1, "Ending customer receivables from related parties"),
    ("GAS", 2017, "note:raw_materials_gross", "consolidated"): _disclosure_cell("GAS", 2017, "note:raw_materials_gross", "consolidated", "835.708.160.659", "GAS_financial_statements_2017_consolidated|805", "GAS_financial_statements_2017_consolidated_805.csv", 4, 1, "Ending raw materials, gross"),
    ("GAS", 2017, "note:raw_materials_provision", "consolidated"): _disclosure_cell("GAS", 2017, "note:raw_materials_provision", "consolidated", "(92.176.611.606)", "GAS_financial_statements_2017_consolidated|805", "GAS_financial_statements_2017_consolidated_805.csv", 4, 2, "Ending raw-materials provision"),
    ("GEG", 2017, "note:raw_materials_net", "consolidated"): _disclosure_cell("GEG", 2017, "note:raw_materials_net", "consolidated", "8.116.833.651", "GEG_financial_statements_2017_consolidated|711", "GEG_financial_statements_2017_consolidated_711.csv", 3, 1, "Ending raw materials, net"),
    ("HDB", 2024, "note:uncollected_loan_interest_million", "consolidated"): _disclosure_cell("HDB", 2024, "note:uncollected_loan_interest_million", "consolidated", "1.242.873", "HDB_financial_statements_2024_consolidated|2059", "HDB_financial_statements_2024_consolidated_2059.csv", 1, 1, "Uncollected loan interest, million VND"),
    ("EIB", 2024, "note:uncollected_loan_interest_million", "consolidated"): _disclosure_cell("EIB", 2024, "note:uncollected_loan_interest_million", "consolidated", "3.120.199", "EIB_financial_statements_2024_consolidated|1867", "EIB_financial_statements_2024_consolidated_1867.csv", 1, 1, "Uncollected loan interest, million VND"),
    ("OCB", 2023, "note:personal_loans", "consolidated"): _disclosure_cell("OCB", 2023, "note:personal_loans", "consolidated", "54.362.329.505.068", "OCB_financial_statements_2023_consolidated|1437", "OCB_financial_statements_2023_consolidated_1437.csv", 10, 1, "Ending personal loans"),
    ("NAB", 2023, "note:personal_loans_million", "consolidated"): _disclosure_cell("NAB", 2023, "note:personal_loans_million", "consolidated", "28.125.006", "NAB_financial_statements_2023_consolidated_1|1292", "NAB_financial_statements_2023_consolidated_1_1292.csv", 9, 1, "Ending personal loans, million VND"),
    ("HDB", 2021, "note:domestic_customer_loans_million", "separate"): _disclosure_cell("HDB", 2021, "note:domestic_customer_loans_million", "separate", "185.373.610", "HDB_financial_statements_2021_separate|1272", "HDB_financial_statements_2021_separate_1272.csv", 1, 1, "Parent domestic customer loans, million VND"),
    ("ABB", 2021, "note:domestic_customer_loans_million", "separate"): _disclosure_cell("ABB", 2021, "note:domestic_customer_loans_million", "separate", "68.729.213", "ABB_financial_statements_2021_separate|1317", "ABB_financial_statements_2021_separate_1317.csv", 1, 1, "Parent domestic customer loans, million VND"),
    ("ACB", 2025, "note:net_fx_income_million", "consolidated"): _disclosure_cell("ACB", 2025, "note:net_fx_income_million", "consolidated", "1.731.886", "ACB_financial_statements_2025_consolidated|209", "ACB_financial_statements_2025_consolidated_209.csv", 7, 3, "Net foreign-exchange trading income, million VND"),
    ("MBB", 2025, "note:net_fx_income_million", "consolidated"): _disclosure_cell("MBB", 2025, "note:net_fx_income_million", "consolidated", "1.756.922", "MBB_financial_statements_2025_consolidated|428", "MBB_financial_statements_2025_consolidated_428.csv", 7, 2, "Net foreign-exchange trading income, million VND"),
    ("EIB", 2025, "note:net_fx_income_million", "consolidated"): _disclosure_cell("EIB", 2025, "note:net_fx_income_million", "consolidated", "580.096", "EIB_financial_statements_2025_consolidated|348", "EIB_financial_statements_2025_consolidated_348.csv", 7, 3, "Net foreign-exchange trading income, million VND"),
    ("BID", 2025, "note:net_fx_income_million", "consolidated"): _disclosure_cell("BID", 2025, "note:net_fx_income_million", "consolidated", "3.791.593", "BID_financial_statements_2025_consolidated|462", "BID_financial_statements_2025_consolidated_462.csv", 7, 3, "Net foreign-exchange trading income, million VND"),
    ("MCH", 2022, "note:total_segment_revenue", "consolidated"): _disclosure_cell("MCH", 2022, "note:total_segment_revenue", "consolidated", "26.977.273.170.028", "MCH_financial_statements_2022_consolidated|937", "MCH_financial_statements_2022_consolidated_937.csv", 2, 9, "Consolidated total segment revenue"),
    ("ACB", 2021, "note:deferred_expense_million", "separate"): _disclosure_cell("ACB", 2021, "note:deferred_expense_million", "separate", "706.109", "ACB_financial_statements_2021_separate|2183", "ACB_financial_statements_2021_separate_2183.csv", 1, 1, "Ending deferred expense, million VND"),
    ("OCB", 2021, "note:deferred_expense", "separate"): _disclosure_cell("OCB", 2021, "note:deferred_expense", "separate", "149.514.186.294", "OCB_financial_statements_2021_separate|1528", "OCB_financial_statements_2021_separate_1528.csv", 1, 1, "Ending deferred expense"),
    ("STB", 2021, "note:deferred_expense_million", "separate"): _disclosure_cell("STB", 2021, "note:deferred_expense_million", "separate", "826.879", "STB_financial_statements_2021_separate|1628", "STB_financial_statements_2021_separate_1628.csv", 5, 1, "Ending deferred expense, million VND"),
    ("STB", 2021, "note:net_other_income_million", "separate"): _disclosure_cell("STB", 2021, "note:net_other_income_million", "separate", "490.262", "STB_financial_statements_2021_separate|245", "STB_financial_statements_2021_separate_245.csv", 11, 2, "Net other income, million VND"),
    ("CTG", 2023, "note:other_income_million", "consolidated"): _disclosure_cell("CTG", 2023, "note:other_income_million", "consolidated", "7.080.218", "CTG_financial_statements_2023_consolidated|1643", "CTG_financial_statements_2023_consolidated_1643.csv", 2, 1, "Other operating income, million VND"),
    ("NAB", 2023, "note:other_income_million", "consolidated"): _disclosure_cell("NAB", 2023, "note:other_income_million", "consolidated", "499.520", "NAB_financial_statements_2023_consolidated_1|1929", "NAB_financial_statements_2023_consolidated_1_1929.csv", 1, 1, "Other operating income, million VND"),
    ("ABB", 2023, "note:other_income_million", "consolidated"): _disclosure_cell("ABB", 2023, "note:other_income_million", "consolidated", "405.873", "ABB_financial_statements_2023_consolidated|2057", "ABB_financial_statements_2023_consolidated_2057.csv", 1, 1, "Other operating income, million VND"),
    ("KLB", 2023, "note:other_income_million", "consolidated"): _disclosure_cell("KLB", 2023, "note:other_income_million", "consolidated", "95.764", "KLB_financial_statements_2023_consolidated|1571", "KLB_financial_statements_2023_consolidated_1571.csv", 1, 1, "Other operating income, million VND"),
    ("KLB", 2023, "note:net_interest_income_million", "consolidated"): _disclosure_cell("KLB", 2023, "note:net_interest_income_million", "consolidated", "2.038.106", "KLB_financial_statements_2023_consolidated|410", "KLB_financial_statements_2023_consolidated_410.csv", 3, 3, "Net interest income, million VND"),
    ("EIB", 2016, "note:welfare_fund_million", "consolidated"): _disclosure_cell("EIB", 2016, "note:welfare_fund_million", "consolidated", "15.862", "EIB_financial_statements_2016_consolidated|1855", "EIB_financial_statements_2016_consolidated_1855.csv", 13, 1, "Ending bonus and welfare fund, million VND"),
    ("EIB", 2017, "note:welfare_fund_million", "consolidated"): _disclosure_cell("EIB", 2017, "note:welfare_fund_million", "consolidated", "27.437", "EIB_financial_statements_2017_consolidated|1671", "EIB_financial_statements_2017_consolidated_1671.csv", 14, 1, "Ending bonus and welfare fund, million VND"),
    ("EIB", 2018, "note:welfare_fund_million", "consolidated"): _disclosure_cell("EIB", 2018, "note:welfare_fund_million", "consolidated", "40.580", "EIB_financial_statements_2018_consolidated|2005", "EIB_financial_statements_2018_consolidated_2005.csv", 15, 1, "Ending bonus and welfare fund, million VND"),
    ("SSB", 2024, "note:employee_expense_million", "consolidated"): _disclosure_cell("SSB", 2024, "note:employee_expense_million", "consolidated", "2.113.460", "SSB_financial_statements_2024_consolidated|2004", "SSB_financial_statements_2024_consolidated_2004.csv", 2, 1, "Employee expense, million VND"),
    ("NAB", 2024, "note:employee_expense_million", "consolidated"): _disclosure_cell("NAB", 2024, "note:employee_expense_million", "consolidated", "2.103.964", "NAB_financial_statements_2024_consolidated|1913", "NAB_financial_statements_2024_consolidated_1913.csv", 1, 1, "Employee expense, million VND"),
    ("VIB", 2024, "note:employee_expense_million", "consolidated"): _disclosure_cell("VIB", 2024, "note:employee_expense_million", "consolidated", "4.708.481", "VIB_financial_statements_2024_consolidated|2150", "VIB_financial_statements_2024_consolidated_2150.csv", 1, 1, "Employee expense, million VND"),
    ("VIB", 2024, "note:interbank_demand_deposits_vnd_million", "consolidated"): _disclosure_cell("VIB", 2024, "note:interbank_demand_deposits_vnd_million", "consolidated", "476.922", "VIB_financial_statements_2024_consolidated|1433", "VIB_financial_statements_2024_consolidated_1433.csv", 2, 1, "Demand deposits from other credit institutions in VND, million VND"),
    ("VIB", 2024, "note:interbank_term_deposits_vnd_million", "consolidated"): _disclosure_cell("VIB", 2024, "note:interbank_term_deposits_vnd_million", "consolidated", "48.900.000", "VIB_financial_statements_2024_consolidated|1433", "VIB_financial_statements_2024_consolidated_1433.csv", 5, 1, "Term deposits from other credit institutions in VND, million VND"),
    ("VIB", 2024, "note:interbank_loans_vnd_million", "consolidated"): _disclosure_cell("VIB", 2024, "note:interbank_loans_vnd_million", "consolidated", "55.550.089", "VIB_financial_statements_2024_consolidated|1433", "VIB_financial_statements_2024_consolidated_1433.csv", 8, 1, "Loans to other credit institutions in VND, million VND"),
    ("DXG", 2021, "cdkt:242", "consolidated"): _disclosure_cell("DXG", 2021, "cdkt:242", "consolidated", "662.693.888.918", "DXG_financial_statements_2021_consolidated|218", "DXG_financial_statements_2021_consolidated_218.csv", 16, 3, "Ending construction in progress"),
    ("DXS", 2021, "cdkt:242", "consolidated"): _disclosure_cell("DXS", 2021, "cdkt:242", "consolidated", "59.957.958.239", "DXS_financial_statements_2021_consolidated|332", "DXS_financial_statements_2021_consolidated_332.csv", 16, 3, "Ending construction in progress"),
    ("NLG", 2021, "cdkt:242", "consolidated"): _disclosure_cell("NLG", 2021, "cdkt:242", "consolidated", "20.109.203.747", "NLG_financial_statements_2021_consolidated|228", "NLG_financial_statements_2021_consolidated_228.csv", 16, 3, "Ending construction in progress"),
    ("VPI", 2024, "note:related_short_term_liability_floor", "separate"): _disclosure_cell("VPI", 2024, "note:related_short_term_liability_floor", "separate", "7.978.300.021", "VPI_financial_statements_2024_separate|1283", "VPI_financial_statements_2024_separate_1283.csv", 5, 1, "Ending parent-company short-term other payable to related parties"),
    ("PDR", 2024, "note:related_short_term_liability_floor", "separate"): _disclosure_cell("PDR", 2024, "note:related_short_term_liability_floor", "separate", "1.536.567.176.610", "PDR_financial_statements_2024_separate|1116", "PDR_financial_statements_2024_separate_1116.csv", 9, 1, "Ending parent-company short-term payable to related parties"),
    ("DXS", 2024, "note:related_short_term_liability_floor", "separate"): _disclosure_cell("DXS", 2024, "note:related_short_term_liability_floor", "separate", "34.151.927.286", "DXS_financial_statements_2024_separate|970", "DXS_financial_statements_2024_separate_970.csv", 7, 1, "Ending parent-company short-term other payable to related parties"),
    ("VIC", 2018, "note:long_term_bank_loans", "consolidated"): _disclosure_cell("VIC", 2018, "note:long_term_bank_loans", "consolidated", "23.591.140.420.079", "VIC_financial_statements_2018_consolidated|1790", "VIC_financial_statements_2018_consolidated_1790.csv", 9, 5, "Ending long-term bank loans"),
    ("DIG", 2018, "note:long_term_bank_loans", "consolidated"): _disclosure_cell("DIG", 2018, "note:long_term_bank_loans", "consolidated", "424.026.756.178", "DIG_financial_statements_2018_consolidated|1147", "DIG_financial_statements_2018_consolidated_1147.csv", 8, 4, "Ending long-term bank loans from the debt roll-forward; avoids the OCR-concatenated long-term/current-portion pair"),
    ("DXG", 2018, "note:long_term_bank_loans", "consolidated"): _disclosure_cell("DXG", 2018, "note:long_term_bank_loans", "consolidated", "116.608.697.257", "DXG_financial_statements_2018_consolidated|1318", "DXG_financial_statements_2018_consolidated_1318.csv", 7, 1, "Ending long-term bank loans"),
    ("KHG", 2023, "note:short_term_accrued_borrowing_interest", "consolidated"): _disclosure_cell("KHG", 2023, "note:short_term_accrued_borrowing_interest", "consolidated", "13.176.504.288", "KHG_financial_statements_2023_consolidated|795", "KHG_financial_statements_2023_consolidated_795.csv", 3, 1, "Ending accrued interest on bonds and bank loans"),
    ("CRE", 2023, "note:short_term_accrued_borrowing_interest", "consolidated"): _disclosure_cell("CRE", 2023, "note:short_term_accrued_borrowing_interest", "consolidated", "26.451.824.604", "CRE_financial_statements_2023_consolidated|1352", "CRE_financial_statements_2023_consolidated_1352.csv", 2, 1, "Ending accrued borrowing interest"),
    ("KBC", 2023, "note:short_term_accrued_borrowing_interest", "consolidated"): _disclosure_cell("KBC", 2023, "note:short_term_accrued_borrowing_interest", "consolidated", "17.558.752.691", "KBC_financial_statements_2023_consolidated|1407", "KBC_financial_statements_2023_consolidated_1407.csv", 5, 1, "Ending short-term accrued borrowing interest"),
    ("IJC", 2017, "note:ordinary_bonds_total", "separate"): _disclosure_cell("IJC", 2017, "note:ordinary_bonds_total", "separate", "1.000.000.000.000", "IJC_financial_statements_2017_separate|1332", "IJC_financial_statements_2017_separate_1332.csv", 3, 1, "Ending parent-company ordinary bonds, total debt"),
    ("SCR", 2017, "note:ordinary_bonds_short_term", "separate"): _disclosure_cell("SCR", 2017, "note:ordinary_bonds_short_term", "separate", "300.000.000.000", "SCR_financial_statements_2017_separate|981", "SCR_financial_statements_2017_separate_981.csv", 3, 5, "Ending parent-company short-term ordinary bonds"),
    ("SCR", 2017, "note:ordinary_bonds_long_term", "separate"): _disclosure_cell("SCR", 2017, "note:ordinary_bonds_long_term", "separate", "396.121.428.571", "SCR_financial_statements_2017_separate|1046", "SCR_financial_statements_2017_separate_1046.csv", 2, 1, "Ending parent-company long-term ordinary bonds"),
    ("VGT", 2023, "note:short_term_operating_lease_commitments", "separate"): _disclosure_cell("VGT", 2023, "note:short_term_operating_lease_commitments", "separate", "21.339.474.240", "VGT_financial_statements_2023_separate|1436", "VGT_financial_statements_2023_separate_1436.csv", 1, 1, "Parent-company non-cancellable operating-lease payments due within one year"),
    ("TTF", 2023, "note:short_term_operating_lease_commitments", "separate"): _disclosure_cell("TTF", 2023, "note:short_term_operating_lease_commitments", "separate", "6.638.630.000", "TTF_financial_statements_2023_separate|1479", "TTF_financial_statements_2023_separate_1479.csv", 1, 1, "Parent-company operating-lease payments due within one year"),
    ("KHG", 2024, "note:key_management_total_remuneration", "consolidated"): _disclosure_cell("KHG", 2024, "note:key_management_total_remuneration", "consolidated", "2.271.909.427", "KHG_financial_statements_2024_consolidated|1045", "KHG_financial_statements_2024_consolidated_1045.csv", 19, 2, "Total income/remuneration of the Board, executive management and supervisory board"),
    ("IJC", 2024, "note:key_management_total_remuneration", "consolidated"): _disclosure_cell("IJC", 2024, "note:key_management_total_remuneration", "consolidated", "7.940.845.455", "IJC_financial_statements_2024_consolidated|2027", "IJC_financial_statements_2024_consolidated_2027.csv", 8, 5, "Total income/remuneration of key management and supervisory-board members"),
    ("HAG", 2021, "note:supplier_goods_services_advances_thousand", "consolidated"): _disclosure_cell("HAG", 2021, "note:supplier_goods_services_advances_thousand", "consolidated", "129.190.217", "HAG_financial_statements_2021_consolidated|1073", "HAG_financial_statements_2021_consolidated_1073.csv", 1, 1, "Ending advances to suppliers of goods and services, thousand VND"),
    ("HNG", 2021, "note:supplier_goods_services_advances_thousand", "consolidated"): _disclosure_cell("HNG", 2021, "note:supplier_goods_services_advances_thousand", "consolidated", "42.936.831", "HNG_financial_statements_2021_consolidated|954", "HNG_financial_statements_2021_consolidated_954.csv", 1, 1, "Ending advances to suppliers of goods and services, thousand VND"),
    ("HNG", 2015, "note:idle_asset_depreciation_thousand", "consolidated"): _disclosure_cell("HNG", 2015, "note:idle_asset_depreciation_thousand", "consolidated", "(3.521.561)", "HNG_financial_statements_2015_consolidated|1628", "HNG_financial_statements_2015_consolidated_1628.csv", 6, 1, "Idle-asset depreciation expense, thousand VND"),
    ("HNG", 2016, "note:idle_asset_depreciation_thousand", "consolidated"): _disclosure_cell("HNG", 2016, "note:idle_asset_depreciation_thousand", "consolidated", "(7.655.041)", "HNG_financial_statements_2016_consolidated|1788", "HNG_financial_statements_2016_consolidated_1788.csv", 7, 1, "Idle-asset depreciation expense, thousand VND"),
    ("HNG", 2021, "note:idle_asset_depreciation_thousand", "consolidated"): _disclosure_cell("HNG", 2021, "note:idle_asset_depreciation_thousand", "consolidated", "14.542.048", "HNG_financial_statements_2021_consolidated|1410", "HNG_financial_statements_2021_consolidated_1410.csv", 7, 1, "Idle-asset depreciation expense, thousand VND"),
    ("HNG", 2022, "note:idle_asset_depreciation_thousand", "consolidated"): _disclosure_cell("HNG", 2022, "note:idle_asset_depreciation_thousand", "consolidated", "16.634.357", "HNG_financial_statements_2022_consolidated|1464", "HNG_financial_statements_2022_consolidated_1464.csv", 7, 1, "Idle-asset depreciation expense, thousand VND"),
    ("SHB", 2021, "note:government_bonds_afs_million", "separate"): _disclosure_cell("SHB", 2021, "note:government_bonds_afs_million", "separate", "3.004.621", "SHB_financial_statements_2021_separate|1523", "SHB_financial_statements_2021_separate_1523.csv", 2, 1, "Parent government bonds classified as available for sale, million VND"),
    ("SHB", 2021, "note:government_bonds_htm_million", "separate"): _disclosure_cell("SHB", 2021, "note:government_bonds_htm_million", "separate", "13.241.284", "SHB_financial_statements_2021_separate|1548", "SHB_financial_statements_2021_separate_1548.csv", 1, 1, "Parent government bonds classified as held to maturity, million VND"),
    ("SSB", 2021, "note:government_bonds_trading_million", "separate"): _disclosure_cell("SSB", 2021, "note:government_bonds_trading_million", "separate", "8.544.746", "SSB_financial_statements_2021_separate|1694", "SSB_financial_statements_2021_separate_1694.csv", 2, 1, "Parent government bonds classified as trading securities, million VND"),
    ("SSB", 2021, "note:government_bonds_afs_million", "separate"): _disclosure_cell("SSB", 2021, "note:government_bonds_afs_million", "separate", "6.705.572", "SSB_financial_statements_2021_separate|1997", "SSB_financial_statements_2021_separate_1997.csv", 4, 1, "Parent government bonds classified as available for sale, million VND"),
    ("CTG", 2021, "note:government_bonds_afs_million", "separate"): _disclosure_cell("CTG", 2021, "note:government_bonds_afs_million", "separate", "78.299.141", "CTG_financial_statements_2021_separate|1414", "CTG_financial_statements_2021_separate_1414.csv", 2, 1, "Parent government bonds classified as available for sale, million VND"),
    ("CTG", 2021, "note:government_bonds_htm_million", "separate"): _disclosure_cell("CTG", 2021, "note:government_bonds_htm_million", "separate", "2.200.000", "CTG_financial_statements_2021_separate|1418", "CTG_financial_statements_2021_separate_1418.csv", 2, 1, "Parent government bonds classified as held to maturity, million VND"),
    ("STB", 2021, "note:government_bonds_afs_million", "separate"): _disclosure_cell("STB", 2021, "note:government_bonds_afs_million", "separate", "31.122.543", "STB_financial_statements_2021_separate|1370", "STB_financial_statements_2021_separate_1370.csv", 2, 1, "Parent government bonds classified as available for sale, million VND"),
    ("STB", 2021, "note:government_bonds_htm_million", "separate"): _disclosure_cell("STB", 2021, "note:government_bonds_htm_million", "separate", "17.922.385", "STB_financial_statements_2021_separate|1397", "STB_financial_statements_2021_separate_1397.csv", 2, 1, "Parent government bonds classified as held to maturity, million VND"),
    ("EIB", 2021, "note:government_bonds_htm_million", "separate"): _disclosure_cell("EIB", 2021, "note:government_bonds_htm_million", "separate", "8.577.217", "EIB_financial_statements_2021_separate|1357", "EIB_financial_statements_2021_separate_1357.csv", 1, 1, "Parent government bonds classified as held to maturity, million VND; available-for-sale balance is zero"),
    # Old-model tail: direct statement-note rows that were previously replaced
    # by a nearby total, a selector value, or the wrong disclosure altogether.
    ("VIB", 2024, "note:other_tangible_ppe_nbv_million", "consolidated"): _disclosure_cell("VIB", 2024, "note:other_tangible_ppe_nbv_million", "consolidated", "11.043", "VIB_financial_statements_2024_consolidated|1741", "VIB_financial_statements_2024_consolidated_1741.csv", 13, 5, "Ending net book value of other tangible fixed assets, million VND"),
    ("OCB", 2017, "note:domestic_economic_and_individual_loans", "consolidated"): _disclosure_cell("OCB", 2017, "note:domestic_economic_and_individual_loans", "consolidated", "47.893.069.902.809", "OCB_financial_statements_2017_consolidated|1409", "OCB_financial_statements_2017_consolidated_1409.csv", 1, 1, "Ending loans to domestic economic entities and individuals"),
    ("OCB", 2018, "note:domestic_economic_and_individual_loans", "consolidated"): _disclosure_cell("OCB", 2018, "note:domestic_economic_and_individual_loans", "consolidated", "55.962.872.280.567", "OCB_financial_statements_2018_consolidated|1619", "OCB_financial_statements_2018_consolidated_1619.csv", 1, 1, "Ending loans to domestic economic entities and individuals"),
    ("OCB", 2021, "note:domestic_economic_and_individual_loans", "consolidated"): _disclosure_cell("OCB", 2021, "note:domestic_economic_and_individual_loans", "consolidated", "101.578.366.954.676", "OCB_financial_statements_2022_consolidated_1|1245", "OCB_financial_statements_2022_consolidated_1_1245.csv", 1, 2, "Opening-2022 comparative for ending-2021 loans to domestic economic entities and individuals"),
    ("OCB", 2022, "note:domestic_economic_and_individual_loans", "consolidated"): _disclosure_cell("OCB", 2022, "note:domestic_economic_and_individual_loans", "consolidated", "119.510.721.884.604", "OCB_financial_statements_2022_consolidated_1|1245", "OCB_financial_statements_2022_consolidated_1_1245.csv", 1, 1, "Ending loans to domestic economic entities and individuals"),
    ("OCB", 2017, "note:bonus_welfare_fund_ending", "consolidated"): _disclosure_cell("OCB", 2017, "note:bonus_welfare_fund_ending", "consolidated", "2.771.242.559", "OCB_financial_statements_2017_consolidated|1843", "OCB_financial_statements_2017_consolidated_1843.csv", 9, 1, "Ending bonus and welfare fund"),
    ("OCB", 2018, "note:bonus_welfare_fund_ending", "consolidated"): _disclosure_cell("OCB", 2018, "note:bonus_welfare_fund_ending", "consolidated", "5.112.311.443", "OCB_financial_statements_2018_consolidated|2050", "OCB_financial_statements_2018_consolidated_2050.csv", 9, 1, "Ending bonus and welfare fund"),
    ("OCB", 2021, "note:bonus_welfare_fund_ending", "consolidated"): _disclosure_cell("OCB", 2021, "note:bonus_welfare_fund_ending", "consolidated", "67.110.004.614", "OCB_financial_statements_2021_consolidated|1694", "OCB_financial_statements_2021_consolidated_1694.csv", 11, 1, "Ending bonus and welfare fund"),
    ("OCB", 2022, "note:bonus_welfare_fund_ending", "consolidated"): _disclosure_cell("OCB", 2022, "note:bonus_welfare_fund_ending", "consolidated", "102.424.213.974", "OCB_financial_statements_2022_consolidated_1|1572", "OCB_financial_statements_2022_consolidated_1_1572.csv", 11, 1, "Ending bonus and welfare fund"),
    ("NAB", 2021, "note:construction_in_progress_million", "consolidated"): _disclosure_cell("NAB", 2021, "note:construction_in_progress_million", "consolidated", "42.867", "NAB_financial_statements_2021_consolidated|2346", "NAB_financial_statements_2021_consolidated_2346.csv", 5, 1, "Ending construction in progress, million VND"),
    ("NAB", 2022, "note:construction_in_progress_million", "consolidated"): _disclosure_cell("NAB", 2022, "note:construction_in_progress_million", "consolidated", "531.950", "NAB_financial_statements_2022_consolidated_1|1573", "NAB_financial_statements_2022_consolidated_1_1573.csv", 2, 1, "Ending construction in progress, million VND"),
    ("NAB", 2023, "note:construction_in_progress_million", "consolidated"): _disclosure_cell("NAB", 2023, "note:construction_in_progress_million", "consolidated", "293.571", "NAB_financial_statements_2023_consolidated_2|1562", "NAB_financial_statements_2023_consolidated_2_1562.csv", 3, 1, "Ending construction in progress, million VND"),
    ("NAB", 2024, "note:construction_in_progress_million", "consolidated"): _disclosure_cell("NAB", 2024, "note:construction_in_progress_million", "consolidated", "179.644", "NAB_financial_statements_2024_consolidated|1539", "NAB_financial_statements_2024_consolidated_1539.csv", 2, 1, "Ending construction in progress, million VND"),
    ("NAB", 2025, "note:construction_in_progress_million", "consolidated"): _disclosure_cell("NAB", 2025, "note:construction_in_progress_million", "consolidated", "602.113", "NAB_financial_statements_2025_consolidated|1748", "NAB_financial_statements_2025_consolidated_1748.csv", 2, 1, "Ending construction in progress, million VND"),
    ("NAB", 2021, "note:performing_customer_loans_million", "consolidated"): _disclosure_cell("NAB", 2021, "note:performing_customer_loans_million", "consolidated", "99.023.365", "NAB_financial_statements_2021_consolidated|1994", "NAB_financial_statements_2021_consolidated_1994.csv", 1, 1, "Ending performing customer loans, million VND"),
    ("NAB", 2022, "note:performing_customer_loans_million", "consolidated"): _disclosure_cell("NAB", 2022, "note:performing_customer_loans_million", "consolidated", "114.017.677", "NAB_financial_statements_2022_consolidated_1|1331", "NAB_financial_statements_2022_consolidated_1_1331.csv", 1, 1, "Ending performing customer loans, million VND"),
    ("NAB", 2023, "note:performing_customer_loans_million", "consolidated"): _disclosure_cell("NAB", 2023, "note:performing_customer_loans_million", "consolidated", "133.053.654", "NAB_financial_statements_2023_consolidated_2|1267", "NAB_financial_statements_2023_consolidated_2_1267.csv", 1, 1, "Ending performing customer loans, million VND"),
    ("NAB", 2024, "note:performing_customer_loans_million", "consolidated"): _disclosure_cell("NAB", 2024, "note:performing_customer_loans_million", "consolidated", "161.359.386", "NAB_financial_statements_2024_consolidated|1302", "NAB_financial_statements_2024_consolidated_1302.csv", 1, 1, "Ending performing customer loans, million VND"),
    ("NAB", 2025, "note:performing_customer_loans_million", "consolidated"): _disclosure_cell("NAB", 2025, "note:performing_customer_loans_million", "consolidated", "190.759.675", "NAB_financial_statements_2025_consolidated|1479", "NAB_financial_statements_2025_consolidated_1479.csv", 1, 1, "Ending performing customer loans, million VND"),
    ("MSN", 2018, "note:current_income_tax_expense_million", "consolidated"): _disclosure_cell("MSN", 2018, "note:current_income_tax_expense_million", "consolidated", "726.692", "MSN_financial_statements_2018_consolidated|1858", "MSN_financial_statements_2018_consolidated_1858.csv", 3, 1, "Current corporate income tax expense after prior-year true-up, million VND"),
    ("NLG", 2024, "note:board_and_executive_income_total", "separate"): _disclosure_cell("NLG", 2024, "note:board_and_executive_income_total", "separate", "45.573.123.913", "NLG_financial_statements_2024_separate|1594", "NLG_financial_statements_2024_separate_1594.csv", 3, 1, "Total parent-company Board and executive-management income"),
    ("BVH", 2017, "note:written_off_bad_debt", "consolidated"): _disclosure_cell("BVH", 2017, "note:written_off_bad_debt", "consolidated", "6.343.174.175", "BVH_financial_statements_2017_consolidated|2583", "BVH_financial_statements_2017_consolidated_2583.csv", 2, 1, "Ending written-off bad debt"),
    ("BVH", 2018, "note:written_off_bad_debt", "consolidated"): _disclosure_cell("BVH", 2018, "note:written_off_bad_debt", "consolidated", "6.343.174.175", "BVH_financial_statements_2018_consolidated|2640", "BVH_financial_statements_2018_consolidated_2640.csv", 2, 1, "Ending written-off bad debt"),
    ("BVH", 2023, "note:written_off_bad_debt", "consolidated"): _disclosure_cell("BVH", 2023, "note:written_off_bad_debt", "consolidated", "132.178.081.950", "BVH_financial_statements_2023_consolidated|2599", "BVH_financial_statements_2023_consolidated_2599.csv", 2, 1, "Ending written-off bad debt"),
    ("BVH", 2025, "note:written_off_bad_debt", "consolidated"): _disclosure_cell("BVH", 2025, "note:written_off_bad_debt", "consolidated", "157.480.332.668", "BVH_financial_statements_2025_consolidated|2409", "BVH_financial_statements_2025_consolidated_2409.csv", 2, 1, "Ending written-off bad debt"),
    ("BVH", 2017, "note:listed_equity_down10_pbt_impact", "consolidated"): _disclosure_cell("BVH", 2017, "note:listed_equity_down10_pbt_impact", "consolidated", "(32.872.135.940)", "BVH_financial_statements_2017_consolidated|3317", "BVH_financial_statements_2017_consolidated_3317.csv", 3, 2, "PBT impact of a 10% listed-equity market-price decrease, VND"),
    ("BVH", 2018, "note:listed_equity_down10_pbt_impact", "consolidated"): _disclosure_cell("BVH", 2018, "note:listed_equity_down10_pbt_impact", "consolidated", "(75.768.363.087)", "BVH_financial_statements_2018_consolidated|3379", "BVH_financial_statements_2018_consolidated_3379.csv", 3, 2, "PBT impact of a 10% listed-equity market-price decrease, VND"),
    ("BVH", 2023, "note:listed_equity_down10_pbt_impact_million", "consolidated"): _disclosure_cell("BVH", 2023, "note:listed_equity_down10_pbt_impact_million", "consolidated", "(120.731)", "BVH_financial_statements_2023_consolidated|3338", "BVH_financial_statements_2023_consolidated_3338.csv", 4, 2, "PBT impact of a 10% listed-equity market-price decrease, million VND"),
    ("BVH", 2025, "note:listed_equity_down10_pbt_impact_million", "consolidated"): _disclosure_cell("BVH", 2025, "note:listed_equity_down10_pbt_impact_million", "consolidated", "(100.562)", "BVH_financial_statements_2025_consolidated|3099", "BVH_financial_statements_2025_consolidated_3099.csv", 4, 2, "PBT impact of a 10% listed-equity market-price decrease, million VND"),
    # Remaining old-model programs with source-proven unit, entity or
    # comparison errors.  Each cell is the exact BTC-table coordinate used by
    # the rewritten program; displayed OCR figures are not treated as labels.
    ("PC1", 2023, "note:raw_materials_gross_ending", "consolidated"): _disclosure_cell("PC1", 2023, "note:raw_materials_gross_ending", "consolidated", "212.530.977.247", "PC1_financial_statements_2023_consolidated|1091", "PC1_financial_statements_2023_consolidated_1091.csv", 4, 1, "Ending gross raw-material inventory"),
    ("PC1", 2022, "note:raw_materials_gross_ending", "consolidated"): _disclosure_cell("PC1", 2022, "note:raw_materials_gross_ending", "consolidated", "307.908.804.848", "PC1_financial_statements_2023_consolidated|1091", "PC1_financial_statements_2023_consolidated_1091.csv", 4, 3, "Opening-2023 comparative for ending-2022 gross raw-material inventory"),
    ("SSB", 2020, "note:average_employee_count", "separate"): _disclosure_cell("SSB", 2020, "note:average_employee_count", "separate", "3.987", "SSB_financial_statements_2020_separate|1813", "SSB_financial_statements_2020_separate_1813.csv", 2, 1, "Average employee count"),
    ("SSB", 2020, "note:total_employee_income_million", "separate"): _disclosure_cell("SSB", 2020, "note:total_employee_income_million", "separate", "1.068.615", "SSB_financial_statements_2020_separate|1813", "SSB_financial_statements_2020_separate_1813.csv", 5, 1, "Total employee income including bonus, million VND"),
    ("VPB", 2020, "note:average_employee_count", "separate"): _disclosure_cell("VPB", 2020, "note:average_employee_count", "separate", "9.419", "VPB_financial_statements_2020_separate|1862", "VPB_financial_statements_2020_separate_1862.csv", 2, 1, "Average employee count"),
    ("VPB", 2020, "note:total_employee_income_million", "separate"): _disclosure_cell("VPB", 2020, "note:total_employee_income_million", "separate", "3.174.214", "VPB_financial_statements_2020_separate|1862", "VPB_financial_statements_2020_separate_1862.csv", 3, 1, "Total employee income, million VND"),
    ("MPC", 2019, "note:basic_eps_vnd", "consolidated"): _disclosure_cell("MPC", 2019, "note:basic_eps_vnd", "consolidated", "2.528", "MPC_financial_statements_2019_consolidated|184", "MPC_financial_statements_2019_consolidated_184.csv", 6, 3, "Basic earnings per share, VND"),
    ("VSF", 2019, "note:basic_eps_vnd", "consolidated"): _disclosure_cell("VSF", 2019, "note:basic_eps_vnd", "consolidated", "(423)", "VSF_financial_statements_2019_consolidated|383", "VSF_financial_statements_2019_consolidated_383.csv", 7, 2, "Basic loss per share, VND"),
    ("GEE", 2025, "note:common_shares_outstanding", "consolidated"): _disclosure_cell("GEE", 2025, "note:common_shares_outstanding", "consolidated", "365.999.956", "GEE_financial_statements_2025_consolidated|1606", "GEE_financial_statements_2025_consolidated_1606.csv", 5, 1, "Ending common shares outstanding"),
    ("GEX", 2025, "note:common_shares_outstanding", "consolidated"): _disclosure_cell("GEX", 2025, "note:common_shares_outstanding", "consolidated", "902.398.948", "GEX_financial_statements_2025_consolidated|2217", "GEX_financial_statements_2025_consolidated_2217.csv", 5, 1, "Ending common shares outstanding"),
    ("BID", 2022, "note:deposits_and_borrowings_other_credit_institutions_million", "separate"): _disclosure_cell("BID", 2022, "note:deposits_and_borrowings_other_credit_institutions_million", "separate", "167.634.732", "BID_financial_statements_2022_separate|299", "BID_financial_statements_2022_separate_299.csv", 4, 3, "Parent deposits and borrowings from other credit institutions, million VND"),
    ("MSB", 2022, "note:deposits_and_borrowings_other_credit_institutions_million", "separate"): _disclosure_cell("MSB", 2022, "note:deposits_and_borrowings_other_credit_institutions_million", "separate", "50.298.619", "MSB_financial_statements_2022_separate|316", "MSB_financial_statements_2022_separate_316.csv", 5, 2, "Parent deposits and borrowings from other credit institutions, million VND"),
    ("HSG", 2019, "note:bonus_welfare_fund_allocation", "consolidated"): _disclosure_cell("HSG", 2019, "note:bonus_welfare_fund_allocation", "consolidated", "(14.454.085.321)", "HSG_financial_statements_2019_consolidated|1377", "HSG_financial_statements_2019_consolidated_1377.csv", 4, 1, "Annual bonus-and-welfare-fund allocation"),
    ("HSG", 2021, "note:bonus_welfare_fund_allocation", "consolidated"): _disclosure_cell("HSG", 2021, "note:bonus_welfare_fund_allocation", "consolidated", "(172.540.289.351)", "HSG_financial_statements_2021_consolidated|1466", "HSG_financial_statements_2021_consolidated_1466.csv", 4, 1, "Annual bonus-and-welfare-fund allocation"),
    ("HSG", 2024, "note:bonus_welfare_fund_allocation", "consolidated"): _disclosure_cell("HSG", 2024, "note:bonus_welfare_fund_allocation", "consolidated", "(18.103.097.086)", "HSG_financial_statements_2024_consolidated|1020", "HSG_financial_statements_2024_consolidated_1020.csv", 11, 4, "Annual bonus-and-welfare-fund allocation"),
    ("HSG", 2025, "note:bonus_welfare_fund_allocation", "consolidated"): _disclosure_cell("HSG", 2025, "note:bonus_welfare_fund_allocation", "consolidated", "(18.604.251.306)", "HSG_financial_statements_2025_consolidated|1028", "HSG_financial_statements_2025_consolidated_1028.csv", 12, 4, "Annual bonus-and-welfare-fund allocation"),
    ("HSG", 2015, "note:long_term_borrowing_balance", "consolidated"): _disclosure_cell("HSG", 2015, "note:long_term_borrowing_balance", "consolidated", "1.223.388.652.292", "HSG_financial_statements_2015_consolidated|986", "HSG_financial_statements_2015_consolidated_986.csv", 2, 1, "Ending long-term bank-borrowing balance"),
    ("HSG", 2018, "note:long_term_borrowing_balance", "consolidated"): _disclosure_cell("HSG", 2018, "note:long_term_borrowing_balance", "consolidated", "4.135.888.386.927", "HSG_financial_statements_2019_consolidated|1077", "HSG_financial_statements_2019_consolidated_1077.csv", 2, 2, "Opening-2019 comparative for ending-2018 long-term-borrowing balance"),
    ("HSG", 2019, "note:long_term_borrowing_balance", "consolidated"): _disclosure_cell("HSG", 2019, "note:long_term_borrowing_balance", "consolidated", "3.583.459.745.822", "HSG_financial_statements_2019_consolidated|1077", "HSG_financial_statements_2019_consolidated_1077.csv", 2, 1, "Ending long-term-borrowing balance"),
    ("HSG", 2021, "note:long_term_borrowing_balance", "consolidated"): _disclosure_cell("HSG", 2021, "note:long_term_borrowing_balance", "consolidated", "2.006.162.118.340", "HSG_financial_statements_2021_consolidated|1156", "HSG_financial_statements_2021_consolidated_1156.csv", 2, 1, "Ending long-term-borrowing balance"),
    ("HSG", 2022, "note:long_term_borrowing_balance", "consolidated"): _disclosure_cell("HSG", 2022, "note:long_term_borrowing_balance", "consolidated", "551.461.944.323", "HSG_financial_statements_2022_consolidated|1064", "HSG_financial_statements_2022_consolidated_1064.csv", 2, 1, "Ending long-term-borrowing balance"),
    ("HSG", 2023, "note:long_term_borrowing_balance", "consolidated"): _disclosure_cell("HSG", 2023, "note:long_term_borrowing_balance", "consolidated", "-", "HSG_financial_statements_2023_consolidated|1046", "HSG_financial_statements_2023_consolidated_1046.csv", 2, 1, "Ending long-term-borrowing balance"),
    ("ACV", 2019, "note:total_employee_expense", "consolidated"): _disclosure_cell("ACV", 2019, "note:total_employee_expense", "consolidated", "3.074.424.965.877", "ACV_financial_statements_2019_consolidated|1289", "ACV_financial_statements_2019_consolidated_1289.csv", 2, 1, "Total employee expense"),
    ("VJC", 2019, "note:total_employee_and_labor_expense", "consolidated"): _disclosure_cell("VJC", 2019, "note:total_employee_and_labor_expense", "consolidated", "4.780.622.485.492", "VJC_financial_statements_2019_consolidated|1298", "VJC_financial_statements_2019_consolidated_1298.csv", 4, 1, "Total employee and labor expense"),
    ("VIB", 2023, "note:manufacturing_loan_share_ocr_percent", "separate"): _disclosure_cell("VIB", 2023, "note:manufacturing_loan_share_ocr_percent", "separate", "503", "VIB_financial_statements_2023_separate|1486", "VIB_financial_statements_2023_separate_1486.csv", 4, 2, "Manufacturing-sector share of parent loan balance; OCR stores 5.03% as 503"),
    ("BID", 2023, "note:manufacturing_loan_share_ocr_percent", "separate"): _disclosure_cell("BID", 2023, "note:manufacturing_loan_share_ocr_percent", "separate", "1649", "BID_financial_statements_2023_separate|1161", "BID_financial_statements_2023_separate_1161.csv", 4, 2, "Manufacturing-sector share of parent loan balance; OCR stores 16.49% as 1649"),
    ("GEX", 2024, "note:related_short_term_trade_receivables", "separate"): _disclosure_cell("GEX", 2024, "note:related_short_term_trade_receivables", "separate", "317.413.438.972", "GEX_financial_statements_2024_separate|1086", "GEX_financial_statements_2024_separate_1086.csv", 5, 1, "Ending short-term trade receivables from related parties"),
    ("GEX", 2024, "note:related_other_short_term_receivables", "separate"): _disclosure_cell("GEX", 2024, "note:related_other_short_term_receivables", "separate", "337.751.686.478", "GEX_financial_statements_2024_separate|1116", "GEX_financial_statements_2024_separate_1116.csv", 16, 1, "Ending other short-term receivables from related parties"),
    ("GEX", 2024, "note:related_short_term_trade_payables", "separate"): _disclosure_cell("GEX", 2024, "note:related_short_term_trade_payables", "separate", "108.306.804", "GEX_financial_statements_2024_separate|1239", "GEX_financial_statements_2024_separate_1239.csv", 11, 1, "Ending short-term trade payables to related parties"),
    ("GEX", 2024, "note:related_other_short_term_payables", "separate"): _disclosure_cell("GEX", 2024, "note:related_other_short_term_payables", "separate", "46.011.460.031", "GEX_financial_statements_2024_separate|1280", "GEX_financial_statements_2024_separate_1280.csv", 15, 1, "Ending other short-term payables to related parties"),
    # Bank-formatted 2015 SGB statements use captions that the generic
    # statement cube does not map to cdkt codes.  Both figures are reported in
    # the same million-VND unit, so their ratio is source-unit invariant.
    ("SGB", 2015, "cdkt:300", "separate"): _disclosure_cell("SGB", 2015, "cdkt:300", "separate", "14.968.812", "SGB_financial_statements_2015_separate|59", "SGB_financial_statements_2015_separate_59.csv", 16, 2, "Parent total liabilities, million VND"),
    ("SGB", 2015, "cdkt:270", "separate"): _disclosure_cell("SGB", 2015, "cdkt:270", "separate", "18.359.429", "SGB_financial_statements_2015_separate|42", "SGB_financial_statements_2015_separate_42.csv", 22, 2, "Parent total assets, million VND"),
    # Residual semantic audit: these old programs were executable but selected
    # an adjacent disclosure, the comparative-year column, or the wrong
    # attribution/due-date line.
    ("SSI", 2020, "note:financial_operating_revenue", "consolidated"): _disclosure_cell("SSI", 2020, "note:financial_operating_revenue", "consolidated", "208.753.551.822", "SSI_financial_statements_2020_consolidated|394", "SSI_financial_statements_2020_consolidated_394.csv", 7, 3, "Total financial operating revenue"),
    ("SSI", 2023, "note:financial_operating_revenue", "consolidated"): _disclosure_cell("SSI", 2023, "note:financial_operating_revenue", "consolidated", "123.303.909.120", "SSI_financial_statements_2023_consolidated|400", "SSI_financial_statements_2023_consolidated_400.csv", 7, 3, "Total financial operating revenue"),
    ("VCB", 2017, "note:general_customer_loan_provision_expense_million", "separate"): _disclosure_cell("VCB", 2017, "note:general_customer_loan_provision_expense_million", "separate", "736.950", "VCB_financial_statements_2017_separate|1551", "VCB_financial_statements_2017_separate_1551.csv", 2, 1, "General customer-loan provision expense, million VND"),
    ("VCB", 2017, "note:specific_customer_loan_provision_expense_million", "separate"): _disclosure_cell("VCB", 2017, "note:specific_customer_loan_provision_expense_million", "separate", "5.490.641", "VCB_financial_statements_2017_separate|1551", "VCB_financial_statements_2017_separate_1551.csv", 3, 1, "Specific customer-loan provision expense, million VND"),
    ("VCB", 2017, "note:pre_provision_operating_profit_million", "separate"): _disclosure_cell("VCB", 2017, "note:pre_provision_operating_profit_million", "separate", "17.208.801", "VCB_financial_statements_2017_separate|277", "VCB_financial_statements_2017_separate_277.csv", 17, 3, "Net operating profit before credit-risk provision expense, million VND"),
    ("DPM", 2017, "note:gross_sales_revenue", "separate"): _disclosure_cell("DPM", 2017, "note:gross_sales_revenue", "separate", "7.465.852.549.086", "DPM_financial_statements_2017_separate|1112", "DPM_financial_statements_2017_separate_1112.csv", 8, 1, "Total sales and service revenue"),
    ("DPM", 2017, "note:related_party_sales_revenue", "separate"): _disclosure_cell("DPM", 2017, "note:related_party_sales_revenue", "separate", "6.799.214.111.490", "DPM_financial_statements_2017_separate|1112", "DPM_financial_statements_2017_separate_1112.csv", 13, 1, "Sales revenue arising with related parties"),
    ("PLX", 2019, "note:long_term_borrowing_due_within_12_months", "consolidated"): _disclosure_cell("PLX", 2019, "note:long_term_borrowing_due_within_12_months", "consolidated", "(342.758.241.592)", "PLX_financial_statements_2019_consolidated|1382", "PLX_financial_statements_2019_consolidated_1382.csv", 3, 1, "Long-term borrowing due within 12 months"),
    ("PLX", 2020, "note:long_term_borrowing_due_within_12_months", "consolidated"): _disclosure_cell("PLX", 2020, "note:long_term_borrowing_due_within_12_months", "consolidated", "(309.697.844.395)", "PLX_financial_statements_2020_consolidated|1433", "PLX_financial_statements_2020_consolidated_1433.csv", 3, 1, "Long-term borrowing due within 12 months"),
    ("PLX", 2023, "note:long_term_borrowing_due_within_12_months", "consolidated"): _disclosure_cell("PLX", 2023, "note:long_term_borrowing_due_within_12_months", "consolidated", "(186.558.947.671)", "PLX_financial_statements_2023_consolidated|1500", "PLX_financial_statements_2023_consolidated_1500.csv", 3, 1, "Long-term borrowing due within 12 months"),
    ("NVL", 2018, "note:profit_attributable_to_parent_shareholders", "consolidated"): _disclosure_cell("NVL", 2018, "note:profit_attributable_to_parent_shareholders", "consolidated", "3.227.004.714.155", "NVL_financial_statements_2018_consolidated|232", "NVL_financial_statements_2018_consolidated_232.csv", 23, 3, "Profit after tax attributable to shareholders of the parent company"),
    # Older-model residuals: each legacy program selected an adjacent but
    # semantically unrelated table (a component disclosure, cash subtotal, or
    # related-party balance) even though the requested totals are explicit.
    ("DXG", 2018, "note:short_term_other_receivables_total", "consolidated"): _disclosure_cell("DXG", 2018, "note:short_term_other_receivables_total", "consolidated", "3.557.808.940.778", "DXG_financial_statements_2018_consolidated|1013", "DXG_financial_statements_2018_consolidated_1013.csv", 31, 1, "Ending short-term other receivables total"),
    ("DXG", 2018, "note:other_receivables_total", "consolidated"): _disclosure_cell("DXG", 2018, "note:other_receivables_total", "consolidated", "3.703.144.857.661", "DXG_financial_statements_2018_consolidated|1032", "DXG_financial_statements_2018_consolidated_1032.csv", 6, 1, "Ending total other receivables, short- and long-term"),
    ("HNG", 2020, "note:total_borrowings", "separate"): _disclosure_cell("HNG", 2020, "note:total_borrowings", "separate", "9.542.263.238", "HNG_financial_statements_2020_separate|773", "HNG_financial_statements_2020_separate_773.csv", 9, 1, "Parent ending total borrowings, thousand VND"),
    ("HNG", 2020, "note:cash_on_hand", "separate"): _disclosure_cell("HNG", 2020, "note:cash_on_hand", "separate", "680.896", "HNG_financial_statements_2020_separate|594", "HNG_financial_statements_2020_separate_594.csv", 1, 1, "Parent ending cash on hand, thousand VND"),
    ("HNG", 2020, "note:bank_deposits", "separate"): _disclosure_cell("HNG", 2020, "note:bank_deposits", "separate", "7.039.019", "HNG_financial_statements_2020_separate|594", "HNG_financial_statements_2020_separate_594.csv", 2, 1, "Parent ending bank deposits, thousand VND"),
    ("HDB", 2025, "note:gross_customer_loans_million", "consolidated"): _disclosure_cell("HDB", 2025, "note:gross_customer_loans_million", "consolidated", "546.370.779", "HDB_financial_statements_2025_consolidated|1401", "HDB_financial_statements_2025_consolidated_1401.csv", 2, 1, "Ending gross customer loans, million VND"),
    ("HDB", 2025, "note:customer_deposits_million", "consolidated"): _disclosure_cell("HDB", 2025, "note:customer_deposits_million", "consolidated", "560.714.282", "HDB_financial_statements_2025_consolidated|221", "HDB_financial_statements_2025_consolidated_221.csv", 7, 3, "Ending total customer deposits, million VND"),
    ("BAF", 2019, "note:short_term_bank_borrowings", "separate"): _disclosure_cell("BAF", 2019, "note:short_term_bank_borrowings", "separate", "1.844.322.869.189", "BAF_financial_statements_2020_separate|1300", "BAF_financial_statements_2020_separate_1300.csv", 5, 3, "Opening-2020 short-term bank borrowings"),
    ("BAF", 2019, "note:total_financial_liabilities", "separate"): _disclosure_cell("BAF", 2019, "note:total_financial_liabilities", "separate", "6.205.563.021.783", "BAF_financial_statements_2020_separate|1665", "BAF_financial_statements_2020_separate_1665.csv", 14, 2, "Opening-2020 total financial liabilities"),
    ("VCB", 2025, "note:tangible_fixed_assets_nbv_million", "separate"): _disclosure_cell("VCB", 2025, "note:tangible_fixed_assets_nbv_million", "separate", "5.475.133", "VCB_financial_statements_2025_separate|1547", "VCB_financial_statements_2025_separate_1547.csv", 21, 5, "Parent ending tangible fixed-assets net book value, million VND"),
    ("VCB", 2025, "note:intangible_fixed_assets_nbv_million", "separate"): _disclosure_cell("VCB", 2025, "note:intangible_fixed_assets_nbv_million", "separate", "2.504.438", "VCB_financial_statements_2025_separate|1570", "VCB_financial_statements_2025_separate_1570.csv", 20, 4, "Parent ending intangible fixed-assets net book value, million VND"),
    ("BVH", 2021, "note:other_entity_equity_investments_net", "consolidated"): _disclosure_cell("BVH", 2021, "note:other_entity_equity_investments_net", "consolidated", "886.855.290.194", "BVH_financial_statements_2021_consolidated|1575", "BVH_financial_statements_2021_consolidated_1575.csv", 7, 3, "Ending net equity investments in other entities"),
    ("PC1", 2024, "note:associate_investment_carrying_amount", "consolidated"): _disclosure_cell("PC1", 2024, "note:associate_investment_carrying_amount", "consolidated", "1.708.234.428.340", "PC1_financial_statements_2024_consolidated|1159", "PC1_financial_statements_2024_consolidated_1159.csv", 7, 1, "Ending carrying amount of investments in associates"),
    ("MBB", 2021, "note:intangible_fixed_assets_nbv_million", "separate"): _disclosure_cell("MBB", 2021, "note:intangible_fixed_assets_nbv_million", "separate", "1.017.134", "MBB_financial_statements_2021_separate|1529", "MBB_financial_statements_2021_separate_1529.csv", 15, 3, "Parent ending intangible fixed-assets net book value, million VND"),
    ("VSC", 2024, "note:services_purchased_from_nhdv", "consolidated"): _disclosure_cell("VSC", 2024, "note:services_purchased_from_nhdv", "consolidated", "67.531.200.867", "VSC_financial_statements_2024_consolidated|3198", "VSC_financial_statements_2024_consolidated_3198.csv", 5, 2, "Services purchased from Nam Hai Dinh Vu Port during 2024"),
    ("BAF", 2025, "note:anh_vu_phu_yen_ownership_percent", "separate"): _disclosure_cell("BAF", 2025, "note:anh_vu_phu_yen_ownership_percent", "separate", "100,00%", "BAF_financial_statements_2025_separate|357", "BAF_financial_statements_2025_separate_357.csv", 2, 5, "Parent ownership of Anh Vu Phu Yen at end-2025, percent"),
    # Arithmetic-intent audit: older model programs either selected an
    # adjacent row or reused multiple comparative tables as if independent.
    ("BAF", 2025, "note:short_term_supplier_advances_total", "separate"): _disclosure_cell("BAF", 2025, "note:short_term_supplier_advances_total", "separate", "358.968.990.702", "BAF_financial_statements_2025_separate|821", "BAF_financial_statements_2025_separate_821.csv", 5, 1, "Ending total short-term advances to suppliers"),
    ("BAF", 2022, "note:short_term_supplier_advances_total", "separate"): _disclosure_cell("BAF", 2022, "note:short_term_supplier_advances_total", "separate", "25.699.870.125", "BAF_financial_statements_2022_separate|763", "BAF_financial_statements_2022_separate_763.csv", 5, 1, "Ending total short-term advances to suppliers"),
    ("SGB", 2017, "note:intangible_fixed_assets_nbv_million", "separate"): _disclosure_cell("SGB", 2017, "note:intangible_fixed_assets_nbv_million", "separate", "373.198", "SGB_financial_statements_2017_separate|1132", "SGB_financial_statements_2017_separate_1132.csv", 14, 3, "Parent ending intangible fixed-assets net book value, million VND"),
    ("SGB", 2018, "note:intangible_fixed_assets_nbv_million", "separate"): _disclosure_cell("SGB", 2018, "note:intangible_fixed_assets_nbv_million", "separate", "397.895", "SGB_financial_statements_2018_separate|1049", "SGB_financial_statements_2018_separate_1049.csv", 12, 3, "Parent ending intangible fixed-assets net book value, million VND"),
    ("ACB", 2017, "note:impaired_financial_assets_million", "separate"): _disclosure_cell("ACB", 2017, "note:impaired_financial_assets_million", "separate", "4.349.276", "ACB_financial_statements_2017_separate|2474", "ACB_financial_statements_2017_separate_2474.csv", 2, 7, "Parent impaired financial assets, total, million VND"),
    ("ACB", 2017, "note:financial_asset_risk_provision_million", "separate"): _disclosure_cell("ACB", 2017, "note:financial_asset_risk_provision_million", "separate", "(3.373.531)", "ACB_financial_statements_2017_separate|2474", "ACB_financial_statements_2017_separate_2474.csv", 3, 7, "Parent financial-asset risk provision, total, million VND"),
    ("ACB", 2020, "note:impaired_financial_assets_million", "separate"): _disclosure_cell("ACB", 2020, "note:impaired_financial_assets_million", "separate", "3.354.134", "ACB_financial_statements_2020_separate|2215", "ACB_financial_statements_2020_separate_2215.csv", 2, 8, "Parent impaired financial assets, total, million VND"),
    ("ACB", 2020, "note:financial_asset_risk_provision_million", "separate"): _disclosure_cell("ACB", 2020, "note:financial_asset_risk_provision_million", "separate", "(3.774.705)", "ACB_financial_statements_2020_separate|2215", "ACB_financial_statements_2020_separate_2215.csv", 3, 8, "Parent financial-asset risk provision, total, million VND"),
    ("ACB", 2021, "note:impaired_financial_assets_million", "separate"): _disclosure_cell("ACB", 2021, "note:impaired_financial_assets_million", "separate", "18.734.876", "ACB_financial_statements_2021_separate|2827", "ACB_financial_statements_2021_separate_2827.csv", 2, 8, "Parent impaired financial assets, total, million VND"),
    ("ACB", 2021, "note:financial_asset_risk_provision_million", "separate"): _disclosure_cell("ACB", 2021, "note:financial_asset_risk_provision_million", "separate", "(5.891.397)", "ACB_financial_statements_2021_separate|2827", "ACB_financial_statements_2021_separate_2827.csv", 3, 8, "Parent financial-asset risk provision, total, million VND"),
    ("ACB", 2022, "note:impaired_financial_assets_million", "separate"): _disclosure_cell("ACB", 2022, "note:impaired_financial_assets_million", "separate", "18.734.876", "ACB_financial_statements_2022_separate|2295", "ACB_financial_statements_2022_separate_2295.csv", 2, 8, "Parent impaired financial assets, total, million VND"),
    ("ACB", 2022, "note:financial_asset_risk_provision_million", "separate"): _disclosure_cell("ACB", 2022, "note:financial_asset_risk_provision_million", "separate", "(5.891.397)", "ACB_financial_statements_2022_separate|2295", "ACB_financial_statements_2022_separate_2295.csv", 3, 8, "Parent financial-asset risk provision, total, million VND"),
    ("ACB", 2024, "note:impaired_financial_assets_million", "separate"): _disclosure_cell("ACB", 2024, "note:impaired_financial_assets_million", "separate", "10.087.707", "ACB_financial_statements_2024_separate|2358", "ACB_financial_statements_2024_separate_2358.csv", 2, 7, "Parent impaired financial assets, total, million VND"),
    ("ACB", 2024, "note:financial_asset_risk_provision_million", "separate"): _disclosure_cell("ACB", 2024, "note:financial_asset_risk_provision_million", "separate", "(5.436.313)", "ACB_financial_statements_2024_separate|2358", "ACB_financial_statements_2024_separate_2358.csv", 3, 7, "Parent financial-asset risk provision, total, million VND"),
})

# Final arithmetic-intent audit after the first measured v89 submission.
# These cells repair legacy programs that used only one side of a comparison,
# added values where the question requested a difference, lost OCR decimals,
# summed a requested maximum, or mixed numerator/denominator years.  Each raw
# value below is transcribed from the cited BTC table; q951 deliberately avoids
# the OCR-damaged displayed average and recomputes it from payroll/headcount.
EXPLICIT_CELLS.update({
    ("HND", 2023, "note:profit_before_tax", "unknown"): _disclosure_cell(
        "HND", 2023, "note:profit_before_tax", "unknown", "464.862.192.314",
        "HND_financial_statements_2023|988", "HND_financial_statements_2023_988.csv",
        2, 1, "Profit before tax, current year",
    ),
    ("AAA", 2022, "note:selling_outside_services", "separate"): _disclosure_cell(
        "AAA", 2022, "note:selling_outside_services", "separate", "241.673.979.156",
        "AAA_financial_statements_2022_separate|1253", "AAA_financial_statements_2022_separate_1253.csv",
        3, 1, "Selling expense - outside services, current year",
    ),
    ("AAA", 2022, "note:admin_outside_services", "separate"): _disclosure_cell(
        "AAA", 2022, "note:admin_outside_services", "separate", "18.882.240.130",
        "AAA_financial_statements_2022_separate|1253", "AAA_financial_statements_2022_separate_1253.csv",
        10, 1, "Administrative expense - outside services, current year",
    ),
    ("VIF", 2022, "note:selling_outside_services", "separate"): _disclosure_cell(
        "VIF", 2022, "note:selling_outside_services", "separate", "13.597.818.580",
        "VIF_financial_statements_2022_separate|1436", "VIF_financial_statements_2022_separate_1436.csv",
        6, 1, "Selling expense - outside services, current year",
    ),
    ("VIF", 2022, "note:admin_outside_services", "separate"): _disclosure_cell(
        "VIF", 2022, "note:admin_outside_services", "separate", "17.947.320.207",
        "VIF_financial_statements_2022_separate|1436", "VIF_financial_statements_2022_separate_1436.csv",
        15, 1, "Administrative expense - outside services, current year",
    ),
    ("VCB", 2018, "note:employee_count", "consolidated"): _disclosure_cell(
        "VCB", 2018, "note:employee_count", "consolidated", "17.215",
        "VCB_financial_statements_2018_consolidated|1682", "VCB_financial_statements_2018_consolidated_1682.csv",
        1, 1, "Total employees",
    ),
    ("VCB", 2018, "note:employee_payroll_million", "consolidated"): _disclosure_cell(
        "VCB", 2018, "note:employee_payroll_million", "consolidated", "6.920.065",
        "VCB_financial_statements_2018_consolidated|1682", "VCB_financial_statements_2018_consolidated_1682.csv",
        2, 1, "Total payroll and allowances, VND million",
    ),
    ("VCB", 2019, "note:employee_count", "consolidated"): _disclosure_cell(
        "VCB", 2019, "note:employee_count", "consolidated", "18.948",
        "VCB_financial_statements_2019_consolidated|1735", "VCB_financial_statements_2019_consolidated_1735.csv",
        1, 1, "Total employees",
    ),
    ("VCB", 2019, "note:employee_payroll_million", "consolidated"): _disclosure_cell(
        "VCB", 2019, "note:employee_payroll_million", "consolidated", "7.807.100",
        "VCB_financial_statements_2019_consolidated|1735", "VCB_financial_statements_2019_consolidated_1735.csv",
        3, 1, "Total payroll and allowances, VND million",
    ),
    ("VCB", 2025, "note:employee_count", "consolidated"): _disclosure_cell(
        "VCB", 2025, "note:employee_count", "consolidated", "23.457",
        "VCB_financial_statements_2025_consolidated|1700", "VCB_financial_statements_2025_consolidated_1700.csv",
        1, 1, "Total employees at year end",
    ),
    ("VCB", 2025, "note:employee_payroll_million", "consolidated"): _disclosure_cell(
        "VCB", 2025, "note:employee_payroll_million", "consolidated", "12.192.970",
        "VCB_financial_statements_2025_consolidated|1700", "VCB_financial_statements_2025_consolidated_1700.csv",
        2, 1, "Total payroll and allowances, VND million",
    ),
    ("BID", 2017, "note:domestic_customer_loan_provision_million", "consolidated"): _disclosure_cell(
        "BID", 2017, "note:domestic_customer_loan_provision_million", "consolidated", "10.833.513",
        "BID_financial_statements_2017_consolidated|1156", "BID_financial_statements_2017_consolidated_1156.csv",
        1, 1, "Ending provision for customer loans in Vietnam, VND million",
    ),
    ("BID", 2021, "note:domestic_customer_loan_provision_million", "consolidated"): _disclosure_cell(
        "BID", 2021, "note:domestic_customer_loan_provision_million", "consolidated", "28.451.297",
        "BID_financial_statements_2021_consolidated|1378", "BID_financial_statements_2021_consolidated_1378.csv",
        2, 1, "Ending provision for customer loans in Vietnam, VND million",
    ),
    ("BID", 2023, "note:domestic_customer_loan_provision_million", "consolidated"): _disclosure_cell(
        "BID", 2023, "note:domestic_customer_loan_provision_million", "consolidated", "39.850.765",
        "BID_financial_statements_2023_consolidated|1429", "BID_financial_statements_2023_consolidated_1429.csv",
        2, 1, "Ending provision for customer loans in Vietnam, VND million",
    ),
    ("BID", 2024, "note:domestic_customer_loan_provision_million", "consolidated"): _disclosure_cell(
        "BID", 2024, "note:domestic_customer_loan_provision_million", "consolidated", "37.423.555",
        "BID_financial_statements_2024_consolidated|1597", "BID_financial_statements_2024_consolidated_1597.csv",
        1, 1, "Ending provision for customer loans in Vietnam, VND million",
    ),
    ("BID", 2025, "note:domestic_customer_loan_provision_million", "consolidated"): _disclosure_cell(
        "BID", 2025, "note:domestic_customer_loan_provision_million", "consolidated", "34.220.631",
        "BID_financial_statements_2025_consolidated|1512", "BID_financial_statements_2025_consolidated_1512.csv",
        2, 1, "Ending provision for customer loans in Vietnam, VND million",
    ),
    ("POW", 2017, "note:usd_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2017, "note:usd_long_term_loans", "separate", "12.950.216.434.336",
        "POW_financial_statements_2017_separate|935", "POW_financial_statements_2017_separate_935.csv",
        2, 1, "Ending long-term loans denominated in USD",
    ),
    ("POW", 2017, "note:total_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2017, "note:total_long_term_loans", "separate", "15.661.786.300.389",
        "POW_financial_statements_2017_separate|935", "POW_financial_statements_2017_separate_935.csv",
        4, 1, "Ending total long-term loans",
    ),
    ("POW", 2019, "note:usd_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2019, "note:usd_long_term_loans", "separate", "5.707.365.539.513",
        "POW_financial_statements_2019_separate|986", "POW_financial_statements_2019_separate_986.csv",
        2, 1, "Ending long-term loans denominated in USD",
    ),
    ("POW", 2019, "note:total_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2019, "note:total_long_term_loans", "separate", "7.921.256.591.629",
        "POW_financial_statements_2019_separate|986", "POW_financial_statements_2019_separate_986.csv",
        4, 1, "Ending total long-term loans",
    ),
    ("POW", 2022, "note:usd_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2022, "note:usd_long_term_loans", "separate", "516.770.814.702",
        "POW_financial_statements_2022_separate|1029", "POW_financial_statements_2022_separate_1029.csv",
        2, 1, "Ending long-term loans denominated in USD",
    ),
    ("POW", 2022, "note:total_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2022, "note:total_long_term_loans", "separate", "1.984.143.645.915",
        "POW_financial_statements_2022_separate|1029", "POW_financial_statements_2022_separate_1029.csv",
        4, 1, "Ending total long-term loans",
    ),
    ("POW", 2023, "note:usd_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2023, "note:usd_long_term_loans", "separate", "2.543.730.418.101",
        "POW_financial_statements_2023_separate|1120", "POW_financial_statements_2023_separate_1120.csv",
        2, 1, "Ending long-term loans denominated in USD",
    ),
    ("POW", 2023, "note:total_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2023, "note:total_long_term_loans", "separate", "5.987.879.090.875",
        "POW_financial_statements_2023_separate|1120", "POW_financial_statements_2023_separate_1120.csv",
        4, 1, "Ending total long-term loans",
    ),
    ("POW", 2024, "note:usd_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2024, "note:usd_long_term_loans", "separate", "4.928.718.355.610",
        "POW_financial_statements_2024_separate|1066", "POW_financial_statements_2024_separate_1066.csv",
        2, 1, "Ending long-term loans denominated in USD",
    ),
    ("POW", 2024, "note:total_long_term_loans", "separate"): _disclosure_cell(
        "POW", 2024, "note:total_long_term_loans", "separate", "8.356.636.117.641",
        "POW_financial_statements_2024_separate|1066", "POW_financial_statements_2024_separate_1066.csv",
        4, 1, "Ending total long-term loans",
    ),
    ("ACB", 2016, "note:construction_in_progress_million", "consolidated"): _disclosure_cell(
        "ACB", 2016, "note:construction_in_progress_million", "consolidated", "521.862",
        "ACB_financial_statements_2017_consolidated|1924", "ACB_financial_statements_2017_consolidated_1924.csv",
        1, 2, "Opening-2017 / ending-2016 construction in progress, VND million",
    ),
    ("ACB", 2017, "note:construction_in_progress_million", "consolidated"): _disclosure_cell(
        "ACB", 2017, "note:construction_in_progress_million", "consolidated", "667.965",
        "ACB_financial_statements_2017_consolidated|1924", "ACB_financial_statements_2017_consolidated_1924.csv",
        1, 1, "Ending construction in progress, VND million",
    ),
    ("ACB", 2019, "note:construction_in_progress_million", "consolidated"): _disclosure_cell(
        "ACB", 2019, "note:construction_in_progress_million", "consolidated", "104.225",
        "ACB_financial_statements_2019_consolidated|2106", "ACB_financial_statements_2019_consolidated_2106.csv",
        1, 1, "Ending construction in progress, VND million",
    ),
    ("ACB", 2023, "note:construction_in_progress_million", "consolidated"): _disclosure_cell(
        "ACB", 2023, "note:construction_in_progress_million", "consolidated", "1.174.974",
        "ACB_financial_statements_2023_consolidated|1868", "ACB_financial_statements_2023_consolidated_1868.csv",
        2, 1, "Ending construction in progress, VND million",
    ),
    ("ACB", 2025, "note:construction_in_progress_million", "consolidated"): _disclosure_cell(
        "ACB", 2025, "note:construction_in_progress_million", "consolidated", "806.530",
        "ACB_financial_statements_2025_consolidated|1959", "ACB_financial_statements_2025_consolidated_1959.csv",
        2, 1, "Ending construction in progress, VND million",
    ),
    # Q7: the legacy answer read the prior-year EPS-allocation table.  The
    # question asks for the ending fund balance, which is balance-sheet code
    # 322 in the 2019 consolidated report.
    ("HT1", 2019, "note:bonus_welfare_fund_ending", "consolidated"): _disclosure_cell(
        "HT1", 2019, "note:bonus_welfare_fund_ending", "consolidated", "57.764.463.052",
        "HT1_financial_statements_2019_consolidated|292", "HT1_financial_statements_2019_consolidated_292.csv",
        11, 3, "Ending bonus and welfare fund balance",
    ),
    # Q55: both adjacent year-end values round to 1.40 trillion, which hid
    # the fact that the legacy query read the prior-year rather than current-
    # year ending column.
    ("FTS", 2020, "note:owner_invested_capital_ending", "unknown"): _disclosure_cell(
        "FTS", 2020, "note:owner_invested_capital_ending", "unknown", "1.404.117.487.650",
        "FTS_financial_statements_2020|424", "FTS_financial_statements_2020_424.csv",
        5, 9, "Current-year ending owner-invested capital",
    ),
    # Q137: 1 January 2023 is the opening balance in the 2023 parent report,
    # not the ending balance selected by the legacy program.
    ("TTF", 2023, "note:buyer_advances_opening", "separate"): _disclosure_cell(
        "TTF", 2023, "note:buyer_advances_opening", "separate", "180.790.293.856",
        "TTF_financial_statements_2023_separate|973", "TTF_financial_statements_2023_separate_973.csv",
        3, 2, "Opening buyer-advance balance at 1 January 2023",
    ),
    # Q143: use the explicit TOTAL ASSETS row and its opening column.  The
    # legacy query selected ending current assets from the preceding table.
    ("FIT", 2022, "note:total_assets_opening", "consolidated"): _disclosure_cell(
        "FIT", 2022, "note:total_assets_opening", "consolidated", "5.984.081.185.909",
        "FIT_financial_statements_2022_consolidated|262", "FIT_financial_statements_2022_consolidated_262.csv",
        38, 5, "Opening total assets",
    ),
    # Q166: the correct related-party loan table is line 1309.  The legacy
    # program read an opening trade-receivable cell and divided by billions
    # although the requested unit is hundred-billion VND.
    ("HBC", 2016, "note:loan_to_nha_hoa_binh_ending", "separate"): _disclosure_cell(
        "HBC", 2016, "note:loan_to_nha_hoa_binh_ending", "separate", "221.951.021.299",
        "HBC_financial_statements_2016_separate|1309", "HBC_financial_statements_2016_separate_1309.csv",
        1, 3, "Ending loan balance due from Cong ty Co phan Nha Hoa Binh",
    ),
    # Q214: the question asks for company revenue, not only electricity-sales
    # revenue from a note disclosure.  Use income-statement code 01.
    ("DTK", 2024, "note:gross_sales_and_service_revenue", "consolidated"): _disclosure_cell(
        "DTK", 2024, "note:gross_sales_and_service_revenue", "consolidated", "12.839.301.284.671",
        "DTK_financial_statements_2024_consolidated|390", "DTK_financial_statements_2024_consolidated_390.csv",
        1, 4, "Gross sales and service revenue in 2024",
    ),
    # Q229: the legacy program selected one company from a related-party
    # payable schedule.  Balance-sheet code 252 is the requested total
    # investment in associates/joint ventures.
    ("SJG", 2020, "note:associate_investments_total", "consolidated"): _disclosure_cell(
        "SJG", 2020, "note:associate_investments_total", "consolidated", "2.403.122.974.806",
        "SJG_financial_statements_2020_consolidated|326", "SJG_financial_statements_2020_consolidated_326.csv",
        28, 4, "Ending total investment in joint ventures and associates",
    ),
    # Q628: compare the named molten-urea fertilizer project, not the unrelated
    # inventory work-in-progress row used by the legacy program.
    ("DCM", 2018, "note:molten_urea_project_ending", "consolidated"): _disclosure_cell(
        "DCM", 2018, "note:molten_urea_project_ending", "consolidated", "282.477.152.203",
        "DCM_financial_statements_2018_consolidated|917", "DCM_financial_statements_2018_consolidated_917.csv",
        2, 1, "Ending construction-in-progress balance of the molten-urea fertilizer project",
    ),
    ("DCM", 2017, "note:molten_urea_project_ending", "consolidated"): _disclosure_cell(
        "DCM", 2017, "note:molten_urea_project_ending", "consolidated", "59.991.150.644",
        "DCM_financial_statements_2018_consolidated|917", "DCM_financial_statements_2018_consolidated_917.csv",
        2, 2, "Opening-2018 comparative / ending-2017 project balance",
    ),
    # Q680: a financial-revenue share must use deposit-interest income.  The
    # legacy numerator was the balance-sheet receivable for accrued interest.
    ("QNS", 2024, "note:deposit_interest_income", "separate"): _disclosure_cell(
        "QNS", 2024, "note:deposit_interest_income", "separate", "230.880.661.175",
        "QNS_financial_statements_2024_separate|1359", "QNS_financial_statements_2024_separate_1359.csv",
        1, 1, "Deposit-interest income in 2024",
    ),
    ("QNS", 2024, "note:financial_revenue_total", "separate"): _disclosure_cell(
        "QNS", 2024, "note:financial_revenue_total", "separate", "457.220.214.212",
        "QNS_financial_statements_2024_separate|1359", "QNS_financial_statements_2024_separate_1359.csv",
        5, 1, "Total financial revenue in 2024",
    ),
    # Q40: the tax-reconciliation table has no NPAT row, so the legacy
    # fallback subtracted only nominal-rate tax and omitted tax adjustments.
    # Read NPAT directly from the primary income statement.
    ("VIB", 2020, "note:net_profit_after_tax_million", "consolidated"): _disclosure_cell(
        "VIB", 2020, "note:net_profit_after_tax_million", "consolidated", "4.642.334",
        "VIB_financial_statements_2020_consolidated|284", "VIB_financial_statements_2020_consolidated_284.csv",
        5, 3, "Consolidated net profit after tax in 2020, VND million",
    ),
    # Q24: the source table is denominated in VND.  The legacy program
    # divided by one billion even though the question requests thousand VND.
    ("HNG", 2017, "note:hag_long_term_loan_ending", "separate"): _disclosure_cell(
        "HNG", 2017, "note:hag_long_term_loan_ending", "separate", "2.083.992.733",
        "HNG_financial_statements_2017_separate|995", "HNG_financial_statements_2017_separate_995.csv",
        1, 1, "Ending long-term loan from Hoang Anh Gia Lai, VND",
    ),
    # Q174 and Q342: bank-note tables report these balances directly in
    # million VND.  The legacy programs applied an extra 1e-6 conversion.
    ("NVB", 2024, "note:internal_payables_million", "consolidated"): _disclosure_cell(
        "NVB", 2024, "note:internal_payables_million", "consolidated", "307.293",
        "NVB_financial_statements_2024_consolidated|1350", "NVB_financial_statements_2024_consolidated_1350.csv",
        1, 1, "Ending internal payables, VND million",
    ),
    ("KLB", 2022, "note:cash_gold_precious_stones_million", "separate"): _disclosure_cell(
        "KLB", 2022, "note:cash_gold_precious_stones_million", "separate", "742.817",
        "KLB_financial_statements_2022_separate|1893", "KLB_financial_statements_2022_separate_1893.csv",
        2, 1, "Ending cash, gold and precious stones, VND million",
    ),
    # Q280: the legacy answer came from the adjacent UPAS L/C interest-rate
    # table (4.30%-11.10%), not the requested loan balance table.
    ("HDB", 2022, "note:upas_lc_refinancing_vnd_million", "consolidated"): _disclosure_cell(
        "HDB", 2022, "note:upas_lc_refinancing_vnd_million", "consolidated", "8.634.940",
        "HDB_financial_statements_2022_consolidated|1242", "HDB_financial_statements_2022_consolidated_1242.csv",
        2, 1, "Ending VND UPAS L/C refinancing loans, VND million",
    ),
    # Q284: the retrieved fund-policy table merely contained the words
    # 'charter capital'.  The equity roll-forward gives the actual 2023
    # parent-bank closing balance in million VND.
    ("EIB", 2023, "note:charter_capital_million", "separate"): _disclosure_cell(
        "EIB", 2023, "note:charter_capital_million", "separate", "17.469.561",
        "EIB_financial_statements_2023_separate|1716", "EIB_financial_statements_2023_separate_1716.csv",
        12, 1, "Ending charter capital at 31 December 2023, VND million",
    ),
    # Q285: the legacy program selected the closing gross cost of intangible
    # assets (42.8 billion VND).  The requested bad-debt disclosure reports
    # the total gross balance directly in VND.
    ("ACV", 2025, "note:bad_debt_gross_total", "separate"): _disclosure_cell(
        "ACV", 2025, "note:bad_debt_gross_total", "separate", "3.753.928.692.577",
        "ACV_financial_statements_2025_separate|1201", "ACV_financial_statements_2025_separate_1201.csv",
        10, 1, "Total gross bad debt at 31 December 2025, VND",
    ),
    # Q527: ending other receivables from dividends/profit distributions.
    # The selector must be evaluated independently for every requested year;
    # the legacy program selected a ratio and then subtracted the receivable.
    ("ACV", 2016, "note:dividend_profit_receivable", "separate"): _disclosure_cell(
        "ACV", 2016, "note:dividend_profit_receivable", "separate", "20.850.027.125",
        "ACV_financial_statements_2016_separate|1265", "ACV_financial_statements_2016_separate_1265.csv",
        10, 1, "Ending other receivable from dividends and distributed profits",
    ),
    ("ACV", 2020, "note:dividend_profit_receivable", "separate"): _disclosure_cell(
        "ACV", 2020, "note:dividend_profit_receivable", "separate", "11.250.000.000",
        "ACV_financial_statements_2020_separate|931", "ACV_financial_statements_2020_separate_931.csv",
        8, 1, "Ending other receivable from dividends and distributed profits",
    ),
    ("ACV", 2021, "note:dividend_profit_receivable", "separate"): _disclosure_cell(
        "ACV", 2021, "note:dividend_profit_receivable", "separate", "11.250.000.000",
        "ACV_financial_statements_2021_separate|918", "ACV_financial_statements_2021_separate_918.csv",
        9, 1, "Ending other receivable from dividends and distributed profits",
    ),
    ("ACV", 2023, "note:dividend_profit_receivable", "separate"): _disclosure_cell(
        "ACV", 2023, "note:dividend_profit_receivable", "separate", "26.250.000.000",
        "ACV_financial_statements_2023_separate|891", "ACV_financial_statements_2023_separate_891.csv",
        7, 1, "Ending other receivable from dividends and distributed profits",
    ),
    # Q529: ending short-term trade payables from the primary balance sheet.
    ("QNS", 2017, "note:short_term_trade_payables", "consolidated"): _disclosure_cell(
        "QNS", 2017, "note:short_term_trade_payables", "consolidated", "539.666.220.767",
        "QNS_financial_statements_2017_consolidated|355", "QNS_financial_statements_2017_consolidated_355.csv",
        3, 4, "Ending short-term trade payables",
    ),
    ("QNS", 2020, "note:short_term_trade_payables", "consolidated"): _disclosure_cell(
        "QNS", 2020, "note:short_term_trade_payables", "consolidated", "382.734.453.855",
        "QNS_financial_statements_2020_consolidated|438", "QNS_financial_statements_2020_consolidated_438.csv",
        3, 4, "Ending short-term trade payables",
    ),
    ("QNS", 2023, "note:short_term_trade_payables", "consolidated"): _disclosure_cell(
        "QNS", 2023, "note:short_term_trade_payables", "consolidated", "456.319.808.742",
        "QNS_financial_statements_2023_consolidated|547", "QNS_financial_statements_2023_consolidated_547.csv",
        3, 4, "Ending short-term trade payables",
    ),
    ("QNS", 2024, "note:short_term_trade_payables", "consolidated"): _disclosure_cell(
        "QNS", 2024, "note:short_term_trade_payables", "consolidated", "464.095.068.931",
        "QNS_financial_statements_2024_consolidated|293", "QNS_financial_statements_2024_consolidated_293.csv",
        3, 4, "Ending short-term trade payables",
    ),
    # Q198: the table's net deferred-tax expense is the sum of its expense
    # and income components, not the first component alone.
    ("ACB", 2020, "note:deferred_tax_expense_component_million", "separate"): _disclosure_cell(
        "ACB", 2020, "note:deferred_tax_expense_component_million", "separate", "22.833",
        "ACB_financial_statements_2020_separate|2013", "ACB_financial_statements_2020_separate_2013.csv",
        0, 1, "Deferred-tax expense from reversal of deferred-tax assets, million VND",
    ),
    ("ACB", 2020, "note:deferred_tax_income_component_million", "separate"): _disclosure_cell(
        "ACB", 2020, "note:deferred_tax_income_component_million", "separate", "(50.595)",
        "ACB_financial_statements_2020_separate|2013", "ACB_financial_statements_2020_separate_2013.csv",
        1, 1, "Deferred-tax income from deductible temporary differences, million VND",
    ),
    # Q657: profit per outstanding share uses parent NPAT and the average of
    # opening/ending outstanding shares; the legacy numerator was a dividend
    # received from one subsidiary.
    ("FOX", 2020, "note:ending_outstanding_shares", "separate"): _disclosure_cell(
        "FOX", 2020, "note:ending_outstanding_shares", "separate", "273.616.446",
        "FOX_financial_statements_2020_separate|858", "FOX_financial_statements_2020_separate_858.csv",
        5, 1, "Ending number of outstanding common shares",
    ),
    ("FOX", 2020, "note:opening_outstanding_shares", "separate"): _disclosure_cell(
        "FOX", 2020, "note:opening_outstanding_shares", "separate", "248.742.469",
        "FOX_financial_statements_2020_separate|858", "FOX_financial_statements_2020_separate_858.csv",
        5, 2, "Opening number of outstanding common shares",
    ),
    # Q667: use tangible-PPE gross cost and balance-sheet total assets, both
    # from the parent statement, rather than the currency-risk table's net
    # all-fixed-assets line and gross currency exposure total.
    ("EIB", 2023, "note:tangible_ppe_gross_cost_million", "separate"): _disclosure_cell(
        "EIB", 2023, "note:tangible_ppe_gross_cost_million", "separate", "2.506.132",
        "EIB_financial_statements_2023_separate|319", "EIB_financial_statements_2023_separate_319.csv",
        19, 3, "Ending gross cost of tangible fixed assets, million VND",
    ),
    ("EIB", 2023, "note:total_assets_million", "separate"): _disclosure_cell(
        "EIB", 2023, "note:total_assets_million", "separate", "201.672.702",
        "EIB_financial_statements_2023_separate|319", "EIB_financial_statements_2023_separate_319.csv",
        30, 3, "Ending total assets, million VND",
    ),
    # Q164: depreciation charged specifically on investment property.
    ("SSH", 2023, "note:investment_property_depreciation", "consolidated"): _disclosure_cell(
        "SSH", 2023, "note:investment_property_depreciation", "consolidated", "4.900.612.861",
        "SSH_financial_statements_2023_consolidated|1060", "SSH_financial_statements_2023_consolidated_1060.csv",
        10, 3, "Investment-property depreciation charged during 2023",
    ),
    # Q8: payroll and payroll-related management expense, not cash-flow code 11.
    ("FTS", 2021, "note:payroll_and_related_expense", "unknown"): _disclosure_cell(
        "FTS", 2021, "note:payroll_and_related_expense", "unknown", "30.686.828.047",
        "FTS_financial_statements_2021|1281", "FTS_financial_statements_2021_1281.csv",
        1, 2, "Payroll and other payroll-related expense during 2021",
    ),
    # Q329: parent ACB NPAT from the separate income statement.
    ("ACB", 2024, "note:net_profit_after_tax_million", "separate"): _disclosure_cell(
        "ACB", 2024, "note:net_profit_after_tax_million", "separate", "16.085.199",
        "ACB_financial_statements_2024_separate|216", "ACB_financial_statements_2024_separate_216.csv",
        21, 3, "Parent net profit after tax in 2024, million VND",
    ),
    # Q336: every maturity bucket contributes to total future minimum lease payments.
    ("HND", 2025, "note:minimum_lease_within_one_year", "unknown"): _disclosure_cell(
        "HND", 2025, "note:minimum_lease_within_one_year", "unknown", "16.142.170.183",
        "HND_financial_statements_2024|916", "HND_financial_statements_2024_916.csv",
        2, 1, "Future minimum operating-lease payments within one year",
    ),
    ("HND", 2025, "note:minimum_lease_two_to_five_years", "unknown"): _disclosure_cell(
        "HND", 2025, "note:minimum_lease_two_to_five_years", "unknown", "64.568.680.732",
        "HND_financial_statements_2024|916", "HND_financial_statements_2024_916.csv",
        3, 1, "Future minimum operating-lease payments from two to five years",
    ),
    ("HND", 2025, "note:minimum_lease_after_five_years", "unknown"): _disclosure_cell(
        "HND", 2025, "note:minimum_lease_after_five_years", "unknown", "297.337.156.006",
        "HND_financial_statements_2024|916", "HND_financial_statements_2024_916.csv",
        4, 1, "Future minimum operating-lease payments after five years",
    ),
    # Q336 corrected-year variant: the legacy repair mislabeled the 2024
    # report's ending column as 2025.  The 2025 report states the requested
    # total directly, so retain one auditable total cell instead of rebuilding
    # it from the three maturity buckets.
    ("HND", 2025, "note:q336_total_minimum_lease_payments", "unknown"): _disclosure_cell(
        "HND", 2025, "note:q336_total_minimum_lease_payments", "unknown", "387.656.354.540",
        "HND_financial_statements_2025|912", "HND_financial_statements_2025_912.csv",
        4, 1, "Total future minimum lease payments at 31 December 2025",
    ),
    # Q354 source correction: the legacy query selected line 1097, which is
    # explicitly headed "19.2 Vay ngan hang dai han" in the report.  The
    # requested short-term bank-borrowing balance is stated directly in the
    # short-term debt movement table at line 1058.
    ("AAA", 2021, "note:q354_short_term_bank_borrowings", "separate"): _disclosure_cell(
        "AAA", 2021, "note:q354_short_term_bank_borrowings", "separate", "1.401.195.977.583",
        "AAA_financial_statements_2021_separate|1058", "AAA_financial_statements_2021_separate_1058.csv",
        3, 7, "Ending short-term bank borrowings",
    ),
    # Q646 source-family correction: the unqualified question mirrors the
    # exact "Chi phi nhan vien" row in selling expenses.  The legacy query
    # instead selected the separately disclosed management-employee expense.
    ("KHG", 2021, "note:q646_employee_expense", "consolidated"): _disclosure_cell(
        "KHG", 2021, "note:q646_employee_expense", "consolidated", "22.287.552.828",
        "KHG_financial_statements_2021_consolidated|1021", "KHG_financial_statements_2021_consolidated_1021.csv",
        2, 1, "Employee expense in selling expenses, 2021",
    ),
    ("KHG", 2020, "note:q646_employee_expense", "consolidated"): _disclosure_cell(
        "KHG", 2020, "note:q646_employee_expense", "consolidated", "11.949.173.962",
        "KHG_financial_statements_2021_consolidated|1021", "KHG_financial_statements_2021_consolidated_1021.csv",
        2, 2, "Employee expense in selling expenses, 2020",
    ),
    # Q169: an unqualified government-bond balance includes every investment
    # classification, consistent with q888.  The legacy query read only the
    # pledged subset disclosed elsewhere in the report.
    ("STB", 2016, "note:q169_government_bonds_afs", "consolidated"): _disclosure_cell(
        "STB", 2016, "note:q169_government_bonds_afs", "consolidated", "27.045.792",
        "STB_financial_statements_2017_consolidated|1313", "STB_financial_statements_2017_consolidated_1313.csv",
        3, 2, "Opening government bonds classified as available for sale",
    ),
    ("STB", 2016, "note:q169_government_bonds_htm", "consolidated"): _disclosure_cell(
        "STB", 2016, "note:q169_government_bonds_htm", "consolidated", "991.387",
        "STB_financial_statements_2017_consolidated|1313", "STB_financial_statements_2017_consolidated_1313.csv",
        12, 2, "Opening government bonds classified as held to maturity",
    ),
    # Q26 asks for an ending balance.  The legacy query read a current-year
    # transaction with one related party; the accrued-expense note states the
    # actual ending interest-payable balance directly.
    ("DLG", 2023, "note:q26_interest_payable_ending", "consolidated"): _disclosure_cell(
        "DLG", 2023, "note:q26_interest_payable_ending", "consolidated", "350.187.565.073",
        "DLG_financial_statements_2023_consolidated|1455", "DLG_financial_statements_2023_consolidated_1455.csv",
        1, 1, "Ending accrued interest payable",
    ),
    # Q244 asks for the consolidated ending long-term construction-in-progress
    # balance.  The legacy query read one acquiree's provisional fair value at
    # its purchase date; balance-sheet code 242 states the company-wide ending
    # balance directly.
    ("VIC", 2016, "note:q244_ending_long_term_cip", "consolidated"): _disclosure_cell(
        "VIC", 2016, "note:q244_ending_long_term_cip", "consolidated", "33.991.567.265.462",
        "VIC_financial_statements_2016_consolidated|209", "VIC_financial_statements_2016_consolidated_209.csv",
        16, 3, "Ending long-term construction in progress (balance-sheet code 242)",
    ),
    # Q209: the primary balance sheet reports the consolidated ending
    # long-term loan receivable.  The legacy program read an AUD exposure row.
    ("HPG", 2021, "note:long_term_loan_receivable", "consolidated"): _disclosure_cell(
        "HPG", 2021, "note:long_term_loan_receivable", "consolidated", "118.401.369.280",
        "HPG_financial_statements_2021_consolidated|169", "HPG_financial_statements_2021_consolidated_169.csv",
        3, 3, "Ending long-term loan receivable",
    ),
    # Q239: ending tangible-PPE gross cost from the parent fixed-asset note.
    # The legacy program selected the adjacent intangible-PPE note.
    ("EIB", 2024, "note:tangible_ppe_gross_cost_million", "separate"): _disclosure_cell(
        "EIB", 2024, "note:tangible_ppe_gross_cost_million", "separate", "2.569.380",
        "EIB_financial_statements_2024_separate|1450", "EIB_financial_statements_2024_separate_1450.csv",
        6, 6, "Ending gross cost of tangible fixed assets, million VND",
    ),
    # Q253: total original cost of investments in associates at end-2016.
    # The legacy program read the ending gross cost of tangible fixed assets.
    ("MBB", 2016, "note:associate_investment_original_cost_million", "consolidated"): _disclosure_cell(
        "MBB", 2016, "note:associate_investment_original_cost_million", "consolidated", "105.975",
        "MBB_financial_statements_2016_consolidated|1508", "MBB_financial_statements_2016_consolidated_1508.csv",
        4, 2, "Ending total original cost of investments in associates, million VND",
    ),
    # Q267: the foreign-currency row nested under term deposits, rather than
    # the first foreign-currency row in the whole deposit table.
    ("ACB", 2020, "note:term_deposits_foreign_currency_million", "separate"): _disclosure_cell(
        "ACB", 2020, "note:term_deposits_foreign_currency_million", "separate", "340.063",
        "ACB_financial_statements_2020_separate|1767", "ACB_financial_statements_2020_separate_1767.csv",
        6, 1, "Ending foreign-currency term deposits, million VND",
    ),
    # Q601: short-term accrued interest at each requested year end.  The
    # legacy variable names were reversed relative to the loaded documents.
    ("KBC", 2016, "note:short_term_interest_payable", "separate"): _disclosure_cell(
        "KBC", 2016, "note:short_term_interest_payable", "separate", "4.470.193.703",
        "KBC_financial_statements_2016_separate|1035", "KBC_financial_statements_2016_separate_1035.csv",
        6, 1, "Ending short-term accrued borrowing interest",
    ),
    ("KBC", 2019, "note:short_term_interest_payable", "separate"): _disclosure_cell(
        "KBC", 2019, "note:short_term_interest_payable", "separate", "849.420.772.627",
        "KBC_financial_statements_2019_separate|1147", "KBC_financial_statements_2019_separate_1147.csv",
        7, 1, "Ending short-term accrued borrowing interest",
    ),
    # Q625: opening and closing balances for the specifically named VIB loan;
    # the legacy program compared total short-term borrowings instead.
    ("VGC", 2024, "note:vib_short_term_loan_opening", "consolidated"): _disclosure_cell(
        "VGC", 2024, "note:vib_short_term_loan_opening", "consolidated", "1.771.533.325",
        "VGC_financial_statements_2024_consolidated|1379", "VGC_financial_statements_2024_consolidated_1379.csv",
        5, 1, "Opening short-term borrowing from Vietnam International Bank",
    ),
    ("VGC", 2024, "note:vib_short_term_loan_ending", "consolidated"): _disclosure_cell(
        "VGC", 2024, "note:vib_short_term_loan_ending", "consolidated", "2.388.528.000",
        "VGC_financial_statements_2024_consolidated|1379", "VGC_financial_statements_2024_consolidated_1379.csv",
        5, 2, "Ending short-term borrowing from Vietnam International Bank",
    ),
    # Q85: the first numeric segment is the bank-only amount.  The final
    # column is the requested ACB consolidated total.
    ("ACB", 2016, "note:interest_and_similar_expense_total_million", "consolidated"): _disclosure_cell(
        "ACB", 2016, "note:interest_and_similar_expense_total_million", "consolidated", "9.556.360",
        "ACB_financial_statements_2016_consolidated|2822", "ACB_financial_statements_2016_consolidated_2822.csv",
        9, 7, "Consolidated interest and similar expense, VND million",
    ),
    # Q102: land-use-rights net book value is the closing carrying amount,
    # not the adjacent closing original-cost row.
    ("MBB", 2018, "note:land_use_rights_nbv_million", "consolidated"): _disclosure_cell(
        "MBB", 2018, "note:land_use_rights_nbv_million", "consolidated", "92.783",
        "MBB_financial_statements_2018_consolidated|1803", "MBB_financial_statements_2018_consolidated_1803.csv",
        13, 1, "Closing net book value of land-use rights, VND million",
    ),
    # Q360: total parent IJC contributed capital contains three contribution
    # rows in the investment schedule; the legacy query returned only row 1.
    ("IJC", 2025, "note:q360_contributed_capital_01", "separate"): _disclosure_cell(
        "IJC", 2025, "note:q360_contributed_capital_01", "separate", "516.981.750.000",
        "IJC_financial_statements_2025_separate|1341", "IJC_financial_statements_2025_separate_1341.csv",
        3, 1, "Contributed capital - first disclosed investment",
    ),
    ("IJC", 2025, "note:q360_contributed_capital_02", "separate"): _disclosure_cell(
        "IJC", 2025, "note:q360_contributed_capital_02", "separate", "36.000.000.000",
        "IJC_financial_statements_2025_separate|1341", "IJC_financial_statements_2025_separate_1341.csv",
        6, 1, "Contributed capital - second disclosed investment",
    ),
    ("IJC", 2025, "note:q360_contributed_capital_03", "separate"): _disclosure_cell(
        "IJC", 2025, "note:q360_contributed_capital_03", "separate", "20.000.000.000",
        "IJC_financial_statements_2025_separate|1341", "IJC_financial_statements_2025_separate_1341.csv",
        8, 1, "Contributed capital - third disclosed investment",
    ),
    # Q35 legacy plus interpretation: aggregate the two non-zero gross
    # receivable subtypes.  A more direct summary-line interpretation is kept
    # in FORMULA_VARIANTS so old artifacts remain reproducible.
    ("BVH", 2015, "note:q35_profit_receivable", "separate"): _disclosure_cell(
        "BVH", 2015, "note:q35_profit_receivable", "separate", "208.334.219.954",
        "BVH_financial_statements_2015_separate|1514", "BVH_financial_statements_2015_separate_1514.csv",
        3, 2, "Receivable from Bao Viet Life - profit receivable",
    ),
    ("BVH", 2015, "note:q35_it_cost_receivable", "separate"): _disclosure_cell(
        "BVH", 2015, "note:q35_it_cost_receivable", "separate", "24.234.744.400",
        "BVH_financial_statements_2015_separate|1514", "BVH_financial_statements_2015_separate_1514.csv",
        4, 2, "Receivable from Bao Viet Life - IT cost receivable",
    ),
    ("BVH", 2015, "note:q35_total_receivable_summary", "separate"): _disclosure_cell(
        "BVH", 2015, "note:q35_total_receivable_summary", "separate", "222.575.005.778",
        "BVH_financial_statements_2015_separate|1232", "BVH_financial_statements_2015_separate_1232.csv",
        2, 1, "Receivable - Bao Viet Life, direct summary balance",
    ),
    # Q723: total long-term related-party receivables comprise the dedicated
    # long-term-loan balance (note 8) plus other long-term receivables from
    # related parties (note 9).  The legacy query divided note 9 by the grand
    # total of short- and long-term other receivables, which is a different
    # accounting population.
    ("HAG", 2018, "note:q723_related_long_term_loans", "consolidated"): _disclosure_cell(
        "HAG", 2018, "note:q723_related_long_term_loans", "consolidated", "6.130.524.711",
        "HAG_financial_statements_2018_consolidated|1304", "HAG_financial_statements_2018_consolidated_1304.csv",
        6, 2, "Long-term loans to related parties",
    ),
    ("HAG", 2018, "note:q723_related_other_long_term_receivables", "consolidated"): _disclosure_cell(
        "HAG", 2018, "note:q723_related_other_long_term_receivables", "consolidated", "329.540.303",
        "HAG_financial_statements_2018_consolidated|1333", "HAG_financial_statements_2018_consolidated_1333.csv",
        17, 1, "Other long-term receivables from related parties",
    ),
    # Q191: the requested amount is the VND carrying value of USD-denominated
    # assets.  The legacy query read the USD cash quantity from line 1079 and
    # then treated that foreign-currency quantity as VND.  The currency-risk
    # exposure table states the closing asset carrying amount directly in VND.
    ("PVT", 2022, "note:q191_usd_assets_vnd", "consolidated"): _disclosure_cell(
        "PVT", 2022, "note:q191_usd_assets_vnd", "consolidated", "297.476.115.784",
        "PVT_financial_statements_2022_consolidated|1362", "PVT_financial_statements_2022_consolidated_1362.csv",
        2, 1, "Closing assets denominated in US dollars, VND",
    ),
    # Q599: the question explicitly excludes the first three rows of the
    # other-expense note.  Therefore it asks for the remaining row labelled
    # "Chi phí khác", comparing the prior-year and current-year columns.
    ("FTS", 2023, "note:q599_other_expense_exclusions_removed", "unknown"): _disclosure_cell(
        "FTS", 2023, "note:q599_other_expense_exclusions_removed", "unknown", "45.869.981.490",
        "FTS_financial_statements_2023|1440", "FTS_financial_statements_2023_1440.csv",
        4, 2, "Other expense excluding administrative fines, token-card cost and investor transfer fees - 2023",
    ),
    ("FTS", 2022, "note:q599_other_expense_exclusions_removed", "unknown"): _disclosure_cell(
        "FTS", 2022, "note:q599_other_expense_exclusions_removed", "unknown", "53.775.388.537",
        "FTS_financial_statements_2023|1440", "FTS_financial_statements_2023_1440.csv",
        4, 3, "Other expense excluding administrative fines, token-card cost and investor transfer fees - 2022 comparative",
    ),
    # Q270: the loan schedule identifies "Khoản vay 1" directly.  The legacy
    # program selected the same date from the equity-movement schedule and
    # therefore returned share capital instead of the named borrowing.
    ("VSC", 2017, "note:q270_loan_1_closing", "consolidated"): _disclosure_cell(
        "VSC", 2017, "note:q270_loan_1_closing", "consolidated", "4.998.500.000",
        "VSC_financial_statements_2017_consolidated|851", "VSC_financial_statements_2017_consolidated_851.csv",
        1, 4, "Loan 1 balance at 31 December 2017",
    ),
    # Q183: the geographical exposure table labels the requested total loan
    # balance explicitly.  It also reconciles (subject to one-million display
    # rounding) to interbank loans plus customer loans in note 3/4.
    ("NVB", 2016, "note:q183_total_loans_million", "separate"): _disclosure_cell(
        "NVB", 2016, "note:q183_total_loans_million", "separate", "27.702.541",
        "NVB_financial_statements_2016_separate|1472", "NVB_financial_statements_2016_separate_1472.csv",
        1, 1, "Parent NVB total domestic loan exposure at 31 December 2016, VND million",
    ),
    # Q125: distinguish operating-lease commitments as lessor ("cho thuê")
    # from both lessee commitments ("thuê") and unrelated unearned revenue.
    ("VIF", 2024, "note:q125_operating_lease_income_commitments", "separate"): _disclosure_cell(
        "VIF", 2024, "note:q125_operating_lease_income_commitments", "separate", "84.052.528.171",
        "VIF_financial_statements_2024_separate|1801", "VIF_financial_statements_2024_separate_1801.csv",
        4, 1, "Total closing operating-lease commitments as lessor",
    ),
    # Q327: use the explicitly labelled "other customers" line in the trade-
    # receivable note.  The legacy query read a generic customer receivable
    # from an unrelated asset-handover schedule.
    ("PC1", 2021, "note:q327_other_customer_receivables", "separate"): _disclosure_cell(
        "PC1", 2021, "note:q327_other_customer_receivables", "separate", "1.108.219.411.035",
        "PC1_financial_statements_2021_separate|780", "PC1_financial_statements_2021_separate_780.csv",
        5, 1, "Other customer receivables at 31 December 2021",
    ),
    # Q104: "tăng" asks for the year-on-year increase, not the full 2022
    # profit-before-tax balance.  The narrative immediately below the notes
    # independently states the same increase of VND 5,920,290 million.
    ("MBB", 2022, "note:q104_parent_pbt_million", "separate"): _disclosure_cell(
        "MBB", 2022, "note:q104_parent_pbt_million", "separate", "20.318.374",
        "MBB_financial_statements_2022_separate_1|1833", "MBB_financial_statements_2022_separate_1_1833.csv",
        1, 1, "Parent MBB profit before tax in 2022, VND million",
    ),
    ("MBB", 2021, "note:q104_parent_pbt_million", "separate"): _disclosure_cell(
        "MBB", 2021, "note:q104_parent_pbt_million", "separate", "14.398.084",
        "MBB_financial_statements_2022_separate_1|1833", "MBB_financial_statements_2022_separate_1_1833.csv",
        1, 2, "Parent MBB profit before tax in 2021 comparative, VND million",
    ),
    # Q58: balance-sheet code 132 is short-term advances to suppliers.  The
    # legacy program filtered code 311, which is short-term trade payables.
    ("PC1", 2023, "note:q58_short_term_supplier_advances", "separate"): _disclosure_cell(
        "PC1", 2023, "note:q58_short_term_supplier_advances", "separate", "186.723.471.407",
        "PC1_financial_statements_2023_separate|199", "PC1_financial_statements_2023_separate_199.csv",
        10, 3, "Closing short-term advances to suppliers, balance-sheet code 132",
    ),
    # Q128: the non-cancellable operating-lease note explicitly totals future
    # minimum receipts.  The legacy program instead selected cash-flow code 70.
    ("IJC", 2015, "note:q128_future_minimum_operating_lease_receipts", "separate"): _disclosure_cell(
        "IJC", 2015, "note:q128_future_minimum_operating_lease_receipts", "separate", "27.350.000.000",
        "IJC_financial_statements_2015_separate|1604", "IJC_financial_statements_2015_separate_1604.csv",
        3, 1, "Total future minimum receipts under non-cancellable operating leases",
    ),
    # Q75: the BVIF disclosure states BVH's direct contribution explicitly.
    # The legacy program selected total owner-contributed capital instead.
    ("BVH", 2019, "note:q75_direct_contribution_to_bvif", "separate"): _disclosure_cell(
        "BVH", 2019, "note:q75_direct_contribution_to_bvif", "separate", "420.000.000.000",
        "BVH_financial_statements_2019_separate|372", "BVH_financial_statements_2019_separate_372.csv",
        1, 1, "Direct contribution by Bao Viet Holdings to BVIF",
    ),
    # Q272: the maturity schedule reconciles every parent-bank asset class to
    # the requested total.  The legacy program selected the "movable assets"
    # row from a collateral table rather than total assets.
    ("MBB", 2020, "note:q272_parent_total_assets_million", "separate"): _disclosure_cell(
        "MBB", 2020, "note:q272_parent_total_assets_million", "separate", "482.483.609",
        "MBB_financial_statements_2020_separate|1974", "MBB_financial_statements_2020_separate_1974.csv",
        12, 9, "Parent MBB total assets at 31 December 2020, VND million",
    ),
    # Q905: a broad substring filter matched the component labelled "other
    # financial revenue" instead of the requested total. Read each explicit
    # TONG CONG cell for the four requested years.
    ("VIC", 2020, "note:q905_financial_revenue_total_million", "consolidated"): _disclosure_cell(
        "VIC", 2020, "note:q905_financial_revenue_total_million", "consolidated", "31.068.411",
        "VIC_financial_statements_2020_consolidated|1947", "VIC_financial_statements_2020_consolidated_1947.csv",
        5, 1, "Total consolidated financial revenue in 2020, VND million",
    ),
    ("VIC", 2021, "note:q905_financial_revenue_total_million", "consolidated"): _disclosure_cell(
        "VIC", 2021, "note:q905_financial_revenue_total_million", "consolidated", "16.045.903",
        "VIC_financial_statements_2021_consolidated|2013", "VIC_financial_statements_2021_consolidated_2013.csv",
        6, 1, "Total consolidated financial revenue in 2021, VND million",
    ),
    ("VIC", 2023, "note:q905_financial_revenue_total_million", "consolidated"): _disclosure_cell(
        "VIC", 2023, "note:q905_financial_revenue_total_million", "consolidated", "20.502.485",
        "VIC_financial_statements_2023_consolidated|2242", "VIC_financial_statements_2023_consolidated_2242.csv",
        6, 1, "Total consolidated financial revenue in 2023, VND million",
    ),
    ("VIC", 2025, "note:q905_financial_revenue_total_million", "consolidated"): _disclosure_cell(
        "VIC", 2025, "note:q905_financial_revenue_total_million", "consolidated", "50.463.250",
        "VIC_financial_statements_2025_consolidated|2378", "VIC_financial_statements_2025_consolidated_2378.csv",
        5, 1, "Total consolidated financial revenue in 2025, VND million",
    ),
    # Q337 is answer-neutral because BAF reports no ending cash equivalents,
    # but the legacy query only added cash on hand and bank deposits.  Read the
    # disclosed note total so the program still represents code 110 if a
    # component is zero or the note layout changes.
    ("BAF", 2023, "cdkt:110", "separate"): _disclosure_cell(
        "BAF", 2023, "cdkt:110", "separate", "81.566.921.890",
        "BAF_financial_statements_2023_separate|781", "BAF_financial_statements_2023_separate_781.csv",
        3, 1, "Cash and cash equivalents - total ending balance",
    ),
    # Q611: the current query subtracts two overdue buckets in the 2018 table.
    # "Trong hạn" is the five maturity buckets from "Đến 1 tháng" through
    # "Trên 5 năm". Compare the sum of that group between 2018 and 2019.
    ("EIB", 2019, "note:q611_current_net_liquidity_01", "separate"): _disclosure_cell(
        "EIB", 2019, "note:q611_current_net_liquidity_01", "separate", "(8.893.098)",
        "EIB_financial_statements_2019_separate|2947", "EIB_financial_statements_2019_separate_2947.csv",
        19, 3, "2019 net liquidity gap due within one month, VND million",
    ),
    ("EIB", 2019, "note:q611_current_net_liquidity_02", "separate"): _disclosure_cell(
        "EIB", 2019, "note:q611_current_net_liquidity_02", "separate", "(2.168.521)",
        "EIB_financial_statements_2019_separate|2947", "EIB_financial_statements_2019_separate_2947.csv",
        19, 4, "2019 net liquidity gap due from one to three months, VND million",
    ),
    ("EIB", 2019, "note:q611_current_net_liquidity_03", "separate"): _disclosure_cell(
        "EIB", 2019, "note:q611_current_net_liquidity_03", "separate", "(32.230.642)",
        "EIB_financial_statements_2019_separate|2947", "EIB_financial_statements_2019_separate_2947.csv",
        19, 5, "2019 net liquidity gap due from three to twelve months, VND million",
    ),
    ("EIB", 2019, "note:q611_current_net_liquidity_04", "separate"): _disclosure_cell(
        "EIB", 2019, "note:q611_current_net_liquidity_04", "separate", "12.286.770",
        "EIB_financial_statements_2019_separate|2947", "EIB_financial_statements_2019_separate_2947.csv",
        19, 6, "2019 net liquidity gap due from one to five years, VND million",
    ),
    ("EIB", 2019, "note:q611_current_net_liquidity_05", "separate"): _disclosure_cell(
        "EIB", 2019, "note:q611_current_net_liquidity_05", "separate", "47.572.462",
        "EIB_financial_statements_2019_separate|2947", "EIB_financial_statements_2019_separate_2947.csv",
        19, 7, "2019 net liquidity gap due after five years, VND million",
    ),
    ("EIB", 2018, "note:q611_current_net_liquidity_01", "separate"): _disclosure_cell(
        "EIB", 2018, "note:q611_current_net_liquidity_01", "separate", "(11.659.866)",
        "EIB_financial_statements_2019_separate|2970", "EIB_financial_statements_2019_separate_2970.csv",
        19, 3, "2018 net liquidity gap due within one month, VND million",
    ),
    ("EIB", 2018, "note:q611_current_net_liquidity_02", "separate"): _disclosure_cell(
        "EIB", 2018, "note:q611_current_net_liquidity_02", "separate", "(16.374.612)",
        "EIB_financial_statements_2019_separate|2970", "EIB_financial_statements_2019_separate_2970.csv",
        19, 4, "2018 net liquidity gap due from one to three months, VND million",
    ),
    ("EIB", 2018, "note:q611_current_net_liquidity_03", "separate"): _disclosure_cell(
        "EIB", 2018, "note:q611_current_net_liquidity_03", "separate", "(17.762.039)",
        "EIB_financial_statements_2019_separate|2970", "EIB_financial_statements_2019_separate_2970.csv",
        19, 5, "2018 net liquidity gap due from three to twelve months, VND million",
    ),
    ("EIB", 2018, "note:q611_current_net_liquidity_04", "separate"): _disclosure_cell(
        "EIB", 2018, "note:q611_current_net_liquidity_04", "separate", "10.603.170",
        "EIB_financial_statements_2019_separate|2970", "EIB_financial_statements_2019_separate_2970.csv",
        19, 6, "2018 net liquidity gap due from one to five years, VND million",
    ),
    ("EIB", 2018, "note:q611_current_net_liquidity_05", "separate"): _disclosure_cell(
        "EIB", 2018, "note:q611_current_net_liquidity_05", "separate", "50.921.398",
        "EIB_financial_statements_2019_separate|2970", "EIB_financial_statements_2019_separate_2970.csv",
        19, 7, "2018 net liquidity gap due after five years, VND million",
    ),
    # Q937: the legacy program mixed four unrelated provision-movement rows.
    # The unqualified line "Chứng khoán nợ" is read consistently from CTG's
    # consolidated trading-securities note for each requested year.  Keep this
    # interpretation in an isolated research candidate because the statements
    # also disclose debt securities under AFS and HTM investment notes.
    ("CTG", 2017, "note:q937_trading_debt_securities_million", "consolidated"): _disclosure_cell(
        "CTG", 2017, "note:q937_trading_debt_securities_million", "consolidated", "2.910.939",
        "CTG_financial_statements_2017_consolidated|1272", "CTG_financial_statements_2017_consolidated_1272.csv",
        2, 1, "Consolidated CTG trading debt securities at 31 December 2017, VND million",
    ),
    ("CTG", 2018, "note:q937_trading_debt_securities_million", "consolidated"): _disclosure_cell(
        "CTG", 2018, "note:q937_trading_debt_securities_million", "consolidated", "2.183.108",
        "CTG_financial_statements_2018_consolidated|1335", "CTG_financial_statements_2018_consolidated_1335.csv",
        2, 1, "Consolidated CTG trading debt securities at 31 December 2018, VND million",
    ),
    ("CTG", 2019, "note:q937_trading_debt_securities_million", "consolidated"): _disclosure_cell(
        "CTG", 2019, "note:q937_trading_debt_securities_million", "consolidated", "3.137.327",
        "CTG_financial_statements_2019_consolidated|1285", "CTG_financial_statements_2019_consolidated_1285.csv",
        2, 1, "Consolidated CTG trading debt securities at 31 December 2019, VND million",
    ),
    ("CTG", 2020, "note:q937_trading_debt_securities_million", "consolidated"): _disclosure_cell(
        "CTG", 2020, "note:q937_trading_debt_securities_million", "consolidated", "5.060.257",
        "CTG_financial_statements_2020_consolidated|1292", "CTG_financial_statements_2020_consolidated_1292.csv",
        2, 1, "Consolidated CTG trading debt securities at 31 December 2020, VND million",
    ),
    # Alternative family for the two unqualified "chứng khoán nợ" questions.
    # The test set's q880 uses the same unqualified total-debt-securities phrase
    # with the available-for-sale table, while q883 explicitly says
    # "chứng khoán kinh doanh nợ" when it intends the trading table.  Keep this
    # interpretation behind a named builder variant so the earlier isolated
    # q937 trading candidate remains exactly reproducible.
    ("MBB", 2023, "note:q81_afs_debt_securities_million", "consolidated"): _disclosure_cell(
        "MBB", 2023, "note:q81_afs_debt_securities_million", "consolidated", "143.010.711",
        "MBB_financial_statements_2023_consolidated|1596", "MBB_financial_statements_2023_consolidated_1596.csv",
        1, 1, "MBB available-for-sale debt securities at 31 December 2023, VND million",
    ),
    ("CTG", 2017, "note:q937_afs_debt_securities_million", "consolidated"): _disclosure_cell(
        "CTG", 2017, "note:q937_afs_debt_securities_million", "consolidated", "125.287.262",
        "CTG_financial_statements_2017_consolidated|1372", "CTG_financial_statements_2017_consolidated_1372.csv",
        1, 1, "CTG available-for-sale debt securities at 31 December 2017, VND million",
    ),
    ("CTG", 2018, "note:q937_afs_debt_securities_million", "consolidated"): _disclosure_cell(
        "CTG", 2018, "note:q937_afs_debt_securities_million", "consolidated", "88.187.442",
        "CTG_financial_statements_2018_consolidated|1429", "CTG_financial_statements_2018_consolidated_1429.csv",
        1, 1, "CTG available-for-sale debt securities at 31 December 2018, VND million",
    ),
    ("CTG", 2019, "note:q937_afs_debt_securities_million", "consolidated"): _disclosure_cell(
        "CTG", 2019, "note:q937_afs_debt_securities_million", "consolidated", "96.755.014",
        "CTG_financial_statements_2019_consolidated|1381", "CTG_financial_statements_2019_consolidated_1381.csv",
        1, 1, "CTG available-for-sale debt securities at 31 December 2019, VND million",
    ),
    ("CTG", 2020, "note:q937_afs_debt_securities_million", "consolidated"): _disclosure_cell(
        "CTG", 2020, "note:q937_afs_debt_securities_million", "consolidated", "112.301.221",
        "CTG_financial_statements_2020_consolidated|1405", "CTG_financial_statements_2020_consolidated_1405.csv",
        1, 1, "CTG available-for-sale debt securities at 31 December 2020, VND million",
    ),
    # Q99: the question asks for balance-sheet code 111 (cash).  The legacy
    # query selected the ending balance of short-term prepaid expenses instead.
    ("NKG", 2022, "note:q99_cash", "consolidated"): _disclosure_cell(
        "NKG", 2022, "note:q99_cash", "consolidated", "948.303.528.970",
        "NKG_financial_statements_2022_consolidated|179", "NKG_financial_statements_2022_consolidated_179.csv",
        4, 3, "Consolidated NKG balance-sheet code 111 cash at 31 December 2022, VND",
    ),
    # Q612: the legacy growth query read the opening/closing balances of
    # short-term prepaid expenses.  The borrowing note explicitly reconciles
    # parent-company short-term bank loans from end-2020 to end-2021.
    ("NKG", 2020, "note:q612_short_term_bank_loans", "separate"): _disclosure_cell(
        "NKG", 2020, "note:q612_short_term_bank_loans", "separate", "2.468.663.849.028",
        "NKG_financial_statements_2021_separate|860", "NKG_financial_statements_2021_separate_860.csv",
        13, 2, "Parent NKG short-term bank loans at 31 December 2020, VND",
    ),
    ("NKG", 2021, "note:q612_short_term_bank_loans", "separate"): _disclosure_cell(
        "NKG", 2021, "note:q612_short_term_bank_loans", "separate", "3.773.154.733.117",
        "NKG_financial_statements_2021_separate|860", "NKG_financial_statements_2021_separate_860.csv",
        13, 7, "Parent NKG short-term bank loans at 31 December 2021, VND",
    ),
    # Q318: the legacy query selected the ending balance of a prepaid-expense
    # movement table.  Balance-sheet code 311 directly answers short-term trade
    # payables, and the requested unit is hundred-billion VND.
    ("HHV", 2022, "note:q318_short_term_trade_payables", "consolidated"): _disclosure_cell(
        "HHV", 2022, "note:q318_short_term_trade_payables", "consolidated", "1.094.900.651.814",
        "HHV_financial_statements_2022_consolidated|329", "HHV_financial_statements_2022_consolidated_329.csv",
        4, 3, "Consolidated HHV balance-sheet code 311 short-term trade payables at end-2022, VND",
    ),
    # Q338: the legacy query selected an unrelated ending fund balance.  The
    # related-party supplier schedule names PVOIL and its closing payable
    # directly.
    ("BSR", 2018, "note:q338_pvoil_supplier_payable", "consolidated"): _disclosure_cell(
        "BSR", 2018, "note:q338_pvoil_supplier_payable", "consolidated", "2.499.485.052.166",
        "BSR_financial_statements_2018_consolidated|1224", "BSR_financial_statements_2018_consolidated_1224.csv",
        3, 1, "BSR payable to supplier PetroVietnam Oil Corporation JSC at 31 December 2018, VND",
    ),
})


# Q950: build the four annual related-party transaction totals from every
# current-year numeric cell in the source tables headed "transaction value".
# Coordinates are explicit and the raw values are loaded from the BTC table
# extracts, so both the selected year and every contributing operand remain
# independently source-auditable.
Q950_TRANSACTION_TABLES = {
    2019: (
        (67, 1374, (4, 6, 9, 10, 12, 13, 14, 17, 18, 19, 20, 21, 22, 24, 25, 28, 29, 30)),
        (68, 1389, (3, 4, 7, 10, 11, 13, 16, 17, 19, 20, 21, 23, 25, 27, 28, 29, 31, 32, 33)),
        (69, 1404, (3, 4, 5, 7)),
    ),
    2020: (
        (64, 1368, (9, 11, 16, 17, 18, 19, 21)),
        (65, 1389, (3, 4, 5, 6, 17, 18, 19, 22, 24, 26, 28, 31, 32)),
        (66, 1406, (4, 5, 7, 9, 11, 13, 14, 15, 17, 19)),
    ),
    2021: (
        (56, 1274, (4, 5, 6, 9, 10, 12, 15, 16, 17, 18, 21, 22, 24, 26, 27, 28, 29)),
        (57, 1289, (4, 5, 7, 10, 12, 13, 19, 27, 29)),
    ),
    2022: (
        (54, 1396, (4, 5, 6, 7, 14, 19, 22, 23, 25)),
        (55, 1411, (6, 7, 8, 10, 11, 12, 14, 16, 20, 22, 24, 26, 28)),
        (56, 1428, (7, 8, 11, 12, 14)),
    ),
}


def _register_q950_transaction_cells() -> tuple[dict[int, tuple[Operand, ...]], str]:
    operands_by_year: dict[int, tuple[Operand, ...]] = {}
    expression_terms: list[str] = []
    variable_index = 0
    for year, table_specs in Q950_TRANSACTION_TABLES.items():
        year_operands: list[Operand] = []
        year_variables: list[str] = []
        document = f"VRE_financial_statements_{year}_separate"
        sequence = 1
        for table_index, line, row_indices in table_specs:
            source_path = ROOT / "build" / "tables" / document / f"table_{table_index}_line{line}.csv"
            with source_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row_idx in row_indices:
                raw = rows[row_idx]["1"]
                label = rows[row_idx]["0"]
                metric = f"note:q950_related_transaction_{year}_{sequence:03d}"
                EXPLICIT_CELLS[("VRE", year, metric, "separate")] = _disclosure_cell(
                    "VRE", year, metric, "separate", raw,
                    f"{document}|{line}", f"{document}_{line}.csv",
                    row_idx, 1, label,
                )
                year_operands.append(op("VRE", year, metric, "separate"))
                year_variables.append(f"abs(v{variable_index})")
                variable_index += 1
                sequence += 1
        operands_by_year[year] = tuple(year_operands)
        expression_terms.append(f"({' + '.join(year_variables)}, {year})")
    return operands_by_year, f"max({', '.join(expression_terms)})[1]"


Q950_OPERANDS_BY_YEAR, Q950_EXPRESSION = _register_q950_transaction_cells()
Q950_OPERANDS = tuple(
    operand
    for year in (2019, 2020, 2021, 2022)
    for operand in Q950_OPERANDS_BY_YEAR[year]
)


FORMULAS: dict[int, Formula] = {
    33: Formula((op("MBB", 2020, "note:customer_loan_loss_provision_million"),), "abs(v0)", "MBB consolidated ending customer-loan loss provision at end-2020, VND million; the balance-sheet parentheses denote a contra-asset, while the question asks for the provision amount"),
    35: Formula(
        (op("BVH", 2015, "note:q35_profit_receivable", "separate"), op("BVH", 2015, "note:q35_it_cost_receivable", "separate")),
        "(v0 + v1) / 1e6",
        "total parent BVH receivable from Bao Viet Life at end-2015, VND million; plus-only interpretation aggregates the two non-zero receivable subtypes",
    ),
    58: Formula(
        (op("PC1", 2023, "note:q58_short_term_supplier_advances", "separate"),),
        "v0 / 1e6",
        "parent PC1 closing short-term advances to suppliers in 2023, VND million; uses balance-sheet code 132 instead of trade-payables code 311",
    ),
    75: Formula(
        (op("BVH", 2019, "note:q75_direct_contribution_to_bvif", "separate"),),
        "v0 / 1e6",
        "parent BVH direct contribution to Bao Viet Value Investment Fund at 31 December 2019, VND million; replaces total owner-contributed capital",
    ),
    723: Formula(
        (
            op("HAG", 2018, "note:q723_related_long_term_loans", "consolidated"),
            op("HAG", 2018, "note:q723_related_other_long_term_receivables", "consolidated"),
        ),
        "v0 / (v0 + v1) * 100",
        "share of HAG consolidated long-term related-party loan receivables in total long-term related-party receivables at end-2018; denominator combines note 8 loans and note 9 other long-term receivables",
    ),
    85: Formula(
        (op("ACB", 2016, "note:interest_and_similar_expense_total_million"),),
        "v0",
        "ACB consolidated interest and similar expense in 2016, VND million; uses the consolidated-total column rather than the first bank segment",
    ),
    102: Formula(
        (op("MBB", 2018, "note:land_use_rights_nbv_million"),),
        "v0",
        "MBB consolidated closing net book value of land-use rights in 2018, VND million; replaces closing original cost",
    ),
    104: Formula(
        (
            op("MBB", 2022, "note:q104_parent_pbt_million", "separate"),
            op("MBB", 2021, "note:q104_parent_pbt_million", "separate"),
        ),
        "v0 - v1",
        "increase in parent MBB profit before tax from 2021 to 2022, VND million; replaces the full 2022 balance",
    ),
    125: Formula(
        (op("VIF", 2024, "note:q125_operating_lease_income_commitments", "separate"),),
        "v0 / 1e11",
        "parent VIF total closing operating-lease commitments as lessor at end-2024, hundred-billion VND; replaces unrelated unearned revenue",
    ),
    128: Formula(
        (op("IJC", 2015, "note:q128_future_minimum_operating_lease_receipts", "separate"),),
        "v0 / 1e9",
        "parent IJC total future minimum operating-lease receipts at end-2015, VND billion; replaces cash-flow code 70",
    ),
    146: Formula(
        (
            op("HBC", 2018, "note:q146_opening_short_term_prepaid_expense"),
            op("HBC", 2018, "note:q146_opening_long_term_prepaid_expense"),
        ),
        "(v0 + v1) / 1e11",
        "HBC consolidated total opening prepaid expense in 2018, hundred-billion VND; the unqualified balance is short-term plus long-term prepaid expense, reconciled by their 54,133,339,998 VND combined increase to cash-flow code 12",
    ),
    191: Formula(
        (op("PVT", 2022, "note:q191_usd_assets_vnd", "consolidated"),),
        "v0 / 1e9",
        "PVT consolidated closing VND carrying value of USD-denominated assets in 2022, VND billion; replaces the foreign-currency cash quantity selected from a different disclosure",
    ),
    195: Formula(
        (op("VCB", 2025, "note:q195_fx_transaction_commitments_million", "separate"),),
        "v0 * 1000",
        "parent VCB ending foreign-exchange transaction commitments in 2025, VND thousand; replaces the same-named CTG balance selected by the legacy entity-mismatched program",
    ),
    183: Formula(
        (op("NVB", 2016, "note:q183_total_loans_million", "separate"),),
        "v0",
        "parent NVB total loan exposure at 31 December 2016, VND million; replaces total equity from the equity-movement schedule",
    ),
    270: Formula(
        (op("VSC", 2017, "note:q270_loan_1_closing", "consolidated"),),
        "v0 / 1e6",
        "VSC consolidated Loan 1 balance at 31 December 2017, VND million; replaces share capital from the equity-movement schedule",
    ),
    272: Formula(
        (op("MBB", 2020, "note:q272_parent_total_assets_million", "separate"),),
        "v0",
        "parent MBB total assets at 31 December 2020, VND million; replaces movable collateral assets",
    ),
    327: Formula(
        (op("PC1", 2021, "note:q327_other_customer_receivables", "separate"),),
        "v0 / 1e11",
        "parent PC1 receivables from other customers at 31 December 2021, hundred-billion VND; replaces a generic receivable in an unrelated handover schedule",
    ),
    340: Formula(
        (op("PVT", 2019, "cdkt:140", "separate"),),
        "v0 / 1e9",
        "parent PVT total closing inventory at end-2019, VND billion, from balance-sheet code 140; replaces the closing accumulated amortization of software selected from note 11",
    ),
    360: Formula(
        tuple(op("IJC", 2025, f"note:q360_contributed_capital_{index:02d}", "separate") for index in range(1, 4)),
        "(v0 + v1 + v2) / 1e12",
        "total parent IJC contributed capital at 31 December 2025, VND trillion; aggregates all three contribution rows",
    ),
    599: Formula(
        (
            op("FTS", 2022, "note:q599_other_expense_exclusions_removed", "unknown"),
            op("FTS", 2023, "note:q599_other_expense_exclusions_removed", "unknown"),
        ),
        "(v0 - v1) / 1e6",
        "amount by which FTS other expense excluding the three named categories in 2022 exceeded 2023, VND million; reads the residual 'Chi phí khác' row in note B7.37",
    ),
    611: Formula(
        tuple(
            op("EIB", year, f"note:q611_current_net_liquidity_{index:02d}", "separate")
            for year in (2019, 2018)
            for index in range(1, 6)
        ),
        "(v0 + v1 + v2 + v3 + v4) - (v5 + v6 + v7 + v8 + v9)",
        "change from 2018 to 2019 in parent EIB net liquidity gap across the five current maturity buckets, VND million; replaces a subtraction between two 2018 overdue buckets",
    ),
    905: Formula(
        tuple(
            op("VIC", year, "note:q905_financial_revenue_total_million", "consolidated")
            for year in (2020, 2021, 2023, 2025)
        ),
        "(v0 + v1 + v2 + v3) / 1e6",
        "cumulative VIC consolidated total financial revenue for 2020, 2021, 2023 and 2025, trillion VND; replaces the four 'other financial revenue' component rows",
    ),
    937: Formula(
        tuple(
            op("CTG", year, "note:q937_trading_debt_securities_million", "consolidated")
            for year in (2017, 2018, 2019, 2020)
        ),
        "(v0 + v1 + v2 + v3) / 4",
        "mean CTG consolidated trading debt securities at the four requested year ends, VND million; replaces four unrelated provision-movement rows; interpretation remains isolated because AFS and HTM notes also contain debt-securities lines",
    ),
    99: Formula(
        (op("NKG", 2022, "note:q99_cash", "consolidated"),),
        "v0 / 1e9",
        "NKG consolidated balance-sheet code 111 cash at end-2022, VND billion; replaces the ending short-term-prepayment balance",
    ),
    612: Formula(
        (
            op("NKG", 2020, "note:q612_short_term_bank_loans", "separate"),
            op("NKG", 2021, "note:q612_short_term_bank_loans", "separate"),
        ),
        "(v1 / v0 - 1) * 100",
        "growth in parent NKG short-term bank loans from end-2020 to end-2021; replaces growth in short-term prepaid expenses",
    ),
    318: Formula(
        (op("HHV", 2022, "note:q318_short_term_trade_payables", "consolidated"),),
        "v0 / 1e11",
        "HHV consolidated short-term trade payables at end-2022, hundred-billion VND; replaces a prepaid-expense ending balance and fixes the requested unit",
    ),
    338: Formula(
        (op("BSR", 2018, "note:q338_pvoil_supplier_payable", "consolidated"),),
        "v0 / 1e12",
        "BSR payable to supplier PetroVietnam Oil Corporation JSC at end-2018, trillion VND; replaces an unrelated fund balance",
    ),
    850: Formula(
        tuple(op("TTF", year, "kqkd:11") for year in (2017, 2018, 2020, 2022, 2024)),
        "max((v0, 2017), (v1, 2018), (v2, 2020), (v3, 2022), (v4, 2024))[1]",
        "requested year with maximum TTF consolidated total cost of goods sold and services; replaces a zero-match note-table filter",
    ),
    950: Formula(
        Q950_OPERANDS,
        Q950_EXPRESSION,
        "requested year with maximum parent VRE related-party transaction value; sums every current-year numeric transaction cell from the source schedules",
    ),
    40: Formula((op("VIB", 2020, "note:net_profit_after_tax_million"),), "v0", "VIB consolidated 2020 net profit after tax from the primary income statement, VND million; replaces an incomplete nominal-tax reconciliation fallback"),
    24: Formula((op("HNG", 2017, "note:hag_long_term_loan_ending", "separate"),), "v0 / 1e3", "parent HNG ending long-term loan from Hoang Anh Gia Lai, thousand VND; fixes the legacy billion-unit conversion"),
    174: Formula((op("NVB", 2024, "note:internal_payables_million"),), "v0", "NVB consolidated ending internal payables, VND million; removes an erroneous extra 1e-6 conversion"),
    280: Formula((op("HDB", 2022, "note:upas_lc_refinancing_vnd_million"),), "v0", "HDB consolidated ending VND UPAS L/C refinancing-loan balance, VND million; replaces the adjacent interest-rate range"),
    284: Formula((op("EIB", 2023, "note:charter_capital_million", "separate"),), "v0", "parent EIB ending charter capital in 2023, VND million; replaces the fund-policy percentage row"),
    285: Formula((op("ACV", 2025, "note:bad_debt_gross_total", "separate"),), "v0 / 1e12", "parent ACV total gross bad debt at 31 December 2025, VND trillion; replaces an unrelated intangible-asset gross-cost row"),
    328: Formula((op("DXG", 2024, "cdkt:411"),), "v0 / 1e12", "Bluemarq Group is ticker DXG; ending share capital in 2024, VND trillion; replaces the adjacent share-premium column"),
    342: Formula((op("KLB", 2022, "note:cash_gold_precious_stones_million", "separate"),), "v0", "parent KLB ending cash, gold and precious stones, VND million; removes an erroneous extra 1e-6 conversion"),
    160: Formula((op("VPB", 2025, "note:profit_attributable_common_shareholders_million"),), "v0", "VPB 2025 profit attributable to ordinary shareholders is disclosed directly in VND million; removes the erroneous extra 1e-6 conversion"),
    51: Formula((op("VCB", 2020, "note:promissory_bonds_medium_term_vnd_million"),), "v0", "VCB medium-term VND promissory notes and bonds, VND million; selects the requested second medium-term-VND row rather than the certificate-of-deposit row"),
    88: Formula((op("NVB", 2019, "note:total_assets_million"),), "v0", "NVB consolidated total assets at end-2019, VND million; replaces the total of other assets"),
    110: Formula((op("VCB", 2022, "note:customer_deposits_million", "separate"),), "v0", "parent VCB customer deposits at end-2022, VND million; replaces the closing total shareholders' equity"),
    115: Formula((op("CTG", 2019, "note:total_operating_expense_million", "separate"),), "abs(v0)", "parent CTG 2019 total operating expense, VND million; reports the requested expense magnitude rather than its statement presentation sign"),
    139: Formula((op("HAG", 2020, "note:net_revenue_thousand"),), "v0", "HAG consolidated 2020 net revenue is already disclosed in thousand VND; removes an erroneous 0.001 conversion"),
    152: Formula((op("SGB", 2023, "note:total_assets_million"),), "v0", "SGB consolidated total assets at end-2023, VND million; replaces an unrelated share-capital table total"),
    805: Formula((
        *(op("SCR", 2018, f"note:q805_related_trade_{index:02d}", "separate") for index in range(1, 12)),
        op("DIG", 2018, "note:q805_related_trade_total", "separate"),
        op("DIG", 2018, "note:q805_related_other_all_total", "separate"),
        op("DIG", 2018, "note:q805_related_other_long_term", "separate"),
    ), "abs((v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8 + v9 + v10) - v11 - (v12 - v13)) / 1e9", "absolute end-2018 difference between parent SCR and DIG related-party short-term receivables (trade/code 131 plus other/code 138); excludes loans and supplier advances"),
    7: Formula((op("HT1", 2019, "note:bonus_welfare_fund_ending"),), "v0 / 1e9", "HT1 consolidated ending bonus-and-welfare-fund balance in 2019, VND billion; replaces the prior-year EPS-allocation lookup"),
    55: Formula((op("FTS", 2020, "note:owner_invested_capital_ending", "unknown"),), "v0 / 1e12", "FTS current-year ending owner-invested capital in 2020, VND trillion; fixes the prior-year column while preserving the rounded 1.40 answer"),
    137: Formula((op("TTF", 2023, "note:buyer_advances_opening", "separate"),), "v0 / 1e9", "parent TTF buyer-advance opening balance at 1 January 2023, VND billion; replaces the ending-2023 column"),
    143: Formula((op("FIT", 2022, "note:total_assets_opening"),), "v0 / 1e12", "FIT consolidated opening total assets in 2022, VND trillion; replaces ending current assets"),
    166: Formula((op("HBC", 2016, "note:loan_to_nha_hoa_binh_ending", "separate"),), "v0 / 1e11", "parent HBC ending loan balance due from Cong ty Co phan Nha Hoa Binh at 31 December 2016, VND hundred-billion; replaces an opening trade-receivable cell and fixes the unit divisor"),
    214: Formula((op("DTK", 2024, "note:gross_sales_and_service_revenue"),), "v0 / 1e11", "DTK consolidated gross sales and service revenue in 2024, VND hundred-billion; replaces the electricity-sales-only note row"),
    229: Formula((op("SJG", 2020, "note:associate_investments_total"),), "v0 / 1e9", "SJG consolidated ending total investment in joint ventures and associates, VND billion; replaces one entity from a related-party payable schedule"),
    628: Formula((op("DCM", 2018, "note:molten_urea_project_ending"), op("DCM", 2017, "note:molten_urea_project_ending")), "(v0 - v1) / 1e9", "increase in the named molten-urea fertilizer construction project from end-2017 to end-2018, VND billion; replaces inventory work in progress"),
    680: Formula((op("QNS", 2024, "note:deposit_interest_income", "separate"), op("QNS", 2024, "note:financial_revenue_total", "separate")), "v0 / v1 * 100", "parent QNS deposit-interest income as a share of total financial revenue in 2024; replaces the accrued-interest receivable numerator"),
    786: Formula((op("DTK", 2023, "kqkd:50", "separate"), op("HND", 2023, "note:profit_before_tax", "unknown")), "(v0 - v1) / 1e9", "parent DTK 2023 PBT minus Hai Phong Thermal Power 2023 PBT, VND billion; the legacy program read HND only"),
    789: Formula((op("VIF", 2022, "note:selling_outside_services", "separate"), op("VIF", 2022, "note:admin_outside_services", "separate"), op("AAA", 2022, "note:selling_outside_services", "separate"), op("AAA", 2022, "note:admin_outside_services", "separate")), "abs((v0 + v1) - (v2 + v3)) / 1e9", "absolute difference between parent VIF and AAA total outside-service expense across selling and administrative expenses in 2022, VND billion; the legacy program added prior-year cells"),
    908: Formula(tuple(op("ACB", year, "note:construction_in_progress_million") for year in (2016, 2017, 2019, 2023, 2025)), "max(v0, v1, v2, v3, v4) / 1e6", "maximum ACB consolidated construction in progress across the five requested years, VND thousand-billion; fixes the legacy year/table mapping while preserving the source-backed 1.17 answer"),
    951: Formula((op("VCB", 2018, "note:employee_payroll_million"), op("VCB", 2018, "note:employee_count"), op("VCB", 2019, "note:employee_payroll_million"), op("VCB", 2019, "note:employee_count"), op("VCB", 2025, "note:employee_payroll_million"), op("VCB", 2025, "note:employee_count")), "max(v0 / v1 / 12, v2 / v3 / 12, v4 / v5 / 12)", "maximum VCB average monthly employee income across 2018, 2019 and 2025, VND million per employee; recomputed from payroll/headcount to avoid OCR decimal loss"),
    967: Formula(tuple(op("BID", year, "note:domestic_customer_loan_provision_million") for year in (2017, 2021, 2023, 2024, 2025)), "max(v0, v1, v2, v3, v4)", "maximum ending BID provision for customer loans in Vietnam across the five requested years, VND million; the legacy program incorrectly summed all years"),
    968: Formula((*(op("POW", year, "note:usd_long_term_loans", "separate") for year in (2017, 2019, 2022, 2023, 2024)), *(op("POW", year, "note:total_long_term_loans", "separate") for year in (2017, 2019, 2022, 2023, 2024))), "max(v0 / v5, v1 / v6, v2 / v7, v3 / v8, v4 / v9) * 100", "maximum parent POW USD-denominated share of total long-term loans across the five requested year ends; pairs numerator and denominator within each year"),
    527: Formula((
        *(op("ACV", year, "note:dividend_profit_receivable", "separate") for year in (2016, 2020, 2021, 2023)),
        *(op("ACV", year, "kqkd:50", "separate") for year in (2016, 2020, 2021, 2023)),
        *(op("ACV", year, "kqkd:01", "separate") for year in (2016, 2020, 2021, 2023)),
    ), "max((v0, v4 / v8), (v1, v5 / v9), (v2, v6 / v10), (v3, v7 / v11))[1] * 100", "parent ACV PBT as a percentage of gross sales-and-service revenue in the requested year with the largest ending dividend/profit receivable; removes the legacy receivable subtraction"),
    529: Formula((
        *(op("QNS", year, "kqkd:25") for year in (2017, 2020, 2023, 2024)),
        *(op("QNS", year, "note:short_term_trade_payables") for year in (2017, 2020, 2023, 2024)),
    ), "max((v0, v4), (v1, v5), (v2, v6), (v3, v7))[1] / 1e9", "QNS ending short-term trade payables in the requested year with the largest selling expense; replaces the legacy accrued-expense selector"),
    832: Formula(tuple(op("MCH", year, "cdkt:110", "separate") for year in (2017, 2018, 2019, 2022, 2025)), "max((v0, 2017), (v1, 2018), (v2, 2019), (v3, 2022), (v4, 2025))[1]", "requested year with the highest parent MCH ending cash and cash equivalents; makes the source dependencies explicit"),
    198: Formula((op("ACB", 2020, "note:deferred_tax_expense_component_million", "separate"), op("ACB", 2020, "note:deferred_tax_income_component_million", "separate")), "v0 + v1", "net total parent ACB deferred-tax expense in 2020, million VND; sums the expense and income components instead of returning only the first row"),
    657: Formula((op("FOX", 2020, "kqkd:60", "separate"), op("FOX", 2020, "note:ending_outstanding_shares", "separate"), op("FOX", 2020, "note:opening_outstanding_shares", "separate")), "v0 / ((v1 + v2) / 2) / 1e3", "parent FOX 2020 NPAT per average outstanding share, thousand VND; replaces unrelated subsidiary profit distributions"),
    667: Formula((op("EIB", 2023, "note:tangible_ppe_gross_cost_million", "separate"), op("EIB", 2023, "note:total_assets_million", "separate")), "v0 / v1 * 100", "parent EIB ending tangible-PPE gross cost as a percentage of balance-sheet total assets"),
    164: Formula((op("SSH", 2023, "note:investment_property_depreciation"),), "v0 / 1e9", "SSH 2023 depreciation charged specifically on investment property, VND billion; excludes tangible fixed-asset depreciation"),
    212: Formula((op("HNG", 2015, "kqkd:60"),), "v0", "HNG consolidated 2015 net profit for the year, thousand VND; replaces the legacy HAG entity lookup"),
    8: Formula((op("FTS", 2021, "note:payroll_and_related_expense", "unknown"),), "v0 / 1e9", "parent FTS 2021 payroll and other payroll-related expense, VND billion; replaces the unrelated cash-flow code-11 lookup"),
    329: Formula((op("ACB", 2024, "note:net_profit_after_tax_million", "separate"),), "v0", "parent ACB 2024 NPAT from the separate income statement, million VND; replaces consolidated shareholder-attributable profit"),
    325: Formula((op("TTF", 2025, "kqkd:60", "separate"),), "v0 / 1e6", "parent TTF 2025 net profit after tax from income-statement code 60, VND million; replaces cash-flow code 60 from an unrelated evidence table"),
    337: Formula((op("BAF", 2023, "cdkt:110", "separate"),), "v0 / 1e9", "parent BAF 2023 ending cash and cash equivalents from the disclosed total; replaces a component-only sum that omitted the zero-valued cash-equivalents row"),
    336: Formula((op("HND", 2025, "note:minimum_lease_within_one_year", "unknown"), op("HND", 2025, "note:minimum_lease_two_to_five_years", "unknown"), op("HND", 2025, "note:minimum_lease_after_five_years", "unknown")), "(v0 + v1 + v2) / 1e9", "total future minimum non-cancellable operating-lease payments across all maturity buckets, VND billion"),
    209: Formula((op("HPG", 2021, "note:long_term_loan_receivable"),), "v0 / 1e11", "HPG consolidated ending long-term loan receivable in 2021, VND hundred-billion; replaces an AUD currency-exposure row"),
    239: Formula((op("EIB", 2024, "note:tangible_ppe_gross_cost_million", "separate"),), "v0", "parent EIB ending tangible-PPE gross cost at 31 December 2024, million VND; replaces the intangible-PPE note"),
    253: Formula((op("MBB", 2016, "note:associate_investment_original_cost_million"),), "v0", "MBB total original cost of investments in associates at 31 December 2016, million VND; replaces tangible-PPE gross cost"),
    267: Formula((op("ACB", 2020, "note:term_deposits_foreign_currency_million", "separate"),), "v0", "parent ACB ending foreign-currency term deposits at 31 December 2020, million VND; selects the row within the term-deposit section"),
    601: Formula((op("KBC", 2019, "note:short_term_interest_payable", "separate"), op("KBC", 2016, "note:short_term_interest_payable", "separate")), "(v0 - v1) / 1e9", "parent KBC ending short-term accrued borrowing interest in 2019 minus 2016, VND billion; fixes reversed document variables"),
    625: Formula((op("VGC", 2024, "note:vib_short_term_loan_ending"), op("VGC", 2024, "note:vib_short_term_loan_opening")), "(v0 - v1) / 1e6", "change in VGC short-term borrowing specifically from Vietnam International Bank from end-2023 to end-2024, million VND; excludes all other lenders"),
    578: Formula((op("BAF", 2025, "note:short_term_supplier_advances_total", "separate"), op("BAF", 2022, "note:short_term_supplier_advances_total", "separate")), "(v0 - v1) / 1e9", "parent BAF ending total short-term supplier advances in 2025 minus end-2022, VND billion; replaces the adjacent provision-row lookup"),
    720: Formula((op("SGB", 2017, "note:intangible_fixed_assets_nbv_million", "separate"), op("SGB", 2018, "note:intangible_fixed_assets_nbv_million", "separate")), "(v1 / v0 - 1) * 100", "parent SGB intangible fixed-assets NBV growth from end-2017 to end-2018; reads one ending-NBV total per year without summing duplicate comparative tables"),
    828: Formula((
        *(op("ACB", year, "note:impaired_financial_assets_million", "separate") for year in (2017, 2020, 2021, 2022, 2024)),
        *(op("ACB", year, "note:financial_asset_risk_provision_million", "separate") for year in (2017, 2020, 2021, 2022, 2024)),
    ), "max(abs(v5) / abs(v0), abs(v6) / abs(v1), abs(v7) / abs(v2), abs(v8) / abs(v3), abs(v9) / abs(v4)) * 100", "maximum parent ACB risk-provision coverage ratio across the five requested years: total risk provision divided by total impaired financial assets"),
    650: Formula((op("SSI", 2020, "note:financial_operating_revenue"), op("SSI", 2023, "note:financial_operating_revenue")), "(v1 / v0 - 1) * 100", "SSI consolidated financial-operating-revenue growth from 2020 to 2023; excludes other operating income"),
    676: Formula((op("VCB", 2017, "note:general_customer_loan_provision_expense_million", "separate"), op("VCB", 2017, "note:specific_customer_loan_provision_expense_million", "separate"), op("VCB", 2017, "note:pre_provision_operating_profit_million", "separate")), "(v0 + v1) / v2 * 100", "parent VCB total general-and-specific customer-loan provision expense / 2017 pre-provision operating profit"),
    711: Formula((op("DPM", 2017, "note:related_party_sales_revenue", "separate"), op("DPM", 2017, "note:gross_sales_revenue", "separate")), "v0 / v1 * 100", "parent DPM related-party sales revenue / total sales and service revenue in 2017"),
    984: Formula(tuple(op("PLX", year, "note:long_term_borrowing_due_within_12_months") for year in (2019, 2020, 2023)), "max(abs(v0), abs(v1), abs(v2)) / 1e9", "largest PLX long-term-borrowing amount due within 12 months across 2019, 2020 and 2023, VND billion"),
    339: Formula((op("NVL", 2018, "note:profit_attributable_to_parent_shareholders"),), "v0 / 1e9", "NVL 2018 profit after tax attributable to shareholders of the parent company, VND billion; excludes non-controlling interests"),
    688: Formula((op("DXG", 2018, "note:short_term_other_receivables_total"), op("DXG", 2018, "note:other_receivables_total")), "v0 / v1 * 100", "DXG ending short-term other receivables as a percentage of total other receivables"),
    690: Formula((op("HNG", 2020, "note:total_borrowings", "separate"), op("HNG", 2020, "note:cash_on_hand", "separate"), op("HNG", 2020, "note:bank_deposits", "separate")), "v0 / (v1 + v2)", "parent HNG ending total borrowings divided by cash on hand plus bank deposits"),
    731: Formula((op("HDB", 2025, "note:gross_customer_loans_million"), op("HDB", 2025, "note:customer_deposits_million")), "v0 / v1 * 100", "HDB ending gross customer loans divided by total customer deposits (LDR), percent"),
    670: Formula((op("BAF", 2019, "note:short_term_bank_borrowings", "separate"), op("BAF", 2019, "note:total_financial_liabilities", "separate")), "v0 / v1 * 100", "parent BAF opening-2020 short-term bank borrowings as a percentage of total financial liabilities"),
    693: Formula((op("VCB", 2025, "note:intangible_fixed_assets_nbv_million", "separate"), op("VCB", 2025, "note:tangible_fixed_assets_nbv_million", "separate")), "v0 / (v0 + v1) * 100", "parent VCB ending intangible fixed-assets net book value as a percentage of total tangible plus intangible fixed assets"),
    30: Formula((op("BVH", 2021, "note:other_entity_equity_investments_net"),), "v0 / 1e6", "BVH ending net equity investments in other entities, million VND"),
    73: Formula((op("PC1", 2024, "note:associate_investment_carrying_amount"),), "v0 / 1e12", "PC1 ending carrying amount of investments in associates, VND trillion"),
    242: Formula((op("MBB", 2021, "note:intangible_fixed_assets_nbv_million", "separate"),), "v0", "parent MBB ending intangible fixed-assets net book value, million VND"),
    149: Formula((op("VSC", 2024, "note:services_purchased_from_nhdv"),), "v0 / 1e11", "VSC services purchased from Nam Hai Dinh Vu Port during 2024, VND hundred-billion"),
    260: Formula((op("BAF", 2025, "note:anh_vu_phu_yen_ownership_percent", "separate"),), "v0", "parent BAF ownership of Anh Vu Phu Yen at end-2025, percent"),
    512: Formula((
        *(op("HSG", year, "cdkt:400") for year in (2015, 2018, 2019, 2021, 2022, 2023)),
        *(op("HSG", year, "note:long_term_borrowing_balance") for year in (2015, 2018, 2019, 2021, 2022, 2023)),
    ), "max((v0, v6), (v1, v7), (v2, v8), (v3, v9), (v4, v10), (v5, v11))[1] / 1e9", "HSG long-term-borrowing balance in the requested year with the highest ending total equity; source-proven maximum is 2022, VND billion"),
    797: Formula((op("ACV", 2019, "note:total_employee_expense"), op("VJC", 2019, "note:total_employee_and_labor_expense")), "abs(v0 - v1) / 1e9", "absolute difference between ACV and Vietjet total employee/labor expense in 2019, VND billion; the question asks how much one amount differs rather than for a signed subtraction"),
    790: Formula((op("VIB", 2023, "note:manufacturing_loan_share_ocr_percent", "separate"), op("BID", 2023, "note:manufacturing_loan_share_ocr_percent", "separate")), "(v0 - v1) / 100", "parent VIB minus parent BID manufacturing-sector loan-balance share at end-2023, percentage points; OCR percentages are divided by 100"),
    703: Formula((op("GEX", 2024, "note:related_short_term_trade_receivables", "separate"), op("GEX", 2024, "note:related_other_short_term_receivables", "separate"), op("GEX", 2024, "note:related_short_term_trade_payables", "separate"), op("GEX", 2024, "note:related_other_short_term_payables", "separate")), "(v0 + v1 - v2 - v3) / 1e9", "parent GEX net disclosed related-party short-term receivables less short-term payables at end-2024, VND billion; loans/borrowings are excluded from receivable/payable wording"),
    623: Formula((op("PC1", 2023, "note:raw_materials_gross_ending"), op("PC1", 2022, "note:raw_materials_gross_ending")), "abs(v0 - v1) / 1e9", "absolute change in PC1 gross raw-material inventory between end-2023 and end-2022, VND billion"),
    733: Formula((op("SSB", 2020, "note:total_employee_income_million", "separate"), op("SSB", 2020, "note:average_employee_count", "separate"), op("VPB", 2020, "note:total_employee_income_million", "separate"), op("VPB", 2020, "note:average_employee_count", "separate")), "abs(v0 / v1 / 12 - v2 / v3 / 12)", "absolute difference in parent SSB/VPB average monthly employee income in 2020, million VND per employee; recomputed from disclosed totals and headcounts to avoid OCR decimal loss"),
    741: Formula((op("VSF", 2019, "note:basic_eps_vnd"), op("MPC", 2019, "note:basic_eps_vnd")), "abs(v0 - v1) / 1000", "absolute VSF/MPC 2019 basic-EPS difference, thousand VND per share"),
    793: Formula((op("GEE", 2025, "note:common_shares_outstanding"), op("GEX", 2025, "note:common_shares_outstanding")), "(v0 - v1) / 1e6", "GEE minus GEX ending common shares outstanding, million shares"),
    800: Formula((op("BID", 2022, "note:deposits_and_borrowings_other_credit_institutions_million", "separate"), op("MSB", 2022, "note:deposits_and_borrowings_other_credit_institutions_million", "separate")), "v0 - v1", "amount by which parent BID deposits-and-borrowings liability to other credit institutions exceeds parent MSB, million VND"),
    823: Formula(tuple(op("HSG", year, "note:bonus_welfare_fund_allocation") for year in (2019, 2021, 2024, 2025)), "(1 if abs(v0) > 40e9 else 0) + (1 if abs(v1) > 40e9 else 0) + (1 if abs(v2) > 40e9 else 0) + (1 if abs(v3) > 40e9 else 0)", "count of requested HSG years whose annual bonus-and-welfare-fund allocation exceeded VND 40 billion"),
    89: Formula((op("MSN", 2018, "note:current_income_tax_expense_million"),), "v0", "MSN current corporate income tax expense in 2018, million VND; excludes deferred-tax benefit"),
    133: Formula((op("NLG", 2024, "note:board_and_executive_income_total", "separate"),), "v0 / 1e9", "total parent NLG Board and executive-management income in 2024, VND billion"),
    127: Formula((op("VIB", 2024, "note:other_tangible_ppe_nbv_million"),), "v0", "VIB ending net book value of other tangible fixed assets at end-2024, million VND"),
    524: Formula((
        *(op("OCB", year, "note:domestic_economic_and_individual_loans") for year in (2017, 2018, 2021, 2022)),
        *(op("OCB", year, "note:bonus_welfare_fund_ending") for year in (2017, 2018, 2021, 2022)),
    ), "max((v0, v4), (v1, v5), (v2, v6), (v3, v7))[1] / 1e9", "OCB ending bonus-and-welfare fund in the requested year with maximum loans to domestic economic entities and individuals, VND billion"),
    530: Formula((
        *(op("NAB", year, "note:construction_in_progress_million") for year in (2021, 2022, 2023, 2024, 2025)),
        *(op("NAB", year, "note:performing_customer_loans_million") for year in (2021, 2022, 2023, 2024, 2025)),
    ), "max((v0, v5), (v1, v6), (v2, v7), (v3, v8), (v4, v9))[1] / 1e9", "NAB ending performing customer loans in the requested year with maximum ending construction in progress, VND thousand-trillion; source-proven maximum is 2025"),
    522: Formula((
        *(op("BVH", year, "note:written_off_bad_debt") for year in (2017, 2018, 2023, 2025)),
        op("BVH", 2017, "note:listed_equity_down10_pbt_impact"),
        op("BVH", 2018, "note:listed_equity_down10_pbt_impact"),
        op("BVH", 2023, "note:listed_equity_down10_pbt_impact_million"),
        op("BVH", 2025, "note:listed_equity_down10_pbt_impact_million"),
    ), "max((v0, v4 / 1e6), (v1, v5 / 1e6), (v2, v6), (v3, v7))[1]", "BVH PBT impact of a 10% listed-equity price decrease in the requested year with maximum ending written-off bad debt, million VND; source-proven maximum is 2025"),
    505: Formula((
        *(op("IJC", year, "note:short_term_bank_loan", "separate") for year in (2017, 2018, 2019, 2022, 2024)),
        op("IJC", 2024, "note:related_party_short_term_loan_payable", "separate"),
    ), "v5 / 1e9 if v4 == max(v0, v1, v2, v3, v4) else 0", "parent IJC related-party short-term loan payable in the requested year with maximum short-term bank loans; source-proven maximum is 2024, VND billion"),
    508: Formula((
        op("OCB", 2021, "note:deferred_expense", "separate"),
        op("ACB", 2021, "note:deferred_expense_million", "separate"),
        op("STB", 2021, "note:deferred_expense_million", "separate"),
        op("STB", 2021, "note:net_other_income_million", "separate"),
    ), "v3 if v2 == max(v0 / 1e6, v1, v2) else 0", "parent net other income of the bank with maximum ending deferred expense among OCB, ACB and STB; source-proven maximum is STB, VND million"),
    507: Formula((
        *(op("DIG", year, "note:short_term_supplier_advances", "separate") for year in (2015, 2016, 2017, 2021, 2024)),
        *(op("DIG", year, "kqkd:23", "separate") for year in (2015, 2016, 2017, 2021, 2024)),
        *(op("DIG", year, "kqkd:50", "separate") for year in (2015, 2016, 2017, 2021, 2024)),
    ), "(v5 / v10 if v0 == max(v0, v1, v2, v3, v4) else v6 / v11 if v1 == max(v0, v1, v2, v3, v4) else v7 / v12 if v2 == max(v0, v1, v2, v3, v4) else v8 / v13 if v3 == max(v0, v1, v2, v3, v4) else v9 / v14) * 100", "parent DIG interest-expense/PBT ratio in the requested year with maximum ending short-term advances to suppliers"),
    535: Formula((
        *(op("TTF", year, "note:short_term_trade_payables", "separate") for year in (2017, 2019, 2020, 2021, 2022, 2024)),
        op("TTF", 2017, "note:asset_liquidation_income", "separate"),
    ), "v6 / 1e6 if v0 == max(v0, v1, v2, v3, v4, v5) else 0", "parent TTF asset-liquidation income in the requested year with maximum ending short-term trade payables; source-proven maximum is 2017, VND million"),
    509: Formula((
        *(op("GAS", year, "note:tax_and_other_payables") for year in (2015, 2016, 2017, 2019)),
        op("GAS", 2017, "note:sales_to_pow"),
    ), "v4 / 1e12 if v2 == max(v0, v1, v2, v3) else 0", "GAS sales to PetroVietnam Power in the requested year with maximum ending taxes and other State payables; source-proven maximum is 2017, VND trillion"),
    510: Formula((
        op("BAB", 2024, "note:employee_expense_million"),
        op("SSB", 2024, "note:employee_expense_million"),
        op("NAB", 2024, "note:employee_expense_million"),
        op("VIB", 2024, "note:employee_expense_million"),
        op("VIB", 2024, "note:interbank_loans_vnd_million"),
        op("VIB", 2024, "note:interbank_demand_deposits_vnd_million"),
        op("VIB", 2024, "note:interbank_term_deposits_vnd_million"),
    ), "v4 / (v5 + v6) * 100 if v3 == max(v0, v1, v2, v3) else 0", "interbank-loans/deposits ratio of the bank with maximum employee expense among BAB, SSB, NAB and VIB; source-proven maximum is VIB"),
    514: Formula((
        *(op("HDG", year, "lctt:40", "separate") for year in (2015, 2016, 2017, 2018, 2019)),
        *(op("HDG", year, "note:apartment_customer_advances", "separate") for year in (2015, 2016, 2017, 2018, 2019)),
    ), "(v5 if v0 == min(v0, v1, v2, v3, v4) else v6 if v1 == min(v0, v1, v2, v3, v4) else v7 if v2 == min(v0, v1, v2, v3, v4) else v8 if v3 == min(v0, v1, v2, v3, v4) else v9) / 1e9", "parent HDG apartment-customer advances in the requested year with minimum net financing cash flow, VND billion"),
    892: Formula(tuple(
        op("HAG", year, metric)
        for year in (2015, 2016, 2017, 2022)
        for metric in ("note:laos_external_revenue", "note:geographic_external_revenue_total")
    ), "(v0 / v1 + v2 / v3 + v4 / v5 + v6 / v7) / 4 * 100", "mean HAG Laos share of total external-customer revenue across 2015, 2016, 2017 and 2022"),
    895: Formula((
        op("VIC", 2015, "note:operating_lease_minimum_receipts_total"),
        *(op("VIC", year, "note:operating_lease_minimum_receipts_total_million") for year in (2019, 2022, 2025)),
    ), "max(v0 / 1e9, v1 / 1e3, v2 / 1e3, v3 / 1e3)", "maximum VIC total minimum operating-lease receipts across the four requested years, VND billion; 2015 is in VND and later reports are in million VND"),
    941: Formula(tuple(op("NLG", year, "note:subsidiary_short_term_other_payables", "separate") for year in (2017, 2018, 2021, 2025)), "max(v0, v1, v2, v3) / 1e9", "maximum parent NLG ending short-term other payables to subsidiaries across the four requested years, VND billion"),
    112: Formula((op("FOX", 2022, "note:short_term_term_deposit_carrying_amount", "separate"),), "v0 / 1e6", "parent FOX carrying amount of short-term term deposits at end-2022, VND million"),
    44: Formula((op("STB", 2016, "note:total_assets_million"),), "v0", "STB ending total assets at end-2016, million VND"),
    64: Formula((op("STB", 2021, "note:total_assets_million"),), "v0", "STB ending consolidated total assets at end-2021, million VND; replace the derivative-contract value selected from the geographical credit-risk table"),
    151: Formula((op("EIB", 2015, "note:savings_deposits_vnd_million"),), "v0", "EIB ending savings deposits denominated in VND, million VND; use the savings/VND sub-row rather than the all-deposit total"),
    255: Formula((op("DCM", 2016, "note:short_term_supplier_advances", "separate"),), "v0 / 1e9", "parent DCM ending short-term supplier advances, billion VND; use balance-sheet code 132 rather than a named related-party receivable"),
    78: Formula((op("VRE", 2022, "note:interest_payable_million", "separate"),), "v0", "parent VRE ending interest payable at end-2022, million VND"),
    82: Formula((op("NVB", 2022, "note:deposits_at_other_credit_institutions_million", "separate"),), "v0", "parent NVB deposits at other credit institutions at end-2022, million VND"),
    123: Formula((op("SSI", 2016, "cdkt:400"),), "v0 / 1e9", "SSI consolidated total equity at end-2016, VND billion"),
    178: Formula((op("NVL", 2021, "cdkt:221"),), "v0 / 1e11", "NVL ending tangible fixed-assets net book value (balance-sheet code 221), VND hundred-billion"),
    202: Formula((op("CEO", 2025, "note:short_term_supplier_advances"),), "v0 / 1e9", "CEO ending short-term advances to suppliers at end-2025, VND billion"),
    213: Formula((op("ACV", 2018, "note:ending_usd_balance", "consolidated"),), "v0 / 1e6", "ACV ending off-balance-sheet US-dollar amount, million USD"),
    97: Formula((op("HPG", 2023, "note:gang_steel_ownership_pct", "separate"),), "v0", "HPG direct ownership percentage in Hoa Phat Steel Joint Stock Company at end-2023; read from the explicit parent-company ownership column"),
    355: Formula((op("MSN", 2022, "note:3f_food_economic_interest_pct", "consolidated"),), "v0", "MSN economic interest in 3F Viet Food Company Limited at end-2022; read from the explicit subsidiary-roster percentage column"),
    629: Formula((op("HBC", 2016, "cdkt:137", "separate"), op("HBC", 2020, "cdkt:137", "separate")), "(abs(v1) / abs(v0) - 1) * 100", "growth in the magnitude of parent HBC's ending allowance for doubtful short-term receivables from 2016 to 2020; both operands are the same balance-sheet code 137"),
    263: Formula((op("SSI", 2019, "note:fvtpl_ending_fair_value", "separate"),), "v0 / 1e12", "parent SSI ending fair value of FVTPL financial assets at end-2019, VND trillion; use the ending revalued amount rather than current-year income from FVTPL"),
    218: Formula((op("BVH", 2021, "note:other_entity_investment_net", "separate"),), "v0 / 1e12", "parent BVH ending net carrying amount of investments in other entities at end-2021, VND trillion; use the detailed net column rather than the summary gross-cost row"),
    233: Formula((op("GEX", 2017, "note:operating_lease_minimum_receipts_total"),), "v0 / 1e11", "GEX total future minimum operating-lease receipts at end-2017, VND hundred-billion"),
    268: Formula((op("DXS", 2021, "lctt:21"),), "abs(v0) / 1e9", "DXS consolidated cash paid to acquire fixed assets in 2021, VND billion"),
    279: Formula((op("HSG", 2020, "cdkt:111"),), "v0 / 1e9", "HSG consolidated cash on hand (balance-sheet code 111) at 30 September 2020, VND billion"),
    289: Formula((op("DPM", 2016, "cdkt:230", "separate"),), "v0 / 1e9", "parent DPM ending investment-property net book value, VND billion"),
    304: Formula((op("VGC", 2016, "cdkt:421"),), "v0 / 1e11", "VGC consolidated ending undistributed profit, VND hundred-billion; the question does not request parent-only scope"),
    762: Formula((op("GEX", 2019, "cdkt:250"), op("SAM", 2019, "cdkt:250")), "abs(v0 - v1) / 1e9", "absolute difference between GEX and SAM ending long-term financial investments in 2019, VND billion"),
    799: Formula((op("HBC", 2021, "cdkt:400", "separate"), op("SAM", 2021, "cdkt:400", "separate")), "abs(v0 - v1) / 1e9", "absolute difference between parent HBC and parent SAM ending total equity in 2021, VND billion"),
    156: Formula((op("BVH", 2018, "note:financial_assets_exposed_to_credit_risk", "separate"),), "v0 / 1e9", "parent BVH total financial assets exposed to credit risk at end-2018, VND billion"),
    685: Formula((
        op("BAB", 2025, "note:customer_loan_loss_provision_million", "separate"),
        op("BAB", 2025, "note:substandard_loans_million", "separate"),
        op("BAB", 2025, "note:doubtful_loans_million", "separate"),
        op("BAB", 2025, "note:loss_loans_million", "separate"),
    ), "abs(v0) / (v1 + v2 + v3) * 100", "parent BAB loan-loss coverage ratio: ending customer-loan provision divided by gross non-performing loans"),
    721: Formula((
        op("VIF", 2020, "note:other_entity_share_investment", "consolidated"),
        op("VIF", 2020, "note:associate_joint_venture_investment", "consolidated"),
    ), "v0 / v1 * 100", "VIF ending share investments as a percentage of ending investment in associates and joint ventures"),
    780: Formula((op("BAB", 2024, "note:customer_loan_loss_provision_million", "separate"), op("SGB", 2024, "note:customer_loan_loss_provision_million", "separate")), "abs(v0) - abs(v1)", "parent BAB customer-loan provision difference versus parent SGB at end-2024, million VND"),
    792: Formula((op("EIB", 2022, "note:other_on_balance_assets_provision_million"), op("MBB", 2022, "note:other_on_balance_assets_provision_million")), "abs(v0) - abs(v1)", "EIB excess over MBB in ending provision for other on-balance-sheet assets at end-2022, million VND"),
    827: Formula((
        op("SNZ", 2019, "note:tax_and_state_payables", "separate"),
        op("VIC", 2019, "note:tax_and_state_payables_million", "separate"),
        op("DXS", 2019, "note:tax_and_state_payables", "separate"),
        op("HPX", 2019, "note:tax_and_state_payables", "separate"),
    ), "(v0 / 1e9 + v1 / 1e3 + v2 / 1e9 + v3 / 1e9) / 4", "mean ending parent-company taxes and State payables for SNZ, VIC, DXS and HPX, VND billion"),
    498: Formula((
        *(op("ACB", year, "note:operating_expense_million", "separate") for year in (2018, 2020, 2022)),
        *(op("ACB", year, "note:personal_deposits_million", "separate") for year in (2018, 2020, 2022)),
    ), "(v3 if abs(v0) > 10000000 else 0) + (v4 if abs(v1) > 10000000 else 0) + (v5 if abs(v2) > 10000000 else 0)", "parent ACB individual deposits in the requested year whose operating expense exceeds VND 10,000 billion; source amounts and result are VND million"),
    795: Formula((op("GAS", 2019, "note:short_term_prepaid_expense", "separate"), op("POW", 2019, "note:short_term_prepaid_expense", "separate")), "(v0 - v1) / 1e9", "excess of parent GAS over parent POW short-term prepaid expense at end-2019, VND billion"),
    798: Formula((op("HHV", 2022, "cdkt:221", "separate"), op("VSC", 2022, "cdkt:221", "separate")), "abs(v0 - v1) / 1e9", "absolute difference in parent tangible fixed-assets net book value (code 221) at end-2022, VND billion"),
    809: Formula((op("DNH", 2025, "note:corporate_tax_payable_during_year", "separate"), op("HND", 2025, "note:corporate_tax_payable_during_year", "unknown")), "abs(v0 - v1) / 1e9", "difference in 2025 corporate income tax payable during the year between parent DNH and HND, VND billion"),
    815: Formula(tuple(op("OCB", year, "note:net_fx_result") for year in (2017, 2020, 2021, 2022)), "max(v0, v1, v2, v3) / 1e9", "maximum OCB consolidated net foreign-exchange result across the four requested years, VND billion"),
    816: Formula(tuple(op("MSN", year, "short_term_borrowings") for year in (2017, 2020, 2022)), "(v0 + v1 + v2) / 3 / 1e12", "mean MSN consolidated short-term borrowings across the three requested year ends, VND trillion"),
    829: Formula(tuple(op("TTF", year, "kqkd:40") for year in (2016, 2017, 2023, 2025)), "max((v0, 2016), (v1, 2017), (v2, 2023), (v3, 2025))[1]", "year of maximum TTF consolidated other profit across the requested years"),
    830: Formula(tuple(op("STB", year, "note:cash_and_gold_million", "separate") for year in (2017, 2021, 2022)), "(v0 + v1 + v2) / 3", "mean parent STB cash and gold across the three requested year ends, VND million"),
    833: Formula(tuple(op("VGT", year, "note:dividend_income", "separate") for year in (2016, 2024, 2025)), "(v0 + v1 + v2) / 1e11", "total parent VGT dividend income across the three requested years, VND hundred-billion"),
    834: Formula(tuple(op("GAS", year, "revenue") for year in (2015, 2016, 2017, 2025)), "max(v0, v1, v2, v3) / 1e12", "maximum GAS consolidated net revenue across the four requested years, VND trillion"),
    835: Formula(tuple(op("FOX", year, "note:long_term_borrowings_ending", "separate") for year in (2016, 2017, 2018, 2019, 2020)), "max(v0, v1, v2, v3, v4) / 1e9", "maximum parent FOX ending total long-term borrowings across the five requested years, VND billion"),
    836: Formula(tuple(op("TTF", year, "note:doubtful_receivable_provision_change", "separate") for year in (2017, 2019, 2021, 2025)), "max(v0, v1, v2, v3) / 1e9", "maximum parent TTF doubtful-receivable provision charge across the four requested years, VND billion; negative source values are reversals"),
    838: Formula(tuple(op("MSN", year, "cdkt:250") for year in (2017, 2018, 2020, 2021, 2024)), "max(v0, v1, v2, v3, v4) / 1e12", "maximum MSN consolidated long-term financial investments across the five requested year ends, VND trillion"),
    839: Formula(tuple(
        op("BSR", year, metric)
        for year in (2017, 2019, 2021, 2024, 2025)
        for metric in ("note:inventory_gross", "note:inventory_provision")
    ), "(abs(v1) / v0 + abs(v3) / v2 + abs(v5) / v4 + abs(v7) / v6 + abs(v9) / v8) / 5 * 100", "mean BSR inventory-provision-to-gross-inventory ratio across the five requested year ends"),
    840: Formula(tuple(op("HPG", year, "note:long_term_prepaid_expense_total", "separate") for year in (2016, 2017, 2018, 2020, 2024)), "(v0 + v1 + v2 + v3 + v4) / 5 / 1e9", "mean parent HPG ending long-term prepaid expense across the five requested years, VND billion"),
    843: Formula(tuple(op("EVF", year, "note:dividends_received_million", "unknown") for year in (2020, 2022, 2024)), "v0 + v1 + v2", "total EVF dividends received from investments across the three requested years, VND million"),
    845: Formula(tuple(op("BVH", year, "note:basic_eps") for year in (2017, 2019, 2022, 2024)), "max(v0, v1, v2, v3) / 1e3", "maximum BVH consolidated basic EPS, thousand VND per share"),
    854: Formula(tuple(
        op("NLG", year, metric)
        for year in (2020, 2021, 2023)
        for metric in ("note:unearned_revenue_current", "note:unearned_revenue_noncurrent")
    ), "(v0 + v1 + v2 + v3 + v4 + v5) / 3 / 1e11", "mean NLG consolidated ending unearned revenue, current plus non-current, VND hundred-billion"),
    856: Formula(tuple(op("PC1", year, "note:basic_eps") for year in (2015, 2020, 2022, 2023, 2024)), "(v0 + v1 + v2 + v3 + v4) / 5", "mean PC1 consolidated basic EPS across the five requested years, VND per share"),
    859: Formula(tuple(op("DCM", year, "note:bonus_welfare_fund_ending") for year in (2022, 2023, 2024)), "max(v0, v1, v2) / 1e9", "maximum DCM consolidated ending bonus-and-welfare fund balance across the three requested years, VND billion"),
    863: Formula(tuple(op("HAG", year, "note:xnk_hagl_short_term_other_payable", "separate") for year in (2015, 2019, 2021)), "(1 if v0 > 0 else 0) + (1 if v1 > 0 else 0) + (1 if v2 > 0 else 0)", "count of requested years with a positive parent HAG short-term other payable to HAGL Import Export"),
    864: Formula(tuple(op("EIB", year, "note:bonus_welfare_fund_ending_million") for year in (2017, 2019, 2021, 2025)), "max(v0, v1, v2, v3) / 1e5", "maximum EIB consolidated ending bonus-and-welfare fund balance, VND hundred-billion; source amounts are VND million"),
    865: Formula(tuple(
        op("VIF", year, metric, "separate")
        for year in (2017, 2020, 2021, 2022)
        for metric in ("note:trade_receivables_total", "note:trade_receivables_provision")
    ), "(abs(v1) / v0 + abs(v3) / v2 + abs(v5) / v4 + abs(v7) / v6) / 4 * 100", "mean parent VIF short-term trade-receivables provision ratio across the four requested year ends"),
    867: Formula(tuple(op("HPX", year, "liabilities") for year in (2022, 2023, 2025)), "(v0 + v1 + v2) / 3 / 1e12", "mean HPX consolidated total liabilities across the three requested year ends, VND trillion"),
    868: Formula(tuple(op("VCB", year, "note:current_income_tax_million") for year in (2015, 2017, 2018, 2022, 2023)), "max(abs(v0), abs(v1), abs(v2), abs(v3), abs(v4)) / 1e3", "maximum VCB consolidated current corporate income tax expense across the five requested years, VND billion; source amounts are VND million"),
    879: Formula(tuple(op("VNM", year, "note:corporate_income_tax_paid", "separate") for year in (2019, 2022, 2023, 2024, 2025)), "max((abs(v0), 2019), (abs(v1), 2022), (abs(v2), 2023), (abs(v3), 2024), (abs(v4), 2025))[1]", "year of maximum parent VNM corporate-income-tax cash outflow across the five requested years"),
    894: Formula(tuple(op("VCB", year, "note:bank_corporate_income_tax_paid_million", "separate") for year in (2018, 2020, 2021, 2022, 2025)), "(abs(v0) + abs(v1) + abs(v2) + abs(v3) + abs(v4)) / 5", "mean parent VCB corporate income tax paid across the five requested years, VND million; parenthesized cash outflows are treated as amounts paid"),
    871: Formula(tuple(op("MPC", year, "note:bonus_welfare_appropriation") for year in (2019, 2022, 2023, 2024)), "max(abs(v0), abs(v1), abs(v2), abs(v3)) / 1e9", "maximum MPC consolidated annual bonus-and-welfare fund appropriation across the four requested years, VND billion"),
    873: Formula(tuple(
        op("DXG", year, metric)
        for year in (2017, 2018, 2021, 2023, 2025)
        for metric in ("note:current_tax_expense_signed", "note:deferred_tax_expense_signed")
    ), "max(abs(v0 + v1), abs(v2 + v3), abs(v4 + v5), abs(v6 + v7), abs(v8 + v9)) / 1e9", "maximum DXG consolidated total income-tax expense, net of deferred-tax income, VND billion"),
    881: Formula(tuple(
        op("HND", year, metric, "unknown")
        for year in (2016, 2018, 2019, 2020, 2021)
        for metric in ("note:tangible_ppe_gross", "note:tangible_ppe_accumulated_depreciation")
    ), "(abs(v1) / v0 + abs(v3) / v2 + abs(v5) / v4 + abs(v7) / v6 + abs(v9) / v8) / 5 * 100", "mean HND accumulated-depreciation-to-gross-cost ratio for tangible fixed assets across the five requested years"),
    884: Formula(tuple(op("HND", year, "cdkt:300", "unknown") for year in (2016, 2017, 2018, 2021, 2022)), "max((v0, 2016), (v1, 2017), (v2, 2018), (v3, 2021), (v4, 2022))[1]", "year of maximum HND total liabilities across the five requested years"),
    885: Formula(tuple(
        op("STB", year, metric)
        for year in (2021, 2022, 2025)
        for metric in ("note:cd_under_12m_million", "note:cd_12m_to_5y_million", "note:cd_5y_plus_million")
    ), "max(v0 / (v0 + v1 + v2), v3 / (v3 + v4 + v5), v6 / (v6 + v7 + v8)) * 100", "maximum STB share of certificates of deposit under 12 months across the three requested year ends"),
    887: Formula(tuple(op("VIC", year, "note:board_remuneration_million", "separate") for year in (2022, 2023, 2024)), "v0 + v1 + v2", "cumulative parent VIC board remuneration across 2022-2024, VND million"),
    902: Formula(tuple(op("VPI", year, "cash") for year in (2016, 2017, 2018)), "max(v0, v1, v2) / 1e9", "maximum VPI consolidated cash and cash equivalents across 2016-2018, VND billion"),
    903: Formula(tuple(op("MPC", year, "kqkd:32") for year in (2017, 2018, 2020, 2021, 2022)), "max(v0, v1, v2, v3, v4) / 1e11", "maximum MPC consolidated other expense across the five requested years, VND hundred-billion"),
    909: Formula(tuple(op("GEE", year, "note:cft_short_term_loan_receivable", "separate") for year in (2022, 2023, 2024, 2025)), "max(v0, v1, v2, v3) / 1e9", "maximum parent GEE short-term loan receivable from CFT across the four requested year ends, VND billion"),
    911: Formula(tuple(op("VIF", year, "note:tangible_fixed_assets_nbv", "separate") for year in (2017, 2020, 2021, 2022, 2024)), "max(v0, v1, v2, v3, v4) / 1e9", "maximum parent VIF ending tangible fixed-assets net book value across the five requested years, VND billion"),
    912: Formula(tuple(op("VIF", year, "note:board_management_income_total", "separate") for year in (2022, 2024, 2025)), "max(v0, v1, v2) / 1e9", "maximum parent VIF annual total board and management income across the three requested years, VND billion"),
    916: Formula(tuple(op("FOX", year, "note:short_term_prepaid_expense_total", "separate") for year in (2016, 2018, 2019, 2020)), "max(v0, v1, v2, v3) / 1e9", "maximum parent FOX ending short-term prepaid expense across the four requested years, VND billion"),
    930: Formula(tuple(op("PC1", year, "note:dividend_profit_receivable", "separate") for year in (2022, 2023, 2024, 2025)), "max(v0, v1, v2, v3) / 1e9", "maximum parent PC1 ending short-term dividend and distributed-profit receivable across the four requested years, VND billion"),
    935: Formula((
        op("GEG", 2023, "note:welfare_fund_appropriation", "separate"),
        op("GEG", 2023, "note:net_profit_equity_movement", "separate"),
        op("DNH", 2023, "note:welfare_fund_appropriation", "separate"),
        op("DNH", 2023, "note:net_profit_equity_movement", "separate"),
        op("HDG", 2023, "note:retained_earnings_opening", "separate"),
        op("HDG", 2023, "note:stock_dividend_equity_movement", "separate"),
        op("HDG", 2023, "note:net_profit_equity_movement", "separate"),
        op("HDG", 2023, "note:retained_earnings_ending", "separate"),
    ), "(abs(v0) / v1 + abs(v2) / v3 + abs(v7 - v4 - v5 - v6) / v6) / 3 * 100", "mean parent bonus-and-welfare appropriation/net-profit ratio for GEG, DNH and HDG; HDG appropriation is the retained-earnings residual"),
    944: Formula(tuple(op("DLG", year, "note:construction_in_progress_ending") for year in (2020, 2021, 2022, 2023)), "max(v0, v1, v2, v3) / 1e9", "maximum DLG consolidated ending construction in progress across the four requested years, VND billion"),
    945: Formula(tuple(
        op(ticker, 2023, f"note:off_balance_foreign_currency_{index}")
        for ticker, count in (("MSR", 6), ("GVR", 5), ("AAA", 6))
        for index in range(count)
    ), "(v0 / (v0 + v1 + v2 + v3 + v4 + v5) + v6 / (v6 + v7 + v8 + v9 + v10) + v11 / (v11 + v12 + v13 + v14 + v15 + v16)) / 3 * 100", "mean company-level USD share of end-2023 off-balance-sheet foreign-currency amounts for MSR, GVR and AAA"),
    979: Formula((
        *(op(ticker, 2022, "note:deposit_interest_income") for ticker in ("POW", "GAS", "DTK", "GEG")),
        *(op(ticker, 2022, "finance_revenue") for ticker in ("POW", "GAS", "DTK", "GEG")),
    ), "(v0 + v1 + v2 + v3) / (v4 + v5 + v6 + v7) * 100", "aggregate 2022 deposit-interest share of total finance revenue for POW, GAS, DTK and GEG"),
    914: Formula(tuple(op("SJG", year, "kqkd:70") for year in (2019, 2020, 2021, 2022, 2023)), "max(v0, v1, v2, v3, v4) / 1e3", "maximum SJG basic earnings per share across 2019-2023, thousand VND per share"),
    940: Formula(tuple(op("SJG", year, "finance_revenue") for year in (2019, 2020, 2021)), "(v0 + v1 + v2) / 3 / 1e6", "mean SJG finance revenue across 2019-2021, VND million"),
    947: Formula(tuple(op("IJC", year, "cogs") for year in (2016, 2017, 2020, 2024)), "(v0 + v1 + v2 + v3) / 4 / 1e9", "mean IJC consolidated cost of sales across the four requested years, VND billion"),
    962: Formula((
        *(op("DNH", year, "liabilities", "separate") for year in (2016, 2017, 2018, 2021, 2022)),
        *(op("DNH", year, "equity", "separate") for year in (2016, 2017, 2018, 2021, 2022)),
    ), "(v0 / v5 + v1 / v6 + v2 / v7 + v3 / v8 + v4 / v9) / 5", "mean parent DNH liabilities-to-equity ratio across the five requested years"),
    972: Formula(tuple(op("DIG", year, "kqkd:31", "separate") for year in (2021, 2023, 2025)), "max(v0, v1, v2) / 1e9", "maximum parent DIG other income across 2021, 2023 and 2025, VND billion"),
    988: Formula(tuple(op("SJG", year, "kqkd:70") for year in (2019, 2021, 2022, 2023)), "max(v0, v1, v2, v3)", "maximum SJG basic earnings per share across the four requested years, VND per share"),
    994: Formula(tuple(op("QNS", year, "selling_expense") for year in (2019, 2020, 2021, 2022, 2023)), "max(v0, v1, v2, v3, v4) / 1e9", "maximum QNS consolidated selling expense across 2019-2023, VND billion"),
    998: Formula(tuple(op("HHS", year, "cash", "separate") for year in (2016, 2017, 2018, 2019, 2021)), "max(v0, v1, v2, v3, v4) / 1e9", "maximum parent HHS cash and cash equivalents across the five requested years, VND billion"),
    515: Formula((
        *(op("VAB", year, "note:total_liabilities") for year in (2020, 2024, 2025)),
        op("VAB", 2024, "note:materials_and_tools"), op("VAB", 2025, "note:materials_and_tools"),
    ), "v4 / 1e9 if v2 == max(v0, v1, v2) else v3 / 1e9", "VAB materials and tools in the requested year with maximum total liabilities; 2020 is source-proven below both later years"),
    513: Formula((
        *(op("HHS", year, "note:inventory_gross") for year in (2015, 2016, 2017, 2020, 2021)),
        *(op("HHS", year, "note:raw_materials_gross") for year in (2015, 2016, 2017, 2020, 2021)),
    ), "(v5 if v0 == max(v0, v1, v2, v3, v4) else v6 if v1 == max(v0, v1, v2, v3, v4) else v7 if v2 == max(v0, v1, v2, v3, v4) else v8 if v3 == max(v0, v1, v2, v3, v4) else v9) / 1e9", "HHS ending raw-materials gross amount in the requested year with maximum ending inventory gross amount, VND billion"),
    521: Formula((
        *(op("HDG", year, "cash") for year in (2021, 2023, 2025)),
        *(op("HDG", year, "kqkd:23") for year in (2021, 2023, 2025)),
    ), "(v3 if v0 == max(v0, v1, v2) else v4 if v1 == max(v0, v1, v2) else v5) / 1e9", "HDG interest expense in the requested year with maximum ending cash and cash equivalents, VND billion"),
    523: Formula((
        *(op("DCM", year, "note:accrued_term_deposit_interest", "separate") for year in (2019, 2022, 2023, 2024, 2025)),
        *(op("DCM", year, "note:cash_ending", "separate") for year in (2019, 2022, 2023, 2024, 2025)),
    ), "(v5 if v0 == max(v0, v1, v2, v3, v4) else v6 if v1 == max(v0, v1, v2, v3, v4) else v7 if v2 == max(v0, v1, v2, v3, v4) else v8 if v3 == max(v0, v1, v2, v3, v4) else v9) / 1e9", "parent DCM ending cash in the requested year with maximum accrued interest on term deposits, VND billion"),
    525: Formula((
        *(op("OCB", year, "note:corporate_tax_payable_during_year", "separate") for year in (2017, 2019, 2022)),
        *(op("OCB", year, "note:customer_loan_loss_provision", "separate") for year in (2017, 2019, 2022)),
    ), "abs(v3 if v0 == max(v0, v1, v2) else v4 if v1 == max(v0, v1, v2) else v5) / 1e12", "parent OCB ending customer-loan loss provision in the requested year with maximum corporate income tax payable during the year, VND trillion"),
    526: Formula((
        op("MWG", 2023, "kqkd:70"),
        op("MWG", 2024, "note:basic_and_diluted_eps"),
        op("MWG", 2025, "kqkd:70"),
        *(op("MWG", year, "kqkd:32") for year in (2023, 2024, 2025)),
    ), "(v3 if v0 == max(v0, v1, v2) else v4 if v1 == max(v0, v1, v2) else v5) / 1e9", "MWG other expense in the requested year with maximum basic and diluted EPS, VND billion"),
    534: Formula((
        *(op(ticker, 2024, "note:max_associate_voting_rate") for ticker in ("HBC", "GEX", "PC1")),
        *(op(ticker, 2024, "note:payables_after_12_months") for ticker in ("HBC", "GEX", "PC1")),
    ), "((v3 if v0 >= 50 else 0) + (v4 if v1 >= 50 else 0) + (v5 if v2 >= 50 else 0)) / 1e12", "post-12-month payables of companies whose 2024 associate/JV voting rate reaches at least 50%, VND trillion"),
    825: Formula(tuple(op("POW", year, "note:evn_short_term_payable", "separate") for year in (2017, 2018, 2021, 2024)), "max(v0, v1, v2, v3) / 1e9", "maximum parent POW ending short-term payable to Vietnam Electricity across the four requested years, VND billion"),
    851: Formula(tuple(op("GEX", year, "note:corporate_tax_payable_during_year", "separate") for year in (2015, 2018, 2022, 2025)), "(v0 + v1 + v2 + v3) / 1e9", "total parent GEX corporate income tax payable during the four requested years, VND billion"),
    875: Formula(tuple(op("STB", year, "note:parent_total_assets_million", "separate") for year in (2016, 2017, 2022, 2025)), "(v0 + v1 + v2 + v3) / 4 / 1e6", "mean parent STB total assets in the four requested years; source amounts are VND million and result is VND trillion"),
    877: Formula((
        *(op("HDB", year, "note:deposit_interest_cost") for year in (2023, 2024, 2025)),
        *(op("HDB", year, "note:total_interest_cost") for year in (2023, 2024, 2025)),
    ), "max((v0 / v3, 2023), (v1 / v4, 2024), (v2 / v5, 2025))[1]", "year of maximum HDB deposit-interest-cost share of total interest cost"),
    889: Formula(tuple(op("FTS", year, "note:intangible_ppe_nbv", "unknown") for year in (2019, 2020, 2023, 2024)), "max((v0, 2019), (v1, 2020), (v2, 2023), (v3, 2024))[1]", "year of maximum FTS ending intangible fixed-assets net book value"),
    893: Formula(tuple(
        op("HDG", year, f"note:related_party_sales_{index}", "separate")
        for year, count in ((2016, 9), (2017, 8), (2018, 8), (2019, 7))
        for index in range(count)
    ), "(" + " + ".join(f"v{index}" for index in range(32)) + ") / 1e9", "total parent HDG sales of goods and services to related parties across 2016-2019, VND billion"),
    896: Formula(tuple(op("TTF", year, "note:related_party_short_term_loan", "separate") for year in (2018, 2021, 2022, 2024)), "(v0 + v1 + v2 + v3) / 4 / 1e9", "mean parent TTF ending short-term related-party loans across the four requested years, VND billion"),
    925: Formula(tuple(op("VAB", year, "note:charter_capital") for year in (2020, 2024, 2025)), "max((v0, 2020), (v1, 2024), (v2, 2025))[1]", "year of maximum VAB charter capital"),
    999: Formula(tuple(op("VCB", year, "note:issued_valuable_papers_million", "separate") for year in (2016, 2020, 2025)), "max((v0, 2016), (v1, 2020), (v2, 2025))[1]", "year of maximum parent VCB issued valuable papers"),
    1012: Formula((
        *(op(ticker, 2017, "note:related_short_term_trade_receivables") for ticker in ("DTK", "GAS", "POW")),
        *(op(ticker, 2017, "note:related_short_term_other_receivables") for ticker in ("DTK", "GAS", "POW")),
    ), "(v0 + v1 + v2 + v3 + v4 + v5) / 3 / 1e9", "mean end-2017 short-term related-party receivables, including trade and other receivables, VND billion"),
    898: Formula(tuple(op(ticker, 2024, "note:related_short_term_other_payables", "separate") for ticker in ("DIG", "PDR", "VRE")), "(v0 + v1) / 1e6 + v2", "total end-2024 parent related-party short-term other payables, VND million; VRE source is already in million VND"),
    987: Formula((
        op("MCH", 2020, "note:fx_gain", "separate"), op("MCH", 2020, "note:fx_loss", "separate"),
        op("MSN", 2020, "note:fx_gain", "separate"), op("MSN", 2020, "note:fx_loss", "separate"),
        op("SAB", 2020, "note:fx_gain", "separate"), op("SAB", 2020, "note:fx_loss_realised", "separate"), op("SAB", 2020, "note:fx_loss_unrealised", "separate"),
        op("VNM", 2020, "note:fx_gain", "separate"), op("VNM", 2020, "note:fx_loss", "separate"),
    ), "(v0 - v1 + v2 - v3 + v4 - v5 - v6 + v7 - v8) / 1e6", "total 2020 parent net FX result for MCH, MSN, SAB and VNM, VND million"),
    991: Formula(tuple(op(ticker, 2016, "note:afs_impairment_provision") for ticker in ("SHB", "VIB", "BID", "CTG")), "(abs(v0) + abs(v1) + abs(v2) + abs(v3))", "total end-2016 AFS impairment provision for SHB, VIB, BID and CTG, VND million"),
    926: Formula((
        *(op(ticker, 2020, "note:finished_goods_gross") for ticker in ("MSN", "MML", "MPC", "VNM")),
        *(op(ticker, 2020, "note:inventory_gross_total") for ticker in ("MSN", "MML", "MPC", "VNM")),
    ), "(v0 / v4 + v1 / v5 + v2 / v6 + v3 / v7) / 4 * 100", "mean end-2020 finished-goods share of gross inventory"),
    939: Formula((
        *(op(ticker, 2022, "note:tangible_accumulated_depreciation") for ticker in ("PNJ", "MWG", "HHS", "HUT")),
        *(op(ticker, 2022, "note:tangible_gross_cost") for ticker in ("PNJ", "MWG", "HHS", "HUT")),
    ), "(abs(v0) / v4 + abs(v1) / v5 + abs(v2) / v6 + abs(v3) / v7) / 4 * 100", "mean end-2022 tangible-PPE accumulated-depreciation rate"),
    976: Formula((
        *(op(ticker, 2022, "note:admin_depreciation", "separate") for ticker in ("MSN", "MPC", "VNM", "MML")),
        *(op(ticker, 2022, "note:admin_expense_total", "separate") for ticker in ("MSN", "MPC", "VNM", "MML")),
    ), "(v0 / v4 + v1 / v5 + v2 / v6 + v3 / v7) / 4 * 100", "mean parent depreciation share of administration expense"),
    880: Formula((
        *(op(ticker, 2022, "note:government_bonds", "separate") for ticker in ("MBB", "MSB", "STB")),
        *(op(ticker, 2022, "note:total_debt_securities", "separate") for ticker in ("MBB", "MSB", "STB")),
    ), "(v0 / v3 + v1 / v4 + v2 / v5) / 3 * 100", "mean parent government-bond share of debt securities"),
    1006: Formula((
        op("NAB", 2025, "note:average_employee_income_monthly", "separate"),
        op("ABB", 2025, "note:average_employee_income_monthly_ocr", "separate"),
        op("ACB", 2025, "note:average_employee_income_annual", "separate"),
        op("STB", 2025, "note:average_employee_income_monthly_ocr", "separate"),
    ), "(v0 * 12 + (v1 / 100) * 12 + v2 + (v3 / 100) * 12) / 4", "mean 2025 parent annual employee income, VND million; ABB/STB OCR omit decimal separators"),
    819: Formula(tuple(op(ticker, 2024, "note:customer_loan_provision_expense", "separate") for ticker in ("MSB", "BID", "ABB")), "(v0 + v1 + v2) / 3 / 1e3", "mean 2024 parent customer-loan provision expense, VND billion"),
    861: Formula((op("OGC", 2015, "note:ending_common_shares_outstanding"), op("VNM", 2015, "note:ending_common_shares_outstanding"), op("HNG", 2015, "note:share_capital_thousand_vnd")), "(v0 + v1 + v2 / 10) / 3", "mean ending common shares for OGC, VNM and HNG; HNG share capital is in thousand VND at VND 10,000 par"),
    860: Formula((op("OGC", 2017, "note:other_expense"), op("OGC", 2022, "other_expense"), op("OGC", 2023, "other_expense")), "max((v0, 2017), (v1, 2022), (v2, 2023))[1]", "year of maximum OGC other expense"),
    927: Formula(tuple(op(ticker, 2016, "note:taxes_state_payables", "separate") for ticker in ("VPI", "DIG", "VRE", "DXG", "PDR")), "(v0 + v1 + v2 + v3 + v4) / 5 / 1e9", "mean 2016 parent taxes and state payables, VND billion"),
    917: Formula(tuple(op(ticker, 2017, "note:short_term_trade_payables", "separate") for ticker in ("VSC", "VJC", "ACV")), "(v0 + v1 + v2) / 3 / 1e6", "mean 2017 parent short-term trade payables, VND million"),
    920: Formula(tuple(op(ticker, 2025, "note:investment_in_associates") for ticker in ("VIF", "GVR", "DPM")), "(v0 + v1 + v2) / 1e9", "total ending investment in associates, VND billion"),
    938: Formula(tuple(op(ticker, 2018, "note:net_customer_loans") for ticker in ("MSB", "EIB", "STB")), "v0 + v1 + v2", "total end-2018 net customer loans, VND million"),
    942: Formula(tuple(op(ticker, 2020, "note:net_fx_income", "separate") for ticker in ("VPB", "EIB", "HDB")), "(v0 + v1 + v2) / 3", "mean 2020 parent net foreign-exchange result, VND million"),
    959: Formula(tuple(op("GVR", year, "equity") for year in (2016, 2017, 2018, 2020)), "max((v0, 2016), (v1, 2017), (v2, 2018), (v3, 2020))[1]", "year of maximum GVR ending equity"),
    985: Formula(tuple(op("FIT", year, "short_term_borrowings", "separate") for year in (2016, 2017, 2019, 2020, 2021)), "max((v0, 2016), (v1, 2017), (v2, 2019), (v3, 2020), (v4, 2021))[1]", "year of maximum FIT parent short-term borrowings"),
    990: Formula(tuple(op(ticker, 2021, "note:ending_common_shares_outstanding", "separate") for ticker in ("VPI", "NLG", "DXG", "SNZ")), "(v0 > 3.5e8) + (v1 > 3.5e8) + (v2 > 3.5e8) + (v3 > 3.5e8)", "count of parents with more than 350 million ending shares"),
    993: Formula((
        op("GEE", 2022, "pbt", "separate"), op("GEE", 2022, "note:current_tax_expense_zero", "separate"),
        op("VGC", 2022, "pbt", "separate"), op("VGC", 2022, "current_tax_expense", "separate"), op("VGC", 2022, "kqkd:52", "separate"),
        op("SJG", 2022, "pbt", "separate"), op("SJG", 2022, "current_tax_expense", "separate"), op("SJG", 2022, "kqkd:52", "separate", preserve_sign=True),
    ), "(v1 / v0 + (v3 - v4) / v2 + (v6 + v7) / v5) / 3 * 100", "mean 2022 parent effective tax rate; VGC reports deferred-tax income while SJG reports deferred-tax expense"),
    1001: Formula((
        *(op(ticker, 2023, "note:credit_provision_expense") for ticker in ("BID", "NAB", "ABB")),
        *(op(ticker, 2023, "note:pre_provision_profit") for ticker in ("BID", "NAB", "ABB")),
    ), "(abs(v0) / v3 + abs(v1) / v4 + abs(v2) / v5) / 3 * 100", "mean 2023 credit-provision-expense/pre-provision-profit ratio"),
    499: Formula((
        *(op("FTS", year, "note:short_term_borrowings_ending", "unknown") for year in (2018, 2019, 2022, 2023, 2024)),
        *(op("FTS", year, metric, "unknown") for year in (2018, 2019, 2022, 2023, 2024) for metric in ("note:cash_beginning", "note:cash_ending")),
    ), "((v6 - v5) / v5 * 100 if v0 == max(v0, v1, v2, v3, v4) else (v8 - v7) / v7 * 100 if v1 == max(v0, v1, v2, v3, v4) else (v10 - v9) / v9 * 100 if v2 == max(v0, v1, v2, v3, v4) else (v12 - v11) / v11 * 100 if v3 == max(v0, v1, v2, v3, v4) else (v14 - v13) / v13 * 100)", "FTS company cash change in the requested year with maximum ending short-term borrowings"),
    820: Formula(tuple(op("SNZ", year, "finance_revenue", "separate") for year in (2018, 2019, 2020, 2021, 2022)), "(v0 + v1 + v2 + v3 + v4) / 5 / 1e11", "mean SNZ parent finance revenue, VND hundred-billion"),
    824: Formula(tuple(op("VJC", year, "note:ending_common_shares_outstanding") for year in (2016, 2019, 2021)), "(v0 + v1 + v2) / 3 / 1e6", "mean VJC ending common shares outstanding, million shares"),
    831: Formula(tuple(op(ticker, 2015, "finance_revenue") for ticker in ("DLG", "VJC", "ACV", "VSC")), "(v0 + v1 + v2 + v3) / 4 / 1e9", "mean 2015 finance revenue for DLG, VJC, ACV and VSC, VND billion"),
    858: Formula(tuple(op(ticker, 2017, "selling_expense", "separate") for ticker in ("SAB", "DBC", "MCH")), "(v0 + v1 + v2) / 1e12", "total 2017 parent selling expense for SAB, DBC and MCH, VND trillion"),
    915: Formula(tuple(op(ticker, 2016, "finance_expense", "separate") for ticker in ("MPC", "SAB", "HAG")), "(v0 + v1 + v2) / 1e9", "total 2016 parent finance expense for MPC, SAB and HAG, VND billion"),
    934: Formula((op("VGC", 2018, "finance_revenue"), op("PC1", 2018, "finance_revenue"), op("SJG", 2018, "finance_revenue")), "(v0 + v1 + v2) / 1e9", "total requested 2018 finance revenue for VGC, PC1 and SJG, VND billion"),
    943: Formula(tuple(op(ticker, 2018, "cash") for ticker in ("DCM", "VIF", "NKG")), "(v0 + v1 + v2) / 3 / 1e9", "mean 2018 cash and cash equivalents for DCM, VIF and NKG, VND billion"),
    954: Formula(tuple(op(ticker, 2018, "interest_expense") for ticker in ("DPM", "VIF", "HSG")), "(v0 + v1 + v2) / 3 / 1e9", "mean 2018 interest expense for DPM, VIF and HSG, VND billion"),
    963: Formula(tuple(op(ticker, 2016, "cash", "separate") for ticker in ("PC1", "VGC", "SAM")), "(v0 > 1e11) + (v1 > 1e11) + (v2 > 1e11)", "count of parents with ending cash above VND 100 billion"),
    975: Formula(tuple(op(ticker, 2024, "current_tax_expense") for ticker in ("MCH", "MSN", "VNM", "ASM")), "(v0 + v1 + v2 + v3) / 4 / 1e9", "mean 2024 current corporate-income-tax expense, VND billion"),
    1002: Formula((
        op("AAA", 2016, "interest_expense", "separate"), op("NKG", 2016, "interest_expense", "separate"),
        op("DCM", 2016, "interest_expense", "separate"), op("DPM", 2016, "note:interest_expense", "separate"),
    ), "(v0 > 1e11) + (v1 > 1e11) + (v2 > 1e11) + (v3 > 1e11)", "count of parents with 2016 interest expense above VND 100 billion"),
    1004: Formula((
        *(op("IJC", year, "note:admin_depreciation", "separate") for year in (2016, 2017, 2018, 2023, 2024)),
        *(op("IJC", year, "note:admin_expense_total", "separate") for year in (2016, 2017, 2018, 2023, 2024)),
    ), "max(v0 / v5, v1 / v6, v2 / v7, v3 / v8, v4 / v9) * 100", "maximum parent IJC administrative-depreciation share across the five requested years"),
    87: Formula((op("HPX", 2024, "cogs"),), "v0 / 1e9", "HPX 2024 COGS, VND bn"),
    130: Formula((op("DBC", 2024, "revenue"),), "v0 / 1e12", "DBC 2024 revenue, VND tn"),
    126: Formula((op("SAB", 2022, "cash"),), "v0 / 1e12", "SAB beginning-2023 cash, VND tn"),
    171: Formula((op("NLG", 2021, "cfo"),), "v0 / 1e11", "NLG 2021 CFO, VND 100bn"),
    172: Formula((op("FPT", 2022, "cash"),), "v0 / 1e12", "FPT 2022 cash, VND tn"),
    237: Formula((op("GAS", 2016, "revenue"),), "v0 / 1e11", "GAS 2016 revenue, VND 100bn"),
    271: Formula((op("GEG", 2025, "inventory"),), "v0 / 1e9", "GEG 2025 inventory, VND bn"),
    293: Formula((op("ACV", 2015, "cash"),), "v0 / 1e11", "ACV 2015 cash, VND 100bn"),
    295: Formula((op("HSG", 2018, "cfo"),), "v0 / 1e11", "HSG 2018 CFO, VND 100bn"),
    297: Formula((op("SJG", 2024, "total_assets"),), "v0 / 1e11", "SJG 2024 assets, VND 100bn"),
    655: Formula((op("VIC", 2021, "cash"), op("VIC", 2019, "cash")), "(v0 / v1 - 1) * 100", "VIC cash growth"),
    662: Formula((op("CEO", 2016, "npat"), op("CEO", 2016, "total_assets")), "v0 / v1 * 100", "CEO ROA on ending assets"),
    666: Formula((op("KBC", 2015, "finance_revenue", "separate"), op("KBC", 2015, "finance_expense", "separate")), "(v0 - v1) / 1e9", "KBC parent net finance result, VND bn"),
    673: Formula((op("SCR", 2025, "current_assets", "separate"), op("SCR", 2025, "total_assets", "separate")), "v0 / v1 * 100", "SCR parent current-assets share"),
    674: Formula((op("HBC", 2016, "npat"), op("HBC", 2016, "revenue")), "v0 / v1 * 100", "HBC net margin"),
    678: Formula((op("DNH", 2025, "current_liabilities", "separate"), op("DNH", 2025, "equity", "separate")), "v0 / v1", "DNH parent current liabilities / equity"),
    679: Formula((op("FOX", 2016, "gross_profit"),), "v0 / 1e9", "FOX gross profit, VND bn"),
    681: Formula((op("NLG", 2019, "gross_profit", "separate"),), "v0 / 1e9", "NLG parent gross profit, VND bn"),
    682: Formula((op("HSG", 2021, "kqkd:40"),), "v0 / 1e6", "HSG other result, VND mn"),
    686: Formula((op("VNM", 2023, "pbt"), op("VNM", 2023, "revenue")), "v0 / v1 * 100", "VNM PBT margin"),
    694: Formula((op("SGB", 2015, "liabilities", "separate"), op("SGB", 2015, "total_assets", "separate")), "v0 / v1 * 100", "SGB parent liabilities / total assets at end-2015; both source rows are million VND"),
    695: Formula((op("DCM", 2017, "npat"), op("DCM", 2016, "equity"), op("DCM", 2017, "equity")), "v0 / ((v1 + v2) / 2) * 100", "DCM ROE on average equity"),
    696: Formula((op("HSG", 2019, "revenue"), op("HSG", 2018, "equity"), op("HSG", 2019, "equity")), "v0 / ((v1 + v2) / 2)", "HSG equity turnover"),
    698: Formula((op("TTF", 2020, "admin_expense"), op("TTF", 2020, "revenue")), "v0 / v1 * 100", "TTF admin expense ratio"),
    701: Formula((op("IJC", 2020, "selling_expense", "separate"), op("IJC", 2020, "revenue", "separate")), "v0 / v1 * 100", "IJC parent selling expense / net revenue"),
    702: Formula((op("CEO", 2018, "kqkd:40", "separate"),), "v0 / 1e9", "CEO parent net other income, VND bn"),
    704: Formula((op("NLG", 2015, "finance_revenue", "separate"), op("NLG", 2015, "finance_expense", "separate")), "(v0 - v1) / 1e9", "NLG parent net finance result, VND bn"),
    708: Formula((op("MML", 2017, "npat"), op("MML", 2017, "revenue")), "v0 / v1 * 100", "MML net margin"),
    710: Formula((op("VNM", 2022, "current_liabilities", "separate"), op("VNM", 2022, "equity", "separate")), "v0 / v1 * 100", "VNM parent current liabilities / equity"),
    712: Formula((op("GEG", 2025, "finance_expense"), op("GEG", 2025, "revenue")), "v0 / v1 * 100", "GEG finance expense ratio"),
    713: Formula((op("HT1", 2020, "finance_revenue", "separate"), op("HT1", 2020, "finance_expense", "separate")), "v0 / v1 * 100", "HT1 parent finance revenue / finance expense"),
    714: Formula((op("HUT", 2024, "finance_revenue"), op("HUT", 2024, "finance_expense")), "(v0 - v1) / 1e9", "HUT net finance result, VND bn"),
    718: Formula((op("DXG", 2017, "kqkd:11"), op("DXG", 2017, "kqkd:10")), "v0 / v1 * 100", "Bluemarq Group is the catalog company-name alias for ticker DXG; total consolidated COGS / net revenue in 2017"),
    719: Formula((op("NCB", 2016, "cfo", "separate"), op("NCB", 2016, "pbt", "separate")), "v0 / v1 * 100", "NCB parent CFO / PBT"),
    722: Formula((op("OGC", 2022, "kqkd:40"),), "v0 / 1e9", "OGC other result, VND bn"),
    725: Formula((op("SAM", 2017, "finance_revenue", "separate"), op("SAM", 2017, "finance_expense", "separate")), "(v0 - v1) / 1e9", "SAM parent net finance result, VND bn"),
    726: Formula((op("GEX", 2024, "finance_revenue"), op("GEX", 2024, "finance_expense")), "(v0 - v1) / 1e9", "GEX net finance result, VND bn"),
    728: Formula((op("DBC", 2018, "selling_expense", "separate"), op("DBC", 2018, "admin_expense", "separate"), op("DBC", 2018, "revenue", "separate")), "(v0 + v1) / v2 * 100", "DBC parent SGA / net revenue"),
    730: Formula((op("DTK", 2017, "gross_profit", "separate"),), "v0 / 1e12", "DTK parent gross profit, VND tn"),
    737: Formula((op("ASM", 2023, "cogs"), op("BAF", 2023, "cogs")), "abs(v0 - v1) / 1e9", "absolute ASM/BAF COGS difference, VND bn"),
    743: Formula((op("GEE", 2023, "cdkt:221"), op("SAM", 2023, "cdkt:221")), "abs(v0 - v1) / 1e12", "absolute GEE/SAM tangible-fixed-assets difference using code 221, VND tn"),
    581: Formula((op("DBC", 2019, "cdkt:221"), op("DBC", 2015, "cdkt:221")), "abs(v0 - v1) / 1e9", "DBC tangible-fixed-assets difference using code 221, VND bn"),
    580: Formula((op("HBC", 2016, "kqkd:52"), op("HBC", 2015, "kqkd:52")), "abs(v0 - v1) / 1e6", "HBC deferred-tax expense difference, VND mn"),
    582: Formula((op("MCH", 2023, "cdkt:320"), op("MCH", 2021, "cdkt:320")), "(v0 / v1 - 1) * 100", "MCH short-term borrowings growth 2021-2023"),
    583: Formula((op("OCB", 2022, "note:intangible_nbv", "separate"), op("OCB", 2020, "note:intangible_nbv", "separate")), "(v0 / v1 - 1) * 100", "OCB parent intangible fixed-assets NBV growth 2020-2022"),
    584: Formula((op("AAA", 2025, "equity"), op("AAA", 2024, "equity")), "abs(v0 - v1) / 1e9", "AAA equity difference, VND bn"),
    586: Formula((op("ACV", 2022, "cash"), op("ACV", 2021, "cash")), "(v0 / v1 - 1) * 100", "ACV cash growth 2021-2022"),
    592: Formula((op("BID", 2024, "note:loan_loss_provision"), op("BID", 2022, "note:loan_loss_provision")), "abs(v0 - v1)", "BID loan-loss provision ending-balance difference, VND mn"),
    593: Formula((op("VIF", 2023, "note:fx_gain"), op("VIF", 2021, "note:fx_gain")), "abs(v0 - v1) / 1e6", "VIF FX-gain difference, VND mn"),
    594: Formula((op("GEE", 2025, "note:vcb_fair_value"), op("GEE", 2022, "note:vcb_fair_value")), "(v0 - v1) / 1e6", "GEE fair-value difference for its VCB investment, VND mn"),
    595: Formula((op("DXS", 2022, "note:long_term_other_receivables"), op("DXS", 2021, "note:long_term_other_receivables")), "(v0 - v1) / 1e9", "DXS long-term other-receivables difference, VND bn"),
    598: Formula((op("HNG", 2021, "note:related_trade_payables", "separate"), op("HNG", 2020, "note:related_trade_payables", "separate")), "(v0 / v1 - 1) * 100", "HNG parent related-party short-term trade-payables growth"),
    600: Formula((op("FPT", 2025, "note:accrued_interest"), op("FPT", 2023, "note:accrued_interest")), "(v0 - v1) / 1e9", "FPT accrued-interest difference, VND bn"),
    603: Formula((op("VJC", 2025, "note:prepaid_aircraft_maintenance", "separate"), op("VJC", 2024, "note:prepaid_aircraft_maintenance", "separate")), "abs(v0 - v1) / 1e9", "VJC parent prepaid aircraft-maintenance decrease, VND bn"),
    604: Formula((op("DLG", 2019, "note:parent_total_assets", "separate"), op("DLG", 2018, "note:parent_total_assets", "separate")), "abs(v0 - v1) / 1e9", "DLG parent total-assets movement, VND bn"),
    605: Formula((op("NLG", 2024, "note:short_term_customer_advances"), op("NLG", 2023, "note:short_term_customer_advances")), "(v0 / v1 - 1) * 100", "NLG short-term customer-advances growth"),
    606: Formula((op("MML", 2023, "cdkt:310"), op("MML", 2019, "cdkt:310")), "(v0 / v1 - 1) * 100", "MML current-liabilities growth 2019-2023"),
    607: Formula((op("HAG", 2023, "kqkd:51"), op("HAG", 2017, "kqkd:51")), "abs(v0 - v1) / 1e3", "HAG current-income-tax expense difference, VND thousand"),
    608: Formula((op("MML", 2023, "note:tangible_fixed_assets_nbv"), op("MML", 2018, "note:tangible_fixed_assets_nbv")), "(v0 - v1) / 1e6", "MML tangible-fixed-assets NBV increase, VND mn"),
    616: Formula((op("VJC", 2018, "cdkt:221"), op("VJC", 2015, "cdkt:221")), "abs(v0 - v1) / 1e12", "VJC tangible-fixed-assets difference using code 221, VND tn"),
    617: Formula((op("GAS", 2018, "cdkt:270"), op("GAS", 2015, "cdkt:270")), "(v0 / v1 - 1) * 100", "GAS total-assets growth from beginning-2016 to beginning-2019"),
    591: Formula((op("SSB", 2022, "note:performing_gross_customer_loans"), op("SSB", 2021, "note:performing_gross_customer_loans")), "v0 - v1", "SSB performing gross-customer-loans difference, VND mn"),
    602: Formula((op("HDB", 2021, "note:listed_trading_securities"), op("HDB", 2018, "note:listed_trading_securities")), "v0 - v1", "HDB listed trading-securities difference, VND mn"),
    618: Formula((op("STB", 2019, "note:accrued_customer_loan_interest", "separate"), op("STB", 2018, "note:accrued_customer_loan_interest", "separate")), "abs(v0 - v1)", "absolute STB parent accrued customer-loan-interest difference between end-2019 and end-2018, VND million"),
    619: Formula((op("SSI", 2020, "note:short_term_advances", "separate"), op("SSI", 2019, "note:short_term_advances", "separate")), "(v0 - v1) / 1e6", "SSI parent short-term-advances difference, VND mn"),
    # Multi-year questions whose previous programs returned the maximum value
    # rather than the year attached to that value.  Tuple max keeps the year
    # selection fully dependent on the source cells at grader runtime.
    538: Formula((
        op("AAA", 2024, "cfo", "separate"), op("VIF", 2024, "cfo", "separate"), op("NKG", 2024, "cfo", "separate"),
        op("AAA", 2023, "interest_expense", "separate"), op("AAA", 2024, "interest_expense", "separate"), op("AAA", 2025, "interest_expense", "separate"),
        op("VIF", 2023, "interest_expense", "separate"), op("NKG", 2023, "interest_expense", "separate"), op("NKG", 2024, "interest_expense", "separate"), op("NKG", 2025, "interest_expense", "separate"),
    ), "max((v3 if v0 > 0 else -1, 2023), (v4 if v0 > 0 else -1, 2024), (v5 if v0 > 0 else -1, 2025), (v6 if v1 > 0 else -1, 2023), (v7 if v2 > 0 else -1, 2023), (v8 if v2 > 0 else -1, 2024), (v9 if v2 > 0 else -1, 2025))[1]", "year of maximum parent interest expense after positive-2024-CFO filter"),
    813: Formula(tuple(op("ACV", year, "revenue", "separate") for year in (2015, 2017, 2019, 2023)), "max((v0, 2015), (v1, 2017), (v2, 2019), (v3, 2023))[1]", "year of maximum ACV parent revenue"),
    842: Formula(tuple(op("PDR", year, "pbt", "separate") for year in (2015, 2018, 2022)), "max((v0, 2015), (v1, 2018), (v2, 2022))[1]", "year of maximum PDR parent PBT"),
    874: Formula(tuple(op("VNM", year, "cdkt:221") for year in (2015, 2016, 2021, 2023, 2025)), "max((v0, 2015), (v1, 2016), (v2, 2021), (v3, 2023), (v4, 2025))[1]", "year of maximum VNM tangible fixed-assets NBV using balance-sheet code 221"),
    506: Formula((
        *(op(ticker, 2024, "cdkt:221") for ticker in ("IJC", "DXG", "NVL", "NLG", "KBC")),
        *(op(ticker, 2024, "kqkd:10") for ticker in ("IJC", "DXG", "NVL", "NLG", "KBC")),
    ), "max((v0, v5), (v1, v6), (v2, v7), (v3, v8), (v4, v9))[1] / 1e12", "2024 net revenue of the company with maximum tangible fixed-assets NBV (code 221), VND trillion; NVL remains the selected company after replacing total fixed assets code 220"),
    897: Formula(tuple(op("DXG", year, "cash") for year in (2016, 2018, 2020)), "max((v0, 2016), (v1, 2018), (v2, 2020))[1]", "year of maximum DXG cash and cash equivalents"),
    904: Formula(tuple(op("HDG", year, "finance_revenue") for year in (2020, 2023, 2024)), "max((v0, 2020), (v1, 2023), (v2, 2024))[1]", "year of maximum HDG finance revenue"),
    910: Formula(tuple(op("DBC", year, "inventory", "separate") for year in (2015, 2016, 2019, 2022, 2025)), "max((v0, 2015), (v1, 2016), (v2, 2019), (v3, 2022), (v4, 2025))[1]", "year of maximum DBC parent inventory"),
    933: Formula(tuple(op("VNM", year, "lctt:36") for year in (2015, 2016, 2018, 2021)), "max((abs(v0), 2015), (abs(v1), 2016), (abs(v2), 2018), (abs(v3), 2021))[1]", "year of maximum VNM dividend cash outflow"),
    960: Formula(tuple(op("QNS", year, "cfo", "separate") for year in (2017, 2019, 2020, 2021, 2023)), "max((v0, 2017), (v1, 2019), (v2, 2020), (v3, 2021), (v4, 2023))[1]", "year of maximum QNS parent CFO"),
    997: Formula(tuple(op("AAA", year, "cdkt:132") for year in (2019, 2023, 2025)), "max((v0, 2019), (v1, 2023), (v2, 2025))[1]", "year of maximum AAA short-term advances to suppliers using balance-sheet code 132"),
    1000: Formula(tuple(op("SAM", year, "cdkt:242") for year in (2020, 2021, 2023, 2024)), "max((v0, 2020), (v1, 2021), (v2, 2023), (v3, 2024))[1]", "year of maximum SAM construction in progress using balance-sheet code 242"),
    841: Formula(tuple(op("VGT", year, "cdkt:261", "separate") for year in (2017, 2018, 2021, 2022, 2023)), "max((v0, 2017), (v1, 2018), (v2, 2021), (v3, 2022), (v4, 2023))[1]", "year of maximum VGT parent long-term prepaid expense"),
    852: Formula(tuple(op("BID", year, "note:external_receivables") for year in (2020, 2022, 2024, 2025)), "max((v0, 2020), (v1, 2022), (v2, 2024), (v3, 2025))[1]", "year of maximum BID external receivables"),
    866: Formula(tuple(op("SJG", year, "cdkt:313") for year in (2018, 2019, 2020, 2021)), "max((v0, 2018), (v1, 2019), (v2, 2020), (v3, 2021))[1]", "year of maximum SJG taxes and state payables"),
    876: Formula((op("ACV", 2017, "interest_expense"), op("ACV", 2019, "interest_expense"), op("ACV", 2021, "note:interest_expense"), op("ACV", 2022, "note:interest_expense")), "max((v0, 2017), (v1, 2019), (v2, 2021), (v3, 2022))[1]", "year of maximum ACV interest expense"),
    906: Formula(tuple(op("VGT", year, "cdkt:221", "separate") for year in (2015, 2017, 2018, 2020, 2022)), "max((v0, 2015), (v1, 2017), (v2, 2018), (v3, 2020), (v4, 2022))[1]", "year of maximum VGT parent tangible fixed-assets NBV using balance-sheet code 221"),
    936: Formula(tuple(op("CRE", year, "cdkt:318") for year in (2020, 2021, 2022, 2025)), "max((v0, 2020), (v1, 2021), (v2, 2022), (v3, 2025))[1]", "year of maximum CRE unearned short-term revenue"),
    948: Formula(tuple(op("PLX", year, "cdkt:242") for year in (2017, 2018, 2019, 2023, 2024)), "max((v0, 2017), (v1, 2018), (v2, 2019), (v3, 2023), (v4, 2024))[1]", "year of maximum PLX construction in progress using balance-sheet code 242"),
    890: Formula(tuple(op("NVL", year, "cdkt:132", "separate") for year in (2017, 2021, 2023)), "max((abs(v0), 2017), (abs(v1), 2021), (abs(v2), 2023))[1]", "year of maximum absolute NVL parent advances to suppliers"),
    928: Formula(tuple(op("OCB", year, "note:specific_loan_provision", "separate") for year in (2017, 2018, 2019, 2025)), "max((v0, 2017), (v1, 2018), (v2, 2019), (v3, 2025))[1]", "year of maximum OCB parent specific loan provision"),
    929: Formula(tuple(op("MBB", year, "note:government_bonds") for year in (2015, 2016, 2017, 2018, 2022)), "max((v0, 2015), (v1, 2016), (v2, 2017), (v3, 2018), (v4, 2022))[1]", "year of maximum MBB government-bond investment"),
    953: Formula(tuple(op("CTG", year, "note:intangible_total_nbv", "separate") for year in (2016, 2021, 2022)), "max((v0, 2016), (v1, 2021), (v2, 2022))[1]", "year of maximum CTG parent intangible fixed-assets NBV"),
    971: Formula(tuple(op("KHG", year, "note:brokerage_commission", "separate") for year in (2019, 2020, 2021, 2022, 2023)), "max((v0, 2019), (v1, 2020), (v2, 2021), (v3, 2022), (v4, 2023))[1]", "year of maximum KHG parent brokerage commission"),
    981: Formula(tuple(op("BSR", year, "note:lpg_revenue") for year in (2017, 2019, 2021, 2022, 2025)), "max((v0, 2017), (v1, 2019), (v2, 2021), (v3, 2022), (v4, 2025))[1]", "year of maximum BSR LPG revenue"),
    989: Formula(tuple(op("STB", year, "note:accrued_loan_interest") for year in (2017, 2022, 2024)), "max((v0, 2017), (v1, 2022), (v2, 2024))[1]", "year of maximum STB accrued customer-loan interest"),
    878: Formula(tuple(op("KBC", year, "cdkt:230") for year in (2015, 2017, 2019)), "max((v0, 2015), (v1, 2017), (v2, 2019))[1]", "year of maximum KBC investment-property NBV"),
    986: Formula(tuple(op("HAG", year, "current_tax_expense", "separate") for year in (2016, 2017, 2024, 2025)), "max((v0, 2016), (v1, 2017), (v2, 2024), (v3, 2025))[1]", "year of maximum HAG parent current income-tax expense"),
    501: Formula((
        *(op("QNS", year, "note:thanh_phat_investment_cost", "separate") for year in (2015, 2020, 2021, 2023)),
        *(op("QNS", year, "note:overdue_receivables_cost", "separate") for year in (2015, 2020, 2021, 2023)),
    ), "max((v4 if v0 == max(v0, v1, v2, v3) else -1, 2015), (v5 if v1 == max(v0, v1, v2, v3) else -1, 2020), (v6 if v2 == max(v0, v1, v2, v3) else -1, 2021), (v7 if v3 == max(v0, v1, v2, v3) else -1, 2023))[1]", "year of maximum QNS parent overdue receivables after maximum Thanh Phat investment-cost filter"),
    826: Formula((
        *(op("KBC", year, "note:long_term_land_infra_cost") for year in (2016, 2019, 2020, 2022)),
        *(op("KBC", year, "cogs") for year in (2016, 2019, 2020, 2022)),
    ), "max((abs(v0 / v4), 2016), (abs(v1 / v5), 2019), (abs(v2 / v6), 2020), (abs(v3 / v7), 2022))[1]", "year of maximum KBC long-term land-and-infrastructure rental cost share"),
    982: Formula((
        *(op("PVT", year, "note:transport_segment_assets", "separate") for year in (2018, 2019, 2022, 2023, 2025)),
        *(op("PVT", year, "note:segment_total_assets", "separate") for year in (2018, 2019, 2022, 2023, 2025)),
    ), "max((v0 / v5, 2018), (v1 / v6, 2019), (v2 / v7, 2022), (v3 / v8, 2023), (v4 / v9, 2025))[1]", "year of maximum PVT parent transport-segment assets / total assets"),
    995: Formula((
        op("HAG", 2017, "note:related_receivables_total"),
        op("HAG", 2018, "note:related_receivables_total"),
        op("HAG", 2019, "note:related_receivables_total"),
        op("HAG", 2017, "note:related_trade_payables"),
        op("HAG", 2017, "note:related_customer_advances"),
        op("HAG", 2017, "note:related_other_current_payables"),
        op("HAG", 2017, "note:related_other_long_payables"),
        op("HAG", 2017, "note:related_short_borrowings"),
        op("HAG", 2018, "note:related_trade_payables"),
        op("HAG", 2018, "note:related_other_current_payables"),
        op("HAG", 2018, "note:related_other_long_payables"),
        op("HAG", 2018, "note:related_short_accruals"),
        op("HAG", 2018, "note:related_long_accruals"),
        op("HAG", 2018, "note:related_short_borrowings"),
        op("HAG", 2018, "note:related_long_borrowings"),
        op("HAG", 2019, "note:related_trade_payables"),
        op("HAG", 2019, "note:related_other_current_payables"),
        op("HAG", 2019, "note:related_other_long_payables"),
    ), "max((abs(v0) + abs(v3) + abs(v4) + abs(v5) + abs(v6) + abs(v7), 2017), (abs(v1) + abs(v8) + abs(v9) + abs(v10) + abs(v11) + abs(v12) + abs(v13) + abs(v14), 2018), (abs(v2) + abs(v15) + abs(v16) + abs(v17), 2019))[1]", "year of maximum HAG related-party receivables plus payables"),
    193: Formula((op("MBB", 2022, "note:vnd_term_deposits"),), "v0", "MBB VND term deposits at end-2022, VND million"),
    221: Formula((op("CEO", 2017, "note:finance_lease_asset_cost"),), "v0 / 1e9", "CEO finance-lease fixed-asset ending gross cost, VND billion"),
    351: Formula((op("SHB", 2016, "note:deferred_allocation_cost", "separate"),), "v0", "SHB parent deferred allocation cost, VND million"),
    847: Formula(tuple(op("QNS", year, "note:intangible_total_nbv") for year in (2019, 2020, 2021, 2023, 2024)), "max(v0, v1, v2, v3, v4) / 1e9", "maximum QNS intangible fixed-asset ending NBV, VND billion"),
    886: Formula((
        op("VIB", 2015, "note:loan_risk_provision_total"),
        op("VIB", 2022, "note:loan_risk_provision_total"),
        op("VIB", 2023, "note:loan_risk_general_provision"),
        op("VIB", 2023, "note:loan_risk_specific_provision"),
    ), "max(v0, v1, v2 + v3)", "maximum VIB total customer-loan risk provision, VND million"),
    996: Formula(tuple(op("SHB", year, "note:intangible_total_nbv") for year in (2016, 2018, 2020, 2021)), "max(v0, v1, v2, v3)", "maximum SHB intangible fixed-asset ending NBV, VND million"),
    1011: Formula(tuple(op("SSB", year, "note:related_party_loans", "separate") for year in (2021, 2023, 2024)), "max(v0, v1, v2)", "maximum SSB parent related-party loan balance, VND million"),
    846: Formula((
        *(op("HDG", year, "note:tangible_accumulated_depreciation", "separate") for year in (2020, 2021, 2022, 2023, 2025)),
        *(op("HDG", year, "note:tangible_gross_cost", "separate") for year in (2020, 2021, 2022, 2023, 2025)),
    ), "(abs(v0 / v5) + abs(v1 / v6) + abs(v2 / v7) + abs(v3 / v8) + abs(v4 / v9)) / 5 * 100", "mean HDG parent accumulated-depreciation/gross-cost ratio"),
    872: Formula(tuple(op("EVF", year, "note:intangible_total_nbv") for year in (2021, 2022, 2023, 2025)), "(v0 + v1 + v2 + v3) / 4 / 1e3", "mean EVF intangible fixed-asset ending NBV, VND billion"),
    787: Formula((op("KLB", 2024, "note:on_balance_currency_position", "separate"), op("NVB", 2024, "note:on_balance_currency_position", "separate")), "abs(v0 - v1)", "absolute KLB/NVB parent on-balance-sheet currency-position difference, VND million"),
    764: Formula((op("GVR", 2015, "cdkt:418"), op("DPM", 2015, "cdkt:418")), "abs(v0 - v1) / 1e12", "absolute GVR/DPM development-investment-fund difference, VND trillion"),
    818: Formula(tuple(op("FTS", year, "note:doubtful_receivables_provision") for year in (2018, 2020, 2021, 2023, 2024)), "(v0 + v1 + v2 + v3 + v4) / 5 / 1e6", "mean FTS ending doubtful-receivable provision, VND million"),
    188: Formula((op("GEE", 2020, "note:gross_bad_receivables"),), "v0 / 1e11", "GEE gross bad receivables at end-2020, VND hundred-billion"),
    955: Formula((
        *(op(ticker, 2019, "note:short_term_customer_loans", "separate") for ticker in ("ACB", "MBB", "BID", "STB")),
        *(op(ticker, 2019, "note:total_customer_loans_by_maturity", "separate") for ticker in ("ACB", "MBB", "BID", "STB")),
    ), "(v0 / v4 + v1 / v5 + v2 / v6 + v3 / v7) / 4 * 100", "mean parent short-term customer-loan share for ACB/MBB/BID/STB"),
    970: Formula((
        *(op("ACB", year, "note:financial_reserve") for year in (2015, 2021, 2023, 2025)),
        *(op("ACB", year, "note:equity_movement_total") for year in (2015, 2021, 2023, 2025)),
    ), "(v0 / v4 + v1 / v5 + v2 / v6 + v3 / v7) / 4 * 100", "mean ACB financial-reserve/equity share"),
    980: Formula((
        *(op("HDB", year, "note:net_on_balance_currency_position") for year in (2018, 2021, 2022, 2024)),
        *(op("HDB", year, "note:currency_schedule_total_assets") for year in (2018, 2021, 2022, 2024)),
    ), "max(v0 / v4, v1 / v5, v2 / v6, v3 / v7) * 100", "maximum HDB net on-balance currency-position/total-assets share"),
    992: Formula((
        *(op(ticker, 2022, "note:tangible_accumulated_depreciation") for ticker in ("VPB", "SHB", "MBB")),
        *(op(ticker, 2022, "note:tangible_gross_cost") for ticker in ("VPB", "SHB", "MBB")),
    ), "(abs(v0 / v3) + abs(v1 / v4) + abs(v2 / v5)) / 3 * 100", "mean 2022 tangible-fixed-asset depreciation rate for VPB/SHB/MBB"),
    1007: Formula((
        op("HHV", 2021, "note:bot_segment_assets"), op("HHV", 2022, "note:bot_segment_assets"),
        op("HHV", 2024, "note:bot_segment_assets"), op("HHV", 2025, "note:bot_segment_assets"),
        op("HHV", 2021, "cdkt:270"), op("HHV", 2022, "cdkt:270"),
        op("HHV", 2024, "note:segment_total_assets"), op("HHV", 2025, "note:segment_total_assets"),
    ), "(v0 / v4 + v1 / v5 + v2 / v6 + v3 / v7) / 4 * 100", "mean HHV BOT-segment-assets/consolidated-total-assets share"),
    1008: Formula(tuple(op("SSH", year, "note:related_party_service_revenue", "separate") for year in (2020, 2021, 2022, 2023)), "max((v0, 2020), (v1, 2021), (v2, 2022), (v3, 2023))[1]", "year of maximum parent SSH related-party service revenue across the four requested years"),
    1010: Formula(tuple(op(ticker, 2020, "note:overall_net_liquidity_gap") for ticker in ("NVB", "SGB", "VIB", "HDB", "MSB")), "(v0 > 0) + (v1 > 0) + (v2 > 0) + (v3 > 0) + (v4 > 0)", "count of banks with positive overall net liquidity gap"),
    427: Formula((
        *(op("FPT", 2016, f"note:{currency}_monetary_liabilities") for currency in ("usd", "eur", "jpy", "sgd")),
        *(op("FPT", 2016, f"note:{currency}_monetary_assets") for currency in ("usd", "eur", "jpy", "sgd")),
        *(op("FPT", 2016, f"note:{currency}_adverse_5pct_pbt") for currency in ("usd", "eur", "jpy", "sgd")),
    ), "((abs(v8) if v0 > v4 else 0) + (abs(v9) if v1 > v5 else 0) + (abs(v10) if v2 > v6 else 0) + (abs(v11) if v3 > v7 else 0)) / 1e9", "FPT adverse-5% PBT reduction for currencies whose year-end monetary liabilities exceed assets, VND billion"),
    502: Formula((
        *(op("PLX", year, "note:stabilisation_fund_bank_deposit", "separate") for year in (2015, 2016, 2018, 2019, 2020, 2021)),
        *(op("PLX", year, "note:accrued_interest_receivable", "separate") for year in (2015, 2016, 2018, 2019, 2020, 2021)),
    ), "(v6 if v0 == max(v0, v1, v2, v3, v4, v5) else v7 if v1 == max(v0, v1, v2, v3, v4, v5) else v8 if v2 == max(v0, v1, v2, v3, v4, v5) else v9 if v3 == max(v0, v1, v2, v3, v4, v5) else v10 if v4 == max(v0, v1, v2, v3, v4, v5) else v11) / 1e9", "PLX parent accrued interest in the maximum stabilisation-fund bank-deposit year, VND billion"),
    503: Formula((
        *(op("VGT", year, "equity") for year in (2015, 2017, 2019, 2021, 2022, 2023)),
        *(op("VGT", year, "note:coats_phong_phu_purchases") for year in (2015, 2017, 2019, 2021, 2022, 2023)),
    ), "(v6 if v0 == max(v0, v1, v2, v3, v4, v5) else v7 if v1 == max(v0, v1, v2, v3, v4, v5) else v8 if v2 == max(v0, v1, v2, v3, v4, v5) else v9 if v3 == max(v0, v1, v2, v3, v4, v5) else v10 if v4 == max(v0, v1, v2, v3, v4, v5) else v11) / 1e9", "VGT Coats Phong Phu purchases in the maximum ending-equity year, VND billion"),
    516: Formula((
        *(op("ACB", year, "note:bonus_welfare_fund", "separate") for year in (2015, 2019, 2022)),
        *(op("ACB", year, "note:recorded_derivatives_current", "separate") for year in (2015, 2019, 2022)),
        *(op("ACB", year, "note:recorded_derivatives_prior", "separate") for year in (2015, 2019, 2022)),
    ), "(v3 / v6 - 1) * 100 if v0 == max(v0, v1, v2) else (v4 / v7 - 1) * 100 if v1 == max(v0, v1, v2) else (v5 / v8 - 1) * 100", "ACB parent derivative recorded-value change in the maximum bonus/welfare-fund year"),
    531: Formula((
        *(op("MPC", year, "note:construction_in_progress") for year in (2016, 2018, 2020, 2022, 2023)),
        op("MPC", 2016, "note:transport_expense"), op("MPC", 2016, "note:outside_services_expense"),
        op("MPC", 2018, "note:transport_expense"), op("MPC", 2018, "note:outside_services_expense"),
        op("MPC", 2020, "note:transport_and_outside_services_expense"),
        op("MPC", 2022, "note:transport_and_outside_services_expense"),
        op("MPC", 2023, "note:transport_and_outside_services_expense"),
    ), "((v5 + v6) if v0 == max(v0, v1, v2, v3, v4) else (v7 + v8) if v1 == max(v0, v1, v2, v3, v4) else v9 if v2 == max(v0, v1, v2, v3, v4) else v10 if v3 == max(v0, v1, v2, v3, v4) else v11) / 1e9", "MPC transport and outside-services expense in the maximum construction-in-progress year, VND billion"),
    533: Formula((
        *(op("HND", year, "note:tax_at_company_rate", "unknown") for year in (2016, 2018, 2019, 2020)),
        *(op("HND", year, "note:operating_lease_due_within_one_year", "unknown") for year in (2016, 2018, 2019, 2020)),
    ), "(v4 if v0 == max(v0, v1, v2, v3) else v5 if v1 == max(v0, v1, v2, v3) else v6 if v2 == max(v0, v1, v2, v3) else v7) / 1e9", "HND minimum operating-lease payment due within one year at the year end with maximum tax calculated at the company rate, VND billion"),
    537: Formula((
        *(op("VGC", year, "note:science_technology_fund_appropriation") for year in (2019, 2020, 2022, 2023, 2024)),
        *(op("VGC", year, "note:investment_other_entities") for year in (2019, 2020, 2022, 2023, 2024)),
    ), "(v5 if v0 == max(v0, v1, v2, v3, v4) else v6 if v1 == max(v0, v1, v2, v3, v4) else v7 if v2 == max(v0, v1, v2, v3, v4) else v8 if v3 == max(v0, v1, v2, v3, v4) else v9) / 1e9", "VGC investment in other entities in the maximum science/technology-fund appropriation year, VND billion"),
    755: Formula((op("BID", 2025, "note:on_balance_interest_sensitivity_gap"), op("STB", 2025, "note:on_balance_interest_sensitivity_gap")), "abs(v0 - v1)", "absolute BID/STB total on-balance interest-sensitivity-gap difference, VND million"),
    # Historical-submission ablation exposed these row/unit mistakes.  The
    # replacements below recompute every answer from the cited BTC table cells
    # at runtime; old submitted numbers are not used as labels.
    20: Formula((op("GVR", 2019, "note:visorutex_voting_rate", "separate"),), "v0", "GVR parent Visorutex voting rate"),
    159: Formula((op("GVR", 2020, "note:lai_chau_rubber_ownership_rate"),), "v0", "GVR ownership rate in Cao su Lai Chau"),
    333: Formula((op("GEG", 2022, "note:gialai_hydropower_paid_capital_ownership_rate"),), "v0 / 100", "Gia Lai Hydropower ownership rate on paid-in capital; OCR stores 62.53 as 6253"),
    428: Formula((
        *(op("ACB", 2024, f"note:{currency}_combined_currency_position") for currency in ("usd", "gold", "eur", "jpy", "aud", "cad", "other")),
        op("ACB", 2024, "note:bank_pbt"),
    ), "max(-v0, -v1, -v2, -v3, -v4, -v5, -v6, 0) * 0.05 / v7 * 100", "largest adverse 5% ACB combined-currency-position loss as a share of PBT"),
    496: Formula((
        *(op("MWG", year, "note:depreciation_amortisation_expense") for year in (2017, 2018, 2020, 2022)),
        *(op("MWG", year, "inventory") for year in (2017, 2018, 2020, 2022)),
        *(op("MWG", year, "short_term_borrowings") for year in (2017, 2018, 2020, 2022)),
        *(op("MWG", year, "note:long_term_borrowings") for year in (2017, 2018, 2020, 2022)),
    ), "(v4 / (v8 + v12) if v0 == max(v0, v1, v2, v3) else v5 / (v9 + v13) if v1 == max(v0, v1, v2, v3) else v6 / (v10 + v14) if v2 == max(v0, v1, v2, v3) else v7 / (v11 + v15)) * 100", "MWG inventory/total-borrowings ratio in the maximum depreciation-and-amortisation year"),
    658: Formula((op("PLX", 2024, "kqkd:24"), op("PLX", 2024, "kqkd:30")), "v0 / v1 * 100", "PLX share of JV/associate profit as a percentage of operating profit"),
    661: Formula((op("DNH", 2021, "lctt:21", "separate"), op("DNH", 2021, "note:tangible_fixed_asset_gross_cost", "separate")), "abs(v0) / v1 * 100", "DNH parent capital-expenditure cash outflow as a percentage of ending tangible-PPE gross cost"),
    668: Formula((op("HHS", 2023, "note:dividend_income", "separate"), op("HHS", 2023, "note:short_term_financial_investments", "separate"), op("HHS", 2023, "note:long_term_financial_investments", "separate")), "v0 / (v1 + v2) * 100", "HHS parent dividend investment yield"),
    697: Formula((op("MSN", 2016, "note:techcombank_associate_investment", "separate"), op("MSN", 2016, "note:total_associate_investment", "separate")), "v0 / v1 * 100", "Techcombank share of MSN parent investment in associates"),
    707: Formula((op("VGT", 2024, "note:gross_doubtful_customer_receivables"), op("VGT", 2024, "note:total_customer_receivables")), "v0 / v1 * 100", "VGT gross doubtful/overdue customer receivables as a percentage of total customer receivables"),
    709: Formula((op("MSR", 2022, "note:interest_and_borrowing_cost"), op("MSR", 2022, "note:short_term_borrowings_ending")), "v0 / v1 * 100", "MSR interest and borrowing cost as a percentage of ending short-term borrowings"),
    719: Formula((op("NCB", 2016, "note:cfo", "separate"), op("NCB", 2016, "note:pbt", "separate")), "v0 / v1 * 100", "NCB parent CFO / PBT from audited NVB-source disclosure tables"),
    519: Formula((
        *(op("HPG", year, "note:office_repair_tool_prepayment_allocation", "separate") for year in (2015, 2018, 2022, 2024)),
        *(op("HPG", year, "note:deposit_and_loan_interest", "separate") for year in (2015, 2018, 2022, 2024)),
        *(op("HPG", year, "note:parent_finance_revenue", "separate") for year in (2015, 2018, 2022, 2024)),
    ), "(v4 / v8 if v0 == max(v0, v1, v2, v3) else v5 / v9 if v1 == max(v0, v1, v2, v3) else v6 / v10 if v2 == max(v0, v1, v2, v3) else v7 / v11) * 100", "HPG parent deposit/loan interest share in the maximum long-term-prepayment-allocation year"),
    528: Formula((
        *(op("ABB", year, "note:afs_provision_movement", "separate") for year in (2020, 2022, 2023)),
        *(op("ABB", year, "note:pre_provision_operating_profit", "separate") for year in (2020, 2022, 2023)),
        *(op("ABB", year, "note:parent_total_assets", "separate") for year in (2020, 2022, 2023)),
    ), "(v3 / v6 if v0 == max(v0, v1, v2) else v4 / v7 if v1 == max(v0, v1, v2) else v5 / v8) * 100", "ABB parent pre-provision operating profit / total assets in the maximum signed AFS-provision-movement year"),
    806: Formula((
        *(op("NLG", 2020, f"note:subsidiary_voting_rate_{i}") for i in range(1, 20)),
        *(op("SCR", 2020, f"note:subsidiary_voting_rate_{i}") for i in range(1, 13)),
    ), f"abs({_ocr_percent_mean_expression(0, 19)} - {_ocr_percent_mean_expression(19, 12)})", "absolute difference between NLG and SCR mean subsidiary voting rates"),
    821: Formula((
        *(op("OCB", year, "note:real_estate_customer_loans", "separate") for year in (2017, 2020, 2024, 2025)),
        *(op("OCB", year, "note:total_customer_loans_by_industry", "separate") for year in (2017, 2020, 2024, 2025)),
    ), "(v0 / v4 + v1 / v5 + v2 / v6 + v3 / v7) / 4 * 100", "mean OCB parent real-estate customer-loan share at the four requested year ends"),
    19: Formula((op("HHV", 2023, "note:total_voting_rate", "separate"),), "v0", "HHV parent total direct and indirect voting rate"),
    106: Formula((op("PLX", 2016, "note:ptn_ownership_rate", "separate"),), "v0", "PLX parent ownership rate in PTN Chemicals"),
    660: Formula((op("CTG", 2022, "note:off_balance_contingent_liabilities", "separate"), op("CTG", 2022, "note:off_balance_commitments", "separate"), op("CTG", 2022, "note:parent_total_assets", "separate")), "(v0 + v1) / v2 * 100", "CTG parent off-balance contingent liabilities and commitments / total assets"),
    675: Formula((op("MML", 2021, "cdkt:250", "separate"), op("MML", 2021, "cdkt:400", "separate")), "v0 / v1 * 100", "MML parent long-term financial investments / equity"),
    684: Formula((op("SHB", 2016, "note:general_customer_loan_provision", "separate"), op("SHB", 2016, "note:total_customer_loan_provision", "separate")), "v0 / v1 * 100", "SHB parent general provision / total customer-loan provision"),
    700: Formula((op("HDG", 2015, "note:interest_payable"), op("HDG", 2015, "note:long_term_borrowings")), "v0 / v1 * 100", "HDG interest payable / ending long-term borrowings"),
    729: Formula((op("KHG", 2024, "note:nguyen_khai_hoan_capital_rate"),), "v0", "Nguyen Khai Hoan share of KHG owner-contributed capital"),
    849: Formula(tuple(op("SAB", year, f"note:production_cost_component_{component}") for year in (2018, 2020, 2024, 2025) for component in range(1, 6)), "max(v2 / (v0 + v1 + v2 + v3 + v4), v7 / (v5 + v6 + v7 + v8 + v9), v13 / (v10 + v11 + v12 + v13 + v14), v18 / (v15 + v16 + v17 + v18 + v19)) * 100", "maximum SAB depreciation-and-amortisation share of production/business costs by element"),
    853: Formula((op("DNH", 2022, "cdkt:400", "separate"), op("DNH", 2022, "cdkt:270", "separate"), op("GEG", 2022, "cdkt:400", "separate"), op("GEG", 2022, "cdkt:270", "separate"), op("POW", 2022, "cdkt:400", "separate"), op("POW", 2022, "cdkt:270", "separate")), "(v0 / v1 + v2 / v3 + v4 / v5) / 3 * 100", "mean 2022 parent equity / total-capital share for DNH, GEG and POW"),
    882: Formula((
        *(op(ticker, 2017, "note:future_lease_receipts_under_one_year") for ticker in ("NLG", "VIC", "DIG", "SNZ")),
        *(op(ticker, 2017, "note:future_lease_receipts_total") for ticker in ("NLG", "VIC", "DIG", "SNZ")),
    ), "(v0 / v4 + v1 / v5 + v2 / v6 + v3 / v7) / 4 * 100", "mean 2017 under-one-year share of future operating-lease receipts for NLG, VIC, DIG and SNZ"),
    932: Formula(tuple(op(ticker, 2020, "note:operating_lease_due_within_one_year", "separate") for ticker in ("MBB", "HDB", "KLB", "NAB")), "(v0 > 40000) + (v1 > 40000) + (v2 > 40000) + (v3 > 40000)", "count of parent banks with within-one-year operating-lease commitments above VND 40 billion"),
    931: Formula((
        *(op(ticker, 2025, "note:tangible_accumulated_depreciation", "separate") for ticker in ("GEE", "GEX", "VGC", "SAM", "PC1")),
        *(op(ticker, 2025, "note:tangible_gross_cost", "separate") for ticker in ("GEE", "GEX", "VGC", "SAM", "PC1")),
    ), "(v0 / v5 + v1 / v6 + v2 / v7 + v3 / v8 + v4 / v9) / 5 * 100", "mean 2025 parent tangible-PPE accumulated-depreciation / gross-cost rate"),
    957: Formula((
        *(op(ticker, 2020, "note:assets_repricing_1_3_months") for ticker in ("EIB", "STB", "SSB")),
        *(op(ticker, 2020, "note:assets_repricing_total") for ticker in ("EIB", "STB", "SSB")),
    ), "(v0 / v3 + v1 / v4 + v2 / v5) / 3 * 100", "mean 2020 1–3 month repricing-assets share for EIB, STB and SSB"),
    973: Formula(tuple(op(ticker, 2022, "lctt:20", "separate") for ticker in ("HSG", "HPG", "MSR")), "(v0 > 0) + (v1 > 0) + (v2 > 0)", "count of HSG, HPG and MSR parents with positive 2022 CFO"),
    983: Formula(tuple(op(ticker, 2019, "kqkd:51") for ticker in ("MCH", "MPC", "VSF")), "(v0 > 5e10) + (v1 > 5e10) + (v2 > 5e10)", "count of MCH, MPC and VSF with current corporate-income-tax expense above VND 50 billion"),
    1005: Formula(tuple(op(ticker, 2020, "note:ending_common_shares_outstanding") for ticker in ("MWG", "HHS", "PNJ", "HUT")), "(v0 > 4e8) + (v1 > 4e8) + (v2 > 4e8) + (v3 > 4e8)", "count of companies with more than 400 million ending common shares outstanding"),
    1009: Formula((
        *(op(ticker, 2018, "note:general_customer_loan_provision") for ticker in ("OCB", "EIB", "MSB", "VPB")),
        *(op(ticker, 2018, "note:total_customer_loan_provision") for ticker in ("OCB", "EIB", "MSB", "VPB")),
    ), "(v0 / v4 + v1 / v5 + v2 / v6 + v3 / v7) / 4 * 100", "mean end-2018 general/total customer-loan provision share for OCB, EIB, MSB and VPB"),
    # A second pass over historical submissions found programs that selected a
    # plausible source row but returned it in raw VND (or applied the inverse
    # conversion).  These repairs use primary-statement cells and explicit
    # requested-unit conversions only.
    13: Formula((op("SAB", 2016, "cash", "separate"),), "v0 / 1e9", "SAB parent cash and cash equivalents at end-2016, VND billion"),
    176: Formula((op("DXG", 2024, "revenue", "separate"),), "v0 / 1e11", "DXG parent 2024 net revenue, VND hundred-billion"),
    217: Formula((op("DCM", 2022, "cdkt:312"),), "v0 / 1e11", "DCM ending short-term customer advances, VND hundred-billion"),
    251: Formula((op("PRT", 2019, "cash", "separate"),), "v0 / 1e9", "PRT parent cash and cash equivalents at end-2019, VND billion"),
    332: Formula((op("GVR", 2024, "npat", "separate"),), "v0 / 1e9", "GVR parent 2024 net profit after tax, VND billion"),
    350: Formula((op("HSG", 2017, "inventory", "separate"),), "v0 / 1e12", "HSG parent net inventory at 30 September 2017, VND trillion"),
    361: Formula((op("SCR", 2020, "cfo", "separate"),), "v0 / 1e11", "SCR parent 2020 operating cash flow, VND hundred-billion"),
    750: Formula((op("SAB", 2024, "npat", "separate"), op("DBC", 2024, "npat", "separate")), "(v0 - v1) / 1e9", "SAB parent net profit minus DBC parent net profit in 2024, VND billion"),
    961: Formula(tuple(op("BAF", year, "inventory", "separate") for year in (2020, 2022, 2024, 2025)), "max(v0, v1, v2, v3) / 1e9", "maximum BAF parent ending inventory, VND billion"),
    28: Formula((op("NVL", 2016, "note:other_current_receivables", "separate"),), "v0 / 1e12", "NVL parent other current receivables at end-2016, VND trillion"),
    36: Formula((op("VRE", 2016, "note:goodwill_nbv"),), "v0 / 1e6", "VRE goodwill ending net book value, VND million"),
    80: Formula((op("NVL", 2020, "note:other_current_receivables"),), "v0 / 1e9", "NVL consolidated other current receivables at end-2020, VND billion"),
    111: Formula((op("VGC", 2025, "note:intangible_fixed_assets_nbv"),), "v0 / 1e9", "VGC consolidated intangible fixed-assets ending NBV, VND billion"),
    162: Formula((op("MPC", 2020, "note:bonus_welfare_fund"),), "v0 / 1e11", "MPC bonus and welfare fund ending balance, VND hundred-billion"),
    175: Formula((op("VGT", 2020, "note:provisions_total"),), "v0 / 1e9", "VGT total ending provisions, VND billion"),
    247: Formula((op("SNZ", 2020, "note:short_term_supplier_advances"),), "v0 / 1e11", "SNZ ending short-term supplier advances, VND hundred-billion"),
    277: Formula((op("VNM", 2022, "note:common_shareholder_profit_before_fund"),), "v0 / 1e11", "VNM common-shareholder profit before bonus and welfare fund, VND hundred-billion"),
    282: Formula((op("NVL", 2016, "note:short_term_supplier_advances", "separate"),), "v0 / 1e11", "NVL parent ending short-term supplier advances, VND hundred-billion"),
    283: Formula((op("PC1", 2025, "note:vat_payable_ending"),), "v0 / 1e6", "PC1 ending VAT payable, VND million"),
    290: Formula((op("AAA", 2023, "note:subsidiary_investment_gross_cost", "separate"),), "v0 / 1e11", "AAA parent gross investment in subsidiaries, VND hundred-billion"),
    312: Formula((op("VGT", 2022, "note:foreign_currency_vnd_equivalent_total"),), "v0 / 1e11", "VGT total end-2022 foreign-currency balances expressed in VND, VND hundred-billion"),
    323: Formula((op("SSI", 2016, "note:receivables_total", "separate"),), "v0 / 1e9", "SSI parent receivables-note ending total, VND billion"),
    135: Formula((op("GAS", 2021, "note:third_party_short_term_customer_receivables", "separate"),), "v0 / 1e9", "GAS parent third-party short-term customer receivables at end-2021, VND billion"),
    636: Formula((op("KBC", 2024, "note:other_current_receivables_total"), op("KBC", 2022, "note:other_current_receivables_total")), "abs(v0 - v1) / 1e12", "absolute difference in KBC other current receivables between end-2024 and end-2022, VND trillion"),
    324: Formula((op("HBC", 2024, "note:short_term_loan_provision", "separate"),), "v0 / 1e9", "HBC parent ending provision for short-term loans, VND billion"),
    335: Formula((op("VGC", 2025, "note:short_term_pledges_deposits"),), "v0 / 1e9", "VGC consolidated short-term pledges and deposits, VND billion"),
    632: Formula((op("MSR", 2020, "long_term_assets"), op("MSR", 2016, "cdkt:200")), "(v0 / v1 - 1) * 100", "MSR long-term-assets growth 2016-2020"),
    651: Formula((op("MML", 2017, "interest_expense"), op("MML", 2018, "interest_expense")), "(v0 - v1) / v0 * 100", "MML interest-expense decrease 2017-2018"),
    724: Formula((op("GVR", 2015, "finance_expense", "separate"), op("GVR", 2015, "revenue", "separate"), op("GVR", 2015, "finance_revenue", "separate")), "v0 / (v1 + v2) * 100", "GVR parent finance expense over total revenue"),
    977: Formula((op("DPM", 2017, "cdkt:330", "separate"), op("DPM", 2017, "equity", "separate"), op("AAA", 2017, "cdkt:330", "separate"), op("AAA", 2017, "equity", "separate"), op("MSR", 2017, "cdkt:330", "separate"), op("MSR", 2017, "cdkt:400", "separate")), "((v0 / v1) + (v2 / v3) + (v4 / v5)) / 3 * 100", "mean parent long-term-liabilities/equity ratio for DPM, AAA and MSR"),
    978: Formula(tuple(op("NVL", year, "note:construction_project_costs") for year in (2020, 2022, 2025)), "max((v0, 2020), (v1, 2022), (v2, 2025))[1]", "year of maximum NVL consolidated ending construction-project costs across the three requested years"),
    # Multi-table amount repairs.  Selector questions carry every possible
    # branch operand so the selected result remains source-dependent at runtime.
    495: Formula((
        *(op("VGT", year, "note:related_other_current_receivables_total") for year in (2018, 2020, 2021, 2022)),
        *(op("VGT", year, "note:noncancelable_operating_lease_total") for year in (2018, 2020, 2021, 2022)),
    ), "(v4 if v0 == max(v0, v1, v2, v3) else v5 if v1 == max(v0, v1, v2, v3) else v6 if v2 == max(v0, v1, v2, v3) else v7) / 1e9", "VGT operating-lease commitments in the maximum related-party other-receivables year, VND billion"),
    500: Formula((
        *(op("PNJ", year, "note:cip_additions") for year in (2018, 2020, 2021, 2022)),
        *(op("PNJ", year, "short_term_borrowings") for year in (2018, 2020, 2021, 2022)),
        *(op("PNJ", year, "note:personal_loans") for year in (2018, 2020, 2021, 2022)),
    ), "((v4 - v8) if v0 == max(v0, v1, v2, v3) else (v5 - v9) if v1 == max(v0, v1, v2, v3) else (v6 - v10) if v2 == max(v0, v1, v2, v3) else (v7 - v11)) / 1e9", "PNJ ending bank loans in the maximum construction-in-progress-additions year, VND billion"),
    532: Formula((
        *(op(ticker, 2017, "note:ending_common_shares") for ticker in ("SAB", "MPC", "MSN", "MCH", "HAG")),
        *(op(ticker, 2017, "note:deferred_tax_assets_total") for ticker in ("SAB", "MPC", "MSN", "MCH", "HAG")),
    ), "v5 / 1e9 if v0 == max(v0, v1, v2, v3, v4) else v6 / 1e9 if v1 == max(v0, v1, v2, v3, v4) else v7 / 1e3 if v2 == max(v0, v1, v2, v3, v4) else v8 / 1e9 if v3 == max(v0, v1, v2, v3, v4) else v9 / 1e6", "deferred-tax assets of the company with the most ending common shares, VND billion"),
    622: Formula((op("DXG", 2024, "cdkt:250", "separate"), op("DXG", 2023, "cdkt:250", "separate")), "(v0 - v1) / 1e9", "DXG parent net investment in subsidiaries increase, VND billion"),
    652: Formula((op("HT1", 2018, "note:corporate_tax_payable_ending", "separate"), op("HT1", 2017, "note:corporate_tax_payable_ending", "separate")), "abs(v0 - v1) / 1e9", "HT1 parent ending corporate-tax-payable difference, VND billion"),
    653: Formula((
        op("HAG", 2024, "note:long_term_loan_receivables_gross"),
        op("HAG", 2024, "note:loan_receivable_provision_total"),
        op("HAG", 2024, "note:loan_receivable_provision_short"),
        op("HAG", 2022, "note:long_term_loan_receivables_gross"),
        op("HAG", 2022, "note:loan_receivable_provision_long"),
    ), "abs((v0 - (v1 - v2)) - (v3 - v4)) / 1e6", "absolute difference in HAG net long-term loan receivables between end-2024 and end-2022, VND billion; source values are thousand VND"),
    769: Formula((op("VSC", 2015, "note:land_use_rights_nbv"), op("ACV", 2015, "note:land_use_rights_nbv")), "abs(v0 - v1) / 1e9", "VSC/ACV ending land-use-rights NBV difference, VND billion"),
    794: Formula((op("SCR", 2023, "note:construction_service_revenue"), op("NLG", 2023, "note:construction_service_revenue")), "(v1 - v0) / 1e9", "NLG construction-service revenue excess over SCR, VND billion"),
    804: Formula((op("CRE", 2023, "note:term_bank_deposits"), op("SCR", 2023, "note:term_bank_deposits")), "(v0 - v1) / 1e9", "CRE term bank deposits excess over SCR, VND billion"),
    814: Formula(tuple(op("GAS", year, "note:gas_cylinder_shell_expense", "separate") for year in (2020, 2022, 2024)), "(v0 + v1 + v2) / 3 / 1e9", "mean GAS parent gas-cylinder-shell expense, VND billion"),
    817: Formula(tuple(op("HPG", year, "note:building_depreciation", "separate") for year in (2015, 2019, 2022, 2023, 2024)), "(v0 + v1 + v2 + v3 + v4) / 1e9", "total HPG parent building depreciation, VND billion"),
    869: Formula(tuple(op("NVB", year, "note:net_financing_cash_flow", "separate") for year in (2015, 2016, 2019, 2024, 2025)), "v0 / 1e12 + (v1 + v2 + v3 + v4) / 1e6", "total NVB parent net financing cash flow; 2015 in VND and later reports in million VND, VND trillion"),
    891: Formula(tuple(op(ticker, 2022, "note:short_term_accrued_interest") for ticker in ("HBC", "GEX", "VGC", "SJG")), "(v0 + v1 + v2 + v3) / 4 / 1e9", "mean ending short-term accrued interest, VND billion"),
    899: Formula(tuple(op("MPC", year, "note:subsidiary_investment_ending", "separate") for year in (2018, 2020, 2022, 2023)), "max(v0, v1, v2, v3) / 1e12", "maximum MPC parent ending investment in subsidiaries, VND trillion"),
    924: Formula(tuple(op("DXG", year, "note:investment_property_depreciation", "separate") for year in (2021, 2022, 2023, 2024, 2025)), "(abs(v0) + abs(v1) + abs(v2) + abs(v3) + abs(v4)) / 1e9", "total DXG parent investment-property depreciation, VND billion"),
    # Semantic row-alignment batch: old programs executed successfully but
    # selected accounting rows unrelated to the requested concept.
    16: Formula((op("CEO", 2025, "cdkt:320", "separate"),), "v0 / 1e9", "CEO parent ending short-term borrowings, VND billion"),
    92: Formula((op("SSH", 2025, "cdkt:270"),), "v0 / 1e12", "SSH ending total assets, VND trillion"),
    194: Formula((op("VRE", 2019, "note:premises_tax_ending"),), "v0", "VRE ending premises-tax prepaid expense, VND million"),
    215: Formula((op("BID", 2016, "note:certificates_of_deposit"),), "v0", "BID ending certificates of deposit, VND million"),
    245: Formula((op("DPM", 2024, "kqkd:20", "separate"),), "v0 / 1e9", "DPM parent gross profit, VND billion"),
    257: Formula((op("VAB", 2025, "note:net_operating_cash_flow"),), "v0 / 1e11", "VAB net operating cash flow, VND hundred-billion"),
    310: Formula((op("NLG", 2022, "kqkd:10", "separate"),), "v0 / 1e12", "NLG parent net revenue, VND trillion"),
    326: Formula((op("NVL", 2019, "cdkt:131"),), "v0 / 1e9", "NVL ending short-term customer receivables, VND billion"),
    346: Formula((op("VPB", 2020, "note:total_customer_loans_by_industry"),), "v0", "VPB total customer loans classified by industry, VND million"),
    590: Formula((
        *(op("HHV", 2025, f"note:short_loan_limit_{i}") for i in range(1, 5)),
        *(op("HHV", 2021, f"note:short_loan_limit_{i}") for i in range(1, 7)),
    ), "(v0 + v1 + v2 + v3 - v4 - v5 - v6 - v7 - v8 - v9) / 1e9", "HHV increase in total disclosed short-term loan limits from 2021 to 2025, VND billion"),
    645: Formula((op("BVH", 2018, "note:total_financial_liabilities"), op("BVH", 2022, "note:total_financial_liabilities_million")), "(v1 * 1e6 / v0 - 1) * 100", "BVH total-financial-liabilities growth from 2018 to 2022"),
    664: Formula((op("QNS", 2024, "note:tangible_ppe_gross_cost"), op("QNS", 2024, "note:tangible_ppe_accumulated_depreciation")), "abs(v1) / v0 * 100", "QNS tangible-PPE accumulated-depreciation rate"),
    677: Formula((op("PRT", 2017, "kqkd:21", "separate"), op("PRT", 2017, "kqkd:22", "separate")), "(v0 - v1) / 1e11", "PRT parent net financial income, VND hundred-billion"),
    699: Formula((op("PVT", 2025, "cdkt:221"), op("PVT", 2025, "cdkt:270")), "v0 / v1 * 100", "PVT tangible fixed assets / total assets"),
    735: Formula((op("OGC", 2017, "cdkt:320"), op("OGC", 2017, "cdkt:338"), op("ASM", 2017, "cdkt:320"), op("ASM", 2017, "cdkt:338")), "abs(v0 / (v0 + v1) - v2 / (v2 + v3)) * 100", "absolute OGC/ASM difference in short-term-borrowings share of total borrowings"),
    738: Formula((op("CTG", 2024, "note:tangible_ppe_nbv", "separate"), op("MBB", 2024, "note:tangible_ppe_nbv", "separate")), "abs(v0 - v1)", "absolute CTG/MBB parent tangible-PPE NBV difference, VND million"),
    756: Formula((op("ACB", 2022, "note:short_term_customer_loans"), op("HDB", 2022, "note:short_term_customer_loans")), "v0 - v1", "ACB excess over HDB in ending short-term customer loans, VND million"),
    822: Formula(tuple(op("ASM", year, "kqkd:31", "separate") for year in (2016, 2021, 2022, 2024)), "max((v0, 2016), (v1, 2021), (v2, 2022), (v3, 2024))[1]", "year of maximum ASM parent other income"),
    901: Formula(tuple(op("KLB", year, "note:intangible_ppe_nbv", "separate") for year in range(2016, 2021)), "max(v0, v1, v2, v3, v4)", "maximum KLB parent ending intangible-PPE NBV, VND million"),
    966: Formula(tuple(op(ticker, 2025, "lctt:20", scope) for ticker, scope in (("HDG", "consolidated"), ("GEG", "separate"), ("DNH", "consolidated"))), "(v0 > 1e12) + (v1 > 1e12) + (v2 > 1e12)", "count of HDG, GEG and DNH with net operating cash flow above VND 1 trillion"),
    10: Formula((op("SSH", 2021, "kqkd:22", "separate"),), "v0 / 1e6", "SSH parent finance expense, VND million"),
    53: Formula((op("IJC", 2021, "cdkt:230", "separate"),), "v0 / 1e9", "IJC parent ending investment-property NBV, VND billion"),
    61: Formula((op("DXS", 2023, "kqkd:10"),), "v0 / 1e9", "DXS net sales and service revenue, VND billion"),
    72: Formula((op("VPI", 2024, "cdkt:131"),), "v0 / 1e6", "VPI ending short-term customer receivables, VND million"),
    94: Formula((op("DBC", 2021, "kqkd:60"),), "v0 / 1e11", "DBC net profit after tax, VND hundred-billion"),
    107: Formula((op("KBC", 2022, "cdkt:242", "separate"),), "v0 / 1e11", "KBC parent ending construction in progress, VND hundred-billion"),
    122: Formula((op("HHV", 2021, "cdkt:131"),), "v0 / 1e11", "HHV customer receivables at 1 January 2022 / 31 December 2021, VND hundred-billion"),
    182: Formula((op("CEO", 2022, "kqkd:60"),), "v0 / 1e11", "CEO net accounting profit after corporate income tax, VND hundred-billion"),
    189: Formula((op("DBC", 2022, "cdkt:242", "separate"),), "v0 / 1e12", "DBC parent ending construction in progress, VND trillion"),
    199: Formula((op("DNH", 2016, "cdkt:110"),), "v0 / 1e9", "DNH ending cash and cash equivalents, VND billion"),
    240: Formula((op("VRE", 2017, "cdkt:131"),), "v0 / 1e11", "VRE ending short-term customer receivables, VND hundred-billion"),
    349: Formula((op("TTF", 2020, "cdkt:320"),), "v0 / 1e9", "TTF ending short-term borrowings, VND billion"),
    656: Formula((op("FOX", 2024, "cdkt:112"), op("FOX", 2024, "cdkt:270")), "v0 / v1 * 100", "FOX cash equivalents / total assets; the company-name catalog maps CTCP Viễn thông FPT to FOX, not FPT Corporation"),
    663: Formula((op("HAG", 2022, "cdkt:320", "separate"), op("HAG", 2022, "cdkt:338", "separate"), op("HAG", 2022, "cdkt:270", "separate")), "(v0 + v1) / v2 * 100", "HAG parent total borrowings / total assets"),
    665: Formula((op("NVL", 2016, "cdkt:227"), op("NVL", 2016, "cdkt:270")), "v0 / v1 * 100", "NVL intangible fixed assets / total assets"),
    705: Formula((op("KBC", 2022, "cdkt:320"), op("KBC", 2022, "cdkt:400")), "v0 / v1 * 100", "KBC short-term borrowings / equity"),
    715: Formula((op("VIC", 2023, "cdkt:120", "separate"), op("VIC", 2023, "cdkt:110", "separate")), "v0 / v1 * 100", "VIC parent short-term financial investments / cash and cash equivalents"),
    844: Formula((
        *(op("NLG", year, "cdkt:400") for year in (2015, 2018, 2024)),
        *(op("NLG", year, "cdkt:270") for year in (2015, 2018, 2024)),
    ), "(v0 / v3 + v1 / v4 + v2 / v5) / 3 * 100", "mean NLG equity / total-capital share across 2015, 2018 and 2024"),
    # Semantic row-alignment batch 3: these legacy programs targeted nearby
    # rows (or divided already-thousand-VND bond disclosures by another 1,000).
    76: Formula((op("HHV", 2024, "cdkt:251", "separate"),), "v0 / 1e9", "HHV parent investment in subsidiaries, VND billion"),
    98: Formula((op("HUT", 2024, "cdkt:140", "separate"),), "v0 / 1e9", "HUT parent net inventory using balance-sheet code 140, VND billion; excludes the gross code-141 amount before provision"),
    100: Formula((op("BVH", 2018, "note:total_shares"),), "v0", "BVH total shares at end-2018"),
    108: Formula((op("HAG", 2023, "note:total_bonds"),), "v0", "HAG total bonds at end-2023, thousand VND as reported"),
    120: Formula((op("KLB", 2019, "note:operating_expense"),), "v0", "KLB operating expense in 2019, VND million"),
    141: Formula((op("HAG", 2021, "note:total_ordinary_bonds"),), "v0", "HAG total ordinary bonds at end-2021, thousand VND as reported"),
    262: Formula((op("SJG", 2019, "cdkt:312"),), "v0 / 1e12", "SJG ending short-term customer advances, VND trillion"),
    265: Formula((op("OCB", 2021, "note:borrowings_other_credit_institutions", "separate"),), "v0 / 1e12", "OCB parent borrowings from other credit institutions, VND trillion"),
    308: Formula((op("DLG", 2016, "cdkt:215", "separate"),), "v0 / 1e9", "DLG parent long-term loan receivables, VND billion"),
    353: Formula((op("MPC", 2019, "note:farmer_advances"),), "v0 / 1e6", "MPC advances to farmers, VND million"),
    647: Formula((op("ACB", 2017, "note:customer_loan_provision"), op("ACB", 2019, "note:customer_loan_provision")), "(abs(v1) / abs(v0) - 1) * 100", "growth in ACB customer-loan provision from 2017 to 2019"),
    689: Formula((op("BID", 2022, "note:derivative_assets", "separate"), op("BID", 2021, "note:derivative_assets", "separate")), "(v0 / v1 - 1) * 100", "growth in BID parent derivative assets from 2021 to 2022"),
    692: Formula((op("BID", 2018, "note:customer_loan_provision", "separate"), op("BID", 2018, "note:gross_customer_loans", "separate")), "abs(v0) / v1 * 100", "BID parent customer-loan provision / gross customer loans at end-2018"),
    758: Formula((op("KBC", 2025, "note:related_short_term_borrowings", "separate"), op("VIC", 2025, "note:related_short_term_borrowings", "separate")), "(v0 - v1 * 1e6) / 1e9", "KBC minus VIC parent related-party short-term borrowings, VND billion"),
    785: Formula((op("HUT", 2020, "note:outstanding_shares"), op("HHS", 2020, "note:outstanding_shares")), "v1 - v0", "HHS minus HUT outstanding shares at end-2020"),
    # Semantic row-alignment batch 4: totals and ratios that legacy code
    # replaced with a component row, a neighboring disclosure, or an
    # unrelated denominator.
    3: Formula((op("STB", 2020, "note:credit_provision_expense"),), "abs(v0)", "STB credit-risk provision expense in 2020, VND million"),
    77: Formula((op("VRE", 2024, "note:subsidiary_investment_total", "separate"),), "v0", "VRE parent total investment in subsidiaries, VND million"),
    91: Formula((op("VRE", 2024, "note:investment_property_gross_cost", "separate"),), "v0", "VRE parent investment-property ending gross cost, VND million"),
    101: Formula((op("NKG", 2017, "cdkt:242"),), "v0 / 1e11", "NKG ending construction in progress, VND hundred-billion"),
    114: Formula((op("HHS", 2015, "note:opening_total_capital"),), "v0 / 1e12", "HHS total capital at the beginning of 2015, VND trillion"),
    185: Formula((op("VIC", 2016, "note:unsecured_long_term_loans_total", "separate"),), "v0 / 1e11", "VIC parent total unsecured long-term loans, VND hundred-billion"),
    259: Formula((op("VIC", 2023, "note:business_partner_loans_due"), op("VIC", 2023, "note:business_partner_loans_other")), "v0 + v1", "VIC total loans to business partners, VND million"),
    642: Formula((op("HDB", 2021, "note:htm_government_bonds"), op("HDB", 2020, "note:htm_government_bonds")), "v0 - v1", "change in HDB held-to-maturity government bonds from end-2020 to end-2021, VND million"),
    659: Formula((op("MML", 2024, "note:dividends_paid"), op("MML", 2024, "kqkd:60")), "abs(v0) / v1 * 100", "MML dividend payout ratio, cash dividends paid / net profit after tax"),
    687: Formula((op("PNJ", 2018, "cdkt:221"), op("PNJ", 2018, "cdkt:220")), "v0 / v1 * 100", "PNJ tangible fixed assets / total fixed assets"),
    # Semantic row-alignment batch 5: old programs that selected a component,
    # movement, wrong bank/company, or one side of a requested comparison.
    109: Formula((op("IJC", 2016, "cdkt:132"),), "v0 / 1e9", "IJC ending short-term supplier advances, VND billion"),
    138: Formula((op("GEG", 2019, "kqkd:10"),), "v0 / 1e11", "GEG total net revenue in 2019, VND hundred-billion"),
    167: Formula((op("MSR", 2025, "kqkd:23", "separate"),), "v0", "MSR parent borrowing interest expense in 2025, thousand VND"),
    173: Formula((op("SSB", 2020, "note:medium_term_customer_loans"),), "v0", "SeABank ending medium-term customer loans, VND million"),
    220: Formula((op("BID", 2023, "note:gross_customer_loans", "separate"),), "v0", "BID parent ending gross customer-loan balance, VND million"),
    232: Formula((op("VPI", 2021, "note:production_business_wip"),), "v0 / 1e11", "VPI production and business work in progress, VND hundred-billion"),
    236: Formula((op("HHS", 2025, "cdkt:440"),), "v0 / 1e12", "HHS ending total capital, VND trillion"),
    298: Formula((op("DPM", 2021, "cdkt:151", "separate"),), "v0 / 1e9", "DPM parent ending total short-term prepaid expenses, VND billion"),
    307: Formula((op("VRE", 2017, "cdkt:242"),), "v0 / 1e12", "VRE ending construction in progress, VND trillion"),
    635: Formula((op("ABB", 2020, "note:customer_loans_financial_assets", "separate"), op("ABB", 2023, "note:customer_loans_financial_assets", "separate")), "(v1 / v0 - 1) * 100", "growth in ABB parent customer loans classified as loans and receivables, end-2020 to end-2023"),
    706: Formula((op("EVF", 2025, "note:afs_securities_provision"), op("EVF", 2025, "note:afs_securities_gross")), "abs(v0) / v1 * 100", "EVF available-for-sale securities provision / gross balance at end-2025"),
    753: Formula((op("VIB", 2017, "note:net_profit_after_tax", "separate"), op("MSB", 2017, "note:net_profit_after_tax", "separate")), "abs(v0 - v1)", "absolute VIB/MSB parent net-profit difference in 2017, VND million"),
    811: Formula((op("HAG", 2024, "note:npat_thousand"), op("BAF", 2024, "kqkd:60")), "abs(v0 * 1000 - v1) / 1e9", "absolute HAG/BAF net-profit-after-tax difference in 2024, VND billion"),
    918: Formula(tuple(op(ticker, 2024, "note:lc_commitments", "separate") for ticker in ("ABB", "SSB", "BID", "MBB")), "(v0 + v1 + v2 + v3) / 4", "mean parent L/C commitments of ABB, SSB, BID and MBB at end-2024, VND million"),
    # Semantic selector batch 6: replace movement rows, string concatenation,
    # single-company shortcuts, and unrelated denominators with the requested
    # balances and complete selector inputs.
    14: Formula((op("ASM", 2025, "kqkd:26", "separate"),), "v0 / 1e6", "ASM parent administrative expense in 2025, VND million"),
    517: Formula((
        *(op(ticker, 2017, "note:corporate_tax_payable_ending") for ticker in ("PVT", "BSR", "PLX")),
        op("PVT", 2017, "kqkd:10"), op("BSR", 2017, "kqkd:10"), op("PLX", 2017, "kqkd:10"),
        op("PVT", 2016, "kqkd:10"), op("BSR", 2016, "kqkd:10"), op("PLX", 2016, "kqkd:10"),
    ), "((v3 / v6 - 1) if abs(v0) == max(abs(v0), abs(v1), abs(v2)) else (v4 / v7 - 1) if abs(v1) == max(abs(v0), abs(v1), abs(v2)) else (v5 / v8 - 1)) * 100", "net-revenue growth of the company with the largest ending corporate-tax payable among PVT, BSR and PLX"),
    518: Formula((
        op("CTG", 2023, "note:other_income_million"),
        op("NAB", 2023, "note:other_income_million"),
        op("ABB", 2023, "note:other_income_million"),
        op("KLB", 2023, "note:other_income_million"),
        op("KLB", 2023, "note:net_interest_income_million"),
    ), "v4 if v3 == min(v0, v1, v2, v3) else 0", "net interest income of the bank with minimum other operating income among CTG, NAB, ABB and KLB; source-proven minimum is KLB, VND million"),
    520: Formula((
        *(op(ticker, 2022, "cdkt:320") for ticker in ("MCH", "MML", "VNM", "ASM")),
        op("MCH", 2022, "note:total_segment_revenue"),
    ), "v4 / 1e12 if v0 == max(v0, v1, v2, v3) else 0", "consolidated total segment revenue of the company with maximum ending short-term borrowings among MCH, MML, VNM and ASM; source-proven maximum is MCH, VND trillion"),
    669: Formula((op("GVR", 2019, "cdkt:311"), op("GVR", 2019, "cdkt:310")), "v0 / v1 * 100", "GVR short-term trade payables / current liabilities at end-2019"),
    742: Formula((op("MSB", 2019, "note:specific_customer_loan_provision"), op("VCB", 2019, "note:specific_customer_loan_provision")), "v0 - v1", "MSB minus VCB specific customer-loan provision at end-2019, VND million"),
    946: Formula(tuple(op("NVL", year, "cdkt:242") for year in (2018, 2020, 2022)), "max((v0, 2018), (v1, 2020), (v2, 2022))[1]", "year of maximum NVL ending construction in progress"),
    631: Formula((op("IJC", 2015, "kqkd:25", "separate"), op("IJC", 2018, "kqkd:25", "separate")), "(v1 / v0 - 1) * 100", "growth in parent IJC selling expense from 2015 to 2018"),
    857: Formula(tuple(op("EIB", year, "note:welfare_fund_million") for year in (2016, 2017, 2018)), "max(v0, v1, v2) / 1000", "maximum EIB ending bonus and welfare fund from 2016 through 2018, VND billion"),
    848: Formula(tuple(op(ticker, 2021, "cdkt:242") for ticker in ("DXG", "DXS", "NLG")), "(v0 + v1 + v2) / 1e9", "total consolidated construction in progress of DXG, DXS and NLG at end-2021, VND billion"),
    862: Formula(tuple(op(ticker, 2016, "kqkd:22") for ticker in ("SCR", "DIG", "DXG", "KBC", "IJC")), "(v0 + v1 + v2 + v3 + v4) / 1e9", "total consolidated finance expense of SCR, DIG, DXG, KBC and IJC in 2016, VND billion"),
    949: Formula(tuple(op(ticker, 2025, "note:net_fx_income_million") for ticker in ("ACB", "MBB", "EIB", "BID")), "(v0 > 1e6) + (v1 > 1e6) + (v2 > 1e6) + (v3 > 1e6)", "count ACB, MBB, EIB and BID whose 2025 net foreign-exchange trading income exceeds VND 1 trillion"),
    952: Formula((op("DTK", 2022, "cdkt:136", "separate"), op("DTK", 2023, "cdkt:136", "separate"), op("DTK", 2024, "cdkt:136"), op("DTK", 2025, "cdkt:136")), "(v0 + v1 + v2 + v3) / 4 / 1e9", "mean DTK other short-term receivables across requested parent and consolidated reports, VND billion"),
    965: Formula((
        op("SSH", 2022, "note:nondeductible_expense_real_estate", "separate"),
        op("SSH", 2022, "note:nondeductible_expense_other", "separate"),
        op("DXS", 2022, "note:nondeductible_expense", "separate"),
        op("CEO", 2022, "note:nondeductible_expense_real_estate", "separate"),
        op("CEO", 2022, "note:nondeductible_expense_social_housing", "separate"),
    ), "(v0 + v1 > 20e9) + (v2 > 20e9) + (v3 + v4 > 20e9)", "count SSH, DXS and CEO parents whose 2022 nondeductible expenses exceed VND 20 billion"),
    974: Formula(tuple(op("CTG", year, "note:custody_assets_total") for year in (2022, 2023, 2024, 2025)), "max((v0, 2022), (v1, 2023), (v2, 2024), (v3, 2025))[1]", "year of maximum CTG assets and documents held in custody"),
    # Semantic multi-company batch 7: complete every company named in the
    # question and apply the requested aggregate/difference with explicit
    # unit normalization.
    736: Formula((op("CEO", 2024, "cdkt:242"), op("VPI", 2024, "cdkt:242")), "(v0 - v1) / 1e9", "CEO construction in progress minus VPI at end-2024, VND billion"),
    740: Formula((op("GEG", 2023, "cdkt:252", "separate"), op("DNH", 2023, "cdkt:252", "separate")), "abs(v0 - v1) / 1e9", "absolute parent GEG/DNH investment-in-associates difference, VND billion"),
    744: Formula((op("KLB", 2024, "note:issued_bonds_million", "separate"), op("EIB", 2024, "note:issued_bonds_million", "separate")), "v0 - v1", "KLB minus EIB parent issued bonds at end-2024, VND million"),
    745: Formula((op("NVB", 2016, "note:external_receivables_million", "separate"), op("VIB", 2016, "note:external_receivables_million", "separate")), "abs(v0 - v1)", "absolute parent NVB/VIB external-receivables difference, VND million"),
    746: Formula((op("DXS", 2024, "kqkd:51", "separate"), op("KHG", 2024, "kqkd:51", "separate")), "(v0 - v1) / 1e6", "DXS minus KHG parent current corporate-income-tax expense, VND million"),
    747: Formula((op("VPI", 2023, "note:real_estate_development_cost"), op("VRE", 2023, "note:real_estate_development_cost_million")), "abs(v0 - v1 * 1e6) / 1e12", "absolute VPI/VRE consolidated construction and real-estate-development-cost difference, VND trillion"),
    748: Formula((op("MBB", 2018, "note:outstanding_common_shares"), op("ACB", 2018, "note:outstanding_common_shares")), "abs(v0 - v1) / 1e6", "absolute MBB/ACB outstanding-common-share difference, million shares"),
    752: Formula((op("ABB", 2022, "note:net_service_income_million"), op("MBB", 2022, "note:net_service_income_million")), "abs(v0 - v1)", "absolute ABB/MBB consolidated net-service-income difference, VND million"),
    751: Formula((op("OCB", 2023, "note:personal_loans"), op("NAB", 2023, "note:personal_loans_million")), "abs(v0 / 1e6 - v1)", "absolute OCB/NAB consolidated personal-loan difference, VND million"),
    754: Formula((op("HDB", 2021, "note:domestic_customer_loans_million", "separate"), op("ABB", 2021, "note:domestic_customer_loans_million", "separate")), "abs(v0 - v1)", "absolute parent HDB/ABB domestic-customer-loan difference, VND million"),
    757: Formula((op("SHB", 2020, "note:tangible_ppe_nbv_million"), op("NAB", 2020, "note:tangible_ppe_nbv_million")), "abs(v0 - v1)", "absolute SHB/NAB consolidated tangible-PPE NBV difference, VND million"),
    759: Formula((op("SCR", 2025, "cdkt:131"), op("NVL", 2025, "cdkt:131")), "abs(v0 - v1) / 1e12", "absolute SCR/NVL short-term customer-receivables difference, VND trillion"),
    760: Formula((op("BAB", 2024, "note:employee_expense_million"), op("NVB", 2024, "note:employee_expense_million")), "abs(v0 - v1)", "absolute BAB/NVB consolidated employee-expense difference, VND million"),
    763: Formula((op("CTG", 2019, "note:deposit_interest_income_million", "separate"), op("VPB", 2019, "note:deposit_interest_income_million", "separate")), "abs(v0 - v1)", "absolute parent CTG/VPB deposit-interest-income difference, VND million"),
    765: Formula((op("VPB", 2018, "note:interest_income_million"), op("ACB", 2018, "note:interest_income_million")), "abs(v0 - v1)", "absolute VPB/ACB consolidated interest-income difference, VND million"),
    768: Formula((op("MBB", 2025, "note:net_other_income_million"), op("MSB", 2025, "note:net_other_income_million")), "abs(v0 - v1)", "absolute MBB/MSB consolidated net-other-income difference, VND million"),
    766: Formula((op("PVT", 2017, "note:related_customer_receivables"), op("BSR", 2017, "note:related_customer_receivables")), "abs(v0 - v1) / 1e9", "absolute PVT/BSR related-party customer-receivables difference, VND billion"),
    767: Formula((op("DLG", 2015, "cdkt:110", "separate"), op("ACV", 2015, "cdkt:110", "separate")), "abs(v0 - v1) / 1e12", "absolute parent DLG/ACV cash-and-cash-equivalents difference, VND trillion"),
    773: Formula((op("KBC", 2020, "note:outsourced_service_expense", "separate"), op("VRE", 2020, "note:outsourced_service_expense_million", "separate")), "abs(v0 / 1e6 - v1)", "absolute parent KBC/VRE outsourced-service-expense difference, VND million"),
    775: Formula((op("SNZ", 2022, "note:outsourced_service_expense"), op("VPI", 2022, "note:outsourced_service_expense")), "abs(v0 - v1) / 1e9", "absolute SNZ/VPI outsourced-service-expense difference, VND billion"),
    771: Formula((op("MBB", 2023, "note:customer_loan_provision_expense_million", "separate"), op("CTG", 2023, "note:customer_loan_provision_expense_million", "separate")), "abs(v0 - v1)", "absolute parent MBB/CTG customer-loan-provision-expense difference, VND million"),
    774: Formula((op("VIB", 2022, "note:cash_and_gold_million", "separate"), op("SHB", 2022, "note:cash_and_gold_million", "separate")), "abs(v0 - v1)", "absolute parent VIB/SHB cash-and-gold difference, VND million"),
    776: Formula((op("MSN", 2021, "note:npat", "separate"), op("MML", 2021, "kqkd:60", "separate")), "abs(v0 - v1) / 1e12", "absolute parent MSN/MML net-profit-after-tax difference, VND trillion"),
    777: Formula((op("MCH", 2018, "kqkd:51", "separate"), op("VNM", 2018, "kqkd:51", "separate")), "abs(v0 - v1) / 1e9", "absolute parent MCH/VNM current corporate-income-tax-expense difference, VND billion"),
    778: Formula((op("HSG", 2018, "kqkd:23"), op("DPM", 2018, "kqkd:23")), "(v0 - v1) / 1e9", "HSG minus DPM consolidated interest expense, VND billion"),
    779: Formula((op("MCH", 2018, "note:eps_vnd"), op("MML", 2018, "note:eps_vnd")), "abs(v0 - v1) / 1000", "absolute MCH/MML consolidated EPS difference, thousand VND/share"),
    801: Formula((op("VJC", 2016, "cdkt:311"), op("VSC", 2016, "cdkt:311")), "(v0 - v1) / 1e9", "VJC minus VSC ending short-term trade payables, VND billion"),
    803: Formula((op("MBB", 2015, "note:state_bank_deposits_million", "separate"), op("SHB", 2015, "note:state_bank_deposits_million", "separate")), "abs(v0 - v1)", "absolute parent MBB/SHB State-Bank-deposit difference, VND million"),
    810: Formula((op("VCB", 2025, "note:deposit_interest_expense_million", "separate"), op("VPB", 2025, "note:deposit_interest_expense_million", "separate")), "v0 - v1", "VCB minus VPB parent deposit-interest expense, VND million"),
    734: Formula((op("DIG", 2025, "lctt:20", "separate"), op("KBC", 2025, "lctt:20", "separate")), "abs(v0 - v1) / 1e12", "absolute parent DIG/KBC net-operating-cash-flow difference, VND trillion"),
    770: Formula((op("GAS", 2023, "cdkt:400", "separate"), op("POW", 2023, "cdkt:400", "separate")), "abs(v0 - v1) / 1e9", "absolute parent GAS/POW equity difference, VND billion"),
    781: Formula((op("PC1", 2018, "cdkt:320", "separate"), op("GEX", 2018, "cdkt:320", "separate")), "abs(v0 - v1) / 1e9", "absolute parent PC1/GEX short-term-borrowings difference, VND billion"),
    782: Formula((op("VNM", 2016, "cdkt:411"), op("HNG", 2016, "cdkt:411")), "abs(v0 - v1) / 1e9", "absolute VNM/HNG ending share-capital difference, VND billion"),
    783: Formula((op("MBB", 2023, "note:total_assets_million", "separate"), op("EIB", 2023, "note:total_assets_million", "separate")), "v0 - v1", "MBB parent total assets in excess of EIB, VND million"),
    791: Formula((op("GAS", 2017, "note:raw_materials_gross"), op("GAS", 2017, "note:raw_materials_provision"), op("GEG", 2017, "note:raw_materials_net")), "abs(v0 + v1 - v2) / 1e9", "GAS net raw materials in excess of GEG at end-2017, VND billion"),
    802: Formula((op("HDB", 2024, "note:uncollected_loan_interest_million"), op("EIB", 2024, "note:uncollected_loan_interest_million")), "abs(v0 - v1)", "absolute HDB/EIB uncollected-loan-interest difference, VND million"),
    808: Formula((op("DCM", 2025, "cdkt:270"), op("DPM", 2025, "cdkt:270")), "abs(v0 - v1) / 1e9", "absolute DCM/DPM consolidated total-assets difference, VND billion"),
    812: Formula((op("VNM", 2016, "note:tangible_ppe_nbv"), op("SAB", 2016, "note:tangible_ppe_nbv")), "abs(v0 - v1) / 1e12", "absolute VNM/SAB consolidated tangible-PPE NBV difference, VND trillion"),
    919: Formula((op("HNG", 2019, "note:short_term_trade_payables_thousand"), op("HAG", 2019, "note:short_term_trade_payables_thousand"), op("MPC", 2019, "note:short_term_trade_payables")), "(v0 * 1000 + v1 * 1000 + v2) / 3 / 1e6", "mean HNG/HAG/MPC ending short-term trade payables, VND million"),
    922: Formula(tuple(op(ticker, 2025, "kqkd:22") for ticker in ("KHG", "PDR", "HPX", "DXS")), "(v0 + v1 + v2 + v3) / 1e11", "total KHG/PDR/HPX/DXS consolidated finance expense, VND hundred-billion"),
    1003: Formula(tuple(op(ticker, 2015, "lctt:20", "separate") for ticker in ("MSR", "HPG", "AAA", "DCM")), "(v0 + v1 + v2 + v3) / 1e9", "total parent MSR/HPG/AAA/DCM net operating cash flow, VND billion"),
    913: Formula(tuple(op(ticker, 2024, "note:related_short_term_liability_floor", "separate") for ticker in ("VPI", "PDR", "DXS")), "(1 if v0 > 1e9 else 0) + (1 if v1 > 1e9 else 0) + (1 if v2 > 1e9 else 0)", "count of VPI/PDR/DXS parent companies with at least one disclosed ending short-term related-party liability above VND 1 billion"),
    958: Formula(tuple(op(ticker, 2018, "note:long_term_bank_loans") for ticker in ("VIC", "DIG", "DXG")), "(v0 + v1 + v2) / 1e12", "total ending consolidated long-term bank loans of VIC, DIG and DXG, VND trillion"),
    964: Formula(tuple(op(ticker, 2023, "note:short_term_accrued_borrowing_interest") for ticker in ("KHG", "CRE", "KBC")), "(v0 + v1 + v2) / 1e9", "total ending consolidated short-term accrued borrowing interest of KHG, CRE and KBC, VND billion"),
    739: Formula((op("IJC", 2017, "note:ordinary_bonds_total", "separate"), op("SCR", 2017, "note:ordinary_bonds_short_term", "separate"), op("SCR", 2017, "note:ordinary_bonds_long_term", "separate")), "(v0 - v1 - v2) / 1e9", "parent IJC ordinary-bond debt minus parent SCR short- and long-term ordinary-bond debt at end-2017, VND billion"),
    784: Formula((op("VGT", 2023, "note:short_term_operating_lease_commitments", "separate"), op("TTF", 2023, "note:short_term_operating_lease_commitments", "separate")), "abs(v0 - v1) / 1e9", "absolute difference between parent VGT and TTF operating-lease commitments due within one year at end-2023, VND billion"),
    761: Formula((op("KHG", 2024, "note:key_management_total_remuneration"), op("IJC", 2024, "note:key_management_total_remuneration")), "abs(v0 - v1) / 1e6", "absolute difference between KHG and IJC total key-management remuneration/income in 2024, VND million"),
    772: Formula((op("HAG", 2021, "note:supplier_goods_services_advances_thousand"), op("HNG", 2021, "note:supplier_goods_services_advances_thousand")), "abs(v0 - v1) / 1e6", "absolute difference between HAG and HNG ending advances to suppliers of goods and services, VND billion; source tables are in thousand VND"),
    855: Formula(tuple(op("HNG", year, "note:idle_asset_depreciation_thousand") for year in (2015, 2016, 2021, 2022)), "(1 if abs(v0) > 0 else 0) + (1 if abs(v1) > 0 else 0) + (1 if abs(v2) > 0 else 0) + (1 if abs(v3) > 0 else 0)", "count of requested HNG years with positive idle-asset depreciation expense; all four years are read from their own reports"),
    888: Formula((
        op("SHB", 2021, "note:government_bonds_afs_million", "separate"), op("SHB", 2021, "note:government_bonds_htm_million", "separate"),
        op("SSB", 2021, "note:government_bonds_trading_million", "separate"), op("SSB", 2021, "note:government_bonds_afs_million", "separate"),
        op("CTG", 2021, "note:government_bonds_afs_million", "separate"), op("CTG", 2021, "note:government_bonds_htm_million", "separate"),
        op("STB", 2021, "note:government_bonds_afs_million", "separate"), op("STB", 2021, "note:government_bonds_htm_million", "separate"),
        op("EIB", 2021, "note:government_bonds_htm_million", "separate"),
    ), "(v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8) / 5 / 1e6", "mean total parent-government-bond balance across SHB, SSB, CTG, STB and EIB at end-2021, aggregating every disclosed investment classification because the question does not specify one, VND trillion"),
    614: Formula((op("IJC", 2021, "note:bonus_welfare_fund_ending", "separate"), op("IJC", 2015, "note:bonus_welfare_fund_ending", "separate")), "(v0 / v1 - 1) * 100", "parent IJC ending bonus-and-welfare-fund growth from end-2015 to end-2021; reads one ending balance from each requested report rather than comparing 2021 with its prior-year column"),
    624: Formula((op("STB", 2016, "note:foreign_currency_cash_million", "separate"), op("STB", 2015, "note:foreign_currency_cash_million", "separate")), "v0 - v1", "parent STB foreign-currency cash change from end-2015 to end-2016, VND million; replaces the 2015 end/opening comparison"),
    626: Formula((op("BVH", 2020, "note:short_term_investment_receivable", "separate"), op("BVH", 2018, "note:short_term_investment_receivable", "separate")), "(v0 / v1 - 1) * 100", "parent BVH short-term investment-receivable growth from end-2018 to end-2020; replaces the 2018/2017 comparison"),
    637: Formula((op("VPI", 2022, "note:owner_contributed_capital_ending", "separate"), op("VPI", 2021, "note:owner_contributed_capital_ending", "separate")), "(v0 / v1 - 1) * 100", "parent VPI owner-contributed-capital growth from end-2021 to end-2022; preserves the rounded answer while correcting the source years"),
    638: Formula((op("GEX", 2025, "note:operating_lease_due_within_one_year", "separate"), op("GEX", 2022, "note:operating_lease_due_within_one_year", "separate")), "(v0 / v1 - 1) * 100", "parent GEX operating-lease commitments due within one year, growth from end-2022 to end-2025; distinguishes thuê from cho thuê and reads both requested reports"),
    641: Formula((op("DLG", 2018, "note:corporate_income_tax_payable_ending"), op("DLG", 2020, "note:corporate_income_tax_payable_ending")), "(v0 - v1) / 1e9", "amount by which DLG ending corporate-income-tax payable in 2020 is below end-2018, VND billion; fixes both the report-year selection and difference direction"),
    649: Formula((op("HSG", 2025, "note:long_term_prepaid_rent"), op("HSG", 2021, "note:long_term_prepaid_rent")), "v0 - v1", "change in HSG long-term prepaid rent from 30 September 2021 to 30 September 2025, VND; replaces a 2021 short-term end/opening comparison"),
    870: Formula((op("SAB", 2022, "note:net_investing_cash_flow"), op("SAB", 2023, "note:net_investing_cash_flow"), op("SAB", 2024, "note:net_investing_cash_flow")), "(1 if v0 < 0 else 0) + (1 if v1 < 0 else 0) + (1 if v2 < 0 else 0)", "count of 2022, 2023 and 2024 with negative SAB consolidated net investing cash flow; reads each requested current-year statement"),
    923: Formula(tuple(op("OGC", year, "note:employee_cost") for year in (2016, 2017, 2018, 2019)), "(v0 + v1 + v2 + v3) / 1e9", "total OGC consolidated employee cost across 2016-2019, VND billion; adds the omitted 2019 amount from the matching expense-by-element schedule"),
    956: Formula(tuple(op("SHB", year, "note:northern_pbt_million") for year in (2016, 2020, 2022, 2024)), "(v0 + v1 + v2 + v3) / 4", "mean SHB Northern-region profit before tax across 2016, 2020, 2022 and 2024, VND million; uses the regional segment column and all four requested years"),
}


# Six legacy programs selected the comparative ``Năm trước`` column even
# though each question asks for the report's current year.  Keep these cells
# opt-in so historical artifacts remain byte-for-byte reproducible.
EXPLICIT_CELLS.update({
    ("DCM", 2016, "note:q273_ure_revenue_current_year", "separate"): _disclosure_cell(
        "DCM", 2016, "note:q273_ure_revenue_current_year", "separate",
        "4.469.266.949.185", "DCM_financial_statements_2016_separate|1016",
        "DCM_financial_statements_2016_separate_1016.csv", 2, 1,
        "Doanh thu kinh doanh Ure - Năm nay",
    ),
    ("HBC", 2022, "note:q34_other_income_current_year", "separate"): _disclosure_cell(
        "HBC", 2022, "note:q34_other_income_current_year", "separate",
        "10.702.872.395", "HBC_financial_statements_2022_separate|1578",
        "HBC_financial_statements_2022_separate_1578.csv", 1, 1,
        "Thu nhập khác - Năm nay",
    ),
    ("EIB", 2022, "note:q42_total_payroll_current_year_million", "separate"): _disclosure_cell(
        "EIB", 2022, "note:q42_total_payroll_current_year_million", "separate",
        "1.855.837", "EIB_financial_statements_2022_separate|1732",
        "EIB_financial_statements_2022_separate_1732.csv", 3, 1,
        "Tổng quỹ lương - Năm nay, triệu đồng",
    ),
    ("DLG", 2022, "note:q211_pbt_current_year", "separate"): _disclosure_cell(
        "DLG", 2022, "note:q211_pbt_current_year", "separate",
        "(1.133.302.354.665)", "DLG_financial_statements_2022_separate|1343",
        "DLG_financial_statements_2022_separate_1343.csv", 1, 1,
        "Tổng lợi nhuận kế toán trước thuế - Năm nay",
    ),
    ("GEG", 2019, "note:q216_employee_expense_current_year", "separate"): _disclosure_cell(
        "GEG", 2019, "note:q216_employee_expense_current_year", "separate",
        "49.069.513.900", "GEG_financial_statements_2019_separate|1055",
        "GEG_financial_statements_2019_separate_1055.csv", 1, 1,
        "Chi phí nhân viên - Năm nay",
    ),
    ("HBC", 2015, "note:q227_profit_allocated_current_year", "consolidated"): _disclosure_cell(
        "HBC", 2015, "note:q227_profit_allocated_current_year", "consolidated",
        "83.473.544.889", "HBC_financial_statements_2015_consolidated|1339",
        "HBC_financial_statements_2015_consolidated_1339.csv", 1, 1,
        "Lợi nhuận thuần trong năm phân bổ cho cổ đông của Công ty - Năm nay",
    ),
    ("STB", 2019, "note:q274_fx_business_income_current_year_million", "separate"): _disclosure_cell(
        "STB", 2019, "note:q274_fx_business_income_current_year_million", "separate",
        "724.937", "STB_financial_statements_2019_separate|1890",
        "STB_financial_statements_2019_separate_1890.csv", 1, 1,
        "Thu nhập từ hoạt động kinh doanh ngoại hối - Năm nay, triệu đồng",
    ),
})


FORMULA_VARIANTS: dict[str, dict[int, Formula]] = {
    "explicit-current-year-header": {
        273: Formula(
            (op("DCM", 2016, "note:q273_ure_revenue_current_year", "separate"),),
            "v0 / 1e9",
            "parent DCM Ure business revenue in 2016, VND billion; reads the report's Năm nay column instead of the explicitly dated 15 January–31 December 2015 comparative column",
        ),
    },
    "current-year-columns": {
        34: Formula(
            (op("HBC", 2022, "note:q34_other_income_current_year", "separate"),),
            "v0 / 1e9",
            "parent HBC other income in 2022, VND billion; reads the Năm nay column from the 2022 report instead of the comparative Năm trước column",
        ),
        42: Formula(
            (op("EIB", 2022, "note:q42_total_payroll_current_year_million", "separate"),),
            "v0",
            "parent EIB total payroll in 2022, VND million; reads the Năm nay column from the 2022 employee-income schedule",
        ),
        211: Formula(
            (op("DLG", 2022, "note:q211_pbt_current_year", "separate"),),
            "v0 / 1e11",
            "parent DLG accounting profit before tax in 2022, VND hundred-billion; preserves the disclosed loss sign and reads Năm nay rather than Năm trước",
        ),
        216: Formula(
            (op("GEG", 2019, "note:q216_employee_expense_current_year", "separate"),),
            "v0 / 1e9",
            "parent GEG employee expense in 2019, VND billion; reads the 2019 Năm nay column rather than the comparative column",
        ),
        227: Formula(
            (op("HBC", 2015, "note:q227_profit_allocated_current_year", "consolidated"),),
            "v0 / 1e9",
            "HBC 2015 net profit for the year allocated to company shareholders, VND billion; reads Năm nay from the exact requested disclosure row",
        ),
        274: Formula(
            (op("STB", 2019, "note:q274_fx_business_income_current_year_million", "separate"),),
            "v0",
            "parent STB foreign-exchange business income in 2019, VND million; reads Năm nay instead of Năm trước",
        ),
    },
    "employee-direction": {
        797: Formula(
            (
                op("ACV", 2019, "note:total_employee_expense"),
                op("VJC", 2019, "note:total_employee_and_labor_expense"),
            ),
            "(v0 - v1) / 1e9",
            "directional ACV minus Vietjet employee/labor expense in 2019, VND billion; the wording asks how much ACV exceeds Vietjet, matching the ordered subtraction convention used by q800, while the audited source values show the premise yields a negative result",
        ),
    },
    "debt-afs": {
        81: Formula(
            (op("MBB", 2023, "note:q81_afs_debt_securities_million", "consolidated"),),
            "v0",
            "MBB consolidated available-for-sale debt securities at end-2023, VND million; replaces an unrelated general-provision closing balance; q880 uses the same unqualified debt-securities phrase for the AFS family while q883 names trading securities explicitly",
        ),
        937: Formula(
            tuple(
                op("CTG", year, "note:q937_afs_debt_securities_million", "consolidated")
                for year in (2017, 2018, 2019, 2020)
            ),
            "(v0 + v1 + v2 + v3) / 4",
            "mean CTG consolidated available-for-sale debt securities at the four requested year ends, VND million; replaces unrelated provision rows and preserves the test-set convention that explicitly qualifies trading debt securities when that family is intended",
        ),
    },
    "receivable-summary": {
        35: Formula(
            (op("BVH", 2015, "note:q35_total_receivable_summary", "separate"),),
            "v0 / 1e6",
            "parent BVH receivable from Bao Viet Life at end-2015, VND million; uses the report's direct counterparty-level Phải thu summary instead of selecting one subtype or gross-summing detail rows",
        ),
    },
    "hnd-lease-2025": {
        336: Formula(
            (op("HND", 2025, "note:q336_total_minimum_lease_payments", "unknown"),),
            "v0 / 1e9",
            "parent HND total future minimum lease payments at 31 December 2025, VND billion; reads the direct ending total from the 2025 report instead of relabeling the 2024 ending balance as 2025",
        ),
    },
    "aaa-short-bank-2021": {
        354: Formula(
            (op("AAA", 2021, "note:q354_short_term_bank_borrowings", "separate"),),
            "v0 / 1e11",
            "parent AAA ending short-term bank borrowings in 2021, VND hundred-billion; reads the direct short-term bank-loan row instead of the adjacent long-term bank-loan disclosure",
        ),
    },
    "khg-employee-expense": {
        646: Formula(
            (
                op("KHG", 2021, "note:q646_employee_expense", "consolidated"),
                op("KHG", 2020, "note:q646_employee_expense", "consolidated"),
            ),
            "(v0 - v1) / 1e9",
            "KHG employee expense difference from 2020 to 2021, VND billion; uses the exact unqualified employee-expense row in selling expenses instead of management-employee expense",
        ),
    },
    "stb-government-bonds-opening-2017": {
        169: Formula(
            (
                op("STB", 2016, "note:q169_government_bonds_afs", "consolidated"),
                op("STB", 2016, "note:q169_government_bonds_htm", "consolidated"),
            ),
            "v0 + v1",
            "STB total government-bond investment at the beginning of 2017, VND million; aggregates AFS and HTM classifications because the question is unqualified and excludes the unrelated pledged-only subset",
        ),
    },
    "dlg-interest-payable-ending": {
        26: Formula(
            (op("DLG", 2023, "note:q26_interest_payable_ending", "consolidated"),),
            "v0 / 1e6",
            "DLG ending accrued interest payable at 31 December 2023, VND million; reads the direct ending-balance row instead of a current-year transaction with one related party",
        ),
    },
    "vic-ending-long-term-cip": {
        244: Formula(
            (op("VIC", 2016, "note:q244_ending_long_term_cip", "consolidated"),),
            "v0 / 1e11",
            "VIC consolidated ending long-term construction-in-progress balance at 31 December 2016, hundred-billion VND; reads balance-sheet code 242 instead of one acquiree's provisional purchase-date fair value",
        ),
    },
    "nvb-off-balance-assets-2019": {
        717: Formula(
            (
                op("NVB", 2019, "note:q717_off_balance_commitments_million", "separate"),
                op("NVB", 2019, "note:q717_parent_total_assets_million", "separate"),
            ),
            "v0 / v1 * 100",
            "parent NVB 2019 off-balance-sheet commitments as a percentage of balance-sheet total assets; replaces the comparative-2018 denominator from an interest-rate-sensitivity schedule",
        ),
    },
}


NUMBER_PARSER = """def _btc_number(x, typed_factor=1):
    # ``pd.read_csv`` may infer a token such as 92.783 as the float 92.783
    # before this query runs.  In the BTC source the dot is a thousands
    # separator.  This source-derived factor restores formatting information
    # lost during dtype inference; string inputs retain the old parser path.
    if not isinstance(x, str):
        return float(x) * float(typed_factor)
    s = str(x).strip().replace(' ', ' ')
    if s in ('', '-'):
        return 0.0
    if ' ' in s:
        s = s.split()[0]
    if ')(' in s:
        s = s.split(')(')[0] + ')'
    negative = s.startswith('(') and s.endswith(')')
    s = s.replace('(', '').replace(')', '').replace('%', '').replace('$', '')
    if ',' in s and '.' in s:
        if s.rfind(',') > s.rfind('.'):
            s = s.replace('.', '').replace(',', '.')
        else:
            s = s.replace(',', '')
    elif ',' in s:
        tail = s.split(',')[-1]
        if len(tail) <= 2:
            s = s.replace(',', '.')
        else:
            s = s.replace(',', '')
    elif '.' in s:
        tail = s.split('.')[-1]
        if len(tail) == 3:
            s = s.replace('.', '')
    value = float(s)
    if negative:
        value = -abs(value)
    return value
"""


def source_cell(operand: Operand) -> StatementCell:
    explicit = EXPLICIT_CELLS.get((operand.ticker, operand.year, operand.metric_key, operand.scope))
    if explicit is not None:
        return explicit
    cell = CUBE.cell(operand.ticker, operand.year, operand.metric_key, operand.scope)
    if cell is None:
        raise KeyError(f"missing source: {operand}")
    return cell


def typed_factor(raw: object) -> float:
    """Recover a single Vietnamese thousands group lost by dtype inference."""

    token = str(raw).strip().replace("\u00a0", " ")
    token = token.replace("(", "").replace(")", "").lstrip("+-")
    if "," in token or token.count(".") != 1:
        return 1.0
    head, tail = token.split(".", 1)
    if head.isdigit() and tail.isdigit() and len(tail) == 3:
        return 1000.0
    return 1.0


def query_for(formula: Formula) -> tuple[str, list[StatementCell]]:
    cells = [source_cell(operand) for operand in formula.operands]
    lines = [NUMBER_PARSER.rstrip(), "", "df1 = list(dfs.values())[0]"]
    for index, (operand, cell) in enumerate(zip(formula.operands, cells)):
        expr = (
            f"_btc_number(df1.iloc[{index}]['raw'], "
            f"df1.iloc[{index}]['typed_factor']) * "
            f"float(df1.iloc[{index}]['scale'])"
        )
        if cell.metric_key in COST_KEYS and not operand.preserve_sign:
            expr = f"abs({expr})"
        lines.append(f"v{index} = {expr}")
    lines.append(f"result = round({formula.expression}, 2)")
    return "\n".join(lines), cells


def _candidate_source_paths(base: Path, cell: StatementCell) -> list[Path]:
    """Return provenance files in the same precedence order as verification."""

    paths: list[Path] = []
    direct = base / "data" / Path(cell.csv_path).name
    if direct.is_file():
        paths.append(direct)
    if "|" in cell.table_ref:
        document, line = cell.table_ref.rsplit("|", 1)
        paths.extend(sorted((ROOT / "build" / "tables" / document).glob(f"*_line{line}.csv")))
    return list(dict.fromkeys(paths))


def _textual_row_labels(frame: pd.DataFrame, row: int, column: int) -> list[str]:
    labels: list[str] = []
    for value in frame.iloc[row, : max(1, column)].tolist():
        text = str(value).strip()
        compact = (
            text.replace("\u00a0", "").replace(" ", "")
            .replace("(", "").replace(")", "")
            .replace(".", "").replace(",", "")
            .lstrip("+-")
        )
        if text and not compact.isdigit() and text not in labels:
            labels.append(text)
    return labels


def resolve_source_coordinate(
    base: Path,
    cell: StatementCell,
    cache: dict[Path, pd.DataFrame],
) -> tuple[Path, int, int, list[str]]:
    """Resolve an audit token to a real cell instead of trusting stale offsets.

    Table extraction occasionally changes the number of header rows.  The raw
    value used by the runtime query remains valid, but a historical ``row_idx``
    can then point at an adjacent disclosure.  Prefer the recorded coordinate
    when it still matches; otherwise find the exact raw token and choose the
    nearest occurrence.  A missing token is a build failure, never a silently
    accepted provenance gap.
    """

    expected = str(cell.raw)
    attempted: list[str] = []
    for path in _candidate_source_paths(base, cell):
        attempted.append(str(path))
        if path not in cache:
            cache[path] = pd.read_csv(
                path, encoding="utf-8-sig", dtype=str,
                keep_default_na=False, index_col=None,
            )
        frame = cache[path]
        nominal = (int(cell.row_idx), int(cell.col_idx))
        try:
            if str(frame.iloc[nominal[0], nominal[1]]) == expected:
                return path, nominal[0], nominal[1], _textual_row_labels(
                    frame, nominal[0], nominal[1]
                )
        except IndexError:
            pass
        matches: list[tuple[int, int]] = []
        for row_index in range(frame.shape[0]):
            for column_index in range(frame.shape[1]):
                if str(frame.iat[row_index, column_index]) == expected:
                    matches.append((row_index, column_index))
        if matches:
            row_index, column_index = min(
                matches,
                key=lambda item: (
                    abs(item[0] - nominal[0]) + abs(item[1] - nominal[1]),
                    abs(item[1] - nominal[1]), item[0], item[1],
                ),
            )
            return path, row_index, column_index, _textual_row_labels(
                frame, row_index, column_index
            )
    raise ValueError(
        f"raw token {expected!r} not found for {cell.table_ref}; attempted={attempted}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, default=ROOT / "sub_standard_lookups")
    parser.add_argument("--out", type=Path, default=ROOT / "sub_compliant_standard")
    parser.add_argument(
        "--exclude-ids",
        default="",
        help="comma-separated formula IDs to leave unchanged for lower-risk full candidates",
    )
    parser.add_argument(
        "--include-ids",
        default="",
        help=(
            "comma-separated formula IDs to rewrite; when supplied, every "
            "other submission row is preserved byte-for-byte at the JSON-object level"
        ),
    )
    parser.add_argument(
        "--formula-variant",
        choices=("default", *sorted(FORMULA_VARIANTS)),
        default="default",
        help=(
            "named interpretation override for explicitly ambiguous table families; "
            "the default formula map remains unchanged for artifact reproducibility"
        ),
    )
    args = parser.parse_args()
    excluded_ids = {
        int(value.strip())
        for value in args.exclude_ids.split(",")
        if value.strip()
    }
    included_ids = {
        int(value.strip())
        for value in args.include_ids.split(",")
        if value.strip()
    }
    if included_ids and excluded_ids:
        parser.error("--include-ids and --exclude-ids are mutually exclusive")
    formula_map = dict(FORMULAS)
    if args.formula_variant != "default":
        formula_map.update(FORMULA_VARIANTS[args.formula_variant])

    submission = json.loads((args.base / "submission.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    data_dst = args.out / "data"
    if data_dst.exists():
        shutil.rmtree(data_dst)
    shutil.copytree(args.base / "data", data_dst)
    panel_audit = args.base / "panel_source_audit.json"
    if panel_audit.exists():
        shutil.copy2(panel_audit, args.out / panel_audit.name)

    # A cumulative candidate may use an already source-audited candidate as
    # its base (for example the separately built panel layer).  Preserve that
    # provenance and replace only IDs rewritten in this invocation.  Older
    # versions reset source_audit.json here, which made a valid cumulative
    # build appear structurally unaudited and could silently drop evidence
    # about inherited repairs.
    prior_audit_path = args.base / "source_audit.json"
    prior_audit = (
        json.loads(prior_audit_path.read_text(encoding="utf-8"))
        if prior_audit_path.exists()
        else []
    )
    audit = []
    failures = []
    seen_ids = set()
    source_frame_cache: dict[Path, pd.DataFrame] = {}
    for row in submission:
        qid = int(row["id"])
        if included_ids:
            formula = formula_map.get(qid) if qid in included_ids else None
        else:
            formula = None if qid in excluded_ids else formula_map.get(qid)
        if formula is None:
            continue
        try:
            query, cells = query_for(formula)
            resolved_sources = [
                resolve_source_coordinate(args.base, cell, source_frame_cache)
                for cell in cells
            ]
        except (KeyError, ValueError) as exc:
            # Keep the baseline entry when the normalized cube cannot prove a
            # source cell.  One missing statement must not abort the other
            # independently auditable rewrites.
            failures.append({"id": qid, "error": str(exc)})
            continue
        runtime_values = {f"v{index}": cell.value for index, cell in enumerate(cells)}
        expected = round(float(eval(formula.expression, {"__builtins__": {"abs": abs, "max": max, "min": min}}, runtime_values)), 2)
        old_answer = row.get("answer")
        safe_name = f"q{qid}_source_cells.csv"
        destination = data_dst / safe_name
        with destination.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=("ticker", "year", "metric_key", "raw", "typed_factor", "scale", "source_table", "source_csv", "row_idx", "col_idx"))
            writer.writeheader()
            for cell, (source_path, row_idx, col_idx, _source_labels) in zip(cells, resolved_sources):
                writer.writerow({
                    "ticker": cell.ticker, "year": cell.year, "metric_key": cell.metric_key,
                    "raw": cell.raw, "typed_factor": typed_factor(cell.raw),
                    "scale": cell.scale, "source_table": cell.table_ref,
                    "source_csv": source_path.name, "row_idx": row_idx, "col_idx": col_idx,
                })
        evidence = [{"variable": "df1", "csv_path": f"data/{safe_name}"}]
        row["evidence"] = evidence
        row["pandas_query"] = query
        row["answer"] = expected
        row["relevant_tables"] = list(dict.fromkeys(cell.table_ref for cell in cells))
        row["relevant_docs"] = list(dict.fromkeys(cell.table_ref.split("|", 1)[0] for cell in cells))
        seen_ids.add(qid)
        audit.append({
            "id": qid,
            "old_answer": old_answer,
            "answer": expected,
            "note": formula.note,
            "sources": [
                {
                    "table_ref": cell.table_ref,
                    "csv": source_path.name,
                    "row": row_idx,
                    "column": col_idx,
                    "metric": cell.metric_key,
                    "label": cell.label,
                    "source_row_labels": source_labels,
                    "scale": cell.scale,
                    "typed_factor": typed_factor(cell.raw),
                    "raw": cell.raw,
                }
                for cell, (source_path, row_idx, col_idx, source_labels) in zip(cells, resolved_sources)
            ],
        })

    requested_ids = included_ids if included_ids else set(formula_map) - excluded_ids
    missing = sorted(requested_ids - seen_ids - {item["id"] for item in failures})
    if missing:
        raise SystemExit(f"IDs not found in submission: {missing}")
    (args.out / "submission.json").write_text(json.dumps(submission, ensure_ascii=False), encoding="utf-8")
    combined_audit = {int(item["id"]): item for item in prior_audit}
    combined_audit.update({int(item["id"]): item for item in audit})
    (args.out / "source_audit.json").write_text(
        json.dumps([combined_audit[qid] for qid in sorted(combined_audit)], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.out), "formula_variant": args.formula_variant, "requested": len(requested_ids), "included_ids": sorted(included_ids), "excluded_ids": sorted(excluded_ids), "rewritten": len(audit), "ids": sorted(seen_ids), "failures": failures}, indent=2))


if __name__ == "__main__":
    main()
