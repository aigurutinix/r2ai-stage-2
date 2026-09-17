
from __future__ import annotations

from vifinqa.common.corpus.company_meta import CompanyInfo
from vifinqa.generation.intermediate_formulas.base import FormulaDefinition, FormulaRole

FORMULAS: dict[str, FormulaDefinition] = {
    "roa": FormulaDefinition(
        formula_id="roa",
        name_vi="Tỷ suất sinh lời trên tổng tài sản (ROA)",
        formula_text=(
            "Lợi nhuận sau thuế / Bình quân tổng tài sản (trung bình số đầu năm và số cuối năm)"
        ),
        required_roles=(
            FormulaRole(
                role_id="net_income",
                label_vi="Lợi nhuận sau thuế",
                concept_names=("Lợi nhuận sau thuế", "Lợi nhuận sau thuế thu nhập doanh nghiệp"),
            ),
            FormulaRole(
                role_id="total_assets_begin",
                label_vi="Tổng cộng tài sản đầu năm",
                concept_names=("Tổng cộng tài sản", "Tổng tài sản"),
            ),
            FormulaRole(
                role_id="total_assets_end",
                label_vi="Tổng cộng tài sản cuối năm",
                concept_names=("Tổng cộng tài sản", "Tổng tài sản"),
            ),
        ),
        answer_type="percentage",
        unit="%",
        time_basis="Lợi nhuận sau thuế là số phát sinh trong năm báo cáo; tổng tài sản lấy đúng cột Số đầu năm/Số cuối năm của CÙNG năm báo cáo đó.",
        allowed_report_scopes=("consolidated", "parent"),
        industry_policy="Loại tổ chức tín dụng (ngân hàng) — chart of accounts khác Thông tư 200 phi tài chính.",
        denominator_policy="Bình quân tổng tài sản (đầu kỳ + cuối kỳ)/2 phải > 0.",
        sign_policy="Lợi nhuận sau thuế có thể âm (doanh nghiệp lỗ); không loại vì âm.",
        terminology_notes="ROA là thuật ngữ chuẩn, dùng phổ biến trong phân tích tài chính doanh nghiệp.",
        source_urls=(
            "https://www.investopedia.com/terms/r/returnonassets.asp",
        ),
        enabled=True,
    ),
    "roe": FormulaDefinition(
        formula_id="roe",
        name_vi="Tỷ suất sinh lời trên vốn chủ sở hữu (ROE)",
        formula_text=(
            "Lợi nhuận sau thuế / Bình quân vốn chủ sở hữu (trung bình số đầu năm và số cuối năm)"
        ),
        required_roles=(
            FormulaRole(
                role_id="net_income",
                label_vi="Lợi nhuận sau thuế",
                concept_names=("Lợi nhuận sau thuế", "Lợi nhuận sau thuế thu nhập doanh nghiệp"),
            ),
            FormulaRole(
                role_id="equity_begin",
                label_vi="Vốn chủ sở hữu đầu năm",
                concept_names=("Vốn chủ sở hữu",),
            ),
            FormulaRole(
                role_id="equity_end",
                label_vi="Vốn chủ sở hữu cuối năm",
                concept_names=("Vốn chủ sở hữu",),
            ),
        ),
        answer_type="percentage",
        unit="%",
        time_basis="Lợi nhuận sau thuế là số phát sinh trong năm báo cáo; vốn chủ sở hữu lấy đúng cột Số đầu năm/Số cuối năm của CÙNG năm báo cáo đó.",
        allowed_report_scopes=("consolidated", "parent"),
        industry_policy="Loại tổ chức tín dụng (ngân hàng) — chart of accounts khác Thông tư 200 phi tài chính.",
        denominator_policy="Bình quân vốn chủ sở hữu (đầu kỳ + cuối kỳ)/2 phải > 0.",
        sign_policy="Lợi nhuận sau thuế có thể âm (doanh nghiệp lỗ); không loại vì âm.",
        terminology_notes="ROE là thuật ngữ chuẩn, dùng phổ biến trong phân tích tài chính doanh nghiệp.",
        source_urls=(
            "https://www.investopedia.com/terms/r/returnonequity.asp",
        ),
        enabled=True,
    ),
    "quick_ratio": FormulaDefinition(
        formula_id="quick_ratio",
        name_vi="Hệ số thanh toán nhanh (Quick ratio)",
        formula_text="(Tài sản ngắn hạn - Hàng tồn kho) / Nợ ngắn hạn",
        required_roles=(
            FormulaRole(
                role_id="current_assets",
                label_vi="Tài sản ngắn hạn",
                concept_names=("Tài sản ngắn hạn",),
            ),
            FormulaRole(
                role_id="inventory",
                label_vi="Hàng tồn kho",
                concept_names=("Hàng tồn kho",),
            ),
            FormulaRole(
                role_id="current_liabilities",
                label_vi="Nợ ngắn hạn",
                concept_names=("Nợ ngắn hạn",),
            ),
        ),
        answer_type="number",
        unit="lần",
        time_basis="Point-in-time — cả 3 role lấy đúng cùng 1 cột (Số cuối năm) của cùng năm báo cáo.",
        allowed_report_scopes=("consolidated", "parent"),
        industry_policy="Loại tổ chức tín dụng (ngân hàng) — không có tài sản/nợ ngắn hạn tách biệt như doanh nghiệp phi tài chính.",
        denominator_policy="Nợ ngắn hạn phải > 0.",
        sign_policy="Tài sản ngắn hạn, hàng tồn kho, nợ ngắn hạn đều phải >= 0 (khoản mục bảng cân đối kế toán).",
        terminology_notes="Quick ratio (hệ số thanh toán nhanh/hệ số khả năng thanh toán nhanh) là thuật ngữ chuẩn.",
        source_urls=(
            "https://www.investopedia.com/terms/q/quickratio.asp",
        ),
        enabled=True,
    ),
}

FORMULA_EXPRESSIONS: dict[str, str] = {
    "roa": "net_income / ((total_assets_begin + total_assets_end) / 2)",
    "roe": "net_income / ((equity_begin + equity_end) / 2)",
    "quick_ratio": "(current_assets - inventory) / current_liabilities",
}

_CREDIT_INSTITUTION_INDUSTRY_L2 = "Tổ chức tín dụng"


def get_formula(formula_id: str) -> FormulaDefinition:
    return FORMULAS[formula_id]


def enabled_formulas() -> list[FormulaDefinition]:
    return [f for f in FORMULAS.values() if f.enabled]


def formula_excludes_company(formula: FormulaDefinition, company: CompanyInfo | None) -> bool:
    if company is None:
        return True
    return company.industry_l2 == _CREDIT_INSTITUTION_INDUSTRY_L2


def compute_formula(formula_id: str, values: dict[str, float]) -> float:
    expression = FORMULA_EXPRESSIONS[formula_id]
    return eval(expression, {"__builtins__": {}}, values)  # noqa: S307
