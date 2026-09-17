
from __future__ import annotations

from vifinqa.generation.common import CandidateTable
from vifinqa.generation.intermediate_formulas.base import FormulaDefinition, FormulaScenarioPlan


_MAPPING_SYSTEM = """\
Bạn là chuyên gia phân tích tài chính đang xây bộ dữ liệu hỏi-đáp (QA) từ báo cáo tài chính doanh \
nghiệp Việt Nam đã OCR.

# MAPPING VAI TRÒ CÔNG THỨC
Nhiệm vụ DUY NHẤT ở bước này: với công thức tài chính đã cho, xác định ĐÚNG 1 ô (dòng, cột) trong \
các bảng bên dưới cho từng "role" (thành phần input) của công thức.

# QUY TẮC BẮT BUỘC
- Công thức và role đã được khóa trước — bạn KHÔNG được đổi công thức, KHÔNG được thêm/bớt role, \
KHÔNG được tự viết biểu thức tính. Bạn CHỈ định vị ô dữ liệu.
- Với mỗi role, nếu tìm thấy ô đúng nghĩa: đặt `found=true`, `table_ref` đúng bằng mã bảng đã cho, \
`row_label` COPY NGUYÊN VĂN (không sửa dấu, không viết tắt, không rút gọn, không tự thêm/bớt \
khoảng trắng) đúng ô nhãn dòng (thường là cột đầu tiên/tên khoản mục) trong CSV, và `column_label` \
COPY NGUYÊN VĂN đúng tên cột (dòng header CSV) chứa giá trị đó.
- Hai role khác nhau có thể cùng `table_ref` (ví dụ 2 cột "Số đầu năm"/"Số cuối năm" của CÙNG 1 \
dòng trong CÙNG 1 bảng) — đây là bình thường, không phải lỗi.
- Nếu KHÔNG có bảng nào chứa đúng khoản mục cho role đó, đặt `found=false` và giải thích ngắn ở \
`reason` — TUYỆT ĐỐI không chọn đại 1 dòng gần giống nhưng khác bản chất kinh tế (ví dụ không dùng \
"Tài sản dài hạn" thay cho "Tài sản ngắn hạn", không dùng "Lợi nhuận trước thuế" thay cho "Lợi \
nhuận sau thuế").
- KHÔNG tự tính chỉ số nào ở bước này, KHÔNG suy đoán giá trị bằng số học, CHỈ định vị vị trí ô.
- Phải trả về ĐỦ 1 phần tử `mappings` cho MỖI role được liệt kê bên dưới, đúng `role_id`.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"mappings": [{"role_id": "...", "found": true, "table_ref": "...", "row_label": "...", "column_label": "...", "reason": ""}, ...]}
"""


def _table_block(table_ref: str, table: CandidateTable) -> str:
    return f"""### Bảng: {table_ref}
Công ty: {table.company_name} (mã {table.ticker})
Tài liệu: {table.doc_name} (năm {table.year})

Nội dung CSV:
{table.csv_text}
"""


def build_formula_role_mapping_prompt(
    formula: FormulaDefinition, tables_by_role: dict[str, list[CandidateTable]]
) -> tuple[str, str]:
    role_lines: list[str] = []
    seen_tables: dict[str, CandidateTable] = {}
    for role in formula.required_roles:
        candidates = tables_by_role.get(role.role_id, [])
        refs = ", ".join(c.table_ref for c in candidates) or "(không có ứng viên)"
        role_lines.append(
            f"- role_id=`{role.role_id}` ({role.label_vi}, khái niệm chuẩn: "
            f"{' / '.join(role.concept_names)}). Bảng ứng viên: {refs}"
        )
        for c in candidates:
            seen_tables.setdefault(c.table_ref, c)

    blocks = "\n\n".join(_table_block(ref, table) for ref, table in seen_tables.items())
    user = (
        f"Công thức: {formula.name_vi} = {formula.formula_text}\n\n"
        "CÁC ROLE CẦN ĐỊNH VỊ:\n" + "\n".join(role_lines) + "\n\n"
        f"CÁC BẢNG ỨNG VIÊN:\n{blocks}\n\n"
        "Hãy định vị đúng ô cho từng role, trả về đúng JSON yêu cầu."
    )
    return _MAPPING_SYSTEM, user



_FINANCE_JUDGE_SYSTEM = """\
Bạn là kiểm soát viên chất lượng dữ liệu tài chính. Nhiệm vụ: kiểm tra các Ô đã định vị (bên dưới) \
có ĐÚNG là các thành phần input thật của công thức đã cho hay không — KHÔNG được sửa lại binding, \
chỉ được kết luận đạt/không đạt.

Trả `valid=false` nếu:
- Một ô có tên dòng/cột KHÔNG đúng ý nghĩa kinh tế của role (ví dụ dùng "Lợi nhuận trước thuế" thay \
"Lợi nhuận sau thuế", dùng "Tài sản dài hạn" thay "Tài sản ngắn hạn").
- Giá trị bất thường rõ ràng cho thấy chọn nhầm dòng/cột (ví dụ số quá nhỏ/lớn bất thường so với \
các role khác cùng đơn vị tiền, dấu không hợp lý với bản chất khoản mục).
- Các ô không cùng năm báo cáo, hoặc không cùng phạm vi báo cáo (hợp nhất/công ty mẹ).

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"valid": true, "reason": ""}
"""


def build_formula_finance_judge_prompt(
    formula: FormulaDefinition, plan: FormulaScenarioPlan
) -> tuple[str, str]:
    role_by_id = {role.role_id: role for role in formula.required_roles}
    lines = [
        f"- {role_by_id[role_id].label_vi} (role_id={role_id}): bảng {cell.table_ref}, "
        f"dòng {cell.row_label!r}, cột {cell.column_label!r}, giá trị thô {cell.raw_value!r}"
        for role_id, cell in plan.bindings.items()
    ]
    user = (
        f"Công thức: {formula.name_vi} = {formula.formula_text}\n"
        f"Công ty: {plan.entity_name} ({plan.entity_ticker}); năm {plan.year}; "
        f"phạm vi {plan.report_scope}\n\n"
        "CÁC Ô ĐÃ ĐỊNH VỊ:\n" + "\n".join(lines) + "\n\n"
        "Hãy kết luận các ô này có đúng vai trò kinh tế của công thức hay không."
    )
    return _FINANCE_JUDGE_SYSTEM, user



_QUESTION_SYSTEM = """\
Bạn là chuyên gia phân tích tài chính đang xây bộ dữ liệu hỏi-đáp (QA).

# VIẾT CÂU HỎI CÔNG THỨC
Nhiệm vụ DUY NHẤT ở bước này: viết 1 câu hỏi tự nhiên hỏi giá trị của MỘT chỉ số tài chính đã khóa \
cho MỘT công ty.

# RÀNG BUỘC CÂU HỎI
- CHỈ hỏi về ĐÚNG chỉ số đã cho (dùng tên chuẩn hoặc cách diễn đạt tự nhiên tương đương) — KHÔNG tự \
đổi công thức, KHÔNG ghép thêm chỉ số khác, KHÔNG nêu dòng/cột cụ thể sẽ dùng để tính.
- Câu hỏi tự nhiên như người thật hỏi; không nhại tên dòng/cột CSV; không cộc lốc.
- Nêu rõ CÔNG TY (tên đầy đủ + mã trong ngoặc) và NĂM báo cáo.
- Nếu là báo cáo CÔNG TY MẸ (không phải hợp nhất), phải nêu rõ "công ty mẹ <tên>"; hợp nhất là mặc \
định, KHÔNG cần nhắc.
- Đáp án phải là ĐÚNG 1 số (không hỏi công ty nào/năm nào, không hỏi Có/Không).
- Không dùng "tại ngày", "tại thời điểm", "tại cuối năm"; không mở đầu bằng "trong báo cáo"/"theo \
báo cáo"/"trong BCTC"/"theo BCTC"; không viết tắt "VND".
- KHÔNG dùng mẫu câu mở đầu cố định; xáo trộn linh hoạt cách mở đầu và cấu trúc câu giữa các lần \
viết khác nhau, giống nhiều người khác nhau đặt câu hỏi.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"question": "..."}
"""


def build_formula_question_prompt(
    formula: FormulaDefinition, plan: FormulaScenarioPlan
) -> tuple[str, str]:
    scope_label = "công ty mẹ" if plan.report_scope == "parent" else "hợp nhất (mặc định)"
    user = (
        f"Chỉ số đã khóa: {formula.name_vi} = {formula.formula_text}\n"
        f"Đơn vị: {plan.unit}\n"
        f"Công ty: {plan.entity_name} (mã {plan.entity_ticker})\n"
        f"Năm báo cáo: {plan.year}\n"
        f"Phạm vi báo cáo: {scope_label}\n\n"
        "Hãy viết 1 câu hỏi theo mô tả ở trên, trả về đúng JSON yêu cầu."
    )
    return _QUESTION_SYSTEM, user
