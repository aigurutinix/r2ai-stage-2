
from __future__ import annotations

from vifinqa.generation.schemas import ConceptSelection

_SYSTEM = """\
Bạn là kiểm soát viên contract cho dữ liệu QA tài chính. Hãy kiểm tra câu hỏi có khớp chính xác \
kế hoạch tính toán đã khóa hay không.

Trả `valid=false` nếu câu hỏi đổi phép toán, đổi kiểu đáp án, đổi đơn vị, đổi gross/net, thiếu hoặc thêm đối tượng, \
đổi kỳ/phạm vi báo cáo, hỏi mô tả nhiều giá trị thay vì một scalar, hoặc thêm một yêu cầu không có \
trong kế hoạch. Danh tính và phạm vi bảng bên dưới là một phần của contract đã khóa: câu hỏi được \
phép và phải nêu phạm vi `parent` bằng cách nói công ty mẹ; không coi thông tin này là tự thêm chỉ \
vì nó không nằm trong `concept_formula`. Phạm vi `consolidated` là mặc định và KHÔNG được yêu cầu \
câu hỏi nêu báo cáo hợp nhất; việc không nhắc scope này không phải là thiếu thông tin. Phạm vi \
`parent` phải được diễn đạt trực tiếp là chỉ tiêu "của công ty mẹ <doanh nghiệp>", không đóng khung \
nguồn bằng "trong/theo báo cáo" hoặc "trong/theo BCTC". Không sửa câu hỏi.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"valid": true, "reason": ""}
"""


def build_alignment_judge_prompt(
    concept: ConceptSelection, question: str, table_identities: str
) -> tuple[str, str]:
    user = f"""KẾ HOẠCH ĐÃ KHÓA
Khái niệm: {concept.concept_name}
Công thức: {concept.concept_formula}
Phép toán: {concept.operation}
Kiểu đáp án: {concept.answer_type}
Đơn vị: {concept.unit}
Population: {concept.population}
Measurement basis: {concept.measurement_basis}
Các bước: {concept.calculation_steps}

DANH TÍNH VÀ PHẠM VI BẢNG ĐƯỢC DÙNG
{table_identities}

CÂU HỎI CẦN KIỂM TRA
{question}"""
    return _SYSTEM, user
