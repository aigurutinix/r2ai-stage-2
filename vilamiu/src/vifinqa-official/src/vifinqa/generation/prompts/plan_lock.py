
from __future__ import annotations

from typing import TYPE_CHECKING

from vifinqa.generation.schemas import ConceptSelection, TableConceptMapping

if TYPE_CHECKING:
    from vifinqa.generation.common import CandidateTable

_SYSTEM = """\
Bạn đang ở bước KHÓA KẾ HOẠCH sau khi mapping đã chọn xong các bảng thực sự được dùng. Hãy chuẩn \
hóa kế hoạch ban đầu để áp dụng chính xác cho TOÀN BỘ và CHỈ các bảng được liệt kê bên dưới.

Yêu cầu:
- Giữ nguyên bản chất khái niệm và mục tiêu phân tích. Không phát minh chỉ tiêu mới để ép bảng.
- `concept_formula`, `population` và `calculation_steps` phải khớp đúng số bảng/đối tượng được \
chọn; không nhắc bất kỳ table_ref, công ty hoặc năm nào ngoài danh sách đã chọn.
- Chuẩn hóa kỳ đo lường theo nhãn và ngữ cảnh: phân biệt số dư đầu/cuối kỳ với chỉ tiêu phát sinh \
trong năm. Không gọi số dư đầu/cuối kỳ là tăng trưởng năm nếu bản chất không đúng.
- Khóa `measurement_basis` trên đúng các mapping: `gross` = giá gốc/trước dự phòng, `net` = giá \
trị thuần/sau dự phòng, `not_applicable` = metric không có phân biệt này. Tất cả bảng phải cùng một \
basis; nếu có mapping `unknown` hoặc trộn basis, trả `feasible=false`.
- Nếu `operation=difference`, hai đầu vào phải là cùng một metric qua kỳ/đối tượng, hoặc tạo thành \
một quan hệ chênh lệch/net/subtotal được dùng trong phân tích tài chính. Không khóa kế hoạch lấy hiệu \
giữa hai khoản mục khác bản chất chỉ vì cả hai cùng là số tiền.
- Với nhiều doanh nghiệp, không khóa chênh lệch tiền tuyệt đối nếu ý nghĩa chỉ là bên nào có quy mô \
lớn hơn. Cùng ngành, cùng kỳ và cùng đơn vị chưa đủ; `financial_rationale` phải nêu được quyết định \
phân tích cụ thể dựa trực tiếp trên mức tiền tuyệt đối.
- Không khóa phép chia trực tiếp số tiền của doanh nghiệp A cho số tiền của doanh nghiệp B rồi gọi là \
tỷ lệ/hệ số. So sánh chéo doanh nghiệp phải bắt đầu từ metric đã được chuẩn hóa trong từng công ty.
- `operation`, `answer_type` và `unit` phải xác định duy nhất một kết quả scalar.
- Nếu các bảng đã chọn không thể thực hiện cùng một kế hoạch nhất quán, trả `feasible=false`; không \
đổi nghĩa hoặc ghép các khoản mục gần tên để hợp thức hóa.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"feasible": true, "concept_name": "...", "concept_formula": "...", "operation": "difference|ratio|growth|share|sum|average|minimum|maximum|argmin|argmax|count|boolean", "answer_type": "money|percentage|number|year|company|boolean", "unit": "...", "table_topic": "...", "population": "...", "financial_rationale": "...", "measurement_basis": "gross|net|not_applicable|unknown", "calculation_steps": [], "reason": ""}
"""


def build_plan_lock_prompt(
    concept: ConceptSelection,
    tables: list[CandidateTable],
    mappings: list[TableConceptMapping],
) -> tuple[str, str]:
    hints = {mapping.table_ref: mapping.row_or_column_hint for mapping in mappings}
    blocks = "\n\n".join(
        f"""Bảng được chọn: {table.table_ref}
Công ty: {table.company_name} ({table.ticker}); năm: {table.year}; tài liệu: {table.doc_name}
Mapping: {hints.get(table.table_ref, '')}
Measurement basis đã map: {next((m.measurement_basis for m in mappings if m.table_ref == table.table_ref), 'unknown')}
Nhãn: {table.table_labels}
Ngữ cảnh: {table.surrounding_pages}"""
        for table in tables
    )
    user = f"""KẾ HOẠCH BAN ĐẦU
Khái niệm: {concept.concept_name}
Công thức: {concept.concept_formula}
Phép toán: {concept.operation}
Kiểu đáp án: {concept.answer_type}
Đơn vị: {concept.unit}
Chủ đề: {concept.table_topic}
Population: {concept.population}
Measurement basis ban đầu: {concept.measurement_basis}
Ý nghĩa: {concept.financial_rationale}
Các bước: {concept.calculation_steps}

ĐÚNG CÁC BẢNG ĐÃ CHỌN
{blocks}

Hãy khóa lại kế hoạch trên đúng tập bảng này."""
    return _SYSTEM, user
