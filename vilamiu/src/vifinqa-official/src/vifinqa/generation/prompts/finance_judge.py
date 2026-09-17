
from __future__ import annotations

from typing import TYPE_CHECKING

from vifinqa.generation.schemas import ConceptSelection, TableConceptMapping

if TYPE_CHECKING:
    from vifinqa.generation.common import CandidateTable

_SYSTEM = """\
Bạn là kiểm soát viên chất lượng dữ liệu tài chính. Hãy đánh giá KẾ HOẠCH TÍNH TOÁN bên dưới trước \
khi viết câu hỏi. Nếu có `pandas_query`, phải kiểm tra code thực hiện đúng kế hoạch trên các bảng \
đã map. Không viết lại kế hoạch, query hoặc đề xuất một câu hỏi khác.

Trả `valid=false` nếu có bất kỳ lỗi nào sau:
- Các bảng nói về khoản mục, bản chất kinh tế, kỳ đo lường hoặc phạm vi báo cáo khác nhau.
- Tử số và mẫu số không cùng population; một bên là toàn danh mục nhưng bên kia chỉ là một phần.
- So sánh các tập đối tượng thay đổi qua các năm rồi lấy tổng/trung bình như thể cùng population.
- Cộng, trừ, chia hoặc lấy trung bình các ô chỉ vì cùng xuất hiện trong bảng, nhưng kết quả không \
phải chỉ tiêu được dùng trong phân tích tài chính hoặc không trả lời một quyết định thực tế.
- Với phép trừ: hai số là các khoản mục khác bản chất và kết quả chỉ là khoảng cách quy mô tùy ý. \
Chỉ chấp nhận chênh lệch của cùng một metric qua kỳ/đối tượng, hoặc quan hệ net/subtotal được giới \
phân tích tài chính sử dụng; cùng đơn vị tiền là chưa đủ.
- Chênh lệch tiền tuyệt đối giữa các doanh nghiệp mà kết quả chủ yếu chỉ phản ánh bên nào có quy mô \
lớn hơn. Mặc định phải loại; chỉ chấp nhận khi `financial_rationale` giải thích được một quyết định \
phân tích cụ thể dựa trực tiếp trên mức tiền tuyệt đối, không phụ thuộc chủ yếu vào quy mô. Cùng \
ngành, cùng kỳ, cùng đơn vị và cùng tên khoản mục tự chúng chưa đủ.
- Lấy một số tiền tuyệt đối của doanh nghiệp A chia cho số tiền của doanh nghiệp B rồi gọi thương số \
là "tỷ lệ", "hệ số" hoặc "quy mô tương đối". Đây không phải tỷ số tài chính chuẩn; muốn so chéo \
doanh nghiệp phải dùng metric đã chuẩn hóa trong từng doanh nghiệp.
- Công thức, operation, answer_type hoặc unit mâu thuẫn nhau.
- Gợi ý mapping giữa các bảng dùng các dòng/cột có tên gần giống nhưng khác bản chất.
- `measurement_basis` chưa xác định, hoặc các observation trộn giá gốc/trước dự phòng (`gross`) \
với giá trị thuần/sau dự phòng (`net`). Với metric không có phân biệt này, tất cả mapping phải là \
`not_applicable`.
- `pandas_query` đổi công thức, population, kỳ, đơn vị hoặc tự chèn một phép hiệu chỉnh không có \
trong kế hoạch. Kết quả phải được tính từ dữ liệu CSV theo đúng công thức, không được nắn để khớp \
một con số định trước.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"valid": true, "reason": ""}
"""


def build_finance_judge_prompt(
    concept: ConceptSelection,
    tables: list[CandidateTable],
    mappings: list[TableConceptMapping],
    *,
    pandas_query: str | None = None,
    actual_result: object | None = None,
) -> tuple[str, str]:
    hints = {mapping.table_ref: mapping.row_or_column_hint for mapping in mappings}
    table_blocks = "\n\n".join(
        f"""Bảng: {table.table_ref}
Công ty: {table.company_name} ({table.ticker}); năm: {table.year}; tài liệu: {table.doc_name}
Mapping dự kiến: {hints.get(table.table_ref, '')}
Measurement basis đã map: {next((m.measurement_basis for m in mappings if m.table_ref == table.table_ref), 'unknown')}
Nhãn bảng: {table.table_labels}
Ngữ cảnh: {table.surrounding_pages}"""
        for table in tables
    )
    query_block = (
        f"\n\nPANDAS QUERY ĐÃ CHẠY TRÊN CSV GỐC\n{pandas_query}\n\nRESULT THỰC THI\n{actual_result!r}"
        if pandas_query is not None
        else ""
    )
    user = f"""KẾ HOẠCH TÍNH TOÁN
Khái niệm: {concept.concept_name}
Công thức: {concept.concept_formula}
Phép toán: {concept.operation}
Kiểu đáp án: {concept.answer_type}
Đơn vị: {concept.unit}
Chủ đề bảng: {concept.table_topic}
Population phải nhất quán: {concept.population}
Measurement basis phải nhất quán: {concept.measurement_basis}
Ý nghĩa tài chính: {concept.financial_rationale}
Các bước tính phụ thuộc nhau: {concept.calculation_steps}

CÁC BẢNG VÀ MAPPING
{table_blocks}
{query_block}

Hãy kết luận kế hoạch và query có ý nghĩa tài chính, nhất quán và khớp nhau hay không."""
    return _SYSTEM, user
