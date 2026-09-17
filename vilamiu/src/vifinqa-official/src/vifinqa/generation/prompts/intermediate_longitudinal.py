
from __future__ import annotations

from vifinqa.generation.common import CandidateTable
from vifinqa.generation.intermediate_formulas.longitudinal import (
    LongitudinalInputMetric,
    LongitudinalScenarioPlan,
    TransformDefinition,
)

# Keep period handling explicit and deterministic.

_MAPPING_SYSTEM = """\
Bạn là chuyên gia phân tích tài chính đang xây bộ dữ liệu hỏi-đáp (QA) từ báo cáo tài chính doanh \
nghiệp Việt Nam đã OCR.

# MAPPING VAI TRÒ CÔNG THỨC LONGITUDINAL
Nhiệm vụ DUY NHẤT ở bước này: với MỘT chỉ tiêu tài chính đã cho, xác định ĐÚNG 1 ô (dòng, cột) cho
TỪNG cell `(công ty, năm)` được liệt kê bên dưới — tổng cộng 9 cell (3 công ty x 3 năm liên tiếp).

# QUY TẮC BẮT BUỘC
- Chỉ tiêu, 3 công ty và 3 năm đã được khóa trước — bạn KHÔNG được đổi chỉ tiêu, KHÔNG được
thêm/bớt công ty hoặc năm, KHÔNG được tự viết công thức tính. Bạn CHỈ định vị ô dữ liệu.
- Với mỗi cell, nếu tìm thấy ô đúng nghĩa: đặt `found=true`, `table_ref` đúng bằng mã bảng đã cho,
`row_label` COPY NGUYÊN VĂN đúng ô nhãn dòng (thường là cột đầu tiên/tên khoản mục) trong CSV, và
`column_label` COPY NGUYÊN VĂN đúng tên cột (dòng header CSV) chứa giá trị của ĐÚNG năm đó.
- `role_id` của mỗi cell có dạng `"<mã công ty>__<năm>"` — PHẢI giữ nguyên định dạng này khi trả lời.
- Nếu KHÔNG có bảng nào chứa đúng khoản mục cho cell đó, đặt `found=false` và giải thích ngắn ở
`reason` — TUYỆT ĐỐI không chọn đại 1 dòng gần giống nhưng khác bản chất kinh tế, và TUYỆT ĐỐI
không lấy nhầm cột của năm khác.
- Phải trả về ĐỦ 1 phần tử `mappings` cho MỖI cell được liệt kê bên dưới, đúng `role_id`.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"mappings": [{"role_id": "...", "found": true, "table_ref": "...", "row_label": "...", "column_label": "...", "reason": ""}, ...]}
"""


def cell_role_id(ticker: str, period: str) -> str:
    return f"{ticker}__{period}"


def _table_block(table_ref: str, table: CandidateTable) -> str:
    return f"""### Bảng: {table_ref}
Công ty: {table.company_name} (mã {table.ticker})
Tài liệu: {table.doc_name} (năm {table.year})

Nội dung CSV:
{table.csv_text}
"""


def build_longitudinal_cell_mapping_prompt(
    metric: LongitudinalInputMetric,
    entities: tuple[str, str, str],
    entity_names: dict[str, str],
    periods: tuple[str, str, str],
    tables_by_cell: dict[tuple[str, str], list[CandidateTable]],
) -> tuple[str, str]:
    cell_lines: list[str] = []
    seen_tables: dict[str, CandidateTable] = {}
    for ticker in entities:
        for period in periods:
            candidates = tables_by_cell.get((ticker, period), [])
            refs = ", ".join(c.table_ref for c in candidates) or "(không có ứng viên)"
            cell_lines.append(
                f"- role_id=`{cell_role_id(ticker, period)}` "
                f"({entity_names.get(ticker, ticker)}, năm {period}). Bảng ứng viên: {refs}"
            )
            for c in candidates:
                seen_tables.setdefault(c.table_ref, c)

    blocks = "\n\n".join(_table_block(ref, table) for ref, table in seen_tables.items())
    user = (
        f"Chỉ tiêu: {metric.label_vi} ({' / '.join(metric.concept_names)})\n\n"
        "CÁC CELL CẦN ĐỊNH VỊ:\n" + "\n".join(cell_lines) + "\n\n"
        f"CÁC BẢNG ỨNG VIÊN:\n{blocks}\n\n"
        "Hãy định vị đúng ô cho từng cell, trả về đúng JSON yêu cầu."
    )
    return _MAPPING_SYSTEM, user



_FINANCE_JUDGE_SYSTEM = """\
Bạn là kiểm soát viên chất lượng dữ liệu tài chính. Nhiệm vụ: kiểm tra 9 ô đã định vị (bên dưới) \
có ĐÚNG là quan sát thật của MỘT chỉ tiêu hay không — KHÔNG được sửa lại binding, chỉ được kết luận \
đạt/không đạt.

# LƯU Ý QUAN TRỌNG VỀ ĐƠN VỊ/SCALE
Giá trị của mỗi ô đã được hệ thống tự động quy đổi về đơn vị đồng chuẩn dựa trên khai báo "nghìn/
triệu/tỷ đồng" trong header cột — bạn KHÔNG cần và KHÔNG được từ chối chỉ vì các công ty khác nhau \
khai báo scale khác nhau (ví dụ công ty A ghi "Triệu VND" còn công ty B ghi "VND đầy đủ") — đây là \
BÌNH THƯỜNG và đã được xử lý đúng. Phép tính cuối cùng chỉ so sánh TỶ LỆ TĂNG TRƯỞNG tính riêng cho \
từng công ty, không cộng/so trực tiếp số tiền tuyệt đối giữa các công ty.

Trả `valid=false` nếu:
- Một ô có tên dòng KHÔNG đúng ý nghĩa kinh tế của chỉ tiêu đã cho (khác khoản mục).
- Một ô lấy nhầm cột của năm khác (column_label không khớp năm của cell đó).
- Ba năm của CÙNG một công ty trộn lẫn measurement basis (gross/net) với nhau.
- Giá trị bất thường rõ ràng cho thấy chọn nhầm dòng/cột (không liên quan tới việc khác scale với
công ty khác).

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"valid": true, "reason": ""}
"""


def build_longitudinal_finance_judge_prompt(
    metric: LongitudinalInputMetric, plan: LongitudinalScenarioPlan
) -> tuple[str, str]:
    lines = [
        f"- {plan.entity_names.get(ticker, ticker)} ({ticker}), năm {period}: bảng {cell.table_ref}, "
        f"dòng {cell.row_label!r}, cột {cell.column_label!r}, giá trị thô {cell.raw_value!r}"
        for (ticker, period), cell in sorted(plan.bindings.items())
    ]
    user = (
        f"Chỉ tiêu: {metric.label_vi} ({' / '.join(metric.concept_names)})\n"
        f"Phạm vi báo cáo: {plan.report_scope}\n\n"
        "CÁC Ô ĐÃ ĐỊNH VỊ:\n" + "\n".join(lines) + "\n\n"
        "Hãy kết luận các ô này có đúng cùng một chỉ tiêu, nhất quán qua công ty/năm hay không."
    )
    return _FINANCE_JUDGE_SYSTEM, user



_QUESTION_SYSTEM = """\
Bạn là chuyên gia phân tích tài chính đang xây bộ dữ liệu hỏi-đáp (QA).

# VIẾT CÂU HỎI LONGITUDINAL
Nhiệm vụ DUY NHẤT ở bước này: viết 1 câu hỏi tự nhiên hỏi kết quả tổng hợp (reducer) của một phép
biến đổi (transform) trên MỘT chỉ tiêu, áp dụng cho ĐÚNG 3 công ty qua ĐÚNG 3 năm liên tiếp đã cho.

# RÀNG BUỘC CÂU HỎI
- CHỈ hỏi về ĐÚNG chỉ tiêu/transform/reducer đã cho — KHÔNG tự đổi công thức, KHÔNG ghép thêm chỉ
tiêu khác, KHÔNG nêu dòng/cột cụ thể sẽ dùng để tính.
- BẮT BUỘC nêu rõ CẢ 3 công ty (tên đầy đủ và/hoặc mã) và CẢ 3 năm (hoặc khoảng năm liên tiếp, ví
dụ "giai đoạn 2021-2023" — miễn đủ 3 năm này xuất hiện trong câu).
- Câu hỏi tự nhiên như người thật hỏi; không nhại tên dòng/cột CSV; không cộc lốc.
- Đáp án phải là ĐÚNG 1 số (không hỏi công ty nào/năm nào).
- Không dùng "tại ngày", "tại thời điểm", "tại cuối năm"; không mở đầu bằng "trong báo cáo"/"theo
báo cáo"/"trong BCTC"/"theo BCTC"; không viết tắt "VND".
- KHÔNG dùng mẫu câu mở đầu cố định (ví dụ luôn mở đầu bằng "Xét riêng..."); xáo trộn linh hoạt
cách mở đầu và cấu trúc câu giữa các lần viết khác nhau, giống nhiều người khác nhau đặt câu hỏi.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"question": "..."}
"""


_REDUCER_LABEL_VI = {
    "average": "trung bình",
    "median": "trung vị",
    "minimum": "giá trị nhỏ nhất",
    "maximum": "giá trị lớn nhất",
    "range": "khoảng biến động (giá trị lớn nhất trừ giá trị nhỏ nhất)",
}


def build_longitudinal_question_prompt(
    metric: LongitudinalInputMetric,
    transform: TransformDefinition,
    reducer_id: str,
    plan: LongitudinalScenarioPlan,
) -> tuple[str, str]:
    entities_desc = "; ".join(
        f"{plan.entity_names.get(ticker, ticker)} ({ticker})" for ticker in plan.entities
    )
    user = (
        f"Chỉ tiêu đầu vào: {metric.label_vi}\n"
        f"Transform riêng từng công ty: {transform.name_vi} ({transform.formula_text})\n"
        f"Reducer tổng hợp qua 3 công ty: {_REDUCER_LABEL_VI.get(reducer_id, reducer_id)}\n"
        f"Công ty: {entities_desc}\n"
        f"Các năm: {', '.join(plan.periods)}\n"
        f"Phạm vi báo cáo: {'công ty mẹ' if plan.report_scope == 'parent' else 'hợp nhất (mặc định)'}\n\n"
        "Hãy viết 1 câu hỏi theo mô tả ở trên, trả về đúng JSON yêu cầu."
    )
    return _QUESTION_SYSTEM, user
