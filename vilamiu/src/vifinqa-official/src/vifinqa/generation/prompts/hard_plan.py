
from __future__ import annotations

from typing import TYPE_CHECKING

from vifinqa.generation.hard.schemas import (
    HardMultiMetricPlanDraft,
    HardP3Plan,
    HardPlan,
    MetricBinding,
)

if TYPE_CHECKING:
    from vifinqa.generation.common import CandidateTable
    from vifinqa.generation.retrieval.base import TableDescriptor


def _label_block(table: CandidateTable) -> str:
    return f"""### Bảng: {table.table_ref}
Công ty: {table.company_name} (mã {table.ticker}); năm: {table.year}; tài liệu: {table.doc_name}
Nhãn bảng: {table.table_labels}
Ngữ cảnh trang lân cận: {table.surrounding_pages}"""


def build_hard_plan_prompt(
    candidates: list[CandidateTable], *, min_entities: int
) -> tuple[str, str]:
    system = """Bạn là chuyên gia phân tích tài chính. Các bảng dưới đây thuộc nhiều công ty \
cùng ngành, cùng năm. Hãy chọn MỘT chỉ tiêu tài chính (metric) có cùng bản chất, cùng đơn vị, \
xuất hiện được ở ít nhất __MIN_ENTITIES__ công ty, để sau đó hệ thống lọc công ty theo 1 ngưỡng \
trên chỉ tiêu này rồi đếm/tổng hợp — bạn KHÔNG cần tự nghĩ ngưỡng, không viết code, không liệt kê \
các bước tính. Các bảng còn lại là ứng viên dự phòng cho bước mapping; không bắt buộc chỉ tiêu phải \
xuất hiện trong toàn bộ bảng.

Yêu cầu:
- `metric_role`: tên ngắn gọn, không dấu, dùng làm khoá nội bộ (vd "loi_nhuan_sau_thue").
- `concept_formula`: mô tả đầy đủ khái niệm sẽ lọc/đếm (vd "đếm số công ty có lợi nhuận sau thuế \
năm 2023 vượt một ngưỡng").
- `final_operation`: `count` (đếm số công ty vượt ngưỡng), `sum` (tổng chỉ tiêu của các công ty đã \
lọc — CHỈ khi việc cộng các công ty đã lọc có ý nghĩa phân tích thật), hoặc `boolean` (có tồn tại \
công ty nào vượt ngưỡng không).
- `answer_type` là kiểu của ĐÁP ÁN CUỐI: `count` bắt buộc là `number`, `boolean` bắt buộc là \
`boolean`; với `sum` dùng `money`/`percentage`/`number` theo chỉ tiêu được cộng.
- `metric_value_kind` là kiểu giá trị CỦA CHÍNH metric_role dùng để lọc (độc lập với `answer_type` \
— vd `final_operation=count` thì `answer_type=number` nhưng metric_role vẫn có thể là tiền): \
`money` (chỉ tiêu tiền tệ, cần đơn vị/scale), `percentage` (tỷ lệ %), `number` (số đếm/tỷ số \
không mang tiền).
- `comparison`: `gt`/`gte` nếu câu hỏi tự nhiên nên hỏi "vượt/trên", `lt`/`lte` nếu nên hỏi "dưới".
- Không chọn chỉ tiêu có ở ít hơn __MIN_ENTITIES__ công ty hoặc cần phân biệt gross/net mà ngữ cảnh \
chưa rõ trên tập công ty sẽ dùng.
- Nếu không có chỉ tiêu nào đủ chung cho ít nhất __MIN_ENTITIES__ công ty, trả `feasible=false`.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"feasible": true, "concept_name": "...", "metric_role": "...", "concept_formula": "...", \
"financial_rationale": "...", "population": "...", "unit": "...", \
"answer_type": "money|percentage|number|boolean", \
"metric_value_kind": "money|percentage|number", \
"measurement_basis": "gross|net|not_applicable|unknown", \
"final_operation": "count|sum|boolean", "comparison": "gt|gte|lt|lte", "reason": ""}"""
    system = system.replace("__MIN_ENTITIES__", str(min_entities))
    return system, "\n\n".join(_label_block(table) for table in candidates)


_UNIT_CLAIM_INSTRUCTIONS = """Với mỗi chỉ tiêu tiền tệ (metric_value_kind/answer_type/value_kind là \
`money`), PHẢI khai `unit` (hoặc `selector_unit`/`answer_unit` ở P3): \
{"kind": "vnd|thousand_vnd|million_vnd|billion_vnd|percentage|number|unknown", \
"evidence": "...", "source": "csv_header|unit_snippet|cell|none"}.
- `evidence` PHẢI là chuỗi NGUYÊN VĂN lấy từ đúng "CSV header" hoặc đúng 1 dòng trong "Đơn vị tính" \
đã cho — không tự diễn giải, không lấy từ đoạn văn khác.
- `source="csv_header"` nếu evidence lấy từ dòng CSV header; `source="unit_snippet"` nếu evidence \
lấy từ 1 trong các dòng "Đơn vị tính" đã cho.
- Nếu là tiền mà KHÔNG xác định được scale (không thấy `triệu/tỷ/nghìn/đồng/VND` rõ ràng ở header \
hoặc dòng đơn vị), trả `kind="unknown"` — hệ thống sẽ tự loại, KHÔNG được đoán bừa `kind="vnd"`.
- Chỉ tiêu phần trăm dùng `kind="percentage"`; số đếm/tỷ số không mang tiền dùng `kind="number"`."""


def _full_block(table: CandidateTable) -> str:
    unit_snippets = "\n".join(table.unit_snippets) if table.unit_snippets else "(không có)"
    return f"""### Bảng: {table.table_ref}
Công ty: {table.company_name} ({table.ticker}); năm: {table.year}; tài liệu: {table.doc_name}
CSV header: {table.csv_header}
CSV gốc:
{table.csv_text}
Đơn vị tính (unit_snippet ứng viên, mỗi dòng 1 candidate cho evidence):
{unit_snippets}"""


def build_hard_mapping_prompt(
    candidates: list[CandidateTable], draft_summary: str, metric_role: str
) -> tuple[str, str]:
    system = f"""Với MỖI bảng, xác định bảng có chứa chỉ tiêu đã chọn hay không, và nếu có thì \
`row_label`/`column_label` PHẢI là chuỗi xuất hiện NGUYÊN VĂN trong CSV (đúng tên dòng ở cột đầu \
tiên, đúng tên cột ở header) — hệ thống sẽ tự định vị ô bằng chuỗi này, không tự viết code.

{_UNIT_CLAIM_INSTRUCTIONS}

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{{"mappings": [{{"table_ref": "...", "has_concept": true, "row_label": "...", "column_label": "...", \
"measurement_basis": "gross|net|not_applicable|unknown", \
"unit": {{"kind": "...", "evidence": "", "source": "none"}}}}]}}"""
    blocks = "\n\n".join(_full_block(table) for table in candidates)
    user = f"""Chỉ tiêu cần định vị (metric_role={metric_role}): {draft_summary}

{blocks}"""
    return system, user


def build_hard_finance_judge_prompt(
    plan: HardPlan,
    bindings: list[MetricBinding],
    *,
    pandas_query: str,
    actual_result: object,
) -> tuple[str, str]:
    system = """Bạn là kiểm soát viên chất lượng dữ liệu tài chính cho câu hỏi Hard dạng \
lọc-rồi-tổng hợp. Hãy đánh giá kế hoạch + query bên dưới. Trả `valid=false` nếu:
- Chỉ tiêu, phạm vi báo cáo, kỳ hoặc đơn vị không nhất quán giữa các công ty.
- `final_operation=sum` nhưng việc cộng chỉ tiêu giữa các công ty đã lọc không có ý nghĩa phân \
tích tài chính thật (vd cộng dồn các công ty khác quy mô chỉ để ra 1 con số không ai dùng).
- Ngưỡng lọc (threshold) không tự nhiên hoặc không nằm trong khoảng giá trị hợp lý của chỉ tiêu.
- `measurement_basis` chưa xác định hoặc trộn gross/net.
- Query không khớp đúng kế hoạch (đọc sai bảng, sai công thức, tự chèn hệ số hiệu chỉnh).

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"valid": true, "reason": ""}"""
    bindings_block = "\n".join(
        f"- {b.ticker}: bảng {b.table_ref}, dòng {b.row_label!r}, cột {b.column_label!r}, "
        f"basis {b.measurement_basis}, đơn vị gốc {b.raw_unit or 'không rõ'} (x{b.scale:g})"
        for b in bindings
    )
    draft = plan.draft
    user = f"""KẾ HOẠCH
Khái niệm: {draft.concept_name}
Công thức: {draft.concept_formula}
final_operation: {draft.final_operation}; comparison: {draft.comparison}; threshold: {plan.threshold}
Đơn vị: {draft.unit}; answer_type: {draft.answer_type}
Population: {draft.population}
Measurement basis: {draft.measurement_basis}
Ý nghĩa tài chính: {draft.financial_rationale}
Các bước: {plan.calculation_steps}

BINDINGS
{bindings_block}

PANDAS QUERY ĐÃ CHẠY TRÊN CSV GỐC
{pandas_query}

RESULT THỰC THI
{actual_result!r}"""
    return system, user


def build_hard_alignment_judge_prompt(
    plan: HardPlan, question: str, table_identities: str
) -> tuple[str, str]:
    system = """Bạn là kiểm soát viên contract cho câu hỏi Hard dạng lọc-rồi-tổng hợp. Kiểm tra \
câu hỏi có khớp CHÍNH XÁC kế hoạch đã khoá hay không.

Trả `valid=false` nếu câu hỏi đổi final_operation (hỏi tổng nhưng kế hoạch là đếm, hỏi Có/Không \
nhưng kế hoạch là đếm/tổng...), đổi chiều so sánh (vượt/dưới), đổi chỉ tiêu/đơn vị/gross-net, \
thiếu hoặc thêm công ty ngoài nhóm đã xét, đổi kỳ/phạm vi báo cáo, hoặc thêm yêu cầu không có \
trong kế hoạch. Không sửa câu hỏi.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"valid": true, "reason": ""}"""
    draft = plan.draft
    user = f"""KẾ HOẠCH ĐÃ KHOÁ
Khái niệm: {draft.concept_name}
Công thức: {draft.concept_formula}
final_operation: {draft.final_operation}; comparison: {draft.comparison}; threshold: {plan.threshold}
Đơn vị: {draft.unit}; answer_type: {draft.answer_type}
Population: {draft.population}
Measurement basis: {draft.measurement_basis}

DANH TÍNH VÀ PHẠM VI CÔNG TY ĐƯỢC DÙNG
{table_identities}

CÂU HỎI CẦN KIỂM TRA
{question}"""
    return system, user


def build_hard_question_prompt(plan: HardPlan, tables: list[CandidateTable]) -> tuple[str, str]:
    system = """Hãy viết một câu hỏi tài chính tự nhiên cho kế hoạch Hard đã khoá (lọc theo 1 \
ngưỡng rồi đếm/tổng hợp/kiểm tra tồn tại). Câu hỏi phải nêu đủ: nhóm công ty/ngành đang xét, kỳ, \
chỉ tiêu, và ngưỡng so sánh — nhưng không lộ tên dòng/cột bảng gốc hay cách tra bảng.

Hợp nhất là mặc định, không nhắc `báo cáo hợp nhất`; nếu là công ty mẹ, gắn cụm này trực tiếp với \
tên/nhóm doanh nghiệp. Đi thẳng vào chỉ tiêu, không dùng `trong/theo báo cáo`, `trong/theo BCTC`, \
không nhắc nguồn tài liệu/file/bảng. Không dùng `tại ngày`, `tại thời điểm`, `tại cuối năm`, `thay \
đổi như thế nào` hoặc `diễn biến ra sao`. Dùng đơn vị tiếng Việt, không viết `VND`. Nếu \
`final_operation=count`, câu hỏi phải hỏi SỐ LƯỢNG công ty; nếu `sum`, hỏi TỔNG chỉ tiêu của các \
công ty thoả điều kiện; nếu `boolean`, hỏi dạng Có/Không có công ty nào thoả điều kiện.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"question": "..."}"""
    draft = plan.draft
    identities = "\n".join(f"- {t.company_name} ({t.ticker}), năm {t.year}" for t in tables)
    comparison_vn = {"gt": "vượt", "gte": "từ", "lt": "dưới", "lte": "tối đa"}[draft.comparison]
    user = f"""Khái niệm: {draft.concept_name}
Chỉ tiêu: {draft.concept_formula}
final_operation: {draft.final_operation}
Ngưỡng ({comparison_vn}): {plan.threshold} {draft.unit}
Population: {draft.population}
Measurement basis: {draft.measurement_basis}

Các công ty trong nhóm xét:
{identities}"""
    return system, user


# Keep period handling explicit and deterministic.
#
# Keep period handling explicit and deterministic.
# Keep period handling explicit and deterministic.


def _descriptor_block(descriptor: TableDescriptor) -> str:
    return f"""### Bảng: {descriptor.table_ref}
Công ty: {descriptor.company_name} (mã {descriptor.ticker}); kỳ: {descriptor.period}; phạm vi: {descriptor.report_scope}
Nhãn bảng: {descriptor.table_labels}
Ngữ cảnh quanh bảng: {descriptor.anchor_context}"""


def build_hard_p3_plan_prompt(inventory: list[TableDescriptor], *, min_periods: int) -> tuple[str, str]:
    system = """Bạn là chuyên gia phân tích tài chính. Các bảng dưới đây là TOÀN BỘ bảng hợp lệ \
của 1 công ty tại 1 kỳ đại diện (anchor period) trong 1 dải nhiều kỳ liên tiếp — CHỈ thấy tên \
bảng/tên dòng/tên cột (KHÔNG thấy số liệu thật). Hãy chọn HAI chỉ tiêu tài chính (metric), MỖI \
chỉ tiêu PHẢI neo (`selector_anchor_ref`/`answer_anchor_ref`) vào ĐÚNG 1 bảng có thật trong danh \
sách dưới đây:
- `selector_metric_role` (neo vào `selector_anchor_ref`): chỉ tiêu dùng để CHỌN RA đúng 1 kỳ \
trong dải kỳ (kỳ có giá trị cao nhất hoặc thấp nhất của chỉ tiêu này).
- `answer_metric_role` (neo vào `answer_anchor_ref`): chỉ tiêu THỰC SỰ được hỏi, lấy ĐÚNG TẠI kỳ \
đã chọn ở trên — KHÔNG được lấy tại một kỳ cố định khác.

Cả hai chỉ tiêu phải là thuật ngữ tài chính CHUẨN, tra cứu được rộng rãi (không tự bịa tên nghe \
"có vẻ đúng"), cùng bản chất/đơn vị xuyên suốt __MIN_PERIODS__ kỳ, và PHẢI khác nhau. Hai bảng neo \
CÓ THỂ trùng nhau (2 chỉ tiêu cùng nằm 1 bảng) — không sao. Nhưng KHÔNG được chọn cặp chỉ vì 2 \
dòng tình cờ nằm cùng bảng: `answer_metric_role` KHÔNG được là alias/diễn giải lại của \
`selector_metric_role`, KHÔNG được là subtotal/component TRỰC TIẾP của selector (vd tổng và 1 \
thành phần của chính tổng đó), KHÔNG được là 1 accounting identity trực tiếp của selector. Cặp \
phải có Ý NGHĨA PHÂN TÍCH THẬT: sau khi chọn kỳ theo selector, việc hỏi answer tại đúng kỳ đó phải \
trả lời được 1 câu hỏi phân tích tài chính thật mà nhà phân tích thực sự quan tâm.

- `analysis_intent`: 1 câu mô tả ý nghĩa phân tích của việc "chọn kỳ theo selector rồi xem answer \
tại kỳ đó" (vd "xem chất lượng sinh lời trong năm có lợi nhuận tuyệt đối cao nhất").
- `relationship_rationale`: vì sao 2 chỉ tiêu này liên quan nhau trong phân tích tài chính thật — \
KHÔNG chỉ vì chúng cùng nằm 1 bảng.
- `selector_operation`: `argmax` (kỳ selector CAO NHẤT) hoặc `argmin` (kỳ THẤP NHẤT).
- `selector_value_kind`/`answer_type`: `money|percentage|number` — 2 chỉ tiêu ĐƯỢC PHÉP khác loại \
đơn vị (vd selector là tiền, answer là %).
- Không chọn chỉ tiêu cần phân biệt gross/net mà ngữ cảnh chưa rõ.
- Nếu không có cặp nào đạt các yêu cầu trên trong TOÀN BỘ danh sách, trả `feasible=false`.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"feasible": true, \
"selector_anchor_ref": "...", \
"selector_concept_name": "...", "selector_metric_role": "...", "selector_concept_formula": "...", \
"selector_financial_rationale": "...", "selector_unit": "...", "selector_value_kind": "money|percentage|number", \
"selector_measurement_basis": "gross|net|not_applicable|unknown", "selector_operation": "argmax|argmin", \
"answer_anchor_ref": "...", \
"answer_concept_name": "...", "answer_metric_role": "...", "answer_concept_formula": "...", \
"answer_financial_rationale": "...", "answer_unit": "...", \
"answer_measurement_basis": "gross|net|not_applicable|unknown", \
"answer_type": "money|percentage|number", "population": "...", \
"analysis_intent": "...", "relationship_rationale": "...", "reason": ""}"""
    system = system.replace("__MIN_PERIODS__", str(min_periods))
    return system, "\n\n".join(_descriptor_block(d) for d in inventory)


def build_hard_p3_pair_judge_prompt(
    draft: HardMultiMetricPlanDraft,
    selector_descriptor: TableDescriptor,
    answer_descriptor: TableDescriptor,
) -> tuple[str, str]:
    system = """Bạn là kiểm soát viên chất lượng cặp chỉ tiêu cho câu hỏi Hard multi-hop chọn khoá. \
Đánh giá CẶP (selector, answer) dưới đây — CHƯA thấy số liệu thật, chỉ thấy khái niệm + bảng neo. \
Trả `valid=false` nếu:
- `answer_metric_role` là alias/diễn giải lại của `selector_metric_role` (cùng 1 chỉ tiêu, khác tên).
- `answer_metric_role` là subtotal/component TRỰC TIẾP của selector (hoặc ngược lại) — vd tổng và \
một thành phần của chính tổng đó, nguyên giá và giá trị còn lại của cùng 1 tài sản nếu rationale \
chỉ đơn thuần là decomposition chứ không có ý nghĩa phân tích riêng.
- Đây là 1 accounting identity trực tiếp (answer suy ra thẳng từ selector bằng 1 phép cộng/trừ \
đơn giản trong cùng báo cáo, không cần phân tích gì thêm).
- Hai chỉ tiêu không có quan hệ phân tích tài chính thật nào (rationale gượng ép, chỉ đúng vì 2 \
dòng tình cờ nằm cùng bảng).
- Tên chỉ tiêu không phải thuật ngữ tài chính chuẩn, tra cứu được (tự bịa/diễn giải sáng tạo).
- Câu hỏi kết quả sẽ chỉ Hard về mặt topology (chọn-kỳ-rồi-tra-cứu) nhưng vô nghĩa với nhà phân \
tích thật.

KHÔNG dùng danh sách từ khoá cứng — tự đánh giá theo bản chất tài chính. Không sửa kế hoạch.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"valid": true, "reason": ""}"""
    user = f"""SELECTOR: {draft.selector_concept_name} ({draft.selector_concept_formula})
Bảng neo selector: {_descriptor_block(selector_descriptor)}

ANSWER: {draft.answer_concept_name} ({draft.answer_concept_formula})
Bảng neo answer: {_descriptor_block(answer_descriptor)}

analysis_intent: {draft.analysis_intent}
relationship_rationale: {draft.relationship_rationale}"""
    return system, user


def _p3_full_block(table: CandidateTable) -> str:
    unit_snippets = "\n".join(table.unit_snippets) if table.unit_snippets else "(không có)"
    return f"""### Bảng: {table.table_ref}
Công ty: {table.company_name} ({table.ticker}); kỳ: {table.year}; tài liệu: {table.doc_name}
CSV header: {table.csv_header}
CSV gốc:
{table.csv_text}
Đơn vị tính (unit_snippet ứng viên, mỗi dòng 1 candidate cho evidence):
{unit_snippets}"""


def build_hard_p3_mapping_prompt(
    selector_by_period: dict[str, list[CandidateTable]],
    answer_by_period: dict[str, list[CandidateTable]],
    draft: HardMultiMetricPlanDraft,
) -> tuple[str, str]:
    system = f"""Với MỖI bảng, xác định bảng có chứa selector_metric_role và/hoặc answer_metric_role \
hay không — một bảng có thể chứa CẢ HAI, chỉ MỘT, hoặc KHÔNG chỉ tiêu nào. Nếu có, `row_label`/ \
`column_label` tương ứng PHẢI là chuỗi xuất hiện NGUYÊN VĂN trong CSV (đúng tên dòng ở cột đầu \
tiên, đúng tên cột ở header) — hệ thống sẽ tự định vị ô bằng chuỗi này, không tự viết code.

{_UNIT_CLAIM_INSTRUCTIONS}

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{{"mappings": [{{"table_ref": "...", \
"has_selector": true, "selector_row_label": "...", "selector_column_label": "...", \
"selector_measurement_basis": "gross|net|not_applicable|unknown", \
"selector_unit": {{"kind": "...", "evidence": "", "source": "none"}}, \
"has_answer": true, "answer_row_label": "...", "answer_column_label": "...", \
"answer_measurement_basis": "gross|net|not_applicable|unknown", \
"answer_unit": {{"kind": "...", "evidence": "", "source": "none"}}}}]}}"""

    def _section(title: str, by_period: dict[str, list[CandidateTable]]) -> str:
        parts = [title]
        for period in sorted(by_period, key=int):
            parts.append(f"-- Kỳ {period} --")
            parts.extend(_p3_full_block(c) for c in by_period[period])
        return "\n\n".join(parts)

    user = f"""selector_metric_role={draft.selector_metric_role}: {draft.selector_concept_formula}
answer_metric_role={draft.answer_metric_role}: {draft.answer_concept_formula}

{_section("SELECTOR CANDIDATES BY PERIOD", selector_by_period)}

{_section("ANSWER CANDIDATES BY PERIOD", answer_by_period)}"""
    return system, user


def build_hard_p3_finance_judge_prompt(
    plan: HardP3Plan,
    selector_bindings: list[MetricBinding],
    answer_bindings: list[MetricBinding],
    *,
    pandas_query: str,
    actual_result: object,
    selected_period: str,
) -> tuple[str, str]:
    system = """Bạn là kiểm soát viên chất lượng dữ liệu tài chính cho câu hỏi Hard dạng multi-hop \
chọn khoá (chọn kỳ theo 1 chỉ tiêu, rồi lấy chỉ tiêu khác TẠI đúng kỳ đó). Hãy đánh giá kế hoạch + \
query bên dưới. Trả `valid=false` nếu:
- selector_metric_role và answer_metric_role không phải hai chỉ tiêu tài chính thật, khác nhau.
- Đơn vị, phạm vi báo cáo hoặc measurement_basis không nhất quán qua các kỳ của CÙNG một chỉ tiêu.
- Việc "chọn kỳ theo selector rồi lấy answer tại kỳ đó" không có ý nghĩa phân tích tài chính thật \
(vd hai chỉ tiêu không liên quan gì đến nhau trong phân tích thực tế).
- Query không khớp đúng kế hoạch (đọc sai bảng, sai kỳ, tự chèn hệ số hiệu chỉnh).

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"valid": true, "reason": ""}"""
    draft = plan.draft
    selector_block = "\n".join(
        f"- kỳ {b.period}: bảng {b.table_ref}, dòng {b.row_label!r}, cột {b.column_label!r}, "
        f"basis {b.measurement_basis}, đơn vị gốc {b.raw_unit or 'không rõ'} (x{b.scale:g})"
        for b in selector_bindings
    )
    answer_block = "\n".join(
        f"- kỳ {b.period}: bảng {b.table_ref}, dòng {b.row_label!r}, cột {b.column_label!r}, "
        f"basis {b.measurement_basis}, đơn vị gốc {b.raw_unit or 'không rõ'} (x{b.scale:g})"
        for b in answer_bindings
    )
    user = f"""KẾ HOẠCH
Selector: {draft.selector_concept_name} ({draft.selector_concept_formula}); operation: {draft.selector_operation}
Answer: {draft.answer_concept_name} ({draft.answer_concept_formula})
Đơn vị selector: {draft.selector_unit}; đơn vị answer: {draft.answer_unit}; answer_type: {draft.answer_type}
Population: {draft.population}
Ý nghĩa tài chính selector: {draft.selector_financial_rationale}
Ý nghĩa tài chính answer: {draft.answer_financial_rationale}
Các bước: {plan.calculation_steps}

BINDINGS SELECTOR (theo kỳ)
{selector_block}

BINDINGS ANSWER (theo kỳ)
{answer_block}

KỲ ĐÃ CHỌN (bởi {draft.selector_operation}): {selected_period}

PANDAS QUERY ĐÃ CHẠY TRÊN CSV GỐC
{pandas_query}

RESULT THỰC THI
{actual_result!r}"""
    return system, user


def build_hard_p3_question_prompt(
    plan: HardP3Plan,
    *,
    company_name: str,
    ticker: str,
    periods: list[str],
    report_scope: str,
    canonical_answer_unit: str,
) -> tuple[str, str]:
    system = """Hãy viết một câu hỏi tài chính tự nhiên cho kế hoạch Hard multi-hop chọn khoá đã \
khoá: hỏi giá trị của MỘT chỉ tiêu (answer) TẠI kỳ mà MỘT chỉ tiêu khác (selector) đạt cao nhất/thấp \
nhất, trong 1 dải kỳ đã cho. Câu hỏi KHÔNG được tự nêu rõ kỳ cụ thể (để hệ thống tự xác định), chỉ \
nêu dải kỳ đang xét và công ty. Không lộ tên dòng/cột bảng gốc hay cách tra bảng.

Hợp nhất là mặc định, không nhắc `báo cáo hợp nhất`; nếu là công ty mẹ (report_scope=parent), PHẢI \
gắn rõ "công ty mẹ" trực tiếp với tên doanh nghiệp. Đi thẳng vào chỉ tiêu, không dùng `trong/theo \
báo cáo`, `trong/theo BCTC`, không nhắc nguồn tài liệu/file/bảng. Không dùng `tại ngày`, `tại thời \
điểm`, `tại cuối năm`, `thay đổi như thế nào` hoặc `diễn biến ra sao`. Dùng đơn vị tiếng Việt \
(`canonical_answer_unit` đã cho), không viết `VND`.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"question": "..."}"""
    draft = plan.draft
    direction_vn = "cao nhất" if draft.selector_operation == "argmax" else "thấp nhất"
    scope_note = "công ty mẹ" if report_scope == "parent" else "hợp nhất (mặc định, không cần nhắc)"
    user = f"""Công ty: {company_name} ({ticker}); phạm vi báo cáo: {scope_note}
Dải kỳ đang xét: {", ".join(periods)}
Selector: {draft.selector_concept_name} ({draft.selector_concept_formula}) — chọn kỳ {direction_vn}
Answer: {draft.answer_concept_name} ({draft.answer_concept_formula})
canonical_answer_unit: {canonical_answer_unit}"""
    return system, user


def build_hard_p3_question_judge_prompt(
    plan: HardP3Plan,
    question: str,
    *,
    company_name: str,
    ticker: str,
    periods: list[str],
    report_scope: str,
) -> tuple[str, str]:
    system = """Bạn là kiểm soát viên cuối cho câu hỏi Hard multi-hop chọn khoá — kiểm tra ĐỒNG \
THỜI alignment với kế hoạch đã khoá VÀ chất lượng tự nhiên của câu hỏi. Trả `valid=false` nếu bất \
kỳ điều nào sau đúng:
- Đổi selector_metric_role hoặc answer_metric_role, đổi hướng chọn kỳ (cao nhất/thấp nhất).
- TỰ NÊU RÕ kỳ cụ thể đã được chọn (câu hỏi phải để người trả lời TỰ XÁC ĐỊNH kỳ).
- Đổi công ty, đổi phạm vi báo cáo (thiếu "công ty mẹ" khi report_scope=parent), đổi dải kỳ.
- Thêm yêu cầu không có trong kế hoạch, hoặc lộ tên dòng/cột/bảng gốc/cách tra bảng.
- Dùng đơn vị sai (vd viết `VND` thay vì tiếng Việt).
- Câu hỏi không tự nhiên: nhại nguyên văn tên dòng/cột, dùng thuật ngữ không chuẩn, hoặc dùng cụm \
bị cấm (`tại ngày`, `tại thời điểm`, `tại cuối năm`, `thay đổi như thế nào`, `diễn biến ra sao`).

Không sửa câu hỏi.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"valid": true, "reason": ""}"""
    draft = plan.draft
    scope_note = "công ty mẹ" if report_scope == "parent" else "hợp nhất"
    user = f"""KẾ HOẠCH ĐÃ KHOÁ
Selector: {draft.selector_concept_name} ({draft.selector_concept_formula}); operation: {draft.selector_operation}
Answer: {draft.answer_concept_name} ({draft.answer_concept_formula})
Population: {draft.population}

DANH TÍNH: {company_name} ({ticker}), phạm vi {scope_note}, dải kỳ {", ".join(periods)}

CÂU HỎI CẦN KIỂM TRA
{question}"""
    return system, user
