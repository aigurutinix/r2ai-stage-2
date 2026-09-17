
from __future__ import annotations

import json

from vifinqa.generation.common import CandidateTable
from vifinqa.generation.hard.depth3_schemas import PeriodFilterSelectLookupDraft, ResolvedMetricExpression
from vifinqa.generation.retrieval.base import TableDescriptor
from vifinqa.generation.retrieval.roles import RoleCandidates

_UNIT_INSTRUCTIONS = """Unit claim cho mỗi role có mặt:
- kind: vnd|thousand_vnd|million_vnd|billion_vnd|percentage|number|unknown.
- money phải trích evidence nguyên văn từ csv_header hoặc unit_snippet và khai đúng source.
- percentage dùng kind=percentage; number dùng kind=number.
"""

def _descriptor_block(descriptor: TableDescriptor) -> str:
    labels = descriptor.table_labels[:1200]
    context = " ".join(descriptor.anchor_context.split())[:500]
    return f"### {descriptor.table_ref}\n{labels}\nNgữ cảnh: {context}"


def build_depth3_plan_prompt(
    inventory: list[TableDescriptor], *, periods: tuple[str, ...]
) -> tuple[str, str]:
    system = f"""Bạn lập kế hoạch cho đúng recipe Hard cố định trên một công ty qua các năm:

filter metric A theo threshold -> tập năm hợp lệ
-> argmax/argmin metric B chỉ trong tập đó -> năm được chọn
-> lookup metric C tại năm được chọn -> một scalar.

Inventory chỉ cung cấp danh mục metric khả dụng để neo role vào table_ref; tên bảng/dòng không
phải evidence cho một luận điểm phân tích. Luận điểm phải đến từ kiến thức tài chính của bạn.
Thực hiện theo thứ tự: hình thành analysis_intent trước, rồi mới chọn bộ A/B/C phục vụ đúng intent
đó. Không chọn ba metric trước rồi viết relationship_rationale để hợp thức hóa hậu nghiệm.

Một plan chỉ feasible khi đồng thời thỏa:
- A tạo một điều kiện/regime có ý nghĩa, làm thay đổi cách diễn giải B và C trong tập kỳ còn lại.
- B là tiêu chí hợp lý để chọn kỳ quan sát C và có liên hệ tài chính trực tiếp, có thể giải thích
  độc lập với C; C là outcome hợp lý để quan sát tại kỳ do B chọn.
- Toàn bộ A -> B -> C tạo thành một góc phân tích tài chính có chủ đích, không chỉ là chuỗi thao
  tác lọc/chọn/lookup đúng kỹ thuật.

Trước khi trả feasible=true, tự làm substitution test: nếu có thể thay B hoặc C bằng một metric
bất kỳ khác trong inventory mà relationship_rationale vẫn nghe xuôi, plan không có dependency
ngữ nghĩa thật và phải trả feasible=false. Cũng trả infeasible nếu lý do duy nhất là ba metric
"cùng có trong báo cáo", hoặc câu hỏi dự kiến chỉ là trò tìm ô nhiều bước. Không được lấy cách
diễn đạt trôi chảy làm bằng chứng cho chất lượng quan hệ.

Chọn BA chỉ tiêu tài chính chuẩn, khác nhau và neo mỗi role vào một table_ref có thật trong
inventory. Filter phải có khả năng tạo proper subset; ưu tiên predicate tự nhiên như dương/âm
hoặc vượt một threshold chuẩn, không chọn threshold tùy tiện để ép dữ liệu.
Với measurement_basis: dùng not_applicable cho một line item độc lập như doanh thu, lợi nhuận,
chi phí hay số dư. Chỉ dùng gross/net khi chính concept và nhãn bảng thật sự phân biệt cách đo
gộp/ròng đó; không suy diễn net chỉ vì line item nằm sau một khoản khấu trừ hoặc dự phòng.
Inventory bên dưới CHỈ thuộc kỳ anchor, không phải coverage của toàn giai đoạn. Chỉ dùng nó để
chọn/neo ba metric role; hệ thống sẽ tự retrieval và kiểm tra coverage ở các kỳ còn lại sau bước
plan. KHÔNG trả infeasible chỉ vì inventory không chứa table_ref của mọi kỳ.
Recipe sẽ áp dụng xuyên suốt {len(periods)} kỳ: {', '.join(periods)}. Không phát minh topology.

Chỉ trả JSON hợp lệ theo shape:
{{"feasible":true,"reason":"",
"filter_role":{{"anchor_ref":"...","concept_name":"...","metric_role":"...","concept_formula":"...","financial_rationale":"...","unit":"...","value_kind":"money|percentage|number","measurement_basis":"gross|net|not_applicable|unknown"}},
"filter_comparison":"gt|gte|lt|lte","filter_threshold":0,
"selector_role":{{... cùng shape ...}},"selector_operation":"argmax|argmin",
"answer_role":{{... cùng shape ...}},"answer_type":"money|percentage|number|year|boolean",
"population":"...","analysis_intent":"...","relationship_rationale":"..."}}
"""
    return system, "\n\n".join(_descriptor_block(descriptor) for descriptor in inventory)


def _candidate_block(candidate: CandidateTable) -> str:
    units = "\n".join(candidate.unit_snippets) if candidate.unit_snippets else "(không có)"
    return (
        f"### Bảng: {candidate.table_ref}\nKỳ: {candidate.year}\n"
        f"CSV header: {candidate.csv_header}\nCSV gốc:\n{candidate.csv_text}\n"
        f"Unit snippets:\n{units}"
    )


def build_depth3_mapping_prompt(
    role_candidates: RoleCandidates, plan: PeriodFilterSelectLookupDraft
) -> tuple[str, str]:
    system = f"""Map BA metric_role của recipe thành đúng MỘT expression cho mỗi role × period.
Retrieval đã đưa shortlist nhỏ; tự chọn bảng/cell đúng trong shortlist, không mặc định candidate
đứng trước là đúng. row_label và column_label phải COPY NGUYÊN VĂN từ CSV, bao gồm cả lỗi OCR,
dấu câu và dấu tiếng Việt bất thường. Không được sửa chính tả hoặc chuẩn hóa bất kỳ ký tự nào.
Trước khi trả JSON, kiểm tra chuỗi đã copy xuất hiện đúng từng ký tự trong CSV; nếu không thể
copy chắc chắn thì trả feasible=false.

Expression hỗ trợ:
- direct: đúng một cell chứa trực tiếp metric.
- sum: ít nhất hai cell thành phần có quan hệ cộng thật sự để suy ra metric; không cộng các dòng
  chỉ vì chúng có sẵn, không cộng subtotal với component, và không dùng sum nếu công thức tài
  chính cần trừ/chia. Mọi cell trong sum phải cùng unit và measurement_basis.

Nếu không dựng đủ expression đáng tin cậy cho toàn bộ role × period, trả feasible=false kèm reason.
Không tự tính answer, không viết Python.

Giữ measurement_basis đã khóa trong ROLE PLAN nếu bảng thực sự chứa đúng concept. Line item độc
lập dùng not_applicable; gross/net chỉ hợp lệ khi concept và nhãn bảng phân biệt rõ basis đó.
Nếu basis của cell không khớp role thì trả present=false, không tự đổi contract của role.

{_UNIT_INSTRUCTIONS}

Chỉ trả JSON hợp lệ:
{{"feasible":true,"reason":"","expressions":[{{"metric_role":"...","period":"2024",
"operation":"direct|sum","measurement_basis":"gross|net|not_applicable|unknown",
"cells":[{{"table_ref":"...","row_label":"...","column_label":"...",
"unit":{{"kind":"...","evidence":"...","source":"csv_header|unit_snippet|cell|none"}}}}]}}]}}

Phải có đúng một expression cho từng ROLE section × kỳ, không trả record trạng thái cho từng bảng.
"""
    plan_roles = {
        role.metric_role: {
            "concept": role.concept_name,
            "formula": role.concept_formula,
            "value_kind": role.value_kind,
            "unit": role.unit,
            "measurement_basis": role.measurement_basis,
        }
        for role in (plan.filter_role, plan.selector_role, plan.answer_role)
    }
    parts = ["ROLE PLAN:\n" + json.dumps(plan_roles, ensure_ascii=False)]
    for metric_role, by_period in role_candidates.items():
        parts.append(f"\n## ROLE {metric_role}")
        for period in sorted(by_period, key=int):
            parts.append(f"\n-- Kỳ {period} --")
            parts.extend(_candidate_block(candidate) for candidate in by_period[period])
    return system, "\n\n".join(parts)


def build_depth3_question_prompt(
    plan: PeriodFilterSelectLookupDraft,
    *,
    company_name: str,
    ticker: str,
    periods: tuple[str, ...],
    answer_unit: str,
) -> tuple[str, str]:
    system = """Viết đúng một câu hỏi tài chính tự nhiên bằng tiếng Việt từ analysis intent đã
được duyệt; đừng dịch cơ học graph filter -> selector -> lookup thành văn hoặc tuần tự hóa từng
node thành một chuỗi mệnh đề lồng nhau khó đọc.

Chỉ hỏi MỘT scalar terminal C; không hỏi "năm nào", vì năm được chọn chỉ là key trung gian và
không thuộc answer. Không tiết lộ năm thắng hay đáp án. Không thêm quan hệ nhân quả hoặc luận
điểm mới để làm một plan yếu có vẻ hợp lý hơn; chỉ diễn đạt trung thành quan hệ đã có. Câu hỏi
phải tự nhiên như một nhu cầu phân tích tài chính, không như hướng dẫn tìm ô nhiều bước.
Chỉ trả JSON {"question":"..."}."""
    user = f"""Công ty: {company_name} ({ticker}); giai đoạn: {periods[0]}-{periods[-1]}.
Filter: {plan.filter_role.concept_name} {plan.filter_comparison} {plan.filter_threshold}.
Selector: {plan.selector_operation} {plan.selector_role.concept_name} trong tập năm đã lọc.
Answer: {plan.answer_role.concept_name}, đơn vị canonical {answer_unit}.
Ý nghĩa: {plan.analysis_intent}"""
    return system, user


def build_depth3_final_judge_prompt(
    plan: PeriodFilterSelectLookupDraft,
    expressions: list[ResolvedMetricExpression],
    *,
    question: str,
    pandas_query: str,
    answer: object,
    periods: tuple[str, ...],
    company_name: str,
) -> tuple[str, str]:
    system = """Bạn là final judge duy nhất cho câu hỏi Hard depth 3 và phải đánh giá độc lập.
analysis_intent, financial_rationale và relationship_rationale của planner chỉ là lời khai chưa
được tin; không dùng độ trôi chảy của chúng làm bằng chứng. Tự kiểm tra từ kiến thức tài chính
xem A -> B -> C có phải một cách phân tích có chủ đích và phổ biến hay không.

Trả valid=false nếu chỉ một trong các kiểm tra quan hệ sau thất bại:
- A không tạo regime làm thay đổi cách diễn giải B/C, mà chỉ thu hẹp số năm một cách tùy tiện.
- B không phải tiêu chí hợp lý, có liên hệ trực tiếp với C để chọn kỳ quan sát C.
- Có thể tráo B hoặc C với metric bất kỳ mà rationale hầu như vẫn giữ nguyên.
- Rationale chỉ diễn giải hậu nghiệm việc ba metric cùng xuất hiện trong báo cáo.
- Câu hỏi chỉ phức tạp về thao tác nhưng không có ý nghĩa phân tích tài chính.
- Câu văn máy móc, lồng quá nhiều mệnh đề hoặc khó hiểu.

Sau kiểm tra độc lập trên, vẫn phải giữ nguyên toàn bộ gate correctness hiện có: trả valid=false
nếu ba thuật ngữ/công thức không chuẩn hoặc quan hệ gượng ép; filter không điều khiển tập năm
selector được nhìn; selector không điều khiển key lookup; query lệch bindings/plan;
scope/unit/basis không nhất quán; câu hỏi lộ năm thắng/đáp án, sai company/window, hoặc không tự
nhiên. Unit evidence là đơn vị nguồn; query được phép nhân scale để canonicalize sang
normalized_unit (ví dụ triệu đồng × 1e6 -> đồng), đây không phải unit mismatch. Câu hỏi chỉ được
hỏi scalar metric C, không hỏi thêm selected period. reason phải nêu gate đầu tiên bị fail.
Chỉ trả JSON {"valid":true,"reason":""}."""
    expression_summary = [
        {
            "role": expression.metric_role,
            "period": expression.period,
            "operation": expression.operation,
            "cells": [
                {
                    "table_ref": binding.table_ref,
                    "row_label": binding.row_label,
                    "column_label": binding.column_label,
                    "raw_unit": binding.raw_unit,
                    "scale": binding.scale,
                    "normalized_unit": binding.normalized_unit,
                    "basis": binding.measurement_basis,
                }
                for binding in expression.cells
            ],
        }
        for expression in expressions
    ]
    user = json.dumps(
        {
            "company": company_name,
            "periods": periods,
            "plan": plan.model_dump(),
            "expressions": expression_summary,
            "pandas_query": pandas_query,
            "canonical_answer": answer,
            "question": question,
        },
        ensure_ascii=False,
    )
    return system, user
