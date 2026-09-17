
from __future__ import annotations

from typing import Literal

from vifinqa.generation.common import CandidateTable
from vifinqa.generation.scenarios import ScenarioSpec, scenario_prompt_rules
from vifinqa.generation.schemas import ConceptSelection, TableConceptMapping

MediumCase = Literal["same_doc", "same_company_diff_year", "same_year_diff_company"]

_CASE_RULES: dict[MediumCase, str] = {
    "same_doc": """Hai scalar phải đến từ hai bảng trong cùng tài liệu. Chỉ dùng phép chia/tỷ \
trọng khi đó là một tỷ số tài chính chuẩn với tử số và mẫu số cùng phạm vi; chỉ dùng phép trừ khi \
hai giá trị thật sự cùng metric và cùng cơ sở đo lường. Không lấy chênh lệch quy mô giữa hai khoản \
mục khác bản chất (ví dụ hai nhóm tài sản hoặc hai nhóm chi phí khác nhau); việc cả hai cùng có đơn \
vị tiền không làm phép trừ trở nên có ý nghĩa.""",
    "same_company_diff_year": """Hai scalar phải là CÙNG MỘT metric cụ thể của cùng doanh nghiệp, \
mỗi scalar lấy từ đúng một năm/tài liệu đã chọn. Phép tính so sánh trực tiếp hai năm đó. Không tính \
growth riêng bằng cặp cột năm nay/năm trước trong từng bảng rồi tổng hợp hai kết quả.""",
    "same_year_diff_company": """Hai scalar phải là CÙNG MỘT metric cụ thể, cùng kỳ và cùng đơn vị, \
mỗi scalar thuộc đúng một doanh nghiệp. Chỉ so sánh trực tiếp hai scalar đó. Không tính một tỷ số \
hoặc growth riêng trong từng công ty rồi tiếp tục tổng hợp. Ưu tiên metric đã chuẩn hóa theo quy mô. \
Chênh lệch tiền tuyệt đối chỉ khả thi khi giá trị tuyệt đối của chính khoản mục đó có ý nghĩa phân \
tích cụ thể không phụ thuộc chủ yếu vào quy mô doanh nghiệp; cùng ngành, cùng kỳ và cùng đơn vị chưa \
đủ để chứng minh điều này. Không lấy số tiền tuyệt đối của công ty A chia cho số tiền của công ty B \
rồi gọi kết quả là tỷ lệ/hệ số; phải so sánh một metric đã được chuẩn hóa trong từng công ty.""",
}


_CONCEPT_SYSTEM = """\
Bạn là chuyên gia phân tích tài chính đang xây bộ dữ liệu hỏi-đáp (QA) từ báo cáo tài chính \
doanh nghiệp Việt Nam đã OCR. Đây là BƯỚC 1 trong chuỗi 4 bước sinh 1 câu hỏi mức MEDIUM-DỄ — bạn \
CHƯA cần chọn bảng cụ thể, CHƯA viết câu hỏi, CHƯA viết code.

# BƯỚC 1: CHỌN KHÁI NIỆM
Bạn được cung cấp NHIỀU bảng ứng viên (có thể cùng 1 tài liệu, cùng công ty khác năm, hoặc khác \
công ty cùng năm) — CHỈ thấy tên cột/tên dòng của mỗi bảng (KHÔNG thấy số liệu thật). Nhiệm vụ: \
tìm ĐÚNG 1 khái niệm/tỷ số tài chính CHUẨN mà ÍT NHẤT 2 trong số các bảng này có thể dùng để tính \
— hiệu (chênh lệch tuyệt đối), thương/tỉ lệ, hoặc phần trăm (tăng/giảm % theo thời gian, tỉ \
trọng...). Chỉ 1 phép tính đơn giản (không gộp nhiều phép tính — đó là mức khó hơn).

Khái niệm phải chỉ rõ ĐÚNG HAI GIÁ TRỊ SCALAR đầu vào và MỘT metric/khoản mục tài chính cụ thể. \
Không chọn kế hoạch áp dụng cho "mỗi dòng", "các khoản mục tương ứng", toàn bộ bảng hoặc một danh \
sách metric vì chúng tạo nhiều đáp án. Không tính một tỷ số riêng trong từng bảng rồi tiếp tục so \
sánh hai tỷ số, vì như vậy dùng nhiều hơn hai thông tin và nhiều hơn một phép tính.

# THẾ NÀO LÀ "HỢP LÝ"
Phép tính giữa 2 số đó phải ứng với 1 KHÁI NIỆM/TỶ SỐ TÀI CHÍNH ĐƯỢC CÔNG NHẬN RỘNG RÃI. Tử số và \
mẫu số phải cùng phạm vi; tỷ trọng phải dùng đúng tổng chứa thành phần đó; tăng trưởng phải áp dụng \
cho cùng khoản mục qua các kỳ. Không ghép hai khoản mục chỉ vì tên gọi gần giống khi bản chất hoặc \
phạm vi khác nhau. Không tính lại một kết quả bằng phép trừ/chia nếu kết quả đã có sẵn thành một \
dòng riêng trong bảng. Với `operation=difference`, chỉ chấp nhận cùng một metric qua hai kỳ/hai \
doanh nghiệp, hoặc một quan hệ chênh lệch/net/subtotal có ý nghĩa tài chính được công nhận. Loại bỏ \
chênh lệch tuyệt đối giữa hai khoản mục khác bản chất chỉ vì chúng cùng đơn vị tiền.

Nếu xem qua tên dòng/cột của tất cả bảng ứng viên mà KHÔNG tìm được khái niệm nào thoả các điều \
kiện trên, TUYỆT ĐỐI KHÔNG được bịa — trả `feasible=false` kèm lý do ngắn gọn.

- Kế hoạch phải khai rõ `operation`, `answer_type`, `unit`, chủ đề bảng, population được so sánh, \
`measurement_basis` và \
lý do phép tính có ý nghĩa tài chính. `concept_formula` phải xác định duy nhất một kết quả scalar.
- `measurement_basis` là `gross` nếu dùng giá gốc/trước dự phòng, `net` nếu dùng giá trị thuần/sau \
dự phòng, `not_applicable` nếu metric không có phân biệt gross/net, và `unknown` nếu ngữ cảnh chưa \
đủ để xác định. Không chọn concept cần phân biệt gross/net khi basis còn `unknown`.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"feasible": true, "concept_name": "...", "concept_formula": "...", "operation": "difference|ratio|growth|share", "answer_type": "money|percentage|number", "unit": "...", "table_topic": "...", "population": "...", "financial_rationale": "...", "measurement_basis": "gross|net|not_applicable|unknown", "reason": ""}
"""


def _label_block(c: CandidateTable) -> str:
    return f"""### Bảng: {c.table_ref}
Công ty: {c.company_name} (mã {c.ticker})
Tài liệu: {c.doc_name} (năm {c.year})

Tên cột/tên dòng: {c.table_labels}

Ngữ cảnh trang lân cận:
{c.surrounding_pages}
"""


def build_concept_prompt(
    candidates: list[CandidateTable],
    case: MediumCase = "same_doc",
    scenario: ScenarioSpec | None = None,
) -> tuple[str, str]:
    system = _CONCEPT_SYSTEM
    if scenario is not None:
        system += "\n" + scenario_prompt_rules(scenario)
    blocks = "\n\n".join(_label_block(c) for c in candidates)
    user = (
        f"CASE ĐÃ RANDOM CHO RECORD NÀY: {case}\n"
        f"Contract riêng của case: {_CASE_RULES[case]}\n\n"
        f"{blocks}\n\n"
        "Hãy tìm 1 khái niệm/tỷ số tài chính chuẩn khả thi theo đúng case, trả về JSON yêu cầu."
    )
    return system, user



_MAPPING_SYSTEM = """\
Đây là BƯỚC 2 trong chuỗi 4 bước sinh 1 câu hỏi mức MEDIUM-DỄ. Ở bước trước, khái niệm/tỷ số tài \
chính đã được chọn (xem trong phần input).

# BƯỚC 2: MAP DÒNG/CỘT
Nhiệm vụ: với TỪNG bảng ứng viên bên dưới (chỉ thấy tên cột/tên dòng, KHÔNG thấy số liệu thật), \
xác định bảng đó CÓ hay KHÔNG có dòng/cột tương ứng ĐÚNG NGHĨA với khái niệm đó.

# QUY TẮC
- Nếu bảng thiếu hẳn dòng/cột tương ứng, hoặc chỉ có 1 khoản mục gần giống nhưng khác phạm vi/bản \
chất, PHẢI báo `has_concept=false` — TUYỆT ĐỐI không nhận đại 1 dòng gần giống rồi hợp thức hoá.
- Nếu `has_concept=true`, điền `row_or_column_hint` mô tả ngắn gọn dòng/cột sẽ dùng (chỉ để tham \
khảo nội bộ cho bước viết code sau — KHÔNG ảnh hưởng câu hỏi).
- Với mỗi mapping, khai `measurement_basis` của đúng giá trị sẽ dùng: `gross` = giá gốc/trước dự \
phòng; `net` = giá trị thuần/sau dự phòng; `not_applicable` = metric không có phân biệt này. Nếu \
metric có thể có gross/net nhưng nhãn và ngữ cảnh không đủ xác định, đặt `measurement_basis=unknown` \
và `has_concept=false`. Không trộn gross và net giữa các bảng.
- Phải trả về ĐỦ 1 phần tử `mappings` cho MỖI bảng ứng viên được cung cấp, đúng `table_ref`.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"mappings": [{"table_ref": "...", "has_concept": true, "row_or_column_hint": "...", "measurement_basis": "gross|net|not_applicable|unknown"}, ...]}
"""


def build_mapping_prompt(
    candidates: list[CandidateTable],
    concept: ConceptSelection,
    scenario: ScenarioSpec | None = None,
) -> tuple[str, str]:
    system = _MAPPING_SYSTEM
    if scenario is not None:
        system += "\n" + scenario_prompt_rules(scenario)
    blocks = "\n\n".join(_label_block(c) for c in candidates)
    user = (
        f"Khái niệm/tỷ số đã chọn ở Bước 1: {concept.concept_name} ({concept.concept_formula})\n"
        f"Measurement basis mục tiêu: {concept.measurement_basis}\n\n"
        f"{blocks}\n\nHãy map từng bảng theo mô tả ở trên, trả về đúng JSON yêu cầu."
    )
    return system, user



_QUESTION_SYSTEM = """\
Đây là BƯỚC 3 trong chuỗi 4 bước sinh 1 câu hỏi mức MEDIUM-DỄ. Bạn CHỈ được cung cấp khái niệm/tỷ \
số tài chính đã chọn + danh tính (công ty/mã/năm/tên tài liệu) của ĐÚNG 2 bảng đã được chọn ở Bước \
2 — bạn KHÔNG thấy CSV, KHÔNG thấy tên dòng/cột của 2 bảng này.

# BƯỚC 3: VIẾT CÂU HỎI
Nhiệm vụ: viết 1 câu hỏi tự nhiên hỏi về khái niệm đó áp dụng cho 2 bảng này.

# RÀNG BUỘC CÂU HỎI
- Câu hỏi phải TỰ NHIÊN, rõ nghĩa, đúng như người thật sẽ hỏi khi đọc báo cáo; không cộc lốc.
- CHỈ hỏi về ĐÚNG 1 khái niệm/tỷ số đã cho — TUYỆT ĐỐI KHÔNG được tự ghép thêm bất kỳ khái niệm/tỷ \
số/phép so sánh nào khác vào câu hỏi, kể cả khi bạn nghĩ nó có vẻ liên quan đến 2 bảng này. Câu hỏi \
chỉ được hỏi 1 điều duy nhất, có 1 đáp án số duy nhất.
- TUYỆT ĐỐI KHÔNG hỏi kiểu "thay đổi như thế nào"/"diễn biến ra sao" — dạng này đòi hỏi mô tả nhiều \
giá trị, không quy về được 1 đáp án. Hỏi thẳng kết quả scalar mà phép tính xác định.
- CHỈ được nêu TÊN KHÁI NIỆM/TỶ SỐ đã cho — TUYỆT ĐỐI KHÔNG được nêu \
dòng/cột cụ thể nào sẽ dùng để tính (bạn cũng không thấy thông tin đó nên không thể nêu ra).
- BẮT BUỘC TỰ ĐỦ NGHĨA — nêu rõ với TỪNG bảng được dùng:
  (a) CÔNG TY (tên đầy đủ + mã trong ngoặc);
  (b) KỲ/THỜI ĐIỂM đúng theo năm của bảng; câu so 2 mốc phải nêu cả 2 mốc;
  (c) LOẠI BÁO CÁO — suy từ tên tài liệu: "Hopnhat" => HỢP NHẤT, "Congtyme" => CÔNG TY MẸ. QUY ƯỚC \
BẮT BUỘC: hợp nhất là MẶC ĐỊNH — TUYỆT ĐỐI KHÔNG được thêm cụm kiểu "theo báo cáo hợp nhất" / "trong \
BCTC hợp nhất của" khi đó là hợp nhất (thừa, không tự nhiên); CHỈ khi là CÔNG TY MẸ mới BẮT BUỘC nêu \
rõ "công ty mẹ" và gắn trực tiếp với tên doanh nghiệp, chẳng hạn "của công ty mẹ <doanh nghiệp>".
- Đi thẳng vào chỉ tiêu cần hỏi. Không mở câu hoặc đóng khung nguồn bằng "trong báo cáo", "theo báo \
cáo", "trong BCTC", "theo BCTC", kể cả với tài liệu công ty mẹ; không nhắc tên tài liệu/file/bảng.
- KHÔNG dùng mẫu câu mở đầu cố định; xáo trộn linh hoạt vị trí công ty/năm/loại báo cáo cho tự \
nhiên. Không lặp lại cấu trúc ngữ pháp giữa các câu.

- Không dùng các cụm "tại ngày", "tại thời điểm", "tại cuối năm". Với số dư, diễn đạt trực tiếp \
theo mốc như "cuối năm 2024" hoặc "đến ngày 31/12/2024"; với số liệu theo kỳ, dùng "năm 2024".
- Dùng đơn vị tiếng Việt trong câu hỏi; không viết `VND`.
- Nếu `measurement_basis=gross`, câu hỏi phải nói rõ giá gốc/trước dự phòng; nếu là `net`, phải \
nói rõ giá trị thuần/sau dự phòng. Với `not_applicable`, không tự thêm cụm gross/net.
- Hình thức câu hỏi phải khớp chính xác `operation`, `answer_type` và `unit` trong kế hoạch; không \
được đổi từ hỏi giá trị sang hỏi năm/công ty hoặc ngược lại.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"question": "..."}
"""


def _identity_block(c: CandidateTable) -> str:
    return f"- Công ty: {c.company_name} (mã {c.ticker}); Tài liệu: {c.doc_name} (năm {c.year})"


def build_question_prompt(
    concept: ConceptSelection,
    selected_tables: list[CandidateTable],
    scenario: ScenarioSpec | None = None,
) -> tuple[str, str]:
    system = _QUESTION_SYSTEM
    if scenario is not None:
        system += "\n" + scenario_prompt_rules(scenario)
    identities = "\n".join(_identity_block(c) for c in selected_tables)
    user = (
        f"Kế hoạch đã khóa:\n"
        f"- Khái niệm: {concept.concept_name}\n"
        f"- Công thức: {concept.concept_formula}\n"
        f"- Phép toán: {concept.operation}\n"
        f"- Kiểu đáp án: {concept.answer_type}\n"
        f"- Đơn vị: {concept.unit}\n"
        f"- Population: {concept.population}\n"
        f"- Cơ sở đo lường: {concept.measurement_basis}\n\n"
        f"Danh tính các bảng đã chọn:\n{identities}\n\n"
        "Hãy viết 1 câu hỏi Medium-dễ theo mô tả ở trên, trả về đúng JSON yêu cầu."
    )
    return system, user



_QUERY_SYSTEM = """\
Đây là BƯỚC 4 (cuối cùng) trong chuỗi 4 bước sinh 1 câu hỏi mức MEDIUM-DỄ — bước DUY NHẤT bạn thấy \
CSV số liệu thật.

# BƯỚC 4: VIẾT PANDAS QUERY
Nhiệm vụ: viết code `pandas_query` tính đúng khái niệm/tỷ số đã chọn trên 2 bảng đã cho. Hệ thống \
sẽ chạy query trên CSV gốc và lấy chính biến `result` làm answer.

# KẾ HOẠCH ĐÃ KHÓA
`concept_formula`, `operation`, `answer_type`, `unit`, `population` và `measurement_basis` xác định duy nhất thứ cần \
tính. Không tự đổi mục tiêu sang một phép tính, kiểu đáp án hay tập đối tượng khác.

# QUY TẮC CÁC TRƯỜNG
- "pandas_query": code Python ĐẦY ĐỦ, THỰC THI ĐƯỢC, gán kết quả cuối vào biến `result`. QUY TẮC:
  + Biến `dfs` đã có sẵn — là dict, key là ĐÚNG chuỗi table_ref, value là pd.read_csv của đúng bảng đó.
  + Các ô số trong CSV là CHUỖI định dạng Việt Nam ("1.234.567", số âm "(1.234)") — code PHẢI tự \
parse thành số trước khi tính.
  + KHÔNG tự bọc `round(...)`. Chỉ làm tròn nếu kế hoạch đã khóa yêu cầu rõ số chữ số thập phân; \
khi đó câu hỏi cũng phải nêu yêu cầu này. Câu Có/Không: gán `result = "Có"` hoặc `result = "Không"` \
(KHÔNG dùng True/False).
  + KHÔNG cộng/trừ/nhân/chia thêm hằng số hiệu chỉnh ngoài kế hoạch; mọi thành phần trong kết quả \
phải xuất phát từ CSV và đúng kế hoạch đã khóa.
  + CHỈ dùng `pd` và `dfs`. KHÔNG import, KHÔNG dùng `np` hay thư viện khác.
  + Gợi ý dòng/cột từ Bước 2 chỉ mang tính tham khảo — hãy tự kiểm tra lại đúng dòng/cột thật trong \
CSV trước khi dùng, không tin tuyệt đối gợi ý nếu không khớp CSV thực tế.
  + Chỉ dùng đúng `measurement_basis` đã khóa. Không tự lấy gross thay cho net, tự trừ dự phòng để \
đổi basis, hoặc ngược lại.
Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"pandas_query": "..."}
"""


def _full_block(c: CandidateTable, hint: str) -> str:
    hint_line = f"\nGợi ý dòng/cột từ Bước 2: {hint}" if hint else ""
    return f"""### Bảng: {c.table_ref}
Công ty: {c.company_name} (mã {c.ticker})
Tài liệu: {c.doc_name} (năm {c.year}){hint_line}

Nội dung CSV:
{c.csv_text}

Ngữ cảnh trang lân cận:
{c.surrounding_pages}
"""


def build_query_prompt(
    concept: ConceptSelection,
    selected_tables: list[CandidateTable],
    mappings: list[TableConceptMapping],
    scenario: ScenarioSpec | None = None,
) -> tuple[str, str]:
    system = _QUERY_SYSTEM
    if scenario is not None:
        system += "\n" + scenario_prompt_rules(scenario)
    hint_by_ref = {m.table_ref: m.row_or_column_hint for m in mappings}
    blocks = "\n\n".join(_full_block(c, hint_by_ref.get(c.table_ref, "")) for c in selected_tables)
    user = (
        f"Kế hoạch cần tính:\n"
        f"- Khái niệm: {concept.concept_name}\n"
        f"- Công thức: {concept.concept_formula}\n"
        f"- Phép toán: {concept.operation}\n"
        f"- Kiểu đáp án: {concept.answer_type}\n"
        f"- Đơn vị: {concept.unit}\n"
        f"- Chủ đề bảng: {concept.table_topic}\n"
        f"- Population: {concept.population}\n"
        f"- Cơ sở đo lường: {concept.measurement_basis}\n"
        f"- Ý nghĩa tài chính: {concept.financial_rationale}\n\n"
        f"{blocks}\n\nHãy viết pandas_query + answer theo mô tả ở trên, trả về đúng JSON yêu cầu."
    )
    return system, user
