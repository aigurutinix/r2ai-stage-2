
from __future__ import annotations

from typing import Literal

from vifinqa.generation.common import CandidateTable
from vifinqa.generation.scenarios import ScenarioSpec, scenario_prompt_rules
from vifinqa.generation.schemas import ConceptSelection, TableConceptMapping

IntermediateMode = Literal["multi_company_same_year", "multi_year_same_company"]


_CONCEPT_INTRO_BY_MODE: dict[IntermediateMode, str] = {
    "multi_company_same_year": """\
Bạn được cung cấp bảng ứng viên từ NHIỀU CÔNG TY KHÁC NHAU nhưng CÙNG 1 NĂM — chỉ thấy tên cột/tên \
dòng của mỗi bảng (KHÔNG thấy số liệu thật). Nhiệm vụ: tìm ĐÚNG 1 khái niệm/tỷ số tài chính CHUẨN \
áp dụng nhất quán được cho >=3 công ty khác nhau trong số các bảng này, để thực hiện NHIỀU phép \
tính đơn giản HOẶC 1 phép tổng hợp (tổng, trung bình, giá trị min/max, đếm số công ty thoả điều \
kiện...) so sánh GIỮA CÁC CÔNG TY này. KHÔNG chọn dạng hỏi công ty nào cao/thấp nhất vì đáp án \
là tên công ty.
""",
    "multi_year_same_company": """\
Bạn được cung cấp bảng ứng viên của CÙNG 1 CÔNG TY nhưng qua NHIỀU NĂM khác nhau — chỉ thấy tên \
cột/tên dòng của mỗi bảng (KHÔNG thấy số liệu thật). Nhiệm vụ: tìm ĐÚNG 1 khái niệm/tỷ số tài \
chính CHUẨN áp dụng nhất quán được cho >=3 năm khác nhau trong số các bảng này, để thực hiện NHIỀU \
phép tính đơn giản HOẶC 1 phép tổng hợp (tổng, trung bình, min/max, năm nào cao/thấp nhất, đếm số \
năm thoả điều kiện...) QUA CÁC NĂM này.
""",
}

_CONCEPT_SUFFIX = """
# THẾ NÀO LÀ "HỢP LÝ"
Các bảng ứng viên chỉ được lọc/retrieval theo tiêu chí kỹ thuật (cùng năm khác công ty / cùng công \
ty khác năm) — KHÔNG đảm bảo mọi công ty/năm đều có dòng số liệu tương ứng đúng nghĩa với khái \
niệm bạn chọn, VÀ KHÔNG đảm bảo các công ty/năm đó THỰC SỰ đáng để so sánh với nhau. Khi chọn khái \
niệm, tính đến các ràng buộc sau (bước map từng bảng cụ thể sẽ làm ở bước sau, ở đây bạn chỉ cần \
chọn 1 khái niệm mà PHẦN LỚN candidates có khả năng đáp ứng được):
- Phải gọi tên một metric tài chính cụ thể áp dụng nhất quán cho từng đối tượng. Không dùng kế \
hoạch chung chung trên "mỗi dòng", "các khoản mục tương ứng" hoặc toàn bộ bảng; các dạng đó tạo \
nhiều kết quả không đồng nhất thay vì một scalar có nghĩa.
- Với phép đếm hoặc lấy trung bình trên các dòng/đơn vị trong bảng, mỗi bảng phải thể hiện cùng một \
population ĐẦY ĐỦ. Không dùng bảng bị chia tiếp sang trang/bảng khác, bảng trộn nhiều loại đối tượng \
(ví dụ công ty con lẫn công ty liên kết), hoặc bảng chỉ chứa một phân nhóm khác với các bảng còn lại.
- KHÔNG chọn khái niệm chỉ có ý nghĩa trong một ngành cụ thể nếu candidates có công ty khác hẳn \
ngành nghề hoặc mô hình kinh doanh, trừ khi khái niệm đó thực sự có ý nghĩa xuyên ngành.
- KHÔNG chọn phép so sánh/cộng dồn số liệu TUYỆT ĐỐI (VND) nếu các công ty/năm chênh lệch quy mô \
lớn và kết quả chỉ phản ánh "cái nào to hơn" chứ không có ý nghĩa phân tích gì khác — ưu tiên khái \
niệm dạng TỶ LỆ/TỶ TRỌNG (%) thay vì hiệu số/tổng tuyệt đối trong trường hợp này.
- BẮT BUỘC quy về ĐÚNG 1 GIÁ TRỊ DUY NHẤT (số, %, năm, tên 1 đối tượng, hoặc Có/Không) — TUYỆT ĐỐI \
KHÔNG chọn khái niệm chỉ trả lời được bằng cách MÔ TẢ DIỄN BIẾN qua nhiều mốc/nhiều đối tượng; \
loại này không quy về được 1 con số nên KHÔNG dùng được. Nếu muốn hỏi về sự thay đổi qua nhiều \
mốc, PHẢI thu hẹp lại thành 1 trong các dạng \
sau: (a) chênh lệch/tỷ lệ tăng-giảm giữa ĐÚNG 2 mốc cụ thể; (b) năm nào đạt cao nhất/thấp nhất; \
(c) tổng/trung bình/min/max qua các mốc; (d) đếm số mốc/đối tượng thoả 1 điều kiện. Với so sánh \
nhiều công ty, KHÔNG được hỏi công ty nào cao/thấp nhất vì Intermediate không nhận tên công ty làm đáp án.
Nếu không tìm được khái niệm nào thoả các điều kiện trên cho đủ >=3 đối tượng, TUYỆT ĐỐI KHÔNG bịa \
— trả `feasible=false` kèm lý do ngắn gọn.

- Kế hoạch phải khai rõ `operation`, `answer_type`, `unit`, chủ đề bảng, population được so sánh, \
`measurement_basis` và \
lý do phép tính có ý nghĩa tài chính. `concept_formula` phải xác định duy nhất một kết quả scalar.
- `measurement_basis` là `gross` nếu dùng giá gốc/trước dự phòng, `net` nếu dùng giá trị thuần/sau \
dự phòng, `not_applicable` nếu metric không có phân biệt gross/net, và `unknown` nếu ngữ cảnh chưa \
đủ để xác định. Không chọn concept cần phân biệt gross/net khi basis còn `unknown`.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"feasible": true, "concept_name": "...", "concept_formula": "...", "operation": "difference|ratio|growth|share|sum|average|minimum|maximum|argmin|argmax|count|boolean", "answer_type": "money|percentage|number|year|boolean", "unit": "...", "table_topic": "...", "population": "...", "financial_rationale": "...", "measurement_basis": "gross|net|not_applicable|unknown", "reason": ""}
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
    mode: IntermediateMode,
    scenario: ScenarioSpec | None = None,
) -> tuple[str, str]:
    system = (
        "Bạn là chuyên gia phân tích tài chính đang xây bộ dữ liệu hỏi-đáp (QA) từ báo cáo tài "
        "chính doanh nghiệp Việt Nam đã OCR. Đây là BƯỚC 1 trong chuỗi 4 bước sinh 1 câu hỏi mức "
        "MEDIUM-KHÓ — bạn CHƯA cần chọn bảng cụ thể, CHƯA viết câu hỏi, CHƯA viết code.\n\n"
        "# BƯỚC 1: CHỌN KHÁI NIỆM\n" + _CONCEPT_INTRO_BY_MODE[mode] + _CONCEPT_SUFFIX
    )
    if scenario is not None:
        system += "\n" + scenario_prompt_rules(scenario)
    blocks = "\n\n".join(_label_block(c) for c in candidates)
    user = f"{blocks}\n\nHãy tìm 1 khái niệm/tỷ số tài chính chuẩn khả thi theo mô tả ở trên, trả về đúng JSON yêu cầu."
    return system, user



_MAPPING_SYSTEM = """\
Đây là BƯỚC 2 trong chuỗi 4 bước sinh 1 câu hỏi mức MEDIUM-KHÓ. Ở bước trước, khái niệm/tỷ số tài \
chính đã được chọn (xem trong phần input).

# BƯỚC 2: MAP DÒNG/CỘT
Nhiệm vụ: với TỪNG bảng ứng viên bên dưới (chỉ thấy tên cột/tên dòng, KHÔNG thấy số liệu thật), \
xác định bảng đó CÓ hay KHÔNG có dòng/cột tương ứng ĐÚNG NGHĨA với khái niệm đó.

# QUY TẮC
- Nếu bảng thiếu hẳn dòng/cột tương ứng, hoặc chỉ có 1 khoản mục gần giống nhưng khác phạm vi/bản \
chất, PHẢI báo `has_concept=false` — TUYỆT ĐỐI không nhận đại 1 dòng gần giống rồi hợp thức hoá. \
Loại công ty/năm đó khỏi phép so sánh còn hơn ép số liệu sai bản chất.
- LOẠI BÁO CÁO PHẢI NHẤT QUÁN: suy loại báo cáo của mỗi bảng từ tên tài liệu ("Hopnhat" => HỢP \
NHẤT, "Congtyme" => CÔNG TY MẸ). Xem toàn bộ danh sách bảng ứng viên bên dưới — nếu ĐA SỐ bảng \
thuộc 1 loại báo cáo mà một vài bảng khác lại là loại còn lại của CÙNG \
công ty, PHẢI báo `has_concept=false` cho (các) bảng lệch loại đó, dù bảng đó có dòng/cột tương \
ứng khái niệm — vì phạm vi hợp nhất khác công ty mẹ (hợp nhất gồm cả công ty con), trộn lẫn 2 loại \
khi so sánh giữa các năm/công ty sẽ cho kết quả không nhất quán, không có ý nghĩa phân tích thật.
- Nếu `has_concept=true`, điền `row_or_column_hint` mô tả ngắn gọn dòng/cột sẽ dùng (chỉ để tham \
khảo nội bộ cho bước viết code sau — KHÔNG ảnh hưởng câu hỏi).
- Với mỗi mapping, khai `measurement_basis` của đúng giá trị sẽ dùng: `gross` = giá gốc/trước dự \
phòng; `net` = giá trị thuần/sau dự phòng; `not_applicable` = metric không có phân biệt này. Nếu \
metric có thể có gross/net nhưng nhãn và ngữ cảnh không đủ xác định, đặt `measurement_basis=unknown` \
và `has_concept=false`. Không trộn gross và net giữa các bảng.
- Với kế hoạch đếm/trung bình trên các dòng hoặc đơn vị, chỉ đặt `has_concept=true` khi bảng bao phủ \
đúng cùng một population và đầy đủ. Nếu ngữ cảnh cho thấy bảng còn tiếp ở trang/bảng khác, trộn công \
ty con với công ty liên kết, hoặc chỉ là một phân nhóm không tương đương, PHẢI đặt `has_concept=false`.
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



_QUESTION_SYSTEM_TEMPLATE = """\
Đây là BƯỚC 3 trong chuỗi 4 bước sinh 1 câu hỏi mức MEDIUM-KHÓ. Bạn CHỈ được cung cấp khái niệm/tỷ \
số tài chính đã chọn + danh tính (công ty/mã/năm/tên tài liệu) của các bảng đã được chọn ở Bước 2 \
— bạn KHÔNG thấy CSV, KHÔNG thấy tên dòng/cột của các bảng này.

# BƯỚC 3: VIẾT CÂU HỎI
Nhiệm vụ: viết 1 câu hỏi tự nhiên hỏi về khái niệm đó, so sánh/tổng hợp __ENTITY_PLURAL__.

# RÀNG BUỘC CÂU HỎI
- Câu hỏi TỰ NHIÊN, rõ nghĩa, đủ ý như người thật hỏi; không cộc lốc.
- CHỈ hỏi về ĐÚNG 1 khái niệm/tỷ số đã cho — TUYỆT ĐỐI KHÔNG được tự ghép thêm bất kỳ khái niệm/tỷ \
số/phép so sánh nào khác vào câu hỏi, kể cả khi bạn nghĩ nó có vẻ liên quan đến các bảng này. Câu \
hỏi chỉ được hỏi 1 điều duy nhất, có 1 đáp án duy nhất (số, năm hoặc Có/Không), không hỏi tên công ty.
- TUYỆT ĐỐI KHÔNG hỏi kiểu "thay đổi như thế nào"/"diễn biến ra sao" — dạng này đòi hỏi mô tả nhiều \
giá trị, không quy về được 1 đáp án. Nếu concept_formula là 1 phép cực trị, chênh lệch hoặc tổng \
hợp thì hỏi thẳng kết quả scalar mà phép tính đó xác định.
- CHỈ được nêu TÊN KHÁI NIỆM/TỶ SỐ đã cho, áp dụng CHUNG một cách hiểu cho tất cả đối tượng so \
sánh — TUYỆT ĐỐI KHÔNG được nêu dòng/cột cụ thể nào sẽ dùng để tính cho từng đối tượng (bạn cũng \
không thấy thông tin đó nên không thể nêu ra).
- BẮT BUỘC TỰ ĐỦ NGHĨA — với mỗi đối tượng được nhắc, nêu rõ:
  (a) CÔNG TY (tên đầy đủ + mã trong ngoặc);
  (b) KỲ/THỜI ĐIỂM đúng theo năm của bảng (liệt kê các năm/mốc khi cần);
  (c) LOẠI BÁO CÁO — suy từ tên tài liệu: "Hopnhat" => HỢP NHẤT, "Congtyme" => CÔNG TY MẸ. QUY ƯỚC \
BẮT BUỘC: hợp nhất là MẶC ĐỊNH — TUYỆT ĐỐI KHÔNG được thêm cụm kiểu "theo báo cáo hợp nhất" khi đó \
là hợp nhất (thừa, không tự nhiên); CHỈ khi là CÔNG TY MẸ mới BẮT BUỘC nêu rõ "công ty mẹ" và gắn \
trực tiếp với tên doanh nghiệp.
- Đi thẳng vào chỉ tiêu cần hỏi. Không mở câu hoặc đóng khung nguồn bằng "trong báo cáo", "theo báo \
cáo", "trong BCTC", "theo BCTC", kể cả với tài liệu công ty mẹ; không nhắc tên tài liệu/file/bảng.
- KHÔNG dùng mẫu câu mở đầu cố định; xáo trộn linh hoạt cho tự nhiên, không lặp cấu trúc ngữ pháp.

- Không dùng các cụm "tại ngày", "tại thời điểm", "tại cuối năm". Với số dư, diễn đạt trực tiếp \
theo mốc như "cuối năm 2024" hoặc "đến ngày 31/12/2024"; với số liệu theo kỳ, dùng "năm 2024".
- Dùng đơn vị tiếng Việt trong câu hỏi; không viết `VND`.
- Nếu `measurement_basis=gross`, câu hỏi phải nói rõ giá gốc/trước dự phòng; nếu là `net`, phải \
nói rõ giá trị thuần/sau dự phòng. Với `not_applicable`, không tự thêm cụm gross/net.
- Hình thức câu hỏi phải khớp chính xác `operation`, `answer_type` và `unit` trong kế hoạch; không \
được đổi sang hỏi tên công ty.

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"question": "..."}
"""


def _identity_block(c: CandidateTable) -> str:
    return f"- Công ty: {c.company_name} (mã {c.ticker}); Tài liệu: {c.doc_name} (năm {c.year})"


def build_question_prompt(
    concept: ConceptSelection,
    selected_tables: list[CandidateTable],
    mode: IntermediateMode,
    scenario: ScenarioSpec | None = None,
) -> tuple[str, str]:
    entity_plural = "giữa các công ty này" if mode == "multi_company_same_year" else "qua các năm này"
    system = _QUESTION_SYSTEM_TEMPLATE.replace("__ENTITY_PLURAL__", entity_plural)
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
        "Hãy viết 1 câu hỏi Medium-khó theo mô tả ở trên, trả về đúng JSON yêu cầu."
    )
    return system, user



_QUERY_SYSTEM = """\
Đây là BƯỚC 4 (cuối cùng) trong chuỗi 4 bước sinh 1 câu hỏi mức MEDIUM-KHÓ — bước DUY NHẤT bạn \
thấy CSV số liệu thật.

# BƯỚC 4: VIẾT PANDAS QUERY
Nhiệm vụ: viết code `pandas_query` tính đúng khái niệm/tỷ số đã chọn trên các bảng đã cho. Hệ thống \
sẽ chạy query trên CSV gốc và lấy chính biến `result` làm answer.

# KẾ HOẠCH ĐÃ KHÓA
`concept_formula`, `operation`, `answer_type`, `unit`, `population` và `measurement_basis` xác định duy nhất thứ cần \
tính. Không tự đổi mục tiêu sang một phép tính, kiểu đáp án hay tập đối tượng khác.

# QUY TẮC CÁC TRƯỜNG
- "pandas_query": code Python ĐẦY ĐỦ, THỰC THI ĐƯỢC, gán kết quả cuối vào biến `result`. QUY TẮC:
  + Biến `dfs` đã có sẵn — là dict, key là ĐÚNG chuỗi table_ref, value là pd.read_csv của đúng \
bảng đó.
  + Các ô số trong CSV là CHUỖI định dạng Việt Nam ("1.234.567", số âm "(1.234)") — code PHẢI tự \
parse thành số trước khi tính.
  + KHÔNG tự bọc `round(...)`. Chỉ làm tròn nếu kế hoạch đã khóa yêu cầu rõ số chữ số thập phân; \
khi đó câu hỏi cũng phải nêu yêu cầu này. Câu Có/Không: gán `result = "Có"` hoặc `result = "Không"`.
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


# Keep period handling explicit and deterministic.

_SAME_DOC_MULTI_CONCEPT_INTRO = """\
Bạn được cung cấp bảng ứng viên TRONG CÙNG MỘT TÀI LIỆU (cùng công ty, cùng năm/kỳ báo cáo) — chỉ \
thấy tên cột/tên dòng của mỗi bảng (KHÔNG thấy số liệu thật). Nhiệm vụ: tìm ĐÚNG 1 khái niệm/tỷ số \
tài chính CHUẨN áp dụng nhất quán được cho >=3 bảng khác nhau trong số các bảng này, để thực hiện \
NHIỀU phép tính đơn giản HOẶC 1 phép tổng hợp (tổng, trung bình, giá trị min/max, đếm số khoản mục \
thoả điều kiện...) trên các bảng đó.
"""


def build_same_doc_multi_concept_prompt(
    candidates: list[CandidateTable], scenario: ScenarioSpec | None = None
) -> tuple[str, str]:
    system = (
        "Bạn là chuyên gia phân tích tài chính đang xây bộ dữ liệu hỏi-đáp (QA) từ báo cáo tài "
        "chính doanh nghiệp Việt Nam đã OCR. Đây là BƯỚC 1 trong chuỗi 4 bước sinh 1 câu hỏi mức "
        "MEDIUM-KHÓ — bạn CHƯA cần chọn bảng cụ thể, CHƯA viết câu hỏi, CHƯA viết code.\n\n"
        "# BƯỚC 1: CHỌN KHÁI NIỆM\n" + _SAME_DOC_MULTI_CONCEPT_INTRO + _CONCEPT_SUFFIX
    )
    if scenario is not None:
        system += "\n" + scenario_prompt_rules(scenario)
    blocks = "\n\n".join(_label_block(c) for c in candidates)
    user = (
        f"{blocks}\n\nHãy tìm 1 khái niệm/tỷ số tài chính chuẩn khả thi theo mô tả ở trên, "
        "trả về đúng JSON yêu cầu."
    )
    return system, user


def build_same_doc_multi_question_prompt(
    concept: ConceptSelection,
    selected_tables: list[CandidateTable],
    scenario: ScenarioSpec | None = None,
) -> tuple[str, str]:
    system = _QUESTION_SYSTEM_TEMPLATE.replace("__ENTITY_PLURAL__", "trong tài liệu này")
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
        "Hãy viết 1 câu hỏi Medium-khó theo mô tả ở trên, trả về đúng JSON yêu cầu."
    )
    return system, user
