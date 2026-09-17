
from __future__ import annotations

from vifinqa.generation.common import CandidateTable
from vifinqa.generation.schemas import EasyFactQuery

_FACT_SYSTEM = """\
Bạn là chuyên gia đọc báo cáo tài chính doanh nghiệp Việt Nam. Hãy chọn đúng một số liệu có sẵn \
trực tiếp trong bảng CSV để tạo fact mức Easy. Không tính toán, không cộng/trừ/chia, không so sánh.

Yêu cầu:
- Chọn khoản mục có ý nghĩa tài chính rõ ràng; dùng ngữ cảnh trang để hiểu đúng nhãn, đơn vị và kỳ.
- `metric_name` là cách gọi tự nhiên, chính xác; không chép mã thuyết minh hay ký tự OCR.
- `period_label` là kỳ của đúng ô được lấy. `time_basis=period` cho chỉ tiêu phát sinh trong kỳ; \
`point_in_time` cho số dư tại một mốc.
- `unit` phải ghi rõ đơn vị của giá trị bằng cách gọi tiếng Việt (`đồng`, `nghìn đồng`, `triệu \
đồng`, `tỷ đồng`, `%`); không dùng `VND`. `report_scope` là `consolidated` hoặc `parent` theo tài liệu.
- `pandas_query` là code Python đầy đủ, chỉ dùng `pd` và biến `df` đã có sẵn, gán kết quả scalar \
vào `result`. CSV là bản gốc; tự parse số Việt Nam và số âm trong ngoặc.
- `result` phải giữ nguyên độ chính xác tính được từ CSV. Không tự làm tròn; chỉ được làm tròn nếu \
yêu cầu đó được nêu rõ trong fact cần khóa. Hệ thống sẽ chạy query và lấy chính `result` làm answer.
- Không cộng, trừ, nhân hoặc chia thêm hằng số hiệu chỉnh ngoài fact cần khóa; mọi thành phần của \
kết quả phải xuất phát từ CSV và đúng fact.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"metric_name": "...", "period_label": "...", "time_basis": "period|point_in_time", "unit": "...", "report_scope": "consolidated|parent", "pandas_query": "..."}
"""

_QUESTION_SYSTEM = """\
Bạn là chuyên gia tài chính Việt Nam. Hãy diễn đạt fact đã khóa thành đúng một câu hỏi Easy tự \
nhiên, tự đủ nghĩa và chỉ hỏi một giá trị duy nhất.

Yêu cầu:
- Nêu tên đầy đủ doanh nghiệp và mã chứng khoán trong ngoặc, khoản mục, kỳ và đơn vị khi cần.
- Báo cáo hợp nhất là mặc định, không nhắc cụm `báo cáo hợp nhất`. Nếu scope là `parent`, phải nêu \
rõ `công ty mẹ` và gắn cụm này trực tiếp với doanh nghiệp lập báo cáo, không để nó có thể bị hiểu \
là đang bổ nghĩa cho khách hàng, bên liên quan hoặc khoản mục.
- Đi thẳng vào chỉ tiêu của doanh nghiệp. Không mở câu hoặc đóng khung nguồn bằng `trong báo cáo`, \
`theo báo cáo`, `trong BCTC`, `theo BCTC`. Khi scope là `parent`, dùng cấu trúc tự nhiên như \
`... của công ty mẹ <tên doanh nghiệp> (<mã>) ...`, không viết `trong BCTC công ty mẹ của ...`.
- Với chỉ tiêu theo kỳ, dùng cách nói theo năm/kỳ. Với số dư, dùng `cuối năm ...`, `đầu năm ...` \
hoặc `đến ngày ...` tùy `period_label`.
- Không dùng các cụm `tại ngày`, `tại thời điểm`, `tại cuối năm`.
- Dùng đơn vị tiếng Việt; không viết `VND` trong câu hỏi.
- Không nêu công thức, vị trí dòng/cột, mã thuyết minh, tên file, báo cáo nguồn hay cách tra bảng.

Chỉ trả JSON hợp lệ, không thêm text và không rào ```:
{"question": "..."}
"""


def _table_shape(csv_text: str) -> str:
    """State the columns and the row count the query has to stay inside.

    LOCAL ADDITION (not the organisers'). Measured over 27 discarded attempts,
    20 failed because the generated `pandas_query` referenced something that is
    not in the frame — `index N is out of bounds`, `KeyError: ''`,
    `KeyError: 'Unnamed: N'`, `KeyError` on a row label copied from the page text
    rather than the table. The model was shown the raw CSV and left to infer the
    frame's shape from it, which is the same positional-counting task that our own
    submission model fails at.

    Naming the columns verbatim and giving the valid `.iloc` range converts that
    inference into copying. Nothing is loosened: the execution gate downstream is
    unchanged, so a query that still misses is still discarded.
    """

    import csv
    import io

    try:
        rows = list(csv.reader(io.StringIO(csv_text)))
    except (csv.Error, ValueError):
        return ""
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    columns = ", ".join(f"[{i}] {name!r}" for i, name in enumerate(header))
    return (
        f"\nCột của `df` (đúng theo thứ tự, dùng nguyên văn):\n{columns}\n"
        f"`df` có {len(body)} dòng dữ liệu, nên .iloc hợp lệ trong khoảng "
        f"0..{max(len(body) - 1, 0)}.\n"
    )


def build_easy_fact_prompt(
    ctx: CandidateTable, *, known_scope: str = "unknown"
) -> tuple[str, str]:
    # `known_scope` is a LOCAL ADDITION. `report_scope(doc_name)` derives the
    # scope deterministically from the document name, and easy.py *rejects* a
    # fact whose `report_scope` disagrees with it — but the prompt never told the
    # model which one it was. That accounted for 7 of 27 discarded attempts:
    # the pipeline knew the answer and threw away work for not guessing it.
    scope_line = ""
    if known_scope in ("parent", "consolidated"):
        scope_line = (f"Phạm vi báo cáo của tài liệu này là `{known_scope}`. "
                      f"Dùng đúng giá trị đó cho `report_scope`.\n")

    user = f"""Công ty: {ctx.company_name} (mã {ctx.ticker})
Tài liệu: {ctx.doc_name} (năm {ctx.year})
Bảng: {ctx.table_ref}
{scope_line}
CSV gốc:
{ctx.csv_text}
{_table_shape(ctx.csv_text)}
Ngữ cảnh trang lân cận:
{ctx.surrounding_pages}

Hãy chọn và khóa một fact Easy bằng query trên CSV gốc."""
    return _FACT_SYSTEM, user


def build_easy_question_prompt(ctx: CandidateTable, fact: EasyFactQuery) -> tuple[str, str]:
    user = f"""Công ty: {ctx.company_name} (mã {ctx.ticker})
Năm tài liệu: {ctx.year}
Phạm vi báo cáo: {fact.report_scope}
Khoản mục: {fact.metric_name}
Kỳ/mốc của số liệu: {fact.period_label}
Cơ sở thời gian: {fact.time_basis}
Đơn vị: {fact.unit}

Hãy viết câu hỏi khớp chính xác fact trên."""
    return _QUESTION_SYSTEM, user
