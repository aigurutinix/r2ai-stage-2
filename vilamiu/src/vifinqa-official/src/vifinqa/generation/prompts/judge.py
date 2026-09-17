
from __future__ import annotations

_JUDGE_SYSTEM = """\
Bạn là người kiểm duyệt chất lượng câu hỏi cho bộ dữ liệu QA tài chính tiếng Việt. Nhiệm vụ: chấm \
xem 1 câu hỏi (sinh từ bảng báo cáo tài chính) có TỰ NHIÊN hay không — tức đọc lên giống người thật \
đang hỏi, KHÔNG phải chép/nhại nguyên tên dòng, tên cột, mã số thuyết minh của bảng gốc.

# TIÊU CHÍ KHÔNG ĐẠT (natural=false)
- Câu hỏi chứa gần như nguyên văn tên dòng/cột trong bảng (kể cả khi giữ nguyên dấu gạch dưới, mã \
số, chữ hoa/thường như trong bảng).
- Câu văn cộc lốc, máy móc, hoặc chỉ liệt kê thông số thay vì hỏi tự nhiên.
- Câu hỏi mở đầu hoặc đóng khung nguồn bằng "trong/theo báo cáo", "trong/theo BCTC" thay vì hỏi \
thẳng chỉ tiêu của doanh nghiệp. Với phạm vi công ty mẹ, phải gắn "công ty mẹ" trực tiếp với tên \
doanh nghiệp.
- Còn sót ký tự/mã kỹ thuật rõ ràng của bảng nguồn (mã thuyết minh, công thức kiểu "(60 = 50 - 51)", \
ký tự OCR thừa).

# TIÊU CHÍ ĐẠT (natural=true)
- Diễn giải khoản mục bằng ngôn ngữ tài chính tự nhiên; người đọc không cần nhìn bảng CSV vẫn hiểu \
đúng đang hỏi gì.

Nếu natural=false, HÃY viết lại câu hỏi (field "rewrite") sao cho tự nhiên hơn — BẮT BUỘC giữ nguyên \
chính xác: công ty (tên + mã), kỳ/thời điểm, loại báo cáo (hợp nhất/công ty mẹ), và đúng ý nghĩa số \
liệu đang được hỏi trong câu gốc. CHỈ được đổi cách diễn đạt — TUYỆT ĐỐI không đổi số liệu, thực thể, \
hay kỳ báo cáo (câu hỏi vẫn phải khớp với đáp án/pandas_query đã có sẵn, chỉ bạn không thấy 2 thứ đó).

Chỉ trả lời bằng JSON hợp lệ, KHÔNG thêm text nào khác, KHÔNG rào ```:
{"natural": true, "reason": "...", "rewrite": null}
"""


def build_naturalness_judge_prompt(*, question: str, table_labels: str) -> tuple[str, str]:
    user = f"""\
Câu hỏi cần chấm:
{question}

Tên cột + tên hàng gốc của (các) bảng liên quan (để đối chiếu xem câu hỏi có nhại nguyên nhãn không):
{table_labels}

Hãy chấm câu hỏi trên, trả về đúng JSON yêu cầu.
"""
    return _JUDGE_SYSTEM, user
