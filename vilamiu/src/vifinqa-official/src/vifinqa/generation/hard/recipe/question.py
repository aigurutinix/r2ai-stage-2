
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from collections.abc import Mapping
from typing import Protocol

from pydantic import BaseModel, Field

from vifinqa.generation.hard.recipe.grounded.recipes import (
    EQ_01_ID,
    GRO_03_ID,
    LEV_05_ID,
    LIQ_02_ID,
    PRO_04_ID,
    WCA_10_ID,
)
from vifinqa.generation.hard.recipe.planner import PublicSpec
from vifinqa.generation.hard.recipe.semantic_auditor import (
    _GROWTH_TRANSFORM_RECIPE_IDS,
    _threshold_display_value,
    _threshold_display_text,
    describe_public_spec,
)
from vifinqa.generation.parsing import parse_json_object
from vifinqa.llm.base import ChatLLM

logger = logging.getLogger(__name__)

MAX_QUESTION_BATCH_SIZE = 10
PROMPT_VERSION = "question-builder-v3"

_OPERATOR_WORDS: dict[str, tuple[str, ...]] = {
    ">": ("cao hơn", "lớn hơn", "vượt", "trên mức", "trên"),
    ">=": ("cao hơn", "lớn hơn", "vượt", "trên mức", "trên", "từ"),
    "<": ("thấp hơn", "nhỏ hơn", "dưới mức", "dưới", "ít hơn"),
    "<=": (
        "thấp hơn",
        "nhỏ hơn",
        "dưới mức",
        "dưới",
        "ít hơn",
        "tối đa",
        "không cao hơn",
    ),
}
_DIRECTION_WORDS: dict[str, tuple[str, ...]] = {
    "argmax": ("cao nhất", "lớn nhất", "mạnh nhất", "nhiều nhất"),
    "argmin": ("thấp nhất", "nhỏ nhất"),
}
_STATISTIC_WORDS: dict[str, tuple[str, ...]] = {
    "median": ("trung vị",),
    "average_threshold": ("trung bình", "bình quân"),
    "percentile": ("phân vị",),
}
_AGGREGATE_WORDS: dict[str, tuple[str, ...]] = {
    "sum": ("tổng cộng", "tổng số", "tổng "),
    "average": ("trung bình", "bình quân"),
    "minimum": ("thấp nhất", "nhỏ nhất"),
    "maximum": ("cao nhất", "lớn nhất"),
}
_GROWTH_WORDS: tuple[str, ...] = (
    "tăng trưởng",
    "thay đổi",
    "biến động",
    "mức tăng",
    "tăng",
    "giảm",
)
_NEGATIVE_TEMPORAL_OPERATOR_WORDS: dict[str, tuple[str, ...]] = {
    "<": ("giảm hơn", "sụt hơn"),
    "<=": ("giảm ít nhất", "sụt ít nhất"),
}
_POSITIVE_PREDICATE_WORDS: tuple[str, ...] = (
    "dương",
    "tăng",
    "cao hơn 0",
    "lớn hơn 0",
    "trên 0",
)
_NEGATIVE_PREDICATE_WORDS: tuple[str, ...] = (
    "âm",
    "giảm",
    "thấp hơn 0",
    "nhỏ hơn 0",
    "dưới 0",
)
_TERMINAL_WORDS: dict[str, tuple[str, ...]] = {
    "count": (
        "bao nhiêu công ty",
        "bao nhiêu doanh nghiệp",
        "số công ty",
        "số doanh nghiệp",
    ),
    "difference": ("chênh lệch", "trừ", "thay đổi", "biến động"),
    "ratio_of_sums": (
        "gấp bao nhiêu lần",
        "bằng bao nhiêu lần",
        "tỷ số",
        "tỷ lệ giữa",
        "chia cho",
    ),
    "share": ("chiếm", "nắm giữ", "đóng góp", "tỷ trọng"),
    "mean": ("trung bình", "bình quân"),
    "average": ("trung bình", "bình quân"),
}

_SYSTEM = """\
Bạn là trợ lý viết câu hỏi phân tích tài chính tiếng Việt NGẮN GỌN, TỰ NHIÊN từ 1 đặc tả ngữ nghĩa \
đã bị khoá sẵn (recipe, công ty, giai đoạn, tên chỉ tiêu, ngưỡng, hướng chọn, đơn vị đáp án). Bạn \
KHÔNG được nhìn thấy số liệu thật, đáp án, hay công ty/kỳ nào là kết quả trung gian — chỉ diễn đạt \
lại đúng ngữ nghĩa đã cho thành câu hỏi.

QUY TẮC BẮT BUỘC:
- Chỉ viết CÂU HỎI tiếng Việt, KHÔNG được đổi metric/công thức/toán tử/ngưỡng/topology so với đặc tả.
- Diễn đạt mọi toán tử bằng TỪ tiếng Việt tự nhiên ("cao hơn", "thấp hơn", "ít nhất", ...);
  TUYỆT ĐỐI KHÔNG chép ký hiệu `>`, `<`, `>=`, `<=` vào câu hỏi.
- Nêu đúng universe. Nếu đặc tả có dòng "Universe", dùng đúng mô tả đó và KHÔNG liệt kê ticker; \
nếu đặc tả có dòng "Công ty" thì mới liệt kê ticker hoặc tên công ty như được yêu cầu. Nêu giai \
đoạn hoặc năm tham chiếu, và tên TỪNG chỉ tiêu. Giữ NGUYÊN cụm tên chỉ tiêu đã cho (chỉ được đổi
chữ cái đầu thành chữ thường khi đặt giữa câu), không rút gọn, đổi nhãn hay thay bằng tên gần nghĩa.
- Với nhiều ticker, dùng cách gọi tự nhiên "trong nhóm A, B và C" hoặc "xét nhóm A, B và C"; \
KHÔNG viết "trong các mã A, B, C". Khi mô tả cohort, ưu tiên "xét các công ty..."/"trong số các \
công ty..."; KHÔNG dùng cấu trúc sượng "với nhóm có...".
- Trong cùng một batch, bám topology để thay đổi vị trí mệnh đề một cách tự nhiên (mốc thời gian, \
peer group, điều kiện cohort, phép chọn/tổng hợp); KHÔNG nhét mọi câu vào cùng một vỏ câu. Việc đổi \
cấu trúc chỉ là diễn đạt, tuyệt đối không được đổi semantics.
- Danh sách ticker phải có danh từ chỉ đối tượng đi kèm; đặt từ chỉ phép bình quân ở vị trí tự \
nhiên so với tên chỉ tiêu; tránh dùng đồng thời danh từ và động từ cùng diễn tả một biến động trong \
cùng cụm.
- Khi đặc tả ghi "Năm phải nêu", chỉ viết tự nhiên "năm X"; không tự dựng cụm "giai đoạn A, B".
- Với metric lọc được biến đổi thành growth/period_difference và ngưỡng âm: toán tử `<` phải viết
  "[metric] giảm hơn X%/X điểm phần trăm", còn `<=` mới viết "giảm ít nhất"; không viết
  "tăng trưởng [metric] giảm" và không dùng đơn vị "lần".
- Với phép đếm, thể hiện đơn vị bằng danh từ đối tượng trong cụm hỏi và không thêm cụm giải thích \
đơn vị ở cuối câu.
- Nếu ngưỡng là quy ước tài chính chuẩn (threshold_source=convention): nêu đúng hướng (cao hơn/\
thấp hơn/vượt/dưới...) và đúng giá trị ngưỡng đã cho.
- Nếu ngưỡng suy từ dữ liệu (threshold_source=derived): nêu đúng hướng và ĐÚNG TÊN THỐNG KÊ \
(median->"trung vị", average_threshold->"trung bình", percentile->"phân vị X") — TUYỆT ĐỐI KHÔNG \
được viết ngưỡng suy từ dữ liệu thành một con số cụ thể vì con số đó chỉ được tính lúc thực thi.
- Nếu có hướng chọn (selector_direction: argmax/argmin): nêu đúng "cao nhất"/"thấp nhất".
- Nếu có phép tổng hợp (aggregate_operation): nêu đúng tổng/trung bình/thấp nhất/cao nhất.
- Với cohort quantile/phân vị: phải nêu tỷ lệ %, hướng cao nhất/thấp nhất, quy tắc LÀM TRÒN LÊN,
  quy mô TỐI THIỂU BA doanh nghiệp mỗi cohort và LOẠI trường hợp hòa/đồng hạng tại RANH GIỚI.
- Với cohort top-N: phải nêu đúng "top N" gắn với chỉ tiêu xếp hạng và nêu quy tắc loại trường
  hợp hòa/đồng hạng tại ranh giới nếu đặc tả yêu cầu reject_boundary_ties.
- Với nhiều predicate/cohort/terminal branch: diễn đạt đủ TỪNG hướng điều kiện và quan hệ đồng
  thời; nêu đúng phép cuối (hai tổng được so sánh, tỷ trọng trên tổng, chênh lệch hai lookup hoặc
  bình quân). Không được lược bỏ một nhánh chỉ vì metric còn lại đã xuất hiện trong câu.
- Với terminal difference có thứ tự hai nhánh, phải nêu rõ dấu bằng cấu trúc "nhánh thứ nhất trừ
  nhánh thứ hai"; không chỉ viết "chênh lệch giữa A và B" vì cách đó mơ hồ về dấu, và không viết
  cụm lặp sượng "chênh lệch X trừ Y" — dùng trực tiếp "X trừ Y".
- "Giới hạn diễn giải" chỉ ràng buộc điều KHÔNG được suy diễn sai; KHÔNG tự biến từng giới hạn
  thành điều kiện phải đọc ra trong câu hỏi. Chỉ nêu điều kiện khi nó nằm trong Predicate, Cohort,
  Terminal branch, Ngưỡng, Hướng chọn hoặc "Từ/cụm từ ngữ nghĩa bắt buộc" của đặc tả.
- Nêu đúng đơn vị đáp án (%, lần, đồng) phù hợp target_unit_label.
- Không lặp từ liền nhau (ví dụ "bình quân bình quân"). Nếu tên chỉ tiêu đã chứa "tài sản bình
quân" mà phép tổng hợp cũng là average, viết tự nhiên theo dạng "bình quân của [chỉ tiêu]", không
ghép thêm "bình quân" ngay sau tên chỉ tiêu.
- Nếu tên chỉ tiêu đã bắt đầu bằng "Thay đổi" hoặc "Tăng trưởng", dùng trực tiếp tên đó; KHÔNG
  thêm tiền tố "mức thay đổi"/"tăng trưởng" lần nữa thành cụm lặp.
- Nếu chuỗi operation có `target_period_difference -> restrict -> minimum -> absolute_magnitude`,
  diễn đạt tự nhiên phép cuối bằng "[chỉ tiêu gốc] giảm mạnh nhất" hoặc "mức giảm lớn nhất của
  [chỉ tiêu gốc]". Đây là độ lớn dương của mức giảm âm nhất; KHÔNG viết kép "Thay đổi [chỉ tiêu]
  giảm" và không đổi thành "tăng mạnh nhất".
- KHÔNG được lộ công ty/kỳ nào là kết quả trung gian hay giá trị đáp án.
- KHÔNG hỏi trực tiếp công ty/năm nào (đó chỉ là bước trung gian) — luôn hỏi một giá trị SỐ cuối cùng.
- KHÔNG dùng thuật ngữ tài chính tự chế, KHÔNG nhại nguyên tên dòng/cột OCR, KHÔNG nhồi thêm công \
ty/điều kiện ngoài đặc tả đã cho.
- Xử lý tối đa 10 candidate/lần, ánh xạ đúng theo "candidate_id", không bỏ sót, không tạo ID mới.
- Chỉ trả JSON hợp lệ, không thêm text, không rào ```:
{"items": [{"candidate_id": "...", "question": "..."}]}
"""


@dataclass(frozen=True, slots=True)
class QuestionItem:
    candidate_id: str
    question: str


class QuestionBuilder(Protocol):
    def build(self, specs: tuple[PublicSpec, ...]) -> tuple[QuestionItem, ...]:
        ...

    def revise(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        rejection_feedback: Mapping[str, str],
    ) -> tuple[QuestionItem, ...]:
        ...


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(w in folded for w in words)


def _number_candidates(value: float) -> tuple[str, ...]:
    rounded = round(value, 4)
    if float(rounded).is_integer():
        return (str(int(rounded)),)
    dot = f"{rounded:g}"
    comma = dot.replace(".", ",")
    return (dot, comma)


def _threshold_operator_words(
    operator: str,
    *,
    value: float | None,
    transform: str | None,
) -> tuple[str, ...]:
    if (
        transform in {"growth", "period_difference"}
        and value is not None
        and value < 0
        and operator in _NEGATIVE_TEMPORAL_OPERATOR_WORDS
    ):
        return _NEGATIVE_TEMPORAL_OPERATOR_WORDS[operator]
    return _OPERATOR_WORDS.get(operator, ())


_PROXIMITY_WINDOW = 80


def _role_key(spec: PublicSpec, role: str) -> str | None:
    for key, roles in spec.metric_roles.items():
        if role in roles:
            return key
    return None


def _label_positions(folded_question: str, label: str) -> list[int]:
    folded_label = label.casefold()
    if not folded_label:
        return []
    positions: list[int] = []
    start = 0
    while True:
        idx = folded_question.find(folded_label, start)
        if idx == -1:
            break
        positions.append(idx)
        start = idx + 1
    return positions


def _marker_near_label(
    folded_question: str,
    label: str,
    markers: tuple[str, ...],
    *,
    window: int = _PROXIMITY_WINDOW,
) -> bool:
    label_positions = _label_positions(folded_question, label)
    if not label_positions:
        return False
    for marker in markers:
        marker_idx = folded_question.find(marker)
        while marker_idx != -1:
            if any(
                lp - window <= marker_idx <= lp + len(label) + window
                for lp in label_positions
            ):
                return True
            marker_idx = folded_question.find(marker, marker_idx + 1)
    return False


def _aggregate_words(spec: PublicSpec) -> tuple[str, ...]:
    """Lexical contract for the terminal reducer.

    A signed minimum followed by ``absolute_magnitude`` is naturally expressed in Vietnamese as
    the *largest decline*, not as the "smallest change".  Keep this exception tied to the locked
    operation order so ordinary ``minimum`` questions cannot pass with maximum wording.
    """
    words = _AGGREGATE_WORDS.get(spec.aggregate_operation or "", ())
    if (
        spec.aggregate_operation == "minimum"
        and spec.target_transform == "period_difference"
    ):
        operations = spec.operation_sequence
        try:
            minimum_index = operations.index("minimum")
            absolute_index = operations.index("absolute_magnitude")
        except ValueError:
            pass
        else:
            if minimum_index < absolute_index:
                words += (
                    "giảm mạnh nhất",
                    "sụt mạnh nhất",
                    "mức giảm lớn nhất",
                    "mức sụt giảm lớn nhất",
                )
    return words


def _template_sequence_errors(folded_question: str, spec: PublicSpec) -> list[str]:
    """Verify every structured predicate/cohort/terminal relation exposed by a reviewed template.

    The legacy scalar fields remain useful for R1-R7 and simple analytical frames.  Reviewed
    templates additionally expose lists because reducing a conjunction or a branched terminal to
    the first filter/aggregate silently changes the intent.
    """
    errors: list[str] = []
    for predicate in spec.predicates:
        label = spec.metric_labels.get(predicate.metric_role, "")
        if not label:
            errors.append(
                f"predicate {predicate.predicate_id!r} tham chiếu metric role không tồn tại "
                f"{predicate.metric_role!r}"
            )
            continue
        if predicate.threshold_source == "zero":
            markers = (
                _POSITIVE_PREDICATE_WORDS
                if predicate.operator in {">", ">="}
                else _NEGATIVE_PREDICATE_WORDS
            )
            if not _marker_near_label(folded_question, label, markers):
                errors.append(
                    f"thiếu hướng điều kiện {predicate.operator} 0 gắn với predicate "
                    f"{predicate.predicate_id} ({label!r})"
                )
        elif predicate.threshold_source == "relation":
            related = spec.metric_labels.get(predicate.related_metric_role or "", "")
            words = _OPERATOR_WORDS.get(predicate.operator, ())
            if not related or related.casefold() not in folded_question:
                errors.append(
                    f"thiếu metric đối chiếu của predicate quan hệ {predicate.predicate_id}"
                )
            if not words or not _marker_near_label(folded_question, label, words):
                errors.append(
                    f"thiếu toán tử quan hệ {predicate.operator} gắn với predicate "
                    f"{predicate.predicate_id} ({label!r})"
                )
        elif predicate.threshold_source == "derived":
            words = _OPERATOR_WORDS.get(predicate.operator, ())
            statistic_words = _STATISTIC_WORDS.get(predicate.statistic or "", ())
            if not words or not _marker_near_label(folded_question, label, words):
                errors.append(
                    f"thiếu hướng ngưỡng suy ra {predicate.operator} gắn với predicate "
                    f"{predicate.predicate_id} ({label!r})"
                )
            if not statistic_words or not _marker_near_label(
                folded_question, label, statistic_words
            ):
                errors.append(
                    f"thiếu thống kê {predicate.statistic!r} gắn với predicate "
                    f"{predicate.predicate_id} ({label!r})"
                )
        elif predicate.threshold_source == "fixed":
            value = predicate.threshold_value
            # Some analytical-frame specs keep the temporal transform on the
            # metric role (`PublicSpec.filter_transform`) instead of repeating
            # it on every fixed predicate.  Use that locked role-level value so
            # a threshold such as growth <= -10% is checked against the natural
            # Keep unit and scale handling explicit.
            # -0.1).  Predicates with an explicit transform remain authoritative.
            transform = predicate.transform
            if (
                transform is None
                and "filter" in spec.metric_roles.get(predicate.metric_role, ())
            ):
                transform = spec.filter_transform
            words = _threshold_operator_words(
                predicate.operator,
                value=value,
                transform=transform,
            )
            if not words or not _marker_near_label(folded_question, label, words):
                errors.append(
                    f"thiếu hướng ngưỡng cố định {predicate.operator} gắn với predicate "
                    f"{predicate.predicate_id} ({label!r})"
                )
            if value is not None:
                display_value = _threshold_display_value(
                    spec.metric_keys.get(predicate.metric_role, ""),
                    value,
                    transform=transform,
                )
                candidates = _number_candidates(display_value)
                if transform in {"growth", "period_difference"} and display_value < 0:
                    candidates += _number_candidates(abs(display_value))
                if transform in {None, "identity"} and abs(display_value) >= 1_000_000_000:
                    candidates += (f"{display_value / 1_000_000_000:g} tỷ",)
                if not any(
                    _marker_near_label(folded_question, label, (candidate,))
                    for candidate in candidates
                ):
                    errors.append(
                        f"thiếu giá trị ngưỡng cố định của predicate "
                        f"{predicate.predicate_id} ({label!r})"
                    )
        if predicate.transform == "period_difference" and len(spec.periods) > 2:
            prior = spec.periods[-2]
            current = spec.periods[-1]
            baseline_markers = (
                "so với năm liền trước",
                f"so với năm {prior}",
                f"so với {prior}",
                f"{prior}-{current}",
            )
            if not _marker_near_label(
                folded_question, label, baseline_markers, window=120
            ):
                errors.append(
                    f"predicate {predicate.predicate_id} dùng period_difference tại {current} "
                    f"nhưng chưa nêu baseline {prior}/năm liền trước gắn với {label!r}"
                )

    if any(
        cohort.construction == "intersection" and len(cohort.predicate_ids) > 1
        for cohort in spec.cohorts
    ) and not _contains_any(folded_question, ("đồng thời", "vừa", " và ", "chỉ tính")):
        errors.append("missing an intersection/simultaneous relationship between predicates")

    terminal_operations = {branch.operation for branch in spec.terminal_branches}
    for operation, markers in _TERMINAL_WORDS.items():
        if operation in terminal_operations and not _contains_any(
            folded_question, markers
        ):
            errors.append(f"missing wording for terminal operation {operation!r}")
    if "ratio_of_sums" in terminal_operations and "tổng" not in folded_question:
        errors.append("ratio_of_sums must clearly identify the two totals being compared")
    if "difference" in terminal_operations and "trừ" not in folded_question:
        errors.append(
            "terminal difference phải nêu rõ thứ tự nhánh thứ nhất trừ nhánh thứ hai"
        )
    if "share" in terminal_operations and "tổng" not in folded_question:
        errors.append("share must clearly identify the denominator total")

    if "argmin_and_argmax" in spec.operation_sequence:
        if not _contains_any(folded_question, _DIRECTION_WORDS["argmin"]):
            errors.append("missing the argmin branch of the two-lookup terminal")
        if not _contains_any(folded_question, _DIRECTION_WORDS["argmax"]):
            errors.append("missing the argmax branch of the two-lookup terminal")
    return errors


def question_round_trip_error(question: str, spec: PublicSpec) -> str:
    errors: list[str] = []
    folded_question = question.casefold()

    errors.extend(_template_sequence_errors(folded_question, spec))

    if spec.universe_description:
        if spec.universe_description.casefold() not in folded_question:
            errors.append(
                f"thiếu hoặc diễn đạt lệch universe đã khóa {spec.universe_description!r}"
            )
    elif len(spec.entities) > 1:
        missing = [e for e in spec.entities if e.casefold() not in folded_question]
        if missing:
            errors.append(f"question is missing company tickers: {missing}")
    elif spec.entities:
        ticker = spec.entities[0]
        name = spec.company_names[0] if spec.company_names else ""
        if ticker.casefold() not in folded_question:
            errors.append(f"question is missing company ticker {ticker!r}")
        if name and name.casefold() not in folded_question:
            errors.append(f"question is missing company name {name!r}")

    if spec.reference_period and spec.reference_period not in question:
        errors.append(f"question is missing reference year {spec.reference_period!r}")
    requires_explicit_period_span = (
        spec.reference_period is None
        or spec.filter_transform in {"growth", "period_difference", "temporal_all"}
        or spec.rank_transform in {"growth", "period_difference"}
        or spec.target_transform in {"growth", "period_difference"}
    )
    if len(spec.periods) > 1 and requires_explicit_period_span:
        if not (spec.periods[0] in question and spec.periods[-1] in question):
            errors.append(
                f"thiếu mốc giai đoạn {spec.periods[0]}-{spec.periods[-1]} trong câu hỏi"
            )
    elif (
        len(spec.periods) == 1
        and not spec.reference_period
        and spec.periods[0] not in question
    ):
        errors.append(f"question is missing year {spec.periods[0]!r}")

    for role in sorted(spec.metric_labels):
        label = spec.metric_labels[role]
        if label.casefold() not in folded_question:
            errors.append(f"question is missing metric name {role}={label!r}")

    for term in spec.required_question_terms:
        if term.casefold() not in folded_question:
            errors.append(f"question is missing required condition/semantics {term!r}")

    filter_key = _role_key(spec, "filter")
    filter_label = spec.metric_labels.get(
        filter_key or "", ""
    ) or spec.metric_labels.get("A", "")
    if (
        spec.filter_transform in {"growth", "period_difference"}
        and (spec.threshold_convention_value or 0.0) < 0
        and filter_label
    ):
        folded_label = filter_label.casefold()
        prefixed_change = folded_label.startswith(
            ("tăng trưởng ", "thay đổi ", "mức thay đổi ", "biến động ")
        ) or any(
            marker in folded_question
            for prefix in ("tăng trưởng", "thay đổi", "mức thay đổi", "biến động")
            for marker in (
                f"{prefix} {folded_label}",
                f"{prefix} của {folded_label}",
            )
        )
        if prefixed_change and _marker_near_label(
            folded_question,
            filter_label,
            ("giảm hơn", "giảm ít nhất", "sụt hơn", "sụt ít nhất"),
        ):
            errors.append(
                f"chỉ tiêu lọc {filter_label!r} đã được biến đổi theo thời gian và có ngưỡng "
                "âm; phải diễn đạt chỉ tiêu gốc giảm, không viết kiểu 'tăng trưởng/thay đổi "
                "[chỉ tiêu] giảm'"
            )
    rank_key = _role_key(spec, "rank")
    rank_label = spec.metric_labels.get(rank_key or "", "")
    target_key = _role_key(spec, "target")
    target_label = (
        spec.metric_labels.get(target_key or "", "")
        or spec.terminal_measurement_name
        or spec.metric_labels.get(max(spec.metric_labels, default=""), "")
    )

    if spec.threshold_source == "convention":
        if spec.threshold_convention_value == 0.0:
            sign_words = (
                ("dương",) if spec.threshold_operator in (">", ">=") else ("âm",)
            )
            if not _marker_near_label(folded_question, filter_label, sign_words):
                errors.append(
                    f"thiếu từ diễn tả điều kiện dấu ({sign_words[0]}) GẮN VỚI chỉ tiêu lọc "
                    f"{filter_label!r} trong câu hỏi (quan hệ filter->điều kiện)"
                )
        else:
            operator_words = _threshold_operator_words(
                spec.threshold_operator or "",
                value=spec.threshold_convention_value,
                transform=spec.filter_transform,
            )
            if not operator_words or not _marker_near_label(
                folded_question, filter_label, operator_words
            ):
                errors.append(
                    f"thiếu từ diễn tả hướng ngưỡng ({spec.threshold_operator}) GẮN VỚI chỉ tiêu "
                    f"lọc {filter_label!r} trong câu hỏi (quan hệ filter->điều kiện)"
                )
            display_value = _threshold_display_value(
                spec.metric_keys.get(filter_key or "A", ""),
                spec.threshold_convention_value or 0.0,
                transform=spec.filter_transform,
            )
            display_candidates = _number_candidates(display_value)
            if (
                spec.filter_transform in {"growth", "period_difference"}
                and display_value < 0
            ):
                display_candidates += _number_candidates(abs(display_value))
            if (
                spec.filter_transform in {None, "identity"}
                and abs(display_value) >= 1_000_000_000
            ):
                display_candidates += (f"{display_value / 1_000_000_000:g} tỷ",)
            if not any(
                _marker_near_label(folded_question, filter_label, (c,))
                for c in display_candidates
            ):
                errors.append(
                    f"thiếu giá trị ngưỡng quy ước {display_value} GẮN VỚI chỉ tiêu lọc "
                    f"{filter_label!r} trong câu hỏi"
                )
    elif spec.threshold_source == "derived":
        operator_words = _OPERATOR_WORDS.get(spec.threshold_operator or "", ())
        if not operator_words or not _marker_near_label(
            folded_question, filter_label, operator_words
        ):
            errors.append(
                f"thiếu từ diễn tả hướng ngưỡng ({spec.threshold_operator}) GẮN VỚI chỉ tiêu lọc "
                f"{filter_label!r} trong câu hỏi (quan hệ filter->điều kiện)"
            )
        stat_words = _STATISTIC_WORDS.get(spec.threshold_statistic or "", ())
        if not _marker_near_label(folded_question, filter_label, stat_words):
            errors.append(
                f"thiếu từ thống kê ngưỡng ({spec.threshold_statistic}) GẮN VỚI chỉ tiêu lọc "
                f"{filter_label!r} trong câu hỏi"
            )

    if spec.selector_direction:
        words = _DIRECTION_WORDS.get(spec.selector_direction, ())
        anchor_label = rank_label or filter_label
        if not anchor_label or not _marker_near_label(
            folded_question, anchor_label, words
        ):
            errors.append(
                f"thiếu từ diễn tả hướng chọn ({spec.selector_direction}) GẮN VỚI chỉ tiêu xếp "
                f"hạng {anchor_label!r} trong câu hỏi (quan hệ selector->terminal)"
            )
        if (
            spec.recipe_id in _GROWTH_TRANSFORM_RECIPE_IDS
            or spec.rank_transform in {"growth", "period_difference"}
        ) and anchor_label:
            if not _marker_near_label(folded_question, anchor_label, _GROWTH_WORDS):
                errors.append(
                    f"chỉ tiêu xếp hạng {anchor_label!r} được biến đổi thành tăng trưởng/mức thay "
                    "đổi trước khi xếp hạng — câu hỏi phải nêu rõ 'tăng trưởng'/'thay đổi'/'biến "
                    "động' gắn với chỉ tiêu này, không diễn đạt như đang xếp hạng theo giá trị thô"
                )

    if spec.target_transform in {"growth", "period_difference"}:
        if not target_label or not _marker_near_label(
            folded_question, target_label, _GROWTH_WORDS
        ):
            errors.append(
                f"chỉ tiêu đích {target_label!r} được biến đổi theo thời gian — câu hỏi phải nêu "
                "rõ 'tăng trưởng'/'thay đổi'/'biến động' gắn với chỉ tiêu này"
            )

    if spec.aggregate_operation and spec.aggregate_operation != "set_count":
        words = _aggregate_words(spec)
        if not target_label or not _marker_near_label(
            folded_question, target_label, words
        ):
            errors.append(
                f"thiếu từ diễn tả phép tổng hợp ({spec.aggregate_operation}) GẮN VỚI chỉ tiêu "
                f"được hỏi {target_label!r} trong câu hỏi (quan hệ selector->terminal)"
            )

    quantile_cohorts = tuple(
        cohort
        for cohort in spec.cohorts
        if cohort.construction in {"top_quantile", "bottom_quantile"}
    )
    for cohort in quantile_cohorts:
        role = cohort.source_roles[0] if cohort.source_roles else ""
        label = spec.metric_labels.get(role, "")
        direction_words = (
            _DIRECTION_WORDS["argmax"]
            if cohort.construction == "top_quantile"
            else _DIRECTION_WORDS["argmin"]
        )
        if not label or not _marker_near_label(folded_question, label, direction_words):
            errors.append(
                f"thiếu hướng {cohort.construction} gắn với chỉ tiêu cohort {label!r}"
            )
        percent = cohort.quantile_percent
        if percent is not None and not any(
            _marker_near_label(folded_question, label, (candidate,))
            for candidate in _number_candidates(percent)
        ):
            errors.append(
                f"thiếu tỷ lệ phân vị {percent:g}% gắn với chỉ tiêu cohort {label!r}"
            )
    if quantile_cohorts:
        if any(
            c.rounding_policy == "ceil" for c in quantile_cohorts
        ) and not _contains_any(folded_question, ("làm tròn lên",)):
            errors.append("missing the ceiling rule for quantile cohort size")
        minimum_size = max((c.minimum_size or 0 for c in quantile_cohorts), default=0)
        if minimum_size and not re.search(
            rf"(?:ít nhất|tối thiểu)\s+(?:{minimum_size}|ba)\s+(?:doanh nghiệp|công ty)",
            folded_question,
        ):
            errors.append(
                f"thiếu quy mô tối thiểu {minimum_size} doanh nghiệp cho cohort phân vị"
            )
        if any(
            c.equality_policy == "reject_boundary_ties" for c in quantile_cohorts
        ) and not (
            _contains_any(folded_question, ("hòa", "đồng hạng", "bằng nhau"))
            and _contains_any(
                folded_question, ("ranh giới", "biên phân vị", "ngưỡng phân vị")
            )
        ):
            errors.append("missing the tie-exclusion rule at the quantile boundary")

    top_n_cohorts = tuple(
        cohort for cohort in spec.cohorts if cohort.construction == "top_n"
    )
    for cohort in top_n_cohorts:
        role = cohort.source_roles[0] if cohort.source_roles else ""
        label = spec.metric_labels.get(role, "")
        size = cohort.size
        top_markers = (f"top {size}", f"{size} doanh nghiệp", f"{size} công ty")
        if not size or not label or not _marker_near_label(
            folded_question, label, top_markers, window=100
        ):
            errors.append(
                f"thiếu quy mô top-N={size} gắn với chỉ tiêu xếp hạng {label!r}"
            )
        if cohort.equality_policy == "reject_boundary_ties" and not (
            _contains_any(folded_question, ("hòa", "đồng hạng", "bằng nhau"))
            and _contains_any(folded_question, ("ranh giới", "vị trí thứ"))
        ):
            errors.append("missing the tie-exclusion rule at the top-N boundary")

    if spec.rank_transform == "period_difference" and len(spec.periods) > 2:
        prior = spec.periods[-2]
        current = spec.periods[-1]
        baseline_markers = (
            "so với năm liền trước",
            f"so với năm {prior}",
            f"so với {prior}",
            f"{prior}-{current}",
        )
        if rank_label and not _marker_near_label(
            folded_question, rank_label, baseline_markers, window=120
        ):
            errors.append(
                f"chỉ tiêu xếp hạng {rank_label!r} dùng period_difference tại {current} "
                f"nhưng chưa nêu baseline {prior}/năm liền trước"
            )

    unit_checks = {
        "%": _contains_any(question, ("%", "phần trăm")),
        "đồng": "đồng" in folded_question,
        "lần": "lần" in folded_question,
        "điểm phần trăm": _contains_any(question, ("điểm phần trăm", "điểm %", "pp")),
        "công ty": _contains_any(question, ("công ty", "doanh nghiệp")),
    }
    if not unit_checks.get(spec.target_unit_label, spec.target_unit_label in question):
        errors.append(f"question is missing unit {spec.target_unit_label!r}")

    return "; ".join(errors)


def question_answer_leak_error(question: str, formatted_answer: float) -> str:
    for candidate in _number_candidates(round(formatted_answer, 2)):
        if len(candidate) >= 3 and candidate in question:
            return f"question may reveal the literal answer: {formatted_answer} (matched {candidate!r})"
    return ""


_ADJACENT_REPETITION_RES = (
    re.compile(r"\b(\w+\s+\w+)\s+\1\b", flags=re.IGNORECASE | re.UNICODE),
    re.compile(r"\b(\w+)\s+\1\b", flags=re.IGNORECASE | re.UNICODE),
)
_BANNED_JARGON_RE = re.compile(r"\bcohort\b", flags=re.IGNORECASE | re.UNICODE)
_BANNED_AWKWARD_RES = (
    re.compile(r"\bcó trung bình\b", flags=re.IGNORECASE | re.UNICODE),
    re.compile(r"\btính theo đơn vị\b", flags=re.IGNORECASE | re.UNICODE),
    re.compile(
        r"\btrong số\s+(?!các công ty\b|những công ty\b)[A-Z]{2,5}\b",
        flags=re.IGNORECASE | re.UNICODE,
    ),
    re.compile(
        r"\bmức thay đổi\b.{0,80}\btăng (?:cao|mạnh|lớn) nhất\b",
        flags=re.IGNORECASE | re.UNICODE,
    ),
    re.compile(r"\bbao nhiêu\s*\(%\)", flags=re.IGNORECASE | re.UNICODE),
    re.compile(
        r"\btăng trưởng\s+[^,.?]{1,60}\s+giảm ít nhất\b",
        flags=re.IGNORECASE | re.UNICODE,
    ),
)


def question_naturalness_error(question: str) -> str:
    for pattern in _ADJACENT_REPETITION_RES:
        match = pattern.search(question)
        if match:
            return f"adjacent repeated word or phrase: {match.group(0)!r}"
    jargon = _BANNED_JARGON_RE.search(question)
    if jargon:
        return f"unnatural English term in the question: {jargon.group(0)!r}"
    for pattern in _BANNED_AWKWARD_RES:
        match = pattern.search(question)
        if match:
            return f"awkward phrasing pattern: {match.group(0)!r}"
    return ""


class _QuestionResponseItem(BaseModel):
    candidate_id: str
    question: str = ""


class _QuestionResponse(BaseModel):
    items: list[_QuestionResponseItem] = Field(default_factory=list)


class LLMQuestionBuilder:

    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm
        self.prompt_version = PROMPT_VERSION

    def _complete(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        rejection_feedback: Mapping[str, str] | None = None,
    ) -> tuple[QuestionItem, ...]:
        if not specs:
            return ()
        valid_ids = {s.candidate_id for s in specs}
        blocks = "\n\n".join(describe_public_spec(s) for s in specs)
        if rejection_feedback:
            errors = "\n".join(
                f"- {spec.candidate_id}: {rejection_feedback.get(spec.candidate_id, 'không đạt gate')}"
                for spec in specs
            )
            user = (
                "Các câu hỏi trước chưa đạt critic hoặc deterministic gate. Hãy viết lại theo "
                "ĐẶC TẢ KHÓA; chỉ sửa cách diễn đạt, không đổi/bỏ metric, universe, giai đoạn, "
                "toán tử, hướng so sánh, ngưỡng, phép chọn, phép cuối hoặc đơn vị. Mọi predicate "
                "và terminal branch trong đặc tả đều phải được diễn đạt đủ, nhưng câu vẫn phải "
                "ngắn gọn, tự nhiên và có mục đích phân tích tài chính rõ.\n"
                f"Lỗi cần sửa:\n{errors}\n\nĐặc tả candidate:\n\n{blocks}"
            )
        else:
            user = f"Danh sách candidate cần viết câu hỏi:\n\n{blocks}"
        raw = self._llm.complete(system=_SYSTEM, user=user)
        try:
            response = _QuestionResponse.model_validate(parse_json_object(raw))
        except Exception as exc:
            logger.warning(
                "QuestionBuilder returned invalid JSON; skipping the entire batch: %s", exc
            )
            return ()

        results: list[QuestionItem] = []
        seen: set[str] = set()
        for item in response.items:
            if item.candidate_id not in valid_ids:
                logger.info(
                    "QuestionBuilder: skipping candidate_id outside the shortlist: %s",
                    item.candidate_id,
                )
                continue
            if item.candidate_id in seen:
                logger.info(
                    "QuestionBuilder: skipping duplicate candidate_id: %s",
                    item.candidate_id,
                )
                continue
            if not item.question.strip():
                logger.info(
                    "QuestionBuilder: skipping empty question for %s", item.candidate_id
                )
                continue
            seen.add(item.candidate_id)
            results.append(
                QuestionItem(
                    candidate_id=item.candidate_id, question=item.question.strip()
                )
            )
        return tuple(results)

    def build(self, specs: tuple[PublicSpec, ...]) -> tuple[QuestionItem, ...]:
        return self._complete(specs)

    def revise(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        rejection_feedback: Mapping[str, str],
    ) -> tuple[QuestionItem, ...]:
        return self._complete(specs, rejection_feedback=rejection_feedback)


class FakeQuestionBuilder:

    def build(self, specs: tuple[PublicSpec, ...]) -> tuple[QuestionItem, ...]:
        return tuple(
            QuestionItem(candidate_id=s.candidate_id, question=_fake_question(s))
            for s in specs
        )

    def revise(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        rejection_feedback: Mapping[str, str],
    ) -> tuple[QuestionItem, ...]:
        del rejection_feedback
        return self.build(specs)


def _fake_question(spec: PublicSpec) -> str:
    if spec.intent_id is not None:
        return _fake_grounded_question(spec)
    parts: list[str] = []
    if len(spec.entities) > 1:
        parts.append(f"Trong các công ty {', '.join(spec.entities)}")
    else:
        name = spec.company_names[0] if spec.company_names else spec.entities[0]
        parts.append(f"Tại công ty {name} ({spec.entities[0]})")

    if spec.threshold_source == "convention" and spec.threshold_convention_value == 0.0:
        sign_word = "dương" if spec.threshold_operator in (">", ">=") else "âm"
        parts.append(f"có {spec.metric_labels.get('A', '')} {sign_word} liên tục")
    elif spec.threshold_source == "convention":
        display_value = _threshold_display_text(
            spec.metric_keys.get("A", ""),
            spec.threshold_convention_value or 0.0,
            transform=spec.filter_transform,
        )
        direction = _OPERATOR_WORDS.get(spec.threshold_operator or ">", ("cao hơn",))[0]
        parts.append(
            f"có {spec.metric_labels.get('A', '')} {direction} {display_value}"
        )
    elif spec.threshold_source == "derived":
        direction = _OPERATOR_WORDS.get(spec.threshold_operator or ">", ("cao hơn",))[0]
        stat_word = _STATISTIC_WORDS.get(
            spec.threshold_statistic or "median", ("trung vị",)
        )[0]
        parts.append(
            f"có {spec.metric_labels.get('A', '')} {direction} {stat_word} của nhóm"
        )
    elif spec.selector_direction and "C" not in spec.metric_labels:
        direction_word = _DIRECTION_WORDS.get(spec.selector_direction, ("cao nhất",))[0]
        if len(spec.entities) > 1 and len(spec.periods) > 1:
            parts.append(
                f"có cặp công ty-năm đạt tốc độ tăng trưởng {spec.metric_labels.get('A', '')} "
                f"{direction_word}"
            )
        else:
            parts.append(
                f"có tốc độ tăng trưởng {spec.metric_labels.get('A', '')} {direction_word}"
            )

    if spec.selector_direction and "C" in spec.metric_labels:
        direction_word = _DIRECTION_WORDS.get(spec.selector_direction, ("cao nhất",))[0]
        parts.append(f", tại năm {spec.metric_labels.get('B', '')} {direction_word}")

    if len(spec.periods) > 1:
        if spec.reference_period:
            parts.append(
                f"trong giai đoạn {spec.periods[0]}-{spec.periods[-1]}, năm tham chiếu {spec.reference_period}"
            )
        else:
            parts.append(f"trong giai đoạn {spec.periods[0]}-{spec.periods[-1]}")
    elif spec.reference_period:
        parts.append(f"năm {spec.reference_period}")

    if spec.aggregate_operation:
        agg_word = _AGGREGATE_WORDS.get(spec.aggregate_operation, ("tổng cộng",))[0]
        target_role = "C" if "C" in spec.metric_labels else "B"
        parts.append(f", {agg_word} {spec.metric_labels.get(target_role, '')}")
    else:
        target_role = "C" if "C" in spec.metric_labels else "B"
        parts.append(f", {spec.metric_labels.get(target_role, '')}")

    parts.append(f"là bao nhiêu {spec.target_unit_label}?")
    return " ".join(parts)


def _fake_grounded_question(spec: PublicSpec) -> str:
    tickers = ", ".join(spec.entities)
    label_a = spec.metric_labels.get("A", "")
    label_b = spec.metric_labels.get("B", "")

    if spec.intent_id == EQ_01_ID:
        year = spec.reference_period or spec.periods[-1]
        return (
            f"Xét nhóm công ty {tickers}, trong số những công ty có {label_a} dương năm {year}, "
            f"công ty có {label_a} cao nhất thì {spec.terminal_measurement_name or label_b} của công "
            f"ty đó là bao nhiêu {spec.target_unit_label}?"
        )
    if spec.intent_id == LIQ_02_ID:
        year = spec.reference_period or spec.periods[-1]
        return (
            f"Trong số các công ty {tickers} vào năm {year}, công ty dẫn đầu về {label_a} cao nhất "
            f"có {label_b} tương đương bao nhiêu {spec.target_unit_label}?"
        )
    if spec.intent_id == GRO_03_ID:
        prior, current = spec.periods[0], spec.periods[-1]
        return (
            f"So sánh nhóm công ty {tickers} qua hai năm {prior} và {current}: công ty có mức thay "
            f"đổi {label_a} lớn nhất trong giai đoạn này ghi nhận {spec.terminal_measurement_name or label_b} "
            f"thay đổi bao nhiêu {spec.target_unit_label} so với năm {prior}?"
        )
    if spec.intent_id == PRO_04_ID:
        year = spec.reference_period or spec.periods[-1]
        return (
            f"Năm {year}, trong nhóm công ty {tickers}, công ty có {label_a} cao nhất thì "
            f"{label_b} là bao nhiêu {spec.target_unit_label}?"
        )
    if spec.intent_id == LEV_05_ID:
        year = spec.reference_period or spec.periods[-1]
        return (
            f"Xếp hạng nhóm công ty {tickers} theo {label_a} năm {year}, công ty đứng đầu (hệ số cao "
            f"nhất) có {label_b} là bao nhiêu {spec.target_unit_label}?"
        )
    if spec.intent_id == WCA_10_ID:
        year = spec.reference_period or spec.periods[-1]
        display_value = _threshold_display_value(
            spec.metric_keys.get("A", ""), spec.threshold_convention_value or 0.0
        )
        return (
            f"Trong các công ty {tickers} năm {year}, xét những công ty có {label_a} thấp hơn "
            f"{display_value}, công ty có {label_a} thấp nhất thì {label_b} là bao nhiêu {spec.target_unit_label}?"
        )
    raise ValueError(
        f"FakeQuestionBuilder: unknown intent_id: {spec.intent_id!r}"
    )


__all__ = [
    "QuestionBuilder",
    "QuestionItem",
    "LLMQuestionBuilder",
    "FakeQuestionBuilder",
    "MAX_QUESTION_BATCH_SIZE",
    "question_round_trip_error",
    "question_answer_leak_error",
    "question_naturalness_error",
]
