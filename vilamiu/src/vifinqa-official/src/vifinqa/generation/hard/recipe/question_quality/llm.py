"""OpenAI-compatible LLM implementation of the locked-spec question critic."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from vifinqa.generation.hard.recipe.planner import PublicSpec
from vifinqa.generation.hard.recipe.question_quality.base import (
    QuestionQualityAssessment,
    QuestionQualityDecision,
    QuestionQualityResponseError,
)
from vifinqa.generation.hard.recipe.semantic_auditor import describe_public_spec
from vifinqa.generation.parsing import parse_json_object
from vifinqa.llm.base import ChatLLM

PROMPT_VERSION = "question-critic-v2"
LiteralPassFail = Literal["pass", "fail"]

_SYSTEM = """\
Bạn là biên tập viên độc lập cho câu hỏi phân tích tài chính tiếng Việt. Mỗi candidate gồm một đặc
tả ngữ nghĩa đã khóa và một câu hỏi được viết từ đặc tả đó. Bạn KHÔNG được thay đổi metric, universe,
giai đoạn, toán tử, ngưỡng, phép chọn, phép tổng hợp, đơn vị hoặc mục tiêu số cuối cùng.

Chấm bốn tiêu chí:
1. brevity: câu đủ ý nhưng không dài dòng, không lặp lại cùng một điều kiện bằng nhiều cách.
2. naturalness: tiếng Việt tự nhiên, không dùng jargon tiếng Anh như "cohort", "terminal", không
   có cụm máy móc hoặc cấu trúc khó đọc.
3. financial_purpose: câu thể hiện rõ mục đích phân tích tài chính của chuỗi điều kiện và phép cuối,
   không chỉ liệt kê thao tác dữ liệu hay ghép các chỉ tiêu tình cờ có sẵn.
4. locked_spec_alignment: câu giữ đủ và đúng universe, kỳ, từng predicate/cohort/terminal branch,
   toán tử, hướng so sánh, ngưỡng, phép chọn, phép cuối và đơn vị của đặc tả khóa.

Quy tắc đọc đặc tả:
- Toán tử trong câu hỏi phải được viết bằng từ tiếng Việt tự nhiên; ký hiệu `>`, `<`, `>=`, `<=`
  là lỗi naturalness và phải rewrite.
- Cụm tên của từng chỉ tiêu phải được giữ nguyên (chỉ cho phép đổi chữ cái đầu thành chữ thường),
  không chấp nhận rút gọn hoặc thay bằng một nhãn gần nghĩa.
- Nếu tên chỉ tiêu đã bắt đầu bằng "Thay đổi" hoặc "Tăng trưởng", dùng trực tiếp tên đó là đủ để
  thể hiện transform; không yêu cầu thêm tiền tố "mức thay đổi"/"tăng trưởng" gây lặp từ.
- Nếu chuỗi operation khóa `target_period_difference -> restrict -> minimum -> absolute_magnitude`,
  cách viết "[chỉ tiêu gốc] giảm mạnh nhất" hoặc "mức giảm lớn nhất của [chỉ tiêu gốc]" vừa tự
  nhiên vừa đúng semantics: lấy minimum có dấu rồi báo độ lớn dương. Không bắt câu đổi thành cụm
  kép "Thay đổi [chỉ tiêu] giảm" và không chấp nhận "tăng mạnh nhất" cho chuỗi này.
- Nếu đặc tả biến đổi một chỉ tiêu gốc theo growth/period_difference rồi lọc bằng ngưỡng âm,
  cách viết tự nhiên và đúng là "[chỉ tiêu gốc] giảm X%/X điểm phần trăm". KHÔNG yêu cầu và KHÔNG
  đề xuất thêm "tăng trưởng", "thay đổi" hay "mức thay đổi của" trước chỉ tiêu gốc, vì sẽ tạo
  cấu trúc kép sai nghĩa kiểu "mức thay đổi của biên lợi nhuận giảm".
- Các dòng "Giới hạn diễn giải" là ràng buộc chống suy diễn sai, KHÔNG phải danh sách điều kiện
  bắt buộc phải xuất hiện. Không yêu cầu thêm điều kiện nội bộ từ các dòng này trừ khi nội dung đó
  đồng thời nằm trong Predicate, Cohort, Terminal branch, Ngưỡng, Hướng chọn hoặc danh sách từ/cụm
  từ ngữ nghĩa bắt buộc.
- Với terminal difference có thứ tự, câu phải nêu rõ "nhánh thứ nhất trừ nhánh thứ hai"; cụm
  "chênh lệch giữa A và B" một mình là mơ hồ về dấu và phải rewrite. Cũng phải rewrite cụm lặp
  sượng "chênh lệch X trừ Y" thành trực tiếp "X trừ Y".

decision="accept" chỉ khi CẢ BỐN tiêu chí đều pass. Nếu bất kỳ tiêu chí
nào fail, decision="rewrite" và feedback phải nêu chính xác phần cần sửa, nhưng không đề xuất đổi
semantics. Không được tự viết câu thay thế. Phải trả đúng một assessment cho mọi candidate_id, không
thiếu, thừa hoặc trùng. Chỉ trả JSON hợp lệ, không thêm text, không rào ```:
{"assessments":[{"candidate_id":"...","decision":"accept","brevity":"pass","naturalness":"pass","financial_purpose":"pass","locked_spec_alignment":"pass","feedback":"Đạt."}]}
"""


class _Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    decision: QuestionQualityDecision
    brevity: LiteralPassFail
    naturalness: LiteralPassFail
    financial_purpose: LiteralPassFail
    locked_spec_alignment: LiteralPassFail
    feedback: str = Field(min_length=1, max_length=400)


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[_Assessment]


def _validate_response(
    response: _Response,
    valid_ids: set[str],
) -> tuple[QuestionQualityAssessment, ...]:
    seen: set[str] = set()
    result: list[QuestionQualityAssessment] = []
    for item in response.assessments:
        if item.candidate_id not in valid_ids:
            raise QuestionQualityResponseError(
                f"candidate_id is outside the shortlist: {item.candidate_id!r}"
            )
        if item.candidate_id in seen:
            raise QuestionQualityResponseError(
                f"duplicate candidate_id: {item.candidate_id!r}"
            )
        seen.add(item.candidate_id)
        criteria = (
            item.brevity,
            item.naturalness,
            item.financial_purpose,
            item.locked_spec_alignment,
        )
        expected = "accept" if all(value == "pass" for value in criteria) else "rewrite"
        if item.decision != expected:
            raise QuestionQualityResponseError(
                f"decision does not match the three criteria for {item.candidate_id!r}: "
                f"expected={expected!r}, got={item.decision!r}"
            )
        if item.decision == "rewrite" and len(item.feedback.strip()) < 10:
            raise QuestionQualityResponseError(
                f"rewrite feedback is too short for {item.candidate_id!r}"
            )
        result.append(
            QuestionQualityAssessment(
                candidate_id=item.candidate_id,
                decision=item.decision,
                feedback=item.feedback.strip(),
            )
        )
    missing = valid_ids - seen
    if missing:
        raise QuestionQualityResponseError(
            f"missing assessment for candidate_id: {sorted(missing)}"
        )
    return tuple(result)


class LLMQuestionCritic:
    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm
        self.prompt_version = PROMPT_VERSION

    def critique(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        questions: Mapping[str, str],
        feedback: str | None = None,
    ) -> tuple[QuestionQualityAssessment, ...]:
        if not specs:
            return ()
        valid_ids = {spec.candidate_id for spec in specs}
        if set(questions) != valid_ids:
            raise QuestionQualityResponseError(
                "questions must exactly match candidate_id values in specs"
            )
        blocks = "\n\n".join(
            f"{describe_public_spec(spec)}\nCâu hỏi cần chấm: {questions[spec.candidate_id]}"
            for spec in specs
        )
        user = f"Danh sách candidate cần chấm:\n\n{blocks}"
        if feedback:
            user = f"Response trước vi phạm contract: {feedback}\n\n{user}"
        raw = self._llm.complete(system=_SYSTEM, user=user)
        try:
            response = _Response.model_validate(parse_json_object(raw))
        except Exception as exc:
            raise QuestionQualityResponseError(
                f"LLM response could not be parsed: {exc}"
            ) from exc
        return _validate_response(response, valid_ids)
