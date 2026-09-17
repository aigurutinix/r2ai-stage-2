
from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from vifinqa.generation.hard.recipe.audit.base import (
    DependencyReviewItem,
    DependencyReviewResult,
    RejectionReason,
)
from vifinqa.generation.parsing import parse_json_object
from vifinqa.llm.base import ChatLLM

logger = logging.getLogger(__name__)

PROMPT_VERSION = "dependency-auditor-v2"

_SYSTEM = """\
Bạn là kiểm toán viên dữ liệu, xác minh 1 danh sách "dependency" trích từ báo cáo tài chính đã \
OCR có đúng khớp với chỉ số tài chính được khai hay kỳ/cơ sở thời gian được yêu cầu hay không.

Với MỖI dependency (có "dependency_id" và "shortlist" riêng), chọn ĐÚNG MỘT giá trị trong đúng \
"shortlist" của dependency đó:
- "confirm": bằng chứng (label dòng, ngữ cảnh) khớp với chỉ số đã khai.
- "reject": bằng chứng KHÔNG khớp hoặc không đủ căn cứ để khẳng định khớp.

QUY TẮC BẮT BUỘC:
- KHÔNG được trả về số liệu, toạ độ ô, hay bất kỳ giá trị nào ngoài đúng 1 chuỗi trong shortlist.
- KHÔNG được trả corrected value, công thức mới, binding mới, coordinate mới, hoặc tự cộng nhiều kỳ.
- KHÔNG được tạo dependency_id mới, KHÔNG bỏ sót dependency nào được liệt kê.
- KHÔNG suy đoán ý nghĩa ngoài evidence đã cho; nếu evidence không đủ rõ, chọn "reject".
- Với evidence period_basis, chỉ xác minh TARGET selected column có đúng time basis mà dependency yêu cầu.
- Không dùng range của bảng/page khác nếu evidence không chứng minh liên hệ với đúng target anchor/table/column.
- Evidence thiếu hoặc xung đột => chọn "reject".
- Chỉ trả JSON hợp lệ, không thêm text, không rào ```:
{"items": [{"dependency_id": "...", "chosen_id": "confirm", "reason": "..."}]}
"""


class _ResponseItem(BaseModel):
    dependency_id: str
    chosen_id: str = ""
    reason: str = ""


class _Response(BaseModel):
    items: list[_ResponseItem] = Field(default_factory=list)


def _build_prompt(items: tuple[DependencyReviewItem, ...]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            "\n".join(
                [
                    f"--- dependency_id={item.dependency_id} ---",
                    f"Công ty: {item.ticker}, kỳ: {item.period}, metric_key: {item.metric_key}",
                    f"Lý do cần xác minh: {item.reason_hint.value}",
                    f"Evidence: {item.evidence}",
                    f"shortlist: {list(item.shortlist)}",
                ]
            )
        )
    return "\n\n".join(blocks) + "\n\nHãy trả quyết định cho từng dependency_id, đúng JSON yêu cầu."


class LLMDependencyAuditor:

    def __init__(self, llm: ChatLLM, *, model_id: str) -> None:
        self._llm = llm
        self.prompt_version = PROMPT_VERSION
        self.model_id = model_id

    def audit(self, items: tuple[DependencyReviewItem, ...]) -> tuple[DependencyReviewResult, ...]:
        if not items:
            return ()
        user = _build_prompt(items)
        raw = self._llm.complete(system=_SYSTEM, user=user)
        try:
            response = _Response.model_validate(parse_json_object(raw))
        except Exception as exc:
            logger.warning("Dependency audit batch returned invalid JSON; rejecting the entire batch: %s", exc)
            return tuple(
                DependencyReviewResult(
                    dependency_id=item.dependency_id,
                    accept=False,
                    interpretation_id=None,
                    reason=item.reason_hint,
                    detail=f"LLM response could not be parsed: {exc}",
                )
                for item in items
            )

        by_id = {r.dependency_id: r for r in response.items}
        results: list[DependencyReviewResult] = []
        for item in items:
            resp = by_id.get(item.dependency_id)
            if resp is None or resp.chosen_id not in item.shortlist:
                results.append(
                    DependencyReviewResult(
                        dependency_id=item.dependency_id,
                        accept=False,
                        interpretation_id=None,
                        reason=item.reason_hint,
                        detail="missing response or chosen_id is outside the shortlist",
                    )
                )
                continue
            accept = resp.chosen_id == "confirm"
            results.append(
                DependencyReviewResult(
                    dependency_id=item.dependency_id,
                    accept=accept,
                    interpretation_id=resp.chosen_id if accept else None,
                    reason=None if accept else item.reason_hint,
                    detail=resp.reason,
                )
            )
        return tuple(results)


class FakeDependencyAuditor:

    def __init__(self, *, reject_ids: frozenset[str] = frozenset()) -> None:
        self.prompt_version = "fake-dependency-auditor-v1"
        self.model_id = "fake"
        self._reject_ids = reject_ids

    def audit(self, items: tuple[DependencyReviewItem, ...]) -> tuple[DependencyReviewResult, ...]:
        results = []
        for item in items:
            if item.dependency_id in self._reject_ids:
                results.append(
                    DependencyReviewResult(
                        dependency_id=item.dependency_id,
                        accept=False,
                        interpretation_id=None,
                        reason=item.reason_hint,
                        detail="fake reject (test)",
                    )
                )
            else:
                results.append(
                    DependencyReviewResult(
                        dependency_id=item.dependency_id,
                        accept=True,
                        interpretation_id="confirm",
                        reason=None,
                        detail="fake confirm (test)",
                    )
                )
        return tuple(results)


__all__ = ["LLMDependencyAuditor", "FakeDependencyAuditor", "PROMPT_VERSION", "RejectionReason"]
