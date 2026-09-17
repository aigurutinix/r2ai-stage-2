
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from vifinqa.generation.hard.recipe.grounded.recipes import GRO_03_ID
from vifinqa.generation.hard.recipe.grounded.analytical_planner import (
    PERSISTENT_CASH_GROWTH_ROA,
    REVENUE_GROWTH_ASSET_TURNOVER_PANEL_LOOKUP,
    REVENUE_GROWTH_GROSS_MARGIN_PANEL_LOOKUP,
)
from vifinqa.generation.hard.recipe.planner import PublicSpec
from vifinqa.generation.hard.recipe.recipes import R1_RECIPE_ID, RECIPE_MEANINGS
from vifinqa.generation.parsing import parse_json_object
from vifinqa.llm.base import ChatLLM
from vifinqa.generation.panel.catalog import (
    GROUNDED_DERIVED_FORMULAS,
    GROUNDED_DERIVED_UNIT_LABELS,
    RatioDefinition,
    get_ratio,
    metric_name,
)

logger = logging.getLogger(__name__)

_GROWTH_TRANSFORM_RECIPE_IDS = frozenset(
    {
        R1_RECIPE_ID,
        GRO_03_ID,
        PERSISTENT_CASH_GROWTH_ROA,
        REVENUE_GROWTH_GROSS_MARGIN_PANEL_LOOKUP,
        REVENUE_GROWTH_ASSET_TURNOVER_PANEL_LOOKUP,
    }
)

MAX_SEMANTIC_AUDIT_BATCH_SIZE = 10
PROMPT_VERSION = "semantic-auditor-v2"

SemanticDecision = Literal["accept", "reject", "needs_review"]

_ROLE_HEADER: dict[str, str] = {
    "filter": "Metric dùng để lọc",
    "rank": "Metric dùng để xếp hạng",
    "target": "Metric được hỏi",
}
_ROLE_ORDER: tuple[str, ...] = ("filter", "rank", "target")

_SYSTEM = """\
Bạn là chuyên gia phân tích tài chính. Mọi candidate dưới đây đã được code xác nhận chạy đúng dữ \
liệu, đúng công thức và đúng pandas query.

Nhiệm vụ của bạn KHÔNG phải kiểm tra số học và KHÔNG được thay đổi candidate. Bạn phải đánh giá \
liệu chuỗi filter/rank/target có tạo thành một câu hỏi phân tích tài chính hợp lý và tự nhiên hay \
không.

Với MỖI candidate:

1. Xác định metric filter tạo ra nhóm công ty/kỳ có đặc điểm tài chính gì.
2. Nếu có metric rank, xác định việc xếp hạng theo metric đó phục vụ mục tiêu phân tích nào trong \
nhóm đã lọc.
3. Xác định target giúp diễn giải hoặc đánh giá đối tượng được chọn như thế nào.
4. Với recipe aggregate, giải thích vì sao tổng hợp target trên nhóm đã lọc là một kết quả có ý \
nghĩa.
5. Đánh giá toàn bộ chuỗi, không đánh giá từng metric riêng lẻ.

QUY TẮC BẮT BUỘC:
- KHÔNG được accept chỉ vì các metric đều là chỉ tiêu tài chính phổ biến.
- KHÔNG được accept nếu rationale chỉ kể lại thao tác lọc/xếp hạng/tính target.
- KHÔNG được phát minh quan hệ nhân quả.
- KHÔNG bắt buộc các metric phải cùng nhóm tài chính (không có allowlist/denylist family) — một \
quan hệ khác nhóm vẫn có thể hợp lý nếu có mục tiêu phân tích rõ.
- Nếu target chỉ là một con số tình cờ có sẵn và không giúp diễn giải bước chọn, phải reject.
- Nếu metric hoặc công thức không đủ rõ để đánh giá, trả needs_review.
- Nếu không thể giải thích mạch phân tích cụ thể trong tối đa hai câu, không được accept.
- Phải audit ĐÚNG MỘT LẦN mọi candidate_id được cho, không bỏ sót, không tự thêm ID mới.
- rationale 20-400 ký tự, không được rỗng, không kể lại thao tác thuần tuý.
- Chỉ trả JSON hợp lệ, không thêm text, không rào ```:
{"assessments": [{"candidate_id": "...", "decision": "accept", "rationale": "..."}]}
"""


@dataclass(frozen=True, slots=True)
class SemanticAssessment:
    candidate_id: str
    decision: SemanticDecision
    rationale: str


class SemanticAuditor(Protocol):
    def audit(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        feedback: str | None = None,
    ) -> tuple[SemanticAssessment, ...]:
        ...


class SemanticAuditResponseError(ValueError):
    pass


def _threshold_display_value(
    metric_key: str, raw_threshold: float, *, transform: str | None = None
) -> float:
    if transform == "growth":
        return round(raw_threshold * 100, 2)
    ratio = get_ratio(metric_key)
    derived_unit = GROUNDED_DERIVED_UNIT_LABELS.get(metric_key)
    if (
        ratio is not None
        and ratio.value_kind in ("percentage", "percentage_point")
    ) or derived_unit in {"%", "điểm phần trăm"}:
        return round(raw_threshold * 100, 2)
    return round(raw_threshold, 2)


def _threshold_display_text(
    metric_key: str, raw_threshold: float, *, transform: str | None = None
) -> str:
    ratio = get_ratio(metric_key)
    derived_unit = GROUNDED_DERIVED_UNIT_LABELS.get(metric_key)
    value = _threshold_display_value(metric_key, raw_threshold, transform=transform)
    value_text = f"{value:g}"
    if (
        transform in {None, "identity"}
        and ratio is None
        and abs(raw_threshold) >= 1_000_000_000
    ):
        billions = raw_threshold / 1_000_000_000
        return f"{billions:g} tỷ đồng"
    if (
        transform == "period_difference"
        and (
            (
                ratio is not None
                and ratio.value_kind in ("percentage", "percentage_point")
            )
            or derived_unit in {"%", "điểm phần trăm"}
        )
    ):
        return f"{value_text} điểm phần trăm"
    suffix = (
        "%"
        if transform == "growth"
        else (
            "%"
            if (
                ratio is not None
                and ratio.value_kind in ("percentage", "percentage_point")
            )
            or derived_unit in {"%", "điểm phần trăm"}
            else "lần"
        )
    )
    return f"{value_text}{suffix if suffix == '%' else f' {suffix}'}"


def _ratio_unit_label(ratio: RatioDefinition) -> str:
    return "%" if ratio.value_kind in ("percentage", "percentage_point") else "lần"


def _ratio_side_text(terms: tuple[tuple[float, str], ...]) -> str:
    parts: list[str] = []
    for coefficient, metric_key in terms:
        name = metric_name(metric_key) or metric_key
        if not parts:
            parts.append(f"-{name}" if coefficient < 0 else name)
        else:
            parts.append(f" - {name}" if coefficient < 0 else f" + {name}")
    return "".join(parts)


def _ratio_formula_text(ratio: RatioDefinition) -> str:
    return (
        f"{_ratio_side_text(ratio.numerator)} / {_ratio_side_text(ratio.denominator)}"
    )


def _company_line(spec: PublicSpec) -> str:
    if spec.universe_description:
        return f"Universe (dùng đúng mô tả này, không liệt kê ticker): {spec.universe_description}"
    if len(spec.entities) > 1:
        return f"Công ty (dùng đúng mã ticker sau đây trong câu hỏi): {', '.join(spec.entities)}"
    name = spec.company_names[0] if spec.company_names else spec.entities[0]
    return f"Công ty: {name} ({spec.entities[0]})"


def _metric_role_block(spec: PublicSpec, role: str) -> list[str]:
    role_keys = [key for key, roles in spec.metric_roles.items() if role in roles]
    if not role_keys:
        return []
    lines = [f"{_ROLE_HEADER[role]}:"]
    for key_label in role_keys:
        metric_key = spec.metric_keys.get(key_label, "")
        label = spec.metric_labels.get(key_label, "")
        lines.extend((f"- {label}", f"- Key: {metric_key}"))
        ratio = get_ratio(metric_key)
        if ratio is not None:
            lines.append(f"- Công thức: {_ratio_formula_text(ratio)}")
            lines.append(f"- Đơn vị: {_ratio_unit_label(ratio)}")
        elif metric_key in GROUNDED_DERIVED_FORMULAS:
            lines.append(f"- Công thức: {GROUNDED_DERIVED_FORMULAS[metric_key]}")
            lines.append(f"- Đơn vị: {GROUNDED_DERIVED_UNIT_LABELS[metric_key]}")
        transform = (
            spec.rank_transform
            if role == "rank"
            else spec.target_transform
            if role == "target"
            else spec.filter_transform
            if role == "filter"
            else None
        )
        if (
            spec.recipe_id in _GROWTH_TRANSFORM_RECIPE_IDS and role == "rank"
        ) or transform in {"growth", "period_difference"}:
            transform_name = (
                "tốc độ tăng trưởng" if transform == "growth" else "mức thay đổi"
            )
            action = (
                "trước khi lọc"
                if role == "filter"
                else "trước khi xếp hạng"
                if role == "rank"
                else "trước khi lấy đáp án"
            )
            if label.casefold().startswith(("thay đổi", "tăng trưởng")):
                lines.append(
                    f"- Lưu ý: tên chỉ tiêu đã bao gồm {transform_name}; dùng TRỰC TIẾP nguyên "
                    f"tên này và nêu 'so với năm liền trước' {action}, KHÔNG thêm tiền tố "
                    "'mức thay đổi'/'tăng trưởng' lần nữa."
                )
            else:
                lines.append(
                    f"- Lưu ý: metric này được biến đổi thành {transform_name} so với năm liền "
                    f"trước {action}, không phải giá trị tuyệt đối — câu hỏi phải diễn đạt bằng "
                    "đúng 'tăng trưởng'/'mức thay đổi', KHÔNG mô tả như đang dùng giá trị thô."
                )
        if len(spec.metric_roles.get(key_label, ())) > 1:
            other_roles = [r for r in spec.metric_roles[key_label] if r != role]
            other_labels = "/".join(_ROLE_HEADER[r] for r in other_roles)
            lines.append(
                "- Lưu ý: metric này CÒN giữ vai trò khác trong cùng chuỗi suy luận: "
                f"{other_labels}."
            )
    return lines


def describe_public_spec(spec: PublicSpec) -> str:
    meaning = spec.analytical_frame or RECIPE_MEANINGS.get(spec.recipe_id, "")
    lines = [
        f"--- candidate_id={spec.candidate_id} ---",
        f"Recipe: {spec.recipe_id}",
        f"Ý nghĩa: {meaning}",
        _company_line(spec),
    ]
    if spec.template_id:
        lines.append(
            f"Template ID (metadata, không viết vào câu hỏi): {spec.template_id}"
        )
        lines.append(
            f"Chuỗi operation bắt buộc: {' -> '.join(spec.operation_sequence)}"
        )
        for predicate in spec.predicates:
            if predicate.related_metric_role:
                relation = f" metric {predicate.related_metric_role}"
            elif predicate.threshold_source == "zero":
                sign = "dương" if predicate.operator in {">", ">="} else "âm"
                relation = (
                    f" 0 (diễn đạt tự nhiên là '{sign}', không viết đơn vị cho số 0)"
                )
            elif predicate.threshold_value is not None:
                relation = " " + _threshold_display_text(
                    spec.metric_keys.get(predicate.metric_role, ""),
                    predicate.threshold_value,
                    transform=predicate.transform,
                )
            else:
                relation = f" {predicate.statistic or predicate.threshold_source}"
            lines.append(
                f"Predicate {predicate.predicate_id}: metric {predicate.metric_role} "
                f"{predicate.operator}{relation}; transform={predicate.transform or 'identity'}"
            )
        for cohort in spec.cohorts:
            lines.append(
                f"Cohort {cohort.cohort_id}: {cohort.construction}; roles={cohort.source_roles}; "
                f"predicates={cohort.predicate_ids}; size={cohort.size}; "
                f"quantile={cohort.quantile_percent}; "
                f"equality={cohort.equality_policy}; rounding={cohort.rounding_policy}; "
                f"minimum_size={cohort.minimum_size}"
            )
        for branch in spec.terminal_branches:
            lines.append(
                f"Terminal branch {branch.branch_id}: {branch.operation}; "
                f"inputs={branch.input_branches}; metric={branch.metric_role}; "
                f"cohort={branch.cohort_id}; selector={branch.selector_step}"
            )
    if spec.story_family:
        lines.append(f"Story family: {spec.story_family}")
    has_temporal_transform = any(
        transform in {"growth", "period_difference", "temporal_all"}
        for transform in (
            spec.filter_transform,
            spec.rank_transform,
            spec.target_transform,
        )
    )
    if spec.reference_period and not has_temporal_transform:
        period_line = f"Năm phải nêu trong câu hỏi: {spec.reference_period}"
    else:
        period_line = f"Giai đoạn phải nêu: {spec.periods[0]}-{spec.periods[-1]}"
        if spec.reference_period:
            period_line += f" (năm tham chiếu {spec.reference_period})"
    lines.append(period_line)

    for role in _ROLE_ORDER:
        lines.extend(_metric_role_block(spec, role))

    if spec.terminal_measurement_name:
        lines.append(
            "Phép đo terminal được hỏi (phải nêu đúng tên trong câu hỏi): "
            f"{spec.terminal_measurement_name}"
        )

    if spec.threshold_source == "convention" and spec.threshold_convention_value == 0.0:
        sign_word = "dương" if spec.threshold_operator in (">", ">=") else "âm"
        lines.append(
            f"Điều kiện dấu (diễn đạt bằng đúng 1 từ '{sign_word}', KHÔNG viết ra số 0 hay "
            f"'cao hơn 0'): {spec.metric_labels.get('A', '')} {sign_word}"
        )
    elif spec.threshold_source == "convention":
        display_text = _threshold_display_text(
            spec.metric_keys.get("A", ""),
            spec.threshold_convention_value or 0.0,
            transform=spec.filter_transform,
        )
        if (
            spec.filter_transform == "growth"
            and (spec.threshold_convention_value or 0.0) < 0
        ):
            decrease_phrase = (
                "giảm hơn" if spec.threshold_operator == "<" else "giảm ít nhất"
            )
            lines.append(
                f"Ngưỡng (diễn đạt tự nhiên là '{decrease_phrase}', không dùng đơn vị lần): "
                f"{spec.metric_labels.get('A', '')} {decrease_phrase} "
                f"{display_text.removeprefix('-')} "
                f"(tương đương growth {spec.threshold_operator} {display_text})"
            )
        else:
            lines.append(
                f"Ngưỡng (quy ước tài chính chuẩn, dùng đúng con số này trong câu hỏi): "
                f"{spec.metric_labels.get('A', '')} {spec.threshold_operator} {display_text}"
            )
    elif spec.threshold_source == "derived":
        lines.append(
            f"Ngưỡng (tính runtime từ population): {spec.metric_labels.get('A', '')} "
            f"{spec.threshold_operator} {spec.threshold_statistic} của nhóm"
        )
    if spec.selector_direction:
        lines.append(f"Hướng chọn: {spec.selector_direction}")
    if spec.filter_transform:
        lines.append(f"Biến đổi metric lọc: {spec.filter_transform}")
    if spec.rank_transform:
        lines.append(f"Biến đổi metric xếp hạng: {spec.rank_transform}")
    if spec.target_transform:
        lines.append(f"Biến đổi metric được hỏi: {spec.target_transform}")
    if spec.aggregate_operation:
        lines.append(f"Phép tổng hợp: {spec.aggregate_operation}")
    lines.append(f"Đơn vị đáp án: {spec.target_unit_label}")
    if spec.interpretation_limits:
        lines.append("Giới hạn diễn giải BẮT BUỘC tôn trọng:")
        for limit in spec.interpretation_limits:
            lines.append(f"- {limit}")
    if spec.required_question_terms:
        lines.append(
            "Từ/cụm từ ngữ nghĩa bắt buộc phải xuất hiện trong câu hỏi: "
            + ", ".join(spec.required_question_terms)
        )
    return "\n".join(lines)


class _Assessment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    decision: SemanticDecision
    rationale: str = Field(min_length=20, max_length=400)


class _Response(BaseModel):
    model_config = ConfigDict(extra="forbid")

    assessments: list[_Assessment]


def _validate_response(
    response: _Response, valid_ids: set[str]
) -> tuple[SemanticAssessment, ...]:
    seen: set[str] = set()
    for item in response.assessments:
        if item.candidate_id not in valid_ids:
            raise SemanticAuditResponseError(
                f"candidate_id is outside the shortlist: {item.candidate_id!r}"
            )
        if item.candidate_id in seen:
            raise SemanticAuditResponseError(
                f"duplicate candidate_id: {item.candidate_id!r}"
            )
        seen.add(item.candidate_id)
    missing = valid_ids - seen
    if missing:
        raise SemanticAuditResponseError(
            f"missing assessment for candidate_id: {sorted(missing)}"
        )
    return tuple(
        SemanticAssessment(
            candidate_id=item.candidate_id,
            decision=item.decision,
            rationale=item.rationale,
        )
        for item in response.assessments
    )


def _build_user_prompt(specs: tuple[PublicSpec, ...], *, feedback: str | None) -> str:
    blocks = "\n\n".join(describe_public_spec(s) for s in specs)
    user = f"Danh sách candidate:\n\n{blocks}\n\nHãy audit từng candidate, trả đúng JSON yêu cầu."
    if feedback:
        user = f"{feedback}\n\n{user}"
    return user


class LLMSemanticAuditor:

    def __init__(self, llm: ChatLLM) -> None:
        self._llm = llm
        self.prompt_version = PROMPT_VERSION

    def audit(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        feedback: str | None = None,
    ) -> tuple[SemanticAssessment, ...]:
        if not specs:
            return ()
        valid_ids = {s.candidate_id for s in specs}
        user = _build_user_prompt(specs, feedback=feedback)
        raw = self._llm.complete(system=_SYSTEM, user=user)
        try:
            parsed = _Response.model_validate(parse_json_object(raw))
        except Exception as exc:
            raise SemanticAuditResponseError(
                f"LLM response could not be parsed: {exc}"
            ) from exc
        return _validate_response(parsed, valid_ids)


class FakeSemanticAuditor:

    def __init__(self, decisions: dict[str, SemanticDecision] | None = None) -> None:
        self._decisions = decisions or {}

    def audit(
        self,
        specs: tuple[PublicSpec, ...],
        *,
        feedback: str | None = None,
    ) -> tuple[SemanticAssessment, ...]:
        del feedback
        return tuple(
            SemanticAssessment(
                candidate_id=spec.candidate_id,
                decision=self._decisions.get(spec.candidate_id, "accept"),
                rationale="Fake semantic audit for deterministic test.",
            )
            for spec in specs
        )


__all__ = [
    "SemanticAuditor",
    "SemanticAssessment",
    "SemanticDecision",
    "SemanticAuditResponseError",
    "LLMSemanticAuditor",
    "FakeSemanticAuditor",
    "MAX_SEMANTIC_AUDIT_BATCH_SIZE",
    "describe_public_spec",
]
